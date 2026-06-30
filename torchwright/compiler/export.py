"""Compile a torchwright graph to a KV-cached ONNX model.

One exporter returning an :class:`OnnxArtifact` (paths + small build
metadata; ``artifact.load()`` / ``artifact.debug_session(...)``):

    compile_to_onnx(output_node, embedding, path, ...)
        Token I/O: token_ids -> logits.  Sidecar format
        ``torchwright.token.v1`` carries the vocab.  Consumer:
        ``OnnxTokenModule`` via
        :func:`torchwright.compiler.onnx_load.load_onnx`.

Both speak the STATIC-cache prefill/decode protocol (the vanilla
HF-StaticCache / vLLM pattern, chosen so ONNX Runtime can capture a CUDA
graph for the decode step — see plan_cuda_graph_decode.md):

    graph inputs:  <seq-input>, cache_position, past_K_i, past_V_i
    graph outputs: <seq-output>, delta_K_i, delta_V_i

K/V are sequence-major.  ``past_K_i`` is the FULL static cache buffer
``(S, n_heads, d_head)`` where ``S = cache_stride`` is baked at export
(static first dim — ORT rejects shorter feeds); slots at positions >=
the committed length must be zero-filled (zero-init, never garbage).
``cache_position`` (int64, ``(n_new,)``) carries the absolute positions
of the new rows; the causal mask (``slot j hidden iff j > p``) and the
positional-encoding rows are derived from it IN-GRAPH on GPU — no CPU
shape plane, no Memcpy nodes, so the whole graph stays capturable.  The
new rows enter the current step's attention via an in-graph ``ScatterND``
into the static buffer; ``delta_K_i`` (the new rows only, ``(n_new,
...)``) remains a graph output that the runtime persists into its owned
cache slots after the run (ORT forbids output buffers aliasing inputs).
Prefill = zero-filled past + ``cache_position = [0..n)``.  Decode =
``cache_position = [base]`` against the same full-S binding.

Both exporters stream each layer's weights into ONNX initializers (with
per-tensor sparsification) as the layer is compiled, then null out the
torch tensor references.  Peak in-memory weight footprint stays around
one dense layer's worth regardless of model depth — the path that lets
big graphs (e.g. the DOOM renderer) fit in realistic RAM.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from torchwright.compiler.residual_assignment import ResidualAssignment

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from torchwright.compiler.forward.compile import forward_compile
from torchwright.graph import Concatenate, Embedding, LiteralValue, Node
from torchwright.graph.attn import CAUSAL_MASK_SENTINEL
from torchwright.graph.rope import ROPE_BASE
from torchwright.graph.misc import Assert, DebugWatch, InputNode

TOKEN_META_FORMAT = "torchwright.token.v2"
DEBUG_META_FORMAT = "torchwright.debug.v1"


def _unwrap_output_node(node: Node) -> Node:
    """Strip Assert/DebugWatch wrappers from an output node.

    Assert and DebugWatch are stripped at compile time (GraphAnalyzer), so
    the residual assignment only carries indices for the *wrapped* node —
    looking the wrapper itself up KeyErrors.  ``compile_headless`` (the
    in-process path) already unwraps via ``GraphAnalyzer.get_output_node``;
    the ONNX exporters do their own residual lookups and need the same
    treatment (ops like ``compare``/``select`` return Assert-wrapped
    outputs).
    """
    while isinstance(node, (Assert, DebugWatch)):
        node = node.inputs[0]
    return node


# ---------------------------------------------------------------------------
# Sidecar plumbing
# ---------------------------------------------------------------------------


def meta_path_for(onnx_path: str) -> str:
    base, _ = os.path.splitext(onnx_path)
    return base + ".meta.json"


def _write_meta(onnx_path: str, meta: dict) -> str:
    meta_path = meta_path_for(onnx_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta_path


def debug_meta_path_for(onnx_path: str) -> str:
    base, _ = os.path.splitext(onnx_path)
    return base + ".debug.json"


@dataclass(frozen=True)
class OnnxArtifact:
    """Paths + small build metadata for a finished ONNX export.

    Returned by :func:`compile_to_onnx`
    so consumers stop reconstructing paths and build facts by convention
    (re-reading layer counts out of the ONNX file, rebuilding
    ``vocab_size`` from globals).

    HARD INVARIANT: built strictly from paths and scalars AFTER export
    completes — holds no graph, no weights, no exporter state.  The
    exporters' streaming memory bound (one dense layer's worth of weights
    in RAM regardless of depth) is sacred; this handle must never grow a
    field that would anchor the compiled model in memory.
    """

    path: str
    meta_path: str
    debug_path: Optional[str]  # None when debug_sidecar=False
    kind: str  # always "token"
    n_layers: int
    per_layer_n_heads: Tuple[int, ...]  # tuple copy, not the exporter's live list
    d: int
    d_head: int
    cache_stride: int
    vocab_size: Optional[int] = None  # token kind only

    def load(self, providers=None):
        """Load the artifact via :func:`torchwright.compiler.onnx_load.load_onnx`.

        Returns an ``OnnxTokenModule``.
        """
        # Function-level import: onnx_load imports from this module at
        # module level, so importing it at the top would be a cycle.
        from torchwright.compiler.onnx_load import load_onnx

        return load_onnx(self.path, providers=providers)

    def debug_session(self, output_node, providers=None):
        """Open a :class:`torchwright.debug.onnx_debug.OnnxDebugSession`.

        ``output_node`` must come from the same deterministic
        graph-construction code the export used (the session
        fingerprint-checks the rebuild).
        """
        # Function-level import: onnx_debug imports from this module at
        # module level, so importing it at the top would be a cycle.
        from torchwright.debug.onnx_debug import OnnxDebugSession

        return OnnxDebugSession(self.path, output_node, providers=providers)


# ---------------------------------------------------------------------------
# 2-D weight-matrix occupancy (floor-plan viz)
# ---------------------------------------------------------------------------

# Axis labels per matrix kind: (axis0 = what rows index, axis1 = what cols
# index).  The attention head and d_head dims are flattened into a single
# "head" axis of width n_heads*d_head == d, so head ``h`` occupies columns
# (or rows, for W_O) ``range(h*d_head, (h+1)*d_head)``.
_MATRIX_AXES = {
    "attn.W_Q": ("residual_in", "head"),
    "attn.W_K": ("residual_in", "head"),
    "attn.W_V": ("residual_in", "head"),
    "attn.W_O": ("head", "residual_out"),
    "mlp.W_in": ("residual_in", "hidden"),
    "mlp.W_out": ("hidden", "residual_out"),
}


def _dense_rects(matrix_id, rows, cols):
    """A dense write fills the full cross product ``rows × cols``.

    Run-length-encode each axis independently (order is irrelevant for a
    set of cells) and emit one rectangle per row-run × col-run pair.
    """
    from torchwright.compiler.graph_identity import encode_cols

    rects = []
    row_runs = encode_cols(sorted(rows))
    col_runs = encode_cols(sorted(cols))
    for rs, rl in row_runs:
        for cs, cl in col_runs:
            rects.append({"matrix": matrix_id, "a0": [rs, rl], "a1": [cs, cl]})
    return rects


def _diag_rects(matrix_id, rows, cols):
    """A diagonal (paired) write fills only cells ``(rows[k], cols[k])``.

    Pairing order is meaningful, so we do NOT sort.  Coalesce into maximal
    segments where both axes advance by 1 in lockstep — each becomes one
    ``diag``-flagged rectangle whose cell ``k`` is ``(a0.start+k,
    a1.start+k)``.  With contiguous hidden slots this yields the same run
    count as the node's 1-D residual occupancy (compact, exact).
    """
    rects = []
    i, n = 0, len(rows)
    while i < n:
        j = i
        while j + 1 < n and rows[j + 1] == rows[j] + 1 and cols[j + 1] == cols[j] + 1:
            j += 1
        length = j - i + 1
        rects.append(
            {
                "matrix": matrix_id,
                "a0": [rows[i], length],
                "a1": [cols[i], length],
                "diag": True,
            }
        )
        i = j + 1
    return rects


def _build_matrix_occupancy(compiled, canon, d: int, d_head: int):
    """Build the ``matrices`` table and ``placements`` map for the sidecar.

    Returns ``(matrices, placements, n_heads, d_hidden_per_layer)``.  All
    geometry comes from scalar params / int attributes (never weight
    tensors, which may be trimmed/freed by export time).  Placements are
    keyed by canonical node id; ops with no graph node (e.g. ``cancel``)
    or whose node is unreachable from the output fold under reserved
    ``_<op_type>`` / ``_unreachable`` buckets — they occupy real matrix
    area, so completeness needs them, but they are not user-graph nodes.
    """
    from torchwright.compiler.graph_identity import unwrap_debug

    n_heads = d // d_head
    head_dim = n_heads * d_head  # == d

    matrices: Dict[str, dict] = {}
    d_hidden_per_layer: List[int] = []
    for i, layer in enumerate(compiled.layers):
        d_hidden = int(getattr(layer.mlp, "d_hidden", d))
        d_hidden_per_layer.append(d_hidden)
        shapes = {
            "attn.W_Q": (d, head_dim),
            "attn.W_K": (d, head_dim),
            "attn.W_V": (d, head_dim),
            "attn.W_O": (head_dim, d),
            "mlp.W_in": (d, d_hidden),
            "mlp.W_out": (d_hidden, d),
        }
        for kind, (a_rows, a_cols) in shapes.items():
            axis0, axis1 = _MATRIX_AXES[kind]
            matrices[f"L{i}.{kind}"] = {
                "layer": i,
                "kind": kind,
                "shape": [int(a_rows), int(a_cols)],
                "axis0": axis0,
                "axis1": axis1,
            }

    placements: Dict[str, list] = {}
    recorder = getattr(compiled, "placements", None)
    if recorder is not None:
        for e in recorder.entries:
            matrix_id = f"L{e.layer}.{e.matrix_kind}"
            if e.node is None:
                key = f"_{e.op_type}"
            else:
                cid = canon.get(unwrap_debug(e.node).node_id)
                key = str(cid) if cid is not None else "_unreachable"
            if e.mode == "diag":
                rects = _diag_rects(matrix_id, e.rows, e.cols)
            else:
                rects = _dense_rects(matrix_id, e.rows, e.cols)
            placements.setdefault(key, []).extend(rects)

    return matrices, placements, n_heads, d_hidden_per_layer


def _write_debug_sidecar(
    onnx_path: str,
    *,
    compiled,
    output_node: Node,
    d: int,
    d_head: int,
    kind: str,
    input_specs: List[tuple],
    asserts: List["Assert"],
    watches: List["DebugWatch"],
    cache_stride: int,
    verbose: bool,
    optimize: int = 0,
    extra: Optional[dict] = None,
) -> str:
    """Write ``<stem>.debug.json`` — everything OnnxDebugSession needs.

    The sidecar carries the residual assignment keyed by CANONICAL node
    id (see :mod:`torchwright.compiler.graph_identity`) per capture
    state, a structural fingerprint of the compiled graph for rebuild
    validation, a per-node metadata table (``nodes``) — also keyed by
    canonical id, one entry per reachable node, carrying op type,
    annotation path, output width, baked-weight parameter count/shapes,
    input ids, and the layer/sublayer the node is scheduled into — and
    the Assert/DebugWatch coverage present at compile time (so the loader
    can warn when a rebuilt graph carries fewer checks than the compiled
    one did — the fingerprint is deliberately wrapper-transparent and
    cannot see that).  ``optimize`` records the compile-optimization
    level the artifact was built at; ``extra`` is the caller's free-form
    ``extra_metadata`` dict passed straight through (torchwright does not
    interpret its keys), mirroring the meta sidecar's ``extra``.

    State keys correspond one-to-one with the per-layer residual tensor
    names in the emitted ONNX graph: ``"input"`` ↔ ``res_0``,
    ``"L{i}.attn"`` ↔ ``l{i}_res_attn``, ``"L{i}.mlp"`` ↔
    ``l{i}_res_next``.  States whose node→columns table is the same
    dict object as an earlier state's (``duplicate_state`` sharing) are
    stored as ``{"same_as": <key>}`` to keep the file small.

    ``asserts``/``watches`` must have been collected BEFORE
    ``forward_compile`` ran — compilation strips both wrapper kinds from
    the graph in-place.
    """
    from torchwright.compiler.graph_identity import (
        canonical_ids,
        debug_fingerprint,
        encode_cols,
        nodes_by_canonical_id,
        unwrap_debug,
    )

    out = unwrap_debug(output_node)
    canon = canonical_ids(out)
    ra = compiled.residual_assignment
    assert ra is not None

    state_list: List[tuple] = [("input", compiled.layers[0].attn.in_state)]
    for i, layer in enumerate(compiled.layers):
        state_list.append((f"L{i}.attn", layer.attn.out_state))
        state_list.append((f"L{i}.mlp", layer.mlp.out_state))

    seen_tables: Dict[int, str] = {}  # id(mapping dict) -> first state key
    state_entries: List[dict] = []
    # Earliest state in which each canonical id appears.  The state_list
    # is ordered input → L0.attn → L0.mlp → L1.attn …, and a node sits in
    # the residual stream from the sublayer that computes it until it is
    # freed, so its FIRST appearance pins where it was computed.  Used as
    # the layer/sublayer source for nodes the placement recorder doesn't
    # log (literals, concatenations).
    first_state: Dict[str, str] = {}
    for key, st in state_list:
        table = ra.mapping.get(st)
        if table is None:
            state_entries.append({"key": key, "nodes": {}})
            continue
        prior = seen_tables.get(id(table))
        if prior is not None:
            state_entries.append({"key": key, "same_as": prior})
            continue
        seen_tables[id(table)] = key
        nodes: Dict[str, list] = {}
        for node, cols in table.items():
            # Alias keys may be Assert/DebugWatch wrappers
            # (ResidualAssignment.add_alias) — fold onto the wrapped
            # node; the loader unwraps before lookup anyway.
            cid = canon.get(unwrap_debug(node).node_id)
            if cid is None:
                # Not reachable from the output via inputs (e.g. the
                # PosEncoding leaf) — cannot be keyed canonically.
                continue
            nodes.setdefault(str(cid), encode_cols(list(cols)))
            first_state.setdefault(str(cid), key)
        state_entries.append({"key": key, "nodes": nodes})

    assert_targets = sorted(
        {
            canon[unwrap_debug(a.inputs[0]).node_id]
            for a in asserts
            if unwrap_debug(a.inputs[0]).node_id in canon
        }
    )
    matrices, placements, n_heads, d_hidden_per_layer = _build_matrix_occupancy(
        compiled, canon, d, d_head
    )

    # Per-node metadata keyed by canonical id — the same key space as
    # ``placements`` and the residual ``states``.  One entry per node
    # reachable from the output.
    #
    # layer/sublayer: the placement recorder logs the layer + matrix for
    # every weight-bearing op (Linear/Attn/ReLU/Add), so it is the
    # authoritative source where present; matrix_kind "attn.*"/"mlp.*"
    # gives the sublayer.  Nodes it doesn't log (literals, concatenations)
    # fall back to their first residual-state appearance, and pre-layer
    # input nodes (Embedding) report sublayer "embed".
    place_loc: Dict[str, tuple] = {}
    recorder = getattr(compiled, "placements", None)
    if recorder is not None:
        for e in recorder.entries:
            if e.node is None:
                continue
            cid = canon.get(unwrap_debug(e.node).node_id)
            if cid is None:
                continue
            cid_s = str(cid)
            if cid_s in place_loc:
                continue
            sub = "attn" if e.matrix_kind.startswith("attn") else "mlp"
            place_loc[cid_s] = (int(e.layer), sub)

    def _layer_sublayer(cid_s: str, node: Node) -> tuple:
        if cid_s in place_loc:
            return place_loc[cid_s]
        key = first_state.get(cid_s)
        if key is not None and key != "input":
            lpart, sub = key.split(".")  # "L{k}", "attn"|"mlp"
            return int(lpart[1:]), sub
        if isinstance(node, Embedding):
            return None, "embed"
        return None, None

    # Baked weight tensors probed by attribute name, summed into a
    # parameter count and per-tensor shape list.  0 / [] for pure ops.
    baked_attrs = ("table", "output_matrix", "matrix", "weight", "value")
    nodes_meta: Dict[str, dict] = {}
    for cid, node in nodes_by_canonical_id(out).items():
        cid_s = str(cid)
        weight_params = 0
        weight_shapes: List[list] = []
        for attr in baked_attrs:
            t = getattr(node, attr, None)
            shape = getattr(t, "shape", None)
            if shape is None:
                continue
            dims = [int(s) for s in shape]
            if not dims:  # 0-d scalar — no parameters to attribute
                continue
            weight_params += int(t.numel()) if hasattr(t, "numel") else 1
            weight_shapes.append([attr, dims])
        input_cids: List[str] = []
        for inp in getattr(node, "inputs", None) or []:
            icid = canon.get(unwrap_debug(inp).node_id)
            if icid is not None:
                input_cids.append(str(icid))
        layer, sublayer = _layer_sublayer(cid_s, node)
        nodes_meta[cid_s] = {
            "op": type(node).__name__,
            "annotation": node.annotation,
            "width": len(node),
            "weight_params": weight_params,
            "weight_shapes": weight_shapes,
            "inputs": input_cids,
            "layer": layer,
            "sublayer": sublayer,
            "name": getattr(node, "name", None),
        }
    payload = {
        "format": DEBUG_META_FORMAT,
        "kind": kind,  # always "token"
        "fingerprint": debug_fingerprint(out, d=d, d_head=d_head),
        "d": d,
        "d_head": d_head,
        "n_heads": n_heads,
        "d_hidden": d_hidden_per_layer,
        "n_layers": len(compiled.layers),
        "matrices": matrices,
        "placements": placements,
        "nodes": nodes_meta,
        "optimize": int(optimize),
        "extra": dict(extra) if extra else {},
        "input_specs": [list(spec) for spec in input_specs],
        "cache_stride": int(cache_stride),
        "assert_coverage": {
            "n_asserts": len(asserts),
            "n_watches": len(watches),
            "assert_targets": assert_targets,
        },
        "states": state_entries,
    }
    path = debug_meta_path_for(onnx_path)
    with open(path, "w") as f:
        json.dump(payload, f)
    if verbose:
        print(f"Wrote {path} ({os.path.getsize(path):,} bytes)")
    return path


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sparse-or-dense initializer conversion
# ---------------------------------------------------------------------------


_SPARSITY_THRESHOLD = 0.75
_MIN_SPARSE_ELEMENTS = 1024


def _tensor_to_proto(name: str, arr: np.ndarray):
    """Convert a float32 numpy array to (dense_tp, sparse_tp).

    Exactly one of the returned values is non-None.  Float tensors with
    zero-fraction >= 75% and at least 1024 elements become
    SparseTensorProto (COO, flat int64 indices).  Everything else is a
    dense TensorProto with ``raw_data``.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n_elem = arr.size
    dims = list(arr.shape)

    if n_elem < _MIN_SPARSE_ELEMENTS:
        dense_tp = helper.make_tensor(
            name=name,
            data_type=TensorProto.FLOAT,
            dims=dims,
            vals=arr.tobytes(),
            raw=True,
        )
        return dense_tp, None

    flat = arr.reshape(-1)
    nnz = int(np.count_nonzero(flat))
    zero_frac = 1.0 - nnz / n_elem
    if zero_frac < _SPARSITY_THRESHOLD:
        dense_tp = helper.make_tensor(
            name=name,
            data_type=TensorProto.FLOAT,
            dims=dims,
            vals=arr.tobytes(),
            raw=True,
        )
        return dense_tp, None

    nz_idx = np.flatnonzero(flat)
    nz_val = flat[nz_idx]
    values_tp = helper.make_tensor(
        name=name,  # SparseTensorProto identified by values.name
        data_type=TensorProto.FLOAT,
        dims=[nnz],
        vals=nz_val.astype(np.float32, copy=False).tobytes(),
        raw=True,
    )
    indices_tp = helper.make_tensor(
        name=name + "__indices",
        data_type=TensorProto.INT64,
        dims=[nnz],
        vals=nz_idx.astype(np.int64, copy=False).tobytes(),
        raw=True,
    )
    sparse_tp = helper.make_sparse_tensor(
        values=values_tp,
        indices=indices_tp,
        dims=dims,
    )
    return None, sparse_tp


def _append_proto(
    dense_tp,
    sparse_tp,
    dense_inits: list,
    sparse_inits: list,
) -> None:
    if dense_tp is not None:
        dense_inits.append(dense_tp)
    else:
        sparse_inits.append(sparse_tp)


def _add_float_init(
    name: str, arr: np.ndarray, dense_inits: list, sparse_inits: list
) -> None:
    dense_tp, sparse_tp = _tensor_to_proto(name, arr)
    _append_proto(dense_tp, sparse_tp, dense_inits, sparse_inits)


def _add_int64_init(name: str, arr: np.ndarray, dense_inits: list) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    dense_inits.append(
        helper.make_tensor(
            name=name,
            data_type=TensorProto.INT64,
            dims=list(arr.shape),
            vals=arr.tobytes(),
            raw=True,
        )
    )


def _add_scalar_inits(dense_inits: list) -> None:
    """Register the scalar / tiny 1-D helper initializers used by the
    cached preamble and layers (Where sentinel, Unsqueeze axes).

    Initializer-fed axes inputs do NOT create Memcpy nodes (initializers
    are materialized on every EP), so these are CUDA-graph-safe.
    """
    dense_inits.append(
        helper.make_tensor(
            name="_f32_causal_sentinel_s",
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=np.array(CAUSAL_MASK_SENTINEL, dtype=np.float32).tobytes(),
            raw=True,
        )
    )
    _add_int64_init("_axes0_1d", np.array([0], dtype=np.int64), dense_inits)
    _add_int64_init("_axes1_1d", np.array([1], dtype=np.int64), dense_inits)


# ---------------------------------------------------------------------------
# Streaming weight emission callback (shared by both exporters)
# ---------------------------------------------------------------------------


def _rope_freq_row(d_rot: int, base: float) -> np.ndarray:
    """``(1, d_rot)`` per-plane RoPE angular frequencies in the half-split layout.

    Matches :func:`torchwright.graph.rope.rope_inv_freq` (``base^(-2k/d_rot)``,
    normalized by the rotary width ``d_rot``) with each plane's frequency repeated
    across both halves of the rotary front, so the runtime ``angle = pos · freq``
    reproduces the in-process / oracle rotation.  ``d_rot == d_head`` is full
    rotary (identical to the LLaMA3 end state).
    """
    p = np.arange(0, d_rot, 2, dtype=np.float64)
    inv = base ** (-p / d_rot)  # (d_rot/2,)
    return np.concatenate([inv, inv]).astype(np.float32).reshape(1, d_rot)


def _add_rope_inits(
    per_layer_rotary: list, d_head: int, dense_inits: list
) -> tuple[float, int]:
    """Add the global RoPE initializers (``rope_freq``, ``rope_split``, and — for
    partial rotary — ``rope_partial_split``).  Returns ``(base, d_rot)`` for the
    sidecar meta.  All layers must share one ``base`` AND one ``d_rot``."""
    bases = {r["base"] for r in per_layer_rotary if r.get("base") is not None}
    if len(bases) > 1:
        raise NotImplementedError(
            f"ONNX RoPE export requires one global base; got {sorted(bases)}."
        )
    base = bases.pop() if bases else ROPE_BASE

    # Partial-rotary width must also be global (one rotate_half front for the
    # whole model).  The plane-based content/recency/global-recency heads are
    # full-rotary (d_rot == d_head); only the rotary offset head may set a
    # partial d_rot, and all heads must agree.
    d_rots = {r["d_rot"] for r in per_layer_rotary if r.get("d_rot") is not None}
    if len(d_rots) > 1:
        raise NotImplementedError(
            f"ONNX RoPE export requires one global d_rot (partial-rotary width); "
            f"got {sorted(d_rots)}.  The plane-based content/recency/global-recency "
            f"heads are full-rotary; only the rotary offset head sets a partial "
            f"d_rot, and every head must use the same value."
        )
    d_rot = d_rots.pop() if d_rots else d_head

    freq = _rope_freq_row(d_rot, base)  # (1, d_rot), guaranteed dense
    dense_inits.append(
        helper.make_tensor(
            name="rope_freq",
            data_type=TensorProto.FLOAT,
            dims=list(freq.shape),
            vals=freq.tobytes(),
            raw=True,
        )
    )
    # rotate_half over the rotary front d_rot (== d_head for full rotary, so this
    # init is byte-identical to the pre-partial export when d_rot == d_head).
    _add_int64_init(
        "rope_split",
        np.array([d_rot // 2, d_rot // 2], dtype=np.int64),
        dense_inits,
    )
    # Partial rotary only: split each head into the rotary front (d_rot) and the
    # unrotated NoPE tail (d_head - d_rot).  Omitted for full rotary so the
    # full-width init set is unchanged.
    if d_rot != d_head:
        _add_int64_init(
            "rope_partial_split",
            np.array([d_rot, d_head - d_rot], dtype=np.int64),
            dense_inits,
        )
    return float(base), int(d_rot)


def _make_stream_layer_weights_cb(
    d: int,
    d_head: int,
    dense_inits: list,
    sparse_inits: list,
    per_layer_n_heads: list,
    per_layer_rotary: Optional[list] = None,
    trim_heads: bool = True,
) -> Callable[[int, object], None]:
    """Factory for the forward_compile on_layer_compiled callback.

    Emits each freshly-compiled layer's dense weights into the
    dense/sparse init lists (sparsified on the fly when mostly zero) and
    nulls out the layer's tensor attributes so GC can reclaim them.

    When ``trim_heads`` is set (the default), each layer is trimmed *in
    place here* — before its tensors are read — so the unused (all-zero)
    attention heads and MLP hidden slots are physically removed: the
    emitted WQ/WK/WV/WO matrices, the per-layer reshape constants, and
    the ``past_K_i``/``past_V_i`` ValueInfo all shrink to the used count,
    cutting both the KV cache and the inference MatMuls.  This is the
    *only* place the ONNX export trims; the post-loop trims in
    ``forward_compile`` are skipped whenever a streaming callback is
    installed (their tensors are already nulled here).  With
    ``trim_heads=False`` the full-width (sparsified-but-not-trimmed)
    model is emitted instead.

    After trimming, each layer's attention matrices have shape
    ``(nh, d, d_head)`` where ``nh <= d // d_head``.  Weight matrices
    are emitted at the trimmed size and per-layer reshape constants
    are added so the ONNX graph can reshape to the correct head count.

    Appends each layer's ``nh`` to ``per_layer_n_heads`` for Phase 2.
    """

    def on_layer_compiled(i: int, layer) -> None:
        attn = layer.attn.attn
        mlp = layer.mlp

        # Trim each layer here, before its tensors are read and nulled
        # below.  ``used_heads`` (attn) and the per-slot weight columns
        # (mlp) were populated by write_attn_sublayer/write_mlp_sublayer
        # during this layer's compile, so both trims see final counts.
        if trim_heads:
            attn.trim_unused_heads()  # n_heads -> used heads (KV cache + attn MatMuls)
            mlp.trim_unused_slots()  # d_hidden -> used slots (MLP MatMuls)

        nh = attn.n_heads  # post-trim head count
        hd = nh * d_head
        per_layer_n_heads.append(nh)

        # RoPE on the one global grid (rotate_half over the rotary front d_rot,
        # rotation by absolute position).  Record the shared base and the
        # partial-rotary width; the rotation is emitted unconditionally below
        # (there is no per-head enable).  base/d_rot are asserted global in
        # _add_rope_inits.
        if per_layer_rotary is not None:
            per_layer_rotary.append(
                {
                    "base": getattr(attn, "rope_base", None),
                    "d_rot": getattr(attn, "rope_d_rot", None),
                }
            )

        def emit(name: str, arr: np.ndarray) -> None:
            dense_tp, sparse_tp = _tensor_to_proto(name, arr)
            _append_proto(dense_tp, sparse_tp, dense_inits, sparse_inits)

        # (nh, d, d_head) → (d, nh, d_head) → (d, hd)
        emit(
            f"l{i}_WQ",
            attn.query_matrix.permute(1, 0, 2)
            .reshape(d, hd)
            .contiguous()
            .cpu()
            .numpy(),
        )
        attn.query_matrix = None
        emit(
            f"l{i}_WK",
            attn.key_matrix.permute(1, 0, 2).reshape(d, hd).contiguous().cpu().numpy(),
        )
        attn.key_matrix = None
        emit(
            f"l{i}_WV",
            attn.value_matrix.permute(1, 0, 2)
            .reshape(d, hd)
            .contiguous()
            .cpu()
            .numpy(),
        )
        attn.value_matrix = None
        # (nh, d_head, d) → (hd, d): canonical W_O layout that feeds
        # one (t, hd) @ (hd, d) MatMul at inference time.
        emit(
            f"l{i}_WO",
            attn.output_matrix.reshape(hd, d).contiguous().cpu().numpy(),
        )
        attn.output_matrix = None

        # Per-layer reshape constants for the ONNX graph.
        _add_int64_init(
            f"l{i}_qkv_view_shape",
            np.array([0, nh, d_head], dtype=np.int64),
            dense_inits,
        )
        _add_int64_init(
            f"l{i}_ctx_flat_shape",
            np.array([0, hd], dtype=np.int64),
            dense_inits,
        )

        emit(f"l{i}_W1", mlp.linear1.output_matrix.cpu().numpy())
        mlp.linear1.output_matrix = None
        emit(f"l{i}_b1", mlp.linear1.output_bias.cpu().numpy())
        mlp.linear1.output_bias = None
        emit(f"l{i}_W2", mlp.linear2.output_matrix.cpu().numpy())
        mlp.linear2.output_matrix = None
        emit(f"l{i}_b2", mlp.linear2.output_bias.cpu().numpy())
        mlp.linear2.output_bias = None

    return on_layer_compiled


# ---------------------------------------------------------------------------
# Cached preamble and per-layer node emission
# ---------------------------------------------------------------------------


def _emit_cached_preamble(nodes: list) -> None:
    """Emit nodes producing ``pos``, ``mask_bool_3d`` and ``_cache_pos_col``.

    Requires:
      - graph input ``cache_position``: int64 ``(n_new,)`` — the absolute
        positions of the new rows (``[base, base+1, …]``).  The ONLY
        position fact the host provides; the causal mask and the RoPE
        cos/sin rotation both derive from it in-graph.
      - graph inputs ``past_K_0``…: sequence-major KV cache with a
        SYMBOLIC first dim ``cache_slots`` — the bound length ``S_eff``
        (a prefix window of the runtime's full-stride cache buffer) sets
        this step's attention width.  The mask width derives from it via
        ``Shape(past_K_0)`` below.
      - initializer ``arange_S``: int64 ``(S,)`` baked ``[0 .. S)``, where
        ``S = cache_stride`` is the FULL static slot count (the maximum
        any binding may use).
      - initializer ``rope_freq``: (d_rot,) — the half-split per-plane RoPE
        frequencies over the rotary front (d_rot == d_head for full rotary);
        rope_rotate applies them to the front and passes the NoPE tail through.
      - helper initializers from :func:`_add_scalar_inits`

    Memcpy invariant (the CUDA-graph capture requirement): a single
    Memcpy node makes ORT refuse capture (hard error at session creation
    under ``enable_cuda_graph``).  The real rule is *no Memcpy*, not "no
    CPU node": the ``Shape`` -> shape-``Slice`` chain here is CPU-resident
    (ORT's ``GetCpuPreferredNodes`` pass places it there) and feeds the
    arange-``Slice``'s ``starts``/``ends``/``axes`` inputs, which the CUDA
    Slice kernel declares CPU-native (``OrtMemTypeCPUInput``) — so no
    CPU<->GPU handoff exists and no Memcpy is inserted (verified:
    "MemcpyTransformer modified: 0" + capture enabled, ORT 1.26, see
    plan_stride_bucketing.md).  ORT warns once about "shape massaging
    nodes" under capture; that is safe here because shapes are frozen per
    ``gpu_graph_id`` — every replay of a bucket binds the same ``S_eff``,
    so the launch parameters baked from the CPU-computed bounds stay
    correct.  A dynamic-size CPU-produced tensor (e.g. ``Range``) feeding
    a GPU op WOULD be a Memcpy — that is why the slots come from slicing
    the baked GPU-resident ``arange_S`` instead.

    Produces:
      - ``_rope_cos`` / ``_rope_sin``: (n_new, d_rot) — the RoPE rotation
        factors at ``cache_position`` (``cos``/``sin`` of ``pos · rope_freq``),
        width d_rot (the rotary front; == d_head for full rotary),
        broadcast per layer onto Q and the new K rows.
      - ``mask_bool_3d``: (1, n_new, S_eff) bool, True where blocked: slot
        ``j`` is hidden from the query at position ``p`` iff ``j > p``.
        Applied via ``Where(mask_bool_3d, CAUSAL_MASK_SENTINEL, logits)``
        — an overwrite, not an additive penalty (additive is numerically
        unsafe when real logits can be very negative).  Hidden slots
        contain zeros (runtime zero-init discipline) and get softmax
        weight exactly 0.0 in fp32; equivalence across bindings (and to
        the old dynamic-concat graph) is TOKEN-level, not float-bit-level
        (each matmul width reselects cuBLAS kernels — see
        plan_cuda_graph_decode.md / plan_stride_bucketing.md).
      - ``_cache_pos_col``: (n_new, 1) int64 — shared by the mask
        comparison and by every layer's ``ScatterND`` indices.
    """

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # S_eff = bound first dim of the cache; slots = arange_S[:S_eff].
    # _axes0_1d/_axes1_1d double as the [0]/[1] bounds constants.
    add("Shape", ["past_K_0"], ["_pastK0_shape"])  # (3,) int64, CPU
    add(
        "Slice",
        ["_pastK0_shape", "_axes0_1d", "_axes1_1d", "_axes0_1d"],
        ["_s_eff_1d"],
    )  # (1,) = [S_eff], CPU
    add(
        "Slice",
        ["arange_S", "_axes0_1d", "_s_eff_1d", "_axes0_1d"],
        ["_slots"],
    )  # (S_eff,) GPU data, CPU bounds
    add("Unsqueeze", ["_slots", "_axes0_1d"], ["_slots_row"])  # (1, S_eff)
    add("Unsqueeze", ["cache_position", "_axes1_1d"], ["_cache_pos_col"])  # (n_new, 1)
    add("Greater", ["_slots_row", "_cache_pos_col"], ["_mask_bool"])  # (n_new, S_eff)
    add("Unsqueeze", ["_mask_bool", "_axes0_1d"], ["mask_bool_3d"])  # (1, n_new, S_eff)

    # RoPE cos/sin from absolute positions, shared across layers (always emitted).
    # angle(pos, plane) = pos * rope_freq[plane]; rope_freq is the half-split
    # per-plane frequency baked by the exporter, width d_rot (the rotary front;
    # == d_head for full rotary).  rope_rotate applies these to the front d_rot
    # dims and passes the NoPE tail through.
    add("Cast", ["cache_position"], ["_rope_pos_f"], to=TensorProto.FLOAT)
    add("Unsqueeze", ["_rope_pos_f", "_axes1_1d"], ["_rope_pos_col"])  # (n_new,1)
    add("Mul", ["_rope_pos_col", "rope_freq"], ["_rope_ang"])  # (n_new, d_rot)
    add("Cos", ["_rope_ang"], ["_rope_cos"])
    add("Sin", ["_rope_ang"], ["_rope_sin"])
    # Broadcast forms: head-major Q is (nh, n_new, d_head) -> (1,n_new,d_head);
    # sequence-major delta_K is (n_new, nh, d_head) -> (n_new,1,d_head).
    add("Unsqueeze", ["_rope_cos", "_axes0_1d"], ["_rope_cos_q"])
    add("Unsqueeze", ["_rope_sin", "_axes0_1d"], ["_rope_sin_q"])
    add("Unsqueeze", ["_rope_cos", "_axes1_1d"], ["_rope_cos_k"])
    add("Unsqueeze", ["_rope_sin", "_axes1_1d"], ["_rope_sin_k"])


def _emit_cached_layer_nodes(
    nodes: list,
    layer_idx: int,
    current_res: str,
    d: int,
    d_head: int,
    n_heads: int,
    d_rot: int,
    scatter_idx_col: str = "_cache_pos_col",
) -> str:
    """Emit cached attention + FFN nodes for one layer.

    Reads graph inputs ``past_K_{i}`` / ``past_V_{i}`` — the FULL static
    cache, sequence-major ``(S, n_heads, d_head)`` with zero-filled slots
    at positions >= the committed length — and writes graph outputs
    ``delta_K_{i}`` / ``delta_V_{i}``, the *new rows only* ``(n_new,
    n_heads, d_head)``.  The new rows are injected into THIS step's
    attention by an in-graph ``ScatterND`` at the absolute slots
    ``cache_position`` (causal self-attention needs the query to see its
    own key — the diagonal); the runtime separately persists the deltas
    into its owned cache buffer after the run, because ORT does not
    support binding an output that aliases an input.  The scattered
    ``(S, ...)`` K/V is a per-layer transient, freed before the next
    layer.  Attention uses the original head-major ``Transpose+MatMul``
    ops (the ONNX ``Einsum`` form is broken on the CUDA EP for this
    graph).  Uses the shared ``mask_bool_3d`` / ``_cache_pos_col`` from
    :func:`_emit_cached_preamble`; every shape in the decode step
    (``n_new=1``) is static, which is what makes the step CUDA-graph
    capturable.

    ``n_heads`` is the (possibly trimmed) head count for this layer.
    Per-layer reshape constants ``l{i}_qkv_view_shape`` and
    ``l{i}_ctx_flat_shape`` are expected to have been emitted by the
    streaming weight callback.

    ``scatter_idx_col`` names the (n_new, 1) int64 tensor used as the
    ScatterND row indices — ``_cache_pos_col`` (slot == position, the
    unbounded protocol).

    Returns the name of the next residual stream tensor.
    """
    p = f"l{layer_idx}"

    def node(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    def rope_rotate(src: str, dst: str, cos: str, sin: str) -> None:
        """rotate_half RoPE over the rotary front ``d_rot``: rotate the first
        ``d_rot`` dims (``front*cos + rotate_half(front)*sin``) and pass the last
        ``d_head - d_rot`` dims (the NoPE tail) through unchanged.  Matches
        graph/rope.py (vanilla partial rotary, half-split over d_rot);
        modeling_torchwright.py mirrors this op sequence.

        ``d_rot == d_head`` (full rotary) emits the exact pre-partial 6-node
        sequence — byte-for-byte the same graph, so the cancel-head denormal-ULP
        note below still holds and existing exports are unchanged.

        No ``src + (rot - src)`` reconstruction: cross-backend onnxruntime/torch
        agreement is not algebraic — the cancel-head rows that cancel to denormal
        magnitude differ by one denormal ULP regardless of the form (the
        test_convert parity bound tolerates it; no token or meaningful logit
        moves), and the full suite is bit-exact with the direct form, so the
        extra Sub/Add bought nothing.
        """
        if d_rot == d_head:
            # Full rotary: rotate_half over the whole head.
            node("Split", [src, "rope_split"], [f"{dst}_h1", f"{dst}_h2"], axis=-1)
            node("Neg", [f"{dst}_h2"], [f"{dst}_h2n"])
            node("Concat", [f"{dst}_h2n", f"{dst}_h1"], [f"{dst}_rh"], axis=-1)
            node("Mul", [src, cos], [f"{dst}_c"])
            node("Mul", [f"{dst}_rh", sin], [f"{dst}_s"])
            node("Add", [f"{dst}_c", f"{dst}_s"], [dst])
            return
        # Partial rotary: split off the rotary front (d_rot) and the NoPE tail,
        # rotate_half the front, then re-concat the untouched tail.  cos/sin are
        # width d_rot and broadcast over the front.
        node(
            "Split",
            [src, "rope_partial_split"],
            [f"{dst}_front", f"{dst}_tail"],
            axis=-1,
        )
        node(
            "Split", [f"{dst}_front", "rope_split"], [f"{dst}_h1", f"{dst}_h2"], axis=-1
        )
        node("Neg", [f"{dst}_h2"], [f"{dst}_h2n"])
        node("Concat", [f"{dst}_h2n", f"{dst}_h1"], [f"{dst}_rh"], axis=-1)
        node("Mul", [f"{dst}_front", cos], [f"{dst}_c"])
        node("Mul", [f"{dst}_rh", sin], [f"{dst}_s"])
        node("Add", [f"{dst}_c", f"{dst}_s"], [f"{dst}_rotfront"])
        node("Concat", [f"{dst}_rotfront", f"{dst}_tail"], [dst], axis=-1)

    # Project Q, K_new, V_new from the new rows in sequence-major
    # (n_new, n_heads, d_head).  The deltas (new rows only) are the graph
    # outputs the runtime writes into its owned cache tail; the reshape
    # constant l{i}_qkv_view_shape = [0, n_heads, d_head] copies n_new from
    # the flat (n_new, hd) projection.
    node("MatMul", [current_res, f"{p}_WQ"], [f"{p}_Q_flat"])
    node("Reshape", [f"{p}_Q_flat", f"{p}_qkv_view_shape"], [f"{p}_Q_sm"])

    node("MatMul", [current_res, f"{p}_WK"], [f"{p}_K_flat"])
    # Rotate the new K (sequence-major) by absolute position, so the cache stores
    # already-rotated K (matches the in-process component and HF).
    node("Reshape", [f"{p}_K_flat", f"{p}_qkv_view_shape"], [f"{p}_K_sm"])
    rope_rotate(
        f"{p}_K_sm",
        f"delta_K_{layer_idx}",
        "_rope_cos_k",
        "_rope_sin_k",
    )

    node("MatMul", [current_res, f"{p}_WV"], [f"{p}_V_flat"])
    node("Reshape", [f"{p}_V_flat", f"{p}_qkv_view_shape"], [f"delta_V_{layer_idx}"])

    # Attention runs in the ORIGINAL head-major Transpose+MatMul form — the
    # ONNX Einsum form is numerically broken on the CUDA EP for this graph
    # (wrong argmax from the very first token), so we keep the exact ops the
    # pre-cache-rewrite export used.  Transpose the sequence-major Q / past /
    # delta into head-major (n_heads, seq, d_head); the past inputs and delta
    # outputs stay sequence-major for the runtime's contiguous-slice binding.
    node("Transpose", [f"{p}_Q_sm"], [f"{p}_Q"], perm=[1, 0, 2])
    # Rotate Q (head-major) by absolute position.
    rope_rotate(
        f"{p}_Q",
        f"{p}_Q_roped",
        "_rope_cos_q",
        "_rope_sin_q",
    )
    q_name = f"{p}_Q_roped"
    # Inject the new rows into the static cache at their absolute slots
    # (sequence-major ScatterND: indices (n_new, 1) select rows, updates are
    # the (n_new, nh, d_head) deltas), then transpose to head-major for
    # attention.  This replaces the old Concat(past, new) — the
    # 21%-of-compute KV-bandwidth term — and keeps the attention width at
    # the static S every step, which is what makes the decode shape
    # replay-stable for CUDA-graph capture.
    node(
        "ScatterND",
        [f"past_K_{layer_idx}", scatter_idx_col, f"delta_K_{layer_idx}"],
        [f"{p}_K_static"],
    )
    node(
        "ScatterND",
        [f"past_V_{layer_idx}", scatter_idx_col, f"delta_V_{layer_idx}"],
        [f"{p}_V_static"],
    )
    node("Transpose", [f"{p}_K_static"], [f"{p}_K_full"], perm=[1, 0, 2])
    node("Transpose", [f"{p}_V_static"], [f"{p}_V_full"], perm=[1, 0, 2])

    # Attention over the full (past + new) K and V.
    node("Transpose", [f"{p}_K_full"], [f"{p}_K_T"], perm=[0, 2, 1])
    node("MatMul", [q_name, f"{p}_K_T"], [f"{p}_logits"])
    # Overwrite-mask with CAUSAL_MASK_SENTINEL (equivalent to torch's
    # masked_fill).  An additive penalty would leave masked positions at
    # "logit + penalty", which is not dominated by real logits when they
    # are very negative (e.g. attend_argmin_unmasked with high query gain).
    # mask_bool_3d is (1, n_new, n_total); it broadcasts over the head axis.
    node(
        "Where",
        ["mask_bool_3d", "_f32_causal_sentinel_s", f"{p}_logits"],
        [f"{p}_logits_masked"],
    )
    node("Softmax", [f"{p}_logits_masked"], [f"{p}_weights"], axis=-1)
    node("MatMul", [f"{p}_weights", f"{p}_V_full"], [f"{p}_ctx"])

    # Fused output projection: (n_heads, n_new, d_head) → (n_new, hd) → (n_new, d)
    node("Transpose", [f"{p}_ctx"], [f"{p}_ctx_t"], perm=[1, 0, 2])
    node("Reshape", [f"{p}_ctx_t", f"{p}_ctx_flat_shape"], [f"{p}_ctx_flat"])
    node("MatMul", [f"{p}_ctx_flat", f"{p}_WO"], [f"{p}_attn_sum"])
    node("Add", [current_res, f"{p}_attn_sum"], [f"{p}_res_attn"])

    # FFN + skip
    node("MatMul", [f"{p}_res_attn", f"{p}_W1"], [f"{p}_l1_m"])
    node("Add", [f"{p}_l1_m", f"{p}_b1"], [f"{p}_l1_b"])
    node("Relu", [f"{p}_l1_b"], [f"{p}_l1_r"])
    node("MatMul", [f"{p}_l1_r", f"{p}_W2"], [f"{p}_l2_m"])
    node("Add", [f"{p}_l2_m", f"{p}_b2"], [f"{p}_l2_b"])
    node("Add", [f"{p}_res_attn", f"{p}_l2_b"], [f"{p}_res_next"])

    return f"{p}_res_next"


def _kv_io_value_info(per_layer_n_heads: List[int], d_head: int) -> tuple[list, list]:
    """Build ValueInfoProto entries for the KV-cache inputs and outputs.

    ``per_layer_n_heads`` gives the (possibly trimmed) head count for
    each layer, so each layer's KV tensors carry only the heads it uses.

    Sequence-major layout with a SYMBOLIC first dim ``cache_slots``: the
    feeder binds a contiguous prefix view ``cache[: S_eff]`` of its
    full-stride buffer (any ``S_eff <= cache_stride`` covering the
    committed length + this pass's rows), and the step's attention runs
    ``S_eff`` wide — the stride-bucketing affordance.  Binding the full
    buffer reproduces the previous static-dim behavior exactly.  Every
    layer must be bound at the SAME ``S_eff`` per run; the shared symbol
    documents that contract but ORT does not enforce it declaratively (a
    mismatch fails mid-run, not at bind) — the runtime enforces it by
    constructing all layers' bindings from one ``S_eff``.  Stable
    addresses + one frozen shape per ``gpu_graph_id`` are the CUDA-graph
    replay requirements; the runtime persists the ``delta_K_i`` output
    rows into slots ``[base : base+n_new)`` after the run.

    The stride ``S`` itself no longer appears in the input shapes — the
    loaders read it from the sidecar meta (``cache_stride`` key).

    Returns:
        (past_vis, new_vis) — each a list of length 2*n_layers
        alternating ``past_K_i`` / ``past_V_i`` (inputs, shape
        ``("cache_slots", nh, d_head)``) and ``delta_K_i`` / ``delta_V_i``
        (outputs — the new rows only, shape ``(n_new, nh, d_head)``).
    """
    past_vis: list = []
    new_vis: list = []
    for i, nh in enumerate(per_layer_n_heads):
        past_vis.append(
            helper.make_tensor_value_info(
                f"past_K_{i}", TensorProto.FLOAT, ["cache_slots", nh, d_head]
            )
        )
        past_vis.append(
            helper.make_tensor_value_info(
                f"past_V_{i}", TensorProto.FLOAT, ["cache_slots", nh, d_head]
            )
        )
        new_vis.append(
            helper.make_tensor_value_info(
                f"delta_K_{i}", TensorProto.FLOAT, ["n_new", nh, d_head]
            )
        )
        new_vis.append(
            helper.make_tensor_value_info(
                f"delta_V_{i}", TensorProto.FLOAT, ["n_new", nh, d_head]
            )
        )
    return past_vis, new_vis


# ---------------------------------------------------------------------------
# Public exporters
# ---------------------------------------------------------------------------


def _resolve_cache_stride(
    cache_stride: Optional[int],
    max_seq_len: int,
) -> int:
    """Resolve the static cache slot count ``S`` (default: max_seq_len).

    ``S`` must be <= max_seq_len, the model's maximum supported absolute
    position.  Under RoPE the cos/sin rotation is computed from
    ``cache_position`` on the fly (there is no longer a ``pos_encoding_full``
    table to index), but a compiled model still cannot serve positions beyond
    ``max_seq_len``.
    """
    s = max_seq_len if cache_stride is None else int(cache_stride)
    if not (1 <= s <= max_seq_len):
        raise ValueError(
            f"cache_stride {s} must be in [1, max_seq_len={max_seq_len}]: the "
            f"static cache cannot hold positions beyond the model's maximum"
        )
    return s


def compile_to_onnx(
    output_node: Node,
    embedding: Embedding,
    output_path: str,
    d: int = 1024,
    d_head: int = 16,
    max_seq_len: int = 512,
    max_layers: int = 400,
    verbose: bool = False,
    trim_heads: bool = True,
    optimize: int = 0,
    assume_zero_init: bool = True,
    d_hidden: Optional[int] = None,
    extra_metadata: Optional[dict] = None,
    cache_stride: Optional[int] = None,
    debug_sidecar: bool = True,
) -> OnnxArtifact:
    """Compile a token-I/O graph to a KV-cached ONNX model.

    Returns an :class:`OnnxArtifact` (paths + small build metadata;
    ``artifact.load()`` for the runtime, ``artifact.debug_session(...)``
    for the debug surface).

    ``extra_metadata`` is a free-form dict written under the sidecar's
    ``"extra"`` key (surfaced via ``OnnxTokenModule.metadata``); top-level
    sidecar keys are unchanged.

    Writes three files:
        ``<output_path>``     — the ONNX model
        ``<stem>.meta.json``  — ``{"format": "torchwright.token.v1",
                                   "vocab": [...]}``
        ``<stem>.debug.json`` — the debug sidecar (residual assignment
                                keyed by canonical node id, structural
                                fingerprint, assert coverage) consumed
                                by :class:`torchwright.debug.onnx_debug.
                                OnnxDebugSession`.  Disable with
                                ``debug_sidecar=False``.

    The graph speaks the static-cache prefill/decode protocol:
        inputs:  token_ids (n_new,) int64, cache_position (n_new,) int64,
                 past_K_i, past_V_i (cache_slots, n_heads, d_head)
                 [cache_slots SYMBOLIC: the bound prefix length S_eff]
        outputs: logits (n_new, vocab_size),
                 delta_K_i, delta_V_i (n_new, n_heads, d_head)

    K/V are sequence-major; ``past_K_i`` is a contiguous prefix view
    ``cache[: S_eff]`` of the runtime's full-stride cache buffer
    (``S = cache_stride``, defaulting to ``max_seq_len``; slots at
    positions >= the committed length must be zero-filled; any
    ``base + n_new <= S_eff <= S`` is valid — stride bucketing),
    ``delta_K_i`` is the new rows only, persisted by the runtime into its
    owned cache slots after the run.  Prefill = zero past +
    cache_position [0..n); decode = cache_position [base] against a
    covering prefix binding.  Binding the full buffer reproduces the
    previous static-dim behavior exactly.  All mask/pos arithmetic is
    in-graph (the mask width derives from Shape(past_K_0) via a
    CPU-native shape chain — no Memcpy nodes, see
    :func:`_emit_cached_preamble`), and each (n_new, S_eff) binding has
    fully static shapes per run — the properties that make decode steps
    CUDA-graph capturable, one captured graph per (S_eff, width) bucket.

    ``assume_zero_init`` defaults to ``True`` here (unlike ``forward_compile``,
    which defaults ``False``): the ONNX runtime always constructs the residual
    stream from zeros plus the input projections (see
    ``HeadlessTransformer.get_input_res_stream`` /
    ``_OnnxRuntime._build_res_stream``), so the initially-free residual columns
    are guaranteed clean on entry.  The compiler can therefore skip the
    BIRTH-layer dirty-column cancels that defend against a non-zero residual
    stream — those cancels are pure overhead the ONNX runtime never needs, and
    modelling them per-allocation makes the CP-SAT attention-head cumulative
    over-tight under width pressure (it charges every fresh allocation a
    full-width dirty cancel, while the replay only clears the genuinely-dirty
    initial-pool subset).  No ONNX caller can supply a non-zero stream, so
    ``True`` is always sound for this entry point.
    """
    # Validate the cache config up front — a ValueError after the
    # (potentially very long) streaming compile would waste the whole run.
    cache_stride_resolved = _resolve_cache_stride(cache_stride, max_seq_len)

    # Assert/DebugWatch coverage must be collected BEFORE forward_compile
    # strips both wrapper kinds from the graph in-place.
    from torchwright.graph.asserts import collect_debug_nodes

    all_asserts, all_watches = collect_debug_nodes(output_node)

    dense_inits: list = []
    sparse_inits: list = []
    per_layer_n_heads: list = []
    per_layer_rotary: list = []

    on_layer_compiled = _make_stream_layer_weights_cb(
        d,
        d_head,
        dense_inits,
        sparse_inits,
        per_layer_n_heads,
        per_layer_rotary,
        trim_heads=trim_heads,
    )

    # --- Phase 1: streaming compile ---------------------------------------
    t0 = time.perf_counter()
    compiled = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        verbose=verbose,
        max_layers=max_layers,
        device=None,
        on_layer_compiled=on_layer_compiled,
        trim_heads=trim_heads,
        optimize=optimize,
        assume_zero_init=assume_zero_init,
        d_hidden=d_hidden,
    )
    t_compile = time.perf_counter() - t0
    if verbose and per_layer_n_heads:
        _max_heads = d // d_head
        _total = _max_heads * len(per_layer_n_heads)
        _kept = sum(per_layer_n_heads)
        print(
            f"  Head pruning (ONNX): {_total - _kept}/{_total} heads pruned; "
            f"per-layer heads range [{min(per_layer_n_heads)}, "
            f"{max(per_layer_n_heads)}] of {_max_heads}"
        )

    # --- Phase 2: metadata + graph assembly -------------------------------
    assert compiled.residual_assignment is not None
    n_layers = len(compiled.layers)

    t0 = time.perf_counter()
    in_state = compiled.layers[0].attn.in_state
    out_state = compiled.layers[-1].mlp.out_state

    embedding_indices: Optional[List[int]] = None
    constant_values = np.zeros(d, dtype=np.float32)

    for node in compiled.residual_assignment.get_nodes(in_state):
        indices = compiled.residual_assignment.get_node_indices(in_state, node)
        if isinstance(node, Embedding):
            embedding_indices = indices
        elif isinstance(node, LiteralValue):
            # Includes the reserved const-1 self-match column, seeded to 1.0.
            for k, idx in enumerate(indices):
                constant_values[idx] = float(node.value[k])
        elif isinstance(node, Concatenate):
            pass

    assert embedding_indices is not None, "No Embedding node in residual assignment"

    d_embed = len(embedding_indices)

    embed_table_compact = (
        embedding.table.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    vocab_size, d_embed_check = embed_table_compact.shape
    assert d_embed_check == d_embed, (
        f"Embedding table last dim {d_embed_check} disagrees with "
        f"d_embed {d_embed} derived from feature assignment"
    )

    output_indices = compiled.residual_assignment.get_node_indices(
        out_state, _unwrap_output_node(output_node)
    )
    assert len(output_indices) == d_embed, (
        f"output column count {len(output_indices)} != embedding width "
        f"{d_embed}; the unembed reuses the embedding table, so the output "
        f"node must be d_embed wide"
    )

    # Vanilla untied (llama3-style) layout: fold the residual column-placement
    # scatters into the weights, so the runtime does a plain (vocab, d) table
    # lookup and a full-width unembed — no embedding_proj / pos_proj / output
    # column gather.  Each folded table is mostly zeros (only d_embed / d_pos of
    # the d columns are populated), so _add_float_init stores them COO-sparse:
    # the ONNX file stays compact even though the dense HF weights are (vocab, d).
    embed_table = np.zeros((vocab_size, d), dtype=np.float32)
    embed_table[:, embedding_indices] = embed_table_compact

    # Untied unembed weight in nn.Linear (out, in) == (vocab, d) convention,
    # nonzero only at the output node's residual columns.  logits = res @ W.T
    # sums over all d columns, but W is exactly zero off the output columns, so
    # the rest of the residual contributes nothing.
    lm_head = np.zeros((vocab_size, d), dtype=np.float32)
    lm_head[:, output_indices] = embed_table_compact

    # Initializers
    _add_float_init("constant_values", constant_values, dense_inits, sparse_inits)
    _add_float_init("embed_table", embed_table, dense_inits, sparse_inits)
    _add_float_init("lm_head", lm_head, dense_inits, sparse_inits)
    # Baked slot indices [0..S), S = the FULL stride: the preamble slices
    # this GPU-resident constant to the bound prefix length S_eff and the
    # causal mask compares the slice against cache_position.  Keeping the
    # arange baked (vs a Range op) is what keeps the slot data GPU-side —
    # only scalar slice bounds live on CPU, so no Memcpy (the CUDA-graph
    # capture requirement; see _emit_cached_preamble).
    _add_int64_init(
        "arange_S", np.arange(cache_stride_resolved, dtype=np.int64), dense_inits
    )
    # Per-layer reshape constants (l{i}_qkv_view_shape, l{i}_ctx_flat_shape)
    # are emitted by the streaming weight callback.
    _add_scalar_inits(dense_inits)

    # Nodes: preamble (mask + pos), token embed, residual stream, layers,
    # full-width unembed.
    nodes: list = []

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # RoPE: bake the global rope inits when any layer is rotary; the preamble
    # emits cos/sin from cache_position when active (no-op otherwise).
    rope_base_val, rope_d_rot_val = _add_rope_inits(
        per_layer_rotary, d_head, dense_inits
    )
    _emit_cached_preamble(nodes)
    # Vanilla token embedding: the (vocab, d) table is gathered straight into
    # the residual seed (no projection).  Position is a rotation applied inside
    # attention (RoPE), so there is no additive position table — the seed is the
    # token embedding plus the residual constants (incl. the const-1 self-match
    # column).
    add("Gather", ["embed_table", "token_ids"], ["inp_res"], axis=0)
    add("Add", ["inp_res", "constant_values"], ["res_0"])

    current_res = "res_0"
    for i in range(n_layers):
        current_res = _emit_cached_layer_nodes(
            nodes,
            i,
            current_res,
            d,
            d_head,
            per_layer_n_heads[i],
            rope_d_rot_val,
            scatter_idx_col="_cache_pos_col",
        )

    # Untied unembed over the full residual stream: logits = res @ lm_head.T.
    # lm_head is zero off the output node's columns, so the rest of the residual
    # (live scratch, freed columns) multiplies to zero and never reaches the
    # logits — provided those columns stay finite (a NaN/Inf in scratch would
    # now poison every logit, where the old column gather ignored them).
    add("Transpose", ["lm_head"], ["_lm_head_T"], perm=[1, 0])
    add("MatMul", [current_res, "_lm_head_T"], ["logits"])

    # Graph I/O value infos
    token_ids_vi = helper.make_tensor_value_info(
        "token_ids", TensorProto.INT64, ["n_new"]
    )
    cache_position_vi = helper.make_tensor_value_info(
        "cache_position", TensorProto.INT64, ["n_new"]
    )
    past_vis, new_vis = _kv_io_value_info(per_layer_n_heads, d_head)
    logits_vi = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, ["n_new", vocab_size]
    )

    graph = helper.make_graph(
        nodes,
        "token_transformer_cached",
        inputs=[token_ids_vi, cache_position_vi, *past_vis],
        outputs=[logits_vi, *new_vis],
        initializer=dense_inits,
        sparse_initializer=sparse_inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 14)],
        producer_name="torchwright",
    )
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    onnx.save_model(model, output_path)
    t_save = time.perf_counter() - t0

    token_meta = {
        "format": TOKEN_META_FORMAT,
        "vocab": list(embedding.tokenizer.vocab),
        # The full static slot count S: with the symbolic cache_slots
        # first dim on past_K_i, loaders read S from here.
        "cache_stride": cache_stride_resolved,
        # RoPE on the one global grid; the converter rebuilds the rotation from
        # the base and the partial-rotary width d_rot (== d_head for full rotary).
        "rope_base": rope_base_val,
        "d_rot": rope_d_rot_val,
    }
    if extra_metadata:
        token_meta["extra"] = dict(extra_metadata)
    meta_path = _write_meta(output_path, token_meta)

    if debug_sidecar:
        _write_debug_sidecar(
            output_path,
            compiled=compiled,
            output_node=output_node,
            d=d,
            d_head=d_head,
            kind="token",
            # The token graph reads its ids from the Embedding node's
            # input slot; one 1-wide column matches the convention
            # CompiledHeadless uses for the same graph.
            input_specs=[(embedding.input_name, 0, 1)],
            asserts=all_asserts,
            watches=all_watches,
            cache_stride=cache_stride_resolved,
            optimize=optimize,
            extra=extra_metadata,
            verbose=verbose,
        )

    if verbose:
        print(
            f"Phases: compile+emit {t_compile:.2f}s, "
            f"build {t_build:.2f}s, save {t_save:.2f}s"
        )
        print(
            f"{n_layers} layers, {vocab_size} vocab, "
            f"{len(sparse_inits)} sparse inits, {len(dense_inits)} dense inits"
        )
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {meta_path}")

    return OnnxArtifact(
        path=output_path,
        meta_path=meta_path,
        debug_path=debug_meta_path_for(output_path) if debug_sidecar else None,
        kind="token",
        n_layers=n_layers,
        per_layer_n_heads=tuple(per_layer_n_heads),
        d=d,
        d_head=d_head,
        cache_stride=cache_stride_resolved,
        vocab_size=int(vocab_size),
    )


# ---------------------------------------------------------------------------
# In-process headless callable: a thin adapter over HeadlessTransformer.compute()
# presenting an (inputs) -> outputs interface.
#
# This is the in-process debug/test backend (the DebugRuntime behind
# debug=True forwards and the probe_* tools) — no ONNX round-trip.  For
# production inference, compile a token ONNX artifact with compile_to_onnx
# and load it with load_onnx.
# ---------------------------------------------------------------------------


@dataclass
class _DebugState:
    """Captured residual-stream snapshots from a debug=True forward pass."""

    state_tensor: Dict  # ResidualStreamState -> (tensor, label)
    ordered_states: List  # [ResidualStreamState, ...]
    ra: "ResidualAssignment"


def _ordered_mlp_state_triples(net, ra) -> List[tuple]:
    """Post-MLP sublayer states in execution order.

    Returns ``(layer_index, state_name, state)`` triples, one per
    transformer layer whose ``mlp.out_state`` is recorded in
    ``ra.mapping``.  The final layer's ``mlp.out_state`` is always
    appended (even if missing from ``ra.mapping``) so the top-level
    output is reachable when the last layer happens to receive no new
    assignments.
    """
    ordered: List[tuple] = []
    for i, layer in enumerate(net.layers):
        st = layer.mlp.out_state
        if st in ra.mapping:
            ordered.append((i, f"L{i}.mlp_out", st))
    last_i = len(net.layers) - 1
    last_st = net.layers[-1].mlp.out_state
    if not any(s is last_st for _, _, s in ordered):
        ordered.append((last_i, f"L{last_i}.mlp_out", last_st))
    return ordered


class CompiledHeadless:
    """Callable wrapper around :class:`HeadlessTransformer` — the
    in-process debug/test backend.

    Exposes a three-method surface:

    - ``module(inputs)``: stateless per-query inference — runs the
      non-cached ``forward()`` path and returns outputs.
    - ``module.step(inputs, past)``: autoregressive step — runs
      ``forward_cached()`` with the given past and returns
      ``(outputs, new_past)``.
    - ``module.empty_past()``: zero-length KV cache tuple suitable as
      the initial state for a decode sequence.
    """

    def __init__(
        self,
        net,
        input_specs: List[tuple],
        output_indices: torch.Tensor,
        metadata: Optional[dict] = None,
        asserts: Optional[List] = None,
        watches: Optional[List] = None,
    ) -> None:
        self._net = net
        # input_specs: list of (name, start_col, width) in input-tensor column order.
        self._input_specs = list(input_specs)
        self._output_indices = output_indices
        self.input_names: List[str] = [name for name, _, _ in input_specs]
        self.metadata: dict = dict(metadata or {})
        self._asserts = list(asserts) if asserts else []
        self._watches = list(watches) if watches else []
        self._debug_state: Optional[_DebugState] = None

        # KV cache shape metadata — discovered from the compiled transformer
        # so empty_past() can build zero-length tensors of the right shape.
        # After head trimming each layer may have a different n_heads.
        self._per_layer_n_heads = [layer.attn.attn.n_heads for layer in net.layers]
        self._d_head = net.layers[0].attn.attn.d_head
        self._n_layers = len(net.layers)

    def _build_res_stream(self, inputs: torch.Tensor, past_len: int) -> torch.Tensor:
        n_new = inputs.shape[0]
        input_values = {
            name: inputs[:, start : start + width]
            for name, start, width in self._input_specs
        }
        return self._net.get_input_res_stream(
            n_new, input_values, past_len=past_len
        ).to(self._net.device)

    def __call__(
        self,
        inputs: torch.Tensor,
        debug: bool = False,
        debug_atol: float = 1e-7,
    ) -> torch.Tensor:
        """Stateless per-query inference — uses the non-cached ``forward()``.

        When ``debug=True``, captures per-layer residual-stream snapshots
        and checks all Assert nodes (raises on failure) and DebugWatch
        nodes (prints on trigger).  ``debug_atol`` is the maximum
        per-column drift tolerated by the residual-stream self-consistency
        check; diffs at or below ``debug_atol`` are treated as fp-rounding
        noise.
        """
        res_stream = self._build_res_stream(inputs, past_len=0)
        if debug and (self._asserts or self._watches):
            res, _ = self._run_debug_checks(res_stream, past_kvs=None, atol=debug_atol)
        else:
            res = self._net.forward(res_stream)
        return res[:, self._output_indices]

    def empty_past(
        self,
        max_len: Optional[int] = None,
    ) -> tuple:
        """Zero-length past tensors suitable for a first prefill call.

        ``max_len`` is accepted and ignored for call-site symmetry with the
        owned-cache ONNX runtime (this in-process reference path grows its own
        head-major tuples each step rather than preallocating).
        """
        device = self._net.device
        past_K = tuple(
            torch.zeros(nh, 0, self._d_head, device=device)
            for nh in self._per_layer_n_heads
        )
        past_V = tuple(
            torch.zeros(nh, 0, self._d_head, device=device)
            for nh in self._per_layer_n_heads
        )
        return (past_K, past_V)

    def step(
        self,
        inputs: torch.Tensor,
        past: tuple,
        past_len: Optional[int] = None,
        debug: bool = False,
        debug_atol: float = 1e-7,
    ) -> tuple:
        """Cached forward step.

        Supports three ``n_new`` regimes uniformly:

        * Pure prefill (``past`` empty, ``n_new == seq_len``).
        * Single-step decode (``n_new == 1`` with a non-empty ``past``).
        * **Batched decode with past** (``n_new > 1`` with a non-empty
          ``past``) — the speculative-decoding shape. Each new row sees
          the entire past unconditionally, plus a lower-triangular slice
          of the new block (row ``i`` sees new rows ``0..i``). Row ``i``
          of the returned outputs is bit-identical to the same row from
          a sequential rollout starting from the same past.

        Speculative-decoding partial-commit pattern. After running with
        ``n_new = K + 1`` (one current input + ``K`` drafts + bonus), a
        caller may decide to commit only the first ``commit_count``
        rows. Slice the returned ``new_past`` directly::

            outputs, (new_K, new_V) = compiled.step(inputs_batch, past)
            # ... compare outputs[:K] to the K drafts, decide commit_count ...
            target_len = past[0][0].shape[1] + commit_count
            committed = (
                tuple(K[:, :target_len] for K in new_K),
                tuple(V[:, :target_len] for V in new_V),
            )

        ``committed`` is the past for the next step. The discarded rows
        (``commit_count..n_new``) carry no state outside ``new_past``,
        so the slice is the only commit primitive needed.

        Args:
            inputs: ``(n_new, d_input)`` float tensor for the new rows.
            past: ``(past_K_tuple, past_V_tuple)`` from a prior step or
                :meth:`empty_past`.  Each tuple has length ``n_layers``
                and each entry is ``(n_heads, n_past, d_head)``.
            past_len: Optional absolute query position for the new rows.
                When ``None`` (default), derived from
                ``past_K[0].shape[1]`` — i.e. "the new rows sit right
                after everything in the cache", which is the normal
                decoding protocol.  Callers using a sliding-window
                runtime may pass the true global position here while
                handing over a trimmed cache; the positional encoding
                slice uses this value while the attention mask uses the
                cache's actual shape.
            debug: When True, capture residual-stream snapshots and
                check all Assert nodes (raises) and DebugWatch nodes
                (prints).
            debug_atol: Maximum per-column drift tolerated by the
                residual-stream self-consistency check.  Diffs at or
                below ``debug_atol`` are treated as fp-rounding noise.

        Returns:
            ``(outputs, new_past)`` where ``outputs`` is
            ``(n_new, d_output)`` and ``new_past`` has the same shape
            as ``past`` but with the new rows appended.
        """
        past_K, past_V = past
        assert len(past_K) == self._n_layers
        assert len(past_V) == self._n_layers
        if past_len is None:
            past_len = int(past_K[0].shape[1])

        res_stream = self._build_res_stream(inputs, past_len=past_len)
        past_kvs = [(past_K[i], past_V[i]) for i in range(self._n_layers)]
        if debug and (self._asserts or self._watches):
            res, new_kvs = self._run_debug_checks(
                res_stream, past_kvs=past_kvs, atol=debug_atol
            )
        else:
            res, new_kvs = self._net.forward_cached(res_stream, past_kvs=past_kvs)

        new_K = tuple(kv[0] for kv in new_kvs)
        new_V = tuple(kv[1] for kv in new_kvs)
        outputs = res[:, self._output_indices]
        return outputs, (new_K, new_V)

    def eval(self) -> "CompiledHeadless":
        return self

    @property
    def device(self) -> torch.device:
        return self._net.device

    def input_slice(self, name: str, inputs: torch.Tensor) -> torch.Tensor:
        """Return the slice of ``inputs`` for the named input field."""
        for n, s, w in self._input_specs:
            if n == name:
                return inputs[..., s : s + w]
        raise KeyError(f"input field {name!r} not found")

    def debug_value(self, node: "Node") -> Optional[torch.Tensor]:
        """Return the compiled value of ``node`` from the last debug=True forward.

        Requires a prior call to ``__call__(inputs, debug=True)`` or
        ``step(inputs, past, debug=True)``.  Returns ``None`` if the node
        has no residual assignment in any captured state (e.g. it was
        never materialized, or it's a Concatenate whose children aren't
        all present at any single state).

        Raises ``RuntimeError`` if no debug forward has been run yet.
        """
        if self._debug_state is None:
            raise RuntimeError("debug_value() requires a prior debug=True forward pass")
        from torchwright.debug.extraction import (
            extract_compiled_value,
            first_state_with,
        )
        from torchwright.graph.misc import Assert, DebugWatch

        while isinstance(node, (Assert, DebugWatch)):
            node = node.inputs[0]

        ds = self._debug_state
        state = first_state_with(node, ds.ra, ds.ordered_states)
        if state is None:
            return None
        tensor_pair = ds.state_tensor.get(state)
        if tensor_pair is None:
            return None
        res_tensor, _ = tensor_pair
        return extract_compiled_value(node, ds.ra, state, res_tensor)

    # ---- DebugRuntime protocol surface ---------------------------------
    #
    # Shared with torchwright.debug.onnx_debug.OnnxDebugSession so the
    # probes in torchwright/debug/probe.py work on either backend.

    def _capture_states(self, res_stream: torch.Tensor, past_kvs=None):
        """Forward once with per-sublayer residual snapshots.

        Prefill (``past_kvs=None``): ``net.forward(return_states=True)``.
        Decode (``past_kvs`` a list): manual layer walk mirroring
        ``HeadlessTransformer.forward_cached`` plus state capture —
        ``net.forward`` has no KV-cache entrypoint.

        Returns ``(res, new_kvs_or_None, state_tensor)``.
        """
        net = self._net
        state_tensor: Dict = {}
        if past_kvs is None:
            res, all_states = net.forward(res_stream, return_states=True)
            for key, (state, tensor) in all_states.items():
                state_tensor[state] = (tensor, key)
            new_kvs = None
        else:
            res = res_stream
            new_kvs_list = []
            with torch.no_grad():
                for i, layer in enumerate(net.layers):
                    res, kv = layer.attn.forward_cached(res, past_kvs[i])
                    new_kvs_list.append(kv)
                    state_tensor[layer.attn.out_state] = (
                        res,
                        f"layer_{i}_attn_skip_out_state",
                    )
                    res = layer.mlp.forward(res)
                    state_tensor[layer.mlp.out_state] = (
                        res,
                        f"layer_{i}_mlp_out_state",
                    )
            new_kvs = new_kvs_list
        return res, new_kvs, state_tensor

    def debug_layout(self) -> tuple:
        """``(residual_assignment, ordered)`` without running a forward.

        ``ordered`` is ``[(layer_index, state_name, state), ...]`` —
        the post-MLP sublayer states in execution order.
        """
        ra = self._net.residual_assignment
        assert ra is not None, "compiled module has no residual_assignment"
        return ra, _ordered_mlp_state_triples(self._net, ra)

    def run_with_states(
        self,
        prefill: torch.Tensor,
        past_len: int = 0,
        past_kvs=None,
    ) -> tuple:
        """Forward once with state capture; returns ``(ra, ordered, state_tensor)``.

        ``ordered`` is the post-MLP ``(layer_index, name, state)`` triple
        list; ``state_tensor`` maps each captured state to
        ``(residual_tensor, label)``.
        """
        ra, ordered = self.debug_layout()
        res_stream = self._build_res_stream(prefill, past_len=past_len)
        _, _, state_tensor = self._capture_states(res_stream, past_kvs=past_kvs)
        return ra, ordered, state_tensor

    def build_prefill(
        self,
        input_values: Dict[str, torch.Tensor],
        n_pos: int,
    ) -> torch.Tensor:
        """Pack an input-name → tensor dict into the flat row-tensor layout."""
        d_input = max(start + width for _, start, width in self._input_specs)
        out = torch.zeros(n_pos, d_input)
        for name, start, width in self._input_specs:
            if name not in input_values:
                raise ValueError(f"missing input '{name}'")
            out[:, start : start + width] = input_values[name]
        return out

    def capture_attention(
        self,
        layer_index: int,
        prefill: torch.Tensor,
        past_len: int = 0,
        past_kvs=None,
    ) -> tuple:
        """Softmax ``(weights, logits)`` at one attention layer, each
        ``(n_heads, n_queries, n_keys)``."""
        from torchwright.debug.probe import attention_capture

        net = self._net
        with attention_capture(net, layer_index) as captured:
            res_stream = self._build_res_stream(prefill, past_len=past_len)
            with torch.no_grad():
                # The cached path so the patched ``forward_cached`` fires;
                # ``net.forward`` uses the fused kernel path which never
                # calls ``attn.attn.forward_cached``.
                net.forward_cached(res_stream, past_kvs=past_kvs)
        weights, logits = captured["weights"], captured["logits"]
        assert (
            weights is not None and logits is not None
        ), "attention_capture did not fire — hook installed on wrong layer?"
        return weights, logits

    def _run_debug_checks(self, res_stream, past_kvs=None, atol: float = 1e-7):
        """Run forward with state capture, then check consistency, asserts, and watches.

        For prefill (past_kvs=None): uses net.forward(return_states=True).
        For decode (past_kvs provided): manually walks layers to capture
        per-layer residual-stream states.

        After capturing states:
        1. Stashes the captured state on ``self._debug_state`` so
           :meth:`debug_value` can extract node values after the call.
        2. Checks residual-stream self-consistency: for every node that
           appears in multiple captured states, verifies the value at its
           assigned columns is identical across all of them (up to
           ``atol`` per column, to tolerate fp-rounding noise).  A diff
           exceeding ``atol`` means something overwrote the node's
           columns before all consumers read them — a compiler or
           scheduling bug.
        3. Runs Assert predicates (raises on failure) and DebugWatch
           predicates (prints on trigger).

        The check logic itself lives in
        :mod:`torchwright.debug.extraction`, shared with the ONNX debug
        backend so semantics cannot drift between the two.

        Returns (res, new_kvs) where new_kvs is None for prefill or a
        list of (K, V) tuples for decode.
        """
        from torchwright.debug.extraction import (
            check_debug_predicates,
            run_consistency_check,
        )

        net = self._net
        ra = net.residual_assignment
        assert ra is not None

        res, new_kvs, state_tensor = self._capture_states(res_stream, past_kvs=past_kvs)
        ordered_states = [s for _, _, s in _ordered_mlp_state_triples(net, ra)]

        self._debug_state = _DebugState(
            state_tensor=state_tensor,
            ordered_states=ordered_states,
            ra=ra,
        )

        run_consistency_check(ordered_states, state_tensor, ra, atol)
        check_debug_predicates(
            self._asserts, self._watches, ra, ordered_states, state_tensor
        )

        return res, new_kvs


def compile_headless(
    graph: Node,
    *,
    d: int = 1024,
    d_head: int = 16,
    max_layers: int = 400,
    verbose: bool = False,
    device: str = "cpu",
    extra_metadata: Optional[dict] = None,
    d_hidden: Optional[int] = None,
    trim_heads: bool = True,
    optimize: int = 0,
    assume_zero_init: bool = False,
) -> CompiledHeadless:
    """Compile a graph to an in-process callable.

    Returns a :class:`CompiledHeadless` that evaluates the graph via
    :meth:`HeadlessTransformer.forward` behind the standard
    ``module(inputs) -> outputs`` interface.  This is the in-process
    debug/test backend; for production inference, compile a token ONNX
    artifact with :func:`compile_to_onnx`.

    ``graph`` is a single output :class:`Node`; its outputs are gathered
    at the node's natural residual columns.

    ``d_hidden`` is the per-layer MLP hidden width.  Defaults to ``d``
    when omitted; pass an explicit value to decouple the MLP intermediate
    width from the residual stream width.

    ``optimize`` and ``assume_zero_init`` thread straight to
    ``forward_compile`` (same meaning as on :func:`compile_to_onnx`) —
    so this in-process debug backend can reproduce a production
    ``optimize=2`` / ``assume_zero_init=True`` schedule exactly.  The
    defaults reproduce ``forward_compile``'s own defaults (today's
    behavior).
    """
    from torchwright.graph.asserts import collect_debug_nodes

    if not isinstance(graph, Node):
        raise TypeError(
            f"compile_headless expects an output Node as its first "
            f"argument, got {type(graph).__name__}"
        )

    # Unwrap Assert nodes at the output root — compilation strips them from
    # the interior of the graph, but the caller's reference may still point
    # at one, and downstream lookups must match the compiled terminal node.
    all_asserts, all_watches = collect_debug_nodes(graph)
    combined_output = _unwrap_output_node(graph)

    net = forward_compile(
        d=d,
        d_head=d_head,
        output_node=combined_output,
        verbose=verbose,
        max_layers=max_layers,
        device=device,
        d_hidden=d_hidden,
        trim_heads=trim_heads,
        optimize=optimize,
        assume_zero_init=assume_zero_init,
    )

    assert net.residual_assignment is not None
    out_state = net.layers[-1].mlp.out_state
    in_state = net.layers[0].attn.in_state

    input_nodes_list: List[tuple] = []  # (name, width)
    declared_names: set = set()
    for node in net.residual_assignment.get_nodes(in_state):
        if isinstance(node, InputNode):
            indices = net.residual_assignment.get_node_indices(in_state, node)
            input_nodes_list.append((node.name, len(indices)))
            declared_names.add(node.name)
    # Embedding leaves read their raw token-ID input from
    # ``input_values[embedding.input_name]`` during get_input_res_stream.
    # When the user graph doesn't also wire a matching InputNode
    # (common in tests that only pass the Embedding as part of the
    # graph without declaring inputs["token_ids"]), the ID slot still
    # needs an entry in input_specs so CompiledHeadless._build_res_stream
    # can populate the dict from the flat input tensor.  Allocate a
    # 1-wide pass-through slot per distinct embedding input_name.
    from torchwright.graph.embedding import Embedding as _Embedding

    for node in net.residual_assignment.get_nodes(in_state):
        if isinstance(node, _Embedding) and node.input_name not in declared_names:
            input_nodes_list.append((node.input_name, 1))
            declared_names.add(node.input_name)
    input_nodes_list.sort(key=lambda x: x[0])

    node_input_specs: List[tuple] = []
    offset = 0
    for name, width in input_nodes_list:
        node_input_specs.append((name, offset, width))
        offset += width

    # Direct residual-stream gather handles Concatenate output nodes
    # (which compute()'s per-node result dict does not populate).
    node_output_indices = torch.tensor(
        net.residual_assignment.get_node_indices(out_state, combined_output),
        dtype=torch.long,
    )

    return CompiledHeadless(
        net,
        node_input_specs,
        node_output_indices,
        metadata=extra_metadata,
        asserts=all_asserts,
        watches=all_watches,
    )
