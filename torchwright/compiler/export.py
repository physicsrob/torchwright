"""Compile a torchwright graph to a KV-cached ONNX model.

One exporter returning an :class:`OnnxArtifact` (paths + small build
metadata; ``artifact.load()`` / ``artifact.debug_session(...)``):

    compile_to_onnx(output_node, embedding, path, ...)
        Token I/O: token_ids -> logits.  Sidecar format
        ``TOKEN_META_FORMAT`` (currently ``torchwright.token.v6``)
        carries the vocab.  Consumer: ``OnnxTokenModule`` via
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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

if TYPE_CHECKING:
    from torchwright.compiler.residual_assignment import ResidualAssignment

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from torchwright.compiler.forward.compile import (
    forward_compile,
    rms_norm_width_supported,
)
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.compiler.token_model import (
    CompileHeader,
    CompileProfile,
    ReluLayerWeights,
    SwishLayerWeights,
    TokenModelSpec,
    build_token_weights,
    make_layer_callback,
    schedule_provenance,
)
from torchwright.compiler.utils import resolve_n_heads
from torchwright.graph import Concatenate, Embedding, LiteralValue, Node
from torchwright.graph.attn import CAUSAL_MASK_SENTINEL
from torchwright.graph.rope import ROPE_BASE
from torchwright.graph.misc import InputNode

# v6: the token table is physically tied.  The compiler returns the logical
# output in the embedding's exact ordered residual bank, clears every folded
# seed before readout, and the final MatMul transposes ``embed_table`` itself.
# There is no ``lm_head`` initializer and no unembed mask.  The binary layout
# is intentionally incompatible with every earlier token artifact.  (v5 was
# the input-state constant-seed fold into ``embed_table`` rows; v4 the rotary
# positional encoding plus the optional identity RMSNorm.)
TOKEN_META_FORMAT = "torchwright.token.v6"
DEBUG_META_FORMAT = "torchwright.debug.v1"


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


def _schedule_provenance(compiled, optimize: int) -> dict:
    """Selected-schedule provenance plus the independent solver attempt."""
    return schedule_provenance(compiled, optimize).to_dict()


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
    # Machine kind ("relu" | "swish") — which MLP-sublayer emission the
    # artifact carries.  Uniform per network by the compile-time check.
    activation: str = "relu"
    # Emission mode: False means no bias parameters exist in the artifact
    # (biases folded against the pinned constant-1 column; no_bias_plan.md).
    bias: bool = True

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
# "head" axis of width n_heads*d_head, so head ``h`` occupies columns
# (or rows, for W_O) ``range(h*d_head, (h+1)*d_head)``.
_MATRIX_AXES = {
    "attn.W_Q": ("residual_in", "head"),
    "attn.W_K": ("residual_in", "head"),
    "attn.W_V": ("residual_in", "head"),
    "attn.W_O": ("head", "residual_out"),
    "mlp.W_in": ("residual_in", "hidden"),
    # Swish machine only: the gated sublayer's up projection ("mlp.W_in" is
    # the gate there; the ReLU machine has no W_up).
    "mlp.W_up": ("residual_in", "hidden"),
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


def _build_matrix_occupancy(
    compiled, canon, d: int, d_head: int, n_heads: Optional[int] = None
):
    """Build the ``matrices`` table and ``placements`` map for the sidecar.

    Returns ``(matrices, placements, n_heads, d_hidden_per_layer)``.  All
    geometry comes from scalar params / int attributes (never weight
    tensors, which may be trimmed/freed by export time).  Placements are
    keyed by canonical node id; ops with no graph node (e.g. ``cancel``)
    or whose node is unreachable from the output fold under reserved
    ``_<op_type>`` / ``_unreachable`` buckets — they occupy real matrix
    area, so completeness needs them, but they are not user-graph nodes.
    """
    n_heads = resolve_n_heads(d, d_head, n_heads)
    head_dim = n_heads * d_head

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
        if getattr(layer.mlp, "activation", "relu") == "swish":
            shapes["mlp.W_up"] = (d, d_hidden)
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
                cid = canon.get(e.node.node_id)
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
    n_heads: int,
    kind: str,
    input_specs: List[tuple],
    checked_nodes: List[Node],
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
    the attached-check coverage present at compile time (so the loader
    can warn when a rebuilt graph carries fewer checks than the compiled
    one did — checks are node metadata outside the fingerprint, which
    therefore cannot see that).  ``optimize`` records the
    compile-optimization
    level the artifact was built at; ``extra`` is the caller's free-form
    ``extra_metadata`` dict passed straight through (torchwright does not
    interpret its keys), mirroring the meta sidecar's ``extra``.

    State keys correspond one-to-one with the per-layer residual tensor
    names in the emitted ONNX graph: ``"input"`` ↔ ``res_0``,
    ``"L{i}.attn"`` ↔ ``l{i}_res_attn``, ``"L{i}.mlp"`` ↔
    ``l{i}_res_next``.  States whose node→columns table is the same
    dict object as an earlier state's (``duplicate_state`` sharing) are
    stored as ``{"same_as": <key>}`` to keep the file small.

    ``checked_nodes`` come from the source graph (compilation never
    mutates it, so its checks are always where they were attached).
    """
    from torchwright.compiler.graph_identity import (
        canonical_ids,
        debug_fingerprint,
        encode_cols,
        nodes_by_canonical_id,
    )

    out = output_node
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
            cid = canon.get(node.node_id)
            if cid is None:
                # Not reachable from the output via inputs (e.g. the
                # PosEncoding leaf) — cannot be keyed canonically.
                continue
            nodes.setdefault(str(cid), encode_cols(list(cols)))
            first_state.setdefault(str(cid), key)
        state_entries.append({"key": key, "nodes": nodes})

    assert_targets = sorted(
        {
            canon[n.node_id]
            for n in checked_nodes
            if n.node_id in canon and any(c.kind == "assert" for c in n.checks)
        }
    )
    matrices, placements, n_heads, d_hidden_per_layer = _build_matrix_occupancy(
        compiled, canon, d, d_head, n_heads
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
            cid = canon.get(e.node.node_id)
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
            t: Any = getattr(node, attr, None)
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
            icid = canon.get(inp.node_id)
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
        # Machine kind ("relu" | "swish").  Deliberately OUTSIDE the
        # fingerprint (whose topology encoding predates the swish machine and
        # must keep hashing byte-identically for existing artifacts); the
        # debug session cross-checks it against the rebuilt graph explicitly.
        "activation": getattr(compiled, "activation", "relu"),
        # Emission mode — also outside the fingerprint (a compile option,
        # not a graph property); the debug session cross-checks it against
        # the artifact's own initializer set.
        "bias": bool(getattr(compiled, "bias", True)),
        # The univariate-collapse lowering always runs (retired escape
        # hatch, 2026-07-06); recorded top-level like "bias" (never inside
        # "extra") so pre-flip sidecars (False) stay distinguishable.
        "collapse_univariate": True,
        # Same for the v2 general-PL S1 pass (retired the same day);
        # absence identifies pre-v2 sidecars.
        "collapse_pl": True,
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
            # Check counts, not node counts: several checks can attach
            # to one node, and the pre-metadata sidecars counted wrapper
            # nodes — one check each — so the numbers stay comparable.
            "n_asserts": sum(
                1 for n in checked_nodes for c in n.checks if c.kind == "assert"
            ),
            "n_watches": sum(
                1 for n in checked_nodes for c in n.checks if c.kind == "watch"
            ),
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
    n_heads: int,
    dense_inits: list,
    sparse_inits: list,
    per_layer_n_heads: list,
    per_layer_rotary: Optional[list] = None,
    trim_heads: bool = True,
    bias: bool = True,
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
    ``(nh, d, d_head)`` where ``nh <= n_heads``.  Weight matrices
    are emitted at the trimmed size and per-layer reshape constants
    are added so the ONNX graph can reshape to the correct head count.

    Appends each layer's ``nh`` to ``per_layer_n_heads`` for Phase 2.
    """

    class OnnxLayerSink:
        def begin(self, header):
            assert header.d == d and header.d_head == d_head
            assert header.n_heads == n_heads
            assert header.trim_heads == trim_heads and header.bias == bias

        def write_layer(self, i, weights):
            attn = weights.attention
            nh, hd = attn.n_heads, attn.n_heads * d_head
            per_layer_n_heads.append(nh)
            if per_layer_rotary is not None:
                per_layer_rotary.append({"base": attn.rope_base, "d_rot": attn.d_rot})

            def emit(name: str, arr: np.ndarray) -> None:
                dense_tp, sparse_tp = _tensor_to_proto(name, arr)
                _append_proto(dense_tp, sparse_tp, dense_inits, sparse_inits)

            emit(f"l{i}_WQ", attn.wq)
            emit(f"l{i}_WK", attn.wk)
            emit(f"l{i}_WV", attn.wv)
            emit(f"l{i}_WO", attn.wo)
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

            def emit_bias(name, value):
                if value is not None:
                    emit(name, value)

            if isinstance(weights, SwishLayerWeights):
                emit(f"l{i}_Wgate", weights.wgate)
                emit_bias(f"l{i}_bgate", weights.bgate)
                emit(f"l{i}_Wup", weights.wup)
                emit_bias(f"l{i}_bup", weights.bup)
                emit(f"l{i}_Wdown", weights.wdown)
                emit_bias(f"l{i}_bdown", weights.bdown)
            else:
                assert isinstance(weights, ReluLayerWeights)
                emit(f"l{i}_W1", weights.w1)
                emit_bias(f"l{i}_b1", weights.b1)
                emit(f"l{i}_W2", weights.w2)
                emit_bias(f"l{i}_b2", weights.b2)

        def finalize(self, spec, weights):
            # Graph assembly below consumes the finalized semantic values.
            self.spec = spec
            self.token_weights = weights

    return make_layer_callback(
        CompileHeader(d, d_head, trim_heads, bias, n_heads=n_heads), OnnxLayerSink()
    )


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


def _emit_rmsnorm(node, x: str, gain: str, eps: str, out: str, scratch: str) -> None:
    """Emit opset-14 RMSNorm ops: ``out = x / sqrt(mean(x^2, -1) + eps) * gain``.

    ``node`` is the layer's ``make_node`` closure; ``scratch`` namespaces the
    intermediate tensor names.  With the pinned constant forcing
    ``mean(x^2) == 2^(2m)`` exactly, ``sqrt`` returns ``2^m`` and both the
    ``Div`` and the final ``Mul`` (gain ``= 2^m``) are pure exponent shifts —
    the norm is the bit-exact identity.  (ReduceMean takes ``axes`` as an
    attribute in opset 14; it became an input only in opset 18.)
    """
    node("Mul", [x, x], [f"{scratch}_sq"])
    node("ReduceMean", [f"{scratch}_sq"], [f"{scratch}_ms"], axes=[-1], keepdims=1)
    node("Add", [f"{scratch}_ms", eps], [f"{scratch}_msx"])
    node("Sqrt", [f"{scratch}_msx"], [f"{scratch}_rms"])
    node("Div", [x, f"{scratch}_rms"], [f"{scratch}_normed"])
    node("Mul", [f"{scratch}_normed", gain], [out])


def _emit_cached_layer_nodes(
    nodes: list,
    layer_idx: int,
    current_res: str,
    d: int,
    d_head: int,
    n_heads: int,
    d_rot: int,
    scatter_idx_col: str = "_cache_pos_col",
    rms_norm: bool = False,
    rms_eps_name: Optional[str] = None,
    activation: str = "relu",
    bias: bool = True,
) -> str:
    """Emit cached attention + MLP-sublayer nodes for one layer.

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

    # Pre-norm: the attention sublayer reads norm(current_res); the residual
    # skip (below) stays the un-normed current_res, so the stream is never
    # overwritten with normed values.  The norm is the bit-exact identity, and
    # it runs BEFORE the Q/K/V projection — the rotary rotation below acts on
    # the projected (and normed) Q/K, so the pinned-constant residual the norm
    # reads is never rotated and the RMS stays position-independent.
    attn_in = current_res
    if rms_norm:
        attn_in = f"{p}_attn_normed"
        _emit_rmsnorm(
            node,
            current_res,
            f"{p}_input_layernorm",
            cast(str, rms_eps_name),
            attn_in,
            f"{p}_attn_norm",
        )

    def rope_rotate(src: str, dst: str, cos: str, sin: str) -> None:
        """rotate_half RoPE over the rotary front ``d_rot``: rotate the first
        ``d_rot`` dims (``front*cos + rotate_half(front)*sin``) and pass the last
        ``d_head - d_rot`` dims (the NoPE tail) through unchanged.  Matches
        graph/rope.py (vanilla partial rotary, half-split over d_rot);
        modeling_torchwright_custom.py mirrors this op sequence.

        ``d_rot == d_head`` (full rotary) emits the exact pre-partial 6-node
        sequence — byte-for-byte the same graph, so the cancel-head denormal-ULP
        note below still holds and existing exports are unchanged.

        No ``src + (rot - src)`` reconstruction: cross-backend onnxruntime/torch
        agreement is not algebraic — the cancel-head rows that cancel to denormal
        magnitude differ by one denormal ULP regardless of the form (the
        direct-HF parity bound tolerates it; no token or meaningful logit
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
    node("MatMul", [attn_in, f"{p}_WQ"], [f"{p}_Q_flat"])
    node("Reshape", [f"{p}_Q_flat", f"{p}_qkv_view_shape"], [f"{p}_Q_sm"])

    node("MatMul", [attn_in, f"{p}_WK"], [f"{p}_K_flat"])
    # Rotate the new K (sequence-major) by absolute position, so the cache stores
    # already-rotated K (matches the in-process component and HF).  Projected
    # from the normed input, like Q/V above.
    node("Reshape", [f"{p}_K_flat", f"{p}_qkv_view_shape"], [f"{p}_K_sm"])
    rope_rotate(
        f"{p}_K_sm",
        f"delta_K_{layer_idx}",
        "_rope_cos_k",
        "_rope_sin_k",
    )

    node("MatMul", [attn_in, f"{p}_WV"], [f"{p}_V_flat"])
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

    # Pre-norm: the MLP reads norm(res_attn); the residual skip (below) stays
    # the un-normed res_attn.  Bit-exact identity, same as the attention norm.
    mlp_in = f"{p}_res_attn"
    if rms_norm:
        mlp_in = f"{p}_mlp_normed"
        _emit_rmsnorm(
            node,
            f"{p}_res_attn",
            f"{p}_post_attention_layernorm",
            cast(str, rms_eps_name),
            mlp_in,
            f"{p}_mlp_norm",
        )

    # MLP sublayer + skip.  Under bias=False the bias Adds do not exist —
    # the writer folded every bias into the matrices (docs/no_bias_plan.md)
    # — so each projection is its bare MatMul.  Residual tensor names
    # ({p}_res_attn / {p}_res_next) are identical in both modes, keeping
    # the debug session's taps mode-blind.
    if activation == "swish":
        # Gated (SwiGLU) sublayer: down(swish(gate(x)) * up(x)) + x.  Swish is
        # emitted as Mul(z, Sigmoid(z)) — the exact op pair whose fp32
        # saturation profile the A0 probes pinned (tests/docs/
        # test_ort_cpu_saturation.py; ORT fuses it where profitable).
        if bias:
            node("MatMul", [mlp_in, f"{p}_Wgate"], [f"{p}_gate_m"])
            node("Add", [f"{p}_gate_m", f"{p}_bgate"], [f"{p}_gate"])
        else:
            node("MatMul", [mlp_in, f"{p}_Wgate"], [f"{p}_gate"])
        node("Sigmoid", [f"{p}_gate"], [f"{p}_gate_sig"])
        node("Mul", [f"{p}_gate", f"{p}_gate_sig"], [f"{p}_gate_sw"])
        if bias:
            node("MatMul", [mlp_in, f"{p}_Wup"], [f"{p}_up_m"])
            node("Add", [f"{p}_up_m", f"{p}_bup"], [f"{p}_up"])
        else:
            node("MatMul", [mlp_in, f"{p}_Wup"], [f"{p}_up"])
        node("Mul", [f"{p}_gate_sw", f"{p}_up"], [f"{p}_hidden"])
        node("MatMul", [f"{p}_hidden", f"{p}_Wdown"], [f"{p}_down_m"])
        if bias:
            node("Add", [f"{p}_down_m", f"{p}_bdown"], [f"{p}_down_b"])
            node("Add", [f"{p}_res_attn", f"{p}_down_b"], [f"{p}_res_next"])
        else:
            node("Add", [f"{p}_res_attn", f"{p}_down_m"], [f"{p}_res_next"])
    else:
        node("MatMul", [mlp_in, f"{p}_W1"], [f"{p}_l1_m"])
        if bias:
            node("Add", [f"{p}_l1_m", f"{p}_b1"], [f"{p}_l1_b"])
            node("Relu", [f"{p}_l1_b"], [f"{p}_l1_r"])
        else:
            node("Relu", [f"{p}_l1_m"], [f"{p}_l1_r"])
        node("MatMul", [f"{p}_l1_r", f"{p}_W2"], [f"{p}_l2_m"])
        if bias:
            node("Add", [f"{p}_l2_m", f"{p}_b2"], [f"{p}_l2_b"])
            node("Add", [f"{p}_res_attn", f"{p}_l2_b"], [f"{p}_res_next"])
        else:
            node("Add", [f"{p}_res_attn", f"{p}_l2_m"], [f"{p}_res_next"])

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
    d_hidden: Optional[int] = None,
    extra_metadata: Optional[dict] = None,
    cache_stride: Optional[int] = None,
    debug_sidecar: bool = True,
    rms_norm: Optional[bool] = None,
    rms_norm_eps: float = 1e-5,
    rms_norm_const_exp: Optional[int] = None,
    bias: bool = True,
    profile: Optional[Union[CompileProfile, str]] = None,
    n_heads: Optional[int] = None,
    policy: Optional[SchedulingPolicy] = None,
    _solver_seed: Optional[int] = None,
    _force_resolve: bool = False,
) -> OnnxArtifact:
    """Compile a token-I/O graph to a KV-cached ONNX model.

    Returns an :class:`OnnxArtifact` (paths + small build metadata;
    ``artifact.load()`` for the runtime, ``artifact.debug_session(...)``
    for the debug surface).

    ``extra_metadata`` is a free-form dict written under the sidecar's
    ``"extra"`` key (surfaced via ``OnnxTokenModule.metadata``); top-level
    sidecar keys are unchanged.

    ``n_heads`` is the attention-head capacity of each compiled layer. It
    defaults to ``d // d_head``. When explicit, ``n_heads * d_head`` is the
    flattened attention width and may differ from ``d``.

    ``policy`` threads the scheduling policy to ``forward_compile``
    (default: ``SchedulingPolicy()``).  Under the default, standalone
    Linears route to MLP and Adds to attention; ``LEGACY_POLICY`` puts
    both on attention, ``SchedulingPolicy(add_in_attention="never")``
    both on MLP.  On ``optimize>0`` paths the solver's flex routing
    overrides the static choice for eligible nodes.

    Writes three files:
        ``<output_path>``     — the ONNX model
        ``<stem>.meta.json``  — ``{"format": "torchwright.token.v6",
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

    ``bias`` threads to ``forward_compile`` (see its docstring for the fold
    mechanics): ``bias=False`` emits no bias initializers and no bias Adds —
    each MLP projection is its bare MatMul, the standard biasless decoder
    layout.  The emission mode is recorded in the artifact, the token meta,
    and the debug sidecar (``"bias"``).

    The compiler assumes the runtime zero-initialises the residual stream:
    the ONNX runtime always constructs it from zeros plus the input
    projections (see ``HeadlessTransformer.get_input_res_stream`` /
    ``_OnnxRuntime._build_res_stream``, and the embed-table zero-scatter into
    non-allocated columns), so the initially-free residual columns are clean
    on entry and a fresh allocation's first additive write needs no prior
    cancel.  This is now universal across every entry point (the
    ``assume_zero_init`` flag was retired 2026-07); no runtime surface can
    supply a non-zero stream.

    ``rms_norm`` emits a real RMSNorm — pre-norm on each sublayer input plus a
    final norm. Reserved columns hold pinned constants so the forced RMS is an
    exact power of two and the gain makes every learned coordinate bit-exactly
    unchanged. Pre-layer gains are uniform; the final gain is exact zero only
    on the pinned coordinates, clearing them before tied readout without an
    unembed mask. This makes the artifact a stock Llama-style decoder (a
    skeptic can't say "no normalization"); it adds the gain weights as
    initializers and folds the constants into
    ``embed_table`` (no new buffer).  The norm is **on by default**: ``None``
    enables it.  With the norm on, ``d`` must be a **supported width: any
    multiple of 1024 up to 16384, or any power of two** (the full table and
    the mechanism are in ``docs/rms_norm_dmodel.md``); the default (and an
    explicit ``True``) **raises** on an unsupported ``d`` rather than
    silently shipping an un-normalized artifact — pass ``rms_norm=False`` to
    opt out deliberately.  ``rms_norm_eps`` (Llama default ``1e-5``) is
    recorded in the meta; it sits below the forced RMS's LSB, so it does not
    affect bit-exactness.  ``rms_norm_const_exp`` (``q``) overrides the
    pinned-constant exponent for a graph whose deepest-layer energy needs
    more margin (see ``forward_compile``).

    ``_solver_seed`` / ``_force_resolve`` are MEASUREMENT-ONLY (CP-SAT
    expressiveness plan §0) and never set in production configs: they thread
    a reproducible descent seed and a schedule-cache bypass to
    ``forward_compile`` so the sound production draw can be sampled across
    seeds through this real export path.  The relaxed (``_disabled_families``)
    solve-only diagnostic is deliberately NOT exposed here — a relaxed
    schedule is never exported; use ``forward_compile`` directly for it.
    """
    # Validate the cache config up front — a ValueError after the
    # (potentially very long) streaming compile would waste the whole run.
    cache_stride_resolved = _resolve_cache_stride(cache_stride, max_seq_len)

    machine = None
    if profile is not None:
        resolved_profile = CompileProfile(profile)
        machine = resolved_profile.machine
        bias = resolved_profile.bias
        rms_norm = resolved_profile.rms_norm

    # Resolve the norm policy.  RMSNorm is on by default (None -> True): every
    # shipped artifact should carry the real norm.  Refuse an unsupported
    # width with the norm on — fail fast and loudly here (before the long
    # streaming compile) rather than silently dropping it.  ``rms_norm=False``
    # is the explicit opt-out.
    rms_norm_on = True if rms_norm is None else bool(rms_norm)
    if rms_norm_on and not rms_norm_width_supported(d):
        raise ValueError(
            f"rms_norm is on (the default) but d={d} is not a supported "
            f"width. Supported: any multiple of 1024 up to 16384, or any "
            f"power of two (docs/rms_norm_dmodel.md has the full table). "
            f"Use a supported d, or pass rms_norm=False to export without "
            f"the norm."
        )
    n_heads = resolve_n_heads(d, d_head, n_heads)

    # Attached-check coverage for the debug sidecar.  Collection order
    # doesn't matter: compilation never mutates the source graph, so its
    # checks are always where they were attached.
    from torchwright.graph.asserts import collect_debug_nodes

    checked_nodes = collect_debug_nodes(output_node)

    dense_inits: list = []
    sparse_inits: list = []
    per_layer_n_heads: list = []
    per_layer_rotary: list = []

    on_layer_compiled = _make_stream_layer_weights_cb(
        d,
        d_head,
        n_heads,
        dense_inits,
        sparse_inits,
        per_layer_n_heads,
        per_layer_rotary,
        trim_heads=trim_heads,
        bias=bias,
    )

    # --- Phase 1: streaming compile ---------------------------------------
    t0 = time.perf_counter()
    compiled = forward_compile(
        d=d,
        d_head=d_head,
        n_heads=n_heads,
        output_node=output_node,
        verbose=verbose,
        max_layers=max_layers,
        device=None,
        on_layer_compiled=on_layer_compiled,
        trim_heads=trim_heads,
        optimize=optimize,
        bias=bias,
        output_layout_source=embedding,
        machine=machine,
        d_hidden=d_hidden,
        policy=policy,
        # Measurement-only (§0): reproducible descent seed + fresh-solve for
        # the sound production draw.  Never set in production configs.
        _solver_seed=_solver_seed,
        _force_resolve=_force_resolve,
        rms_norm=rms_norm_on,
        rms_norm_eps=rms_norm_eps,
        # None -> forward_compile's default q; an explicit value tunes the
        # pinned constant for a graph whose deepest-layer energy needs it.
        **cast(
            Dict[str, Any],
            (
                {}
                if rms_norm_const_exp is None
                else {"rms_norm_const_exp": rms_norm_const_exp}
            ),
        ),
    )
    rms_spec = compiled.rms_norm_spec
    t_compile = time.perf_counter() - t0
    if verbose and per_layer_n_heads:
        _max_heads = n_heads
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

    embedding_indices = compiled.residual_assignment.get_node_indices(
        in_state, embedding
    )
    # Input-state literal seeds as (column, value) pairs, folded into
    # embed_table rows below (token.v6) — there is no separate
    # constant_values initializer or post-Gather seed Add anymore.
    literal_seeds: List[Tuple[int, float]] = []

    for node in compiled.residual_assignment.get_nodes(in_state):
        indices = compiled.residual_assignment.get_node_indices(in_state, node)
        if isinstance(node, Embedding):
            assert node is embedding, (
                "the token compiler contract permits exactly one reachable "
                "Embedding, and it must be the explicit source"
            )
        elif isinstance(node, LiteralValue):
            # The rotary Δ=0 self-match const-1 column: an input-state
            # LiteralValue explicitly placed there by forward_compile, seeded
            # to 1.0.  The graph's own constants are NOT here — they are
            # JIT-materialized into the per-layer MLP near their consumer.
            for k, idx in enumerate(indices):
                literal_seeds.append((idx, float(node.value[k])))
        elif isinstance(node, (Concatenate, InputNode)):
            # Concatenate is structural; a raw InputNode is populated at
            # forward-time by get_input_res_stream, not folded into embed_table.
            # Both are intentionally not seeded here.
            pass
        else:
            # Only Embedding / LiteralValue / Concatenate / InputNode are
            # expected in the token seed; anything else would be silently
            # dropped by the vanilla-embedding seed below — fail loud instead.
            raise AssertionError(
                f"unexpected input-state node {type(node).__name__} in the "
                f"token seed; only Embedding / LiteralValue / Concatenate / "
                f"InputNode are expected"
            )

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
        out_state, output_node
    )
    assert output_indices == embedding_indices, (
        "tied output bank order differs from the embedding bank: "
        f"embedding={embedding_indices}, output={output_indices}"
    )

    # Fold the residual column placement into one full-width tied table.  Its
    # learned coordinates occupy the ordered held bank; folded seed columns
    # initialize the transformer but are exact zero by final readout.
    # ZERO-INIT CONTRACT.  The table starts as all-zeros and only the
    # embedding, pinned-RMSNorm, and literal-seed columns below are written;
    # every column allocated to a graph node stays zero.  A per-token
    # ``Gather("embed_table", ...)`` therefore hands the transformer a residual
    # stream that is exactly zero in every non-allocated column — which is the
    # zero-init the scheduler assumes when it skips BIRTH-layer dirty cancels
    # for fresh allocations (compile.py; the ``assume_zero_init`` flag was
    # retired in favour of this being universal).  This zeros-scatter IS that
    # contract; ``test_token_onnx_embed_table_zeroes_unallocated_columns`` pins
    # it.  Do not seed non-allocated columns here.
    embed_table = np.zeros((vocab_size, d), dtype=np.float32)
    embed_table[:, embedding_indices] = embed_table_compact

    # Pinned-constant RMSNorm seeds: write each reserved column's constant for
    # EVERY vocab row, so the per-token gather reproduces the constants at
    # every position.  The reserved columns are allocated to no node, so every
    # weight row that reads them is zero — the constants contribute nothing to
    # any transformer matmul, yet their energy dominates mean(x^2) so the
    # forced RMS is an exact power of two.  These are the only genuine new
    # seed constants.
    if rms_spec is not None:
        for j, v in zip(rms_spec.reserved_cols, rms_spec.const_values):
            embed_table[:, j] = v

    # Input-state literal seeds (the const-1 self-match column): folded into
    # every vocab row exactly like the RMSNorm constant above, so the
    # per-token gather reproduces the constant at every position.  The
    # literal's columns are disjoint from the embedding's and the RMSNorm's
    # (residual allocation is pairwise disjoint, I1), so these cells are zero
    # before the fold and the gathered row is bit-identical to the pre-v6
    # ``Gather -> Add(constant_values)`` pair this replaces.
    for idx, val in literal_seeds:
        embed_table[:, idx] = val

    # The backend-neutral builder is the contract for these semantic folds.
    # Keep the local variables used by the mature graph emitter, but replace
    # them with the shared builder's results and assert equivalence while this
    # exporter assembly is incrementally moved behind the sink boundary.
    # v6: there is no ``lm_head`` initializer — the tied readout transposes
    # ``embed_table`` itself (the builder still materializes the compact
    # untied projection for the stock-architecture HF target).
    token_weights = build_token_weights(compiled, output_node, embedding, d)
    assert np.array_equal(embed_table, token_weights.embed_table)
    embed_table = token_weights.embed_table

    # Initializers
    _add_float_init("embed_table", embed_table, dense_inits, sparse_inits)
    # Pinned-constant RMSNorm gain weights (Llama3-named on the HF side): one
    # per pre-attention norm, one per pre-MLP norm, and a final norm.  Each is a
    # uniform (d,) vector = 2^m, which cancels the forced rms = 2^m exactly.  A
    # single shared eps scalar feeds every norm.
    if rms_spec is not None:
        gain_vec = token_weights.norm_gain
        assert gain_vec is not None
        for i in range(n_layers):
            _add_float_init(
                f"l{i}_input_layernorm", gain_vec, dense_inits, sparse_inits
            )
            _add_float_init(
                f"l{i}_post_attention_layernorm", gain_vec, dense_inits, sparse_inits
            )
        final_gain_vec = gain_vec.copy()
        final_gain_vec[list(rms_spec.reserved_cols)] = 0.0
        _add_float_init("final_norm", final_gain_vec, dense_inits, sparse_inits)
        _add_float_init(
            "_rms_eps",
            np.array([rms_spec.eps], dtype=np.float32),
            dense_inits,
            sparse_inits,
        )
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
    # full-width tied unembed.
    nodes: list = []

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # RoPE: bake the global rope inits when any layer is rotary; the preamble
    # emits cos/sin from cache_position when active (no-op otherwise).
    rope_base_val, rope_d_rot_val = _add_rope_inits(
        per_layer_rotary, d_head, dense_inits
    )
    cast(Any, on_layer_compiled).token_model_sink.finalize(
        TokenModelSpec(
            d=d,
            d_head=d_head,
            max_seq_len=max_seq_len,
            vocab=tuple(embedding.tokenizer.vocab),
            vocab_size=vocab_size,
            activation=cast(Literal["relu", "swish"], compiled.activation),
            bias=bool(bias),
            rms_norm=rms_spec is not None,
            rms_norm_eps=float(rms_spec.eps if rms_spec else rms_norm_eps),
            rope_base=rope_base_val,
            d_rot=rope_d_rot_val,
            n_layers=n_layers,
            per_layer_n_heads=tuple(per_layer_n_heads),
            per_layer_d_hidden=tuple(
                int(getattr(layer.mlp, "d_hidden", d)) for layer in compiled.layers
            ),
            schedule_provenance=schedule_provenance(compiled, optimize),
            extra_metadata=dict(extra_metadata or {}),
        ),
        token_weights,
    )
    _emit_cached_preamble(nodes)
    # the residual seed (no projection).  Position is a rotation applied inside
    # attention (RoPE), so there is no additive position table — the seed is
    # the token embedding alone, whose rows carry the pinned RMSNorm constant
    # and the const-1 self-match column (both folded above, token.v6).
    add("Gather", ["embed_table", "token_ids"], ["res_0"], axis=0)

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
            rms_norm=rms_spec is not None,
            rms_eps_name="_rms_eps" if rms_spec is not None else None,
            activation=compiled.activation,
            bias=bias,
        )

    # Final norm before the unembed (Llama-style), the bit-exact identity.  It
    # reads ALL d columns for mean(x^2), so the identity needs the total non-pinned
    # energy bounded — which forward_compile's _certify_rms_norm_energy guarantees
    # from the graph's value ranges (cancel-freed columns are zero, reassigned ones
    # are owned by a snapshot node; either way the per-column bound holds).  Every
    # column must also stay finite: the pinned constant is finite by construction.
    if rms_spec is not None:
        _emit_rmsnorm(
            add, current_res, "final_norm", "_rms_eps", "_final_normed", "_final_norm"
        )
        current_res = "_final_normed"

    # Physical tying: the same full-width table feeds token Gather and final
    # MatMul.  The compiler placed the logical output in its learned bank and
    # cleared the const-one seed; final_norm zeroed only the pinned RMS seeds.
    add("Transpose", ["embed_table"], ["_embed_table_T"], perm=[1, 0])
    add("MatMul", [current_res, "_embed_table_T"], ["logits"])

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
        # RoPE on one global grid; sibling backends rebuild the rotation from
        # the base and partial-rotary width d_rot (== d_head for full rotary).
        "rope_base": rope_base_val,
        "d_rot": rope_d_rot_val,
        # Whether a real (identity) RMSNorm is emitted, and its epsilon — the HF
        # Sibling backends need eps in their config; unlike gain, it is not a tensor.
        "rms_norm": rms_spec is not None,
        "rms_norm_eps": rms_spec.eps if rms_spec is not None else None,
        # Machine kind ("relu" | "swish"): which MLP-sublayer emission the
        # artifact carries (l{i}_W1/W2 + Relu vs l{i}_Wgate/Wup/Wdown +
        # Sigmoid·Mul).  Uniform per network by the compile-time check.
        "activation": compiled.activation,
        # Emission mode: False means no bias initializers or bias Adds exist
        # anywhere (docs/no_bias_plan.md; biases folded into the matrices
        # against the pinned constant-1 column).
        "bias": bool(bias),
        # The univariate-collapse lowering always runs (default flipped
        # and the escape hatch retired 2026-07-06 —
        # docs/univariate_collapse_plan.md); False identifies pre-flip
        # artifacts.
        "collapse_univariate": True,
        # Same for the v2 general-PL S1 pass (retired the same day);
        # absence identifies pre-v2 artifacts.
        "collapse_pl": True,
        # Solver provenance: distinguishes a real CP-SAT schedule (status
        # OPTIMAL/FEASIBLE/CACHED) from the heuristic fallback (UNKNOWN/
        # INFEASIBLE with optimize>0) — a fallback artifact is value-correct
        # but unoptimized, and nothing else in the artifact records that.
        "schedule": _schedule_provenance(compiled, optimize),
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
            n_heads=n_heads,
            kind="token",
            # The token graph reads its ids from the Embedding node's
            # input slot; one 1-wide column matches the convention
            # CompiledHeadless uses for the same graph.
            input_specs=[(embedding.input_name, 0, 1)],
            checked_nodes=checked_nodes,
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
        activation=compiled.activation,
        bias=bool(bias),
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
        checked_nodes: Optional[List] = None,
    ) -> None:
        self._net = net
        # input_specs: list of (name, start_col, width) in input-tensor column order.
        self._input_specs = list(input_specs)
        self._output_indices = output_indices
        self.input_names: List[str] = [name for name, _, _ in input_specs]
        self.metadata: dict = dict(metadata or {})
        # Source-graph nodes carrying attached checks (node.checks) —
        # re-checked against compiled residual values on debug=True.
        self._checked_nodes = list(checked_nodes) if checked_nodes else []
        self._debug_state: Optional[_DebugState] = None

        # KV cache shape metadata — discovered from the compiled transformer
        # so empty_past() can build zero-length tensors of the right shape.
        # After head trimming each layer may have a different n_heads.
        self._per_layer_n_heads = [layer.attn.attn.n_heads for layer in net.layers]
        self._d_head = net.layers[0].attn.attn.d_head
        self._n_layers = len(net.layers)

    @property
    def n_layers(self) -> int:
        """Compiled layer count — the number the depth-oriented passes
        (and their before/after reports) are measured by."""
        return self._n_layers

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
        and runs every attached check (assert-kind raises, watch-kind
        prints).  ``debug_atol`` is the maximum
        per-column drift tolerated by the residual-stream self-consistency
        check; diffs at or below ``debug_atol`` are treated as fp-rounding
        noise.
        """
        res_stream = self._build_res_stream(inputs, past_len=0)
        if debug and self._checked_nodes:
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
                run every attached check (assert-kind raises,
                watch-kind prints).
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
        if debug and self._checked_nodes:
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
        3. Runs every attached check (assert-kind raises on failure,
           watch-kind prints on trigger).

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
        check_debug_predicates(self._checked_nodes, ra, ordered_states, state_tensor)

        return res, new_kvs


def compile_headless(
    graph: Node,
    *,
    d: int = 1024,
    d_head: int = 16,
    n_heads: Optional[int] = None,
    max_layers: int = 400,
    verbose: bool = False,
    device: str = "cpu",
    extra_metadata: Optional[dict] = None,
    d_hidden: Optional[int] = None,
    trim_heads: bool = True,
    optimize: int = 0,
    bias: bool = True,
    output_layout_source: Optional[Node] = None,
    policy: Optional[SchedulingPolicy] = None,
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

    ``n_heads`` is the per-layer attention-head capacity. It defaults to
    ``d // d_head``; pass it explicitly to decouple the flattened attention
    width ``n_heads * d_head`` from ``d``.

    ``optimize``, ``bias``, ``output_layout_source``, and ``policy``
    thread straight to ``forward_compile`` (same meaning as on
    :func:`compile_to_onnx`) — so this in-process debug backend can
    reproduce a production ``optimize=2`` / ``bias=False`` schedule
    exactly.  ``compile_to_onnx`` always compiles
    with ``output_layout_source=<the graph's Embedding>`` (the tied
    token.v6 held-bank contract), so reproducing a token artifact's
    schedule requires passing the same Embedding here; leaving it ``None``
    compiles an untied schedule with a different structure and CP-SAT
    fingerprint.  The residual stream is always zero-initialised (the
    ``assume_zero_init`` flag was retired 2026-07).
    """
    from torchwright.graph.asserts import collect_debug_nodes

    if not isinstance(graph, Node):
        raise TypeError(
            f"compile_headless expects an output Node as its first "
            f"argument, got {type(graph).__name__}"
        )

    # The collected checked nodes are re-checked against compiled values
    # on debug=True forwards (compilation never mutates the source
    # graph, so its checks are always where they were attached).
    checked_nodes = collect_debug_nodes(graph)
    combined_output = graph

    net = forward_compile(
        d=d,
        d_head=d_head,
        n_heads=n_heads,
        output_node=combined_output,
        verbose=verbose,
        max_layers=max_layers,
        device=device,
        d_hidden=d_hidden,
        trim_heads=trim_heads,
        optimize=optimize,
        bias=bias,
        output_layout_source=output_layout_source,
        policy=policy,
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
        checked_nodes=checked_nodes,
    )
