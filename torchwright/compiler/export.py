"""Compile a torchwright graph to a KV-cached ONNX model.

Two exporters, symmetric:

    compile_to_onnx(output_node, pos_encoding, embedding, path, ...)
        Token I/O: token_ids -> logits.  Sidecar format
        ``torchwright.token.v1`` carries the vocab.  Consumer:
        :mod:`torchwright.compiler.repl`.

    compile_headless_to_onnx(output_node, pos_encoding, path, ...)
        Float I/O: inputs -> outputs.  Sidecar format
        ``torchwright.headless.v1`` carries the alphabetically-ordered
        input column names.  Consumer:
        :mod:`torchwright.compiler.onnx_load`.

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

**Windowed-cache variant** (``cache_window=C``, the attention-sink +
sliding-window pattern — StreamingLLM sinks / Mistral sliding window):
the committed cache is a fixed ``C``-slot host-managed window instead of
one-slot-per-position.  Every pass binds ``past_K_i`` exactly
``(C + n_new, nh, d_head)``: slots ``[0, C)`` hold committed rows placed
by the host (any policy — e.g. a permanent sink prefix plus a ring that
wraps), and the new rows scatter in-graph to the constant staging tail
``[C, C + n_new)``.  Committed slot ``j`` is visible iff ``j <
cache_position[0]`` (slots the host has written so far — requires the
host to fill committed slots in slot order until all ``C`` are written
once; after that every committed slot stays visible and eviction is
physical overwrite, not masking).  Staging slot ``t`` is visible iff its
position ``cache_position[t] <= p`` (the causal triangle).  The graph
never learns which position a committed slot holds — it doesn't need
to: committed rows are all from previous passes, hence always causally
visible.  ``delta_K_i`` outputs are unchanged; the host persists them at
slots of its choosing (the window policy lives entirely host-side).
Positions stay ABSOLUTE throughout (pos-encoding gather, recency
scores), so reads whose span exceeds what the host keeps resident
return whatever the mask still exposes — windowed output equals
unbounded output ONLY IF every attention read lands on a slot the host
kept (the span condition; callers must size ``C`` against their worst
read span).  Mask and scatter indices still derive solely from
``cache_position``, with no Shape/CPU nodes at all in windowed mode —
the capture story strictly improves.

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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from torchwright.compiler.residual_assignment import ResidualAssignment

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from torchwright.compiler.forward.compile import forward_compile
from torchwright.graph import Concatenate, Embedding, LiteralValue, Node, PosEncoding
from torchwright.graph.attn import CAUSAL_MASK_SENTINEL
from torchwright.graph.misc import Assert, DebugWatch, InputNode

HEADLESS_META_FORMAT = "torchwright.headless.v1"
TOKEN_META_FORMAT = "torchwright.token.v1"
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


def _write_headless_meta(
    onnx_path: str,
    input_names: List[str],
    extra: Optional[dict] = None,
    cache_stride: Optional[int] = None,
    cache_window: Optional[int] = None,
) -> str:
    """Write the headless sidecar JSON, optionally with an ``extra`` dict.

    ``extra`` is a free-form dict for per-export metadata (e.g. DOOM's
    ``rows_per_patch`` — surfaced to the host via
    :class:`OnnxHeadlessModule.metadata`). Kept totally general so the
    compiler layer has no project-specific keys.

    ``cache_stride`` is the full static slot count ``S`` — the loaders
    need it from the sidecar because ``past_K_i``'s first dim is the
    symbolic ``cache_slots``, not a readable int.

    ``cache_window`` marks the windowed-cache protocol variant (the
    committed slot count ``C``); absent for the default unbounded
    protocol.  A windowed model's bindings must be exactly
    ``C + n_new`` wide — loaders that don't know the key must not be
    fed windowed models (they'd bind prefix views and fail loudly on
    the mask width).
    """
    meta: dict = {
        "format": HEADLESS_META_FORMAT,
        "input_names": list(input_names),
    }
    if cache_stride is not None:
        meta["cache_stride"] = int(cache_stride)
    if cache_window is not None:
        meta["cache_window"] = int(cache_window)
    if extra:
        meta["extra"] = dict(extra)
    return _write_meta(onnx_path, meta)


def debug_meta_path_for(onnx_path: str) -> str:
    base, _ = os.path.splitext(onnx_path)
    return base + ".debug.json"


def _write_debug_sidecar(
    onnx_path: str,
    *,
    compiled,
    output_node: Node,
    pos_encoding: PosEncoding,
    d: int,
    d_head: int,
    kind: str,
    input_specs: List[tuple],
    asserts: List["Assert"],
    watches: List["DebugWatch"],
    cache_stride: int,
    cache_window: Optional[int],
    verbose: bool,
) -> str:
    """Write ``<stem>.debug.json`` — everything OnnxDebugSession needs.

    The sidecar carries the residual assignment keyed by CANONICAL node
    id (see :mod:`torchwright.compiler.graph_identity`) per capture
    state, a structural fingerprint of the compiled graph for rebuild
    validation, and the Assert/DebugWatch coverage present at compile
    time (so the loader can warn when a rebuilt graph carries fewer
    checks than the compiled one did — the fingerprint is deliberately
    wrapper-transparent and cannot see that).

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
        state_entries.append({"key": key, "nodes": nodes})

    assert_targets = sorted(
        {
            canon[unwrap_debug(a.inputs[0]).node_id]
            for a in asserts
            if unwrap_debug(a.inputs[0]).node_id in canon
        }
    )
    payload = {
        "format": DEBUG_META_FORMAT,
        "kind": kind,  # "token" | "headless"
        "fingerprint": debug_fingerprint(out, pos_encoding, d=d, d_head=d_head),
        "d": d,
        "d_head": d_head,
        "n_layers": len(compiled.layers),
        "input_specs": [list(spec) for spec in input_specs],
        "cache_stride": int(cache_stride),
        "cache_window": int(cache_window) if cache_window is not None else None,
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
# Positional encoding buffer (numpy; no torch dependency at runtime)
# ---------------------------------------------------------------------------


def _compute_pos_encoding(d_pos: int, max_seq_len: int) -> np.ndarray:
    """Precomputed pos encoding buffer for the ONNX ``pos_encoding_full``
    initializer.  Delegates to :meth:`PosEncoding.get_pos_encoding` so
    the ONNX graph and ``HeadlessTransformer.compute`` share one source
    of truth — any drift would silently break reference parity.
    """
    return (
        PosEncoding(d_pos)
        .get_pos_encoding(max_seq_len)
        .numpy()
        .astype(np.float32, copy=False)
    )


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


# A committed slot the host has not written yet must read as "infinitely
# far in the future" so the causal Greater masks it for every query.  Any
# value > max representable position works; 2^62 keeps clear of int64
# overflow in the comparison (no arithmetic ever touches it).
_FAR_FUTURE_SLOT_POS = 1 << 62


def _add_windowed_scalar_inits(dense_inits: list, cache_window: int) -> None:
    """Scalar initializers used only by the windowed-cache preamble.

    All int64 scalars; initializer-fed, so GPU-materialized on the CUDA
    EP (no Memcpy — same argument as :func:`_add_scalar_inits`).
    """
    for name, val in (
        # already-written committed slots: "position -1" — visible to
        # every query (-1 > p is never true)
        ("_i64_neg1_s", -1),
        # not-yet-written committed slots: masked for every query
        ("_i64_far_future_s", _FAR_FUTURE_SLOT_POS),
        # the staging-tail offset: new row r scatters to slot C + r
        ("_i64_cwin_s", cache_window),
    ):
        dense_inits.append(
            helper.make_tensor(
                name=name,
                data_type=TensorProto.INT64,
                dims=[],
                vals=np.array(val, dtype=np.int64).tobytes(),
                raw=True,
            )
        )


# ---------------------------------------------------------------------------
# Streaming weight emission callback (shared by both exporters)
# ---------------------------------------------------------------------------


def _make_stream_layer_weights_cb(
    d: int,
    d_head: int,
    dense_inits: list,
    sparse_inits: list,
    per_layer_n_heads: list,
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


def _emit_cached_preamble(
    nodes: list, cache_window: Optional[int] = None
) -> None:
    """Emit nodes producing ``pos``, ``mask_bool_3d`` and ``_cache_pos_col``.

    Requires:
      - graph input ``cache_position``: int64 ``(n_new,)`` — the absolute
        positions of the new rows (``[base, base+1, …]``).  The ONLY
        position fact the host provides; the causal mask and the
        positional-encoding rows both derive from it in-graph.
      - graph inputs ``past_K_0``…: sequence-major KV cache with a
        SYMBOLIC first dim ``cache_slots`` — the bound length ``S_eff``
        (a prefix window of the runtime's full-stride cache buffer) sets
        this step's attention width.  The mask width derives from it via
        ``Shape(past_K_0)`` below.
      - initializer ``arange_S``: int64 ``(S,)`` baked ``[0 .. S)``, where
        ``S = cache_stride`` is the FULL static slot count (the maximum
        any binding may use).
      - initializer ``pos_encoding_full``: (max_seq_len, d_pos)
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
      - ``pos``: (n_new, d_pos) float — ``pos_encoding_full`` rows gathered
        at ``cache_position``.
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

    **Windowed mode** (``cache_window=C``): the same three tensors plus
    ``_write_slot_col`` (the layers' ScatterND indices — the staging
    tail ``[C, C+n_new)``, computed as ``C + (cache_position - base)``;
    ``_cache_pos_col`` then serves the mask only).  The mask compares
    the query positions against a per-slot vector ``_slot_pos``:
    committed slots map to ``-1`` (written → always visible) or
    ``2^62`` (unwritten → never visible) via ``arange_C < base``;
    staging slots map to their own ``cache_position`` (the causal
    triangle).  ``arange_S`` must be baked at length exactly ``C`` (the
    binding is ``C + n_new`` wide; a mismatch fails loudly at the
    mask-broadcast).  No ``Shape`` chain at all in this mode — every
    mask/scatter tensor is GPU-resident elementwise int64 arithmetic on
    ``cache_position`` and initializers, so the Memcpy invariant holds
    with zero CPU-resident nodes.
    """

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    if cache_window is None:
        # S_eff = bound first dim of the cache; slots = arange_S[:S_eff].
        # _axes0_1d/_axes1_1d double as the [0]/[1] bounds constants.
        add("Shape", ["past_K_0"], ["_pastK0_shape"])  # (3,) int64, CPU
        add(
            "Slice", ["_pastK0_shape", "_axes0_1d", "_axes1_1d", "_axes0_1d"],
            ["_s_eff_1d"],
        )  # (1,) = [S_eff], CPU
        add(
            "Slice", ["arange_S", "_axes0_1d", "_s_eff_1d", "_axes0_1d"],
            ["_slots"],
        )  # (S_eff,) GPU data, CPU bounds
        add("Unsqueeze", ["_slots", "_axes0_1d"], ["_slots_row"])  # (1, S_eff)
        add(
            "Unsqueeze", ["cache_position", "_axes1_1d"], ["_cache_pos_col"]
        )  # (n_new, 1)
        add(
            "Greater", ["_slots_row", "_cache_pos_col"], ["_mask_bool"]
        )  # (n_new, S_eff)
        add(
            "Unsqueeze", ["_mask_bool", "_axes0_1d"], ["mask_bool_3d"]
        )  # (1, n_new, S_eff)
        add("Gather", ["pos_encoding_full", "cache_position"], ["pos"], axis=0)
        return

    # --- Windowed mode: fixed C committed slots + n_new staging slots. ---
    # base = cache_position[0], the committed-position count.  The [0]/[1]
    # axes constants double as the Slice bounds, exactly as above.
    add(
        "Slice", ["cache_position", "_axes0_1d", "_axes1_1d", "_axes0_1d"],
        ["_base_1d"],
    )  # (1,) = [base], GPU
    # Committed slot j: written iff j < base (the host fills committed
    # slots in slot order until all C are written once — after that this
    # is uniformly true).  Written -> "position -1" (visible to every
    # query); unwritten -> far-future (masked for every query).
    add("Less", ["arange_S", "_base_1d"], ["_committed_written"])  # (C,) bool
    add(
        "Where", ["_committed_written", "_i64_neg1_s", "_i64_far_future_s"],
        ["_committed_slot_pos"],
    )  # (C,) int64
    # Staging slot t holds the new row at cache_position[t]; appending
    # cache_position itself gives the causal triangle under the same
    # Greater comparison the default mask uses.
    add(
        "Concat", ["_committed_slot_pos", "cache_position"], ["_slot_pos"],
        axis=0,
    )  # (C + n_new,)
    add("Unsqueeze", ["_slot_pos", "_axes0_1d"], ["_slot_pos_row"])  # (1, C+n_new)
    add(
        "Unsqueeze", ["cache_position", "_axes1_1d"], ["_cache_pos_col"]
    )  # (n_new, 1)
    add(
        "Greater", ["_slot_pos_row", "_cache_pos_col"], ["_mask_bool"]
    )  # (n_new, C+n_new)
    add(
        "Unsqueeze", ["_mask_bool", "_axes0_1d"], ["mask_bool_3d"]
    )  # (1, n_new, C+n_new)
    add("Gather", ["pos_encoding_full", "cache_position"], ["pos"], axis=0)
    # ScatterND indices: new row r -> staging slot C + r.  Derived as
    # C + (cache_position - base) — GPU-resident elementwise int64, no
    # Range op (a CPU-produced dynamic-size tensor feeding a GPU op
    # would be a Memcpy; see the capture invariant above).
    add("Sub", ["cache_position", "_base_1d"], ["_stage_offsets"])  # (n_new,)
    add("Add", ["_stage_offsets", "_i64_cwin_s"], ["_write_slots"])  # (n_new,)
    add("Unsqueeze", ["_write_slots", "_axes1_1d"], ["_write_slot_col"])  # (n_new, 1)


def _emit_cached_layer_nodes(
    nodes: list,
    layer_idx: int,
    current_res: str,
    d: int,
    d_head: int,
    n_heads: int,
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
    ScatterND row indices.  Default ``_cache_pos_col`` = slot ==
    position (the unbounded protocol); the windowed preamble emits
    ``_write_slot_col`` (the constant staging tail) instead.

    Returns the name of the next residual stream tensor.
    """
    p = f"l{layer_idx}"

    def node(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # Project Q, K_new, V_new from the new rows in sequence-major
    # (n_new, n_heads, d_head).  The deltas (new rows only) are the graph
    # outputs the runtime writes into its owned cache tail; the reshape
    # constant l{i}_qkv_view_shape = [0, n_heads, d_head] copies n_new from
    # the flat (n_new, hd) projection.
    node("MatMul", [current_res, f"{p}_WQ"], [f"{p}_Q_flat"])
    node("Reshape", [f"{p}_Q_flat", f"{p}_qkv_view_shape"], [f"{p}_Q_sm"])

    node("MatMul", [current_res, f"{p}_WK"], [f"{p}_K_flat"])
    node("Reshape", [f"{p}_K_flat", f"{p}_qkv_view_shape"], [f"delta_K_{layer_idx}"])

    node("MatMul", [current_res, f"{p}_WV"], [f"{p}_V_flat"])
    node("Reshape", [f"{p}_V_flat", f"{p}_qkv_view_shape"], [f"delta_V_{layer_idx}"])

    # Attention runs in the ORIGINAL head-major Transpose+MatMul form — the
    # ONNX Einsum form is numerically broken on the CUDA EP for this graph
    # (wrong argmax from the very first token), so we keep the exact ops the
    # pre-cache-rewrite export used.  Transpose the sequence-major Q / past /
    # delta into head-major (n_heads, seq, d_head); the past inputs and delta
    # outputs stay sequence-major for the runtime's contiguous-slice binding.
    node("Transpose", [f"{p}_Q_sm"], [f"{p}_Q"], perm=[1, 0, 2])
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
    node("MatMul", [f"{p}_Q", f"{p}_K_T"], [f"{p}_logits"])
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


def _kv_io_value_info(
    per_layer_n_heads: List[int], d_head: int
) -> tuple[list, list]:
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
    cache_window: Optional[int] = None,
) -> int:
    """Resolve the static cache slot count ``S`` (default: max_seq_len).

    ``S`` must be <= max_seq_len because ``Gather(pos_encoding_full,
    cache_position)`` indexes a table sized by max_seq_len, and a compiled
    model can never serve positions beyond its pos-encoding buffer anyway.

    In windowed mode (``cache_window=C``) the committed slot count IS the
    stride: ``arange_S`` is baked at length exactly ``C`` and the sidecar's
    ``cache_stride`` reads ``C``.  ``cache_stride`` and ``cache_window``
    are mutually exclusive — the window fixes the slot count, so a
    separate stride has nothing left to mean.  ``C > max_seq_len`` is
    rejected: such a window could never fill (positions cap at
    max_seq_len) and signals caller confusion, even though it would be
    mechanically harmless.
    """
    if cache_window is not None:
        if cache_stride is not None:
            raise ValueError(
                "cache_stride and cache_window are mutually exclusive: the "
                "windowed cache fixes the committed slot count at cache_window"
            )
        c = int(cache_window)
        if not (1 <= c <= max_seq_len):
            raise ValueError(
                f"cache_window {c} must be in [1, max_seq_len={max_seq_len}]: "
                f"a window wider than the position space can never fill"
            )
        return c
    s = max_seq_len if cache_stride is None else int(cache_stride)
    if not (1 <= s <= max_seq_len):
        raise ValueError(
            f"cache_stride {s} must be in [1, max_seq_len={max_seq_len}]: the "
            f"static cache cannot hold positions the pos-encoding table lacks"
        )
    return s


def compile_headless_to_onnx(
    output_node: Node,
    pos_encoding: PosEncoding,
    output_path: str,
    d: int = 1024,
    d_head: int = 16,
    max_seq_len: int = 512,
    max_layers: int = 200,
    verbose: bool = True,
    extra_metadata: Optional[dict] = None,
    d_hidden: Optional[int] = None,
    trim_heads: bool = True,
    assume_zero_init: bool = True,
    cache_stride: Optional[int] = None,
    cache_window: Optional[int] = None,
    debug_sidecar: bool = True,
) -> None:
    """Compile a float-I/O graph to a KV-cached ONNX model.

    Writes three files:
        ``<output_path>``     — the ONNX model
        ``<stem>.meta.json``  — ``{"format": "torchwright.headless.v1",
                                   "input_names": [...]}``
        ``<stem>.debug.json`` — the debug sidecar consumed by
                                :class:`torchwright.debug.onnx_debug.
                                OnnxDebugSession` (see
                                :func:`compile_to_onnx`).  Disable with
                                ``debug_sidecar=False``.

    The graph speaks the static-cache prefill/decode protocol:
        inputs:  inputs (n_new, d_input), cache_position (n_new,) int64,
                 past_K_i, past_V_i (cache_slots, n_heads, d_head)
                 [cache_slots SYMBOLIC: the bound prefix length S_eff]
        outputs: outputs (n_new, d_output),
                 delta_K_i, delta_V_i (n_new, n_heads, d_head)

    K/V are sequence-major; ``past_K_i`` is a contiguous prefix view
    ``cache[: S_eff]`` of the feeder's full-stride cache (slots at
    positions >= the committed length zero-filled; any
    ``base + n_new <= S_eff <= cache_stride`` is valid — stride
    bucketing), ``delta_K_i`` is the new rows only, persisted by the
    runtime into its owned cache slots after the run.  Prefill = zero
    past + cache_position [0..n).  Decode = cache_position [base]
    against a covering prefix binding.  Binding the full buffer
    reproduces the previous static-dim behavior exactly.

    ``cache_stride`` is the FULL slot count ``S`` (the ``arange_S`` mask
    constant's length, the maximum any binding may use, and the sidecar
    meta's ``cache_stride`` key); defaults to ``max_seq_len``.  A
    compiled model hard-caps at ``prefill + decode <= S``.

    ``cache_window=C`` selects the windowed-cache protocol instead (see
    the module docstring): bindings are exactly ``C + n_new`` wide,
    committed slot placement is the host's policy, positions stay
    absolute and may run to ``max_seq_len`` regardless of ``C``.
    Mutually exclusive with ``cache_stride``; the sidecar carries both
    ``cache_stride == C`` and the ``cache_window`` discriminator key.

    ``d_hidden`` is the per-layer MLP hidden width.  Defaults to ``d``
    when omitted; pass an explicit value to decouple the MLP intermediate
    width from the residual stream width.

    ``assume_zero_init`` defaults to ``True`` (see :func:`compile_to_onnx`):
    the ONNX runtime always builds the residual stream from zeros plus the
    input projections, so BIRTH-layer dirty-column cancels are unnecessary and
    skipping them keeps the CP-SAT attention-head cumulative from being
    over-tight under width pressure.
    """
    # Validate the cache config up front — a ValueError after the
    # (potentially very long) streaming compile would waste the whole run.
    cache_stride_resolved = _resolve_cache_stride(
        cache_stride, max_seq_len, cache_window
    )

    # Assert/DebugWatch coverage must be collected BEFORE forward_compile
    # strips both wrapper kinds from the graph in-place.
    from torchwright.graph.asserts import collect_debug_nodes

    all_asserts, all_watches = collect_debug_nodes(output_node)

    dense_inits: list = []
    sparse_inits: list = []
    per_layer_n_heads: list = []

    on_layer_compiled = _make_stream_layer_weights_cb(
        d,
        d_head,
        dense_inits,
        sparse_inits,
        per_layer_n_heads,
        trim_heads=trim_heads,
    )

    # --- Phase 1: streaming compile ---------------------------------------
    t0 = time.perf_counter()
    compiled = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=verbose,
        max_layers=max_layers,
        device=None,
        on_layer_compiled=on_layer_compiled,
        d_hidden=d_hidden,
        trim_heads=trim_heads,
        assume_zero_init=assume_zero_init,
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

    input_nodes_list: List[tuple] = []
    pos_indices: Optional[List[int]] = None
    constant_values = np.zeros(d, dtype=np.float32)

    for node in compiled.residual_assignment.get_nodes(in_state):
        indices = compiled.residual_assignment.get_node_indices(in_state, node)
        if isinstance(node, InputNode):
            input_nodes_list.append((node.name, indices))
        elif isinstance(node, PosEncoding):
            pos_indices = indices
        elif isinstance(node, LiteralValue):
            for k, idx in enumerate(indices):
                constant_values[idx] = float(node.value[k])
        elif isinstance(node, (Concatenate, Embedding)):
            pass

    assert len(input_nodes_list) > 0, "No InputNode found in residual assignment"
    assert pos_indices is not None, "No PosEncoding node found in residual assignment"

    input_nodes_list.sort(key=lambda x: x[0])
    input_names = [name for name, _ in input_nodes_list]

    all_input_indices: List[int] = []
    for _, idx in input_nodes_list:
        all_input_indices.extend(idx)
    d_input = len(all_input_indices)

    input_proj = np.zeros((d_input, d), dtype=np.float32)
    for k, idx in enumerate(all_input_indices):
        input_proj[k, idx] = 1.0

    d_pos = len(pos_indices)
    pos_proj = np.zeros((d_pos, d), dtype=np.float32)
    for k, idx in enumerate(pos_indices):
        pos_proj[k, idx] = 1.0

    pos_encoding_buf = _compute_pos_encoding(d_pos, max_seq_len)

    output_indices = compiled.residual_assignment.get_node_indices(
        out_state, _unwrap_output_node(output_node)
    )
    output_gather_indices = np.asarray(output_indices, dtype=np.int64)
    d_output = len(output_gather_indices)

    # Initializers
    _add_float_init("input_proj", input_proj, dense_inits, sparse_inits)
    _add_float_init("pos_proj", pos_proj, dense_inits, sparse_inits)
    _add_float_init("constant_values", constant_values, dense_inits, sparse_inits)
    _add_float_init("pos_encoding_full", pos_encoding_buf, dense_inits, sparse_inits)
    _add_int64_init("output_gather_indices_init", output_gather_indices, dense_inits)
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
    if cache_window is not None:
        _add_windowed_scalar_inits(dense_inits, cache_stride_resolved)

    # Nodes: preamble (mask + pos), residual stream, layers, postamble.
    nodes: list = []

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    _emit_cached_preamble(nodes, cache_window=cache_window)
    add("MatMul", ["inputs", "input_proj"], ["inp_res"])
    add("MatMul", ["pos", "pos_proj"], ["pos_res"])
    add("Add", ["inp_res", "pos_res"], ["res_pi"])
    add("Add", ["res_pi", "constant_values"], ["res_0"])

    scatter_idx_col = (
        "_cache_pos_col" if cache_window is None else "_write_slot_col"
    )
    current_res = "res_0"
    for i in range(n_layers):
        current_res = _emit_cached_layer_nodes(
            nodes,
            i,
            current_res,
            d,
            d_head,
            per_layer_n_heads[i],
            scatter_idx_col=scatter_idx_col,
        )

    add(
        "Gather",
        [current_res, "output_gather_indices_init"],
        ["outputs"],
        axis=1,
    )

    # Graph I/O value infos
    inputs_vi = helper.make_tensor_value_info(
        "inputs", TensorProto.FLOAT, ["n_new", d_input]
    )
    cache_position_vi = helper.make_tensor_value_info(
        "cache_position", TensorProto.INT64, ["n_new"]
    )
    past_vis, new_vis = _kv_io_value_info(per_layer_n_heads, d_head)
    outputs_vi = helper.make_tensor_value_info(
        "outputs", TensorProto.FLOAT, ["n_new", d_output]
    )

    graph = helper.make_graph(
        nodes,
        "headless_transformer_cached",
        inputs=[inputs_vi, cache_position_vi, *past_vis],
        outputs=[outputs_vi, *new_vis],
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

    meta_path = _write_headless_meta(
        output_path,
        list(input_names),
        extra=extra_metadata,
        cache_stride=cache_stride_resolved,
        cache_window=cache_window,
    )

    if debug_sidecar:
        # (name, offset, width) in the alphabetical order the flat
        # ``inputs`` tensor is packed in (input_proj rows follow
        # all_input_indices, which concatenates per-input indices in
        # sorted-name order above).
        debug_input_specs: List[tuple] = []
        offset = 0
        for name, idx in input_nodes_list:
            debug_input_specs.append((name, offset, len(idx)))
            offset += len(idx)
        _write_debug_sidecar(
            output_path,
            compiled=compiled,
            output_node=output_node,
            pos_encoding=pos_encoding,
            d=d,
            d_head=d_head,
            kind="headless",
            input_specs=debug_input_specs,
            asserts=all_asserts,
            watches=all_watches,
            cache_stride=cache_stride_resolved,
            cache_window=cache_window,
            verbose=verbose,
        )

    if verbose:
        print(
            f"Phases: compile+emit {t_compile:.2f}s, "
            f"build {t_build:.2f}s, save {t_save:.2f}s"
        )
        print(
            f"{n_layers} layers, "
            f"{len(sparse_inits)} sparse inits, {len(dense_inits)} dense inits"
        )
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {meta_path}")


def compile_to_onnx(
    output_node: Node,
    pos_encoding: PosEncoding,
    embedding: Embedding,
    output_path: str,
    d: int = 1024,
    d_head: int = 16,
    max_seq_len: int = 512,
    max_layers: int = 200,
    verbose: bool = True,
    trim_heads: bool = True,
    optimize: int = 0,
    assume_zero_init: bool = True,
    d_hidden: Optional[int] = None,
    cache_stride: Optional[int] = None,
    cache_window: Optional[int] = None,
    debug_sidecar: bool = True,
) -> None:
    """Compile a token-I/O graph to a KV-cached ONNX model.

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

    ``cache_window=C`` selects the windowed-cache protocol instead (see
    the module docstring): bindings are exactly ``C + n_new`` wide,
    committed slot placement is the host's policy, positions stay
    absolute and may run to ``max_seq_len`` regardless of ``C``.
    Mutually exclusive with ``cache_stride``; the sidecar carries both
    ``cache_stride == C`` and the ``cache_window`` discriminator key.

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
    cache_stride_resolved = _resolve_cache_stride(
        cache_stride, max_seq_len, cache_window
    )

    # Assert/DebugWatch coverage must be collected BEFORE forward_compile
    # strips both wrapper kinds from the graph in-place.
    from torchwright.graph.asserts import collect_debug_nodes

    all_asserts, all_watches = collect_debug_nodes(output_node)

    dense_inits: list = []
    sparse_inits: list = []
    per_layer_n_heads: list = []

    on_layer_compiled = _make_stream_layer_weights_cb(
        d,
        d_head,
        dense_inits,
        sparse_inits,
        per_layer_n_heads,
        trim_heads=trim_heads,
    )

    # --- Phase 1: streaming compile ---------------------------------------
    t0 = time.perf_counter()
    compiled = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
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
    pos_indices: Optional[List[int]] = None
    constant_values = np.zeros(d, dtype=np.float32)

    for node in compiled.residual_assignment.get_nodes(in_state):
        indices = compiled.residual_assignment.get_node_indices(in_state, node)
        if isinstance(node, Embedding):
            embedding_indices = indices
        elif isinstance(node, PosEncoding):
            pos_indices = indices
        elif isinstance(node, LiteralValue):
            for k, idx in enumerate(indices):
                constant_values[idx] = float(node.value[k])
        elif isinstance(node, Concatenate):
            pass

    assert embedding_indices is not None, "No Embedding node in residual assignment"
    assert pos_indices is not None, "No PosEncoding node in residual assignment"

    d_embed = len(embedding_indices)
    d_pos = len(pos_indices)

    embedding_proj = np.zeros((d_embed, d), dtype=np.float32)
    for k, idx in enumerate(embedding_indices):
        embedding_proj[k, idx] = 1.0

    pos_proj = np.zeros((d_pos, d), dtype=np.float32)
    for k, idx in enumerate(pos_indices):
        pos_proj[k, idx] = 1.0

    pos_encoding_buf = _compute_pos_encoding(d_pos, max_seq_len)

    embed_table_np = (
        embedding.table.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    vocab_size, d_embed_check = embed_table_np.shape
    assert d_embed_check == d_embed, (
        f"Embedding table last dim {d_embed_check} disagrees with "
        f"d_embed {d_embed} derived from feature assignment"
    )

    output_indices = compiled.residual_assignment.get_node_indices(
        out_state, _unwrap_output_node(output_node)
    )
    output_gather_indices = np.asarray(output_indices, dtype=np.int64)

    # Initializers
    _add_float_init("embedding_proj", embedding_proj, dense_inits, sparse_inits)
    _add_float_init("pos_proj", pos_proj, dense_inits, sparse_inits)
    _add_float_init("constant_values", constant_values, dense_inits, sparse_inits)
    _add_float_init("pos_encoding_full", pos_encoding_buf, dense_inits, sparse_inits)
    _add_float_init("embed_table", embed_table_np, dense_inits, sparse_inits)
    _add_int64_init("output_gather_indices_init", output_gather_indices, dense_inits)
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
    if cache_window is not None:
        _add_windowed_scalar_inits(dense_inits, cache_stride_resolved)

    # Nodes: preamble (mask + pos), token embed, residual stream, layers,
    # output gather, unembed.
    nodes: list = []

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    _emit_cached_preamble(nodes, cache_window=cache_window)
    # Token embedding lookup: (vocab, d_embed) gather rows by token_ids.
    add("Gather", ["embed_table", "token_ids"], ["_token_emb"], axis=0)
    add("MatMul", ["_token_emb", "embedding_proj"], ["inp_res"])
    add("MatMul", ["pos", "pos_proj"], ["pos_res"])
    add("Add", ["inp_res", "pos_res"], ["res_pi"])
    add("Add", ["res_pi", "constant_values"], ["res_0"])

    scatter_idx_col = (
        "_cache_pos_col" if cache_window is None else "_write_slot_col"
    )
    current_res = "res_0"
    for i in range(n_layers):
        current_res = _emit_cached_layer_nodes(
            nodes,
            i,
            current_res,
            d,
            d_head,
            per_layer_n_heads[i],
            scatter_idx_col=scatter_idx_col,
        )

    add(
        "Gather",
        [current_res, "output_gather_indices_init"],
        ["_output_emb"],
        axis=1,
    )
    # logits = output_emb @ embed_table.T
    add("Transpose", ["embed_table"], ["_embed_table_T"], perm=[1, 0])
    add("MatMul", ["_output_emb", "_embed_table_T"], ["logits"])

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
    }
    if cache_window is not None:
        # Windowed-cache protocol discriminator: bindings must be exactly
        # C + n_new wide (see the module docstring).
        token_meta["cache_window"] = int(cache_window)
    meta_path = _write_meta(output_path, token_meta)

    if debug_sidecar:
        _write_debug_sidecar(
            output_path,
            compiled=compiled,
            output_node=output_node,
            pos_encoding=pos_encoding,
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
            cache_window=cache_window,
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


# ---------------------------------------------------------------------------
# In-process headless callable: a thin adapter over HeadlessTransformer.compute()
# that presents the same (inputs) -> outputs interface as OnnxHeadlessModule.
#
# Used for compiler-output tests that don't need an ONNX round-trip.  For
# production inference, export with compile_headless_to_onnx instead —
# that path streams weights and runs under onnxruntime.
# ---------------------------------------------------------------------------


def _compute_io_layout(
    io: Dict[str, Tuple[Optional[Node], Optional[Node]]],
) -> Tuple[List[tuple], List[tuple], Dict[Node, Tuple[Optional[Node], List[int]]], int]:
    """Compute column assignments from io spec.

    The io dict declares the I/O contract:
    - Key: field name (string) — used for alphabetical ordering
    - Value: (input_node, output_node) tuple where:
      - (in, out) → overlaid: output lands at input's columns via delta transfer
      - (in, None) → input-only: columns hold input value, no output
      - (None, out) → output-only: overflow columns appended after input region

    Returns:
        input_specs: List[(name, offset, width, input_node)] for input fields
        output_specs: List[(name, offset, width, output_node)] for output fields
        overlays: Dict[output_node -> (input_node or None, target_cols)] for delta transfer
            - For overlaid: (input_node, target_cols) - subtract input before adding output
            - For overflow: (None, target_cols) - just copy to target (subtract zero)
        d_input: Total width of input region
    """
    # Sort by name (alphabetical)
    sorted_names = sorted(io.keys())

    # Input region: all entries with non-None input
    input_specs = []
    input_name_to_offset = {}
    offset = 0
    for name in sorted_names:
        in_node, out_node = io[name]
        if in_node is not None:
            width = len(in_node)
            input_specs.append((name, offset, width, in_node))
            input_name_to_offset[name] = offset
            offset += width
    d_input = offset

    # Output region: overlaid at input positions, overflow after
    output_specs = []
    overlays: Dict[Node, Tuple[Optional[Node], List[int]]] = {}
    overflow_offset = d_input

    for name in sorted_names:
        in_node, out_node = io[name]
        if out_node is not None:
            width = len(out_node)
            if in_node is not None:
                # Overlaid: output at input's columns via delta transfer
                in_offset = input_name_to_offset[name]
                output_specs.append((name, in_offset, width, out_node))
                target_cols = list(range(in_offset, in_offset + width))
                overlays[out_node] = (in_node, target_cols)
            else:
                # Overflow: output after input region, also via delta
                # transfer.  Like the overlay case, the delta layer
                # subtracts whatever is currently at ``target_cols`` (see
                # forward/compile.py where ``subtract_cols = target_cols``
                # unconditionally), so the overflow path does not rely on
                # the caller zero-initialising the overflow region.
                output_specs.append((name, overflow_offset, width, out_node))
                target_cols = list(range(overflow_offset, overflow_offset + width))
                overlays[out_node] = (None, target_cols)
                overflow_offset += width

    return input_specs, output_specs, overlays, d_input


def _validate_io_spec(io: Dict[str, Tuple[Optional[Node], Optional[Node]]]) -> None:
    """Validate the io spec.

    Raises ValueError if:
    - Any entry has both nodes as None (empty tuple)
    - Overlaid pairs have mismatched widths
    - Duplicate nodes across different entries (same node in two different names)

    Note: It's valid for in_node == out_node (identity case) within the same entry.
    """
    seen_input_nodes = set()
    seen_output_nodes = set()

    for name, (in_node, out_node) in io.items():
        # Check for empty tuple
        if in_node is None and out_node is None:
            raise ValueError(f"io entry '{name}' has both input and output as None")

        # Check for duplicate input nodes across entries
        if in_node is not None:
            if in_node in seen_input_nodes:
                raise ValueError(f"Input node {in_node} appears in multiple io entries")
            seen_input_nodes.add(in_node)

        # Check for duplicate output nodes across entries
        # Allow same node to appear as both input and output within the same entry
        if out_node is not None and out_node is not in_node:
            if out_node in seen_output_nodes:
                raise ValueError(
                    f"Output node {out_node} appears in multiple io entries"
                )
            seen_output_nodes.add(out_node)

        # Check width mismatch for overlaid pairs
        if in_node is not None and out_node is not None:
            if len(in_node) != len(out_node):
                raise ValueError(
                    f"io entry '{name}' has width mismatch: "
                    f"input width {len(in_node)} != output width {len(out_node)}"
                )


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
    """Callable wrapper around :class:`HeadlessTransformer`.

    Exposes the same three-method surface as
    :class:`torchwright.compiler.onnx_load.OnnxHeadlessModule`:

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
        output_specs: Optional[List[tuple]] = None,
        asserts: Optional[List] = None,
        watches: Optional[List] = None,
    ) -> None:
        self._net = net
        # input_specs: list of (name, start_col, width) in input-tensor column order.
        self._input_specs = list(input_specs)
        self._output_indices = output_indices
        # output_specs: list of (name, offset_in_out, width) in gathered-output
        # column order.  None for legacy callers that compile a single
        # concatenated output_node and do not declare field names.
        self._output_specs = list(output_specs) if output_specs is not None else []
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

    def output_slice(self, name: str, outputs: torch.Tensor) -> torch.Tensor:
        """Return the slice of ``outputs`` (post-gather) for the named output field."""
        for n, s, w in self._output_specs:
            if n == name:
                return outputs[..., s : s + w]
        raise KeyError(f"output field {name!r} not found")

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

        res, new_kvs, state_tensor = self._capture_states(
            res_stream, past_kvs=past_kvs
        )
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
    first_arg,
    second_arg=None,
    d: int = 1024,
    d_head: int = 16,
    max_layers: int = 100,
    verbose: bool = True,
    device: str = "cpu",
    extra_metadata: Optional[dict] = None,
    d_hidden: Optional[int] = None,
    trim_heads: bool = True,
    # Named parameters for new API
    io: Optional[Dict[str, Tuple[Optional[Node], Optional[Node]]]] = None,
    # Legacy named parameter
    output_node: Optional[Node] = None,
) -> CompiledHeadless:
    """Compile a headless graph to an in-process callable.

    Returns a :class:`CompiledHeadless` that evaluates the graph via
    :meth:`HeadlessTransformer.forward` behind the standard
    ``module(inputs) -> outputs`` interface.  For production use (saved
    artifact, autoregressive decode, fast startup), use
    :func:`compile_headless_to_onnx` instead.

    Supports two calling patterns:
    - New API: ``compile_headless(pos_encoding, io={"name": (input, output)}, ...)``
    - Legacy API: ``compile_headless(output_node, pos_encoding, ...)``

    The ``io`` dict declares the I/O contract:
    - Key: field name (string) — used for alphabetical ordering
    - Value: ``(input_node, output_node)`` tuple where:
      - ``(in, out)`` → overlaid: output lands at input's columns
      - ``(in, None)`` → input-only: columns hold input value, no output
      - ``(None, out)`` → output-only: overflow columns appended after input region

    For overlaid entries, the output is placed at the same columns as the
    input via delta transfer, enabling autoregressive feedback where the
    transformer output IS the next input.

    ``d_hidden`` is the per-layer MLP hidden width.  Defaults to ``d``
    when omitted; pass an explicit value to decouple the MLP intermediate
    width from the residual stream width.
    """
    # Detect API based on argument types
    # Legacy: compile_headless(output_node, pos_encoding, ...)
    # New: compile_headless(pos_encoding, io=..., ...)

    if isinstance(first_arg, PosEncoding):
        # New API: first_arg is pos_encoding
        pos_encoding = first_arg
        # second_arg should be None or io dict (not used positionally in new API)
        if second_arg is not None and io is None:
            # Allow compile_headless(pos_encoding, io_dict, ...) positionally
            io = second_arg
    else:
        # Legacy API: first_arg is output_node, second_arg is pos_encoding
        if output_node is not None:
            raise ValueError(
                "Cannot specify output_node both positionally and as keyword"
            )
        output_node = first_arg
        pos_encoding = second_arg

    # Handle legacy API
    if output_node is not None:
        if io is not None:
            raise ValueError("Cannot specify both io and output_node")
        return _compile_headless_legacy(
            output_node=output_node,
            pos_encoding=pos_encoding,
            d=d,
            d_head=d_head,
            max_layers=max_layers,
            verbose=verbose,
            device=device,
            extra_metadata=extra_metadata,
            d_hidden=d_hidden,
            trim_heads=trim_heads,
        )

    # New io-based path
    if io is None:
        raise ValueError("Either io or output_node must be provided")

    # New io-based path
    assert io is not None
    _validate_io_spec(io)

    # Set names on InputNodes from io dict keys
    # This enables HeadlessTransformer.get_input_res_stream to look up values
    for name, (in_node, out_node) in io.items():
        if in_node is not None and isinstance(in_node, InputNode):
            in_node.name = name

    input_specs, output_specs, overlays, d_input = _compute_io_layout(io)

    # Build the combined output node for forward_compile
    # Collect all output nodes (both overlaid and overflow)
    output_nodes = [spec[3] for spec in output_specs]
    if len(output_nodes) == 0:
        raise ValueError("io must have at least one output")
    elif len(output_nodes) == 1:
        combined_output = output_nodes[0]
    else:
        combined_output = Concatenate(output_nodes)

    # Collect Assert and DebugWatch nodes before forward_compile strips them.
    from torchwright.graph.asserts import collect_debug_nodes

    all_asserts = []
    all_watches = []
    _seen_ids: set = set()
    for _name, (in_node, out_node) in io.items():
        for root in (in_node, out_node):
            if root is None:
                continue
            node_asserts, node_watches = collect_debug_nodes(root)
            for a in node_asserts:
                if a.node_id not in _seen_ids:
                    _seen_ids.add(a.node_id)
                    all_asserts.append(a)
            for w in node_watches:
                if w.node_id not in _seen_ids:
                    _seen_ids.add(w.node_id)
                    all_watches.append(w)

    # Map input nodes to their names for the residual assignment
    input_node_to_name = {spec[3]: spec[0] for spec in input_specs}

    net = forward_compile(
        d=d,
        d_head=d_head,
        output_node=combined_output,
        pos_encoding=pos_encoding,
        verbose=verbose,
        max_layers=max_layers,
        device=device,
        d_hidden=d_hidden,
        trim_heads=trim_heads,
        overlays=overlays,
    )

    assert net.residual_assignment is not None
    out_state = net.layers[-1].mlp.out_state

    # Build input_specs for CompiledHeadless (name, offset, width)
    ch_input_specs = [(name, offset, width) for name, offset, width, _ in input_specs]

    # Build output indices from output_specs
    # For overlaid outputs, offset is the input's column offset
    # For overflow outputs, offset is in the overflow region
    output_indices: list[int] = []
    ch_output_specs: List[tuple] = []
    running = 0
    for name, offset, width, out_node in output_specs:
        output_indices.extend(range(offset, offset + width))
        ch_output_specs.append((name, running, width))
        running += width

    output_indices_tensor = torch.tensor(output_indices, dtype=torch.long)

    return CompiledHeadless(
        net,
        ch_input_specs,
        output_indices_tensor,
        metadata=extra_metadata,
        output_specs=ch_output_specs,
        asserts=all_asserts,
        watches=all_watches,
    )


def _compile_headless_legacy(
    output_node: Node,
    pos_encoding: PosEncoding,
    d: int,
    d_head: int,
    max_layers: int,
    verbose: bool,
    device: str,
    extra_metadata: Optional[dict],
    d_hidden: Optional[int],
    trim_heads: bool,
) -> CompiledHeadless:
    """Legacy compile_headless implementation using output_node parameter."""
    # Unwrap Assert nodes at the output root — compilation strips them
    # from the interior of the graph, but the caller's output_node
    # reference may still point at one.  Downstream lookups
    # (residual-stream indices, etc.) must match the compiled graph's
    # effective terminal node.
    from torchwright.graph.misc import Assert, DebugWatch
    from torchwright.graph.asserts import collect_debug_nodes

    # Collect Assert and DebugWatch nodes before forward_compile strips them.
    all_asserts, all_watches = collect_debug_nodes(output_node)

    while isinstance(output_node, (Assert, DebugWatch)):
        output_node = output_node.inputs[0]

    net = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=verbose,
        max_layers=max_layers,
        device=device,
        d_hidden=d_hidden,
        trim_heads=trim_heads,
    )

    assert net.residual_assignment is not None
    in_state = net.layers[0].attn.in_state
    out_state = net.layers[-1].mlp.out_state

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

    input_specs: List[tuple] = []
    offset = 0
    for name, width in input_nodes_list:
        input_specs.append((name, offset, width))
        offset += width

    # Direct residual-stream gather handles Concatenate output nodes
    # (which compute()'s per-node result dict does not populate).
    output_indices = torch.tensor(
        net.residual_assignment.get_node_indices(out_state, output_node),
        dtype=torch.long,
    )

    return CompiledHeadless(
        net,
        input_specs,
        output_indices,
        metadata=extra_metadata,
        asserts=all_asserts,
        watches=all_watches,
    )
