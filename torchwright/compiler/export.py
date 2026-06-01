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

Both speak the KV-cache prefill/decode protocol:

    graph inputs:  <seq-input>, past_len, past_K_i, past_V_i  (i in 0..n_layers-1)
    graph outputs: <seq-output>, new_K_i, new_V_i

Prefill uses empty past tensors (shape ``(n_heads, 0, d_head)``) and
``past_len = 0``.  Decode uses past tensors produced by a prior run.

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
from torchwright.graph.misc import InputNode

HEADLESS_META_FORMAT = "torchwright.headless.v1"
TOKEN_META_FORMAT = "torchwright.token.v1"


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
) -> str:
    """Write the headless sidecar JSON, optionally with an ``extra`` dict.

    ``extra`` is a free-form dict for per-export metadata (e.g. DOOM's
    ``rows_per_patch`` — surfaced to the host via
    :class:`OnnxHeadlessModule.metadata`). Kept totally general so the
    compiler layer has no project-specific keys.
    """
    meta: dict = {
        "format": HEADLESS_META_FORMAT,
        "input_names": list(input_names),
    }
    if extra:
        meta["extra"] = dict(extra)
    return _write_meta(onnx_path, meta)


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
    """Register 0-D scalar and tiny 1-D helper initializers used by the
    cached preamble (Range, Where, Unsqueeze/Slice axes).
    """
    dense_inits.append(
        helper.make_tensor(
            name="_i64_zero_s",
            data_type=TensorProto.INT64,
            dims=[],
            vals=np.array(0, dtype=np.int64).tobytes(),
            raw=True,
        )
    )
    dense_inits.append(
        helper.make_tensor(
            name="_i64_one_s",
            data_type=TensorProto.INT64,
            dims=[],
            vals=np.array(1, dtype=np.int64).tobytes(),
            raw=True,
        )
    )
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


def _make_stream_layer_weights_cb(
    d: int,
    d_head: int,
    dense_inits: list,
    sparse_inits: list,
    per_layer_n_heads: list,
) -> Callable[[int, object], None]:
    """Factory for the forward_compile on_layer_compiled callback.

    Emits each freshly-compiled layer's dense weights into the
    dense/sparse init lists (sparsified on the fly when mostly zero) and
    nulls out the layer's tensor attributes so GC can reclaim them.

    After head trimming, each layer's attention matrices have shape
    ``(nh, d, d_head)`` where ``nh <= d // d_head``.  Weight matrices
    are emitted at the trimmed size and per-layer reshape constants
    are added so the ONNX graph can reshape to the correct head count.

    Appends each layer's ``nh`` to ``per_layer_n_heads`` for Phase 2.
    """

    def on_layer_compiled(i: int, layer) -> None:
        attn = layer.attn.attn
        mlp = layer.mlp

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


def _emit_cached_preamble(nodes: list, seq_input_name: str) -> None:
    """Emit nodes that produce ``pos`` and ``mask_bool_3d`` tensors.

    Requires:
      - graph input ``past_len``: scalar int64 — the absolute *query*
        position the new rows live at.  Drives the positional-encoding
        slice.  Does **not** have to equal ``past_K_0.shape[1]``; the
        host may hand over a trimmed cache with a larger ``past_len`` to
        get correct pos encodings for a sliding-window runtime.
      - graph input ``seq_input_name``: first dim is ``n_new``
      - graph input ``past_K_0``: shape ``(n_heads, past_cache_len,
        d_head)`` — the actual past cache length is read from its shape
        at runtime and drives the mask geometry.
      - initializer ``pos_encoding_full``: (max_seq_len, d_pos)
      - scalar initializers from :func:`_add_scalar_inits`

    Produces:
      - ``pos``: (n_new, d_pos) float, sliced from pos_encoding_full at
        ``[past_len : past_len + n_new]``.
      - ``mask_bool_3d``: (1, n_new, past_cache_len + n_new) bool, True
        where a query position must NOT attend to a key position.
        Applied via ``Where(mask_bool_3d, -10000, logits)`` — this
        overwrites masked logits (unlike an additive -10000 mask, which
        is numerically unsafe when the original logits can dominate the
        additive shift).

    The mask is derived from the *actual cache length* (the shape-[1] of
    ``past_K_0``) rather than the ``past_len`` input, so that callers
    can pass a trimmed cache with a different ``past_len`` without
    creating a shape mismatch against the attention logits.  When
    ``past_len == past_cache_len`` (the normal non-trimmed case) this
    emission is bit-identical in behaviour to the simpler
    ``past_len + n_new`` construction it replaces.
    """

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # n_new as 0-D scalar: Shape(seq_input)[0]
    add("Shape", [seq_input_name], ["_seq_shape"])
    add("Gather", ["_seq_shape", "_i64_zero_s"], ["_n_new_s"], axis=0)

    # past_cache_len as 0-D scalar: Shape(past_K_0)[1]
    # Every layer's past_K has the same shape-[1]; pick layer 0.
    add("Shape", ["past_K_0"], ["_past_K_shape"])
    add("Gather", ["_past_K_shape", "_i64_one_s"], ["_past_cache_len_s"], axis=0)

    # Mask geometry uses the actual cache length, not past_len.
    # n_total_mask = past_cache_len + n_new
    add("Add", ["_past_cache_len_s", "_n_new_s"], ["_n_total_mask_s"])

    # Dynamic causal mask — True where a new row must NOT attend to a column.
    # Row r is local to the cache+new window: abs-within-window = past_cache_len + r.
    # Column c is the same local index. mask[r, c] = c > past_cache_len + r.
    add("Range", ["_i64_zero_s", "_n_new_s", "_i64_one_s"], ["_rows"])
    add("Range", ["_i64_zero_s", "_n_total_mask_s", "_i64_one_s"], ["_cols"])
    add("Add", ["_rows", "_past_cache_len_s"], ["_abs_rows"])  # (n_new,)
    add("Unsqueeze", ["_abs_rows", "_axes1_1d"], ["_abs_rows_col"])  # (n_new, 1)
    add("Unsqueeze", ["_cols", "_axes0_1d"], ["_cols_row"])  # (1, n_total_mask)
    add(
        "Greater", ["_cols_row", "_abs_rows_col"], ["_mask_bool"]
    )  # (n_new, n_total_mask)
    add(
        "Unsqueeze", ["_mask_bool", "_axes0_1d"], ["mask_bool_3d"]
    )  # (1, n_new, n_total_mask)

    # Positional encoding slice uses past_len directly — this is the one
    # place past_len still matters, so the host can set the "absolute
    # query position" independently of the cache length.
    add("Add", ["past_len", "_n_new_s"], ["_pos_slice_end_s"])
    add("Unsqueeze", ["past_len", "_axes0_1d"], ["_past_len_1d"])
    add("Unsqueeze", ["_pos_slice_end_s", "_axes0_1d"], ["_pos_slice_end_1d"])
    add(
        "Slice",
        ["pos_encoding_full", "_past_len_1d", "_pos_slice_end_1d", "_axes0_1d"],
        ["pos"],
    )


def _emit_cached_layer_nodes(
    nodes: list,
    layer_idx: int,
    current_res: str,
    d: int,
    d_head: int,
    n_heads: int,
) -> str:
    """Emit cached attention + FFN nodes for one layer.

    Reads graph inputs ``past_K_{i}`` / ``past_V_{i}`` and writes graph
    outputs ``new_K_{i}`` / ``new_V_{i}``.  Uses the shared ``mask_3d``
    produced by :func:`_emit_cached_preamble`.

    ``n_heads`` is the (possibly trimmed) head count for this layer.
    Per-layer reshape constants ``l{i}_qkv_view_shape`` and
    ``l{i}_ctx_flat_shape`` are expected to have been emitted by the
    streaming weight callback.

    Returns the name of the next residual stream tensor.
    """
    p = f"l{layer_idx}"

    def node(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # Project Q, K_new, V_new from the new rows only, reshape to heads.
    node("MatMul", [current_res, f"{p}_WQ"], [f"{p}_Q_flat"])
    node("Reshape", [f"{p}_Q_flat", f"{p}_qkv_view_shape"], [f"{p}_Q_view"])
    node("Transpose", [f"{p}_Q_view"], [f"{p}_Q"], perm=[1, 0, 2])

    node("MatMul", [current_res, f"{p}_WK"], [f"{p}_K_flat"])
    node("Reshape", [f"{p}_K_flat", f"{p}_qkv_view_shape"], [f"{p}_K_view"])
    node("Transpose", [f"{p}_K_view"], [f"{p}_K_new"], perm=[1, 0, 2])

    node("MatMul", [current_res, f"{p}_WV"], [f"{p}_V_flat"])
    node("Reshape", [f"{p}_V_flat", f"{p}_qkv_view_shape"], [f"{p}_V_view"])
    node("Transpose", [f"{p}_V_view"], [f"{p}_V_new"], perm=[1, 0, 2])

    # Concatenate the cached past with the new rows along the seq axis.
    # These Concat outputs are also exposed as graph outputs (new_K_i /
    # new_V_i) so callers can feed them back as the next step's past.
    node(
        "Concat",
        [f"past_K_{layer_idx}", f"{p}_K_new"],
        [f"new_K_{layer_idx}"],
        axis=1,
    )
    node(
        "Concat",
        [f"past_V_{layer_idx}", f"{p}_V_new"],
        [f"new_V_{layer_idx}"],
        axis=1,
    )

    # Attention over the full (past + new) K and V.
    node("Transpose", [f"new_K_{layer_idx}"], [f"{p}_K_T"], perm=[0, 2, 1])
    node("MatMul", [f"{p}_Q", f"{p}_K_T"], [f"{p}_logits"])
    # Overwrite-mask with CAUSAL_MASK_SENTINEL (equivalent to torch's
    # masked_fill).  An additive penalty would leave masked positions at
    # "logit + penalty", which is not dominated by real logits when they
    # are very negative (e.g. attend_argmin_unmasked with high query gain).
    node(
        "Where",
        ["mask_bool_3d", "_f32_causal_sentinel_s", f"{p}_logits"],
        [f"{p}_logits_masked"],
    )
    node("Softmax", [f"{p}_logits_masked"], [f"{p}_weights"], axis=-1)
    node("MatMul", [f"{p}_weights", f"new_V_{layer_idx}"], [f"{p}_ctx"])

    # Fused output projection: (n_heads, t, d_head) → (t, hd) → (t, d)
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

    Returns:
        (past_vis, new_vis) — each a list of length 2*n_layers
        alternating ``past_K_i`` / ``past_V_i`` (inputs) and
        ``new_K_i`` / ``new_V_i`` (outputs).
    """
    past_vis: list = []
    new_vis: list = []
    for i, nh in enumerate(per_layer_n_heads):
        past_vis.append(
            helper.make_tensor_value_info(
                f"past_K_{i}", TensorProto.FLOAT, [nh, "n_past", d_head]
            )
        )
        past_vis.append(
            helper.make_tensor_value_info(
                f"past_V_{i}", TensorProto.FLOAT, [nh, "n_past", d_head]
            )
        )
        new_vis.append(
            helper.make_tensor_value_info(
                f"new_K_{i}", TensorProto.FLOAT, [nh, "n_total", d_head]
            )
        )
        new_vis.append(
            helper.make_tensor_value_info(
                f"new_V_{i}", TensorProto.FLOAT, [nh, "n_total", d_head]
            )
        )
    return past_vis, new_vis


# ---------------------------------------------------------------------------
# Public exporters
# ---------------------------------------------------------------------------


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
) -> None:
    """Compile a float-I/O graph to a KV-cached ONNX model.

    Writes two files:
        ``<output_path>``    — the ONNX model
        ``<stem>.meta.json`` — ``{"format": "torchwright.headless.v1",
                                  "input_names": [...]}``

    The graph speaks the KV-cache prefill/decode protocol:
        inputs:  inputs (n_new, d_input), past_len (scalar),
                 past_K_i, past_V_i (n_heads, n_past, d_head)
        outputs: outputs (n_new, d_output),
                 new_K_i, new_V_i (n_heads, n_total, d_head)

    Prefill = empty past tensors + past_len=0.  Decode = feed back the
    new_K_i / new_V_i from a previous run and set past_len accordingly.

    ``d_hidden`` is the per-layer MLP hidden width.  Defaults to ``d``
    when omitted; pass an explicit value to decouple the MLP intermediate
    width from the residual stream width.
    """
    dense_inits: list = []
    sparse_inits: list = []
    per_layer_n_heads: list = []

    on_layer_compiled = _make_stream_layer_weights_cb(
        d,
        d_head,
        dense_inits,
        sparse_inits,
        per_layer_n_heads,
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
    )
    t_compile = time.perf_counter() - t0

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
        out_state, output_node
    )
    output_gather_indices = np.asarray(output_indices, dtype=np.int64)
    d_output = len(output_gather_indices)

    # Initializers
    _add_float_init("input_proj", input_proj, dense_inits, sparse_inits)
    _add_float_init("pos_proj", pos_proj, dense_inits, sparse_inits)
    _add_float_init("constant_values", constant_values, dense_inits, sparse_inits)
    _add_float_init("pos_encoding_full", pos_encoding_buf, dense_inits, sparse_inits)
    _add_int64_init("output_gather_indices_init", output_gather_indices, dense_inits)
    # Per-layer reshape constants (l{i}_qkv_view_shape, l{i}_ctx_flat_shape)
    # are emitted by the streaming weight callback.
    _add_scalar_inits(dense_inits)

    # Nodes: preamble (mask + pos), residual stream, layers, postamble.
    nodes: list = []

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    _emit_cached_preamble(nodes, seq_input_name="inputs")
    add("MatMul", ["inputs", "input_proj"], ["inp_res"])
    add("MatMul", ["pos", "pos_proj"], ["pos_res"])
    add("Add", ["inp_res", "pos_res"], ["res_pi"])
    add("Add", ["res_pi", "constant_values"], ["res_0"])

    current_res = "res_0"
    for i in range(n_layers):
        current_res = _emit_cached_layer_nodes(
            nodes, i, current_res, d, d_head, per_layer_n_heads[i]
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
    past_len_vi = helper.make_tensor_value_info("past_len", TensorProto.INT64, [])
    past_vis, new_vis = _kv_io_value_info(per_layer_n_heads, d_head)
    outputs_vi = helper.make_tensor_value_info(
        "outputs", TensorProto.FLOAT, ["n_new", d_output]
    )

    graph = helper.make_graph(
        nodes,
        "headless_transformer_cached",
        inputs=[inputs_vi, past_len_vi, *past_vis],
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
) -> None:
    """Compile a token-I/O graph to a KV-cached ONNX model.

    Writes two files:
        ``<output_path>``    — the ONNX model
        ``<stem>.meta.json`` — ``{"format": "torchwright.token.v1",
                                  "vocab": [...]}``

    The graph speaks the KV-cache prefill/decode protocol:
        inputs:  token_ids (n_new,) int64, past_len (scalar),
                 past_K_i, past_V_i (n_heads, n_past, d_head)
        outputs: logits (n_new, vocab_size),
                 new_K_i, new_V_i (n_heads, n_total, d_head)

    Prefill = empty past + past_len=0; decode = feed back the new_K_i /
    new_V_i from a prior run with the accumulated past_len.
    """
    dense_inits: list = []
    sparse_inits: list = []
    per_layer_n_heads: list = []

    on_layer_compiled = _make_stream_layer_weights_cb(
        d,
        d_head,
        dense_inits,
        sparse_inits,
        per_layer_n_heads,
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
    )
    t_compile = time.perf_counter() - t0

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
        out_state, output_node
    )
    output_gather_indices = np.asarray(output_indices, dtype=np.int64)

    # Initializers
    _add_float_init("embedding_proj", embedding_proj, dense_inits, sparse_inits)
    _add_float_init("pos_proj", pos_proj, dense_inits, sparse_inits)
    _add_float_init("constant_values", constant_values, dense_inits, sparse_inits)
    _add_float_init("pos_encoding_full", pos_encoding_buf, dense_inits, sparse_inits)
    _add_float_init("embed_table", embed_table_np, dense_inits, sparse_inits)
    _add_int64_init("output_gather_indices_init", output_gather_indices, dense_inits)
    # Per-layer reshape constants (l{i}_qkv_view_shape, l{i}_ctx_flat_shape)
    # are emitted by the streaming weight callback.
    _add_scalar_inits(dense_inits)

    # Nodes: preamble (mask + pos), token embed, residual stream, layers,
    # output gather, unembed.
    nodes: list = []

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    _emit_cached_preamble(nodes, seq_input_name="token_ids")
    # Token embedding lookup: (vocab, d_embed) gather rows by token_ids.
    add("Gather", ["embed_table", "token_ids"], ["_token_emb"], axis=0)
    add("MatMul", ["_token_emb", "embedding_proj"], ["inp_res"])
    add("MatMul", ["pos", "pos_proj"], ["pos_res"])
    add("Add", ["inp_res", "pos_res"], ["res_pi"])
    add("Add", ["res_pi", "constant_values"], ["res_0"])

    current_res = "res_0"
    for i in range(n_layers):
        current_res = _emit_cached_layer_nodes(
            nodes, i, current_res, d, d_head, per_layer_n_heads[i]
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
    past_len_vi = helper.make_tensor_value_info("past_len", TensorProto.INT64, [])
    past_vis, new_vis = _kv_io_value_info(per_layer_n_heads, d_head)
    logits_vi = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, ["n_new", vocab_size]
    )

    graph = helper.make_graph(
        nodes,
        "token_transformer_cached",
        inputs=[token_ids_vi, past_len_vi, *past_vis],
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

    meta_path = _write_meta(
        output_path,
        {"format": TOKEN_META_FORMAT, "vocab": list(embedding.tokenizer.vocab)},
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
    ) -> tuple:
        """Zero-length past tensors suitable for a first prefill call."""
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
        from torchwright.debug.probe import (
            _extract_compiled_value,
            _first_state_with,
        )
        from torchwright.graph.misc import Assert, DebugWatch

        while isinstance(node, (Assert, DebugWatch)):
            node = node.inputs[0]

        ds = self._debug_state
        state = _first_state_with(node, ds.ra, ds.ordered_states)
        if state is None:
            return None
        tensor_pair = ds.state_tensor.get(state)
        if tensor_pair is None:
            return None
        res_tensor, _ = tensor_pair
        return _extract_compiled_value(node, ds.ra, state, res_tensor)

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

        Returns (res, new_kvs) where new_kvs is None for prefill or a
        list of (K, V) tuples for decode.
        """
        from torchwright.debug.probe import (
            _extract_compiled_value,
            _first_state_with,
            _ordered_mlp_states,
        )
        from torchwright.graph.misc import Assert, DebugWatch

        net = self._net
        ra = net.residual_assignment
        assert ra is not None

        state_tensor = {}
        new_kvs = None

        if past_kvs is None:
            res, all_states = net.forward(res_stream, return_states=True)
            for _key, (state, tensor) in all_states.items():
                state_tensor[state] = (tensor, _key)
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

        ordered_states = [s for _, _, s in _ordered_mlp_states(net, ra)]

        self._debug_state = _DebugState(
            state_tensor=state_tensor,
            ordered_states=ordered_states,
            ra=ra,
        )

        # Self-consistency: for each node present in multiple captured
        # states, verify the value at its columns is the same everywhere.
        # A mismatch means a column was overwritten while the node was
        # still live — a scheduling or allocator bug.
        #
        # We collect ALL violating nodes (one entry per node, the first
        # state where its value diverged from the initial reading) and
        # raise a single error listing every one.  Aborting on the first
        # failure used to hide secondary aliasings behind the first, which
        # made it easy to misdiagnose the compiler bug as a single isolated
        # issue when in practice a whole class of nodes was affected.
        captured_states = [s for s in ordered_states if s in state_tensor]
        # Each entry: (value_tensor, label_from_state_tensor, cols_list).
        # The label comes from ``state_tensor[state][1]`` which embeds the
        # layer index ("layer_5_mlp_out_state") — ``state.name`` alone is
        # ambiguous because every layer's MLP out state shares the same
        # name.
        node_first_value: Dict[Node, Tuple[torch.Tensor, str, list]] = {}
        violations: List[tuple] = []
        violated_nodes: set = set()
        for state in captured_states:
            state_label = state_tensor[state][1]
            for node in ra.get_nodes(state):
                if node in violated_nodes:
                    continue
                res_tensor, _ = state_tensor[state]
                val = _extract_compiled_value(node, ra, state, res_tensor)
                if val is None:
                    continue
                cols = list(ra.get_node_indices(state, node))
                if node in node_first_value:
                    first_val, first_label, first_cols = node_first_value[node]
                    if first_val.shape == val.shape:
                        diff = (first_val - val).abs()
                        max_diff = diff.max().item()
                        if max_diff > atol:
                            violations.append(
                                (
                                    node,
                                    first_label,
                                    state_label,
                                    cols,
                                    first_val.detach().clone(),
                                    val.detach().clone(),
                                    max_diff,
                                )
                            )
                            violated_nodes.add(node)
                else:
                    node_first_value[node] = (
                        val.detach().clone(),
                        state_label,
                        cols,
                    )

        if violations:
            # Sort by max_abs_diff descending so the biggest offenders are
            # at the top.
            violations.sort(key=lambda v: v[6], reverse=True)
            msg_lines = [
                f"Residual-stream self-consistency failure (atol={atol:g}) — "
                f"{len(violations)} node(s):"
            ]
            for i, (
                node,
                first_label,
                later_label,
                cols,
                first_val,
                val,
                max_diff,
            ) in enumerate(violations, 1):
                diff = (first_val - val).abs()
                if diff.ndim >= 2:
                    per_col_max = diff.max(dim=0).values
                else:
                    per_col_max = diff
                worst_col_idx = int(per_col_max.argmax().item())
                worst_col = cols[worst_col_idx] if worst_col_idx < len(cols) else None
                node_name = node.annotation or node.name or f"node_{node.node_id}"
                # Summarise value tensors at the worst col if they span many
                # positions: show min/max across positions rather than dumping
                # the whole list so the error stays readable when the prefill
                # is long.
                a_slice = first_val[..., worst_col_idx]
                b_slice = val[..., worst_col_idx]
                if a_slice.ndim == 0 or a_slice.numel() <= 8:
                    a_fmt = a_slice.tolist()
                    b_fmt = b_slice.tolist()
                else:
                    a_fmt = (
                        f"n={a_slice.numel()} min={float(a_slice.min()):g} "
                        f"max={float(a_slice.max()):g}"
                    )
                    b_fmt = (
                        f"n={b_slice.numel()} min={float(b_slice.min()):g} "
                        f"max={float(b_slice.max()):g}"
                    )
                msg_lines.append(
                    f"\n  [{i}] {node_name} (id={node.node_id}, "
                    f"type={type(node).__name__}, width={len(cols)})\n"
                    f"      first:     {first_label}\n"
                    f"      later:     {later_label}\n"
                    f"      cols:      {cols if len(cols) <= 16 else str(cols[:16]) + '...'}\n"
                    f"      worst col: residual[{worst_col}] (node index {worst_col_idx})\n"
                    f"      first val: {a_fmt}\n"
                    f"      later val: {b_fmt}\n"
                    f"      max_abs_diff: {max_diff:.6g}"
                )
            raise RuntimeError("".join(msg_lines))

        for assert_node in self._asserts:
            target = assert_node.inputs[0]
            while isinstance(target, (Assert, DebugWatch)):
                target = target.inputs[0]
            state = _first_state_with(target, ra, ordered_states)
            if state is None:
                continue
            tensor_pair = state_tensor.get(state)
            if tensor_pair is None:
                continue
            res_tensor, _ = tensor_pair
            compiled_val = _extract_compiled_value(target, ra, state, res_tensor)
            if compiled_val is None:
                continue
            assert_node._check(compiled_val)

        for watch_node in self._watches:
            target = watch_node.inputs[0]
            while isinstance(target, (Assert, DebugWatch)):
                target = target.inputs[0]
            state = _first_state_with(target, ra, ordered_states)
            if state is None:
                continue
            tensor_pair = state_tensor.get(state)
            if tensor_pair is None:
                continue
            res_tensor, _ = tensor_pair
            compiled_val = _extract_compiled_value(target, ra, state, res_tensor)
            if compiled_val is None:
                continue
            watch_node._check(compiled_val)

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
