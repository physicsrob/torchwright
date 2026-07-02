"""Writes weight matrices into TransformerLayer components.

Each operation directly sets matrix entries using column indices from the
ResidualStreamMap. No strategies, no ResidualAssignment — just scatter writes.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set

import torch
import torch.nn.functional as F

from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.graph import Node, Linear, Attn, Add, Concatenate
from torchwright.graph.misc import LiteralValue
from torchwright.graph.ffn import FFN
from torchwright.graph.rope import ROPE_BASE

# Self-match attention hardness: scales the constant-1 self-match logit so the
# diagonal softmax weight is 1.0 to fp32 (Δ=0 transport is bit-identical).
_SELF_MATCH_HARDNESS = 100.0


@dataclass
class PlacementEntry:
    """One op's contribution to one weight matrix, in flat matrix coords.

    ``matrix_kind`` is the per-layer suffix (e.g. ``"attn.W_Q"``,
    ``"mlp.W_in"``); the full ``matrix_id`` is built at sidecar-emit time
    from ``layer`` + this suffix.  ``rows`` / ``cols`` are index lists into
    the matrix's two axes (the head/d_head dims are pre-flattened, so a
    head ``h`` occupies columns ``range(h*d_head, (h+1)*d_head)``).

    ``mode`` distinguishes how the write filled the region:
      * ``"dense"`` — full cross product ``rows × cols`` (a real weight
        block, written with broadcast/outer indexing).
      * ``"diag"``  — paired write: cell ``k`` is ``(rows[k], cols[k])``
        only (an identity / passthrough, written with aligned 1-D index
        tensors).  ``rows`` and ``cols`` have equal length and the pairing
        order is meaningful — do not reorder.
    """

    layer: int
    matrix_kind: str
    node: Optional[Node]
    op_type: str
    rows: List[int]
    cols: List[int]
    mode: Literal["dense", "diag"]


class PlacementRecorder:
    """Accumulates per-op 2-D weight-matrix occupancy during weight writing.

    Holds only Python ints (never tensors), so it does not pin trimmed
    layer weights and its footprint is O(emitted ops) — the same order as
    the residual assignment.  ``set_layer`` is called once per sublayer
    write; subsequent ``add`` calls attribute to that layer.
    """

    def __init__(self):
        self.entries: List[PlacementEntry] = []
        self._layer = -1

    def set_layer(self, layer: int) -> None:
        self._layer = layer

    def add(
        self,
        matrix_kind: str,
        node: Optional[Node],
        op_type: str,
        rows: List[int],
        cols: List[int],
        mode: Literal["dense", "diag"],
    ) -> None:
        self.entries.append(
            PlacementEntry(
                self._layer, matrix_kind, node, op_type, list(rows), list(cols), mode
            )
        )


@dataclass
class AttnHeadOp:
    op_type: Literal[
        "compute_attn",
        "compute_linear",
        "compute_add",
        "cancel",
        "add_into",
    ]
    node: Optional[Node]
    target_cols: List[int]
    # Primary source columns captured at schedule time.  Meaning depends on
    # op_type:
    #   compute_attn: V input columns
    #   compute_linear: input columns
    #   compute_add: first addend's columns
    #   add_into: live addend's columns
    source_cols: Optional[List[int]] = None
    # compute_add: second addend's columns.
    source_cols_b: Optional[List[int]] = None
    # compute_attn: Q and K input columns (V is in source_cols).
    q_source_cols: Optional[List[int]] = None
    k_source_cols: Optional[List[int]] = None


@dataclass
class MLPOp:
    op_type: Literal[
        "compute_ffn",
        "compute_literal_value",
        "compute_bias",
        "compute_linear_bypass",
    ]
    node: Node
    target_cols: List[int]
    mlp_slots: List[int] = field(default_factory=list)
    # Input columns captured at schedule time.  Used by compute_ffn (the
    # FFN's input) and compute_linear_bypass (the Linear's input).
    source_cols: Optional[List[int]] = None


def write_attn_sublayer(
    layer: TransformerLayer,
    ops: List[AttnHeadOp],
    residual_map: ResidualStreamMap,
    pos_encoding=None,
    recorder: Optional[PlacementRecorder] = None,
    const_one: Optional[Node] = None,
):
    """Write attention head operations into a layer's AttnLayerComponent.

    The four arithmetic-transport ops (``compute_linear``/``compute_add``/
    ``cancel``/``add_into``) build a **rotary** Δ=0 self-match head that reads
    the reserved constant-1 :class:`LiteralValue` column (``const_one``) and
    lets the runtime rotation place the softmax on the diagonal — the RoPE
    end-state self-match (``docs/rope_port_plan.md`` §3, Phase 5).  ``pos_encoding``
    is accepted for signature stability but is always ``None`` (position is no
    longer a residual substrate).
    """
    attn = layer.attn.attn
    assert const_one is not None, "const_one required for the rotary self-match"
    for op in ops:
        if op.op_type == "compute_attn":
            _write_compute_attn(attn, op, residual_map, recorder)
        elif op.op_type == "compute_linear":
            _write_compute_linear(attn, op, residual_map, recorder, const_one)
        elif op.op_type == "cancel":
            _write_cancel(attn, op, residual_map, recorder, const_one)
        elif op.op_type == "compute_add":
            _write_compute_add(attn, op, residual_map, recorder, const_one)
        elif op.op_type == "add_into":
            _write_add_into(attn, op, residual_map, recorder, const_one)
        else:
            raise ValueError(f"Unknown attn op_type: {op.op_type}")


def write_mlp_sublayer(
    layer: TransformerLayer,
    ops: List[MLPOp],
    residual_map: ResidualStreamMap,
    biased_linears: Optional[Set[Node]] = None,
    recorder: Optional[PlacementRecorder] = None,
):
    """Write MLP operations into a layer's MLPSubLayer components."""
    if biased_linears is None:
        biased_linears = set()
    for op in ops:
        if op.op_type == "compute_ffn":
            _write_compute_ffn(layer.mlp, op, residual_map, biased_linears, recorder)
        elif op.op_type == "compute_literal_value":
            _write_compute_literal_value(layer.mlp, op)
        elif op.op_type == "compute_bias":
            _write_compute_bias(layer.mlp, op)
        elif op.op_type == "compute_linear_bypass":
            _write_compute_linear_bypass(
                layer.mlp, op, residual_map, biased_linears, recorder
            )
        else:
            raise ValueError(f"Unknown mlp op_type: {op.op_type}")


# ---------------------------------------------------------------------------
# Attention operations
# ---------------------------------------------------------------------------


def _scatter_attn_head(
    attn,
    head,
    q_idx,
    k_idx,
    v_idx,
    o_idx,
    q_mat,
    k_mat,
    v_mat,
    o_mat,
    d_head,
    *,
    recorder: Optional[PlacementRecorder] = None,
    node: Optional[Node] = None,
    op_type: Optional[str] = None,
):
    """Scatter strategy matrices into one attention head's weight tensors."""
    if recorder is not None:
        # Every attention write fills the head's full d_head columns for the
        # captured rows (the head is the allocation unit) — a dense block.
        # Q/K/V: rows = residual source cols, cols = this head's flat slice.
        # O:     rows = this head's flat slice, cols = residual output cols.
        head_cols = list(range(head * d_head, (head + 1) * d_head))
        recorder.add("attn.W_Q", node, op_type, q_idx, head_cols, "dense")
        recorder.add("attn.W_K", node, op_type, k_idx, head_cols, "dense")
        recorder.add("attn.W_V", node, op_type, v_idx, head_cols, "dense")
        recorder.add("attn.W_O", node, op_type, head_cols, o_idx, "dense")
    q_idx_t = torch.as_tensor(q_idx, dtype=torch.long)
    k_idx_t = torch.as_tensor(k_idx, dtype=torch.long)
    v_idx_t = torch.as_tensor(v_idx, dtype=torch.long)
    o_idx_t = torch.as_tensor(o_idx, dtype=torch.long)
    # Source matrices may be Long (e.g. integer Linear weights); the
    # destination is always Float.  Original scalar-loop assignment
    # auto-cast; vectorized assignment does not.
    target_dtype = attn.query_matrix.dtype
    attn.query_matrix[head, q_idx_t, :d_head] = q_mat[: len(q_idx), :d_head].to(
        target_dtype
    )
    attn.key_matrix[head, k_idx_t, :d_head] = k_mat[: len(k_idx), :d_head].to(
        target_dtype
    )
    attn.value_matrix[head, v_idx_t, :d_head] = v_mat[: len(v_idx), :d_head].to(
        target_dtype
    )
    attn.output_matrix[head, :d_head, o_idx_t] = o_mat[:d_head, : len(o_idx)].to(
        target_dtype
    )


def _allocate_head(attn):
    """Allocate the next available attention head."""
    assert attn.used_heads < attn.n_heads, "Ran out of attention heads"
    head = attn.used_heads
    attn.used_heads += 1
    return head


def _write_compute_attn(
    attn,
    op: AttnHeadOp,
    rmap: ResidualStreamMap,
    recorder: Optional[PlacementRecorder] = None,
):
    """Copy an Attn node's Q/K/V/O matrices into attention heads.

    When the node's d_v exceeds the layer's d_head, V/O are split across
    multiple heads that share duplicated Q/K matrices.  This is correct
    because V/O is applied after softmax (purely linear):
    sum_i(weights @ V_i @ O_i) == weights @ V @ O.
    """
    node = op.node
    assert isinstance(node, Attn)

    assert op.q_source_cols is not None, "compute_attn requires q_source_cols"
    assert op.k_source_cols is not None, "compute_attn requires k_source_cols"
    assert op.source_cols is not None, "compute_attn requires source_cols (V)"
    q_idx = op.q_source_cols
    k_idx = op.k_source_cols
    v_idx = op.source_cols
    o_idx = op.target_cols

    q_in, k_in, v_in = node.inputs
    assert len(q_idx) == len(q_in), (
        f"compute_attn Q row-width mismatch for {node!r}: "
        f"q_source_cols len={len(q_idx)} but Q input {q_in!r} has "
        f"width {len(q_in)}. "
        f"Likely a Concatenate-resolution bug."
    )
    assert len(k_idx) == len(k_in), (
        f"compute_attn K row-width mismatch for {node!r}: "
        f"k_source_cols len={len(k_idx)} but K input {k_in!r} has "
        f"width {len(k_in)}."
    )
    assert len(v_idx) == len(v_in), (
        f"compute_attn V row-width mismatch for {node!r}: "
        f"source_cols len={len(v_idx)} but V input {v_in!r} has "
        f"width {len(v_in)}."
    )
    assert len(o_idx) == node.d_output, (
        f"compute_attn O col-width mismatch for {node!r}: "
        f"target_cols len={len(o_idx)} but node.d_output={node.d_output}."
    )

    layer_d_head = attn.d_head

    # The head must FILL the grid: its Q/K projection occupies the whole d_head
    # (the rotary front + the unrotated NoPE tail both live in real columns).
    # d_rot (how many of those planes rotate) is orthogonal and threaded
    # separately below; this assert only guards the Q/K *width*.  Negative test
    # (both directions) lives in tests/compile/forward/test_compiler_assertions.py.
    assert node.d_qk == layer_d_head, (
        f"Attn '{node.name}' requires d_qk == d_head; got "
        f"d_qk={node.d_qk}, d_head={layer_d_head}.  Build the head full-width "
        f"(d_qk={layer_d_head}); a narrower Q/K width is not supported "
        f"(partial rotary is set via rope_d_rot, not by shrinking d_qk)."
    )

    # Q/K fill the rotary grid exactly (d_qk == d_head, asserted above), so they
    # need no padding and are shared across all V/O chunk heads.
    q_mat = node.query_matrix
    k_mat = node.key_matrix

    # Split V/O across ceil(d_v / layer_d_head) heads
    n_vo_heads = (node.d_v + layer_d_head - 1) // layer_d_head
    for chunk_idx in range(n_vo_heads):
        v_start = chunk_idx * layer_d_head
        v_end = min(v_start + layer_d_head, node.d_v)
        chunk_size = v_end - v_start

        v_chunk = F.pad(
            node.value_matrix[:, v_start:v_end], (0, layer_d_head - chunk_size)
        )
        o_chunk = F.pad(
            node.output_matrix[v_start:v_end, :], (0, 0, 0, layer_d_head - chunk_size)
        )

        head = _allocate_head(attn)
        # Record the rotary base and partial-rotary width (the component rotates
        # every head's front d_rot planes on the one global grid).  Like
        # rope_base, this is one value per layer-component.  d_rot is one global
        # value per model — forward_compile asserts every graph Attn node shares
        # it before compiling, so all heads packed into this layer agree and the
        # last write here is always the same value.
        attn.rope_base = node.rope_base
        attn.rope_d_rot = node.rope_d_rot
        _scatter_attn_head(
            attn,
            head,
            q_idx,
            k_idx,
            v_idx,
            o_idx,
            q_mat,
            k_mat,
            v_chunk,
            o_chunk,
            layer_d_head,
            recorder=recorder,
            node=op.node,
            op_type=op.op_type,
        )


def _self_match_source(d_head, rmap, const_one):
    """Q/K source columns + matrices for a Δ=0 self-match head.

    Reads the single reserved constant-1 column and projects it to
    ``hardness · ones`` (query) / ``ones`` (key) across all ``d_head`` dims.
    The runtime rotates this head by absolute position (rotate_half over the
    rotary front), and the logit ``∝ Σ_p cos((i − j)·θ_p)`` (plus a constant from
    any unrotated NoPE tail) peaks at ``i == j`` — a Δ=0 transport that is
    rotation-invariant, so it holds for full or partial rotary alike.  This is a
    compiler-inserted head; it leaves rope_d_rot unset, so it never constrains the
    model's global d_rot.

    Returns ``(qk_idx, q_mat, k_mat)`` where ``qk_idx`` is used for both the query
    and key source (Δ=0 self-match reads one source on both sides).
    """
    const_idx = rmap.get_indices(const_one)  # width 1
    q_mat = _SELF_MATCH_HARDNESS * torch.ones(1, d_head)
    k_mat = torch.ones(1, d_head)
    return const_idx, q_mat, k_mat


def _write_compute_linear(
    attn,
    op: AttnHeadOp,
    rmap: ResidualStreamMap,
    recorder: Optional[PlacementRecorder] = None,
    const_one: Optional[Node] = None,
):
    """Compile a zero-bias Linear via current-position attention.

    Q/K attend to the current position (Δ=0 self-match) — over the trig grid
    (``const_one is None``) or the rotary constant-1 column (``const_one`` set).
    V reads from the Linear's input columns.
    O applies the Linear's weight matrix to target columns.

    For d_input > d_head, splits across multiple heads. Each head handles
    a d_head-sized chunk of the input and applies the corresponding slice
    of the weight matrix. Attention heads are additive, so results sum
    to give the full W @ input.
    """
    node = op.node
    assert isinstance(node, Linear)

    input_node = node.inputs[0]
    d_head = attn.d_head
    d_input = len(input_node)

    assert op.source_cols is not None, "compute_linear requires source_cols"
    qk_idx, q_mat, k_mat = _self_match_source(d_head, rmap, const_one)
    v_idx = op.source_cols
    o_idx = op.target_cols

    # Split input across ceil(d_input / d_head) heads
    for start in range(0, d_input, d_head):
        end = min(start + d_head, d_input)
        chunk_size = end - start

        v_chunk_idx = v_idx[start:end]
        v_mat = torch.eye(chunk_size, d_head)

        # Weight matrix slice for this chunk: rows [start:end]
        weight_slice = node.output_matrix[start:end, :]  # (chunk_size, d_output)
        o_mat = F.pad(weight_slice, (0, 0, 0, d_head - chunk_size))

        head = _allocate_head(attn)
        attn.rope_base = ROPE_BASE
        _scatter_attn_head(
            attn,
            head,
            qk_idx,
            qk_idx,
            v_chunk_idx,
            o_idx,
            q_mat,
            k_mat,
            v_mat,
            o_mat,
            d_head,
            recorder=recorder,
            node=op.node,
            op_type=op.op_type,
        )


def _write_compute_add(
    attn,
    op: AttnHeadOp,
    rmap: ResidualStreamMap,
    recorder: Optional[PlacementRecorder] = None,
    const_one: Optional[Node] = None,
):
    """Compute Add(a, b) by copying both inputs to fresh columns via attention.

    Used when neither input is dead (so add_into can't reuse columns).
    Allocates new output columns and uses attention heads to copy both inputs.
    Since attention heads are additive, the result in the output columns is
    0 + a + b = a + b.  Q/K self-match the current position over the trig grid
    (``const_one is None``) or the rotary constant-1 column (``const_one`` set).

    When 2 * chunk_size <= d_head, both inputs are combined into a single
    head: a0 maps into head dims [0:cs] and a1 into [cs:2cs], with the
    output matrix summing both contributions to the same output columns.
    Otherwise falls back to one head per input.
    """
    node = op.node
    assert isinstance(node, Add)
    assert op.source_cols is not None, "compute_add requires source_cols"
    assert op.source_cols_b is not None, "compute_add requires source_cols_b"

    d_head = attn.d_head
    d_output = len(node)
    o_idx = op.target_cols
    qk_idx, q_mat, k_mat = _self_match_source(d_head, rmap, const_one)

    v_idx_a = op.source_cols
    v_idx_b = op.source_cols_b

    for start in range(0, d_output, d_head):
        end = min(start + d_head, d_output)
        chunk_size = end - start
        o_chunk_idx = o_idx[start:end]

        if 2 * chunk_size <= d_head:
            # Both inputs fit in a single head.
            # a0 occupies head dims [0:cs], a1 occupies [cs:2*cs].
            # Output matrix sums both into the same output columns.
            combined_v_idx = v_idx_a[start:end] + v_idx_b[start:end]
            v_mat = torch.eye(2 * chunk_size, d_head)
            o_mat = torch.zeros(d_head, chunk_size)
            o_mat[:chunk_size, :chunk_size] = torch.eye(chunk_size)
            o_mat[chunk_size : 2 * chunk_size, :chunk_size] = torch.eye(chunk_size)
            head = _allocate_head(attn)
            attn.rope_base = ROPE_BASE
            _scatter_attn_head(
                attn,
                head,
                qk_idx,
                qk_idx,
                combined_v_idx,
                o_chunk_idx,
                q_mat,
                k_mat,
                v_mat,
                o_mat,
                d_head,
                recorder=recorder,
                node=op.node,
                op_type=op.op_type,
            )
        else:
            # Too wide to combine — one head per input.
            for v_idx_x in [v_idx_a, v_idx_b]:
                v_chunk_idx = v_idx_x[start:end]
                v_mat = torch.eye(chunk_size, d_head)
                o_mat = torch.eye(d_head, chunk_size)
                head = _allocate_head(attn)
                attn.rope_base = ROPE_BASE
                _scatter_attn_head(
                    attn,
                    head,
                    qk_idx,
                    qk_idx,
                    v_chunk_idx,
                    o_chunk_idx,
                    q_mat,
                    k_mat,
                    v_mat,
                    o_mat,
                    d_head,
                    recorder=recorder,
                    node=op.node,
                    op_type=op.op_type,
                )


def _write_cancel(
    attn,
    op: AttnHeadOp,
    rmap: ResidualStreamMap,
    recorder: Optional[PlacementRecorder] = None,
    const_one: Optional[Node] = None,
):
    """Cancel target_cols: V=identity, O=-identity. Skip adds x + (-x) = 0.

    Splits across multiple heads when wider than d_head.  Used both to
    zero a dead node's columns before reuse and to clear dirty columns
    that the caller's initial residual stream may have populated with
    garbage.  Q/K self-match the current position over the trig grid
    (``const_one is None``) or the rotary constant-1 column (``const_one`` set).
    """
    d_head = attn.d_head
    d_width = len(op.target_cols)

    node_idx = op.target_cols
    qk_idx, q_mat, k_mat = _self_match_source(d_head, rmap, const_one)

    for start in range(0, d_width, d_head):
        end = min(start + d_head, d_width)
        chunk_size = end - start

        chunk_idx = node_idx[start:end]
        v_mat = torch.eye(chunk_size, d_head)
        o_mat = -torch.eye(d_head, chunk_size)

        head = _allocate_head(attn)
        attn.rope_base = ROPE_BASE
        _scatter_attn_head(
            attn,
            head,
            qk_idx,
            qk_idx,
            chunk_idx,
            chunk_idx,
            q_mat,
            k_mat,
            v_mat,
            o_mat,
            d_head,
            recorder=recorder,
            node=op.node,
            op_type=op.op_type,
        )


def _write_add_into(
    attn,
    op: AttnHeadOp,
    rmap: ResidualStreamMap,
    recorder: Optional[PlacementRecorder] = None,
    const_one: Optional[Node] = None,
):
    """Add(dead, live): copy live's values to dead's columns via attention.

    target_cols are the dead addend's columns (now owned by the Add node
    after reassign). The live addend is whichever input is still allocated
    in the residual map. Skip connection adds: dead + live = Add(dead, live).
    Q/K self-match the current position over the trig grid (``const_one is
    None``) or the rotary constant-1 column (``const_one`` set).

    Splits across multiple heads for operands wider than d_head.
    """
    node = op.node
    assert isinstance(node, Add)

    assert op.source_cols is not None, "add_into requires source_cols (live addend)"
    d_head = attn.d_head
    v_idx = op.source_cols  # live addend cols, captured at schedule time
    d_live = len(v_idx)

    o_idx = op.target_cols  # dead addend's columns
    qk_idx, q_mat, k_mat = _self_match_source(d_head, rmap, const_one)

    for start in range(0, d_live, d_head):
        end = min(start + d_head, d_live)
        chunk_size = end - start

        v_chunk_idx = v_idx[start:end]
        o_chunk_idx = o_idx[start:end]

        v_mat = torch.eye(chunk_size, d_head)
        o_mat = torch.eye(d_head, chunk_size)

        head = _allocate_head(attn)
        attn.rope_base = ROPE_BASE
        _scatter_attn_head(
            attn,
            head,
            qk_idx,
            qk_idx,
            v_chunk_idx,
            o_chunk_idx,
            q_mat,
            k_mat,
            v_mat,
            o_mat,
            d_head,
            recorder=recorder,
            node=op.node,
            op_type=op.op_type,
        )


# ---------------------------------------------------------------------------
# MLP operations
# ---------------------------------------------------------------------------


def _write_compute_ffn(
    mlp,
    op: MLPOp,
    rmap: ResidualStreamMap,
    biased_linears: Optional[Set[Node]] = None,
    recorder: Optional[PlacementRecorder] = None,
):
    """Compile a degenerate-ReLU :class:`FFN` through the MLP sublayer.

    The FFN is the only MLP composite: it maps its input through the gate
    projection into the hidden slots, applies the sublayer's built-in ReLU, and
    maps back out through the output projection:

    - ``linear1``: input columns -> mlp_slots, weight ``gate_proj.T`` (the
      FFN stores ``gate_proj`` as ``(n_lanes, d_input)``; the MLP linear1
      matrix is ``(d_input, n_lanes)``), bias ``gate_bias``.
    - ReLU: applied elementwise (the sublayer's built-in nonlinearity).
    - ``linear2``: mlp_slots -> target columns, weight ``out_proj``
      ``(n_lanes, d_output)``, bias ``out_bias``.
    """
    ffn = op.node
    assert isinstance(ffn, FFN)
    assert ffn.activation == "relu" and ffn.up_proj is None, (
        f"compute_ffn supports only degenerate ReLU FFNs in step 1, got "
        f"activation={ffn.activation!r}, gated={ffn.up_proj is not None}"
    )

    assert op.source_cols is not None, "compute_ffn requires source_cols"
    input_node = ffn.inputs[0]
    in_idx = op.source_cols
    mlp_slots = op.mlp_slots
    out_idx = op.target_cols

    in_idx_t = torch.as_tensor(in_idx, dtype=torch.long)
    slots_t = torch.as_tensor(mlp_slots, dtype=torch.long)
    out_idx_t = torch.as_tensor(out_idx, dtype=torch.long)

    target_dtype = mlp.linear1.output_matrix.dtype

    # linear1: rows=input columns, cols=mlp slots.  gate matrix is
    # (d_input, n_lanes) = gate_proj.T; gate bias is (n_lanes,).
    gate_matrix = ffn.gate_proj.t()
    mlp.linear1.output_matrix[in_idx_t.unsqueeze(1), slots_t.unsqueeze(0)] = (
        gate_matrix.to(target_dtype)
    )
    mlp.linear1.output_bias[slots_t] = ffn.gate_bias.to(target_dtype)

    # Fold deferred biased-Linear inputs into the gate's hidden bias, because
    # the FFN reads those columns in linear1 before the deferred output_bias
    # is written.
    if biased_linears:
        leaves = (
            flatten_concat_nodes([input_node])
            if isinstance(input_node, Concatenate)
            else [input_node]
        )
        offset = 0
        for leaf in leaves:
            if leaf in biased_linears:
                assert isinstance(leaf, Linear)
                contrib = leaf.output_bias @ gate_matrix[offset : offset + len(leaf), :]
                mlp.linear1.output_bias[slots_t] += contrib.to(target_dtype)
            offset += len(leaf)

    # linear2: rows=mlp slots, cols=output columns.  out_proj is
    # (n_lanes, d_output); out_bias is (d_output,).
    mlp.linear2.output_matrix[slots_t.unsqueeze(1), out_idx_t.unsqueeze(0)] = (
        ffn.out_proj.to(target_dtype)
    )
    mlp.linear2.output_bias[out_idx_t] = ffn.out_bias.to(target_dtype)

    if recorder is not None:
        # Both dense weight blocks attributed to the FFN node (it carries all
        # the weights, unlike a mined chain's split across L1/L2).
        recorder.add("mlp.W_in", ffn, op.op_type, in_idx, mlp_slots, "dense")
        recorder.add("mlp.W_out", ffn, op.op_type, mlp_slots, out_idx, "dense")


def _write_compute_literal_value(mlp, op: MLPOp):
    """Write a constant value via MLP output bias."""
    node = op.node
    assert isinstance(node, LiteralValue)
    assert len(op.target_cols) == node.value.numel(), (
        f"Literal truncation would drop values from {node!r}: "
        f"value.numel()={node.value.numel()}, "
        f"target_cols len={len(op.target_cols)}. "
        f"Slicing would silently lose trailing entries — compiler bug."
    )
    cols_t = torch.as_tensor(op.target_cols, dtype=torch.long)
    target_dtype = mlp.linear2.output_bias.dtype
    mlp.linear2.output_bias[cols_t] = node.value.to(target_dtype)


def _write_compute_bias(mlp, op: MLPOp):
    """Add bias to MLP output bias (for biased Linear split)."""
    node = op.node
    assert isinstance(node, Linear)
    assert len(op.target_cols) == node.output_bias.numel(), (
        f"Bias truncation would drop values from {node!r}: "
        f"output_bias.numel()={node.output_bias.numel()}, "
        f"target_cols len={len(op.target_cols)}. "
        f"Slicing would silently lose trailing entries — compiler bug."
    )
    cols_t = torch.as_tensor(op.target_cols, dtype=torch.long)
    target_dtype = mlp.linear2.output_bias.dtype
    mlp.linear2.output_bias[cols_t] += node.output_bias.to(target_dtype)


def _write_compute_linear_bypass(
    mlp,
    op: MLPOp,
    rmap: ResidualStreamMap,
    biased_linears: Optional[Set[Node]] = None,
    recorder: Optional[PlacementRecorder] = None,
):
    """Compile a standalone Linear via MLP using the ReLU bypass trick.

    ReLU(z) - ReLU(-z) = z, so two MLP slots per output column recover
    an arbitrary linear function without the ReLU nonlinearity.

    Slot layout: mlp_slots[:d_output] are the positive slots,
    mlp_slots[d_output:] are the negative slots.
    """
    node = op.node
    assert isinstance(node, Linear)

    assert op.source_cols is not None, "compute_linear_bypass requires source_cols"
    in_idx = op.source_cols
    out_idx = op.target_cols
    mlp_slots = op.mlp_slots
    d_output = node.d_output

    assert len(mlp_slots) == 2 * d_output, (
        f"compute_linear_bypass needs 2*d_output={2 * d_output} slots, "
        f"got {len(mlp_slots)}"
    )

    pos_slots = mlp_slots[:d_output]
    neg_slots = mlp_slots[d_output:]

    in_idx_t = torch.as_tensor(in_idx, dtype=torch.long)
    out_idx_t = torch.as_tensor(out_idx, dtype=torch.long)
    pos_slots_t = torch.as_tensor(pos_slots, dtype=torch.long)
    neg_slots_t = torch.as_tensor(neg_slots, dtype=torch.long)

    target_dtype = mlp.linear1.output_matrix.dtype
    W = node.output_matrix.to(target_dtype)  # (d_input, d_output)

    # linear1: positive slots get +W columns, negative slots get -W columns.
    # Each positive slot j computes W[:,j]^T @ x; after ReLU → max(0, W[:,j]^T @ x).
    # Each negative slot j computes -W[:,j]^T @ x; after ReLU → max(0, -W[:,j]^T @ x).
    mlp.linear1.output_matrix[in_idx_t.unsqueeze(1), pos_slots_t.unsqueeze(0)] = W
    mlp.linear1.output_matrix[in_idx_t.unsqueeze(1), neg_slots_t.unsqueeze(0)] = -W

    # Fold deferred biased-Linear inputs into linear1 slot biases.
    if biased_linears:
        input_node = node.inputs[0]
        leaves = (
            flatten_concat_nodes([input_node])
            if isinstance(input_node, Concatenate)
            else [input_node]
        )
        bias_dtype = mlp.linear1.output_bias.dtype
        offset = 0
        for leaf in leaves:
            if leaf in biased_linears:
                assert isinstance(leaf, Linear)
                # contrib[j] = sum_i W[offset+i, j] * leaf.bias[i]
                contrib = (leaf.output_bias @ W[offset : offset + len(leaf), :]).to(
                    bias_dtype
                )
                mlp.linear1.output_bias[pos_slots_t] += contrib
                mlp.linear1.output_bias[neg_slots_t] -= contrib
            offset += len(leaf)

    # linear2: positive slots write +1 to output columns, negative write -1.
    # Combined: max(0, W[:,j]^T @ x) - max(0, -W[:,j]^T @ x) = W[:,j]^T @ x.
    mlp.linear2.output_matrix[pos_slots_t, out_idx_t] = 1.0
    mlp.linear2.output_matrix[neg_slots_t, out_idx_t] = -1.0

    # Output bias via linear2 output bias.
    mlp.linear2.output_bias[out_idx_t] += node.output_bias.to(target_dtype)

    if recorder is not None:
        # W_in: dense ±W blocks into the positive and negative slot halves.
        # W_out: paired ±1 identity from each slot half back to out_idx.
        recorder.add("mlp.W_in", node, op.op_type, in_idx, pos_slots, "dense")
        recorder.add("mlp.W_in", node, op.op_type, in_idx, neg_slots, "dense")
        recorder.add("mlp.W_out", node, op.op_type, pos_slots, out_idx, "diag")
        recorder.add("mlp.W_out", node, op.op_type, neg_slots, out_idx, "diag")
