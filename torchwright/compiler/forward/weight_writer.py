"""Writes weight matrices into TransformerLayer components.

Each operation directly sets matrix entries using column indices from the
ResidualStreamMap. No strategies, no ResidualAssignment — just scatter writes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from torchwright.compiler.realization import linear_attn_chunks
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.graph import Node, Linear, Attn, Add, Concatenate
from torchwright.graph.misc import LiteralValue
from torchwright.graph.ffn import FFN
from torchwright.graph.rope import ROPE_BASE

# The swish machine's hinge-sharpening constant (a sideways import of a leaf
# constants module — ops/const.py has no imports, so no cycle).  Used only by
# the swish bypass pair below; ops fold it into FFN weights themselves.
# The bias-lane constants build the no-bias constant lane (see BiasFold).
from torchwright.ops.const import (
    bias_lane_gate as _BIAS_LANE_GATE,
    bias_lane_up as _BIAS_LANE_UP,
    scale as _SWISH_SCALE,
)

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
        "cancel_bypass",
    ]
    # None for ``cancel_bypass`` — the op zeroes a dying node's columns but is
    # not itself a graph node (like the attention ``cancel``, which is outside
    # invariant I2's literal/bias scope).
    node: Optional[Node]
    target_cols: List[int]
    mlp_slots: List[int] = field(default_factory=list)
    # Input columns captured at schedule time.  Used by compute_ffn (the FFN's
    # input), compute_linear_bypass (the Linear's input), and cancel_bypass
    # (the dying node's own columns — it reads what it negates).
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
    const_one: Optional[Node] = None,
    bias: bool = True,
):
    """Write MLP operations into a layer's MLPSubLayer components.

    Under ``bias=False`` (docs/no_bias_plan.md) every bias write folds into
    the weight matrices against the pinned constant-1 column: hidden-side
    biases as const-column rows, output-side constants through the constant
    lane at hidden slot 0 (see :class:`BiasFold`).  ``const_one`` is required
    then; the scheduler must have reserved slot 0 (packing from slot 1).
    """
    if biased_linears is None:
        biased_linears = set()
    bias_fold: Optional[BiasFold] = None
    if not bias:
        assert const_one is not None, "bias=False requires const_one"
        const_col = residual_map.get_indices(const_one)[0]
        bias_fold = BiasFold(layer.mlp, const_col, recorder)
        for op in ops:
            assert BiasFold.LANE_SLOT not in op.mlp_slots, (
                f"{op.op_type} for {op.node!r} claims hidden slot "
                f"{BiasFold.LANE_SLOT}, which bias=False reserves for the "
                f"constant lane — the scheduler must pack from slot 1. "
                f"Compiler bug."
            )
    for op in ops:
        if op.op_type == "compute_ffn":
            _write_compute_ffn(
                layer.mlp, op, residual_map, biased_linears, recorder, bias_fold
            )
        elif op.op_type == "compute_literal_value":
            _write_compute_literal_value(layer.mlp, op, bias_fold)
        elif op.op_type == "compute_bias":
            _write_compute_bias(layer.mlp, op, bias_fold)
        elif op.op_type == "compute_linear_bypass":
            _write_compute_linear_bypass(
                layer.mlp, op, residual_map, biased_linears, recorder, bias_fold
            )
        elif op.op_type == "cancel_bypass":
            _write_cancel_bypass(layer.mlp, op, recorder, bias_fold)
        else:
            raise ValueError(f"Unknown mlp op_type: {op.op_type}")


# ---------------------------------------------------------------------------
# Attention operations
# ---------------------------------------------------------------------------


def _coalesce_source_rows(
    idx: List[int], mat: torch.Tensor
) -> "Tuple[List[int], torch.Tensor]":
    """Sum matrix rows that share a source column index.

    Weight scatters assign rows by advanced indexing, which is
    last-write-wins on duplicate indices.  Duplicate source columns are
    legal — a ``Concatenate`` can hold the same leaf more than once, so
    ``resolve_indices`` yields that leaf's columns once per occurrence —
    and a residual value read through k matrix rows contributes the sum
    of those rows, so summing is exact.  First-occurrence order is kept
    (assignment order is irrelevant once indices are unique; this just
    keeps the result deterministic).
    """
    if len(set(idx)) == len(idx):
        return idx, mat
    order: List[int] = []
    row_of: Dict[int, int] = {}
    rows: List[torch.Tensor] = []
    for i, col in enumerate(idx):
        j = row_of.get(col)
        if j is None:
            row_of[col] = len(order)
            order.append(col)
            rows.append(mat[i])
        else:
            rows[j] = rows[j] + mat[i]
    return order, torch.stack(rows)


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
    # A source-column list can carry duplicates (a Concatenate holding the
    # same leaf twice resolves that leaf's columns once per occurrence);
    # coalesce them before the vectorized assignment below, which is
    # last-write-wins on duplicate indices and would silently drop all but
    # the final occurrence's rows.  Target columns (O) are allocator-unique
    # by I1 and need no coalescing.
    q_idx, q_rows = _coalesce_source_rows(q_idx, q_mat[: len(q_idx), :d_head])
    k_idx, k_rows = _coalesce_source_rows(k_idx, k_mat[: len(k_idx), :d_head])
    v_idx, v_rows = _coalesce_source_rows(v_idx, v_mat[: len(v_idx), :d_head])
    q_idx_t = torch.as_tensor(q_idx, dtype=torch.long)
    k_idx_t = torch.as_tensor(k_idx, dtype=torch.long)
    v_idx_t = torch.as_tensor(v_idx, dtype=torch.long)
    o_idx_t = torch.as_tensor(o_idx, dtype=torch.long)
    # Source matrices may be Long (e.g. integer Linear weights); the
    # destination is always Float.  Original scalar-loop assignment
    # auto-cast; vectorized assignment does not.
    target_dtype = attn.query_matrix.dtype
    attn.query_matrix[head, q_idx_t, :d_head] = q_rows.to(target_dtype)
    attn.key_matrix[head, k_idx_t, :d_head] = k_rows.to(target_dtype)
    attn.value_matrix[head, v_idx_t, :d_head] = v_rows.to(target_dtype)
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

    Only chunks with a live weight row get a head: a chunk whose rows of
    ``output_matrix`` are all zero would emit a head with O == 0, which
    contributes exactly zero to the target columns.  The chunk list comes
    from ``realization.linear_attn_chunks`` — the same list every
    scheduler charge site counts — so the head budget and this emission
    cannot desync.  (Its floor keeps one zero-weight head for a
    zero-support Linear; the target columns then correctly hold 0 plus
    whatever ``compute_bias`` writes.)
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

    for chunk in linear_attn_chunks(node, d_head):
        start = chunk * d_head
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

    A self-add — ``Add(h, h)``, the one case where both addends resolve
    to the SAME residual columns (I1 forbids column sharing between
    distinct live nodes) — cannot use the combined-head scatter: the
    duplicated column indices would collide in the vectorized V-matrix
    assignment (last write wins), silently dropping one addend.  It gets
    a single V mapping with a doubled output matrix instead, which is
    exact.
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
    self_add = v_idx_a == v_idx_b

    for start in range(0, d_output, d_head):
        end = min(start + d_head, d_output)
        chunk_size = end - start
        o_chunk_idx = o_idx[start:end]

        if self_add:
            # One copy of the shared columns, output matrix doubled.
            v_mat = torch.eye(chunk_size, d_head)
            o_mat = 2.0 * torch.eye(d_head, chunk_size)
            head = _allocate_head(attn)
            attn.rope_base = ROPE_BASE
            _scatter_attn_head(
                attn,
                head,
                qk_idx,
                qk_idx,
                v_idx_a[start:end],
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
        elif 2 * chunk_size <= d_head:
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


class BiasFold:
    """Bias-write redirector for ``bias=False`` compiles (docs/no_bias_plan.md).

    Hidden-side biases (**fold 1**) become rows of the input-side matrices at
    the pinned constant-1 column: that residual column always holds exactly
    1.0, so the matmul picks up the bias term.  Output-side constants
    (**fold 2** — literals, deferred Linear biases, FFN out biases) route
    through the **constant lane**: hidden slot 0, reserved by the scheduler
    under ``bias=False`` and written lazily here to compute exactly 1.0, so
    its down-projection row carries each constant verbatim.

    Lane exactness: ReLU machine ``ReLU(1·1) = 1``; swish machine
    ``swish(32·1) · ((1/32)·1) = 32 · σ(32) · 2⁻⁵ = 1.0`` bit-exactly — fp32
    σ saturates to 1.0 from z ≥ 17 (torch/ORT-CUDA; z ≥ 18 CPU-ORT, the A0
    probes) and both lane constants are powers of two, so no product rounds.
    Constants live in ``ops/const.py`` (``bias_lane_gate``/``bias_lane_up``).
    """

    LANE_SLOT = 0

    def __init__(
        self,
        mlp,
        const_col: int,
        recorder: Optional[PlacementRecorder] = None,
    ):
        self.mlp = mlp
        self.const_col = const_col
        self.recorder = recorder
        self._lane_written = False

    def hidden_bias(
        self,
        lin,
        matrix_kind: str,
        slots: List[int],
        values: torch.Tensor,
        *,
        node: Node,
        op_type: str,
        accumulate: bool = False,
    ) -> None:
        """Fold 1: write ``values`` at ``lin.output_matrix[const_col, slots]``."""
        slots_t = torch.as_tensor(slots, dtype=torch.long)
        vals = values.to(lin.output_matrix.dtype)
        if accumulate:
            lin.output_matrix[self.const_col, slots_t] += vals
        else:
            lin.output_matrix[self.const_col, slots_t] = vals
        if self.recorder is not None:
            self.recorder.add(
                matrix_kind, node, op_type, [self.const_col], list(slots), "dense"
            )

    def out_bias(
        self,
        out_lin,
        cols: List[int],
        values: torch.Tensor,
        *,
        node: Node,
        op_type: str,
        accumulate: bool = False,
    ) -> None:
        """Fold 2: write ``values`` at the constant lane's down-projection row."""
        vals = values.to(out_lin.output_matrix.dtype)
        if not bool((vals != 0).any()):
            # An all-zero write contributes nothing whether or not the lane
            # exists — skip so a zero-bias layer never activates the lane.
            return
        self._ensure_lane(node, op_type)
        cols_t = torch.as_tensor(cols, dtype=torch.long)
        if accumulate:
            out_lin.output_matrix[self.LANE_SLOT, cols_t] += vals
        else:
            out_lin.output_matrix[self.LANE_SLOT, cols_t] = vals
        if self.recorder is not None:
            self.recorder.add(
                "mlp.W_out", node, op_type, [self.LANE_SLOT], list(cols), "dense"
            )

    def _ensure_lane(self, node: Node, op_type: str) -> None:
        """Write the lane's gate/up rows once per layer, on first use."""
        if self._lane_written:
            return
        self._lane_written = True
        in_lin, _ = _mlp_in_out(self.mlp)
        assert in_lin.output_matrix[self.const_col, self.LANE_SLOT] == 0.0, (
            f"constant-lane slot {self.LANE_SLOT} gate cell already written — "
            f"the scheduler must reserve slot {self.LANE_SLOT} under "
            f"bias=False.  Compiler bug."
        )
        if self.mlp.activation == "swish":
            in_lin.output_matrix[self.const_col, self.LANE_SLOT] = _BIAS_LANE_GATE
            self.mlp.up_proj.output_matrix[self.const_col, self.LANE_SLOT] = (
                _BIAS_LANE_UP
            )
        else:
            in_lin.output_matrix[self.const_col, self.LANE_SLOT] = 1.0
        if self.recorder is not None:
            self.recorder.add(
                "mlp.W_in", node, op_type, [self.const_col], [self.LANE_SLOT], "dense"
            )
            if self.mlp.activation == "swish":
                self.recorder.add(
                    "mlp.W_up",
                    node,
                    op_type,
                    [self.const_col],
                    [self.LANE_SLOT],
                    "dense",
                )


def _mlp_in_out(mlp):
    """The hidden pool's (input-side, output-side) projections, either machine.

    ReLU machine: ``(linear1, linear2)``.  Swish machine: ``(gate_proj,
    down_proj)`` — the gate is the input-side projection every op writes;
    the up projection is swish-machine-specific and accessed explicitly.
    """
    if mlp.activation == "swish":
        return mlp.gate_proj, mlp.down_proj
    return mlp.linear1, mlp.linear2


def _write_compute_ffn(
    mlp,
    op: MLPOp,
    rmap: ResidualStreamMap,
    biased_linears: Optional[Set[Node]] = None,
    recorder: Optional[PlacementRecorder] = None,
    bias_fold: Optional[BiasFold] = None,
):
    """Compile an :class:`FFN` through the MLP sublayer (either machine).

    The FFN is the only MLP composite: it maps its input through the gate
    projection into the hidden slots, applies the machine's nonlinearity, and
    maps back out through the output projection.

    ReLU machine (``mlp.activation == "relu"``, degenerate FFNs only):

    - ``linear1``: input columns -> mlp_slots, weight ``gate_proj.T`` (the
      FFN stores ``gate_proj`` as ``(n_lanes, d_input)``; the MLP linear1
      matrix is ``(d_input, n_lanes)``), bias ``gate_bias``.
    - ReLU: applied elementwise (the sublayer's built-in nonlinearity).
    - ``linear2``: mlp_slots -> target columns, weight ``out_proj``
      ``(n_lanes, d_output)``, bias ``out_bias``.

    Swish machine (``mlp.activation == "swish"``): gate rows land the same
    way in ``gate_proj``; every lane additionally gets its up-side row in
    ``up_proj`` — the FFN's own ``up_proj``/``up_bias`` for gated lanes, or
    up-row 0 / up-bias 1 for degenerate lanes (the constant-1 up factor);
    ``down_proj`` takes ``out_proj``/``out_bias``.

    The machine kind is uniform per network and selected from the graph's
    FFNs (the compile-time uniformity check), so a node/machine activation
    mismatch here is a compiler bug, not a user error.
    """
    ffn = op.node
    assert isinstance(ffn, FFN)
    assert ffn.activation == mlp.activation, (
        f"compute_ffn machine mismatch for {ffn!r}: node activation "
        f"{ffn.activation!r} on a {mlp.activation!r}-machine MLP sublayer. "
        f"The uniformity check selects the machine from the graph's FFNs, "
        f"so this is a compiler bug."
    )
    if mlp.activation == "relu":
        assert ffn.up_proj is None, (
            f"compute_ffn: gated FFN {ffn!r} on the ReLU machine — the ReLU "
            f"MLP sublayer has no up projection to realize a gated lane."
        )

    assert op.source_cols is not None, "compute_ffn requires source_cols"
    input_node = ffn.inputs[0]
    in_idx = op.source_cols
    mlp_slots = op.mlp_slots
    out_idx = op.target_cols

    slots_t = torch.as_tensor(mlp_slots, dtype=torch.long)
    out_idx_t = torch.as_tensor(out_idx, dtype=torch.long)

    in_lin, out_lin = _mlp_in_out(mlp)
    target_dtype = in_lin.output_matrix.dtype

    # The const-column rows are the bias-fold destination and must never
    # alias a real input row (const_one is compiler-internal, so no graph
    # node can resolve to its column — this pins that invariant).
    if bias_fold is not None:
        assert bias_fold.const_col not in set(in_idx), (
            f"compute_ffn input for {ffn!r} reads the pinned constant-1 "
            f"column {bias_fold.const_col}, which bias=False uses as the "
            f"bias-fold row.  Compiler bug."
        )

    # Input-side projection: rows=input columns, cols=mlp slots.  gate matrix
    # is (d_input, n_lanes) = gate_proj.T; gate bias is (n_lanes,).
    # Duplicate source columns (same Concatenate leaf twice) coalesce by
    # row-summing at the scatter only — the vectorized assignment is
    # last-write-wins.  ``gate_matrix`` keeps the per-occurrence row layout
    # (the biased-linears fold below walks it by leaf offset).
    gate_matrix = ffn.gate_proj.t()
    scatter_idx, scatter_rows = _coalesce_source_rows(in_idx, gate_matrix)
    scatter_idx_t = torch.as_tensor(scatter_idx, dtype=torch.long)
    in_lin.output_matrix[scatter_idx_t.unsqueeze(1), slots_t.unsqueeze(0)] = (
        scatter_rows.to(target_dtype)
    )
    if bias_fold is None:
        in_lin.output_bias[slots_t] = ffn.gate_bias.to(target_dtype)
    else:
        bias_fold.hidden_bias(
            in_lin, "mlp.W_in", mlp_slots, ffn.gate_bias, node=ffn, op_type=op.op_type
        )

    # Swish machine: the up-side rows.  A degenerate lane's up factor is the
    # constant 1 (row 0, bias 1 — or, under bias=False, const-column row 1);
    # a gated lane carries its own affine.
    up_matrix = None
    if mlp.activation == "swish":
        if ffn.up_proj is None:
            up_bias = torch.ones(len(mlp_slots))
        else:
            up_matrix = ffn.up_proj.t()
            up_bias = ffn.up_bias
            _, up_rows = _coalesce_source_rows(in_idx, up_matrix)
            mlp.up_proj.output_matrix[
                scatter_idx_t.unsqueeze(1), slots_t.unsqueeze(0)
            ] = up_rows.to(target_dtype)
        if bias_fold is None:
            mlp.up_proj.output_bias[slots_t] = up_bias.to(target_dtype)
        else:
            bias_fold.hidden_bias(
                mlp.up_proj,
                "mlp.W_up",
                mlp_slots,
                up_bias,
                node=ffn,
                op_type=op.op_type,
            )

    # Fold deferred biased-Linear inputs into the hidden biases, because the
    # FFN reads those columns in the input-side matmuls (gate, and up when
    # gated) before the deferred output_bias is written.
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
                if bias_fold is None:
                    in_lin.output_bias[slots_t] += contrib.to(target_dtype)
                else:
                    bias_fold.hidden_bias(
                        in_lin,
                        "mlp.W_in",
                        mlp_slots,
                        contrib,
                        node=ffn,
                        op_type=op.op_type,
                        accumulate=True,
                    )
                if up_matrix is not None:
                    up_contrib = (
                        leaf.output_bias @ up_matrix[offset : offset + len(leaf), :]
                    )
                    if bias_fold is None:
                        mlp.up_proj.output_bias[slots_t] += up_contrib.to(target_dtype)
                    else:
                        bias_fold.hidden_bias(
                            mlp.up_proj,
                            "mlp.W_up",
                            mlp_slots,
                            up_contrib,
                            node=ffn,
                            op_type=op.op_type,
                            accumulate=True,
                        )
            offset += len(leaf)

    # Output-side projection: rows=mlp slots, cols=output columns.  out_proj
    # is (n_lanes, d_output); out_bias is (d_output,).
    out_lin.output_matrix[slots_t.unsqueeze(1), out_idx_t.unsqueeze(0)] = (
        ffn.out_proj.to(target_dtype)
    )
    if bias_fold is None:
        out_lin.output_bias[out_idx_t] = ffn.out_bias.to(target_dtype)
    else:
        bias_fold.out_bias(out_lin, out_idx, ffn.out_bias, node=ffn, op_type=op.op_type)

    if recorder is not None:
        # All dense weight blocks attributed to the FFN node (it carries all
        # the weights, unlike a mined chain's split across L1/L2).  The up
        # matrix is recorded only when real rows were written (a degenerate
        # FFN touches only the up bias, which occupancy does not track).
        recorder.add("mlp.W_in", ffn, op.op_type, in_idx, mlp_slots, "dense")
        if up_matrix is not None:
            recorder.add("mlp.W_up", ffn, op.op_type, in_idx, mlp_slots, "dense")
        recorder.add("mlp.W_out", ffn, op.op_type, mlp_slots, out_idx, "dense")


def _write_compute_literal_value(mlp, op: MLPOp, bias_fold: Optional[BiasFold] = None):
    """Write a constant value via MLP output bias (or the constant lane).

    Under ``bias=False`` the value rides the constant lane's down-projection
    row instead: the lane's value is exactly 1.0 and no other lane touches a
    literal's columns, so the constant still lands bit-exactly.
    """
    node = op.node
    assert isinstance(node, LiteralValue)
    assert len(op.target_cols) == node.value.numel(), (
        f"Literal truncation would drop values from {node!r}: "
        f"value.numel()={node.value.numel()}, "
        f"target_cols len={len(op.target_cols)}. "
        f"Slicing would silently lose trailing entries — compiler bug."
    )
    out_lin = _mlp_in_out(mlp)[1]
    if bias_fold is not None:
        bias_fold.out_bias(
            out_lin, op.target_cols, node.value, node=node, op_type=op.op_type
        )
        return
    cols_t = torch.as_tensor(op.target_cols, dtype=torch.long)
    target_dtype = out_lin.output_bias.dtype
    out_lin.output_bias[cols_t] = node.value.to(target_dtype)


def _write_compute_bias(mlp, op: MLPOp, bias_fold: Optional[BiasFold] = None):
    """Add bias to MLP output bias (for biased Linear split).

    Under ``bias=False`` the deferred bias rides the constant lane's
    down-projection row — same layer, same ordering as the bias add.
    """
    node = op.node
    assert isinstance(node, Linear)
    assert len(op.target_cols) == node.output_bias.numel(), (
        f"Bias truncation would drop values from {node!r}: "
        f"output_bias.numel()={node.output_bias.numel()}, "
        f"target_cols len={len(op.target_cols)}. "
        f"Slicing would silently lose trailing entries — compiler bug."
    )
    out_lin = _mlp_in_out(mlp)[1]
    if bias_fold is not None:
        bias_fold.out_bias(
            out_lin,
            op.target_cols,
            node.output_bias,
            node=node,
            op_type=op.op_type,
            accumulate=True,
        )
        return
    cols_t = torch.as_tensor(op.target_cols, dtype=torch.long)
    target_dtype = out_lin.output_bias.dtype
    out_lin.output_bias[cols_t] += node.output_bias.to(target_dtype)


def _write_bypass_lane_pair(
    mlp,
    in_idx: List[int],
    out_idx: List[int],
    mlp_slots: List[int],
    W: torch.Tensor,
    *,
    node: Optional[Node],
    op_type: str,
    bias_fold: Optional[BiasFold] = None,
    recorder: Optional[PlacementRecorder] = None,
):
    """Emit the activation bypass lane-pair computing ``W.T @ x`` from
    ``in_idx`` into ``out_idx`` over ``mlp_slots`` (== ``2 * len(out_idx)``).

    Shared by :func:`_write_compute_linear_bypass` (``W = node.output_matrix``)
    and :func:`_write_cancel_bypass` (``W = -I``, so the pair emits ``-x`` and
    the skip connection turns ``x`` into ``x + (-x) = 0``).  Value path only:
    ±in_gain·W on the two slot halves, the degenerate up-lanes (up ≡ 1 on the
    swish machine), ±out_gain on the output side, and placement records.  Bias
    folds are the caller's.

    ReLU machine: ``ReLU(z) - ReLU(-z) = z``.  Swish machine:
    ``Swish(scale·z)/scale - Swish(-scale·z)/scale = z`` — the ``scale`` fold is
    self-normalizing (gate rows carry ``scale·W``, output side ``1/scale``), so
    no value path is amplified.  1.0 on the ReLU machine (no sharpening needed).
    """
    d_output = len(out_idx)
    assert len(mlp_slots) == 2 * d_output, (
        f"bypass lane-pair needs 2*d_output={2 * d_output} slots, "
        f"got {len(mlp_slots)}"
    )
    pos_slots = mlp_slots[:d_output]
    neg_slots = mlp_slots[d_output:]
    out_idx_t = torch.as_tensor(out_idx, dtype=torch.long)
    pos_slots_t = torch.as_tensor(pos_slots, dtype=torch.long)
    neg_slots_t = torch.as_tensor(neg_slots, dtype=torch.long)

    in_lin, out_lin = _mlp_in_out(mlp)
    swish = mlp.activation == "swish"
    in_gain = _SWISH_SCALE if swish else 1.0
    out_gain = 1.0 / _SWISH_SCALE if swish else 1.0

    target_dtype = in_lin.output_matrix.dtype
    W = W.to(target_dtype)
    # Input side: positive slots get +in_gain·W columns, negative -in_gain·W.
    # Duplicate source columns (same Concatenate leaf twice) coalesce by
    # row-summing — the vectorized assignment is last-write-wins.
    w_idx, W_rows = _coalesce_source_rows(in_idx, W)
    w_idx_t = torch.as_tensor(w_idx, dtype=torch.long)
    in_lin.output_matrix[w_idx_t.unsqueeze(1), pos_slots_t.unsqueeze(0)] = (
        in_gain * W_rows
    )
    in_lin.output_matrix[w_idx_t.unsqueeze(1), neg_slots_t.unsqueeze(0)] = (
        -in_gain * W_rows
    )

    # Swish machine: both slot halves are degenerate lanes (up ≡ 1).
    if swish:
        if bias_fold is None:
            mlp.up_proj.output_bias[pos_slots_t] = 1.0
            mlp.up_proj.output_bias[neg_slots_t] = 1.0
        else:
            bias_fold.hidden_bias(
                mlp.up_proj,
                "mlp.W_up",
                mlp_slots,
                torch.ones(len(mlp_slots)),
                node=node,
                op_type=op_type,
            )

    # Output side: +out_gain on the positive slots, -out_gain on the negative;
    # combined they recover W[:,j]^T @ x exactly.
    out_lin.output_matrix[pos_slots_t, out_idx_t] = out_gain
    out_lin.output_matrix[neg_slots_t, out_idx_t] = -out_gain

    if recorder is not None:
        recorder.add("mlp.W_in", node, op_type, in_idx, pos_slots, "dense")
        recorder.add("mlp.W_in", node, op_type, in_idx, neg_slots, "dense")
        recorder.add("mlp.W_out", node, op_type, pos_slots, out_idx, "diag")
        recorder.add("mlp.W_out", node, op_type, neg_slots, out_idx, "diag")


def _write_cancel_bypass(
    mlp,
    op: MLPOp,
    recorder: Optional[PlacementRecorder] = None,
    bias_fold: Optional[BiasFold] = None,
):
    """Zero a dying node's columns from the MLP sublayer via the bypass pair
    with ``W = -I``: the lane-pair emits ``-x`` and the skip connection turns
    the column's ``x`` into ``x + (-x) = 0`` (up to the swish two-lane fp32
    residue measured in ``scripts/measure_mlp_cancel_residue.py``; the ReLU
    pair is bit-exact).  ``source_cols == target_cols`` — it reads what it
    negates — and there is no graph node (``op.node is None``).
    """
    assert op.source_cols is not None, "cancel_bypass requires source_cols"
    cols = op.target_cols
    assert op.source_cols == cols, (
        "cancel_bypass reads and writes the same columns: "
        f"source={op.source_cols} target={cols}"
    )
    in_lin, _ = _mlp_in_out(mlp)
    neg_I = -torch.eye(len(cols), dtype=in_lin.output_matrix.dtype)
    _write_bypass_lane_pair(
        mlp,
        cols,
        cols,
        op.mlp_slots,
        neg_I,
        node=None,
        op_type=op.op_type,
        bias_fold=bias_fold,
        recorder=recorder,
    )


def _write_compute_linear_bypass(
    mlp,
    op: MLPOp,
    rmap: ResidualStreamMap,
    biased_linears: Optional[Set[Node]] = None,
    recorder: Optional[PlacementRecorder] = None,
    bias_fold: Optional[BiasFold] = None,
):
    """Compile a standalone Linear via MLP using the activation bypass pair.

    ReLU machine: ``ReLU(z) - ReLU(-z) = z``.  Swish machine:
    ``Swish(scale·z)/scale - Swish(-scale·z)/scale = z`` — exact at any
    sharpening (``σ(z) + σ(-z) = 1``), sharpened by the module ``scale``
    by convention: the value is exact either way, but the affine-bound
    sandwich slack on the pair is ``±0.2785/scale`` sharpened versus
    ``±0.2785`` raw (see the spec's ``min`` entry).  Either way, two MLP
    slots per output column recover an arbitrary linear function without
    the nonlinearity; on the swish machine both slots are degenerate
    lanes (up-row 0, up-bias 1).

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

    in_lin, out_lin = _mlp_in_out(mlp)
    # See the matching assert in _write_compute_ffn: the const-column rows
    # are the bias-fold destination and must never alias a real input row.
    if bias_fold is not None:
        assert bias_fold.const_col not in set(in_idx), (
            f"compute_linear_bypass input for {node!r} reads the pinned "
            f"constant-1 column {bias_fold.const_col}, which bias=False uses "
            f"as the bias-fold row.  Compiler bug."
        )

    target_dtype = in_lin.output_matrix.dtype
    W = node.output_matrix.to(target_dtype)  # (d_input, d_output)

    # Value path — the activation bypass lane-pair (shared with cancel_bypass).
    _write_bypass_lane_pair(
        mlp,
        in_idx,
        out_idx,
        mlp_slots,
        W,
        node=node,
        op_type=op.op_type,
        bias_fold=bias_fold,
        recorder=recorder,
    )

    # Bias path (compute_linear-specific — cancel_bypass has no bias).
    swish = mlp.activation == "swish"
    in_gain = _SWISH_SCALE if swish else 1.0
    pos_slots = mlp_slots[:d_output]
    neg_slots = mlp_slots[d_output:]
    pos_slots_t = torch.as_tensor(pos_slots, dtype=torch.long)
    neg_slots_t = torch.as_tensor(neg_slots, dtype=torch.long)
    out_idx_t = torch.as_tensor(out_idx, dtype=torch.long)

    # Fold deferred biased-Linear inputs into the input-side slot biases
    # (scaled with the gate rows; the up rows are zero, so no up-side fold).
    if biased_linears:
        input_node = node.inputs[0]
        leaves = (
            flatten_concat_nodes([input_node])
            if isinstance(input_node, Concatenate)
            else [input_node]
        )
        bias_dtype = in_lin.output_bias.dtype
        offset = 0
        for leaf in leaves:
            if leaf in biased_linears:
                assert isinstance(leaf, Linear)
                # contrib[j] = in_gain * sum_i W[offset+i, j] * leaf.bias[i]
                contrib = (
                    in_gain * (leaf.output_bias @ W[offset : offset + len(leaf), :])
                ).to(bias_dtype)
                if bias_fold is None:
                    in_lin.output_bias[pos_slots_t] += contrib
                    in_lin.output_bias[neg_slots_t] -= contrib
                else:
                    bias_fold.hidden_bias(
                        in_lin,
                        "mlp.W_in",
                        pos_slots,
                        contrib,
                        node=node,
                        op_type=op.op_type,
                        accumulate=True,
                    )
                    bias_fold.hidden_bias(
                        in_lin,
                        "mlp.W_in",
                        neg_slots,
                        -contrib,
                        node=node,
                        op_type=op.op_type,
                        accumulate=True,
                    )
            offset += len(leaf)

    # Output bias via the output-side projection's bias (or the constant lane).
    # The output-side value rows (±out_gain) and the placement records are
    # emitted by _write_bypass_lane_pair above.
    if bias_fold is None:
        out_lin.output_bias[out_idx_t] += node.output_bias.to(target_dtype)
    else:
        bias_fold.out_bias(
            out_lin,
            out_idx,
            node.output_bias,
            node=node,
            op_type=op.op_type,
            accumulate=True,
        )
