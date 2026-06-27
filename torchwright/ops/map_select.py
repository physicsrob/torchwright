from torchwright.graph import Node, Concatenate, Linear
from torchwright.graph.misc import LiteralValue
from typing import List, Dict
import math
import numbers
import torch

from torchwright.graph.asserts import assert_integer, assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import step_sharpness, embedding_step_sharpness
from torchwright.ops.logic_ops import (
    cond_add_vector,
    cond_gate,
    _max_abs_or_raise,
    _intersect_intervals,
    per_column_offsets,
)


def _select_per_column_offsets(true_node, false_node, scalar_M):
    """Per-output-column gate offsets for a two-way select. The output column
    ``j`` equals ``true_j`` or ``false_j``, so size ``M_j`` from the union of
    the two operands' per-column affine intervals (falls back to the scalar
    ``M`` if a per-column bound is unavailable). See ``per_column_offsets``."""
    ti = _intersect_intervals(true_node)
    fi = _intersect_intervals(false_node)
    if ti is None or fi is None or len(ti) != len(fi):
        return torch.full((len(true_node),), scalar_M)
    union = [Range(min(t.lo, f.lo), max(t.hi, f.hi)) for t, f in zip(ti, fi)]
    return per_column_offsets(union, scalar_M)


def _broadcast_select_per_column_offsets(
    true_value, false_value, n_slots, d_fill, true_bc, false_bc, scalar_M
):
    """Per-output-column gate offsets for broadcast_select. Output column
    ``i*d_fill+j`` equals ``true``/``false`` at its (possibly broadcast)
    source column, so union those two per-column intervals."""
    ti = _intersect_intervals(true_value)
    fi = _intersect_intervals(false_value)
    if ti is None or fi is None:
        return torch.full((n_slots * d_fill,), scalar_M)
    union = []
    for i in range(n_slots):
        for j in range(d_fill):
            ts = j if true_bc else i * d_fill + j
            fs = j if false_bc else i * d_fill + j
            union.append(Range(min(ti[ts].lo, fi[fs].lo), max(ti[ts].hi, fi[fs].hi)))
    return per_column_offsets(union, scalar_M)


from torchwright.ops.arithmetic_ops import sum_nodes
from torchwright.ops.linear_relu_linear import linear_relu_linear


def map_to_table(
    inp: Node, key_to_value: Dict[torch.Tensor, torch.Tensor], default: torch.Tensor
) -> Node:
    """
    Maps the value of the input node to a lookup table.

    Args:
        inp (Node): Node whose values will be looked up.
        key_to_value (Dict[torch.Tensor, torch.Tensor]): Lookup table mapping from keys to values.
        default (torch.Tensor): Default tensor to return if the input value doesn't exist in the table.

    Returns:
        Node: Output node with mapped values.
    """
    d_keys = {len(x) for x in key_to_value.keys()}
    d_values = {len(x) for x in key_to_value.values()}
    assert len(d_keys) == 1
    assert len(d_values) == 1
    d_key = d_keys.pop()
    d_value = d_values.pop()
    assert len(inp) == d_key
    assert len(default) == d_value

    d_hidden = len(key_to_value)
    speed = embedding_step_sharpness
    # We'll use 1 MLP entry per item in the table, and an overall output bias of the default value
    # So roughly speaking:
    # input_proj will be (d_hidden x d_key), where input_proj[i, :] = table.keys()[i]
    # input_bias will be (d_hidden), where input_bias[i] = 1.0/speed - (table.keys()[i] @ table.keys()[i])
    # output_proj will be (d_hidden, d_value), where output_proj[i, :] = speed * (table.values()[i] - default)
    # output_bias will be (d_value), equal to default

    input_proj = torch.zeros(d_hidden, d_key)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, d_value)

    for i, (key, value) in enumerate(key_to_value.items()):
        input_proj[i, :] = key
        input_bias[i] = 1.0 / speed - (key @ key)
        output_proj[i, :] = speed * (value - default)

    result = linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=default,
    )

    # Output = default + sum_i ReLU_i * speed * (value_i - default).
    # When multiple keys overlap, multiple ReLU_i can fire at once, so
    # bound per channel by |default[j]| + sum_i |value_i[j] - default[j]|.
    # Wider than [min, max] over the table (tight only when keys are
    # cleanly separated) but covers the overlapping-key case. Without
    # this claim, Linear's pessimistic interval arithmetic on the wide
    # MLP blows up after a few chained map_to_tables.
    diff_abs_sum = torch.zeros(d_value)
    for value in key_to_value.values():
        diff_abs_sum = diff_abs_sum + (value - default).abs()
    lo = float((default - diff_abs_sum).min().item())
    hi = float((default + diff_abs_sum).max().item())
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=Range(lo, hi)),
    )


def _lookup_axis_scale(index_scale, axis: int, n_axes: int = 2) -> float:
    if isinstance(index_scale, numbers.Real):
        scale = float(index_scale)
    else:
        if len(index_scale) != n_axes:
            raise ValueError(
                f"index_scale must be a scalar or length-{n_axes} tuple, "
                f"got {index_scale!r}"
            )
        scale = float(index_scale[axis])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"index_scale values must be finite and > 0, got {scale}")
    return scale


def _scale_lookup_index(inp: Node, scale: float, name: str) -> Node:
    if scale == 1.0:
        return inp
    return Linear(inp, torch.tensor([[scale]]), name=name)


def _constant_vector(values: torch.Tensor, name: str) -> Node:
    return LiteralValue(values.to(dtype=torch.float32), name=name)


def _lookup_numeric_slack(max_abs: float, sharpness: float, n_steps: int) -> float:
    # Per-element slack for the row-vector / staircase output-range *guards*
    # (assert_matches_value_type), not the correctness path. The guard must
    # have margin above accumulated fp32 noise in the wide PWL AND above GPU
    # cross-test FP variation (cuBLAS algorithm selection / TF32), which the
    # noise notes peg at ~1e-5..1e-6 — an order of magnitude above the fp32
    # single-run unit. At the 16x128x128 target (rows = A*B = 2048) the hidden
    # PWL activations reach ~20k*sharpness, so reduced-precision matmul on A100
    # can push the guarded value to ~max_abs*sharpness*rows*1e-5; a 1e-6 budget
    # tripped intermittently in the full sharded suite (passes in isolation and
    # under fp32). Using 1e-5 gives the guard headroom for GPU variation without
    # loosening real correctness (the caller's value-match test stays tight).
    return max(1e-3, max_abs * sharpness * max(n_steps, 1) * 1e-5)


def _saturating_step_select(
    index: Node,
    *,
    n: int,
    top_value: torch.Tensor,
    deltas: torch.Tensor,
    sharpness: float,
    d_max: int,
    name: str,
) -> Node:
    """Cancellation-free integer-index selection from an ``n``-row table.

    Returns ``value[clamp(round(index), 0, n-1)]`` as a width-``d`` vector,
    where the rows are reconstructed from their consecutive differences:
    ``value[n-1] == top_value`` and ``deltas[k-1] == value[k] - value[k-1]``
    (so ``value[k] = top_value - sum_{j>k} deltas[j-1]``).

    Built as ``floor_int``'s saturating-step staircase rather than the single
    projection ``value[0] + sum_i delta_i * relu(s*(x - b_i))`` the old
    ``piecewise_linear`` builders used.  That old form summed ~``n`` ReLU terms
    of magnitude ~``s*x`` into one accumulator, so for a large scaled index
    (out-of-range ``x``) or a tall table (large ``n``) the partial sums
    overflowed float32's 2^24 exact-integer limit and the alternating-sign
    terms cancelled into garbage (e.g. an in-range index in a 20000-row table
    collapsed to 0).  Here:

    * A leading ``min(index, n-1)`` clamp bounds the out-of-range case so the
      step width ``W`` (sized for the in-range span ``s*(n-1)``) always
      resolves the saturating difference.  The lower edge needs no clamp — a
      hugely negative index leaves every step off (``relu`` of a negative is
      exactly 0), so it selects row 0 exactly.
    * Each boundary ``k - 0.5`` (``k = 1..n-1``) contributes a *bounded* step
      ``step_k = relu(t_k) - relu(t_k - W)`` with ``t_k = s*(x - (k-0.5)) +
      0.5``, then the output accumulates ``relu(1 - step_k)`` (the "row not yet
      reached" indicator, in ``[0, 1]``) weighted by the row difference:
      ``out = top_value - sum_k relu(1 - step_k) * deltas[k-1]``.  Every partial
      sum is bounded by the table's total variation ``sum_k |deltas[k-1]|``,
      not by ``s*x`` — that is what removes the cancellation.

    The ``+ 0.5`` centers the ramp on the boundary (indicator ``= 0.5`` at the
    boundary), reproducing the two-row linear blend the old ``piecewise_linear``
    form produced in the transition band, so off-grid values are unchanged.

    Cost vs the old single sublayer: one ``min`` sublayer plus the two chained
    ReLU sublayers of the staircase.
    """
    d = int(top_value.shape[0])
    s = float(sharpness)
    top_idx = float(n - 1)

    # Upper-clamp the index to n-1: (n-1) - relu((n-1) - x).  One sublayer,
    # cancellation-free (a single ReLU of a magnitude, not a sum of them).
    x_clamped = linear_relu_linear(
        input_node=index,
        input_proj=torch.tensor([[-1.0]]),
        input_bias=torch.tensor([top_idx]),
        output_proj=torch.tensor([[-1.0]]),
        output_bias=torch.tensor([top_idx]),
        name=f"{name}_clamp_hi",
    )

    # W: the saturating-step cap.  relu(t) - relu(t - W) collapses to 0 once
    # ulp(t) >= W, so size W a few ulp above the largest |t| (= s*(n-1)+1),
    # matching floor_int.  W = 2 for the common small range; larger only as
    # s*(n-1) approaches 2^24.
    max_t = s * float(n - 1) + 1.0
    ulp = 2.0 ** (math.floor(math.log2(max_t)) - 23) if max_t >= 1.0 else 2.0**-23
    step_cap = max(2.0, 8.0 * ulp)  # W

    _CHUNK = max(1, d_max // 2)  # cap stage-1 hidden width (2 ReLUs per step)
    partials: list[Node] = []
    for c0 in range(1, n, _CHUNK):
        ks = list(range(c0, min(c0 + _CHUNK, n)))  # boundary indices k = 1..n-1
        c = len(ks)
        # stage 1 (MLP sublayer): step_k = relu(t_k) - relu(t_k - W),
        #   t_k = s*x - s*(k - 0.5) + 0.5.   hidden width 2c, output width c.
        in_proj = torch.full((2 * c, 1), s)
        in_bias = torch.empty(2 * c)
        out_proj = torch.zeros((2 * c, c))
        for j, k in enumerate(ks):
            t_bias = 0.5 - s * (float(k) - 0.5)
            in_bias[2 * j] = t_bias  # t_k
            in_bias[2 * j + 1] = t_bias - step_cap  # t_k - W
            out_proj[2 * j, j] = 1.0
            out_proj[2 * j + 1, j] = -1.0
        step = linear_relu_linear(
            input_node=x_clamped,
            input_proj=in_proj,
            input_bias=in_bias,
            output_proj=out_proj,
            output_bias=torch.zeros(c),
            name=f"{name}_step",
        )
        # stage 2 (MLP sublayer): out += output_bias - sum_k relu(1 - step_k) *
        #   deltas[k-1].   hidden = relu(1 - step_k) (width c), output width d.
        chunk_out_proj = torch.zeros((c, d))
        for j, k in enumerate(ks):
            chunk_out_proj[j, :] = -deltas[k - 1]
        # top_value rides on the first chunk's output bias; the chunks sum to
        # top_value - sum_k relu(1 - step_k) * deltas[k-1].
        out_bias = top_value.clone() if c0 == 1 else torch.zeros(d)
        partials.append(
            linear_relu_linear(
                input_node=step,
                input_proj=-torch.eye(c),
                input_bias=torch.ones(c),
                output_proj=chunk_out_proj,
                output_bias=out_bias,
                name=f"{name}_saturate",
            )
        )

    return partials[0] if len(partials) == 1 else sum_nodes(partials)


def _table_lookup_row_vector(
    index: Node,
    table: torch.Tensor,
    *,
    sharpness: float,
    d_max: int,
    name: str,
) -> Node:
    rows, cols = table.shape
    if rows == 1:
        return _constant_vector(table[0], name=f"{name}_constant_row")

    # value[k] = table[k]; deltas[k-1] = table[k] - table[k-1]; top = table[-1].
    result = _saturating_step_select(
        index,
        n=rows,
        top_value=table[rows - 1].clone(),
        deltas=table[1:] - table[:-1],
        sharpness=sharpness,
        d_max=d_max,
        name=f"{name}_row",
    )
    max_abs = float(table.abs().max().item())
    return assert_matches_value_type(
        result,
        NodeValueType(
            value_range=Range(float(table.min().item()), float(table.max().item()))
        ),
        atol=_lookup_numeric_slack(max_abs, sharpness, rows),
    )


def _table_lookup_column_mask(
    index: Node,
    n_cols: int,
    *,
    sharpness: float,
    d_max: int,
    name: str,
) -> Node:
    if n_cols == 1:
        return _constant_vector(torch.tensor([1.0]), name=f"{name}_constant_mask")

    # mask[k] = +1 at column k, -1 elsewhere.  top = mask[n_cols-1]; crossing
    # boundary k flips column k-1 (+1 -> -1, delta -2) and column k (-1 -> +1,
    # delta +2).
    top_value = -torch.ones(n_cols)
    top_value[n_cols - 1] = 1.0
    deltas = torch.zeros((n_cols - 1, n_cols))
    for k in range(1, n_cols):
        deltas[k - 1, k - 1] = -2.0
        deltas[k - 1, k] = 2.0
    return _saturating_step_select(
        index,
        n=n_cols,
        top_value=top_value,
        deltas=deltas,
        sharpness=sharpness,
        d_max=d_max,
        name=f"{name}_mask",
    )


def table_lookup_2d(
    i: Node,
    j: Node,
    table,
    *,
    index_scale=1.0,
    sharpness: float = 100.0,
    d_max: int = 1024,
    name: str = "table_lookup_2d",
) -> Node:
    """Lookup a scalar from a compile-time constant 2D table.

    Computes an integer-index lookup: scaled inputs near integer ``k``
    select table index ``k``, with out-of-range scaled indices clamped to
    the table edge.  The discontinuous lookup is represented by narrow
    ReLU ramps with width ``eps = 1 / sharpness`` centered at
    half-integer boundaries ``k + 0.5``.

    The implementation builds a selected row vector from ``i`` and then
    gates the selected column from ``j``.  Inside a ramp, the output is
    the bounded local handoff induced by the row ramp and saturating
    column gate: exact outside ramps and at integer indices, defined and
    local inside ramps, but not a bilinear interpolation guarantee.
    """
    assert len(i) == 1, "i must be a 1D scalar node"
    assert len(j) == 1, "j must be a 1D scalar node"
    s = float(sharpness)
    if not math.isfinite(s) or s <= 1.0:
        raise ValueError(f"sharpness must be finite and > 1, got {s}")
    if d_max < 2:
        raise ValueError(f"d_max must be >= 2, got {d_max}")

    table_t = torch.as_tensor(table, dtype=torch.float32)
    if table_t.ndim != 2:
        raise ValueError(f"table must be 2D, got shape {tuple(table_t.shape)}")
    rows, cols = table_t.shape
    if rows < 1 or cols < 1:
        raise ValueError(f"table must be non-empty, got shape {tuple(table_t.shape)}")
    if not torch.isfinite(table_t).all():
        raise ValueError("table must contain only finite values")

    scale_i = _lookup_axis_scale(index_scale, 0)
    scale_j = _lookup_axis_scale(index_scale, 1)
    i_scaled = _scale_lookup_index(i, scale_i, name=f"{name}_scale_i")
    j_scaled = _scale_lookup_index(j, scale_j, name=f"{name}_scale_j")

    row = _table_lookup_row_vector(
        i_scaled,
        table_t,
        sharpness=s,
        d_max=d_max,
        name=name,
    )
    if cols == 1:
        return row

    mask = _table_lookup_column_mask(
        j_scaled,
        cols,
        sharpness=s,
        d_max=d_max,
        name=name,
    )

    max_abs = float(table_t.abs().max().item())
    row_slack = _lookup_numeric_slack(max_abs, s, rows)
    offset = max_abs + row_slack + 1.0
    x = Concatenate([mask, row])
    cols_per_chunk = max(1, d_max // 2)
    chunks: list[Node] = []
    for start in range(0, cols, cols_per_chunk):
        end = min(cols, start + cols_per_chunk)
        chunk_cols = end - start
        d_hidden = 2 * chunk_cols
        input_proj = torch.zeros(d_hidden, 2 * cols)
        input_bias = torch.zeros(d_hidden)
        output_proj = torch.zeros(d_hidden, 1)
        output_bias = torch.zeros(1)

        for local_c, c in enumerate(range(start, end)):
            pos_value = 2 * local_c
            neg_value = pos_value + 1
            input_proj[pos_value, c] = offset
            input_proj[pos_value, cols + c] = 1.0
            input_proj[neg_value, c] = offset
            input_proj[neg_value, cols + c] = -1.0
            output_proj[pos_value, 0] = 0.5
            output_proj[neg_value, 0] = -0.5

        chunk_name = f"{name}_gate_{start}_{end}" if cols > cols_per_chunk else name
        chunks.append(
            linear_relu_linear(
                input_node=x,
                input_proj=input_proj,
                input_bias=input_bias,
                output_proj=output_proj,
                output_bias=output_bias,
                name=chunk_name,
            )
        )

    result = chunks[0] if len(chunks) == 1 else sum_nodes(chunks)
    lo = float(table_t.min().item())
    hi = float(table_t.max().item())
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=Range(lo, hi)),
        atol=max(1e-3, offset * 0.005, row_slack),
    )


def _table_lookup_index_staircase(
    index: Node,
    n: int,
    *,
    sharpness: float,
    d_max: int,
    name: str,
) -> Node:
    """Scalar centered integer-index staircase: scaled ``index`` -> clamped
    nearest-integer index, flat in stable bins with narrow ramps at the
    half-integer boundaries.  The scalar analogue of the row-vector lookup;
    width-1 output."""
    if n == 1:
        return _constant_vector(torch.tensor([0.0]), name=f"{name}_zero_index")

    # value[k] = k; deltas[k-1] = 1; top = n-1.
    result = _saturating_step_select(
        index,
        n=n,
        top_value=torch.tensor([float(n - 1)]),
        deltas=torch.ones((n - 1, 1)),
        sharpness=sharpness,
        d_max=d_max,
        name=f"{name}_staircase",
    )
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=Range(0.0, float(n - 1))),
        atol=_lookup_numeric_slack(float(n - 1), sharpness, n),
    )


def table_lookup_3d(
    i: Node,
    j: Node,
    k: Node,
    table,
    *,
    index_scale=1.0,
    sharpness: float = 100.0,
    outer_axis=None,
    d_max: int = 1024,
    name: str = "table_lookup_3d",
) -> Node:
    """Lookup a scalar from a compile-time constant 3D table.

    A ``table_lookup_2d`` whose row index is a flattened, pre-rounded
    ``(A, B)`` index: the two flattened axes are rounded to integer indices
    ``idx_a, idx_b`` and combined as ``q = B * idx_a + idx_b`` into the row
    axis of ``table.reshape(A * B, C)``; the C axis is the 2D column gate.

    Semantics match ``table_lookup_2d`` per axis at integer inputs.  Off the
    integer grid the three axes differ: the vector axis ``C`` is the
    saturating-gate handoff, the inner flattened axis ``B`` is a two-neighbor
    linear blend, and the outer flattened axis ``A`` is degenerate (its
    transition sweeps a whole block of ``B`` rows).  ``A`` is therefore
    asserted integer-valued.

    Internal axis order is fixed by a heuristic: ``A`` = ``outer_axis``
    (defaults to the smallest axis, asserted integer), ``C`` = the larger of
    the remaining two (the contiguous vector axis), ``B`` = the smaller
    remaining axis.  ``index_scale`` may be a scalar or length-3 tuple mapping
    to ``(i, j, k)``.
    """
    inputs = [i, j, k]
    for axis, inp in enumerate(inputs):
        if len(inp) != 1:
            raise ValueError(f"input for axis {axis} must be a 1D scalar node")
    s = float(sharpness)
    if not math.isfinite(s) or s <= 1.0:
        raise ValueError(f"sharpness must be finite and > 1, got {s}")
    if d_max < 2:
        raise ValueError(f"d_max must be >= 2, got {d_max}")

    table_t = torch.as_tensor(table, dtype=torch.float32)
    if table_t.ndim != 3:
        raise ValueError(f"table must be 3D, got shape {tuple(table_t.shape)}")
    if any(d < 1 for d in table_t.shape):
        raise ValueError(f"table must be non-empty, got shape {tuple(table_t.shape)}")
    if not torch.isfinite(table_t).all():
        raise ValueError("table must contain only finite values")

    sizes = list(table_t.shape)
    if outer_axis is None:
        a_axis = int(min(range(3), key=lambda ax: sizes[ax]))
    else:
        a_axis = int(outer_axis)
        if a_axis not in (0, 1, 2):
            raise ValueError(f"outer_axis must be 0, 1, or 2, got {outer_axis!r}")
    # C is the larger of the two remaining axes (kept contiguous as the
    # vector axis); B is the smaller, so A*B stays small.
    remaining = sorted(
        (ax for ax in range(3) if ax != a_axis), key=lambda ax: sizes[ax]
    )
    b_axis, c_axis = remaining[0], remaining[1]

    a_size, b_size, c_size = sizes[a_axis], sizes[b_axis], sizes[c_axis]

    a_scaled = _scale_lookup_index(
        inputs[a_axis],
        _lookup_axis_scale(index_scale, a_axis, n_axes=3),
        name=f"{name}_scale_a",
    )
    b_scaled = _scale_lookup_index(
        inputs[b_axis],
        _lookup_axis_scale(index_scale, b_axis, n_axes=3),
        name=f"{name}_scale_b",
    )
    c_scaled = _scale_lookup_index(
        inputs[c_axis],
        _lookup_axis_scale(index_scale, c_axis, n_axes=3),
        name=f"{name}_scale_c",
    )

    # The outer axis has no graceful off-grid handoff, so require it integer.
    a_scaled = assert_integer(a_scaled)

    idx_a = _table_lookup_index_staircase(
        a_scaled, a_size, sharpness=s, d_max=d_max, name=f"{name}_a"
    )
    idx_b = _table_lookup_index_staircase(
        b_scaled, b_size, sharpness=s, d_max=d_max, name=f"{name}_b"
    )
    q = Linear(
        Concatenate([idx_a, idx_b]),
        torch.tensor([[float(b_size)], [1.0]]),
        name=f"{name}_flatten",
    )

    table_2d = (
        table_t.permute(a_axis, b_axis, c_axis)
        .contiguous()
        .reshape(a_size * b_size, c_size)
    )
    return table_lookup_2d(
        q,
        c_scaled,
        table_2d,
        index_scale=1.0,
        sharpness=s,
        d_max=d_max,
        name=f"{name}_2d",
    )


def switch(conditions: List[Node], values: List[Node]) -> Node:
    """
    Select one of N values based on which condition is true.

    Assumes exactly one condition is true (1.0), rest are false (-1.0).

    Args:
        conditions (List[Node]): Boolean condition nodes (each length 1).
        values (List[Node]): Value nodes (all same length).

    Returns:
        Node: The value whose corresponding condition is true.
    """
    return sum_nodes([cond_gate(c, v) for c, v in zip(conditions, values)])


def _select_output_type(
    cond: Node,
    true_node: Node,
    false_node: Node,
) -> NodeValueType:
    tv = true_node.value_type
    fv = false_node.value_type
    r = tv.value_range.union(fv.value_range)
    if not r.is_finite():
        return NodeValueType()
    return NodeValueType(value_range=r)


def _select_offset(true_node: Node, false_node: Node, caller: str) -> float:
    union_range = true_node.value_type.value_range.union(
        false_node.value_type.value_range
    )
    union_vt = NodeValueType(value_range=union_range)
    return _max_abs_or_raise(union_vt, caller)


def select(
    cond: Node,
    true_node: Node,
    false_node: Node,
    *,
    approximate: bool = True,
    c_tol: float = 0.005,
) -> Node:
    """
    Outputs one of two nodes based on a boolean condition.

    Args:
        cond (Node): Condition node that outputs either true or false.
        true_node (Node): Node to be outputted if the condition is true.
        false_node (Node): Node to be outputted if the condition is false.
        approximate: When ``True`` (default), uses a single L→ReLU→L sublayer
            with an additive cancellation trick. Both branches compute
            ``(M + v) − M`` where ``M`` is derived from the union of
            ``true_node`` / ``false_node`` ranges; this loses precision for
            ``|v| ≪ ULP(M)``. When ``False``, uses two sublayers: the first
            maps ``cond`` to ``c_on = ReLU(cond)`` and ``c_off = ReLU(−cond)``;
            the second gates each branch with ReLU clipping (no cancellation).
            The winning branch is float-exact and immune to cond noise; costs
            one extra MLP sublayer.
        c_tol: Maximum acceptable deviation of ``|cond|`` from 1. When
            ``approximate=True``, an Assert checks ``||cond| - 1| <= c_tol``
            and the output's semantic bound is widened by ``c_tol * M`` to
            account for the amplified condition noise. Default 0.005
            matches typical far-field compare noise.

    Returns:
        Node: Either true_node or false_node based on the condition.
    """
    from torchwright.ops.const import step_sharpness

    assert len(cond) == 1
    assert len(true_node) == len(false_node)

    d = len(true_node)
    M = _select_offset(true_node, false_node, "select")

    if approximate:
        from torchwright.graph.misc import Assert

        def _cond_check(x: torch.Tensor) -> tuple:
            deviation = (x.abs() - 1.0).abs()
            bad = deviation > c_tol
            if not bad.any():
                return True, ""
            from torchwright.graph.asserts import _format_bad

            return False, (f"expected ||cond| - 1| <= {c_tol}; {_format_bad(x, bad)}")

        cond = Assert(
            cond,
            _cond_check,
            message=f"cond near ±1 (c_tol={c_tol})",
            claimed_type=NodeValueType(value_range=Range(-1.0 - c_tol, 1.0 + c_tol)),
        )
        # Per-column offsets sized from the union of the two operands' columns
        # (see per_column_offsets) so a narrow column is not forced to a wide
        # sibling's M in the `(M + v) - M` cancellation.
        M_cols = _select_per_column_offsets(true_node, false_node, M)

        d_hidden = 2 * d
        input_proj = torch.zeros(d_hidden, 1 + 2 * d)
        input_bias = torch.zeros(d_hidden)
        output_proj = torch.zeros(d_hidden, d)
        output_bias = -M_cols.clone()

        for j in range(d):
            a = j
            b = d + j
            input_proj[a, 0] = M_cols[j]
            input_proj[a, 1 + j] = 1.0
            input_proj[b, 0] = -M_cols[j]
            input_proj[b, 1 + d + j] = 1.0
            output_proj[a, j] = 1.0
            output_proj[b, j] = 1.0

        x = Concatenate([cond, true_node, false_node])
        result = linear_relu_linear(
            input_node=x,
            input_proj=input_proj,
            input_bias=input_bias,
            output_proj=output_proj,
            output_bias=output_bias,
            name="select",
        )
    else:
        cond_gates = linear_relu_linear(
            input_node=cond,
            input_proj=torch.tensor([[-1.0], [1.0]]),
            input_bias=torch.tensor([0.0, 0.0]),
            output_proj=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            output_bias=torch.tensor([0.0, 0.0]),
            name="select_cond_gates",
        )
        d_hidden = 4 * d
        d_input = 2 + 2 * d
        input_proj = torch.zeros(d_hidden, d_input)
        input_bias = torch.zeros(d_hidden)
        output_proj = torch.zeros(d_hidden, d)
        output_bias = torch.zeros(d)

        for j in range(d):
            t_pos = j
            t_neg = d + j
            f_pos = 2 * d + j
            f_neg = 3 * d + j
            input_proj[t_pos, 0] = -M
            input_proj[t_pos, 2 + j] = 1.0
            input_proj[t_neg, 0] = -M
            input_proj[t_neg, 2 + j] = -1.0
            input_proj[f_pos, 1] = -M
            input_proj[f_pos, 2 + d + j] = 1.0
            input_proj[f_neg, 1] = -M
            input_proj[f_neg, 2 + d + j] = -1.0
            output_proj[t_pos, j] = 1.0
            output_proj[t_neg, j] = -1.0
            output_proj[f_pos, j] = 1.0
            output_proj[f_neg, j] = -1.0

        x = Concatenate([cond_gates, true_node, false_node])
        result = linear_relu_linear(
            input_node=x,
            input_proj=input_proj,
            input_bias=input_bias,
            output_proj=output_proj,
            output_bias=output_bias,
            name="select",
        )

    vt = _select_output_type(cond, true_node, false_node)
    if vt != NodeValueType.unknown():
        gate_atol = M * c_tol if approximate else 1e-3
        result = assert_matches_value_type(result, vt, atol=gate_atol)
    tolerance = c_tol * M if approximate else 0.0
    from torchwright.graph.affine_rules import (
        _apply_semantic_override,
        _select_semantic_bound,
    )

    _apply_semantic_override(
        result,
        _select_semantic_bound(
            true_node._affine_bound, false_node._affine_bound, tolerance=tolerance
        ),
    )
    return result


def in_range(lower: Node, upper: Node, n_slots: int) -> Node:
    """Test each integer position against a runtime interval.

    For each position i in {0, 1, ..., n_slots-1}, returns 1.0 (true)
    if lower <= i + 0.5 < upper, or -1.0 (false) otherwise.

    The +0.5 offset means position i is "in range" when the interval
    covers the center of the integer bin.

    Args:
        lower: Scalar node, lower bound of the interval.
        upper: Scalar node, upper bound of the interval.
        n_slots: Number of integer positions to test (0 through n_slots-1).

    Returns:
        Node of width n_slots, each value 1.0 (in range) or -1.0 (out of range).
    """
    assert len(lower) == 1
    assert len(upper) == 1

    S = step_sharpness
    d_hidden = 4 * n_slots  # 4 neurons per position

    # Input is [lower, upper], d_input=2
    inp = Concatenate([lower, upper])

    input_proj = torch.zeros(d_hidden, 2)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, n_slots)
    output_bias = torch.full((n_slots,), -1.0)

    for i in range(n_slots):
        center = i + 0.5
        base = 4 * i

        # step_past_lower: step(center - lower) using 2 neurons
        # Unit 0: ReLU(S*center - S*lower) = ReLU(-S*lower + S*center)
        input_proj[base, 0] = -S  # reads lower
        input_bias[base] = S * center
        # Unit 1: ReLU(S*center - S*lower - 1) = ReLU(-S*lower + S*center - 1)
        input_proj[base + 1, 0] = -S
        input_bias[base + 1] = S * center - 1.0

        # step_past_upper: step(center - upper) using 2 neurons
        # Unit 2: ReLU(S*center - S*upper)
        input_proj[base + 2, 1] = -S  # reads upper
        input_bias[base + 2] = S * center
        # Unit 3: ReLU(S*center - S*upper - 1)
        input_proj[base + 3, 1] = -S
        input_bias[base + 3] = S * center - 1.0

        # output_i = 2*(step_past_lower - step_past_upper) - 1
        # step_past_lower = unit0 - unit1, step_past_upper = unit2 - unit3
        # output_i = 2*((u0 - u1) - (u2 - u3)) - 1
        #          = 2*u0 - 2*u1 - 2*u2 + 2*u3 - 1
        output_proj[base, i] = 2.0
        output_proj[base + 1, i] = -2.0
        output_proj[base + 2, i] = -2.0
        output_proj[base + 3, i] = 2.0
        # output_bias[i] = -1.0 (already set)

    result = linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
        name="in_range",
    )
    return assert_matches_value_type(
        result, NodeValueType(value_range=Range(-1.0, 1.0))
    )


def dynamic_extract(
    table: Node,
    idx: Node,
    n_entries: int,
    d_fill: int,
) -> Node:
    """Read a ``d_fill``-wide slice from a runtime-valued table at a runtime index.

    Given ``table`` of width ``n_entries * d_fill``, laid out slot-major
    so entry ``i`` occupies columns ``[i*d_fill, (i+1)*d_fill)``, and a
    scalar ``idx`` carrying an integer in ``[0, n_entries - 1]``,
    returns the ``d_fill``-wide slice
    ``table[idx*d_fill : (idx+1)*d_fill]``.

    This is the missing resampling primitive: torchwright has
    :func:`map_to_table` for *compile-time constant* tables and
    :func:`broadcast_select` for *runtime masks over runtime values*,
    but nothing that directly implements "index a runtime table at a
    runtime scalar".  The composition is small —

    1. ``in_range(idx, idx + 1, n_entries)`` emits a width-``n_entries``
       mask with ``+1`` at slot ``floor(idx)`` and ``-1`` elsewhere.
    2. ``broadcast_select(mask, table, zero_d_fill, n_entries, d_fill)``
       keeps the selected slot's ``d_fill`` values and zeros out the
       rest, producing a width-``n_entries*d_fill`` intermediate.
    3. A free ``Linear`` sums across slots to collapse the intermediate
       down to a ``d_fill``-wide output.

    — but pulling it out into its own op lets callers say "extract row
    ``idx`` from the table" instead of hand-assembling masks and
    worrying about whether ``broadcast_select``'s broadcasting rules
    match their layout.  Most of the recent DOOM fill bugs came from
    ad-hoc hand-assembled versions of this exact pattern.

    Contract:

    * ``idx`` must carry an integer in ``[0, n_entries - 1]``.  Exact
      integer values select cleanly; off-integer inputs round toward
      the nearest slot with the boundary at ``k + 0.5``.  Out-of-range
      inputs produce an all-zero output.  Callers who need clamp
      semantics should clamp before calling (one extra MLP sublayer).
    * ``table`` is read once per forward pass — the mask is applied at
      build time to the same runtime node, not recomputed per row.

    Cost: two MLP sublayers (the ``in_range`` and ``broadcast_select``)
    plus one free ``Linear``.  Hidden-width use is
    ``4*n_entries + 2*n_entries*d_fill`` neurons.

    Args:
        table: Runtime node of width ``n_entries * d_fill``.
        idx: Scalar node carrying an integer in ``[0, n_entries - 1]``.
        n_entries: Number of logical entries in ``table`` (compile-time).
        d_fill: Width of each entry (compile-time).

    Returns:
        A ``d_fill``-wide node carrying the selected entry.
    """
    from torchwright.graph import Linear
    from torchwright.graph.misc import LiteralValue
    from torchwright.ops.arithmetic_ops import add_const

    assert len(idx) == 1, "idx must be a 1D scalar node"
    assert len(table) == n_entries * d_fill, (
        f"table has width {len(table)}; expected n_entries*d_fill = "
        f"{n_entries * d_fill}"
    )
    assert n_entries >= 1, "n_entries must be at least 1"
    assert d_fill >= 1, "d_fill must be at least 1"

    # Step 1: one-hot mask over n_entries.  in_range(idx, idx+1, n) fires
    # at the single slot whose center is in [idx, idx+1) — that slot is
    # floor(idx) for integer idx and the nearest slot under rounding
    # otherwise.
    idx_plus_one = add_const(idx, 1.0)
    one_hot = in_range(idx, idx_plus_one, n_entries)

    # Step 2: zero out every slot except the selected one.  The output
    # is width n_entries * d_fill with zeros at every slot the mask
    # marks as -1.
    zero_d_fill = LiteralValue(
        torch.zeros(d_fill),
        name="dynamic_extract_zero",
    )
    masked = broadcast_select(
        masks=one_hot,
        true_value=table,
        false_value=zero_d_fill,
        n_slots=n_entries,
        d_fill=d_fill,
    )

    # Step 3: collapse n_entries slots down to d_fill via a sparse free
    # Linear.  Because exactly one slot is non-zero (the selected one),
    # the "sum across slots" degenerates to "copy the selected slot" —
    # no arithmetic error, even under the ReLU-approximation wiggle of
    # the mask at its boundaries.
    sum_matrix = torch.zeros(n_entries * d_fill, d_fill)
    for slot in range(n_entries):
        for c in range(d_fill):
            sum_matrix[slot * d_fill + c, c] = 1.0
    return Linear(masked, sum_matrix, name="dynamic_extract_sum")


def broadcast_select(
    masks: Node,
    true_value: Node,
    false_value: Node,
    n_slots: int,
    d_fill: int,
    *,
    approximate: bool = True,
    c_tol: float = 0.005,
) -> Node:
    """Select between two values at each of N slots, based on per-slot masks.

    This is a vectorized version of select(). Each slot independently
    picks true_value or false_value based on its mask. Values can be
    broadcast (same for all slots) or per-slot (different per slot).

    Args:
        masks: Node of width n_slots. Each value is 1.0 (true) or -1.0
            (false). Fractional values are safe but produce a smooth
            blend of ``true`` and ``false`` bounded by ``O(max|v|)``;
            no catastrophic sentinel leaks.
        true_value: Node of width d_fill (broadcast to all slots)
            or n_slots*d_fill (per-slot values).
        false_value: Same shape options as true_value.
        n_slots: Number of slots.
        d_fill: Width of the value at each slot.
        approximate: When ``True`` (default), uses a single L→ReLU→L
            sublayer with four units per ``(slot, channel)`` that cancel
            the mask-offset carry per-unit (no output bias). Offset is
            ``M = max|true ∪ false|`` derived from value ranges. When
            ``False``, uses two sublayers: sublayer 1 produces
            ``c_off[i] = ReLU(-mask_i)`` and ``c_on[i] = ReLU(mask_i)``;
            sublayer 2 gates each branch by ReLU clipping against
            ``M·c_on`` / ``M·c_off`` — no cancellation, winning branch
            is float-exact at mask=±1.
        c_tol: Maximum acceptable deviation of ``|masks[i]|`` from 1.
            When ``approximate=True``, an Assert checks
            ``||masks| - 1| <= c_tol`` per element and the output's
            semantic bound is widened by ``c_tol * M``. Default 0.005.

    Returns:
        Node of width n_slots * d_fill.
    """
    assert len(masks) == n_slots
    true_is_broadcast = len(true_value) == d_fill
    false_is_broadcast = len(false_value) == d_fill
    assert true_is_broadcast or len(true_value) == n_slots * d_fill
    assert false_is_broadcast or len(false_value) == n_slots * d_fill

    M = _select_offset(true_value, false_value, "broadcast_select")

    if approximate:
        # Per-output-column offsets (see per_column_offsets) so a narrow column
        # is not forced to a wide sibling's M in the `(M + v) - M` cancellation.
        M_cols = _broadcast_select_per_column_offsets(
            true_value,
            false_value,
            n_slots,
            d_fill,
            true_is_broadcast,
            false_is_broadcast,
            M,
        )

        d_hidden = 4 * n_slots * d_fill
        inp = Concatenate([masks, true_value, false_value])
        d_input = len(inp)

        # Offsets into the concatenated input
        mask_offset = 0
        true_offset = n_slots
        false_offset = n_slots + len(true_value)

        input_proj = torch.zeros(d_hidden, d_input)
        input_bias = torch.zeros(d_hidden)
        output_proj = torch.zeros(d_hidden, n_slots * d_fill)
        # Output bias is zero — every M offset is cancelled locally by
        # the matching ``_b`` carrier unit.
        output_bias = torch.zeros(n_slots * d_fill)

        for i in range(n_slots):
            for j in range(d_fill):
                out_idx = i * d_fill + j
                M_o = M_cols[out_idx]
                unit_pos_t = 4 * out_idx
                unit_pos_b = 4 * out_idx + 1
                unit_neg_t = 4 * out_idx + 2
                unit_neg_b = 4 * out_idx + 3

                # unit_pos_t = ReLU(M * mask_i + true_ij)
                input_proj[unit_pos_t, mask_offset + i] = M_o
                if true_is_broadcast:
                    input_proj[unit_pos_t, true_offset + j] = 1.0
                else:
                    input_proj[unit_pos_t, true_offset + i * d_fill + j] = 1.0

                # unit_pos_b = ReLU(M * mask_i)
                input_proj[unit_pos_b, mask_offset + i] = M_o

                # unit_neg_t = ReLU(-M * mask_i + false_ij)
                input_proj[unit_neg_t, mask_offset + i] = -M_o
                if false_is_broadcast:
                    input_proj[unit_neg_t, false_offset + j] = 1.0
                else:
                    input_proj[unit_neg_t, false_offset + i * d_fill + j] = 1.0

                # unit_neg_b = ReLU(-M * mask_i)
                input_proj[unit_neg_b, mask_offset + i] = -M_o

                # output = (unit_pos_t - unit_pos_b) + (unit_neg_t - unit_neg_b)
                output_proj[unit_pos_t, out_idx] = 1.0
                output_proj[unit_pos_b, out_idx] = -1.0
                output_proj[unit_neg_t, out_idx] = 1.0
                output_proj[unit_neg_b, out_idx] = -1.0

        result = linear_relu_linear(
            input_node=inp,
            input_proj=input_proj,
            input_bias=input_bias,
            output_proj=output_proj,
            output_bias=output_bias,
            name="broadcast_select",
        )
        tv = true_value.value_type
        fv = false_value.value_type
        r = tv.value_range.union(fv.value_range)
        if r.is_finite():
            gate_atol = M * c_tol
            result = assert_matches_value_type(
                result, NodeValueType(value_range=r), atol=gate_atol
            )
        from torchwright.graph.affine_rules import (
            _apply_semantic_override,
            _broadcast_select_semantic_bound,
        )

        _apply_semantic_override(
            result,
            _broadcast_select_semantic_bound(
                true_value._affine_bound,
                false_value._affine_bound,
                n_slots,
                d_fill,
                true_is_broadcast,
                false_is_broadcast,
                tolerance=c_tol * M,
            ),
        )
        return result

    # approximate=False: two sublayers, cancellation-free.
    # Sublayer 1: c_off[i] = ReLU(-masks[i]), c_on[i] = ReLU(masks[i]).
    # Output layout is [c_off_0..c_off_{n-1}, c_on_0..c_on_{n-1}] (width 2n).
    s1_in_proj = torch.zeros(2 * n_slots, n_slots)
    s1_out_proj = torch.zeros(2 * n_slots, 2 * n_slots)
    for i in range(n_slots):
        s1_in_proj[i, i] = -1.0  # c_off row
        s1_in_proj[n_slots + i, i] = 1.0  # c_on row
        s1_out_proj[i, i] = 1.0
        s1_out_proj[n_slots + i, n_slots + i] = 1.0
    cond_gates = linear_relu_linear(
        input_node=masks,
        input_proj=s1_in_proj,
        input_bias=torch.zeros(2 * n_slots),
        output_proj=s1_out_proj,
        output_bias=torch.zeros(2 * n_slots),
        name="broadcast_select_cond_gates",
    )

    # Sublayer 2 reads [c_off (n_slots), c_on (n_slots), true_value, false_value].
    # Column layout:
    c_off_col = 0  # c_off[i] at col i
    c_on_col = n_slots  # c_on[i]  at col n_slots + i
    true_col = 2 * n_slots  # true_value slice
    false_col = 2 * n_slots + len(true_value)  # false_value slice

    d_hidden = 4 * n_slots * d_fill
    d_input = 2 * n_slots + len(true_value) + len(false_value)
    input_proj = torch.zeros(d_hidden, d_input)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, n_slots * d_fill)
    output_bias = torch.zeros(n_slots * d_fill)

    for i in range(n_slots):
        for j in range(d_fill):
            out_idx = i * d_fill + j
            t_pos = 4 * out_idx
            t_neg = 4 * out_idx + 1
            f_pos = 4 * out_idx + 2
            f_neg = 4 * out_idx + 3
            true_src = true_col + (j if true_is_broadcast else i * d_fill + j)
            false_src = false_col + (j if false_is_broadcast else i * d_fill + j)

            # true branch gated by c_off[i]
            input_proj[t_pos, c_off_col + i] = -M
            input_proj[t_pos, true_src] = 1.0
            input_proj[t_neg, c_off_col + i] = -M
            input_proj[t_neg, true_src] = -1.0
            # false branch gated by c_on[i]
            input_proj[f_pos, c_on_col + i] = -M
            input_proj[f_pos, false_src] = 1.0
            input_proj[f_neg, c_on_col + i] = -M
            input_proj[f_neg, false_src] = -1.0

            output_proj[t_pos, out_idx] = 1.0
            output_proj[t_neg, out_idx] = -1.0
            output_proj[f_pos, out_idx] = 1.0
            output_proj[f_neg, out_idx] = -1.0

    x = Concatenate([cond_gates, true_value, false_value])
    return linear_relu_linear(
        input_node=x,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
        name="broadcast_select",
    )


def soft_blend(cond: Node, t: Node, f: Node, *, atol: float = 1e-4) -> Node:
    """Bounded crisp-handoff switch between ``t`` and ``f`` driven by ``cond``.

    This is **not** a cond-blender.  It is for the case where ``cond`` is crisp
    (near ±1) wherever ``t`` and ``f`` differ, and soft (near 0) only where
    ``t ≈ f`` — so almost no resolution flows through ``cond``; the resolution
    lives in ``t``/``f``, read during the crisp-cond interiors.  Where ``cond``
    *is* soft, ``t`` and ``f`` are (by the caller's construction) nearly equal,
    so any blend between them is acceptable.

    The motivating use is the octant recency ramp (docs/rope_port_plan.md §3,
    Phase 1b): the per-octant branch values are constructed to be *exactly*
    equal at each shared octant boundary, which is exactly where the selecting
    ``cond`` (a ``compare`` on ``|u|−|v|``, ``u``, or ``v``) passes through 0
    and cannot saturate.  ``select``'s ``(M+v)−M`` cancellation core drives the
    output to ≈ ``−M`` there (a non-monotone dip); ``soft_blend`` cannot, because
    its core has **no** ``−M`` output bias and the result is then clamped into
    the ``[min(t,f), max(t,f)]`` box.

    Construction — a median-of-three, ``out = median(raw, min(t,f), max(t,f))``
    ``= min(max(raw, min(t,f)), max(t,f))``:

    - ``raw`` reuses ``broadcast_select``'s per-unit carrier core with
      ``output_bias = 0`` (instantiated ``n_slots=1``)::

          raw_j = ReLU(M_j·cond + t_j) − ReLU(M_j·cond)
                + ReLU(−M_j·cond + f_j) − ReLU(−M_j·cond)

      At ``cond=+1`` this is ``t_j``; at ``cond=−1`` it is ``f_j``; at
      ``cond=0`` it is ``ReLU(t_j)+ReLU(f_j)`` — bounded by ``|t_j|+|f_j|``,
      **never** the ``−M`` dip that ``select`` produces.
    - The ``min``/``max`` clamp is load-bearing: for **same-sign** ``t,f`` the
      ``raw`` core overshoots the ``[min,max]`` box (e.g. ``t=f=a>0`` gives
      ``raw=2a`` at ``cond=0``); the clamp restores both in-box-ness *and*
      monotonicity.

    No activation×activation multiply, no ``compare`` postcondition assert.

    Args:
        cond: length-1 switch, crisp (±1) where ``t``/``f`` differ.
        t: value when ``cond`` is +1 (any width ``d``).
        f: value when ``cond`` is −1 (same width ``d``).
        atol: tolerance for the output value-type assert; covers the PL noise
            of the two ``min``/``max`` clamp ops.

    Returns:
        Node of width ``d``, value in ``[min(t,f), max(t,f)]`` elementwise.

    .. noise-footer::

       Max error: 1.192e-07 abs, 1.672e-07 rel over 4096 samples;
       measured at commit 2e04d93. See docs/numerical_noise.md.
    """
    from torchwright.ops.arithmetic_ops import min as op_min, max as op_max

    assert len(cond) == 1
    assert len(t) == len(f)
    d = len(t)

    # Per-column carrier offsets from the union of t/f ranges (raises if either
    # lacks a finite value-type — soft_blend requires bounded operands).
    M = _select_offset(t, f, "soft_blend")
    M_cols = _broadcast_select_per_column_offsets(t, f, 1, d, False, False, M)

    # raw: broadcast_select's carrier core, n_slots=1, output_bias=0, and
    # WITHOUT the crisp-mask union assert / c_tol override tail (which would
    # fire on a soft cond).
    d_hidden = 4 * d
    inp = Concatenate([cond, t, f])
    cond_off = 0
    t_off = 1
    f_off = 1 + d
    input_proj = torch.zeros(d_hidden, len(inp))
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, d)
    output_bias = torch.zeros(d)
    for j in range(d):
        M_o = M_cols[j]
        u_pt = 4 * j  # ReLU(M·cond + t_j)
        u_pb = 4 * j + 1  # ReLU(M·cond)
        u_nt = 4 * j + 2  # ReLU(-M·cond + f_j)
        u_nb = 4 * j + 3  # ReLU(-M·cond)
        input_proj[u_pt, cond_off] = M_o
        input_proj[u_pt, t_off + j] = 1.0
        input_proj[u_pb, cond_off] = M_o
        input_proj[u_nt, cond_off] = -M_o
        input_proj[u_nt, f_off + j] = 1.0
        input_proj[u_nb, cond_off] = -M_o
        output_proj[u_pt, j] = 1.0
        output_proj[u_pb, j] = -1.0
        output_proj[u_nt, j] = 1.0
        output_proj[u_nb, j] = -1.0
    raw = linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
        name="soft_blend_raw",
    )

    lo = op_min(t, f)
    hi = op_max(t, f)
    out = op_min(op_max(raw, lo), hi)

    # Output is clamped into [min(t,f), max(t,f)] ⊆ union(t,f) box, regardless
    # of cond — so the union value-type holds up to the min/max PL noise.
    tv = t.value_type
    fv = f.value_type
    r = tv.value_range.union(fv.value_range)
    if r.is_finite():
        out = assert_matches_value_type(out, NodeValueType(value_range=r), atol=atol)
    return out
