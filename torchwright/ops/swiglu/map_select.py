"""Selection ops on the swish machine.

``select`` is the two-branch complementary gate pair per its entry in
``docs/ops_plain_english.md``; ``switch`` composes ``cond_gate``.  The
ReLU-era offset apparatus (``per_column_offsets``, ``scalar_M``, the
finite-range requirement) does not exist on this machine — cond noise
lands proportionally to the actual branch values.
"""

from typing import Dict, List

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.misc import LiteralValue
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import (
    embedding_step_sharpness,
    scale,
    step_sharpness,
    swish_dip,
)

# sum_nodes is purely linear (Add hardware, no activation) — machine-neutral
# in substance, shared with the frozen relu package until its retirement
# relocates the linear ops.
from torchwright.ops.relu.arithmetic_ops import add_const, sum_nodes
from torchwright.ops.swiglu.logic_ops import _GATE_C_TOL, _assert_cond_pm1, cond_gate
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn


def _select_output_type(true_node: Node, false_node: Node) -> NodeValueType:
    r = true_node.value_type.value_range.union(false_node.value_type.value_range)
    if not r.is_finite():
        return NodeValueType()
    return NodeValueType(value_range=r)


def select(cond: Node, true_node: Node, false_node: Node) -> Node:
    """Output one of two nodes based on a boolean condition.

    A complementary pair of gated lanes per component::

        select(cond, a, b) = Swish(scale·cond)·a/scale + Swish(−scale·cond)·b/scale

    ``Swish(scale·cond)/scale ≈ ReLU(cond)`` — 1 at ``cond=+1``, 0 at
    ``cond=−1`` — so the two gates are complementary on/off indicators:
    one passes ``a``, the other ``b``.  ``cond`` is ±1, enforced by an
    assert.  At clean conds in fp32 the losing branch contributes
    exactly zero (``σ(−scale)`` computes as 0.0) and the winner passes
    with at most ~1 ulp relative rounding.  A cond off ±1 by δ
    mis-scales the winner by exactly δ·|actual value| — there is no
    offset ``M``, no finite-range requirement on the branches, and the
    semantic bound widens relatively, not by ``c_tol·M``.

    Args:
        cond (Node): Condition node that outputs either true or false.
        true_node (Node): Node to be outputted if the condition is true.
        false_node (Node): Node to be outputted if the condition is false.

    Returns:
        Node: Either true_node or false_node based on the condition.

    .. noise-footer::

       Max error: 4.768e-07 abs, 1.19e-07 rel over 4096 samples;
       measured at commit 249512e. See docs/numerical_noise.md.
    """
    assert len(cond) == 1
    assert len(true_node) == len(false_node)

    d = len(true_node)
    cond = _assert_cond_pm1(cond)

    # 2·d gated lanes: d with gate row +scale·cond and up rows picking
    # true_node's components, d mirrored with −scale·cond picking
    # false_node's; the /scale folds into out_proj.
    gate_proj = torch.zeros(2 * d, 1 + 2 * d)
    gate_proj[:d, 0] = scale
    gate_proj[d:, 0] = -scale
    up_proj = torch.zeros(2 * d, 1 + 2 * d)
    up_proj[:d, 1 : 1 + d] = torch.eye(d)
    up_proj[d:, 1 + d :] = torch.eye(d)
    output_proj = torch.cat([torch.eye(d), torch.eye(d)], dim=0) / scale

    x = Concatenate([cond, true_node, false_node])
    result = swiglu_ffn(
        x,
        gate_proj,
        torch.zeros(2 * d),
        output_proj,
        torch.zeros(d),
        up_proj=up_proj,
        up_bias=torch.zeros(2 * d),
        name="select",
    )

    vt = _select_output_type(true_node, false_node)
    if vt != NodeValueType.unknown():
        r = vt.value_range
        gate_atol = _GATE_C_TOL * max(abs(r.lo), abs(r.hi))
        result = assert_matches_value_type(result, vt, atol=gate_atol)
    from torchwright.graph.affine_rules import (
        _apply_semantic_override,
        _select_semantic_bound,
    )

    _apply_semantic_override(
        result,
        _select_semantic_bound(
            true_node._affine_bound,
            false_node._affine_bound,
            rel_tolerance=_GATE_C_TOL,
        ),
    )
    return result


def switch(conditions: List[Node], values: List[Node]) -> Node:
    """Select one of N values based on which condition is true.

    Assumes exactly one condition is true (1.0), rest are false (-1.0).
    A :func:`cond_gate` per branch summed on Add hardware — each losing
    branch contributes exactly zero at clean conds (see cond_gate), so
    the sum is the winning value.

    Args:
        conditions (List[Node]): Boolean condition nodes (each length 1).
        values (List[Node]): Value nodes (all same length).

    Returns:
        Node: The value whose corresponding condition is true.
    """
    return sum_nodes([cond_gate(c, v) for c, v in zip(conditions, values)])

_MASK_TOL = 4.0 * swish_dip / scale
"""Mask tolerance for :func:`broadcast_select` (≈ 0.0111 at scale=100).

Unlike select/cond_gate's ±1 cond contract (c_tol = 0.005),
broadcast_select's masks arrive from :func:`in_range`, whose in-contract
outputs deviate from ±1 by up to ``4·swish_dip/scale`` (two ramps, each
with a possible fillet dip).  A saturated gate is linear in the mask, so
a mask off by δ mis-scales the winner by exactly δ·|value| — the ReLU-era
0.005 budget cannot survive for in_range-fed masks; consumer budgets
re-derive from δ·|value| with this δ.  (Spec: broadcast_select's noise
interlock note in docs/ops_plain_english.md.)"""


def _is_zero_literal(node: Node) -> bool:
    return isinstance(node, LiteralValue) and bool((node.value == 0).all())


def in_range(lower: Node, upper: Node, n_slots: int) -> Node:
    """Test each integer position against a runtime interval.

    For each position i in {0, 1, ..., n_slots-1}, returns 1.0 (true)
    if lower <= i + 0.5 < upper, or -1.0 (false) otherwise.  Per slot,
    two compare-shaped sharpened ramps combined in out_proj::

        past_lower_i = hinge(S·(center_i − lower)) − hinge(S·(center_i − lower) − 1)
        past_upper_i = hinge(S·(center_i − upper)) − hinge(S·(center_i − upper) − 1)
        out_i        = 2·(past_lower_i − past_upper_i) − 1

    Integer-valued bounds are bit-exact across the whole slot vector:
    the +0.5 center offset keeps every hinge argument at least 4 units
    from its bend — fully saturated/underflowed at scale=100 (modulo
    the folded-/scale product-rounding ulp class).  Continuous bounds
    inherit compare's contract per boundary: a bound inside a center's
    ramp zone makes that slot an interpolated intermediate (as today),
    and a bound within ``~17/(scale·S)`` of a ramp edge adds a fillet
    dip.  At most two hinges per slot can sit in fillets at once, so the
    worst in-contract deviation from ±1 is ``4·swish_dip/scale`` ≈ 0.011
    — the value-range assert carries that slack, and it is what lands on
    :func:`broadcast_select`'s mask contract (see ``_MASK_TOL``).

    Args:
        lower: Scalar node, lower bound of the interval.
        upper: Scalar node, upper bound of the interval.
        n_slots: Number of integer positions to test (0 through n_slots-1).

    Returns:
        Node of width n_slots, each value 1.0 (in range) or -1.0 (out of range).

    .. noise-footer::

       Max error: 5.96e-08 abs, 5.96e-08 rel over 4096 samples;
       measured at commit 249512e. See docs/numerical_noise.md.
    """
    assert len(lower) == 1
    assert len(upper) == 1

    S = step_sharpness
    n_lanes = 4 * n_slots  # 4 degenerate lanes per slot

    inp = Concatenate([lower, upper])

    gate_proj = torch.zeros(n_lanes, 2)
    gate_bias = torch.zeros(n_lanes)
    output_proj = torch.zeros(n_lanes, n_slots)
    output_bias = torch.full((n_slots,), -1.0)

    for i in range(n_slots):
        center = i + 0.5
        base = 4 * i

        # past_lower ramp: hinges at lower = center and lower = center − 1/S
        gate_proj[base, 0] = -scale * S
        gate_bias[base] = scale * S * center
        gate_proj[base + 1, 0] = -scale * S
        gate_bias[base + 1] = scale * (S * center - 1.0)

        # past_upper ramp: same shape reading upper
        gate_proj[base + 2, 1] = -scale * S
        gate_bias[base + 2] = scale * S * center
        gate_proj[base + 3, 1] = -scale * S
        gate_bias[base + 3] = scale * (S * center - 1.0)

        # out_i = 2·(past_lower − past_upper) − 1, /scale folded in
        output_proj[base, i] = 2.0 / scale
        output_proj[base + 1, i] = -2.0 / scale
        output_proj[base + 2, i] = -2.0 / scale
        output_proj[base + 3, i] = 2.0 / scale

    result = swiglu_ffn(
        inp,
        gate_proj,
        gate_bias,
        output_proj,
        output_bias,
        name="in_range",
    )
    slack = 4.0 * swish_dip / scale
    return assert_matches_value_type(
        result, NodeValueType(value_range=Range(-1.0 - slack, 1.0 + slack))
    )


def broadcast_select(
    masks: Node,
    true_value: Node,
    false_value: Node,
    n_slots: int,
    d_fill: int,
) -> Node:
    """Select between two values at each of N slots, based on per-slot masks.

    A vectorized :func:`~torchwright.ops.swiglu.map_select.select`: per
    slot, the complementary gated pair with ``mask_i`` as the cond::

        out[i·d_fill + j] = Swish(scale·mask_i)·t_ij/scale
                          + Swish(−scale·mask_i)·f_ij/scale

    One gated form serves every caller — the ReLU-era ``approximate``
    flag, the ``(M+v)−M`` offset apparatus, and the two-sublayer exact
    path all die.  At clean ±1 masks in fp32 the losing branch
    contributes exactly zero and the winner passes with ~1 ulp relative
    rounding.  **Junk masks are safe by construction** — no ±1 assert:
    the flagship builds picks eagerly and discards rows whose masks are
    fractional; once ``|mask| ≥ 0.17`` the gate saturates to the mask
    value itself, so a fractional mask in [-1, 1] blends
    ``≈ ReLU(m)·t + ReLU(−m)·f``, bounded by the branch hull plus the
    dip term ``swish_dip/scale·|branch|``.  Where the output is
    consumed, masks must sit within ``_MASK_TOL`` of ±1 (sized for
    in_range-fed masks — see the module constant); a mask off by δ
    mis-scales the winner by exactly δ·|actual value|, never δ·M.

    A branch that is an all-zero :class:`LiteralValue` contributes
    nothing — its lanes drop at build time (``dynamic_extract``'s false
    branch: the op degenerates to a per-slot cond_gate).

    Args:
        masks: Node of width n_slots, ±1 where consumed (see above).
        true_value: Node of width d_fill (broadcast to all slots)
            or n_slots*d_fill (per-slot values).
        false_value: Same shape options as true_value.
        n_slots: Number of slots.
        d_fill: Width of the value at each slot.

    Returns:
        Node of width n_slots * d_fill.

    .. noise-footer::

       Max error: 4.768e-07 abs, 1.192e-07 rel over 4096 samples;
       measured at commit 249512e. See docs/numerical_noise.md.
    """
    assert len(masks) == n_slots
    true_is_broadcast = len(true_value) == d_fill
    false_is_broadcast = len(false_value) == d_fill
    assert true_is_broadcast or len(true_value) == n_slots * d_fill
    assert false_is_broadcast or len(false_value) == n_slots * d_fill

    d_out = n_slots * d_fill
    drop_true = _is_zero_literal(true_value)
    drop_false = _is_zero_literal(false_value)

    # Build the concatenated input from the live branches only; a dropped
    # zero branch never enters the graph.
    parts = [masks]
    branch_offset = {}
    off = n_slots
    for name, node, dropped in (
        ("true", true_value, drop_true),
        ("false", false_value, drop_false),
    ):
        if not dropped:
            parts.append(node)
            branch_offset[name] = off
            off += len(node)
    inp = Concatenate(parts)
    d_input = off

    lanes_per_col = 2 - int(drop_true) - int(drop_false)
    n_lanes = lanes_per_col * d_out
    gate_proj = torch.zeros(n_lanes, d_input)
    up_proj = torch.zeros(n_lanes, d_input)
    output_proj = torch.zeros(n_lanes, d_out)

    lane = 0
    for i in range(n_slots):
        for j in range(d_fill):
            out_idx = i * d_fill + j
            for name, sign, node, is_bcast, dropped in (
                ("true", 1.0, true_value, true_is_broadcast, drop_true),
                ("false", -1.0, false_value, false_is_broadcast, drop_false),
            ):
                if dropped:
                    continue
                gate_proj[lane, i] = sign * scale
                src = branch_offset[name] + (j if is_bcast else out_idx)
                up_proj[lane, src] = 1.0
                output_proj[lane, out_idx] = 1.0 / scale
                lane += 1

    result = swiglu_ffn(
        inp,
        gate_proj,
        torch.zeros(n_lanes),
        output_proj,
        torch.zeros(d_out),
        up_proj=up_proj,
        up_bias=torch.zeros(n_lanes),
        name="broadcast_select",
    )

    tv = true_value.value_type
    fv = false_value.value_type
    r = tv.value_range.union(fv.value_range)
    if r.is_finite():
        # Junk masks in [-1, 1] blend inside the hull plus the two gates'
        # dips; masks within _MASK_TOL of ±1 add δ·|value|.
        max_abs = max(abs(r.lo), abs(r.hi))
        gate_atol = (_MASK_TOL + 2.0 * swish_dip / scale) * max_abs
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
            rel_tolerance=_MASK_TOL,
        ),
    )
    return result


def dynamic_extract(
    table: Node,
    idx: Node,
    n_entries: int,
    d_fill: int,
) -> Node:
    """Read a ``d_fill``-wide slice from a runtime-valued table at a runtime index.

    Given ``table`` of width ``n_entries * d_fill``, laid out slot-major
    so entry ``i`` occupies columns ``[i*d_fill, (i+1)*d_fill)``, and a
    scalar ``idx`` carrying an integer in ``[0, n_entries - 1]``, returns
    the ``d_fill``-wide slice ``table[idx*d_fill : (idx+1)*d_fill]``.

    The composition (unchanged from the ReLU form):

    1. ``in_range(idx, idx + 1, n_entries)`` emits a width-``n_entries``
       mask with ``+1`` at slot ``floor(idx)`` and ``-1`` elsewhere.
    2. ``broadcast_select(mask, table, zero, ...)`` keeps the selected
       slot's values.  The false branch is a zero literal, so its lanes
       drop at build time — the stage is ``n_entries·d_fill`` gated
       lanes, a per-slot cond_gate.
    3. A free ``Linear`` sums across slots.  Every losing slot is
       *exactly* zero at clean masks (``σ(−scale)`` computes as 0.0), so
       the sum degenerates to a copy of the selected slot.

    Contract: ``idx`` integer in ``[0, n_entries - 1]``; off-integer
    inputs round toward the nearest slot with the boundary at
    ``k + 0.5``; out-of-range inputs produce an all-zero output.  The
    in_range→broadcast_select noise interlock (mask deviation up to
    ``_MASK_TOL``, landing as δ·|table value|) is this op's error story.

    Args:
        table: Runtime node of width ``n_entries * d_fill``.
        idx: Scalar node carrying an integer in ``[0, n_entries - 1]``.
        n_entries: Number of logical entries in ``table`` (compile-time).
        d_fill: Width of each entry (compile-time).

    Returns:
        A ``d_fill``-wide node carrying the selected entry.

    .. noise-footer::

       Max error: 4.768e-07 abs, 1.191e-07 rel over 4096 samples;
       measured at commit 249512e. See docs/numerical_noise.md.
    """
    assert len(idx) == 1, "idx must be a 1D scalar node"
    assert len(table) == n_entries * d_fill, (
        f"table has width {len(table)}; expected n_entries*d_fill = "
        f"{n_entries * d_fill}"
    )
    assert n_entries >= 1, "n_entries must be at least 1"
    assert d_fill >= 1, "d_fill must be at least 1"

    idx_plus_one = add_const(idx, 1.0)
    one_hot = in_range(idx, idx_plus_one, n_entries)

    masked = broadcast_select(
        masks=one_hot,
        true_value=table,
        false_value=LiteralValue(
            torch.zeros(d_fill), name="dynamic_extract_zero"
        ),
        n_slots=n_entries,
        d_fill=d_fill,
    )

    sum_matrix = torch.zeros(n_entries * d_fill, d_fill)
    for slot in range(n_entries):
        for c in range(d_fill):
            sum_matrix[slot * d_fill + c, c] = 1.0
    return Linear(masked, sum_matrix, name="dynamic_extract_sum")

def map_to_table(
    inp: Node, key_to_value: Dict[torch.Tensor, torch.Tensor], default: torch.Tensor
) -> Node:
    """Map the value of the input node to a lookup table.

    A bank of equals_vector-shaped hinges — one degenerate lane per
    table entry, rescaled to 0/1, with each entry's value-delta folded
    into out_proj::

        m_i     = key_i·inp − key_i·key_i
        match_i = speed · Swish(scale·(m_i + 1/speed))/scale
        result  = default + Σ_i match_i · (value_i − default)

    Today's ReLU construction ported hinge-for-hinge (the deltas are
    constants, so no live multiply anywhere).  A matching entry's
    indicator is bit-exact 1 (hinge argument ``scale/speed``, fully
    saturated), so the result is ``value_i`` to ~1 ulp (the
    ``×scale/÷scale`` round trip) plus leakage from the other entries —
    and in real vocabularies that leakage is zero: a non-matching key's
    margin sits far below the bend, deep in fp32 underflow, so a
    no-match input returns ``default`` bit-exactly.  Sensitivity to
    embedding noise at a match is ``speed`` per dot-product unit —
    identical to the ReLU form (self-normalizing hinge).  Between-keys
    inputs can partially fire several indicators, as today: the
    contract is approximate match, not exact selection.

    Args:
        inp (Node): Node whose values will be looked up.
        key_to_value (Dict[torch.Tensor, torch.Tensor]): Lookup table
            mapping from keys to values.
        default (torch.Tensor): Default tensor to return if the input
            value doesn't exist in the table.

    Returns:
        Node: Output node with mapped values.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit 249512e. See docs/numerical_noise.md.
    """
    d_keys = {len(x) for x in key_to_value.keys()}
    d_values = {len(x) for x in key_to_value.values()}
    assert len(d_keys) == 1
    assert len(d_values) == 1
    d_key = d_keys.pop()
    d_value = d_values.pop()
    assert len(inp) == d_key
    assert len(default) == d_value

    n_lanes = len(key_to_value)
    speed = embedding_step_sharpness

    gate_proj = torch.zeros(n_lanes, d_key)
    gate_bias = torch.zeros(n_lanes)
    output_proj = torch.zeros(n_lanes, d_value)

    for i, (key, value) in enumerate(key_to_value.items()):
        gate_proj[i, :] = scale * key
        gate_bias[i] = scale * (1.0 / speed - (key @ key))
        output_proj[i, :] = speed * (value - default) / scale

    result = swiglu_ffn(
        inp,
        gate_proj,
        gate_bias,
        output_proj,
        default,
        name="map_to_table",
    )

    # Output = default + sum_i match_i * (value_i - default).  When
    # multiple keys overlap, multiple indicators can fire at once, so
    # bound per channel by |default[j]| + sum_i |value_i[j] - default[j]|
    # — the hand-written claim ports unchanged (without it, interval
    # arithmetic blows up after a few chained lookups).  Dips sit
    # comfortably inside it (each entry's worst contribution is
    # swish_dip/scale·|Δ_i| against a claim of |Δ_i|): no new slack.
    diff_abs_sum = torch.zeros(d_value)
    for value in key_to_value.values():
        diff_abs_sum = diff_abs_sum + (value - default).abs()
    lo = float((default - diff_abs_sum).min().item())
    hi = float((default + diff_abs_sum).max().item())
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=Range(lo, hi)),
    )

