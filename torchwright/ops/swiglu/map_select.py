"""Selection ops on the swish machine.

``select`` is the two-branch complementary gate pair per its entry in
``docs/ops_plain_english.md``; ``switch`` composes ``cond_gate``.  The
ReLU-era offset apparatus (``per_column_offsets``, ``scalar_M``, the
finite-range requirement) does not exist on this machine — cond noise
lands proportionally to the actual branch values.
"""

import math
from typing import Dict, List, Tuple

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.misc import LiteralValue
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import (
    embedding_step_sharpness,
    min_d_hidden,
    scale,
    step_sharpness,
    swish_dip,
)

from torchwright.ops._math import _lookup_axis_scale, _lookup_numeric_slack
from torchwright.ops.linear import add_const, sum_nodes
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

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit eb4a0f8. See docs/numerical_noise.md.
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
"""Mask tolerance for :func:`broadcast_select` (≈ 0.0087 at scale=128).

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
    from its bend — fully saturated/underflowed, and with ``scale`` a
    power of two the folded ``2/scale`` products of the saturated
    integer hinge values are exact too (at scale=100 those products
    rounded, leaking ~1e-5 per slot above the winner — see
    docs/onehot_accumulated_leak_postmortem.md).  Continuous bounds
    inherit compare's contract per boundary: a bound inside a center's
    ramp zone makes that slot an interpolated intermediate (as today),
    and a bound within ``~17/(scale·S)`` of a ramp edge adds a fillet
    dip.  At most two hinges per slot can sit in fillets at once, so the
    worst in-contract deviation from ±1 is ``4·swish_dip/scale`` ≈ 0.0087
    — the value-range assert carries that slack, and it is what lands on
    :func:`broadcast_select`'s mask contract (see ``_MASK_TOL``).

    Args:
        lower: Scalar node, lower bound of the interval.
        upper: Scalar node, upper bound of the interval.
        n_slots: Number of integer positions to test (0 through n_slots-1).

    Returns:
        Node of width n_slots, each value 1.0 (in range) or -1.0 (out of range).

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit eb4a0f8. See docs/numerical_noise.md.
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
    ``≈ ReLU(m)·t + ReLU(−m)·f``, bounded by the hull of **zero and the
    two branch ranges** plus the dip term ``swish_dip/scale·|branch|`` —
    at m ≈ 0 both gates vanish and the blend collapses toward zero, so
    the junk-mask bound stays inside the asserted range only when zero
    lies in the union of the branch ranges (true for every current
    caller; two live same-signed branches would fire the retained
    value-range assert on exactly the discarded rows).  Where the output is
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

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit eb4a0f8. See docs/numerical_noise.md.
    """
    assert len(masks) == n_slots
    true_is_broadcast = len(true_value) == d_fill
    false_is_broadcast = len(false_value) == d_fill
    assert true_is_broadcast or len(true_value) == n_slots * d_fill
    assert false_is_broadcast or len(false_value) == n_slots * d_fill

    d_out = n_slots * d_fill
    drop_true = _is_zero_literal(true_value)
    drop_false = _is_zero_literal(false_value)

    if drop_true and drop_false:
        # Both branches are all-zero literals: the gated pair is identically
        # zero at every mask value (junk or clean), so the op collapses to
        # the zero literal itself.  Without this, lanes_per_col == 0 below
        # and a zero-lane FFN reaches the affine rules, where the empty
        # per-lane comparison tensors default to float32 and torch.where
        # rejects them.  The flagship hits this through pick_by_one_hot over
        # an all-zero table (a missing-texture bank's palette rows).
        return LiteralValue(torch.zeros(d_out), name="broadcast_select_zero")

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

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit eb4a0f8. See docs/numerical_noise.md.
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
        false_value=LiteralValue(torch.zeros(d_fill), name="dynamic_extract_zero"),
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
       measured at commit eb4a0f8. See docs/numerical_noise.md.
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


def _upper_clamp(index: Node, top: float, gate_mult: float, name: str) -> Node:
    """``min(gate_mult·index, top)`` as one 3-lane degenerate FFN.

    The swish :func:`min` with a constant second operand folded into the
    biases: a hinge lane subtracting ``hinge(g·x − top)`` plus the
    sharpened bypass pair carrying ``g·x`` exactly.  ``gate_mult`` folds
    the caller's index_scale into the gate rows — no separate scaling
    Linear.
    """
    g = gate_mult
    gate_proj = torch.tensor([[scale * g], [scale * g], [-scale * g]])
    gate_bias = torch.tensor([-scale * top, 0.0, 0.0])
    out_proj = torch.tensor([[-1.0], [1.0], [-1.0]]) / scale
    return swiglu_ffn(
        index,
        gate_proj,
        gate_bias,
        out_proj,
        torch.zeros(1),
        name=name,
    )


def _clamp_or_skip(
    index: Node, top: float, gate_mult: float, name: str
) -> Tuple[Node, float]:
    """Feed an axis index to the staircase, spending the clamp FFN only
    when the input range doesn't already prove it in-bounds.

    ``_upper_clamp`` both scales by ``gate_mult`` and caps at ``top`` in
    one FFN sublayer.  When the input's static value range proves
    ``0 ≤ gate_mult·index ≤ top`` the cap is the identity, so we skip the
    clamp FFN and defer the scaling into the staircase gate instead
    (``_bounded_step_chunks(gate_mult=…)``) — the raw index feeds the
    steps and no sublayer is spent.  Returns ``(step_input,
    step_gate_mult)``: the clamped scaled index with unit step-gain, or
    the raw index with ``gate_mult`` deferred.

    ``gate_mult`` is finite and > 0 (``_lookup_axis_scale``), so it
    preserves the ordering of the range endpoints.  A missing, infinite,
    or out-of-bounds range keeps the clamp.
    """
    r = index.value_type.value_range
    if r.is_finite() and 0.0 <= gate_mult * r.lo and gate_mult * r.hi <= top:
        return index, gate_mult
    return _upper_clamp(index, top, gate_mult, name=name), 1.0


def _bounded_step_chunks(
    x_clamped: Node,
    n: int,
    s: float,
    chunk_size: int,
    name: str,
    *,
    gate_mult: float = 1.0,
) -> List[tuple]:
    """floor_int's bounded-step stage per boundary ``k − 0.5``, chunked.

    Returns ``[(ks, step_node), ...]`` where ``step_node`` is the
    width-``len(ks)`` vector of steps
    ``step_k = hinge(t_k) − hinge(t_k − W)``,
    ``t_k = s·(gate_mult·x − (k − 0.5)) + 0.5`` — bounded in [0, W], so
    downstream accumulation never sees ``s·x``-magnitude terms (the fp32
    2^24 rationale, unchanged from relu).  W keeps today's sizing plus the
    swish saturation floor ``W ≥ 1 + 17/scale`` (dominated by the 2).

    ``gate_mult`` folds the axis ``index_scale`` into the boundary gate so
    the raw (unscaled) index can feed the staircase directly when the
    upstream clamp is skipped; it is 1.0 when ``x`` is already scaled.
    """
    max_t = s * float(n - 1) + 1.0
    ulp = 2.0 ** (math.floor(math.log2(max_t)) - 23) if max_t >= 1.0 else 2.0**-23
    step_cap = max(2.0, 8.0 * ulp)  # W

    out = []
    for c0 in range(1, n, chunk_size):
        ks = list(range(c0, min(c0 + chunk_size, n)))
        c = len(ks)
        gate_proj = torch.full((2 * c, 1), scale * s * gate_mult)
        gate_bias = torch.empty(2 * c)
        out_proj = torch.zeros((2 * c, c))
        for j, k in enumerate(ks):
            t_bias = 0.5 - s * (float(k) - 0.5)
            gate_bias[2 * j] = scale * t_bias  # t_k
            gate_bias[2 * j + 1] = scale * (t_bias - step_cap)  # t_k − W
            out_proj[2 * j, j] = 1.0 / scale
            out_proj[2 * j + 1, j] = -1.0 / scale
        step = swiglu_ffn(
            x_clamped,
            gate_proj,
            gate_bias,
            out_proj,
            torch.zeros(c),
            name=f"{name}_step",
        )
        out.append((ks, step))
    return out


def _table_lookup_row_vector(
    index: Node,
    table: torch.Tensor,
    *,
    sharpness: float,
    gate_mult: float,
    chunk_size: int,
    name: str,
) -> Node:
    """Row stage: the selected table row as a live width-``cols`` vector.

    floor_int's two-stage staircase with vector-valued deltas — the
    third stage is 1 *degenerate* lane per boundary (the deltas are
    table constants, folded into out_proj): ``row = T[A−1] −
    Σ_k hinge(1 − step_k)·(T[k] − T[k−1])``.
    """
    rows, cols = table.shape
    if rows == 1:
        return LiteralValue(
            table[0].to(dtype=torch.float32), name=f"{name}_constant_row"
        )

    step_input, step_mult = _clamp_or_skip(
        index, float(rows - 1), gate_mult, name=f"{name}_clamp_i"
    )
    deltas = table[1:] - table[:-1]
    partials: List[Node] = []
    for ci, (ks, step) in enumerate(
        _bounded_step_chunks(
            step_input, rows, sharpness, chunk_size, f"{name}_row", gate_mult=step_mult
        )
    ):
        c = len(ks)
        out_proj = torch.zeros((c, cols))
        for j, k in enumerate(ks):
            out_proj[j, :] = -deltas[k - 1] / scale
        out_bias = table[rows - 1].clone() if ci == 0 else torch.zeros(cols)
        partials.append(
            swiglu_ffn(
                step,
                -scale * torch.eye(c),
                scale * torch.ones(c),
                out_proj,
                out_bias,
                name=f"{name}_row_deltas",
            )
        )
    result = partials[0] if len(partials) == 1 else sum_nodes(partials)
    max_abs = float(table.abs().max().item())
    return assert_matches_value_type(
        result,
        NodeValueType(
            value_range=Range(float(table.min().item()), float(table.max().item()))
        ),
        atol=_lookup_numeric_slack(max_abs, sharpness, rows),
    )


def table_lookup_2d(
    i: Node,
    j: Node,
    table,
    *,
    index_scale=1.0,
    sharpness: float = 100.0,
    name: str = "table_lookup_2d",
) -> Node:
    """Lookup a scalar from a compile-time constant 2D table.

    Both axes are the same machine — floor_int's two-stage saturating
    staircase, a bounded per-boundary step then a "boundary not yet
    reached" indicator ``hinge(1 − step_k)`` weighting that boundary's
    table delta.  **The two axes differ only in whether the deltas are
    live, and that is the whole redesign**: row-axis deltas are table
    constants (folded into out_proj, degenerate lanes, hinge-for-hinge
    with relu); column-axis deltas are differences of the just-built
    live row vector — a live multiply, one *gated* lane per boundary
    (gate row ``scale·(1 − step_k(j))``, up row ``row_{k−1} − row_k``)::

        row_c = T[A−1, c] − Σ_k hinge(1 − step_k(i)) · (T[k,c] − T[k−1,c])
        out   = row_{B−1} + Σ_k hinge(1 − step_k(j)) · (row_{k−1} − row_k)

    The ReLU machine's column detour — a second full mask staircase plus
    the ``0.5·[ReLU(offset·mask + v) − ReLU(offset·mask − v)]`` offset
    gate — collapses into that one gated lane; the mask staircase, the
    offset, and the ``offset·0.005`` guard term all die.  The gate is a
    hinge (not the exact ± multiply pair) because the indicator must be
    *snapped*: a deeply-on step carries fp noise at ``ulp(scale·s·x)``
    scale, and the hinge's saturated ends read it as exactly 0 or 1
    where the exact pair would faithfully reproduce it times the delta.

    Exactness: on the integer grid and outside boundary bands every
    indicator is exactly 0 or 1 (off steps exactly 0; on steps land in
    the W-slack where ``1 − step ≤ −1`` saturates the hinge), so the
    value path is the delta telescoping's fp32 accumulation — what
    disappears is the offset gate's absolute error at ``ulp(offset)``
    scale.  **In-band behavior upgrades from disclaimer to contract**:
    the band coefficient is the blend fraction itself (``hinge(α) = α``
    exactly for ``α ≥ 17/scale``), so a band on one axis gives the clean
    two-entry linear blend and a corner band genuine bilinear
    interpolation; below ``α = 0.17`` the coefficient rolls through the
    dip (error ``≤ swish_dip/scale·|Δ|`` per band edge).

    What survives: ``sharpness`` and its contract (ramp width ``1/s`` on
    half-integer boundaries), ``index_scale`` (now folded into the
    clamp/step gate rows — the separate scaling Linear is deleted), the
    chunking (derived from ``min_d_hidden``), the min-clamps (swish
    ``min``, 3 lanes), and the output-range guard with
    ``_lookup_numeric_slack``.

    Args:
        i: 1D scalar row index (integer-ish; out-of-range clamps).
        j: 1D scalar column index.
        table: 2D compile-time constant array.
        index_scale: scalar or per-axis (row, col) multiplier folded
            into the staircase gate rows.
        sharpness: ramp width ``1/sharpness`` at half-integer
            boundaries (flagship uses 1000).
        name: debug label prefix.

    Returns:
        1D scalar node with ``table[round(i·scale_i), round(j·scale_j)]``.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit eb4a0f8. See docs/numerical_noise.md.
    """
    assert len(i) == 1, "i must be a 1D scalar node"
    assert len(j) == 1, "j must be a 1D scalar node"
    s = float(sharpness)
    if not math.isfinite(s) or s <= 1.0:
        raise ValueError(f"sharpness must be finite and > 1, got {s}")
    chunk_size = max(1, min_d_hidden // 2)

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

    row = _table_lookup_row_vector(
        i,
        table_t,
        sharpness=s,
        gate_mult=scale_i,
        chunk_size=chunk_size,
        name=name,
    )
    if cols == 1:
        return row

    # Column stage: gated lanes reading the live row vector.
    j_input, j_mult = _clamp_or_skip(
        j, float(cols - 1), scale_j, name=f"{name}_clamp_j"
    )
    step_chunks = _bounded_step_chunks(
        j_input, cols, s, chunk_size, f"{name}_col", gate_mult=j_mult
    )

    chunks: List[Node] = []
    for ci, (ks, step) in enumerate(step_chunks):
        c = len(ks)
        first = ci == 0
        n_lanes = c + (1 if first else 0)
        d_input = c + cols
        gate_proj = torch.zeros(n_lanes, d_input)
        gate_bias = torch.zeros(n_lanes)
        up_proj = torch.zeros(n_lanes, d_input)
        out_proj = torch.zeros(n_lanes, 1)
        for lj, k in enumerate(ks):
            # gate: scale·(1 − step_k(j)) — indicator snapped by saturation
            gate_proj[lj, lj] = -scale
            gate_bias[lj] = scale
            # up: row_{k−1} − row_k, read from the live row section
            up_proj[lj, c + k - 1] = 1.0
            up_proj[lj, c + k] = -1.0
            out_proj[lj, 0] = 1.0 / scale
        if first:
            # Always-on lane carrying the live top entry (out_bias cannot —
            # row_{B−1} is a live value, not a constant).
            gate_bias[c] = scale
            up_proj[c, c + cols - 1] = 1.0
            out_proj[c, 0] = 1.0 / scale
        x = Concatenate([step, row])
        chunks.append(
            swiglu_ffn(
                x,
                gate_proj,
                gate_bias,
                out_proj,
                torch.zeros(1),
                up_proj=up_proj,
                up_bias=torch.zeros(n_lanes),
                name=f"{name}_col_gate",
            )
        )

    result = chunks[0] if len(chunks) == 1 else sum_nodes(chunks)
    lo = float(table_t.min().item())
    hi = float(table_t.max().item())
    max_abs = float(table_t.abs().max().item())
    row_slack = _lookup_numeric_slack(max_abs, s, rows)
    # The relu guard's offset·0.005 term dies with the offset gate; the
    # band-edge dip term (swish_dip/scale·max|Δ| per axis, ≤ 2 live) is
    # the swish-specific addition.
    max_delta = 0.0
    if rows > 1:
        max_delta = float((table_t[1:] - table_t[:-1]).abs().max().item())
    if cols > 1:
        max_delta = max(
            max_delta, float((table_t[:, 1:] - table_t[:, :-1]).abs().max().item())
        )
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=Range(lo, hi)),
        atol=max(1e-3, row_slack) + 2.0 * swish_dip / scale * max_delta,
    )
