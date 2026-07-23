"""swiglu select, cond_gate, switch: the complementary gated pair and its
compositions.

Spec: docs/ops_plain_english.md (select, cond_gate entries; switch is a
composition).  Pinned facts these lean on — losing branch exactly zero at
clean conds (σ(−scale) = 0 in fp32), winner ~1 ulp relative, saturated
gate linear in the mask — live in tests/docs/test_swish_constants.py
(the gated_select tests).
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN
from torchwright.ops.const import scale
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.swiglu import cond_gate, select, switch
from torchwright.ops.swiglu.logic_ops import _GATE_C_TOL

D = 64
D_HEAD = 8


def _unwrap(node):
    while not isinstance(node, FFN):
        node = node.inputs[0]
    return node


def _cond(*vals):
    return torch.tensor(vals).reshape(-1, 1)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def test_select_structure():
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    a = create_input("a", 3, value_range=(-5.0, 5.0))
    b = create_input("b", 3, value_range=(-2.0, 8.0))
    out = select(c, a, b)
    ffn = _unwrap(out)
    assert ffn.activation == "swish"
    assert not ffn.is_degenerate
    assert ffn.n_lanes == 6  # 2·w gated lanes
    assert len(out) == 3


def test_select_picks_branch_winner_ulp_class():
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    a = create_input("a", 3, value_range=(-5.0, 5.0))
    b = create_input("b", 3, value_range=(-2.0, 8.0))
    out = select(c, a, b)
    g = torch.Generator().manual_seed(51)
    at = torch.rand(4, 3, generator=g) * 10.0 - 5.0
    bt = torch.rand(4, 3, generator=g) * 10.0 - 2.0
    ct = _cond(1.0, -1.0, 1.0, -1.0)
    val = out.compute(4, {"c": ct, "a": at, "b": bt})
    ref = torch.where(ct > 0, at, bt)
    # Winner passes with ~1 ulp relative rounding (the ×scale/÷scale
    # round trip) — often bit-exact, not always; don't pin equality.
    assert torch.allclose(val, ref, rtol=1e-6, atol=1e-7)


def test_cond_gate_losing_branch_exactly_zero():
    """At cond=-1, σ(-scale) computes as exactly 0.0 in fp32, so the
    gated value is exactly zero — the pinned losing-branch claim.
    """
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    v = create_input("v", 2, value_range=(-1000.0, 1000.0))
    out = cond_gate(c, v)
    vt = torch.tensor([[123.456, -987.654]])
    val = out.compute(1, {"c": _cond(-1.0), "v": vt})
    assert torch.equal(val, torch.zeros(1, 2))


def test_cond_gate_structure_and_pass_through():
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    v = create_input("v", 2, value_range=(-10.0, 10.0))
    out = cond_gate(c, v)
    ffn = _unwrap(out)
    assert not ffn.is_degenerate
    assert ffn.n_lanes == 2  # w gated lanes
    vt = torch.tensor([[3.25, -7.5]])
    val = out.compute(1, {"c": _cond(1.0), "v": vt})
    assert torch.allclose(val, vt, rtol=1e-6, atol=1e-7)


def test_no_finite_range_requirement():
    """The ReLU-era offset apparatus is gone: unbounded branch ranges
    build fine (relu select/cond_gate raise TypeError via the M
    derivation).
    """
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    a = create_input("a", 1)  # no value_range — unbounded
    b = create_input("b", 1)
    out = select(c, a, b)  # must not raise
    val = out.compute(1, {"c": _cond(1.0), "a": _cond(4.0), "b": _cond(9.0)})
    assert torch.allclose(val, torch.tensor([[4.0]]), rtol=1e-6, atol=1e-7)
    out2 = cond_gate(c, a)  # must not raise
    val2 = out2.compute(1, {"c": _cond(-1.0), "a": _cond(4.0)})
    assert torch.equal(val2, torch.zeros(1, 1))


def test_cond_assert_fires_on_junk_cond():
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    a = create_input("a", 1, value_range=(0.0, 10.0))
    b = create_input("b", 1, value_range=(0.0, 10.0))
    out = select(c, a, b)
    with pytest.raises(AssertionError, match="cond near"):
        out.compute(1, {"c": _cond(0.5), "a": _cond(1.0), "b": _cond(2.0)})


def test_cond_deviation_scales_with_actual_value():
    """A cond off ±1 by δ mis-scales the winner by exactly δ·|value| —
    the saturated gate is linear in the cond (no M anywhere).
    """
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    v = create_input("v", 1, value_range=(-1000.0, 1000.0))
    out = cond_gate(c, v)
    delta = 0.004  # inside c_tol
    for value in (0.125, 800.0):
        val = out.compute(1, {"c": _cond(1.0 + delta), "v": _cond(value)})
        expected = (1.0 + delta) * value  # gate saturated: passes 1+δ itself
        assert val.item() == pytest.approx(expected, rel=1e-6)


def test_select_semantic_bound_relative_widening():
    """The semantic hull widens by c_tol·|side| per side — actual-value
    terms, replacing the ReLU-era c_tol·M.
    """
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    a = create_input("a", 1, value_range=(2.0, 5.0))
    b = create_input("b", 1, value_range=(-3.0, 4.0))
    out = select(c, a, b)
    iv = out.affine_bound.to_interval()
    assert len(iv) == 1
    assert iv[0].lo == pytest.approx(-3.0 - _GATE_C_TOL * 3.0)
    assert iv[0].hi == pytest.approx(5.0 + _GATE_C_TOL * 5.0)


def test_cond_gate_semantic_bound_sign_determined_scaling():
    """Sign-determined inputs keep the pass-through affine bound, scaled
    by (1 + c_tol) for cond noise.
    """
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    v = create_input("v", 1, value_range=(1.0, 6.0))
    out = cond_gate(c, v)
    iv = out.affine_bound.to_interval()
    assert iv[0].lo == pytest.approx(0.0)
    assert iv[0].hi == pytest.approx(6.0 * (1.0 + _GATE_C_TOL))


# ---------------------------------------------------------------------------
# switch
# ---------------------------------------------------------------------------


def test_switch_picks_the_true_branch():
    conds = [create_input(f"c{i}", 1, value_range=(-1.0, 1.0)) for i in range(3)]
    vals = [create_input(f"v{i}", 2, value_range=(-9.0, 9.0)) for i in range(3)]
    out = switch(conds, vals)
    inputs = {
        "c0": _cond(-1.0),
        "c1": _cond(1.0),
        "c2": _cond(-1.0),
        "v0": torch.tensor([[1.0, 2.0]]),
        "v1": torch.tensor([[3.0, 4.0]]),
        "v2": torch.tensor([[5.0, 6.0]]),
    }
    val = out.compute(1, inputs)
    # Losing branches are exactly zero, so the sum is the winner alone
    # (up to its own ulp-class rounding).
    assert torch.allclose(val, torch.tensor([[3.0, 4.0]]), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# compiled
# ---------------------------------------------------------------------------


def test_select_and_switch_compile_clean():
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    a = create_input("a", 2, value_range=(-10.0, 10.0))
    b = create_input("b", 2, value_range=(-10.0, 10.0))
    picked = select(c, a, b)
    gated = cond_gate(c, picked)
    compiled = compile_headless(gated, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    inputs = {
        "c": _cond(1.0, -1.0, 1.0, -1.0),
        "a": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        "b": torch.tensor([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0]]),
    }
    report = probe_compiled(compiled, gated, inputs, 4, atol=1e-3)
    assert report.first_divergent is None, report.format_short()
    # debug=True runs the ±1-cond asserts on compiled values; pack the
    # flat input row per the module's own input layout.
    packed = torch.zeros(4, sum(w for _, _, w in compiled._input_specs))
    for name, start, w in compiled._input_specs:
        packed[:, start : start + w] = inputs[name]
    compiled(packed, debug=True)


# ---------------------------------------------------------------------------
# in_range / broadcast_select / dynamic_extract
# ---------------------------------------------------------------------------


def test_in_range_structure_and_integer_bounds():
    from torchwright.ops.swiglu import in_range

    lo = create_input("lo", 1, value_range=(0.0, 8.0))
    hi = create_input("hi", 1, value_range=(0.0, 8.0))
    out = in_range(lo, hi, 8)
    ffn = _unwrap(out)
    assert ffn.is_degenerate
    assert ffn.n_lanes == 32  # 4 per slot
    # Integer bounds: slots [lo, hi) read +1, others -1, to the folded
    # ulp class (products at scale·S·center magnitudes).
    val = out.compute(
        2, {"lo": torch.tensor([[2.0], [0.0]]), "hi": torch.tensor([[5.0], [8.0]])}
    )
    ref = torch.tensor([[-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0], [1.0] * 8])
    assert torch.allclose(val, ref, atol=1e-5, rtol=0.0)


def test_in_range_dip_slack_and_claimed_range():
    """Continuous bounds near a ramp edge dip past ±1 by up to
    4·swish_dip/scale; the claimed value range carries that slack.
    """
    from torchwright.ops.const import swish_dip
    from torchwright.ops.swiglu import in_range

    lo = create_input("lo", 1, value_range=(0.0, 4.0))
    hi = create_input("hi", 1, value_range=(0.0, 4.0))
    out = in_range(lo, hi, 4)
    slack = 4.0 * swish_dip / scale
    r = out.value_type.value_range
    assert r.lo == pytest.approx(-1.0 - slack)
    assert r.hi == pytest.approx(1.0 + slack)
    # Sweep lower bound through a slot's ramp+fillet zone; outputs stay
    # within the claimed slack and do exceed 1 (the dip is real).
    los = torch.linspace(0.3, 0.6, 3001).unsqueeze(1)
    his = torch.full((3001, 1), 4.0)
    val = out.compute(3001, {"lo": los, "hi": his})
    assert val.min() >= -1.0 - slack - 1e-6
    assert val.max() <= 1.0 + slack + 1e-6
    assert val.max() > 1.0 + 1e-4


def test_broadcast_select_per_slot_and_broadcast():
    from torchwright.ops.swiglu import broadcast_select

    m = create_input("m", 2, value_range=(-1.0, 1.0))
    t = create_input("t", 4, value_range=(-9.0, 9.0))  # per-slot, d_fill=2
    f = create_input("f", 2, value_range=(-9.0, 9.0))  # broadcast
    out = broadcast_select(m, t, f, n_slots=2, d_fill=2)
    ffn = _unwrap(out)
    assert not ffn.is_degenerate
    assert ffn.n_lanes == 8  # 2 per output column
    inputs = {
        "m": torch.tensor([[1.0, -1.0]]),
        "t": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "f": torch.tensor([[7.0, 8.0]]),
    }
    val = out.compute(1, inputs)
    # slot 0 true -> t[0:2]; slot 1 false -> broadcast f
    assert torch.allclose(
        val, torch.tensor([[1.0, 2.0, 7.0, 8.0]]), rtol=1e-6, atol=1e-7
    )


def test_broadcast_select_junk_mask_safe_no_assert():
    """Fractional masks blend smoothly — no ±1 assert fires, and the
    blend stays inside the hull plus the dip term.
    """
    from torchwright.ops.swiglu import broadcast_select

    m = create_input("m", 1, value_range=(-1.0, 1.0))
    t = create_input("t", 1, value_range=(0.0, 8.0))
    f = create_input("f", 1, value_range=(-2.0, 0.0))
    out = broadcast_select(m, t, f, n_slots=1, d_fill=1)
    ms = torch.linspace(-1.0, 1.0, 201).unsqueeze(1)
    ts = torch.full((201, 1), 8.0)
    fs = torch.full((201, 1), -2.0)
    val = out.compute(201, {"m": ms, "t": ts, "f": fs})  # must not raise
    # saturated-gate blend: ReLU(m)·8 + ReLU(-m)·(-2) within hull+dips
    assert val.min() >= -2.0 - 0.028 - 1e-6
    assert val.max() <= 8.0 + 0.028 + 1e-6


def test_broadcast_select_zero_literal_branch_drops_lanes():
    from torchwright.graph.misc import LiteralValue
    from torchwright.ops.swiglu import broadcast_select

    m = create_input("m", 3, value_range=(-1.0, 1.0))
    t = create_input("t", 3, value_range=(-5.0, 5.0))
    zero = LiteralValue(torch.zeros(1), name="z")
    out = broadcast_select(m, t, zero, n_slots=3, d_fill=1)
    ffn = _unwrap(out)
    assert ffn.n_lanes == 3  # false lanes dropped: per-slot cond_gate
    inputs = {
        "m": torch.tensor([[1.0, -1.0, 1.0]]),
        "t": torch.tensor([[2.0, 3.0, 4.0]]),
    }
    val = out.compute(1, inputs)
    # losing slots exactly zero (sigma(-scale) = 0)
    assert val[0, 1].item() == 0.0
    assert torch.allclose(val, torch.tensor([[2.0, 0.0, 4.0]]), rtol=1e-6, atol=1e-7)


def test_broadcast_select_both_branches_zero_collapses_to_literal():
    """BOTH branches all-zero literals: the op is identically zero at any
    mask value, so it must collapse to a zero LiteralValue rather than build
    a zero-lane FFN — the zero-lane bound reaches _gated_lane_affine with
    empty per-lane comparison lists, whose torch.tensor([]) defaults to
    float32 and crashes torch.where.  The flagship hits this construction
    through pick_by_one_hot over an all-zero table (a missing-texture
    bank's palette rows).
    """
    from torchwright.graph.misc import LiteralValue
    from torchwright.ops.swiglu import broadcast_select

    m = create_input("m", 4, value_range=(-1.0, 1.0))
    table_zero = LiteralValue(torch.zeros(4), name="tz")
    zero = LiteralValue(torch.zeros(1), name="z")
    out = broadcast_select(m, table_zero, zero, n_slots=4, d_fill=1)  # no raise
    assert isinstance(out, LiteralValue)
    assert len(out) == 4
    masks = torch.tensor([[1.0, -1.0, 1.0, -1.0], [0.3, 0.0, -0.2, 1.0]])
    val = out.compute(2, {"m": masks})
    assert val.shape == (2, 4)
    assert (val == 0.0).all()


def test_broadcast_select_semantic_bound_mask_tol_widening():
    from torchwright.ops.swiglu import broadcast_select
    from torchwright.ops.swiglu.map_select import _MASK_TOL

    m = create_input("m", 1, value_range=(-1.0, 1.0))
    t = create_input("t", 1, value_range=(2.0, 6.0))
    f = create_input("f", 1, value_range=(-4.0, 1.0))
    out = broadcast_select(m, t, f, n_slots=1, d_fill=1)
    iv = out.affine_bound.to_interval()
    assert iv[0].lo == pytest.approx(-4.0 - _MASK_TOL * 4.0)
    assert iv[0].hi == pytest.approx(6.0 + _MASK_TOL * 6.0)


def test_dynamic_extract_picks_row_and_out_of_range_zeros():
    from torchwright.ops.swiglu import dynamic_extract

    table = create_input("table", 8, value_range=(-9.0, 9.0))
    idx = create_input("idx", 1, value_range=(0.0, 3.0))
    out = dynamic_extract(table, idx, n_entries=4, d_fill=2)
    assert len(out) == 2
    tt = torch.arange(1.0, 9.0).unsqueeze(0).repeat(5, 1)
    ii = torch.tensor([[0.0], [1.0], [2.0], [3.0], [9.0]])
    val = out.compute(5, {"table": tt, "idx": ii})
    ref = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [0.0, 0.0]])
    assert torch.allclose(val, ref, rtol=1e-5, atol=1e-5)


def test_dynamic_extract_compiles_clean():
    from torchwright.ops.swiglu import dynamic_extract

    table = create_input("table", 6, value_range=(-9.0, 9.0))
    idx = create_input("idx", 1, value_range=(0.0, 2.0))
    out = dynamic_extract(table, idx, n_entries=3, d_fill=2)
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    inputs = {
        "table": torch.arange(1.0, 7.0).unsqueeze(0).repeat(3, 1),
        "idx": torch.tensor([[0.0], [1.0], [2.0]]),
    }
    report = probe_compiled(compiled, out, inputs, 3, atol=1e-3)
    assert report.first_divergent is None, report.format_short()
