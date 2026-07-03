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
    gated value is exactly zero — the pinned losing-branch claim."""
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
    derivation)."""
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
    the saturated gate is linear in the cond (no M anywhere)."""
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
    terms, replacing the ReLU-era c_tol·M."""
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
    by (1 + c_tol) for cond noise."""
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
