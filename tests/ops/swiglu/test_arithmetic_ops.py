"""swiglu multiply / square: the exact ± gated-lane pairs.

Spec: docs/ops_plain_english.md (multiply, square entries); the identity
``Swish(a)·b + Swish(-a)·(-b) = a·b`` is pinned in exact math by
tests/docs/test_swish_constants.py::test_multiply_identity_exact.  These
tests cover the *op* — lane structure, oracle values in fp32, and a
compiled pure-swish graph matching the oracle.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.swiglu import multiply, square

N_POS = 64
D = 64
D_HEAD = 8


def _pair(seed=7, lo=-10.0, hi=10.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(N_POS, 1, generator=g) * (hi - lo) + lo
    b = torch.rand(N_POS, 1, generator=g) * (hi - lo) + lo
    return a, b


def test_multiply_structure():
    a = create_input("a", 1, value_range=(-10.0, 10.0))
    b = create_input("b", 1, value_range=(-10.0, 10.0))
    out = multiply(a, b)
    assert isinstance(out, FFN)
    assert out.activation == "swish"
    assert not out.is_degenerate
    assert out.n_lanes == 2
    assert len(out) == 1


def test_multiply_oracle_matches_product():
    a = create_input("a", 1, value_range=(-10.0, 10.0))
    b = create_input("b", 1, value_range=(-10.0, 10.0))
    out = multiply(a, b)
    at, bt = _pair()
    val = out.compute(N_POS, {"a": at, "b": bt})
    ref = at * bt
    # Exact in real math; fp32 evaluation leaves ~ulp-level rounding.
    assert torch.allclose(val, ref, rtol=1e-5, atol=1e-6)


def test_multiply_no_range_limit():
    """Exact at magnitudes far beyond any ReLU-era grid — no max_value."""
    a = create_input("a", 1, value_range=(-1e4, 1e4))
    b = create_input("b", 1, value_range=(-1e4, 1e4))
    out = multiply(a, b)
    at, bt = _pair(seed=11, lo=-1e4, hi=1e4)
    val = out.compute(N_POS, {"a": at, "b": bt})
    assert torch.allclose(val, at * bt, rtol=1e-5, atol=1e-3)


def test_multiply_dead_lane_exact_zero_bit_exact_fp32():
    """At |a| = 100, fp32 σ(-|a|) computes as exactly 0.0 (the pinned
    losing-branch zero: e^100 overflows fp32), so the dead lane
    contributes nothing and the live lane is a·b in one rounding —
    bit-identical to the reference.  (At merely-saturated |a| ≈ 17-88
    the dead lane's σ is a representable ~e^{-|a|}, leaving ulp-class
    relative error — covered by the allclose tests above.)
    """
    at = torch.tensor([100.0, -100.0, 150.0, -200.0]).unsqueeze(1)
    bt = torch.tensor([3.25, -0.5, -123.456, 0.875]).unsqueeze(1)
    a = create_input("a", 1, value_range=(-200.0, 200.0))
    b = create_input("b", 1, value_range=(-200.0, 200.0))
    out = multiply(a, b)
    val = out.compute(4, {"a": at, "b": bt})
    assert torch.equal(val, at * bt)


def test_multiply_rejects_vector_inputs():
    a = create_input("a", 2, value_range=(-1.0, 1.0))
    b = create_input("b", 1, value_range=(-1.0, 1.0))
    with pytest.raises(AssertionError, match="1D scalar"):
        multiply(a, b)


def test_square_structure_and_oracle():
    x = create_input("x", 1, value_range=(-50.0, 50.0))
    out = square(x)
    assert isinstance(out, FFN)
    assert out.activation == "swish"
    assert not out.is_degenerate
    assert out.n_lanes == 2

    g = torch.Generator().manual_seed(13)
    xt = torch.rand(N_POS, 1, generator=g) * 100.0 - 50.0
    val = out.compute(N_POS, {"x": xt})
    assert torch.allclose(val, xt * xt, rtol=1e-5, atol=1e-6)
    # Both lane terms are x²·σ(±x) ≥ 0: never negative.
    assert (val >= 0).all()


def test_square_near_zero_relative_error_gone():
    """The ReLU-era piecewise square was worst near zero (231x rel error
    in the noise table); the gated form stays ~ulp-relative there.
    """
    xt = torch.linspace(-0.1, 0.1, 201).unsqueeze(1)
    x = create_input("x", 1, value_range=(-1.0, 1.0))
    val = square(x).compute(201, {"x": xt})
    ref = xt * xt
    mask = ref.abs() >= 1e-8
    rel = ((val - ref).abs()[mask] / ref[mask]).max()
    assert rel < 1e-5


@pytest.mark.parametrize("op_name", ["multiply", "square"])
def test_compiles_clean(op_name):
    if op_name == "multiply":
        a = create_input("a", 1, value_range=(-10.0, 10.0))
        b = create_input("b", 1, value_range=(-10.0, 10.0))
        out = multiply(a, b)
        at, bt = _pair(seed=23)
        inputs = {"a": at, "b": bt}
    else:
        x = create_input("x", 1, value_range=(-10.0, 10.0))
        out = square(x)
        g = torch.Generator().manual_seed(29)
        inputs = {"x": torch.rand(N_POS, 1, generator=g) * 20.0 - 10.0}

    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    report = probe_compiled(compiled, out, inputs, N_POS, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


# ---------------------------------------------------------------------------
# abs / min
# ---------------------------------------------------------------------------


def test_abs_structure_and_integer_grid_bit_exact():
    """2·d degenerate lanes; for |x| >= 0.2 tanh saturates and abs is
    bit-exact in fp32 — the entire integer grid (pinned in
    test_swish_constants::test_abs_integer_grid_bit_exact_fp32).
    """
    from torchwright.ops.swiglu import abs as abs_op

    x = create_input("x", 2, value_range=(-1000.0, 1000.0))
    out = abs_op(x)
    assert isinstance(out, FFN)
    assert out.is_degenerate
    assert out.n_lanes == 4
    xs = torch.tensor([[1.0, -1.0], [-737.0, 0.25], [0.0, -1000.0]])
    val = out.compute(3, {"x": xs})
    assert torch.equal(val, xs.abs())


def test_abs_one_sided_dip_near_zero():
    """Output lies in [0, |x|]: never negative, never above the true
    value; worst underestimate 2·swish_dip/scale at |x| = 1.278/scale.
    """
    from torchwright.ops.const import scale, swish_dip
    from torchwright.ops.swiglu import abs as abs_op

    x = create_input("x", 1, value_range=(-1.0, 1.0))
    out = abs_op(x)
    xs = torch.linspace(-0.2, 0.2, 4001).unsqueeze(1)
    val = out.compute(len(xs), {"x": xs})
    err = xs.abs() - val
    assert (val >= 0).all()
    assert (err >= -1e-7).all()  # one-sided: never above |x|
    peak = 2.0 * swish_dip / scale
    assert err.max().item() <= peak + 1e-7
    assert err.max().item() > 0.9 * peak  # the dip is really there


def test_min_ties_and_integer_far_field():
    from torchwright.ops.swiglu import min as min_op

    a = create_input("a", 1, value_range=(-100.0, 100.0))
    b = create_input("b", 1, value_range=(-100.0, 100.0))
    out = min_op(a, b)
    ffn = out
    assert ffn.is_degenerate
    assert ffn.n_lanes == 3
    # Ties exact (hinge at its bend, Swish(0)=0); far-apart integers
    # carry the folded-/scale product-rounding class, a few output ulps.
    at = torch.tensor([[3.0], [-2.0], [0.0], [7.0], [-5.0]])
    bt = torch.tensor([[3.0], [-2.0], [0.0], [1.0], [4.0]])
    val = out.compute(5, {"a": at, "b": bt})
    ref = torch.minimum(at, bt)
    assert torch.equal(val[:3], ref[:3])  # ties bit-exact
    assert torch.allclose(val, ref, rtol=1e-6, atol=1e-6)


def test_min_one_sided_overestimate_and_symmetry():
    """Min is over-estimated by at most swish_dip/scale, only when
    |a-b| is small; and min(a,b) == min(b,a) despite the asymmetric
    construction (the hinge's gap to ReLU is even in a-b).
    """
    from torchwright.ops.const import scale, swish_dip
    from torchwright.ops.swiglu import min as min_op

    a = create_input("a", 1, value_range=(-2.0, 2.0))
    b = create_input("b", 1, value_range=(-2.0, 2.0))
    out_ab = min_op(a, b)
    out_ba = min_op(b, a)
    g = torch.Generator().manual_seed(61)
    at = torch.rand(2001, 1, generator=g) * 4.0 - 2.0
    bt = at + torch.linspace(-0.3, 0.3, 2001).unsqueeze(1)  # sweep the tie zone
    inputs = {"a": at, "b": bt}
    val = out_ab.compute(2001, inputs)
    ref = torch.minimum(at, bt)
    err = val - ref
    assert (err >= -1e-6).all()  # one-sided: never under-reads
    assert err.max().item() <= swish_dip / scale + 1e-6
    val_swapped = out_ba.compute(2001, inputs)
    assert torch.allclose(val, val_swapped, atol=1e-6, rtol=0.0)


def test_abs_min_compile_clean():
    from torchwright.ops.swiglu import abs as abs_op
    from torchwright.ops.swiglu import min as min_op

    a = create_input("a", 2, value_range=(-50.0, 50.0))
    b = create_input("b", 2, value_range=(-50.0, 50.0))
    out = min_op(abs_op(a), b)
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    g = torch.Generator().manual_seed(67)
    inputs = {
        "a": torch.rand(8, 2, generator=g) * 100.0 - 50.0,
        "b": torch.rand(8, 2, generator=g) * 100.0 - 50.0,
    }
    report = probe_compiled(compiled, out, inputs, 8, atol=1e-3)
    assert report.first_divergent is None, report.format_short()
