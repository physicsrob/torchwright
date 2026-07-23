"""swiglu piecewise_linear and its compositions (clamp, reciprocal,
thermometer_floor_div, mod_const, global_position_from_bos).

Spec: docs/ops_plain_english.md (piecewise_linear entry; the compositions
inherit it).  Fillet radius/dip magnitudes are pinned in
tests/docs/test_swish_constants.py (test_hinge_fillet_width).
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN, InputNode
from torchwright.ops.const import scale, swish_dip
from torchwright.ops.inout_nodes import create_input, create_rope_config
from torchwright.ops.swiglu import (
    clamp,
    global_position_from_bos,
    mod_const,
    piecewise_linear,
    reciprocal,
    thermometer_floor_div,
)

D = 64
D_HEAD = 8


def _unwrap(node):
    while not isinstance(node, FFN):
        node = node.inputs[0]
    return node


# ---------------------------------------------------------------------------
# piecewise_linear
# ---------------------------------------------------------------------------


def test_pl_knots_and_segment_interiors():
    """Away from the fillets (|x - x_i| > 17/K) the op computes the exact
    interpolation to the folded ulp class; knots included.
    """
    x = create_input("x", 1, value_range=(-2.0, 6.0))
    out = piecewise_linear(x, [0.0, 1.0, 3.0, 4.0], lambda v: v * v)
    xs = torch.tensor([[0.0], [1.0], [3.0], [4.0], [0.5], [2.0], [3.5]])
    val = out.compute(7, {"x": xs})
    # exact PL: knots exact; interiors interpolate knot values
    ref = torch.tensor([[0.0], [1.0], [9.0], [16.0], [0.5], [5.0], [12.5]])
    assert torch.allclose(val, ref, rtol=1e-6, atol=1e-6)


def test_pl_clamp_holds_outside_range():
    x = create_input("x", 1, value_range=(-10.0, 10.0))
    out = piecewise_linear(x, [0.0, 1.0, 2.0], lambda v: 2.0 * v)
    xs = torch.tensor([[-5.0], [7.0]])
    val = out.compute(2, {"x": xs})
    assert torch.allclose(val, torch.tensor([[0.0], [4.0]]), rtol=1e-6, atol=1e-6)


def test_pl_no_clamp_extrapolates():
    x = create_input("x", 1, value_range=(-10.0, 10.0))
    out = piecewise_linear(x, [0.0, 1.0, 2.0], lambda v: 2.0 * v, clamp=False)
    xs = torch.tensor([[-5.0], [7.0]])
    val = out.compute(2, {"x": xs})
    assert torch.allclose(val, torch.tensor([[-10.0], [14.0]]), rtol=1e-6, atol=1e-5)


def test_pl_fillet_dip_bounded_by_slope_change():
    """Inside a fillet the gap to the exact PL is ≤ swish_dip·|Δm|/K,
    and the fillet is really there.
    """
    x = create_input("x", 1, value_range=(-2.0, 4.0))
    # single corner at 1.0 with slope change 3.0 (0 -> 3)
    out = piecewise_linear(x, [-1.0, 1.0, 3.0], lambda v: 3.0 * max(v - 1.0, 0.0))
    xs = torch.linspace(0.8, 1.2, 4001).unsqueeze(1)
    val = out.compute(len(xs), {"x": xs})
    ref = 3.0 * torch.clamp(xs - 1.0, min=0.0)
    err = (val - ref).abs()
    bound = swish_dip * 3.0 / scale
    assert err.max().item() <= bound + 1e-6
    assert err.max().item() > 0.9 * bound


def test_pl_vector_fn_and_range_slack():
    x = create_input("x", 1, value_range=(-1.0, 3.0))
    out = piecewise_linear(x, [0.0, 1.0, 2.0], lambda v: [v, -2.0 * v])
    xs = torch.tensor([[0.5], [1.5]])
    val = out.compute(2, {"x": xs})
    assert torch.allclose(
        val, torch.tensor([[0.5, -1.0], [1.5, -3.0]]), rtol=1e-6, atol=1e-6
    )
    # Range claim: knot hull ± windowed stacked dip. Breakpoints 1.0
    # apart >> 34/K, so the slack is the single worst windowed sum.
    r = out.value_type.value_range
    assert r.lo < -4.0 and r.lo > -4.1  # -4 minus small slack
    assert r.hi > 2.0 and r.hi < 2.1


def test_pl_chunking_matches_single_ffn():
    """d_max chunking splits lanes across FFNs joined by sum_nodes; the
    math is identical.
    """
    bps = [float(k) for k in range(9)]

    def f(v):
        return float(v * v)

    x1 = create_input("x", 1, value_range=(-1.0, 9.0))
    whole = piecewise_linear(x1, bps, f)
    x2 = create_input("x", 1, value_range=(-1.0, 9.0))
    chunked = piecewise_linear(x2, bps, f, d_max=3)
    g = torch.Generator().manual_seed(71)
    xs = torch.rand(16, 1, generator=g) * 10.0 - 1.0
    v1 = whole.compute(16, {"x": xs})
    v2 = chunked.compute(16, {"x": xs})
    assert torch.allclose(v1, v2, rtol=1e-6, atol=1e-5)


def test_pl_compiles_clean():
    x = create_input("x", 1, value_range=(-1.0, 5.0))
    out = piecewise_linear(x, [0.0, 1.0, 2.0, 4.0], lambda v: 1.0 / (1.0 + v * v))
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    g = torch.Generator().manual_seed(73)
    xs = torch.rand(16, 1, generator=g) * 6.0 - 1.0
    report = probe_compiled(compiled, out, {"x": xs}, 16, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


# ---------------------------------------------------------------------------
# compositions
# ---------------------------------------------------------------------------


def test_clamp_identity_and_edges():
    x = create_input("x", 1, value_range=(-50.0, 50.0))
    out = clamp(x, 2.0, 8.0)
    xs = torch.tensor([[-50.0], [2.0], [5.0], [8.0], [50.0]])
    val = out.compute(5, {"x": xs})
    ref = torch.tensor([[2.0], [2.0], [5.0], [8.0], [8.0]])
    assert torch.allclose(val, ref, rtol=1e-6, atol=1e-5)


def test_reciprocal_accuracy_vs_true_function():
    """Smooth-target grid: fillets bend toward 1/x, so the dense low-end
    grid keeps the error in the relu class (measured entry is the
    authority; this is the sanity ceiling).
    """
    x = create_input("x", 1, value_range=(0.3, 200.0))
    out = reciprocal(x, min_value=0.3, max_value=200.0, step=1.0)
    g = torch.Generator().manual_seed(79)
    xs = torch.rand(2048, 1, generator=g) * 199.7 + 0.3
    val = out.compute(2048, {"x": xs})
    err = (val - 1.0 / xs).abs()
    assert err.max().item() < 5e-3, err.max().item()


def test_thermometer_floor_div_and_mod_const_exact_on_integers():
    x = create_input("x", 1, value_range=(0.0, 100.0))
    q = thermometer_floor_div(x, 10, 100)
    xs = torch.tensor([[0.0], [9.0], [10.0], [35.0], [99.0], [100.0]])
    val = q.compute(6, {"x": xs})
    ref = torch.floor(xs / 10.0)
    assert torch.allclose(val, ref, rtol=0.0, atol=1e-5)

    x2 = create_input("x", 1, value_range=(0.0, 100.0))
    m = mod_const(x2, 10, 100)
    val2 = m.compute(6, {"x": xs})
    assert torch.allclose(val2, xs - 10.0 * torch.floor(xs / 10.0), atol=1e-4)


def test_global_position_from_bos_integer_recovery():
    """Position recovery within the relu op's empirical ceiling (0.15;
    hard requirement is 0.5 for integer rounding).  The inversion table
    is the library's densest grid — this is its stacked-fillet audit.
    """
    rope = create_rope_config(d_head=256, max_positions=61440)
    n = 80
    bos_indicator = InputNode("bos", 1, value_range=(0.0, 1.0))
    pos = global_position_from_bos(rope, bos_indicator)
    bos_in = torch.zeros(n, 1)
    bos_in[0, 0] = 1.0
    result = pos.compute(n_pos=n, input_values={"bos": bos_in}).reshape(-1)
    errs = (result - torch.arange(n, dtype=torch.float32)).abs()
    assert errs.max().item() < 0.15, (
        f"max position error {errs.max().item():.4f} at pos {errs.argmax().item()}"
    )
