"""Unit layer for the v2 PLFunction algebra (D6, Phase A).

The composed-function representation behind continuous-source collapse:
candidate kinks from weights (gate-row zero crossings pulled back
through the composed upstream function), values from the seeded exact
oracle, midpoint-linearity certificate with a near-knot ladder, and
the S1/S2 emission-shape models.  Pins: affine composition, relu-FFN
kinks recovered from weights, pullback through a monotone map, kink
multiplication through a sawtooth, saturation tails on a huge claimed
range, Add/Concatenate unions, vector-valued members, kink counts for
floor_int / sawtooth / table_lookup_2d / clamp, the keystone property
(PLFunction.eval vs reference_eval on both machines), and the
midpoint-linearity decline on a genuinely curved member.
"""

import pytest
import torch

from torchwright.compiler.collapse import scalar_sources
from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.pl_function import (
    PLFunction,
    certify_subgraph,
    model_s1,
    model_s2,
    transition_runs,
)
from torchwright.debug.probe import reference_eval
from torchwright.graph.linear import Linear
from torchwright.graph.misc import Concatenate
from torchwright.graph.node import suppress_checks
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear import add


def _ops(machine):
    if machine == "relu":
        from torchwright.ops.relu import arithmetic_ops as ops
    else:
        from torchwright.ops.swiglu import arithmetic_ops as ops
    return ops


def _certify(out, x, **kwargs):
    """Group ``x``'s univariate subgraph under ``out`` and certify it."""
    order = topological_order(out)
    src = scalar_sources(order)
    members = [n for n in order if src[n] is x and n is not x]
    return certify_subgraph(x, members, **kwargs)


def _interior_kinks(cert, node):
    return cert.members[node].fn.x[1:-1].tolist()


def _reference(node, xs):
    with suppress_checks():
        vals = reference_eval(node, {"x": xs.float().reshape(-1, 1)}, xs.numel())
    return vals[node].to(torch.float64)


# ---------------------------------------------------------------------------
# Algebra: affine composition, crossings
# ---------------------------------------------------------------------------


def test_affine_of_affine_closed_form():
    f = PLFunction.line(torch.tensor([2.0]), torch.tensor([1.0]), -5.0, 5.0)
    g = f.map_affine(torch.tensor([[3.0]]), torch.tensor([-4.0]))
    xs = torch.linspace(-5.0, 5.0, 11, dtype=torch.float64)
    torch.testing.assert_close(g.eval(xs)[:, 0], 6.0 * xs - 1.0)
    cross = g.zero_crossings()
    assert cross.numel() == 1
    assert abs(float(cross[0]) - 1.0 / 6.0) < 1e-12


def test_affine_graph_chain_has_no_kinks():
    x = create_input("x", 1, value_range=(-5.0, 5.0))
    a = Linear(x, torch.tensor([[2.0]]), torch.tensor([1.0]))
    b = Linear(a, torch.tensor([[3.0]]), torch.tensor([-4.0]))
    cert = _certify(b, x)
    assert cert.declined is None
    cb = cert.members[b]
    assert cb.n_kinks == 0
    assert cb.deviation < 1e-5
    xs = torch.linspace(-5.0, 5.0, 7, dtype=torch.float64)
    torch.testing.assert_close(
        cb.fn.eval(xs)[:, 0], 6.0 * xs - 1.0, atol=1e-4, rtol=1e-6
    )


def test_relu_ffn_kinks_recovered_from_weights():
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(-4.0, 6.0))
    bps = [-1.0, 0.5, 2.0]
    pl = ops.piecewise_linear(x, bps, lambda t: abs(t - 0.5))
    cert = _certify(pl, x)
    m = cert.members[pl]
    assert sorted(_interior_kinks(cert, pl)) == pytest.approx(bps)
    assert m.deviation < 1e-5
    xs = torch.linspace(-4.0, 6.0, 101, dtype=torch.float64)
    torch.testing.assert_close(m.fn.eval(xs), _reference(pl, xs), atol=2e-5, rtol=0.0)


def test_pullback_through_monotone_pl():
    """A downstream threshold pulls back through u = x/2 to x = 2·thresh."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 10.0))
    u = ops.piecewise_linear(x, [0.0, 10.0], lambda t: 0.5 * t)
    c = ops.compare(u, 2.0)  # hinges at u = 2.0 and u = 2.1
    cert = _certify(c, x)
    kk = sorted(_interior_kinks(cert, c))
    assert kk == pytest.approx([4.0, 4.2], abs=1e-4)
    assert cert.members[c].deviation < 1e-5


def _sawtooth_knots(n_teeth=3, period=4.0, drop=0.2):
    knots, vals = [], []
    for k in range(n_teeth):
        knots += [k * period, (k + 1) * period - drop]
        vals += [0.0, period - drop]
    knots.append(n_teeth * period)
    vals.append(0.0)
    return knots, vals


def test_sawtooth_pullback_multiplies_crossings():
    """One threshold crossing per monotone run of the sawtooth — the
    reason per-lane counts are not a valid kink bound."""
    knots, vals = _sawtooth_knots()
    saw = PLFunction(torch.tensor(knots), torch.tensor(vals).reshape(-1, 1), 0.0, 0.0)
    phi = saw.map_affine(torch.tensor([[1.0]]), torch.tensor([-2.0]))
    assert phi.zero_crossings().numel() == 6  # 3 rises + 3 falls


def test_sawtooth_graph_kink_pin():
    """Graph-level: compare-after-sawtooth candidates = the sawtooth's 5
    interior knots + 2 hinges x 6 monotone runs = 17."""
    ops = _ops("relu")
    knots, vals = _sawtooth_knots()
    x = create_input("x", 1, value_range=(0.0, 12.0))
    saw = ops.piecewise_linear(x, knots, dict(zip(knots, vals)).__getitem__)
    c = ops.compare(saw, 2.0)
    cert = _certify(c, x)
    assert cert.members[saw].n_kinks == 5
    assert cert.members[c].n_kinks == 17
    # Steep composed ramps carry fp32 position-quantization noise of
    # ~eps32(x)·|slope| (slope ≈ 380 here) — real machine behavior,
    # measured, and inside the budget.
    assert cert.members[c].deviation < 1e-3
    # S1 lane pin for the sawtooth itself: 2 slope changes per period
    # boundary plus the leading edge (7 total on 3 teeth).
    s1 = model_s1(cert.members[saw].fn, cert.members[saw].deviation)
    assert s1.lanes == 7


def test_saturation_tails_keep_kinks_finite():
    """A huge interval-arithmetic source range costs nothing: every kink
    lives inside the clamp contract and the tails are flat."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(-1e9, 1e9))
    c = ops.clamp(x, -100.0, 100.0)
    out = ops.compare(c, 3.0)
    cert = _certify(out, x)
    m = cert.members[out]
    assert m.n_kinks == 4  # clamp corners ±100, compare hinges 3.0/3.1
    assert m.deviation < 1e-4
    v = m.fn.eval(torch.tensor([-9e8, -101.0, 101.0, 9e8]))
    torch.testing.assert_close(v[0], v[1])
    torch.testing.assert_close(v[2], v[3])


def test_add_and_concat_union_kinks():
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    a = ops.compare(x, 2.5)
    b = ops.compare(x, 5.5)
    s = add(a, b)
    cat = Concatenate([a, b])
    out = Concatenate([s, cat])
    cert = _certify(out, x)
    union = pytest.approx([2.5, 2.6, 5.5, 5.6], abs=1e-6)
    assert sorted(_interior_kinks(cert, s)) == union
    assert cert.members[cat].fn.d == 2
    assert sorted(set(cert.members[cat].fn.x[1:-1].tolist())) == union
    ks = torch.arange(0.0, 10.0, dtype=torch.float64)
    torch.testing.assert_close(
        cert.members[s].fn.eval(ks), _reference(s, ks), atol=2e-5, rtol=0.0
    )
    torch.testing.assert_close(
        cert.members[cat].fn.eval(ks), _reference(cat, ks), atol=2e-5, rtol=0.0
    )


@pytest.mark.parametrize("machine", ["relu", "swish"])
def test_vector_valued_member(machine):
    ops = _ops(machine)
    x = create_input("x", 1, value_range=(0.0, 4.0))
    table = {1.0: [0.0, 1.0, 5.0], 2.0: [1.0, 1.0, 2.0], 3.0: [0.0, 3.0, 2.0]}
    pl = ops.piecewise_linear(x, [1.0, 2.0, 3.0], table.__getitem__)
    cert = _certify(pl, x)
    m = cert.members[pl]
    assert m.fn.d == 3
    assert sorted(_interior_kinks(cert, pl)) == pytest.approx([1.0, 2.0, 3.0])
    assert m.deviation < (1e-5 if machine == "relu" else 1e-3)


# ---------------------------------------------------------------------------
# Kink-count pins for the op constructions the doom chains are made of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("machine", ["relu", "swish"])
def test_kink_pin_clamp(machine):
    ops = _ops(machine)
    x = create_input("x", 1, value_range=(-500.0, 500.0))
    c = ops.clamp(x, -50.0, 70.0)
    cert = _certify(c, x)
    assert sorted(_interior_kinks(cert, c)) == pytest.approx([-50.0, 70.0])
    assert cert.members[c].deviation < 1e-3


def test_kink_pin_floor_int():
    """floor_int realizes 2 slope changes per boundary; its two-stage
    construction contributes at most 3 candidates per boundary."""
    ops = _ops("swish")
    x = create_input("x", 1, value_range=(0.0, 8.0))
    f = ops.floor_int(x, 0, 8)
    cert = _certify(f, x)
    m = cert.members[f]
    # 2 stage-1 + 1 stage-2 candidates per boundary; boundary 8's own
    # crossing and post-ramp edge sit at/past the domain end (clipped).
    assert m.n_kinks == 3 * 8 - 2
    assert m.deviation < 1e-3
    s1 = model_s1(m.fn, m.deviation)
    assert s1.lanes == 16  # 2 per boundary — the plan's pin
    s2 = model_s2(m.fn, m.deviation, machine="swish")
    assert s2.n_steps == 8
    assert s2.stage1_lanes == 16 and s2.stage1_cols == 8 and s2.stage2_lanes == 8


def test_kink_pin_table_lookup_2d():
    """(rows-1) row transitions + (cols-1) column transitions when the
    two index chains hit their half-integer boundaries at distinct x."""
    from torchwright.ops.swiglu.map_select import table_lookup_2d

    x = create_input("x", 1, value_range=(0.0, 6.0))
    j = Linear(x, torch.tensor([[0.35]]))
    table = [[10.0 * r + c for c in range(3)] for r in range(4)]
    out = table_lookup_2d(x, j, table)
    cert = _certify(out, x)
    m = cert.members[out]
    assert m.deviation < 1e-3
    s2 = model_s2(m.fn, m.deviation, machine="swish")
    assert s2.n_steps == (4 - 1) + (3 - 1)


# ---------------------------------------------------------------------------
# Keystone property: PLFunction matches reference_eval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("machine", ["relu", "swish"])
def test_keystone_random_chains_match_reference(machine):
    """Random small op chains: the certified PL function reproduces the
    exact oracle on a dense mid-segment grid — exactly on relu,
    fillet-bounded on swish."""
    from torchwright.ops.const import step_sharpness

    ops = _ops(machine)
    gen = torch.Generator().manual_seed(7)
    tol = 5e-5 if machine == "relu" else 2e-3
    for trial in range(3):
        x = create_input("x", 1, value_range=(0.0, 10.0))
        bps = torch.sort(torch.rand(5, generator=gen) * 8.0 + 1.0).values
        bps = [0.0] + [float(v) for v in bps]  # spacing can be tight; scale sharpens
        vals = {b: float(torch.rand(1, generator=gen) * 6.0 - 3.0) for b in bps}
        p1 = ops.piecewise_linear(x, bps, vals.__getitem__, input_scale=step_sharpness)
        thresh = float(torch.rand(1, generator=gen) * 4.0 - 2.0)
        c1 = ops.compare(p1, thresh)
        c2 = ops.compare(x, float(torch.rand(1, generator=gen) * 8.0 + 1.0))
        out = add(ops.min(c1, c2), p1)
        cert = _certify(out, x)
        assert cert.declined is None
        m = cert.members[out]
        assert m.deviation <= 1e-3, (trial, m.deviation, m.deviation_at)
        # Independent dense check at mid-segment fractions (0.3/0.5/0.7),
        # away from the swish corner fillets.
        knots = m.fn.x
        fracs = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        dense = (
            knots[:-1].unsqueeze(1)
            + fracs.unsqueeze(0) * (knots[1:] - knots[:-1]).unsqueeze(1)
        ).reshape(-1)
        dense = dense.float().double()  # fp32-representable positions
        got = m.fn.eval(dense)
        want = _reference(out, dense)
        assert float((got - want).abs().max()) <= tol, trial


# ---------------------------------------------------------------------------
# Declines and the deviation split
# ---------------------------------------------------------------------------


def test_same_source_multiply_declines_midpoint_linearity():
    """x·x is piecewise-quadratic: measured, located decline."""
    ops = _ops("swish")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    m = ops.multiply(x, x)
    cert = _certify(m, x)
    c = cert.members[m]
    assert not c.linear()
    assert c.deviation > 1.0  # chord vs x² mid-domain
    assert 0.0 < c.deviation_at < 9.0


def test_relu_square_is_pl_and_certifies():
    """The relu machine's square is a PL *approximation* — the same
    shape that declines as an exact product certifies as a table."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    s = ops.square(x, max_value=9.0)
    cert = _certify(s, x)
    assert cert.members[s].linear()


def test_swish_abs_curvature_lands_in_fillet_split():
    """|x|'s rounding near zero is fillet-class: reported in
    fillet_deviation, not charged against the chord certificate."""
    ops = _ops("swish")
    x = create_input("x", 1, value_range=(-5.0, 5.0))
    a = ops.abs(x)
    cert = _certify(a, x)
    c = cert.members[a]
    assert c.deviation < 1e-3  # mid-segment: the V is linear
    assert c.fillet_deviation > 1e-3  # the near-zero dip is real and visible


def test_kink_explosion_declines():
    ops = _ops("relu")
    knots, vals = _sawtooth_knots()
    x = create_input("x", 1, value_range=(0.0, 12.0))
    saw = ops.piecewise_linear(x, knots, dict(zip(knots, vals)).__getitem__)
    c = ops.compare(saw, 2.0)
    cert = _certify(c, x, max_kinks=3)
    assert cert.declined is not None and "kink explosion" in cert.declined


def test_unbounded_source_declines():
    import math

    x = create_input("x", 1, value_range=(-math.inf, math.inf))
    c = Linear(x, torch.tensor([[2.0]]))
    cert = _certify(c, x)
    assert cert.declined == "source value_range is unbounded"


# ---------------------------------------------------------------------------
# Emission-shape models on the doom-scale synthetic staircase
# ---------------------------------------------------------------------------


def _synthetic_staircase(n_steps=2046, ramp=1e-3):
    """A doom-floor-shaped staircase: unit steps at integer boundaries
    across [-1023, 1023], ramps ``ramp`` wide (slope 1/ramp)."""
    xs, ys = [-1023.5], [0.0]
    for k in range(n_steps):
        b = -1023.0 + k
        xs += [b, b + ramp]
        ys += [float(k), float(k + 1)]
    xs.append(1023.5)
    ys.append(float(n_steps))
    return PLFunction(torch.tensor(xs), torch.tensor(ys).reshape(-1, 1), 0.0, 0.0)


def test_s1_infeasible_s2_feasible_on_doom_scale_staircase():
    """The recon insight, pinned: fp32 kills the single-FFN emission of
    a sharp wide staircase; the bounded-step shape stays in budget."""
    fn = _synthetic_staircase()
    s1 = model_s1(fn, measured_dev=0.0)
    assert s1.lanes == 2 * 2046
    assert s1.fp_bound > 1e-3  # ulp32 of slope·span — hopeless
    assert not s1.admissible(lane_cap=8192)
    s2 = model_s2(fn, measured_dev=0.0, machine="relu")
    assert s2.n_steps == 2046
    assert s2.total_bound <= 1e-3
    assert s2.admissible(lane_cap=8192)
    assert not s2.admissible(lane_cap=1024)  # stage-1 lanes past the cap
    is_plateau, runs = transition_runs(fn)
    assert len(runs) == 2046


# ---------------------------------------------------------------------------
# Analysis mode (report-only) smoke: verdicts + modeled floor
# ---------------------------------------------------------------------------


def test_analysis_takes_v1_declined_continuous_chain():
    """A staircase chain with NO integer claim: v1 declines (gate 1),
    v2 certifies the composed PL directly and takes it as S1."""
    from torchwright.compiler.collapse_analysis import analyze_collapse_v2
    from torchwright.compiler.lower import lower

    ops = _ops("swish")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    out = ops.min(ops.compare(x, 2.5), ops.compare(x, 5.5))
    lowered = lower(out, collapse_univariate=True, collapse_lane_cap=64)
    assert lowered.collapse_report.n_collapsed == 0  # v1: no assert_integer
    report = analyze_collapse_v2(lowered.output_node, lane_cap=64)
    (sg,) = [s for s in report.subgraphs if s.source == "x"]
    assert sg.verdict == "S1", sg.format_line()
    assert report.floor_on < report.floor_off


def test_analysis_picks_s2_for_sharp_wide_staircase_chain():
    """A sharp wide floor chain: the single-FFN shape dies on fp32
    accumulation, the bounded-step shape is taken at chain -> 2."""
    from torchwright.compiler.collapse_analysis import analyze_collapse_v2
    from torchwright.compiler.lower import lower

    ops = _ops("swish")
    x = create_input("x", 1, value_range=(0.0, 600.0))
    f = ops.floor_int(x, 0, 600, sharpness=1000.0)
    out = ops.compare(f, 300.5)
    lowered = lower(out, collapse_univariate=True, collapse_lane_cap=4096)
    report = analyze_collapse_v2(lowered.output_node, lane_cap=4096)
    (sg,) = [s for s in report.subgraphs if s.n_synthesized]
    assert sg.verdict == "S2", sg.format_line()
    m = sg.members[0]
    assert not m.s1_ok and m.s2_ok  # fp32 kills S1; S2 fits
    assert sg.stage1_cols == 1  # one composed value transition
    assert report.floor_on == 2
    assert report.floor_on < report.floor_off


def test_s2_swish_fillet_is_reported_not_charged():
    """Per-step jumps of 100 (a colormap-scale delta) carry an in-band
    fillet excursion of 2·swish_dip/scale·|δ| — the class the original
    chain already has.  The model reports it and keeps it out of the
    charged bound."""
    xs = [0.0]
    ys = [0.0]
    for k in range(4):
        b = 1.0 + k
        xs += [b, b + 1e-3]
        ys += [k * 100.0, (k + 1) * 100.0]
    xs.append(6.0)
    ys.append(400.0)
    fn = PLFunction(torch.tensor(xs), torch.tensor(ys).reshape(-1, 1), 0.0, 0.0)
    s2 = model_s2(fn, measured_dev=0.0, machine="swish")
    assert s2.max_abs_delta == pytest.approx(100.0)
    assert s2.fillet_bound > 0.1
    assert s2.total_bound < 1e-3
    assert s2.admissible(lane_cap=4096)
