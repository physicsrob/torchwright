"""Unit set for the v2 S1 emitter (torchwright/compiler/collapse_pl.py).

The pass synthesizes one interpolating ``piecewise_linear`` FFN per
certified boundary member of the univariate subgraphs v1 leaves —
strict policy, production budget, S1 shape only (the Phase B descope).
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.lower import lower
from torchwright.debug.probe import reference_eval
from torchwright.graph.asserts import debug_watch
from torchwright.graph.misc import Concatenate, LiteralValue
from torchwright.graph.node import suppress_checks
from torchwright.ops.inout_nodes import create_input


def _ops(machine):
    if machine == "relu":
        from torchwright.ops.relu import arithmetic_ops as ops
    else:
        from torchwright.ops.swiglu import arithmetic_ops as ops
    return ops


def _eval(node, xs):
    with suppress_checks():
        vals = reference_eval(node, {"x": xs.float().reshape(-1, 1)}, xs.numel())
    return vals[node].to(torch.float64)


def _chain(machine, sharpness=50.0):
    """add_const -> compare -> add_const: a depth-3 univariate chain.

    A continuous source can never hand this chain to v1 (no integer claim).
    """
    ops = _ops(machine)
    x = create_input("x", 1, value_range=(0.0, 10.0))
    c = ops.compare(ops.add_const(x, 1.0), 5.0, sharpness=sharpness)
    return x, ops.add_const(c, 2.0)


def test_s1_take_matches_source_both_machines():
    for machine in ("relu", "swish"):
        _x, out = _chain(machine)
        lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
        rep = lowered.collapse_pl_report
        assert rep is not None
        assert rep.n_collapsed == 1, (machine, rep.format())
        xs = torch.linspace(0.0, 10.0, 4001, dtype=torch.float64)
        err = (_eval(lowered.output_node, xs) - _eval(out, xs)).abs().amax(dim=1)
        # Outside the compare's transition window (step at x=4, designed
        # ramp 1/50 wide plus the hinge bands) the claim budget holds;
        # inside it the emission inherits the ramp GEOMETRY but not the
        # source's exact silu curve — the reported in-band class,
        # bounded well under the step height.
        outside = (xs - 4.0).abs() > 0.05
        assert float(err[outside].max()) < 2.5e-3, (machine, float(err[outside].max()))
        assert float(err.max()) < 0.5, (machine, float(err.max()))


def test_pl_takes_what_v1_leaves():
    """v1 (staircase, integer-gated) declines the continuous chain.

    The pl pass takes it in the same lower() run.
    """
    _x, out = _chain("swish")
    lowered = lower(
        out, collapse_univariate=True, collapse_pl=True, collapse_lane_cap=64
    )
    assert lowered.collapse_report is not None
    assert lowered.collapse_report.n_collapsed == 0  # no integer claim
    assert lowered.collapse_pl_report.n_collapsed == 1


def test_same_source_multiply_declines():
    ops = _ops("swish")
    x = create_input("x", 1, value_range=(1.0, 4.0))
    m = ops.multiply(ops.add_const(x, 1.0), ops.add_const(x, 2.0))
    out = ops.add_const(m, 0.5)
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    (o,) = [o for o in lowered.collapse_pl_report.outcomes if o.chain_depth >= 2]
    assert not o.collapsed
    assert "not PL within budget" in o.reason, o.reason


def test_lane_cap_declines_s1():
    # Candidates (19 knots) stay under the kink pre-screen (4 x cap =
    # 32) so the decline exercises the true S1 lane gate, not the
    # pre-screen in front of it; quadratic knot values give every
    # segment a distinct slope, so nothing simplifies below the cap.
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 100.0))
    knots = [float(k) for k in range(1, 20)]
    p = ops.piecewise_linear(x, knots, lambda t: t * t / 10.0, d_max=4096)
    out = ops.add_const(p, 1.0)
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=8)
    (o,) = [o for o in lowered.collapse_pl_report.outcomes if o.chain_depth >= 2]
    assert not o.collapsed
    assert "S1 inadmissible" in o.reason, o.reason


def test_kink_prescreen_declines_before_the_sweep():
    """A member whose candidate-kink population exceeds 4 x lane_cap declines early.

    It declines at the pre-screen (the kink-explosion seam, before the
    member's oracle sweep) — the suite-cost mitigation for the default
    flip. ~98 candidates (quadratic values: one gate lane per knot)
    against cap 8 (screen 32).
    """
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 100.0))
    knots = [float(k) for k in range(1, 99)]
    p = ops.piecewise_linear(x, knots, lambda t: t * t / 10.0, d_max=4096)
    out = ops.add_const(p, 1.0)
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=8)
    (o,) = [o for o in lowered.collapse_pl_report.outcomes if o.chain_depth >= 2]
    assert not o.collapsed
    assert "kink explosion" in o.reason, o.reason


def test_no_depth_gain_declines():
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 10.0))
    out = ops.add_const(x, 2.0)  # depth-1 chain: nothing to gain
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    assert lowered.collapse_pl_report.n_collapsed == 0
    assert all("no depth gain" in o.reason for o in lowered.collapse_pl_report.outcomes)


def test_orphaned_checks_are_counted():
    _x, out = _chain("relu")
    # A watch on the interior compare member: the rewiring orphans it.
    interior = out.inputs[0]
    debug_watch(interior, lambda _v: (True, ""), "interior watch")
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    (o,) = [o for o in lowered.collapse_pl_report.outcomes if o.collapsed]
    assert o.n_checks_orphaned >= 1


def test_vector_valued_boundary_member():
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 8.0))
    # compare (an FFN) as the interior member: linear fusion cannot
    # fold it into the vector piecewise_linear, so the chain stays
    # depth 2 into the pass.
    a = ops.compare(x, 3.0, true_level=1.0, false_level=0.0, sharpness=50.0)
    out = ops.piecewise_linear(a, [0.25, 0.5, 0.75], lambda t: [t, 2.0 * t - 1.0])
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    assert lowered.collapse_pl_report.n_collapsed == 1
    xs = torch.linspace(0.0, 8.0, 1601, dtype=torch.float64)
    err = (_eval(lowered.output_node, xs) - _eval(out, xs)).abs().max()
    assert float(err) < 2.5e-3, float(err)
    assert lowered.output_node.d_output == 2


def test_node_map_follows_synthesized_value():
    _x, out = _chain("relu")
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    survivor = lowered.copy_of(out)
    assert (survivor.name or "").startswith("collapse_pl_"), survivor.name


def test_flag_off_is_inert():
    _x, out = _chain("relu")
    lowered = lower(out)
    assert lowered.collapse_pl_report is None


def test_emitted_step_survives_shallow_crossing_bands():
    """The regression the emitter's first sweep caught.

    A swish compare carries a shallow-slope gate crossing whose analytic
    band, unclamped, spans the whole domain — every sample classified as
    fillet, the member 'certified' trivially, and the emitted skeleton
    was a straight chord through the step. The locality bound on bands
    (pl_function) plus emitted-vs-ORIGINAL verification (collapse_pl)
    each independently prevent it; the value sweep pins the end-to-end
    behavior.
    """
    _x, out = _chain("swish", sharpness=50.0)
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    xs = torch.linspace(0.0, 10.0, 4001, dtype=torch.float64)
    err = (_eval(lowered.output_node, xs) - _eval(out, xs)).abs().amax(dim=1)
    # The step at x=4 must survive: the chord regression read ~0.8 on
    # the plateaus (mid-domain), the healthy emission reads the claim
    # budget there.
    outside = (xs - 4.0).abs() > 0.05
    assert float(err[outside].max()) < 2.5e-3, float(err[outside].max())


def test_output_concat_with_literal_field_compiles():
    """The doom emit-row shape (flag-on sweep regression, 2026-07-06).

    The OUTPUT node is a Concatenate whose whole row is univariate in one
    source — constant fields are literal leaves — so the pass synthesizes
    the Concatenate as a unit and the leaves orphan with the interior.
    The source-facing output gather must then use the synthesized
    member's direct residual entry; flattening the source Concatenate
    into its (now orphaned) leaves was a KeyError.
    """
    x = create_input("x", 1, value_range=(0.0, 10.0))
    ops = _ops("relu")
    chain = ops.add_const(ops.compare(ops.add_const(x, 1.0), 5.0, sharpness=50.0), 2.0)
    lit = LiteralValue(torch.tensor([3.0, -1.0]), name="const_field")
    out = Concatenate([lit, chain])

    # Pin the shape: the pass takes the concat as one synthesized unit
    # (a future decline here would make the compile check vacuous).
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    assert lowered.collapse_pl_report.n_collapsed == 1
    assert (lowered.output_node.name or "").startswith("collapse_pl_")

    compiled = compile_headless(out, d=64, d_head=8, device="cpu")
    inp = torch.tensor([[0.0], [7.0]], dtype=torch.float32)
    res = compiled(inp)
    with suppress_checks():
        want = reference_eval(out, {"x": inp}, 2)[out]
    assert float((res.cpu() - want).abs().max()) < 1e-3


def _steep_count_chain():
    """The mean -> count chain behind the 2026-07 calculator truncation.

    ``count_since_marker``'s reciprocal uses a geometric grid whose
    hinges at the steep end sit closer together than their fillet
    width.  Pre-fix, the per-hinge analytic bands chained into one
    interval spanning four real slope changes, ``band_skeleton``
    dropped those knots, and the S1 emission replaced the curve there
    with a single chord — 0.62 off, certified at the 1e-3 budget
    because the verifier never samples inside bands.
    """
    import math

    from torchwright.ops._math import _RECIP_REL_SAFETY
    from torchwright.ops.swiglu.arithmetic_ops import add_const, reciprocal

    max_gap = 13
    lo_v, hi_v = 1.0 / (max_gap + 1.5), 1.5
    target_rel = 0.5 / (max_gap + 1.0) / _RECIP_REL_SAFETY
    r_max = 1.0 + math.sqrt(8.0 * target_rel)
    n_bp = max(32, int(math.log(hi_v / lo_v) / math.log(r_max)) + 2)
    step = (hi_v - lo_v) / (n_bp - 1)
    x = create_input("x", 1, value_range=(lo_v, 1.0))
    return x, add_const(reciprocal(x, min_value=lo_v, max_value=hi_v, step=step), -1.0)


def test_steep_dense_grid_keeps_count_contract():
    """Collapsing a dense steep grid must not lose real corners.

    Whether the pass emits (with the corners kept) or declines (the
    chain keeps its original layers), the lowered graph must track the
    original reference at the count's operating points and across the
    steep end.  Thresholds sit far under the pre-fix 0.44/0.62 errors
    but leave room for legitimate in-band rounding differences.
    """
    _x, out = _steep_count_chain()
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=512)
    gaps = torch.tensor([1.0 / (g + 1.0) for g in range(14)], dtype=torch.float64)
    err = (_eval(lowered.output_node, gaps) - _eval(out, gaps)).abs()
    assert float(err.max()) < 5e-2, float(err.max())
    fine = torch.linspace(1.0 / 14.5, 0.13, 1001, dtype=torch.float64)
    err_fine = (_eval(lowered.output_node, fine) - _eval(out, fine)).abs()
    assert float(err_fine.max()) < 5e-2, float(err_fine.max())


def test_overlapping_hinge_bands_void_excusal():
    """Unit set for ``_excusable_bands`` (the band-chaining fix)."""
    from torchwright.compiler.pl_function import _excusable_bands

    def rows(*pairs):
        return torch.tensor(pairs, dtype=torch.float64)

    # An isolated band survives untouched.
    assert _excusable_bands(rows([0.0, 1.0])).tolist() == [[0.0, 1.0]]
    # Rows whose centers coincide within the fillet radius (a hinge
    # pair, an inherited duplicate) union into ONE band.
    assert _excusable_bands(rows([0.0, 1.0], [0.001, 1.001])).shape[0] == 1
    # Distinct corners whose bands overlap void each other's excusal.
    assert _excusable_bands(rows([0.0, 1.0], [0.8, 1.8])).shape[0] == 0
    # The calculator steep end: a chain of five overlapping bands all
    # void; the first non-overlapping neighbors survive.
    chain = rows(
        [0.0637, 0.0743],
        [0.0709, 0.0815],
        [0.0788, 0.0894],
        [0.0876, 0.0982],
        [0.0973, 0.1079],
        [0.1080, 0.1186],
        [0.1199, 0.1305],
    )
    kept = _excusable_bands(chain)
    assert kept.shape[0] == 2, kept.tolist()
    assert float(kept[0, 0]) == 0.1080
