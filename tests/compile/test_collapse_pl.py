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
    """add_const -> compare -> add_const: depth-3 univariate chain a
    continuous source can never hand to v1 (no integer claim)."""
    ops = _ops(machine)
    x = create_input("x", 1, value_range=(0.0, 10.0))
    c = ops.compare(ops.add_const(x, 1.0), 5.0, sharpness=sharpness)
    return x, ops.add_const(c, 2.0)


def test_s1_take_matches_source_both_machines():
    for machine in ("relu", "swish"):
        x, out = _chain(machine)
        lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
        rep = lowered.collapse_pl_report
        assert rep is not None and rep.n_collapsed == 1, (machine, rep.format())
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
    """v1 (staircase, integer-gated) declines the continuous chain;
    the pl pass takes it in the same lower() run."""
    x, out = _chain("swish")
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
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 100.0))
    knots = [float(k) for k in range(1, 99)]
    p = ops.piecewise_linear(x, knots, lambda t: (t % 7.0) - 3.0, d_max=4096)
    out = ops.add_const(p, 1.0)
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=8)
    (o,) = [o for o in lowered.collapse_pl_report.outcomes if o.chain_depth >= 2]
    assert not o.collapsed
    assert "S1 inadmissible" in o.reason, o.reason


def test_no_depth_gain_declines():
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 10.0))
    out = ops.add_const(x, 2.0)  # depth-1 chain: nothing to gain
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    assert lowered.collapse_pl_report.n_collapsed == 0
    assert all("no depth gain" in o.reason for o in lowered.collapse_pl_report.outcomes)


def test_orphaned_checks_are_counted():
    x, out = _chain("relu")
    # A watch on the interior compare member: the rewiring orphans it.
    interior = out.inputs[0]
    debug_watch(interior, lambda v: (True, ""), "interior watch")
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
    x, out = _chain("relu")
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    survivor = lowered.copy_of(out)
    assert (survivor.name or "").startswith("collapse_pl_"), survivor.name


def test_flag_off_is_inert():
    x, out = _chain("relu")
    lowered = lower(out)
    assert lowered.collapse_pl_report is None


def test_emitted_step_survives_shallow_crossing_bands():
    """The regression the emitter's first sweep caught: a swish
    compare carries a shallow-slope gate crossing whose analytic band,
    unclamped, spans the whole domain — every sample classified as
    fillet, the member 'certified' trivially, and the emitted skeleton
    was a straight chord through the step.  The locality bound on
    bands (pl_function) plus emitted-vs-ORIGINAL verification
    (collapse_pl) each independently prevent it; the value sweep pins
    the end-to-end behavior."""
    x, out = _chain("swish", sharpness=50.0)
    lowered = lower(out, collapse_pl=True, collapse_lane_cap=64)
    xs = torch.linspace(0.0, 10.0, 4001, dtype=torch.float64)
    err = (_eval(lowered.output_node, xs) - _eval(out, xs)).abs().amax(dim=1)
    # The step at x=4 must survive: the chord regression read ~0.8 on
    # the plateaus (mid-domain), the healthy emission reads the claim
    # budget there.
    outside = (xs - 4.0).abs() > 0.05
    assert float(err[outside].max()) < 2.5e-3, float(err[outside].max())


def test_output_concat_with_literal_field_compiles():
    """The doom emit-row shape (flag-on sweep regression, 2026-07-06):
    the OUTPUT node is a Concatenate whose whole row is univariate in
    one source — constant fields are literal leaves — so the pass
    synthesizes the Concatenate as a unit and the leaves orphan with
    the interior.  The source-facing output gather must then use the
    synthesized member's direct residual entry; flattening the source
    Concatenate into its (now orphaned) leaves was a KeyError."""
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

    compiled = compile_headless(out, d=64, d_head=8, device="cpu", collapse_pl=True)
    inp = torch.tensor([[0.0], [7.0]], dtype=torch.float32)
    res = compiled(inp)
    with suppress_checks():
        want = reference_eval(out, {"x": inp}, 2)[out]
    assert float((res.cpu() - want).abs().max()) < 1e-3
