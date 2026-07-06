"""Unit layer for the univariate-subgraph collapse pass (D6).

Small staircase subgraphs on both machines, collapsed via ``lower()``
with the flag on, swept over the integer grid (± the plateau slack)
against the source graph's exact oracle.  Negative tests pin every
feasibility-gate decline path from docs/univariate_collapse_plan.md:
missing integer contract, over-budget plateau count, emitted-lane
overflow past a pre-screen that passes, non-staircase member, no depth
gain, and the kept depth-1 boundary member.
"""

import pytest
import torch

from torchwright.compiler.collapse import _PLATEAU_SLACK, scalar_sources
from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.lower import lower
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.debug.probe import reference_eval
from torchwright.graph.asserts import assert_in_range, assert_integer
from torchwright.graph.ffn import FFN
from torchwright.graph.linear import Linear
from torchwright.graph.misc import InputNode
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear import add, sum_nodes


def _ops(machine):
    if machine == "relu":
        from torchwright.ops.relu import arithmetic_ops as ops
    else:
        from torchwright.ops.swiglu import arithmetic_ops as ops
    return ops


def _staircase_chain(machine, lo=0.0, hi=9.0, integer=True):
    """A depth->=2 univariate chain of x: min(cmp(x,2.5), cmp(x,5.5)).

    Composed function: -1 for x <= 5, +1 for x >= 6 — a one-step
    staircase reached through a multi-sublayer chain (two compares, the
    min's subtract/abs/rescale).
    """
    ops = _ops(machine)
    x = create_input("x", 1, value_range=(lo, hi))
    xi = assert_integer(x) if integer else x
    a = ops.compare(xi, 2.5)
    b = ops.compare(xi, 5.5)
    return xi, ops.min(a, b)


def _collapse(out, lane_cap=64):
    return lower(out, collapse_univariate=True, collapse_lane_cap=lane_cap)


def _only_outcome(lowered, source_hint="x"):
    outcomes = [o for o in lowered.collapse_report.outcomes if source_hint in o.source]
    assert len(outcomes) == 1, lowered.collapse_report.format()
    return outcomes[0]


# ---------------------------------------------------------------------------
# Positive path: collapse happens, values match the oracle on plateaus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("machine", ["relu", "swish"])
def test_collapse_matches_oracle_on_plateaus(machine):
    _, out = _staircase_chain(machine)
    ks = torch.arange(0.0, 10.0).unsqueeze(1)
    want = reference_eval(out, {"x": ks}, 10)[out]

    lowered = _collapse(out)
    assert _only_outcome(lowered).collapsed, lowered.collapse_report.format()

    # Structural depth: the collapsed output is one FFN reading the
    # source input directly (chain length -> 1).
    new_out = lowered.output_node
    assert isinstance(new_out, FFN)
    assert isinstance(new_out.inputs[0], InputNode)

    # Bit-for-bit-ish parity at plateau centers (the folded-projection
    # ulp class is the only slack the construction has).
    got = reference_eval(new_out, {"x": ks}, 10)[new_out]
    torch.testing.assert_close(got, want, atol=1e-5, rtol=0.0)

    # Constant across the whole plateau slack band by construction.
    # The band deliberately violates the source's integer contract
    # (x = k ± slack), so suppress its attached checks for the sweep.
    from torchwright.graph.node import suppress_checks

    offs = torch.tensor([-_PLATEAU_SLACK, _PLATEAU_SLACK])
    grid = (ks.unsqueeze(1) + offs.view(1, -1, 1)).reshape(-1, 1)
    with suppress_checks():
        banded = reference_eval(new_out, {"x": grid}, 20)[new_out].reshape(10, 2, -1)
    torch.testing.assert_close(
        banded, want.unsqueeze(1).expand_as(banded), atol=1e-5, rtol=0.0
    )


@pytest.mark.parametrize("machine", ["relu", "swish"])
def test_emitted_lane_count_is_two_per_changing_step(machine):
    """A 9-step full staircase costs exactly 2 lanes per value change."""
    ops = _ops(machine)
    x = create_input("x", 1, value_range=(0.0, 9.0))
    xi = assert_integer(x)
    steps = [ops.compare(xi, k + 0.5) for k in range(9)]
    out = sum_nodes(steps)  # f(k) = 2k - 9: distinct on every plateau

    lowered = _collapse(out, lane_cap=64)
    outcome = _only_outcome(lowered)
    assert outcome.collapsed, lowered.collapse_report.format()
    assert outcome.emitted_lanes == 18
    assert isinstance(lowered.output_node, FFN)
    assert lowered.output_node.n_lanes == 18

    ks = torch.arange(0.0, 10.0).unsqueeze(1)
    want = reference_eval(out, {"x": ks}, 10)[out]
    got = reference_eval(lowered.output_node, {"x": ks}, 10)[lowered.output_node]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=0.0)


def test_kept_depth1_boundary_member():
    """A boundary member at depth 1 is kept as-is, not re-synthesized."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    xi = assert_integer(x)
    m1 = ops.compare(xi, 2.5)  # depth 1, consumed outside the subgraph
    m2 = ops.min(m1, ops.compare(xi, 5.5))  # depth >= 2
    y = create_input("y", 1, value_range=(-1.0, 1.0))
    out = add(m2, add(m1, y))

    lowered = _collapse(out)
    outcome = _only_outcome(lowered)
    assert outcome.collapsed
    assert outcome.n_kept == 1

    m1_copy = lowered.copy_of(m1)
    m2_copy = lowered.copy_of(m2)
    live = get_ancestor_nodes({lowered.output_node})
    assert m1_copy in live and not m1_copy.name.startswith("collapse_")
    assert m2_copy in live and m2_copy.name.startswith("collapse_")


def test_interior_member_value_ceases_to_exist():
    """Interior members are orphaned; their node_map entries drop, the
    same contract fusion orphans already have."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    xi = assert_integer(x)
    interior = assert_in_range(ops.compare(xi, 2.5), -1.0, 1.0)
    out = ops.min(interior, ops.compare(xi, 5.5))

    lowered = _collapse(out)
    assert _only_outcome(lowered).collapsed
    with pytest.raises(KeyError):
        lowered.copy_of(interior)


def test_flag_off_is_a_noop():
    _, out = _staircase_chain("relu")
    lowered = lower(out)
    assert lowered.collapse_report is None
    assert not any(
        (n.name or "").startswith("collapse_")
        for n in get_ancestor_nodes({lowered.output_node})
    )


# ---------------------------------------------------------------------------
# Decline paths — every gate, in order
# ---------------------------------------------------------------------------


def _decline_reasons(lowered):
    return [o.reason for o in lowered.collapse_report.outcomes if not o.collapsed]


def test_declines_source_without_integer_assert():
    _, out = _staircase_chain("relu", integer=False)
    lowered = _collapse(out)
    assert lowered.collapse_report.n_collapsed == 0
    assert any("assert_integer" in r for r in _decline_reasons(lowered))


def test_declines_over_budget_plateau_count():
    _, out = _staircase_chain("relu", hi=1.0e6)
    lowered = _collapse(out, lane_cap=64)
    assert lowered.collapse_report.n_collapsed == 0
    assert any("plateaus exceed" in r for r in _decline_reasons(lowered))


def test_declines_emitted_lane_overflow_past_prescreen():
    """10 plateaus pass a 12-lane pre-screen, but 9 changing steps emit
    18 lanes — the post-tabulation lane gate must decline."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    xi = assert_integer(x)
    out = sum_nodes([ops.compare(xi, k + 0.5) for k in range(9)])

    lowered = _collapse(out, lane_cap=12)
    assert lowered.collapse_report.n_collapsed == 0
    assert any("18 lanes" in r for r in _decline_reasons(lowered))


def test_declines_predicted_accumulation_error():
    """Large plateau steps break the error model's budget: the staircase's
    fp32 lane-sum error is ~ulp(step_sharpness * R * max_step), and a
    +-100 plateau step over range 9 puts the 4-ulp bound (~7.8e-3) past
    the synthesized claim's 1e-3 tolerance — declined, per the measured
    staircase entries in docs/op_noise_data.json.  The +-1 chain is
    bit-constant on plateaus and the x100 Linear is exact in fp32, so
    this reaches the error-model gate rather than the staircase check."""
    _, chain = _staircase_chain("relu")  # bit-exact +-1 staircase of x
    out = Linear(chain, torch.tensor([[100.0]]))

    lowered = _collapse(out, lane_cap=64)
    assert lowered.collapse_report.n_collapsed == 0
    assert any(
        "predicted fp32 accumulation" in r for r in _decline_reasons(lowered)
    ), lowered.collapse_report.format()


def test_declines_non_staircase_member():
    """|x - 4.5| varies inside every plateau by the full band offset
    (0.05, far past the 1e-3 budget) — not a function of round(x); the
    measured-deviation check must decline it."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    xi = assert_integer(x)
    shifted = Linear(xi, torch.tensor([[1.0]]), torch.tensor([-4.5]))
    out = ops.abs(shifted)

    lowered = _collapse(out)
    assert lowered.collapse_report.n_collapsed == 0
    assert any("not constant on the plateau" in r for r in _decline_reasons(lowered))


def test_collapses_saturating_min_over_interior_variation():
    """min(|x - 4.5|, cmp(x, 5.5)) saturates: the interior |x - 4.5|
    varies in-band, but the *boundary* member is constant on every
    plateau up to fp32 associativity residue (ulp-scale), which the
    composite budget measures and admits.  Under the pre-composite
    bit-identical contract this was declined — the pin for the 2026-07-06
    contract change."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    xi = assert_integer(x)
    shifted = Linear(xi, torch.tensor([[1.0]]), torch.tensor([-4.5]))
    out = ops.min(ops.abs(shifted), ops.compare(xi, 5.5))

    ks = torch.arange(0.0, 10.0).unsqueeze(1)
    want = reference_eval(out, {"x": ks}, 10)[out]

    lowered = _collapse(out)
    assert _only_outcome(lowered).collapsed, lowered.collapse_report.format()
    got = reference_eval(lowered.output_node, {"x": ks}, 10)[lowered.output_node]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=0.0)


def test_collapses_sub_budget_band_deviation():
    """A staircase plus a tiny linear leak (1e-5·x) deviates from
    plateau-constancy by 1e-5·slack = 5e-7 — measured and charged
    against the composite budget (band deviation + modeled accumulation
    <= the synthesized claim's 1e-3), and admitted.  This is the swish
    fillet-tail shape (ulp-scale band residue) in a machine-independent
    construction; the collapse replaces the leak with its
    plateau-center value."""
    xi, chain = _staircase_chain("relu")
    out = add(chain, Linear(xi, torch.tensor([[1e-5]])))

    ks = torch.arange(0.0, 10.0).unsqueeze(1)
    want = reference_eval(out, {"x": ks}, 10)[out]

    lowered = _collapse(out)
    assert _only_outcome(lowered).collapsed, lowered.collapse_report.format()
    got = reference_eval(lowered.output_node, {"x": ks}, 10)[lowered.output_node]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=0.0)


def test_declines_composite_budget_overflow():
    """Each error term fits the budget alone; their sum does not.
    Plateau steps of ±15 over range 9 put the modeled fp32 accumulation
    at 4·ulp32(10·9·30) = 2^-10 ≈ 9.8e-4 (< 1e-3), and a 1e-3·x leak
    adds a measured band deviation of 1e-3·slack = 5e-5 (< 1e-3);
    together they exceed the synthesized claim's tolerance and the
    composite gate declines."""
    xi, chain = _staircase_chain("relu")
    out = add(
        Linear(chain, torch.tensor([[15.0]])),
        Linear(xi, torch.tensor([[1e-3]])),
    )

    lowered = _collapse(out)
    assert lowered.collapse_report.n_collapsed == 0
    assert any(
        "band deviation" in r for r in _decline_reasons(lowered)
    ), lowered.collapse_report.format()


def test_declines_no_depth_gain():
    """A subgraph whose only boundary member sits at depth 1 has nothing
    to collapse."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    xi = assert_integer(x)
    m = ops.compare(xi, 2.5)  # depth 1, and it is the output
    lowered = _collapse(m)
    assert lowered.collapse_report.n_collapsed == 0
    assert any("no depth gain" in r for r in _decline_reasons(lowered))


# ---------------------------------------------------------------------------
# End to end: the flag threads through compile_headless and saves layers
# ---------------------------------------------------------------------------


def test_compile_headless_collapse_saves_layers_and_matches():
    from torchwright.compiler.export import compile_headless

    inputs = torch.arange(0.0, 10.0).unsqueeze(1)  # the single "x" column

    _, out_a = _staircase_chain("relu")
    oracle = reference_eval(out_a, {"x": inputs}, 10)[out_a]
    # Explicit False: the baseline must stay the off-path even after the
    # default flips (and during flag-forced-on sweeps).
    baseline = compile_headless(
        out_a, d=64, d_head=8, verbose=False, collapse_univariate=False
    )
    _, out_b = _staircase_chain("relu")
    collapsed = compile_headless(
        out_b, d=64, d_head=8, verbose=False, collapse_univariate=True
    )

    assert collapsed.n_layers < baseline.n_layers
    # Each backend within compiled-path noise of the exact oracle (the
    # residual-stream matmul writes cost ~1e-5 on top of per-op noise;
    # values here are +-1, steps of size 2).
    torch.testing.assert_close(baseline(inputs).cpu(), oracle, atol=1e-4, rtol=0.0)
    torch.testing.assert_close(collapsed(inputs).cpu(), oracle, atol=1e-4, rtol=0.0)


# ---------------------------------------------------------------------------
# Finder unit
# ---------------------------------------------------------------------------


def test_scalar_sources_reseeds_at_two_source_meet():
    """Checks are node metadata, so the finder walks the ops-layer
    graph directly — a claim never interrupts a subgraph."""
    ops = _ops("relu")
    x = create_input("x", 1, value_range=(0.0, 9.0))
    y = create_input("y", 1, value_range=(0.0, 9.0))
    fx = ops.compare(x, 2.5)
    gy = ops.compare(y, 2.5)
    mix = add(fx, gy)  # two sources meet: ends both subgraphs, reseeds
    deeper = ops.compare(mix, 0.0)

    src = scalar_sources(topological_order(deeper))
    assert src[fx] is x
    assert src[gy] is y
    assert src[mix] is mix  # 1-D reseed
    assert src[deeper] is mix
