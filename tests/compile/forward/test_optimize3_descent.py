"""optimize=3 = in-compile iterated descent (CP-SAT expressiveness plan).

``optimize=3`` runs a 600 s iterated descent *inside a single*
``forward_compile``: rung 0 is the heuristic-hinted solve, each later rung
re-solves hinted with the best-so-far at horizon best+1 (a soft ceiling —
the plan is explicit a hard K-ceiling kills the hint), until the budget is
spent or a rung proves optimality; the shallowest schedule is returned.

This is uniform per compile — no cross-compile state — so a first-seen
graph behaves identically to a previously-seen one (Rob's no-preseed rule).
"""

import pytest
import torch

from torchwright.compiler.forward import compile as cmod
from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    ScheduleAssignment,
    SolveStats,
)
from torchwright.graph import Concatenate, Linear
from torchwright.ops.inout_nodes import create_input


def _width_graph():
    torch.manual_seed(1)
    x = create_input("x", 4)
    mids = []
    for i in range(6):
        li = Linear(x, torch.randn(4, 12), torch.zeros(12), name=f"L{i}")
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Ma{i}"))
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Mb{i}"))
    return Linear(Concatenate(mids), torch.randn(24, 4), torch.zeros(4), name="out")


def test_optimize3_compiles_replays_and_is_no_worse_than_optimize1():
    """A real optimize=3 compile produces a valid, replayable schedule no
    deeper than optimize=1 (the descent never regresses)."""
    graph = _width_graph()
    net3 = forward_compile(
        d=80, d_head=80, output_node=graph, device="cpu", verbose=False, optimize=3
    )
    net1 = forward_compile(
        d=80, d_head=80, output_node=_width_graph(), device="cpu",
        verbose=False, optimize=1,
    )
    assert len(net3.layers) <= len(net1.layers)

    inp = torch.randn(3, 4)
    ref = graph.compute(3, {"x": inp})
    got = net3.compute(3, {"x": inp})[graph]
    assert torch.allclose(got.cpu(), ref, atol=1e-3)


def test_optimize3_descent_budget_override_is_accepted():
    """The measurement-only ``_descent_budget_s`` overrides the total budget
    (production default 600s stays put) and still yields a valid solve."""
    graph = _width_graph()
    net = forward_compile(
        d=80, d_head=80, output_node=graph, device="cpu", verbose=False,
        optimize=3, _solve_only=True, _force_resolve=True, _descent_budget_s=5.0,
    )
    assert net.cpsat_assignment is not None
    assert net.cpsat_assignment.n_layers >= 1


def _fake_stats(n_layers, optimal):
    return SolveStats(
        status_name="OPTIMAL" if optimal else "FEASIBLE",
        objective_value=n_layers,
        best_objective_bound=float(n_layers if optimal else 3),
        wall_time_s=1.0,
        solver_log="",
        total_attn_heads=-1,
        total_mlp_bypass_slots=-1,
        is_optimal=optimal,
    )


def test_optimize3_runs_hinted_rungs_and_stops_on_optimal(monkeypatch):
    """The rung loop re-solves hinted with the best-so-far and stops when a
    rung proves optimality.  solve_schedule is stubbed to a deterministic
    improving sequence so the loop logic is tested exactly (``_solve_only``
    returns before the replay, so the fake schedule is never executed)."""
    # Deterministic improving sequence: 6 (feasible) -> 5 (feasible)
    # -> 5 (proven optimal, ends the descent).
    seq = [(6, False), (5, False), (5, True)]
    calls = []

    def fake_solve(output_node, pos_encoding=None, **kw):
        calls.append(kw)
        n, opt = seq[min(len(calls) - 1, len(seq) - 1)]
        asg = ScheduleAssignment(
            node_to_layer={1: n - 1},
            node_to_cancel_layer={1: n},  # parked at horizon -> dropped from hint
            node_to_routing={1: "mlp"},
            n_layers=n,
            node_to_cancel_mech={},
        )
        return asg, _fake_stats(n, opt)

    monkeypatch.setattr(cmod, "solve_schedule", fake_solve)

    net = forward_compile(
        d=80, d_head=80, output_node=_width_graph(), device="cpu",
        verbose=False, optimize=3, _solve_only=True, _force_resolve=True,
    )

    # Three rungs: 6, 5, 5(optimal) — stopped on the proven-optimal rung.
    assert len(calls) == 3
    # Rung 0 is heuristic-hinted (real warm start); rungs 1+ are hinted with
    # the previous rung's best node_to_layer.
    assert calls[1]["hint_layers"] == {1: 5}  # rung 0 returned n=6 -> {1: 5}
    assert calls[2]["hint_layers"] == {1: 4}  # rung 1 returned n=5 -> {1: 4}
    # Horizon is best+1 (soft ceiling), tightening as the incumbent drops.
    assert calls[1]["max_layers"] == 7  # incumbent 6 -> horizon 7
    assert calls[2]["max_layers"] == 6  # incumbent 5 -> horizon 6
    # Parked cancels (>= n_layers) are dropped from the hint.
    assert calls[1]["hint_cancel"] in (None, {})
    # Returns the shallowest, with proven-optimal status.
    assert net.cpsat_assignment.n_layers == 5
    assert net.cpsat_solve_stats.is_optimal


def test_optimize3_loop_is_bounded_when_never_optimal(monkeypatch):
    """A rung that never improves and never proves optimality does not spin
    forever: in production the 600 s wall budget stops it; the 64-rung safety
    cap is the backstop (which this instant-stub test exercises, since the
    stub consumes no wall time)."""
    calls = []

    def fake_solve(output_node, pos_encoding=None, **kw):
        calls.append(kw)
        asg = ScheduleAssignment(
            node_to_layer={1: 5},
            node_to_cancel_layer={1: 6},
            node_to_routing={1: "mlp"},
            n_layers=6,
            node_to_cancel_mech={},
        )
        return asg, _fake_stats(6, False)  # never improves, never optimal

    monkeypatch.setattr(cmod, "solve_schedule", fake_solve)

    net = forward_compile(
        d=80, d_head=80, output_node=_width_graph(), device="cpu",
        verbose=False, optimize=3, _solve_only=True, _force_resolve=True,
    )
    assert len(calls) == 64  # bounded by the safety cap, not infinite
    assert net.cpsat_assignment.n_layers == 6
