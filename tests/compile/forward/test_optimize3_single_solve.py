"""optimize=3 = one continuous 600 s solve (the production configuration).

``optimize=3`` runs a single hinted CP-SAT solve at the full 600 s budget
inside ``forward_compile`` — the same shape as levels 1/2, differing only
in budget.  It replaced the 2026-07-08 iterated descent (180 s rungs, each
re-hinted with the best-so-far at horizon best+1) on 2026-07-10: under the
pinned-cancel model the deep improvements arrive as large in-solve LNS
jumps, and every rung rebuild burned ~25-35 s of re-presolve and reset
exactly that LNS state — measured pinned-descent depths [38,39,37,38,33]
median 38 vs pinned single-solve median 37 over 15 draws on the d=8192
fixture (umbrella ``cpsat_pinned_cancel_plan.md``, steps 2-3).

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
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear


def _width_graph():
    torch.manual_seed(1)
    x = create_input("x", 4)
    mids = []
    for i in range(6):
        li = Linear(x, torch.randn(4, 12), torch.zeros(12), name=f"L{i}")
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Ma{i}"))
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Mb{i}"))
    return Linear(Concatenate(mids), torch.randn(24, 4), torch.zeros(4), name="out")


def _ffn_chain_graph():
    """x -> FFN -> L_mid -> FFN -> L_out: nonlinearities keep the
    intermediates alive through lowering, so the heuristic warm start has
    real nodes to free — non-empty cancel + cancel-mechanism hints."""
    torch.manual_seed(0)
    x = create_input("x", 8)
    block_a = linear_relu_linear(
        x, torch.randn(16, 8), torch.zeros(16),
        torch.randn(16, 12), torch.zeros(12), name="a",
    )
    L_mid = Linear(block_a, torch.randn(12, 16), torch.zeros(16), name="L_mid")
    block_c = linear_relu_linear(
        L_mid, torch.randn(16, 16), torch.zeros(16),
        torch.randn(16, 8), torch.zeros(8), name="c",
    )
    return Linear(block_c, torch.randn(8, 4), torch.zeros(4), name="L_out")


def test_optimize3_compiles_replays_and_is_no_worse_than_optimize1():
    """A real optimize=3 compile produces a valid, replayable schedule no
    deeper than optimize=1 (the bigger budget never regresses)."""
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


def test_solve_budget_override_is_accepted():
    """The measurement-only ``_solve_budget_s`` overrides the solve budget
    (production default 600s stays put) and still yields a valid solve."""
    graph = _width_graph()
    net = forward_compile(
        d=80, d_head=80, output_node=graph, device="cpu", verbose=False,
        optimize=3, _solve_only=True, _force_resolve=True, _solve_budget_s=5.0,
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


def test_optimize3_is_one_solve_with_the_full_budget_and_the_mech_hint(
    monkeypatch,
):
    """optimize=3 performs exactly ONE ``solve_schedule`` call at the full
    600 s budget — no rung loop, no re-solve — warm-started with all four
    hint families from the heuristic.  The cancel-MECHANISM hint reaching the
    solver is load-bearing under the pinned-cancel default: without those
    bits the pinned model completed the hint into ZERO incumbents in
    5x600 s on the d=8192 production fixture (cpsat_pinned_cancel_plan.md
    step 2, batch 1) — the solver, not the compile, is where the
    cancel-LAYER values get dropped.  (``_solve_only`` returns before the
    replay, so the fake schedule is never executed.)"""
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
        return asg, _fake_stats(6, False)

    monkeypatch.setattr(cmod, "solve_schedule", fake_solve)

    net = forward_compile(
        d=64, d_head=8, output_node=_ffn_chain_graph(), device="cpu",
        verbose=False, optimize=3, _solve_only=True, _force_resolve=True,
    )

    # One continuous solve; a non-optimal FEASIBLE result is accepted as-is
    # (no rung loop re-solving behind it).
    assert len(calls) == 1
    assert net.cpsat_assignment.n_layers == 6

    kw = calls[0]
    # The full production budget goes to the single solve.
    assert kw["time_budget_s"] == 600.0
    # All four hint families are passed from the heuristic warm start; the
    # mechanism bits are the load-bearing ones under the pin.
    assert kw["hint_layers"], "no layer hint reached the solver"
    assert kw["hint_routing"], "no routing hint reached the solver"
    assert kw["hint_cancel"], "no cancel hint reached the solver"
    assert kw["hint_cancel_mech"], "no cancel-mechanism hint reached the solver"
    # Horizon is the heuristic depth + 1 slack layer (a soft ceiling).
    assert kw["max_layers"] == max(kw["hint_layers"].values()) + 2
