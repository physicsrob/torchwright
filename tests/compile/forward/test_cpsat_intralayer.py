"""CP-SAT intra-layer reuse — attention mechanism (Unit 1).

Covers the routing-aware cancel bound + parked escape in the model, and the
directed replay realizing a **solver-produced** same-layer handoff (including
self-consumer reuse — a node's last consumer reusing its dying input's own
columns).  The replay test is the load-bearing one: it drives a schedule the
solver produced on its own (not a hint replay) through ``DirectedLayerScheduler``
end to end, which is the exact path that breaks if the model/replay coupling is
mishandled.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    ATTN,
    MLP,
    build_cpsat_model,
    critical_path_layers,
    solve_schedule,
)
from torchwright.graph import Concatenate, Linear
from torchwright.ops.inout_nodes import create_input


def _width_starved_graph():
    """8 independent chains x -> Li(12 cols) -> {Ma_i, Mb_i}(2 cols each).

    At d=48 the Li's cannot coexist, so a shallow schedule requires freeing
    each Li's columns within its consumers' layer — intra-layer reuse, and in
    particular self-consumer reuse (Ma_i is Li's last attention consumer and
    reuses Li's own columns).  The solver finds this on its own.
    """
    torch.manual_seed(0)
    x = create_input("x", 4)
    mids = []
    for i in range(8):
        li = Linear(x, torch.randn(4, 12), torch.zeros(12), name=f"L{i}")
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Ma{i}"))
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Mb{i}"))
    out = Linear(Concatenate(mids), torch.randn(32, 4), torch.zeros(4), name="out")
    return x, out


def _hard_fix_and_solve(built, hint_layers, hint_routing, hint_cancel):
    """Add an equality per hinted variable and solve: is the point feasible?"""
    from ortools.sat.python import cp_model

    model = built.model
    for nid, L in hint_layers.items():
        if nid in built.layer_var:
            model.Add(built.layer_var[nid] == L)
    for nid, route in hint_routing.items():
        if nid in built.is_attn:
            model.Add(built.is_attn[nid] == (1 if route == ATTN else 0))
    for nid, L in hint_cancel.items():
        if nid in built.cancel_layer:
            model.Add(built.cancel_layer[nid] == L)
        elif nid in built.input_cancel_layer:
            model.Add(built.input_cancel_layer[nid] == L)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    return solver.StatusName(solver.Solve(model))


def test_solver_same_layer_handoff_replays_correctly():
    """The solver produces a gap-0 intra-layer schedule on its own, and the
    directed replay realizes it (self-consumer reuse included) with output
    matching the reference.  Before Unit 1 this graph either hit I4 (base
    replay) or dead-locked; the assertion pins that it now compiles correctly
    AND that a real same-layer handoff occurred."""
    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer

    x, out = _width_starved_graph()
    net = forward_compile(
        d=48, d_head=8, output_node=out, device="cpu", verbose=False, optimize=1
    )
    assert net.cpsat_solve_stats is not None

    # solve_schedule reproduces the assignment the compile used (deterministic).
    asg, _ = solve_schedule(out, d=48, d_head=8, d_hidden=48, max_layers=40)
    n2l, n2cl, n2r = asg.node_to_layer, asg.node_to_cancel_layer, asg.node_to_routing

    # A same-layer handoff: some node cancelled at the very layer an
    # attention-routed consumer of it runs (gap 0).  Its presence proves the
    # solver used intra-layer reuse and the replay realized it — else I4 /
    # no-progress would have fired instead of producing this compile.
    g = GraphAnalyzer(out)
    gap0 = any(
        n2r.get(v.node_id) == ATTN and n2cl.get(u.node_id) == n2l.get(v.node_id)
        for u in g.get_all_nodes()
        if u.node_id in n2cl
        for v in g.get_consumers(u)
    )
    assert gap0, "expected at least one solver-produced gap-0 handoff"

    inp = torch.randn(3, 4)
    ref = out.compute(3, {"x": inp})
    got = net.compute(3, {"x": inp})[out].cpu()
    assert torch.allclose(got, ref, atol=1e-3), (
        f"replayed intra-layer schedule diverges from reference: "
        f"max err {(got - ref).abs().max().item():.2e}"
    )


def test_optimize1_compiles_where_eager_heuristic_cannot():
    """The width-starved graph needs self-consumer reuse (all 16 M-nodes live
    until ``out``, so a live L cannot coexist at d=48) — a schedule only the
    directed replay realizes.  optimize=1 compiles it correctly; the eager
    heuristic (optimize=0) legitimately dead-locks, since it never
    self-consumer-reuses (that would change every golden layer count).  The
    optimize=1-vs-optimize=0 depth comparison on graphs both can schedule lives
    in the step-8 example sweep, not here."""
    import pytest

    x, out = _width_starved_graph()
    opt = forward_compile(
        d=48, d_head=8, output_node=out, device="cpu", verbose=False, optimize=1
    )
    inp = torch.randn(2, 4)
    ref = out.compute(2, {"x": inp})
    got = opt.compute(2, {"x": inp})[out].cpu()
    assert torch.allclose(got, ref, atol=1e-3)

    with pytest.raises(RuntimeError, match="No progress"):
        forward_compile(
            d=48, d_head=8, output_node=out, device="cpu", verbose=False, optimize=0
        )


def _chain_ab():
    """x -> A(Linear) -> B(Linear); B is the output.  A's only consumer is B."""
    torch.manual_seed(0)
    x = create_input("x", 8)
    a = Linear(x, torch.randn(8, 8), torch.zeros(8), name="A")
    b = Linear(a, torch.randn(8, 4), torch.zeros(4), name="B")
    return x, a, b


def test_same_layer_attn_handoff_feasible_mlp_infeasible():
    """Routing-aware cancel bound: A cancelled at B's layer is feasible when B
    is attention-routed (gap 0) and infeasible when B is MLP-routed (gap 1)."""
    x, a, b = _chain_ab()
    built = build_cpsat_model(b, d=64, d_head=8, d_hidden=64, max_layers=20)
    hint_layers = {a.node_id: 0, b.node_id: 1}
    hint_cancel = {a.node_id: 1}  # A cancelled at B's own layer

    # B in attention: A's bound is cancel >= layer[B] + 1 - is_attn[B] = 1.
    built_attn = build_cpsat_model(b, d=64, d_head=8, d_hidden=64, max_layers=20)
    status_attn = _hard_fix_and_solve(
        built_attn, hint_layers, {b.node_id: ATTN}, hint_cancel
    )
    assert status_attn in ("OPTIMAL", "FEASIBLE"), (
        f"same-layer cancel with an attention consumer should be feasible, "
        f"got {status_attn}"
    )

    # B in MLP: bound becomes cancel >= layer[B] + 1 = 2; cancel==1 is rejected.
    built_mlp = build_cpsat_model(b, d=64, d_head=8, d_hidden=64, max_layers=20)
    status_mlp = _hard_fix_and_solve(
        built_mlp, hint_layers, {b.node_id: MLP}, hint_cancel
    )
    assert status_mlp == "INFEASIBLE", (
        f"same-layer cancel with an MLP consumer must be infeasible (gap 1), "
        f"got {status_mlp}"
    )


def test_parked_escape_leaves_node_unfreed_and_charges_no_head():
    """A schedule that never frees a dead node (cancel == max_layers) is
    feasible via the parked escape, even though its last consumer ran much
    earlier — the cancel-head interval is gated absent so no head is charged
    in-horizon."""
    x, a, b = _chain_ab()
    max_layers = 20
    built = build_cpsat_model(b, d=64, d_head=8, d_hidden=64, max_layers=max_layers)
    # A dead after layer 1 (B reads it there) but parked to max_layers.
    hint_layers = {a.node_id: 0, b.node_id: 1}
    hint_cancel = {a.node_id: max_layers}
    status = _hard_fix_and_solve(built, hint_layers, {b.node_id: ATTN}, hint_cancel)
    assert status in (
        "OPTIMAL",
        "FEASIBLE",
    ), f"parked escape (cancel == max_layers) should be feasible, got {status}"
