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

    The concat groups all Ma's before all Mb's so that no two *adjacent* leaves
    share an input.  Adjacent same-input Linear leaves are merged by
    ``fuse_consecutive_linears``' sibling fold; merging Ma_i with Mb_i would
    leave Li single-consumer, unravel the chain into ``x -> M_i``, and compile
    the whole graph in one layer — the starvation this fixture exists to create.
    """
    torch.manual_seed(0)
    x = create_input("x", 4)
    mas, mbs = [], []
    for i in range(8):
        li = Linear(x, torch.randn(4, 12), torch.zeros(12), name=f"L{i}")
        mas.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Ma{i}"))
        mbs.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Mb{i}"))
    out = Linear(Concatenate(mas + mbs), torch.randn(32, 4), torch.zeros(4), name="out")
    return x, out


def _hard_fix_and_solve(
    built, hint_layers, hint_routing, hint_cancel, hint_cancel_mech=None
):
    """Add an equality per hinted variable and solve: is the point feasible?

    ``hint_cancel_mech`` (node_id -> "attn"/"mlp") pins the cancel mechanism
    bools; without it the solver is free to flip a node from attention-cancel
    to MLP-cancel (which relaxes its cancel bound to the uniform gap-0 form),
    so a feasibility assertion about the attention-cancel bound would test
    nothing.
    """
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
    for nid, mech in (hint_cancel_mech or {}).items():
        if nid in built.cancel_in_mlp:
            model.Add(built.cancel_in_mlp[nid] == (1 if mech == MLP else 0))
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
    """Routing-aware cancel bound under an ATTENTION-mech cancel: A cancelled at
    B's layer is feasible when B is attention-routed (gap 0) and infeasible when
    B is MLP-routed (gap 1).  A's mechanism is pinned to attention — otherwise
    the solver could flip A to MLP-cancel and satisfy the gap-0 bound even with
    an MLP-routed B (that case is the next test)."""
    x, a, b = _chain_ab()
    hint_layers = {a.node_id: 0, b.node_id: 1}
    hint_cancel = {a.node_id: 1}  # A cancelled at B's own layer
    attn_mech = {a.node_id: ATTN}

    # B in attention: A's bound is cancel >= layer[B] + 1 - is_attn[B] = 1.
    built_attn = build_cpsat_model(b, d=64, d_head=8, d_hidden=64, max_layers=20)
    status_attn = _hard_fix_and_solve(
        built_attn, hint_layers, {b.node_id: ATTN}, hint_cancel, attn_mech
    )
    assert status_attn in ("OPTIMAL", "FEASIBLE"), (
        f"same-layer cancel with an attention consumer should be feasible, "
        f"got {status_attn}"
    )

    # B in MLP: bound becomes cancel >= layer[B] + 1 = 2; cancel==1 is rejected.
    built_mlp = build_cpsat_model(b, d=64, d_head=8, d_hidden=64, max_layers=20)
    status_mlp = _hard_fix_and_solve(
        built_mlp, hint_layers, {b.node_id: MLP}, hint_cancel, attn_mech
    )
    assert status_mlp == "INFEASIBLE", (
        f"same-layer cancel with an MLP consumer must be infeasible (gap 1), "
        f"got {status_mlp}"
    )


def test_mlp_cancel_same_layer_feasible_with_mlp_consumer():
    """Mechanism-conditional bound: with an MLP-routed consumer B at layer 1,
    A cancelled at layer 1 (gap 0) is INFEASIBLE under an attention-mech cancel
    (bound cancel >= layer[B]+1) but FEASIBLE under an MLP-mech cancel (uniform
    bound cancel >= layer[B]).  This is the mirror of the routing test above and
    pins that `cancel_in_mlp` actually relaxes the bound."""
    x, a, b = _chain_ab()
    hint_layers = {a.node_id: 0, b.node_id: 1}
    hint_cancel = {a.node_id: 1}
    mlp_route = {b.node_id: MLP}

    built_attn = build_cpsat_model(b, d=64, d_head=8, d_hidden=64, max_layers=20)
    status_attn = _hard_fix_and_solve(
        built_attn, hint_layers, mlp_route, hint_cancel, {a.node_id: ATTN}
    )
    assert status_attn == "INFEASIBLE", (
        f"attention-mech cancel at an MLP consumer's layer must be infeasible, "
        f"got {status_attn}"
    )

    built_mlp = build_cpsat_model(b, d=64, d_head=8, d_hidden=64, max_layers=20)
    status_mlp = _hard_fix_and_solve(
        built_mlp, hint_layers, mlp_route, hint_cancel, {a.node_id: MLP}
    )
    assert status_mlp in ("OPTIMAL", "FEASIBLE"), (
        f"MLP-mech cancel at an MLP consumer's layer (gap 0) should be "
        f"feasible, got {status_mlp}"
    )


def _reuse_pressure_graph():
    """Two independent width-8 chains off a width-2 input plus their narrow
    tails.  A (chain 1) dies at layer 1; C (chain 2) is born at layer 1 and can
    only fit if A's columns are already reclaimed there.  Tuned so d=19 is
    exactly the width where an attention-cancel of A (frees mid-attention,
    interval [0,1)) leaves room for C but an MLP-cancel of A (frees at the end
    of layer 1, interval [0,2)) does not."""
    torch.manual_seed(0)
    x = create_input("x", 2)
    a = Linear(x, torch.randn(2, 8), torch.zeros(8), name="A")
    b = Linear(a, torch.randn(8, 2), torch.zeros(2), name="B")
    c = Linear(x, torch.randn(2, 8), torch.zeros(8), name="C")
    cc = Linear(c, torch.randn(8, 2), torch.zeros(2), name="Cc")
    out = Linear(Concatenate([b, cc]), torch.randn(4, 2), torch.zeros(2), name="out")
    return x, a, b, c, cc, out


def test_mlp_cancel_residual_extends_through_cancel_layer():
    """Soundness of the MLP-cancel residual accounting.  An MLP cancel fires in
    the MLP sublayer, so the dying node's columns stay occupied through the
    WHOLE cancel layer (interval [layer, cancel+1)), not just up to it like an
    attention cancel (interval [layer, cancel)).  At the tuned width, freeing A
    at layer 1 to make room for C is feasible via an attention cancel but NOT
    via an MLP cancel — the columns are still live during layer 1's attention
    sublayer, where the replay would need them for C.  Were the model to give
    MLP-cancel the same [layer, cancel) interval, this would be (wrongly)
    feasible and the directed replay would hit I4."""
    _, a, b, c, cc, out = _reuse_pressure_graph()
    hint_layers = {
        a.node_id: 0,
        b.node_id: 1,
        c.node_id: 1,
        cc.node_id: 2,
        out.node_id: 3,
    }
    hint_routing = {n.node_id: ATTN for n in (a, b, c, cc, out)}
    hint_cancel = {a.node_id: 1, c.node_id: 2}
    # Pin every other node's mechanism to attention so only A's mechanism moves.
    base_mech = {b.node_id: ATTN, c.node_id: ATTN, cc.node_id: ATTN}

    built_attn = build_cpsat_model(out, d=19, d_head=2, d_hidden=38, max_layers=12)
    status_attn = _hard_fix_and_solve(
        built_attn,
        hint_layers,
        hint_routing,
        hint_cancel,
        {**base_mech, a.node_id: ATTN},
    )
    assert status_attn in ("OPTIMAL", "FEASIBLE"), (
        f"attention-cancel of A (frees mid-attention, [0,1)) leaves room for C "
        f"at layer 1 — should be feasible, got {status_attn}"
    )

    built_mlp = build_cpsat_model(out, d=19, d_head=2, d_hidden=38, max_layers=12)
    status_mlp = _hard_fix_and_solve(
        built_mlp, hint_layers, hint_routing, hint_cancel, {**base_mech, a.node_id: MLP}
    )
    assert status_mlp == "INFEASIBLE", (
        f"MLP-cancel of A occupies layer 1 ([0,2)), so C cannot fit — must be "
        f"infeasible; feasible here would mean an unreplayable schedule, got "
        f"{status_mlp}"
    )


def test_solver_mlp_cancel_replays_correctly():
    """A one-head-per-layer geometry (d_head == d) forces the solver to route
    some death cancels to the MLP mechanism (a single attention op saturates the
    layer's one head).  The directed replay realizes those `cancel_bypass` ops
    and the output matches the exact-math reference — the coupled model + replay
    executing MLP-cancel end to end."""
    from torchwright.compiler.forward.cpsat_scheduler import MLP as _MLP

    torch.manual_seed(1)
    x = create_input("x", 4)
    mids = []
    for i in range(6):
        li = Linear(x, torch.randn(4, 12), torch.zeros(12), name=f"L{i}")
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Ma{i}"))
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Mb{i}"))
    out = Linear(Concatenate(mids), torch.randn(24, 4), torch.zeros(4), name="out")

    asg, _ = solve_schedule(out, d=80, d_head=80, d_hidden=80, max_layers=30)
    n_mlp = sum(1 for v in asg.node_to_cancel_mech.values() if v == _MLP)
    assert n_mlp > 0, "expected the solver to use the MLP cancel mechanism"

    net = forward_compile(
        d=80, d_head=80, output_node=out, device="cpu", verbose=False, optimize=1
    )
    inp = torch.randn(3, 4)
    ref = out.compute(3, {"x": inp})
    got = net.compute(3, {"x": inp})[out].cpu()
    assert torch.allclose(got, ref, atol=1e-3), (
        f"MLP-cancel schedule diverges from reference: "
        f"max err {(got - ref).abs().max().item():.2e}"
    )


def _eager_warm_start_hint(out, d, d_head, d_hidden, max_layers):
    """Run the eager heuristic in schedule-only mode (mirroring
    ``_run_heuristic_warm_start``) and return the layer / routing / cancel /
    cancel-mechanism hints plus the hint layer count.  Now that the warm start
    is eager, this is the schedule the production compile hands CP-SAT."""
    import copy

    from torchwright.compiler.forward.compile import _TrackingResidualStreamMap
    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
    from torchwright.compiler.forward.residual_map import ResidualStreamMap
    from torchwright.compiler.forward.scheduler import LayerScheduler
    from torchwright.compiler.residual_assignment import flatten_concat_nodes

    graph = GraphAnalyzer(out)
    rmap = _TrackingResidualStreamMap(copy.deepcopy(ResidualStreamMap(d)))
    inputs = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]
    for n in inputs:
        rmap.allocate(n)
    computed = set(inputs)
    sched = LayerScheduler(graph, d, d_head, None, d_hidden=d_hidden)
    hint_layers, hint_routing = {}, {}
    for hi in range(max_layers):
        if out in computed:
            break
        rmap.current_layer = hi
        prev = set(computed)
        attn_ops, mlp_ops, _ = sched.schedule_layer(rmap, computed)
        for op in attn_ops:
            if op.op_type == "compute_linear" and op.node is not None:
                hint_routing[op.node.node_id] = ATTN
        for op in mlp_ops:
            if op.op_type == "compute_linear_bypass":
                hint_routing[op.node.node_id] = MLP
        for n in graph.get_all_nodes():
            if isinstance(n, Concatenate) and n not in computed:
                if all(leaf in computed for leaf in flatten_concat_nodes([n])):
                    computed.add(n)
        for n in computed - prev:
            hint_layers[n.node_id] = hi
        if not attn_ops and not mlp_ops:
            break
    hint_n = max(hint_layers.values()) + 1 if hint_layers else 0
    return (
        hint_layers,
        hint_routing,
        dict(rmap.cancel_layer),
        dict(rmap.cancel_mech),
        hint_n,
    )


def test_eager_warm_start_hint_is_accepted_not_dropped():
    """Incident-class regression (the June 2026 eager-free and July 2026
    deferred-cancel silently-dropped-hint incidents named in the
    ``_validate_hint`` docstring).  On a width-pressure graph the eager
    warm-start schedule must be a schedule CP-SAT can USE: (a) ``_validate_hint``
    finds zero violations, and (b) the solver's returned ``n_layers`` is no
    worse than the hint's — i.e. the incumbent was accepted, not silently
    dropped into a cold search.  Uses a one-head-per-layer geometry so the
    eager schedule leans on the MLP-cancel mechanism (15 MLP-cancels), the exact
    density that was previously an infeasible hint."""
    from torchwright.compiler.forward.cpsat_scheduler import _validate_hint

    torch.manual_seed(0)
    x = create_input("x", 4)
    mids = []
    for i in range(6):
        li = Linear(x, torch.randn(4, 12), torch.zeros(12), name=f"L{i}")
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Ma{i}"))
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Mb{i}"))
    out = Linear(Concatenate(mids), torch.randn(24, 4), torch.zeros(4), name="out")

    d = d_head = d_hidden = 80
    max_layers = 40
    hint_layers, hint_routing, hint_cancel, hint_mech, hint_n = _eager_warm_start_hint(
        out, d, d_head, d_hidden, max_layers
    )
    assert hint_n > 0, "eager warm start deadlocked"
    assert any(v == MLP for v in hint_mech.values()), "expected MLP-cancels in the hint"

    built = build_cpsat_model(
        out,
        d=d,
        d_head=d_head,
        d_hidden=d_hidden,
        max_layers=max_layers,
        hint_layers=hint_layers,
        hint_cancel=hint_cancel,
    )
    violations = _validate_hint(
        built, hint_layers, hint_routing, hint_cancel, hint_mech, max_layers=max_layers
    )
    assert violations == [], f"eager warm-start hint has violations: {violations}"

    # strict_hint=True raises if the model would drop the hint; the solve must
    # accept the incumbent and return a schedule no deeper than the hint.
    asg, stats = solve_schedule(
        out,
        d=d,
        d_head=d_head,
        d_hidden=d_hidden,
        max_layers=max_layers,
        hint_layers=hint_layers,
        hint_routing=hint_routing,
        hint_cancel=hint_cancel,
        hint_cancel_mech=hint_mech,
        strict_hint=True,
    )
    assert asg is not None, f"solver found no schedule ({stats.status_name})"
    assert asg.n_layers <= hint_n, (
        f"solver returned {asg.n_layers} layers, worse than the accepted "
        f"hint's {hint_n} — the incumbent was silently dropped"
    )


def test_parked_escape_leaves_node_unfreed_and_charges_no_head():
    """A schedule that never frees a dead node (cancel == max_layers) is
    feasible via the parked escape, even though its last consumer ran much
    earlier — the cancel-head interval is gated absent so no head is charged
    in-horizon.  Legacy model only: the pinned production default
    (``_pin_cancels``) removes the parked escape (every cancel is pinned to
    its earliest legal layer), so this pins the ``_pin_cancels=False``
    escape hatch."""
    x, a, b = _chain_ab()
    max_layers = 20
    built = build_cpsat_model(
        b, d=64, d_head=8, d_hidden=64, max_layers=max_layers,
        _pin_cancels=False,
    )
    # A dead after layer 1 (B reads it there) but parked to max_layers.
    hint_layers = {a.node_id: 0, b.node_id: 1}
    hint_cancel = {a.node_id: max_layers}
    status = _hard_fix_and_solve(built, hint_layers, {b.node_id: ATTN}, hint_cancel)
    assert status in (
        "OPTIMAL",
        "FEASIBLE",
    ), f"parked escape (cancel == max_layers) should be feasible, got {status}"
