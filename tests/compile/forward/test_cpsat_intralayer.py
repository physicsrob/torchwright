"""CP-SAT intra-layer reuse — attention mechanism.

Covers the routing-aware cancel bound + parked escape in the model, and the
directed replay realizing same-layer attention handoffs as ONE atomic batch
transition (docs/cpsat_atomic_attention_replay_plan.md): capture every batch
source, release every assigned cancel, then place every output against the
aggregate-freed pool.

Two kinds of replay coverage, deliberately separate:

- **Deterministic hard-fixed fixtures** (the collective-readers, dual-release,
  and entry-dead fixtures) pin the batch semantics and the model/replay
  contract assertions against exact assignments, independent of which
  equal-depth optimum a parallel CP-SAT run happens to return.
- **The width-starved smoke test** drives a real optimize=1 solve end to end
  and asserts only properties of the compile it actually ran (success, depth
  equality with its own solve, numerical parity) — a second independent
  ``solve_schedule`` is NOT guaranteed to reproduce the compile's assignment.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    ATTN,
    DiagnosticHint,
    MLP,
    ScheduleAssignment,
    SolveStats,
    build_cpsat_model,
    critical_path_layers,
    solve_schedule,
)
from torchwright.graph import Add, Concatenate, Linear
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
    """Broad solver/replay smoke test on the width-starved graph: an
    optimize=1 compile succeeds, emits exactly the depth its own solve
    claimed, and matches the exact-math reference.  Asserts only properties
    of the compile it actually ran — parallel CP-SAT does not guarantee that
    a second independent ``solve_schedule`` reproduces the compile's
    assignment, so nothing here re-solves (which equal-depth optimum a
    16-worker solve returns is a race).  Gap-zero and collective same-layer
    handoff semantics are pinned deterministically by the hard-fixed
    fixtures below (atomic-attention-replay plan §§5.1–5.2)."""
    x, out = _width_starved_graph()
    net = forward_compile(
        d=48, d_head=8, output_node=out, device="cpu", verbose=False, optimize=1
    )
    stats = net.cpsat_solve_stats
    assert stats is not None
    # Replay-depth equality against the compile's OWN recorded solve (the
    # in-compile tripwire enforces this on every directed compile; asserting
    # it here keeps the smoke test honest if the tripwire ever regresses).
    assert len(net.layers) == stats.objective_value // stats.objective_scale

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

    hint = DiagnosticHint(
        layers=hint_layers,
        routing=hint_routing,
        cancel=hint_cancel,
        cancel_mech=hint_mech,
    )
    built = build_cpsat_model(
        out,
        d=d,
        d_head=d_head,
        d_hidden=d_hidden,
        max_layers=max_layers,
        diagnostic_hint=hint,
    )
    violations = _validate_hint(
        built,
        hint,
        max_layers=max_layers,
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
        _diagnostic_hint=hint,
        strict_hint=True,
    )
    assert asg is not None, f"solver found no schedule ({stats.status_name})"
    assert asg.n_layers <= hint_n, (
        f"solver returned {asg.n_layers} layers, worse than the accepted "
        f"hint's {hint_n} — the incumbent was silently dropped"
    )


def test_validate_hint_flags_mlp_mechanism_on_input():
    """Inputs have no MLP cancel mechanism in the model (no cancel_in_mlp var
    for freeable inputs), so an MLP-mech hint on an input is a hint the model
    cannot represent and must be flagged — the former leniency read the MLP
    mechanism as gap-0-for-everything and stayed silent."""
    from torchwright.compiler.forward.cpsat_scheduler import _validate_hint

    torch.manual_seed(0)
    x = create_input("x", 4)
    y = Linear(x, torch.randn(4, 4), torch.zeros(4), name="y")
    out = Linear(y, torch.randn(4, 4), torch.zeros(4), name="out")
    built = build_cpsat_model(out, d=16, d_head=4, d_hidden=16, max_layers=6)
    hint_layers = {y.node_id: 0, out.node_id: 1}

    violations = _validate_hint(
        built,
        DiagnosticHint(
            layers=hint_layers,
            cancel={x.node_id: 1},
            cancel_mech={x.node_id: MLP},
        ),
        max_layers=6,
    )
    assert any("MLP cancel mechanism" in v for v in violations), violations

    clean = _validate_hint(
        built,
        DiagnosticHint(
            layers=hint_layers,
            cancel={x.node_id: 1},
            cancel_mech={x.node_id: ATTN},
        ),
        max_layers=6,
    )
    assert clean == [], clean


def test_validate_hint_checks_add_consumer_free_gap():
    """The former deliberate blind spot: an Add consumer's cancel bound is
    `layer + is_free` regardless of mechanism, and is_free is a pure function
    of the layer assignment — so the validator derives it from the layer
    hints.  Here w's only consumer is the Add (dead addend -> free add), so
    u's cancel needs gap 1; the gap-0 hint used to pass silently."""
    from torchwright.compiler.forward.cpsat_scheduler import _validate_hint

    torch.manual_seed(0)
    x = create_input("x", 4)
    u = Linear(x, torch.randn(4, 4), torch.zeros(4), name="u")
    w = Linear(x, torch.randn(4, 4), torch.zeros(4), name="w")
    add = Add(w, u)
    out = Linear(add, torch.randn(4, 4), torch.zeros(4), name="out")
    built = build_cpsat_model(out, d=16, d_head=4, d_hidden=16, max_layers=6)
    hint_layers = {u.node_id: 0, w.node_id: 0, add.node_id: 1, out.node_id: 2}

    violations = _validate_hint(
        built,
        DiagnosticHint(layers=hint_layers, cancel={u.node_id: 1}),
        max_layers=6,
    )
    assert any("consumer's layer+1" in v for v in violations), violations

    clean = _validate_hint(
        built,
        DiagnosticHint(layers=hint_layers, cancel={u.node_id: 2}),
        max_layers=6,
    )
    assert clean == [], clean


def test_validate_hint_held_target_add_permits_gap_zero():
    """The held target is a forced fresh compute (the model pins its is_free
    to 0), so a gap-0 cancel of an addend at the target's own layer is legal
    — the derivation must special-case it instead of reading the sole-consumer
    addend as `free` and demanding gap 1."""
    from torchwright.compiler.forward.cpsat_scheduler import _validate_hint

    torch.manual_seed(0)
    src = create_input("src", 4)
    left = Linear(src, torch.eye(4), torch.zeros(4), name="left")
    out = Add(left, src)
    built = build_cpsat_model(
        out,
        d=16,
        d_head=4,
        d_hidden=16,
        max_layers=6,
        held_source_id=src.node_id,
        held_target_id=out.node_id,
    )
    hint_layers = {left.node_id: 0, out.node_id: 1}
    clean = _validate_hint(
        built,
        DiagnosticHint(
            layers=hint_layers,
            cancel={src.node_id: 1},
            cancel_mech={src.node_id: ATTN},
        ),
        max_layers=6,
    )
    assert clean == [], clean


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
        b,
        d=64,
        d_head=8,
        d_hidden=64,
        max_layers=max_layers,
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


# ---------------------------------------------------------------------------
# Collective same-layer handoff — atomic attention replay
# (docs/cpsat_atomic_attention_replay_plan.md §§5.1–5.2)
# ---------------------------------------------------------------------------


def _collective_readers_graph():
    """One source, two simultaneous same-layer attention readers (plan §5.1).

    ``S`` has two attention-routed consumers ``A`` and ``B`` assigned to the
    same layer, where ``S`` is also assigned its attention-mechanism cancel.
    On entry to that layer exactly one residual column is free, so the layer
    is feasible only as one aggregate attention transition (release S's eight
    columns, allocate A and B, then the MLP ``out``) — never as a per-output
    ordering, because neither A nor B is individually S's last uncomputed
    consumer while the other is pending.

    Two lowering gates keep this fixture intact; a change to either would
    silently collapse it:

    - ``fuse_consecutive_linears``' fold-through-Concatenate case applies
      structurally (A and B are sole-consumer Linear leaves of a Concatenate
      whose sole consumer is a Linear) and is declined only by its
      never-grow-parameter-count gate: folding A[8->2] into out's 2x4 block
      would cost 8*4 = 32 parameters against the current 8*2 + 2*4 = 24.
    - the width-four ``x`` keeps the graph out of the univariate collapse
      pass (which rebuilds whole single-input subgraphs).
    """
    torch.manual_seed(0)
    x = create_input("x", 4)
    blocker = create_input("blocker", 10)
    s = Linear(x, torch.randn(4, 8), torch.zeros(8), name="S")
    a = Linear(s, torch.randn(8, 2), torch.zeros(2), name="A")
    b = Linear(s, torch.randn(8, 2), torch.zeros(2), name="B")
    out = Linear(
        Concatenate([a, x, b, blocker]),
        torch.randn(18, 4),
        torch.zeros(4),
        name="out",
    )
    return x, blocker, s, a, b, out


def test_collective_readers_assignment_is_cpsat_feasible():
    """The plan §5.1 hard-fixed assignment is model-feasible at depth two:
    S at layer 0 (attention); A, B at layer 1 (attention); out at layer 1
    (MLP); S cancelled at layer 1 via the attention mechanism.  Layer 1 uses
    exactly six of six attention heads (A reads S[8]: 2; B reads S[8]: 2;
    coalesced cancel of S[8]: 2) and exactly fits the residual stream (one
    free column on entry, S's eight released, A+B's four plus the MLP out's
    four allocated).  Paired with ``test_collective_readers_replay_atomically``
    this proves any replay failure is a model/replay mismatch, not an
    infeasible assignment."""
    _, _, s, a, b, out = _collective_readers_graph()
    built = build_cpsat_model(out, d=24, d_head=4, d_hidden=24, max_layers=8)
    built.model.Add(built.n_layers_var == 2)
    status = _hard_fix_and_solve(
        built,
        {s.node_id: 0, a.node_id: 1, b.node_id: 1, out.node_id: 1},
        {s.node_id: ATTN, a.node_id: ATTN, b.node_id: ATTN, out.node_id: MLP},
        {s.node_id: 1},
        {s.node_id: ATTN},
    )
    assert status in ("OPTIMAL", "FEASIBLE"), (
        f"the collective-handoff assignment must be CP-SAT feasible at depth "
        f"two, got {status}"
    )


def test_collective_readers_replay_atomically(monkeypatch):
    """Directed replay of the exact plan §5.1 assignment: the compile must
    emit exactly the assigned two layers, use exactly six of six attention
    heads at the handoff layer, and match the graph reference — without
    modifying or re-solving the assignment.  Before the atomic-batch fix this
    raised the one-free-column ``No progress`` deadlock: replay demanded a
    per-output bootstrap order (each output allocating after releasing at
    most one input) that the aggregate-feasible layer does not admit."""
    from torchwright.compiler.forward import compile as compile_mod
    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer

    x, blocker, s, a, b, out = _collective_readers_graph()

    solve_calls = []

    def fixed_solve(lowered_out, *args, **kwargs):
        g = GraphAnalyzer(lowered_out)
        by_name = {
            n.name: n.node_id for n in g.get_all_nodes() if getattr(n, "name", "")
        }
        asg = ScheduleAssignment(
            node_to_layer={
                by_name["S"]: 0,
                by_name["A"]: 1,
                by_name["B"]: 1,
                by_name["out"]: 1,
            },
            node_to_cancel_layer={
                by_name["S"]: 1,
                # Everything else lives past the two-layer horizon.
                by_name["A"]: 2,
                by_name["B"]: 2,
                by_name["out"]: 2,
                by_name["x"]: 2,
                by_name["blocker"]: 2,
            },
            node_to_routing={
                by_name["S"]: ATTN,
                by_name["A"]: ATTN,
                by_name["B"]: ATTN,
                by_name["out"]: MLP,
            },
            n_layers=2,
            node_to_cancel_mech={by_name["S"]: ATTN},
        )
        stats = SolveStats(
            status_name="OPTIMAL",
            objective_value=2,
            best_objective_bound=2.0,
            wall_time_s=0.0,
            solver_log="",
            total_attn_heads=-1,
            total_mlp_bypass_slots=-1,
            is_optimal=True,
        )
        solve_calls.append(
            (
                asg,
                dict(asg.node_to_layer),
                dict(asg.node_to_cancel_layer),
                dict(asg.node_to_routing),
                dict(asg.node_to_cancel_mech),
            )
        )
        return asg, stats

    monkeypatch.setattr(compile_mod, "solve_schedule", fixed_solve)

    net = forward_compile(
        d=24,
        d_head=4,
        d_hidden=24,
        output_node=out,
        device="cpu",
        verbose=False,
        optimize=1,
        require_solver=True,
    )

    # The fixed assignment was used once, unmodified — never re-solved.
    assert len(solve_calls) == 1, "the compile must not re-solve"
    asg, n2l, n2cl, n2r, n2m = solve_calls[0]
    assert asg.node_to_layer == n2l
    assert asg.node_to_cancel_layer == n2cl
    assert asg.node_to_routing == n2r
    assert asg.node_to_cancel_mech == n2m

    # Exactly the assigned depth — no slack layer.
    assert len(net.layers) == 2

    # The handoff layer uses exactly six of six attention heads, read from
    # the compile's recorded per-layer head counts (not re-derived by hand):
    # A reads S[8] (2 heads), B reads S[8] (2 heads), coalesced cancel of
    # S[8] (2 heads).
    counts = net.per_layer_head_counts[1]
    assert counts.get("compute_linear", 0) == 4
    assert counts.get("cancel", 0) == 2
    assert sum(counts.values()) == 24 // 4

    ix = torch.randn(3, 4)
    ib = torch.randn(3, 10)
    ref = out.compute(3, {"x": ix, "blocker": ib})
    got = net.compute(3, {"x": ix, "blocker": ib})[out].cpu()
    assert torch.allclose(got, ref, atol=1e-3), (
        f"collective-handoff replay diverges from reference: "
        f"max err {(got - ref).abs().max().item():.2e}"
    )


def _directed_scheduler(out, asg, inputs, *, d=24, d_head=4, d_hidden=24):
    """A ``DirectedLayerScheduler`` plus residual map set up the way
    ``forward_compile`` would: a width-1 stand-in for the reserved const-1
    column, then every input, allocated in order — so the scheduler-level
    column arithmetic matches the production allocation exactly."""
    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
    from torchwright.compiler.forward.residual_map import ResidualStreamMap
    from torchwright.compiler.forward.scheduler import DirectedLayerScheduler

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(d)
    const = create_input("const_stand_in", 1)
    for node in (const, *inputs):
        rmap.allocate(node)
    computed = set(inputs)
    sched = DirectedLayerScheduler(
        graph, d, d_head, None, assignment=asg, d_hidden=d_hidden
    )
    return sched, rmap, computed


def _dual_release_graph():
    """One reader, multiple dying inputs (plan §5.2) — the dual fixture.

    ``C`` is the single last consumer of both ``Sx`` and ``Sy``.  On entry to
    C's layer one column is free; releasing only one four-column source
    leaves five, still short of C[6] — only the combined release of both
    admits the layer.  This pins the other incompleteness of a unary
    (one-dying-input-per-placement) handoff API.

    The general linear lowering intentionally collapses this algebraically
    linear graph to a single Linear, so the replay half is tested directly
    through ``DirectedLayerScheduler`` on the source graph (no lowering
    escape hatch); the CP feasibility half runs on the source graph, which
    is what ``build_cpsat_model`` receives here.
    """
    torch.manual_seed(0)
    x = create_input("x", 4)
    y = create_input("y", 4)
    blocker = create_input("blocker", 6)
    sx = Linear(x, torch.randn(4, 4), torch.zeros(4), name="Sx")
    sy = Linear(y, torch.randn(4, 4), torch.zeros(4), name="Sy")
    c = Linear(Concatenate([sx, sy]), torch.randn(8, 6), torch.zeros(6), name="C")
    out = Linear(
        Concatenate([c, x, blocker, y]),
        torch.randn(20, 2),
        torch.zeros(2),
        name="out",
    )
    return x, y, blocker, sx, sy, c, out


def _dual_release_assignment(x, y, blocker, sx, sy, c, out):
    """The plan §5.2 hard-fixed assignment: Sx, Sy at layer 0; C (attention)
    and out (MLP) at layer 1; both source cancels at layer 1 via the
    attention mechanism; depth two."""
    return ScheduleAssignment(
        node_to_layer={sx.node_id: 0, sy.node_id: 0, c.node_id: 1, out.node_id: 1},
        node_to_cancel_layer={
            sx.node_id: 1,
            sy.node_id: 1,
            c.node_id: 2,
            out.node_id: 2,
            x.node_id: 2,
            y.node_id: 2,
            blocker.node_id: 2,
        },
        node_to_routing={
            sx.node_id: ATTN,
            sy.node_id: ATTN,
            c.node_id: ATTN,
            out.node_id: MLP,
        },
        n_layers=2,
        node_to_cancel_mech={sx.node_id: ATTN, sy.node_id: ATTN},
    )


def test_dual_release_assignment_is_cpsat_feasible():
    """The plan §5.2 hard-fixed assignment is model-feasible at depth two on
    the source graph.  Compute (C reads Sx+Sy[8]: 2 heads) plus the coalesced
    eight-column cancel (2 heads) uses four of six attention heads; entry has
    one free column and the combined release of both sources fits C and the
    MLP out exactly."""
    x, y, blocker, sx, sy, c, out = _dual_release_graph()
    built = build_cpsat_model(out, d=24, d_head=4, d_hidden=24, max_layers=8)
    built.model.Add(built.n_layers_var == 2)
    status = _hard_fix_and_solve(
        built,
        {sx.node_id: 0, sy.node_id: 0, c.node_id: 1, out.node_id: 1},
        {sx.node_id: ATTN, sy.node_id: ATTN, c.node_id: ATTN, out.node_id: MLP},
        {sx.node_id: 1, sy.node_id: 1},
        {sx.node_id: ATTN, sy.node_id: ATTN},
    )
    assert status in ("OPTIMAL", "FEASIBLE"), (
        f"the dual-release assignment must be CP-SAT feasible at depth two, "
        f"got {status}"
    )


def test_dual_release_replays_atomically():
    """Directed replay of the plan §5.2 assignment, driven straight through
    ``DirectedLayerScheduler`` on the source graph: both dying sources are
    released in one coalesced cancel, C's source columns are the pre-release
    capture, and C plus the layer's MLP output place without an extra layer.
    Before the atomic-batch fix, the unary reuse released one source, failed
    C's six-column allocation against five free, and deferred the layer."""
    x, y, blocker, sx, sy, c, out = _dual_release_graph()
    asg = _dual_release_assignment(x, y, blocker, sx, sy, c, out)
    sched, rmap, computed = _directed_scheduler(out, asg, (x, y, blocker))

    sched.set_current_layer(0)
    sched.schedule_layer(rmap, computed)
    assert sx in computed and sy in computed

    # Layer-1 entry: const(1) + x(4) + y(4) + blocker(6) + Sx(4) + Sy(4)
    # occupy 23 of 24 columns.
    assert rmap.get_free_count() == 1

    sx_cols = list(rmap.get_indices(sx))
    sy_cols = list(rmap.get_indices(sy))

    sched.set_current_layer(1)
    attn_ops, mlp_ops, _ = sched.schedule_layer(rmap, computed)

    # C placed via attention with the captured pre-release concatenation.
    c_ops = [op for op in attn_ops if op.op_type == "compute_linear" and op.node is c]
    assert len(c_ops) == 1, f"C was not placed at its assigned layer: {attn_ops}"
    assert c_ops[0].source_cols == sx_cols + sy_cols

    # Both dying sources in the ONE coalesced cancel batch.
    cancels = [op for op in attn_ops if op.op_type == "cancel"]
    assert len(cancels) == 1
    assert set(cancels[0].target_cols) == set(sx_cols + sy_cols)

    # The MLP output places in the same layer — depth two, no extra layer.
    assert any(
        op.op_type == "compute_linear_bypass" and op.node is out for op in mlp_ops
    )
    assert out in computed


# ---------------------------------------------------------------------------
# Contract-assertion negative tests (plan §5.6)
#
# Each fail-loud check the atomic batch added — A2 whole-batch last-reader
# validation, the A3 head preflight, the A4 width preflight, the overdue-
# cancel scan, and the post-preflight must-allocate assertion — gets one
# test feeding an assignment corrupted in exactly one way and pinning the
# error shape.  These are directed-replay model/replay contract assertions,
# NOT extensions of the canonical compiler invariants I1–I4 (no CLAUDE.md
# entry, no test_compiler_assertions.py test); the assertion-plus-negative-
# test pairing discipline is borrowed because it is what keeps an assertion
# honest across refactors.
# ---------------------------------------------------------------------------


def test_batch_contract_a2_consumer_outside_batch():
    """A2: a value cancelled at a layer where one uncomputed consumer is not
    in the attention batch (here: its only reader rerouted to the MLP) fails
    before any mutation."""
    import dataclasses

    import pytest

    x, y, blocker, sx, sy, c, out = _dual_release_graph()
    asg = _dual_release_assignment(x, y, blocker, sx, sy, c, out)
    # One-field corruption: C runs in the MLP sublayer, so Sx/Sy's assigned
    # attention cancels at C's layer have an uncomputed consumer outside the
    # attention batch (the model's routing-aware bound would push both
    # cancels a layer later; this assignment is model-infeasible).
    asg = dataclasses.replace(
        asg, node_to_routing={**asg.node_to_routing, c.node_id: MLP}
    )
    sched, rmap, computed = _directed_scheduler(out, asg, (x, y, blocker))

    sched.set_current_layer(0)
    sched.schedule_layer(rmap, computed)

    sched.set_current_layer(1)
    with pytest.raises(AssertionError, match=r"\(A2\)"):
        sched.schedule_layer(rmap, computed)


def test_batch_contract_a3_head_overcharge():
    """A3: one extra attention-routed node moved into the handoff layer makes
    the batch's exact head charge exceed n_heads and fails before mutation.
    The moved node reads only the input, so it is genuinely ready during the
    handoff layer's attention pass (a same-layer reader of A or B would be
    filtered by readiness, not by the head preflight)."""
    import pytest

    torch.manual_seed(0)
    x = create_input("x", 4)
    blocker = create_input("blocker", 8)
    s = Linear(x, torch.randn(4, 8), torch.zeros(8), name="S")
    a = Linear(s, torch.randn(8, 2), torch.zeros(2), name="A")
    b = Linear(s, torch.randn(8, 2), torch.zeros(2), name="B")
    t = Linear(x, torch.randn(4, 2), torch.zeros(2), name="T")
    out = Linear(
        Concatenate([a, x, b, t, blocker]),
        torch.randn(18, 4),
        torch.zeros(4),
        name="out",
    )
    # The sound assignment places T at layer 0 (the handoff layer then uses
    # exactly six of six heads).  One-field corruption: T moves into the
    # handoff layer, so the batch charge becomes A(2) + B(2) + T(1) +
    # coalesced cancel of S[8](2) = 7 > 6.
    asg = ScheduleAssignment(
        node_to_layer={
            s.node_id: 0,
            t.node_id: 1,
            a.node_id: 1,
            b.node_id: 1,
            out.node_id: 1,
        },
        node_to_cancel_layer={
            s.node_id: 1,
            t.node_id: 2,
            a.node_id: 2,
            b.node_id: 2,
            out.node_id: 2,
            x.node_id: 2,
            blocker.node_id: 2,
        },
        node_to_routing={
            s.node_id: ATTN,
            t.node_id: ATTN,
            a.node_id: ATTN,
            b.node_id: ATTN,
            out.node_id: MLP,
        },
        n_layers=2,
        node_to_cancel_mech={s.node_id: ATTN},
    )
    sched, rmap, computed = _directed_scheduler(out, asg, (x, blocker))

    sched.set_current_layer(0)
    sched.schedule_layer(rmap, computed)

    sched.set_current_layer(1)
    with pytest.raises(AssertionError, match=r"\(A3\)"):
        sched.schedule_layer(rmap, computed)


def test_batch_contract_a4_width_underrelease():
    """A4: removing one release from an exactly-fitting assignment makes the
    ordinary width preflight fail before any mutation."""
    import dataclasses

    import pytest

    x, y, blocker, sx, sy, c, out = _dual_release_graph()
    asg = _dual_release_assignment(x, y, blocker, sx, sy, c, out)
    # One-field corruption: Sy's cancel moves one layer later, so the batch
    # releases only Sx (4 cols) against C's 6-column demand with 1 free.
    asg = dataclasses.replace(
        asg, node_to_cancel_layer={**asg.node_to_cancel_layer, sy.node_id: 2}
    )
    sched, rmap, computed = _directed_scheduler(out, asg, (x, y, blocker))

    sched.set_current_layer(0)
    sched.schedule_layer(rmap, computed)

    sched.set_current_layer(1)
    with pytest.raises(AssertionError, match=r"\(A4\)"):
        sched.schedule_layer(rmap, computed)


def test_batch_contract_overdue_cancel():
    """Overdue: a cancel assigned one layer earlier than the value goes dead
    is still allocated when its assigned layer has passed — an assertion,
    not a silent reschedule (an assignment that fit only because the cancel
    occurred on time is not soundly replayed by delaying it)."""
    import dataclasses

    import pytest

    x, y, blocker, sx, sy, c, out = _dual_release_graph()
    asg = _dual_release_assignment(x, y, blocker, sx, sy, c, out)
    # One-field corruption: Sx's cancel at layer 0 — its own birth layer,
    # one layer before its reader C makes it dead.  Layer 0's hook runs
    # before Sx is allocated (it is a placement candidate), so the missed
    # cancel surfaces as overdue on entry to layer 1.
    asg = dataclasses.replace(
        asg, node_to_cancel_layer={**asg.node_to_cancel_layer, sx.node_id: 0}
    )
    sched, rmap, computed = _directed_scheduler(out, asg, (x, y, blocker))

    sched.set_current_layer(0)
    sched.schedule_layer(rmap, computed)

    sched.set_current_layer(1)
    with pytest.raises(AssertionError, match=r"\(overdue cancel\)"):
        sched.schedule_layer(rmap, computed)


def test_batch_contract_must_allocate(monkeypatch):
    """Must-allocate: reachable only through a preflight bug, so it is pinned
    with a direct unit seam — force ``_try_allocate`` to return ``None`` past
    a passing preflight — rather than by weakening one of the four honest
    corruptions above."""
    import pytest

    x, y, blocker, sx, sy, c, out = _dual_release_graph()
    asg = _dual_release_assignment(x, y, blocker, sx, sy, c, out)
    sched, rmap, computed = _directed_scheduler(out, asg, (x, y, blocker))

    sched.set_current_layer(0)
    sched.schedule_layer(rmap, computed)

    monkeypatch.setattr(sched, "_try_allocate", lambda node, residual_map: None)
    sched.set_current_layer(1)
    with pytest.raises(AssertionError, match=r"\(post-preflight allocation\)"):
        sched.schedule_layer(rmap, computed)


# ---------------------------------------------------------------------------
# Entry-dead and mid-batch releases share one batch (plan §5.7)
# ---------------------------------------------------------------------------


def _entry_dead_graph():
    """Both release classes in one layer (plan §5.7): ``W`` is already dead
    on entry to layer 1 (its only consumer ``P`` ran in layer 0's MLP, so its
    attention-mechanism cancel pins to layer 1), while ``S`` dies mid-batch
    under its two layer-1 readers.  Before the atomic batch, entry-dead
    values flowed through the promotion/leftover paths that the batch now
    bypasses — a botched bypass (entry-dead values dropped instead of
    batched) passes the §§5.1–5.2 fixtures and is caught only here."""
    torch.manual_seed(0)
    x = create_input("x", 4)
    blocker = create_input("blocker", 10)
    w = Linear(x, torch.randn(4, 2), torch.zeros(2), name="W")
    p = Linear(w, torch.randn(2, 2), torch.zeros(2), name="P")
    s = Linear(x, torch.randn(4, 4), torch.zeros(4), name="S")
    a = Linear(s, torch.randn(4, 2), torch.zeros(2), name="A")
    b = Linear(s, torch.randn(4, 2), torch.zeros(2), name="B")
    out = Linear(
        Concatenate([a, b, p, x, blocker]),
        torch.randn(20, 2),
        torch.zeros(2),
        name="out",
    )
    return x, blocker, w, p, s, a, b, out


def _entry_dead_assignment(x, blocker, w, p, s, a, b, out):
    return ScheduleAssignment(
        node_to_layer={
            w.node_id: 0,
            p.node_id: 0,
            s.node_id: 0,
            a.node_id: 1,
            b.node_id: 1,
            out.node_id: 1,
        },
        node_to_cancel_layer={
            w.node_id: 1,
            s.node_id: 1,
            p.node_id: 2,
            a.node_id: 2,
            b.node_id: 2,
            out.node_id: 2,
            x.node_id: 2,
            blocker.node_id: 2,
        },
        node_to_routing={
            w.node_id: ATTN,
            p.node_id: MLP,
            s.node_id: ATTN,
            a.node_id: ATTN,
            b.node_id: ATTN,
            out.node_id: MLP,
        },
        n_layers=2,
        node_to_cancel_mech={w.node_id: ATTN, s.node_id: ATTN},
    )


def test_entry_dead_assignment_is_cpsat_feasible():
    """The §5.7 assignment is model-feasible at depth two (W's cancel is
    legal at layer 1 because its MLP consumer P ran at layer 0; S's is legal
    gap-0 under its two attention readers)."""
    x, blocker, w, p, s, a, b, out = _entry_dead_graph()
    built = build_cpsat_model(out, d=24, d_head=4, d_hidden=24, max_layers=8)
    built.model.Add(built.n_layers_var == 2)
    status = _hard_fix_and_solve(
        built,
        {
            w.node_id: 0,
            p.node_id: 0,
            s.node_id: 0,
            a.node_id: 1,
            b.node_id: 1,
            out.node_id: 1,
        },
        {
            w.node_id: ATTN,
            p.node_id: MLP,
            s.node_id: ATTN,
            a.node_id: ATTN,
            b.node_id: ATTN,
            out.node_id: MLP,
        },
        {w.node_id: 1, s.node_id: 1},
        {w.node_id: ATTN, s.node_id: ATTN},
    )
    assert status in ("OPTIMAL", "FEASIBLE"), (
        f"the entry-dead assignment must be CP-SAT feasible at depth two, "
        f"got {status}"
    )


def test_entry_dead_and_mid_batch_releases_share_one_batch():
    """Bypass regression pin (green before and after the fix — the old
    promotion path also handled W): the entry-dead ``W`` and the mid-batch
    dying ``S`` are released by the same coalesced cancel, the overdue scan
    stays silent, and the layer places everything at depth two."""
    x, blocker, w, p, s, a, b, out = _entry_dead_graph()
    asg = _entry_dead_assignment(x, blocker, w, p, s, a, b, out)
    sched, rmap, computed = _directed_scheduler(out, asg, (x, blocker))

    sched.set_current_layer(0)
    sched.schedule_layer(rmap, computed)
    assert w in computed and p in computed and s in computed

    # Layer-1 entry: const(1) + x(4) + blocker(10) + W(2) + P(2) + S(4)
    # occupy 23 of 24 columns.
    assert rmap.get_free_count() == 1

    w_cols = list(rmap.get_indices(w))
    s_cols = list(rmap.get_indices(s))

    sched.set_current_layer(1)
    attn_ops, mlp_ops, _ = sched.schedule_layer(rmap, computed)

    # Entry-dead W and mid-batch S share the single coalesced cancel op.
    cancels = [op for op in attn_ops if op.op_type == "cancel"]
    assert len(cancels) == 1
    assert set(cancels[0].target_cols) == set(w_cols + s_cols)

    # A, B place via attention; out places via the MLP — depth two.
    placed = {op.node for op in attn_ops if op.op_type == "compute_linear"}
    assert placed == {a, b}
    assert any(
        op.op_type == "compute_linear_bypass" and op.node is out for op in mlp_ops
    )
    assert out in computed
