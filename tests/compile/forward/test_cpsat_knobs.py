"""Tests for the CP-SAT solver knobs added for the scheduling campaign
(2026-06): domain tightening, solver parameter overrides, incumbent trace
capture, and lexicographic secondary objectives.

All tests run on a small two-FFN repro graph — a handful of nodes, so
each solve is sub-second.
"""

import inspect

import pytest
import torch

from torchwright.compiler.forward.cpsat_scheduler import (
    Costs,
    DiagnosticHint,
    _compute_layer_bounds,
    build_cpsat_model,
    build_graph_model,
    solve_schedule,
)
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.graph import Linear
from torchwright.graph.optimize import fuse_consecutive_linears
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear


def _repro_graph():
    """x -> FFN -> L_mid -> FFN -> L_out (a small serial FFN graph,
    lightly fused)."""
    torch.manual_seed(0)
    x = create_input("x", 8)
    block_a = linear_relu_linear(
        x,
        torch.randn(16, 8),
        torch.zeros(16),
        torch.randn(16, 12),
        torch.zeros(12),
        name="a",
    )
    L_mid = Linear(block_a, torch.randn(12, 16), torch.zeros(16), name="L_mid")
    block_c = linear_relu_linear(
        L_mid,
        torch.randn(16, 16),
        torch.zeros(16),
        torch.randn(16, 8),
        torch.zeros(8),
        name="c",
    )
    L_out = Linear(block_c, torch.randn(8, 4), torch.zeros(4), name="L_out")
    fuse_consecutive_linears({L_out})
    return L_out


def test_parallel_hint_api_is_isolated_behind_one_diagnostic_value():
    solve_params = inspect.signature(solve_schedule).parameters
    build_params = inspect.signature(build_cpsat_model).parameters
    legacy = {"hint_layers", "hint_routing", "hint_cancel", "hint_cancel_mech"}

    assert legacy.isdisjoint(solve_params)
    assert legacy.isdisjoint(build_params)
    assert "_diagnostic_hint" in solve_params
    assert "diagnostic_hint" in build_params


def test_diagnostic_hint_defensively_freezes_parallel_mappings():
    layers = {2: 1}
    hint = DiagnosticHint(layers=layers, routing={2: "attn"})
    layers[2] = 9

    assert hint.layers == {2: 1}
    with pytest.raises(TypeError):
        hint.layers[2] = 3


_SOLVE_KW = dict(d=64, d_head=8, d_hidden=128, time_budget_s=10.0, max_layers=20)


def test_tighten_domains_preserves_optimum():
    out = _repro_graph()
    plain, _ = solve_schedule(out, **_SOLVE_KW)
    tight, tight_stats = solve_schedule(out, tighten_domains=True, **_SOLVE_KW)
    assert plain is not None and tight is not None
    assert tight_stats.status_name == "OPTIMAL"
    assert tight.n_layers == plain.n_layers


def test_layer_bounds_sound_against_solved_schedule():
    """Every layer in an actual feasible schedule satisfies [es, ls]."""
    out = _repro_graph()
    assignment, _ = solve_schedule(out, **_SOLVE_KW)
    assert assignment is not None
    gm = build_graph_model(out)
    es, ls = _compute_layer_bounds(
        gm, SchedulingPolicy(), True, _SOLVE_KW["max_layers"]
    )
    for nid, layer in assignment.node_to_layer.items():
        assert (
            es[nid] <= layer <= ls[nid]
        ), f"node {nid} scheduled at {layer} outside [{es[nid]}, {ls[nid]}]"


def test_solver_params_applied_and_solve_unchanged():
    out = _repro_graph()
    assignment, stats = solve_schedule(
        out,
        solver_params={
            "random_seed": 7,
            "shared_tree_num_workers": 0,
            "ignore_subsolvers": ["objective_lb_search"],
        },
        **_SOLVE_KW,
    )
    assert assignment is not None
    assert stats.status_name == "OPTIMAL"


def test_drop_decision_strategy_solves_same_depth():
    """Dropping the hand-rolled ``AddDecisionStrategy`` (C1 arm
    ``no_decision_strategy``) is search-only: on a graph the solver proves
    optimal, the achieved depth is identical — only the search path changes."""
    out = _repro_graph()
    with_strategy, s0 = solve_schedule(out, **_SOLVE_KW)
    without, s1 = solve_schedule(out, drop_decision_strategy=True, **_SOLVE_KW)
    assert with_strategy is not None and without is not None
    assert s0.status_name == "OPTIMAL" and s1.status_name == "OPTIMAL"
    assert without.n_layers == with_strategy.n_layers


def test_solution_trace_captures_incumbents():
    out = _repro_graph()
    trace: list = []
    assignment, _ = solve_schedule(out, solution_trace=trace, **_SOLVE_KW)
    assert assignment is not None
    assert len(trace) >= 1
    last = trace[-1]
    assert last["n_layers"] == assignment.n_layers
    # Snapshots carry the full assignment.
    assert set(last["layers"]) == set(assignment.node_to_layer)
    assert set(last["cancels"]) == set(assignment.node_to_cancel_layer) - set(
        last["input_cancels"]
    )


@pytest.mark.parametrize(
    "costs",
    [
        Costs(alpha=1, earliness=1),
        Costs(alpha=1, waste=1),
        Costs(alpha=1, earliness=1, waste=1),
    ],
    ids=["earliness", "waste", "both"],
)
def test_secondary_objectives_are_lexicographic(costs):
    """Secondaries must never trade a layer: primary optimum is preserved
    and ``objective_value // objective_scale`` recovers it exactly."""
    out = _repro_graph()
    plain, _ = solve_schedule(out, **_SOLVE_KW)
    sec, stats = solve_schedule(out, costs=costs, **_SOLVE_KW)
    assert plain is not None and sec is not None
    assert sec.n_layers == plain.n_layers
    assert stats.objective_scale > 1
    assert stats.objective_value // stats.objective_scale == plain.n_layers


def test_plain_costs_keep_scale_one():
    out = _repro_graph()
    _, stats = solve_schedule(out, **_SOLVE_KW)
    assert stats.objective_scale == 1


# ---------------------------------------------------------------------------
# Single-phase warm-start descent (optimize >= 2 in forward_compile)
#
# The floor-probe phase was removed 2026-07-08; optimize>=2 now runs a single
# warm-started descent whose time budget absorbs the retired probe's budget
# (cpsat_time_budget_s + min(150, cpsat_time_budget_s)).  These tests pin the
# two regimes the probe used to straddle: with width slack the descent reaches
# the dependency floor with a real solve; when width binds it still produces a
# valid, deeper compile.
# ---------------------------------------------------------------------------


def test_descent_reaches_floor_at_slack_width():
    """With width slack, optimize=2's single warm-start descent lands at (or
    below) critical_path+1 with a real solve (no floor probe)."""
    from torchwright.compiler.forward.compile import forward_compile
    from torchwright.compiler.forward.cpsat_scheduler import (
        critical_path_layers,
    )

    out = _repro_graph()
    cp = critical_path_layers(out)
    net = forward_compile(
        d=64,
        d_head=8,
        output_node=out,
        device="cpu",
        verbose=False,
        optimize=2,
    )
    assert net.cpsat_solve_stats is not None
    assert net.cpsat_solve_stats.status_name in ("OPTIMAL", "FEASIBLE")
    assert len(net.layers) <= cp + 1


def test_schedule_cache_round_trip(tmp_path, monkeypatch):
    """Second compile of the same topology+geometry replays the cached
    schedule (status CACHED), skips the solver, and computes identically."""
    from torchwright.compiler.forward.compile import forward_compile

    monkeypatch.setenv("TW_SCHEDULE_CACHE_DIR", str(tmp_path))
    kw = dict(
        d=64,
        d_head=8,
        device="cpu",
        verbose=False,
        optimize=2,
    )
    out1 = _repro_graph()
    net1 = forward_compile(output_node=out1, **kw)
    assert net1.cpsat_solve_stats.status_name in ("OPTIMAL", "FEASIBLE")
    assert len(list(tmp_path.glob("*.json"))) == 1

    # Rebuild the graph from scratch: fresh node objects with SHIFTED raw
    # node ids (the global counter keeps counting), exactly like a warm
    # Modal container compiling twice in one process.  The canonical-id
    # fingerprint must still hit.
    out2 = _repro_graph()
    net2 = forward_compile(output_node=out2, **kw)
    assert net2.cpsat_solve_stats.status_name == "CACHED"
    assert len(net2.layers) == len(net1.layers)
    fresh_provenance = net1.schedule_result.provenance
    cached_provenance = net2.schedule_result.provenance
    assert fresh_provenance.delivery == "fresh"
    assert cached_provenance.delivery == "cache"
    assert cached_provenance.origin == fresh_provenance.origin
    assert cached_provenance.selected_objective == fresh_provenance.selected_objective
    assert (
        cached_provenance.selected_objective_blocks
        == fresh_provenance.selected_objective_blocks
    )
    assert cached_provenance.selected_is_optimal == fresh_provenance.selected_is_optimal
    assert cached_provenance.solver_attempt is not None
    assert fresh_provenance.solver_attempt is not None
    assert (
        cached_provenance.solver_attempt.status_name
        == fresh_provenance.solver_attempt.status_name
    )

    inputs = {"x": torch.randn(3, 8)}
    torch.testing.assert_close(
        net1.compute(3, inputs)[out1], net2.compute(3, inputs)[out2]
    )


def test_schedule_cache_disabled_without_env(monkeypatch):
    """Without TW_SCHEDULE_CACHE_DIR nothing is read or written."""
    from torchwright.compiler.forward.cpsat_scheduler import (
        ScheduleAssignment,
    )
    from torchwright.compiler.forward.schedule_cache import (
        cache_dir,
        load_assignment,
        store_assignment,
    )

    monkeypatch.delenv("TW_SCHEDULE_CACHE_DIR", raising=False)
    dummy = create_input("dummy", 1)
    assert cache_dir() is None
    assert load_assignment("deadbeef", dummy) is None
    assert not store_assignment(
        "deadbeef", ScheduleAssignment({}, {}, {}, 1), {}, dummy
    )


def test_schedule_cache_keys_on_compiler_code(monkeypatch):
    """Any torchwright source change must invalidate cached schedules: the
    fingerprint includes a content hash of the package sources, so an edit
    the topology hash cannot see (warm-start heuristic, solver model) still
    misses instead of replaying a stale schedule."""
    from torchwright.compiler import graph_identity

    out = _repro_graph()
    kw = dict(
        d=64,
        d_head=8,
        d_hidden=64,
        flex_routing=True,
        cancel_slack=2,
        policy=None,
    )
    fp_before = graph_identity.graph_fingerprint(out, **kw)
    assert fp_before == graph_identity.graph_fingerprint(out, **kw)
    monkeypatch.setattr(
        graph_identity, "compiler_code_fingerprint", lambda: "edited-sources"
    )
    assert graph_identity.graph_fingerprint(out, **kw) != fp_before


def test_schedule_cache_min_optimize_gate(tmp_path, monkeypatch):
    """An entry solved at a lower optimize level misses a higher-level
    request; a failed higher-level improvement upgrades the entry's level
    in place so the re-solve happens once, not per compile."""
    from torchwright.compiler.forward.cpsat_scheduler import (
        ScheduleAssignment,
    )
    from torchwright.compiler.forward.schedule_cache import (
        load_assignment,
        store_assignment,
    )

    monkeypatch.setenv("TW_SCHEDULE_CACHE_DIR", str(tmp_path))
    dummy = create_input("dummy", 1)
    fp = "cafe" * 16

    assert store_assignment(
        fp, ScheduleAssignment({}, {}, {}, 3), {"optimize": 1}, dummy
    )
    assert load_assignment(fp, dummy, min_optimize=1) is not None
    assert load_assignment(fp, dummy, min_optimize=2) is None

    # A WORSE schedule from an optimize=3 solve: the schedule ratchet
    # refuses it, but the fact that level 3 couldn't beat the entry
    # certifies the entry at level 3.
    assert not store_assignment(
        fp, ScheduleAssignment({}, {}, {}, 5), {"optimize": 3}, dummy
    )
    hit = load_assignment(fp, dummy, min_optimize=3)
    assert hit is not None
    assignment, meta = hit
    assert assignment.n_layers == 3  # the better schedule survived
    assert meta["optimize"] == 3

    # Lower-level requests replay a higher-level entry (safe direction).
    assert load_assignment(fp, dummy, min_optimize=0) is not None


def test_schedule_cache_optimize_gate_end_to_end(tmp_path, monkeypatch):
    """Raising ``optimize`` in a compile actually re-solves instead of
    replaying the lower level's cached draw; the level then sticks and the
    next same-level compile is CACHED."""
    from torchwright.compiler.forward.compile import forward_compile

    monkeypatch.setenv("TW_SCHEDULE_CACHE_DIR", str(tmp_path))
    kw = dict(d=64, d_head=8, device="cpu", verbose=False)

    net1 = forward_compile(output_node=_repro_graph(), optimize=1, **kw)
    assert net1.cpsat_solve_stats.status_name in ("OPTIMAL", "FEASIBLE")

    net2 = forward_compile(output_node=_repro_graph(), optimize=2, **kw)
    assert net2.cpsat_solve_stats.status_name != "CACHED"

    net3 = forward_compile(output_node=_repro_graph(), optimize=2, **kw)
    assert net3.cpsat_solve_stats.status_name == "CACHED"
    assert len(net3.layers) == len(net2.layers)


# ---------------------------------------------------------------------------
# Hint-aware cancel-window widening (the silent optimize=2 fallback fix).
# The window/widening machinery belongs to the LEGACY model — the pinned
# production default (``_pin_cancels``) neither builds windows nor keeps
# cancel-layer hints — so every build/solve in this section passes
# ``_pin_cancels=False``.
# ---------------------------------------------------------------------------


def _deferred_cancel_hint(max_layers, cancel_slack=2):
    """Solve the repro graph, then craft a warm-start hint whose cancel for
    one node is pushed to ``last_consumer + 1 + K + 1`` — the shape the
    heuristic produces when a layer's attention heads are full and it defers
    the free to the next layer.  Returns ``(out, hints, target_id)``."""
    from torchwright.compiler.forward.cpsat_scheduler import Concatenate

    out = _repro_graph()
    assignment, stats = solve_schedule(out, _pin_cancels=False, **_SOLVE_KW)
    assert assignment is not None and stats.status_name == "OPTIMAL"

    gm = build_graph_model(out)
    pinned_ids = {n.node_id for n in gm.pinned_nodes}
    target_id = last_cons_layer = None
    for n in gm.schedulable:
        if n.node_id in pinned_ids:
            continue
        consumers = gm.consumers_eff.get(n, set())
        if any(isinstance(c, Concatenate) for c in consumers):
            continue
        cons_layers = [
            assignment.node_to_layer[c.node_id]
            for c in consumers
            if c.node_id in assignment.node_to_layer
        ]
        if not cons_layers:
            continue
        if max(cons_layers) + 1 + cancel_slack + 1 <= max_layers:
            target_id = n.node_id
            last_cons_layer = max(cons_layers)
            break
    assert target_id is not None, "repro graph has no widenable node"

    hint_layers = dict(assignment.node_to_layer)
    hint_routing = dict(assignment.node_to_routing)
    hint_cancel = dict(assignment.node_to_cancel_layer)
    hint_cancel[target_id] = last_cons_layer + 1 + cancel_slack + 1
    return out, (hint_layers, hint_routing, hint_cancel), target_id


def _hard_fix_and_solve(built, hint_layers, hint_routing, hint_cancel):
    """Add an equality per hinted variable and solve — is the hint a model
    point?  Mirrors phase 2 of ``torchwright_doom/scripts/cpsat_hint_audit``."""
    from ortools.sat.python import cp_model

    from torchwright.compiler.forward.cpsat_scheduler import ATTN

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


def test_deferred_cancel_hint_rejected_without_widening():
    """Pins the bug: a heuristic-shaped hint (one cancel deferred one layer
    past the uniform window) is INFEASIBLE under the hint-blind model — the
    root cause of the silent optimize=2 fallback — and becomes a model point
    once ``build_cpsat_model`` sees the hint and widens that node's window."""
    max_layers = _SOLVE_KW["max_layers"]
    out, (hint_layers, hint_routing, hint_cancel), target_id = _deferred_cancel_hint(
        max_layers
    )
    build_kw = dict(
        d=_SOLVE_KW["d"],
        d_head=_SOLVE_KW["d_head"],
        d_hidden=_SOLVE_KW["d_hidden"],
        max_layers=max_layers,
    )

    blind = build_cpsat_model(out, _pin_cancels=False, **build_kw)
    assert blind.cancel_window_delta is None
    status = _hard_fix_and_solve(blind, hint_layers, hint_routing, hint_cancel)
    assert status == "INFEASIBLE", (
        f"hint-blind model accepted the deferred-cancel hint ({status}) — "
        f"the reproducer no longer reproduces"
    )

    aware = build_cpsat_model(
        out,
        diagnostic_hint=DiagnosticHint(layers=hint_layers, cancel=hint_cancel),
        _pin_cancels=False,
        **build_kw,
    )
    assert aware.cancel_window_delta == {target_id: 1}
    status = _hard_fix_and_solve(aware, hint_layers, hint_routing, hint_cancel)
    assert status in (
        "OPTIMAL",
        "FEASIBLE",
    ), f"widened model rejected the deferred-cancel hint ({status})"


def test_deferred_cancel_hint_accepted_by_solve_schedule():
    """End-to-end through ``solve_schedule``: the deferral-shaped hint passes
    strict validation (the widened window admits it) and solves."""
    out, (hint_layers, hint_routing, hint_cancel), _ = _deferred_cancel_hint(
        _SOLVE_KW["max_layers"]
    )
    assignment, stats = solve_schedule(
        out,
        _diagnostic_hint=DiagnosticHint(
            layers=hint_layers,
            routing=hint_routing,
            cancel=hint_cancel,
        ),
        strict_hint=True,
        _pin_cancels=False,
        **_SOLVE_KW,
    )
    assert assignment is not None
    assert stats.status_name in ("OPTIMAL", "FEASIBLE")


def test_strict_hint_validation_raises_on_invalid_hint():
    """A genuinely invalid hint (cancel before birth+1) raises under
    ``strict_hint=True`` and warns under the default.  Legacy model only:
    the pinned default drops cancel-layer hints before validation (their
    values are forced by the pin), so this contract lives behind
    ``_pin_cancels=False``."""
    out = _repro_graph()
    assignment, _ = solve_schedule(out, _pin_cancels=False, **_SOLVE_KW)
    assert assignment is not None
    hint_layers = dict(assignment.node_to_layer)
    # Pick any non-keep-forever node and hint its cancel AT its birth layer
    # (the model requires cancel >= birth + 1).
    gm = build_graph_model(out)
    pinned_ids = {n.node_id for n in gm.pinned_nodes}
    target = next(
        nid
        for nid, cl in assignment.node_to_cancel_layer.items()
        if nid in hint_layers and nid not in pinned_ids and cl < _SOLVE_KW["max_layers"]
    )
    bad_cancel = {target: hint_layers[target]}

    with pytest.raises(ValueError, match="before birth"):
        solve_schedule(
            out,
            _diagnostic_hint=DiagnosticHint(
                layers=hint_layers,
                cancel=bad_cancel,
            ),
            strict_hint=True,
            _pin_cancels=False,
            **_SOLVE_KW,
        )

    with pytest.warns(RuntimeWarning, match="hint validation"):
        assignment2, _ = solve_schedule(
            out,
            _diagnostic_hint=DiagnosticHint(
                layers=hint_layers,
                cancel=bad_cancel,
            ),
            _pin_cancels=False,
            **_SOLVE_KW,
        )
    # Default mode keeps the fall-back-don't-fail contract: still solves.
    assert assignment2 is not None


def test_strict_hint_validation_raises_on_keep_forever_cancel():
    """A cancel hint below max_layers for a keep-forever node (pinned or
    Concatenate-consumed) is a hint the model pins to max_layers — strict
    mode names it.  Legacy model only (the pinned default drops cancel-layer
    hints before validation)."""
    out = _repro_graph()
    assignment, _ = solve_schedule(out, _pin_cancels=False, **_SOLVE_KW)
    assert assignment is not None
    gm = build_graph_model(out)
    # The output node is the schedulable pinned node (inputs are modeled as
    # freeable, so they are not keep-forever despite being pinned).
    keep = gm.output_node
    with pytest.raises(ValueError, match="keep-forever"):
        solve_schedule(
            out,
            _diagnostic_hint=DiagnosticHint(
                layers=assignment.node_to_layer,
                cancel={keep.node_id: 1},
            ),
            strict_hint=True,
            _pin_cancels=False,
            **_SOLVE_KW,
        )


def test_descent_valid_compile_on_width_starved_graph():
    """Width-starved graph: the dependency floor cannot fit (parallel wide
    chains must serialize), so optimize=2's warm-start descent produces a
    valid compile necessarily deeper than critical_path+1."""
    from torchwright.compiler.forward.compile import forward_compile
    from torchwright.compiler.forward.cpsat_scheduler import (
        critical_path_layers,
    )
    from torchwright.graph import Concatenate

    torch.manual_seed(0)
    x = create_input("x", 4)
    # 8 independent chains x -> Li(12 cols) -> two Mi(2 cols) each.  The
    # critical path is short, but at d=48 the Li's cannot coexist, so any
    # schedule is far deeper than critical_path + 1 — the probe horizon
    # is infeasible by construction.  Each Li feeds TWO Ms so the
    # lowering boundary's linear fusion can't absorb it (multi-consumer),
    # and the Mi->out concat fold is declined by the parameter guard
    # (12->2 bottleneck against a width-4 output).  The concat groups all
    # Ma's before all Mb's so no two adjacent leaves share an Li — adjacent
    # same-input leaves are collapsed by the sibling fold, which would leave
    # each Li single-consumer and unravel the starvation entirely.
    mas, mbs = [], []
    for i in range(8):
        li = Linear(x, torch.randn(4, 12), torch.zeros(12), name=f"L{i}")
        mas.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Ma{i}"))
        mbs.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Mb{i}"))
    out = Linear(Concatenate(mas + mbs), torch.randn(32, 4), torch.zeros(4), name="out")
    cp = critical_path_layers(out)
    net = forward_compile(
        d=48,
        d_head=8,
        output_node=out,
        device="cpu",
        verbose=False,
        optimize=2,
    )
    assert net.cpsat_solve_stats is not None
    # The compile is valid and necessarily deeper than the probe horizon.
    assert len(net.layers) > cp + 1
