"""Tests for the CP-SAT solver knobs added for the scheduling campaign
(2026-06): domain tightening, solver parameter overrides, incumbent trace
capture, and lexicographic secondary objectives.

All tests run on a small two-Block repro graph — a handful of nodes, so
each solve is sub-second.
"""

import pytest
import torch

from torchwright.compiler.forward.cpsat_scheduler import (
    Costs,
    _compute_layer_bounds,
    build_graph_model,
    solve_schedule,
)
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.graph import Linear
from torchwright.graph.optimize import fuse_consecutive_linears
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear_relu_linear import linear_relu_linear


def _repro_graph():
    """x -> Block -> L_mid -> Block -> L_out (a small serial block graph,
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
# Floor-probe ladder (optimize >= 2 in forward_compile)
# ---------------------------------------------------------------------------


def test_floor_probe_succeeds_at_slack_width():
    """With width slack, optimize=2 cold-probes at critical_path+1 and the
    compile lands at (or below) the heuristic's depth with a real solve."""
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


def test_floor_probe_infeasible_falls_back_to_descent():
    """Width-starved graph: the floor horizon cannot fit (parallel wide
    chains must serialize), so optimize=2 must fall through the probe and
    still produce a valid compile via the descent/heuristic path."""
    from torchwright.compiler.forward.compile import forward_compile
    from torchwright.compiler.forward.cpsat_scheduler import (
        critical_path_layers,
    )
    from torchwright.graph import Concatenate

    torch.manual_seed(0)
    x = create_input("x", 4)
    # 8 independent chains x -> Li(12 cols) -> Mi(2 cols).  The critical
    # path is short, but at d=48 the Li's cannot coexist, so any schedule
    # is far deeper than critical_path + 1 — the probe horizon is
    # infeasible by construction.
    mids = []
    for i in range(8):
        li = Linear(x, torch.randn(4, 12), torch.zeros(12), name=f"L{i}")
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"M{i}"))
    out = Linear(Concatenate(mids), torch.randn(16, 4), torch.zeros(4), name="out")
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
