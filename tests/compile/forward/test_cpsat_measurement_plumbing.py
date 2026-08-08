"""Measurement-only plumbing for CP-SAT gap attribution (Unit 0, plan §0).

``forward_compile`` threads three measurement knobs down to
``solve_schedule`` so the gap-attribution sweep can run *through the real
compile path* (plan decision #2) rather than a standalone replica:

* ``_disabled_families`` — a solve-only diagnostic: solve with the named
  constraint families relaxed, return the assignment/stats before the
  replay, never compile or cache the (unsound) relaxed schedule;
* ``_solver_seed`` — seed the solve lottery reproducibly;
* ``_force_resolve`` — skip the schedule cache so a measured solve
  re-solves instead of replaying a cached draw;
* ``_solver_params`` — general ``CpSolver.parameters`` overrides (C1 sweep),
  merged with the seed and applied to the solve;
* ``_drop_decision_strategy`` — skip the hand-rolled ``AddDecisionStrategy``
  (C1 arm ``no_decision_strategy``).
"""

import torch

import torchwright.compiler.forward.compile as compile_mod
from torchwright.compiler.forward.compile import forward_compile
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


def _compile(out, **kw):
    return forward_compile(
        d=80,
        d_head=80,
        output_node=out,
        device="cpu",
        verbose=False,
        optimize=1,
        **kw,
    )


def test_disabled_families_is_solve_only():
    """Return the assignment + stats without replaying on a relaxed solve.

    Zero compiled layers, and the relaxed depth is a lower bound on the
    sound one (``dependency`` relaxed lands strictly below, the §0 sanity
    family).
    """
    sound = _compile(_width_graph())
    assert len(sound.layers) > 0  # sound path compiles + replays
    sound_n = sound.cpsat_solve_stats.objective_value

    relaxed = _compile(_width_graph(), _disabled_families=frozenset({"dependency"}))
    assert len(relaxed.layers) == 0  # never replayed
    assert relaxed.cpsat_assignment is not None
    assert relaxed.cpsat_assignment.n_layers <= sound_n
    # best_objective_bound is a certified lower bound on the sound optimum.
    assert relaxed.cpsat_solve_stats.best_objective_bound <= sound_n


def test_add_live_addend_gap_family_is_accepted():
    """Resolve gap #2's diagnostic family name and keep the relaxed solve solve-only.

    The family name resolves (unknown names raise), the relaxed solve stays
    solve-only, and its depth is a valid lower bound on the sound optimum.
    The graph carries a free Add so the relaxed `layer[A]` form of the
    Add-consumer cancel term is actually constructed.
    """
    torch.manual_seed(0)
    x = create_input("x", 4)
    u = Linear(x, torch.randn(4, 4), torch.zeros(4), name="u")
    w = Linear(x, torch.randn(4, 4), torch.zeros(4), name="w")
    from torchwright.graph import Add

    out = Linear(Add(w, u), torch.randn(4, 4), torch.zeros(4), name="out")

    sound = _compile(out)
    assert len(sound.layers) > 0
    sound_n = sound.cpsat_solve_stats.objective_value

    relaxed = _compile(out, _disabled_families=frozenset({"add_live_addend_gap"}))
    assert len(relaxed.layers) == 0  # never replayed
    assert relaxed.cpsat_assignment is not None
    assert relaxed.cpsat_assignment.n_layers <= sound_n


def test_relaxed_solve_never_touches_the_cache(tmp_path, monkeypatch):
    """Never let a relaxed solve read or write the fingerprint-keyed cache.

    Neither read a cached sound schedule nor write its own (unsound)
    schedule into the cache the production compile replays from.
    """
    monkeypatch.setenv("TW_SCHEDULE_CACHE_DIR", str(tmp_path))

    # A sound compile populates the cache.
    _compile(_width_graph())
    entries = list(tmp_path.glob("*.json"))
    assert len(entries) == 1, "sound compile should cache its schedule"

    # A relaxed solve writes nothing new (and does not overwrite the sound one).
    before = entries[0].read_text()
    _compile(_width_graph(), _disabled_families=frozenset({"mlp_cancel_occupancy"}))
    assert list(tmp_path.glob("*.json")) == entries
    assert entries[0].read_text() == before


def test_force_resolve_bypasses_a_cached_schedule(tmp_path, monkeypatch):
    """Re-solve via ``_force_resolve`` even when a cache entry exists.

    A cached hit would otherwise silently stand in for the measurement.
    """
    monkeypatch.setenv("TW_SCHEDULE_CACHE_DIR", str(tmp_path))

    cached = _compile(_width_graph())
    assert cached.cpsat_solve_stats.status_name in ("OPTIMAL", "FEASIBLE")

    hit = _compile(_width_graph())
    assert hit.cpsat_solve_stats.status_name == "CACHED"

    fresh = _compile(_width_graph(), _force_resolve=True)
    assert fresh.cpsat_solve_stats.status_name != "CACHED"
    assert len(fresh.layers) == len(cached.layers)


def test_solve_only_returns_sound_depth_without_replaying():
    """Return the sound solve's assignment/stats before the replay via ``_solve_only``.

    The G0 attribution reads the sound n_layers through the same real path
    as the relaxed cells, and it matches the full compile's depth.
    """
    full = _compile(_width_graph())
    solve_only = _compile(_width_graph(), _solve_only=True, _force_resolve=True)
    assert len(solve_only.layers) == 0  # replay skipped
    assert solve_only.cpsat_assignment is not None
    assert solve_only.cpsat_assignment_payload is not None
    assert (
        solve_only.cpsat_assignment_payload["n_layers"]
        == solve_only.cpsat_assignment.n_layers
    )
    assert solve_only.schedule_fingerprint is not None
    assert len(solve_only.schedule_fingerprint) == 64
    assert solve_only.cpsat_assignment.n_layers == len(full.layers)


def test_solver_seed_is_accepted_and_replays():
    """Produce a sound, replayable schedule from a seeded solve.

    The seed only perturbs the search, never soundness.
    """
    graph = _width_graph()
    net = _compile(graph, _solver_seed=12345)
    assert len(net.layers) > 0
    inp = torch.randn(3, 4)
    ref = graph.compute(3, {"x": inp})
    got = net.compute(3, {"x": inp})[graph]
    assert torch.allclose(got.cpu(), ref, atol=1e-3)


def test_solver_params_merge_reaches_the_solve(monkeypatch):
    """Merge ``_solver_params`` with the seed and forward ``_drop_decision_strategy``.

    Captured at the solve boundary so the plumbing is proven independent of
    solver behavior.
    """
    captured = {}
    real = compile_mod.solve_schedule

    def _spy(*args, **kw):
        captured["solver_params"] = kw.get("solver_params")
        captured["drop_decision_strategy"] = kw.get("drop_decision_strategy")
        return real(*args, **kw)

    monkeypatch.setattr(compile_mod, "solve_schedule", _spy)
    _compile(
        _width_graph(),
        _solver_seed=9,
        _solver_params={"linearization_level": 2},
        _drop_decision_strategy=True,
    )
    assert captured["solver_params"] == {"random_seed": 9, "linearization_level": 2}
    assert captured["drop_decision_strategy"] is True


def test_solver_params_none_when_unset(monkeypatch):
    """Keep ``solver_params`` at None and the decision strategy on with no overrides.

    Default production behavior is byte-unchanged.
    """
    captured = {}
    real = compile_mod.solve_schedule

    def _spy(*args, **kw):
        captured["solver_params"] = kw.get("solver_params")
        captured["drop_decision_strategy"] = kw.get("drop_decision_strategy")
        return real(*args, **kw)

    monkeypatch.setattr(compile_mod, "solve_schedule", _spy)
    _compile(_width_graph())
    assert captured["solver_params"] is None
    assert captured["drop_decision_strategy"] is False


def test_solver_params_applied_and_schedule_replays():
    """Yield a sound, replayable schedule under a general parameter override.

    A valid ``CpSolver`` field can only change search, never feasibility.
    """
    graph = _width_graph()
    net = _compile(graph, _solver_params={"linearization_level": 2})
    assert len(net.layers) > 0
    inp = torch.randn(3, 4)
    ref = graph.compute(3, {"x": inp})
    got = net.compute(3, {"x": inp})[graph]
    assert torch.allclose(got.cpu(), ref, atol=1e-3)


def test_drop_decision_strategy_replays():
    """Keep the compile sound and replayable when dropping the decision strategy.

    Search-only: the compile still produces a sound, replayable schedule.
    """
    graph = _width_graph()
    net = _compile(graph, _drop_decision_strategy=True)
    assert len(net.layers) > 0
    inp = torch.randn(3, 4)
    ref = graph.compute(3, {"x": inp})
    got = net.compute(3, {"x": inp})[graph]
    assert torch.allclose(got.cpu(), ref, atol=1e-3)
