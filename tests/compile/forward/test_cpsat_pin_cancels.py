"""_pin_cancels: earliest-legal cancel pinning (the production default).

Pins the ``_pin_cancels`` contract points (shipped 2026-07-10; evidence in
the umbrella ``cpsat_pinned_cancel_plan.md``):

- **default is pinned** — the default build IS the pinned model (proto equal
  to an explicit ``_pin_cancels=True`` build, no ``parked`` vars).
- **knob-off reproduces the legacy model** — ``_pin_cancels=False`` is the
  escape hatch: its proto differs from the default and rebuilds the
  window/parked/widening machinery (``parked`` vars present).
- **restriction invariant** — a pinned-model solution, hard-fixed into the
  UNPINNED legacy model (layers, routings, cancel mechanisms, cancel layers,
  input cancels), is feasible there: the pin only ADDS constraints, so every
  schedule it emits is a valid legacy-model solution (machine-valid by
  construction).
- **hint contract** — a full four-family hint from a LEGACY-model solve
  passed to a pinned solve with ``strict_hint=True`` does not raise: the
  cancel-LAYER hints are dropped before validation (forced by the pin);
  layers, routing, and cancel-MECHANISM hints are kept.
- **end-to-end replay** — a default (pinned) ``optimize=1`` compile replays
  cleanly through ``DirectedLayerScheduler`` (the always-on replay-depth
  tripwire and I1–I4 stay silent), and so does a knob-off compile.

CP-SAT solves are CPU-only, so this file runs on the plain suite.
"""

import pytest
import torch
from ortools.sat.python import cp_model

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    ATTN,
    DiagnosticHint,
    MLP,
    build_cpsat_model,
    solve_schedule,
)
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.compiler.lower import lower


# ---------------------------------------------------------------------------
# The six example graphs (matches test_cpsat_sym1.py / test_cpsat_snapshot.py).
# ---------------------------------------------------------------------------


def _build_bucketed_argmin():
    from torchwright.graph import InputNode
    from torchwright.ops.attention_ops import attend_argmin_above_in_bucket
    from torchwright.ops.inout_nodes import create_rope_config

    nb, nt, vw, d_head = 3, 5, 4, 32
    score = InputNode("baib_score", 1, value_range=(-100.0, 100.0))
    validity = InputNode("baib_validity", 1, value_range=(-2.0, 2.0))
    kb = InputNode("baib_kb", nb, value_range=(-2.0, 2.0))
    above = InputNode("baib_above", nt, value_range=(-2.0, 2.0))
    qb = InputNode("baib_qb", nb, value_range=(-2.0, 2.0))
    th = InputNode("baib_th", nt, value_range=(-2.0, 2.0))
    value = InputNode("baib_value", vw, value_range=(-100.0, 100.0))
    return attend_argmin_above_in_bucket(
        create_rope_config(d_head=d_head, max_positions=512),
        score, validity, kb, above, qb, th, value,
    )


def _example_specs():
    from examples import (
        binary_increment,
        caesar_cipher,
        calculator_simple,
        fibonacci,
        sort_digits_v1,
    )

    return {
        "calculator": (lambda: calculator_simple.create_network_parts()[0], calculator_simple.D_MODEL, calculator_simple.D_HEAD),
        "caesar": (lambda: caesar_cipher.create_network_parts()[0], 512, 16),
        "sort_digits": (lambda: sort_digits_v1.create_network_parts()[0], sort_digits_v1.D_MODEL, sort_digits_v1.D_HEAD),
        "fibonacci": (lambda: fibonacci.create_network_parts()[0], 512, 16),
        "binary_increment": (
            lambda: binary_increment.create_network_parts()[0], 256, 16
        ),
        "bucketed_argmin": (_build_bucketed_argmin, 512, 32),
    }


def _build(name):
    build, d, d_head = _example_specs()[name]
    torch.manual_seed(0)
    return build(), d, d_head


_NAMES = list(_example_specs().keys())

_MAX_LAYERS = 60


def _proto_text(built):
    return str(built.model.Proto())


def _lower(out, d):
    return lower(
        out,
        verbose=False,
        collapse_univariate=True,
        collapse_pl=True,
        collapse_lane_cap=d // 4,
    ).output_node


# ---------------------------------------------------------------------------
# Default is pinned / knob-off reproduces the legacy model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_default_build_is_pinned(name):
    """The default build IS the pinned model: proto equal to an explicit
    ``_pin_cancels=True`` build, no ``parked`` vars, the pin equalities
    really posted."""
    node, d, d_head = _build(name)
    cfg = dict(d=d, d_head=d_head, d_hidden=d, max_layers=_MAX_LAYERS)
    default = _proto_text(build_cpsat_model(node, **cfg))
    on = _proto_text(build_cpsat_model(node, _pin_cancels=True, **cfg))
    assert default == on, f"{name}: default proto differs from pinned build"
    assert "parked" not in default, f"{name}: default build has parked vars"
    assert "pin_attn" in default or "pin_mlp" in default, (
        f"{name}: default build posts no pin aux vars (pin is dead)"
    )


@pytest.mark.parametrize("name", _NAMES)
def test_knob_off_reproduces_legacy_model(name):
    """``_pin_cancels=False`` is the escape hatch: the proto differs from the
    (pinned) default and rebuilds the window/parked/widening machinery.

    Every example graph has at least one non-keep-forever node, so the legacy
    build gets its ``parked`` var + upper window back and loses the pins."""
    node, d, d_head = _build(name)
    cfg = dict(d=d, d_head=d_head, d_hidden=d, max_layers=_MAX_LAYERS)
    default = _proto_text(build_cpsat_model(node, **cfg))
    off = _proto_text(build_cpsat_model(node, _pin_cancels=False, **cfg))
    assert off != default, f"{name}: knob-off proto identical to default"
    assert "parked" in off, f"{name}: legacy build has no parked vars"
    assert "pin_attn" not in off and "pin_mlp" not in off, (
        f"{name}: legacy build still posts pin aux vars"
    )


# ---------------------------------------------------------------------------
# Restriction invariant: a pinned solution is a valid unpinned-model solution
# ---------------------------------------------------------------------------


# (name, d) cells; the two pinch cells (fibonacci d=208, binary_increment
# d=96) exercise real width pressure, caesar is the mid-size control.
_SOLVE_CELLS = [
    ("fibonacci", 208),
    ("binary_increment", 96),
    ("caesar", 512),
]


def _solve_cfg(d, d_head):
    return dict(
        d=d, d_head=d_head, d_hidden=d, max_layers=100,
        time_budget_s=60.0, policy=SchedulingPolicy(), tighten_domains=True,
    )


@pytest.mark.parametrize("name,d", _SOLVE_CELLS)
def test_pinned_solution_valid_in_unpinned_model(name, d):
    """Solve the pinned model, then hard-fix its full decision assignment into
    the UNPINNED legacy model and re-solve: feasibility there proves the pin
    only added constraints (every pinned schedule is machine-valid by
    construction)."""
    build, _, d_head = _example_specs()[name]
    torch.manual_seed(0)
    low = _lower(build(), d)
    cfg = _solve_cfg(d, d_head)
    asg, stats = solve_schedule(low, _pin_cancels=True, **cfg)
    assert asg is not None, (
        f"{name} d={d}: pinned solve found no feasible schedule "
        f"({stats.status_name})"
    )

    built = build_cpsat_model(
        low, d=d, d_head=d_head, d_hidden=d, max_layers=100,
        policy=SchedulingPolicy(), tighten_domains=True,
        _pin_cancels=False,  # the legacy model is the verification target
    )
    for nid, L in asg.node_to_layer.items():
        built.model.Add(built.layer_var[nid] == L)
    for nid, route in asg.node_to_routing.items():
        built.model.Add(built.is_attn[nid] == (1 if route == ATTN else 0))
    for nid, mech in asg.node_to_cancel_mech.items():
        built.model.Add(built.cancel_in_mlp[nid] == (1 if mech == MLP else 0))
    for nid, cl in asg.node_to_cancel_layer.items():
        if nid in built.cancel_layer:
            built.model.Add(built.cancel_layer[nid] == cl)
        elif nid in built.input_cancel_layer:
            built.model.Add(built.input_cancel_layer[nid] == cl)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(built.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
        f"{name} d={d}: pinned solution rejected by the unpinned model "
        f"({solver.StatusName(status)}) — the pin relaxed something"
    )
    assert solver.Value(built.n_layers_var) == asg.n_layers


@pytest.mark.parametrize("name,d", _SOLVE_CELLS)
def test_pinned_optimum_no_shallower_than_unpinned(name, d):
    """A restriction can never beat the model it restricts: when both solves
    prove optimality, pinned depth >= unpinned depth."""
    build, _, d_head = _example_specs()[name]

    def _solve(pin):
        torch.manual_seed(0)
        low = _lower(build(), d)
        return solve_schedule(low, _pin_cancels=pin, **_solve_cfg(d, d_head))

    off_asg, off_stats = _solve(False)
    on_asg, on_stats = _solve(True)
    assert off_asg is not None and off_stats.is_optimal, (
        f"{name} d={d}: unpinned not optimal in budget ({off_stats.status_name})"
    )
    assert on_asg is not None and on_stats.is_optimal, (
        f"{name} d={d}: pinned not optimal in budget ({on_stats.status_name})"
    )
    assert on_asg.n_layers >= off_asg.n_layers, (
        f"{name} d={d}: pinned optimum {on_asg.n_layers} beats unpinned "
        f"{off_asg.n_layers} — impossible for a pure restriction"
    )


# ---------------------------------------------------------------------------
# Snapshot-path plumbing: the knob reaches build_model_from_snapshot
# ---------------------------------------------------------------------------


def test_pin_reaches_snapshot_path():
    """Fixture-based solves go through ``solve_schedule_from_snapshot``; prove
    the snapshot path agrees with the live path on the new default — the
    default snapshot build is pinned (equals the pinned live build) and the
    knob-off escape hatch is live there too."""
    from torchwright.compiler.forward.cpsat_scheduler import (
        build_graph_model,
        build_model_from_snapshot,
    )
    from torchwright.compiler.forward.cpsat_snapshot import (
        snapshot_from_graph_model,
    )

    node, d, d_head = _build("fibonacci")
    problem = snapshot_from_graph_model(build_graph_model(node))
    cfg = dict(d=d, d_head=d_head, d_hidden=d, max_layers=_MAX_LAYERS)
    snap_default = _proto_text(build_model_from_snapshot(problem, **cfg))
    snap_off = _proto_text(
        build_model_from_snapshot(problem, _pin_cancels=False, **cfg)
    )
    live_default = _proto_text(build_cpsat_model(node, **cfg))
    assert snap_default != snap_off, "knob dead on the snapshot path"
    assert snap_default == live_default, (
        "default (pinned) snapshot proto differs from default live"
    )


# ---------------------------------------------------------------------------
# Hint contract: a full four-family legacy hint + strict validation is fine
# ---------------------------------------------------------------------------


def test_full_hint_with_pin_passes_strict_validation():
    """A full four-family hint from a LEGACY-model solve does not blow up a
    strict pinned solve: the cancel-LAYER hints — which may contradict the
    pin (the legacy model can cancel later than earliest-legal) — are dropped
    before ``_validate_hint``; layer, routing, and cancel-MECHANISM hints are
    kept.  ``strict_hint=True`` would raise on any violation the validator
    can see."""
    build, _, d_head = _example_specs()["fibonacci"]
    d = 208
    torch.manual_seed(0)
    low = _lower(build(), d)
    cfg = _solve_cfg(d, d_head)
    donor, donor_stats = solve_schedule(low, _pin_cancels=False, **cfg)
    assert donor is not None, f"donor solve failed ({donor_stats.status_name})"
    asg, stats = solve_schedule(
        low,
        _diagnostic_hint=DiagnosticHint.from_assignment(donor),
        strict_hint=True,
        _pin_cancels=True,
        **cfg,
    )
    assert asg is not None, (
        f"pinned solve with full hint found nothing ({stats.status_name})"
    )


# ---------------------------------------------------------------------------
# End-to-end: a default (pinned) optimize=1 compile replays cleanly — the
# always-on replay-depth tripwire and I1-I4 stay silent, so the pinned
# schedule is machine-valid all the way through DirectedLayerScheduler.
# The knob-off escape hatch must replay clean too.
# ---------------------------------------------------------------------------


def test_default_pinned_compile_replays_clean():
    build, d, d_head = _example_specs()["caesar"]
    torch.manual_seed(0)
    net = forward_compile(
        d=d, d_head=d_head, output_node=build(), device="cpu",
        verbose=False, optimize=1,
    )
    # A raise above would be the replay-depth tripwire (or any compile error)
    # firing; a clean return with real layers is the tripwire staying silent.
    assert len(net.layers) > 0


def test_knob_off_compile_replays_clean():
    build, d, d_head = _example_specs()["caesar"]
    torch.manual_seed(0)
    net = forward_compile(
        d=d, d_head=d_head, output_node=build(), device="cpu",
        verbose=False, optimize=1, _pin_cancels=False, _force_resolve=True,
    )
    assert len(net.layers) > 0
