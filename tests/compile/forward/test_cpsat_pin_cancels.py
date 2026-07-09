"""_pin_cancels: earliest-legal cancel pinning (measurement-only knob).

Pins the ``_pin_cancels`` knob's contract points (plan
``cpsat_pinned_cancel_plan.md``, step 2):

- **byte-identity off** — with the knob off the built CP-SAT model proto is
  identical to the default build (production is never perturbed).
- **knob is live** — with the knob on the proto differs (the pin equalities
  are actually posted).
- **restriction invariant** — a pinned-model solution, hard-fixed into the
  UNPINNED model (layers, routings, cancel mechanisms, cancel layers, input
  cancels), is feasible there: the pin only ADDS constraints, so every
  schedule it emits is a valid default-model solution (machine-valid by
  construction).
- **hint dropping** — a full four-family hint passed to a pinned solve with
  ``strict_hint=True`` does not raise: the cancel + mechanism hints are
  dropped before validation (their families are equality-pinned), layers +
  routing are kept.
- **end-to-end replay** — an ``optimize=1`` compile with the knob on replays
  cleanly through ``DirectedLayerScheduler`` (the always-on replay-depth
  tripwire and I1–I4 stay silent).

CP-SAT solves are CPU-only, so this file runs on the plain suite.
"""

import pytest
import torch
from ortools.sat.python import cp_model

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    ATTN,
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
        calculator_v2,
        fibonacci,
        sort_digits_v1,
    )

    return {
        "calculator": (lambda: calculator_v2.create_network_parts()[0], 1024, 16),
        "caesar": (lambda: caesar_cipher.create_network_parts()[0], 512, 16),
        "sort_digits": (lambda: sort_digits_v1.create_network_parts()[0], 384, 32),
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
# Byte-identity off / knob-is-live
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_knob_off_is_byte_identical(name):
    """With the knob off the proto equals the default build (no perturbation)."""
    node, d, d_head = _build(name)
    cfg = dict(d=d, d_head=d_head, d_hidden=d, max_layers=_MAX_LAYERS)
    default = _proto_text(build_cpsat_model(node, **cfg))
    off = _proto_text(build_cpsat_model(node, _pin_cancels=False, **cfg))
    assert off == default, f"{name}: knob-off proto differs from default build"


@pytest.mark.parametrize("name", _NAMES)
def test_knob_on_changes_proto(name):
    """With the knob on the proto differs — the pins are really posted.

    Every example graph has at least one non-keep-forever node, so each gets
    its equality pin and loses its ``parked`` var + upper window."""
    node, d, d_head = _build(name)
    cfg = dict(d=d, d_head=d_head, d_hidden=d, max_layers=_MAX_LAYERS)
    off = _proto_text(build_cpsat_model(node, _pin_cancels=False, **cfg))
    on = _proto_text(build_cpsat_model(node, _pin_cancels=True, **cfg))
    assert on != off, f"{name}: knob-on proto identical to off (knob is dead)"
    assert "parked" not in on, f"{name}: pinned model still builds parked vars"


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
    the UNPINNED model and re-solve: feasibility there proves the pin only
    added constraints (every pinned schedule is machine-valid by construction).
    """
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
# Hint dropping: a full four-family hint + strict validation does not raise
# ---------------------------------------------------------------------------


def test_full_hint_with_pin_passes_strict_validation():
    """Cancel + mechanism hints are dropped before ``_validate_hint`` when the
    pin is on (the pinned model has no cancel freedom for them to warm-start);
    layer + routing hints stay.  ``strict_hint=True`` would raise on any
    violation the validator can see."""
    build, _, d_head = _example_specs()["fibonacci"]
    d = 208
    torch.manual_seed(0)
    low = _lower(build(), d)
    cfg = _solve_cfg(d, d_head)
    donor, donor_stats = solve_schedule(low, **cfg)
    assert donor is not None, f"donor solve failed ({donor_stats.status_name})"
    asg, stats = solve_schedule(
        low,
        hint_layers=donor.node_to_layer,
        hint_routing=donor.node_to_routing,
        hint_cancel=donor.node_to_cancel_layer,
        hint_cancel_mech=donor.node_to_cancel_mech,
        strict_hint=True,
        _pin_cancels=True,
        **cfg,
    )
    assert asg is not None, (
        f"pinned solve with full hint found nothing ({stats.status_name})"
    )


# ---------------------------------------------------------------------------
# End-to-end: a knob-on optimize=1 compile replays cleanly — the always-on
# replay-depth tripwire and I1-I4 stay silent, so the pinned schedule is
# machine-valid all the way through DirectedLayerScheduler.
# ---------------------------------------------------------------------------


def test_pin_on_compile_replays_clean():
    build, d, d_head = _example_specs()["caesar"]
    torch.manual_seed(0)
    net = forward_compile(
        d=d, d_head=d_head, output_node=build(), device="cpu",
        verbose=False, optimize=1, _pin_cancels=True,
    )
    # A raise above would be the replay-depth tripwire (or any compile error)
    # firing; a clean return with real layers is the tripwire staying silent.
    assert len(net.layers) > 0
