"""sym1: parked/cancel-mechanism canonicalization (measurement-only knob).

The knob targets the ``parked`` booleans of the legacy cancel-window model,
which the pinned-cancel production default (``_pin_cancels``, shipped
2026-07-10) does not build — so every build/solve here passes
``_pin_cancels=False``: sym1 has scope only on the legacy model, where it
remains reachable through the escape hatch.

Pins the ``_canonical_cancel_reps`` knob's four contract points (plan
``cpsat_symmetry_sym1_plan.md`` §5.2):

- **byte-identity off** — with the knob off the built CP-SAT model proto is
  identical to the default build (production is never perturbed).
- **knob is live** — with the knob on the proto differs (the two implications
  are actually posted for every eligible node).
- **canonical form on** — in a solve with the knob on, no parked node
  (``cancel_layer == max_layers``) keeps an MLP cancel mechanism: D-1 pins the
  dangling ``cancel_in_mlp`` boolean to the attention side.
- **depth invariance** — knob on vs off returns the identical proven-optimal
  layer count, including the two pinch cells (fibonacci d=208,
  binary_increment d=96) where the floor is tight.

CP-SAT solves are CPU-only, so this file runs on the plain suite.
"""

import pytest
import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    ATTN,
    build_cpsat_model,
    solve_schedule,
)
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.compiler.lower import lower


# ---------------------------------------------------------------------------
# The six example graphs (matches scripts/intralayer_example_sweep.py and
# test_cpsat_snapshot.py).
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
    """With the knob off the proto equals the plain legacy build."""
    node, d, d_head = _build(name)
    cfg = dict(
        d=d, d_head=d_head, d_hidden=d, max_layers=_MAX_LAYERS,
        _pin_cancels=False,
    )
    default = _proto_text(build_cpsat_model(node, **cfg))
    off = _proto_text(build_cpsat_model(node, _canonical_cancel_reps=False, **cfg))
    assert off == default, f"{name}: knob-off proto differs from legacy build"


@pytest.mark.parametrize("name", _NAMES)
def test_knob_on_changes_proto(name):
    """With the knob on the legacy proto differs — the implications are really
    posted.

    Every example graph has at least one non-keep-forever node with a cancel
    window (``cancel_slack`` default 2), so each gets a ``parked`` var and the
    D-2 converse ``cancel_layer <= max_layers - 1`` under ``parked.Not()``."""
    node, d, d_head = _build(name)
    cfg = dict(
        d=d, d_head=d_head, d_hidden=d, max_layers=_MAX_LAYERS,
        _pin_cancels=False,
    )
    off = _proto_text(build_cpsat_model(node, _canonical_cancel_reps=False, **cfg))
    on = _proto_text(build_cpsat_model(node, _canonical_cancel_reps=True, **cfg))
    assert on != off, f"{name}: knob-on proto identical to off (knob is dead)"


# ---------------------------------------------------------------------------
# Canonical form: no parked node keeps an MLP cancel (D-1 observable)
# ---------------------------------------------------------------------------


# (name, d) cells; the two pinch cells solve fast and exercise real parking.
_SOLVE_CELLS = [
    ("fibonacci", 208),
    ("binary_increment", 96),
    ("caesar", 512),
    ("bucketed_argmin", 512),
]


@pytest.mark.parametrize("name,d", _SOLVE_CELLS)
def test_canonical_form_on_solved_model(name, d):
    build, _, d_head = _example_specs()[name]
    torch.manual_seed(0)
    low = _lower(build(), d)
    asg, stats = solve_schedule(
        low, d=d, d_head=d_head, d_hidden=d, max_layers=100,
        time_budget_s=60.0, policy=SchedulingPolicy(),
        tighten_domains=True, _canonical_cancel_reps=True,
        _pin_cancels=False,
    )
    assert asg is not None and stats.is_optimal, (
        f"{name} d={d}: expected a proven-optimal solve (got "
        f"{stats.status_name})"
    )
    max_layers = 100
    # node_to_cancel_mech is keyed by exactly the non-keep-forever schedulable
    # nodes (those with a cancel_in_mlp var).  D-1: a parked node
    # (cancel_layer == max_layers) must carry the attention mechanism.
    parked_mech = [
        (nid, mech)
        for nid, mech in asg.node_to_cancel_mech.items()
        if asg.node_to_cancel_layer[nid] == max_layers
    ]
    assert all(mech == ATTN for _, mech in parked_mech), (
        f"{name} d={d}: parked node with non-attention cancel mechanism "
        f"{[p for p in parked_mech if p[1] != ATTN]}"
    )
    assert stats.parked_count >= 0


# ---------------------------------------------------------------------------
# Depth invariance (incl. the two pinch cells): knob on vs off ties, both
# proven optimal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,d", _SOLVE_CELLS)
def test_depth_invariance_on_vs_off(name, d):
    build, _, d_head = _example_specs()[name]

    def _solve(canonical):
        torch.manual_seed(0)
        low = _lower(build(), d)
        return solve_schedule(
            low, d=d, d_head=d_head, d_hidden=d, max_layers=100,
            time_budget_s=60.0, policy=SchedulingPolicy(),
            tighten_domains=True, _canonical_cancel_reps=canonical,
            _pin_cancels=False,
        )

    off_asg, off_stats = _solve(False)
    on_asg, on_stats = _solve(True)
    assert off_asg is not None and off_stats.is_optimal, f"{name} d={d}: off not optimal"
    assert on_asg is not None and on_stats.is_optimal, f"{name} d={d}: on not optimal"
    assert on_asg.n_layers == off_asg.n_layers, (
        f"{name} d={d}: canonicalization changed the optimal depth "
        f"(off={off_asg.n_layers}, on={on_asg.n_layers})"
    )


# ---------------------------------------------------------------------------
# End-to-end: a knob-on optimize=1 compile replays cleanly — the always-on
# replay-depth tripwire (gap #5) stays silent and the depth matches knob-off.
# The removed model solutions replay identically to retained ones (plan §3),
# so the sound directed replay must land at the solved n_layers.
# ---------------------------------------------------------------------------


def test_knob_on_compile_replays_clean():
    build, d, d_head = _example_specs()["caesar"]

    def _compile(canonical):
        torch.manual_seed(0)
        net = forward_compile(
            d=d, d_head=d_head, output_node=build(), device="cpu",
            verbose=False, optimize=1, _canonical_cancel_reps=canonical,
            _pin_cancels=False, _force_resolve=True,
        )
        return len(net.layers)

    # A raise here would be the replay-depth tripwire firing (or any compile
    # error); a clean return is the tripwire staying silent.
    off_layers = _compile(False)
    on_layers = _compile(True)
    assert on_layers == off_layers, (
        f"caesar: knob-on compile depth {on_layers} != knob-off {off_layers}"
    )
