"""SchedulingProblem snapshot: round-trip + CP-SAT proto equivalence.

The load-bearing guarantee (the fixture layer's whole point): the CP-SAT model
built from a live graph and the model built from that graph's round-tripped
snapshot are IDENTICAL.  Pinned on the six example graphs via proto text
comparison (OR-Tools 9.15's pybind proto has no binary ``SerializeToString``;
``str(proto)`` is the canonical, complete text format).

Also covers: object round-trip, canonical persistence + the stale-fixture
fingerprint gate, the hint-widening build path, solve equivalence, and the
proto-dump zero-rebuild re-solve.
"""

import pytest
import torch

from torchwright.compiler.forward.cpsat_scheduler import (
    build_cpsat_model,
    build_graph_model,
    build_model_from_snapshot,
    dump_model_proto,
    resolve_model_proto,
    solve_schedule,
    solve_schedule_from_snapshot,
)
from torchwright.compiler.forward.cpsat_snapshot import (
    FORMAT_VERSION,
    FrozenHint,
    SchedulingProblem,
    snapshot_from_graph_model,
)
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy


# ---------------------------------------------------------------------------
# The six example graphs (matches scripts/intralayer_example_sweep.py)
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


def _proto_text(built):
    return str(built.model.Proto())


# ---------------------------------------------------------------------------
# Proto equivalence: live graph vs round-tripped snapshot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("tighten", [False, True])
def test_proto_identical_live_vs_roundtripped_snapshot(name, tighten):
    node, d, d_head = _build(name)
    cfg = dict(d=d, d_head=d_head, d_hidden=d, max_layers=60, tighten_domains=tighten)

    live = _proto_text(build_cpsat_model(node, **cfg))

    gm = build_graph_model(node)
    problem = snapshot_from_graph_model(gm)
    roundtripped = SchedulingProblem.loads(problem.dumps())
    snap = _proto_text(build_model_from_snapshot(roundtripped, **cfg))

    assert snap == live, f"{name} tighten={tighten}: snapshot proto != live proto"


@pytest.mark.parametrize("name", _NAMES)
def test_problem_object_roundtrips(name):
    node, _, _ = _build(name)
    problem = snapshot_from_graph_model(build_graph_model(node))
    assert SchedulingProblem.loads(problem.dumps()).to_json() == problem.to_json()


def test_canonicalized_snapshot_rebuilds_a_valid_model():
    """A canonical (on-disk) snapshot rebuilds a self-consistent model that
    re-solves; its ids differ from live, so its proto is its own — but stable."""
    node, d, d_head = _build("caesar")
    cfg = dict(d=d, d_head=d_head, d_hidden=d, max_layers=40)
    problem = snapshot_from_graph_model(build_graph_model(node)).canonicalized(node)
    a = _proto_text(build_model_from_snapshot(SchedulingProblem.loads(problem.dumps()), **cfg))
    b = _proto_text(build_model_from_snapshot(SchedulingProblem.loads(problem.dumps()), **cfg))
    assert a == b  # canonical rebuild is deterministic across loads


# ---------------------------------------------------------------------------
# Hint-widening build path (build_cpsat_model reads hints to size cancel windows)
# ---------------------------------------------------------------------------


def test_hinted_build_proto_identical():
    # Legacy (knob-off) model: the widening machinery only exists there —
    # the pinned production default skips window sizing entirely, which
    # would make this proto comparison vacuous on the widening path.
    node, d, d_head = _build("caesar")
    cfg = dict(
        d=d, d_head=d_head, d_hidden=d, max_layers=40, tighten_domains=True,
        _pin_cancels=False,
    )
    # A feasible schedule -> hint dicts that exercise the cancel-window widening.
    asg, _ = solve_schedule(node, time_budget_s=20.0, **cfg)
    assert asg is not None
    hint_layers = dict(asg.node_to_layer)
    hint_cancel = dict(asg.node_to_cancel_layer)

    live = _proto_text(
        build_cpsat_model(node, hint_layers=hint_layers, hint_cancel=hint_cancel, **cfg)
    )
    problem = SchedulingProblem.loads(
        snapshot_from_graph_model(build_graph_model(node)).dumps()
    )
    snap = _proto_text(
        build_model_from_snapshot(
            problem, hint_layers=hint_layers, hint_cancel=hint_cancel, **cfg
        )
    )
    assert snap == live


def test_frozen_hint_roundtrips():
    h = FrozenHint(
        layers={1: 0, 2: 3},
        routing={1: "attn", 2: "mlp"},
        cancel={1: 5, 2: 6},
        cancel_mech={2: "mlp"},
        n_layers=7,
    )
    assert FrozenHint.from_json(h.to_json()) == h
    remapped = h.remap({1: 100, 2: 200})
    assert remapped.layers == {100: 0, 200: 3}
    assert remapped.n_layers == 7


# ---------------------------------------------------------------------------
# Solve equivalence
# ---------------------------------------------------------------------------


def test_solve_from_snapshot_matches_live_solve():
    node, d, d_head = _build("caesar")
    kw = dict(
        d=d, d_head=d_head, d_hidden=d, max_layers=40, time_budget_s=30.0,
        tighten_domains=True, policy=SchedulingPolicy(),
        solver_params={"random_seed": 1},
    )
    live_asg, live_stats = solve_schedule(node, **kw)
    problem = SchedulingProblem.loads(
        snapshot_from_graph_model(build_graph_model(node)).canonicalized(node).dumps()
    )
    snap_asg, snap_stats = solve_schedule_from_snapshot(problem, **kw)
    assert live_asg is not None and snap_asg is not None
    assert live_stats.status_name == snap_stats.status_name == "OPTIMAL"
    assert live_asg.n_layers == snap_asg.n_layers


# ---------------------------------------------------------------------------
# Persistence: fingerprint gate, refusals, format-version
# ---------------------------------------------------------------------------


def test_save_load_with_fingerprint_gate(tmp_path):
    node, d, d_head = _build("binary_increment")
    geom = dict(
        d=d, d_head=d_head, d_hidden=d, flex_routing=True, cancel_slack=2,
        policy=SchedulingPolicy(), max_layers=60,
    )
    cp = 5  # arbitrary stand-in for the lowered critical path in this unit test
    problem = (
        snapshot_from_graph_model(build_graph_model(node))
        .canonicalized(node)
        .with_identity(output_node=node, critical_path_layers=cp, **geom)
    )
    path = tmp_path / "fixture.json"
    problem.save(path)
    fp = problem.identity.fingerprint

    loaded = SchedulingProblem.load(path, expected_fingerprint=fp)
    assert loaded.identity.fingerprint == fp
    # A wrong (stale) fingerprint must fail loudly, never silently measure.
    with pytest.raises(ValueError, match="does not match the current graph"):
        SchedulingProblem.load(path, expected_fingerprint="deadbeef")


def test_save_refuses_live_ids_and_missing_identity(tmp_path):
    node, _, _ = _build("binary_increment")
    live = snapshot_from_graph_model(build_graph_model(node))
    with pytest.raises(ValueError, match="live-id snapshot"):
        live.save(tmp_path / "x.json")
    with pytest.raises(ValueError, match="without identity"):
        live.canonicalized(node).save(tmp_path / "x.json")


def test_format_version_mismatch_raises():
    node, _, _ = _build("binary_increment")
    d = snapshot_from_graph_model(build_graph_model(node)).to_json()
    d["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="format_version"):
        SchedulingProblem.from_json(d)


# ---------------------------------------------------------------------------
# Proto dump + zero-rebuild re-solve
# ---------------------------------------------------------------------------


def test_proto_dump_and_resolve(tmp_path):
    node, d, d_head = _build("caesar")
    cfg = dict(d=d, d_head=d_head, d_hidden=d, max_layers=40, tighten_domains=True)
    built = build_cpsat_model(node, **cfg)
    path = tmp_path / "model.cpproto"
    dump_model_proto(built, path)
    assert path.exists() and path.with_suffix(".cpproto.meta.json").exists()

    result = resolve_model_proto(path, time_budget_s=30.0, workers=4)
    assert result["status_name"] == "OPTIMAL"
    # depth recovered from the proto matches a direct solve of the same model.
    direct_asg, _ = solve_schedule(
        node, time_budget_s=30.0, solver_params={"random_seed": 0}, **cfg
    )
    assert direct_asg is not None
    assert result["n_layers"] == direct_asg.n_layers
