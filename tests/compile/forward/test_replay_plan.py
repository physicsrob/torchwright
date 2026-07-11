import dataclasses

import pytest
import torch

from torchwright.compiler.forward.compile import (
    _choose_dominating_replay_plan,
    _replay_plan_objective,
    forward_compile,
)
from torchwright.compiler.forward.cpsat_scheduler import Costs, ScheduleAssignment
from torchwright.compiler.forward.replay_plan import (
    PlannedAttentionOp,
    PlannedLayer,
    PlannedMlpOp,
    ReplayPlan,
    planned_layer_shape,
)
from torchwright.compiler.forward.scheduler import DirectedLayerScheduler
from torchwright.compiler.forward.schedule_cache import store_assignment
from torchwright.compiler.forward.weight_writer import AttnHeadOp, write_attn_sublayer
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.compiler.token_model import LayerShape
from torchwright.graph import Linear
from torchwright.ops.inout_nodes import create_input


def test_planned_operations_defensively_freeze_scheduler_lists():
    node = create_input("x", 2)
    target = [3, 4]
    source = [1, 2]
    mutable = AttnHeadOp("cancel", node, target, source_cols=source)

    planned = PlannedAttentionOp.from_scheduler_op(mutable)
    target.append(5)
    source[0] = 99

    assert planned.target_cols == (3, 4)
    assert planned.source_cols == (1, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        planned.target_cols = ()


def test_planned_accounting_drives_shape_and_weighted_counts():
    x = create_input("x", 20)
    add_like = PlannedAttentionOp(
        "compute_add",
        x,
        tuple(range(20)),
        source_cols=tuple(range(20)),
        source_cols_b=tuple(range(20, 40)),
    )
    bypass = PlannedMlpOp("compute_linear_bypass", x, (1, 2), mlp_slots=(3, 4, 5, 6))

    shape, heads, bypass_slots = planned_layer_shape(
        (add_like,), (bypass,), d=64, d_head=16, d_hidden=32, trim_heads=True
    )

    assert shape == LayerShape(n_heads=3, d_hidden=7)
    assert heads == 3
    assert bypass_slots == 4


def test_writer_rejects_mutable_scheduler_operations():
    node = create_input("x", 1)
    op = AttnHeadOp("cancel", node, [0])
    with pytest.raises(TypeError, match="PlannedAttentionOp"):
        write_attn_sublayer(TransformerLayer(16, 16), [op], 15)


def test_selected_assignment_has_one_directed_walk(monkeypatch):
    x = create_input("x", 2)
    out = Linear(x, torch.eye(2), torch.zeros(2))
    calls = 0
    real_schedule_layer = DirectedLayerScheduler.schedule_layer

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return real_schedule_layer(self, *args, **kwargs)

    monkeypatch.setattr(DirectedLayerScheduler, "schedule_layer", counted)
    net = forward_compile(d=16, d_head=16, output_node=out, optimize=0, verbose=False)

    assert calls == net.schedule_result.assignment.n_layers
    assert len(net.layers) == net.schedule_result.assignment.n_layers


def test_trivial_graph_uses_placeholder_layer_without_planned_operations():
    x = create_input("x", 2)
    net = forward_compile(d=16, d_head=16, output_node=x, optimize=0, verbose=False)

    assert net.schedule_result.assignment.n_layers == 0
    assert len(net.layers) == 1
    assert net.residual_assignment.get_node_indices(net.layers[0].attn.in_state, x)


def test_weighted_objective_uses_concrete_snapshot_occupancy():
    x = create_input("x", 4)
    y = Linear(x, torch.eye(4), torch.zeros(4))
    assignment = ScheduleAssignment(
        node_to_layer={y.node_id: 0},
        # The assignment asks for cancellation at layer 0, but the concrete
        # snapshots retain x: realized dominance must charge that deferral.
        node_to_cancel_layer={x.node_id: 0, y.node_id: 2},
        node_to_routing={y.node_id: "mlp"},
        n_layers=2,
    )
    layer = PlannedLayer(
        attention_ops=(),
        mlp_ops=(),
        biased_linear_ids=frozenset(),
        shape=LayerShape(1, 1),
        residual_snapshot=(
            (x.node_id, (0, 1, 2, 3)),
            (y.node_id, (4, 5, 6, 7)),
        ),
        newly_computed_ids=(y.node_id,),
        emitted_attention_heads=0,
        mlp_bypass_slots=0,
    )
    plan = ReplayPlan(
        assignment=assignment,
        layers=(layer, layer),
        input_indices=((x.node_id, (0, 1, 2, 3)),),
        final_indices=((y.node_id, (4, 5, 6, 7)),),
        nodes_by_id=((x.node_id, x), (y.node_id, y)),
        const_one_col=15,
    )

    # primary=2 layers, realized waste=4 columns * two retained snapshots
    assert (
        _replay_plan_objective(plan, Costs(alpha=1, waste=1), objective_scale=100)
        == 208
    )


def test_worse_realized_weighted_candidate_cannot_replace_incumbent():
    x = create_input("x", 1)
    assignment = ScheduleAssignment(
        node_to_layer={},
        node_to_cancel_layer={x.node_id: 1},
        node_to_routing={},
        n_layers=1,
    )
    base_layer = PlannedLayer(
        attention_ops=(),
        mlp_ops=(),
        biased_linear_ids=frozenset(),
        shape=LayerShape(1, 1),
        residual_snapshot=((x.node_id, (0,)),),
        newly_computed_ids=(),
        emitted_attention_heads=1,
        mlp_bypass_slots=0,
    )
    incumbent = ReplayPlan(
        assignment,
        (base_layer,),
        ((x.node_id, (0,)),),
        ((x.node_id, (0,)),),
        ((x.node_id, x),),
        15,
    )
    candidate = dataclasses.replace(
        incumbent,
        layers=(dataclasses.replace(base_layer, emitted_attention_heads=3),),
    )

    winner, diagnostics = _choose_dominating_replay_plan(
        Costs(alpha=1, beta=10), incumbent, candidate, objective_scale=1
    )

    assert winner is incumbent
    assert diagnostics["candidate_objective"] > diagnostics["incumbent_objective"]


def test_weighted_cache_ratchets_on_realized_objective(tmp_path, monkeypatch):
    monkeypatch.setenv("TW_SCHEDULE_CACHE_DIR", str(tmp_path))
    x = create_input("x", 1)
    y = Linear(x, torch.ones(1, 1), torch.zeros(1))
    assignment = ScheduleAssignment(
        node_to_layer={y.node_id: 0},
        node_to_cancel_layer={x.node_id: 1, y.node_id: 1},
        node_to_routing={y.node_id: "mlp"},
        n_layers=1,
    )

    assert store_assignment("weighted", assignment, {"realized_objective": 20}, y)
    assert store_assignment("weighted", assignment, {"realized_objective": 10}, y)
    assert not store_assignment("weighted", assignment, {"realized_objective": 30}, y)
