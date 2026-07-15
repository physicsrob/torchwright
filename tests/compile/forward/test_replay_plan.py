import dataclasses
from types import SimpleNamespace

import pytest
import torch

from torchwright.compiler.forward.compile import (
    _choose_dominating_replay_plan,
    _count_heads_by_type,
    _count_layer_params,
    _replay_plan_objective,
    forward_compile,
)
from torchwright.compiler.forward.cpsat_scheduler import (
    Costs,
    ScheduleAssignment,
    ScheduleResult,
    SchedulingProvenance,
    SolveStats,
)
from torchwright.compiler.forward.replay_plan import (
    PlannedAttentionOp,
    PlannedLayer,
    PlannedMlpOp,
    ReplayPlan,
    planned_layer_shape,
)
from torchwright.compiler.forward.scheduler import DirectedLayerScheduler, _AttentionOp
from torchwright.compiler.forward.schedule_cache import store_assignment
from torchwright.compiler.forward.weight_writer import write_attn_sublayer
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.compiler.token_model import (
    CompileHeader,
    LayerShape,
    make_layer_callback,
    schedule_provenance,
)
from torchwright.graph import Attn, Linear
from torchwright.ops.inout_nodes import create_input


def test_planned_operations_defensively_freeze_scheduler_lists():
    node = create_input("x", 2)
    target = [3, 4]
    source = [1, 2]
    mutable = _AttentionOp("cancel", node, target, source_cols=source)

    planned = PlannedAttentionOp.from_scheduler_op(mutable)
    target.append(5)
    source[0] = 99

    assert planned.target_cols == (3, 4)
    assert planned.source_cols == (1, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        planned.target_cols = ()


def test_planned_operations_reject_tensor_indices():
    node = create_input("x", 2)
    with pytest.raises(TypeError, match="target_cols must contain only integers"):
        PlannedAttentionOp("cancel", node, torch.tensor([0, 1]))


def test_reuse_records_require_the_occurrence_index():
    """Every reuse record (add_into, add_into_bypass) carries exactly one
    valid target-occurrence index; every fresh or unrelated record rejects
    one (docs/plan_additional_mlp_routing.md)."""
    from torchwright.graph import Add

    a = create_input("a", 2)
    b = create_input("b", 2)
    add = Add(a, b)

    # add_into: required, 0 or 1 only.
    with pytest.raises(ValueError, match="requires reuse_input_index"):
        PlannedAttentionOp("add_into", add, (0, 1), source_cols=(2, 3))
    with pytest.raises(ValueError, match="requires reuse_input_index"):
        PlannedAttentionOp(
            "add_into", add, (0, 1), source_cols=(2, 3), reuse_input_index=2
        )
    op = PlannedAttentionOp(
        "add_into", add, (0, 1), source_cols=(2, 3), reuse_input_index=1
    )
    assert op.reuse_input_index == 1

    # Fresh and unrelated attention records reject an index.
    with pytest.raises(ValueError, match="must not carry"):
        PlannedAttentionOp(
            "compute_add",
            add,
            (0, 1),
            source_cols=(2, 3),
            source_cols_b=(4, 5),
            reuse_input_index=0,
        )
    with pytest.raises(ValueError, match="must not carry"):
        PlannedAttentionOp("cancel", None, (0, 1), reuse_input_index=0)

    # add_into_bypass: required; source_cols required.
    with pytest.raises(ValueError, match="requires reuse_input_index"):
        PlannedMlpOp(
            "add_into_bypass", add, (0, 1), mlp_slots=(2, 3, 4, 5), source_cols=(6, 7)
        )
    with pytest.raises(ValueError, match="requires source_cols"):
        PlannedMlpOp(
            "add_into_bypass", add, (0, 1), mlp_slots=(2, 3, 4, 5), reuse_input_index=0
        )
    mlp_op = PlannedMlpOp(
        "add_into_bypass",
        add,
        (0, 1),
        mlp_slots=(2, 3, 4, 5),
        source_cols=(6, 7),
        reuse_input_index=0,
    )
    assert mlp_op.reuse_input_index == 0

    # Fresh and unrelated MLP records reject an index.
    with pytest.raises(ValueError, match="must not carry"):
        PlannedMlpOp(
            "compute_add_bypass",
            add,
            (0, 1),
            mlp_slots=(2, 3, 4, 5),
            source_cols=(6, 7),
            source_cols_b=(8, 9),
            reuse_input_index=0,
        )
    with pytest.raises(ValueError, match="must not carry"):
        PlannedMlpOp(
            "compute_linear_bypass",
            None,
            (0, 1),
            mlp_slots=(2, 3, 4, 5),
            source_cols=(6, 7),
            reuse_input_index=0,
        )


def test_compute_add_bypass_source_field_rules():
    """compute_add_bypass requires both source lists; every other MLP record
    rejects source_cols_b."""
    from torchwright.graph import Add

    a = create_input("a", 2)
    b = create_input("b", 2)
    add = Add(a, b)

    with pytest.raises(ValueError, match="source_cols and source_cols_b"):
        PlannedMlpOp(
            "compute_add_bypass",
            add,
            (0, 1),
            mlp_slots=(2, 3, 4, 5),
            source_cols=(6, 7),
        )
    with pytest.raises(ValueError, match="must not carry source_cols_b"):
        PlannedMlpOp(
            "compute_linear_bypass",
            None,
            (0, 1),
            mlp_slots=(2, 3, 4, 5),
            source_cols=(6, 7),
            source_cols_b=(8, 9),
        )
    op = PlannedMlpOp(
        "compute_add_bypass",
        add,
        (0, 1),
        mlp_slots=(2, 3, 4, 5),
        source_cols=(6, 7),
        source_cols_b=(8, 9),
    )
    assert op.source_cols_b == (8, 9)


def test_add_bypass_ops_count_as_bypass_slot_consumers():
    from torchwright.graph import Add

    a = create_input("a", 2)
    b = create_input("b", 2)
    add = Add(a, b)
    reused = PlannedMlpOp(
        "add_into_bypass",
        add,
        (0, 1),
        mlp_slots=(2, 3, 4, 5),
        source_cols=(6, 7),
        reuse_input_index=0,
    )
    fresh = PlannedMlpOp(
        "compute_add_bypass",
        add,
        (0, 1),
        mlp_slots=(2, 3, 4, 5),
        source_cols=(6, 7),
        source_cols_b=(8, 9),
    )
    bias = PlannedMlpOp("compute_bias", None, (0, 1))
    assert reused.bypass_slot_count == 4
    assert fresh.bypass_slot_count == 4
    assert bias.bypass_slot_count == 0


def test_replay_plan_rejects_missing_node_resolution():
    x = create_input("x", 1)
    assignment = ScheduleAssignment(
        node_to_layer={},
        node_to_cancel_layer={x.node_id: 0},
        node_to_routing={},
        n_layers=0,
    )
    layer = PlannedLayer(
        attention_ops=[],
        mlp_ops=[],
        biased_linear_ids=set(),
        shape=LayerShape(1, 1),
        residual_snapshot=[(x.node_id, [0])],
        newly_computed_ids=[],
        emitted_attention_heads=0,
        mlp_bypass_slots=0,
    )

    assert layer.attention_ops == ()
    assert layer.residual_snapshot == ((x.node_id, (0,)),)
    with pytest.raises(ValueError, match="does not resolve"):
        ReplayPlan(
            assignment=assignment,
            layers=[layer],
            input_indices=[(x.node_id, [0])],
            final_indices=[(x.node_id, [0])],
            nodes_by_id=[],
            const_one_col=15,
        )


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
    assert _count_heads_by_type((add_like,), 16) == {"compute_add": 3}
    assert _count_layer_params((add_like,), (bypass,), 64, 16) == (
        3 * 4 * 64 * 16 + 4 * (2 * 64 + 2)
    )


def test_planned_attention_heads_exclude_zero_output_chunks():
    q = create_input("q", 4)
    k = create_input("k", 4)
    v = create_input("v", 4)
    output = torch.zeros(8, 2)
    output[4:, :] = 1.0
    node = Attn(
        q,
        k,
        v,
        torch.eye(4),
        torch.eye(4),
        torch.ones(4, 8),
        output,
    )
    op = PlannedAttentionOp(
        "compute_attn",
        node,
        (8, 9),
        source_cols=(4, 5, 6, 7),
        q_source_cols=(0, 1, 2, 3),
        k_source_cols=(0, 1, 2, 3),
    )

    shape, heads, _ = planned_layer_shape(
        (op,), (), d=16, d_head=4, d_hidden=16, trim_heads=True
    )

    assert heads == 1
    assert shape.n_heads == 1


def test_writer_rejects_mutable_scheduler_operations():
    node = create_input("x", 1)
    op = _AttentionOp("cancel", node, [0])
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
    class Sink:
        def begin(self, header):
            self.header = header

        def write_layer(self, index, weights):
            self.written = (index, weights)

    x = create_input("x", 2)
    sink = Sink()
    net = forward_compile(
        d=16,
        d_head=16,
        output_node=x,
        optimize=0,
        verbose=False,
        on_layer_compiled=make_layer_callback(CompileHeader(16, 16, True, True), sink),
    )

    assert net.schedule_result.assignment.n_layers == 0
    assert len(net.layers) == 1
    assert sink.header.layer_shapes == (LayerShape(1, 1),)
    assert sink.written[0] == 0
    assert sink.written[1].attention.n_heads == 1
    assert sink.written[1].d_hidden == 1
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

    assert store_assignment(
        "weighted", assignment, {"realized_objective_blocks": (20, 0)}, y
    )
    assert store_assignment(
        "weighted", assignment, {"realized_objective_blocks": (10, 0)}, y
    )
    assert not store_assignment(
        "weighted", assignment, {"realized_objective_blocks": (30, 0)}, y
    )


def test_weighted_cache_compares_blocks_across_different_scales(tmp_path, monkeypatch):
    monkeypatch.setenv("TW_SCHEDULE_CACHE_DIR", str(tmp_path))
    x = create_input("x", 1)
    y = Linear(x, torch.ones(1, 1), torch.zeros(1))
    assignment = ScheduleAssignment(
        node_to_layer={y.node_id: 0},
        node_to_cancel_layer={x.node_id: 1, y.node_id: 1},
        node_to_routing={y.node_id: "mlp"},
        n_layers=1,
    )

    assert store_assignment(
        "scaled",
        assignment,
        {
            "realized_objective": 210,
            "realized_objective_blocks": (2, 10),
            "objective_scale": 100,
        },
        y,
    )
    # The raw total is smaller only because its scale differs. Its secondary
    # block is worse, so it must not replace the incumbent.
    assert not store_assignment(
        "scaled",
        assignment,
        {
            "realized_objective": 22,
            "realized_objective_blocks": (2, 20),
            "objective_scale": 1,
        },
        y,
    )


def test_artifact_provenance_does_not_transfer_solver_optimality_to_selection():
    stats = SolveStats(
        status_name="OPTIMAL",
        objective_value=10,
        best_objective_bound=10,
        wall_time_s=1.0,
        solver_log="",
        total_attn_heads=1,
        total_mlp_bypass_slots=0,
        is_optimal=True,
    )
    assignment = ScheduleAssignment({}, {}, {}, 0)
    compiled = SimpleNamespace(
        cpsat_solve_stats=stats,
        schedule_result=ScheduleResult(
            assignment,
            SchedulingProvenance(
                origin="heuristic",
                delivery="fresh",
                selected_is_optimal=False,
                selected_objective=9,
                selected_objective_blocks=(9, 0),
                solver_attempt=stats,
            ),
        ),
    )

    provenance = schedule_provenance(compiled, optimize=1).to_dict()

    assert provenance["selected_origin"] == "heuristic"
    assert provenance["selected_is_optimal"] is False
    assert provenance["selected_objective"] == 9
    assert provenance["solver_status"] == "OPTIMAL"
    assert provenance["solver_is_optimal"] is True
