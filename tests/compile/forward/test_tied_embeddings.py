"""Held-bank embedding/output allocation and scheduling regressions."""

import torch
import pytest

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    build_cpsat_model,
    build_graph_model,
    build_model_from_snapshot,
    solve_schedule,
)
from torchwright.compiler.forward.cpsat_snapshot import snapshot_from_graph_model
from torchwright.compiler.graph_identity import graph_fingerprint
from torchwright.graph import Add, Concatenate, Embedding, Linear
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import LiteralValue


def _embedding(width=4):
    table = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, 3.0]],
        dtype=torch.float32,
    )
    assert width == 4
    return Embedding(["a", "b"], width, table=table, special_tokens=[])


def _identity(inp, name):
    return Linear(inp, torch.eye(len(inp)), name=name)


def _relu_identity(inp, name):
    width = len(inp)
    eye = torch.eye(width)
    return FFN(
        inp,
        gate_proj=torch.cat([eye, -eye], dim=0),
        gate_bias=torch.zeros(2 * width),
        out_proj=torch.cat([eye, -eye], dim=0),
        out_bias=torch.zeros(width),
        name=name,
    )


def _banks(net, embedding, output):
    ra = net.residual_assignment
    assert ra is not None
    in_state = net.layers[0].attn.in_state
    out_state = net.layers[-1].mlp.out_state
    return (
        ra.get_node_indices(in_state, embedding),
        ra.get_node_indices(out_state, output),
    )


def test_direct_attention_handoff_preserves_order_and_clears_const_seed():
    embedding = _embedding()
    output = _identity(embedding, "output")
    net = forward_compile(
        d=16,
        d_head=4,
        output_node=output,
        output_layout_source=embedding,
        rms_norm=False,
        verbose=False,
    )

    assert _banks(net, embedding, output)[0] == _banks(net, embedding, output)[1]
    got = net.compute(2, {embedding.input_name: torch.tensor([0, 1])})[output]
    assert torch.equal(got.cpu(), embedding.table)

    ra = net.residual_assignment
    in_state = net.layers[0].attn.in_state
    const_nodes = [
        n
        for n in ra.get_nodes(in_state)
        if isinstance(n, LiteralValue) and n.name == "rope_self_match_const_one"
    ]
    assert len(const_nodes) == 1
    const_col = ra.get_node_indices(in_state, const_nodes[0])[0]
    assert const_nodes[0] not in ra.get_nodes(net.layers[-1].mlp.out_state)
    assert net.layers[-1].mlp.linear2.output_bias[const_col].item() == -1.0


def test_bank_can_be_held_across_intermediate_layers():
    embedding = _embedding()
    # FFNs cannot collapse into one affine node and consecutive MLP writers
    # require distinct layers, creating a real clean-held interval.
    a = _relu_identity(embedding, "a")
    b = _relu_identity(a, "b")
    output = _relu_identity(b, "output")
    net = forward_compile(
        d=20,
        d_head=4,
        d_hidden=16,
        output_node=output,
        output_layout_source=embedding,
        rms_norm=False,
        verbose=False,
    )
    bank, output_bank = _banks(net, embedding, output)
    assert bank == output_bank

    # At least one post-layer state is inside the clean held interval: the
    # source has died and the target has not yet been born.  No ordinary live
    # node may occupy even one bank column in that state.
    ra = net.residual_assignment
    held_states = [
        layer.mlp.out_state
        for layer in net.layers
        if embedding not in ra.get_nodes(layer.mlp.out_state)
        and output not in ra.get_nodes(layer.mlp.out_state)
    ]
    assert held_states
    for state in held_states:
        for node in ra.get_nodes(state):
            assert set(ra.get_node_indices(state, node)).isdisjoint(bank)


def test_intermediate_add_cannot_inherit_tied_source_bank():
    embedding = _embedding()
    branch = _identity(embedding, "branch")
    intermediate = Add(embedding, branch)
    output = _identity(intermediate, "output")
    net = forward_compile(
        d=20,
        d_head=4,
        d_hidden=20,
        output_node=output,
        output_layout_source=embedding,
        rms_norm=False,
        verbose=False,
    )

    bank, output_bank = _banks(net, embedding, output)
    assert bank == output_bank
    ra = net.residual_assignment
    for layer in net.layers[:-1]:
        state = layer.mlp.out_state
        if intermediate in ra.get_nodes(state):
            assert set(ra.get_node_indices(state, intermediate)).isdisjoint(bank)


def test_output_add_is_fresh_compute_not_add_into():
    embedding = _embedding()
    left = _identity(embedding, "left")
    right = _identity(embedding, "right")
    output = Add(left, right)
    net = forward_compile(
        d=20,
        d_head=4,
        d_hidden=20,
        output_node=output,
        output_layout_source=embedding,
        rms_norm=False,
        verbose=False,
    )
    output_ops = [e.op_type for e in net.placements.entries if e.node is output]
    assert "compute_add" in output_ops
    assert "add_into" not in output_ops
    assert _banks(net, embedding, output)[0] == _banks(net, embedding, output)[1]


def test_direct_mlp_only_target_is_rejected_before_scheduling():
    embedding = _embedding()
    output = FFN(
        embedding,
        gate_proj=torch.eye(4),
        gate_bias=torch.zeros(4),
        out_proj=torch.eye(4),
        out_bias=torch.zeros(4),
    )
    with pytest.raises(ValueError, match="no direct MLP handoff"):
        forward_compile(
            d=16,
            d_head=4,
            output_node=output,
            output_layout_source=embedding,
            rms_norm=False,
            verbose=False,
        )


def test_no_bias_clear_uses_reserved_constant_lane():
    embedding = _embedding()
    output = _identity(embedding, "output")
    net = forward_compile(
        d=16,
        d_head=4,
        d_hidden=8,
        output_node=output,
        output_layout_source=embedding,
        rms_norm=False,
        bias=False,
        verbose=False,
    )

    ra = net.residual_assignment
    in_state = net.layers[0].attn.in_state
    const_one = next(
        n
        for n in ra.get_nodes(in_state)
        if isinstance(n, LiteralValue) and n.name == "rope_self_match_const_one"
    )
    const_col = ra.get_node_indices(in_state, const_one)[0]
    mlp = net.layers[-1].mlp
    assert mlp.linear1.output_matrix[const_col, 0].item() == 1.0
    assert mlp.linear2.output_matrix[0, const_col].item() == -1.0
    assert mlp.linear2.output_bias[const_col].item() == 0.0
    assert any(e.op_type == "clear_literal_seed" for e in net.placements.entries)


def test_cpsat_charges_direct_compute_and_cancel_in_same_layer():
    embedding = _embedding()
    output = _identity(embedding, "output")
    common = dict(
        output_node=output,
        d_head=2,
        d_hidden=16,
        max_layers=4,
        time_budget_s=5.0,
        held_source_id=embedding.node_id,
        held_target_id=output.node_id,
    )

    impossible, _ = solve_schedule(d=5, **common)
    assert impossible is None

    assignment, _ = solve_schedule(d=9, **common)
    assert assignment is not None
    assert assignment.node_to_routing[output.node_id] == "attn"
    assert (
        assignment.node_to_cancel_layer[embedding.node_id]
        == assignment.node_to_layer[output.node_id]
    )

    statically_routed, _ = solve_schedule(d=9, **common, flex_routing=False)
    assert statically_routed is not None
    assert statically_routed.node_to_routing[output.node_id] == "attn"

    hinted, _ = solve_schedule(
        d=9,
        **common,
        hint_layers={output.node_id: assignment.node_to_layer[output.node_id]},
        hint_routing={output.node_id: "attn"},
        hint_cancel={
            embedding.node_id: assignment.node_to_cancel_layer[embedding.node_id]
        },
        strict_hint=True,
    )
    assert hinted is not None


def test_held_live_and_snapshot_models_are_identical():
    embedding = _embedding()
    output = _identity(embedding, "output")
    cfg = dict(
        d=9,
        d_head=2,
        d_hidden=16,
        max_layers=4,
        held_source_id=embedding.node_id,
        held_target_id=output.node_id,
    )
    live = build_cpsat_model(output, **cfg)
    problem = snapshot_from_graph_model(build_graph_model(output))
    snap = build_model_from_snapshot(problem, **cfg)
    assert str(live.model.Proto()) == str(snap.model.Proto())


def test_held_contract_participates_in_schedule_cache_identity():
    embedding = _embedding()
    output = _identity(embedding, "output")
    cfg = dict(
        d=16,
        d_head=4,
        d_hidden=16,
        flex_routing=True,
        cancel_slack=2,
        policy=None,
    )
    generic = graph_fingerprint(output, **cfg)
    held = graph_fingerprint(
        output,
        **cfg,
        held_output_source=embedding,
        held_output_target=output,
    )
    assert held != generic
    with pytest.raises(ValueError, match="requires both endpoints"):
        graph_fingerprint(output, **cfg, held_output_source=embedding)


def test_cpsat_no_incumbent_fallback_keeps_held_contract(monkeypatch):
    from torchwright.compiler.forward import compile as compile_mod
    from torchwright.compiler.forward.cpsat_scheduler import SolveStats

    stats = SolveStats(
        status_name="UNKNOWN",
        objective_value=-1,
        best_objective_bound=0.0,
        wall_time_s=0.0,
        solver_log="",
        total_attn_heads=-1,
        total_mlp_bypass_slots=-1,
        is_optimal=False,
    )
    monkeypatch.setattr(
        compile_mod, "solve_schedule", lambda *args, **kwargs: (None, stats)
    )

    embedding = _embedding()
    output = _identity(embedding, "output")
    with pytest.warns(RuntimeWarning, match="falling back"):
        net = forward_compile(
            d=16,
            d_head=4,
            d_hidden=16,
            output_node=output,
            output_layout_source=embedding,
            rms_norm=False,
            optimize=1,
            verbose=False,
        )
    assert _banks(net, embedding, output)[0] == _banks(net, embedding, output)[1]


def test_held_schedule_cache_replay_keeps_bank(monkeypatch, tmp_path):
    monkeypatch.setenv("TW_SCHEDULE_CACHE_DIR", str(tmp_path))
    embedding = _embedding()
    output = _identity(embedding, "output")
    kwargs = dict(
        d=16,
        d_head=4,
        d_hidden=16,
        output_node=output,
        output_layout_source=embedding,
        rms_norm=False,
        optimize=1,
        require_solver=True,
        verbose=False,
    )

    first = forward_compile(**kwargs)
    assert first.cpsat_solve_stats.status_name != "CACHED"
    second = forward_compile(**kwargs)
    assert second.cpsat_solve_stats.status_name == "CACHED"
    assert _banks(second, embedding, output)[0] == _banks(second, embedding, output)[1]


def test_direct_held_handoff_shares_the_atomic_attention_batch():
    """Scheduler-level pin of the held handoff inside the directed atomic
    attention batch (atomic-attention-replay plan §5.3): the target's source
    columns are captured BEFORE ``hold(source)`` (they are exactly the bank
    plus the ordinary leaf's columns), the held bank is claimed exactly and
    in order by the target while an ordinary release in the same batch goes
    to the free pool, and the one coalesced cancel covers both.  The
    allocator-level facts (held columns are never ordinary-free; ordinary
    ``allocate`` cannot draw them; only the full ordered bank claim succeeds)
    are unit-pinned in ``test_residual_map.py``; the end-to-end tied parity
    tests remain the primary artifact check."""
    from torchwright.compiler.forward.cpsat_scheduler import ScheduleAssignment
    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
    from torchwright.compiler.forward.residual_map import (
        HeldOutputLayout,
        ResidualStreamMap,
    )
    from torchwright.compiler.forward.scheduler import DirectedLayerScheduler
    from torchwright.ops.inout_nodes import create_input

    torch.manual_seed(0)
    emb = _embedding()
    x = create_input("x", 2)
    w = Linear(x, torch.eye(2), name="W")
    out = Linear(
        Concatenate([emb, w]),
        torch.randn(6, 4),
        torch.zeros(4),
        name="out",
    )

    asg = ScheduleAssignment(
        node_to_layer={w.node_id: 0, out.node_id: 1},
        node_to_cancel_layer={
            emb.node_id: 1,  # gap-0: dies at its direct reader's layer
            w.node_id: 1,
            out.node_id: 2,
            x.node_id: 2,
        },
        node_to_routing={w.node_id: "attn", out.node_id: "attn"},
        n_layers=2,
        node_to_cancel_mech={w.node_id: "attn"},
    )

    d = 12
    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(d)
    const = create_input("const_stand_in", 1)
    for node in (const, emb, x):
        rmap.allocate(node)
    computed = {emb, x}
    bank = tuple(rmap.get_indices(emb))
    layout = HeldOutputLayout(source=emb, target=out, bank=bank)
    sched = DirectedLayerScheduler(
        graph, d, 3, None, assignment=asg, d_hidden=d, held_output_layout=layout
    )

    sched.set_current_layer(0)
    sched.schedule_layer(rmap, computed)
    assert w in computed
    w_cols = list(rmap.get_indices(w))

    sched.set_current_layer(1)
    attn_ops, _, _ = sched.schedule_layer(rmap, computed)

    # Source columns were captured before hold(source): the op reads exactly
    # the bank followed by the ordinary leaf's columns.
    out_ops = [
        op for op in attn_ops if op.op_type == "compute_linear" and op.node is out
    ]
    assert len(out_ops) == 1
    assert out_ops[0].source_cols == list(bank) + w_cols

    # One coalesced cancel covers the held source and the ordinary release.
    cancels = [op for op in attn_ops if op.op_type == "cancel"]
    assert len(cancels) == 1
    assert set(cancels[0].target_cols) == set(bank) | set(w_cols)

    # The target claims exactly the bank, in order; the ordinary release went
    # to the free pool (d - const(1) - x(2) - out(4) = 5 free), so the bank
    # was never reported as ordinary free.
    assert list(rmap.get_indices(out)) == list(bank)
    assert not rmap._held
    assert rmap.get_free_count() == d - 1 - 2 - 4
