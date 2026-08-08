"""Held-bank embedding/output allocation and scheduling regressions."""

import pytest
import torch

from torchwright.compiler.forward.compile import (
    _build_heuristic_schedule_trace,
    forward_compile,
)
from torchwright.compiler.forward.cpsat_scheduler import (
    build_cpsat_model,
    build_graph_model,
    build_model_from_snapshot,
    solve_schedule,
)
from torchwright.compiler.forward.cpsat_snapshot import snapshot_from_graph_model
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import (
    HeldOutputLayout,
    ResidualStreamMap,
)
from torchwright.compiler.forward.scheduler import LayerScheduler
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.compiler.graph_identity import graph_fingerprint
from torchwright.compiler.realization import (
    ATTN_TRANSPORT,
    RealizationTable,
    usable_hidden_slots,
)
from torchwright.graph import Add, Concatenate, Embedding, Linear
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import LiteralValue


def _embedding(width=4):
    # "<unk>" appended last (a/b keep their token ids) with a zeros row:
    # the token exporters require an addressable unknown token, and the
    # structural tests here never read vocab-size-dependent surfaces.
    table = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, 3.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    assert width == 4
    return Embedding(["a", "b", "<unk>"], width, table=table, special_tokens=[])


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
    assert torch.equal(got.cpu(), embedding.table[:2])

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


def _add_output_graph():
    """The canonical tied Add shape: each addend's only consumer is the output.

    So the model's E_dead for both addends is the constant 1.
    """
    embedding = _embedding()
    left = _identity(embedding, "left")
    right = _identity(embedding, "right")
    return embedding, Add(left, right)


def test_add_target_held_model_is_feasible():
    """Regression: the held-target pin made the model hard-INFEASIBLE.

    The pin `is_free == 0` posted ON TOP of the is_free <=> OR(E_dead)
    biconditional made the model hard-INFEASIBLE for any Add target with
    a sole-consumer addend, the canonical
    `logits = transported_embedding + correction` shape.
    """
    embedding, output = _add_output_graph()
    asg, stats = solve_schedule(
        output,
        d=20,
        d_head=4,
        d_hidden=20,
        max_layers=8,
        time_budget_s=30.0,
        held_source_id=embedding.node_id,
        held_target_id=output.node_id,
    )
    assert asg is not None, (
        f"held Add-target model has no solution (status={stats.status_name}) "
        f"— the is_free pin contradicts the E_dead biconditional"
    )


def test_output_add_compiles_at_optimize_1_without_fallback():
    """The production-path companion: an optimize>=1 tied Add compile solves for real.

    It must come from a real solve, never the silent eager fallback
    (require_solver=True turns the fallback into a hard error).
    """
    embedding, output = _add_output_graph()
    net = forward_compile(
        d=20,
        d_head=4,
        d_hidden=20,
        output_node=output,
        output_layout_source=embedding,
        rms_norm=False,
        verbose=False,
        optimize=1,
        require_solver=True,
    )
    output_ops = [e.op_type for e in net.placements.entries if e.node is output]
    assert "compute_add" in output_ops
    assert "add_into" not in output_ops
    assert _banks(net, embedding, output)[0] == _banks(net, embedding, output)[1]


def _held_warm_start_hints(output, embedding, *, d, d_head, d_hidden, max_layers=12):
    """Run the REAL warm-start seam (`compile._build_heuristic_schedule_trace`).

    Runs it on a held-layout graph, mirroring forward_compile's setup:
    inputs allocated in node_id order, the bank captured immediately
    after the source allocates.
    """
    graph = GraphAnalyzer(output)
    nodes = graph.get_all_nodes()
    input_nodes = sorted(
        (n for n in nodes if graph.is_input_node(n)), key=lambda n: n.node_id
    )
    rmap = ResidualStreamMap(d)
    for n in input_nodes:
        rmap.allocate(n)
    layout = HeldOutputLayout(
        source=embedding, target=output, bank=tuple(rmap.get_indices(embedding))
    )
    table = RealizationTable.build(nodes).resolve_static(
        nodes, SchedulingPolicy(), usable_hidden_slots(d_hidden, bias=True)
    )
    trace = _build_heuristic_schedule_trace(
        graph=graph,
        d=d,
        d_head=d_head,
        n_heads=None,
        pos_encoding=None,
        d_hidden=d_hidden,
        residual_map=rmap,
        computed=set(input_nodes),
        clusters=None,
        admission_budget_fraction=0.4,
        policy=None,
        realization_table=table,
        held_output_layout=layout,
        output_node=output,
        max_layers=max_layers,
    )
    return trace, layout


def _ffn_chain_graph():
    """The held-across-layers fixture shape.

    The source's last (and only) reader is MLP-routed, so an
    MLP-mechanism hold of the source would fire at the reader's own
    layer, a timing the model cannot represent.
    """
    embedding = _embedding()
    a = _relu_identity(embedding, "a")
    b = _relu_identity(a, "b")
    output = _relu_identity(b, "output")
    return embedding, a, output


def test_warm_start_holds_source_via_attention_at_reader_layer_plus_one():
    """Regression (warm-start/model contract): inputs have no MLP cancel mechanism.

    There is no cancel_in_mlp var in the CP-SAT model; MLP readers bound
    the held source at cl >= layer + 1, so the heuristic must hold the
    tied bank via an attention cancel at the last MLP reader's layer + 1,
    never via an MLP cancel at the reader's own layer.
    """
    embedding, a, output = _ffn_chain_graph()
    trace, _ = _held_warm_start_hints(output, embedding, d=20, d_head=4, d_hidden=16)
    assert trace.n_layers > 0, "warm start deadlocked"
    assert trace.observed_cancel_mech[embedding.node_id] == "attn"
    assert (
        trace.observed_cancel_layer[embedding.node_id]
        == trace.node_to_layer[a.node_id] + 1
    )


def test_warm_start_held_hint_is_model_feasible_knob_off():
    """The hint the warm start emits must BE a schedule of the knob-off held model.

    The knob-off model is (_pin_cancels=False): hard-fix every hinted
    variable as an equality and assert the point is feasible. Before the
    fix the source's MLP-mechanism gap-0 hold hard-fixed cancel[emb]
    below the model's lower bound and this point was INFEASIBLE.
    """
    from ortools.sat.python import cp_model

    embedding, _a, output = _ffn_chain_graph()
    trace, _ = _held_warm_start_hints(output, embedding, d=20, d_head=4, d_hidden=16)
    assert trace.n_layers > 0, "warm start deadlocked"
    hl = trace.node_to_layer
    hc = trace.observed_cancel_layer
    hm = trace.observed_cancel_mech
    built = build_cpsat_model(
        output,
        d=20,
        d_head=4,
        d_hidden=16,
        max_layers=8,
        held_source_id=embedding.node_id,
        held_target_id=output.node_id,
        _pin_cancels=False,
    )
    # Hard-fix (mirrors test_cpsat_intralayer._hard_fix_and_solve): equality
    # per hinted variable, then a plain feasibility solve.
    model = built.model
    for nid, layer in hl.items():
        if nid in built.layer_var:
            model.Add(built.layer_var[nid] == layer)
    for nid, layer in hc.items():
        if nid in built.cancel_layer:
            model.Add(built.cancel_layer[nid] == layer)
        elif nid in built.input_cancel_layer:
            model.Add(built.input_cancel_layer[nid] == layer)
    for nid, mech in hm.items():
        if nid in built.cancel_in_mlp:
            model.Add(built.cancel_in_mlp[nid] == (1 if mech == "mlp" else 0))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status = solver.StatusName(solver.Solve(model))
    assert status in ("OPTIMAL", "FEASIBLE"), (
        f"warm-start hint is not a schedule of the held model ({status}) — "
        f"CP-SAT would silently drop the incumbent"
    )


def test_held_handoff_declines_same_layer_live_addend():
    """Guard parity: the held handoff must honor the add_into live-addend exclusion.

    It must exclude like the other same-layer cancel paths (the
    cancel-candidate and freshly-dead filters). This state is
    unreachable through schedule_layer today: a free Add consuming the
    source is an ancestor of the single-output target, so the two are
    never in one layer-start ready set, so the attention sublayer is
    driven directly with the co-resident state to pin the guard against
    future placement-rule changes.
    """
    embedding = _embedding()
    x = _identity(embedding, "x")
    free_add = Add(x, embedding)
    output = Linear(Concatenate([free_add, embedding]), torch.randn(8, 4), name="out")
    graph = GraphAnalyzer(output)
    nodes = graph.get_all_nodes()
    rmap = ResidualStreamMap(20)
    rmap.allocate(embedding)
    bank = tuple(rmap.get_indices(embedding))
    rmap.allocate(x)
    layout = HeldOutputLayout(source=embedding, target=output, bank=bank)
    table = RealizationTable.build(nodes).resolve_static(
        nodes,
        SchedulingPolicy(),
        usable_hidden_slots(16, bias=True),
        forced_classes={output.node_id: ATTN_TRANSPORT},
    )
    sched = LayerScheduler(
        graph,
        20,
        4,
        None,
        d_hidden=16,
        realization_table=table,
        held_output_layout=layout,
    )
    computed = {embedding, x}
    sched._skips = []
    sched._schedule_attn_sublayer({output}, [], [free_add], [], rmap, computed)
    # The free Add reused x's columns with the source as its LIVE addend; the
    # handoff must decline the same-layer hold (the target defers a layer).
    assert free_add in computed
    assert output not in computed
    assert rmap.is_allocated(embedding)
    assert not rmap._held


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
    common = {
        "output_node": output,
        "d_head": 2,
        "d_hidden": 16,
        "max_layers": 4,
        "time_budget_s": 5.0,
        "held_source_id": embedding.node_id,
        "held_target_id": output.node_id,
    }

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

    from torchwright.compiler.forward.cpsat_scheduler import DiagnosticHint

    hinted, _ = solve_schedule(
        d=9,
        **common,
        _diagnostic_hint=DiagnosticHint(
            layers={output.node_id: assignment.node_to_layer[output.node_id]},
            routing={output.node_id: "attn"},
            cancel={
                embedding.node_id: assignment.node_to_cancel_layer[embedding.node_id]
            },
        ),
        strict_hint=True,
    )
    assert hinted is not None


def test_held_live_and_snapshot_models_are_identical():
    """The snapshot alone carries the held contract.

    A captured-with-held problem round-trips through JSON and rebuilds
    the live proto with NO re-supplied kwargs. A held-less capture
    builds a different (relaxed) proto, the non-vacuity check.
    """
    from torchwright.compiler.forward.cpsat_snapshot import SchedulingProblem

    embedding = _embedding()
    output = _identity(embedding, "output")
    geometry = {"d": 9, "d_head": 2, "d_hidden": 16, "max_layers": 4}
    live = build_cpsat_model(
        output,
        **geometry,
        held_source_id=embedding.node_id,
        held_target_id=output.node_id,
    )
    problem = snapshot_from_graph_model(
        build_graph_model(output),
        held_source_id=embedding.node_id,
        held_target_id=output.node_id,
    )
    roundtripped = SchedulingProblem.loads(problem.dumps())
    snap = build_model_from_snapshot(roundtripped, **geometry)
    assert str(live.model.Proto()) == str(snap.model.Proto())

    # Idempotent re-supply of the stored pair is allowed.
    resupplied = build_model_from_snapshot(
        roundtripped,
        **geometry,
        held_source_id=embedding.node_id,
        held_target_id=output.node_id,
    )
    assert str(resupplied.model.Proto()) == str(live.model.Proto())

    # A held-less capture solves a strictly relaxed model.
    relaxed = build_model_from_snapshot(
        snapshot_from_graph_model(build_graph_model(output)), **geometry
    )
    assert str(relaxed.model.Proto()) != str(live.model.Proto())


def test_snapshot_held_contract_validation():
    """Loud misses on every malformed held contract.

    This covers unpaired endpoints, ids that name no captured node,
    kwargs conflicting with the stored contract, and unknown ids
    reaching the model builder.
    """
    from torchwright.compiler.forward.cpsat_snapshot import SchedulingProblem

    embedding = _embedding()
    output = _identity(embedding, "output")
    gm = build_graph_model(output)
    geometry = {"d": 9, "d_head": 2, "d_hidden": 16, "max_layers": 4}

    with pytest.raises(ValueError, match="supplied together"):
        snapshot_from_graph_model(gm, held_source_id=embedding.node_id)

    with pytest.raises(ValueError, match="not a captured node"):
        snapshot_from_graph_model(
            gm, held_source_id=10**9, held_target_id=output.node_id
        )

    # from_json enforces pairing on hand-edited/corrupt fixtures too.
    d = snapshot_from_graph_model(gm).to_json()
    d["held_source_id"] = output.node_id
    with pytest.raises(ValueError, match="supplied together"):
        SchedulingProblem.from_json(d)

    problem = snapshot_from_graph_model(
        gm, held_source_id=embedding.node_id, held_target_id=output.node_id
    )
    with pytest.raises(ValueError, match="conflict"):
        build_model_from_snapshot(
            problem,
            **geometry,
            held_source_id=output.node_id,
            held_target_id=output.node_id,
        )

    # The model builder names an unknown id instead of crashing on
    # Node.__eq__(None) deep inside a list-membership check.
    with pytest.raises(ValueError, match="does not name a node"):
        build_cpsat_model(
            output,
            **geometry,
            held_source_id=10**9,
            held_target_id=output.node_id,
        )


def test_canonicalized_snapshot_remaps_and_rebuilds_held_ids():
    """canonicalized() carries the held endpoints into the canonical id space.

    The canonical problem still rebuilds without re-supplied kwargs.
    """
    from torchwright.compiler.graph_identity import canonical_ids

    embedding = _embedding()
    output = _identity(embedding, "output")
    geometry = {"d": 9, "d_head": 2, "d_hidden": 16, "max_layers": 4}
    problem = snapshot_from_graph_model(
        build_graph_model(output),
        held_source_id=embedding.node_id,
        held_target_id=output.node_id,
    )
    canon = problem.canonicalized(output)
    mapping = canonical_ids(output)
    assert canon.held_source_id == mapping[embedding.node_id]
    assert canon.held_target_id == mapping[output.node_id]

    snap = build_model_from_snapshot(canon, **geometry)
    # Canonical relabeling renames proto variables, so byte-identity is a
    # live-id-space pin (above); here it suffices that the rebuilt model
    # actually carries the held contract — the direct-handoff attention pin
    # only exists when the endpoints survived the remap.
    assert "held_pinned" in str(snap.model.Proto())
    relaxed = build_model_from_snapshot(
        snapshot_from_graph_model(build_graph_model(output)).canonicalized(output),
        **geometry,
    )
    assert "held_pinned" not in str(relaxed.model.Proto())


def test_snapshot_identity_matches_production_cache_key_for_tied_graph(tmp_path):
    """with_identity derives the held endpoints from the snapshot's own stored ids.

    So a tied fixture's fingerprint equals the production schedule-cache
    key, and load() gates on exactly that key.
    """
    embedding = _embedding()
    output = _identity(embedding, "output")
    fp_cfg = {
        "d": 16,
        "d_head": 4,
        "d_hidden": 16,
        "flex_routing": True,
        "cancel_slack": 2,
        "policy": None,
    }
    production_fp = graph_fingerprint(
        output,
        **fp_cfg,
        held_output_source=embedding,
        held_output_target=output,
    )
    problem = (
        snapshot_from_graph_model(
            build_graph_model(output),
            held_source_id=embedding.node_id,
            held_target_id=output.node_id,
        )
        .canonicalized(output)
        .with_identity(
            output_node=output,
            d=16,
            d_head=4,
            d_hidden=16,
            flex_routing=True,
            cancel_slack=2,
            policy=None,
            critical_path_layers=1,
        )
    )
    assert problem.identity.fingerprint == production_fp

    path = problem.save(tmp_path / "tied.json")
    from torchwright.compiler.forward.cpsat_snapshot import SchedulingProblem

    loaded = SchedulingProblem.load(path, expected_fingerprint=production_fp)
    assert loaded.held_source_id == problem.held_source_id

    heldless_fp = graph_fingerprint(output, **fp_cfg)
    assert heldless_fp != production_fp
    with pytest.raises(ValueError, match="does not match the"):
        SchedulingProblem.load(path, expected_fingerprint=heldless_fp)


def test_held_contract_participates_in_schedule_cache_identity():
    embedding = _embedding()
    output = _identity(embedding, "output")
    cfg = {
        "d": 16,
        "d_head": 4,
        "d_hidden": 16,
        "flex_routing": True,
        "cancel_slack": 2,
        "policy": None,
    }
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
        compile_mod, "solve_schedule", lambda *_args, **_kwargs: (None, stats)
    )

    embedding = _embedding()
    output = _identity(embedding, "output")
    with pytest.warns(RuntimeWarning, match="retaining the"):
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
    kwargs = {
        "d": 16,
        "d_head": 4,
        "d_hidden": 16,
        "output_node": output,
        "output_layout_source": embedding,
        "rms_norm": False,
        "optimize": 1,
        "require_solver": True,
        "verbose": False,
    }

    first = forward_compile(**kwargs)
    assert first.cpsat_solve_stats.status_name != "CACHED"
    second = forward_compile(**kwargs)
    assert second.cpsat_solve_stats.status_name == "CACHED"
    assert _banks(second, embedding, output)[0] == _banks(second, embedding, output)[1]


def test_direct_held_handoff_shares_the_atomic_attention_batch():
    """Scheduler-level pin of the held handoff inside the atomic attention batch.

    This is the directed atomic-attention-replay plan §5.3: the target's source
    columns are captured BEFORE ``hold(source)`` (they are exactly the
    bank plus the ordinary leaf's columns), the held bank is claimed
    exactly and in order by the target while an ordinary release in the
    same batch goes to the free pool, and the one coalesced cancel
    covers both. The allocator-level facts (held columns are never
    ordinary-free; ordinary ``allocate`` cannot draw them; only the full
    ordered bank claim succeeds) are unit-pinned in
    ``test_residual_map.py``; the end-to-end tied parity tests remain
    the primary artifact check.
    """
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
    assert not rmap.has_held()
    assert rmap.get_free_count() == d - 1 - 2 - 4


def test_compile_headless_reproduces_the_tied_schedule(tmp_path):
    """compile_headless(output_layout_source=...) solves the same tied schedule.

    It solves the same (token.v6) schedule compile_to_onnx always
    builds, so the documented OnnxDebugSession discrimination recompile
    (CLAUDE.md, D1) reproduces a v6 artifact's structure instead of
    silently compiling an untied one.
    """
    from torchwright.compiler.export import compile_headless, compile_to_onnx

    embedding = _embedding()
    output = _identity(embedding, "output")
    compiled = compile_headless(output, d=16, d_head=4, output_layout_source=embedding)
    in_bank, out_bank = _banks(compiled._net, embedding, output)
    assert in_bank == out_bank  # ordered held-bank handoff, the v6 contract

    artifact = compile_to_onnx(
        output,
        embedding,
        str(tmp_path / "tied.onnx"),
        d=16,
        d_head=4,
        rms_norm=False,  # compile_headless has no rms_norm path
    )
    assert artifact.n_layers == len(compiled._net.layers)


def test_directed_scheduler_rejects_sibling_clusters():
    """The CP-SAT model has no admission constraint, so the replay runs ungated.

    An admission deferral after the atomic attention batch committed its
    releases would strand a batch member whose inputs were already
    cancelled. forward_compile rejects optimize>0 with
    admission_control=True; direct construction must be rejected too.
    """
    from torchwright.compiler.forward.cpsat_scheduler import ScheduleAssignment
    from torchwright.compiler.forward.scheduler import DirectedLayerScheduler
    from torchwright.compiler.forward.sibling_clusters import SiblingClusters

    emb = _embedding()
    out = _identity(emb, "output")
    asg = ScheduleAssignment(
        node_to_layer={out.node_id: 0},
        node_to_cancel_layer={},
        node_to_routing={out.node_id: "attn"},
        n_layers=1,
    )
    with pytest.raises(ValueError, match="admission control"):
        DirectedLayerScheduler(
            GraphAnalyzer(out),
            8,
            4,
            None,
            assignment=asg,
            clusters=SiblingClusters(),
        )


def test_unheld_bank_skip_names_the_held_output_bank():
    """A held target that cannot allocate is skipped for the "held output bank".

    That happens because the bank is not yet held; it is skipped for
    "held output bank", not for "residual columns", since the latter
    would report demand <= free, a self-contradictory line pointing at
    column exhaustion when the real blocker is that the tied source was
    never cancelled into the held state.
    """
    emb = _embedding()
    out = _identity(emb, "output")

    d = 8
    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(d)
    rmap.allocate(emb)
    computed = {emb}
    bank = tuple(rmap.get_indices(emb))
    layout = HeldOutputLayout(source=emb, target=out, bank=bank)
    all_nodes = graph.get_all_nodes()
    table = RealizationTable.build(all_nodes).resolve_static(
        all_nodes,
        SchedulingPolicy(),
        usable_hidden_slots(d, bias=True),
        forced_classes={out.node_id: ATTN_TRANSPORT},
    )
    # d_head == d leaves a single attention head: the dying-source escape
    # (cancel emb AND compute out in one layer: two heads) cannot fit, so
    # the target is passed over with the bank still unheld.
    sched = LayerScheduler(
        graph, d, d, None, realization_table=table, held_output_layout=layout
    )
    with pytest.raises(RuntimeError, match="No progress"):
        sched.schedule_layer(rmap, computed)
    skips = [s for s in sched._skips if s.node is out]
    assert skips
    assert skips[-1].resource == "held output bank"
    assert skips[-1].available == 0
    assert skips[-1].capacity == len(bank)
