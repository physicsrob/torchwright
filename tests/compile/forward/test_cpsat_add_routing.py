"""CP-SAT Add routing (docs/plan_additional_mlp_routing.md, Step 4).

The solver can choose either route family for a fitting Add, every returned
assignment replays through the directed scheduler, and a graph whose
attention-only Add schedule exceeds ``n_heads`` becomes feasible by routing
Adds to MLP.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    ScheduleAssignment,
    _solve_built,
    build_cpsat_model,
)
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)
from torchwright.debug.probe import probe_compiled
from torchwright.graph import Add, Concatenate, Linear
from torchwright.ops.inout_nodes import create_input

D = 64
D_HEAD = 8


def _mixed_add_graph():
    """One reusable and one fresh Add (mirrors test_mlp_add_routing)."""
    torch.manual_seed(0)
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    y = create_input("y", 4, value_range=(-1.0, 1.0))
    z = create_input("z", 4, value_range=(-1.0, 1.0))
    lin = Linear(y, torch.randn(4, 4) * 0.2, torch.randn(4) * 0.1, name="lin")
    add_reuse = Add(x, lin, name="add_reuse")
    add_fresh = Add(add_reuse, z, name="add_fresh")
    out = Concatenate([add_fresh, add_reuse, z])
    return out, x, add_reuse, add_fresh


def _capture_plan(**compile_kwargs):
    captured = {}

    def on_layer(_i, _layer):
        pass

    on_layer.on_replay_plan = lambda plan: captured.__setitem__("plan", plan)
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        device="cpu",
        verbose=False,
        on_layer_compiled=on_layer,
        **compile_kwargs,
    )
    return net, captured["plan"]


def _op_types(plan):
    attn = [op.op_type for layer in plan.layers for op in layer.attention_ops]
    mlp = [op.op_type for layer in plan.layers for op in layer.mlp_ops]
    return attn, mlp


def test_flex_solve_replays_and_matches_oracle():
    """Replay any solver-chosen route through the directed scheduler and match oracle.

    optimize=1 with default flex routing: whatever routes the solver
    picks, the assignment replays through the directed scheduler and the
    compiled values match the recursive oracle.
    """
    out, _x, _add_reuse, _add_fresh = _mixed_add_graph()
    compiled = compile_headless(out, d=D, d_head=D_HEAD, d_hidden=64, optimize=1)
    torch.manual_seed(1)
    values = {
        "x": torch.rand(3, 4) * 2 - 1,
        "y": torch.rand(3, 4) * 2 - 1,
        "z": torch.rand(3, 4) * 2 - 1,
    }
    report = probe_compiled(compiled, out, values, n_pos=3, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_legacy_policy_with_flex_off_pins_cpsat_adds_to_attention():
    """Keep every Add on the attention writers under the documented legacy config.

    policy=LEGACY_POLICY plus cpsat_flex_routing=False keeps every Add on
    the attention writers.
    """
    out, *_ = _mixed_add_graph()
    _net, plan = _capture_plan(
        output_node=out,
        optimize=1,
        policy=LEGACY_POLICY,
        cpsat_flex_routing=False,
        require_solver=True,
    )
    attn_types, mlp_types = _op_types(plan)
    assert "add_into" in attn_types or "compute_add" in attn_types
    assert "add_into_bypass" not in mlp_types
    assert "compute_add_bypass" not in mlp_types
    for add_name in ("add_reuse", "add_fresh"):
        nid = next(
            node_id
            for node_id, node in plan.nodes_by_id
            if getattr(node, "name", None) == add_name
        )
        assert plan.assignment.node_to_routing[nid] == "attn"


def _head_starved_graph():
    """A width-16 Add that no attention placement fits at n_heads=1.

    Both addends are Concatenates (physically non-reassignable, so reused
    placement is structurally impossible) and fresh placement wants
    ``2 * ceil(16/8) = 4`` heads against a 1-head budget — the Add is
    schedulable only on the MLP route (``2*16 = 32`` hidden slots).  The
    geometry deliberately avoids two PRE-EXISTING low-head model artifacts
    unrelated to Add routing: every keep-forever value is 8 wide (their
    cancel intervals pile at the virtual layer ``max_layers``, and each
    must fit the 8-unit head capacity alone), and only ONE 8-wide input
    dies (freeable-input cancels are attention-only and equality-pinned,
    so two inputs dying at the same pinned layer would exceed one head).
    The 16-wide Add itself dies through the MLP cancel mechanism.
    """
    torch.manual_seed(0)
    a = create_input("a", 8, value_range=(-1.0, 1.0))
    cat = Concatenate([a, a])
    add = Add(cat, cat, name="wide_add")
    return Linear(add, torch.randn(16, 8) * 0.2, torch.zeros(8), name="out")


def test_low_head_geometry_feasible_via_mlp_adds():
    """Make the CP-SAT compile feasible via MLP Add routing where attention-only fails.

    The headline win: attention-only Add routing cannot fit n_heads=1,
    and MLP Add routing makes the CP-SAT compile feasible (require_solver
    turns the silent heuristic fallback into a raise) and exact.
    """
    out = _head_starved_graph()
    _net, plan = _capture_plan(
        output_node=out,
        optimize=1,
        n_heads=1,
        d_hidden=64,
        require_solver=True,
    )
    attn_types, mlp_types = _op_types(plan)
    assert "add_into" not in attn_types
    assert "compute_add" not in attn_types
    assert "compute_add_bypass" in mlp_types

    compiled = compile_headless(
        out, d=D, d_head=D_HEAD, d_hidden=64, n_heads=1, optimize=1
    )
    torch.manual_seed(2)
    inputs = {"a": torch.rand(3, 8) * 2 - 1}
    report = probe_compiled(compiled, out, inputs, n_pos=3, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_low_head_geometry_feasible_via_mlp_adds_heuristic_path():
    """Route the Add to MLP and compile it via the heuristic + directed replay.

    Same geometry at optimize=0: an MLP-preferring Add policy routes the
    Add to MLP and the heuristic + directed replay compile it. (The
    shipping default keeps Adds on attention statically — at optimize=0
    this geometry needs the explicit knob; at optimize>0 the solver's flex
    routing rescues it regardless, per the test above.)
    """
    out = _head_starved_graph()
    compiled = compile_headless(
        out,
        d=D,
        d_head=D_HEAD,
        d_hidden=64,
        n_heads=1,
        policy=SchedulingPolicy(add_in_attention="never"),
    )
    torch.manual_seed(3)
    inputs = {"a": torch.rand(3, 8) * 2 - 1}
    report = probe_compiled(compiled, out, inputs, n_pos=3, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_reused_input_target_gets_virtual_handoff_in_assignment():
    """Record the canonical virtual cancel `layer[A] + 1` for a reused input target.

    A CP-SAT solve that reuses a graph-input target records the canonical
    virtual cancel `layer[A] + 1` for it, with no mechanism entry.
    """
    out, _x, _add_reuse, _add_fresh = _mixed_add_graph()
    _net, plan = _capture_plan(output_node=out, optimize=1, require_solver=True)
    assignment = plan.assignment
    reused_ops = [
        op
        for layer in plan.layers
        for op in layer.mlp_ops + layer.attention_ops
        if op.op_type in ("add_into_bypass", "add_into")
    ]
    if not reused_ops:
        pytest.skip("solver chose fresh placement for every Add on this run")
    for op in reused_ops:
        target = op.node.inputs[op.reuse_input_index]
        add_layer = assignment.node_to_layer[op.node.node_id]
        assert assignment.node_to_cancel_layer[target.node_id] == add_layer + 1
        if target.node_id not in assignment.node_to_layer:
            # Graph-input target: no mechanism entry (assignment contract).
            assert target.node_id not in assignment.node_to_cancel_mech


def test_extraction_tripwire_fires_on_corrupted_literal():
    """Trip the named invariant before a ScheduleAssignment is returned.

    A deliberately inconsistent model — the extraction reads a literal
    that no constraint ties to the layer assignment — must trip the named
    invariant before a ScheduleAssignment is returned.
    """
    from torchwright.compiler.lower import lower

    out, *_ = _mixed_add_graph()
    lowered = lower(out)
    built = build_cpsat_model(
        lowered.output_node, d=D, d_head=D_HEAD, d_hidden=64, max_layers=8
    )
    add_id, lit = next(iter(built.is_free.items()))
    # Replace the extraction's view of is_free with an unconstrained pin to
    # the WRONG value: the model still solves (the real literal keeps its
    # constraints), but the recomputed selector cannot match.
    wrong = built.model.NewBoolVar("corrupted_is_free")
    built.model.Add(wrong == lit.Not())
    built.is_free[add_id] = wrong
    with pytest.raises(AssertionError, match="extraction tripwire"):
        _solve_built(built, max_layers=8, time_budget_s=30.0)


def test_a2_trips_on_mlp_add_source_with_same_layer_attention_cancel():
    """Raise contract check A2 at replay rather than mis-execute a corrupted assignment.

    A corrupted assignment that gives an MLP-routed Add's source a
    same-layer attention cancel must raise contract check A2 at replay
    rather than mis-execute (the batch would release a value whose
    uncomputed consumer is outside the attention batch).
    """
    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
    from torchwright.compiler.forward.residual_map import ResidualStreamMap
    from torchwright.compiler.forward.scheduler import DirectedLayerScheduler

    torch.manual_seed(0)
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    u = Linear(x, torch.randn(4, 4) * 0.2, torch.zeros(4), name="u")
    v = Linear(x, torch.randn(4, 4) * 0.2, torch.zeros(4), name="v")
    add = Add(u, v, name="add")

    graph = GraphAnalyzer(add)
    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    computed = {x}

    assignment = ScheduleAssignment(
        node_to_layer={u.node_id: 0, v.node_id: 0, add.node_id: 1},
        node_to_cancel_layer={
            x.node_id: 1,
            # CORRUPTED: u is the MLP Add's source but is assigned an
            # attention-mechanism cancel in the Add's own layer.
            u.node_id: 1,
            v.node_id: 4,
            add.node_id: 4,
        },
        node_to_routing={u.node_id: "mlp", v.node_id: "mlp", add.node_id: "mlp"},
        n_layers=4,
        node_to_cancel_mech={u.node_id: "attn", v.node_id: "attn"},
    )
    sched = DirectedLayerScheduler(graph, D, D_HEAD, None, assignment=assignment)
    sched.set_current_layer(0)
    sched.schedule_layer(rmap, computed)
    sched.set_current_layer(1)
    with pytest.raises(AssertionError, match="A2"):
        sched.schedule_layer(rmap, computed)
