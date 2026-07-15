"""End-to-end MLP Add routing (docs/plan_additional_mlp_routing.md).

Step 3 gate: ``optimize=0`` compiles representative reused and fresh Adds
with zero Add heads under an MLP-preferring Add policy
(``add_in_attention="never"``), while the shipping default and
``LEGACY_POLICY`` retain the attention Add operations (the static default
keeps Adds on attention — see ``SchedulingPolicy.add_in_attention``).
Because every ``optimize=0`` emission replays through the directed
scheduler, these compiles exercise the directed MLP-Add placement, the
observed-versus-derived tripwire, and the reused-target canonicalization
on every run.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import ScheduleAssignment
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
    """One reusable and one fresh Add.

    ``add_reuse = x + lin(y)``: x's only consumer is the Add, so occurrence
    0 (a graph input) is the reuse target.  ``add_fresh = add_reuse + z``:
    both addends are retained by the output Concatenate, so neither is
    reusable.
    """
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


MLP_ADD_POLICY = SchedulingPolicy(add_in_attention="never")


def test_mlp_add_policy_routes_adds_to_mlp_with_zero_add_heads():
    out, x, add_reuse, add_fresh = _mixed_add_graph()
    net, plan = _capture_plan(output_node=out, policy=MLP_ADD_POLICY)

    attn_types, mlp_types = _op_types(plan)
    assert "add_into" not in attn_types and "compute_add" not in attn_types
    assert "add_into_bypass" in mlp_types
    assert "compute_add_bypass" in mlp_types

    reused_ops = [
        op
        for layer in plan.layers
        for op in layer.mlp_ops
        if op.op_type == "add_into_bypass"
    ]
    assert len(reused_ops) == 1
    assert reused_ops[0].reuse_input_index == 0  # x, the graph input

    # Canonical virtual cancel for the reassigned graph-input target: the
    # handoff boundary layer[add_reuse] + 1, with no cancel-mechanism entry
    # (the assignment contract for inputs).  No physical cancel is emitted:
    # ownership ended through reassign.
    assignment = plan.assignment
    x_id = plan.node_resolver()[reused_ops[0].node.node_id].inputs[0].node_id
    add_layer = assignment.node_to_layer[reused_ops[0].node.node_id]
    assert assignment.node_to_cancel_layer[x_id] == add_layer + 1
    assert x_id not in assignment.node_to_cancel_mech


@pytest.mark.parametrize(
    "policy",
    [None, LEGACY_POLICY],
    ids=["default", "legacy"],
)
def test_default_and_legacy_policies_retain_attention_add_ops(policy):
    """The shipping default keeps Adds on attention (the 2026-07-14 fallback
    decision: a static MLP default costs one layer per Add wedged between
    MLP-sublayer ops), as does LEGACY_POLICY."""
    out, *_ = _mixed_add_graph()
    kwargs = {} if policy is None else {"policy": policy}
    net, plan = _capture_plan(output_node=out, **kwargs)

    attn_types, mlp_types = _op_types(plan)
    assert "add_into" in attn_types
    assert "compute_add" in attn_types
    assert "add_into_bypass" not in mlp_types
    assert "compute_add_bypass" not in mlp_types


def test_mlp_routed_adds_match_compute():
    """Value parity through compile_headless: the compiled transformer with
    MLP-routed Adds matches the recursive oracle.  (Per-machine ReLU/Swish
    coverage of the writers lives in test_weight_writer.py; the machine
    here is whatever the default compile selects.)"""
    out, *_ = _mixed_add_graph()
    compiled = compile_headless(
        out, d=D, d_head=D_HEAD, d_hidden=64, policy=MLP_ADD_POLICY
    )
    torch.manual_seed(1)
    values = {
        "x": torch.rand(3, 4) * 2 - 1,
        "y": torch.rand(3, 4) * 2 - 1,
        "z": torch.rand(3, 4) * 2 - 1,
    }
    report = probe_compiled(compiled, out, values, n_pos=3, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_corrupted_observation_trips_the_derivation_tripwire():
    """A trace whose observed Add placement disagrees with the assignment-
    level derivation must raise the named invariant before an assignment is
    returned — for the fresh/reused form and for the occurrence, including
    add(x, x)."""
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    a = Linear(x, torch.randn(4, 3) * 0.2, name="a")
    b = Linear(x, torch.randn(4, 3) * 0.2, name="b")
    add = Add(a, b, name="add")

    layers = {a.node_id: 0, b.node_id: 0, add.node_id: 1}
    routing = {a.node_id: "mlp", b.node_id: "mlp", add.node_id: "mlp"}

    def complete(observed):
        return ScheduleAssignment.from_heuristic_trace(
            add,
            node_to_layer=layers,
            node_to_routing=routing,
            observed_cancel_layer={},
            observed_cancel_mech={},
            n_layers=2,
            observed_add_placement=observed,
        )

    # Both addends are single-consumer, so the derivation selects
    # occurrence 0.  The honest observation completes and canonicalizes.
    assignment = complete({add.node_id: (True, 0)})
    assert assignment.node_to_cancel_layer[a.node_id] == 2
    assert assignment.node_to_cancel_mech[a.node_id] == "attn"

    # Wrong occurrence.
    with pytest.raises(ValueError, match="disagrees with the assignment-level"):
        complete({add.node_id: (True, 1)})
    # Wrong form.
    with pytest.raises(ValueError, match="disagrees with the assignment-level"):
        complete({add.node_id: (False, None)})
    # Missing observation.
    with pytest.raises(ValueError, match="no observed placement"):
        complete({})

    # add(x, x): both occurrences name one node; only occurrence 0 is the
    # legal target.
    self_add = Add(a, a, name="self_add")
    self_layers = {a.node_id: 0, self_add.node_id: 1}
    self_routing = {a.node_id: "mlp", self_add.node_id: "mlp"}
    good = ScheduleAssignment.from_heuristic_trace(
        self_add,
        node_to_layer=self_layers,
        node_to_routing=self_routing,
        observed_cancel_layer={},
        observed_cancel_mech={},
        n_layers=2,
        observed_add_placement={self_add.node_id: (True, 0)},
    )
    assert good.node_to_cancel_layer[a.node_id] == 2
    with pytest.raises(ValueError, match="disagrees with the assignment-level"):
        ScheduleAssignment.from_heuristic_trace(
            self_add,
            node_to_layer=self_layers,
            node_to_routing=self_routing,
            observed_cancel_layer={},
            observed_cancel_mech={},
            n_layers=2,
            observed_add_placement={self_add.node_id: (True, 1)},
        )
