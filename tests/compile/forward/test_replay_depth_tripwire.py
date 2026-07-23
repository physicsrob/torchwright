"""Replay-depth tripwire (optimality gap #5) — Unit 1.

``forward_compile`` replays a CP-SAT ``ScheduleAssignment`` through
``DirectedLayerScheduler``.  A faithful replay computes every schedulable
node at exactly ``assignment.node_to_layer[n]`` and therefore emits exactly
``assignment.n_layers`` transformer layers.  The tripwire
(``compile._check_replay_depth``) fires when the emitted depth or a node's
replayed birth layer diverges from the assignment — the only guard, before
this check, was the global ``max_layers`` convergence raise, so a
valid-but-deeper replay used to ship silently.

Two firing modes are covered:

* the emitted-depth mode (the "valid-but-deeper transformer" of gap #5),
  driven end to end through the real ``forward_compile`` replay by feeding
  it an assignment whose output node is placed one layer past the claimed
  depth; and
* the placement-divergence mode (a future executor that slips a node to a
  layer other than its assigned one), unit-tested against the helper
  directly because the current strict ready filter cannot slip a node — a
  mis-set *cancel* layer starves the replay into the existing "No progress"
  raise rather than producing a clean one-layer overrun.

Silence on a faithful replay is asserted here and, more broadly, by every
existing ``optimize>0`` test staying green.
"""

import dataclasses

import pytest
import torch

from torchwright.compiler.forward import compile as compile_mod
from torchwright.compiler.forward.compile import _check_replay_depth, forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    ScheduleAssignment,
    solve_schedule,
)
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.graph import Concatenate, Linear
from torchwright.ops.inout_nodes import create_input


def _build_width_graph():
    """A width-tight fan-out/fan-in graph (6 Linears -> pairs -> concat -> out).

    Small, deterministic, and CPU-compilable; the same geometry the
    intra-layer tests use to exercise the directed replay.
    """
    torch.manual_seed(1)
    x = create_input("x", 4)
    mids = []
    for i in range(6):
        li = Linear(x, torch.randn(4, 12), torch.zeros(12), name=f"L{i}")
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Ma{i}"))
        mids.append(Linear(li, torch.randn(12, 2), torch.zeros(2), name=f"Mb{i}"))
    out = Linear(Concatenate(mids), torch.randn(24, 4), torch.zeros(4), name="out")
    return out


def test_tripwire_silent_on_faithful_replay():
    """A normal optimize=1 compile replays the assignment faithfully, so the
    tripwire never fires and the compile produces the reference output.
    """
    out = _build_width_graph()
    net = forward_compile(
        d=80, d_head=80, output_node=out, device="cpu", verbose=False, optimize=1
    )
    inp = torch.randn(3, 4)
    ref = out.compute(3, {"x": inp})
    got = net.compute(3, {"x": inp})[out].cpu()
    assert torch.allclose(got, ref, atol=1e-3)


def test_tripwire_fires_on_deeper_replay(monkeypatch):
    """Gap #5: a schedule the machine replays one layer deeper than the model
    claimed.  We intercept the solve and push the output node one layer past
    its assigned layer (keeping ``n_layers``); the faithful replay then emits
    ``n_layers + 1`` layers and the tripwire must catch it.
    """
    out = _build_width_graph()
    real_solve = solve_schedule

    def deeper(*args, **kwargs):
        asg, stats = real_solve(*args, **kwargs)
        n2l = dict(asg.node_to_layer)
        n2cl = dict(asg.node_to_cancel_layer)
        # solve_schedule's first arg is the (lowered) output node — the graph
        # sink.  Push it one layer deeper (all its inputs stay put, so the
        # replay still satisfies its dependencies) and keep its cancel parked
        # above the unchanged claimed depth.  Moving the sink, not a tied
        # interior node, is what keeps the replay convergent yet one deeper.
        out_id = args[0].node_id
        n2l[out_id] += 1
        if out_id in n2cl:
            n2cl[out_id] = max(n2cl[out_id], asg.n_layers + 1)
        # The mutation must stay replay-SOUND (its only defect the extra
        # emitted layer): a value the moved sink reads whose cancel now sits
        # at or before the sink's new layer would be released by the atomic
        # attention batch while the sink still needs it, and the batch's A2
        # whole-batch last-reader validation correctly rejects that as an
        # unsound assignment before the depth tripwire can fire.  Push such
        # cancels past the new horizon (they simply never fire).
        for leaf in flatten_concat_nodes(list(args[0].inputs)):
            cl = n2cl.get(leaf.node_id)
            if cl is not None and cl <= n2l[out_id]:
                n2cl[leaf.node_id] = n2l[out_id] + 1
        mutated = dataclasses.replace(asg, node_to_layer=n2l, node_to_cancel_layer=n2cl)
        return mutated, stats

    monkeypatch.setattr(compile_mod, "solve_schedule", deeper)
    monkeypatch.setattr(
        compile_mod,
        "choose_dominating_assignment",
        lambda costs, incumbent, candidate: candidate,
    )

    with pytest.raises(RuntimeError, match="Replay-depth tripwire"):
        forward_compile(
            d=80, d_head=80, output_node=out, device="cpu", verbose=False, optimize=1
        )


def test_tripwire_helper_placement_divergence():
    """The placement-divergence branch: a node replayed at a layer other than
    its assigned one.  Tested against the helper directly — the current
    executor cannot slip a node (the strict ready filter starves instead), so
    this guards a future relaxed executor.
    """
    assignment = ScheduleAssignment(
        node_to_layer={1: 0, 2: 1, 3: 2},
        node_to_cancel_layer={1: 1, 2: 2, 3: 3},
        node_to_routing={1: "mlp", 2: "mlp", 3: "mlp"},
        n_layers=3,
    )
    # Node 2 was assigned layer 1 but the replay computed it at layer 2.
    replay_birth = {1: 0, 2: 2, 3: 2}
    with pytest.raises(RuntimeError, match=r"node 2 was assigned to layer 1.*layer 2"):
        _check_replay_depth(assignment, replay_birth, emitted_layers=3)


def test_tripwire_helper_silent_on_match():
    """The helper is silent when replay birth layers and emitted depth all
    agree with the assignment.
    """
    assignment = ScheduleAssignment(
        node_to_layer={1: 0, 2: 1, 3: 2},
        node_to_cancel_layer={1: 1, 2: 2, 3: 3},
        node_to_routing={1: "mlp", 2: "mlp", 3: "mlp"},
        n_layers=3,
    )
    replay_birth = {1: 0, 2: 1, 3: 2}
    _check_replay_depth(assignment, replay_birth, emitted_layers=3)  # no raise
