"""Regression: re-compiling a long-lived graph after a node-id-counter reset.

`forward_compile` mints one graph node of its own — the RoPE self-match
constant-1 column (`compile.py`, `rope_self_match_const_one`) — and inserts it
into the residual-stream map.  `Node.__eq__`/`__hash__` key on `node_id`, so
that mint must not share an id with any graph node, or the two compare equal and
collide in `ResidualStreamMap._node_to_indices`.

In production the id counter is monotonic and never reset, so the mint is always
born past every graph node.  But `tests/conftest.py` resets the counter to 0
before every test (for deterministic column assignment).  A module-scoped graph
built in one test keeps its low ids; when a later test re-compiles it, the
freshly-reset counter mints `const_one` at id 0, colliding with the graph's
id-0 input node — its residual-map entry is overwritten, its column orphaned,
and invariant I1 fires (`column 0 neither free, allocated, nor reserved`).

`reserve_node_id_above` pushes the counter past the graph before minting, which
restores the production invariant locally.  This test drives the exact
collision condition (graph holds id 0, counter reset to 0, re-compile) and
asserts the compile both succeeds and stays numerically correct.
"""

import torch

import torchwright.graph.node as _node_module
from torchwright.compiler.forward.compile import forward_compile
from torchwright.graph import Linear
from torchwright.ops.inout_nodes import create_input

D = 256
D_HEAD = 16


def test_recompile_after_counter_reset_does_not_collide():
    # Build a tiny graph whose input node owns id 0 (the conftest autouse
    # fixture has just reset the counter, so the first node born here is id 0).
    x = create_input("x", 4)
    assert x.node_id == 0, "precondition: the input node must hold id 0"
    out = Linear(x, torch.randn(4, 3), torch.randn(3), name="lin")

    # Simulate the between-test reset that strands a long-lived (module-scoped)
    # graph's low ids: the counter goes back to 0 while `x` keeps id 0.  Without
    # the fix, the const_one mint inside forward_compile is also born at id 0 and
    # collides with `x`, firing I1.
    _node_module.global_node_id = 0

    net = forward_compile(d=D, d_head=D_HEAD, output_node=out, verbose=False)

    inputs = {"x": torch.randn(3, 4)}
    result = net.compute(3, inputs)
    actual = result[out].cpu()
    expected = out.compute(3, inputs)
    assert torch.allclose(
        actual, expected, atol=1e-4
    ), f"Max diff: {(actual - expected).abs().max().item():.6f}"


def test_reserve_node_id_above_only_advances():
    """The counter helper never moves the counter backward and is a no-op when
    the counter already leads — the property that makes it inert in production
    and in every single-compile test."""
    a = create_input("a", 1)  # some node with a concrete id
    _node_module.global_node_id = a.node_id + 100
    before = _node_module.global_node_id
    _node_module.reserve_node_id_above([a])  # a.node_id < before -> no-op
    assert _node_module.global_node_id == before

    _node_module.global_node_id = 0
    _node_module.reserve_node_id_above([a])  # counter behind a -> advance past it
    assert _node_module.global_node_id == a.node_id + 1
