"""Tests for scheduling hints — sequential_scope and related helpers.

Verifies that ``scheduling_predecessors`` on a Node gate readiness, and
that ``sequential_scope`` correctly wires deps so batched chains run in
the expected concurrency.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.graph import Concatenate, Linear
from torchwright.graph.scheduling_hints import (
    add_scheduling_dependency,
    sequential_scope,
)
from torchwright.ops.inout_nodes import create_input


def _linear(inp, d_out, name=""):
    return Linear(
        inp,
        torch.zeros(len(inp), d_out),
        torch.zeros(d_out),
        name=name,
    )


def test_scheduling_predecessor_gates_readiness():
    """A node with an unsatisfied scheduling predecessor is not ready."""
    x = create_input("x", 4)
    a = _linear(x, 2, name="a")
    b = _linear(x, 2, name="b")
    out = Concatenate([a, b])

    graph = GraphAnalyzer(out)

    # Without scheduling deps, a and b are both ready after {x}.
    available = {x}
    ready = graph.get_ready_nodes(available)
    assert a in ready
    assert b in ready

    # Add scheduling dep: b must wait for a.
    add_scheduling_dependency(b, a)

    ready = graph.get_ready_nodes(available)
    assert a in ready
    assert b not in ready  # gated on a

    # After a is computed, b becomes ready.
    ready = graph.get_ready_nodes({x, a})
    assert b in ready


def test_sequential_scope_batch_1():
    """batch_size=1 fully sequentializes iterations."""
    x = create_input("x", 4)

    terminals = sequential_scope(
        [lambda i=i: _linear(x, 2, name=f"row_{i}") for i in range(4)],
        batch_size=1,
    )

    assert len(terminals) == 4
    # row_0 has no scheduling deps; row_i (i>=1) gated on row_{i-1}.
    assert not terminals[0].scheduling_predecessors
    for i in range(1, 4):
        assert terminals[i - 1] in terminals[i].scheduling_predecessors


def test_sequential_scope_batch_2():
    """batch_size=2 allows 2 iterations in flight."""
    x = create_input("x", 4)

    terminals = sequential_scope(
        [lambda i=i: _linear(x, 2, name=f"row_{i}") for i in range(6)],
        batch_size=2,
    )

    # row_0, row_1 unrestricted. row_i (i>=2) waits for row_{i-2}.
    assert not terminals[0].scheduling_predecessors
    assert not terminals[1].scheduling_predecessors
    for i in range(2, 6):
        assert terminals[i - 2] in terminals[i].scheduling_predecessors


def test_sequential_scope_finds_deep_entry_nodes():
    """The entry-node detector walks past Concatenate and chain nodes."""
    x = create_input("x", 4)
    # Build a multi-node "iteration" per row: Linear → Linear → Linear.
    # The actual entry is the first Linear (reads from x).

    def build_row(i):
        a = _linear(x, 8, name=f"a_{i}")
        b = _linear(a, 8, name=f"b_{i}")
        return _linear(b, 2, name=f"c_{i}")

    terminals = sequential_scope(
        [lambda i=i: build_row(i) for i in range(3)],
        batch_size=1,
    )

    # Walk back from terminals[1] to find its "a_1" entry node, which
    # should carry a scheduling dep on terminals[0].
    graph = GraphAnalyzer(Concatenate(terminals))
    entries = [
        n for n in graph.get_all_nodes() if terminals[0] in n.scheduling_predecessors
    ]
    # Exactly one entry per iteration (the first Linear in the chain).
    assert len(entries) == 1
    (entry,) = entries
    # It should be the FIRST node of iteration 1 (not the terminal).
    assert entry is not terminals[1]
    # Specifically, the first Linear reading from x.
    assert entry.inputs[0] is x


def test_sequential_scope_does_not_deadlock_compile():
    """A sequentially-scoped graph compiles without infinite loops."""
    x = create_input("x", 4)

    rows = sequential_scope(
        [lambda i=i: _linear(x, 3, name=f"r_{i}") for i in range(4)],
        batch_size=2,
    )
    out = Concatenate(rows)

    net = forward_compile(
        d=64,
        d_head=16,
        output_node=out,
        verbose=False,
        device=None,
    )
    assert len(net.layers) > 0

    # Correctness: identity-ish weights mean output is zero; just verify
    # compute works end-to-end.
    inp = torch.randn(2, 4)
    result = net.compute(2, {"x": inp})
    # All rows are zero-weight Linears → output is all zeros
    assert result[rows[0]].shape[-1] == 3
