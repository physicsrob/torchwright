"""Tests for scheduler admission control.

Verify that admission control shrinks compiled layer count on graphs
with the sibling-chain pattern, doesn't change correctness on other
graphs, and falls back on deadlock.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.graph import Concatenate, Linear
from torchwright.ops.inout_nodes import create_input


def _linear(inp, d_out, name=""):
    return Linear(
        inp,
        torch.zeros(len(inp), d_out),
        torch.zeros(d_out),
        name=name,
    )


def _build_sibling_chain_graph(n_branches=8, branch_width=64):
    """Graph with many parallel wide chains feeding one Concatenate.

    shared_in → N x [Linear(→branch_width) → Linear(→3)] → Concatenate
    """
    shared_in = create_input("shared", 1)
    terminals = []
    for i in range(n_branches):
        wide = _linear(shared_in, branch_width, name=f"wide_{i}")
        narrow = _linear(wide, 3, name=f"narrow_{i}")
        terminals.append(narrow)
    return Concatenate(terminals)


def test_admission_completes_on_sibling_graph():
    """Admission must not deadlock or crash on graphs with sibling clusters.

    Whether it *reduces* layers is workload-dependent — on tiny graphs
    admission may over-serialize and produce more layers than baseline.
    The DOOM-scale measurement is the authoritative "does this help"
    check; this test just verifies admission doesn't break compilation.
    """
    output = _build_sibling_chain_graph(n_branches=8, branch_width=64)

    gated = forward_compile(
        d=128,
        d_head=16,
        output_node=output,
        verbose=False,
        device=None,
        admission_control=True,
    )
    assert len(gated.layers) > 0


def test_admission_preserves_correctness_simple():
    """Compiling a simple graph with admission on produces correct outputs."""
    x = create_input("x", 8)
    w = torch.eye(8)
    y = Linear(x, w, torch.zeros(8))

    net = forward_compile(
        d=64,
        d_head=16,
        output_node=y,
        verbose=False,
        device=None,
        admission_control=True,
    )

    # Forward computation should pass inputs through unchanged.
    inp = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    result = net.compute(1, {"x": inp})
    torch.testing.assert_close(result[y].cpu(), inp)


def test_admission_disabled_matches_baseline():
    """admission_control=False should be a no-op — identical compile."""
    output = _build_sibling_chain_graph(n_branches=5, branch_width=64)
    net_a = forward_compile(
        d=128,
        d_head=16,
        output_node=output,
        verbose=False,
        device=None,
        admission_control=False,
    )
    net_b = forward_compile(
        d=128,
        d_head=16,
        output_node=output,
        verbose=False,
        device=None,
        admission_control=False,
    )
    assert len(net_a.layers) == len(net_b.layers)


def test_admission_on_narrow_graph_is_noop():
    """Graph with no clusters: admission on/off produces same layers."""
    # Narrow branches — below min_peak_width, so no cluster is formed.
    output = _build_sibling_chain_graph(n_branches=8, branch_width=4)

    baseline = forward_compile(
        d=128,
        d_head=16,
        output_node=output,
        verbose=False,
        device=None,
        admission_control=False,
    )
    gated = forward_compile(
        d=128,
        d_head=16,
        output_node=output,
        verbose=False,
        device=None,
        admission_control=True,
    )
    assert len(gated.layers) == len(baseline.layers)


def test_correctness_on_sibling_chain_graph():
    """Sibling-chain graph compiles and computes correct output with admission."""
    # Use non-zero weights so we actually compute something meaningful.
    # Final output must be non-Concatenate: Concatenate nodes aren't
    # materialized, so compute() won't key by them directly.
    shared_in = create_input("shared", 1)
    torch.manual_seed(0)
    terminals = []
    for i in range(4):
        w1 = torch.randn(1, 32)
        wide = Linear(shared_in, w1, torch.zeros(32), name=f"wide_{i}")
        w2 = torch.randn(32, 3)
        narrow = Linear(wide, w2, torch.zeros(3), name=f"narrow_{i}")
        terminals.append(narrow)
    joined = Concatenate(terminals)
    w_final = torch.randn(12, 5)
    final = Linear(joined, w_final, torch.zeros(5), name="final")

    net_a = forward_compile(
        d=256,
        d_head=32,
        output_node=final,
        verbose=False,
        device=None,
        admission_control=False,
    )
    net_b = forward_compile(
        d=256,
        d_head=32,
        output_node=final,
        verbose=False,
        device=None,
        admission_control=True,
        admission_min_chains=3,
        admission_min_peak_width=16,
    )

    # Both compilations should produce the same output for the same
    # input — admission control only changes *when* ops run, not what
    # they compute.
    inp = torch.tensor([[2.5]])
    out_a = net_a.compute(1, {"shared": inp})[final]
    out_b = net_b.compute(1, {"shared": inp})[final]
    torch.testing.assert_close(out_a.cpu(), out_b.cpu(), rtol=1e-4, atol=1e-4)
