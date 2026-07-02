"""Tests for graph optimization passes."""

import torch
import pytest

from torchwright.compiler.export import compile_headless
from torchwright.graph import FFN, Concatenate, InputNode, Linear
from torchwright.graph.optimize import fuse_consecutive_linears


def test_fuse_simple_chain():
    """Fuse L1 -> L2 into a single Linear."""
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(inp, torch.randn(4, 3), torch.randn(3), name="l1")
    l2 = Linear(l1, torch.randn(3, 2), torch.randn(2), name="l2")

    # Compute before fusion
    n_pos = 5
    x = torch.randn(n_pos, 4)
    out_before = l2.compute(n_pos, {"x": x})

    # Fuse
    fused = fuse_consecutive_linears({l2})
    assert fused == 1
    assert l2.output_matrix.shape == (4, 2)
    assert l2.inputs[0] is inp

    # Output should match
    out_after = l2.compute(n_pos, {"x": x})
    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_fuse_chain_of_three():
    """Fuse L1 -> L2 -> L3 in two passes."""
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(inp, torch.randn(4, 3), name="l1")
    l2 = Linear(l1, torch.randn(3, 2), name="l2")
    l3 = Linear(l2, torch.randn(2, 1), name="l3")

    n_pos = 5
    x = torch.randn(n_pos, 4)
    out_before = l3.compute(n_pos, {"x": x})

    # First pass fuses l1+l2, second pass fuses (l1+l2)+l3
    total = 0
    while True:
        fused = fuse_consecutive_linears({l3})
        if fused == 0:
            break
        total += fused

    assert total == 2
    assert l3.output_matrix.shape == (4, 1)
    assert l3.inputs[0] is inp

    out_after = l3.compute(n_pos, {"x": x})
    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_fuse_chain_ordering_regression():
    """Fusion count must be exactly 2 regardless of global_node_id offset.

    With offset=5, CPython's set iteration places L3 (ID=8, slot 8%8=0)
    before L2 (ID=7, slot 7%8=7), so without a topological sort the
    candidates loop appends (L2,L3) before (L1,L2).  Processing bottom-up
    leaves L3 depending on L1, which a second while-True pass fuses again,
    reporting total=3 instead of 2.

    The topological sort (by l1.node_id) makes the function correct for
    any starting offset.
    """
    import torchwright.graph.node as node_module

    for offset in [0, 5, 101, 997]:
        node_module.global_node_id = offset

        inp = InputNode("x", 4, value_range=(-100.0, 100.0))
        l1 = Linear(inp, torch.randn(4, 3), name="l1")
        l2 = Linear(l1, torch.randn(3, 2), name="l2")
        l3 = Linear(l2, torch.randn(2, 1), name="l3")

        n_pos = 5
        x = torch.randn(n_pos, 4)
        out_before = l3.compute(n_pos, {"x": x})

        total = 0
        while True:
            fused = fuse_consecutive_linears({l3})
            if fused == 0:
                break
            total += fused

        assert total == 2, f"offset={offset}: expected 2 fusions, got {total}"
        assert l3.output_matrix.shape == (4, 1)
        assert l3.inputs[0] is inp

        out_after = l3.compute(n_pos, {"x": x})
        assert torch.allclose(
            out_before, out_after, atol=1e-5
        ), f"offset={offset}: numeric mismatch after fusion"


def test_no_fuse_multiple_consumers():
    """Don't fuse when L1 has multiple consumers."""
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(inp, torch.randn(4, 3), name="l1")
    l2 = Linear(l1, torch.randn(3, 2), name="l2")
    l3 = Linear(l1, torch.randn(3, 2), name="l3")  # Another consumer of l1

    fused = fuse_consecutive_linears({l2, l3})
    assert fused == 0  # Can't fuse because l1 has two consumers


def test_no_fuse_concatenate_input():
    """Don't fuse when L2's input is a Concatenate (even if it wraps a Linear)."""
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(inp, torch.randn(4, 3), name="l1")
    concat = Concatenate([l1])  # Wrap l1 in a Concatenate
    l2 = Linear(concat, torch.randn(3, 2), name="l2")

    fused = fuse_consecutive_linears({l2})
    assert fused == 0  # Skip Concatenate inputs


def test_fuse_preserves_annotation():
    """Fused node keeps L2's annotation."""
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(inp, torch.randn(4, 3), name="l1")
    l1.annotation = "first"
    l2 = Linear(l1, torch.randn(3, 2), name="l2")
    l2.annotation = "second"

    fuse_consecutive_linears({l2})
    assert l2.annotation == "second"


def test_no_fuse_param_increase():
    """Don't fuse when fusion would increase params (bottleneck patterns).

    Example: L1 (4 -> 1) -> L2 (1 -> 100) uses 4*1 + 1 + 1*100 + 100 = 206 params.
    Fused (4 -> 100) would use 4*100 + 100 = 500 params — almost 2.5x more.

    This guards against "inverse bottleneck" patterns where the intermediate
    dimension is smaller than both input and output.
    """
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(inp, torch.randn(4, 1), torch.randn(1), name="bottleneck")
    l2 = Linear(l1, torch.randn(1, 100), torch.randn(100), name="expand")

    # Original params: 4*1 + 1 + 1*100 + 100 = 206
    # Fused params: 4*100 + 100 = 500
    fused = fuse_consecutive_linears({l2})
    assert fused == 0  # Should skip because it would increase params

    # The nodes should be unchanged
    assert l2.inputs[0] is l1
    assert l1.inputs[0] is inp


def test_fuse_param_decrease():
    """Fusion that reduces params should proceed.

    Example: L1 (100 -> 10) -> L2 (10 -> 3) uses 100*10 + 10 + 10*3 + 3 = 1043 params.
    Fused (100 -> 3) uses 100*3 + 3 = 303 params — ~70% reduction.
    """
    inp = InputNode("x", 100, value_range=(-100.0, 100.0))
    l1 = Linear(inp, torch.randn(100, 10), torch.randn(10), name="compress")
    l2 = Linear(l1, torch.randn(10, 3), torch.randn(3), name="final")

    n_pos = 5
    x = torch.randn(n_pos, 100)
    out_before = l2.compute(n_pos, {"x": x})

    # Original params: 100*10 + 10 + 10*3 + 3 = 1043
    # Fused params: 100*3 + 3 = 303
    fused = fuse_consecutive_linears({l2})
    assert fused == 1  # Should fuse

    # The fused node should produce same output (looser tolerance for 100-dim
    # matrix operations where floating-point error accumulates)
    out_after = l2.compute(n_pos, {"x": x})
    assert torch.allclose(out_before, out_after, atol=1e-4)


# ---------------------------------------------------------------------------
# FFN-aware folds (Phase 2c)
# ---------------------------------------------------------------------------


def _block(inp, d_input, n_lanes, d_output, seed=0):
    g = torch.Generator().manual_seed(seed)
    return FFN(
        inp,
        gate_proj=torch.randn(n_lanes, d_input, generator=g) * 0.3,
        gate_bias=torch.randn(n_lanes, generator=g) * 0.1,
        out_proj=torch.randn(n_lanes, d_output, generator=g) * 0.3,
        out_bias=torch.randn(d_output, generator=g) * 0.1,
    )


def test_fold_upstream_linear_into_block_gate():
    """A Linear whose sole consumer is an FFN folds into the gate projection."""
    inp = InputNode("x", 10, value_range=(-2.0, 2.0))
    u = Linear(inp, torch.randn(10, 6) * 0.2, torch.randn(6) * 0.1, name="u")
    b = _block(u, 6, 8, 4, seed=1)

    n_pos = 5
    x = torch.randn(n_pos, 10)
    before = b.compute(n_pos, {"x": x})

    fused = fuse_consecutive_linears({b})
    assert fused == 1
    assert isinstance(b, FFN)
    assert b.inputs[0] is inp  # gate now reads x directly
    assert b.d_input == 10
    assert b.gate_proj.shape == (8, 10)

    after = b.compute(n_pos, {"x": x})
    assert torch.allclose(before, after, atol=1e-5)


def test_fold_block_out_into_downstream_linear():
    """An FFN whose sole consumer is a Linear folds its out_proj into it."""
    inp = InputNode("x", 6, value_range=(-2.0, 2.0))
    b = _block(inp, 6, 8, 4, seed=2)
    l = Linear(b, torch.randn(4, 3) * 0.2, torch.randn(3) * 0.1, name="l")
    sink = Concatenate([l])  # keep l off the output boundary

    n_pos = 5
    x = torch.randn(n_pos, 6)
    before = sink.compute(n_pos, {"x": x})

    fused = fuse_consecutive_linears({sink})
    assert fused == 1
    assert b.d_output == 3
    assert b.out_proj.shape == (8, 3)
    assert sink.inputs[0] is b  # downstream Linear orphaned, consumer rewired

    after = sink.compute(n_pos, {"x": x})
    assert torch.allclose(before, after, atol=1e-5)


def test_block_out_fold_declined_at_output_boundary():
    """The FFN-into-Linear fold is declined when the Linear is a caller-held
    output node, preserving the caller's output identity."""
    inp = InputNode("x", 6, value_range=(-2.0, 2.0))
    b = _block(inp, 6, 8, 4, seed=3)
    l = Linear(b, torch.randn(4, 3) * 0.2, torch.randn(3) * 0.1, name="l")

    fused = fuse_consecutive_linears({l})
    assert fused == 0
    assert l.inputs[0] is b  # unchanged


def test_fold_linear_block_linear_both_sides():
    """Linear -> FFN -> Linear folds on both sides in one fixpoint call."""
    inp = InputNode("x", 10, value_range=(-2.0, 2.0))
    u = Linear(inp, torch.randn(10, 6) * 0.2, torch.randn(6) * 0.1, name="u")
    b = _block(u, 6, 8, 4, seed=4)
    l = Linear(b, torch.randn(4, 3) * 0.2, torch.randn(3) * 0.1, name="l")
    sink = Concatenate([l])

    n_pos = 5
    x = torch.randn(n_pos, 10)
    before = sink.compute(n_pos, {"x": x})

    fused = fuse_consecutive_linears({sink})
    assert fused == 2
    assert b.inputs[0] is inp
    assert b.d_input == 10
    assert b.d_output == 3
    assert sink.inputs[0] is b

    after = sink.compute(n_pos, {"x": x})
    assert torch.allclose(before, after, atol=1e-5)


def test_block_folds_preserve_compiled_output():
    """Folding a Linear -> FFN -> Linear graph must not change compiled output."""

    def build():
        g = torch.Generator().manual_seed(700)
        inp = InputNode("x", 12, value_range=(-1.0, 1.0))
        u = Linear(
            inp,
            torch.randn(12, 8, generator=g) * 0.2,
            torch.randn(8, generator=g) * 0.1,
            name="u",
        )
        b = _block(u, 8, 10, 6, seed=7)
        l = Linear(
            b,
            torch.randn(6, 5, generator=g) * 0.2,
            torch.randn(5, generator=g) * 0.1,
            name="l",
        )
        return inp, Concatenate([l])

    n_pos = 4
    g = torch.Generator().manual_seed(21)
    xt = torch.randn(n_pos, 12, generator=g)

    _, out_plain = build()
    c_plain = compile_headless(out_plain, d=64, d_head=8)

    _, out_fused = build()
    fuse_consecutive_linears({out_fused})
    c_fused = compile_headless(out_fused, d=64, d_head=8)

    assert torch.allclose(c_plain(xt), c_fused(xt), atol=1e-4)


def test_fusion_refreshes_stale_bounds():
    """Regression (RMSNorm-cert soundness crash): a fold rewrites a surviving
    node's weights/inputs in place, so its eagerly-cached ``_affine_bound`` /
    ``_structural_type`` — and every downstream node's — go stale.  A stale
    bound stays hidden until a downstream ``Assert`` tightens a structural type
    (GraphAnalyzer strip), at which point the affine and structural ranges
    disagree and the RMSNorm energy certification's soundness check fires.
    ``fuse_consecutive_linears`` must refresh every mutated/downstream bound, so
    the cached bound equals a fresh recompute for every node.
    """
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.graph.affine_rules import compute_affine_bound

    # Linear -> FFN -> Linear exercises Fold 1 (Linear into the gate) and
    # Fold 2 (FFN's out_proj into the downstream Linear, which also changes
    # the FFN's d_output 5 -> 4 — the clearest stale-bound signature).
    inp = InputNode("x", 6, value_range=(-1.0, 1.0))
    u = Linear(inp, torch.randn(6, 8) * 0.2, torch.randn(8) * 0.1, name="u")
    b = _block(u, 8, 10, 5, seed=3)
    l = Linear(b, torch.randn(5, 4) * 0.2, torch.randn(4) * 0.1, name="l")
    sink = Concatenate([l])

    n = fuse_consecutive_linears({sink})
    assert n == 2, f"expected both folds to fire, got {n}"
    assert b.d_output == 4  # Fold 2 rewrote the FFN's output width

    for node in get_ancestor_nodes({sink}):
        cached = node._affine_bound.to_scalar_range()
        fresh = compute_affine_bound(node).to_scalar_range()
        assert cached.lo == fresh.lo and cached.hi == fresh.hi, (
            f"stale bound on {type(node).__name__} id={node.node_id}: "
            f"cached={cached} fresh={fresh}"
        )
