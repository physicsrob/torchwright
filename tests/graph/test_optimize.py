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


def test_fuse_through_single_leaf_concatenate():
    """A Linear leaf folds through a 1-leaf Concatenate; the concat is bypassed."""
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(inp, torch.randn(4, 3), torch.randn(3), name="l1")
    concat = Concatenate([l1])  # Wrap l1 in a Concatenate
    l2 = Linear(concat, torch.randn(3, 2), torch.randn(2), name="l2")

    n_pos = 5
    x = torch.randn(n_pos, 4)
    out_before = l2.compute(n_pos, {"x": x})

    fused = fuse_consecutive_linears({l2})
    assert fused == 1
    assert l2.inputs[0] is inp  # single surviving leaf bypasses the concat
    assert l2.output_matrix.shape == (4, 2)

    out_after = l2.compute(n_pos, {"x": x})
    assert torch.allclose(out_before, out_after, atol=1e-5)


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


# ---------------------------------------------------------------------------
# Linear-through-Concatenate folds
# ---------------------------------------------------------------------------


def test_fold_linear_leaves_through_concat():
    """Single-consumer Linear leaves fold into the downstream Linear's row
    blocks; the concat survives, rewired to the leaves' inputs."""
    in1 = InputNode("a", 4, value_range=(-2.0, 2.0))
    in2 = InputNode("b", 5, value_range=(-2.0, 2.0))
    l1 = Linear(in1, torch.randn(4, 3) * 0.3, torch.randn(3) * 0.1, name="l1")
    l2 = Linear(in2, torch.randn(5, 2) * 0.3, torch.randn(2) * 0.1, name="l2")
    c = Concatenate([l1, l2])
    top = Linear(c, torch.randn(5, 4) * 0.3, torch.randn(4) * 0.1, name="top")

    n_pos = 5
    vals = {"a": torch.randn(n_pos, 4), "b": torch.randn(n_pos, 5)}
    before = top.compute(n_pos, vals)

    fused = fuse_consecutive_linears({top})
    assert fused == 2
    assert c.inputs == [in1, in2]
    assert len(c) == 9
    assert top.output_matrix.shape == (9, 4)
    assert top.d_input == 9

    after = top.compute(n_pos, vals)
    assert torch.allclose(before, after, atol=1e-5)


def test_concat_fold_declined_multiconsumer_concat():
    """No leaf folds when the Concatenate itself has two consumers."""
    in1 = InputNode("a", 4, value_range=(-2.0, 2.0))
    l1 = Linear(in1, torch.randn(4, 3), name="l1")
    c = Concatenate([l1, in1])
    t1 = Linear(c, torch.randn(7, 2), name="t1")
    t2 = Linear(c, torch.randn(7, 2), name="t2")

    fused = fuse_consecutive_linears({t1, t2})
    assert fused == 0
    assert c.inputs == [l1, in1]


def test_concat_fold_skips_multiconsumer_leaf():
    """A leaf with a second consumer is kept; its sibling still folds."""
    in1 = InputNode("a", 4, value_range=(-2.0, 2.0))
    in2 = InputNode("b", 5, value_range=(-2.0, 2.0))
    shared = Linear(in1, torch.randn(4, 3) * 0.3, name="shared")
    other_consumer = Linear(shared, torch.randn(3, 2) * 0.3, name="other")
    foldable = Linear(in2, torch.randn(5, 2) * 0.3, name="foldable")
    c = Concatenate([shared, foldable])
    top = Linear(c, torch.randn(5, 4) * 0.3, name="top")

    n_pos = 5
    vals = {"a": torch.randn(n_pos, 4), "b": torch.randn(n_pos, 5)}
    before = top.compute(n_pos, vals)

    fused = fuse_consecutive_linears({top, other_consumer})
    assert fused == 1
    assert c.inputs == [shared, in2]

    after = top.compute(n_pos, vals)
    assert torch.allclose(before, after, atol=1e-5)


def test_concat_fold_declined_output_leaf():
    """A leaf that is a caller-held output node is never absorbed."""
    in1 = InputNode("a", 4, value_range=(-2.0, 2.0))
    leaf = Linear(in1, torch.randn(4, 3), name="leaf")
    c = Concatenate([leaf, in1])
    top = Linear(c, torch.randn(7, 2), name="top")

    fused = fuse_consecutive_linears({top, leaf})
    assert fused == 0
    assert c.inputs == [leaf, in1]


def test_literal_leaf_folds_into_bias():
    """A LiteralValue leaf's contribution moves into the bias; the leaf and
    its block rows are dropped."""
    from torchwright.graph import LiteralValue

    in1 = InputNode("a", 4, value_range=(-2.0, 2.0))
    lit = LiteralValue(torch.tensor([1.5, -2.0, 0.5]), name="konst")
    c = Concatenate([in1, lit])
    top = Linear(c, torch.randn(7, 2) * 0.3, torch.randn(2) * 0.1, name="top")

    n_pos = 5
    vals = {"a": torch.randn(n_pos, 4)}
    before = top.compute(n_pos, vals)

    fused = fuse_consecutive_linears({top})
    assert fused == 1
    assert top.inputs[0] is in1  # one leaf left -> concat bypassed
    assert top.output_matrix.shape == (4, 2)

    after = top.compute(n_pos, vals)
    assert torch.allclose(before, after, atol=1e-5)


def test_all_literal_concat_declined():
    """Literal folds never empty the concat (a 0-input Linear is not
    representable)."""
    from torchwright.graph import LiteralValue

    lit1 = LiteralValue(torch.tensor([1.0, 2.0]))
    lit2 = LiteralValue(torch.tensor([3.0]))
    c = Concatenate([lit1, lit2])
    top = Linear(c, torch.randn(3, 2), name="top")

    fused = fuse_consecutive_linears({top})
    assert fused == 0
    assert c.inputs == [lit1, lit2]


def test_concat_fold_param_guard():
    """A bottleneck leaf (1-wide output, wide fan-out) is declined."""
    in1 = InputNode("a", 4, value_range=(-2.0, 2.0))
    in2 = InputNode("b", 3, value_range=(-2.0, 2.0))
    bottleneck = Linear(in1, torch.randn(4, 1), name="bottleneck")
    c = Concatenate([bottleneck, in2])
    # old for the leaf: 4*1 + 1 + 1*100 = 105; new: 4*100 = 400 -> declined
    top = Linear(c, torch.randn(4, 100), name="top")

    fused = fuse_consecutive_linears({top})
    assert fused == 0
    assert c.inputs == [bottleneck, in2]


def test_accumulator_chain_collapses():
    """The ``sum_nodes`` fanout-limited accumulator shape —
    ``Linear(Concat(acc_Linear, ...))`` chains — collapses to one flat
    Linear over the leaves' inputs across passes (fold, splice, fold)."""
    ins = [InputNode(f"x{i}", 4, value_range=(-2.0, 2.0)) for i in range(3)]
    gates = [
        Linear(ins[i], torch.randn(4, 3) * 0.3, torch.randn(3) * 0.1, name=f"g{i}")
        for i in range(3)
    ]
    acc1 = Linear(
        Concatenate([gates[0], gates[1]]),
        torch.randn(6, 3) * 0.3,
        torch.randn(3) * 0.1,
        name="acc1",
    )
    top = Linear(
        Concatenate([acc1, gates[2]]),
        torch.randn(6, 2) * 0.3,
        torch.randn(2) * 0.1,
        name="top",
    )

    n_pos = 5
    vals = {f"x{i}": torch.randn(n_pos, 4) for i in range(3)}
    before = top.compute(n_pos, vals)

    # Pass 1 folds acc1 + g2 into top (acc1's concat becomes a leaf);
    # pass 2 splices that concat inline and folds g0 + g1.
    fused = fuse_consecutive_linears({top})
    assert fused == 5
    assert isinstance(top.inputs[0], Concatenate)
    assert top.inputs[0].inputs == [ins[0], ins[1], ins[2]]
    assert top.output_matrix.shape == (12, 2)

    after = top.compute(n_pos, vals)
    assert torch.allclose(before, after, atol=1e-5)


def test_concat_fold_refreshes_stale_bounds():
    """Concat folds mutate the survivor Linear and the concat in place; the
    cached bounds of both — and everything downstream — must equal a fresh
    recompute afterwards."""
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.graph.affine_rules import compute_affine_bound

    in1 = InputNode("a", 4, value_range=(-1.0, 1.0))
    in2 = InputNode("b", 5, value_range=(-1.0, 1.0))
    l1 = Linear(in1, torch.randn(4, 3) * 0.3, torch.randn(3) * 0.1, name="l1")
    l2 = Linear(in2, torch.randn(5, 2) * 0.3, torch.randn(2) * 0.1, name="l2")
    c = Concatenate([l1, l2])
    top = Linear(c, torch.randn(5, 4) * 0.3, torch.randn(4) * 0.1, name="top")
    sink = Concatenate([top])

    n = fuse_consecutive_linears({sink})
    assert n == 2

    for node in get_ancestor_nodes({sink}):
        cached = node._affine_bound.to_scalar_range()
        fresh = compute_affine_bound(node).to_scalar_range()
        assert cached.lo == fresh.lo and cached.hi == fresh.hi, (
            f"stale bound on {type(node).__name__} id={node.node_id}: "
            f"cached={cached} fresh={fresh}"
        )


def test_fusion_refreshes_stale_bounds():
    """Regression (RMSNorm-cert soundness crash): a fold rewrites a surviving
    node's weights/inputs in place, so its eagerly-cached ``_affine_bound`` /
    ``_structural_type`` — and every downstream node's — go stale.  A stale
    bound stays hidden until a claim tightens a structural type, at which
    point the affine and structural ranges
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


def test_concat_fold_merges_duplicate_leaves():
    """Two absorbed selector Linears over the SAME input leave the same node
    in two concat slots; the fold must merge their blocks (x reads through
    B1 + B2) instead of leaving duplicates — duplicated source columns broke
    the weight-writer scatter (the doom instance-index regression).

    Since the sibling fold landed, ``p`` and ``vp`` are contiguous same-input
    leaves and merge into one wide Linear *before* the absorb, so the duplicate
    the coalesce step exists to clean up is never created here.  Same final
    matrix, one fewer fold.  ``test_concat_fold_merges_hand_built_duplicate_leaves``
    (leaves are InputNodes, which the sibling fold never touches) still covers
    the coalesce itself.
    """
    occ = InputNode("occ", 4, value_range=(0.0, 300.0))
    p = Linear(occ, torch.tensor([[0.0], [1.0], [0.0], [0.0]]), name="p")
    vp = Linear(occ, torch.tensor([[0.0], [0.0], [1.0], [0.0]]), name="vp")
    idx = Linear(Concatenate([p, vp]), torch.tensor([[8.0], [1.0]]), name="idx")

    n_pos = 3
    vals = {"occ": torch.randn(n_pos, 4)}
    before = idx.compute(n_pos, vals)

    fused = fuse_consecutive_linears({idx})
    assert fused == 2  # sibling merge of p+vp, then one leaf fold
    assert idx.inputs[0] is occ  # merged to one leaf -> concat bypassed
    assert idx.output_matrix.shape == (4, 1)
    # merged block = 8·p_selector + vp_selector
    assert torch.equal(idx.output_matrix, torch.tensor([[0.0], [8.0], [1.0], [0.0]]))

    after = idx.compute(n_pos, vals)
    assert torch.allclose(before, after, atol=1e-5)


def test_concat_fold_merges_hand_built_duplicate_leaves():
    """A hand-built Concat([x, x]) under a Linear merges to a single leaf
    with summed blocks even though no leaf fold fires."""
    x = InputNode("x", 2, value_range=(-5.0, 5.0))
    l = Linear(
        Concatenate([x, x]),
        torch.tensor([[1.0], [2.0], [10.0], [20.0]]),
        name="l",
    )

    n_pos = 3
    vals = {"x": torch.randn(n_pos, 2)}
    before = l.compute(n_pos, vals)

    fused = fuse_consecutive_linears({l})
    assert fused == 1  # the merge alone
    assert l.inputs[0] is x
    assert torch.equal(l.output_matrix, torch.tensor([[11.0], [22.0]]))

    after = l.compute(n_pos, vals)
    assert torch.allclose(before, after, atol=1e-5)


def test_duplicate_selector_shape_compiles_correct():
    """End-to-end pin of the doom instance-index regression: two selectors
    over one input, concatenated into a combining Linear, must compile to
    the oracle value (it compiled to the second block only)."""
    from torchwright.ops.inout_nodes import create_input

    occ = create_input("occ", 4, value_range=(0.0, 300.0))
    p = Linear(occ, torch.tensor([[0.0], [1.0], [0.0], [0.0]]), name="p")
    vp = Linear(occ, torch.tensor([[0.0], [0.0], [1.0], [0.0]]), name="vp")
    idx = Linear(Concatenate([p, vp]), torch.tensor([[8.0], [1.0]]), name="idx")

    compiled = compile_headless(idx, d=64, d_head=8)
    x = torch.tensor([[1.0, 31.0, 0.0, 15.0], [1.0, 3.0, 2.0, 7.0]])
    got = compiled(x)
    expected = idx.compute(2, {"occ": x})
    assert torch.allclose(got, expected, atol=1e-4), (got, expected)


# ---------------------------------------------------------------------------
# Fold policy for checked values (docs/assert_metadata_plan.md): a fold
# that would erase a checked value is declined; the FFN->Linear fold
# migrates the orphan's checks instead (its value survives on the FFN).
# ---------------------------------------------------------------------------


def test_linear_fold_declined_on_checked_producer():
    from torchwright.graph.asserts import assert_in_range

    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(inp, torch.randn(4, 3), torch.randn(3), name="l1")
    assert_in_range(l1, -2000.0, 2000.0)
    l2 = Linear(l1, torch.randn(3, 2), torch.randn(2), name="l2")

    assert fuse_consecutive_linears({l2}) == 0
    assert l2.inputs[0] is l1  # l1's checked value stays materialized


def test_gate_fold_declined_on_checked_upstream_linear():
    from torchwright.graph.asserts import assert_in_range

    inp = InputNode("x", 6, value_range=(-1.0, 1.0))
    u = Linear(inp, torch.randn(6, 8) * 0.2, torch.randn(8) * 0.1, name="u")
    assert_in_range(u, -10.0, 10.0)
    b = _block(u, 8, 10, 5, seed=21)

    assert fuse_consecutive_linears({b}) == 0
    assert b.inputs[0] is u


def test_ffn_fold_declined_on_checked_ffn():
    from torchwright.graph.asserts import assert_in_range

    inp = InputNode("x", 6, value_range=(-1.0, 1.0))
    b = _block(inp, 6, 8, 4, seed=22)
    assert_in_range(b, -100.0, 100.0)  # b's pre-fold value is checked
    l = Linear(b, torch.randn(4, 3) * 0.2, torch.randn(3) * 0.1, name="l")
    sink = Concatenate([l])

    assert fuse_consecutive_linears({sink}) == 0
    assert l.inputs[0] is b


def test_ffn_fold_migrates_checks_from_orphaned_linear():
    """The one fold where the orphan's VALUE survives (on the FFN): its
    checks and claim migrate to the survivor instead of blocking."""
    from torchwright.graph.asserts import assert_in_range

    inp = InputNode("x", 6, value_range=(-1.0, 1.0))
    b = _block(inp, 6, 8, 4, seed=23)
    l = Linear(b, torch.randn(4, 3) * 0.2, torch.randn(3) * 0.1, name="l")
    assert_in_range(l, -1000.0, 1000.0)
    sink = Concatenate([l])

    n_pos = 5
    x = torch.randn(n_pos, 6)
    l_before = l.compute(n_pos, {"x": x})

    assert fuse_consecutive_linears({sink}) == 1
    assert len(b.checks) == 1  # migrated with the value
    assert b.claimed_type is not None
    assert torch.allclose(b.compute(n_pos, {"x": x}), l_before, atol=1e-5)


def test_concat_fold_declined_on_checked_concat():
    """The composite two-input asserts attach to a Concatenate; every
    concat fold changes its value or width, so a checked concat stays
    whole (splices included)."""
    from torchwright.graph.asserts import assert_unique_values

    a = InputNode("a", 2, value_range=(-2.0, 2.0))
    leaf = Linear(a, torch.randn(2, 3) * 0.3, torch.randn(3) * 0.1, name="leaf")
    c = Concatenate([leaf, a])
    assert_unique_values(c, margin=0.1)
    top = Linear(c, torch.randn(5, 4) * 0.3, torch.randn(4) * 0.1, name="top")

    assert fuse_consecutive_linears({top}) == 0
    assert top.inputs[0] is c
    assert c.inputs == [leaf, a]


def test_concat_fold_skips_checked_leaf_only():
    """A checked leaf is skipped; unchecked sibling leaves still fold."""
    from torchwright.graph.asserts import assert_in_range

    a = InputNode("a", 4, value_range=(-2.0, 2.0))
    b = InputNode("b", 5, value_range=(-2.0, 2.0))
    leaf1 = Linear(a, torch.randn(4, 3) * 0.3, torch.randn(3) * 0.1, name="leaf1")
    leaf2 = Linear(b, torch.randn(5, 2) * 0.3, torch.randn(2) * 0.1, name="leaf2")
    assert_in_range(leaf1, -10.0, 10.0)
    c = Concatenate([leaf1, leaf2])
    top = Linear(c, torch.randn(5, 4) * 0.3, torch.randn(4) * 0.1, name="top")

    n_pos = 5
    vals = {"a": torch.randn(n_pos, 4), "b": torch.randn(n_pos, 5)}
    before = top.compute(n_pos, vals)

    assert fuse_consecutive_linears({top}) == 1  # only leaf2 absorbed
    assert leaf1 in c.inputs
    assert leaf2 not in c.inputs
    assert torch.allclose(top.compute(n_pos, vals), before, atol=1e-5)
