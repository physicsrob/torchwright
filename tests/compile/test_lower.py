"""Tests for the lowering boundary (:mod:`torchwright.compiler.lower`).

``lower()`` certifies a graph for scheduling — closed vocabulary (no raw
``ReLU`` / hand-built ``Linear -> ReLU -> Linear`` chain — the check that
used to live in the deleted ``graph/blockify.py``) — and returns a
compiler-private *copy* with fresh derived caches computed at clone time
(semantic overrides re-applied) and wrappers stripped.  The source graph
is never touched.  ``forward_compile`` runs it as step 0, so an
uncertified graph cannot reach the scheduler.
"""

import pytest
import torch

from torchwright.compiler.lower import LoweredGraph, LoweringError, lower
from torchwright.graph import FFN, Linear, Node, ReLU
from torchwright.graph.affine_rules import compute_affine_bound
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.misc import Concatenate
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear


def _block(x, d_input, n_lanes, d_output, seed=0, name=""):
    g = torch.Generator().manual_seed(seed)
    return linear_relu_linear(
        x,
        torch.randn(n_lanes, d_input, generator=g),
        torch.randn(n_lanes, generator=g),
        torch.randn(n_lanes, d_output, generator=g),
        torch.randn(d_output, generator=g),
        name=name,
    )


# ---------------------------------------------------------------------------
# Vocabulary certification (ports the deleted test_blockify.py cases)
# ---------------------------------------------------------------------------


def test_linear_relu_linear_builds_block_natively():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    node = _block(x, 4, 6, 3, name="c")
    assert isinstance(node, FFN)


def test_lower_passes_on_native_block_graph():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    blk = _block(x, 4, 6, 3, name="c")
    downstream = Linear(blk, torch.randn(3, 2), torch.randn(2))
    lowered = lower(downstream)
    assert isinstance(lowered, LoweredGraph)
    # The pipeline gets a compiler-private copy, never the source.
    assert lowered.output_node is not downstream
    assert lowered.output_node is lowered.copy_of(downstream)
    assert lowered.source_output_node is downstream


def test_lower_passes_on_stacked_blocks():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    b1 = _block(x, 4, 6, 4, seed=1)
    b2 = _block(b1, 4, 5, 3, seed=2)
    lowered = lower(b2)
    assert lowered.output_node is lowered.copy_of(b2)
    assert lowered.output_node is not b2


def test_lower_raises_on_raw_chain():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    l1 = Linear(x, torch.randn(4, 6), torch.randn(6))
    relu = ReLU(l1)
    l2 = Linear(relu, torch.randn(6, 3), torch.randn(3))
    with pytest.raises(LoweringError, match="chain"):
        lower(l2)


def test_lower_detects_raw_chain_through_internal_wrapper():
    # A wrapper on a chain-internal value does not hide the chain from the
    # detector (Assert/DebugWatch-transparent mining).
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    l1 = Linear(x, torch.randn(4, 6), torch.randn(6))
    relu = ReLU(l1)
    guarded = assert_in_range(relu, 0.0, 1000.0)
    l2 = Linear(guarded, torch.randn(6, 3), torch.randn(3))
    with pytest.raises(LoweringError, match="chain"):
        lower(l2)


def test_lower_raises_on_lone_relu():
    # A ReLU with no chain shape is still outside the vocabulary — the
    # scheduler has no write path for it.
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    relu = ReLU(x)
    with pytest.raises(LoweringError, match="lone ReLU|no chain shape"):
        lower(relu)


def test_lower_raises_on_unknown_node_type():
    # A type with no affine rule cannot even be constructed (Node.__init__
    # computes bounds eagerly and the dispatch raises), so simulate a future
    # node type that *has* graph-layer support but no scheduler write path
    # by swapping the class after construction.  The boundary must reject
    # anything outside the vocabulary regardless of how it got there.
    class Mystery(Node):
        def compute(self, n_pos, input_values):
            return self.inputs[0].compute(n_pos, input_values)

    x = create_input("x", 4, value_range=(-1.0, 1.0))
    stray = Linear(x, torch.randn(4, 3), torch.randn(3))
    stray.__class__ = Mystery
    with pytest.raises(LoweringError, match="Mystery"):
        lower(stray)


def test_lower_rejects_non_node():
    with pytest.raises(TypeError, match="output Node"):
        lower("not a node")


# ---------------------------------------------------------------------------
# Derived-cache freshness
# ---------------------------------------------------------------------------


def test_lower_copy_has_fresh_bounds_source_keeps_stale():
    """Boundary twin of test_fusion_refreshes_stale_bounds: an in-place
    weight mutation (here a hand edit) leaves the eagerly-cached bounds
    describing the pre-mutation graph.  The *copy* lower() builds must
    have cached == fresh for every reachable node (its caches are
    recomputed at clone time), while the source — compilation is a pure
    function — keeps whatever caches it had, stale or not."""
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    l1 = Linear(x, torch.randn(4, 6) * 0.2, torch.randn(6) * 0.1)
    blk = _block(l1, 6, 8, 5, seed=3)
    sink = Concatenate([blk])

    stale = l1._affine_bound.to_scalar_range()
    l1.output_matrix = l1.output_matrix * 3.0  # stale-making mutation

    lowered = lower(sink)

    from torchwright.compiler.utils import get_ancestor_nodes

    for node in get_ancestor_nodes({lowered.output_node}):
        cached = node._affine_bound.to_scalar_range()
        fresh = compute_affine_bound(node).to_scalar_range()
        assert cached.lo == fresh.lo and cached.hi == fresh.hi, (
            f"stale bound on copy {type(node).__name__} id={node.node_id}: "
            f"cached={cached} fresh={fresh}"
        )
    copy_range = lowered.copy_of(l1)._affine_bound.to_scalar_range()
    assert (
        copy_range.lo != stale.lo or copy_range.hi != stale.hi
    ), "mutation should have widened the copy's l1 bound"
    # Source untouched: still carrying its (now stale) cached bound.
    src_range = l1._affine_bound.to_scalar_range()
    assert (src_range.lo, src_range.hi) == (stale.lo, stale.hi)


def test_lower_preserves_semantic_overrides_on_copy():
    """Ops install semantic affine overrides (tighter than pure propagation)
    via _apply_semantic_override; the copy's clone-time recompute must
    re-apply them, not wipe them — a wipe would silently loosen every
    downstream bound.

    ``compare()`` installs its override on the Assert *wrapper* it
    returns.  The wrapper is stripped from the copy (as it always was
    from the compiled graph), so the override's compiled effect lives in
    the bounds of consumers built on the wrapper — the clone computes
    those through the cloned wrapper before the strip rewires it, so
    they must match the source's bit-identically."""
    from torchwright.ops.relu.arithmetic_ops import compare

    x = create_input("x", 1, value_range=(-10.0, 10.0))
    cmp = compare(x, 0.0, true_level=1.0, false_level=-1.0)
    consumer = Linear(cmp, torch.full((1, 1), 2.0))

    from torchwright.compiler.utils import get_ancestor_nodes

    overridden = [
        n
        for n in get_ancestor_nodes({consumer})
        if n._semantic_affine_override is not None
    ]
    assert overridden, "compare() should install a semantic override"
    r_src = consumer._affine_bound.to_scalar_range()
    assert (r_src.lo, r_src.hi) == (-2.0, 2.0)  # 2 * override range [-1, 1]

    lowered = lower(consumer)
    consumer_copy = lowered.copy_of(consumer)
    r_copy = consumer_copy._affine_bound.to_scalar_range()
    assert (r_copy.lo, r_copy.hi) == (r_src.lo, r_src.hi), (
        f"override-derived consumer bound lost on the copy: "
        f"source={r_src} copy={r_copy}"
    )
    # Overrides carried on non-wrapper nodes survive on their clones
    # (covered exhaustively in test_graph_clone); wrapper-carried
    # overrides vanish with the stripped wrapper, whose map entry
    # resolves to the wrapped node's clone.
    for n in overridden:
        from torchwright.graph.misc import Assert, DebugWatch

        copy = lowered.copy_of(n)
        if isinstance(n, (Assert, DebugWatch)):
            assert not isinstance(copy, (Assert, DebugWatch))
        else:
            assert copy._semantic_affine_override is not None


def test_lower_twice_is_bit_identical_and_source_untouched():
    """D6 for L1: lowering the same source twice yields per-node
    value_types that are bit-identical across the two copies, and the
    source — node set, wrappers, bounds — is untouched by both."""
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.graph.asserts import assert_in_range, collect_asserts

    x = create_input("x", 4, value_range=(-2.0, 2.0))
    blk = _block(x, 4, 6, 3, seed=5)
    out = Linear(assert_in_range(blk, -100.0, 100.0), torch.randn(3, 2))

    src_nodes_before = {n.node_id for n in get_ancestor_nodes({out})}
    src_ranges_before = {
        n.node_id: n.value_type.value_range for n in get_ancestor_nodes({out})
    }
    n_asserts_before = len(collect_asserts(out))

    lowered1 = lower(out)
    lowered2 = lower(out)

    for src in lowered1.node_map:
        c1, c2 = lowered1.copy_of(src), lowered2.copy_of(src)
        r1, r2 = c1.value_type.value_range, c2.value_type.value_range
        assert (r1.lo, r1.hi) == (r2.lo, r2.hi), (
            f"value_type differs across two lowers of the same source "
            f"({type(src).__name__} id={src.node_id}): {r1} vs {r2}"
        )

    assert {n.node_id for n in get_ancestor_nodes({out})} == src_nodes_before
    for n in get_ancestor_nodes({out}):
        assert n.value_type.value_range == src_ranges_before[n.node_id]
    assert len(collect_asserts(out)) == n_asserts_before


# ---------------------------------------------------------------------------
# The boundary is unbypassable from the compile entry
# ---------------------------------------------------------------------------


def test_forward_compile_rejects_uncertified_graph():
    from torchwright.compiler.forward.compile import forward_compile

    x = create_input("x", 4, value_range=(-2.0, 2.0))
    l1 = Linear(x, torch.randn(4, 6), torch.randn(6))
    relu = ReLU(l1)
    l2 = Linear(relu, torch.randn(6, 3), torch.randn(3))
    with pytest.raises(LoweringError, match="chain"):
        forward_compile(d=64, d_head=8, output_node=l2, verbose=False)
