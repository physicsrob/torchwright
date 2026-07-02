"""Tests for the lowering boundary (:mod:`torchwright.compiler.lower`).

``lower()`` certifies a graph for scheduling: closed vocabulary (no raw
``ReLU`` / hand-built ``Linear -> ReLU -> Linear`` chain — the check that
used to live in the deleted ``graph/blockify.py``) and fresh derived
caches (``_affine_bound`` / ``_structural_type`` recomputed in topological
order, semantic overrides re-applied).  ``forward_compile`` runs it as
step 0, so an uncertified graph cannot reach the scheduler.
"""

import pytest
import torch

from torchwright.compiler.lower import LoweredGraph, LoweringError, lower
from torchwright.graph import Block, Linear, Node, ReLU
from torchwright.graph.affine_rules import compute_affine_bound
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.misc import Concatenate
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear_relu_linear import linear_relu_linear


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
    assert isinstance(node, Block)


def test_lower_passes_on_native_block_graph():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    blk = _block(x, 4, 6, 3, name="c")
    downstream = Linear(blk, torch.randn(3, 2), torch.randn(2))
    lowered = lower(downstream)
    assert isinstance(lowered, LoweredGraph)
    assert lowered.output_node is downstream


def test_lower_passes_on_stacked_blocks():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    b1 = _block(x, 4, 6, 4, seed=1)
    b2 = _block(b1, 4, 5, 3, seed=2)
    assert lower(b2).output_node is b2


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


def test_lower_refreshes_stale_bounds():
    """Boundary twin of test_fusion_refreshes_stale_bounds: an in-place
    weight mutation (here a hand edit; in production a pass that forgot the
    per-pass refresh discipline) leaves the eagerly-cached bounds describing
    the pre-mutation graph.  lower() must restore cached == fresh for every
    reachable node."""
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    l1 = Linear(x, torch.randn(4, 6) * 0.2, torch.randn(6) * 0.1)
    blk = _block(l1, 6, 8, 5, seed=3)
    sink = Concatenate([blk])

    before = l1._affine_bound.to_scalar_range()
    l1.output_matrix = l1.output_matrix * 3.0  # stale-making mutation

    lower(sink)

    from torchwright.compiler.utils import get_ancestor_nodes

    for node in get_ancestor_nodes({sink}):
        cached = node._affine_bound.to_scalar_range()
        fresh = compute_affine_bound(node).to_scalar_range()
        assert cached.lo == fresh.lo and cached.hi == fresh.hi, (
            f"stale bound on {type(node).__name__} id={node.node_id}: "
            f"cached={cached} fresh={fresh}"
        )
    after = l1._affine_bound.to_scalar_range()
    assert (
        after.lo != before.lo or after.hi != before.hi
    ), "mutation should have widened l1's bound; refresh did not pick it up"


def test_lower_preserves_semantic_overrides():
    """Ops install semantic affine overrides (tighter than pure propagation)
    via _apply_semantic_override; lower()'s recompute must re-apply them, not
    wipe them — a wipe would silently loosen every downstream bound."""
    from torchwright.ops.arithmetic_ops import compare

    x = create_input("x", 1, value_range=(-10.0, 10.0))
    cmp = compare(x, 0.0, true_level=1.0, false_level=-1.0)

    # Find the node carrying the override (the op applies it to its result).
    from torchwright.compiler.utils import get_ancestor_nodes

    overridden = [
        n for n in get_ancestor_nodes({cmp}) if n._semantic_affine_override is not None
    ]
    assert overridden, "compare() should install a semantic override"

    ranges_before = {n.node_id: n._affine_bound.to_scalar_range() for n in overridden}
    lower(cmp)
    for n in overridden:
        r_before = ranges_before[n.node_id]
        r_after = n._affine_bound.to_scalar_range()
        assert (r_after.lo, r_after.hi) == (r_before.lo, r_before.hi), (
            f"semantic override lost on node {n.node_id}: "
            f"before={r_before} after={r_after}"
        )
        assert n._semantic_affine_override is not None


def test_lower_is_idempotent():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    blk = _block(x, 4, 6, 3, seed=5)
    lower(blk)
    r1 = blk._affine_bound.to_scalar_range()
    lower(blk)
    r2 = blk._affine_bound.to_scalar_range()
    assert (r1.lo, r1.hi) == (r2.lo, r2.hi)


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
