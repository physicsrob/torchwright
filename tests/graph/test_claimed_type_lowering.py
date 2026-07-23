"""Claim application on node metadata (docs/assert_metadata_plan.md).

An attached range claim (``node.claimed_type``) tightens the node's
``value_type`` immediately, composes with further claims by
intersection, and survives the lowering copy — the clone's fresh
bounds recompute re-applies it at the ``refresh_node_caches`` choke
point, so a post-copy recompute can never silently widen a
claim-tightened bound (the bug class the old strip-time one-shot
transfer allowed).
"""

import torch

from torchwright.compiler.lower import lower
from torchwright.graph import InputNode, LiteralValue, NodeValueType, Range
from torchwright.graph.affine_rules import refresh_node_caches
from torchwright.graph.asserts import (
    assert_01,
    assert_in_range,
    assert_integer,
    assert_onehot,
)
from torchwright.graph.linear import Linear
from torchwright.graph.value_type import tightened_with


def test_tightened_with_intersects_range():
    a = NodeValueType.bounded(-5.0, 5.0)
    b = NodeValueType.bounded(0.0, 3.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 3.0)


def test_tightened_with_bounded_and_01():
    a = NodeValueType.bounded(0.0, 10.0)
    b = NodeValueType.bounded(0.0, 1.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 1.0)


def test_integer_claim_lands_on_node_and_copy():
    inp = InputNode("x", 3, value_range=(-100.0, 100.0))
    out = assert_integer(inp)
    assert out is inp  # attach-and-return, no wrapper node
    assert inp.integer_claim
    assert inp.value_type.value_range == Range(-100.0, 100.0)

    lowered = lower(inp)
    copy = lowered.copy_of(inp)
    assert lowered.output_node is copy
    assert copy.value_type.value_range == Range(-100.0, 100.0)
    assert copy.integer_claim  # rides the clone


def test_binary_claim_tightens_node_and_copy():
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    assert_01(inp)
    assert inp.value_type.value_range == Range(0.0, 1.0)
    lowered = lower(inp)
    assert lowered.copy_of(inp).value_type.value_range == Range(0.0, 1.0)


def test_onehot_claim_tightens_node_and_copy():
    inp = InputNode("x", 5, value_range=(-100.0, 100.0))
    assert_onehot(inp)
    assert inp.value_type.value_range == Range(0.0, 1.0)
    lowered = lower(inp)
    assert lowered.copy_of(inp).value_type.value_range == Range(0.0, 1.0)


def test_chained_claims_compose_by_intersection():
    inp = InputNode("x", 2, value_range=(-100.0, 100.0))
    assert_01(assert_integer(inp))
    assert len(inp.checks) == 2  # both predicates on the one node
    assert inp.value_type.value_range == Range(0.0, 1.0)
    assert inp.integer_claim
    lowered = lower(inp)
    assert lowered.copy_of(inp).value_type.value_range == Range(0.0, 1.0)


def test_claim_does_not_regress_existing_inferred_type():
    lit = LiteralValue(torch.tensor([1.0, 2.0, 3.0]))
    before = lit.value_type
    assert_integer(lit)
    assert lit.value_type.value_range == before.value_range
    lowered = lower(lit)
    assert lowered.copy_of(lit).value_type.value_range == before.value_range


def test_claim_survives_cache_refresh():
    """The refresh-proof property itself: recomputing a claimed node's
    caches (any pass may do this at any time) re-applies the claim
    instead of widening back to the propagated bound.
    """
    x = InputNode("x", 2, value_range=(-10.0, 10.0))
    lin = Linear(x, torch.ones(2, 2) * 3.0)  # propagated range [-60, 60]
    assert_in_range(lin, -1.0, 1.0)
    assert lin.value_type.value_range == Range(-1.0, 1.0)

    refresh_node_caches(lin)  # the recompute that used to lose claims

    assert lin.value_type.value_range == Range(-1.0, 1.0)


def test_general_target_claim_reaches_consumer_bounds():
    """A general (non-leaf) target whose finite claim is strictly tighter
    than its propagated bound: the claim degenerates the node's affine
    bound to the claim-intersected constant box, and a consumer built
    after the attach derives through that box — on the source and
    bit-identically on the lowering copy.
    """
    x = InputNode("x", 2, value_range=(-10.0, 10.0))
    lin = Linear(x, torch.ones(2, 2) * 3.0)  # propagated range [-60, 60]
    assert lin.value_type.value_range == Range(-60.0, 60.0)

    assert_in_range(lin, -1.0, 1.0)  # strictly tighter claim
    assert lin.value_type.value_range == Range(-1.0, 1.0)

    consumer = Linear(lin, torch.ones(2, 2))  # built after the attach
    consumer_range_at_construction = consumer.value_type.value_range
    # Each output sums two components of the box [-1, 1] -> [-2, 2].
    assert consumer_range_at_construction == Range(-2.0, 2.0)

    lowered = lower(consumer)

    lin_copy = lowered.copy_of(lin)
    assert lin_copy.value_type.value_range == Range(-1.0, 1.0)

    consumer_copy = lowered.copy_of(consumer)
    cr = consumer_copy.value_type.value_range
    assert (cr.lo, cr.hi) == (
        consumer_range_at_construction.lo,
        consumer_range_at_construction.hi,
    )


def test_leaf_claim_tightens_input_ranges_channel():
    """A claim on an InputNode tightens the leaf's own input_ranges entry
    (the leaf channel), so downstream bounds inherit it through normal
    affine propagation — with coefficients intact, not a constant box.
    """
    x = InputNode("x", 2, value_range=(-10.0, 10.0))
    assert_in_range(x, -2.0, 2.0)

    lo, hi = x._affine_bound.input_ranges[x.node_id]
    assert lo.tolist() == [-2.0, -2.0]
    assert hi.tolist() == [2.0, 2.0]

    consumer = Linear(x, torch.ones(2, 2))
    r = consumer.value_type.value_range
    assert (r.lo, r.hi) == (-4.0, 4.0)  # 2 components of [-2, 2] summed

    lowered = lower(consumer)
    cr = lowered.copy_of(consumer).value_type.value_range
    assert (cr.lo, cr.hi) == (-4.0, 4.0)
