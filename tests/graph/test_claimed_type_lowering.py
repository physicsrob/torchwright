"""Claim transfer at the lowering boundary: an Assert's ``claimed_type``
must survive the wrapper strip and appear on the *copy* of the wrapped
node's ``value_type`` — while the source graph (wrappers, bounds, types)
stays untouched.  Compilation is a pure function of the source
(``docs/lowering_copy_plan.md``); the strip runs on the compiler-private
copy inside ``lower()``.
"""

import torch

from torchwright.compiler.lower import lower
from torchwright.graph import InputNode, LiteralValue, NodeValueType, Range
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


def test_claim_transfers_integer_range_to_copied_node():
    inp = InputNode("x", 3, value_range=(-100.0, 100.0))
    wrapped = assert_integer(inp)
    lowered = lower(wrapped)
    # The copy's output is the (stripped) clone of the wrapped node.
    assert lowered.output_node is lowered.copy_of(inp)
    assert lowered.copy_of(inp).value_type.value_range == Range(-100.0, 100.0)
    # Source untouched — wrapper intact, type unchanged.
    assert wrapped.inputs[0] is inp
    assert inp.value_type.value_range == Range(-100.0, 100.0)


def test_claim_transfers_binary_range():
    inp = InputNode("x", 4, value_range=(-100.0, 100.0))
    wrapped = assert_01(inp)
    lowered = lower(wrapped)
    assert lowered.copy_of(inp).value_type.value_range == Range(0.0, 1.0)
    assert inp.value_type.value_range == Range(-100.0, 100.0)


def test_claim_transfers_onehot_range():
    inp = InputNode("x", 5, value_range=(-100.0, 100.0))
    wrapped = assert_onehot(inp)
    lowered = lower(wrapped)
    assert lowered.copy_of(inp).value_type.value_range == Range(0.0, 1.0)
    assert inp.value_type.value_range == Range(-100.0, 100.0)


def test_chained_assert_claims_compose():
    inp = InputNode("x", 2, value_range=(-100.0, 100.0))
    inner = assert_integer(inp)
    outer = assert_01(inner)
    lowered = lower(outer)
    assert lowered.copy_of(inp).value_type.value_range == Range(0.0, 1.0)
    # Both wrapper entries in the map resolve to the same stripped clone.
    assert lowered.copy_of(inner) is lowered.copy_of(inp)
    assert lowered.copy_of(outer) is lowered.copy_of(inp)
    assert inp.value_type.value_range == Range(-100.0, 100.0)


def test_claim_does_not_regress_existing_inferred_type():
    lit = LiteralValue(torch.tensor([1.0, 2.0, 3.0]))
    before = lit.value_type
    wrapped = assert_integer(lit)
    lowered = lower(wrapped)
    assert lowered.copy_of(lit).value_type.value_range == before.value_range
    assert lit.value_type.value_range == before.value_range


def test_general_target_claim_with_wrapper_consumer():
    """The case the old suite always lacked: a general (non-leaf) target
    whose finite claim is strictly tighter than its propagated bound,
    with a consumer built on the wrapper.

    The claim reaches bounds through two channels (decision 2 of the
    lowering-copy plan): the wrapper's affine rule collapses the
    *wrapper's* bound to a constant box (claim ∩ propagated), which the
    consumer's construction-time bound derives through; and the strip
    tightens the *wrapped node's* structural type.  The copy must
    reproduce both bit-identically — the consumer clone's bound is
    computed through the cloned wrapper before the strip rewires it.
    """
    x = InputNode("x", 2, value_range=(-10.0, 10.0))
    lin = Linear(x, torch.ones(2, 2) * 3.0)  # propagated range [-60, 60]
    propagated = lin.value_type.value_range
    assert propagated == Range(-60.0, 60.0)

    wrapped = assert_in_range(lin, -1.0, 1.0)  # strictly tighter claim
    consumer = Linear(wrapped, torch.ones(2, 2))  # built on the wrapper
    consumer_range_at_construction = consumer.value_type.value_range
    # The consumer's bound derives through the wrapper's constant box:
    # each output sums two components of the box [-1, 1] -> [-2, 2].
    assert consumer_range_at_construction == Range(-2.0, 2.0)

    lowered = lower(consumer)

    # Copy of the target: structural type tightened by the claim.
    lin_copy = lowered.copy_of(lin)
    assert lin_copy.value_type.value_range == Range(-1.0, 1.0)

    # Copy of the consumer: bound bit-identical to the source's
    # (derived through the wrapper box, exactly as at construction).
    consumer_copy = lowered.copy_of(consumer)
    cr = consumer_copy.value_type.value_range
    assert (cr.lo, cr.hi) == (
        consumer_range_at_construction.lo,
        consumer_range_at_construction.hi,
    )

    # Source untouched: consumer still wired through the wrapper, and
    # the target's own type still claim-free.
    assert consumer.inputs[0] is wrapped
    assert lin.value_type.value_range == propagated
