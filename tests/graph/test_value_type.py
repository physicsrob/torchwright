"""Unit tests for ``Range`` arithmetic and ``NodeValueType``."""

import math

import pytest

from torchwright.graph import NodeValueType, Range
from torchwright.graph.value_type import tightened_with

# --- Range ------------------------------------------------------------


def test_range_default_is_unbounded():
    r = Range()
    assert r.lo == -math.inf
    assert r.hi == math.inf


def test_range_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Range(1.0, 0.0)


def test_range_point():
    r = Range.point(3.0)
    assert r.lo == 3.0
    assert r.hi == 3.0


def test_range_add():
    a = Range(0.0, 1.0)
    b = Range(2.0, 5.0)
    assert a + b == Range(2.0, 6.0)


def test_range_neg_and_sub():
    a = Range(1.0, 3.0)
    assert -a == Range(-3.0, -1.0)
    assert a - Range(0.0, 1.0) == Range(0.0, 3.0)


def test_range_union_and_intersect():
    a = Range(0.0, 2.0)
    b = Range(1.0, 3.0)
    assert a.union(b) == Range(0.0, 3.0)
    assert a.intersect(b) == Range(1.0, 2.0)


def test_range_relu_clamps_negatives_to_zero():
    assert Range(-2.0, 3.0).relu() == Range(0.0, 3.0)
    assert Range(-5.0, -1.0).relu() == Range(0.0, 0.0)
    assert Range(2.0, 5.0).relu() == Range(2.0, 5.0)


def test_range_contains():
    outer = Range(0.0, 10.0)
    assert outer.contains(Range(1.0, 5.0))
    assert not outer.contains(Range(-1.0, 5.0))


# --- NodeValueType ---------------------------------------------------


def test_unknown_has_unbounded_range():
    t = NodeValueType.unknown()
    assert t.value_range == Range.unbounded()


def test_bounded_factory():
    t = NodeValueType.bounded(0.0, 9.0)
    assert t.value_range == Range(0.0, 9.0)


# --- Combinators ------------------------------------------------------


def test_tightened_with_intersects_ranges():
    a = NodeValueType.bounded(-5.0, 5.0)
    b = NodeValueType.bounded(0.0, 3.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 3.0)


def test_tightened_with_unbounded_and_bounded():
    a = NodeValueType.unknown()
    b = NodeValueType.bounded(0.0, 9.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 9.0)


def test_tightened_with_both_same_range():
    a = NodeValueType.bounded(0.0, 9.0)
    b = NodeValueType.bounded(0.0, 9.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 9.0)


# --- Node.value_type: affine-vs-structural reconciliation -------------
#
# The affine bound is float64 interval arithmetic; its rounding can push a
# bound a few ULPs past the structural type (e.g. a clamp gadget whose output
# is structurally in [0, 1] but whose affine bound collapses to the point
# 1+2^-26).  `value_type` must reconcile that fp-noise crossing by clamping
# onto the exact structural boundary rather than raising on the empty
# intersection — but a *gross* (non-rounding) disjointness must still raise.
# Regression for the DOOM `compile_to_onnx` failure exposed by the RMSNorm
# energy certification reading every residual node's `value_type`.


def _node_with_bounds(point_value, structural):
    import torch

    from torchwright.graph import LiteralValue
    from torchwright.graph.affine_bound import AffineBound

    node = LiteralValue(torch.tensor([1.0], dtype=torch.float32))
    # Pin a float64 affine point at `point_value` (the real clamp gadget's
    # affine bound is float64 and carries sub-float32-ULP slop that a float32
    # literal would round away) and the tighter, exact structural range.
    node._affine_bound = AffineBound.constant(
        torch.tensor([point_value], dtype=torch.float64)
    )
    node._structural_type = NodeValueType.bounded(*structural)
    return node


def test_value_type_clamps_fp_noise_affine_overshoot():
    # Affine point 1+2^-26 sits 1.5e-8 above the structural hi of 1.0.
    node = _node_with_bounds(1.0 + 2.0**-26, (0.0, 1.0))
    assert node._affine_bound.to_scalar_range().lo > 1.0  # genuinely overshoots
    vt = node.value_type
    assert vt.value_range == Range(1.0, 1.0)


def test_value_type_clamps_fp_noise_affine_undershoot():
    # Symmetric: an affine point a hair below the structural lo.
    node = _node_with_bounds(-(2.0**-26), (0.0, 1.0))
    assert node._affine_bound.to_scalar_range().hi < 0.0
    assert node.value_type.value_range == Range(0.0, 0.0)


def test_value_type_raises_on_gross_disjointness():
    # 5.0 vs [0, 1] is far beyond fp-rounding noise: a real soundness bug.
    node = _node_with_bounds(5.0, (0.0, 1.0))
    with pytest.raises(ValueError, match="disjoint from structural type"):
        _ = node.value_type
