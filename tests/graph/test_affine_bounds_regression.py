"""Regression gate for affine bound tightening.

Asserts that affine-derived intervals at bound-consumer sites never
*loosen* relative to the eager scalar interval arithmetic — they may
tighten (which is the whole point) but must never grow.

With eager affine bounds (computed in Node.__init__), there is no
separate finalize step. The value_type already reflects tightened
ranges at construction time.
"""

import pytest
import torch

from torchwright.graph import (
    Add,
    Concatenate,
    InputNode,
    Linear,
    LiteralValue,
    ReLU,
)
from torchwright.graph.session import fresh_graph_session


class TestTighteningContract:
    """Affine bounds must be at least as tight as eager scalar bounds."""

    def test_linear_tighter_than_eager(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            W = torch.tensor([[1.0, -1.0], [1.0, 1.0]])
            lin = Linear(x, W)
            affine = lin.affine_bound.to_scalar_range()
            eager_range = lin.value_type.value_range
            assert eager_range.lo >= affine.lo - 1e-10
            assert eager_range.hi <= affine.hi + 1e-10

    def test_add_cancel_tighter_than_eager(self):
        """X + (-x) = 0: affine sees [0,0], eager interval would give [-2,2]."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            neg = Linear(x, torch.tensor([[-1.0]]))
            s = Add(x, neg)
            r = s.value_type.value_range
            assert r.lo == pytest.approx(0.0)
            assert r.hi == pytest.approx(0.0)

    def test_relu_tighter_than_eager(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-2.0, 3.0))
            r = ReLU(x)
            affine = r.affine_bound.to_scalar_range()
            eager_range = r.value_type.value_range
            assert eager_range.lo >= affine.lo - 1e-10
            assert eager_range.hi <= affine.hi + 1e-10

    def test_concat_tighter_than_eager(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            y = InputNode(1, name="y", value_range=(0.0, 5.0))
            c = Concatenate([x, y])
            affine = c.affine_bound.to_scalar_range()
            eager_range = c.value_type.value_range
            assert eager_range.lo >= affine.lo - 1e-10
            assert eager_range.hi <= affine.hi + 1e-10


class TestCancellationTightening:
    """The canonical use case: offset cancellation in cond_gate-style patterns."""

    def test_add_offset_then_subtract(self):
        """X + M + (y - M) should give range(x) + range(y), not range(x) + range(y) + 2*M."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            y = InputNode(1, name="y", value_range=(-1.0, 1.0))
            M = 100.0
            offset = LiteralValue(torch.tensor([M]))
            neg_offset = LiteralValue(torch.tensor([-M]))
            x_plus_M = Add(x, offset)
            y_minus_M = Add(y, neg_offset)
            total = Add(x_plus_M, y_minus_M)
            r = total.value_type.value_range
            assert r.lo == pytest.approx(-2.0)
            assert r.hi == pytest.approx(2.0)

    def test_scale_then_cancel(self):
        """2x + (-2x) = 0 regardless of offset magnitude."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-50.0, 50.0))
            scaled = Linear(x, torch.tensor([[2.0]]))
            neg_scaled = Linear(x, torch.tensor([[-2.0]]))
            s = Add(scaled, neg_scaled)
            r = s.value_type.value_range
            assert r.lo == pytest.approx(0.0)
            assert r.hi == pytest.approx(0.0)
