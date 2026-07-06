"""Unit tests for eager affine bound propagation: AffineBound factories,
alignment, to_interval() concretization, and per-op rules.
"""

import math

import pytest
import torch

from torchwright.graph import InputNode, LiteralValue
from torchwright.graph.affine_bound import AffineBound
from torchwright.graph.session import fresh_graph_session
from torchwright.graph.value_type import Range

# --- AffineBound factories -----------------------------------------------


class TestAffineBoundFactories:
    def test_identity(self):
        with fresh_graph_session():
            x = InputNode(3, name="x", value_range=(-1.0, 1.0))
            ab = AffineBound.identity(x)
            assert ab.d_output == 3
            assert ab.n_cols == 3
            assert ab.columns == {x.node_id: (0, 3)}
            assert torch.equal(ab.A_lo, torch.eye(3, dtype=torch.float64))
            assert torch.equal(ab.A_hi, torch.eye(3, dtype=torch.float64))
            assert torch.equal(ab.b_lo, torch.zeros(3, dtype=torch.float64))
            assert torch.equal(ab.b_hi, torch.zeros(3, dtype=torch.float64))

    def test_constant(self):
        vals = torch.tensor([3.0, 7.0])
        ab = AffineBound.constant(vals)
        assert ab.d_output == 2
        assert ab.n_cols == 0
        assert ab.columns == {}
        assert torch.allclose(ab.b_lo, vals.double())
        assert torch.allclose(ab.b_hi, vals.double())

    def test_degenerate(self):
        ab = AffineBound.degenerate(4, lo=-5.0, hi=10.0)
        assert ab.d_output == 4
        assert ab.n_cols == 0
        assert torch.allclose(ab.b_lo, torch.full((4,), -5.0, dtype=torch.float64))
        assert torch.allclose(ab.b_hi, torch.full((4,), 10.0, dtype=torch.float64))

    def test_degenerate_defaults_to_inf(self):
        ab = AffineBound.degenerate(2)
        assert ab.b_lo[0].item() == float("-inf")
        assert ab.b_hi[0].item() == float("inf")


# --- to_interval() -------------------------------------------------------


class TestToInterval:
    def test_identity_interval_matches_input_range(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-3.0, 5.0))
            ab = AffineBound.identity(x)
            intervals = ab.to_interval()
            assert len(intervals) == 2
            assert intervals[0].lo == pytest.approx(-3.0)
            assert intervals[0].hi == pytest.approx(5.0)
            assert intervals[1].lo == pytest.approx(-3.0)
            assert intervals[1].hi == pytest.approx(5.0)

    def test_constant_interval(self):
        ab = AffineBound.constant(torch.tensor([3.0, 7.0]))
        intervals = ab.to_interval()
        assert intervals[0].lo == pytest.approx(3.0)
        assert intervals[0].hi == pytest.approx(3.0)
        assert intervals[1].lo == pytest.approx(7.0)
        assert intervals[1].hi == pytest.approx(7.0)

    def test_degenerate_interval(self):
        ab = AffineBound.degenerate(1, lo=-5.0, hi=10.0)
        intervals = ab.to_interval()
        assert intervals[0].lo == pytest.approx(-5.0)
        assert intervals[0].hi == pytest.approx(10.0)

    def test_scalar_range_union(self):
        ab = AffineBound.constant(torch.tensor([3.0, 7.0]))
        r = ab.to_scalar_range()
        assert r.lo == pytest.approx(3.0)
        assert r.hi == pytest.approx(7.0)


# --- Alignment -----------------------------------------------------------


class TestAlign:
    def test_identical_columns_fast_path(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            a = AffineBound.identity(x)
            b = AffineBound.identity(x)
            a2, b2 = AffineBound.align(a, b)
            assert a2.columns == a.columns
            assert torch.equal(a2.A_lo, a.A_lo)

    def test_disjoint_inputs_merge(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            y = InputNode(3, name="y", value_range=(0.0, 5.0))
            ax = AffineBound.identity(x)
            ay = AffineBound.identity(y)
            ax2, ay2 = AffineBound.align(ax, ay)
            assert ax2.n_cols == 5
            assert ay2.n_cols == 5
            assert ax2.columns == ay2.columns
            # x's identity is in first 2 cols, zeros in last 3
            assert ax2.A_lo[0, 0].item() == 1.0
            assert ax2.A_lo[0, 2].item() == 0.0
            # y's identity is in last 3 cols, zeros in first 2
            assert ay2.A_lo[0, 0].item() == 0.0
            x_id, y_id = x.node_id, y.node_id
            y_start = ay2.columns[y_id][0]
            assert ay2.A_lo[0, y_start].item() == 1.0

    def test_ranges_intersected(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-5.0, 5.0))
            a = AffineBound(
                A_lo=torch.ones(1, 1, dtype=torch.float64),
                A_hi=torch.ones(1, 1, dtype=torch.float64),
                b_lo=torch.zeros(1, dtype=torch.float64),
                b_hi=torch.zeros(1, dtype=torch.float64),
                columns={x.node_id: (0, 1)},
                input_ranges={
                    x.node_id: (
                        torch.tensor([-5.0], dtype=torch.float64),
                        torch.tensor([5.0], dtype=torch.float64),
                    )
                },
            )
            b = AffineBound(
                A_lo=torch.ones(1, 1, dtype=torch.float64),
                A_hi=torch.ones(1, 1, dtype=torch.float64),
                b_lo=torch.zeros(1, dtype=torch.float64),
                b_hi=torch.zeros(1, dtype=torch.float64),
                columns={x.node_id: (0, 1)},
                input_ranges={
                    x.node_id: (
                        torch.tensor([-2.0], dtype=torch.float64),
                        torch.tensor([3.0], dtype=torch.float64),
                    )
                },
            )
            a2, b2 = AffineBound.align(a, b)
            lo, hi = a2.input_ranges[x.node_id]
            assert lo.item() == pytest.approx(-2.0)
            assert hi.item() == pytest.approx(3.0)
            lo2, hi2 = b2.input_ranges[x.node_id]
            assert lo2.item() == pytest.approx(-2.0)
            assert hi2.item() == pytest.approx(3.0)


# --- Session management --------------------------------------------------


class TestSession:
    def test_fresh_session_isolates_inputs(self):
        with fresh_graph_session() as s1:
            x = InputNode(2, name="x", value_range=(-100.0, 100.0))
            assert len(s1.input_nodes) == 1
        with fresh_graph_session() as s2:
            y = InputNode(3, name="y", value_range=(-100.0, 100.0))
            assert len(s2.input_nodes) == 1
            assert s2.input_nodes[0] is y

    def test_nested_session_raises(self):
        with fresh_graph_session():
            with pytest.raises(RuntimeError, match="Nested"):
                with fresh_graph_session():
                    pass


# --- Eager bounds (computed in __init__) -----------------------------------


class TestEagerBounds:
    def test_input_has_affine_bound(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            assert x._affine_bound is not None
            assert x.affine_bound.d_output == 2

    def test_add_has_affine_bound(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            lit = LiteralValue(torch.tensor([3.0, 5.0]))
            from torchwright.graph import Add

            s = Add(x, lit)
            assert s._affine_bound is not None
            assert x._affine_bound is not None
            assert lit._affine_bound is not None

    def test_input_identity_interval(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            ab = x.affine_bound
            intervals = ab.to_interval()
            assert intervals[0].lo == pytest.approx(-1.0)
            assert intervals[0].hi == pytest.approx(1.0)

    def test_repr_works(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            r = repr(x.affine_bound)
            assert "AffineBound" in r

    def test_column_map_two_inputs_add(self):
        """Add of two InputNodes merges their column maps."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(0.0, 1.0))
            y = InputNode(1, name="y", value_range=(2.0, 3.0))
            from torchwright.graph import Add

            s = Add(x, y)
            assert x.node_id in s.affine_bound.columns
            assert y.node_id in s.affine_bound.columns
            assert s.affine_bound.n_cols == 2


# --- Exact affine rules ---------------------------------------------------


class TestLinearRule:
    def test_identity_matrix_preserves_input(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-3.0, 5.0))
            lin = __import__("torchwright.graph", fromlist=["Linear"]).Linear(
                x, torch.eye(2), name="id"
            )
            intervals = lin.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(-3.0)
            assert intervals[0].hi == pytest.approx(5.0)

    def test_scaling_matrix(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(0.0, 2.0))
            W = torch.tensor([[3.0]])
            from torchwright.graph import Linear

            lin = Linear(x, W, name="scale")
            intervals = lin.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(0.0)
            assert intervals[0].hi == pytest.approx(6.0)

    def test_negative_weight(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(1.0, 3.0))
            W = torch.tensor([[-2.0]])
            from torchwright.graph import Linear

            lin = Linear(x, W, name="neg")
            intervals = lin.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(-6.0)
            assert intervals[0].hi == pytest.approx(-2.0)

    def test_bias(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(0.0, 1.0))
            W = torch.tensor([[1.0]])
            b = torch.tensor([5.0])
            from torchwright.graph import Linear

            lin = Linear(x, W, b, name="bias")
            intervals = lin.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(5.0)
            assert intervals[0].hi == pytest.approx(6.0)


class TestAddRule:
    def test_add_tracks_correlation(self):
        """x + (-x) should give [0, 0] via affine tracking, not [-2, 2]."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            from torchwright.graph import Linear, Add

            neg_x = Linear(x, torch.tensor([[-1.0]]), name="neg")
            s = Add(x, neg_x, name="cancel")
            intervals = s.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(0.0)
            assert intervals[0].hi == pytest.approx(0.0)

    def test_add_independent_inputs(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(0.0, 1.0))
            y = InputNode(1, name="y", value_range=(2.0, 3.0))
            from torchwright.graph import Add

            s = Add(x, y, name="sum")
            intervals = s.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(2.0)
            assert intervals[0].hi == pytest.approx(4.0)


class TestConcatRule:
    def test_concat_stacks_bounds(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            y = InputNode(1, name="y", value_range=(0.0, 5.0))
            from torchwright.graph import Concatenate

            c = Concatenate([x, y])
            intervals = c.affine_bound.to_interval()
            assert len(intervals) == 3
            assert intervals[0].lo == pytest.approx(-1.0)
            assert intervals[0].hi == pytest.approx(1.0)
            assert intervals[2].lo == pytest.approx(0.0)
            assert intervals[2].hi == pytest.approx(5.0)


class TestLiteralRule:
    def test_literal_constant_bound(self):
        with fresh_graph_session():
            lit = LiteralValue(torch.tensor([3.0, 7.0]))
            intervals = lit.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(3.0)
            assert intervals[0].hi == pytest.approx(3.0)
            assert intervals[1].lo == pytest.approx(7.0)
            assert intervals[1].hi == pytest.approx(7.0)


class TestDualRail:
    def test_value_type_tightens_from_affine(self):
        """Affine bounds should tighten value_type range eagerly."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            from torchwright.graph import Linear, Add

            neg_x = Linear(x, torch.tensor([[-1.0]]), name="neg")
            s = Add(x, neg_x, name="cancel")
            # Eager range would be [-2, 2] but affine tightens to [0, 0]
            assert s.value_type.value_range.lo == pytest.approx(0.0)
            assert s.value_type.value_range.hi == pytest.approx(0.0)

    def test_value_type_preserves_range(self):
        with fresh_graph_session():
            lit = LiteralValue(torch.tensor([0.0, 1.0]))
            assert lit.value_type.value_range == Range(0.0, 1.0)


class TestSoundness:
    """Randomized checks: sample from the basis box and verify actual <= bound."""

    def test_linear_soundness(self):
        import random

        random.seed(42)
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-3.0, 3.0))
            W = torch.randn(2, 3)
            b = torch.randn(3)
            from torchwright.graph import Linear

            lin = Linear(x, W, b, name="test")
            intervals = lin.affine_bound.to_interval()
            for _ in range(100):
                xv = torch.FloatTensor(1, 2).uniform_(-3.0, 3.0)
                y = (xv @ W + b).squeeze(0)
                for j in range(3):
                    assert y[j].item() >= intervals[j].lo - 1e-5
                    assert y[j].item() <= intervals[j].hi + 1e-5

    def test_add_cancel_soundness(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-10.0, 10.0))
            from torchwright.graph import Linear, Add

            neg = Linear(x, torch.tensor([[-1.0]]))
            s = Add(x, neg)
            r = s.affine_bound.to_interval()[0]
            assert r.lo == pytest.approx(0.0)
            assert r.hi == pytest.approx(0.0)


# --- ReLU envelope --------------------------------------------------------


class TestReluRule:
    def test_relu_identity_positive(self):
        """When input is fully positive, ReLU is identity."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(1.0, 3.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            intervals = r.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(1.0)
            assert intervals[0].hi == pytest.approx(3.0)

    def test_relu_zero_negative(self):
        """When input is fully negative, ReLU output is 0."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-5.0, -1.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            intervals = r.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(0.0)
            assert intervals[0].hi == pytest.approx(0.0)

    def test_relu_straddling(self):
        """Straddling case uses linear envelope."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-2.0, 4.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            intervals = r.affine_bound.to_interval()
            assert intervals[0].lo >= -2.0 - 1e-5
            assert intervals[0].hi == pytest.approx(4.0)
            assert intervals[0].hi <= 4.0 + 1e-5

    def test_relu_soundness(self):
        """Randomized soundness: actual relu values within affine bounds."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-3.0, 5.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            intervals = r.affine_bound.to_interval()
            for _ in range(200):
                xv = torch.FloatTensor(1).uniform_(-3.0, 5.0)
                yv = torch.clamp(xv, min=0.0).item()
                assert yv >= intervals[0].lo - 1e-5
                assert yv <= intervals[0].hi + 1e-5


# --- Claim channels (leaf + general) ---------------------------------------


class TestClaimChannels:
    def test_leaf_claim_preserves_coefficients(self):
        """A claim on an InputNode tightens its input_ranges entry, not
        its coefficients (the leaf channel keeps the affine structure)."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-5.0, 5.0))
            from torchwright.graph.asserts import assert_in_range

            before_A = x.affine_bound.A_lo.clone()
            a = assert_in_range(x, -3.0, 3.0)
            assert a is x
            assert torch.equal(a.affine_bound.A_lo, before_A)
            assert torch.equal(a.affine_bound.A_hi, before_A)
            assert a.value_type.value_range.lo == pytest.approx(-3.0)
            assert a.value_type.value_range.hi == pytest.approx(3.0)

    def test_assert_tightens_downstream(self):
        """Tightened input_ranges propagate through downstream Linear."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-10.0, 10.0))
            from torchwright.graph import Linear
            from torchwright.graph.asserts import assert_in_range

            a = assert_in_range(x, -2.0, 3.0)
            scaled = Linear(a, torch.tensor([[2.0]]), name="scale")
            intervals = scaled.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(-4.0)
            assert intervals[0].hi == pytest.approx(6.0)

    def test_assert_chain_tightens(self):
        """Chained claims (assert_01(assert_integer(x))) intersect."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-10.0, 10.0))
            from torchwright.graph.asserts import assert_01, assert_integer

            a1 = assert_integer(x)
            a2 = assert_01(a1)
            intervals = a2.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(0.0)
            assert intervals[0].hi == pytest.approx(1.0)

    def test_claim_tightens_the_node_itself(self):
        """The claim is a fact about the node's value: the node's own
        bound tightens, and every consumer — including ones reading the
        node through another handle — sees it (sound: the claim is
        runtime-checked)."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-10.0, 10.0))
            from torchwright.graph.asserts import assert_in_range

            assert_in_range(x, -2.0, 3.0)
            x_intervals = x.affine_bound.to_interval()
            assert x_intervals[0].lo == pytest.approx(-2.0)
            assert x_intervals[0].hi == pytest.approx(3.0)

    def test_parallel_claims_intersect_for_all_consumers(self):
        """Two claims on the same node intersect; every consumer sees
        the intersection (claims commute — attach order irrelevant)."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-10.0, 10.0))
            from torchwright.graph import Linear
            from torchwright.graph.asserts import assert_in_range

            a1 = assert_in_range(x, -2.0, 3.0)
            a2 = assert_in_range(x, -5.0, 5.0)
            assert a1 is x and a2 is x
            lin1 = Linear(a1, torch.tensor([[1.0]]))
            lin2 = Linear(a2, torch.tensor([[1.0]]))
            for lin in (lin1, lin2):
                assert lin.affine_bound.to_interval()[0].lo == pytest.approx(-2.0)
                assert lin.affine_bound.to_interval()[0].hi == pytest.approx(3.0)

    def test_multiple_asserts_intersect_in_add(self):
        """Multiple claims on the same InputNode intersect when added."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-100.0, 100.0))
            from torchwright.graph import Add
            from torchwright.graph.asserts import assert_in_range

            a1 = assert_in_range(x, -5.0, 10.0)
            a2 = assert_in_range(x, -3.0, 20.0)
            s = Add(a1, a2)
            # align intersects input_ranges: max(-5,-3)=-3, min(10,20)=10
            # s = 2*x with x in [-3, 10] -> [-6, 20]
            intervals = s.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(-6.0)
            assert intervals[0].hi == pytest.approx(20.0)


# --- Attn degenerate ------------------------------------------------------


class TestSemanticBounds:
    """Semantic affine bounds for composite ops: cond_gate, select, compare."""

    def test_cond_gate_positive_input(self):
        """cond_gate with positive input: upper = identity, lower = 0.

        c_tol=0.005 widens the semantic bound by M*c_tol per side.
        """
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            inp = InputNode(2, name="inp", value_range=(1.0, 5.0))
            from torchwright.ops.relu.logic_ops import cond_gate

            result = cond_gate(cond, inp)
            intervals = result.affine_bound.to_interval()
            for iv in intervals:
                assert iv.lo == pytest.approx(0.0, abs=0.06)
                assert iv.hi == pytest.approx(5.0, abs=0.06)

    def test_cond_gate_negative_input(self):
        """cond_gate with negative input: upper = 0, lower = identity."""
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            inp = InputNode(2, name="inp", value_range=(-5.0, -1.0))
            from torchwright.ops.relu.logic_ops import cond_gate

            result = cond_gate(cond, inp)
            intervals = result.affine_bound.to_interval()
            for iv in intervals:
                assert iv.lo == pytest.approx(-5.0, abs=0.06)
                assert iv.hi == pytest.approx(0.0, abs=0.06)

    def test_cond_gate_straddling(self):
        """cond_gate with straddling input: bounded by [lo, hi]."""
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            inp = InputNode(1, name="inp", value_range=(-3.0, 5.0))
            from torchwright.ops.relu.logic_ops import cond_gate

            result = cond_gate(cond, inp)
            iv = result.affine_bound.to_interval()[0]
            assert iv.lo >= -3.0 - 0.06
            assert iv.hi <= 5.0 + 0.06
            assert iv.lo <= 0.0
            assert iv.hi >= 0.0

    def test_cond_gate_soundness(self):
        """Randomized: actual cond_gate output within semantic bounds."""
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            inp = InputNode(2, name="inp", value_range=(-3.0, 7.0))
            from torchwright.ops.relu.logic_ops import cond_gate

            result = cond_gate(cond, inp)
            intervals = result.affine_bound.to_interval()
            for _ in range(200):
                c = torch.FloatTensor(1).uniform_(-1.0, 1.0)
                v = torch.FloatTensor(2).uniform_(-3.0, 7.0)
                actual = torch.where(c > 0, v, torch.zeros_like(v))
                for j in range(2):
                    assert actual[j].item() >= intervals[j].lo - 1e-5
                    assert actual[j].item() <= intervals[j].hi + 1e-5

    def test_cond_gate_tighter_than_naive(self):
        """Semantic bound should be tighter than the MLP-derived bound."""
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            inp = InputNode(1, name="inp", value_range=(1.0, 5.0))
            from torchwright.ops.relu.logic_ops import cond_gate

            result = cond_gate(cond, inp)
            iv = result.affine_bound.to_interval()[0]
            width = iv.hi - iv.lo
            assert width <= 6.0, f"Semantic bound width {width} should be <= 6 (0 to 5)"

    def test_select_hull(self):
        """select bound is the hull of true/false intervals, widened by c_tol * M."""
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            a = InputNode(1, name="a", value_range=(2.0, 5.0))
            b = InputNode(1, name="b", value_range=(-1.0, 3.0))
            from torchwright.ops.relu.map_select import select

            result = select(cond, a, b)
            iv = result.affine_bound.to_interval()[0]
            assert iv.lo == pytest.approx(-1.0, abs=0.06)
            assert iv.hi == pytest.approx(5.0, abs=0.06)

    def test_select_soundness(self):
        """Randomized: actual select output within semantic bounds."""
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            a = InputNode(2, name="a", value_range=(0.0, 10.0))
            b = InputNode(2, name="b", value_range=(-5.0, 3.0))
            from torchwright.ops.relu.map_select import select

            result = select(cond, a, b)
            intervals = result.affine_bound.to_interval()
            for _ in range(200):
                c = torch.FloatTensor(1).uniform_(-1.0, 1.0)
                av = torch.FloatTensor(2).uniform_(0.0, 10.0)
                bv = torch.FloatTensor(2).uniform_(-5.0, 3.0)
                actual = av if c.item() > 0 else bv
                for j in range(2):
                    assert actual[j].item() >= intervals[j].lo - 1e-5
                    assert actual[j].item() <= intervals[j].hi + 1e-5

    def test_compare_definite_above(self):
        """When input is definitely above threshold, compare is constant."""
        with fresh_graph_session():
            inp = InputNode(1, name="inp", value_range=(5.0, 10.0))
            from torchwright.ops.relu.arithmetic_ops import compare

            result = compare(inp, thresh=3.0, true_level=1.0, false_level=-1.0)
            iv = result.affine_bound.to_interval()[0]
            assert iv.lo == pytest.approx(1.0, abs=1e-5)
            assert iv.hi == pytest.approx(1.0, abs=1e-5)

    def test_compare_definite_below(self):
        """When input is definitely below threshold, compare is constant."""
        with fresh_graph_session():
            inp = InputNode(1, name="inp", value_range=(-10.0, -1.0))
            from torchwright.ops.relu.arithmetic_ops import compare

            result = compare(inp, thresh=0.0, true_level=1.0, false_level=-1.0)
            iv = result.affine_bound.to_interval()[0]
            assert iv.lo == pytest.approx(-1.0, abs=1e-5)
            assert iv.hi == pytest.approx(-1.0, abs=1e-5)

    def test_compare_straddling(self):
        """When input straddles threshold, compare bound is [min, max] of levels."""
        with fresh_graph_session():
            inp = InputNode(1, name="inp", value_range=(-5.0, 5.0))
            from torchwright.ops.relu.arithmetic_ops import compare

            result = compare(inp, thresh=0.0, true_level=1.0, false_level=-1.0)
            iv = result.affine_bound.to_interval()[0]
            assert iv.lo == pytest.approx(-1.0, abs=1e-5)
            assert iv.hi == pytest.approx(1.0, abs=1e-5)

    def test_compare_soundness(self):
        """Randomized: actual compare output within semantic bounds."""
        with fresh_graph_session():
            inp = InputNode(1, name="inp", value_range=(-5.0, 5.0))
            from torchwright.ops.relu.arithmetic_ops import compare

            result = compare(inp, thresh=2.0, true_level=7.0, false_level=-3.0)
            iv = result.affine_bound.to_interval()[0]
            for _ in range(200):
                v = torch.FloatTensor(1).uniform_(-5.0, 5.0).item()
                actual = 7.0 if v > 2.0 else -3.0
                assert actual >= iv.lo - 1e-5
                assert actual <= iv.hi + 1e-5

    def test_cond_gate_preserves_correlation(self):
        """Semantic bound preserves inp correlation through the gate.

        Also pins the override-vs-claim precedence on the affine channel
        (see ``_apply_semantic_override``): cond_gate's result carries both
        a finite range claim and a semantic override, and if the claim box
        were applied on top of the override it would degenerate the bound
        to a constant box with no input columns — the ``columns`` assertion
        below fails under exactly that regression.
        """
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            inp = InputNode(1, name="inp", value_range=(2.0, 5.0))
            from torchwright.ops.relu.logic_ops import cond_gate

            result = cond_gate(cond, inp)
            ab = result.affine_bound
            assert inp.node_id in ab.columns


class TestEmbeddingRule:
    def test_embedding_identity_bound(self):
        """Embedding produces an identity A-matrix with per-column ranges."""
        from torchwright.graph import Embedding

        emb = Embedding(vocab=["a", "b", "c"])
        ab = emb.affine_bound
        assert ab.n_cols > 0, "Embedding must have non-degenerate bound"
        assert ab.d_output == emb.d_output
        assert emb.node_id in ab.columns
        assert emb.node_id in ab.input_ranges
        assert torch.equal(ab.A_lo, torch.eye(emb.d_output, dtype=torch.float64))
        intervals = ab.to_interval()
        t = emb.table.to(torch.float64)
        for i in range(emb.d_output):
            assert intervals[i].lo == pytest.approx(t[:, i].min().item())
            assert intervals[i].hi == pytest.approx(t[:, i].max().item())

    def test_embedding_per_column_no_wider_than_global(self):
        """Per-column ranges must never be wider than global min/max."""
        from torchwright.graph import Embedding

        emb = Embedding(vocab=["a", "b", "c"])
        ab = emb.affine_bound
        intervals = ab.to_interval()
        t = emb.table.to(torch.float64)
        global_lo = float(t.min().item())
        global_hi = float(t.max().item())
        for iv in intervals:
            assert iv.lo >= global_lo - 1e-10
            assert iv.hi <= global_hi + 1e-10


class TestAttnRule:
    def test_attn_propagates_value_range(self):
        with fresh_graph_session():
            from torchwright.graph import Attn

            # Plain bounded input standing in for the query/key source; the
            # test only checks Attn propagates value range from value_in.
            pe = InputNode(9, name="pe", value_range=(-1.0, 1.0))
            value = LiteralValue(torch.tensor([2.0, 3.0]))
            attn = Attn(
                query_in=pe,
                key_in=pe,
                value_in=value,
                query_matrix=torch.eye(9, 2),
                key_matrix=torch.eye(9, 2),
                value_matrix=torch.eye(2),
                output_matrix=torch.eye(2),
            )
            r = attn.affine_bound.to_scalar_range()
            assert r.lo <= 2.0 + 1e-5
            assert r.hi >= 3.0 - 1e-5


# --- NaN safety ---------------------------------------------------------------


class TestNanSafety:
    """Unbounded InputNode ranges must not produce NaN in bounds."""

    def test_unbounded_relu(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(float("-inf"), float("inf")))
            from torchwright.graph import ReLU

            r = ReLU(x)
            iv = r.affine_bound.to_interval()[0]
            assert not math.isnan(iv.lo)
            assert not math.isnan(iv.hi)
            assert iv.lo == 0.0
            assert iv.hi == float("inf")

    def test_unbounded_linear(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(float("-inf"), float("inf")))
            from torchwright.graph import Linear

            W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            lin = Linear(x, W)
            for iv in lin.affine_bound.to_interval():
                assert not math.isnan(iv.lo)
                assert not math.isnan(iv.hi)

    def test_unbounded_relu_then_linear(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(float("-inf"), float("inf")))
            from torchwright.graph import Linear, ReLU

            r = ReLU(x)
            lin = Linear(r, torch.tensor([[2.0]]))
            iv = lin.affine_bound.to_interval()[0]
            assert not math.isnan(iv.lo)
            assert not math.isnan(iv.hi)

    def test_half_bounded_relu(self):
        """ReLU with finite upper, infinite lower."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(float("-inf"), 5.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            iv = r.affine_bound.to_interval()[0]
            assert not math.isnan(iv.lo)
            assert not math.isnan(iv.hi)
            assert iv.lo == 0.0
            assert iv.hi == pytest.approx(5.0)


# --- ReLU chord tightness ----------------------------------------------------


class TestReluChord:
    """Continuous chord lower bound tightens cancellation through ReLU."""

    def test_chord_tighter_than_zero(self):
        """ReLU(x) + (-ReLU(x)) with chord should be tighter than [-4, 4]."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-2.0, 4.0))
            from torchwright.graph import Add, Linear, ReLU

            r = ReLU(x)
            neg_r = Linear(r, torch.tensor([[-1.0]]))
            cancel = Add(r, neg_r)
            iv = cancel.affine_bound.to_interval()[0]
            # With chord alpha=h/(h-l)=4/6=2/3: interval is [-4/3, 4/3]
            assert iv.lo == pytest.approx(-4.0 / 3.0, abs=1e-10)
            assert iv.hi == pytest.approx(4.0 / 3.0, abs=1e-10)

    def test_chord_soundness(self):
        """Actual ReLU values must remain within chord-derived bounds."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-3.0, 5.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            intervals = r.affine_bound.to_interval()
            for _ in range(200):
                xv = torch.FloatTensor(1).uniform_(-3.0, 5.0)
                yv = torch.clamp(xv, min=0.0).item()
                assert yv >= intervals[0].lo - 1e-5
                assert yv <= intervals[0].hi + 1e-5

    def test_chord_lower_bound_negative(self):
        """Chord lower bound is negative for straddling (looser per-component, tighter correlation)."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-2.0, 4.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            iv = r.affine_bound.to_interval()[0]
            # alpha = 4/6, lower = alpha * (-2) = -4/3
            assert iv.lo == pytest.approx(-4.0 / 3.0, abs=1e-10)
            assert iv.hi == pytest.approx(4.0)


# --- General claim channel: degenerate box ---------------------------------


class TestClaimDegenerate:
    """A finite claim on a non-leaf collapses its bound to the
    claim-intersected constant box (the general channel)."""

    def test_degenerate_tight_range(self):
        """Claim on a Linear gives per-component intersected intervals."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-10.0, 10.0))
            from torchwright.graph import Linear
            from torchwright.graph.asserts import assert_in_range

            scaled = Linear(x, torch.tensor([[2.0]]))
            asserted = assert_in_range(scaled, -5.0, 5.0)
            iv = asserted.affine_bound.to_interval()[0]
            assert iv.lo == pytest.approx(-5.0)
            assert iv.hi == pytest.approx(5.0)

    def test_degenerate_zero_coefficients(self):
        """Claim on a non-leaf produces zero A matrices."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-10.0, 10.0))
            from torchwright.graph import Linear
            from torchwright.graph.asserts import assert_in_range

            scaled = Linear(x, torch.tensor([[2.0]]))
            asserted = assert_in_range(scaled, -5.0, 5.0)
            ab = asserted.affine_bound
            assert ab.n_cols == 0

    def test_value_type_reflects_claimed_range(self):
        """The claimed node's value_type.value_range reflects the claim."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-10.0, 10.0))
            from torchwright.graph import Linear
            from torchwright.graph.asserts import assert_in_range

            scaled = Linear(x, torch.tensor([[2.0]]))
            asserted = assert_in_range(scaled, -3.0, 7.0)
            r = asserted.value_type.value_range
            assert r.lo == pytest.approx(-3.0)
            assert r.hi == pytest.approx(7.0)


class TestToIntervalFpCrossing:
    """``to_interval`` snaps a sub-tolerance lower>upper crossing — fp
    accumulation noise on a point-valued component — instead of raising; a gross
    crossing still raises. Regression: a token embedding column at exactly 0.5
    had its lower bound eval land ~1 ULP above 0.5, tripping the strict ``Range``
    check during ``value_type`` propagation through a ``select``."""

    @staticmethod
    def _point_bound(lo_b: float, hi_b: float) -> AffineBound:
        # x-free (n_cols=0) component with lower offset lo_b, upper offset hi_b.
        return AffineBound(
            A_lo=torch.zeros(1, 0, dtype=torch.float64),
            A_hi=torch.zeros(1, 0, dtype=torch.float64),
            b_lo=torch.tensor([lo_b], dtype=torch.float64),
            b_hi=torch.tensor([hi_b], dtype=torch.float64),
            columns={},
            input_ranges={},
        )

    def test_sub_tolerance_crossing_snaps(self):
        ab = self._point_bound(0.5 + 7.45e-9, 0.5)
        intervals = ab.to_interval()
        assert len(intervals) == 1
        assert intervals[0].lo == intervals[0].hi == 0.5
        # to_scalar_range goes through the same path.
        r = ab.to_scalar_range()
        assert r.lo == pytest.approx(0.5) and r.hi == pytest.approx(0.5)

    def test_gross_crossing_still_raises(self):
        ab = self._point_bound(1.0, 0.5)
        with pytest.raises(ValueError):
            ab.to_interval()


# --- Gated / swish FFN rule ------------------------------------------------


def _gated_ffn(x, gate_proj, up_proj, out_proj, activation="swish"):
    from torchwright.graph.ffn import FFN

    n_lanes = gate_proj.shape[0]
    d_output = out_proj.shape[1]
    return FFN(
        x,
        gate_proj=gate_proj,
        gate_bias=torch.zeros(n_lanes),
        out_proj=out_proj,
        out_bias=torch.zeros(d_output),
        up_proj=up_proj,
        up_bias=torch.zeros(n_lanes),
        activation=activation,
    )


def _ffn_forward_f64(blk, x64):
    """The FFN lane math evaluated in float64 from the node's own weights."""
    gate = x64 @ blk.gate_proj.double().t() + blk.gate_bias.double()
    if blk.activation == "relu":
        act = torch.clamp(gate, min=0.0)
    else:
        act = gate * torch.sigmoid(gate)
    if blk.up_proj is not None:
        act = act * (x64 @ blk.up_proj.double().t() + blk.up_bias.double())
    return act @ blk.out_proj.double() + blk.out_bias.double()


def _pointwise_slack(blk, lo, hi, n_samples=20_000, seed=0):
    """Min over samples of (f - lower) and (upper - f); both must be >= 0."""
    ab = blk.affine_bound
    g = torch.Generator().manual_seed(seed)
    lo64 = torch.as_tensor(lo, dtype=torch.float64)
    hi64 = torch.as_tensor(hi, dtype=torch.float64)
    x = lo64 + (hi64 - lo64) * torch.rand(
        n_samples, ab.n_cols, dtype=torch.float64, generator=g
    )
    y = _ffn_forward_f64(blk, x)
    lower = x @ ab.A_lo.T + ab.b_lo
    upper = x @ ab.A_hi.T + ab.b_hi
    return min((y - lower).min().item(), (upper - y).min().item())


class TestSwishSandwich:
    def test_relu_sandwich_holds_on_dense_grid(self):
        from torchwright.graph.affine_rules import _SWISH_SANDWICH_C

        z = torch.linspace(-80.0, 80.0, 2_000_001, dtype=torch.float64)
        sw = z * torch.sigmoid(z)
        relu = torch.clamp(z, min=0.0)
        assert (sw - relu).max().item() <= 0.0
        assert ((relu - _SWISH_SANDWICH_C) - sw).max().item() <= 0.0

    def test_swish_interval_exact(self):
        from torchwright.graph.affine_rules import _swish_interval

        # Interval containing the interior minimum.
        lo, hi = _swish_interval(-5.0, 5.0)
        assert lo == pytest.approx(-0.278465, abs=1e-6)
        assert hi == pytest.approx(5.0 / (1.0 + math.exp(-5.0)))
        # Entirely left of the argmin: swish is decreasing there.
        lo, hi = _swish_interval(-10.0, -5.0)
        assert lo == pytest.approx(-5.0 / (1.0 + math.exp(5.0)))
        assert hi == pytest.approx(-10.0 / (1.0 + math.exp(10.0)))
        # Infinite endpoints stay sound and NaN-free.
        lo, hi = _swish_interval(float("-inf"), float("inf"))
        assert lo == pytest.approx(-0.278465, abs=1e-6) and math.isinf(hi)


class TestGatedFFNRule:
    def test_random_gated_swish_pointwise_soundness(self):
        for seed in range(5):
            with fresh_graph_session():
                g = torch.Generator().manual_seed(100 + seed)
                d_in, n_lanes, d_out = 5, 7, 3
                scale = 1.0 + 4.0 * torch.rand(1, generator=g).item()
                x = InputNode(d_in, name="x", value_range=(-4.0, 4.0))
                blk = _gated_ffn(
                    x,
                    gate_proj=torch.randn(n_lanes, d_in, generator=g) * scale,
                    up_proj=torch.randn(n_lanes, d_in, generator=g),
                    out_proj=torch.randn(n_lanes, d_out, generator=g),
                )
                slack = _pointwise_slack(blk, -4.0, 4.0, seed=seed)
                assert slack >= -1e-9, f"seed {seed}: bound violated by {-slack}"

    def test_gated_relu_pointwise_soundness(self):
        with fresh_graph_session():
            g = torch.Generator().manual_seed(7)
            x = InputNode(4, name="x", value_range=(-3.0, 3.0))
            blk = _gated_ffn(
                x,
                gate_proj=torch.randn(6, 4, generator=g) * 2.0,
                up_proj=torch.randn(6, 4, generator=g),
                out_proj=torch.randn(6, 2, generator=g),
                activation="relu",
            )
            assert _pointwise_slack(blk, -3.0, 3.0) >= -1e-9

    def test_degenerate_swish_pointwise_soundness(self):
        from torchwright.graph.ffn import FFN

        with fresh_graph_session():
            g = torch.Generator().manual_seed(11)
            x = InputNode(4, name="x", value_range=(-3.0, 3.0))
            blk = FFN(
                x,
                gate_proj=torch.randn(6, 4, generator=g) * 2.0,
                gate_bias=torch.randn(6, generator=g),
                out_proj=torch.randn(6, 2, generator=g),
                out_bias=torch.randn(2, generator=g),
                activation="swish",
            )
            assert _pointwise_slack(blk, -3.0, 3.0) >= -1e-9

    def test_gated_swish_value_type_finite(self):
        """The RMSNorm energy certification requirement: a gated swish FFN
        over bounded inputs must expose a finite value range."""
        with fresh_graph_session():
            g = torch.Generator().manual_seed(13)
            x = InputNode(5, name="x", value_range=(-10.0, 10.0))
            blk = _gated_ffn(
                x,
                gate_proj=torch.randn(8, 5, generator=g),
                up_proj=torch.randn(8, 5, generator=g),
                out_proj=torch.randn(8, 3, generator=g),
            )
            r = blk.value_type.value_range
            assert math.isfinite(r.lo) and math.isfinite(r.hi)

    def test_unbounded_input_degenerates_without_crash(self):
        """An unbounded gated lane must fall back to an unbounded row —
        constructed and concretized NaN-free (to_interval asserts on NaN),
        never a crash at graph build."""
        with fresh_graph_session():
            g = torch.Generator().manual_seed(17)
            x = InputNode(3, name="x", value_range=(float("-inf"), float("inf")))
            blk = _gated_ffn(
                x,
                gate_proj=torch.randn(4, 3, generator=g),
                up_proj=torch.randn(4, 3, generator=g),
                out_proj=torch.randn(4, 2, generator=g),
            )
            intervals = blk.affine_bound.to_interval()
            assert all(math.isinf(r.lo) and math.isinf(r.hi) for r in intervals)


class TestGatedFFNTightness:
    """The ops_plain_english.md constructions: the rule must stay within a
    small factor of the true output range (regression-pinned from the spike:
    multiply 1.06x, select 1.10x, cond_gate 1.05x)."""

    def test_multiply_construction(self):
        # multiply(a, b) = swish(a)*b + swish(-a)*(-b), a, b in [-10, 10];
        # exact output a*b in [-100, 100].
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-10.0, 10.0))
            blk = _gated_ffn(
                x,
                gate_proj=torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
                up_proj=torch.tensor([[0.0, 1.0], [0.0, -1.0]]),
                out_proj=torch.tensor([[1.0], [1.0]]),
            )
            r = blk.affine_bound.to_interval()[0]
            assert r.lo <= -100.0 <= 100.0 <= r.hi  # contains the true range
            assert r.lo >= -110.0 and r.hi <= 110.0  # within 1.10x
            assert _pointwise_slack(blk, -10.0, 10.0) >= -1e-9

    def test_select_construction(self):
        # select(cond, a, b) = swish(s*cond)/s * a + swish(-s*cond)/s * b,
        # s = 12; output within ~[-10, 10] and reaches ~±9.99 at cond = ±1.
        s = 12.0
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            ab = InputNode(2, name="ab", value_range=(-10.0, 10.0))
            from torchwright.graph import Concatenate

            inp = Concatenate([cond, ab])
            blk = _gated_ffn(
                inp,
                gate_proj=torch.tensor([[s, 0.0, 0.0], [-s, 0.0, 0.0]]),
                up_proj=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                out_proj=torch.tensor([[1.0 / s], [1.0 / s]]),
            )
            r = blk.affine_bound.to_interval()[0]
            assert r.lo <= -9.9 and r.hi >= 9.9
            assert r.lo >= -11.5 and r.hi <= 11.5

    def test_cond_gate_construction(self):
        # cond_gate(cond, inp) = swish(s*cond)/s * inp: single gated lane.
        s = 12.0
        with fresh_graph_session():
            cond = InputNode(1, name="cond", value_range=(-1.0, 1.0))
            val = InputNode(1, name="val", value_range=(-10.0, 10.0))
            from torchwright.graph import Concatenate

            inp = Concatenate([cond, val])
            blk = _gated_ffn(
                inp,
                gate_proj=torch.tensor([[s, 0.0]]),
                up_proj=torch.tensor([[0.0, 1.0]]),
                out_proj=torch.tensor([[1.0 / s]]),
            )
            r = blk.affine_bound.to_interval()[0]
            assert r.lo <= -9.9 and r.hi >= 9.9
            assert r.lo >= -11.0 and r.hi <= 11.0
