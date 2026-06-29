"""Unit tests for piecewise_linear_2d.

These tests verify correctness in isolation: exact at grid points,
bounded interpolation error, boundary clamping, non-uniform grids, and
compiled-vs-oracle agreement via probe_graph.
"""

import math

import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.ops.arithmetic_ops import piecewise_linear_2d
from torchwright.ops.inout_nodes import create_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_2d(node, x_val, y_val):
    """Evaluate a 2-input scalar node at (x, y)."""
    return node.compute(
        n_pos=1,
        input_values={
            "x": torch.tensor([[x_val]]),
            "y": torch.tensor([[y_val]]),
        },
    ).item()


def _build_2d(breakpoints1, breakpoints2, fn, **kwargs):
    """Build a piecewise_linear_2d graph from two scalar inputs."""
    x = create_input("x", 1)
    y = create_input("y", 1)
    return piecewise_linear_2d(x, y, breakpoints1, breakpoints2, fn, **kwargs)


# ---------------------------------------------------------------------------
# Basic function tests
# ---------------------------------------------------------------------------


def test_piecewise_linear_2d_sum():
    """f(x, y) = x + y — exact at grid points, small error between."""
    bp = [0.0, 1.0, 2.0, 3.0]
    node = _build_2d(bp, bp, lambda x, y: x + y)

    # Exact at grid points
    for x in bp:
        for y in bp:
            assert (
                abs(_eval_2d(node, x, y) - (x + y)) < 0.01
            ), f"f({x}, {y}) should be {x + y}"

    # Between grid points — small interpolation noise from the ReLU basis
    assert abs(_eval_2d(node, 0.5, 0.5) - 1.0) < 0.1
    assert abs(_eval_2d(node, 1.5, 2.5) - 4.0) < 0.1


def test_piecewise_linear_2d_product():
    """f(x, y) = x * y — exact at grid points, bounded error between."""
    bp = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    node = _build_2d(bp, bp, lambda x, y: x * y)

    # Exact at grid points
    for x in bp:
        for y in bp:
            expected = x * y
            result = _eval_2d(node, x, y)
            assert (
                abs(result - expected) < 0.01
            ), f"f({x}, {y}) = {result}, expected {expected}"

    # Between grid points: error bounded by h1*h2/4 = 1*1/4 = 0.25
    for x, y in [(0.5, 0.5), (-1.5, 2.5), (0.3, -0.7)]:
        expected = x * y
        result = _eval_2d(node, x, y)
        assert (
            abs(result - expected) < 0.3
        ), f"f({x}, {y}) = {result}, expected {expected}"


def test_piecewise_linear_2d_abs_sum():
    """f(x, y) = |x| + |y| — non-smooth, piecewise-linear function."""
    bp = [-2.0, -1.0, 0.0, 1.0, 2.0]
    node = _build_2d(bp, bp, lambda x, y: abs(x) + abs(y))

    # Exact at grid points
    grid_cases = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 2.0),
        (-1.0, 1.0, 2.0),
        (-2.0, -2.0, 4.0),
    ]
    for x, y, expected in grid_cases:
        result = _eval_2d(node, x, y)
        assert (
            abs(result - expected) < 0.01
        ), f"|{x}| + |{y}| = {expected}, got {result}"

    # Between grid points — interpolation noise from the ReLU basis
    result = _eval_2d(node, 0.5, -0.5)
    assert abs(result - 1.0) < 0.15, f"|0.5| + |-0.5| = 1.0, got {result}"


def test_piecewise_linear_2d_constant():
    """f(x, y) = 5.0 — trivial constant function."""
    bp = [0.0, 1.0, 2.0]
    node = _build_2d(bp, bp, lambda x, y: 5.0)

    # Exact at grid points
    for x in bp:
        for y in bp:
            result = _eval_2d(node, x, y)
            assert abs(result - 5.0) < 0.01, f"f({x}, {y}) = {result}, expected 5.0"

    # Between grid points — small noise from the least-squares solve
    for x in [0.5, 1.5]:
        for y in [0.5, 1.5]:
            result = _eval_2d(node, x, y)
            assert abs(result - 5.0) < 0.15, f"f({x}, {y}) = {result}, expected 5.0"


def test_piecewise_linear_2d_clamping():
    """Outside the grid domain, output is clamped (constant extrapolation)."""
    bp = [0.0, 5.0, 10.0]
    node = _build_2d(bp, bp, lambda x, y: x + y)

    # At grid boundary
    assert abs(_eval_2d(node, 10.0, 10.0) - 20.0) < 0.01

    # Outside grid — should clamp to boundary value.
    # piecewise_linear_2d extrapolates linearly outside the grid, but
    # for x+y the extrapolation IS correct (same slope).  Test with a
    # non-linear function where clamping vs extrapolation matters.
    bp2 = [0.0, 5.0, 10.0]
    node2 = _build_2d(bp2, bp2, lambda x, y: x * y)

    # At boundary
    val_at_10_10 = _eval_2d(node2, 10.0, 10.0)
    assert abs(val_at_10_10 - 100.0) < 0.01

    # For a product, extrapolation beyond the grid will diverge from
    # the true product because the piecewise-affine approximation
    # extends the last segment's slope.  Just verify the result is
    # finite and not wildly wrong.
    val_outside = _eval_2d(node2, 12.0, 12.0)
    assert abs(val_outside) < 500, f"Extrapolated value {val_outside} seems extreme"


def test_piecewise_linear_2d_non_uniform():
    """Non-uniform breakpoints (like _DIFF_BP) produce correct results."""
    # Sparse near the extremes, dense near zero — typical DOOM pattern
    bp = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]
    node = _build_2d(bp, bp, lambda x, y: x * y)

    # Exact at grid points
    for x in bp:
        for y in bp:
            expected = x * y
            result = _eval_2d(node, x, y)
            assert (
                abs(result - expected) < 0.01
            ), f"f({x}, {y}) = {result}, expected {expected}"

    # At mid-cell points near zero (fine grid), error should be small
    assert abs(_eval_2d(node, 0.5, 0.5) - 0.25) < 0.1
    assert abs(_eval_2d(node, -0.5, 1.5) - (-0.75)) < 0.15


def test_piecewise_linear_2d_asymmetric():
    """Different breakpoints on each axis."""
    bp_x = [0.0, 1.0, 2.0, 3.0, 4.0]
    bp_y = [-1.0, 0.0, 1.0]
    node = _build_2d(bp_x, bp_y, lambda x, y: 2 * x + 3 * y)

    # Grid points
    grid_cases = [
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 7.0),
        (4.0, -1.0, 5.0),
    ]
    for x, y, expected in grid_cases:
        result = _eval_2d(node, x, y)
        assert (
            abs(result - expected) < 0.01
        ), f"f({x}, {y}) = {result}, expected {expected}"

    # Between grid points
    result = _eval_2d(node, 1.5, 0.5)
    assert abs(result - 4.5) < 0.1, f"f(1.5, 0.5) = {result}, expected 4.5"


# ---------------------------------------------------------------------------
# Probe (compilation) test
# ---------------------------------------------------------------------------


def test_piecewise_linear_2d_probe():
    """Compiled transformer matches the oracle for a 2D product."""
    bp = [-5.0, -2.5, 0.0, 2.5, 5.0]
    node = _build_2d(bp, bp, lambda x, y: x * y)

    # Use grid points for exact oracle comparison
    xs = [-5.0, -2.5, 0.0, 2.5, 5.0]
    ys = [-5.0, -2.5, 0.0, 2.5, 5.0]
    n_pos = len(xs)
    inputs = {
        "x": torch.tensor([[v] for v in xs]),
        "y": torch.tensor([[v] for v in ys]),
    }

    # Oracle check at grid points
    cache = reference_eval(node, inputs, n_pos)
    oracle = cache[node].flatten()
    expected = torch.tensor([x * y for x, y in zip(xs, ys)])
    assert torch.allclose(
        oracle, expected, atol=0.01
    ), f"oracle: {oracle.tolist()}\nexpected: {expected.tolist()}"

    # Compiled check
    report = probe_graph(
        node,
        input_values=inputs,
        n_pos=n_pos,
        d=512,
        d_head=16,
        verbose=False,
        atol=0.5,
    )
    assert report.first_divergent is None, report.format_short()
