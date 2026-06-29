"""Unit tests for low_rank_2d.

The op approximates a 2-D function as a sum of ``K`` separable rank-1
terms (SVD truncation).  The contract:

- Rank-1 functions (e.g. ``x*y``) are reproduced exactly at K=1.
- For higher-rank functions, the error at every point is bounded by the
  first truncated singular value ``σ_{K+1}`` (a compile-time quantity).
- Works on uniform *and* non-uniform grids — the inner multiplications
  use bounded uniform grids where ``piecewise_linear_2d`` is well-posed.
"""

import math

import numpy as np
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.ops.arithmetic_ops import low_rank_2d
from torchwright.ops.inout_nodes import create_input


def _eval_2d(node, x_val, y_val):
    return node.compute(
        n_pos=1,
        input_values={
            "x": torch.tensor([[x_val]]),
            "y": torch.tensor([[y_val]]),
        },
    ).item()


def _build(bp_x, bp_y, fn, rank, **kwargs):
    x = create_input("x", 1)
    y = create_input("y", 1)
    return low_rank_2d(x, y, bp_x, bp_y, fn, rank=rank, **kwargs)


def test_low_rank_2d_product_rank1_exact():
    """``x*y`` has rank 1 exactly; K=1 must be exact at and between vertices."""
    bp = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    node = _build(bp, bp, lambda a, b: a * b, rank=1)

    # Near-exact at vertices (limited only by the inner multiply_2d's
    # own grid quantization — not by the SVD truncation, which is zero
    # for rank-1 functions).
    for x in bp:
        for y in bp:
            got = _eval_2d(node, x, y)
            assert abs(got - x * y) < 0.1, f"{x}*{y}: got {got}, expected {x * y}"

    # Exact (modulo inner multiply precision) at cell interiors too —
    # the decomposition is algebraically exact, only multiply_2d's own
    # grid introduces drift.  The inner multiply has step ≈ bound/10,
    # giving ~1% precision.
    for x, y in [(0.5, 0.5), (-1.5, 2.5), (2.7, -0.9)]:
        got = _eval_2d(node, x, y)
        assert abs(got - x * y) < 0.25, f"{x}*{y}: got {got}, expected {x * y}"


def test_low_rank_2d_product_non_uniform_rank1_exact():
    """Rank-1 exactness survives non-uniform breakpoints."""
    bp = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]
    node = _build(bp, bp, lambda a, b: a * b, rank=1)

    # Vertices exact.
    for x in bp:
        for y in bp:
            got = _eval_2d(node, x, y)
            # Absolute tolerance accommodates the inner multiply's grid
            # quantization; product magnitudes up to 100.
            assert abs(got - x * y) < 1.0, f"{x}*{y}: got {got}, expected {x * y}"


def test_low_rank_2d_constant_function():
    """Constant function: rank 1, single nonzero singular value."""
    bp = [0.0, 1.0, 2.0, 3.0]
    node = _build(bp, bp, lambda a, b: 5.0, rank=1)

    for x in [0.5, 1.5, 2.5, 0.0, 3.0]:
        for y in [0.5, 1.5, 2.5, 0.0, 3.0]:
            got = _eval_2d(node, x, y)
            assert abs(got - 5.0) < 0.5, f"f({x}, {y}) = {got}, expected 5.0"


def test_low_rank_2d_atan_non_uniform_K3():
    """``atan(x/y)`` on the TODO pathological grid — K=3 gets σ_4 ≈ 0.008."""
    bp_x = [
        -2.0,
        -1.5,
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        -0.1,
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
    ]
    bp_y = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

    node = _build(bp_x, bp_y, lambda x, y: math.atan(x / y), rank=3)

    # Predicted worst-cell error bound from the SVD spectrum.
    V = np.array([[math.atan(x / y) for y in bp_y] for x in bp_x])
    s = np.linalg.svd(V, compute_uv=False)
    sigma_bound = s[3]  # σ_{K+1} = σ_4
    # Loose factor of 10 covers the inner multiply's own grid drift on
    # top of the SVD truncation.
    tol = max(10 * sigma_bound, 0.1)

    for x in bp_x:
        for y in bp_y:
            got = _eval_2d(node, x, y)
            expected = math.atan(x / y)
            assert (
                abs(got - expected) < tol
            ), f"atan({x}/{y}): got {got}, expected {expected}"

    # The specific interior probe that was the TODO's failing case.
    got = _eval_2d(node, -1.25, 8.5)
    # Triangulated-PL value on this diagonal-midpoint (avg of corners)
    # is about -0.155; true atan is about -0.146; rank-3 low-rank should
    # be closer to true than to triangulated-PL, and certainly within
    # 0.1 of either.  Pre-fix buggy value was -1.04 — far outside this.
    assert abs(got - math.atan(-1.25 / 8.5)) < 0.1, (
        f"interior probe (-1.25, 8.5): got {got}, " f"oracle {math.atan(-1.25 / 8.5)}"
    )


def test_low_rank_2d_rank_clamped_to_grid_size():
    """rank > min(n1, n2) silently clamps, doesn't raise."""
    bp_x = [0.0, 1.0, 2.0]  # n1 = 3
    bp_y = [0.0, 1.0]  # n2 = 2
    # rank=10 should clamp to min(3, 2) = 2.
    node = _build(bp_x, bp_y, lambda a, b: a + b, rank=10)
    assert abs(_eval_2d(node, 1.0, 0.5) - 1.5) < 0.3


def test_low_rank_2d_sum_is_rank_2():
    """``x + y = x·1 + 1·y`` has rank 2.  K=2 reproduces it."""
    bp = [-2.0, -1.0, 0.0, 1.0, 2.0]
    node = _build(bp, bp, lambda a, b: a + b, rank=2)

    for x in bp:
        for y in bp:
            got = _eval_2d(node, x, y)
            assert abs(got - (x + y)) < 0.3, f"{x}+{y}: got {got}, expected {x + y}"


def test_low_rank_2d_compiled_product():
    """Compiled transformer matches oracle for the rank-1 product."""
    bp = [-5.0, -2.5, 0.0, 2.5, 5.0]
    x = create_input("x", 1)
    y = create_input("y", 1)
    node = low_rank_2d(x, y, bp, bp, lambda a, b: a * b, rank=1)

    xs = [-5.0, -2.5, 0.0, 2.5, 5.0]
    ys = [5.0, 2.5, 0.0, -2.5, -5.0]
    n_pos = len(xs)
    inputs = {
        "x": torch.tensor([[v] for v in xs]),
        "y": torch.tensor([[v] for v in ys]),
    }

    cache = reference_eval(node, inputs, n_pos)
    oracle = cache[node].flatten()
    expected = torch.tensor([a * b for a, b in zip(xs, ys)])
    assert torch.allclose(
        oracle, expected, atol=0.3
    ), f"oracle: {oracle.tolist()}\nexpected: {expected.tolist()}"

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
