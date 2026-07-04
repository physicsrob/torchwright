"""Unit tests for multiply_2d.

Verifies that the 2D piecewise-linear multiplication primitive produces
correct results across sign combinations, grid points, interpolated
values, custom breakpoints, unsigned ranges, output clamping, and
compiled transformer agreement.
"""

import time
import tracemalloc

import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.ops.relu.arithmetic_ops import multiply_2d
from torchwright.ops.inout_nodes import create_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_mul(node, a_val, b_val):
    """Evaluate a multiply_2d node at (a, b)."""
    return node.compute(
        n_pos=1,
        input_values={
            "a": torch.tensor([[a_val]]),
            "b": torch.tensor([[b_val]]),
        },
    ).item()


def _build_multiply_2d(**kwargs):
    """Build a multiply_2d graph from two scalar inputs named 'a' and 'b'."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    return multiply_2d(a, b, **kwargs)


# ---------------------------------------------------------------------------
# Grid-point accuracy
# ---------------------------------------------------------------------------


def test_multiply_2d_grid_points():
    """Exact at integer multiples of step."""
    node = _build_multiply_2d(max_abs1=5.0, max_abs2=5.0, step1=1.0, step2=1.0)

    for a in range(-5, 6):
        for b in range(-5, 6):
            expected = float(a * b)
            result = _eval_mul(node, float(a), float(b))
            assert abs(result - expected) < 0.01, f"{a}*{b} = {expected}, got {result}"


def test_multiply_2d_fine_step():
    """Finer step gives tighter accuracy at grid points."""
    node = _build_multiply_2d(max_abs1=3.0, max_abs2=3.0, step1=0.5, step2=0.5)

    for a in [-3.0, -1.5, 0.0, 1.5, 3.0]:
        for b in [-3.0, -1.5, 0.0, 1.5, 3.0]:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < 0.01, f"{a}*{b} = {expected}, got {result}"


# ---------------------------------------------------------------------------
# Sign combinations and zeros
# ---------------------------------------------------------------------------


def test_multiply_2d_sign_combinations():
    """All four quadrants plus zero produce correct results."""
    node = _build_multiply_2d(max_abs1=10.0, max_abs2=10.0, step1=1.0, step2=1.0)

    cases = [
        (3.0, 4.0, 12.0),
        (-3.0, 4.0, -12.0),
        (3.0, -4.0, -12.0),
        (-3.0, -4.0, 12.0),
        (0.0, 5.0, 0.0),
        (5.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    for a, b, expected in cases:
        result = _eval_mul(node, a, b)
        assert abs(result - expected) < 0.5, f"{a}*{b} = {expected}, got {result}"


# ---------------------------------------------------------------------------
# Interpolation accuracy
# ---------------------------------------------------------------------------


def test_multiply_2d_interpolation():
    """Between grid points, error is bounded by h1*h2/4."""
    step1, step2 = 1.0, 1.0
    node = _build_multiply_2d(max_abs1=10.0, max_abs2=10.0, step1=step1, step2=step2)
    max_error = step1 * step2 / 4 + 0.05  # small margin for float noise

    # Test at half-step offsets (worst case for interpolation)
    for a in [0.5, 1.5, -2.5, 4.5]:
        for b in [0.5, -1.5, 3.5]:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < max_error, (
                f"{a}*{b} = {expected}, got {result}, "
                f"error {abs(result - expected):.4f} > {max_error:.4f}"
            )


# ---------------------------------------------------------------------------
# Custom breakpoints
# ---------------------------------------------------------------------------


def test_multiply_2d_custom_breakpoints():
    """Non-uniform breakpoints (DOOM _DIFF_BP style) work correctly."""
    diff_bp = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]
    trig_bp = [-1.0, -0.5, 0.0, 0.5, 1.0]
    node = _build_multiply_2d(
        max_abs1=10.0,
        max_abs2=1.0,
        breakpoints1=diff_bp,
        breakpoints2=trig_bp,
    )

    # Exact at grid points
    for a in diff_bp:
        for b in trig_bp:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < 0.01, f"{a}*{b} = {expected}, got {result}"


# ---------------------------------------------------------------------------
# Positive-input ranges
# ---------------------------------------------------------------------------


def test_multiply_2d_unsigned():
    """Positive-only second input (e.g., inv_range)."""
    node = _build_multiply_2d(
        max_abs1=10.0,
        max_abs2=2.0,
        step1=1.0,
        step2=0.25,
    )

    # Positive second input
    for a in [-5.0, 0.0, 3.0, 10.0]:
        for b in [0.0, 0.5, 1.0, 2.0]:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < 0.3, f"{a}*{b} = {expected}, got {result}"


def test_multiply_2d_both_unsigned():
    """Both inputs non-negative."""
    node = _build_multiply_2d(
        max_abs1=5.0,
        max_abs2=5.0,
        step1=1.0,
        step2=1.0,
    )

    for a in [0.0, 1.0, 3.0, 5.0]:
        for b in [0.0, 1.0, 3.0, 5.0]:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < 0.01, f"{a}*{b} = {expected}, got {result}"


# ---------------------------------------------------------------------------
# Probe (compilation) test
# ---------------------------------------------------------------------------


def test_multiply_2d_probe():
    """Compiled transformer matches the oracle."""
    node = _build_multiply_2d(max_abs1=5.0, max_abs2=5.0, step1=1.0, step2=1.0)

    # Sweep of integer grid points
    n_pos = 11
    a_vals = torch.tensor([[float(i - 5)] for i in range(n_pos)])
    b_vals = torch.tensor([[float(5 - i)] for i in range(n_pos)])
    inputs = {"a": a_vals, "b": b_vals}

    # Oracle check
    cache = reference_eval(node, inputs, n_pos)
    oracle = cache[node].flatten()
    expected = torch.tensor(
        [a_vals[i, 0].item() * b_vals[i, 0].item() for i in range(n_pos)]
    )
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


# ---------------------------------------------------------------------------
# Construction cost (analytic bilinear fast path)
# ---------------------------------------------------------------------------


def test_multiply_2d_build_is_linear_in_breakpoints():
    """257 breakpoints/axis builds in O(n) time and memory, not O(n⁴).

    Regression for the build-time OOM: the generic ``piecewise_linear_2d``
    path solved a least-squares system over an O(n²)-row design matrix and
    a ``full_matrices=True`` SVD whose discarded left-singular matrix is
    ``(n², n²)`` — ~35 GB of float64 at n=257, which OOM'd the build.  A
    product is exactly bilinear, so ``multiply_2d`` now builds the
    quarter-square interpolant analytically (O(n) neurons, no solve).  This
    test pins that property: the build must finish quickly and allocate a
    trivial amount of memory at a breakpoint count that previously OOM'd.
    """
    a = create_input("a", 1)
    b = create_input("b", 1)

    max_abs = 1500.0
    step = (2 * max_abs) / 256.0  # 257 breakpoints per axis

    tracemalloc.start()
    t0 = time.perf_counter()
    node = multiply_2d(a, b, max_abs1=max_abs, max_abs2=max_abs, step1=step, step2=step)
    elapsed = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # O(n) construction: well under a second and a few MB of Python-traced
    # allocation.  The pre-fix O(n⁴) path took tens of seconds and OOM'd a
    # 30 GB machine; generous bounds here still catch any regression to it.
    assert elapsed < 5.0, f"build took {elapsed:.2f}s — expected O(n), well under 1s"
    assert peak < 200e6, f"build allocated {peak / 1e6:.1f} MB — expected a few MB"

    # Sanity: the node still computes the product accurately at the scale
    # the breakpoints cover.
    for x, y in [(1000.0, 1000.0), (-1234.0, 567.0), (1500.0, -1500.0)]:
        result = node.compute(
            n_pos=1,
            input_values={
                "a": torch.tensor([[x]]),
                "b": torch.tensor([[y]]),
            },
        ).item()
        expected = x * y
        # Bilinear interpolation error bound is step²/4 at this grid.
        assert (
            abs(result - expected) < step * step / 4 + 1.0
        ), f"{x}*{y} = {expected}, got {result}"


def test_multiply_2d_chunks_across_d_max():
    """The analytic product bank chunks correctly when it exceeds d_max.

    The quarter-square fast path emits ~2·(2n−2) neurons; a small d_max
    forces them across multiple linear_relu_linear sublayers summed
    together.  Grid points must stay exact regardless of chunk boundary.
    """
    a = create_input("a", 1)
    b = create_input("b", 1)
    node = multiply_2d(a, b, max_abs1=5.0, max_abs2=5.0, step1=1.0, step2=1.0, d_max=4)
    for x in range(-5, 6):
        for y in range(-5, 6):
            result = node.compute(
                n_pos=1,
                input_values={
                    "a": torch.tensor([[float(x)]]),
                    "b": torch.tensor([[float(y)]]),
                },
            ).item()
            assert abs(result - x * y) < 0.01, f"{x}*{y} != {result}"
