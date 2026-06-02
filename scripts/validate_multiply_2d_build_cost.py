"""Validate multiply_2d's build cost and accuracy at high breakpoint counts.

Background.  ``multiply_2d`` used to lower to ``piecewise_linear_2d``'s
generic least-squares solve, which built an O(n²)-row vertex design matrix
and ran ``numpy.linalg.svd(..., full_matrices=True)`` whose left-singular
matrix is ``(n², n²)``.  At ~257 breakpoints per axis that discarded U
matrix is ~35 GB of float64 — a single graph build OOM'd a 30 GB machine.

A product ``x·y`` is exactly bilinear (rank-1), so ``multiply_2d`` now
builds its piecewise-linear interpolant analytically via the quarter-square
identity ``x·y = ((x+y)² − (x−y)²)/4`` — O(n) neurons, no global solve.

This script:

* (i)  builds a 257×257 grid over a realistic operand range and confirms it
       constructs in well under a second using a trivial amount of memory;
* (ii) checks the max abs error of the product against exact ``x·y`` over a
       dense sample, at magnitudes up to ~1500·1500, stays inside the
       bilinear ``step²/4`` interpolation bound.

Run with the workspace venv::

    /data/torchdoom/.venv/bin/python -m scripts.validate_multiply_2d_build_cost
"""

from __future__ import annotations

import time
import tracemalloc

import torch

from torchwright.ops.arithmetic_ops import multiply_2d
from torchwright.ops.inout_nodes import create_input


def main() -> None:
    max_abs = 1500.0
    n_bp = 257
    step = (2 * max_abs) / (n_bp - 1)  # 257 breakpoints per axis over [-1500, 1500]

    # (i) Construction cost.
    a = create_input("a", 1)
    b = create_input("b", 1)
    tracemalloc.start()
    t0 = time.perf_counter()
    node = multiply_2d(
        a, b, max_abs1=max_abs, max_abs2=max_abs, step1=step, step2=step
    )
    elapsed = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"breakpoints per axis : {n_bp}")
    print(f"operand range        : [-{max_abs}, {max_abs}], step {step:.4f}")
    print(f"build time           : {elapsed * 1000:.1f} ms")
    print(f"build alloc (traced) : {peak / 1e6:.2f} MB")
    print()
    print(
        "old generic path at this n built a (n², n²) = "
        f"({n_bp ** 2}, {n_bp ** 2}) float64 SVD U matrix "
        f"≈ {n_bp ** 4 * 8 / 1e9:.1f} GB → OOM"
    )
    print()

    assert elapsed < 1.0, f"build took {elapsed:.2f}s — expected well under a second"
    assert peak < 200e6, f"build allocated {peak / 1e6:.1f} MB — expected a few MB"

    # (ii) Accuracy vs exact product over a dense sample.
    torch.manual_seed(0)
    n_samples = 200_000
    xs = (torch.rand(n_samples) * 2 - 1) * max_abs
    ys = (torch.rand(n_samples) * 2 - 1) * max_abs
    worst = 0.0
    worst_at = None
    # compute() takes one position at a time here; batch through n_pos.
    inputs = {"a": xs.unsqueeze(1), "b": ys.unsqueeze(1)}
    out = node.compute(n_pos=n_samples, input_values=inputs).flatten()
    err = (out - xs * ys).abs()
    worst = err.max().item()
    worst_at = (xs[err.argmax()].item(), ys[err.argmax()].item())

    bound = step * step / 4
    print(f"max abs error        : {worst:.4f} at {worst_at}")
    print(f"bilinear step²/4 bnd : {bound:.4f}")
    assert worst < bound + 1.0, f"error {worst} exceeds bilinear bound {bound}"

    print()
    print("PASS: builds in O(n) time/memory and stays within the interpolation bound.")


if __name__ == "__main__":
    main()
