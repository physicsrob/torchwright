"""Verify the float32-cancellation model for piecewise-linear `log`.

Compares observed worst-case absolute log error against the predicted
floor `(x_max / x_min) · 2⁻²³` across input ranges spanning different
dynamic ranges.

If the model is right, error scales linearly with `x_max / x_min`,
independent of `n_breakpoints`.
"""

from __future__ import annotations

import math

import torch

from torchwright.ops.relu.arithmetic_ops import log
from torchwright.ops.inout_nodes import create_input


def _measure(x_min: float, x_max: float, n_bp: int, n_samples: int = 4096) -> dict:
    x = create_input("x", 1)
    f = log(x, min_value=x_min, max_value=x_max, n_breakpoints=n_bp)

    # Sample log-uniformly so we hit values across the whole range,
    # including the high-x tail where cancellation peaks.
    log_lo = math.log(x_min)
    log_hi = math.log(x_max)
    gen = torch.Generator().manual_seed(0)
    u = torch.rand(n_samples, generator=gen)
    xs = torch.exp(log_lo + u * (log_hi - log_lo)).unsqueeze(1)
    # Suppress log's own value-range assert so we can *measure* the
    # drift instead of having compute() raise on it.
    from torchwright.graph.node import suppress_checks

    with suppress_checks():
        ys = f.compute(n_pos=n_samples, input_values={"x": xs})

    expected = torch.log(xs.squeeze(1))
    err = (ys.squeeze(1) - expected).abs()
    worst_idx = int(err.argmax().item())

    predicted = (x_max / x_min) * 2**-23
    return {
        "x_min": x_min,
        "x_max": x_max,
        "dynamic_range": x_max / x_min,
        "n_bp": n_bp,
        "max_abs_err": err.max().item(),
        "p99_abs_err": err.quantile(0.99).item(),
        "worst_x": xs[worst_idx].item(),
        "worst_log_actual": ys[worst_idx].item(),
        "worst_log_expected": expected[worst_idx].item(),
        "predicted_floor": predicted,
        "ratio_observed_to_predicted": err.max().item() / predicted,
    }


def main() -> None:
    cases = [
        # User's failing case
        (0.01, 30000.0, 256, "user's case (failing)"),
        (0.01, 30000.0, 1024, "user's case @ 1024 BPs"),
        # Prediction 1: should pass cleanly
        (1.0, 30000.0, 256, "prediction 1: bp0=1, expect clean"),
        # Prediction 2: borderline
        (0.01, 1000.0, 256, "prediction 2: borderline"),
        # Prediction 3: should fail badly
        (0.001, 30000.0, 256, "prediction 3: 7 decades, expect bad"),
        # Sanity: tight range
        (1.0, 100.0, 256, "sanity: 2 decades"),
    ]

    print(
        f"{'x_min':>8} {'x_max':>10} {'dyn_rng':>10} {'n_bp':>5} "
        f"{'max_err':>10} {'predicted':>10} {'obs/pred':>8}  label"
    )
    for x_min, x_max, n_bp, label in cases:
        r = _measure(x_min, x_max, n_bp)
        print(
            f"{r['x_min']:>8g} {r['x_max']:>10g} {r['dynamic_range']:>10.1e} "
            f"{r['n_bp']:>5d} {r['max_abs_err']:>10.4f} "
            f"{r['predicted_floor']:>10.4f} "
            f"{r['ratio_observed_to_predicted']:>8.2f}  {label}"
        )


if __name__ == "__main__":
    main()
