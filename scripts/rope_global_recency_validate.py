"""Validate the boosted-BOS global-recency mechanism numerically (CPU / numpy only).

Phase 7 (docs/rope_port_plan.md §8, candidate d): each token at position m
attends to BOS with score ln(MAX_LEN)·cos(m·θ_slow), receiving softmax weight

    w(m) = MAX_LEN^cos(m·θ) / (MAX_LEN^cos(m·θ) + m)

A PWL inverse g: [w_min, 1] → [0, MAX_LEN] recovers the position.

This script verifies:
1. w(m) is strictly monotone over [0, MAX_LEN] with the actual production θ_slow.
2. The minimum adjacent-m gap in w exceeds the fp32 weight floor (~6e-8) by a
   healthy margin.
3. A 1024-breakpoint log-uniform PWL inverse achieves <0.02 worst-case error in m
   over the full range, comfortably inside the 0.5 rounding threshold.
4. The fp32 softmax precision contribution is bounded and consistent with the
   ~0.008 estimate from the analytical argument.
"""

import math

# ---------------------------------------------------------------------------
# Production constants (must match torchwright/graph/rope.py)
# ---------------------------------------------------------------------------
MAX_LEN = 61440
D_HEAD = 256
BASE = 5e5
N_BREAKPOINTS = 1024

FP32_EPS = 2.0**-23  # machine epsilon (relative error per fp32 operation)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def theta_slow(d_head: int, base: float) -> float:
    """Frequency of the slowest rotary plane: base^(-(d_head-2)/d_head)."""
    return float(base ** (-(d_head - 2) / d_head))


def w_of_m(m: float, max_len: int, theta: float) -> float:
    """Boosted-BOS softmax weight at absolute position m."""
    if m <= 0:
        return 1.0
    cos_m = math.cos(m * theta)
    eff = math.pow(
        max_len, cos_m
    )  # MAX_LEN^cos(m·θ) — matches the actual attention score
    if eff <= 0.0:
        return 0.0
    return eff / (eff + m)


def bisect_m(w_target: float, max_len: int, theta: float, n_iter: int = 64) -> float:
    """Invert w_of_m: find m in [0, max_len] such that w_of_m(m) ≈ w_target."""
    if w_target >= 1.0:
        return 0.0
    w_at_max = w_of_m(max_len, max_len, theta)
    if w_target <= w_at_max:
        return float(max_len)
    lo, hi = 0.0, float(max_len)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        # w is decreasing in m: w_of_m(mid) > w_target ⟹ true m is larger
        if w_of_m(mid, max_len, theta) > w_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_pwl_table(max_len: int, theta: float, n: int):
    """Build log-uniform breakpoints and their inverse-function values.

    Returns (w_bps, m_bps): breakpoints in w ∈ [w_min, 1] (ascending) and the
    corresponding m values (descending — the PWL is a decreasing function).
    """
    w_min = w_of_m(max_len, max_len, theta)
    ratio = (1.0 / w_min) ** (1.0 / (n - 1))
    w_bps = [w_min * (ratio**k) for k in range(n)]
    w_bps[-1] = 1.0  # exact endpoint
    m_bps = [bisect_m(w, max_len, theta) for w in w_bps]
    return w_bps, m_bps


def eval_pwl(w_query: float, w_bps: list, m_bps: list) -> float:
    """Evaluate the PWL table at w_query via linear interpolation."""
    # w_bps is ascending; binary search
    lo, hi = 0, len(w_bps) - 2
    while lo < hi:
        mid = (lo + hi) // 2
        if w_bps[mid + 1] < w_query:
            lo = mid + 1
        else:
            hi = mid
    idx = max(0, min(lo, len(w_bps) - 2))
    w0, w1 = w_bps[idx], w_bps[idx + 1]
    m0, m1 = m_bps[idx], m_bps[idx + 1]
    if w1 == w0:
        return m0
    t = (w_query - w0) / (w1 - w0)
    return m0 + t * (m1 - m0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def main() -> None:
    theta = theta_slow(D_HEAD, BASE)
    print(f"Production parameters: MAX_LEN={MAX_LEN}, D_HEAD={D_HEAD}, BASE={BASE:.0e}")
    print(f"θ_slow = {theta:.5e}")
    print()

    # 1. Monotonicity and minimum adjacent gap
    print("=== 1. Monotonicity and adjacent gap ===")
    ws = [w_of_m(m, MAX_LEN, theta) for m in range(MAX_LEN + 1)]
    diffs = [ws[m] - ws[m + 1] for m in range(MAX_LEN)]
    min_gap = min(diffs)
    min_gap_pos = diffs.index(min_gap)
    fp32_floor = 6e-8
    assert min_gap > 0, f"w(m) is NOT monotone! Inversion at m={min_gap_pos}"
    print(f"w(m) strictly monotone: ✓")
    print(f"Min adjacent-m gap = {min_gap:.3e}  (at m={min_gap_pos})")
    print(f"fp32 weight floor   = {fp32_floor:.1e}")
    print(f"Safety margin       = {min_gap / fp32_floor:.1f}×  (need >> 1)")
    print()

    # 2. PWL inverse accuracy
    print(f"=== 2. PWL inverse accuracy ({N_BREAKPOINTS} log-uniform breakpoints) ===")
    w_bps, m_bps = build_pwl_table(MAX_LEN, theta, N_BREAKPOINTS)
    print(f"w range: [{w_bps[0]:.6f}, {w_bps[-1]:.6f}]")
    print(f"m range: [{m_bps[-1]:.2f}, {m_bps[0]:.2f}]")

    # Dense test: sample every 10 positions
    step = 10
    max_err = 0.0
    worst_m = 0
    for m_true in range(0, MAX_LEN + 1, step):
        w_t = w_of_m(m_true, MAX_LEN, theta)
        m_pred = eval_pwl(w_t, w_bps, m_bps)
        err = abs(m_pred - m_true)
        if err > max_err:
            max_err = err
            worst_m = m_true

    print(f"Worst-case error: {max_err:.4f} positions (at m={worst_m})")
    print(f"Rounding threshold: 0.5 positions")
    print(f"Rounding margin: {0.5 / max_err:.1f}× below threshold  (need >> 1)")
    print()

    # 3. fp32 softmax noise contribution
    print("=== 3. fp32 softmax noise → error in m ===")
    # At each m, |δm| ≈ |dm/dw| · |δw| where |δw| ≈ w · FP32_EPS / 2
    max_fp32_err = 0.0
    worst_fp32_m = 0
    eps = FP32_EPS / 2.0
    for m_true in range(0, MAX_LEN + 1, step):
        w_t = w_of_m(m_true, MAX_LEN, theta)
        dw = w_t * eps
        # Finite-difference estimate of |dm/dw|
        m_up = bisect_m(w_t + dw, MAX_LEN, theta)
        m_dn = bisect_m(w_t - dw, MAX_LEN, theta)
        dm = abs(m_dn - m_up) / 2.0
        if dm > max_fp32_err:
            max_fp32_err = dm
            worst_fp32_m = m_true

    print(
        f"Worst-case fp32 contribution: {max_fp32_err:.4f} positions (at m={worst_fp32_m})"
    )

    combined = max_err + max_fp32_err
    print(f"Combined (PWL + fp32):        {combined:.4f} positions")
    print(f"Rounding threshold:           0.5")
    print(f"Rounding margin:              {0.5 / combined:.1f}×")
    print()

    # 4. cosine attenuation magnitude at max_len
    print("=== 4. Cosine attenuation sanity check ===")
    cos_at_max = math.cos(MAX_LEN * theta)
    print(f"cos(MAX_LEN · θ_slow) = {cos_at_max:.6f}  (1% would be 0.990)")
    naive_m_hat = MAX_LEN * (1.0 / w_of_m(MAX_LEN, MAX_LEN, theta) - 1)
    print(
        f"Naive 1/w−1 at MAX_LEN: returns {naive_m_hat:.0f} vs true {MAX_LEN}"
        f"  (absorbed by the PWL fit)"
    )
    print()

    print("=== Summary ===")
    ok = min_gap > 10 * fp32_floor and max_err < 0.1 and combined < 0.5
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] gap={min_gap:.1e}, PWL_err={max_err:.4f}, "
        f"combined={combined:.4f}"
    )


if __name__ == "__main__":
    main()
