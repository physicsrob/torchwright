"""RETIRED (2026-06-27) — reference only, NOT on the build path.

No consumer needs an exact absolute integer: recency needs only a uniform-
resolution monotone ranking (the octant two-head readout, see
`scripts/rope_recency_replay.py` and `docs/rope_port_plan.md` §3 bucket 2),
and pixel-difference consumers need bounded LOCAL differences. This exact-
integer decode is kept only as a validated mechanism should a future consumer
ever need a true absolute index.

Prototype: recover an exact integer absolute position from BOS-relative
rotary phases, with a hierarchical (mixed-radix) decode and NO full-range
round.

Context (RoPE port plan, docs/rope_port_plan.md, R12):
  Position is recovered by reading each token's rotary phase RELATIVE TO A
  PINNED BOS token.  BOS sits at position 0, so the relative phase on a plane
  of frequency theta is exactly t*theta -- a direct, non-recurrent encoding of
  the absolute position t.  We must turn that phase back into the exact integer
  t for the consumers (pixel arithmetic, recency) over the full rollout range
  (~64k positions), inside the compiler's piecewise-linear op budget.

What this prototype ASSUMES (the open readout piece, R12(a) -- NOT validated here):
  - That we can read, per plane i, the BOS-relative phase as a (cos, sin) pair
    cos(t*theta_i), sin(t*theta_i).  In the real graph this comes out of an
    attention readout (the phase lives in the self-vs-BOS score, since RoPE
    rotates Q/K not V), and landing it as a clean residual value is unsolved.
    Here we generate it directly and inject angle noise to stand in for readout
    + PL + fp imperfection.

What this prototype VALIDATES (the decode, R12(b)):
  1. EXACTNESS: with integer positions and a mixed-radix phase ladder, the
     decode returns t EXACTLY over the whole range, with no full-range round.
  2. COST: it needs only ~ L * r breakpoints (L levels, radix r), i.e.
     O(L * N^(1/L)) -- e.g. radix-16/4-level decodes 0..65535 with ~64 knots,
     vs a single ~64k-knot round.  The only rounds are per-level "stitch"
     rounds over the small radix r (cheap), never over the full range.
  3. NOISE MARGIN: how much angle error the decode tolerates before any
     position in the range is misdecoded.  This tells us the budget the
     (still-open) readout has to hit -- the decode is not the bottleneck.

Run:  python scripts/rope_position_decode.py        (CPU, numpy only)
"""

from __future__ import annotations

import math
import numpy as np

TWO_PI = 2.0 * math.pi


def plane_periods(radix: int, n_levels: int) -> list[int]:
    """Period (in positions) of each level's rotary plane.

    Level l completes one full 2*pi cycle every ``radix**(l+1)`` positions, so
    its phase encodes ``t mod radix**(l+1)``.  The coarsest level
    (``radix**n_levels``) must cover the whole range without wrapping.
    """
    return [radix ** (l + 1) for l in range(n_levels)]


# Per-level phase offsets (in POSITION units), chosen to keep every in-range
# integer position away from the 0/2pi seam, where additive angle noise would
# wrap a measurement by a full period:
#   - Fine levels (which wrap every `radix` positions and self-correct at the
#     next stitch): a half-bin nudge suffices.
#   - The COARSEST level has nothing above it to correct a wrap, so its whole
#     working range [0, n_max) must sit in the middle of the circle. That needs
#     real headroom (period >= ~2*n_max) plus centering at phase pi.
FINE_BIAS = 0.5


def level_offsets(periods: list[int], n_max: int) -> np.ndarray:
    """Offset (in positions) per level; see module note above."""
    offs = np.full(len(periods), FINE_BIAS, dtype=np.float64)
    # Center [0, n_max) at phase pi on the coarsest level: position (n_max-1)/2
    # should map to period/2.
    offs[-1] = periods[-1] / 2.0 - (n_max - 1) / 2.0
    return offs


def make_phases(t: np.ndarray, periods: list[int], offsets: np.ndarray) -> np.ndarray:
    """BOS-relative rotary phase per level, as an angle in [0, 2*pi).

    phase_l(t) = ((t + off_l) * theta_l) mod 2*pi,  theta_l = 2*pi / period_l.
    Because BOS is at position 0, this relative phase IS t*theta_l (plus the
    seam-avoidance offset) -- the direct absolute-position encoding.  Returns
    shape (n_levels, len(t)).
    """
    angles = np.empty((len(periods), t.size), dtype=np.float64)
    for l, period in enumerate(periods):
        angles[l] = (TWO_PI * (t + offsets[l]) / period) % TWO_PI
    return angles


def decode(angles: np.ndarray, periods: list[int], offsets: np.ndarray) -> np.ndarray:
    """Hierarchical mixed-radix decode: phases -> exact integer position.

    Coarse-to-fine successive refinement.  The coarsest level (period >= range)
    seeds a rough estimate; each finer level l measures ``t mod period_l`` and
    snaps the running estimate onto the lattice it pins down.  The snap is the
    only rounding -- and it is over the small radix step, NOT the full range.
    """
    # measured y_l = (angle_l / 2pi) * period_l - off_l  ~=  t mod period_l
    measured = (angles / TWO_PI) * np.array(periods)[:, None] - offsets[:, None]

    # Coarsest level covers the whole range (period >= N), so t mod period == t.
    t_hat = measured[-1].copy()
    for l in range(len(periods) - 2, -1, -1):
        period = periods[l]
        # Snap t_hat onto { k*period + y_l } nearest to the coarse estimate.
        k = np.round((t_hat - measured[l]) / period)
        t_hat = k * period + measured[l]
    return np.round(t_hat).astype(np.int64)


def n_breakpoints(radix: int, n_levels: int) -> int:
    """Rough piecewise-linear knot budget for the decode.

    Per level: a phase->index map and a radix-wide stitch snap, each ~radix
    knots.  Total ~ 2 * radix * n_levels.  The point is it scales with
    radix*levels = O(L * N^(1/L)), never with N.
    """
    return 2 * radix * n_levels


def _require_headroom(periods: list[int], n_max: int) -> None:
    assert periods[-1] >= 2 * n_max, (
        f"coarsest period {periods[-1]} < 2*range {2 * n_max}: too little "
        f"headroom to center the working range off the seam; add a level"
    )


def verify_exact(radix: int, n_levels: int, n_max: int) -> bool:
    """Zero-noise exactness over every integer position in [0, n_max)."""
    periods = plane_periods(radix, n_levels)
    _require_headroom(periods, n_max)
    offsets = level_offsets(periods, n_max)
    t = np.arange(n_max, dtype=np.float64)
    got = decode(make_phases(t, periods, offsets), periods, offsets)
    return bool(np.array_equal(got, t.astype(np.int64)))


def noise_margin(radix: int, n_levels: int, n_max: int, *, seed: int) -> float:
    """Largest angle-noise sigma (radians) keeping EVERY position exact.

    Binary-searches sigma; at each sigma, adds Gaussian noise to every plane's
    angle and requires a perfect decode across the whole range.  Stands in for
    the readout/PL/fp error budget the decode can absorb.
    """
    periods = plane_periods(radix, n_levels)
    offsets = level_offsets(periods, n_max)
    t = np.arange(n_max, dtype=np.float64)
    clean = make_phases(t, periods, offsets)
    rng = np.random.default_rng(seed)

    def ok(sigma: float) -> bool:
        noisy = (clean + rng.normal(0.0, sigma, clean.shape)) % TWO_PI
        return bool(np.array_equal(decode(noisy, periods, offsets), t.astype(np.int64)))

    lo, hi = 0.0, math.pi  # pi/radix is the analytic ceiling; pi is a safe hi
    if not ok(1e-9):
        return 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def fixtures() -> None:
    """Self-check: hand-picked positions and full-range exactness."""
    radix, n_levels, n_max = 16, 5, 64000
    periods = plane_periods(radix, n_levels)
    offsets = level_offsets(periods, n_max)

    # Spot-check a few positions, including the fragile seam (t=0) and the top.
    for t_val in [0, 1, 255, 256, 4095, 4096, 42248, 63999]:
        t = np.array([t_val], dtype=np.float64)
        got = decode(make_phases(t, periods, offsets), periods, offsets)
        assert got[0] == t_val, f"decode({t_val}) = {got[0]}"

    # Full-range exactness (the real fixture).
    assert verify_exact(radix, n_levels, n_max), "full-range decode not exact"
    print("fixtures: OK (spot-checks + exact over [0, 64000))\n")


def main() -> None:
    fixtures()
    n_max = 64000  # full e1m1 rollout range (~42k frame, 64k cap)
    print(
        f"Decoding exact integer position over [0, {n_max}) "
        f"from BOS-relative rotary phases.\n"
    )
    print(
        f"{'radix':>6} {'levels':>7} {'planes':>7} {'~knots':>7} "
        f"{'exact?':>7} {'noise_margin(rad)':>18} {'pi/radix':>10}"
    )
    print("-" * 72)

    # Each config needs radix**levels >= 2*n_max (headroom to center the range).
    configs = [(64, 3), (32, 4), (16, 5), (8, 6), (4, 9)]
    for radix, n_levels in configs:
        periods = plane_periods(radix, n_levels)
        if periods[-1] < 2 * n_max:
            print(
                f"{radix:>6} {n_levels:>7}  -- coarsest period "
                f"{periods[-1]} < 2*{n_max}, skipped"
            )
            continue
        exact = verify_exact(radix, n_levels, n_max)
        margin = noise_margin(radix, n_levels, n_max, seed=0)
        knots = n_breakpoints(radix, n_levels)
        print(
            f"{radix:>6} {n_levels:>7} {n_levels:>7} {knots:>7} "
            f"{str(exact):>7} {margin:>18.4f} {math.pi/radix:>10.4f}"
        )

    print("\nReadings:")
    print(" - 'exact?' True  => decode reproduces t for EVERY position, no")
    print("   full-range round (only per-level radix-wide stitch snaps).")
    print(" - knots ~ 2*radix*levels: O(L*N^(1/L)), never O(N). Smaller radix")
    print("   is cheaper AND more noise-tolerant (margin ~ pi/radix).")
    print(" - noise_margin is the angle error the DECODE absorbs; the open")
    print("   readout (R12a) must land inside it. The decode is not the floor.")


if __name__ == "__main__":
    main()
