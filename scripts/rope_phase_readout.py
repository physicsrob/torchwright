"""RETIRED (2026-06-27) — reference only, NOT on the build path.

This prototype assumed the compiled softmax is piecewise-linear-approximated
(it modeled a "softmax-weight noise" budget). It is NOT — compiled attention
uses an EXACT fp32 softmax (SDPA MATH backend, `components/attn.py`), so the
readout weight is ~fp32-exact and this two-head-atan2-with-un-sigmoid design is
unnecessary. The live recency readout is the OCTANT TWO-HEAD scheme (direct-read
weights + steep-channel select + offset chain, no un-sigmoid, no atan2):
`scripts/rope_recency_replay.py`, `docs/rope_port_plan.md` §3 bucket 2 / R12.

Prototype: read the BOS-relative rotary phase out as a residual value (R12).

Context (RoPE port plan, docs/rope_port_plan.md, §3 bucket 2, R12/R13):
  Recency needs a globally MONOTONE signal in absolute position t. The candidate
  is the BOS-relative rotary phase on the fastest non-wrapping plane (one turn
  over the rollout). R13 set the correctness bar: the readout must recover the
  phase to within ~3e-3 rad (the winner-vs-runner-up separation at the measured
  min gap ~60). This script asks: CAN we read that phase out that accurately?

The mechanism we must respect (this is why the readout is non-trivial):
  RoPE rotates Q/K, NOT V. So the phase t*theta does NOT appear as a value we
  can copy from BOS -- it appears in the SCORE between the current token (pos t)
  and BOS (pos 0): on a plane, score = M*cos(t*theta - psi). To get it into the
  residual stream we read it as an ATTENTION WEIGHT (a 2-key softmax: BOS vs a
  position-independent reference), which is a SIGMOID of the score, then
  un-squash it. A single cosine is non-monotone and 2-to-1 ambiguous, so we use
  TWO heads (cos and sin) and atan2 to recover a monotone angle.

What this prototype VALIDATES:
  1. The two-head (cos,sin) -> un-sigmoid -> atan2 readout recovers the phase
     with ~uniform accuracy across the whole rollout (no end-of-range collapse).
  2. The attention-WEIGHT noise budget: how accurately the softmax weight must
     be read for the recovered angle to stay under ~3e-3 rad. (This is the bar
     the compiler's softmax/PL ops must hit -- compare to docs/op_noise_data.json.)
  3. Why a SINGLE cosine fails (resolution collapses near t=0 and t=N).

What it ASSUMES (still abstracted):
  - The 2-key softmax is modeled as w = sigmoid(g * score); the exact head
    construction (BOS key, reference key, gain g) is what Phase-1b builds. The
    un-sigmoid and atan2 are modeled as exact math + injected weight noise; the
    real PL `logit`/`atan2` ops add their own (separately measurable) error.

Run:  python scripts/rope_phase_readout.py        (CPU, numpy only)
"""

from __future__ import annotations

import math

import numpy as np


def _phase(t: np.ndarray, n: int, margin: float) -> tuple[np.ndarray, float]:
    """Monotone BOS-relative phase over [0, n), centered with headroom.

    Maps t in [0, n) to phase in (-pi+margin, pi-margin] so atan2 is monotone
    with no wrap seam inside the range. Returns (phase, theta_per_pos).
    """
    theta = (2.0 * math.pi - 2.0 * margin) / (n - 1)
    phi0 = -math.pi + margin - theta * 0.5
    return theta * (t + 0.5) + phi0, theta


def read_two_head(
    phase: np.ndarray,
    g: float,
    sigma_w: float,
    rng: np.random.Generator,
    k_planes: int = 1,
) -> np.ndarray:
    """Two-head (cos,sin) readout through a sigmoid attention weight + noise.

    With ``k_planes`` > 1, average K independent (cos,sin) reads of the SAME
    angle (redundant readout heads); angle noise falls ~sqrt(K).
    """

    def channel(signal: np.ndarray) -> np.ndarray:  # signal in [-1, 1]
        w = 1.0 / (1.0 + np.exp(-g * signal))  # 2-key softmax weight
        w = np.clip(w + rng.normal(0.0, sigma_w, w.shape), 1e-9, 1 - 1e-9)
        return np.log(w / (1.0 - w)) / g  # un-sigmoid -> ~signal

    cos_p, sin_p = np.cos(phase), np.sin(phase)
    angles = [np.arctan2(channel(sin_p), channel(cos_p)) for _ in range(k_planes)]
    cx = np.mean([np.cos(a) for a in angles], axis=0)  # average as unit vectors
    sx = np.mean([np.sin(a) for a in angles], axis=0)
    return np.arctan2(sx, cx)


def max_angle_error(
    n: int,
    g: float,
    sigma_w: float,
    *,
    seed: int,
    margin: float = 0.1,
    k_planes: int = 1,
) -> float:
    """Worst-case recovered-angle error (rad) over every position in [0, n)."""
    t = np.arange(n, dtype=np.float64)
    phase, _ = _phase(t, n, margin)
    rng = np.random.default_rng(seed)
    err = read_two_head(phase, g, sigma_w, rng, k_planes) - phase
    err = (err + math.pi) % (2 * math.pi) - math.pi  # wrap to (-pi, pi]
    return float(np.abs(err).max())


def weight_noise_budget(
    n: int, g: float, target_rad: float, *, seed: int, k_planes: int = 1
) -> float:
    """Largest softmax-weight noise sigma keeping max angle error < target."""
    lo, hi = 0.0, 0.5
    if max_angle_error(n, g, 1e-9, seed=seed, k_planes=k_planes) > target_rad:
        return 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if max_angle_error(n, g, mid, seed=seed, k_planes=k_planes) < target_rad:
            lo = mid
        else:
            hi = mid
    return lo


def single_cosine_end_resolution(n: int, margin: float = 0.1) -> None:
    """Show why ONE cosine head is not enough: resolution collapses at the ends.

    A single monotone cosine (half-turn) ranks recency by cos(phase). The
    winner-vs-runner-up separation in cos-space is ~|d cos/dt| * gap, which goes
    to zero as phase -> 0 or pi (the ends of the rollout). Report the
    cos-separation for a gap-60 pair near the start vs the middle.
    """
    theta_half = (math.pi - 2 * margin) / (n - 1)  # half turn, monotone
    gap = 60
    for label, t0 in [
        ("start t=300", 300),
        ("mid t=32000", n // 2),
        ("end t=63000", 63000),
    ]:
        p0 = theta_half * t0
        p1 = theta_half * (t0 + gap)
        dcos = abs(math.cos(p0) - math.cos(p1))
        print(
            f"    single-cosine sep at {label:>14}: {dcos:.2e} "
            f"(uniform two-head sep would be ~{theta_half * gap * 2:.2e})"
        )


def main() -> None:
    n = 64000
    # R13 budget: winner-vs-runner-up separation at min gap ~60 on the one-turn
    # plane; the readout angle error must stay under half of it.
    _, theta = _phase(np.array([0.0]), n, 0.1)
    min_gap = 60
    target = 0.5 * theta * min_gap
    print(
        f"Reading the BOS-relative phase over [0, {n}); recency budget "
        f"= {target:.3e} rad (gap {min_gap} @ theta={theta:.3e}).\n"
    )

    print(f"{'gain g':>7} {'max_err@noiseless':>18} {'weight-noise budget':>20}")
    print("-" * 50)
    for g in [0.5, 1.0, 2.0, 4.0, 8.0]:
        noiseless = max_angle_error(n, g, 1e-9, seed=0)
        budget = weight_noise_budget(n, g, target, seed=0)
        print(f"{g:>7.1f} {noiseless:>18.3e} {budget:>20.4e}")

    print("\nMargin lever: average K redundant readout planes (g=2):")
    print(f"  {'K planes':>9} {'weight-noise budget':>20}")
    for k in [1, 2, 4, 8]:
        b = weight_noise_budget(n, 2.0, target, seed=0, k_planes=k)
        print(f"  {k:>9} {b:>20.4e}")

    print("\nEstimated compiler softmax-weight noise (from op_noise_data.json):")
    print("  exp rel-error ~1.9e-4 (256-BP grid) + reciprocal; a 2-key softmax")
    print("  weight near 0.5 lands ~1.5e-4 abs. Budget at g=2, K=1 is ~2e-4 ->")
    print("  BORDERLINE (~1.3x). Levers: more exp breakpoints; K>1 planes.")

    print("\nWhy two heads (single-cosine resolution collapses at the ends):")
    single_cosine_end_resolution(n)

    print("\nReadings:")
    print(" - two-head atan2 recovers the phase with ~uniform accuracy; the")
    print("   noiseless error is just fp/atan2 round-off.")
    print(" - 'weight-noise budget' = how accurately the softmax weight must be")
    print("   read to keep recency correct. Compare to the compiler's softmax/PL")
    print("   weight noise (docs/op_noise_data.json) -- that comparison decides")
    print("   whether R12 is feasible. The decode is NOT the floor; the readout is.")


if __name__ == "__main__":
    main()
