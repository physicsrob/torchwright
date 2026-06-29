"""RoPE recency-resolution sweep.

Question: under a pure-RoPE construction with base=1e6, can a phase-aligned
sum of rotary planes rank attention keys by recency (relative separation
Delta = query_pos - key_pos) such that:

  (M) score(Delta) is STRICTLY decreasing in Delta over [0, R]
      -- otherwise a farther key outscores a nearer one (recency inversion),
  (G) the per-step gap g(Delta) = score(Delta) - score(Delta+1) stays above
      the fp32 resolution floor at the logit magnitude actually used.

Key correction vs the additive-counter scheme: RoPE recency is BOUNDED
(|score| <= sum of plane amplitudes), it does NOT grow with sequence length.
So the content gate that must dominate it is ~O(#planes), not ~3e5. The fp32
floor is therefore set by a logit magnitude of order (sum of amplitudes), and
the decision quantity is the DIMENSIONLESS ratio g(Delta) / score(0).

We sweep d_head and a few plane-subset / apodization choices.
"""

import sys

import numpy as np

try:
    from torchwright.graph.rope import ROPE_BASE as _DEFAULT_BASE
except Exception:  # pragma: no cover - script may run without the package on path
    _DEFAULT_BASE = 500000.0

# Defaults to the live ROPE_BASE (5e5) so figures reconcile with the runtime
# grid; pass another base as the first CLI arg (e.g. the historical 1e6).
BASE = float(sys.argv[1]) if len(sys.argv) > 1 else float(_DEFAULT_BASE)
R = 65536  # full sequence range to test monotonicity / resolution across


def freqs(d_head, base=BASE):
    k = np.arange(d_head // 2)
    return base ** (-2.0 * k / d_head)  # theta_k, shape (d_head/2,)


def recency_score(deltas, theta, amp):
    # score(Delta) = sum_k amp_k * cos(Delta * theta_k), phase-aligned at 0.
    c = np.cos(np.outer(deltas, theta))  # (len(deltas), n_planes)
    return c @ amp


def analyze(label, theta, amp):
    deltas = np.arange(0, R + 1)
    s = recency_score(deltas, theta, amp)
    s0 = s[0]  # == sum(amp), the global max (all cos=1)
    gap = s[:-1] - s[1:]  # g(Delta) for Delta=0..R-1
    # Monotonicity: is the most-recent always strictly highest? Equivalent to
    # gap > 0 everywhere. Report first inversion (gap <= 0).
    inv = np.where(gap <= 0)[0]
    first_inv = int(inv[0]) if inv.size else None
    # Normalized gap (dimensionless: gap relative to max score).
    ng = gap / s0
    # fp32 relative floor for a logit dot-product accumulation of ~d_head terms.
    fp32_floor = (d_head_for[label]) ** 0.5 * 2.0**-23

    # Main-lobe vs sidelobe: the REAL recency question is whether a recent
    # match (small Delta) outranks ALL older matches. The most-recent write at
    # Delta_r wins iff score(Delta_r) > max score over every larger Delta.
    running_max_from_right = np.maximum.accumulate(s[::-1])[::-1]  # max of s[d:]
    wins = s[:-1] > running_max_from_right[1:]
    W = int(np.argmin(wins)) if not wins.all() else R
    sidelobe_peak = float(s[first_inv:].max()) if first_inv else float("nan")

    print(f"\n=== {label} ===")
    print(f"  n_planes={len(theta)}  theta_min={theta.min():.3g}  score(0)={s0:.3f}")
    print(
        f"  first inversion at Delta={first_inv}   sidelobe peak (Delta>=lobe)"
        f"={sidelobe_peak:.3f} ({100 * sidelobe_peak / s0:.1f}% of peak)"
    )
    print(
        f"  >>> GUARANTEED recency window W = {W}  "
        f"(most-recent write within W positions always wins)"
    )


d_head_for = {}

for d_head in [64, 128, 256]:
    theta = freqs(d_head)

    # (a) all planes, uniform amplitude
    lab = f"d_head={d_head}, all planes, uniform amp"
    d_head_for[lab] = d_head
    analyze(lab, theta, np.ones_like(theta))

    # (b) only "recency-suitable" planes: drop the slowest (content) planes and
    #     the very fastest (alias < a few positions). Keep a mid band.
    keep = (theta < 0.3) & (theta > np.pi / R)  # monotone-over-R-ish, not too fast
    lab = f"d_head={d_head}, mid-band planes [pi/R, 0.3], uniform amp"
    d_head_for[lab] = d_head
    if keep.sum() >= 2:
        analyze(lab, theta[keep], np.ones(keep.sum()))

    # (c) apodized: Hann taper over the kept band (suppress sidelobes/bumps)
    lab = f"d_head={d_head}, mid-band planes, Hann taper"
    d_head_for[lab] = d_head
    if keep.sum() >= 2:
        n = keep.sum()
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
        analyze(lab, theta[keep], hann + 1e-3)
