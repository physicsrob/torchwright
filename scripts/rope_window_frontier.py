"""Show how the plane cutoff trades window width W for near-Delta resolution.

Local-recency lobe frontier (docs/rope_port_plan.md Phase 6). W = the largest
distance at which a nearer key always outscores every farther one, for the
Hann-tapered mid-band lobe ``s(Δ) = Σ_p amp_p cos(Δ·θ_p)``. Lower
``theta_max`` drops fast planes and widens W at the cost of coarser near-Δ
resolution.

BASE defaults to the live ``ROPE_BASE`` (5e5) so the printed W reconciles with
the runtime grid; pass another base as the first CLI arg (e.g. the historical
1e6 the earlier measurements used).
"""

import sys
from collections.abc import Callable

import numpy as np

try:
    from torchwright.graph.rope import ROPE_BASE as _DEFAULT_BASE
except ImportError:  # pragma: no cover - script may run without the package on path
    _DEFAULT_BASE = 500000.0

BASE = float(sys.argv[1]) if len(sys.argv) > 1 else float(_DEFAULT_BASE)
R = 65536
MIN_PLANES = 2  # need at least 2 kept planes for a usable Hann-tapered lobe


def freqs(d: int, base: float = BASE) -> np.ndarray:
    k = np.arange(d // 2)
    return base ** (-2.0 * k / d)


def usable_window(
    theta: np.ndarray, amp: np.ndarray
) -> tuple[int, float, Callable[[int], float]]:
    d = np.arange(R + 1)
    s = np.cos(np.outer(d, theta)) @ amp
    rmax = np.maximum.accumulate(s[::-1])[::-1]
    wins = s[:-1] > rmax[1:]
    W = int(np.argmin(wins)) if not wins.all() else R

    # smallest-separation resolution: normalized per-step gap near a target sep
    def ngap(D: int) -> float:
        return (s[D] - s[D + 1]) / s[0]

    return W, s[0], ngap


print(
    f"Frontier (base={BASE:g}): lower theta_max widens W but kills "
    f"near-Delta resolution."
)
print("Local recency (Phase 6) targets W >= ~100; the natural band gives ~415.\n")
print(
    f"{'theta_max':>10} {'n_planes':>8} {'W':>7} {'peak':>7} {'ngap@1':>10} "
    f"{'ngap@60':>10} {'ngap@100':>10}"
)
d_head = 256
theta_all = freqs(d_head)
for tmax in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
    keep = (theta_all < tmax) & (theta_all > np.pi / R)
    th = theta_all[keep]
    n = len(th)
    if n < MIN_PLANES:
        continue
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1)) + 1e-3
    W, peak, ngap = usable_window(th, hann)
    print(
        f"{tmax:>10.3g} {n:>8} {W:>7} {peak:>7.2f} {ngap(1):>10.2e} "
        f"{ngap(60):>10.2e} {ngap(100):>10.2e}"
    )
