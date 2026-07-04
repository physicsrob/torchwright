"""Private pure-math helpers shared by both op libraries.

Nothing here builds a graph node — these are sizing and inversion
formulas used by the relu and swiglu implementations of the same ops
(lookup-table grids in ``map_select``, the BOS-weight inversion in
``global_recency``, the reciprocal grid in ``marker_count``).  That is
what makes them machine-neutral: they contain no activation choice.
"""

import math
import numbers

from torchwright.graph import RopeConfig
from torchwright.graph.rope import rope_inv_freq

# ---------------------------------------------------------------------------
# map_select: lookup-table grid sizing
# ---------------------------------------------------------------------------


def _lookup_axis_scale(index_scale, axis: int, n_axes: int = 2) -> float:
    if isinstance(index_scale, numbers.Real):
        scale = float(index_scale)
    else:
        if len(index_scale) != n_axes:
            raise ValueError(
                f"index_scale must be a scalar or length-{n_axes} tuple, "
                f"got {index_scale!r}"
            )
        scale = float(index_scale[axis])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"index_scale values must be finite and > 0, got {scale}")
    return scale


def _lookup_numeric_slack(max_abs: float, sharpness: float, n_steps: int) -> float:
    # Per-element slack for the row-vector / staircase output-range *guards*
    # (assert_matches_value_type), not the correctness path. The guard must
    # have margin above accumulated fp32 noise in the wide PWL AND above GPU
    # cross-test FP variation (cuBLAS algorithm selection / TF32), which the
    # noise notes peg at ~1e-5..1e-6 — an order of magnitude above the fp32
    # single-run unit. At the 16x128x128 target (rows = A*B = 2048) the hidden
    # PWL activations reach ~20k*sharpness, so reduced-precision matmul on A100
    # can push the guarded value to ~max_abs*sharpness*rows*1e-5; a 1e-6 budget
    # tripped intermittently in the full sharded suite (passes in isolation and
    # under fp32). Using 1e-5 gives the guard headroom for GPU variation without
    # loosening real correctness (the caller's value-match test stays tight).
    return max(1e-3, max_abs * sharpness * max(n_steps, 1) * 1e-5)


# ---------------------------------------------------------------------------
# marker_count: reciprocal grid sizing
# ---------------------------------------------------------------------------

# Per-segment relative interpolation error of 1/x on a geometric grid with
# ratio r is ~ (r-1)^2 / 8.  We want gap+1 within +/-0.5 at the bound, i.e.
# relative error < 0.5/(max_gap+1).  Solving for the breakpoint count with a
# safety factor keeps the inversion comfortably inside that budget.
_RECIP_REL_SAFETY = 16.0


# ---------------------------------------------------------------------------
# global_recency: BOS-weight inversion (w → m)
# ---------------------------------------------------------------------------

# How many log-uniform breakpoints to use for the PWL inverse w → m.
# Validation (scripts/rope_global_recency_validate.py) shows 1024 achieves
# max error ~0.013 positions, well within the 0.5 rounding threshold.
# 1024 neurons fit in a single MLP sublayer (d_max=1024 default).
_N_BPS = 1024


def _theta_slow(rope: RopeConfig) -> float:
    """Frequency of the slowest **rotated** plane for this rope config.

    The runtime rotation runs over the rotary front ``d_rot`` (``apply_rope`` uses
    width ``d_rot``, so per-plane frequencies are ``base^(-2p/d_rot)``), so the
    slowest rotated plane is index ``d_rot/2 − 1`` with frequency
    ``rope_inv_freq(d_rot, base)[-1]``.  Under full rotary ``d_rot == d_head`` and
    this is the slowest plane of the ``d_head`` grid — byte-identical to the
    pre-partial form.  This is the plane the BOS-weight feature must ride so its
    ``cos(m·θ_slow)`` attenuation matches the PWL inversion table."""
    return float(rope_inv_freq(rope.d_rot, rope.base)[-1])


def _w_of_m(m: float, max_len: int, theta: float) -> float:
    """True BOS softmax weight at position m (exact math)."""
    if m <= 0.0:
        return 1.0
    cos_m = math.cos(m * theta)
    eff = math.pow(max_len, cos_m)  # MAX_LEN^cos(m·θ) — the actual attention score
    if eff <= 0.0:
        return 0.0
    return eff / (eff + m)


def _bisect_m(w_target: float, max_len: int, theta: float) -> float:
    """Invert _w_of_m: find m ∈ [0, max_len] with _w_of_m(m) ≈ w_target."""
    if w_target >= 1.0:
        return 0.0
    w_at_max = _w_of_m(max_len, max_len, theta)
    if w_target <= w_at_max:
        return float(max_len)
    lo, hi = 0.0, float(max_len)
    for _ in range(64):  # converges to <1e-15 in 64 steps
        mid = 0.5 * (lo + hi)
        # w is decreasing in m: w_of_m(mid) > w_target ⟹ true m is larger
        if _w_of_m(mid, max_len, theta) > w_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
