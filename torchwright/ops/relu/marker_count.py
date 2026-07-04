"""Bucket-1 near-marker count — the bounded position-difference capability.

docs/rope_port_plan.md §3 (bucket 1) / Phase 1.  DOOM's pixel/row/prefix
arithmetic is all ``pos - a recent reference``: wall/flat pixel index =
``pos - span_start``, weapon/HUD row = ``sy_value + (pos - sy_pos)``.  The
difference (the *gap*) is bounded by a screen dimension (< ~350), and the
reference is a recent, graph-identified **marker** token.

Mechanism — a uniform-attention count, no position counter:
``attend_mean_where`` over the keys in the window ``[marker, now]``, reading a
value that is ``1`` at the single marker key and ``0`` at the others.  Uniform
attention assigns each of the ``gap+1`` window keys equal weight ``1/(gap+1)``,
so the mean is exactly ``1/(gap+1)``.  Inverting (``reciprocal``) gives the
window size ``gap+1``; subtract 1 for the gap.

This is RoPE-clean: ``attend_mean_where`` uses a literal query and
validity-driven keys, and under the end-state global rotation it rides the
**slowest plane** (``cos((i-j)·θ_slow) ≈ 1`` over the bounded window), so the
mean stays uniform to ~fp32.  Position enters only through the
``window_validity`` / ``marker_onehot`` features, which the consumer derives
from in-context marker tokens — never from an absolute counter.

**Resolution (gate b).** The mean values ``{1, 1/2, ..., 1/(N+1)}`` bunch near
0 for large gaps: ``1/350`` vs ``1/351`` differ by ~8e-6.  The softmax weight
is fp32-exact (~6e-8), so the *signal* resolves; the inversion must too.
``reciprocal`` uses geometric breakpoints (constant relative error), and we
size the breakpoint count so the relative error is well under ``0.5/(N+1)`` —
i.e. ``gap+1`` lands within ±0.5 of the integer out to the ``max_gap`` bound.
"""

import math

from torchwright.graph import Node, RopeConfig
from torchwright.graph.rope import rope_inv_freq
from torchwright.ops._math import _RECIP_REL_SAFETY
from torchwright.ops.arithmetic_ops import reciprocal
from torchwright.ops.linear import add_const
from torchwright.ops.attention_ops import attend_mean_where


def count_since_marker(
    rope: RopeConfig,
    window_validity: Node,
    marker_onehot: Node,
    *,
    max_gap: int,
) -> Node:
    """Integer gap ``pos - marker_pos`` via a uniform-attention count.

    Args:
        rope: the RoPE config.  ``attend_mean_where`` places the validity on the
            slowest plane, so the rotation is quasi-static
            (``cos((i-j)·θ_slow) ≈ 1``) over the bounded window and the mean
            stays uniform; ``rope.d_head`` / ``base`` size that plane.
        window_validity: length-1 boolean (+1 / -1).  +1 for keys in the window
            ``[marker, now]``, -1 outside.  The caller derives this from the
            marker token (e.g. "at or after the most recent marker").
        marker_onehot: length-1 value, ``1.0`` at the single marker key inside
            the window and ``0.0`` at every other window key.  Exactly one
            window key must carry the 1.0 (the count's correctness contract).
        max_gap: the worst-case gap bound (< ~350 for DOOM).  Sizes both the
            reciprocal's input range and its breakpoint density.

    Returns:
        length-1 node: ``gap`` in ``[0, max_gap]``, accurate to well under ±0.5
        of the true integer out to ``max_gap``.  **Only meaningful where the
        window is non-empty** (the marker is visible); at empty-window queries
        the result is bounded garbage (~``max_gap``), not an error.
    """
    assert len(window_validity) == 1, "window_validity must be 1-D"
    assert len(marker_onehot) == 1, "marker_onehot must be 1-D"
    assert max_gap >= 1, "max_gap must be >= 1"

    # Guard: the quasi-static approximation holds only when the slowest-plane
    # frequency is small enough.  The analytic bound on window non-uniformity
    # is ~333 × θ_slow² × max_gap³; exceeding 0.45 pushes the total gap error
    # past the ±0.5 rounding threshold.  At base=5e5 this requires d_head ≥ 64
    # for max_gap=350.
    theta_slow = float(rope_inv_freq(rope.d_head, rope.base)[-1])
    approx_err = 333.0 * theta_slow**2 * max_gap**3
    if approx_err > 0.45:
        raise ValueError(
            f"count_since_marker: estimated gap error {approx_err:.2f} > 0.45 "
            f"(theta_slow={theta_slow:.2e}, max_gap={max_gap}, "
            f"d_head={rope.d_head}).  "
            f"Increase d_head/base or reduce max_gap — at base=5e5, d_head≥64 "
            f"is safe for max_gap=350."
        )

    # mean over the window = (1 + 0*gap) / (gap+1) = 1/(gap+1).  At a query
    # whose window is empty (no valid key), attend_mean_where degrades to
    # garbage (its documented contract); reciprocal's clamp then bounds the
    # result to ~max_gap rather than crashing, but the gap is meaningless
    # there.  The caller must only consume the gap at positions whose window
    # holds the marker (e.g. at/after a span start) -- see docstring.
    mean = attend_mean_where(rope, window_validity, marker_onehot)

    # reciprocal over a padded range; size breakpoints for the per-gap
    # resolution at the bound (see module docstring / _RECIP_REL_SAFETY).
    recip_lo = 1.0 / (max_gap + 1.5)
    recip_hi = 1.5
    target_rel = 0.5 / (max_gap + 1.0) / _RECIP_REL_SAFETY
    # (r-1)^2/8 <= target_rel  =>  r <= 1 + sqrt(8*target_rel)
    r_max = 1.0 + math.sqrt(8.0 * target_rel)
    n_breakpoints = max(32, int(math.log(recip_hi / recip_lo) / math.log(r_max)) + 2)
    step = (recip_hi - recip_lo) / (n_breakpoints - 1)

    window_size = reciprocal(mean, min_value=recip_lo, max_value=recip_hi, step=step)
    return add_const(window_size, -1.0)
