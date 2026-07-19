"""Positional counting relative to marker tokens, on the swish machine.

``count_since_marker`` is attention hardware (a uniform mean over the
marker window — machine-neutral) plus one :func:`reciprocal` inversion;
only the reciprocal is machine-specific, so this module reuses the
shared guard math from ``torchwright/ops/_math.py`` verbatim and swaps
in the swiglu reciprocal (whose grid-spacing audit derives
``input_scale`` internally).
"""

import math

from torchwright.graph import Node, RopeConfig
from torchwright.graph.rope import rope_inv_freq

from torchwright.ops._math import _RECIP_REL_SAFETY
from torchwright.ops.attention_ops import attend_mean_where
from torchwright.ops.linear import add_const
from torchwright.ops.swiglu.arithmetic_ops import reciprocal


def count_since_marker(
    rope: RopeConfig,
    window_validity: Node,
    marker_onehot: Node,
    *,
    max_gap: int,
) -> Node:
    """Integer gap ``pos - marker_pos`` via a uniform-attention count.

    See the relu twin for the full mechanism notes (the quasi-static
    guard, the empty-window contract); structure identical, reciprocal
    swiglu.

    Args:
        rope: the RoPE config.
        window_validity: length-1 boolean (+1 / -1) marking keys in the
            window ``[marker, now]``.  Must be flat across keys to ~1e-4 —
            the mean head's 1000× validity gain amplifies any per-key wobble
            into a logit tilt (see the relu twin's Args note).
        marker_onehot: length-1 value, 1.0 at the single marker key
            inside the window, 0.0 elsewhere.
        max_gap: the worst-case gap bound; sizes the reciprocal's range
            and density.

    Returns:
        length-1 node: ``gap`` in ``[0, max_gap]``, accurate to well
        under ±0.5 out to ``max_gap``; only meaningful where the window
        is non-empty.
    """
    assert len(window_validity) == 1, "window_validity must be 1-D"
    assert len(marker_onehot) == 1, "marker_onehot must be 1-D"
    assert max_gap >= 1, "max_gap must be >= 1"

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

    # mean over the window = 1/(gap+1); invert and shift.
    mean = attend_mean_where(rope, window_validity, marker_onehot)

    recip_lo = 1.0 / (max_gap + 1.5)
    recip_hi = 1.5
    target_rel = 0.5 / (max_gap + 1.0) / _RECIP_REL_SAFETY
    r_max = 1.0 + math.sqrt(8.0 * target_rel)
    n_breakpoints = max(32, int(math.log(recip_hi / recip_lo) / math.log(r_max)) + 2)
    step = (recip_hi - recip_lo) / (n_breakpoints - 1)

    window_size = reciprocal(mean, min_value=recip_lo, max_value=recip_hi, step=step)
    return add_const(window_size, -1.0)
