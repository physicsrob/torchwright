"""Phase 6 — local recency: the intrinsic rotary distance-decay lobe.

docs/rope_port_plan.md Phase 6.  "Most recent matching" is no longer a
precomputed octant ramp read against an injected ``<ref>`` token; it is the
self-similarity of the rotation itself — a constant feature on a Hann-tapered
mid-band of planes gives the logit ``Σ_p amp_p cos(Δ·θ_p)``
(:func:`~torchwright.graph.rope.rope_lobe_band`), maximal at ``Δ=0`` and
**monotone-decreasing only within the lobe window** ``W``.

These are the smallest-layer (oracle, ``node.compute``) confidence tests:

- the lobe is strictly decreasing to ``W`` and **breaks down past it** (a farther
  key outscores a nearer one — the load-bearing limit, tested not footnoted);
- ``get_prev_value`` latches the most-recent true position;
- ``get_prev_value`` **refuses to build on a degenerate lobe band** (a
  ≤2-plane band Hann-tapers to its endpoint zeros, the gate sized from the
  collapsed peak leaks softmax weight to false-cond keys, and the latch
  silently drifts — the failure that once skewed ``count_since_marker`` by
  1.2 counts at ``d_head=32, max_positions=64``).

The compiled-path parity / prefill==decode checks are in
``tests/compile/forward/test_rope_local_recency.py``.
"""

import pytest
import torch

from torchwright.graph import InputNode
from torchwright.graph.rope import rope_lobe_band, rope_inv_freq
from torchwright.ops.attention_ops import get_prev_value
from torchwright.ops.inout_nodes import create_rope_config

# Production-faithful recency grid: the lobe must ride real slow frequencies, so
# the recency band (planes 12..96) is disjoint from the content slow-planes only
# at a wide d_head (§9 calibration discipline — test at the real config).
D_HEAD = 256
CAP = 61440


def _lobe_curve():
    """``(W, peak, lobe(Δ))`` for the default production band."""
    planes, amps = rope_lobe_band(D_HEAD, 5e5, CAP)
    inv = rope_inv_freq(D_HEAD, 5e5)
    theta = torch.tensor([float(inv[p]) for p in planes])

    def lobe(deltas):
        return (torch.cos(torch.outer(deltas.float(), theta)) * amps).sum(dim=1)

    return float(amps.sum()), theta, amps, lobe


# --------------------------------------------------------------------------- #
# The lobe shape (pure rope_lobe_band math)                                    #
# --------------------------------------------------------------------------- #
def test_lobe_strictly_decreasing_within_window():
    """The distance-decay lobe is strictly decreasing over ``[0, W)`` — a nearer
    key always outscores every farther one inside the window."""
    peak, _, _, lobe = _lobe_curve()
    R = 600
    s = lobe(torch.arange(0, R + 1))
    # Monotone window W: first Δ where some farther key ties/beats it.
    rmax = torch.flip(torch.cummax(torch.flip(s, [0]), 0).values, [0])
    wins = s[:-1] > rmax[1:]
    W = int(torch.argmin(wins.int())) if not bool(wins.all()) else R
    assert W >= 400, f"local-recency window W={W} below the ~415 target (4× ~100)"
    diffs = s[:W] - s[1 : W + 1]
    assert bool((diffs > 0).all()), f"lobe not strictly decreasing on [0,{W})"
    assert peak > 40.0  # Σ amp_p ≈ 42.6 at the production band


def test_lobe_breaks_down_past_window():
    """Past ``W`` the lobe is non-monotone: a concretely farther key outscores a
    nearer one.  This is the local limit — tested, not a footnote (Phase 6)."""
    _, _, _, lobe = _lobe_curve()
    R = 1500
    s = lobe(torch.arange(0, R + 1))
    # Find the first Δ1 (just past the window) that loses to some farther Δ2.
    rmax = torch.flip(torch.cummax(torch.flip(s, [0]), 0).values, [0])
    wins = s[:-1] > rmax[1:]
    W = int(torch.argmin(wins.int()))
    inversion = None
    for d1 in range(W, R):
        tail = s[d1 + 1 :]
        if tail.numel() and float(tail.max()) > float(s[d1]):
            d2 = d1 + 1 + int(torch.argmax(tail))
            inversion = (d1, d2)
            break
    assert inversion is not None, "no recency inversion found past W"
    d1, d2 = inversion
    assert d2 > d1 and float(lobe(torch.tensor([d2]))) > float(lobe(torch.tensor([d1])))


def _rope():
    return create_rope_config(d_head=D_HEAD, max_positions=CAP)


# --------------------------------------------------------------------------- #
# get_prev_value — most-recent true position                                   #
# --------------------------------------------------------------------------- #


def test_get_prev_value_latches_most_recent_true():
    """get_prev_value reads the value at the most recent position where cond is
    true; a single far trigger still latches (the content gate dominates the
    bounded lobe regardless of distance)."""
    rope = _rope()
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    cond = InputNode("cond", 1, value_range=(-1.0, 1.0))
    out = get_prev_value(rope, value, cond)

    n = 120
    value_in = (10.0 + torch.arange(n, dtype=torch.float32)).unsqueeze(1)
    # cond true only at position 3 (one trigger, far back by the end).
    cond_in = -torch.ones(n, 1)
    cond_in[3, 0] = 1.0
    out_v = out.compute(
        n_pos=n, input_values={"value": value_in, "cond": cond_in}
    ).reshape(-1)
    # From position 3 onward, the latched value is value[3] = 13.
    for p in range(3, n):
        assert abs(out_v[p].item() - 13.0) < 1e-2, (p, out_v[p].item())


# --------------------------------------------------------------------------- #
# get_prev_value — degenerate lobe bands refuse to build                       #
# --------------------------------------------------------------------------- #


def test_get_prev_value_raises_on_degenerate_lobe_band():
    """A rope config whose lobe band collapses (2 surviving planes both
    Hann-tapered to the endpoint floor: d_head=32, max_positions=64) must fail
    loudly at build time.  Before the guard this built a silently-leaky latch
    whose validity wobble, amplified 1000x by attend_mean_where, skewed
    count_since_marker by 1.2 counts at a gap of 4."""
    rope = create_rope_config(d_head=32, max_positions=64)
    value = InputNode("value", 1, value_range=(-1.0, 1.0))
    cond = InputNode("cond", 1, value_range=(-1.0, 1.0))
    with pytest.raises(ValueError, match="too weak to latch"):
        get_prev_value(rope, value, cond)


def test_get_prev_value_builds_on_healthy_bands():
    """Every config the repo actually uses clears the leak bound with orders of
    magnitude to spare — including the narrowest (d_head=16, max_positions=512,
    a 3-plane band whose Hann midpoint survives)."""
    for d_head, max_positions in [(16, 512), (32, 512), (64, 4096), (256, 61440)]:
        rope = create_rope_config(d_head=d_head, max_positions=max_positions)
        value = InputNode("value", 1, value_range=(-1.0, 1.0))
        cond = InputNode("cond", 1, value_range=(-1.0, 1.0))
        get_prev_value(rope, value, cond)  # must not raise
