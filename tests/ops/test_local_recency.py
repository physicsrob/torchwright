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
- ``attend_most_recent_matching`` picks the nearest content match within ``W``,
  and **exhibits the inversion** when the only matches sit more than ``W`` apart;
- ``get_prev_value`` latches the most-recent true position.

The compiled-path parity / prefill==decode checks are in
``tests/compile/forward/test_rope_local_recency.py``.
"""

import torch

from torchwright.graph import InputNode
from torchwright.graph.rope import rope_lobe_band, rope_inv_freq
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.attention_ops import attend_most_recent_matching, get_prev_value
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


# --------------------------------------------------------------------------- #
# attend_most_recent_matching — oracle selection                              #
# --------------------------------------------------------------------------- #
def _rope():
    return create_rope_config(d_head=D_HEAD, max_positions=CAP)


def _e8_selection(n_pos, match_positions, value_fn, *, exclude_self=False):
    """All non-match positions carry a distinct E8 code; ``match_positions`` carry
    the queried code.  Returns the per-position selected value (E8 content gives a
    large dot gap so the default match_gain dominates the bounded lobe)."""
    rope = _rope()
    q = InputNode("q", 8, value_range=(-20.0, 20.0))
    k = InputNode("k", 8, value_range=(-20.0, 20.0))
    v = InputNode("v", 1, value_range=(-1.0e4, 1.0e4))
    out = attend_most_recent_matching(rope, q, k, v, exclude_self=exclude_self)
    target, other = index_to_vector(3), index_to_vector(0)
    key_in = torch.stack(
        [target if p in match_positions else other for p in range(n_pos)]
    )
    query_in = target.unsqueeze(0).expand(n_pos, -1).contiguous()
    value_in = torch.tensor(
        [value_fn(p) for p in range(n_pos)], dtype=torch.float32
    ).reshape(n_pos, 1)
    return out.compute(
        n_pos=n_pos, input_values={"q": query_in, "k": key_in, "v": value_in}
    ).reshape(-1)


def test_picks_nearest_match_within_window():
    """With many matches in the window, the nearest (most recent) is picked — the
    immediate predecessor under ``exclude_self`` over a 130-token prefill (the
    nearest of up to 129 candidates, all within W)."""
    n = 130
    out = _e8_selection(
        n, set(range(n)), value_fn=lambda p: float(p), exclude_self=True
    )
    # out[j] == predecessor index j-1 for every j>=1.
    for j in range(1, n):
        assert abs(out[j].item() - float(j - 1)) < 1e-2, (j, out[j].item())


def test_sparse_matches_pick_most_recent():
    """Sparse matches every 30 positions: each query picks the most recent."""
    n, every = 200, 30
    matches = set(range(0, n, every))
    out = _e8_selection(n, matches, value_fn=lambda p: float(p))
    for p in (60, 100, 150, 199):
        assert abs(out[p].item() - float((p // every) * every)) < 1e-1, (p, out[p])


def test_selection_breaks_down_past_window():
    """The load-bearing limit, made concrete: when the only two matches sit more
    than ``W`` apart, "most recent" inverts — the query selects the *farther*
    match, because past ``W`` the lobe is non-monotone (Phase 6 / Phase 7 split).
    Matches at positions 10 and 20; query at 439 is Δ=419 from 20 and Δ=429 from
    10, and lobe(429) > lobe(419), so it (wrongly) picks position 10."""
    n = 440
    out = _e8_selection(n, {10, 20}, value_fn=lambda p: float(p))
    # The nearer match (pos 20, Δ=419) LOSES to the farther (pos 10, Δ=429).
    assert abs(out[439].item() - 10.0) < 1e-1, (
        f"expected the inversion to pick the farther match (pos 10); got "
        f"{out[439].item()} — if this picks 20 the window/gain changed"
    )


# --------------------------------------------------------------------------- #
# get_prev_value — most-recent true position                                   #
# --------------------------------------------------------------------------- #
def test_value_wider_than_planes_splits():
    """Value wider than ``d_head/2`` passes through unchanged — the V/O head split
    (same as ``attend_argmax_dot``).  Width-24 value, E8 content, a single match
    so the pick is unambiguous (content gate) and the row passes through exactly."""
    rope = _rope()
    q = InputNode("q", 8, value_range=(-20.0, 20.0))
    k = InputNode("k", 8, value_range=(-20.0, 20.0))
    v = InputNode("v", 24, value_range=(-1.0e4, 1.0e4))
    out = attend_most_recent_matching(rope, q, k, v)
    target, other = index_to_vector(3), index_to_vector(0)
    n = 6
    # Only position 2 matches the query; every query asks for the target type.
    key_in = torch.stack([target if p == 2 else other for p in range(n)])
    query_in = target.unsqueeze(0).expand(n, -1).contiguous()
    value_in = torch.arange(0.0, float(n * 24)).reshape(n, 24)
    res = out.compute(n_pos=n, input_values={"q": query_in, "k": key_in, "v": value_in})
    # From pos 2 on, the only match is row 2 — the wide row passes through exactly.
    for p in range(2, n):
        assert torch.allclose(res[p], value_in[2], atol=1e-2), (p, res[p])


def test_wide_content_build_and_disjointness_guard():
    """Content width up to ``d_head/2 − max(lobe plane)`` builds; wider content
    overlaps the recency lobe band and raises a clear error (Phase 6)."""
    rope = _rope()
    q = InputNode("q", 16, value_range=(-1.0, 1.0))
    k = InputNode("k", 16, value_range=(-1.0, 1.0))
    v = InputNode("v", 1, value_range=(-100.0, 100.0))
    # Width 16: content planes 112..127, lobe band 12..96 — disjoint, builds.
    attend_most_recent_matching(rope, q, k, v)
    attend_most_recent_matching(rope, q, k, v, exclude_self=True)

    # Width 40 would need content planes 88..127, overlapping the lobe band.
    wq = InputNode("wq", 40, value_range=(-1.0, 1.0))
    wk = InputNode("wk", 40, value_range=(-1.0, 1.0))
    import pytest

    with pytest.raises(ValueError, match="lobe band"):
        attend_most_recent_matching(rope, wq, wk, v)


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
