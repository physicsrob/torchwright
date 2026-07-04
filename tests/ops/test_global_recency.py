"""Phase 7 — global recency: oracle-level tests.

These tests exercise `global_position_from_bos` and
`attend_most_recent_globally` at the oracle level (``node.compute`` — exact
math), verifying:

1. The BOS-weight mechanism gives per-position outputs that round to the correct
   absolute position (integer recovery within 0.025).
2. `attend_most_recent_globally` picks the most recent matching key across a gap
   of 490 positions — the exact scenario where Phase 6 local recency FAILS
   (gap > W ≈ 415) — and picks correctly.
3. Content dominates position: a non-matching key closer to the query loses to
   a matching key further away.

Compiled-path parity / prefill==decode tests live in
``tests/compile/forward/test_rope_global_recency.py``.
"""

import math

import torch

from torchwright.graph import InputNode
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops._math import _theta_slow, _w_of_m
from torchwright.ops.relu.global_recency import (
    attend_most_recent_globally,
    global_position_from_bos,
)
from torchwright.ops.inout_nodes import create_rope_config

D_HEAD = 256
CAP = 61440


def _rope():
    return create_rope_config(d_head=D_HEAD, max_positions=CAP)


# ---------------------------------------------------------------------------
# global_position_from_bos — integer recovery
# ---------------------------------------------------------------------------


def test_bos_weight_gives_correct_position():
    """Output of global_position_from_bos rounds to the correct absolute
    position at oracle precision.

    Tests n=80 positions to keep oracle compute fast.

    The validation script (scripts/rope_global_recency_validate.py) confirms
    PWL-only error < 0.009 when sampled every 10 positions.  However, the
    piecewise_linear implementation uses a sum of 1024 ReLU slope-changes
    in float32; fp32 accumulation in this sum adds up to ~0.09 at small
    positions (the initial slope is O(250000), and each of the ~1022 active
    terms carries rounding error).  Tolerance 0.15 is the empirical ceiling
    seen here — still 3.3× below the 0.5 rounding threshold that matters for
    downstream use.
    """
    rope = _rope()
    n = 80

    bos_indicator = InputNode("bos", 1, value_range=(0.0, 1.0))
    pos = global_position_from_bos(rope, bos_indicator)

    bos_in = torch.zeros(n, 1)
    bos_in[0, 0] = 1.0

    result = pos.compute(n_pos=n, input_values={"bos": bos_in}).reshape(-1)

    # All positions must round to the correct integer.  The 0.5 threshold
    # is the hard requirement; 0.15 is our empirical float32 ceiling.
    max_err = 0.0
    worst = -1
    for m in range(n):
        err = abs(result[m].item() - float(m))
        if err > max_err:
            max_err = err
            worst = m

    assert max_err < 0.15, (
        f"max position error {max_err:.4f} at pos {worst}; "
        f"expected < 0.15 (float32 ReLU-sum ceiling; rounding threshold is 0.5)"
    )


def test_position_strictly_increases():
    """The recovered position increases monotonically with absolute position
    (oracle precision, n=60).

    This confirms the PWL inverse faithfully reflects w(m) being strictly
    monotone decreasing.
    """
    rope = _rope()
    n = 60

    bos_indicator = InputNode("bos", 1, value_range=(0.0, 1.0))
    pos = global_position_from_bos(rope, bos_indicator)

    bos_in = torch.zeros(n, 1)
    bos_in[0, 0] = 1.0

    result = pos.compute(n_pos=n, input_values={"bos": bos_in}).reshape(-1)

    for m in range(1, n - 1):
        assert result[m].item() < result[m + 1].item(), (
            f"position not monotone: result[{m}]={result[m].item():.3f} "
            f">= result[{m+1}]={result[m+1].item():.3f}"
        )


# ---------------------------------------------------------------------------
# attend_most_recent_globally — selection tests
# ---------------------------------------------------------------------------


def _global_selection(n_pos, match_positions, value_fn, *, exclude_self=False):
    """E8-coded content, one match type vs one non-match type.

    Builds global_position_from_bos and attend_most_recent_globally, then
    evaluates on oracle values.  Returns per-position selected values.
    """
    rope = _rope()

    bos_indicator = InputNode("bos", 1, value_range=(0.0, 1.0))
    q = InputNode("q", 8, value_range=(-20.0, 20.0))
    k = InputNode("k", 8, value_range=(-20.0, 20.0))
    v = InputNode("v", 1, value_range=(-1.0e4, 1.0e4))

    global_pos = global_position_from_bos(rope, bos_indicator)
    out = attend_most_recent_globally(
        rope, q, k, global_pos, v, exclude_self=exclude_self
    )

    target, other = index_to_vector(3), index_to_vector(0)
    bos_in = torch.zeros(n_pos, 1)
    bos_in[0, 0] = 1.0
    key_in = torch.stack(
        [target if p in match_positions else other for p in range(n_pos)]
    )
    query_in = target.unsqueeze(0).expand(n_pos, -1).contiguous()
    value_in = torch.tensor(
        [value_fn(p) for p in range(n_pos)], dtype=torch.float32
    ).reshape(n_pos, 1)

    return out.compute(
        n_pos=n_pos,
        input_values={"bos": bos_in, "q": query_in, "k": key_in, "v": value_in},
    ).reshape(-1)


def test_picks_most_recent_within_window():
    """Sparse matches within the local window: most recent is selected.

    Matches every 10 positions (0, 10, 20, …, 120).  Adjacent match logit gap
    is recency_scale × 10 = 10, giving softmax weight exp(10)/(exp(10)+1) >
    99.99% — essentially one-hot on the most recent match.
    """
    n = 130
    stride = 10
    matches = set(range(0, n, stride))
    out = _global_selection(n, matches, value_fn=lambda p: float(p))

    for p in range(1, n):
        # m <= p: the query at p can see itself if it's a match (causal, inclusive).
        most_recent = max(m for m in matches if m <= p)
        assert (
            abs(out[p].item() - float(most_recent)) < 0.5
        ), f"pos {p}: expected most-recent match {most_recent}, got {out[p].item():.2f}"


def test_fixes_phase6_breakdown():
    """The exact scenario where Phase-6 local recency FAILS: two matches, the
    only recent one is at position 500 and the older at position 10 — gap=490
    exceeds the local window W ≈ 415.

    Phase 6 `attend_most_recent_matching` would invert here and pick pos 10
    (the farther match) because the lobe is non-monotone past W.

    Phase 7 `attend_most_recent_globally` must pick pos 500 (the correct,
    more recent match) for all query positions 501..599.
    """
    n = 600
    recent, older = 500, 10
    out = _global_selection(n, {older, recent}, value_fn=lambda p: float(p))

    for p in range(recent + 1, n):
        # The most recent match is at position 500; value_fn(500) = 500.0.
        assert abs(out[p].item() - float(recent)) < 0.5, (
            f"query {p}: expected most-recent pick = pos {recent} "
            f"(value {float(recent)}), got {out[p].item():.2f} — "
            f"gap {p - recent} from recent, {p - older} from older"
        )


def test_content_dominates_recency():
    """A non-matching key at a very recent position loses to a matching key at
    a much older position — content score dominates the position tiebreak.

    Scenario: one match at position 5, non-matches everywhere else (including
    position n-2, very recent).  Query from position n-1 should pick position 5.
    """
    n = 80
    match_pos = 5
    out = _global_selection(n, {match_pos}, value_fn=lambda p: float(p))

    for p in range(match_pos + 1, n):
        assert abs(out[p].item() - float(match_pos)) < 0.5, (
            f"query {p}: expected content-dominant pick = {match_pos}, "
            f"got {out[p].item():.2f}"
        )


def test_very_far_gap_still_correct():
    """Gap of 600 positions (well past W ≈ 415): picks the more recent match.

    Two matches at positions 2 and 602; queries from 603..699.
    Phase 6 would fail; Phase 7 must pick 602.
    """
    n = 700
    recent, older = 602, 2
    out = _global_selection(n, {older, recent}, value_fn=lambda p: float(p))

    for p in range(recent + 1, n):
        assert abs(out[p].item() - float(recent)) < 0.5, (
            f"query {p}: expected pos {recent}, got {out[p].item():.2f} "
            f"(gap from recent={p - recent}, from older={p - older})"
        )


def test_exclude_self_does_not_pick_self():
    """With exclude_self=True, a query at a match position picks the prior match.

    Stride-10 matches {0, 10, 20, 30, 40}.  exclude_self shifts each key back
    one position, so the self-key at position j appears at shifted position j+1
    (past the causal boundary).  The most recent VISIBLE match for query j (a
    multiple of 10) is at shifted position j-9, carrying value j-10.

    Adjacent shifted-match logit gap is recency_scale × 10 = 10, so exp(10) ≈
    22026 → effectively one-hot on the previous match.
    """
    n = 50
    stride = 10
    matches = set(range(0, n, stride))  # {0, 10, 20, 30, 40}
    out = _global_selection(n, matches, value_fn=lambda p: float(p), exclude_self=True)
    # At a match position p, self-key is shifted out; most-recent visible match
    # carries value p-stride.
    for p in [20, 30, 40]:
        expected = float(p - stride)
        assert abs(out[p].item() - expected) < 0.5, (
            f"query {p} (exclude_self): expected previous-match value {expected}, "
            f"got {out[p].item():.2f}"
        )


# ---------------------------------------------------------------------------
# Formula correctness at large m — regression guard for exp vs linear bug
# ---------------------------------------------------------------------------


def test_position_recovery_at_n500():
    """Oracle position recovery at n=500 — mid-range beyond the n=80 basic test.

    The full-cap validation (n=61440) lives in
    ``scripts/rope_global_recency_validate.py`` (analytic path).  This test
    confirms integer recovery at an intermediate scale reachable in the oracle
    (exact math) path without a compilation step.
    """
    rope = _rope()
    n = 500

    bos_indicator = InputNode("bos", 1, value_range=(0.0, 1.0))
    pos = global_position_from_bos(rope, bos_indicator)

    bos_in = torch.zeros(n, 1)
    bos_in[0, 0] = 1.0

    result = pos.compute(n_pos=n, input_values={"bos": bos_in}).reshape(-1)
    for m in range(n):
        err = abs(result[m].item() - float(m))
        assert err < 0.5, (
            f"pos {m}: recovered {result[m].item():.3f}, expected {m}, "
            f"error {err:.3f} ≥ 0.5"
        )


def test_wider_content_W16():
    """attend_most_recent_globally works with W=16 content vectors.

    Confirms that the extra planes used by wider content (position tiebreak
    moves to the 17th-slowest plane) do not break recency ordering.
    """
    rope = _rope()
    W = 16

    bos_indicator = InputNode("bos", 1, value_range=(0.0, 1.0))
    q = InputNode("q", W, value_range=(-20.0, 20.0))
    k = InputNode("k", W, value_range=(-20.0, 20.0))
    v = InputNode("v", 1, value_range=(-1.0e4, 1.0e4))

    global_pos = global_position_from_bos(rope, bos_indicator)
    out = attend_most_recent_globally(rope, q, k, global_pos, v)

    n, stride = 80, 10
    matches = set(range(0, n, stride))

    match_vec = torch.zeros(W)
    match_vec[0] = 10.0  # match_gain(200) × dot(100) = 20000 >> max_position tiebreak
    other_vec = torch.zeros(W)

    bos_in = torch.zeros(n, 1)
    bos_in[0, 0] = 1.0
    key_in = torch.stack([match_vec if p in matches else other_vec for p in range(n)])
    query_in = match_vec.unsqueeze(0).expand(n, -1).contiguous()
    value_in = torch.tensor([float(p) for p in range(n)]).reshape(n, 1)

    result = out.compute(
        n_pos=n,
        input_values={"bos": bos_in, "q": query_in, "k": key_in, "v": value_in},
    ).reshape(-1)

    for p in range(1, n):
        most_recent = max(m for m in matches if m <= p)
        assert (
            abs(result[p].item() - float(most_recent)) < 0.5
        ), f"pos {p}: expected most-recent {most_recent}, got {result[p].item():.2f}"


def test_w_of_m_uses_exponential_formula():
    """_w_of_m must compute MAX_LEN^cos(m·θ), not MAX_LEN*cos(m·θ).

    For m ≤ 1000 the two formulas agree to five decimal places (cos ≈ 1,
    so MAX_LEN^cos ≈ MAX_LEN and MAX_LEN*cos ≈ MAX_LEN).  Past m ≈ 5000
    they diverge sharply: the error in recovered position reaches ~670 at
    m=30,000 and ~3,100 at m=50,000 with the linear formula — both well
    beyond the 0.5 rounding threshold.

    This test directly checks the formula against the exact exponential at
    several large m values, so a regression to MAX_LEN*cos fails loudly
    rather than silently (all currently committed oracle tests use n ≤ 700).
    """
    rope = _rope()
    theta = _theta_slow(rope)
    max_len = rope.max_positions

    for m in [1000, 5000, 20000, 50000]:
        cos_m = math.cos(m * theta)
        eff_correct = math.pow(max_len, cos_m)
        w_expected = eff_correct / (eff_correct + m)
        w_actual = _w_of_m(m, max_len, theta)
        assert abs(w_actual - w_expected) < 1e-9, (
            f"At m={m}: _w_of_m={w_actual:.12f} but MAX_LEN^cos/(MAX_LEN^cos+m)="
            f"{w_expected:.12f}.  "
            f"Check for MAX_LEN*cos vs MAX_LEN^cos formula mismatch in _w_of_m."
        )
