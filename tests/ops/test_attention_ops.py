"""Unit tests for the content-based attention primitives in ``attention_ops``.

These primitives live in ``torchwright.ops.attention_ops``. These tests
run the ``Attn`` node's Python ``compute`` path directly (no
compiler round-trip) against a small hand-built graph. They verify that
the Q/K/V matrices produce the selections we expect under the standard
causal softmax.

The compile-path round-trip is exercised by the sort-variant end-to-end
tests in ``tests/compile/forward/test_sort_digits.py``.

A few things worth knowing before reading the assertions:

- The primitives use a non-hardness-scaled query projection, so the
  softmax is decisive only when key-space deltas are at least ~1 unit.
  Unique integer scores are the supported case; exact ties are softly
  averaged (the ``get_prev_value`` primitive has the same property). The
  tests therefore only use unique scores.
- Scores must be comfortably inside ``|score| <= 120`` so the resulting
  logit stays above the ``Attn.compute`` causal-mask sentinel of ``-1000``.
  Where tests exercise "a single valid position beats any other score"
  they stay below that ceiling.
"""

import torch

from torchwright.graph import InputNode
from torchwright.graph.asserts import assert_01, assert_integer, assert_onehot
from torchwright.graph.attn import Attn
from torchwright.ops.attention_ops import (
    _ABOVE_BONUS,
    _ABOVE_MATCH_BONUS,
    _BUCKET_BONUS,
    _QUERY_GAIN,
    _VALIDITY_BONUS,
    attend_argmax_dot,
    attend_argmin_above_in_bucket,
    attend_argmin_above_integer,
    attend_argmin_unmasked,
    attend_causal_mean,
    attend_mean_where,
)
from torchwright.ops.inout_nodes import create_rope_config

# Uniform rotary substrate for these oracle (``node.compute``) tests.  Under
# RoPE every content head is full-width rotary on the ``d_head`` ``rotate_half``
# grid, with the content relocated onto the slowest planes
# (:func:`place_on_slow_planes`), so the match is position-quasi-static over a
# bounded window.  d_head=64 is wide enough for every content head in this file
# (the widest is the build-only wide ``exclude_self`` matcher at content width
# 25, which needs ``d_head/2 >= 25``) and keeps the slow planes slow enough that
# the selection is unaffected by the rotation over these short windows.
D_HEAD = 64
MAX_POSITIONS = 2048


def _rope():
    return create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)


# ``attend_most_recent_matching`` / ``get_prev_value`` (the local rotary-lobe
# recency consumers) are exercised in ``tests/ops/test_local_recency.py`` (oracle)
# and ``tests/compile/forward/test_rope_local_recency.py`` (compiled) — they need
# the production d_head=256 grid so the recency band is disjoint from content.


def _run(out_node, n_pos, **inputs):
    """Call ``compute`` on an attention output with the given input tensors."""
    return out_node.compute(n_pos=n_pos, input_values=inputs)


# ---------------------------------------------------------------------------
# attend_argmin_unmasked
# ---------------------------------------------------------------------------


def test_attend_argmin_unmasked_empty_mask_picks_min_score():
    """With an all-zero mask, this degenerates to a plain argmin."""
    rope = _rope()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_unmasked(rope, score, mask, onehot, value)

    n_pos = 4
    # Scores: min is pos 2 (score 1.0).
    score_in = torch.tensor([[5.0], [3.0], [1.0], [4.0]])
    # Each key position is one-hot at its own slot index.
    onehot_in = torch.eye(4, 4)
    # Empty mask everywhere.
    mask_in = torch.zeros(4, 4)
    # Value payload — tag each input position uniquely.
    value_in = torch.eye(4, 4) * 2.0

    result = _run(
        out, n_pos, score=score_in, mask=mask_in, onehot=onehot_in, value=value_in
    )
    # Argmin over each prefix: pos 0, pos 1, pos 2, pos 2.
    assert torch.allclose(result[0], value_in[0], atol=1e-2), f"pos 0: {result[0]}"
    assert torch.allclose(result[1], value_in[1], atol=1e-2), f"pos 1: {result[1]}"
    assert torch.allclose(result[2], value_in[2], atol=1e-2), f"pos 2: {result[2]}"
    assert torch.allclose(result[3], value_in[2], atol=1e-2), f"pos 3: {result[3]}"


def test_attend_argmin_unmasked_skips_masked_index():
    """Masking a specific input slot skips it in favour of the next-best."""
    rope = _rope()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_unmasked(rope, score, mask, onehot, value)

    n_pos = 4
    # Position 2 is the global min, but it's already been "used": its slot
    # index (2) is set in the mask from position 2 onwards.
    score_in = torch.tensor([[5.0], [3.0], [1.0], [4.0]])
    onehot_in = torch.eye(4, 4)
    # Mask is zero until position 2, then slot 2 is set.
    mask_in = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    value_in = torch.eye(4, 4) * 2.0

    result = _run(
        out, n_pos, score=score_in, mask=mask_in, onehot=onehot_in, value=value_in
    )
    # At pos 2, slot 2 is masked, so the next-best in {0, 1, 2} is pos 1
    # (score 3). At pos 3, {0, 1, 3} are unmasked — min score is 3 at pos 1.
    assert torch.allclose(result[2], value_in[1], atol=1e-2), f"pos 2: {result[2]}"
    assert torch.allclose(result[3], value_in[1], atol=1e-2), f"pos 3: {result[3]}"


# ---------------------------------------------------------------------------
# attend_argmin_above_integer
# ---------------------------------------------------------------------------


def _build_indicators(scores, thresholds):
    """Build the key-side I(score_i > t_c) indicator basis in 0/1 form."""
    out = torch.zeros(len(scores), len(thresholds))
    for i, s in enumerate(scores):
        for c, t in enumerate(thresholds):
            out[i, c] = 1.0 if s > t else 0.0
    return out


def test_attend_argmin_above_integer_picks_smallest_above_threshold():
    """At each query position, pick the smallest score strictly above the threshold.

    The threshold is indicated by the one-hot query.
    """
    rope = _rope()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    indicators = InputNode("indicators", 10, value_range=(-100.0, 100.0))
    threshold_onehot = InputNode("threshold_onehot", 10, value_range=(-100.0, 100.0))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_above_integer(rope, score, indicators, threshold_onehot, value)

    n_pos = 5
    scores_list = [4.0, 2.0, 6.0, 1.0, 5.0]  # distinct integer-ish scores
    score_in = torch.tensor([[s] for s in scores_list])
    # Thresholds covered: d ∈ {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8} (10 slots).
    thresholds = list(range(-1, 9))
    indicators_in = _build_indicators(scores_list, thresholds)
    value_in = torch.eye(5, 4)

    # At every query position, test threshold = 2 (slot index 3 in
    # the -1..8 list). Expected: the smallest score > 2 that is in the
    # causal window.
    threshold_onehot_in = torch.zeros(n_pos, 10)
    threshold_onehot_in[:, 3] = 1.0  # select threshold d=2 at every query

    result = _run(
        out,
        n_pos,
        score=score_in,
        indicators=indicators_in,
        threshold_onehot=threshold_onehot_in,
        value=value_in,
    )
    # Scores > 2 over each prefix:
    #   pos 0: {4} → pos 0
    #   pos 1: {4} → pos 0 (pos 1 has score 2, not strictly above)
    #   pos 2: {4, 6} → pos 0 (smallest above 2)
    #   pos 3: {4, 6} → pos 0
    #   pos 4: {4, 6, 5} → pos 0
    assert torch.allclose(result[0], value_in[0], atol=1e-2)
    assert torch.allclose(result[1], value_in[0], atol=1e-2)
    assert torch.allclose(result[2], value_in[0], atol=1e-2)
    assert torch.allclose(result[3], value_in[0], atol=1e-2)
    assert torch.allclose(result[4], value_in[0], atol=1e-2)


def test_attend_argmin_above_integer_threshold_varies_per_query():
    """The one-hot threshold can differ per query position, giving different selections.

    Varying the threshold per position changes which score each query
    selects.
    """
    rope = _rope()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    indicators = InputNode("indicators", 10, value_range=(-100.0, 100.0))
    threshold_onehot = InputNode("threshold_onehot", 10, value_range=(-100.0, 100.0))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_above_integer(rope, score, indicators, threshold_onehot, value)

    n_pos = 4
    scores_list = [3.0, 1.0, 5.0, 2.0]
    score_in = torch.tensor([[s] for s in scores_list])
    thresholds = list(range(-1, 9))
    indicators_in = _build_indicators(scores_list, thresholds)
    value_in = torch.eye(4, 4)

    # Row k selects threshold index (2 + k): i.e. d = 1, 2, 3, 4.
    threshold_onehot_in = torch.zeros(n_pos, 10)
    threshold_onehot_in[0, 2] = 1.0  # d = 1
    threshold_onehot_in[1, 3] = 1.0  # d = 2
    threshold_onehot_in[2, 4] = 1.0  # d = 3
    threshold_onehot_in[3, 5] = 1.0  # d = 4

    result = _run(
        out,
        n_pos,
        score=score_in,
        indicators=indicators_in,
        threshold_onehot=threshold_onehot_in,
        value=value_in,
    )
    # Query 0: threshold 1, causal {3}. Smallest > 1: 3 (pos 0). → value[0]
    # Query 1: threshold 2, causal {3, 1}. Smallest > 2: 3 (pos 0). → value[0]
    # Query 2: threshold 3, causal {3, 1, 5}. Smallest > 3: 5 (pos 2). → value[2]
    # Query 3: threshold 4, causal {3, 1, 5, 2}. Smallest > 4: 5 (pos 2). → value[2]
    assert torch.allclose(result[0], value_in[0], atol=1e-2), f"q0: {result[0]}"
    assert torch.allclose(result[1], value_in[0], atol=1e-2), f"q1: {result[1]}"
    assert torch.allclose(result[2], value_in[2], atol=1e-2), f"q2: {result[2]}"
    assert torch.allclose(result[3], value_in[2], atol=1e-2), f"q3: {result[3]}"


def test_attend_argmin_unmasked_advances_through_all_slots():
    """Simulate selection sort: each step masks the previous winner."""
    rope = _rope()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_unmasked(rope, score, mask, onehot, value)

    n_pos = 4
    # Scores: 9, 3, 5, 1. Ascending sort order: pos 3 (1), pos 1 (3),
    # pos 2 (5), pos 0 (9). We feed the mask that "has already been
    # picked at each step" by hand, so we can check the selection advances.
    score_in = torch.tensor([[9.0], [3.0], [5.0], [1.0]])
    onehot_in = torch.eye(4, 4)
    # At query position 0, nothing picked yet → picks pos 0 (the only option).
    # At query position 1, mask pos 0 → picks pos 1 (the only other option,
    # score 3 < 9).
    # At query position 2, mask pos 0 and pos 1 → picks pos 2 (score 5).
    # At query position 3, mask pos 0, 1, 2 → picks pos 3 (score 1).
    mask_in = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    value_in = torch.eye(4, 4) * 2.0

    result = _run(
        out, n_pos, score=score_in, mask=mask_in, onehot=onehot_in, value=value_in
    )
    assert torch.allclose(result[0], value_in[0], atol=1e-2), f"pos 0: {result[0]}"
    assert torch.allclose(result[1], value_in[1], atol=1e-2), f"pos 1: {result[1]}"
    assert torch.allclose(result[2], value_in[2], atol=1e-2), f"pos 2: {result[2]}"
    assert torch.allclose(result[3], value_in[3], atol=1e-2), f"pos 3: {result[3]}"


# ---------------------------------------------------------------------------
# attend_mean_where
# ---------------------------------------------------------------------------


def test_attend_mean_where_averages_valid_positions():
    """Mean of value across valid positions, ignoring invalid ones."""
    rope = _rope()
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 3, value_range=(-100.0, 100.0))
    out = attend_mean_where(rope, validity, value)

    n_pos = 5
    # Positions 0, 1, 3 are valid; 2, 4 are invalid.
    validity_in = torch.tensor([[1.0], [1.0], [-1.0], [1.0], [-1.0]])
    value_in = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [99.0, 99.0, 99.0],  # invalid — should be ignored
            [0.0, 0.0, 6.0],
            [99.0, 99.0, 99.0],  # invalid
        ]
    )

    result = _run(out, n_pos, validity=validity_in, value=value_in)

    # At pos 3: causal window = {0, 1, 2, 3}. Valid = {0, 1, 3}.
    # Mean = (2+0+0)/3, (0+4+0)/3, (0+0+6)/3 = (0.667, 1.333, 2.0)
    expected_3 = torch.tensor([2.0 / 3, 4.0 / 3, 6.0 / 3])
    assert torch.allclose(result[3], expected_3, atol=1e-2), f"pos 3: {result[3]}"


def test_attend_mean_where_single_valid():
    """With one valid position, the mean is that position's value."""
    rope = _rope()
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 2, value_range=(-100.0, 100.0))
    out = attend_mean_where(rope, validity, value)

    n_pos = 3
    validity_in = torch.tensor([[-1.0], [1.0], [-1.0]])
    value_in = torch.tensor([[0.0, 0.0], [7.0, 3.0], [0.0, 0.0]])

    result = _run(out, n_pos, validity=validity_in, value=value_in)
    # At pos 1 and pos 2, only pos 1 is valid.
    assert torch.allclose(result[1], torch.tensor([7.0, 3.0]), atol=1e-2)
    assert torch.allclose(result[2], torch.tensor([7.0, 3.0]), atol=1e-2)


def test_attend_mean_where_wide_value():
    """Value wider than d_head — exercises the no-width-constraint path."""
    rope = _rope()
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 32, value_range=(-100.0, 100.0))  # wider than d_head
    out = attend_mean_where(rope, validity, value)

    n_pos = 3
    validity_in = torch.tensor([[1.0], [1.0], [-1.0]])
    v0 = torch.arange(32).float().unsqueeze(0)
    v1 = (torch.arange(32).float() * 2).unsqueeze(0)
    value_in = torch.cat([v0, v1, torch.zeros(1, 32)], dim=0)

    result = _run(out, n_pos, validity=validity_in, value=value_in)
    # At pos 1: mean of v0 and v1
    expected = (v0 + v1).squeeze(0) / 2
    assert torch.allclose(result[1], expected, atol=1e-2), f"pos 1: {result[1]}"


def test_attend_mean_where_all_valid_uniform():
    """With all positions valid, result is the running cumulative mean."""
    rope = _rope()
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_mean_where(rope, validity, value)

    n_pos = 4
    validity_in = torch.ones(n_pos, 1)
    value_in = torch.tensor([[4.0], [8.0], [12.0], [16.0]])

    result = _run(out, n_pos, validity=validity_in, value=value_in)
    # pos 0: mean(4) = 4
    # pos 1: mean(4, 8) = 6
    # pos 2: mean(4, 8, 12) = 8
    # pos 3: mean(4, 8, 12, 16) = 10
    assert abs(result[0].item() - 4.0) < 0.1
    assert abs(result[1].item() - 6.0) < 0.1
    assert abs(result[2].item() - 8.0) < 0.1
    assert abs(result[3].item() - 10.0) < 0.1


# ---------------------------------------------------------------------------
# attend_causal_mean
# ---------------------------------------------------------------------------


def test_attend_causal_mean_is_exact_cumulative_mean():
    """Zero-Q/K logits make the softmax an exact uniform 1/(t+1)."""
    rope = _rope()
    value = InputNode("value", 2, value_range=(-100.0, 100.0))
    out = attend_causal_mean(rope, value)

    n_pos = 16
    torch.manual_seed(3)
    value_in = torch.randn(n_pos, 2) * 50.0
    result = _run(out, n_pos, value=value_in)
    expected = torch.cumsum(value_in, dim=0) / torch.arange(1, n_pos + 1).unsqueeze(1)
    assert torch.allclose(result, expected, atol=1e-4), (
        f"max err {(result - expected).abs().max()}"
    )


def test_attend_causal_mean_exact_under_full_rotary():
    """The exactness is layout-proof: rotating zero Q/K vectors is a no-op.

    So full rotary gives the same exact uniform mean as partial (the
    distinction that makes attend_mean_where only quasi-uniform there).
    """
    rope_full = create_rope_config(d_head=64, max_positions=4096)  # d_rot = d_head
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_causal_mean(rope_full, value)

    n_pos = 64
    value_in = torch.linspace(-100.0, 100.0, n_pos).unsqueeze(1)
    result = _run(out, n_pos, value=value_in)
    expected = torch.cumsum(value_in, dim=0) / torch.arange(1, n_pos + 1).unsqueeze(1)
    assert torch.allclose(result, expected, atol=1e-4), (
        f"max err {(result - expected).abs().max()}"
    )


def test_attend_causal_mean_output_scale_folds_into_o():
    """output_scale multiplies the mean without an extra graph op."""
    rope = _rope()
    value = InputNode("value", 1, value_range=(0.0, 100.0))
    out = attend_causal_mean(rope, value, output_scale=2.0)

    n_pos = 8
    value_in = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    result = _run(out, n_pos, value=value_in)
    # 2 * mean(0..t) = t exactly — the smoothed-position identity.
    expected = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    assert torch.allclose(result, expected, atol=1e-4), result.squeeze()


# ---------------------------------------------------------------------------
# attend_argmax_dot
# ---------------------------------------------------------------------------


def test_attend_argmax_dot_selects_best_match():
    """Selects the key position whose key_vector best matches query_vector."""
    rope = _rope()
    query_vector = InputNode("qv", 4, value_range=(-100.0, 100.0))
    key_vector = InputNode("kv", 4, value_range=(-100.0, 100.0))
    value = InputNode("value", 2, value_range=(-100.0, 100.0))
    out = attend_argmax_dot(rope, query_vector, key_vector, value)

    n_pos = 4
    # Query: one-hot selecting column 2 (0/1 convention)
    qv_in = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    # Key: ±1 masks. Pos 0: col 2 = -1 (not matching).
    # Pos 1: col 2 = +1 (matching). Pos 2: col 2 = +1 (matching).
    # Pos 3: col 2 = -1 (not matching).
    kv_in = torch.tensor(
        [
            [-1.0, 1.0, -1.0, 1.0],  # col 2 = -1
            [1.0, -1.0, 1.0, -1.0],  # col 2 = +1
            [-1.0, 1.0, 1.0, -1.0],  # col 2 = +1
            [1.0, -1.0, -1.0, 1.0],  # col 2 = -1
        ]
    )
    value_in = torch.tensor(
        [
            [10.0, 0.0],
            [20.0, 1.0],
            [30.0, 2.0],
            [40.0, 3.0],
        ]
    )

    result = _run(out, n_pos, qv=qv_in, kv=kv_in, value=value_in)

    # pos 0: only pos 0 visible, col 2 = -1 → forced to pick it
    assert torch.allclose(result[0], value_in[0], atol=1e-2)
    # pos 1: pos 1 has col 2 = +1, pos 0 has -1 → picks pos 1
    assert torch.allclose(result[1], value_in[1], atol=1e-2)
    # pos 2: pos 1 and 2 both have col 2 = +1 (tied dot product);
    # result is a soft average of value_in[1] and value_in[2].
    # Just verify it's between the two matching values and far from
    # the non-matching one.
    assert result[2, 0].item() > 19.0  # well above value_in[0]=10
    assert result[2, 0].item() < 31.0  # bounded by value_in[2]=30
    # pos 3: pos 1 and 2 match (col 2 = +1), pos 3 doesn't → soft avg of 1,2
    assert result[3, 0].item() > 19.0
    assert result[3, 0].item() < 31.0


def test_attend_argmax_dot_zero_key_isolation():
    """A zero key_vector (from cond_gate) produces dot product 0, losing to any match.

    It loses to any matching position.
    """
    rope = _rope()
    query_vector = InputNode("qv", 3, value_range=(-100.0, 100.0))
    key_vector = InputNode("kv", 3, value_range=(-100.0, 100.0))
    value = InputNode("value", 2, value_range=(-100.0, 100.0))
    out = attend_argmax_dot(rope, query_vector, key_vector, value)

    n_pos = 3
    # Query: one-hot column 0
    qv_in = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    # Key: pos 0 is gated to zero (non-participating), pos 1 matches,
    # pos 2 doesn't match.
    kv_in = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # gated zero (dot product = 0)
            [1.0, -1.0, -1.0],  # col 0 = +1 (dot product = +1)
            [-1.0, 1.0, 1.0],  # col 0 = -1 (dot product = -1)
        ]
    )
    value_in = torch.tensor(
        [
            [0.0, 0.0],  # gated zero value
            [5.0, 5.0],
            [9.0, 9.0],
        ]
    )

    result = _run(out, n_pos, qv=qv_in, kv=kv_in, value=value_in)
    # At pos 2: pos 1 (dot=+200) beats pos 0 (dot=0) and pos 2 (dot=-200)
    assert torch.allclose(result[2], value_in[1], atol=1e-2), f"pos 2: {result[2]}"


def test_attend_argmax_dot_different_queries_per_position():
    """Different query positions can select different matches."""
    rope = _rope()
    query_vector = InputNode("qv", 3, value_range=(-100.0, 100.0))
    key_vector = InputNode("kv", 3, value_range=(-100.0, 100.0))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_argmax_dot(rope, query_vector, key_vector, value)

    n_pos = 4
    # Key positions 0-2 each have a different column set to +1.
    # Position 3 queries column 0.
    kv_in = torch.tensor(
        [
            [1.0, -1.0, -1.0],  # "visible at col 0"
            [-1.0, 1.0, -1.0],  # "visible at col 1"
            [-1.0, -1.0, 1.0],  # "visible at col 2"
            [0.0, 0.0, 0.0],  # gated (non-participating)
        ]
    )
    # Queries: each position queries a different column.
    qv_in = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # query col 0
            [0.0, 1.0, 0.0],  # query col 1
            [0.0, 0.0, 1.0],  # query col 2
            [1.0, 0.0, 0.0],  # query col 0 again
        ]
    )
    value_in = torch.tensor([[10.0], [20.0], [30.0], [0.0]])

    result = _run(out, n_pos, qv=qv_in, kv=kv_in, value=value_in)
    # pos 0 queries col 0 → pos 0 (only option)
    assert abs(result[0].item() - 10.0) < 0.5
    # pos 1 queries col 1 → pos 1 (col 1 = +1)
    assert abs(result[1].item() - 20.0) < 0.5
    # pos 2 queries col 2 → pos 2 (col 2 = +1)
    assert abs(result[2].item() - 30.0) < 0.5
    # pos 3 queries col 0 → pos 0 (col 0 = +1, among {0,1,2,3})
    assert abs(result[3].item() - 10.0) < 0.5


# ---------------------------------------------------------------------------
# attend_argmin_above_in_bucket
# ---------------------------------------------------------------------------
#
# Filtered argmin over the past: at each query position, pick the smallest
# `score` among earlier rows that are valid, in a requested bucket, and
# above a requested threshold.  Bucket = equality lookup (one-hot row +
# one-hot picker); threshold = greater-than lookup (a "run of 1s up to the
# score" row + one-hot picker).  These tests run the exact-math
# `node.compute` path; the compiled fp32 round-trip lives in
# tests/compile/forward/test_bucketed_argmin.py.


def _onehot(idx, width):
    v = torch.zeros(width)
    v[idx] = 1.0
    return v


def _onehot_rows(indices, width):
    return torch.stack([_onehot(i, width) for i in indices])


def _above_table(scores, thresholds):
    """Key-side `score_above_each_threshold`: slot c == 1 iff score > thresholds[c]."""
    out = torch.zeros(len(scores), len(thresholds))
    for i, s in enumerate(scores):
        for c, t in enumerate(thresholds):
            out[i, c] = 1.0 if s > t else 0.0
    return out


def _unwrap_attn(node):
    """Hard-selection outputs are the Attn itself (claims are metadata)."""
    assert isinstance(node, Attn), f"expected Attn, got {type(node).__name__}"
    return node


# --- slow-plane layout helpers (RoPE) ------------------------------------- #
# Under RoPE a content head is a full-width rotary Attn: its compact ``(rows, W)``
# query/key matrices are relocated onto the slowest ``W`` planes of the
# ``d_head`` grid (``place_on_slow_planes``), so ``attn.d_qk == d_head`` and
# compact content column ``c`` lands at first-half physical column
# ``d_head/2 - 1 - c`` (its rotate_half partner left zero).  The structural
# regressions below therefore read cells through ``_slow_col`` and recover the
# logical content width via ``_content_width`` rather than indexing the old
# compact layout directly.


def _slow_col(attn, c):
    """Physical column holding compact content column ``c`` after relocation."""
    return attn.d_qk // 2 - 1 - c


def _content_width(attn):
    """The logical content width W = number of slow planes carrying content.

    ``place_on_slow_planes`` fills first-half columns ``half-1 .. half-W`` (one
    per content scalar) and leaves the rest zero, so W is ``half`` minus the
    lowest nonzero first-half column across the query and key matrices.
    """
    half = attn.d_qk // 2
    first_half = torch.cat([attn.query_matrix, attn.key_matrix], dim=0)[:, :half]
    used = torch.nonzero((first_half.abs() > 0).any(dim=0)).reshape(-1)
    assert used.numel() > 0, "no content columns found on the slow planes"
    return half - int(used.min())


def _baib_nodes(nb, nt, value_width=4, *, assert_hardness_gt=None):
    score = assert_integer(InputNode("baib_score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("baib_validity", 1, value_range=(-2.0, 2.0))
    key_bucket = InputNode("baib_kb", nb, value_range=(-2.0, 2.0))
    above = InputNode("baib_above", nt, value_range=(-2.0, 2.0))
    query_bucket = InputNode("baib_qb", nb, value_range=(-2.0, 2.0))
    threshold = InputNode("baib_th", nt, value_range=(-2.0, 2.0))
    value = InputNode("baib_value", value_width, value_range=(-100.0, 100.0))
    return attend_argmin_above_in_bucket(
        _rope(),
        score,
        validity,
        key_bucket,
        above,
        query_bucket,
        threshold,
        value,
        assert_hardness_gt=assert_hardness_gt,
    )


def _run_baib(
    out,
    n_pos,
    *,
    scores,
    buckets,
    valid,
    thresholds,
    q_bucket,
    q_thresh,
    value_in,
    nb,
    nt,
    above_override=None,
    perturb=None,
):
    kb = _onehot_rows(buckets, nb)
    qb = _onehot_rows(q_bucket, nb)
    th = _onehot_rows(q_thresh, nt)
    above = (
        _above_table(scores, thresholds) if above_override is None else above_override
    )
    if perturb is not None:
        # Soften every "1.0" toward a near-clean value to mimic compiled noise.
        for t in (kb, qb, th, above):
            t[t == 1.0] = perturb
    return _run(
        out,
        n_pos,
        baib_score=torch.tensor([[float(s)] for s in scores]),
        baib_validity=torch.tensor([[float(v)] for v in valid]),
        baib_kb=kb,
        baib_above=above,
        baib_qb=qb,
        baib_th=th,
        baib_value=value_in,
    )


def test_baib_picks_lowest_valid_in_bucket_above():
    """Among eligible rows, the smallest score wins.

    Eligible rows are valid, in the queried bucket, and above the
    queried threshold.
    """
    nb, nt = 3, 5
    out = _baib_nodes(nb, nt)
    n_pos = 5
    scores = [6, 4, 8, 3, 5]
    buckets = [1, 1, 1, 0, 1]  # row 3 is in a different bucket
    valid = [1, 1, 1, 1, 1]
    thresholds = [0, 1, 2, 3, 4]  # slot c -> score > c
    res = _run_baib(
        out,
        n_pos,
        scores=scores,
        buckets=buckets,
        valid=valid,
        thresholds=thresholds,
        q_bucket=[1] * n_pos,
        q_thresh=[2] * n_pos,
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # threshold slot 2 -> score > 2; bucket 1.
    #  q0: {row0=6}                              -> row0
    #  q1: {6, 4}                                -> row1 (4)
    #  q4: {6, 4, 8, 5}  (row3 bucket 0 excluded)-> row1 (4)
    assert res[0].argmax().item() == 0, res[0]
    assert res[1].argmax().item() == 1, res[1]
    assert res[4].argmax().item() == 1, res[4]


def test_baib_ignores_wrong_bucket():
    """A valid, above-threshold row in the WRONG bucket loses to a worse-score row.

    The winning row is in the right bucket.
    """
    nb, nt = 3, 4
    out = _baib_nodes(nb, nt)
    n_pos = 2
    res = _run_baib(
        out,
        n_pos,
        scores=[1, 5],
        buckets=[0, 1],
        valid=[1, 1],
        thresholds=[0, 1, 2, 3],
        q_bucket=[1, 1],
        q_thresh=[0, 0],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # row0 has the lower score (1) but is in bucket 0; query wants bucket 1.
    assert res[1].argmax().item() == 1, res[1]


def test_baib_ignores_not_above_threshold():
    """A valid, in-bucket row whose score is NOT above the threshold loses."""
    nb, nt = 2, 6
    out = _baib_nodes(nb, nt)
    n_pos = 2
    res = _run_baib(
        out,
        n_pos,
        scores=[3, 5],
        buckets=[1, 1],
        valid=[1, 1],
        thresholds=[0, 1, 2, 3, 4, 5],
        q_bucket=[1, 1],
        q_thresh=[4, 4],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # threshold slot 4 -> score > 4. row0 score 3 not above; row1 score 5 above.
    assert res[1].argmax().item() == 1, res[1]


def test_baib_ignores_invalid():
    """An invalid row otherwise matching (and with the lowest score) loses."""
    nb, nt = 2, 4
    out = _baib_nodes(nb, nt)
    n_pos = 2
    res = _run_baib(
        out,
        n_pos,
        scores=[1, 5],
        buckets=[1, 1],
        valid=[-1, 1],
        thresholds=[0, 1, 2, 3],
        q_bucket=[1, 1],
        q_thresh=[0, 0],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # row0 score 1 is invalid; row1 score 5 valid -> row1.
    assert res[1].argmax().item() == 1, res[1]


def test_baib_query_bucket_varies_per_position():
    """A per-position query bucket selects a different row at each position."""
    nb, nt = 3, 4
    out = _baib_nodes(nb, nt)
    n_pos = 3
    res = _run_baib(
        out,
        n_pos,
        scores=[5, 5, 5],
        buckets=[0, 1, 2],
        valid=[1, 1, 1],
        thresholds=[0, 1, 2, 3],
        q_bucket=[0, 1, 2],
        q_thresh=[0, 0, 0],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # Each row is alone in its bucket; query k asks bucket k (visible by pos k).
    assert res[0].argmax().item() == 0, res[0]
    assert res[1].argmax().item() == 1, res[1]
    assert res[2].argmax().item() == 2, res[2]


def test_baib_query_threshold_varies_per_position():
    """A per-position query threshold selects a different row at each position."""
    nb, nt = 1, 5
    out = _baib_nodes(nb, nt)
    n_pos = 3
    res = _run_baib(
        out,
        n_pos,
        scores=[2, 4, 6],
        buckets=[0, 0, 0],
        valid=[1, 1, 1],
        thresholds=[1, 2, 3, 4, 5],
        q_bucket=[0, 0, 0],
        q_thresh=[0, 2, 4],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # slot0 -> >1, slot2 -> >3, slot4 -> >5.
    #  q0: >1 over {2}        -> row0 (2)
    #  q1: >3 over {2, 4}     -> row1 (4)
    #  q2: >5 over {2, 4, 6}  -> row2 (6)
    assert res[0].argmax().item() == 0, res[0]
    assert res[1].argmax().item() == 1, res[1]
    assert res[2].argmax().item() == 2, res[2]


def test_baib_bucket_boundary():
    """Query bucket k: only the bucket-k row wins, even with lower-scoring neighbors.

    Candidates come from k-1, k, k+1.
    """
    nb, nt = 3, 4
    out = _baib_nodes(nb, nt)
    n_pos = 3
    res = _run_baib(
        out,
        n_pos,
        scores=[1, 5, 2],
        buckets=[0, 1, 2],
        valid=[1, 1, 1],
        thresholds=[0, 1, 2, 3],
        q_bucket=[1, 1, 1],
        q_thresh=[0, 0, 0],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # At q2: buckets 0 (score 1) and 2 (score 2) are off by one; only bucket 1.
    assert res[2].argmax().item() == 1, res[2]


def test_baib_threshold_boundary_equal_is_not_above():
    """Score == threshold is NOT strictly above."""
    nb, nt = 1, 5
    out = _baib_nodes(nb, nt)
    n_pos = 2
    res = _run_baib(
        out,
        n_pos,
        scores=[3, 4],
        buckets=[0, 0],
        valid=[1, 1],
        thresholds=[0, 1, 2, 3, 4],
        q_bucket=[0, 0],
        q_thresh=[3, 3],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # slot 3 -> threshold value 3. row0 score 3 (3 > 3 is False) excluded;
    # row1 score 4 (4 > 3) included.
    assert res[1].argmax().item() == 1, res[1]


def test_baib_arbitrary_value_width_does_not_grow_dqk():
    """A wide `value` does not change the logical Q/K (content) width."""
    nb, nt = 3, 4
    for vw in (1, 4, 20):
        out = _baib_nodes(nb, nt, value_width=vw)
        attn = _unwrap_attn(out)
        # Full-width rotary head: physical d_qk is the grid width, and the
        # content rides exactly 2 + nb + nt slow planes regardless of vw.
        assert attn.d_qk == D_HEAD, (vw, attn.d_qk)
        assert _content_width(attn) == 2 + nb + nt, (vw, _content_width(attn))
        assert attn.d_v == vw, (vw, attn.d_v)
    # And selection still works with a wide payload.
    out = _baib_nodes(nb, nt, value_width=20)
    n_pos = 2
    value_in = torch.zeros(n_pos, 20)
    value_in[0, 7] = 9.0
    value_in[1, 13] = 4.0
    res = _run_baib(
        out,
        n_pos,
        scores=[5, 6],
        buckets=[1, 1],
        valid=[1, 1],
        thresholds=[0, 1, 2, 3],
        q_bucket=[1, 1],
        q_thresh=[0, 0],
        value_in=value_in,
        nb=nb,
        nt=nt,
    )
    # q1: smallest score is row0 (5) -> payload with 9.0 at index 7.
    assert torch.allclose(res[1], value_in[0], atol=1e-2), res[1]


def test_baib_duplicate_matching_rows_with_identical_payload_blend_harmlessly():
    """Two matching rows that tie on score and payload produce that payload.

    The blend is a no-op.
    """
    nb, nt = 2, 4
    out = _baib_nodes(nb, nt)
    n_pos = 2
    payload = torch.tensor([1.0, 2.0, 3.0, 4.0])
    value_in = torch.stack([payload, payload])
    res = _run_baib(
        out,
        n_pos,
        scores=[5, 5],
        buckets=[1, 1],
        valid=[1, 1],
        thresholds=[0, 1, 2, 3],
        q_bucket=[1, 1],
        q_thresh=[0, 0],
        value_in=value_in,
        nb=nb,
        nt=nt,
    )
    assert torch.allclose(res[1], payload, atol=1e-3), res[1]


def test_baib_near_one_hot_inputs_do_not_change_selection():
    """Softened bucket / threshold tables still select the same row.

    Softened to ~0.97 (as compiled values are); the predicate-bonus
    margin absorbs it.
    """
    nb, nt = 3, 5
    out = _baib_nodes(nb, nt)
    n_pos = 5
    kwargs = {
        "scores": [6, 4, 8, 3, 5],
        "buckets": [1, 1, 1, 0, 1],
        "valid": [1, 1, 1, 1, 1],
        "thresholds": [0, 1, 2, 3, 4],
        "q_bucket": [1] * n_pos,
        "q_thresh": [2] * n_pos,
        "value_in": torch.eye(n_pos, 4),
        "nb": nb,
        "nt": nt,
    }
    clean = _run_baib(out, n_pos, **kwargs)
    noisy = _run_baib(out, n_pos, perturb=0.97, **kwargs)
    for q in range(n_pos):
        assert clean[q].argmax().item() == noisy[q].argmax().item(), (
            q,
            clean[q],
            noisy[q],
        )
    assert noisy[4].argmax().item() == 1, noisy[4]


def test_baib_threshold_above_max_has_no_match_but_does_not_crash():
    """With no row above the queried threshold, the output never NaNs or raises.

    Output is a finite blend (undefined by contract), never NaN, never
    raises.
    """
    nb, nt = 1, 6
    out = _baib_nodes(nb, nt)
    n_pos = 3
    res = _run_baib(
        out,
        n_pos,
        scores=[2, 3, 4],
        buckets=[0, 0, 0],
        valid=[1, 1, 1],
        thresholds=[0, 1, 2, 3, 4, 9],
        q_bucket=[0, 0, 0],
        q_thresh=[5, 5, 5],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # slot 5 -> threshold value 9; no score is above 9.
    assert torch.isfinite(res).all(), res


def test_baib_all_invalid_does_not_crash():
    """Every row invalid: output is finite, never NaN, never raises."""
    nb, nt = 2, 4
    out = _baib_nodes(nb, nt)
    n_pos = 3
    res = _run_baib(
        out,
        n_pos,
        scores=[2, 3, 4],
        buckets=[1, 1, 1],
        valid=[-1, -1, -1],
        thresholds=[0, 1, 2, 3],
        q_bucket=[1, 1, 1],
        q_thresh=[0, 0, 0],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    assert torch.isfinite(res).all(), res


def test_baib_scalar_bucket_recompute_is_unsafe_under_blend():
    """Negative result: recomputing presence from an averaged bucket id is unsafe.

    Two wrong-bucket rows can blend to the query bucket id.
    """
    nb, nt = 3, 4
    out = _baib_nodes(nb, nt, value_width=1)
    n_pos = 2
    # Query bucket 1, but the only rows are in buckets 0 and 2 (both wrong),
    # both valid + above, tied on score -> a 50/50 blend.  value carries the
    # scalar bucket id.
    value_in = torch.tensor([[0.0], [2.0]])
    res = _run_baib(
        out,
        n_pos,
        scores=[5, 5],
        buckets=[0, 2],
        valid=[1, 1],
        thresholds=[0, 1, 2, 3],
        q_bucket=[1, 1],
        q_thresh=[0, 0],
        value_in=value_in,
        nb=nb,
        nt=nt,
    )
    # (0 + 2) / 2 == 1 == the query bucket id: a FALSE positive if you trust it.
    assert abs(res[1].item() - 1.0) < 1e-2, res[1]


def test_baib_onehot_recompute_stays_false_under_blend():
    """Positive result: the robust recomputation stays low in the same no-match blend.

    The robust recomputation carries the selected `key_bucket_onehot`
    and dots it against the query one-hot.
    """
    nb, nt = 3, 4
    out = _baib_nodes(nb, nt, value_width=nb)
    n_pos = 2
    value_in = _onehot_rows([0, 2], nb)  # carry each row's bucket one-hot
    res = _run_baib(
        out,
        n_pos,
        scores=[5, 5],
        buckets=[0, 2],
        valid=[1, 1],
        thresholds=[0, 1, 2, 3],
        q_bucket=[1, 1],
        q_thresh=[0, 0],
        value_in=value_in,
        nb=nb,
        nt=nt,
    )
    # Blended one-hot ~ [0.5, 0, 0.5]; dot with query one-hot [0,1,0] ~ 0.
    match = float((res[1] * _onehot(1, nb)).sum())
    assert match < 0.9, (match, res[1])


def test_baib_assert_hardness_passes_on_clean_matches():
    """With `assert_hardness_gt`, a clean scene with a unique winner per query passes.

    It passes the reference-eval hardness predicate (does not raise).
    """
    nb, nt = 1, 5
    out = _baib_nodes(nb, nt, assert_hardness_gt=0.99)
    n_pos = 4
    res = _run_baib(
        out,
        n_pos,
        scores=[4, 2, 5, 3],
        buckets=[0, 0, 0, 0],
        valid=[1, 1, 1, 1],
        thresholds=[0, 1, 2, 3, 4],
        q_bucket=[0] * n_pos,
        q_thresh=[0] * n_pos,
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # Smallest score over each prefix: row0, row1, row1, row1.
    assert [res[q].argmax().item() for q in range(n_pos)] == [0, 1, 1, 1], res


def test_baib_d_qk_is_two_plus_buckets_plus_thresholds():
    """Width regression: the logical content width (slow planes used) is fixed.

    It is exactly 2 + n_buckets + n_thresholds.
    """
    for nb, nt in [(1, 1), (3, 5), (8, 6)]:
        attn = _unwrap_attn(_baib_nodes(nb, nt, value_width=7))
        assert _content_width(attn) == 2 + nb + nt, (nb, nt, _content_width(attn))


def test_baib_identity_vo_is_decoupled_from_dqk():
    """V/O regression: value passes through identity, decoupled from content width.

    The identity matrices have width len(value).
    """
    nb, nt, vw = 2, 3, 11
    attn = _unwrap_attn(_baib_nodes(nb, nt, value_width=vw))
    assert attn.d_v == vw
    assert torch.equal(attn.value_matrix, torch.eye(vw))
    assert torch.equal(attn.output_matrix, torch.eye(vw))
    # The content width does not include the value width.
    assert _content_width(attn) == 2 + nb + nt


def test_baib_bonus_magnitudes_are_op_local_not_inherited_1000():
    """The validity/bucket/above coefficients are op-local and dominate the score swing.

    They are baked into the matrices, op-local, and large enough to
    dominate the score swing.  The compiled probe cannot tell 256 from
    a too-small (or 1000) bonus in fp32, so these structural checks are
    what pin the constants.
    """
    nb, nt = 3, 4
    attn = _unwrap_attn(_baib_nodes(nb, nt))
    q, k = attn.query_matrix, attn.key_matrix
    # Compact content columns are relocated onto slow planes; read them through
    # _slow_col(attn, content_col).  Content layout (compact): col 0 score gain,
    # col 1 validity, cols 2..1+nb bucket, cols 2+nb.. above.
    assert q[0, _slow_col(attn, 0)].item() == _QUERY_GAIN
    assert q[0, _slow_col(attn, 1)].item() == 1.0
    assert k[1, _slow_col(attn, 1)].item() == _VALIDITY_BONUS
    # bucket cols start at content col 2: query carries the bonus, key a bare 1.0.
    for c in range(nb):
        assert q[1 + c, _slow_col(attn, 2 + c)].item() == _BUCKET_BONUS
        assert k[2 + c, _slow_col(attn, 2 + c)].item() == 1.0
    # above cols start at content col 2 + n_buckets; same query/key split.
    for c in range(nt):
        assert q[1 + nb + c, _slow_col(attn, 2 + nb + c)].item() == _ABOVE_MATCH_BONUS
        assert k[2 + nb + c, _slow_col(attn, 2 + nb + c)].item() == 1.0
    # The load-bearing invariant: each bonus must dominate the worst-case
    # gained score swing over the documented range S <= 12 — the smallest
    # of 2*_VALIDITY_BONUS, _BUCKET_BONUS, and _ABOVE_MATCH_BONUS must
    # exceed _QUERY_GAIN times 12.
    # An undersized "minimal" bonus (e.g. 50) silently mis-selects; this is the
    # only check that catches that, since fp32 compiled selection cannot.
    assert 2 * _VALIDITY_BONUS > _QUERY_GAIN * 12
    assert _BUCKET_BONUS > _QUERY_GAIN * 12
    assert _ABOVE_MATCH_BONUS > _QUERY_GAIN * 12
    # Op-local and distinct from the inherited 1000-unit globals.
    assert _VALIDITY_BONUS != 1000.0
    assert _BUCKET_BONUS != 1000.0
    assert _ABOVE_MATCH_BONUS != 1000.0
    # The shared global is untouched by introducing this op.
    assert _ABOVE_BONUS == 1000.0


def test_baib_bonus_dominates_worst_case_score_gap():
    """Behavioral pin: even at the worst-case score gap, the matching row must win.

    The filter-FAILING row has the best score (0) and the matching row
    the worst (30, a near-maximal in-range gap under the ~32 diameter).
    An undersized bonus mis-selects here even though the small-gap
    fixtures stay green.
    """
    nb, nt = 3, 12
    out = _baib_nodes(nb, nt)
    n_pos = 4
    thresholds = list(range(nt))  # slot c -> score > c
    res = _run_baib(
        out,
        n_pos,
        scores=[0, 0, 0, 30],
        buckets=[0, 1, 1, 1],  # row 0: wrong bucket
        valid=[1, -1, 1, 1],  # row 1: invalid
        thresholds=thresholds,
        q_bucket=[1, 1, 1, 1],  # query bucket 1
        q_thresh=[0, 0, 0, 0],  # > 0: row 2 (score 0) is NOT above
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    # Only row 3 passes all three filters; despite its worst score it wins.
    assert res[3].argmax().item() == 3, res[3]


def test_baib_single_column_bucket_and_threshold_select():
    """n_buckets == 1 and n_thresholds == 1, the asserted lower bound, still select.

    A 1-wide bucket is a constant no-op filter, a 1-wide threshold gates
    on the single above-column, and argmin-of-score still wins.
    """
    nb, nt = 1, 1
    out = _baib_nodes(nb, nt)
    n_pos = 3
    res = _run_baib(
        out,
        n_pos,
        scores=[6, 4, 8],
        buckets=[0, 0, 0],
        valid=[1, 1, 1],
        thresholds=[-1],  # every non-negative score is "above"
        q_bucket=[0, 0, 0],
        q_thresh=[0, 0, 0],
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
    assert [res[q].argmax().item() for q in range(n_pos)] == [0, 1, 1], res


def test_baib_all_zero_selectors_stay_finite():
    """All-zero query selectors drop that filter rather than crash, staying finite.

    This happens when no bucket / threshold column is chosen — e.g. an
    out-of-range query index.
    """
    nb, nt = 3, 4
    out = _baib_nodes(nb, nt)
    n_pos = 3
    res = _run(
        out,
        n_pos,
        baib_score=torch.tensor([[6.0], [4.0], [8.0]]),
        baib_validity=torch.ones(n_pos, 1),
        baib_kb=_onehot_rows([1, 1, 1], nb),
        baib_above=_above_table([6, 4, 8], [0, 1, 2, 3]),
        baib_qb=torch.zeros(n_pos, nb),  # no bucket selected
        baib_th=torch.zeros(n_pos, nt),  # no threshold selected
        baib_value=torch.eye(n_pos, 4),
    )
    assert torch.isfinite(res).all(), res
