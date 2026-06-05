"""Unit tests for the content-based attention primitives in
``torchwright.ops.attention_ops``.

These tests run the ``Attn`` node's Python ``compute`` path directly (no
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

from torchwright.graph import InputNode, PosEncoding
from torchwright.graph.asserts import assert_01, assert_integer, assert_onehot
from torchwright.ops.attention_ops import (
    attend_argmin,
    attend_argmax,
    attend_argmax_dot,
    attend_argmax_dot_where,
    attend_argmin_dot_where,
    attend_argmin_where,
    attend_argmax_where,
    attend_argmin_above_integer,
    attend_argmin_above_in_bucket,
    attend_argmin_unmasked,
    attend_argmin_valid_unmasked,
    attend_mean_where,
    attend_most_recent_matching,
    _QUERY_GAIN,
    _VALIDITY_BONUS,
    _BUCKET_BONUS,
    _ABOVE_MATCH_BONUS,
    _ABOVE_BONUS,
)
from torchwright.graph.attn import Attn
from torchwright.graph.misc import Assert


def _pe() -> PosEncoding:
    return PosEncoding(17)


def _run(out_node, n_pos, **inputs):
    """Call ``compute`` on an attention output with the given input tensors."""
    return out_node.compute(n_pos=n_pos, input_values=inputs)


# ---------------------------------------------------------------------------
# attend_argmin
# ---------------------------------------------------------------------------


def test_attend_argmin_happy_path():
    """Unique scores — argmin lands on the single minimum position."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin(pe, score, value)

    n_pos = 5
    # Scores are distinct.
    score_in = torch.tensor([[5.0], [3.0], [4.0], [2.0], [6.0]])
    value_in = torch.eye(5, 4)
    # Expected argmin over each causal prefix: 0, 1, 1, 3, 3.
    expected = torch.stack(
        [
            value_in[0],
            value_in[1],
            value_in[1],
            value_in[3],
            value_in[3],
        ]
    )

    result = _run(out, n_pos, score=score_in, value=value_in)
    assert result.shape == (n_pos, 4)
    assert torch.allclose(result, expected, atol=1e-3), f"got {result}"


def test_attend_argmin_scalar_value():
    """Width-1 value — smoke test with small unique scores."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_argmin(pe, score, value)

    n_pos = 3
    score_in = torch.tensor([[2.0], [1.0], [5.0]])
    value_in = torch.tensor([[10.0], [20.0], [30.0]])

    result = _run(out, n_pos, score=score_in, value=value_in)
    # pos 0 → value[0]=10. pos 1 → value[1]=20 (min 1.0). pos 2 → value[1]=20.
    assert abs(result[0].item() - 10.0) < 0.1
    assert abs(result[1].item() - 20.0) < 0.1
    assert abs(result[2].item() - 20.0) < 0.1


def test_attend_argmin_negative_scores():
    """argmin works with negative scores too (they're larger in -score space)."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    value = InputNode("value", 3, value_range=(-100.0, 100.0))
    out = attend_argmin(pe, score, value)

    n_pos = 3
    # Most negative score = minimum → argmin picks it.
    score_in = torch.tensor([[-1.0], [-5.0], [-3.0]])
    value_in = torch.eye(3, 3)

    result = _run(out, n_pos, score=score_in, value=value_in)
    assert torch.allclose(result[0], value_in[0], atol=1e-3)
    assert torch.allclose(result[1], value_in[1], atol=1e-3)
    assert torch.allclose(result[2], value_in[1], atol=1e-3)


# ---------------------------------------------------------------------------
# attend_argmax
# ---------------------------------------------------------------------------


def test_attend_argmax_happy_path():
    """Unique scores — argmax lands on the single maximum position."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmax(pe, score, value)

    n_pos = 5
    score_in = torch.tensor([[3.0], [1.0], [4.0], [2.0], [5.0]])
    value_in = torch.eye(5, 4)
    # argmax over prefixes: 0, 0, 2, 2, 4.
    expected = torch.stack(
        [
            value_in[0],
            value_in[0],
            value_in[2],
            value_in[2],
            value_in[4],
        ]
    )

    result = _run(out, n_pos, score=score_in, value=value_in)
    assert torch.allclose(result, expected, atol=1e-3), f"got {result}"


def test_attend_argmax_dual_of_argmin():
    """``attend_argmax(score)`` equals ``attend_argmin(-score)``."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    value = InputNode("value", 3, value_range=(-100.0, 100.0))
    out_max = attend_argmax(pe, score, value)

    # Build an argmin on negated scores by flipping the input.
    score_in = torch.tensor([[1.0], [5.0], [3.0]])
    value_in = torch.eye(3, 3)

    result_max = _run(out_max, 3, score=score_in, value=value_in)
    # Max score 5 is at position 1; argmax picks it from position 1 onwards.
    assert torch.allclose(result_max[0], value_in[0], atol=1e-3)
    assert torch.allclose(result_max[1], value_in[1], atol=1e-3)
    assert torch.allclose(result_max[2], value_in[1], atol=1e-3)


# ---------------------------------------------------------------------------
# attend_argmin_where
# ---------------------------------------------------------------------------


def test_attend_argmin_where_mask_overrides_score():
    """A valid high score beats an invalid low score."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_where(pe, score, validity, value)

    n_pos = 4
    # Position 1 has the lowest score (0) but is invalid.
    # Position 2 has score 1.0 and is valid.
    # Position 3 has score 2.0 and is valid.
    score_in = torch.tensor([[5.0], [0.0], [1.0], [2.0]])
    validity_in = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
    value_in = torch.eye(4, 4)

    result = _run(out, n_pos, score=score_in, validity=validity_in, value=value_in)
    # From pos 2 onwards the only valid minimum is pos 2.
    assert torch.allclose(result[2], value_in[2], atol=1e-3), f"pos 2: {result[2]}"
    assert torch.allclose(result[3], value_in[2], atol=1e-3), f"pos 3: {result[3]}"


def test_attend_argmin_where_single_valid_wins_any_score():
    """A single valid position is selected regardless of other scores."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_where(pe, score, validity, value)

    n_pos = 4
    # Only position 1 is valid, and it has a moderate score (kept under
    # _MAX_SCORE_ABS = 120 so its logit stays above the causal mask).
    score_in = torch.tensor([[0.0], [50.0], [1.0], [2.0]])
    validity_in = torch.tensor([[-1.0], [1.0], [-1.0], [-1.0]])
    value_in = torch.eye(4, 4)

    result = _run(out, n_pos, score=score_in, validity=validity_in, value=value_in)
    # From pos 1 onwards, the argmin-where must pick pos 1.
    for p in range(1, n_pos):
        assert torch.allclose(
            result[p], value_in[1], atol=1e-3
        ), f"pos {p}: {result[p]}"


def test_attend_argmin_where_advances_with_mask():
    """As more positions become valid over time, the selection advances."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_where(pe, score, validity, value)

    n_pos = 4
    # Scores: 4, 2, 1, 3. Validity: T, T, F, T.
    # At pos 0: valid={0}, min=4 → pos 0.
    # At pos 1: valid={0,1}, min=2 → pos 1.
    # At pos 2: valid={0,1}, min=2 → pos 1 (pos 2 invalid).
    # At pos 3: valid={0,1,3}, min=2 → pos 1 (pos 3's score is 3 > 2).
    score_in = torch.tensor([[4.0], [2.0], [1.0], [3.0]])
    validity_in = torch.tensor([[1.0], [1.0], [-1.0], [1.0]])
    value_in = torch.eye(4, 4)

    result = _run(out, n_pos, score=score_in, validity=validity_in, value=value_in)
    assert torch.allclose(result[0], value_in[0], atol=1e-3)
    assert torch.allclose(result[1], value_in[1], atol=1e-3)
    assert torch.allclose(result[2], value_in[1], atol=1e-3)
    assert torch.allclose(result[3], value_in[1], atol=1e-3)


# ---------------------------------------------------------------------------
# attend_argmax_where
# ---------------------------------------------------------------------------


def test_attend_argmax_where_mask_overrides_score():
    """A valid low score beats an invalid high score."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmax_where(pe, score, validity, value)

    n_pos = 4
    # Invalid high score at pos 1 should be ignored; max among
    # {pos 0 (1), pos 2 (2), pos 3 (3)} is pos 3.
    score_in = torch.tensor([[1.0], [99.0], [2.0], [3.0]])
    validity_in = torch.tensor([[1.0], [-1.0], [1.0], [1.0]])
    value_in = torch.eye(4, 4)

    result = _run(out, n_pos, score=score_in, validity=validity_in, value=value_in)
    assert torch.allclose(result[3], value_in[3], atol=1e-3), f"pos 3: {result[3]}"


def test_attend_argmax_where_single_valid_wins_any_score():
    """A single valid position is selected even with a tiny score."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmax_where(pe, score, validity, value)

    n_pos = 4
    # Only pos 2 is valid; other positions have higher but invalid scores.
    score_in = torch.tensor([[50.0], [40.0], [1.0], [30.0]])
    validity_in = torch.tensor([[-1.0], [-1.0], [1.0], [-1.0]])
    value_in = torch.eye(4, 4)

    result = _run(out, n_pos, score=score_in, validity=validity_in, value=value_in)
    assert torch.allclose(result[2], value_in[2], atol=1e-3)
    assert torch.allclose(result[3], value_in[2], atol=1e-3)


# ---------------------------------------------------------------------------
# attend_argmin_unmasked
# ---------------------------------------------------------------------------


def test_attend_argmin_unmasked_empty_mask_picks_min_score():
    """With an all-zero mask, this degenerates to a plain argmin."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_unmasked(pe, score, mask, onehot, value)

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
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_unmasked(pe, score, mask, onehot, value)

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
    """At each query position, pick the smallest score strictly above the
    threshold indicated by the one-hot query."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    indicators = InputNode("indicators", 10, value_range=(-100.0, 100.0))
    threshold_onehot = InputNode("threshold_onehot", 10, value_range=(-100.0, 100.0))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_above_integer(pe, score, indicators, threshold_onehot, value)

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
    """The one-hot threshold can differ per query position, giving
    different selections at each."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    indicators = InputNode("indicators", 10, value_range=(-100.0, 100.0))
    threshold_onehot = InputNode("threshold_onehot", 10, value_range=(-100.0, 100.0))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_above_integer(pe, score, indicators, threshold_onehot, value)

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
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_unmasked(pe, score, mask, onehot, value)

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
# attend_argmin_valid_unmasked
# ---------------------------------------------------------------------------


def test_attend_argmin_valid_unmasked_all_valid_empty_mask_picks_min():
    """All keys valid, empty mask — behaves like a plain argmin."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_valid_unmasked(pe, score, validity, mask, onehot, value)

    n_pos = 4
    score_in = torch.tensor([[5.0], [3.0], [1.0], [4.0]])
    validity_in = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    onehot_in = torch.eye(4, 4)
    mask_in = torch.zeros(4, 4)
    value_in = torch.eye(4, 4) * 2.0

    result = _run(
        out,
        n_pos,
        score=score_in,
        validity=validity_in,
        mask=mask_in,
        onehot=onehot_in,
        value=value_in,
    )
    # Argmin over prefixes: pos 0, pos 1, pos 2, pos 2.
    assert torch.allclose(result[0], value_in[0], atol=1e-2), f"pos 0: {result[0]}"
    assert torch.allclose(result[1], value_in[1], atol=1e-2), f"pos 1: {result[1]}"
    assert torch.allclose(result[2], value_in[2], atol=1e-2), f"pos 2: {result[2]}"
    assert torch.allclose(result[3], value_in[2], atol=1e-2), f"pos 3: {result[3]}"


def test_attend_argmin_valid_unmasked_validity_overrides_low_score():
    """The lowest-score key is invalid — attention picks the next-lowest valid."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_valid_unmasked(pe, score, validity, mask, onehot, value)

    n_pos = 4
    # pos 2 has lowest score (1.0) but is invalid.  Next-lowest valid = pos 1 (3.0).
    score_in = torch.tensor([[5.0], [3.0], [1.0], [4.0]])
    validity_in = torch.tensor([[1.0], [1.0], [-1.0], [1.0]])
    onehot_in = torch.eye(4, 4)
    mask_in = torch.zeros(4, 4)
    value_in = torch.eye(4, 4) * 2.0

    result = _run(
        out,
        n_pos,
        score=score_in,
        validity=validity_in,
        mask=mask_in,
        onehot=onehot_in,
        value=value_in,
    )
    assert torch.allclose(result[1], value_in[1], atol=1e-2), f"pos 1: {result[1]}"
    assert torch.allclose(result[2], value_in[1], atol=1e-2), f"pos 2: {result[2]}"
    assert torch.allclose(result[3], value_in[1], atol=1e-2), f"pos 3: {result[3]}"


def test_attend_argmin_valid_unmasked_mask_excludes_picked():
    """All valid, but the min-score slot is masked — expect next-best unmasked."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_valid_unmasked(pe, score, validity, mask, onehot, value)

    n_pos = 4
    score_in = torch.tensor([[5.0], [3.0], [1.0], [4.0]])
    validity_in = torch.ones(4, 1)
    onehot_in = torch.eye(4, 4)
    # From pos 2 onwards, slot 2 is masked.
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
        out,
        n_pos,
        score=score_in,
        validity=validity_in,
        mask=mask_in,
        onehot=onehot_in,
        value=value_in,
    )
    # pos 2: slot 2 masked, min of {0,1} is pos 1 (score 3).
    # pos 3: {0,1,3} unmasked, min is pos 1 (score 3).
    assert torch.allclose(result[2], value_in[1], atol=1e-2), f"pos 2: {result[2]}"
    assert torch.allclose(result[3], value_in[1], atol=1e-2), f"pos 3: {result[3]}"


def test_attend_argmin_valid_unmasked_mask_and_validity_combined():
    """Lowest score is masked, second-lowest invalid — pick third (valid + unmasked)."""
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_valid_unmasked(pe, score, validity, mask, onehot, value)

    n_pos = 4
    # Scores: 5 (pos 0), 3 (pos 1), 1 (pos 2), 4 (pos 3).
    # pos 2 (lowest) is masked; pos 1 (second-lowest) is invalid.
    # Expect pos 3 (third-lowest, valid, unmasked) at query pos 3.
    score_in = torch.tensor([[5.0], [3.0], [1.0], [4.0]])
    validity_in = torch.tensor([[1.0], [-1.0], [1.0], [1.0]])
    onehot_in = torch.eye(4, 4)
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
        out,
        n_pos,
        score=score_in,
        validity=validity_in,
        mask=mask_in,
        onehot=onehot_in,
        value=value_in,
    )
    # pos 3: valid-unmasked set = {0 (s=5), 3 (s=4)}, argmin = pos 3.
    assert torch.allclose(result[3], value_in[3], atol=1e-2), f"pos 3: {result[3]}"


def test_attend_argmin_valid_unmasked_all_valid_masked_repicks_masked():
    """No valid+unmasked key available — attention falls back to a masked-valid key.

    Documents the wasteful-but-safe end-of-sort behavior: when the whole
    valid set has been picked, the softmax prefers a masked-valid key
    over any unmasked-invalid key (validity dominates mask).  The caller
    gets back one of the previously-picked valid values rather than
    garbage from an invalid position.
    """
    pe = _pe()
    score = assert_integer(InputNode("score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    mask = assert_01(InputNode("mask", 4, value_range=(-100.0, 100.0)))
    onehot = assert_onehot(InputNode("onehot", 4, value_range=(-100.0, 100.0)))
    value = InputNode("value", 4, value_range=(-100.0, 100.0))
    out = attend_argmin_valid_unmasked(pe, score, validity, mask, onehot, value)

    n_pos = 4
    # Only pos 1 is valid — and its slot is masked.
    # Pos 0, 2, 3 are invalid (unmasked).
    score_in = torch.tensor([[5.0], [3.0], [1.0], [4.0]])
    validity_in = torch.tensor([[-1.0], [1.0], [-1.0], [-1.0]])
    onehot_in = torch.eye(4, 4)
    # Mask bit 1 on from query pos 1 onward (the one valid slot).
    mask_in = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    value_in = torch.eye(4, 4) * 2.0

    result = _run(
        out,
        n_pos,
        score=score_in,
        validity=validity_in,
        mask=mask_in,
        onehot=onehot_in,
        value=value_in,
    )
    # At pos 3, masked-valid (pos 1) must still win over any unmasked-invalid.
    assert torch.allclose(result[3], value_in[1], atol=1e-2), f"pos 3: {result[3]}"


# ---------------------------------------------------------------------------
# attend_mean_where
# ---------------------------------------------------------------------------


def test_attend_mean_where_averages_valid_positions():
    """Mean of value across valid positions, ignoring invalid ones."""
    pe = _pe()
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 3, value_range=(-100.0, 100.0))
    out = attend_mean_where(pe, validity, value)

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
    pe = _pe()
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 2, value_range=(-100.0, 100.0))
    out = attend_mean_where(pe, validity, value)

    n_pos = 3
    validity_in = torch.tensor([[-1.0], [1.0], [-1.0]])
    value_in = torch.tensor([[0.0, 0.0], [7.0, 3.0], [0.0, 0.0]])

    result = _run(out, n_pos, validity=validity_in, value=value_in)
    # At pos 1 and pos 2, only pos 1 is valid.
    assert torch.allclose(result[1], torch.tensor([7.0, 3.0]), atol=1e-2)
    assert torch.allclose(result[2], torch.tensor([7.0, 3.0]), atol=1e-2)


def test_attend_mean_where_wide_value():
    """Value wider than d_pos — exercises the no-width-constraint path."""
    pe = _pe()  # d_pos = 16
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 32, value_range=(-100.0, 100.0))  # wider than d_pos
    out = attend_mean_where(pe, validity, value)

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
    pe = _pe()
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_mean_where(pe, validity, value)

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
# attend_argmax_dot
# ---------------------------------------------------------------------------


def test_attend_argmax_dot_selects_best_match():
    """Selects the key position whose key_vector best matches query_vector."""
    pe = _pe()
    query_vector = InputNode("qv", 4, value_range=(-100.0, 100.0))
    key_vector = InputNode("kv", 4, value_range=(-100.0, 100.0))
    value = InputNode("value", 2, value_range=(-100.0, 100.0))
    out = attend_argmax_dot(query_vector, key_vector, value)

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
    """Zero key_vector (from cond_gate) produces dot product 0, losing to
    any matching position."""
    pe = _pe()
    query_vector = InputNode("qv", 3, value_range=(-100.0, 100.0))
    key_vector = InputNode("kv", 3, value_range=(-100.0, 100.0))
    value = InputNode("value", 2, value_range=(-100.0, 100.0))
    out = attend_argmax_dot(query_vector, key_vector, value)

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
    pe = _pe()
    query_vector = InputNode("qv", 3, value_range=(-100.0, 100.0))
    key_vector = InputNode("kv", 3, value_range=(-100.0, 100.0))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_argmax_dot(query_vector, key_vector, value)

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
# attend_argmax_dot_where / attend_argmin_dot_where
# ---------------------------------------------------------------------------


def test_attend_argmax_dot_where_validity_dominates_supported_range():
    """A valid low dot score beats an invalid high dot score."""
    query_vector = InputNode("qv", 1, value_range=(-100.0, 100.0))
    key_vector = InputNode("kv", 1, value_range=(-100.0, 100.0))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 2, value_range=(-100.0, 100.0))
    out = attend_argmax_dot_where(query_vector, key_vector, validity, value)

    n_pos = 2
    qv_in = torch.ones(n_pos, 1)
    # At the default match_gain=200, dot magnitudes 4.5 stay inside the
    # documented supported range (abs(match_gain * dot) <= 960).
    kv_in = torch.tensor([[-4.5], [4.5]])
    validity_in = torch.tensor([[1.0], [-1.0]])
    value_in = torch.tensor([[7.0, 1.0], [99.0, 9.0]])

    result = _run(
        out,
        n_pos,
        qv=qv_in,
        kv=kv_in,
        validity=validity_in,
        value=value_in,
    )
    assert torch.allclose(result[1], value_in[0], atol=1e-3), f"pos 1: {result[1]}"


def test_attend_argmin_dot_where_validity_dominates_supported_range():
    """A valid high dot score beats an invalid low dot score."""
    query_vector = InputNode("qv", 1, value_range=(-100.0, 100.0))
    key_vector = InputNode("kv", 1, value_range=(-100.0, 100.0))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 2, value_range=(-100.0, 100.0))
    out = attend_argmin_dot_where(query_vector, key_vector, validity, value)

    n_pos = 2
    qv_in = torch.ones(n_pos, 1)
    kv_in = torch.tensor([[4.5], [-4.5]])
    validity_in = torch.tensor([[1.0], [-1.0]])
    value_in = torch.tensor([[3.0, 4.0], [99.0, 9.0]])

    result = _run(
        out,
        n_pos,
        qv=qv_in,
        kv=kv_in,
        validity=validity_in,
        value=value_in,
    )
    assert torch.allclose(result[1], value_in[0], atol=1e-3), f"pos 1: {result[1]}"


def test_attend_argmin_dot_where_zero_key_regression():
    """Invalid zero keys must not win argmin over valid positive-dot keys."""
    query_vector = InputNode("qv", 2, value_range=(-100.0, 100.0))
    key_vector = InputNode("kv", 2, value_range=(-100.0, 100.0))
    validity = InputNode("validity", 1, value_range=(-100.0, 100.0))
    value = InputNode("value", 3, value_range=(-100.0, 100.0))
    out = attend_argmin_dot_where(query_vector, key_vector, validity, value)

    n_pos = 3
    qv_in = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    # Pos 1 is invalid and has a zero key. Without explicit validity,
    # its dot score 0 would beat the valid positive dot scores under argmin.
    kv_in = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
        ]
    )
    validity_in = torch.tensor([[1.0], [-1.0], [1.0]])
    value_in = torch.eye(3, 3)

    result = _run(
        out,
        n_pos,
        qv=qv_in,
        kv=kv_in,
        validity=validity_in,
        value=value_in,
    )
    assert torch.allclose(result[1], value_in[0], atol=1e-3), f"pos 1: {result[1]}"
    assert torch.allclose(result[2], value_in[0], atol=1e-3), f"pos 2: {result[2]}"


# ---------------------------------------------------------------------------
# attend_most_recent_matching
# ---------------------------------------------------------------------------


def test_most_recent_matching_single_match():
    """When only one causal-window position matches, the output equals
    that position's value."""
    pe = _pe()
    query = InputNode("query", 4, value_range=(-1.0, 1.0))
    key = InputNode("key", 4, value_range=(-1.0, 1.0))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_most_recent_matching(pe, query, key, value)

    n_pos = 5
    # One-hot types, only position 2 has type 0.
    key_in = torch.eye(4)[torch.tensor([1, 2, 0, 3, 1])]
    # All queries look for type 0.
    query_in = torch.eye(4)[torch.tensor([0, 0, 0, 0, 0])]
    value_in = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])

    result = _run(out, n_pos, query=query_in, key=key_in, value=value_in)
    # From pos 2 onwards, position 2 is the only match → value 30.
    for p in range(2, n_pos):
        assert abs(result[p].item() - 30.0) < 1e-2, f"pos {p}: {result[p]}"


def test_most_recent_matching_picks_most_recent_of_ties():
    """With multiple matches, recency breaks the tie."""
    pe = _pe()
    query = InputNode("query", 4, value_range=(-1.0, 1.0))
    key = InputNode("key", 4, value_range=(-1.0, 1.0))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_most_recent_matching(pe, query, key, value)

    n_pos = 6
    # pos 0,3 = type 0; pos 1,4 = type 1; pos 2,5 = type 2.
    key_in = torch.eye(4)[torch.tensor([0, 1, 2, 0, 1, 2])]
    # Each query asks for its own type — so recency matters.
    query_in = torch.eye(4)[torch.tensor([0, 1, 2, 0, 1, 2])]
    # Distinct values so tied matches are distinguishable.
    value_in = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])

    result = _run(out, n_pos, query=query_in, key=key_in, value=value_in)
    # Each position is the most recent match for its own query.
    expected = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])
    assert torch.allclose(
        result, expected, atol=1e-2
    ), f"got {result.squeeze().tolist()}"


def test_most_recent_matching_recency_advances_with_time():
    """As new matches appear, the attended position advances."""
    pe = _pe()
    query = InputNode("query", 4, value_range=(-1.0, 1.0))
    key = InputNode("key", 4, value_range=(-1.0, 1.0))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_most_recent_matching(pe, query, key, value)

    n_pos = 6
    # All positions have type 0 — every position matches every query.
    key_in = torch.eye(4)[torch.tensor([0, 0, 0, 0, 0, 0])]
    query_in = torch.eye(4)[torch.tensor([0, 0, 0, 0, 0, 0])]
    value_in = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

    # Every query's "most recent match" is the query's own position.
    result = _run(out, n_pos, query=query_in, key=key_in, value=value_in)
    for p in range(n_pos):
        assert abs(result[p].item() - float(p + 1)) < 1e-2, f"pos {p}: {result[p]}"


def test_most_recent_matching_e8_type_lookup():
    """Realistic DOOM-style usage: query and key are 10×-scaled E8 codes
    (from ``index_to_vector``), matching SORTED positions among a mix.

    This is the shape M3 will use: RENDER attends to 'most recent
    SORTED_WALL' by feeding token_type as key and a literal E8 code as
    query.  10×-scaled codes have self-dot 1600 and worst off-diagonal
    ~800; default ``match_gain = 200`` gives a 200·800 = 160000 logit
    gap between match and worst no-match, plenty to dominate the
    ~``8·n_pos`` recency swing.
    """
    from torchwright.graph.spherical_codes import index_to_vector

    pe = _pe()
    # Use a few DOOM-used E8 indices.  Only index 3 is treated as "match".
    TYPES = [1, 3, 5, 7, 3, 4, 3]  # positions 1, 4, 6 are type 3
    target_e8 = index_to_vector(3)

    query = InputNode("query", 8, value_range=(-20.0, 20.0))
    key = InputNode("key", 8, value_range=(-20.0, 20.0))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))
    out = attend_most_recent_matching(pe, query, key, value)

    n_pos = 7
    key_in = torch.stack([index_to_vector(t) for t in TYPES])
    # Every query carries the same target E8 code.
    query_in = target_e8.unsqueeze(0).expand(n_pos, -1).contiguous()
    value_in = torch.arange(10.0, 10.0 + n_pos).unsqueeze(1)  # 10, 11, ..., 16

    result = _run(out, n_pos, query=query_in, key=key_in, value=value_in)
    # Type-3 positions: 1, 4, 6.  Their values: 11, 14, 16.
    # From pos 1→3: most recent match is pos 1 → value 11.
    # From pos 4→5: most recent match is pos 4 → value 14.
    # At pos 6: most recent match is pos 6 → value 16.
    expected_vals = [11.0, 11.0, 11.0, 11.0, 14.0, 14.0, 16.0]
    # Positions 0 has no match in the causal prefix → soft-average
    # biased by recency; we don't assert on it.
    for p in range(1, n_pos):
        assert (
            abs(result[p].item() - expected_vals[p]) < 1e-2
        ), f"pos {p}: got {result[p].item()}, expected {expected_vals[p]}"


def test_most_recent_matching_scale_1500_tokens():
    """Scale derisking: with a ~1500-token causal window and sparse
    matches (every ~30 positions), the softmax concentrates cleanly on
    the most recent match.

    This exercises the sizing claim in the docstring — that the
    ``_QUERY_GAIN · n_pos`` recency swing stays well below the match
    gap at the default ``match_gain`` with E8-scaled vectors.
    """
    from torchwright.graph.spherical_codes import index_to_vector

    pe = _pe()
    n_pos = 1500
    match_every = 30  # one match every 30 positions
    target_e8 = index_to_vector(3)
    non_target_e8 = index_to_vector(0)

    query = InputNode("query", 8, value_range=(-20.0, 20.0))
    key = InputNode("key", 8, value_range=(-20.0, 20.0))
    value = InputNode("value", 1, value_range=(-2000.0, 2000.0))
    out = attend_most_recent_matching(pe, query, key, value)

    # Place matches at every `match_every`-th position.
    key_in = torch.stack(
        [target_e8 if (p % match_every == 0) else non_target_e8 for p in range(n_pos)]
    )
    query_in = target_e8.unsqueeze(0).expand(n_pos, -1).contiguous()
    # Encode position into value so we can verify which one we picked.
    value_in = torch.arange(0.0, float(n_pos)).unsqueeze(1)

    result = _run(out, n_pos, query=query_in, key=key_in, value=value_in)
    # Check a sample of query positions across the range.
    for p in (50, 200, 500, 900, 1499):
        expected_pick = (p // match_every) * match_every
        got = result[p].item()
        assert (
            abs(got - float(expected_pick)) < 1e-1
        ), f"pos {p}: got {got}, expected {float(expected_pick)}"


def test_most_recent_matching_value_wider_than_d_pos():
    """Value can be wider than the positional encoding — the compiler
    splits V/O across physical heads, same as ``attend_argmax_dot``."""
    pe = _pe()  # d_pos = 16
    query = InputNode("query", 4, value_range=(-1.0, 1.0))
    key = InputNode("key", 4, value_range=(-1.0, 1.0))
    # Width 24 > d_pos = 16.
    value = InputNode("value", 24, value_range=(-100.0, 100.0))
    out = attend_most_recent_matching(pe, query, key, value)

    n_pos = 3
    key_in = torch.eye(4)[torch.tensor([0, 1, 0])]
    query_in = torch.eye(4)[torch.tensor([0, 0, 0])]
    value_in = torch.arange(0.0, float(n_pos * 24)).reshape(n_pos, 24)

    result = _run(out, n_pos, query=query_in, key=key_in, value=value_in)
    # pos 0: match is pos 0 → value row 0.
    # pos 1: match is pos 0 → value row 0.
    # pos 2: match is pos 2 → value row 2.
    assert torch.allclose(result[0], value_in[0], atol=1e-2)
    assert torch.allclose(result[1], value_in[0], atol=1e-2)
    assert torch.allclose(result[2], value_in[2], atol=1e-2)


def test_most_recent_matching_exclude_self():
    """``exclude_self=True`` skips the current query position even when
    its own key matches — picking the most recent matching position
    strictly before self.

    Setup: every position queries type 0; positions 0, 2, 3 are
    type-0 keys (matches) and position 1 is type 1 (no match).

    Default (exclude_self=False) behaviour:
        pos 1 → match at pos 0  → value 10
        pos 2 → match at pos 2  → value 30   (self matches)
        pos 3 → match at pos 3  → value 40   (self matches)

    With exclude_self=True the head shifts key/value back by one, so
    self can no longer contribute and the predecessor's value is
    returned at the rightmost slot:
        pos 1 → predecessor of pos 1 is pos 0 (match) → value 10
        pos 2 → predecessor of pos 2 is pos 1 (no match); next
                most-recent match below self is pos 0      → value 10
        pos 3 → predecessor of pos 3 is pos 2 (match)      → value 30
    Position 0 is degenerate under the shift (no prior position exists)
    and is not asserted.
    """
    pe = _pe()
    query = InputNode("query", 4, value_range=(-1.0, 1.0))
    key = InputNode("key", 4, value_range=(-1.0, 1.0))
    value = InputNode("value", 1, value_range=(-100.0, 100.0))

    out_default = attend_most_recent_matching(pe, query, key, value)
    out_excl = attend_most_recent_matching(pe, query, key, value, exclude_self=True)

    n_pos = 4
    # types per position: 0, 1, 0, 0 — positions 0, 2, 3 match type 0.
    key_in = torch.eye(4)[torch.tensor([0, 1, 0, 0])]
    # All queries look for type 0.
    query_in = torch.eye(4)[torch.tensor([0, 0, 0, 0])]
    value_in = torch.tensor([[10.0], [20.0], [30.0], [40.0]])

    result_default = _run(
        out_default, n_pos, query=query_in, key=key_in, value=value_in
    )
    result_excl = _run(out_excl, n_pos, query=query_in, key=key_in, value=value_in)

    # Sanity: default mode picks self when self matches.
    assert abs(result_default[1].item() - 10.0) < 1e-2, result_default[1]
    assert abs(result_default[2].item() - 30.0) < 1e-2, result_default[2]
    assert abs(result_default[3].item() - 40.0) < 1e-2, result_default[3]

    # exclude_self mode: self's match doesn't count; we get the most
    # recent strict-prior match instead.
    assert abs(result_excl[1].item() - 10.0) < 1e-2, result_excl[1]
    assert abs(result_excl[2].item() - 10.0) < 1e-2, result_excl[2]
    assert abs(result_excl[3].item() - 30.0) < 1e-2, result_excl[3]


def test_most_recent_matching_exclude_self_accepts_wide_width():
    """``exclude_self=True`` no longer caps key/value width: the pre-shift
    uses ``attend_to_offset``, whose V/O auto-splits across heads.  Both modes
    build for widths wider than the encoding (regression for the removed caps).
    """
    pe = _pe()  # PosEncoding(17), trig_width 16
    query = InputNode("query", 4, value_range=(-1.0, 1.0))
    key = InputNode("key", 4, value_range=(-1.0, 1.0))
    wide_value = InputNode("value", 24, value_range=(-100.0, 100.0))

    # Wide value (24 > trig_width 16) builds in both modes.
    attend_most_recent_matching(pe, query, key, wide_value)
    attend_most_recent_matching(pe, query, key, wide_value, exclude_self=True)

    # Wide key_vector likewise.
    wide_query = InputNode("query", 24, value_range=(-1.0, 1.0))
    wide_key = InputNode("key", 24, value_range=(-1.0, 1.0))
    narrow_value = InputNode("value", 1, value_range=(-100.0, 100.0))
    attend_most_recent_matching(
        pe, wide_query, wide_key, narrow_value, exclude_self=True
    )


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
    """Peel any Assert wrappers off a hard-selection output to reach the Attn."""
    while isinstance(node, Assert):
        node = node.inputs[0]
    assert isinstance(node, Attn), f"expected Attn, got {type(node).__name__}"
    return node


def _baib_nodes(nb, nt, value_width=4, *, assert_hardness_gt=None):
    score = assert_integer(InputNode("baib_score", 1, value_range=(-100.0, 100.0)))
    validity = InputNode("baib_validity", 1, value_range=(-2.0, 2.0))
    key_bucket = InputNode("baib_kb", nb, value_range=(-2.0, 2.0))
    above = InputNode("baib_above", nt, value_range=(-2.0, 2.0))
    query_bucket = InputNode("baib_qb", nb, value_range=(-2.0, 2.0))
    threshold = InputNode("baib_th", nt, value_range=(-2.0, 2.0))
    value = InputNode("baib_value", value_width, value_range=(-100.0, 100.0))
    return attend_argmin_above_in_bucket(
        _pe(),
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
    """Among rows that are valid, in the queried bucket, and above the queried
    threshold, the smallest score wins."""
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
    """A valid, above-threshold row in the WRONG bucket loses to a worse-score
    row in the right bucket."""
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
    """Query bucket k; candidates in k-1, k, k+1 — only the bucket-k row wins,
    even when neighbors have lower scores."""
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
    """score == threshold is NOT strictly above."""
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
    """A wide `value` does not change the logical Q/K width."""
    nb, nt = 3, 4
    for vw in (1, 4, 20):
        out = _baib_nodes(nb, nt, value_width=vw)
        attn = _unwrap_attn(out)
        assert attn.d_qk == 2 + nb + nt, (vw, attn.d_qk)
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
    """Two matching rows that tie on score and carry the same payload produce
    that payload (the blend is a no-op)."""
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
    """Bucket / threshold tables softened to ~0.97 (as compiled values are)
    still select the same row — the predicate-bonus margin absorbs it."""
    nb, nt = 3, 5
    out = _baib_nodes(nb, nt)
    n_pos = 5
    kwargs = dict(
        scores=[6, 4, 8, 3, 5],
        buckets=[1, 1, 1, 0, 1],
        valid=[1, 1, 1, 1, 1],
        thresholds=[0, 1, 2, 3, 4],
        q_bucket=[1] * n_pos,
        q_thresh=[2] * n_pos,
        value_in=torch.eye(n_pos, 4),
        nb=nb,
        nt=nt,
    )
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
    """No row above the queried threshold: output is a finite blend (undefined
    by contract), never NaN, never raises."""
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
    """Negative result: recomputing presence from an AVERAGED scalar bucket id
    is unsafe — two wrong-bucket rows can blend to the query bucket id."""
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
    """Positive result: the robust recomputation — carry the selected
    `key_bucket_onehot` and dot it against the query one-hot — stays low in the
    same no-match blend."""
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
    """With `assert_hardness_gt`, a clean scene with a unique winner per query
    passes the reference-eval hardness predicate (does not raise)."""
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
    """Width regression: logical Q/K width is exactly 2 + n_buckets + n_thresholds."""
    for nb, nt in [(1, 1), (3, 5), (8, 6)]:
        attn = _unwrap_attn(_baib_nodes(nb, nt, value_width=7))
        assert attn.d_qk == 2 + nb + nt, (nb, nt, attn.d_qk)


def test_baib_identity_vo_is_decoupled_from_dqk():
    """V/O regression: value passes through identity matrices of width
    len(value), decoupled from d_qk."""
    nb, nt, vw = 2, 3, 11
    attn = _unwrap_attn(_baib_nodes(nb, nt, value_width=vw))
    assert attn.d_v == vw
    assert torch.equal(attn.value_matrix, torch.eye(vw))
    assert torch.equal(attn.output_matrix, torch.eye(vw))
    # d_qk does not include the value width.
    assert attn.d_qk == 2 + nb + nt


def test_baib_bonus_magnitudes_are_op_local_not_inherited_1000():
    """The validity / bucket / above coefficients baked into the matrices are
    op-local AND large enough to dominate the score swing.  The compiled probe
    cannot tell 256 from a too-small (or 1000) bonus in fp32, so these
    structural checks are what pin the constants."""
    nb, nt = 3, 4
    attn = _unwrap_attn(_baib_nodes(nb, nt))
    q, k = attn.query_matrix, attn.key_matrix
    # col 0: score gain;  col 1: validity (Q=1.0, K=±_VALIDITY_BONUS).
    assert q[0, 0].item() == _QUERY_GAIN
    assert q[0, 1].item() == 1.0
    assert k[1, 1].item() == _VALIDITY_BONUS
    # bucket cols start at 2: query side carries the bonus, key side a bare 1.0.
    for c in range(nb):
        assert q[1 + c, 2 + c].item() == _BUCKET_BONUS
        assert k[2 + c, 2 + c].item() == 1.0
    # above cols start at 2 + n_buckets; same query/key split.
    for c in range(nt):
        assert q[1 + nb + c, 2 + nb + c].item() == _ABOVE_MATCH_BONUS
        assert k[2 + nb + c, 2 + nb + c].item() == 1.0
    # The load-bearing invariant: each bonus must dominate the worst-case
    # gained score swing over the documented range S <= 12, i.e.
    # min(2*_VALIDITY_BONUS, _BUCKET_BONUS, _ABOVE_MATCH_BONUS) > _QUERY_GAIN*12.
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
    """Behavioral pin for the sizing invariant: when the filter-FAILING row
    has the best score (0) and the matching row the worst (30, a near-maximal
    in-range gap under the ~32 diameter), the matching row must still win.  An
    undersized bonus mis-selects here even though the small-gap fixtures stay
    green."""
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
    """n_buckets == 1 and n_thresholds == 1 (the asserted lower bound) still
    select: a 1-wide bucket is a constant no-op filter, a 1-wide threshold
    gates on the single above-column, and argmin-of-score still wins."""
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
    """All-zero query selectors (no bucket / threshold column chosen — e.g. an
    out-of-range query index) drop that filter rather than crash: finite out."""
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
