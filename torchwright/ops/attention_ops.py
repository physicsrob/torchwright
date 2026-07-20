"""Content-based attention primitives built on top of ``graph.Attn``.

Each primitive compiles to a single vanilla attention head. The trick is all
in the Q/K/V weight matrices: by carefully choosing what the attention logit
computes, one head can search for "the position with the smallest score",
"the position with the smallest score among valid positions", or "the
position with the smallest score whose input-index isn't already in a
running mask".

All primitives here follow this template:

- ``query_in`` is an exact constant ``LiteralValue([1.0])`` (and, for the
  ``_above_integer`` / ``_unmasked`` / ``_dot`` variants, also the
  per-query signal that rendezvous with a key-side vector).
  ``query_matrix`` scales that ``1.0`` by ``_QUERY_GAIN`` into column 0 of
  the logit, giving a stable positive per-query gain.
- Every head's content is placed by
  :func:`~torchwright.graph.rope.rotary_content_head` so the match survives the
  global rotation, with the placement **routed by the config's ``d_rot``**: under
  full rotary the content rides the slowest planes of the ``rope.d_head``
  ``rotate_half`` grid (quasi-static, ``cos((i−j)·θ_slow) ≈ 1``); under partial
  rotary (``d_rot < d_head``) it rides the unrotated NoPE tail ``[d_rot:d_head]``,
  an *exact* position-free match (``docs/rope_port_plan.md`` §3).  The builders
  therefore take a :class:`~torchwright.graph.RopeConfig` (carrying ``d_head`` /
  ``d_rot`` / ``base``) where they used to take a ``PosEncoding`` node — position
  is a rotation applied inside attention, no longer a residual feature.
- ``key_in`` contains **only** what ``key_matrix`` reads from: the
  content nodes that drive selection (score, validity, indicators,
  onehot, etc.).  Never concat a node into ``key_in`` without wiring up
  at least one non-zero ``key_matrix`` row for it; ``Attn.__init__``
  enforces this.
- ``value_in`` is whatever node we want to read at the selected key
  position; ``value_matrix`` and ``output_matrix`` are identity projections
  that copy it through unchanged.

Query gain.  ``_QUERY_GAIN = 8`` scales the exact ``1.0`` query constant,
giving a per-unit-score logit delta of 8 → ``exp(8) ≈ 2981`` softmax
weight ratio — ``≥ 99.9 %`` concentration for any integer score gap.  All
current callers operate on integer-valued scores (BSP rank, digit, slot
index) with gap ≥ 1, so the integer-score invariant is what secures hard
selection, not a large gain.

Validity is additive, not gained.  The ``_where`` variants route
validity through a dedicated ``d_qk`` column (``Q = 1.0``,
``K = _VALIDITY_DIRECT``) rather than combining it with the score
column under ``_QUERY_GAIN``.  This keeps worst-case ``|Q·K|`` in the
low thousands instead of tens of thousands, so pre-softmax logits
stay well-resolved in the compiled fp32 path (SDPA MATH backend, TF32 off).

Step-function logits (e.g. strict ``>`` comparisons against a runtime
threshold) are not expressible in bilinear Q·K. The ``_where`` and
``_unmasked`` variants take a pre-computed validity / mask signal as
input rather than synthesising the step function inside the attention op.
"""

import math
from typing import Optional

import torch

from torchwright.graph import Node, Concatenate, Attn, LiteralValue, RopeConfig
from torchwright.graph.asserts import (
    assert_in_range,
    assert_matches_value_type,
    assert_softmax_hardness,
)
from torchwright.graph.rope import (
    rope_inv_freq,
    rope_lobe_band,
    rotary_content_head,
    rotary_offset_head,
    rotary_recency_head,
)
from torchwright.graph.value_type import NodeValueType
from torchwright.ops._math import _theta_slow

# Default tolerance for hard-selection output assertions.  At
# ``_QUERY_GAIN = 8`` the runner-up softmax weight is ``exp(-8) ≈ 3.4e-4``,
# so contamination of the winning value is at most
# ``3.4e-4 × value_range_width``.  For typical sort-digit value widths
# (≤ 10), the observed deviation is ≤ 3e-3; 5e-3 absorbs that plus
# position-scalar PL fuzz.  Callers producing larger-magnitude values
# should pass a larger ``atol``.
_HARD_SELECTION_ATOL = 5e-3


def _wrap_hard_selection_output(
    attn: Attn,
    value: Node,
    atol: float = _HARD_SELECTION_ATOL,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Bake a value-type guarantee onto a hard-selection primitive's output.

    Hard-selection primitives construct Q/K so that softmax concentrates
    overwhelmingly on one key per query, and use identity (or identity-
    embedded) V/O so the selected row passes through unchanged.  Under
    those preconditions the output equals exactly one row of ``value``,
    inheriting its full static ``value_type``.

    This helper attaches a check to the Attn's output that (a) promotes
    the claim statically via the node's ``claimed_type``, and (b) runs a
    runtime predicate during reference_eval checking each claimed
    property to within ``atol`` — the safety net that catches
    construction errors (insufficient gain, score ties, non-identity
    V/O, etc.).

    If ``value.value_type`` is ``unknown``, skips wrapping (no claim to
    promote, no predicate to run).

    When ``assert_hardness_gt`` is set, also wraps the output in a
    softmax hardness assert that checks the maximum attention weight
    per query exceeds the threshold.
    """
    result: Node = attn
    if assert_hardness_gt is not None:
        result = assert_softmax_hardness(result, assert_hardness_gt)
    vt = value.value_type
    if vt == NodeValueType.unknown():
        return result
    return assert_matches_value_type(result, vt, atol=atol)


# Coefficient applied to the exact ``1.0`` query constant inside the query
# projection, so ``Q[·, 0] = _QUERY_GAIN`` independent of query position. A
# unit score delta then produces a logit delta of 8 → ``exp(8) ≈ 2981``
# softmax weight ratio, i.e. ``≥ 99.9 %`` concentration. All current callers
# produce integer-valued scores with gap ≥ 1, so 8 is sufficient; larger
# gains (e.g. 80) are historical and bought only unused margin.  The compiled
# path runs fp32 through the SDPA MATH backend with TF32 off, so the
# discriminating gap resolves with ample headroom regardless — keeping the
# gain modest is about not needing more, not a precision ceiling.
_QUERY_GAIN = 8.0

# Direct (not gained) logit bonus for valid positions in the simple
# scalar ``_where`` variants (``attend_argmin_where``,
# ``attend_argmax_where``, ``attend_mean_where``) and the dot-product
# ``_dot_where`` variants.  Routed through a dedicated ``d_qk`` column
# with ``Q = 1.0`` and ``K = ± _VALIDITY_DIRECT``, so the contribution
# to the logit is literally ``± _VALIDITY_DIRECT`` — not multiplied by
# ``_QUERY_GAIN``.  Must exceed the one-sided score swing
# ``_QUERY_GAIN · _MAX_SCORE_ABS = 8 × 120 = 960`` so validity
# dominates score; ``1000`` buys a small but sufficient margin.
_VALIDITY_DIRECT = 1000.0

# Key-side validity coefficient for ``attend_argmin_valid_unmasked``,
# which keeps the *multiplicative* (gained) validity encoding because
# its mask_vector input can accumulate integer values above 1 (see the
# op's docstring for why).  Under the gain, the effective validity
# logit contribution is ``_QUERY_GAIN · _VALIDITY_KEY_COEFF = 8000``,
# giving ``2 · 8000 = 16000`` of validity swing — enough to dominate
# ``_UNMASKED_PENALTY · max_walls`` for typical max_walls ≤ 15.
_VALIDITY_KEY_COEFF = 1000.0

# Maximum ``|score|`` supported by these primitives. With gain=8, the
# worst valid-position logit contribution from score is ``8 × 120 =
# 960``, under the 1000-unit ``_VALIDITY_DIRECT`` bonus (or well under
# the 8000-unit gained ``_VALIDITY_KEY_COEFF`` contribution in
# ``attend_argmin_valid_unmasked``).
_MAX_SCORE_ABS = 120.0

# Maximum absolute dot-product contribution supported by
# ``attend_argmax_dot_where`` / ``attend_argmin_dot_where`` *after*
# multiplying by ``match_gain``.  Validity contributes
# ``± _VALIDITY_DIRECT`` directly to the logit, so keeping the match
# term inside ±960 leaves the same 40-logit margin as the scalar
# ``_where`` variants in the worst valid-vs-invalid case.
_MAX_DOT_LOGIT_ABS = 960.0

# Penalty (in *logit* space, not key space) applied by
# ``attend_argmin_unmasked`` to masked positions. Must exceed
# ``_QUERY_GAIN * _MAX_SCORE_UNMASKED_ABS`` so a masked position with
# the best score still loses to an unmasked position with the worst
# score. With gain=8 and max_score=100: 8×100 = 800, so 1000 gives
# ~25% margin.
_UNMASKED_PENALTY = 1000.0

# Maximum ``|score|`` supported by ``attend_argmin_unmasked``.
_MAX_SCORE_UNMASKED_ABS = 100.0

# Bonus applied to "above threshold" positions in
# ``attend_argmin_above_integer``. Added directly to the logit (not
# scaled by _QUERY_GAIN, because the bonus rendezvous goes straight into
# its own query_matrix columns rather than through the _QUERY_GAIN column).
# Must exceed ``_QUERY_GAIN · (max_score - min_score)`` so a valid
# position with the worst score still beats any invalid position with
# the best score.
#
# For ``score ∈ [0, 9]`` (the sort_digits toy) a bonus of 100 buys
# ~40% margin.  For production use with piecewise-linear softmax under
# residual-stream noise, 1000 matches _VALIDITY_DIRECT's headroom —
# both route directly through the logit (not through the gained score
# column) and both need to dominate noise from competing attention
# values in the compiled residual.
_ABOVE_BONUS = 1000.0


# Op-local predicate bonuses for ``attend_argmin_above_in_bucket``.  That op
# stacks THREE predicate filters (validity, bucket, strict-above) on one
# head; each bonus is routed directly into its own ``d_qk`` column, so its
# logit contribution is the bare constant.  Each must dominate a single
# predicate miss against that op's worst-case gained score swing
# ``_QUERY_GAIN · S``; the downstream range is ``S <= 12`` (so ``96``), and
# ~256 gives a ~2.7x margin.  These are an order of magnitude below the
# single-bonus ``_VALIDITY_DIRECT`` / ``_ABOVE_BONUS`` (1000) because those
# were sized for a 10x larger swing (``_QUERY_GAIN · _MAX_SCORE_ABS = 960``),
# NOT for a precision budget — the compiled path is fp32 (SDPA MATH backend,
# TF32 off), where the gained unit-score gap resolves with thousands of ULP
# of margin at 256 or 1000 alike.  Note the distinct name
# ``_ABOVE_MATCH_BONUS`` (NOT ``_ABOVE_BONUS``): a second module-level
# ``_ABOVE_BONUS`` would rebind the constant above and undersize
# ``attend_argmin_above_integer``.
_VALIDITY_BONUS = 256.0
_BUCKET_BONUS = 256.0
_ABOVE_MATCH_BONUS = 256.0


# Default coefficient on the intrinsic rotary recency lobe (the local "most
# recent" tiebreak in ``get_prev_value``, the lobe's only remaining caller since
# ``attend_most_recent_matching`` was retired for
# :func:`attend_most_recent_globally`).
# The lobe peak is ``Σ amp_p ≈ 42.6`` (:func:`~torchwright.graph.rope.rope_lobe_band`
# at the production config), so ``recency_gain · peak ≈ 2.5e4``.  Sized against
# the *smallest* near-Δ step over the working window: the Hann taper rounds the
# peak, so the Δ=0→1 step (self vs immediate predecessor, normalized ``≈ 3.0e-4``)
# is smaller than the Δ=1→2 gap-1 step (``≈ 8.8e-4``); at ``600`` the worst step
# over the ~100-token target is ``≈ 7.5`` logits → ``exp(7.5) ≈ 1.8e3`` softmax
# ratio (≥ 99.9 % concentration).  Unlike the old octant ramp's ``rank_gain ≈
# 2e5``, the lobe is bounded (does not grow with sequence length), so the content
# gate that must dominate it is small (``get_prev_value`` sets it automatically).
_LOCAL_RECENCY_GAIN = 600.0

# Ceiling on the summed softmax weight that false-cond keys may steal from a
# true key in ``get_prev_value``, enforced at build time.  The ±1 cond swing
# rides the slowest rotated plane, so a key at distance ``Δ`` contributes
# ``±gate·cos(Δ·θ_slow)`` — the bound therefore has two enforced inputs, not
# one assumption: (1) ``max_positions · θ_slow < π/2`` (past that a distant
# false key's cosine goes negative and the cond ordering can invert outright
# — a separate build-time raise), and (2) with every cosine then at least
# ``cos_floor = cos(max_positions·θ_slow) > 0``, each false key's logit sits
# at least ``2·gate·cos_floor`` below any true key's, so its weight against
# the winner is at most ``exp(−2·gate·cos_floor)`` and the total over at most
# ``max_positions`` keys is ``max_positions · exp(−2·gate·cos_floor)``.
# (Pre-2026-07 this bound used the bare ``2·gate``, silently assuming the
# rotation factor ≈ 1; a config could pass it while the slow-plane rotation
# flipped the gate's sign well inside the rollout.)  The tolerance
# is sized for the harshest consumer: ``attend_mean_where`` multiplies a
# latched validity by ``_VALIDITY_DIRECT`` (1000), so a leak-driven validity
# wobble ``ε`` becomes a ``1000·ε`` per-key logit tilt that skews the
# "uniform" mean (the failure mode that broke ``count_since_marker`` at
# ``d_head=32, max_positions=64``: a 2-plane band Hann-tapers to its endpoint
# zeros, leaving ``Σ amp ≈ 2e-3``, gate ≈ 4.8, leak ≈ 4e-3 — the count read
# 2.8 at a true gap of 4).  At 1e-6 total leak the tilt is ≤ 2e-3 logits —
# invisible — while every healthy config passes by hundreds of orders of
# magnitude (gate ≥ ~2400 ⇒ the bound underflows to 0) and the observed
# degenerate config fails by ~3600×.
_MAX_RECENCY_LEAK = 1e-6

# Per-position logit gain for the position tiebreak in attend_most_recent_globally.
# Each unit of absolute position contributes recency_scale to the attention logit
# so adjacent positions differ by recency_scale in logit.
#
# Constraints the caller must verify:
#   (1) representability: recency_scale >> ULP(content_score)
#       For E8 content (score≈320000): ULP≈0.038, so recency_scale=1.0 gives 26× margin.
#   (2) content dominance: match_gain · min_match_dot_gap > recency_scale · max_positions
#       For E8 (match_gain=200, dot_gap=1600): 320000 > 1.0 · 61440 ✓ (5.2× margin).
_RECENCY_SCALE = 1.0


def attend_argmin_above_integer(
    rope: RopeConfig,
    score: Node,
    indicators_above: Node,
    threshold_onehot: Node,
    value: Node,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Argmin of ``score`` among positions strictly above a runtime threshold.

    This is the "next-above-threshold" selection primitive used by the
    V1 sort variant. It solves the pattern "give me the smallest score
    strictly greater than a threshold that's known per-query-position"
    under the vanilla-attention constraint.

    The key trick is an **indicator basis** on the key side. Because
    the comparison ``score_i > threshold_j`` is a step function that
    mixes per-key and per-query info, bilinear ``Q·K^T`` cannot compute
    it directly. Instead, for a fixed integer-valued score with a known
    set of possible thresholds ``{t_0, …, t_{N-1}}``, we precompute at
    each key position a width-``N`` indicator vector
    ``indicators_above[c] = I(score_i > t_c)``. The query side provides
    a one-hot ``threshold_onehot`` selecting which threshold applies at
    this query position. The bilinear sum then evaluates to

        Σ_c threshold_onehot_j[c] · indicators_above_i[c]
            = I(score_i > threshold_j)

    exactly, entirely inside the attention head.

    Combining this with a ``-score_i`` term in column 0 and a large
    above-bonus in columns ``1..N`` gives an attention logit of

        _QUERY_GAIN · (−score_i)
            + _ABOVE_BONUS_LOGIT · I(score_i > threshold_j)

    whose argmax (argmin of score) is the smallest-score position
    strictly above the threshold.

    Caller responsibilities.

    - ``indicators_above`` must be built once at each key position.
      The caller decides the set of possible thresholds. A typical
      pattern for sorting digits 0..9: precompute
      ``[I(digit > -1), I(digit > 0), I(digit > 1), …, I(digit > 8)]``
      (width 10), one slot per possible ``prev_digit`` value in
      ``{-1, 0, 1, …, 8}``.
    - ``threshold_onehot`` must be a ``{0, 1}`` one-hot of the same
      width, with exactly one entry set to 1 at each query position
      indicating which threshold that position uses.
    - If the query's threshold is such that no valid position exists
      (e.g. threshold 9 with max digit 9), the output is garbage — wrap
      the result in a ``select`` against a sentinel at the call site.

    Compile cost: one vanilla attention head. The width of the
    indicator basis determines ``d_head`` (roughly
    ``1 + N + len(value)``).

    Args:
        rope: RoPE config (``d_head`` / ``base``) for the slow-plane placement.
        score: 1D scalar node (score at each key position).
        indicators_above: Width-``N`` node where slot ``c`` is the
            precomputed indicator ``I(score_i > threshold_c)``.
        threshold_onehot: Width-``N`` node whose value at each query
            position is a one-hot selecting the active threshold.
        value: Node to read at the selected key position.

    Returns:
        Attn node of width ``len(value)``.
    """
    assert len(score) == 1, "attend_argmin_above_integer expects a 1D scalar score"
    assert len(indicators_above) == len(threshold_onehot), (
        "indicators_above and threshold_onehot must have the same width "
        f"(got {len(indicators_above)} and {len(threshold_onehot)})"
    )
    n_thresholds = len(indicators_above)
    # Content Q/K layout (relocated onto slow planes by rotary_content_head;
    # value rides identity V/O, decoupled from this width):
    #   col 0:                  score logit
    #   cols 1..n_thresholds:   threshold_onehot · indicators_above terms
    W = 1 + n_thresholds

    # Query: an exact 1.0 for the score gain (col 0) plus the threshold
    # one-hot for the bilinear above-test.  No position info needed.
    query_one = LiteralValue(torch.tensor([1.0]), name="above_query_one")
    query_in = Concatenate([query_one, threshold_onehot])
    key_in = Concatenate([score, indicators_above])

    # --- Query matrix, shape (1 + n_thresholds, W) ---
    query_matrix = torch.zeros((len(query_in), W))
    # Col 0: stable positive gain for the scoring logit (from the 1.0 literal).
    query_matrix[0, 0] = _QUERY_GAIN
    # Cols 1..n_thresholds: _ABOVE_BONUS · threshold_onehot[c] routed
    # to the matching column for the bilinear rendezvous with
    # indicators_above on the key side.
    for c in range(n_thresholds):
        query_matrix[1 + c, 1 + c] = _ABOVE_BONUS

    # --- Key matrix, shape (1 + n_thresholds, W) ---
    key_matrix = torch.zeros((len(key_in), W))
    score_row = 0
    indicators_start_row = 1
    # Col 0: -score.
    key_matrix[score_row, 0] = -1.0
    # Cols 1..n_thresholds: each indicator_above column.
    for c in range(n_thresholds):
        key_matrix[indicators_start_row + c, 1 + c] = 1.0

    attn = rotary_content_head(
        query_in,
        key_in,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        d_rot=rope.d_rot,
        base=rope.base,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def attend_argmin_above_in_bucket(
    rope: RopeConfig,
    score: Node,
    validity: Node,
    key_bucket_onehot: Node,
    score_above_each_threshold: Node,
    query_bucket_onehot: Node,
    threshold_onehot: Node,
    value: Node,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Smallest-``score`` row that is valid, in a requested bucket, and above
    a requested threshold — read its ``value``.

    One vanilla attention head.  For each query position it selects, among
    all earlier positions ("rows"), the one with the minimum ``score`` that
    passes three per-row filters, and returns that row's ``value``::

        valid:      validity == +1   (torchwright's +1 / -1 convention)
        in bucket:  the row's bucket == the bucket this query asks for
        above:      score > the threshold this query asks for

    The requested bucket and threshold are chosen per query position, at
    runtime, via one-hot selectors.

    Why bucket / threshold arrive as tables, not numbers.  A bilinear
    ``Q·K`` logit can multiply and add but cannot test equality or a strict
    ``>``.  So each of those two filters is a small *table of pre-answered
    yes/no questions* — one row per position, one column per choice — and
    the query names a column; the attention dot product reads out that one
    cell::

        bucket table (equality)        threshold table (greater-than)
              g0 g1 g2 g3                     >t0 >t1 >t2 >t3 >t4
        s(g2)  0  0  1  0               s(5)   1   1   1   1   1
        s(g0)  1  0  0  0               s(2)   1   1   0   0   0

    A bucket row is a one-hot (you are in exactly one group); a threshold
    row is a run of 1s that stops at the score (you are above every
    threshold up to your score).  ``key_bucket_onehot`` /
    ``score_above_each_threshold`` are the key-side table rows;
    ``query_bucket_onehot`` / ``threshold_onehot`` are the query-side column
    pickers.  ``dot(query_bucket_onehot, key_bucket_onehot)`` is 1 iff the
    groups match; ``dot(threshold_onehot, score_above_each_threshold)``
    reads the precomputed "score > threshold" answer.

    The attention logit at ``(query j, key i)`` is::

        _QUERY_GAIN·(−score_i)
            + _VALIDITY_BONUS·validity_i
            + _BUCKET_BONUS·dot(query_bucket_onehot_j, key_bucket_onehot_i)
            + _ABOVE_MATCH_BONUS·dot(threshold_onehot_j, score_above_each_threshold_i)

    The three bonuses dominate, so only rows passing all three filters
    compete; the small ``−score`` term then picks the smallest score.

    Layout.  Decoupled identity V/O — ``value`` passes through unchanged and
    may be any width without enlarging the head's logical Q/K width
    (``d_qk = 2 + n_buckets + n_thresholds``); the compiler splits a wide
    ``value`` over ``ceil(d_v / d_head)`` physical heads.  Compile with
    ``d_head >= d_qk``.

    No-match is undefined.  If no row passes all three filters the output is
    a soft blend of whatever is present, NOT a sentinel — this op does not
    report presence.  Prove a match exists upstream, or carry the selected
    ``validity`` / ``key_bucket_onehot`` / ``score_above_each_threshold`` in
    ``value`` and re-test them after attention (a blend of non-matching
    one-hots dotted against the query one-hot stays well below 1).  Tied
    scores blend their ``value`` payloads.

    Args:
        rope: RoPE config (``d_head`` / ``base``) for the slow-plane placement;
            the content match is purely bilinear, the rotation rides quasi-
            static planes.
        score: width-1 scalar; lower wins.  Integer-valued for hard
            selection.  Bounded score DIAMETER: a predicate bonus only
            dominates the gained score term while ``_QUERY_GAIN *
            (max_score - min_score)`` stays below it, so the supported
            diameter is ``min(2*_VALIDITY_BONUS, _BUCKET_BONUS,
            _ABOVE_MATCH_BONUS) / _QUERY_GAIN`` — ``32`` at the shipped
            constants (the bucket / above bound; validity holds to 64).
            Past that, a filter-failing low-score row can silently outscore
            a matching high-score row (no error, no NaN).  The Task 3 range
            (``S <= 12``) sits well inside it.
        validity: width-1, +1 valid / -1 invalid.
        key_bucket_onehot: width ``n_buckets``; the row's own bucket as a
            0/1 one-hot.
        score_above_each_threshold: width ``n_thresholds``; slot ``c`` is 1
            iff ``score`` is strictly above threshold ``c`` (a monotone run
            of 1s — NOT a one-hot).
        query_bucket_onehot: width ``n_buckets``; the requested bucket.
        threshold_onehot: width ``n_thresholds``; the requested threshold.
        value: payload (any width) read at the selected row.
        assert_hardness_gt: optional softmax-hardness floor checked at
            reference-eval / ``debug=True`` time; keep it ``<= ~0.9997``,
            the adjacent-score ceiling ``softmax(_QUERY_GAIN·1)``.

    Returns:
        Attn node of width ``len(value)``.
    """
    assert len(score) == 1, "attend_argmin_above_in_bucket expects a width-1 score"
    assert (
        len(validity) == 1
    ), "attend_argmin_above_in_bucket expects a width-1 validity"
    assert len(key_bucket_onehot) == len(query_bucket_onehot), (
        "key_bucket_onehot and query_bucket_onehot must have the same width "
        f"(got {len(key_bucket_onehot)} and {len(query_bucket_onehot)})"
    )
    assert len(score_above_each_threshold) == len(threshold_onehot), (
        "score_above_each_threshold and threshold_onehot must have the same "
        f"width (got {len(score_above_each_threshold)} and {len(threshold_onehot)})"
    )
    assert len(value) >= 1, "value must be non-empty"
    n_buckets = len(key_bucket_onehot)
    n_thresholds = len(score_above_each_threshold)
    assert n_buckets >= 1, "n_buckets must be >= 1"
    assert n_thresholds >= 1, "n_thresholds must be >= 1"

    d_qk = 2 + n_buckets + n_thresholds
    d_v = len(value)

    # d_qk column layout:
    #   col 0:                         score logit  (Q = _QUERY_GAIN, K = -score)
    #   col 1:                         validity     (Q = 1.0, K = ± _VALIDITY_BONUS)
    #   cols 2 .. 1+n_buckets:         bucket-equality rendezvous
    #   cols 2+n_buckets .. d_qk-1:    strict-above rendezvous
    bucket_col0 = 2
    above_col0 = 2 + n_buckets

    # query_in rows: [1.0 literal (1), query_bucket_onehot (n_buckets),
    #                 threshold_onehot (n_thresholds)]
    query_one = LiteralValue(torch.tensor([1.0]), name="bucketed_argmin_query_one")
    query_in = Concatenate([query_one, query_bucket_onehot, threshold_onehot])
    # key_in rows: [score (1), validity (1), key_bucket_onehot (n_buckets),
    #               score_above_each_threshold (n_thresholds)]
    key_in = Concatenate(
        [score, validity, key_bucket_onehot, score_above_each_threshold]
    )

    # --- Query matrix, shape (len(query_in), d_qk) ---
    query_matrix = torch.zeros((len(query_in), d_qk))
    # Literal 1.0 (row 0) → score gain (col 0) and direct validity Q (col 1).
    query_matrix[0, 0] = _QUERY_GAIN
    query_matrix[0, 1] = 1.0
    # query_bucket_onehot (rows 1..n_buckets) → bucket cols, coeff _BUCKET_BONUS.
    for c in range(n_buckets):
        query_matrix[1 + c, bucket_col0 + c] = _BUCKET_BONUS
    # threshold_onehot (rows 1+n_buckets..) → above cols, coeff _ABOVE_MATCH_BONUS.
    for c in range(n_thresholds):
        query_matrix[1 + n_buckets + c, above_col0 + c] = _ABOVE_MATCH_BONUS

    # --- Key matrix, shape (len(key_in), d_qk) ---
    key_matrix = torch.zeros((len(key_in), d_qk))
    score_row = 0
    validity_row = 1
    bucket_row0 = 2
    above_row0 = 2 + n_buckets
    key_matrix[score_row, 0] = -1.0  # col 0: -score
    key_matrix[validity_row, 1] = _VALIDITY_BONUS  # col 1: ± _VALIDITY_BONUS
    for c in range(n_buckets):
        key_matrix[bucket_row0 + c, bucket_col0 + c] = 1.0
    for c in range(n_thresholds):
        key_matrix[above_row0 + c, above_col0 + c] = 1.0

    # Content relocated onto slow planes; identity V/O passes ``value`` through
    # unchanged and may be wider than d_qk (compiler splits over physical heads).
    attn = rotary_content_head(
        query_in,
        key_in,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        d_rot=rope.d_rot,
        base=rope.base,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def attend_argmin_unmasked(
    rope: RopeConfig,
    score: Node,
    mask_vector: Node,
    position_onehot: Node,
    value: Node,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Argmin of ``score`` over positions whose index isn't set in the mask.

    This is the primitive that the V4 selection-sort variant exists to
    motivate. It solves the pattern "I have a per-query-position mask
    vector that says which *input-position indices* have already been
    emitted, and I want to find the smallest-score input position whose
    index isn't in the mask yet."

    The mask / position-onehot rendezvous. At each *key* position ``i``
    we precompute a one-hot ``position_onehot_i`` of width ``N``, where
    ``N`` is the number of distinguishable input slots. At each *query*
    position ``j`` we carry a width-``N`` mask vector ``mask_vector_j``
    whose bit ``c`` is 1 iff input slot ``c`` has already been selected
    at or before ``j``. The attention logit at ``(j, i)`` then has the
    shape

        _QUERY_GAIN · (−score_i)
        −_UNMASKED_PENALTY · mask_vector_j[position_i]

    The first term lives in ``d_head`` column 0 as usual. The second term
    is built by putting ``−_UNMASKED_PENALTY · mask_vector_j[c]`` in
    ``Q[j, c+1]`` and ``position_onehot_i[c]`` in ``K[i, c+1]`` for each
    ``c ∈ {0, …, N-1}``. The bilinear sum then equals
    ``−_UNMASKED_PENALTY · mask_vector_j[position_i]`` — a very negative
    penalty exactly when the query's mask has a bit set at the key's
    position index. An additional slab of ``d_head`` columns at the end
    carries ``value`` through unchanged.

    **Score constraint.** Scores must satisfy
    ``|score| <= _MAX_SCORE_UNMASKED_ABS`` (= 100). This is tighter than
    the ``_MAX_SCORE_ABS`` constraint on the other primitives because the
    penalty machinery needs headroom to let an unmasked worst-score
    position still beat a masked best-score position. If you need to sort
    larger-magnitude scores, normalise them first.

    **Score uniqueness and stable sort.** The caller controls stability.
    For a selection-sort over possibly-duplicate input items, use a
    lexicographic score like ``digit_i · N + pos_scalar_i`` so the
    scores are distinct — otherwise the softmax will weighted-average
    duplicate-score unmasked positions.

    **When every unmasked position is exhausted.** If the mask covers
    every causally-visible position, the best remaining logit is
    ``−_UNMASKED_PENALTY`` which equals ``-10000`` — still above the
    ``CAUSAL_MASK_SENTINEL``, so the attention will return the
    weighted-average of the *least-bad* masked positions rather than
    wandering into "future" positions. Callers that care about this edge
    case should wrap the result in a ``select`` against a sentinel.

    Compile cost: exactly one vanilla attention head. The cost is
    primarily in d_head, which grows with the mask width
    (``d_head = 1 + N + len(value)``).

    Args:
        rope: RoPE config (``d_head`` / ``base``) for the slow-plane placement.
        score: 1D scalar node.
        mask_vector: Width-``N`` node whose value at query position ``j``
            is a ``{0, 1}`` mask — bit ``c`` set means "input slot ``c``
            has been emitted already, skip it".
        position_onehot: Width-``N`` node whose value at key position
            ``i`` is the one-hot of that position's *input slot index*.
            Must have the same width as ``mask_vector``.
        value: Node to read.  Any width — V/O is identity and the compiler
            splits a wide value across physical heads.

    Returns:
        Attn node of width ``len(value)`` equal to ``value`` at the
        unmasked argmin-of-``score`` position within the causal window.
    """
    assert len(score) == 1, "attend_argmin_unmasked expects a 1D scalar score"
    assert len(mask_vector) == len(position_onehot), (
        "mask_vector and position_onehot must have the same width "
        f"(got {len(mask_vector)} and {len(position_onehot)})"
    )
    n_slots = len(mask_vector)
    # Content Q/K layout (relocated onto slow planes; value rides identity V/O):
    #   col 0:               score logit
    #   cols 1 .. n_slots:   mask · position_onehot dot-product terms
    W = 1 + n_slots

    # Query: an exact 1.0 for the score gain (col 0) plus the per-query mask.
    query_one = LiteralValue(torch.tensor([1.0]), name="unmasked_query_one")
    query_in = Concatenate([query_one, mask_vector])
    key_in = Concatenate([score, position_onehot])

    # --- Query matrix, shape (1 + n_slots, W) ---
    query_matrix = torch.zeros((len(query_in), W))
    # Col 0: stable positive gain from the exact 1.0 literal.
    query_matrix[0, 0] = _QUERY_GAIN
    # Cols 1 .. n_slots: -_UNMASKED_PENALTY · mask_vector[c].
    for c in range(n_slots):
        query_matrix[1 + c, 1 + c] = -_UNMASKED_PENALTY

    # --- Key matrix, shape (1 + n_slots, W) ---
    key_matrix = torch.zeros((len(key_in), W))
    # Row order in key_in: [score (1), onehot (n_slots)]
    score_row = 0
    onehot_start_row = 1
    # Col 0: -score.
    key_matrix[score_row, 0] = -1.0
    # Cols 1 .. n_slots: position_onehot[c]  (identity into cols 1..n_slots).
    for c in range(n_slots):
        key_matrix[onehot_start_row + c, 1 + c] = 1.0

    attn = rotary_content_head(
        query_in,
        key_in,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        d_rot=rope.d_rot,
        base=rope.base,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def attend_mean_where(
    rope: RopeConfig,
    validity: Node,
    value: Node,
) -> Node:
    """Uniform mean of ``value`` across positions where ``validity`` is true.

    At each query position, the attention returns the uniform average of
    ``value`` over all causally-visible positions where ``validity`` is
    +1.  Invalid positions (``validity`` = −1) receive a large negative
    logit penalty and contribute negligibly to the output.

    All valid positions share the same logit (no score term), so softmax
    assigns them equal weight — producing an exact mean rather than a
    weighted combination.

    A typical use is reduce-any: map boolean flags to 0/1 with
    :func:`~torchwright.ops.relu.arithmetic_ops.bool_to_01`, average them
    with this primitive, and threshold the result.

    **When no position is valid.** The softmax still runs and produces a
    weighted average over all positions — effectively garbage.  Callers
    must ensure at least one valid position exists within the causal
    window at every query position whose output is consumed.

    Compile cost: one attention head (auto-split across multiple
    physical heads by the compiler when ``d_v > d_head``).
    ``d_qk = 1``, ``d_v = len(value)``.

    Args:
        rope: RoPE config (``d_head`` / ``base``) for the slow-plane placement.
        validity: 1D boolean node (+1 valid, −1 invalid).
        value: Node to average.  No width constraint — the compiler
            splits wide V/O across multiple physical heads.

    Returns:
        Attn node of width ``len(value)`` equal to the uniform mean of
        ``value`` across valid key positions in the causal window.

    See also:
        :func:`attend_argmin_where` — selects one position (min score)
        instead of averaging.
    """
    assert len(validity) == 1, "attend_mean_where expects a 1D boolean validity"

    # Content width 1: the only column carries the direct validity bonus,
    # relocated onto the slowest plane by rotary_content_head so the rotation
    # is quasi-static (cos((i−j)·θ_slow) ≈ 1) over a bounded valid window and
    # the mean stays uniform.  Q is an exact 1.0 (a LiteralValue) — unscaled,
    # since validity here is a direct logit contribution, not combined with any
    # gained score.  K reads only validity.  All valid positions get the same
    # logit → uniform softmax → mean.
    d_qk = 1

    query_one = LiteralValue(torch.tensor([1.0]), name="mean_where_query_one")
    query_matrix = torch.zeros((1, d_qk))
    query_matrix[0, 0] = 1.0

    # Only validity drives K — no additional position feature in key_in.
    key_matrix = torch.zeros((len(validity), d_qk))
    key_matrix[0, 0] = _VALIDITY_DIRECT

    attn = rotary_content_head(
        query_one,
        validity,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        d_rot=rope.d_rot,
        base=rope.base,
    )
    # Mean of values in [lo, hi] stays in [lo, hi] (convex combination),
    # but integer-ness / binary-ness / one-hot-ness do not survive the
    # soft mean.  Only promote the range claim.
    r = value.value_type.value_range
    if math.isfinite(r.lo) and math.isfinite(r.hi):
        return assert_in_range(attn, r.lo, r.hi, atol=_HARD_SELECTION_ATOL)
    return attn


def attend_causal_mean(
    rope: RopeConfig,
    value: Node,
    *,
    output_scale: float = 1.0,
    claim_range: bool = True,
) -> Node:
    """Uniform mean of ``value`` over ALL causally-visible positions.

    At each query position t, returns ``output_scale`` times the uniform
    average of ``value`` over positions 0..t (the current position included).
    Every position always participates — there is no validity signal; use
    :func:`attend_mean_where` to average a subset.

    **Exactly uniform on every rotary layout.**  The Q and K projection
    matrices are entirely zero, so every pre-softmax logit is exactly 0 —
    the rotation of a zero vector is zero — and the softmax is an exact
    ``1/(t+1)`` under full rotary, partial rotary, and any ``d_rot``.  This
    is stronger than :func:`attend_mean_where` with a constant-true
    validity, whose relocated validity column is exactly position-free only
    under partial rotary (on the NoPE tail) and merely quasi-static under
    full rotary.

    ``output_scale`` folds a scalar multiply into the O projection — e.g.
    the smoothed global position (``global_position_from_bos(...,
    smoothed=True)``) uses ``output_scale=2.0`` so ``2 × mean(0..t) ≈ t``
    costs no extra sublayer.

    Compile cost: one attention head (auto-split across multiple physical
    heads when ``d_v > d_head``).  ``d_qk = rope.d_head``, ``d_v = len(value)``.

    Args:
        rope: RoPE config (``d_head`` / ``d_rot`` / ``base``) — fixes the
            head geometry; the zero projections make the output independent
            of it.
        value: node to average.  No width constraint.
        output_scale: scalar folded into the output projection.
        claim_range: promote the value's (scaled) range claim onto the
            output.  Pass ``False`` when the input's compiled values are
            allowed to sit slightly outside their claimed range (an assert
            slack the convexity claim cannot see) and attach a caller-side
            claim with a justified tolerance instead.

    Returns:
        Attn node of width ``len(value)`` equal to ``output_scale`` times the
        uniform causal mean of ``value``.

    See also:
        :func:`attend_mean_where` — validity-selected mean (one subset).
    """
    query_one = LiteralValue(torch.tensor([1.0]), name="causal_mean_query_one")
    d_v = len(value)
    attn = Attn(
        query_in=query_one,
        key_in=query_one,
        value_in=value,
        query_matrix=torch.zeros((1, rope.d_head)),
        key_matrix=torch.zeros((1, rope.d_head)),
        value_matrix=torch.eye(d_v),
        output_matrix=output_scale * torch.eye(d_v),
        rope_base=rope.base,
        rope_d_rot=rope.d_rot,
    )
    # Convexity promotes the value's range claim, scaled by output_scale
    # (positive scale assumed for the lo/hi order).  The atol scales with the
    # output: the mean's own fp32 accumulation error grows with value
    # magnitude and count (measured ~1.5e-2 on position-scale values at 54k
    # positions), but the claim only binds at the range ends, where the mean
    # sits only when nearly every averaged value does.
    r = value.value_type.value_range
    if claim_range and output_scale > 0 and math.isfinite(r.lo) and math.isfinite(r.hi):
        return assert_in_range(
            attn,
            output_scale * r.lo,
            output_scale * r.hi,
            atol=_HARD_SELECTION_ATOL * max(1.0, output_scale),
        )
    return attn


def attend_argmax_dot(
    rope: RopeConfig,
    query_vector: Node,
    key_vector: Node,
    value: Node,
    match_gain: float = 200.0,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Argmax of a vector dot-product score.

    At each query position, the attention returns ``value`` at the
    causal-window position whose ``key_vector`` has the highest dot
    product with ``query_vector``.  When multiple positions share the
    highest dot product, the output is a soft average of their values.

    The logit at key position ``i`` seen from query position ``j`` is

        match_gain · (query_vector_j · key_vector_i)

    **Type isolation.**  This primitive does not include a validity
    parameter.  Callers should use
    :func:`~torchwright.ops.relu.logic_ops.cond_gate` to zero out
    ``key_vector`` and ``value`` at non-participating positions.  A
    zero ``key_vector`` produces a dot product of 0, well below
    ``match_gain`` for any matching position — providing effective type
    isolation without a separate validity signal.

    Compile cost: one attention head (auto-split across multiple
    physical heads by the compiler when ``d_v > d_head``).
    ``d_qk = len(query_vector)``, ``d_v = len(value)``.

    Args:
        query_vector: Width-``W`` node at each query position (e.g. a
            column one-hot mapped to 0/1 via ``bool_to_01``).
        key_vector: Width-``W`` node at each key position (e.g. a
            visibility mask in ±1).  Must have the same width as
            ``query_vector``.
        value: Node to read at the winning position.  No width
            constraint — the compiler splits wide V/O across
            multiple physical heads.
        match_gain: Coefficient applied to the dot-product term.

    Returns:
        Attn node of width ``len(value)`` equal to ``value`` at the
        best-matching key position within the causal window.

    See also:
        :func:`attend_argmax_where` — scalar-score variant with
        explicit validity.
    """
    assert len(query_vector) == len(key_vector), (
        "query_vector and key_vector must have the same width "
        f"(got {len(query_vector)} and {len(key_vector)})"
    )
    W = len(query_vector)

    # Content Q/K layout: cols 0..W-1 are match dims (query_vector · key_vector),
    # relocated onto slow planes by rotary_content_head.
    # --- Query ---
    # Columns 0..W-1: match_gain * query_vector[c]
    query_in = query_vector
    query_matrix = torch.zeros((W, W))
    for c in range(W):
        query_matrix[c, c] = match_gain

    # --- Key ---
    # Columns 0..W-1: key_vector[c] (identity)
    key_in = key_vector
    key_matrix = torch.zeros((W, W))
    for c in range(W):
        key_matrix[c, c] = 1.0

    attn = rotary_content_head(
        query_in,
        key_in,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        d_rot=rope.d_rot,
        base=rope.base,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def attend_to_offset(rope: RopeConfig, value: Node, delta_pos: int = -1) -> Node:
    """Read ``value`` from the position ``delta_pos`` away — a rotary offset head.

    The RoPE-native replacement for the old ``PosEncoding.attend_to_offset``:
    a pure-rotary "attend to position ``j + delta_pos``" head that carries **no**
    positional input (position enters only through the runtime rotation).  At
    each query position ``j`` it returns ``value`` at key ``j + delta_pos``;
    ``delta_pos = -1`` is the previous position.  Out-of-range targets (before
    BOS) are a causal don't-care — do not consume them.

    ``delta_pos = 0`` is a no-op (returns ``value``).  Rotary on the
    ``rope.d_head`` grid with the config's ``rope.d_rot`` rotary front (full
    rotary by default; partial rotary is fine here — the unrotated NoPE tail
    contributes a position-independent constant the softmax cancels), so it works
    on all three runtime surfaces.
    """
    return rotary_offset_head(
        value, delta_pos, d_qk=rope.d_head, base=rope.base, d_rot=rope.d_rot
    )


def get_prev_value(
    rope: RopeConfig,
    value: Node,
    cond: Node,
) -> Node:
    """Most-recent previous ``value`` at a position where ``cond`` is true.

    The RoPE-native most-recent read: recency is the **intrinsic rotary
    distance-decay lobe** (:func:`~torchwright.graph.rope.rotary_recency_head`),
    not a precomputed rank (``docs/rope_port_plan.md`` Phase 6 — local recency,
    superseding the octant ramp).  For each query position it selects ``value``
    at the most recent causal position where ``cond`` is true.  ``cond`` follows
    the usual torchwright boolean convention (true ``= +1``; false ``<= 0``).

    The logit at key ``i`` is ``gate · cond_i + recency_gain · lobe(Δ_i)`` where
    ``lobe(Δ) = Σ_p amp_p cos(Δ·θ_p)`` (peak ``≈ 42.6``).  ``gate = 4 ·
    recency_gain · lobe(0)`` so the condition dominates the bounded lobe swing:
    any true key beats any false key, and among true keys the **most recent**
    (largest lobe, i.e. smallest Δ) wins.

    **Recency is local.**  The nearest true key is guaranteed to win only within
    the lobe window ``W ≈ 415`` (the default config); a single true key that is
    farther back is still selected because the content gate dominates (the lobe
    only *ranks* among multiple true keys).  When **two or more** true keys sit
    more than ``W`` apart, the ordering can silently invert — use the Phase-7
    global mechanism there.

    **All-false window.**  If no causal position has ``cond`` true, the gate is
    uniform and the head degrades to pure recency (the most recent position's
    value) — callers must ensure at least one true key exists in every consumed
    window, or gate the result.

    **Degenerate configs raise.**  Two build-time guards, one per way the
    cond gate can fail to latch:

    * **Quasi-static precondition.**  The gate rides the slowest rotated
      plane, so a key at distance ``Δ`` contributes ``±gate·cos(Δ·θ_slow)``.
      If ``max_positions · θ_slow ≥ π/2`` a distant false key's cosine goes
      negative and the cond *ordering* can invert outright — the latch reads
      a false-cond position (the mechanism behind e.g. ``d_head=32,
      max_positions=20, base=10``, which fails to latch by position 15).
      Raises ``ValueError``; increase ``d_head`` (slowing the slowest plane)
      or reduce ``max_positions``.  (Raising the rope ``base`` also slows
      ``θ_slow``, but thins the lobe band the other guard needs — prefer
      ``d_head``.)
    * **Leak bound.**  The gate is sized from the lobe peak, so a rope
      config whose band collapses (a ≤2-plane band Hann-tapers to its
      endpoint zeros — e.g. ``d_head=32, max_positions=64``) leaves the gate
      too weak: false-cond keys leak softmax weight and the latched value
      silently drifts (downstream, ``count_since_marker`` read 2.8 at a true
      gap of 4 before this guard existed).  The worst-case leak —
      ``max_positions · exp(−2·gate·cos_floor)``, the gate attenuated by the
      enforced cosine floor — raises ``ValueError`` when it exceeds
      ``_MAX_RECENCY_LEAK``; increase ``max_positions`` or ``d_head``
      (widening the lobe band).

    Args:
        rope: the RoPE config (``d_head`` / ``base`` / ``max_positions``).
        value: node to read at the selected position.
        cond: length-1 boolean (true = +1).
    """
    assert len(cond) == 1, "get_prev_value expects a 1-D boolean cond"
    _, amps = rope_lobe_band(rope.d_head, rope.base, rope.max_positions)
    lobe_peak = _LOCAL_RECENCY_GAIN * float(amps.sum())
    gate = 4.0 * lobe_peak  # cond dominates the bounded lobe swing

    # The cond gate's ±gate swing is attenuated to ±gate·cos(Δ·θ_slow) at key
    # distance Δ (the gate column rides the slowest rotated plane).  Keep the
    # whole rollout on the cosine's positive, quasi-static side; past π/2 a
    # distant false key's contribution flips sign and the ordering the leak
    # bound below protects can invert outright.  Mirrors the
    # attend_most_recent_globally position-tiebreak guard.
    theta_slow = _theta_slow(rope)
    if rope.max_positions * theta_slow >= math.pi / 2:
        raise ValueError(
            f"get_prev_value: the cond gate's slow-plane rotation is not "
            f"quasi-static at d_head={rope.d_head} base={rope.base:g} "
            f"max_positions={rope.max_positions}: the slowest rotated plane "
            f"(θ={theta_slow:.3e}) turns max_positions × θ = "
            f"{rope.max_positions * theta_slow:.3f} ≥ π/2 ({math.pi / 2:.3f}), "
            f"so a distant key's gate contribution goes negative and the "
            f"cond ordering can invert — the latch would silently read a "
            f"false-cond position.  Increase d_head (slowing the slowest "
            f"plane) or reduce max_positions."
        )
    cos_floor = math.cos(rope.max_positions * theta_slow)

    leak_bound = rope.max_positions * math.exp(-2.0 * gate * cos_floor)
    if leak_bound > _MAX_RECENCY_LEAK:
        raise ValueError(
            f"get_prev_value: the recency-lobe cond gate is too weak to latch "
            f"at d_head={rope.d_head} base={rope.base:g} "
            f"max_positions={rope.max_positions}: the lobe band's Hann taper "
            f"leaves amplitude sum {float(amps.sum()):.2e} (a <=2-plane band "
            f"tapers to its endpoint zeros), gate {gate:.3g} (rotation floor "
            f"cos={cos_floor:.3f}), worst-case false-key softmax leak "
            f"{leak_bound:.2e} > {_MAX_RECENCY_LEAK:g} — the latched value "
            f"would silently drift.  Increase max_positions or d_head "
            f"(widening the lobe band)."
        )

    # Content col: the cond gate, relocated onto the slowest plane.  The recency
    # lobe is added by rotary_recency_head on a disjoint faster-plane band.
    query_one = LiteralValue(torch.tensor([1.0]), name="prev_value_query_one")
    query_matrix = torch.tensor([[gate]])  # (1, 1)
    key_matrix = torch.tensor([[1.0]])  # (1, 1): cond -> gate column

    return rotary_recency_head(
        query_one,
        cond,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        d_rot=rope.d_rot,
        max_positions=rope.max_positions,
        base=rope.base,
        recency_gain=_LOCAL_RECENCY_GAIN,
    )


def attend_most_recent_globally(
    rope: RopeConfig,
    query_vector: Node,
    key_vector: Node,
    global_position: Node,
    value: Node,
    *,
    match_gain: float = 200.0,
    recency_scale: float = _RECENCY_SCALE,
    exclude_self: bool = False,
    assert_hardness_gt: float | None = None,
) -> Node:
    """Attend to the **most recent** key whose content matches — **global** recency.

    The unbounded companion to the local recency lobe (:func:`get_prev_value`).
    Uses a precomputed per-token absolute position (from
    :func:`~torchwright.ops.relu.global_recency.global_position_from_bos` or its
    swiglu twin) as the tiebreak among matching keys, so "most recent" is resolved
    by true global position rather than the local lobe — no window limit.

    The logit at ``(query j, key i)`` is

        match_gain · (query_vector_j · key_vector_i) + recency_scale · global_position_i

    Among keys whose content matches (high dot product), the one with the largest
    ``global_position`` (i.e. the most recent) wins.

    **Content-dominance invariant.**  A matching key at *any* position must beat a
    non-matching key at *any* position, which requires

        match_gain · min_match_dot_gap > recency_scale · max_positions

    With E8 content codes (dot gap = 1600, match_gain=200): 320,000 > 1.0 · 61,440 ✓.

    **Float32 representability.**  Adjacent positions must differ in logit by more
    than the fp32 ULP at the content score scale.  For E8 content (score ≈ 320,000)
    the ULP is ≈ 0.038; ``recency_scale = 1.0`` gives an adjacent-position gap of
    1.0 — 26× above the floor.  For content types with smaller scores, reduce
    ``recency_scale`` so that ``recency_scale · max_positions`` stays below the
    content gap.

    **No window limit.**  Unlike the lobe-based Phase-6 mechanism, this head
    correctly orders matching keys at any distance — the position signal is
    monotone over the full ``[0, max_positions]`` range.  The compile cost is 1
    additional attention head vs Phase 6 (plus the upstream
    ``global_position_from_bos`` MLP sublayer, shared across all callers).

    **Placement (full vs partial rotary).**  Under full rotary the content match
    and the position tiebreak both ride slow planes of the ``d_head`` grid (the
    content on the slowest ``W``, the tiebreak on the next).  Under partial rotary
    (``rope.d_rot < rope.d_head``) the content rides the unrotated NoPE tail — an
    *exact* position-free match, dissolving the slow-plane ``d_head`` budget — and
    only the position tiebreak rides a rotated plane (the slowest, ``d_rot/2−1``).
    The two signals are on disjoint dims either way, so they never interact; the
    partial path is what lets the wide unbounded clip read coexist with the global
    position tiebreak in one feasible-``d_head`` head.

    Args:
        rope: RoPE config.
        query_vector: width-W content query.
        key_vector: width-W content key.  Must have same width as
            ``query_vector``.
        global_position: length-1 node; each key's approximate absolute
            position from ``global_position_from_bos``.
        value: node to read at the selected key position.
        match_gain: coefficient on the content dot-product term.
        recency_scale: per-unit-position logit gain for the position tiebreak.
            Must satisfy ``recency_scale · max_positions < match_gain · min_match_dot_gap``
            (content-dominance invariant; see above).
        exclude_self: if True, shift ``key_vector``, ``global_position``,
            and ``value`` back one position so the current token cannot
            select itself.
        assert_hardness_gt: if set, wraps the output in a softmax hardness
            assert checked during ``debug=True`` passes.

    Returns:
        Attn node of width ``len(value)``.
    """
    assert len(query_vector) == len(key_vector), (
        "query_vector and key_vector must have the same width "
        f"(got {len(query_vector)} and {len(key_vector)})"
    )
    assert len(global_position) == 1, "global_position must be 1-D"

    if exclude_self:
        key_vector = attend_to_offset(rope, key_vector, delta_pos=-1)
        global_position = attend_to_offset(rope, global_position, delta_pos=-1)
        value = attend_to_offset(rope, value, delta_pos=-1)

    W = len(query_vector)

    # Per-position logit gain: each unit of absolute position contributes
    # recency_scale to the logit.  NOT divided by max_positions — the raw
    # position value ≈ i is used directly so adjacent positions differ by
    # recency_scale in logit.  Float32 at content score ~320000 has ULP ≈ 0.038;
    # recency_scale=1.0 gives adjacent diff = 1.0 >> 0.038 (26× margin).
    alpha = recency_scale

    # Shared inputs: content vector + a constant 1.0 (query side) / global_position
    # (key side, ≈ absolute position i ∈ [0, max_positions]) for the tiebreak.
    query_one = LiteralValue(torch.tensor([1.0]), name="global_recency_query_one")
    query_in = Concatenate([query_vector, query_one])
    key_in = Concatenate([key_vector, global_position])

    d_head = rope.d_head
    if rope.d_rot < d_head:
        # --- Partial rotary ---
        # Content (W match dims) rides the unrotated NoPE tail [d_rot:d_rot+W]: an
        # EXACT, position-free content dot product at any distance.  The position
        # tiebreak rides the slowest rotated plane d_rot/2-1, where the RoPE
        # attenuation cos((i−j)·θ_slow) stays positive (the guard below), so among
        # content-matching keys the largest global_position (most recent) wins.
        # The two signals live on disjoint dims, so they do not interact.  This
        # cannot delegate to rotary_content_head, which would route ALL columns
        # uniformly (content to the tail AND the position column to the tail, where
        # there is no rotation to order keys at distance — wrong).
        tail = d_head - rope.d_rot
        if W > tail:
            raise ValueError(
                f"attend_most_recent_globally: content width W={W} exceeds the "
                f"{tail}-wide NoPE tail (d_head={d_head}, d_rot={rope.d_rot}); "
                f"raise d_head or lower d_rot."
            )
        pos_plane = rope.d_rot // 2 - 1
        theta_pos = _theta_slow(rope)
        if rope.max_positions * theta_pos >= math.pi / 2:
            raise ValueError(
                f"attend_most_recent_globally: the position tiebreak on the "
                f"slowest rotated plane {pos_plane} (θ={theta_pos:.3e}) has "
                f"max_positions={rope.max_positions} × θ = "
                f"{rope.max_positions * theta_pos:.3f} ≥ π/2 ({math.pi/2:.3f}); a "
                f"negative cosine would reverse the tiebreak ordering.  Increase "
                f"d_rot/base or reduce max_positions."
            )
        query_matrix = torch.zeros((W + 1, d_head))
        key_matrix = torch.zeros((W + 1, d_head))
        for c in range(W):
            query_matrix[c, rope.d_rot + c] = match_gain  # content on the tail
            key_matrix[c, rope.d_rot + c] = 1.0
        query_matrix[W, pos_plane] = alpha  # position tiebreak on a rotated plane
        key_matrix[W, pos_plane] = 1.0  # global_position passes through identity
        d_v = len(value)
        attn = Attn(
            query_in=query_in,
            key_in=key_in,
            value_in=value,
            query_matrix=query_matrix,
            key_matrix=key_matrix,
            value_matrix=torch.eye(d_v),
            output_matrix=torch.eye(d_v),
            rope_base=rope.base,
            rope_d_rot=rope.d_rot,
        )
    else:
        # --- Full rotary ---
        # Content + position both on slow planes of the d_head grid; the position
        # column occupies the (W+1)-th slowest plane (after the W content planes).
        # Guard: that plane's frequency θ_pos must satisfy max_positions × θ_pos <
        # π/2 so cos((i−j)·θ_pos) stays positive for all key offsets; a negative
        # cosine would reverse the tiebreak ordering.
        theta_pos = float(rope_inv_freq(d_head, rope.base)[d_head // 2 - 1 - W])
        if rope.max_positions * theta_pos >= math.pi / 2:
            raise ValueError(
                f"attend_most_recent_globally: content width W={W} places the "
                f"position tiebreak on plane {d_head // 2 - 1 - W} "
                f"(θ={theta_pos:.3e}); max_positions={rope.max_positions} × θ = "
                f"{rope.max_positions * theta_pos:.3f} ≥ π/2 ({math.pi/2:.3f}).  "
                f"Narrow the content vector, increase d_head/base, or reduce "
                f"max_positions."
            )
        # d_qk layout (W+1 columns placed on slowest W+1 planes):
        #   cols 0..W-1: content match (match_gain · Q · K)
        #   col W:       position tiebreak (recency_scale · 1_query · gp_key)
        d_qk = W + 1
        query_matrix = torch.zeros((W + 1, d_qk))
        for c in range(W):
            query_matrix[c, c] = match_gain
        query_matrix[W, W] = alpha  # constant 1.0 × recency_scale for position col
        key_matrix = torch.zeros((W + 1, d_qk))
        for c in range(W):
            key_matrix[c, c] = 1.0
        key_matrix[W, W] = 1.0  # global_position passes through identity
        attn = rotary_content_head(
            query_in,
            key_in,
            value,
            query_matrix,
            key_matrix,
            d_head=d_head,
            d_rot=rope.d_rot,
            base=rope.base,
        )

    if assert_hardness_gt is not None:
        attn = assert_softmax_hardness(attn, assert_hardness_gt)
    return attn
