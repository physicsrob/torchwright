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
- Every head is **full-width rotary on slow planes**: the content columns are
  relocated onto the slowest planes of the ``rope.d_head`` ``rotate_half`` grid
  (:func:`~torchwright.graph.rope.rotary_content_head`), so the match survives
  the end-state global rotation (``docs/rope_port_plan.md`` §3).  The builders
  therefore take a :class:`~torchwright.graph.RopeConfig` (carrying
  ``d_head`` / ``base``) where they used to take a ``PosEncoding`` node —
  position is a rotation applied inside attention, no longer a residual
  feature.
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
    rope_lobe_band,
    rotary_content_head,
    rotary_offset_head,
    rotary_recency_head,
)
from torchwright.graph.value_type import NodeValueType

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

    This helper wraps the Attn in an Assert that (a) promotes the
    claim statically via ``claimed_type``, and (b) runs a runtime
    predicate during reference_eval checking each claimed property to
    within ``atol`` — the safety net that catches construction errors
    (insufficient gain, score ties, non-identity V/O, etc.).

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
# path runs fp32 through the SDPA MATH backend with TF32 off (see the
# precision-policy note on ``attend_most_recent_matching``), so the
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
# recent" tiebreak in ``get_prev_value`` / ``attend_most_recent_matching``).
# The lobe peak is ``Σ amp_p ≈ 42.6`` (:func:`~torchwright.graph.rope.rope_lobe_band`
# at the production config), so ``recency_gain · peak ≈ 2.5e4``.  Sized against
# the *smallest* near-Δ step over the working window: the Hann taper rounds the
# peak, so the Δ=0→1 step (self vs immediate predecessor, normalized ``≈ 3.0e-4``)
# is smaller than the Δ=1→2 gap-1 step (``≈ 8.8e-4``); at ``600`` the worst step
# over the ~100-token target is ``≈ 7.5`` logits → ``exp(7.5) ≈ 1.8e3`` softmax
# ratio (≥ 99.9 % concentration).  Unlike the old octant ramp's ``rank_gain ≈
# 2e5``, the lobe is bounded (does not grow with sequence length), so the content
# gate that must dominate it is small (``get_prev_value`` sets it automatically;
# :func:`attend_most_recent_matching` callers size ``match_gain`` per the
# content-dominance bound in its docstring — low-dot content needs a large
# ``match_gain``, high-dot E8 content clears it at the default).
_LOCAL_RECENCY_GAIN = 600.0


def _build_selection_attn(
    rope: RopeConfig,
    key_in: Node,
    key_matrix: torch.Tensor,
    value: Node,
) -> Attn:
    """Wire up a selection-style attention head — rotary on slow planes.

    Callers supply ``key_in`` (the content node(s) driving selection) and a
    populated ``key_matrix`` of content width ``W``; this helper fills in the
    query matrix and relocates the content onto the slowest ``W`` planes of the
    full ``rope.d_head`` ``rotate_half`` grid via :func:`rotary_content_head`,
    so selection survives the end-state global rotation (the match rides
    quasi-static planes — ``docs/rope_port_plan.md`` §3).  The score logit lives
    in content column 0: ``Q`` projects an exact constant ``1.0`` (a
    ``LiteralValue``) scaled by ``_QUERY_GAIN``, so ``Q[·, 0]`` is a stable
    positive gain independent of query position.  A unit score delta is then
    decisive (``exp(8) ≈ 3000``).  Value passes through identity V/O and is
    split across physical heads by the compiler when wider than ``d_head``.

    ``rope`` carries the ``d_head`` / ``base`` the slow-plane placement needs.
    """
    d_qk = key_matrix.shape[1]

    query_one = LiteralValue(torch.tensor([1.0]), name="selection_query_one")
    query_matrix = torch.zeros((1, d_qk))
    query_matrix[0, 0] = _QUERY_GAIN

    return rotary_content_head(
        query_one,
        key_in,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        base=rope.base,
    )


def _build_where_attn(
    rope: RopeConfig,
    score: Node,
    validity: Node,
    value: Node,
    *,
    score_sign: float,
) -> Attn:
    """Shared construction for ``attend_argmin_where`` / ``attend_argmax_where``.

    Content (``d_qk``) layout, relocated onto slow planes by
    :func:`rotary_content_head`:
      * col 0: gained score (``Q = _QUERY_GAIN``).
      * col 1: additive validity (``Q = 1.0``, ``K = ± _VALIDITY_DIRECT``),
        not multiplied by the gain.

    ``score_sign`` is ``-1`` for argmin (small score → large logit) and
    ``+1`` for argmax.  ``rope`` carries the ``d_head`` / ``base`` for the
    slow-plane placement; the query constant is an exact ``LiteralValue([1.0])``.
    """
    # key_in row layout: [score (1), validity (1)]
    key_in = Concatenate([score, validity])

    # --- Query: an exact 1.0 projected to col 0 (gained) and col 1 (direct). ---
    query_one = LiteralValue(torch.tensor([1.0]), name="where_query_one")
    query_matrix = torch.zeros((1, 2))
    query_matrix[0, 0] = _QUERY_GAIN
    query_matrix[0, 1] = 1.0

    # --- Key: col 0 score, col 1 direct validity. ---
    key_matrix = torch.zeros((len(key_in), 2))
    key_matrix[0, 0] = score_sign
    key_matrix[1, 1] = _VALIDITY_DIRECT

    return rotary_content_head(
        query_one,
        key_in,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        base=rope.base,
    )


def attend_argmin(
    rope: RopeConfig,
    score: Node,
    value: Node,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Attend to the position with the *minimum* score.

    For each query position, this returns ``value`` at the position within
    the causal window (positions ``<= current``) whose ``score`` is
    smallest.  When multiple positions share the same score, the output
    is a soft average of their values — callers that need deterministic
    selection should ensure distinct scores.

    To mask positions you want the attention to ignore, pass a score that
    is very large at those positions (a few hundred is enough). For a
    cleaner valid/invalid API, use :func:`attend_argmin_where` instead.

    Compile cost: exactly one vanilla attention head.

    Args:
        rope: RoPE config (``d_head`` / ``base``) for the slow-plane placement.
        score: 1D scalar node (``len(score) == 1``).
        value: Node whose value to read at the winning position.
            No width constraint — wide V/O is auto-split across physical
            heads by the compiler.

    Returns:
        A new ``Attn`` node of width ``len(value)`` equal to ``value`` at
        the argmin-of-``score`` key position within the causal window.

    See also:
        :func:`attend_argmax`, :func:`attend_argmin_where`.
    """
    assert len(score) == 1, "attend_argmin expects a 1D scalar score node"

    key_matrix = torch.zeros((len(score), 1))
    key_matrix[0, 0] = -1.0  # smaller score → larger logit

    attn = _build_selection_attn(rope, score, key_matrix, value)
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def attend_argmax(
    rope: RopeConfig,
    score: Node,
    value: Node,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Attend to the position with the *maximum* score.

    Sign-flipped twin of :func:`attend_argmin`.

    Args:
        rope: RoPE config (``d_head`` / ``base``) for the slow-plane placement.
        score: 1D scalar node.
        value: Node whose value to read. No width constraint (wide V/O
            auto-splits across heads).

    Returns:
        Attn node of width ``len(value)`` equal to ``value`` at the
        argmax-of-``score`` key position within the causal window.
    """
    assert len(score) == 1, "attend_argmax expects a 1D scalar score node"

    key_matrix = torch.zeros((len(score), 1))
    key_matrix[0, 0] = 1.0  # larger score → larger logit

    attn = _build_selection_attn(rope, score, key_matrix, value)
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def attend_argmin_where(
    rope: RopeConfig,
    score: Node,
    validity: Node,
    value: Node,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Argmin of ``score`` restricted to positions where ``validity`` is true.

    The workhorse primitive for selection-sort variants. At each query
    position, the attention returns ``value`` at the causal-window
    position where ``validity`` is true **and** ``score`` is smallest.

    ``validity`` follows the usual torchwright boolean convention: +1.0
    means "valid", −1.0 means "invalid". Validity is routed through a
    dedicated ``d_qk`` column (``Q = 1.0``, ``K = ± _VALIDITY_DIRECT``)
    rather than combined with the score column under ``_QUERY_GAIN``,
    so the logit at key position ``i`` is

        _QUERY_GAIN · (−score[i]) + _VALIDITY_DIRECT · validity[i]

    Because ``_VALIDITY_DIRECT > _QUERY_GAIN · _MAX_SCORE_ABS``, validity
    dominates score: the softmax always prefers a valid position over an
    invalid one regardless of their scores; among valid positions,
    smaller score wins.  Tied scores produce a soft average of the tied
    positions' values.

    **When no position is valid.** The softmax still runs and produces a
    weighted average over all positions — effectively garbage. Callers
    must ensure at least one valid position exists within the causal
    window at every query position whose output is actually consumed, or
    wrap the result in a ``select`` against a sentinel literal.

    Compile cost: exactly one vanilla attention head.

    Args:
        rope: RoPE config (``d_head`` / ``base``) for the slow-plane placement.
        score: 1D scalar node.
        validity: 1D boolean node (+1 valid, −1 invalid).
        value: Node to read. No width constraint (wide V/O auto-splits
            across heads).

    Returns:
        Attn node of width ``len(value)``.

    See also:
        :func:`attend_argmax_where` — maximum-score dual.
    """
    assert len(score) == 1, "attend_argmin_where expects a 1D scalar score"
    assert len(validity) == 1, "attend_argmin_where expects a 1D boolean validity"
    attn = _build_where_attn(
        rope,
        score,
        validity,
        value,
        score_sign=-1.0,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def attend_argmax_where(
    rope: RopeConfig,
    score: Node,
    validity: Node,
    value: Node,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Argmax of ``score`` restricted to positions where ``validity`` is true.

    Sign-flipped twin of :func:`attend_argmin_where`. Same caveats about
    all-invalid windows.

    Args:
        rope: RoPE config (``d_head`` / ``base``) for the slow-plane placement.
        score: 1D scalar node.
        validity: 1D boolean node (+1 valid, −1 invalid).
        value: Node to read.

    Returns:
        Attn node of width ``len(value)``.
    """
    assert len(score) == 1, "attend_argmax_where expects a 1D scalar score"
    assert len(validity) == 1, "attend_argmax_where expects a 1D boolean validity"
    attn = _build_where_attn(
        rope,
        score,
        validity,
        value,
        score_sign=+1.0,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


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
        base=rope.base,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def attend_argmin_valid_unmasked(
    rope: RopeConfig,
    score: Node,
    validity: Node,
    mask_vector: Node,
    position_onehot: Node,
    value: Node,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Argmin of ``score`` restricted to valid keys, with a per-query mask.

    Combines ``attend_argmin_where``'s per-key validity signal with
    ``attend_argmin_unmasked``'s per-query mask rendezvous. The logit at
    key position ``i`` under query position ``j`` is

        _QUERY_GAIN · (−score[i] + _VALIDITY_KEY_COEFF · validity[i])
            − _UNMASKED_PENALTY · mask_vector_j[position_onehot_i]

    Unlike the simple ``_where`` variants, validity is kept in the
    *gained* (multiplicative) column rather than an additive one — the
    caller's mask_vector can accumulate integer values above 1 as the
    same slot is re-picked, and the multiplicative validity budget
    (``_QUERY_GAIN · _VALIDITY_KEY_COEFF = 8000``) must dominate
    ``_UNMASKED_PENALTY · max_walls`` for the masked-valid fallback to
    keep working.

    Separation (with ``_QUERY_GAIN=8``, ``_VALIDITY_KEY_COEFF=1000``,
    ``_UNMASKED_PENALTY=1000``, ``|score| ≤ 100`` one-sided):

    * worst valid-unmasked logit ≈ ``8 · (-100 + 1000) = +7200``
    * valid-masked logit at mask-bit ``k`` ≈ ``8000 − 1000 · k``
    * worst invalid-unmasked logit ≈ ``8 · (0 − 1000) = -8000``

    Since ``2 · _QUERY_GAIN · _VALIDITY_KEY_COEFF = 16000`` and
    ``_UNMASKED_PENALTY = 1000``, validity dominates mask up to
    ``max_walls ≤ 15``: a masked-valid key (bit accumulated up to 15)
    still beats any invalid key.  For larger ``max_walls`` the caller
    must either cap accumulation via a saturating mask update or raise
    ``_VALIDITY_KEY_COEFF``.

    **End-of-sort behavior.** When ``N_renderable < max_walls``, after
    all valid keys are picked the attention re-picks the last-picked
    valid key (masked-valid). Wasteful but correct — callers that want
    early termination must gate downstream consumers on a compiled
    "done" signal.

    Compile cost: one vanilla attention head.
    ``d_head = 1 + n_slots + len(value)``.

    Args:
        rope: RoPE config (``d_head`` / ``base``) for the slow-plane placement.
        score: 1D scalar node.
        validity: 1D boolean node (+1 valid, −1 invalid).
        mask_vector: Width-``N`` per-query ``{0, 1}`` mask.
        position_onehot: Width-``N`` per-key one-hot of input-slot index.
        value: Node to read at the selected key position.

    Returns:
        Attn node of width ``len(value)``.
    """
    assert len(score) == 1, "attend_argmin_valid_unmasked expects a 1D scalar score"
    assert (
        len(validity) == 1
    ), "attend_argmin_valid_unmasked expects a 1D boolean validity"
    assert len(mask_vector) == len(position_onehot), (
        "mask_vector and position_onehot must have the same width "
        f"(got {len(mask_vector)} and {len(position_onehot)})"
    )
    n_slots = len(mask_vector)
    # Content Q/K layout (relocated onto slow planes; value rides identity V/O):
    #   col 0:               score + gained validity
    #   cols 1 .. n_slots:   mask · position_onehot terms
    W = 1 + n_slots

    # Query: an exact 1.0 for the score/validity gain (col 0) plus the mask.
    query_one = LiteralValue(torch.tensor([1.0]), name="valid_unmasked_query_one")
    query_in = Concatenate([query_one, mask_vector])
    key_in = Concatenate([score, validity, position_onehot])

    # --- Query matrix, shape (1 + n_slots, W) ---
    query_matrix = torch.zeros((len(query_in), W))
    query_matrix[0, 0] = _QUERY_GAIN
    for c in range(n_slots):
        query_matrix[1 + c, 1 + c] = -_UNMASKED_PENALTY

    # --- Key matrix, shape (2 + n_slots, W) ---
    # Row order in key_in: [score (1), validity (1), onehot (n_slots)]
    key_matrix = torch.zeros((len(key_in), W))
    score_row = 0
    validity_row = 1
    onehot_start_row = 2
    key_matrix[score_row, 0] = -1.0  # smaller score → larger logit
    key_matrix[validity_row, 0] = (
        _VALIDITY_KEY_COEFF  # gained validity dominates mask accumulation
    )
    for c in range(n_slots):
        key_matrix[onehot_start_row + c, 1 + c] = 1.0

    attn = rotary_content_head(
        query_in,
        key_in,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
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
    :func:`~torchwright.ops.arithmetic_ops.bool_to_01`, average them
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

    # pos_encoding doesn't appear in key_in — only validity drives K.
    key_matrix = torch.zeros((len(validity), d_qk))
    key_matrix[0, 0] = _VALIDITY_DIRECT

    attn = rotary_content_head(
        query_one,
        validity,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        base=rope.base,
    )
    # Mean of values in [lo, hi] stays in [lo, hi] (convex combination),
    # but integer-ness / binary-ness / one-hot-ness do not survive the
    # soft mean.  Only promote the range claim.
    r = value.value_type.value_range
    if math.isfinite(r.lo) and math.isfinite(r.hi):
        return assert_in_range(attn, r.lo, r.hi, atol=_HARD_SELECTION_ATOL)
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
    :func:`~torchwright.ops.logic_ops.cond_gate` to zero out
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
        base=rope.base,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def _build_dot_where_attn(
    rope: RopeConfig,
    query_vector: Node,
    key_vector: Node,
    validity: Node,
    value: Node,
    *,
    score_sign: float,
    match_gain: float,
) -> Attn:
    """Shared construction for dot-product selection with validity.

    ``score_sign`` is ``+1`` for argmax and ``-1`` for argmin.  Content is
    relocated onto slow planes by :func:`rotary_content_head`; ``rope`` carries
    the ``d_head`` / ``base``.
    """
    assert len(query_vector) == len(key_vector), (
        "query_vector and key_vector must have the same width "
        f"(got {len(query_vector)} and {len(key_vector)})"
    )
    assert len(validity) == 1, "attend_*_dot_where expects 1D boolean validity"
    assert match_gain > 0, "attend_*_dot_where expects positive match_gain"

    W = len(query_vector)
    # Content Q/K layout:
    #   cols 0..W-1: dot-product score
    #   col W:       direct validity bonus
    d_qk = W + 1

    # The validity term needs a query-side constant.  A LiteralValue keeps
    # this op independent of position while preserving the same direct
    # validity-logit pattern used by the scalar _where variants.
    query_one = LiteralValue(torch.tensor([1.0]), name="dot_where_query_one")
    query_in = Concatenate([query_vector, query_one])
    key_in = Concatenate([key_vector, validity])

    query_matrix = torch.zeros((W + 1, d_qk))
    for c in range(W):
        query_matrix[c, c] = score_sign * match_gain
    query_matrix[W, W] = 1.0

    key_matrix = torch.zeros((W + 1, d_qk))
    for c in range(W):
        key_matrix[c, c] = 1.0
    key_matrix[W, W] = _VALIDITY_DIRECT

    return rotary_content_head(
        query_in,
        key_in,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        base=rope.base,
    )


def attend_argmax_dot_where(
    rope: RopeConfig,
    query_vector: Node,
    key_vector: Node,
    validity: Node,
    value: Node,
    match_gain: float = 200.0,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Argmax of ``query_vector · key_vector`` over valid key positions.

    At each query position, returns ``value`` at the causal-window
    position where ``validity`` is +1 and the dot product with the query
    vector is largest.  ``validity`` follows the usual torchwright
    boolean convention: +1.0 means "valid", -1.0 means "invalid".

    The logit at key position ``i`` seen from query position ``j`` is

        match_gain · (query_vector[j] · key_vector[i])
            + _VALIDITY_DIRECT · validity[i]

    Invalid rows cannot win as long as every caller-provided dot score
    satisfies ``abs(match_gain * dot_score) <= _MAX_DOT_LOGIT_ABS``.
    With the default ``match_gain=200``, this means
    ``abs(dot_score) <= 4.8``.  If your dot products are larger, scale
    the vectors or lower ``match_gain`` so the validity term has room to
    dominate; tied valid scores produce a soft average.

    **When no position is valid.** The softmax still returns a weighted
    average over invalid positions.  Callers must ensure each consumed
    query has at least one valid causal-window key, or gate the result
    downstream.

    Compile cost: one attention head (auto-split across multiple
    physical heads by the compiler when ``d_v > d_head``).
    ``d_qk = len(query_vector) + 1``, ``d_v = len(value)``.

    Args:
        query_vector: Width-``W`` node at each query position.
        key_vector: Width-``W`` node at each key position.
        validity: 1D boolean node (+1 valid, -1 invalid).
        value: Node to read at the winning position.
        match_gain: Positive coefficient applied to the dot-product term.
        assert_hardness_gt: If set, checks the winning softmax weight.

    Returns:
        Attn node of width ``len(value)``.

    See also:
        :func:`attend_argmin_dot_where` — minimum-dot dual.
        :func:`attend_argmax_dot` — original unmasked dot-product variant.
    """
    attn = _build_dot_where_attn(
        rope,
        query_vector,
        key_vector,
        validity,
        value,
        score_sign=+1.0,
        match_gain=match_gain,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )


def attend_argmin_dot_where(
    rope: RopeConfig,
    query_vector: Node,
    key_vector: Node,
    validity: Node,
    value: Node,
    match_gain: float = 200.0,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Argmin of ``query_vector · key_vector`` over valid key positions.

    Sign-flipped twin of :func:`attend_argmax_dot_where`.  This is the
    variant to use when zeroed invalid keys are unsafe: in an argmin, an
    invalid zero key can otherwise beat valid keys with positive dot
    scores.

    The logit at key position ``i`` seen from query position ``j`` is

        -match_gain · (query_vector[j] · key_vector[i])
            + _VALIDITY_DIRECT · validity[i]

    The same supported range applies:
    ``abs(match_gain * dot_score) <= _MAX_DOT_LOGIT_ABS`` for all key
    rows in the consumed causal window.  At the default
    ``match_gain=200``, that is ``abs(dot_score) <= 4.8``.

    Args:
        query_vector: Width-``W`` node at each query position.
        key_vector: Width-``W`` node at each key position.
        validity: 1D boolean node (+1 valid, -1 invalid).
        value: Node to read at the winning position.
        match_gain: Positive coefficient applied to the dot-product term.
        assert_hardness_gt: If set, checks the winning softmax weight.

    Returns:
        Attn node of width ``len(value)``.
    """
    attn = _build_dot_where_attn(
        rope,
        query_vector,
        key_vector,
        validity,
        value,
        score_sign=-1.0,
        match_gain=match_gain,
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

    ``delta_pos = 0`` is a no-op (returns ``value``).  Full-width rotary on the
    ``rope.d_head`` grid (the §6 LLaMA3 end state), so it works on all three
    runtime surfaces.
    """
    return rotary_offset_head(value, delta_pos, d_qk=rope.d_head, base=rope.base)


def get_prev_value(
    rope: RopeConfig,
    value: Node,
    cond: Node,
    *,
    recency_gain: float = _LOCAL_RECENCY_GAIN,
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

    Args:
        rope: the RoPE config (``d_head`` / ``base`` / ``max_positions``).
        value: node to read at the selected position.
        cond: length-1 boolean (true = +1).
        recency_gain: coefficient on the rotary recency lobe; sets the softmax
            sharpness of the tiebreak among true keys.
    """
    assert len(cond) == 1, "get_prev_value expects a 1-D boolean cond"
    _, amps = rope_lobe_band(rope.d_head, rope.base, rope.max_positions)
    lobe_peak = recency_gain * float(amps.sum())
    gate = 4.0 * lobe_peak  # cond dominates the bounded lobe swing

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
        max_positions=rope.max_positions,
        base=rope.base,
        recency_gain=recency_gain,
    )


def attend_most_recent_matching(
    rope: RopeConfig,
    query_vector: Node,
    key_vector: Node,
    value: Node,
    *,
    match_gain: float = 200.0,
    recency_gain: float = _LOCAL_RECENCY_GAIN,
    exclude_self: bool = False,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    """Attend to the **most recent** key whose ``key_vector`` matches
    ``query_vector`` — RoPE-native, with **local** recency.

    The content match rides slow planes and the recency tiebreak is the
    **intrinsic rotary distance-decay lobe** built into the rotation itself
    (:func:`~torchwright.graph.rope.rotary_recency_head`) — no precomputed rank,
    no reference token (``docs/rope_port_plan.md`` Phase 6, superseding the
    octant ramp / ``{BOS, REF}`` readout).

    At each query position ``j`` the logit at key ``i`` is

        match_gain · (query_vector_j · key_vector_i) + recency_gain · lobe(Δ)

    where ``Δ = j − i`` and ``lobe(Δ) = Σ_p amp_p cos(Δ·θ_p)`` is the Hann-tapered
    self-similarity of the recency plane band (peak ``≈ 42.6``, strictly
    decreasing over the window ``W``).  So among keys whose content matches, the
    one with the smallest ``Δ`` (the **most recent**) wins.

    **Recency is local — the load-bearing limit.**  ``lobe`` is monotone only
    within the window ``W ≈ 415`` (the default config — measured by
    ``scripts/rope_window_frontier.py``; ~4× the ~100-token Phase-6 target).  If
    two or more matching keys lie more than ``W`` apart, a *farther* key can
    outscore a nearer one and "most recent" silently picks the wrong key.
    Consumers whose matching-key separation can exceed ``W`` need the Phase-7
    global mechanism.  (A *single* matching key far back is still selected: the
    content gate dominates the bounded lobe; the lobe only ranks *among* matches.)

    **Required invariant on ``match_gain`` (content dominance).**  A
    content-matched but older key must still beat an unmatched but newer key, so

        match_gain · (min_match_dot − max_no_match_dot) > recency_gain · lobe(0)

    where ``recency_gain · lobe(0) ≈ recency_gain · 42.6`` is the *bounded* lobe
    peak (it does **not** grow with sequence length — the key advantage over the
    old ``rank_gain · rank_range ≈ 4.1e5`` octant ramp).  At the default
    ``recency_gain = 600`` that is ``≈ 2.5e4``, so **low-dot content needs a
    large ``match_gain``** (one-hot, dot gap 1, needs ``match_gain > 2.5e4``);
    high-dot E8 codes (self-dot 1600, off-diagonal ~800, gap ~800) clear it at
    the default ``match_gain = 200`` (``200·800 = 1.6e5 ≫ 2.5e4``).

    **All-matching / no-match windows.**  At least one matching key must exist in
    every consumed window, else the head degrades to pure recency (most recent
    position).

    Args:
        rope: the RoPE config (``d_head`` / ``base`` / ``max_positions``).
        query_vector: Width-``W`` node — what we're looking for.
        key_vector: Width-``W`` node — each key's identity.
        value: Node to read at the selected key position.
        match_gain: Coefficient on the dot-product term (see the invariant).
        recency_gain: Coefficient on the rotary recency lobe; sets the softmax
            sharpness of the tiebreak among matching keys.
        exclude_self: If ``True``, the current query position is excluded —
            the head returns ``value`` at the most recent matching position
            strictly before self.  Implemented by pre-shifting ``key_vector``
            and ``value`` back one position via the rotary
            :func:`attend_to_offset`.  At query position 0 the result is
            degenerate; the caller must not consume it there.
        assert_hardness_gt: If set, wraps the output in a softmax hardness
            assertion checked during ``debug=True`` passes.

    Returns:
        Attn node of width ``len(value)``.
    """
    assert len(query_vector) == len(key_vector), (
        "query_vector and key_vector must have the same width "
        f"(got {len(query_vector)} and {len(key_vector)})"
    )
    if exclude_self:
        # Shift key and value back one position so slot i carries the
        # predecessor's data — self can no longer contribute its own match.
        key_vector = attend_to_offset(rope, key_vector, delta_pos=-1)
        value = attend_to_offset(rope, value, delta_pos=-1)
    W = len(query_vector)

    # Content Q/K (relocated onto slow planes by rotary_recency_head); the
    # recency lobe is added by rotary_recency_head on a disjoint faster band.
    query_matrix = match_gain * torch.eye(W)
    key_matrix = torch.eye(W)

    attn = rotary_recency_head(
        query_vector,
        key_vector,
        value,
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        max_positions=rope.max_positions,
        base=rope.base,
        recency_gain=recency_gain,
    )
    return _wrap_hard_selection_output(
        attn, value, assert_hardness_gt=assert_hardness_gt
    )
