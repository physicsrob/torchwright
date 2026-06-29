"""Phase 7 — global most-recent: unbounded recency via BOS-weight inversion.

docs/rope_port_plan.md Phase 7, candidate (d).

**The problem Phase 6 left open.** The intrinsic rotary recency lobe
(:func:`~torchwright.ops.attention_ops.attend_most_recent_matching`) is
monotone within the lobe window W ≈ 415; past W a farther key can beat a
nearer one.  DOOM clip look-backs can exceed W, so Phase 6 needs a global
companion for unbounded "most recent matching."

**Mechanism.** Every token at absolute position m can recover m from the
softmax weight it places on BOS.  When the BOS key carries a special feature
of magnitude sqrt(ln(MAX_LEN)) on the slowest rotary plane — and every other
key carries zero on that plane — the Q·K score from query m to BOS is

    S_BOS = ln(MAX_LEN) · cos(m · θ_slow)

and the score to every other key is 0, giving softmax weight

    w(m) = MAX_LEN^cos(m·θ) / (MAX_LEN^cos(m·θ) + m)

``w`` is strictly monotone in m (validated in
``scripts/rope_global_recency_validate.py``), with minimum adjacent-m gap
~4.9e-6 (~81× the fp32 weight floor).  A 1024-breakpoint log-uniform PWL
inverts w → m with max error ~0.009; combined with the fp32 softmax noise
(~0.006) the total positional error is ~0.15 (fp32 ReLU accumulation in the PWL sum
dominates at small positions — see ``tests/ops/test_global_recency.py``) —
still 3.3× below the 0.5 rounding threshold for integer recovery.

The cosine attenuation at large m (cos(MAX_LEN·θ_slow) ≈ 0.991 — a ~1%
variation) is **baked into the PWL fit**: the table is built against the true
attenuated function w(m), not the idealized 1/(1+m/MAX_LEN), so no separate
correction is needed.

**API.**

    # Once per graph: compute each token's approximate absolute position.
    pos = global_position_from_bos(rope, bos_indicator)

    # Wherever you need "most recent matching" beyond the lobe window W:
    result = attend_most_recent_globally(
        rope, query_vec, key_vec, pos, value
    )

``bos_indicator`` is a 1D node that is 1.0 at position 0 (BOS) and 0.0
everywhere else.  The caller derives it from the BOS token's embedding or a
graph-level one-hot.
"""

import math

import torch

from torchwright.graph import Concatenate, LiteralValue, Node, RopeConfig
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.rope import (
    rope_inv_freq,
    rotary_content_head,
)
from torchwright.ops.arithmetic_ops import piecewise_linear
from torchwright.ops.attention_ops import attend_to_offset

# How many log-uniform breakpoints to use for the PWL inverse w → m.
# Validation (scripts/rope_global_recency_validate.py) shows 1024 achieves
# max error ~0.013 positions, well within the 0.5 rounding threshold.
# 1024 neurons fit in a single MLP sublayer (d_max=1024 default).
_N_BPS = 1024

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _theta_slow(rope: RopeConfig) -> float:
    """Frequency of the slowest rotary plane for this rope config."""
    return float(rope_inv_freq(rope.d_head, rope.base)[-1])


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def global_position_from_bos(
    rope: RopeConfig,
    bos_indicator: Node,
    *,
    n_breakpoints: int = _N_BPS,
) -> Node:
    """Approximate absolute position (0-indexed) via boosted-BOS attention.

    Builds one attention head where BOS carries a special feature of magnitude
    ``sqrt(ln(max_positions))`` on the slowest rotary plane, and all other keys
    carry zero.  At query position m the softmax weight at BOS is

        w(m) = MAX_LEN^cos(m·θ) / (MAX_LEN^cos(m·θ) + m)

    A log-uniform PWL table (``n_breakpoints`` entries) inverts w → m.

    Precision: max error ~0.15 positions (PWL fitting ~0.009; fp32 softmax
    ~0.006; fp32 ReLU accumulation in the PWL sum up to ~0.09 at small
    positions — empirically measured in ``tests/ops/test_global_recency.py``),
    still 3.3× below the 0.5 rounding threshold for integer recovery.

    Compile cost: 1 attention head + 1 MLP sublayer (the PWL inversion).

    Args:
        rope: RoPE config — ``d_head``, ``base``, ``max_positions``.
        bos_indicator: length-1 node; 1.0 at position 0 (BOS), 0.0 elsewhere.
            Derive from the BOS token's embedding one-hot or a graph-level
            indicator.
        n_breakpoints: breakpoints for the PWL inverse table.  1024 (default)
            fits in one MLP sublayer; 512 gives max error ~0.05 — still fine.

    Returns:
        length-1 node whose value at position m approximates m (float,
        in ``[0, max_positions]``).
    """
    assert len(bos_indicator) == 1, "bos_indicator must be 1-D"

    max_len = rope.max_positions
    theta = _theta_slow(rope)

    # Guard: the BOS-weight w(m) is monotone only when the slowest plane rotates
    # less than π/2 over the full rollout.  Past π/2, cos(m·θ) goes negative,
    # w(m) eventually starts increasing, and the PWL inversion silently maps
    # large positions to near-zero — wrong output, no crash.
    product = max_len * theta
    if product >= math.pi / 2:
        raise ValueError(
            f"global_position_from_bos requires max_positions × theta_slow < π/2 "
            f"for the BOS weight to be monotone, but got "
            f"max_positions={max_len} × theta_slow={theta:.3e} = {product:.3f} ≥ {math.pi/2:.3f}.  "
            f"Increase d_head or base, or reduce max_positions."
        )

    A = math.sqrt(math.log(max_len))  # sqrt(ln(MAX_LEN)) so Q·K = ln(MAX_LEN)

    # --- BOS-weight attention head ---
    # Q: constant 1.0 → projected to A on the slowest plane
    # K: bos_indicator (1 at BOS, 0 elsewhere) → projected to A on slowest plane
    # V: bos_indicator (so output = w_BOS · 1 + sum_others w_i · 0 = w_BOS)
    query_one = LiteralValue(torch.tensor([1.0]), name="bos_weight_query_one")
    query_matrix = torch.tensor([[A]])  # (1, 1): 1.0 → A on slowest plane
    key_matrix = torch.tensor([[A]])  # (1, 1): bos_ind → A·bos_ind on slowest plane

    bos_weight = rotary_content_head(
        query_one,
        bos_indicator,
        bos_indicator,  # value: 1 at BOS, 0 elsewhere → output = w_BOS
        query_matrix,
        key_matrix,
        d_head=rope.d_head,
        base=rope.base,
    )

    # --- PWL inverse: w → m ---
    # Build log-uniform breakpoints in w ∈ [w_min, 1] (ascending).
    w_min = _w_of_m(max_len, max_len, theta)
    ratio = (1.0 / w_min) ** (1.0 / (n_breakpoints - 1))
    w_bps = [w_min * (ratio**k) for k in range(n_breakpoints)]
    w_bps[-1] = 1.0  # exact endpoint

    # clamp=False: suppress piecewise_linear's auto tight-range assertion
    # (atol 0.001), which fires at position 0 when float32 ReLU-sum accumulation
    # rounds -0.04 below zero.  assert_in_range(atol=0.1) absorbs that noise
    # while still catching any large-scale problems.
    raw = piecewise_linear(
        bos_weight,
        w_bps,
        lambda w: _bisect_m(w, max_len, theta),
        clamp=False,
        name="bos_weight_to_position",
    )
    return assert_in_range(raw, 0.0, float(max_len), atol=0.1)


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

    The unbounded companion to
    :func:`~torchwright.ops.attention_ops.attend_most_recent_matching`.  Uses a
    precomputed per-token absolute position (from :func:`global_position_from_bos`)
    as the tiebreak among matching keys, so "most recent" is resolved by true
    global position rather than the local lobe — no window limit.

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
    :func:`global_position_from_bos` MLP sublayer, shared across all callers).

    Args:
        rope: RoPE config.
        query_vector: width-W content query.
        key_vector: width-W content key.  Must have same width as
            ``query_vector``.
        global_position: length-1 node; each key's approximate absolute
            position from :func:`global_position_from_bos`.
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
    # Guard: the position column occupies the (W+1)-th slowest plane.  That
    # plane's frequency θ_pos must satisfy max_positions × θ_pos < π/2 so the
    # RoPE attenuation cos((i−j)·θ_pos) remains positive for all key offsets;
    # a negative cosine would reverse the tiebreak ordering.
    theta_pos = float(rope_inv_freq(rope.d_head, rope.base)[rope.d_head // 2 - 1 - W])
    if rope.max_positions * theta_pos >= math.pi / 2:
        raise ValueError(
            f"attend_most_recent_globally: content width W={W} places the "
            f"position tiebreak on plane {rope.d_head // 2 - 1 - W} "
            f"(θ={theta_pos:.3e}); max_positions={rope.max_positions} × θ = "
            f"{rope.max_positions * theta_pos:.3f} ≥ π/2 ({math.pi/2:.3f}).  "
            f"Narrow the content vector, increase d_head/base, or reduce "
            f"max_positions."
        )

    # Per-position logit gain: each unit of absolute position contributes
    # recency_scale to the logit.  NOT divided by max_positions — the raw
    # position value ≈ i is used directly so adjacent positions differ by
    # recency_scale in logit.  Float32 at content score ~320000 has ULP ≈ 0.038;
    # recency_scale=1.0 gives adjacent diff = 1.0 >> 0.038 (26× margin).
    alpha = recency_scale

    # d_qk layout (W+1 columns placed on slowest W+1 planes):
    #   cols 0..W-1: content match (match_gain · Q · K)
    #   col W:       position tiebreak (recency_scale · 1_query · global_position_key)
    d_qk = W + 1

    # Query: content vector + constant 1.0 for the position column
    query_one = LiteralValue(torch.tensor([1.0]), name="global_recency_query_one")
    query_in = Concatenate([query_vector, query_one])

    # Key: content vector + global position (≈ absolute position i, range [0, max_positions])
    key_in = Concatenate([key_vector, global_position])

    query_matrix = torch.zeros((W + 1, d_qk))
    for c in range(W):
        query_matrix[c, c] = match_gain
    query_matrix[W, W] = alpha  # constant 1.0 × recency_scale for the position column

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
        d_head=rope.d_head,
        base=rope.base,
    )

    if assert_hardness_gt is not None:
        from torchwright.graph.asserts import assert_softmax_hardness

        attn = assert_softmax_hardness(attn, assert_hardness_gt)
    return attn
