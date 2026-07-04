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

``bos_indicator`` is a 1D node that is 1.0 at position 0 (BOS) and 0.0
everywhere else.  The caller derives it from the BOS token's embedding or a
graph-level one-hot.

The consumer of ``pos`` — :func:`~torchwright.ops.attention_ops.
attend_most_recent_globally`, the "most recent matching beyond the lobe
window" head — is machine-neutral attention hardware and lives in
``ops/attention_ops.py``.
"""

import math

import torch

from torchwright.graph import Attn, LiteralValue, Node, RopeConfig
from torchwright.graph.asserts import assert_in_range
from torchwright.ops._math import _N_BPS, _bisect_m, _theta_slow, _w_of_m
from torchwright.ops.relu.arithmetic_ops import piecewise_linear

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def global_position_from_bos(
    rope: RopeConfig,
    bos_indicator: Node,
) -> Node:
    """Approximate absolute position (0-indexed) via boosted-BOS attention.

    Builds one attention head where BOS carries a special feature of magnitude
    ``sqrt(ln(max_positions))`` on the slowest rotary plane, and all other keys
    carry zero.  At query position m the softmax weight at BOS is

        w(m) = MAX_LEN^cos(m·θ) / (MAX_LEN^cos(m·θ) + m)

    A log-uniform PWL table (``_N_BPS`` entries) inverts w → m.

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
    # The BOS feature must ride a ROTATED plane: the mechanism is the
    # cos(m·θ_slow) attenuation of the BOS score, and the "+m" in w(m)'s
    # denominator is the count of the m non-BOS keys (each scoring 0).  Placing it
    # on the unrotated NoPE tail would strip the cosine and mismatch the PWL table
    # (_w_of_m bakes in cos(m·θ)), so this head cannot delegate to
    # rotary_content_head under partial rotary (that routes the (1,1) matrix to the
    # tail).  Place A explicitly on the slowest rotated plane d_rot/2-1
    # (rotate_half partner d_rot-1 left zero); under full rotary that is d_head/2-1,
    # byte-identical to the old place_on_slow_planes((1,1)) layout.
    #   Q: constant 1.0 → A on the slowest rotated plane
    #   K: bos_indicator (1 at BOS, 0 elsewhere) → A·bos_ind on that plane
    #   V: bos_indicator (so output = w_BOS · 1 + Σ_others w_i · 0 = w_BOS)
    query_one = LiteralValue(torch.tensor([1.0]), name="bos_weight_query_one")
    plane = rope.d_rot // 2 - 1
    query_matrix = torch.zeros((1, rope.d_head))
    query_matrix[0, plane] = A
    key_matrix = torch.zeros((1, rope.d_head))
    key_matrix[0, plane] = A

    bos_weight = Attn(
        query_in=query_one,
        key_in=bos_indicator,
        value_in=bos_indicator,  # value: 1 at BOS, 0 elsewhere → output = w_BOS
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=torch.eye(1),
        output_matrix=torch.eye(1),
        rope_base=rope.base,
        rope_d_rot=rope.d_rot,
    )

    # --- PWL inverse: w → m ---
    # Build log-uniform breakpoints in w ∈ [w_min, 1] (ascending).
    w_min = _w_of_m(max_len, max_len, theta)
    ratio = (1.0 / w_min) ** (1.0 / (_N_BPS - 1))
    w_bps = [w_min * (ratio**k) for k in range(_N_BPS)]
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

