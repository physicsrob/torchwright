"""Global-position recovery on the swish machine.

``global_position_from_bos`` is attention hardware plus one
piecewise-linear inversion table — only the PL stage is
machine-specific, so this module rebuilds the same boosted-BOS head (the
mechanism and its monotonicity guard are identical) and inverts the BOS
weight through the swiglu :func:`piecewise_linear`.  The pure-math
helpers (``_theta_slow``, ``_w_of_m``, ``_bisect_m``) and the table size
are shared with the frozen relu module — they contain no machine choice.

The inversion table is this library's densest committed grid: 1024
log-uniform breakpoints on ``w ∈ [w_min, 1]``, far below the ``34/K``
fillet-spacing bound at ``K = scale``.  It is a smooth-target grid
(the fillets bend toward the true inverse curve), and the closing
``assert_in_range(atol=0.1)`` — ported unchanged — is the empirical
budget: the unit test measures the end-to-end position error against
exact math on a full rollout, the same way the relu op is validated.
"""

import math

import torch

from torchwright.graph import Attn, LiteralValue, Node, RopeConfig
from torchwright.graph.asserts import assert_in_range
from torchwright.ops.const import scale
from torchwright.ops.relu.global_recency import (
    _N_BPS,
    _bisect_m,
    _theta_slow,
    _w_of_m,
)
from torchwright.ops.swiglu.arithmetic_ops import piecewise_linear


def global_position_from_bos(
    rope: RopeConfig,
    bos_indicator: Node,
) -> Node:
    """Approximate absolute position (0-indexed) via boosted-BOS attention.

    Builds one attention head where BOS carries a special feature of
    magnitude ``sqrt(ln(max_positions))`` on the slowest rotary plane,
    and all other keys carry zero.  At query position m the softmax
    weight at BOS is

        w(m) = MAX_LEN^cos(m·θ) / (MAX_LEN^cos(m·θ) + m)

    A log-uniform PWL table (``_N_BPS`` entries, swiglu hinges) inverts
    w → m.  See the relu twin for the full mechanism notes; everything
    but the inversion's activation is identical.

    Args:
        rope: RoPE config — ``d_head``, ``base``, ``max_positions``.
        bos_indicator: length-1 node; 1.0 at position 0 (BOS), 0.0
            elsewhere.

    Returns:
        length-1 node whose value at position m approximates m (float,
        in ``[0, max_positions]``).
    """
    assert len(bos_indicator) == 1, "bos_indicator must be 1-D"

    max_len = rope.max_positions
    theta = _theta_slow(rope)

    # Guard: w(m) is monotone only while the slowest plane rotates less
    # than π/2 over the full rollout (see the relu twin).
    product = max_len * theta
    if product >= math.pi / 2:
        raise ValueError(
            f"global_position_from_bos requires max_positions × theta_slow < π/2 "
            f"for the BOS weight to be monotone, but got "
            f"max_positions={max_len} × theta_slow={theta:.3e} = {product:.3f} ≥ {math.pi/2:.3f}.  "
            f"Increase d_head or base, or reduce max_positions."
        )

    A = math.sqrt(math.log(max_len))  # sqrt(ln(MAX_LEN)) so Q·K = ln(MAX_LEN)

    # Boosted-BOS attention head, identical to the relu twin: the BOS
    # feature rides the slowest *rotated* plane so its cos(m·θ_slow)
    # attenuation matches the inversion table.
    query_one = LiteralValue(torch.tensor([1.0]), name="bos_weight_query_one")
    plane = rope.d_rot // 2 - 1
    query_matrix = torch.zeros((1, rope.d_head))
    query_matrix[0, plane] = A
    key_matrix = torch.zeros((1, rope.d_head))
    key_matrix[0, plane] = A

    bos_weight = Attn(
        query_in=query_one,
        key_in=bos_indicator,
        value_in=bos_indicator,  # output = w_BOS · 1 + Σ_others w_i · 0
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=torch.eye(1),
        output_matrix=torch.eye(1),
        rope_base=rope.base,
        rope_d_rot=rope.d_rot,
    )

    # PWL inverse: w → m, log-uniform breakpoints in w ∈ [w_min, 1].
    w_min = _w_of_m(max_len, max_len, theta)
    ratio = (1.0 / w_min) ** (1.0 / (_N_BPS - 1))
    w_bps = [w_min * (ratio**k) for k in range(_N_BPS)]
    w_bps[-1] = 1.0  # exact endpoint

    # Spacing audit (docs/ops_plain_english.md, piecewise_linear): near
    # w = 1 the inverse slope is O(max_len) and the log grid's spacing is
    # ~0.01 — far under 34/scale — so at input_scale=1 the stacked fillets
    # mis-read position 0 by ~10 positions (measured).  Derive input_scale
    # from the grid's smallest gap (at w_min) so no two fillets overlap;
    # K multiplies out of the value path, and the gate magnitudes
    # (~K·1 ≈ 1e8) stay well inside fp32's exact-integer window.
    min_gap = w_bps[1] - w_bps[0]
    input_scale = max(1.0, 34.0 / (scale * min_gap))

    # clamp=False: suppress piecewise_linear's auto range assertion, as
    # in the relu twin (fp32 accumulation rounds slightly below zero at
    # position 0); assert_in_range(atol=0.1) is the ported budget.
    raw = piecewise_linear(
        bos_weight,
        w_bps,
        lambda w: _bisect_m(w, max_len, theta),
        clamp=False,
        input_scale=input_scale,
        name="bos_weight_to_position",
    )
    return assert_in_range(raw, 0.0, float(max_len), atol=0.1)
