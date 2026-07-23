"""Arithmetic ops on the swish machine.

Each op assembles gated-FFN lanes per its entry in
``docs/ops_plain_english.md``; every numeric claim there is pinned by
``tests/docs/test_swish_constants.py``.
"""

import builtins
import math
from collections.abc import Callable
from typing import cast

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import min_d_hidden, scale, step_sharpness, swish_dip
from torchwright.ops.linear import (
    add_const,
    multiply_const,
    negate,
    subtract,
    sum_nodes,
)
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn

# Below this magnitude, treat a slope change as zero (avoids emitting a
# zero-weight gate lane / degenerate hinge).
_SLOPE_EPS = 1e-12

# A piecewise-linear function needs at least two breakpoints to define one
# segment.
_MIN_PIECEWISE_BREAKPOINTS = 2

# radix_floor_int's radix divisor D must be at least 2 (base-2 split).
_MIN_RADIX_DIVISOR = 2


def compare(
    inp: Node,
    thresh: float,
    true_level: float = 1.0,
    false_level: float = -1.0,
    sharpness: float | None = None,
) -> Node:
    """Compare input with threshold, returning true_level above and false_level below.

    ``true_level`` is +1 by default, ``false_level`` is -1. A saturating
    ramp built from two sharpened hinges
    (``hinge(z) = Swish(scale·z)/scale ≈ ReLU(z)``)::

        z       = sharpness · (inp - thresh)
        compare = false_level + (true_level - false_level)·(hinge(z) - hinge(z-1))

    The caller contract is unchanged from the ReLU form: the ramp is
    ``1/sharpness`` wide in input units — inputs at least ``1/sharpness``
    above ``thresh`` read true, inputs at or below ``thresh`` read false —
    and contract-point outputs are bit-exact in fp32 (the sigmoid's input
    there is ±scale, far past saturation).  What's new: the output is not
    exactly confined to ``[false_level, true_level]`` — inputs landing in
    a fillet (within ``~1.3/(scale·sharpness)`` of one of the two bends)
    can overshoot either level by up to
    ``swish_dip/scale · |true_level - false_level|`` (0.0022 at
    scale=128).  The value-range assert and the semantic bound both carry
    that slack, and every downstream ``c_tol`` budget must too.

    Args:
        inp: Node to compare. Must be length 1.
        thresh: Threshold to use.
        true_level: Value to return if inp is greater than thresh.
        false_level: Value to return if inp is less than thresh.
        sharpness: Override for the ramp sharpness.  Saturation requires
            ``(inp - thresh) * sharpness >= 1``, so the ramp width is
            ``1/sharpness`` in input units.  ``None`` (default) uses the
            module-level ``step_sharpness``.

    Returns:
        Node with a value of true_level if inp is greater than thresh,
        false_level otherwise.

    .. noise-footer::

       Max error: 1.999 abs, 1.999 rel over 8192 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"

    s = step_sharpness if sharpness is None else sharpness

    # Two degenerate lanes; scale·sharpness folds into the gate rows and
    # /scale into out_proj, so the sharpening is free of the value path.
    gate_proj = torch.tensor([[scale * s], [scale * s]])
    gate_bias = torch.tensor([-scale * s * thresh, -scale * (s * thresh + 1.0)])
    output_proj = (
        torch.tensor([[true_level - false_level], [false_level - true_level]]) / scale
    )
    output_bias = false_level * torch.ones(1)

    result = swiglu_ffn(
        inp,
        gate_proj,
        gate_bias,
        output_proj,
        output_bias,
        name="compare",
    )

    slack = swish_dip / scale * builtins.abs(true_level - false_level)
    lo = builtins.min(true_level, false_level)
    hi = builtins.max(true_level, false_level)
    result = assert_matches_value_type(
        result, NodeValueType(value_range=Range(lo - slack, hi + slack))
    )
    from torchwright.graph.affine_rules import (
        _apply_semantic_override,
        _compare_semantic_bound,
    )

    _apply_semantic_override(
        result,
        _compare_semantic_bound(
            inp.affine_bound, thresh, true_level, false_level, slack=slack
        ),
    )
    return result


def multiply(inp1: Node, inp2: Node) -> Node:
    """Multiply two live values.

        multiply(a, b) = Swish(a)·b + Swish(-a)·(-b)  =  a·b

    Exact for all ``a``, ``b`` — no range limit, no grid: the ± pair
    makes the Swish sigmoid factors cancel (``Swish(z) = z·sigma(z)`` and
    ``sigma(a) + sigma(-a) = 1``, so the two lanes sum to ``a·b``).  Both terms
    share the sign of ``a·b``, so they add constructively — no
    catastrophic cancellation.  This replaces the ReLU-era workarounds
    for multiplication (the quarter-square construction in
    ``multiply_2d``, the ``multiply_integers`` chain).

    Args:
        inp1: 1D scalar node — the gate-side operand.
        inp2: 1D scalar node — the up-side operand.

    Returns:
        1D scalar node containing ``inp1 * inp2``.

    .. noise-footer::

       Max error: 0.0009766 abs, 2.241e-07 rel over 8192 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp1) == 1, "Input must be a 1D scalar node"
    assert len(inp2) == 1, "Input must be a 1D scalar node"

    x = Concatenate([inp1, inp2])
    return swiglu_ffn(
        x,
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),  # gate rows +a / -a
        torch.zeros(2),
        torch.tensor([[1.0], [1.0]]),
        torch.zeros(1),
        up_proj=torch.tensor([[0.0, 1.0], [0.0, -1.0]]),  # up rows +b / -b
        up_bias=torch.zeros(2),
        name="multiply",
    )


def abs(inp: Node) -> Node:
    """Element-wise absolute value.

        abs(x) = Swish(scale·x)/scale + Swish(-scale·x)/scale  =  x·tanh(scale·x/2)

    The ReLU identity (``|x| = ReLU(x) + ReLU(-x)``, exact) with each
    ReLU replaced by the sharpened hinge.  Unlike ``multiply``'s ± pair,
    the two sigmoids here *add* instead of cancelling — an
    approximation, the rare op that regresses under swish (there is no
    exact swish form: ``|x|`` has a corner, and every finite sum of
    Swish lanes is smooth).  The error is one-sided and bounded: the
    output always lies in ``[0, |x|]`` — never negative, never above the
    true value.  Worst underestimate is ``2·swish_dip/scale`` (0.0044 at
    scale=128), hit at ``|x| = 1.278/scale``; for ``|x| ≳ 0.2`` tanh
    saturates and the op is bit-exact in fp32 — the entire integer
    grid.  Only consumers that need ``abs`` to *not under-read* near the
    origin (dividing by it, comparing it against a small threshold)
    must budget the ``2·swish_dip/scale``.

    Args:
        inp: Node of any width.

    Returns:
        Node of the same width containing ``|x|`` element-wise.

    .. noise-footer::

       Max error: 0.004351 abs, 0.9954 rel over 8192 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    d = len(inp)
    eye = torch.eye(d)
    return swiglu_ffn(
        inp,
        torch.cat([scale * eye, -scale * eye]),  # gate rows +scale·x / -scale·x
        torch.zeros(2 * d),
        torch.cat([eye, eye]) / scale,
        torch.zeros(d),
        name="abs",
    )


def min(inp1: Node, inp2: Node) -> Node:
    """Element-wise minimum of two nodes.

        min(a, b) = a - Swish(scale·(a-b))/scale        # a - hinge(a-b)

    Replaces the ReLU-era abs route (``(a+b-|a-b|)/2``) — under swish
    the two forms have identical error, so the choice falls to graph
    simplicity: the hinge form is self-contained, with no dependency on
    ``abs``'s budget.  The ``a`` pass-through is a sharpened bypass pair
    (``Swish(scale·a)/scale - Swish(-scale·a)/scale = a``, exact at any
    sharpening — the identity the ``mlp_bypass`` realization class
    relies on).  Error is one-sided: min is *over*-estimated by at most
    ``swish_dip/scale`` (0.0022 at scale=128), and only when
    ``|a-b| ≲ 0.2``; ties are exact (``Swish(0) = 0``).  The
    construction is asymmetric but the error is not: the hinge's gap to
    ReLU is an even function of ``a-b``.  fp note: min of far-apart
    magnitudes inherits the larger operand's relative fp error (the
    ``a - (a-b)`` cancellation), plus the folded ``/scale`` product
    rounding at the lane-contribution magnitude — both unchanged in
    class from the ReLU machine.

    Args:
        inp1: First node.
        inp2: Second node (same width as *inp1*).

    Returns:
        Node of the same width containing ``min(inp1, inp2)`` element-wise.

    .. noise-footer::

       Max error: 3.815e-06 abs, 1.953e-06 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp1) == len(inp2)
    d = len(inp1)
    eye = torch.eye(d)
    zero = torch.zeros(d, d)

    # 3 degenerate lanes per component: hinge on (a-b), then a's bypass
    # pair; the /scale folds into out_proj.
    gate_proj = scale * torch.cat(
        [
            torch.cat([eye, -eye], dim=1),  # hinge rows: +a, -b
            torch.cat([eye, zero], dim=1),  # bypass +a
            torch.cat([-eye, zero], dim=1),  # bypass -a
        ]
    )
    output_proj = torch.cat([-eye, eye, -eye]) / scale

    x = Concatenate([inp1, inp2])
    return swiglu_ffn(
        x,
        gate_proj,
        torch.zeros(3 * d),
        output_proj,
        torch.zeros(d),
        name="min",
    )


def square(inp: Node) -> Node:
    """Compute ``inp²``.

        square(x) = Swish(x)·x + Swish(-x)·(-x)  =  x²

    :func:`multiply` with both operands the same node — exact for all
    ``inp`` (see there for the ± cancellation).  Both terms are
    ``x²·sigma(±x)`` — non-negative, so they add cleanly.  Drops the
    ReLU-era ``[0, max_value]`` restriction, the ``step`` grid, and the
    huge near-zero relative error of the piecewise-linear version.

    Args:
        inp: 1D scalar node.

    Returns:
        1D scalar node containing ``inp²``.

    .. noise-footer::

       Max error: 3.052e-05 abs, 2.266e-07 rel over 8192 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"

    return swiglu_ffn(
        inp,
        torch.tensor([[1.0], [-1.0]]),  # gate rows +x / -x
        torch.zeros(2),
        torch.tensor([[1.0], [1.0]]),
        torch.zeros(1),
        up_proj=torch.tensor([[1.0], [-1.0]]),  # up rows +x / -x
        up_bias=torch.zeros(2),
        name="square",
    )


def _slope_change_hinges(
    breakpoints: list[float],
    values: list[list[float]],
    d_out: int,
    *,
    clamp: bool,
) -> list[tuple[float, float, list[float]]]:
    """Hinge list ``(input_weight, threshold, [output_weights_per_dim])``.

    One entry per breakpoint where the slope changes; a final entry
    cancels the last slope when *clamp* is true, and two extrapolation
    entries extend the end segments when it is false.
    """
    n = len(breakpoints)

    slopes = []
    for i in range(n - 1):
        dx = breakpoints[i + 1] - breakpoints[i]
        slopes.append([(values[i + 1][j] - values[i][j]) / dx for j in range(d_out)])

    relus: list[tuple[float, float, list[float]]] = []
    prev_slopes = [0.0] * d_out

    for i in range(n - 1):
        deltas = [slopes[i][j] - prev_slopes[j] for j in range(d_out)]
        if any(builtins.abs(d) > _SLOPE_EPS for d in deltas):
            relus.append((1.0, breakpoints[i], deltas))
        prev_slopes = list(slopes[i])

    # Cancel final slope (clamp)
    if any(builtins.abs(s) > _SLOPE_EPS for s in prev_slopes):
        relus.append((1.0, breakpoints[-1], [-s for s in prev_slopes]))

    if not clamp:
        if any(builtins.abs(s) > _SLOPE_EPS for s in slopes[0]):
            relus.append((-1.0, breakpoints[0], [-s for s in slopes[0]]))
        if any(builtins.abs(s) > _SLOPE_EPS for s in slopes[-1]):
            relus.append((1.0, breakpoints[-1], list(slopes[-1])))

    return relus


def piecewise_linear(
    inp: Node,
    breakpoints: list[float],
    fn: Callable[[float], float | list[float]],
    *,
    clamp: bool = True,
    d_max: int = min_d_hidden,
    input_scale: float = 1.0,
    name: str = "piecewise_linear",
) -> Node:
    """Evaluate a piecewise-linear function defined by breakpoints and a callable.

    The ReLU construction ported verbatim — one degenerate lane per slope
    *change* (equal-slope segments free), the clamp trick, ``d_max``
    chunking, vector-valued *fn* sharing lanes — with each ReLU replaced
    by the sharpened hinge ``Swish(K·z)/K``, ``K = scale·input_scale``
    (``input_scale``'s argument amplification is the same trick as
    sharpening, so the two knobs multiply into the gate rows)::

        f(x) = y_0 + Σ delta_m_i · hinge(x - x_i)

    Error structure: the exact PL function with each corner rounded in a
    radius-``~17/K`` fillet.  Outside the fillets it is bit-exact in fp32
    (segment interiors compute the exact interpolation and the function
    passes through every knot exactly, modulo the folded-projection ulp
    class); inside a fillet the error is ``≤ swish_dip·|delta_m_i|/K`` —
    it scales with the slope change, not the value magnitudes.  Two
    breakpoints closer than ``~34/K`` have overlapping fillets whose
    errors stack additively — the clamp range claim widens by the worst
    windowed sum (see below), and each call site's committed grid should
    be audited against ``34/K`` (staircase ops place hinge pairs
    ``1/sharpness`` apart — fine when ``input_scale ≈ sharpness``).
    Monotonicity is no longer exact: a monotone target acquires a dip of
    up to ``swish_dip·|delta_m|/K`` just before each rising ramp.  The
    chord error *between* breakpoints (curved *fn* approximated by
    segments) is untouched, and smooth-target grids need no redesign —
    the fillet bends toward the curve the corner was approximating.

    Args:
        inp: 1D scalar node.
        breakpoints: Strictly ascending x-coordinates (length n >= 2).
        fn: ``fn(x) -> float`` or ``fn(x) -> List[float]`` evaluated at
            each breakpoint.  Vector returns must all have the same
            length.
        clamp: If True (default), hold constant outside the range.
            If False, extrapolate linearly.
        d_max: Maximum lanes per FFN (chunks beyond this join via
            ``sum_nodes`` — Add hardware, the one place this op leaves
            the FFN).  Defaults to ``min_d_hidden``: a chunk must fit
            one MLP sublayer's hidden pool.
        input_scale: Multiplier for the gate-row weights.  Each hinge
            ``delta · hinge(x - b)`` is rewritten as
            ``(delta/s) · hinge(s·x - s·b)`` so the bias ``-s·b`` can be
            exact in float32 where ``-b`` is not; it also narrows the
            fillets (``K = scale·s``).  Use ``step_sharpness`` for
            step-function staircases.
        name: Debug label prefix.

    Returns:
        Node of width 1 (scalar fn) or D (vector fn).

    .. noise-footer::

       Max error: 0.25 abs, 146.3 rel over 16384 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    n = len(breakpoints)
    assert n >= _MIN_PIECEWISE_BREAKPOINTS, "Need >= 2 breakpoints"
    assert all(breakpoints[i] < breakpoints[i + 1] for i in range(n - 1)), (
        "Breakpoints must be strictly ascending"
    )

    raw_values = [fn(x) for x in breakpoints]

    scalar = not isinstance(raw_values[0], (list, tuple))
    values = (
        [[cast("float", v)] for v in raw_values]
        if scalar
        else [list(cast("list[float]", v)) for v in raw_values]
    )
    d_out = len(values[0])
    assert all(len(v) == d_out for v in values)

    relus = _slope_change_hinges(breakpoints, values, d_out, clamp=clamp)

    if len(relus) == 0:
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor(values[0]))

    y0 = torch.tensor(values[0])
    K = scale * input_scale

    chunks = []
    for chunk_start in range(0, len(relus), d_max):
        chunk = relus[chunk_start : chunk_start + d_max]
        d = len(chunk)

        gate_proj = torch.tensor([[r[0] * K] for r in chunk])  # (d, 1)
        gate_bias = torch.tensor([-r[0] * K * r[1] for r in chunk])  # (d,)
        output_proj = torch.tensor(
            [[r[2][j] / K for j in range(d_out)] for r in chunk]
        )  # (d, d_out)

        ob = y0 if chunk_start == 0 else torch.zeros(d_out)

        chunks.append(
            swiglu_ffn(
                inp,
                gate_proj,
                gate_bias,
                output_proj,
                ob,
                name=f"{name}_{chunk_start}_{chunk_start + d}",
            )
        )

    result = chunks[0] if len(chunks) == 1 else sum_nodes(chunks)

    # With clamp=True the exact PL output is bounded by the min/max of fn
    # at the breakpoints (per-channel); the swish fillets can dip past
    # that by the worst *windowed* sum of per-bend dips — fillets of
    # bends within 34/K of each other overlap and stack additively, so
    # the slack is max over bends of sum_{|x_j - x_i| <= 34/K}
    # swish_dip·|delta_m_j|/K per channel.  Well-spaced grids reduce to
    # the single worst bend's dip.
    if clamp:
        per_channel_los = [
            builtins.min(values[i][j] for i in range(n)) for j in range(d_out)
        ]
        per_channel_his = [
            builtins.max(values[i][j] for i in range(n)) for j in range(d_out)
        ]
        window = 34.0 / K
        xs = [r[1] for r in relus]
        slack = 0.0
        for i in range(len(relus)):
            for j in range(d_out):
                s = sum(
                    swish_dip * builtins.abs(r[2][j]) / K
                    for r, x in zip(relus, xs, strict=False)
                    if builtins.abs(x - xs[i]) <= window
                )
                slack = builtins.max(slack, s)
        lo = float(builtins.min(per_channel_los)) - slack
        hi = float(builtins.max(per_channel_his)) + slack
        result = assert_matches_value_type(
            result, NodeValueType(value_range=Range(lo, hi))
        )
    return result


def clamp(inp: Node, lo: float, hi: float) -> Node:
    """Clamp a scalar to [lo, hi] in a single FFN.

    Uses :func:`piecewise_linear` with 4 breakpoints to implement an
    identity passthrough in [lo, hi] with sharp clamping at the edges.
    Inherits piecewise_linear's entry: the two corners are
    ``1/step_sharpness`` apart with ``K = scale·step_sharpness``, so the
    fillets never overlap (spacing 0.1 vs 34/K = 0.034).

    Args:
        inp: 1D scalar node.
        lo: Lower bound.
        hi: Upper bound (must be > lo).

    Returns:
        1D scalar node clamped to [lo, hi].

    .. noise-footer::

       Max error: 1.907e-06 abs, 0.0002899 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert hi > lo, "hi must exceed lo"

    eps = 1.0 / step_sharpness
    return piecewise_linear(
        inp,
        [lo, lo + eps, hi - eps, hi],
        lambda x: x,
        input_scale=step_sharpness,
        name="clamp",
    )


def reciprocal(
    inp: Node,
    min_value: float,
    max_value: float,
    step: float = 1.0,
) -> Node:
    """Compute 1/x via piecewise-linear interpolation.

    Uses **geometric** breakpoint spacing so that relative interpolation
    error is roughly constant across the entire ``[min_value, max_value]``
    range; inherits :func:`piecewise_linear`'s entry.  A smooth-target
    grid: the fillets bend toward the curve the corners approximate, so
    the dense low-end grid (spacing well under ``34/K``) does not need a
    redesign — the stacked-dip slack lands in the range claim, and the
    measured noise entry is the authority on the net effect.

    Args:
        inp: 1D scalar node with value in [min_value, max_value].
        min_value: Lower bound on input (must be > 0).
        max_value: Upper bound on input.
        step: Controls breakpoint density.  Smaller step = more
            breakpoints = higher accuracy.

    Returns:
        1D scalar node containing 1/x.

    .. noise-footer::

       Max error: 0.0008245 abs, 0.1117 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert min_value > 0, "min_value must be positive"
    assert max_value > min_value, "max_value must exceed min_value"

    n_breakpoints = builtins.max(int((max_value - min_value) / step) + 1, 32)
    ratio = (max_value / min_value) ** (1.0 / (n_breakpoints - 1))
    breakpoints = [min_value * (ratio**k) for k in range(n_breakpoints)]
    breakpoints[0] = min_value
    breakpoints[-1] = max_value

    # Spacing audit (docs/ops_plain_english.md, piecewise_linear): the
    # geometric grid's smallest gap (at min_value) sits far below 34/scale,
    # so at input_scale=1 the fillets of ~tens of bends overlap and their
    # dips stack (~2e-2 measured at the default grid — a 20x regression on
    # the relu entry).  Raising input_scale multiplies into K and shrinks
    # the fillets below the smallest gap; the value path is K-neutral.
    min_gap = breakpoints[1] - breakpoints[0]
    input_scale = builtins.max(1.0, 34.0 / (scale * min_gap))

    return piecewise_linear(
        inp,
        breakpoints,
        lambda x: 1.0 / x,
        input_scale=input_scale,
        name="reciprocal",
    )


def thermometer_floor_div(inp: Node, divisor: int, max_value: int) -> Node:
    """Compute floor(inp / divisor) using a piecewise-linear staircase.

    Places a steep ramp at each multiple of the divisor, with
    half-integer thresholds (9.5 not 10.0) so integer inputs sit in the
    flat zones.  Inherits :func:`piecewise_linear`'s entry; the hinge
    pairs sit ``1/step_sharpness`` apart with
    ``K = scale·step_sharpness``, clearing the ``34/K`` spacing audit
    3x (the same closed-form fact as scalar_to_embedding's).

    .. warning::
       **Integer inputs only.**  Each ramp is centred *between* two
       valid integer outputs — a continuous float input near a bin
       boundary lands inside the ramp and interpolates.  Use
       ``floor_int`` for continuous scalars.

    Args:
        inp: 1D scalar node with **integer** value in [0, max_value].
        divisor: The divisor for floor division.
        max_value: Upper bound on input (determines number of steps).

    Returns:
        1D scalar node containing floor(inp / divisor).

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    n = max_value // divisor
    if n == 0:
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor([0.0]))

    eps = 1.0 / step_sharpness
    breakpoints = [0.0 - eps]
    for k in range(1, n + 1):
        threshold = k * divisor - 0.5  # Half-integer: 9.5, 19.5, ...
        breakpoints.extend([threshold - eps / 2, threshold + eps / 2])
    breakpoints.append(max_value + eps)

    def _staircase(x: float) -> float:
        return float(sum(1 for k in range(1, n + 1) if x > k * divisor - 0.5))

    result = piecewise_linear(
        inp,
        breakpoints,
        _staircase,
        input_scale=step_sharpness,
        name="thermometer_floor_div",
    )
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=Range(0.0, float(n))),
    )


def mod_const(inp: Node, divisor: int, max_value: int) -> Node:
    """Compute inp % divisor for non-negative integer inputs.

    Uses the identity ``x % d = x - d * floor(x / d)`` — a
    :func:`thermometer_floor_div` plus linear hardware.

    Args:
        inp: 1D scalar node with integer value in [0, max_value].
        divisor: The constant divisor (positive integer).
        max_value: Upper bound on input.

    Returns:
        1D scalar node containing inp % divisor.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert divisor > 0, "divisor must be positive"
    q = thermometer_floor_div(inp, divisor, max_value)
    return subtract(inp, multiply_const(q, float(divisor)))


def floor_int(
    inp: Node,
    min_value: int,
    max_value: int,
    sharpness: float | None = None,
    output_map: Callable[[int], float] | None = None,
) -> Node:
    """Compute floor(x) — or f(floor(x)) — for a continuous scalar input.

    **Not a flat staircase, and the depth is load-bearing.**  The
    single-FFN form piecewise_linear would emit sums ~n terms of
    magnitude ``sharpness·x`` in one projection, whose partial sums
    overflow fp32's 2^24 exact-integer window at production magnitudes
    and collapse.  The two-stage form keeps every accumulated term
    bounded — that constraint is about fp32 accumulation, not the
    activation; do not "simplify" this back to one layer::

        t_k    = sharpness·(x - k) + 1                # per boundary k
        step_k = hinge(t_k) - hinge(t_k - W)          # FFN 1: bounded step ∈ [0, W]
        floor  = min + n - Σ_k hinge(1 - step_k)      # FFN 2: count not-yet-ON steps

    ``hinge(1 - step_k)`` is the exact indicator ``[x < k]`` (1 while
    boundary ``k`` is un-crossed, 0 once crossed): stage 2 subtracts it
    from ``min + n`` so the result counts crossed boundaries, i.e.
    ``floor(x)``.

    **``output_map`` collapses a following piecewise-constant function
    into this op.**  Any ``g(floor(x))`` with ``g`` defined on the
    integers is itself piecewise-constant with breakpoints *at the same
    integers* floor_int already switches on — so instead of computing
    the floor and feeding it to a separate op, pass ``g`` as
    ``output_map`` and the downstream op disappears.  Writing
    ``δ_k = g(k) - g(k-1)`` and telescoping
    ``g(floor(x)) = g(min) + Σ_k δ_k·[x ≥ k] = g(max) - Σ_k δ_k·[x < k]``,
    stage 2 becomes::

        out = g(max) - Σ_k δ_k · hinge(1 - step_k)    # FFN 2, per-boundary δ_k

    i.e. exactly today's stage 2 with the all-ones output weights
    replaced by the ``δ_k`` and the closing constant ``min + n`` replaced
    by ``g(max)``.  The default (``output_map is None``) is the identity
    ``g(k) = k`` — every ``δ_k = 1``, ``g(max) = min + n`` — and the
    emitted weights are byte-identical to before.

    Contract unchanged from the ReLU form: inputs stay out of the
    ``1/sharpness``-wide ramp zone just below each boundary; the flat
    zone ``[k, k+1-1/sharpness]`` is the home of legal inputs.
    Flat-zone interiors and exact integer inputs are exact to the folded
    ulp class (at ``x = k`` the critical hinges sit exactly on a bend,
    where ``Swish(0) = 0``, or fully saturated).  The port adds fillet
    zones of width ``~17/(scale·sharpness)`` at each ramp edge
    contributing ``≤ swish_dip/scale`` apiece — at most a couple live at
    once, so the closing range claim carries ``2·swish_dip/scale`` of
    slack.  **The W-slack absorbs fillets too**: an ON step parks stage
    2's hinge argument at ``1 - W = -1`` — ``scale`` past saturation —
    so stage-1 fillet noise on an ON step still reads exactly 0 in stage
    2, the same mechanism that absorbs fp ulps today; the existing
    sizing ``W = max(2, 8·ulp(sharpness·n))`` already dominates the
    swish requirement ``W ≥ 1 + 17/scale``.

    **``output_map`` numerics — the δ-amplification.**  Reweighting exact
    0/1 indicators by constants is itself exact, so the *noise* story is
    the same saturating stage as the default path — but every error term
    that path carries is now multiplied by the local ``|δ_k|``:

    - *Near-boundary error.*  An input inside the ``1/sharpness`` ramp
      just below boundary ``k`` reads off-by-one in the floor and hence
      off-by-``δ_k`` in the output (vs off-by-1 for plain floor).  The
      composed op sees its breakpoints on the *raw* pre-floor ``x``,
      whereas a separate ``g`` applied to the snapped integer floor never
      sees its own transition band — but this introduces **no new** band,
      because ``g``'s breakpoints are a subset of floor's (the integers):
      the only ramp zones are floor's own, unchanged in width and
      location.  The restriction the caller inherits is exactly floor's:
      keep inputs out of the sub-integer ramp; there the error scales
      with ``|δ_k|`` instead of 1.
    - *Fillet slack.*  The closing range carries ``2·swish_dip/scale ·
      max_k|δ_k|`` (a couple fillets live at once, each now weighted by
      its boundary's ``|δ_k|``) instead of the plain ``2·swish_dip/scale``.
    - *Chunk-sum pin.*  Each chunk's stage-2 sum is pinned to the
      ``δ_k``-weighted hinge bounds (see the per-chunk assert); the fp32
      accumulation pad grows to ``max_k|δ_k|`` per chunk, which still
      dominates the true ``~c·max|δ_k|·ulp`` rounding by orders of
      magnitude.

    ``output_map`` must be defined (and finite) on every integer in
    ``[min_value, max_value]``.

    Args:
        inp: 1D scalar node with value in [min_value, max_value].
        min_value: Lower bound (integer).
        max_value: Upper bound (integer).
        sharpness: Override the global ``step_sharpness`` for this op.
            Higher values narrow the ramp zone at each boundary.
        output_map: Optional ``g`` applied as ``g(floor(x))``.  Must be
            defined on every integer in ``[min_value, max_value]``.  When
            ``None`` the op returns ``floor(x)`` and emits weights
            byte-identical to the pre-``output_map`` form.

    Returns:
        1D scalar node containing ``floor(x)`` (default) or
        ``output_map(floor(x))``.

    .. noise-footer::

       Max error: 100.7 abs, 0.9531 rel over 12288 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_value >= min_value

    # δ_k = g(k) - g(k-1) per boundary; the identity g leaves every δ_k = 1
    # and g(max) = min + n, reproducing the plain-floor weights exactly.
    g = (float) if output_map is None else output_map

    def _delta(k: int) -> float:
        return float(g(k)) - float(g(k - 1))

    if max_value == min_value:
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor([float(g(min_value))]))

    s = float(sharpness if sharpness is not None else step_sharpness)
    n = max_value - min_value
    # Stage-1 lanes are 2 per boundary: a chunk must fit one MLP
    # sublayer's hidden pool (min_d_hidden).
    _CHUNK = min_d_hidden // 2

    max_t = s * n + 1.0
    ulp = 2.0 ** (math.floor(math.log2(max_t)) - 23) if max_t >= 1.0 else 2.0**-23
    step_cap = builtins.max(2.0, 8.0 * ulp)  # W

    neg_partials = []  # each chunk contributes -Σ_k δ_k·hinge(1 - step_k)
    for c0 in range(0, n, _CHUNK):
        ks = list(
            range(
                min_value + 1 + c0,
                builtins.min(min_value + 1 + c0 + _CHUNK, max_value + 1),
            )
        )
        c = len(ks)
        # stage 1: step_k = hinge(t_k) - hinge(t_k - W), t_k = s·x - s·k + 1;
        # scale folds into the gate rows, /scale into out_proj.
        gate_proj = torch.full((2 * c, 1), scale * s)
        gate_bias = torch.empty(2 * c)
        out_proj = torch.zeros((2 * c, c))
        for j, k in enumerate(ks):
            gate_bias[2 * j] = scale * (1.0 - s * k)  # t_k
            gate_bias[2 * j + 1] = scale * (1.0 - step_cap - s * k)  # t_k - W
            out_proj[2 * j, j] = 1.0 / scale
            out_proj[2 * j + 1, j] = -1.0 / scale
        step = swiglu_ffn(
            inp,
            gate_proj,
            gate_bias,
            out_proj,
            torch.zeros(c),
            name="floor_int_step",
        )
        # Pin the stage's true range: hinge(z) <= relu(z) and hinge(z) >=
        # relu(z) - swish_dip/scale pointwise, so step_k = hinge(t_k) -
        # hinge(t_k - W) lies in [-dip, W + dip] for ANY input — garbage
        # rows included, no ramp-zone assumption.  The computed fp32 value
        # additionally carries folded-gate rounding of the ulp(s·n) class —
        # the same class the W = max(2, 8·ulp) sizing absorbs — so the
        # claim widens by W/4 per side (measured overshoot ~W/1000; W/4 is
        # ~2x the sizing's own worst case).  Without the pin, the affine
        # relaxation declares ~sharpness·range here (~1e16 on a production
        # floor), and the rms_norm residual-energy certifier — which reads
        # every residual-resident intermediate, not just the op's asserted
        # output — blows its fp32-feasible budget.
        dip = swish_dip / scale
        fp_slack = step_cap / 4.0
        step = assert_matches_value_type(
            step,
            NodeValueType(
                value_range=Range(-dip - fp_slack, step_cap + dip + fp_slack)
            ),
            atol=1e-5,
        )
        # stage 2: -Σ_k δ_k·hinge(1 - step_k), single output column.  Default
        # δ_k = 1 makes out_proj byte-identical to the old -ones/scale.
        dlist = [_delta(k) for k in ks]
        deltas = torch.tensor(dlist)
        neg_partial = swiglu_ffn(
            step,
            -scale * torch.eye(c),
            scale * torch.ones(c),
            (-deltas / scale).unsqueeze(1),
            torch.zeros(1),
            name="floor_int_saturate",
        )
        # Same universal bound one level up: 1 - step_k ∈ [1-W-dip, 1+dip],
        # so hinge(1 - step_k) ∈ [-dip, 1+dip]; weighting lane k by δ_k, its
        # term -δ_k·hinge lies in [min(δ_k·dip, -δ_k(1+dip)), max(...)].  Sum
        # the per-lane extremes, then pad by max|δ_k| for the summed fp32
        # rounding — that pad dominates the true ~c·max|δ_k|·ulp by orders of
        # magnitude.  Default δ_k = 1 recovers [-c(1+dip)-1, c·dip+1].
        lo_sum = sum(builtins.min(d * dip, -d * (1.0 + dip)) for d in dlist)
        hi_sum = sum(builtins.max(d * dip, -d * (1.0 + dip)) for d in dlist)
        pad = builtins.max(1.0, *(builtins.abs(d) for d in dlist))
        neg_partials.append(
            assert_matches_value_type(
                neg_partial,
                NodeValueType(value_range=Range(lo_sum - pad, hi_sum + pad)),
                atol=1e-5,
            )
        )

    summed = neg_partials[0] if len(neg_partials) == 1 else sum_nodes(neg_partials)
    result = add_const(summed, float(g(max_value)))  # g(max) - Σ δ_k·hinge(1 - step_k)
    # Output spans g over the integer floors; the couple-of-fillets closing
    # slack is δ-amplified, so it scales with max|δ_k| (default max|δ_k| = 1
    # recovers the plain [min - 2·dip, max + 2·dip]).
    g_vals = [float(g(j)) for j in range(min_value, max_value + 1)]
    max_abs_delta = builtins.max(
        builtins.abs(_delta(k)) for k in range(min_value + 1, max_value + 1)
    )
    slack = 2.0 * swish_dip / scale * max_abs_delta
    return assert_matches_value_type(
        result,
        NodeValueType(
            value_range=Range(
                builtins.min(g_vals) - slack, builtins.max(g_vals) + slack
            )
        ),
    )


def ceil_int(
    inp: Node,
    min_value: int,
    max_value: int,
    sharpness: float | None = None,
) -> Node:
    """Compute ceil(x) using the identity ``ceil(x) = -floor(-x)``.

    Inherits :func:`floor_int`'s entry (contract, two-stage depth,
    fillet slack).

    Args:
        inp: 1D scalar node with value in [min_value, max_value].
        min_value: Lower bound (integer).
        max_value: Upper bound (integer).
        sharpness: Override the global ``step_sharpness`` for the inner
            ``floor_int``.

    Returns:
        1D scalar node containing ceil(x).

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    return negate(floor_int(negate(inp), -max_value, -min_value, sharpness=sharpness))


def radix_floor_int(
    inp: Node,
    min_value: int,
    max_value: int,
    divisor: int | None = None,
    sharpness: float | None = None,
    hi_sharpness: float | None = None,
) -> Node:
    r"""Compute floor(x) as a radix split of three small :func:`floor_int`\\ s.

    A flat ``floor_int`` over ``N = max_value - min_value`` boundaries
    costs ``3N`` hidden lanes in 2 sublayers and holds an up-to-512-wide
    step vector live on the residual between its stages.  This form
    splits with divisor ``D`` (default: the power of two nearest
    ``√(2N)``)::

        hi_raw = floor_int((x - min) / D)  # ⌈N/D⌉ boundaries
        hi = floor_int(hi_raw + 0.5)  # integer snap, ⌈N/D⌉+1
        lo = floor_int(x - min - D·hi)  # over [-1, D]: D+1
        floor = min + D·hi + lo

    Cost: ``≈ 3·(2·⌈N/D⌉ + D + 2)`` lanes (≈ ``8.5·√N`` at the default
    divisor, vs ``3N`` flat), 6 FFN sublayers plus 3 one-wide Linears
    (vs 2 sublayers flat), and residual intermediates at most
    ``2·max(⌈N/D⌉, D+1)`` wide (vs ``min(512, N)``).  Use it where the
    lane/width saving matters and the chain has depth slack; ``floor_int``
    remains the right op on zero-slack chains.

    **Why the snap makes the split exact (the digit-quad argument,
    extended).**  The hi floor's ramp sits just below each multiple of
    D, where it emits a *fractional* ``hi_raw``; the low part would
    amplify that fraction by D (the boundary-sliver hazard of the
    two-digit emit split).  Rounding ``hi_raw`` to the nearest integer
    (add 0.5, floor) collapses it to one of the two neighboring
    integers — and *either* neighbor reconstructs ``floor(x)`` exactly,
    because ``lo`` is floored over the extended range ``[-1, D]`` and its
    input ``x - min - D·hi`` is computed from ``x`` directly, so it
    carries x's own fractional part: with ``hi`` one too high, ``lo``
    lands in ``[-1, 0)`` and compensates; one too low, in ``[D-1, D)``.
    Unlike the emit digit-quad — whose low byte is recovered affinely
    and therefore inherits a ±1-step truncation in the sliver — the
    composed floors reconstruct *exactly* throughout the hi ramp.

    Contract (matches ``floor_int``, no new legal-input restriction):

    - Legal inputs stay out of the ``1/sharpness``-wide ramp just below
      each **integer** — the LO floor's ramp, the same zone flat
      ``floor_int`` excludes.  There, the result is exact to the folded
      ulp class: hi is a snapped exact integer, ``D·hi`` is exact
      (D a small power of two), the one-Linear ``x - min - D·hi``
      rounds at ~ulp(D), well inside lo's flat zone.
    - An input *inside* that ramp yields a fractional value within the
      same ±1-step window as flat ``floor_int`` — no D-amplification.
    - Residual hazard, by the digit-quad's product-of-slivers argument:
      the snap's own half-integer ramp is hit only when the hi ramp
      already produced a fraction within ``1/step_sharpness`` of 0.5 —
      two independent slivers (~``D/(hi_sharpness·N) x
      1/step_sharpness`` of the range), where the D-amplified error
      survives.  Callers who care push ``hi_sharpness`` up, exactly as
      the emit digit-quad does (``_DQ_HI_SHARPNESS``).

    Args:
        inp: 1D scalar node with value in [min_value, max_value].
        min_value: Lower bound (integer).
        max_value: Upper bound (integer).
        divisor: Radix divisor D ≥ 2.  Default: the power of two
            nearest ``√(2N)`` in log space (a power of two keeps
            ``x/D`` and ``D·hi`` exact in fp32).
        sharpness: Ramp sharpness for the LO floor — the knob that sets
            the op's precision window, same meaning as ``floor_int``'s.
        hi_sharpness: Ramp sharpness for the HI and snap floors.
            Defaults to ``sharpness``; raise it to shrink the
            product-of-slivers hazard window.

    Returns:
        1D scalar node containing floor(x).

    .. noise-footer::

       Max error: 1 abs, 1 rel over 12288 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_value >= min_value
    n = max_value - min_value
    if divisor is None:
        # Power of two nearest sqrt(2n) in log space: minimizes
        # 2*ceil(n/D) + D over powers of two.
        divisor = (
            2 ** builtins.max(1, round(0.5 * math.log2(2 * n)))
            if n >= _MIN_RADIX_DIVISOR
            else 2
        )
    d = int(divisor)
    assert d >= _MIN_RADIX_DIVISOR, "divisor must be >= 2"
    if n <= d:
        # Nothing to split: the flat form is already at or below the
        # composed form's lo-floor cost.
        return floor_int(inp, min_value, max_value, sharpness=sharpness)

    hi_s = hi_sharpness if hi_sharpness is not None else sharpness
    n_hi = -(-n // d)  # ceil(n/d): hi ∈ [0, n_hi]

    # (x - min)/D — one 1-wide Linear; exact when D is a power of two.
    hi_in = Linear(
        inp,
        torch.tensor([[1.0 / d]], dtype=torch.float32),
        torch.tensor([-float(min_value) / d], dtype=torch.float32),
        name="radix_floor_hi_div",
    )
    hi_raw = floor_int(hi_in, 0, n_hi, sharpness=hi_s)
    # Integer snap: round-to-nearest via floor(hi_raw + 0.5).
    hi_half = Linear(
        hi_raw,
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
        name="radix_floor_hi_snap_half",
    )
    hi = floor_int(hi_half, 0, n_hi + 1, sharpness=hi_s)
    # lo input x - min - D·hi as ONE Linear over (x, hi) — the chained
    # subtract/multiply_const form leaves extra unfusable Linears.
    lo_in = Linear(
        Concatenate([inp, hi]),
        torch.tensor([[1.0], [-float(d)]], dtype=torch.float32),
        torch.tensor([-float(min_value)], dtype=torch.float32),
        name="radix_floor_lo_in",
    )
    # [-1, D]: the extended range that absorbs both snap outcomes.
    lo = floor_int(lo_in, -1, d, sharpness=sharpness)
    result = Linear(
        Concatenate([hi, lo]),
        torch.tensor([[float(d)], [1.0]], dtype=torch.float32),
        torch.tensor([float(min_value)], dtype=torch.float32),
        name="radix_floor_recombine",
    )
    # Same closing claim as floor_int, widened one step down: an
    # in-ramp input can land just below its integer (lo = -1 against
    # the true floor's min), exactly the flat form's ±1-step window.
    slack = 2.0 * swish_dip / scale
    return assert_matches_value_type(
        result,
        NodeValueType(
            value_range=Range(float(min_value) - 1.0 - slack, float(max_value) + slack)
        ),
    )
