"""Arithmetic ops on the swish machine.

Each op assembles gated-FFN lanes per its entry in
``docs/ops_plain_english.md``; every numeric claim there is pinned by
``tests/docs/test_swish_constants.py``.
"""

import builtins
from typing import List

import torch

from torchwright.graph import Concatenate, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import min_d_hidden, scale, step_sharpness, swish_dip

# Purely linear ops (no activation) — machine-neutral in substance, shared
# with the frozen relu package until its retirement relocates them.
from torchwright.ops.relu.arithmetic_ops import (
    multiply_const,
    subtract,
    sum_nodes,
)
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn


def compare(
    inp: Node,
    thresh: float,
    true_level: float = 1.0,
    false_level: float = -1.0,
    sharpness: float | None = None,
) -> Node:
    """Compare input with threshold: ``true_level`` (+1 by default) if
    ``inp`` is above ``thresh``, ``false_level`` (-1) if below.

    A saturating ramp built from two sharpened hinges
    (``hinge(z) = Swish(scale·z)/scale ≈ ReLU(z)``)::

        z       = sharpness · (inp − thresh)
        compare = false_level + (true_level − false_level)·(hinge(z) − hinge(z−1))

    The caller contract is unchanged from the ReLU form: the ramp is
    ``1/sharpness`` wide in input units — inputs at least ``1/sharpness``
    above ``thresh`` read true, inputs at or below ``thresh`` read false —
    and contract-point outputs are bit-exact in fp32 (the sigmoid's input
    there is ±scale, far past saturation).  What's new: the output is not
    exactly confined to ``[false_level, true_level]`` — inputs landing in
    a fillet (within ``~1.3/(scale·sharpness)`` of one of the two bends)
    can overshoot either level by up to
    ``swish_dip/scale · |true_level − false_level|`` (0.0028 at
    scale=100).  The value-range assert and the semantic bound both carry
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
       measured at commit 3d08c2e. See docs/numerical_noise.md.
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
            inp._affine_bound, thresh, true_level, false_level, slack=slack
        ),
    )
    return result


def multiply(inp1: Node, inp2: Node) -> Node:
    """Multiply two live values.

        multiply(a, b) = Swish(a)·b + Swish(-a)·(-b)  =  a·b

    Exact for all ``a``, ``b`` — no range limit, no grid: the ± pair
    makes the Swish sigmoid factors cancel (``Swish(z) = z·σ(z)`` and
    ``σ(a) + σ(-a) = 1``, so the two lanes sum to ``a·b``).  Both terms
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
       measured at commit 3d08c2e. See docs/numerical_noise.md.
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
    true value.  Worst underestimate is ``2·swish_dip/scale`` (0.0056 at
    scale=100), hit at ``|x| = 1.278/scale``; for ``|x| ≳ 0.2`` tanh
    saturates and the op is bit-exact in fp32 — the entire integer
    grid.  Only consumers that need ``abs`` to *not under-read* near the
    origin (dividing by it, comparing it against a small threshold)
    must budget the ``2·swish_dip/scale``.

    Args:
        inp: Node of any width.

    Returns:
        Node of the same width containing ``|x|`` element-wise.

    .. noise-footer::

       Max error: 0.005569 abs, 0.9964 rel over 8192 samples;
       measured at commit 3d08c2e. See docs/numerical_noise.md.
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
    ``swish_dip/scale`` (0.0028 at scale=100), and only when
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

       Max error: 1.144e-05 abs, 6.759e-06 rel over 4096 samples;
       measured at commit 3d08c2e. See docs/numerical_noise.md.
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
    ``x²·σ(±x)`` — non-negative, so they add cleanly.  Drops the
    ReLU-era ``[0, max_value]`` restriction, the ``step`` grid, and the
    huge near-zero relative error of the piecewise-linear version.

    Args:
        inp: 1D scalar node.

    Returns:
        1D scalar node containing ``inp²``.

    .. noise-footer::

       Max error: 3.052e-05 abs, 2.266e-07 rel over 8192 samples;
       measured at commit 3d08c2e. See docs/numerical_noise.md.
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

def piecewise_linear(
    inp: Node,
    breakpoints: List[float],
    fn,
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

       Max error: 0.25 abs, 139.7 rel over 4096 samples;
       measured at commit 3d08c2e. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    n = len(breakpoints)
    assert n >= 2, "Need >= 2 breakpoints"
    assert all(
        breakpoints[i] < breakpoints[i + 1] for i in range(n - 1)
    ), "Breakpoints must be strictly ascending"

    raw_values = [fn(x) for x in breakpoints]

    scalar = not isinstance(raw_values[0], (list, tuple))
    if scalar:
        values = [[v] for v in raw_values]
    else:
        values = [list(v) for v in raw_values]
    d_out = len(values[0])
    assert all(len(v) == d_out for v in values)

    slopes = []
    for i in range(n - 1):
        dx = breakpoints[i + 1] - breakpoints[i]
        slopes.append([(values[i + 1][j] - values[i][j]) / dx for j in range(d_out)])

    # Hinge list: (input_weight, threshold, [output_weights_per_dim])
    relus: list = []
    prev_slopes = [0.0] * d_out

    for i in range(n - 1):
        deltas = [slopes[i][j] - prev_slopes[j] for j in range(d_out)]
        if any(builtins.abs(d) > 1e-12 for d in deltas):
            relus.append((1.0, breakpoints[i], deltas))
        prev_slopes = list(slopes[i])

    # Cancel final slope (clamp)
    if any(builtins.abs(s) > 1e-12 for s in prev_slopes):
        relus.append((1.0, breakpoints[-1], [-s for s in prev_slopes]))

    if not clamp:
        if any(builtins.abs(s) > 1e-12 for s in slopes[0]):
            relus.append((-1.0, breakpoints[0], [-s for s in slopes[0]]))
        if any(builtins.abs(s) > 1e-12 for s in slopes[-1]):
            relus.append((1.0, breakpoints[-1], list(slopes[-1])))

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

    if len(chunks) == 1:
        result = chunks[0]
    else:
        result = sum_nodes(chunks)

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
                    for r, x in zip(relus, xs)
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

       Max error: 2.861e-06 abs, 0.000331 rel over 4096 samples;
       measured at commit 3d08c2e. See docs/numerical_noise.md.
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
       measured at commit 3d08c2e. See docs/numerical_noise.md.
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
       measured at commit 3d08c2e. See docs/numerical_noise.md.
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

    def _staircase(x):
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
       measured at commit 3d08c2e. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert divisor > 0, "divisor must be positive"
    q = thermometer_floor_div(inp, divisor, max_value)
    return subtract(inp, multiply_const(q, float(divisor)))

