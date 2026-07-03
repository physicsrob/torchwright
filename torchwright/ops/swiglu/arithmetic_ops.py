"""Arithmetic ops on the swish machine.

Each op assembles gated-FFN lanes per its entry in
``docs/ops_plain_english.md``; every numeric claim there is pinned by
``tests/docs/test_swish_constants.py``.
"""

import builtins

import torch

from torchwright.graph import Concatenate, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import scale, step_sharpness, swish_dip
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
       measured at commit 23fee36. See docs/numerical_noise.md.
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
       measured at commit 23fee36. See docs/numerical_noise.md.
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
       measured at commit 23fee36. See docs/numerical_noise.md.
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
       measured at commit 23fee36. See docs/numerical_noise.md.
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
       measured at commit 23fee36. See docs/numerical_noise.md.
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
