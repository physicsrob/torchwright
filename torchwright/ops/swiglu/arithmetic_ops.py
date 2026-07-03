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
       measured at commit 95cf02b. See docs/numerical_noise.md.
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
       measured at commit 95cf02b. See docs/numerical_noise.md.
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
       measured at commit 95cf02b. See docs/numerical_noise.md.
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
