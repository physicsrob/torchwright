"""Arithmetic ops on the swish machine.

Each op assembles gated-FFN lanes per its entry in
``docs/ops_plain_english.md``; every numeric claim there is pinned by
``tests/docs/test_swish_constants.py``.
"""

import torch

from torchwright.graph import Concatenate, Node
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn


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
       measured at commit 7a0636d. See docs/numerical_noise.md.
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
       measured at commit 7a0636d. See docs/numerical_noise.md.
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
