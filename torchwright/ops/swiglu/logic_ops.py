"""Logic ops on the swish machine.

The bool ops are pure :func:`~torchwright.ops.swiglu.arithmetic_ops.compare`
compositions — they inherit compare's entry in ``docs/ops_plain_english.md``,
not their own.  ``equals_vector`` is a one-sided compare on a dot product;
``cond_gate`` is the one-branch gated select.
"""

from typing import List

import torch

from torchwright.graph import Concatenate, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import embedding_step_sharpness, scale, swish_dip

# sum_nodes is purely linear (Add hardware, no activation) — machine-neutral
# in substance, shared with the frozen relu package until its retirement
# relocates the linear ops.
from torchwright.ops.relu.arithmetic_ops import sum_nodes
from torchwright.ops.swiglu.arithmetic_ops import compare
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn

_GATE_C_TOL = 0.005
"""Maximum acceptable deviation of ``|cond|`` from 1 in the gated ops
(``cond_gate``, ``select``, ``switch``).  A saturated swish gate is linear in
the cond, so a cond off ±1 by δ mis-scales the gated value by exactly
δ·|actual value| — the semantic bounds widen by that relative amount (no
range maximum ``M`` anywhere).  0.005 matches typical far-field compare
noise and clears compare's fillet dip (0.0028) with margin."""


def _assert_cond_pm1(cond: Node, c_tol: float = _GATE_C_TOL) -> Node:
    """Wrap ``cond`` in the gated ops' shared ±1 assert."""
    from torchwright.graph.misc import Assert

    def _cond_check(x: torch.Tensor) -> tuple:
        deviation = (x.abs() - 1.0).abs()
        bad = deviation > c_tol
        if not bad.any():
            return True, ""
        from torchwright.graph.asserts import _format_bad

        return False, (f"expected ||cond| - 1| <= {c_tol}; {_format_bad(x, bad)}")

    return Assert(
        cond,
        _cond_check,
        message=f"cond near ±1 (c_tol={c_tol})",
        claimed_type=NodeValueType(value_range=Range(-1.0 - c_tol, 1.0 + c_tol)),
    )


def bool_any_true(inp_list: List[Node]) -> Node:
    """
    Returns a node that evaluates to True if any of the input nodes are true.

    Args:
        inp_list (List[Node]): List of nodes to be evaluated.

    Returns:
        Node: Output node that is True if any input nodes are true, otherwise False.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit 23fee36. See docs/numerical_noise.md.
    """
    # Convert all the values to 1.0 if they're > 0.0 and 0.0 otherwise
    # then sum them, and if the sum is > 0.5, return 1.0, otherwise -1.0.
    # Each 0/1 indicator carries compare's dip slack (±swish_dip/scale),
    # so the sum sits within N·0.0028 of an integer — far outside the
    # final compare's (0.5, 0.6) ramp.
    sum_node = sum_nodes(
        [compare(n, thresh=0.0, true_level=1.0, false_level=0.0) for n in inp_list]
    )
    return compare(sum_node, thresh=0.5, true_level=1.0, false_level=-1.0)


def bool_all_true(inp_list: List[Node]) -> Node:
    """
    Returns a node that evaluates to True if all of the input nodes are true.

    Inputs must be clean ±1.0 booleans (as produced by compare/bool_* ops).
    Sum of N such inputs is +N only when all are +1; otherwise ≤ N-2.
    A threshold at N-1 cleanly separates the two cases.

    Args:
        inp_list (List[Node]): List of nodes to be evaluated.

    Returns:
        Node: Output node that is True if all input nodes are true, otherwise False.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit 23fee36. See docs/numerical_noise.md.
    """
    return compare(
        sum_nodes(inp_list),
        thresh=len(inp_list) - 1.0,
        true_level=1.0,
        false_level=-1.0,
    )


def bool_not(inp: Node) -> Node:
    """
    Returns a node that evaluates to 1.0 if the input node is false, and -1.0 if the input node is true.

    Args:
        inp: Input node to be evaluated

    Returns:
        Node: Output node that is 1.0 if the input node is false, and -1.0 if the input node is true.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit 23fee36. See docs/numerical_noise.md.
    """
    return compare(inp, thresh=0.0, true_level=-1.0, false_level=1.0)


def equals_vector(inp: Node, vector: torch.Tensor) -> Node:
    """Compare a node's value to a fixed key vector: +1 at a match, -1
    otherwise.

    A one-sided compare on a dot product — one sharpened hinge whose bend
    sits a ``1/speed`` margin below the match::

        m      = vector·inp − vector·vector       # 0 at match, ≤ −1/speed at non-match
        result = 2·speed · Swish(scale·(m + 1/speed))/scale − 1

    ``speed`` is ``embedding_step_sharpness``: the margin, in dot-product
    units, that distinct keys must clear — sized to absorb embedding
    noise, unchanged from the ReLU form (self-normalizing hinge, so the
    module ``scale`` applies and match sensitivity stays ``2·speed`` per
    dot unit).  Matches are bit-exact +1 in fp32 (hinge argument
    ``scale/speed``, fully saturated); a non-match exactly at the margin
    is exact -1; non-matches within ``~17/scale`` past the margin land in
    the hinge's dip and read as low as ``-1 - 2·swish_dip·speed/scale``
    (-1.0056 at scale=100) — the value-range assert carries that low-side
    slack.  The top stays unclamped, as today: ``m > 0`` (a vector
    out-dotting the key) exceeds +1 — contract-excluded, caught by the
    assert.

    Args:
        inp (Node): The node to be compared.
        vector (torch.Tensor): The vector tensor for comparison.

    Returns:
        Node: Node with the result of the comparison.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit 23fee36. See docs/numerical_noise.md.
    """
    speed = embedding_step_sharpness
    gate_proj = scale * vector.unsqueeze(0)
    gate_bias = scale * (1.0 / speed - vector @ vector).reshape(1)
    output_proj = torch.tensor([[2.0 * speed / scale]])
    output_bias = torch.tensor([-1.0])
    result = swiglu_ffn(
        inp,
        gate_proj,
        gate_bias,
        output_proj,
        output_bias,
        name="equals_vector",
    )
    low_slack = 2.0 * swish_dip * speed / scale
    return assert_matches_value_type(
        result, NodeValueType(value_range=Range(-1.0 - low_slack, 1.0))
    )


def _cond_gate_output_type(inp: Node) -> NodeValueType:
    r = inp.value_type.value_range
    if not r.is_finite():
        return NodeValueType()
    return NodeValueType(value_range=Range(min(0.0, r.lo), max(0.0, r.hi)))


def cond_gate(cond: Node, inp: Node) -> Node:
    """Output ``inp`` if ``cond`` is true, else 0.

    One gated lane per component::

        cond_gate(cond, inp) = Swish(scale·cond) · inp / scale

    ``Swish(scale·cond)/scale ≈ ReLU(cond)``: 1 at ``cond=+1``, 0 at
    ``cond=−1`` — ``select`` with the false branch pinned to zero.
    ``cond`` is ±1, enforced by an assert (same ``e^{−scale}``-class gate
    error as select: at clean conds the off-path is exactly zero in fp32
    and the on-path passes with ~1 ulp relative rounding).  Because the
    multiply is direct, cond noise lands proportionally to the *actual*
    value of ``inp``, not a range maximum — the ReLU-era offset
    apparatus (``M``, the finite-range requirement, the ``c_tol·M``
    widening) does not exist here; ``inp`` may have an unbounded range.

    Args:
        cond (Node): Condition node.
        inp (Node): The node whose value is to be gated.

    Returns:
        Node: Output node after applying the gate based on condition.

    .. noise-footer::

       Max error: 4.768e-07 abs, 1.188e-07 rel over 4096 samples;
       measured at commit 23fee36. See docs/numerical_noise.md.
    """
    assert len(cond) == 1
    d = len(inp)

    cond = _assert_cond_pm1(cond)

    # d gated lanes: every gate row reads scale·cond, up row j picks
    # component j of inp, /scale folds into out_proj.
    gate_proj = torch.zeros(d, 1 + d)
    gate_proj[:, 0] = scale
    up_proj = torch.zeros(d, 1 + d)
    up_proj[:, 1:] = torch.eye(d)
    output_proj = torch.eye(d) / scale

    x = Concatenate([cond, inp])
    result = swiglu_ffn(
        x,
        gate_proj,
        torch.zeros(d),
        output_proj,
        torch.zeros(d),
        up_proj=up_proj,
        up_bias=torch.zeros(d),
        name="cond_gate",
    )

    vt = _cond_gate_output_type(inp)
    if vt != NodeValueType.unknown():
        r = inp.value_type.value_range
        gate_atol = _GATE_C_TOL * max(abs(r.lo), abs(r.hi))
        result = assert_matches_value_type(result, vt, atol=gate_atol)
    from torchwright.graph.affine_rules import (
        _apply_semantic_override,
        _cond_gate_semantic_bound,
    )

    _apply_semantic_override(
        result,
        _cond_gate_semantic_bound(inp._affine_bound, inp, rel_tol=_GATE_C_TOL),
    )
    return result
