import math
from typing import List

from torchwright.graph import Node, Add, Concatenate
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.linear_relu_linear import linear_relu_linear

import torch

from torchwright.ops.arithmetic_ops import sum_nodes, compare
from torchwright.ops.const import (
    step_sharpness,
    embedding_step_sharpness,
)

_GATE_OFFSET_SAFETY_FACTOR = 2.0
"""Headroom over the declared ``max|value|`` so that activation noise
from the compiled transformer's ReLU approximations doesn't leak
through the gate's off-path. The old global ``big_offset = 1000`` gave
~100× headroom over typical values; a 2× factor keeps the precision
win (M ≈ 2·max_abs rather than 1000) while tolerating modest drift."""


_MAX_REASONABLE_OFFSET = 1e6


def _max_abs_or_raise(vt: NodeValueType, caller: str) -> float:
    r = vt.value_range
    m = max(abs(r.lo), abs(r.hi))
    if not math.isfinite(m):
        raise TypeError(
            f"{caller} requires a bounded value_range on its gated input; "
            f"got {vt}. Wrap the upstream node with "
            f"`assert_matches_value_type(node, NodeValueType(value_range=Range(lo, hi)))`."
        )
    M = _GATE_OFFSET_SAFETY_FACTOR * m
    assert M <= _MAX_REASONABLE_OFFSET, (
        f"{caller}: M offset {M:.2e} exceeds sanity bound {_MAX_REASONABLE_OFFSET:.0e}. "
        f"Input value_range={r} is likely a stale or un-clamped range — "
        f"check that upstream value_type propagation returns bounded ranges."
    )
    return M


def bool_any_true(inp_list: List[Node]) -> Node:
    """
    Returns a node that evaluates to True if any of the input nodes are true.

    Args:
        inp_list (List[Node]): List of nodes to be evaluated.

    Returns:
        Node: Output node that is True if any input nodes are true, otherwise False.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    # Strategy:
    # Convert all the values to 1.0 if they're > 0.0 and 0.0 otherwise
    # then sum them, and if the sum is > 0.5, return 1.0, otherwise -1.0
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
       measured at commit fe133f9. See docs/numerical_noise.md.
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
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    return compare(inp, thresh=0.0, true_level=-1.0, false_level=1.0)


def equals_vector(inp: Node, vector: torch.Tensor) -> Node:
    """
    Compares a node's value to a vector tensor.

    Args:
        inp (Node): The node to be compared.
        vector (torch.Tensor): The vector tensor for comparison.

    Returns:
        Node: Node with the result of the comparison.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    # If value1 == c, result is 1
    # else result is -1
    # We'll use an MLP:
    # y = 2.0*speed * max(1.0/speed + c @ value - c @ c, 0) - 1.0
    # d_hidden = 1
    speed = embedding_step_sharpness
    input_proj = vector.unsqueeze(0)  # We're dotting vector into value
    input_bias = 1.0 / speed - vector @ vector
    output_proj = torch.tensor([[2.0 * speed]])
    output_bias = torch.tensor([-1.0])
    result = linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )
    return assert_matches_value_type(
        result, NodeValueType(value_range=Range(-1.0, 1.0))
    )


def cond_add_vector(
    cond: Node, inp: Node, true_vector: torch.Tensor, false_vector: torch.Tensor
) -> Node:
    """
    Conditionally adds a vector to the input node based on the value of the condition node.

    If the value from the `cond` node is true, this function adds the `true_vector` to the `input_node`.
    If the value from the `cond` node is false, it adds the `false_vector` to the `input_node`.

    Args:
        cond (Node): A boolean input node that determines which vector gets added.
        inp (Node): The node whose values are to be modified based on the condition.
        true_vector (torch.Tensor): The vector to add if the condition is true.
        false_vector (torch.Tensor): The vector to add if the condition is false.

    Returns:
        Node: A new node with the modified values based on the condition and input vectors.
    """
    assert len(cond) == 1
    assert len(true_vector) == len(false_vector) == len(inp)

    # We need 2 MLP entries, we'll use the equation:
    # y= c * [max(step_sharpness*x, 0) - max(step_sharpness*x - 1, 0)]
    # And rely on the residual connection

    d_input = len(inp)

    input_proj = torch.tensor([[step_sharpness], [step_sharpness]])
    input_bias = torch.tensor([0.0, -1.0])
    output_proj = torch.zeros((2, d_input))
    output_bias = false_vector

    for d in range(d_input):
        output_proj[0, d] = true_vector[d] - false_vector[d]
        output_proj[1, d] = -(true_vector[d] - false_vector[d])

    return Add(
        inp,
        linear_relu_linear(
            input_node=cond,
            input_proj=input_proj,
            input_bias=input_bias,
            output_proj=output_proj,
            output_bias=output_bias,
        ),
    )


def _cond_gate_output_type(cond: Node, inp: Node) -> NodeValueType:
    vt = inp.value_type
    r = vt.value_range
    if not r.is_finite():
        return NodeValueType()
    return NodeValueType(value_range=Range(min(0.0, r.lo), max(0.0, r.hi)))


def cond_gate(
    cond: Node,
    inp: Node,
    *,
    approximate: bool = True,
    c_tol: float = 0.005,
) -> Node:
    """
    Gates the value of a node based on a condition. If the condition is true,
    outputs the value. If false, outputs a zero tensor of the same shape as value.

    Args:
        cond (Node): Condition node.
        inp (Node): The node whose value is to be gated.
        approximate: When ``True`` (default), uses a single L→ReLU→L sublayer
            with an additive cancellation trick. The on-path computes
            ``(M + v) − M`` where ``M`` is derived from ``inp.value_type``; this
            loses precision for ``|v| ≪ ULP(M)`` and amplifies approximate-cond
            error as ``M·ε``. When ``False``, uses two sublayers: the first
            maps ``cond`` to ``c_off = ReLU(−cond) ∈ {0, 1}`` (clipping cond
            noise on the on-side); the second gates ``inp`` via
            ``ReLU(±inp − M·c_off)``. The on-path is float-exact and immune to
            cond noise; costs one extra MLP sublayer.
        c_tol: Maximum acceptable deviation of ``|cond|`` from 1. When
            ``approximate=True``, an Assert checks ``||cond| - 1| <= c_tol``
            and the output's semantic bound is widened by ``M * c_tol`` to
            account for the amplified condition noise. Default 0.005
            matches typical far-field compare noise.

    Returns:
        Node: Output node after applying the gate based on condition.

    .. noise-footer::

       Max error: 0.0009766 abs, 0.3885 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(cond) == 1
    d = len(inp)
    M = _max_abs_or_raise(inp.value_type, "cond_gate")

    if approximate:
        from torchwright.graph.misc import Assert

        def _cond_check(x: torch.Tensor) -> tuple:
            deviation = (x.abs() - 1.0).abs()
            bad = deviation > c_tol
            if not bad.any():
                return True, ""
            from torchwright.graph.asserts import _format_bad

            return False, (f"expected ||cond| - 1| <= {c_tol}; {_format_bad(x, bad)}")

        cond = Assert(
            cond,
            _cond_check,
            message=f"cond near ±1 (c_tol={c_tol})",
            claimed_type=NodeValueType(value_range=Range(-1.0 - c_tol, 1.0 + c_tol)),
        )
        d_hidden = 2 * d
        input_proj = torch.zeros(d_hidden, 1 + d)
        input_bias = torch.zeros(d_hidden)
        output_proj = torch.zeros(d_hidden, d)
        output_bias = torch.full((d,), -M)

        for j in range(d):
            a = j
            b = d + j
            input_proj[a, 0] = M
            input_proj[a, 1 + j] = 1.0
            input_proj[b, 0] = -M
            output_proj[a, j] = 1.0
            output_proj[b, j] = 1.0

        x = Concatenate([cond, inp])
        result = linear_relu_linear(
            input_node=x,
            input_proj=input_proj,
            input_bias=input_bias,
            output_proj=output_proj,
            output_bias=output_bias,
            name="cond_gate",
        )
    else:
        c_off = linear_relu_linear(
            input_node=cond,
            input_proj=torch.tensor([[-1.0]]),
            input_bias=torch.tensor([0.0]),
            output_proj=torch.tensor([[1.0]]),
            output_bias=torch.tensor([0.0]),
            name="cond_gate_c_off",
        )
        d_hidden = 2 * d
        input_proj = torch.zeros(d_hidden, 1 + d)
        input_bias = torch.zeros(d_hidden)
        output_proj = torch.zeros(d_hidden, d)
        output_bias = torch.zeros(d)

        for j in range(d):
            a = j
            b = d + j
            input_proj[a, 0] = -M
            input_proj[a, 1 + j] = 1.0
            input_proj[b, 0] = -M
            input_proj[b, 1 + j] = -1.0
            output_proj[a, j] = 1.0
            output_proj[b, j] = -1.0

        x = Concatenate([c_off, inp])
        result = linear_relu_linear(
            input_node=x,
            input_proj=input_proj,
            input_bias=input_bias,
            output_proj=output_proj,
            output_bias=output_bias,
            name="cond_gate",
        )

    vt = _cond_gate_output_type(cond, inp)
    if vt != NodeValueType.unknown():
        gate_atol = M * c_tol if approximate else 1e-3
        result = assert_matches_value_type(result, vt, atol=gate_atol)
    from torchwright.graph.affine_rules import (
        _apply_semantic_override,
        _cond_gate_semantic_bound,
    )

    _apply_semantic_override(
        result,
        _cond_gate_semantic_bound(
            inp._affine_bound, inp, c_tol=c_tol if approximate else 0.0, M=M
        ),
    )
    return result
