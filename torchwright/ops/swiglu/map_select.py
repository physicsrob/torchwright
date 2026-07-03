"""Selection ops on the swish machine.

``select`` is the two-branch complementary gate pair per its entry in
``docs/ops_plain_english.md``; ``switch`` composes ``cond_gate``.  The
ReLU-era offset apparatus (``per_column_offsets``, ``scalar_M``, the
finite-range requirement) does not exist on this machine — cond noise
lands proportionally to the actual branch values.
"""

from typing import List

import torch

from torchwright.graph import Concatenate, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType
from torchwright.ops.const import scale

# sum_nodes is purely linear (Add hardware, no activation) — machine-neutral
# in substance, shared with the frozen relu package until its retirement
# relocates the linear ops.
from torchwright.ops.relu.arithmetic_ops import sum_nodes
from torchwright.ops.swiglu.logic_ops import _GATE_C_TOL, _assert_cond_pm1, cond_gate
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn


def _select_output_type(true_node: Node, false_node: Node) -> NodeValueType:
    r = true_node.value_type.value_range.union(false_node.value_type.value_range)
    if not r.is_finite():
        return NodeValueType()
    return NodeValueType(value_range=r)


def select(cond: Node, true_node: Node, false_node: Node) -> Node:
    """Output one of two nodes based on a boolean condition.

    A complementary pair of gated lanes per component::

        select(cond, a, b) = Swish(scale·cond)·a/scale + Swish(−scale·cond)·b/scale

    ``Swish(scale·cond)/scale ≈ ReLU(cond)`` — 1 at ``cond=+1``, 0 at
    ``cond=−1`` — so the two gates are complementary on/off indicators:
    one passes ``a``, the other ``b``.  ``cond`` is ±1, enforced by an
    assert.  At clean conds in fp32 the losing branch contributes
    exactly zero (``σ(−scale)`` computes as 0.0) and the winner passes
    with at most ~1 ulp relative rounding.  A cond off ±1 by δ
    mis-scales the winner by exactly δ·|actual value| — there is no
    offset ``M``, no finite-range requirement on the branches, and the
    semantic bound widens relatively, not by ``c_tol·M``.

    Args:
        cond (Node): Condition node that outputs either true or false.
        true_node (Node): Node to be outputted if the condition is true.
        false_node (Node): Node to be outputted if the condition is false.

    Returns:
        Node: Either true_node or false_node based on the condition.

    .. noise-footer::

       Max error: 4.768e-07 abs, 1.19e-07 rel over 4096 samples;
       measured at commit 23fee36. See docs/numerical_noise.md.
    """
    assert len(cond) == 1
    assert len(true_node) == len(false_node)

    d = len(true_node)
    cond = _assert_cond_pm1(cond)

    # 2·d gated lanes: d with gate row +scale·cond and up rows picking
    # true_node's components, d mirrored with −scale·cond picking
    # false_node's; the /scale folds into out_proj.
    gate_proj = torch.zeros(2 * d, 1 + 2 * d)
    gate_proj[:d, 0] = scale
    gate_proj[d:, 0] = -scale
    up_proj = torch.zeros(2 * d, 1 + 2 * d)
    up_proj[:d, 1 : 1 + d] = torch.eye(d)
    up_proj[d:, 1 + d :] = torch.eye(d)
    output_proj = torch.cat([torch.eye(d), torch.eye(d)], dim=0) / scale

    x = Concatenate([cond, true_node, false_node])
    result = swiglu_ffn(
        x,
        gate_proj,
        torch.zeros(2 * d),
        output_proj,
        torch.zeros(d),
        up_proj=up_proj,
        up_bias=torch.zeros(2 * d),
        name="select",
    )

    vt = _select_output_type(true_node, false_node)
    if vt != NodeValueType.unknown():
        r = vt.value_range
        gate_atol = _GATE_C_TOL * max(abs(r.lo), abs(r.hi))
        result = assert_matches_value_type(result, vt, atol=gate_atol)
    from torchwright.graph.affine_rules import (
        _apply_semantic_override,
        _select_semantic_bound,
    )

    _apply_semantic_override(
        result,
        _select_semantic_bound(
            true_node._affine_bound,
            false_node._affine_bound,
            rel_tolerance=_GATE_C_TOL,
        ),
    )
    return result


def switch(conditions: List[Node], values: List[Node]) -> Node:
    """Select one of N values based on which condition is true.

    Assumes exactly one condition is true (1.0), rest are false (-1.0).
    A :func:`cond_gate` per branch summed on Add hardware — each losing
    branch contributes exactly zero at clean conds (see cond_gate), so
    the sum is the winning value.

    Args:
        conditions (List[Node]): Boolean condition nodes (each length 1).
        values (List[Node]): Value nodes (all same length).

    Returns:
        Node: The value whose corresponding condition is true.
    """
    return sum_nodes([cond_gate(c, v) for c, v in zip(conditions, values)])
