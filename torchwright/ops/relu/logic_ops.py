import math

import torch

from torchwright.graph import Concatenate, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import embedding_step_sharpness
from torchwright.ops.linear import sum_nodes
from torchwright.ops.relu.arithmetic_ops import compare
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

_GATE_OFFSET_SAFETY_FACTOR = 2.0
"""Headroom over the declared ``max|value|`` so that activation noise
from the compiled transformer's ReLU approximations doesn't leak
through the gate's off-path. The old global ``big_offset = 1000`` gave
~100× headroom over typical values; a 2× factor keeps the precision
win (M ≈ 2·max_abs rather than 1000) while tolerating modest drift."""


_MAX_REASONABLE_OFFSET = 1e6

_COND_GATE_C_TOL = 0.005
"""Maximum acceptable deviation of ``|cond|`` from 1 in ``cond_gate``. An
Assert checks ``||cond| - 1| <= _COND_GATE_C_TOL`` and the output's semantic
bound is widened by ``M * _COND_GATE_C_TOL`` to account for amplified condition
noise. 0.005 matches typical far-field compare noise."""


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


def per_column_offsets(intervals, scalar_M: float) -> "torch.Tensor":
    """Per-output-column gate offsets ``M_j = safety * max|range_j|``.

    The additive-cancellation gate ``(M + v) - M`` rounds ``v`` to
    ``ULP(M)``; a single scalar ``M`` is forced to the *widest* column, so
    a narrow column bundled with a wide sibling (e.g. the lifted-id key
    ``[child, -child^2, 1]``, where ``child`` needs ~``2q``× finer precision
    than ``-child^2``) is rounded far more coarsely than it needs to be.
    Sizing ``M`` per column from the per-column affine interval keeps each
    column's rounding at its own scale -- the Plan-K Step-1 edge-key fix.
    See ``/data/torchdoom/k_step1_divergence_characterization.md``.

    ``intervals`` is a per-component ``Range`` list (already intersected with
    the scalar value_type by the caller); ``scalar_M`` is the fallback used
    for any non-finite component. Every returned ``M_j <= scalar_M`` (the
    per-column max never exceeds the union), so the gate stays sound.
    """
    out = torch.empty(len(intervals))
    fallback = scalar_M / _GATE_OFFSET_SAFETY_FACTOR
    for j, r in enumerate(intervals):
        m = max(abs(r.lo), abs(r.hi))
        if not math.isfinite(m):
            m = fallback
        out[j] = _GATE_OFFSET_SAFETY_FACTOR * m
    return out


def _intersect_intervals(node: "Node"):
    """Per-component intervals of ``node`` intersected with its scalar
    value_type, or ``None`` if a per-column affine bound is unavailable /
    width-mismatched (caller falls back to the scalar offset).
    """
    try:
        intervals = node.affine_bound.to_interval()
    except Exception:
        return None
    if intervals is None or len(intervals) != len(node):
        return None
    vt = node.value_type.value_range
    from torchwright.graph.value_type import Range as _Range

    return [
        _Range(max(r.lo, vt.lo), min(r.hi, vt.hi)) if vt.is_finite() else r
        for r in intervals
    ]


def bool_any_true(inp_list: list[Node]) -> Node:
    """Returns a node that evaluates to True if any of the input nodes are true.

    Args:
        inp_list (List[Node]): List of nodes to be evaluated.

    Returns:
        Node: Output node that is True if any input nodes are true, otherwise False.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    # Strategy:
    # Convert all the values to 1.0 if they're > 0.0 and 0.0 otherwise
    # then sum them, and if the sum is > 0.5, return 1.0, otherwise -1.0
    sum_node = sum_nodes(
        [compare(n, thresh=0.0, true_level=1.0, false_level=0.0) for n in inp_list]
    )
    return compare(sum_node, thresh=0.5, true_level=1.0, false_level=-1.0)


def bool_all_true(inp_list: list[Node]) -> Node:
    """Returns a node that evaluates to True if all of the input nodes are true.

    Inputs must be clean ±1.0 booleans (as produced by compare/bool_* ops).
    Sum of N such inputs is +N only when all are +1; otherwise ≤ N-2.
    A threshold at N-1 cleanly separates the two cases.

    Args:
        inp_list (List[Node]): List of nodes to be evaluated.

    Returns:
        Node: Output node that is True if all input nodes are true, otherwise False.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    return compare(
        sum_nodes(inp_list),
        thresh=len(inp_list) - 1.0,
        true_level=1.0,
        false_level=-1.0,
    )


def bool_not(inp: Node) -> Node:
    """Returns a node that evaluates to 1.0 if the input node is false, and -1.0 if the input node is true.

    Args:
        inp: Input node to be evaluated

    Returns:
        Node: Output node that is 1.0 if the input node is false, and -1.0 if the input node is true.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    return compare(inp, thresh=0.0, true_level=-1.0, false_level=1.0)


def equals_vector(inp: Node, vector: torch.Tensor) -> Node:
    """Compares a node's value to a vector tensor.

    Args:
        inp (Node): The node to be compared.
        vector (torch.Tensor): The vector tensor for comparison.

    Returns:
        Node: Node with the result of the comparison.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
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


def _cond_gate_output_type(cond: Node, inp: Node) -> NodeValueType:
    vt = inp.value_type
    r = vt.value_range
    if not r.is_finite():
        return NodeValueType()
    return NodeValueType(value_range=Range(min(0.0, r.lo), max(0.0, r.hi)))


def cond_gate(cond: Node, inp: Node) -> Node:
    """Gates the value of a node based on a condition. If the condition is true,
    outputs the value. If false, outputs a zero tensor of the same shape as value.

    Uses a single L→ReLU→L sublayer with an additive cancellation trick: the
    on-path computes ``(M + v) − M`` where ``M`` is derived from
    ``inp.value_type``. This loses precision for ``|v| ≪ ULP(M)`` and amplifies
    approximate-cond error as ``M·ε``. An Assert checks ``||cond| - 1|`` stays
    within ``_COND_GATE_C_TOL`` and the output's semantic bound is widened by
    ``M * _COND_GATE_C_TOL`` accordingly.

    Args:
        cond (Node): Condition node.
        inp (Node): The node whose value is to be gated.

    Returns:
        Node: Output node after applying the gate based on condition.

    .. noise-footer::

       Max error: 0.0009766 abs, 0.3885 rel over 4096 samples;
       measured at commit a39e4c6. See docs/numerical_noise.md.
    """
    assert len(cond) == 1
    d = len(inp)
    M = _max_abs_or_raise(inp.value_type, "cond_gate")

    from torchwright.graph.asserts import attach_assert

    def _cond_check(x: torch.Tensor) -> tuple:
        deviation = (x.abs() - 1.0).abs()
        bad = deviation > _COND_GATE_C_TOL
        if not bad.any():
            return True, ""
        from torchwright.graph.asserts import _format_bad

        return False, (
            f"expected ||cond| - 1| <= {_COND_GATE_C_TOL}; {_format_bad(x, bad)}"
        )

    cond = attach_assert(
        cond,
        _cond_check,
        message=f"cond near ±1 (c_tol={_COND_GATE_C_TOL})",
        claimed_type=NodeValueType(
            value_range=Range(-1.0 - _COND_GATE_C_TOL, 1.0 + _COND_GATE_C_TOL)
        ),
    )
    # Per-column offsets: the gate bakes M into `(M + v) - M`, so size M
    # per column from the per-column affine interval (a narrow column is
    # not forced to a wide sibling's M). Falls back to the scalar M.
    intervals = _intersect_intervals(inp)
    M_cols = (
        per_column_offsets(intervals, M)
        if intervals is not None
        else torch.full((d,), M)
    )

    d_hidden = 2 * d
    input_proj = torch.zeros(d_hidden, 1 + d)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, d)
    output_bias = -M_cols.clone()

    for j in range(d):
        a = j
        b = d + j
        input_proj[a, 0] = M_cols[j]
        input_proj[a, 1 + j] = 1.0
        input_proj[b, 0] = -M_cols[j]
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

    vt = _cond_gate_output_type(cond, inp)
    if vt != NodeValueType.unknown():
        gate_atol = M * _COND_GATE_C_TOL
        result = assert_matches_value_type(result, vt, atol=gate_atol)
    from torchwright.graph.affine_rules import (
        _apply_semantic_override,
        _cond_gate_semantic_bound,
    )

    _apply_semantic_override(
        result,
        _cond_gate_semantic_bound(inp._affine_bound, inp, c_tol=_COND_GATE_C_TOL, M=M),
    )
    return result
