"""Eager affine bound computation rules for each node type.

Called from ``Node.__init__`` to compute ``_affine_bound`` at
graph-construction time.  Each rule reads the already-set
``_affine_bound`` of its inputs (which are constructed first in
a dataflow graph) and produces the output bound.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch

from torchwright.graph.affine_bound import AffineBound

if TYPE_CHECKING:
    from torchwright.graph.node import Node


def _safe_matvec(W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``W @ b`` treating ``0 * ±inf`` as 0."""
    product = W * b.unsqueeze(0)
    product = torch.where(W == 0, torch.zeros_like(product), product)
    return product.sum(dim=1)


def compute_affine_bound(node: "Node") -> AffineBound:
    """Dispatch to the appropriate affine rule for *node*."""
    from torchwright.graph.misc import (
        InputNode,
        LiteralValue,
        Add,
        Concatenate,
        Assert,
        DebugWatch,
        ValueLogger,
    )
    from torchwright.graph.linear import Linear
    from torchwright.graph.relu import ReLU
    from torchwright.graph.block import Block
    from torchwright.graph.embedding import Embedding
    from torchwright.graph.attn import Attn

    if isinstance(node, InputNode):
        ab = AffineBound.identity(node)
        assert ab.n_cols > 0, "InputNode must produce non-degenerate affine bound"
        return ab

    if isinstance(node, LiteralValue):
        import torch

        return AffineBound.constant(node.value.to(dtype=torch.float64))

    if isinstance(node, (ValueLogger, DebugWatch)):
        return node.inputs[0]._affine_bound

    if isinstance(node, Linear):
        return _linear_rule(node)

    if isinstance(node, Add):
        return _add_rule(node)

    if isinstance(node, Concatenate):
        return _concat_rule(node)

    if isinstance(node, ReLU):
        return _relu_rule(node)

    if isinstance(node, Block):
        return _block_rule(node)

    if isinstance(node, Assert):
        return _assert_rule(node)

    if isinstance(node, Attn):
        return _attn_rule(node)

    if isinstance(node, Embedding):
        ab = _embedding_rule(node)
        assert ab.n_cols > 0, "Embedding must produce non-degenerate affine bound"
        return ab

    from torchwright.graph.misc import Placeholder

    if not isinstance(node, Placeholder):
        raise NotImplementedError(
            f"No affine rule for {type(node).__name__}. Add one to "
            f"affine_rules.py or register it as an explicit pass-through."
        )
    return AffineBound.degenerate(node.d_output)


def _linear_affine(
    inp_ab: AffineBound, W: torch.Tensor, c: torch.Tensor
) -> AffineBound:
    """y = x @ W + c through a bound: sign-split GEMM.

    ``W`` is ``(d_input, d_output)`` and ``c`` is ``(d_output,)``, both
    float64.  Pure function of the input bound and the affine params — shared
    by :func:`_linear_rule` (the ``Linear`` node) and :func:`_block_rule` (the
    gate/out projections of a ``Block``)."""
    W_plus = torch.clamp(W, min=0)
    W_minus = torch.clamp(W, max=0)

    A_lo = W_plus.T @ inp_ab.A_lo + W_minus.T @ inp_ab.A_hi
    b_lo = (
        _safe_matvec(W_plus.T, inp_ab.b_lo) + _safe_matvec(W_minus.T, inp_ab.b_hi) + c
    )
    A_hi = W_plus.T @ inp_ab.A_hi + W_minus.T @ inp_ab.A_lo
    b_hi = (
        _safe_matvec(W_plus.T, inp_ab.b_hi) + _safe_matvec(W_minus.T, inp_ab.b_lo) + c
    )

    return AffineBound(
        A_lo=A_lo,
        A_hi=A_hi,
        b_lo=b_lo,
        b_hi=b_hi,
        columns=inp_ab.columns,
        input_ranges=inp_ab.input_ranges,
    )


def _linear_rule(node) -> AffineBound:
    """y = x @ W + c: sign-split GEMM."""
    inp_ab = node.inputs[0]._affine_bound
    W = node.output_matrix.to(torch.float64)
    c = node.output_bias.to(torch.float64)
    return _linear_affine(inp_ab, W, c)


def _add_rule(node) -> AffineBound:
    u = node.inputs[0]._affine_bound
    v = node.inputs[1]._affine_bound
    if u.d_output != v.d_output:
        raise ValueError(
            f"Add node {node.node_id}: input affine bounds have mismatched "
            f"d_output ({u.d_output} vs {v.d_output})"
        )
    u, v = AffineBound.align(u, v)
    return AffineBound(
        A_lo=u.A_lo + v.A_lo,
        A_hi=u.A_hi + v.A_hi,
        b_lo=u.b_lo + v.b_lo,
        b_hi=u.b_hi + v.b_hi,
        columns=u.columns,
        input_ranges=u.input_ranges,
    )


def _concat_rule(node) -> AffineBound:
    import torch

    from torchwright.graph.affine_bound import _merge_layouts, _scatter

    bounds = [inp._affine_bound for inp in node.inputs]

    if len(bounds) == 1:
        return bounds[0]

    merged_columns, merged_ranges, n = _merge_layouts(*bounds)

    parts_lo, parts_hi, parts_blo, parts_bhi = [], [], [], []
    for b in bounds:
        scattered = _scatter(b, merged_columns, merged_ranges, n)
        parts_lo.append(scattered.A_lo)
        parts_hi.append(scattered.A_hi)
        parts_blo.append(scattered.b_lo)
        parts_bhi.append(scattered.b_hi)

    return AffineBound(
        A_lo=torch.cat(parts_lo, dim=0),
        A_hi=torch.cat(parts_hi, dim=0),
        b_lo=torch.cat(parts_blo, dim=0),
        b_hi=torch.cat(parts_bhi, dim=0),
        columns=merged_columns,
        input_ranges=merged_ranges,
    )


def _relu_affine(
    inp_ab: AffineBound, warn_node: Optional["Node"] = None
) -> AffineBound:
    """ReLU per-component case analysis using the linear envelope.

    Pure function of the input bound.  ``warn_node`` (optional) is only used
    to emit the "infinite affine interval but finite value_type" warning
    against a real node; pass ``None`` when there is no single node behind the
    bound (e.g. a ``Block``'s internal gate values)."""
    intervals = inp_ab.to_interval()
    d = inp_ab.d_output
    n = inp_ab.n_cols

    A_lo = torch.zeros(d, n, dtype=torch.float64)
    A_hi = torch.zeros(d, n, dtype=torch.float64)
    b_lo = torch.zeros(d, dtype=torch.float64)
    b_hi = torch.zeros(d, dtype=torch.float64)

    for i in range(d):
        l, h = intervals[i].lo, intervals[i].hi
        if l >= 0:
            A_lo[i] = inp_ab.A_lo[i]
            A_hi[i] = inp_ab.A_hi[i]
            b_lo[i] = inp_ab.b_lo[i]
            b_hi[i] = inp_ab.b_hi[i]
        elif h <= 0:
            pass
        elif math.isinf(l) or math.isinf(h):
            if warn_node is not None:
                vt_range = warn_node.value_type.value_range
                if math.isfinite(vt_range.lo) and math.isfinite(vt_range.hi):
                    import warnings

                    warnings.warn(
                        f"Node {warn_node.node_id}: ReLU input affine interval is "
                        f"infinite ({l}, {h}) but value_type is finite "
                        f"{vt_range}. Upstream affine rule may be missing or "
                        f"degenerate."
                    )
            b_hi[i] = float(h)
        else:
            slope = h / (h - l)
            A_hi[i] = slope * inp_ab.A_hi[i]
            b_hi[i] = slope * (inp_ab.b_hi[i] - l)
            alpha = h / (h - l)
            A_lo[i] = alpha * inp_ab.A_lo[i]
            b_lo[i] = alpha * inp_ab.b_lo[i]

    return AffineBound(
        A_lo=A_lo,
        A_hi=A_hi,
        b_lo=b_lo,
        b_hi=b_hi,
        columns=inp_ab.columns,
        input_ranges=inp_ab.input_ranges,
    )


def _relu_rule(node) -> AffineBound:
    """ReLU per-component case analysis using linear envelope."""
    return _relu_affine(node.inputs[0]._affine_bound, warn_node=node.inputs[0])


def _block_rule(node) -> AffineBound:
    """Block: compose gate projection -> ReLU envelope -> output projection.

    The three sub-steps are exactly the rules the mined ``Linear -> ReLU ->
    Linear`` chain applies across its three nodes today, run here over one
    node's projections.  Step 1 is ReLU-only; a swish envelope is step 2's
    problem, so a gated/swish block is rejected here (the compiler path is
    ReLU-degenerate this phase anyway)."""
    if node.activation != "relu" or node.up_proj is not None:
        raise NotImplementedError(
            "Block affine rule supports only degenerate ReLU lanes in step 1 "
            f"(got activation={node.activation!r}, "
            f"gated={node.up_proj is not None}); a swish/gated envelope is "
            "step 2's affine rule."
        )
    inp_ab = node.inputs[0]._affine_bound
    # Gate projection: y = x @ gate_proj.T + gate_bias.  gate_proj is
    # (n_lanes, d_input) in math orientation, so W = gate_proj.T.
    gate_W = node.gate_proj.to(torch.float64).T
    gate_c = node.gate_bias.to(torch.float64)
    gate_ab = _linear_affine(inp_ab, gate_W, gate_c)
    # ReLU envelope per lane.
    lane_ab = _relu_affine(gate_ab, warn_node=None)
    # Output projection: out = lane @ out_proj + out_bias.  out_proj is already
    # (n_lanes, d_output).
    out_W = node.out_proj.to(torch.float64)
    out_c = node.out_bias.to(torch.float64)
    return _linear_affine(lane_ab, out_W, out_c)


def _assert_rule(node) -> AffineBound:
    """Assert: pass through coefficients, optionally tighten input_ranges."""
    from torchwright.graph.misc import Assert, InputNode
    from torchwright.graph.embedding import Embedding

    inp_ab = node.inputs[0]._affine_bound
    if node.claimed_type is not None:
        claimed_range = node.claimed_type.value_range

        target = node.inputs[0]
        while isinstance(target, Assert):
            target = target.inputs[0]

        if isinstance(target, (InputNode, Embedding)):
            new_ranges = dict(inp_ab.input_ranges)
            if target.node_id in new_ranges:
                old_lo, old_hi = new_ranges[target.node_id]
                new_lo = torch.maximum(
                    old_lo, torch.full_like(old_lo, claimed_range.lo)
                )
                new_hi = torch.minimum(
                    old_hi, torch.full_like(old_hi, claimed_range.hi)
                )
                new_ranges[target.node_id] = (new_lo, new_hi)
                return AffineBound(
                    A_lo=inp_ab.A_lo,
                    A_hi=inp_ab.A_hi,
                    b_lo=inp_ab.b_lo,
                    b_hi=inp_ab.b_hi,
                    columns=inp_ab.columns,
                    input_ranges=new_ranges,
                )

        if claimed_range.is_finite():
            intervals = inp_ab.to_interval()
            d = node.d_output
            b_lo = torch.tensor(
                [max(iv.lo, claimed_range.lo) for iv in intervals],
                dtype=torch.float64,
            )
            b_hi = torch.tensor(
                [min(iv.hi, claimed_range.hi) for iv in intervals],
                dtype=torch.float64,
            )
            return AffineBound(
                A_lo=torch.zeros(d, 0, dtype=torch.float64),
                A_hi=torch.zeros(d, 0, dtype=torch.float64),
                b_lo=b_lo,
                b_hi=b_hi,
                columns={},
                input_ranges={},
            )
    return inp_ab


def _attn_rule(node) -> AffineBound:
    """Attn: propagate value bounds through V then O.

    Softmax produces convex-combination weights, so per-component
    affine bounds on ``value @ V`` carry through unchanged.  O is
    then a standard linear step.
    """
    value_ab = node.inputs[2]._affine_bound
    V = node.value_matrix.to(torch.float64)
    O = node.output_matrix.to(torch.float64)

    V_plus = torch.clamp(V, min=0)
    V_minus = torch.clamp(V, max=0)
    proj_A_lo = V_plus.T @ value_ab.A_lo + V_minus.T @ value_ab.A_hi
    proj_b_lo = _safe_matvec(V_plus.T, value_ab.b_lo) + _safe_matvec(
        V_minus.T, value_ab.b_hi
    )
    proj_A_hi = V_plus.T @ value_ab.A_hi + V_minus.T @ value_ab.A_lo
    proj_b_hi = _safe_matvec(V_plus.T, value_ab.b_hi) + _safe_matvec(
        V_minus.T, value_ab.b_lo
    )

    O_plus = torch.clamp(O, min=0)
    O_minus = torch.clamp(O, max=0)
    A_lo = O_plus.T @ proj_A_lo + O_minus.T @ proj_A_hi
    b_lo = _safe_matvec(O_plus.T, proj_b_lo) + _safe_matvec(O_minus.T, proj_b_hi)
    A_hi = O_plus.T @ proj_A_hi + O_minus.T @ proj_A_lo
    b_hi = _safe_matvec(O_plus.T, proj_b_hi) + _safe_matvec(O_minus.T, proj_b_lo)

    return AffineBound(
        A_lo=A_lo,
        A_hi=A_hi,
        b_lo=b_lo,
        b_hi=b_hi,
        columns=value_ab.columns,
        input_ranges=value_ab.input_ranges,
    )


def _embedding_rule(node) -> AffineBound:
    """Identity A-matrix with per-column min/max ranges from the embedding table."""
    import torch

    t = node.table.to(torch.float64)
    d = node.d_output
    assert t.numel() > 0
    A = torch.eye(d, dtype=torch.float64)
    b = torch.zeros(d, dtype=torch.float64)
    col_lo = t.min(dim=0).values
    col_hi = t.max(dim=0).values
    return AffineBound(
        A_lo=A.clone(),
        A_hi=A.clone(),
        b_lo=b.clone(),
        b_hi=b.clone(),
        columns={node.node_id: (0, d)},
        input_ranges={node.node_id: (col_lo, col_hi)},
    )


# --- Semantic overrides for composite ops ----------------------------------


def _apply_semantic_override(node: "Node", semantic_ab: Optional[AffineBound]) -> None:
    """Replace *node*'s affine bound with a semantic override."""
    if semantic_ab is None:
        return
    propagated = node._affine_bound.to_scalar_range()
    semantic = semantic_ab.to_scalar_range()
    assert semantic.lo <= propagated.hi and semantic.hi >= propagated.lo, (
        f"Semantic override on node {node.node_id} is disjoint from "
        f"propagated bound: semantic={semantic}, propagated={propagated}"
    )
    node._affine_bound = semantic_ab


def _cond_gate_semantic_bound(
    inp_ab: AffineBound,
    inp_node: Optional["Node"] = None,
    c_tol: float = 0.0,
    M: float = 0.0,
) -> AffineBound:
    """Per-component [min(0, inp), max(0, inp)] envelope for cond_gate.

    When ``c_tol > 0`` and ``M > 0``, widens the envelope by
    ``c_tol * M`` per component to account for amplified condition noise
    in the approximate gate path.
    """
    intervals = inp_ab.to_interval()
    d = inp_ab.d_output
    n = inp_ab.n_cols

    A_lo = torch.zeros(d, n, dtype=torch.float64)
    A_hi = torch.zeros(d, n, dtype=torch.float64)
    b_lo = torch.zeros(d, dtype=torch.float64)
    b_hi = torch.zeros(d, dtype=torch.float64)

    for i in range(d):
        l, h = intervals[i].lo, intervals[i].hi
        if l >= 0:
            A_hi[i] = inp_ab.A_hi[i]
            b_hi[i] = inp_ab.b_hi[i]
        elif h <= 0:
            A_lo[i] = inp_ab.A_lo[i]
            b_lo[i] = inp_ab.b_lo[i]
        elif math.isinf(l) or math.isinf(h):
            if inp_node is not None:
                vt_range = inp_node.value_type.value_range
                if math.isfinite(vt_range.lo) and math.isfinite(vt_range.hi):
                    import warnings

                    warnings.warn(
                        f"Node {inp_node.node_id}: cond_gate input affine "
                        f"interval is infinite ({l}, {h}) but value_type is "
                        f"finite {vt_range}. Upstream affine rule may be "
                        f"missing or degenerate."
                    )
            b_lo[i] = float(min(0, l))
            b_hi[i] = float(max(0, h))
        else:
            s_hi = h / (h - l)
            A_hi[i] = s_hi * inp_ab.A_hi[i]
            b_hi[i] = s_hi * (inp_ab.b_hi[i] - l)
            s_lo = -l / (h - l)
            A_lo[i] = s_lo * inp_ab.A_lo[i]
            b_lo[i] = s_lo * inp_ab.b_lo[i] + l * h / (h - l)

    widen = c_tol * M
    b_lo -= widen
    b_hi += widen

    return AffineBound(
        A_lo=A_lo,
        A_hi=A_hi,
        b_lo=b_lo,
        b_hi=b_hi,
        columns=inp_ab.columns,
        input_ranges=inp_ab.input_ranges,
    )


def _select_semantic_bound(
    a_ab: AffineBound,
    b_ab: AffineBound,
    tolerance: float = 0.0,
) -> AffineBound:
    """Per-component hull of a and b intervals for select.

    When ``tolerance > 0``, widens the hull by that amount per component
    to account for amplified condition noise in the approximate gate path.
    """
    import torch

    from torchwright.graph.affine_bound import _merge_layouts, _scatter

    a_ab, b_ab = AffineBound.align(a_ab, b_ab)
    a_intervals = a_ab.to_interval()
    b_intervals = b_ab.to_interval()
    d = a_ab.d_output

    b_lo = torch.zeros(d, dtype=torch.float64)
    b_hi = torch.zeros(d, dtype=torch.float64)
    for i in range(d):
        b_lo[i] = min(a_intervals[i].lo, b_intervals[i].lo)
        b_hi[i] = max(a_intervals[i].hi, b_intervals[i].hi)

    b_lo -= tolerance
    b_hi += tolerance

    return AffineBound(
        A_lo=torch.zeros(d, a_ab.n_cols, dtype=torch.float64),
        A_hi=torch.zeros(d, a_ab.n_cols, dtype=torch.float64),
        b_lo=b_lo,
        b_hi=b_hi,
        columns=a_ab.columns,
        input_ranges=a_ab.input_ranges,
    )


def _broadcast_select_semantic_bound(
    true_ab: AffineBound,
    false_ab: AffineBound,
    n_slots: int,
    d_fill: int,
    true_is_broadcast: bool,
    false_is_broadcast: bool,
    tolerance: float = 0.0,
) -> AffineBound:
    """Per-component hull across slots for broadcast_select.

    Each output slot picks true or false, so the per-channel bound is
    the hull of the corresponding true/false channel. When a branch is
    broadcast, its single-slot interval applies to every slot.
    """
    import torch

    true_intervals = true_ab.to_interval()
    false_intervals = false_ab.to_interval()
    d = n_slots * d_fill

    b_lo = torch.zeros(d, dtype=torch.float64)
    b_hi = torch.zeros(d, dtype=torch.float64)
    for i in range(n_slots):
        for j in range(d_fill):
            out_idx = i * d_fill + j
            t_idx = j if true_is_broadcast else out_idx
            f_idx = j if false_is_broadcast else out_idx
            t_iv = (
                true_intervals[t_idx]
                if t_idx < len(true_intervals)
                else true_intervals[j]
            )
            f_iv = (
                false_intervals[f_idx]
                if f_idx < len(false_intervals)
                else false_intervals[j]
            )
            b_lo[out_idx] = min(t_iv.lo, f_iv.lo)
            b_hi[out_idx] = max(t_iv.hi, f_iv.hi)

    b_lo -= tolerance
    b_hi += tolerance

    return AffineBound(
        A_lo=torch.zeros(d, 0, dtype=torch.float64),
        A_hi=torch.zeros(d, 0, dtype=torch.float64),
        b_lo=b_lo,
        b_hi=b_hi,
        columns={},
        input_ranges={},
    )


def _compare_semantic_bound(
    inp_ab: AffineBound,
    thresh: float,
    true_level: float,
    false_level: float,
) -> AffineBound:
    """Constant or degenerate bound for compare, depending on inp vs thresh."""
    import torch

    intervals = inp_ab.to_interval()
    assert len(intervals) == 1
    l, h = intervals[0].lo, intervals[0].hi

    lo = min(true_level, false_level)
    hi = max(true_level, false_level)

    if l > thresh:
        return AffineBound.constant(torch.tensor([true_level], dtype=torch.float64))
    if h <= thresh:
        return AffineBound.constant(torch.tensor([false_level], dtype=torch.float64))
    return AffineBound.degenerate(1, lo=lo, hi=hi)
