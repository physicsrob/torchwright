"""Eager affine bound computation rules for each node type.

Called from ``Node.__init__`` to compute ``_affine_bound`` at
graph-construction time.  Each rule reads the already-set
``_affine_bound`` of its inputs (which are constructed first in
a dataflow graph) and produces the output bound.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from torchwright.graph.affine_bound import AffineBound

if TYPE_CHECKING:
    from torchwright.graph.attn import Attn
    from torchwright.graph.embedding import Embedding
    from torchwright.graph.ffn import FFN
    from torchwright.graph.linear import Linear
    from torchwright.graph.misc import Add, Concatenate
    from torchwright.graph.node import Node
    from torchwright.graph.relu import ReLU


def _safe_matvec(W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``W @ b`` treating ``0 * ±inf`` as 0."""
    product = W * b.unsqueeze(0)
    product = torch.where(W == 0, torch.zeros_like(product), product)
    return product.sum(dim=1)


def compute_affine_bound(node: Node) -> AffineBound:
    """Dispatch to the appropriate affine rule for *node*."""
    from torchwright.graph.attn import Attn
    from torchwright.graph.embedding import Embedding
    from torchwright.graph.ffn import FFN
    from torchwright.graph.linear import Linear
    from torchwright.graph.misc import (
        Add,
        Concatenate,
        InputNode,
        LiteralValue,
        ValueLogger,
    )
    from torchwright.graph.relu import ReLU

    if isinstance(node, InputNode):
        result = AffineBound.identity(node)
        assert result.n_cols > 0, "InputNode must produce non-degenerate affine bound"
    elif isinstance(node, LiteralValue):
        import torch

        result = AffineBound.constant(node.value.to(dtype=torch.float64))
    elif isinstance(node, ValueLogger):
        result = node.inputs[0]._affine_bound
    elif isinstance(node, Linear):
        result = _linear_rule(node)
    elif isinstance(node, Add):
        result = _add_rule(node)
    elif isinstance(node, Concatenate):
        result = _concat_rule(node)
    elif isinstance(node, ReLU):
        result = _relu_rule(node)
    elif isinstance(node, FFN):
        result = _ffn_rule(node)
    elif isinstance(node, Attn):
        result = _attn_rule(node)
    elif isinstance(node, Embedding):
        result = _embedding_rule(node)
        assert result.n_cols > 0, "Embedding must produce non-degenerate affine bound"
    else:
        from torchwright.graph.misc import Placeholder

        if not isinstance(node, Placeholder):
            raise NotImplementedError(
                f"No affine rule for {type(node).__name__}. Add one to "
                f"affine_rules.py or register it as an explicit pass-through."
            )
        result = AffineBound.degenerate(node.d_output)
    return result


def _linear_affine(
    inp_ab: AffineBound, W: torch.Tensor, c: torch.Tensor
) -> AffineBound:
    """Y = x @ W + c through a bound: sign-split GEMM.

    ``W`` is ``(d_input, d_output)`` and ``c`` is ``(d_output,)``, both
    float64.  Pure function of the input bound and the affine params — shared
    by :func:`_linear_rule` (the ``Linear`` node) and :func:`_ffn_rule` (the
    gate/out projections of an ``FFN``).
    """
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


def _linear_rule(node: Linear) -> AffineBound:
    """Y = x @ W + c: sign-split GEMM."""
    inp_ab = node.inputs[0]._affine_bound
    W = node.output_matrix.to(torch.float64)
    c = node.output_bias.to(torch.float64)
    return _linear_affine(inp_ab, W, c)


def _add_rule(node: Add) -> AffineBound:
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


def _concat_rule(node: Concatenate) -> AffineBound:
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


def _relu_affine(inp_ab: AffineBound, warn_node: Node | None = None) -> AffineBound:
    """ReLU per-component case analysis using the linear envelope.

    Pure function of the input bound.  ``warn_node`` (optional) is only used
    to emit the "infinite affine interval but finite value_type" warning
    against a real node; pass ``None`` when there is no single node behind the
    bound (e.g. an ``FFN``'s internal gate values).
    """
    intervals = inp_ab.to_interval()
    d = inp_ab.d_output
    n = inp_ab.n_cols

    A_lo = torch.zeros(d, n, dtype=torch.float64)
    A_hi = torch.zeros(d, n, dtype=torch.float64)
    b_lo = torch.zeros(d, dtype=torch.float64)
    b_hi = torch.zeros(d, dtype=torch.float64)

    for i in range(d):
        lo_i, h = intervals[i].lo, intervals[i].hi
        if lo_i >= 0:
            A_lo[i] = inp_ab.A_lo[i]
            A_hi[i] = inp_ab.A_hi[i]
            b_lo[i] = inp_ab.b_lo[i]
            b_hi[i] = inp_ab.b_hi[i]
        elif h <= 0:
            pass
        elif math.isinf(lo_i) or math.isinf(h):
            if warn_node is not None:
                vt_range = warn_node.value_type.value_range
                if math.isfinite(vt_range.lo) and math.isfinite(vt_range.hi):
                    import warnings

                    warnings.warn(
                        f"Node {warn_node.node_id}: ReLU input affine interval is "
                        f"infinite ({lo_i}, {h}) but value_type is finite "
                        f"{vt_range}. Upstream affine rule may be missing or "
                        f"degenerate.",
                        stacklevel=2,
                    )
            b_hi[i] = float(h)
        else:
            slope = h / (h - lo_i)
            A_hi[i] = slope * inp_ab.A_hi[i]
            b_hi[i] = slope * (inp_ab.b_hi[i] - lo_i)
            alpha = h / (h - lo_i)
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


def _relu_rule(node: ReLU) -> AffineBound:
    """ReLU per-component case analysis using linear envelope."""
    return _relu_affine(node.inputs[0]._affine_bound, warn_node=node.inputs[0])


# Swish is sandwiched by ReLU globally: relu(z) - C <= swish(z) <= relu(z)
# for every z (upper: z*sigmoid(z) <= z for z >= 0 and <= 0 for z <= 0;
# lower: swish's global minimum is -0.2784645... at z = -1.2784645..., and
# for z > 0 the gap z - swish(z) = z*sigmoid(-z) has the same maximum by
# symmetry).  C is the minimum's magnitude rounded UP so the sandwich stays
# sound in float64.
_SWISH_SANDWICH_C = 0.278465
_SWISH_ARGMIN = -1.278464542761074


def _swish_affine(inp_ab: AffineBound) -> AffineBound:
    """Swish envelope: the ReLU envelope with the lower offset shifted by C.

    Sound pointwise because relu(z) - C <= swish(z) <= relu(z) everywhere,
    so any linear under-estimator of ReLU minus C under-estimates swish and
    any linear over-estimator of ReLU over-estimates it.  The slack is the
    constant C per lane — it does not grow with the value magnitude.
    """
    ab = _relu_affine(inp_ab, warn_node=None)
    return AffineBound(
        A_lo=ab.A_lo,
        A_hi=ab.A_hi,
        b_lo=ab.b_lo - _SWISH_SANDWICH_C,
        b_hi=ab.b_hi,
        columns=ab.columns,
        input_ranges=ab.input_ranges,
    )


def _swish_scalar(v: float) -> float:
    """swish(v) = v * sigmoid(v), overflow-safe for any float (incl. ±inf)."""
    if math.isinf(v):
        return v if v > 0 else 0.0
    if v >= 0:
        return v / (1.0 + math.exp(-v))
    t = math.exp(v)  # underflows to 0 for very negative v; v*t then rounds to -0.0
    return v * t / (1.0 + t)


def _swish_interval(lo: float, hi: float) -> tuple:
    """Exact range of swish over [lo, hi].

    Endpoints plus the interior minimum when the interval contains
    swish's argmin.  (Swish decreases until the argmin and increases
    after it, so the maximum is always at an endpoint.)  Uses -C for
    the interior minimum — rounded below the true minimum, keeping
    the range sound.
    """
    f_lo, f_hi = _swish_scalar(lo), _swish_scalar(hi)
    out_lo, out_hi = min(f_lo, f_hi), max(f_lo, f_hi)
    if lo <= _SWISH_ARGMIN <= hi:
        out_lo = min(out_lo, -_SWISH_SANDWICH_C)
    return out_lo, out_hi


def _bilinear_comb(
    a_s: torch.Tensor,
    a_u: torch.Tensor,
    g: torch.Tensor,
    s_ab: AffineBound,
    u_ab: AffineBound,
    lower: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-lane affine under- or over-estimator of ``a_s*s + a_u*u + g``.

    ``lower=True`` selects the under-estimator, ``lower=False`` the
    over-estimator: route each coefficient to the factor's lower or
    upper affine expression by sign — the same clamp-and-route logic
    as the sign-split GEMM in :func:`_linear_affine`, per lane.
    """

    def pick(coef: torch.Tensor, ab: AffineBound) -> tuple[torch.Tensor, torch.Tensor]:
        take_lo = (coef >= 0) if lower else (coef < 0)
        A = torch.where(take_lo.unsqueeze(1), ab.A_lo, ab.A_hi)
        b = torch.where(take_lo, ab.b_lo, ab.b_hi)
        # 0 * ±inf must contribute 0, matching _safe_matvec semantics.
        return coef.unsqueeze(1) * A, torch.where(
            coef == 0, torch.zeros_like(b), coef * b
        )

    A_s, b_s = pick(a_s, s_ab)
    A_u, b_u = pick(a_u, u_ab)
    return A_s + A_u, b_s + b_u + g


def _gated_lane_affine(
    s_ab: AffineBound, u_ab: AffineBound, s_iv: list, u_iv: list
) -> AffineBound:
    """Affine bound on the per-lane product ``w = s * u``.

    ``s`` is the activated gate and ``u`` is the up projection.  A
    product of two affine-bounded quantities is not affine, but given
    interval endpoints ``s in [sl, sh]``, ``u in [ul, uh]`` that hold
    pointwise, the four standard linear product relaxations (the McCormick
    inequalities) are::

        w >= ul*s + sl*u - sl*ul      w <= ul*s + sh*u - sh*ul
        w >= uh*s + sh*u - sh*uh      w <= uh*s + sl*u - sl*uh

    Each is affine in (s, u), so substituting the factors' affine bounds
    keeps the result affine in the graph inputs.  Per side, both candidates
    are built and the tighter one (by concretized interval over the input
    box) is kept per lane.  A lane with a non-finite factor interval falls
    back to an unbounded row — sound, and it degenerates only that lane.
    """
    assert s_ab.columns == u_ab.columns, "gate/up bounds must share a basis"
    sl = torch.tensor([iv[0] for iv in s_iv], dtype=torch.float64)
    sh = torch.tensor([iv[1] for iv in s_iv], dtype=torch.float64)
    ul = torch.tensor([iv[0] for iv in u_iv], dtype=torch.float64)
    uh = torch.tensor([iv[1] for iv in u_iv], dtype=torch.float64)
    finite = (
        torch.isfinite(sl)
        & torch.isfinite(sh)
        & torch.isfinite(ul)
        & torch.isfinite(uh)
    )
    # Zero non-finite corner coefficients so candidate construction stays
    # NaN-free; those lanes' rows are overwritten with ±inf offsets below.
    zero = torch.zeros((), dtype=torch.float64)
    sl_, sh_ = torch.where(finite, sl, zero), torch.where(finite, sh, zero)
    ul_, uh_ = torch.where(finite, ul, zero), torch.where(finite, uh, zero)

    A_L1, b_L1 = _bilinear_comb(ul_, sl_, -sl_ * ul_, s_ab, u_ab, lower=True)
    A_L2, b_L2 = _bilinear_comb(uh_, sh_, -sh_ * uh_, s_ab, u_ab, lower=True)
    A_U1, b_U1 = _bilinear_comb(ul_, sh_, -sh_ * ul_, s_ab, u_ab, lower=False)
    A_U2, b_U2 = _bilinear_comb(uh_, sl_, -sl_ * uh_, s_ab, u_ab, lower=False)

    bad = ~finite
    for A, b, fill in (
        (A_L1, b_L1, float("-inf")),
        (A_L2, b_L2, float("-inf")),
        (A_U1, b_U1, float("inf")),
        (A_U2, b_U2, float("inf")),
    ):
        A[bad] = 0.0
        b[bad] = fill

    def bound(
        A_lo: torch.Tensor,
        b_lo: torch.Tensor,
        A_hi: torch.Tensor,
        b_hi: torch.Tensor,
    ) -> AffineBound:
        return AffineBound(
            A_lo=A_lo,
            A_hi=A_hi,
            b_lo=b_lo,
            b_hi=b_hi,
            columns=s_ab.columns,
            input_ranges=s_ab.input_ranges,
        )

    iv1 = bound(A_L1, b_L1, A_U1, b_U1).to_interval()
    iv2 = bound(A_L2, b_L2, A_U2, b_U2).to_interval()
    # dtype=torch.bool: an empty comparison list would otherwise default to
    # float32, which torch.where rejects as a condition (a zero-lane FFN's
    # bound is legitimately empty and must flow through well-typed).
    lo2 = torch.tensor(
        [r2.lo > r1.lo for r1, r2 in zip(iv1, iv2, strict=False)], dtype=torch.bool
    )
    hi2 = torch.tensor(
        [r2.hi < r1.hi for r1, r2 in zip(iv1, iv2, strict=False)], dtype=torch.bool
    )
    return bound(
        torch.where(lo2.unsqueeze(1), A_L2, A_L1),
        torch.where(lo2, b_L2, b_L1),
        torch.where(hi2.unsqueeze(1), A_U2, A_U1),
        torch.where(hi2, b_U2, b_U1),
    )


def _ffn_rule(node: FFN) -> AffineBound:
    """FFN: gate projection, activation envelope, up projection, then output.

    The degenerate-ReLU form composes exactly the rules the mined
    ``Linear -> ReLU -> Linear`` chain applied across three nodes.  The
    swish envelope is the ReLU envelope shifted by the sandwich constant
    (see :func:`_swish_affine`); a gated lane adds the product relaxation
    (see :func:`_gated_lane_affine`), whose pointwise-valid factor
    intervals come from the exact 1-D activation range over the gate's
    concretized interval — not from concretizing the activation envelope,
    which is far looser at wide gate intervals.
    """
    inp_ab = node.inputs[0]._affine_bound
    # Gate projection: y = x @ gate_proj.T + gate_bias.  gate_proj is
    # (n_lanes, d_input) in math orientation, so W = gate_proj.T.
    gate_W = node.gate_proj.to(torch.float64).T
    gate_c = node.gate_bias.to(torch.float64)
    gate_ab = _linear_affine(inp_ab, gate_W, gate_c)
    if node.activation == "relu":
        act_ab = _relu_affine(gate_ab, warn_node=None)
    else:
        act_ab = _swish_affine(gate_ab)
    if node.up_proj is None:
        lane_ab = act_ab
    else:
        # FFN.__init__ enforces up_proj/up_bias both-or-neither, so up_bias
        # is not None here (mypy can't see that cross-attribute invariant).
        assert node.up_bias is not None
        up_W = node.up_proj.to(torch.float64).T
        up_c = node.up_bias.to(torch.float64)
        u_ab = _linear_affine(inp_ab, up_W, up_c)
        gate_iv = gate_ab.to_interval()
        if node.activation == "relu":
            s_iv = [(max(r.lo, 0.0), max(r.hi, 0.0)) for r in gate_iv]
        else:
            s_iv = [_swish_interval(r.lo, r.hi) for r in gate_iv]
        u_iv = [(r.lo, r.hi) for r in u_ab.to_interval()]
        lane_ab = _gated_lane_affine(act_ab, u_ab, s_iv, u_iv)
    # Output projection: out = lane @ out_proj + out_bias.  out_proj is already
    # (n_lanes, d_output).
    out_W = node.out_proj.to(torch.float64)
    out_c = node.out_bias.to(torch.float64)
    return _linear_affine(lane_ab, out_W, out_c)


def _apply_claim_to_bound(node: Node) -> None:
    """Fold ``node.claimed_type`` into ``node._affine_bound`` in place.

    The two channels that used to live in the Assert wrapper's affine
    rule, applied to the claimed node itself:

    * **Leaf channel** — a claim on an ``InputNode``/``Embedding``
      tightens that leaf's own ``input_ranges`` entry, so every
      downstream bound inherits it through normal topological
      recomputation.
    * **General channel** — a finite claim degenerates the bound to the
      claim-intersected constant box (identical math to the old wrapper
      rule, one node earlier — the wrapper was a pass-through directly
      above the node).

    A non-finite claim on a non-leaf changes nothing here (it still
    tightens ``_structural_type`` in :func:`refresh_node_caches`).
    Idempotent: intersecting the same claim twice is a no-op.
    """
    if node.claimed_type is None:
        return
    from torchwright.graph.embedding import Embedding
    from torchwright.graph.misc import InputNode

    claimed_range = node.claimed_type.value_range
    ab = node._affine_bound

    if isinstance(node, (InputNode, Embedding)) and node.node_id in ab.input_ranges:
        old_lo, old_hi = ab.input_ranges[node.node_id]
        new_ranges = dict(ab.input_ranges)
        new_ranges[node.node_id] = (
            torch.maximum(old_lo, torch.full_like(old_lo, claimed_range.lo)),
            torch.minimum(old_hi, torch.full_like(old_hi, claimed_range.hi)),
        )
        node._affine_bound = AffineBound(
            A_lo=ab.A_lo,
            A_hi=ab.A_hi,
            b_lo=ab.b_lo,
            b_hi=ab.b_hi,
            columns=ab.columns,
            input_ranges=new_ranges,
        )
        return

    if claimed_range.is_finite():
        intervals = ab.to_interval()
        d = node.d_output
        b_lo = torch.tensor(
            [max(iv.lo, claimed_range.lo) for iv in intervals],
            dtype=torch.float64,
        )
        b_hi = torch.tensor(
            [min(iv.hi, claimed_range.hi) for iv in intervals],
            dtype=torch.float64,
        )
        node._affine_bound = AffineBound(
            A_lo=torch.zeros(d, 0, dtype=torch.float64),
            A_hi=torch.zeros(d, 0, dtype=torch.float64),
            b_lo=b_lo,
            b_hi=b_hi,
            columns={},
            input_ranges={},
        )


def _attn_rule(node: Attn) -> AffineBound:
    """Attn: propagate value bounds through V then O.

    Softmax produces convex-combination weights, so per-component
    affine bounds on ``value @ V`` carry through unchanged.  O is
    then a standard linear step.
    """
    value_ab = node.inputs[2]._affine_bound
    V = node.value_matrix.to(torch.float64)
    O_mat = node.output_matrix.to(torch.float64)

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

    O_plus = torch.clamp(O_mat, min=0)
    O_minus = torch.clamp(O_mat, max=0)
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


def _embedding_rule(node: Embedding) -> AffineBound:
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


def _apply_semantic_override(node: Node, semantic_ab: AffineBound | None) -> None:
    """Replace *node*'s affine bound with a semantic override.

    The override is also recorded on the node
    (``_semantic_affine_override``) so :func:`refresh_node_caches` can
    re-apply it after recomputing the propagated bound — otherwise any
    post-construction recompute would silently drop the semantic
    tightening and loosen every downstream bound derived from it.

    An override WINS over the node's range claim on the affine channel:
    ops install overrides after attaching their claims, and the
    override's correlation structure (input columns) is worth more
    downstream than the claim's constant box.  The claim still tightens
    ``_structural_type``, so ``value_type`` stays claim-tight.
    Pinned by ``test_cond_gate_preserves_correlation``
    (tests/graph/test_affine_bounds.py).
    """
    if semantic_ab is None:
        return
    propagated = node._affine_bound.to_scalar_range()
    semantic = semantic_ab.to_scalar_range()
    disjoint_msg = (
        f"Semantic override on node {node.node_id} is disjoint from "
        f"propagated bound: semantic={semantic}, propagated={propagated}"
    )
    assert semantic.lo <= propagated.hi, disjoint_msg
    assert semantic.hi >= propagated.lo, disjoint_msg
    node._semantic_affine_override = semantic_ab
    node._affine_bound = semantic_ab


def refresh_node_caches(node: Node) -> None:
    """Recompute *node*'s eagerly-cached derived data from its inputs.

    Recomputes ``_structural_type`` and ``_affine_bound`` (both computed
    eagerly in ``Node.__init__`` and therefore stale after any in-place
    mutation of the node or an ancestor), then re-applies the node's
    semantic affine override if an op installed one, and finally its
    range claim (``claimed_type``) on both the structural and affine
    channels.  This is the single choke point for claim application —
    there is no code path that recomputes bounds without re-applying
    claims, so claims are refresh-proof by construction
    (docs/assert_metadata_plan.md).  Callers must invoke
    this in an order where a node's inputs are refreshed before the node
    itself (any topological order works; the fusion pass uses node-id
    order, which its folds keep topological).
    """
    node._structural_type = node.compute_value_type()
    if node.claimed_type is not None:
        from torchwright.graph.value_type import tightened_with

        node._structural_type = tightened_with(node._structural_type, node.claimed_type)
    node._affine_bound = compute_affine_bound(node)
    if node._semantic_affine_override is not None:
        # Override wins on the affine channel (see _apply_semantic_override);
        # the claim reached _structural_type above, keeping value_type tight.
        _apply_semantic_override(node, node._semantic_affine_override)
    else:
        _apply_claim_to_bound(node)


def _cond_gate_semantic_bound(
    inp_ab: AffineBound,
    inp_node: Node | None = None,
    c_tol: float = 0.0,
    M: float = 0.0,
    rel_tol: float = 0.0,
) -> AffineBound:
    """Per-component [min(0, inp), max(0, inp)] envelope for cond_gate.

    Two widening modes for condition noise, matching the two gate
    machines:

    - ReLU (additive-cancellation gate): ``c_tol > 0`` and ``M > 0``
      widen the envelope by the absolute ``c_tol * M`` per component —
      a cond off ±1 by δ leaks δ·M of the offset.
    - swish (direct gated lanes): ``rel_tol > 0`` scales the whole
      envelope by ``(1 + rel_tol)`` — a saturated swish gate is linear
      in the cond, so a cond off ±1 by δ mis-scales the output by
      exactly δ·|actual value|, never δ·M.  Scaling is sound on both
      sides: the upper envelope bounds a pointwise non-negative
      function (so ``(1+δ)·U ≥ max(0, (1+δ')·v)``), the lower a
      pointwise non-positive one.
    """
    intervals = inp_ab.to_interval()
    d = inp_ab.d_output
    n = inp_ab.n_cols

    A_lo = torch.zeros(d, n, dtype=torch.float64)
    A_hi = torch.zeros(d, n, dtype=torch.float64)
    b_lo = torch.zeros(d, dtype=torch.float64)
    b_hi = torch.zeros(d, dtype=torch.float64)

    for i in range(d):
        lo, h = intervals[i].lo, intervals[i].hi
        if lo >= 0:
            A_hi[i] = inp_ab.A_hi[i]
            b_hi[i] = inp_ab.b_hi[i]
        elif h <= 0:
            A_lo[i] = inp_ab.A_lo[i]
            b_lo[i] = inp_ab.b_lo[i]
        elif math.isinf(lo) or math.isinf(h):
            if inp_node is not None:
                vt_range = inp_node.value_type.value_range
                if math.isfinite(vt_range.lo) and math.isfinite(vt_range.hi):
                    import warnings

                    warnings.warn(
                        f"Node {inp_node.node_id}: cond_gate input affine "
                        f"interval is infinite ({lo}, {h}) but value_type is "
                        f"finite {vt_range}. Upstream affine rule may be "
                        f"missing or degenerate.",
                        stacklevel=2,
                    )
            b_lo[i] = float(min(0, lo))
            b_hi[i] = float(max(0, h))
        else:
            s_hi = h / (h - lo)
            A_hi[i] = s_hi * inp_ab.A_hi[i]
            b_hi[i] = s_hi * (inp_ab.b_hi[i] - lo)
            s_lo = -lo / (h - lo)
            A_lo[i] = s_lo * inp_ab.A_lo[i]
            b_lo[i] = s_lo * inp_ab.b_lo[i] + lo * h / (h - lo)

    widen = c_tol * M
    b_lo -= widen
    b_hi += widen

    if rel_tol > 0.0:
        s = 1.0 + rel_tol
        A_lo, A_hi = s * A_lo, s * A_hi
        b_lo, b_hi = s * b_lo, s * b_hi

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
    rel_tolerance: float = 0.0,
) -> AffineBound:
    """Per-component hull of a and b intervals for select.

    Two widening modes for condition noise, matching the two gate
    machines: ``tolerance`` widens the hull by an absolute amount per
    component (ReLU gate: a cond off ±1 by δ leaks δ·M of the offset);
    ``rel_tolerance`` widens each hull side by ``rel_tolerance·|side|``
    (swish gate: the saturated gate is linear in the cond, so a cond off
    by δ mis-scales the winning branch by exactly δ·|actual value| —
    the extreme scaled values are ``lo - δ·|lo|`` and ``hi + δ·|hi|``).
    """
    import torch

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

    if rel_tolerance > 0.0:
        b_lo -= rel_tolerance * b_lo.abs()
        b_hi += rel_tolerance * b_hi.abs()

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
    rel_tolerance: float = 0.0,
) -> AffineBound:
    """Per-component hull across slots for broadcast_select.

    Each output slot picks true or false, so the per-channel bound is
    the hull of the corresponding true/false channel. When a branch is
    broadcast, its single-slot interval applies to every slot.

    Widening modes mirror :func:`_select_semantic_bound`: ``tolerance``
    is the ReLU machine's absolute ``c_tol·M`` per component;
    ``rel_tolerance`` is the swish machine's per-side
    ``rel_tolerance·|hull side|`` (a saturated gate is linear in the
    mask, so a mask off ±1 by δ mis-scales the winner by δ·|actual
    value| — for broadcast_select δ is the *mask* tolerance, sized to
    absorb in_range's dip, not the stricter select/cond_gate c_tol).
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

    if rel_tolerance > 0.0:
        b_lo -= rel_tolerance * b_lo.abs()
        b_hi += rel_tolerance * b_hi.abs()

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
    slack: float = 0.0,
) -> AffineBound:
    """Constant or degenerate bound for compare, depending on inp vs thresh.

    The soundness argument is the caller contract, not the construction:
    inputs never land inside the ramp, so an input interval clearing
    ``thresh`` collapses the bound to the corresponding level.

    ``slack`` widens every case symmetrically.  The ReLU compare passes 0
    (its saturated levels are exact); the swish compare passes its bend
    overshoot ``swish_dip/scale · |true_level - false_level|`` — inputs
    within the fillet just past a contract point legitimately read up to
    that far beyond the level, so the constant collapse is an interval
    there, not a constant.
    """
    import torch

    intervals = inp_ab.to_interval()
    assert len(intervals) == 1
    iv_lo, iv_hi = intervals[0].lo, intervals[0].hi

    lo = min(true_level, false_level)
    hi = max(true_level, false_level)

    if iv_lo > thresh:
        if slack == 0.0:
            return AffineBound.constant(torch.tensor([true_level], dtype=torch.float64))
        return AffineBound.degenerate(1, lo=true_level - slack, hi=true_level + slack)
    if iv_hi <= thresh:
        if slack == 0.0:
            return AffineBound.constant(
                torch.tensor([false_level], dtype=torch.float64)
            )
        return AffineBound.degenerate(1, lo=false_level - slack, hi=false_level + slack)
    return AffineBound.degenerate(1, lo=lo - slack, hi=hi + slack)
