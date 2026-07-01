import builtins
from typing import List, Optional, Tuple

from torchwright.graph import Node, Add, Concatenate, Linear
import torch

from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.relu import ReLU
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.linear_relu_linear import linear_relu_linear

from torchwright.ops.const import step_sharpness

_builtin_abs = builtins.abs  # save before module-level def abs() shadows the builtin


# ---------------------------------------------------------------------------
# Linear ops (no MLP sublayers — compiled into residual-stream wiring)
# ---------------------------------------------------------------------------


def add(inp1: Node, inp2: Node) -> Node:
    """
    Performs element-wise addition of two input nodes.

    Args:
        inp1 (Node): First node for addition.
        inp2 (Node): Second node for addition.

    Returns:
        Node: Node resulting from element-wise addition.
    """
    return Add(inp1, inp2)


def subtract(inp1: Node, inp2: Node) -> Node:
    """
    Subtracts inp2 from inp1 element-wise.

    Args:
        inp1 (Node): Node to subtract from.
        inp2 (Node): Node to subtract.

    Returns:
        Node: Node resulting from inp1 - inp2.
    """
    return add(inp1, negate(inp2))


def negate(inp: Node) -> Node:
    """
    Negates the input node (multiplies by -1).

    Args:
        inp (Node): Node to negate.

    Returns:
        Node: Node with negated values.
    """
    d = len(inp)
    return Linear(inp, -torch.eye(d), name="negate")


def add_const(inp: Node, scalar: float) -> Node:
    """
    Adds a scalar value to each entry of the input node.

    Args:
        inp (Node): Node whose values will have the scalar added.
        scalar (float): Scalar value to add.

    Returns:
        Node: Output node with scalar added to each entry.
    """
    d = len(inp)
    return Linear(
        inp,
        torch.eye(d),
        torch.tensor([scalar] * d),
        name="add_const",
    )


def multiply_const(inp: Node, scalar: float) -> Node:
    """
    Multiplies each entry of the input node by a scalar.

    Args:
        inp (Node): Node to scale.
        scalar (float): Scalar multiplier.

    Returns:
        Node: Node with scaled values.
    """
    d = len(inp)
    return Linear(inp, scalar * torch.eye(d), name="multiply_const")


def bool_to_01(inp: Node) -> Node:
    """Map a ±1 boolean node to 0/1.

    Converts the torchwright boolean convention (+1 = true, −1 = false)
    to a 0/1 scale (1 = true, 0 = false).  This is a free operation
    (no MLP sublayers — two linear transforms).

    Args:
        inp (Node): Boolean node with values in {-1, +1}.

    Returns:
        Node: Node with values in {0, 1}.
    """
    return multiply_const(add_const(inp, 1.0), 0.5)


def add_scaled_nodes(scale1: float, inp1: Node, scale2: float, inp2: Node) -> Node:
    """
    Computes the linear combination of two nodes using specified coefficients.

    Args:
        scale1 (float): Coefficient for the first node.
        inp1 (Node): First node.
        scale2 (float): Coefficient for the second node.
        inp2 (Node): Second node.

    Returns:
        Node: Node resulting from the linear combination of input nodes.
    """
    assert len(inp1) == len(inp2)
    d = len(inp1)

    concat = Concatenate([inp1, inp2])
    M = torch.zeros(len(concat), d)
    for i in range(d):
        M[i, i] = scale1
        M[d + i, i] = scale2

    return Linear(concat, M)


def sum_nodes(inp_list: List[Node], *, max_fanout: Optional[int] = None) -> Node:
    """Compute the sum of all input nodes.

    Args:
        inp_list: List of nodes to be summed.  All must have the same
            output width.
        max_fanout: Optional cap on the number of operands combined in a
            single reduction step.  ``None`` (default) produces a single
            flat ``Linear`` that takes all N operands at once — shallow
            but holds all inputs on the residual stream simultaneously.
            Setting ``max_fanout=k >= 2`` chains the reduction through a
            running accumulator so at most ``k`` operands are alive at
            any reduction step: wider input lists trade one Linear per
            chunk for a correspondingly lower peak stream footprint.
            Prefer the dial when the input list is large and each
            operand is wide (e.g. H*3 pixel bands in the renderer).

    Returns:
        Node holding the elementwise sum.
    """
    d_values = {len(node) for node in inp_list}
    assert len(d_values) == 1
    d = d_values.pop()

    if max_fanout is not None and max_fanout < 2:
        raise ValueError(f"max_fanout must be >= 2, got {max_fanout}")

    def _flat(nodes: List[Node]) -> Node:
        x = Concatenate(nodes)
        output_matrix = torch.zeros(len(x), d)
        for i in range(len(x)):
            output_matrix[i, i % d] = 1.0
        return Linear(input_node=x, output_matrix=output_matrix)

    if max_fanout is None or len(inp_list) <= max_fanout:
        return _flat(inp_list)

    # Chain into a running accumulator so at most ``max_fanout`` operands
    # are live per reduction step.  The first chunk uses the full fanout;
    # subsequent chunks leave one slot for the running accumulator so the
    # total alive per step stays at ``max_fanout``.
    running = _flat(inp_list[:max_fanout])
    chunk = max_fanout - 1
    for start in range(max_fanout, len(inp_list), chunk):
        group = inp_list[start : start + chunk]
        running = _flat([running] + list(group))
    return running


def concat(inp_list: List[Node]) -> Node:
    """
    Concatenates all the Nodes in inp_list.

    Args:
        inp_list (List[Node]): List of nodes to concatenate

    Returns:
        Node: Node resulting from concatenation
    """
    return Concatenate(inp_list)


# ---------------------------------------------------------------------------
# ReLU-based ops (1 MLP sublayer each)
# ---------------------------------------------------------------------------


def relu_add(inp1: Node, inp2: Node) -> Node:
    """
    Applies the ReLU function to both input nodes and then adds them together.

    Args:
        inp1 (Node): First node for ReLU and addition.
        inp2 (Node): Second node for ReLU and addition.

    Returns:
        Node: Node resulting from ReLU application and addition.
    """
    # Rectifies val1 and val2 and then adds them together.
    # Equivalent to torch.clamp(val1, min=0) + torch.clamp(val2, min=0)
    assert len(inp1) == len(inp2)
    x = Concatenate([inp1, inp2])

    input_proj = torch.eye(len(x))
    input_bias = torch.zeros(len(x))
    output_proj = torch.zeros((len(x), len(inp1)))
    output_bias = torch.zeros(len(inp1))

    for i in range(len(inp1)):
        output_proj[i, i] = 1.0
        output_proj[len(inp1) + i, i] = 1.0

    return linear_relu_linear(
        input_node=x,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )


def abs(inp: Node) -> Node:
    """Element-wise absolute value.

    Equivalent to ``ReLU(x) + ReLU(-x)``.

    Args:
        inp: Node of any width.

    Returns:
        Node of the same width containing ``|x|`` element-wise.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    return relu_add(inp, negate(inp))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _compare_output_type(true_level: float, false_level: float) -> NodeValueType:
    lo = builtins.min(true_level, false_level)
    hi = builtins.max(true_level, false_level)
    return NodeValueType(value_range=Range(lo, hi))


def compare(
    inp: Node,
    thresh: float,
    true_level: float = 1.0,
    false_level: float = -1.0,
    sharpness: float | None = None,
) -> Node:
    """
    Compare input with threshold and return boolean valued node (1.0 for true, -1.0 for false)

    Args:
        inp: Node to compare. Must be length 1.
        thresh: Threshold to use.
        true_level: Value to return if inp is greater than thresh.
        false_level: Value to return if inp is less than thresh.
        sharpness: Override for the ramp sharpness.  Saturation requires
            ``(inp - thresh) * sharpness >= 1``, so the ramp width is
            ``1/sharpness`` in input units.  ``None`` (default) uses the
            module-level ``step_sharpness``.  Use a large override when
            the caller knows the closest non-matching input sits very
            near ``thresh`` (e.g. triangle-wave bit extraction where the
            feature is only ~7.6e-6 away from 0.5 at the critical input
            values).

    Returns:
        Node: Node with a value of true_level if inp is greater than thresh, false_level otherwise.

    .. noise-footer::

       Max error: 1.998 abs, 1.998 rel over 8192 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """

    assert len(inp) == 1, "Input must be a 1D scalar node"

    s = step_sharpness if sharpness is None else sharpness

    # We need 2 MLP entries, we'll use the equation:
    # y= (true_level-false_level) * [
    #   max(s*x - s*thresh, 0) - max(s*x - s*thresh - 1, 0)
    # ] + false_level

    input_proj = torch.tensor([[s], [s]])
    input_bias = torch.tensor([-s * thresh, -s * thresh - 1.0])
    output_proj = torch.tensor([[true_level - false_level], [false_level - true_level]])
    output_bias = false_level * torch.ones(1)

    result = linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )

    vt = _compare_output_type(true_level, false_level)
    if vt != NodeValueType.unknown():
        result = assert_matches_value_type(result, vt)
    from torchwright.graph.affine_rules import (
        _apply_semantic_override,
        _compare_semantic_bound,
    )

    _apply_semantic_override(
        result,
        _compare_semantic_bound(inp._affine_bound, thresh, true_level, false_level),
    )
    return result


# ---------------------------------------------------------------------------
# Element-wise min / max
# ---------------------------------------------------------------------------


def min(inp1: Node, inp2: Node) -> Node:
    """Element-wise minimum of two nodes.

    Computed as ``(a + b - |a - b|) / 2``.

    Args:
        inp1: First node.
        inp2: Second node (same width as *inp1*).

    Returns:
        Node of the same width containing ``min(inp1, inp2)`` element-wise.

    .. noise-footer::

       Max error: 3.815e-06 abs, 1.953e-06 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp1) == len(inp2)
    diff = subtract(inp1, inp2)
    abs_diff = abs(diff)
    sum_ab = add(inp1, inp2)
    return add_scaled_nodes(0.5, sum_ab, -0.5, abs_diff)


# ---------------------------------------------------------------------------
# Piecewise-linear foundation
# ---------------------------------------------------------------------------


def piecewise_linear(
    inp: Node,
    breakpoints: List[float],
    fn,
    clamp: bool = True,
    d_max: int = 1024,
    input_scale: float = 1.0,
    name: str = "piecewise_linear",
) -> Node:
    """Evaluate a piecewise-linear function defined by breakpoints and a callable.

    The callable *fn* is evaluated at each breakpoint to obtain the
    target values, then the function linearly interpolates between
    consecutive breakpoints.  By default the output is clamped to the
    endpoint values outside the breakpoint range.

    *fn* may return a scalar (``float``) or a vector (``List[float]``).
    When vector-valued, all output channels share the same breakpoints
    and neurons — only the output projection differs.  The output
    node has width D matching the vector length.

    Implemented as a sum of ReLU slope-changes::

        f(x) = y_0 + Σ delta_m_i · ReLU(x - x_i)

    where ``delta_m_i`` is the change in slope at breakpoint ``x_i``.
    Uses one neuron per slope change, so segments with equal slopes
    are free.  Cost: 1 MLP sublayer regardless of output width.

    Args:
        inp: 1D scalar node.
        breakpoints: Strictly ascending x-coordinates (length n >= 2).
        fn: ``fn(x) -> float`` or ``fn(x) -> List[float]`` evaluated at
            each breakpoint.  Vector returns must all have the same
            length.
        clamp: If True (default), hold constant outside the range.
            If False, extrapolate linearly.
        d_max: Maximum neurons per MLP sublayer (chunks beyond this).
        input_scale: Multiplier for the first-layer weights.  Each ReLU
            ``delta · ReLU(x - b)`` is rewritten as
            ``(delta/s) · ReLU(s·x - s·b)`` so the intermediate
            activations are amplified while the mathematical result is
            unchanged.  This matters when breakpoints are large: the
            bias ``-b`` may not be exact in float32, but ``-s·b`` can
            be (e.g. ``-999.45`` rounds, but ``-9994.5`` is exact).
            Use ``step_sharpness`` for step-function staircases where
            precision compounds across chained operations.
        name: Debug label prefix.

    Returns:
        Node of width 1 (scalar fn) or D (vector fn).

    .. noise-footer::

       Max error: 0.25 abs, 231.1 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    n = len(breakpoints)
    assert n >= 2, "Need >= 2 breakpoints"
    assert all(
        breakpoints[i] < breakpoints[i + 1] for i in range(n - 1)
    ), "Breakpoints must be strictly ascending"

    raw_values = [fn(x) for x in breakpoints]

    # Normalise to vector form: values[i] is always a list of length d_out.
    scalar = not isinstance(raw_values[0], (list, tuple))
    if scalar:
        values = [[v] for v in raw_values]
    else:
        values = [list(v) for v in raw_values]
    d_out = len(values[0])
    assert all(len(v) == d_out for v in values)

    # Per-segment slopes for each output dimension.
    slopes = []
    for i in range(n - 1):
        dx = breakpoints[i + 1] - breakpoints[i]
        slopes.append([(values[i + 1][j] - values[i][j]) / dx for j in range(d_out)])

    # Build ReLU list: (input_weight, threshold, [output_weights_per_dim])
    relus: list = []
    prev_slopes = [0.0] * d_out

    for i in range(n - 1):
        deltas = [slopes[i][j] - prev_slopes[j] for j in range(d_out)]
        if any(_builtin_abs(d) > 1e-12 for d in deltas):
            relus.append((1.0, breakpoints[i], deltas))
        prev_slopes = list(slopes[i])

    # Cancel final slope (clamp)
    if any(_builtin_abs(s) > 1e-12 for s in prev_slopes):
        relus.append((1.0, breakpoints[-1], [-s for s in prev_slopes]))

    if not clamp:
        if any(_builtin_abs(s) > 1e-12 for s in slopes[0]):
            relus.append((-1.0, breakpoints[0], [-s for s in slopes[0]]))
        if any(_builtin_abs(s) > 1e-12 for s in slopes[-1]):
            relus.append((1.0, breakpoints[-1], list(slopes[-1])))

    if len(relus) == 0:
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor(values[0]))

    y0 = torch.tensor(values[0])

    chunks = []
    for chunk_start in range(0, len(relus), d_max):
        chunk = relus[chunk_start : chunk_start + d_max]
        d = len(chunk)
        s = input_scale

        input_proj = torch.tensor([[r[0] * s] for r in chunk])  # (d, 1)
        input_bias = torch.tensor([-r[0] * s * r[1] for r in chunk])  # (d,)
        output_proj = torch.tensor(
            [[r[2][j] / s for j in range(d_out)] for r in chunk]
        )  # (d, d_out)

        ob = y0 if chunk_start == 0 else torch.zeros(d_out)

        chunks.append(
            linear_relu_linear(
                input_node=inp,
                input_proj=input_proj,
                input_bias=input_bias,
                output_proj=output_proj,
                output_bias=ob,
                name=f"{name}_{chunk_start}_{chunk_start + d}",
            )
        )

    if len(chunks) == 1:
        result = chunks[0]
    else:
        result = sum_nodes(chunks)

    # With clamp=True the output is bounded by the min/max of fn evaluated
    # on the breakpoints (per-channel). Declare that range so downstream
    # gating ops can derive a tight offset.
    if clamp:
        per_channel_los = [
            builtins.min(values[i][j] for i in range(n)) for j in range(d_out)
        ]
        per_channel_his = [
            builtins.max(values[i][j] for i in range(n)) for j in range(d_out)
        ]
        lo = float(builtins.min(per_channel_los))
        hi = float(builtins.max(per_channel_his))
        result = assert_matches_value_type(
            result, NodeValueType(value_range=Range(lo, hi))
        )
    return result


# ---------------------------------------------------------------------------
# Piecewise-linear derived ops
# ---------------------------------------------------------------------------


def _product_2d_quarter_square(
    u_node: Node,
    v_node: Node,
    n: int,
    lo1: float,
    range1: float,
    lo2: float,
    range2: float,
    d_max: int = 1024,
    name: str = "multiply_2d",
) -> Node:
    """Closed-form O(n) piecewise-linear interpolant of a bilinear product.

    Given two inputs already affine-mapped onto the common unit grid
    (``u_node, v_node`` ∈ [0, 1] sampled at ``n`` equally-spaced
    breakpoints, spacing ``h = 1/(n-1)``), build a single MLP sublayer
    that evaluates::

        P(u, v) = (lo1 + range1·u)·(lo2 + range2·v)

    without the generic least-squares solve.  ``P`` is bilinear, so the
    only nonlinear term is ``u·v``, and ``u·v = ((u+v)² − (u−v)²)/4``.
    The piecewise-linear interpolant of ``t²`` on a uniform grid of
    spacing ``h`` has a *constant* slope change of ``2h`` at every
    interior breakpoint, so each square is a ReLU staircase of
    equal-weight neurons.  The sum axis ``u+v`` ranges over ``[0, 2]``
    and the difference axis ``u−v`` over ``[−1, 1]``, both uniform with
    spacing ``h``: ``2·(2n−2)`` neurons total, all assembled into one
    ``linear_relu_linear``.

    The two squares use ``clamp=False`` extrapolation (the leading
    segment's slope, carried by the base affine term, extends past both
    ends).  That matches the product's own linear extrapolation outside
    the grid — beyond the grid a product's piecewise-affine approximation
    follows the last segment's slope, exactly as the generic
    :func:`piecewise_linear_2d` product did.

    Returns a 1-D scalar node holding ``P(u, v)``.
    """
    h = 1.0 / (n - 1)

    # square(t) on a uniform grid {t_k} (spacing h, clamp=False) expands to
    #     square(t) = const + m0·t + Σ_{k=1}^{m-2} 2h·ReLU(t − t_k)
    # where m0 = t_0 + t_1 is the leading segment slope (carried by the
    # affine part so the ReLUs, inactive left of t_0, never need a left
    # ramp) and const = t_0² − m0·t_0.
    def _square_expansion(t_bps: List[float]) -> Tuple[float, float, list]:
        t0 = t_bps[0]
        m0 = t_bps[0] + t_bps[1]
        const = t0 * t0 - m0 * t0
        relus = [(t_bps[k], 2.0 * h) for k in range(1, len(t_bps) - 1)]
        return const, m0, relus

    s_bps = [i * h for i in range(2 * n - 1)]  # u+v ∈ [0, 2]
    d_bps = [-1.0 + i * h for i in range(2 * n - 1)]  # u−v ∈ [−1, 1]
    cS, mS, rS = _square_expansion(s_bps)
    cD, mD, rD = _square_expansion(d_bps)

    # uv = (S2 − D2)/4, so the product's
    #   constant term  = lo1·lo2 + range1·range2·(cS − cD)/4
    #   u coefficient  = lo2·range1 + range1·range2·(mS − mD)/4
    #   v coefficient  = lo1·range2 + range1·range2·(mS + mD)/4
    prod_scale = range1 * range2
    const_term = lo1 * lo2 + prod_scale * (cS - cD) / 4.0
    coef_u = lo2 * range1 + prod_scale * (mS - mD) / 4.0
    coef_v = lo1 * range2 + prod_scale * (mS + mD) / 4.0

    inp = Concatenate([u_node, v_node])

    # Sum-axis neuron k: ReLU(u + v − thr) contributes +2h·(prod_scale/4).
    # Diff-axis neuron k: ReLU(u − v − thr) contributes −2h·(prod_scale/4).
    # (proj_u, proj_v, bias, output_weight)
    relus: list = []
    for thr, w in rS:
        relus.append((1.0, 1.0, -thr, prod_scale * w / 4.0))
    for thr, w in rD:
        relus.append((1.0, -1.0, -thr, -prod_scale * w / 4.0))

    base = Linear(
        inp,
        torch.tensor([[coef_u], [coef_v]]),
        name=f"{name}_base",
    )

    chunks = []
    multi = len(relus) > d_max
    for chunk_start in range(0, len(relus), d_max):
        chunk = relus[chunk_start : chunk_start + d_max]
        d = len(chunk)
        input_proj = torch.zeros(d, 2)
        input_bias = torch.zeros(d)
        output_proj = torch.zeros(d, 1)
        for k, (pu, pv, b, ow) in enumerate(chunk):
            input_proj[k, 0] = pu
            input_proj[k, 1] = pv
            input_bias[k] = b
            output_proj[k, 0] = ow
        ob = torch.tensor([const_term]) if chunk_start == 0 else torch.zeros(1)
        chunk_name = f"{name}_{chunk_start}_{chunk_start + d}" if multi else name
        chunks.append(
            linear_relu_linear(
                input_node=inp,
                input_proj=input_proj,
                input_bias=input_bias,
                output_proj=output_proj,
                output_bias=ob,
                name=chunk_name,
            )
        )

    relu_part = chunks[0] if len(chunks) == 1 else sum_nodes(chunks)
    return Add(base, relu_part)


def multiply_2d(
    inp1: Node,
    inp2: Node,
    max_abs1: float,
    max_abs2: float,
    step1: float = 1.0,
    step2: float = 1.0,
    breakpoints1: Optional[List[float]] = None,
    breakpoints2: Optional[List[float]] = None,
    d_max: int = 1024,
    name: str = "multiply_2d",
) -> Node:
    """Multiply two signed scalars via a 2D piecewise-linear lookup.

    Computes ``inp1 * inp2`` in a **single MLP sublayer** via an analytic
    quarter-square construction: both axes are normalized to a common unit
    grid and the bilinear product is built from ``((u+v)² − (u−v)²)/4`` as
    one ReLU bank of ~``2*(2n−2)`` neurons (O(n) in the breakpoint count).

    **When to prefer over** ``signed_multiply``:

    * The pipeline is depth-bound (most layers have spare MLP slots).
    * Input ranges are moderate — the grid precision ``step1 * step2 / 4``
      must be acceptable for the downstream consumer.  Feeding the output
      into ``floor_int`` requires ``step1 * step2 < 2`` to avoid bin errors.
    * You need one multiplication per layer rather than a 3-layer chain.

    **Breakpoint generation.** When ``breakpoints1`` / ``breakpoints2`` are
    not provided, uniform spacing from ``-max_abs1`` to ``max_abs1`` (and
    similarly for axis 2) at ``step`` intervals is used.  Custom breakpoints
    (like the non-uniform ``_DIFF_BP`` used elsewhere in the renderer) can be
    passed directly.

    Args:
        inp1: 1D scalar node.
        inp2: 1D scalar node.
        max_abs1: Upper bound on ``|inp1|``.
        max_abs2: Upper bound on ``|inp2|``.
        step1: Grid spacing for auto-generated breakpoints on axis 1.
        step2: Grid spacing for auto-generated breakpoints on axis 2.
        breakpoints1: Explicit breakpoints for axis 1.  Overrides
            ``max_abs1`` / ``step1``.
        breakpoints2: Explicit breakpoints for axis 2.  Overrides
            ``max_abs2`` / ``step2``.
        d_max: Maximum neurons in the underlying MLP sublayer.
        name: Node name for debugging.

    Returns:
        1D scalar node containing ``inp1 * inp2``.

    .. noise-footer::

       Max error: 0.06249 abs, 2.15 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp1) == 1, "inp1 must be a 1D scalar node"
    assert len(inp2) == 1, "inp2 must be a 1D scalar node"

    # --- Build breakpoints ---
    if breakpoints1 is None:
        lo1 = -max_abs1
        n1 = builtins.max(int(round((max_abs1 - lo1) / step1)) + 1, 2)
        breakpoints1 = [lo1 + i * step1 for i in range(n1)]
        breakpoints1[-1] = max_abs1  # pin endpoint
    if breakpoints2 is None:
        lo2 = -max_abs2
        n2 = builtins.max(int(round((max_abs2 - lo2) / step2)) + 1, 2)
        breakpoints2 = [lo2 + i * step2 for i in range(n2)]
        breakpoints2[-1] = max_abs2  # pin endpoint

    # --- Normalize both axes to [0, 1] with a common step ---
    #
    # The quarter-square construction places ReLU staircases along the
    # normalized sum axis u+v and difference axis u−v.  On a square grid
    # (equal step on both axes) these are uniform with a single spacing;
    # differing steps would make the sums/differences all distinct → O(n²)
    # neurons.  Affine-mapping both axes to [0, 1] with a common breakpoint
    # count guarantees equal step → O(n) neurons.  The mapping is done by two
    # free Linear nodes; the product fn(u, v) = (lo₁ + range₁·u)·(lo₂ +
    # range₂·v) is reconstructed below so the output is mathematically
    # identical.
    lo1_f = float(breakpoints1[0])
    hi1_f = float(breakpoints1[-1])
    lo2_f = float(breakpoints2[0])
    hi2_f = float(breakpoints2[-1])
    range1 = hi1_f - lo1_f
    range2 = hi2_f - lo2_f

    if range1 < 1e-12 or range2 < 1e-12:
        raise ValueError(
            f"{name}: degenerate zero-width input axis "
            f"(range1={range1}, range2={range2}); multiply_2d requires each "
            "axis to span a positive range."
        )
    else:
        n = builtins.max(len(breakpoints1), len(breakpoints2))
        bp_unit = [i / (n - 1) for i in range(n)]
        bp_unit[-1] = 1.0

        inp1_norm = Linear(
            inp1,
            torch.tensor([[1.0 / range1]]),
            torch.tensor([-lo1_f / range1]),
            name=f"{name}_norm1",
        )
        inp2_norm = Linear(
            inp2,
            torch.tensor([[1.0 / range2]]),
            torch.tensor([-lo2_f / range2]),
            name=f"{name}_norm2",
        )

        # Analytic O(n) product construction (quarter-square multiplier).
        #
        # The generic piecewise_linear_2d path solves a least-squares
        # system over an O(n²)-row vertex design matrix and an SVD-derived
        # nullspace — O(n⁴) work and O(n²) memory that OOMs at ~257
        # breakpoints per axis.  A product is exactly bilinear, so we build
        # its piecewise-linear interpolant in closed form instead:
        #
        #     P(u, v) = (lo₁ + r₁·u)·(lo₂ + r₂·v)
        #             = lo₁·lo₂ + lo₁·r₂·v + lo₂·r₁·u + r₁·r₂·(u·v)
        #
        # The only nonlinear term is u·v, and u·v = ((u+v)² − (u−v)²)/4.
        # The piecewise-linear interpolant of t² on a uniform grid of
        # spacing h has a *constant* slope change of 2h at every interior
        # breakpoint, so each square is a ReLU staircase with equal-weight
        # neurons.  On the common unit grid (h = 1/(n−1)) the sum axis
        # u+v ∈ [0, 2] and the difference axis u−v ∈ [−1, 1] are both
        # uniform with spacing h, so the whole product is one ReLU bank of
        # ~2·(2n−2) neurons (O(n)) — same single MLP sublayer, no solve.
        result = _product_2d_quarter_square(
            inp1_norm,
            inp2_norm,
            n,
            lo1_f,
            range1,
            lo2_f,
            range2,
            d_max=d_max,
            name=name,
        )

        # Declare the output range.  The interpolant equals the exact
        # product at grid vertices and extrapolates linearly outside, so
        # over an axis-aligned input box its extreme is attained at a
        # corner.  Bound over the box covering both the grid extent and
        # the (finite) declared input ranges; a product over a rectangle
        # is monotone in each argument, so the min/max sit at the four
        # corners.  This mirrors the tight range the generic
        # piecewise_linear_2d path declared, so downstream gating that
        # reads multiply_2d's value range is unaffected.
        r1v = inp1.value_type.value_range
        r2v = inp2.value_type.value_range
        if r1v.is_finite() and r2v.is_finite():
            ax_lo = builtins.min(lo1_f, r1v.lo)
            ax_hi = builtins.max(hi1_f, r1v.hi)
            bx_lo = builtins.min(lo2_f, r2v.lo)
            bx_hi = builtins.max(hi2_f, r2v.hi)
            corners = [
                ax_lo * bx_lo,
                ax_lo * bx_hi,
                ax_hi * bx_lo,
                ax_hi * bx_hi,
            ]
            result = assert_matches_value_type(
                result,
                NodeValueType(
                    value_range=Range(builtins.min(corners), builtins.max(corners))
                ),
            )

    return result


def square(inp: Node, max_value: float, step: float = 1.0) -> Node:
    """Compute x² via piecewise-linear interpolation.

    Exact when x is a multiple of ``step``. Between grid points the
    result is a piecewise-linear interpolation (always an underestimate
    of x²).

    Implemented via :func:`piecewise_linear` with breakpoints at
    multiples of ``step`` from 0 to ``max_value``.

    Args:
        inp: 1D scalar node with value in [0, max_value].
        max_value: Upper bound on input.
        step: Grid spacing. Exact for multiples of step. Smaller values
            give better accuracy for non-grid inputs at the cost of more
            breakpoints.

    Returns:
        1D scalar node containing x² (exact at grid points).

    .. noise-footer::

       Max error: 0.25 abs, 231.1 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_value > 0, "max_value must be positive"

    breakpoints = []
    x = 0.0
    while x <= max_value + step / 2.0:
        breakpoints.append(x)
        x += step

    return piecewise_linear(inp, breakpoints, lambda x: x * x, name="square")


def thermometer_floor_div(inp: Node, divisor: int, max_value: int) -> Node:
    """Compute floor(inp / divisor) using a piecewise-linear staircase.

    Places a steep ramp at each multiple of the divisor.  Half-integer
    thresholds (9.5 not 10.0) ensure clean separation for integer inputs.

        x = 35, divisor = 10  →  output = 3 = floor(35/10)

    Implemented via :func:`piecewise_linear` with a staircase whose
    transition width is ``1 / step_sharpness``.

    .. warning::
       **Integer inputs only.**  Each staircase ramp is centred on
       ``k * divisor - 0.5`` — that's *between* two valid integer
       outputs, exactly where real-valued inputs near a bin boundary
       sit.  A float input like ``0.5`` lands directly inside the
       ramp zone ``[0.45, 0.55]`` and the staircase interpolates to
       ``~0.54`` instead of rounding cleanly.  If your input is a
       continuous float scalar (e.g. the output of a ``multiply`` or
       ``piecewise_linear``), use :func:`floor_int` instead — it
       places its ramps *at* integer boundaries so the flat zones
       cover the between-integer range where floats live.

    Args:
        inp: 1D scalar node with **integer** value in [0, max_value].
            Continuous-float inputs will interpolate junk inside the
            ramp zones — see warning.
        divisor: The divisor for floor division.
        max_value: Upper bound on input (determines number of steps).

    Returns:
        1D scalar node containing floor(inp / divisor).

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    n = max_value // divisor
    if n == 0:
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor([0.0]))

    # Build staircase breakpoints: each step is a steep ramp of width
    # 1/step_sharpness centred on the half-integer threshold.
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

    Uses the identity ``x % d = x - d * floor(x / d)``.

    Args:
        inp: 1D scalar node with integer value in [0, max_value].
        divisor: The constant divisor (positive integer).
        max_value: Upper bound on input.

    Returns:
        1D scalar node containing inp % divisor.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert divisor > 0, "divisor must be positive"
    q = thermometer_floor_div(inp, divisor, max_value)
    return subtract(inp, multiply_const(q, float(divisor)))


def clamp(inp: Node, lo: float, hi: float) -> Node:
    """Clamp a scalar to [lo, hi] in a single MLP sublayer.

    Uses :func:`piecewise_linear` with 4 breakpoints to implement an
    identity passthrough in [lo, hi] with sharp clamping at the edges.
    Much cheaper than a compare+select pair (1 MLP sublayer vs 6).

    Args:
        inp: 1D scalar node.
        lo: Lower bound.
        hi: Upper bound (must be > lo).

    Returns:
        1D scalar node clamped to [lo, hi].

    .. noise-footer::

       Max error: 1.907e-06 abs, 0.0002899 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert hi > lo, "hi must exceed lo"

    from torchwright.ops.const import step_sharpness

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
    range.  ``step`` controls breakpoint density: the number of
    breakpoints is ``(max_value - min_value) / step``, with a floor
    of 32 to guarantee reasonable accuracy.

    Args:
        inp: 1D scalar node with value in [min_value, max_value].
        min_value: Lower bound on input (must be > 0).
        max_value: Upper bound on input.
        step: Controls breakpoint density.  Smaller step = more
            breakpoints = higher accuracy.

    Returns:
        1D scalar node containing 1/x.

    .. noise-footer::

       Max error: 0.0008245 abs, 0.1361 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert min_value > 0, "min_value must be positive"
    assert max_value > min_value, "max_value must exceed min_value"

    n_breakpoints = builtins.max(int((max_value - min_value) / step) + 1, 32)
    ratio = (max_value / min_value) ** (1.0 / (n_breakpoints - 1))
    breakpoints = [min_value * (ratio**k) for k in range(n_breakpoints)]
    breakpoints[0] = min_value
    breakpoints[-1] = max_value

    return piecewise_linear(inp, breakpoints, lambda x: 1.0 / x, name="reciprocal")


def floor_int(
    inp: Node,
    min_value: int,
    max_value: int,
    sharpness: Optional[float] = None,
) -> Node:
    """Compute floor(x) for a continuous-valued scalar input.

    Places a steep ramp at each integer boundary ``k`` spanning
    ``[k - eps, k]`` (where ``eps = 1 / sharpness``).  The flat
    zone between ramps covers ``[k, k + 1 - eps]`` — the natural home
    of floating-point scalars like ``0.5`` or ``k + 0.3`` — so
    float inputs well inside a bin produce exact integer output.

    Use ``floor_int`` when the input is a *continuous* scalar
    (output of ``multiply``, ``piecewise_linear``, etc.) and you want
    its ``floor`` as an integer.  Use :func:`thermometer_floor_div`
    *only* when the input is already integer-valued with a
    compile-time-known bound — the two ops place their staircase
    ramps at different thresholds and aren't interchangeable.

    Both ops have a ``~eps``-wide ramp zone near each integer boundary
    where the output is an interpolated intermediate value.  If a
    caller is passing inputs that are specifically near integer
    boundaries, either clamp/round upstream, shift by ``0.5 - eps/2``
    to move the boundary into the flat zone, or raise ``sharpness``
    to narrow the ramp at the cost of MLP sublayer precision.

    Args:
        inp: 1D scalar node with value in [min_value, max_value].
        min_value: Lower bound (integer).
        max_value: Upper bound (integer).
        sharpness: Override the global ``step_sharpness`` for this op.
            Higher values narrow the ramp zone (e.g. 100 gives a
            0.01-wide ramp vs. the default 10 which gives 0.1).

    Returns:
        1D scalar node containing floor(x).

    .. noise-footer::

       Max error: 0.9993 abs, 0.953 rel over 8192 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_value >= min_value

    if max_value == min_value:
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor([float(min_value)]))

    # floor(x) = min_value + Σ_{k=min+1..max} step_k, with the unit step
    #   step_k = clamp(s·(x−k)+1, 0, 2)        (∈ [0, 2], = relu(t)−relu(t−2))
    # fully ON (==2) once x reaches the integer k (with a 1/s-wide ramp just
    # below), OFF (==0) below it. floor counts the steps with step_k < 1:
    #   floor = min + n − Σ_k relu(1 − step_k).
    # Two properties make this cancellation-free AND compiled-precise:
    #  * Each step_k is a 2-term per-output difference (NOT a sum over all
    #    breakpoints), so its fp32 error is bounded (~ulp of s·x), and the final
    #    Σ relu(1 − step_k) accumulates only bounded [0,1] terms — partial sums
    #    never exceed n. The old single-projection Σ ±s·ReLU(x−b_i) summed ~n
    #    terms of magnitude ~s·x; its partial sums overflowed float32's 2^24
    #    exact-integer limit and collapsed to ~0 at large magnitude/sharpness.
    #  * Clamping the step to [0, W] (not [0, 1]) keeps slack below the stage-2
    #    threshold of 1, so an ON step absorbs to relu(1 − W) = 0 *exactly* even
    #    with a few ulp of error, and the stage-1 intermediate stays small (≤ W,
    #    not ~s·x) so it does not amplify upstream compiled noise. W must exceed a
    #    few ulp(s·n): the per-step difference relu(t) − relu(t − W) collapses to
    #    0 once t and t − W round to the same float32 value (ulp(t) ≥ W), so we
    #    size W = max(2, 8·ulp(s·n)) — 2 for the common small-range case, larger
    #    only when s·n approaches 2^24.
    # Cost: one extra MLP sublayer (two chained ReLUs) vs the old form.
    import math as _math

    s = float(sharpness if sharpness is not None else step_sharpness)
    n = max_value - min_value
    _CHUNK = 512  # cap stage-1 hidden width (2 ReLUs per step) at ~1024

    max_t = s * n + 1.0
    ulp = 2.0 ** (_math.floor(_math.log2(max_t)) - 23) if max_t >= 1.0 else 2.0**-23
    step_cap = builtins.max(2.0, 8.0 * ulp)  # W

    neg_partials = []  # each chunk contributes −Σ_k relu(1 − step_k)
    for c0 in range(0, n, _CHUNK):
        ks = list(
            range(
                min_value + 1 + c0,
                builtins.min(min_value + 1 + c0 + _CHUNK, max_value + 1),
            )
        )
        c = len(ks)
        # stage 1 (MLP sublayer): step_k = relu(t_k) − relu(t_k − W)   width c
        # hidden = [relu(t_k), relu(t_k − W)] per step, t_k = s·x − s·k + 1.
        in_proj = torch.full((2 * c, 1), s)
        in_bias = torch.empty(2 * c)
        out_proj = torch.zeros((2 * c, c))
        for j, k in enumerate(ks):
            in_bias[2 * j] = 1.0 - s * k  # t_k
            in_bias[2 * j + 1] = 1.0 - step_cap - s * k  # t_k − W
            out_proj[2 * j, j] = 1.0
            out_proj[2 * j + 1, j] = -1.0
        step = linear_relu_linear(
            input_node=inp,
            input_proj=in_proj,
            input_bias=in_bias,
            output_proj=out_proj,
            output_bias=torch.zeros(c),
            name="floor_int_step",
        )
        # stage 2 (MLP sublayer): −Σ_k relu(1 − step_k)               width 1
        neg_partials.append(
            linear_relu_linear(
                input_node=step,
                input_proj=-torch.eye(c),
                input_bias=torch.ones(c),
                output_proj=-torch.ones((c, 1)),
                output_bias=torch.zeros(1),
                name="floor_int_saturate",
            )
        )

    summed = neg_partials[0] if len(neg_partials) == 1 else sum_nodes(neg_partials)
    result = add_const(summed, float(min_value + n))  # min + n − Σ relu(1 − step_k)
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=Range(float(min_value), float(max_value))),
    )


def ceil_int(
    inp: Node,
    min_value: int,
    max_value: int,
    sharpness: Optional[float] = None,
) -> Node:
    """Compute ceil(x) using the identity ``ceil(x) = -floor(-x)``.

    Args:
        inp: 1D scalar node with value in [min_value, max_value].
        min_value: Lower bound (integer).
        max_value: Upper bound (integer).
        sharpness: Override the global ``step_sharpness`` for the inner
            ``floor_int``. Higher values narrow the ramp zone at each
            integer boundary (e.g. 10000 gives a 1e-4-wide ramp vs. the
            default 10 which gives 0.1) so inputs carrying a few tenths of
            piecewise-linear noise stay in the flat zone and ceil exactly.
            Mirrors :func:`floor_int`'s ``sharpness``.

    Returns:
        1D scalar node containing ceil(x).

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    return negate(floor_int(negate(inp), -max_value, -min_value, sharpness=sharpness))


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def multiply_integers(
    inp1: Node,
    inp2: Node,
    max_value: int,
) -> Node:
    """Multiply two non-negative integer scalars using the polarization identity.

    a * b = ((a+b)² - (a-b)²) / 4

    Both inputs must be 1D scalar nodes with integer values in [0, max_value].
    The polarization identity is exact for all reals, but ``square`` is only
    exact at grid points (integers by default), so the inputs must be integers.

    Implementation:
        s = a + b                          range [0, 2*max_value], Linear (free)
        d = a - b                          range [-max_value, max_value], Linear (free)
        s² = square(s)                     1+ MLP sublayers
        d² = square(abs(d))                2 MLP sublayers (|d| = abs(d), then square)
        result = (s² - d²) / 4             Linear (free)

    Total cost: 3 MLP sublayers.

    Args:
        inp1: 1D scalar node, integer value in [0, max_value].
        inp2: 1D scalar node, integer value in [0, max_value].
        max_value: Upper bound on each input.

    Returns:
        1D scalar node containing inp1 * inp2.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit aafc5f3. See docs/numerical_noise.md.
    """
    assert len(inp1) == 1, "Input must be a 1D scalar node"
    assert len(inp2) == 1, "Input must be a 1D scalar node"

    s = add(inp1, inp2)  # a+b
    d = subtract(inp1, inp2)  # a-b (may be negative)

    sq_sum = square(s, 2 * max_value)  # (a+b)²
    sq_diff = square(abs(d), max_value)  # (a-b)²

    # a*b = ((a+b)² - (a-b)²) / 4
    return add_scaled_nodes(0.25, sq_sum, -0.25, sq_diff)
