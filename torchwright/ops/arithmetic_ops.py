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


def relu(inp: Node) -> Node:
    """
    Applies the Rectified Linear Unit (ReLU) function to the input node.

    Args:
        inp (Node): Node to apply ReLU.

    Returns:
        Node: Node with ReLU applied.
    """
    return ReLU(inp)


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
       measured at commit fe133f9. See docs/numerical_noise.md.
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
       measured at commit fe133f9. See docs/numerical_noise.md.
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
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp1) == len(inp2)
    diff = subtract(inp1, inp2)
    abs_diff = abs(diff)
    sum_ab = add(inp1, inp2)
    return add_scaled_nodes(0.5, sum_ab, -0.5, abs_diff)


def max(inp1: Node, inp2: Node) -> Node:
    """Element-wise maximum of two nodes.

    Computed as ``(a + b + |a - b|) / 2``.

    Args:
        inp1: First node.
        inp2: Second node (same width as *inp1*).

    Returns:
        Node of the same width containing ``max(inp1, inp2)`` element-wise.

    .. noise-footer::

       Max error: 3.815e-06 abs, 1.953e-06 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp1) == len(inp2)
    diff = subtract(inp1, inp2)
    abs_diff = abs(diff)
    sum_ab = add(inp1, inp2)
    return add_scaled_nodes(0.5, sum_ab, 0.5, abs_diff)


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
       measured at commit fe133f9. See docs/numerical_noise.md.
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


def piecewise_linear_2d(
    inp1: Node,
    inp2: Node,
    breakpoints1: List[float],
    breakpoints2: List[float],
    fn,
    d_max: int = 1024,
    name: str = "piecewise_linear_2d",
) -> Node:
    """Evaluate a piecewise-linear function of two scalar inputs.

    The callable *fn(x, y)* is evaluated at every grid vertex in
    *breakpoints1* × *breakpoints2* to obtain target values.  The
    result is a piecewise-linear interpolant on a triangulated
    rectangular grid (**1 MLP sublayer**).

    Outside the grid the output is clamped to the nearest edge/corner
    value (constant extrapolation).

    **Cost:** 1 MLP sublayer.  The number of neurons (hidden units) is
    approximately 2·n1·n2 for an n1 × n2 grid.  For a 30×10 grid this
    is ~600 neurons, well within typical ``d`` values.

    Args:
        inp1: 1D scalar node (first input variable).
        inp2: 1D scalar node (second input variable).
        breakpoints1: Strictly ascending x-coordinates (length n1 ≥ 2).
        breakpoints2: Strictly ascending y-coordinates (length n2 ≥ 2).
        fn: ``fn(x, y) -> float`` evaluated at each grid vertex.
        d_max: Maximum neurons per MLP sublayer.
        name: Debug label prefix.

    Returns:
        1D scalar node containing the interpolated value.

    .. noise-footer::

       Max error: 7.778 abs, 0.6024 rel over 8192 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp1) == 1, "inp1 must be a 1D scalar node"
    assert len(inp2) == 1, "inp2 must be a 1D scalar node"
    n1 = len(breakpoints1)
    n2 = len(breakpoints2)
    assert n1 >= 2 and n2 >= 2, "Need >= 2 breakpoints per axis"
    values = [
        [fn(breakpoints1[i], breakpoints2[j]) for j in range(n2)] for i in range(n1)
    ]
    assert all(
        breakpoints1[i] < breakpoints1[i + 1] for i in range(n1 - 1)
    ), "breakpoints1 must be strictly ascending"
    assert all(
        breakpoints2[i] < breakpoints2[i + 1] for i in range(n2 - 1)
    ), "breakpoints2 must be strictly ascending"

    inp = Concatenate([inp1, inp2])

    # ---------------------------------------------------------------
    # Strategy: solve for ReLU output weights via constrained
    # least-squares.
    #
    # Hyperplane family (4 directions through every grid vertex):
    #   1. Vertical:   x = x_i
    #   2. Horizontal: y = y_j
    #   3. Sum:        x + y = x_i + y_j
    #   4. Difference: x - y = x_i - y_j
    #
    # On non-uniform grids the sum/diff families expand to O(n1·n2)
    # distinct lines, making the system heavily underdetermined (more
    # hyperplanes than vertex constraints).  pinv's min-L2-norm
    # solution then packs large cancelling ReLU weights that agree at
    # vertices but oscillate across cell interiors.
    #
    # Fix: constrained least-squares.  Vertex values are enforced as
    # hard equality constraints (preserving exact interpolation at
    # grid points).  Interior sample points — fn evaluated at several
    # points per cell — provide a soft objective that pins down the
    # free DOF in the nullspace of the vertex system, eliminating the
    # oscillation.
    # ---------------------------------------------------------------

    seen: set = set()
    hyperplanes: list = []  # (a, b, c) tuples

    def _add(a: float, b: float, c: float) -> None:
        key = (round(a, 10), round(b, 10), round(c, 10))
        if key not in seen:
            seen.add(key)
            hyperplanes.append((a, b, c))

    for i in range(n1):
        _add(1.0, 0.0, -breakpoints1[i])
    for j in range(n2):
        _add(0.0, 1.0, -breakpoints2[j])

    for i in range(n1):
        for j in range(n2):
            _add(1.0, 1.0, -(breakpoints1[i] + breakpoints2[j]))
            _add(1.0, -1.0, -(breakpoints1[i] - breakpoints2[j]))

    K = len(hyperplanes)

    # -- Vectorized design-matrix builder (float64) --
    a_arr = torch.tensor([h[0] for h in hyperplanes], dtype=torch.float64)
    b_arr = torch.tensor([h[1] for h in hyperplanes], dtype=torch.float64)
    c_arr = torch.tensor([h[2] for h in hyperplanes], dtype=torch.float64)

    def _design(xs: list, ys: list) -> torch.Tensor:
        xt = torch.tensor(xs, dtype=torch.float64)
        yt = torch.tensor(ys, dtype=torch.float64)
        M = torch.zeros(len(xs), 3 + K, dtype=torch.float64)
        M[:, 0] = 1.0
        M[:, 1] = xt
        M[:, 2] = yt
        M[:, 3:] = torch.clamp(
            a_arr.unsqueeze(0) * xt.unsqueeze(1)
            + b_arr.unsqueeze(0) * yt.unsqueeze(1)
            + c_arr.unsqueeze(0),
            min=0.0,
        )
        return M

    # Vertex constraints (hard equality).
    xs_v = [breakpoints1[i] for i in range(n1) for _ in range(n2)]
    ys_v = [breakpoints2[j] for _ in range(n1) for j in range(n2)]
    bv = torch.tensor(
        [values[i][j] for i in range(n1) for j in range(n2)],
        dtype=torch.float64,
    )
    A_v = _design(xs_v, ys_v)

    # Interior sample constraints (soft, resolved in nullspace).
    # 4 points per cell, spread to cover both triangles of the SW-NE
    # triangulation.  fn is evaluated at these points to get the true
    # target — this is what pins down the nullspace DOF.
    _OFFSETS = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    xs_int: list = []
    ys_int: list = []
    vals_int: list = []
    for i in range(n1 - 1):
        for j in range(n2 - 1):
            xi, xi1 = breakpoints1[i], breakpoints1[i + 1]
            yj, yj1 = breakpoints2[j], breakpoints2[j + 1]
            for u, v in _OFFSETS:
                x = xi + u * (xi1 - xi)
                y = yj + v * (yj1 - yj)
                xs_int.append(x)
                ys_int.append(y)
                vals_int.append(float(fn(x, y)))

    # Constrained least-squares via nullspace parameterization:
    #   x_part = vertex-exact particular solution (min-norm)
    #   N_basis = nullspace of A_v
    #   z = argmin ||A_int · (x_part + N_basis · z) - b_int||²
    #   solution = x_part + N_basis · z
    x_part = torch.linalg.pinv(A_v) @ bv

    if xs_int:
        # We need Vh's rows up to the column dimension (3+K) to span the
        # nullspace basis N_basis = Vh[rank:].T below.
        #
        # When A_v is TALL (rows n1*n2 >= cols 3+K — uniform / collapsed
        # grids, e.g. the unit grid multiply_2d used to feed here), the
        # reduced SVD already returns all 3+K right-singular vectors, so
        # full_matrices=False yields a bit-identical Vh while avoiding the
        # (n1*n2, n1*n2) left-singular matrix U that full_matrices=True
        # materializes and we discard — that U is ~35 GB of float64 at
        # n1*n2 ~ 66049 and is the build-time OOM this guard removes.
        #
        # When A_v is WIDE (rows < cols — non-uniform grids, where the
        # sum/diff hyperplane families do not collapse), the reduced SVD
        # returns only the first `rows` right-singular vectors and the
        # nullspace rows rank..(3+K) would be missing.  There we must keep
        # full_matrices=True so N_basis spans the full nullspace; the U it
        # builds is only (rows, rows) and the grids that hit this path are
        # small.
        tall = A_v.shape[0] >= A_v.shape[1]
        _U, S, Vh = torch.linalg.svd(A_v, full_matrices=not tall)
        tol = S.max().item() * 1e-10 if S.numel() else 0.0
        rank = int((S > tol).sum().item())
        N_basis = Vh[rank:].T  # (3+K, nulldim)

        if N_basis.shape[1] > 0:
            A_int = _design(xs_int, ys_int)
            b_int = torch.tensor(vals_int, dtype=torch.float64)
            A_red = A_int @ N_basis
            b_red = b_int - A_int @ x_part
            # Tikhonov-regularized solve: penalize large z to prevent
            # the nullspace solve from exploding when A_red is
            # ill-conditioned (some nullspace directions are nearly
            # invisible to the interior samples).
            AtA = A_red.T @ A_red
            lam = AtA.diag().max().item() * 1e-6
            z = torch.linalg.solve(
                AtA + lam * torch.eye(AtA.shape[0], dtype=torch.float64),
                A_red.T @ b_red,
            )
            solution = (x_part + N_basis @ z).float()
        else:
            solution = x_part.float()
    else:
        solution = x_part.float()

    bias_val = solution[0].item()
    base_sx = solution[1].item()
    base_sy = solution[2].item()
    weights = solution[3:]

    # Filter out near-zero neurons
    active = []
    for k in range(K):
        if _builtin_abs(weights[k].item()) > 1e-10:
            active.append((hyperplanes[k], weights[k].item()))

    if (
        len(active) == 0
        and _builtin_abs(base_sx) < 1e-10
        and _builtin_abs(base_sy) < 1e-10
    ):
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor([bias_val]))

    # Build L -> ReLU -> L weight matrices, chunking across sublayers if
    # the active neuron count exceeds d_max.  Each chunk is an independent
    # linear_relu_linear that contributes a partial sum of ReLU terms; the
    # constant bias_val is carried by the first chunk only.
    if not active:
        # Degenerate placeholder: bias_val needs a carrier, but all ReLU
        # weights are zero.  One dead neuron keeps the shape valid.
        input_proj = torch.zeros(1, 2)
        input_bias = torch.zeros(1)
        output_proj = torch.zeros(1, 1)
        result = linear_relu_linear(
            input_node=inp,
            input_proj=input_proj,
            input_bias=input_bias,
            output_proj=output_proj,
            output_bias=torch.tensor([bias_val]),
            name=name,
        )
    else:
        chunks = []
        multi = len(active) > d_max
        for chunk_start in range(0, len(active), d_max):
            chunk = active[chunk_start : chunk_start + d_max]
            d = len(chunk)
            input_proj = torch.zeros(d, 2)
            input_bias = torch.zeros(d)
            output_proj = torch.zeros(d, 1)
            for k, ((a, b, c), w) in enumerate(chunk):
                input_proj[k, 0] = a
                input_proj[k, 1] = b
                input_bias[k] = c
                output_proj[k, 0] = w
            ob = torch.tensor([bias_val]) if chunk_start == 0 else torch.zeros(1)
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
        result = chunks[0] if len(chunks) == 1 else sum_nodes(chunks)

    # Add base linear term (sx*x + sy*y) — free (Linear node).
    if _builtin_abs(base_sx) > 1e-10 or _builtin_abs(base_sy) > 1e-10:
        base_weight = torch.tensor([[base_sx], [base_sy]])
        base_linear = Linear(inp, base_weight, name=f"{name}_base")
        result = Add(base_linear, result)

    # Declare the tight output range.  Inside the grid the piecewise-
    # linear interpolation is bounded by the function's vertex min/max.
    # A non-zero base linear term (base_sx*x + base_sy*y) extrapolates
    # linearly outside the grid — but grid_lo/grid_hi already include
    # the base linear contribution at grid vertices, so only the input
    # range that extends BEYOND the grid boundaries adds extra range.
    grid_vals = [v for row in values for v in row]
    grid_lo = builtins.min(grid_vals)
    grid_hi = builtins.max(grid_vals)
    base_lo = 0.0
    base_hi = 0.0
    can_tighten = True
    if _builtin_abs(base_sx) > 1e-10:
        r1 = inp1.value_type.value_range
        if not r1.is_finite():
            can_tighten = False
        else:
            ext_below = builtins.min(r1.lo - breakpoints1[0], 0.0)
            ext_above = builtins.max(r1.hi - breakpoints1[-1], 0.0)
            cand = (base_sx * ext_below, base_sx * ext_above)
            base_lo += builtins.min(cand)
            base_hi += builtins.max(cand)
    if can_tighten and _builtin_abs(base_sy) > 1e-10:
        r2 = inp2.value_type.value_range
        if not r2.is_finite():
            can_tighten = False
        else:
            ext_below = builtins.min(r2.lo - breakpoints2[0], 0.0)
            ext_above = builtins.max(r2.hi - breakpoints2[-1], 0.0)
            cand = (base_sy * ext_below, base_sy * ext_above)
            base_lo += builtins.min(cand)
            base_hi += builtins.max(cand)
    if can_tighten:
        result = assert_matches_value_type(
            result,
            NodeValueType(value_range=Range(grid_lo + base_lo, grid_hi + base_hi)),
        )
    return result


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
    min1: Optional[float] = None,
    min2: Optional[float] = None,
    max_abs_output: Optional[float] = None,
    d_max: int = 1024,
    name: str = "multiply_2d",
) -> Node:
    """Multiply two signed scalars via a 2D piecewise-linear lookup.

    Computes ``inp1 * inp2`` in a **single MLP sublayer** by tabulating
    the product on a 2D grid and delegating to :func:`piecewise_linear_2d`.
    This trades MLP width for depth: ``signed_multiply`` uses ~3 MLP
    sublayers but few neurons; ``multiply_2d`` uses 1 sublayer but
    ~2*n1*n2 neurons (non-uniform grids) or ~3*(n1+n2) neurons (uniform).

    **When to prefer over** ``signed_multiply``:

    * The pipeline is depth-bound (most layers have spare MLP slots).
    * Input ranges are moderate — the grid precision ``step1 * step2 / 4``
      must be acceptable for the downstream consumer.  Feeding the output
      into ``floor_int`` requires ``step1 * step2 < 2`` to avoid bin errors.
    * You need one multiplication per layer rather than a 3-layer chain.

    **Breakpoint generation.** When ``breakpoints1`` / ``breakpoints2`` are
    not provided, uniform spacing from ``min1`` to ``max_abs1`` (and
    similarly for axis 2) at ``step`` intervals is used.  ``min1`` defaults
    to ``-max_abs1``; setting ``min1=0`` halves the breakpoints for
    inputs known to be non-negative (e.g. ``inv_range``).  Custom
    breakpoints (like the non-uniform ``_DIFF_BP`` used elsewhere in the
    renderer) can be passed directly.

    Args:
        inp1: 1D scalar node.
        inp2: 1D scalar node.
        max_abs1: Upper bound on ``|inp1|``.
        max_abs2: Upper bound on ``|inp2|``.
        step1: Grid spacing for auto-generated breakpoints on axis 1.
        step2: Grid spacing for auto-generated breakpoints on axis 2.
        breakpoints1: Explicit breakpoints for axis 1.  Overrides
            ``max_abs1`` / ``step1`` / ``min1``.
        breakpoints2: Explicit breakpoints for axis 2.  Overrides
            ``max_abs2`` / ``step2`` / ``min2``.
        min1: Lower bound for auto-generated axis-1 breakpoints.
            Defaults to ``-max_abs1``.
        min2: Lower bound for auto-generated axis-2 breakpoints.
            Defaults to ``-max_abs2``.
        max_abs_output: If set, clamp the product to
            ``[-max_abs_output, max_abs_output]`` via an extra
            :func:`clamp` node (1 additional MLP sublayer).
        d_max: Maximum neurons in the underlying MLP sublayer.
        name: Node name for debugging.

    Returns:
        1D scalar node containing ``inp1 * inp2`` (optionally clamped).

    .. noise-footer::

       Max error: 0.06249 abs, 2.15 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp1) == 1, "inp1 must be a 1D scalar node"
    assert len(inp2) == 1, "inp2 must be a 1D scalar node"

    # --- Build breakpoints ---
    if breakpoints1 is None:
        lo1 = -max_abs1 if min1 is None else min1
        n1 = builtins.max(int(round((max_abs1 - lo1) / step1)) + 1, 2)
        breakpoints1 = [lo1 + i * step1 for i in range(n1)]
        breakpoints1[-1] = max_abs1  # pin endpoint
    if breakpoints2 is None:
        lo2 = -max_abs2 if min2 is None else min2
        n2 = builtins.max(int(round((max_abs2 - lo2) / step2)) + 1, 2)
        breakpoints2 = [lo2 + i * step2 for i in range(n2)]
        breakpoints2[-1] = max_abs2  # pin endpoint

    # --- Normalize both axes to [0, 1] with a common step ---
    #
    # piecewise_linear_2d places ReLU hyperplanes at x+y=const and
    # x−y=const through every grid vertex.  On a square grid (equal step
    # on both axes), these collapse to O(n) distinct lines.  When the
    # steps differ the sums/differences are all distinct → O(n²)
    # hyperplanes, which can exceed the MLP width budget.
    #
    # Affine-mapping both axes to [0, 1] with a common breakpoint count
    # guarantees equal step → diagonal collapse → O(n) hyperplanes.
    # The mapping is done by two free Linear nodes; the product
    # fn(u, v) = (lo₁ + range₁·u)·(lo₂ + range₂·v) is passed to
    # piecewise_linear_2d so the output is mathematically identical.
    lo1_f = float(breakpoints1[0])
    hi1_f = float(breakpoints1[-1])
    lo2_f = float(breakpoints2[0])
    hi2_f = float(breakpoints2[-1])
    range1 = hi1_f - lo1_f
    range2 = hi2_f - lo2_f

    if range1 < 1e-12 or range2 < 1e-12:
        result = piecewise_linear_2d(
            inp1,
            inp2,
            breakpoints1,
            breakpoints2,
            lambda a, b: a * b,
            d_max=d_max,
            name=name,
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

    if max_abs_output is not None:
        result = clamp(result, -max_abs_output, max_abs_output)

    return result


def low_rank_2d(
    inp1: Node,
    inp2: Node,
    breakpoints1: List[float],
    breakpoints2: List[float],
    fn,
    rank: int,
    multiply_steps_per_axis: int = 20,
    max_abs_output: Optional[float] = None,
    d_max: int = 1024,
    name: str = "low_rank_2d",
) -> Node:
    """Evaluate a 2D function via rank-K separable approximation.

    Samples ``fn`` at every grid vertex, SVD-truncates the value matrix
    to rank *K*, and emits the approximation as a sum of ``K`` separable
    rank-1 terms::

        f(x, y) ≈ Σ_{k=1..K} U_k(x) · V_k(y)

    where ``U_k`` and ``V_k`` are 1-D piecewise-linear interpolants of
    the scaled left/right singular vectors.  The output is the
    SVD-optimal rank-K fit in Frobenius norm — in particular, the
    worst-cell error is bounded above by ``σ_{K+1}`` (the first
    truncated singular value), which is deterministic and computable at
    compile time.

    **When to prefer over** :func:`piecewise_linear_2d`:

    * The grid is non-uniform (``piecewise_linear_2d``'s least-squares
      solve oscillates in cell interiors on non-uniform grids).
    * The function has low effective rank.  Products ``x·y`` are rank-1
      exactly, so K=1 is lossless.  Smooth functions like ``atan(x/y)``
      typically need K=2–3 for ~1% precision.
    * You want a compile-time error bound that isn't dependent on the
      pinv condition number.

    **Cost:** 2 MLP sublayers:

    * Sublayer 1: two 1D piecewise-linear lookups (one per axis), each
      with vector-valued output of width *K*.  ``~(n1 + n2)`` neurons
      total, independent of *K* (channels share ReLUs).
    * Sublayer 2: *K* scalar multiplications via :func:`multiply_2d` on
      uniform bounded grids (where the pinv issue doesn't bite).

    Args:
        inp1: 1-D scalar node.
        inp2: 1-D scalar node.
        breakpoints1: Strictly ascending x-coordinates (length n1 ≥ 2).
        breakpoints2: Strictly ascending y-coordinates (length n2 ≥ 2).
        fn: ``fn(x, y) -> float`` evaluated at each grid vertex.
        rank: Number of separable terms *K* in the decomposition.
            Clamped to ``min(n1, n2)``.  Compile time scales linearly
            with *K*; pick the smallest *K* that meets your tolerance.
        multiply_steps_per_axis: Grid resolution of each inner
            :func:`multiply_2d` call (uniform breakpoints).  20 gives
            typical ~1% multiply precision.
        max_abs_output: If set, clamp the sum to
            ``[−max_abs_output, max_abs_output]`` (one extra sublayer).
        d_max: Maximum neurons per MLP sublayer.
        name: Debug label prefix.

    Returns:
        1-D scalar node containing the rank-K approximation.

    .. noise-footer::

       Max error: 0.0651 abs, 0.3315 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp1) == 1, "inp1 must be a 1D scalar node"
    assert len(inp2) == 1, "inp2 must be a 1D scalar node"
    n1 = len(breakpoints1)
    n2 = len(breakpoints2)
    assert n1 >= 2 and n2 >= 2, "Need >= 2 breakpoints per axis"
    assert rank >= 1, "rank must be >= 1"
    assert all(
        breakpoints1[i] < breakpoints1[i + 1] for i in range(n1 - 1)
    ), "breakpoints1 must be strictly ascending"
    assert all(
        breakpoints2[i] < breakpoints2[i + 1] for i in range(n2 - 1)
    ), "breakpoints2 must be strictly ascending"

    K = builtins.min(rank, builtins.min(n1, n2))

    V = torch.zeros(n1, n2, dtype=torch.float64)
    for i, xi in enumerate(breakpoints1):
        for j, yj in enumerate(breakpoints2):
            V[i, j] = float(fn(xi, yj))

    U_full, S_full, Vh_full = torch.linalg.svd(V, full_matrices=False)

    U_k = U_full[:, :K]  # (n1, K)
    S_k = S_full[:K]
    Vh_k = Vh_full[:K, :]  # (K, n2)

    # Absorb √σ into each factor so both U_scaled and V_scaled have
    # comparable magnitudes — makes the downstream multiply grid easy
    # to bound.
    sqrt_S = torch.sqrt(S_k)
    U_scaled = (U_k * sqrt_S.unsqueeze(0)).tolist()  # (n1, K)
    V_scaled = (Vh_k * sqrt_S.unsqueeze(1)).tolist()  # (K, n2)

    # Map breakpoint → vector of K component values for piecewise_linear's
    # vector-fn interface.  Float-equality dict lookup works because
    # piecewise_linear iterates the exact same breakpoint floats.
    u_by_bp = {breakpoints1[i]: U_scaled[i] for i in range(n1)}
    v_by_bp = {breakpoints2[j]: [V_scaled[k][j] for k in range(K)] for j in range(n2)}

    u_vec = piecewise_linear(
        inp1,
        breakpoints1,
        lambda x: u_by_bp[x],
        d_max=d_max,
        name=f"{name}_u_vec",
    )
    v_vec = piecewise_linear(
        inp2,
        breakpoints2,
        lambda y: v_by_bp[y],
        d_max=d_max,
        name=f"{name}_v_vec",
    )

    # Per-component amplitude bounds (used to size the multiply grid).
    u_abs_max = [
        builtins.max(_builtin_abs(U_scaled[i][k]) for i in range(n1)) for k in range(K)
    ]
    v_abs_max = [
        builtins.max(_builtin_abs(V_scaled[k][j]) for j in range(n2)) for k in range(K)
    ]

    products: list = []
    for k in range(K):
        proj_u = torch.zeros(K, 1)
        proj_u[k, 0] = 1.0
        u_k = Linear(u_vec, proj_u, name=f"{name}_u{k}")

        proj_v = torch.zeros(K, 1)
        proj_v[k, 0] = 1.0
        v_k = Linear(v_vec, proj_v, name=f"{name}_v{k}")

        # Pad 5% around the empirical max so the input lands strictly
        # inside the multiply's interpolation grid.  Floor at 1e-6 so
        # a degenerate all-zero component doesn't produce step=0.
        u_bound = builtins.max(u_abs_max[k] * 1.05, 1e-6)
        v_bound = builtins.max(v_abs_max[k] * 1.05, 1e-6)
        step_u = u_bound / builtins.max(multiply_steps_per_axis // 2, 1)
        step_v = v_bound / builtins.max(multiply_steps_per_axis // 2, 1)

        products.append(
            multiply_2d(
                u_k,
                v_k,
                max_abs1=u_bound,
                max_abs2=v_bound,
                step1=step_u,
                step2=step_v,
                d_max=d_max,
                name=f"{name}_prod{k}",
            )
        )

    result = products[0] if len(products) == 1 else sum_nodes(products)

    if max_abs_output is not None:
        result = clamp(result, -max_abs_output, max_abs_output)

    return result


def square(inp: Node, max_value: float, step: float = 1.0, d_max: int = 1024) -> Node:
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
        d_max: Maximum neurons per MLP sublayer. When more breakpoints
            are needed, they are split into chunks of this size.

    Returns:
        1D scalar node containing x² (exact at grid points).

    .. noise-footer::

       Max error: 0.25 abs, 231.1 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_value > 0, "max_value must be positive"

    breakpoints = []
    x = 0.0
    while x <= max_value + step / 2.0:
        breakpoints.append(x)
        x += step

    return piecewise_linear(
        inp, breakpoints, lambda x: x * x, d_max=d_max, name="square"
    )


def square_signed(
    inp: Node,
    max_abs: float,
    step: float = 1.0,
    d_max: int = 1024,
) -> Node:
    """Compute x² for signed inputs via piecewise-linear interpolation.

    Unlike :func:`square` (which only handles non-negative inputs),
    this handles x in [-max_abs, max_abs] directly — no ``abs`` needed.
    This saves one MLP sublayer when used inside :func:`signed_multiply`.

    Args:
        inp: 1D scalar node with value in [-max_abs, max_abs].
        max_abs: Maximum absolute value of input.
        step: Grid spacing.
        d_max: Maximum neurons per MLP sublayer.

    Returns:
        1D scalar node containing x².

    .. noise-footer::

       Max error: 0.25 abs, 276.7 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_abs > 0, "max_abs must be positive"

    breakpoints = []
    x = -max_abs
    while x <= max_abs + step / 2.0:
        breakpoints.append(x)
        x += step

    return piecewise_linear(
        inp, breakpoints, lambda x: x * x, d_max=d_max, name="square_signed"
    )


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
       measured at commit fe133f9. See docs/numerical_noise.md.
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
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert divisor > 0, "divisor must be positive"
    q = thermometer_floor_div(inp, divisor, max_value)
    return subtract(inp, multiply_const(q, float(divisor)))


def linear_bin_index(
    x: Node,
    x_min: Node,
    x_max: Node,
    n_bins: int,
    min_range: float = 0.5,
    max_range: float = 200.0,
    n_reciprocal_breakpoints: int = 32,
    mul_step: float = 0.5,
    name: str = "linear_bin_index",
    inv_range: Optional[Node] = None,
) -> Node:
    """Map a continuous coordinate onto an integer bin index.

    Computes ::

        bin = clamp(floor((x - x_min) * n_bins / (x_max - x_min)),
                    0, n_bins - 1)

    for runtime scalars ``x``, ``x_min``, ``x_max`` and a compile-time
    ``n_bins``.  This is the "continuous → discrete bin" side of
    resampling: paired with :func:`dynamic_extract` it gives texture
    sampling, histogram bucketing, dispatch tables, and every other
    "which of ``n_bins`` things am I looking at right now" query.

    Decomposition (and why each piece exists):

    1. ``range_ = x_max - x_min`` — free Linear.
    2. ``clamp(range_, min_range, max_range)`` — a runtime range that
       drops to zero would blow up the reciprocal.  Clamping bounds the
       reciprocal at ``1/min_range`` instead.  Costs 1 MLP sublayer.
    3. ``inv_range = 1/clamped_range`` — **geometric** breakpoints, not
       linear, because ``1/x`` has steep gradient near small ``x`` and
       geometric spacing gives constant *relative* error across the
       whole range.  Costs 1 MLP sublayer with ~``n_reciprocal_breakpoints``
       neurons.
    4. ``delta = x - x_min`` — free Linear.
    5. ``clamped_delta = clamp(delta, -max_range, max_range)`` — keeps
       the multiplication within its declared input bounds even when
       the caller sends an ``x`` far outside ``[x_min, x_max]``.
    6. ``normalized = signed_multiply(clamped_delta, inv_range,
       max_abs_output=max_range/min_range)`` — the one non-trivial cost,
       ~3 MLP sublayers.  Bounded via ``max_abs_output`` so downstream
       ops see a defined output range.
    7. ``bin_f = multiply_const(normalized, float(n_bins))`` — free.
    8. ``clamp(bin_f, 0, n_bins - 0.5)`` — clamp to the valid floor
       domain.  Callers who pass out-of-range ``x`` land on the nearest
       endpoint bin instead of wandering off into the staircase's
       extrapolation zone.
    9. ``thermometer_floor_div(..., divisor=1, max_value=n_bins-1)`` —
       the actual floor-to-integer step.

    Total cost: ~6 MLP sublayers at step sharpness ~1.  Depth is
    dominated by the ``signed_multiply``.  The primitive is designed
    for "call once per query"; if a caller needs many bin indices over
    the same ``(x_min, x_max)`` with different ``x`` values, the cheap
    path is to hoist ``inv_range`` out and pass it via the ``inv_range``
    parameter, saving 2 MLP sublayers per call::

        # Hoist the shared computation:
        range_ = subtract(x_max, x_min)
        clamped = clamp(range_, min_range, max_range)
        inv = reciprocal(clamped, min_value=min_range, max_value=max_range)

        for y_idx in range(rows_per_patch):
            idx = linear_bin_index(y, x_min, x_max, n_bins,
                                   min_range=min_range, max_range=max_range,
                                   inv_range=inv)

    Args:
        x: Scalar node — continuous coordinate to bin.
        x_min: Scalar node — lower edge of the value range.
        x_max: Scalar node — upper edge.  Must satisfy
            ``x_max - x_min >= min_range`` for correct output; smaller
            ranges are clamped to ``min_range`` before the reciprocal.
            Ignored (but still required for API stability) when
            ``inv_range`` is provided.
        n_bins: Number of discrete bins (compile-time).  Output is an
            integer in ``[0, n_bins - 1]``.
        min_range: Smallest representable value of ``x_max - x_min``.
            Smaller values need more reciprocal breakpoints to stay
            accurate.  Also sets the ``signed_multiply`` bound
            ``max_abs2 = 1/min_range``, so it is required even when
            ``inv_range`` is provided.
        max_range: Largest representable value of ``x_max - x_min``.
            Also sets the ``signed_multiply`` bound ``max_abs1 = max_range``
            and the delta clamp, so it is required even when ``inv_range``
            is provided.
        n_reciprocal_breakpoints: Number of geometrically-spaced
            breakpoints used by the internal ``1/range`` lookup.  More
            breakpoints → tighter relative error on the reciprocal;
            typically ``log(max_range/min_range) / log(1 + tolerance)``.
            Default 32 gives ≲1% relative error over a 400× range.
            Ignored when ``inv_range`` is provided.
        mul_step: Grid spacing passed to the internal ``signed_multiply``
            for the ``delta × inv_range`` product.
        inv_range: Pre-computed ``1 / clamp(x_max - x_min, min_range,
            max_range)`` node.  When provided, steps 1-3 above are
            skipped and this node is used directly in the multiplication.
            The caller is responsible for computing this with adequate
            precision (see the hoisting example above).

    Returns:
        Scalar node carrying an integer in ``[0, n_bins - 1]``.

    .. noise-footer::

       Max error: 1 abs, 0.9854 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(x) == 1, "x must be a 1D scalar node"
    assert len(x_min) == 1, "x_min must be a 1D scalar node"
    assert len(x_max) == 1, "x_max must be a 1D scalar node"
    assert n_bins >= 1, "n_bins must be at least 1"
    assert (
        0 < min_range < max_range
    ), "need 0 < min_range < max_range for the reciprocal lookup"

    if inv_range is not None:
        # Caller pre-computed 1/clamped_range — skip steps 1-3.
        assert len(inv_range) == 1, "inv_range must be a 1D scalar node"
    else:
        assert (
            n_reciprocal_breakpoints >= 2
        ), "need at least 2 breakpoints for the geometric reciprocal lookup"

        # 1. range and its clamp.
        range_ = subtract(x_max, x_min)
        clamped_range = clamp(range_, min_range, max_range)

        # 2. 1/range via a geometric breakpoint lookup.  Geometric spacing
        #    gives constant relative error per segment: error_rel ≈ (r-1)²/4
        #    where r is the per-step ratio.  For 32 breakpoints over a
        #    400× range, r ≈ 1.22 and error_rel ≈ 1.2%.
        ratio = (max_range / min_range) ** (1.0 / (n_reciprocal_breakpoints - 1))
        bps: List[float] = [
            min_range * (ratio**k) for k in range(n_reciprocal_breakpoints)
        ]
        # Pin the endpoints so float rounding can't drift them.
        bps[0] = min_range
        bps[-1] = max_range
        # The breakpoints must be strictly ascending — trivially true for
        # geometric spacing but assert in case min_range == max_range sneaks
        # through numerically.
        assert all(
            bps[i] < bps[i + 1] for i in range(len(bps) - 1)
        ), "geometric breakpoints collapsed — check min_range/max_range"
        inv_range = piecewise_linear(
            clamped_range,
            bps,
            lambda r: 1.0 / r,
            name=f"{name}_inv_range",
        )

    # 3. delta and its clamp.  Pre-clamping keeps the multiplication
    #    inputs inside the declared bounds even under adversarial
    #    caller behaviour.
    delta = subtract(x, x_min)
    clamped_delta = clamp(delta, -max_range, max_range)

    # 4. normalized = delta × (1/range).  Signed because delta may be
    #    negative when x < x_min.
    max_abs_normalized = max_range / min_range
    normalized = signed_multiply(
        clamped_delta,
        inv_range,
        max_abs1=max_range,
        max_abs2=1.0 / min_range,
        max_abs_output=max_abs_normalized,
        step=mul_step,
    )

    # 5. Scale by n_bins (free Linear), clamp to the valid floor domain,
    #    then staircase-floor.  The (n_bins - 0.5) upper clamp ensures
    #    that an x at or past x_max lands on bin (n_bins - 1) rather
    #    than the non-existent bin n_bins.  We use ``floor_int`` rather
    #    than ``thermometer_floor_div`` because ``bin_f`` is a continuous
    #    float that rarely lands on an integer — ``floor_int`` places
    #    its staircase ramps AT integer boundaries (leaving the flat
    #    between-integer region clean), whereas ``thermometer_floor_div``
    #    places them at ``k - 0.5`` which is designed for integer
    #    inputs and produces interpolated junk on non-integer floats.
    bin_f = multiply_const(normalized, float(n_bins))
    clamped_bin_f = clamp(bin_f, 0.0, float(n_bins) - 0.5)
    return floor_int(clamped_bin_f, min_value=0, max_value=n_bins - 1)


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
       measured at commit fe133f9. See docs/numerical_noise.md.
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
    d_max: int = 1024,
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
        d_max: Maximum neurons per MLP sublayer.

    Returns:
        1D scalar node containing 1/x.

    .. noise-footer::

       Max error: 0.0009137 abs, 0.1686 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert min_value > 0, "min_value must be positive"
    assert max_value > min_value, "max_value must exceed min_value"

    n_breakpoints = builtins.max(int((max_value - min_value) / step) + 1, 32)
    ratio = (max_value / min_value) ** (1.0 / (n_breakpoints - 1))
    breakpoints = [min_value * (ratio**k) for k in range(n_breakpoints)]
    breakpoints[0] = min_value
    breakpoints[-1] = max_value

    return piecewise_linear(
        inp, breakpoints, lambda x: 1.0 / x, d_max=d_max, name="reciprocal"
    )


def _log_single(
    inp: Node,
    min_value: float,
    max_value: float,
    n_breakpoints: int,
    d_max: int,
    name: str = "log_section",
) -> Node:
    """Single-section piecewise-linear log over ``[min_value, max_value]``.

    Geometric breakpoint spacing. Used both as the fallback path of
    :func:`log` (for ranges that fit in one section) and as each
    section's interior approximation in the sectioned path.
    """
    import math as _math

    ratio = (max_value / min_value) ** (1.0 / (n_breakpoints - 1))
    breakpoints = [min_value * (ratio**k) for k in range(n_breakpoints)]
    breakpoints[0] = min_value
    breakpoints[-1] = max_value
    return piecewise_linear(
        inp, breakpoints, lambda x: _math.log(x), d_max=d_max, name=name
    )


_LOG_BOUNDARY_SHARPNESS = 10.0
"""Step-sharpness for :func:`log`'s section-routing compares.

Trade-off here is between **ramp width** (``1/sharpness`` in input
units, drives the section-overlap requirement) and **compare-output
cancellation noise** (``sharpness · x_max · 2⁻²³``, since the compare
realises a 0→1 step as ``ReLU(s·(x-t)) - ReLU(s·(x-t) - 1)`` whose
two terms grow with ``s·x_max``).

At ``sharpness=10`` and ``x_max≈3·10⁴``, cancellation magnitude is
``3·10⁵`` so output noise is ``~0.04`` absolute — manageable with
the widened ``_LOG_COMPARE_ATOL`` below. Higher sharpness narrows
the ramp but explodes cancellation noise; lower sharpness widens
the ramp past where multiply_2d blending stays accurate.
"""

_LOG_COMPARE_ATOL = 0.1
"""Tolerance on ``_log_compare_01``'s value-range assertion.

Float32 cancellation in ``ReLU(s·(x-t)) - ReLU(s·(x-t) - 1)`` at
extreme ``x`` can push the output as far as ``±0.07`` outside the
nominal ``[0, 1]`` range. ``0.1`` gives a safety margin without
masking real bugs.
"""


def _log_compare_01(inp: Node, thresh: float) -> Node:
    """0/1 thermometer compare for :func:`log`'s section routing.

    Inline ``linear_relu_linear`` (rather than calling :func:`compare`)
    so we can attach a widened value-range assertion that accommodates
    the float32 cancellation noise at extreme ``x``. Same structure
    as ``compare``: two ReLUs realise a 0→1 ramp of width
    ``1/_LOG_BOUNDARY_SHARPNESS`` starting at ``thresh``.
    """
    s = _LOG_BOUNDARY_SHARPNESS
    input_proj = torch.tensor([[s], [s]])
    input_bias = torch.tensor([-s * float(thresh), -s * float(thresh) - 1.0])
    output_proj = torch.tensor([[1.0], [-1.0]])
    output_bias = torch.tensor([0.0])
    result = linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
        name="log_cmp",
    )
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=Range(0.0, 1.0)),
        atol=_LOG_COMPARE_ATOL,
    )


def log(
    inp: Node,
    min_value: float,
    max_value: float,
    n_breakpoints: int = 256,
    section_factor: float = 10.0,
    d_max: int = 1024,
) -> Node:
    """Natural log of a positive scalar via per-section piecewise-linear.

    The naïve single-section piecewise-linear log accumulates
    ``slope[0] · x_max ≈ x_max / x_min`` in pre-cancellation magnitude
    at the right end of its range, hitting a float32 cancellation
    floor of ``(x_max/x_min) · 2⁻²³``. For ``x_max/x_min > ~10⁴`` this
    exceeds typical noise budgets — and adding more breakpoints makes
    it *worse* by lengthening the cancellation chain.

    This implementation sections the input range geometrically by
    ``section_factor`` (default decades), computes one piecewise-
    linear log per section in parallel, and routes via thermometer
    compare plus :func:`multiply_2d` blending. Each section's pre-
    cancellation magnitude is bounded by its own ``B_{i+1}/B_i =
    section_factor``, so the per-section precision floor is
    ``section_factor · 2⁻²³`` — independent of the overall range.

    **Routing detail.** Section ``i``'s piecewise-linear is extended
    by ``1/_LOG_BOUNDARY_SHARPNESS`` past its right boundary so that
    when an input lands in the ramp zone of the boundary compare,
    both adjacent sections compute ``log(x)`` correctly (not clamped).
    The 0/1 thermometer indicators (telescoping differences) sum to 1
    by construction, so :func:`multiply_2d` blending gives the linear
    interpolation between two correct values at boundaries — i.e.,
    still the correct value.

    For ranges that fit in one section
    (``x_max/x_min <= section_factor``), falls through to a single
    :func:`piecewise_linear` with no routing overhead.

    Args:
        inp: 1D scalar node with value in ``[min_value, max_value]``.
        min_value: Lower bound on input (must be > 0).
        max_value: Upper bound on input (must exceed ``min_value``).
        n_breakpoints: Approximate total breakpoint budget. Distributed
            across sections in the sectioned path; sets BP density in
            the single-section path. Default 256.
        section_factor: Geometric width of each section (default 10).
            Smaller factor means more sections — tighter precision per
            section, more total neurons.
        d_max: Maximum neurons per MLP sublayer.

    Returns:
        1D scalar node containing ``log(x)``.

    .. noise-footer::

       Max error: 0.003023 abs, 0.003705 rel over 8192 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    import math as _math

    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert min_value > 0, "min_value must be positive"
    assert max_value > min_value, "max_value must exceed min_value"
    assert n_breakpoints >= 2, "need >= 2 breakpoints"
    assert section_factor > 1.0, "section_factor must be > 1"

    log_ratio = _math.log(max_value / min_value)
    log_factor = _math.log(section_factor)
    n_sections = builtins.max(int(_math.ceil(log_ratio / log_factor)), 1)

    if n_sections == 1:
        return _log_single(
            inp, min_value, max_value, n_breakpoints, d_max, name="log"
        )

    # Geometric section endpoints.
    section_ratio = (max_value / min_value) ** (1.0 / n_sections)
    endpoints = [min_value * (section_ratio**i) for i in range(n_sections + 1)]
    endpoints[0] = min_value
    endpoints[-1] = max_value

    # Per-section breakpoint count: distribute the budget but keep a
    # floor so each section is well-resolved with many sections.
    n_bp_per_section = builtins.max(n_breakpoints // n_sections, 16)

    # Section overlap on the right side: extend each section's PWL
    # past its boundary by the ramp width of the routing compare. At
    # an input in the ramp zone, both adjacent sections then produce
    # log(x) correctly rather than the clamped boundary value.
    ramp_width = 1.0 / _LOG_BOUNDARY_SHARPNESS

    # Parallel section logs. We unwrap each section's value-range
    # Assert: its declared range (e.g. ``[log(B_i), log(B_{i+1})]``)
    # is exceeded by ~1e-3 absolute when the PWL is evaluated far
    # outside its section (float32 cancellation in the slope-delta
    # sum at large ``x``), but those out-of-section values are routed
    # to weight≈0 via the indicator gating below. The global value
    # range is re-asserted on the final sum.
    section_logs = []
    for i in range(n_sections):
        bp_lo = endpoints[i]
        bp_hi = endpoints[i + 1] + (
            ramp_width if i < n_sections - 1 else 0.0
        )
        sec = _log_single(
            inp, bp_lo, bp_hi, n_bp_per_section, d_max, name=f"log_sec_{i}"
        )
        # piecewise_linear with clamp=True wraps in assert_matches_value_type;
        # the wrapped node is `sec.inputs[0]`.
        section_logs.append(sec.inputs[0])

    # Thermometer compares against interior boundaries (0/1 outputs).
    is_above = [
        _log_compare_01(inp, endpoints[k + 1])
        for k in range(n_sections - 1)
    ]

    # 0/1 indicators via thermometer differences. Sum to 1 by
    # telescoping (including in ramp zones where adjacent indicators
    # are non-{0,1} but sum to 1).
    from torchwright.ops.inout_nodes import create_literal_value

    one_lit = create_literal_value(torch.tensor([1.0]), name="log_one_lit")
    indicators = []
    indicators.append(subtract(one_lit, is_above[0]))
    for i in range(1, n_sections - 1):
        indicators.append(subtract(is_above[i - 1], is_above[i]))
    indicators.append(is_above[-1])

    # multiply_2d blending: indicator (∈[0,1]) × section_log. At a
    # fuzzy boundary, sum = 0.5·log(B) + 0.5·log(B) = log(B).
    log_min = _math.log(min_value)
    log_max = _math.log(max_value)
    max_abs_log = builtins.max(builtins.abs(log_min), builtins.abs(log_max))

    # Indicator nominal range is [0, 1] but noisy in compare-cancellation
    # zones (extreme x). Widen multiply_2d's first-axis range so the
    # blending op still sees its inputs as valid.
    weight_lo = -_LOG_COMPARE_ATOL * 2
    weight_hi = 1.0 + _LOG_COMPARE_ATOL * 2
    weighted = []
    for i in range(n_sections):
        weighted.append(
            multiply_2d(
                indicators[i],
                section_logs[i],
                max_abs1=weight_hi,
                max_abs2=max_abs_log,
                min1=weight_lo,
                step1=0.05,
                step2=0.5,
                d_max=d_max,
                name=f"log_sec_blend_{i}",
            )
        )

    result = sum_nodes(weighted)
    return assert_matches_value_type(
        result, NodeValueType(value_range=Range(log_min, log_max))
    )


def exp(
    inp: Node,
    min_value: float,
    max_value: float,
    n_breakpoints: int = 256,
    d_max: int = 1024,
) -> Node:
    """Natural exponential via piecewise-linear interpolation.

    Uses **uniform** breakpoint spacing.  The second derivative of
    ``exp(x)`` is ``exp(x)`` itself, so the linear-interpolation error
    between two breakpoints ``[x_i, x_{i+1}]`` is approximately
    ``(Δx)² · exp(x_mid) / 8``.  Uniform spacing (constant ``Δx``)
    therefore gives constant *relative* error of ``(Δx)² / 8`` in the
    output across the whole range — the natural pairing for the
    geometric-spaced :func:`log`.

    With ``n_breakpoints = 256`` over a range of width 10
    (``Δx ≈ 0.0392``), the worst-case relative error is
    ``Δx² / 8 ≈ 1.9·10⁻⁴``.

    Args:
        inp: 1D scalar node with value in ``[min_value, max_value]``.
        min_value: Lower bound on input.
        max_value: Upper bound on input (must exceed ``min_value``).
        n_breakpoints: Number of breakpoints (default 256).  Must be
            >= 2.
        d_max: Maximum neurons per MLP sublayer.

    Returns:
        1D scalar node containing ``exp(x)``.

    .. noise-footer::

       Max error: 0.02797 abs, 0.0001924 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    import math as _math

    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_value > min_value, "max_value must exceed min_value"
    assert n_breakpoints >= 2, "need >= 2 breakpoints"

    step = (max_value - min_value) / (n_breakpoints - 1)
    breakpoints = [min_value + k * step for k in range(n_breakpoints)]
    breakpoints[-1] = max_value

    return piecewise_linear(
        inp, breakpoints, lambda x: _math.exp(x), d_max=d_max, name="exp"
    )


_LOG_ABS_SINGLE_RATIO_THRESHOLD = 1e4
"""Above this ``max_abs/min_abs`` ratio, single-piecewise :func:`log_abs`'s
float32 cancellation floor exceeds ~1.2e-3 absolute. Use the abs+sectioned
log fallback above this threshold (3 sublayers but precision bounded by
section width)."""


def _log_abs_single(
    inp: Node,
    min_abs: float,
    max_abs: float,
    n_breakpoints: int,
    d_max: int,
) -> Node:
    """Single-sublayer V-shape ``log|·|`` via :func:`piecewise_linear` over
    signed ``x``.

    Breakpoints are placed at ``[-max_abs, ..., -min_abs, min_abs, ...,
    max_abs]`` with geometric spacing on each arm. The flat V-bottom is
    represented implicitly: between the consecutive breakpoints
    ``-min_abs`` and ``+min_abs`` (no BP between), :func:`piecewise_linear`
    interpolates linearly — and both endpoints have value ``log(min_abs)``,
    so the interpolation is constant. Outside ``±max_abs``, :func:`piecewise_linear`'s
    default ``clamp=True`` holds the value at ``log(max_abs)``.
    """
    import math as _math

    n_arm = builtins.max(n_breakpoints // 2, 8)

    # Right arm: geometric BPs from min_abs to max_abs.
    ratio = (max_abs / min_abs) ** (1.0 / (n_arm - 1))
    right_arm = [min_abs * (ratio**k) for k in range(n_arm)]
    right_arm[0] = min_abs
    right_arm[-1] = max_abs

    # Left arm: mirror around 0 (so breakpoints stay strictly ascending
    # when concatenated).
    left_arm = [-v for v in reversed(right_arm)]

    breakpoints = left_arm + right_arm  # length 2 * n_arm

    def _fn(v: float) -> float:
        a = _builtin_abs(v)
        clamped = builtins.max(min_abs, builtins.min(max_abs, a))
        return _math.log(clamped)

    return piecewise_linear(
        inp, breakpoints, _fn, d_max=d_max, name="log_abs"
    )


def log_abs(
    inp: Node,
    min_abs: float = 0.1,
    max_abs: float = 100.0,
    n_breakpoints: int = 256,
    section_factor: float = 10.0,
    d_max: int = 1024,
) -> Node:
    """``log(clamp(|x|, min_abs, max_abs))`` for signed ``x``.

    Even-symmetric, V-shaped with a flat bottom at ``log(min_abs)`` over
    ``[-min_abs, +min_abs]``, and clamped to ``log(max_abs)`` outside
    ``[-max_abs, +max_abs]``. Pairs with :func:`exp` and Linear addition
    for log-domain multiplication of a signed by a positive value (e.g.
    ``signed_coord · positive_scale``). Saves the explicit
    ``abs`` + ``log`` layering by treating the V-shaped ``log|x|`` as a
    single piecewise op.

    **Implementation paths.**

    * For ratios ``max_abs/min_abs ≤ 10⁴`` (the typical case, including
      the default 3-decade ``[0.1, 100]``): a single piecewise_linear
      over signed ``x`` with V-shape breakpoints. Cost: **1 MLP
      sublayer.** The V-spikes at ``±min_abs`` (where the log slope
      transitions to/from the flat zone) each contribute roughly
      ``(max_abs/min_abs) · log_slope_jump`` magnitude to the matmul
      partial sum at extreme ``x`` — but with the threshold at 10⁴ that
      magnitude stays under ~10³, giving float32 ULP ~1.2e-4 absolute.
    * For wider ratios: compose ``log(abs(x), ..., section_factor=...)``
      with the sectioned :func:`log` op. Cost: **3 MLP sublayers** (1 for
      ``abs``, 2 for sectioned ``log``) — at parity with the explicit
      ``abs`` + ``log`` composition this op fuses below the threshold.

    Args:
        inp: 1D scalar node (signed).
        min_abs: Lower clamp on ``|x|`` (must be > 0). Below ``|x| = min_abs``
            the output is constant ``log(min_abs)``.
        max_abs: Upper clamp on ``|x|`` (must exceed ``min_abs``). Above
            ``|x| = max_abs`` the output is constant ``log(max_abs)``.
        n_breakpoints: Approximate total breakpoint budget (default 256).
            Distributed roughly evenly across the two arms in the single-
            piecewise path; passed through to :func:`log` in the fallback.
        section_factor: Geometric section width used by the wide-range
            fallback's :func:`log` (default 10). Mirrors :func:`log`'s knob
            for API consistency; unused on the single-piecewise path.
        d_max: Maximum neurons per MLP sublayer.

    Returns:
        1D scalar node containing ``log(clamp(|x|, min_abs, max_abs))``.

    .. noise-footer::

       Max error: 0.000699 abs, 0.1428 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert min_abs > 0, "min_abs must be positive"
    assert max_abs > min_abs, "max_abs must exceed min_abs"
    assert n_breakpoints >= 4, "need >= 4 breakpoints (>= 2 per arm)"
    assert section_factor > 1.0, "section_factor must be > 1"

    if max_abs / min_abs <= _LOG_ABS_SINGLE_RATIO_THRESHOLD:
        return _log_abs_single(inp, min_abs, max_abs, n_breakpoints, d_max)

    # Wide-range fallback: abs + sectioned log = 3 sublayers.
    abs_x = abs(inp)
    return log(
        abs_x,
        min_value=min_abs,
        max_value=max_abs,
        n_breakpoints=n_breakpoints,
        section_factor=section_factor,
        d_max=d_max,
    )


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
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_value >= min_value

    if max_value == min_value:
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor([float(min_value)]))

    # Build staircase: each step is a steep ramp of width eps ending at
    # the integer boundary.  This ensures floor(k) = k exactly for all
    # integers, and floor(k - delta) = k-1 for delta > eps.
    import math as _math

    s = sharpness if sharpness is not None else step_sharpness
    eps = 1.0 / s
    breakpoints = [float(min_value) - eps]
    for k in range(min_value + 1, max_value + 1):
        breakpoints.extend([float(k) - eps, float(k)])
    breakpoints.append(float(max_value) + eps)

    lo, hi = float(min_value), float(max_value)

    result = piecewise_linear(
        inp,
        breakpoints,
        lambda x: builtins.max(lo, builtins.min(hi, float(_math.floor(x)))),
        input_scale=s,
        name="floor_int",
    )
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=Range(float(min_value), float(max_value))),
    )


def ceil_int(inp: Node, min_value: int, max_value: int) -> Node:
    """Compute ceil(x) using the identity ``ceil(x) = -floor(-x)``.

    Args:
        inp: 1D scalar node with value in [min_value, max_value].
        min_value: Lower bound (integer).
        max_value: Upper bound (integer).

    Returns:
        1D scalar node containing ceil(x).

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    return negate(floor_int(negate(inp), -max_value, -min_value))


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def multiply_integers(
    inp1: Node,
    inp2: Node,
    max_value: int,
    strategy: str = "deep",
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
        d²:
          deep    → |d| = abs(d), |d|² = square(|d|)        2 MLP sublayers
          shallow → square_signed(d)                         1 MLP sublayer (~2× width)
        result = (s² - d²) / 4              Linear (free)

    Total cost: 3 MLP sublayers (deep, default) or 2 MLP sublayers (shallow).

    Args:
        inp1: 1D scalar node, integer value in [0, max_value].
        inp2: 1D scalar node, integer value in [0, max_value].
        max_value: Upper bound on each input.
        strategy: ``"deep"`` (default, abs+square, narrower) or ``"shallow"``
            (square_signed, saves 1 MLP sublayer at ~2× the width on the
            d branch).  Use ``"shallow"`` only when the target ``d`` can
            fit ``2*max_value + 1`` neurons in a single MLP sublayer.

    Returns:
        1D scalar node containing inp1 * inp2.

    .. noise-footer::

       Max error: 0 abs, 0 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert strategy in ("deep", "shallow"), f"unknown strategy: {strategy}"
    assert len(inp1) == 1, "Input must be a 1D scalar node"
    assert len(inp2) == 1, "Input must be a 1D scalar node"

    s = add(inp1, inp2)  # a+b
    d = subtract(inp1, inp2)  # a-b (may be negative)

    sq_sum = square(s, 2 * max_value)  # (a+b)²
    if strategy == "shallow":
        sq_diff = square_signed(d, max_abs=max_value)  # (a-b)²
    else:
        sq_diff = square(abs(d), max_value)  # (a-b)²

    # a*b = ((a+b)² - (a-b)²) / 4
    return add_scaled_nodes(0.25, sq_sum, -0.25, sq_diff)


def signed_multiply(
    inp1: Node,
    inp2: Node,
    max_abs1: float,
    max_abs2: float,
    step: float = 1.0,
    max_abs_output: float | None = None,
    d_max: int = 1024,
    strategy: str = "deep",
) -> Node:
    """Multiply two signed scalars using the polarization identity.

    ``a * b = (|a+b|² - |a-b|²) / 4``

    Exact when both inputs are multiples of ``step``.  Piecewise-linear
    between grid points.

    Two implementations of the squarings are available:

    - ``"deep"`` (default): ``abs(s)`` then ``square(abs_s)`` per branch.
      Depth 2 per branch (1 abs layer + 1 square layer), narrower MLP
      hidden width (~``2*ceil(max_sum/step)`` neurons total).
    - ``"shallow"``: ``square_signed(s)`` per branch.  Depth 1 per branch,
      roughly 2× the MLP hidden width (~``2*ceil(2*max_sum/step)`` neurons).
      Saves 1 MLP sublayer.

    **Precision vs. bounds (read this if the caller cares about small
    relative errors).** The absolute error in the output scales with
    ``step × max_sum`` where ``max_sum = max_abs1 + max_abs2``, not
    with ``|inp1 * inp2|``.  Loose bounds are quietly expensive: a
    caller declaring ``max_abs1 = 200`` when its actual data never
    exceeds ``10`` pays the full ``200``-scale error budget on every
    output.  The relative error on a product whose magnitude is
    ``|a*b|`` is roughly ``(step × max_sum) / (4 * |a*b|)`` — so
    halving ``max_sum`` doubles effective precision at zero neuron
    cost.  Hidden width scales linearly with ``max_sum / step``, so
    tightening the bounds also *reduces* neuron count.

    If tuning bounds isn't possible but finer precision is needed,
    shrink ``step`` — precision improves linearly, hidden width grows
    linearly.  ``step = 0.25`` with ``max_sum = 20`` gives ``~80``
    neurons per square op; ``step = 0.1`` gives ``~200``.

    Pathological inputs: when one of the factors is near zero
    (e.g. ``a = 1e-3``) and the other is near its bound (``|b| =
    max_abs2``), the polarization identity subtracts two nearly-equal
    squares, magnifying interpolation error.  Callers computing small
    products at the tail of a wide input distribution should either
    clamp upstream or drop ``step`` further.

    Args:
        inp1: 1D scalar node with value in [-max_abs1, max_abs1].
        inp2: 1D scalar node with value in [-max_abs2, max_abs2].
        max_abs1: Maximum absolute value of *inp1*.  Tighter bounds
            → better precision AND fewer neurons; see the precision
            note above.
        max_abs2: Maximum absolute value of *inp2*.  Same story.
        step: Grid spacing for accuracy.  Smaller → more neurons,
            more precision.
        max_abs_output: Optional tighter bound on the result magnitude.
            When provided, the output is clamped to [-max_abs_output,
            max_abs_output].
        d_max: Maximum neurons per MLP sublayer.
        strategy: ``"deep"`` (default) or ``"shallow"``.  Use
            ``"shallow"`` only when the target ``d`` can fit
            ``2 * (2*max_sum/step + 1)`` neurons in a single MLP sublayer.

    Returns:
        1D scalar node containing inp1 * inp2.

    .. noise-footer::

       Max error: 0.06248 abs, 2.15 rel over 4096 samples;
       measured at commit fe133f9. See docs/numerical_noise.md.
    """
    assert strategy in ("deep", "shallow"), f"unknown strategy: {strategy}"
    assert len(inp1) == 1, "Input must be a 1D scalar node"
    assert len(inp2) == 1, "Input must be a 1D scalar node"

    s = add(inp1, inp2)  # a+b
    d = subtract(inp1, inp2)  # a-b
    max_sum = max_abs1 + max_abs2

    if strategy == "shallow":
        sq_sum = square_signed(s, max_abs=max_sum, step=step, d_max=d_max)
        sq_diff = square_signed(d, max_abs=max_sum, step=step, d_max=d_max)
    else:  # "deep"
        abs_s = abs(s)  # |a+b|
        abs_d = abs(d)  # |a-b|
        sq_sum = square(abs_s, max_value=max_sum, step=step, d_max=d_max)
        sq_diff = square(abs_d, max_value=max_sum, step=step, d_max=d_max)

    result = add_scaled_nodes(0.25, sq_sum, -0.25, sq_diff)

    if max_abs_output is not None:
        result = piecewise_linear(
            result,
            [-max_abs_output, max_abs_output],
            lambda x: x,
            clamp=True,
            name="signed_multiply_clamp",
        )

    return result


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def reduce_min(keys: List[Node], values: List[Node]) -> Tuple[Node, Node]:
    """Find the (key, value) pair with the minimum key.

    Uses a binary tree reduction with ``ceil(log2(N))`` stages.

    Args:
        keys: N scalar nodes (each ``d_output=1``).
        values: N nodes (all same width).

    Returns:
        ``(winning_key, winning_value)`` tuple.
    """
    from torchwright.ops.map_select import select

    assert len(keys) == len(values) and len(keys) >= 1
    assert all(len(k) == 1 for k in keys)
    if len(values) > 1:
        d_val = len(values[0])
        assert all(len(v) == d_val for v in values)

    cur_keys = list(keys)
    cur_vals = list(values)

    while len(cur_keys) > 1:
        nxt_keys: List[Node] = []
        nxt_vals: List[Node] = []
        for i in range(0, len(cur_keys), 2):
            if i + 1 >= len(cur_keys):
                nxt_keys.append(cur_keys[i])
                nxt_vals.append(cur_vals[i])
            else:
                diff = subtract(cur_keys[i], cur_keys[i + 1])
                # cond = 1.0 when diff > 0 (k1 > k2) → pick k2
                cond = compare(diff, 0.0)
                nxt_keys.append(select(cond, cur_keys[i + 1], cur_keys[i]))
                nxt_vals.append(select(cond, cur_vals[i + 1], cur_vals[i]))
        cur_keys = nxt_keys
        cur_vals = nxt_vals

    return cur_keys[0], cur_vals[0]


def reduce_max(keys: List[Node], values: List[Node]) -> Tuple[Node, Node]:
    """Find the (key, value) pair with the maximum key.

    Uses a binary tree reduction with ``ceil(log2(N))`` stages.

    Args:
        keys: N scalar nodes (each ``d_output=1``).
        values: N nodes (all same width).

    Returns:
        ``(winning_key, winning_value)`` tuple.
    """
    from torchwright.ops.map_select import select

    assert len(keys) == len(values) and len(keys) >= 1
    assert all(len(k) == 1 for k in keys)
    if len(values) > 1:
        d_val = len(values[0])
        assert all(len(v) == d_val for v in values)

    cur_keys = list(keys)
    cur_vals = list(values)

    while len(cur_keys) > 1:
        nxt_keys: List[Node] = []
        nxt_vals: List[Node] = []
        for i in range(0, len(cur_keys), 2):
            if i + 1 >= len(cur_keys):
                nxt_keys.append(cur_keys[i])
                nxt_vals.append(cur_vals[i])
            else:
                diff = subtract(cur_keys[i], cur_keys[i + 1])
                # cond = 1.0 when diff > 0 (k1 > k2) → pick k1
                cond = compare(diff, 0.0)
                nxt_keys.append(select(cond, cur_keys[i], cur_keys[i + 1]))
                nxt_vals.append(select(cond, cur_vals[i], cur_vals[i + 1]))
        cur_keys = nxt_keys
        cur_vals = nxt_vals

    return cur_keys[0], cur_vals[0]
