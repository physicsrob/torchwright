"""Purely linear ops — compiled into residual-stream wiring, no MLP sublayers.

No hidden lane means no activation, which is why these ops carry no
machine choice: both op libraries (``torchwright/ops/relu/``,
``torchwright/ops/swiglu/``) build on them.  Anything added here must
preserve that property — an op that needs an MLP sublayer belongs in a
machine library, not in this module.
"""

import torch

from torchwright.graph import Add, Concatenate, Linear, Node, op_scope


@op_scope
def add(inp1: Node, inp2: Node) -> Node:
    """Performs element-wise addition of two input nodes.

    Args:
        inp1 (Node): First node for addition.
        inp2 (Node): Second node for addition.

    Returns:
        Node: Node resulting from element-wise addition.
    """
    return Add(inp1, inp2)


@op_scope
def subtract(inp1: Node, inp2: Node) -> Node:
    """Subtracts inp2 from inp1 element-wise.

    Args:
        inp1 (Node): Node to subtract from.
        inp2 (Node): Node to subtract.

    Returns:
        Node: Node resulting from inp1 - inp2.
    """
    return add(inp1, negate(inp2))


@op_scope
def negate(inp: Node) -> Node:
    """Negates the input node (multiplies by -1).

    Args:
        inp (Node): Node to negate.

    Returns:
        Node: Node with negated values.
    """
    d = len(inp)
    return Linear(inp, -torch.eye(d), name="negate")


@op_scope
def add_const(inp: Node, scalar: float) -> Node:
    """Adds a scalar value to each entry of the input node.

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


@op_scope
def multiply_const(inp: Node, scalar: float) -> Node:
    """Multiplies each entry of the input node by a scalar.

    Args:
        inp (Node): Node to scale.
        scalar (float): Scalar multiplier.

    Returns:
        Node: Node with scaled values.
    """
    d = len(inp)
    return Linear(inp, scalar * torch.eye(d), name="multiply_const")


@op_scope
def bool_to_01(inp: Node) -> Node:
    """Map a ±1 boolean node to 0/1.

    Converts the torchwright boolean convention (+1 = true, -1 = false)
    to a 0/1 scale (1 = true, 0 = false).  This is a free operation
    (no MLP sublayers — two linear transforms).

    Args:
        inp (Node): Boolean node with values in {-1, +1}.

    Returns:
        Node: Node with values in {0, 1}.
    """
    return multiply_const(add_const(inp, 1.0), 0.5)


@op_scope
def add_scaled_nodes(scale1: float, inp1: Node, scale2: float, inp2: Node) -> Node:
    """Computes the linear combination of two nodes using specified coefficients.

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


_MIN_FANOUT = 2  # a fanout chunk must combine at least 2 operands to be useful


@op_scope
def sum_nodes(inp_list: list[Node], *, max_fanout: int | None = None) -> Node:
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

    if max_fanout is not None and max_fanout < _MIN_FANOUT:
        raise ValueError(f"max_fanout must be >= {_MIN_FANOUT}, got {max_fanout}")

    def _flat(nodes: list[Node]) -> Node:
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
        running = _flat([running, *list(group)])
    return running


@op_scope
def concat(inp_list: list[Node]) -> Node:
    """Concatenates all the Nodes in inp_list.

    Args:
        inp_list (List[Node]): List of nodes to concatenate

    Returns:
        Node: Node resulting from concatenation
    """
    return Concatenate(inp_list)


@op_scope
def slice_columns(inp: Node, start: int, width: int, name: str = "slice") -> Node:
    """Take ``width`` consecutive components of ``inp`` starting at ``start``.

    The narrowing dual of :func:`concat`.  Realized as a ``Linear`` whose
    matrix is a shifted identity — purely linear, so it normally folds
    into a neighboring op's weights and costs nothing.  That fold is
    declined when the sliced value must stay materialized (e.g. it
    carries an attached check), in which case the slice occupies
    residual-stream wiring like any other ``Linear``.

    Args:
        inp (Node): Node to slice.
        start (int): Index of the first component to keep.
        width (int): Number of consecutive components to keep.
        name (str): Name for the resulting node.

    Returns:
        Node: ``width``-wide node holding components
        ``[start, start + width)`` of ``inp``.
    """
    if width < 1 or start < 0 or start + width > len(inp):
        raise ValueError(
            f"slice [{start}, {start + width}) out of range for width-{len(inp)} node"
        )
    proj = torch.zeros(len(inp), width)
    proj[start : start + width] = torch.eye(width)
    return Linear(inp, proj, name=name)
