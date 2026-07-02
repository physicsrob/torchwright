import torch

from torchwright.graph import Node
from torchwright.graph.block import Block


def linear_relu_linear(
    input_node: Node,
    input_proj: torch.Tensor,
    input_bias: torch.Tensor,
    output_proj: torch.Tensor,
    output_bias: torch.Tensor,
    name: str = "",
) -> Node:
    """Build a degenerate-ReLU :class:`~torchwright.graph.block.Block`.

    This is the fundamental building block for piecewise-linear functions in
    the computation graph: an input projection, a ReLU, and an output
    projection.  It returns a single :class:`Block` node (not the three
    ``Linear -> ReLU -> Linear`` nodes it once built) — the ReLU and the two
    projections are the Block's gate projection, activation, and output
    projection.

    A Block is a **packable unit**, not a sublayer: the scheduler bins many
    blocks' lanes into one MLP sublayer's hidden pool, so several calls can
    share one compiled MLP sublayer.  (The old "one call = one MLP sublayer"
    description was never quite true and is easy to over-rely on.)

    Args:
        input_node: Upstream node whose output feeds the gate projection.
        input_proj: Gate-projection weight matrix, shape ``(d_hidden, d_input)``.
        input_bias: Gate-projection bias, shape ``(d_hidden,)``.
        output_proj: Output-projection weight matrix, shape
            ``(d_hidden, d_output)``.
        output_bias: Output-projection bias, shape ``(d_output,)``.
        name: Label for the Block (for debugging).

    Returns:
        The :class:`Block` node computing
        ``ReLU(x @ input_proj.T + input_bias) @ output_proj + output_bias``.
    """
    if len(input_proj.shape) == 1:
        input_proj = input_proj.unsqueeze(0)
    if len(input_bias.shape) == 0:
        input_bias = input_bias.unsqueeze(0)
    if len(output_proj.shape) == 1:
        output_proj = output_proj.unsqueeze(0)

    d_hidden = input_proj.shape[0]
    d_input = input_proj.shape[1]

    assert input_proj.shape == (d_hidden, d_input)
    assert input_bias.shape == (d_hidden,)

    d_output = output_proj.shape[1]
    assert output_proj.shape == (d_hidden, d_output)
    assert output_bias.shape == (d_output,)

    return Block(
        input_node,
        gate_proj=input_proj,
        gate_bias=input_bias,
        out_proj=output_proj,
        out_bias=output_bias,
        activation="relu",
        name=name,
    )
