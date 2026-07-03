from typing import Optional

import torch

from torchwright.graph import Node
from torchwright.graph.ffn import FFN


def swiglu_ffn(
    input_node: Node,
    gate_proj: torch.Tensor,
    gate_bias: torch.Tensor,
    output_proj: torch.Tensor,
    output_bias: torch.Tensor,
    *,
    up_proj: Optional[torch.Tensor] = None,
    up_bias: Optional[torch.Tensor] = None,
    name: str = "",
) -> Node:
    """Build a swish :class:`~torchwright.graph.ffn.FFN`.

    The swish-machine counterpart of :func:`linear_relu_linear`, and the
    building block every op in ``torchwright/ops/swiglu/`` assembles its
    lanes with.  For an input row ``x``::

        lane[j] = Swish(gate_proj[j] · x + gate_bias[j]) * up[j](x)
        output  = lane @ output_proj + output_bias

    where ``up[j](x)`` is the constant 1 when ``up_proj is None`` (every
    lane degenerate) or the affine ``up_proj[j] · x + up_bias[j]`` when
    set (every lane gated) — one call is uniformly one or the other.

    The name describes the compiled substrate the call becomes a piece
    of, not the per-call math: a degenerate bank still occupies gate
    *and* up rows of the physical SwiGLU sublayer (the weight writer
    gives its lanes up-row 0, up-bias 1).  An FFN is a **packable
    unit**, not a sublayer: the scheduler bins many FFNs' lanes into one
    MLP sublayer's hidden pool.

    Args:
        input_node: Upstream node whose output feeds the projections.
        gate_proj: Gate-projection weight matrix, shape
            ``(n_lanes, d_input)``.
        gate_bias: Gate-projection bias, shape ``(n_lanes,)``.
        output_proj: Output-projection weight matrix, shape
            ``(n_lanes, d_output)``.
        output_bias: Output-projection bias, shape ``(d_output,)``.
        up_proj: Up-projection weight matrix, shape
            ``(n_lanes, d_input)``, or ``None`` for degenerate lanes.
        up_bias: Up-projection bias, shape ``(n_lanes,)``; set exactly
            when ``up_proj`` is.
        name: Label for the FFN (for debugging).

    Returns:
        The swish :class:`FFN` node.
    """
    if len(gate_proj.shape) == 1:
        gate_proj = gate_proj.unsqueeze(0)
    if len(gate_bias.shape) == 0:
        gate_bias = gate_bias.unsqueeze(0)
    if len(output_proj.shape) == 1:
        output_proj = output_proj.unsqueeze(0)
    if up_proj is not None and len(up_proj.shape) == 1:
        up_proj = up_proj.unsqueeze(0)
    if up_bias is not None and len(up_bias.shape) == 0:
        up_bias = up_bias.unsqueeze(0)

    n_lanes = gate_proj.shape[0]
    d_input = gate_proj.shape[1]

    assert gate_proj.shape == (n_lanes, d_input)
    assert gate_bias.shape == (n_lanes,)

    d_output = output_proj.shape[1]
    assert output_proj.shape == (n_lanes, d_output)
    assert output_bias.shape == (d_output,)

    return FFN(
        input_node,
        gate_proj=gate_proj,
        gate_bias=gate_bias,
        out_proj=output_proj,
        out_bias=output_bias,
        up_proj=up_proj,
        up_bias=up_bias,
        activation="swish",
        name=name,
    )
