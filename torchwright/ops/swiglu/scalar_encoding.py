"""Scalar↔embedding conversion on the swish machine.

The digit pipeline: ``digits_to_number`` → scalar arithmetic →
``number_to_digit_scalars`` → ``scalar_to_embedding``.  The conversions
are compositions of swiglu ingredients (map_to_table banks and the
thermometer staircase); the pipeline structure is unchanged from relu.
"""

from typing import List

import torch

from torchwright.graph import Node
from torchwright.graph.embedding import Embedding
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import scale, step_sharpness

from torchwright.ops.linear import add_scaled_nodes, sum_nodes
from torchwright.ops.swiglu.arithmetic_ops import thermometer_floor_div
from torchwright.ops.swiglu.map_select import map_to_table
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn


def scalar_to_embedding(inp: Node, embedding: Embedding) -> Node:
    """Convert a scalar digit (0.0-9.0) back to its embedding vector.

    A :func:`piecewise_linear` special case ported hinge-for-hinge: nine
    unit steps at half-integer thresholds, vector-valued (the embedding
    deltas fold into out_proj, all lanes degenerate)::

        step_k = hinge(z_k) − hinge(z_k − 1),  z_k = S·(x − (k+0.5))
        result = embed(0) + Σ_k step_k · (embed(k+1) − embed(k))

    The single-FFN form stays safe — no floor_int-style two-stage split:
    there are 9 boundaries and the sharpened arguments top out near
    ``scale·S·9 ≈ 1e4``, the small-and-safe end of the construction.  An
    integer digit puts every hinge argument on an exact integer with
    ``|z| ≥ 5`` — every sharpened argument ≥ 500, saturated or
    underflowed exactly — so the indicators are exact 0/1 and the only
    error is out_proj rounding (~ulps of the embedding components).
    Input-noise headroom unchanged: a digit scalar off by up to ±0.4
    reconstructs the same embedding; mid-ramp inputs blend the two
    adjacent embeddings linearly, as today.  The piecewise_linear
    spacing audit closes in closed form here: the pair's hinges sit
    ``1/S`` apart with fillet radius ``17/(scale·S)``, non-overlapping
    exactly when ``scale > 34`` — the module scale clears it 3x.

    Args:
        inp: 1D scalar node with value in [0.0, 9.0].
        embedding: Embedding table (must contain "0"-"9").

    Returns:
        Node of width ``embedding.d_embed`` containing the reconstructed
        embedding vector.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    d_embed = embedding.d_embed
    n_thresholds = 9
    n_lanes = 2 * n_thresholds
    S = step_sharpness

    gate_proj = torch.zeros(n_lanes, 1)
    gate_bias = torch.zeros(n_lanes)
    output_proj = torch.zeros(n_lanes, d_embed)

    for k in range(n_thresholds):
        threshold = k + 0.5
        row = 2 * k

        # Unit-step hinge pair: bends at z_k = 0 and z_k = 1.
        gate_proj[row, 0] = scale * S
        gate_proj[row + 1, 0] = scale * S
        gate_bias[row] = -scale * S * threshold
        gate_bias[row + 1] = -scale * (S * threshold + 1.0)

        # The step contributes the embedding delta; /scale folds in.
        delta = embedding.get_embedding(str(k + 1)) - embedding.get_embedding(str(k))
        output_proj[row, :] = delta / scale
        output_proj[row + 1, :] = -delta / scale

    # Start from embed("0"); deltas telescope up to embed(d).
    output_bias = embedding.get_embedding(str(0)).clone()

    return swiglu_ffn(
        inp,
        gate_proj,
        gate_bias,
        output_proj,
        output_bias,
        name="scalar_to_embedding",
    )


def digit_to_scaled_scalar(
    embedding: Embedding, digit_node: Node, place_value: float
) -> Node:
    """Convert a digit embedding to a scalar multiplied by place_value.

    Example: embed("5") with place_value=100 → 500.0

    A 10-entry :func:`map_to_table` lookup — inherits that entry.
    """
    table = {}
    for i in range(10):
        table[embedding.get_embedding(str(i))] = torch.tensor([float(i) * place_value])
    return map_to_table(inp=digit_node, key_to_value=table, default=torch.tensor([0.0]))


def digits_to_number(embedding: Embedding, digit_nodes: List[Node]) -> Node:
    """Convert digit embeddings (MSB first) to a single scalar.

    Example: [embed("1"), embed("2"), embed("3")]
           → 1*100 + 2*10 + 3*1 = 123.0
    """
    num_digits = len(digit_nodes)
    scaled = []
    for i, digit in enumerate(digit_nodes):
        place_value = 10.0 ** (num_digits - 1 - i)
        scaled.append(digit_to_scaled_scalar(embedding, digit, place_value))
    return sum_nodes(scaled)


def number_to_digit_scalars(inp: Node, num_digits: int, max_value: int) -> List[Node]:
    """Extract individual digit scalars (0.0-9.0) from a scalar number, MSB first.

    Greedy extraction via :func:`thermometer_floor_div`: peel off the most
    significant digit, subtract it, repeat on the remainder — structure
    unchanged from relu.
    """
    digits = []
    remainder = inp
    for i in range(num_digits):
        place = 10 ** (num_digits - 1 - i)
        if place == 1:
            digits.append(
                assert_matches_value_type(
                    remainder,
                    NodeValueType(value_range=Range(0.0, 9.0)),
                )
            )
        else:
            digit = thermometer_floor_div(remainder, place, max_value)
            digits.append(digit)
            remainder = add_scaled_nodes(1.0, remainder, -float(place), digit)
            remainder = assert_matches_value_type(
                remainder,
                NodeValueType(value_range=Range(0.0, float(place - 1))),
            )
            max_value = place - 1
    return digits
