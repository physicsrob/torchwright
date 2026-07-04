"""Bridge between embedding and scalar representations.

These functions convert digit embeddings into scalar numbers and back again,
enabling arithmetic in scalar space. The typical pipeline is:

    digits_to_number  →  scalar arithmetic  →  number_to_digit_scalars  →  scalar_to_embedding

Use this when numbers arrive as digit tokens and you want to do arithmetic
as plain addition/subtraction/multiplication on scalars, then convert the
result back to token embeddings. The scalar-space approach avoids the
combinatorial lookup tables of embedding-space arithmetic, but requires
thermometer coding (ReLU-based threshold detectors) for the conversions.

See also:
    arithmetic_ops.thermometer_floor_div — the core integer division primitive
    arithmetic_ops.square — squaring via ReLU ramps
    embedding_arithmetic — the alternative: stay in embedding space throughout
"""

from typing import List

import torch

from torchwright.graph import Node, Embedding
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.arithmetic_ops import thermometer_floor_div
from torchwright.ops.linear import add_scaled_nodes, sum_nodes
from torchwright.ops.const import step_sharpness
from torchwright.ops.linear_relu_linear import linear_relu_linear
from torchwright.ops.map_select import map_to_table


def digit_to_scaled_scalar(
    embedding: Embedding, digit_node: Node, place_value: float
) -> Node:
    """Convert a digit embedding to a scalar multiplied by place_value.

    Example: embed("5") with place_value=100 → 500.0

    Uses a 10-entry lookup table: embed("0")→0, embed("1")→place_value, ...,
    embed("9")→9*place_value.
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

    Greedy extraction: peel off the most significant digit, subtract it,
    repeat on the remainder.

    Example: 579 → 3 digits

        digit[0] = floor(579 / 100) = 5    remainder = 579 - 5*100 = 79
        digit[1] = floor(79 / 10)   = 7    remainder = 79  - 7*10  = 9
        digit[2] = remainder         = 9

    max_value shrinks at each step (999→99→9) so the thermometer MLPs
    only need thresholds for the remaining range.
    """
    digits = []
    remainder = inp
    for i in range(num_digits):
        place = 10 ** (num_digits - 1 - i)
        if place == 1:
            # Last digit: just the remainder, no division needed.
            # Declare the tight [0, 9] range so downstream ops see it.
            digits.append(
                assert_matches_value_type(
                    remainder,
                    NodeValueType(value_range=Range(0.0, 9.0)),
                )
            )
        else:
            digit = thermometer_floor_div(remainder, place, max_value)
            digits.append(digit)
            # remainder = remainder - digit * place. The math guarantees
            # the new remainder is in [0, place - 1]; declare that so
            # Linear's interval arithmetic doesn't pessimistically inflate
            # it on subsequent iterations.
            remainder = add_scaled_nodes(1.0, remainder, -float(place), digit)
            remainder = assert_matches_value_type(
                remainder,
                NodeValueType(value_range=Range(0.0, float(place - 1))),
            )
            max_value = place - 1
    return digits


def scalar_to_embedding(inp: Node, embedding: Embedding) -> Node:
    """Convert a scalar digit (0.0-9.0) back to its embedding vector.

    Same paired-ReLU thermometer as thermometer_floor_div, but instead of
    each threshold contributing 1.0, it contributes an embedding delta.
    This reconstructs the embedding by telescoping from embed(0):

        input=0 → embed(0)
        input=1 → embed(0) + [embed(1) - embed(0)]              = embed(1)
        input=2 → embed(0) + [embed(1)-embed(0)] + [embed(2)-embed(1)] = embed(2)
        ...
        input=d → embed(d)

    9 thresholds at 0.5, 1.5, ..., 8.5 cover digits 0-9.

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
    d_hidden = 2 * n_thresholds

    input_proj = torch.zeros(d_hidden, 1)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, d_embed)

    for k in range(n_thresholds):
        threshold = k + 0.5
        row = 2 * k

        # Same paired-ReLU step function as thermometer_floor_div
        input_proj[row, 0] = step_sharpness
        input_proj[row + 1, 0] = step_sharpness
        input_bias[row] = -step_sharpness * threshold
        input_bias[row + 1] = -step_sharpness * threshold - 1.0

        # Instead of contributing 1.0, contribute the embedding delta
        delta = embedding.get_embedding(str(k + 1)) - embedding.get_embedding(str(k))
        output_proj[row, :] = delta
        output_proj[row + 1, :] = -delta

    # Start from embed("0"); deltas telescope up to embed(d)
    output_bias = embedding.get_embedding(str(0)).clone()

    return linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )
