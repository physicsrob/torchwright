"""Digit-by-digit arithmetic in embedding space via lookup tables.

These functions implement school-math carry-propagation addition entirely in
embedding space. Every digit pair is enumerated in a map_to_table lookup,
making the approach exhaustive but exact.

This is the embedding-space counterpart to scalar arithmetic via
scalar_encoding. Use embedding-space when the number of possible values is
small (e.g., single digits 0-9), or when you want to avoid the thermometer
threshold count that scalar-space requires. Use scalar-space (scalar_encoding
+ arithmetic_ops) when operating on numbers as wholes is simpler than
digit-by-digit propagation.

All sequences are MSB-first: seq[0] is the most significant digit.
"""

from typing import Tuple, List

import torch

from torchwright.graph import Node, Embedding
from torchwright.ops.arithmetic_ops import concat
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.map_select import map_to_table

# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


def sum_digits(
    embedding: Embedding, num1: Node, num2: Node, carry_in: Node
) -> Tuple[Node, Node]:
    """Add two single-digit embeddings plus a carry bit.

    Args:
        embedding: The embedding table (must contain "0"-"9").
        num1: Embedding-valued node for the first digit.
        num2: Embedding-valued node for the second digit.
        carry_in: Boolean node (1.0 = carry, -1.0 = no carry).

    Returns:
        (result_digit, carry_out): result_digit is an embedding-valued node,
        carry_out is a boolean node.
    """
    sum_out_table = {}
    carry_out_table = {}

    for A in range(10):
        for B in range(10):
            for C in [0, 1]:
                entry_key = torch.cat(
                    [
                        embedding.get_embedding(str(A)),
                        embedding.get_embedding(str(B)),
                        torch.tensor([1.0 if C else -1.0]),
                    ]
                )
                sum_out_table[entry_key] = embedding.get_embedding(
                    str((A + B + C) % 10)
                )
                carry_out_table[entry_key] = torch.tensor(
                    [1.0 if (A + B + C) >= 10 else -1.0]
                )

    key = concat([num1, num2, carry_in])

    return (
        map_to_table(
            inp=key,
            key_to_value=sum_out_table,
            default=embedding.get_embedding("0"),
        ),
        map_to_table(key, key_to_value=carry_out_table, default=torch.tensor([-1.0])),
    )


def sum_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
    """Add two digit sequences with carry propagation, right-to-left.

    Sequences are MSB-first: seq[0] is the most significant digit,
    seq[-1] is the least significant.

    Args:
        embedding: The embedding table (must contain "0"-"9").
        seq1: List of embedding-valued digit nodes (MSB-first).
        seq2: List of embedding-valued digit nodes (same length as *seq1*).

    Returns:
        List of embedding-valued digit nodes for the sum (MSB-first),
        same length as the inputs (overflow digit not included).
    """
    carry = create_literal_value(torch.tensor([-1.0]))
    out = []
    for digit1, digit2 in reversed(list(zip(seq1, seq2))):
        sum, carry = sum_digits(embedding, digit1, digit2, carry)
        out.append(sum)

    return list(reversed(out))
