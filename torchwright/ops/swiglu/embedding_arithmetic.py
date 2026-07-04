"""Digit-by-digit arithmetic in embedding space, on the swish machine.

School-math carry-propagation addition via exhaustive
:func:`~torchwright.ops.swiglu.map_select.map_to_table` lookups — the
compositions are structurally identical to relu's; only the lookup
banks' activation changed (they inherit map_to_table's entry).

All sequences are MSB-first: seq[0] is the most significant digit.
"""

from typing import List, Tuple

import torch

from torchwright.graph import Embedding, Node

from torchwright.ops.linear import concat
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.swiglu.map_select import map_to_table


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

    Sequences are MSB-first; overflow digit not included.
    """
    carry = create_literal_value(torch.tensor([-1.0]))
    out = []
    for digit1, digit2 in reversed(list(zip(seq1, seq2))):
        sum, carry = sum_digits(embedding, digit1, digit2, carry)
        out.append(sum)

    return list(reversed(out))
