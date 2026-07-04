"""1-digit adder: the simplest example of programming a transformer.

Parses "A+B\\n" where A and B are single digits, and outputs their sum.
All arithmetic is done via a single 100-entry lookup table that maps
every (A, B) pair to (A+B) mod 10.

This example keeps its helpers inline for self-containment — see
embedding_arithmetic for the general multi-digit versions.
"""

from typing import Tuple

import torch

from torchwright.graph import Node, Embedding
from torchwright.graph.embedding import Unembedding
from torchwright.ops.linear import concat
from torchwright.ops.attention_ops import attend_to_offset, get_prev_value
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_embedding,
    create_rope_config,
    create_unembedding,
)
from torchwright.ops.logic_ops import equals_vector
from torchwright.ops.map_select import map_to_table, select

# Rotary width the graph is built against; must match the d_head it is
# compiled at (the token-example harness compiles at d_head=16).
D_HEAD = 16
MAX_POSITIONS = 512


def check_is_num(embedding_value: Node, embedding: Embedding) -> Node:
    return map_to_table(
        inp=embedding_value,
        key_to_value={
            embedding.get_embedding(str(i)): torch.tensor([1.0]) for i in range(10)
        },
        default=torch.tensor([-1.0]),
    )


def sum_numbers(embedding: Embedding, num1: Node, num2: Node) -> Tuple[Node, Node]:
    """
    Adds num1 with num2.
    Assumes num1 and num2 are both embedding-valued nodes.
    return result as embedding-valued node and carry as boolean.
    """
    result_table = {}
    carry_table = {}
    for A in range(10):
        for B in range(10):
            numcat = torch.cat(
                [embedding.get_embedding(str(A)), embedding.get_embedding(str(B))]
            )
            result_table[numcat] = embedding.get_embedding(str((A + B) % 10))
            carry_table[numcat] = torch.tensor([1.0 if (A + B) >= 10 else -1.0])

    num1_num2 = concat([num1, num2])
    return (
        map_to_table(
            inp=num1_num2,
            key_to_value=result_table,
            default=embedding.get_embedding("0"),
        ),
        map_to_table(num1_num2, key_to_value=carry_table, default=torch.tensor([-1.0])),
    )


def create_network() -> Unembedding:
    # --- Phase 1: Vocabulary and parsing ---
    vocab = list(
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["\n", "<bos>", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)

    # Determine the current digit (default to "0" for non-digit tokens).
    zero_constant = create_literal_value(embedding.get_embedding("0"))
    is_num = check_is_num(embedding_value=embedding, embedding=embedding)
    current_num = select(cond=is_num, true_node=embedding, false_node=zero_constant)

    # Detect operator positions: "+" ends the first number, "=" ends the second.
    is_first_num = equals_vector(inp=embedding, vector=embedding.get_embedding("+"))
    is_second_num = equals_vector(inp=embedding, vector=embedding.get_embedding("\n"))

    # --- Phase 2: Capture operands and compute ---
    # Look one position back to get the digit that just completed.
    just_completed_num = attend_to_offset(rope, current_num, delta_pos=-1)
    # Latch: remember the digit at "+", carry it forward to all later positions.
    first_num = get_prev_value(rope, just_completed_num, is_first_num)
    # Latch: remember the digit at "=".
    second_num = get_prev_value(rope, just_completed_num, is_second_num)

    summed, carry = sum_numbers(embedding, first_num, second_num)

    # --- Phase 3: Output ---
    return create_unembedding(summed, embedding)
