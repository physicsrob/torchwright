"""3-digit adder using embedding-space arithmetic.

Parses "A+B\\n" where A and B are up to 3-digit numbers, and outputs their
sum autoregressively. All arithmetic is done digit-by-digit in embedding
space: each digit pair is looked up in a table, with carry propagation
right-to-left — exactly like pencil-and-paper addition.

See adder_v2 for the scalar-space alternative, which converts digits to
a single number, adds as a scalar, and converts back.
"""

from typing import Tuple

import torch

from torchwright.graph import Node, Embedding, RopeConfig
from torchwright.graph.embedding import Unembedding
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_embedding,
    create_rope_config,
    create_unembedding,
)
from torchwright.ops.swiglu.logic_ops import equals_vector
from torchwright.ops.swiglu.embedding_arithmetic import sum_digit_seqs
from torchwright.ops.swiglu.sequence_ops import (
    NumericSequence,
    output_sequence,
    remove_leading_0s,
)

max_digits = 3
D_MODEL = 256
# Rotary width the graph is built against; must match the d_head it is
# compiled at (the token-example harness compiles at d_head=16).
D_HEAD = 16
MAX_POSITIONS = 512


def create_network_parts() -> Tuple[Node, Embedding]:
    """Build the 3-digit adder graph and return (output_node, embedding)."""
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["\n", "<bos>", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)

    # --- Phase 1: Parse operand digits from the token stream ---
    # NumericSequence tracks a sliding window of digits as tokens arrive.
    # get_digits_at_event captures the window when the trigger token appears
    # (recency is the intrinsic rotary lobe — no rank node, local-only).
    num_seq = NumericSequence(rope, embedding, max_digits)

    is_end_of_first_num = equals_vector(
        inp=embedding, vector=embedding.get_embedding("+")
    )
    is_end_of_second_num = equals_vector(
        inp=embedding, vector=embedding.get_embedding("\n")
    )

    first_num_digits = num_seq.get_digits_at_event(is_end_of_first_num)
    second_num_digits = num_seq.get_digits_at_event(is_end_of_second_num)

    # --- Phase 2: Add digit-by-digit with carry propagation ---
    sum_digits = sum_digit_seqs(embedding, first_num_digits, second_num_digits) + [
        create_literal_value(embedding.get_embedding("<eos>"))
    ]
    sum_digits = remove_leading_0s(embedding, sum_digits, max_removals=max_digits - 1)

    # --- Phase 3: Output result autoregressively ---
    output_node = output_sequence(
        rope,
        is_end_of_second_num,
        sum_digits,
        embedding.get_embedding(" "),
    )
    return output_node, embedding


def create_network() -> Unembedding:
    output_node, embedding = create_network_parts()
    return create_unembedding(output_node, embedding)
