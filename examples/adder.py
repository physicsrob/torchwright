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

from torchwright.graph import Node, Embedding, PosEncoding
from torchwright.graph.embedding import Unembedding
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_embedding,
    create_pos_encoding,
    create_unembedding,
)
from torchwright.ops.logic_ops import equals_vector
from torchwright.ops.embedding_arithmetic import sum_digit_seqs
from torchwright.ops.sequence_ops import (
    NumericSequence,
    output_sequence,
    remove_leading_0s,
)

max_digits = 3
D_MODEL = 256


def create_network_parts() -> Tuple[Node, PosEncoding, Embedding]:
    """Build the 3-digit adder graph and return (output_node, pos_encoding, embedding)."""
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["\n", "<bos>", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    # --- Phase 1: Parse operand digits from the token stream ---
    # NumericSequence tracks a sliding window of digits as tokens arrive.
    # get_digits_at_event captures the window when the trigger token appears.
    num_seq = NumericSequence(pos_encoding, embedding, max_digits)

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
        pos_encoding,
        is_end_of_second_num,
        sum_digits,
        embedding.get_embedding(" "),
    )
    return output_node, pos_encoding, embedding


def create_network() -> Unembedding:
    output_node, pos_encoding, embedding = create_network_parts()
    return create_unembedding(output_node, embedding)
