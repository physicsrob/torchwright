"""Binary increment.

Parses a binary string like "1011\\n" and outputs the incremented value
"1100". Handles carry propagation through consecutive trailing 1s
(e.g., "1111\\n" → "10000", overflow adds a leading bit).

The algorithm reads all bits at the trigger position via backward
attention, then runs a sequential carry chain from LSB to MSB: carry
starts as True (adding 1), propagates through "1" bits (which flip to
"0"), and stops at the first "0" bit (which flips to "1"). Overflow
(all 1s input) adds a leading bit.

    Input:  <bos> 1 0 1 1 \\n
    Output: 1 1 0 0

    Input:  <bos> 1 1 1 1 \\n
    Output: 1 0 0 0 0  (overflow adds a bit)
"""

from typing import Tuple

import torch

from torchwright.graph import Node, Embedding, RopeConfig
from torchwright.graph.embedding import Unembedding
from torchwright.ops.attention_ops import attend_to_offset, get_prev_value
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_embedding,
    create_rope_config,
    create_unembedding,
)

from torchwright.ops.swiglu.logic_ops import bool_all_true, equals_vector
from torchwright.ops.swiglu.map_select import map_to_table, select
from torchwright.ops.swiglu.sequence_ops import output_sequence, remove_leading_0s

D_MODEL = 256
# Rotary width the graph is built against; must match the d_head it is
# compiled at (the token-example harness compiles at this d_head).
D_HEAD = 16
MAX_POSITIONS = 512


def create_network_parts(
    max_bits: int = 4,
) -> Tuple[Node, Embedding]:
    """Build the binary increment computation graph.

    Args:
        max_bits: Maximum number of input bits (handles up to max_bits-bit
            binary numbers). Output may be max_bits+1 bits on overflow.
    """
    vocab = list("01 ") + ["\n", "<bos>", "<eos>"]
    embedding = create_embedding(vocab=vocab)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)
    embed = embedding.get_embedding

    is_trigger = equals_vector(embedding, embed("\n"))

    zero_embed = create_literal_value(embed("0"))
    one_embed = create_literal_value(embed("1"))
    eos_embed = create_literal_value(embed("<eos>"))

    # --- Read all bit positions at trigger (LSB first) ---
    # bits[0] = LSB (rightmost), bits[max_bits-1] = MSB (leftmost)
    bits_raw = []
    for i in range(max_bits):
        raw = attend_to_offset(rope, embedding, delta_pos=-(i + 1))
        latched = get_prev_value(rope, raw, is_trigger)
        bits_raw.append(latched)

    # --- Normalize padding to "0" ---
    # Positions that read <bos> (shorter inputs) become "0".
    bits = [
        map_to_table(b, {embed("1"): embed("1")}, default=embed("0")) for b in bits_raw
    ]
    is_one = [equals_vector(b, embed("1")) for b in bits_raw]

    # --- Sequential carry chain (LSB to MSB) ---
    # carry[0] = True: we're adding 1
    carry = [create_literal_value(torch.tensor([1.0]))]
    for i in range(max_bits):
        # Carry propagates only through "1" bits
        carry.append(bool_all_true([is_one[i], carry[i]]))

    # --- Compute new bits ---
    new_bits = []
    for i in range(max_bits):
        # When carry reaches this bit, flip it: 1→0, 0→1
        flipped = select(is_one[i], zero_embed, one_embed)
        new_bits.append(select(carry[i], flipped, bits[i]))

    # --- Overflow: carry past MSB means all bits were 1 ---
    overflow_bit = select(carry[max_bits], one_embed, zero_embed)

    # --- Build output (MSB first) ---
    output_seq = [overflow_bit] + list(reversed(new_bits)) + [eos_embed]
    output_seq = remove_leading_0s(embedding, output_seq, max_removals=max_bits)

    output_node = output_sequence(
        rope,
        is_trigger,
        output_seq,
        embed(" "),
    )
    return output_node, embedding


def create_network(max_bits: int = 4) -> Unembedding:
    """Create a binary increment network."""
    output_node, embedding = create_network_parts(max_bits)
    return create_unembedding(output_node, embedding)
