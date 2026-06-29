"""Autoregressive Fibonacci generator.

After a trigger token, generates Fibonacci numbers autoregressively.
Each number is emitted as a fixed-width block of W digit tokens (zero-padded),
so the transformer can use fixed attend_to_offset to read the previous
two numbers' digits.

    Input:  <bos> f i b o n a c c i \\n
    Output: 01 01 02 03 05 08 13 21
            (W=2, n_terms=8)

This is the first example with genuine autoregressive recurrence — each
output number depends on previously generated tokens. The transformer
attends to the two prior W-digit blocks, adds them in scalar space,
and emits the next number digit by digit. The seeds (first two terms)
are precomputed constants; the remaining terms are computed live from
the model's own prior output.

The input prompt must have enough tokens before \\n to avoid out-of-bounds
attention at the trigger position (at least n_terms * digit_width tokens).
"""

from typing import Tuple

import torch

from torchwright.graph import Node, Embedding
from torchwright.graph.embedding import Unembedding
from torchwright.ops.arithmetic_ops import add
from torchwright.ops.attention_ops import attend_to_offset
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_embedding,
    create_rope_config,
    create_unembedding,
)
from torchwright.ops.logic_ops import equals_vector
from torchwright.ops.recency_heads import recency_rank_from_tokens
from torchwright.ops.scalar_encoding import (
    digits_to_number,
    number_to_digit_scalars,
    scalar_to_embedding,
)
from torchwright.ops.sequence_ops import output_sequence

D_MODEL = 512
# Rotary width the graph is built against; must match the d_head it is
# compiled at (the token-example harness default is d_head=16).
D_HEAD = 16
MAX_POSITIONS = 512

# Fixed digit width per Fibonacci number. W=2 handles values up to 99.
DIGIT_WIDTH = 2

# Number of terms to generate.
N_TERMS = 8


def create_network_parts(
    digit_width: int = DIGIT_WIDTH,
    n_terms: int = N_TERMS,
) -> Tuple[Node, Embedding]:
    """Build the Fibonacci generator computation graph.

    Args:
        digit_width: Number of digit tokens per Fibonacci number (zero-padded).
            Determines the maximum representable value (10^digit_width - 1).
        n_terms: Number of Fibonacci terms to generate (including the two seeds).
    """
    vocab = list("0123456789 abcdefghijklmnopqrstuvwxyz") + [
        "\n",
        "<bos>",
        "<ref>",
        "<eos>",
    ]
    embedding = create_embedding(vocab=vocab)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)
    # Bucket-2 recency rank from the <bos>/<ref> markers — drives the
    # "has the trigger fired" latch inside output_sequence.
    recency_rank = recency_rank_from_tokens(rope, embedding)
    embed = embedding.get_embedding

    is_trigger = equals_vector(embedding, embed("\n"))
    W = digit_width
    max_value = 10**W - 1  # e.g., 99 for W=2

    # --- Build recurrence computation for each digit position ---
    # For digit position d within a W-digit block, the offsets to read
    # F(n-1) and F(n-2) depend on d. Generated tokens start at trigger+1
    # (trigger+0 is still \n in the input), so offsets are shifted by 1:
    #   F(n-1) digit i is at offset -(W - 1 + d - i)
    #   F(n-2) digit i is at offset -(2W - 1 + d - i)
    recurrence_digits = []
    for d in range(W):
        # Read all W digits of F(n-1) (MSB first)
        prev1 = [
            attend_to_offset(rope, embedding, delta_pos=-(W - 1 + d - i))
            for i in range(W)
        ]
        # Read all W digits of F(n-2) (MSB first)
        prev2 = [
            attend_to_offset(rope, embedding, delta_pos=-(2 * W - 1 + d - i))
            for i in range(W)
        ]
        # Convert to scalar numbers and add
        num1 = digits_to_number(embedding, prev1)
        num2 = digits_to_number(embedding, prev2)
        fib_sum = add(num1, num2)

        # Extract digit d of the result
        result_scalars = number_to_digit_scalars(fib_sum, W, 2 * max_value)
        recurrence_digits.append(scalar_to_embedding(result_scalars[d], embedding))

    # --- Build output sequence ---
    # First 2*W positions: seed digits for F(0)=1 and F(1)=1
    # Remaining (n_terms-2)*W positions: recurrence computation
    seed_fib = [1, 1]
    seed_tokens = []
    for f in seed_fib:
        for i in range(W):
            place = 10 ** (W - 1 - i)
            digit = (f // place) % 10
            seed_tokens.append(create_literal_value(embed(str(digit))))

    recurrence_tokens = []
    for _ in range(n_terms - 2):
        recurrence_tokens.extend(recurrence_digits)

    all_tokens = (
        seed_tokens + recurrence_tokens + [create_literal_value(embed("<eos>"))]
    )

    output_node = output_sequence(
        rope,
        is_trigger,
        all_tokens,
        embed(" "),
        recency_rank,
    )
    return output_node, embedding


def create_network(
    digit_width: int = DIGIT_WIDTH, n_terms: int = N_TERMS
) -> Unembedding:
    """Create a Fibonacci generator network."""
    output_node, embedding = create_network_parts(digit_width, n_terms)
    return create_unembedding(output_node, embedding)
