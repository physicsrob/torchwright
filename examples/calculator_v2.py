"""Calculator V2: Scalar-space +, -, * on positive integers.

All three operations work in scalar space — digit embeddings are converted
to plain numbers, arithmetic is performed on scalars, then results are
converted back to digit embeddings.

  "12*34\n"
  → [embed("1"), embed("2")], [embed("3"), embed("4")]   parse digits
  → 12.0, 34.0                                           digits_to_number
  → 408.0                                                multiply_integers
  → [4.0, 0.0, 8.0]                                      number_to_digit_scalars
  → [embed("4"), embed("0"), embed("8")]                  scalar_to_embedding

Addition and subtraction are trivial in scalar space (one Add/Subtract node).
Multiplication uses the polarization identity: a*b = ((a+b)^2 - (a-b)^2) / 4,
where squaring is implemented via thermometer coding (see arithmetic_ops).
"""

from typing import Tuple, List

import torch

from torchwright.graph import Node, Embedding, RopeConfig
from torchwright.graph.embedding import Unembedding

# This example stays on the ReLU machine: it is the graph behind
# calculator_hf_export.py (and tests/hf/), and the HF converter's native
# module is relu-only by design — it refuses swish artifacts.  It is
# also the remaining exerciser of relu-only ops (relu_add,
# multiply_integers).  Every other example builds the swish machine via
# ops/swiglu.
from torchwright.ops.relu.arithmetic_ops import compare, relu_add, multiply_integers
from torchwright.ops.linear import add, negate, subtract
from torchwright.ops.attention_ops import get_prev_value
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_embedding,
    create_rope_config,
    create_unembedding,
)
from torchwright.ops.relu.logic_ops import (
    equals_vector,
    bool_any_true,
    bool_all_true,
    bool_not,
)
from torchwright.ops.relu.map_select import select, switch
from torchwright.ops.relu.scalar_encoding import (
    digits_to_number,
    number_to_digit_scalars,
    scalar_to_embedding,
)
from torchwright.ops.relu.sequence_ops import (
    NumericSequence,
    output_sequence,
    remove_leading_0s,
)

D_MODEL = 1024
# Rotary width the graph is built against; must match the d_head it is
# compiled at (the token-example harness and HF parity both compile at
# d_head=16, the default for compile_to_onnx).
D_HEAD = 16
MAX_POSITIONS = 512


def create_network_parts(
    max_digits: int = 3,
) -> Tuple[Node, Embedding]:
    """Build the calculator graph and return (output_node, embedding).

    Parses "A op B\\n" where op is +, -, or *, then outputs result digits
    autoregressively. Subtraction can produce negative results (prefixed with "-").

    Three phases:
      1. Parse: detect operator, extract digit embeddings for A and B
      2. Compute: all three operations in scalar space, dispatch by operator
      3. Output: emit result digits left-to-right, then <eos>
    """
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["\n", "<bos>", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)

    # --- Phase 1: Parse operands and operator ---
    num_seq = NumericSequence(rope, embedding, max_digits)

    # Detect which operator token appears
    is_plus = equals_vector(embedding, embedding.get_embedding("+"))
    is_minus = equals_vector(embedding, embedding.get_embedding("-"))
    is_times = equals_vector(embedding, embedding.get_embedding("*"))
    is_operator = bool_any_true([is_plus, is_minus, is_times])
    is_equals = equals_vector(embedding, embedding.get_embedding("\n"))

    # Guard: only detect operators before "=" — output tokens like "-" must
    # not re-trigger operator signals during autoregressive decoding.
    has_seen_equals = get_prev_value(rope, is_equals, is_equals)
    before_equals = bool_not(has_seen_equals)
    is_operator_input = bool_all_true([is_operator, before_equals])

    # Latch which operator was seen (captured at the operator position,
    # carried forward to all later positions via attention).
    which_plus = get_prev_value(rope, is_plus, is_operator_input)
    which_minus = get_prev_value(rope, is_minus, is_operator_input)
    which_times = get_prev_value(rope, is_times, is_operator_input)

    # Extract operand digits (MSB first)
    first_num_digits = num_seq.get_digits_at_event(is_operator_input)
    second_num_digits = num_seq.get_digits_at_event(is_equals)

    # Convert digit embeddings to scalar numbers (shared across all operations)
    number_a = digits_to_number(embedding, first_num_digits)
    number_b = digits_to_number(embedding, second_num_digits)

    eos_embed = create_literal_value(embedding.get_embedding("<eos>"))
    minus_embed = create_literal_value(embedding.get_embedding("-"))

    # Multiplication produces up to 2*max_digits digits; all sequences
    # are padded to this length so switch() can select element-by-element.
    seq_len = 2 * max_digits + 2

    # --- Phase 2a: Addition (scalar-space) ---
    # Result can overflow by one digit (e.g. 999+999=1998)
    add_result = add(number_a, number_b)
    max_result_add = 2 * (10**max_digits - 1)
    add_digit_scalars = number_to_digit_scalars(
        add_result, max_digits + 1, max_result_add
    )
    add_digit_embeds = [scalar_to_embedding(d, embedding) for d in add_digit_scalars]
    add_seq = add_digit_embeds + [eos_embed] * (max_digits + 1)
    add_seq = remove_leading_0s(embedding, add_seq, max_removals=max_digits)

    # --- Phase 2b: Subtraction (scalar-space) ---
    # Result can be negative (e.g. 100-999=-899)
    sub_result = subtract(number_a, number_b)
    # compare returns true_level when inp > thresh; result > -0.5 means NOT negative
    is_negative = compare(sub_result, thresh=-0.5, true_level=-1.0, false_level=1.0)
    # |result| via ReLU: ReLU(x) + ReLU(-x) = |x|
    abs_result = relu_add(sub_result, negate(sub_result))
    max_abs = 10**max_digits - 1
    sub_digit_scalars = number_to_digit_scalars(abs_result, max_digits, max_abs)
    sub_digit_embeds = [scalar_to_embedding(d, embedding) for d in sub_digit_scalars]

    # Positive case: digits + eos padding
    sub_pos = sub_digit_embeds + [eos_embed] * (max_digits + 2)
    sub_pos = remove_leading_0s(embedding, sub_pos, max_removals=max_digits - 1)

    # Negative case: "-" then digits + eos padding, remove leading zeros from digits
    sub_abs_clean = remove_leading_0s(
        embedding,
        sub_digit_embeds + [eos_embed] * (max_digits + 1),
        max_removals=max_digits - 1,
    )
    sub_neg = [minus_embed] + sub_abs_clean[: seq_len - 1]

    # Select between positive and negative per position
    sub_seq = [select(is_negative, sub_neg[i], sub_pos[i]) for i in range(seq_len)]

    # --- Phase 2c: Multiplication (scalar-space via polarization) ---
    # a*b = ((a+b)^2 - (a-b)^2) / 4, using thermometer squaring
    mul_result = multiply_integers(number_a, number_b, max_value=10**max_digits - 1)
    max_result_mul = (10**max_digits - 1) ** 2  # e.g. 999^2=998001
    num_mul_digits = 2 * max_digits
    mul_digit_scalars = number_to_digit_scalars(
        mul_result, num_mul_digits, max_result_mul
    )
    mul_digit_embeds = [scalar_to_embedding(d, embedding) for d in mul_digit_scalars]
    mul_seq = mul_digit_embeds + [eos_embed, eos_embed]
    mul_seq = remove_leading_0s(embedding, mul_seq, max_removals=2 * max_digits - 1)

    # --- Phase 3: Dispatch by operator and output ---
    # switch() selects the result corresponding to whichever operator was used.
    result_digits = []
    for i in range(seq_len):
        result_digits.append(
            switch(
                [which_plus, which_minus, which_times],
                [add_seq[i], sub_seq[i], mul_seq[i]],
            )
        )

    output_node = output_sequence(
        rope,
        is_equals,
        result_digits,
        embedding.get_embedding(" "),
    )
    return output_node, embedding


def create_network(max_digits: int = 3) -> Unembedding:
    """Create a calculator network: parses "A op B\\n", outputs result autoregressively."""
    output_node, embedding = create_network_parts(max_digits)
    return create_unembedding(output_node, embedding)
