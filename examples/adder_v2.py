"""3-digit adder using scalar-space arithmetic.

Instead of propagating carries digit-by-digit (see adder.py), this converts
digit embeddings into plain numbers, adds them as scalars, then converts back.

  "123" + "456"
  → [embed("1"), embed("2"), embed("3")]   input: 8D embedding per digit
  → 123.0                                  digits_to_number
  + 456.0                                  digits_to_number
  = 579.0                                  scalar addition — one Add node
  → [5.0, 7.0, 9.0]                       number_to_digit_scalars
  → [embed("5"), embed("7"), embed("9")]   scalar_to_embedding

The scalar↔digit conversions use thermometer coding: a ReLU-based technique
that counts how many thresholds an input crosses, like mercury rising in a
thermometer. See arithmetic_ops.thermometer_floor_div for the detailed
explanation.
"""

from torchwright.graph.embedding import Unembedding
from torchwright.ops.inout_nodes import (
    create_embedding,
    create_literal_value,
    create_rope_config,
    create_unembedding,
)
from torchwright.ops.linear import add
from torchwright.ops.swiglu.logic_ops import equals_vector
from torchwright.ops.swiglu.scalar_encoding import (
    digits_to_number,
    number_to_digit_scalars,
    scalar_to_embedding,
)
from torchwright.ops.swiglu.sequence_ops import (
    NumericSequence,
    output_sequence,
    remove_leading_0s,
)

# Rotary width the graph is built against; must match the d_head it is
# compiled at (the token-example harness compiles at d_head=16).
D_HEAD = 16
MAX_POSITIONS = 512


def create_network(max_digits: int = 3) -> Unembedding:
    r"""Create a v2 adder: parses "A+B\\n", outputs result digits autoregressively.

    Three phases:
      1. Parse: extract digit embeddings for A and B from the token stream
      2. Compute: embeddings → scalars → add → scalars → embeddings
      3. Output: emit result digits left-to-right, then <eos>
    """
    vocab = [
        *list(
            " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "!@#$%^&*()-+="
        ),
        "\n",
        "<bos>",
        "<eos>",
        "default",
    ]
    embedding = create_embedding(vocab=vocab)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)

    # --- Phase 1: Parse operand digits from the token stream ---
    num_seq = NumericSequence(rope, embedding, max_digits)

    is_end_of_first_num = equals_vector(
        inp=embedding, vector=embedding.get_embedding("+")
    )
    is_end_of_second_num = equals_vector(
        inp=embedding, vector=embedding.get_embedding("\n")
    )

    first_num_digits = num_seq.get_digits_at_event(is_end_of_first_num)
    second_num_digits = num_seq.get_digits_at_event(is_end_of_second_num)

    # --- Phase 2: Scalar-space addition ---
    # Convert digit embeddings to numbers, add, convert back.
    number_a = digits_to_number(embedding, first_num_digits)
    number_b = digits_to_number(embedding, second_num_digits)
    result = add(number_a, number_b)

    # Result can overflow by one digit (e.g. 999+999=1998)
    max_result = 2 * (10**max_digits - 1)
    num_output_digits = max_digits + 1
    digit_scalars = number_to_digit_scalars(result, num_output_digits, max_result)
    result_digits = [scalar_to_embedding(d, embedding) for d in digit_scalars]

    # --- Phase 3: Format and output ---
    result_digits = [
        *result_digits,
        create_literal_value(embedding.get_embedding("<eos>")),
    ]
    result_digits = remove_leading_0s(embedding, result_digits, max_removals=max_digits)

    return create_unembedding(
        output_sequence(
            rope,
            is_end_of_second_num,
            result_digits,
            embedding.get_embedding(" "),
        ),
        embedding,
    )
