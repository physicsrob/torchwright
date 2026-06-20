"""Calculator compiled into a transformer: ``+``, ``-``, ``*`` on integers up
to ``max_digits`` digits, parsed from ``"A op B\\n"`` and emitted digit by
digit.

Every operation is the same two ideas — an exhaustive **lookup table** over
digit combinations, and a **fold** that threads one piece of state along the
sequence: a carry for addition, a borrow for subtraction, a running
less/equal/greater verdict for comparison, a stack of single-digit partial
products for multiplication.  With one-hot embeddings each digit is a unit
vector, each lookup is exact integer counting, the compiled weight matrix *is*
the arithmetic table, and the identity unembed makes argmax-decode exact.  See
``torchwright/ops/onehot_table.py`` and ``torchwright/ops/onehot_arithmetic.py``.

The token-stream plumbing — sliding a digit window, latching operands,
emitting the result autoregressively, trimming leading zeros — is quarantined
behind :func:`parse_expression` and :func:`emit_result` so the body reads as
parse → compute → emit and the four algorithms stay in the foreground.

``calculator_v2`` is the scalar-space alternative (digits → a number,
thermometer-coded arithmetic); ``embedding_arithmetic`` is the older
spherical-code / ``map_to_table`` version of these same algorithms.
"""

from typing import List, Tuple

from torchwright.graph import Node, Embedding, PosEncoding
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_onehot_embedding,
    create_pos_encoding,
)
from torchwright.ops.logic_ops import (
    equals_vector,
    bool_not,
    bool_all_true,
    bool_any_true,
)
from torchwright.ops.map_select import select, switch
from torchwright.ops import onehot_arithmetic
from torchwright.ops.sequence_ops import (
    NumericSequence,
    output_sequence,
    remove_leading_0s,
)

D_MODEL = 1024

# Compact, calculator-only vocabulary: 10 digits, 3 operators, the newline that
# ends the input, a space (the pre-result placeholder), a BOS, and an EOS that
# pads / terminates the result.  17 tokens -> d_embed = 17 one-hot columns.
CALC_VOCAB = [str(d) for d in range(10)] + ["+", "-", "*", "\n", " ", "<bos", "<eos>"]


def parse_expression(
    pos_encoding: PosEncoding, embedding: Embedding, max_digits: int
) -> Tuple[List[Node], List[Node], Node, Node, Node, Node]:
    """Parse ``"A op B\\n"`` from the token stream.

    Returns ``(first, second, is_plus, is_minus, is_times, saw_newline)``:
    the two operand digit windows (MSB-first), three latched ±1 flags for which
    operator appeared, and the ±1 newline trigger that ends the input and
    starts result emission.
    """
    num_seq = NumericSequence(pos_encoding, embedding, max_digits)

    is_plus = equals_vector(embedding, embedding.get_embedding("+"))
    is_minus = equals_vector(embedding, embedding.get_embedding("-"))
    is_times = equals_vector(embedding, embedding.get_embedding("*"))
    is_operator = bool_any_true([is_plus, is_minus, is_times])
    saw_newline = equals_vector(embedding, embedding.get_embedding("\n"))

    # Only treat an operator as such *before* the newline, so a "-" emitted as
    # a negative sign during decoding does not re-trigger operator parsing.
    seen_newline = pos_encoding.get_prev_value(saw_newline, saw_newline)
    is_input_operator = bool_all_true([is_operator, bool_not(seen_newline)])

    # Latch which operator was used (captured at the operator position, held
    # forward to every later position by attention).
    which_plus = pos_encoding.get_prev_value(is_plus, is_input_operator)
    which_minus = pos_encoding.get_prev_value(is_minus, is_input_operator)
    which_times = pos_encoding.get_prev_value(is_times, is_input_operator)

    # First operand's window is complete at the operator; second's at newline.
    first = num_seq.get_digits_at_event(is_input_operator)
    second = num_seq.get_digits_at_event(saw_newline)
    return first, second, which_plus, which_minus, which_times, saw_newline


def _format_result(
    embedding: Embedding, digits: List[Node], seq_len: int
) -> List[Node]:
    """Pad a digit sequence to ``seq_len`` with ``<eos>``, then drop leading
    zeros (keeping at least one digit) so ``"007"`` prints as ``"7"``."""
    eos = create_literal_value(embedding.get_embedding("<eos>"))
    padded = digits + [eos] * (seq_len - len(digits))
    return remove_leading_0s(embedding, padded, max_removals=len(digits) - 1)


def emit_result(
    pos_encoding: PosEncoding,
    embedding: Embedding,
    saw_newline: Node,
    result_digits: List[Node],
) -> Node:
    """Emit ``result_digits`` autoregressively once the newline fires, printing
    a space at every position before then."""
    return output_sequence(
        pos_encoding, saw_newline, result_digits, embedding.get_embedding(" ")
    )


def build_calculator(arith, max_digits: int) -> Tuple[Node, PosEncoding, Embedding]:
    """Build the calculator graph over an arithmetic module ``arith``.

    ``arith`` supplies the four digit-sequence algorithms — ``add_digit_seqs``,
    ``subtract_digit_seqs``, ``compare_digit_seqs``, ``multiply_digit_seqs`` —
    so the same parse → compute → emit body wires up either the legible
    ``onehot_arithmetic`` (this file's :func:`create_network_parts`) or the
    depth-optimized ``onehot_arithmetic_fast`` (``calculator_advanced``).
    Returns ``(output_node, pos_encoding, embedding)``.
    """
    embedding = create_onehot_embedding(CALC_VOCAB)
    pos_encoding = create_pos_encoding()

    first, second, is_plus, is_minus, is_times, saw_newline = parse_expression(
        pos_encoding, embedding, max_digits
    )

    # Multiplication is the widest result (2*max_digits digits); the others are
    # padded with <eos> to this length so the operator switch is per-position.
    seq_len = 2 * max_digits + 2
    zero = create_literal_value(embedding.get_embedding("0"))

    # --- Addition: pad each operand by one digit so the top carry has a home. ---
    add_digits = arith.add_digit_seqs(embedding, [zero] + first, [zero] + second)
    add_seq = _format_result(embedding, add_digits, seq_len)

    # --- Subtraction: |A - B| by a borrow fold, sign from the comparison. ---
    a_ge_b = arith.compare_digit_seqs(embedding, first, second)
    bigger = [select(a_ge_b, a, b) for a, b in zip(first, second)]
    smaller = [select(a_ge_b, b, a) for a, b in zip(first, second)]
    magnitude = _format_result(
        embedding, arith.subtract_digit_seqs(embedding, bigger, smaller), seq_len
    )
    minus = create_literal_value(embedding.get_embedding("-"))
    negative = [minus] + magnitude[: seq_len - 1]
    sub_seq = [select(a_ge_b, magnitude[i], negative[i]) for i in range(seq_len)]

    # --- Multiplication: long multiplication, product fits in 2*max_digits. ---
    mul_seq = _format_result(
        embedding, arith.multiply_digit_seqs(embedding, first, second), seq_len
    )

    # --- Dispatch by operator, then emit. ---
    result_digits = [
        switch([is_plus, is_minus, is_times], [add_seq[i], sub_seq[i], mul_seq[i]])
        for i in range(seq_len)
    ]
    output_node = emit_result(pos_encoding, embedding, saw_newline, result_digits)
    return output_node, pos_encoding, embedding


def create_network_parts(
    max_digits: int = 3,
) -> Tuple[Node, PosEncoding, Embedding]:
    """The simple calculator: :func:`build_calculator` over ``onehot_arithmetic``."""
    return build_calculator(onehot_arithmetic, max_digits)
