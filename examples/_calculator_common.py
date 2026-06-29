"""Shared scaffolding for the two standalone calculator examples.

``calculator_simple`` and ``calculator_advanced`` are two *independent*
implementations of the same calculator: they differ only in their arithmetic —
the legible serial carry/borrow folds vs. the depth-optimized carry-lookahead /
carry-save versions.  Everything else is identical, so it lives here rather than
being duplicated or having one calculator import the other:

* the vocabulary and residual width (``CALC_VOCAB`` / ``D_MODEL``);
* the one-hot helpers both arithmetics build on (``_state`` / ``_slice`` and the
  carry/borrow state constants);
* **comparison** — a lexicographic fold of a less/equal/greater verdict.  It is
  not a depth target (subtraction's sign just needs the answer), so both
  variants use this one verbatim;
* the token-stream plumbing — sliding the digit window, latching operands,
  dispatching on the operator, emitting the result autoregressively, trimming
  leading zeros — quarantined behind :func:`parse_expression`,
  :func:`emit_result`, and :func:`build_calculator` so each calculator file
  reads as its three arithmetic algorithms in the foreground.

Each calculator supplies its own ``add_digit_seqs`` / ``subtract_digit_seqs`` /
``multiply_digit_seqs`` and hands them to :func:`build_calculator`; nothing here
imports either calculator, and neither calculator imports the other.
"""

from typing import Dict, List, Tuple

import torch

from torchwright.graph import Embedding, Linear, Node, RopeConfig
from torchwright.graph.embedding import bos_token
from torchwright.ops.arithmetic_ops import compare as _compare_scalar, concat
from torchwright.ops.attention_ops import get_prev_value
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_onehot_embedding,
    create_rope_config,
)
from torchwright.ops.logic_ops import (
    bool_all_true,
    bool_any_true,
    bool_not,
    equals_vector,
)
from torchwright.ops.map_select import select, switch
from torchwright.ops.onehot_table import onehot_lookup
from torchwright.ops.recency_heads import recency_rank_from_tokens
from torchwright.ops.sequence_ops import (
    NumericSequence,
    output_sequence,
    remove_leading_0s,
)

D_MODEL = 1024
# Rotary width the calculator graph is built against; must equal the d_head the
# token-example harness compiles at (whose default is 16).  32 leaves 16 slow
# planes — the widest content head in the family is the scratchpad multiply's
# pointer gather, whose width is 2n+1 (the answer-column one-hot plus the
# recency tiebreak); the scratchpad depth test builds up to max_digits=6, where
# that is 13, so the family needs d_head >= 26 (place_on_slow_planes runs at
# build time).  32 is the next clean even width with margin.
D_HEAD = 32
MAX_POSITIONS = 512

# Compact, calculator-only vocabulary: 10 digits, 3 operators, the newline that
# ends the input, a space (the pre-result placeholder), a BOS, a REF (the second
# always-visible marker the RoPE recency rank reads), and an EOS that
# pads / terminates the result.  18 tokens -> d_embed = 18 one-hot columns.
CALC_VOCAB = [str(d) for d in range(10)] + [
    "+",
    "-",
    "*",
    "\n",
    " ",
    bos_token,
    "<ref>",
    "<eos>",
]


# ---------------------------------------------------------------------------
# One-hot helpers shared by both arithmetics.
# ---------------------------------------------------------------------------

# State one-hots.  Carry and borrow are 2-state (no / yes); the lexicographic
# comparison verdict is 3-state.
_NO, _YES = 0, 1
_CARRY_W = 2

_LESS, _EQUAL, _GREATER = 0, 1, 2
_CMP_W = 3


def _state(index: int, width: int) -> torch.Tensor:
    """A width-``width`` one-hot with a 1 at ``index``."""
    v = torch.zeros(width)
    v[index] = 1.0
    return v


def _slice(node: Node, start: int, width: int, name: str = "slice") -> Node:
    """Take a ``width``-wide consecutive slice via a free ``Linear`` (no layer)."""
    proj = torch.zeros(len(node), width)
    for i in range(width):
        proj[start + i, i] = 1.0
    return Linear(node, proj, name=name)


# ---------------------------------------------------------------------------
# Comparison (shared verbatim — not a depth target).
# ---------------------------------------------------------------------------


def compare_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> Node:
    """Lexicographic comparison via an MSB-first fold of a 3-state verdict.

    The state is a one-hot over {less, equal, greater}.  Folding from the most
    significant digit down, the first non-equal digit decides the verdict and
    every later digit leaves it unchanged.  Returns a ±1 boolean: ``+1`` if
    ``seq1 >= seq2`` (equal counts as ``>=``), ``-1`` otherwise.
    """
    assert len(seq1) == len(seq2)

    # combine: key = concat([state, a, b]) -> next verdict state.
    combine_table: Dict[torch.Tensor, torch.Tensor] = {}
    for verdict in range(_CMP_W):
        for a in range(10):
            for b in range(10):
                key = torch.cat(
                    [
                        _state(verdict, _CMP_W),
                        embedding.get_embedding(str(a)),
                        embedding.get_embedding(str(b)),
                    ]
                )
                if verdict != _EQUAL:
                    nxt = verdict  # already decided at a more significant digit
                elif a > b:
                    nxt = _GREATER
                elif a < b:
                    nxt = _LESS
                else:
                    nxt = _EQUAL
                combine_table[key] = _state(nxt, _CMP_W)
    default_state = _state(_EQUAL, _CMP_W)

    state: Node = create_literal_value(_state(_EQUAL, _CMP_W))
    for a, b in zip(seq1, seq2):  # MSB-first
        key = concat([state, a, b])
        state = onehot_lookup(key, combine_table, default_state)

    # Collapse the 3-state verdict to a ±1 score, then sharpen to a clean ±1
    # boolean (a fuzzy final one-hot would otherwise yield a slightly off-±1
    # score that downstream selects amplify).
    score = onehot_lookup(
        state,
        {
            _state(_LESS, _CMP_W): torch.tensor([-1.0]),
            _state(_EQUAL, _CMP_W): torch.tensor([1.0]),
            _state(_GREATER, _CMP_W): torch.tensor([1.0]),
        },
        default=torch.tensor([1.0]),
    )
    return _compare_scalar(score, thresh=0.0, true_level=1.0, false_level=-1.0)


# ---------------------------------------------------------------------------
# Token-stream plumbing: parse -> compute -> emit.
# ---------------------------------------------------------------------------


def parse_expression(
    rope: RopeConfig,
    embedding: Embedding,
    max_digits: int,
    recency_rank: Node,
) -> Tuple[List[Node], List[Node], Node, Node, Node, Node]:
    """Parse ``"A op B\\n"`` from the token stream.

    Returns ``(first, second, is_plus, is_minus, is_times, saw_newline)``:
    the two operand digit windows (MSB-first), three latched ±1 flags for which
    operator appeared, and the ±1 newline trigger that ends the input and
    starts result emission.
    """
    num_seq = NumericSequence(rope, embedding, max_digits, recency_rank)

    is_plus = equals_vector(embedding, embedding.get_embedding("+"))
    is_minus = equals_vector(embedding, embedding.get_embedding("-"))
    is_times = equals_vector(embedding, embedding.get_embedding("*"))
    is_operator = bool_any_true([is_plus, is_minus, is_times])
    saw_newline = equals_vector(embedding, embedding.get_embedding("\n"))

    # Only treat an operator as such *before* the newline, so a "-" emitted as
    # a negative sign during decoding does not re-trigger operator parsing.
    seen_newline = get_prev_value(rope, saw_newline, saw_newline, recency_rank)
    is_input_operator = bool_all_true([is_operator, bool_not(seen_newline)])

    # Latch which operator was used (captured at the operator position, held
    # forward to every later position by attention).
    which_plus = get_prev_value(rope, is_plus, is_input_operator, recency_rank)
    which_minus = get_prev_value(rope, is_minus, is_input_operator, recency_rank)
    which_times = get_prev_value(rope, is_times, is_input_operator, recency_rank)

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
    rope: RopeConfig,
    embedding: Embedding,
    saw_newline: Node,
    result_digits: List[Node],
    recency_rank: Node,
) -> Node:
    """Emit ``result_digits`` autoregressively once the newline fires, printing
    a space at every position before then."""
    return output_sequence(
        rope, saw_newline, result_digits, embedding.get_embedding(" "), recency_rank
    )


def build_calculator(
    max_digits: int,
    *,
    add_digit_seqs,
    subtract_digit_seqs,
    multiply_digit_seqs,
) -> Tuple[Node, Embedding]:
    """Assemble the calculator graph from one variant's arithmetic.

    The three depth-differentiating algorithms — ``add_digit_seqs``,
    ``subtract_digit_seqs``, ``multiply_digit_seqs`` — come from the calling
    calculator (the legible ``calculator_simple`` or the depth-optimized
    ``calculator_advanced``).  Comparison is identical across variants, so this
    wires up the shared :func:`compare_digit_seqs` directly.  Returns
    ``(output_node, embedding)``.
    """
    embedding = create_onehot_embedding(CALC_VOCAB)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)

    # Bucket-2 recency rank from the <bos>/<ref> markers — replaces the old
    # position counter that drove "most recent" selection in get_prev_value /
    # NumericSequence / output_sequence.
    recency_rank = recency_rank_from_tokens(rope, embedding)

    first, second, is_plus, is_minus, is_times, saw_newline = parse_expression(
        rope, embedding, max_digits, recency_rank
    )

    # Multiplication is the widest result (2*max_digits digits); the others are
    # padded with <eos> to this length so the operator switch is per-position.
    seq_len = 2 * max_digits + 2
    zero = create_literal_value(embedding.get_embedding("0"))

    # --- Addition: pad each operand by one digit so the top carry has a home. ---
    add_digits = add_digit_seqs(embedding, [zero] + first, [zero] + second)
    add_seq = _format_result(embedding, add_digits, seq_len)

    # --- Subtraction: |A - B| by a borrow fold, sign from the comparison. ---
    a_ge_b = compare_digit_seqs(embedding, first, second)
    bigger = [select(a_ge_b, a, b) for a, b in zip(first, second)]
    smaller = [select(a_ge_b, b, a) for a, b in zip(first, second)]
    magnitude = _format_result(
        embedding, subtract_digit_seqs(embedding, bigger, smaller), seq_len
    )
    minus = create_literal_value(embedding.get_embedding("-"))
    negative = [minus] + magnitude[: seq_len - 1]
    sub_seq = [select(a_ge_b, magnitude[i], negative[i]) for i in range(seq_len)]

    # --- Multiplication: long multiplication, product fits in 2*max_digits. ---
    mul_seq = _format_result(
        embedding, multiply_digit_seqs(embedding, first, second), seq_len
    )

    # --- Dispatch by operator, then emit. ---
    result_digits = [
        switch([is_plus, is_minus, is_times], [add_seq[i], sub_seq[i], mul_seq[i]])
        for i in range(seq_len)
    ]
    output_node = emit_result(rope, embedding, saw_newline, result_digits, recency_rank)
    return output_node, embedding
