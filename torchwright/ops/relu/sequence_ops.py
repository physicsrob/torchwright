"""Token-stream parsing and autoregressive output for numeric sequences.

These patterns handle the common task of extracting multi-digit numbers from
a token stream and emitting computed results one token at a time. They are
mode-agnostic — they work with embedding-valued nodes regardless of whether
downstream computation uses embedding-space arithmetic (embedding_arithmetic)
or scalar-space arithmetic (scalar_encoding + arithmetic_ops).

Key components:
    NumericSequence — parses digit tokens into a sliding window of embeddings,
        captures the window when a delimiter (like "+" or "=") appears.
    output_sequence — gates a precomputed sequence of values for left-to-right
        autoregressive emission, starting when a trigger condition fires.
    check_is_digit — boolean predicate: is the current token a digit 0-9?
    remove_leading_0s — shifts a digit sequence left to drop leading zeros.
"""

from typing import List

import torch

from torchwright.graph import Node, Embedding, RopeConfig
from torchwright.ops.linear import sum_nodes
from torchwright.ops.attention_ops import attend_to_offset, get_prev_value
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.relu.logic_ops import (
    equals_vector,
    cond_gate,
    bool_not,
    bool_all_true,
)
from torchwright.ops.relu.map_select import map_to_table, select


def check_is_digit(embedding: Embedding) -> Node:
    """Check if the current embedding value is a digit (0-9).

    Args:
        embedding: Embedding input node to test.

    Returns:
        Boolean node: 1.0 if the token is a digit, -1.0 otherwise.
    """
    return map_to_table(
        inp=embedding,
        key_to_value={
            embedding.get_embedding(str(i)): torch.tensor([1.0]) for i in range(10)
        },
        default=torch.tensor([-1.0]),
    )


class NumericSequence:
    """Tracks a sliding window of digit embeddings across a token stream.

    As tokens arrive, this maintains a window of the last N digit embeddings.
    When a non-digit token appears (like "+" or "="), the window holds the
    complete number that preceded it. Use get_digits_at_event() to capture
    the window at a specific trigger position.

    Example: for the token stream "123+456=", with digits=3:
        At position "+": window = [embed("1"), embed("2"), embed("3")]
        At position "=": window = [embed("4"), embed("5"), embed("6")]

    Args:
        rope: RoPE config for the rotary offset / recency attention ops.
        embedding: Embedding table (must contain "0"-"9").
        digits: Number of digits to track in the sliding window.
    """

    def __init__(
        self,
        rope: RopeConfig,
        embedding: Embedding,
        digits: int,
    ):
        self.rope = rope
        zero_constant = create_literal_value(embedding.get_embedding("0"))
        is_digit = check_is_digit(embedding)

        # Detect the start of a new number: current token is a digit,
        # but the previous token was not.
        is_num_start = bool_all_true(
            [is_digit, bool_not(attend_to_offset(rope, is_digit))]
        )

        # Build the sliding window: current_digits[0] = current token,
        # current_digits[1] = token one position back, etc.
        # At number boundaries, reset earlier positions to "0".
        current_digits: List[Node] = [embedding]
        for i in range(digits - 1):
            current_digits.append(
                select(
                    cond=is_num_start,
                    true_node=zero_constant,
                    false_node=attend_to_offset(rope, current_digits[-1]),
                )
            )

        # Shift by one position so digit values are available at the
        # delimiter token (one step after the last digit).
        self.digit_values = [attend_to_offset(rope, digit) for digit in current_digits]

    def get_digits_at_event(self, termination_event: Node) -> List[Node]:
        """Capture the digit window at the position where termination_event fires.

        The captured values persist forward via attention -- once latched,
        they're available at all subsequent positions.

        Args:
            termination_event: Boolean node indicating the capture position.

        Returns:
            List of embedding-valued digit nodes, MSB-first.
        """
        return [
            get_prev_value(self.rope, digit, termination_event)
            for digit in reversed(self.digit_values)
        ]


def output_sequence(
    rope: RopeConfig,
    trigger_condition: Node,
    seq: List[Node],
    default_output: torch.Tensor,
):
    """Gate a sequence of values for left-to-right autoregressive emission.

    Before the trigger fires, outputs default_output. Once the trigger fires
    (at some position P), outputs seq[0] at position P, seq[1] at P+1, etc.

    This is the standard pattern for outputting a computed result one token
    at a time during autoregressive decoding.

    Args:
        rope: RoPE config for the rotary offset / recency attention ops.
        trigger_condition: Boolean node — emission starts when this is true.
        seq: List of embedding-valued nodes to emit in order.
        default_output: Tensor to output before the trigger fires.
    """
    # has_triggered is true at all positions from the trigger onward.  The
    # trigger fires once, so this is a single-match get_prev_value: the content
    # gate selects that one key regardless of distance (local recency only ranks
    # *among* matches), so the latch holds for the whole rollout.
    has_triggered = get_prev_value(rope, trigger_condition, trigger_condition)

    out_values = []
    for i, value in enumerate(seq):
        delta = -i
        # At position (trigger + i), this fires and gates seq[i] through.
        trigger = attend_to_offset(rope, trigger_condition, delta_pos=delta)
        out_values.append(cond_gate(trigger, value))

    return select(
        cond=has_triggered,
        true_node=sum_nodes(out_values),
        false_node=create_literal_value(default_output),
    )


def remove_leading_0s(
    embedding: Embedding, seq: List[Node], max_removals: int
) -> List[Node]:
    """Remove leading zeros from a digit sequence by shifting left.

    If seq[0] is embed("0"), shifts the entire sequence one position left
    and pads with the last element. Applies recursively up to max_removals
    times.

    Args:
        embedding: Embedding table (must contain "0").
        seq: List of embedding-valued digit nodes (MSB-first).
        max_removals: Maximum number of leading zeros to remove.
    """
    if max_removals == 0:
        return seq

    is_leading_zero = equals_vector(inp=seq[0], vector=embedding.get_embedding("0"))

    out = []
    seq = seq + [seq[-1]]
    for i, _ in enumerate(seq[:-1]):
        out.append(
            select(cond=is_leading_zero, true_node=seq[i + 1], false_node=seq[i])
        )
    return remove_leading_0s(embedding, out, max_removals - 1)
