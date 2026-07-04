"""Token-stream parsing and autoregressive output, on the swish machine.

Structurally identical to the relu module — these patterns compose
embedding-valued lookups, ±1 gates, and attention moves; the attention
ops are machine-neutral hardware, and every MLP ingredient
(map_to_table, select, cond_gate, the bool ops, equals_vector) is the
swiglu version, inheriting its entry.
"""

from typing import List

import torch

from torchwright.graph import Embedding, Node, RopeConfig

from torchwright.ops.attention_ops import attend_to_offset, get_prev_value
from torchwright.ops.linear import sum_nodes
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.swiglu.logic_ops import (
    bool_all_true,
    bool_not,
    cond_gate,
    equals_vector,
)
from torchwright.ops.swiglu.map_select import map_to_table, select


def check_is_digit(embedding: Embedding) -> Node:
    """Check if the current embedding value is a digit (0-9).

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

    See the relu twin for the mechanism; the structure is identical.

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

        # Sliding window; at number boundaries, reset earlier positions
        # to "0".
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
        # delimiter token.
        self.digit_values = [attend_to_offset(rope, digit) for digit in current_digits]

    def get_digits_at_event(self, termination_event: Node) -> List[Node]:
        """Capture the digit window at the position where termination_event
        fires; the captured values persist forward via attention.

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

    Before the trigger fires, outputs default_output. Once the trigger
    fires (at position P), outputs seq[0] at P, seq[1] at P+1, etc.  The
    losing cond_gates contribute exactly zero at clean conds on this
    machine, so the summed emission is the winning value alone.
    """
    has_triggered = get_prev_value(rope, trigger_condition, trigger_condition)

    out_values = []
    for i, value in enumerate(seq):
        delta = -i
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

    Applies recursively up to max_removals times.
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
