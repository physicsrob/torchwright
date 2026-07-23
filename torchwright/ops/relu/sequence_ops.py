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

import torch

from torchwright.graph import Embedding, Linear, Node, RopeConfig
from torchwright.ops.attention_ops import attend_to_offset, get_prev_value
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.linear import add_const, bool_to_01, concat, negate, sum_nodes
from torchwright.ops.relu.arithmetic_ops import compare
from torchwright.ops.relu.logic_ops import (
    bool_all_true,
    bool_not,
    cond_gate,
    equals_vector,
)
from torchwright.ops.relu.map_select import (
    broadcast_select,
    in_range,
    map_to_table,
    select,
)
from torchwright.ops.relu.marker_count import count_since_marker


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
    ) -> None:
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
        current_digits: list[Node] = [embedding]
        for _i in range(digits - 1):
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

    def get_digits_at_event(self, termination_event: Node) -> list[Node]:
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
    seq: list[Node],
    default_output: torch.Tensor,
):
    r"""Gate a sequence of values for left-to-right autoregressive emission.

    Before the trigger fires, outputs default_output. Once the trigger fires
    (at some position P), outputs seq[0] at position P, seq[1] at P+1, etc.

    This is the standard pattern for outputting a computed result one token
    at a time during autoregressive decoding.

    The trigger must fire at most once per context (the standard usage: a
    delimiter like "=" or "\\n" that appears exactly once).

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

    # Slot gating rides an exact step counter, NOT per-slot offset reads.
    # Gating slot i with attend_to_offset(trigger_condition, delta_pos=-i)
    # is broken when the trigger fires fewer than i positions after the
    # sequence start: the read targets a position before BOS, and with no
    # real key to match the sharp positional softmax locks onto an arbitrary
    # in-range key (out-of-range targets are a causal don't-care per
    # attend_to_offset — they must not be consumed).  A wrong key where the
    # trigger reads true spuriously sums a deep slot's value into the
    # emission.  The counter is the bucket-1 near-marker count
    # pos - trigger_pos, meaningful from the trigger onward; at earlier
    # positions it is bounded garbage, which the has_triggered select
    # below discards.
    steps_since = count_since_marker(
        rope,
        window_validity=has_triggered,
        marker_onehot=bool_to_01(trigger_condition),
        max_gap=len(seq) + 1,
    )

    out_values = []
    for i, value in enumerate(seq):
        # Fires iff steps_since == i: a ±0.5 band around the integer, wide
        # enough to absorb the count's sub-integer error.
        at_slot_i = bool_all_true(
            [
                compare(steps_since, thresh=i - 0.5),
                compare(negate(steps_since), thresh=-(i + 0.5)),
            ]
        )
        out_values.append(cond_gate(at_slot_i, value))

    return select(
        cond=has_triggered,
        true_node=sum_nodes(out_values),
        false_node=create_literal_value(default_output),
    )


def remove_leading_0s(
    embedding: Embedding, seq: list[Node], max_removals: int
) -> list[Node]:
    """Remove leading zeros from a digit sequence by shifting left.

    With ``k`` the length of the leading run of ``"0"`` tokens capped at
    ``max_removals``, output slot ``i`` holds ``seq[min(i + k, n - 1)]``:
    the sequence shifted left by ``k``, padded on the right with the last
    element.  (Same semantics as the retired chained-select form, which
    re-tested the shifted front once per removal and so cost two MLP
    sublayers per removal on the critical path.)

    Constant depth in ``max_removals`` — four sublayer stages:

    1. An ``equals_vector`` zero flag per leading slot, all parallel.
    2. A ``compare`` per leading slot on a free prefix sum of the 0/1
       flags: slot ``i``'s prefix is all-"0" iff the sum is at least
       ``i + 1`` (threshold ``i + 0.5``).  The prefix flags are monotone
       (all-true then all-false), so the shift amount ``k`` is their
       free 0/1 sum — a near-integer scalar.
    3. ``in_range(k, k + 1, ...)`` — the shift-amount one-hot, computed
       once and shared by every output slot.
    4. A ``broadcast_select`` per output slot over that slot's shift
       candidates, plus the free cross-slot collapse.  Steps 3–4 are
       ``dynamic_extract`` with the index one-hot hoisted out of the
       per-slot loop.

    Margins: each prefix sum sits a half-integer from its threshold and
    ``k`` sits near-integer between the ``in_range`` centers, so both
    stages saturate.  Per-flag deviation accumulates linearly in the
    number of leading slots against those ~0.5 bands — comfortable at
    digit-sequence widths; a much wider window would want the prefix
    sums re-sharpened.

    Args:
        embedding: Embedding table (must contain "0").
        seq: List of embedding-valued digit nodes (MSB-first), all the
            same width.
        max_removals: Maximum number of leading zeros to remove.
    """
    n = len(seq)
    # Shifting by n-1 or more pins every slot to the last element, so
    # larger removal budgets are no-ops — cap the candidate table there.
    n_shifts = min(max_removals, n - 1)
    if n_shifts <= 0:
        return seq

    d = len(seq[0])
    assert all(len(node) == d for node in seq)

    zero_vec = embedding.get_embedding("0")
    z01 = [
        bool_to_01(equals_vector(inp=seq[i], vector=zero_vec)) for i in range(n_shifts)
    ]
    prefix01 = [
        bool_to_01(compare(sum_nodes(z01[: i + 1]), thresh=i + 0.5))
        for i in range(n_shifts)
    ]
    shift = sum_nodes(prefix01)

    n_entries = n_shifts + 1
    one_hot = in_range(shift, add_const(shift, 1.0), n_entries)
    zero_fill = create_literal_value(torch.zeros(d), name="remove_leading_0s_zero")
    collapse = torch.eye(d).repeat(n_entries, 1)

    out: list[Node] = []
    for i in range(n):
        candidates = concat([seq[min(i + k, n - 1)] for k in range(n_entries)])
        masked = broadcast_select(
            masks=one_hot,
            true_value=candidates,
            false_value=zero_fill,
            n_slots=n_entries,
            d_fill=d,
        )
        # Losing slots are exactly zero at clean masks, so the free sum
        # degenerates to a copy of the selected candidate.
        out.append(Linear(masked, collapse, name="remove_leading_0s_sum"))
    return out
