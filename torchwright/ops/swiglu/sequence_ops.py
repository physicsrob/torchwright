"""Token-stream parsing and autoregressive output, on the swish machine.

Structurally identical to the relu module — these patterns compose
embedding-valued lookups, ±1 gates, and attention moves; the attention
ops are machine-neutral hardware, and every MLP ingredient
(map_to_table, select, cond_gate, the bool ops, equals_vector) is the
swiglu version, inheriting its entry.  Exception: :class:`IndexedRegion`
is swiglu-only — no relu twin exists because no relu user does (the
calculators that need it are swiglu graphs).
"""

from typing import List

import torch

from torchwright.graph import Embedding, Linear, Node, RopeConfig

from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_to_offset,
    get_prev_value,
)
from torchwright.ops.linear import (
    add,
    add_const,
    bool_to_01,
    concat,
    negate,
    sum_nodes,
)
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.swiglu.arithmetic_ops import compare
from torchwright.ops.swiglu.logic_ops import (
    bool_all_true,
    bool_not,
    cond_gate,
    equals_vector,
)
from torchwright.ops.swiglu.map_select import in_range, map_to_table, select
from torchwright.ops.swiglu.marker_count import count_since_marker


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


# Softmax sharpness for IndexedRegion's pointer gather.  Each region position's
# key is a 0/1 index one-hot, so a query that has a match sees a dot of exactly
# 1 against exactly one key and at most _SENTINEL_FLOOR against everything
# else; the gain only sets how hard the softmax locks onto that gap.  Same
# magnitude as the scratchpad calculator's answer gather — overwhelmingly
# hard, and with the match dot at exactly 1 there is no fp32 precision concern.
_INDEX_MATCH_GAIN = 2_000_000.0

# The default sentinel's dot with any in-range index query.  The marker
# position carries a key of [floor, ..., floor, 1] whose payload is the
# region's default: a query whose member exists out-dots it 1 vs floor (the
# member wins), a query whose member does not exist (index at or past the
# region's actual length) sees floor vs 0 (the sentinel wins and the read is
# the default), and the all-zero query of a negative index matches the
# sentinel's dedicated last lane outright.  Any value strictly inside (0, 1)
# works; 0.5 maximizes the smaller of the two margins.
_SENTINEL_FLOOR = 0.5


class IndexedRegion:
    """Constant-depth random access into a marker-delimited run of tokens.

    A *region* is a contiguous run of stream positions that starts immediately
    after a single marker position and whose membership is decided by a
    caller-supplied boolean (e.g. "is a digit and the operator has not appeared
    yet").  Each member position is stamped with a one-hot key of its index
    within the region (0 for the position right after the marker), computed
    from the bounded marker-distance count.  :meth:`token_at` then reads the
    region's token at a runtime-computed index with a single content-match
    attention; an index outside ``[0, region length)`` reads the region's
    ``default`` instead, via a sentinel key at the marker position whose
    payload is the default (see ``_SENTINEL_FLOOR``) — no post-attention
    boolean test, so an unconsumed read at a not-yet-meaningful position
    yields bounded garbage without tripping any ±1-cond assert.

    The depth of every read is constant in the region length — this is the
    random-access alternative to :class:`NumericSequence`, whose sliding
    window threads each position's value through the previous position's
    *computed* node and therefore unrolls into a chain as deep as the window
    is wide.

    Caller contract:

    * ``marker`` is true at exactly one stream position, and the region's
      members occupy the positions immediately after it with no gaps —
      member ``i`` sits at ``marker_pos + 1 + i``.  (The index keys are
      derived purely from marker distance, so a gap would mislabel every
      member after it.)
    * ``in_region`` is true exactly at member positions.  Its real job is
      killing lookalike positions elsewhere in the stream — e.g. digits of a
      *different* operand, or digits the model itself emits later — whose
      marker distance would otherwise land in ``[0, max_len)`` and collide
      with a member's key.
    * ``rope.max_positions`` must be large enough for a healthy recency-lobe
      band (the production 512 is).  At small values like 64 the band's
      amplitudes collapse and ``get_prev_value`` refuses to build the latch —
      a loud ``ValueError`` at construction, not silent drift.

    Index scalars are marker-count-derived (accurate to well under ±0.5, not
    exact), so — deliberately — nothing here carries an integer claim; the
    one-hot encodings absorb the sub-integer error, the same idiom as the
    scratchpad calculator's column pointer.

    Args:
        rope: RoPE config for the counting and gather heads.
        embedding: the raw token embedding; :meth:`token_at` reads its rows.
        marker: ±1 boolean node, true exactly at the marker position.
        in_region: ±1 boolean node, true exactly at member positions; must be
            a clean ±1 boolean at *every* position (it gates the payload).
        max_len: maximum region length; sizes the count and the key width.
            The gather's content match places ``max_len + 1`` columns on
            slow rotary planes, so it must fit ``rope.d_head // 2 - 1``.
        default: the embedding row an out-of-range :meth:`token_at` reads
            (e.g. ``embedding.get_embedding("0")`` for implicit zero-padding).
    """

    def __init__(
        self,
        rope: RopeConfig,
        embedding: Embedding,
        *,
        marker: Node,
        in_region: Node,
        max_len: int,
        default: torch.Tensor,
    ):
        assert len(marker) == 1, "marker must be a 1-D boolean"
        assert len(in_region) == 1, "in_region must be a 1-D boolean"
        assert max_len >= 1, "max_len must be >= 1"
        assert default.numel() == len(embedding), "default must be an embedding row"
        self.rope = rope
        self.max_len = max_len

        # Distance to the marker, valid (to well under ±0.5) out to the last
        # position that reads it: the event position one past the last member.
        seen_marker = get_prev_value(rope, marker, marker)
        self._count = count_since_marker(
            rope, seen_marker, bool_to_01(marker), max_gap=max_len + 1
        )

        # Each member's key: the one-hot of its own region index (distance
        # minus one), in the first max_len lanes.  Off-region positions are
        # hard-zeroed by the gate — range alone cannot exclude them, because
        # any position within max_len of the marker (the operator right after
        # an operand, say) has an in-range distance.  The marker position
        # itself carries the sentinel key: _SENTINEL_FLOOR in every index
        # lane plus a 1 in the dedicated last lane, so it loses to a present
        # member but beats an absent one (and catches the all-zero query).
        own_index = add_const(self._count, -1.0)
        own_onehot = bool_to_01(in_range(own_index, add_const(own_index, 1.0), max_len))
        member_keys = concat(
            [cond_gate(in_region, own_onehot), create_literal_value(torch.zeros(1))]
        )
        sentinel = torch.cat([torch.full((max_len,), _SENTINEL_FLOOR), torch.ones(1)])
        self._keys = add(member_keys, cond_gate(marker, create_literal_value(sentinel)))

        # The gather's payload: the member's token where in_region holds, the
        # region default everywhere else — in particular at the marker, so a
        # sentinel win reads the default.
        self._value = select(in_region, embedding, create_literal_value(default))

    def length_at(self, event: Node) -> Node:
        """The region's length, latched where ``event`` fires.

        ``event`` must fire at the position immediately after the region's
        last member (the operator after an operand, the newline after the
        second one) — the marker distance there is ``length + 1``.  The
        latched scalar persists to every later position; positions before
        the event read bounded garbage, as with any event latch.
        """
        assert len(event) == 1, "event must be a 1-D boolean"
        return add_const(get_prev_value(self.rope, self._count, event), -1.0)

    def token_at(self, index: Node) -> Node:
        """The region's token at runtime ``index`` (0 = first member), or the
        region's ``default`` where ``index`` falls outside ``[0, region
        length)``.

        One content-match attention.  The index becomes a one-hot query with
        a computed "out of ``[0, max_len)``" last lane; at most one member
        key can match it, and the marker's sentinel key catches every
        no-member case — a negative index, an index past ``max_len``, and an
        in-range index at or past the region's actual length.
        """
        assert len(index) == 1, "index must be a 1-D scalar"
        onehot = bool_to_01(in_range(index, add_const(index, 1.0), self.max_len))
        # 1 - sum(onehot): 1 exactly when no index lane is set.
        no_lane = add_const(
            Linear(onehot, -torch.ones(self.max_len, 1), name="region_no_lane"), 1.0
        )
        query = concat([onehot, no_lane])
        return attend_argmax_dot(
            self.rope,
            query_vector=query,
            key_vector=self._keys,
            value=self._value,
            match_gain=_INDEX_MATCH_GAIN,
        )


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

    The trigger must fire at most once per context.  See the relu twin for
    why the slot gating rides the near-marker step counter rather than
    per-slot ``attend_to_offset`` reads (out-of-range offset targets land
    on an arbitrary key and would leak deep slots into the sum).
    """
    has_triggered = get_prev_value(rope, trigger_condition, trigger_condition)

    steps_since = count_since_marker(
        rope,
        window_validity=has_triggered,
        marker_onehot=bool_to_01(trigger_condition),
        max_gap=len(seq) + 1,
    )

    out_values = []
    for i, value in enumerate(seq):
        # Fires iff steps_since == i: a ±0.5 band around the integer.
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
