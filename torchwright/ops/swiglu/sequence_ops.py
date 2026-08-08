"""Token-stream parsing and autoregressive output, on the swish machine.

Structurally identical to the relu module — these patterns compose
embedding-valued lookups, ±1 gates, and attention moves; the attention
ops are machine-neutral hardware, and every MLP ingredient
(map_to_table, select, cond_gate, the bool ops, equals_vector) is the
swiglu version, inheriting its entry.  Exception: :class:`IndexedRegion`
is swiglu-only — no relu twin exists because no relu user does (the
calculators that need it are swiglu graphs).
"""

import math
from typing import cast

import torch

from torchwright.graph import Embedding, Linear, Node, RopeConfig, op_scope
from torchwright.graph.rope import require_full_rotary, rope_inv_freq
from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_to_offset,
    get_prev_value,
)
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.linear import (
    add,
    add_const,
    bool_to_01,
    concat,
    multiply_const,
    negate,
    sum_nodes,
)
from torchwright.ops.swiglu.arithmetic_ops import compare
from torchwright.ops.swiglu.logic_ops import (
    bool_all_true,
    bool_not,
    cond_gate,
    equals_vector,
)
from torchwright.ops.swiglu.map_select import (
    broadcast_select,
    in_range,
    map_to_table,
    select,
)
from torchwright.ops.swiglu.marker_count import count_since_marker


@op_scope
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

    @op_scope
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

        # Sliding window; at number boundaries, reset earlier positions
        # to "0".
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
        # delimiter token.
        self.digit_values = [attend_to_offset(rope, digit) for digit in current_digits]

    @op_scope
    def get_digits_at_event(self, termination_event: Node) -> list[Node]:
        """Capture the digit window at the position where termination_event fires.

        The captured values persist forward via attention.

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

# Cosine floor for the pointer gather's content match under the global
# rotation.  The dominance ordering above (member 1 > sentinel 0.5 > zero 0)
# is stated in bare dot products, but each content lane's Q·K contribution is
# really multiplied by cos(Δ·θ_lane), where Δ is the distance from the query
# to the key and θ_lane is the frequency of the slow plane that lane rides
# (place_on_slow_planes: lane c → plane d_head/2 - 1 - c, so the *last* lane —
# the sentinel's dedicated no-member lane — rides the fastest selected plane).
# Requiring cos(max_read_distance · θ) ≥ 0.75 on every lane keeps the ordering
# intact with margin: member-beats-sentinel needs worst-member cos 0.75 >
# _SENTINEL_FLOOR · best-sentinel cos 1 (a 1.5x margin — and in fact the
# sentinel sits at the marker, farther than every member, so its cosine on the
# shared lane never exceeds the member's); sentinel-beats-zero needs cos > 0.
# Measured against reference eval (d_head=32, base=5e5, prompt-length reads):
# the parse is numerically exact through max_len=12, degrades at 13 (worst
# window-slot error 6e-2), and misreads outright at 14; this floor admits
# max_len ≤ 10 there — comfortably inside the break.
_CONTENT_COS_FLOOR = 0.75


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
    * Every position where a :meth:`token_at` read is *consumed* must sit
      within ``max_read_distance`` of the marker.  The gather's content match
      rides slow rotary planes, so each key lane's dot product is attenuated
      by ``cos(Δ·θ_lane)`` at distance ``Δ``; past the guarded distance the
      dominance ordering (member beats sentinel beats zero) breaks and the
      read silently returns a wrong member or a blend.  The constructor
      enforces ``cos(max_read_distance · θ) ≥ 0.75`` on the fastest selected
      plane — a loud ``ValueError``, not silent drift.  Wide regions select
      fast planes (``max_len + 1`` lanes reach down to plane
      ``d_head/2 - max_len - 1``), so the practical remedy is raising
      ``rope.d_head``; shrinking ``max_len`` or consuming reads closer to
      the marker (e.g. latching them at a delimiter with
      :func:`~torchwright.ops.attention_ops.get_prev_value`, the calculator
      parse's pattern) also works.
    * ``rope.max_positions`` must be large enough for a healthy recency-lobe
      band (the production 512 is).  At small values like 64 the band's
      amplitudes collapse and ``get_prev_value`` refuses to build the latch —
      a loud ``ValueError`` at construction, not silent drift.

    Index scalars are marker-count-derived (accurate to well under ±0.5, not
    exact), so — deliberately — nothing here carries an integer claim; the
    one-hot encodings absorb the sub-integer error, the same idiom as the
    scratchpad calculator's column pointer.

    Args:
        rope: RoPE config for the counting and gather heads.  Must be full
            rotary — the latches ride the recency lobe, which has no
            partial-rotary form.
        embedding: the raw token embedding; :meth:`token_at` reads its rows.
        marker: ±1 boolean node, true exactly at the marker position.
        in_region: ±1 boolean node, true exactly at member positions; must be
            a clean ±1 boolean at *every* position (it gates the payload).
        max_len: maximum region length; sizes the count and the key width.
            The gather's content match places ``max_len + 1`` columns on slow
            rotary planes (they must fit ``rope.d_head // 2``), and how *fast*
            the fastest of those planes is bounds the usable read distance —
            see ``max_read_distance``.
        max_read_distance: upper bound on ``query position - marker
            position`` over every position where a :meth:`token_at` read is
            consumed (unconsumed reads stay bounded garbage as documented on
            :meth:`token_at`).  Raises ``ValueError`` when the rotation over
            this distance would attenuate the fastest content lane below the
            dominance floor (``_CONTENT_COS_FLOOR``).
        default: the embedding row an out-of-range :meth:`token_at` reads
            (e.g. ``embedding.get_embedding("0")`` for implicit zero-padding).
    """

    @op_scope
    def __init__(
        self,
        rope: RopeConfig,
        embedding: Embedding,
        *,
        marker: Node,
        in_region: Node,
        max_len: int,
        max_read_distance: int,
        default: torch.Tensor,
    ) -> None:
        assert len(marker) == 1, "marker must be a 1-D boolean"
        assert len(in_region) == 1, "in_region must be a 1-D boolean"
        assert max_len >= 1, "max_len must be >= 1"
        assert max_read_distance >= 1, "max_read_distance must be >= 1"
        assert default.numel() == len(embedding), "default must be an embedding row"
        require_full_rotary(cast("int", rope.d_rot), rope.d_head, "IndexedRegion")

        # Build-time dominance bound (see _CONTENT_COS_FLOOR): the gather's
        # max_len + 1 content lanes land on planes d_head/2 - 1 (slowest)
        # down to d_head/2 - max_len - 1 (fastest); cos is monotone on the
        # guarded range, so checking the fastest plane at the farthest
        # consumed distance covers every lane and every nearer key.
        half = rope.d_head // 2
        w_content = max_len + 1
        if w_content > half:
            raise ValueError(
                f"IndexedRegion: max_len={max_len} needs {w_content} content "
                f"lanes but only {half} rotary planes exist at "
                f"d_head={rope.d_head}; raise d_head."
            )
        theta_fast = float(rope_inv_freq(rope.d_head, rope.base)[half - w_content])
        angle = max_read_distance * theta_fast
        if angle > math.acos(_CONTENT_COS_FLOOR):
            raise ValueError(
                f"IndexedRegion: the pointer gather cannot honor its dominance "
                f"ordering at max_len={max_len}, "
                f"max_read_distance={max_read_distance} with d_head="
                f"{rope.d_head}, base={rope.base:g}: the fastest of its "
                f"{w_content} content lanes rides plane {half - w_content} "
                f"(θ={theta_fast:.3e}), and the rotation over the consumed "
                f"read distance turns cos(Δ·θ) = cos({angle:.3f}) = "
                f"{math.cos(angle):.3f} < {_CONTENT_COS_FLOOR}, so a member "
                f"key can lose to the default sentinel (or both to the "
                f"all-zero keys) and the read silently returns a wrong digit "
                f"or a blend.  Raise d_head (more slow planes), shrink "
                f"max_len, or consume the reads closer to the marker (latch "
                f"them at a delimiter with get_prev_value)."
            )

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

    @op_scope
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

    @op_scope
    def token_at(self, index: Node) -> Node:
        """The region's token at runtime ``index``, or ``default`` if out of range.

        ``index`` of 0 is the first member; the region's ``default`` is
        returned where ``index`` falls outside ``[0, region length)``.

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


@op_scope
def output_sequence(
    rope: RopeConfig,
    trigger_condition: Node,
    seq: list[Node],
    default_output: torch.Tensor,
) -> Node:
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


@op_scope
def remove_leading_0s(
    embedding: Embedding,
    seq: list[Node],
    max_removals: int,
    *,
    sign_cond: Node | None = None,
    sign_token: torch.Tensor | None = None,
) -> list[Node]:
    """Remove leading zeros from a digit sequence by shifting left.

    With ``k`` the length of the leading run of ``"0"`` tokens capped at
    ``max_removals``, output slot ``i`` holds ``seq[min(i + k, n - 1)]``:
    the sequence shifted left by ``k``, padded on the right with the last
    element.  (Same semantics as the retired chained-select form, which
    re-tested the shifted front once per removal and so cost two FFN
    stages per removal on the critical path.)

    Constant depth in ``max_removals`` — four FFN stages:

    1. An ``equals_vector`` zero flag per leading slot, all parallel.
    2. A ``compare`` per leading slot on a free prefix sum of the 0/1
       flags: slot ``i``'s prefix is all-"0" iff the sum is at least
       ``i + 1`` (threshold ``i + 0.5``).  The prefix flags are monotone
       (all-true then all-false), so the shift amount ``k`` is their
       free 0/1 sum — a near-integer scalar.
    3. ``in_range(k, k + 1, ...)`` — the shift-amount one-hot, computed
       once and shared by every output slot.
    4. A ``broadcast_select`` per output slot over that slot's shift
       candidates, plus the free cross-slot collapse.  Steps 3-4 are
       ``dynamic_extract`` with the index one-hot hoisted out of the
       per-slot loop; the single-call form would also recompute the
       one-hot per slot and the monolithic single-table form would put
       all ``(k_max+1)·n·d`` gated lanes in one unsplittable FFN.

    Margins: each prefix sum sits a half-integer from its threshold and
    ``k`` sits near-integer between the ``in_range`` centers, so both
    stages saturate.  Per-flag deviation accumulates linearly in the
    number of leading slots against those ~0.5 bands — comfortable at
    digit-sequence widths; a much wider window would want the prefix
    sums re-sharpened.

    **Signed variant** (``sign_cond`` + ``sign_token``, both or neither):
    when the ±1 cond ``sign_cond`` is true, the output is instead
    ``sign_token`` at slot 0 followed by the trimmed sequence — the
    sign character prepended, everything shifted right one slot (the
    slot that falls off the end is the caller's padding).  Costs no
    extra depth: the mux scalar becomes ``k + (n_shifts+1)·sign01``,
    so one ``in_range`` over twice the entries jointly keys (shift,
    sign) and each slot's candidate list doubles.  ``sign_cond`` must be a
    clean ±1 cond (``bool_to_01`` margins); it enters at stage 3, so it
    may resolve later than the sequence without deepening the trim.
    """
    if (sign_cond is None) != (sign_token is None):
        raise ValueError(
            "remove_leading_0s: sign_cond and sign_token must be passed together"
        )
    n = len(seq)
    # Shifting by n-1 or more pins every slot to the last element, so
    # larger removal budgets are no-ops — cap the candidate table there.
    n_shifts = min(max_removals, n - 1)
    if n_shifts <= 0:
        if sign_cond is not None:
            raise ValueError(
                "remove_leading_0s: the signed variant needs at least one "
                "removable slot (n >= 2 and max_removals >= 1)"
            )
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
    if sign_cond is not None:
        # Joint (shift, sign) key: the negative half of the entry range.
        shift = add(shift, multiply_const(bool_to_01(sign_cond), float(n_entries)))
    n_slots = n_entries * (2 if sign_cond is not None else 1)
    one_hot = in_range(shift, add_const(shift, 1.0), n_slots)
    zero_fill = create_literal_value(torch.zeros(d), name="remove_leading_0s_zero")
    collapse = torch.eye(d).repeat(n_slots, 1)
    sign = (
        create_literal_value(sign_token, name="remove_leading_0s_sign")
        if sign_token is not None
        else None
    )

    out: list[Node] = []
    for i in range(n):
        candidates = [seq[min(i + k, n - 1)] for k in range(n_entries)]
        if sign_cond is not None:
            # Negative half: the sign at slot 0, the trimmed sequence
            # shifted right one everywhere else.
            candidates += [
                cast("Node", sign) if i == 0 else seq[min(i - 1 + k, n - 1)]
                for k in range(n_entries)
            ]
        masked = broadcast_select(
            masks=one_hot,
            true_value=concat(candidates),
            false_value=zero_fill,
            n_slots=n_slots,
            d_fill=d,
        )
        # Losing slots are exactly zero at clean masks, so the free sum
        # degenerates to a copy of the selected candidate.
        out.append(Linear(masked, collapse, name="remove_leading_0s_sum"))
    return out
