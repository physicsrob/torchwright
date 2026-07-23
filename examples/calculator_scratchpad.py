"""Flat-depth, scratchpad ("thinking tokens") calculator.

``calculator_simple`` and ``calculator_advanced`` compute ``A op B`` (``+ - *``,
one-hot digits) with the serial carry/borrow/comparison work folded *inside the
graph*.  That fold is a chain of computed nodes, so the compiled model's depth
grows linearly with ``max_digits``.  (The operand parse used to be a second
``O(n)`` chain — the ``NumericSequence`` sliding window — stacked on top; the
shared ``parse_expression`` is now a constant-depth pointer-gather parse, so
what keeps those variants linear is their in-graph serial work: the
arithmetic folds, the shared comparison, and the leading-zero trim.)

This variant moves every serial recurrence off the *layer* axis and onto the
*decode-step* axis: the carry sweep, the borrow sweep, the comparison, and the
leading-zero normalization are written out as visible "thinking" tokens, one per
column, and each step reads the previous column's state back from the model's
**own emitted token stream**.  The compiled graph is one static, constant-depth
DAG; arbitrarily many digits cost more decode *steps*, not more *layers*.  It is
the first calculator whose whole model — parse + arithmetic + emit — is
flat-depth in ``max_digits``.

The load-bearing rule (why depth stays flat): **every cross-column read targets
a graph root** — the input ``embedding`` at a fixed offset, which during decode
holds a previously-emitted token — never a computed node from an earlier
column.  A recurrence threaded through a computed node would be unrolled by the
compiler into an ``O(n)`` chain (that is exactly ``calculator_simple``'s serial
fold).  Closing the recurrence through the token stream (emit → re-embed as
input → read back) resets the depth every step, and the argmax on each emitted
token re-quantizes the state to an exact one-hot so approximation noise cannot
accumulate across columns.  ``examples/fibonacci.py`` is the precedent that
reads its own emitted tokens; this generalizes that pattern to three ops.

Operands are variable-width — the same ``"A op B\\n"`` prompts the other two
calculators accept, no zero-padding required.  The parse is the shared
:func:`examples._calculator_common.parse_expression`: each prompt digit
carries a one-hot key of its index within its operand, and each window slot
reads the digit it needs with one pointer attention (an
``IndexedRegion`` gather, defaulting to ``"0"`` for the pad columns), so the
parse is ``O(1)`` depth at any operand width.

The *answer* is trimmed (no leading zeros, MSB-first), even though the column
arithmetic is fixed-width.  Each fixed-width answer digit is derived **once**
into a **scratch-digit** region (plain ``0``–``9``, MSB-first); a streamed
**normalization sweep** then counts the leading zeros (one count glyph per
answer column, reading each scratch digit's ``is_zero`` back at a static
offset), and each answer slot reads the count ``L`` and **points** at the
scratch digit ``L + t`` it needs via a single runtime-index attention
(``attend_argmax_dot`` with width-``N`` one-hot keys — a unique content match),
stopping at
``<eos>``.  The trim is therefore *more decode steps* (scratch + sweep) but no
extra depth and — crucially — only ~``n³`` width: the pointer read replaces the
old per-slot materialization of all ``N`` digits (the ~``n⁴`` term) with one
cheap attention off a region derived once.

Token protocol (rendered)::

    7*13\\n  <THINKING>⁰ ⁰ ... 9 1 ... ₀ ₁ ...</THINKING>91
                         │       │       │             └ trimmed answer (MSB-first) ┘
                         │       │       └ normalization sweep (MSB-first count) ┘
                         │       └ scratch digits (fixed-width answer, MSB-first) ┘
                         └ carry/borrow/verdict sweep (LSB-first) ┘

``THINKING_START`` (``<THINKING>``) opens the scratch region; the work region
holds the thinking glyphs — carry / borrow / verdict (superscripts, one per
column), then the scratch answer digits (plain ``0``–``9``), then the
leading-zero count (subscripts, one per answer column); ``RESULT``
(``</THINKING>``) separates scratch from answer; the answer region holds only
digit tokens (and ``-`` for a negative difference); then ``<eos>``.  The two
glyph families are deliberately disjoint so the value-readers fail safe: a count
reader that lands on a carry glyph returns its default ``0``, not a
wrong-but-valid number.
"""

from collections.abc import Callable
from typing import cast

import torch

from examples._calculator_common import (
    _CMP_W,
    _EQUAL,
    _GREATER,
    _LESS,
    CALC_VOCAB,
    D_HEAD,
    D_HIDDEN,
    D_MODEL,
    MAX_POSITIONS,
    _slice,
    _state,
    parse_expression,
)
from torchwright.graph import Embedding, Node, RopeConfig
from torchwright.graph.asserts import assert_integer
from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_to_offset,
    get_prev_value,
)
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_onehot_embedding,
    create_rope_config,
)
from torchwright.ops.linear import (
    add,
    add_const,
    bool_to_01,
    concat,
    negate,
    subtract,
    sum_nodes,
)
from torchwright.ops.swiglu.arithmetic_ops import compare
from torchwright.ops.swiglu.arithmetic_ops import min as min_node
from torchwright.ops.swiglu.logic_ops import (
    bool_all_true,
    bool_not,
    cond_gate,
    equals_vector,
)
from torchwright.ops.swiglu.map_select import in_range, select, switch
from torchwright.ops.swiglu.marker_count import count_since_marker
from torchwright.ops.swiglu.onehot_table import onehot_lookup

# Geometry notes (the constants themselves are the family-wide set in
# ``_calculator_common``):
#
# * ``D_HEAD`` — this module is why the family runs 64-wide heads: the
#   multiply answer gather's content one-hot spans ``2n+1`` columns and must
#   fit ``d_head/2`` slow rotary planes, so ``max_digits=10`` needs
#   ``d_head >= 42``.  64 also lands the deepest content column on plane 11
#   instead of plane 0 (``θ`` shrinks from ``base^0 = 1`` to
#   ``base^{-22/64} ≈ 0.011`` rad/token), easing — not yet verifying — the
#   transcript-length ``cos(Δ·θ)`` attenuation concern flagged at d_head=32
#   (decode tests still run max_digits=3).
#
# * ``D_MODEL`` — the dispatch computes all three streamed ops in parallel;
#   each carry/borrow column carries a wide one-hot total.  The leading-zero
#   trim derives each answer digit *once* into a scratch-digit region and each
#   answer slot reads the one it needs with a single pointer (runtime-index)
#   attention — a width-``N`` one-hot match — instead of re-materializing all
#   ``N`` column digits per slot.  That drops the per-slot ``N``-wide digit
#   table (the old ``dynamic_extract`` ~``N²`` width term) so the peak live
#   residual is back to ~``n³``.  Depth stays flat in ``max_digits``; the
#   width (and the decode-step count) is the axis that grows with operand
#   size.  At ``max_digits=3`` the dispatched graph schedules from ~2560
#   columns (the old ``dynamic_extract`` trim needed 8192).

__all__ = [
    "D_HEAD",
    "D_HIDDEN",
    "D_MODEL",
    "RESULT",
    "THINKING_START",
    "create_network_parts",
    "decode_steps",
    "scratch_vocab",
]


def decode_steps(max_digits: int) -> int:
    """**Worst-case** decode steps (emitted tokens, incl. ``THINKING_START`` /
    ``RESULT`` / ``<eos>``) for ``max_digits``-wide operands.

    This is a max over the three per-op transcript lengths, and the multiply
    transcript is the longest (``8·max_digits + 3``: open, ``2n`` carry glyphs,
    ``2n`` scratch digits, ``2n`` normalization glyphs, close, up to ``2n``
    answer digits, ``<eos>``) — an add or subtract query emits its own
    ``<eos>`` at its shorter transcript length (``4n+7`` / ``5n+4``) and
    generation stops there, so it is *not* the per-query emission count.  Use
    it as the ceiling: the decode-buffer size a rollout must provision, and
    the axis that grows with operand size — the serial work the legible
    calculators put on the *layer* axis lives here, on the *step* axis —
    while the layer depth stays flat.  The leading-zero trim adds one scratch
    digit *and* one normalization glyph per answer column (``+2N`` steps), the
    price of an MSB-first variable-length answer with no ``O(n)``-depth shift.
    """
    n = max_digits
    # open, carry, scratch, norm, close, ans, eos
    add_len = 1 + (n + 1) + (n + 1) + (n + 1) + 1 + (n + 1) + 1  # 4n + 7
    # open, verdict+borrow, scratch, norm, close, sign+n, eos
    sub_len = 1 + 2 * n + n + n + 1 + (n + 1) + 1  # 5n + 4
    # open, carry, scratch, norm, close, ans, eos
    mul_len = 1 + 2 * n + 2 * n + 2 * n + 1 + 2 * n + 1  # 8n + 3
    return max(add_len, sub_len, mul_len)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Two disjoint "thinking" glyph families, both the small-integer rendering of a
# streamed state, one token per value (value-indexed):
#   * superscripts ``⁰¹²`` — the carry / borrow / comparison verdict (counts
#     "up" as the sweep ripples);
#   * subscripts  ``₀₁₂`` — the running leading-zero count in the normalization
#     sweep (counts "down" the answer's leading zeros).
# Keeping them disjoint makes the value-readers fail safe: a count reader that
# lands on a carry glyph (or vice versa) matches no key and returns the default
# ``0`` rather than a plausible-but-wrong integer.
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
THINKING_START = "<THINKING>"
RESULT = "</THINKING>"


def _thinking_text(k: int) -> str:
    """Display string for the carry/borrow/verdict glyph carrying value ``k``
    (e.g. ``"¹² "``).
    """
    return str(k).translate(_SUP) + " "


def _count_text(k: int) -> str:
    """Display string for the normalization glyph carrying leading-zero count
    ``k`` (e.g. ``"₁₂ "``).
    """
    return str(k).translate(_SUB) + " "


def _n_thinking(max_digits: int) -> int:
    """Number of distinct streamed-state values (shared by both glyph families).

    A carry in the multiply sweep is the widest state: a column total is at most
    ``20·n`` (``2n`` digit contributions, each ``≤ 9``, plus an incoming carry
    ``≤ 2n``), so the carry-out is at most ``2n``.  The leading-zero count is at
    most the answer width, also ``2n`` (the all-zeros multiply ``2n``-digit
    result).  ``2n + 1`` tokens (value ``0 .. 2n``) therefore cover every
    reachable carry, borrow (``0/1``), verdict (``0/1/2``), and count.
    """
    return 2 * max_digits + 1


def scratch_vocab(max_digits: int) -> list[str]:
    """Full vocabulary for a ``max_digits``-wide scratchpad calculator.

    The base calculator alphabet (digits, ``+ - *``, newline, space, BOS, EOS)
    plus the two thinking alphabets (carry/borrow/verdict superscripts and
    normalization subscripts), both of which grow with ``max_digits``.
    ``d_embed`` grows with them; depth does not.
    """
    n_state = _n_thinking(max_digits)
    return (
        CALC_VOCAB
        + [THINKING_START]
        + [_thinking_text(k) for k in range(n_state)]
        + [_count_text(k) for k in range(n_state)]
        + [RESULT]
    )


# ---------------------------------------------------------------------------
# Small graph helpers
# ---------------------------------------------------------------------------


def _scalar(x: float) -> Node:
    return create_literal_value(torch.tensor([float(x)]))


def _digit_scalar(embedding: Embedding, digit_onehot: Node) -> Node:
    """One-hot digit embedding → its numeric value (a scalar ``0..9``).

    Integer by construction (table values 0..9, default 0.0), so the
    result carries the integer claim.
    """
    return assert_integer(
        onehot_lookup(
            digit_onehot,
            {
                embedding.get_embedding(str(d)): torch.tensor([float(d)])
                for d in range(10)
            },
            torch.tensor([0.0]),
        )
    )


def _thinking_value(embedding: Embedding, token: Node, n_state: int) -> Node:
    """Carry/borrow/verdict glyph embedding → the integer it carries (default
    ``0``).

    Non-superscript tokens (digits, normalization glyphs, ``THINKING_START``,
    ``RESULT``, …) map to ``0``, so a column whose preceding token is not a carry
    glyph reads carry-in ``0``.
    """
    return onehot_lookup(
        token,
        {
            embedding.get_embedding(_thinking_text(c)): torch.tensor([float(c)])
            for c in range(n_state)
        },
        torch.tensor([0.0]),
    )


def _count_value(embedding: Embedding, token: Node, n_state: int) -> Node:
    """Normalization glyph embedding → the leading-zero count it carries
    (default ``0``).

    The subscript twin of :func:`_thinking_value`: non-subscript tokens map to
    ``0``, so a stray read of a carry glyph (or ``RESULT``) reads count ``0``
    rather than a plausible-but-wrong integer.
    """
    return onehot_lookup(
        token,
        {
            embedding.get_embedding(_count_text(c)): torch.tensor([float(c)])
            for c in range(n_state)
        },
        torch.tensor([0.0]),
    )


# ---------------------------------------------------------------------------
# Seq-index layout: every cross-token read offset is derived from these named
# list-index boundaries, so the formulas stay legible after the normalization
# region was inserted between the carry sweep and the answer.
# ---------------------------------------------------------------------------


class _SeqLayout:
    """Named list-index boundaries of one op's emitted token list.

    The full list is ``[THINKING_START] + carry + scratch + norm + [RESULT] +
    answer + [<eos>]`` (see :func:`_assemble`).  The scratch region holds the
    ``N`` fixed-width answer digits (MSB-first, plain ``0``–``9``), derived once;
    the normalization sweep and the answer both read *it* (no re-derivation).
    Every cross-token read offset is the target list-index minus the emitting
    list-index plus one (the autoregressive shift — see :func:`_read_offset`),
    so naming the boundaries is enough to write every offset.
    """

    def __init__(
        self,
        n_carry: int,
        n_scratch: int,
        n_norm: int,
        n_answer: int,
        n_state: int,
    ):
        self.carry_start = 1  # first carry/verdict glyph (THINKING_START at 0)
        self.scratch_start = 1 + n_carry  # first scratch digit
        self.norm_start = self.scratch_start + n_scratch  # first normalization glyph
        self.norm_last = self.norm_start + n_norm - 1  # carries the total count
        self.result_idx = self.norm_start + n_norm  # RESULT
        self.answer_start = self.result_idx + 1  # first answer slot
        self.n_answer = n_answer
        self.n_state = n_state


def _read_offset(target_idx: int, here_idx: int) -> int:
    """``delta_pos`` for ``attend_to_offset(embedding, ...)`` so the token
    emitted at list-index ``here_idx`` reads the token emitted at list-index
    ``target_idx``.

    The ``+1`` is the autoregressive shift: the raw input ``embedding`` at a
    decode position holds the *previously* emitted token, so reading list-index
    ``t`` from position ``h`` (which emits list element ``h``) means attending
    ``t - (h - 1) = t - h + 1`` positions back.
    """
    return target_idx - here_idx + 1


def _steps_since_newline(
    rope: RopeConfig,
    embedding: Embedding,
    *,
    max_gap: int,
) -> Node:
    """``pos - newline_pos`` (the decode-step index), via the bucket-1
    near-marker count.

    The RoPE-native replacement for the old
    ``pos.get_position_scalar() - pos.get_prev_value(pos_scalar, saw_newline)``:
    there is no absolute position counter under RoPE, but ``count_since_marker``
    recovers the bounded gap to a recent marker token directly.  The newline is
    the marker (exactly one per transcript): ``marker_onehot`` is 1 at the
    newline and 0 elsewhere, and ``window_validity`` is +1 at and after the
    newline (the "has the newline fired yet" latch).  The result is meaningful
    only at and after the newline; earlier positions return bounded garbage,
    which every consumer here gates away (``has_triggered`` / an out-of-range
    column one-hot).  ``max_gap`` bounds the largest step index read — the full
    padded transcript length.
    """
    embed = embedding.get_embedding
    saw_newline = equals_vector(embedding, embed("\n"))
    marker_onehot = bool_to_01(saw_newline)  # 1 at the newline, 0 elsewhere
    window_validity = get_prev_value(rope, saw_newline, saw_newline)
    return count_since_marker(rope, window_validity, marker_onehot, max_gap=max_gap)


# ---------------------------------------------------------------------------
# The shared streaming skeleton: a per-column total, then a single-state carry
# sweep emitting one thinking token per column that reads the previous column's
# carry from the emitted prefix.  Addition and multiplication differ only in how
# the column total ``T[k]`` is formed (same spirit as ``calculator_simple``:
# "differ only in the table").  A per-column digit-derivation closure (the digit
# the answer would emit at MSB position ``m``) is shared by the normalization
# sweep (which uses only ``is_zero`` of it) and the trimmed answer.
# ---------------------------------------------------------------------------

# A column digit is derived MSB-first at a given *list-index*: ``digit(here, m)``
# returns the answer's digit embedding at MSB position ``m`` as seen from the
# token emitted at list-index ``here`` (the reads back into the carry/borrow
# sweep are relative to ``here``).
DigitFn = Callable[[int, int], Node]


def _emit_from_total(
    embedding: Embedding,
    total: Node,
    table: dict[torch.Tensor, torch.Tensor],
    default: torch.Tensor,
    W: int,
) -> Node:
    """Turn a column total (an integer ``0..W-1``) into a one-hot index, then
    read off the emitted token via ``table`` — the carry-sweep idiom shared with
    ``calculator_simple``'s multiply: ``in_range`` makes the one-hot,
    ``onehot_lookup`` reads the table.
    """
    onehot = bool_to_01(in_range(total, add_const(total, 1.0), W))
    return onehot_lookup(onehot, table, default)


def _carry_sweep(embedding: Embedding, T: list[Node], W: int, n: int) -> list[Node]:
    """Carry sweep over LSB-first column totals ``T`` → one thinking glyph per
    column (each carrying that column's carry-out).

    Column ``k`` forms ``total = T[k] + carry_in`` and emits its carry-out
    (``total // 10``); the carry-in is read from the immediately-preceding
    emitted token (the raw embedding at decode time); column 0 starts from a
    literal 0.  ``W`` is one past the largest possible column total (carry-in
    included).
    """
    embed = embedding.get_embedding
    n_state = _n_thinking(n)
    carry_table = {_state(t, W): embed(_thinking_text(t // 10)) for t in range(W)}
    carry_default = embed(_thinking_text(0))
    prev_carry = _thinking_value(embedding, embedding, n_state)  # offset-0 read
    return [
        _emit_from_total(
            embedding,
            add(T[k], _scalar(0.0) if k == 0 else prev_carry),
            carry_table,
            carry_default,
            W,
        )
        for k in range(len(T))
    ]


def _carry_digit_fn(
    rope: RopeConfig,
    embedding: Embedding,
    layout: _SeqLayout,
    T: list[Node],
    W: int,
    n: int,
) -> DigitFn:
    """Closure deriving the addition/multiplication answer digit at MSB position
    ``m`` (place-value column ``kc = len(T)-1-m``): ``total = T[kc] + carry_in``,
    where the carry-in is the carry-out of column ``kc-1`` read from the carry
    sweep at a statically-known offset.  Only ``total % 10`` (the digit) is
    returned; the carry half lives in :func:`_carry_sweep`.
    """
    embed = embedding.get_embedding
    n_state = _n_thinking(n)
    num_cols = len(T)
    digit_table = {_state(t, W): embed(str(t % 10)) for t in range(W)}
    digit_default = embed("0")

    def digit(here: int, m: int) -> Node:
        kc = num_cols - 1 - m
        if kc == 0:
            cin = _scalar(0.0)
        else:
            target = layout.carry_start + (kc - 1)  # carry-out of column kc-1
            token = attend_to_offset(
                rope, embedding, delta_pos=_read_offset(target, here)
            )
            cin = _thinking_value(embedding, token, n_state)
        return _emit_from_total(
            embedding, add(T[kc], cin), digit_table, digit_default, W
        )

    return digit


def _stream_normalize(
    rope: RopeConfig,
    embedding: Embedding,
    layout: _SeqLayout,
    N: int,
) -> list[Node]:
    """Normalization sweep: MSB-first, one count glyph per answer column,
    threading a single running leading-zero count.

    The trick that lets one integer carry the whole state (no separate "still in
    the run" flag): the count can equal the number of columns processed *only if
    every one was zero*, so "still scanning leading zeros at column ``m``" is
    exactly ``prev_count == m`` — a compare against a static threshold (the count
    can never exceed ``m``).  Per column ``m`` (MSB index): read the previous
    count from the immediately-preceding emitted glyph (``m == 0`` starts at 0),
    read this column's answer digit back from the **scratch region** at a static
    offset (only its ``is_zero`` is used), and increment the count iff still in
    the run *and* the digit is zero.  The scratch digit was derived once (the
    answer reads the same region), so the normalization sweep no longer
    re-materializes the column arithmetic.
    """
    embed = embedding.get_embedding
    n_state = layout.n_state
    count_table = {_state(k, n_state): embed(_count_text(k)) for k in range(n_state)}
    count_default = embed(_count_text(0))

    glyphs: list[Node] = []
    for m in range(N):
        here = layout.norm_start + m
        prev = _scalar(0.0) if m == 0 else _count_value(embedding, embedding, n_state)
        scratch_token = attend_to_offset(
            rope, embedding, delta_pos=_read_offset(layout.scratch_start + m, here)
        )
        is_zero = equals_vector(inp=scratch_token, vector=embed("0"))
        in_run = compare(prev, thresh=m - 0.5)  # prev == m (count never exceeds m)
        still_zero = bool_all_true([in_run, is_zero])
        # A count of leading zeros — genuinely integer (the ±1 bools and
        # the count table are exact up to op noise).
        this_count = assert_integer(add(prev, bool_to_01(still_zero)))
        glyphs.append(
            _emit_from_total(embedding, this_count, count_table, count_default, n_state)
        )
    return glyphs


def _read_count(
    rope: RopeConfig, embedding: Embedding, layout: _SeqLayout, here: int
) -> Node:
    """The total leading-zero count ``L`` carried by the last normalization
    glyph (``layout.norm_last``), read from the token emitted at list-index
    ``here``.
    """
    token = attend_to_offset(
        rope, embedding, delta_pos=_read_offset(layout.norm_last, here)
    )
    return _count_value(embedding, token, layout.n_state)


# The pointer-gather attention's content-match coefficient.  Each scratch column
# is written exactly once, so the width-``N`` one-hot query has a *unique*
# matching key — this is a pure content gather (``attend_argmax_dot``), no
# recency tiebreak (docs/rope_port_plan.md Phase 6: recency is the intrinsic
# local lobe, used only where multiple keys match; here at most one does).
# ``match_gain`` just sets the softmax sharpness of the one-hot argmax (match dot
# 1 vs no-match 0); ``2e6`` is overwhelmingly hard and the match dot is exactly
# 1, so there is no fp32 precision concern at this magnitude.
_MATCH_GAIN = 2_000_000.0


def _col_onehot(steps_since: Node, layout: _SeqLayout, N: int) -> Node:
    """Width-``N`` one-hot of *this position's* scratch-digit column index.

    At the input position holding scratch digit for column ``i`` the one-hot is
    ``e_i``; everywhere else the column scalar lands outside ``[0, N)`` so the
    one-hot is all-zero (a key that matches no answer-slot query).  The column
    index is the decode-step index measured from the scratch region's start:
    ``steps_since`` is ``pos − newline_pos`` (the bucket-1 near-marker count),
    and scratch digit ``i`` is the input at ``newline_pos + scratch_start + i +
    1`` (list element ``k`` is emitted at ``newline_pos + k`` and re-read one
    position later), so ``col = steps_since − (scratch_start + 1)``.

    Deliberately **no** ``assert_integer`` on ``steps_since``/``col_scalar``:
    ``count_since_marker`` is a reciprocal approximation, accurate only to
    well under ±0.5 (not 1e-3), and pre-newline positions carry bounded
    non-integer garbage.  The subgraph sourced here therefore stays out of
    the univariate collapse — by design, not by omission.
    """
    col_scalar = add_const(steps_since, -float(layout.scratch_start + 1))
    return bool_to_01(in_range(col_scalar, add_const(col_scalar, 1.0), N))


def _emit_gathered_answer(
    rope: RopeConfig,
    embedding: Embedding,
    layout: _SeqLayout,
    col_onehot: Node,
    N: int,
    *,
    sign_at: Callable[[int], Node] | None = None,
) -> list[Node]:
    """The trimmed, MSB-first answer, one slot per ``layout.n_answer``.

    Each slot reads the leading-zero count ``L`` (capped to ``min(L, N-1)`` so an
    all-zeros result still prints one ``0``) and **points** at the scratch digit
    it needs: ``idx = L + t``, a width-``N`` one-hot query, gathered from the
    scratch region with a single ``attend_argmax_dot`` (each scratch column is
    written exactly once, so the one-hot query has a unique matching key — a
    content gather, no recency tiebreak; the value is the emitted token).  The
    digit was derived once into scratch; the slot just reads it — no per-slot
    materialization of all ``N`` digits.  Past the last digit (``idx >= N``) the
    slot emits ``<eos>``; the decode loop stops there, so the *emitted* answer is
    genuinely variable-length.

    ``sign_at`` (subtraction only): a callable ``here -> a_ge_b`` (the ±1
    ``A>=B`` verdict).  Slot 0 becomes ``select(a_ge_b, first_digit, '-')`` and
    every slot's index is shifted back by ``bool_to_01(not a_ge_b)`` so the ``-``
    occupies slot 0 and the magnitude follows from slot 1.
    """
    embed = embedding.get_embedding
    eos = create_literal_value(embed("<eos>"))
    minus = create_literal_value(embed("-"))

    answer: list[Node] = []
    for t in range(layout.n_answer):
        here = layout.answer_start + t
        L_capped = min_node(
            _read_count(rope, embedding, layout, here), _scalar(float(N - 1))
        )
        idx = add_const(L_capped, float(t))
        ge = None
        if sign_at is not None:
            ge = sign_at(here)
            idx = subtract(idx, bool_to_01(bool_not(ge)))  # -1 slot when negative
        query = bool_to_01(in_range(idx, add_const(idx, 1.0), N))
        picked = attend_argmax_dot(
            rope,
            query_vector=query,
            key_vector=col_onehot,
            value=embedding,
            match_gain=_MATCH_GAIN,
        )
        if t == 0 and sign_at is not None:
            answer.append(select(cast("Node", ge), picked, minus))
        else:
            answer.append(select(compare(idx, thresh=N - 0.5), eos, picked))
    return answer


def _build_carry_op(
    rope: RopeConfig,
    embedding: Embedding,
    T: list[Node],
    W: int,
    n: int,
    steps_since: Node,
) -> tuple[list[Node], list[Node]]:
    """Carry-based op (addition / multiplication): carry sweep, scratch digits,
    normalization sweep, pointer-gathered answer.  Returns ``(thinking, answer)``
    where ``thinking`` is ``carry ++ scratch ++ norm``.
    """
    N = len(T)
    # Column totals are sums of digit-table lookups — genuinely integer.
    T = [assert_integer(t) for t in T]
    carry = _carry_sweep(embedding, T, W, n)
    layout = _SeqLayout(
        n_carry=len(carry), n_scratch=N, n_norm=N, n_answer=N, n_state=_n_thinking(n)
    )
    digit = _carry_digit_fn(rope, embedding, layout, T, W, n)
    scratch = [digit(layout.scratch_start + m, m) for m in range(N)]
    norm = _stream_normalize(rope, embedding, layout, N)
    col_onehot = _col_onehot(steps_since, layout, N)
    answer = _emit_gathered_answer(rope, embedding, layout, col_onehot, N)
    return carry + scratch + norm, answer


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


def _add_op(
    rope: RopeConfig,
    embedding: Embedding,
    A: list[Node],
    B: list[Node],
    n: int,
    steps_since: Node,
) -> tuple[list[Node], list[Node]]:
    """Streamed addition.  ``num_cols = n + 1`` columns give the top carry a home;
    the column total is just ``a_k + b_k``.
    """
    a_scalar = [_digit_scalar(embedding, d) for d in A]  # MSB-first
    b_scalar = [_digit_scalar(embedding, d) for d in B]

    # Place value k (LSB-first) → MSB-first index n-1-k; column n is the pad.
    T = [add(a_scalar[n - 1 - k], b_scalar[n - 1 - k]) for k in range(n)] + [
        _scalar(0.0)
    ]
    W = 20  # a + b + carry_in ≤ 9 + 9 + 1 = 19
    return _build_carry_op(rope, embedding, T, W, n, steps_since)


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def _mul_op(
    rope: RopeConfig,
    embedding: Embedding,
    A: list[Node],
    B: list[Node],
    n: int,
    steps_since: Node,
) -> tuple[list[Node], list[Node]]:
    """Streamed multiplication.  The ``n²`` digit products and the per-column
    number sums are ``calculator_simple``'s multiply steps 1–2 verbatim (all
    ``O(1)`` depth); only the serial carry loop is replaced by the streamed
    sweep.  ``num_cols = 2n``; a column total plus carry never exceeds ``20n``.
    """
    embed = embedding.get_embedding

    # Step 1: the times table — two one-hot digits → (tens, ones) as numbers.
    product_table = {
        torch.cat([embed(str(a)), embed(str(b))]): torch.tensor(
            [float(a * b // 10), float(a * b % 10)]
        )
        for a in range(10)
        for b in range(10)
    }
    default_product = torch.tensor([0.0, 0.0])

    # Steps 1–2: drop each product's digits into their place-value columns and
    # collect the (free) per-column number contributions.
    columns: list[list[Node]] = [[] for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            product = onehot_lookup(
                concat([A[i], B[j]]), product_table, default_product
            )
            place = (n - 1 - i) + (n - 1 - j)  # place value of the ones digit
            columns[place].append(_slice(product, 1, 1, name="product_ones"))
            columns[place + 1].append(_slice(product, 0, 1, name="product_tens"))

    T = [sum_nodes(col) if col else _scalar(0.0) for col in columns]
    W = 20 * n + 1
    return _build_carry_op(rope, embedding, T, W, n, steps_since)


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


def _sub_op(
    rope: RopeConfig,
    embedding: Embedding,
    A: list[Node],
    B: list[Node],
    n: int,
    steps_since: Node,
) -> tuple[list[Node], list[Node]]:
    """Streamed subtraction.  Three streamed thinking phases precede the answer:

    1. **comparison** (MSB-first) — a 3-state less/equal/greater verdict folded
       one column at a time, each column reading the prior verdict from the
       previous emitted token; the last (LSB) verdict is the overall ``A ≥ B``.
    2. **borrow sweep** (LSB-first) — ``|A − B|`` by a borrow fold over
       ``bigger_k = (a_k if A≥B else b_k)`` / ``smaller_k``.
    3. **normalization** (MSB-first) — the leading-zero count of the magnitude.

    The answer is ``n+1`` MSB-first slots: a leading sign slot (``-`` if
    ``A < B``) then the trimmed magnitude.  Verdict, borrow, and count state are
    all read back from the emitted prefix at statically-known offsets.
    """
    embed = embedding.get_embedding
    n_state = _n_thinking(n)
    W = 20  # shifted column total (bigger - smaller - borrow_in + 10) ∈ 0..19

    a_scalar = [_digit_scalar(embedding, d) for d in A]  # MSB-first
    b_scalar = [_digit_scalar(embedding, d) for d in B]

    # --- comparison phase: MSB-first 3-state verdict fold — the serial form
    # the graph calculators retired when compare_digit_seqs went
    # constant-depth; here the recurrence rides the decode axis, where
    # serial is the point. ---
    combine_table: dict[torch.Tensor, torch.Tensor] = {}
    nxt: int | Node
    for v in range(_CMP_W):
        for a in range(10):
            for b in range(10):
                key = torch.cat([_state(v, _CMP_W), embed(str(a)), embed(str(b))])
                if v != _EQUAL:
                    nxt = v  # already decided at a more significant digit
                elif a > b:
                    nxt = _GREATER
                elif a < b:
                    nxt = _LESS
                else:
                    nxt = _EQUAL
                combine_table[key] = _state(nxt, _CMP_W)
    cmp_default = _state(_EQUAL, _CMP_W)
    verdict_to_token = {
        _state(v, _CMP_W): embed(_thinking_text(v)) for v in range(_CMP_W)
    }
    token_to_verdict = {
        embed(_thinking_text(v)): _state(v, _CMP_W) for v in range(_CMP_W)
    }

    prev_verdict = onehot_lookup(embedding, token_to_verdict, cmp_default)  # offset 0
    verdict_tok = []
    for k in range(n):
        nxt = onehot_lookup(
            concat([prev_verdict, A[k], B[k]]), combine_table, cmp_default
        )
        verdict_tok.append(
            onehot_lookup(nxt, verdict_to_token, embed(_thinking_text(_EQUAL)))
        )

    # The thinking region is verdict (n) ++ borrow (n) ++ scratch (n) ++ norm
    # (n); the answer is the sign slot ++ n magnitude slots.  Name the boundaries
    # once.  (verdict ++ borrow occupy the layout's "carry" span: 2n glyphs.)
    layout = _SeqLayout(
        n_carry=2 * n, n_scratch=n, n_norm=n, n_answer=n + 1, n_state=n_state
    )
    verdict_last = layout.carry_start + n - 1  # final verdict's list-index (= seq[n])
    borrow_start = layout.carry_start + n  # borrow[k] at borrow_start + k

    # Read the final verdict (at ``verdict_last``) from list-index ``here``,
    # sharpened to a clean ±1 A≥B boolean (less → −1, equal/greater → +1).
    def a_ge_b_at(here: int) -> Node:
        token = attend_to_offset(
            rope, embedding, delta_pos=_read_offset(verdict_last, here)
        )
        score = onehot_lookup(
            token,
            {
                embed(_thinking_text(_LESS)): torch.tensor([-1.0]),
                embed(_thinking_text(_EQUAL)): torch.tensor([1.0]),
                embed(_thinking_text(_GREATER)): torch.tensor([1.0]),
            },
            torch.tensor([1.0]),
        )
        return compare(score, thresh=0.0, true_level=1.0, false_level=-1.0)

    # --- borrow phase: LSB-first borrow fold on bigger/smaller.  A shifted
    # column total (bigger − smaller − borrow_in + 10) ∈ 0..19 makes both the
    # borrow-out (1 if total < 10) and the digit (total % 10) one table lookup. ---
    borrow_table = {
        _state(t, W): embed(_thinking_text(1 if t < 10 else 0)) for t in range(W)
    }
    digit_table = {_state(t, W): embed(str(t % 10)) for t in range(W)}
    prev_borrow = _thinking_value(embedding, embedding, n_state)  # offset 0

    def shifted_total(bigger: Node, smaller: Node, borrow_in: Node) -> Node:
        return add_const(subtract(subtract(bigger, smaller), borrow_in), 10.0)

    borrow_tok = []
    for k in range(n):
        here = borrow_start + k
        ge = a_ge_b_at(here)  # final verdict at seq[n]
        idx = n - 1 - k  # MSB-first index for place value k
        bigger = select(ge, a_scalar[idx], b_scalar[idx])
        smaller = select(ge, b_scalar[idx], a_scalar[idx])
        bin_ = _scalar(0.0) if k == 0 else prev_borrow
        borrow_tok.append(
            _emit_from_total(
                embedding,
                shifted_total(bigger, smaller, bin_),
                borrow_table,
                embed(_thinking_text(0)),
                W,
            )
        )

    # Magnitude digit at MSB position m (place-value column kc = n-1-m): the
    # borrow-in is the borrow-out of column kc-1, read from the borrow sweep.
    def digit(here: int, m: int) -> Node:
        kc = n - 1 - m
        ge = a_ge_b_at(here)
        bigger = select(ge, a_scalar[m], b_scalar[m])
        smaller = select(ge, b_scalar[m], a_scalar[m])
        if kc == 0:
            bin_ = _scalar(0.0)
        else:
            target = borrow_start + (kc - 1)
            token = attend_to_offset(
                rope, embedding, delta_pos=_read_offset(target, here)
            )
            bin_ = _thinking_value(embedding, token, n_state)
        return _emit_from_total(
            embedding, shifted_total(bigger, smaller, bin_), digit_table, embed("0"), W
        )

    scratch = [digit(layout.scratch_start + m, m) for m in range(n)]
    norm = _stream_normalize(rope, embedding, layout, n)
    col_onehot = _col_onehot(steps_since, layout, n)
    answer = _emit_gathered_answer(
        rope, embedding, layout, col_onehot, n, sign_at=a_ge_b_at
    )
    return verdict_tok + borrow_tok + scratch + norm, answer


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _assemble(
    embedding: Embedding, thinking: list[Node], answer: list[Node]
) -> list[Node]:
    """Wrap a (thinking, answer) pair into the full emitted token list:
    ``[THINKING_START] + thinking + [RESULT] + answer + [<eos>]``.

    The seq-index layout this produces (``THINKING_START`` at 0, ``thinking[k]``
    at ``1+k``, ``RESULT`` at ``1+len(thinking)``, ``answer[m]`` at
    ``2+len(thinking)+m``) is exactly what each op's :class:`_SeqLayout`
    encodes, so every read offset lines up by construction.
    """
    embed = embedding.get_embedding
    return (
        [create_literal_value(embed(THINKING_START))]
        + thinking
        + [create_literal_value(embed(RESULT))]
        + answer
        + [create_literal_value(embed("<eos>"))]
    )


def _emit_by_slot_index(
    rope: RopeConfig,
    steps_since: Node,
    is_trigger: Node,
    seq: list[Node],
    default_output: torch.Tensor,
) -> Node:
    """Autoregressive emission gated by the exact integer step counter.

    A drop-in replacement for :func:`~torchwright.ops.swiglu.sequence_ops.output_sequence`
    that avoids ``attend_to_offset`` for the slot-index gating — the same helper
    ``sort_digits_v1`` carries, lifted here for the same
    reason.  ``output_sequence`` gates slot ``k`` with
    ``attend_to_offset(is_trigger, delta_pos=-k)``; at an early emit position the
    deep slots target a position *before the sequence start*, where — with no
    real key to match — the sharp positional softmax locks onto a wrong existing
    key (the trigger itself when ``k`` nears a multiple of the fastest sinusoid's
    ~6.28 period).  That spuriously gates a deep slot's padding ``<eos>`` into an
    early token, and **widening the trig grid does not fix it** (the fastest
    sinusoid is period ~6.28 at any width).  The trimmed multiply transcript
    (``8n+3 = 27`` tokens for ``n=3``) has slots out to ``k=26``, well past the
    newline-to-start distance, so it trips this where the pre-trim ``4n+3`` did
    not.

    Instead the step index is the bucket-1 near-marker count ``steps_since =
    pos - newline_pos`` (the trigger is the newline; ``count_since_marker``
    recovers the bounded gap with no absolute position counter, accurate to well
    under ±0.5 over the short transcript), and ``compare`` tests ``steps_since ==
    k`` exactly for each compile-time ``k`` (the ±0.5 band absorbs the count's
    sub-integer error).  Costs one ``compare`` pair per slot instead of one
    ``attend_to_offset``, but the gating is immune to aliasing.  (The ops' own
    back-references — answer digit reading the carry glyph, etc. — keep using
    ``attend_to_offset``: those target *real, in-range* tokens, where the
    exact-match key always exists and outscores any near-match, so they are
    reliable at any distance.)  ``steps_since`` is built once in
    :func:`create_network_parts` and threaded in (the trigger ``is_trigger`` is
    the same newline that defines the count).
    """
    has_triggered = get_prev_value(rope, is_trigger, is_trigger)

    gated = []
    for k in range(len(seq)):
        # cond_k: true iff steps_since == k, a ±0.5-wide band around the integer.
        cond_k = bool_all_true(
            [
                compare(steps_since, thresh=k - 0.5),
                compare(negate(steps_since), thresh=-(k + 0.5)),
            ]
        )
        gated.append(cond_gate(cond_k, seq[k]))

    return select(
        cond=has_triggered,
        true_node=sum_nodes(gated),
        false_node=create_literal_value(default_output),
    )


def create_network_parts(
    max_digits: int = 3,
) -> tuple[Node, Embedding]:
    """Build the scratchpad calculator graph: parse, three streamed ops, dispatch.

    The parse is the shared constant-depth
    :func:`examples._calculator_common.parse_expression` (variable-width
    operands, same prompts as the other two calculators).  Each op builds its
    own full token list; the lists are padded to a common
    length with ``<eos>`` and dispatched per-position by the latched operator
    flags (exactly ``_calculator_common.build_calculator``'s pattern).  The
    operator is latched, so the same op is selected at every decode position —
    the selected op reads its own emitted prefix back correctly, while the
    non-selected ops misread it and are discarded by the switch.

    Emission uses the counter-based :func:`_emit_by_slot_index` rather than
    ``output_sequence``: the trimmed transcript is long enough (``8n+3`` tokens)
    that ``output_sequence``'s ``attend_to_offset`` slot gating would alias a
    deep slot onto an early position (see that helper).
    """
    n = max_digits
    embedding = create_onehot_embedding(scratch_vocab(n))
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)

    # Bucket-1 decode-step index pos - newline_pos, built once and threaded into
    # the per-column reads and the slot-gated emission (replaces the old
    # pos.get_position_scalar()).  max_gap is the longest padded transcript.
    steps_since = _steps_since_newline(rope, embedding, max_gap=decode_steps(n) + 2)

    A, B, is_plus, is_minus, is_times, saw_newline = parse_expression(
        rope, embedding, n
    )

    seqs = [
        _assemble(embedding, *op(rope, embedding, A, B, n, steps_since))
        for op in (_add_op, _sub_op, _mul_op)
    ]
    length = max(len(s) for s in seqs)
    eos = embedding.get_embedding("<eos>")
    seqs = [
        s + [create_literal_value(eos) for _ in range(length - len(s))] for s in seqs
    ]
    add_seq, sub_seq, mul_seq = seqs

    result = [
        switch([is_plus, is_minus, is_times], [add_seq[i], sub_seq[i], mul_seq[i]])
        for i in range(length)
    ]
    output_node = _emit_by_slot_index(
        rope,
        steps_since,
        saw_newline,
        result,
        embedding.get_embedding(" "),
    )
    return output_node, embedding
