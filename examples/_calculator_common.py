"""Shared scaffolding for the three calculator examples.

``calculator_simple`` and ``calculator_advanced`` are two *independent*
implementations of the same calculator: they differ only in their arithmetic —
the legible serial carry/borrow folds vs. the depth-optimized carry-lookahead /
carry-save versions.  Everything else is identical, so it lives here rather than
being duplicated or having one calculator import the other:

* the vocabulary and residual width (``CALC_VOCAB`` / ``D_MODEL``);
* the one-hot helpers both arithmetics build on (``_state`` / ``_slice`` and the
  carry/borrow state constants);
* **comparison** — lexicographic, computed positionally at constant depth
  (the verdict is the greater-flag at the first differing digit).
  Subtraction's sign just needs the answer, so both variants use this one
  verbatim;
* the token-stream plumbing — parsing the operands, dispatching on the
  operator, emitting the result autoregressively, trimming leading zeros —
  quarantined behind :func:`parse_expression`, :func:`emit_result`, and
  :func:`build_calculator` so each calculator file reads as its three
  arithmetic algorithms in the foreground.

Each calculator supplies its own ``add_digit_seqs`` / ``subtract_digit_seqs`` /
``multiply_digit_seqs`` and hands them to :func:`build_calculator`; nothing here
imports either calculator, and neither calculator imports the other.

``calculator_scratchpad`` — the flat-depth variant that streams its serial
work out as thinking tokens — shares :func:`parse_expression` too (all three
calculators accept the same variable-width ``"A op B\\n"`` prompts) plus the
small one-hot helpers, but replaces the rest of the plumbing: its emission,
dispatch, and leading-zero trim live on the decode-step axis.  The parse is
constant-depth in ``max_digits`` (an
:class:`~torchwright.ops.swiglu.sequence_ops.IndexedRegion` pointer gather per
operand digit), so it holds that variant's flat model depth and no longer adds
an ``O(max_digits)`` chain of its own to the other two.  (``simple`` still
grows linearly end to end through its serial arithmetic; ``advanced``'s
end-to-end depth now tracks its carry-lookahead / carry-save arithmetic —
the shared parse, comparison, and leading-zero trim are all constant-depth.)
"""

from typing import Dict, List, Tuple

import torch

from torchwright.graph import Embedding, Linear, Node, RopeConfig
from torchwright.graph.embedding import bos_token
from torchwright.ops.swiglu.arithmetic_ops import compare as _compare_scalar
from torchwright.ops.linear import add_const, bool_to_01, concat, sum_nodes
from torchwright.ops.attention_ops import get_prev_value
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_onehot_embedding,
    create_rope_config,
)
from torchwright.ops.swiglu.logic_ops import (
    bool_all_true,
    bool_any_true,
    bool_not,
    equals_vector,
)
from torchwright.ops.swiglu.map_select import broadcast_select, in_range, select, switch
from torchwright.ops.swiglu.onehot_table import onehot_lookup
from torchwright.ops.swiglu.sequence_ops import (
    IndexedRegion,
    check_is_digit,
    output_sequence,
    remove_leading_0s,
)

D_MODEL = 1024
# Rotary width the simple/advanced calculator graphs are built against; must
# equal the d_head the token-example harness compiles at.  32 leaves 16 slow
# planes, comfortably above the widest content head the shared parse / compare
# / emission plumbing builds (place_on_slow_planes checks at build time).
# ``calculator_scratchpad`` overrides this with its own wider D_HEAD: its
# multiply answer gather's content one-hot spans 2n+1 columns, which outgrows
# 16 planes past max_digits=7 (see the comment there, including the standing
# cos(Δ·θ) read-distance attenuation caveat on that gather's fastest lanes).
D_HEAD = 32
MAX_POSITIONS = 512

# Compact, calculator-only vocabulary: 10 digits, 3 operators, the newline that
# ends the input, a space (the pre-result placeholder), a BOS, and an EOS that
# pads / terminates the result.  17 tokens -> d_embed = 17 one-hot columns.
CALC_VOCAB = [str(d) for d in range(10)] + [
    "+",
    "-",
    "*",
    "\n",
    " ",
    bos_token,
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
    """Lexicographic comparison at constant depth: weight the digit flags so
    the first difference outweighs everything after it.

    This is how a person compares equal-width numbers — walk from the most
    significant digit, and the first position where the digits differ
    decides — encoded arithmetically in three FFN stages regardless of
    digit count:

    1. one ``onehot_lookup`` per position maps the concatenated digit pair
       to a three-way flag: ``+1`` (a > b), ``0`` (equal), ``-1`` (a < b)
       — all positions in parallel;
    2. each flag is re-sharpened to an exact ``-1/0/+1`` (two saturated
       ``compare`` ramps, combined for free).  This stage is load-bearing:
       the position weights below multiply each flag by up to ``3^(n-1)``,
       which would amplify the lookup's pass-through noise (proportional
       to the gathered digits' one-hot deviation) far past any budget —
       sharpened flags are bit-exact, so the weighted sum is exact integer
       arithmetic;
    3. a free ``Linear`` sums the flags with weights ``3^(n-1-i)`` (MSB
       heaviest).  ``3^k`` strictly exceeds ``2·(3^{k-1}+...+1)``, so the
       first nonzero flag dominates every later position combined and the
       sign of the sum IS the verdict; a final ``compare`` at ``-0.5``
       reads it off, with equal (sum exactly 0) counting as ``>=``.

    Returns a ±1 boolean: ``+1`` if ``seq1 >= seq2``, ``-1`` otherwise —
    the final compare keeps the cond clean for the selects that consume it.
    """
    assert len(seq1) == len(seq2)
    n = len(seq1)

    # key = concat([a, b]) -> three-way flag.  The default reads "equal"
    # (0.0) — a non-digit pair defers the verdict to later positions.
    pair_table: Dict[torch.Tensor, torch.Tensor] = {}
    for a in range(10):
        for b in range(10):
            key = torch.cat(
                [embedding.get_embedding(str(a)), embedding.get_embedding(str(b))]
            )
            pair_table[key] = torch.tensor(
                [0.0 if a == b else (1.0 if a > b else -1.0)]
            )
    default_flag = torch.tensor([0.0])

    flags = [
        onehot_lookup(concat([a, b]), pair_table, default_flag)
        for a, b in zip(seq1, seq2)  # MSB-first
    ]
    # Sharpen each flag to an exact three-level value: gt = 1 iff flag > 0.5,
    # lt = 1 iff flag < -0.5 (both bit-exact in saturation; the flag sits
    # 0.5 from either threshold, well clear of the 1/sharpness ramps).
    sharpened = []
    for f in flags:
        gt = _compare_scalar(f, thresh=0.5, true_level=1.0, false_level=0.0)
        lt = _compare_scalar(f, thresh=-0.5, true_level=0.0, false_level=1.0)
        sharpened.extend([gt, lt])

    stacked = concat(sharpened)
    weights = torch.zeros(len(stacked), 1)
    for i in range(n):
        weights[2 * i, 0] = 3.0 ** (n - 1 - i)  # gt adds the position weight
        weights[2 * i + 1, 0] = -(3.0 ** (n - 1 - i))  # lt subtracts it
    score = Linear(stacked, weights, name="cmp_score")
    return _compare_scalar(score, thresh=-0.5, true_level=1.0, false_level=-1.0)


# ---------------------------------------------------------------------------
# Token-stream plumbing: parse -> compute -> emit.
# ---------------------------------------------------------------------------


def parse_expression(
    rope: RopeConfig,
    embedding: Embedding,
    max_digits: int,
) -> Tuple[List[Node], List[Node], Node, Node, Node, Node]:
    """Parse ``"A op B\\n"`` from the token stream, at constant depth.

    Returns ``(first, second, is_plus, is_minus, is_times, saw_newline)``:
    the two operand digit windows (MSB-first, zero-padded to ``max_digits``),
    three latched ±1 flags for which operator appeared, and the ±1 newline
    trigger that ends the input and starts result emission.

    Operands are variable-width — no zero-padding required in the prompt.
    Each one is an :class:`~torchwright.ops.swiglu.sequence_ops.IndexedRegion`
    (the first delimited by ``<bos>`` and the operator, the second by the
    operator and the newline): every prompt digit carries a one-hot key of
    its index within its operand, the operand length is latched at the
    delimiter, and each window slot reads the digit it needs with one
    pointer attention — ``index = length - max_digits + i`` — falling back
    to the region default ``"0"`` where the index is negative.  Each window
    slot is latched at the newline, so every decode position reads the same
    operand values and the gather's slow-plane content match only has to
    survive the rotary attenuation over the prompt length (the regions'
    ``max_read_distance`` contract), not the whole rollout.  That makes
    the whole parse constant-depth in ``max_digits``, unlike the old
    ``NumericSequence`` sliding window, whose slot-to-slot shift chained
    through computed nodes and unrolled into ``O(max_digits)`` layers.
    """
    embed = embedding.get_embedding
    is_plus = equals_vector(embedding, embed("+"))
    is_minus = equals_vector(embedding, embed("-"))
    is_times = equals_vector(embedding, embed("*"))
    is_operator = bool_any_true([is_plus, is_minus, is_times])
    saw_newline = equals_vector(embedding, embed("\n"))

    # Only treat an operator as such *before* the newline, so a "-" emitted as
    # a negative sign during decoding does not re-trigger operator parsing.
    # (Single-trigger get_prev_value: the content gate selects the one matching
    # key regardless of distance, so this latch holds for the whole rollout.)
    seen_newline = get_prev_value(rope, saw_newline, saw_newline)
    is_input_operator = bool_all_true([is_operator, bool_not(seen_newline)])

    # Latch which operator was used (captured at the operator position, held
    # forward to every later position by attention).
    which_plus = get_prev_value(rope, is_plus, is_input_operator)
    which_minus = get_prev_value(rope, is_minus, is_input_operator)
    which_times = get_prev_value(rope, is_times, is_input_operator)

    # The two operand regions.  The membership gates carry the region
    # contract: only *prompt* digits qualify (a digit the model emits later
    # sits after the newline and must not collide with an operand key), and
    # the operator latch splits the prompt digits between the two operands.
    saw_bos = equals_vector(embedding, embed(bos_token))
    seen_operator = get_prev_value(rope, is_input_operator, is_input_operator)
    is_digit = check_is_digit(embedding)
    in_first = bool_all_true([is_digit, bool_not(seen_operator)])
    in_second = bool_all_true([is_digit, seen_operator, bool_not(seen_newline)])
    # The gathers below are consumed only at the newline (the get_prev_value
    # latch), so the farthest consumed read sits one prompt length from its
    # marker: "<bos> A op B \n" is at most 2·max_digits + 3 tokens, giving
    # the regions a max_read_distance bound of 2·max_digits + 4 with margin.
    max_read_distance = 2 * max_digits + 4
    first_region = IndexedRegion(
        rope,
        embedding,
        marker=saw_bos,
        in_region=in_first,
        max_len=max_digits,
        max_read_distance=max_read_distance,
        default=embed("0"),
    )
    second_region = IndexedRegion(
        rope,
        embedding,
        marker=is_input_operator,
        in_region=in_second,
        max_len=max_digits,
        max_read_distance=max_read_distance,
        default=embed("0"),
    )

    # First operand's length is known at the operator; second's at newline.
    # MSB-first window slot i holds the digit at index length - max_digits + i;
    # a negative index (operand narrower than the window) reads the "0" default.
    # Each pointer gather is latched at the newline: every decode position then
    # reads the value the gather computed *at the newline*, instead of
    # re-running the gather per position.  Two effects: the operand windows
    # cannot drift across the rollout, and the gather's content match only has
    # to survive the rotation over the prompt length (query at the newline,
    # keys at the prompt digits) rather than over the whole
    # rope.max_positions rollout — that prompt-length bound is the
    # max_read_distance contract the regions enforce above.  Cost: one extra
    # constant-depth attention sublayer.
    len_first = first_region.length_at(is_input_operator)
    len_second = second_region.length_at(saw_newline)
    first = [
        get_prev_value(
            rope,
            first_region.token_at(add_const(len_first, float(i - max_digits))),
            saw_newline,
        )
        for i in range(max_digits)
    ]
    second = [
        get_prev_value(
            rope,
            second_region.token_at(add_const(len_second, float(i - max_digits))),
            saw_newline,
        )
        for i in range(max_digits)
    ]
    return first, second, which_plus, which_minus, which_times, saw_newline


def _pad_result(embedding: Embedding, digits: List[Node], seq_len: int) -> List[Node]:
    """Align a digit sequence into the shared result frame (free literals):
    leading ``"0"`` digits out to the widest answer width (``seq_len - 2``,
    multiplication's ``2·max_digits``), then ``<eos>`` padding.  Every branch
    then has the same digit width, so the one post-dispatch trim's
    ``max_removals`` cap keeps exactly one digit of an all-zero answer no
    matter which branch won."""
    zero = create_literal_value(embedding.get_embedding("0"))
    eos = create_literal_value(embedding.get_embedding("<eos>"))
    return [zero] * (seq_len - 2 - len(digits)) + digits + [eos, eos]


def emit_result(
    rope: RopeConfig,
    embedding: Embedding,
    saw_newline: Node,
    result_digits: List[Node],
) -> Node:
    """Emit ``result_digits`` autoregressively once the newline fires, printing
    a space at every position before then."""
    return output_sequence(
        rope, saw_newline, result_digits, embedding.get_embedding(" ")
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

    # Recency ("most recent") is the intrinsic rotary lobe inside get_prev_value /
    # output_sequence — no rank node, no <ref> token, local-only.
    first, second, is_plus, is_minus, is_times, saw_newline = parse_expression(
        rope, embedding, max_digits
    )

    # Multiplication is the widest result (2*max_digits digits); the others are
    # padded with <eos> to this length so the operator switch is per-position.
    seq_len = 2 * max_digits + 2
    zero = create_literal_value(embedding.get_embedding("0"))

    # --- Addition: pad each operand by one digit so the top carry has a home. ---
    add_seq = _pad_result(
        embedding, add_digit_seqs(embedding, [zero] + first, [zero] + second), seq_len
    )

    # --- Subtraction: |A - B| by a borrow fold, sign from the comparison. ---
    a_ge_b = compare_digit_seqs(embedding, first, second)
    bigger = [select(a_ge_b, a, b) for a, b in zip(first, second)]
    smaller = [select(a_ge_b, b, a) for a, b in zip(first, second)]
    sub_seq = _pad_result(
        embedding, subtract_digit_seqs(embedding, bigger, smaller), seq_len
    )

    # --- Multiplication: long multiplication, product fits in 2*max_digits. ---
    mul_seq = _pad_result(
        embedding, multiply_digit_seqs(embedding, first, second), seq_len
    )

    # --- Dispatch by operator, THEN format the one selected answer. ---
    # Formatting after the dispatch instead of per branch runs one
    # leading-zero trim instead of three.  The trim's mux materializes a
    # whole formatted sequence on the residual stream, so three parallel
    # trims were the graph's peak-width hotspot — wide enough to starve
    # the d=1024 publish geometry.  Same depth either way: the operator
    # switch, the trim, and the sign select below are serial in both
    # orders.
    answer = [
        switch([is_plus, is_minus, is_times], [add_seq[i], sub_seq[i], mul_seq[i]])
        for i in range(seq_len)
    ]
    # A negative subtraction prepends the sign, shifting everything right
    # one slot; the trim's signed variant folds that into its own mux, so
    # the sign costs no extra stage.
    is_negative = bool_all_true([is_minus, bool_not(a_ge_b)])
    result_digits = remove_leading_0s(
        embedding,
        answer,
        max_removals=2 * max_digits - 1,
        sign_cond=is_negative,
        sign_token=embedding.get_embedding("-"),
    )
    output_node = emit_result(rope, embedding, saw_newline, result_digits)
    return output_node, embedding
