r"""Calculator compiled into a transformer.

``+``, ``-``, ``*`` on integers up to ``max_digits`` digits, parsed from
``"A op B\\n"`` and emitted digit by digit.  This is the *legible* variant,
built to be read line by line.

Every operation is the same two ideas — an exhaustive **lookup table** over
digit combinations (:func:`torchwright.ops.swiglu.onehot_table.onehot_lookup`), and a
**fold** that threads one piece of state along the digits: a carry for
addition, a borrow for subtraction, a running less/equal/greater verdict for
comparison, a per-column running sum for multiplication.  With one-hot
embeddings each digit is a unit vector, each lookup is exact integer counting,
the compiled weight matrix *is* the arithmetic table, and the identity unembed
makes argmax-decode exact.

This file is one of two standalone calculator implementations.  It defines its
own ``add`` / ``subtract`` / ``multiply`` (the legible serial folds) and hands
them to :func:`examples._calculator_common.build_calculator`, which supplies the
token-stream plumbing and the shared comparison fold.  ``calculator_advanced``
is the depth-optimized peer (carry-lookahead add/subtract, carry-save multiply,
``O(log n)`` depth); ``scripts/arithmetic_scaling.py`` measures the two against
each other.  Sequences are MSB-first (``seq[0]`` most significant); each
function documents how wide the caller must size the inputs so the fixed-width
result never drops a nonzero carry.
"""

import torch

from examples._calculator_common import (
    _CARRY_W,
    _NO,
    _YES,
    CALC_VOCAB,
    D_HEAD,
    D_HIDDEN,
    D_MODEL,
    MAX_POSITIONS,
    _slice,
    _state,
    build_calculator,
    compare_digit_seqs,  # shared verbatim — re-exported as part of this variant
)
from torchwright.graph import Embedding, Node
from torchwright.ops.const import step_sharpness
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.linear import add, add_const, bool_to_01, concat, sum_nodes
from torchwright.ops.swiglu.arithmetic_ops import piecewise_linear
from torchwright.ops.swiglu.map_select import in_range
from torchwright.ops.swiglu.onehot_table import onehot_lookup

_DECIMAL_BASE = 10

__all__ = [
    "CALC_VOCAB",
    "D_HEAD",
    "D_HIDDEN",
    "D_MODEL",
    "MAX_POSITIONS",
    "add_digit_seqs",
    "compare_digit_seqs",
    "create_network_parts",
    "multiply_digit_seqs",
    "subtract_digit_seqs",
]

# Model-card fields consumed by ``examples.compile`` when publishing this
# example as a Hugging Face bundle.
CARD_TASK = "a computation graph for integer arithmetic (`A op B` with `op` in `+ - *`)"
DEMO_PROMPTS = ["12*34\n", "7+8\n", "100-99\n", "123*4\n"]


# ---------------------------------------------------------------------------
# The shared shape: a lookup table threaded by a state fold.
# ---------------------------------------------------------------------------


def digitwise_fold(
    embedding: Embedding,
    seq1: list[Node],
    seq2: list[Node],
    *,
    digit_table: dict[torch.Tensor, torch.Tensor],
    state_table: dict[torch.Tensor, torch.Tensor],
    init_state: torch.Tensor,
) -> list[Node]:
    """Right-to-left (LSB-first) fold threading one state one-hot.

    At each digit position the key ``concat([a, b, state])`` is looked up in
    two tables built over the same key space: ``digit_table`` gives the output
    digit (an embedding row), ``state_table`` gives the carry/borrow passed to
    the next, more-significant position.  This is the carry-propagation /
    borrow-propagation shape; addition and subtraction differ only in their
    two tables.

    Args:
        embedding: The one-hot embedding (its ``"0"`` row is the digit default).
        seq1: Equal-length MSB-first digit sequence (the ``a`` operand).
        seq2: Equal-length MSB-first digit sequence (the ``b`` operand).
        digit_table: Lookup keyed on ``concat([a, b, state])`` giving the
            output digit.
        state_table: Lookup keyed on ``concat([a, b, state])`` giving the
            carry/borrow passed to the next position.
        init_state: The state one-hot entering the least-significant position.

    Returns:
        Output digits, MSB-first, same length as the inputs.  The final state
        is dropped — callers size the inputs so it is always ``init_state``.
    """
    assert len(seq1) == len(seq2)
    default_digit = embedding.get_embedding("0")
    default_state = init_state
    state = create_literal_value(init_state)
    out: list[Node] = []
    for a, b in reversed(list(zip(seq1, seq2, strict=False))):
        key = concat([a, b, state])
        out.append(onehot_lookup(key, digit_table, default_digit))
        state = onehot_lookup(key, state_table, default_state)
    return list(reversed(out))


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


def add_digit_seqs(
    embedding: Embedding, seq1: list[Node], seq2: list[Node]
) -> list[Node]:
    """Add two equal-length MSB-first digit sequences with a carry fold.

    Returns a sequence of the **same length**, dropping the final carry.

    **No-overflow cap:** the caller must size the inputs so the sum fits in
    that width — e.g. prepend a ``"0"`` digit to each ``n``-digit operand so an
    ``n``-digit + ``n``-digit sum has a home for its top carry.  Within the cap
    the dropped carry is always zero, so the result is exact.
    """
    digit_table: dict[torch.Tensor, torch.Tensor] = {}
    state_table: dict[torch.Tensor, torch.Tensor] = {}
    for a in range(10):
        for b in range(10):
            for carry in range(2):
                key = torch.cat(
                    [
                        embedding.get_embedding(str(a)),
                        embedding.get_embedding(str(b)),
                        _state(carry, _CARRY_W),
                    ]
                )
                total = a + b + carry
                digit_table[key] = embedding.get_embedding(str(total % _DECIMAL_BASE))
                state_table[key] = _state(
                    _YES if total >= _DECIMAL_BASE else _NO, _CARRY_W
                )
    return digitwise_fold(
        embedding,
        seq1,
        seq2,
        digit_table=digit_table,
        state_table=state_table,
        init_state=_state(_NO, _CARRY_W),
    )


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


def subtract_digit_seqs(
    embedding: Embedding, seq1: list[Node], seq2: list[Node]
) -> list[Node]:
    """``seq1 - seq2`` digit by digit with a borrow fold (assumes ``seq1 >= seq2``).

    Equal length in, equal length out.  Because ``seq1 >= seq2`` the final
    borrow is always zero, so dropping it is exact — the caller handles the
    sign separately (see :func:`compare_digit_seqs`).
    """
    digit_table: dict[torch.Tensor, torch.Tensor] = {}
    state_table: dict[torch.Tensor, torch.Tensor] = {}
    for a in range(10):
        for b in range(10):
            for borrow in range(2):
                key = torch.cat(
                    [
                        embedding.get_embedding(str(a)),
                        embedding.get_embedding(str(b)),
                        _state(borrow, _CARRY_W),
                    ]
                )
                diff = a - b - borrow
                digit_table[key] = embedding.get_embedding(str(diff % 10))
                state_table[key] = _state(_YES if diff < 0 else _NO, _CARRY_W)
    return digitwise_fold(
        embedding,
        seq1,
        seq2,
        digit_table=digit_table,
        state_table=state_table,
        init_state=_state(_NO, _CARRY_W),
    )


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def multiply_digit_seqs(
    embedding: Embedding, seq1: list[Node], seq2: list[Node]
) -> list[Node]:
    """Multiply ``seq1 * seq2`` (both ``n`` digits MSB-first) -> ``2n`` digits.

    Rather than building whole partial-product rows and adding them with a
    carry-propagating addition each time (an ``O(n)`` serial stack of carry
    folds), this accumulates the digit products *as plain numbers* and carries
    exactly once:

    1. **Times table.** Every digit pair ``seq1[i] * seq2[j]`` is one lookup
       returning its two-digit product as a pair of plain numbers
       ``(tens, ones)`` — all ``n^2`` in parallel, one layer.
    2. **Column sums.** Each product's tens/ones land in their place-value
       columns, and a column's contributions are summed *as numbers*.  Number
       addition is free (residual wiring, no carry to ripple), so this costs
       no layers — but a column total can exceed 9.
    3. **One carry sweep.** Sweeping the ``2n`` columns least-significant
       first, each column's total (plus the incoming carry) is turned back
       into a digit, passing the overflow to the next column.  This is the
       only place carries ripple.

    **No-overflow cap:** ``(10^n - 1)^2 < 10^(2n)``, so ``2n`` digits hold the
    product.  A column receives at most ``2n`` digit contributions (each
    ``<= 9``) and the incoming carry stays ``<= 2n``, so a column total plus
    carry never exceeds ``20n`` — the width of the carry-sweep lookup table.
    """
    n = len(seq1)
    assert len(seq2) == n
    zero_scalar = create_literal_value(torch.tensor([0.0]))

    # Step 1: the times table.  Two one-hot digits -> (tens, ones) as numbers.
    product_table: dict[torch.Tensor, torch.Tensor] = {}
    for a in range(10):
        for b in range(10):
            key = torch.cat(
                [embedding.get_embedding(str(a)), embedding.get_embedding(str(b))]
            )
            product_table[key] = torch.tensor([float(a * b // 10), float(a * b % 10)])
    default_product = torch.tensor([0.0, 0.0])

    # Steps 1-2: drop each product's digits into their place-value columns and
    # collect the (free) per-column number contributions.  Like the operands
    # and result, columns are MSB-first: column 0 is the most significant
    # product digit and column 2n-1 is the units digit.
    columns: list[list[Node]] = [[] for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            product = onehot_lookup(
                concat([seq1[i], seq2[j]]), product_table, default_product
            )
            tens = _slice(product, 0, 1, name="product_tens")
            ones = _slice(product, 1, 1, name="product_ones")
            columns[i + j].append(tens)
            columns[i + j + 1].append(ones)

    # Step 3: one carry sweep, least-significant column first.  Turn each
    # column total (a number 0..20n) into a one-hot index and read off its
    # digit with a free selection-matrix lookup; the carry is "how many
    # tens fit" — a 2n-step staircase over the total, one FFN.
    #
    # The carry is deliberately NOT read off the wide one-hot with a
    # 0..2n-valued selection row: a saturated in_range slot at index i
    # carries fp32 rounding jitter of ulp(scale·sharpness·i)/scale (its
    # hinge pair straddles a binade boundary at some off-integer inputs),
    # and a selection row weighted up to 2n amplifies two adjacent slots'
    # jitter past the 1e-3 claim budget of the compiler's univariate
    # collapse pass — leaving the whole sweep at three compiled layers
    # per column instead of one.  The staircase reads the same step
    # structure with unit hinge weights, keeping the composed noise an
    # order of magnitude under the budget.
    max_total = 20 * n
    digit_table = {
        _state(t, max_total + 1): embedding.get_embedding(str(t % 10))
        for t in range(max_total + 1)
    }
    # Staircase knot pairs (one step per ten): carry k-1 up to 10k-0.5,
    # carry k from 10k-0.4 — the standard 1/sharpness ramp width, so
    # integer totals sit >= 0.4 from every knot (bit-exact reads).
    carry_knots: dict[float, float] = {}
    for k in range(1, max_total // 10 + 1):
        carry_knots[10.0 * k - 0.5] = float(k - 1)
        carry_knots[10.0 * k - 0.4] = float(k)

    carry: Node = zero_scalar
    out_lsb_first: list[Node] = []
    for column in reversed(columns):
        contributions = column or [zero_scalar]
        total = add(sum_nodes(contributions), carry)  # free number addition
        total_onehot = bool_to_01(in_range(total, add_const(total, 1.0), max_total + 1))
        out_lsb_first.append(
            onehot_lookup(total_onehot, digit_table, embedding.get_embedding("0"))
        )
        carry = piecewise_linear(
            total,
            sorted(carry_knots),
            carry_knots.__getitem__,
            input_scale=step_sharpness,
            name="carry_staircase",
        )
    return list(reversed(out_lsb_first))


def create_network_parts(
    max_digits: int = 3,
) -> tuple[Node, Embedding]:
    """Build the simple calculator.

    The legible serial folds wired up by
    :func:`examples._calculator_common.build_calculator`.
    """
    return build_calculator(
        max_digits,
        add_digit_seqs=add_digit_seqs,
        subtract_digit_seqs=subtract_digit_seqs,
        multiply_digit_seqs=multiply_digit_seqs,
    )
