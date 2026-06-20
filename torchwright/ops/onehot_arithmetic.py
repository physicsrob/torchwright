"""School-math arithmetic on one-hot digits via exact selection tables.

The calculator's thesis, made literal: every operation is an exhaustive lookup
over digit combinations (:func:`torchwright.ops.onehot_table.onehot_lookup`),
wired together the way the pencil-and-paper algorithm wires it:

* **addition / subtraction** thread a single carry (borrow) one-hot along the
  digits from least significant to most: at each column one lookup over
  ``concat([a, b, carry])`` reads off the output digit and another reads the
  carry passed to the next column;
* **comparison** folds a running less/equal/greater verdict from the most
  significant digit down;
* **multiplication** multiplies every digit pair (the times table), sums the
  products per place-value column as plain numbers, and carries once.

Every digit is a one-hot embedding row and every status/verdict/carry is a
small one-hot, so each lookup key is a concatenation of one-hots and each
lookup is exact integer counting.  This module is the one-hot counterpart of
``embedding_arithmetic`` (which uses the soft ``map_to_table`` over
spherical-code embeddings) and shares its MSB-first convention: ``seq[0]`` is
the most significant digit, ``seq[-1]`` the least.

**No overflow by construction.**  Each function returns a fixed width and
documents how wide the caller must make the inputs so the true result always
fits.  Within those caps the carry/borrow folds never drop a nonzero top
carry, so the fixed-width output is exact.
"""

from typing import Dict, List

import torch

from torchwright.graph import Node, Embedding, Linear
from torchwright.ops.arithmetic_ops import (
    add,
    add_const,
    bool_to_01,
    compare as _compare_scalar,
    concat,
    sum_nodes,
)
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.map_select import in_range
from torchwright.ops.onehot_table import onehot_lookup

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
# The shared shape: a lookup table threaded by a state fold.
# ---------------------------------------------------------------------------


def digitwise_fold(
    embedding: Embedding,
    seq1: List[Node],
    seq2: List[Node],
    *,
    digit_table: Dict[torch.Tensor, torch.Tensor],
    state_table: Dict[torch.Tensor, torch.Tensor],
    init_state: torch.Tensor,
) -> List[Node]:
    """Right-to-left (LSB-first) fold threading one state one-hot.

    At each digit position the key ``concat([a, b, state])`` is looked up in
    two tables built over the same key space: ``digit_table`` gives the output
    digit (an embedding row), ``state_table`` gives the carry/borrow passed to
    the next, more-significant position.  This is the carry-propagation /
    borrow-propagation shape; addition and subtraction differ only in their
    two tables.

    Args:
        embedding: The one-hot embedding (its ``"0"`` row is the digit default).
        seq1, seq2: Equal-length MSB-first digit sequences.
        digit_table, state_table: Lookups keyed on ``concat([a, b, state])``.
        init_state: The state one-hot entering the least-significant position.

    Returns:
        Output digits, MSB-first, same length as the inputs.  The final state
        is dropped — callers size the inputs so it is always ``init_state``.
    """
    assert len(seq1) == len(seq2)
    default_digit = embedding.get_embedding("0")
    default_state = init_state
    state = create_literal_value(init_state)
    out: List[Node] = []
    for a, b in reversed(list(zip(seq1, seq2))):
        key = concat([a, b, state])
        out.append(onehot_lookup(key, digit_table, default_digit))
        state = onehot_lookup(key, state_table, default_state)
    return list(reversed(out))


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


def add_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
    """Add two equal-length MSB-first digit sequences with a carry fold.

    Returns a sequence of the **same length**, dropping the final carry.

    **No-overflow cap:** the caller must size the inputs so the sum fits in
    that width — e.g. prepend a ``"0"`` digit to each ``n``-digit operand so an
    ``n``-digit + ``n``-digit sum has a home for its top carry.  Within the cap
    the dropped carry is always zero, so the result is exact.
    """
    digit_table: Dict[torch.Tensor, torch.Tensor] = {}
    state_table: Dict[torch.Tensor, torch.Tensor] = {}
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
                digit_table[key] = embedding.get_embedding(str(total % 10))
                state_table[key] = _state(_YES if total >= 10 else _NO, _CARRY_W)
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
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
    """``seq1 - seq2`` digit by digit with a borrow fold (assumes ``seq1 >= seq2``).

    Equal length in, equal length out.  Because ``seq1 >= seq2`` the final
    borrow is always zero, so dropping it is exact — the caller handles the
    sign separately (see :func:`compare_digit_seqs`).
    """
    digit_table: Dict[torch.Tensor, torch.Tensor] = {}
    state_table: Dict[torch.Tensor, torch.Tensor] = {}
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
# Comparison
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
# Multiplication
# ---------------------------------------------------------------------------


def multiply_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
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
    d_embed = embedding.d_embed
    zero_scalar = create_literal_value(torch.tensor([0.0]))

    # Step 1: the times table.  Two one-hot digits -> (tens, ones) as numbers.
    product_table: Dict[torch.Tensor, torch.Tensor] = {}
    for a in range(10):
        for b in range(10):
            key = torch.cat(
                [embedding.get_embedding(str(a)), embedding.get_embedding(str(b))]
            )
            product_table[key] = torch.tensor([float(a * b // 10), float(a * b % 10)])
    default_product = torch.tensor([0.0, 0.0])

    # Steps 1-2: drop each product's digits into their place-value columns and
    # collect the (free) per-column number contributions.  Column p is the
    # 10^p place; columns run 0 (least significant) .. 2n-1.
    columns: List[List[Node]] = [[] for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            product = onehot_lookup(
                concat([seq1[i], seq2[j]]), product_table, default_product
            )
            tens = _slice(product, 0, 1, name="product_tens")
            ones = _slice(product, 1, 1, name="product_ones")
            place = (n - 1 - i) + (n - 1 - j)  # place value of the ones digit
            columns[place].append(ones)
            columns[place + 1].append(tens)

    # Step 3: one carry sweep, least-significant column first.  Turn each
    # column total (a number 0..20n) into a one-hot index, then read off its
    # digit and its carry with two free selection-matrix lookups.
    max_total = 20 * n
    digit_table = {
        _state(t, max_total + 1): embedding.get_embedding(str(t % 10))
        for t in range(max_total + 1)
    }
    carry_table = {
        _state(t, max_total + 1): torch.tensor([float(t // 10)])
        for t in range(max_total + 1)
    }

    carry: Node = zero_scalar
    out_lsb_first: List[Node] = []
    for place in range(2 * n):
        contributions = columns[place] or [zero_scalar]
        total = add(sum_nodes(contributions), carry)  # free number addition
        total_onehot = bool_to_01(in_range(total, add_const(total, 1.0), max_total + 1))
        out_lsb_first.append(
            onehot_lookup(total_onehot, digit_table, embedding.get_embedding("0"))
        )
        carry = onehot_lookup(total_onehot, carry_table, torch.tensor([0.0]))
    return list(reversed(out_lsb_first))
