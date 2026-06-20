"""Depth-optimized one-hot arithmetic: the *advanced* calculator's algorithms.

This is the fast counterpart of :mod:`torchwright.ops.onehot_arithmetic`.  It
computes the same functions but trades the legible serial folds for the
logarithmic-depth structures a hardware multiplier uses.  It is **correctness
only** — it is never explained line-by-line in the blog, so its intricacy
costs nothing against the approachability budget; the only contract is that it
agrees, digit for digit, with the simple module.

The two depth wins, both ``O(log n)`` deep instead of ``O(n)``:

* **addition / subtraction — carry-lookahead.**  Read each column's carry
  *status* (does it start / pass / stop a carry) in parallel, fold those
  statuses with a Hillis-Steele parallel-prefix scan to get every carry at
  once, then look up every output digit in parallel.  Subtraction is the same
  scan with "borrow" read for "carry" (``a < b`` starts a borrow, ``a == b``
  passes one).

* **multiplication — carry-save (Wallace) reduction.**  Build the digit-product
  times table, drop each product's two digits into their place-value columns
  (so a column is a *bag* of one-hot digit addends), then repeatedly compress
  three addends in a column into one sum digit (kept) plus one carry digit
  (sent to the next column) with a base-10 "3:2 compressor" lookup.  A column
  bag shrinks to two addends in ``O(log n)`` rounds; a single carry-lookahead
  addition of the two surviving rows finishes the product.  Everything stays in
  one-hot space — no number↔one-hot conversion — so each compressor and each
  lookahead step is one exact lookup layer.

Comparison is not a depth target, so it reuses the simple module's
lexicographic fold verbatim (re-exported here so ``calculator_advanced`` finds
all four algorithms on this module).

Conventions match the simple module: digits are one-hot embedding rows,
sequences are MSB-first (``seq[0]`` most significant), and each function
documents the input width the caller must supply so the fixed-width result is
exact with no dropped nonzero carry.
"""

from typing import Dict, List

import torch

from torchwright.graph import Node, Embedding, Linear
from torchwright.ops.arithmetic_ops import add_const, bool_to_01, concat, sum_nodes
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.map_select import in_range
from torchwright.ops.onehot_table import onehot_lookup

# Reuse the simple module's shared one-hot primitives and its comparison.
from torchwright.ops.onehot_arithmetic import (  # noqa: F401  (compare re-exported)
    _CARRY_W,
    _NO,
    _YES,
    _slice,
    _state,
    compare_digit_seqs,
)

# ---------------------------------------------------------------------------
# Carry-lookahead addition / subtraction
#
# A column either GENERATEs a carry on its own (sum >= 10), PROPAGATEs an
# incoming carry (sum exactly 9: a carry-in tips it to 10), or KILLs any carry
# (sum <= 8).  These statuses combine associatively, so a parallel-prefix scan
# resolves every carry in O(log n) lookups instead of an O(n) serial walk.
# ---------------------------------------------------------------------------

_KILL, _PROPAGATE, _GENERATE = 0, 1, 2
_SEG_W = 3


def _combine_table() -> Dict[torch.Tensor, torch.Tensor]:
    """The associative carry-status combine for a lower block X then a higher Y.

    The combined block GENERATEs iff Y does, or Y propagates X's generate; it
    PROPAGATEs iff both propagate.  Equivalently: ``Y`` wins unless ``Y``
    propagates, in which case the block behaves like ``X``.
    """
    table: Dict[torch.Tensor, torch.Tensor] = {}
    for x in range(_SEG_W):
        for y in range(_SEG_W):
            combined = x if y == _PROPAGATE else y
            table[torch.cat([_state(x, _SEG_W), _state(y, _SEG_W)])] = _state(
                combined, _SEG_W
            )
    return table


def carry_lookahead(segments: List[Node]) -> List[Node]:
    """Carry into each column (LSB-first) from per-column carry statuses.

    ``segments`` is the LSB-first list of carry-status one-hots
    ({KILL, PROPAGATE, GENERATE}).  A Hillis-Steele parallel-prefix scan
    computes, for every column, the combined status of all columns up to and
    including it (``O(log n)`` combine-lookups deep instead of an ``O(n)``
    serial walk).  The carry into column ``j`` is then whether the block below
    it ends up GENERATEing — the incoming carry to column 0 is zero.

    Returns the LSB-first list of carry one-hots (2-state), same length as
    ``segments``.
    """
    n = len(segments)
    combine = _combine_table()
    default_seg = _state(_KILL, _SEG_W)

    prefix = list(segments)
    offset = 1
    while offset < n:
        nxt = list(prefix)
        for j in range(offset, n):  # combine lower block into higher, in parallel
            nxt[j] = onehot_lookup(
                concat([prefix[j - offset], prefix[j]]), combine, default_seg
            )
        prefix = nxt
        offset *= 2

    # carry into column j = does the block [0..j-1] generate a carry?  (With a
    # zero carry-in, a block emits a carry iff its combined status is GENERATE.)
    generated = {
        _state(_KILL, _SEG_W): _state(_NO, _CARRY_W),
        _state(_PROPAGATE, _SEG_W): _state(_NO, _CARRY_W),
        _state(_GENERATE, _SEG_W): _state(_YES, _CARRY_W),
    }
    carries: List[Node] = [create_literal_value(_state(_NO, _CARRY_W))]
    for j in range(1, n):
        carries.append(onehot_lookup(prefix[j - 1], generated, _state(_NO, _CARRY_W)))
    return carries


def _carry_lookahead_op(
    embedding: Embedding,
    seq1: List[Node],
    seq2: List[Node],
    *,
    status_of,
    digit_of,
) -> List[Node]:
    """Add or subtract two MSB-first digit sequences via carry-lookahead.

    ``status_of(a, b)`` returns the column's carry status (KILL / PROPAGATE /
    GENERATE); ``digit_of(a, b, carry)`` returns the output digit.  Per-column
    statuses and per-column digits are each one parallel layer of lookups; the
    carries between them come from :func:`carry_lookahead`.
    """
    assert len(seq1) == len(seq2)
    status_table: Dict[torch.Tensor, torch.Tensor] = {}
    digit_table: Dict[torch.Tensor, torch.Tensor] = {}
    for a in range(10):
        for b in range(10):
            ab = torch.cat(
                [embedding.get_embedding(str(a)), embedding.get_embedding(str(b))]
            )
            status_table[ab] = _state(status_of(a, b), _SEG_W)
            for carry in range(2):
                key = torch.cat([ab, _state(carry, _CARRY_W)])
                digit_table[key] = embedding.get_embedding(str(digit_of(a, b, carry)))

    pairs = list(reversed(list(zip(seq1, seq2))))  # LSB-first
    segments = [
        onehot_lookup(concat([a, b]), status_table, _state(_KILL, _SEG_W))
        for a, b in pairs
    ]
    carries = carry_lookahead(segments)
    out_lsb = [
        onehot_lookup(concat([a, b, carry]), digit_table, embedding.get_embedding("0"))
        for (a, b), carry in zip(pairs, carries)
    ]
    return list(reversed(out_lsb))


def add_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
    """Add two equal-length MSB-first digit sequences with carry-lookahead.

    Returns a sequence of the **same length**, dropping the final carry.  Size
    the inputs (e.g. a leading ``"0"`` on each ``n``-digit operand) so the sum
    fits; within that cap the dropped carry is always zero.
    """
    return _carry_lookahead_op(
        embedding,
        seq1,
        seq2,
        status_of=lambda a, b: (
            _GENERATE if a + b >= 10 else _PROPAGATE if a + b == 9 else _KILL
        ),
        digit_of=lambda a, b, carry: (a + b + carry) % 10,
    )


def subtract_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
    """``seq1 - seq2`` via borrow-lookahead (assumes ``seq1 >= seq2``).

    Equal length in, equal length out.  A column generates a borrow when
    ``a < b`` and passes one along when ``a == b``, so it reuses
    :func:`carry_lookahead`.  Because ``seq1 >= seq2`` the final borrow is
    always zero; the caller handles the sign separately.
    """
    return _carry_lookahead_op(
        embedding,
        seq1,
        seq2,
        status_of=lambda a, b: (
            _GENERATE if a < b else _PROPAGATE if a == b else _KILL
        ),
        digit_of=lambda a, b, borrow: (a - b - borrow) % 10,
    )


# ---------------------------------------------------------------------------
# Carry-save (Wallace) multiplication
# ---------------------------------------------------------------------------


def _times_table(embedding: Embedding) -> Dict[torch.Tensor, torch.Tensor]:
    """``a ⊕ b`` (two one-hot digits) -> ``tens ⊕ ones`` of ``a*b`` (one-hots)."""
    table: Dict[torch.Tensor, torch.Tensor] = {}
    for a in range(10):
        for b in range(10):
            key = torch.cat(
                [embedding.get_embedding(str(a)), embedding.get_embedding(str(b))]
            )
            table[key] = torch.cat(
                [
                    embedding.get_embedding(str(a * b // 10)),
                    embedding.get_embedding(str(a * b % 10)),
                ]
            )
    return table


_COMPRESS_W = 28  # x + y + z ranges 0..27, so 28 one-hot buckets cover every total


def _digit_value_proj(embedding: Embedding) -> torch.Tensor:
    """A ``(d_embed, 1)`` matrix turning a one-hot digit into its scalar value."""
    proj = torch.zeros(embedding.d_embed, 1)
    for d in range(10):
        proj[int(embedding.get_embedding(str(d)).argmax()), 0] = float(d)
    return proj


def _make_compressor(embedding: Embedding):
    """Return a 3:2 compressor ``(x, y, z one-hot digits) -> (sum, carry)``.

    Rather than a 10x10x10 = 1000-row lookup (1000 neurons — fat enough that a
    reduction level's parallel compressors spill across many transformer layers
    and inflate the apparent depth), this reuses the simple multiply's carry
    trick: read each one-hot digit's value with a free ``Linear``, add the three
    as plain numbers (free), then encode the small total (0..27) into a one-hot
    and read the sum digit ``total % 10`` and carry digit ``total // 10`` with
    two free selection-matrix lookups.  Cost is one ``in_range`` of width 28
    (~28 neurons) instead of 1000, so a level stays ~one layer deep.
    """
    value_proj = _digit_value_proj(embedding)
    zero_digit = embedding.get_embedding("0")
    sum_table = {
        _state(t, _COMPRESS_W): embedding.get_embedding(str(t % 10))
        for t in range(_COMPRESS_W)
    }
    carry_table = {
        _state(t, _COMPRESS_W): embedding.get_embedding(str(t // 10))
        for t in range(_COMPRESS_W)
    }

    def compress(x: Node, y: Node, z: Node):
        values = [Linear(d, value_proj, name="digit_value") for d in (x, y, z)]
        total = sum_nodes(values)  # plain-number add of the three digits, 0..27
        bucket = bool_to_01(in_range(total, add_const(total, 1.0), _COMPRESS_W))
        sum_digit = onehot_lookup(bucket, sum_table, zero_digit)
        carry_digit = onehot_lookup(bucket, carry_table, zero_digit)
        return sum_digit, carry_digit

    return compress


def multiply_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
    """Multiply ``seq1 * seq2`` (both ``n`` digits MSB-first) -> ``2n`` digits.

    Carry-save (Wallace) multiplication, all in one-hot space:

    1. **Times table.** Every digit pair ``seq1[i] * seq2[j]`` becomes a
       ``(tens, ones)`` one-hot pair, dropped into its place-value columns.
       Reading those bags across the columns gives up to ``2n`` *rows*, each a
       ``2n``-digit number (one digit per place, zero where a column is short).
    2. **Carry-save reduction.** Reduce the rows three at a time: a 3:2
       compressor turns rows ``(r0, r1, r2)`` into a *sum row* (the per-column
       ``(r0+r1+r2) % 10``) and a *carry row* (the per-column ``// 10``, shifted
       one place up).  Every column's compressor in a level runs in parallel and
       the carries become a row for the next level — no carry ripples within a
       level — so the row count falls ``3 -> 2`` each level and reaches two rows
       after ``O(log n)`` levels.
    3. **Final add.** The two surviving rows are summed with one carry-lookahead
       addition (``O(log n)`` deep).

    The carry-save invariant — the place-weighted sum of all rows always equals
    the product — means the two rows sum to exactly the product, which fits in
    ``2n`` digits.  So every column at place ``>= 2n`` is identically zero: the
    top column's carry (and the lookahead's final carry) are dropped as
    provably zero.  Returns the ``2n`` product digits, MSB-first.
    """
    n = len(seq1)
    assert len(seq2) == n
    d_embed = embedding.d_embed
    zero_digit = embedding.get_embedding("0")
    zero = create_literal_value(zero_digit)
    width = 2 * n  # places 0 (least significant) .. 2n-1; the product fits here

    times_table = _times_table(embedding)
    default_pair = torch.cat([zero_digit, zero_digit])

    # Step 1: drop each product's (tens, ones) into its place-value columns, then
    # transpose the column bags into rows (one digit per place, zero-padded).
    columns: List[List[Node]] = [[] for _ in range(width)]
    for i in range(n):
        for j in range(n):
            product = onehot_lookup(
                concat([seq1[i], seq2[j]]), times_table, default_pair
            )
            tens = _slice(product, 0, d_embed, name="pp_tens")
            ones = _slice(product, d_embed, d_embed, name="pp_ones")
            place = (n - 1 - i) + (n - 1 - j)  # place value of the ones digit
            columns[place].append(ones)
            columns[place + 1].append(tens)  # place+1 <= 2n-1 = width-1

    height = max((len(bag) for bag in columns), default=0)
    rows: List[List[Node]] = [
        [columns[c][r] if len(columns[c]) > r else zero for c in range(width)]
        for r in range(height)
    ] or [[zero] * width]

    # Step 2: carry-save reduction — 3 rows -> 2 rows per level, all columns in
    # parallel, until two rows survive.
    compress = _make_compressor(embedding)

    while len(rows) > 2:
        nxt: List[List[Node]] = []
        k = 0
        while k + 3 <= len(rows):
            r0, r1, r2 = rows[k], rows[k + 1], rows[k + 2]
            sum_row = [zero] * width
            carry_row = [zero] * width
            for c in range(width):
                sum_digit, carry_digit = compress(r0[c], r1[c], r2[c])
                sum_row[c] = sum_digit
                if c + 1 < width:  # the place-2n carry is provably zero -> drop
                    carry_row[c + 1] = carry_digit
            nxt.append(sum_row)
            nxt.append(carry_row)
            k += 3
        while k < len(rows):  # 1 or 2 leftover rows pass through
            nxt.append(rows[k])
            k += 1
        rows = nxt

    while len(rows) < 2:  # pad to exactly two rows for the final add
        rows.append([zero] * width)

    # Step 3: add the two surviving rows (MSB-first) with carry-lookahead.
    row_a = list(reversed(rows[0]))
    row_b = list(reversed(rows[1]))
    return add_digit_seqs(embedding, row_a, row_b)  # 2n digits, MSB-first
