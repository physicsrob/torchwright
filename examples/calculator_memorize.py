"""Calculator that computes nothing: every answer is a memorized fact.

The fourth corner of the calculator design space.  ``calculator_simple``
computes in depth (a serial fold per digit), ``calculator_advanced`` in
less depth (carry-lookahead / carry-save), ``calculator_scratchpad`` in
decode steps (flat depth, serial work streamed as thinking tokens) —
this variant does not compute at all.  One swish lane per possible
expression holds the complete formatted answer (sign, leading-zero trim,
and <eos> padding pre-applied), keyed on the concatenated operand digit
one-hots and the operator one-hot.  Comparison, carries, borrows, and
the leading-zero trim all vanish: memorization includes formatting.

The price is the exponential fact table, and stating it exactly is the
point.  With ``n = max_digits`` (operands zero-padded to width ``n``):

    facts(n)  = 3 * 10^(2n)          (one lane per expression)
    params(n) = facts(n) * (d_key + d_answer + 1) + 30 * d_answer
              = 3 * 10^(2n) * (68n + 72) + 30 * (34n + 34)

where ``d_key = 34n + 3`` (two digit windows + the operator one-hot)
and ``d_answer = (2n + 2) * 17`` (the widest result frame, embedding
rows per slot); the ``+1`` is each lane's gate bias and the ``30 *
d_answer`` term is the partition FFNs' output biases.  Layer count
splits into two regimes, both measured (2026-07-20, optimize=0 via
``scripts.measure_calculator_compiled_layers --impl calculator_memorize``):

    floor                 = 13, constant in n  (the parse -> one lookup
                                                -> emit chain;
                                                width-unlimited)
    n_layers(n, d_hidden) = 14 + ceil(facts(n) / d_hidden)
                                               (capacity-bound: an FFN
                                                bank packs ~d_hidden
                                                lanes per layer; exact
                                                at n=1 d_hidden=8192 ->
                                                15, n=2 8192 -> 18,
                                                n=2 16384 -> 16)

and the residual stream must additionally carry the table's read-out —
the 30 partitions' ``d_answer``-wide outputs are all alive at the sum,
so ``d`` below ~4096 has no schedule at all (measured: n=1 and n=2 both
deadlock at d=2048 and schedule at d=4096; the family's canonical
``d=8192`` clears it with margin).  Because the capacity term binds at
EVERY geometry,
:func:`compiled_layers` — not the 13-layer floor — is what a compile
produces; the committed layer table reports the witnessed ``optimize=2``
compile at the canonical geometry for the buildable row and extends the
law to the refused rows.  The parameter/fact formulas are validated against
the built graph's actual lane and weight counts in
``tests/examples/test_calculator_memorize.py``, which also verifies all
300 n=1 facts end to end.  ``max_digits <= 2`` is enforced at build
time: n=3 is 3,000,000 facts (~726M parameters, ~12 GB of fp32 weights,
and ~370 capacity layers at d_hidden=8192) — the formulas extrapolate
it; building it teaches nothing more.

The fact table is partitioned by (operator, leading digit of A) into 30
lookups over the same key — each partition holds ``10^(2n-1)`` lanes, so
every partition fits a single MLP sublayer at any supported geometry and
the scheduler packs them across consecutive layers by capacity.  A key
outside a partition mismatches at least one one-hot block, so exactly
one partition fires and their free sum is the answer.
"""

import torch

from examples._calculator_common import (
    CALC_VOCAB,
    D_HEAD,
    D_HIDDEN,
    D_MODEL,
    MAX_POSITIONS,
    _slice,
    emit_result,
    parse_expression,
)
from torchwright.graph import Embedding, Node
from torchwright.ops.inout_nodes import create_onehot_embedding, create_rope_config
from torchwright.ops.linear import bool_to_01, concat, sum_nodes
from torchwright.ops.swiglu.onehot_table import onehot_lookup

__all__ = [
    "CALC_VOCAB",
    "D_HEAD",
    "D_HIDDEN",
    "D_MODEL",
    "MAX_POSITIONS",
    "compiled_layers",
    "create_network_parts",
    "n_facts",
    "n_params",
]

# Build-time cap: the fact table is 3 * 10^(2n) lanes.  n=2 is 30,000
# (5.2M parameters — builds and compiles); n=3 is 3,000,000 (~720M
# parameters, ~12 GB fp32) — expressible only through the formulas above.
MAX_SUPPORTED_DIGITS = 2

# Model-card fields consumed by ``examples.compile`` when publishing this
# example as a Hugging Face bundle.  Prompts stay within the 2-digit width.
CARD_TASK = (
    "a computation graph for integer arithmetic (`A op B` with `op` in `+ - *`) "
    "that computes nothing: every answer is a memorized fact, looked up from "
    "the operand pair"
)
DEMO_PROMPTS = ["12*34\n", "7+8\n", "10-99\n", "99*99\n"]


def n_facts(max_digits: int) -> int:
    """One lane per possible expression: ``3 * 10^(2n)``."""
    return 3 * 10 ** (2 * max_digits)


def n_params(max_digits: int) -> int:
    """Fact-table parameter count (see the module docstring's derivation)."""
    n = max_digits
    d_key = 34 * n + 3
    d_answer = (2 * n + 2) * 17
    return n_facts(n) * (d_key + d_answer + 1) + 30 * d_answer


def compiled_layers(max_digits: int, d_hidden: int = 16384) -> int:
    """The layer count a compile at width ``d_hidden`` produces.

    Unlike the computing calculators — whose compiled layer count equals
    their dependency floor once width saturates — memorize is always
    capacity-bound: an FFN bank packs ~``d_hidden`` fact lanes per layer,
    so the table's ``ceil(facts / d_hidden)`` sublayers are paid at every
    geometry and the 13-layer dependency floor is never attained.  The
    default is the family's canonical ``D_HIDDEN``.

    Measured (optimize=0, the eager walk): n=1 d_hidden=8192 -> 15,
    n=2 8192 -> 18, n=2 16384 -> 16 — exact at all three points.  An
    ``optimize=2`` compile can land below the law at small n (witnessed:
    n=2 at d=4096/d_hidden=8192 -> 16, two under the law's 18, and at
    the canonical d=8192/d_hidden=16384 -> 14, two under the law's 16 —
    CP-SAT packs the bank into fewer layers than the eager walk, by a
    mechanism not traced further); at large n the ``ceil(facts /
    d_hidden)`` capacity term dominates either schedule, so the law is
    what the refused rows extrapolate with — asymptotically right,
    conservative by ~2 at small n.
    """
    return 14 + -(-n_facts(max_digits) // d_hidden)


def _format_answer(embedding: Embedding, value: int, seq_len: int) -> torch.Tensor:
    """The complete formatted answer as a flat ``seq_len * 17`` vector.

    Sign and decimal digits, then <eos> padding - exactly what the other
    calculators spend their comparison / borrow / trim machinery producing.
    """
    text = str(value)
    assert len(text) <= seq_len
    slots = list(text) + ["<eos>"] * (seq_len - len(text))
    return torch.cat([embedding.get_embedding(t) for t in slots])


def create_network_parts(max_digits: int = 2) -> tuple[Node, Embedding]:
    """The memorizing calculator: parse, one fact lookup, emit."""
    if max_digits > MAX_SUPPORTED_DIGITS:
        raise ValueError(
            f"calculator_memorize: max_digits={max_digits} needs "
            f"{n_facts(max_digits):,} fact lanes (~{n_params(max_digits):,} "
            f"parameters) — the table is exponential in the digit count and "
            f"only max_digits <= {MAX_SUPPORTED_DIGITS} is buildable; use "
            f"n_facts()/n_params() for the extrapolated numbers"
        )
    n = max_digits
    seq_len = 2 * n + 2

    embedding = create_onehot_embedding(CALC_VOCAB)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)
    first, second, is_plus, is_minus, is_times, saw_newline = parse_expression(
        rope, embedding, n
    )

    # The lookup key: both zero-padded operand windows plus the operator
    # as a third one-hot block (exactly one flag is true).
    op_onehot = concat([bool_to_01(f) for f in (is_plus, is_minus, is_times)])
    key = concat([*first, *second, op_onehot])

    # The fact table, partitioned by (operator, leading digit of A): 30
    # partitions of 10^(2n-1) lanes over the same key, zero default, so
    # exactly one partition fires and the free sum is its stored answer.
    # Every partition's d_answer-wide output is alive until the sum
    # consumes it (30 * 102 = 3,060 residual columns at n=2 — grouping
    # the sum does not help, linear fusion legitimately flattens it
    # back), so the compile geometry must carry the table's read-out
    # width; see the module docstring's d_model row.
    ops = [("+", 10), ("-", 11), ("*", 12)]
    d_answer = seq_len * 17
    partitions: list[Node] = []
    for op_text, op_index in ops:
        for lead in range(10):
            table = {}
            for a in range(lead * 10 ** (2 * n - 1), (lead + 1) * 10 ** (2 * n - 1)):
                a_hi, b = divmod(a, 10**n)
                value = {"+": a_hi + b, "-": a_hi - b, "*": a_hi * b}[op_text]
                digits = str(a_hi).zfill(n) + str(b).zfill(n)
                fact_key = torch.cat(
                    [embedding.get_embedding(c) for c in digits]
                    + [torch.eye(3)[op_index - 10]]
                )
                table[fact_key] = _format_answer(embedding, value, seq_len)
            partitions.append(onehot_lookup(key, table, torch.zeros(d_answer)))
    answer = sum_nodes(partitions)

    result_digits = [
        _slice(answer, i * 17, 17, name="memorized_slot") for i in range(seq_len)
    ]
    output_node = emit_result(rope, embedding, saw_newline, result_digits)
    return output_node, embedding
