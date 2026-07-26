"""Full-width arithmetic coverage for the sized calculator family.

Until 2026-07-26 no test exercised operands at a variant's own width
limit: every demo prompt and parse case used <= 3-digit operands — the
one region where all digit widths agree — and the collapse_pl
band-chaining bug shipped in exactly the untested region (digits near
the width limit read as "0").  Two layers of coverage here:

* the three arithmetic algorithms on 9-digit literal digit sequences
  (exact graph math, no attention, no parse) — carries and borrows
  rippling through every column;
* the shared parse at 9-digit width through ``lower()`` WITH both
  collapse passes — the lowering is what corrupted the marker-distance
  count, so the parse must be checked post-lowering, not just on the
  raw graph.
"""

import pytest
import torch

from examples._calculator_common import CALC_VOCAB, D_HEAD, MAX_POSITIONS
from torchwright.compiler.lower import lower
from torchwright.debug.probe import reference_eval
from torchwright.graph.embedding import bos_token
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_onehot_embedding,
    create_rope_config,
)
from torchwright.ops.linear import concat

N = 9


def _embedding():
    return create_onehot_embedding(CALC_VOCAB)


def _digit_nodes(embedding, digits: str):
    return [create_literal_value(embedding.get_embedding(c)) for c in digits]


def _decode_digits(embedding, value: torch.Tensor, width: int) -> str:
    """Decode a row of ``width`` concatenated one-hots to a digit string."""
    vocab = list(embedding.tokenizer.vocab)
    d_embed = len(embedding)
    out = []
    for i in range(width):
        lane = value[i * d_embed : (i + 1) * d_embed]
        out.append(vocab[int(lane.argmax())])
    return "".join(out)


@pytest.mark.parametrize("impl", ["calculator_simple", "calculator_advanced"])
@pytest.mark.parametrize(
    ("a", "b"),
    [
        (999999999, 999999999),  # every product column carries
        (123456789, 987654321),  # dense mixed digits
        (999999999, 2),  # asymmetric widths
        (100000000, 100000000),  # zero-heavy
    ],
)
def test_multiply_digit_seqs_full_width(impl, a, b):
    import importlib

    mod = importlib.import_module(f"examples.{impl}")
    embedding = _embedding()
    seq1 = _digit_nodes(embedding, str(a).zfill(N))
    seq2 = _digit_nodes(embedding, str(b).zfill(N))
    out = concat(mod.multiply_digit_seqs(embedding, seq1, seq2))
    got = _decode_digits(embedding, reference_eval(out, {}, 1)[out][0], 2 * N)
    assert got == str(a * b).zfill(2 * N), (impl, a, b, got)


@pytest.mark.parametrize("impl", ["calculator_simple", "calculator_advanced"])
def test_add_and_subtract_digit_seqs_full_width(impl):
    import importlib

    mod = importlib.import_module(f"examples.{impl}")
    embedding = _embedding()
    zero = create_literal_value(embedding.get_embedding("0"))

    a, b = 999999999, 1  # the carry ripples through every column
    seq1 = [zero, *_digit_nodes(embedding, str(a).zfill(N))]
    seq2 = [zero, *_digit_nodes(embedding, str(b).zfill(N))]
    out = concat(mod.add_digit_seqs(embedding, seq1, seq2))
    got = _decode_digits(embedding, reference_eval(out, {}, 1)[out][0], N + 1)
    assert got == str(a + b).zfill(N + 1), (impl, got)

    a, b = 100000000, 99999999  # the borrow ripples through every column
    seq1 = _digit_nodes(embedding, str(a).zfill(N))
    seq2 = _digit_nodes(embedding, str(b).zfill(N))
    out = concat(mod.subtract_digit_seqs(embedding, seq1, seq2))
    got = _decode_digits(embedding, reference_eval(out, {}, 1)[out][0], N)
    assert got == str(a - b).zfill(N), (impl, got)


def test_lowered_parse_reads_full_width_operands():
    """The parse windows, post-lowering, at 9-digit width.

    This is the smallest graph-level test that would have caught the
    shipped truncation: the raw graph parsed correctly, the LOWERED
    graph (collapse passes on, the compile default) read digits at
    region index >= 6 as "0" at this width.
    """
    from examples._calculator_common import parse_expression

    embedding = _embedding()
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)
    first, second, _p, _m, _t, _nl = parse_expression(rope, embedding, N)
    out = concat(first + second)
    lowered = lower(
        out, collapse_univariate=True, collapse_pl=True, collapse_lane_cap=4096
    )

    a, b = "999999999", "123456789"
    tokens = [bos_token, *a, "*", *b, "\n"]
    ids = torch.tensor(
        [embedding.tokenizer.get_token_id(t) for t in tokens], dtype=torch.long
    ).reshape(-1, 1)
    vals = reference_eval(lowered.output_node, {"embedding_input": ids}, len(tokens))
    got = _decode_digits(embedding, vals[lowered.output_node][-1], 2 * N)
    assert got == a + b, got
