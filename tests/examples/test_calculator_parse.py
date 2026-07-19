"""The shared calculator parse: variable-width operands, reference evaluation.

``parse_expression`` is the front end all three calculators share.  These
tests reference-eval the parse subgraph alone (no arithmetic, no compile, no
GPU) over real token streams and pin its contract:

* operands of any width up to ``max_digits`` come back MSB-first,
  zero-padded to ``max_digits`` — the implicit padding the old
  ``NumericSequence`` window synthesized, now produced by the
  ``IndexedRegion`` pointer gathers at constant depth;
* explicitly zero-padded prompts remain valid (the full-width special case);
* exactly one operator flag latches, and a ``-`` the model would emit
  *after* the newline cannot re-trigger the operator latch (the flags are
  gated to the pre-newline prompt).
"""

import pytest

from examples._calculator_common import (
    CALC_VOCAB,
    D_HEAD,
    MAX_POSITIONS,
    parse_expression,
)
from torchwright.graph.embedding import bos_token
from torchwright.ops.inout_nodes import create_onehot_embedding, create_rope_config

MAX_DIGITS = 3


@pytest.fixture(scope="module")
def parse():
    embedding = create_onehot_embedding(CALC_VOCAB)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)
    nodes = parse_expression(rope, embedding, MAX_DIGITS)
    return embedding, nodes


def _run(parse, tokens):
    embedding, (first, second, is_plus, is_minus, is_times, _) = parse
    iv = {"embedding_input": tokens}
    n = len(tokens)

    def dec(node):
        return embedding.tokenizer.vocab[int(node.compute(n, iv)[-1].argmax())]

    a = "".join(dec(d) for d in first)
    b = "".join(dec(d) for d in second)
    flags = "".join(
        "+-*"[i]
        for i, f in enumerate((is_plus, is_minus, is_times))
        if f.compute(n, iv)[-1, 0].item() > 0
    )
    return a, b, flags


def _prompt(text):
    return [bos_token] + list(text)


@pytest.mark.parametrize(
    "prompt,a,b,op",
    [
        ("12+7\n", "012", "007", "+"),  # mixed widths
        ("123+456\n", "123", "456", "+"),  # full width
        ("5-13\n", "005", "013", "-"),
        ("99*1\n", "099", "001", "*"),
        ("0+0\n", "000", "000", "+"),  # single digits
        ("007+013\n", "007", "013", "+"),  # zero-padded input stays valid
        ("100-999\n", "100", "999", "-"),
    ],
)
def test_operands_and_operator(parse, prompt, a, b, op):
    assert _run(parse, _prompt(prompt)) == (a, b, op)


def test_emitted_minus_does_not_retrigger_latch(parse):
    # After the newline the model emits "-4" (a negative difference); the
    # latched flags and operand windows must not move.
    tokens = _prompt("1-5\n") + ["-", "4"]
    assert _run(parse, tokens) == ("001", "005", "-")


def test_emitted_digits_do_not_pollute_operands(parse):
    # Emitted answer digits are stream digits too; the region gates must keep
    # them out of the operand windows for the rest of the rollout.
    tokens = _prompt("12+7\n") + ["1", "9"]
    assert _run(parse, tokens) == ("012", "007", "+")
