"""swiglu compositions: the digit pipeline, embedding arithmetic, and
sequence predicates.

These are structural ports — every MLP ingredient is a landed swiglu op
and inherits its entry; the tests pin end-to-end behavior on the same
cases the relu example tests use.  The full autoregressive patterns
(NumericSequence/output_sequence over token streams) are exercised at
Phase C (examples cutover); here the pure-graph pieces are validated.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.ops.inout_nodes import create_embedding, create_input
from torchwright.ops.swiglu import (
    check_is_digit,
    digit_to_scaled_scalar,
    digits_to_number,
    number_to_digit_scalars,
    remove_leading_0s,
    scalar_to_embedding,
    sum_digit_seqs,
)

D = 128
D_HEAD = 8


@pytest.fixture
def embedding():
    vocab = list("0123456789+-=") + ["<eos>", "default"]
    return create_embedding(vocab=vocab)


def test_digit_to_scaled_scalar(embedding):
    node = digit_to_scaled_scalar(embedding, embedding, place_value=100.0)
    for digit in (0, 3, 9):
        result = node.compute(n_pos=1, input_values={"embedding_input": [str(digit)]})
        assert result[0, 0].item() == pytest.approx(digit * 100.0, abs=1e-2)


def test_digits_to_number(embedding):
    node = digits_to_number(embedding, [embedding, embedding, embedding])
    # All three digit nodes read the same stream position here; feeding
    # "5" gives 5*100 + 5*10 + 5 = 555.
    result = node.compute(n_pos=1, input_values={"embedding_input": ["5"]})
    assert result[0, 0].item() == pytest.approx(555.0, abs=1e-2)


def test_number_to_digit_scalars_roundtrip():
    x = create_input("x", 1, value_range=(0.0, 999.0))
    digits = number_to_digit_scalars(x, num_digits=3, max_value=999)
    xs = torch.tensor([[579.0], [40.0], [999.0], [0.0]])
    vals = [d.compute(4, {"x": xs}) for d in digits]
    expected = [
        [5.0, 0.0, 9.0, 0.0],
        [7.0, 4.0, 9.0, 0.0],
        [9.0, 0.0, 9.0, 0.0],
    ]
    # abs=1e-2: on GPU, the remainder subtraction amplifies staircase
    # matmul noise by the place value (100·~2e-5 ≈ 2e-3 — the
    # _lookup_numeric_slack GPU class); the downstream consumer
    # (scalar_to_embedding) tolerates ±0.4.
    for place, exp in enumerate(expected):
        for row, e in enumerate(exp):
            assert vals[place][row, 0].item() == pytest.approx(e, abs=1e-2), (
                place,
                row,
            )


def test_full_digit_pipeline_compiles(embedding):
    """digits_to_number → arithmetic → number_to_digit_scalars →
    scalar_to_embedding, compiled end to end on the swish machine."""
    from torchwright.ops.linear import add_const

    number = digits_to_number(embedding, [embedding])
    shifted = add_const(number, 3.0)
    digits = number_to_digit_scalars(shifted, num_digits=2, max_value=12)
    out = scalar_to_embedding(digits[1], embedding)
    compiled = compile_headless(out, d=2048, d_head=32)
    assert compiled._net.activation == "swish"
    ids = torch.tensor([embedding.tokenizer.get_token_id("4")])
    report = probe_compiled(compiled, out, {"embedding_input": ids}, 1, atol=1e-2)
    assert report.first_divergent is None, report.format_short()
    # 4 + 3 = 7; ones digit 7 → embed("7")
    val = out.compute(1, {"embedding_input": ["4"]})
    assert torch.allclose(val[0], embedding.get_embedding("7"), atol=1e-3)


def test_sum_digit_seqs(embedding):
    seqs = sum_digit_seqs(embedding, [embedding, embedding], [embedding, embedding])
    # Feeding "7": 77 + 77 = 154 → kept digits (no overflow digit): "5", "4"
    v0 = seqs[0].compute(1, {"embedding_input": ["7"]})
    v1 = seqs[1].compute(1, {"embedding_input": ["7"]})
    assert torch.allclose(v0[0], embedding.get_embedding("5"), atol=1e-3)
    assert torch.allclose(v1[0], embedding.get_embedding("4"), atol=1e-3)


def test_check_is_digit(embedding):
    node = check_is_digit(embedding)
    for tok, exp in (("0", 1.0), ("9", 1.0), ("+", -1.0), ("<eos>", -1.0)):
        val = node.compute(1, {"embedding_input": [tok]})
        assert val.item() == pytest.approx(exp, abs=1e-3), tok


def test_remove_leading_0s(embedding):
    zero = digit_to_scaled_scalar(embedding, embedding, 1.0)  # unused warm-up
    del zero
    from torchwright.ops.inout_nodes import create_literal_value

    seq = [
        create_literal_value(embedding.get_embedding("0")),
        create_literal_value(embedding.get_embedding("4")),
        create_literal_value(embedding.get_embedding("2")),
    ]
    out = remove_leading_0s(embedding, seq, max_removals=1)
    v = [n.compute(1, {"embedding_input": ["0"]}) for n in out]
    assert torch.allclose(v[0][0], embedding.get_embedding("4"), atol=1e-3)
    assert torch.allclose(v[1][0], embedding.get_embedding("2"), atol=1e-3)
    assert torch.allclose(v[2][0], embedding.get_embedding("2"), atol=1e-3)
