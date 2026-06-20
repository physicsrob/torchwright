"""Correctness of the one-hot school-math algorithms (reference evaluation).

Each operand is a literal one-hot digit sequence, so ``node.compute`` runs the
exact lookup/fold arithmetic with no inputs.  Results are decoded by argmax
over the identity embedding.
"""

import pytest
import torch

from torchwright.ops import onehot_arithmetic, onehot_arithmetic_fast
from torchwright.ops.inout_nodes import create_literal_value, create_onehot_embedding

VOCAB = [str(d) for d in range(10)]

# One entry per arithmetic implementation; every test below runs once per module
# via the ``arith`` fixture (the legible serial folds and the depth-optimized
# carry-lookahead / carry-save versions must agree digit for digit).
ARITHMETIC_MODULES = [onehot_arithmetic, onehot_arithmetic_fast]


@pytest.fixture(params=ARITHMETIC_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def arith(request):
    return request.param


def _embedding():
    return create_onehot_embedding(vocab=VOCAB)


def _digits(embedding, text, width):
    text = text.zfill(width)
    return [create_literal_value(embedding.get_embedding(c)) for c in text]


def _decode(embedding, digit_nodes):
    chars = []
    for node in digit_nodes:
        v = node.compute(n_pos=1, input_values={})[0]
        chars.append(embedding.tokenizer.vocab[int(v.argmax())])
    return "".join(chars)


# ---------------------------------------------------------------------------
# Addition (caller pads so the carry has a home)
# ---------------------------------------------------------------------------


def test_add(arith):
    e = _embedding()
    cases = [("12", "34", "046"), ("099", "001", "100"), ("000", "000", "000")]
    for a, b, expected in cases:
        out = arith.add_digit_seqs(e, _digits(e, a, 3), _digits(e, b, 3))
        assert _decode(e, out) == expected, (a, b, _decode(e, out))


def test_add_top_carry(arith):
    e = _embedding()
    # 999 + 999 = 1998; widen to 4 digits so the carry lands.
    out = arith.add_digit_seqs(e, _digits(e, "999", 4), _digits(e, "999", 4))
    assert _decode(e, out) == "1998"


# ---------------------------------------------------------------------------
# Subtraction (assumes seq1 >= seq2)
# ---------------------------------------------------------------------------


def test_subtract(arith):
    e = _embedding()
    cases = [("456", "123", "333"), ("100", "001", "099"), ("500", "500", "000")]
    for a, b, expected in cases:
        out = arith.subtract_digit_seqs(e, _digits(e, a, 3), _digits(e, b, 3))
        assert _decode(e, out) == expected, (a, b, _decode(e, out))


# ---------------------------------------------------------------------------
# Comparison (lexicographic, returns +1 if seq1 >= seq2)
# ---------------------------------------------------------------------------


def test_compare(arith):
    e = _embedding()
    cases = [
        ("456", "123", 1.0),
        ("123", "456", -1.0),
        ("100", "100", 1.0),  # equal counts as >=
        ("190", "200", -1.0),  # decided at the most significant differing digit
        ("219", "210", 1.0),
    ]
    for a, b, expected in cases:
        out = arith.compare_digit_seqs(e, _digits(e, a, 3), _digits(e, b, 3))
        got = out.compute(n_pos=1, input_values={})[0, 0]
        assert torch.sign(got).item() == expected, (a, b, got.item())


# ---------------------------------------------------------------------------
# Multiplication (2n digits)
# ---------------------------------------------------------------------------


def test_multiply(arith):
    e = _embedding()
    cases = [("12", "34", "0408"), ("99", "99", "9801"), ("00", "55", "0000")]
    for a, b, expected in cases:
        out = arith.multiply_digit_seqs(e, _digits(e, a, 2), _digits(e, b, 2))
        assert _decode(e, out) == expected, (a, b, _decode(e, out))


def test_multiply_three_digit(arith):
    e = _embedding()
    out = arith.multiply_digit_seqs(e, _digits(e, "123", 3), _digits(e, "456", 3))
    assert _decode(e, out) == "056088"  # 123*456 = 56088, 6 digits
