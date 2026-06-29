"""V2 scalar-space calculator: export to ONNX and verify +, -, * via argmax
decode (converted from ``forward_compile``; see ``_example_onnx``).
"""

import pytest

from examples.calculator_v2 import D_HEAD

from ._example_onnx import load_example, run


def _build(digits):
    from examples.calculator_v2 import create_network_parts

    return create_network_parts(digits)


@pytest.fixture(scope="module")
def calc_1digit(tmp_path_factory):
    return load_example(
        lambda: _build(1),
        tmp_path_factory.mktemp("calcv2_1"),
        d_head=D_HEAD,
        name="calcv2_1",
    )


@pytest.fixture(scope="module")
def calc_3digit(tmp_path_factory):
    # square splits across layers when needed, so d=1024 suffices
    return load_example(
        lambda: _build(3),
        tmp_path_factory.mktemp("calcv2_3"),
        d_head=D_HEAD,
        name="calcv2_3",
    )


def _check(model, input_str, expected):
    result = run(model, input_str)
    assert (
        result == expected
    ), f"For {input_str!r}: expected {expected!r} but got {result!r}"


# ---------------------------------------------------------------------------
# Phase 1: Addition
# ---------------------------------------------------------------------------


def test_calc_addition_1digit(calc_1digit):
    model, _ = calc_1digit
    _check(model, "1+1\n", "2")
    _check(model, "4+5\n", "9")
    _check(model, "0+0\n", "0")


def test_calc_addition_3digit(calc_3digit):
    model, _ = calc_3digit
    _check(model, "1+1\n", "2")
    _check(model, "123+456\n", "579")
    _check(model, "99+1\n", "100")
    _check(model, "0+0\n", "0")


# ---------------------------------------------------------------------------
# Phase 2: Subtraction
# ---------------------------------------------------------------------------


def test_calc_subtraction_1digit(calc_1digit):
    model, _ = calc_1digit
    _check(model, "5-3\n", "2")
    _check(model, "9-0\n", "9")
    _check(model, "0-0\n", "0")


def test_calc_subtraction_3digit(calc_3digit):
    model, _ = calc_3digit
    _check(model, "456-123\n", "333")
    _check(model, "100-100\n", "0")


def test_calc_subtraction_negative(calc_3digit):
    model, _ = calc_3digit
    _check(model, "1-5\n", "-4")
    _check(model, "100-999\n", "-899")


# ---------------------------------------------------------------------------
# Phase 3: Multiplication
# ---------------------------------------------------------------------------


def test_calc_multiplication_1digit(calc_1digit):
    model, _ = calc_1digit
    _check(model, "2*3\n", "6")
    _check(model, "9*9\n", "81")
    _check(model, "0*5\n", "0")


def test_calc_multiplication_3digit(calc_3digit):
    model, _ = calc_3digit
    _check(model, "12*34\n", "408")
    _check(model, "123*456\n", "56088")
    _check(model, "100*100\n", "10000")
