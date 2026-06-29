"""Adder example: export 1-digit and 3-digit adders to ONNX and verify
arithmetic via argmax decode.

Converted from the in-process ``forward_compile`` path to ``compile_to_onnx``
+ ``OnnxTokenModule.generate``; see ``_example_onnx`` for why the decoded
output is token-identical to the old nearest-embedding decode.
"""

import pytest

from examples.adder import create_network_parts

from ._example_onnx import load_example, run


def _build_1digit():
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        return create_network_parts()
    finally:
        adder_module.max_digits = original


@pytest.fixture(scope="module")
def adder_1digit(tmp_path_factory):
    return load_example(_build_1digit, tmp_path_factory.mktemp("adder1"), name="adder1")


@pytest.fixture(scope="module")
def adder_3digit(tmp_path_factory):
    return load_example(
        create_network_parts, tmp_path_factory.mktemp("adder3"), name="adder3"
    )


# ---------------------------------------------------------------------------
# 1-digit adder
# ---------------------------------------------------------------------------


def test_1digit_adder(adder_1digit):
    model, artifact = adder_1digit
    # RoPE recency: the octant ramp + two graded {BOS, REF} heads behind
    # recency_rank are structurally deeper than the old counter column, so the
    # budget is higher than the pre-port ≤20 (docs/rope_port_plan.md §8 Phase 5).
    assert artifact.n_layers <= 50, f"Too many layers: {artifact.n_layers}"

    test_cases = [
        ("1+1\n", "2"),
        ("2+3\n", "5"),
        ("0+0\n", "0"),
        ("4+5\n", "9"),
        ("7+2\n", "9"),
        ("6+3\n", "9"),
    ]
    for input_str, expected in test_cases:
        result = run(model, input_str, ref_token="<ref>")
        assert (
            result == expected
        ), f"For {input_str!r}: expected {expected!r} but got {result!r}"


# ---------------------------------------------------------------------------
# 3-digit adder
# ---------------------------------------------------------------------------


def test_3digit_adder(adder_3digit):
    model, artifact = adder_3digit
    assert artifact.n_layers <= 50, f"Too many layers: {artifact.n_layers}"

    test_cases = [
        ("1+1\n", "2"),
        ("12+34\n", "46"),
        ("123+456\n", "579"),
        ("100+200\n", "300"),
        ("0+0\n", "0"),
        ("99+1\n", "100"),
    ]
    for input_str, expected in test_cases:
        result = run(model, input_str, ref_token="<ref>")
        assert (
            result == expected
        ), f"For {input_str!r}: expected {expected!r} but got {result!r}"


def test_3digit_autoregressive(adder_3digit):
    model, _ = adder_3digit
    test_cases = [
        ("1+2\n", "3"),
        ("99+1\n", "100"),
        ("100+200\n", "300"),
        ("111+222\n", "333"),
        ("456+123\n", "579"),
    ]
    for input_str, expected in test_cases:
        result = run(model, input_str, ref_token="<ref>")
        assert (
            result == expected
        ), f"For {input_str!r}: expected {expected!r} but got {result!r}"
