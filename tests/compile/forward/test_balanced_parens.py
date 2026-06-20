"""Balanced-parentheses checker: export to ONNX and verify detection.

Tests both equal-count detection (shared with token_balance) and underflow
detection (unique to this example — catches cases like ')(' that have equal
counts but invalid nesting).  Converted from ``forward_compile``; see
``_example_onnx``.
"""

import pytest

from examples.balanced_parens import create_network_parts

from ._example_onnx import load_example, run


@pytest.fixture(scope="module")
def parens(tmp_path_factory):
    return load_example(
        create_network_parts, tmp_path_factory.mktemp("parens"), name="parens"
    )


def test_balanced_parens(parens):
    model, artifact = parens
    assert artifact.n_layers <= 60, f"Too many layers: {artifact.n_layers}"

    test_cases = [
        # Balanced
        ("()", "Y"),
        ("(())", "Y"),
        ("()()", "Y"),
        ("((()))", "Y"),
        ("(()())", "Y"),
        ("()()()", "Y"),
        ("(())()", "Y"),
        ("(((())))", "Y"),
        ("", "Y"),
        # Unbalanced — wrong count
        ("(", "N"),
        ("(()", "N"),
        ("())(", "N"),
        # Unbalanced — underflow (equal counts but bad nesting)
        (")(", "N"),
        (")()(", "N"),
        ("()))", "N"),
        ("())()(", "N"),
    ]
    for input_str, expected in test_cases:
        result = run(model, input_str + "\n", bos_token="<bos>")
        assert (
            result == expected
        ), f"For {input_str!r}: expected {expected!r} but got {result!r}"
