"""Token-balance checker: export to ONNX and verify equal-count detection.

This validates the parallel prefix sum + position-gating strategy that the
balanced-parentheses checker builds on.  Converted from ``forward_compile``;
see ``_example_onnx``.
"""

import pytest

from examples.token_balance import create_network_parts

from ._example_onnx import load_example, run


@pytest.fixture(scope="module")
def token_balance(tmp_path_factory):
    return load_example(
        create_network_parts, tmp_path_factory.mktemp("tokbal"), name="tokbal"
    )


def test_token_balance(token_balance):
    model, artifact = token_balance
    assert artifact.n_layers <= 40, f"Too many layers: {artifact.n_layers}"

    test_cases = [
        # Balanced
        ("ab", "Y"),
        ("ba", "Y"),
        ("aabb", "Y"),
        ("abab", "Y"),
        ("abba", "Y"),
        ("aaabbb", "Y"),
        ("ababab", "Y"),
        ("", "Y"),
        # Unbalanced
        ("a", "N"),
        ("b", "N"),
        ("aab", "N"),
        ("abb", "N"),
        ("aaab", "N"),
        ("abbb", "N"),
    ]
    for input_str, expected in test_cases:
        result = run(model, input_str + "\n", bos_token="<bos>")
        assert (
            result == expected
        ), f"For {input_str!r}: expected {expected!r} but got {result!r}"
