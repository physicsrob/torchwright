"""Binary increment: export to ONNX and verify correctness.

Tests carry propagation, overflow, and variable-length inputs up to 4 bits.
Converted from ``forward_compile``; see ``_example_onnx``.
"""

import pytest

from examples.binary_increment import create_network_parts

from ._example_onnx import load_example, run


@pytest.fixture(scope="module")
def binary_increment(tmp_path_factory):
    return load_example(
        create_network_parts, tmp_path_factory.mktemp("binc"), name="binc"
    )


def test_binary_increment(binary_increment):
    model, artifact = binary_increment
    assert artifact.n_layers <= 40, f"Too many layers: {artifact.n_layers}"

    test_cases = [
        # Simple increments
        ("0", "1"),
        ("1", "10"),
        ("10", "11"),
        ("11", "100"),
        ("100", "101"),
        ("101", "110"),
        ("110", "111"),
        ("111", "1000"),
        # Multi-bit carry propagation
        ("1011", "1100"),
        ("1111", "10000"),
    ]
    for binary_in, expected in test_cases:
        result = run(model, binary_in + "\n", bos_token="<bos>")
        assert (
            result == expected
        ), f"For {binary_in!r}: expected {expected!r} but got {result!r}"
