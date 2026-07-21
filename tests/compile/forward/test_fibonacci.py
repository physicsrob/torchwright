"""Fibonacci generator: export to ONNX and verify correctness.

Tests that the autoregressive recurrence produces correct Fibonacci numbers.
The input prompt must have enough tokens before \\n to avoid out-of-bounds
attention (at least n_terms * digit_width tokens).  Converted from
``forward_compile``; see ``_example_onnx``.
"""

import pytest

from examples.fibonacci import D_HEAD, D_MODEL, create_network_parts

from ._example_onnx import load_example, run


@pytest.fixture(scope="module")
def fibonacci(tmp_path_factory):
    return load_example(
        create_network_parts,
        tmp_path_factory.mktemp("fib"),
        d=D_MODEL,
        d_head=D_HEAD,
        name="fib",
    )


def test_fibonacci(fibonacci):
    model, artifact = fibonacci
    assert artifact.n_layers <= 50, f"Too many layers: {artifact.n_layers}"

    # Use "fibonacci" as the prompt — 9 letters provides enough tokens
    # before \n to avoid OOB attention for the output entries.
    # Expected: 8 Fibonacci terms, each zero-padded to 2 digits.
    # F = 1, 1, 2, 3, 5, 8, 13, 21
    expected_fibs = [1, 1, 2, 3, 5, 8, 13, 21]
    expected = "".join(f"{f:02d}" for f in expected_fibs)

    result = run(model, "fibonacci\n", bos_token="<bos>", max_new_tokens=20)
    assert result == expected, f"Expected {expected!r} but got {result!r}"
