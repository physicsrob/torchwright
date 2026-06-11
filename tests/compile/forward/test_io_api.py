"""Tests for the io-dict form of compile_headless.

The io form enables overlaid I/O where output values land at input columns
via delta transfer, enabling autoregressive feedback where the transformer
output IS the next input.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.ops.inout_nodes import (
    create_input,
    create_pos_encoding,
    create_literal_value,
)
from torchwright.ops.arithmetic_ops import add_const, multiply_const, add

# ---------------------------------------------------------------------------
# Basic io API tests
# ---------------------------------------------------------------------------


def test_io_identity():
    """Trivial overlay: output = input (identity)."""
    pos = create_pos_encoding()
    x = create_input(4)

    module = compile_headless(
        {"x": (x, x)},  # output = input
        pos,
        d=64,
        verbose=False,
    )

    inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = module(inp)
    assert torch.allclose(out, inp, atol=0.1)


def test_io_overlaid_single():
    """Single overlaid field: output replaces input."""
    pos = create_pos_encoding()
    x = create_input(4)
    y = multiply_const(x, 2.0)  # y = 2*x

    module = compile_headless(
        {"x": (x, y)},  # x -> y, overlaid
        pos,
        d=64,
        verbose=False,
    )

    inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = module(inp)

    expected = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    assert torch.allclose(out, expected, atol=0.1)


def test_io_overlaid_add_const():
    """Overlaid field with add_const."""
    pos = create_pos_encoding()
    x = create_input(4)
    y = add_const(x, 10.0)  # y = x + 10

    module = compile_headless(
        {"x": (x, y)},
        pos,
        d=64,
        verbose=False,
    )

    inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = module(inp)

    expected = torch.tensor([[11.0, 12.0, 13.0, 14.0]])
    assert torch.allclose(out, expected, atol=0.1)


def test_io_overlaid_multiple():
    """Multiple overlaid fields."""
    pos = create_pos_encoding()
    a = create_input(2)
    b = create_input(3)

    out_a = multiply_const(a, 2.0)
    out_b = add_const(b, 1.0)

    module = compile_headless(
        {
            "a": (a, out_a),  # Overlaid
            "b": (b, out_b),  # Overlaid
        },
        pos,
        d=64,
        verbose=False,
    )

    # Alphabetical: a then b
    inp = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    out = module(inp)

    # out_a=[2,4], out_b=[4,5,6]
    expected = torch.tensor([[2.0, 4.0, 4.0, 5.0, 6.0]])
    assert torch.allclose(out, expected, atol=0.1)


def test_io_input_only_not_in_output():
    """Input-only field doesn't appear in output."""
    pos = create_pos_encoding()
    a = create_input(2)
    b = create_input(2)

    out_b = multiply_const(b, 2.0)

    module = compile_headless(
        {
            "a": (a, None),  # Input-only: NOT in output
            "b": (b, out_b),  # Overlaid
        },
        pos,
        d=64,
        verbose=False,
    )

    # Input: a=[1,2], b=[3,4] (alphabetical)
    inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = module(inp)

    # Output only has b's overlaid result: [6.0, 8.0]
    expected = torch.tensor([[6.0, 8.0]])
    assert torch.allclose(out, expected, atol=0.1)


def test_io_with_overflow():
    """Overlaid + output-only (overflow) fields."""
    pos = create_pos_encoding()
    a = create_input(2)

    out_a = multiply_const(a, 2.0)
    out_extra = create_literal_value(torch.tensor([99.0]))

    module = compile_headless(
        {
            "a": (a, out_a),  # Overlaid at columns [0:2]
            "extra": (None, out_extra),  # Overflow at column [2]
        },
        pos,
        d=64,
        verbose=False,
    )

    inp = torch.tensor([[3.0, 4.0]])
    out = module(inp)

    # Output: [6.0, 8.0, 99.0]
    expected = torch.tensor([[6.0, 8.0, 99.0]])
    assert torch.allclose(out, expected, atol=0.1)


def test_io_autoregressive():
    """Verify overlaid output can be fed directly as next input."""
    pos = create_pos_encoding()
    state = create_input(4)
    next_state = add_const(state, 1.0)

    module = compile_headless(
        {"state": (state, next_state)},
        pos,
        d=64,
        verbose=False,
    )

    current = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    for i in range(5):
        current = module(current)  # Output IS next input
        expected = torch.full((1, 4), float(i + 1))
        assert torch.allclose(
            current, expected, atol=0.1
        ), f"Step {i}: expected {expected}, got {current}"


def test_io_uses_input_value():
    """Overlaid output that depends on the input value."""
    pos = create_pos_encoding()
    x = create_input(2)
    y = create_input(2)

    # Output is x + y, placed at x's columns
    out_xy = add(x, y)

    module = compile_headless(
        {
            "x": (x, out_xy),  # Overlaid: output x+y at x's columns
            "y": (y, None),  # Input-only
        },
        pos,
        d=64,
        verbose=False,
    )

    # Input: x=[1,2], y=[10,20]
    inp = torch.tensor([[1.0, 2.0, 10.0, 20.0]])
    out = module(inp)

    # Output: [11, 22] (x + y)
    expected = torch.tensor([[11.0, 22.0]])
    assert torch.allclose(out, expected, atol=0.1)


# ---------------------------------------------------------------------------
# Error case tests
# ---------------------------------------------------------------------------


def test_io_width_mismatch_error():
    """Overlaid fields must have matching widths."""
    pos = create_pos_encoding()
    a = create_input(4)
    out_a = create_literal_value(torch.zeros(2))  # Wrong width

    with pytest.raises(ValueError, match="width"):
        compile_headless({"a": (a, out_a)}, pos, d=64, verbose=False)


def test_io_empty_tuple_error():
    """Each io entry must have at least one node."""
    pos = create_pos_encoding()

    with pytest.raises(ValueError, match="both input and output as None"):
        compile_headless({"a": (None, None)}, pos, d=64, verbose=False)


def test_io_no_output_error():
    """io must have at least one output."""
    pos = create_pos_encoding()
    x = create_input(4)

    with pytest.raises(ValueError, match="at least one output"):
        compile_headless({"x": (x, None)}, pos, d=64, verbose=False)


# ---------------------------------------------------------------------------
# Calling-convention tests
# ---------------------------------------------------------------------------


def test_node_mode():
    """A bare output Node compiles in node mode (natural-column gather)."""
    pos = create_pos_encoding()
    x = create_input("x", 4)  # node mode: named input
    y = multiply_const(x, 2.0)

    module = compile_headless(y, pos, d=64, verbose=False)

    inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = module(inp)

    expected = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    assert torch.allclose(out, expected, atol=0.1)


def test_output_node_keyword_rejected():
    """The removed output_node= keyword raises TypeError."""
    pos = create_pos_encoding()
    x = create_input(4)
    y = multiply_const(x, 2.0)

    with pytest.raises(TypeError):
        compile_headless({"x": (x, y)}, pos, output_node=y, d=64, verbose=False)


def test_pos_encoding_first_rejected_with_hint():
    """The old (pos_encoding, io) order raises a TypeError naming the fix."""
    pos = create_pos_encoding()
    x = create_input(4)

    with pytest.raises(TypeError, match="graph, pos_encoding"):
        compile_headless(pos, {"x": (x, x)}, d=64, verbose=False)


# ---------------------------------------------------------------------------
# Anonymous input tests
# ---------------------------------------------------------------------------


def test_anonymous_input_with_io():
    """Anonymous inputs work with io API."""
    pos = create_pos_encoding()
    x = create_input(4)  # No name - anonymous

    assert x.name == ""  # Verify it's anonymous

    module = compile_headless(
        {"my_input": (x, x)},  # Name comes from io key
        pos,
        d=64,
        verbose=False,
    )

    inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = module(inp)
    assert torch.allclose(out, inp, atol=0.1)
