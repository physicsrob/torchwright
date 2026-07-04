from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.inout_nodes import create_literal_value, create_input
from torchwright.ops.relu.logic_ops import (
    equals_vector,
    cond_gate,
    bool_any_true,
    bool_all_true,
    bool_not,
)

import torch


def test_equals_vector():
    for i in range(10):
        for j in range(10):
            x = create_literal_value(index_to_vector(i))
            c = index_to_vector(j)
            y = equals_vector(x, c)
            output = y.compute(n_pos=1, input_values={})
            expected_output = torch.tensor(1.0 if i == j else -1.0)
            assert torch.allclose(output, expected_output, atol=1.0e-3)


def test_cond_gate():
    x = create_input("x", 1)
    x_bounded = assert_matches_value_type(
        x, NodeValueType(value_range=Range(-2.0, 2.0))
    )
    cond_input = create_input("cond", 1)
    out = cond_gate(cond_input, x_bounded)
    for cond_value in [-1.0, 1.0]:
        for x_value in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            output = out.compute(
                n_pos=1,
                input_values={
                    "cond": torch.tensor([[cond_value]]),
                    "x": torch.tensor([[x_value]]),
                },
            )
            if cond_value > 0.0:
                expected_value = x_value
            else:
                expected_value = 0.0
            assert output.item() == expected_value


def test_cond_gate_adaptive_M_uses_value_range():
    """cond_gate picks M from inp.value_type.value_range, so small inputs survive
    that would be lost under the old global big_offset=1000."""
    x = create_input("x", 1)
    x_bounded = assert_matches_value_type(
        x, NodeValueType(value_range=Range(-1.0, 1.0))
    )
    cond_input = create_input("cond", 1)
    out = cond_gate(cond_input, x_bounded)

    small = 1.0e-5
    output = out.compute(
        n_pos=1,
        input_values={
            "cond": torch.tensor([[1.0]]),
            "x": torch.tensor([[small]]),
        },
    )
    # M=1 here, so ULP(M)≈1.2e-7; 1e-5 survives cancellation cleanly.
    assert abs(output.item() - small) < 1.0e-6


def test_cond_gate_builds_eagerly():
    """cond_gate with bounded inputs builds eagerly (no placeholder)."""
    x = create_input("x", 1)
    cond_input = create_input("cond", 1)
    result = cond_gate(cond_input, x)
    assert result._affine_bound is not None
    assert result.value_type.value_range.is_finite()


def test_bool_any_true():
    x = create_input("x", 1)
    y = create_input("y", 1)
    z = create_input("z", 1)
    out = bool_any_true([x, y, z])
    for x_value in [-1.0, 1.0]:
        for y_value in [-1.0, 1.0]:
            for z_value in [-1.0, 1.0]:
                output = out.compute(
                    n_pos=1,
                    input_values={
                        "x": torch.tensor([[x_value]]),
                        "y": torch.tensor([[y_value]]),
                        "z": torch.tensor([[z_value]]),
                    },
                )
                expected_value = (
                    1.0 if (x_value > 0.0 or y_value > 0.0 or z_value > 0.0) else -1.0
                )
                assert output.item() == expected_value


def test_bool_all_true():
    x = create_input("x", 1)
    y = create_input("y", 1)
    z = create_input("z", 1)
    out = bool_all_true([x, y, z])
    for x_value in [-1.0, 1.0]:
        for y_value in [-1.0, 1.0]:
            for z_value in [-1.0, 1.0]:
                output = out.compute(
                    n_pos=1,
                    input_values={
                        "x": torch.tensor([[x_value]]),
                        "y": torch.tensor([[y_value]]),
                        "z": torch.tensor([[z_value]]),
                    },
                )
                expected_value = (
                    1.0 if (x_value > 0.0 and y_value > 0.0 and z_value > 0.0) else -1.0
                )
                assert output.item() == expected_value


def test_bool_not():
    x = create_input("x", 1)
    out = bool_not(x)
    for x_value in [-1.0, 1.0]:
        output = out.compute(
            n_pos=1,
            input_values={
                "x": torch.tensor([[x_value]]),
            },
        )
        expected_value = 1.0 if x_value < 0.0 else -1.0
        assert output.item() == expected_value
