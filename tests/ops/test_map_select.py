import pytest

from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.arithmetic_ops import add_const
from torchwright.ops.inout_nodes import (
    create_input,
    create_literal_value,
    create_embedding,
)
from torchwright.ops.map_select import (
    select,
    map_to_table,
    switch,
    in_range,
    broadcast_select,
)

import torch


def test_select():
    start = 100.0
    offset = 123.0
    cond_input = create_input("cond", 1)
    x = create_literal_value(torch.tensor([start]))
    x = select(cond=cond_input, true_node=add_const(x, offset), false_node=x)
    for cond in [1.0, -1.0]:
        print("\n\n")
        output = x.compute(n_pos=1, input_values={"cond": torch.tensor([[cond]])})
        expected_value = start + (offset if cond > 0 else 0)
        assert output.tolist() == [[expected_value]]


def test_select_builds_eagerly():
    """select with bounded inputs builds eagerly (no placeholder)."""
    cond_input = create_input("cond", 1)
    t = create_input("t", 1)
    f = create_literal_value(torch.tensor([1.0]))
    result = select(cond_input, t, f)
    assert result._affine_bound is not None
    assert result.value_type.value_range.is_finite()


def test_broadcast_select_exact_branch():
    """Two-sublayer broadcast_select preserves small inputs bit-for-bit at mask=±1."""
    n_slots = 3
    d_fill = 1

    masks_input = create_input("masks", n_slots)
    t = create_input("t", d_fill)
    f = create_input("f", d_fill)
    t_bounded = assert_matches_value_type(
        t, NodeValueType(value_range=Range(-1.0, 1.0))
    )
    f_bounded = assert_matches_value_type(
        f, NodeValueType(value_range=Range(-1.0, 1.0))
    )
    out = broadcast_select(
        masks_input, t_bounded, f_bounded, n_slots, d_fill, approximate=False
    )

    t_val = torch.tensor([[1.0e-5]])
    f_val = torch.tensor([[-1.0e-5]])

    # alternating mask: slot 0 true, slot 1 false, slot 2 true
    masks = torch.tensor([[1.0, -1.0, 1.0]])
    output = out.compute(
        n_pos=1,
        input_values={"masks": masks, "t": t_val, "f": f_val},
    )
    expected = torch.tensor([[1.0e-5, -1.0e-5, 1.0e-5]])
    assert torch.equal(output, expected)


def test_switch():
    """Select one of three values based on which condition is true."""
    c1 = create_input("c1", 1)
    c2 = create_input("c2", 1)
    c3 = create_input("c3", 1)
    v1 = create_literal_value(torch.tensor([10.0, 20.0]))
    v2 = create_literal_value(torch.tensor([30.0, 40.0]))
    v3 = create_literal_value(torch.tensor([50.0, 60.0]))
    out = switch([c1, c2, c3], [v1, v2, v3])

    # Condition 1 true
    result = out.compute(
        n_pos=1,
        input_values={
            "c1": torch.tensor([[1.0]]),
            "c2": torch.tensor([[-1.0]]),
            "c3": torch.tensor([[-1.0]]),
        },
    )
    assert torch.allclose(result, torch.tensor([[10.0, 20.0]]), atol=1e-4)

    # Condition 2 true
    result = out.compute(
        n_pos=1,
        input_values={
            "c1": torch.tensor([[-1.0]]),
            "c2": torch.tensor([[1.0]]),
            "c3": torch.tensor([[-1.0]]),
        },
    )
    assert torch.allclose(result, torch.tensor([[30.0, 40.0]]), atol=1e-4)

    # Condition 3 true
    result = out.compute(
        n_pos=1,
        input_values={
            "c1": torch.tensor([[-1.0]]),
            "c2": torch.tensor([[-1.0]]),
            "c3": torch.tensor([[1.0]]),
        },
    )
    assert torch.allclose(result, torch.tensor([[50.0, 60.0]]), atol=1e-4)


def test_map_to_table():
    table_values = {
        "x": torch.tensor([10.0, 1.0, 1.0]),
        "y": torch.tensor([10.0, 2.0, 2.0]),
        "z": torch.tensor([10.0, 3.0, 3.0]),
    }
    default_value = torch.tensor([-1.0, -1.0, -1.0])

    embedding = create_embedding(vocab=["x", "y", "z"])
    x = map_to_table(
        inp=embedding,
        key_to_value={
            embedding.get_embedding(text): value for text, value in table_values.items()
        },
        default=default_value,
    )
    for text in ["x", "y", "z", "other"]:
        output = x.compute(n_pos=1, input_values={"embedding_input": [text]})
        if text in table_values:
            assert torch.allclose(output[0], table_values[text], atol=1.0e-2)
        else:
            assert torch.allclose(output[0], default_value, atol=1.0e-2)


def test_in_range():
    lower = create_input("lower", 1)
    upper = create_input("upper", 1)
    n_slots = 12

    result = in_range(lower, upper, n_slots)
    assert len(result) == n_slots

    # Interval [4, 8) — positions 4,5,6,7 should be in range
    output = result.compute(
        n_pos=1,
        input_values={
            "lower": torch.tensor([[4.0]]),
            "upper": torch.tensor([[8.0]]),
        },
    )
    expected = torch.tensor([[-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]]).float()
    assert torch.allclose(output, expected, atol=0.1)

    # All in range: [0, 12)
    output = result.compute(
        n_pos=1,
        input_values={
            "lower": torch.tensor([[0.0]]),
            "upper": torch.tensor([[12.0]]),
        },
    )
    expected = torch.ones(1, 12)
    assert torch.allclose(output, expected, atol=0.1)

    # Empty range: lower == upper
    output = result.compute(
        n_pos=1,
        input_values={
            "lower": torch.tensor([[5.0]]),
            "upper": torch.tensor([[5.0]]),
        },
    )
    expected = torch.full((1, 12), -1.0)
    assert torch.allclose(output, expected, atol=0.1)


def test_broadcast_select():
    """Broadcast mode: same true/false value for all slots."""
    n_slots = 6
    d_fill = 3

    masks_input = create_input("masks", n_slots)
    true_val = create_literal_value(torch.tensor([1.0, 0.0, 0.0]))  # red
    false_val = create_literal_value(torch.tensor([0.0, 0.0, 1.0]))  # blue

    result = broadcast_select(masks_input, true_val, false_val, n_slots, d_fill)
    assert len(result) == n_slots * d_fill

    # First 3 slots true, last 3 false
    masks = torch.tensor([[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]])
    output = result.compute(n_pos=1, input_values={"masks": masks})

    expected = torch.tensor(
        [[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]]
    ).float()
    assert torch.allclose(output, expected, atol=0.1)


def test_broadcast_select_per_slot():
    """Per-slot false_value, broadcast true_value."""
    n_slots = 4
    d_fill = 2

    masks_input = create_input("masks", n_slots)
    true_val = create_literal_value(torch.tensor([10.0, 20.0]))  # broadcast
    # Per-slot false values: each slot gets a different pair
    false_val = create_literal_value(
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    )

    result = broadcast_select(masks_input, true_val, false_val, n_slots, d_fill)

    # Slots 1 and 3 true, slots 0 and 2 false
    masks = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])
    output = result.compute(n_pos=1, input_values={"masks": masks})

    # slot 0: false → [1, 2], slot 1: true → [10, 20],
    # slot 2: false → [5, 6], slot 3: true → [10, 20]
    expected = torch.tensor([[1.0, 2.0, 10.0, 20.0, 5.0, 6.0, 10.0, 20.0]])
    assert torch.allclose(output, expected, atol=0.1)
