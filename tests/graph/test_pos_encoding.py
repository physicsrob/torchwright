from torchwright.graph import PosEncoding, InputNode
from torchwright.graph.pos_encoding import get_pos_delta_matrix
import torch

atol = 1.0e-4  # Absolute tolerance for comparing tensors

d_pos = 32
N_pos = 16

pos_encoding_node = PosEncoding(d_pos)
pos_encoding = pos_encoding_node.compute(n_pos=N_pos, input_values={})


def test_get_prev_value():
    input_values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    cond_values = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]])
    expected_prev_values = torch.tensor([[1.0], [1.0], [1.0], [4.0], [4.0]])

    value_input = InputNode("value", 1, value_range=(-100.0, 100.0))
    cond_input = InputNode("cond", 1, value_range=(-100.0, 100.0))
    pos_encoding = PosEncoding(16)
    last_input = pos_encoding.get_prev_value(value_input, cond_input)
    output = last_input.compute(
        n_pos=5, input_values={"value": input_values, "cond": cond_values}
    )
    assert torch.allclose(output, expected_prev_values)


def test_attend_to_offset():
    input_values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    value_input = InputNode("value", 1, value_range=(-100.0, 100.0))
    pos_encoding = PosEncoding(16)
    last_input = pos_encoding.attend_to_offset(value_input, delta_pos=-1)
    output = last_input.compute(n_pos=5, input_values={"value": input_values})
    assert torch.allclose(output[0], input_values[0])
    assert torch.allclose(output[1:], input_values[:-1])


def test_attend_to_offset_value_can_be_wider_than_position_encoding():
    input_values = torch.arange(5 * 20, dtype=torch.float32).reshape(5, 20)
    value_input = InputNode("value", 20, value_range=(-100.0, 100.0))
    pos_encoding = PosEncoding(16)
    last_input = pos_encoding.attend_to_offset(value_input, delta_pos=-1)
    output = last_input.compute(n_pos=5, input_values={"value": input_values})
    assert torch.allclose(output[0], input_values[0])
    assert torch.allclose(output[1:], input_values[:-1])


def test_delta_matrix():
    # The last pair (d_pos-2, d_pos-1) contains the counter — exclude it.
    s = d_pos - 2
    delta_matrix = get_pos_delta_matrix(delta_pos=1, d=d_pos)[:s, :s]
    sin_part = pos_encoding[:, :s]
    delta_pos_encoding = sin_part @ delta_matrix
    assert torch.allclose(delta_pos_encoding[1:, :], sin_part[:-1, :], atol=atol)


def test_delta_matrix2():
    s = d_pos - 2
    delta_matrix = get_pos_delta_matrix(delta_pos=2, d=d_pos)[:s, :s]
    sin_part = pos_encoding[:, :s]
    delta_pos_encoding = sin_part @ delta_matrix
    assert torch.allclose(delta_pos_encoding[2:, :], sin_part[:-2, :], atol=atol)


def test_delta_matrix_neg():
    s = d_pos - 2
    delta_matrix = get_pos_delta_matrix(delta_pos=-1, d=d_pos)[:s, :s]
    sin_part = pos_encoding[:, :s]
    delta_pos_encoding = sin_part @ delta_matrix
    assert torch.allclose(delta_pos_encoding[:-1, :], sin_part[1:, :], atol=atol)

    assert torch.allclose(delta_pos_encoding[1], sin_part[2], atol=atol)
