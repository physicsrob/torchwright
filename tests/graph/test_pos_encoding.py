from torchwright.graph import PosEncoding, InputNode
from torchwright.graph.pos_encoding import trig_shift_matrix
import pytest
import torch

atol = 1.0e-4  # Absolute tolerance for comparing tensors

d_pos = 33
N_pos = 16

pos_encoding_node = PosEncoding(d_pos)
pos_encoding = pos_encoding_node.compute(n_pos=N_pos, input_values={})


def test_layout_properties():
    pe = PosEncoding(17)
    assert pe.trig_width == 16
    assert pe.counter_col == 16
    assert pe.trig_slice == slice(0, 16)


def test_even_d_pos_rejected():
    with pytest.raises(ValueError):
        PosEncoding(16)


def test_counter_column_is_exact_position():
    counter = pos_encoding[:, pos_encoding_node.counter_col]
    assert torch.allclose(counter, torch.arange(N_pos, dtype=counter.dtype))


def test_get_prev_value():
    input_values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    cond_values = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]])
    expected_prev_values = torch.tensor([[1.0], [1.0], [1.0], [4.0], [4.0]])

    value_input = InputNode("value", 1, value_range=(-100.0, 100.0))
    cond_input = InputNode("cond", 1, value_range=(-100.0, 100.0))
    pos_encoding = PosEncoding(17)
    last_input = pos_encoding.get_prev_value(value_input, cond_input)
    output = last_input.compute(
        n_pos=5, input_values={"value": input_values, "cond": cond_values}
    )
    assert torch.allclose(output, expected_prev_values)


def test_attend_to_offset():
    input_values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    value_input = InputNode("value", 1, value_range=(-100.0, 100.0))
    pos_encoding = PosEncoding(17)
    last_input = pos_encoding.attend_to_offset(value_input, delta_pos=-1)
    output = last_input.compute(n_pos=5, input_values={"value": input_values})
    assert torch.allclose(output[0], input_values[0])
    assert torch.allclose(output[1:], input_values[:-1])


def test_attend_to_offset_value_can_be_wider_than_position_encoding():
    input_values = torch.arange(5 * 20, dtype=torch.float32).reshape(5, 20)
    value_input = InputNode("value", 20, value_range=(-100.0, 100.0))
    pos_encoding = PosEncoding(17)
    last_input = pos_encoding.attend_to_offset(value_input, delta_pos=-1)
    output = last_input.compute(n_pos=5, input_values={"value": input_values})
    assert torch.allclose(output[0], input_values[0])
    assert torch.allclose(output[1:], input_values[:-1])


def _check_shift(k: int):
    # Row-convention shift over the trig block: trig(pos) @ S == trig(pos+k).
    tw = pos_encoding_node.trig_width
    trig = pos_encoding[:, :tw]
    shifted = trig @ trig_shift_matrix(k, tw)
    if k >= 0:
        assert torch.allclose(shifted[: N_pos - k], trig[k:], atol=atol)
    else:
        assert torch.allclose(shifted[-k:], trig[: N_pos + k], atol=atol)


def test_trig_shift_matrix_pos1():
    _check_shift(1)


def test_trig_shift_matrix_pos2():
    _check_shift(2)


def test_trig_shift_matrix_neg1():
    _check_shift(-1)
