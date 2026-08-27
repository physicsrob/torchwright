"""End-to-end continuous HF bundle and recurrent runtime tests."""

import json

import pytest
import torch
from safetensors.torch import load_file

import torchwright
from torchwright.compiler.hf import (
    CONTINUOUS_BASE_FILENAME,
    CONTINUOUS_IO_FILENAME,
    CONTINUOUS_IO_FORMAT,
    ContinuousRunner,
    compile_continuous_hf_bundle,
)
from torchwright.graph import Add, InputNode, LiteralValue
from torchwright.ops.swiglu import multiply

_D = 64
_D_HEAD = 8
_N_POSITIONS = 3


def test_continuous_api_is_available_from_top_level() -> None:
    assert torchwright.ContinuousRunner is ContinuousRunner
    assert torchwright.compile_continuous_hf_bundle is compile_continuous_hf_bundle


def _transition_graph():
    state = InputNode("state", 2, value_range=(-10.0, 10.0))
    update = InputNode("update", 2, value_range=(-10.0, 10.0))
    offset = LiteralValue(torch.tensor([1.0, -2.0]))
    next_state = Add(Add(state, update), offset)
    converged = LiteralValue(torch.tensor([1.0]))
    return state, update, next_state, converged


@pytest.fixture(scope="module")
def continuous_bundle(tmp_path_factory):
    _state, _update, next_state, converged = _transition_graph()
    path = tmp_path_factory.mktemp("continuous") / "bundle"
    report = compile_continuous_hf_bundle(
        {"state": next_state, "converged": converged},
        path,
        n_positions=_N_POSITIONS,
        d=_D,
        d_head=_D_HEAD,
        rms_norm=True,
        verbose=False,
    )
    return path, report


def test_bundle_persists_named_layout_and_static_residual(continuous_bundle) -> None:
    path, report = continuous_bundle
    spec = json.loads((path / CONTINUOUS_IO_FILENAME).read_text())

    assert report.n_positions == _N_POSITIONS
    assert report.n_layers > 0
    assert spec["format"] == CONTINUOUS_IO_FORMAT
    assert spec["n_positions"] == _N_POSITIONS
    assert spec["d_model"] == _D
    assert set(spec["inputs"]) == {"state", "update"}
    assert set(spec["outputs"]) == {"state", "converged"}
    assert spec["inputs"]["state"]["shape"] == [_N_POSITIONS, 2]
    assert spec["inputs"]["state"]["dtype"] == "float32"
    assert len(spec["inputs"]["state"]["residual_columns"]) == 2
    assert (path / CONTINUOUS_BASE_FILENAME).is_file()
    base = load_file(path / CONTINUOUS_BASE_FILENAME)["base_residual"]
    assert base.abs().max() > 1e10
    torch.testing.assert_close(base[0], base[-1])
    assert not list(path.glob(".tensor_*.npy"))


def test_runner_matches_graph_values_and_supports_batching(continuous_bundle) -> None:
    path, _report = continuous_bundle
    runner = ContinuousRunner.from_pretrained(path)
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-2.0, 8.0]])
    update = torch.tensor([[0.5, -1.0], [2.0, 3.0], [1.0, -4.0]])

    result = runner(state=state, update=update)
    expected = state + update + torch.tensor([1.0, -2.0])
    torch.testing.assert_close(result["state"], expected)
    torch.testing.assert_close(result["converged"], torch.ones(_N_POSITIONS, 1))

    batched = runner(
        state=torch.stack([state, state * 2]),
        update=torch.stack([update, update * 3]),
    )
    expected_batch = torch.stack(
        [
            expected,
            state * 2 + update * 3 + torch.tensor([1.0, -2.0]),
        ]
    )
    torch.testing.assert_close(batched["state"], expected_batch)


def test_runner_validates_names_shapes_and_batching(continuous_bundle) -> None:
    path, _report = continuous_bundle
    runner = ContinuousRunner.from_pretrained(path)
    state = torch.zeros(_N_POSITIONS, 2)
    update = torch.zeros(_N_POSITIONS, 2)

    with pytest.raises(ValueError, match="missing"):
        runner(state=state)
    with pytest.raises(ValueError, match="unexpected"):
        runner(state=state, update=update, extra=update)
    with pytest.raises(ValueError, match="end in shape"):
        runner(state=torch.zeros(2, 2), update=update)
    with pytest.raises(ValueError, match="consistent batching"):
        runner(state=state.unsqueeze(0), update=update)


def test_run_until_feeds_state_output_into_fresh_invocation(continuous_bundle) -> None:
    path, _report = continuous_bundle
    runner = ContinuousRunner.from_pretrained(path)
    state = torch.zeros(_N_POSITIONS, 2)
    update = torch.ones(_N_POSITIONS, 2)

    result = runner.run_until(state, max_steps=5, update=update)

    torch.testing.assert_close(
        result["state"], state + update + torch.tensor([1.0, -2.0])
    )


def test_swiglu_bundle_executes_continuous_product(tmp_path) -> None:
    """SwiGLU continuous export preserves the range-free gated product."""
    left = InputNode("left", 1, value_range=(-1000.0, 1000.0))
    right = InputNode("right", 1, value_range=(-1000.0, 1000.0))
    path = tmp_path / "swiglu"
    compile_continuous_hf_bundle(
        {"product": multiply(left, right)},
        path,
        n_positions=_N_POSITIONS,
        d=_D,
        d_head=_D_HEAD,
        verbose=False,
    )
    config = json.loads((path / "config.json").read_text())
    assert config["activation"] == "swish"
    runner = ContinuousRunner.from_pretrained(path)
    left_values = torch.tensor([[3.5], [-8.25], [999.0]])
    right_values = torch.tensor([[-2.0], [0.125], [-777.0]])

    result = runner(left=left_values, right=right_values)

    torch.testing.assert_close(
        result["product"], left_values * right_values, rtol=3e-7, atol=1e-4
    )
