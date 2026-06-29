"""Scale tests for thermometer_floor_div and mod_const.

These ops build a piecewise-linear staircase with
``n = max_value // divisor`` steps.  The game graph invokes them with
``max_value = W * (H // rows_per_patch)`` — at the DOOM shipping scale
(W=160, H=100, rp=10) that is 1600, so the staircase has 160 steps.
The pre-existing test_arithmetic_ops.test_mod_const only exercises
``max_value <= 15``, and runs via the uncompiled graph.compute() path,
so the compiler's handling of wide staircases is unverified.

These tests feed a clean integer input (via create_input, not
position_scalar) through a minimally-compiled headless module and
check every integer in ``[0, max_value]`` against the ground truth.
Isolating from position_scalar keeps this focused on the staircase
op itself; see tests/graph/test_position_scalar.py for the separate
position_scalar accuracy tests.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.arithmetic_ops import mod_const, thermometer_floor_div
from torchwright.ops.inout_nodes import create_input

# Representative (divisor, max_value) sweep:
#   (2, 32)    — baseline, small divisor/small scale
#   (10, 320)  — 32-step staircase, divisor 10 (mid scale)
#   (10, 1600) — new DOOM default shape (shards_per_col=10, W=160), full scale
SCALE_CASES = [
    (2, 32),
    (10, 320),
    (10, 1600),
]


def _compile_unary(op_fn, input_name: str = "x", d: int = 1024):
    """Compile a module computing ``y = op_fn(x)`` on a 1D scalar input.

    The module takes a (seq_len, 1) input tensor and returns a
    (seq_len, 1) output tensor.  We pack the output in a Concatenate so
    compile_headless always sees a proper output node even when op_fn
    returns a single-width node.
    """
    x = create_input(input_name, 1)
    y = op_fn(x)
    output = Concatenate([y])
    return compile_headless(
        output,
        d=d,
        d_head=16,
        max_layers=20,
        verbose=False,
    )


def _run_module_on_integer_range(module, max_value: int) -> torch.Tensor:
    """Feed every integer in [0, max_value] through the module as one
    prefill pass and return the (max_value+1,) output row."""
    inputs = torch.arange(max_value + 1, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        return module(inputs).squeeze(-1)


@pytest.mark.parametrize("divisor,max_value", SCALE_CASES)
def test_thermometer_floor_div_exhaustive(divisor, max_value):
    """For every integer v in [0, max_value], floor_div(v, divisor) == v // divisor."""
    module = _compile_unary(
        lambda x: thermometer_floor_div(x, divisor, max_value),
    )
    outputs = _run_module_on_integer_range(module, max_value)
    expected = torch.tensor(
        [v // divisor for v in range(max_value + 1)],
        dtype=outputs.dtype,
    )
    diff = (outputs - expected).abs()
    max_err = diff.max().item()
    argmax = int(diff.argmax().item())
    assert max_err < 0.5, (
        f"thermometer_floor_div(divisor={divisor}, max_value={max_value}): "
        f"max error {max_err:.3f} at v={argmax} "
        f"(got {outputs[argmax].item():.3f}, expected {expected[argmax].item():.0f})"
    )


@pytest.mark.parametrize("divisor,max_value", SCALE_CASES)
def test_mod_const_exhaustive(divisor, max_value):
    """For every integer v in [0, max_value], mod_const(v, divisor) == v % divisor."""
    module = _compile_unary(
        lambda x: mod_const(x, divisor, max_value),
    )
    outputs = _run_module_on_integer_range(module, max_value)
    expected = torch.tensor(
        [v % divisor for v in range(max_value + 1)],
        dtype=outputs.dtype,
    )
    diff = (outputs - expected).abs()
    max_err = diff.max().item()
    argmax = int(diff.argmax().item())
    assert max_err < 0.5, (
        f"mod_const(divisor={divisor}, max_value={max_value}): "
        f"max error {max_err:.3f} at v={argmax} "
        f"(got {outputs[argmax].item():.3f}, expected {expected[argmax].item():.0f})"
    )


@pytest.mark.parametrize("divisor,max_value", SCALE_CASES)
def test_mod_const_crosses_every_wraparound(divisor, max_value):
    """Stricter check: every ``v`` that sits immediately on either side of a
    multiple of ``divisor`` is within 0.25 of the ground truth.  This
    catches drift near the staircase transitions that a looser tolerance
    might hide."""
    module = _compile_unary(
        lambda x: mod_const(x, divisor, max_value),
    )
    outputs = _run_module_on_integer_range(module, max_value)

    critical = set()
    for k in range(1, max_value // divisor + 1):
        boundary = k * divisor
        for v in (boundary - 1, boundary, boundary + 1):
            if 0 <= v <= max_value:
                critical.add(v)
    critical = sorted(critical)
    for v in critical:
        got = outputs[v].item()
        expected = v % divisor
        assert abs(got - expected) < 0.25, (
            f"mod_const(divisor={divisor}, max_value={max_value}) at wraparound "
            f"v={v}: got {got:.3f}, expected {expected}"
        )
