"""Tests for the compiler's zero-init assumption on the residual stream.

These tests corrupt the initial residual stream by injecting noise into
columns that the compiler expects to be zero on entry (everything except
input / literal / pos-encoding / embedding assignments).  They should
fail until the compiler is changed to not rely on that assumption.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.ops.arithmetic_ops import add, add_const, multiply_const, relu_add
from torchwright.ops.inout_nodes import create_input


def _patch_noisy_init(
    monkeypatch, noise_seed: int = 42, scale: float = 0.3, bias: float = 0.2
):
    """Replace HeadlessTransformer.get_input_res_stream with a version that
    starts with *noise* in every residual column that is not explicitly
    overwritten by an input / literal / pos / embedding assignment.

    The assigned-column values are unchanged, so if the compiled forward
    pass is correct the final output must also be unchanged.
    """
    orig = HeadlessTransformer.get_input_res_stream

    def noisy(self, n_pos, input_values, past_len=0):
        res_stream = orig(self, n_pos, input_values, past_len)
        in_state = self.layers[0].attn.in_state
        assigned_cols = set()
        for node in self.residual_assignment.get_nodes(in_state):
            indices = self.residual_assignment.get_node_indices(in_state, node)
            assigned_cols.update(int(i) for i in indices)
        unused = [c for c in range(self.d) if c not in assigned_cols]
        g = torch.Generator(device=res_stream.device).manual_seed(noise_seed)
        noise = (
            torch.randn(
                res_stream.shape[0],
                len(unused),
                generator=g,
                device=res_stream.device,
                dtype=res_stream.dtype,
            )
            * scale
            + bias
        )
        res_stream[:, unused] = noise
        return res_stream

    monkeypatch.setattr(HeadlessTransformer, "get_input_res_stream", noisy)


def _run_clean_and_noisy(module, inp, monkeypatch):
    clean = module(inp)
    _patch_noisy_init(monkeypatch)
    noisy = module(inp)
    return clean, noisy


# ---------------------------------------------------------------------------
# Test 1 — intermediate-col path (attention/linear write)
# ---------------------------------------------------------------------------


def test_nonzero_init_intermediate_col(monkeypatch):
    """multiply_const compiles through an attention write to an
    intermediate residual column.  If that column isn't zero at input,
    the additive write produces noise + value, not value."""
    x = create_input(4)
    y = multiply_const(x, 2.0)

    module = compile_headless(y, d=64, verbose=False)
    inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[2.0, 4.0, 6.0, 8.0]])

    clean, noisy = _run_clean_and_noisy(module, inp, monkeypatch)
    assert torch.allclose(clean, expected, atol=0.1)
    assert torch.allclose(
        noisy, expected, atol=0.1
    ), f"noisy init broke output: got {noisy}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 3 — MLP path (Linear → ReLU → Linear)
# ---------------------------------------------------------------------------


def test_nonzero_init_mlp_path(monkeypatch):
    """relu_add compiles to a Linear→ReLU→Linear chain (an MLP op).
    MLP linear2 writes additively to its output column, which the
    compiler assumes starts at zero."""
    x = create_input(4)
    y = relu_add(x, multiply_const(x, -1.0))  # ReLU(x) + ReLU(-x) = |x|

    module = compile_headless(y, d=128, verbose=False)
    inp = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    clean, noisy = _run_clean_and_noisy(module, inp, monkeypatch)
    assert torch.allclose(clean, expected, atol=0.1)
    assert torch.allclose(
        noisy, expected, atol=0.1
    ), f"noisy init broke MLP output: got {noisy}, expected {expected}"
