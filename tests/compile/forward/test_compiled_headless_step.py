"""In-process ``CompiledHeadless.step`` / ``empty_past`` cached-path tests.

``compile_headless`` (node form) returns the in-process ``CompiledHeadless``
debug/test backend — the DebugRuntime behind ``debug=True`` forwards and the
``probe_*`` tools.  These pin its autoregressive surface: ``step`` with an
empty past equals a stateless call, and prefill-then-decode equals a single
full-sequence forward.  (Relocated from the retired ``test_headless_onnx.py``;
the float graph is incidental — ``CompiledHeadless`` runs any graph.)
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.ops.arithmetic_ops import multiply_2d
from torchwright.ops.inout_nodes import create_input

D = 256
D_HEAD = 16


def _build_sample_graph():
    a = create_input("a", 1)
    b = create_input("b", 1)
    out = multiply_2d(a, b, max_abs1=10, max_abs2=10)
    return out


def test_compiled_headless_step_matches_call():
    """step(inputs, empty_past) on a full sequence == module(inputs)."""
    out = _build_sample_graph()
    module = compile_headless(out, d=D, d_head=D_HEAD, verbose=False)

    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs = torch.cat([a_vals, b_vals], dim=1)

    with torch.no_grad():
        full = module(inputs)
        step_out, _ = module.step(inputs, module.empty_past())

    assert torch.allclose(
        full, step_out, atol=1e-4
    ), f"step diff: {(full - step_out).abs().max().item():.6f}"


def test_compiled_headless_step_prefill_decode_matches_full():
    """Prefill 4 + decode 1 matches a single full-sequence forward."""
    out = _build_sample_graph()
    module = compile_headless(out, d=D, d_head=D_HEAD, verbose=False)

    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs = torch.cat([a_vals, b_vals], dim=1)

    with torch.no_grad():
        full = module(inputs)
        past = module.empty_past()
        prefill_out, past = module.step(inputs[:4], past)
        decode_out, past = module.step(inputs[4:5], past)

    assert torch.allclose(
        full[:4], prefill_out, atol=1e-4
    ), f"prefill diff: {(full[:4] - prefill_out).abs().max().item():.6f}"
    assert torch.allclose(
        full[4], decode_out[0], atol=1e-4
    ), f"decode diff: {(full[4] - decode_out[0]).abs().max().item():.6f}"
    # past_K should have grown to n_total = 5
    past_K, _ = past
    assert past_K[0].shape[1] == 5
