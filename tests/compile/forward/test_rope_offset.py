"""Phase-0 RoPE validation: a pure-rotary offset head.

The Phase-0 gate (``docs/rope_port_plan.md`` §8): reimplement
``attend_to_offset(-1)`` as a rotary head and prove (a) the oracle selects the
same key as the existing trig-shift, (b) the compiled head matches the oracle
(``probe_compiled`` — the R15 oracle-first contract), and (c) prefill and
unbounded cached decode produce identical offset-head output (the cache-rotation
invariant: K is stored already-rotated, Q rotates by absolute position).

The head transports a per-position payload, so ``output[m]`` equals the
*previous* position's payload (and the position-0 payload at position 0, which
attends to itself under the causal mask).  That makes the selection directly
observable.
"""

import os
import tempfile

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph.rope import rotary_offset_head
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

N_POS = 12
D = 256
D_HEAD = 16


def _payload(n_pos: int) -> torch.Tensor:
    # Distinct per-position values so "attend one back" is observable.
    return (100.0 + torch.arange(n_pos, dtype=torch.float32)).unsqueeze(1)


def _expected_prev(payload: torch.Tensor) -> torch.Tensor:
    # output[0] = payload[0] (self under causal mask); output[m] = payload[m-1].
    shifted = payload.squeeze(1).roll(1)
    shifted[0] = payload[0, 0]
    return shifted


def test_rotary_offset_oracle_selects_prev_position():
    """The rotary head's oracle selects the previous position, token-identical
    to the existing trig-shift ``attend_to_offset(-1)``."""
    pos = create_pos_encoding()
    payload = create_input("payload", 1)
    vals = _payload(N_POS)

    rotary = rotary_offset_head(payload, delta_pos=-1)
    trig = pos.attend_to_offset(payload, delta_pos=-1)

    rotary_out = rotary.compute(N_POS, {"payload": vals}).squeeze(1)
    trig_out = trig.compute(N_POS, {"payload": vals}).squeeze(1)
    expected = _expected_prev(vals)

    assert torch.allclose(rotary_out, expected, atol=1e-3), rotary_out
    # Token-identical selection vs the trig-shift head.
    assert torch.allclose(rotary_out, trig_out, atol=1e-3), (rotary_out, trig_out)


def test_rotary_offset_compiled_matches_oracle():
    """probe_compiled: the compiled rotary head matches its oracle everywhere
    (no divergent node) — the oracle-first / R15 contract."""
    pos = create_pos_encoding()
    payload = create_input("payload", 1)
    vals = _payload(N_POS)
    rotary = rotary_offset_head(payload, delta_pos=-1)

    compiled = compile_headless(rotary, pos, d=D, d_head=D_HEAD, verbose=False)
    report = probe_compiled(compiled, rotary, {"payload": vals}, N_POS, atol=1e-2)
    assert report.first_divergent is None, report.format_short()

    out = compiled(vals.to(compiled._net.device)).squeeze(1).cpu()
    assert torch.allclose(out, _expected_prev(vals), atol=1e-2), out


def test_rotary_offset_prefill_decode_identical():
    """Prefill and token-by-token cached decode produce identical offset-head
    output — the cache-rotation invariant (K stored rotated, Q rotated by
    absolute position)."""
    pos = create_pos_encoding()
    payload = create_input("payload", 1)
    vals = _payload(N_POS)
    rotary = rotary_offset_head(payload, delta_pos=-1)

    compiled = compile_headless(rotary, pos, d=D, d_head=D_HEAD, verbose=False)
    device = compiled._net.device

    full = compiled(vals.to(device)).squeeze(1).cpu()

    past = compiled.empty_past()
    decoded = []
    for t in range(N_POS):
        out_t, past = compiled.step(vals[t : t + 1].to(device), past)
        decoded.append(out_t.squeeze(0).squeeze(0).item())
    decoded = torch.tensor(decoded)

    assert torch.allclose(full, decoded, atol=1e-4), (full, decoded)


# The ONNX/HF runtime paths apply RoPE full-width (rotate_half over d_head), so
# the rotary head must have d_qk == d_head there.  These use a full-width
# (d_qk == D_HEAD) offset graph.


def _onnx_offset_graph():
    pos = create_pos_encoding()
    payload = create_input("payload", 1)
    rotary = rotary_offset_head(payload, delta_pos=-1, d_qk=D_HEAD)
    return rotary, pos


def test_rotary_offset_onnx_prefill_and_decode():
    """The ONNX export rotates Q and stores rotated K, so the offset head
    selects the previous position on both prefill and cached decode (the
    runtime mirror of the in-process invariant)."""
    pytest.importorskip("onnxruntime")
    from torchwright.compiler.export import compile_headless_to_onnx
    from torchwright.compiler.onnx_load import OnnxHeadlessModule

    rotary, pos = _onnx_offset_graph()
    vals = _payload(N_POS)
    expected = _expected_prev(vals)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rope_offset.onnx")
        compile_headless_to_onnx(
            rotary, pos, path, d=D, d_head=D_HEAD, max_seq_len=64, verbose=False
        )
        module = OnnxHeadlessModule(path)

        full = module(vals).squeeze(1)
        assert torch.allclose(full, expected, atol=1e-2), full

        past = module.empty_past()
        decoded = []
        for t in range(N_POS):
            out_t, past = module.step(vals[t : t + 1], past)
            decoded.append(float(out_t.squeeze()))
        assert torch.allclose(torch.tensor(decoded), expected, atol=1e-2), decoded


def test_rotary_offset_onnx_partial_width_rejected():
    """ONNX RoPE requires full-width rotary heads (rotate_half over d_head); a
    partial-width rotary head fails loud rather than rotating the wrong dims."""
    pytest.importorskip("onnxruntime")
    from torchwright.compiler.export import compile_headless_to_onnx

    pos = create_pos_encoding()
    payload = create_input("payload", 1)
    rotary = rotary_offset_head(payload, delta_pos=-1, d_qk=8)  # < D_HEAD

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "partial.onnx")
        with pytest.raises(NotImplementedError, match="rotary_width == d_head"):
            compile_headless_to_onnx(
                rotary, pos, path, d=D, d_head=D_HEAD, max_seq_len=64, verbose=False
            )
