"""Vanilla partial rotary (``d_rot``): rotate the first ``d_rot`` dims of each
head, pass the last ``d_head - d_rot`` through unrotated (the NoPE tail).

This is HF ``partial_rotary_factor`` (GPT-NeoX / Phi / StableLM): the same
``rotate_half`` as the LLaMA3 end state, restricted to the rotary front and
normalized by ``d_rot``.  Coverage:

* the primitive ``apply_rope`` leaves the NoPE tail exactly position-invariant
  and reduces to the full-rotary expression at ``d_rot == d_head``;
* ``RopeConfig`` validates ``d_rot``;
* the plane-based content heads reject a partial ``d_rot`` (full-rotary only);
* the rotary offset head still selects the previous position at ``d_rot < d_head``
  — oracle == compiled (``probe_compiled``) and prefill == cached decode.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph.rope import (
    RopeConfig,
    apply_rope,
    rope_cos_sin,
    rotary_offset_head,
    rotate_half,
)
from torchwright.ops.attention_ops import attend_argmin_where
from torchwright.ops.inout_nodes import create_input, create_rope_config

N_POS = 12
D = 256
D_HEAD = 16
D_ROT = 8  # >= D_HEAD/2 so the offset head keeps a clean argmax at hardness=100
BASE = 500000.0


def _payload(n_pos: int) -> torch.Tensor:
    return (100.0 + torch.arange(n_pos, dtype=torch.float32)).unsqueeze(1)


def _expected_prev(payload: torch.Tensor) -> torch.Tensor:
    shifted = payload.squeeze(1).roll(1)
    shifted[0] = payload[0, 0]
    return shifted


def test_apply_rope_tail_is_position_invariant():
    """The NoPE tail ``x[..., d_rot:]`` is byte-identical across positions; only
    the rotary front changes."""
    x = torch.randn(1, D_HEAD)
    cos0, sin0 = rope_cos_sin(torch.tensor([0]), D_ROT, BASE)
    cos7, sin7 = rope_cos_sin(torch.tensor([7]), D_ROT, BASE)
    y0 = apply_rope(x, cos0, sin0)
    y7 = apply_rope(x, cos7, sin7)

    # Tail unchanged from the input, and identical across two different positions.
    assert torch.equal(y0[..., D_ROT:], x[..., D_ROT:])
    assert torch.equal(y7[..., D_ROT:], x[..., D_ROT:])
    # Front actually rotates (so the test isn't vacuous).
    assert not torch.allclose(y0[..., :D_ROT], y7[..., :D_ROT])


def test_apply_rope_full_width_reduces_to_exact_expression():
    """At ``d_rot == d_head`` (cos as wide as x) the partial path takes the exact
    pre-partial expression, byte-for-byte."""
    x = torch.randn(3, D_HEAD)
    cos, sin = rope_cos_sin(torch.tensor([0, 5, 11]), D_HEAD, BASE)
    exact = x * cos + rotate_half(x) * sin
    assert torch.equal(apply_rope(x, cos, sin), exact)


@pytest.mark.parametrize("bad", [3, 0, D_HEAD + 2])
def test_rope_config_rejects_bad_d_rot(bad):
    with pytest.raises(ValueError, match="d_rot"):
        RopeConfig(d_head=D_HEAD, max_positions=512, d_rot=bad)


def test_rope_config_default_d_rot_is_full():
    assert RopeConfig(d_head=D_HEAD, max_positions=512).d_rot == D_HEAD
    assert RopeConfig(d_head=D_HEAD, max_positions=512, d_rot=D_ROT).d_rot == D_ROT


def test_content_head_rejects_partial_rotary():
    """The plane-based content heads ride specific planes of the full grid, so a
    partial ``d_rot`` config is rejected loudly (full-rotary only)."""
    rope = create_rope_config(d_head=D_HEAD, max_positions=512, d_rot=D_ROT)
    score = create_input("score", 1)
    validity = create_input("validity", 1)
    value = create_input("value", 1)
    with pytest.raises(NotImplementedError, match="full rotary"):
        attend_argmin_where(rope, score, validity, value)


def test_compile_rejects_mixed_d_rot():
    """d_rot is global: a graph mixing two rotary widths fails fast at
    forward_compile (one shared cos/sin grid can't honor both)."""
    from torchwright.graph import Concatenate

    payload = create_input("payload", 1)
    h_full = rotary_offset_head(payload, delta_pos=-1, d_qk=D_HEAD)  # d_rot=d_head
    h_part = rotary_offset_head(payload, delta_pos=-1, d_qk=D_HEAD, d_rot=D_ROT)
    out = Concatenate([h_full, h_part])
    with pytest.raises(ValueError, match="one global value"):
        compile_headless(out, d=D, d_head=D_HEAD, verbose=False)


def test_partial_offset_oracle_selects_prev_position():
    payload = create_input("payload", 1)
    vals = _payload(N_POS)
    rotary = rotary_offset_head(payload, delta_pos=-1, d_qk=D_HEAD, d_rot=D_ROT)
    assert rotary.rope_d_rot == D_ROT
    out = rotary.compute(N_POS, {"payload": vals}).squeeze(1)
    assert torch.allclose(out, _expected_prev(vals), atol=1e-3), out


def test_partial_offset_compiled_matches_oracle():
    payload = create_input("payload", 1)
    vals = _payload(N_POS)
    rotary = rotary_offset_head(payload, delta_pos=-1, d_qk=D_HEAD, d_rot=D_ROT)

    compiled = compile_headless(rotary, d=D, d_head=D_HEAD, verbose=False)
    report = probe_compiled(compiled, rotary, {"payload": vals}, N_POS, atol=1e-2)
    assert report.first_divergent is None, report.format_short()

    out = compiled(vals.to(compiled._net.device)).squeeze(1).cpu()
    assert torch.allclose(out, _expected_prev(vals), atol=1e-2), out


def test_partial_offset_prefill_decode_identical():
    payload = create_input("payload", 1)
    vals = _payload(N_POS)
    rotary = rotary_offset_head(payload, delta_pos=-1, d_qk=D_HEAD, d_rot=D_ROT)

    compiled = compile_headless(rotary, d=D, d_head=D_HEAD, verbose=False)
    device = compiled._net.device
    full = compiled(vals.to(device)).squeeze(1).cpu()

    past = compiled.empty_past()
    decoded = []
    for t in range(N_POS):
        out_t, past = compiled.step(vals[t : t + 1].to(device), past)
        decoded.append(out_t.squeeze(0).squeeze(0).item())
    decoded = torch.tensor(decoded)

    assert torch.allclose(full, decoded, atol=1e-4), (full, decoded)
