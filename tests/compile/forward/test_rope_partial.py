"""Vanilla partial rotary (``d_rot``): rotate the first ``d_rot`` dims of each
head, pass the last ``d_head - d_rot`` through unrotated (the NoPE tail).

This is HF ``partial_rotary_factor`` (GPT-NeoX / Phi / StableLM): the same
``rotate_half`` as the LLaMA3 end state, restricted to the rotary front and
normalized by ``d_rot``.  Coverage:

* the primitive ``apply_rope`` leaves the NoPE tail exactly position-invariant
  and reduces to the full-rotary expression at ``d_rot == d_head``;
* ``RopeConfig`` validates ``d_rot``;
* the content heads route onto the NoPE tail under partial ``d_rot`` — they build,
  select by content, and the selection is EXACT across distance (no slow-plane
  attenuation), oracle == compiled (``probe_compiled``);
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
from torchwright.ops.attention_ops import attend_argmax_dot
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


def _pack(module, named, n):
    """Pack named per-input tensors into the compiled module's flat input row."""
    total = sum(w for _, _, w in module._input_specs)
    out = torch.zeros(n, total)
    for name, start, w in module._input_specs:
        if name in named:
            out[:, start : start + w] = named[name]
    return out


def test_apply_rope_tail_is_position_invariant():
    """The NoPE tail ``x[..., d_rot:]`` is byte-identical across positions; only
    the rotary front changes.
    """
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
    pre-partial expression, byte-for-byte.
    """
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


def _content_dot_graph():
    """A content (dot-match) head under partial rotary. ``attend_argmax_dot``
    picks the causal key with the largest ``query·key`` — a pure content head,
    so under ``d_rot < d_head`` its content must ride the NoPE tail.
    """
    rope = create_rope_config(d_head=D_HEAD, max_positions=512, d_rot=D_ROT)
    query = create_input("query", 1)
    key = create_input("key", 1)
    value = create_input("value", 1)
    return attend_argmax_dot(rope, query, key, value)


def test_content_head_builds_on_nope_tail_partial_rotary():
    """A content head *builds* under partial rotary (no longer rejected): the
    content rides the NoPE tail, so the Attn carries ``rope_d_rot == D_ROT`` and the
    rotary front of its Q/K projection is all zero (the logit is pure content
    dot).
    """
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.graph.attn import Attn

    sel = _content_dot_graph()
    attns = [n for n in get_ancestor_nodes({sel}) if isinstance(n, Attn)]
    assert attns, "expected a content Attn head"
    head = attns[0]
    assert head.rope_d_rot == D_ROT
    assert head.query_matrix.shape[1] == D_HEAD  # head fills the grid (d_qk==d_head)
    # The content lives in the unrotated tail; the rotary front is zero.
    assert head.query_matrix[:, :D_ROT].abs().sum() == 0.0
    assert head.key_matrix[:, :D_ROT].abs().sum() == 0.0
    assert head.query_matrix[:, D_ROT:].abs().sum() > 0.0


def test_partial_content_compiled_matches_oracle_and_selects():
    """Compiled partial-rotary content-dot head == oracle, and selects the
    highest-dot valid key EXACTLY across distance (tail content has no slow-plane
    attenuation): a unique max planted at position 1 is selected by every later
    query, near and far alike.
    """
    sel = _content_dot_graph()
    n = 24
    q = torch.ones(n, 1)  # constant query
    k = torch.ones(n, 1)
    k[1, 0] = 50.0  # unique max-dot key at a near position
    val = (torch.arange(n, dtype=torch.float32) * 10.0).unsqueeze(1)
    named = {"query": q, "key": k, "value": val}

    compiled = compile_headless(sel, d=D, d_head=D_HEAD, verbose=False)
    report = probe_compiled(compiled, sel, named, n, atol=1e-2)
    assert report.first_divergent is None, report.format_short()

    out = compiled(_pack(compiled, named, n).to(compiled._net.device)).reshape(-1).cpu()
    # Every query qq >= 1 selects position 1 (value 10), Δ from 0 (qq=1) to 22.
    for qq in range(1, n):
        assert abs(out[qq].item() - 10.0) < 1e-2, f"pos {qq}: got {out[qq].item():.4f}"


def test_compile_rejects_mixed_d_rot():
    """d_rot is global: a graph mixing two rotary widths fails fast at
    forward_compile (one shared cos/sin grid can't honor both).
    """
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
