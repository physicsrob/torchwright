"""Phase-0 RoPE on the HF token path: a rotary "predict the previous token" model.

Builds a one-hot token model whose single rotary offset head transports each
token's embedding one position forward, so the tied unembed predicts the
*previous* token. It compiles directly to stock Phi-3 and checks the HF runtime:

* carries the RoPE config (``rope_base`` + per-head enable) through the direct compiler,
* predicts the previous token (the rotary selection survives direct compilation),
* prefill == token-by-token cached decode (cache stores rotated K).

CPU-only; skips where onnxruntime / transformers are unavailable (e.g. the
Modal test image), like the other HF parity tests.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.cache_utils import DynamicCache

from torchwright.compiler.hf import compile_to_hf
from torchwright.graph.rope import ROPE_BASE, rotary_offset_head
from torchwright.ops.inout_nodes import create_onehot_embedding

_VOCAB = ["<bos>", "<eos>", "a", "b", "c", "d", "e"]
D = 256
D_HEAD = 16


def _build_direct(_tmpdir=None):
    emb = create_onehot_embedding(_VOCAB)
    prev = rotary_offset_head(emb, delta_pos=-1, d_qk=D_HEAD)  # full-width rotary
    return compile_to_hf(prev, emb, d=D, d_head=D_HEAD, max_seq_len=64)


def test_hf_rope_config_carried():
    hf = _build_direct()
    # Every head is full-width rotary on the one global grid (no per-head enable
    # to check); the single shared base carried through the direct compiler.  The
    # predict-previous + prefill==cached-decode assertions below prove the
    # rotation actually fires correctly.
    assert hf.config.rope_parameters["rope_theta"] == ROPE_BASE


def test_hf_rope_predicts_previous_token():
    seq = ["<bos>", "a", "b", "c", "d", "e"]
    ids = torch.tensor([[_VOCAB.index(t) for t in seq]])
    hf = _build_direct()
    with torch.no_grad():
        logits = hf(ids).logits[0]
    pred = [_VOCAB[i] for i in logits.argmax(-1).tolist()]
    assert pred[1:] == seq[:-1], pred


def test_hf_rope_prefill_equals_cached_decode():
    seq = ["<bos>", "a", "b", "c", "d", "e"]
    ids = torch.tensor([[_VOCAB.index(t) for t in seq]])
    hf = _build_direct()
    with torch.no_grad():
        full = hf(ids).logits[0]
        cache = DynamicCache()
        rows = []
        for t in range(ids.shape[1]):
            out = hf(
                ids[:, t : t + 1], past_key_values=cache, use_cache=True,
                cache_position=torch.arange(t, t + 1),
            )
            cache = out.past_key_values
            rows.append(out.logits[0, -1])
        decode = torch.stack(rows)
    assert torch.allclose(full, decode, atol=1e-4), (full - decode).abs().max()
