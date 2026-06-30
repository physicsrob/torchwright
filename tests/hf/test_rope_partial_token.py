"""Partial rotary (``d_rot``) on the HF token path.

Same "predict the previous token" rotary offset model as ``test_rope_token.py``,
but built with a partial rotary width (``d_rot < d_head``).  Exports to ONNX,
converts to ``TorchwrightForCausalLM``, and checks the HF runtime:

* carries ``d_rot`` (and ``rope_base``) through the converter,
* still predicts the previous token (the partial rotation survives export +
  convert — the NoPE tail is a position-independent constant the softmax ignores),
* prefill == token-by-token cached decode (cache stores rotated K).

CPU-only; skips where onnxruntime / transformers are unavailable.
"""

from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")

from transformers.cache_utils import DynamicCache

from torchwright.compiler.export import compile_to_onnx
from torchwright.compiler.hf.convert import convert_onnx_to_hf
from torchwright.graph.rope import ROPE_BASE, rotary_offset_head
from torchwright.ops.inout_nodes import create_onehot_embedding

_VOCAB = ["<bos>", "<eos>", "a", "b", "c", "d", "e"]
D = 256
D_HEAD = 16
D_ROT = 8  # partial: the first 8 dims rotate, the last 8 are the NoPE tail


def _build_and_convert(tmpdir):
    emb = create_onehot_embedding(_VOCAB)
    prev = rotary_offset_head(emb, delta_pos=-1, d_qk=D_HEAD, d_rot=D_ROT)
    path = os.path.join(tmpdir, "prev_token.onnx")
    compile_to_onnx(prev, emb, path, d=D, d_head=D_HEAD, max_seq_len=64, verbose=False)
    return convert_onnx_to_hf(path, bos_token="<bos>", eos_token="<eos>")


def test_torchwright_config_validates_d_rot():
    from torchwright.compiler.hf.configuration_torchwright import TorchwrightConfig

    # All-zero defaults (HF instantiates configs this way) must not raise.
    TorchwrightConfig()
    # A real model rejects an odd or out-of-range d_rot.
    with pytest.raises(ValueError, match="d_rot"):
        TorchwrightConfig(d=256, d_head=16, d_rot=3)
    with pytest.raises(ValueError, match="d_rot"):
        TorchwrightConfig(d=256, d_head=16, d_rot=18)
    # Valid partial width is accepted and stored.
    assert TorchwrightConfig(d=256, d_head=16, d_rot=8).d_rot == 8


def test_hf_partial_rope_config_carried():
    with tempfile.TemporaryDirectory() as tmp:
        hf = _build_and_convert(tmp)
    assert hf.config.rope_base == ROPE_BASE
    assert hf.config.d_rot == D_ROT
    assert hf.config.d_head == D_HEAD


def test_hf_partial_rope_predicts_previous_token():
    seq = ["<bos>", "a", "b", "c", "d", "e"]
    ids = torch.tensor([[_VOCAB.index(t) for t in seq]])
    with tempfile.TemporaryDirectory() as tmp:
        hf = _build_and_convert(tmp)
        with torch.no_grad():
            logits = hf(ids).logits[0]
    pred = [_VOCAB[i] for i in logits.argmax(-1).tolist()]
    assert pred[1:] == seq[:-1], pred


def test_hf_partial_rope_prefill_equals_cached_decode():
    seq = ["<bos>", "a", "b", "c", "d", "e"]
    ids = torch.tensor([[_VOCAB.index(t) for t in seq]])
    with tempfile.TemporaryDirectory() as tmp:
        hf = _build_and_convert(tmp)
        with torch.no_grad():
            full = hf(ids).logits[0]
            cache = DynamicCache()
            rows = []
            for t in range(ids.shape[1]):
                out = hf(
                    ids[:, t : t + 1],
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.arange(t, t + 1),
                )
                cache = out.past_key_values
                rows.append(out.logits[0, -1])
            decode = torch.stack(rows)
    assert torch.allclose(full, decode, atol=1e-4), (full - decode).abs().max()
