"""Partial rotary (``d_rot``) on the HF token path.

Same "predict the previous token" rotary offset model as ``test_rope_token.py``,
but built with a partial rotary width (``d_rot < d_head``).  Exports to ONNX,
converts to ``TorchwrightForCausalLM``, and checks the HF runtime:

* carries ``d_rot`` (and ``rope_base``) through the converter,
* still predicts the previous token (the partial rotation survives export +
  convert — the NoPE tail is a position-independent constant the softmax ignores),
* prefill == token-by-token cached decode (cache stores rotated K),
* a **content head** (``attend_argmax_dot``) whose content rides the NoPE tail
  survives export + convert and still selects by content (predicts the
  highest-vocab-index token seen so far) — the partial-rotary content placement
  this phase added, exercised on the real ONNX + HF surfaces.

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
from torchwright.graph.rope import ROPE_BASE, RopeConfig, rotary_offset_head
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


def _build_content_model(tmpdir):
    """Predict the highest-vocab-index token seen so far via a content head whose
    content (a width-1 score) rides the NoPE tail under partial rotary."""
    from torchwright.graph.linear import Linear
    from torchwright.ops.attention_ops import attend_argmax_dot
    from torchwright.ops.inout_nodes import create_literal_value

    emb = create_onehot_embedding(_VOCAB)
    # Per-position scalar score = the token's vocab index (one-hot · arange).
    idx = torch.arange(len(_VOCAB), dtype=torch.float32).reshape(len(_VOCAB), 1)
    score = Linear(emb, idx, name="vocab_index_score")
    rope = RopeConfig(d_head=D_HEAD, max_positions=64, d_rot=D_ROT)
    # A constant-1 query dotted with the score key ranks causal keys by score, so
    # the dot-content head selects the max-score (highest vocab index) token so far.
    query_one = create_literal_value(torch.tensor([1.0]), name="content_query_one")
    out = attend_argmax_dot(rope, query_one, score, emb)
    path = os.path.join(tmpdir, "max_index_token.onnx")
    compile_to_onnx(out, emb, path, d=D, d_head=D_HEAD, max_seq_len=64, verbose=False)
    return convert_onnx_to_hf(path, bos_token="<bos>", eos_token="<eos>")


def test_hf_partial_content_head_selects_by_content():
    """A content head on the NoPE tail survives ONNX export + HF convert and still
    selects by content: each position predicts the highest-vocab-index token in its
    causal window."""
    seq = ["<bos>", "a", "c", "b", "e", "d"]  # indices 0,2,4,3,6,5
    ids = torch.tensor([[_VOCAB.index(t) for t in seq]])
    with tempfile.TemporaryDirectory() as tmp:
        hf = _build_content_model(tmp)
        assert hf.config.d_rot == D_ROT
        with torch.no_grad():
            logits = hf(ids).logits[0]
    pred = [_VOCAB[i] for i in logits.argmax(-1).tolist()]
    # running-max vocab index → token: <bos>,a,c,c,e,e
    assert pred == ["<bos>", "a", "c", "c", "e", "e"], pred


def test_hf_partial_content_head_prefill_equals_cached_decode():
    """The partial-rotary content head caches cleanly: prefill == cached decode."""
    seq = ["<bos>", "a", "c", "b", "e", "d"]
    ids = torch.tensor([[_VOCAB.index(t) for t in seq]])
    with tempfile.TemporaryDirectory() as tmp:
        hf = _build_content_model(tmp)
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
