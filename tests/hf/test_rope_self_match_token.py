"""Phase-2 part 2 on the ONNX + HF token path: rotary compiler self-match.

The Δ=0 self-match heads behind ``Linear``/``Add``/``Cancel``/``add_into`` are
compiler-internal — they never appear as a graph ``Attn`` node — so the only way
to exercise them through the real ONNX emission and the native HF runtime is a
token model whose lowering emits them.  The "predict the previous token" model
of ``test_rope_token.py`` does: with ``rotary_self_match=True`` it compiles to
three rotary heads (the one offset head plus two self-match transport heads),
versus one with the flag off.

This pins that the migration survives export + convert on both non-in-process
surfaces:

* the reserved constant-1 column rides the ONNX/HF ``constant_values`` seed
  (no new emission code) and reaches the HF model as a literal 1.0 column,
* the self-match rotation propagates via per-head ``rotary_width`` → the HF
  per-head enable mask, so transport stays correct and the model still predicts
  the previous token,
* prefill == token-by-token cached decode (the self-match K is stored rotated
  at absolute position, same as the offset head).

If the const-1 column or the rotary emission were wrong, the self-match would
corrupt arithmetic transport and both the prediction and the cache parity would
break — these are the load-bearing cross-surface checks.

CPU-only; skips where onnxruntime / transformers are unavailable, like the other
HF parity tests.
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
from torchwright.ops.inout_nodes import create_onehot_embedding, create_pos_encoding

_VOCAB = ["<bos>", "<eos>", "a", "b", "c", "d", "e"]
D = 256
D_HEAD = 16


def _build_and_convert(tmpdir):
    emb = create_onehot_embedding(_VOCAB)
    prev = rotary_offset_head(emb, delta_pos=-1, d_qk=D_HEAD)  # full-width rotary
    pos = create_pos_encoding()
    path = os.path.join(tmpdir, "prev_token_self_match.onnx")
    compile_to_onnx(
        prev,
        pos,
        emb,
        path,
        d=D,
        d_head=D_HEAD,
        max_seq_len=64,
        verbose=False,
        rotary_self_match=True,
    )
    return convert_onnx_to_hf(path, bos_token="<bos>", eos_token="<eos>")


def test_hf_self_match_rotary_heads_and_const_column():
    """The self-match heads are rotary in the HF config, and the reserved
    constant-1 column reached the HF model as a literal 1.0 column."""
    with tempfile.TemporaryDirectory() as tmp:
        hf = _build_and_convert(tmp)
    assert hf.config.rope_base == ROPE_BASE
    # With the flag off this model has exactly one rotary head (the offset
    # head); turning self-match rotary on adds the transport heads, so the
    # count is strictly greater.  The exact number depends on how many
    # BIRTH-layer cancels the schedule emits — compile_to_onnx defaults to
    # assume_zero_init=True, which elides some — so assert "> 1", not a literal.
    total_rotary = sum(sum(row) for row in hf.config.rotary_enable_per_layer)
    assert total_rotary > 1, hf.config.rotary_enable_per_layer
    # The const-1 column rode the constant_values seed: at least one residual
    # column is a literal 1.0.
    assert bool(
        (hf.model.constant_values == 1.0).any()
    ), "no constant-1 column in the HF constant_values seed"


def test_hf_self_match_predicts_previous_token():
    """Rotary self-match transports the embedding correctly through export +
    convert: the model still predicts the previous token."""
    seq = ["<bos>", "a", "b", "c", "d", "e"]
    ids = torch.tensor([[_VOCAB.index(t) for t in seq]])
    with tempfile.TemporaryDirectory() as tmp:
        hf = _build_and_convert(tmp)
        with torch.no_grad():
            logits = hf(ids).logits[0]
    pred = [_VOCAB[i] for i in logits.argmax(-1).tolist()]
    assert pred[1:] == seq[:-1], pred


def test_hf_self_match_prefill_equals_cached_decode():
    """Prefill == token-by-token cached decode with rotary self-match heads."""
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
