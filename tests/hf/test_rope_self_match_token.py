"""Phase-2 part 2 on the ONNX + HF token path: rotary compiler self-match.

The Δ=0 self-match heads behind ``Linear``/``Add``/``Cancel``/``add_into`` are
compiler-internal — they never appear as a graph ``Attn`` node — so the only way
to exercise them through the real ONNX emission and the stock Phi-3 runtime is a
token model whose lowering emits them.  The "predict the previous token" model
of ``test_rope_token.py`` does: under RoPE the self-match transport heads are
*unconditionally* rotary (the old ``rotary_self_match`` flag is gone), so this
model compiles to more than one rotary head — the offset head plus the self-match
transport head(s) the schedule emits.

This pins that the migration survives direct compilation on both non-in-process
surfaces:

* the reserved constant-1 column rides the embedding table's rows (the
  token.v6 fold) and reaches the HF model as a literal 1.0 column in every
  vocab row,
* the self-match heads are full-width rotary like every other head (one global
  grid, no per-head enable), so transport stays correct and the model still
  predicts the previous token,
* prefill == token-by-token cached decode (the self-match K is stored rotated
  at absolute position, same as the offset head).

If the const-1 column or the rotary emission were wrong, the self-match would
corrupt arithmetic transport and both the prediction and the cache parity would
break — these are the load-bearing cross-surface checks.

CPU-only; skips where onnxruntime / transformers are unavailable, like the other
HF parity tests.
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
    return compile_to_hf(
        prev,
        emb,
        d=D,
        d_head=D_HEAD,
        max_seq_len=64,
        verbose=False,
    )


def test_hf_self_match_rotary_heads_and_const_column():
    """The self-match heads are rotary in the HF config, and the reserved
    constant-1 column reached the HF model as a literal 1.0 column."""
    hf = _build_direct()
    # Every head (offset + the unconditionally-rotary self-match transport heads)
    # is full-width rotary on the one global grid; the single shared base carried
    # through the direct compiler (there is no per-head enable).
    assert hf.config.rope_parameters["rope_theta"] == ROPE_BASE
    # The const-1 column is folded into the embedding table (token.v6): at
    # least one residual column is a literal 1.0 in EVERY vocab row, so the
    # per-token lookup reproduces it at every position.
    emb = hf.model.embed_tokens.weight
    assert bool((emb == 1.0).all(dim=0).any()), (
        "no all-rows constant-1 column in the HF embedding table"
    )


def test_hf_self_match_predicts_previous_token():
    """Rotary self-match transports the embedding correctly through export +
    compilation: the model still predicts the previous token."""
    seq = ["<bos>", "a", "b", "c", "d", "e"]
    ids = torch.tensor([[_VOCAB.index(t) for t in seq]])
    hf = _build_direct()
    with torch.no_grad():
        logits = hf(ids).logits[0]
    pred = [_VOCAB[i] for i in logits.argmax(-1).tolist()]
    assert pred[1:] == seq[:-1], pred


def test_hf_self_match_prefill_equals_cached_decode():
    """Prefill == token-by-token cached decode with rotary self-match heads."""
    seq = ["<bos>", "a", "b", "c", "d", "e"]
    ids = torch.tensor([[_VOCAB.index(t) for t in seq]])
    hf = _build_direct()
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
