"""Regression: the rotary HF model survives a save_pretrained -> from_pretrained
round trip.

The rotary frequency grid and per-head enable mask are derived purely from the
serialized config.  An earlier port registered them as ``persistent=False``
buffers computed in ``__init__``; ``from_pretrained``'s meta-device
(``low_cpu_mem_usage``) load path does NOT materialize a non-persistent buffer —
it is absent from the checkpoint and never re-run — so the reloaded buffer held
uninitialized memory (NaN on the A100 fp32 path) and the rotary multiply
propagated NaN to every logit.  The freshly-direct_model model passed because
``compile_to_hf`` builds it with a plain constructor (no meta path), which
is exactly why the bug only showed up on save -> load.

This pins the contract directly at the smallest layer — a tiny config built by
hand, no compile / ONNX artifact — so it is fast and independent of the
direct compiler.  It asserts behaviour (finite, bit-identical logits before and after
the round trip), not a specific buffer, so it stays valid however the rotary
constants are stored.
"""

import pytest
import torch

from torchwright.compiler.hf.configuration_torchwright_custom import TorchwrightCustomConfig
from torchwright.compiler.hf.modeling_torchwright_custom import TorchwrightCustomForCausalLM


def _tiny_model(rope_base):
    torch.manual_seed(0)
    cfg = TorchwrightCustomConfig(
        d=48,
        d_head=12,
        vocab_size=13,
        n_layers=3,
        n_heads_per_layer=[3, 3, 3],
        d_hidden_per_layer=[24, 24, 24],
        max_position_embeddings=64,
        rope_base=rope_base,
    )
    model = TorchwrightCustomForCausalLM(cfg).eval()
    # A finite, non-zero embedding table so a broken rotary path can't hide
    # behind an all-zero residual (token.v5: const seeds ride the embedding
    # rows; there is no separate constant_values buffer to perturb).
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(torch.randn(cfg.vocab_size, cfg.d) * 0.1)
    return model


# Every head is full-width rotary on the one global grid; a couple of bases
# exercise the recomputed frequency grid.
@pytest.mark.parametrize("rope_base", [500000.0, 10000.0])
def test_save_load_logits_identical(tmp_path, rope_base):
    model = _tiny_model(rope_base)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        before = model(input_ids=ids, use_cache=True).logits
    assert torch.isfinite(before).all(), "fresh model already non-finite"

    model.save_pretrained(tmp_path)
    reloaded = TorchwrightCustomForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(input_ids=ids, use_cache=True).logits

    assert torch.isfinite(after).all(), "reloaded model emits non-finite logits"
    assert torch.equal(before, after), (
        f"save->load changed the logits (max abs diff "
        f"{(before - after).abs().max().item():.3e})"
    )


def test_save_load_cached_decode_finite(tmp_path):
    """The reloaded model also decodes (cached single step after prefill) finite —
    the path generate() exercises."""
    model = _tiny_model(500000.0)
    model.save_pretrained(tmp_path)
    reloaded = TorchwrightCustomForCausalLM.from_pretrained(tmp_path).eval()

    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        out = reloaded(input_ids=ids, use_cache=True)
        step = reloaded(
            input_ids=torch.tensor([[7]]),
            past_key_values=out.past_key_values,
            use_cache=True,
        ).logits
    assert torch.isfinite(step).all(), "reloaded cached decode emits non-finite logits"
