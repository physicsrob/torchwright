"""Continuous-input and raw-residual behavior of the custom HF model."""

import pytest
import torch

from torchwright.compiler.hf import (
    TorchwrightCustomConfig,
    TorchwrightCustomForCausalLM,
)


def _model() -> TorchwrightCustomForCausalLM:
    config = TorchwrightCustomConfig(
        d=8,
        d_head=4,
        vocab_size=3,
        n_layers=1,
        n_heads_per_layer=[1],
        d_hidden_per_layer=[8],
        max_position_embeddings=4,
        rms_norm=True,
    )
    return TorchwrightCustomForCausalLM(config).float().eval()


def test_inputs_embeds_matches_embedding_lookup() -> None:
    """The continuous hidden-state path follows the HF inputs_embeds contract."""
    model = _model()
    input_ids = torch.tensor([[0, 1, 2]])
    inputs_embeds = model.model.embed_tokens(input_ids)

    with torch.no_grad():
        from_ids = model(input_ids=input_ids).logits
        from_embeds = model(inputs_embeds=inputs_embeds).logits

    torch.testing.assert_close(from_embeds, from_ids)


def test_forward_residual_bypasses_final_norm_and_lm_head() -> None:
    """Raw output is exactly the decoder-block result before token readout."""
    model = _model()
    inputs_embeds = torch.randn(2, 3, model.config.d)

    with torch.no_grad():
        raw = model.forward_residual(inputs_embeds=inputs_embeds)
        normalized = model.model(inputs_embeds=inputs_embeds).last_hidden_state

    assert raw.shape == inputs_embeds.shape
    torch.testing.assert_close(normalized, model.model.norm(raw))
    assert not torch.equal(raw, normalized)


def test_inputs_embeds_validates_exclusive_source_and_shape() -> None:
    model = _model()
    ids = torch.tensor([[0]])

    with pytest.raises(ValueError, match="exactly one"):
        model.model()
    with pytest.raises(ValueError, match="not both"):
        model.model(input_ids=ids, inputs_embeds=torch.zeros(1, 1, 8))
    with pytest.raises(ValueError, match="shape"):
        model.model(inputs_embeds=torch.zeros(1, 1, 7))
