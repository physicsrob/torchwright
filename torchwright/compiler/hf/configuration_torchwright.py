"""Torchwright model configuration."""

from __future__ import annotations

from transformers import PretrainedConfig


class TorchwrightConfig(PretrainedConfig):
    """Configuration for a Torchwright causal decoder.

    The model is a pre-norm decoder with a uniform residual width ``d``: a
    ``(vocab_size, d)`` token embedding feeds ``n_layers`` attention+MLP
    blocks, and the same ``(vocab_size, d)`` table is tied to ``lm_head`` for
    readout. Attention is causal with ``scale=1.0`` and rotary position
    embeddings; the MLP is ``fc2(relu(fc1(x)))``. Head count and MLP hidden
    width may vary per layer. When ``rms_norm`` is set, each sublayer input
    and the final hidden state pass through a Llama-style RMSNorm. Weights
    and activations are fp32; the modeling code enforces this.

    Args:
        d: Residual stream width (also the embedding/unembedding width).
        d_head: Per-head dimension, shared across all layers and heads.
        vocab_size: Number of rows in the embedding table.
        n_layers: Number of attention+MLP blocks.
        n_heads_per_layer: List of length ``n_layers`` — attention head count
            for each layer.
        d_hidden_per_layer: List of length ``n_layers`` — MLP hidden width
            for each layer.
        max_position_embeddings: Largest absolute position the model is
            built for.
        rope_base: Rotary frequency base.
        d_rot: Rotary width. The first ``d_rot`` dims of every head rotate;
            the remaining ``d_head - d_rot`` pass through unrotated. ``None``
            means full rotary (``d_rot == d_head``).
        rms_norm: Whether the model applies RMSNorm — pre-norm on each
            sublayer input plus a final norm.
        rms_norm_eps: RMSNorm epsilon.
    """

    model_type = "torchwright"

    def __init__(
        self,
        d: int = 0,
        d_head: int = 0,
        vocab_size: int = 0,
        n_layers: int = 0,
        n_heads_per_layer: list[int] | None = None,
        d_hidden_per_layer: list[int] | None = None,
        max_position_embeddings: int = 0,
        rope_base: float = 500000.0,
        d_rot: int | None = None,
        rms_norm: bool = False,
        rms_norm_eps: float = 1e-5,
        **kwargs,
    ):
        self.d = int(d)
        self.d_head = int(d_head)
        self.vocab_size = int(vocab_size)
        self.n_layers = int(n_layers)
        self.n_heads_per_layer = [int(n) for n in (n_heads_per_layer or [])]
        self.d_hidden_per_layer = [int(n) for n in (d_hidden_per_layer or [])]
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_base = float(rope_base)
        self.d_rot = int(d_rot) if d_rot is not None else self.d_head
        # Validate only for a real model (d_head > 0): transformers
        # instantiates configs with all-zero defaults, which must not raise.
        if self.d_head and (self.d_rot % 2 != 0 or not (0 < self.d_rot <= self.d_head)):
            raise ValueError(
                f"TorchwrightConfig.d_rot must be even and in (0, "
                f"d_head={self.d_head}]; got {self.d_rot}."
            )
        self.rms_norm = bool(rms_norm)
        self.rms_norm_eps = float(rms_norm_eps)
        # Canonical aliases consulted by generic transformers utilities (cache
        # sizing, repr, sharding). Recomputed from our own fields; drop any
        # stale values round-tripped in from a serialized config.
        kwargs.pop("num_hidden_layers", None)
        kwargs.pop("hidden_size", None)
        self.num_hidden_layers = self.n_layers
        self.hidden_size = self.d
        # Torchwright models are storage-tied (token.v6): lm_head shares
        # embed_tokens' storage, so an untied config has no meaning here.
        # Overriding an explicit False silently would let HF's tie_weights()
        # clone the embedding over a checkpoint's real lm_head (pre-v6
        # converters saved tie_word_embeddings: false with a distinct
        # unembed) — wrong logits with no error.  Reject it loudly instead.
        if kwargs.get("tie_word_embeddings", True) is False:
            raise ValueError(
                "TorchwrightConfig requires tie_word_embeddings=True: the "
                "unembed is storage-tied to the embedding (token.v6).  A "
                "config with tie_word_embeddings=False is a pre-v6 artifact "
                "— regenerate it with the current converter."
            )
        kwargs["tie_word_embeddings"] = True
        super().__init__(**kwargs)
