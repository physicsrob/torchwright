"""PyTorch Torchwright model.

A pre-norm causal decoder with rotary position embeddings, per-layer head
and MLP widths, and a tied LM head.

The forward path:

    h = embed_tokens(input_ids)                    # (B, T, d)
    for each layer:
        h = h + attn(norm(h))                      # causal, scale=1.0, RoPE, no bias
        h = h + mlp(norm(h))                       # fc2(relu(fc1(h))), biased
    logits = lm_head(norm(h))                      # tied to embed_tokens, no bias

``norm`` is a Llama-style RMSNorm when ``config.rms_norm`` is set and
``nn.Identity`` otherwise.

The model is fp32-only: it is built for exact fp32 arithmetic, and reduced
precision or non-deterministic attention kernels change its outputs.
``forward`` raises on non-fp32 hidden states or active autocast, and
attention is pinned to the deterministic math SDPA backend.

Padding is not modeled: the mask is causal-only, so batched generation with
left/right padding is unsupported. The intended use is a single greedy
sequence — stock ``generate(do_sample=False)`` over an unbounded
``DynamicCache``.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_torchwright_custom import TorchwrightCustomConfig

# The fused flash/efficient attention kernels round differently across devices
# and runs; this model's outputs must be bit-reproducible, so attention is
# pinned to the deterministic math backend (plain softmax + matmul).
_SDPA_BACKEND = [SDPBackend.MATH]

# A supported key-padding mask is 2D: (batch, total_keys).
_KEY_PADDING_MASK_NDIM = 2


class TorchwrightCustomAttention(nn.Module):
    """Causal multi-head attention, ``scale=1.0``, no bias.

    ``n_heads`` is this layer's own head count (layers may differ);
    ``d_head`` is shared across all layers.
    """

    def __init__(self, config: TorchwrightCustomConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.d = config.d
        self.d_head = config.d_head
        self.n_heads = config.n_heads_per_layer[layer_idx]
        hd = self.n_heads * self.d_head
        self.q_proj = nn.Linear(self.d, hd, bias=False)
        self.k_proj = nn.Linear(self.d, hd, bias=False)
        self.v_proj = nn.Linear(self.d, hd, bias=False)
        self.o_proj = nn.Linear(hd, self.d, bias=False)

        # Rotary embedding (rotate_half), partial width: the first ``d_rot``
        # dims of every head rotate, the rest pass through unrotated;
        # ``d_rot == d_head`` is full rotary. The frequency grid is derived
        # purely from the config and is cheap, so it is recomputed in
        # ``forward`` on the input's device rather than stored as a buffer —
        # ``from_pretrained``'s meta-device (low_cpu_mem_usage) load path
        # does not materialize non-persistent buffers.
        self.rope_base = float(config.rope_base)
        self.d_rot = int(getattr(config, "d_rot", self.d_head))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, n_heads, T, d_head); cos/sin: (T, d_rot) -> (1, 1, T, d_rot).
        # Rotate the first d_rot dims; dims past d_rot pass through unchanged.
        cos = cos[None, None]
        sin = sin[None, None]
        d_rot = cos.shape[-1]
        if d_rot == x.shape[-1]:
            return x * cos + self._rotate_half(x) * sin
        x_front = x[..., :d_rot]
        x_pass = x[..., d_rot:]
        rotated = x_front * cos + self._rotate_half(x_front) * sin
        return torch.cat([rotated, x_pass], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_values: Cache | None,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        shape = (B, T, self.n_heads, self.d_head)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)

        # Rotate Q and the new K by absolute position (cache_position) before
        # the cache update, so the cache holds already-rotated keys.
        device = hidden_states.device
        p = torch.arange(0, self.d_rot, 2, dtype=torch.float64, device=device)
        inv = (self.rope_base ** (-p / self.d_rot)).to(torch.float32)
        rope_freq = torch.cat([inv, inv])  # (d_rot,)
        ang = cache_position.to(torch.float32)[:, None] * rope_freq[None, :]
        cos = torch.cos(ang)  # (T, d_rot)
        sin = torch.sin(ang)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # attn_mask is (1, 1, T, T_total) boolean, True == visible; it
        # broadcasts over the batch and head axes.
        with sdpa_kernel(_SDPA_BACKEND):
            attn = functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=1.0
            )

        attn = attn.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.o_proj(attn)


class TorchwrightCustomMLP(nn.Module):
    """``fc2(relu(fc1(x)))`` — both linears biased."""

    def __init__(self, config: TorchwrightCustomConfig, layer_idx: int) -> None:
        super().__init__()
        d_hidden = config.d_hidden_per_layer[layer_idx]
        self.fc1 = nn.Linear(config.d, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, config.d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(functional.relu(self.fc1(x)))


class TorchwrightCustomRMSNorm(nn.Module):
    """Root-mean-square norm.

    Computes ``x / sqrt(mean(x^2, -1) + eps) * weight`` — the same form as
    Llama's.
    """

    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.weight


class TorchwrightCustomDecoderLayer(nn.Module):
    """Pre-norm decoder block.

    ``x = x + attn(norm(x)); x = x + mlp(norm(x))``. The norms are
    ``nn.Identity`` when ``config.rms_norm`` is off.
    """

    def __init__(self, config: TorchwrightCustomConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = TorchwrightCustomAttention(config, layer_idx)
        self.mlp = TorchwrightCustomMLP(config, layer_idx)
        # Norm weights exist in the checkpoint only when rms_norm is on;
        # Identity keeps the no-norm state dict free of stray parameters.
        if config.rms_norm:
            self.input_layernorm: nn.Module = TorchwrightCustomRMSNorm(
                config.d, config.rms_norm_eps
            )
            self.post_attention_layernorm: nn.Module = TorchwrightCustomRMSNorm(
                config.d, config.rms_norm_eps
            )
        else:
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_values: Cache | None,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            attn_mask,
            past_key_values,
            cache_position,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TorchwrightCustomPreTrainedModel(PreTrainedModel):
    config_class = TorchwrightCustomConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_sdpa = True

    def __init__(
        self, config: TorchwrightCustomConfig, *inputs: object, **kwargs: object
    ) -> None:
        super().__init__(config, *inputs, **kwargs)
        # Instance attribute (not a class-level mutable default): shared list
        # literals as class attributes are a footgun across instances.
        self._no_split_modules = ["TorchwrightCustomDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        # Real weights come from the checkpoint; a fresh construction only has
        # to be finite and shape-correct. Use ``nn.init.*`` on the parameter
        # (not ``.data.normal_()``): under ``from_pretrained``, transformers
        # patches ``torch.nn.init`` to skip already-loaded parameters, and raw
        # ``.data`` mutation bypasses that guard.
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class TorchwrightCustomModel(TorchwrightCustomPreTrainedModel):
    """Decoder trunk: token embedding, ``n_layers`` decoder blocks, final norm."""

    def __init__(self, config: TorchwrightCustomConfig) -> None:
        super().__init__(config)
        self.embed_tokens: nn.Module = nn.Embedding(config.vocab_size, config.d)
        self.layers = nn.ModuleList(
            [TorchwrightCustomDecoderLayer(config, i) for i in range(config.n_layers)]
        )
        # Final norm before the LM head; Identity when rms_norm is off.
        self.norm = (
            TorchwrightCustomRMSNorm(config.d, config.rms_norm_eps)
            if config.rms_norm
            else nn.Identity()
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **_kwargs: object,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds is not supported; pass input_ids.")
        if input_ids is None:
            raise ValueError("TorchwrightCustomModel requires input_ids.")
        # `use_cache` is a generation parameter, which transformers 5.x strips
        # off the config dataclass — read it defensively, defaulting to True
        # (a bare forward must not crash on the missing attribute).
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", True)

        hidden_states = self.embed_tokens(input_ids)  # (B, T, d)
        _B, T = hidden_states.shape[0], hidden_states.shape[1]
        device = hidden_states.device

        # fp32-only: reduced precision changes this model's outputs. Fail loud
        # rather than emit wrong tokens.
        if hidden_states.dtype != torch.float32:
            raise RuntimeError(
                "This model is fp32-only; got "
                f"{hidden_states.dtype}. Load/keep the model in float32 (no "
                "torch_dtype=fp16/bf16, no .half())."
            )
        try:
            autocast_on = torch.is_autocast_enabled(device.type)
        except TypeError:  # older torch: no device-type arg (CUDA only)
            autocast_on = torch.is_autocast_enabled()
        if autocast_on:
            raise RuntimeError(
                "This model is fp32-only; autocast is active, which runs "
                "matmuls in reduced precision. Run outside autocast."
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_seen = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if cache_position is None:
            if position_ids is not None:
                # Honor caller-supplied absolute positions (single sequence):
                # the rotary rotation and the causal mask both key off
                # cache_position, so map position_ids onto it rather than
                # silently ignoring it.
                cache_position = cast(
                    "torch.LongTensor",
                    position_ids[0].to(device=device, dtype=torch.long),
                )
            else:
                cache_position = cast(
                    "torch.LongTensor",
                    torch.arange(past_seen, past_seen + T, device=device),
                )

        # Causal mask over absolute positions: key j is visible to query p iff
        # j <= p.
        total = past_seen + T
        key_pos = torch.arange(total, device=device)
        mask = (key_pos[None, :] <= cache_position[:, None])[None, None, :, :]
        # Fold in a key-padding mask when one is supplied. The supported shape
        # is a 2D (batch, total_keys) mask covering every key — what `generate`
        # passes for a single unpadded sequence. Anything else is refused
        # loudly rather than silently reduced to causal-only.
        if attention_mask is not None:
            if (
                attention_mask.dim() == _KEY_PADDING_MASK_NDIM
                and attention_mask.shape[-1] == total
            ):
                pad = attention_mask[:, None, None, :].to(torch.bool)
                mask = mask & pad  # -> (B, 1, T, total)
            else:
                raise NotImplementedError(
                    "Unsupported attention_mask of shape "
                    f"{tuple(attention_mask.shape)}; pass a 2D (batch, {total}) "
                    "key-padding mask covering all keys, or None."
                )

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, past_key_values, cache_position)

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class TorchwrightCustomForCausalLM(TorchwrightCustomPreTrainedModel, GenerationMixin):
    """Torchwright decoder with a language-modeling head.

    The head is storage-tied to ``model.embed_tokens`` and has no bias.
    """

    def __init__(self, config: TorchwrightCustomConfig) -> None:
        super().__init__(config)
        self.model = TorchwrightCustomModel(config)
        self.lm_head: nn.Module = nn.Linear(config.d, config.vocab_size, bias=False)
        # Instance attribute (not a class-level mutable default): a dict
        # literal as a class attribute is shared across every instance.
        self._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    def get_decoder(self) -> TorchwrightCustomModel:
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int = 0,
        **kwargs: object,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden = outputs.last_hidden_state  # (B, T, d)

        # Keep only the trailing rows generate asks for (T'==1 during decode).
        if isinstance(logits_to_keep, int):
            slice_idx = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        else:  # a tensor of explicit indices
            slice_idx = logits_to_keep
        hidden = hidden[:, slice_idx, :]

        logits = self.lm_head(hidden)  # (B, T', vocab)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=cast("torch.FloatTensor | None", loss),
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


__all__ = [
    "TorchwrightCustomConfig",
    "TorchwrightCustomForCausalLM",
    "TorchwrightCustomModel",
    "TorchwrightCustomPreTrainedModel",
]
