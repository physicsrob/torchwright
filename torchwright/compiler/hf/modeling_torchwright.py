"""``TorchwrightForCausalLM`` — a native HuggingFace causal LM that reproduces a
compiled torchwright token model.

This is a **shipped** file: it rides into every saved model directory (copied
verbatim by ``transformers``' ``custom_object_save``, together with its one
relative-imported sibling ``configuration_torchwright``) and must load on a
stranger's machine with only ``torch`` + ``transformers`` installed — no
``torchwright``. So it imports only the standard library, ``torch``,
``transformers``, and ``.configuration_torchwright``. The hermetic test
``tests/hf/test_shipped_model.py`` enforces that statically.

The forward path is transcribed one-for-one from the compiler's ONNX emission
(``compiler/export.py`` ``compile_to_onnx`` token head +
``_emit_cached_layer_nodes``):

    tok      = embed_table[input_ids]              # (B, T, d_embed)
    inp_res  = tok @ embedding_proj                # (B, T, d)
    pos      = pos_encoding_full[cache_position]   # gather PE by ABSOLUTE pos
    pos_res  = pos @ pos_proj                      # (B, T, d)
    res      = inp_res + pos_res + constant_values # additive absolute PE + const
    for each layer:
        res  = res + attn(res)                     # causal, scale=1.0, no bias
        res  = res + linear2(relu(linear1(res)))   # both linears biased
    gathered = res[:, :, output_gather_indices]    # (B, T, d_output == d_embed)
    logits   = gathered @ embed_table.T            # TIED unembed, no bias

Correctness invariants (all from the compiler source; see the config docstring):

* No normalization anywhere; no final norm.
* Attention uses ``scale=1.0`` (no ``1/sqrt(d_head)``) and the exact-math SDPA
  backend. The cancel-heads trick the compiler uses relies on
  ``attn_out + skip == 0`` *algebraically*; a single fp32-LSB perturbation from
  the flash / efficient attention kernels flips output bits, so we force
  ``SDPBackend.MATH`` and keep everything fp32. **Never downcast to fp16/bf16.**
* The mask is an overwrite of hidden positions, equivalent to the compiler's
  ``Where(slot > query_pos, sentinel, logit)``: a key at absolute position ``j``
  is visible to a query at absolute position ``p`` iff ``j <= p``. The mask
  derives purely from ``cache_position`` — the only position fact the host
  supplies — exactly as the ONNX preamble does.

Generation is plain greedy argmax-append with an EOS stop, i.e. stock
``generate(do_sample=False)``. The native model uses a stock unbounded
``DynamicCache``.

Padding is not modeled: the mask is causal-only, so batched generation with
left/right padding is unsupported. The consumers (calculator, DOOM) run a
single greedy sequence, which is exactly what this covers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_torchwright import TorchwrightConfig

# The compiler emits attention against the exact-math softmax+matmul. On A100 the
# default fp32 SDPA backend is EFFICIENT_ATTENTION, which perturbs values by a
# single fp32 mantissa-LSB on some inputs — enough to break the algebraic
# cancellation the compiled heads depend on. MATH matches the compiler bit-for-bit.
_SDPA_BACKEND = [SDPBackend.MATH]


class TorchwrightAttention(nn.Module):
    """Per-layer causal multi-head attention, ``scale=1.0``, no bias.

    Q/K/V/O are plain ``nn.Linear(bias=False)`` whose weights are the compiled
    projection matrices (transposed into ``nn.Linear``'s ``(out, in)`` layout by
    the converter). The head count ``n_heads`` is this layer's trimmed count, so
    each layer carries only the heads it uses; ``d_head`` is shared across all
    layers.
    """

    def __init__(self, config: TorchwrightConfig, layer_idx: int):
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

        # RoPE (LLaMA3 rotate_half, half-split, scale baked into base). A head is
        # rotary iff its enable bit is set; rotation is full-width over d_head, by
        # ABSOLUTE position (cache_position). rope_base 0.0 => the sinusoidal
        # (non-rotary) model, in which case nothing below runs. Buffers are
        # non-persistent: rebuilt from the serialized config on load.
        enable = (
            config.rotary_enable_per_layer[layer_idx]
            if config.rotary_enable_per_layer
            else []
        )
        self.rope_active = config.rope_base > 0.0 and any(enable)
        if self.rope_active:
            p = torch.arange(0, self.d_head, 2, dtype=torch.float64)
            inv = (config.rope_base ** (-p / self.d_head)).to(torch.float32)
            self.register_buffer(
                "rope_freq", torch.cat([inv, inv]), persistent=False
            )  # (d_head,)
            self.register_buffer(
                "rope_enable",
                torch.tensor(enable, dtype=torch.float32).view(1, self.n_heads, 1, 1),
                persistent=False,
            )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, n_heads, T, d_head); cos/sin: (T, d_head) -> (1, 1, T, d_head).
        # Per-head enable blend keeps non-rotary heads untouched.
        cos = cos[None, None]
        sin = sin[None, None]
        rot = x * cos + self._rotate_half(x) * sin
        return x + self.rope_enable * (rot - x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_values: Optional[Cache],
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        shape = (B, T, self.n_heads, self.d_head)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)

        # RoPE rotates Q and the NEW K by absolute position, before the cache
        # update, so the cache stores already-rotated K (slot == position).
        if self.rope_active:
            ang = cache_position.to(torch.float32)[:, None] * self.rope_freq[None, :]
            cos = torch.cos(ang)  # (T, d_head)
            sin = torch.sin(ang)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # attn_mask is (1, 1, T, T_total) boolean, True == visible; it broadcasts
        # over the batch and head axes (the head count varies per layer).
        with sdpa_kernel(_SDPA_BACKEND):
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=1.0
            )

        attn = attn.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.o_proj(attn)


class TorchwrightMLP(nn.Module):
    """``linear2(relu(linear1(x)))`` — both linears biased."""

    def __init__(self, config: TorchwrightConfig, layer_idx: int):
        super().__init__()
        d_hidden = config.d_hidden_per_layer[layer_idx]
        self.linear1 = nn.Linear(config.d, d_hidden, bias=True)
        self.linear2 = nn.Linear(d_hidden, config.d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class TorchwrightDecoderLayer(nn.Module):
    """One block: ``x = x + attn(x); x = x + mlp(x)`` — no normalization."""

    def __init__(self, config: TorchwrightConfig, layer_idx: int):
        super().__init__()
        self.self_attn = TorchwrightAttention(config, layer_idx)
        self.mlp = TorchwrightMLP(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_values: Optional[Cache],
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            hidden_states, attn_mask, past_key_values, cache_position
        )
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states


class TorchwrightPreTrainedModel(PreTrainedModel):
    config_class = TorchwrightConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["TorchwrightDecoderLayer"]
    _supports_sdpa = True

    def _init_weights(self, module):
        # Real weights are loaded over this by the converter / from_pretrained;
        # this only has to be finite and shape-correct for a fresh construction.
        #
        # Use ``nn.init.*`` ON THE PARAMETER (not ``.data.normal_()``): under
        # ``from_pretrained`` transformers patches the ``torch.nn.init``
        # functions to respect each param's ``_is_hf_initialized`` flag, so
        # already-loaded weights are skipped. Calling the raw ``Tensor.normal_``
        # on ``.data`` bypasses that guard and silently re-randomizes loaded
        # weights — the bug that made reloaded models emit garbage.
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class TorchwrightModel(TorchwrightPreTrainedModel):
    """The token transformer trunk: embed + project + additive PE + N blocks."""

    def __init__(self, config: TorchwrightConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_embed)
        # Fixed projection matrices the compiler emits (mostly scatter/permute);
        # stored as parameters so they serialize with the model.
        self.embedding_proj = nn.Parameter(torch.zeros(config.d_embed, config.d))
        self.pos_proj = nn.Parameter(torch.zeros(config.d_pos, config.d))
        # Precomputed absolute positional-encoding table and the input constant
        # vector are lookup/bias data, not matmul weights — registered as
        # persistent buffers (still saved into the safetensors state dict).
        self.register_buffer(
            "pos_encoding_full",
            torch.zeros(config.max_seq, config.d_pos),
            persistent=True,
        )
        self.register_buffer("constant_values", torch.zeros(config.d), persistent=True)
        self.register_buffer(
            "output_gather_indices",
            torch.zeros(config.d_output, dtype=torch.long),
            persistent=True,
        )
        self.layers = nn.ModuleList(
            [TorchwrightDecoderLayer(config, i) for i in range(config.n_layers)]
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is not None:
            # The compiler emits a table lookup into d_embed, not a (B, T, d)
            # hidden-state seed, so the usual inputs_embeds contract doesn't
            # apply here. Reject it rather than silently misinterpret its width.
            raise NotImplementedError(
                "inputs_embeds is not supported for torchwright token models; "
                "pass input_ids."
            )
        if input_ids is None:
            raise ValueError("TorchwrightModel requires input_ids.")
        # `use_cache` is a generation parameter, which transformers 5.x strips
        # off the config dataclass — read it defensively, defaulting to True
        # (a bare forward must not crash on the missing attribute).
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", True)

        tok = self.embed_tokens(input_ids)  # (B, T, d_embed)
        B, T = tok.shape[0], tok.shape[1]
        device = tok.device

        # fp32-only: the compiled attention relies on exact algebraic
        # cancellation (cancel heads: attn_out + skip == 0); a fp16/bf16
        # downcast or active autocast runs the matmuls in reduced precision and
        # silently breaks correctness. Fail loud rather than emit wrong tokens.
        if tok.dtype != torch.float32:
            raise RuntimeError(
                "torchwright token models are fp32-only; got "
                f"{tok.dtype}. Load/keep the model in float32 (no "
                "torch_dtype=fp16/bf16, no .half())."
            )
        try:
            autocast_on = torch.is_autocast_enabled(device.type)
        except TypeError:  # older torch: no device-type arg (CUDA only)
            autocast_on = torch.is_autocast_enabled()
        if autocast_on:
            raise RuntimeError(
                "torchwright token models are fp32-only; autocast is active, "
                "which runs matmuls in reduced precision and breaks the "
                "compiled cancellation contract. Run outside autocast."
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_seen = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if cache_position is None:
            if position_ids is not None:
                # Honor caller-supplied absolute positions (single sequence):
                # the PE gather and the causal mask both key off cache_position,
                # so map position_ids onto it rather than silently ignoring it.
                cache_position = position_ids[0].to(device=device, dtype=torch.long)
            else:
                cache_position = torch.arange(past_seen, past_seen + T, device=device)

        # Token + absolute-PE + constant residual seed.
        inp_res = tok @ self.embedding_proj  # (B, T, d)
        pos = self.pos_encoding_full[cache_position]  # (T, d_pos)
        pos_res = pos @ self.pos_proj  # (T, d)
        res = inp_res + pos_res + self.constant_values  # (B, T, d)

        # Causal mask over absolute positions: key j visible to query p iff j<=p.
        total = past_seen + T
        key_pos = torch.arange(total, device=device)
        mask = (key_pos[None, :] <= cache_position[:, None])[None, None, :, :]
        # Fold in a key-padding mask when one is supplied. The supported shape is
        # a 2D (batch, total_keys) mask covering every key — exactly what
        # `generate` passes (all-ones, a no-op, for the single greedy sequence).
        # Anything else (4D prebuilt masks, or a 2D mask that doesn't cover all
        # keys) is NOT part of this model's contract: refuse it loudly rather
        # than silently fall back to causal-only and emit plausible-but-wrong
        # logits on padded input.
        if attention_mask is not None:
            if attention_mask.dim() == 2 and attention_mask.shape[-1] == total:
                pad = attention_mask[:, None, None, :].to(torch.bool)
                mask = mask & pad  # -> (B, 1, T, total)
            else:
                raise NotImplementedError(
                    "Unsupported attention_mask of shape "
                    f"{tuple(attention_mask.shape)}; pass a 2D (batch, {total}) "
                    "key-padding mask covering all keys, or None."
                )

        for layer in self.layers:
            res = layer(res, mask, past_key_values, cache_position)

        return BaseModelOutputWithPast(
            last_hidden_state=res,
            past_key_values=past_key_values if use_cache else None,
        )


class TorchwrightForCausalLM(TorchwrightPreTrainedModel, GenerationMixin):
    """Compiled torchwright token model as a standard causal LM.

    The unembed is tied to the embedding table: ``lm_head.weight`` shares
    storage with ``model.embed_tokens.weight`` (both are the compiler's
    ``embed_table``), so ``logits = gathered @ embed_table.T``.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: TorchwrightConfig):
        super().__init__(config)
        self.model = TorchwrightModel(config)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        **kwargs,
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

        # Gather the output columns, then unembed against the tied table.
        gathered = hidden[:, :, self.model.output_gather_indices]  # (B, T', d_embed)
        logits = self.lm_head(gathered)  # (B, T', vocab)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


__all__ = [
    "TorchwrightConfig",
    "TorchwrightPreTrainedModel",
    "TorchwrightModel",
    "TorchwrightForCausalLM",
]
