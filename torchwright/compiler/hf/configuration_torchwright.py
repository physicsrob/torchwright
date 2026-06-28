"""``TorchwrightConfig`` — the HuggingFace config for a compiled torchwright
token model.

This is a **shipped** file: it rides into every saved model directory (copied
verbatim by ``transformers``' ``custom_object_save``) and must load on a
stranger's machine with only ``transformers`` installed — so it imports nothing
beyond the standard library and ``transformers`` (no ``torch``, no
``torchwright``, no relative imports). The hermetic test
``tests/hf/test_shipped_model.py`` enforces that statically.

Every field describes one shape of the pinned forward path that
``torchwright``'s compiler emits for *any* token-I/O graph (see
``compiler/export.py`` ``_emit_cached_layer_nodes`` / the token head at the end
of ``compile_to_onnx``). The values are filled in by
``compiler/hf/convert.py`` from the ONNX artifact's initializer shapes; nothing
here is a free hyperparameter.

Architecture invariants this config does NOT carry a knob for, because the
compiler never varies them:

* No normalization anywhere (no LayerNorm / RMSNorm, no final norm).
* Attention is causal with ``scale=1.0`` (no ``1/sqrt(d_head)``), no bias.
* MLP is ``linear2(relu(linear1(x)))``, both linears biased.
* Positional encoding is plain additive absolute PE.
* The unembedding is an untied ``lm_head`` Linear over the full residual
  stream; no output bias.
* fp32 throughout — a downcast to fp16/bf16 breaks correctness (the
  cancel-heads trick relies on exact algebraic cancellation).
"""

from __future__ import annotations

from transformers import PretrainedConfig


class TorchwrightConfig(PretrainedConfig):
    """Config for a compiled torchwright token transformer.

    The residual stream is a uniform width ``d`` across all layers (per-layer
    head trimming keeps it uniform; only the internal head count and MLP hidden
    width vary per layer). The token head looks up a ``(vocab_size, d)``
    embedding table straight into the residual stream, adds a ``(max_seq, d)``
    absolute positional encoding and a constant vector, runs ``n_layers``
    attention+MLP blocks, and unembeds with an untied ``(vocab_size, d)``
    ``lm_head`` over the full residual.

    Args:
        d: Residual stream width (also the embedding/unembedding table width).
        d_head: Per-head dimension (shared across all layers and heads).
        vocab_size: Number of rows in the embedding table.
        n_layers: Number of attention+MLP blocks.
        n_heads_per_layer: List of length ``n_layers`` — trimmed head count
            for each layer.
        d_hidden_per_layer: List of length ``n_layers`` — trimmed MLP hidden
            width for each layer.
        max_seq: Number of precomputed positional-encoding rows (the largest
            absolute position the model can encode).
        head_kind: ``"token"`` — the only head built today.
        cache_stride: The static KV-cache slot count ``S`` baked into the
            source ONNX artifact, kept for provenance. The native model uses a
            stock unbounded cache.
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
        max_seq: int = 0,
        head_kind: str = "token",
        cache_stride: int | None = None,
        rope_base: float = 0.0,
        rotary_enable_per_layer: list | None = None,
        **kwargs,
    ):
        self.d = int(d)
        self.d_head = int(d_head)
        self.vocab_size = int(vocab_size)
        self.n_layers = int(n_layers)
        self.n_heads_per_layer = [int(n) for n in (n_heads_per_layer or [])]
        self.d_hidden_per_layer = [int(n) for n in (d_hidden_per_layer or [])]
        self.max_seq = int(max_seq)
        self.head_kind = head_kind
        self.cache_stride = None if cache_stride is None else int(cache_stride)
        # RoPE: rope_base 0.0 means "no rotary" (back-compat for the sinusoidal
        # models).  rotary_enable_per_layer[i] is a per-head 0/1 list (length
        # n_heads_per_layer[i]) marking which heads of layer i rotate full-width
        # (rotate_half over d_head).  See docs/rope_port_plan.md §6.
        self.rope_base = float(rope_base)
        self.rotary_enable_per_layer = [
            [int(x) for x in row] for row in (rotary_enable_per_layer or [])
        ]
        # Aliases so generic transformers utilities that reach for the canonical
        # field names (cache sizing, repr, sharding heuristics) find them. We own
        # them here and recompute from our own fields, so drop any (possibly
        # stale) values round-tripped in from a serialized config.
        kwargs.pop("num_hidden_layers", None)
        kwargs.pop("hidden_size", None)
        self.num_hidden_layers = self.n_layers
        self.hidden_size = self.d
        # Untied embeddings by default: the compiler emits a separate lm_head.
        # PretrainedConfig defaults this to True, which would make HF's
        # tie_weights() clone lm_head onto embed_tokens and corrupt the unembed
        # (the two tables live at different residual columns).
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
