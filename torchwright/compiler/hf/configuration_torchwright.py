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
* The unembedding is tied to ``embed_table.T`` after a column gather; no
  output bias.
* fp32 throughout — a downcast to fp16/bf16 breaks correctness (the
  cancel-heads trick relies on exact algebraic cancellation).
"""

from __future__ import annotations

from transformers import PretrainedConfig


class TorchwrightConfig(PretrainedConfig):
    """Config for a compiled torchwright token transformer.

    The residual stream is a uniform width ``d`` across all layers (per-layer
    head trimming keeps it uniform; only the internal head count and MLP hidden
    width vary per layer). The token head projects a ``d_embed``-wide table
    lookup into ``d``, adds a ``d_pos``-wide absolute positional encoding
    projected into ``d``, runs ``n_layers`` attention+MLP blocks, gathers
    ``d_output`` columns, and unembeds against the tied ``(vocab_size, d_embed)``
    table.

    Args:
        d: Residual stream width.
        d_head: Per-head dimension (shared across all layers and heads).
        d_embed: Token embedding table width (also the gathered output width).
        d_pos: Positional-encoding width (before projection into ``d``).
        vocab_size: Number of rows in the embedding table.
        n_layers: Number of attention+MLP blocks.
        n_heads_per_layer: List of length ``n_layers`` — trimmed head count
            for each layer.
        d_hidden_per_layer: List of length ``n_layers`` — trimmed MLP hidden
            width for each layer.
        max_seq: Number of precomputed positional-encoding rows (the largest
            absolute position the model can encode).
        d_output: Width of the output column gather feeding the unembed; equals
            ``d_embed`` for a token model.
        head_kind: ``"token"`` — the only head built today (a headless
            float-I/O head is a noted future extension).
        cache_stride: The static KV-cache slot count ``S`` baked into the
            source ONNX artifact, kept for provenance. The native model uses a
            stock unbounded cache.
    """

    model_type = "torchwright"

    def __init__(
        self,
        d: int = 0,
        d_head: int = 0,
        d_embed: int = 0,
        d_pos: int = 0,
        vocab_size: int = 0,
        n_layers: int = 0,
        n_heads_per_layer: list[int] | None = None,
        d_hidden_per_layer: list[int] | None = None,
        max_seq: int = 0,
        d_output: int = 0,
        head_kind: str = "token",
        cache_stride: int | None = None,
        **kwargs,
    ):
        self.d = int(d)
        self.d_head = int(d_head)
        self.d_embed = int(d_embed)
        self.d_pos = int(d_pos)
        self.vocab_size = int(vocab_size)
        self.n_layers = int(n_layers)
        self.n_heads_per_layer = [int(n) for n in (n_heads_per_layer or [])]
        self.d_hidden_per_layer = [int(n) for n in (d_hidden_per_layer or [])]
        self.max_seq = int(max_seq)
        self.d_output = int(d_output)
        self.head_kind = head_kind
        self.cache_stride = None if cache_stride is None else int(cache_stride)
        # Aliases so generic transformers utilities that reach for the canonical
        # field names (cache sizing, repr, sharding heuristics) find them. We own
        # them here and recompute from our own fields, so drop any (possibly
        # stale) values round-tripped in from a serialized config.
        kwargs.pop("num_hidden_layers", None)
        kwargs.pop("hidden_size", None)
        self.num_hidden_layers = self.n_layers
        self.hidden_size = self.d
        super().__init__(**kwargs)
