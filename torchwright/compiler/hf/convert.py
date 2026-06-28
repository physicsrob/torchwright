"""Build-time converter: a torchwright token-I/O ONNX artifact → a native
``TorchwrightForCausalLM`` (weights + ``TorchwrightConfig``).

This is **not** a shipped file — it imports ``onnx`` and reads the artifact at
build time. The output (safetensors + ``config.json`` + the shipped
``modeling_torchwright`` / ``configuration_torchwright`` / tokenizer files) is
what gets published; nothing here is needed to *load* a published model.

It reads every initializer from the ONNX graph (densifying the COO-sparse ones
the exporter emits for mostly-zero matrices), maps them one-for-one onto the
native module's parameters/buffers, and asserts that no initializer is left
unmapped — so a future exporter weight that this converter doesn't know about
fails loudly instead of silently dropping.

Initializer → parameter map (ground truth: ``compiler/export.py``
``compile_to_onnx``; layouts confirmed against ``components/*``):

    embed_table              -> model.embed_tokens.weight  (vocab, d)
    lm_head                  -> lm_head.weight              (vocab, d)  UNTIED
    pos_encoding_full        -> model.pos_encoding_full    (max_seq, d)
    constant_values          -> model.constant_values      (d,)
    l{i}_WQ/WK/WV  (d, hd)   -> q/k/v_proj.weight  = M.T    (hd, d)
    l{i}_WO        (hd, d)   -> o_proj.weight      = M.T    (d, hd)
    l{i}_W1        (d, d_h)  -> linear1.weight     = M.T    (d_h, d)
    l{i}_b1        (d_h,)    -> linear1.bias
    l{i}_W2        (d_h, d)  -> linear2.weight     = M.T    (d, d_h)
    l{i}_b2        (d,)      -> linear2.bias

``nn.Linear`` stores weights as ``(out, in)`` and computes ``x @ W.T``; the
compiler stores ``M`` and computes ``x @ M``. So every Linear weight is ``M.T``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from torchwright.compiler.export import (
    TOKEN_META_FORMAT,
    debug_meta_path_for,
    meta_path_for,
)

# These initializers are the cached-protocol scaffolding (mask sentinel, slot
# arange, reshape constants) — structural, not weights. Every other initializer
# must map to a model parameter or the converter raises.
_IGNORABLE_EXACT = {
    "arange_S",
    "_f32_causal_sentinel_s",
    "_axes0_1d",
    "_axes1_1d",
}
_IGNORABLE_SUFFIX = ("_qkv_view_shape", "_ctx_flat_shape")


def _load_all_initializers(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    """Name -> dense ndarray for every initializer, densifying COO-sparse ones.

    The exporter stores mostly-zero float matrices as ``SparseTensorProto`` with
    flat int64 indices (``export.py:_tensor_to_proto``); reverse that here.
    """
    out: Dict[str, np.ndarray] = {}
    for tp in model.graph.initializer:
        out[tp.name] = numpy_helper.to_array(tp)
    for sp in model.graph.sparse_initializer:
        dims = list(sp.dims)
        values = numpy_helper.to_array(sp.values)
        indices = numpy_helper.to_array(sp.indices).reshape(-1)
        dense = np.zeros(int(np.prod(dims)), dtype=values.dtype)
        dense[indices] = values
        out[sp.values.name] = dense.reshape(dims)
    return out


def _read_meta(onnx_path: str) -> dict:
    meta_path = meta_path_for(onnx_path)
    with open(meta_path) as f:
        return json.load(f)


def _read_debug(onnx_path: str) -> Optional[dict]:
    debug_path = debug_meta_path_for(onnx_path)
    if not os.path.exists(debug_path):
        return None
    with open(debug_path) as f:
        return json.load(f)


def build_config(
    inits: Dict[str, np.ndarray],
    meta: dict,
    *,
    bos_token: Optional[str] = None,
    eos_token: Optional[str] = None,
):
    """Derive a :class:`TorchwrightConfig` from the ONNX initializer shapes."""
    from .configuration_torchwright import TorchwrightConfig

    fmt = meta.get("format")
    assert fmt == TOKEN_META_FORMAT, (
        f"artifact meta format {fmt!r} != expected {TOKEN_META_FORMAT!r}; this "
        f"converter reads the vanilla untied token layout — re-export the "
        f"artifact with the current exporter."
    )

    vocab: List[str] = list(meta["vocab"])

    # The embedding table is over-allocated: its row count is the model's logit
    # width (the unembed produces one logit per row), which the compiler pads
    # beyond the meaningful token count. Rows past len(vocab) are zero
    # embeddings → zero logits → never argmax-selected over a real token, and
    # are never valid INPUT ids. So vocab_size (the model's output dimension) is
    # the table's row count, not the tokenizer's token count.
    embed_table = inits["embed_table"]
    # Vanilla untied layout: embed_table is (vocab, d) — the full residual width
    # — and the unembed is a separate (vocab, d) lm_head, so there is no
    # d_embed / d_pos / output-gather width left to read.
    vocab_size, d = embed_table.shape
    assert (
        len(vocab) <= vocab_size
    ), f"meta vocab len {len(vocab)} exceeds embed_table rows {vocab_size}"
    max_seq = inits["pos_encoding_full"].shape[0]

    n_layers = sum(1 for k in inits if re.fullmatch(r"l\d+_WQ", k))
    assert n_layers > 0, "no l{i}_WQ initializers — not a token transformer?"

    # d_head is shared across layers; read it from the per-layer reshape const
    # l{i}_qkv_view_shape = [0, nh, d_head].
    d_head = int(inits["l0_qkv_view_shape"][2])

    n_heads_per_layer: List[int] = []
    d_hidden_per_layer: List[int] = []
    for i in range(n_layers):
        nh = int(inits[f"l{i}_qkv_view_shape"][1])
        hd = inits[f"l{i}_WQ"].shape[1]
        assert (
            hd == nh * d_head
        ), f"layer {i}: WQ width {hd} != n_heads {nh} * d_head {d_head}"
        n_heads_per_layer.append(nh)
        d_hidden_per_layer.append(int(inits[f"l{i}_W1"].shape[1]))

    # RoPE: the exporter records `rope_base` in the sidecar meta and bakes, for
    # each layer with a rotary head, an `l{i}_rope_enable_q` (nh, 1, 1) per-head
    # enable.  Absent = the sinusoidal (non-rotary) model.
    rope_base = float(meta.get("rope_base", 0.0))
    rotary_enable_per_layer: List[List[int]] = []
    for i in range(n_layers):
        key = f"l{i}_rope_enable_q"
        if key in inits:
            rotary_enable_per_layer.append(
                [int(round(float(x))) for x in inits[key].reshape(-1)]
            )
        else:
            rotary_enable_per_layer.append([0] * n_heads_per_layer[i])

    # Resolve bos/eos ids. A token of None means "this model has no bos/eos";
    # a non-None token that isn't in the vocab is a caller error (e.g. wrong
    # bracketing) and must fail loudly rather than silently become None.
    def _token_id(token, kind):
        if token is None:
            return None
        if token not in vocab:
            raise ValueError(
                f"requested {kind}_token {token!r} is not in the artifact vocab; "
                f"pass the correct {kind}_token (or {kind}_token=None if the "
                f"model has no {kind})"
            )
        return vocab.index(token)

    bos_id = _token_id(bos_token, "bos")
    eos_id = _token_id(eos_token, "eos")

    return TorchwrightConfig(
        d=int(d),
        d_head=int(d_head),
        vocab_size=int(vocab_size),
        n_layers=int(n_layers),
        n_heads_per_layer=n_heads_per_layer,
        d_hidden_per_layer=d_hidden_per_layer,
        max_seq=int(max_seq),
        head_kind="token",
        cache_stride=meta.get("cache_stride"),
        rope_base=rope_base,
        rotary_enable_per_layer=rotary_enable_per_layer,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        tie_word_embeddings=False,
        use_cache=True,
    )


def _t(arr: np.ndarray):
    import torch

    # .copy() — the ONNX-backed arrays are read-only; torch needs writable.
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32).copy())


def build_state_dict(config, inits: Dict[str, np.ndarray]) -> Tuple[dict, set]:
    """Map initializers to the native state dict; return (state_dict, consumed).

    ``consumed`` is the set of initializer names this used, so the caller can
    assert nothing weight-like was left behind.
    """
    consumed: set = set()

    def take(name: str) -> np.ndarray:
        consumed.add(name)
        return inits[name]

    embed_table = take("embed_table")  # (vocab, d)
    sd: dict = {
        "model.embed_tokens.weight": _t(embed_table),
        "lm_head.weight": _t(take("lm_head")),  # untied (vocab, d)
        "model.pos_encoding_full": _t(take("pos_encoding_full")),  # (max_seq, d)
        "model.constant_values": _t(take("constant_values")),  # (d,)
    }

    for i in range(config.n_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_proj.weight"] = _t(take(f"l{i}_WQ").T)  # (hd, d)
        sd[f"{p}.self_attn.k_proj.weight"] = _t(take(f"l{i}_WK").T)
        sd[f"{p}.self_attn.v_proj.weight"] = _t(take(f"l{i}_WV").T)
        sd[f"{p}.self_attn.o_proj.weight"] = _t(take(f"l{i}_WO").T)  # (d, hd)
        sd[f"{p}.mlp.linear1.weight"] = _t(take(f"l{i}_W1").T)  # (d_h, d)
        sd[f"{p}.mlp.linear1.bias"] = _t(take(f"l{i}_b1"))
        sd[f"{p}.mlp.linear2.weight"] = _t(take(f"l{i}_W2").T)  # (d, d_h)
        sd[f"{p}.mlp.linear2.bias"] = _t(take(f"l{i}_b2"))

    # RoPE constants are config-derived (rebuilt in the model from rope_base +
    # the per-head enable), not state-dict weights — mark them consumed so the
    # all-mapped check passes.
    for name in ("rope_freq", "rope_base", "rope_split"):
        if name in inits:
            consumed.add(name)
    for i in range(config.n_layers):
        for suffix in ("rope_enable_q", "rope_enable_k"):
            key = f"l{i}_{suffix}"
            if key in inits:
                consumed.add(key)

    return sd, consumed


def _assert_all_mapped(inits: Dict[str, np.ndarray], consumed: set) -> None:
    leftover = []
    for name in inits:
        if name in consumed:
            continue
        if name in _IGNORABLE_EXACT:
            continue
        if any(name.endswith(suf) for suf in _IGNORABLE_SUFFIX):
            continue
        leftover.append(name)
    assert not leftover, (
        f"unmapped initializers (a new weight the converter doesn't handle?): "
        f"{sorted(leftover)}"
    )


def convert_onnx_to_hf(
    onnx_path: str,
    *,
    bos_token: Optional[str] = "<bos>",
    eos_token: Optional[str] = "<eos>",
):
    """Load an ONNX token artifact and return an in-memory ``TorchwrightForCausalLM``.

    fp32, ``eval()`` mode. ``bos_token`` / ``eos_token`` are looked up in the
    vocab to populate the config's token ids, and must exist in it (a wrong
    token raises; pass ``None`` for a model with no bos/eos). The generic
    example vocab's control tokens are ``"<bos>"`` / ``"<eos>"`` (hence the
    defaults); DOOM passes ``begin`` / ``done``.
    """
    import torch

    from .modeling_torchwright import TorchwrightForCausalLM

    model_proto = onnx.load(onnx_path)
    inits = _load_all_initializers(model_proto)
    meta = _read_meta(onnx_path)

    config = build_config(inits, meta, bos_token=bos_token, eos_token=eos_token)
    sd, consumed = build_state_dict(config, inits)
    _assert_all_mapped(inits, consumed)

    model = TorchwrightForCausalLM(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Untied: embed_tokens.weight and lm_head.weight are loaded as two separate
    # tensors, so neither may be missing.
    assert not missing, f"missing params after load: {missing}"
    assert not unexpected, f"unexpected params in state dict: {unexpected}"

    model = model.to(torch.float32).eval()
    return model


def read_vocab(onnx_path: str) -> List[str]:
    """The token vocabulary list from the artifact's meta sidecar."""
    return list(_read_meta(onnx_path)["vocab"])


def save_bundle(
    onnx_path: str,
    save_dir: str,
    *,
    bos_token: Optional[str] = "<bos>",
    eos_token: Optional[str] = "<eos>",
    add_bos_token: bool = True,
    write_tokenizer: bool = True,
):
    """Convert and ``save_pretrained`` a full trust-remote-code bundle.

    Writes safetensors + ``config.json`` + the shipped modeling/config files
    (via ``register_for_auto_class``), and — unless ``write_tokenizer`` is
    False — the generic ``TorchwrightTokenizer`` over the artifact vocab.
    Returns the saved model.
    """
    from .configuration_torchwright import TorchwrightConfig
    from .modeling_torchwright import TorchwrightForCausalLM

    model = convert_onnx_to_hf(onnx_path, bos_token=bos_token, eos_token=eos_token)
    TorchwrightConfig.register_for_auto_class()
    TorchwrightForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)

    if write_tokenizer:
        from .tokenization_torchwright import TorchwrightTokenizer

        vocab = read_vocab(onnx_path)
        vocab_path = os.path.join(save_dir, "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        tok = TorchwrightTokenizer(
            vocab_file=vocab_path,
            bos_token=bos_token,
            eos_token=eos_token,
            add_bos_token=add_bos_token,
        )
        TorchwrightTokenizer.register_for_auto_class()
        tok.save_pretrained(save_dir)

    return model
