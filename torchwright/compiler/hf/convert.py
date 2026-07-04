"""Build-time converter: a torchwright token-I/O ONNX artifact → an HF model.

This is **not** a shipped file — it imports ``onnx`` and reads the artifact at
build time. The output (safetensors + ``config.json`` + tokenizer files, plus
the shipped ``modeling_torchwright`` / ``configuration_torchwright`` files on
the native path) is what gets published; nothing here is needed to *load* a
published model.

Two targets, routed on the artifact's emission meta
(:func:`_conversion_target`; docs/phi3_conversion_plan.md):

- **relu, biased** → the native ``TorchwrightForCausalLM`` (custom modeling
  code, ``trust_remote_code`` bundle).  Unchanged legacy path.
- **swish, bias=False, rms_norm** → stock ``Phi3ForCausalLM`` — no custom
  modeling file, no custom tokenizer code.  Phi-3 is the one stock
  architecture whose modeling code *honors* ``partial_rotary_factor`` with
  torchwright's exact rotation semantics (half-split pairing over the rotary
  front, NoPE tail passed through bit-exactly).
- anything else refuses loudly.

Both paths read every initializer from the ONNX graph (densifying the
COO-sparse ones the exporter emits for mostly-zero matrices), map them onto
the target module's parameters, and assert that no initializer is left
unmapped — so a future exporter weight that this converter doesn't know about
fails loudly instead of silently dropping.

Native initializer → parameter map (ground truth: ``compiler/export.py``
``compile_to_onnx``; layouts confirmed against ``components/*``):

    embed_table              -> model.embed_tokens.weight  (vocab, d)
                                (rows carry the folded const seeds, token.v5)
    lm_head                  -> lm_head.weight              (vocab, d)  UNTIED
    rope_freq/_base/_split   -> TorchwrightConfig.rope_base (config-derived; not a param)
    l{i}_input_layernorm     -> model.layers.{i}.input_layernorm.weight          (d,)
    l{i}_post_attention_layernorm -> model.layers.{i}.post_attention_layernorm.weight (d,)
    final_norm               -> model.norm.weight          (d,)   [rms_norm only]
    l{i}_WQ/WK/WV  (d, hd)   -> q/k/v_proj.weight  = M.T    (hd, d)
    l{i}_WO        (hd, d)   -> o_proj.weight      = M.T    (d, hd)
    l{i}_W1        (d, d_h)  -> fc1.weight         = M.T    (d_h, d)
    l{i}_b1        (d_h,)    -> fc1.bias
    l{i}_W2        (d_h, d)  -> fc2.weight         = M.T    (d, d_h)
    l{i}_b2        (d,)      -> fc2.bias

Phi-3 map (docs/phi3_conversion_plan.md "The weight mapping"): fused
``qkv_proj`` / ``gate_up_proj`` are plain row concatenations, ``√d_head`` is
folded into the Q rows (the artifact's scores are raw ``Q·Kᵀ``; Phi-3
multiplies logits by ``d_head^-0.5`` at runtime), and per-layer trimmed
head counts / MLP widths are zero-padded to the uniform stock shape —
every padded contribution is exactly zero (see
:func:`build_phi3_state_dict`).

``nn.Linear`` stores weights as ``(out, in)`` and computes ``x @ W.T``; the
compiler stores ``M`` and computes ``x @ M``. So every Linear weight is ``M.T``.
"""

from __future__ import annotations

import json
import math
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
    # RMSNorm epsilon scalar: a structural constant, not a weight — the eps
    # rides into the HF config via the meta, so this initializer is unmapped.
    "_rms_eps",
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


def _assert_meta_format(meta: dict) -> None:
    fmt = meta.get("format")
    assert fmt == TOKEN_META_FORMAT, (
        f"artifact meta format {fmt!r} != expected {TOKEN_META_FORMAT!r}; this "
        f"converter reads the vanilla untied token layout — re-export the "
        f"artifact with the current exporter."
    )


def _conversion_target(meta: dict) -> str:
    """Route an artifact to its HF target on the emission meta.

    Returns ``"native"`` (relu, biased → ``TorchwrightForCausalLM``) or
    ``"phi3"`` (swish, bias=False, rms_norm → stock ``Phi3ForCausalLM``);
    every other combination refuses loudly.  This is the routing that
    replaced ``build_config``'s standalone swish / bias=False refusals
    (docs/phi3_conversion_plan.md P1).
    """
    activation = meta.get("activation", "relu")
    bias = meta.get("bias", True)
    rms_norm = meta.get("rms_norm", False)
    if activation == "relu":
        if bias is False:
            raise NotImplementedError(
                "HF conversion of a bias=False artifact is not implemented "
                "for the relu machine: the native module hardcodes biased "
                "MLP linears, and the stock Phi-3 path takes swish "
                "artifacts only (docs/phi3_conversion_plan.md)."
            )
        return "native"
    if activation == "swish":
        if bias is not False or not rms_norm:
            raise NotImplementedError(
                f"Phi-3 conversion of a swish artifact requires the stock "
                f"biasless normed emission; this artifact has bias={bias}, "
                f"rms_norm={rms_norm}.  Re-export with "
                f"compile_to_onnx(..., bias=False) (rms_norm is on by "
                f"default at a power-of-two d)."
            )
        return "phi3"
    raise NotImplementedError(
        f"HF conversion of the {activation!r} machine is not implemented: "
        f"the native module models fc1/ReLU/fc2 MLPs and the Phi-3 path "
        f"models gate/up/down SwiGLU MLPs only."
    )


def build_config(
    inits: Dict[str, np.ndarray],
    meta: dict,
    *,
    bos_token: Optional[str] = None,
    eos_token: Optional[str] = None,
):
    """Derive a :class:`TorchwrightConfig` from the ONNX initializer shapes."""
    from .configuration_torchwright import TorchwrightConfig

    _assert_meta_format(meta)
    # Routing guard for direct callers: this builds the NATIVE config; a
    # swish artifact routes to build_phi3_config (and oddballs raise inside).
    target = _conversion_target(meta)
    assert target == "native", (
        f"build_config builds the native TorchwrightConfig but this artifact "
        f"routes to {target!r} — use convert_onnx_to_hf (which routes) or "
        f"build_phi3_config directly"
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
    # No additive PE table under RoPE; the artifact's position bound is the
    # baked slot count arange_S.
    max_position_embeddings = inits["arange_S"].shape[0]

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

    # RoPE on one global grid; the exporter records the single shared `rope_base`
    # and the partial-rotary width `d_rot` in the sidecar meta, and the converter
    # rebuilds the rotation from them.  `d_rot` defaults to `d_head` (full rotary)
    # for artifacts exported before partial rotary existed.
    rope_base = float(meta.get("rope_base", 500000.0))
    rope_d_rot = int(meta.get("d_rot", d_head))

    bos_id = _resolve_token_id(vocab, bos_token, "bos")
    eos_id = _resolve_token_id(vocab, eos_token, "eos")

    # RMSNorm: the gain weights are read from the initializers (build_state_dict);
    # the eps is not a tensor, so it rides in via the meta.  A norm-on artifact
    # carries a `final_norm` weight — cross-check the meta flag against it so a
    # malformed pairing fails loudly rather than silently dropping the norm.
    rms_norm = bool(meta.get("rms_norm", False))
    has_final_norm = "final_norm" in inits
    assert rms_norm == has_final_norm, (
        f"meta rms_norm={rms_norm} disagrees with the presence of a final_norm "
        f"initializer ({has_final_norm}); the artifact is inconsistent — re-export"
    )
    # Explicit None check, not `or`: a legitimately-recorded eps of 0.0 is
    # falsy and must survive (coercing it to 1e-5 would mismatch the artifact).
    _eps = meta.get("rms_norm_eps")
    rms_norm_eps = 1e-5 if _eps is None else float(_eps)

    return TorchwrightConfig(
        d=int(d),
        d_head=int(d_head),
        vocab_size=int(vocab_size),
        n_layers=int(n_layers),
        n_heads_per_layer=n_heads_per_layer,
        d_hidden_per_layer=d_hidden_per_layer,
        max_position_embeddings=int(max_position_embeddings),
        rope_base=rope_base,
        d_rot=rope_d_rot,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        tie_word_embeddings=False,
        use_cache=True,
        rms_norm=rms_norm,
        rms_norm_eps=rms_norm_eps,
    )


def _resolve_token_id(vocab: List[str], token: Optional[str], kind: str):
    """Resolve a control token to its vocab id.

    A token of None means "this model has no bos/eos"; a non-None token that
    isn't in the vocab is a caller error (e.g. wrong bracketing) and must
    fail loudly rather than silently become None.
    """
    if token is None:
        return None
    if token not in vocab:
        raise ValueError(
            f"requested {kind}_token {token!r} is not in the artifact vocab; "
            f"pass the correct {kind}_token (or {kind}_token=None if the "
            f"model has no {kind})"
        )
    return vocab.index(token)


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

    embed_table = take("embed_table")  # (vocab, d) — rows carry the const seeds
    sd: dict = {
        "model.embed_tokens.weight": _t(embed_table),
        "lm_head.weight": _t(take("lm_head")),  # untied (vocab, d)
    }

    for i in range(config.n_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_proj.weight"] = _t(take(f"l{i}_WQ").T)  # (hd, d)
        sd[f"{p}.self_attn.k_proj.weight"] = _t(take(f"l{i}_WK").T)
        sd[f"{p}.self_attn.v_proj.weight"] = _t(take(f"l{i}_WV").T)
        sd[f"{p}.self_attn.o_proj.weight"] = _t(take(f"l{i}_WO").T)  # (d, hd)
        sd[f"{p}.mlp.fc1.weight"] = _t(take(f"l{i}_W1").T)  # (d_h, d)
        sd[f"{p}.mlp.fc1.bias"] = _t(take(f"l{i}_b1"))
        sd[f"{p}.mlp.fc2.weight"] = _t(take(f"l{i}_W2").T)  # (d, d_h)
        sd[f"{p}.mlp.fc2.bias"] = _t(take(f"l{i}_b2"))
        # RMSNorm gains (Llama3 names): uniform (d,) = 2^m.  Present iff norm on.
        if config.rms_norm:
            sd[f"{p}.input_layernorm.weight"] = _t(take(f"l{i}_input_layernorm"))
            sd[f"{p}.post_attention_layernorm.weight"] = _t(
                take(f"l{i}_post_attention_layernorm")
            )

    if config.rms_norm:
        sd["model.norm.weight"] = _t(take("final_norm"))

    # RoPE constants are config-derived (the model rebuilds the rotation from
    # rope_base and d_rot), not state-dict weights — mark them consumed so the
    # all-mapped check passes.  rope_partial_split is emitted only for partial
    # rotary; the `in inits` guard skips it on full-rotary artifacts.
    for name in ("rope_freq", "rope_base", "rope_split", "rope_partial_split"):
        if name in inits:
            consumed.add(name)

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


# ---------------------------------------------------------------------------
# Phi-3 path (swish, bias=False, rms_norm — docs/phi3_conversion_plan.md)
# ---------------------------------------------------------------------------


def _per_layer_shapes(
    inits: Dict[str, np.ndarray],
) -> Tuple[int, int, List[int], List[int]]:
    """(n_layers, d_head, n_heads_per_layer, d_hidden_per_layer) from inits.

    Head counts come from the per-layer reshape const ``l{i}_qkv_view_shape``
    = ``[0, nh, d_head]``; the swish MLP hidden width is ``Wgate``'s column
    count.  Cross-checked against WQ's width so a drifted emission fails
    loudly.
    """
    n_layers = sum(1 for k in inits if re.fullmatch(r"l\d+_WQ", k))
    assert n_layers > 0, "no l{i}_WQ initializers — not a token transformer?"
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
        d_hidden_per_layer.append(int(inits[f"l{i}_Wgate"].shape[1]))
    return n_layers, d_head, n_heads_per_layer, d_hidden_per_layer


def build_phi3_config(
    inits: Dict[str, np.ndarray],
    meta: dict,
    *,
    bos_token: Optional[str] = None,
    eos_token: Optional[str] = None,
):
    """Derive a stock :class:`transformers.Phi3Config` from the artifact.

    Consumes initializer shapes + the sidecar meta only.  Non-uniform
    per-layer head counts / MLP widths become the uniform stock shape by
    taking the max; ``build_phi3_state_dict`` zero-pads each layer up to it.
    ``head_dim`` is set explicitly (P0(c): ``Phi3Config`` accepts, persists,
    and honors it — pinned by ``tests/hf/test_phi3_rope_parity.py``), so
    heads pad to the per-layer max instead of ``d / d_head``.
    """
    from transformers import Phi3Config

    _assert_meta_format(meta)
    target = _conversion_target(meta)
    assert target == "phi3", (
        f"build_phi3_config builds a Phi3Config but this artifact routes to "
        f"{target!r} — use convert_onnx_to_hf (which routes) or build_config"
    )

    vocab: List[str] = list(meta["vocab"])
    # vocab_size is the model's logit width = the embed table's row count,
    # which the compiler pads past len(vocab) (same reasoning as the native
    # config; padded rows are zero embeddings → zero logits).
    vocab_size, d = inits["embed_table"].shape
    assert (
        len(vocab) <= vocab_size
    ), f"meta vocab len {len(vocab)} exceeds embed_table rows {vocab_size}"

    n_layers, d_head, n_heads_per_layer, d_hidden_per_layer = _per_layer_shapes(inits)
    num_heads = max(n_heads_per_layer)
    intermediate_size = max(d_hidden_per_layer)
    assert num_heads > 0, "no attention heads on any layer"
    assert intermediate_size > 0, "no MLP hidden width on any layer"

    # RoPE: rope_theta and the partial-rotary width from the sidecar meta.
    # Phi-3's rotary embedding computes dim = int(head_dim * factor), so the
    # factor must reproduce d_rot exactly; d_head is a power of two in
    # practice, making d_rot/d_head exact in binary fp, but verify rather
    # than assume.
    rope_base = float(meta["rope_base"])
    d_rot = int(meta.get("d_rot", d_head))
    partial_rotary_factor = d_rot / d_head
    assert int(d_head * partial_rotary_factor) == d_rot, (
        f"partial_rotary_factor {partial_rotary_factor} does not reproduce "
        f"d_rot={d_rot} at head_dim={d_head} — Phi3RotaryEmbedding would "
        f"build a wrong-width frequency table"
    )

    # rms_norm is guaranteed on by the routing; cross-check the initializer
    # so a malformed pairing fails loudly (same guard as the native path).
    assert "final_norm" in inits, (
        "meta says rms_norm=True but the artifact has no final_norm "
        "initializer; the artifact is inconsistent — re-export"
    )
    eps = meta.get("rms_norm_eps")
    assert eps is not None, (
        "meta rms_norm=True but rms_norm_eps is missing — re-export with "
        "the current exporter"
    )

    return Phi3Config(
        vocab_size=int(vocab_size),
        hidden_size=int(d),
        intermediate_size=int(intermediate_size),
        num_hidden_layers=int(n_layers),
        num_attention_heads=int(num_heads),
        num_key_value_heads=int(num_heads),  # no GQA: every head owns its K/V
        head_dim=int(d_head),
        hidden_act="silu",
        max_position_embeddings=int(meta["cache_stride"]),
        rms_norm_eps=float(eps),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": rope_base,
            "partial_rotary_factor": partial_rotary_factor,
        },
        sliding_window=None,
        attention_dropout=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        use_cache=True,
        tie_word_embeddings=False,
        bos_token_id=_resolve_token_id(vocab, bos_token, "bos"),
        eos_token_id=_resolve_token_id(vocab, eos_token, "eos"),
        # Phi3Config defaults pad_token_id=32000, which is out of range for
        # these vocabs (nn.Embedding(padding_idx=...) would raise).
        pad_token_id=None,
    )


def _pad_zeros(arr: np.ndarray, target: int, axis: int) -> np.ndarray:
    """Zero-pad ``arr`` along ``axis`` up to ``target`` (no-op if already there)."""
    have = arr.shape[axis]
    if have == target:
        return arr
    assert have < target, f"cannot pad axis {axis} from {have} down to {target}"
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (0, target - have)
    return np.pad(arr, pad)


def build_phi3_state_dict(config, inits: Dict[str, np.ndarray]) -> Tuple[dict, set]:
    """Map initializers onto ``Phi3ForCausalLM``'s state dict.

    Returns ``(state_dict, consumed)`` like :func:`build_state_dict`.

    Two things happen beyond the transpose-and-concatenate mapping:

    - **The ``√d_head`` fold.**  The artifact's attention scores are raw
      ``Q·Kᵀ`` (every gain folded into the weights); Phi-3 multiplies logits
      by ``d_head^-0.5`` at runtime.  Scaling the Q rows by ``√d_head``
      cancels it up to one fp32 rounding per element (the product is formed
      in float64 and rounded once) plus the runtime's own rounded
      ``fl(d_head^-0.5)`` — a ~1e-7 relative logit perturbation, bounded
      against measured score gaps by the parity gate.

    - **Padding to uniform.**  Per-layer trimmed heads pad to
      ``config.num_attention_heads`` with all-zero Q/K/V rows and zero
      ``o_proj`` columns; MLP widths pad to ``config.intermediate_size``
      with all-zero gate/up rows and zero ``down_proj`` columns.  Every
      padded contribution is exactly zero: a zero head's K rows give
      identical logits for every key → uniform softmax over V rows that are
      exactly zero → head output exactly zero into zero o-columns; a zero
      gate/up lane contributes ``silu(0)·0 = 0`` and its down-column reads
      nothing.  (Whole-tensor bitwise equality against a hypothetical
      unpadded module is NOT guaranteed — a wider matmul can regroup the
      reduction of the *real* terms by a few ulps, kernel-dependent; that
      sits inside the parity bound.)  Cost is parameters and KV cache only.
    """
    consumed: set = set()

    def take(name: str) -> np.ndarray:
        consumed.add(name)
        return inits[name]

    d_head = int(config.head_dim)
    qkv_rows = config.num_attention_heads * d_head
    inter = int(config.intermediate_size)

    sd: dict = {
        "model.embed_tokens.weight": _t(take("embed_table")),  # (vocab, d)
        "lm_head.weight": _t(take("lm_head")),  # untied (vocab, d)
    }

    # One float64 √d_head, rounded per-element at the fp32 cast below.
    sqrt_dh = math.sqrt(float(d_head))

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        wq_t = take(f"l{i}_WQ").T  # (hd_i, d)
        wq_t = (wq_t.astype(np.float64) * sqrt_dh).astype(np.float32)
        qkv = np.concatenate(
            [
                _pad_zeros(wq_t, qkv_rows, axis=0),
                _pad_zeros(take(f"l{i}_WK").T, qkv_rows, axis=0),
                _pad_zeros(take(f"l{i}_WV").T, qkv_rows, axis=0),
            ],
            axis=0,
        )
        sd[f"{p}.self_attn.qkv_proj.weight"] = _t(qkv)  # (3·H·dh, d)
        sd[f"{p}.self_attn.o_proj.weight"] = _t(
            _pad_zeros(take(f"l{i}_WO").T, qkv_rows, axis=1)  # (d, H·dh)
        )
        gate_up = np.concatenate(
            [
                _pad_zeros(take(f"l{i}_Wgate").T, inter, axis=0),
                _pad_zeros(take(f"l{i}_Wup").T, inter, axis=0),
            ],
            axis=0,
        )
        sd[f"{p}.mlp.gate_up_proj.weight"] = _t(gate_up)  # (2·I, d)
        sd[f"{p}.mlp.down_proj.weight"] = _t(
            _pad_zeros(take(f"l{i}_Wdown").T, inter, axis=1)  # (d, I)
        )
        sd[f"{p}.input_layernorm.weight"] = _t(take(f"l{i}_input_layernorm"))
        sd[f"{p}.post_attention_layernorm.weight"] = _t(
            take(f"l{i}_post_attention_layernorm")
        )

    sd["model.norm.weight"] = _t(take("final_norm"))

    # RoPE constants are config-derived (Phi3RotaryEmbedding rebuilds the
    # rotation from rope_theta and partial_rotary_factor), not state-dict
    # weights — mark them consumed so the all-mapped check passes.
    for name in ("rope_freq", "rope_base", "rope_split", "rope_partial_split"):
        if name in inits:
            consumed.add(name)

    return sd, consumed


def build_fast_tokenizer(
    vocab: List[str],
    *,
    bos_token: Optional[str] = "<bos>",
    eos_token: Optional[str] = "<eos>",
    add_bos_token: bool = True,
):
    """A stock ``PreTrainedTokenizerFast`` over the artifact vocab.

    Reproduces ``TorchwrightTokenizer``'s semantics with zero custom code in
    the bundle: WordLevel over the vocab, character-level pre-tokenization
    (each char is one token; multi-char specials are split out first as
    added tokens), ``bos`` prepended on encode when ``add_bos_token``, and
    byte-exact decode (tokens fused with no separator).

    Unknown characters: if ``"<unk>"`` is in the vocab (the
    ``create_embedding`` tokenizer puts it at id 0) it is used as the
    WordLevel unk; otherwise encoding an out-of-vocab character raises —
    loud, where the slow tokenizer silently mapped to id 0.
    """
    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocab_dict = {t: i for i, t in enumerate(vocab)}
    for name, token in (("bos", bos_token), ("eos", eos_token)):
        if token is not None and token not in vocab_dict:
            raise ValueError(
                f"requested {name}_token {token!r} is not in the artifact "
                f"vocab; pass the correct {name}_token (or None)"
            )

    unk_token = "<unk>" if "<unk>" in vocab_dict else None
    # WordLevel always gets "<unk>" as its unk name: when the vocab carries
    # it, unknown characters map there; when it doesn't, the failed unk
    # lookup raises at encode time (the loud path — never silent).
    tok = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="<unk>"))
    # [\s\S] = any character including newline (the Oniguruma engine rejects
    # the (?s) inline flag); "isolated" splits every character into its own
    # token for the WordLevel lookup.
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(
        [t for t in (unk_token, bos_token, eos_token) if t is not None]
    )
    if add_bos_token and bos_token is not None:
        tok.post_processor = processors.TemplateProcessing(
            single=f"{bos_token} $A",
            pair=f"{bos_token} $A {bos_token} $B",
            special_tokens=[(bos_token, vocab_dict[bos_token])],
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )


def convert_onnx_to_hf(
    onnx_path: str,
    *,
    bos_token: Optional[str] = "<bos>",
    eos_token: Optional[str] = "<eos>",
):
    """Load an ONNX token artifact and return an in-memory HF model.

    Routes on the artifact's emission meta (:func:`_conversion_target`):
    relu → ``TorchwrightForCausalLM``, swish/bias=False/rms_norm → stock
    ``Phi3ForCausalLM``.  fp32, ``eval()`` mode.  The Phi-3 model is pinned
    to ``attn_implementation="eager"`` — the backend the parity claims are
    measured against (plain init would silently pick SDPA); the pin is not
    serialized, so bundle consumers still get their stock default.

    ``bos_token`` / ``eos_token`` are looked up in the vocab to populate the
    config's token ids, and must exist in it (a wrong token raises; pass
    ``None`` for a model with no bos/eos). The generic example vocab's
    control tokens are ``"<bos>"`` / ``"<eos>"`` (hence the defaults); DOOM
    passes ``begin`` / ``done``.
    """
    import torch

    model_proto = onnx.load(onnx_path)
    inits = _load_all_initializers(model_proto)
    meta = _read_meta(onnx_path)

    if _conversion_target(meta) == "phi3":
        from transformers import Phi3ForCausalLM

        config = build_phi3_config(
            inits, meta, bos_token=bos_token, eos_token=eos_token
        )
        sd, consumed = build_phi3_state_dict(config, inits)
        _assert_all_mapped(inits, consumed)
        model = Phi3ForCausalLM._from_config(config, attn_implementation="eager")
    else:
        from .modeling_torchwright import TorchwrightForCausalLM

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
    """Convert and ``save_pretrained`` a loadable bundle.

    Native (relu) artifacts get the full trust-remote-code bundle:
    safetensors + ``config.json`` + the shipped modeling/config files (via
    ``register_for_auto_class``), and — unless ``write_tokenizer`` is
    False — the generic ``TorchwrightTokenizer`` over the artifact vocab.

    Phi-3 (swish) artifacts get a fully stock bundle: safetensors +
    ``config.json`` (``architectures: ["Phi3ForCausalLM"]``, no ``auto_map``)
    and a ``tokenizer.json``-backed ``PreTrainedTokenizerFast`` — zero
    custom code, loadable with plain ``AutoModelForCausalLM`` /
    ``AutoTokenizer``.

    Returns the saved model.
    """
    meta = _read_meta(onnx_path)
    model = convert_onnx_to_hf(onnx_path, bos_token=bos_token, eos_token=eos_token)
    os.makedirs(save_dir, exist_ok=True)

    if _conversion_target(meta) == "phi3":
        model.save_pretrained(save_dir)
        if write_tokenizer:
            tok = build_fast_tokenizer(
                list(meta["vocab"]),
                bos_token=bos_token,
                eos_token=eos_token,
                add_bos_token=add_bos_token,
            )
            tok.save_pretrained(save_dir)
        return model

    from .configuration_torchwright import TorchwrightConfig
    from .modeling_torchwright import TorchwrightForCausalLM

    TorchwrightConfig.register_for_auto_class()
    TorchwrightForCausalLM.register_for_auto_class("AutoModelForCausalLM")
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
