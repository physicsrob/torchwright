"""Phi-3 converter round-trip: swish ONNX artifact → stock ``Phi3ForCausalLM``.

The Phi-3 mirror of ``test_convert.py`` (docs/phi3_conversion_plan.md P2).
Compiles the 1-digit adder — a swish, ``bias=False``, ``rms_norm`` example
with non-uniform per-layer head counts and MLP widths, so the padding path
is genuinely exercised — converts it with ``compiler/hf/convert``, and gates:

1. the derived config matches the artifact (dims, rope, untied head);
2. **bounded-relative logit parity + token-exact decode** against the
   ``load_onnx`` oracle, eager attention pinned.  NOT bit-exactness: the
   claim is weaker than the native converter's by design, from four
   non-removable rounding sources (rope tables, the ``√d_head`` fold, mask
   semantics, kernel choice) — see the plan's "Numerical implications";
3. the P0(b) score-path audit: additive ``finfo.min`` masking yields masked
   softmax weights of **exactly 0.0** at engineered logit magnitudes, both
   in isolation and through the converted model, and the measured logit
   perturbation sits under the derived bound with a recorded margin factor
   against the decisive logit gaps;
4. D6-scale repros for each mapping piece: the ``√d_head`` fold, head and
   MLP padding (bit-exact no-ops), the pinned-constant RMSNorm identity,
   and the ``tokenizer.json`` round trip;
5. a ``save_bundle`` output that is maximally ordinary: loads through stock
   ``AutoModelForCausalLM`` / ``AutoTokenizer`` with no custom code files
   and no ``auto_map``;
6. the partial-rotary path end-to-end (a ``d_rot < d_head`` swish artifact
   converts and still predicts by position).

CPU-only; both backends are deterministic there.
"""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

from torchwright.compiler.export import compile_to_onnx, debug_meta_path_for
from torchwright.compiler.hf.convert import (
    build_fast_tokenizer,
    convert_onnx_to_hf,
    save_bundle,
)
from torchwright.compiler.onnx_load import load_onnx
from torchwright.graph.rope import ROPE_BASE

from tests.hf._hf_parity import hf_teacher_forced, max_logit_diff, oracle_decode

_BOS = "<bos>"
_EOS = "<eos>"
_PROMPTS = ["1+2\n", "7+2\n", "0+0\n"]
_N_STEPS = 4

# Derived teacher-forced logit-parity bound, relative to the largest compared
# logit magnitude.  Sources (docs/phi3_conversion_plan.md "Numerical
# implications"): (1) rope tables fp32-rounding-equal within 1 ulp, reaching
# scores as ~pos·θ·2^-23; (2) the √d_head fold — one fp32 rounding per WQ
# element plus the runtime's fl(d_head^-0.5) — ~2·2^-24 relative per score.
# Each is ~1e-7 relative; the bound allows 100× for accumulation through
# depth and the unembed's cancellation.  Measured at pin time: the adder
# fixture is BIT-EXACT (max |Δlogit| = 0.0 — its attention weights and swish
# lanes saturate, so the Q-side perturbations never reach a value path); an
# unsaturated random-weight swish graph measures ~2e-7 relative.  The bound
# covers the unsaturated case.
_LOGIT_REL_BOUND = 1e-5

# The decisive logit gap (oracle top-1 minus top-2 on rows with signal) must
# dominate the observed cross-backend perturbation by at least this factor —
# the P0(b) margin gate protecting token-exactness.  Measured at pin time:
# min gap 400 over a perturbation of exactly 0 (unbounded margin; the 1e-30
# guard below keeps the ratio finite).
_MIN_MARGIN_FACTOR = 1e3


def _compile_adder1(tmpdir) -> str:
    """The 1-digit adder as a swish/bias=False/rms_norm token artifact."""
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        output_node, embedding = adder_module.create_network_parts()
    finally:
        adder_module.max_digits = original
    path = os.path.join(str(tmpdir), "adder1.onnx")
    compile_to_onnx(
        output_node,
        embedding,
        path,
        d=adder_module.D_MODEL,
        d_head=adder_module.D_HEAD,
        max_seq_len=64,
        bias=False,
    )
    return path


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory):
    return _compile_adder1(tmp_path_factory.mktemp("phi3_adder1"))


@pytest.fixture(scope="module")
def converted(artifact_path):
    model = convert_onnx_to_hf(artifact_path, bos_token=_BOS, eos_token=_EOS)
    oracle = load_onnx(artifact_path)
    return model, oracle


def _prefill_ids(oracle, text: str):
    tok2id = {t: i for i, t in enumerate(oracle.vocab)}
    return [tok2id[t] for t in ([_BOS] + list(text))]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_matches_artifact(artifact_path, converted):
    model, oracle = converted
    cfg = model.config
    assert type(model).__name__ == "Phi3ForCausalLM"
    with open(debug_meta_path_for(artifact_path)) as f:
        dbg = json.load(f)
    assert cfg.hidden_size == dbg["d"]
    assert cfg.head_dim == dbg["d_head"]
    assert cfg.num_hidden_layers == dbg["n_layers"]
    # Uniform stock widths = the per-layer maxima (padding fills the rest).
    assert cfg.intermediate_size == max(int(x) for x in dbg["d_hidden"])
    assert cfg.num_attention_heads == max(oracle.per_layer_n_heads)
    assert cfg.num_key_value_heads == cfg.num_attention_heads
    # vocab_size is the logit width (embed rows), padded past the token count.
    assert cfg.vocab_size >= len(oracle.vocab)
    assert cfg.max_position_embeddings == oracle.cache_stride
    assert cfg.rope_parameters["rope_theta"] == ROPE_BASE
    assert cfg.rope_parameters["partial_rotary_factor"] == 1.0  # full rotary
    assert not cfg.tie_word_embeddings
    assert cfg.pad_token_id is None
    # The parity claims below are measured against eager attention.
    assert model.config._attn_implementation == "eager"


# ---------------------------------------------------------------------------
# P0(b): score-path audit — mask semantics and perturbation margin
# ---------------------------------------------------------------------------


def test_masked_softmax_weights_exactly_zero_at_engineered_magnitudes():
    """Phi-3 masks *additively* with ``finfo.min`` where the artifact
    *overwrites* with its sentinel.  The delta is harmless iff masked
    weights are still exactly 0.0 with real logits at the engineered
    magnitudes (~1e5–1e6, either sign): ``exp(finfo.min - rowmax)``
    underflows to exact zero as long as the row max is a real logit.
    Verified, not assumed (the plan's item 3)."""
    fmin = torch.finfo(torch.float32).min
    for mag in (1e5, 1e6, 3e6):
        for sign in (1.0, -1.0):
            logits = torch.tensor([sign * mag, sign * mag - 30.0, 500.0, -2e6])
            mask = torch.tensor([0.0, 0.0, fmin, fmin])
            w = torch.softmax(logits + mask, dim=-1)
            assert (w[2:] == 0.0).all(), (mag, sign, w)
            # The unmasked pair still resolves: top weight saturated.
            assert w[0] == 1.0, (mag, sign, w)


def test_converted_model_masked_attention_weights_exactly_zero(converted):
    """Through the real converted model (eager path, real engineered
    logits): every future-position attention weight is exactly 0.0."""
    model, oracle = converted
    ids = torch.tensor([_prefill_ids(oracle, _PROMPTS[0])])
    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True)
    n = ids.shape[1]
    future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    for li, weights in enumerate(out.attentions):
        masked = weights[0][:, future]
        assert (masked == 0.0).all(), f"layer {li}: nonzero masked weight"


def test_logit_perturbation_margin_factor(converted):
    """The P0(b) gate: the decisive logit gap dominates the measured
    cross-backend perturbation by ≥ ``_MIN_MARGIN_FACTOR``.  The gap is the
    oracle's top-1 − top-2 logit on rows with signal (what greedy decode
    actually resolves); the perturbation is the teacher-forced max |Δlogit|
    across all prompts."""
    model, oracle = converted
    min_gap = float("inf")
    max_diff = 0.0
    for text in _PROMPTS:
        prefill = _prefill_ids(oracle, text)
        o_ids, o_logits = oracle_decode(oracle, prefill, _N_STEPS)
        h_logits = hf_teacher_forced(model, prefill, o_ids)
        max_diff = max(max_diff, max_logit_diff(o_logits, h_logits))
        for row in o_logits:
            if (row.abs() >= 1e-30).any():  # skip denormal cancel-head rows
                top2 = torch.topk(row, 2).values
                min_gap = min(min_gap, float(top2[0] - top2[1]))
    margin = min_gap / max(max_diff, 1e-30)
    assert margin >= _MIN_MARGIN_FACTOR, (
        f"decisive gap {min_gap:.3e} over perturbation {max_diff:.3e} gives "
        f"margin {margin:.1f} < {_MIN_MARGIN_FACTOR:g}: token-exactness is "
        f"no longer comfortably protected — investigate before shipping"
    )


# ---------------------------------------------------------------------------
# Parity gate: bounded-relative logits, token-exact decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", _PROMPTS)
def test_prefill_and_decode_parity(converted, text):
    """Teacher-forced on the oracle's token stream (see ``test_convert`` for
    why free-running comparison is unsound past the meaningful output):
    every logit row agrees within the derived relative bound, and every row
    with signal decodes the same token."""
    model, oracle = converted
    prefill = _prefill_ids(oracle, text)
    o_ids, o_logits = oracle_decode(oracle, prefill, _N_STEPS)
    h_logits = hf_teacher_forced(model, prefill, o_ids)

    scale = max(row.abs().max().item() for row in o_logits)
    diff = max_logit_diff(o_logits, h_logits)
    assert diff <= _LOGIT_REL_BOUND * scale, (
        f"max |Δlogit| {diff:.3e} exceeds {_LOGIT_REL_BOUND:g} × max|logit| "
        f"{scale:.3e} — beyond the derived rope-table + √d_head-fold budget"
    )
    for i, (o, h) in enumerate(zip(o_logits, h_logits)):
        normal = o.abs() >= 1e-30
        if normal.any():
            assert int(o.argmax()) == int(h.argmax()), f"row {i}: argmax diverged"


@pytest.mark.parametrize("text", [("1+2\n", "3"), ("7+2\n", "9")])
def test_greedy_decode_token_exact(converted, text):
    """Free-running greedy decode agrees token-for-token through <eos>
    (after <eos> both streams are denormal garbage — not compared)."""
    from tests.hf._hf_parity import hf_decode

    prompt, expected = text
    model, oracle = converted
    prefill = _prefill_ids(oracle, prompt)
    o_ids, _ = oracle_decode(oracle, prefill, _N_STEPS)
    h_ids, _ = hf_decode(model, prefill, _N_STEPS)
    eos_id = oracle.vocab.index(_EOS)
    n = o_ids.index(eos_id) + 1 if eos_id in o_ids else len(o_ids)
    assert h_ids[:n] == o_ids[:n]
    assert "".join(oracle.vocab[i] for i in o_ids[: n - 1]) == expected


def test_sdpa_decodes_same_tokens(converted, tmp_path):
    """SDPA consumers get token-level claims only (the parity bound is
    measured on eager) — pin that the tokens do hold."""
    from transformers import AutoModelForCausalLM

    model, oracle = converted
    model.save_pretrained(tmp_path / "m")
    sdpa = AutoModelForCausalLM.from_pretrained(
        tmp_path / "m", attn_implementation="sdpa"
    ).eval()
    from tests.hf._hf_parity import hf_decode

    prefill = _prefill_ids(oracle, "1+2\n")
    o_ids, _ = oracle_decode(oracle, prefill, _N_STEPS)
    s_ids, _ = hf_decode(sdpa, prefill, _N_STEPS)
    eos_id = oracle.vocab.index(_EOS)
    n = o_ids.index(eos_id) + 1 if eos_id in o_ids else len(o_ids)
    assert s_ids[:n] == o_ids[:n]


# ---------------------------------------------------------------------------
# D6 repros: one per mapping piece
# ---------------------------------------------------------------------------


def test_fold_round_trip_error_bound():
    """The √d_head fold cancels Phi-3's runtime ``d_head^-0.5`` up to a few
    fp32 roundings.  Each folded weight is ``w·√d_head`` perturbed by ≤ 1
    ulp, so a dot product's error is bounded by the *absolute-value* dot
    times a few ulps — NOT by a relative tolerance on the result, which
    cancellation can inflate arbitrarily.  The elementwise bound
    ``(|x|·|w| + |q|)·4·2^-24`` is the derived form of rounding source (2)
    in the plan's numerical-implications section."""
    d_head = 128
    g = torch.Generator().manual_seed(11)
    w = torch.randn(64, 32, generator=g)
    x = torch.randn(8, 32, generator=g)
    folded = (w.double() * (d_head**0.5)).float()
    q_folded = (x @ folded.T) * torch.tensor(d_head**-0.5, dtype=torch.float32)
    q_raw = x @ w.T
    bound = (x.abs() @ w.abs().T + q_raw.abs()) * (4 * 2.0**-24)
    assert ((q_folded - q_raw).abs() <= bound).all(), (
        (q_folded - q_raw).abs() / bound
    ).max()


def _tiny_phi3_config(n_heads: int, d: int = 64, d_head: int = 16):
    from transformers import Phi3Config

    cfg = Phi3Config(
        vocab_size=16,
        hidden_size=d,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        head_dim=d_head,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e4,
            "partial_rotary_factor": 0.5,
        },
        max_position_embeddings=64,
        tie_word_embeddings=False,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_head_padding_contributes_exactly_zero():
    """The head-padding no-op, pinned at the level IEEE actually
    guarantees.  (Whole-tensor bitwise equality between a 2-head and a
    padded 5-head module does NOT hold on real BLAS — a wider matmul may
    regroup the reduction of the real terms by an ulp — so the claim is
    exact-zero *contributions*, plus a regrouping-only bound.)

    (a) all-zero Q/K/V rows produce exactly-zero padded projections (a dot
        against a zero vector is exact zero);
    (b) the zero o_proj columns read nothing: replacing the padded heads'
        Q/K/V rows with large-but-finite garbage — so the padded heads
        compute arbitrary nonzero contexts — leaves the output bit-for-bit
        identical (same shapes → same kernels, so any change could only
        come through the o-columns);
    (c) the padded module agrees with the 2-head module to reduction-
        regrouping ulps."""
    from transformers.models.phi3.modeling_phi3 import (
        Phi3Attention,
        Phi3RotaryEmbedding,
    )

    d, d_head, real_h, padded_h = 64, 16, 2, 5
    hd_r, hd_p = real_h * d_head, padded_h * d_head
    g = torch.Generator().manual_seed(5)
    wq = torch.randn(hd_r, d, generator=g)
    wk = torch.randn(hd_r, d, generator=g)
    wv = torch.randn(hd_r, d, generator=g)
    wo = torch.randn(d, hd_r, generator=g)

    def pad_rows(m, fill=0.0):
        return torch.cat([m, torch.full((hd_p - m.shape[0], d), fill)], 0)

    def build(n_heads, qkv, o):
        attn = Phi3Attention(
            _tiny_phi3_config(n_heads, d=d, d_head=d_head), layer_idx=0
        ).eval()
        with torch.no_grad():
            attn.qkv_proj.weight.copy_(qkv)
            attn.o_proj.weight.copy_(o)
        return attn

    o_padded = torch.cat([wo, torch.zeros(d, hd_p - hd_r)], 1)
    attn_r = build(real_h, torch.cat([wq, wk, wv], 0), wo)
    attn_p = build(
        padded_h, torch.cat([pad_rows(wq), pad_rows(wk), pad_rows(wv)], 0), o_padded
    )
    attn_g = build(
        padded_h,
        torch.cat([pad_rows(wq, 1e3), pad_rows(wk, 1e3), pad_rows(wv, 1e3)], 0),
        o_padded,
    )

    x = torch.randn(1, 6, d, generator=g)
    pos = torch.arange(6)[None]
    cfg_p = attn_p.config

    with torch.no_grad():
        # (a) zero rows -> exactly-zero padded projections.
        qkv_act = attn_p.qkv_proj(x)
        for base in (0, hd_p, 2 * hd_p):  # Q, K, V blocks
            assert (qkv_act[..., base + hd_r : base + hd_p] == 0.0).all()

        cos_sin = Phi3RotaryEmbedding(cfg_p)(x, pos)
        out_p = attn_p(x, position_embeddings=cos_sin, attention_mask=None)[0]
        # (b) garbage in the padded heads changes nothing downstream.
        out_g = attn_g(x, position_embeddings=cos_sin, attention_mask=None)[0]
        assert torch.equal(out_p, out_g)
        # (c) real-vs-padded: identical sums, possibly regrouped.
        out_r = attn_r(x, position_embeddings=cos_sin, attention_mask=None)[0]
        assert torch.allclose(out_r, out_p, rtol=1e-6, atol=1e-4)


def test_mlp_padding_contributes_exactly_zero():
    """The MLP-padding no-op, pinned like the head padding above:

    (a) zero gate/up rows give exactly-zero padded hidden lanes
        (``silu(0)·0 = 0``, and the zero-weight dots are exact zeros);
    (b) the zero down_proj columns read nothing: garbage in the padded
        gate/up rows — arbitrary nonzero hidden lanes — leaves the output
        bit-for-bit identical at identical shapes;
    (c) the padded module agrees with the unpadded one to reduction-
        regrouping ulps (bitwise equality is kernel-dependent)."""
    from transformers.models.phi3.modeling_phi3 import Phi3MLP

    d, real_i, padded_i = 64, 5, 8
    g = torch.Generator().manual_seed(9)
    gate = torch.randn(real_i, d, generator=g)
    up = torch.randn(real_i, d, generator=g)
    down = torch.randn(d, real_i, generator=g)

    def build(inter, gate_up, down_w):
        cfg = _tiny_phi3_config(2, d=d)
        cfg.intermediate_size = inter
        mlp = Phi3MLP(cfg).eval()
        with torch.no_grad():
            mlp.gate_up_proj.weight.copy_(gate_up)
            mlp.down_proj.weight.copy_(down_w)
        return mlp

    def pad_rows(m, fill=0.0):
        return torch.cat([m, torch.full((padded_i - m.shape[0], d), fill)], 0)

    down_padded = torch.cat([down, torch.zeros(d, padded_i - real_i)], 1)
    mlp_r = build(real_i, torch.cat([gate, up], 0), down)
    mlp_p = build(padded_i, torch.cat([pad_rows(gate), pad_rows(up)], 0), down_padded)
    mlp_g = build(
        padded_i, torch.cat([pad_rows(gate, 1e3), pad_rows(up, 1e3)], 0), down_padded
    )

    x = torch.randn(3, d, generator=g)
    with torch.no_grad():
        # (a) padded hidden lanes are exactly zero.
        gu = mlp_p.gate_up_proj(x)
        g_act, u_act = gu.chunk(2, dim=-1)
        hidden = u_act * torch.nn.functional.silu(g_act)
        assert (hidden[..., real_i:] == 0.0).all()
        # (b) garbage lanes change nothing through the zero down-columns.
        assert torch.equal(mlp_p(x), mlp_g(x))
        # (c) real-vs-padded: identical sums, possibly regrouped.
        assert torch.allclose(mlp_r(x), mlp_p(x), rtol=1e-6, atol=1e-4)


def test_rms_norm_is_bitexact_identity_on_real_rows(converted):
    """The pinned-constant RMSNorm survives Phi3RMSNorm bit-exactly: on real
    embedded rows (which carry the pinned constant) every norm in the
    converted model returns its input unchanged — the forced rms is an exact
    power of two and the uniform gain cancels it."""
    model, oracle = converted
    ids = torch.tensor([_prefill_ids(oracle, "1+2\n")])
    with torch.no_grad():
        rows = model.model.embed_tokens(ids)
        l0 = model.model.layers[0]
        assert torch.equal(l0.input_layernorm(rows), rows)
        assert torch.equal(model.model.norm(rows), rows)


def test_fast_tokenizer_round_trip(converted, tmp_path):
    """The tokenizer.json bundle reproduces TorchwrightTokenizer semantics
    with zero custom code: char-level encode with bos prepended, multi-char
    specials split out, byte-exact decode, save/load through the stock
    PreTrainedTokenizerFast."""
    from transformers import PreTrainedTokenizerFast

    _, oracle = converted
    vocab = list(oracle.vocab)
    tok = build_fast_tokenizer(vocab, bos_token=_BOS, eos_token=_EOS)

    text = "12+34\n"
    ids = tok(text)["input_ids"]
    assert ids == [vocab.index(_BOS)] + [vocab.index(c) for c in text]
    assert tok.decode(ids, skip_special_tokens=True) == text
    # Multi-char specials are split out before the character pass.
    assert tok("1<eos>2")["input_ids"] == [
        vocab.index(_BOS),
        vocab.index("1"),
        vocab.index(_EOS),
        vocab.index("2"),
    ]
    # The artifact vocab carries "<unk>" (create_embedding puts it at id 0):
    # an out-of-vocab character maps there, matching the oracle's fallback.
    assert "?" not in vocab
    assert tok("1?2")["input_ids"] == [
        vocab.index(_BOS),
        vocab.index("1"),
        vocab.index("<unk>"),
        vocab.index("2"),
    ]
    # A vocab WITHOUT "<unk>" fails loudly on unknown characters instead of
    # silently mapping somewhere.
    no_unk = build_fast_tokenizer(
        ["<bos>", "<eos>", "0", "1"], bos_token=_BOS, eos_token=_EOS
    )
    with pytest.raises(Exception, match="(?i)unk"):
        no_unk("1?0")

    # add_bos_token=False drops the prefix.
    no_bos = build_fast_tokenizer(
        vocab, bos_token=_BOS, eos_token=_EOS, add_bos_token=False
    )
    assert no_bos(text)["input_ids"] == [vocab.index(c) for c in text]

    # Stock save/load round trip.
    tok.save_pretrained(tmp_path)
    reloaded = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    assert reloaded(text)["input_ids"] == ids
    assert reloaded.decode(ids, skip_special_tokens=True) == text


# ---------------------------------------------------------------------------
# The bundle: fully stock, no custom code
# ---------------------------------------------------------------------------


def test_save_load_round_trip_bit_exact(converted, tmp_path):
    """save_pretrained → from_pretrained (same eager backend) reproduces the
    in-memory model's logits bit-exactly, with the head still untied."""
    from transformers import AutoModelForCausalLM, Phi3ForCausalLM

    model, oracle = converted
    save_dir = tmp_path / "bundle"
    model.save_pretrained(save_dir)
    reloaded = AutoModelForCausalLM.from_pretrained(
        save_dir, attn_implementation="eager"
    ).eval()
    assert isinstance(reloaded, Phi3ForCausalLM)

    prefill = _prefill_ids(oracle, "1+2\n")
    with torch.no_grad():
        a = model(input_ids=torch.tensor([prefill])).logits
        b = reloaded(input_ids=torch.tensor([prefill])).logits
    assert torch.equal(a, b)
    assert (
        reloaded.lm_head.weight.data_ptr()
        != reloaded.model.embed_tokens.weight.data_ptr()
    )


def test_save_bundle_is_fully_stock(artifact_path, tmp_path):
    """The published bundle carries no custom code at all: no .py files, no
    auto_map, stock architectures; AutoTokenizer + AutoModelForCausalLM +
    generate produce the right answer end-to-end."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Phi3ForCausalLM,
        PreTrainedTokenizerFast,
    )

    save_dir = tmp_path / "bundle"
    save_bundle(artifact_path, str(save_dir), bos_token=_BOS, eos_token=_EOS)

    files = os.listdir(save_dir)
    assert not [f for f in files if f.endswith(".py")], files
    with open(save_dir / "config.json") as f:
        cfg = json.load(f)
    assert cfg["architectures"] == ["Phi3ForCausalLM"]
    assert "auto_map" not in cfg

    tok = AutoTokenizer.from_pretrained(save_dir)
    assert type(tok) is PreTrainedTokenizerFast
    model = AutoModelForCausalLM.from_pretrained(
        save_dir, attn_implementation="eager"
    ).eval()
    assert isinstance(model, Phi3ForCausalLM)

    ids = tok("1+2\n", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=4, do_sample=False)
    answer = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
    assert answer == "3"


# ---------------------------------------------------------------------------
# Routing refusals
# ---------------------------------------------------------------------------


def _tiny_swish_graph():
    """A 2-FFN swish token graph (the test_no_bias_onnx fixture shape)."""
    from torchwright.graph import FFN
    from torchwright.ops.inout_nodes import create_embedding

    emb = create_embedding(vocab=list("01+") + ["\n", _BOS, _EOS, "default"])
    d = len(emb)
    g = torch.Generator().manual_seed(23)
    out = FFN(
        emb,
        gate_proj=torch.randn(24, d, generator=g) * 0.2,
        gate_bias=torch.randn(24, generator=g) * 0.1,
        out_proj=torch.randn(24, d, generator=g) * 0.2,
        out_bias=torch.randn(d, generator=g) * 0.1,
        up_proj=torch.randn(24, d, generator=g) * 0.2,
        up_bias=torch.randn(24, generator=g) * 0.1,
        activation="swish",
        name="gated",
    )
    return out, emb


def test_swish_biased_artifact_refused(tmp_path):
    """swish + bias=True has no stock target (Phi-3 is biasless): refuse."""
    out, emb = _tiny_swish_graph()
    path = str(tmp_path / "swish_biased.onnx")
    compile_to_onnx(out, emb, path, d=256, d_head=16, max_seq_len=16, bias=True)
    with pytest.raises(NotImplementedError, match="bias=True"):
        convert_onnx_to_hf(path, bos_token=_BOS, eos_token=_EOS)


def test_swish_unnormed_artifact_refused(tmp_path):
    """swish + bias=False without the RMSNorm cannot be a Phi3 checkpoint
    (the architecture always norms): refuse."""
    out, emb = _tiny_swish_graph()
    path = str(tmp_path / "swish_unnormed.onnx")
    compile_to_onnx(
        out, emb, path, d=256, d_head=16, max_seq_len=16, bias=False, rms_norm=False
    )
    with pytest.raises(NotImplementedError, match="rms_norm"):
        convert_onnx_to_hf(path, bos_token=_BOS, eos_token=_EOS)


# ---------------------------------------------------------------------------
# Partial rotary end-to-end
# ---------------------------------------------------------------------------

_PVOCAB = [_BOS, _EOS, "a", "b", "c", "d", "e"]
_P_D_HEAD = 16
_P_D_ROT = 8


def _build_partial_swish():
    """Predict the previous token via a partial-rotary offset head, routed
    through a swiglu identity map so the artifact is the swish machine."""
    from torchwright.graph.rope import rotary_offset_head
    from torchwright.ops.inout_nodes import create_embedding
    from torchwright.ops.swiglu.map_select import map_to_table

    emb = create_embedding(vocab=_PVOCAB)
    prev = rotary_offset_head(emb, delta_pos=-1, d_qk=_P_D_HEAD, d_rot=_P_D_ROT)
    out = map_to_table(
        prev,
        {emb.get_embedding(t): emb.get_embedding(t) for t in _PVOCAB},
        default=emb.get_embedding(_EOS),
    )
    return out, emb


def test_partial_rotary_swish_converts_and_predicts(tmp_path):
    out, emb = _build_partial_swish()
    path = str(tmp_path / "partial_swish.onnx")
    compile_to_onnx(out, emb, path, d=256, d_head=_P_D_HEAD, max_seq_len=64, bias=False)
    model = convert_onnx_to_hf(path, bos_token=_BOS, eos_token=_EOS)
    oracle = load_onnx(path)
    assert type(model).__name__ == "Phi3ForCausalLM"
    assert model.config.rope_parameters["partial_rotary_factor"] == (
        _P_D_ROT / _P_D_HEAD
    )
    # The factor must be honored by the built model, not just the config.
    assert model.model.rotary_emb.inv_freq.shape[0] == _P_D_ROT // 2

    # Token ids come from the artifact's own vocab list — the
    # create_embedding tokenizer prepends "<unk>" and assigns ids itself,
    # so positional indexing into the construction-time list is wrong.
    vocab = oracle.vocab
    seq = [_BOS, "a", "b", "c", "d", "e"]
    ids = torch.tensor([[vocab.index(t) for t in seq]])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    pred = [vocab[i] for i in logits.argmax(-1).tolist()]
    assert pred[1:] == seq[:-1], pred

    # And the oracle agrees within the parity bound on the prefill.
    o_logits, _ = oracle.step(ids[0].to(torch.int64), oracle.empty_past())
    diff = (o_logits - logits).abs().max().item()
    scale = o_logits.abs().max().item()
    assert diff <= _LOGIT_REL_BOUND * scale, (diff, scale)
