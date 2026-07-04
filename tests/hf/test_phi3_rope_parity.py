"""Phi-3 rotary parity probe + config constraint pins (phi3_conversion_plan P0).

The Phi-3 conversion target rests on one structural fact about the pinned
transformers version: ``Phi3ForCausalLM``'s modeling code **honors**
``partial_rotary_factor`` with exactly torchwright's rotation semantics —
it rotates the first ``rotary_dim`` dims of every head with half-split
pairing (dim ``i`` with ``i + rotary_dim/2``) and passes the tail through
untouched, character-for-character ``graph/rope.py::apply_rope``.  Stock
Llama *accepts* the factor into its config but ignores it in modeling code,
which is why Llama was rejected (see the plan's architecture sweep).

This file is the A0-pattern tripwire: a transformers upgrade that breaks or
removes Phi-3's partial-rotary path fails loudly here, and the recorded
fallback decision triggers (GLM/GLM4: honors the factor with the right body,
but pairs dims interleaved — costs an exact Q/K row permutation — and ships
attention biases by default).

Pinned here, at flagship geometry (``head_dim=128``, ``d_rot=64`` ⇒ factor
0.5) and at both bases in play (1e4 = the Phi-3 default, 5e5 = torchwright's
``ROPE_BASE``):

* the factor is honored (``inv_freq`` width from the *module*, not the
  config — config fields lie);
* ``inv_freq`` agrees with ``rope_inv_freq`` to ≤ 1 ulp (their table is fp32
  storage of one differently-rounded intermediate);
* the rotation is **semantically identical**: on shared cos/sin tables,
  ``apply_rotary_pos_emb`` and ``apply_rope`` are bit-exact;
* the NoPE tail passes through **bit-exactly** even on HF's own tables;
* the cos/sin tables diverge only by the 1-ulp ``inv_freq`` error propagated
  through the angle (grows linearly with position — bounded, not bit-exact);
* P0(c): ``Phi3Config`` accepts an explicit ``head_dim``, persists it
  through a save/load round trip, and the modeling code honors it — the
  finding that selects pad-to-per-layer-max over pad-to-``d/d_head`` in the
  converter — and ``tie_word_embeddings=False`` keeps ``lm_head`` untied.

CPU-only, no artifact compile — this is a pure modeling-code probe.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Phi3Config

from torchwright.graph.rope import apply_rope, rope_cos_sin, rope_inv_freq

D_HEAD = 128  # flagship head width (both e1m1 configs)
D_ROT = 64  # flagship partial-rotary width ⇒ factor 0.5
FACTOR = D_ROT / D_HEAD
BASES = [10_000.0, 500_000.0]  # Phi-3 default; torchwright ROPE_BASE


def _phi3_config(base: float, **overrides) -> Phi3Config:
    kwargs = dict(
        vocab_size=64,
        hidden_size=D_HEAD * 4,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=D_HEAD,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": base,
            "partial_rotary_factor": FACTOR,
        },
        max_position_embeddings=65536,
        tie_word_embeddings=False,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    )
    kwargs.update(overrides)
    return Phi3Config(**kwargs)


def _phi3_rope(base: float):
    from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding

    return Phi3RotaryEmbedding(_phi3_config(base))


def _qk(n_pos: int, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 4, n_pos, D_HEAD, generator=g)
    k = torch.randn(1, 4, n_pos, D_HEAD, generator=g)
    return q, k


# ---------------------------------------------------------------------------
# P0(a): the partial-rotary path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", BASES)
def test_partial_rotary_factor_is_honored(base):
    """THE tripwire: the built module's frequency table has the partial
    width ``d_rot/2``, not the full ``d_head/2``.  Stock Llama fails exactly
    this check (factor accepted by the config, full-width table built); if a
    transformers upgrade regresses Phi-3 the same way, the GLM fallback
    decision triggers (docs/phi3_conversion_plan.md, Risks)."""
    emb = _phi3_rope(base)
    assert emb.inv_freq.shape[0] == D_ROT // 2, (
        f"Phi3RotaryEmbedding built a {emb.inv_freq.shape[0] * 2}-dim rotation "
        f"where partial_rotary_factor={FACTOR} at head_dim={D_HEAD} requires "
        f"{D_ROT} — this transformers version no longer honors the factor; "
        f"the Phi-3 conversion target is broken (GLM is the recorded fallback)"
    )


@pytest.mark.parametrize("base", BASES)
def test_inv_freq_one_ulp_agreement(base):
    """Phi-3's ``inv_freq`` equals ``rope_inv_freq(d_rot, base)`` to ≤ 1 ulp.

    Both are fp32 storage of ``base^(-2p/d_rot)`` computed through one
    differently-rounded intermediate; the residual table difference is the
    only non-bit-exact ingredient of the conversion's rotation parity."""
    hf = _phi3_rope(base).inv_freq.float()
    tw = rope_inv_freq(D_ROT, base)
    assert hf.shape == tw.shape
    ulp = (hf.view(torch.int32) - tw.view(torch.int32)).abs().max().item()
    assert ulp <= 1, f"inv_freq diverged by {ulp} ulp (expected ≤ 1)"


@pytest.mark.parametrize("base", BASES)
def test_rotation_semantics_bit_exact_on_shared_tables(base):
    """On the SAME cos/sin tables, Phi-3's ``apply_rotary_pos_emb`` is
    bit-exact with ``graph/rope.py::apply_rope`` — pinning the split (first
    ``rotary_dim`` dims), the pairing (half-split: dim ``i`` with
    ``i + rotary_dim/2``), and the tail passthrough in one shot.  Any
    re-pairing (e.g. GLM's interleaved ``2p, 2p+1``) breaks this exactly."""
    from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb

    n_pos = 512
    q, k = _qk(n_pos)
    cos, sin = rope_cos_sin(torch.arange(n_pos), D_ROT, base)  # (P, d_rot)
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos[None], sin[None])
    q_tw = apply_rope(q, cos, sin)
    k_tw = apply_rope(k, cos, sin)
    assert torch.equal(q_hf, q_tw)
    assert torch.equal(k_hf, k_tw)


@pytest.mark.parametrize("base", BASES)
def test_nope_tail_bit_exact_through_hf_stack(base):
    """The unrotated tail ``[d_rot:d_head]`` passes through Phi-3's rotary
    application bit-exactly at every position, on HF's OWN tables — the
    property the engineered content-equality heads rely on (a converted
    checkpoint that rotated these dims would be silently wrong, not noisy)."""
    from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb

    n_pos = 512
    q, k = _qk(n_pos)
    emb = _phi3_rope(base)
    cos, sin = emb(q, torch.arange(n_pos)[None])
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.equal(q_hf[..., D_ROT:], q[..., D_ROT:])
    assert torch.equal(k_hf[..., D_ROT:], k[..., D_ROT:])


@pytest.mark.parametrize("base", BASES)
def test_cos_sin_tables_within_propagated_ulp_bound(base):
    """HF's cos/sin tables differ from ``rope_cos_sin`` only by the ≤ 1-ulp
    ``inv_freq`` error propagated through the angle: ``|Δcos| ≲ pos · θ ·
    2^-23`` (θ ≤ 1), so the divergence grows linearly with position and is
    *bounded*, never bit-exact.  This is rounding source (1) of the
    conversion's numerical budget (docs/phi3_conversion_plan.md)."""
    n_pos = 4096
    emb = _phi3_rope(base)
    x = torch.zeros(1, 1, n_pos, D_HEAD)
    cos_hf, sin_hf = emb(x, torch.arange(n_pos)[None])
    cos_tw, sin_tw = rope_cos_sin(torch.arange(n_pos), D_ROT, base)
    # Per-position bound: angle error ≤ pos · θ_max · 1 ulp, θ_max = 1 (plane
    # p=0 is base^0, exact in both).  Factor 2 for the cos evaluation's own
    # rounding at slightly different fp32 arguments.
    eps = 2.0**-23
    bound = 2.0 * eps * (torch.arange(n_pos, dtype=torch.float32) + 1.0)
    dcos = (cos_hf[0] - cos_tw).abs().max(dim=-1).values
    dsin = (sin_hf[0] - sin_tw).abs().max(dim=-1).values
    assert (dcos <= bound).all(), (dcos / bound).max()
    assert (dsin <= bound).all(), (dsin / bound).max()


# ---------------------------------------------------------------------------
# P0(c): config constraints
# ---------------------------------------------------------------------------


def test_head_dim_accepted_persisted_and_honored(tmp_path):
    """``Phi3Config`` accepts an explicit ``head_dim`` decoupled from
    ``hidden_size / num_attention_heads``, persists it through a save/load
    round trip, and the modeling code honors it (attention projections and
    the rotary table are sized from it).  This finding selects the cheaper
    pad-to-per-layer-max padding in the converter; if an upgrade drops the
    field, the converter must fall back to padding heads to ``d/d_head``."""
    from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

    cfg = _phi3_config(10_000.0, num_attention_heads=3)  # 3·128 ≠ hidden 512
    cfg.save_pretrained(tmp_path)
    reloaded = Phi3Config.from_pretrained(tmp_path)
    assert getattr(reloaded, "head_dim", None) == D_HEAD
    assert reloaded.rope_parameters["partial_rotary_factor"] == FACTOR

    model = Phi3ForCausalLM(reloaded)
    attn = model.model.layers[0].self_attn
    assert attn.head_dim == D_HEAD
    assert attn.qkv_proj.weight.shape == (3 * 3 * D_HEAD, cfg.hidden_size)
    assert attn.o_proj.weight.shape == (cfg.hidden_size, 3 * D_HEAD)
    assert model.model.rotary_emb.inv_freq.shape[0] == D_ROT // 2

    # A forward pass at the decoupled geometry must run.
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[0, 1, 2]]))
    assert out.logits.shape == (1, 3, cfg.vocab_size)


def test_untied_lm_head():
    """``tie_word_embeddings=False`` keeps ``lm_head`` a separate tensor —
    the artifact's unembed is genuinely untied from its embedding."""
    from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

    model = Phi3ForCausalLM(_phi3_config(10_000.0))
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
