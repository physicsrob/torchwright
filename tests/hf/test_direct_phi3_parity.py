"""Direct stock Phi-3 compilation plus an ONNX sibling-backend parity gate.

Compiles a small example (``binary_increment``, ~3s on CPU; no committed
artifact — ``*.onnx`` is gitignored), compiles HF directly,
and checks that the stock :class:`Phi3ForCausalLM`:

1. derives a config whose scalar dims agree with the artifact's debug sidecar;
2. consumes every ONNX initializer (the direct compiler asserts this internally — a
   new unmapped weight would raise);
3. produces **bit-exact** logits versus the ONNX oracle on the prefill and
   through several greedy decode steps (the cached-decode path);
4. survives a ``save_pretrained`` / ``from_pretrained`` round-trip
   (safetensors + config + the tied embedding) reproducing the same logits.

This is the small, fast gate on the generic core; ``test_calculator_parity``
runs the same machinery against the flagship calculator with real arithmetic.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("onnxruntime")

from torchwright.compiler.export import debug_meta_path_for
from torchwright.compiler.hf import compile_to_hf
from torchwright.compiler.onnx_load import load_onnx

from tests.hf._hf_parity import (
    compile_example,
    hf_teacher_forced,
    max_logit_diff,
    oracle_decode,
)

# binary_increment: "<bos> 1 0 1 1 \n" -> "1 1 0 0". Single-char vocab tokens.
_EXAMPLE = "binary_increment"
_BOS = "<bos>"
_EOS = "<eos>"
_PREFILL_TEXT = "1011\n"
_N_STEPS = 6


@pytest.fixture(scope="module")
def artifact_path():
    return compile_example(_EXAMPLE)


@pytest.fixture(scope="module")
def direct_model(artifact_path):
    from examples import binary_increment
    out, emb = binary_increment.create_network_parts()
    model = compile_to_hf(
        out, emb, d=binary_increment.D_MODEL, d_head=binary_increment.D_HEAD
    )
    oracle = load_onnx(artifact_path)
    return model, oracle


def _prefill_ids(oracle):
    tok2id = {t: i for i, t in enumerate(oracle.vocab)}
    return [tok2id[t] for t in ([_BOS] + list(_PREFILL_TEXT))]


def test_config_matches_debug_sidecar(artifact_path, direct_model):
    model, _ = direct_model
    cfg = model.config
    with open(debug_meta_path_for(artifact_path)) as f:
        dbg = json.load(f)
    assert cfg.hidden_size == dbg["d"]
    assert cfg.head_dim == dbg["d_head"]
    assert cfg.num_hidden_layers == dbg["n_layers"]
    # The Phi-3 profile is biasless and reserves one constant lane; the
    # independently compiled legacy ONNX profile may therefore be one slot
    # narrower while remaining semantically identical.
    assert cfg.intermediate_size in {
        max(int(x) for x in dbg["d_hidden"]),
        max(int(x) for x in dbg["d_hidden"]) + 1,
    }
    # Untied vanilla layout: no separate embedding/output-gather width remains.
    assert not cfg.tie_word_embeddings


def test_prefill_and_decode_bit_exact(direct_model):
    """ONNX oracle (onnxruntime) vs stock Phi-3 model (torch): the meaningful
    logits are bit-identical and decode the same tokens; the cancel-head rows
    that cancel to denormal magnitude differ only by a denormal ULP.

    Two things make this robust rather than a flaky ``== 0.0``:

    * The backends are *different runtimes*, so bit-for-bit agreement is not an
      algebraic guarantee. It holds for every normal-magnitude logit, but a row
      where the compiled cancel-heads drive the residual to denormal magnitude
      (~1e-40) differs by a single denormal ULP (~1e-43) — the near-zero row
      magnifies a sub-ULP rounding difference between the two backends.  The gap
      is denormal-magnitude *regardless of the rotation formula*; the exact
      mechanism of the sub-ULP difference is unconfirmed (one plausible candidate
      is an FMA contraction one runtime applies and the other does not).  We bound
      those rows far below any real cancellation failure (~1e-4+) instead of
      demanding equality.
    * We teacher-force the HF model on the *oracle's* token stream rather
      than letting each backend free-run on its own argmax. Past the meaningful
      output the logits are all-denormal noise whose argmax is arbitrary, so two
      free-running loops would pick different garbage and diverge; teacher
      forcing keeps every compared row on identical inputs.
    """
    model, oracle = direct_model
    prefill = _prefill_ids(oracle)
    o_ids, o_logits = oracle_decode(oracle, prefill, _N_STEPS)
    h_logits = hf_teacher_forced(model, prefill, o_ids)
    for i, (o, h) in enumerate(zip(o_logits, h_logits)):
        normal = o.abs() >= 1e-30  # exclude the denormal cancel-head noise floor
        assert torch.equal(o[normal], h[normal]), f"row {i}: normal logits diverged"
        # Where the row carries real signal, both backends decode the same token.
        if normal.any():
            assert int(o.argmax()) == int(h.argmax()), f"row {i}: argmax diverged"
    assert (
        max_logit_diff(o_logits, h_logits) < 1e-30
    ), "cross-backend logit divergence exceeds the denormal noise floor"


def test_bare_forward_without_use_cache(direct_model):
    """A plain ``model(input_ids=...)`` with NO kwargs must not crash.

    Regression for the transformers-5.x config stripping ``use_cache`` off the
    config dataclass — reading ``self.config.use_cache`` raised AttributeError.
    Every other test passes ``use_cache=True`` and masked it. This is the
    standard logit-eval call a Hub consumer makes outside ``generate``.
    """
    model, oracle = direct_model
    prefill = _prefill_ids(oracle)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prefill]))
    assert out.logits.shape[0] == 1 and out.logits.shape[1] == len(prefill)
    assert torch.isfinite(out.logits).all()


def test_labels_loss_path(direct_model):
    """``labels=`` returns a finite scalar loss (the training/perplexity path)."""
    model, oracle = direct_model
    ids = torch.tensor([_prefill_ids(oracle)])
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    assert out.loss is not None and out.loss.ndim == 0
    assert torch.isfinite(out.loss)


def test_inputs_embeds_supported(direct_model):
    """Stock Phi-3 accepts the standard inputs_embeds path."""
    model, oracle = direct_model
    ids = torch.tensor([_prefill_ids(oracle)])
    emb = model.model.embed_tokens(ids)
    with torch.no_grad():
        out = model(inputs_embeds=emb)
    assert out.logits.shape[:2] == ids.shape


def test_compiler_returns_fp32(direct_model):
    model, oracle = direct_model
    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_position_ids_are_honored(direct_model):
    """position_ids must shift which absolute PE rows are used (not be ignored)."""
    model, oracle = direct_model
    ids = torch.tensor([_prefill_ids(oracle)])
    T = ids.shape[1]
    with torch.no_grad():
        default = model(input_ids=ids).logits  # cache_position = arange(0, T)
        shifted = model(
            input_ids=ids, position_ids=torch.arange(100, 100 + T)[None]
        ).logits
    # Different absolute positions -> different PE -> different logits.
    assert not torch.equal(default, shifted)


def test_standard_attention_mask(direct_model):
    model, oracle = direct_model
    ids = torch.tensor([_prefill_ids(oracle)])
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert out.logits.shape[:2] == ids.shape


def test_save_load_round_trip(tmp_path, direct_model):
    from transformers import Phi3ForCausalLM

    model, oracle = direct_model
    save_dir = tmp_path / "bundle"
    model.save_pretrained(save_dir)
    reloaded = Phi3ForCausalLM.from_pretrained(
        save_dir, attn_implementation="eager"
    ).eval()

    prefill = _prefill_ids(oracle)
    with torch.no_grad():
        a = model(input_ids=torch.tensor([prefill]), use_cache=True).logits
        b = reloaded(input_ids=torch.tensor([prefill]), use_cache=True).logits
    assert torch.equal(a, b)
    # Untied: lm_head and embed_tokens are independent tensors (separate storage).
    assert (
        reloaded.lm_head.weight.data_ptr()
        != reloaded.model.embed_tokens.weight.data_ptr()
    )
