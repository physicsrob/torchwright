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
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("onnxruntime")

from tests.hf._hf_parity import (
    compile_example,
    hf_teacher_forced,
    max_logit_diff,
    oracle_decode,
)
from torchwright.compiler.export import debug_meta_path_for
from torchwright.compiler.hf import compile_to_hf
from torchwright.compiler.onnx_load import load_onnx

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
    return [tok2id[t] for t in ([_BOS, *list(_PREFILL_TEXT)])]


def test_custom_config_rejects_explicit_untied():
    """An explicit ``tie_word_embeddings=False`` must raise, not be silently overridden.

    This covers a pre-v6 saved config, or a deliberate untied
    experiment. HF's ``tie_weights()`` would otherwise clone the
    embedding over a checkpoint's real lm_head and return wrong logits
    with no error. (The stock Phi3 target ties through its ordinary
    ``tie_word_embeddings`` flag instead — no custom guard needed
    there.)
    """
    from torchwright.compiler.hf.configuration_torchwright_custom import (
        TorchwrightCustomConfig,
    )

    with pytest.raises(ValueError, match="tie_word_embeddings"):
        TorchwrightCustomConfig(tie_word_embeddings=False)
    assert TorchwrightCustomConfig().tie_word_embeddings
    assert TorchwrightCustomConfig(tie_word_embeddings=True).tie_word_embeddings


def test_config_matches_debug_sidecar(artifact_path, direct_model):
    model, _ = direct_model
    cfg = model.config
    with Path(debug_meta_path_for(artifact_path)).open() as f:
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
    # token.v6 tied layout: one token table serves lookup and readout.
    assert cfg.tie_word_embeddings


def test_artifact_uses_one_physically_tied_table(artifact_path):
    import onnx

    proto = onnx.load(artifact_path)
    names = {x.name for x in proto.graph.initializer}
    names |= {x.values.name for x in proto.graph.sparse_initializer}
    assert "embed_table" in names
    assert "lm_head" not in names
    transpose_inputs = [
        node.input[0]
        for node in proto.graph.node
        if node.op_type == "Transpose" and list(node.output) == ["_embed_table_T"]
    ]
    assert transpose_inputs == ["embed_table"]


def test_prefill_and_decode_bit_exact(direct_model):
    """ONNX oracle (onnxruntime) vs stock Phi-3 (torch): logits are bit-identical.

    They decode the same tokens; the cancel-head rows that cancel to
    denormal magnitude differ only by a denormal ULP.

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
    for i, (o, h) in enumerate(zip(o_logits, h_logits, strict=False)):
        normal = o.abs() >= 1e-30  # exclude the denormal cancel-head noise floor
        assert torch.equal(o[normal], h[normal]), f"row {i}: normal logits diverged"
        # Where the row carries real signal, both backends decode the same token.
        if normal.any():
            assert int(o.argmax()) == int(h.argmax()), f"row {i}: argmax diverged"
    assert max_logit_diff(o_logits, h_logits) < 1e-30, (
        "cross-backend logit divergence exceeds the denormal noise floor"
    )


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
    assert out.logits.shape[0] == 1
    assert out.logits.shape[1] == len(prefill)
    assert torch.isfinite(out.logits).all()


def test_labels_loss_path(direct_model):
    """``labels=`` returns a finite scalar loss (the training/perplexity path)."""
    model, oracle = direct_model
    ids = torch.tensor([_prefill_ids(oracle)])
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    assert out.loss is not None
    assert out.loss.ndim == 0
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
    model, _oracle = direct_model
    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_position_ids_are_honored(direct_model):
    """Non-uniform position IDs must change relative rotary offsets.

    A uniform shift cannot test this: RoPE attention is translation-invariant,
    so adding the same constant to every position preserves every relative
    offset. Stretching the positions changes those offsets and therefore must
    change this position-sensitive compiled model's logits.
    """
    model, oracle = direct_model
    ids = torch.tensor([_prefill_ids(oracle)])
    T = ids.shape[1]
    with torch.no_grad():
        default = model(input_ids=ids).logits  # cache_position = arange(0, T)
        stretched = model(
            input_ids=ids, position_ids=(2 * torch.arange(T))[None]
        ).logits
    assert not torch.equal(default, stretched)


def test_standard_attention_mask(direct_model):
    model, oracle = direct_model
    ids = torch.tensor([_prefill_ids(oracle)])
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert out.logits.shape[:2] == ids.shape


def test_direct_model_is_storage_tied(direct_model):
    """The compiled model's lm_head and embed_tokens are one tensor, not two copies.

    The token.v6 tie survives the from_pretrained load path.
    """
    model, _ = direct_model
    assert model.config.tie_word_embeddings
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_save_load_round_trip(tmp_path, direct_model):
    from safetensors import safe_open
    from transformers import Phi3ForCausalLM

    model, oracle = direct_model
    save_dir = tmp_path / "bundle"
    model.save_pretrained(save_dir)
    # token.v6 tied: exactly one serialized token table, under the embedding's
    # key; lm_head is reconstructed as an alias by tie_weights() at load.
    saved_keys = set()
    for shard in save_dir.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            saved_keys |= set(handle.keys())
    assert "model.embed_tokens.weight" in saved_keys
    assert "lm_head.weight" not in saved_keys
    reloaded = Phi3ForCausalLM.from_pretrained(
        save_dir, attn_implementation="eager"
    ).eval()

    prefill = _prefill_ids(oracle)
    with torch.no_grad():
        a = model(input_ids=torch.tensor([prefill]), use_cache=True).logits
        b = reloaded(input_ids=torch.tensor([prefill]), use_cache=True).logits
    assert torch.equal(a, b)
    # Tied: lm_head and embed_tokens share storage after reload as well.
    assert (
        reloaded.lm_head.weight.data_ptr()
        == reloaded.model.embed_tokens.weight.data_ptr()
    )
