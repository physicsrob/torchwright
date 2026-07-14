"""Converter round-trip: ONNX artifact → native model → bit-exact parity.

Compiles a small example (``binary_increment``, ~3s on CPU; no committed
artifact — ``*.onnx`` is gitignored), converts it with ``compiler/hf/convert``,
and checks that the native :class:`TorchwrightForCausalLM`:

1. derives a config whose scalar dims agree with the artifact's debug sidecar;
2. consumes every ONNX initializer (the converter asserts this internally — a
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

from torchwright.compiler.export import debug_meta_path_for
from torchwright.compiler.hf.convert import convert_onnx_to_hf
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
def converted(artifact_path):
    model = convert_onnx_to_hf(artifact_path, bos_token=_BOS, eos_token=_EOS)
    oracle = load_onnx(artifact_path)
    return model, oracle


def _prefill_ids(oracle):
    tok2id = {t: i for i, t in enumerate(oracle.vocab)}
    return [tok2id[t] for t in ([_BOS] + list(_PREFILL_TEXT))]


def test_config_rejects_explicit_untied():
    """An explicit tie_word_embeddings=False (a pre-v6 saved config, or a
    deliberate untied experiment) must raise, not be silently overridden —
    HF's tie_weights() would otherwise clone the embedding over the
    checkpoint's real lm_head and return wrong logits with no error."""
    from torchwright.compiler.hf.configuration_torchwright import (
        TorchwrightConfig,
    )

    with pytest.raises(ValueError, match="tie_word_embeddings"):
        TorchwrightConfig(tie_word_embeddings=False)
    assert TorchwrightConfig().tie_word_embeddings
    assert TorchwrightConfig(tie_word_embeddings=True).tie_word_embeddings


def test_config_matches_debug_sidecar(artifact_path, converted):
    model, _ = converted
    cfg = model.config
    with open(debug_meta_path_for(artifact_path)) as f:
        dbg = json.load(f)
    assert cfg.d == dbg["d"]
    assert cfg.d_head == dbg["d_head"]
    assert cfg.n_layers == dbg["n_layers"]
    # Per-layer trimmed hidden widths come straight from the debug sidecar list.
    assert cfg.d_hidden_per_layer == [int(x) for x in dbg["d_hidden"]]
    assert len(cfg.n_heads_per_layer) == cfg.n_layers
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


def test_prefill_and_decode_bit_exact(converted):
    """ONNX oracle (onnxruntime) vs native HF model (torch): the meaningful
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
    * We teacher-force the native model on the *oracle's* token stream rather
      than letting each backend free-run on its own argmax. Past the meaningful
      output the logits are all-denormal noise whose argmax is arbitrary, so two
      free-running loops would pick different garbage and diverge; teacher
      forcing keeps every compared row on identical inputs.
    """
    model, oracle = converted
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


def test_bare_forward_without_use_cache(converted):
    """A plain ``model(input_ids=...)`` with NO kwargs must not crash.

    Regression for the transformers-5.x config stripping ``use_cache`` off the
    config dataclass — reading ``self.config.use_cache`` raised AttributeError.
    Every other test passes ``use_cache=True`` and masked it. This is the
    standard logit-eval call a Hub consumer makes outside ``generate``.
    """
    model, oracle = converted
    prefill = _prefill_ids(oracle)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prefill]))
    assert out.logits.shape[0] == 1 and out.logits.shape[1] == len(prefill)
    assert torch.isfinite(out.logits).all()


def test_labels_loss_path(converted):
    """``labels=`` returns a finite scalar loss (the training/perplexity path)."""
    model, oracle = converted
    ids = torch.tensor([_prefill_ids(oracle)])
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    assert out.loss is not None and out.loss.ndim == 0
    assert torch.isfinite(out.loss)


def test_inputs_embeds_rejected(converted):
    """inputs_embeds isn't supported (the table-lookup width is non-standard)."""
    import pytest as _pytest

    model, oracle = converted
    ids = torch.tensor([_prefill_ids(oracle)])
    emb = model.model.embed_tokens(ids)
    with _pytest.raises(NotImplementedError):
        model(inputs_embeds=emb)


def test_fp32_only_guard(converted):
    """A downcast model must refuse to run (fp32-only invariant), not emit garbage."""
    import pytest as _pytest

    from torchwright.compiler.hf.modeling_torchwright import TorchwrightForCausalLM

    model, oracle = converted
    half = TorchwrightForCausalLM(
        model.config
    ).half()  # fresh; guard fires pre-correctness
    with _pytest.raises(RuntimeError, match="fp32"):
        half(input_ids=torch.tensor([_prefill_ids(oracle)]))


def test_position_ids_are_honored(converted):
    """position_ids must shift which absolute PE rows are used (not be ignored)."""
    model, oracle = converted
    ids = torch.tensor([_prefill_ids(oracle)])
    T = ids.shape[1]
    with torch.no_grad():
        default = model(input_ids=ids).logits  # cache_position = arange(0, T)
        shifted = model(
            input_ids=ids, position_ids=torch.arange(100, 100 + T)[None]
        ).logits
    # Different absolute positions -> different PE -> different logits.
    assert not torch.equal(default, shifted)


def test_unsupported_attention_mask_raises(converted):
    """A 4D / wrong-length mask must fail loudly, not silently drop to causal."""
    import pytest as _pytest

    model, oracle = converted
    ids = torch.tensor([_prefill_ids(oracle)])
    T = ids.shape[1]
    with _pytest.raises(NotImplementedError):
        model(input_ids=ids, attention_mask=torch.ones(1, 1, T, T))  # 4D
    with _pytest.raises(NotImplementedError):
        model(input_ids=ids, attention_mask=torch.ones(1, T + 5))  # wrong length


def test_save_load_round_trip(tmp_path, converted):
    from torchwright.compiler.hf.modeling_torchwright import TorchwrightForCausalLM

    model, oracle = converted
    save_dir = tmp_path / "bundle"
    model.save_pretrained(save_dir)
    reloaded = TorchwrightForCausalLM.from_pretrained(save_dir).eval()

    prefill = _prefill_ids(oracle)
    with torch.no_grad():
        a = model(input_ids=torch.tensor([prefill]), use_cache=True).logits
        b = reloaded(input_ids=torch.tensor([prefill]), use_cache=True).logits
    assert torch.equal(a, b)
    assert (
        reloaded.lm_head.weight.data_ptr()
        == reloaded.model.embed_tokens.weight.data_ptr()
    )
