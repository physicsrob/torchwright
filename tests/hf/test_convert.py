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
    hf_decode,
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
    # Untied vanilla layout: no separate embedding/output-gather width remains.
    assert not cfg.tie_word_embeddings


def test_prefill_and_decode_bit_exact(converted):
    model, oracle = converted
    prefill = _prefill_ids(oracle)
    o_ids, o_logits = oracle_decode(oracle, prefill, _N_STEPS)
    h_ids, h_logits = hf_decode(model, prefill, _N_STEPS)
    assert h_ids == o_ids, f"decode tokens diverged: hf={h_ids} oracle={o_ids}"
    assert max_logit_diff(o_logits, h_logits) == 0.0


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
    # Untied: lm_head and embed_tokens are independent tensors (separate storage).
    assert (
        reloaded.lm_head.weight.data_ptr()
        != reloaded.model.embed_tokens.weight.data_ptr()
    )
