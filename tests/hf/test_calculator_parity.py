"""Flagship calculator parity: stock Phi-3 == ONNX oracle, token-for-token.

Compiles ``examples/calculator_simple.py`` — the one-hot lookup-table
calculator — to ONNX in-process (the artifact is gitignored, never
committed), compiles it directly to stock :class:`Phi3ForCausalLM`, and
asserts the HF model reproduces the ONNX runtime's behaviour on real
arithmetic:

* **token-identical** prefill (per-position argmax) versus :class:`OnnxTokenModule`,
  with logits within a small cross-backend fp floor, and
* **token-identical** greedy generation that matches the correct answer,

on a gate spanning all three ops at every operand magnitude the 3-digit
build supports — the one-hot arithmetic is exact table lookup, so there is
no reduced-precision edge to carve out (999*999 decodes exactly; a
"reliable range" carve-out existed only for the retired scalar-space
calculator, whose thermometer-coded squaring path saturated its noise
budget at large operands).

Why "within an fp floor" and not bit-exact: the RoPE local-recency port (Phase 6)
reshaped the graph, and onnxruntime vs torch now round a few logits differently
at the operator-latch positions — a cross-backend fp floor (~5 logits, amplitude-
independent; see ``docs/numerical_noise_findings.md`` "local-recency cross-path
fp floor"). It changes no argmax, so generation stays token-identical; the
prefill check asserts per-position argmax-equality (the real token bar) plus a
bounded logit floor.

CPU-only and deterministic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("onnxruntime")

from tests.hf._hf_parity import compile_example
from torchwright.compiler.hf import compile_to_hf
from torchwright.compiler.onnx_load import load_onnx

_BOS = "<bos>"
_EOS = "<eos>"

# Cross-backend (onnxruntime vs torch) logit floor introduced by the Phase-6
# local-recency graph reshape: amplitude-independent, ~5 logits observed on the
# gate set (Modal CPU pair ~2.3, other build pairs ~4.7), flips no argmax.  The
# real correctness bar is per-position argmax-equality; this bounds the residual
# logit divergence so a structural regression (free-floating 10s-1000s) still
# fails.  See docs/numerical_noise_findings.md.
_PARITY_FP_FLOOR = 8.0

# All three ops at every magnitude the 3-digit build supports, largest-operand
# multiplies included — the one-hot arithmetic is exact lookup, so nothing is
# carved out.  Every entry lands the correct answer (confirmed empirically).
_GATE = [
    "0+0\n",
    "7+8\n",
    "12+34\n",
    "999+1\n",
    "500+499\n",
    "123+456\n",
    "9-4\n",
    "100-99\n",
    "45-67\n",
    "0-7\n",
    "999-1\n",
    "320-200\n",
    "12*34\n",
    "9*9\n",
    "25*25\n",
    "100*7\n",
    "321*3\n",
    "123*4\n",
    "900*900\n",
    "950*950\n",
    "999*999\n",
]


# ONNX + HF compiled at the same pinned export width (parity is between the
# two backends, so they must share a geometry).  1024 keeps both artifacts
# test-sized; the family's canonical publish width (D_MODEL=8192) is witnessed
# by the layer table, not here.
_EXPORT_D = 1024


@pytest.fixture(scope="module")
def artifact_path():
    return compile_example("calculator_simple", d=_EXPORT_D)


@pytest.fixture(scope="module")
def model(artifact_path):
    from examples import calculator_simple

    out, emb = calculator_simple.create_network_parts()
    return compile_to_hf(out, emb, d=_EXPORT_D, d_head=calculator_simple.D_HEAD)


@pytest.fixture(scope="module")
def oracle(artifact_path):
    return load_onnx(artifact_path)


@pytest.fixture(scope="module")
def tok2id(oracle):
    return {t: i for i, t in enumerate(oracle.vocab)}


def _prefill_logits(model, oracle, tok2id, text):
    ids = [tok2id[t] for t in ([_BOS, *list(text)])]
    o = oracle(torch.tensor(ids, dtype=torch.int64))
    with torch.no_grad():
        h = model(input_ids=torch.tensor([ids], dtype=torch.int64), use_cache=True)
    return o, h.logits[0]


def _oracle_gen(oracle, text):
    return "".join(
        oracle.generate(text, max_new_tokens=10, bos_token=_BOS, eos_token=_EOS)
    )


def _hf_gen(model, tok2id, vocab, text):
    ids = torch.tensor([[tok2id[t] for t in ([_BOS, *list(text)])]], dtype=torch.int64)
    with torch.no_grad():
        g = model.generate(
            ids,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
            eos_token_id=tok2id[_EOS],
            pad_token_id=tok2id[_EOS],
        )
    new = g[0, ids.shape[1] :].tolist()
    return "".join(vocab[i] for i in new if i != tok2id[_EOS])


@pytest.mark.parametrize("text", _GATE)
def test_prefill_token_identical_within_fp_floor(model, oracle, tok2id, text):
    o, h = _prefill_logits(model, oracle, tok2id, text)
    # Real bar: stock Phi-3 and the ONNX oracle pick the same token at every
    # teacher-forced position (no argmax flip).
    assert (o.argmax(-1) == h.argmax(-1)).all(), (
        f"{text!r}: Phi-3/oracle argmax differs at a prefill position (token flip)"
    )
    # Residual logit divergence stays within the cross-backend fp floor (Phase-6
    # local-recency reshape) — a structural regression would blow past it.
    assert (o - h).abs().max().item() <= _PARITY_FP_FLOOR


@pytest.mark.parametrize("text", _GATE)
def test_greedy_token_identical_and_correct(model, oracle, tok2id, artifact_path, text):
    vocab = oracle.vocab
    expected = str(eval(text.strip()))
    o_out = _oracle_gen(oracle, text)
    h_out = _hf_gen(model, tok2id, vocab, text)
    assert o_out == h_out, f"{text!r}: hf={h_out!r} oracle={o_out!r}"
    assert h_out == expected, f"{text!r}: got {h_out!r}, want {expected!r}"


def test_save_load_generate_round_trip(tmp_path, artifact_path, model, oracle, tok2id):
    """Save → reload (classes imported) → tokenizer-driven generate == '408'."""
    from transformers import Phi3ForCausalLM

    from torchwright.compiler.hf import build_fast_tokenizer

    save_dir = tmp_path / "calc"
    model.save_pretrained(save_dir)
    tok = build_fast_tokenizer(oracle.vocab, bos_token=_BOS, eos_token=_EOS)
    tok.save_pretrained(save_dir)

    reloaded = Phi3ForCausalLM.from_pretrained(save_dir).eval()
    from transformers import AutoTokenizer

    reloaded_tok = AutoTokenizer.from_pretrained(save_dir)

    enc = reloaded_tok("12*34\n", return_tensors="pt")
    with torch.no_grad():
        g = reloaded.generate(
            enc["input_ids"],
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
            eos_token_id=reloaded_tok.eos_token_id,
            pad_token_id=reloaded_tok.eos_token_id,
        )
    out = reloaded_tok.decode(
        g[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
    )
    assert out == "408", f"round-trip generate gave {out!r}"
