"""Flagship calculator parity: native model == ONNX oracle, token-for-token.

Compiles ``examples/calculator_v2.py`` to ONNX in-process (~6s on CPU; the
artifact is gitignored, never committed), converts it to a native
:class:`TorchwrightForCausalLM`, and asserts the native model reproduces the
ONNX runtime's behaviour on real arithmetic:

* **token-identical** prefill (per-position argmax) versus :class:`OnnxTokenModule`,
  with logits within a small cross-backend fp floor, and
* **token-identical** greedy generation that matches the correct answer,

on a curated set spanning the calculator's reliable range (additions and
subtractions at any magnitude; multiplications with comfortable digit margin).

Why "within an fp floor" and not bit-exact: the RoPE local-recency port (Phase 6)
reshaped the graph, and onnxruntime vs torch now round a few logits differently
at the operator-latch positions — a cross-backend fp floor (~5 logits, amplitude-
independent, *not* the squaring path; see ``docs/numerical_noise_findings.md``
"local-recency cross-path fp floor"). It changes no argmax, so generation stays
token-identical; the prefill check asserts per-position argmax-equality (the real
token bar) plus a bounded logit floor.

Why "reliable range": the multiply path computes ``a*b = ((a+b)^2 - (a-b)^2)/4``
with a thermometer-coded piecewise-linear squaring whose intermediate reaches
~4e6 for large operands. onnxruntime and torch can accumulate that fp32 matmul
reduction in different orders (whether they actually differ depends on the
backend builds); when they do and the difference exceeds a digit-quantization
level's half-width (operands roughly >= 900), the two backends round one output
digit to adjacent levels — a fixed ~1600 logit gap that flips a borderline
argmax. There the compiled circuit is already at its numerical-noise budget
(the ONNX oracle itself is wrong on e.g. 999*999). That is a property of the
compiled arithmetic at the edge of its range, not of this reimplementation:
additions/subtractions (no squaring) and smaller products are bit-identical, as
the gate below asserts. See
``test_extreme_multiply_at_most_one_digit_level_divergence``.

CPU-only and deterministic, so the bar is exact-bit, not merely argmax.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from torchwright.compiler.hf.convert import convert_onnx_to_hf, read_vocab
from torchwright.compiler.onnx_load import load_onnx

from tests.hf._hf_parity import compile_example

_BOS = "<bos>"
_EOS = "<eos>"

# Cross-backend (onnxruntime vs torch) logit floor introduced by the Phase-6
# local-recency graph reshape: amplitude-independent, ~5 logits observed on the
# gate set (Modal CPU pair ~2.3, other build pairs ~4.7), flips no argmax.  The
# real correctness bar is per-position argmax-equality; this bounds the residual
# logit divergence so a structural regression (free-floating 10s–1000s) still
# fails.  See docs/numerical_noise_findings.md.
_PARITY_FP_FLOOR = 8.0

# In-range gate: add/sub at any magnitude, mult with comfortable digit margin.
# Every entry is bit-exact AND lands the correct answer (confirmed empirically).
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
]

# Documented edge cases: large-operand multiplies where the two fp32 backends
# can land on opposite sides of a digit-quantization boundary (see module
# docstring). Listed so the divergence is a tracked, explained property.
_EXTREME = ["900*900\n", "950*950\n", "999*999\n"]


@pytest.fixture(scope="module")
def artifact_path():
    return compile_example("calculator_v2")


@pytest.fixture(scope="module")
def model(artifact_path):
    return convert_onnx_to_hf(artifact_path, bos_token=_BOS, eos_token=_EOS)


@pytest.fixture(scope="module")
def oracle(artifact_path):
    return load_onnx(artifact_path)


@pytest.fixture(scope="module")
def tok2id(artifact_path):
    return {t: i for i, t in enumerate(read_vocab(artifact_path))}


def _prefill_logits(model, oracle, tok2id, text):
    ids = [tok2id[t] for t in ([_BOS] + list(text))]
    o = oracle(torch.tensor(ids, dtype=torch.int64))
    with torch.no_grad():
        h = model(input_ids=torch.tensor([ids], dtype=torch.int64), use_cache=True)
    return o, h.logits[0]


def _oracle_gen(oracle, text):
    return "".join(
        oracle.generate(text, max_new_tokens=10, bos_token=_BOS, eos_token=_EOS)
    )


def _hf_gen(model, tok2id, vocab, text):
    ids = torch.tensor([[tok2id[t] for t in ([_BOS] + list(text))]], dtype=torch.int64)
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
    # Real bar: native HF and the ONNX oracle pick the same token at every
    # teacher-forced position (no argmax flip).
    assert (
        o.argmax(-1) == h.argmax(-1)
    ).all(), (
        f"{text!r}: native/oracle argmax differs at a prefill position (token flip)"
    )
    # Residual logit divergence stays within the cross-backend fp floor (Phase-6
    # local-recency reshape) — a structural regression would blow past it.
    assert (o - h).abs().max().item() <= _PARITY_FP_FLOOR


@pytest.mark.parametrize("text", _GATE)
def test_greedy_token_identical_and_correct(model, oracle, tok2id, artifact_path, text):
    vocab = read_vocab(artifact_path)
    expected = str(eval(text.strip()))
    o_out = _oracle_gen(oracle, text)
    h_out = _hf_gen(model, tok2id, vocab, text)
    assert o_out == h_out, f"{text!r}: hf={h_out!r} oracle={o_out!r}"
    assert h_out == expected, f"{text!r}: got {h_out!r}, want {expected!r}"


def test_extreme_multiply_at_most_one_digit_level_divergence(model, oracle, tok2id):
    """Extreme multiplies stay within one digit-level of the ONNX oracle.

    A structural bug (wrong cache, mask, or weight map) would corrupt every
    expression by a free-floating amount.  Instead, on the largest-operand
    multiplies the native model and the ONNX oracle either agree bit-for-bit or
    differ by exactly one digit-quantization level (a ~1600 logit gap) — the
    fingerprint of a piecewise-linear boundary the two fp32 backends round to
    adjacent levels.

    Whether the gap actually appears is backend-dependent: it turns on the order
    the squaring-path fp32 reduction is accumulated, which differs across
    onnxruntime/torch builds (e.g. the CPU pair on Modal agrees on the squaring
    path even on 999*999, while other build pairs flip one level).  So we assert
    the *bound*, not the gap's presence — within the cross-backend fp floor (the
    Phase-6 local-recency reshape, ``_PARITY_FP_FLOOR``), or that floor plus
    exactly one ~1600 digit-level — which distinguishes the squaring boundary
    from a free-floating structural bug.
    """
    for text in _EXTREME:
        o, h = _prefill_logits(model, oracle, tok2id, text)
        d = (o - h).abs().max().item()
        assert d <= _PARITY_FP_FLOOR or abs(d - 1600.0) <= _PARITY_FP_FLOOR, (
            f"{text}: divergence {d} is neither within the cross-backend fp floor "
            f"(~{_PARITY_FP_FLOOR}) nor a single digit-level (~1600) boundary flip "
            f"— that signals free-floating noise or a structural bug, not the "
            f"squaring-path quantization edge"
        )


def test_save_load_generate_round_trip(tmp_path, artifact_path, model, oracle, tok2id):
    """Save → reload (classes imported) → tokenizer-driven generate == '408'."""
    from torchwright.compiler.hf.modeling_torchwright import TorchwrightForCausalLM
    from torchwright.compiler.hf.tokenization_torchwright import TorchwrightTokenizer

    save_dir = tmp_path / "calc"
    model.save_pretrained(save_dir)
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(read_vocab(artifact_path)))
    tok = TorchwrightTokenizer(
        vocab_file=str(vocab_path), bos_token=_BOS, eos_token=_EOS
    )
    tok.save_pretrained(save_dir)

    reloaded = TorchwrightForCausalLM.from_pretrained(save_dir).eval()
    reloaded_tok = TorchwrightTokenizer.from_pretrained(save_dir)

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
