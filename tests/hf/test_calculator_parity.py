"""Flagship calculator parity: native model == ONNX oracle, token-for-token.

Compiles ``examples/calculator_v2.py`` to ONNX in-process (~6s on CPU; the
artifact is gitignored, never committed), converts it to a native
:class:`TorchwrightForCausalLM`, and asserts the native model reproduces the
ONNX runtime's behaviour on real arithmetic:

* **bit-exact** prefill logits versus :class:`OnnxTokenModule`, and
* **token-identical** greedy generation that matches the correct answer,

on a curated set spanning the calculator's reliable range (additions and
subtractions at any magnitude; multiplications with comfortable digit margin).

Why "reliable range": the multiply path computes ``a*b = ((a+b)^2 - (a-b)^2)/4``
with a thermometer-coded piecewise-linear squaring whose intermediate reaches
~4e6 for large operands. onnxruntime and torch accumulate that fp32 matmul
reduction in different orders; once the difference exceeds a digit-quantization
level's half-width (operands roughly >= 900), the two backends round one output
digit to adjacent levels — a fixed ~1600 logit gap that flips a borderline
argmax. There the compiled circuit is already at its numerical-noise budget
(the ONNX oracle itself is wrong on e.g. 999*999). That is a property of the
compiled arithmetic at the edge of its range, not of this reimplementation:
additions/subtractions (no squaring) and smaller products are bit-identical, as
the gate below asserts. See ``test_extreme_multiply_is_the_only_divergence``.

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

_BOS = "<bos"
_EOS = "<eos>"

# In-range gate: add/sub at any magnitude, mult with comfortable digit margin.
# Every entry is bit-exact AND lands the correct answer (confirmed empirically).
_GATE = [
    "0+0\n", "7+8\n", "12+34\n", "999+1\n", "500+499\n", "123+456\n",
    "9-4\n", "100-99\n", "45-67\n", "0-7\n", "999-1\n", "320-200\n",
    "12*34\n", "9*9\n", "25*25\n", "100*7\n", "321*3\n", "123*4\n",
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
    ids = torch.tensor(
        [[tok2id[t] for t in ([_BOS] + list(text))]], dtype=torch.int64
    )
    with torch.no_grad():
        g = model.generate(
            ids,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
            eos_token_id=tok2id[_EOS],
            pad_token_id=tok2id[_EOS],
        )
    new = g[0, ids.shape[1]:].tolist()
    return "".join(vocab[i] for i in new if i != tok2id[_EOS])


@pytest.mark.parametrize("text", _GATE)
def test_prefill_bit_exact(model, oracle, tok2id, text):
    o, h = _prefill_logits(model, oracle, tok2id, text)
    assert (o - h).abs().max().item() == 0.0


@pytest.mark.parametrize("text", _GATE)
def test_greedy_token_identical_and_correct(model, oracle, tok2id, artifact_path, text):
    vocab = read_vocab(artifact_path)
    expected = str(eval(text.strip()))
    o_out = _oracle_gen(oracle, text)
    h_out = _hf_gen(model, tok2id, vocab, text)
    assert o_out == h_out, f"{text!r}: hf={h_out!r} oracle={o_out!r}"
    assert h_out == expected, f"{text!r}: got {h_out!r}, want {expected!r}"


def test_extreme_multiply_is_the_only_divergence(model, oracle, tok2id):
    """Document that divergence appears ONLY at the squaring-path noise budget.

    A structural bug (wrong cache, mask, or weight map) would corrupt every
    expression; instead divergence is confined to large-operand multiplies, and
    when it appears it is exactly one digit-level (a ~1600 logit gap) — the
    fingerprint of a piecewise-linear boundary rounded differently by the two
    fp32 backends, not noise in this reimplementation.
    """
    diverged = []
    for text in _EXTREME:
        o, h = _prefill_logits(model, oracle, tok2id, text)
        d = (o - h).abs().max().item()
        if d != 0.0:
            diverged.append((text, d))
            assert d == pytest.approx(1600.0, abs=1.0), (
                f"{text}: unexpected divergence magnitude {d} — expected a single "
                f"digit-level (~1600) boundary flip, not free-floating noise"
            )
    assert diverged, (
        "expected the extreme multiplies to exercise the squaring-path boundary; "
        "if they no longer diverge the calculator's precision improved — update "
        "this test and the module docstring"
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
        g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
    )
    assert out == "408", f"round-trip generate gave {out!r}"
