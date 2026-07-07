"""The identity RMSNorm is bit-exact on the shipping graph, end to end.

Compiles ``examples/calculator_v2.py`` (the production HF-export graph, d=1024)
both with the RMSNorm forced on and with it off, and asserts:

* the ONNX oracle (real onnxruntime execution) produces **bit-identical** logits
  with and without the norm — i.e. the emitted RMSNorm is exactly the identity,
  including through the scheduler's cancel heads, on the squaring-path
  expressions whose energy broke the original q=30 setting; and
* the converted HF model carries the norm as Llama3-named ``weight`` parameters
  (``input_layernorm`` / ``post_attention_layernorm`` / ``model.norm``), each a
  uniform power-of-two gain, while the norm-off model carries none.

CPU-only and deterministic, so the bar is exact-bit (see ``_hf_parity``).

A second section repeats the ONNX bit-exactness bar at d=5120 (= 5·2^10),
the first supported non-power-of-two width, whose pinned layout carries two
unequal constants (see ``docs/rms_norm_dmodel.md``).
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from torchwright.compiler.export import compile_to_onnx
from torchwright.compiler.hf.convert import convert_onnx_to_hf, read_vocab
from torchwright.compiler.onnx_load import load_onnx

_BOS, _EOS = "<bos>", "<eos>"
# squaring-path expressions whose residual energy (~2.6e13) exceeded the
# original q=30 bound and silently broke the identity; plus easy ones.
_EXPRS = ["12*34\n", "999+1\n", "0-7\n", "999*999\n", "321*3\n"]


def _compile(rms_norm: bool) -> str:
    from examples import calculator_v2

    out_node, emb = calculator_v2.create_network_parts()
    d_dir = tempfile.mkdtemp(prefix=f"tw_rmsnorm_{rms_norm}_")
    art = compile_to_onnx(
        out_node,
        emb,
        os.path.join(d_dir, "m.onnx"),
        d=calculator_v2.D_MODEL,
        rms_norm=rms_norm,
    )
    return art.path


@pytest.fixture(scope="module")
def path_on():
    return _compile(rms_norm=True)


@pytest.fixture(scope="module")
def path_off():
    return _compile(rms_norm=False)


def _last_logits(oracle, vocab, text):
    t2i = {t: i for i, t in enumerate(vocab)}
    ids = torch.tensor([t2i[t] for t in ([_BOS] + list(text))], dtype=torch.int64)
    return oracle(ids)[-1]


@pytest.mark.parametrize("text", _EXPRS)
def test_norm_is_bit_exact_identity_in_onnx(path_on, path_off, text):
    """norm-on oracle == norm-off oracle, bit-for-bit (the norm is identity)."""
    on, off = load_onnx(path_on), load_onnx(path_off)
    vocab = read_vocab(path_on)
    lo_on = _last_logits(on, vocab, text)
    lo_off = _last_logits(off, vocab, text)
    assert (lo_on - lo_off).abs().max().item() == 0.0


def test_converted_model_has_llama3_norm_params(path_on):
    model = convert_onnx_to_hf(path_on, bos_token=_BOS, eos_token=_EOS)
    sd = model.state_dict()
    assert "model.norm.weight" in sd, "missing final norm"
    assert "model.layers.0.input_layernorm.weight" in sd
    assert "model.layers.0.post_attention_layernorm.weight" in sd
    # every gain is a single power-of-two value (the cancel constant)
    norm_keys = [
        k for k in sd if k.endswith("layernorm.weight") or k == "model.norm.weight"
    ]
    gains = {round(float(sd[k].min()), 6) for k in norm_keys}
    gains |= {round(float(sd[k].max()), 6) for k in norm_keys}
    assert len(gains) == 1, f"gains not uniform: {gains}"
    g = next(iter(gains))
    assert math.log2(g) == int(math.log2(g)), f"gain {g} is not a power of two"


def test_norm_off_model_has_no_norm_params(path_off):
    model = convert_onnx_to_hf(path_off, bos_token=_BOS, eos_token=_EOS)
    sd = model.state_dict()
    assert not [
        k for k in sd if "layernorm" in k or k == "model.norm.weight"
    ], "norm-off model unexpectedly carries norm weights"


# ===========================================================================
# Non-power-of-two width: d=5120 = 5·2^10, the first contract width whose
# pinned layout uses two UNEQUAL constants (2^q and 2^(q+1)).  A tiny token
# graph keeps the artifact small (d_hidden bounds the fat matrices); the bar
# is the same as above — norm-on and norm-off artifacts produce bit-identical
# logits through real onnxruntime execution.
# ===========================================================================

_TINY_VOCAB = list("0123456789+") + ["\n", _BOS, _EOS, "default"]
_TINY_TOKENS = [_BOS, "1", "+", "2", "\n"]


def _compile_tiny_5120(rms_norm: bool) -> str:
    import torch

    from torchwright.ops.inout_nodes import create_embedding
    from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

    emb = create_embedding(vocab=_TINY_VOCAB)
    d_e = len(emb)
    g = torch.Generator().manual_seed(3)
    out = linear_relu_linear(
        emb,
        torch.randn(24, d_e, generator=g) * 0.2,
        torch.randn(24, generator=g) * 0.1,
        torch.randn(24, d_e, generator=g) * 0.2,
        torch.randn(d_e, generator=g) * 0.1,
        name="ffn",
    )
    d_dir = tempfile.mkdtemp(prefix=f"tw_rmsnorm5120_{rms_norm}_")
    art = compile_to_onnx(
        out,
        emb,
        os.path.join(d_dir, "m.onnx"),
        d=5120,
        d_hidden=64,
        max_seq_len=16,
        rms_norm=rms_norm,
    )
    return art.path


def test_norm_is_bit_exact_identity_at_5120():
    path_on = _compile_tiny_5120(rms_norm=True)
    path_off = _compile_tiny_5120(rms_norm=False)
    on, off = load_onnx(path_on), load_onnx(path_off)
    t2i = {t: i for i, t in enumerate(read_vocab(path_on))}
    import torch

    ids = torch.tensor([t2i[t] for t in _TINY_TOKENS], dtype=torch.int64)
    lo_on = on(ids)[-1]
    lo_off = off(ids)[-1]
    assert (lo_on - lo_off).abs().max().item() == 0.0
