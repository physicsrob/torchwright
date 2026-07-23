"""The identity RMSNorm is bit-exact on the shipping graph, end to end.

Compiles ``examples/calculator_simple.py`` (the production HF-export graph,
built at the family's canonical ``D_HEAD``; exported at the smaller
``_EXPORT_D`` — see its comment) both with the RMSNorm forced on and with it
off, and asserts:

* the ONNX oracle (real onnxruntime execution) produces **bit-identical** logits
  with and without the norm — i.e. the emitted RMSNorm is exactly the identity,
  including through the scheduler's cancel heads, on the squaring-path
  expressions whose energy broke the original q=30 setting; and
* the direct-built HF model carries the norm as Llama3-named ``weight``
  parameters (``input_layernorm`` / ``post_attention_layernorm`` /
  ``model.norm``); transformer-layer gains are uniform powers of two, while
  the final gain zeros only the pinned seed columns after using them in its
  denominator (token.v6 tied readout), and the norm-off model carries none.

CPU-only and deterministic, so the bar is exact-bit (see ``_hf_parity``).

A second section repeats the ONNX bit-exactness bar at d=5120 (= 5·2^10),
the first supported non-power-of-two width, whose pinned layout carries two
unequal constants (see ``docs/rms_norm_dmodel.md``).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from torchwright.compiler.export import compile_to_onnx
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy

# Everything on attention, cancels included: the layout-stable value path
# the bit-exact norm-on/norm-off bar requires (see _compile's comment).
_BIT_STABLE_POLICY = SchedulingPolicy(
    local_in_attention="always",
    add_in_attention="always",
    cancel_in_attention="always",
)
from torchwright.compiler.hf import compile_to_hf
from torchwright.compiler.onnx_load import load_onnx

_BOS, _EOS = "<bos>", "<eos>"

# Export width for the calculator compiles below.  The bar here is the norm's
# bit-exact identity, not the publish geometry: the family's canonical
# D_MODEL=8192 would make each of the two ONNX artifacts tens of GB of dense
# fp32 for nothing (d=1024 exercises the same pinned-column layout class —
# one constant column).  d_head must stay the family D_HEAD: it is baked into
# the graph at build time.
_EXPORT_D = 1024
# squaring-path expressions whose residual energy (~2.6e13) exceeded the
# original q=30 bound and silently broke the identity; plus easy ones.
_EXPRS = ["12*34\n", "999+1\n", "0-7\n", "999*999\n", "321*3\n"]


def _compile(*, rms_norm: bool) -> str:
    from examples import calculator_simple

    out_node, emb = calculator_simple.create_network_parts()
    d_dir = tempfile.mkdtemp(prefix=f"tw_rmsnorm_{rms_norm}_")
    art = compile_to_onnx(
        out_node,
        emb,
        str(Path(d_dir) / "m.onnx"),
        d=_EXPORT_D,
        d_head=calculator_simple.D_HEAD,
        rms_norm=rms_norm,
        # The bit-exact norm-on/norm-off comparison needs a value path that
        # is bit-stable across the two compiles' layouts.  MLP-routed Adds
        # AND MLP bypass cancels on the swish machine carry a
        # schedule-dependent fp32 residue (~2^-41 on near-zero lanes), and
        # the norm's pinned-column reservation makes the two scheduling
        # problems differ — so pin the attention Add routing and the
        # attention cancel mechanism here.  (Adds stay on attention under
        # the shipping default too; cancels do not — the eager walk
        # MLP-cancels whatever overflows a layer's attention batch, and
        # which nodes overflow is layout luck.  Measured: the 999*999
        # rollout differs by exactly one 2^-40 crumb's column without the
        # cancel pin.)  This test certifies the NORM; the MLP Add and
        # cancel paths have their own oracle-parity coverage
        # (tests/compile/forward/test_mlp_add_routing.py and the HF
        # parity suite).
        policy=_BIT_STABLE_POLICY,
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
    ids = torch.tensor([t2i[t] for t in ([_BOS, *list(text)])], dtype=torch.int64)
    return oracle(ids)[-1]


@pytest.mark.parametrize("text", _EXPRS)
def test_norm_is_bit_exact_identity_in_onnx(path_on, path_off, text):
    """norm-on oracle == norm-off oracle, bit-for-bit (the norm is identity)."""
    on, off = load_onnx(path_on), load_onnx(path_off)
    vocab = load_onnx(path_on).vocab
    lo_on = _last_logits(on, vocab, text)
    lo_off = _last_logits(off, vocab, text)
    assert (lo_on - lo_off).abs().max().item() == 0.0


def _compile_hf(rms_norm):
    from examples import calculator_simple

    out, emb = calculator_simple.create_network_parts()
    return compile_to_hf(
        out,
        emb,
        d=_EXPORT_D,
        d_head=calculator_simple.D_HEAD,
        rms_norm=rms_norm,
    )


def test_direct_model_model_has_llama3_norm_params(path_on):
    model = _compile_hf(rms_norm=True)
    sd = model.state_dict()
    assert "model.norm.weight" in sd, "missing final norm"
    assert "model.layers.0.input_layernorm.weight" in sd
    assert "model.layers.0.post_attention_layernorm.weight" in sd
    layer_keys = [k for k in sd if k.endswith("layernorm.weight")]
    gains = {round(float(sd[k].min()), 6) for k in layer_keys}
    gains |= {round(float(sd[k].max()), 6) for k in layer_keys}
    assert len(gains) == 1, f"layer gains not uniform: {gains}"
    g = next(iter(gains))
    assert math.log2(g) == int(math.log2(g)), f"gain {g} is not a power of two"
    final = sd["model.norm.weight"]
    assert float(final.max()) == g
    assert (final == 0.0).any(), "final norm must clear pinned RMS seed columns"
    assert set(final.unique().tolist()) == {0.0, g}


def test_norm_off_model_has_no_norm_params(path_off):
    with pytest.raises(ValueError, match="requires rms_norm=True"):
        _compile_hf(rms_norm=False)


# ===========================================================================
# Non-power-of-two width: d=5120 = 5·2^10, the first contract width whose
# pinned layout uses two UNEQUAL constants (2^q and 2^(q+1)).  A tiny token
# graph keeps the artifact small (d_hidden bounds the fat matrices); the bar
# is the same as above — norm-on and norm-off artifacts produce bit-identical
# logits through real onnxruntime execution.
# ===========================================================================

_TINY_VOCAB = [*list("0123456789+"), "\n", _BOS, _EOS, "default"]
_TINY_TOKENS = [_BOS, "1", "+", "2", "\n"]


def _compile_tiny_5120(*, rms_norm: bool) -> str:
    import torch

    from torchwright.graph import Linear
    from torchwright.ops.inout_nodes import create_embedding
    from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

    emb = create_embedding(vocab=_TINY_VOCAB)
    d_e = len(emb)
    g = torch.Generator().manual_seed(3)
    ffn = linear_relu_linear(
        emb,
        torch.randn(24, d_e, generator=g) * 0.2,
        torch.randn(24, generator=g) * 0.1,
        torch.randn(24, d_e, generator=g) * 0.2,
        torch.randn(d_e, generator=g) * 0.1,
        name="ffn",
    )
    # The held-bank contract deliberately has no direct MLP
    # read/cancel/overwrite event. Keep this RMS fixture's output behind an
    # attention-routable identity writer.
    out = Linear(ffn, torch.eye(d_e), name="output")
    d_dir = tempfile.mkdtemp(prefix=f"tw_rmsnorm5120_{rms_norm}_")
    art = compile_to_onnx(
        out,
        emb,
        str(Path(d_dir) / "m.onnx"),
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
    t2i = {t: i for i, t in enumerate(load_onnx(path_on).vocab)}
    import torch

    ids = torch.tensor([t2i[t] for t in _TINY_TOKENS], dtype=torch.int64)
    lo_on = on(ids)[-1]
    lo_off = off(ids)[-1]
    assert (lo_on - lo_off).abs().max().item() == 0.0
