"""ONNX emission + debug surface for bias=False (no-bias) artifacts.

docs/no_bias_plan.md Phase N3: a ``bias=False`` export carries no bias
initializers and no bias Adds (each projection is its bare MatMul), records
the emission mode in the artifact / token meta / debug sidecar, and
round-trips the debug surface — probe_compiled over the executing artifact
is clean and its logits match the biased twin.  A sidecar/artifact pairing
mismatch trips the explicit emission-mode cross-check, and the custom HF profile
refuses biasless artifacts loudly.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch

from torchwright.compiler.export import compile_to_onnx, meta_path_for
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN, Linear
from torchwright.ops.inout_nodes import create_embedding
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

onnxruntime = pytest.importorskip("onnxruntime")

from torchwright.debug.onnx_debug import OnnxDebugSession

D = 256
D_HEAD = 16
VOCAB = [*list("0123456789+"), "\n", "<bos>", "<eos>", "default"]
TOKENS = ["<bos>", "1", "+", "2"]
_ONNX_PROBE_ATOL = 2.5e-3


def _build(machine="swish"):
    """A token graph with real bias content on every FFN (gate/up/out)."""
    emb = create_embedding(vocab=VOCAB)
    d = len(emb)
    g = torch.Generator().manual_seed(23)
    if machine == "swish":
        h = FFN(
            emb,
            gate_proj=torch.randn(24, d, generator=g) * 0.2,
            gate_bias=torch.randn(24, generator=g) * 0.1,
            out_proj=torch.randn(24, d, generator=g) * 0.2,
            out_bias=torch.randn(d, generator=g) * 0.1,
            up_proj=torch.randn(24, d, generator=g) * 0.2,
            up_bias=torch.randn(24, generator=g) * 0.1,
            activation="swish",
            name="gated",
        )
        out = FFN(
            h,
            gate_proj=torch.randn(16, d, generator=g) * 0.2,
            gate_bias=torch.randn(16, generator=g) * 0.1,
            out_proj=torch.randn(16, d, generator=g) * 0.2,
            out_bias=torch.randn(d, generator=g) * 0.1,
            activation="swish",
            name="degenerate",
        )
    else:
        out = linear_relu_linear(
            emb,
            torch.randn(24, d, generator=g) * 0.2,
            torch.randn(24, generator=g) * 0.1,
            torch.randn(24, d, generator=g) * 0.2,
            torch.randn(d, generator=g) * 0.1,
            name="ffn",
        )
    # Token.v6 deliberately rejects a direct MLP-only tied handoff. The final
    # identity writer preserves this fixture's FFN/bias behavior while giving
    # the held bank an attention-routable output operation.
    return Linear(out, torch.eye(d), name="output"), emb


def _token_ids(emb) -> torch.Tensor:
    return torch.tensor(
        [[emb.tokenizer.get_token_id(t)] for t in TOKENS], dtype=torch.long
    )


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """The same swish token graph exported both ways."""
    base = tmp_path_factory.mktemp("no_bias_onnx")
    paths = {}
    for mode, flag in (("biased", True), ("biasless", False)):
        out, emb = _build()
        path = str(base / f"{mode}.onnx")
        artifact = compile_to_onnx(
            out,
            emb,
            path,
            d=D,
            d_head=D_HEAD,
            max_seq_len=16,
            verbose=False,
            bias=flag,
        )
        paths[mode] = (artifact, path)
    return paths


def test_no_bias_artifact_records_mode(artifacts):
    artifact, onnx_path = artifacts["biasless"]
    assert artifact.bias is False
    with Path(meta_path_for(onnx_path)).open() as f:
        assert json.load(f)["bias"] is False
    with Path(artifact.debug_path).open() as f:
        assert json.load(f)["bias"] is False
    biased_artifact, biased_path = artifacts["biased"]
    assert biased_artifact.bias is True
    with Path(meta_path_for(biased_path)).open() as f:
        assert json.load(f)["bias"] is True


def test_artifact_records_collapse_provenance(artifacts):
    """Meta and debug sidecar both record that the collapse lowerings ran.

    (docs/univariate_collapse_plan.md; v1 staircase and the v2 general-PL
    S1 pass) — provenance for depth comparisons across artifacts,
    top-level like "bias" (never inside "extra", which is the caller's
    verbatim metadata). Lives here to reuse this module's exported
    artifacts; both passes are unconditional, so this pins the emitted
    keys, not a knob.
    """
    artifact, onnx_path = artifacts["biased"]
    with Path(meta_path_for(onnx_path)).open() as f:
        meta = json.load(f)
    assert meta["collapse_univariate"] is True
    assert meta["collapse_pl"] is True
    with Path(artifact.debug_path).open() as f:
        sidecar = json.load(f)
    assert sidecar["collapse_univariate"] is True
    assert sidecar["collapse_pl"] is True


def test_no_bias_emission_has_no_bias_tensors(artifacts):
    """No bias initializers exist and no node reads one.

    Each projection is its bare MatMul.
    """
    import onnx

    _, onnx_path = artifacts["biasless"]
    model = onnx.load(onnx_path)
    init_names = {i.name for i in model.graph.initializer} | {
        s.values.name for s in model.graph.sparse_initializer
    }
    bias_names = {
        n
        for n in init_names
        if n.split("_", 1)[-1] in ("b1", "b2", "bgate", "bup", "bdown")
    }
    assert not bias_names, f"bias initializers in a bias=False artifact: {bias_names}"
    for node in model.graph.node:
        for inp in node.input:
            assert not inp.endswith(("_bgate", "_bup", "_bdown", "_b1", "_b2")), (
                f"node {node.name} reads a bias tensor {inp}"
            )


def test_no_bias_logits_match_biased_twin(artifacts):
    """The two emissions of the same graph agree at the logits.

    The folds are the same math, shifted into the matmuls.
    """
    _, biased_path = artifacts["biased"]
    _, biasless_path = artifacts["biasless"]
    out_a, emb = _build()
    ids = _token_ids(emb)
    sess_a = OnnxDebugSession(biased_path, out_a)
    out_b, _ = _build()
    sess_b = OnnxDebugSession(biasless_path, out_b)
    la = sess_a(ids)
    lb = sess_b(ids)
    # Not bit-identical by design: the folds move bias adds into the matmul
    # accumulation — ulp-class shifts amplified by cancellation in the
    # unembed dot products.  Measured on this fixture: max abs 3.7e-4 on
    # ~700-magnitude logits (5e-7 relative); worst relative 8e-5 on small
    # logits.  Tolerance set just above with margin.
    assert torch.allclose(la, lb, rtol=1e-6, atol=1e-3)


def test_no_bias_debug_session_roundtrips(artifacts):
    """probe_compiled over the executing biasless artifact matches the exact oracle.

    debug=True passes residual self-consistency.
    """
    _, onnx_path = artifacts["biasless"]
    out2, emb2 = _build()
    session = OnnxDebugSession(onnx_path, out2)

    ids = _token_ids(emb2)
    report = probe_compiled(
        session,
        out2,
        {"embedding_input": ids},
        n_pos=len(TOKENS),
        atol=_ONNX_PROBE_ATOL,
    )
    assert report.first_divergent is None, report.format_short()

    session(ids, debug=True)
    assert session.debug_value(out2) is not None


def test_sidecar_artifact_mode_mismatch_trips(artifacts, tmp_path):
    """A biasless model paired with a biased compile's sidecars must trip the check.

    That's the emission-mode cross-check (the fingerprint cannot see the
    flag).
    """
    _, biased_path = artifacts["biased"]
    _, biasless_path = artifacts["biasless"]
    # Assemble a mismatched pair: the biased export's sidecars next to the
    # biasless model file.
    target = tmp_path / "mismatch.onnx"
    shutil.copy(biasless_path, target)
    shutil.copy(meta_path_for(biased_path), meta_path_for(str(target)))
    shutil.copy(
        biased_path.replace(".onnx", ".debug.json"),
        str(target).replace(".onnx", ".debug.json"),
    )
    out2, _ = _build()
    with pytest.raises(ValueError, match="emission-mode mismatch"):
        OnnxDebugSession(str(target), out2)


def test_custom_hf_profile_refuses_biasless_compile(tmp_path):
    """The custom HF model has biased ReLU linears and refuses bias=False."""
    from torchwright.compiler.hf import compile_to_hf

    out, emb = _build(machine="relu")
    with pytest.raises(ValueError, match="requires bias=True"):
        compile_to_hf(
            out,
            emb,
            d=D,
            d_head=D_HEAD,
            max_seq_len=16,
            architecture="custom",
            bias=False,
        )
