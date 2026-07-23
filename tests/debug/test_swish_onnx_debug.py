"""ONNX export + debug surface for the swish (gated) machine — swiglu plan A4.

A pure-swish token graph exports through :func:`compile_to_onnx` with the
gated MLP emission (``l{i}_Wgate/Wup/Wdown`` + ``Sigmoid``·``Mul``), records
the machine kind in the artifact / meta / debug sidecar, and round-trips the
debug surface: an :class:`OnnxDebugSession` over a fresh rebuild gives a
clean ``probe_compiled`` verdict (the artifact itself executes under
onnxruntime, so this exercises the real gated emission end to end) and a
passing ``debug=True`` step.  A same-shape ReLU rebuild — which the frozen
topology fingerprint cannot distinguish — trips the explicit machine check.
"""

import json

import pytest
import torch

from torchwright.compiler.export import compile_to_onnx, meta_path_for
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN
from torchwright.ops.inout_nodes import create_embedding
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

onnxruntime = pytest.importorskip("onnxruntime")

from torchwright.debug.onnx_debug import OnnxDebugSession  # noqa: E402

D = 256
D_HEAD = 16
VOCAB = list("0123456789+") + ["\n", "<bos>", "<eos>", "default"]
TOKENS = ["<bos>", "1", "+", "2"]
_ONNX_PROBE_ATOL = 2.5e-3


def _build(machine="swish"):
    """A token graph: embedding -> gated FFN -> degenerate FFN, output
    d_embed-wide.  ``machine="relu"`` builds the same-shape ReLU twin (both
    FFNs degenerate) for the machine-mismatch negative test — identical
    topology under the wrapper-transparent fingerprint.
    """
    emb = create_embedding(vocab=VOCAB)
    d = len(emb)
    g = torch.Generator().manual_seed(17)
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
        h = linear_relu_linear(
            emb,
            torch.randn(24, d, generator=g) * 0.2,
            torch.randn(24, generator=g) * 0.1,
            torch.randn(24, d, generator=g) * 0.2,
            torch.randn(d, generator=g) * 0.1,
            name="gated",
        )
        out = linear_relu_linear(
            h,
            torch.randn(16, d, generator=g) * 0.2,
            torch.randn(16, generator=g) * 0.1,
            torch.randn(16, d, generator=g) * 0.2,
            torch.randn(d, generator=g) * 0.1,
            name="degenerate",
        )
    return out, emb


def _token_ids(emb) -> torch.Tensor:
    return torch.tensor(
        [[emb.tokenizer.get_token_id(t)] for t in TOKENS], dtype=torch.long
    )


@pytest.fixture(scope="module")
def swish_artifact(tmp_path_factory):
    onnx_path = str(tmp_path_factory.mktemp("swish_onnx") / "swish.onnx")
    out, emb = _build()
    artifact = compile_to_onnx(
        out, emb, onnx_path, d=D, d_head=D_HEAD, max_seq_len=16, verbose=False
    )
    return artifact, onnx_path


def test_swish_artifact_records_machine_kind(swish_artifact):
    artifact, onnx_path = swish_artifact
    assert artifact.activation == "swish"
    with open(meta_path_for(onnx_path)) as f:
        assert json.load(f)["activation"] == "swish"
    with open(artifact.debug_path) as f:
        assert json.load(f)["activation"] == "swish"


def test_swish_onnx_emission_is_gated(swish_artifact):
    """The artifact carries the gated emission: gate/up/down initializers and
    Sigmoid·Mul swish, no Relu nodes, no W1/W2.
    """
    import onnx

    _, onnx_path = swish_artifact
    model = onnx.load(onnx_path)
    init_names = {i.name for i in model.graph.initializer} | {
        s.values.name for s in model.graph.sparse_initializer
    }
    assert "l0_Wgate" in init_names
    assert "l0_Wup" in init_names
    assert "l0_Wdown" in init_names
    assert not any(n.startswith("l0_W1") for n in init_names)
    op_types = {n.op_type for n in model.graph.node}
    assert "Sigmoid" in op_types
    assert "Relu" not in op_types


def test_swish_onnx_debug_session_roundtrips(swish_artifact):
    """probe_compiled over the executing artifact matches the exact oracle;
    debug=True passes residual self-consistency.
    """
    _, onnx_path = swish_artifact
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


def test_relu_rebuild_trips_machine_check(swish_artifact):
    """A same-shape ReLU rebuild passes the (frozen, activation-blind)
    topology fingerprint but must trip the explicit machine cross-check.
    """
    _, onnx_path = swish_artifact
    relu_out, _ = _build(machine="relu")
    with pytest.raises(ValueError, match="machine mismatch"):
        OnnxDebugSession(onnx_path, relu_out)
