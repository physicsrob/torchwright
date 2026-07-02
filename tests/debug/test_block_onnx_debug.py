"""ONNX debug-surface parity for a graph with Block nodes.

Confirms the :class:`~torchwright.graph.block.Block` node type round-trips the
debug sidecar / canonical-id remap / rebuild fingerprint: a token graph built
natively with Blocks, exported with :func:`compile_to_onnx`, then reopened as an
:class:`OnnxDebugSession` over a *freshly rebuilt* graph, gives a clean
``probe_compiled`` verdict and a passing ``debug=True`` step.
"""

import pytest
import torch

from torchwright.compiler.export import compile_to_onnx
from torchwright.debug.probe import probe_compiled
from torchwright.graph import Block
from torchwright.ops.inout_nodes import create_embedding
from torchwright.ops.linear_relu_linear import linear_relu_linear

onnxruntime = pytest.importorskip("onnxruntime")

from torchwright.debug.onnx_debug import OnnxDebugSession  # noqa: E402

D = 256
D_HEAD = 16
VOCAB = list("0123456789+") + ["\n", "<bos>", "<eos>", "default"]
TOKENS = ["<bos>", "1", "+", "2"]
_ONNX_PROBE_ATOL = 2.5e-3


def _build():
    """A token graph (embedding -> L/R/L, output d_embed-wide) — new ids each call."""
    emb = create_embedding(vocab=VOCAB)
    d = len(emb)
    g = torch.Generator().manual_seed(3)
    out = linear_relu_linear(
        emb,
        torch.randn(24, d, generator=g) * 0.2,
        torch.randn(24, generator=g) * 0.1,
        torch.randn(24, d, generator=g) * 0.2,
        torch.randn(d, generator=g) * 0.1,
        name="ffn",
    )
    return out, emb


def _token_ids(emb) -> torch.Tensor:
    return torch.tensor(
        [[emb.tokenizer.get_token_id(t)] for t in TOKENS], dtype=torch.long
    )


def test_onnx_debug_session_roundtrips_block_graph(tmp_path):
    onnx_path = str(tmp_path / "blk.onnx")
    out_block, emb = _build()
    assert isinstance(out_block, Block)
    compile_to_onnx(
        out_block, emb, onnx_path, d=D, d_head=D_HEAD, max_seq_len=16, verbose=False
    )

    # Fresh rebuild: different node ids, same canonical topology.
    out2_block, emb2 = _build()
    session = OnnxDebugSession(onnx_path, out2_block)

    ids = _token_ids(emb2)
    report = probe_compiled(
        session,
        out2_block,
        {"embedding_input": ids},
        n_pos=len(TOKENS),
        atol=_ONNX_PROBE_ATOL,
    )
    assert report.first_divergent is None, report.format_short()

    # debug=True step over the real artifact: residual self-consistency passes.
    session(ids, debug=True)
    val = session.debug_value(out2_block)
    assert val is not None
