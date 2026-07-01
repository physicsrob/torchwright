"""Compiled-equivalence tests for the Block node and the blockify pass.

The safety contract of Phase 2a: a graph compiled with ``blockify`` produces
bit-identical output to the same graph compiled via the chain-mined path, and
``probe_compiled`` agrees with the oracle on the blockified graph.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import Block
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.blockify import blockify
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear_relu_linear import linear_relu_linear


def _build():
    """A two-MLP graph with a chain feeding a chain — new node ids each call."""
    x = create_input("x", 8, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(11)
    h = linear_relu_linear(
        x,
        torch.randn(16, 8, generator=g) * 0.3,
        torch.randn(16, generator=g) * 0.1,
        torch.randn(16, 8, generator=g) * 0.3,
        torch.randn(8, generator=g) * 0.1,
        name="mlp1",
    )
    out = linear_relu_linear(
        h,
        torch.randn(12, 8, generator=g) * 0.3,
        torch.randn(12, generator=g) * 0.1,
        torch.randn(12, 4, generator=g) * 0.3,
        torch.randn(4, generator=g) * 0.1,
        name="mlp2",
    )
    return x, out


N_POS = 4
D = 64
D_HEAD = 8


@pytest.fixture
def xt():
    g = torch.Generator().manual_seed(99)
    return torch.randn(N_POS, 8, generator=g)


def test_blockify_matches_chain_optimize0(xt):
    _, out_chain = _build()
    c_chain = compile_headless(out_chain, d=D, d_head=D_HEAD)

    _, out2 = _build()
    out_block = blockify(out2)
    c_block = compile_headless(out_block, d=D, d_head=D_HEAD)

    oc = c_chain(xt)
    ob = c_block(xt)
    assert torch.equal(oc, ob), "blockified compile must match the chain-mined compile"


def test_blockify_matches_chain_cpsat(xt):
    _, out_chain = _build()
    c_chain = compile_headless(out_chain, d=D, d_head=D_HEAD)

    _, out2 = _build()
    out_block = blockify(out2)
    c_block = compile_headless(out_block, d=D, d_head=D_HEAD, optimize=1)

    assert torch.allclose(c_chain(xt), c_block(xt), atol=1e-4)


def test_probe_clean_on_blockified_graph(xt):
    _, out = _build()
    out_block = blockify(out)
    compiled = compile_headless(out_block, d=D, d_head=D_HEAD)
    report = probe_compiled(compiled, out_block, {"x": xt}, N_POS, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_debug_value_and_assert_on_block_output(xt):
    _, out = _build()
    out_block = blockify(out)
    assert isinstance(out_block, Block)
    wrapped = assert_in_range(out_block, -1000.0, 1000.0)
    compiled = compile_headless(wrapped, d=D, d_head=D_HEAD)

    # debug=True runs the residual self-consistency check and the assert
    # predicate on the compiled Block value; both must pass.
    compiled(xt, debug=True)
    val = compiled.debug_value(out_block)
    ref = out_block.compute(N_POS, {"x": xt})
    assert val is not None
    assert torch.allclose(val, ref, atol=1e-3)


def test_swish_block_rejected_by_compiler(xt):
    """A gated/swish Block has no ReLU-degenerate lowering this phase; the
    MLP writer's precondition assert fires rather than silently miscompiling."""
    x = create_input("x", 8, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(5)
    blk = Block(
        x,
        gate_proj=torch.randn(6, 8, generator=g) * 0.2,
        gate_bias=torch.randn(6, generator=g) * 0.1,
        out_proj=torch.randn(6, 4, generator=g) * 0.2,
        out_bias=torch.randn(4, generator=g) * 0.1,
        up_proj=torch.randn(6, 8, generator=g) * 0.2,
        up_bias=torch.randn(6, generator=g) * 0.1,
        activation="swish",
    )
    with pytest.raises(AssertionError, match="degenerate ReLU"):
        compile_headless(blk, d=D, d_head=D_HEAD)
