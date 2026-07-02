"""Compiled-equivalence tests for the Block node (Phase 2b native construction).

The safety contract: a graph built natively from :func:`linear_relu_linear`
(which returns a :class:`Block`) compiles to bit-identical output as the same
graph hand-built as a ``Linear -> ReLU -> Linear`` chain, and ``probe_compiled``
agrees with the oracle on the block graph.

The manual-chain arm exercises the scheduler's chain-mining path, which Phase 3
deletes; when it does, the equivalence is re-pinned by the op-noise / oracle
tests and this manual arm is removed.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import Block, Linear, ReLU
from torchwright.graph.asserts import assert_in_range
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear_relu_linear import linear_relu_linear


def _manual_chain(x, input_proj, input_bias, output_proj, output_bias):
    """Hand-built Linear -> ReLU -> Linear with the same weights
    ``linear_relu_linear`` would give its Block."""
    l1 = Linear(x, input_proj.t(), input_bias)
    r = ReLU(l1)
    return Linear(r, output_proj, output_bias)


def _build(as_chain):
    """A two-MLP graph (chain feeding chain) — new node ids each call.  Built as
    native Blocks or as hand-built chains from identical weights."""
    x = create_input("x", 8, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(11)
    w1i = torch.randn(16, 8, generator=g) * 0.3
    b1i = torch.randn(16, generator=g) * 0.1
    w1o = torch.randn(16, 8, generator=g) * 0.3
    b1o = torch.randn(8, generator=g) * 0.1
    w2i = torch.randn(12, 8, generator=g) * 0.3
    b2i = torch.randn(12, generator=g) * 0.1
    w2o = torch.randn(12, 4, generator=g) * 0.3
    b2o = torch.randn(4, generator=g) * 0.1
    if as_chain:
        h = _manual_chain(x, w1i, b1i, w1o, b1o)
        out = _manual_chain(h, w2i, b2i, w2o, b2o)
    else:
        h = linear_relu_linear(x, w1i, b1i, w1o, b1o, name="mlp1")
        out = linear_relu_linear(h, w2i, b2i, w2o, b2o, name="mlp2")
    return x, out


N_POS = 4
D = 64
D_HEAD = 8


@pytest.fixture
def xt():
    g = torch.Generator().manual_seed(99)
    return torch.randn(N_POS, 8, generator=g)


def test_native_block_matches_manual_chain_optimize0(xt):
    _, out_chain = _build(as_chain=True)
    c_chain = compile_headless(out_chain, d=D, d_head=D_HEAD)

    _, out_block = _build(as_chain=False)
    c_block = compile_headless(out_block, d=D, d_head=D_HEAD)

    assert torch.equal(
        c_chain(xt), c_block(xt)
    ), "native Block compile must match the hand-built chain compile"


def test_native_block_matches_manual_chain_cpsat(xt):
    _, out_chain = _build(as_chain=True)
    c_chain = compile_headless(out_chain, d=D, d_head=D_HEAD)

    _, out_block = _build(as_chain=False)
    c_block = compile_headless(out_block, d=D, d_head=D_HEAD, optimize=1)

    assert torch.allclose(c_chain(xt), c_block(xt), atol=1e-4)


def test_probe_clean_on_block_graph(xt):
    _, out_block = _build(as_chain=False)
    compiled = compile_headless(out_block, d=D, d_head=D_HEAD)
    report = probe_compiled(compiled, out_block, {"x": xt}, N_POS, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_debug_value_and_assert_on_block_output(xt):
    _, out_block = _build(as_chain=False)
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
