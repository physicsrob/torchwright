"""Compiled-correctness tests for the FFN node.

A graph built natively from :func:`linear_relu_linear` (which returns a
:class:`FFN`) compiles cleanly and ``probe_compiled`` agrees with the oracle,
on both the heuristic and CP-SAT schedules.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN
from torchwright.graph.asserts import assert_in_range
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear


def _build():
    """A two-FFN graph (FFN feeding FFN) — new node ids each call."""
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


def test_heuristic_matches_cpsat(xt):
    _, out0 = _build()
    c0 = compile_headless(out0, d=D, d_head=D_HEAD)

    _, out1 = _build()
    c1 = compile_headless(out1, d=D, d_head=D_HEAD, optimize=1)

    assert torch.allclose(c0(xt), c1(xt), atol=1e-4)


def test_probe_clean_on_block_graph(xt):
    _, out_block = _build()
    compiled = compile_headless(out_block, d=D, d_head=D_HEAD)
    report = probe_compiled(compiled, out_block, {"x": xt}, N_POS, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_debug_value_and_assert_on_block_output(xt):
    _, out_block = _build()
    assert isinstance(out_block, FFN)
    wrapped = assert_in_range(out_block, -1000.0, 1000.0)
    compiled = compile_headless(wrapped, d=D, d_head=D_HEAD)

    # debug=True runs the residual self-consistency check and the assert
    # predicate on the compiled FFN value; both must pass.
    compiled(xt, debug=True)
    val = compiled.debug_value(out_block)
    ref = out_block.compute(N_POS, {"x": xt})
    assert val is not None
    assert torch.allclose(val, ref, atol=1e-3)


def _build_swish(gated_second=True):
    """A two-FFN swish graph (gated feeding degenerate-or-gated) — new node
    ids each call.  Directly-authored fixture (plan phase A: no swiglu ops
    exist yet).
    """
    x = create_input("x", 8, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(21)
    h = FFN(
        x,
        gate_proj=torch.randn(10, 8, generator=g) * 0.3,
        gate_bias=torch.randn(10, generator=g) * 0.1,
        out_proj=torch.randn(10, 8, generator=g) * 0.3,
        out_bias=torch.randn(8, generator=g) * 0.1,
        up_proj=torch.randn(10, 8, generator=g) * 0.3,
        up_bias=torch.randn(10, generator=g) * 0.1,
        activation="swish",
        name="swish1",
    )
    kwargs = {}
    if gated_second:
        kwargs["up_proj"] = torch.randn(6, 8, generator=g) * 0.3
        kwargs["up_bias"] = torch.randn(6, generator=g) * 0.1
    out = FFN(
        h,
        gate_proj=torch.randn(6, 8, generator=g) * 0.3,
        gate_bias=torch.randn(6, generator=g) * 0.1,
        out_proj=torch.randn(6, 4, generator=g) * 0.3,
        out_bias=torch.randn(4, generator=g) * 0.1,
        activation="swish",
        name="swish2",
        **kwargs,
    )
    return x, out


def test_swish_graph_compiles_to_gated_machine(xt):
    """A pure-swish graph selects the gated MLP sublayer and matches the
    oracle everywhere (probe_compiled), gated and degenerate lanes alike.
    """
    from torchwright.compiler.groups.mlp_sublayer import GatedMLPSubLayer

    for gated_second in (True, False):
        _, out = _build_swish(gated_second=gated_second)
        # verbose=True on one variant exercises the swish param-accounting
        # path (3-matrix layer capacity, per-slot cost, trim savings).
        compiled = compile_headless(out, d=D, d_head=D_HEAD, verbose=gated_second)
        net = compiled._net
        assert net.activation == "swish"
        assert all(isinstance(layer.mlp, GatedMLPSubLayer) for layer in net.layers)
        report = probe_compiled(compiled, out, {"x": xt}, N_POS, atol=1e-3)
        assert report.first_divergent is None, report.format_short()


def test_swish_debug_forward_and_debug_value(xt):
    """debug=True self-consistency + assert predicates hold on the swish
    machine; debug_value extracts the FFN's compiled value.
    """
    _, out = _build_swish()
    wrapped = assert_in_range(out, -1000.0, 1000.0)
    compiled = compile_headless(wrapped, d=D, d_head=D_HEAD)

    compiled(xt, debug=True)
    val = compiled.debug_value(out)
    ref = out.compute(N_POS, {"x": xt})
    assert val is not None
    assert torch.allclose(val, ref, atol=1e-3)


def test_mixed_activation_graph_rejected(xt):
    """No mixed networks: a graph with both ReLU and swish FFNs is a compile
    error (uniformity check, swiglu plan A3).
    """
    x = create_input("x", 8, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(5)
    relu_h = linear_relu_linear(
        x,
        torch.randn(6, 8, generator=g) * 0.3,
        torch.randn(6, generator=g) * 0.1,
        torch.randn(6, 8, generator=g) * 0.3,
        torch.randn(8, generator=g) * 0.1,
        name="relu_ffn",
    )
    swish_out = FFN(
        relu_h,
        gate_proj=torch.randn(6, 8, generator=g) * 0.3,
        gate_bias=torch.randn(6, generator=g) * 0.1,
        out_proj=torch.randn(6, 4, generator=g) * 0.3,
        out_bias=torch.randn(4, generator=g) * 0.1,
        activation="swish",
        name="swish_ffn",
    )
    with pytest.raises(ValueError, match="uniformly one machine"):
        compile_headless(swish_out, d=D, d_head=D_HEAD)


def test_relu_gated_ffn_rejected(xt):
    """A ReLU FFN carrying gated lanes has no physical substrate — rejected
    at the compile boundary, not deep in the weight writer.
    """
    x = create_input("x", 8, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(6)
    blk = FFN(
        x,
        gate_proj=torch.randn(6, 8, generator=g) * 0.2,
        gate_bias=torch.randn(6, generator=g) * 0.1,
        out_proj=torch.randn(6, 4, generator=g) * 0.2,
        out_bias=torch.randn(4, generator=g) * 0.1,
        up_proj=torch.randn(6, 8, generator=g) * 0.2,
        up_bias=torch.randn(6, generator=g) * 0.1,
        activation="relu",
    )
    with pytest.raises(ValueError, match="no up"):
        compile_headless(blk, d=D, d_head=D_HEAD)
