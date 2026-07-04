"""Tests proving the MLP intermediate width (``d_hidden``) is decoupled
from the residual stream width (``d``).

The "smoking gun" is :func:`test_compile_with_d_hidden_larger_than_d`,
which compiles a graph whose ``L1->ReLU->L2`` chain is wider than the
residual stream — impossible before the decoupling.
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.groups.mlp_sublayer import MLPSubLayer
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear


def _build_relu_chain_graph(d_input: int, d_hidden_chain: int, d_output: int):
    """Build a tiny graph: input -> Linear -> ReLU -> Linear with random weights."""
    inp = create_input("x", d_input)
    torch.manual_seed(0)
    input_proj = torch.randn(d_hidden_chain, d_input)
    input_bias = torch.randn(d_hidden_chain)
    output_proj = torch.randn(d_hidden_chain, d_output)
    output_bias = torch.randn(d_output)
    out = linear_relu_linear(
        inp, input_proj, input_bias, output_proj, output_bias, name="chain"
    )
    return out, inp


# ---------------------------------------------------------------------------
# 1. Component-level shape check
# ---------------------------------------------------------------------------


def test_mlp_sublayer_rectangular_shapes():
    """``MLPSubLayer(d, d_hidden)`` allocates rectangular weight matrices
    and forwards correctly."""
    mlp = MLPSubLayer(d=32, d_hidden=8)
    assert mlp.linear1.output_matrix.shape == (32, 8)
    assert mlp.linear1.output_bias.shape == (8,)
    assert mlp.linear2.output_matrix.shape == (8, 32)
    assert mlp.linear2.output_bias.shape == (32,)
    assert mlp.relu.d_hidden == 8

    out = mlp.forward(torch.randn(5, 32))
    assert out.shape == (5, 32)


# ---------------------------------------------------------------------------
# 2. d_hidden < d compiles correctly (memory-savings case)
# ---------------------------------------------------------------------------


def test_compile_with_small_d_hidden():
    out_node, inp = _build_relu_chain_graph(d_input=4, d_hidden_chain=4, d_output=2)

    module = compile_headless(
        out_node,
        d=32,
        d_head=8,
        d_hidden=8,
        device="cpu",
        verbose=False,
    )

    assert module._net.d == 32
    assert module._net.d_hidden == 8
    # All MLP weight matrices in every compiled layer should be (d, d_hidden) /
    # (d_hidden, d) — never (d, d).  trim_unused_slots() shrinks d_hidden to
    # the number of slots actually used; the chain only uses 4, so the
    # per-layer width trims down to at most 4 (but is still ≤ the requested
    # d_hidden=8 and still decoupled from d=32).
    for layer in module._net.layers:
        assert 1 <= layer.mlp.linear1.output_matrix.shape[1] <= 8
        assert layer.mlp.linear1.output_matrix.shape[0] == 32
        assert (
            layer.mlp.linear2.output_matrix.shape[0]
            == layer.mlp.linear1.output_matrix.shape[1]
        )
        assert layer.mlp.linear2.output_matrix.shape[1] == 32

    x = torch.tensor([[0.5, -1.0, 2.0, 0.25]])
    expected = out_node.compute(n_pos=1, input_values={"x": x})
    actual = module(x)
    assert torch.allclose(
        actual, expected, atol=1e-3
    ), f"max diff: {(actual - expected).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# 3. The "smoking gun": d_hidden larger than d
# ---------------------------------------------------------------------------


def test_compile_with_d_hidden_larger_than_d():
    """A chain whose hidden width exceeds ``d``.  Before the decoupling
    the scheduler would reject the chain because ``next_slot + d_hidden
    > self.d`` (the per-layer pool was the residual stream itself)."""
    out_node, inp = _build_relu_chain_graph(d_input=4, d_hidden_chain=48, d_output=2)

    # d=32 is smaller than the chain's hidden width (48) — impossible
    # before the decoupling (the scheduler's pool was ``self.d``).
    module = compile_headless(
        out_node,
        d=32,
        d_head=8,
        d_hidden=64,
        device="cpu",
        verbose=False,
    )

    assert module._net.d == 32
    assert module._net.d_hidden == 64
    # trim_unused_slots() trims trailing zeros; the chain fills 48 of the 64
    # slots, so each layer's per-layer width lands in (32, 64].  The smoking
    # gun is still intact: every layer's d_hidden > d, which was impossible
    # before the decoupling.
    for layer in module._net.layers:
        assert layer.mlp.linear1.output_matrix.shape[0] == 32
        assert (
            32 < layer.mlp.linear1.output_matrix.shape[1] <= 64
        ), "per-layer d_hidden must exceed d=32 to exercise the decoupling"
        assert (
            layer.mlp.linear2.output_matrix.shape[0]
            == layer.mlp.linear1.output_matrix.shape[1]
        )
        assert layer.mlp.linear2.output_matrix.shape[1] == 32

    x = torch.tensor([[0.5, -1.0, 2.0, 0.25]])
    expected = out_node.compute(n_pos=1, input_values={"x": x})
    actual = module(x)
    assert torch.allclose(
        actual, expected, atol=1e-3
    ), f"max diff: {(actual - expected).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# 4. Default d_hidden=None is byte-identical to d_hidden=d
# ---------------------------------------------------------------------------


def test_compile_default_d_hidden_equals_d():
    out_node, _ = _build_relu_chain_graph(d_input=4, d_hidden_chain=4, d_output=2)

    net_default = forward_compile(
        d=32,
        d_head=8,
        output_node=out_node,
        device="cpu",
        verbose=False,
    )
    net_explicit = forward_compile(
        d=32,
        d_head=8,
        d_hidden=32,
        output_node=out_node,
        device="cpu",
        verbose=False,
    )

    assert net_default.d_hidden == net_default.d == 32
    assert net_explicit.d_hidden == net_explicit.d == 32
    assert len(net_default.layers) == len(net_explicit.layers)

    for la, lb in zip(net_default.layers, net_explicit.layers):
        for attr in ("output_matrix", "output_bias"):
            assert torch.equal(
                getattr(la.mlp.linear1, attr), getattr(lb.mlp.linear1, attr)
            )
            assert torch.equal(
                getattr(la.mlp.linear2, attr), getattr(lb.mlp.linear2, attr)
            )
        assert torch.equal(la.attn.attn.query_matrix, lb.attn.attn.query_matrix)
        assert torch.equal(la.attn.attn.key_matrix, lb.attn.attn.key_matrix)
        assert torch.equal(la.attn.attn.value_matrix, lb.attn.attn.value_matrix)
        assert torch.equal(la.attn.attn.output_matrix, lb.attn.attn.output_matrix)
