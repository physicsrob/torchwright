"""Attention-head count is independent of residual and per-head widths."""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.forward.cpsat_scheduler import build_cpsat_model
from torchwright.compiler.forward.scheduling_policy import LEGACY_POLICY
from torchwright.compiler.graph_identity import graph_fingerprint
from torchwright.graph import Linear
from torchwright.ops.inout_nodes import create_input


def _graph():
    x = create_input("x", 16, value_range=(-1.0, 1.0))
    weights = torch.arange(32, dtype=torch.float32).reshape(16, 2) / 100.0
    return Linear(x, weights, name="projection")


def test_explicit_n_heads_allows_decoupled_attention_width():
    """The residual width is 30 while the attention width is 3 * 8 = 24."""
    out = _graph()
    module = compile_headless(
        out,
        d=30,
        d_head=8,
        n_heads=3,
        trim_heads=False,
        verbose=False,
        device="cpu",
    )

    assert module._net.d == 30
    assert module._net.d_head == 8
    assert module._net.n_heads == 3
    for layer in module._net.layers:
        attn = layer.attn.attn
        assert attn.n_heads == 3
        assert attn.query_matrix.shape == (3, 30, 8)
        assert attn.output_matrix.shape == (3, 8, 30)

    values = torch.tensor([[0.25] * 16, [-0.5] * 16])
    expected = out.compute(n_pos=2, input_values={"x": values})
    torch.testing.assert_close(module(values), expected, rtol=0, atol=1e-5)


def test_default_n_heads_retains_coupled_geometry():
    out = _graph()
    default = compile_headless(
        out, d=32, d_head=8, trim_heads=False, verbose=False, device="cpu"
    )
    explicit = compile_headless(
        out,
        d=32,
        d_head=8,
        n_heads=4,
        trim_heads=False,
        verbose=False,
        device="cpu",
    )

    assert default._net.n_heads == explicit._net.n_heads == 4
    assert default.n_layers == explicit.n_layers
    assert [layer.attn.attn.query_matrix.shape for layer in default._net.layers] == [
        layer.attn.attn.query_matrix.shape for layer in explicit._net.layers
    ]


def test_explicit_attention_width_may_exceed_residual_width():
    module = compile_headless(
        _graph(),
        d=24,
        d_head=8,
        n_heads=4,
        trim_heads=False,
        verbose=False,
        device="cpu",
    )

    assert module._net.n_heads * module._net.d_head == 32
    assert module._net.d == 24
    assert all(layer.attn.attn.n_heads == 4 for layer in module._net.layers)


@pytest.mark.parametrize("n_heads", [0, -1, 1.5, True])
def test_invalid_explicit_n_heads_rejected(n_heads):
    with pytest.raises(ValueError, match="n_heads must be a positive integer"):
        compile_headless(_graph(), d=30, d_head=8, n_heads=n_heads, verbose=False)


def test_omitted_n_heads_requires_divisible_default_geometry():
    with pytest.raises(ValueError, match="pass n_heads explicitly"):
        compile_headless(_graph(), d=30, d_head=8, verbose=False)


def test_cpsat_capacity_uses_explicit_n_heads():
    built = build_cpsat_model(
        _graph(), d=30, d_head=8, n_heads=3, d_hidden=30, max_layers=8
    )
    assert built.n_heads_per_layer == 3


def test_schedule_fingerprint_includes_decoupled_head_capacity():
    common = {
        "d": 32,
        "d_head": 8,
        "d_hidden": 32,
        "flex_routing": True,
        "cancel_slack": 2,
        "policy": LEGACY_POLICY,
    }
    out = _graph()
    default = graph_fingerprint(out, **common)
    assert default == graph_fingerprint(out, n_heads=4, **common)
    assert default != graph_fingerprint(out, n_heads=3, **common)
