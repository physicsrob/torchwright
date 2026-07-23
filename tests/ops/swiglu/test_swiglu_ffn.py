"""The swiglu_ffn builder: returns a swish FFN, degenerate vs gated by
``up_proj`` presence, and pure-swish graphs built from it compile clean
(``compile_headless`` + ``probe_compiled``).
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.swiglu import swiglu_ffn

N_POS = 4
D = 64
D_HEAD = 8


def _build(gated_second=True):
    """A two-FFN swish graph (gated feeding degenerate-or-gated) — new
    node ids each call.
    """
    x = create_input("x", 8, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(31)
    h = swiglu_ffn(
        x,
        torch.randn(10, 8, generator=g) * 0.3,
        torch.randn(10, generator=g) * 0.1,
        torch.randn(10, 8, generator=g) * 0.3,
        torch.randn(8, generator=g) * 0.1,
        up_proj=torch.randn(10, 8, generator=g) * 0.3,
        up_bias=torch.randn(10, generator=g) * 0.1,
        name="gated",
    )
    kwargs = {}
    if gated_second:
        kwargs["up_proj"] = torch.randn(6, 8, generator=g) * 0.3
        kwargs["up_bias"] = torch.randn(6, generator=g) * 0.1
    out = swiglu_ffn(
        h,
        torch.randn(6, 8, generator=g) * 0.3,
        torch.randn(6, generator=g) * 0.1,
        torch.randn(6, 4, generator=g) * 0.3,
        torch.randn(4, generator=g) * 0.1,
        name="second",
        **kwargs,
    )
    return x, out


@pytest.fixture
def xt():
    g = torch.Generator().manual_seed(99)
    return torch.randn(N_POS, 8, generator=g)


def test_returns_swish_ffn():
    _, out = _build()
    assert isinstance(out, FFN)
    assert out.activation == "swish"


def test_degenerate_vs_gated_by_up_proj_presence():
    _, gated = _build(gated_second=True)
    assert not gated.is_degenerate
    assert gated.up_proj is not None and gated.up_bias is not None

    _, degen = _build(gated_second=False)
    assert degen.is_degenerate
    assert degen.up_proj is None and degen.up_bias is None


def test_up_proj_requires_up_bias():
    x = create_input("x", 8, value_range=(-1.0, 1.0))
    with pytest.raises(ValueError, match="up_proj and up_bias"):
        swiglu_ffn(
            x,
            torch.zeros(4, 8),
            torch.zeros(4),
            torch.zeros(4, 2),
            torch.zeros(2),
            up_proj=torch.zeros(4, 8),
        )


def test_scalar_shapes_unsqueeze():
    """1-D projections / 0-D biases are accepted, as in linear_relu_linear."""
    x = create_input("x", 1, value_range=(-1.0, 1.0))
    out = swiglu_ffn(
        x,
        torch.ones(1),
        torch.tensor(0.0),
        torch.ones(1),
        torch.zeros(1),
        up_proj=torch.ones(1),
        up_bias=torch.tensor(0.0),
    )
    assert out.n_lanes == 1
    # Swish(x) * x at x=1: 1*sigmoid(1)*1
    val = out.compute(1, {"x": torch.ones(1, 1)})
    assert torch.allclose(val, torch.sigmoid(torch.ones(1, 1)))


@pytest.mark.parametrize("gated_second", [True, False])
def test_compiles_clean(xt, gated_second):
    _, out = _build(gated_second=gated_second)
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    report = probe_compiled(compiled, out, {"x": xt}, N_POS, atol=1e-3)
    assert report.first_divergent is None, report.format_short()
