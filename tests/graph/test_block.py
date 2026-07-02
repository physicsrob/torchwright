"""Tests for the :class:`~torchwright.graph.block.Block` node.

Step 1 (degenerate ReLU lanes): a Block must compute the identical function,
and produce the identical affine bound, as the equivalent
``Linear -> ReLU -> Linear`` subgraph — that equivalence is the whole safety
argument for the block-IR refactor.
"""

import pytest
import torch

from torchwright.graph import Block
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear_relu_linear import linear_relu_linear


def _rand_block_params(d_input, n_lanes, d_output, seed=0):
    g = torch.Generator().manual_seed(seed)
    gate_proj = torch.randn(n_lanes, d_input, generator=g)
    gate_bias = torch.randn(n_lanes, generator=g)
    out_proj = torch.randn(n_lanes, d_output, generator=g)
    out_bias = torch.randn(d_output, generator=g)
    return gate_proj, gate_bias, out_proj, out_bias


def test_block_compute_matches_hand_computed():
    d_input, n_lanes, d_output, n_pos = 3, 4, 2, 5
    x = create_input("x", d_input, value_range=(-2.0, 2.0))
    gate_proj, gate_bias, out_proj, out_bias = _rand_block_params(
        d_input, n_lanes, d_output
    )
    blk = Block(
        x,
        gate_proj=gate_proj,
        gate_bias=gate_bias,
        out_proj=out_proj,
        out_bias=out_bias,
    )

    xv = torch.randn(n_pos, d_input)
    got = blk.compute(n_pos, {"x": xv})

    # Hand-computed: relu(x @ gate_proj.T + gate_bias) @ out_proj + out_bias
    gate = xv @ gate_proj.t() + gate_bias
    lane = torch.clamp(gate, min=0.0)
    expected = lane @ out_proj + out_bias
    assert torch.allclose(got, expected, atol=1e-6)
    assert got.shape == (n_pos, d_output)
    assert len(blk) == d_output
    assert blk.n_lanes == n_lanes
    assert blk.is_degenerate


def test_block_compute_matches_linear_relu_linear_builder():
    d_input, n_lanes, d_output, n_pos = 6, 8, 3, 7
    gate_proj, gate_bias, out_proj, out_bias = _rand_block_params(
        d_input, n_lanes, d_output, seed=1
    )

    x = create_input("x", d_input, value_range=(-3.0, 3.0))
    # linear_relu_linear takes input_proj (d_hidden, d_input) = gate_proj and
    # output_proj (d_hidden, d_output) = out_proj — same orientation as the
    # Block's gate_proj / out_proj rows — and now returns a Block directly.
    built = linear_relu_linear(x, gate_proj, gate_bias, out_proj, out_bias)
    assert isinstance(built, Block)
    blk = Block(
        x,
        gate_proj=gate_proj,
        gate_bias=gate_bias,
        out_proj=out_proj,
        out_bias=out_bias,
    )

    xv = torch.randn(n_pos, d_input)
    cv = built.compute(n_pos, {"x": xv})
    bv = blk.compute(n_pos, {"x": xv})
    assert torch.equal(cv, bv), "builder Block must match a directly-constructed Block"


def test_block_affine_bound_matches_builder():
    d_input, n_lanes, d_output = 5, 6, 4
    gate_proj, gate_bias, out_proj, out_bias = _rand_block_params(
        d_input, n_lanes, d_output, seed=2
    )
    x = create_input("x", d_input, value_range=(-2.0, 2.0))
    built = linear_relu_linear(x, gate_proj, gate_bias, out_proj, out_bias)
    blk = Block(
        x,
        gate_proj=gate_proj,
        gate_bias=gate_bias,
        out_proj=out_proj,
        out_bias=out_bias,
    )

    cr = built.value_type.value_range
    br = blk.value_type.value_range
    assert br.lo == pytest.approx(cr.lo)
    assert br.hi == pytest.approx(cr.hi)


def test_block_rejects_bad_shapes():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    # gate_proj d_input != len(input)
    with pytest.raises(AssertionError):
        Block(
            x,
            gate_proj=torch.randn(3, 5),
            gate_bias=torch.randn(3),
            out_proj=torch.randn(3, 2),
            out_bias=torch.randn(2),
        )
    # out_proj rows != n_lanes
    with pytest.raises(AssertionError):
        Block(
            x,
            gate_proj=torch.randn(3, 4),
            gate_bias=torch.randn(3),
            out_proj=torch.randn(2, 2),
            out_bias=torch.randn(2),
        )
    # bad activation
    with pytest.raises(ValueError):
        Block(
            x,
            gate_proj=torch.randn(3, 4),
            gate_bias=torch.randn(3),
            out_proj=torch.randn(3, 2),
            out_bias=torch.randn(2),
            activation="gelu",
        )
    # up_proj without up_bias
    with pytest.raises(ValueError):
        Block(
            x,
            gate_proj=torch.randn(3, 4),
            gate_bias=torch.randn(3),
            out_proj=torch.randn(3, 2),
            out_bias=torch.randn(2),
            up_proj=torch.randn(3, 4),
        )


def test_gated_swish_block_compute():
    """A gated swish Block computes the SwiGLU lane math (oracle only — the
    compiler path is degenerate-ReLU this phase; this pins the node's spec)."""
    d_input, n_lanes, d_output, n_pos = 4, 5, 3, 6
    g = torch.Generator().manual_seed(3)
    gate_proj = torch.randn(n_lanes, d_input, generator=g)
    gate_bias = torch.randn(n_lanes, generator=g)
    up_proj = torch.randn(n_lanes, d_input, generator=g)
    up_bias = torch.randn(n_lanes, generator=g)
    out_proj = torch.randn(n_lanes, d_output, generator=g)
    out_bias = torch.randn(d_output, generator=g)
    blk = Block(
        create_input("x", d_input, value_range=(-2.0, 2.0)),
        gate_proj=gate_proj,
        gate_bias=gate_bias,
        out_proj=out_proj,
        out_bias=out_bias,
        up_proj=up_proj,
        up_bias=up_bias,
        activation="swish",
    )
    assert not blk.is_degenerate

    xv = torch.randn(n_pos, d_input)
    got = blk.compute(n_pos, {"x": xv})
    gate = xv @ gate_proj.t() + gate_bias
    up = xv @ up_proj.t() + up_bias
    lane = (gate * torch.sigmoid(gate)) * up
    expected = lane @ out_proj + out_bias
    assert torch.allclose(got, expected, atol=1e-6)
