"""Tests for the :func:`~torchwright.graph.blockify.blockify` graph pass."""

import pytest
import torch

from torchwright.graph import Block, Concatenate, Linear, ReLU
from torchwright.graph.asserts import assert_in_range, debug_watch
from torchwright.graph.blockify import blockify
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear_relu_linear import linear_relu_linear


def _chain(x, d_input, n_lanes, d_output, seed=0, name=""):
    g = torch.Generator().manual_seed(seed)
    return linear_relu_linear(
        x,
        torch.randn(n_lanes, d_input, generator=g),
        torch.randn(n_lanes, generator=g),
        torch.randn(n_lanes, d_output, generator=g),
        torch.randn(d_output, generator=g),
        name=name,
    )


def test_blockify_finds_chain_and_rewires_consumer():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    chain = _chain(x, 4, 6, 3, name="c")
    downstream = Linear(chain, torch.randn(3, 2), torch.randn(2))

    xv = torch.randn(5, 4)
    ref = downstream.compute(5, {"x": xv})  # value before blockify

    new_out = blockify(downstream)
    # Output object unchanged (it was the downstream Linear, not the chain L2),
    # but its input is now the Block.
    assert new_out is downstream
    assert isinstance(downstream.inputs[0], Block)

    got = downstream.compute(5, {"x": xv})
    assert torch.allclose(got, ref, atol=1e-6)


def test_blockify_output_is_chain_returns_block():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    chain = _chain(x, 4, 6, 3)
    new_out = blockify(chain)
    assert isinstance(new_out, Block)

    xv = torch.randn(5, 4)
    # The Block reproduces the chain's math.
    ref_chain = _chain(create_input("y", 4, value_range=(-2.0, 2.0)), 4, 6, 3)
    assert torch.allclose(
        new_out.compute(5, {"x": xv}), ref_chain.compute(5, {"y": xv}), atol=1e-6
    )


def test_blockify_multiple_chains():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    c1 = _chain(x, 4, 6, 4, seed=1)
    c2 = _chain(c1, 4, 5, 3, seed=2)
    new_out = blockify(c2)
    assert isinstance(new_out, Block)
    # The Block's input is the first chain's Block (both chains blockified).
    assert isinstance(new_out.inputs[0], Block)


def test_blockify_preserves_external_assert_on_output():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    chain = _chain(x, 4, 6, 3)
    wrapped = assert_in_range(chain, -1000.0, 1000.0)  # external assert on L2 out
    new_out = blockify(wrapped)
    # The external assert stays; its input is now the Block.
    assert new_out is wrapped
    assert isinstance(wrapped.inputs[0], Block)


def test_blockify_non_exclusive_l1_raises():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    l1 = Linear(x, torch.randn(4, 6), torch.randn(6))
    relu = ReLU(l1)
    l2 = Linear(relu, torch.randn(6, 3), torch.randn(3))
    extra = Linear(l1, torch.randn(6, 2), torch.randn(2))  # 2nd consumer of L1
    out = Concatenate([l2, extra])
    with pytest.raises(AssertionError, match="not exclusive"):
        blockify(out)


def test_blockify_internal_assert_on_relu_raises():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    l1 = Linear(x, torch.randn(4, 6), torch.randn(6))
    relu = ReLU(l1)
    guarded = assert_in_range(relu, 0.0, 1000.0)  # assert on chain-internal value
    l2 = Linear(guarded, torch.randn(6, 3), torch.randn(3))
    with pytest.raises(AssertionError, match="chain-internal"):
        blockify(l2)


def test_blockify_internal_watch_on_l1_raises():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    l1 = Linear(x, torch.randn(4, 6), torch.randn(6))
    watched = debug_watch(l1, lambda t: (True, ""))  # watch on L1 output
    relu = ReLU(watched)
    l2 = Linear(relu, torch.randn(6, 3), torch.randn(3))
    with pytest.raises(AssertionError, match="chain-internal"):
        blockify(l2)
