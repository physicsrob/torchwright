"""Tests for the :func:`~torchwright.graph.blockify.blockify` verification pass.

Since Phase 2b, ``blockify`` no longer converts chains — the op layer builds
:class:`Block` nodes directly.  It is now a check: it asserts no raw
``Linear -> ReLU -> Linear`` chain survives in the graph.
"""

import pytest
import torch

from torchwright.graph import Block, Linear, ReLU
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.blockify import blockify
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear_relu_linear import linear_relu_linear


def _block(x, d_input, n_lanes, d_output, seed=0, name=""):
    g = torch.Generator().manual_seed(seed)
    return linear_relu_linear(
        x,
        torch.randn(n_lanes, d_input, generator=g),
        torch.randn(n_lanes, generator=g),
        torch.randn(n_lanes, d_output, generator=g),
        torch.randn(d_output, generator=g),
        name=name,
    )


def test_linear_relu_linear_builds_block_natively():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    node = _block(x, 4, 6, 3, name="c")
    assert isinstance(node, Block)


def test_blockify_passes_on_native_block_graph():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    blk = _block(x, 4, 6, 3, name="c")
    downstream = Linear(blk, torch.randn(3, 2), torch.randn(2))
    # No raw chain anywhere; verification returns the output unchanged.
    assert blockify(downstream) is downstream


def test_blockify_passes_on_stacked_blocks():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    b1 = _block(x, 4, 6, 4, seed=1)
    b2 = _block(b1, 4, 5, 3, seed=2)
    assert blockify(b2) is b2


def test_blockify_raises_on_raw_chain():
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    l1 = Linear(x, torch.randn(4, 6), torch.randn(6))
    relu = ReLU(l1)
    l2 = Linear(relu, torch.randn(6, 3), torch.randn(3))
    with pytest.raises(AssertionError, match="unclaimed"):
        blockify(l2)


def test_blockify_detects_raw_chain_through_internal_wrapper():
    # A wrapper on a chain-internal value does not hide the chain from the
    # detector (Assert/DebugWatch-transparent mining).
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    l1 = Linear(x, torch.randn(4, 6), torch.randn(6))
    relu = ReLU(l1)
    guarded = assert_in_range(relu, 0.0, 1000.0)
    l2 = Linear(guarded, torch.randn(6, 3), torch.randn(3))
    with pytest.raises(AssertionError, match="unclaimed"):
        blockify(l2)
