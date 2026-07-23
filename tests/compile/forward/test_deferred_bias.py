"""Tests for deferred-bias correctness across MLP operation types.

When a biased Linear is compiled in the attention sublayer, its bias is
deferred to the MLP's output bias (compute_bias). MLP operations that
read from the biased Linear's residual columns in the same layer see
the pre-bias value via mlp.linear1. These tests verify that the compiler
handles this interaction correctly for each MLP op type.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.graph import Add, Linear
from torchwright.graph.misc import LiteralValue
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.map_select import in_range


def test_block_with_two_biased_linear_inputs():
    """FFN (in_range) whose Concatenate input has two biased Linears.

    Both biases must be folded into the FFN's gate bias.
    """
    a = create_input("a", 1)
    b = create_input("b", 1)

    # Two biased Linears
    lower = Linear(a, torch.tensor([[1.0]]), torch.tensor([10.0]))  # a + 10
    upper = Linear(b, torch.tensor([[1.0]]), torch.tensor([20.0]))  # b + 20

    H = 8
    masks = in_range(lower, upper, H)

    net = forward_compile(
        d=64,
        d_head=16,
        output_node=masks,
        verbose=False,
    )

    # a=-5, b=-12 → lower=5, upper=8 → positions 5,6,7 in range
    # If biases dropped: lower=-5, upper=-12 → empty range → all -1
    vals = {"a": torch.tensor([[-5.0]]), "b": torch.tensor([[-12.0]])}

    graph_out = masks.compute(1, vals)[0]
    compiled_out = net.compute(1, vals)[masks][0]

    expected = [-1, -1, -1, -1, -1, 1, 1, 1]
    for i in range(H):
        assert abs(graph_out[i].item() - expected[i]) < 0.5, (
            f"Graph mismatch at slot {i}: got {graph_out[i].item():.1f}, "
            f"expected {expected[i]}"
        )
        assert abs(compiled_out[i].item() - expected[i]) < 0.5, (
            f"Compiled mismatch at slot {i}: got {compiled_out[i].item():.1f}, "
            f"expected {expected[i]}"
        )


def test_biased_linear_fanout_block_and_add():
    """Biased Linear consumed by both an FFN and an Add.

    Verifies that the FFN gate-bias fold and compute_bias coexist correctly:
    - FFN must see the bias (via fold into its gate bias)
    - Residual stream must have the bias (via compute_bias for the Add)
    Neither should double-apply the bias.

    Expected output = 3. Diagnostic values if something breaks:
    - 13  → fold missing (chain sees y=0 instead of y=5)
    - -2  → compute_bias missing (Add sees y=0 instead of y=5)
    - -7  → bias doubled in fold (chain sees y=10 instead of y=5)
    """
    x = create_input("x", 1)

    # biased Linear: y = x + 5 (consumed by both chain and Add)
    y = Linear(x, torch.tensor([[1.0]]), torch.tensor([5.0]))
    upper = LiteralValue(torch.tensor([10.0]))
    n_slots = 12

    # Path 1: chain (in_range uses y as lower bound)
    masks = in_range(y, upper, n_slots)

    # Path 2: collapse masks to scalar, then add y
    mask_sum = Linear(masks, torch.ones(n_slots, 1), torch.zeros(1))
    output = Add(mask_sum, y)

    net = forward_compile(
        d=64,
        d_head=16,
        output_node=output,
        verbose=False,
    )

    # x=0 → y=5
    # in_range(5, 10, 12) → positions 5-9 in range: 5x(+1) + 7x(-1) = -2
    # output = mask_sum + y = -2 + 5 = 3
    vals = {"x": torch.tensor([[0.0]])}
    graph_out = output.compute(1, vals)[0]
    compiled_out = net.compute(1, vals)[output][0]

    assert abs(graph_out[0].item() - 3.0) < 0.5, (
        f"Graph: expected 3, got {graph_out[0].item():.1f}"
    )
    assert abs(compiled_out[0].item() - 3.0) < 0.5, (
        f"Compiled: expected 3, got {compiled_out[0].item():.1f}. "
        f"13 → fold missing, -2 → compute_bias missing, -7 → bias doubled"
    )
