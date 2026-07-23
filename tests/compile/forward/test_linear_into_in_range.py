"""Minimal reproduction of compiler bug: Linear → in_range.

When a Linear node with a non-zero bias feeds into in_range, the compiled
transformer ignores the Linear's bias, producing wrong range masks.

Direct InputNode → in_range works correctly.
Linear(input, weight, bias) → in_range produces wrong results.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.graph.linear import Linear
from torchwright.graph.misc import LiteralValue
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.map_select import in_range


def test_linear_into_in_range():
    """Linear with bias feeding into in_range should preserve the bias."""
    hh = create_input("hh", 1)

    H = 8

    # lower = 4 - hh (Linear with weight=-1 and bias=4)
    lower = Linear(hh, torch.tensor([[-1.0]]), torch.tensor([4.0]))
    upper = LiteralValue(torch.tensor([6.0]))

    masks = in_range(lower, upper, H)

    net = forward_compile(
        d=64,
        d_head=16,
        output_node=masks,
        verbose=False,
    )

    # hh=1.0 → lower=3.0, upper=6.0 → rows 3,4,5 should be in range
    vals = {"hh": torch.tensor([[1.0]])}

    graph_out = masks.compute(1, vals)[0]
    compiled_out = net.compute(1, vals)[masks][0]

    expected = [-1, -1, -1, 1, 1, 1, -1, -1]
    for i in range(H):
        assert abs(graph_out[i].item() - expected[i]) < 0.5, (
            f"Graph mismatch at slot {i}: got {graph_out[i].item()}, "
            f"expected {expected[i]}"
        )
        assert abs(compiled_out[i].item() - expected[i]) < 0.5, (
            f"Compiled mismatch at slot {i}: got {compiled_out[i].item()}, "
            f"expected {expected[i]}. "
            f"Full compiled output: {[f'{v:.1f}' for v in compiled_out.tolist()]}"
        )
