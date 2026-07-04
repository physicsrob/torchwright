"""Test that the compiler handles Concatenate as the output node.

The scheduler currently deadlocks when the output node is a Concatenate
because Concatenate nodes are filtered out of the remaining-nodes check
but never added to the computed set, so the termination condition
``output_node in computed`` is never satisfied.
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.relu.arithmetic_ops import compare
from torchwright.ops.inout_nodes import create_input


def test_concatenate_output_node():
    """Compilation should succeed when the output node is a Concatenate."""
    a = create_input("a", 1)
    b = create_input("b", 1)

    r1 = compare(a, 0.5)
    r2 = compare(b, 0.5)
    output = Concatenate([r1, r2])

    module = compile_headless(
        output,
        d=512,
        d_head=16,
        verbose=False,
    )

    inp = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    result = module(inp)
    assert result.shape == (2, 2)
    # Both > 0.5 → both true (1.0)
    assert result[0, 0].item() > 0.5
    assert result[0, 1].item() > 0.5
    # Both < 0.5 → both false (-1.0)
    assert result[1, 0].item() < -0.5
    assert result[1, 1].item() < -0.5
