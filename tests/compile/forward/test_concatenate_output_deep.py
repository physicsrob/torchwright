"""Test that Concatenate output nodes preserve leaf columns until compilation ends.

When the output node is a Concatenate whose leaf children finish at
different depths, the scheduler must keep the shallow leaf's residual
stream columns allocated until the final ResidualAssignment step.
Otherwise `residual_map.get_indices(leaf)` raises KeyError because
the columns were freed while deeper subgraphs were still compiling.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.arithmetic_ops import compare, multiply_2d, reciprocal
from torchwright.ops.relu.map_select import select


def test_concatenate_output_mixed_depth():
    """Concatenate of a shallow and a deep subgraph should compile."""
    # Shallow subgraph (~3 layers)
    a = create_input("a", 1, value_range=(-10.0, 10.0))
    b = create_input("b", 1, value_range=(-10.0, 10.0))
    shallow = select(compare(a, 0.5), a, b)

    # Deep subgraph (~15 layers): two multiply_2ds + reciprocal
    c = create_input("c", 1, value_range=(-10.0, 10.0))
    d = create_input("d", 1, value_range=(-10.0, 10.0))
    p = multiply_2d(c, d, max_abs1=10, max_abs2=10, step1=0.5, step2=0.5)
    p2 = multiply_2d(p, c, max_abs1=100, max_abs2=10, step1=1.0, step2=1.0)
    deep = reciprocal(p2, min_value=0.5, max_value=100, step=1.0)

    output = Concatenate([shallow, deep])

    module = compile_headless(
        output,
        d=1024,
        d_head=16,
        verbose=False,
    )

    # Verify correctness: a=1 > 0.5 so shallow selects a=1; deep = 1/(1*1 * 1) = 1.0
    inp = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    result = module(inp)
    assert result.shape == (1, 2)
    assert result[0, 0].item() == pytest.approx(1.0, abs=0.2)  # shallow = a = 1.0
    assert result[0, 1].item() == pytest.approx(1.0, abs=0.2)  # deep = 1/1.0 = 1.0
