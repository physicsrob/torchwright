"""Node-keyed lookup surfaces resolve through Assert/DebugWatch wrappers.

The source graph keeps its wrappers through compilation (only the
compiler-private copy is stripped), so a source output Concatenate's
children may be wrappers and callers may hand in the wrapped node an
ops helper returned.  D6 smallest-layer repro for the post-copy
KeyError class (surfaced by test_concatenate_output and the
range_printer example tests).
"""

import torch

from torchwright.compiler.residual_assignment import (
    ResidualAssignment,
    ResidualStreamState,
    flatten_concat_nodes,
)
from torchwright.graph import Linear
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.misc import Concatenate
from torchwright.ops.inout_nodes import create_input


def _wrapped_pair():
    x = create_input("x", 2, value_range=(-1.0, 1.0))
    lin = Linear(x, torch.randn(2, 3) * 0.2)
    wrapped = assert_in_range(lin, -10.0, 10.0)
    return x, lin, wrapped


def test_flatten_concat_steps_through_wrappers():
    x, lin, wrapped = _wrapped_pair()
    cat = Concatenate([wrapped, x])
    assert flatten_concat_nodes([cat]) == [lin, x]


def test_assignment_lookups_unwrap():
    x, lin, wrapped = _wrapped_pair()
    st = ResidualStreamState(name="s")
    ra = ResidualAssignment({st})
    ra.assign(st, lin, [4, 5, 6])
    ra.assign(st, x, [0, 1])

    assert ra.has_node(st, wrapped)
    assert ra.get_node_indices(st, wrapped) == [4, 5, 6]
    # A Concatenate with a wrapper child resolves to the wrapped node's cols.
    cat = Concatenate([wrapped, x])
    assert ra.get_node_indices(st, cat) == [4, 5, 6, 0, 1]
