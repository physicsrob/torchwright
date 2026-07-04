"""Old-vs-new parity gate for the lowering copy (L1 of
``docs/lowering_copy_plan.md``).

The pre-copy pipeline mutated the graph: ``lower()`` refreshed every
node's derived caches in place, then ``GraphAnalyzer`` stripped the
wrappers in place with claim transfer.  Decision 2(a) (clone the
wrappers, run the same strip on the copy) promises first-compile bounds
bit-identical to that pipeline *by construction* — this gate measures
it: run the old mutation sequence on one fresh rebuild, ``lower()`` on
another, and compare per-node ``value_type`` through canonical ids.
"""

import torch

from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.graph_identity import canonical_ids, unwrap_debug
from torchwright.compiler.lower import _strip_debug_wrappers, lower
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph.affine_rules import refresh_node_caches
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.ffn import FFN
from torchwright.graph.linear import Linear
from torchwright.ops.inout_nodes import create_input


def _old_pipeline_bounds(output_node):
    """Reproduce the pre-copy compile's bound state on a throwaway rebuild.

    Refresh caches in topological order (the old ``lower()`` refresh
    loop), then strip wrappers in place with claim transfer (the old
    ``GraphAnalyzer._strip_asserts``, which ``_strip_debug_wrappers`` is
    the verbatim relocation of).  Mutates its argument — callers pass a
    rebuild they own.
    """
    for node in topological_order(output_node):
        refresh_node_caches(node)
    stripped = _strip_debug_wrappers(output_node)
    canon = canonical_ids(stripped)
    return {
        canon[n.node_id]: n.value_type.value_range
        for n in get_ancestor_nodes({stripped})
    }


def _new_pipeline_bounds(output_node):
    lowered = lower(output_node)
    canon = canonical_ids(lowered.output_node)
    return {
        canon[n.node_id]: n.value_type.value_range
        for n in get_ancestor_nodes({lowered.output_node})
    }


def _assert_bit_identical(old, new):
    assert old.keys() == new.keys()
    for cid in old:
        o, n = old[cid], new[cid]
        assert (o.lo, o.hi) == (
            n.lo,
            n.hi,
        ), f"canonical node {cid}: old pipeline {o} vs lowering copy {n}"


def _adder_1digit():
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        from examples.adder import create_network_parts

        output_node, _embedding = create_network_parts()
        return output_node
    finally:
        adder_module.max_digits = original


def test_parity_adder_1digit():
    old = _old_pipeline_bounds(_adder_1digit())
    new = _new_pipeline_bounds(_adder_1digit())
    _assert_bit_identical(old, new)


def _swish_graph():
    """A small swish-machine graph with assert wrappers on both a leaf and
    a general (FFN) target — the two claim channels."""

    def w(*shape, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=g) * 0.4

    x = create_input("x", 4, value_range=(-3.0, 3.0))
    guarded_x = assert_in_range(x, -2.0, 2.0)  # leaf claim
    ffn = FFN(
        guarded_x,
        gate_proj=w(6, 4, seed=1),
        gate_bias=w(6, seed=2),
        out_proj=w(6, 3, seed=3),
        out_bias=w(3, seed=4),
        up_proj=w(6, 4, seed=5),
        up_bias=w(6, seed=6),
        activation="swish",
    )
    guarded_ffn = assert_in_range(ffn, -5.0, 5.0)  # general-target claim
    return Linear(guarded_ffn, w(3, 2, seed=7))


def test_parity_swish_graph():
    old = _old_pipeline_bounds(_swish_graph())
    new = _new_pipeline_bounds(_swish_graph())
    _assert_bit_identical(old, new)
