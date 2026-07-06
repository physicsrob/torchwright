"""The lowering copy is bounds-transparent: an in-place run of the
lowering passes and ``lower()``'s private-copy run agree bit-for-bit.

Two fresh rebuilds of the same graph: one gets the passes applied
**in place** (fuse consecutive Linears, refresh every node's derived
caches, strip wrappers with claim transfer), the other goes through
``lower()`` (clone, then the same passes on the copy).  Comparing
per-node ``value_type`` through canonical ids pins three living
invariants at once:

1. **Clone fidelity** — the copy carries every piece of state that
   feeds bound computation (a ``graph_clone`` edit that drops a cached
   field diverges here).
2. **Wrapper-strip equivalence** — stripping Assert wrappers before vs
   after the bounds refresh yields the same bounds.
3. **Canonical-id stability** — two independent rebuilds of the same
   construction code map node-for-node (the property the schedule
   cache and the debug sidecar rely on).

Maintenance rule: when ``lower()`` gains a structural pass, the
in-place twin (``_inplace_pipeline_bounds``) must apply the same pass —
a key-set mismatch in ``_assert_bit_identical`` means the twin is
stale, not that the compiler broke.  (Historical note: this began as
the old-vs-new migration gate for the lowering copy, L1 of
``docs/lowering_copy_plan.md``; the pre-copy pipeline it replayed no
longer exists, so the twin now tracks ``lower()``'s own pass list.)
"""

import torch

from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.graph_identity import canonical_ids, unwrap_debug
from torchwright.compiler.lower import _strip_debug_wrappers, lower
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph.affine_rules import refresh_node_caches
from torchwright.graph.optimize import fuse_consecutive_linears
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.ffn import FFN
from torchwright.graph.linear import Linear
from torchwright.ops.inout_nodes import create_input


def _inplace_pipeline_bounds(output_node):
    """Apply ``lower()``'s passes in place on a throwaway rebuild.

    Fuse consecutive Linears, refresh caches in topological order, then
    strip wrappers in place with claim transfer — the same passes
    ``lower()`` runs on its private copy, minus the clone.  Must be kept
    in step with ``lower()``'s pass list (see module docstring).
    Mutates its argument — callers pass a rebuild they own.
    """
    fuse_consecutive_linears({output_node})
    for node in topological_order(output_node):
        refresh_node_caches(node)
    stripped, _integer_claimed = _strip_debug_wrappers(output_node)
    canon = canonical_ids(stripped)
    return {
        canon[n.node_id]: n.value_type.value_range
        for n in get_ancestor_nodes({stripped})
    }


def _lowered_pipeline_bounds(output_node):
    lowered = lower(output_node)
    canon = canonical_ids(lowered.output_node)
    return {
        canon[n.node_id]: n.value_type.value_range
        for n in get_ancestor_nodes({lowered.output_node})
    }


def _assert_bit_identical(inplace, lowered):
    # A key-set mismatch means the in-place twin's pass list is stale
    # relative to lower()'s, not a compiler bug — see module docstring.
    assert inplace.keys() == lowered.keys()
    for cid in inplace:
        i, l = inplace[cid], lowered[cid]
        assert (i.lo, i.hi) == (
            l.lo,
            l.hi,
        ), f"canonical node {cid}: in-place run {i} vs lowering copy {l}"


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


def test_bounds_transparent_adder_1digit():
    inplace = _inplace_pipeline_bounds(_adder_1digit())
    lowered = _lowered_pipeline_bounds(_adder_1digit())
    _assert_bit_identical(inplace, lowered)


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


def test_bounds_transparent_swish_graph():
    inplace = _inplace_pipeline_bounds(_swish_graph())
    lowered = _lowered_pipeline_bounds(_swish_graph())
    _assert_bit_identical(inplace, lowered)


# --- Univariate collapse variant (same twin rule, flag on) ----------------

_COLLAPSE_LANE_CAP = 64


def _collapsible_graph():
    """An integer-asserted scalar chain the collapse pass rewrites."""
    from torchwright.graph.asserts import assert_integer
    from torchwright.ops.relu.arithmetic_ops import compare, min as ops_min

    x = create_input("x", 1, value_range=(0.0, 9.0))
    xi = assert_integer(x)
    return ops_min(compare(xi, 2.5), compare(xi, 5.5))


def _inplace_pipeline_bounds_collapsed(output_node):
    """The in-place twin with the collapse pass in lower()'s round
    order: fuse, refresh, strip, collapse, re-strip."""
    from torchwright.compiler.collapse import collapse_univariate_subgraphs
    from torchwright.graph.optimize import FoldLog

    fuse_consecutive_linears({output_node})
    for node in topological_order(output_node):
        refresh_node_caches(node)
    stripped, integer_claimed = _strip_debug_wrappers(output_node)
    stripped, report = collapse_univariate_subgraphs(
        stripped,
        integer_claimed=integer_claimed,
        lane_cap=_COLLAPSE_LANE_CAP,
        fold_log=FoldLog(),
    )
    assert report.n_collapsed, report.format()
    stripped, _ = _strip_debug_wrappers(stripped)
    canon = canonical_ids(stripped)
    return {
        canon[n.node_id]: n.value_type.value_range
        for n in get_ancestor_nodes({stripped})
    }


def test_bounds_transparent_collapsed_graph():
    inplace = _inplace_pipeline_bounds_collapsed(_collapsible_graph())
    lowered_graph = lower(
        _collapsible_graph(),
        collapse_univariate=True,
        collapse_lane_cap=_COLLAPSE_LANE_CAP,
    )
    canon = canonical_ids(lowered_graph.output_node)
    lowered = {
        canon[n.node_id]: n.value_type.value_range
        for n in get_ancestor_nodes({lowered_graph.output_node})
    }
    _assert_bit_identical(inplace, lowered)
