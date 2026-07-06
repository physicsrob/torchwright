"""Clone-pass completeness (:mod:`torchwright.compiler.graph_clone`).

The clone pass is the foundation of compile-as-a-pure-function
(``docs/lowering_copy_plan.md``): a semantic field the clone forgets is a
silent miscompile, so these tests compare representative variant
instances of every vocabulary type across (i) topology, (ii) every
``__dict__`` field including the underscored caches, and (iii) behavior
(``value_type``, affine scalar range, ``compute()`` on random inputs).
"""

import pytest
import torch

from torchwright.compiler.graph_clone import (
    GraphCloneError,
    build_clone_dispatch,
    clone_graph,
    topological_order,
)
from torchwright.compiler.graph_identity import topology_entries
from torchwright.compiler.lower import VOCABULARY
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import Node
from torchwright.graph.asserts import assert_in_range, debug_watch
from torchwright.graph.attn import Attn
from torchwright.graph.embedding import Embedding
from torchwright.graph.ffn import FFN
from torchwright.graph.linear import Linear
from torchwright.graph.misc import (
    Add,
    Concatenate,
    LiteralValue,
    Placeholder,
    ValueLogger,
)
from torchwright.ops.inout_nodes import create_input

DISPATCH = build_clone_dispatch(VOCABULARY)


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _variant_graph():
    """One graph containing a representative variant of every vocabulary type.

    Machine uniformity (all-relu vs all-swish FFNs) is a *compile* rule,
    not a clone rule, so both FFN variants coexist here deliberately.
    """
    x = create_input("x", 4, value_range=(-2.0, 2.0))
    y = create_input("y", 4, value_range=(-1.0, 1.0))
    lit = LiteralValue(torch.tensor([0.5, -1.5, 2.0, 0.0]), name="lit")
    added = Add(x, y, name="added")
    cat = Concatenate([added, lit])
    lin = Linear(cat, _rand(8, 6, seed=1) * 0.3, _rand(6, seed=2) * 0.1, name="lin")
    guarded = assert_in_range(lin, -100.0, 100.0)

    ffn_relu = FFN(
        guarded,
        gate_proj=_rand(5, 6, seed=3) * 0.3,
        gate_bias=_rand(5, seed=4) * 0.1,
        out_proj=_rand(5, 4, seed=5) * 0.3,
        out_bias=_rand(4, seed=6) * 0.1,
        name="ffn_relu",
    )
    ffn_gated = FFN(
        lin,
        gate_proj=_rand(5, 6, seed=7) * 0.3,
        gate_bias=_rand(5, seed=8) * 0.1,
        out_proj=_rand(5, 4, seed=9) * 0.3,
        out_bias=_rand(4, seed=10) * 0.1,
        up_proj=_rand(5, 6, seed=11) * 0.3,
        up_bias=_rand(5, seed=12) * 0.1,
        activation="swish",
        name="ffn_gated",
    )

    attn_full = Attn(
        lin,
        lin,
        lin,
        query_matrix=_rand(6, 4, seed=13) * 0.3,
        key_matrix=_rand(6, 4, seed=14) * 0.3,
        value_matrix=_rand(6, 4, seed=15) * 0.3,
        output_matrix=_rand(4, 4, seed=16) * 0.3,
    )
    attn_partial = Attn(
        lin,
        lin,
        lin,
        query_matrix=_rand(6, 4, seed=17) * 0.3,
        key_matrix=_rand(6, 4, seed=18) * 0.3,
        value_matrix=_rand(6, 4, seed=19) * 0.3,
        output_matrix=_rand(4, 4, seed=20) * 0.3,
        rope_d_rot=2,
    )

    emb = Embedding(vocab=["a", "b", "c"], d_embed=8)
    emb_lin = Linear(emb, _rand(8, 4, seed=21) * 0.2, name="emb_lin")

    watched = debug_watch(ffn_gated, lambda t: (True, ""), "never fires")
    logged = ValueLogger(attn_partial, name="vlog")

    out = Concatenate([ffn_relu, watched, attn_full, logged, emb_lin])

    # A scheduling-only edge (the hint helpers' field) — must remap, not
    # copy verbatim.
    ffn_relu.scheduling_predecessors = {added}

    inputs = {
        "x": _rand(3, 4, seed=22),
        "y": _rand(3, 4, seed=23),
        "embedding_input": torch.tensor([0, 2, 1]),
    }
    return out, inputs


def _node_field_matches(key, sval, cval, node_map):
    """Field-level source-vs-clone comparison; node references go through the map."""
    if key == "node_id":
        return cval != sval
    if key == "inputs":
        return all(c is node_map[s] for s, c in zip(sval, cval)) and len(sval) == len(
            cval
        )
    if key == "scheduling_predecessors":
        return cval == {node_map[s] for s in sval} and all(
            c is not s for s in sval for c in cval
        )
    if key == "checks":
        # Fresh list object per clone; Check entries shared by reference.
        return (
            cval is not sval or not sval
        ) and all(c is s for c, s in zip(cval, sval)) and len(cval) == len(sval)
    if key == "_structural_type":
        return cval.value_range == sval.value_range
    if key == "_affine_bound":
        sr, cr = sval.to_scalar_range(), cval.to_scalar_range()
        return (sr.lo, sr.hi) == (cr.lo, cr.hi)
    if key == "_semantic_affine_override":
        if sval is None or cval is None:
            return sval is None and cval is None
        sr, cr = sval.to_scalar_range(), cval.to_scalar_range()
        return (sr.lo, sr.hi) == (cr.lo, cr.hi)
    if isinstance(sval, torch.Tensor):
        return cval is sval  # weights shared by reference, never copied
    try:
        return cval is sval or cval == sval
    except Exception:
        return cval is sval


def test_clone_completeness_every_vocabulary_variant():
    out, inputs = _variant_graph()
    copy = clone_graph(out, DISPATCH)

    # (i) Structural equality under the canonical walk.
    assert topology_entries(out) == topology_entries(copy.output_node)

    # Every vocabulary type is actually exercised (guards test rot).
    types_here = {type(n) for n in copy.node_map}
    for t in VOCABULARY:
        if t is Placeholder:
            continue  # zero-width sentinel; covered by its own test below
        assert t in types_here, f"variant graph exercises no {t.__name__}"

    # (ii) Every field of every node carried, with node references remapped.
    for src, clone in copy.node_map.items():
        assert type(clone) is type(src)
        assert clone is not src
        assert set(clone.__dict__) == set(src.__dict__), (
            f"{type(src).__name__}: clone field set differs "
            f"({set(src.__dict__) ^ set(clone.__dict__)})"
        )
        for key, sval in src.__dict__.items():
            assert _node_field_matches(key, sval, clone.__dict__[key], copy.node_map), (
                f"{type(src).__name__}.{key} not carried faithfully "
                f"(source={sval!r}, clone={clone.__dict__[key]!r})"
            )

    # (iii) Behavior: value_type, affine range (checked per-field above),
    # and compute() on random inputs, node by node.
    for src, clone in copy.node_map.items():
        s_vt, c_vt = src.value_type.value_range, clone.value_type.value_range
        assert (s_vt.lo, s_vt.hi) == (c_vt.lo, c_vt.hi), (
            f"{type(src).__name__} id={src.node_id}: value_type "
            f"{s_vt} != clone's {c_vt}"
        )
        s_val = src.compute(3, inputs)
        c_val = clone.compute(3, inputs)
        assert torch.equal(s_val, c_val), (
            f"{type(src).__name__} id={src.node_id}: compute() differs "
            f"(max diff {(s_val - c_val).abs().max().item():g})"
        )


def test_clone_shares_weight_tensors_by_reference():
    out, _ = _variant_graph()
    copy = clone_graph(out, DISPATCH)
    checked = 0
    for src, clone in copy.node_map.items():
        for attr in (
            "output_matrix",
            "output_bias",
            "query_matrix",
            "key_matrix",
            "value_matrix",
            "gate_proj",
            "gate_bias",
            "up_proj",
            "up_bias",
            "out_proj",
            "out_bias",
            "table",
            "value",
        ):
            sval = getattr(src, attr, None)
            if isinstance(sval, torch.Tensor):
                assert getattr(clone, attr) is sval, f"{type(src).__name__}.{attr}"
                checked += 1
        if isinstance(src, Embedding):
            assert clone.tokenizer is src.tokenizer
        for c_check, s_check in zip(clone.checks, src.checks):
            assert c_check is s_check  # Check entries shared by reference
    assert checked > 10


def test_clone_leaves_source_untouched():
    out, _ = _variant_graph()
    nodes_before = get_ancestor_nodes({out})
    snapshot = {
        n.node_id: (
            type(n).__name__,
            [id(i) for i in n.inputs],
            n.value_type.value_range,
            n._affine_bound.to_scalar_range(),
        )
        for n in nodes_before
    }

    clone_graph(out, DISPATCH)

    nodes_after = get_ancestor_nodes({out})
    assert {n.node_id for n in nodes_after} == set(snapshot)
    for n in nodes_after:
        tname, input_ids, vt, ab = snapshot[n.node_id]
        assert type(n).__name__ == tname
        assert [id(i) for i in n.inputs] == input_ids
        now_vt = n.value_type.value_range
        now_ab = n._affine_bound.to_scalar_range()
        assert (now_vt.lo, now_vt.hi) == (vt.lo, vt.hi)
        assert (now_ab.lo, now_ab.hi) == (ab.lo, ab.hi)


def test_clone_preserves_relative_id_order():
    out, _ = _variant_graph()
    copy = clone_graph(out, DISPATCH)
    sources_by_id = sorted(copy.node_map, key=lambda n: n.node_id)
    clone_ids_in_source_order = [copy.node_map[s].node_id for s in sources_by_id]
    assert clone_ids_in_source_order == sorted(clone_ids_in_source_order)
    # Fresh ids, disjoint from the sources' (Node.__eq__ keys on node_id).
    assert min(clone_ids_in_source_order) > max(s.node_id for s in sources_by_id)


def test_clone_is_deterministic_construction_order():
    # Same source topology -> clones constructed in the same structural
    # order both times (the order bound recomputation runs in).
    out, _ = _variant_graph()
    order1 = [type(n).__name__ for n in topological_order(out)]
    order2 = [type(n).__name__ for n in topological_order(out)]
    assert order1 == order2
    copy1 = clone_graph(out, DISPATCH)
    copy2 = clone_graph(out, DISPATCH)
    for src in copy1.node_map:
        c1, c2 = copy1.node_map[src], copy2.node_map[src]
        r1 = c1._affine_bound.to_scalar_range()
        r2 = c2._affine_bound.to_scalar_range()
        assert (r1.lo, r1.hi) == (r2.lo, r2.hi)


def test_clone_does_not_register_inputs_with_session():
    from torchwright.graph.session import current_session

    out, _ = _variant_graph()
    n_before = len(current_session().input_nodes)
    clone_graph(out, DISPATCH)
    assert len(current_session().input_nodes) == n_before


def test_clone_remaps_semantic_override_basis():
    from torchwright.ops.relu.arithmetic_ops import compare

    x = create_input("x", 1, value_range=(-10.0, 10.0))
    cmp = compare(x, 0.0, true_level=1.0, false_level=-1.0)
    overridden = [
        n for n in get_ancestor_nodes({cmp}) if n._semantic_affine_override is not None
    ]
    assert overridden, "compare() should install a semantic override"

    copy = clone_graph(cmp, DISPATCH)
    source_ids = {n.node_id for n in copy.node_map}
    for src in overridden:
        clone = copy.node_map[src]
        ov = clone._semantic_affine_override
        assert ov is not None
        for nid in list(ov.columns) + list(ov.input_ranges):
            assert (
                nid not in source_ids
            ), f"clone override basis still keyed by source node id {nid}"
        sr = src._semantic_affine_override.to_scalar_range()
        cr = ov.to_scalar_range()
        assert (sr.lo, sr.hi) == (cr.lo, cr.hi)
        # And the override is live: the clone's bound IS the override.
        br = clone._affine_bound.to_scalar_range()
        assert (br.lo, br.hi) == (sr.lo, sr.hi)


def test_clone_placeholder():
    ph = Placeholder(0)
    copy = clone_graph(ph, DISPATCH)
    clone = copy.node_map[ph]
    assert type(clone) is Placeholder
    assert clone.d_output == 0
    assert clone.node_id != ph.node_id


def test_clone_rejects_unhandled_exact_type():
    class Mystery(Node):
        def compute(self, n_pos, input_values):
            return self.inputs[0].compute(n_pos, input_values)

    x = create_input("x", 4, value_range=(-1.0, 1.0))
    stray = Linear(x, _rand(4, 3), _rand(3))
    stray.__class__ = Mystery
    with pytest.raises(GraphCloneError, match="Mystery"):
        clone_graph(stray, DISPATCH)


def test_dispatch_generation_rejects_uncovered_vocabulary():
    class FutureType(Node):
        pass

    with pytest.raises(GraphCloneError, match="FutureType"):
        build_clone_dispatch(VOCABULARY + (FutureType,))


def test_clone_catches_stray_node_reference_field():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    other = LiteralValue(torch.tensor([1.0]))
    lin = Linear(x, _rand(4, 3), _rand(3))
    lin.my_stray_ref = other  # a field the clone pass cannot know about
    with pytest.raises(GraphCloneError, match="my_stray_ref"):
        clone_graph(lin, DISPATCH)


def test_clone_raises_on_out_of_cone_scheduling_predecessor():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    lin = Linear(x, _rand(4, 3), _rand(3))
    outside = LiteralValue(torch.tensor([1.0]))  # not an ancestor of lin
    lin.scheduling_predecessors = {outside}
    with pytest.raises(GraphCloneError, match="ancestor cone"):
        clone_graph(lin, DISPATCH)


def test_clone_remaps_sibling_scheduling_predecessor():
    """A scheduling predecessor may be a *sibling* (scheduling-only edge,
    not an ancestor via inputs), which the data-topological clone order
    can visit after its dependent — the remap runs as a second pass."""
    from torchwright.graph.scheduling_hints import sequential_scope

    x = create_input("x", 4, value_range=(-1.0, 1.0))

    def factory(i):
        return lambda: Linear(x, _rand(4, 3, seed=31 + i), name=f"r_{i}")

    r1, r2, r3 = sequential_scope([factory(0), factory(1), factory(2)])
    out = Concatenate([r1, r2, r3])
    assert any(
        n.scheduling_predecessors for n in (r1, r2, r3)
    ), "sequential_scope should have wired sibling predecessors"

    copy = clone_graph(out, DISPATCH)
    for src_node in (r1, r2, r3):
        clone = copy.node_map[src_node]
        assert clone.scheduling_predecessors == {
            copy.node_map[p] for p in src_node.scheduling_predecessors
        }
        for pred in src_node.scheduling_predecessors:
            assert copy.node_map[pred] is not pred  # remapped, not verbatim
