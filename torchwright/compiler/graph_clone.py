"""Deterministic deep-copy of a compilable graph — thin nodes, shared weights.

``clone_graph`` builds the compiler-private copy that ``lower()`` hands to
the rest of the compile pipeline (``docs/lowering_copy_plan.md``): every
reachable node object is duplicated, every weight payload (``LiteralValue``
tensors, ``Linear``/``Attn`` matrices, embedding tables, predicate
closures) is shared by reference, and the copy's derived caches
(``_structural_type`` / ``_affine_bound``) are recomputed fresh in
topological order — the same guarantee ``refresh_node_caches`` provides,
which is why the copy pass subsumes ``lower()``'s old refresh loop.

Two determinism properties are load-bearing:

* **Construction order is structural.**  Clones are built in a
  topological order derived purely from the graph's ``inputs`` wiring
  (never from ``node_id`` values or set iteration), so the same source
  topology always produces the same copy in the same order — in any
  process.

* **Relative id order is preserved.**  Clone ids are pre-reserved from
  the global counter in *source-id-sorted* order, so
  ``sorted(copies, key=node_id)`` visits clones exactly where
  ``sorted(sources, key=node_id)`` visits their sources.  Affine-bound
  bases are merged in id-sorted order (``affine_bound._merge_layouts``),
  so preserving relative order keeps every bound's column layout — and
  therefore its float64 accumulation order — bit-identical to the
  source's.  Do NOT renumber clones from 0: ``Node.__eq__`` keys on
  ``node_id``, so a clone sharing an id with a live source node would
  compare equal to it and corrupt any set or dict holding both.

The per-type dispatch is generated from the vocabulary tuple the caller
passes (``lower.VOCABULARY``) — a vocabulary type without a registered
clone implementation fails loudly at generation time, and an instance
whose exact type has no entry fails loudly at clone time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import Node
from torchwright.graph.affine_bound import AffineBound
from torchwright.graph.affine_rules import refresh_node_caches
from torchwright.graph.attn import Attn
from torchwright.graph.embedding import Embedding
from torchwright.graph.ffn import FFN
from torchwright.graph.linear import Linear
from torchwright.graph.misc import (
    Add,
    Concatenate,
    InputNode,
    LiteralValue,
    Placeholder,
    ValueLogger,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class GraphCloneError(RuntimeError):
    """A graph could not be cloned (unhandled type or unremapped reference)."""


def topological_order(output_node: Node) -> list[Node]:
    """Inputs-before-consumers order over the ancestor cone (iterative).

    Purely structural: the order depends only on the ``inputs`` wiring,
    never on ``node_id`` values, so it is identical for a graph and its
    clone (and across processes for deterministic construction code).
    """
    order: list[Node] = []
    visited: set[Node] = set()
    stack: list[tuple[Node, bool]] = [(output_node, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.append((node, True))
        stack.extend((inp, False) for inp in node.inputs if inp not in visited)
    return order


# Fields rebuilt explicitly by the clone pass; everything else in a node's
# ``__dict__`` is carried by reference.  A new semantic field added to any
# node type is therefore carried automatically; a new field holding *node
# references* is caught by the stray-reference walk below.
#
# ``checks`` is rebuilt as a shallow list copy: the Check entries (and
# their predicate closures) are shared like weight payloads, but the
# list object must be the clone's own — a check attached to the source
# after cloning must not appear on the compiler-private copy.
# ``claimed_type`` / ``integer_claim`` are immutable values and ride
# the generic copy; refresh_node_caches re-applies the claim on the
# clone's fresh bounds.
_REBUILT_FIELDS = frozenset(
    {
        "inputs",
        "node_id",
        "scheduling_predecessors",
        "checks",
        "_structural_type",
        "_affine_bound",
        "_semantic_affine_override",
        # Weight-support cache (realization.live_weight_row_ranges).  Must
        # NOT ride the generic copy: lowering's folds mutate the clone's
        # output_matrix, and a stale copied cache would make the head charge
        # — and the emitter's chunk list — describe the pre-fold weights,
        # skipping chunks that are live post-fold (a miscompile, not just
        # accounting).  The clone recomputes it on first query, which
        # happens only after fusion has finished.
        "_live_weight_row_ranges",
    }
)


def _remap_bound_basis(ab: AffineBound, id_map: dict[int, int]) -> AffineBound:
    """Re-key a bound's ``columns`` / ``input_ranges`` onto clone node ids.

    An ``AffineBound``'s basis is keyed by the ``node_id`` of its leaf
    nodes (InputNode / Embedding).  A bound carried from a source node
    (the semantic affine override) still references *source* leaf ids;
    downstream basis merges on the copy would treat those as distinct
    inputs and silently lose cancellation structure.  Coefficient and
    range tensors are shared by reference — only the dict keys move.
    """
    if not ab.columns and not ab.input_ranges:
        return ab
    return AffineBound(
        A_lo=ab.A_lo,
        A_hi=ab.A_hi,
        b_lo=ab.b_lo,
        b_hi=ab.b_hi,
        columns={id_map[nid]: v for nid, v in ab.columns.items()},
        input_ranges={id_map[nid]: v for nid, v in ab.input_ranges.items()},
    )


def _iter_stray_node_refs(value: object, depth: int = 0) -> Iterator[Node]:
    """Yield ``Node`` instances found in ``value`` (one container level deep)."""
    if isinstance(value, Node):
        yield value
        return
    if depth >= 1:
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_stray_node_refs(item, depth + 1)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _iter_stray_node_refs(k, depth + 1)
            yield from _iter_stray_node_refs(v, depth + 1)


def _clone_default(
    src: Node,
    clone_map: dict[Node, Node],
    new_ids: dict[Node, int],
    id_map: dict[int, int],
) -> Node:
    """Structural clone: shared payload fields, rewired graph fields, fresh caches.

    ``__init__`` is deliberately not re-run: it has side effects the copy
    must not repeat (``InputNode`` registers with the graph session) and
    several types cannot reconstruct their state from their own fields
    (``Embedding``'s tokenizer/table split).  Instead every ``__dict__``
    entry is carried by reference except the graph-structural fields,
    which are rebuilt: ``inputs`` and ``scheduling_predecessors`` remap
    through the source→copy map, ``node_id`` takes its pre-reserved id,
    and the derived caches are recomputed from the cloned inputs via
    ``refresh_node_caches`` (with the semantic affine override re-keyed
    onto clone leaf ids first, so the re-apply lands on the copy's basis).
    """
    cls = type(src)
    clone = cls.__new__(cls)
    for key, value in src.__dict__.items():
        if key in _REBUILT_FIELDS:
            continue
        clone.__dict__[key] = value

    clone.node_id = new_ids[src]
    clone.inputs = [clone_map[inp] for inp in src.inputs]
    clone.checks = list(src.checks)
    # Remapped in clone_graph's second pass: a scheduling predecessor is
    # a scheduling-only edge, so it may be a *sibling* (not an ancestor
    # via ``inputs``) that the data-topological clone order has not
    # reached yet.
    clone.scheduling_predecessors = set()

    override = src._semantic_affine_override
    clone._semantic_affine_override = (
        None if override is None else _remap_bound_basis(override, id_map)
    )
    refresh_node_caches(clone)

    # Loud completeness guard: no other field may hold node references —
    # a field the generic copy carried verbatim would silently point the
    # clone back at source nodes (the ``scheduling_predecessors`` bug
    # class).  If a future type adds such a field, teach the clone pass
    # about it explicitly.
    for key, value in clone.__dict__.items():
        if key in ("inputs", "scheduling_predecessors"):
            continue
        for stray in _iter_stray_node_refs(value):
            raise GraphCloneError(
                f"{cls.__name__}.{key} holds a graph-node reference "
                f"({stray!r}) the clone pass does not know how to remap; "
                f"add explicit handling for this field in graph_clone."
            )
    return clone


# One entry per compilable-vocabulary type — the conscious per-type
# decision the dispatch generation checks against ``lower.VOCABULARY``.
# All current types clone structurally via ``_clone_default``; notes
# record the per-type reasoning that is not obvious from the code:
#
# * ``InputNode`` — clones do NOT register with the graph session (the
#   copy is compiler-private; input binding is name-keyed).
# * ``ValueLogger`` — cloned, not rejected: it is vocabulary and the
#   canonical walk does not skip it, so dropping it would break the
#   copy's canonical-id equality with the source.
# * ``Embedding`` — tokenizer object and table tensor shared by
#   reference (``__init__`` cannot reconstruct a caller-supplied
#   special-token split from stored fields).
_CLONE_IMPLS: dict[type, Callable] = {
    FFN: _clone_default,
    Attn: _clone_default,
    Linear: _clone_default,
    Add: _clone_default,
    InputNode: _clone_default,
    LiteralValue: _clone_default,
    Embedding: _clone_default,
    Concatenate: _clone_default,
    Placeholder: _clone_default,
    ValueLogger: _clone_default,
}


def build_clone_dispatch(vocabulary: tuple[type, ...]) -> dict[type, Callable]:
    """Generate the exact-type clone dispatch from a vocabulary tuple.

    Raises loudly if any vocabulary member lacks a registered clone
    implementation — adding a type to ``lower.VOCABULARY`` without
    teaching the clone pass about it must fail at import, not miscompile.
    """
    missing = [t for t in vocabulary if t not in _CLONE_IMPLS]
    if missing:
        raise GraphCloneError(
            f"no clone implementation registered for vocabulary type(s) "
            f"{[t.__name__ for t in missing]}; add entries to "
            f"graph_clone._CLONE_IMPLS."
        )
    return {t: _CLONE_IMPLS[t] for t in vocabulary}


@dataclass(frozen=True)
class GraphCopy:
    """A compiler-private copy of a graph plus its source↔copy node map."""

    #: The copy's output node.
    output_node: Node
    #: The source output node the copy was built from.
    source_output_node: Node
    #: source node -> its clone, one entry per reachable source node.
    node_map: dict[Node, Node]


def _remap_scheduling_predecessors(
    src: Node, clone: Node, clone_map: dict[Node, Node]
) -> None:
    """Remap one clone's scheduling predecessors through ``clone_map``."""
    try:
        clone.scheduling_predecessors = {
            clone_map[pred] for pred in src.scheduling_predecessors
        }
    except KeyError as e:
        raise GraphCloneError(
            f"scheduling predecessor of {src!r} is outside the "
            f"output's ancestor cone (no clone exists for "
            f"{e.args[0]!r}).  A predecessor that is not reachable "
            f"from the output can never be scheduled — fix the hint "
            f"wiring."
        ) from None


def clone_graph(output_node: Node, dispatch: dict[type, Callable]) -> GraphCopy:
    """Deep-copy the graph reachable from ``output_node``.

    ``dispatch`` is the exact-type clone table from
    :func:`build_clone_dispatch`; an instance whose exact type has no
    entry (including subclasses of vocabulary types) raises
    :class:`GraphCloneError`.
    """
    import torchwright.graph.node as node_module
    from torchwright.graph.node import reserve_node_id_above

    order = topological_order(output_node)

    # Pre-reserve clone ids in source-id-sorted order so relative id
    # order is preserved (see module docstring).  reserve_node_id_above
    # first: tests reset the global counter, and a clone id colliding
    # with a live source id would make the two compare equal.
    reserve_node_id_above(order)
    new_ids: dict[Node, int] = {}
    for src in sorted(order, key=lambda n: n.node_id):
        new_ids[src] = node_module.global_node_id
        node_module.global_node_id += 1
    id_map = {src.node_id: cid for src, cid in new_ids.items()}

    clone_map: dict[Node, Node] = {}
    for src in order:
        impl = dispatch.get(type(src))
        if impl is None:
            raise GraphCloneError(
                f"no clone path for exact type {type(src).__name__} "
                f"(node {src!r}); the clone dispatch covers exactly the "
                f"lowering vocabulary."
            )
        clone_map[src] = impl(src, clone_map, new_ids, id_map)

    # Second pass: scheduling predecessors are scheduling-only edges and
    # may point at siblings the data-topological order visited later, so
    # they can only be remapped once every clone exists.
    for src, clone in clone_map.items():
        _remap_scheduling_predecessors(src, clone, clone_map)

    return GraphCopy(
        output_node=clone_map[output_node],
        source_output_node=output_node,
        node_map=clone_map,
    )
