"""The lowering boundary: certify a graph and hand the pipeline a private copy.

``lower()`` is the single moment a graph transitions from "whatever the
ops layer built" to "certified compilable vocabulary"
(``docs/lowering_boundary_plan.md``, ``docs/lowering_copy_plan.md``).
Constructing a :class:`LoweredGraph` does four things:

1. **Closed vocabulary** — every reachable node is one of the compilable
   types (FFN, Attn, Linear, Add), bookkeeping (InputNode, LiteralValue,
   Embedding, Concatenate, Placeholder), or a debug wrapper (Assert,
   DebugWatch, ValueLogger).  In particular no raw ``ReLU`` survives: since
   Phase 2b the ops layer builds FFN nodes natively, so a ReLU in a graph
   is a construction bug (a hand-built ``Linear -> ReLU -> Linear`` chain,
   or a stray nonlinearity the scheduler has no write path for).  The chain
   miner that used to live in ``graph/blockify.py`` is folded in here as
   the error-reporting detector.

2. **A compiler-private copy** — compilation is a pure function of the
   source graph.  The copy duplicates every node object (weights shared
   by reference), recomputes ``_affine_bound`` / ``_structural_type``
   fresh in topological order (subsuming the refresh loop that used to
   live here — the stale-bounds bug class of commit 0570af1 stays
   structurally impossible), and is the only graph any compile pass ever
   touches.  The source keeps its Assert/DebugWatch wrappers and its
   bounds forever; recompiling is re-lowering the same pristine source.

3. **Linear fusion, on the copy** — ``fuse_consecutive_linears`` runs on
   the copy before the wrapper strip (wrappers block folds exactly as
   they did when callers pre-fused their source graphs, and the bounds
   refresh re-derives claims through the Assert rule).  Every compile
   gets the fused schedule; no caller-side pre-pass is needed, and the
   source graph stays unfused.  Fusion orphans some copy nodes and
   changes what a few survivors compute, so ``node_map`` tracks values:
   a source node whose value no longer exists in the copy has no entry.

4. **The wrapper strip, on the copy** — Assert/DebugWatch wrappers are
   removed from the *copy* (their claimed ranges tightened onto the
   wrapped nodes' types first), so the scheduler, weight writer, and
   certifications see a wrapper-free graph with claim-tightened bounds.
   ``GraphAnalyzer`` is pure analysis and receives no wrappers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from torchwright.compiler.graph_clone import (
    build_clone_dispatch,
    clone_graph,
)
from torchwright.compiler.realization import (
    CostSummary,
    RealizationTable,
    summarize_cost,
)
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import Node
from torchwright.graph.optimize import FoldLog, fuse_consecutive_linears
from torchwright.graph.attn import Attn
from torchwright.graph.ffn import FFN
from torchwright.graph.embedding import Embedding
from torchwright.graph.linear import Linear
from torchwright.graph.misc import (
    Add,
    Assert,
    Concatenate,
    DebugWatch,
    InputNode,
    LiteralValue,
    Placeholder,
    ValueLogger,
)
from torchwright.graph.relu import ReLU

# The closed vocabulary a compilable graph may contain.  Anything outside
# this tuple has no scheduler routing and no weight-writer emission; it is
# rejected at the boundary with a construction-bug error instead of failing
# deep inside scheduling.
VOCABULARY: Tuple[type, ...] = (
    FFN,
    Attn,
    Linear,
    Add,
    InputNode,
    LiteralValue,
    Embedding,
    Concatenate,
    Placeholder,
    Assert,
    DebugWatch,
    ValueLogger,
)

# Clone dispatch generated from the vocabulary — a vocabulary type without
# a registered clone path fails HERE, at import, not mid-compile.
_CLONE_DISPATCH = build_clone_dispatch(VOCABULARY)


class LoweringError(ValueError):
    """A graph failed certification at the lowering boundary."""


@dataclass(frozen=True)
class LoweredGraph:
    """Certificate that a graph passed the lowering boundary.

    Holds the **compiler-private copy** of the graph (``output_node``, a
    wrapper-free clone with claim-tightened fresh bounds), the untouched
    source graph's output (``source_output_node``), the source→copy node
    map, and the **unresolved realization table** built from the copy —
    every schedulable node's candidate realization classes
    (:mod:`torchwright.compiler.realization`).  Produced only by
    :func:`lower`; the compile pipeline's scheduling stages consume this
    type, so an uncertified graph cannot reach the scheduler.  A resolver
    (the static policy at ``optimize=0``; the CP-SAT solve on the directed
    path) turns the table into the resolved artifact the layer walk reads.

    ``node_map`` maps each reachable source node to the copy node that
    computes its value; source Assert/DebugWatch wrappers map to the
    clone of the node they wrap (the copy is stripped, so the wrappers'
    clones are not part of it).  The map is **partial**: a source node
    fused away during lowering (its value absorbed into a survivor's
    weights) has no entry, and a source node whose value moved onto a
    fold survivor maps to that survivor.
    The realization table — like everything the compile pipeline sees —
    is keyed by *copy* node ids; translate through :meth:`copy_of` when
    querying it with source nodes.
    """

    output_node: Node
    realization_table: RealizationTable
    source_output_node: Optional[Node] = None
    node_map: Dict[Node, Node] = field(default_factory=dict)

    def copy_of(self, source_node: Node) -> Node:
        """The copy node computing ``source_node``'s value (wrappers
        resolve to the clone of the node they wrap)."""
        try:
            return self.node_map[source_node]
        except KeyError:
            raise KeyError(
                f"{source_node!r} has no counterpart in the lowered graph — "
                f"either it is not reachable from the source output, or "
                f"linear fusion at the lowering boundary absorbed its value "
                f"into a survivor's weights (fused-away nodes are not "
                f"probeable)"
            ) from None

    def cost_summary(
        self,
        d_head: int,
        realization_table: Optional[RealizationTable] = None,
        policy: Optional["SchedulingPolicy"] = None,
    ) -> CostSummary:
        """Hardware demand readable *before* scheduling.

        Aggregates heads by class, MLP bypass slot demand, and FFN lane
        counts from a resolved realization table plus each class's
        resource signature.  Pass the resolved table you intend to compile
        with; by default the static policy resolves the free choices (the
        optimize=0 rule).  ``Add`` demand is reported as its two
        conditional bounds.
        """
        from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy

        if realization_table is None:
            realization_table = self.realization_table.resolve_static(
                policy if policy is not None else SchedulingPolicy()
            )
        nodes = get_ancestor_nodes({self.output_node})
        return summarize_cost(nodes, realization_table, d_head)


def _unwrap(node: Node) -> Node:
    """Step through Assert/DebugWatch wrappers to the wrapped node."""
    while isinstance(node, (Assert, DebugWatch)):
        node = node.inputs[0]
    return node


def _strip_debug_wrappers(output_node: Node) -> Node:
    """Remove Assert/DebugWatch wrappers from the (compiler-private) graph.

    Runs only on the copy ``lower()`` just built — never on a user's
    graph.  Two steps, ported unchanged from the pre-copy
    ``GraphAnalyzer._strip_asserts``:

    1. Each Assert's claimed range is transferred onto the node it wraps
       — the wrapped node's ``_structural_type`` is tightened with the
       claim, and any input-range tightening the Assert's affine bound
       carries (the leaf-claim channel of ``_assert_rule``) is
       intersected into the wrapped node's bound.  This is what makes
       the copy's ``value_type``s claim-tightened even though the
       wrappers are gone.

    2. Every consumer of a wrapper is rewired to the wrapped node, and
       the unwrapped output is returned.

    Order note: asserts are processed in ``node_id`` order for
    determinism, though the transfer itself commutes (range
    intersections are min/max).
    """
    from torchwright.graph.value_type import tightened_with

    all_nodes = get_ancestor_nodes({output_node})
    asserts = sorted(
        (n for n in all_nodes if isinstance(n, Assert)), key=lambda n: n.node_id
    )
    watches = [n for n in all_nodes if isinstance(n, DebugWatch)]
    if not asserts and not watches:
        return output_node

    for a in asserts:
        if a.claimed_type is None:
            continue
        target = _unwrap(a)
        target._structural_type = tightened_with(
            target._structural_type, a.claimed_type
        )

        import torch

        from torchwright.graph.affine_bound import AffineBound

        a_ab = a._affine_bound
        t_ab = target._affine_bound
        new_ranges = dict(t_ab.input_ranges)
        changed = False
        for nid, (a_lo, a_hi) in a_ab.input_ranges.items():
            if nid in new_ranges:
                old_lo, old_hi = new_ranges[nid]
                tighter_lo = torch.maximum(old_lo, a_lo)
                tighter_hi = torch.minimum(old_hi, a_hi)
                if not (
                    torch.equal(tighter_lo, old_lo) and torch.equal(tighter_hi, old_hi)
                ):
                    new_ranges[nid] = (tighter_lo, tighter_hi)
                    changed = True
        if changed:
            target._affine_bound = AffineBound(
                A_lo=t_ab.A_lo,
                A_hi=t_ab.A_hi,
                b_lo=t_ab.b_lo,
                b_hi=t_ab.b_hi,
                columns=t_ab.columns,
                input_ranges=new_ranges,
            )

    for node in all_nodes:
        if isinstance(node, (Assert, DebugWatch)):
            continue
        for i, inp in enumerate(node.inputs):
            if isinstance(inp, (Assert, DebugWatch)):
                node.inputs[i] = _unwrap(inp)

    return _unwrap(output_node)


class _ChainMiner:
    """Assert/Concatenate-transparent ``L1 -> ReLU -> L2`` chain detector.

    Ported from the deleted ``graph/blockify.py``.  Used only to enrich the
    vocabulary error: when raw ReLU nodes are found, chains among them are
    reported as (L1, ReLU, L2) id triples so the fix ("build an FFN") is
    obvious.  Resolves effective consumers transparently through
    ``Concatenate`` **and** ``Assert``/``DebugWatch`` so a hand-built chain
    is detected even when a wrapper sits on an internal value.
    """

    def __init__(self, all_nodes: Set[Node]):
        self._direct_consumers: Dict[Node, List[Node]] = {n: [] for n in all_nodes}
        for node in all_nodes:
            for inp in node.inputs:
                if inp in self._direct_consumers:
                    self._direct_consumers[inp].append(node)
        self._eff_cache: Dict[Node, Set[Node]] = {}

    def effective_consumers(self, node: Node) -> Set[Node]:
        """Consumers resolving through Concatenate/Assert/DebugWatch.

        A terminal transparent wrapper (an output Concatenate/Assert with no
        further consumers) is kept as the effective consumer so the node it
        wraps is never treated as dead."""
        if node in self._eff_cache:
            return self._eff_cache[node]
        result: Set[Node] = set()
        for consumer in self._direct_consumers.get(node, []):
            if isinstance(consumer, (Concatenate, Assert, DebugWatch)):
                downstream = self.effective_consumers(consumer)
                if downstream:
                    result |= downstream
                else:
                    result.add(consumer)
            else:
                result.add(consumer)
        self._eff_cache[node] = result
        return result

    def mine(self) -> List[Tuple[Linear, ReLU, Linear]]:
        chains: List[Tuple[Linear, ReLU, Linear]] = []
        seen_relus: Set[Node] = set()
        seen_linears: Set[Node] = set()

        linears = sorted(
            (n for n in self._direct_consumers if isinstance(n, Linear)),
            key=lambda n: n.node_id,
        )
        for l1 in linears:
            if l1 in seen_linears:
                continue
            l1_eff = self.effective_consumers(l1)
            relu_candidates = [c for c in l1_eff if isinstance(c, ReLU)]
            if len(relu_candidates) != 1:
                continue
            relu = relu_candidates[0]
            if relu in seen_relus:
                continue
            relu_eff = self.effective_consumers(relu)
            l2_candidates = [c for c in relu_eff if isinstance(c, Linear)]
            if len(relu_eff) != 1 or len(l2_candidates) != 1:
                continue
            l2 = l2_candidates[0]
            if _unwrap(l2.inputs[0]) is not relu:
                continue
            if l2 in seen_linears:
                continue
            chains.append((l1, relu, l2))
            seen_relus.add(relu)
            seen_linears.add(l1)
            seen_linears.add(l2)

        return chains


def _check_vocabulary(all_nodes: Set[Node]) -> None:
    """Raise :class:`LoweringError` if any node is outside the vocabulary."""
    strays = [n for n in all_nodes if not isinstance(n, VOCABULARY)]
    if not strays:
        return

    parts: List[str] = []
    relus = [n for n in strays if isinstance(n, ReLU)]
    if relus:
        miner = _ChainMiner(all_nodes)
        chains = miner.mine()
        chain_ids = sorted(
            (l1.node_id, relu.node_id, l2.node_id) for l1, relu, l2 in chains
        )
        chained = {relu.node_id for _, relu, _ in chains}
        lone = sorted(r.node_id for r in relus if r.node_id not in chained)
        if chain_ids:
            parts.append(
                f"{len(chain_ids)} hand-built Linear->ReLU->Linear chain(s) "
                f"(L1,ReLU,L2 node ids) {chain_ids} — since Phase 2b ops "
                f"build FFN nodes natively (linear_relu_linear); build an "
                f"FFN instead"
            )
        if lone:
            parts.append(
                f"raw ReLU node(s) {lone} with no chain shape — the "
                f"scheduler has no write path for a lone ReLU; use an FFN"
            )
    others = [n for n in strays if not isinstance(n, ReLU)]
    if others:
        by_type: Dict[str, List[int]] = {}
        for n in others:
            by_type.setdefault(type(n).__name__, []).append(n.node_id)
        desc = ", ".join(f"{t} {sorted(ids)}" for t, ids in sorted(by_type.items()))
        parts.append(f"non-compilable node type(s): {desc}")

    raise LoweringError(
        "lower(): graph failed vocabulary certification: " + "; ".join(parts)
    )


def lower(output_node: Node, *, verbose: bool = False) -> LoweredGraph:
    """Certify the graph reachable from ``output_node`` and return its copy.

    Compilation is a pure function of the source graph: this validates
    the closed vocabulary, builds the compiler-private copy (fresh
    derived caches by construction), runs linear fusion on the copy
    (every compile gets the fused schedule — callers no longer pre-fuse),
    strips Assert/DebugWatch wrappers from the copy with their claims
    tightened onto the wrapped nodes, and returns the
    :class:`LoweredGraph` certificate the compile pipeline consumes.
    The source graph — nodes, wiring, wrappers, cached bounds — is never
    touched.

    Args:
        output_node: The source graph's output node.
        verbose: Print the certified node count.

    Returns:
        A :class:`LoweredGraph` holding the copy's output node, the
        source output, and the source→copy node map.

    Raises:
        LoweringError: if any reachable node is outside the compilable
            vocabulary (including any raw ``ReLU`` / hand-built chain).
    """
    if not isinstance(output_node, Node):
        raise TypeError(
            f"lower() expects the graph's output Node, got "
            f"{type(output_node).__name__}"
        )
    all_nodes = get_ancestor_nodes({output_node})
    _check_vocabulary(all_nodes)

    copy = clone_graph(output_node, _CLONE_DISPATCH)

    # Linear fusion, on the copy, BEFORE the wrapper strip: with the
    # wrappers still present the folds fire exactly as they did when
    # callers fused their source graphs (a wrapper on a value blocks the
    # fold that would absorb it), and the post-fusion bounds refresh can
    # re-derive claims through the Assert affine rule.  The source graph
    # is never touched — fusing here is what lets it stay pristine.
    fold_log = FoldLog()
    n_fused = fuse_consecutive_linears({copy.output_node}, fold_log=fold_log)

    copy_output = _strip_debug_wrappers(copy.output_node)
    copy_nodes = get_ancestor_nodes({copy_output})

    if not n_fused:
        # Wrapper sources resolve to the clone of the node they wrap —
        # the stripped wrapper clones are no longer part of the copy.
        node_map = {src: _unwrap(clone) for src, clone in copy.node_map.items()}
    else:
        # Fusion orphans copy nodes and changes what some survivors
        # compute, so the map must track VALUES, not object identity: a
        # source node maps to the copy node that computes its value, or
        # to nothing if that value no longer exists in the copy.
        # Wrapper sources are resolved through their SOURCE wrapped node
        # (a wrapper's value is its wrapped node's value) — resolving
        # through the clone would be wrong when a fold rewired the clone
        # wrapper onto a survivor whose own mapping is dropped.
        node_map: Dict[Node, Node] = {}
        for src, clone in copy.node_map.items():
            if isinstance(src, (Assert, DebugWatch)):
                continue
            survivor = fold_log.moves.get(clone)
            if survivor is not None:
                node_map[src] = survivor
            elif clone in fold_log.value_changed or clone not in copy_nodes:
                continue  # value gone (changed in place, or orphaned)
            else:
                node_map[src] = clone
        for src in copy.node_map:
            if isinstance(src, (Assert, DebugWatch)):
                target = node_map.get(_unwrap(src))
                if target is not None:
                    node_map[src] = target

    table = RealizationTable.build(copy_nodes)
    if verbose:
        print(
            f"lower(): certified {len(all_nodes)} nodes "
            f"(vocabulary + compiler-private copy), fused {n_fused} pair(s)"
        )
    return LoweredGraph(
        output_node=copy_output,
        realization_table=table,
        source_output_node=output_node,
        node_map=node_map,
    )
