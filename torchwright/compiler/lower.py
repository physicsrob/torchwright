"""The lowering boundary: certify a graph is scheduler-ready.

``lower()`` is the single moment a graph transitions from "whatever the
ops layer built" to "certified compilable vocabulary"
(``docs/lowering_boundary_plan.md``).  Constructing a :class:`LoweredGraph`
validates two things:

1. **Closed vocabulary** — every reachable node is one of the compilable
   types (Block, Attn, Linear, Add), bookkeeping (InputNode, LiteralValue,
   Embedding, Concatenate, Placeholder), or a debug wrapper (Assert,
   DebugWatch, ValueLogger).  In particular no raw ``ReLU`` survives: since
   Phase 2b the ops layer builds Block nodes natively, so a ReLU in a graph
   is a construction bug (a hand-built ``Linear -> ReLU -> Linear`` chain,
   or a stray nonlinearity the scheduler has no write path for).  The chain
   miner that used to live in ``graph/blockify.py`` is folded in here as
   the error-reporting detector.

2. **Fresh derived caches** — ``_affine_bound`` / ``_structural_type`` are
   computed eagerly in ``Node.__init__``, so any in-place graph mutation
   (a fusion fold, a hand edit) leaves them describing the pre-mutation
   graph.  ``lower()`` recomputes both for every reachable node in true
   topological order (NOT node-id order — the "inputs have smaller ids"
   property is a fusion-fold invariant, not a general graph property) and
   re-applies each node's semantic affine override, reproducing
   construction-time bounds exactly.  The stale-bounds class of bug
   (commit 0570af1) becomes structurally impossible at this boundary
   instead of being guarded by per-pass discipline.

``forward_compile`` calls ``lower()`` as its first step, ahead of
``GraphAnalyzer`` — the ordering matters: GraphAnalyzer's Assert-strip
tightens structural types and bound input-ranges from claimed ranges, and
that tightening must land on top of *fresh* bounds, not be wiped by a
later recompute.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from torchwright.compiler.realization import RealizationTable
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import Node
from torchwright.graph.affine_rules import refresh_node_caches
from torchwright.graph.attn import Attn
from torchwright.graph.block import Block
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
    Block,
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


class LoweringError(ValueError):
    """A graph failed certification at the lowering boundary."""


@dataclass(frozen=True)
class LoweredGraph:
    """Certificate that a graph passed the lowering boundary.

    Holds the certified output node and the **unresolved realization
    table** — every schedulable node's candidate realization classes
    (:mod:`torchwright.compiler.realization`).  Produced only by
    :func:`lower`; the compile pipeline's scheduling stages consume this
    type, so an uncertified graph cannot reach the scheduler.  A resolver
    (the static policy at ``optimize=0``; the CP-SAT solve on the directed
    path) turns the table into the resolved artifact the layer walk reads.
    """

    output_node: Node
    realization_table: RealizationTable


def _unwrap(node: Node) -> Node:
    """Step through Assert/DebugWatch wrappers to the wrapped node."""
    while isinstance(node, (Assert, DebugWatch)):
        node = node.inputs[0]
    return node


class _ChainMiner:
    """Assert/Concatenate-transparent ``L1 -> ReLU -> L2`` chain detector.

    Ported from the deleted ``graph/blockify.py``.  Used only to enrich the
    vocabulary error: when raw ReLU nodes are found, chains among them are
    reported as (L1, ReLU, L2) id triples so the fix ("build a Block") is
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
                f"build Block nodes natively (linear_relu_linear); build a "
                f"Block instead"
            )
        if lone:
            parts.append(
                f"raw ReLU node(s) {lone} with no chain shape — the "
                f"scheduler has no write path for a lone ReLU; use a Block"
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


def _topological_order(output_node: Node) -> List[Node]:
    """Inputs-before-consumers order over the ancestor cone (iterative)."""
    order: List[Node] = []
    visited: Set[Node] = set()
    stack: List[Tuple[Node, bool]] = [(output_node, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.append((node, True))
        for inp in node.inputs:
            if inp not in visited:
                stack.append((inp, False))
    return order


def lower(output_node: Node, *, verbose: bool = False) -> LoweredGraph:
    """Certify the graph reachable from ``output_node`` for scheduling.

    Runs the two boundary validations (closed vocabulary; fresh derived
    caches) and returns the :class:`LoweredGraph` certificate the compile
    pipeline consumes.  Raises :class:`LoweringError` on a vocabulary
    violation.  The cache refresh mutates the graph's nodes in place
    (recomputed ``_affine_bound`` / ``_structural_type``, semantic
    overrides re-applied) and is idempotent.

    Args:
        output_node: The graph's output node.
        verbose: Print the certified node count.

    Returns:
        A :class:`LoweredGraph` holding ``output_node``.

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
    for node in _topological_order(output_node):
        refresh_node_caches(node)
    table = RealizationTable.build(all_nodes)
    if verbose:
        print(f"lower(): certified {len(all_nodes)} nodes (vocabulary + fresh caches)")
    return LoweredGraph(output_node=output_node, realization_table=table)
