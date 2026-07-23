"""Static detection of "sibling clusters" for scheduler admission control.

A sibling cluster is a graph pattern where ``N ≥ min_chains`` parallel
chains all feed a common N-way join (typically a ``Concatenate``).  Each
chain has wide intermediate nodes (``≥ min_peak_width``) that persist
until the chain's terminal node is placed.  If the scheduler admits too
many sibling chains concurrently, their intermediates saturate the
residual stream and force a long low-productivity "plateau" — see the
optimization_guide §7 for background.

This module runs once before scheduling and produces a
:class:`SiblingClusters` descriptor that the :class:`LayerScheduler`
consults to gate admission of new chains.

Detection rules
---------------

For each candidate join node J (currently: ``Concatenate`` with
``≥ min_chains`` inputs):

    1. Compute, per input branch, the "backward-reachable" set of
       non-``Concatenate`` nodes (traversing ``Concatenate`` inputs
       transparently since they're never placed in the residual stream).
    2. The branch-exclusive set = backward-reachable nodes that aren't
       shared with any other branch of J.
    3. Prune nodes whose direct consumers escape the exclusive set
       (i.e., have any consumer outside ``exclusive U {J}``,
       modulo ``Concatenate`` transparency).
    4. Peak width = the max ``len(n)`` over the surviving branch nodes
       (an FFN's output occupies residual columns like any node).
    5. Accept the cluster iff ≥ ``min_chains`` branches survive and
       the maximum branch peak-width ≥ ``min_peak_width``.

Limitations
-----------

- Only ``Concatenate`` joins are detected.  Multi-input ``Add`` or
  concat-fed ``Linear`` joins can be added later.
- Exclusivity is strict: a node shared between two branches (even
  indirectly) is excluded from both.  This misses valid clusters with
  diamond dependencies but avoids mis-attributing shared work.
- Peak width is a static max-over-nodes; it doesn't account for the
  fact that some intermediates may be freed before the chain's peak
  is reached.  Conservative = admission is slightly more aggressive
  than optimal.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.graph import Concatenate, Node
from torchwright.graph.misc import LiteralValue


@dataclass
class ChainInfo:
    """One parallel branch of a sibling cluster.

    ``nodes`` is the set of branch-exclusive non-``Concatenate`` nodes.
    ``terminal`` is the (non-``Concatenate``) node that feeds the join
    directly — scheduling its placement marks the branch "completed"
    and frees an in-flight slot.  ``peak_width`` is the max
    residual-relevant width over ``nodes``.
    """

    chain_id: int
    nodes: set[Node]
    terminal: Node
    peak_width: int


@dataclass
class ClusterInfo:
    cluster_id: int
    join: Node
    chains: list[ChainInfo]
    peak_chain_width: int


@dataclass
class SiblingClusters:
    """Analysis output consumed by the scheduler.

    ``node_to_chain`` maps each branch-exclusive node to its
    ``(cluster_id, chain_id)`` so the scheduler can look up in O(1)
    whether a candidate node belongs to a gated cluster.

    ``terminal_to_chain`` maps a branch's terminal node to its
    ``(cluster_id, chain_id)`` so the scheduler can detect "branch
    completed" transitions as terminals are scheduled.
    """

    clusters: dict[int, ClusterInfo] = field(default_factory=dict)
    node_to_chain: dict[Node, tuple[int, int]] = field(default_factory=dict)
    terminal_to_chain: dict[Node, tuple[int, int]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.clusters


class SiblingClusterAnalyzer:
    """Detects sibling clusters in a graph.

    Parameters
    ----------
    graph
        A :class:`GraphAnalyzer` over the target graph.
    min_chains
        Minimum number of parallel branches required to register a
        cluster.  Default 4 - captures unrolled loops without firing on
        small (2-3 way) joins that don't benefit from batching.
    min_peak_width
        Minimum per-branch peak intermediate width.  Default 32 —
        excludes scalar-only branches where admission gating would
        over-serialize with no pressure benefit.
    """

    def __init__(
        self,
        graph: GraphAnalyzer,
        min_chains: int = 4,
        min_peak_width: int = 32,
    ) -> None:
        self.graph = graph
        self.min_chains = min_chains
        self.min_peak_width = min_peak_width

    def analyze(self) -> SiblingClusters:
        result = SiblingClusters()
        next_id = 0

        for join in self._find_joins():
            cluster = self._try_build_cluster(join, next_id)
            if cluster is None:
                continue
            result.clusters[cluster.cluster_id] = cluster
            for chain in cluster.chains:
                key = (cluster.cluster_id, chain.chain_id)
                for node in chain.nodes:
                    # If a node ends up in multiple clusters (possible
                    # if graph has unusual topology), first assignment
                    # wins.  This is rare and the alternative
                    # (double-counting) would over-gate.
                    result.node_to_chain.setdefault(node, key)
                result.terminal_to_chain.setdefault(chain.terminal, key)
            next_id += 1

        return result

    # ------------------------------------------------------------------
    # Join discovery
    # ------------------------------------------------------------------

    def _find_joins(self) -> Iterator[Concatenate]:
        for node in self.graph.get_all_nodes():
            if isinstance(node, Concatenate) and len(node.inputs) >= self.min_chains:
                yield node

    # ------------------------------------------------------------------
    # Per-join cluster construction
    # ------------------------------------------------------------------

    def _try_build_cluster(self, join: Node, cluster_id: int) -> ClusterInfo | None:
        inputs = list(join.inputs)

        # Step 1: per-input backward-reachable set (Concatenate-transparent).
        per_input_reachable: list[set[Node]] = [
            self._backward_reachable(inp) for inp in inputs
        ]

        # Step 2: per-input exclusive set = reachable_i \ union_{j≠i} reachable_j.
        chains: list[ChainInfo] = []
        union_others_cache = self._union_others(per_input_reachable)
        for idx, inp in enumerate(inputs):
            exclusive = per_input_reachable[idx] - union_others_cache[idx]
            # Filter out input nodes — they're always live and not
            # candidates for admission gating.  LiteralValue is likewise
            # excluded: constants are width-small and now materialized
            # just-in-time, not the wide-intermediate chains admission
            # control targets.
            exclusive = {
                n
                for n in exclusive
                if not self.graph.is_input_node(n) and not isinstance(n, LiteralValue)
            }
            if not exclusive:
                continue

            # Step 3: prune nodes with external consumers.
            exclusive = self._prune_external_consumers(exclusive, join)
            if not exclusive:
                continue

            # Step 4: compute peak residual width.  An FFN's output occupies
            # residual columns like any node (its internal ReLU activations
            # live in MLP hidden slots, but that is not a graph node here), so
            # every exclusive node contributes its width.
            widths = [len(n) for n in exclusive]
            if not widths:
                continue
            peak = max(widths)

            # Terminal = the branch's direct input to the join.  If inp
            # is a Concatenate (nested Concatenate-into-Concatenate),
            # fall back to the widest exclusive node — the scheduler
            # can handle either.
            terminal = inp if inp in exclusive else max(exclusive, key=len)

            chains.append(
                ChainInfo(
                    chain_id=len(chains),
                    nodes=exclusive,
                    terminal=terminal,
                    peak_width=peak,
                )
            )

        # Step 5: cluster acceptance.
        if len(chains) < self.min_chains:
            return None
        peak_chain_width = max(c.peak_width for c in chains)
        if peak_chain_width < self.min_peak_width:
            return None

        return ClusterInfo(
            cluster_id=cluster_id,
            join=join,
            chains=chains,
            peak_chain_width=peak_chain_width,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _backward_reachable(self, start: Node) -> set[Node]:
        """All non-Concatenate nodes reachable backward from ``start``.

        Concatenates are transparent: walked through but not included
        in the result.  The start node itself is included (unless it's
        a Concatenate, in which case its children are).
        """
        result: set[Node] = set()
        visited: set[Node] = set()
        stack: list[Node] = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if not isinstance(cur, Concatenate):
                result.add(cur)
            stack.extend(cur.inputs)
        return result

    def _union_others(self, per_input_reachable: list[set[Node]]) -> list[set[Node]]:
        """For each index i, return the union of all other reachable sets."""
        out: list[set[Node]] = []
        for i, _ in enumerate(per_input_reachable):
            u: set[Node] = set()
            for j, s_j in enumerate(per_input_reachable):
                if j != i:
                    u |= s_j
            out.append(u)
        return out

    def _prune_external_consumers(self, exclusive: set[Node], join: Node) -> set[Node]:
        """Iteratively drop nodes with consumers outside ``exclusive U {join}``.

        Concatenates on the consumer side are walked through: a node n
        whose immediate consumer is a Concatenate C is fine iff all of
        C's downstream non-Concatenate consumers are in the exclusive
        set or are the join.
        """
        valid_downstream_cache: dict[Node, bool] = {}

        def is_valid(node: Node, within: set[Node]) -> bool:
            if node is join:
                return True
            if node in within:
                return True
            if not isinstance(node, Concatenate):
                return False
            if node in valid_downstream_cache:
                return valid_downstream_cache[node]
            # Tentatively mark True to break cycles (DAG: won't happen,
            # but defensive).  We'll finalize after checking.
            valid_downstream_cache[node] = True
            ok = all(is_valid(c, within) for c in self.graph.get_consumers(node))
            valid_downstream_cache[node] = ok
            return ok

        # Pruning can invalidate the cache, so clear it each iteration.
        while True:
            valid_downstream_cache.clear()
            to_remove: set[Node] = set()
            for n in exclusive:
                for c in self.graph.get_consumers(n):
                    if not is_valid(c, exclusive):
                        to_remove.add(n)
                        break
            if not to_remove:
                break
            exclusive -= to_remove
        return exclusive

    def _effective_consumers(self, node: Node) -> set[Node]:
        """Consumers of ``node``, walking through ``Concatenate``."""
        result: set[Node] = set()
        stack = list(self.graph.get_consumers(node))
        seen: set[Node] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if isinstance(cur, Concatenate):
                stack.extend(self.graph.get_consumers(cur))
            else:
                result.add(cur)
        return result
