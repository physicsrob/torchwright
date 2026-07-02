"""The ``blockify`` verification pass: assert a graph is block-native.

Phase 2a introduced this pass as a *converter* — it mined ``Linear -> ReLU ->
Linear`` chains and rewrote each into a first-class
:class:`~torchwright.graph.block.Block`.  Phase 2b makes the op layer build
Blocks directly (``linear_relu_linear`` returns a Block; no op constructs a raw
chain), so there is nothing left to convert.  What remains is the *check*: run
the same chain miner over the graph and assert it finds **zero** chains.  A
surviving raw ``Linear -> ReLU -> Linear`` shape means some code built a chain
by hand instead of a Block — a construction bug this pass surfaces early,
before the scheduler (which no longer mines chains after Phase 3) would fail to
compile it.

The miner is kept only as that detector; its Concatenate/Assert transparency
matches how the compiler used to see chains, so it catches a hand-built chain
regardless of intervening wrappers.
"""

from typing import Dict, List, Set, Tuple

from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import Node
from torchwright.graph.linear import Linear
from torchwright.graph.misc import Assert, Concatenate, DebugWatch
from torchwright.graph.relu import ReLU


def _unwrap(node: Node) -> Node:
    """Step through Assert/DebugWatch wrappers to the wrapped node."""
    while isinstance(node, (Assert, DebugWatch)):
        node = node.inputs[0]
    return node


class _ChainMiner:
    """Assert/Concatenate-transparent ``L1 -> ReLU -> L2`` chain detector.

    Mirrors the semantics the compiler's chain mining used — each ReLU and each
    Linear participates in at most one chain, iteration in node-id order for
    determinism — resolving effective consumers transparently through
    ``Concatenate`` **and** ``Assert``/``DebugWatch`` so a hand-built chain is
    detected even when a wrapper sits on an internal value.
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

    def direct_consumers(self, node: Node) -> List[Node]:
        return self._direct_consumers.get(node, [])

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


def blockify(output_node: Node, *, verbose: bool = False) -> Node:
    """Verify the graph reachable from ``output_node`` is block-native.

    Runs the chain miner and raises if it finds any ``Linear -> ReLU ->
    Linear`` chain.  After Phase 2b every such shape is built as a
    :class:`Block`, so a surviving raw chain is a construction bug (an op that
    built ``Linear``/``ReLU``/``Linear`` by hand instead of calling
    ``linear_relu_linear`` / constructing a Block).  Returns ``output_node``
    unchanged — the pass no longer rewrites the graph.

    Args:
        output_node: The graph's output node.
        verbose: Print the (zero) chain count on success.

    Returns:
        ``output_node`` (unchanged).

    Raises:
        AssertionError: if any raw chain remains.
    """
    all_nodes = get_ancestor_nodes({output_node})
    miner = _ChainMiner(all_nodes)
    chains = miner.mine()

    if chains:
        ids = sorted((l1.node_id, relu.node_id, l2.node_id) for l1, relu, l2 in chains)
        raise AssertionError(
            f"blockify: found {len(chains)} unclaimed Linear->ReLU->Linear "
            f"chain(s) (L1,ReLU,L2 node ids) {ids}. Since Phase 2b ops build "
            f"Block nodes natively; a surviving raw chain is a construction bug. "
            f"Stop-and-report."
        )

    if verbose:
        print("blockify: 0 chains (graph is block-native)")

    return output_node
