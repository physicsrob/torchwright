from collections import defaultdict, deque
from typing import Dict, List, Set

from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import (
    Node,
    Concatenate,
    InputNode,
    Embedding,
)


class GraphAnalyzer:
    """Pre-computes graph metadata needed by the forward compiler scheduler.

    Pure analysis — nothing here mutates the graph.
    """

    def __init__(self, output_node: Node):
        self._output_node = output_node
        self._all_nodes = get_ancestor_nodes({output_node})

        # Build reverse dependency map: node -> set of nodes that consume it
        self._consumers: Dict[Node, Set[Node]] = defaultdict(set)
        for node in self._all_nodes:
            for inp in node.inputs:
                self._consumers[inp].add(node)

        self._topo_order = self._build_topo_order()
        self._critical_path: Dict[Node, int] = {}
        self._compute_critical_paths()

    def get_output_node(self) -> Node:
        """Return the analyzed graph's output node (as passed in)."""
        return self._output_node

    def _build_topo_order(self) -> List[Node]:
        """Kahn's algorithm — returns nodes with inputs before dependents.

        Seeds and consumer drains iterate in ``node_id`` order: the
        result order feeds critical-path computation and (transitively)
        scheduling decisions, and set iteration keyed on absolute id
        values would not survive a graph rebuild or the lowering copy.
        """
        in_degree: Dict[Node, int] = {node: 0 for node in self._all_nodes}
        for node in self._all_nodes:
            # Use set to avoid double-counting duplicate inputs
            # (e.g. Attn nodes where query_in == key_in)
            for inp in set(node.inputs):
                if inp in in_degree:
                    in_degree[node] += 1

        queue = deque(
            sorted(
                (n for n, deg in in_degree.items() if deg == 0),
                key=lambda n: n.node_id,
            )
        )
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for consumer in sorted(
                self._consumers.get(node, set()), key=lambda n: n.node_id
            ):
                in_degree[consumer] -= 1
                if in_degree[consumer] == 0:
                    queue.append(consumer)
        return order

    def _compute_critical_paths(self):
        """Longest chain from each node to the output, computed bottom-up."""
        # Process in reverse topo order (output first)
        self._critical_path[self._output_node] = 0
        for node in reversed(self._topo_order):
            if node not in self._critical_path:
                self._critical_path[node] = 0
            for inp in node.inputs:
                if inp in self._all_nodes:
                    dist = self._critical_path[node] + 1
                    if dist > self._critical_path.get(inp, 0):
                        self._critical_path[inp] = dist

    def get_consumers(self, node: Node) -> Set[Node]:
        return self._consumers.get(node, set())

    def get_topological_order(self) -> List[Node]:
        return self._topo_order

    def get_critical_path_length(self, node: Node) -> int:
        return self._critical_path.get(node, 0)

    def get_all_nodes(self) -> Set[Node]:
        return self._all_nodes

    def is_input_node(self, node: Node) -> bool:
        # LiteralValue is deliberately NOT an input node.  Constants are
        # first-class *schedulable* nodes, materialized just-in-time via the
        # compute_literal_value op near their consumer and freed after use —
        # not residual columns pre-allocated at layer 0 and held for the whole
        # network.  See constants_plan.md.
        return isinstance(node, (Embedding, InputNode))

    def is_ready(self, node: Node, available: Set[Node]) -> bool:
        """Check if all of a node's inputs are available.

        Concatenate nodes are transparent — we check their leaf children instead.
        Scheduling predecessors (set by hint helpers like
        ``sequential_scope``) also gate readiness but aren't data inputs.
        """
        for inp in node.inputs:
            if isinstance(inp, Concatenate):
                for leaf in flatten_concat_nodes([inp]):
                    if leaf not in available:
                        return False
            else:
                if inp not in available:
                    return False
        for pred in node.scheduling_predecessors:
            if pred not in available:
                return False
        return True

    def get_ready_nodes(self, available: Set[Node]) -> Set[Node]:
        """Return all nodes whose inputs are all in the available set.

        Excludes nodes already in available and Concatenate nodes
        (which are never placed in the residual stream).
        """
        ready = set()
        for node in self._all_nodes:
            if node in available:
                continue
            if isinstance(node, Concatenate):
                continue
            if self.is_ready(node, available):
                ready.add(node)
        return ready
