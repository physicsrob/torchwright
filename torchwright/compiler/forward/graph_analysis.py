from collections import defaultdict, deque
from typing import Dict, List, Set

from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import (
    Node,
    Assert,
    Concatenate,
    DebugWatch,
    InputNode,
    Embedding,
)
from torchwright.graph.value_type import tightened_with


class GraphAnalyzer:
    """Pre-computes graph metadata needed by the forward compiler scheduler.

    Assert nodes are stripped (in-place) during initialization so they
    don't show up in the scheduled graph — they're a reference-eval and
    probe-compiled concern, not a compiled-weights concern.
    """

    def __init__(self, output_node: Node):
        # Strip Assert nodes in-place so scheduler/weight_writer/compile
        # see a graph without them.  Reference-eval callers that run
        # *before* GraphAnalyzer still see Asserts and fire their
        # predicates as expected.
        self._stripped_asserts: List[Assert] = []
        self._assert_aliases: Dict[Assert, Node] = {}
        self._stripped_watches: List[DebugWatch] = []
        self._watch_aliases: Dict[DebugWatch, Node] = {}
        output_node = self._strip_asserts(output_node)

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

    def _strip_asserts(self, output_node: Node) -> Node:
        """Mutate the graph in-place to remove Assert pass-through nodes.

        Rewires every consumer of an Assert to point at the Assert's
        underlying input.  If ``output_node`` itself is an Assert (or a
        chain of them), follow the chain to the real output and return
        that.  The stripped Assert nodes are recorded on
        ``self._stripped_asserts`` so ``probe_compiled`` can find them
        after compile.

        Idempotent: if called on an already-stripped graph, finds
        nothing to do and returns the same output.
        """
        pre_strip_nodes = get_ancestor_nodes({output_node})
        asserts = [n for n in pre_strip_nodes if isinstance(n, Assert)]
        watches = [n for n in pre_strip_nodes if isinstance(n, DebugWatch)]
        if not asserts and not watches:
            return output_node

        def unwrap(node: Node) -> Node:
            while isinstance(node, (Assert, DebugWatch)):
                node = node.inputs[0]
            return node

        # Transfer each Assert's claimed range and tightened
        # input_ranges onto the node it wraps, so downstream graph
        # analysis that runs after stripping still sees the
        # strengthened type and tight ranges.
        for a in asserts:
            if a.claimed_type is None:
                continue
            target = unwrap(a)
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
                        torch.equal(tighter_lo, old_lo)
                        and torch.equal(tighter_hi, old_hi)
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

        for node in pre_strip_nodes:
            if isinstance(node, (Assert, DebugWatch)):
                continue
            for i, inp in enumerate(node.inputs):
                if isinstance(inp, (Assert, DebugWatch)):
                    node.inputs[i] = unwrap(inp)

        self._stripped_asserts.extend(asserts)
        for a in asserts:
            self._assert_aliases[a] = unwrap(a)
        self._stripped_watches.extend(watches)
        for w in watches:
            self._watch_aliases[w] = unwrap(w)
        return unwrap(output_node)

    def get_stripped_asserts(self) -> List[Assert]:
        """Return the Assert nodes removed during initialization."""
        return list(self._stripped_asserts)

    def get_assert_aliases(self) -> Dict[Assert, Node]:
        """Return mapping from stripped Assert nodes to their underlying targets."""
        return dict(self._assert_aliases)

    def get_stripped_watches(self) -> List[DebugWatch]:
        """Return the DebugWatch nodes removed during initialization."""
        return list(self._stripped_watches)

    def get_watch_aliases(self) -> Dict[DebugWatch, Node]:
        """Return mapping from stripped DebugWatch nodes to their underlying targets."""
        return dict(self._watch_aliases)

    def get_output_node(self) -> Node:
        """Return the effective output node after Assert stripping.

        Differs from the ``output_node`` the caller passed in only when
        that node was itself an Assert — in which case this returns the
        Assert's underlying input.  Callers that track whether the
        root's value has been computed should consult this method
        rather than their original reference.
        """
        return self._output_node

    def _build_topo_order(self) -> List[Node]:
        """Kahn's algorithm — returns nodes with inputs before dependents."""
        in_degree: Dict[Node, int] = {node: 0 for node in self._all_nodes}
        for node in self._all_nodes:
            # Use set to avoid double-counting duplicate inputs
            # (e.g. Attn nodes where query_in == key_in)
            for inp in set(node.inputs):
                if inp in in_degree:
                    in_degree[node] += 1

        queue = deque(n for n, deg in in_degree.items() if deg == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for consumer in self._consumers.get(node, set()):
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
