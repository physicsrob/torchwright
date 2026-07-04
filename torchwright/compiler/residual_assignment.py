from torchwright.graph import Node, Concatenate

from typing import Dict, List, Set, Optional

from torchwright.graph.misc import Assert, DebugWatch, Placeholder

global_state_id = 0


def _unwrap(node: Node) -> Node:
    """Step through Assert/DebugWatch wrappers to the wrapped node.

    Wrappers never receive residual columns (the lowering strip removes
    them from the compiler-private copy), but the *source* graph — which
    every post-compile node-keyed surface speaks — keeps its wrappers,
    so lookups must resolve through them: a source output Concatenate's
    children may be wrappers, and callers may hand the wrapped node an
    ops helper returned.
    """
    while isinstance(node, (Assert, DebugWatch)):
        node = node.inputs[0]
    return node


class ResidualStreamState:
    state_id: int
    name: str  # For debugging

    def __init__(self, name: str = ""):
        global global_state_id
        self.state_id = global_state_id
        self.name = name
        global_state_id += 1

    def __repr__(self):
        return f"ResidualStreamState({self.state_id}, name='{self.name}')"


class ResidualAssignment:
    """Maps graph nodes to column indices in the residual stream, per state.

    (Previously named ``FeatureAssignment``; renamed to avoid colliding with
    the circuits-literature meaning of *feature* — an interpretable direction
    in activation space.)
    """

    mapping: Dict[ResidualStreamState, Dict[Node, List[int]]]

    def __init__(self, states: Set[ResidualStreamState]):
        self.mapping = {state: {} for state in states}

    def assign(self, state: ResidualStreamState, node: Node, indices: List[int]):
        self.mapping[state][node] = indices

    def add_alias(self, alias: Node, target: Node):
        """Make *alias* resolve to the same indices as *target* in all states."""
        for state in self.mapping:
            if target in self.mapping[state]:
                self.mapping[state][alias] = self.mapping[state][target]

    def duplicate_state(self, src: ResidualStreamState, dst: ResidualStreamState):
        self.mapping[dst] = self.mapping[src]

    def has_node(self, state: ResidualStreamState, node: Node) -> bool:
        return _unwrap(node) in self.mapping[state]

    def get_nodes(self, state: ResidualStreamState) -> Set[Node]:
        return set(self.mapping.get(state, {}).keys())

    def get_node_indices(self, state: ResidualStreamState, node: Node) -> List[int]:
        node = _unwrap(node)
        if isinstance(node, Placeholder):
            return []
        elif isinstance(node, Concatenate):
            # Concatenate is a logical grouping — gather children's indices in order.
            indices = []
            for child in flatten_concat_nodes([node]):
                indices += self.get_node_indices(state, child)
            return indices
        return self.mapping[state][node]


def flatten_concat_nodes(nodes: List[Node]) -> List[Node]:
    """Flatten Concatenate nodes to leaf children, remove Placeholder.

    Assert/DebugWatch wrappers are transparent, like Concatenate: leaves
    are the wrapped nodes.  (On the compiler-private copy this is a
    no-op — the lowering strip removed every wrapper.)
    """
    simplified_other_nodes = []
    for n in nodes:
        n = _unwrap(n)
        if isinstance(n, Concatenate):
            simplified_other_nodes += flatten_concat_nodes(n.inputs)
        elif isinstance(n, Placeholder):
            pass
        else:
            simplified_other_nodes.append(n)
    return simplified_other_nodes
