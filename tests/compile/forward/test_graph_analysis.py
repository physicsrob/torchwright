import torch

from examples.adder import create_network_parts
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.graph import Add, Concatenate, Linear, ReLU
from torchwright.graph.misc import InputNode


def _make_linear(inp, d_out, name=""):
    """Helper: zero-bias linear node."""
    w = torch.randn(len(inp), d_out)
    b = torch.zeros(d_out)
    return Linear(inp, w, b, name=name)


def test_simple_chain():
    """Input -> Linear -> ReLU -> Linear: verify topo order, critical path, consumers."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = _make_linear(x, 4, "l1")
    r = ReLU(l1, name="relu")
    l2 = _make_linear(r, 2, "l2")

    graph = GraphAnalyzer(l2)

    # Topo order: every node appears after its inputs
    order = graph.get_topological_order()
    idx = {node: i for i, node in enumerate(order)}
    assert idx[x] < idx[l1] < idx[r] < idx[l2]

    # Critical path: distance to output
    assert graph.get_critical_path_length(l2) == 0
    assert graph.get_critical_path_length(r) == 1
    assert graph.get_critical_path_length(l1) == 2
    assert graph.get_critical_path_length(x) == 3

    # Consumers
    assert graph.get_consumers(x) == {l1}
    assert graph.get_consumers(l1) == {r}
    assert graph.get_consumers(r) == {l2}
    assert graph.get_consumers(l2) == set()


def test_diamond_graph():
    """Input -> A, Input -> B, Add(A, B): verify readiness progression."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    a = _make_linear(x, 4, "a")
    b = _make_linear(x, 4, "b")
    out = Add(a, b, name="sum")

    graph = GraphAnalyzer(out)

    # Initially only x is available — a and b are ready
    available = {x}
    ready = graph.get_ready_nodes(available)
    assert a in ready
    assert b in ready
    assert out not in ready  # needs a and b

    # After computing a, out still not ready (needs b)
    available.add(a)
    ready = graph.get_ready_nodes(available)
    assert b in ready
    assert out not in ready

    # After computing b, out is ready
    available.add(b)
    ready = graph.get_ready_nodes(available)
    assert out in ready


def test_concatenate_transparency():
    """Concat([A, B]) -> Linear: readiness depends on A, B not the Concat node."""
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    cat = Concatenate([a, b])
    l = _make_linear(cat, 2, "l")

    graph = GraphAnalyzer(l)

    # Concat should never appear in ready nodes
    available = {a, b}
    ready = graph.get_ready_nodes(available)
    assert l in ready
    assert cat not in ready  # Concat is never "ready" — it's transparent

    # With only a available, l is not ready (needs b through concat)
    available = {a}
    ready = graph.get_ready_nodes(available)
    assert l not in ready


def test_adder_graph():
    """Load the 3-digit adder graph, verify topo order valid and all nodes reachable.

    Analyzes the lowered copy: GraphAnalyzer is pure analysis of the
    graph the lowering boundary produces.
    """
    from torchwright.compiler.lower import lower

    source_output, embedding = create_network_parts()
    output_node = lower(source_output).output_node
    graph = GraphAnalyzer(output_node)

    all_nodes = graph.get_all_nodes()
    order = graph.get_topological_order()

    # Topo order contains all non-Concatenate nodes
    order_set = set(order)
    for node in all_nodes:
        if not isinstance(node, Concatenate):
            assert node in order_set, f"{node} missing from topo order"

    # Topo order is valid: every node appears after all its inputs
    idx = {node: i for i, node in enumerate(order)}
    for node in order:
        for inp in node.inputs:
            if inp in idx:  # Concatenates may not be in order
                assert idx[inp] < idx[node], (
                    f"{inp} should come before {node} in topo order"
                )

    # Output node has critical path 0
    assert graph.get_critical_path_length(output_node) == 0

    # Input nodes exist and have positive critical path
    input_nodes = [n for n in all_nodes if graph.is_input_node(n)]
    assert len(input_nodes) > 0
    for n in input_nodes:
        assert graph.get_critical_path_length(n) > 0


def test_ready_nodes_progression():
    """Iteratively add ready nodes to available set — all nodes eventually computed."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = _make_linear(x, 4, "l1")
    r = ReLU(l1, name="relu")
    a = _make_linear(r, 4, "a")
    b = _make_linear(r, 4, "b")
    out = Add(a, b, name="out")

    graph = GraphAnalyzer(out)

    # Start with input nodes
    available = {n for n in graph.get_all_nodes() if graph.is_input_node(n)}
    assert x in available

    # Iterate: add all ready nodes until output is computed
    iterations = 0
    while out not in available:
        ready = graph.get_ready_nodes(available)
        assert len(ready) > 0, "No progress — stuck"
        available |= ready
        iterations += 1
        assert iterations < 100, "Too many iterations"

    assert out in available
