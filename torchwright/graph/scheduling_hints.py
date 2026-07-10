"""Compiler hints for scheduler behavior.

Hints are graph-level annotations that guide the forward-compile
scheduler beyond what it can infer from topology alone.  They attach
to :class:`torchwright.graph.node.Node` via ``scheduling_predecessors``
— a set of nodes that must be in ``computed_nodes`` before this node
can be scheduled, over and above the natural data-flow dependencies.

The primary use case: serializing a group of parallel chains that
would otherwise create residual-stream pressure.  See
:func:`sequential_scope`.
"""

from typing import Callable, List, Optional

from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.graph.node import Node, global_node_id
from torchwright.graph.misc import Concatenate


def add_scheduling_dependency(node: Node, depends_on: Node) -> None:
    """Declare that ``node`` must not be scheduled until ``depends_on``
    is in ``computed_nodes``.

    This does NOT add a data input — the compiled op won't read from
    ``depends_on``.  It only affects scheduler ordering via
    :meth:`GraphAnalyzer.is_ready`.
    """
    node.scheduling_predecessors.add(depends_on)


def assert_hints_intact(output_node: Node, context: str) -> None:
    """Every scheduling predecessor of every reachable node is itself
    reachable from ``output_node``.

    The same invariant ``clone_graph`` asserts during its remap — but the
    clone runs *before* fusion, so its check cannot see a hint orphaned by
    a later pass.  A dangling hint's downstream symptom is far from its
    cause: ``Compilation did not converge in N layers`` at ``optimize=0``,
    a bare ``KeyError: <node_id>`` inside CP-SAT's bound propagation at
    ``optimize>=1`` — neither names a node or mentions scheduling.  Calling
    this after each graph-mutating pass converts both into an error naming
    the node, the missing predecessor, and (via ``context``) the pass that
    orphaned it.

    O(V + hint edges); walks ``inputs`` only (data reachability — a hint
    edge must point at a node something will actually compute).
    """
    reachable = set()
    stack: List[Node] = [output_node]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        stack.extend(cur.inputs)
    for node in reachable:
        for pred in node.scheduling_predecessors:
            if pred not in reachable:
                raise RuntimeError(
                    f"{context} orphaned a scheduling hint: {node!r} waits "
                    f"on {pred!r}, which is no longer reachable from the "
                    f"output and so can never be computed.  The scheduler "
                    f"would never find {node!r} ready — this graph cannot "
                    f"compile.  The pass named above removed a node that a "
                    f"hint edge depends on; its orphan gate (_blocks_orphan "
                    f"in graph/optimize.py) should have declined the rewrite."
                )


def _current_node_id() -> int:
    """Snapshot the current global_node_id counter."""
    import torchwright.graph.node as _node_mod

    return _node_mod.global_node_id


def _find_entry_nodes(terminal: Node, lo: int, hi: int) -> List[Node]:
    """Non-Concatenate nodes in id-range [lo, hi) reachable backward
    from ``terminal`` whose flattened-input ids are all < lo.

    An "entry node" is the earliest point in this iteration where
    gating scheduling is effective: downstream nodes in the iteration
    will naturally wait on their data inputs, so blocking at the entry
    blocks the whole iteration.
    """

    def in_range(n: Node) -> bool:
        return lo <= n.node_id < hi

    visited = set()
    entries: List[Node] = []
    stack: List[Node] = [terminal]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        if not in_range(cur):
            continue
        if isinstance(cur, Concatenate):
            # Walk through; Concatenates aren't scheduled themselves.
            for inp in cur.inputs:
                stack.append(inp)
            continue
        # Non-Concatenate in-range node.  Check if it's an entry:
        # all of its flattened inputs must be strictly pre-iteration.
        is_entry = True
        for inp in cur.inputs:
            flat = (
                flatten_concat_nodes([inp]) if isinstance(inp, Concatenate) else [inp]
            )
            for leaf in flat:
                if leaf.node_id >= lo:
                    is_entry = False
                    break
            if not is_entry:
                break
        if is_entry:
            entries.append(cur)
        # Keep walking backward through all inputs to find deeper entries.
        for inp in cur.inputs:
            stack.append(inp)
    return entries


def sequential_scope(
    factories: List[Callable[[], Node]],
    batch_size: int = 1,
) -> List[Node]:
    """Call ``factories`` in order, wiring scheduling deps so that at
    most ``batch_size`` iterations are in flight concurrently.

    Each factory builds one iteration's subgraph and returns its
    terminal node — the one that downstream code consumes.  After
    iteration ``i`` runs, its entry nodes (those whose flattened
    inputs are all pre-iteration) are given a scheduling predecessor:
    the terminal of iteration ``i - batch_size``.

    The scheduler's readiness check honors those predecessors, so
    iteration ``i`` can't start until iteration ``i - batch_size`` has
    been fully computed.  With ``batch_size=1``, iterations run
    strictly sequentially.  With ``batch_size=K``, up to K iterations
    are in flight at once.

    Parameters
    ----------
    factories
        One callable per iteration.  Each must return the iteration's
        terminal node.  Factories are called in order.
    batch_size
        Maximum concurrency.  Must be >= 1.

    Returns
    -------
    List of terminal nodes, one per iteration, in the order ``factories``
    were given.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    iterations: List[tuple] = []  # (lo_id, hi_id, terminal)
    for factory in factories:
        lo = _current_node_id()
        terminal = factory()
        hi = _current_node_id()
        if terminal is None:
            raise ValueError(
                "sequential_scope factories must return a terminal Node; "
                f"factory {len(iterations)} returned None"
            )
        iterations.append((lo, hi, terminal))

    for i in range(batch_size, len(iterations)):
        gate_terminal = iterations[i - batch_size][2]
        lo_i, hi_i, term_i = iterations[i]
        for entry in _find_entry_nodes(term_i, lo_i, hi_i):
            add_scheduling_dependency(entry, gate_terminal)

    return [t for _, _, t in iterations]
