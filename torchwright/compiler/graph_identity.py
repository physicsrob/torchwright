"""Canonical node identity, topology fingerprints, and cross-process remap.

A torchwright graph rebuilt by the same deterministic construction code
produces the same topology but different ``node_id`` values (the raw
counter is process-cumulative).  Everything that persists per-node data
across processes — the CP-SAT schedule cache, the ONNX debug sidecar —
keys nodes by a **canonical id** instead: preorder-DFS numbering from
the output node following each node's ordered ``inputs`` list, which
depends only on the topology.

Assert and DebugWatch wrappers are **transparent** to every function in
this module: the traversal steps through them to the wrapped node and
never assigns them an id.  This matches the compiled graph exactly —
``GraphAnalyzer`` strips both wrapper kinds in-place before scheduling —
so a freshly *rebuilt* graph that still carries its Assert wrappers
canonicalizes (and fingerprints) identically to the stripped graph the
compiler actually processed.  Unlike ``GraphAnalyzer``, nothing here
mutates the graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from torchwright.graph import Node
from torchwright.graph.misc import Assert, DebugWatch


def unwrap_debug(node: Node) -> Node:
    """Step through Assert/DebugWatch wrapper chains to the wrapped node."""
    while isinstance(node, (Assert, DebugWatch)):
        node = node.inputs[0]
    return node


def _canonical_walk(output_node: Node) -> List[Node]:
    """Nodes in canonical (preorder-DFS) order, wrappers skipped.

    Preorder DFS from the output following each node's ORDERED
    ``inputs`` list; first visit assigns the next canonical number.
    Assert/DebugWatch wrappers are stepped through transparently — the
    wrapped node is visited at the wrapper's position — so the order is
    identical to walking the ``GraphAnalyzer``-stripped graph.
    """
    seen: set = set()
    ordered: List[Node] = []
    stack: List[Node] = [output_node]
    while stack:
        n = unwrap_debug(stack.pop())
        if n.node_id in seen:
            continue
        seen.add(n.node_id)
        ordered.append(n)
        # Reversed keeps the first input on top of the stack (preorder).
        stack.extend(reversed(getattr(n, "inputs", None) or []))
    return ordered


def canonical_ids(output_node: Node) -> Dict[int, int]:
    """Map current ``node_id`` -> canonical id, independent of creation order."""
    return {n.node_id: i for i, n in enumerate(_canonical_walk(output_node))}


def nodes_by_canonical_id(output_node: Node) -> Dict[int, Node]:
    """Map canonical id -> live node object (the remap direction loaders need)."""
    return dict(enumerate(_canonical_walk(output_node)))


def topology_entries(output_node: Node) -> List[tuple]:
    """Per-node ``(canon_id, type_name, width, input_canon_ids)`` tuples.

    The hashable topology encoding shared by the schedule-cache
    fingerprint and the debug-sidecar fingerprint.  Inputs are unwrapped
    before lookup, so an Assert between producer and consumer does not
    change the encoding.
    """
    ordered = _canonical_walk(output_node)
    canon = {n.node_id: i for i, n in enumerate(ordered)}
    topo = []
    for i, n in enumerate(ordered):
        ins = tuple(
            canon[unwrap_debug(inp).node_id]
            for inp in (getattr(n, "inputs", None) or [])
            if unwrap_debug(inp).node_id in canon
        )
        topo.append((i, type(n).__name__, len(n), ins))
    return topo


def graph_fingerprint(
    output_node: Node,
    *,
    d: int,
    d_head: int,
    d_hidden: int,
    flex_routing: bool,
    assume_zero_init: bool,
    cancel_slack: Optional[int],
    policy,
    reserve_residual: int = 0,
) -> str:
    """Topology + geometry + solver-knob hash for the CP-SAT schedule cache.

    Stable across processes and warm containers; any change to graph
    construction still changes the fingerprint and misses (correct by
    construction).  The payload layout is frozen — it must keep hashing
    byte-identically for existing graphs or every cached schedule entry
    silently misses.

    ``reserve_residual`` (residual columns withheld from the solver, e.g. the
    pinned-constant RMSNorm's 1–2) changes the modeled residual capacity, hence
    the schedule, so it MUST participate in the key.  It is added only when
    non-zero so the common case (no reservation) keeps hashing byte-identically
    to the pre-feature layout and existing cache entries still hit.
    """
    payload = {
        "topology": topology_entries(output_node),
        "d": d,
        "d_head": d_head,
        "d_hidden": d_hidden,
        "flex_routing": flex_routing,
        "assume_zero_init": assume_zero_init,
        "cancel_slack": cancel_slack,
        "policy": asdict(policy) if policy is not None else None,
    }
    if reserve_residual:
        payload["reserve_residual"] = reserve_residual
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def debug_fingerprint(
    output_node: Node,
    *,
    d: int,
    d_head: int,
) -> str:
    """Structural fingerprint for the ONNX debug sidecar.

    Deliberately narrower than :func:`graph_fingerprint`: only the
    topology and the residual geometry (which fix the column layout)
    participate.  Solver knobs don't — the debug loader has no way to
    know them, and they don't change which node lives where in a way
    the sidecar doesn't already record explicitly.

    Because the topology encoding is wrapper-transparent, a rebuilt
    graph with more or fewer Assert/DebugWatch wrappers than the
    compiled one still matches — by design, so debug instrumentation
    can be added to the rebuild without invalidating the sidecar.
    """
    payload = {
        "format": "torchwright.debug.v1",
        "topology": topology_entries(output_node),
        "d": d,
        "d_head": d_head,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_cols(cols: List[int]) -> List[Tuple[int, int]]:
    """Run-length encode an ORDERED column list as ``(start, length)`` runs.

    Column order is meaningful (column k holds component k of the node's
    value), so runs only merge consecutive ASCENDING indices — decoding
    reproduces the exact original order.
    """
    runs: List[Tuple[int, int]] = []
    for c in cols:
        if runs and c == runs[-1][0] + runs[-1][1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((c, 1))
    return runs


def decode_cols(runs: List) -> List[int]:
    """Inverse of :func:`encode_cols` (accepts JSON-decoded lists)."""
    cols: List[int] = []
    for start, length in runs:
        cols.extend(range(int(start), int(start) + int(length)))
    return cols
