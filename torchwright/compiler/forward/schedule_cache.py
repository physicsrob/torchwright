"""Persistent cache of solved ``ScheduleAssignment``s, keyed by topology.

A CP-SAT schedule depends only on the graph topology and the solve
geometry — not on weight values, assets, or git state.  Winning the
search once per graph shape is enough; this cache makes the win durable
so later compiles of the same shape skip the solver entirely (measured
motivation: at d=8192 the 180s solve is a 50-64 layer lottery and the
proven optimum took ~2 minutes to find — re-fighting that per compile is
pure waste).

Opt-in: the cache is active only when ``TW_SCHEDULE_CACHE_DIR`` is set
(one JSON file per fingerprint inside that directory).  Replayed
schedules pass through ``DirectedLayerScheduler`` and the compiler's
I1-I4 invariants, so a stale entry fails loudly rather than corrupting a
compile.  Callers surface hits as ``status_name="CACHED"`` — never
silently (see docs/cpsat_scheduler.md on fallback provenance).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.graph import Node
from torchwright.graph.pos_encoding import PosEncoding

from .cpsat_scheduler import ScheduleAssignment

_ENV_DIR = "TW_SCHEDULE_CACHE_DIR"


def cache_dir() -> Optional[Path]:
    """The active cache directory, or None when the cache is disabled."""
    value = os.environ.get(_ENV_DIR, "").strip()
    return Path(value) if value else None


def _canonical_ids(output_node: Node) -> Dict[int, int]:
    """Map current ``node_id`` -> canonical id, independent of creation order.

    Preorder DFS from the output following each node's ORDERED ``inputs``
    list; first visit assigns the next canonical number.  The raw global
    node-id counter is process-cumulative: a warm process that builds the
    graph twice (observed: Modal containers reuse one process across
    compiles) shifts every id, which would change the fingerprint and
    silently miss the cache.  Canonical ids depend only on the topology.
    """
    canon: Dict[int, int] = {}
    stack = [output_node]
    while stack:
        n = stack.pop()
        if n.node_id in canon:
            continue
        canon[n.node_id] = len(canon)
        # Reversed keeps the first input on top of the stack (preorder).
        stack.extend(reversed(getattr(n, "inputs", None) or []))
    return canon


def graph_fingerprint(
    output_node: Node,
    pos_encoding: PosEncoding,
    *,
    d: int,
    d_head: int,
    d_hidden: int,
    flex_routing: bool,
    assume_zero_init: bool,
    cancel_slack: Optional[int],
    policy: Optional[SchedulingPolicy],
) -> str:
    """Topology + geometry hash over CANONICAL node ids (see
    ``_canonical_ids``) — stable across processes and warm containers; any
    change to graph construction still changes the fingerprint and misses
    (correct by construction)."""
    canon = _canonical_ids(output_node)
    graph = GraphAnalyzer(output_node)
    nodes = sorted(
        (n for n in graph.get_all_nodes() if n.node_id in canon),
        key=lambda n: canon[n.node_id],
    )
    topo = [
        (
            canon[n.node_id],
            type(n).__name__,
            len(n),
            tuple(
                canon[inp.node_id]
                for inp in (getattr(n, "inputs", None) or [])
                if inp.node_id in canon
            ),
        )
        for n in nodes
    ]
    payload = {
        "topology": topo,
        "pos_width": len(pos_encoding),
        "d": d,
        "d_head": d_head,
        "d_hidden": d_hidden,
        "flex_routing": flex_routing,
        "assume_zero_init": assume_zero_init,
        "cancel_slack": cancel_slack,
        "policy": asdict(policy) if policy is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_assignment(
    fingerprint: str,
    output_node: Node,
) -> Optional[Tuple[ScheduleAssignment, Dict[str, Any]]]:
    """Return (assignment, meta) for a cached fingerprint, or None.

    Stored keys are canonical ids; they are remapped onto the CURRENT
    graph's node ids via the same deterministic traversal."""
    base = cache_dir()
    if base is None:
        return None
    path = base / f"{fingerprint}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    current_by_canon = {c: nid for nid, c in _canonical_ids(output_node).items()}

    def _remap(table: Dict[str, Any]) -> Dict[int, Any]:
        return {current_by_canon[int(k)]: v for k, v in table.items()}

    try:
        assignment = ScheduleAssignment(
            node_to_layer=_remap(data["node_to_layer"]),
            node_to_cancel_layer=_remap(data["node_to_cancel_layer"]),
            node_to_routing=_remap(data["node_to_routing"]),
            n_layers=data["n_layers"],
        )
    except KeyError:
        # Canonical id absent from the current graph: the entry predates a
        # construction change the fingerprint failed to capture.  Treat as
        # a miss rather than replaying a wrong schedule.
        return None
    return assignment, data.get("meta", {})


def store_assignment(
    fingerprint: str,
    assignment: ScheduleAssignment,
    meta: Dict[str, Any],
    output_node: Node,
) -> bool:
    """Persist ``assignment`` unless an equal-or-better entry exists.

    Returns True when the entry was written.
    """
    base = cache_dir()
    if base is None:
        return False
    prior_path = base / f"{fingerprint}.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("n_layers", 1 << 30) <= assignment.n_layers:
            return False
    base.mkdir(parents=True, exist_ok=True)
    canon = _canonical_ids(output_node)
    if not set(assignment.node_to_layer) <= set(canon):
        # A scheduled node is unreachable from the output via inputs — it
        # cannot be keyed canonically; skip caching rather than store a
        # partial schedule.
        return False
    payload = {
        "node_to_layer": {canon[k]: v for k, v in assignment.node_to_layer.items()},
        "node_to_cancel_layer": {
            canon[k]: v
            for k, v in assignment.node_to_cancel_layer.items()
            if k in canon
        },
        "node_to_routing": {canon[k]: v for k, v in assignment.node_to_routing.items()},
        "n_layers": assignment.n_layers,
        "meta": meta,
    }
    path = base / f"{fingerprint}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return True
