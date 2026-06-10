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
    """Topology + geometry hash.

    Includes the node ids themselves: a cached assignment is keyed by
    ``node_id``, which is only meaningful if graph construction replays
    identically — any change to construction order changes the ids, the
    fingerprint, and therefore misses (correct by construction).
    """
    graph = GraphAnalyzer(output_node)
    nodes = sorted(graph.get_all_nodes(), key=lambda n: n.node_id)
    topo = [
        (
            n.node_id,
            type(n).__name__,
            len(n),
            tuple(inp.node_id for inp in (getattr(n, "inputs", None) or [])),
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
) -> Optional[Tuple[ScheduleAssignment, Dict[str, Any]]]:
    """Return (assignment, meta) for a cached fingerprint, or None."""
    base = cache_dir()
    if base is None:
        return None
    path = base / f"{fingerprint}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    assignment = ScheduleAssignment(
        node_to_layer={int(k): v for k, v in data["node_to_layer"].items()},
        node_to_cancel_layer={
            int(k): v for k, v in data["node_to_cancel_layer"].items()
        },
        node_to_routing={int(k): v for k, v in data["node_to_routing"].items()},
        n_layers=data["n_layers"],
    )
    return assignment, data.get("meta", {})


def store_assignment(
    fingerprint: str,
    assignment: ScheduleAssignment,
    meta: Dict[str, Any],
) -> bool:
    """Persist ``assignment`` unless an equal-or-better entry exists.

    Returns True when the entry was written.
    """
    base = cache_dir()
    if base is None:
        return False
    existing = load_assignment(fingerprint)
    if existing is not None and existing[0].n_layers <= assignment.n_layers:
        return False
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "node_to_layer": assignment.node_to_layer,
        "node_to_cancel_layer": assignment.node_to_cancel_layer,
        "node_to_routing": assignment.node_to_routing,
        "n_layers": assignment.n_layers,
        "meta": meta,
    }
    path = base / f"{fingerprint}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return True
