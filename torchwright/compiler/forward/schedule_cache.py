"""Persistent cache of solved ``ScheduleAssignment``s.

Keyed by topology + solve geometry + **compiler code identity**
(:func:`torchwright.compiler.graph_identity.compiler_code_fingerprint`):
the solve outcome is a function of the solver model and warm-start code,
not just the graph, so any torchwright change misses and re-solves.
Within one code state, winning the search once per graph shape is enough;
this cache makes the win durable so later compiles of the same shape skip
the solver entirely (measured motivation: at d=8192 the 180s solve is a
50-64 layer lottery and the proven optimum took ~2 minutes to find —
re-fighting that per compile is pure waste).  Hits are additionally gated
on the requested optimize level (``load_assignment(min_optimize=...)``) so
a bigger budget in the config buys a bigger solve, once, instead of
silently replaying a smaller budget's draw.

Opt-in: the cache is active only when ``TW_SCHEDULE_CACHE_DIR`` is set
(one JSON file per fingerprint inside that directory).  Replayed
schedules pass through ``DirectedLayerScheduler`` and the compiler's
I1-I4 invariants, so a stale entry fails loudly rather than corrupting a
compile.  Callers surface hits as ``status_name="CACHED"`` — never
silently (see docs/cpsat_scheduler.md on fallback provenance).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from torchwright.compiler.graph_identity import (
    canonical_ids as _canonical_ids,
)
from torchwright.compiler.graph_identity import (
    graph_fingerprint,
)
from torchwright.graph import Node

from .cpsat_scheduler import ScheduleAssignment

__all__ = [
    "cache_dir",
    "graph_fingerprint",
    "load_assignment",
    "store_assignment",
]

_ENV_DIR = "TW_SCHEDULE_CACHE_DIR"


def cache_dir() -> Path | None:
    """The active cache directory, or None when the cache is disabled."""
    value = os.environ.get(_ENV_DIR, "").strip()
    return Path(value) if value else None


# _canonical_ids and graph_fingerprint live in
# torchwright.compiler.graph_identity (shared with the ONNX debug
# sidecar) and are re-exported here for existing callers.  The
# fingerprint payload is byte-identical to the pre-extraction
# implementation for compiled (wrapper-stripped) graphs, so existing
# cache entries keep hitting.


def load_assignment(
    fingerprint: str,
    output_node: Node,
    *,
    min_optimize: int = 0,
) -> tuple[ScheduleAssignment, dict[str, Any]] | None:
    """Return (assignment, meta) for a cached fingerprint, or None.

    Stored keys are canonical ids; they are remapped onto the CURRENT
    graph's node ids via the same deterministic traversal.

    ``min_optimize`` gates hits on the entry's validated optimize level
    (``meta["optimize"]``): an entry solved at a lower level than the
    request misses, so raising ``optimize`` in a config actually buys a
    bigger solve instead of silently replaying the smaller budget's
    draw.  The level is validated, not just provenance — a higher-level
    re-solve that fails to beat the cached schedule upgrades the entry's
    level in place (see :func:`store_assignment`), so the re-solve
    happens once per level, not per compile.
    """
    base = cache_dir()
    if base is None:
        return None
    path = base / f"{fingerprint}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("meta", {}).get("optimize", 0) or 0) < min_optimize:
        return None
    current_by_canon = {c: nid for nid, c in _canonical_ids(output_node).items()}

    def _remap(table: dict[str, Any]) -> dict[int, Any]:
        return {current_by_canon[int(k)]: v for k, v in table.items()}

    try:
        assignment = ScheduleAssignment(
            node_to_layer=_remap(data["node_to_layer"]),
            node_to_cancel_layer=_remap(data["node_to_cancel_layer"]),
            node_to_routing=_remap(data["node_to_routing"]),
            n_layers=data["n_layers"],
            # Bare indexing (NOT .get): a pre-mechanism cache file lacks this
            # key and must miss cleanly here, not fabricate a mechanism-less
            # assignment that fails later inside the directed replay.
            node_to_cancel_mech=_remap(data["node_to_cancel_mech"]),
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
    meta: dict[str, Any],
    output_node: Node,
) -> bool:
    """Persist ``assignment`` unless an equal-or-better entry exists.

    Returns True when the entry was written.

    The schedule ratchet is one-way by scale-independent realized objective
    blocks when the caller supplies ``meta["realized_objective_blocks"]``.
    Scaled totals are comparable only when both entries record the same scale;
    legacy callers otherwise fall back to fewer layers. The entry's
    validated optimize level (``meta["optimize"]``) ratchets
    independently: when a higher-level solve fails to beat the cached
    schedule, that is proof the entry is as good as that level produces,
    so the level is upgraded in place and future requests at that level
    replay it (the ``min_optimize`` gate in :func:`load_assignment`)
    instead of re-solving every compile.
    """
    base = cache_dir()
    if base is None:
        return False
    prior_path = base / f"{fingerprint}.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_meta = prior.get("meta", {})
        prior_blocks_raw = prior_meta.get("realized_objective_blocks")
        new_blocks_raw = meta.get("realized_objective_blocks")
        prior_blocks = (
            tuple(int(value) for value in prior_blocks_raw)
            if isinstance(prior_blocks_raw, (list, tuple))
            and len(prior_blocks_raw) == 2
            else None
        )
        new_blocks = (
            tuple(int(value) for value in new_blocks_raw)
            if isinstance(new_blocks_raw, (list, tuple)) and len(new_blocks_raw) == 2
            else None
        )
        prior_objective = prior_meta.get("realized_objective")
        new_objective = meta.get("realized_objective")
        prior_scale = prior_meta.get("objective_scale")
        new_scale = meta.get("objective_scale")
        if prior_blocks is not None and new_blocks is not None:
            prior_dominates = prior_blocks <= new_blocks
        elif (
            prior_objective is not None
            and new_objective is not None
            and prior_scale is not None
            and new_scale is not None
            and int(prior_scale) == int(new_scale)
        ):
            prior_dominates = int(prior_objective) <= int(new_objective)
        else:
            prior_dominates = prior.get("n_layers", 1 << 30) <= assignment.n_layers
        if prior_dominates:
            new_level = int(meta.get("optimize", 0) or 0)
            if new_level > int(prior_meta.get("optimize", 0) or 0):
                prior_meta["optimize"] = new_level
                prior["meta"] = prior_meta
                tmp = prior_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(prior), encoding="utf-8")
                tmp.replace(prior_path)
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
        "node_to_cancel_mech": {
            canon[k]: v for k, v in assignment.node_to_cancel_mech.items() if k in canon
        },
        "n_layers": assignment.n_layers,
        "meta": meta,
    }
    path = base / f"{fingerprint}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return True
