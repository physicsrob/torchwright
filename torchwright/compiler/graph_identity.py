"""Canonical node identity, topology fingerprints, and cross-process remap.

A torchwright graph rebuilt by the same deterministic construction code
produces the same topology but different ``node_id`` values (the raw
counter is process-cumulative).  Everything that persists per-node data
across processes — the CP-SAT schedule cache, the ONNX debug sidecar —
keys nodes by a **canonical id** instead: preorder-DFS numbering from
the output node following each node's ordered ``inputs`` list, which
depends only on the topology.

Runtime checks and range claims are node *metadata* (``node.checks`` /
``node.claimed_type``) and do not participate in the encoding, so
attaching or removing them never changes a canonical id or fingerprint.
(Historically they were Assert/DebugWatch wrapper nodes that this
module's traversals stepped through transparently — the encoding is
byte-identical across that representation change; the pinned goldens in
``tests/compile/test_topology_golden.py`` enforce it.)  Nothing here
mutates the graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torchwright.graph import Node


@lru_cache(maxsize=1)
def compiler_code_fingerprint() -> str:
    """Content hash of every ``.py`` source in the torchwright package.

    Participates in the schedule-cache fingerprint so ANY torchwright
    change invalidates cached schedules and forces a fresh solve.  The
    solver's search behavior is a function of compiler code the topology
    hash cannot see (the warm-start heuristic, the CP-SAT model,
    lowering passes); before this, invalidation was a hand-bumped
    generation string that a compiler change could — and did — forget to
    bump, silently replaying a stale schedule.  A spurious re-solve
    after a cosmetic edit costs one solver budget; a stale schedule
    mislabels the production artifact.

    Content-hashed rather than git-derived so it works where ``.git`` is
    absent (the Modal compile container mounts sources only) and is
    dirty-aware by construction.  Cached per process — sources don't
    change mid-run.
    """
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def _canonical_walk(output_node: Node) -> List[Node]:
    """Nodes in canonical (preorder-DFS) order.

    Preorder DFS from the output following each node's ORDERED
    ``inputs`` list; first visit assigns the next canonical number.
    """
    seen: set = set()
    ordered: List[Node] = []
    stack: List[Node] = [output_node]
    while stack:
        n = stack.pop()
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
    fingerprint and the debug-sidecar fingerprint.
    """
    ordered = _canonical_walk(output_node)
    canon = {n.node_id: i for i, n in enumerate(ordered)}
    topo = []
    for i, n in enumerate(ordered):
        ins = tuple(
            canon[inp.node_id]
            for inp in (getattr(n, "inputs", None) or [])
            if inp.node_id in canon
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
    cancel_slack: Optional[int],
    policy,
    reserve_residual: int = 0,
    bias: bool = True,
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

    ``compiler_code`` (:func:`compiler_code_fingerprint`) keys the cache on
    the torchwright sources themselves, so any compiler change — solver
    model, warm-start heuristic, lowering — invalidates every cached
    schedule and forces a fresh solve.  It replaces the earlier hand-bumped
    generation mechanisms (the ``cancel_window: hint-aware-v1`` string, the
    ``assume_zero_init`` payload-layout change), which relied on a human
    remembering that a compiler edit changed the solve; the 2026-07
    warm-start rewrite wasn't bumped and replayed a stale schedule into a
    production compile.  Adding the key changes the payload layout for
    every graph — the resulting global one-time invalidation is intended.
    """
    payload = {
        "topology": topology_entries(output_node),
        "d": d,
        "d_head": d_head,
        "d_hidden": d_hidden,
        "flex_routing": flex_routing,
        "cancel_slack": cancel_slack,
        "policy": asdict(policy) if policy is not None else None,
        "compiler_code": compiler_code_fingerprint(),
    }
    if reserve_residual:
        payload["reserve_residual"] = reserve_residual
    # bias=False reserves hidden slot 0 (the constant lane), changing the
    # modeled slot capacity, hence the schedule — it MUST key the cache.
    # Added only when False (the ``reserve_residual`` compatibility pattern)
    # so every existing biased-compile entry keeps hashing byte-identically.
    if not bias:
        payload["bias"] = False
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

    Checks are node metadata outside the encoding, so a rebuilt graph
    with more or fewer attached checks than the compiled one still
    matches — by design, so debug instrumentation can be added to the
    rebuild without invalidating the sidecar.
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
