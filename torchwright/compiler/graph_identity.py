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
from typing import cast

from torchwright.compiler.utils import resolve_n_heads
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


def _canonical_walk(output_node: Node) -> list[Node]:
    """Nodes in canonical (preorder-DFS) order.

    Preorder DFS from the output following each node's ORDERED
    ``inputs`` list; first visit assigns the next canonical number.
    """
    seen: set = set()
    ordered: list[Node] = []
    stack: list[Node] = [output_node]
    while stack:
        n = stack.pop()
        if n.node_id in seen:
            continue
        seen.add(n.node_id)
        ordered.append(n)
        # Reversed keeps the first input on top of the stack (preorder).
        stack.extend(reversed(getattr(n, "inputs", None) or []))
    return ordered


def canonical_ids(output_node: Node) -> dict[int, int]:
    """Map current ``node_id`` -> canonical id, independent of creation order."""
    return {n.node_id: i for i, n in enumerate(_canonical_walk(output_node))}


def nodes_by_canonical_id(output_node: Node) -> dict[int, Node]:
    """Map canonical id -> live node object (the remap direction loaders need)."""
    return dict(enumerate(_canonical_walk(output_node)))


def topology_entries(output_node: Node) -> list[tuple]:
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
    n_heads: int | None = None,
    d_hidden: int,
    flex_routing: bool,
    cancel_slack: int | None,
    policy,
    reserve_residual: int = 0,
    bias: bool = True,
    held_output_source: Node | None = None,
    held_output_target: Node | None = None,
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

    ``linear_support`` (2026-07, support-aware head charge): every
    ``Linear``'s live weight-row runs, keyed by canonical id.  The head
    charge became a function of weight *sparsity*
    (``realization.linear_attn_chunks``), which is a schedule input
    neither the topology hash nor ``compiler_code`` can see — two graphs
    with identical topology but different support schedule differently,
    and replaying across them would under-book heads in one direction (a
    "Ran out of attention heads" assert deep in the weight-writer, on a
    cache *hit*).
    """
    from torchwright.compiler.realization import live_weight_row_ranges
    from torchwright.graph.linear import Linear as _Linear

    n_heads = resolve_n_heads(d, d_head, n_heads, require_divisible=False)
    canon = canonical_ids(output_node)
    linear_support = {
        canon[n.node_id]: live_weight_row_ranges(n)
        for n in _canonical_walk(output_node)
        if isinstance(n, _Linear)
    }
    payload = {
        "topology": topology_entries(output_node),
        "d": d,
        "d_head": d_head,
        "d_hidden": d_hidden,
        "flex_routing": flex_routing,
        "cancel_slack": cancel_slack,
        "policy": asdict(policy) if policy is not None else None,
        "compiler_code": compiler_code_fingerprint(),
        "linear_support": sorted(linear_support.items()),
    }
    if reserve_residual:
        payload["reserve_residual"] = reserve_residual
    # Preserve the historical payload for the default coupled geometry.  A
    # decoupled capacity changes the feasible schedule and must key separately.
    if d % d_head != 0 or n_heads != d // d_head:
        payload["n_heads"] = n_heads
    # bias=False reserves hidden slot 0 (the constant lane), changing the
    # modeled slot capacity, hence the schedule — it MUST key the cache.
    # Added only when False (the ``reserve_residual`` compatibility pattern)
    # so every existing biased-compile entry keeps hashing byte-identically.
    if not bias:
        payload["bias"] = False
    if (held_output_source is None) != (held_output_target is None):
        raise ValueError("held output fingerprint requires both endpoints")
    if held_output_source is not None:
        try:
            payload["held_output_source"] = canon[held_output_source.node_id]
            payload["held_output_target"] = canon[
                cast("Node", held_output_target).node_id
            ]
        except KeyError as exc:
            raise ValueError(
                "held output endpoint is not reachable from output"
            ) from exc
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


def encode_cols(cols: list[int]) -> list[tuple[int, int]]:
    """Run-length encode an ORDERED column list as ``(start, length)`` runs.

    Column order is meaningful (column k holds component k of the node's
    value), so runs only merge consecutive ASCENDING indices — decoding
    reproduces the exact original order.
    """
    runs: list[tuple[int, int]] = []
    for c in cols:
        if runs and c == runs[-1][0] + runs[-1][1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((c, 1))
    return runs


def decode_cols(runs: list) -> list[int]:
    """Inverse of :func:`encode_cols` (accepts JSON-decoded lists)."""
    cols: list[int] = []
    for start, length in runs:
        cols.extend(range(int(start), int(start) + int(length)))
    return cols
