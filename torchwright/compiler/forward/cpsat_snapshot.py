"""Serializable snapshot of the CP-SAT scheduling problem.

The CP-SAT model builder (:func:`cpsat_scheduler.build_cpsat_model`) reads a
narrow set of *structural* facts off the live graph — per-node op class,
width, dependency edges, effective consumers, pinned / keep-forever status,
critical-path length, and the eager warm-start hint — and nothing else.  This
module freezes exactly that set into a :class:`SchedulingProblem`, a pure-data
structure from which :func:`cpsat_scheduler.graph_model_from_problem` rebuilds
a stand-in ``GraphModel`` the builder consumes in place of a live graph.

Why this exists.  At DOOM scale the graph *build + lower* that produces the
live graph costs ~10 minutes; the CP-SAT solve that follows is where the
scheduling experiments (parameter sweeps, feasibility probes, redundant
capacity rows) actually iterate.  A snapshot lets those experiments skip the
build entirely: load the frozen problem, rebuild the identical CP-SAT model,
re-solve.  It is measurement/testing tooling only — production compiles always
build fresh (compile behavior must stay uniform), and a stale snapshot must
fail loudly, never silently measure the wrong graph.

The single-builder contract.  ``build_cpsat_model`` is
``build_cpsat_model_from_gm(build_graph_model(output_node), ...)``; the
snapshot path is ``build_cpsat_model_from_gm(graph_model_from_problem(problem),
...)``.  Both run the *same* model-construction code over a ``GraphModel``.
The stand-in nodes are real graph-class instances (``object.__new__`` +
attribute injection), so every ``isinstance`` check and every realization /
demand helper in the builder behaves byte-identically — the proto built from a
live graph and the proto built from that graph's round-tripped snapshot are
identical (the equivalence test in ``tests/compile/forward/test_cpsat_snapshot.py``
pins this on the six example graphs).

Node ids.  A :class:`SchedulingProblem` holds whatever node-id space it was
captured in.  In-process capture keeps the live ``node_id`` values (so a
snapshot-built proto is byte-identical to a live-built one — node ids appear in
CP-SAT variable names).  For on-disk persistence,
:meth:`SchedulingProblem.canonicalized` relabels every id to the topology-only
*canonical id* (:mod:`torchwright.compiler.graph_identity`), the same
cross-process-stable numbering the schedule cache and the ONNX debug sidecar
use — so a fixture reloaded in a fresh container reproduces the same problem.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

from torchwright.compiler.graph_identity import canonical_ids, graph_fingerprint

# Bumped whenever the on-disk schema or the set of facts the builder reads
# changes.  The loader refuses a snapshot whose ``format_version`` differs from
# this, so a schema drift is a loud miss, never a silent wrong-graph measure.
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Per-node record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeRecord:
    """Every structural fact the CP-SAT builder reads about one node.

    ``kind`` is the op-class tag (``type(node).__name__``): ``"Add"``,
    ``"Attn"``, ``"FFN"``, ``"Linear"``, ``"LiteralValue"``,
    ``"Concatenate"``, ``"InputNode"``, ``"Embedding"``.  On reconstruction it
    selects the real graph class, so the builder's ``isinstance`` checks and
    the realization helpers (``candidate_classes`` / ``is_flex`` /
    ``is_conditional``) resolve exactly as on the live node.

    The demand fields are what the builder reads off the node:

    - ``d_output`` = ``len(node)`` = ``node.d_output`` (``Node.__len__``
      returns ``d_output``) — residual/cancel/free-add demand, MLP-bypass
      slots (``2·d_output``), the flex-Linear objective term, and — via the
      first input's ``d_output`` — attention-transport head demand.
    - ``d_v`` = ``node.d_v`` (``Attn`` only) — attention-head demand.
    - ``n_lanes`` = ``node.n_lanes`` (``FFN`` only) — MLP hidden-slot demand.

    ``input_ids`` is the ordered raw input list (the ``Add`` free/compute
    classification walks it; reconstruction wires ``node.inputs`` from it);
    ``critical_path_len`` seeds the solver's decision strategy.
    """

    node_id: int
    kind: str
    d_output: int
    input_ids: Tuple[int, ...]
    d_v: Optional[int] = None
    n_lanes: Optional[int] = None
    critical_path_len: int = 0
    name: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "d_output": self.d_output,
            "input_ids": list(self.input_ids),
            "d_v": self.d_v,
            "n_lanes": self.n_lanes,
            "critical_path_len": self.critical_path_len,
            "name": self.name,
        }

    @classmethod
    def from_json(cls, d: dict) -> "NodeRecord":
        return cls(
            node_id=int(d["node_id"]),
            kind=d["kind"],
            d_output=int(d["d_output"]),
            input_ids=tuple(int(x) for x in d["input_ids"]),
            d_v=None if d["d_v"] is None else int(d["d_v"]),
            n_lanes=None if d["n_lanes"] is None else int(d["n_lanes"]),
            critical_path_len=int(d["critical_path_len"]),
            name=d["name"],
        )

    def remap(self, mapping: Dict[int, int]) -> "NodeRecord":
        """Return a copy with every node id relabeled through ``mapping``."""
        return NodeRecord(
            node_id=mapping[self.node_id],
            kind=self.kind,
            d_output=self.d_output,
            input_ids=tuple(mapping[i] for i in self.input_ids if i in mapping),
            d_v=self.d_v,
            n_lanes=self.n_lanes,
            critical_path_len=self.critical_path_len,
            name=self.name,
        )


# ---------------------------------------------------------------------------
# Identity + hint metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotIdentity:
    """Identity + provenance stamped on a persisted snapshot.

    ``fingerprint`` is :func:`graph_fingerprint` over the captured graph +
    geometry — the same key the production schedule cache uses.  A consumer
    rebuilds the graph, recomputes the fingerprint, and compares: a mismatch
    means the fixture is stale for the running graph and must be regenerated,
    not measured.  ``critical_path_layers`` is the lowered dependency floor
    (the second half of the graph-identity gate every measurement row cites).
    """

    fingerprint: str
    critical_path_layers: int
    torchwright_commit: Optional[str]
    geometry: Dict[str, object]

    def to_json(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "critical_path_layers": self.critical_path_layers,
            "torchwright_commit": self.torchwright_commit,
            "geometry": self.geometry,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SnapshotIdentity":
        return cls(
            fingerprint=d["fingerprint"],
            critical_path_layers=int(d["critical_path_layers"]),
            torchwright_commit=d.get("torchwright_commit"),
            geometry=d.get("geometry", {}),
        )


@dataclass(frozen=True)
class FrozenHint:
    """The eager warm-start hint, frozen into the snapshot.

    A shared hint improves A/B matching across re-solves (every arm starts
    from the identical incumbent), and it is exactly what a consumer passes to
    ``solve_schedule`` as ``hint_layers`` / ``hint_routing`` / ``hint_cancel`` /
    ``hint_cancel_mech``.
    """

    layers: Dict[int, int]
    routing: Dict[int, str]
    cancel: Dict[int, int]
    cancel_mech: Dict[int, str]
    n_layers: int

    def to_json(self) -> dict:
        return {
            "layers": {str(k): v for k, v in self.layers.items()},
            "routing": {str(k): v for k, v in self.routing.items()},
            "cancel": {str(k): v for k, v in self.cancel.items()},
            "cancel_mech": {str(k): v for k, v in self.cancel_mech.items()},
            "n_layers": self.n_layers,
        }

    @classmethod
    def from_json(cls, d: dict) -> "FrozenHint":
        return cls(
            layers={int(k): int(v) for k, v in d["layers"].items()},
            routing={int(k): v for k, v in d["routing"].items()},
            cancel={int(k): int(v) for k, v in d["cancel"].items()},
            cancel_mech={int(k): v for k, v in d["cancel_mech"].items()},
            n_layers=int(d["n_layers"]),
        )

    def remap(self, mapping: Dict[int, int]) -> "FrozenHint":
        def rm(t):
            return {mapping[k]: v for k, v in t.items() if k in mapping}

        return FrozenHint(
            layers=rm(self.layers),
            routing=rm(self.routing),
            cancel=rm(self.cancel),
            cancel_mech=rm(self.cancel_mech),
            n_layers=self.n_layers,
        )


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulingProblem:
    """Everything the CP-SAT model builder consumes, as pure data.

    Structure fields mirror :class:`cpsat_scheduler.GraphModel`, keyed by node
    id instead of live ``Node`` objects; ``nodes`` maps every referenced id to
    its :class:`NodeRecord`.  ``schedulable_ids`` / ``input_ids`` / ``edges``
    and each ``consumers`` list preserve the *exact* order the live builder
    iterated them, so the rebuilt proto is byte-identical (constraint /
    variable order is order-sensitive).

    ``id_space`` records whether ids are the live process-cumulative
    ``node_id`` (``"live"``) or topology-canonical (``"canonical"``); the
    latter is what persists to disk.
    """

    nodes: Dict[int, NodeRecord]
    schedulable_ids: Tuple[int, ...]
    input_ids: Tuple[int, ...]
    output_id: int
    pinned_ids: FrozenSet[int]
    edges: Tuple[Tuple[int, int], ...]
    consumers: Dict[int, Tuple[int, ...]]
    id_space: str = "live"
    identity: Optional[SnapshotIdentity] = None
    hint: Optional[FrozenHint] = None

    # -- id-space transforms ----------------------------------------------

    def canonicalized(self, output_node) -> "SchedulingProblem":
        """Return a copy with every id relabeled to canonical (topology) ids.

        ``output_node`` is the live graph the problem was captured from; its
        canonical numbering is what makes the persisted fixture reproduce the
        same problem in any process.
        """
        if self.id_space == "canonical":
            return self
        mapping = canonical_ids(output_node)
        missing = [nid for nid in self.nodes if nid not in mapping]
        if missing:
            raise ValueError(
                f"{len(missing)} snapshot node(s) are unreachable from the "
                f"output via inputs and cannot be canonically keyed "
                f"(e.g. id {missing[0]}); refusing to persist a partial problem."
            )
        return self._remap(mapping, "canonical")

    def _remap(self, mapping: Dict[int, int], id_space: str) -> "SchedulingProblem":
        return SchedulingProblem(
            nodes={mapping[k]: r.remap(mapping) for k, r in self.nodes.items()},
            schedulable_ids=tuple(mapping[i] for i in self.schedulable_ids),
            input_ids=tuple(mapping[i] for i in self.input_ids),
            output_id=mapping[self.output_id],
            pinned_ids=frozenset(mapping[i] for i in self.pinned_ids),
            edges=tuple((mapping[u], mapping[v]) for u, v in self.edges),
            consumers={
                mapping[k]: tuple(mapping[c] for c in v)
                for k, v in self.consumers.items()
            },
            id_space=id_space,
            identity=self.identity,
            hint=None if self.hint is None else self.hint.remap(mapping),
        )

    # -- serialization -----------------------------------------------------

    def to_json(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "id_space": self.id_space,
            "nodes": [r.to_json() for r in self.nodes.values()],
            "schedulable_ids": list(self.schedulable_ids),
            "input_ids": list(self.input_ids),
            "output_id": self.output_id,
            "pinned_ids": sorted(self.pinned_ids),
            "edges": [[u, v] for u, v in self.edges],
            "consumers": {str(k): list(v) for k, v in self.consumers.items()},
            "identity": None if self.identity is None else self.identity.to_json(),
            "hint": None if self.hint is None else self.hint.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "SchedulingProblem":
        version = d.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"snapshot format_version {version!r} != running "
                f"FORMAT_VERSION {FORMAT_VERSION} — regenerate the fixture "
                f"against the current code (stale snapshots must miss loudly)."
            )
        nodes = {}
        for nd in d["nodes"]:
            rec = NodeRecord.from_json(nd)
            nodes[rec.node_id] = rec
        return cls(
            nodes=nodes,
            schedulable_ids=tuple(int(x) for x in d["schedulable_ids"]),
            input_ids=tuple(int(x) for x in d["input_ids"]),
            output_id=int(d["output_id"]),
            pinned_ids=frozenset(int(x) for x in d["pinned_ids"]),
            edges=tuple((int(u), int(v)) for u, v in d["edges"]),
            consumers={
                int(k): tuple(int(c) for c in v) for k, v in d["consumers"].items()
            },
            id_space=d["id_space"],
            identity=(
                None
                if d.get("identity") is None
                else SnapshotIdentity.from_json(d["identity"])
            ),
            hint=None if d.get("hint") is None else FrozenHint.from_json(d["hint"]),
        )

    def dumps(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True)

    @classmethod
    def loads(cls, text: str) -> "SchedulingProblem":
        return cls.from_json(json.loads(text))

    def save(self, path) -> Path:
        """Persist to ``path`` (JSON).  Requires canonical ids + identity so a
        fixture is self-describing and reproducible; a live-id or
        identity-less problem is a program error to persist."""
        if self.id_space != "canonical":
            raise ValueError(
                "refusing to save a live-id snapshot; call .canonicalized("
                "output_node) first so the fixture reproduces across processes."
            )
        if self.identity is None:
            raise ValueError(
                "refusing to save a snapshot without identity metadata; attach "
                "it via with_identity(...) so stale fixtures fail loudly."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.dumps(), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(
        cls,
        path,
        *,
        expected_fingerprint: Optional[str] = None,
    ) -> "SchedulingProblem":
        """Load a fixture, refusing on a stale/mismatched identity.

        ``expected_fingerprint`` (from rebuilding the current graph) gates the
        graph-identity check: a mismatch raises rather than silently measuring
        a different graph — this repo has been burned twice by wrong-graph
        measurements.  ``FORMAT_VERSION`` mismatch already raises in
        :meth:`from_json`.
        """
        problem = cls.loads(Path(path).read_text(encoding="utf-8"))
        if expected_fingerprint is not None:
            got = problem.identity.fingerprint if problem.identity else None
            if got != expected_fingerprint:
                raise ValueError(
                    f"snapshot fingerprint {got!r} != expected "
                    f"{expected_fingerprint!r}: the fixture does not match the "
                    f"current graph+geometry. Regenerate it — a stale fixture "
                    f"must not be measured."
                )
        return problem

    def with_identity(
        self,
        *,
        output_node,
        d: int,
        d_head: int,
        d_hidden: int,
        flex_routing: bool,
        cancel_slack: Optional[int],
        policy,
        critical_path_layers: int,
        reserve_residual: int = 0,
        reserve_heads: int = 0,
        max_layers: int = 60,
        bias: bool = True,
        fingerprint_d_hidden: Optional[int] = None,
    ) -> "SchedulingProblem":
        """Attach identity metadata (fingerprint + critical path + geometry).

        Kept separate from structural capture so the production build path (via
        :func:`snapshot_from_graph_model`) pays no fingerprint-hashing cost;
        only the fixture generator calls this.

        ``d_hidden`` is the value the model is *built* with (stored in
        ``geometry`` and passed to :func:`build_model_from_snapshot`).
        ``fingerprint_d_hidden`` overrides only the fingerprint's ``d_hidden``
        so the stored identity matches the production schedule-cache key —
        ``forward_compile`` keys the cache on the raw ``d_hidden`` but solves
        with ``d_hidden - 1`` when ``bias=False`` (the reserved constant lane).
        """
        from dataclasses import asdict as _asdict

        fingerprint = graph_fingerprint(
            output_node,
            d=d,
            d_head=d_head,
            d_hidden=d_hidden if fingerprint_d_hidden is None else fingerprint_d_hidden,
            flex_routing=flex_routing,
            cancel_slack=cancel_slack,
            policy=policy,
            reserve_residual=reserve_residual,
            bias=bias,
        )
        geometry = {
            "d": d,
            "d_head": d_head,
            "d_hidden": d_hidden,
            "flex_routing": flex_routing,
            "cancel_slack": cancel_slack,
            "reserve_residual": reserve_residual,
            "reserve_heads": reserve_heads,
            "max_layers": max_layers,
            "bias": bias,
            "policy": _asdict(policy) if policy is not None else None,
        }
        identity = SnapshotIdentity(
            fingerprint=fingerprint,
            critical_path_layers=critical_path_layers,
            torchwright_commit=_git_commit(),
            geometry=geometry,
        )
        return self._replace(identity=identity)

    def with_hint(self, hint: Optional[FrozenHint]) -> "SchedulingProblem":
        return self._replace(hint=hint)

    def _replace(self, **changes) -> "SchedulingProblem":
        from dataclasses import replace as _replace

        return _replace(self, **changes)


# ---------------------------------------------------------------------------
# Capture from a live GraphModel
# ---------------------------------------------------------------------------


def _record_for(node, graph) -> NodeRecord:
    d_v = getattr(node, "d_v", None)
    n_lanes = getattr(node, "n_lanes", None)
    inputs = getattr(node, "inputs", None) or []
    return NodeRecord(
        node_id=node.node_id,
        kind=type(node).__name__,
        d_output=len(node),
        input_ids=tuple(inp.node_id for inp in inputs),
        d_v=None if d_v is None else int(d_v),
        n_lanes=None if n_lanes is None else int(n_lanes),
        critical_path_len=graph.get_critical_path_length(node),
        name=getattr(node, "name", None),
    )


def snapshot_from_graph_model(gm) -> SchedulingProblem:
    """Capture the structural problem from a live ``GraphModel`` (cheap).

    Reads only what the CP-SAT builder reads; adds no fingerprint hashing
    (attach that with :meth:`SchedulingProblem.with_identity` when building a
    fixture).  Iteration orders (``schedulable``, ``edges``, per-node consumer
    sets) are frozen exactly as the live builder sees them, so the rebuilt
    proto matches byte-for-byte in the capturing process.
    """
    graph = gm.graph
    nodes: Dict[int, NodeRecord] = {}
    for node in graph.get_all_nodes():
        nodes[node.node_id] = _record_for(node, graph)

    consumers: Dict[int, Tuple[int, ...]] = {}
    for node, cons in gm.consumers_eff.items():
        consumers[node.node_id] = tuple(c.node_id for c in cons)

    return SchedulingProblem(
        nodes=nodes,
        schedulable_ids=tuple(n.node_id for n in gm.schedulable),
        input_ids=tuple(n.node_id for n in gm.input_nodes),
        output_id=gm.output_node.node_id,
        pinned_ids=frozenset(n.node_id for n in gm.pinned_nodes),
        edges=tuple((u.node_id, v.node_id) for u, v in gm.edges),
        consumers=consumers,
        id_space="live",
    )


def _git_commit() -> Optional[str]:
    """Best-effort torchwright commit for provenance (None if unavailable)."""
    try:
        here = Path(__file__).resolve()
        out = subprocess.run(
            ["git", "-C", str(here.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        return None
    return None
