"""CP-SAT scheduler for `forward_compile`.

Produces an optimal placement of every graph node into transformer
layers, an optimal cancellation timing for every node's residual
columns, and (under `flex_routing=True`) an optimal attention-versus-
MLP routing for every standalone `Linear`.

See `docs/cpsat_scheduler.md` for the architecture spec — this module
is its implementation.

Public API:

- `Costs` — objective weights `(alpha, beta, gamma)`.
- `ScheduleAssignment` — solver output contract: per-node layer,
  cancel layer, and routing.
- `DiagnosticHint` — one immutable, explicitly non-production container for
  partial/invalid warm starts used by tests and measurement tooling.
- `SolveStats` — solver metadata (status, objective, LB, walltime,
  log) for diagnostics.
- `solve_schedule(...)` — build and solve, return
  `(assignment, stats)`.  ``assignment is None`` when the solver
  finds no feasible solution within the budget; ``stats.is_optimal``
  distinguishes proven-optimal from feasible-only.  Raises only on
  graph-precondition violations (see ``docs/cpsat_scheduler.md`` §3).

Callers (``forward_compile``) decide how to handle non-optimal /
no-solution outcomes — the forward compiler falls back to the
heuristic schedule.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from ortools.sat.python import cp_model

from torchwright.compiler.forward.add_placement import derive_add_placement
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.realization import (
    CLASS_SUBLAYER,
    candidate_classes,
    has_flex_choice,
    linear_attn_heads,
    static_flex_class,
)
from torchwright.compiler.utils import resolve_n_heads
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.graph import (
    Add,
    Attn,
    Concatenate,
    Embedding,
    InputNode,
    Linear,
    Node,
)
from torchwright.graph.misc import LiteralValue
from torchwright.graph.ffn import FFN
from torchwright.compiler.forward.cpsat_snapshot import (
    SchedulingProblem,
    snapshot_from_graph_model,
)

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Costs:
    """Objective weights for the CP-SAT solver.

    Total objective = `alpha * n_layers + beta * total_attn_heads +
    gamma * total_mlp_bypass_slots`.

    Defaults: `alpha=1, beta=0, gamma=0` — pure layer minimization.

    `beta` (long-sequence regime). Per-token attention compute scales
    as `O(L · d_head)` per head for sequence length `L`; per-layer
    compute (the full `d × d_hidden` MLP matmul plus the `4 · d²`
    attention QKVO matmuls) is independent of `L`. For long sequences,
    set `beta` above zero to push routing toward MLP. Rule of thumb:
    `beta ≈ L` makes one attention head equivalent to one extra layer.

    `gamma` (MLP bypass slot pressure). Less commonly useful — the
    per-layer MLP matmul costs the full `d × d_hidden` regardless of
    how many slots are used. `gamma=0` is the normal case.

    `earliness` (plateau gradient). When above zero, adds a strictly
    LEXICOGRAPHIC secondary term `earliness * sum(layer_var)` — the
    alpha/beta/gamma block is scaled so no amount of earliness can
    trade against it.  A pure-depth objective is a staircase: vast
    plateaus of schedules tie at the same layer count and local-search
    workers get zero reward for compacting moves until one happens to
    drop the max layer.  Earliness rewards every move that shifts any
    node earlier, pulling incumbents toward states one move away from
    a layer-count drop.  `earliness=0` (default) leaves the objective
    byte-identical to the historical three-term form.
    (Measured 2026-06-09 on the d8192 DOOM graph: earliness SLOWED
    layer descent — the gradient mostly rewards irrelevant early-region
    shuffling.  Kept for comparison; prefer `waste`.)

    `waste` (residual-occupancy gradient). Lexicographic secondary like
    `earliness`, but aimed at the binding resource instead of raw
    position: minimizes total residual occupancy area
    `sum(len(n) * (cancel_layer[n] - layer[n]))` — the column-layers a
    value sits in the stream — over every node with a real cancel
    window, plus `len(n) * cancel` for freeable inputs (born at layer
    0).  Keep-forever nodes are excluded: their lifetime is unavoidable
    and the term would reward scheduling them LATER.  Shrinking
    producer-to-last-reader spans frees width at the pinch layers that
    gate layer-count drops.
    """

    alpha: int = 1
    beta: int = 0
    gamma: int = 0
    earliness: int = 0
    waste: int = 0


def lexicographic_objective_scale(max_secondary: int) -> int:
    """Return the shared multiplier that makes the primary block dominant."""
    return int(max_secondary) + 1 if max_secondary else 1


def evaluate_objective_components(
    costs: Costs,
    *,
    n_layers: int,
    total_attn_heads: int,
    total_mlp_bypass_slots: int,
    earliness_sum: int,
    waste_sum: int,
    objective_scale: int,
) -> int:
    """Evaluate concrete counts with the same objective algebra as CP-SAT."""
    primary, secondary = objective_blocks(
        costs,
        n_layers=n_layers,
        total_attn_heads=total_attn_heads,
        total_mlp_bypass_slots=total_mlp_bypass_slots,
        earliness_sum=earliness_sum,
        waste_sum=waste_sum,
    )
    return objective_scale * primary + secondary


def objective_blocks(
    costs: Costs,
    *,
    n_layers: int,
    total_attn_heads: int,
    total_mlp_bypass_slots: int,
    earliness_sum: int,
    waste_sum: int,
) -> tuple[int, int]:
    """Return the primary and secondary blocks before lexicographic scaling."""
    primary = (
        costs.alpha * n_layers
        + costs.beta * total_attn_heads
        + costs.gamma * total_mlp_bypass_slots
    )
    secondary = costs.earliness * earliness_sum + costs.waste * waste_sum
    return int(primary), int(secondary)


@dataclass(frozen=True)
class ScheduleAssignment:
    """Per-node placement, cancellation, and routing decisions.

    Returned by `solve_schedule`. Consumed by `DirectedLayerScheduler`
    to replay the schedule through the existing per-layer code path.

    `node_to_layer[n]` is the transformer layer where node `n`
    executes. `node_to_cancel_layer[n]` is the layer where `n`'s
    residual columns are reclaimed (set to `n_layers` for nodes that
    stay alive forever — inputs, output, output-cone leaves).
    `node_to_routing[n]` is `"attn"` or `"mlp"` — which sublayer of
    `node_to_layer[n]` runs the op.

    Every schedulable node — every non-`Concatenate`, non-input node
    in the ancestor cone of `output_node` — appears in all three
    dicts.  Freeable input nodes (every input except `pos_encoding`)
    additionally appear in `node_to_cancel_layer` only: they are
    pre-computed at layer 0 (no layer/routing decision) but get a
    cancel layer so the replay reclaims their columns once consumed.
    """

    node_to_layer: Mapping[int, int]
    node_to_cancel_layer: Mapping[int, int]
    node_to_routing: Mapping[int, str]
    n_layers: int
    # Which sublayer each node's cancel runs in: "attn" (a batched attention
    # cancel head) or "mlp" (a ``cancel_bypass`` MLP op).  Keyed by the same
    # schedulable node ids as ``node_to_cancel_layer``; freeable inputs are
    # always attention-cancelled and are omitted (the replay defaults absent
    # nodes to "attn").  ``DirectedLayerScheduler`` routes each directed cancel
    # to its assigned mechanism.
    node_to_cancel_mech: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``frozen=True`` does not freeze nested dictionaries. Defensive,
        # canonical copies keep validation, cache identity, and replay stable
        # even if a caller later mutates the mappings used to construct us.
        for name in (
            "node_to_layer",
            "node_to_cancel_layer",
            "node_to_routing",
            "node_to_cancel_mech",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self, name, MappingProxyType(dict(sorted(value.items())))
            )

    @classmethod
    def from_heuristic_trace(
        cls,
        output_node: Node,
        *,
        node_to_layer: Dict[int, int],
        node_to_routing: Dict[int, str],
        observed_cancel_layer: Dict[int, int],
        observed_cancel_mech: Dict[int, str],
        n_layers: int,
        observed_add_placement: Optional[Dict[int, Tuple[bool, Optional[int]]]] = None,
        held_source_id: Optional[int] = None,
        held_target_id: Optional[int] = None,
    ) -> "ScheduleAssignment":
        """Complete and validate a schedule-only heuristic trace.

        Before canonicalizing target metadata, each Add's expected placement
        is derived from the completed layer/route maps
        (``add_placement.derive_add_placement``) and compared with the
        walk's ``observed_add_placement``; a mismatch raises rather than
        silently replacing the physical observation with the derived value.
        Every reused target then gets the canonical virtual cancel layer
        ``layer[Add] + 1`` — bookkeeping only, no operation is emitted;
        ownership ended through ``reassign`` — with an attention mechanism
        recorded for schedulable targets (canonical, never run) and no
        mechanism entry for graph-input targets.
        """
        gm = build_graph_model(output_node)
        sched_ids = {node.node_id for node in gm.schedulable}
        input_ids = {node.node_id for node in gm.input_nodes}
        layer_map = {
            nid: layer for nid, layer in node_to_layer.items() if nid in sched_ids
        }
        if set(layer_map) != sched_ids:
            missing = sorted(sched_ids - set(layer_map))
            extra = sorted(set(layer_map) - sched_ids)
            raise ValueError(
                f"heuristic trace layer coverage mismatch: missing={missing}, extra={extra}"
            )
        if set(node_to_routing) != sched_ids:
            missing = sorted(sched_ids - set(node_to_routing))
            extra = sorted(set(node_to_routing) - sched_ids)
            raise ValueError(
                f"heuristic trace routing coverage mismatch: missing={missing}, extra={extra}"
            )

        # Physical-versus-derived Add placement tripwire + reused-target
        # canonicalization (docs/plan_additional_mlp_routing.md).
        observed_add_placement = observed_add_placement or {}
        canonical_cancel: Dict[int, int] = {}
        canonical_mech: Dict[int, str] = {}
        for node in gm.schedulable:
            if not isinstance(node, Add):
                continue
            derived = derive_add_placement(
                node,
                effective_consumers=lambda n: gm.consumers_eff.get(n, ()),
                node_to_layer=layer_map,
                node_to_routing=node_to_routing,
                held_source_id=held_source_id,
                held_target_id=held_target_id,
            )
            observed = observed_add_placement.get(node.node_id)
            if observed is None:
                raise ValueError(
                    f"heuristic trace has no observed placement for Add "
                    f"{node.node_id} ({node!r}); every scheduled Add must "
                    f"record (is_reused, reuse_input_index) from its emitted "
                    f"operation"
                )
            observed_reused, observed_index = observed
            if (
                derived.is_free != observed_reused
                or derived.reuse_input_index != observed_index
            ):

                def _consumer_summary(occ: Node) -> str:
                    parts = [
                        f"{c!r}@L{layer_map.get(c.node_id)}"
                        f"/{node_to_routing.get(c.node_id)}"
                        for c in sorted(
                            gm.consumers_eff.get(occ, ()),
                            key=lambda c: c.node_id,
                        )
                        if c.node_id != node.node_id
                    ]
                    return ", ".join(parts) or "none"

                a0, a1 = node.inputs
                raise ValueError(
                    f"heuristic Add placement disagrees with the "
                    f"assignment-level derivation for {node!r} (route "
                    f"{node_to_routing.get(node.node_id)!r}, layer "
                    f"{layer_map.get(node.node_id)}): observed "
                    f"(reused={observed_reused}, occurrence={observed_index}) "
                    f"but derived (reusable_0={derived.reusable_0}, "
                    f"reusable_1={derived.reusable_1}, "
                    f"occurrence={derived.reuse_input_index}).  Other "
                    f"consumers of occurrence 0 {a0!r}: "
                    f"{_consumer_summary(a0)}; of occurrence 1 {a1!r}: "
                    f"{_consumer_summary(a1)}."
                )
            if derived.reuse_input_index is not None:
                target = node.inputs[derived.reuse_input_index]
                canonical_cancel[target.node_id] = int(layer_map[node.node_id]) + 1
                if target.node_id in sched_ids:
                    # Canonical bookkeeping only — the target selector gates
                    # the physical cancel absent; graph-input targets carry
                    # no mechanism entry (the assignment contract).
                    canonical_mech[target.node_id] = ATTN

        cancel = {nid: int(n_layers) for nid in sched_ids | input_ids}
        unknown_cancel = set(observed_cancel_layer) - (sched_ids | input_ids)
        if unknown_cancel:
            raise ValueError(
                f"heuristic trace has cancellation for unknown nodes: {sorted(unknown_cancel)}"
            )
        cancel.update({int(k): int(v) for k, v in observed_cancel_layer.items()})
        # A reassigned target never went through free(), so its observed
        # entry is absent; the canonical virtual layer ends its occupancy
        # exactly where the Add's shifted ownership begins.
        cancel.update(canonical_cancel)
        mech = {
            nid: mech_
            for nid, mech_ in observed_cancel_mech.items()
            if nid in sched_ids
        }
        mech.update(canonical_mech)
        assignment = cls(
            node_to_layer=layer_map,
            node_to_cancel_layer=cancel,
            node_to_routing=dict(node_to_routing),
            n_layers=int(n_layers),
            node_to_cancel_mech=mech,
        )
        assignment.validate(output_node)
        return assignment

    def validate(self, output_node: Node) -> None:
        gm = build_graph_model(output_node)
        sched_ids = {node.node_id for node in gm.schedulable}
        input_ids = {node.node_id for node in gm.input_nodes}
        if set(self.node_to_layer) != sched_ids:
            raise ValueError(
                "assignment node_to_layer does not cover schedulable nodes"
            )
        if set(self.node_to_routing) != sched_ids:
            raise ValueError(
                "assignment node_to_routing does not cover schedulable nodes"
            )
        if not (sched_ids | input_ids) <= set(self.node_to_cancel_layer):
            raise ValueError("assignment cancellation map is incomplete")
        for nid, layer in self.node_to_layer.items():
            if not 0 <= int(layer) < self.n_layers:
                raise ValueError(f"node {nid} has invalid layer {layer}")
            if self.node_to_routing[nid] not in (ATTN, MLP):
                raise ValueError(
                    f"node {nid} has invalid routing {self.node_to_routing[nid]!r}"
                )
            cancel = self.node_to_cancel_layer[nid]
            if not int(layer) <= int(cancel) <= self.n_layers:
                raise ValueError(
                    f"node {nid} cancel layer {cancel} precedes birth {layer}"
                )
        if set(self.node_to_cancel_mech) - sched_ids:
            raise ValueError(
                "assignment has cancellation mechanisms for non-schedulable nodes"
            )
        if any(mech not in (ATTN, MLP) for mech in self.node_to_cancel_mech.values()):
            raise ValueError("assignment has an invalid cancellation mechanism")

    def canonical_key(self) -> tuple:
        """Stable tie-break independent of mapping insertion order."""
        return (
            self.n_layers,
            tuple(self.node_to_layer.items()),
            tuple(self.node_to_routing.items()),
            tuple(self.node_to_cancel_layer.items()),
            tuple(self.node_to_cancel_mech.items()),
        )


@dataclass(frozen=True)
class DiagnosticHint:
    """Partial warm start for tests, snapshots, and solver experiments.

    Production compilation passes a complete :class:`ScheduleAssignment` as
    ``incumbent``.  This type deliberately isolates diagnostic cases that need
    an incomplete or intentionally invalid hint without exposing four parallel
    mappings on the production solver API.
    """

    layers: Mapping[int, int] = field(default_factory=dict)
    routing: Mapping[int, str] = field(default_factory=dict)
    cancel: Mapping[int, int] = field(default_factory=dict)
    cancel_mech: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("layers", "routing", "cancel", "cancel_mech"):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(sorted(value.items()))),
            )

    @classmethod
    def from_assignment(cls, assignment: ScheduleAssignment) -> "DiagnosticHint":
        return cls(
            layers=assignment.node_to_layer,
            routing=assignment.node_to_routing,
            cancel=assignment.node_to_cancel_layer,
            cancel_mech=assignment.node_to_cancel_mech,
        )

    def without_cancel_layers(self) -> "DiagnosticHint":
        return DiagnosticHint(
            layers=self.layers,
            routing=self.routing,
            cancel_mech=self.cancel_mech,
        )


def choose_dominating_assignment(
    costs: Costs,
    incumbent: Optional[ScheduleAssignment],
    candidate: Optional[ScheduleAssignment],
) -> Optional[ScheduleAssignment]:
    """Select a candidate only when its comparable objective dominates.

    Pure-depth is the production default and is fully evaluable from an
    assignment. Resource-weighted objectives also depend on concrete replay
    choices (notably free-Add head cost), so CP-SAT's objective governs those
    until ReplayPlan exposes the physical counts at this boundary.
    """
    if incumbent is None:
        return candidate
    if candidate is None:
        return incumbent
    pure_depth = (
        costs.alpha > 0
        and costs.beta == costs.gamma == costs.earliness == costs.waste == 0
    )
    if not pure_depth:
        return candidate
    if candidate.n_layers < incumbent.n_layers:
        return candidate
    if candidate.n_layers > incumbent.n_layers:
        return incumbent
    return min((incumbent, candidate), key=lambda item: item.canonical_key())


@dataclass(frozen=True)
class SolveStats:
    """Solver metadata for diagnostics.

    Returned alongside an optional `ScheduleAssignment` from
    `solve_schedule`. Useful for the probe script and for logging the
    solver gap when accepting suboptimal schedules.
    """

    status_name: str
    objective_value: int  # -1 if no feasible solution
    best_objective_bound: float  # tight LB the solver proved
    wall_time_s: float
    solver_log: str
    total_attn_heads: int  # -1 if no feasible solution
    total_mlp_bypass_slots: int  # -1 if no feasible solution
    is_optimal: bool
    # Lexicographic multiplier on the primary objective block (1 when Costs
    # has no secondary terms).  objective_value // objective_scale recovers
    # the primary value; the raw solver log's best/bound lines are scaled.
    objective_scale: int = 1
    # Number of non-keep-forever nodes (schedulable + freeable inputs) whose
    # returned cancel layer sits at the virtual horizon `max_layers` — i.e.
    # the nodes the machine leaves allocated forever (`parked`).  This is the
    # `2^k` D-1 dangling-boolean multiplicity the sym1 canonicalization removes;
    # logged post-hoc so a null measurement result can be read as "no effect"
    # vs "no parked nodes to deduplicate."  -1 when no feasible solution.
    parked_count: int = -1


@dataclass(frozen=True)
class SchedulingProvenance:
    """Orthogonal origin/delivery metadata for a selected assignment."""

    origin: str  # "heuristic" or "solver"
    delivery: str  # "fresh" or "cache"
    selected_is_optimal: bool = False
    selected_objective: Optional[int] = None
    selected_objective_blocks: Optional[Tuple[int, int]] = None
    solver_attempt: Optional[SolveStats] = None

    @property
    def is_optimal(self) -> bool:
        """Compatibility alias for the selected schedule's certification."""
        return self.selected_is_optimal


@dataclass(frozen=True)
class ScheduleResult:
    assignment: ScheduleAssignment
    provenance: SchedulingProvenance


# Routing constants used both by this module and by the probe script.
ATTN = "attn"
MLP = "mlp"


# Names of the toggleable constraint families in `build_cpsat_model`.  Used
# only by the diagnostic path (`_disabled_families`); the production solve
# never disables any of them.  Kept as a module constant so the diagnostic
# script and any regression test validate names against one source of truth.
CONSTRAINT_FAMILIES = frozenset(
    {
        "dependency",  # edge u->v layer ordering
        "cancel_consumer_lb",  # cancel_layer >= consumer_layer + 1
        "cancel_slack",  # cancel_layer <= last_consumer + 1 + K window
        "attn_cumulative",  # per-layer attention-head + cancel + dirty capacity
        "mlp_cumulative",  # per-layer MLP hidden-slot capacity
        "residual_cumulative",  # residual-stream column capacity
        # DIAGNOSTIC-ONLY relaxation (never disabled by the production solve, and
        # a schedule solved with it disabled must NEVER be compiled/replayed):
        # reverts the MLP-cancel residual occupancy from the sound
        # `[layer, cancel + 1)` to the unsound `[layer, cancel)`.  The resulting
        # layer count is a valid LOWER BOUND on the sound optimum, so
        # sound-minus-relaxed measures how much the `[layer, cancel + 1)`
        # conservatism (known optimality gap #1 — the forgone same-layer MLP->MLP
        # column handoff) actually costs on a given graph.  See the derisk doc's
        # 2026-07-08 second correction.
        "mlp_cancel_occupancy",
        # DIAGNOSTIC-ONLY relaxation (same never-replay rule): relaxes the
        # Add-consumer cancel bound `cancel >= layer[A] + is_free[A]` to
        # `cancel >= layer[A]`, dropping the live-addend gap-1 conservatism
        # (known optimality gap #2 — see docs/cpsat_scheduler.md).  The
        # resulting layer count is a valid LOWER BOUND on the sound optimum,
        # so sound-minus-relaxed measures what the live-addend hold costs.
        "add_live_addend_gap",
    }
)


@dataclass
class BuiltModel:
    """A constructed (but unsolved) CP-SAT model plus its decision variables.

    Returned by :func:`build_cpsat_model`.  ``solve_schedule`` adds the hint,
    the decision strategy, solves, and reads the variables back out; the
    diagnostic path adds hard schedule-fixing constraints and toggles
    constraint families to localize an infeasibility.
    """

    model: cp_model.CpModel
    gm: GraphModel
    layer_var: Dict[int, cp_model.IntVar]
    cancel_layer: Dict[int, cp_model.IntVar]
    is_attn: Dict[int, cp_model.IntVar]
    is_free: Dict[int, cp_model.IntVar]
    # Per non-keep-forever schedulable node: 1 when its cancel runs as a
    # ``cancel_bypass`` MLP op (cost on the MLP hidden-slot budget), 0 when it
    # runs as a batched attention cancel head (cost on the attention-head
    # budget).  Absent for keep-forever nodes (they never cancel in-horizon)
    # and freeable inputs (always attention-cancelled — decision #3).
    cancel_in_mlp: Dict[int, cp_model.IntVar]
    n_layers_var: cp_model.IntVar
    total_attn_heads: cp_model.IntVar
    total_mlp_bypass: cp_model.IntVar
    available_residual: int
    n_heads_per_layer: int
    # Cancel-layer vars for freeable input nodes (born at layer 0).  Separate
    # from `cancel_layer` (which is keyed by schedulable node) so the
    # schedulable iteration stays clean; `solve_schedule` reads these into the
    # assignment so `DirectedLayerScheduler` frees the inputs at replay.
    input_cancel_layer: Dict[int, cp_model.IntVar]
    # Optional held-bank contract.  The source is a freeable input whose
    # physically-cancelled columns remain unavailable until the target (the
    # graph output) is born.  Retained here so hint validation uses the same
    # gap-0 source semantics as the model.
    held_source_id: Optional[int] = None
    held_target_id: Optional[int] = None
    # The lexicographic multiplier applied to the primary objective block
    # when Costs has secondary terms (earliness/waste); 1 otherwise.
    # ObjectiveValue() // objective_scale recovers the primary value.
    objective_scale: int = 1
    # Tightened per-node layer domains `node_id -> (lo, hi)` when the model
    # was built with `tighten_domains=True`; None otherwise.  Retained so
    # hint validation can check layer hints against the actual variable
    # domains without re-deriving them (and without reading IntVar.Proto(),
    # which returns corrupted memory in the installed ortools build and
    # segfaults a later Solve()).
    layer_bounds: Optional[Dict[int, Tuple[int, int]]] = None
    # Schedulable nodes whose cancel var is pinned to `max_layers` (pinned
    # nodes and nodes consumed by a terminal Concatenate), and the same for
    # freeable inputs.  Retained for hint validation.
    keep_forever_ids: FrozenSet[int] = frozenset()
    input_keep_ids: FrozenSet[int] = frozenset()
    # Per-occurrence Add placement literals, keyed (add_id, occurrence):
    # `add_reusable` reifies the sublayer-order deadness predicate;
    # `add_reuse` is the deterministic selector (occurrence 0 wins).
    # Diagnostic/extraction state only — never copied into
    # ScheduleAssignment; solution extraction recomputes the selectors from
    # the extracted layers/routes (`add_placement.derive_add_placement`) and
    # asserts they match while these literals still exist.  The held target
    # has no entries (its is_free is pinned 0 without the biconditional).
    add_reusable: Dict[Tuple[int, int], cp_model.IntVar] = field(default_factory=dict)
    add_reuse: Dict[Tuple[int, int], cp_model.IntVar] = field(default_factory=dict)
    # Old value id -> OR of the selectors that reassign it (an ownership
    # handoff: no physical cancel, parked forced false).
    reused_as_target: Dict[int, cp_model.IntVar] = field(default_factory=dict)
    # Hint-aware cancel-window widening actually applied: `node_id -> delta`
    # for every node whose window was widened past the uniform
    # `last_consumer + 1 + K` (only nonzero deltas appear).  None when the
    # model was built without hints.
    cancel_window_delta: Optional[Dict[int, int]] = None
    # The cancel-window slack K the model actually posted (None when the
    # window family is disabled or `cancel_slack=None`).
    eff_cancel_slack: Optional[int] = None
    # Per-node routing as the model sees it: "attn"/"mlp" for pinned nodes,
    # "flex" when routing is a free decision variable.  Used by hint
    # validation to mirror the routing-aware cancel bounds.
    static_routing: Optional[Dict[int, str]] = None
    # True when the model was built with `_pin_cancels` (cancel layers
    # equality-pinned; no parked/window/widening families) — the production
    # default since 2026-07-10.  `_solve_built` reads it to drop the
    # cancel-LAYER hints, which the pin forces; the cancel-MECHANISM hints
    # are kept (`cancel_in_mlp` stays a free decision, and its hint bits are
    # load-bearing for warm-start completion on head-saturated graphs).
    pin_cancels: bool = False


# ---------------------------------------------------------------------------
# Static graph preprocessing
# ---------------------------------------------------------------------------


@dataclass
class GraphModel:
    """Static analysis of the graph that the CP-SAT model is built over."""

    graph: GraphAnalyzer
    schedulable: List[Node]  # nodes that need a time slot
    edges: List[Tuple[Node, Node]]  # (u, v) Concatenate-transparent
    consumers_eff: Dict[Node, Set[Node]]  # effective consumers (Concat-transparent)
    output_node: Node
    pos_encoding: object  # vestigial (always None) — position is rotary, no node
    input_nodes: List[Node]  # pre-allocated inputs (incl. LiteralValue)
    pinned_nodes: Set[Node]  # never freed


def _effective_consumers(
    graph: GraphAnalyzer, node: Node, cache: Dict[Node, Set[Node]]
) -> Set[Node]:
    """Mirror `LayerScheduler._get_effective_consumers`."""
    if node in cache:
        return cache[node]
    result: Set[Node] = set()
    for consumer in graph.get_consumers(node):
        if isinstance(consumer, Concatenate):
            downstream = _effective_consumers(graph, consumer, cache)
            if downstream:
                result |= downstream
            else:
                # Terminal Concatenate (output): keep as consumer so
                # children stay alive.
                result.add(consumer)
        else:
            result.add(consumer)
    cache[node] = result
    return result


def build_graph_model(output_node: Node, pos_encoding=None) -> GraphModel:
    """Run all the static preprocessing the CP-SAT builder needs."""
    graph = GraphAnalyzer(output_node)
    output_node = graph.get_output_node()
    all_nodes = graph.get_all_nodes()

    # Inputs (Embedding, PosEncoding, InputNode) are pre-allocated by
    # compile.py's initialization; they don't need a time slot. LiteralValue is
    # deliberately NOT an input node — it's materialized just-in-time near its
    # consumer (see is_input_node / constants_plan.md), so it lands in
    # `schedulable` below, not here.
    input_nodes: List[Node] = [n for n in all_nodes if graph.is_input_node(n)]

    schedulable: List[Node] = sorted(
        (
            n
            for n in all_nodes
            if not isinstance(n, Concatenate) and n not in set(input_nodes)
        ),
        key=lambda n: n.node_id,
    )

    # Effective-consumers cache (Concat-transparent).
    consumers_eff: Dict[Node, Set[Node]] = {}
    for n in all_nodes:
        if isinstance(n, Concatenate):
            continue
        _effective_consumers(graph, n, consumers_eff)

    # Edges: every direct dependency, traversing Concatenate inputs.
    edges_set: Set[Tuple[int, int]] = set()
    edges: List[Tuple[Node, Node]] = []
    for v in schedulable:
        for inp in v.inputs:
            if isinstance(inp, Concatenate):
                leaves = flatten_concat_nodes([inp])
            else:
                leaves = [inp]
            for u in leaves:
                key = (u.node_id, v.node_id)
                if key not in edges_set:
                    edges_set.add(key)
                    edges.append((u, v))
        for pred in v.scheduling_predecessors:
            key = (pred.node_id, v.node_id)
            if key not in edges_set:
                edges_set.add(key)
                edges.append((pred, v))

    pinned_nodes: Set[Node] = set(input_nodes)
    pinned_nodes.add(output_node)

    return GraphModel(
        graph=graph,
        schedulable=schedulable,
        edges=edges,
        consumers_eff=consumers_eff,
        output_node=output_node,
        pos_encoding=pos_encoding,
        input_nodes=input_nodes,
        pinned_nodes=pinned_nodes,
    )


# ---------------------------------------------------------------------------
# Routing and cost helpers
# ---------------------------------------------------------------------------
#
# Used internally by the model builder, and by the probe script for
# diagnostic prints. Treat as implementation detail of this module —
# external callers should use `solve_schedule` instead.


def routing(
    node: Node,
    gm: GraphModel,
    policy: SchedulingPolicy,
    usable_slots: Optional[int],
) -> str:
    """Static routing decision under the given policy and geometry.

    Used when `flex_routing=False`: every node has a fixed sublayer.
    With `flex_routing=True`, only `is_flex(n)` nodes' modes become
    CP-SAT decision variables; others still use this routing.

    Derived from the shared option set
    (`torchwright/compiler/realization.py:candidate_classes`) — the same
    declaration the eager path's static resolver reads, so the two paths
    cannot drift apart on what the options are — and from the same
    capacity rule (`static_flex_class`), so they cannot drift apart on
    which options a given geometry actually admits.

    `usable_slots` is the layer's usable MLP hidden-slot count.  It may be
    None only when no node here has a free choice (every caller with
    `flex_routing=True` and no standalone Linears); routing a Linear
    without it raises rather than guessing a sublayer.
    """
    if has_flex_choice(node):
        # Standalone Linear: the policy pins the sublayer here, subject to
        # the MLP-bypass capacity check; with flex_routing=True the CP-SAT
        # choice variable overrides this.
        if usable_slots is None:
            raise ValueError(
                f"routing() needs the layer's usable hidden-slot count to "
                f"place {node!r}: its MLP-bypass realization may not fit the "
                f"geometry, and picking a sublayer without checking deadlocks "
                f"the walk"
            )
        return CLASS_SUBLAYER[static_flex_class(node, policy, usable_slots)]
    (single,) = candidate_classes(node)  # raises TypeError on unknown types
    return CLASS_SUBLAYER[single]


def is_flex(node: Node, gm: GraphModel) -> bool:
    """True iff this node's routing is a CP-SAT decision variable
    when `flex_routing=True`.

    The nodes with a free realization choice
    (`realization.has_flex_choice`): a standalone Linear can run in
    attention (`heads = ⌈d_input/d_head⌉`) or in MLP bypass
    (`slots = 2 · d_output`), and an Add can run in attention (reused or
    fresh placement) or in the MLP bypass pair (`slots = 2 · d_output`,
    docs/plan_additional_mlp_routing.md).  The heuristic picks one
    statically per policy; CP-SAT can pick per-node.  A held-target Add
    is not flex in effect: the routing loop pins its `is_attn` to 1
    before this is consulted.

    `Attn` / `FFN` / `LiteralValue` stay locked (single candidate class).
    """
    return has_flex_choice(node)


def heads_for(node: Node, d_head: int) -> int:
    """Heads consumed if attention-routed.

    Mirrors `LayerScheduler._heads_*`. For `Add`, returns the free-
    add unit count (`⌈d_out/d_head⌉` — one head per `d_head`-wide
    chunk of the live addend, copied into the dead addend's cols);
    the compute-add regime costs `2 ·` this. The CP-SAT model gates
    free vs compute via a per-Add `is_free` boolean derived from
    reified consumer-ordering booleans; see the helper inside
    `solve_schedule` and `docs/cpsat_scheduler.md` §3.

    A `Linear`'s charge is support-aware: one head per `d_head`-wide
    input chunk with any nonzero weight row, floor 1
    (`realization.linear_attn_chunks` — the same list the emitter
    iterates, so the budget and the emission cannot desync).  A dense
    weight matrix charges exactly the old `⌈d_input/d_head⌉`.
    """
    if isinstance(node, Attn):
        return (node.d_v + d_head - 1) // d_head
    if isinstance(node, Linear):
        return linear_attn_heads(node, d_head)
    if isinstance(node, Add):
        d_out = len(node)
        return (d_out + d_head - 1) // d_head
    return 0


def slots_for(node: Node, gm: GraphModel) -> int:
    """MLP slots consumed if MLP-routed.

    An FFN carries one hidden slot per lane (the composite's slot
    demand); a standalone Linear routed to MLP bypass needs `2 ·
    d_output`; an MLP-routed Add needs `2 · d_output` (the bypass lane
    pair, same demand for reused and fresh placement); everything else
    costs no hidden slots.
    """
    if isinstance(node, FFN):
        return node.n_lanes
    if isinstance(node, Linear):
        return 2 * node.d_output  # MLP bypass
    if isinstance(node, Add):
        return 2 * node.d_output  # MLP Add bypass pair
    if isinstance(node, LiteralValue):
        return 0
    return 0


def uses_residual(node: Node, gm: GraphModel) -> bool:
    """True iff this node gets its own residual-stream column allocation.

    Every schedulable node writes its output to the residual stream (a
    FFN's output, a Linear's output, an Add, an Attn, a LiteralValue).
    The FFN's internal ReLU activations live in MLP hidden slots, not
    the residual stream, but the FFN node itself (its output) does use
    residual columns.
    """
    return True


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _compute_layer_bounds(
    gm: "GraphModel",
    policy: SchedulingPolicy,
    flex_routing: bool,
    max_layers: int,
    *,
    usable_slots: Optional[int] = None,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Per-node [earliest, latest] layer bounds from the dependency DAG.

    Mirrors the dependency-constraint semantics exactly: edge u->v allows
    same-layer placement only when u *can* run in the attention sublayer and
    v *can* run in MLP (gap 0); otherwise v must come at least one layer
    after u (gap 1).  "Can" means: flexible routing, or pinned to the
    needed sublayer.  These are the same bounds CP-SAT presolve derives by
    propagation; computing them here shrinks the input model instead.

    ``usable_slots`` is only read when ``flex_routing=False``, where every
    standalone Linear is statically routed and the MLP-bypass capacity check
    decides its sublayer.  Leaving it None there raises inside ``routing()``
    rather than guessing, so the default cannot silently mis-route anything.
    """

    # Per-node allowed sublayers ("modes").  The propagation is mode-aware:
    # an edge u->v is same-layer only for (u=attn, v=mlp), and each node has
    # ONE mode, so two consecutive same-layer hops through a node are
    # impossible (it would need to be mlp as a consumer and attn as a
    # producer).  Tracking earliest/latest per mode captures that; the
    # per-edge relaxation without modes is sound but visibly weaker (proved
    # 40 vs presolve's 50 on the d8192 DOOM graph).
    def modes(n: Node) -> Tuple[str, ...]:
        if flex_routing and is_flex(n, gm):
            return (ATTN, MLP)
        return (routing(n, gm, policy, usable_slots),)

    def gap(a: str, b: str) -> int:
        return 0 if (a == ATTN and b == MLP) else 1

    input_ids = {n.node_id for n in gm.input_nodes}
    node_modes = {n.node_id: modes(n) for n in gm.schedulable}
    edges: List[Tuple[int, int]] = []
    for u, v in gm.edges:
        if u.node_id in input_ids:
            continue
        edges.append((u.node_id, v.node_id))

    ids = [n.node_id for n in gm.schedulable]
    es = {i: {m: 0 for m in node_modes[i]} for i in ids}
    ls = {i: {m: max_layers - 1 for m in node_modes[i]} for i in ids}
    # Fixpoint over the dependency edges; converges in one forward+backward
    # sweep when edges are in topological order (gm.schedulable is
    # build-ordered), but loop defensively in case they are not.
    for _ in range(200):
        changed = False
        for u, v in edges:
            for b in node_modes[v]:
                # u commits to ONE mode; the relaxation lets it pick the
                # best one per edge (sound: real schedules are no earlier).
                lo = min(es[u][a] + gap(a, b) for a in node_modes[u])
                if lo > es[v][b]:
                    es[v][b] = lo
                    changed = True
        for u, v in reversed(edges):
            for a in node_modes[u]:
                hi = max(ls[v][b] - gap(a, b) for b in node_modes[v])
                if hi < ls[u][a]:
                    ls[u][a] = hi
                    changed = True
        if not changed:
            break
    else:
        raise RuntimeError("layer-bound propagation did not converge")
    return (
        {i: min(es[i].values()) for i in ids},
        {i: max(ls[i].values()) for i in ids},
    )


def critical_path_layers(
    output_node: Node,
    pos_encoding=None,
    *,
    policy: Optional[SchedulingPolicy] = None,
    flex_routing: bool = True,
    usable_slots: Optional[int] = None,
) -> int:
    """Exact minimum layer count imposed by the dependency DAG alone.

    This is the mode-aware longest path through the graph (same semantics
    as the CP-SAT dependency constraints), ignoring width entirely.  No
    schedule can be shallower; when the residual stream has slack, the
    optimum EQUALS this value (measured on the DOOM graph at d=8192).
    Costs milliseconds beyond the graph-model build — usable as a
    pre-solve bound or a probe horizon.

    Under the default ``flex_routing=True`` a standalone Linear may take
    either sublayer, so no geometry is consulted and ``usable_slots`` is
    unused.  With ``flex_routing=False`` every Linear is statically routed
    and the layer's usable hidden-slot count decides whether the MLP bypass
    is even available: pass it (:func:`realization.usable_hidden_slots`) or
    the call raises.
    """
    if policy is None:
        policy = LEGACY_POLICY
    gm = build_graph_model(output_node, pos_encoding)
    es, _ = _compute_layer_bounds(
        gm, policy, flex_routing, max_layers=1 << 20, usable_slots=usable_slots
    )
    return max(es.values()) + 1


def _and_presence(model, parked, other, *, name):
    """Presence literal for AND(parked.Not(), ``other``).

    ``other`` is the desired literal already (e.g. ``cim`` or ``cim.Not()``).
    With no parked var the presence is just ``other``; otherwise a fresh aux
    bool is reified to ``parked.Not() AND other`` and returned.
    """
    if parked is None:
        return other
    aux = model.NewBoolVar(name)
    model.AddBoolAnd([parked.Not(), other]).OnlyEnforceIf(aux)
    model.AddBoolOr([parked, other.Not()]).OnlyEnforceIf(aux.Not())
    return aux


def build_cpsat_model(
    output_node: Node,
    pos_encoding=None,
    *,
    d: int,
    d_head: int,
    n_heads: Optional[int] = None,
    d_hidden: int,
    costs: Costs = Costs(),
    flex_routing: bool = True,
    max_layers: int = 60,
    cancel_slack: Optional[int] = 2,
    policy: Optional[SchedulingPolicy] = None,
    reserve_heads: int = 0,
    reserve_residual: int = 0,
    tighten_domains: bool = False,
    diagnostic_hint: Optional[DiagnosticHint] = None,
    held_source_id: Optional[int] = None,
    held_target_id: Optional[int] = None,
    _disabled_families: frozenset = frozenset(),
    _canonical_cancel_reps: bool = False,
    _pin_cancels: bool = True,
) -> BuiltModel:
    """Build (but do not solve) the CP-SAT scheduling model.

    Extracted from :func:`solve_schedule` so the model construction has a
    single definition shared by the production solve and the diagnostic
    path.  ``solve_schedule`` calls this, then adds the warm-start hint, the
    decision strategy, solves, and reads the variables back out of the
    returned :class:`BuiltModel`.

    ``diagnostic_hint.layers`` / ``diagnostic_hint.cancel`` are used only to
    size per-node cancel windows; hint application stays in the solve path.
    With no diagnostic hint the model is byte-identical to a hint-less build.

    ``_disabled_families`` is a diagnostic-only escape hatch: each name in
    :data:`CONSTRAINT_FAMILIES` gates one constraint family, and listing it
    here skips posting that family.  The production path always passes the
    empty set (every family on); the diagnostic path bisects an infeasibility
    by disabling one family at a time over a hard-fixed schedule.  See
    ``torchwright_doom/scripts/cpsat_diagnose.py``.
    """
    gm = build_graph_model(output_node, pos_encoding)
    return build_cpsat_model_from_gm(
        gm,
        d=d,
        d_head=d_head,
        n_heads=n_heads,
        d_hidden=d_hidden,
        costs=costs,
        flex_routing=flex_routing,
        max_layers=max_layers,
        cancel_slack=cancel_slack,
        policy=policy,
        reserve_heads=reserve_heads,
        reserve_residual=reserve_residual,
        tighten_domains=tighten_domains,
        diagnostic_hint=diagnostic_hint,
        held_source_id=held_source_id,
        held_target_id=held_target_id,
        _disabled_families=_disabled_families,
        _canonical_cancel_reps=_canonical_cancel_reps,
        _pin_cancels=_pin_cancels,
    )


def build_cpsat_model_from_gm(
    gm: GraphModel,
    *,
    d: int,
    d_head: int,
    n_heads: Optional[int] = None,
    d_hidden: int,
    costs: Costs = Costs(),
    flex_routing: bool = True,
    max_layers: int = 60,
    cancel_slack: Optional[int] = 2,
    policy: Optional[SchedulingPolicy] = None,
    reserve_heads: int = 0,
    reserve_residual: int = 0,
    tighten_domains: bool = False,
    diagnostic_hint: Optional[DiagnosticHint] = None,
    held_source_id: Optional[int] = None,
    held_target_id: Optional[int] = None,
    _disabled_families: frozenset = frozenset(),
    _canonical_cancel_reps: bool = False,
    _pin_cancels: bool = True,
) -> BuiltModel:
    """Build (but do not solve) the CP-SAT model from a prebuilt GraphModel.

    Shared core of :func:`build_cpsat_model` (live graph) and
    :func:`build_model_from_snapshot` (a stand-in ``GraphModel`` rebuilt from a
    :class:`SchedulingProblem`).  The body reads only structural facts both
    forms provide — op class via ``isinstance``, widths, edges, effective
    consumers, pinned status — so the proto is byte-identical for a live graph
    and its round-tripped snapshot.

    ``d_hidden`` here is the count of hidden slots a layer can hand out, not
    the raw MLP width: ``forward_compile`` passes
    ``realization.usable_hidden_slots(d_hidden, bias)``, which is one less than
    the width when ``bias=False`` reserves slot 0 for the constant lane.  Both
    the hidden-slot cumulative and the static routing rule read it, so they
    admit and place exactly the same set of MLP-bypass Linears.

    ``_canonical_cancel_reps`` is a MEASUREMENT-ONLY knob (default OFF; the
    production path never sets it).  When on, it posts two extra implications
    per node that has both a ``parked`` var and (for D-1) a ``cancel_in_mlp``
    var, collapsing two encoding degeneracies that give one physical schedule
    multiple model representations — see ``cpsat_symmetry_sym1_plan.md``.  With
    it OFF no variable or constraint is added, so the proto is byte-identical
    to today.  It has scope only with ``_pin_cancels=False``: the pinned
    default builds no ``parked`` vars, so both implications are vacuous there.

    ``_pin_cancels`` is the PRODUCTION DEFAULT (on since 2026-07-10; formerly
    the pinned-cancel A/B knob, ``cpsat_pinned_cancel_plan.md`` step 2).  When
    on, every non-keep-forever cancel layer is equality-pinned to its earliest
    legal value given the chosen mechanism (an ``AddMaxEquality`` over the same
    consumer expressions the lower bounds use), replacing the ``parked``
    boolean, the upper-window (``cancel_slack``) constraint, and the
    hint-aware widening for that node; ``cancel_in_mlp`` stays a free decision.
    Every legacy lower bound is kept, so the pinned model only ADDS
    constraints relative to the legacy build — every solution it emits is a
    valid legacy-model solution (machine-valid by construction), verified
    10/10 by re-fixing production-fixture solutions into the unpinned model.
    ``False`` is the escape hatch: it rebuilds the legacy
    window/parked/widening model byte-identically.
    """
    unknown = _disabled_families - CONSTRAINT_FAMILIES
    if unknown:
        raise ValueError(
            f"Unknown constraint family/families {sorted(unknown)}; "
            f"valid names: {sorted(CONSTRAINT_FAMILIES)}"
        )

    hint_layers = diagnostic_hint.layers if diagnostic_hint is not None else None
    hint_cancel = diagnostic_hint.cancel if diagnostic_hint is not None else None

    if policy is None:
        policy = LEGACY_POLICY

    if costs.alpha == 0 and costs.beta == 0 and costs.gamma == 0:
        raise ValueError("alpha=beta=gamma=0 — no objective.")

    if (held_source_id is None) != (held_target_id is None):
        raise ValueError("held_source_id and held_target_id must be supplied together")

    n_heads_per_layer = resolve_n_heads(d, d_head, n_heads, require_divisible=False)
    # ``pos_encoding`` is read by the attention sublayer at (nearly) every
    # layer, so it stays resident for the whole schedule — reserve its columns
    # permanently.  Every OTHER input node is *freeable*: the heuristic frees
    # an input once its last consumer has run and recycles the columns (the
    # wide token ``Embedding`` is freed early and its 600+ columns carry the
    # geometry-stage intermediates).  So those inputs are modelled as residual
    # intervals from layer 0 to a cancel layer, exactly like a scheduled node,
    # rather than being reserved forever.  Reserving every input forever (the
    # previous behaviour) starved intermediates under width pressure and made
    # the residual cumulative reject schedules the heuristic compiles fine.
    # The Δ=0 self-match reads one reserved constant-1 column that stays
    # resident for the whole schedule (the RoPE end state — position is a
    # rotation, no PosEncoding substrate); every input node is freeable
    # (recycled once its last consumer runs).  Additionally, ``reserve_residual``
    # columns are permanently removed from the free pool by ``forward_compile``
    # *before* scheduling (the pinned-constant RMSNorm reserves 1–2 there — see
    # ``_reserve_rms_norm_columns``).  The solver never sees ``residual_map``, so
    # it must subtract BOTH the self-match column and ``reserve_residual`` here,
    # or the modeled capacity over-counts and the solver can emit a peak-occupancy
    # schedule that is infeasible on replay against the reservation-reduced pool
    # (a loud out-of-columns / liveness failure under width pressure, exactly
    # where DOOM-class graphs run).
    freeable_inputs = [n for n in gm.input_nodes if n is not gm.output_node]
    held_source = None
    held_target = None
    if held_source_id is not None:
        by_id = {n.node_id: n for n in gm.graph.get_all_nodes()}
        held_source = by_id.get(held_source_id)
        held_target = by_id.get(held_target_id)
        if held_source is None:
            raise ValueError(
                f"held_source_id {held_source_id} does not name a node in "
                f"the scheduled graph"
            )
        if held_target is None:
            raise ValueError(
                f"held_target_id {held_target_id} does not name a node in "
                f"the scheduled graph"
            )
        if held_source not in freeable_inputs:
            raise ValueError("held source must be a freeable graph input")
        if held_target is not gm.output_node or held_target not in gm.schedulable:
            raise ValueError("held target must be the schedulable graph output")
        if len(held_source) != len(held_target):
            raise ValueError(
                f"held source/target width mismatch: {len(held_source)} != "
                f"{len(held_target)}"
            )
    reserved_residual = 1 + reserve_residual
    available_residual = d - reserved_residual
    if available_residual <= 0:
        raise RuntimeError(
            f"the self-match column (1) plus reserved columns "
            f"({reserve_residual}) require {reserved_residual} residual "
            f"columns, but d={d}. No room for inputs or intermediates."
        )

    model = cp_model.CpModel()

    # ---- layer_var per schedulable node ----
    # With ``tighten_domains`` the domain is [earliest, latest] from DAG
    # propagation instead of the uniform [0, max_layers-1]; presolve would
    # derive the same bounds itself, this just hands them over up front.
    bounds = (
        _compute_layer_bounds(
            gm, policy, flex_routing, max_layers, usable_slots=d_hidden
        )
        if tighten_domains
        else None
    )
    layer_var: Dict[int, cp_model.IntVar] = {}
    layer_var_hi_sum = 0
    layer_var_lo: Dict[int, int] = {}
    layer_bounds: Optional[Dict[int, Tuple[int, int]]] = (
        {} if bounds is not None else None
    )
    for n in gm.schedulable:
        lo, hi = (
            (bounds[0][n.node_id], bounds[1][n.node_id])
            if bounds is not None
            else (0, max_layers - 1)
        )
        if lo > hi:
            raise RuntimeError(
                f"tighten_domains produced empty domain [{lo},{hi}] for "
                f"node {n.node_id} — critical path exceeds max_layers="
                f"{max_layers}"
            )
        layer_var_hi_sum += hi
        layer_var_lo[n.node_id] = lo
        if layer_bounds is not None:
            layer_bounds[n.node_id] = (lo, hi)
        layer_var[n.node_id] = model.NewIntVar(lo, hi, f"L_n{n.node_id}")

    # ---- Routing: is_attn[n] BoolVar (or fixed literal) per node ----
    # is_attn[n] == 1 means the node runs in the attention sublayer
    # at its layer; is_attn[n] == 0 means it runs in MLP.
    is_attn: Dict[int, cp_model.IntVar] = {}
    static_routing: Dict[int, str] = {}
    held_direct_handoff = (
        held_source is not None
        and held_target in gm.consumers_eff.get(held_source, set())
    )
    for n in gm.schedulable:
        if n is held_target and (held_direct_handoff or isinstance(n, Add)):
            # A direct source->target handoff must execute in attention:
            # that sublayer reads the old source value, cancels it, and
            # writes the target into the reclaimed bank in one event.
            # There is intentionally no MLP equivalent.  A held-target Add
            # is pinned to attention even for an INDIRECT handoff (the
            # source cancelled by an earlier consumer, the bank claimed
            # later): the tied contract keeps it on ATTN_ADD with fresh
            # placement under every policy and flex configuration — there
            # is no MLP-phase bank-claim executor
            # (docs/plan_additional_mlp_routing.md, *Tied output*).
            v = model.NewBoolVar(f"is_attn_n{n.node_id}_held_pinned")
            model.Add(v == 1)
            static_routing[n.node_id] = ATTN
        elif flex_routing and is_flex(n, gm):
            if slots_for(n, gm) > d_hidden:
                # The MLP family is structurally infeasible at this geometry
                # (its whole-layer slot demand exceeds the usable pool);
                # constrain the route instead of presenting an infeasible
                # MLP mode to the solver.  Mirrors realization.fits_mlp.
                v = model.NewBoolVar(f"is_attn_n{n.node_id}_capacity_pinned")
                model.Add(v == 1)
                static_routing[n.node_id] = ATTN
            else:
                v = model.NewBoolVar(f"is_attn_n{n.node_id}")
                static_routing[n.node_id] = "flex"
        else:
            r = routing(n, gm, policy, d_hidden)
            v = model.NewBoolVar(f"is_attn_n{n.node_id}_pinned")
            if r == ATTN:
                model.Add(v == 1)
            else:
                model.Add(v == 0)
            static_routing[n.node_id] = r
        is_attn[n.node_id] = v

    # ---- Dependency constraints ----
    # Edge u->v: same-layer ok iff u is_attn AND v is mlp (i.e., NOT
    # v is_attn). Otherwise layer[v] > layer[u].
    input_ids = {n.node_id for n in gm.input_nodes}
    if "dependency" not in _disabled_families:
        for u, v in gm.edges:
            if u.node_id in input_ids:
                continue
            u_attn = is_attn[u.node_id]
            v_attn = is_attn[v.node_id]
            # same_layer_ok = u_attn AND (NOT v_attn)
            same_ok = model.NewBoolVar(f"so_n{u.node_id}_n{v.node_id}")
            model.AddBoolAnd([u_attn, v_attn.Not()]).OnlyEnforceIf(same_ok)
            model.AddBoolOr([u_attn.Not(), v_attn]).OnlyEnforceIf(same_ok.Not())
            model.Add(layer_var[v.node_id] >= layer_var[u.node_id]).OnlyEnforceIf(
                same_ok
            )
            model.Add(layer_var[v.node_id] >= layer_var[u.node_id] + 1).OnlyEnforceIf(
                same_ok.Not()
            )

    # ---- Add reusable-placement classification ----
    # (docs/plan_additional_mlp_routing.md, *Route-aware reusable placement*.)
    # Per Add occurrence i, `reusable_i` reifies the sublayer-order deadness
    # predicate: every OTHER effective consumer C of the occurrence satisfies
    #     layer[C] < layer[A]
    #     or (layer[C] == layer[A] and C attention-routed and A MLP-routed)
    # — a same-layer attention consumer is complete by the MLP phase-start
    # snapshot; every other same-layer consumer is not.  The occurrence's own
    # birth needs no term: the dependency gaps above already order it before
    # A's snapshot.  Target selection is deterministic, never a solver
    # choice:
    #     reuse_0 = reusable_0
    #     reuse_1 = NOT reusable_0 AND reusable_1
    #     is_free = reusable_0 OR reusable_1
    # Graph inputs are legitimate targets (they own residual columns and the
    # heuristic reassigns them); reuse is rejected only for a physically
    # non-reassignable value (Concatenate, the held source — its columns end
    # through the held-bank cancel/hold transition) or an unordered
    # effective consumer (e.g. a terminal Concatenate).  The held target
    # skips the construction entirely: its `is_free` is pinned 0 without the
    # deadness biconditional (the 32983b0 contract).  Assignment-level
    # mirror: `add_placement.derive_add_placement` — extraction asserts the
    # two agree while the literals still exist.
    is_free: Dict[int, cp_model.IntVar] = {}
    add_reusable: Dict[Tuple[int, int], cp_model.IntVar] = {}
    add_reuse: Dict[Tuple[int, int], cp_model.IntVar] = {}
    # `add_attn_gap[A]` = OR(is_free[A], NOT is_attn[A]): the extra layer an
    # ordinary source of A must stay alive under an ATTENTION-mechanism
    # cancel — 1 for any reuse (gap #2 conservatism, both routes) and for an
    # MLP-routed Add (its sources are read after the attention cancel would
    # fire).  The MLP-mechanism term stays `is_free[A]` alone.
    add_attn_gap: Dict[int, cp_model.IntVar] = {}
    # Old value -> the selector literals that reassign it; mutually exclusive
    # by construction (posted below) so one residual owner cannot be
    # reassigned twice.
    target_selectors: Dict[int, List[cp_model.IntVar]] = {}
    # (selector, add_id, target node): the virtual-handoff equality
    # `cancel == layer[A] + 1` is posted after the cancel vars exist.
    target_cancel_specs: List[Tuple[cp_model.IntVar, int, Node]] = []
    for A in gm.schedulable:
        if not isinstance(A, Add):
            continue
        if A is held_target:
            # The output must be a fresh write into the zeroed held bank — a
            # free Add would reassign an addend's unrelated allocation
            # instead — and the executors always compute it fresh
            # (LayerScheduler._is_forced_fresh_add) regardless of addend
            # deadness.  Pin is_free to 0 WITHOUT the deadness biconditional:
            # posting both would (a) go hard-INFEASIBLE whenever an addend's
            # only consumer is this Add (reusable is the constant 1, forcing
            # is_free to 1 against the pin — the canonical
            # `logits = transported_embedding + correction` shape), and
            # (b) otherwise spuriously force every addend not-dead.  The
            # pinned var still lands in `is_free`: every Add is indexed
            # downstream (Add-consumer cancel bounds, residual start shift,
            # attention head charge).
            is_free_held = model.NewBoolVar(f"is_free_A{A.node_id}")
            model.Add(is_free_held == 0)
            is_free[A.node_id] = is_free_held
            gap_held = model.NewBoolVar(f"add_src_attn_gap_A{A.node_id}")
            model.AddBoolOr([is_free_held, is_attn[A.node_id].Not()]).OnlyEnforceIf(
                gap_held
            )
            model.AddBoolAnd([is_free_held.Not(), is_attn[A.node_id]]).OnlyEnforceIf(
                gap_held.Not()
            )
            add_attn_gap[A.node_id] = gap_held
            continue
        a_layer = layer_var[A.node_id]
        a_attn = is_attn[A.node_id]
        occurrence_literals: List[cp_model.IntVar] = []
        for i, E in enumerate(A.inputs):
            if i == 1 and E is A.inputs[0]:
                # add(x, x): one node, one consumer set, one deadness value —
                # share the occurrence-0 literal instead of rebuilding its
                # reified constraints.
                r = occurrence_literals[0]
                occurrence_literals.append(r)
                add_reusable[(A.node_id, i)] = r
                continue
            r = model.NewBoolVar(f"reusable_A{A.node_id}_i{i}")
            add_reusable[(A.node_id, i)] = r
            occurrence_literals.append(r)
            if isinstance(E, Concatenate) or E.node_id == held_source_id:
                model.Add(r == 0)
                continue
            if E.node_id not in layer_var and E not in gm.input_nodes:
                # Neither schedulable nor a residual-owning graph input.
                model.Add(r == 0)
                continue
            other_consumers = [c for c in gm.consumers_eff.get(E, set()) if c is not A]
            if any(
                isinstance(c, Concatenate) or c.node_id not in layer_var
                for c in other_consumers
            ):
                # An unordered read (terminal Concatenate / non-schedulable
                # consumer) can never be sequenced before A's snapshot.
                model.Add(r == 0)
                continue
            complete_bools: List[cp_model.IntVar] = []
            for C in sorted(other_consumers, key=lambda c: c.node_id):
                c_layer = layer_var[C.node_id]
                b_lt = model.NewBoolVar(
                    f"lt_E{E.node_id}_C{C.node_id}_A{A.node_id}_i{i}"
                )
                model.Add(c_layer < a_layer).OnlyEnforceIf(b_lt)
                model.Add(c_layer >= a_layer).OnlyEnforceIf(b_lt.Not())
                b_gt = model.NewBoolVar(
                    f"gt_E{E.node_id}_C{C.node_id}_A{A.node_id}_i{i}"
                )
                model.Add(c_layer > a_layer).OnlyEnforceIf(b_gt)
                model.Add(c_layer <= a_layer).OnlyEnforceIf(b_gt.Not())
                # eq_ok = same layer AND C attention-routed AND A MLP-routed.
                eq_ok = model.NewBoolVar(
                    f"eqok_E{E.node_id}_C{C.node_id}_A{A.node_id}_i{i}"
                )
                model.AddBoolAnd(
                    [b_lt.Not(), b_gt.Not(), is_attn[C.node_id], a_attn.Not()]
                ).OnlyEnforceIf(eq_ok)
                model.AddBoolOr(
                    [b_lt, b_gt, is_attn[C.node_id].Not(), a_attn]
                ).OnlyEnforceIf(eq_ok.Not())
                complete = model.NewBoolVar(
                    f"complete_E{E.node_id}_C{C.node_id}_A{A.node_id}_i{i}"
                )
                model.AddBoolOr([b_lt, eq_ok]).OnlyEnforceIf(complete)
                model.AddBoolAnd([b_lt.Not(), eq_ok.Not()]).OnlyEnforceIf(
                    complete.Not()
                )
                complete_bools.append(complete)
            if complete_bools:
                model.AddBoolAnd(complete_bools).OnlyEnforceIf(r)
                model.AddBoolOr([b.Not() for b in complete_bools]).OnlyEnforceIf(
                    r.Not()
                )
            else:
                model.Add(r == 1)  # E feeds only A
        # Deterministic selector: occurrence 0 wins.
        reuse_0 = occurrence_literals[0]
        add_reuse[(A.node_id, 0)] = reuse_0
        reuse_1 = model.NewBoolVar(f"reuse_A{A.node_id}_i1")
        model.AddBoolAnd(
            [occurrence_literals[0].Not(), occurrence_literals[1]]
        ).OnlyEnforceIf(reuse_1)
        model.AddBoolOr(
            [occurrence_literals[0], occurrence_literals[1].Not()]
        ).OnlyEnforceIf(reuse_1.Not())
        add_reuse[(A.node_id, 1)] = reuse_1
        is_free_A = model.NewBoolVar(f"is_free_A{A.node_id}")
        model.AddBoolOr(occurrence_literals).OnlyEnforceIf(is_free_A)
        model.AddBoolAnd([b.Not() for b in occurrence_literals]).OnlyEnforceIf(
            is_free_A.Not()
        )
        is_free[A.node_id] = is_free_A
        gap_A = model.NewBoolVar(f"add_src_attn_gap_A{A.node_id}")
        model.AddBoolOr([is_free_A, a_attn.Not()]).OnlyEnforceIf(gap_A)
        model.AddBoolAnd([is_free_A.Not(), a_attn]).OnlyEnforceIf(gap_A.Not())
        add_attn_gap[A.node_id] = gap_A
        for i, sel in ((0, reuse_0), (1, reuse_1)):
            E = A.inputs[i]
            target_selectors.setdefault(E.node_id, []).append(sel)
            target_cancel_specs.append((sel, A.node_id, E))

    # One residual owner is reassigned at most once; `reused_as_target[E]`
    # is the OR of the selectors naming E (an ownership handoff, not a
    # parked value or a cancel).
    reused_as_target: Dict[int, cp_model.IntVar] = {}
    for eid, sels in target_selectors.items():
        model.AddAtMostOne(sels)
        rat = model.NewBoolVar(f"reused_as_target_n{eid}")
        model.AddBoolOr(sels).OnlyEnforceIf(rat)
        model.AddBoolAnd([s.Not() for s in sels]).OnlyEnforceIf(rat.Not())
        reused_as_target[eid] = rat

    # ---- Cancel layer per schedulable node ----
    # The natural lower bound on cancel_layer[n] is
    # ``max(layer[c] + 1)`` over consumers — the columns must outlive
    # every reader.  The natural upper bound is ``max_layers``, which
    # leaves ~60 candidate values per node on a DOOM-scale graph.
    # When ``cancel_slack`` is set, restrict to a small window above
    # the lower bound: the heuristic almost always cancels within 1–2
    # layers of the last consumer, so K=2 cuts the cancel decision
    # space ~30x with negligible loss of optimality.
    # Under `_pin_cancels` the equality pin (posted after `is_free` exists,
    # below) subsumes the upper window entirely: forcing `eff_cancel_slack`
    # to None here skips the whole window/parked/widening block per node,
    # exactly the family the pin replaces.
    eff_cancel_slack = (
        None if ("cancel_slack" in _disabled_families or _pin_cancels) else cancel_slack
    )

    # Hint-aware cancel-window widening.  Freeing a node's columns costs
    # attention-head work charged against the same per-layer head budget as
    # compute, so the heuristic warm-start *defers* a free when a layer's
    # heads are full (`try_add_cancel` returns None — the free retries next
    # layer).  A deferred free lands past the uniform
    # `last_consumer + 1 + K` window, making the whole warm-start hint
    # infeasible under this model: CP-SAT silently drops it, cold-searches,
    # and `forward_compile` falls back to the heuristic (the optimize=2
    # fallback incident, 2026-07).  For each hinted node, widen its window by
    # exactly the amount the hint needs:
    # `delta_n = max(0, hint_cancel[n] - (hint_base + 1 + K))` where
    # `hint_base` is the hinted last-consumer layer (or the hinted birth
    # layer when the node has no layer-bound consumers; 0 for freeable
    # inputs).  `delta_n = 0` whenever the node or any needed consumer lacks
    # a hint entry.  This is a pure relaxation: the cancel still pays its
    # attention-head charge at whatever layer it lands and the residual
    # interval still spans [layer, cancel), so no invalid schedule becomes
    # expressible — the window just admits the schedules the heuristic
    # actually emits, preserving the moving near-last-consumer shape.
    cancel_window_delta: Dict[int, int] = {}

    def _widen_delta(nid: int, hint_base: Optional[int]) -> int:
        if hint_cancel is None or hint_base is None or eff_cancel_slack is None:
            return 0
        hinted = hint_cancel.get(nid)
        if hinted is None:
            return 0
        delta = max(0, hinted - (hint_base + 1 + eff_cancel_slack))
        if delta:
            cancel_window_delta[nid] = delta
        return delta

    def _hinted_last_consumer(consumer_ids: List[int]) -> Optional[int]:
        if hint_layers is None or not consumer_ids:
            return None
        hinted = [hint_layers.get(cid) for cid in consumer_ids]
        if any(L is None for L in hinted):
            return None
        return max(hinted)

    cancel_layer: Dict[int, cp_model.IntVar] = {}
    keep_forever_ids: Set[int] = set()
    # Nodes whose cancel window carries a `parked` escape (cl == max_layers
    # allowed instead of the near-last-consumer window).  Their cancel-head
    # interval is gated absent when parked (below), so parking charges no head.
    parked_by_id: Dict[int, cp_model.IntVar] = {}
    # `cancel_in_mlp[n]` == 1 routes n's death cancel to the MLP sublayer (a
    # `cancel_bypass` op charged 2·len(n) hidden slots) instead of a batched
    # attention cancel head.  Built only for non-keep-forever schedulable nodes;
    # freeable inputs never get it (decision #3).  It relaxes the cancel bound
    # to the uniform gap-0 `cl >= layer[c]` for every consumer (an MLP cancel
    # fires after both sublayers' reads) and moves the cancel cost from the
    # attention-head cumulative to the MLP-slot cumulative.
    cancel_in_mlp: Dict[int, cp_model.IntVar] = {}
    # Add-consumer cancel lower bounds are posted with the mechanism-specific
    # Add terms (below): the gap between a source's cancel and the Add's
    # layer depends on the reuse regime, the Add's route, and the cancel
    # mechanism.
    deferred_add_consumer_lbs: List[Tuple[cp_model.IntVar, cp_model.IntVar, int]] = []
    # `_pin_cancels` equality pins are likewise deferred until `is_free`
    # exists (Add-consumer terms read it): per node, (node_id, cl, cim,
    # non-Add consumer ids with layer vars, Add consumer ids with layer vars).
    deferred_pin_specs: List[
        Tuple[int, cp_model.IntVar, cp_model.IntVar, List[int], List[int]]
    ] = []

    def _canonicalize_cancel_reps(cl, parked, cim, rat=None):
        # sym1 (MEASUREMENT-ONLY, gated by `_canonical_cancel_reps`; default OFF
        # posts nothing, keeping the proto byte-identical).  Removes two
        # encoding degeneracies where one physical schedule has multiple model
        # representations — see cpsat_symmetry_sym1_plan.md.
        #   D-1: under `parked` no cancel op executes, so `cancel_in_mlp` only
        #   selects which cost pool a cancel *would* charge — both pool
        #   presences are already gated absent.  Pin it to the attention side
        #   (cim = 0), which also drops the phantom residual-end extension
        #   `rend = cancel_layer + cim`.  Freeable inputs have no `cim`.
        #   D-2: `parked ⇒ cancel_layer == max_layers` is already posted; add
        #   the converse so "never freed in-horizon" has the single `parked`
        #   encoding, not also the duplicate `parked = 0, cancel = max_layers`
        #   (which additionally charges a phantom cancel-head column at the
        #   virtual layer).  A reused target is an ownership handoff, not a
        #   parked value: its parked literal is forced false and its virtual
        #   end may legitimately equal `max_layers` (a final-layer Add), so
        #   the converse is gated off under `reused_as_target`.
        if not _canonical_cancel_reps:
            return
        if cim is not None:
            model.AddImplication(parked, cim.Not())
        enforce = [parked.Not()] if rat is None else [parked.Not(), rat.Not()]
        model.Add(cl <= max_layers - 1).OnlyEnforceIf(enforce)

    for n in gm.schedulable:
        cl = model.NewIntVar(0, max_layers, f"cl_n{n.node_id}")
        cancel_layer[n.node_id] = cl
        model.Add(cl >= layer_var[n.node_id] + 1)
        if n in gm.pinned_nodes:
            model.Add(cl == max_layers)
            keep_forever_ids.add(n.node_id)
            continue
        consumers = gm.consumers_eff.get(n, set())
        if any(isinstance(c, Concatenate) for c in consumers):
            # Consumed by a terminal Concatenate (output cone) — keep forever.
            model.Add(cl == max_layers)
            keep_forever_ids.add(n.node_id)
            continue
        # Non-keep-forever: this node gets a mechanism choice.
        cim = model.NewBoolVar(f"cancel_in_mlp_n{n.node_id}")
        cancel_in_mlp[n.node_id] = cim
        consumer_layer_vars: List[cp_model.IntVar] = []
        consumer_ids: List[int] = []
        for c in consumers:
            if c.node_id in layer_var:
                if "cancel_consumer_lb" not in _disabled_families:
                    # Intra-layer reuse: an attention-sublayer consumer reads
                    # the residual as it entered the layer, and the cancel is
                    # itself an additive delta in that same attention output,
                    # so the consumer's OWN layer may reclaim the columns
                    # (gap 0).  An MLP-routed consumer reads post-attention
                    # state and keeps the layer-after bound (gap 1).
                    # `1 - is_attn[c]` is exactly that gap; presolve folds it
                    # to a constant for pinned-routing consumers.  Under an MLP
                    # cancel the bound relaxes to the uniform gap-0 `cl >=
                    # layer[c]` for every consumer — the MLP cancel fires after
                    # both sublayers' reads (decision #1).  Add consumers depend
                    # on the free/compute regime and are posted after `is_free`;
                    # their `cl >= layer[A] + is_free[A]` stays unconditional
                    # (decision #2) and already dominates the uniform MLP bound.
                    if isinstance(c, Add):
                        deferred_add_consumer_lbs.append((cl, cim, c.node_id))
                    else:
                        model.Add(
                            cl >= layer_var[c.node_id] + 1 - is_attn[c.node_id]
                        ).OnlyEnforceIf(cim.Not())
                        model.Add(cl >= layer_var[c.node_id]).OnlyEnforceIf(cim)
                consumer_layer_vars.append(layer_var[c.node_id])
                consumer_ids.append(c.node_id)
        if _pin_cancels:
            deferred_pin_specs.append(
                (
                    n.node_id,
                    cl,
                    cim,
                    [
                        c.node_id
                        for c in consumers
                        if c.node_id in layer_var and not isinstance(c, Add)
                    ],
                    [
                        c.node_id
                        for c in consumers
                        if c.node_id in layer_var and isinstance(c, Add)
                    ],
                )
            )
        rat = reused_as_target.get(n.node_id)
        if eff_cancel_slack is not None and consumer_layer_vars:
            delta = _widen_delta(n.node_id, _hinted_last_consumer(consumer_ids))
            last_cons = model.NewIntVar(0, max_layers - 1, f"last_cons_n{n.node_id}")
            model.AddMaxEquality(last_cons, consumer_layer_vars)
            # Parked escape: the machine may leave a dead value's columns
            # allocated forever when every layer's head budget is too full
            # to pay the cancel.  cl == max_layers models exactly that; the
            # cancel-head interval is gated absent when parked so no head is
            # charged in-horizon.  Without it the window would FORCE a paid
            # cancel near the last consumer, rejecting machine schedules whose
            # pinch layers have no head slack.  A reused target is an
            # ownership handoff, not a parked value: its parked literal is
            # forced false and the ordinary window is gated off — the
            # selector posts the exact `layer[A] + 1` handoff instead.
            parked = model.NewBoolVar(f"parked_n{n.node_id}")
            parked_by_id[n.node_id] = parked
            window_gate = [parked.Not()] if rat is None else [parked.Not(), rat.Not()]
            model.Add(cl <= last_cons + 1 + eff_cancel_slack + delta).OnlyEnforceIf(
                window_gate
            )
            model.Add(cl == max_layers).OnlyEnforceIf(parked)
            if rat is not None:
                model.AddImplication(rat, parked.Not())
            _canonicalize_cancel_reps(cl, parked, cim, rat)
        elif eff_cancel_slack is not None and not consumer_layer_vars:
            # No layer-bound consumers — cancel can fire right after
            # the node's own birth layer.  (Such a node has no Add consumer,
            # so it can never be a reuse target; no gating needed.)
            delta = _widen_delta(
                n.node_id,
                hint_layers.get(n.node_id) if hint_layers is not None else None,
            )
            parked = model.NewBoolVar(f"parked_n{n.node_id}")
            parked_by_id[n.node_id] = parked
            model.Add(
                cl <= layer_var[n.node_id] + 1 + eff_cancel_slack + delta
            ).OnlyEnforceIf(parked.Not())
            model.Add(cl == max_layers).OnlyEnforceIf(parked)
            _canonicalize_cancel_reps(cl, parked, cim)

    # ---- Freeable input cancel layers ----
    # Freeable inputs are born at layer 0 (pre-allocated by the compiler
    # before the layer loop) and live until their last consumer runs, mirroring
    # the schedulable cancel logic with a fixed birth at 0.  An input feeding a
    # terminal `Concatenate` (output cone) is kept forever.
    input_cancel_layer: Dict[int, cp_model.IntVar] = {}
    input_keep_ids: Set[int] = set()
    parked_input_by_id: Dict[int, cp_model.IntVar] = {}
    held_input_pin_spec = None
    for n in freeable_inputs:
        cl = model.NewIntVar(0, max_layers, f"cl_in{n.node_id}")
        input_cancel_layer[n.node_id] = cl
        is_held_source = n is held_source
        if not is_held_source:
            model.Add(cl >= 1)  # ordinary input lives through layer 0
        keep_forever = False
        consumer_layer_vars = []
        consumer_ids = []
        for c in gm.consumers_eff.get(n, set()):
            if isinstance(c, Concatenate):
                model.Add(cl == max_layers)
                keep_forever = True
                break
            if c.node_id in layer_var:
                if "cancel_consumer_lb" not in _disabled_families:
                    if is_held_source:
                        # Attention readers permit cancel-after-read in their
                        # own layer; MLP readers require the following layer.
                        # Add's free/compute-dependent term is deferred until
                        # is_free exists below.
                        if not isinstance(c, Add):
                            model.Add(
                                cl >= layer_var[c.node_id] + 1 - is_attn[c.node_id]
                            )
                    else:
                        model.Add(cl >= layer_var[c.node_id] + 1)
                consumer_layer_vars.append(layer_var[c.node_id])
                consumer_ids.append(c.node_id)
        if keep_forever:
            input_keep_ids.add(n.node_id)
            continue
        if is_held_source:
            non_add_ids = [
                c.node_id
                for c in gm.consumers_eff.get(n, set())
                if c.node_id in layer_var and not isinstance(c, Add)
            ]
            add_ids = [
                c.node_id
                for c in gm.consumers_eff.get(n, set())
                if c.node_id in layer_var and isinstance(c, Add)
            ]
            held_input_pin_spec = (cl, non_add_ids, add_ids)
            # The bank is held from this physical cancel until target birth.
            model.Add(cl <= layer_var[held_target.node_id])
            # The held source has its own equality/window treatment below,
            # after Add classification exists.
            continue
        rat_in = reused_as_target.get(n.node_id)
        if _pin_cancels:
            # Inputs have no mechanism choice (always attention-cancelled), so
            # the earliest legal cancel is the uniform gap-1 bound the model
            # posts above: max(1, layer[c] + 1) over layer-bound consumers,
            # falling back to the birth-based earliest (layer 1) when none.
            # A selected reuse target's pin evaluates to exactly the
            # `layer[A] + 1` handoff (the Add is one of the consumers and
            # every other consumer completes no later), so no gating is
            # needed here — the selector's equality is consistent.
            if consumer_layer_vars:
                model.AddMaxEquality(cl, [1] + [v + 1 for v in consumer_layer_vars])
            else:
                model.Add(cl == 1)
            continue
        if eff_cancel_slack is not None and consumer_layer_vars:
            delta = _widen_delta(n.node_id, _hinted_last_consumer(consumer_ids))
            last_cons = model.NewIntVar(0, max_layers - 1, f"last_cons_in{n.node_id}")
            model.AddMaxEquality(last_cons, consumer_layer_vars)
            # Parked escape — see the schedulable-node cancel window above.
            # A selected graph-input target is an ownership handoff: parked
            # forced false, ordinary window gated off.
            parked = model.NewBoolVar(f"parked_in{n.node_id}")
            parked_input_by_id[n.node_id] = parked
            window_gate = (
                [parked.Not()] if rat_in is None else [parked.Not(), rat_in.Not()]
            )
            model.Add(cl <= last_cons + 1 + eff_cancel_slack + delta).OnlyEnforceIf(
                window_gate
            )
            model.Add(cl == max_layers).OnlyEnforceIf(parked)
            if rat_in is not None:
                model.AddImplication(rat_in, parked.Not())
            _canonicalize_cancel_reps(cl, parked, None, rat_in)
        elif eff_cancel_slack is not None and not consumer_layer_vars:
            # Born at layer 0, so the hinted base is the fixed birth layer 0.
            delta = _widen_delta(n.node_id, 0)
            parked = model.NewBoolVar(f"parked_in{n.node_id}")
            parked_input_by_id[n.node_id] = parked
            model.Add(cl <= 1 + eff_cancel_slack + delta).OnlyEnforceIf(parked.Not())
            model.Add(cl == max_layers).OnlyEnforceIf(parked)
            _canonicalize_cancel_reps(cl, parked, None)

    # Every Add-consumer cancel bound below shares two mechanism-specific
    # terms.  An Add A bounds an ordinary source's ATTENTION-mechanism
    # cancel at `layer[A] + add_attn_gap[A]` (gap 1 for any reuse — known
    # optimality gap #2, both routes — and for an MLP-routed Add, whose
    # sources are read after a same-layer attention cancel would fire) and
    # its MLP-mechanism cancel at `layer[A] + is_free[A]` (an MLP cancel
    # fires after both sublayers' reads, so only the reuse conservatism
    # remains).  The reused TARGET's exact `layer[A] + 1` handoff is posted
    # separately under the selecting literal — `is_free` alone cannot
    # distinguish the target from the live source.  The diagnostic-only
    # "add_live_addend_gap" family relaxes both terms to `layer[A]` for a
    # bindingness lower bound; a schedule solved with it disabled must NEVER
    # be compiled/replayed.
    def _add_consumer_cancel_expr_attn(add_id: int):
        if "add_live_addend_gap" in _disabled_families:
            return layer_var[add_id]
        return layer_var[add_id] + add_attn_gap[add_id]

    def _add_consumer_cancel_expr_mlp(add_id: int):
        if "add_live_addend_gap" in _disabled_families:
            return layer_var[add_id]
        return layer_var[add_id] + is_free[add_id]

    if held_input_pin_spec is not None:
        cl, non_add_ids, add_ids = held_input_pin_spec
        add_exprs = [_add_consumer_cancel_expr_attn(a) for a in add_ids]
        if "cancel_consumer_lb" not in _disabled_families:
            for expr in add_exprs:
                model.Add(cl >= expr)
        if _pin_cancels:
            model.AddMaxEquality(
                cl,
                [0] + [layer_var[c] + 1 - is_attn[c] for c in non_add_ids] + add_exprs,
            )

    # ---- Deferred Add-consumer cancel lower bounds ----
    # (See the cancel-layer section.)  Mechanism-gated like the non-Add
    # consumer bounds: the attention term carries `add_attn_gap[A]` (reuse
    # conservatism OR the MLP-routed Add's post-cancel read), the MLP term
    # carries `is_free[A]` alone.  For an attention-routed Add both terms
    # equal the historical `layer[A] + is_free[A]`.
    if "cancel_consumer_lb" not in _disabled_families:
        for cl, cim, add_id in deferred_add_consumer_lbs:
            model.Add(cl >= _add_consumer_cancel_expr_attn(add_id)).OnlyEnforceIf(
                cim.Not()
            )
            model.Add(cl >= _add_consumer_cancel_expr_mlp(add_id)).OnlyEnforceIf(cim)

    # ---- Reused-target ownership handoff ----
    # Under the selecting literal, the old target's lifetime ends exactly at
    # the boundary where the Add's shifted residual interval begins: a
    # virtual cancel layer `layer[A] + 1` (bookkeeping only — no cancel op
    # executes; ownership ends through `reassign`).  The physical cancel
    # intervals and the parked machinery are gated off under
    # `reused_as_target` where they are built, and `cancel_in_mlp` is forced
    # to the canonical attention side.  Skipped under the
    # "add_live_addend_gap" relaxation, whose lowered Add terms would
    # contradict the exact handoff.
    if "add_live_addend_gap" not in _disabled_families:
        for sel, add_id, target in target_cancel_specs:
            cl_target = cancel_layer.get(target.node_id)
            if cl_target is None:
                cl_target = input_cancel_layer.get(target.node_id)
            if cl_target is None:
                continue  # non-reassignable occurrence (selector is pinned 0)
            model.Add(cl_target == layer_var[add_id] + 1).OnlyEnforceIf(sel)
    for eid, rat in reused_as_target.items():
        cim = cancel_in_mlp.get(eid)
        if cim is not None:
            # Canonical mechanism for a reassigned target: the attention
            # side (never run — the cancel intervals are gated absent).
            model.AddImplication(rat, cim.Not())

    # ---- Pinned cancel layers (_pin_cancels, the production default) ----
    # Equality-pin each non-keep-forever cancel to its earliest legal value
    # given the mechanism, mirroring the lower bounds above term for term:
    # under an attention cancel (cim = 0) the earliest is
    # max(layer[n] + 1, layer[c] + 1 - is_attn[c] per non-Add consumer,
    # layer[A] + add_attn_gap[A] per Add consumer); under an MLP cancel
    # (cim = 1) the non-Add term relaxes to the uniform gap-0 layer[c] and
    # the Add term to layer[A] + is_free[A].  `cancel_in_mlp` stays free —
    # it still chooses which budget pays.  The kept lower bounds make each
    # pin an upper-bound-only addition, so the pinned model is a pure
    # restriction of the legacy (knob-off) model.  For a selected reuse
    # target the attention pin evaluates to exactly the `layer[A] + 1`
    # handoff (the selecting Add's term dominates and cim is forced 0), so
    # the selector's equality needs no gating here.
    if _pin_cancels:
        for nid, cl, cim, non_add_ids, add_ids in deferred_pin_specs:
            birth = layer_var[nid] + 1
            if not non_add_ids and not add_ids:
                # No layer-bound consumers: both mechanisms share the
                # birth-based earliest.
                model.Add(cl == birth)
                continue
            pin_attn = model.NewIntVar(1, max_layers, f"pin_attn_n{nid}")
            model.AddMaxEquality(
                pin_attn,
                [birth]
                + [layer_var[c] + 1 - is_attn[c] for c in non_add_ids]
                + [_add_consumer_cancel_expr_attn(a) for a in add_ids],
            )
            pin_mlp = model.NewIntVar(1, max_layers, f"pin_mlp_n{nid}")
            model.AddMaxEquality(
                pin_mlp,
                [birth]
                + [layer_var[c] for c in non_add_ids]
                + [_add_consumer_cancel_expr_mlp(a) for a in add_ids],
            )
            model.Add(cl == pin_attn).OnlyEnforceIf(cim.Not())
            model.Add(cl == pin_mlp).OnlyEnforceIf(cim)

    # ---- Combined attn-heads + cancel-cols cumulative ----
    # Per-node attn interval is OPTIONAL (gated by is_attn[n]) when
    # the node could run in either sublayer; for pinned nodes, the
    # bool is constant and CP-SAT presolve drops the unreachable
    # branch.
    #
    # `Add` gets two optional intervals — one demand for the reused
    # placement (`⌈d_out/d_head⌉` heads), one for fresh (`2 ·
    # ⌈d_out/d_head⌉` heads) — each additionally gated by `is_attn[A]`:
    # an MLP-routed Add charges no heads (its `2·d_out` hidden slots ride
    # the MLP cumulative via `slots_for`).
    attn_intervals: List = []
    attn_demands: List[int] = []
    # Reused/fresh presence bools per attention-routed Add, shared with the
    # objective counters below.
    add_attn_free_pres: Dict[int, cp_model.IntVar] = {}
    add_attn_comp_pres: Dict[int, cp_model.IntVar] = {}
    for n in gm.schedulable:
        h = heads_for(n, d_head)
        if h <= 0:
            continue
        if isinstance(n, Add):
            free_pres = model.NewBoolVar(f"add_attn_free_n{n.node_id}")
            model.AddBoolAnd([is_attn[n.node_id], is_free[n.node_id]]).OnlyEnforceIf(
                free_pres
            )
            model.AddBoolOr(
                [is_attn[n.node_id].Not(), is_free[n.node_id].Not()]
            ).OnlyEnforceIf(free_pres.Not())
            add_attn_free_pres[n.node_id] = free_pres
            comp_pres = model.NewBoolVar(f"add_attn_comp_n{n.node_id}")
            model.AddBoolAnd(
                [is_attn[n.node_id], is_free[n.node_id].Not()]
            ).OnlyEnforceIf(comp_pres)
            model.AddBoolOr(
                [is_attn[n.node_id].Not(), is_free[n.node_id]]
            ).OnlyEnforceIf(comp_pres.Not())
            add_attn_comp_pres[n.node_id] = comp_pres

            free_end = model.NewIntVar(1, max_layers, f"aend_free_n{n.node_id}")
            model.Add(free_end == layer_var[n.node_id] + 1)
            iv_free = model.NewOptionalIntervalVar(
                layer_var[n.node_id],
                1,
                free_end,
                free_pres,
                f"aiv_free_n{n.node_id}",
            )
            attn_intervals.append(iv_free)
            attn_demands.append(h * d_head)

            comp_end = model.NewIntVar(1, max_layers, f"aend_comp_n{n.node_id}")
            model.Add(comp_end == layer_var[n.node_id] + 1)
            iv_comp = model.NewOptionalIntervalVar(
                layer_var[n.node_id],
                1,
                comp_end,
                comp_pres,
                f"aiv_comp_n{n.node_id}",
            )
            attn_intervals.append(iv_comp)
            attn_demands.append(2 * h * d_head)
            continue
        end = model.NewIntVar(1, max_layers, f"aend_n{n.node_id}")
        model.Add(end == layer_var[n.node_id] + 1)
        iv = model.NewOptionalIntervalVar(
            layer_var[n.node_id], 1, end, is_attn[n.node_id], f"aiv_n{n.node_id}"
        )
        attn_intervals.append(iv)
        attn_demands.append(h * d_head)

    cancel_intervals: List = []
    cancel_demands: List[int] = []
    # MLP-cancel cost intervals accumulate here and join the MLP-slot cumulative
    # below (demand 2·len — the bypass lane-pair costs two hidden slots per
    # column), positioned at the cancel layer rather than the compute layer.
    mlp_cancel_intervals: List = []
    mlp_cancel_demands: List[int] = []
    for n in gm.schedulable:
        if n in gm.pinned_nodes:
            continue
        if not uses_residual(n, gm):
            # A node with no residual-stream column (none today — every
            # schedulable node writes its output to the stream) has no
            # columns to cancel.
            continue
        c_end = model.NewIntVar(1, max_layers + 1, f"cend_n{n.node_id}")
        model.Add(c_end == cancel_layer[n.node_id] + 1)
        parked = parked_by_id.get(n.node_id)
        cim = cancel_in_mlp.get(n.node_id)
        rat = reused_as_target.get(n.node_id)
        if cim is not None:
            # Non-keep-forever node: parked / reassigned-away / attention-
            # cancel / MLP-cancel are mutually exclusive presences.  A dead
            # value is never freed in-horizon (parked, no cancel op runs), or
            # its ownership ends through an Add's `reassign`
            # (`reused_as_target` — no cancel op runs and NO death cancel is
            # charged; the virtual `layer[A] + 1` handoff is bookkeeping), or
            # its columns are zeroed by a batched attention cancel head
            # (attention pool), or by a `cancel_bypass` MLP op (MLP-slot
            # pool).  The gated intervals are pure COSTS; the booleans only
            # move the charge between pools or drop it where no physical
            # cancel exists.
            attn_literal = (
                cim.Not()
                if rat is None
                else _and_presence(
                    model, rat, cim.Not(), name=f"cpres_attn_rat_n{n.node_id}"
                )
            )
            attn_present = _and_presence(
                model, parked, attn_literal, name=f"cpres_attn_n{n.node_id}"
            )
            iv_attn = model.NewOptionalIntervalVar(
                cancel_layer[n.node_id], 1, c_end, attn_present, f"civ_n{n.node_id}"
            )
            cancel_intervals.append(iv_attn)
            cancel_demands.append(len(n))

            mlp_literal = (
                cim
                if rat is None
                else _and_presence(model, rat, cim, name=f"cpres_mlp_rat_n{n.node_id}")
            )
            mlp_present = _and_presence(
                model, parked, mlp_literal, name=f"cpres_mlp_n{n.node_id}"
            )
            iv_mlp = model.NewOptionalIntervalVar(
                cancel_layer[n.node_id], 1, c_end, mlp_present, f"civ_mlp_n{n.node_id}"
            )
            mlp_cancel_intervals.append(iv_mlp)
            mlp_cancel_demands.append(2 * len(n))
        elif parked is not None:
            # Keep-forever-via-Concatenate nodes reach here only with no parked
            # var; a parked var without a cancel_in_mlp var cannot occur (both
            # are built for exactly the non-keep-forever set), but keep the
            # branch structurally for safety.
            iv = model.NewOptionalIntervalVar(
                cancel_layer[n.node_id], 1, c_end, parked.Not(), f"civ_n{n.node_id}"
            )
            cancel_intervals.append(iv)
            cancel_demands.append(len(n))
        else:
            # Keep-forever-via-Concatenate (cl == max_layers): no mechanism
            # choice; its cancel-head interval piles at the virtual layer
            # max_layers like the pinned keep-forever nodes always have.
            iv = model.NewIntervalVar(
                cancel_layer[n.node_id], 1, c_end, f"civ_n{n.node_id}"
            )
            cancel_intervals.append(iv)
            cancel_demands.append(len(n))
        # No BIRTH-layer dirty-column cancel: the runtime always
        # zero-initialises the residual stream (the ONNX embed-table
        # zero-scatter + get_input_res_stream contract), so a fresh
        # allocation's columns start clean and its first additive write
        # needs no prior cancel.  Recycled columns are cleaned by the
        # death-cancel that freed them.

    # Freeable inputs pay a DEATH-layer cancel head (their columns hold the
    # input value and must be zeroed before any downstream additive write
    # reuses them), so they consume the per-layer head budget exactly like a
    # scheduled node's death cancel.  They never need a BIRTH-dirty cancel:
    # the compiler marks input columns clean at allocation, so the first reuse
    # after the input is freed lands on already-clean columns.
    for n in freeable_inputs:
        cl_in = input_cancel_layer[n.node_id]
        c_end = model.NewIntVar(1, max_layers + 2, f"cend_in{n.node_id}")
        model.Add(c_end == cl_in + 1)
        parked = parked_input_by_id.get(n.node_id)
        rat_in = reused_as_target.get(n.node_id)
        # A selected reuse target's ownership ends through `reassign` (the
        # selector gates the physical cancel absent); a parked input never
        # cancels in-horizon.
        if parked is not None and rat_in is not None:
            literal = _and_presence(
                model, parked, rat_in.Not(), name=f"cpres_in{n.node_id}"
            )
        elif parked is not None:
            literal = parked.Not()
        elif rat_in is not None:
            literal = rat_in.Not()
        else:
            literal = None
        if literal is not None:
            iv = model.NewOptionalIntervalVar(
                cl_in, 1, c_end, literal, f"civ_in{n.node_id}"
            )
        else:
            iv = model.NewIntervalVar(cl_in, 1, c_end, f"civ_in{n.node_id}")
        cancel_intervals.append(iv)
        cancel_demands.append(len(n))

    # `reserve_heads` is a safety knob for graphs whose attention
    # heads are saturated by ops outside the model (e.g. bias writes
    # folded into deferred Linears); default 0.
    effective_capacity = max(0, n_heads_per_layer - reserve_heads) * d_head
    if "attn_cumulative" not in _disabled_families and (
        attn_intervals or cancel_intervals
    ):
        model.AddCumulative(
            attn_intervals + cancel_intervals,
            attn_demands + cancel_demands,
            effective_capacity,
        )

    # ---- MLP slots cumulative ----
    # For flex nodes, MLP demand is gated by NOT(is_attn).  An FFN carries
    # its lane slots and is pinned to MLP (is_attn == 0), so its interval is
    # always present.
    mlp_intervals: List = []
    mlp_demands: List[int] = []
    for n in gm.schedulable:
        s = slots_for(n, gm)
        if s <= 0:
            continue
        end = model.NewIntVar(1, max_layers, f"mend_n{n.node_id}")
        model.Add(end == layer_var[n.node_id] + 1)
        iv = model.NewOptionalIntervalVar(
            layer_var[n.node_id], 1, end, is_attn[n.node_id].Not(), f"miv_n{n.node_id}"
        )
        mlp_intervals.append(iv)
        mlp_demands.append(s)
    # MLP-cancel cost intervals (built in the cancel-interval loop above) share
    # the same hidden-slot budget: a `cancel_bypass` at a node's cancel layer
    # competes with FFN lanes and bypass Linears for the layer's d_hidden slots.
    if "mlp_cumulative" not in _disabled_families and (
        mlp_intervals or mlp_cancel_intervals
    ):
        model.AddCumulative(
            mlp_intervals + mlp_cancel_intervals,
            mlp_demands + mlp_cancel_demands,
            d_hidden,
        )

    # ---- Residual cumulative ----
    residual_nodes = [n for n in gm.schedulable if uses_residual(n, gm)]
    resid_intervals: List = []
    resid_demands: List[int] = []
    for n in residual_nodes:
        # Residual-occupancy end.  An attention cancel frees the columns mid-
        # attention-sublayer, so the node stops occupying at `cancel_layer`
        # (`[layer, cancel)`).  An MLP cancel (`cancel_in_mlp == 1`) fires in the
        # MLP sublayer — AFTER both sublayers' reads (decision #1) — so the
        # columns stay live through the whole cancel layer and free only at its
        # end: the node occupies `[layer, cancel + 1)`.  `end = cancel + cim`
        # captures both.  Without the +1 the model would count an MLP-cancelled
        # node's columns as free during the cancel layer's attention sublayer,
        # where the replay still holds them — an unreplayable (I4) schedule.
        cim = cancel_in_mlp.get(n.node_id)
        if cim is not None and "mlp_cancel_occupancy" not in _disabled_families:
            rend = model.NewIntVar(1, max_layers + 1, f"rend_n{n.node_id}")
            model.Add(rend == cancel_layer[n.node_id] + cim)
        else:
            # Either a keep-forever node (no cim) or the diagnostic relaxation:
            # occupy only `[layer, cancel)` (unsound to execute — lower bound).
            rend = cancel_layer[n.node_id]
        if isinstance(n, Add) and n.node_id in is_free:
            # Free-add reuses a dead addend's already-allocated residual
            # columns (`reassign` in both `LayerScheduler._schedule_attn_
            # sublayer` and the directed replay), so it adds NO fresh
            # residual column at its birth layer.  The reused addend's own
            # interval already covers that layer — `cancel_consumer_lb`
            # forces the addend's cancel >= layer[A] + 1 because A is one of
            # its consumers — so giving the Add a full interval starting at
            # layer[A] double-counts the shared column for exactly one layer.
            # Shift the Add's residual start by `is_free[A]`: a free-add
            # (is_free=1) starts one layer later (the addend covers layer[A]),
            # a compute-add (is_free=0) allocates fresh columns and starts at
            # layer[A].  This is the residual-cumulative analogue of the
            # BIRTH-dirty / attention-head free-add gating above, and removes
            # the residual over-count that rejected schedules the heuristic
            # compiles (the dead addend and its Add never occupy two distinct
            # columns at the add layer).
            start = model.NewIntVar(0, max_layers, f"rstart_n{n.node_id}")
            model.Add(start == layer_var[n.node_id] + is_free[n.node_id])
            size = model.NewIntVar(0, max_layers + 1, f"rsz_n{n.node_id}")
            model.Add(size == rend - start)
            iv = model.NewIntervalVar(start, size, rend, f"riv_n{n.node_id}")
        else:
            size = model.NewIntVar(1, max_layers + 1, f"rsz_n{n.node_id}")
            model.Add(size == rend - layer_var[n.node_id])
            iv = model.NewIntervalVar(
                layer_var[n.node_id],
                size,
                rend,
                f"riv_n{n.node_id}",
            )
        resid_intervals.append(iv)
        resid_demands.append(len(n))
    # Freeable inputs occupy residual columns from layer 0 until their cancel
    # layer.  Counting them here (instead of pre-subtracting their width from
    # `available_residual`) lets the solver reclaim their columns for
    # intermediates once they die — the whole point of the input-freeing fix.
    for n in freeable_inputs:
        cl_in = input_cancel_layer[n.node_id]
        # A held source's physical value dies at cl_in, but its columns remain
        # unavailable to ordinary allocation until target birth.  Model that
        # ownership gap by extending residual occupancy to layer[target].
        rend_in = layer_var[held_target.node_id] if n is held_source else cl_in
        iv = model.NewIntervalVar(0, rend_in, rend_in, f"riv_in{n.node_id}")
        resid_intervals.append(iv)
        resid_demands.append(len(n))
    if "residual_cumulative" not in _disabled_families and resid_intervals:
        model.AddCumulative(resid_intervals, resid_demands, available_residual)

    # ---- Aggregate counters for the objective ----
    attn_term: List = []
    mlp_bypass_term: List = []
    fixed_attn_heads = 0
    for n in gm.schedulable:
        h = heads_for(n, d_head)
        if h == 0:
            continue
        if isinstance(n, Add):
            # Route-aware Add head cost: an MLP-routed Add contributes zero
            # heads; an attention-routed one pays h (reused placement) or
            # 2h (fresh) — h per attention presence plus another h when the
            # fresh-placement presence holds.
            attn_term.append(h * is_attn[n.node_id])
            attn_term.append(h * add_attn_comp_pres[n.node_id])
        elif flex_routing and is_flex(n, gm):
            attn_term.append(h * is_attn[n.node_id])
        else:
            r = static_routing[n.node_id]
            if r == ATTN:
                fixed_attn_heads += h
    total_attn_heads = model.NewIntVar(
        0,
        fixed_attn_heads + 2 * sum(heads_for(n, d_head) for n in gm.schedulable),
        "total_attn_heads",
    )
    if attn_term:
        model.Add(total_attn_heads == fixed_attn_heads + sum(attn_term))
    else:
        model.Add(total_attn_heads == fixed_attn_heads)

    fixed_mlp_bypass = 0
    for n in gm.schedulable:
        if not is_flex(n, gm):
            continue
        if flex_routing:
            mlp_bypass_term.append((2 * n.d_output) * is_attn[n.node_id].Not())
        else:
            r = static_routing[n.node_id]
            if r == MLP:
                fixed_mlp_bypass += 2 * n.d_output
    total_mlp_bypass = model.NewIntVar(
        0,
        fixed_mlp_bypass
        + sum(2 * n.d_output for n in gm.schedulable if is_flex(n, gm)),
        "total_mlp_bypass",
    )
    if mlp_bypass_term:
        model.Add(total_mlp_bypass == fixed_mlp_bypass + sum(mlp_bypass_term))
    else:
        model.Add(total_mlp_bypass == fixed_mlp_bypass)

    # ---- Objective ----
    makespan_layer = model.NewIntVar(0, max_layers, "makespan_layer")
    if gm.schedulable:
        model.AddMaxEquality(
            makespan_layer, [layer_var[n.node_id] for n in gm.schedulable]
        )
    else:
        model.Add(makespan_layer == 0)
    n_layers_var = model.NewIntVar(0, max_layers + 1, "n_layers")
    model.Add(n_layers_var == makespan_layer + 1)

    primary = (
        costs.alpha * n_layers_var
        + costs.beta * total_attn_heads
        + costs.gamma * total_mlp_bypass
    )
    # Lexicographic secondaries (see Costs): the primary block is scaled
    # past the secondaries' maximum possible contribution so they can never
    # trade against a layer.  Bounds are tracked at var creation — the
    # Proto()/.proto accessor in the installed ortools build returns
    # corrupted memory and segfaults.
    secondary_terms = []
    max_secondary = 0
    if costs.earliness > 0:
        secondary_terms.append(costs.earliness * sum(layer_var.values()))
        max_secondary += costs.earliness * layer_var_hi_sum
    if costs.waste > 0:
        occupancy = []
        for n in gm.schedulable:
            if n.node_id in keep_forever_ids:
                continue
            occupancy.append(len(n) * (cancel_layer[n.node_id] - layer_var[n.node_id]))
            max_secondary += (
                costs.waste * len(n) * (max_layers - layer_var_lo[n.node_id])
            )
        for n in freeable_inputs:
            if n.node_id in input_keep_ids:
                continue
            occupancy.append(
                len(n)
                * (
                    layer_var[held_target.node_id]
                    if n is held_source
                    else input_cancel_layer[n.node_id]
                )
            )
            max_secondary += costs.waste * len(n) * max_layers
        secondary_terms.append(costs.waste * sum(occupancy))
    objective_scale = 1
    if secondary_terms:
        objective_scale = lexicographic_objective_scale(max_secondary)
        model.Minimize(objective_scale * primary + sum(secondary_terms))
    else:
        model.Minimize(primary)

    return BuiltModel(
        model=model,
        gm=gm,
        layer_var=layer_var,
        cancel_layer=cancel_layer,
        is_attn=is_attn,
        is_free=is_free,
        cancel_in_mlp=cancel_in_mlp,
        n_layers_var=n_layers_var,
        total_attn_heads=total_attn_heads,
        total_mlp_bypass=total_mlp_bypass,
        available_residual=available_residual,
        n_heads_per_layer=n_heads_per_layer,
        input_cancel_layer=input_cancel_layer,
        held_source_id=held_source_id,
        held_target_id=held_target_id,
        objective_scale=objective_scale,
        layer_bounds=layer_bounds,
        keep_forever_ids=frozenset(keep_forever_ids),
        input_keep_ids=frozenset(input_keep_ids),
        cancel_window_delta=(
            cancel_window_delta
            if (hint_layers is not None or hint_cancel is not None)
            else None
        ),
        eff_cancel_slack=eff_cancel_slack,
        static_routing=static_routing,
        pin_cancels=_pin_cancels,
        add_reusable=add_reusable,
        add_reuse=add_reuse,
        reused_as_target=reused_as_target,
    )


# ---------------------------------------------------------------------------
# Snapshot bridge: rebuild a stand-in GraphModel from a SchedulingProblem
# ---------------------------------------------------------------------------
#
# The builder consumes a GraphModel and reads real graph-node attributes off
# it (``isinstance`` on the graph classes, ``len(n)``, ``n.d_output`` /
# ``n.d_v`` / ``n.n_lanes``, ``n.inputs`` for Add, ``n.node_id``).  Rebuilding
# stand-in *real-class* instances (``object.__new__`` + attribute injection)
# rather than lightweight records means the builder body needs no changes: an
# ``Add`` stand-in IS an ``Add`` to ``isinstance`` and to every realization /
# demand helper, so ``build_cpsat_model_from_gm`` produces the identical proto.

_KIND_TO_CLASS = {
    "Add": Add,
    "Attn": Attn,
    "Concatenate": Concatenate,
    "Embedding": Embedding,
    "InputNode": InputNode,
    "Linear": Linear,
    "FFN": FFN,
    "LiteralValue": LiteralValue,
}


class _StandInGraph:
    """The narrow slice of ``GraphAnalyzer`` the builder + solver read off
    ``gm.graph``: ``get_all_nodes`` (validator id→node map) and
    ``get_critical_path_length`` (solver decision strategy).  Raw-consumer /
    topo-order queries are not reconstructed from a snapshot (the builder uses
    ``gm.consumers_eff``, not ``gm.graph``); they raise loudly if ever read."""

    def __init__(self, all_nodes, critical_path: Dict[int, int]):
        self._all_nodes = set(all_nodes)
        self._critical_path = critical_path

    def get_all_nodes(self):
        return self._all_nodes

    def get_critical_path_length(self, node: Node) -> int:
        return self._critical_path.get(node.node_id, 0)

    def get_output_node(self):  # pragma: no cover - parity with GraphAnalyzer
        raise NotImplementedError("stand-in graph exposes no output-node query")

    def get_consumers(self, node):  # pragma: no cover - not snapshotted
        raise NotImplementedError(
            "raw consumers are not reconstructed from a snapshot; the builder "
            "reads gm.consumers_eff instead"
        )

    def get_topological_order(self):  # pragma: no cover - not snapshotted
        raise NotImplementedError("topological order is not reconstructed")


class _ShapeCarrier:
    """Minimal stand-in for a weight tensor: exposes only ``.shape`` so
    ``FFN.n_lanes`` (``self.gate_proj.shape[0]``) returns the captured lane
    count without materializing a tensor."""

    __slots__ = ("shape",)

    def __init__(self, n_lanes: int):
        self.shape = (n_lanes,)


def _stand_in_node(rec) -> Node:
    cls = _KIND_TO_CLASS.get(rec.kind)
    if cls is None:
        raise ValueError(
            f"snapshot node {rec.node_id} has unknown kind {rec.kind!r}; the "
            f"CP-SAT builder only knows {sorted(_KIND_TO_CLASS)}"
        )
    node = object.__new__(cls)
    node.node_id = rec.node_id
    node.d_output = rec.d_output
    node.name = rec.name
    if rec.d_v is not None:
        node.d_v = rec.d_v
    if rec.n_lanes is not None:
        # FFN.n_lanes is a read-only property over gate_proj.shape[0]; feed it a
        # shape carrier so slots_for(FFN) reads the captured lane count.
        node.gate_proj = _ShapeCarrier(rec.n_lanes)
    if rec.live_row_ranges is not None:
        # Stand-ins carry no output_matrix; injecting the captured runs into
        # the cache attribute makes realization.live_weight_row_ranges — and
        # so heads_for — read the identical value it read on the live node.
        node._live_weight_row_ranges = rec.live_row_ranges
    return node


def graph_model_from_problem(problem: SchedulingProblem) -> GraphModel:
    """Rebuild a stand-in ``GraphModel`` from a captured problem.

    The nodes are real graph-class instances carrying only the attributes the
    builder reads; ``consumers_eff`` values are ordered tuples (the builder
    only iterates them) so the constraint order — and thus the proto — matches
    the capturing process byte-for-byte.
    """
    by_id: Dict[int, Node] = {
        rec.node_id: _stand_in_node(rec) for rec in problem.nodes.values()
    }
    for rec in problem.nodes.values():
        node = by_id[rec.node_id]
        node.inputs = [by_id[i] for i in rec.input_ids]
        # Pre-support-charge snapshots carry no live_row_ranges; fall back to
        # the dense-equivalent single run (== the old width-derived charge).
        # Such snapshots fail the identity fingerprint gate before any solve
        # is trusted, so this is determinism hygiene, not a supported path.
        if isinstance(node, Linear) and rec.live_row_ranges is None:
            node._live_weight_row_ranges = (
                ((0, len(node.inputs[0])),) if node.inputs else ((0, 0),)
            )

    consumers_eff = {
        by_id[nid]: tuple(by_id[c] for c in cons)
        for nid, cons in problem.consumers.items()
    }
    graph = _StandInGraph(
        by_id.values(),
        {rec.node_id: rec.critical_path_len for rec in problem.nodes.values()},
    )
    return GraphModel(
        graph=graph,
        schedulable=[by_id[i] for i in problem.schedulable_ids],
        edges=[(by_id[u], by_id[v]) for u, v in problem.edges],
        consumers_eff=consumers_eff,
        output_node=by_id[problem.output_id],
        pos_encoding=None,
        input_nodes=[by_id[i] for i in problem.input_ids],
        pinned_nodes={by_id[i] for i in problem.pinned_ids},
    )


def build_model_from_snapshot(
    problem: SchedulingProblem,
    *,
    d: int,
    d_head: int,
    n_heads: Optional[int] = None,
    d_hidden: int,
    costs: Costs = Costs(),
    flex_routing: bool = True,
    max_layers: int = 60,
    cancel_slack: Optional[int] = 2,
    policy: Optional[SchedulingPolicy] = None,
    reserve_heads: int = 0,
    reserve_residual: int = 0,
    tighten_domains: bool = False,
    diagnostic_hint: Optional[DiagnosticHint] = None,
    held_source_id: Optional[int] = None,
    held_target_id: Optional[int] = None,
    _disabled_families: frozenset = frozenset(),
    _canonical_cancel_reps: bool = False,
    _pin_cancels: bool = True,
) -> BuiltModel:
    """Build the CP-SAT model from a captured snapshot — no live graph.

    Equivalent to :func:`build_cpsat_model` on the graph the snapshot was
    captured from, at the same geometry; the two protos are identical.

    The held-bank endpoints default to the snapshot's own stored contract
    (``problem.held_source_id`` / ``problem.held_target_id``).  Explicit
    kwargs equal to the stored pair are allowed (idempotent re-supply);
    kwargs that disagree with a stored contract raise — a snapshot solved
    under a different held contract than it captured is a wrong-problem
    measure.  Kwargs against a held-less snapshot are honoured (a capture
    site that has not yet learned to store the contract).
    """
    if held_source_id is None and held_target_id is None:
        held_source_id = problem.held_source_id
        held_target_id = problem.held_target_id
    elif problem.held_source_id is not None or problem.held_target_id is not None:
        if (held_source_id, held_target_id) != (
            problem.held_source_id,
            problem.held_target_id,
        ):
            raise ValueError(
                f"held endpoints ({held_source_id}, {held_target_id}) "
                f"conflict with the snapshot's stored held contract "
                f"({problem.held_source_id}, {problem.held_target_id}); "
                f"drop the kwargs to use the snapshot's values"
            )
    return build_cpsat_model_from_gm(
        graph_model_from_problem(problem),
        d=d,
        d_head=d_head,
        n_heads=n_heads,
        d_hidden=d_hidden,
        costs=costs,
        flex_routing=flex_routing,
        max_layers=max_layers,
        cancel_slack=cancel_slack,
        policy=policy,
        reserve_heads=reserve_heads,
        reserve_residual=reserve_residual,
        tighten_domains=tighten_domains,
        diagnostic_hint=diagnostic_hint,
        held_source_id=held_source_id,
        held_target_id=held_target_id,
        _disabled_families=_disabled_families,
        _canonical_cancel_reps=_canonical_cancel_reps,
        _pin_cancels=_pin_cancels,
    )


# ---------------------------------------------------------------------------
# CpModelProto dump + zero-rebuild re-solve
# ---------------------------------------------------------------------------


def dump_model_proto(built: BuiltModel, path) -> "os.PathLike":
    """Serialize a built CP-SAT model to disk for a zero-rebuild re-solve.

    Writes the OR-Tools model proto (text format — the installed pybind build
    exposes no binary ``SerializeToString`` on the proto) to ``path`` and a
    small ``<path>.meta.json`` recording the objective scale and the
    ``n_layers`` variable's proto index, so :func:`resolve_model_proto` reads
    the depth straight back with no graph and no model rebuild (C1's zero-build
    re-solve).
    """
    import json as _json
    from pathlib import Path as _Path

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(built.model.Proto()), encoding="utf-8")
    meta = {
        "objective_scale": built.objective_scale,
        "n_layers_var_index": built.n_layers_var.Index(),
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        _json.dumps(meta), encoding="utf-8"
    )
    return path


def resolve_model_proto(
    path,
    *,
    time_budget_s: float = 60.0,
    workers: Optional[int] = None,
    solver_params: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Load a dumped model proto and re-solve it — no graph, no model rebuild.

    Returns ``{status_name, objective, best_bound, wall_time_s, n_layers}``;
    ``n_layers`` (the depth) is recovered via the proto index stored by
    :func:`dump_model_proto`.
    """
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    path = _Path(path)
    text = path.read_text(encoding="utf-8")
    meta = _json.loads(
        path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8")
    )
    model = cp_model.CpModel()
    if not model.Proto().parse_text_format(text):
        raise ValueError(f"failed to parse CP-SAT proto from {path}")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_budget_s
    solver.parameters.num_search_workers = int(
        workers if workers is not None else os.environ.get("TW_CPSAT_WORKERS", "16")
    )
    if solver_params:
        for key, value in solver_params.items():
            if isinstance(value, (list, tuple)):
                getattr(solver.parameters, key).extend(value)
            else:
                setattr(solver.parameters, key, value)
    t0 = _time.perf_counter()
    status = solver.Solve(model)
    elapsed = _time.perf_counter() - t0
    has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    n_layers = None
    if has_solution:
        nlv = model.GetIntVarFromProtoIndex(int(meta["n_layers_var_index"]))
        n_layers = solver.Value(nlv)
    return {
        "status_name": solver.StatusName(status),
        "objective": int(solver.ObjectiveValue()) if has_solution else -1,
        "best_bound": float(solver.BestObjectiveBound()),
        "wall_time_s": elapsed,
        "n_layers": n_layers,
    }


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class _IncumbentTrace(cp_model.CpSolverSolutionCallback):
    """Records (time, objective, n_layers, full assignment) per incumbent.

    Used by ``solve_schedule(solution_trace=[...])``.  Each improving
    solution appends one snapshot dict; reading ~20k variable values per
    incumbent costs microseconds each and only runs in capture mode.
    """

    def __init__(self, layer_var, cancel_layer, input_cancel_layer, n_layers_var, out):
        super().__init__()
        self._layer_var = layer_var
        self._cancel_layer = cancel_layer
        self._input_cancel_layer = input_cancel_layer
        self._n_layers_var = n_layers_var
        self._out = out

    def on_solution_callback(self):
        self._out.append(
            {
                "t": self.WallTime(),
                "objective": int(self.ObjectiveValue()),
                "n_layers": self.Value(self._n_layers_var),
                "layers": {nid: self.Value(v) for nid, v in self._layer_var.items()},
                "cancels": {
                    nid: self.Value(v) for nid, v in self._cancel_layer.items()
                },
                "input_cancels": {
                    nid: self.Value(v) for nid, v in self._input_cancel_layer.items()
                },
            }
        )


def _validate_hint(
    built: BuiltModel,
    hint: DiagnosticHint,
    *,
    max_layers: int,
    strict: bool = False,
) -> List[str]:
    """Cross-check a warm-start hint against the built model — the tripwire.

    The hint-application loop in :func:`solve_schedule` guards every
    ``AddHint`` with ``if nid in vars and in-range`` — a hint that fails the
    guard silently vanishes, and a hint that passes the guard but violates a
    model constraint is silently discarded by CP-SAT at solve time (it
    cold-searches instead).  Both failure classes were invisible twice: the
    June 2026 eager-free infeasible hint and the July 2026 deferred-cancel
    infeasible hint (the silent optimize=2 fallback).  This validator mirrors
    every hint-checkable model constraint and reports what the model would
    reject:

    - layer hints inside the tightened domains (when the model has them);
    - guard-dropped hints for non-``Concatenate`` nodes (``Concatenate``
      drops are expected — the warm start records them but the model has no
      vars for them);
    - cancel >= hinted birth + 1 and >= each hinted consumer's layer + 1;
    - keep-forever nodes: cancel hint must be ``max_layers`` or absent;
    - the (widened) cancel windows.  Post-widening this check should never
      fire; firing means a NEW class of hint infeasibility has appeared —
      investigate before trusting optimize>0 results.

    Returns the violation list.  ``strict=True`` raises ``ValueError``
    naming the first violations; the default emits one ``RuntimeWarning``
    (production keeps its fall-back-don't-fail contract).
    """
    hint_layers = hint.layers
    hint_routing = hint.routing
    hint_cancel = hint.cancel
    hint_cancel_mech = hint.cancel_mech
    violations: List[str] = []
    gm = built.gm
    all_nodes: Dict[int, Node] = {n.node_id: n for n in gm.graph.get_all_nodes()}
    input_ids = {n.node_id for n in gm.input_nodes}

    def _desc(nid: int) -> str:
        n = all_nodes.get(nid)
        if n is None:
            return f"id={nid} <not in graph>"
        return f"id={nid} {type(n).__name__} name={getattr(n, 'name', None)!r}"

    def _expected_drop(nid: int) -> bool:
        n = all_nodes.get(nid)
        return isinstance(n, Concatenate)

    K = built.eff_cancel_slack
    deltas = built.cancel_window_delta or {}

    for nid, L in (hint_layers or {}).items():
        if nid in built.layer_var:
            if not (0 <= L < max_layers):
                violations.append(
                    f"layer hint out of range (guard-dropped): "
                    f"{_desc(nid)} hint={L}"
                )
            elif built.layer_bounds is not None:
                lo, hi = built.layer_bounds[nid]
                if not (lo <= L <= hi):
                    violations.append(
                        f"layer hint outside tightened domain [{lo},{hi}]: "
                        f"{_desc(nid)} hint={L}"
                    )
        elif not _expected_drop(nid) and nid not in input_ids:
            violations.append(
                f"layer hint for node with no layer var (guard-dropped): "
                f"{_desc(nid)} hint={L}"
            )

    for nid, route in (hint_routing or {}).items():
        if route not in (ATTN, MLP):
            violations.append(
                f"routing hint with unknown route {route!r}: {_desc(nid)}"
            )
        if nid not in built.is_attn and not _expected_drop(nid):
            violations.append(
                f"routing hint for node with no routing var (guard-dropped): "
                f"{_desc(nid)} hint={route}"
            )

    def _hinted_add_placement(A: Add):
        """Derive A's placement from the LAYER/ROUTING hints through the one
        assignment-level definition (`add_placement.derive_add_placement` —
        the same rule the model's per-occurrence literals reify; keep them
        in sync).  Returns None (check leniently) when a needed hint is
        missing.  The held target short-circuits inside the derivation
        (is_free pinned 0, never derived).
        """
        if hint_layers is None or A.node_id not in hint_layers:
            return None
        routing_map = dict(hint_routing or {})
        if routing_map.get(A.node_id) is None:
            # Layers-only diagnostic hints: assume the attention-route
            # (strict-prior) predicate — exact for attention-routed Adds
            # and conservative (never over-reuses) otherwise.  Consumers
            # with missing route hints block same-layer completion inside
            # the derivation for the same reason.
            routing_map[A.node_id] = ATTN
        try:
            return derive_add_placement(
                A,
                effective_consumers=lambda x: gm.consumers_eff.get(x, ()),
                node_to_layer=hint_layers,
                node_to_routing=routing_map,
                held_source_id=built.held_source_id,
                held_target_id=built.held_target_id,
            )
        except ValueError:
            return None

    def _hinted_add_placement_complete(A: Add) -> bool:
        """True when every hint the derivation read actually existed — a
        missing consumer layer/route makes the derivation conservative
        (not reusable), which must not be mistaken for a derived fresh."""
        if hint_layers is None or A.node_id not in hint_layers:
            return False
        if (hint_routing or {}).get(A.node_id) is None:
            return False
        for E in A.inputs:
            for c in gm.consumers_eff.get(E, set()):
                if c is A or isinstance(c, Concatenate):
                    continue
                if c.node_id not in built.layer_var:
                    continue
                if (hint_layers or {}).get(c.node_id) is None:
                    return False
                if (hint_routing or {}).get(c.node_id) is None:
                    return False
        return True

    # The selected reuse target per the hints: target nid -> (add, placement).
    hinted_target_of: Dict[int, Tuple[Add, object]] = {}
    for _A in gm.schedulable:
        if not isinstance(_A, Add):
            continue
        _p = _hinted_add_placement(_A)
        if (
            _p is not None
            and _p.reuse_input_index is not None
            and _hinted_add_placement_complete(_A)
        ):
            hinted_target_of[_A.inputs[_p.reuse_input_index].node_id] = (_A, _p)

    for nid, L in (hint_cancel or {}).items():
        in_sched = nid in built.cancel_layer
        in_input = nid in built.input_cancel_layer
        if not in_sched and not in_input:
            if not _expected_drop(nid):
                violations.append(
                    f"cancel hint for node with no cancel var (guard-dropped): "
                    f"{_desc(nid)} hint={L}"
                )
            continue
        if not (0 <= L <= max_layers):
            violations.append(
                f"cancel hint out of range (guard-dropped): {_desc(nid)} hint={L}"
            )
            continue
        keep = nid in (built.keep_forever_ids if in_sched else built.input_keep_ids)
        if keep:
            if L != max_layers:
                violations.append(
                    f"cancel hint below max_layers={max_layers} for "
                    f"keep-forever node: {_desc(nid)} hint={L}"
                )
            continue
        birth = (hint_layers or {}).get(nid) if in_sched else 0
        min_lifetime = 0 if nid == built.held_source_id else 1
        if birth is not None and L < birth + min_lifetime:
            violations.append(
                f"cancel hint before birth+{min_lifetime}: {_desc(nid)} "
                f"cancel={L} birth={birth}"
            )
        node = all_nodes.get(nid)
        mech = (hint_cancel_mech or {}).get(nid, ATTN)
        selected = hinted_target_of.get(nid)
        if selected is not None:
            # A selected reuse target's lifetime ends through the reassign
            # handoff: its canonical virtual cancel is layer[A] + 1 with a
            # non-MLP mechanism (bookkeeping only — the physical cancel is
            # gated absent).  A different hint contradicts the model's
            # selector constraints and would sink the incumbent.
            sel_add, _sel_p = selected
            sel_layer = (hint_layers or {}).get(sel_add.node_id)
            if sel_layer is not None and L != sel_layer + 1:
                violations.append(
                    f"selected reuse target's cancel hint is not the "
                    f"virtual handoff layer[A]+1={sel_layer + 1}: "
                    f"{_desc(nid)} cancel={L} (Add {_desc(sel_add.node_id)})"
                )
            if (hint_cancel_mech or {}).get(nid) == MLP:
                violations.append(
                    f"selected reuse target hinted an MLP cancel mechanism "
                    f"(a reassigned target has no physical cancel; its "
                    f"canonical mechanism is attention): {_desc(nid)}"
                )
            continue
        if in_input and mech == MLP:
            # Inputs have no MLP cancel mechanism in the model (no
            # `cancel_in_mlp` var is created for freeable inputs), so an
            # MLP-mech hint on an input can only come from a heuristic
            # emitting a schedule the model cannot represent.  Flag it, then
            # check the gaps as the attention mechanism the model assumes.
            violations.append(
                f"MLP cancel mechanism hinted for an input (inputs are "
                f"always attention-cancelled in the model): {_desc(nid)}"
            )
            mech = ATTN
        hinted_cons: List[int] = []
        all_cons_hinted = True
        for c in gm.consumers_eff.get(node, set()):
            if c.node_id not in built.layer_var:
                continue
            c_hint = (hint_layers or {}).get(c.node_id)
            if c_hint is None:
                all_cons_hinted = False
                continue
            hinted_cons.append(c_hint)
            # Mirror the mechanism-conditional cancel bound.  An MLP cancel
            # fires after both sublayers' reads, so it permits gap 0 for every
            # non-Add consumer (attn or mlp).  An attention cancel keeps the
            # routing-aware gap: an attention-routed consumer permits a
            # same-layer cancel (gap 0); an MLP-routed consumer keeps the
            # layer-after bound (gap 1).  Routing comes from the hint when
            # present, else the model's pinned routing; flex consumers without
            # a routing hint are checked leniently (gap 0).  An Add consumer's
            # bound mirrors the mechanism-specific model terms: the attention
            # term is `layer + (is_free OR MLP-routed)` (reuse conservatism —
            # gap #2 — or the MLP-routed Add's post-cancel read), the MLP term
            # is `layer + is_free`.  Placement derives from the layer/route
            # hints (`_hinted_add_placement`), lenient (gap 0) when a needed
            # hint is missing.  Freeable non-held inputs keep the model's
            # uniform gap-1 bound for every consumer.
            if in_input and nid != built.held_source_id:
                gap = 1
            elif isinstance(c, Add):
                placement = _hinted_add_placement(c)
                add_free = placement.is_free if placement is not None else False
                if mech == MLP:
                    gap = 1 if add_free else 0
                else:
                    c_route = (hint_routing or {}).get(c.node_id)
                    gap = 1 if (add_free or c_route == MLP) else 0
            elif mech == MLP:
                gap = 0
            else:
                route = (hint_routing or {}).get(c.node_id)
                if route is None and built.static_routing is not None:
                    route = built.static_routing.get(c.node_id)
                gap = 1 if route == MLP else 0
            if L < c_hint + gap:
                violations.append(
                    f"cancel hint before consumer's layer+{gap}: {_desc(nid)} "
                    f"cancel={L}, consumer {_desc(c.node_id)} layer={c_hint}"
                )
        if K is not None and all_cons_hinted:
            base = max(hinted_cons) if hinted_cons else birth
            if base is not None:
                ub = base + 1 + K + deltas.get(nid, 0)
                if L > ub:
                    violations.append(
                        f"cancel hint outside the (widened) window: "
                        f"{_desc(nid)} cancel={L} > ub={ub} (base={base}, "
                        f"K={K}, delta={deltas.get(nid, 0)}) — post-widening "
                        f"this should be impossible; a new class of hint "
                        f"infeasibility has appeared"
                    )

    if violations:
        shown = "; ".join(violations[:5])
        more = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
        msg = (
            f"warm-start hint validation found {len(violations)} "
            f"violation(s) — CP-SAT will silently drop the affected hints "
            f"(or the whole incumbent): {shown}{more}"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return violations


def solve_schedule(
    output_node: Node,
    pos_encoding=None,
    *,
    d: int,
    d_head: int,
    n_heads: Optional[int] = None,
    d_hidden: int,
    costs: Costs = Costs(),
    flex_routing: bool = True,
    time_budget_s: float = 60.0,
    max_layers: int = 60,
    incumbent: Optional[ScheduleAssignment] = None,
    held_source_id: Optional[int] = None,
    held_target_id: Optional[int] = None,
    cancel_slack: Optional[int] = 2,
    policy: Optional[SchedulingPolicy] = None,
    log_search_progress: bool = False,
    reserve_heads: int = 0,
    reserve_residual: int = 0,
    tighten_domains: bool = False,
    solver_params: Optional[Dict[str, object]] = None,
    solution_trace: Optional[List[dict]] = None,
    strict_hint: bool = False,
    drop_decision_strategy: bool = False,
    _disabled_families: frozenset = frozenset(),
    _canonical_cancel_reps: bool = False,
    _pin_cancels: bool = True,
    _diagnostic_hint: Optional[DiagnosticHint] = None,
) -> Tuple[Optional[ScheduleAssignment], SolveStats]:
    """Build and solve the CP-SAT scheduling model.

    Returns ``(assignment, stats)``.  ``assignment`` is ``None`` when
    the solver found no feasible solution within the budget; check
    ``stats.is_optimal`` to distinguish proven-optimal from
    feasible-only.  Callers decide what to do with non-optimal /
    no-solution outcomes (the forward compiler falls back to the
    heuristic; the probe script just reports them).

    Args:
        output_node: graph output. Defines the ancestor cone the
            scheduler operates over.
        pos_encoding: vestigial — always ``None`` under RoPE; retained for
            call-site compatibility.
        d, d_head, d_hidden: transformer geometry. Residual budget is
            ``d - input_residual_cols``.
        n_heads: Per-layer attention-head capacity. Defaults to
            ``d // d_head``.
        costs: objective weights. See :class:`Costs`.
        flex_routing: if True, CP-SAT picks attention vs MLP for each
            standalone ``Linear``.  If False, standalone Linears use
            the static routing dictated by ``policy.local_in_attention``.
        time_budget_s: per-solve wall-clock cap.
        max_layers: search horizon.  Should be at least the heuristic's
            layer count.
        incumbent: complete semantic assignment used as the production warm
            start. Tests and measurement tooling that require a partial or
            intentionally invalid hint use the private ``_diagnostic_hint``
            seam with one :class:`DiagnosticHint` value.
        cancel_slack: when not None, restrict each non-pinned node's
            cancel layer to ``[earliest_dead, earliest_dead + K]``
            where ``earliest_dead = max(layer[c] + 1)`` over consumers
            and ``K == cancel_slack``.  Cuts the cancel-decision
            search space ~30x at K=2 with negligible loss of
            optimality (the heuristic almost always cancels within
            1–2 layers of the last consumer).  Set to None to keep
            the wide ``[layer[n]+1, max_layers]`` domain.  Default 2.
        policy: only consulted when ``flex_routing=False``.  Defaults
            to ``LEGACY_POLICY``.
        log_search_progress: if True, the solver's progress log is
            forwarded line-by-line and accumulated in
            ``stats.solver_log``.
        reserve_heads: per-layer attention-head budget reserved
            beyond the modeled compute + cancel + dirty terms.
            Defaults to 0.  Raise it for graphs whose attention heads
            are saturated by ops outside the model.
        reserve_residual: residual columns permanently removed from the
            free pool before scheduling and therefore unavailable to the
            solver (the pinned-constant RMSNorm reserves 1–2; see
            ``forward_compile``).  Subtracted from the residual budget so
            the modeled capacity matches the reservation-reduced replay
            pool.  Defaults to 0.
        strict_hint: if True, a hint the model would drop or reject
            raises ``ValueError`` (see :func:`_validate_hint`).  Default
            False emits one ``RuntimeWarning`` instead — production
            keeps its fall-back-don't-fail contract; tests use strict.

    Raises ``RuntimeError`` only on structural problems (no residual
    columns left after pre-allocated inputs).  Solver-outcome
    handling (no-incumbent, FEASIBLE-not-OPTIMAL) is the caller's
    responsibility.
    """
    if incumbent is not None and _diagnostic_hint is not None:
        raise ValueError("pass either incumbent or _diagnostic_hint, not both")
    if incumbent is not None:
        incumbent.validate(output_node)
        hint = DiagnosticHint.from_assignment(incumbent)
    else:
        hint = _diagnostic_hint

    built = build_cpsat_model(
        output_node,
        pos_encoding,
        d=d,
        d_head=d_head,
        n_heads=n_heads,
        d_hidden=d_hidden,
        costs=costs,
        flex_routing=flex_routing,
        max_layers=max_layers,
        cancel_slack=cancel_slack,
        policy=policy,
        reserve_heads=reserve_heads,
        reserve_residual=reserve_residual,
        tighten_domains=tighten_domains,
        diagnostic_hint=hint,
        held_source_id=held_source_id,
        held_target_id=held_target_id,
        _disabled_families=_disabled_families,
        _canonical_cancel_reps=_canonical_cancel_reps,
        _pin_cancels=_pin_cancels,
    )
    return _solve_built(
        built,
        hint=hint,
        max_layers=max_layers,
        time_budget_s=time_budget_s,
        log_search_progress=log_search_progress,
        solver_params=solver_params,
        solution_trace=solution_trace,
        strict_hint=strict_hint,
        drop_decision_strategy=drop_decision_strategy,
    )


def solve_schedule_from_snapshot(
    problem: SchedulingProblem,
    *,
    d: int,
    d_head: int,
    n_heads: Optional[int] = None,
    d_hidden: int,
    costs: Costs = Costs(),
    flex_routing: bool = True,
    time_budget_s: float = 60.0,
    max_layers: int = 60,
    incumbent: Optional[ScheduleAssignment] = None,
    held_source_id: Optional[int] = None,
    held_target_id: Optional[int] = None,
    cancel_slack: Optional[int] = 2,
    policy: Optional[SchedulingPolicy] = None,
    log_search_progress: bool = False,
    reserve_heads: int = 0,
    reserve_residual: int = 0,
    tighten_domains: bool = False,
    solver_params: Optional[Dict[str, object]] = None,
    solution_trace: Optional[List[dict]] = None,
    strict_hint: bool = False,
    drop_decision_strategy: bool = False,
    _disabled_families: frozenset = frozenset(),
    _canonical_cancel_reps: bool = False,
    _pin_cancels: bool = True,
    _diagnostic_hint: Optional[DiagnosticHint] = None,
) -> Tuple[Optional[ScheduleAssignment], SolveStats]:
    """Build and solve the model from a captured snapshot — no live graph.

    The zero-build re-solve path (C1 parameter sweeps, C2 probes): rebuild the
    identical CP-SAT model from a fixture and solve it.  Returns the same
    ``(assignment, stats)`` as :func:`solve_schedule`; the assignment is keyed
    by the snapshot's node-id space (canonical ids for a loaded fixture).
    """
    if incumbent is not None and _diagnostic_hint is not None:
        raise ValueError("pass either incumbent or _diagnostic_hint, not both")
    hint = (
        DiagnosticHint.from_assignment(incumbent)
        if incumbent is not None
        else _diagnostic_hint
    )
    built = build_model_from_snapshot(
        problem,
        d=d,
        d_head=d_head,
        n_heads=n_heads,
        d_hidden=d_hidden,
        costs=costs,
        flex_routing=flex_routing,
        max_layers=max_layers,
        cancel_slack=cancel_slack,
        policy=policy,
        reserve_heads=reserve_heads,
        reserve_residual=reserve_residual,
        tighten_domains=tighten_domains,
        diagnostic_hint=hint,
        held_source_id=held_source_id,
        held_target_id=held_target_id,
        _disabled_families=_disabled_families,
        _canonical_cancel_reps=_canonical_cancel_reps,
        _pin_cancels=_pin_cancels,
    )
    return _solve_built(
        built,
        hint=hint,
        max_layers=max_layers,
        time_budget_s=time_budget_s,
        log_search_progress=log_search_progress,
        solver_params=solver_params,
        solution_trace=solution_trace,
        strict_hint=strict_hint,
        drop_decision_strategy=drop_decision_strategy,
    )


def _solve_built(
    built: BuiltModel,
    *,
    hint: Optional[DiagnosticHint] = None,
    max_layers: int = 60,
    time_budget_s: float = 60.0,
    log_search_progress: bool = False,
    solver_params: Optional[Dict[str, object]] = None,
    solution_trace: Optional[List[dict]] = None,
    strict_hint: bool = False,
    drop_decision_strategy: bool = False,
) -> Tuple[Optional[ScheduleAssignment], SolveStats]:
    """Validate + apply the warm-start hint, set the decision strategy, solve,
    and read the assignment back off a pre-built model.  Shared by
    :func:`solve_schedule` (live) and :func:`solve_schedule_from_snapshot`.

    ``drop_decision_strategy`` is MEASUREMENT-ONLY (C1 sweep, arm
    ``no_decision_strategy``): when True the hand-rolled critical-path-first
    ``AddDecisionStrategy`` below is not emitted, freeing CP-SAT's fixed-search
    subsolver slot for its default portfolio.  It cannot change the feasible
    set — only which schedule the search finds first — so it is never set in
    production."""
    if built.pin_cancels:
        # The pinned model (`_pin_cancels`) equality-pins every cancel layer,
        # so a captured cancel-layer hint would almost always contradict the
        # pin and CP-SAT would silently discard the whole incumbent.  Drop it
        # (once layers, routings, and `cancel_in_mlp` are decided the cancels
        # are forced by propagation), which also keeps `_validate_hint` off
        # that family.  The layer, routing, and cancel-MECHANISM hints are
        # kept: `cancel_in_mlp` stays a free decision under the pin, and on a
        # head-saturated graph its hint carries exactly the packing choice —
        # which cancels take the same-layer MLP tier — that makes the warm
        # start completable.  Measured 2026-07-09 on the d=8192 fixture:
        # with the mechanism hint also dropped, no seed completed the hint
        # into ANY incumbent in 600 s (0/5), vs first incumbents at 83-104 s
        # for the unpinned control.
        if hint is not None:
            hint = hint.without_cancel_layers()
    if log_search_progress and built.cancel_window_delta:
        print(
            f"  cancel windows widened for {len(built.cancel_window_delta)} "
            f"nodes (max +{max(built.cancel_window_delta.values())})"
        )
    model = built.model
    gm = built.gm
    layer_var = built.layer_var
    cancel_layer = built.cancel_layer
    is_attn = built.is_attn
    cancel_in_mlp = built.cancel_in_mlp
    n_layers_var = built.n_layers_var
    total_attn_heads = built.total_attn_heads
    total_mlp_bypass = built.total_mlp_bypass
    input_cancel_layer = built.input_cancel_layer

    # ---- Hint ----
    # A complete hint (layer + routing + cancel) gives CP-SAT a
    # full feasible incumbent it can verify and improve from, which
    # is much faster than reconstructing routing and cancel timing
    # from a layer-only hint.  Hints are soft — CP-SAT is free to
    # discard them and explore alternatives — which is exactly why a
    # bad hint is validated loudly here instead of vanishing behind
    # the `if nid in ...` guards below.
    if hint is not None:
        _validate_hint(
            built,
            hint,
            max_layers=max_layers,
            strict=strict_hint,
        )
    if hint is not None:
        for nid, L in hint.layers.items():
            if nid in layer_var and 0 <= L < max_layers:
                model.AddHint(layer_var[nid], L)
        for nid, route in hint.routing.items():
            if nid in is_attn:
                model.AddHint(is_attn[nid], 1 if route == ATTN else 0)
        for nid, mech in hint.cancel_mech.items():
            if nid in cancel_in_mlp:
                model.AddHint(cancel_in_mlp[nid], 1 if mech == MLP else 0)
        for nid, L in hint.cancel.items():
            if nid in cancel_layer and 0 <= L <= max_layers:
                model.AddHint(cancel_layer[nid], L)
            elif nid in input_cancel_layer and 0 <= L <= max_layers:
                # The warm-start's tracking residual map records `free()`
                # for input nodes too, so route their captured cancel layer
                # to the input cancel vars — without this the input cancels
                # are unhinted and CP-SAT cannot accept the heuristic
                # schedule as a ready-made feasible incumbent.
                model.AddHint(input_cancel_layer[nid], L)

    # ---- Decision strategy: schedule by critical path first ----
    # MEASUREMENT-ONLY: dropping it (C1 arm ``no_decision_strategy``) leaves
    # the fixed-search subsolver slot to CP-SAT's default portfolio.
    if not drop_decision_strategy:
        nodes_by_cp = sorted(
            gm.schedulable,
            key=lambda n: -gm.graph.get_critical_path_length(n),
        )
        model.AddDecisionStrategy(
            [layer_var[n.node_id] for n in nodes_by_cp],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE,
        )

    # ---- Solve ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_budget_s
    solver.parameters.log_search_progress = log_search_progress
    # TW_CPSAT_WORKERS lets high-core environments (e.g. a 64-CPU Modal
    # compile container) raise the parallel-search width; 16 matches the
    # historical hardcoded value for local runs.
    solver.parameters.num_search_workers = int(os.environ.get("TW_CPSAT_WORKERS", "16"))
    # Experiment escape hatch: apply arbitrary CpSolver parameter overrides
    # (list values extend repeated fields, e.g. ``ignore_subsolvers``).
    if solver_params:
        for key, value in solver_params.items():
            if isinstance(value, (list, tuple)):
                getattr(solver.parameters, key).extend(value)
            else:
                setattr(solver.parameters, key, value)

    log_buf: List[str] = []

    def _log(msg: str) -> None:
        log_buf.append(msg)

    solver.log_callback = _log

    t0 = time.perf_counter()
    if solution_trace is not None:
        # Record every improving incumbent's full assignment for offline
        # trajectory analysis (which schedule metrics lead layer drops).
        # Snapshots fire on objective improvements only — with a pure-depth
        # objective that means layer drops; a dense secondary (earliness /
        # waste) is what surfaces intra-plateau states here.
        callback = _IncumbentTrace(
            layer_var,
            cancel_layer,
            input_cancel_layer,
            n_layers_var,
            solution_trace,
        )
        status = solver.Solve(model, callback)
    else:
        status = solver.Solve(model)
    elapsed = time.perf_counter() - t0

    has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if has_solution:
        node_to_layer: Dict[int, int] = {}
        node_to_cancel_layer: Dict[int, int] = {}
        node_to_routing: Dict[int, str] = {}
        node_to_cancel_mech: Dict[int, str] = {}
        for n in gm.schedulable:
            node_to_layer[n.node_id] = solver.Value(layer_var[n.node_id])
            node_to_cancel_layer[n.node_id] = solver.Value(cancel_layer[n.node_id])
            node_to_routing[n.node_id] = (
                ATTN if solver.Value(is_attn[n.node_id]) else MLP
            )
            cim = cancel_in_mlp.get(n.node_id)
            if cim is not None:
                node_to_cancel_mech[n.node_id] = MLP if solver.Value(cim) else ATTN
        # Freeable inputs get a cancel layer (but no layer/routing — they are
        # pre-computed at layer 0).  ``DirectedLayerScheduler._find_dead_nodes``
        # frees any allocated node whose cancel layer matches the current
        # layer, so adding inputs here makes the replay reclaim their columns.
        for nid, cl_in in input_cancel_layer.items():
            node_to_cancel_layer[nid] = solver.Value(cl_in)
        # Extraction-time Add placement tripwire
        # (docs/plan_additional_mlp_routing.md): recompute every Add's
        # per-occurrence reusability and deterministic selector from the
        # extracted layers/routes and assert they equal the solver's
        # literals WHILE those literals still exist — this is the last
        # moment a bad reification is distinguishable from a bad replay
        # liveness calculation.
        selected_target_ids: Set[int] = set()
        for A in gm.schedulable:
            if not isinstance(A, Add):
                continue
            derived = derive_add_placement(
                A,
                effective_consumers=lambda x: gm.consumers_eff.get(x, ()),
                node_to_layer=node_to_layer,
                node_to_routing=node_to_routing,
                held_source_id=built.held_source_id,
                held_target_id=built.held_target_id,
            )
            solver_free = bool(solver.Value(built.is_free[A.node_id]))
            mismatches = []
            if solver_free != derived.is_free:
                mismatches.append(
                    f"is_free solver={solver_free} derived={derived.is_free}"
                )
            for i, want in ((0, derived.reusable_0), (1, derived.reusable_1)):
                lit = built.add_reusable.get((A.node_id, i))
                if lit is not None and bool(solver.Value(lit)) != want:
                    mismatches.append(
                        f"reusable_{i} solver={bool(solver.Value(lit))} "
                        f"derived={want}"
                    )
            for i in (0, 1):
                lit = built.add_reuse.get((A.node_id, i))
                if lit is not None and bool(solver.Value(lit)) != (
                    derived.reuse_input_index == i
                ):
                    mismatches.append(
                        f"reuse_{i} solver={bool(solver.Value(lit))} "
                        f"derived={derived.reuse_input_index == i}"
                    )
            if mismatches:
                raise AssertionError(
                    f"CP-SAT Add placement extraction tripwire for {A!r} "
                    f"(layer {node_to_layer[A.node_id]}, route "
                    f"{node_to_routing[A.node_id]!r}): {'; '.join(mismatches)} "
                    f"(derived reusable_0={derived.reusable_0}, "
                    f"reusable_1={derived.reusable_1}, "
                    f"occurrence={derived.reuse_input_index}) — a bad "
                    f"reification in the model, caught before the literals "
                    f"are discarded."
                )
            if derived.reuse_input_index is not None:
                selected_target_ids.add(A.inputs[derived.reuse_input_index].node_id)
        # Count the parked (never-freed-in-horizon) nodes: non-keep-forever
        # nodes whose cancel landed at the virtual horizon.  Keyed off the
        # returned assignment so it holds under the sym1 knob and off it.
        # A selected reuse target is an ownership handoff, not a parked
        # value, even when its virtual end equals max_layers (a final-layer
        # Add) — excluded on both sides.
        parked_count = sum(
            1
            for n in gm.schedulable
            if n.node_id not in built.keep_forever_ids
            and n.node_id not in selected_target_ids
            and node_to_cancel_layer[n.node_id] == max_layers
        ) + sum(
            1
            for nid in input_cancel_layer
            if nid not in built.input_keep_ids
            and nid not in selected_target_ids
            and node_to_cancel_layer[nid] == max_layers
        )
        n_layers = solver.Value(n_layers_var)
        total_heads = solver.Value(total_attn_heads)
        total_bypass = solver.Value(total_mlp_bypass)
        objective = int(solver.ObjectiveValue())
        assignment: Optional[ScheduleAssignment] = ScheduleAssignment(
            node_to_layer=node_to_layer,
            node_to_cancel_layer=node_to_cancel_layer,
            node_to_routing=node_to_routing,
            n_layers=n_layers,
            node_to_cancel_mech=node_to_cancel_mech,
        )
    else:
        total_heads = -1
        total_bypass = -1
        objective = -1
        parked_count = -1
        assignment = None

    stats = SolveStats(
        status_name=solver.StatusName(status),
        objective_value=objective,
        best_objective_bound=float(solver.BestObjectiveBound()),
        wall_time_s=elapsed,
        solver_log="\n".join(log_buf),
        total_attn_heads=total_heads,
        total_mlp_bypass_slots=total_bypass,
        is_optimal=status == cp_model.OPTIMAL,
        objective_scale=built.objective_scale,
        parked_count=parked_count,
    )
    return assignment, stats
