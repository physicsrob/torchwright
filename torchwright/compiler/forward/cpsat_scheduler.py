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
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
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
from torchwright.graph.block import Block

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

    node_to_layer: Dict[int, int]
    node_to_cancel_layer: Dict[int, int]
    node_to_routing: Dict[int, str]
    n_layers: int


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
    # Hint-aware cancel-window widening actually applied: `node_id -> delta`
    # for every node whose window was widened past the uniform
    # `last_consumer + 1 + K` (only nonzero deltas appear).  None when the
    # model was built without hints.
    cancel_window_delta: Optional[Dict[int, int]] = None
    # The cancel-window slack K the model actually posted (None when the
    # window family is disabled or `cancel_slack=None`).
    eff_cancel_slack: Optional[int] = None


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


def routing(node: Node, gm: GraphModel, policy: SchedulingPolicy) -> str:
    """Static routing decision under the given policy.

    Used when `flex_routing=False`: every node has a fixed sublayer.
    With `flex_routing=True`, only `is_flex(n)` nodes' modes become
    CP-SAT decision variables; others still use this routing.
    """
    if isinstance(node, Attn):
        return ATTN
    if isinstance(node, Add):
        return ATTN
    if isinstance(node, Block):
        return MLP  # a Block is the L->ReLU->L composite, always MLP-locked
    if isinstance(node, LiteralValue):
        return MLP
    if isinstance(node, Linear):
        if policy.local_in_attention == "always":
            return ATTN
        return MLP
    raise TypeError(f"Unknown schedulable node type: {type(node).__name__}")


def is_flex(node: Node, gm: GraphModel) -> bool:
    """True iff this node's routing is a CP-SAT decision variable
    when `flex_routing=True`.

    Exactly the standalone Linears: a Linear can run in attention
    (`heads = ⌈d_input/d_head⌉`) or in MLP bypass (`slots = 2 ·
    d_output`).  The heuristic picks one statically per policy; CP-SAT
    can pick per-node.

    `Attn` / `Add` / `Block` / `LiteralValue` stay locked because they
    have only one valid sublayer (a Block is always the MLP composite).
    """
    return isinstance(node, Linear)


def heads_for(node: Node, d_head: int) -> int:
    """Heads consumed if attention-routed.

    Mirrors `LayerScheduler._heads_*`. For `Add`, returns the free-
    add unit count (`⌈d_out/d_head⌉` — one head per `d_head`-wide
    chunk of the live addend, copied into the dead addend's cols);
    the compute-add regime costs `2 ·` this. The CP-SAT model gates
    free vs compute via a per-Add `is_free` boolean derived from
    reified consumer-ordering booleans; see the helper inside
    `solve_schedule` and `docs/cpsat_scheduler.md` §3.
    """
    if isinstance(node, Attn):
        return (node.d_v + d_head - 1) // d_head
    if isinstance(node, Linear):
        d_in = len(node.inputs[0])
        return (d_in + d_head - 1) // d_head
    if isinstance(node, Add):
        d_out = len(node)
        return (d_out + d_head - 1) // d_head
    return 0


def slots_for(node: Node, gm: GraphModel) -> int:
    """MLP slots consumed if MLP-routed.

    A Block carries one hidden slot per lane (the composite's slot
    demand); a standalone Linear routed to MLP bypass needs `2 ·
    d_output`; everything else costs no hidden slots.
    """
    if isinstance(node, Block):
        return node.n_lanes
    if isinstance(node, Linear):
        return 2 * node.d_output  # MLP bypass
    if isinstance(node, LiteralValue):
        return 0
    return 0


def uses_residual(node: Node, gm: GraphModel) -> bool:
    """True iff this node gets its own residual-stream column allocation.

    Every schedulable node writes its output to the residual stream (a
    Block's output, a Linear's output, an Add, an Attn, a LiteralValue).
    The Block's internal ReLU activations live in MLP hidden slots, not
    the residual stream, but the Block node itself (its output) does use
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
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Per-node [earliest, latest] layer bounds from the dependency DAG.

    Mirrors the dependency-constraint semantics exactly: edge u->v allows
    same-layer placement only when u *can* run in the attention sublayer and
    v *can* run in MLP (gap 0); otherwise v must come at least one layer
    after u (gap 1).  "Can" means: flexible routing, or pinned to the
    needed sublayer.  These are the same bounds CP-SAT presolve derives by
    propagation; computing them here shrinks the input model instead.
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
        return (routing(n, gm, policy),)

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
) -> int:
    """Exact minimum layer count imposed by the dependency DAG alone.

    This is the mode-aware longest path through the graph (same semantics
    as the CP-SAT dependency constraints), ignoring width entirely.  No
    schedule can be shallower; when the residual stream has slack, the
    optimum EQUALS this value (measured on the DOOM graph at d=8192).
    Costs milliseconds beyond the graph-model build — usable as a
    pre-solve bound or a probe horizon.
    """
    if policy is None:
        policy = LEGACY_POLICY
    gm = build_graph_model(output_node, pos_encoding)
    es, _ = _compute_layer_bounds(gm, policy, flex_routing, max_layers=1 << 20)
    return max(es.values()) + 1


def build_cpsat_model(
    output_node: Node,
    pos_encoding=None,
    *,
    d: int,
    d_head: int,
    d_hidden: int,
    costs: Costs = Costs(),
    flex_routing: bool = True,
    max_layers: int = 60,
    cancel_slack: Optional[int] = 2,
    policy: Optional[SchedulingPolicy] = None,
    reserve_heads: int = 0,
    reserve_residual: int = 0,
    assume_zero_init: bool = False,
    tighten_domains: bool = False,
    hint_layers: Optional[Dict[int, int]] = None,
    hint_cancel: Optional[Dict[int, int]] = None,
    _disabled_families: frozenset = frozenset(),
) -> BuiltModel:
    """Build (but do not solve) the CP-SAT scheduling model.

    Extracted from :func:`solve_schedule` so the model construction has a
    single definition shared by the production solve and the diagnostic
    path.  ``solve_schedule`` calls this, then adds the warm-start hint, the
    decision strategy, solves, and reads the variables back out of the
    returned :class:`BuiltModel`.

    ``hint_layers`` / ``hint_cancel`` are used ONLY to size the per-node
    cancel windows (see the widening comment at the cancel-layer section) —
    hint *application* (``AddHint``) stays in :func:`solve_schedule`.  With
    both left as None the model is byte-identical to the hint-less build,
    so diagnostics, the probe script, and hint-less callers are unchanged.

    ``_disabled_families`` is a diagnostic-only escape hatch: each name in
    :data:`CONSTRAINT_FAMILIES` gates one constraint family, and listing it
    here skips posting that family.  The production path always passes the
    empty set (every family on); the diagnostic path bisects an infeasibility
    by disabling one family at a time over a hard-fixed schedule.  See
    ``torchwright_doom/scripts/cpsat_diagnose.py``.
    """
    unknown = _disabled_families - CONSTRAINT_FAMILIES
    if unknown:
        raise ValueError(
            f"Unknown constraint family/families {sorted(unknown)}; "
            f"valid names: {sorted(CONSTRAINT_FAMILIES)}"
        )

    if policy is None:
        policy = LEGACY_POLICY

    if costs.alpha == 0 and costs.beta == 0 and costs.gamma == 0:
        raise ValueError("alpha=beta=gamma=0 — no objective.")

    gm = build_graph_model(output_node, pos_encoding)

    n_heads_per_layer = d // d_head
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
        _compute_layer_bounds(gm, policy, flex_routing, max_layers)
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
    for n in gm.schedulable:
        if flex_routing and is_flex(n, gm):
            v = model.NewBoolVar(f"is_attn_n{n.node_id}")
        else:
            r = routing(n, gm, policy)
            v = model.NewBoolVar(f"is_attn_n{n.node_id}_pinned")
            if r == ATTN:
                model.Add(v == 1)
            else:
                model.Add(v == 0)
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

    # ---- Cancel layer per schedulable node ----
    # The natural lower bound on cancel_layer[n] is
    # ``max(layer[c] + 1)`` over consumers — the columns must outlive
    # every reader.  The natural upper bound is ``max_layers``, which
    # leaves ~60 candidate values per node on a DOOM-scale graph.
    # When ``cancel_slack`` is set, restrict to a small window above
    # the lower bound: the heuristic almost always cancels within 1–2
    # layers of the last consumer, so K=2 cuts the cancel decision
    # space ~30x with negligible loss of optimality.
    eff_cancel_slack = None if "cancel_slack" in _disabled_families else cancel_slack

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
    for n in gm.schedulable:
        cl = model.NewIntVar(0, max_layers, f"cl_n{n.node_id}")
        cancel_layer[n.node_id] = cl
        model.Add(cl >= layer_var[n.node_id] + 1)
        if n in gm.pinned_nodes:
            model.Add(cl == max_layers)
            keep_forever_ids.add(n.node_id)
            continue
        keep_forever = False
        consumer_layer_vars: List[cp_model.IntVar] = []
        consumer_ids: List[int] = []
        for c in gm.consumers_eff.get(n, set()):
            if isinstance(c, Concatenate):
                model.Add(cl == max_layers)
                keep_forever = True
                break
            if c.node_id in layer_var:
                if "cancel_consumer_lb" not in _disabled_families:
                    model.Add(cl >= layer_var[c.node_id] + 1)
                consumer_layer_vars.append(layer_var[c.node_id])
                consumer_ids.append(c.node_id)
        if keep_forever:
            keep_forever_ids.add(n.node_id)
            continue
        if eff_cancel_slack is not None and consumer_layer_vars:
            delta = _widen_delta(n.node_id, _hinted_last_consumer(consumer_ids))
            last_cons = model.NewIntVar(0, max_layers - 1, f"last_cons_n{n.node_id}")
            model.AddMaxEquality(last_cons, consumer_layer_vars)
            model.Add(cl <= last_cons + 1 + eff_cancel_slack + delta)
        elif eff_cancel_slack is not None and not consumer_layer_vars:
            # No layer-bound consumers — cancel can fire right after
            # the node's own birth layer.
            delta = _widen_delta(
                n.node_id,
                hint_layers.get(n.node_id) if hint_layers is not None else None,
            )
            model.Add(cl <= layer_var[n.node_id] + 1 + eff_cancel_slack + delta)

    # ---- Freeable input cancel layers ----
    # Freeable inputs are born at layer 0 (pre-allocated by the compiler
    # before the layer loop) and live until their last consumer runs, mirroring
    # the schedulable cancel logic with a fixed birth at 0.  An input feeding a
    # terminal `Concatenate` (output cone) is kept forever.
    input_cancel_layer: Dict[int, cp_model.IntVar] = {}
    input_keep_ids: Set[int] = set()
    for n in freeable_inputs:
        cl = model.NewIntVar(0, max_layers, f"cl_in{n.node_id}")
        input_cancel_layer[n.node_id] = cl
        model.Add(cl >= 1)  # born at layer 0; live through at least layer 0
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
                    model.Add(cl >= layer_var[c.node_id] + 1)
                consumer_layer_vars.append(layer_var[c.node_id])
                consumer_ids.append(c.node_id)
        if keep_forever:
            input_keep_ids.add(n.node_id)
            continue
        if eff_cancel_slack is not None and consumer_layer_vars:
            delta = _widen_delta(n.node_id, _hinted_last_consumer(consumer_ids))
            last_cons = model.NewIntVar(0, max_layers - 1, f"last_cons_in{n.node_id}")
            model.AddMaxEquality(last_cons, consumer_layer_vars)
            model.Add(cl <= last_cons + 1 + eff_cancel_slack + delta)
        elif eff_cancel_slack is not None and not consumer_layer_vars:
            # Born at layer 0, so the hinted base is the fixed birth layer 0.
            delta = _widen_delta(n.node_id, 0)
            model.Add(cl <= 1 + eff_cancel_slack + delta)

    # ---- Add free/compute classification ----
    # The heuristic schedules an `Add` via `add_into` (free regime)
    # iff at least one addend is dead at the Add's layer — every
    # other consumer of that addend has already computed in a strictly
    # prior layer.  Free-add costs `⌈d_out/d_head⌉` heads (copy the
    # live addend into the dead addend's already-allocated cols).
    # Otherwise the heuristic falls back to `compute_add`: fresh cols,
    # both inputs copied (≈ 2× heads) plus a BIRTH-layer dirty cancel
    # for the fresh cols.
    #
    # Encode: `is_free[A]` is a boolean equal to OR over addends E of
    # `E_dead_at_A` = AND over E's other consumers C of
    # `layer_var[C] < layer_var[A]`.  Strict inequality matches the
    # heuristic, which reads `computed_nodes` snapshotted at layer
    # start, so a same-layer attention consumer doesn't count as
    # making the addend dead at an MLP-routed Add (Adds always run in
    # attention anyway, so this corner doesn't fire — but the strict
    # form is the conservative/correct encoding).
    is_free: Dict[int, cp_model.IntVar] = {}
    for A in gm.schedulable:
        if not isinstance(A, Add):
            continue
        addend_dead_bools: List[cp_model.IntVar] = []
        for E in A.inputs:
            E_dead = model.NewBoolVar(f"E_dead_A{A.node_id}_E{E.node_id}")
            disqualifying = (
                isinstance(E, Concatenate)
                or E in gm.pinned_nodes
                or E.node_id not in layer_var
            )
            if disqualifying:
                model.Add(E_dead == 0)
                addend_dead_bools.append(E_dead)
                continue
            other_consumers = [c for c in gm.consumers_eff.get(E, set()) if c is not A]
            # Any consumer that lacks a layer_var (terminal Concatenate
            # in the output cone, or any non-schedulable consumer) is
            # "always alive" — E can never be dead at A.
            has_unschedulable = any(
                isinstance(c, Concatenate) or c.node_id not in layer_var
                for c in other_consumers
            )
            if has_unschedulable:
                model.Add(E_dead == 0)
                addend_dead_bools.append(E_dead)
                continue
            before_bools: List[cp_model.IntVar] = []
            for C in other_consumers:
                b = model.NewBoolVar(f"before_E{E.node_id}_C{C.node_id}_A{A.node_id}")
                model.Add(layer_var[C.node_id] < layer_var[A.node_id]).OnlyEnforceIf(b)
                model.Add(layer_var[C.node_id] >= layer_var[A.node_id]).OnlyEnforceIf(
                    b.Not()
                )
                before_bools.append(b)
            if before_bools:
                model.AddBoolAnd(before_bools).OnlyEnforceIf(E_dead)
                model.AddBoolOr([b.Not() for b in before_bools]).OnlyEnforceIf(
                    E_dead.Not()
                )
            else:
                # E feeds only A — E_dead is constant True.
                model.Add(E_dead == 1)
            addend_dead_bools.append(E_dead)
        is_free_A = model.NewBoolVar(f"is_free_A{A.node_id}")
        if addend_dead_bools:
            model.AddBoolOr(addend_dead_bools).OnlyEnforceIf(is_free_A)
            model.AddBoolAnd([b.Not() for b in addend_dead_bools]).OnlyEnforceIf(
                is_free_A.Not()
            )
        else:
            model.Add(is_free_A == 0)
        is_free[A.node_id] = is_free_A

    # ---- Combined attn-heads + cancel-cols cumulative ----
    # Per-node attn interval is OPTIONAL (gated by is_attn[n]) when
    # the node could run in either sublayer; for pinned nodes, the
    # bool is constant and CP-SAT presolve drops the unreachable
    # branch.
    #
    # `Add` gets two optional intervals — one demand for the free-add
    # regime (`⌈d_out/d_head⌉` heads), one for compute-add (`2 ·
    # ⌈d_out/d_head⌉` heads), gated by `is_free[A]` and its negation
    # respectively.  Adds are pinned to attention so we don't gate by
    # `is_attn[A]` here.
    attn_intervals: List = []
    attn_demands: List[int] = []
    for n in gm.schedulable:
        h = heads_for(n, d_head)
        if h <= 0:
            continue
        if isinstance(n, Add):
            free_end = model.NewIntVar(1, max_layers, f"aend_free_n{n.node_id}")
            model.Add(free_end == layer_var[n.node_id] + 1)
            iv_free = model.NewOptionalIntervalVar(
                layer_var[n.node_id],
                1,
                free_end,
                is_free[n.node_id],
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
                is_free[n.node_id].Not(),
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
    dirty_intervals: List = []
    dirty_demands: List[int] = []
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
        iv = model.NewIntervalVar(
            cancel_layer[n.node_id], 1, c_end, f"civ_n{n.node_id}"
        )
        cancel_intervals.append(iv)
        cancel_demands.append(len(n))

        # Birth-layer dirty-column cancel.  When `assume_zero_init` is
        # False (the default, mirroring the heuristic's defensive
        # behaviour), every fresh allocation pays a cancel head to
        # clear the column's prior value before its additive write.
        # When True, the runtime is contracted to zero-initialise the
        # residual stream and the heuristic skips these cancels — so
        # we skip them in the model too.
        #
        # `Add` is conditional: free-add reuses the dead addend's
        # already-clean cols (no dirty bits), so no cancel.  Compute-
        # add allocates fresh cols and pays the dirty cancel.  Gate
        # the dirty interval by `is_free[A].Not()`.
        if assume_zero_init:
            continue
        d_end = model.NewIntVar(1, max_layers, f"dend_n{n.node_id}")
        model.Add(d_end == layer_var[n.node_id] + 1)
        if isinstance(n, Add):
            d_iv = model.NewOptionalIntervalVar(
                layer_var[n.node_id],
                1,
                d_end,
                is_free[n.node_id].Not(),
                f"div_n{n.node_id}",
            )
        else:
            d_iv = model.NewIntervalVar(
                layer_var[n.node_id], 1, d_end, f"div_n{n.node_id}"
            )
        dirty_intervals.append(d_iv)
        dirty_demands.append(len(n))

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
        iv = model.NewIntervalVar(cl_in, 1, c_end, f"civ_in{n.node_id}")
        cancel_intervals.append(iv)
        cancel_demands.append(len(n))

    # `reserve_heads` is a safety knob for graphs whose attention
    # heads are saturated by ops outside the model (e.g. bias writes
    # folded into deferred Linears); default 0.
    effective_capacity = max(0, n_heads_per_layer - reserve_heads) * d_head
    if "attn_cumulative" not in _disabled_families and (
        attn_intervals or cancel_intervals or dirty_intervals
    ):
        model.AddCumulative(
            attn_intervals + cancel_intervals + dirty_intervals,
            attn_demands + cancel_demands + dirty_demands,
            effective_capacity,
        )

    # ---- MLP slots cumulative ----
    # For flex nodes, MLP demand is gated by NOT(is_attn).  A Block carries
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
    if "mlp_cumulative" not in _disabled_families and mlp_intervals:
        model.AddCumulative(mlp_intervals, mlp_demands, d_hidden)

    # ---- Residual cumulative ----
    residual_nodes = [n for n in gm.schedulable if uses_residual(n, gm)]
    resid_intervals: List = []
    resid_demands: List[int] = []
    for n in residual_nodes:
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
            model.Add(size == cancel_layer[n.node_id] - start)
            iv = model.NewIntervalVar(
                start, size, cancel_layer[n.node_id], f"riv_n{n.node_id}"
            )
        else:
            size = model.NewIntVar(1, max_layers + 1, f"rsz_n{n.node_id}")
            model.Add(size == cancel_layer[n.node_id] - layer_var[n.node_id])
            iv = model.NewIntervalVar(
                layer_var[n.node_id],
                size,
                cancel_layer[n.node_id],
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
        iv = model.NewIntervalVar(0, cl_in, cl_in, f"riv_in{n.node_id}")
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
            # Add always runs in attention; cost is h (free) or 2h
            # (compute), gated by `is_free[A]`.  Always pay h, plus
            # an extra h when not free.
            fixed_attn_heads += h
            attn_term.append(h * is_free[n.node_id].Not())
        elif flex_routing and is_flex(n, gm):
            attn_term.append(h * is_attn[n.node_id])
        else:
            r = routing(n, gm, policy)
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
            r = routing(n, gm, policy)
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
            occupancy.append(len(n) * input_cancel_layer[n.node_id])
            max_secondary += costs.waste * len(n) * max_layers
        secondary_terms.append(costs.waste * sum(occupancy))
    objective_scale = 1
    if secondary_terms:
        objective_scale = max_secondary + 1
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
        n_layers_var=n_layers_var,
        total_attn_heads=total_attn_heads,
        total_mlp_bypass=total_mlp_bypass,
        available_residual=available_residual,
        n_heads_per_layer=n_heads_per_layer,
        input_cancel_layer=input_cancel_layer,
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
    )


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
    hint_layers: Optional[Dict[int, int]],
    hint_routing: Optional[Dict[int, str]],
    hint_cancel: Optional[Dict[int, int]],
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
        if birth is not None and L < birth + 1:
            violations.append(
                f"cancel hint before birth+1: {_desc(nid)} cancel={L} " f"birth={birth}"
            )
        node = all_nodes.get(nid)
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
            if L < c_hint + 1:
                violations.append(
                    f"cancel hint before consumer's layer+1: {_desc(nid)} "
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
    d_hidden: int,
    costs: Costs = Costs(),
    flex_routing: bool = True,
    time_budget_s: float = 60.0,
    max_layers: int = 60,
    hint_layers: Optional[Dict[int, int]] = None,
    hint_routing: Optional[Dict[int, str]] = None,
    hint_cancel: Optional[Dict[int, int]] = None,
    cancel_slack: Optional[int] = 2,
    policy: Optional[SchedulingPolicy] = None,
    log_search_progress: bool = False,
    reserve_heads: int = 0,
    reserve_residual: int = 0,
    assume_zero_init: bool = False,
    tighten_domains: bool = False,
    solver_params: Optional[Dict[str, object]] = None,
    solution_trace: Optional[List[dict]] = None,
    strict_hint: bool = False,
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
        d, d_head, d_hidden: transformer geometry. ``n_heads_per_layer
            = d // d_head``.  Residual budget is
            ``d - input_residual_cols``.
        costs: objective weights. See :class:`Costs`.
        flex_routing: if True, CP-SAT picks attention vs MLP for each
            standalone ``Linear``.  If False, standalone Linears use
            the static routing dictated by ``policy.local_in_attention``.
        time_budget_s: per-solve wall-clock cap.
        max_layers: search horizon.  Should be at least the heuristic's
            layer count.
        hint_layers: optional warm-start mapping ``node_id -> layer``.
        hint_routing: optional warm-start mapping
            ``node_id -> "attn"|"mlp"`` for flex Linears.  When the
            heuristic placed a standalone Linear in attention vs
            MLP-bypass, hinting the same routing lets CP-SAT
            reconstruct the heuristic's solution as a starting
            incumbent.
        hint_cancel: optional warm-start mapping ``node_id -> layer``
            for the cancel layer.  Captures when the heuristic freed
            each node's columns; combined with ``hint_layers`` this
            gives a complete schedule the solver can verify and
            improve from.  ``hint_layers``/``hint_cancel`` are also
            forwarded to :func:`build_cpsat_model` to widen each hinted
            node's cancel window just enough to admit the hint (the
            heuristic defers a free when a layer's heads are full, which
            otherwise lands past the uniform window and silently
            invalidates the whole incumbent).
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
        assume_zero_init: if True, the model assumes the runtime
            zero-initialises the residual stream (so the heuristic
            emits no BIRTH-layer dirty-column cancels for fresh
            allocations on the initially-free pool).  Pair this with
            ``forward_compile(assume_zero_init=True)`` so the heuristic
            and CP-SAT model agree.  Defaults to False — the
            conservative model that mirrors the heuristic's defensive
            BIRTH-layer cancellation of fresh allocations.
        strict_hint: if True, a hint the model would drop or reject
            raises ``ValueError`` (see :func:`_validate_hint`).  Default
            False emits one ``RuntimeWarning`` instead — production
            keeps its fall-back-don't-fail contract; tests use strict.

    Raises ``RuntimeError`` only on structural problems (no residual
    columns left after pre-allocated inputs).  Solver-outcome
    handling (no-incumbent, FEASIBLE-not-OPTIMAL) is the caller's
    responsibility.
    """
    built = build_cpsat_model(
        output_node,
        pos_encoding,
        d=d,
        d_head=d_head,
        d_hidden=d_hidden,
        costs=costs,
        flex_routing=flex_routing,
        max_layers=max_layers,
        cancel_slack=cancel_slack,
        policy=policy,
        reserve_heads=reserve_heads,
        reserve_residual=reserve_residual,
        assume_zero_init=assume_zero_init,
        tighten_domains=tighten_domains,
        hint_layers=hint_layers,
        hint_cancel=hint_cancel,
    )
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
    if any(h is not None for h in (hint_layers, hint_routing, hint_cancel)):
        _validate_hint(
            built,
            hint_layers,
            hint_routing,
            hint_cancel,
            max_layers=max_layers,
            strict=strict_hint,
        )
    if hint_layers is not None:
        for nid, L in hint_layers.items():
            if nid in layer_var and 0 <= L < max_layers:
                model.AddHint(layer_var[nid], L)
    if hint_routing is not None:
        for nid, route in hint_routing.items():
            if nid in is_attn:
                model.AddHint(is_attn[nid], 1 if route == ATTN else 0)
    if hint_cancel is not None:
        for nid, L in hint_cancel.items():
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
        for n in gm.schedulable:
            node_to_layer[n.node_id] = solver.Value(layer_var[n.node_id])
            node_to_cancel_layer[n.node_id] = solver.Value(cancel_layer[n.node_id])
            node_to_routing[n.node_id] = (
                ATTN if solver.Value(is_attn[n.node_id]) else MLP
            )
        # Freeable inputs get a cancel layer (but no layer/routing — they are
        # pre-computed at layer 0).  ``DirectedLayerScheduler._find_dead_nodes``
        # frees any allocated node whose cancel layer matches the current
        # layer, so adding inputs here makes the replay reclaim their columns.
        for nid, cl_in in input_cancel_layer.items():
            node_to_cancel_layer[nid] = solver.Value(cl_in)
        n_layers = solver.Value(n_layers_var)
        total_heads = solver.Value(total_attn_heads)
        total_bypass = solver.Value(total_mlp_bypass)
        objective = int(solver.ObjectiveValue())
        assignment: Optional[ScheduleAssignment] = ScheduleAssignment(
            node_to_layer=node_to_layer,
            node_to_cancel_layer=node_to_cancel_layer,
            node_to_routing=node_to_routing,
            n_layers=n_layers,
        )
    else:
        total_heads = -1
        total_bypass = -1
        objective = -1
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
    )
    return assignment, stats
