"""Realization classes and the realization table.

A **realization class** is a write-path choice: which transformer hardware
computes a fixed, already-built graph node (docs/lowering_boundary_plan.md).
The node, its weights, and its consumers are unchanged across classes —
only the weight-writer emission differs.

This module is the **single declaration of the option set**
(:func:`candidate_classes`): it is read by ``lower()`` (which builds the
unresolved table), by the static resolver (``optimize=0``), and by the
CP-SAT model builder (``is_flex`` / static ``routing``).  Before this
module existed the option set was hard-coded twice — a policy read in the
eager scheduler and an ``isinstance`` check in the CP-SAT model — with
nothing keeping the two in agreement.

The **realization table** is the one artifact recording every schedulable
node's class.  Both resolvers write it; the layer walk only reads it.
``Add`` resolves to a *route family* (``ATTN_ADD`` or ``MLP_ADD``) before
the layer walk begins; the walk then chooses the operation within the
family from addend liveness (a dead addend's columns are reused in place,
or the output goes to fresh columns).  Residual placement is scheduler
state, not a realization class (docs/plan_additional_mlp_routing.md).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.graph import Add, Attn, Linear, Node
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import LiteralValue

# ---------------------------------------------------------------------------
# Realization classes (write-path choices)
# ---------------------------------------------------------------------------

#: Standalone Linear as rotary Δ=0 self-match attention head(s).
ATTN_TRANSPORT = "attn_transport"
#: Standalone Linear in the MLP via the bypass identity act(z) - act(-z) = z
#: (two hidden slots per output column; exact for relu and for swish).
MLP_BYPASS = "mlp_bypass"
#: An Attn node on attention heads (the only hardware that attends).
ATTN_HEADS = "attn_heads"
#: An FFN on the MLP sublayer (the L→act→L composite; always MLP-locked).
MLP_COMPOSITE = "mlp_composite"
#: A LiteralValue written by the MLP constant path.
MLP_LITERAL = "mlp_literal"
#: An Add on attention heads.  The layer walk picks the operation from
#: addend liveness: reuse a dead addend's columns (add_into) or copy both
#: addends to fresh columns (compute_add).
ATTN_ADD = "attn_add"
#: An Add in the MLP via the bypass identity (two hidden slots per output
#: column).  Same liveness choice: add_into_bypass or compute_add_bypass.
MLP_ADD = "mlp_add"

#: Which sublayer each class occupies.
CLASS_SUBLAYER: dict[str, str] = {
    ATTN_TRANSPORT: "attn",
    ATTN_HEADS: "attn",
    ATTN_ADD: "attn",
    MLP_BYPASS: "mlp",
    MLP_ADD: "mlp",
    MLP_COMPOSITE: "mlp",
    MLP_LITERAL: "mlp",
}


def candidate_classes(node: Node) -> tuple[str, ...]:
    """The candidate realization classes for a schedulable node.

    THE single declaration of the option set.  A single-candidate result is
    the legitimate degenerate case (most node types have one obviously-right
    hardware form); the standalone ``Linear`` and the ``Add`` have a free
    attention-versus-MLP route choice.
    """
    if isinstance(node, Attn):
        return (ATTN_HEADS,)
    if isinstance(node, FFN):
        return (MLP_COMPOSITE,)
    if isinstance(node, LiteralValue):
        return (MLP_LITERAL,)
    if isinstance(node, Add):
        return (ATTN_ADD, MLP_ADD)
    if isinstance(node, Linear):
        return (ATTN_TRANSPORT, MLP_BYPASS)
    raise TypeError(f"No realization classes for {type(node).__name__}")


def is_schedulable(node: Node) -> bool:
    """True for node types that occupy hardware (get a table entry)."""
    return isinstance(node, (Attn, FFN, LiteralValue, Add, Linear))


def has_flex_choice(node: Node) -> bool:
    """True when the node's class is a *free* choice a resolver must make.

    A CP-SAT decision variable in the directed path — the standalone
    ``Linear`` and the ``Add``.
    """
    return len(candidate_classes(node)) > 1


def flex_route_classes(node: Node) -> tuple[str, str]:
    """The (attention-family, MLP-family) candidates of a free-routing-choice node.

    Raises ``TypeError`` on nodes without exactly one candidate per
    sublayer — the route resolvers and skip diagnostics may only call this
    on :func:`has_flex_choice` nodes.
    """
    candidates = candidate_classes(node)
    attn = [c for c in candidates if CLASS_SUBLAYER[c] == "attn"]
    mlp = [c for c in candidates if CLASS_SUBLAYER[c] == "mlp"]
    if len(attn) != 1 or len(mlp) != 1:
        raise TypeError(
            f"{type(node).__name__} has no attention/MLP route pair: "
            f"candidates={candidates}"
        )
    return attn[0], mlp[0]


def first_hidden_slot(bias: bool) -> int:
    """The lowest hidden slot a scheduled node may pack into.

    ``bias=False`` reserves hidden slot 0 for the weight-writer's constant
    lane (the BiasFold), so packing starts at slot 1.
    """
    return 0 if bias else 1


def usable_hidden_slots(d_hidden: int, bias: bool) -> int:
    """Hidden slots one layer's MLP can hand out to scheduled nodes.

    The eager scheduler's slot packing, the CP-SAT hidden-slot cumulative,
    and :func:`static_flex_class` below must agree on this number: the
    routing rule decides whether a node *can* live in the MLP, and the two
    schedulers then have to be able to place what it decided.
    """
    return d_hidden - first_hidden_slot(bias)


def fits_mlp(node: Node, usable_slots: int) -> bool:
    """Whether *node* has an MLP realization at all under this geometry.

    Both MLP route families spend two hidden slots per output column (the
    bypass identity ``act(z) - act(-z) = z``), so a node whose
    ``2 · d_output`` exceeds a layer's entire usable hidden pool can never
    be placed — not in this layer, not in any layer, no matter how empty
    the MLP is.  That makes this a *structural* property of the node and
    the geometry, unlike the scheduler's per-layer "no slots left right
    now" check, which is transient.
    """
    _, mlp_cls = flex_route_classes(node)
    return class_hidden_slots(node, mlp_cls) <= usable_slots


def static_flex_class(node: Node, policy: SchedulingPolicy, usable_slots: int) -> str:
    """The static policy's pick for a free choice.

    The one rule both the optimize=0 resolver and the CP-SAT model's
    pinned (``flex_routing=False``) routing apply.

    Capacity is checked before policy.  A node too wide for its MLP family
    (:func:`fits_mlp`) has exactly one realization left, so it goes to its
    attention family whatever the policy says; routing it to a sublayer
    that cannot hold it deadlocks the eager walk and makes the CP-SAT
    model infeasible.  Otherwise the node's own knob decides — Adds read
    ``policy.add_in_attention``, standalone Linears read
    ``policy.local_in_attention`` (the two default to opposite sides; see
    :class:`SchedulingPolicy`): attention unless the knob says
    ``"never"``.
    """
    attn_cls, mlp_cls = flex_route_classes(node)
    if not fits_mlp(node, usable_slots):
        return attn_cls
    knob = (
        policy.add_in_attention if isinstance(node, Add) else policy.local_in_attention
    )
    if knob != "never":
        return attn_cls
    return mlp_cls


# ---------------------------------------------------------------------------
# Resource signatures: hardware demand per (node, class)
# ---------------------------------------------------------------------------


def live_weight_row_ranges(node: Node) -> tuple[tuple[int, int], ...]:
    """The nonzero rows of ``node.output_matrix``, as ``(start, length)`` runs.

    THE single support convention for attention-transport head charging —
    nonzero entries of ``output_matrix`` only, matching what
    ``trim_unused_slots`` reads on the MLP side.  ``output_bias`` never
    enters: a bias needs no input read (it is written by the separate
    ``compute_bias`` MLP op), so it cannot make a weight row live.

    Runs rather than a row list or a head count because both compactness
    and geometry matter: a dense matrix collapses to one run, and runs are
    ``d_head``-free, so the CP-SAT snapshot can persist them
    (``cpsat_snapshot.NodeRecord``) and recompute the head charge at
    whatever geometry the model is later built with — a stored integer
    would silently bake in the capture-time ``d_head``.

    Computed once per node and cached on it (``_live_weight_row_ranges``);
    every charge site queries a node many times, and by the time any
    charge runs (scheduling, model build) the lowering folds that mutate
    weights have finished.  Snapshot stand-in nodes carry the attribute
    directly — they have no ``output_matrix`` to scan.
    """
    cached = getattr(node, "_live_weight_row_ranges", None)
    if cached is not None:
        return cached
    live = cast("Any", node).output_matrix.ne(0).any(dim=1)
    runs = []
    start = None
    for i, flag in enumerate(live.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(live) - start))
    ranges = tuple(runs)
    node._live_weight_row_ranges = ranges
    return ranges


def _linear_live_attn_chunks(node: Node, d_head: int) -> tuple[int, ...]:
    """Attention chunks with a nonzero output block, without the model floor."""
    return tuple(
        sorted(
            {
                chunk
                for start, length in live_weight_row_ranges(node)
                for chunk in range(
                    start // d_head,
                    (start + length - 1) // d_head + 1,
                )
            }
        )
    )


def linear_attn_chunks(node: Node, d_head: int) -> tuple[int, ...]:
    """The ``d_head``-wide input chunks a Linear's attention transport actually needs.

    In ascending order.  THE single head-charge declaration.
    ``_write_compute_linear`` splits a
    Linear's input into ``d_head``-wide chunks and emits one Δ=0 self-match
    head per chunk; a chunk whose weight rows are all zero contributes
    exactly zero to the output (its head's O block would be zero), so it is
    neither charged nor emitted.  Every accounting site (the CP-SAT model's
    ``heads_for``, the eager scheduler, :func:`class_heads`, the compile
    diagnostics) and the emitter iterate THIS list, so the budget and the
    emission cannot desync.

    Never empty: a zero-support Linear (its value is just the bias, or
    exactly zero) keeps chunk 0 — one zero-weight head.  The floor costs
    one head on a node type that may not exist in practice, and it keeps
    every Linear inside the CP-SAT model (the builder skips ``h == 0``
    nodes entirely, which would drop a zero-support *flex* Linear's routing
    variable from the objective and its interval from the cumulative).
    """
    chunks = _linear_live_attn_chunks(node, d_head)
    return chunks or (0,)


def linear_attn_heads(node: Node, d_head: int) -> int:
    """Attention-transport head demand of a standalone Linear.

    One head per live input chunk (:func:`linear_attn_chunks`), floor 1.
    """
    return len(linear_attn_chunks(node, d_head))


def linear_attn_live_heads(node: Node, d_head: int) -> int:
    """Post-compaction heads of an attention-transport Linear.

    The scheduler charges a one-head floor for a zero-support Linear, but the
    dense component's trim removes that head because its output block is zero.
    ReplayPlan shape and artifact-resource accounting use this live count.
    """
    return len(_linear_live_attn_chunks(node, d_head))


def add_attn_heads(node: Node, d_head: int, *, reused: bool) -> int:
    """Attention-head demand of an ``ATTN_ADD`` at a known residual placement.

    Reused placement (``add_into``) copies the live addend into the dead
    addend's columns: one head per ``d_head``-wide chunk.  Fresh placement
    (``compute_add``) is charged the conservative model-level ``2 x`` that;
    the walk's emission can share one head when two chunks of the same Add
    fit in a single ``d_head``.
    """
    h = (node.d_output + d_head - 1) // d_head
    return h if reused else 2 * h


def class_heads(node: Node, cls: str, d_head: int) -> int:
    """Attention-head demand of realizing *node* as *cls* (0 for MLP classes).

    ``ATTN_ADD`` demand depends on reused-versus-fresh residual
    placement, which is scheduler state, not part of the class — callers
    that know the placement use :func:`add_attn_heads`; callers that don't
    report the range (see :func:`summarize_cost`).
    """
    if cls == ATTN_HEADS:
        return (cast("Any", node).d_v + d_head - 1) // d_head
    if cls == ATTN_TRANSPORT:
        return linear_attn_heads(node, d_head)
    if cls == ATTN_ADD:
        raise ValueError(
            "ATTN_ADD head demand depends on residual placement; call "
            "add_attn_heads(node, d_head, reused=...) with the placement"
        )
    return 0


def class_hidden_slots(node: Node, cls: str) -> int:
    """MLP hidden-slot demand of realizing *node* as *cls*.

    0 for attention classes; a literal's constant write costs no hidden slot.
    """
    if cls in (MLP_BYPASS, MLP_ADD):
        return 2 * node.d_output
    if cls == MLP_COMPOSITE:
        return cast("int", cast("Any", node).n_lanes)
    return 0


@dataclass(frozen=True)
class CostSummary:
    """Hardware demand read off a lowered graph *before* scheduling.

    Aggregated from the resolved realization table plus each class's
    resource signature.  An attention-routed Add's head demand depends on
    reused-versus-fresh residual placement (scheduler state), so it is
    reported as the two bounds (every ``ATTN_ADD`` reused vs every
    ``ATTN_ADD`` fresh); the schedule lands in between.  An MLP-routed Add
    contributes hidden slots (2 per output column) to ``mlp_bypass_slots``
    and no heads.  These are demand totals, not a layer count — placement
    (and the cancel heads it emits) is the scheduler's business.
    """

    nodes_by_class: dict[str, int]
    heads_by_class: dict[str, int]  # fixed-demand attention classes only
    add_heads_if_all_reuse: int
    add_heads_if_all_copy: int
    mlp_bypass_slots: int
    mlp_lanes: int

    @property
    def attn_heads_min(self) -> int:
        return sum(self.heads_by_class.values()) + self.add_heads_if_all_reuse

    @property
    def attn_heads_max(self) -> int:
        return sum(self.heads_by_class.values()) + self.add_heads_if_all_copy

    def format_short(self) -> str:
        by_class = ", ".join(
            f"{cls}={n}" for cls, n in sorted(self.heads_by_class.items())
        )
        return (
            f"attn heads {self.attn_heads_min}..{self.attn_heads_max} "
            f"({by_class or 'no fixed-head classes'}, "
            f"adds {self.add_heads_if_all_reuse}..{self.add_heads_if_all_copy}); "
            f"mlp: {self.mlp_bypass_slots} bypass slots + {self.mlp_lanes} lanes; "
            f"nodes {dict(sorted(self.nodes_by_class.items()))}"
        )


def summarize_cost(
    nodes: Iterable[Node], table: "RealizationTable", d_head: int
) -> CostSummary:
    """Aggregate hardware demand over *nodes* under a resolved *table*.

    Raises :class:`UnresolvedRealizationError` if the table is incomplete
    for the schedulable nodes (resolve first).
    """
    nodes = [n for n in nodes if is_schedulable(n)]
    table.check_complete(nodes)
    nodes_by_class: dict[str, int] = {}
    heads_by_class: dict[str, int] = {}
    add_reuse = add_copy = bypass_slots = lanes = 0
    for node in nodes:
        entry = table.entries[node.node_id]
        cls = cast("str", entry.resolved)
        nodes_by_class[cls] = nodes_by_class.get(cls, 0) + 1
        if cls == ATTN_ADD:
            add_reuse += add_attn_heads(node, d_head, reused=True)
            add_copy += add_attn_heads(node, d_head, reused=False)
            continue
        h = class_heads(node, cls, d_head)
        if h:
            heads_by_class[cls] = heads_by_class.get(cls, 0) + h
        if cls in (MLP_BYPASS, MLP_ADD):
            bypass_slots += class_hidden_slots(node, cls)
        elif cls == MLP_COMPOSITE:
            lanes += class_hidden_slots(node, cls)
    return CostSummary(
        nodes_by_class=nodes_by_class,
        heads_by_class=heads_by_class,
        add_heads_if_all_reuse=add_reuse,
        add_heads_if_all_copy=add_copy,
        mlp_bypass_slots=bypass_slots,
        mlp_lanes=lanes,
    )


# ---------------------------------------------------------------------------
# The realization table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One node's realization record.

    ``resolved`` is the chosen class (None while unresolved).
    """

    candidates: tuple[str, ...]
    resolved: str | None = None


class UnresolvedRealizationError(RuntimeError):
    """A realization entry was read (or reached the walk) unresolved."""


class RealizationTable:
    """node_id → :class:`Entry` for every schedulable node in a graph.

    Built unresolved by ``lower()``; resolved by exactly one of the two
    resolvers (:meth:`resolve_static` for the heuristic path,
    :meth:`resolve_from_assignment` for the CP-SAT directed path); read by
    the layer walk.  Resolvers return a new table — the unresolved table
    on the ``LoweredGraph`` is never mutated, so several resolutions of one
    lowering (e.g. the CP-SAT warm-start hint pass and the directed replay)
    cannot alias.
    """

    def __init__(self, entries: dict[int, Entry]) -> None:
        self.entries = entries

    @classmethod
    def build(cls, nodes: Iterable[Node]) -> "RealizationTable":
        """Unresolved table over the schedulable nodes of a graph.

        Single-candidate entries are born resolved (the degenerate case);
        multi-candidate entries (standalone Linears and Adds) are born
        unresolved.
        """
        entries: dict[int, Entry] = {}
        for node in nodes:
            if not is_schedulable(node):
                continue
            candidates = candidate_classes(node)
            if len(candidates) == 1:
                entries[node.node_id] = Entry(candidates, resolved=candidates[0])
            else:
                entries[node.node_id] = Entry(candidates)
        return cls(entries)

    # -- resolvers ---------------------------------------------------------

    def resolve_static(
        self,
        nodes: Iterable[Node],
        policy: SchedulingPolicy,
        usable_slots: int,
        forced_classes: dict[int, str] | None = None,
    ) -> "RealizationTable":
        """The optimize=0 resolver: :func:`static_flex_class` picks every free choice.

        :func:`static_flex_class` applies each flexible node's own knob
        (``policy.local_in_attention`` for standalone Linears,
        ``policy.add_in_attention`` for Adds) — except that a node too
        wide for a layer's usable hidden pool (``usable_slots``, from
        :func:`usable_hidden_slots`) has no MLP realization and goes to
        attention regardless.

        Takes *nodes* because the capacity check reads each unresolved node's
        width, which the ``node_id``-keyed entries do not carry.
        """
        by_id = {n.node_id: n for n in nodes if is_schedulable(n)}
        forced_classes = forced_classes or {}
        resolved: dict[int, Entry] = {}
        for node_id, entry in self.entries.items():
            forced = forced_classes.get(node_id)
            if forced is not None:
                if forced not in entry.candidates:
                    raise UnresolvedRealizationError(
                        f"cannot force node {node_id} to realization {forced!r}; "
                        f"candidates={entry.candidates}"
                    )
                resolved[node_id] = replace(entry, resolved=forced)
            elif entry.resolved is None:
                node = by_id.get(node_id)
                if node is None:
                    raise UnresolvedRealizationError(
                        f"node {node_id} has an unresolved entry "
                        f"{entry.candidates} but is absent from the nodes "
                        f"passed to resolve_static"
                    )
                choice = static_flex_class(node, policy, usable_slots)
                resolved[node_id] = replace(entry, resolved=choice)
            else:
                resolved[node_id] = entry
        return RealizationTable(resolved)

    def resolve_from_assignment(
        self, node_to_routing: Mapping[int, str]
    ) -> "RealizationTable":
        """The directed-path resolver: per-node sublayer decisions pick every choice.

        Decisions come from the CP-SAT solve
        (``ScheduleAssignment.node_to_routing``).  Raises if the
        assignment contradicts a locked entry's sublayer — the tripwire
        for the two option sets drifting apart.
        """
        resolved: dict[int, Entry] = {}
        for node_id, entry in self.entries.items():
            routing = node_to_routing.get(node_id)
            if entry.resolved is not None:
                if routing is not None and CLASS_SUBLAYER[entry.resolved] != routing:
                    raise UnresolvedRealizationError(
                        f"solve routed node {node_id} to {routing!r} but its "
                        f"only realization class {entry.resolved!r} lives on "
                        f"{CLASS_SUBLAYER[entry.resolved]!r}"
                    )
                resolved[node_id] = entry
                continue
            if routing is None:
                # Not placed by the solve (e.g. a freeable input literal
                # precomputed at layer 0) — leave unresolved; the
                # completeness check only covers nodes the walk schedules.
                resolved[node_id] = entry
                continue
            matches = [c for c in entry.candidates if CLASS_SUBLAYER[c] == routing]
            if len(matches) != 1:
                raise UnresolvedRealizationError(
                    f"solve routing {routing!r} for node {node_id} does not "
                    f"select exactly one of its candidates {entry.candidates}"
                )
            resolved[node_id] = replace(entry, resolved=matches[0])
        return RealizationTable(resolved)

    # -- readers -----------------------------------------------------------

    def resolved_class(self, node: Node) -> str:
        entry = self.entries.get(node.node_id)
        if entry is None:
            raise UnresolvedRealizationError(
                f"node {node.node_id} ({type(node).__name__}) has no realization entry"
            )
        if entry.resolved is None:
            raise UnresolvedRealizationError(
                f"realization for node {node.node_id} read before resolution "
                f"(candidates {entry.candidates})"
            )
        return entry.resolved

    def is_attention_routed(self, node: Node) -> bool:
        """The layer walk's read: does this node's class live on attention?"""
        return CLASS_SUBLAYER[self.resolved_class(node)] == "attn"

    def check_complete(self, nodes: Iterable[Node]) -> None:
        """Every schedulable node has a resolved entry.

        Runs before the layer walk.
        """
        problems = []
        for node in nodes:
            if not is_schedulable(node):
                continue
            entry = self.entries.get(node.node_id)
            if entry is None:
                problems.append(f"{type(node).__name__} {node.node_id}: no entry")
            elif entry.resolved is None:
                problems.append(
                    f"{type(node).__name__} {node.node_id}: unresolved "
                    f"{entry.candidates}"
                )
        if problems:
            raise UnresolvedRealizationError(
                "realization table incomplete before the layer walk: "
                + "; ".join(problems)
            )
