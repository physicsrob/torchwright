"""Layer scheduler for the forward compiler.

Given the current residual stream state and graph metadata, decides what
to compute/cancel in one transformer layer. Mutable scheduler records stay
private to this module and are frozen into ReplayPlan records at the boundary.

Mutates residual_map (allocate, free, reassign) and computed_nodes (add).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from torchwright.compiler.realization import (
    RealizationTable,
    add_attn_heads,
    class_hidden_slots,
    first_hidden_slot,
    flex_route_classes,
    has_flex_choice,
    linear_attn_heads,
    usable_hidden_slots,
)
from torchwright.compiler.utils import resolve_n_heads
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.forward.cpsat_scheduler import ScheduleAssignment
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import (
    HeldOutputLayout,
    ResidualStreamMap,
)
from torchwright.compiler.forward.sibling_clusters import SiblingClusters
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.graph import Node, Linear, Attn, Add, Concatenate
from torchwright.graph.misc import LiteralValue
from torchwright.graph.ffn import FFN


@dataclass
class _AttentionOp:
    """Mutable attention record used only during one scheduler walk."""

    op_type: Literal[
        "compute_attn",
        "compute_linear",
        "compute_add",
        "cancel",
        "add_into",
    ]
    node: Optional[Node]
    target_cols: List[int]
    source_cols: Optional[List[int]] = None
    source_cols_b: Optional[List[int]] = None
    q_source_cols: Optional[List[int]] = None
    k_source_cols: Optional[List[int]] = None


@dataclass
class _MlpOp:
    """Mutable MLP record used only during one scheduler walk."""

    op_type: Literal[
        "compute_ffn",
        "compute_literal_value",
        "compute_bias",
        "compute_linear_bypass",
        "cancel_bypass",
    ]
    node: Optional[Node]
    target_cols: List[int]
    mlp_slots: List[int] = field(default_factory=list)
    source_cols: Optional[List[int]] = None


@dataclass(frozen=True)
class SkipReason:
    """Why one ready node was not placed during one layer.

    Recorded at every point the layer walk passes over a node it could
    otherwise have scheduled, and folded into the ``No progress`` deadlock
    message.  Three distinct failures used to produce that one string with no
    way to tell them apart: MLP-slot over-demand, attention-head over-demand,
    and residual-column exhaustion.

    ``structural`` is the distinction that matters: the node's demand exceeds
    what a whole layer could ever supply, so no schedule places it and no
    amount of retrying helps.  A non-structural skip is an ordinary deadlock —
    this layer's budget is spent, or the residual stream is full right now.

    Among structural skips, ``misroute`` separates the two causes.  It is
    True only when the *selected* realization violates the structural
    eligibility bound its resolver was required to enforce — an
    MLP-sublayer class wider than a whole layer's usable hidden pool
    (``realization.fits_mlp`` exists to make that unreachable) — which is a
    compiler bug.  Merely having another candidate family is not enough:
    the alternative may be geometry-eliminated, or deliberately policy- or
    solver-fixed.  ``alternative`` records the other route family's demand
    and status so the deadlock message names both sides.
    """

    node: Node
    op_label: str  # the op that would have been emitted
    resource: str  # "attn heads" | "MLP hidden slots" | "residual columns"
    #   | "held output bank" (the tied target waiting for its source's
    #   columns to be cancelled into the held state)
    demand: int
    available: int  # free right now, this layer
    capacity: int  # the most a layer could ever supply
    misroute: bool  # the selected class violates resolver-enforced eligibility
    alternative: Optional[str] = None  # the other route family's demand/status

    @property
    def structural(self) -> bool:
        return self.demand > self.capacity

    def format_line(self) -> str:
        name = f" {self.node.name!r}" if getattr(self.node, "name", "") else ""
        head = (
            f"{type(self.node).__name__}{name} (id {self.node.node_id}) "
            f"via {self.op_label}: needs {self.demand} {self.resource}"
        )
        if not self.structural:
            line = f"{head}, {self.available} free this layer"
        else:
            line = f"{head}, but a layer holds at most {self.capacity}"
        if self.alternative:
            line += f" [{self.alternative}]"
        return line


class LayerScheduler:
    """Decides what operations to schedule in one transformer layer.

    Given the current residual-stream allocation and which nodes have
    been computed, picks attention and MLP operations for the next layer.

    Key concepts:
        - **dead node**: all downstream consumers are already computed,
          so its residual-stream columns can be reclaimed.
        - **pressure**: free columns are below 25% of ``d`` — the
          scheduler prioritises operations that free columns over those
          on the critical path.
        - **free add**: an ``Add`` where one input is dead — the dead
          input's columns are reused in-place (no allocation needed).
    """

    def __init__(
        self,
        graph: GraphAnalyzer,
        d: int,
        d_head: int,
        pos_encoding=None,
        d_hidden: Optional[int] = None,
        clusters: Optional[SiblingClusters] = None,
        admission_budget_fraction: float = 0.4,
        policy: Optional[SchedulingPolicy] = None,
        realization_table: Optional[RealizationTable] = None,
        bias: bool = True,
        held_output_layout: Optional[HeldOutputLayout] = None,
        n_heads: Optional[int] = None,
    ):
        self.graph = graph
        self.d = d
        self.d_hidden = d if d_hidden is None else d_hidden
        self.d_head = d_head
        # bias=False reserves hidden slot 0 for the constant lane (the
        # weight-writer's BiasFold): packing starts at slot 1, so the
        # effective per-layer capacity is d_hidden - 1.  Must agree with the
        # capacity CP-SAT models (forward_compile passes the solver
        # usable_hidden_slots(d_hidden, bias)) or a solver-feasible layer is
        # unpackable on replay.
        self.bias = bias
        self.usable_hidden_slots = usable_hidden_slots(self.d_hidden, bias)
        self.n_heads = resolve_n_heads(d, d_head, n_heads, require_divisible=False)
        self.pos_encoding = pos_encoding
        self.policy = policy if policy is not None else SchedulingPolicy()
        # The resolved realization table the walk reads (one artifact,
        # written by whichever resolver the optimize level provides — see
        # torchwright/compiler/realization.py).  When the caller doesn't
        # pass one (standalone scheduler construction in tests), resolve
        # statically from the policy through the same code path the
        # optimize=0 compile uses.
        if realization_table is None:
            all_nodes = graph.get_all_nodes()
            realization_table = RealizationTable.build(all_nodes).resolve_static(
                all_nodes, self.policy, self.usable_hidden_slots
            )
        self.realization_table = realization_table
        self.held_output_layout = held_output_layout
        # Eager (within-layer) freeing is always on: when a node is placed, free
        # any of its inputs that just became dead so their columns can be reused
        # *in the same layer*.  This is the within-layer column-reuse density
        # that Units 1 and 2 taught the CP-SAT model to represent (gap-0
        # attention cancels + MLP cancels), so the eager schedule is both the
        # production heuristic and the CP-SAT warm-start incumbent.  See
        # ``_freshly_dead_inputs``.
        # Why each ready node was passed over in the layer being scheduled;
        # rebuilt per pass, read only by ``_deadlock_message``.
        self._skips: List[SkipReason] = []

        # Admission control state (see _is_admissible).  When clusters
        # is None or empty, admission is disabled and the scheduler
        # behaves as it did before this feature.
        self._clusters = clusters
        self._admission_budget_fraction = admission_budget_fraction
        self._in_flight: Dict[int, Set[int]] = {}
        if clusters is not None:
            for cluster_id in clusters.clusters:
                self._in_flight[cluster_id] = set()

        # Layer index for diagnostics.  The heuristic walk never sets it;
        # ``DirectedLayerScheduler`` drives it via ``set_current_layer``.
        self._current_layer: Optional[int] = None

    def schedule_layer(
        self, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Tuple[List[_AttentionOp], List[_MlpOp], List[Node]]:
        """Schedule one transformer layer's worth of operations.

        Phases:
            1. Classify ready nodes (free adds, deferred adds, compute-ready).
            2. Attention sublayer: free adds, compute ops (Attn/Linear/Add),
               cancellations of dead nodes.
            3. MLP sublayer: FFNs (the L->ReLU->L composite), standalone
               Linears via MLP bypass, constants, bias writes for biased
               Linears.

        Mutates ``residual_map`` (allocate/free/reassign) and
        ``computed_nodes`` (add newly computed nodes).

        Returns:
            ``(attn_ops, mlp_ops, biased_linears)`` lists for the weight writer.
        """
        # Admission-control / literal-deferral bookkeeping for this call.
        self._admission_deferred = False
        self._literal_deferred = False
        self._admission_bypass = False

        attn_ops, mlp_ops, biased_linears, had_schedulable = self._schedule_layer_inner(
            residual_map, computed_nodes
        )

        # Deadlock guard: if the only thing that could have run was deferred —
        # either an admission-gated chain or a just-in-time-gated constant —
        # and nothing else was schedulable, retry with both gates bypassed.
        # Safe only when no state was mutated (no free_adds, no cancels, no
        # placements) — all of those append to attn_ops, so the emptiness
        # check is sufficient.  ``_admission_bypass`` lifts both gates
        # (admission and the literal JIT gate; see _literal_needed_now).
        if (
            not attn_ops
            and not mlp_ops
            and (self._admission_deferred or self._literal_deferred)
        ):
            self._admission_bypass = True
            attn_ops, mlp_ops, biased_linears, had_schedulable = (
                self._schedule_layer_inner(residual_map, computed_nodes)
            )

        # Progress check: raise only if nothing got placed despite ready
        # work existing (true deadlock).  Moved out of the inner function
        # so the admission retry runs first.
        if not attn_ops and not mlp_ops and had_schedulable:
            remaining = self.graph.get_all_nodes() - computed_nodes
            remaining = {n for n in remaining if not isinstance(n, Concatenate)}
            if remaining:
                raise RuntimeError(self._deadlock_message(remaining, residual_map))

        return attn_ops, mlp_ops, biased_linears

    def _deadlock_message(
        self, remaining: Set[Node], residual_map: ResidualStreamMap
    ) -> str:
        """The ``No progress`` text, itemized by why each ready node was
        passed over.

        The ``No progress:`` prefix and the ``RuntimeError`` type are a
        contract: ``test_optimize1_compiles_where_eager_heuristic_cannot``
        matches on the string, and that test's deadlock (residual-column
        exhaustion at d=48) is a legitimate one the directed replay resolves.
        """
        lines = [
            f"No progress: {len(remaining)} nodes remaining, "
            f"{residual_map.get_free_count()} free columns."
        ]
        if not self._skips:
            lines.append(
                "  No ready node recorded a skip reason — the deadlock is in "
                "the layer walk's bookkeeping, not in a capacity check."
            )
            return "\n".join(lines)

        # One line per node, keeping its first (most specific) skip.  A node
        # skipped for slots in the MLP pass is not also reported for columns.
        seen: Set[int] = set()
        misrouted, too_big, transient = [], [], []
        for skip in self._skips:
            if skip.node.node_id in seen:
                continue
            seen.add(skip.node.node_id)
            if not skip.structural:
                transient.append(skip)
            elif skip.misroute:
                misrouted.append(skip)
            else:
                too_big.append(skip)

        def section(header: str, skips: List[SkipReason], limit: int = 8) -> None:
            if not skips:
                return
            lines.append(header)
            lines.extend(f"    {s.format_line()}" for s in skips[:limit])
            if len(skips) > limit:
                lines.append(f"    ... and {len(skips) - limit} more like these")

        # Structural skips are never truncated: each one is a distinct
        # unschedulable node, and the first is the one to act on.
        section(
            "  COMPILER BUG — these nodes' resolved realization exceeds the "
            "structural bound its resolver enforces "
            "(realization.fits_mlp should have made this unreachable):",
            misrouted,
            limit=len(misrouted),
        )
        section(
            f"  Too big for the geometry — no fitting realization remains "
            f"for these nodes at d={self.d}, d_hidden={self.d_hidden}, "
            f"d_head={self.d_head} (an alternative route's demand and "
            f"status, where one exists, is bracketed). No schedule exists; "
            f"the graph or the geometry has to change:",
            too_big,
            limit=len(too_big),
        )
        section(
            "  Out of budget this layer — a legitimate deadlock at this "
            "geometry (optimize>0 may find a schedule the eager walk "
            "cannot):",
            transient,
        )
        return "\n".join(lines)

    def _record_skip(
        self,
        node: Node,
        op_label: str,
        resource: str,
        demand: int,
        available: int,
        capacity: int,
    ) -> None:
        # A structural skip is a resolver misroute only when the selected
        # realization violates the bound its resolver had to enforce: an
        # MLP-sublayer class needs its whole-layer slot demand to fit the
        # usable pool (realization.fits_mlp).  A structural attention skip
        # is a geometry/policy failure, never a misroute — no resolver
        # enforces a per-node head bound.
        misroute = (
            demand > capacity
            and resource == "MLP hidden slots"
            and has_flex_choice(node)
        )
        self._skips.append(
            SkipReason(
                node,
                op_label,
                resource,
                demand,
                available,
                capacity,
                misroute=misroute,
                alternative=self._route_alternative_note(node),
            )
        )

    def _route_alternative_note(self, node: Node) -> Optional[str]:
        """The other route family's demand and status, for skip diagnostics.

        ``None`` for nodes without a route choice.  Never consults
        liveness: an attention Add's head demand is reported as its
        fresh/reused range because residual placement is not known at skip
        time.
        """
        if not has_flex_choice(node):
            return None
        entry = self.realization_table.entries.get(node.node_id)
        selected = entry.resolved if entry is not None else None
        if selected is None:
            return None
        # The note states the alternative's demand and fit only; WHY it was
        # not selected (geometry elimination, policy, solver choice, or a
        # resolver bug) is the deadlock message's section classification.
        attn_cls, mlp_cls = flex_route_classes(node)
        if selected == attn_cls:
            slots = class_hidden_slots(node, mlp_cls)
            if slots > self.usable_hidden_slots:
                return (
                    f"alternative {mlp_cls} needs {slots} hidden slots, over "
                    f"a layer's usable {self.usable_hidden_slots}"
                )
            return (
                f"alternative {mlp_cls} fits: {slots} of "
                f"{self.usable_hidden_slots} usable hidden slots"
            )
        if isinstance(node, Add):
            lo = add_attn_heads(node, self.d_head, reused=True)
            hi = add_attn_heads(node, self.d_head, reused=False)
            demand = f"{lo}..{hi} heads"
            over = lo > self.n_heads
        else:
            h = linear_attn_heads(node, self.d_head)
            demand = f"{h} heads"
            over = h > self.n_heads
        if over:
            return (
                f"alternative {attn_cls} needs {demand}, over "
                f"n_heads={self.n_heads}"
            )
        return f"alternative {attn_cls} fits: {demand} within n_heads={self.n_heads}"

    def _schedule_layer_inner(
        self, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Tuple[List[_AttentionOp], List[_MlpOp], List[Node], bool]:
        # Skip reasons for THIS pass only.  ``schedule_layer`` may call this
        # twice (the admission-bypass retry), and it is the retry's skips that
        # explain a deadlock — the first pass's were provisional.
        self._skips: List[SkipReason] = []
        # Nodes already computed before this layer.  A node born THIS layer must
        # not be MLP-cancelled this layer even if a same-layer MLP consumer
        # makes it dead: the model requires every cancel at birth+1 or later
        # (`cancel_layer >= layer + 1`), so freeing at birth would emit a hint
        # the model cannot represent and a degenerate zero-length residual
        # liveness window.  The directed replay is already protected by its
        # `cancel_layer <= current` gate (the model never assigns cancel ==
        # birth); this guards the eager heuristic's own choice.
        computed_before_layer = set(computed_nodes)
        # --- 1. Classify ready nodes ---
        all_ready = self._get_ready_nodes(computed_nodes)

        ready = set()
        free_adds = []
        deferred_adds = []
        # Iterate node sets in node_id order wherever the iteration
        # order can reach the schedule (list build order feeds stable
        # sorts and append-order scheduling below) — set iteration is
        # keyed on absolute id values and does not survive a rebuild.
        for node in sorted(all_ready, key=lambda n: n.node_id):
            if isinstance(node, Add):
                if self._is_forced_fresh_add(node):
                    deferred_adds.append(node)
                    continue
                a0, a1 = node.inputs
                d0 = self._is_dead_for_add(a0, node, computed_nodes)
                d1 = self._is_dead_for_add(a1, node, computed_nodes)
                if d0 or d1:
                    free_adds.append(node)
                else:
                    deferred_adds.append(node)
            elif isinstance(node, (Attn, Linear, LiteralValue, FFN)):
                ready.add(node)
            # else: skip unschedulable source nodes (InputNode, Embedding, etc.)

        # Layer-start deadness snapshot for the legacy attention-cancel paths
        # (placement-time promotion and the end-of-sublayer leftover loop).
        # Must be taken here, before the attention sublayer's free-Add
        # placements mutate state — the attention cancel path relies on
        # layer-start deadness to never see a keep-forever node (see
        # ``_is_keep_forever``).  The directed replay never consumes it (its
        # atomic batch is the single definition of the attention transition),
        # so it skips the O(allocated x consumers) scan.
        dead = (
            []
            if self._uses_atomic_attention_batch()
            else self._find_dead_nodes(residual_map, computed_nodes)
        )

        had_schedulable = bool(ready) or bool(free_adds) or bool(deferred_adds)

        # --- 2. Attention sublayer ---
        # An FFN reads its input's residual columns inline in linear1, but the
        # FFN is a normal ready node and an uncomputed consumer of that input
        # until it is scheduled in the MLP sublayer, so the input is never
        # "freshly dead" during the attention sublayer and needs no special
        # protection (the analogue of the chain's old ``chain_protected`` set).
        (
            attn_ops,
            biased_linears,
            bypass_linears,
            cancel_cols,
            add_into_live_addends,
        ) = self._schedule_attn_sublayer(
            ready,
            dead,
            free_adds,
            deferred_adds,
            residual_map,
            computed_nodes,
        )

        # --- 2.5. Re-check readiness after attention ---
        # Nodes computed by attention may unlock MLP-eligible nodes in the same
        # layer (the MLP sublayer reads x + attn(x), so it sees attention results).
        newly_ready = self._get_ready_nodes(computed_nodes) - all_ready
        for node in newly_ready:
            if isinstance(node, Add):
                if self._is_forced_fresh_add(node):
                    continue  # attention pass is over; retry fresh next layer
                a0, a1 = node.inputs
                d0 = self._is_dead_for_add(a0, node, computed_nodes)
                d1 = self._is_dead_for_add(a1, node, computed_nodes)
                if not (d0 or d1):
                    continue  # deferred add, skip
            if isinstance(node, (Linear, LiteralValue, FFN)):
                ready.add(node)

        # Newly-ready MLP-routed Linears: extend bypass_linears so the
        # MLP sublayer picks them up this layer.  bypass_linears was
        # built before the attn pass ran, so any Linear whose inputs
        # only became available after attn (an attn->mlp same-layer
        # dependency) was missed.  Without this catch-up the Linear is
        # deferred a layer — fine for the heuristic, but the CP-SAT
        # model's same_layer_ok rule (docs/cpsat_scheduler.md §3 Dependency
        # constraints) assumes the same-layer placement is realised.
        bypass_set = set(bypass_linears)
        for node in sorted(ready, key=lambda n: n.node_id):
            if not isinstance(node, Linear) or node in bypass_set:
                continue
            if not self.realization_table.is_attention_routed(node):
                bypass_linears.append(node)
                bypass_set.add(node)

        # Dead nodes the attention sublayer did not cancel (its head budget was
        # full, or the directed replay routed them to the MLP mechanism) are
        # offered to the MLP sublayer, which zeroes them via `cancel_bypass`
        # while hidden slots remain — the fall-back-to-MLP half of the
        # mechanism-choice policy.  Recomputed against the CURRENT computed set
        # (not the layer-start `dead`) so a node that died during THIS layer's
        # attention sublayer — its last consumer an attention op here — is
        # MLP-cancelled this layer rather than a layer late, which the directed
        # replay's `[layer, cancel + 1)` residual budget would otherwise
        # overflow.  `_find_dead_nodes` returns allocated, dead nodes (and, on
        # the directed replay, only those whose assigned cancel layer has
        # arrived); the mechanism filter keeps a directed attn-mech leftover
        # deferred rather than silently MLP-cancelling it.
        defer_live = self._mlp_cancel_defers_live_addends()
        mlp_cancel_candidates = [
            n
            for n in self._find_dead_nodes(residual_map, computed_nodes)
            if self._mlp_cancel_eligible(n)
            and not self._is_keep_forever(n)
            and not self.graph.is_input_node(n)  # decision #3: inputs stay
            # resident for snapshot-based value lookup — never MLP-cancelled
            # (the directed replay agrees: inputs carry no cancel-mech entry).
            and n in computed_before_layer  # never cancel at birth (>= birth+1)
            and not (defer_live and n in add_into_live_addends)
        ]

        # --- 3. MLP sublayer ---
        mlp_ops = self._schedule_mlp_sublayer(
            ready,
            bypass_linears,
            biased_linears,
            residual_map,
            computed_nodes,
            mlp_cancel_candidates,
            computed_before_layer,
        )

        # Emit the single batched death-cancel op (built in the attention
        # sublayer) at the end of the layer.  Order within the sublayer is
        # irrelevant (all heads run in parallel and sum into the residual
        # stream), so it's fine to append after compute ops.
        if cancel_cols:
            attn_ops.append(_AttentionOp("cancel", None, cancel_cols))

        # Caller (schedule_layer) handles the progress check after the
        # admission-retry pass.
        return attn_ops, mlp_ops, biased_linears, had_schedulable

    # ------------------------------------------------------------------
    # Attention sublayer
    # ------------------------------------------------------------------

    def _schedule_attn_sublayer(
        self,
        ready,
        dead,
        free_adds,
        deferred_adds,
        residual_map,
        computed_nodes,
    ):
        attn_ops = []
        biased_linears = []
        heads_used = 0

        # The dead-node cancels in this layer (zeroing a dying node's columns
        # so they can be reused) are batched into a single
        # _AttentionOp("cancel", None, cancel_cols) emitted at the end.
        # Coalescing matters: one cancel head can zero d_head cols, so
        # scattering one cancel op per write-site burns heads that would
        # otherwise be shared.  ``heads_used`` tracks main-op heads
        # *plus* the current batched-cancel cost (ceil(|cancel_cols|/d_head)).
        cancel_cols: list[int] = []
        cancel_cols_set: set[int] = set()
        cancel_heads = 0

        def try_add_cancel(new_cols):
            """Try to add ``new_cols`` to the pending cancel batch.

            Returns ``(additions, delta_heads)`` if the merged cancel fits
            in the remaining head budget; ``None`` otherwise.  ``additions``
            is the subset of ``new_cols`` not already in the batch.
            Does NOT commit — the caller decides whether to keep or
            discard.
            """
            additions = [c for c in new_cols if c not in cancel_cols_set]
            if not additions:
                return [], 0
            new_total = len(cancel_cols) + len(additions)
            new_heads = (new_total + self.d_head - 1) // self.d_head
            delta = new_heads - cancel_heads
            if heads_used + delta > self.n_heads:
                return None
            return additions, delta

        def commit_cancel(additions, delta):
            nonlocal heads_used, cancel_heads
            if additions:
                cancel_cols.extend(additions)
                cancel_cols_set.update(additions)
                cancel_heads += delta
                heads_used += delta

        # 2a. Free Adds (highest priority — no allocation needed)
        # Snapshot computed_nodes so dead-for-add checks are consistent across
        # the entire batch. Without this, earlier add_into ops add their Add to
        # computed_nodes, which can flip a shared node from "live" to "dead" on
        # a later iteration — reassigning its columns and orphaning earlier ops.
        computed_snapshot = set(computed_nodes)
        add_into_live_addends = set()
        for add_node in sorted(free_adds, key=self._critical_path_key):
            a0, a1 = add_node.inputs
            d0 = self._is_dead_for_add(a0, add_node, computed_snapshot)
            d1 = self._is_dead_for_add(a1, add_node, computed_snapshot)
            dead_addend = a0 if d0 else a1
            live_addend = a1 if d0 else a0
            n_heads = (len(live_addend) + self.d_head - 1) // self.d_head
            if heads_used + n_heads > self.n_heads:
                # The Add's route family is already resolved (attention); a
                # structural skip here is terminal for this compile — the MLP
                # family was geometry-eliminated or policy-/solver-fixed away,
                # and the skip's alternative note records which.
                self._record_skip(
                    add_node,
                    "add_into",
                    "attn heads",
                    n_heads,
                    self.n_heads - heads_used,
                    self.n_heads,
                )
                continue
            self._require_live(
                dead_addend,
                residual_map,
                f"add_into dead-addend for {add_node!r}",
            )
            self._require_live(
                live_addend,
                residual_map,
                f"add_into live-addend for {add_node!r}",
            )
            target_cols = residual_map.get_indices(dead_addend)
            live_source_cols = residual_map.resolve_indices(live_addend)
            attn_ops.append(
                _AttentionOp(
                    "add_into",
                    add_node,
                    target_cols,
                    source_cols=live_source_cols,
                )
            )
            residual_map.reassign(dead_addend, add_node)
            computed_nodes.add(add_node)
            add_into_live_addends.add(live_addend)
            heads_used += n_heads

        # Build compute candidates: Attn nodes, standalone Linears, deferred Adds.
        # When the routing hook says MLP for a Linear, it's skipped here and
        # scheduled via bypass in _schedule_mlp_sublayer.
        compute_candidates = []
        bypass_linears: list[Node] = []
        for node in sorted(ready, key=lambda n: n.node_id):
            if isinstance(node, Attn):
                n_heads = (node.d_v + self.d_head - 1) // self.d_head
                compute_candidates.append(("compute_attn", node, n_heads))
            elif isinstance(node, Linear):
                if not self.realization_table.is_attention_routed(node):
                    bypass_linears.append(node)
                    continue
                n_heads = self._heads_for_linear(node)
                compute_candidates.append(("compute_linear", node, n_heads))
        # Deferred Adds: neither input is dead, so we can't use add_into.
        # Instead, copy both inputs to fresh columns via attention heads.
        for node in deferred_adds:
            n_heads = self._heads_for_add(node)
            compute_candidates.append(("compute_add", node, n_heads))

        # Sort: Attn first; under column pressure prefer nodes that free columns,
        # otherwise maximize parallelism via critical path.
        under_pressure = residual_map.get_free_count() < self.d * (
            1.0 - self.policy.pressure_threshold
        )
        if under_pressure:
            compute_candidates.sort(
                key=lambda t: (
                    0 if t[0] == "compute_attn" else 1,
                    self._net_column_cost(t[1], computed_nodes, residual_map),
                    *self._critical_path_key(t[1]),
                )
            )
        else:
            compute_candidates.sort(
                key=lambda t: (
                    0 if t[0] == "compute_attn" else 1,
                    *self._critical_path_key(t[1]),
                )
            )

        # --- Atomic attention batch transition (directed replay) -----------
        # The solver-assigned attention sublayer is one physical
        # read/cancel/write event: every head reads the pre-attention
        # residual and the deltas sum, so replay may capture every source,
        # release every assigned cancel, then place every output — no
        # per-output bootstrap order is required, and the residual model only
        # guarantees the aggregate transition exists, never such an ordering
        # (docs/cpsat_atomic_attention_replay_plan.md).  A non-None hook
        # return is the complete release set; the legacy dead-list cancel
        # paths are bypassed below.
        batch_releases = self._assigned_attention_releases(
            [node for _, node, _ in compute_candidates],
            residual_map,
            computed_nodes,
        )
        batch_sources: Optional[Dict[Node, Dict[str, List[int]]]] = None
        if batch_releases is not None:
            layer_label = self._current_layer
            # A1: capture every batch member's sources while all inputs are
            # still allocated, before any release commits.  This is compiler
            # invariant I4's schedule-time check relocated with the capture
            # site — the placement loop consumes these saved captures and
            # must not re-run the liveness lookup after a legitimate release.
            batch_sources = {
                node: self._capture_attn_sources(op_type, node, residual_map)
                for op_type, node, _ in compute_candidates
            }
            release_cols = {
                u: list(residual_map.get_indices(u)) for u in batch_releases
            }

            # A3: exact attention charge, before mutation.  Free-Add heads
            # (already in heads_used) + every compute candidate's heads + the
            # coalesced cancel, charged once over the distinct release
            # columns.  This is equivalent to the CP-SAT attention cumulative
            # (per-op compute charges agree exactly; the pooled column-unit
            # cancel charge rounds to the same head count), so a
            # model-feasible assignment can never fail here — a failure is a
            # model/replay contract violation, never a legitimate deferral.
            distinct_cancel_cols: Set[int] = set()
            for cols in release_cols.values():
                distinct_cancel_cols.update(cols)
            compute_heads = sum(h for _, _, h in compute_candidates)
            cancel_head_charge = (
                len(distinct_cancel_cols) + self.d_head - 1
            ) // self.d_head
            if heads_used + compute_heads + cancel_head_charge > self.n_heads:
                raise AssertionError(
                    f"Model/replay contract violation (A3) at layer "
                    f"{layer_label}: attention head charge {heads_used} "
                    f"(free Adds) + {compute_heads} (compute) + "
                    f"{cancel_head_charge} (coalesced cancel of "
                    f"{len(distinct_cancel_cols)} cols) exceeds "
                    f"n_heads={self.n_heads}."
                )

            # A4: aggregate residual preflight over ORDINARY columns, before
            # mutation.  The held bank and its exact target are excluded from
            # both sides: hold(source) does not make the bank generally free,
            # and allocate_at(target, bank) claims the held bank without
            # consuming ordinary columns.  The held claim is checked directly
            # on the intended transition — the bank is already held from an
            # earlier layer, or the held source is in this batch's release
            # set — because can_allocate_at would false-negative here (the
            # preflight runs before the release commits the bank to the held
            # state).  Aggregate width suffices: the allocator takes
            # arbitrary noncontiguous columns, so there is no fragmentation
            # condition.
            layout = self.held_output_layout
            held_release_width = 0
            if layout is not None and layout.source in release_cols:
                held_release_width = len(release_cols[layout.source])
            ordinary_release_width = (
                sum(len(cols) for cols in release_cols.values()) - held_release_width
            )
            ordinary_demand = 0
            for _, node, _ in compute_candidates:
                if layout is not None and node is layout.target:
                    bank_already_held = residual_map.has_held()
                    if not (bank_already_held or layout.source in release_cols):
                        raise AssertionError(
                            f"Model/replay contract violation (A4) at layer "
                            f"{layer_label}: held target {node!r} is in the "
                            f"attention batch but its bank is neither already "
                            f"held nor released by this batch (held source "
                            f"{layout.source!r})."
                        )
                else:
                    ordinary_demand += len(node)
            ordinary_free = residual_map.get_free_count() + ordinary_release_width
            if ordinary_demand > ordinary_free:
                raise AssertionError(
                    f"Model/replay contract violation (A4) at layer "
                    f"{layer_label}: attention batch needs {ordinary_demand} "
                    f"ordinary residual columns but only {ordinary_free} are "
                    f"available ({residual_map.get_free_count()} free on "
                    f"entry + {ordinary_release_width} released)."
                )

            # Commit: the release columns join the single coalesced cancel op
            # and ownership transitions (ordinary values become free; a held
            # source becomes held).  A3 already covered the merged charge, so
            # the batch always fits the remaining head budget.
            result = try_add_cancel(sorted(distinct_cancel_cols))
            assert result is not None, (
                "A3 preflight passed but the coalesced cancel batch does not "
                "fit the head budget — the two charges diverged."
            )
            commit_cancel(*result)
            for u in batch_releases:
                self._release_cancelled(u, residual_map, mech="attn")

        # Cancellation candidates (exclude live addends of add_into ops, and
        # dead nodes the directed replay routed to the MLP mechanism — those
        # are cancelled in the MLP sublayer, never here).  Under an atomic
        # batch the list stays EMPTY: the batch is the single definition of
        # the attention transition, so the placement-time cancel promotion
        # and the end-of-sublayer leftover loop below never run — a value
        # dead at layer entry is already in the batch's release set, and
        # re-processing it here would double-charge or KeyError on its freed
        # columns.
        if batch_releases is None:
            cancel_candidates = [
                n
                for n in dead
                if n is not self.pos_encoding
                and n not in add_into_live_addends
                and self._attn_cancel_eligible(n)
            ]
            cancel_candidates.sort(key=lambda n: (-len(n), n.node_id))  # largest first
        else:
            cancel_candidates = []

        # 2b-2d. Schedule compute ops with cancellation promotion
        def _try_place(op_type, node, n_heads_needed) -> bool:
            """Place one compute candidate; return True if placed, False if
            deferred (over budget, inadmissible, or no columns available)."""
            nonlocal heads_used
            if heads_used + n_heads_needed > self.n_heads:
                # Recorded, but believed unreachable as a *deadlock* cause: a
                # committed cancel or any placement makes attn_ops non-empty
                # (the batched "cancel" op below), so at a No-progress raise
                # heads_used is 0 and this fires only when the node alone
                # exceeds the whole head budget.  Every op's head demand is
                # bounded by its residual footprint (compute_linear reads
                # d_input <= d columns; compute_attn reads d_v <= d; a deferred
                # Add costs ~2*ceil(w/d_head) heads but needs both w-wide
                # addends live, so 2w <= d caps it at the budget).  If this
                # line ever appears in a deadlock message, that reasoning is
                # wrong and the message is how you find out.
                self._record_skip(
                    node,
                    op_type,
                    "attn heads",
                    n_heads_needed,
                    self.n_heads - heads_used,
                    self.n_heads,
                )
                return False
            if not self._is_admissible(node):
                self._admission_deferred = True
                return False

            # Capture source columns at schedule time, BEFORE allocating this
            # op's target.  The weight-writer reads sources from the op
            # directly, so later free()s (eager freeing, or the held-bank
            # handoff below) don't orphan the lookups.  ``_require_live`` runs
            # inside the capture while every input is still allocated (I4),
            # so the capture also holds when a dying input is freed and its
            # own columns are reused for this op's output.  Under an atomic
            # batch the capture already happened before the batch's releases
            # committed — consume it; re-running the liveness lookup on a
            # legitimately released input would be a wrong duplicate.
            sources = (
                batch_sources[node]
                if batch_sources is not None
                else self._capture_attn_sources(op_type, node, residual_map)
            )

            target_cols = self._try_allocate(node, residual_map)

            # Promotion: cancel dead nodes to free space.  The dead
            # node's cols are added to the batched cancel set.
            while (
                target_cols is None
                and cancel_candidates
                and heads_used + n_heads_needed < self.n_heads
            ):
                cn = cancel_candidates[0]
                cn_cols = residual_map.get_indices(cn)
                result = try_add_cancel(cn_cols)
                if result is None:
                    break
                additions, delta = result
                if heads_used + n_heads_needed + delta > self.n_heads:
                    break
                cancel_candidates.pop(0)
                commit_cancel(additions, delta)
                self._release_cancelled(cn, residual_map, mech="attn")
                target_cols = self._try_allocate(node, residual_map)

            # Held-bank handoff (heuristic only): placing ``node`` may make
            # the registered tied source dead (node is its last consumer).
            # Cancel and hold the dying source now (its value was captured
            # for node's source above), then claim the held bank for node's
            # own output.  Order — capture, cancel, hold, allocate — keeps I1
            # (release precedes allocate) and I4 (require_live ran while the
            # source was live).  Inert on the directed replay: the atomic
            # batch already released everything assigned here, so the lookup
            # below finds nothing allocated.
            if target_cols is None:
                reuse = self._dying_input_to_reuse(node, residual_map, computed_nodes)
                if reuse is not None and reuse in add_into_live_addends:
                    # Same-layer cancel of an add_into live addend — excluded
                    # on every other cancel path (the cancel-candidate and
                    # freshly-dead filters) and forbidden by the model
                    # (`cl >= layer[A] + is_free[A]`).  Unreachable through
                    # schedule_layer today (a free Add consuming the source
                    # is an ancestor of the single-output target, so the two
                    # never share a layer-start ready set); guarded for
                    # parity and against future placement-rule changes.
                    reuse = None
                if reuse is not None:
                    result = try_add_cancel(residual_map.get_indices(reuse))
                    if result is not None:
                        additions, delta = result
                        if heads_used + n_heads_needed + delta <= self.n_heads:
                            commit_cancel(additions, delta)
                            self._release_cancelled(reuse, residual_map, mech="attn")
                            target_cols = self._try_allocate(node, residual_map)

            if target_cols is None:
                if batch_sources is not None:
                    # The batch preflight (A4) proved the aggregate transition
                    # fits, so after the committed releases every directed
                    # attention target must allocate.  Reaching this line
                    # means the preflight and the allocator disagree — an
                    # internal bug, never a normal deferral.
                    raise AssertionError(
                        f"Model/replay contract violation (post-preflight "
                        f"allocation) at layer "
                        f"{self._current_layer}: batch "
                        f"member {node!r} failed to allocate "
                        f"{len(node)} columns with "
                        f"{residual_map.get_free_count()} free after the "
                        f"batch released its cancels "
                        f"(held bank: {residual_map.held_columns() or None})."
                    )
                layout = self.held_output_layout
                if layout is not None and node is layout.target:
                    # The held target allocates only via the complete held
                    # bank (``can_allocate_at``), so this failure means the
                    # bank is not yet held — the tied source is still live.
                    # Free ordinary columns are irrelevant; reporting them
                    # would claim demand <= available on a skipped node.
                    self._record_skip(
                        node,
                        op_type,
                        "held output bank",
                        len(node),
                        0,
                        len(layout.bank),
                    )
                    return False
                # Every cancel-promotion and held-handoff escape above has
                # been tried; the residual stream simply has no room.
                self._record_skip(
                    node,
                    op_type,
                    "residual columns",
                    len(node),
                    residual_map.get_free_count(),
                    self.d,
                )
                return False

            op = _AttentionOp(op_type, node, target_cols)
            for attr, cols in sources.items():
                setattr(op, attr, cols)
            attn_ops.append(op)
            heads_used += n_heads_needed
            computed_nodes.add(node)
            ready.discard(node)
            self._mark_scheduled(node)

            # Eager-freeing: scheduling ``node`` may have just made one
            # of its inputs freshly dead.  Surface those to
            # ``cancel_candidates`` so subsequent compute iterations can
            # promote-cancel them instead of aborting on a full residual
            # stream.  Safe because sources for the just-appended op were
            # captured on ``op`` above, so weight-writer lookups don't
            # depend on the input staying in residual_map.
            already_pending = set(cancel_candidates)
            for fresh in self._freshly_dead_inputs(node, computed_nodes, residual_map):
                if fresh in add_into_live_addends or fresh in already_pending:
                    continue
                if not self._attn_cancel_eligible(fresh):
                    # Routed to the MLP mechanism (directed replay) — the MLP
                    # sublayer picks it up from `dead`; never attention-cancel.
                    continue
                cancel_candidates.append(fresh)
                already_pending.add(fresh)
            cancel_candidates.sort(key=lambda n: (-len(n), n.node_id))

            if (
                op_type == "compute_linear"
                and isinstance(node, Linear)
                and not self._has_zero_bias(node)
            ):
                biased_linears.append(node)
            return True

        # Single order-preserving candidate pass.  The directed replay needs
        # no within-layer retry: every assigned same-layer attention release
        # was committed by the atomic batch above before placement began, so
        # no candidate's allocation depends on another candidate placing
        # first.
        for cand in compute_candidates:
            _try_place(*cand)

        # 2e. Remaining cancellations — try to fold remaining dead cols
        # into the same batch.
        for cn in cancel_candidates:
            cn_cols = residual_map.get_indices(cn)
            result = try_add_cancel(cn_cols)
            if result is None:
                continue
            additions, delta = result
            commit_cancel(additions, delta)
            self._release_cancelled(cn, residual_map, mech="attn")

        # The batched death-cancel op is emitted by the caller.  The MLP
        # sublayer does not extend it (fresh MLP allocations need no cancel
        # under universal zero-init), so only the column list is returned.
        # ``add_into_live_addends`` is returned so the MLP-cancel pass can
        # leave this layer's live addends alone (they die next layer).
        return (
            attn_ops,
            biased_linears,
            bypass_linears,
            cancel_cols,
            add_into_live_addends,
        )

    # ------------------------------------------------------------------
    # MLP sublayer
    # ------------------------------------------------------------------

    def _schedule_mlp_sublayer(
        self,
        ready,
        bypass_linears,
        biased_linears,
        residual_map,
        computed_nodes,
        mlp_cancel_candidates=None,
        computed_before_layer=None,
    ):
        mlp_ops = []
        # Slot 0 is the constant lane under bias=False — see LayerScheduler
        # __init__ and weight_writer.BiasFold.
        next_slot = first_hidden_slot(self.bias)

        # Dead nodes to zero via `cancel_bypass` this layer (attention batch was
        # full, or the directed replay routed them here).  MLP-phase placements
        # below may make more inputs freshly dead; those are appended and share
        # the same slot budget.  The batch is emitted AFTER the compute ops so
        # the freed columns are not reused within this layer — matching the
        # `[layer, cancel + 1)` residual accounting the MLP-cancel model uses
        # (the columns stay live through the whole cancel layer, freed at its
        # end).  Fresh MLP compute allocations still need no BIRTH cancel:
        # universal zero-init leaves them clean.
        mlp_cancel_candidates = list(mlp_cancel_candidates or [])
        mlp_cancel_pending = set(mlp_cancel_candidates)
        computed_before_layer = (
            computed_before_layer if computed_before_layer is not None else set()
        )

        def _surface_mlp_freshly_dead(placed_node):
            # After placing an MLP compute op, its operands may be freshly dead.
            # Offer the MLP-eligible ones to the cancel_bypass batch (their
            # values were captured on the op above, so freeing them is safe).
            # ``_freshly_dead_inputs`` returns [] when eager freeing is off (the
            # no-eager warm-start), so this is inert there.  A node born THIS
            # layer is skipped — no cancel may fire at its own birth layer (the
            # model's `cancel_layer >= layer + 1`); it frees next layer.
            for fresh in self._freshly_dead_inputs(
                placed_node, computed_nodes, residual_map
            ):
                if fresh in mlp_cancel_pending:
                    continue
                if fresh not in computed_before_layer:
                    continue
                if not self._mlp_cancel_eligible(fresh):
                    continue
                mlp_cancel_candidates.append(fresh)
                mlp_cancel_pending.add(fresh)

        # Residual-pressure flag shared by the FFN and bypass-Linear passes:
        # under pressure they sort by net column cost (free columns first) to
        # relieve the residual stream; otherwise by critical path.
        under_pressure = residual_map.get_free_count() < self.d * (
            1.0 - self.policy.pressure_threshold
        )

        # 3a. FFNs (the first-class L->ReLU->L MLP composite).
        # An FFN allocates its d_output residual cols, claims n_lanes hidden
        # slots, and reads its input's residual cols inline in linear1.  It is
        # always realized whole (Gate A); its input projection is exclusive by
        # construction (the gate rows are the FFN's own weights).
        ffns = [n for n in ready if isinstance(n, FFN)]
        if under_pressure:
            ffns.sort(
                key=lambda b: (
                    self._net_column_cost(b, computed_nodes, residual_map),
                    self._critical_path_key(b),
                )
            )
        else:
            ffns.sort(key=self._critical_path_key)
        for ffn in ffns:
            n_lanes = ffn.n_lanes
            if next_slot + n_lanes > self.d_hidden:
                # MLP_COMPOSITE is an FFN's only realization class, so an FFN
                # with more lanes than a layer's usable pool is unschedulable
                # at this d_hidden — no routing rule can rescue it.
                self._record_skip(
                    ffn,
                    "compute_ffn",
                    "MLP hidden slots",
                    n_lanes,
                    self.d_hidden - next_slot,
                    self.usable_hidden_slots,
                )
                continue
            if not self._is_admissible(ffn):
                self._admission_deferred = True
                continue
            target_cols = self._try_allocate(ffn, residual_map)
            if target_cols is None:
                self._record_skip(
                    ffn,
                    "compute_ffn",
                    "residual columns",
                    len(ffn),
                    residual_map.get_free_count(),
                    self.d,
                )
                continue
            mlp_slots = list(range(next_slot, next_slot + n_lanes))
            next_slot += n_lanes
            self._require_live(
                ffn.inputs[0],
                residual_map,
                f"compute_ffn input for {ffn!r}",
            )
            input_cols = residual_map.resolve_indices(ffn.inputs[0])
            mlp_ops.append(
                _MlpOp(
                    "compute_ffn",
                    ffn,
                    target_cols,
                    mlp_slots,
                    source_cols=input_cols,
                )
            )
            computed_nodes.add(ffn)
            ready.discard(ffn)
            self._mark_scheduled(ffn)
            _surface_mlp_freshly_dead(ffn)

        # 3b. Standalone Linears via MLP bypass (ReLU bypass trick).
        # These were skipped in the attention phase because the policy
        # routes them to MLP.  Each output column needs 2 MLP slots.
        if bypass_linears:
            sorted_bypass = sorted(
                bypass_linears,
                key=(
                    (
                        lambda n: (
                            self._net_column_cost(n, computed_nodes, residual_map),
                            self._critical_path_key(n),
                        )
                    )
                    if under_pressure
                    else self._critical_path_key
                ),
            )
            for node in sorted_bypass:
                assert isinstance(node, Linear)
                if node in computed_nodes:
                    continue
                n_slots = 2 * node.d_output
                if next_slot + n_slots > self.d_hidden:
                    # A structural skip here means the node was routed to the
                    # MLP bypass despite not fitting in one — the capacity
                    # check in `realization.static_flex_class` failed to fire,
                    # or the solve's assignment contradicts it.
                    self._record_skip(
                        node,
                        "compute_linear_bypass",
                        "MLP hidden slots",
                        n_slots,
                        self.d_hidden - next_slot,
                        self.usable_hidden_slots,
                    )
                    continue
                if not self._is_admissible(node):
                    self._admission_deferred = True
                    continue
                target_cols = self._try_allocate(node, residual_map)
                if target_cols is None:
                    self._record_skip(
                        node,
                        "compute_linear_bypass",
                        "residual columns",
                        len(node),
                        residual_map.get_free_count(),
                        self.d,
                    )
                    continue
                mlp_slots = list(range(next_slot, next_slot + n_slots))
                next_slot += n_slots
                self._require_live(
                    node.inputs[0],
                    residual_map,
                    f"compute_linear_bypass input for {node!r}",
                )
                input_cols = residual_map.resolve_indices(node.inputs[0])
                mlp_ops.append(
                    _MlpOp(
                        "compute_linear_bypass",
                        node,
                        target_cols,
                        mlp_slots,
                        source_cols=input_cols,
                    )
                )
                computed_nodes.add(node)
                ready.discard(node)
                self._mark_scheduled(node)
                _surface_mlp_freshly_dead(node)

        # 3c. LiteralValues (no slot cost)
        constants = sorted(
            [n for n in ready if isinstance(n, LiteralValue)],
            key=self._critical_path_key,
        )
        for node in constants:
            if not self._literal_needed_now(node, computed_nodes):
                # No consumer needs it yet.  Leave it in ``ready`` (it has no
                # inputs, so it reappears every layer) and materialize it
                # just-in-time when a consumer becomes ready-except-literals.
                self._literal_deferred = True
                continue
            if not self._is_admissible(node):
                self._admission_deferred = True
                continue
            target_cols = self._try_allocate(node, residual_map)
            if target_cols is None:
                self._record_skip(
                    node,
                    "compute_literal_value",
                    "residual columns",
                    len(node),
                    residual_map.get_free_count(),
                    self.d,
                )
                continue
            assert len(target_cols) == len(node) == node.value.numel(), (
                f"Literal allocation width mismatch for {node!r}: "
                f"target_cols={len(target_cols)}, len(node)={len(node)}, "
                f"value.numel()={node.value.numel()}."
            )
            mlp_ops.append(_MlpOp("compute_literal_value", node, target_cols, []))
            computed_nodes.add(node)
            self._mark_scheduled(node)

        # 3d. Bias writes for biased Linears scheduled in attention sublayer
        # Biased Linear target cols were already cancelled when the Linear
        # was scheduled in the attention sublayer, so no extra cancel here.
        for node in biased_linears:
            target_cols = residual_map.get_indices(node)
            mlp_ops.append(_MlpOp("compute_bias", node, target_cols, []))

        # 3e. MLP-cancel: zero dead nodes' columns via `cancel_bypass` while
        # hidden slots remain.  Emitted last so the freed columns are not reused
        # this layer (the `[layer, cancel + 1)` residual accounting).  Each
        # column costs two hidden slots (the bypass lane-pair with W = -I).
        # Larger nodes first, matching the attention cancel batch's ordering.
        for node in sorted(mlp_cancel_candidates, key=lambda n: (-len(n), n.node_id)):
            if not residual_map.is_allocated(node):
                continue
            cols = list(residual_map.get_indices(node))
            n_slots = 2 * len(cols)
            if next_slot + n_slots > self.d_hidden:
                continue  # no slots left — defer the free to a later layer
            mlp_slots = list(range(next_slot, next_slot + n_slots))
            next_slot += n_slots
            mlp_ops.append(
                _MlpOp(
                    "cancel_bypass",
                    None,
                    cols,
                    mlp_slots,
                    source_cols=cols,
                )
            )
            self._release_cancelled(node, residual_map, mech="mlp")

        return mlp_ops

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _net_column_cost(
        self, node: Node, computed_nodes: Set[Node], residual_map: ResidualStreamMap
    ) -> int:
        """Net residual stream columns consumed by scheduling this node.

        Positive = net consumer, negative = net freer (frees more than it allocates).
        """
        cost = len(node)
        for inp in node.inputs:
            leaves = (
                flatten_concat_nodes([inp]) if isinstance(inp, Concatenate) else [inp]
            )
            for leaf in leaves:
                if leaf is self.pos_encoding:
                    continue
                if not residual_map.is_allocated(leaf):
                    continue
                remaining = (
                    self._get_effective_consumers(leaf) - computed_nodes - {node}
                )
                if not remaining:
                    cost -= len(leaf)
        return cost

    def _get_effective_consumers(self, node: Node) -> Set[Node]:
        """Get consumers, resolving through Concatenate nodes.

        Terminal Concatenates (no consumers, i.e. output nodes) are kept
        as effective consumers so their children aren't freed prematurely.
        """
        result = set()
        for consumer in self.graph.get_consumers(node):
            if isinstance(consumer, Concatenate):
                resolved = self._get_effective_consumers(consumer)
                if resolved:
                    result |= resolved
                else:
                    # Terminal Concatenate (output node) — its children's
                    # columns must stay allocated until compilation ends.
                    result.add(consumer)
            else:
                result.add(consumer)
        return result

    def _literal_needed_now(self, node: Node, computed_nodes: Set[Node]) -> bool:
        """Just-in-time gate for a ``LiteralValue`` (heuristic scheduler).

        A constant has no inputs, so it is "ready" from layer 0; scheduling
        it eagerly would hold a residual column across the whole network.
        Instead, materialize it only once some effective consumer has all of
        its *non-constant* inputs computed — the same layer that consumer's
        last such input lands.  The consumer is then ready next layer exactly
        as if the constant had been pre-placed (no added latency to the
        consumer), and the column is held only briefly.  See constants_plan.md.

        Under the deadlock-retry bypass (``_admission_bypass``) the gate is
        lifted so a stalled layer can always make progress (see
        :meth:`schedule_layer`).
        """
        if getattr(self, "_admission_bypass", False):
            return True
        consumers = self._get_effective_consumers(node)
        if not consumers:
            # An output (or otherwise unconsumed) constant has nothing to
            # wait for — materialize it so the output reader can find it.
            return True
        for consumer in consumers:
            if self._ready_except_literals(consumer, computed_nodes):
                return True
        return False

    def _ready_except_literals(self, node: Node, computed_nodes: Set[Node]) -> bool:
        """True if every non-``LiteralValue`` effective input of ``node`` (and
        every scheduling predecessor) is in ``computed_nodes``.

        Concatenate inputs are walked transparently to their leaves.  A node
        whose only inputs are constants is trivially ready-except-literals, so
        a pure-constant computation materializes immediately."""
        for inp in node.inputs:
            leaves = (
                flatten_concat_nodes([inp]) if isinstance(inp, Concatenate) else [inp]
            )
            for leaf in leaves:
                if isinstance(leaf, LiteralValue):
                    continue
                if leaf not in computed_nodes:
                    return False
        for pred in node.scheduling_predecessors:
            if pred not in computed_nodes:
                return False
        return True

    def _is_dead(self, node: Node, computed_nodes: Set[Node]) -> bool:
        if node is self.pos_encoding:
            return False
        if node not in self.graph.get_all_nodes():
            return False
        return self._get_effective_consumers(node).issubset(computed_nodes)

    def _is_keep_forever(self, node: Node) -> bool:
        """The node's residual columns must never be reclaimed within the
        schedule: the output node (no effective consumers) and every
        output-cone leaf feeding a terminal ``Concatenate`` (kept, not
        resolved, by ``_get_effective_consumers``).  Such a node reads as
        vacuously "dead" the moment it is computed, but freeing it would drop
        the value the compile is about to gather — so it is excluded from the
        MLP-cancel batch.  The attention cancel path never hits this because it
        frees only nodes dead at layer start (the output/leaf is computed later)
        and the compile stops once the output is done; the MLP-cancel recompute
        runs within the output's own layer, so it needs the guard.  Mirrors the
        directed replay's own protection (keep-forever nodes carry
        ``cancel_layer == max_layers``)."""
        cons = self._get_effective_consumers(node)
        return (not cons) or any(isinstance(c, Concatenate) for c in cons)

    def _attn_cancel_eligible(self, node: Node) -> bool:
        """May this dead node be zeroed by a batched attention cancel head?

        The eager heuristic tries the attention batch for every dead node (the
        MLP path is the fall-back), so the base returns True.
        ``DirectedLayerScheduler`` overrides this to honour the solver's
        per-node mechanism assignment: an MLP-mech node is never
        attention-cancelled.
        """
        return True

    def _mlp_cancel_eligible(self, node: Node) -> bool:
        """May this dead node be zeroed by an MLP ``cancel_bypass`` op?

        The eager heuristic MLP-cancels any dead node the attention batch could
        not fit (fall-back), so the base returns True — with one exception:
        the held source.  The CP-SAT model gives inputs no MLP cancel
        mechanism (no ``cancel_in_mlp`` var is created for freeable inputs),
        and its held-source bound demands ``cl >= layer[c] + 1`` for MLP
        readers, so an MLP hold at the last reader's own layer is a schedule
        the model cannot represent — as a warm-start hint CP-SAT silently
        drops the incumbent.  Declining here costs nothing: the MLP cancel
        would occupy the bank through the end of its layer anyway
        (``[layer, cancel + 1)``), so the attention hold one layer later
        frees the bank at the same layer and only shifts which budget pays.
        ``DirectedLayerScheduler`` overrides this to MLP-cancel only the nodes
        the solver assigned to the MLP mechanism (never an input).
        """
        if (
            self.held_output_layout is not None
            and node is self.held_output_layout.source
        ):
            return False
        return True

    def _mlp_cancel_defers_live_addends(self) -> bool:
        """Should a node used as an ``add_into`` live addend THIS layer be left
        for a later layer's MLP-cancel rather than freed now?

        The eager heuristic defers it (True): the value was just copied into a
        dead addend's columns, and keeping the live addend resident one more
        layer is the historical, conservative behaviour a bug reproducer pins
        (the calculator ``switch()`` shared-addend case).  The captured
        cancel-layer hint reflects the deferral, so the model stays consistent.
        ``DirectedLayerScheduler`` overrides this to False: it must honour the
        solver's assignment, which may cancel a live addend at the add's own
        layer (freeing it later would overflow the `[layer, cancel + 1)`
        residual budget); reading the live addend in the attention sublayer
        happens before the MLP-sublayer `cancel_bypass` zeroes it, so it is
        safe."""
        return True

    def _assigned_attention_releases(
        self,
        batch_nodes: List[Node],
        residual_map: ResidualStreamMap,
        computed_nodes: Set[Node],
    ) -> Optional[List[Node]]:
        """The complete release set of this layer's atomic attention batch,
        or ``None`` to keep the greedy heuristic behavior.

        ``batch_nodes`` are the attention compute candidates being placed
        this layer.  A non-``None`` return makes the batch the single
        definition of the attention transition: the shared sublayer walk
        captures every candidate's sources, preflights the head charge and
        residual width, commits every returned release into the one coalesced
        cancel, and bypasses the legacy dead-list cancel paths (promotion
        during placement and the end-of-sublayer leftover loop) for this
        sublayer.

        Base: ``None`` — the heuristic keeps its promotion/leftover
        machinery.  ``DirectedLayerScheduler`` returns the solver-assigned
        attention-mechanism cancels for the current layer.
        """
        return None

    def _uses_atomic_attention_batch(self) -> bool:
        """Static companion of :meth:`_assigned_attention_releases`.

        ``True`` promises that hook always returns a batch (never ``None``),
        so the layer-start dead-list snapshot in ``_schedule_layer_inner``
        is never consumed and may be skipped.  Base: ``False``.
        ``DirectedLayerScheduler``: ``True``.
        """
        return False

    def _dying_input_to_reuse(
        self, node: Node, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Optional[Node]:
        """An input leaf of ``node`` that placing ``node`` makes dead, whose
        columns may be cancelled+freed to allocate ``node``'s own output.

        The eager heuristic does not perform ordinary self-consumer reuse; the
        one case it supports is the registered tied-input -> output handoff
        (``optimize=0`` replay of the held bank).  The directed replay never
        reaches this: all its assigned same-layer releases — held source
        included — commit through the atomic attention batch before placement.
        """
        return self._held_handoff_dying_source(node, residual_map, computed_nodes)

    def _is_dead_for_add(
        self, addend: Node, add_node: Add, computed_nodes: Set[Node]
    ) -> bool:
        """True if all effective consumers of addend, except add_node, are computed.

        Concatenate nodes can't be dead addends — they aren't allocated in the
        residual stream, so their columns can't be reused for add_into.
        """
        if addend is self.pos_encoding:
            return False
        if (
            self.held_output_layout is not None
            and addend is self.held_output_layout.source
        ):
            # The tied bank must pass through a physical cancel into the held
            # state; a free Add may never inherit it through reassign().
            return False
        if isinstance(addend, Concatenate):
            return False
        effective = self._get_effective_consumers(addend)
        return (effective - {add_node}).issubset(computed_nodes)

    def _freshly_dead_inputs(
        self,
        node: Node,
        computed_nodes: Set[Node],
        residual_map: ResidualStreamMap,
    ) -> List[Node]:
        """Inputs of ``node`` that are now dead because ``node`` just got placed.

        Walks through ``Concatenate`` inputs since Concatenate nodes aren't
        residual-stream-allocated.  Returns only leaves currently allocated
        whose effective consumers are all in ``computed_nodes``.  Within-layer
        eager freeing is always on (the warm start and the production heuristic
        share it); the density it produces is now CP-SAT-representable.
        """
        result: List[Node] = []
        seen: Set[Node] = set()
        stack: List[Node] = list(node.inputs)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if isinstance(cur, Concatenate):
                stack.extend(cur.inputs)
                continue
            if cur is self.pos_encoding:
                continue
            if not residual_map.is_allocated(cur):
                continue
            if self.graph.is_input_node(cur):
                # Graph source nodes (InputNode, Embedding, LiteralValue,
                # RopeConfig) must stay in the residual stream so
                # callers can read their values via the compiled model's
                # snapshot-based lookup.  Existing dead-node cancellation
                # also leaves them alone until a later layer, so eager-
                # freeing must respect the same invariant.
                if (
                    self.held_output_layout is None
                    or cur is not self.held_output_layout.source
                ):
                    continue
            if self._get_effective_consumers(cur).issubset(computed_nodes):
                result.append(cur)
        return result

    def _find_dead_nodes(
        self, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> List[Node]:
        graph_nodes = self.graph.get_all_nodes()
        dead = []
        for node in sorted(residual_map.get_allocated_nodes(), key=lambda n: n.node_id):
            if node not in graph_nodes:
                continue
            if node not in computed_nodes:
                continue
            if self._is_dead(node, computed_nodes):
                dead.append(node)
        return dead

    # ------------------------------------------------------------------
    # Hook methods — overridden by DirectedLayerScheduler
    # ------------------------------------------------------------------

    def _get_ready_nodes(self, computed_nodes: Set[Node]) -> Set[Node]:
        """Return the set of nodes eligible to schedule this layer.

        The default returns ``graph.get_ready_nodes(...)`` unchanged.
        :class:`DirectedLayerScheduler` filters this to only nodes whose
        precomputed ``ScheduleAssignment`` says to run at the current
        layer.
        """
        return self.graph.get_ready_nodes(computed_nodes)

    def _heads_for_node(self, node: Node) -> int:
        """Number of attention heads needed to copy a node's output."""
        return (len(node) + self.d_head - 1) // self.d_head

    def _heads_for_add(self, node: Node) -> int:
        """Number of attention heads needed for a compute_add op.

        When 2 * chunk_size <= d_head, both inputs share one combined head.
        Otherwise each input needs its own head.
        """
        d_output = len(node)
        d_head = self.d_head
        total = 0
        for start in range(0, d_output, d_head):
            chunk_size = min(start + d_head, d_output) - start
            total += 1 if 2 * chunk_size <= d_head else 2
        return total

    def _heads_for_linear(self, node: Linear) -> int:
        """Number of attention heads needed for a standalone Linear.

        Support-aware: one head per d_head-wide input chunk with any
        nonzero weight row, floor 1 (realization.linear_attn_chunks — the
        same list the weight-writer emits, so this budget and the emission
        cannot desync).  Dense weights charge the old ceil(d_input/d_head).
        """
        return linear_attn_heads(node, self.d_head)

    def _has_zero_bias(self, node: Linear) -> bool:
        return node.output_bias.abs().sum().item() == 0

    def _try_allocate(
        self, node: Node, residual_map: ResidualStreamMap
    ) -> Optional[List[int]]:
        if (
            self.held_output_layout is not None
            and node is self.held_output_layout.target
        ):
            bank = self.held_output_layout.bank
            if not residual_map.can_allocate_at(node, bank):
                return None
            return residual_map.allocate_at(node, bank)
        if len(node) > residual_map.get_free_count():
            return None
        return residual_map.allocate(node)

    def _is_forced_fresh_add(self, node: Node) -> bool:
        return (
            self.held_output_layout is not None
            and node is self.held_output_layout.target
        )

    def _release_cancelled(
        self, node: Node, residual_map: ResidualStreamMap, *, mech: str
    ) -> None:
        """Apply the allocator transition after a charged physical cancel."""
        layout = self.held_output_layout
        if layout is not None and node is layout.source:
            actual = tuple(residual_map.get_indices(node))
            if actual != layout.bank:
                raise AssertionError(
                    f"held source {node!r} moved from bank {layout.bank} to "
                    f"{actual} before cancellation"
                )
            residual_map.hold(node, mech=mech)
            return
        residual_map.free(node, mech=mech)

    def _held_handoff_dying_source(
        self, node: Node, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Optional[Node]:
        """The tied input when ``node`` is its final attention reader/writer."""
        layout = self.held_output_layout
        if layout is None or node is not layout.target:
            return None
        source = layout.source
        if not residual_map.is_allocated(source):
            return None
        if source not in flatten_concat_nodes(list(node.inputs)):
            return None
        if not (self._get_effective_consumers(source) - {node}).issubset(
            computed_nodes
        ):
            return None
        return source

    def _critical_path_key(self, node: Node):
        # node_id tie-break: critical-path ties are common, and a stable
        # sort without a total key inherits whatever order the candidate
        # list was built in — for node sets that is hash-table order
        # keyed on absolute node_id values, which differs between a
        # graph and its fresh rebuild (or its lowered copy).  The
        # tie-break makes every sort over this key reproduce across
        # rebuilds and processes; relative node_id order is
        # construction-order, which deterministic builders (and the
        # lowering clone pass) preserve.
        return (-self.graph.get_critical_path_length(node), node.node_id)

    # ------------------------------------------------------------------
    # Admission control (sibling-cluster-based gating)
    # ------------------------------------------------------------------
    #
    # The scheduler is otherwise fully greedy: if ``N`` sibling chains
    # in a cluster are simultaneously ready, it admits as many as
    # capacity allows.  That creates a residual-pressure plateau (see
    # optimization_guide §7) because each admitted chain pins its wide
    # intermediates until its terminal is placed.
    #
    # Admission control caps the number of *not-yet-in-flight* chains
    # per cluster so that projected peak residual occupancy stays
    # within a configurable budget.  A chain is "in flight" from the
    # moment any of its exclusive nodes is scheduled until its
    # terminal is placed.  Once in flight, the chain is always
    # admitted — we never leave work half-scheduled.

    def _chain_of(self, node: Node) -> Optional[Tuple[int, int]]:
        if self._clusters is None:
            return None
        return self._clusters.node_to_chain.get(node)

    def _is_admissible(self, node: Node) -> bool:
        """True if ``node`` can be scheduled under the admission budget.

        Nodes outside any sibling cluster are always admissible.  A
        node in a cluster is admissible if the chain it belongs to is
        already in flight, or if admitting a fresh chain would keep
        projected residual occupancy for this cluster within
        ``admission_budget_fraction * d``.

        When ``self._admission_bypass`` is set (deadlock guard), admits
        everything — see :meth:`schedule_layer`.
        """
        if getattr(self, "_admission_bypass", False):
            return True
        key = self._chain_of(node)
        if key is None:
            return True
        cluster_id, chain_id = key
        in_flight = self._in_flight.get(cluster_id, set())
        if chain_id in in_flight:
            return True
        assert (
            self._clusters is not None
        )  # key is not None implies _clusters is not None
        cluster = self._clusters.clusters[cluster_id]
        projected = (len(in_flight) + 1) * cluster.peak_chain_width
        budget = int(self._admission_budget_fraction * self.d)
        return projected <= budget

    def _mark_scheduled(self, node: Node) -> None:
        """Update in-flight bookkeeping after a node is placed.

        Scheduling any exclusive node marks the chain in flight.
        Scheduling the chain's terminal marks the chain completed.
        """
        if self._clusters is None:
            return
        key = self._clusters.node_to_chain.get(node)
        if key is None:
            return
        cluster_id, chain_id = key
        self._in_flight.setdefault(cluster_id, set()).add(chain_id)

        term_key = self._clusters.terminal_to_chain.get(node)
        if term_key is not None:
            t_cluster_id, t_chain_id = term_key
            in_flight = self._in_flight.get(t_cluster_id)
            if in_flight is not None:
                in_flight.discard(t_chain_id)

    def _filter_admissible(
        self, candidates: List, node_getter=lambda t: t[1]
    ) -> Tuple[List, List]:
        """Partition candidates into (admissible, deferred).

        Accepts a list of tuples/nodes and a ``node_getter`` to extract
        the underlying Node for the admission check.  Returns the same
        structure, not just nodes, so callers can preserve extra
        metadata (op-type, heads count) without rewrapping.
        """
        if self._clusters is None or not self._clusters.clusters:
            return candidates, []
        admissible = []
        deferred = []
        for c in candidates:
            if self._is_admissible(node_getter(c)):
                admissible.append(c)
            else:
                deferred.append(c)
        return admissible, deferred

    def _capture_attn_sources(
        self, op_type: str, node: Node, residual_map: ResidualStreamMap
    ) -> Dict[str, List[int]]:
        """Capture an attention compute op's source columns while every input
        is still allocated.

        Runs the I4 schedule-time liveness check (``_require_live``) at the
        capture site and returns exactly the attributes later copied onto the
        emitted ``_AttentionOp`` (the weight-writer reads sources from the op,
        never from the residual map at write time).  The heuristic calls this
        lazily at placement; the directed replay calls it for its complete
        attention batch before any release commits.  Either way the capture
        happens while the inputs are live, so a later legitimate release
        never invalidates it — callers consume the saved result and must not
        re-run the liveness lookup afterwards.
        """
        sources: Dict[str, List[int]] = {}
        if op_type == "compute_linear":
            self._require_live(
                node.inputs[0],
                residual_map,
                f"compute_linear input for {node!r}",
            )
            sources["source_cols"] = residual_map.resolve_indices(node.inputs[0])
        elif op_type == "compute_attn":
            q_in, k_in, v_in = node.inputs
            self._require_live(q_in, residual_map, f"compute_attn Q for {node!r}")
            self._require_live(k_in, residual_map, f"compute_attn K for {node!r}")
            self._require_live(v_in, residual_map, f"compute_attn V for {node!r}")
            sources["q_source_cols"] = residual_map.resolve_indices(q_in)
            sources["k_source_cols"] = residual_map.resolve_indices(k_in)
            sources["source_cols"] = residual_map.resolve_indices(v_in)
        elif op_type == "compute_add":
            a0, a1 = node.inputs
            self._require_live(a0, residual_map, f"compute_add a0 for {node!r}")
            self._require_live(a1, residual_map, f"compute_add a1 for {node!r}")
            sources["source_cols"] = residual_map.resolve_indices(a0)
            sources["source_cols_b"] = residual_map.resolve_indices(a1)
        else:
            raise ValueError(f"not an attention compute op: {op_type!r}")
        return sources

    def _require_live(
        self,
        node: Node,
        residual_map: ResidualStreamMap,
        op_label: str,
    ) -> None:
        """Invariant A (schedule-time): ``node`` must be retrievable from
        ``residual_map`` when its value is read as a source.

        Walks through Concatenate to check every leaf.  Raises
        :class:`AssertionError` with op context if any required leaf is
        not currently allocated — surfaces a liveness bug *before* the
        KeyError from get_indices, so the message names the node, the
        consumer op, and the residual_map state.
        """
        if isinstance(node, Concatenate):
            missing = [
                leaf
                for leaf in flatten_concat_nodes([node])
                if not residual_map.is_allocated(leaf)
            ]
            if missing:
                raise AssertionError(
                    f"Live-column invariant violated while scheduling "
                    f"{op_label}: Concatenate {node!r} has unallocated "
                    f"leaves {[repr(m) for m in missing[:4]]}. "
                    f"free_count={residual_map.get_free_count()}, "
                    f"allocated={len(residual_map.get_allocated_nodes())}."
                )
            return
        if not residual_map.is_allocated(node):
            raise AssertionError(
                f"Live-column invariant violated while scheduling "
                f"{op_label}: input {node!r} is not allocated. "
                f"free_count={residual_map.get_free_count()}, "
                f"allocated={len(residual_map.get_allocated_nodes())}."
            )


class DirectedLayerScheduler(LayerScheduler):
    """LayerScheduler driven by a precomputed CP-SAT ``ScheduleAssignment``.

    See ``docs/cpsat_scheduler.md`` §2 for the spec.  Three things change
    relative to the parent heuristic:

    - **Ready filter.** Only nodes with
      ``assignment.node_to_layer[n] == current_layer`` are eligible to
      schedule this layer.  Other ready nodes stay deferred until their
      assigned layer.
    - **Routing.** Each standalone ``Linear`` is forced into the attention
      sublayer or the MLP bypass per the realization table resolved from
      ``assignment.node_to_routing`` (the solve is the resolver).
      ``policy.local_in_attention`` is ignored.
    - **Cancellation.** Attention-mechanism cancels replay as ONE atomic
      batch per layer (``_assigned_attention_releases``): every value whose
      assigned cancel layer equals the current layer is released after all
      batch sources are captured and before any output places — matching the
      aggregate transition the residual cumulative priced, with no
      per-output bootstrap order.  The legacy dead-list promotion/leftover
      paths are bypassed on this scheduler.  MLP-mechanism cancels keep the
      sequential ``cancel_bypass`` path, restricted to their assigned layer;
      mid-layer eager freeing is restricted to the assignment's cancel
      timing (load-bearing for the MLP sublayer's gap-0 cancels).

    What it preserves (by inheriting the parent's per-layer code path):
    cancel coalescing into a single batched ``_AttentionOp("cancel")``,
    source-column capture via ``_require_live``, and the four allocator
    invariants I1–I4 (which run inside ``ResidualStreamMap`` and the
    weight-writer that this subclass doesn't touch).

    The caller must invoke :meth:`set_current_layer` with the layer
    index *before* each :meth:`schedule_layer` call — the subclass has
    no way to know which layer the compile loop is currently building.
    """

    def __init__(
        self,
        graph: GraphAnalyzer,
        d: int,
        d_head: int,
        pos_encoding,
        assignment: ScheduleAssignment,
        d_hidden: Optional[int] = None,
        clusters: Optional[SiblingClusters] = None,
        admission_budget_fraction: float = 0.4,
        policy: Optional[SchedulingPolicy] = None,
        realization_table: Optional[RealizationTable] = None,
        bias: bool = True,
        held_output_layout: Optional[HeldOutputLayout] = None,
        n_heads: Optional[int] = None,
    ):
        # The CP-SAT model does not represent the sibling-cluster admission
        # constraint (docs/cpsat_scheduler.md §3 Model preconditions), so a
        # replay under admission gating could defer a batch member AFTER the
        # atomic attention batch committed its releases — stranding a node
        # whose inputs were already cancelled.  forward_compile rejects the
        # combination up front; reject it here too so direct construction
        # (tests, tooling) cannot reach that state.
        if clusters is not None:
            raise ValueError(
                "DirectedLayerScheduler does not support sibling-cluster "
                "admission control (clusters): the CP-SAT model has no "
                "admission constraint, so the directed replay must run "
                "ungated.  Pass clusters=None (use the heuristic "
                "LayerScheduler for admission-controlled schedules)."
            )
        # The directed path's resolver is the solve itself: its per-node
        # sublayer decisions (node_to_routing) resolve the table the walk
        # reads.  policy.local_in_attention plays no part here.
        if realization_table is None:
            realization_table = RealizationTable.build(
                graph.get_all_nodes()
            ).resolve_from_assignment(assignment.node_to_routing)
        super().__init__(
            graph,
            d,
            d_head,
            pos_encoding,
            n_heads=n_heads,
            d_hidden=d_hidden,
            clusters=clusters,
            admission_budget_fraction=admission_budget_fraction,
            policy=policy,
            realization_table=realization_table,
            bias=bias,
            held_output_layout=held_output_layout,
        )
        self._assignment = assignment
        self._current_layer: int = -1

    def set_current_layer(self, layer: int) -> None:
        """Tell the subclass which transformer layer is being built next.

        Must be called before every :meth:`schedule_layer` invocation —
        the ready/cancel filters key on ``current_layer``.
        """
        self._current_layer = layer

    def _get_ready_nodes(self, computed_nodes: Set[Node]) -> Set[Node]:
        if self._current_layer < 0:
            raise RuntimeError(
                "DirectedLayerScheduler.set_current_layer() must be called "
                "before schedule_layer()."
            )
        all_ready = self.graph.get_ready_nodes(computed_nodes)
        n2l = self._assignment.node_to_layer
        return {n for n in all_ready if n2l.get(n.node_id) == self._current_layer}

    def _attn_cancel_eligible(self, node: Node) -> bool:
        # Honour the solver's mechanism assignment: only attention-mech (and
        # unassigned — freeable inputs, always attention-cancelled) nodes may
        # enter the attention cancel batch.  A default of "attn" for absent
        # nodes matches the assignment (freeable inputs carry no mech entry).
        return self._assignment.node_to_cancel_mech.get(node.node_id, "attn") != "mlp"

    def _mlp_cancel_eligible(self, node: Node) -> bool:
        # Only the nodes the solver routed to the MLP mechanism get a
        # `cancel_bypass`; an attention-mech leftover that a full head budget
        # deferred stays deferred (re-surfaced next layer by _find_dead_nodes).
        return self._assignment.node_to_cancel_mech.get(node.node_id, "attn") == "mlp"

    def _mlp_cancel_defers_live_addends(self) -> bool:
        # Honour the solver's assignment: a live addend the solver routed to an
        # MLP cancel at the add's own layer must free this layer, or the
        # `[layer, cancel + 1)` residual budget the model reserved overflows.
        return False

    def _assigned_attention_releases(
        self,
        batch_nodes: List[Node],
        residual_map: ResidualStreamMap,
        computed_nodes: Set[Node],
    ) -> List[Node]:
        """The solver-assigned attention transition for the current layer.

        Returns every allocated value whose assigned cancel mechanism is
        attention and whose assigned cancel layer EQUALS the current layer —
        values already dead on layer entry and values that die only after
        all batch readers run, alike.  Equality is scoped to this hook; the
        ``<=`` in :meth:`_find_dead_nodes` survives only for MLP-cancel
        deferral re-surfacing.  Two model bounds keep the set well-formed
        (an implementation "defensively" handling either forbidden case
        would be dead code hiding a model regression): free-Add addends are
        bounded ``cancel >= layer[A] + is_free[A]``, one layer after the
        add, so the batch never collides with the free-Add ``reassign``;
        ordinary freeable inputs carry the uniform gap-1 bound, so any input
        here is already dead on entry — the held source is the only input
        the model lets die mid-attention.

        Raises on an overdue attention-mechanism cancel (assigned layer
        already passed but the value is still allocated): an assignment that
        fit only because a cancel occurred on time is not soundly replayed
        by delaying it, and an equality-only match would otherwise implement
        "assert" as "silently leak the columns".  Also validates A2: every
        uncomputed effective consumer of a released value must be in the
        batch, or the assignment and the replay contract disagree — fail
        before mutation.
        """
        n2cl = self._assignment.node_to_cancel_layer
        graph_nodes = self.graph.get_all_nodes()
        batch = set(batch_nodes)
        releases: List[Node] = []
        for node in sorted(residual_map.get_allocated_nodes(), key=lambda n: n.node_id):
            if node not in graph_nodes:
                continue  # compile-internal allocations (const_one) never cancel
            cl = n2cl.get(node.node_id)
            if cl is None or cl > self._current_layer:
                continue
            if not self._attn_cancel_eligible(node):
                continue  # MLP-mechanism cancels flow through cancel_bypass
            if cl < self._current_layer:
                raise AssertionError(
                    f"Model/replay contract violation (overdue cancel) at "
                    f"layer {self._current_layer}: {node!r} was assigned an "
                    f"attention-mechanism cancel at layer {cl} but is still "
                    f"allocated — the batch that should have released it "
                    f"never did, and replaying the cancel late would leak "
                    f"the columns the model already reclaimed."
                )
            releases.append(node)
        for u in releases:
            missing = [
                c
                for c in self._get_effective_consumers(u) - computed_nodes
                if c not in batch
            ]
            if missing:
                raise AssertionError(
                    f"Model/replay contract violation (A2) at layer "
                    f"{self._current_layer}: {u!r} is assigned an "
                    f"attention-mechanism cancel this layer, but its "
                    f"uncomputed consumer(s) "
                    f"{[repr(m) for m in missing[:4]]} are not in this "
                    f"layer's attention batch."
                )
        return releases

    def _uses_atomic_attention_batch(self) -> bool:
        # _assigned_attention_releases above always returns a list, so the
        # layer-start dead-list snapshot is never consumed on this scheduler.
        return True

    def _literal_needed_now(self, node: Node, computed_nodes: Set[Node]) -> bool:
        # Assignment-driven: the CP-SAT layer assignment already places each
        # constant just-in-time, and ``_get_ready_nodes`` only releases it at
        # its assigned layer.  No consumer-readiness gate here.
        return True

    def _find_dead_nodes(
        self, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> List[Node]:
        if self._current_layer < 0:
            raise RuntimeError(
                "DirectedLayerScheduler.set_current_layer() must be called "
                "before schedule_layer()."
            )
        graph_nodes = self.graph.get_all_nodes()
        n2cl = self._assignment.node_to_cancel_layer
        dead: List[Node] = []
        for node in sorted(residual_map.get_allocated_nodes(), key=lambda n: n.node_id):
            if node not in graph_nodes:
                continue
            if node not in computed_nodes:
                continue
            cl = n2cl.get(node.node_id)
            if cl is None or cl > self._current_layer:
                continue
            # The ``<=`` above (rather than ``==``) no longer serves attention
            # cancels — those flow exclusively through the atomic batch hook
            # (``_assigned_attention_releases``: equality matching plus the
            # overdue assertion).  It survives only for the MLP-cancel path,
            # whose ``cancel_bypass`` still defers silently when a layer's
            # hidden slots run out and relies on the ``<=`` to resurface the
            # deferred node (the callers' mechanism filters keep the two
            # paths disjoint).
            if not self._is_dead(node, computed_nodes):
                continue
            dead.append(node)
        return dead

    def _freshly_dead_inputs(
        self,
        node: Node,
        computed_nodes: Set[Node],
        residual_map: ResidualStreamMap,
    ) -> List[Node]:
        # Mid-layer freeing restricted to the assignment's cancel timing:
        # surface a freshly-dead input only when the solver scheduled its
        # cancel at (or before) the current layer.  On the attention side
        # this surface is inert under the atomic batch (batch-released values
        # are no longer allocated, and attention-mech cancels never linger
        # past their layer); it remains load-bearing for the MLP sublayer's
        # ``_surface_mlp_freshly_dead``, which executes the solver's gap-0
        # MLP-mechanism cancels right after the last MLP reader places.
        n2cl = self._assignment.node_to_cancel_layer
        return [
            n
            for n in super()._freshly_dead_inputs(node, computed_nodes, residual_map)
            if n2cl.get(n.node_id) is not None
            and n2cl[n.node_id] <= self._current_layer
        ]
