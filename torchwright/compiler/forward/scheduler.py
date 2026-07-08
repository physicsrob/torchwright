"""Layer scheduler for the forward compiler.

Given the current residual stream state and graph metadata, decides what
to compute/cancel in one transformer layer. Returns AttnHeadOp and MLPOp
lists for the weight writer.

Mutates residual_map (allocate, free, reassign) and computed_nodes (add).
"""

from typing import Dict, List, Optional, Set, Tuple

from torchwright.compiler.realization import RealizationTable
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.forward.cpsat_scheduler import ScheduleAssignment
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.sibling_clusters import SiblingClusters
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.compiler.forward.weight_writer import AttnHeadOp, MLPOp
from torchwright.graph import Node, Linear, Attn, Add, Concatenate
from torchwright.graph.misc import LiteralValue
from torchwright.graph.ffn import FFN


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
        eager_free: bool = True,
        realization_table: Optional[RealizationTable] = None,
        bias: bool = True,
    ):
        self.graph = graph
        self.d = d
        self.d_hidden = d if d_hidden is None else d_hidden
        self.d_head = d_head
        # bias=False reserves hidden slot 0 for the constant lane (the
        # weight-writer's BiasFold): packing starts at slot 1, so the
        # effective per-layer capacity is d_hidden - 1.  Must agree with the
        # capacity CP-SAT models (forward_compile passes the solver
        # d_hidden - 1) or a solver-feasible layer is unpackable on replay.
        self.bias = bias
        self.n_heads = d // d_head
        self.pos_encoding = pos_encoding
        self.policy = policy if policy is not None else SchedulingPolicy()
        # The resolved realization table the walk reads (one artifact,
        # written by whichever resolver the optimize level provides — see
        # torchwright/compiler/realization.py).  When the caller doesn't
        # pass one (standalone scheduler construction in tests), resolve
        # statically from the policy through the same code path the
        # optimize=0 compile uses.
        if realization_table is None:
            realization_table = RealizationTable.build(
                graph.get_all_nodes()
            ).resolve_static(self.policy)
        self.realization_table = realization_table
        # Eager (within-layer) freeing: when a node is placed, free any of its
        # inputs that just became dead so their columns can be reused *in the
        # same layer*.  This is what lets the heuristic exploit within-layer
        # column reuse — a density the layer-granular CP-SAT model cannot
        # represent, which makes the eager schedule an infeasible CP-SAT hint.
        # Set ``eager_free=False`` to produce a model-representable (deeper but
        # feasible) schedule; the CP-SAT warm-start uses this so the solver gets
        # a real incumbent to improve, while the heuristic *fallback* keeps the
        # default eager behavior (shallower).  See ``_freshly_dead_inputs``.
        self._eager_free = eager_free
        # Within-layer retry of deferred attention compute candidates.  False
        # for the heuristic (single order-preserving pass, historical
        # behavior); DirectedLayerScheduler sets it True so a same-layer
        # column handoff resolves regardless of candidate order — see the
        # retry loop in ``_schedule_attn_sublayer``.
        self._retry_within_layer = False

        # Admission control state (see _is_admissible).  When clusters
        # is None or empty, admission is disabled and the scheduler
        # behaves as it did before this feature.
        self._clusters = clusters
        self._admission_budget_fraction = admission_budget_fraction
        self._in_flight: Dict[int, Set[int]] = {}
        if clusters is not None:
            for cluster_id in clusters.clusters:
                self._in_flight[cluster_id] = set()

    def schedule_layer(
        self, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Tuple[List[AttnHeadOp], List[MLPOp], List[Node]]:
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
                raise RuntimeError(
                    f"No progress: {len(remaining)} nodes remaining, "
                    f"{residual_map.get_free_count()} free columns"
                )

        return attn_ops, mlp_ops, biased_linears

    def _schedule_layer_inner(
        self, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Tuple[List[AttnHeadOp], List[MLPOp], List[Node], bool]:
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

        dead = self._find_dead_nodes(residual_map, computed_nodes)

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

        # --- 3. MLP sublayer ---
        mlp_ops = self._schedule_mlp_sublayer(
            ready,
            bypass_linears,
            biased_linears,
            residual_map,
            computed_nodes,
        )

        # Emit the single batched death-cancel op (built in the attention
        # sublayer) at the end of the layer.  Order within the sublayer is
        # irrelevant (all heads run in parallel and sum into the residual
        # stream), so it's fine to append after compute ops.
        if cancel_cols:
            attn_ops.append(AttnHeadOp("cancel", None, cancel_cols))

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
        # AttnHeadOp("cancel", None, cancel_cols) emitted at the end.
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
            if heads_used >= self.n_heads:
                break
            a0, a1 = add_node.inputs
            d0 = self._is_dead_for_add(a0, add_node, computed_snapshot)
            d1 = self._is_dead_for_add(a1, add_node, computed_snapshot)
            dead_addend = a0 if d0 else a1
            live_addend = a1 if d0 else a0
            n_heads = (len(live_addend) + self.d_head - 1) // self.d_head
            if heads_used + n_heads > self.n_heads:
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
                AttnHeadOp(
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

        # Cancellation candidates (exclude live addends of add_into ops)
        cancel_candidates = [
            n
            for n in dead
            if n is not self.pos_encoding and n not in add_into_live_addends
        ]
        cancel_candidates.sort(key=lambda n: (-len(n), n.node_id))  # largest first

        # 2b-2d. Schedule compute ops with cancellation promotion
        def _try_place(op_type, node, n_heads_needed) -> bool:
            """Place one compute candidate; return True if placed, False if
            deferred (over budget, inadmissible, or no columns available)."""
            nonlocal heads_used
            if heads_used + n_heads_needed > self.n_heads:
                return False
            if not self._is_admissible(node):
                self._admission_deferred = True
                return False

            # Capture source columns at schedule time, BEFORE allocating this
            # op's target.  The weight-writer reads sources from the op
            # directly, so later free()s (eager freeing, or the self-consumer
            # reuse below) don't orphan the lookups.  ``_require_live`` runs
            # here while every input is still allocated (I4), so the capture
            # also holds when a dying input is freed and its own columns are
            # reused for this op's output.
            sources: dict = {}
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
                residual_map.free(cn)
                target_cols = self._try_allocate(node, residual_map)

            # Self-consumer reuse (directed replay only): placing ``node`` may
            # make one of its OWN inputs dead (node is that input's last
            # consumer).  The solver schedules that input's cancel at this layer
            # (gap-0 intra-layer reuse), but it is not dead until node places —
            # the chicken-and-egg the promotion path above cannot break.  Cancel
            # and free the dying input now (its value was captured for node's
            # source above), then allocate node's target from the freed pool.
            # Order — capture, cancel, free, allocate — keeps I1 (free precedes
            # allocate) and I4 (require_live ran while the input was live).
            if target_cols is None:
                reuse = self._dying_input_to_reuse(node, residual_map, computed_nodes)
                if reuse is not None:
                    result = try_add_cancel(residual_map.get_indices(reuse))
                    if result is not None:
                        additions, delta = result
                        if heads_used + n_heads_needed + delta <= self.n_heads:
                            commit_cancel(additions, delta)
                            residual_map.free(reuse)
                            target_cols = self._try_allocate(node, residual_map)

            if target_cols is None:
                return False

            op = AttnHeadOp(op_type, node, target_cols)
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

        # Single pass for the heuristic (order-preserving, identical to the
        # historical loop).  The directed replay retries deferred candidates
        # within the layer: a same-layer column handoff means candidate W's
        # allocation can depend on candidate R's placement (R's placement
        # surfaces the dying node whose columns W reuses via
        # ``_freshly_dead_inputs``), and the sorted candidate order does not
        # guarantee R comes before W.  The retry loop reaches a fixpoint —
        # each pass either places a candidate or terminates.
        pending = list(compute_candidates)
        while pending:
            deferred = []
            progress = False
            for cand in pending:
                if _try_place(*cand):
                    progress = True
                else:
                    deferred.append(cand)
            if not progress or not self._retry_within_layer:
                break
            pending = deferred

        # 2e. Remaining cancellations — try to fold remaining dead cols
        # into the same batch.
        for cn in cancel_candidates:
            cn_cols = residual_map.get_indices(cn)
            result = try_add_cancel(cn_cols)
            if result is None:
                continue
            additions, delta = result
            commit_cancel(additions, delta)
            residual_map.free(cn)

        # The batched death-cancel op is emitted by the caller.  The MLP
        # sublayer does not extend it (fresh MLP allocations need no cancel
        # under universal zero-init), so only the column list is returned.
        return (
            attn_ops,
            biased_linears,
            bypass_linears,
            cancel_cols,
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
    ):
        mlp_ops = []
        # Slot 0 is the constant lane under bias=False — see LayerScheduler
        # __init__ and weight_writer.BiasFold.
        next_slot = 0 if self.bias else 1

        # No cancels are emitted from the MLP sublayer: fresh MLP allocations
        # land on clean columns (universal zero-init), so their first additive
        # write needs no prior cancel.

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
                continue
            if not self._is_admissible(ffn):
                self._admission_deferred = True
                continue
            target_cols = self._try_allocate(ffn, residual_map)
            if target_cols is None:
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
                MLPOp(
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
                    continue
                if not self._is_admissible(node):
                    self._admission_deferred = True
                    continue
                target_cols = self._try_allocate(node, residual_map)
                if target_cols is None:
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
                    MLPOp(
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
                continue
            assert len(target_cols) == len(node) == node.value.numel(), (
                f"Literal allocation width mismatch for {node!r}: "
                f"target_cols={len(target_cols)}, len(node)={len(node)}, "
                f"value.numel()={node.value.numel()}."
            )
            mlp_ops.append(MLPOp("compute_literal_value", node, target_cols, []))
            computed_nodes.add(node)
            self._mark_scheduled(node)

        # 3d. Bias writes for biased Linears scheduled in attention sublayer
        # Biased Linear target cols were already cancelled when the Linear
        # was scheduled in the attention sublayer, so no extra cancel here.
        for node in biased_linears:
            target_cols = residual_map.get_indices(node)
            mlp_ops.append(MLPOp("compute_bias", node, target_cols, []))

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

    def _dying_input_to_reuse(
        self, node: Node, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Optional[Node]:
        """An input leaf of ``node`` that placing ``node`` makes dead, whose
        columns may be cancelled+freed to allocate ``node``'s own output
        (self-consumer intra-layer reuse).

        The eager heuristic never self-consumer-reuses — it would change
        heuristic schedules and every golden layer count — so the base class
        returns ``None``.  ``DirectedLayerScheduler`` overrides this to replay
        the solver's gap-0 intra-layer schedules.
        """
        return None

    def _is_dead_for_add(
        self, addend: Node, add_node: Add, computed_nodes: Set[Node]
    ) -> bool:
        """True if all effective consumers of addend, except add_node, are computed.

        Concatenate nodes can't be dead addends — they aren't allocated in the
        residual stream, so their columns can't be reused for add_into.
        """
        if addend is self.pos_encoding:
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
        whose effective consumers are all in ``computed_nodes``.

        Returns an empty list when ``eager_free`` is disabled (the CP-SAT
        warm-start path), so the resulting schedule never frees and reuses a
        column within a consumer's layer — keeping it representable by the
        layer-granular CP-SAT model.
        """
        if not self._eager_free:
            return []
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
        """Number of attention heads needed for a standalone Linear."""
        d_input = len(node.inputs[0])
        return (d_input + self.d_head - 1) // self.d_head

    def _has_zero_bias(self, node: Linear) -> bool:
        return node.output_bias.abs().sum().item() == 0

    def _try_allocate(
        self, node: Node, residual_map: ResidualStreamMap
    ) -> Optional[List[int]]:
        if len(node) > residual_map.get_free_count():
            return None
        return residual_map.allocate(node)

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
    - **Cancellation.** Dead-node candidates restricted to nodes with
      ``assignment.node_to_cancel_layer[n] == current_layer``.  The
      heuristic's eager freeing of freshly-dead inputs is suppressed,
      so cancellation timing follows the assignment exactly.

    What it preserves (by inheriting the parent's per-layer code path):
    cancel coalescing into a single batched ``AttnHeadOp("cancel")``,
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
    ):
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
            d_hidden=d_hidden,
            clusters=clusters,
            admission_budget_fraction=admission_budget_fraction,
            policy=policy,
            realization_table=realization_table,
            bias=bias,
        )
        self._assignment = assignment
        self._current_layer: int = -1
        # Same-layer column handoffs (a cancel at the last attention
        # consumer's own layer) make within-layer placement order matter;
        # the retry pass in ``_schedule_attn_sublayer`` resolves it.
        self._retry_within_layer = True

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
            # A directed cancel at this layer whose last consumer ALSO runs
            # this layer (an attention-sublayer read — the intra-layer-reuse
            # regime) is not dead at layer start; it surfaces through
            # ``_freshly_dead_inputs`` right after that consumer is placed.
            # The ``<=`` above (rather than ``==``) re-surfaces a directed
            # cancel that a full head budget deferred past its assigned layer.
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
        # cancel at (or before) the current layer.  This is how a directed
        # same-layer free-after-read (cancel == the last attention-sublayer
        # consumer's layer) executes — the parent's walk fires right after
        # that consumer is placed, exactly like the eager heuristic.
        n2cl = self._assignment.node_to_cancel_layer
        return [
            n
            for n in super()._freshly_dead_inputs(node, computed_nodes, residual_map)
            if n2cl.get(n.node_id) is not None
            and n2cl[n.node_id] <= self._current_layer
        ]

    def _dying_input_to_reuse(
        self, node: Node, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Optional[Node]:
        # Directed replay of a gap-0 self-consumer handoff: return an input
        # leaf of ``node`` whose solver-assigned cancel is at (or before) this
        # layer and whose only uncomputed consumer is ``node`` itself, so
        # placing ``node`` makes it dead and its columns can be cancelled+freed
        # for ``node``'s own output.  Graph-source leaves are excluded — they
        # stay in the residual stream for the snapshot-based value lookup, the
        # same invariant ``_freshly_dead_inputs`` respects.
        n2cl = self._assignment.node_to_cancel_layer
        leaves: List[Node] = []
        for inp in node.inputs:
            if isinstance(inp, Concatenate):
                leaves.extend(flatten_concat_nodes([inp]))
            else:
                leaves.append(inp)
        for leaf in leaves:
            if leaf is self.pos_encoding:
                continue
            if not residual_map.is_allocated(leaf):
                continue
            if self.graph.is_input_node(leaf):
                continue
            cl = n2cl.get(leaf.node_id)
            if cl is None or cl > self._current_layer:
                continue
            if (self._get_effective_consumers(leaf) - {node}).issubset(computed_nodes):
                return leaf
        return None
