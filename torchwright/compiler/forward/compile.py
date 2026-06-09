"""Forward compiler: wires GraphAnalyzer, ResidualStreamMap, LayerScheduler,
and WeightWriter into a complete compilation pipeline.

Produces a HeadlessTransformer that can compute the output node's value
given input values.
"""

import copy
import os
import time
import warnings
from typing import Callable, Optional, Set

import torch

from torchwright.compiler.device import get_device
from torchwright.compiler.residual_assignment import ResidualAssignment
from torchwright.compiler.forward.cpsat_scheduler import (
    Costs,
    solve_schedule,
)
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.scheduler import (
    DirectedLayerScheduler,
    LayerScheduler,
)
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.compiler.forward.sibling_clusters import (
    SiblingClusterAnalyzer,
)
from torchwright.compiler.forward.weight_writer import (
    AttnHeadOp,
    MLPOp,
    write_attn_sublayer,
    write_mlp_sublayer,
)
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.graph import Node, Linear, Concatenate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.relu import ReLU


class _TrackingResidualStreamMap(ResidualStreamMap):
    """Residual map that records the layer at which each node is freed.

    Used by the heuristic warm-start probe in ``forward_compile`` to
    capture cancel-layer hints for CP-SAT.  The probe sets
    ``current_layer`` before each ``schedule_layer`` call; ``free``
    records that value for every node that gets freed.  Nodes
    consumed via ``reassign`` (free-add path) don't go through
    ``free`` and are correctly omitted from the cancel hint.

    **Rollback-free correction.**  ``LayerScheduler`` speculatively
    allocates a node, then rolls the allocation back with ``free(node)``
    when the op can't be committed under the dirty-cancel / head budget
    (``scheduler.py`` ``residual_map.free(node); continue``).  That
    rollback is not a real death — the node is re-allocated (reborn) in
    a later layer.  Recording the rollback layer as the node's cancel
    produced a hint with ``cancel < birth`` (e.g. freed at layer 6,
    actually born at layer 7), which is hard-infeasible in the CP-SAT
    model.  To keep the emitted ``hint_cancel`` internally consistent,
    a node's stale cancel record is cleared whenever it is re-allocated
    or reborn via ``reassign``: a node is only ever (re)allocated after
    a premature rollback free, so any prior cancel for it must be
    discarded and re-recorded by its genuine later free (or omitted
    entirely if it dies by reassign, per the free-add design above).
    """

    def __init__(self, base: ResidualStreamMap) -> None:
        super().__init__(base.d)
        # Copy state from the cloned base map.  We can't reuse base
        # directly because the tracking subclass is a different type.
        self._free = set(base._free)
        self._node_to_indices = dict(base._node_to_indices)
        self._reserved = set(getattr(base, "_reserved", set()))
        self.current_layer: int = 0
        self.cancel_layer: dict[int, int] = {}

    def allocate(self, node: Node):  # type: ignore[override]
        # A node reaching allocate after a recorded free was rolled back
        # (its earlier free was premature); drop the stale cancel so the
        # genuine death — a later free, or omission if it dies by
        # reassign — is what the hint reflects.
        self.cancel_layer.pop(node.node_id, None)
        return super().allocate(node)

    def free(self, node: Node) -> None:  # type: ignore[override]
        self.cancel_layer[node.node_id] = self.current_layer
        super().free(node)

    def reassign(self, old_node: Node, new_node: Node) -> None:  # type: ignore[override]
        # The new node is born here (free-add reuse), so any stale cancel
        # recorded for it by an earlier rolled-back allocation is wrong.
        self.cancel_layer.pop(new_node.node_id, None)
        super().reassign(old_node, new_node)


def _run_heuristic_warm_start(
    graph: GraphAnalyzer,
    d: int,
    d_head: int,
    pos_encoding: PosEncoding,
    d_hidden: int,
    residual_map: ResidualStreamMap,
    computed: Set[Node],
    clusters,
    admission_budget_fraction: float,
    policy: Optional[SchedulingPolicy],
    overlay_pinned_inputs: Set[Node],
    output_node: Node,
    max_layers: int,
) -> tuple[dict, dict, dict, int]:
    """Run the heuristic LayerScheduler in schedule-only mode and
    capture per-node layer, routing, and cancel-layer hints for
    CP-SAT.  Mutates a *clone* of ``residual_map`` and ``computed``;
    the caller's state is untouched.

    Returns ``(hint_layers, hint_routing, hint_cancel, hint_n_layers)``.
    On heuristic deadlock, all dicts are empty and n_layers is 0 — the
    CP-SAT solve will then cold-start without a hint.
    """
    hint_rmap = _TrackingResidualStreamMap(copy.deepcopy(residual_map))
    hint_computed = set(computed)
    hint_scheduler = LayerScheduler(
        graph,
        d,
        d_head,
        pos_encoding,
        d_hidden=d_hidden,
        clusters=clusters,
        admission_budget_fraction=admission_budget_fraction,
        policy=policy,
        pinned_nodes=overlay_pinned_inputs,
        # The warm-start must hand CP-SAT a *model-representable* schedule.
        # Eager (within-layer) freeing produces a shallower schedule that frees
        # and reuses a column inside a consumer's layer — a density the
        # layer-granular CP-SAT model cannot express, so the eager schedule is
        # an infeasible hint and CP-SAT cold-searches and times out.  The
        # no-eager schedule is deeper but feasible, giving the solver a real
        # incumbent to improve from (d=4096/d_head=32: 87 -> ~81 vs the eager
        # heuristic's 85).  The heuristic *fallback* below stays eager.
        eager_free=False,
    )
    hint_layers: dict = {}
    hint_routing: dict = {}
    for hi in range(max_layers):
        if output_node in hint_computed:
            break
        hint_rmap.current_layer = hi
        prev_hint = set(hint_computed)
        try:
            attn_ops, mlp_ops, _ = hint_scheduler.schedule_layer(
                hint_rmap, hint_computed
            )
        except RuntimeError:
            # Heuristic deadlocked / no progress.  Drop the hint
            # and let CP-SAT cold-start.
            return {}, {}, {}, 0
        # Routing decisions for standalone Linears: heuristic placed
        # compute_linear in attention or compute_linear_bypass in
        # MLP.  Chain Linears are non-flex (always MLP) so we don't
        # hint them.
        for op in attn_ops:
            if op.op_type == "compute_linear" and op.node is not None:
                hint_routing[op.node.node_id] = "attn"
        for op in mlp_ops:
            if op.op_type == "compute_linear_bypass":
                hint_routing[op.node.node_id] = "mlp"
        for node in graph.get_all_nodes():
            if isinstance(node, Concatenate) and node not in hint_computed:
                if all(
                    leaf in hint_computed
                    for leaf in flatten_concat_nodes([node])
                ):
                    hint_computed.add(node)
        for n in hint_computed - prev_hint:
            hint_layers[n.node_id] = hi
        if not attn_ops and not mlp_ops:
            break
    hint_n_layers = max(hint_layers.values()) + 1 if hint_layers else 0
    return hint_layers, hint_routing, dict(hint_rmap.cancel_layer), hint_n_layers


def _effective_consumers(graph: GraphAnalyzer, node: Node) -> Set[Node]:
    """Walk through Concatenate consumers transparently.

    Mirrors ``LayerScheduler._get_effective_consumers``: a Concatenate is
    not a real consumer — its own consumers are.  Terminal Concatenates
    (output nodes) are kept so their leaves never get freed.
    """
    result: Set[Node] = set()
    for consumer in graph.get_consumers(node):
        if isinstance(consumer, Concatenate):
            downstream = _effective_consumers(graph, consumer)
            if downstream:
                result |= downstream
            else:
                result.add(consumer)
        else:
            result.add(consumer)
    return result


def _verify_end_of_layer_liveness(
    graph: GraphAnalyzer,
    residual_map,
    computed: Set[Node],
    layer_idx: int,
) -> None:
    """Invariant A (cross-layer): every node with uncomputed effective
    consumers must still be allocated in the residual map.

    Gated behind ``TW_COMPILER_VERIFY=1`` because the O(|nodes|·fanout)
    walk is noticeable on large compiles.  When enabled, surfaces a
    freed-too-early node at the end of the layer where it was freed
    rather than later at a consumer's KeyError.
    """
    for node in graph.get_all_nodes():
        if isinstance(node, Concatenate):
            continue
        if node not in computed:
            continue
        uncomputed = _effective_consumers(graph, node) - computed
        if not uncomputed:
            continue
        if residual_map.is_allocated(node):
            continue
        sample = [repr(c) for c in list(uncomputed)[:3]]
        raise AssertionError(
            f"End-of-layer {layer_idx} liveness violation: {node!r} has "
            f"{len(uncomputed)} uncomputed consumer(s) "
            f"(e.g. {sample}) but is not allocated in residual_map. "
            f"free_count={residual_map.get_free_count()}."
        )


def _verify_end_of_layer_writes(
    attn_ops: list,
    mlp_ops: list,
    prev_computed: Set[Node],
    prev_allocated: Set[Node],
    computed: Set[Node],
    residual_map,
    layer_idx: int,
) -> None:
    """Invariant B (within-layer): every node freshly added to
    ``computed_nodes`` this layer that owns residual columns must
    have been written by some op in this layer's ``attn_ops`` or
    ``mlp_ops``.

    Catches the failure mode where the scheduler marks a node
    computed and allocates its columns but emits no op that writes
    a value into them.  Without this check, downstream consumers
    silently read uninitialized data and the compile produces
    wrong values with no error.

    Gated behind ``TW_COMPILER_VERIFY=1`` like its sibling
    :func:`_verify_end_of_layer_liveness` — the walk is cheap but
    the assertion machinery is debug-mode only.
    """
    written_nodes: Set[Node] = set()
    for op in attn_ops:
        if op.op_type == "cancel":
            continue
        if op.node is not None:
            written_nodes.add(op.node)
    for op in mlp_ops:
        if op.node is not None:
            written_nodes.add(op.node)

    newly_computed = computed - prev_computed
    for node in newly_computed:
        if isinstance(node, Concatenate):
            continue
        if not residual_map.is_allocated(node):
            # Exclusive chain L1 and chain ReLU live only in MLP hidden
            # slots; LiteralValue may be folded into output bias without
            # owning residual cols.  No residual write expected.
            continue
        if node in prev_allocated:
            # The node's cols were owned by a prior allocation (e.g.
            # the dead addend in an add_into reassign).  The reassign
            # itself does the value write via the add_into op, which
            # is captured in written_nodes via op.node == add_node.
            # Pre-existing cols don't need a separate write.
            continue
        if node in written_nodes:
            continue
        raise AssertionError(
            f"End-of-layer {layer_idx} write-coverage violation: "
            f"{node!r} was added to computed_nodes and allocated to "
            f"cols {residual_map.get_indices(node)} this layer, but "
            f"no op in attn_ops or mlp_ops wrote a value into those "
            f"cols.  Downstream consumers would read uninitialized "
            f"data.  Emitted ops: "
            f"{[(op.op_type, op.node) for op in attn_ops + mlp_ops]}."
        )


def _verify_overlay_target_protection(
    overlays,
    residual_map,
    pos_encoding: Node,
    overlay_pinned_inputs: Set[Node],
) -> None:
    """Pre-delta-transfer guard: every target column must be safe from reuse.

    A safe column is one of:
      - ``pos_encoding``'s (never freed by the scheduler),
      - owned by a node in ``overlay_pinned_inputs`` (pinned against freeing),
      - reserved in the allocator (never handed out by ``allocate``).

    Any other owner means the allocator reused an overlay target position
    for an unrelated live node — the delta write about to happen would
    silently corrupt that node.  Raising here turns the silent value
    corruption into a loud compile-time error.
    """
    for _out_node, (_in_node, target_cols) in overlays.items():
        for col in target_cols:
            if col in residual_map._reserved:
                continue
            owner = None
            for candidate, cols in residual_map._node_to_indices.items():
                if col in cols:
                    owner = candidate
                    break
            if owner is pos_encoding:
                continue
            if owner in overlay_pinned_inputs:
                continue
            raise AssertionError(
                f"Overlay target column {col} is owned by {owner!r} at "
                f"delta-transfer time — the allocator reused a reserved "
                f"position.  Expected: pos_encoding, a pinned overlay "
                f"input, or an allocator-reserved column."
            )


def _count_heads_by_type(
    attn_ops: list[AttnHeadOp],
    d_head: int,
) -> dict[str, int]:
    """Count attention heads consumed by each op type."""
    counts: dict[str, int] = {}
    for attn_op in attn_ops:
        if attn_op.op_type == "compute_attn":
            assert attn_op.node is not None
            from torchwright.graph.attn import Attn as _Attn

            assert isinstance(attn_op.node, _Attn)
            n = (attn_op.node.d_v + d_head - 1) // d_head
        elif attn_op.op_type == "compute_linear":
            assert attn_op.node is not None
            d_input = len(attn_op.node.inputs[0])
            n = (d_input + d_head - 1) // d_head
        elif attn_op.op_type == "compute_add":
            assert attn_op.node is not None
            n = 2 * ((len(attn_op.node) + d_head - 1) // d_head)
        elif attn_op.op_type == "cancel":
            n = (len(attn_op.target_cols) + d_head - 1) // d_head
        elif attn_op.op_type == "add_into":
            assert attn_op.node is not None
            n = (len(attn_op.node) + d_head - 1) // d_head
        elif attn_op.op_type == "delta_transfer":
            n = (len(attn_op.target_cols) + d_head - 1) // d_head
        else:
            n = 0
        counts[attn_op.op_type] = counts.get(attn_op.op_type, 0) + n
    return counts


def _count_layer_params(
    attn_ops: list[AttnHeadOp],
    mlp_ops: list[MLPOp],
    d: int,
    d_head: int,
) -> int:
    """Count transformer parameters used by one layer's ops.

    Attention ops consume whole heads (4 * d * d_head params each).
    MLP ops consume slots (2*d + 2 params each) or bias entries (1 each).
    The per-slot cost is independent of ``d_hidden`` — each occupied
    hidden slot still costs one column in linear1 plus one row in linear2
    plus two biases.
    """
    params_per_head = 4 * d * d_head

    heads_used = 0
    for attn_op in attn_ops:
        if attn_op.op_type == "compute_attn":
            assert attn_op.node is not None
            from torchwright.graph.attn import Attn as _Attn

            assert isinstance(attn_op.node, _Attn)
            heads_used += (attn_op.node.d_v + d_head - 1) // d_head
        elif attn_op.op_type == "compute_linear":
            assert attn_op.node is not None
            d_input = len(attn_op.node.inputs[0])
            heads_used += (d_input + d_head - 1) // d_head
        elif attn_op.op_type == "compute_add":
            assert attn_op.node is not None
            heads_used += 2 * ((len(attn_op.node) + d_head - 1) // d_head)
        elif attn_op.op_type == "cancel":
            heads_used += (len(attn_op.target_cols) + d_head - 1) // d_head
        elif attn_op.op_type == "add_into":
            assert attn_op.node is not None
            heads_used += (len(attn_op.node) + d_head - 1) // d_head

    slots_used = 0
    bias_entries = 0
    for mlp_op in mlp_ops:
        if mlp_op.mlp_slots:
            slots_used += len(mlp_op.mlp_slots)
        if mlp_op.op_type in ("compute_literal_value", "compute_bias"):
            bias_entries += len(mlp_op.target_cols)

    params_per_slot = 2 * d + 2  # linear1 column + bias + linear2 row + bias
    return heads_used * params_per_head + slots_used * params_per_slot + bias_entries


def forward_compile(
    d: int,
    d_head: int,
    output_node: Node,
    pos_encoding: Optional[PosEncoding] = None,
    verbose: bool = True,
    max_layers: int = 100,
    device: Optional[str] = "auto",
    on_layer_compiled: Optional[Callable[[int, TransformerLayer], None]] = None,
    d_hidden: Optional[int] = None,
    on_node_scheduled: Optional[Callable[[Node, int], None]] = None,
    trim_heads: bool = True,
    overlays: Optional[dict] = None,
    admission_control: bool = False,
    admission_budget_fraction: float = 0.4,
    admission_min_chains: int = 4,
    admission_min_peak_width: int = 32,
    policy: Optional[SchedulingPolicy] = None,
    optimize: int = 0,
    cpsat_costs: Costs = Costs(),
    cpsat_flex_routing: bool = True,
    assume_zero_init: bool = False,
    require_solver: bool = False,
) -> HeadlessTransformer:
    """Compile a computation graph into a HeadlessTransformer.

    Args:
        d: Residual stream dimension.
        d_head: Attention head dimension.
        output_node: The graph node whose value should appear in the output.
        pos_encoding: Positional encoding node (required for attention ops).
        verbose: Print compilation progress.
        max_layers: Safety limit on number of layers.
        device: Target device — "auto" (default) uses GPU if available,
                "cpu"/"cuda" to force, or None to skip moving.
        d_hidden: MLP hidden width per layer (the per-layer pool of
            ``L1->ReLU->L2`` neurons).  Independent of ``d``; defaults
            to ``d`` for backwards compatibility.
        on_layer_compiled: Optional streaming hook, called with
            ``(layer_index, layer)`` after each layer's weights are fully
            written.  The callback may extract weight tensors and then
            null the component weight attributes to reclaim memory
            before the next layer is allocated.  The residual-stream
            state objects (``layer.attn.in_state`` / ``layer.mlp.out_state``)
            stay valid regardless and are consumed later when building
            ``residual_assignment``.
        overlays: Optional dict mapping output_node -> (input_node, target_cols)
            for delta transfer. When provided, a final layer is added that
            transfers each output's value to the specified target columns
            via delta: target += (output - target). This enables overlaid
            I/O where output replaces input in-place.
        optimize: Optimization level. ``0`` (default) uses the
            heuristic :class:`LayerScheduler` and skips CP-SAT
            entirely — fastest compile, predictable layer count.
            Higher levels enable CP-SAT with progressively larger
            time budgets, accepting the best feasible schedule at
            budget exhaustion (no proven-optimal requirement) and
            falling back to the heuristic if CP-SAT finds nothing:

            * ``1``: 60s CP-SAT budget — typical iterative-dev win.
            * ``2``: 180s — closer to optimal on d=3072+ geometry.
            * ``3``: 300s — exhaustive; proves optimality on
              parallelism-rich graphs.

            See ``docs/cpsat_scheduler.md`` for the architecture.
        cpsat_costs: Objective weights for CP-SAT (Pareto navigator).
            Defaults to ``Costs()`` (alpha=1, beta=0, gamma=0 — pure
            layer minimization).  Ignored when ``optimize=0``.
        cpsat_flex_routing: When True (default), CP-SAT picks
            attention vs MLP-bypass for each standalone ``Linear``.
            When False, ``policy.local_in_attention`` pins routing.
            Ignored when ``optimize=0``.
        assume_zero_init: When True, the compile assumes the runtime
            zero-initialises the residual stream (the contract met by
            ``HeadlessTransformer.get_input_res_stream``) and skips
            BIRTH-layer dirty-column cancels for fresh allocations on
            the initially-free pool.  Compiles produced this way are
            ~D/d_head fewer attention heads but break under callers
            that pass a non-zero residual stream to ``forward()``
            directly.  Defaults to False — the conservative behaviour
            that runs BIRTH-layer dirty cancels on every fresh
            allocation regardless of the runtime contract.
        require_solver: When True and ``optimize>0``, raise
            ``RuntimeError`` if CP-SAT returns no usable assignment
            instead of silently falling back to the heuristic.  Use it
            in tests and measurement harnesses that must distinguish a
            real optimizer solve from a fallback (a fallback returns a
            valid-but-unoptimized compile with no error).  When False
            (the default) the compile still falls back, but now emits a
            ``warnings.warn`` so the fallback is never silent.  Ignored
            when ``optimize=0``.

    Returns:
        A HeadlessTransformer whose compute() method reproduces
        output_node.compute() for the same inputs.
    """
    # 1. Analyze graph
    graph = GraphAnalyzer(output_node)
    # GraphAnalyzer may have stripped the output if it was an Assert; use
    # the effective output from here on so the loop's termination check
    # matches the graph's actual terminal node.
    output_node = graph.get_output_node()
    input_nodes = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]

    # Unwrap any Assert keys in the overlays dict.
    if overlays:
        from torchwright.graph.misc import Assert

        unwrapped: dict = {}
        for k, v in overlays.items():
            while isinstance(k, Assert):
                k = k.inputs[0]
            unwrapped[k] = v
        overlays = unwrapped

    # Auto-create pos_encoding if needed (required for attention ops)
    if pos_encoding is None:
        pos_encoding = PosEncoding(17)

    # Same-position copy heads match over the full trig grid, so d_head must
    # admit every trig column.  Reject early with an actionable message
    # rather than failing deep in the weight writer.
    if d_head < pos_encoding.trig_width:
        # Largest odd d_pos with trig_width (= d_pos - 1) <= d_head.
        suggested_d_pos = d_head + 1 if d_head % 2 == 0 else d_head
        raise ValueError(
            f"d_head ({d_head}) must be >= pos_encoding.trig_width "
            f"({pos_encoding.trig_width}). Either raise d_head, or pass a "
            f"smaller positional encoding (an odd d_pos with "
            f"d_pos - 1 <= d_head, e.g. PosEncoding({suggested_d_pos}))."
        )

    if d_hidden is None:
        d_hidden = d

    # 2. Initialize
    net = HeadlessTransformer(d, d_head, pos_encoding, d_hidden=d_hidden)
    # Solver provenance, populated only when CP-SAT runs (optimize>0); stays
    # None for the heuristic path so callers can always query it.
    net.cpsat_solve_stats = None
    residual_map = ResidualStreamMap(d)
    residual_map.allocate(pos_encoding)
    # pos_encoding + input_nodes are populated by get_input_res_stream at
    # forward-time, so those cols are guaranteed clean on entry.
    residual_map.mark_clean(residual_map.get_indices(pos_encoding))
    for node in input_nodes:
        if node is pos_encoding:
            continue
        residual_map.allocate(node)
        residual_map.mark_clean(residual_map.get_indices(node))
    # When the caller asserts the runtime zero-initialises the residual
    # stream (the contract `HeadlessTransformer.get_input_res_stream`
    # already provides), the initially-free pool holds zero on entry —
    # not garbage — so mark it clean and the heuristic skips the
    # BIRTH-layer dirty cancel that would otherwise zero each column
    # before its first additive write.  Subsequent recycled columns
    # return to the clean pool via the cancel ops the heuristic already
    # emits at node death.  Default-off because the compiler is
    # defensive against non-zero callers and we don't reverse that
    # without an explicit opt-in.
    if assume_zero_init:
        residual_map.mark_clean(set(residual_map._free))
    computed = set(input_nodes)

    # Static sibling-cluster analysis for admission control.  When
    # disabled or no clusters are found, the scheduler behaves exactly
    # as it did before admission control was added.
    clusters = None
    if admission_control:
        cluster_analyzer = SiblingClusterAnalyzer(
            graph,
            min_chains=admission_min_chains,
            min_peak_width=admission_min_peak_width,
        )
        clusters = cluster_analyzer.analyze()
        if verbose and not clusters.is_empty():
            print(
                f"  Admission control: {len(clusters.clusters)} cluster(s), "
                f"{sum(len(c.chains) for c in clusters.clusters.values())} "
                f"total chains"
            )

    if policy is None:
        policy = SchedulingPolicy()

    # Protect overlay target columns from reuse by intermediate allocations.
    # The delta-transfer layer at end of compile writes to these columns
    # unconditionally (via attention heads), so any intermediate node
    # whose residual-stream columns intersect target_cols would be silently
    # overwritten.  target_cols are *output-writeback* columns (the residual
    # positions the output reader pulls from), not the overlay input node's
    # allocated columns — they may land in pos_encoding's range, an input's
    # range, or the initially-free overflow region.  Protection strategy per
    # column:
    #   - In pos_encoding's cols: no-op, pos_encoding is never freed.
    #   - In some input node's cols: pin that input so the scheduler never
    #     marks it dead.  Its columns stay allocated to it and can't be
    #     reused by intermediates.
    #   - In _free (overflow region): reserve the column directly so the
    #     allocator never picks it.
    overlay_pinned_inputs: set[Node] = set()
    if overlays:
        col_to_owner: dict[int, Node] = {}
        for owner, cols in residual_map._node_to_indices.items():
            for c in cols:
                col_to_owner[c] = owner

        cols_to_reserve: set[int] = set()
        for _out_node, (_in_node, target_cols) in overlays.items():
            for col in target_cols:
                col_owner = col_to_owner.get(col)
                if col_owner is None:
                    cols_to_reserve.add(col)
                elif col_owner is pos_encoding:
                    pass
                else:
                    overlay_pinned_inputs.add(col_owner)
        if cols_to_reserve:
            residual_map.reserve(cols_to_reserve)

    # Map optimize level to CP-SAT time budget.  Level 0 skips CP-SAT
    # entirely; higher levels accept best-feasible (not proven-optimal)
    # at budget exhaustion and fall back to the heuristic if CP-SAT
    # finds nothing.
    _OPTIMIZE_BUDGETS = {1: 60.0, 2: 180.0, 3: 300.0}
    if optimize not in (0, 1, 2, 3):
        raise ValueError(f"optimize must be 0, 1, 2, or 3 (got {optimize})")
    use_cpsat = optimize > 0
    cpsat_time_budget_s = _OPTIMIZE_BUDGETS.get(optimize, 0.0)

    if use_cpsat:
        # Architecture doc §3 marks admission_control as a model
        # precondition; the CP-SAT cumulative does not represent the
        # sibling-cluster admission constraint, so a solver-feasible
        # schedule may not be replayable.  Surface this explicitly
        # rather than producing a corrupted compile.
        if admission_control:
            raise RuntimeError(
                "optimize>0 (CP-SAT) is incompatible with "
                "admission_control=True (see docs/cpsat_scheduler.md "
                "§3 Model preconditions).  Pass optimize=0 to keep "
                "admission control."
            )

        # Warm-start: run the heuristic scheduler in schedule-only
        # mode on a cloned residual_map / computed set, capturing
        # each schedulable node's layer, routing, and cancel layer.
        # CP-SAT consumes the result via `hint_*`; a complete
        # known-feasible incumbent dramatically shrinks the time the
        # solver needs to find any feasible schedule.
        t_hint_start = time.perf_counter()
        hint_layers, hint_routing, hint_cancel, hint_n_layers = (
            _run_heuristic_warm_start(
                graph=graph,
                d=d,
                d_head=d_head,
                pos_encoding=pos_encoding,
                d_hidden=d_hidden,
                residual_map=residual_map,
                computed=computed,
                clusters=clusters,
                admission_budget_fraction=admission_budget_fraction,
                policy=policy,
                overlay_pinned_inputs=overlay_pinned_inputs,
                output_node=output_node,
                max_layers=max_layers,
            )
        )
        if verbose:
            hint_time = time.perf_counter() - t_hint_start
            print(
                f"  Heuristic warm-start: {hint_n_layers} layers "
                f"({hint_time:.2f}s, {len(hint_layers)} hinted nodes)"
            )
            print(
                f"  CP-SAT solver: costs={cpsat_costs}, "
                f"flex_routing={cpsat_flex_routing}, "
                f"time_budget_s={cpsat_time_budget_s}"
            )

        # Use the heuristic's layer count as the search horizon (with
        # one slack layer) when it's tighter than the user-supplied
        # max_layers.  CP-SAT's variable domain shrinks accordingly,
        # which is a big win for graphs where max_layers >> n_layers.
        solver_max_layers = max_layers
        if hint_n_layers > 0:
            solver_max_layers = min(max_layers, hint_n_layers + 1)

        t_solve_start = time.perf_counter()
        assignment, _stats = solve_schedule(
            output_node,
            pos_encoding,
            d=d,
            d_head=d_head,
            d_hidden=d_hidden,
            costs=cpsat_costs,
            flex_routing=cpsat_flex_routing,
            time_budget_s=cpsat_time_budget_s,
            max_layers=solver_max_layers,
            policy=policy,
            assume_zero_init=assume_zero_init,
            hint_layers=hint_layers if hint_layers else None,
            hint_routing=hint_routing if hint_routing else None,
            hint_cancel=hint_cancel if hint_cancel else None,
            log_search_progress=verbose,
        )
        # Surface the solver provenance on the returned net so callers can
        # distinguish a real solve from a fallback without re-deriving it
        # (``stats.status_name``, ``is_optimal``, the LB/objective gap).
        net.cpsat_solve_stats = _stats
        if assignment is None:
            # CP-SAT found no feasible incumbent within budget — fall
            # back to the heuristic schedule.  The warm-start was a
            # sunk cost; we already know the heuristic produces a
            # valid schedule.
            #
            # A fallback returns a valid-but-unoptimized compile with no
            # error, so a layer-count-only measurement cannot tell it from
            # a real solve.  Surface it loudly (and optionally fatally via
            # ``require_solver``) so it can never masquerade as an
            # optimizer result.  ``UNKNOWN`` means CP-SAT found no incumbent
            # within the budget (expected at small budgets / hard geometries);
            # ``INFEASIBLE`` on a graph the heuristic schedules would indicate
            # a CP-SAT model bug — see ``docs/cpsat_scheduler.md``.  The
            # fallback uses the EAGER heuristic (below), which is typically
            # shallower than the no-eager warm-start hint (``hint_n_layers``).
            fallback_msg = (
                f"CP-SAT returned no usable assignment "
                f"(status={_stats.status_name}, "
                f"{cpsat_time_budget_s:.0f}s budget); falling back to the "
                f"eager heuristic schedule (the no-eager warm-start hint was "
                f"{hint_n_layers} layers; the eager fallback is typically "
                f"shallower). The compile is valid but UNOPTIMIZED — "
                f"optimize>0 did not take effect. status=INFEASIBLE (vs "
                f"UNKNOWN/timeout) on a graph the heuristic schedules would "
                f"indicate a CP-SAT model bug, not a hard instance."
            )
            if require_solver:
                raise RuntimeError(
                    fallback_msg
                    + " (require_solver=True: refusing the silent fallback)"
                )
            warnings.warn(fallback_msg, RuntimeWarning, stacklevel=2)
            if verbose and _stats.solver_log:
                print("--- CP-SAT solver log (last 40 lines) ---")
                for line in _stats.solver_log.splitlines()[-40:]:
                    print(f"  {line}")
                print("--- end CP-SAT solver log ---")
            # Eager heuristic fallback (default eager_free=True): a timeout
            # never regresses below the eager heuristic's depth, even though
            # the CP-SAT hint was the deeper no-eager schedule.
            scheduler = LayerScheduler(
                graph,
                d,
                d_head,
                pos_encoding,
                d_hidden=d_hidden,
                clusters=clusters,
                admission_budget_fraction=admission_budget_fraction,
                policy=policy,
                pinned_nodes=overlay_pinned_inputs,
            )
        else:
            if verbose:
                solve_time = time.perf_counter() - t_solve_start
                print(
                    f"  CP-SAT solved in {solve_time:.2f}s: "
                    f"n_layers={assignment.n_layers}"
                )
            scheduler = DirectedLayerScheduler(
                graph,
                d,
                d_head,
                pos_encoding,
                assignment=assignment,
                d_hidden=d_hidden,
                clusters=clusters,
                admission_budget_fraction=admission_budget_fraction,
                policy=policy,
                pinned_nodes=overlay_pinned_inputs,
            )
    else:
        scheduler = LayerScheduler(
            graph,
            d,
            d_head,
            pos_encoding,
            d_hidden=d_hidden,
            clusters=clusters,
            admission_budget_fraction=admission_budget_fraction,
            policy=policy,
            pinned_nodes=overlay_pinned_inputs,
        )

    # Save input indices before scheduling (scheduling may free/reassign them)
    input_indices: dict[Node, list[int]] = {
        pos_encoding: residual_map.get_indices(pos_encoding)
    }
    for node in input_nodes:
        input_indices[node] = residual_map.get_indices(node)

    graph_params = sum(n.num_params() for n in graph.get_all_nodes())

    # Per-layer tensor capacity (Q/K/V/O attention matrices + linear1/linear2
    # weights & biases).  Computed once instead of via `layer.num_params()` so
    # the verbose log still works after `on_layer_compiled` nulls the layer's
    # weight attributes.  Decomposes as 4*d*d (attention QKVO) +
    # 2*d*d_hidden (rectangular MLP matrices) + d_hidden (linear1 bias) +
    # d (linear2 bias).
    layer_capacity = 4 * d * d + 2 * d * d_hidden + d_hidden + d

    if verbose:
        print(
            f"Compiling {len(graph.get_all_nodes())} graph nodes "
            f"({graph_params:,} params) into d={d} transformer"
        )
        print(
            f"  {'Layer':<8} {'Ops':>8}  {'Layer params':>28}  "
            f"{'Stream in':>10}  {'Stream out':>11}  {'Time':>10}"
        )

    # Per-layer snapshots of ``residual_map._node_to_indices``, one per
    # sublayer boundary.  Consumed by :mod:`torchwright.debug.probe` so
    # it can look up where each graph node lives in the compiled
    # residual stream at intermediate layers.  Each entry is
    # ``(ResidualStreamState, {Node: List[int]})`` keyed by the sublayer
    # whose post-skip tensor carries those values.  Only the
    # post-MLP-sublayer state is captured per layer — attn+mlp are both
    # scheduled inside a single ``schedule_layer`` call, so there is no
    # clean observation point between them.
    sublayer_snapshots: list = []

    # 3. Layer loop — seed with input node params (Embedding, etc.)
    total_params = sum(n.num_params() for n in input_nodes)
    total_layer_time = 0.0
    per_layer_head_counts: list[dict[str, int]] = []
    for i in range(max_layers):
        if output_node in computed:
            break

        verify_compiler = bool(os.environ.get("TW_COMPILER_VERIFY"))
        prev_computed = (
            set(computed) if (on_node_scheduled or verify_compiler) else None
        )
        prev_allocated = (
            set(residual_map.get_allocated_nodes()) if verify_compiler else None
        )
        occupied_before = d - residual_map.get_free_count()

        t_layer_start = time.perf_counter()
        layer = net.add_layer(append=True)
        if isinstance(scheduler, DirectedLayerScheduler):
            scheduler.set_current_layer(i)
        t_schedule_start = time.perf_counter()
        attn_ops, mlp_ops, biased_linears = scheduler.schedule_layer(
            residual_map, computed
        )
        t_attn_start = time.perf_counter()
        write_attn_sublayer(layer, attn_ops, residual_map, pos_encoding)
        t_mlp_start = time.perf_counter()
        write_mlp_sublayer(layer, mlp_ops, residual_map, set(biased_linears))
        t_layer_end = time.perf_counter()

        layer_time = t_layer_end - t_layer_start
        alloc_time = t_schedule_start - t_layer_start
        schedule_time = t_attn_start - t_schedule_start
        attn_time = t_mlp_start - t_attn_start
        mlp_time = t_layer_end - t_mlp_start
        total_layer_time += layer_time

        # Mark Concatenate nodes as computed when all leaf inputs are done
        for node in graph.get_all_nodes():
            if isinstance(node, Concatenate) and node not in computed:
                if all(leaf in computed for leaf in flatten_concat_nodes([node])):
                    computed.add(node)

        if verify_compiler:
            _verify_end_of_layer_writes(
                attn_ops, mlp_ops, prev_computed, prev_allocated,
                computed, residual_map, i,
            )
            _verify_end_of_layer_liveness(graph, residual_map, computed, i)

        if on_node_scheduled is not None and prev_computed is not None:
            for node in computed - prev_computed:
                on_node_scheduled(node, i)

        layer_params = _count_layer_params(attn_ops, mlp_ops, d, d_head)
        per_layer_head_counts.append(_count_heads_by_type(attn_ops, d_head))
        total_params += layer_params
        occupied_after = d - residual_map.get_free_count()

        if verbose:
            n_ops = len(attn_ops) + len(mlp_ops)
            pct_params = 100 * layer_params / layer_capacity if layer_capacity else 0
            pct_before = 100 * occupied_before // d
            pct_after = 100 * occupied_after // d
            mlp_slots = sum(len(op.mlp_slots) for op in mlp_ops if op.mlp_slots)
            print(
                f"  {i:<8} {n_ops:>5} ops  "
                f"{layer_params:>9,}/{layer_capacity:,} ({pct_params:>4.1f}%)  "
                f"{occupied_before:>6}/{d} ({pct_before:>2}%)  "
                f"{occupied_after:>6}/{d} ({pct_after:>2}%)  "
                f"MLP {mlp_slots:>4}/{d_hidden}  "
                f"{layer_time*1000:>7.1f}ms "
                f"(alloc {alloc_time*1000:.0f} sch {schedule_time*1000:.0f} "
                f"attn {attn_time*1000:.0f} mlp {mlp_time*1000:.0f})",
                flush=True,
            )

        # Snapshot the live residual-column assignments at the end of
        # this layer's MLP sublayer.  Copying the dict is deliberate:
        # subsequent layers will mutate residual_map via reassign/free
        # and we need the frozen "as of this state" view.
        sublayer_snapshots.append(
            (
                layer.mlp.out_state,
                {n: list(cols) for n, cols in residual_map._node_to_indices.items()},
            )
        )

        if on_layer_compiled is not None:
            on_layer_compiled(i, layer)
    else:
        raise RuntimeError(
            f"Compilation did not converge in {max_layers} layers. "
            f"{len(graph.get_all_nodes() - computed)} nodes remaining."
        )

    # layer_capacity is constant per layer; avoids touching layer tensors,
    # which may have been freed by on_layer_compiled.
    transformer_params = layer_capacity * len(net.layers)
    if verbose:
        pct_used = 100 * total_params / transformer_params if transformer_params else 0
        print(
            f"\n  {len(net.layers)} layers, "
            f"{total_params:,} / {transformer_params:,} params used "
            f"({pct_used:.1f}%), "
            f"{total_layer_time:.2f}s total layer time"
        )

    # 3b. Delta transfer layer for overlaid I/O
    # When overlays is provided, add a final layer that transfers each output
    # value to the input's columns via delta: target += (output - target).
    if overlays:
        _verify_overlay_target_protection(
            overlays, residual_map, pos_encoding, overlay_pinned_inputs
        )
        delta_layer = net.add_layer(append=True)
        delta_ops = []
        for out_node, (in_node, target_cols) in overlays.items():
            # Source columns: where the output value was computed
            source_cols = residual_map.get_indices(out_node)
            # Subtract columns: same as target (the input columns)
            subtract_cols = target_cols
            delta_ops.append(
                AttnHeadOp(
                    op_type="delta_transfer",
                    node=out_node,
                    target_cols=target_cols,
                    source_cols=source_cols,
                    subtract_cols=subtract_cols,
                )
            )
        write_attn_sublayer(delta_layer, delta_ops, residual_map, pos_encoding)
        per_layer_head_counts.append(_count_heads_by_type(delta_ops, d_head))
        if verbose:
            print(f"  Delta transfer layer: {len(delta_ops)} overlays")
        if on_layer_compiled is not None:
            on_layer_compiled(len(net.layers) - 1, delta_layer)

    # Ensure at least one layer exists for ResidualAssignment states.
    # If compile produced zero layers (trivial graph), run the callback on
    # the placeholder too so every layer in net.layers is consistently in
    # the extracted state.
    if not net.layers:
        fallback_layer = net.add_layer(append=True)
        if on_layer_compiled is not None:
            on_layer_compiled(0, fallback_layer)

    # 4. Build ResidualAssignment bridge from saved input indices
    in_state = net.layers[0].attn.in_state
    out_state = net.layers[-1].mlp.out_state
    # Include the per-sublayer snapshots so the debug probe can look up
    # where each graph node lives in the residual stream at any
    # intermediate point.  The top-level in_state / out_state are still
    # populated with input + output indices for the runtime's
    # get_input_res_stream / compute paths.
    all_states = {in_state, out_state}
    for state, _ in sublayer_snapshots:
        all_states.add(state)
    ra = ResidualAssignment(all_states)
    for node, indices in input_indices.items():
        ra.assign(in_state, node, indices)
    for state, snapshot in sublayer_snapshots:
        for node, cols in snapshot.items():
            ra.assign(state, node, list(cols))
    if isinstance(output_node, Concatenate):
        for leaf in flatten_concat_nodes([output_node]):
            ra.assign(out_state, leaf, residual_map.get_indices(leaf))
    else:
        ra.assign(out_state, output_node, residual_map.get_indices(output_node))
    net.residual_assignment = ra
    net.assert_aliases = graph.get_assert_aliases()

    # This post-loop trim is the in-process (no-callback) path's trim.  The ONNX
    # streaming path trims each layer *inside* ``on_layer_compiled`` (see
    # ``_make_stream_layer_weights_cb``) before reading and NULLing its tensors,
    # so by the time we get here those tensors are already trimmed and nulled —
    # re-trimming would slice ``None``.  Hence the ``on_layer_compiled is None``
    # guard: only the in-process return value still carries live weights to trim.
    if trim_heads and on_layer_compiled is None:
        max_heads = d // d_head
        for layer in net.layers:
            layer.attn.attn.trim_unused_heads()
        if verbose:
            heads_per_layer = [layer.attn.attn.n_heads for layer in net.layers]
            total_before = max_heads * len(net.layers)
            total_after = sum(heads_per_layer)
            saved_params = (total_before - total_after) * 4 * d * d_head
            print(
                f"\n  Head pruning: {total_before - total_after}/{total_before} "
                f"heads pruned ({saved_params:,} params saved)"
            )
            # Aggregate heads by op type across all layers
            totals_by_type: dict[str, int] = {}
            for counts in per_layer_head_counts:
                for op_type, n in counts.items():
                    totals_by_type[op_type] = totals_by_type.get(op_type, 0) + n
            cross_pos = totals_by_type.get("compute_attn", 0)
            self_attn = sum(n for t, n in totals_by_type.items() if t != "compute_attn")
            print(
                f"  Heads by purpose: {cross_pos} cross-position (compute_attn), "
                f"{self_attn} self-attending "
                f"({', '.join(f'{n} {t}' for t, n in sorted(totals_by_type.items()) if t != 'compute_attn')})"
            )
            # Per-layer detail
            print(
                f"\n  {'Layer':<8} {'Heads':>12}  {'KV depth':>10}  {'cross-pos':>10}  {'self-attn':>10}"
            )
            for i, layer in enumerate(net.layers):
                attn = layer.attn.attn
                kv_depth = attn.n_heads * d_head
                counts = (
                    per_layer_head_counts[i] if i < len(per_layer_head_counts) else {}
                )
                cp = counts.get("compute_attn", 0)
                sa = sum(n for t, n in counts.items() if t != "compute_attn")
                print(
                    f"  {i:<8} {attn.n_heads:>4}/{max_heads:<7} "
                    f"{kv_depth:>10}  {cp:>10}  {sa:>10}"
                )

    # Trim trailing unused MLP slots (same idea as head trimming, and the
    # in-process counterpart to the per-layer ``mlp.trim_unused_slots()`` the
    # streaming callback runs).  Skipped when streaming because the callback
    # already trimmed and nulled each layer's MLP weights above.
    if on_layer_compiled is None:
        for layer in net.layers:
            layer.mlp.trim_unused_slots()
        if verbose:
            mlp_before = d_hidden * len(net.layers)
            mlp_after = sum(layer.mlp.d_hidden for layer in net.layers)
            mlp_saved = (mlp_before - mlp_after) * (2 * d + 2)
            print(
                f"\n  MLP trimming: {mlp_before - mlp_after}/{mlp_before} "
                f"slots trimmed ({mlp_saved:,} params saved)"
            )

    if device == "auto":
        net.to(get_device(verbose=verbose))
    elif device is not None:
        net.to(torch.device(device))

    return net
