"""Forward compiler: wires GraphAnalyzer, ResidualStreamMap, LayerScheduler,
and WeightWriter into a complete compilation pipeline.

Produces a HeadlessTransformer that can compute the output node's value
given input values.
"""

import copy
import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

import torch

from torchwright.compiler.device import get_device
from torchwright.compiler.realization import (
    ATTN_TRANSPORT,
    RealizationTable,
    is_schedulable,
    linear_attn_heads,
    usable_hidden_slots,
)
from torchwright.compiler.residual_assignment import ResidualAssignment
from torchwright.compiler.forward.cpsat_scheduler import (
    Costs,
    ScheduleAssignment,
    SolveStats,
    solve_schedule,
)
from torchwright.compiler.forward.schedule_cache import (
    graph_fingerprint,
    load_assignment,
    store_assignment,
)
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import (
    HeldOutputLayout,
    ResidualStreamMap,
)
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
    PlacementRecorder,
    write_attn_sublayer,
    write_mlp_sublayer,
)
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.graph import Add, Attn, Embedding, Node, Linear, Concatenate
from torchwright.graph.node import reserve_node_id_above
from torchwright.graph.misc import LiteralValue

# Default constant magnitude exponent q for the pinned-RMS norm: the smallest
# reserved column holds 2^q (at a power-of-two width every reserved column
# does).  Bit-exactness of the identity requires the data energy to stay below
# the smallest pinned column's reduction half-ULP: roughly Sigma data^2 <
# 2^(2q - 24) (the tighter, partial-sum bound; see docs/plan_rmsnorm.md,
# Constraints).  This bound is GRAPH-DEPENDENT, so q is a tunable knob
# (rms_norm_const_exp); this is only the default.
#
# q=44 (gain 2^39 at d=1024) clears the shipping graph with margin: calculator_v2
# reaches Sigma data^2 ~ 2.6e13 (2^44.6) on its squaring path (999*999, 999+1),
# and 2^(2*44-24) = 2^64 ~ 1.8e19 leaves ~2^19 of headroom.  The earlier q=30
# was validated only on calculator_simple (energy ~1e8) and SILENTLY broke the
# identity on calculator_v2 — raise q (or measure the deepest-layer energy)
# before trusting the norm on a new graph.
_RMS_NORM_CONST_EXP = 44


@dataclass(frozen=True)
class RmsNormSpec:
    """The pinned-constant RMSNorm layout chosen at compile time.

    The norm is the identity.  A few reserved residual columns hold large
    power-of-two constants whose combined energy forces ``rms == 2^m``
    exactly; the uniform gain ``2^m`` cancels it (``x / rms * gain == x``),
    so ``÷rms`` and ``×gain`` are pure fp32 exponent shifts and the identity
    is bit-exact for every float at or above ``~2^(m-126)`` (a channel value
    below that underflows to a denormal under ``÷2^m`` and is not recovered
    by the ``×gain`` — the documented near-zero floor, far below any real
    data).  At a power-of-two width this is one column (two at odd
    ``log2(d)``); the general ``odd·2^k`` layout is derived in
    :func:`_rms_norm_pinned_layout`.  See ``docs/plan_rmsnorm.md`` and
    ``docs/rms_norm_dmodel.md``.

    Attributes:
        reserved_cols: the never-allocated residual columns holding the
            constants; seeded into ``embed_table`` at export time.
        const_values: the per-column constants, one per reserved column
            (each a power of two; the smallest is ``2^q``).
        gain: ``2^m``, the uniform RMSNorm weight (every channel, every norm).
        eps: ``rms_norm_eps`` (below the forced RMS's LSB, so bit-exactness is
            eps-independent in the pinned regime).
    """

    reserved_cols: Tuple[int, ...]
    const_values: Tuple[float, ...]
    gain: float
    eps: float


def rms_norm_width_supported(d: int) -> bool:
    """The ``compile_to_onnx`` width contract for RMSNorm-on exports.

    Supported: any multiple of 1024 up to 16384, or any power of two.
    Every width in the contract has a certified bit-exact pinned layout
    (``docs/rms_norm_dmodel.md`` has the full table); the powers of two
    outside it exist mainly for small test artifacts.  The mechanism
    underneath (:func:`_rms_norm_pinned_layout`) is more general — this
    predicate is the deliberately small public promise.
    """
    if d <= 0:
        return False
    if d % 1024 == 0 and d <= 16384:
        return True
    return (d & (d - 1)) == 0


def _rms_norm_pinned_layout(d: int, const_exp: int) -> Tuple[Tuple[int, ...], int]:
    """Pinned-column value exponents and forced-rms exponent for width ``d``.

    Factor ``d = c·2^k`` with ``c`` odd.  The norm is the bit-exact identity
    when the reduction's mean-of-squares lands exactly on an even power of
    two ``2^(2m)``, which needs total pinned energy ``E = d·2^(2m) =
    c·2^(k+2m)``.  Build ``E`` from the binary expansion of ``c``: bit ``p``
    contributes ``2^(p+k+2m)`` — one column holding ``2^((p+k)/2+m)`` when
    ``p+k`` is even, two equal columns holding ``2^((p+k-1)/2+m)`` when odd.
    ``c`` is odd, so bit 0 is always set and the smallest column is anchored
    at ``2^q`` (``const_exp``), i.e. ``m = q - k//2``.  For ``d = 2^k`` this
    reproduces the original layout exactly: one column ``2^q`` for even
    ``k``, two for odd.

    Returns ``(col_exps, m)`` with ``col_exps`` ascending; the reserved
    column count is ``len(col_exps)`` (at most ``2·popcount(c)``).
    """
    c, k = d, 0
    while c % 2 == 0:
        c //= 2
        k += 1
    m = const_exp - k // 2
    col_exps = []
    for p in range(c.bit_length()):
        if not (c >> p) & 1:
            continue
        e = p + k + 2 * m
        if e % 2 == 0:
            col_exps.append(e // 2)
        else:
            col_exps.extend([(e - 1) // 2] * 2)
    return tuple(sorted(col_exps)), m


def _reserve_rms_norm_columns(
    residual_map: ResidualStreamMap, d: int, eps: float, const_exp: int
) -> RmsNormSpec:
    """Reserve the pinned-constant column(s) and return the norm layout.

    Reserves from the free pool *before* the layer loop, so the columns are
    protected from allocation for the whole compile — never written, hence
    holding their constants at every layer.  The layout (how many columns,
    holding which powers of two) comes from :func:`_rms_norm_pinned_layout`;
    ``const_exp`` is ``q`` (the smallest reserved column holds ``2^q``); pick
    it with margin over the deepest-layer data energy (see
    :data:`_RMS_NORM_CONST_EXP`).  Raises ``ValueError`` when no bit-exact
    layout exists for ``d`` under fp32 arithmetic.
    """
    if d <= 0:
        raise ValueError(f"rms_norm requires a positive residual width; got d={d}.")
    col_exps, m = _rms_norm_pinned_layout(d, const_exp)
    q = const_exp

    # Guard the exposed q / eps knobs so a bad value fails loudly here rather
    # than silently producing a non-identity norm or fp32 overflow downstream.
    # The pinned energy E = d·2^(2m) has its top set bit at exactly
    # bitlen(d)-1 + 2m (E's mantissa is d's odd factor), and fp32 holds any
    # such E up to a top bit of 2^127 — the mantissa-width side is certified
    # by the representability check below.  The boundary is inclusive: the
    # DOOM production config (d=8192, q=63) lands exactly on a 2^127 top bit
    # and must be accepted.
    import numpy as _np

    e_exp_top = d.bit_length() - 1 + 2 * m  # exponent of E's top set bit
    if e_exp_top > 127:
        raise ValueError(
            f"rms_norm_const_exp={q} overflows fp32: the pinned energy "
            f"d·2^(2m) ~ 2^{e_exp_top} exceeds the float32 range at d={d}. "
            f"Pick a smaller q."
        )

    # Certify the fp32 mean arithmetic for this exact layout: the pinned
    # energy must be fp32-representable, and BOTH ways a runtime may compute
    # the mean — true division by d, or multiplication by a rounded
    # reciprocal 1/d — must land exactly on 2^(2m).  (ReduceMean's internal
    # strategy is not contractual across ONNX execution providers.)
    energy_exact = float(d) * 2.0 ** (2 * m)  # exact in float64 for any sane q
    energy32 = _np.float32(energy_exact)
    target_ms = _np.float32(2.0 ** (2 * m))
    ms_div = energy32 / _np.float32(d)
    ms_recip = energy32 * (_np.float32(1.0) / _np.float32(d))
    if float(energy32) != energy_exact or ms_div != target_ms or ms_recip != target_ms:
        raise ValueError(
            f"rms_norm has no bit-exact pinned layout at d={d}: the forced "
            f"mean-of-squares does not land exactly on a power of two under "
            f"fp32 arithmetic (odd factor {d // (d & -d)}). Use a supported "
            f"width (docs/rms_norm_dmodel.md), or pass rms_norm=False to "
            f"disable the norm."
        )
    # The forced mean-of-squares is exactly 2^(2m); eps must sit below its fp32
    # LSB or it shifts ms+eps off the power of two and breaks the identity.
    forced_ms = _np.float32(2.0 ** (2 * m))
    if _np.float32(forced_ms + _np.float32(eps)) != forced_ms:
        raise ValueError(
            f"rms_norm_eps={eps} is too large for the forced RMS (mean-square "
            f"2^{2 * m}): it perturbs ms+eps and breaks the identity. Use a "
            f"smaller eps or a larger rms_norm_const_exp."
        )

    n_const = len(col_exps)
    free_sorted = sorted(residual_map._free)
    if len(free_sorted) < n_const:
        raise RuntimeError(
            f"rms_norm needs {n_const} free residual column(s) to pin the RMS at "
            f"d={d}, but only {len(free_sorted)} are free after seeding inputs."
        )
    # The top free columns are highest-indexed; pos/input nodes allocate low, so
    # these are reliably unused.  reserve() asserts they are in the free pool.
    # Values pair with columns in ascending order on both sides — the pairing
    # must be deterministic so a rebuild (OnnxDebugSession) reproduces the
    # exported embed-table seeds.
    reserved = tuple(free_sorted[-n_const:])
    residual_map.reserve(reserved)
    return RmsNormSpec(
        reserved_cols=reserved,
        const_values=tuple(float(2**ce) for ce in col_exps),
        gain=float(2**m),
        eps=float(eps),
    )


def _certify_rms_norm_energy(ra, spec: RmsNormSpec) -> None:
    """Prove the pinned-constant RMSNorm is the bit-exact identity for this graph.

    The norm is an identity only while the data energy the reduction sees stays
    under the *smallest* pinned column's reduction half-ULP:
    ``Σ data² < min(const_values)²·2⁻²⁴``.  Once data energy reaches that
    half-ULP, some summation order (data accumulating before the smallest
    pinned column joins) no longer rounds the fp32 sum to the exact pinned
    energy, the forced RMS drifts off ``2^m``, and the identity breaks.  The
    smallest column is the binding one: any partial sum already containing a
    pinned column is at least as large, so its absorption threshold is at
    least as forgiving.  This bound is graph-dependent, so certify it here
    from the compiler's static value ranges rather than trusting the ``q``
    default.

    Bound (sound, cancel-agnostic): for each residual column, take the largest
    ``max(lo², hi²)`` of any node ever assigned to it across all sublayer
    snapshots — which also covers a freed-but-not-yet-overwritten column, since
    its stale value came from a node that *was* assigned there — and sum over the
    non-pinned columns.  That upper-bounds ``Σ data²`` at every norm.  Raises
    ``ValueError`` (with the smallest sufficient ``q``) if the bound is not met,
    or if any contributing node lacks a finite value range to certify against.

    Why the post-MLP snapshots suffice (the unstated invariant this leans on):
    the norm physically reduces over three residual states — the layer-input
    residual (= the previous layer's post-MLP snapshot), the post-attn residual
    (read by the pre-MLP norm), and the final post-MLP residual — yet only the
    post-MLP states are snapshotted (``sublayer_snapshots``).  This is still a
    sound upper bound because the scheduler never frees a *written* node during
    the MLP sublayer without either driving its columns to exactly zero (a cancel
    head, so the post-attn value there is 0) or transferring ownership via
    ``reassign`` to a live node that survives to the snapshot.  Eager input-frees
    live only in the attn sublayer and zero before freeing.  So every value live
    at any norm read-point is either captured in some post-MLP snapshot or is
    exactly zero — and the per-column max bounds it.
    """
    reserved = set(spec.reserved_cols)
    const_min = min(spec.const_values)
    budget = const_min * const_min * 2.0**-24  # min(const)²·2⁻²⁴

    col_max: dict = {}
    for state, mapping in ra.mapping.items():
        for node, indices in mapping.items():
            data_cols = [c for c in indices if c not in reserved]
            if not data_cols:
                continue
            vr = node.value_type.value_range
            if not vr.is_finite():
                raise ValueError(
                    f"rms_norm cannot be certified as an identity: node "
                    f"{node!r} has a non-finite value range {vr}, so the "
                    f"residual energy the norm reduces over is unbounded. Give "
                    f"the graph finite bounds, or pass rms_norm=False."
                )
            mag2 = max(vr.lo * vr.lo, vr.hi * vr.hi)
            for c in data_cols:
                if mag2 > col_max.get(c, 0.0):
                    col_max[c] = mag2

    energy_bound = sum(col_max.values())
    if energy_bound >= budget:
        # smallest q with const²·2⁻²⁴ = 2^(2q-24) > energy_bound, +1 bit margin
        import math

        need_q = math.ceil((math.log2(max(energy_bound, 1.0)) + 24) / 2) + 1
        raise ValueError(
            f"rms_norm identity not certified: the graph's residual energy "
            f"upper bound ({energy_bound:.3e}) exceeds the pinned-constant "
            f"budget ({budget:.3e} = const²·2⁻²⁴). The forced RMS would drift "
            f"off its power of two and the norm would stop being the identity. "
            f"Pass rms_norm_const_exp>={need_q} (a larger pinned constant) or "
            f"rms_norm=False."
        )


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
        self._held = set(base._held)
        self._reserved = set(base._reserved)
        self.current_layer: int = 0
        self.cancel_layer: dict[int, int] = {}
        # Which sublayer each free ran in ("attn"/"mlp"), captured alongside
        # ``cancel_layer`` so the warm start hints CP-SAT the mechanism the
        # heuristic actually chose.
        self.cancel_mech: dict[int, str] = {}

    def allocate(self, node: Node):  # type: ignore[override]
        # A node reaching allocate after a recorded free was rolled back
        # (its earlier free was premature); drop the stale cancel so the
        # genuine death — a later free, or omission if it dies by
        # reassign — is what the hint reflects.
        self.cancel_layer.pop(node.node_id, None)
        self.cancel_mech.pop(node.node_id, None)
        return super().allocate(node)

    def free(self, node: Node, mech: str = "attn") -> None:  # type: ignore[override]
        self.cancel_layer[node.node_id] = self.current_layer
        self.cancel_mech[node.node_id] = mech
        super().free(node, mech)

    def hold(self, node: Node, mech: str = "attn"):  # type: ignore[override]
        self.cancel_layer[node.node_id] = self.current_layer
        self.cancel_mech[node.node_id] = mech
        return super().hold(node, mech)

    def reassign(self, old_node: Node, new_node: Node) -> None:  # type: ignore[override]
        # The new node is born here (free-add reuse), so any stale cancel
        # recorded for it by an earlier rolled-back allocation is wrong.
        self.cancel_layer.pop(new_node.node_id, None)
        self.cancel_mech.pop(new_node.node_id, None)
        super().reassign(old_node, new_node)


def _run_heuristic_warm_start(
    graph: GraphAnalyzer,
    d: int,
    d_head: int,
    pos_encoding,
    d_hidden: int,
    residual_map: ResidualStreamMap,
    computed: Set[Node],
    clusters,
    *,
    bias: bool = True,
    admission_budget_fraction: float,
    policy: Optional[SchedulingPolicy],
    realization_table: RealizationTable,
    held_output_layout: Optional[HeldOutputLayout],
    output_node: Node,
    max_layers: int,
) -> tuple[dict, dict, dict, dict, int]:
    """Run the heuristic LayerScheduler in schedule-only mode and
    capture per-node layer, routing, cancel-layer, and cancel-mechanism
    hints for CP-SAT.  Mutates a *clone* of ``residual_map`` and
    ``computed``; the caller's state is untouched.

    Returns ``(hint_layers, hint_routing, hint_cancel, hint_cancel_mech,
    hint_n_layers)``.  On heuristic deadlock, all dicts are empty and
    n_layers is 0 — the CP-SAT solve will then cold-start without a hint.
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
        realization_table=realization_table,
        # The warm start emits the eager (within-layer freeing) schedule — the
        # shallower one that frees and reuses a column inside a consumer's
        # layer.  Units 1 and 2 taught the CP-SAT model to represent exactly
        # that density (gap-0 attention cancels + MLP cancels), so the eager
        # schedule is now a FEASIBLE hint the solver accepts as an incumbent
        # and improves from, rather than the deeper no-eager schedule the
        # solver used to need.  (The no-eager mode has been removed;
        # ``LayerScheduler`` is always eager.)
        bias=bias,
        held_output_layout=held_output_layout,
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
            return {}, {}, {}, {}, 0
        # Routing decisions for standalone Linears: heuristic placed
        # compute_linear in attention or compute_linear_bypass in
        # MLP.  FFNs are non-flex (always the MLP composite) so we
        # don't hint them.
        for op in attn_ops:
            if op.op_type == "compute_linear" and op.node is not None:
                hint_routing[op.node.node_id] = "attn"
        for op in mlp_ops:
            if op.op_type == "compute_linear_bypass":
                hint_routing[op.node.node_id] = "mlp"
        for node in graph.get_all_nodes():
            if isinstance(node, Concatenate) and node not in hint_computed:
                if all(leaf in hint_computed for leaf in flatten_concat_nodes([node])):
                    hint_computed.add(node)
        for n in hint_computed - prev_hint:
            hint_layers[n.node_id] = hi
        if not attn_ops and not mlp_ops:
            break
    hint_n_layers = max(hint_layers.values()) + 1 if hint_layers else 0
    return (
        hint_layers,
        hint_routing,
        dict(hint_rmap.cancel_layer),
        dict(hint_rmap.cancel_mech),
        hint_n_layers,
    )


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


def _check_replay_depth(
    assignment: ScheduleAssignment,
    replay_birth_layer: dict[int, int],
    emitted_layers: int,
) -> None:
    """Replay-depth tripwire (optimality gap #5).

    A faithful ``DirectedLayerScheduler`` replay computes every schedulable
    node at exactly ``assignment.node_to_layer[n]`` and therefore emits
    exactly ``assignment.n_layers`` transformer layers.  A divergence means
    the machine ran a schedule the model never proved optimal — an executor
    gap (a model degree of freedom with no replay executor) or a cost-model
    under-charge that forced a deferral — shipping a valid-but-deeper
    transformer that, absent this check, only the global ``max_layers``
    convergence guard would ever notice.

    Raises ``RuntimeError`` naming the first divergent layer.
    """
    n2l = assignment.node_to_layer
    # Placement divergence: the earliest schedulable node whose replayed
    # birth layer differs from its assigned layer (the executor-slip case).
    first_bad: Optional[tuple[int, int, int, int]] = None
    for node_id, assigned in n2l.items():
        actual = replay_birth_layer.get(node_id)
        if actual is not None and actual != assigned:
            layer = min(assigned, actual)
            if first_bad is None or layer < first_bad[0]:
                first_bad = (layer, node_id, assigned, actual)
    if first_bad is not None:
        layer, node_id, assigned, actual = first_bad
        raise RuntimeError(
            f"Replay-depth tripwire (gap #5): node {node_id} was assigned to "
            f"layer {assigned} but the replay computed it at layer {actual} "
            f"(first divergence at layer {layer}).  The machine ran a schedule "
            f"the CP-SAT model never proved: emitted {emitted_layers} layers, "
            f"assignment claims {assignment.n_layers}."
        )
    if emitted_layers != assignment.n_layers:
        # Placements all matched node_to_layer, yet the emitted depth
        # disagrees with the claimed n_layers: the assignment's own makespan
        # is internally inconsistent (a node assigned at or beyond n_layers).
        makespan = max(n2l.values()) + 1 if n2l else 0
        first_over = min(assignment.n_layers, makespan)
        raise RuntimeError(
            f"Replay-depth tripwire (gap #5): replay emitted {emitted_layers} "
            f"layers but the assignment claims n_layers={assignment.n_layers} "
            f"(node_to_layer makespan {makespan}); first layer beyond the "
            f"claimed depth is {first_over}.  The machine shipped a "
            f"valid-but-deeper transformer than the model proved optimal."
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
        if op.op_type == "cancel_bypass":
            # Zeroes an already-freed node's columns (node=None); not a fresh
            # write, exactly like the attention "cancel" carve-out above.
            continue
        if op.node is not None:
            written_nodes.add(op.node)

    newly_computed = computed - prev_computed
    for node in newly_computed:
        if isinstance(node, Concatenate):
            continue
        if not residual_map.is_allocated(node):
            # A LiteralValue may be folded into an output bias without
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
            n = linear_attn_heads(attn_op.node, d_head)
        elif attn_op.op_type == "compute_add":
            assert attn_op.node is not None
            n = 2 * ((len(attn_op.node) + d_head - 1) // d_head)
        elif attn_op.op_type == "cancel":
            n = (len(attn_op.target_cols) + d_head - 1) // d_head
        elif attn_op.op_type == "add_into":
            assert attn_op.node is not None
            n = (len(attn_op.node) + d_head - 1) // d_head
        else:
            n = 0
        counts[attn_op.op_type] = counts.get(attn_op.op_type, 0) + n
    return counts


def _params_per_slot(d: int, activation: str, bias: bool = True) -> int:
    """Parameters one occupied hidden slot costs, by machine.

    ReLU machine: a linear1 column + bias and a linear2 row + bias
    (``2d + 2``).  Swish machine: gate and up columns + biases and a down
    row + bias (``3d + 3``).  Under ``bias=False`` each per-slot bias entry
    relocates 1:1 to a const-column matrix cell except the output-side one,
    which is shared per layer (the constant lane) rather than per slot —
    ``2d + 1`` / ``3d + 2``.  Independent of ``d_hidden``.
    """
    per_slot = (3 * d + 3) if activation == "swish" else (2 * d + 2)
    return per_slot if bias else per_slot - 1


def _count_layer_params(
    attn_ops: list[AttnHeadOp],
    mlp_ops: list[MLPOp],
    d: int,
    d_head: int,
    activation: str = "relu",
    bias: bool = True,
) -> int:
    """Count transformer parameters used by one layer's ops.

    Attention ops consume whole heads (4 * d * d_head params each).
    MLP ops consume slots (:func:`_params_per_slot` each) or bias
    entries (1 each — under ``bias=False`` these are constant-lane
    down-row cells, same count).
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
            heads_used += linear_attn_heads(attn_op.node, d_head)
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
        if mlp_op.op_type in (
            "compute_literal_value",
            "compute_bias",
            "clear_literal_seed",
        ):
            bias_entries += len(mlp_op.target_cols)

    params_per_slot = _params_per_slot(d, activation, bias)
    return heads_used * params_per_head + slots_used * params_per_slot + bias_entries


def forward_compile(
    d: int,
    d_head: int,
    output_node: Node,
    verbose: bool = True,
    max_layers: int = 100,
    device: Optional[str] = "auto",
    on_layer_compiled: Optional[Callable[[int, TransformerLayer], None]] = None,
    d_hidden: Optional[int] = None,
    on_node_scheduled: Optional[Callable[[Node, int], None]] = None,
    trim_heads: bool = True,
    admission_control: bool = False,
    admission_budget_fraction: float = 0.4,
    admission_min_chains: int = 4,
    admission_min_peak_width: int = 32,
    policy: Optional[SchedulingPolicy] = None,
    optimize: int = 0,
    cpsat_costs: Costs = Costs(),
    cpsat_flex_routing: bool = True,
    require_solver: bool = False,
    rms_norm: bool = False,
    rms_norm_eps: float = 1e-5,
    rms_norm_const_exp: int = _RMS_NORM_CONST_EXP,
    bias: bool = True,
    output_layout_source: Optional[Node] = None,
    _disabled_families: frozenset = frozenset(),
    _solver_seed: Optional[int] = None,
    _force_resolve: bool = False,
    _solve_only: bool = False,
    _solve_budget_s: Optional[float] = None,
    _canonical_cancel_reps: bool = False,
    _pin_cancels: bool = True,
    _solver_params: Optional[Dict[str, object]] = None,
    _drop_decision_strategy: bool = False,
) -> HeadlessTransformer:
    """Compile a computation graph into a HeadlessTransformer.

    Args:
        d: Residual stream dimension.
        d_head: Attention head dimension.
        output_node: The graph node whose value should appear in the output.
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
        optimize: Optimization level. ``0`` (default) uses the
            heuristic :class:`LayerScheduler` and skips CP-SAT
            entirely — fastest compile, predictable layer count.
            Higher levels enable CP-SAT with progressively larger
            time budgets, accepting the best feasible schedule at
            budget exhaustion (no proven-optimal requirement) and
            falling back to the heuristic if CP-SAT finds nothing:

            * ``1``: 60s CP-SAT budget — typical iterative-dev win.
            * ``2``: 180s (+150s folded floor-probe budget) — closer to
              optimal on d=3072+ geometry.
            * ``3``: one continuous 600s solve — the production DOOM
              configuration.  Formerly a 600s iterated descent (180s
              rungs, each re-hinted with the best-so-far); retired
              2026-07-10 because rung rebuilds reset the in-solve LNS
              state that produces the deep improvements under the
              pinned-cancel model (descent median 38 vs single-solve
              median 37 on the d=8192 fixture).

            Every level is a single hinted solve; they differ only in
            budget.  See ``docs/cpsat_scheduler.md`` for the
            architecture.
        cpsat_costs: Objective weights for CP-SAT (Pareto navigator).
            Defaults to ``Costs()`` (alpha=1, beta=0, gamma=0 — pure
            layer minimization).  Ignored when ``optimize=0``.
        cpsat_flex_routing: When True (default), CP-SAT picks
            attention vs MLP-bypass for each standalone ``Linear``.
            When False, ``policy.local_in_attention`` pins routing.
            Ignored when ``optimize=0``.
        require_solver: When True and ``optimize>0``, raise
            ``RuntimeError`` if CP-SAT returns no usable assignment
            instead of silently falling back to the heuristic.  Use it
            in tests and measurement harnesses that must distinguish a
            real optimizer solve from a fallback (a fallback returns a
            valid-but-unoptimized compile with no error).  When False
            (the default) the compile still falls back, but now emits a
            ``warnings.warn`` so the fallback is never silent.  Ignored
            when ``optimize=0``.
        rms_norm: When True, reserve the pinned-constant column(s) and record
            an :class:`RmsNormSpec` on the returned net (``net.rms_norm_spec``)
            so the ONNX exporter can emit a real RMSNorm that acts as the
            identity.  ``d`` must factor as ``odd·2^k`` with a bit-exact
            pinned layout (any power of two, and every multiple of 1024 up
            to 16384, qualifies — see ``docs/rms_norm_dmodel.md``); the
            reservation raises ``ValueError`` otherwise.  Default False —
            the in-process forward never applies the norm (it is a no-op),
            so this only changes the column reservation and the recorded
            spec.
        rms_norm_eps: The RMSNorm epsilon recorded on the spec (Llama default
            ``1e-5``).  Below the forced RMS's LSB, so it does not affect
            bit-exactness.  Ignored when ``rms_norm`` is False.
        rms_norm_const_exp: ``q`` — the smallest reserved column holds
            ``2^q``.  Must give the deepest-layer data energy margin under
            ``2^(2q-24)`` or the identity silently breaks (see
            :data:`_RMS_NORM_CONST_EXP`).  Ignored when ``rms_norm`` is
            False.
        bias: When True (default), MLP biases are physical bias vectors —
            today's behavior, unchanged.  When False, no bias parameters
            exist anywhere in the compiled transformer: every bias write
            folds into the weight matrices against the pinned constant-1
            column — hidden-side biases as const-column rows, output-side
            constants (literals, deferred Linear biases, FFN out biases)
            through the constant lane at hidden slot 0, which both
            schedulers reserve (effective capacity ``d_hidden - 1``; the
            schedule-cache fingerprint keys on the flag).  Machine-
            independent (works on ReLU and swish alike); requires
            ``d_hidden >= 2``.  In-process bias tensors remain as zeros so
            probes and debug work unchanged; the ONNX exporter emits no
            bias initializers.  See docs/no_bias_plan.md.
        output_layout_source: Internal held-bank contract for token export.
            When provided, this must be the graph's unique reachable
            :class:`Embedding`, and the equally wide schedulable output is
            freshly written into the embedding's exact ordered residual
            columns.  The source bank is held (never returned to ordinary
            allocation) between source cancellation and output birth.  A
            direct flex ``Linear`` target is forced to attention so source
            cancellation and output writing can share one attention event;
            a direct MLP-only target is rejected.  Generic headless compiles
            leave this as ``None``.  Token export always supplies it.
        _disabled_families: MEASUREMENT-ONLY (CP-SAT expressiveness plan §0
            gap attribution); never set in production configs.  A non-empty
            set of :data:`cpsat_scheduler.CONSTRAINT_FAMILIES` names puts the
            compile into *solve-only diagnostic* mode: the production solve
            runs with those constraint families disabled and forward_compile
            returns immediately after the solve — before the replay — with
            the assignment on ``net.cpsat_assignment`` and the solve metadata
            on ``net.cpsat_solve_stats``.  A relaxed feasible set is a
            superset of the sound one, so the returned schedule is a valid
            LOWER BOUND on the sound optimum but is NOT sound to replay (it
            can require more residual columns / heads than exist); it is
            never replayed, exported, or written to the schedule cache.  The
            returned net therefore has no compiled layers.
        _solver_seed: MEASUREMENT-ONLY.  When set (and ``optimize>0``), seeds
            the CP-SAT solver's ``random_seed`` so the multi-seed draw
            protocol (§0) can sample the descent lottery reproducibly.
        _force_resolve: MEASUREMENT-ONLY.  Skip the schedule-cache lookup so
            a measured solve always re-solves instead of replaying a cached
            draw (a cached hit would silently replace the measurement).
        _solve_only: MEASUREMENT-ONLY.  Return after the solve (before the
            replay), same as a non-empty ``_disabled_families``, but for the
            SOUND model — the G0 attribution reads the sound n_layers / bound
            through the same real solve path as the relaxed cells, without
            paying the full replay + weight-writing (which at production width
            is the ~30-minute, ~28 GB compile).  The sound schedule IS
            replayable; this only skips the replay for the measurement.
        _canonical_cancel_reps: MEASUREMENT-ONLY (sym1; never set in
            production).  When on, the solve posts two extra implications per
            eligible node that collapse two parked/cancel encoding
            degeneracies (a dangling cancel-mechanism boolean under `parked`,
            and a duplicate never-freed encoding) so search workers stop
            branching on distinctions with no physical meaning.  The removed
            model solutions replay identically to retained ones, so the
            schedule stays SOUND (unlike ``_disabled_families``) and the
            forward pass is unchanged — see ``cpsat_symmetry_sym1_plan.md``.
        _pin_cancels: PRODUCTION DEFAULT (on since 2026-07-10; the
            pinned-cancel ship).  When on, the solve equality-pins every
            non-keep-forever cancel layer to its earliest legal value given
            the cancel mechanism (``cancel_in_mlp`` stays a free decision)
            and does not build the parked/window/widening families; the
            warm start keeps the layer, routing, and cancel-MECHANISM
            hints and drops only the cancel-layer values (forced by
            propagation once the mechanism is decided).  A pure
            restriction of the legacy model — every emitted schedule is a
            valid legacy-model solution and stays SOUND to replay — that
            dominates it on search (d=8192 fixture, 600s single solves:
            pinned median 37, min 33 verified, vs unpinned median 40).
            ``False`` is the escape hatch: it rebuilds the legacy
            window/parked/widening model byte-identically.  See
            ``docs/cpsat_scheduler.md`` and the umbrella
            ``cpsat_pinned_cancel_plan.md``.
        _solver_params: MEASUREMENT-ONLY (C1 solver-parameter sweep); never set
            in production configs — same status as ``_disabled_families``.  A
            dict of arbitrary ``CpSolver.parameters`` fields (scalar or list-
            valued) applied to the solve.  Merged with the ``_solver_seed``
            ``random_seed`` (the seed is applied first).  Shipping a winning
            set is a code default edit, not this hatch (C1 plan Decisions #3).
        _solve_budget_s: MEASUREMENT-ONLY override of the solve's wall
            budget (the production budgets come from ``optimize``); never
            set in production.  Formerly ``_descent_budget_s``, the total
            budget of the retired optimize=3 iterated descent.
        _drop_decision_strategy: MEASUREMENT-ONLY (C1 arm
            ``no_decision_strategy``).  When True, the scheduler's hand-rolled
            critical-path-first ``AddDecisionStrategy`` is not emitted, leaving
            CP-SAT's fixed-search subsolver slot to its default portfolio.
            Cannot change the feasible set (search-only); never set in
            production.

    Returns:
        A HeadlessTransformer whose compute() method reproduces
        output_node.compute() for the same inputs.  In ``_disabled_families``
        solve-only mode the returned net carries only the solve outputs
        (``cpsat_assignment`` / ``cpsat_solve_stats``) and cannot compute.
    """
    # 0. Lowering boundary: certify vocabulary and build the
    # compiler-private copy (docs/lowering_copy_plan.md).  Compilation is
    # a pure function of the source graph: from here on every pass —
    # analysis, scheduling, weight writing, certification — operates on
    # the copy; the caller's nodes are never touched.  The copy arrives
    # with fresh, claim-tightened derived caches by construction (checks
    # and range claims are node metadata and ride the clones).
    from torchwright.compiler.lower import lower

    # Both collapse passes always run (docs/univariate_collapse_plan.md
    # — v1 flipped and its escape hatch retired 2026-07-06; the v2
    # general-PL S1 pass likewise, same day; lower() keeps both kwargs
    # as the internal seams the off-path tests exercise).  They change
    # the lowered graph, so the schedule-cache fingerprint follows
    # automatically.
    lowered = lower(
        output_node,
        verbose=verbose,
        collapse_univariate=True,
        collapse_pl=True,
        collapse_lane_cap=(d_hidden if d_hidden else d) // 4,
    )
    output_node = lowered.output_node

    # 1. Analyze graph (pure analysis)
    graph = GraphAnalyzer(output_node)
    # node_id order: these are allocated into the residual stream below,
    # so their iteration order fixes column assignment — set order keyed
    # on absolute ids would not survive a rebuild or the lowering copy.
    input_nodes = sorted(
        (n for n in graph.get_all_nodes() if graph.is_input_node(n)),
        key=lambda n: n.node_id,
    )

    held_source: Optional[Node] = None
    direct_held_handoff = False
    forced_realization_classes: Dict[int, str] = {}
    if output_layout_source is not None:
        try:
            held_source = lowered.copy_of(output_layout_source)
        except KeyError as exc:
            raise ValueError(
                f"output_layout_source {output_layout_source!r} is not a whole "
                f"reachable value after lowering"
            ) from exc
        embeddings = [n for n in graph.get_all_nodes() if isinstance(n, Embedding)]
        if len(embeddings) != 1 or embeddings[0] is not held_source:
            raise ValueError(
                "held output layout requires exactly one reachable Embedding, "
                f"the requested source; found {[repr(n) for n in embeddings]}"
            )
        if not graph.is_input_node(held_source):
            raise ValueError("held output layout source must be a graph input")
        if output_node is held_source:
            raise ValueError("held output layout source and target must differ")
        if isinstance(output_node, Concatenate) or not is_schedulable(output_node):
            raise ValueError(
                "held output layout requires a non-Concatenate schedulable "
                f"target, got {type(output_node).__name__}"
            )
        if len(held_source) != len(output_node) or len(held_source) == 0:
            raise ValueError(
                f"held output layout width mismatch: source={len(held_source)}, "
                f"target={len(output_node)}"
            )
        direct_leaves = flatten_concat_nodes(list(output_node.inputs))
        direct_held_handoff = held_source in direct_leaves
        if direct_held_handoff:
            if isinstance(output_node, Linear):
                forced_realization_classes[output_node.node_id] = ATTN_TRANSPORT
            elif not isinstance(output_node, (Attn, Add)):
                raise ValueError(
                    "held output layout has no direct MLP handoff: target "
                    f"{output_node!r} directly reads the tied Embedding but "
                    f"{type(output_node).__name__} is MLP-only; wrap the "
                    f"output in an identity Linear (e.g. "
                    f"Linear(out, torch.eye(len(out)))) to give the handoff "
                    f"an attention-transport target"
                )

    # RoPE end state (docs/rope_port_plan.md §1, Phase 5): position is a
    # rotation applied inside attention, not a residual node — there is no
    # PosEncoding.  ``pos_encoding`` is kept as an internal ``None`` so the
    # downstream scheduler / fingerprint / weight-writer call sites that still
    # accept it pass through harmlessly (they no longer read any position
    # substrate).
    pos_encoding = None

    # rotate_half pairs dim p with dim p+d_head/2, so every rotary head (which
    # under the end-state global rotation is *every* head) needs an even d_head.
    if d_head % 2 != 0:
        raise ValueError(
            f"d_head must be even (got {d_head}); rotate_half pairs dim p with "
            f"dim p+d_head/2 and every head is rotary on the global grid."
        )

    # d_rot (vanilla partial rotary) is one global value per model: the runtime
    # rotates every head's front rope_d_rot planes on one shared cos/sin grid (a
    # single rope_freq init in ONNX), so a graph mixing rotary widths cannot
    # compile to one consistent transformer.  Assert it here — before compiling,
    # for both backends — so a mixed graph fails fast and identically instead of
    # diverging in-process (the per-layer component holds one width) or only
    # raising at ONNX export.  Build every head from one RopeConfig.d_rot.
    from torchwright.graph.attn import Attn as _Attn

    d_rots = {n.rope_d_rot for n in graph.get_all_nodes() if isinstance(n, _Attn)}
    if len(d_rots) > 1:
        raise ValueError(
            f"rope_d_rot must be one global value per model, but the graph's Attn "
            f"nodes use {sorted(d_rots)}.  Partial rotary (RopeConfig.d_rot) is "
            f"global — build every head from one RopeConfig."
        )

    # Machine uniformity (docs/swiglu_step2_plan.md, settled decision 1): every
    # compiled network is one machine — all-ReLU or all-swish — selected here
    # from the graph's FFN nodes.  A mixed graph is a compile error, not a
    # warning; a graph with no FFNs compiles to the ReLU machine (the machines
    # differ only in the MLP sublayer's realization of FFN lanes).  Not a
    # numbered invariant: this guards a policy, not allocator soundness.
    from torchwright.graph.ffn import FFN as _FFN

    ffn_nodes = [n for n in graph.get_all_nodes() if isinstance(n, _FFN)]
    activations = {n.activation for n in ffn_nodes}
    if len(activations) > 1:
        raise ValueError(
            f"Mixed FFN activations {sorted(activations)}: every compiled graph "
            f"is uniformly one machine (all-ReLU or all-swish).  Build the whole "
            f"graph from a single op package (ops/relu or ops/swiglu)."
        )
    activation = activations.pop() if activations else "relu"
    if activation == "relu":
        gated = [n for n in ffn_nodes if n.up_proj is not None]
        if gated:
            raise ValueError(
                f"{len(gated)} ReLU FFN node(s) carry gated lanes (up_proj), "
                f"e.g. {gated[0]!r} — the ReLU machine's MLP sublayer has no up "
                f"projection.  Gated lanes require activation='swish'."
            )

    if d_hidden is None:
        d_hidden = d
    if not bias and d_hidden < 2:
        raise ValueError(
            f"bias=False reserves hidden slot 0 for the constant lane, so "
            f"d_hidden must be at least 2 (got {d_hidden})."
        )

    # 2. Initialize
    net = HeadlessTransformer(d, d_head, d_hidden=d_hidden, activation=activation)
    # Emission mode: False means no physical bias parameters (the exporter
    # reads this; in-process bias tensors stay as zeros).
    net.bias = bias
    # Solver provenance, populated only when CP-SAT runs (optimize>0); stays
    # None for the heuristic path so callers can always query it.
    net.cpsat_solve_stats = None
    residual_map = ResidualStreamMap(d)
    # Reserve one constant-1 residual column for the Δ=0 self-match heads behind
    # every Linear/Add/Cancel/add_into.  It is a LiteralValue input node, so all
    # three runtime surfaces initialise it to 1.0 with no new emission code
    # (in-process get_input_res_stream's LiteralValue branch; the ONNX/HF
    # embed-table row fold, token.v6).  It is allocated once and no graph node
    # owns or writes it, so the column holds 1.0 through every attention read.
    # A compiler-internal final-MLP op then clears it before tied readout.  The
    # runtime rotates it by absolute position
    # (rotate_half over d_head), making the self-match logit ∝ Σ_p cos((i−j)·θ_p)
    # peak at i==j — a bit-identical Δ=0 transport.
    # Before minting any compile-internal node, push the global id counter past
    # every graph node so the mint cannot share an id (and thus compare equal)
    # with a graph node.  In production the counter is already ahead (it is
    # monotonic and never reset), so this is a no-op; it only fires when a test
    # has reset the counter to 0 and re-compiles a long-lived graph, where a
    # naive mint would collide with a graph node and corrupt the residual map
    # (invariant I1).  See ``reserve_node_id_above``.
    reserve_node_id_above(graph.get_all_nodes())
    const_one: Node = LiteralValue(torch.ones(1), name="rope_self_match_const_one")
    residual_map.allocate(const_one)
    for node in input_nodes:
        residual_map.allocate(node)
    held_output_layout: Optional[HeldOutputLayout] = None
    if held_source is not None:
        held_output_layout = HeldOutputLayout(
            source=held_source,
            target=output_node,
            bank=tuple(residual_map.get_indices(held_source)),
        )
    # The runtime always zero-initialises the residual stream (the contract
    # `HeadlessTransformer.get_input_res_stream` provides, and the ONNX
    # embed-table's zero-scatter into non-allocated columns), so every
    # initially-free column holds zero on entry — not garbage — and a fresh
    # allocation's first additive write needs no BIRTH-layer cancel.  Recycled
    # columns are zeroed by the death-cancel the scheduler emits at node death.
    computed = set(input_nodes)

    # Pinned-constant RMSNorm: reserve the constant column(s) NOW, before the
    # layer loop, so they are never allocated (hence never written) and hold the
    # constant at every layer.  Off by default — a norm-off compile reserves
    # nothing and is unchanged.  The reserved (column, value) + gain are recorded
    # on the net for the ONNX exporter to fold into embed_table and the gain
    # weights; the in-process forward never applies the norm (it is the identity).
    net.rms_norm_spec = None
    if rms_norm:
        net.rms_norm_spec = _reserve_rms_norm_columns(
            residual_map, d, rms_norm_eps, rms_norm_const_exp
        )

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

    # Map optimize level to CP-SAT time budget.  Level 0 skips CP-SAT
    # entirely; higher levels accept best-feasible (not proven-optimal)
    # at budget exhaustion and fall back to the heuristic if CP-SAT
    # finds nothing.  Levels 1/2 run a single warm-start solve
    # (60 s / 330 s incl. the folded floor-probe budget).  Level 3 is a
    # 600 s in-compile ITERATED DESCENT: rung 0 is the heuristic-hinted
    # solve, and each later rung re-solves hinted with the best-so-far at
    # horizon best+1 (the only rung mechanism the plan found to work), all
    # within one forward_compile call — uniform per compile, no cross-compile
    # state (Rob's no-preseed rule), first-seen graphs behave identically.
    _OPTIMIZE_BUDGETS = {1: 60.0, 2: 180.0, 3: 600.0}
    if optimize not in (0, 1, 2, 3):
        raise ValueError(f"optimize must be 0, 1, 2, or 3 (got {optimize})")
    use_cpsat = optimize > 0
    cpsat_time_budget_s = _OPTIMIZE_BUDGETS.get(optimize, 0.0)

    # Columns the pinned-constant RMSNorm reserved out of the free pool above,
    # before scheduling.  The CP-SAT model never sees ``residual_map``, so it
    # must subtract these from its residual budget (and key its schedule cache
    # on them) or it over-counts capacity and can emit a schedule that is
    # infeasible on replay against the reservation-reduced pool.
    n_reserved_residual = (
        len(net.rms_norm_spec.reserved_cols) if net.rms_norm_spec is not None else 0
    )

    # bias=False reserves hidden slot 0 (the constant lane), so the solver's
    # cumulative slot capacity must drop by one — mirroring the heuristic's
    # slot-1 packing base — or a solver-feasible layer that exactly fills
    # d_hidden is unpackable on replay.  The same number decides whether a
    # standalone Linear has an MLP-bypass realization at all, so both
    # resolvers read it: `resolve_static` below, and `routing()` inside the
    # CP-SAT model (which receives it as its `d_hidden`).
    solver_d_hidden = usable_hidden_slots(d_hidden, bias)
    static_realization_table = lowered.realization_table.resolve_static(
        graph.get_all_nodes(),
        policy,
        solver_d_hidden,
        forced_classes=forced_realization_classes,
    )

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

        # ---- Schedule cache (opt-in via TW_SCHEDULE_CACHE_DIR) ----
        # The solve outcome depends only on topology + geometry, so a past
        # win replays directly; the fingerprint keys on node ids and
        # therefore misses whenever graph construction changed.  A hit
        # skips the warm start and the solver entirely and is surfaced as
        # status_name="CACHED" — never silently.
        assignment = None
        hint_n_layers = 0
        t_solve_start = time.perf_counter()
        schedule_fp = graph_fingerprint(
            output_node,
            d=d,
            d_head=d_head,
            d_hidden=d_hidden if d_hidden else d,
            flex_routing=cpsat_flex_routing,
            cancel_slack=2,
            policy=policy,
            reserve_residual=n_reserved_residual,
            bias=bias,
            held_output_source=held_source,
            held_output_target=(output_node if held_source is not None else None),
        )
        if verbose:
            # Print the schedule fingerprint unconditionally (not only on a
            # cache hit/store) so a solve-only measurement emits the graph's
            # identity for the §0 graph-identity gate (compare to `make
            # compile`'s fingerprint).
            print(f"  CP-SAT schedule fingerprint: {schedule_fp}")
        # Measurement modes bypass the cache: _force_resolve forces a fresh
        # solve, and a relaxed (_disabled_families) solve must never read a
        # cached SOUND schedule (same fingerprint) as if it were the relaxed
        # answer.
        cached = (
            None
            if (_force_resolve or _disabled_families)
            else load_assignment(schedule_fp, output_node, min_optimize=optimize)
        )
        if cached is not None:
            assignment, _cached_meta = cached
            if verbose:
                print(
                    f"  CP-SAT schedule cache HIT ({schedule_fp[:12]}...): "
                    f"n_layers={assignment.n_layers} (originally "
                    f"{_cached_meta.get('status_name', '?')})"
                )
            net.cpsat_solve_stats = SolveStats(
                status_name="CACHED",
                objective_value=assignment.n_layers,
                best_objective_bound=float(
                    _cached_meta.get("best_objective_bound", -1)
                ),
                wall_time_s=0.0,
                solver_log="",
                total_attn_heads=-1,
                total_mlp_bypass_slots=-1,
                is_optimal=bool(_cached_meta.get("is_optimal", False)),
            )

        if assignment is None:
            # Warm-start: run the heuristic scheduler in schedule-only
            # mode on a cloned residual_map / computed set, capturing
            # each schedulable node's layer, routing, and cancel layer.
            # CP-SAT consumes the result via `hint_*`; a complete
            # known-feasible incumbent dramatically shrinks the time the
            # solver needs to find any feasible schedule.
            t_hint_start = time.perf_counter()
            hint_layers, hint_routing, hint_cancel, hint_cancel_mech, hint_n_layers = (
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
                    realization_table=static_realization_table,
                    held_output_layout=held_output_layout,
                    output_node=output_node,
                    max_layers=max_layers,
                    bias=bias,
                )
            )
            # ---- Warm-start solve (the sole CP-SAT solve) ----
            # Every optimize level is ONE continuous hinted solve; the levels
            # differ only in budget.
            #
            # 2026-07-08: the floor-probe phase was removed.  It formerly ran
            # a COLD solve at horizon critical_path+1 (budget
            # min(150s, cpsat_time_budget_s)) to try to find and prove the
            # dependency-floor schedule before the main solve.  On the DOOM
            # production graph it never succeeded — it burned its 150s,
            # returned UNKNOWN, and fell through anyway — so it was removed
            # and its budget folded into level 2's solve (the +150 s below)
            # so total CP-SAT wall time is unchanged.  (It also served as a
            # fast INFEASIBLE classifier for width-bound geometries, measured
            # at d=4096; that classification is lost.)
            #
            # 2026-07-10: level 3's iterated descent was retired with the
            # pinned-cancel default (one 600 s solve replaces 180 s rungs
            # re-hinted with the best-so-far at horizon best+1).  Under the
            # pin the deep improvements arrive as large in-solve LNS jumps
            # (e.g. 41->33 at 143 s), and each rung rebuild burned ~25-35 s
            # of re-presolve and reset exactly that LNS state — measured
            # pinned-descent depths [38,39,37,38,33] median 38 vs pinned
            # single-solve median 37 over 15 draws (umbrella
            # cpsat_pinned_cancel_plan.md, steps 2-3).
            solve_budget_s = cpsat_time_budget_s
            if optimize == 2:
                solve_budget_s = cpsat_time_budget_s + min(150.0, cpsat_time_budget_s)
            if _solve_budget_s is not None:
                # MEASUREMENT-ONLY override; never set in production.
                solve_budget_s = _solve_budget_s

            if verbose:
                hint_time = time.perf_counter() - t_hint_start
                print(
                    f"  Heuristic warm-start: {hint_n_layers} layers "
                    f"({hint_time:.2f}s, {len(hint_layers)} hinted nodes)"
                )
                print(
                    f"  CP-SAT solver: costs={cpsat_costs}, "
                    f"flex_routing={cpsat_flex_routing}, "
                    f"time_budget_s={solve_budget_s}"
                )

            # Merge the reproducible-draw seed with any general parameter
            # overrides (C1 sweep, MEASUREMENT-ONLY).  An explicit
            # ``_solver_params["random_seed"]`` would win, but the sweep passes
            # the seed via ``_solver_seed`` and the arm params separately, so
            # the seed is applied first and never shadowed.
            _merged_solver_params: Optional[Dict[str, object]] = None
            if _solver_seed is not None or _solver_params:
                _merged_solver_params = {}
                if _solver_seed is not None:
                    _merged_solver_params["random_seed"] = _solver_seed
                if _solver_params:
                    _merged_solver_params.update(_solver_params)

            # One hinted solve at a given horizon.  Under the pinned-cancel
            # model (the `_pin_cancels` default) the full four-family hint is
            # passed; the solver keeps layers + routing + cancel MECHANISM
            # and drops only the cancel-layer values (forced by propagation
            # once the mechanism is decided).  The mechanism bits are
            # load-bearing: without them the pinned model completed the hint
            # into ZERO incumbents in 5x600 s on the d=8192 fixture
            # (cpsat_pinned_cancel_plan.md step 2, batch 1).
            def _solve_once(hl, hr, hc, hm, horizon, budget):
                return solve_schedule(
                    output_node,
                    pos_encoding,
                    d=d,
                    d_head=d_head,
                    d_hidden=solver_d_hidden,
                    costs=cpsat_costs,
                    flex_routing=cpsat_flex_routing,
                    time_budget_s=budget,
                    max_layers=horizon,
                    policy=policy,
                    reserve_residual=n_reserved_residual,
                    # Sound by construction: the warm-start hint is feasible,
                    # so it always satisfies the tightened domains (verified:
                    # zero violations at both measured geometries).
                    tighten_domains=True,
                    hint_layers=hl if hl else None,
                    hint_routing=hr if hr else None,
                    hint_cancel=hc if hc else None,
                    hint_cancel_mech=hm if hm else None,
                    log_search_progress=verbose,
                    # Measurement-only knobs (§0 gap attribution, C1 sweep);
                    # empty / None / False in production.
                    _disabled_families=_disabled_families,
                    _canonical_cancel_reps=_canonical_cancel_reps,
                    _pin_cancels=_pin_cancels,
                    solver_params=_merged_solver_params,
                    drop_decision_strategy=_drop_decision_strategy,
                    held_source_id=(
                        held_output_layout.source.node_id
                        if held_output_layout is not None
                        else None
                    ),
                    held_target_id=(
                        held_output_layout.target.node_id
                        if held_output_layout is not None
                        else None
                    ),
                )

            def _horizon_for(hn):
                # The heuristic / best-so-far layer count + 1 slack layer is
                # the search horizon when tighter than max_layers — a SOFT
                # ceiling (the hint at depth hn stays feasible; a hard
                # K-ceiling would kill the hint).
                return min(max_layers, hn + 1) if hn > 0 else max_layers

            assignment, _stats = _solve_once(
                hint_layers,
                hint_routing,
                hint_cancel,
                hint_cancel_mech,
                _horizon_for(hint_n_layers),
                solve_budget_s,
            )
            # Surface the solver provenance on the returned net so
            # callers can distinguish a real solve from a fallback
            # without re-deriving it (``stats.status_name``,
            # ``is_optimal``, the LB/objective gap).
            net.cpsat_solve_stats = _stats

            # The schedule-cache commit happens at the successful END of the
            # compile (after directed replay and its validations), not here —
            # defense in depth so no model/replay defect can turn a transient
            # failed solve into a persistent cached failure.  Consequence: a
            # relaxed (_disabled_families) or solve-only (_solve_only) run
            # returns before that point and never writes production cache
            # state (deliberate — a measurement seam should not populate the
            # cache; pinned by the §5.4 cache tests).

        # Solve-only measurement (§0 gap attribution).  Return the solve
        # outputs now — before building the directed replay — for either a
        # relaxed model (``_disabled_families``: a valid lower bound on the
        # sound optimum, unsound to replay) or the sound model
        # (``_solve_only``: replayable, but the replay is skipped so the
        # attribution reads the sound n_layers without the full compile).
        # ``cpsat_solve_stats`` (set above) carries the proven bound /
        # optimality; ``cpsat_assignment`` carries n_layers.
        if _disabled_families or _solve_only:
            net.cpsat_assignment = assignment
            if verbose:
                claim = (
                    "no feasible incumbent"
                    if assignment is None
                    else (f"n_layers={assignment.n_layers}")
                )
                mode = (
                    f"disabled={sorted(_disabled_families)}"
                    if _disabled_families
                    else "sound"
                )
                print(
                    f"  solve-only measurement ({mode}): {claim}, "
                    f"bound={net.cpsat_solve_stats.best_objective_bound}"
                )
            return net

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
            # Eager heuristic fallback: on a CP-SAT timeout the compile runs the
            # same eager schedule the warm start emitted, so a fallback never
            # regresses below the heuristic's depth.
            scheduler = LayerScheduler(
                graph,
                d,
                d_head,
                pos_encoding,
                d_hidden=d_hidden,
                clusters=clusters,
                admission_budget_fraction=admission_budget_fraction,
                policy=policy,
                # Fallback resolves statically, same as optimize=0.
                realization_table=static_realization_table,
                bias=bias,
                held_output_layout=held_output_layout,
            )
        else:
            if verbose and net.cpsat_solve_stats.status_name != "CACHED":
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
                # The solve is the resolver: its per-node sublayer decisions
                # resolve the lowered table into the artifact the walk reads.
                realization_table=lowered.realization_table.resolve_from_assignment(
                    assignment.node_to_routing
                ),
                bias=bias,
                held_output_layout=held_output_layout,
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
            # The static policy is the optimize=0 resolver.
            realization_table=static_realization_table,
            bias=bias,
            held_output_layout=held_output_layout,
        )

    # Realization completeness: every schedulable node's entry is resolved
    # or explicitly conditional before the walk starts (the walk only reads).
    scheduler.realization_table.check_complete(graph.get_all_nodes())
    net.realization_table = scheduler.realization_table

    # Save input indices before scheduling (scheduling may free/reassign them)
    input_indices: dict[Node, list[int]] = {}
    for node in input_nodes:
        input_indices[node] = residual_map.get_indices(node)
    # const_one is in the in_state so get_input_res_stream / the ONNX+HF
    # constant seed write 1.0 into its column (LiteralValue branch).
    input_indices[const_one] = residual_map.get_indices(const_one)

    graph_params = sum(n.num_params() for n in graph.get_all_nodes())

    # Per-layer tensor capacity (Q/K/V/O attention matrices + MLP weights &
    # biases).  Computed once instead of via `layer.num_params()` so the
    # verbose log still works after `on_layer_compiled` nulls the layer's
    # weight attributes.  Decomposes as 4*d*d (attention QKVO) + the machine's
    # rectangular MLP matrices and hidden biases (ReLU: linear1+linear2,
    # 2*d*d_hidden + d_hidden; swish: gate+up+down, 3*d*d_hidden +
    # 2*d_hidden) + d (output-side bias).
    n_mlp_mats = 3 if activation == "swish" else 2
    layer_capacity = (
        4 * d * d + n_mlp_mats * d * d_hidden + (n_mlp_mats - 1) * d_hidden + d
    )
    if not bias:
        # No bias vectors in the emitted artifact (in-process they remain as
        # zero tensors, which this display capacity does not count).
        layer_capacity = 4 * d * d + n_mlp_mats * d * d_hidden

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

    # Records each op's 2-D weight-matrix occupancy as weights are written
    # (see weight_writer.PlacementRecorder).  Always on — it holds only ints,
    # is O(emitted ops), and the debug sidecar reads it on every export
    # (including schedule-cache hits, where weight writing still runs).
    placement_recorder = PlacementRecorder()

    # 3. Layer loop — seed with input node params (Embedding, etc.)
    total_params = sum(n.num_params() for n in input_nodes)
    total_layer_time = 0.0
    per_layer_head_counts: list[dict[str, int]] = []
    # Replay-depth tripwire (gap #5): on the directed replay, record the
    # layer each schedulable node was actually computed at, so the
    # post-loop check can compare it against ``assignment.node_to_layer``.
    directed_replay = isinstance(scheduler, DirectedLayerScheduler)
    replay_birth_layer: dict[int, int] = {}
    for i in range(max_layers):
        if output_node in computed:
            break

        verify_compiler = bool(os.environ.get("TW_COMPILER_VERIFY"))
        prev_computed = (
            set(computed)
            if (on_node_scheduled or verify_compiler or directed_replay)
            else None
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
        clears_literal_seed = held_output_layout is not None and output_node in computed
        if clears_literal_seed:
            const_col = residual_map.get_indices(const_one)[0]
            mlp_ops.append(MLPOp("clear_literal_seed", const_one, [const_col], []))
        t_attn_start = time.perf_counter()
        placement_recorder.set_layer(i)
        write_attn_sublayer(
            layer,
            attn_ops,
            residual_map,
            pos_encoding,
            recorder=placement_recorder,
            const_one=const_one,
        )
        t_mlp_start = time.perf_counter()
        write_mlp_sublayer(
            layer,
            mlp_ops,
            residual_map,
            set(biased_linears),
            recorder=placement_recorder,
            const_one=const_one,
            bias=bias,
        )
        if clears_literal_seed:
            # The runtime value is now exact zero.  Retire allocator ownership
            # before the final snapshot, while retaining input_indices so all
            # runtime surfaces still seed the pre-layer value to one.
            residual_map.free(const_one)
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
                attn_ops,
                mlp_ops,
                prev_computed,
                prev_allocated,
                computed,
                residual_map,
                i,
            )
            _verify_end_of_layer_liveness(graph, residual_map, computed, i)

        if prev_computed is not None:
            newly_computed = computed - prev_computed
            if directed_replay:
                for node in newly_computed:
                    replay_birth_layer[node.node_id] = i
            if on_node_scheduled is not None:
                for node in newly_computed:
                    on_node_scheduled(node, i)

        layer_params = _count_layer_params(
            attn_ops, mlp_ops, d, d_head, activation, bias
        )
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

    # Replay-depth tripwire (gap #5): the directed replay must emit exactly
    # the depth the CP-SAT assignment claims.  Runs on the solved and the
    # cache-replayed directed paths alike (both set `assignment`); the eager
    # fallback (`assignment is None`) has no claimed depth to check.
    if directed_replay and assignment is not None:
        _check_replay_depth(assignment, replay_birth_layer, len(net.layers))

    if held_output_layout is not None:
        if residual_map.has_held():
            raise AssertionError(
                f"held output bank was never claimed: "
                f"{residual_map.held_columns()[:8]}"
            )
        if residual_map.is_allocated(held_output_layout.source):
            raise AssertionError(
                "held source still owns residual columns at completion"
            )
        actual_output_bank = tuple(residual_map.get_indices(output_node))
        if actual_output_bank != held_output_layout.bank:
            raise AssertionError(
                f"held output bank order changed: expected "
                f"{held_output_layout.bank}, got {actual_output_bank}"
            )
        if residual_map.is_allocated(const_one):
            raise AssertionError("final literal seed was not retired")

    # Schedule-cache commit, deferred from the solve to here — after directed
    # replay, the replay-depth tripwire, and the output-layout validation all
    # passed.  Defense in depth: atomic batch replay is the correctness
    # mechanism, but no future model/replay defect should turn a transient
    # failed solve into a persistent cached failure.  A cache hit
    # (status_name == "CACHED") is not rewritten; the eager fallback
    # (assignment is None) has nothing to store; relaxed and solve-only
    # measurement runs returned before the replay and never reach this line.
    if (
        use_cpsat
        and assignment is not None
        and net.cpsat_solve_stats is not None
        and net.cpsat_solve_stats.status_name != "CACHED"
    ):
        if (
            store_assignment(
                schedule_fp,
                assignment,
                {
                    "status_name": net.cpsat_solve_stats.status_name,
                    "best_objective_bound": (
                        net.cpsat_solve_stats.best_objective_bound
                    ),
                    "is_optimal": net.cpsat_solve_stats.is_optimal,
                    "optimize": optimize,
                    "d": d,
                    "d_head": d_head,
                    "d_hidden": d_hidden if d_hidden else d,
                },
                output_node=output_node,
            )
            and verbose
        ):
            print(
                f"  CP-SAT schedule cached ({schedule_fp[:12]}...): "
                f"n_layers={assignment.n_layers}"
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
    net.placements = placement_recorder
    # Per-layer attention-head usage by op type (``_count_heads_by_type``).
    # Observability only: tests read this to assert a layer's exact head
    # charge (e.g. a collective-handoff layer sitting at full capacity)
    # instead of re-deriving head counts by hand.
    net.per_layer_head_counts = per_layer_head_counts

    # Certify the RMSNorm identity against the graph's value ranges: the norm is
    # the identity only while the data energy stays under the pinned constant's
    # reduction half-ULP, and that is graph-dependent.  Do it now that the full
    # residual assignment exists, so a graph whose energy exceeds the budget
    # fails loudly here instead of silently shipping a non-identity norm.
    # Runs BEFORE the source re-key below so the certification reads the
    # copy's claim-tightened bounds (fresh by construction).
    if net.rms_norm_spec is not None:
        _certify_rms_norm_energy(ra, net.rms_norm_spec)

    # The net speaks source.  Compilation ran on the compiler-private copy,
    # but every node-keyed surface the net exposes — the residual
    # assignment (debug_value, probes, the debug sidecar, the runtime's
    # input/output lookups), the placement recorder, the realization
    # table — is re-keyed here to the *source* nodes, so consumers keep
    # speaking the graph the caller actually holds.  Compile-internal
    # nodes with no source counterpart (the rope self-match constant)
    # stay as they are.
    copy_to_source = {copy: src for src, copy in lowered.node_map.items()}
    for state in ra.mapping:
        old = ra.mapping[state]
        new = {copy_to_source.get(n, n): cols for n, cols in old.items()}
        # Sliced values: a source node whose value survives fusion as a column
        # slice of a widened copy node (the sibling merge) gets the matching
        # slice of the holder's columns, wherever the holder is materialized.
        # This is what keeps an output Concatenate of merged leaves gatherable
        # — its source leaves have no whole-node counterpart in the copy.
        # A holder absent from this state (freed, or orphaned by a later fold)
        # simply contributes no entry, same as any other unmaterialized node.
        for src, (holder, off, w) in lowered.slice_map.items():
            holder_cols = old.get(holder)
            if holder_cols is not None:
                new[src] = holder_cols[off : off + w]
        ra.mapping[state] = new
    for entry in placement_recorder.entries:
        if entry.node is not None:
            entry.node = copy_to_source.get(entry.node, entry.node)
    copy_id_to_source_id = {c.node_id: s.node_id for c, s in copy_to_source.items()}
    net.realization_table = RealizationTable(
        {
            copy_id_to_source_id.get(nid, nid): entry
            for nid, entry in net.realization_table.entries.items()
        }
    )

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
            mlp_saved = (mlp_before - mlp_after) * _params_per_slot(d, activation, bias)
            print(
                f"\n  MLP trimming: {mlp_before - mlp_after}/{mlp_before} "
                f"slots trimmed ({mlp_saved:,} params saved)"
            )

    if device == "auto":
        net.to(get_device(verbose=verbose))
    elif device is not None:
        net.to(torch.device(device))

    return net
