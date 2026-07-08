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
from typing import Callable, Optional, Set, Tuple

import torch

from torchwright.compiler.device import get_device
from torchwright.compiler.realization import RealizationTable
from torchwright.compiler.residual_assignment import ResidualAssignment
from torchwright.compiler.forward.cpsat_scheduler import (
    Costs,
    SolveStats,
    critical_path_layers,
    solve_schedule,
)
from torchwright.compiler.forward.schedule_cache import (
    graph_fingerprint,
    load_assignment,
    store_assignment,
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
    PlacementRecorder,
    write_attn_sublayer,
    write_mlp_sublayer,
)
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.graph import Node, Linear, Concatenate
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
        self._reserved = set(getattr(base, "_reserved", set()))
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
        # The warm-start must hand CP-SAT a *model-representable* schedule.
        # Eager (within-layer) freeing produces a shallower schedule that frees
        # and reuses a column inside a consumer's layer — a density the
        # layer-granular CP-SAT model cannot express, so the eager schedule is
        # an infeasible hint and CP-SAT cold-searches and times out.  The
        # no-eager schedule is deeper but feasible, giving the solver a real
        # incumbent to improve from (d=4096/d_head=32: 87 -> ~81 vs the eager
        # heuristic's 85).  The heuristic *fallback* below stays eager.
        eager_free=False,
        bias=bias,
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

    Returns:
        A HeadlessTransformer whose compute() method reproduces
        output_node.compute() for the same inputs.
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
    # embed-table row fold, token.v5).  It is allocated once and never freed — no graph
    # node owns it and no op targets it — so the column holds 1.0 unchanged
    # through every layer.  The runtime rotates it by absolute position
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
    # finds nothing.
    _OPTIMIZE_BUDGETS = {1: 60.0, 2: 180.0, 3: 300.0}
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
    # d_hidden is unpackable on replay.
    solver_d_hidden = d_hidden if bias else d_hidden - 1

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
        )
        cached = load_assignment(schedule_fp, output_node)
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
                    output_node=output_node,
                    max_layers=max_layers,
                    bias=bias,
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

            # ---- Floor probe (optimize >= 2) ----
            # The schedule optimum equals the dependency critical path
            # whenever the residual stream has slack (measured: d=8192
            # optimum 50 = critical path).  A COLD solve at horizon cp+1
            # with tightened domains finds and proves that optimum in
            # ~2 minutes, where warm-start descent is a 50-64 lottery (the
            # hint anchors the search ~20 layers above the floor; exactly
            # the floor leaves no construction slack — the +1 is
            # load-bearing).  When width binds instead (measured: d=4096),
            # the probe is proven INFEASIBLE in seconds, so it doubles as
            # a free regime classifier; a timeout falls through the same
            # way at bounded cost.
            if optimize >= 2:
                cp_layers = critical_path_layers(
                    output_node,
                    pos_encoding,
                    policy=policy,
                    flex_routing=cpsat_flex_routing,
                )
                if cp_layers + 1 < solver_max_layers:
                    probe_budget = min(150.0, cpsat_time_budget_s)
                    if verbose:
                        print(
                            f"  CP-SAT floor probe: horizon {cp_layers + 1} "
                            f"(critical path {cp_layers}), budget "
                            f"{probe_budget:.0f}s"
                        )
                    t_probe = time.perf_counter()
                    assignment, _stats = solve_schedule(
                        output_node,
                        pos_encoding,
                        d=d,
                        d_head=d_head,
                        d_hidden=solver_d_hidden,
                        costs=cpsat_costs,
                        flex_routing=cpsat_flex_routing,
                        time_budget_s=probe_budget,
                        max_layers=cp_layers + 1,
                        policy=policy,
                        reserve_residual=n_reserved_residual,
                        tighten_domains=True,
                        log_search_progress=verbose,
                    )
                    probe_elapsed = time.perf_counter() - t_probe
                    if assignment is not None:
                        net.cpsat_solve_stats = _stats
                        if verbose:
                            print(
                                f"  CP-SAT floor probe SUCCEEDED in "
                                f"{probe_elapsed:.1f}s: "
                                f"n_layers={assignment.n_layers} "
                                f"(status={_stats.status_name})"
                            )
                    elif verbose:
                        print(
                            f"  CP-SAT floor probe {_stats.status_name} in "
                            f"{probe_elapsed:.1f}s — falling back to "
                            f"warm-start descent"
                        )

            if assignment is None:
                t_solve_start = time.perf_counter()
                assignment, _stats = solve_schedule(
                    output_node,
                    pos_encoding,
                    d=d,
                    d_head=d_head,
                    d_hidden=solver_d_hidden,
                    costs=cpsat_costs,
                    flex_routing=cpsat_flex_routing,
                    time_budget_s=cpsat_time_budget_s,
                    max_layers=solver_max_layers,
                    policy=policy,
                    reserve_residual=n_reserved_residual,
                    # Sound by construction: the warm-start hint is
                    # feasible, so it always satisfies the tightened
                    # domains (verified: zero violations at both measured
                    # geometries).  Saves ~7s presolve + ~9s to first
                    # incumbent on the DOOM graph.
                    tighten_domains=True,
                    hint_layers=hint_layers if hint_layers else None,
                    hint_routing=hint_routing if hint_routing else None,
                    hint_cancel=hint_cancel if hint_cancel else None,
                    hint_cancel_mech=hint_cancel_mech if hint_cancel_mech else None,
                    log_search_progress=verbose,
                )
                # Surface the solver provenance on the returned net so
                # callers can distinguish a real solve from a fallback
                # without re-deriving it (``stats.status_name``,
                # ``is_optimal``, the LB/objective gap).
                net.cpsat_solve_stats = _stats

            if assignment is not None:
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
                # Fallback resolves statically, same as optimize=0.
                realization_table=lowered.realization_table.resolve_static(policy),
                bias=bias,
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
            realization_table=lowered.realization_table.resolve_static(policy),
            bias=bias,
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

        if on_node_scheduled is not None and prev_computed is not None:
            for node in computed - prev_computed:
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
        ra.mapping[state] = {
            copy_to_source.get(n, n): cols for n, cols in ra.mapping[state].items()
        }
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
