"""Immutable physical plans produced by directed schedule realization."""

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Union, cast

from torchwright.compiler.forward.cpsat_scheduler import ScheduleAssignment
from torchwright.compiler.realization import linear_attn_live_heads
from torchwright.compiler.token_model import LayerShape
from torchwright.compiler.utils import resolve_n_heads
from torchwright.graph import Node

if TYPE_CHECKING:
    from torchwright.graph import Attn


def _freeze_ints(value, field_name: str) -> tuple[int, ...]:
    try:
        frozen = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of integers") from exc
    if any(not isinstance(item, Integral) or isinstance(item, bool) for item in frozen):
        raise TypeError(f"{field_name} must contain only integers")
    result = tuple(int(item) for item in frozen)
    if any(item < 0 for item in result):
        raise ValueError(f"{field_name} cannot contain negative indices")
    return result


def _freeze_node_indices(value, field_name: str, *, sort: bool) -> "NodeIndices":
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be an iterable of node/index pairs"
        ) from exc
    frozen: list[tuple[int, tuple[int, ...]]] = []
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"{field_name} entries must be (node_id, indices) pairs")
        node_id, indices = item
        if not isinstance(node_id, Integral) or isinstance(node_id, bool):
            raise TypeError(f"{field_name} node IDs must be integers")
        node_id = int(node_id)
        if node_id in seen:
            raise ValueError(f"{field_name} contains duplicate node ID {node_id}")
        seen.add(node_id)
        frozen.append((node_id, _freeze_ints(indices, f"{field_name}[{node_id}]")))
    if sort:
        frozen.sort(key=lambda item: item[0])
    return tuple(frozen)


def _check_reuse_input_index(op_type: str, index, reuse_op_types: tuple[str, ...]):
    """A reuse operation carries exactly one valid target-occurrence index
    (0 or 1); every fresh or unrelated operation carries None.  The index is
    occurrence-level, not node-level — for ``add(x, x)`` both addends name
    one node and only the index says which occurrence owns the reused
    columns (docs/plan_additional_mlp_routing.md)."""
    if op_type in reuse_op_types:
        if index not in (0, 1):
            raise ValueError(
                f"{op_type} requires reuse_input_index 0 or 1, got {index!r}"
            )
    elif index is not None:
        raise ValueError(
            f"{op_type} must not carry a reuse_input_index (got {index!r})"
        )


@dataclass(frozen=True)
class PlannedAttentionOp:
    op_type: Literal[
        "compute_attn",
        "compute_linear",
        "compute_add",
        "cancel",
        "add_into",
    ]
    node: Optional[Node]
    target_cols: tuple[int, ...]
    source_cols: Optional[tuple[int, ...]] = None
    source_cols_b: Optional[tuple[int, ...]] = None
    q_source_cols: Optional[tuple[int, ...]] = None
    k_source_cols: Optional[tuple[int, ...]] = None
    #: The reused target occurrence (0 or 1) of an ``add_into``; None on
    #: every other op.  The attention writer does not read it — the physical
    #: trace and the replay-plan validator preserve which input occurrence
    #: was selected (node identity cannot express it for ``add(x, x)``).
    reuse_input_index: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "target_cols",
            "source_cols",
            "source_cols_b",
            "q_source_cols",
            "k_source_cols",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze_ints(value, name))
        if self.op_type not in (
            "compute_attn",
            "compute_linear",
            "compute_add",
            "cancel",
            "add_into",
        ):
            raise ValueError(f"unknown planned attention op {self.op_type!r}")
        if self.node is not None and not isinstance(self.node, Node):
            raise TypeError("planned attention node must be a Node or None")
        _check_reuse_input_index(self.op_type, self.reuse_input_index, ("add_into",))

    @classmethod
    def from_scheduler_op(cls, op) -> "PlannedAttentionOp":
        def freeze(value):
            return None if value is None else tuple(value)

        return cls(
            op.op_type,
            op.node,
            tuple(op.target_cols),
            freeze(op.source_cols),
            freeze(op.source_cols_b),
            freeze(op.q_source_cols),
            freeze(op.k_source_cols),
            reuse_input_index=op.reuse_input_index,
        )

    def emitted_heads(self, d_head: int) -> int:
        if self.op_type == "compute_attn":
            assert self.node is not None
            return sum(
                bool(
                    cast("Attn", self.node)
                    .output_matrix[start : start + d_head]
                    .ne(0)
                    .any()
                )
                for start in range(0, cast("Attn", self.node).d_v, d_head)
            )
        if self.op_type == "compute_linear":
            assert self.node is not None
            return linear_attn_live_heads(self.node, d_head)
        if self.op_type == "compute_add":
            width = len(self.target_cols)
            self_add = self.source_cols == self.source_cols_b
            heads = 0
            for start in range(0, width, d_head):
                chunk = min(d_head, width - start)
                heads += 1 if self_add or 2 * chunk <= d_head else 2
            return heads
        if self.op_type in ("cancel", "add_into"):
            return (len(self.target_cols) + d_head - 1) // d_head
        raise AssertionError(f"unknown planned attention op {self.op_type!r}")


@dataclass(frozen=True)
class PlannedMlpOp:
    op_type: Literal[
        "compute_ffn",
        "compute_literal_value",
        "compute_bias",
        "compute_linear_bypass",
        "cancel_bypass",
        # MLP-routed Adds (docs/plan_additional_mlp_routing.md): the bypass
        # lane pair adds the live addend into a reused dead addend's columns
        # (add_into_bypass) or both addends into fresh columns
        # (compute_add_bypass).
        "add_into_bypass",
        "compute_add_bypass",
        # v6 tied readout: zeroes the folded const-1 seed column on the
        # output's own layer (appended by _build_replay_plan, never by the
        # scheduler walk).
        "clear_literal_seed",
    ]
    node: Optional[Node]
    target_cols: tuple[int, ...]
    mlp_slots: tuple[int, ...] = ()
    source_cols: Optional[tuple[int, ...]] = None
    #: Second addend of ``compute_add_bypass``; None on every other op.
    source_cols_b: Optional[tuple[int, ...]] = None
    #: The reused target occurrence (0 or 1) of an ``add_into_bypass``; None
    #: on every other op.  The live source occurrence is ``1 - index``; the
    #: writer folds that occurrence's deferred bias and must not infer it
    #: from node identity (``add(x, x)``) or post-reassignment ownership.
    reuse_input_index: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("target_cols", "mlp_slots", "source_cols", "source_cols_b"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze_ints(value, name))
        if self.op_type not in (
            "compute_ffn",
            "compute_literal_value",
            "compute_bias",
            "compute_linear_bypass",
            "cancel_bypass",
            "add_into_bypass",
            "compute_add_bypass",
            "clear_literal_seed",
        ):
            raise ValueError(f"unknown planned MLP op {self.op_type!r}")
        if self.node is not None and not isinstance(self.node, Node):
            raise TypeError("planned MLP node must be a Node or None")
        _check_reuse_input_index(
            self.op_type, self.reuse_input_index, ("add_into_bypass",)
        )
        if self.op_type == "compute_add_bypass":
            if self.source_cols is None or self.source_cols_b is None:
                raise ValueError(
                    "compute_add_bypass requires source_cols and source_cols_b"
                )
        elif self.source_cols_b is not None:
            raise ValueError(f"{self.op_type} must not carry source_cols_b")
        if self.op_type == "add_into_bypass" and self.source_cols is None:
            raise ValueError("add_into_bypass requires source_cols")

    @classmethod
    def from_scheduler_op(cls, op) -> "PlannedMlpOp":
        return cls(
            op.op_type,
            op.node,
            tuple(op.target_cols),
            tuple(op.mlp_slots),
            None if op.source_cols is None else tuple(op.source_cols),
            None if op.source_cols_b is None else tuple(op.source_cols_b),
            reuse_input_index=op.reuse_input_index,
        )

    @property
    def bypass_slot_count(self) -> int:
        # Production dominance uses realized physical pressure, including
        # cancellation bypass lanes that the CP-SAT estimate may omit.
        if self.op_type not in (
            "compute_linear_bypass",
            "cancel_bypass",
            "add_into_bypass",
            "compute_add_bypass",
        ):
            return 0
        return len(self.mlp_slots)


ResidualSnapshot = tuple[tuple[int, tuple[int, ...]], ...]
NodeIndices = tuple[tuple[int, tuple[int, ...]], ...]
PlannedOp = Union[PlannedAttentionOp, PlannedMlpOp]


@dataclass(frozen=True)
class PlannedLayer:
    attention_ops: tuple[PlannedAttentionOp, ...]
    mlp_ops: tuple[PlannedMlpOp, ...]
    biased_linear_ids: frozenset[int]
    shape: LayerShape
    residual_snapshot: ResidualSnapshot
    newly_computed_ids: tuple[int, ...]
    emitted_attention_heads: int
    mlp_bypass_slots: int

    def __post_init__(self) -> None:
        attention_ops = tuple(self.attention_ops)
        mlp_ops = tuple(self.mlp_ops)
        if any(not isinstance(op, PlannedAttentionOp) for op in attention_ops):
            raise TypeError("attention_ops must contain PlannedAttentionOp records")
        if any(not isinstance(op, PlannedMlpOp) for op in mlp_ops):
            raise TypeError("mlp_ops must contain PlannedMlpOp records")
        object.__setattr__(self, "attention_ops", attention_ops)
        object.__setattr__(self, "mlp_ops", mlp_ops)
        object.__setattr__(self, "biased_linear_ids", frozenset(self.biased_linear_ids))
        object.__setattr__(
            self,
            "residual_snapshot",
            _freeze_node_indices(
                self.residual_snapshot, "residual_snapshot", sort=True
            ),
        )
        object.__setattr__(
            self,
            "newly_computed_ids",
            _freeze_ints(self.newly_computed_ids, "newly_computed_ids"),
        )
        if any(
            not isinstance(node_id, Integral) or isinstance(node_id, bool)
            for node_id in self.biased_linear_ids
        ):
            raise TypeError("biased_linear_ids must contain only integers")
        object.__setattr__(
            self,
            "biased_linear_ids",
            frozenset(int(node_id) for node_id in self.biased_linear_ids),
        )
        if not isinstance(self.shape, LayerShape):
            raise TypeError("shape must be a LayerShape")
        if self.shape.n_heads < 1 or self.shape.d_hidden < 1:
            raise ValueError("planned layer dimensions must be positive")
        if self.emitted_attention_heads < 0 or self.mlp_bypass_slots < 0:
            raise ValueError("planned resource counts cannot be negative")


@dataclass(frozen=True)
class ReplayPlan:
    assignment: ScheduleAssignment
    layers: tuple[PlannedLayer, ...]
    input_indices: NodeIndices
    final_indices: NodeIndices
    nodes_by_id: tuple[tuple[int, Node], ...]
    const_one_col: int

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, ScheduleAssignment):
            raise TypeError("assignment must be a ScheduleAssignment")
        layers = tuple(self.layers)
        if not layers or any(not isinstance(layer, PlannedLayer) for layer in layers):
            raise TypeError("layers must contain at least one PlannedLayer")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(
            self,
            "input_indices",
            _freeze_node_indices(self.input_indices, "input_indices", sort=False),
        )
        object.__setattr__(
            self,
            "final_indices",
            _freeze_node_indices(self.final_indices, "final_indices", sort=False),
        )

        resolver_items = tuple(self.nodes_by_id)
        resolver: list[tuple[int, Node]] = []
        resolver_ids: set[int] = set()
        for item in resolver_items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("nodes_by_id entries must be (node_id, Node) pairs")
            node_id, node = item
            if not isinstance(node_id, Integral) or isinstance(node_id, bool):
                raise TypeError("nodes_by_id keys must be integers")
            if not isinstance(node, Node):
                raise TypeError("nodes_by_id values must be Node instances")
            node_id = int(node_id)
            if node.node_id != node_id:
                raise ValueError(
                    f"nodes_by_id key {node_id} does not match node ID {node.node_id}"
                )
            if node_id in resolver_ids:
                raise ValueError(f"nodes_by_id contains duplicate node ID {node_id}")
            resolver_ids.add(node_id)
            resolver.append((node_id, node))
        resolver.sort(key=lambda item: item[0])
        object.__setattr__(self, "nodes_by_id", tuple(resolver))
        resolver_by_id = dict(resolver)

        referenced_ids = {
            node_id
            for mapping in (self.input_indices, self.final_indices)
            for node_id, _ in mapping
        }
        for layer in layers:
            referenced_ids.update(layer.biased_linear_ids)
            referenced_ids.update(layer.newly_computed_ids)
            referenced_ids.update(node_id for node_id, _ in layer.residual_snapshot)
            referenced_ids.update(
                op.node.node_id
                for op in cast(
                    "tuple[PlannedOp, ...]",
                    (*layer.attention_ops, *layer.mlp_ops),
                )
                if op.node is not None
            )
        missing = sorted(referenced_ids - resolver_ids)
        if missing:
            raise ValueError(f"nodes_by_id does not resolve node IDs {missing}")
        for layer in layers:
            for op in cast(
                "tuple[PlannedOp, ...]", (*layer.attention_ops, *layer.mlp_ops)
            ):
                if (
                    op.node is not None
                    and resolver_by_id[op.node.node_id] is not op.node
                ):
                    raise ValueError(
                        f"planned op node {op.node.node_id} differs from its resolver"
                    )
        if not isinstance(self.const_one_col, Integral) or isinstance(
            self.const_one_col, bool
        ):
            raise TypeError("const_one_col must be an integer")
        if self.const_one_col < 0:
            raise ValueError("const_one_col cannot be negative")
        object.__setattr__(self, "const_one_col", int(self.const_one_col))

    def node_resolver(self) -> Mapping[int, Node]:
        return dict(self.nodes_by_id)

    @property
    def total_attention_heads(self) -> int:
        return sum(layer.emitted_attention_heads for layer in self.layers)

    @property
    def total_mlp_bypass_slots(self) -> int:
        return sum(layer.mlp_bypass_slots for layer in self.layers)


def planned_layer_shape(
    attention_ops: tuple[PlannedAttentionOp, ...],
    mlp_ops: tuple[PlannedMlpOp, ...],
    *,
    d: int,
    d_head: int,
    d_hidden: int,
    trim_heads: bool,
    n_heads: Optional[int] = None,
) -> tuple[LayerShape, int, int]:
    n_heads = resolve_n_heads(d, d_head, n_heads, require_divisible=False)
    emitted_heads = sum(op.emitted_heads(d_head) for op in attention_ops)
    bypass_slots = sum(op.bypass_slot_count for op in mlp_ops)
    if not trim_heads:
        shape = LayerShape(n_heads, d_hidden)
    else:
        last_slot = max((slot for op in mlp_ops for slot in op.mlp_slots), default=0)
        shape = LayerShape(max(emitted_heads, 1), max(last_slot + 1, 1))
    return shape, emitted_heads, bypass_slots
