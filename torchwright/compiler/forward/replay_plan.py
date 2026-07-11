"""Immutable physical plans produced by directed schedule realization."""

from dataclasses import dataclass
from typing import Literal, Mapping, Optional

from torchwright.compiler.forward.cpsat_scheduler import ScheduleAssignment
from torchwright.compiler.realization import linear_attn_heads
from torchwright.compiler.token_model import LayerShape
from torchwright.graph import Node


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

    def __post_init__(self) -> None:
        for name in (
            "target_cols",
            "source_cols",
            "source_cols_b",
            "q_source_cols",
            "k_source_cols",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

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
        )

    def emitted_heads(self, d_head: int) -> int:
        if self.op_type == "compute_attn":
            assert self.node is not None
            return (self.node.d_v + d_head - 1) // d_head
        if self.op_type == "compute_linear":
            assert self.node is not None
            return linear_attn_heads(self.node, d_head)
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
    ]
    node: Optional[Node]
    target_cols: tuple[int, ...]
    mlp_slots: tuple[int, ...] = ()
    source_cols: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        for name in ("target_cols", "mlp_slots", "source_cols"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    @classmethod
    def from_scheduler_op(cls, op) -> "PlannedMlpOp":
        return cls(
            op.op_type,
            op.node,
            tuple(op.target_cols),
            tuple(op.mlp_slots),
            None if op.source_cols is None else tuple(op.source_cols),
        )

    @property
    def bypass_slot_count(self) -> int:
        # Production dominance uses realized physical pressure, including
        # cancellation bypass lanes that the CP-SAT estimate may omit.
        if self.op_type not in ("compute_linear_bypass", "cancel_bypass"):
            return 0
        return len(self.mlp_slots)


ResidualSnapshot = tuple[tuple[int, tuple[int, ...]], ...]
NodeIndices = tuple[tuple[int, tuple[int, ...]], ...]


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


@dataclass(frozen=True)
class ReplayPlan:
    assignment: ScheduleAssignment
    layers: tuple[PlannedLayer, ...]
    input_indices: NodeIndices
    final_indices: NodeIndices
    nodes_by_id: tuple[tuple[int, Node], ...]
    const_one_col: int

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
) -> tuple[LayerShape, int, int]:
    emitted_heads = sum(op.emitted_heads(d_head) for op in attention_ops)
    bypass_slots = sum(op.bypass_slot_count for op in mlp_ops)
    if not trim_heads:
        shape = LayerShape(d // d_head, d_hidden)
    else:
        last_slot = max((slot for op in mlp_ops for slot in op.mlp_slots), default=0)
        shape = LayerShape(max(emitted_heads, 1), max(last_slot + 1, 1))
    return shape, emitted_heads, bypass_slots
