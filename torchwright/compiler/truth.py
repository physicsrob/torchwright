"""Mechanistic-interpretability metadata for compiled transformer artifacts.

The truth manifest is captured while the compiler still owns both sides of
the lowering boundary and the immutable replay plan.  Everything returned by
this module is JSON data: it deliberately retains no graph nodes or tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

from torchwright.compiler.graph_identity import (
    canonical_ids,
    compiler_code_fingerprint,
    encode_cols,
    nodes_by_canonical_id,
)
from torchwright.graph import Embedding
from torchwright.graph.attn import Attn
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import InputNode, LiteralValue

if TYPE_CHECKING:
    from torchwright.compiler.forward.replay_plan import NodeIndices, ReplayPlan
    from torchwright.compiler.lower import LoweredGraph
    from torchwright.compiler.transformer import HeadlessTransformer
    from torchwright.graph import Node
    from torchwright.graph.value_type import Range

TRUTH_FORMAT = "torchwright.truth.v1"
TRUTH_FILENAME = "torchwright_truth.json"
TRUTH_SCHEMA_FILENAME = "torchwright_truth_v1.schema.json"
TRUTH_SUPPORT_FILENAME = "torchwright_truth_support.npz"

_MATRIX_AXES = {
    "attn.W_Q": ("residual_in", "head"),
    "attn.W_K": ("residual_in", "head"),
    "attn.W_V": ("residual_in", "head"),
    "attn.W_O": ("head", "residual_out"),
    "mlp.W_in": ("residual_in", "hidden"),
    "mlp.W_up": ("residual_in", "hidden"),
    "mlp.W_out": ("hidden", "residual_out"),
}
_INLINE_LITERAL_LIMIT = 256


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bound(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _literal_value(value: float) -> float | str:
    if math.isfinite(value):
        return float(value)
    if math.isnan(value):
        return "nan"
    return "+inf" if value > 0 else "-inf"


def _range_record(value_range: Range) -> dict[str, Any]:
    return {
        "lower": _bound(value_range.lo),
        "upper": _bound(value_range.hi),
        "lower_unbounded": not math.isfinite(value_range.lo),
        "upper_unbounded": not math.isfinite(value_range.hi),
    }


def _tensor_records(node: Node, tensor_hashes: dict[int, str]) -> list[dict[str, Any]]:
    """Describe and content-bind tensor payloads owned by a graph node."""
    import torch

    records = []
    for name, value in sorted(vars(node).items()):
        if not isinstance(value, torch.Tensor):
            continue
        cpu = value.detach().cpu().contiguous()
        digest = tensor_hashes.get(id(value))
        if digest is None:
            digest = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
            tensor_hashes[id(value)] = digest
        record: dict[str, Any] = {
            "name": name,
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype).removeprefix("torch."),
            "sha256": digest,
        }
        if cpu.numel() <= _INLINE_LITERAL_LIMIT and isinstance(node, LiteralValue):
            record["values"] = [
                _literal_value(float(value)) for value in cpu.reshape(-1).tolist()
            ]
        records.append(record)
    return records


def _node_semantics(node: Node) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(node, Embedding):
        result = {
            "input_name": node.input_name,
            "vocab_size": len(node.tokenizer.vocab),
            "embedding_width": node.d_embed,
        }
    elif isinstance(node, InputNode):
        lo, hi = node.declared_value_range
        result = {
            "input_name": node.name,
            "declared_range": {
                "lower": _bound(float(lo)),
                "upper": _bound(float(hi)),
                "lower_unbounded": not math.isfinite(lo),
                "upper_unbounded": not math.isfinite(hi),
            },
        }
    elif isinstance(node, Attn):
        result = {
            "d_qk": node.d_qk,
            "d_v": node.d_v,
            "rope_base": float(node.rope_base),
            "rope_d_rot": node.rope_d_rot,
            "causal": True,
        }
    elif isinstance(node, FFN):
        result = {
            "activation": node.activation,
            "lanes": node.n_lanes,
            "degenerate_up": node.is_degenerate,
        }
    return result


def _graph_record(
    output: Node, prefix: str, tensor_hashes: dict[int, str]
) -> tuple[dict[str, Any], dict[int, str]]:
    canon = canonical_ids(output)
    refs = {raw: f"{prefix}:{cid}" for raw, cid in canon.items()}
    nodes = []
    for cid, node in nodes_by_canonical_id(output).items():
        value_range = node.value_type.value_range
        claimed = (
            None
            if node.claimed_type is None
            else _range_record(node.claimed_type.value_range)
        )
        checks = [
            {
                "kind": check.kind,
                "message": check.message,
                "annotation": check.annotation,
            }
            for check in node.checks
        ]
        record: dict[str, Any] = {
            "id": f"{prefix}:{cid}",
            "op": type(node).__name__,
            "width": len(node),
            "inputs": [refs[inp.node_id] for inp in node.inputs],
            "scheduling_predecessors": sorted(
                refs[pred.node_id]
                for pred in node.scheduling_predecessors
                if pred.node_id in refs
            ),
            "name": node.name or None,
            "annotation": node.annotation,
            "value_contract": {
                "range": _range_record(value_range),
                "claimed_range": claimed,
                "integer": bool(node.integer_claim),
                "checks": checks,
            },
            "parameters": _tensor_records(node, tensor_hashes),
        }
        semantics = _node_semantics(node)
        if semantics:
            record["semantics"] = semantics
        nodes.append(record)
    graph = {"root": refs[output.node_id], "nodes": nodes}
    graph["sha256"] = _sha256_json(graph)
    return graph, refs


def _dense_rects(matrix: str, rows: list[int], cols: list[int]) -> list[dict]:
    return [
        {"matrix": matrix, "axis0": list(row_run), "axis1": list(col_run)}
        for row_run in encode_cols(sorted(rows))
        for col_run in encode_cols(sorted(cols))
    ]


def _diag_rects(matrix: str, rows: list[int], cols: list[int]) -> list[dict]:
    rects = []
    start = 0
    while start < len(rows):
        end = start
        while (
            end + 1 < len(rows)
            and rows[end + 1] == rows[end] + 1
            and cols[end + 1] == cols[end] + 1
        ):
            end += 1
        length = end - start + 1
        rects.append(
            {
                "matrix": matrix,
                "axis0": [rows[start], length],
                "axis1": [cols[start], length],
                "diagonal": True,
            }
        )
        start = end + 1
    return rects


def _operation_role(op_type: str) -> str:
    if op_type in {"compute_attn", "compute_ffn"}:
        return "semantic_compute"
    if op_type in {"cancel", "cancel_bypass", "clear_literal_seed"}:
        return "memory_management"
    return "transport"


def _column_field(value: tuple[int, ...] | None) -> list[list[int]] | None:
    return None if value is None else [list(run) for run in encode_cols(list(value))]


def _assignment_records(
    plan: ReplayPlan, node_refs: dict[int, str]
) -> list[dict[str, Any]]:
    assignment = plan.assignment
    ids = sorted(
        set(assignment.node_to_layer)
        | set(assignment.node_to_cancel_layer)
        | set(assignment.node_to_routing)
    )
    return [
        {
            "node": node_refs[node_id],
            "layer": assignment.node_to_layer.get(node_id),
            "routing": assignment.node_to_routing.get(node_id),
            "cancel_layer": assignment.node_to_cancel_layer.get(node_id),
            "cancel_sublayer": assignment.node_to_cancel_mech.get(node_id, "attn"),
        }
        for node_id in ids
    ]


def _lowering_records(
    source: Node,
    lowered: LoweredGraph,
    lowered_graph: dict[str, Any],
    lowered_refs: dict[int, str],
) -> list[dict[str, Any]]:
    source_mapping = []
    for source_id, source_node in nodes_by_canonical_id(source).items():
        source_ref = f"s:{source_id}"
        copied = lowered.node_map.get(source_node)
        sliced = lowered.slice_map.get(source_node)
        entry: dict[str, Any]
        if copied is not None and copied.node_id in lowered_refs:
            entry = {
                "source": source_ref,
                "status": "whole",
                "lowered": lowered_refs[copied.node_id],
            }
        elif sliced is not None:
            holder, offset, width = sliced
            entry = {
                "source": source_ref,
                "status": "slice",
                "lowered": lowered_refs[holder.node_id],
                "slice": {"offset": int(offset), "width": int(width)},
            }
        else:
            entry = {"source": source_ref, "status": "not_materialized"}
        source_mapping.append(entry)

    inverse_sources: dict[str, list[str]] = {}
    for entry in source_mapping:
        if "lowered" in entry:
            inverse_sources.setdefault(entry["lowered"], []).append(entry["source"])
    for node in lowered_graph["nodes"]:
        node["source_nodes"] = sorted(inverse_sources.get(node["id"], []))
    lowered_graph["sha256"] = _sha256_json(
        {key: value for key, value in lowered_graph.items() if key != "sha256"}
    )
    return source_mapping


def _node_references(
    plan: ReplayPlan, lowered_refs: dict[int, str]
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    unknown = [
        node for _node_id, node in plan.nodes_by_id if node.node_id not in lowered_refs
    ]
    unknown.sort(key=lambda node: (type(node).__name__, node.name, len(node)))
    node_refs = dict(lowered_refs)
    internal_nodes = []
    for index, node in enumerate(unknown):
        ref = f"i:{index}"
        node_refs[node.node_id] = ref
        internal_nodes.append(
            {
                "id": ref,
                "op": type(node).__name__,
                "width": len(node),
                "name": node.name or None,
                "purpose": "compiler_internal",
            }
        )
    return node_refs, internal_nodes


def _plan_layers(
    plan: ReplayPlan,
    node_refs: dict[int, str],
    *,
    d: int,
    d_head: int,
    activation: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    layers = []
    matrices: dict[str, dict[str, Any]] = {}
    for layer_index, layer in enumerate(plan.layers):
        attention_ops = []
        head_cursor = 0
        for op_index, op in enumerate(layer.attention_ops):
            head_count = op.emitted_heads(d_head)
            attention_ops.append(
                {
                    "id": f"L{layer_index}.attn.{op_index}",
                    "type": op.op_type,
                    "role": _operation_role(op.op_type),
                    "node": None if op.node is None else node_refs[op.node.node_id],
                    "heads": [head_cursor, head_count],
                    "target_columns": _column_field(op.target_cols),
                    "source_columns": _column_field(op.source_cols),
                    "source_columns_b": _column_field(op.source_cols_b),
                    "query_source_columns": _column_field(op.q_source_cols),
                    "key_source_columns": _column_field(op.k_source_cols),
                    "reuse_input_index": op.reuse_input_index,
                }
            )
            head_cursor += head_count
        if head_cursor != layer.emitted_attention_heads:
            raise AssertionError("truth capture disagrees with replay head count")
        mlp_ops = [
            {
                "id": f"L{layer_index}.mlp.{op_index}",
                "type": op.op_type,
                "role": _operation_role(op.op_type),
                "node": None if op.node is None else node_refs[op.node.node_id],
                "target_columns": _column_field(op.target_cols),
                "source_columns": _column_field(op.source_cols),
                "source_columns_b": _column_field(op.source_cols_b),
                "hidden_slots": _column_field(op.mlp_slots),
                "reuse_input_index": op.reuse_input_index,
            }
            for op_index, op in enumerate(layer.mlp_ops)
        ]
        layers.append(
            {
                "index": layer_index,
                "active_attention_heads": layer.shape.n_heads,
                "emitted_attention_heads": layer.emitted_attention_heads,
                "active_mlp_neurons": layer.shape.d_hidden,
                "attention_operations": attention_ops,
                "mlp_operations": mlp_ops,
                "newly_computed": [
                    node_refs[node_id] for node_id in layer.newly_computed_ids
                ],
            }
        )
        flat_heads = layer.shape.n_heads * d_head
        shapes = {
            "attn.W_Q": [d, flat_heads],
            "attn.W_K": [d, flat_heads],
            "attn.W_V": [d, flat_heads],
            "attn.W_O": [flat_heads, d],
            "mlp.W_in": [d, layer.shape.d_hidden],
            "mlp.W_out": [layer.shape.d_hidden, d],
        }
        if activation == "swish":
            shapes["mlp.W_up"] = [d, layer.shape.d_hidden]
        for kind, shape in shapes.items():
            axis0, axis1 = _MATRIX_AXES[kind]
            matrices[f"L{layer_index}.{kind}"] = {
                "shape": shape,
                "axis0": axis0,
                "axis1": axis1,
            }
    return layers, matrices


def _placement_records(
    compiled: HeadlessTransformer, node_refs: dict[int, str]
) -> dict[str, list[dict[str, Any]]]:
    placements: dict[str, list[dict[str, Any]]] = {}
    recorder = compiled.placements
    if recorder is None:
        return placements
    for entry in recorder.entries:
        matrix = f"L{entry.layer}.{entry.matrix_kind}"
        key = (
            f"physical:{entry.op_type}"
            if entry.node is None
            else node_refs[entry.node.node_id]
        )
        rects = (
            _diag_rects(matrix, entry.rows, entry.cols)
            if entry.mode == "diag"
            else _dense_rects(matrix, entry.rows, entry.cols)
        )
        placements.setdefault(key, []).extend(rects)
    return placements


def _state_record(
    key: str, values: NodeIndices, node_refs: dict[int, str]
) -> dict[str, Any]:
    return {
        "key": key,
        "nodes": {
            node_refs[node_id]: [list(run) for run in encode_cols(list(cols))]
            for node_id, cols in values
        },
    }


def _residual_access_records(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    read_fields = (
        ("source_columns", "source"),
        ("source_columns_b", "source_b"),
        ("query_source_columns", "query"),
        ("key_source_columns", "key"),
    )
    for layer in layers:
        operations = [
            *layer["attention_operations"],
            *layer["mlp_operations"],
        ]
        for operation in operations:
            reads = [
                {"port": port, "columns": operation[field]}
                for field, port in read_fields
                if operation.get(field) is not None
            ]
            records.append(
                {
                    "operation": operation["id"],
                    "reads": reads,
                    "writes": operation["target_columns"],
                }
            )
    return records


def capture_compile_truth(
    *,
    lowered: LoweredGraph,
    plan: ReplayPlan,
    compiled: HeadlessTransformer,
    d: int,
    d_head: int,
    n_heads: int,
    d_hidden: int,
    trim_heads: bool,
    optimize: int,
) -> dict[str, Any]:
    """Capture exact compiler facts before lowered nodes are re-keyed."""
    source = lowered.source_output_node
    if source is None:
        raise RuntimeError("truth capture requires a source graph")
    tensor_hashes: dict[int, str] = {}
    source_graph, _source_refs = _graph_record(source, "s", tensor_hashes)
    lowered_graph, lowered_refs = _graph_record(lowered.output_node, "l", tensor_hashes)
    node_refs, internal_nodes = _node_references(plan, lowered_refs)
    source_mapping = _lowering_records(source, lowered, lowered_graph, lowered_refs)
    layers, matrices = _plan_layers(
        plan,
        node_refs,
        d=d,
        d_head=d_head,
        activation=compiled.activation,
    )
    placements = _placement_records(compiled, node_refs)
    states = [_state_record("input", plan.input_indices, node_refs)]
    states.extend(
        _state_record(f"L{index}.post", layer.residual_snapshot, node_refs)
        for index, layer in enumerate(plan.layers)
    )
    states.append(_state_record("output", plan.final_indices, node_refs))
    realizations = [
        {
            "node": node_refs[node_id],
            "candidates": list(entry.candidates),
            "selected": entry.resolved,
        }
        for node_id, entry in sorted(compiled.realization_table.entries.items())
    ]

    rms = compiled.rms_norm_spec
    return {
        "format": TRUTH_FORMAT,
        "$schema": f"./{TRUTH_SCHEMA_FILENAME}",
        "build": {
            "compiler_code_sha256": compiler_code_fingerprint(),
            "schedule_fingerprint": compiled.schedule_fingerprint,
            "options": {
                "d_model": d,
                "d_head": d_head,
                "n_heads_capacity": n_heads,
                "d_hidden_capacity": d_hidden,
                "trim_heads": trim_heads,
                "bias": bool(compiled.bias),
                "activation": compiled.activation,
                "optimize": optimize,
                "collapse_univariate": True,
                "collapse_piecewise_linear": True,
            },
        },
        "graphs": {
            "source": source_graph,
            "lowered": lowered_graph,
            "internal_nodes": internal_nodes,
            "realization_map": source_mapping,
        },
        "schedule": {
            "assignment": _assignment_records(plan, node_refs),
            "realizations": realizations,
            "layers": layers,
            "residual_accesses": _residual_access_records(layers),
        },
        "residual_stream": {
            "width": d,
            "constant_one_column": plan.const_one_col,
            "rms_norm_reserved_columns": (
                [] if rms is None else list(rms.reserved_cols)
            ),
            "states": states,
            "state_semantics": (
                "input and post-layer residual streams; the compiler has no "
                "stable observation boundary between attention and MLP"
            ),
        },
        "physical_layout": {
            "matrix_coordinates": "logical x@W compiler coordinates",
            "matrices": matrices,
            "placements": placements,
        },
        "intervention_contract": {
            "node_value_contracts": "graphs.source.nodes[*].value_contract",
            "physical_coordinates": {
                "residual": "schedule.residual_accesses",
                "attention_heads": "schedule.layers[*].attention_operations[*].heads",
                "mlp_neurons": "schedule.layers[*].mlp_operations[*].hidden_slots",
            },
            "guarantee": (
                "The manifest describes the unmodified compiled program. "
                "Interventions outside a node's stated value contract are not "
                "covered by torchwright's semantic or numerical guarantees."
            ),
        },
    }
