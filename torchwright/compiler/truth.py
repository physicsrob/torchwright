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
    canonical_walk,
    compiler_code_fingerprint,
    encode_cols,
)
from torchwright.graph import Embedding
from torchwright.graph.attn import Attn
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import InputNode, LiteralValue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchwright.compiler.forward.replay_plan import NodeIndices, ReplayPlan
    from torchwright.compiler.lower import LoweredGraph
    from torchwright.compiler.transformer import HeadlessTransformer
    from torchwright.graph import Node, OpScopeRecord
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


def sha256_json(value: object) -> str:
    """Hash of the canonical JSON encoding of ``value``.

    This encoding (sorted keys, compact separators, raw unicode) is the
    contract between every truth-manifest hash producer and validator —
    both sides must call this one function or freshly built bundles fail
    their own hash checks.
    """
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
    output: Node,
    prefix: str,
    tensor_hashes: dict[int, str],
    *,
    stamp_hash: bool = True,
) -> tuple[dict[str, Any], dict[int, str]]:
    """Record a graph's nodes in canonical order.

    ``stamp_hash=False`` skips the content hash for a record that will be
    mutated before it is final (the lowered graph gains ``source_nodes``
    in :func:`_lowering_records`, which stamps the hash afterwards).
    """
    walk = canonical_walk(output)
    refs = {node.node_id: f"{prefix}:{cid}" for cid, node in enumerate(walk)}
    nodes = []
    for cid, node in enumerate(walk):
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
    if stamp_hash:
        graph["sha256"] = sha256_json(graph)
    return graph, refs


def _stamp_semantic_regions(
    source: Node,
    node_records: list[dict[str, Any]],
    refs: dict[int, str],
) -> list[dict[str, Any]]:
    """Stamp per-node region membership and build the region table.

    Walks the canonical order (the order ``node_records`` was built in)
    and assigns ``"r:0"``, ``"r:1"``, ... to op-scope records as they are
    first reached.  Each stamped node's ancestor chain is registered
    outermost-first, so a parent always gets a lower id than its children
    and ids are deterministic across rebuilds of the same graph.
    Collecting ancestors is load-bearing, not an optimization:
    composition ops (``linear.subtract`` = add∘negate) stamp zero nodes
    directly, so their records are reachable only as parents of child
    records.  Operand/result node ids that never reached the compiled
    output are dropped rather than emitted as dangling references.
    """
    region_ids: dict[OpScopeRecord, str] = {}
    table: list[dict[str, Any]] = []

    def register(record: OpScopeRecord) -> str:
        chain: list[OpScopeRecord] = []
        cursor: OpScopeRecord | None = record
        while cursor is not None and cursor not in region_ids:
            chain.append(cursor)
            cursor = cursor.parent
        for entry in reversed(chain):
            region_id = f"r:{len(table)}"
            region_ids[entry] = region_id
            table.append(
                {
                    "id": region_id,
                    "op": entry.op,
                    "parent": (
                        None if entry.parent is None else region_ids[entry.parent]
                    ),
                    "params": entry.params,
                    "operands": [
                        refs[node_id]
                        for node_id in entry.operand_ids
                        if node_id in refs
                    ],
                    "results": [
                        refs[node_id] for node_id in entry.result_ids if node_id in refs
                    ],
                    "annotation": entry.annotation,
                }
            )
        return region_ids[record]

    for node, record in zip(canonical_walk(source), node_records, strict=True):
        region = node.op_region
        record["region"] = None if region is None else register(region)
    return table


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


def column_runs(value: Sequence[int] | None) -> list[list[int]] | None:
    """Encode a column index list as the manifest's ``[start, length]`` runs.

    The one run-list encoding every manifest field uses; ``None`` stays
    ``None`` so optional fields serialize as JSON null.
    """
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
    for source_id, source_node in enumerate(canonical_walk(source)):
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
    lowered_graph["sha256"] = sha256_json(
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
    const_one_col: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    layers = []
    matrices: dict[str, dict[str, Any]] = {}
    for layer_index, layer in enumerate(plan.layers):
        attention_ops = []
        head_cursor = 0
        for op_index, op in enumerate(layer.attention_ops):
            head_count = op.emitted_heads(d_head)
            if op.op_type == "compute_attn":
                q_columns = column_runs(op.q_source_cols)
                k_columns = column_runs(op.k_source_cols)
            else:
                # Every non-compute_attn attention op is a Δ=0 self-match:
                # its Q and K both read the reserved constant-1 column
                # (weight_writer._self_match_source), not graph values.
                q_columns = k_columns = column_runs((const_one_col,))
            attention_ops.append(
                {
                    "id": f"L{layer_index}.attn.{op_index}",
                    "type": op.op_type,
                    "role": _operation_role(op.op_type),
                    "node": None if op.node is None else node_refs[op.node.node_id],
                    "heads": [head_cursor, head_count],
                    "target_columns": column_runs(op.target_cols),
                    "source_columns": column_runs(op.source_cols),
                    "source_columns_b": column_runs(op.source_cols_b),
                    "query_source_columns": q_columns,
                    "key_source_columns": k_columns,
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
                "target_columns": column_runs(op.target_cols),
                "source_columns": column_runs(op.source_cols),
                "source_columns_b": column_runs(op.source_cols_b),
                "hidden_slots": column_runs(op.mlp_slots),
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


def _head_slot_maps(plan: ReplayPlan, d_head: int) -> list[dict[int, int]]:
    """Per-layer maps from the writer's untrimmed head slots to trimmed slots.

    The placement recorder captures attention writes in untrimmed head
    coordinates (the ``trim_unused_heads`` contract), while the manifest
    declares every attention matrix at its trimmed checkpoint width.  This
    map translates between the two; an untrimmed slot absent from its
    layer's map is a dead head that does not exist in the checkpoint.
    """
    maps = []
    for layer in plan.layers:
        mapping: dict[int, int] = {}
        written = trimmed = 0
        for op in layer.attention_ops:
            liveness = op.written_head_liveness(d_head)
            if sum(liveness) != op.emitted_heads(d_head):
                raise AssertionError(
                    "written-head liveness disagrees with emitted head count"
                )
            for is_live in liveness:
                if is_live:
                    mapping[written] = trimmed
                    trimmed += 1
                written += 1
        maps.append(mapping)
    return maps


def _translate_head_axis(
    indices: list[int], slot_map: dict[int, int], d_head: int
) -> list[int] | None:
    """Map head-axis matrix indices to trimmed slots; None if the head is dead."""
    slots = {index // d_head for index in indices}
    live = [slot for slot in slots if slot in slot_map]
    if not live:
        return None
    if len(live) != len(slots):
        raise AssertionError("placement entry spans live and dead head slots")
    return [slot_map[index // d_head] * d_head + index % d_head for index in indices]


def _placement_records(
    compiled: HeadlessTransformer,
    node_refs: dict[int, str],
    head_slot_maps: list[dict[int, int]],
    d_head: int,
) -> dict[str, list[dict[str, Any]]]:
    placements: dict[str, list[dict[str, Any]]] = {}
    recorder = compiled.placements
    if recorder is None:
        return placements
    for entry in recorder.entries:
        matrix = f"L{entry.layer}.{entry.matrix_kind}"
        rows, cols = entry.rows, entry.cols
        axis0_kind, axis1_kind = _MATRIX_AXES[entry.matrix_kind]
        if "head" in (axis0_kind, axis1_kind):
            head_axis = rows if axis0_kind == "head" else cols
            translated = _translate_head_axis(
                head_axis, head_slot_maps[entry.layer], d_head
            )
            if translated is None:
                # A dead head's write was trimmed out of the checkpoint;
                # the manifest describes only weights that exist.
                continue
            if axis0_kind == "head":
                rows = translated
            else:
                cols = translated
        key = (
            f"physical:{entry.op_type}"
            if entry.node is None
            else node_refs[entry.node.node_id]
        )
        rects = (
            _diag_rects(matrix, rows, cols)
            if entry.mode == "diag"
            else _dense_rects(matrix, rows, cols)
        )
        placements.setdefault(key, []).extend(rects)
    return placements


def _state_record(
    key: str, values: NodeIndices, node_refs: dict[int, str]
) -> dict[str, Any]:
    return {
        "key": key,
        "nodes": {node_refs[node_id]: column_runs(cols) for node_id, cols in values},
    }


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
    # The source record is hashed only after the region stamp so the
    # source content hash covers semantic regions and per-node membership.
    source_graph, source_refs = _graph_record(
        source, "s", tensor_hashes, stamp_hash=False
    )
    source_graph["semantic_regions"] = _stamp_semantic_regions(
        source, source_graph["nodes"], source_refs
    )
    source_graph["sha256"] = sha256_json(source_graph)
    # The lowered record is mutated (source_nodes) and hashed by
    # _lowering_records below; hashing it here too would be discarded work.
    lowered_graph, lowered_refs = _graph_record(
        lowered.output_node, "l", tensor_hashes, stamp_hash=False
    )
    node_refs, internal_nodes = _node_references(plan, lowered_refs)
    source_mapping = _lowering_records(source, lowered, lowered_graph, lowered_refs)
    layers, matrices = _plan_layers(
        plan,
        node_refs,
        d=d,
        d_head=d_head,
        activation=compiled.activation,
        const_one_col=plan.const_one_col,
    )
    placements = _placement_records(
        compiled, node_refs, _head_slot_maps(plan, d_head), d_head
    )
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
        },
        "residual_stream": {
            "width": d,
            "constant_one_column": plan.const_one_col,
            "constant_one_semantics": (
                "Holds exactly 1.0 at every position. Read as the Δ=0 "
                "self-match query/key source by every attention operation "
                "except compute_attn (their query/key source columns) and, "
                "under bias=False, by MLP bias folds (their W_in/W_up rows "
                "in physical_layout.placements). Patching it collapses "
                "every self-match head."
            ),
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
                "residual": (
                    "schedule.layers[*].attention_operations[*] and "
                    "mlp_operations[*] target/source/query/key column runs"
                ),
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
