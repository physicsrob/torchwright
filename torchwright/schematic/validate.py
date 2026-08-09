"""Schematic validation — the checks both the builder and every reader run.

Moved out of the HF staging path so consumers can refuse exactly what
the builder refuses without importing torch.  ``validate_manifest`` is
the manifest-internal core (format, section inventory, integrity hash,
per-graph content hashes, reference resolution); ``validate_bound_files``
is the artifact-file binding.  Everything raises
:class:`SchematicValidationError` with the message naming the failing
fact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torchwright.schematic.format import (
    SCHEMATIC_FILENAME,
    SCHEMATIC_FORMAT,
    SCHEMATIC_SCHEMA_FILENAME,
    SCHEMATIC_SCHEMA_SOURCE,
    SCHEMATIC_SUPPORT_FILENAME,
    sha256_file,
    sha256_json,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_RUN_FIELDS = 2


class SchematicValidationError(Exception):
    """A schematic (or its bundle binding) failed a validation check."""


def expected_config_pointer() -> dict[str, str]:
    """The exact ``config.json`` pointer a schematic-carrying bundle declares."""
    return {
        "format": SCHEMATIC_FORMAT,
        "file": SCHEMATIC_FILENAME,
        "schema": SCHEMATIC_SCHEMA_FILENAME,
        "support": SCHEMATIC_SUPPORT_FILENAME,
    }


def validate_manifest(payload: dict[str, Any]) -> None:
    """Run every manifest-internal check on a parsed schematic payload.

    Format string, section inventory (against the packaged schema's
    ``required`` list), whole-manifest integrity hash, per-graph content
    hashes, and internal reference resolution.  File-binding checks are
    separate (:func:`validate_bound_files`) — a bare manifest file can be
    validated without the bundle it describes.
    """
    if payload.get("format") != SCHEMATIC_FORMAT:
        raise SchematicValidationError("schematic has an unsupported format")
    # The packaged schema's required[] list is the one section inventory;
    # enforcing it here keeps the schema shipped in every bundle honest.
    schema = json.loads(SCHEMATIC_SCHEMA_SOURCE.read_text(encoding="utf-8"))
    missing = sorted(set(schema["required"]) - set(payload))
    if missing:
        raise SchematicValidationError(f"schematic is missing sections: {missing}")
    integrity = payload.get("integrity")
    expected_digest = sha256_json(
        {key: value for key, value in payload.items() if key != "integrity"}
    )
    if not isinstance(integrity, dict) or integrity.get("sha256") != expected_digest:
        raise SchematicValidationError("schematic integrity hash mismatch")
    files = payload["artifact"].get("files")
    if not isinstance(files, dict) or not files:
        raise SchematicValidationError("schematic has no artifact file hashes")
    _validate_internals(payload)


def validate_bound_files(
    directory: Path,
    files: dict[str, dict[str, Any]],
    *,
    precomputed_hashes: Mapping[str, dict[str, Any]] | None = None,
    sizes_only: bool = False,
) -> None:
    """Check every schematic-bound artifact file against its manifest record.

    ``precomputed_hashes`` is the hash map the builder computed moments
    earlier in the same process; when a file's entry is present there,
    its digest substitutes for re-reading multi-GB shards.  External
    validation passes nothing and re-hashes from disk — or passes
    ``sizes_only=True`` for the cheap presence-and-size tier.
    """
    for name, record in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise SchematicValidationError(
                f"schematic has unsafe artifact path {name!r}"
            )
        path = directory / relative
        if not path.is_file():
            raise SchematicValidationError(
                f"schematic-bound artifact is missing: {name}"
            )
        if path.stat().st_size != record.get("bytes"):
            raise SchematicValidationError(
                f"schematic-bound artifact size mismatch: {name}"
            )
        if sizes_only:
            continue
        if precomputed_hashes is not None and name in precomputed_hashes:
            actual = precomputed_hashes[name]["sha256"]
        else:
            actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise SchematicValidationError(
                f"schematic-bound artifact hash mismatch: {name}"
            )


def _validate_runs(
    value: object, limit: int, label: str, *, min_length: int = 1
) -> None:
    if not isinstance(value, list):
        raise SchematicValidationError(
            f"{label} must be a list of [start, length] runs"
        )
    for run in value:
        if (
            not isinstance(run, list)
            or len(run) != _RUN_FIELDS
            or not all(isinstance(part, int) for part in run)
        ):
            raise SchematicValidationError(f"{label} has a malformed run")
        start, length = run
        if start < 0 or length < min_length or start + length > limit:
            raise SchematicValidationError(f"{label} run {run} exceeds width {limit}")


def _validate_graph_record(graph: dict[str, Any], prefix: str) -> dict[str, int]:
    label = "source" if prefix == "s" else "lowered"
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise SchematicValidationError(f"{label} graph nodes must be a list")
    widths: dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise SchematicValidationError(f"{label} graph node must be an object")
        node_id, width = node.get("id"), node.get("width")
        if not isinstance(node_id, str) or not isinstance(width, int):
            raise SchematicValidationError(
                f"{label} graph node has an invalid ID or width"
            )
        widths[node_id] = width
    if len(widths) != len(nodes) or any(
        not node.startswith(f"{prefix}:") for node in widths
    ):
        raise SchematicValidationError(
            f"{label} graph has duplicate or invalid node IDs"
        )
    if graph.get("root") not in widths:
        raise SchematicValidationError(f"{label} graph root does not resolve")
    hashed = {key: value for key, value in graph.items() if key != "sha256"}
    if graph.get("sha256") != sha256_json(hashed):
        raise SchematicValidationError(f"{label} graph hash mismatch")
    for node in nodes:
        if any(input_id not in widths for input_id in node.get("inputs", [])):
            raise SchematicValidationError(
                f"{label} graph has an unresolved input reference"
            )
    return widths


def _validate_semantic_regions(
    graph: dict[str, Any], source_widths: dict[str, int]
) -> None:
    regions = graph.get("semantic_regions")
    if not isinstance(regions, list):
        raise SchematicValidationError("source graph semantic regions must be a list")
    region_ids = {region.get("id") for region in regions}
    if len(region_ids) != len(regions) or any(
        not isinstance(region_id, str) or not region_id.startswith("r:")
        for region_id in region_ids
    ):
        raise SchematicValidationError(
            "source graph has duplicate or invalid region IDs"
        )
    for region in regions:
        parent = region.get("parent")
        if parent is not None and parent not in region_ids:
            raise SchematicValidationError("semantic region parent does not resolve")
        for field in ("operands", "results"):
            if any(ref not in source_widths for ref in region.get(field, [])):
                raise SchematicValidationError(
                    f"semantic region has an unresolved {field} node"
                )
    for node in graph["nodes"]:
        region = node.get("region")
        if region is not None and region not in region_ids:
            raise SchematicValidationError(
                "source node names an unknown semantic region"
            )


def _validate_lowering_map(
    records: object, source_widths: dict[str, int], lowered_widths: dict[str, int]
) -> None:
    if not isinstance(records, list) or len(records) != len(source_widths):
        raise SchematicValidationError(
            "schematic lowering map does not cover the source graph"
        )
    if {record.get("source") for record in records} != set(source_widths):
        raise SchematicValidationError(
            "schematic lowering map has invalid source references"
        )
    for record in records:
        status = record.get("status")
        if status == "not_materialized":
            continue
        lowered_id = record.get("lowered")
        if lowered_id not in lowered_widths:
            raise SchematicValidationError(
                "schematic lowering map has an invalid lowered reference"
            )
        if status == "whole":
            continue
        sliced = record.get("slice")
        if status != "slice" or not isinstance(sliced, dict):
            raise SchematicValidationError(
                "schematic lowering map has an invalid status"
            )
        offset, width = sliced.get("offset"), sliced.get("width")
        if (
            not isinstance(offset, int)
            or not isinstance(width, int)
            or offset < 0
            or width < 1
            or offset + width > lowered_widths[lowered_id]
        ):
            raise SchematicValidationError(
                "schematic lowering slice exceeds its holder"
            )


def _validate_attention_operations(
    layer: dict[str, Any], physical_ids: set[str], d_model: int
) -> None:
    for operation in layer.get("attention_operations", []):
        if operation.get("node") not in physical_ids | {None}:
            raise SchematicValidationError("attention operation has an unresolved node")
        # A zero-length span is legitimate here: a zero-support
        # attention-routed op emits no post-trim heads (its writer-allocated
        # floor head is trimmed), so its span is [cursor, 0].
        _validate_runs(
            [operation.get("heads")],
            layer["active_attention_heads"],
            "attention heads",
            min_length=0,
        )
        for field in (
            "target_columns",
            "source_columns",
            "source_columns_b",
            "query_source_columns",
            "key_source_columns",
        ):
            if operation.get(field) is not None:
                _validate_runs(operation[field], d_model, field)


def _validate_mlp_operations(
    layer: dict[str, Any], physical_ids: set[str], d_model: int
) -> None:
    for operation in layer.get("mlp_operations", []):
        if operation.get("node") not in physical_ids | {None}:
            raise SchematicValidationError("MLP operation has an unresolved node")
        _validate_runs(operation.get("target_columns"), d_model, "MLP target")
        _validate_runs(
            operation.get("hidden_slots"),
            layer["active_mlp_neurons"],
            "hidden slots",
        )
        for field in ("source_columns", "source_columns_b"):
            if operation.get(field) is not None:
                _validate_runs(operation[field], d_model, field)


def _validate_schedule_references(
    payload: dict[str, Any], physical_ids: set[str]
) -> None:
    model = payload["model"]
    d_model = model["d_model"]
    layers = payload["schedule"].get("layers", [])
    if len(layers) != model["n_layers"]:
        raise SchematicValidationError(
            "schematic schedule layer count disagrees with the model"
        )
    for index, layer in enumerate(layers):
        if layer.get("index") != index:
            raise SchematicValidationError(
                "schematic schedule layers are not in canonical order"
            )
        _validate_attention_operations(layer, physical_ids, d_model)
        _validate_mlp_operations(layer, physical_ids, d_model)
    for state in payload["residual_stream"].get("states", []):
        for node_id, runs in state.get("nodes", {}).items():
            if node_id not in physical_ids:
                raise SchematicValidationError("residual state has an unresolved node")
            _validate_runs(runs, d_model, "residual columns")


def _validate_physical_layout(payload: dict[str, Any]) -> None:
    layout = payload["physical_layout"]
    matrices = layout.get("matrices", {})
    for owner, rectangles in layout.get("placements", {}).items():
        for rectangle in rectangles:
            matrix = rectangle.get("matrix")
            if matrix not in matrices:
                raise SchematicValidationError(
                    f"placement {owner!r} names an unknown matrix"
                )
            shape = matrices[matrix]["shape"]
            _validate_runs([rectangle.get("axis0")], shape[0], "matrix axis0")
            _validate_runs([rectangle.get("axis1")], shape[1], "matrix axis1")

    tensors = payload["parameter_support"].get("tensors", {})
    for record in layout.get("checkpoint_parameter_map", []):
        tensor = record.get("checkpoint_tensor")
        if tensor not in tensors:
            raise SchematicValidationError(
                "checkpoint parameter map names an unknown tensor"
            )
        shape = tensors[tensor]["shape"]
        if "checkpoint_rows" in record:
            _validate_runs([record["checkpoint_rows"]], shape[0], "checkpoint rows")
            _validate_runs(
                [record["checkpoint_columns"]], shape[1], "checkpoint columns"
            )
        elif "checkpoint_indices" in record:
            _validate_runs(
                [record["checkpoint_indices"]], shape[0], "checkpoint indices"
            )


def _validate_internals(payload: dict[str, Any]) -> None:
    graphs = payload["graphs"]
    source_widths = _validate_graph_record(graphs["source"], "s")
    _validate_semantic_regions(graphs["source"], source_widths)
    lowered_widths = _validate_graph_record(graphs["lowered"], "l")
    _validate_lowering_map(graphs.get("realization_map"), source_widths, lowered_widths)
    internal = graphs.get("internal_nodes", [])
    internal_ids = {node.get("id") for node in internal}
    if len(internal_ids) != len(internal) or any(
        not isinstance(node, str) or not node.startswith("i:") for node in internal_ids
    ):
        raise SchematicValidationError("schematic has invalid internal node IDs")
    physical_ids = set(lowered_widths) | internal_ids
    schedule = payload["schedule"]
    for section in ("assignment", "realizations"):
        if any(record.get("node") not in physical_ids for record in schedule[section]):
            raise SchematicValidationError(
                f"schematic schedule {section} has an unresolved node"
            )
    _validate_schedule_references(payload, physical_ids)
    _validate_physical_layout(payload)
    d_model = payload["model"]["d_model"]
    _validate_runs(
        payload["token_io"]["embedding_residual_columns"],
        d_model,
        "embedding columns",
    )
    _validate_runs(
        payload["token_io"]["output_residual_columns"], d_model, "output columns"
    )
