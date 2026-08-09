"""Typed, validated reader for the schematic.

``load_schematic`` answers questions about a manifest file alone;
``load_schematic_bundle`` binds it to the artifact directory it
describes.  Every view decodes the manifest's ordered run encoding at
the construction boundary — column tuples preserve component order and
must never be sorted — and every index is built lazily and cached.

Attribution rules (stated once, applied everywhere): a node's ``region``
is the innermost op call that *created* it; a node returned unchanged
through an outer decorated call appears in that outer region's
``results`` but keeps its creator's membership; composition ops
(``linear.subtract`` = add∘negate) create no node directly, so their
regions have members only through descendants.  ``region: null`` is
legal — the node was built outside any decorated op.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from torchwright.schematic.format import SCHEMATIC_FILENAME
from torchwright.schematic.support import SupportArchive, validate_support_archive
from torchwright.schematic.validate import (
    SchematicValidationError,
    expected_config_pointer,
    validate_bound_files,
    validate_manifest,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping


class Span(NamedTuple):
    """One ``[start, length]`` range — the manifest's single-run fields."""

    start: int
    length: int

    def __contains__(self, index: object) -> bool:
        return isinstance(index, int) and self.start <= index < self.start + self.length


_MATRIX_NDIM = 2


def _decoded(runs: list | None) -> tuple[int, ...] | None:
    """Decode a run list to an ordered column tuple (None passes through)."""
    if runs is None:
        return None
    cols: list[int] = []
    for start, length in runs:
        cols.extend(range(int(start), int(start) + int(length)))
    return tuple(cols)


def _span(run: list | None) -> Span | None:
    return None if run is None else Span(int(run[0]), int(run[1]))


@dataclass(frozen=True)
class SourceNodeView:
    """One ``graphs.source`` node; ``raw`` is the full record escape hatch."""

    id: str
    op: str
    width: int
    inputs: tuple[str, ...]
    name: str | None
    annotation: str | None
    region: str | None
    value_contract: Mapping[str, Any]
    semantics: Mapping[str, Any] | None
    raw: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class LoweredNodeView:
    """One ``graphs.lowered`` node with its contributing source nodes."""

    id: str
    op: str
    width: int
    inputs: tuple[str, ...]
    source_nodes: tuple[str, ...]
    raw: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class RegionView:
    """One semantic region: an op-library call and its dataflow."""

    id: str
    op: str
    parent: str | None
    params: Mapping[str, Any]
    operands: tuple[str, ...]
    results: tuple[str, ...]
    annotation: str | None


@dataclass(frozen=True)
class RealizationView:
    """How one source node realizes in the lowered graph."""

    source: str
    status: str  # "whole" | "slice" | "not_materialized"
    lowered: str | None
    slice: tuple[int, int] | None  # (offset, width) within the holder


@dataclass(frozen=True)
class PlacementRect:
    """One weight rectangle owned by a node (or a physical op) in a matrix."""

    matrix: str
    axis0: Span
    axis1: Span
    diagonal: bool

    def contains(self, row: int, col: int) -> bool:
        """Membership in logical matrix coordinates."""
        if row not in self.axis0 or col not in self.axis1:
            return False
        if not self.diagonal:
            return True
        return row - self.axis0.start == col - self.axis1.start


@dataclass(frozen=True)
class CheckpointSlice:
    """Where one logical matrix lives in the checkpoint.

    The checkpoint stores ``scale * transpose(logical)`` for 2-D records
    (``transform == "transpose"``); 1-D records (biases) store the
    logical vector at ``indices`` under ``transform == "identity"``.
    """

    logical_matrix: str | None
    checkpoint_tensor: str
    transform: str
    rows: Span | None
    columns: Span | None
    indices: Span | None
    scale: float


@dataclass(frozen=True)
class ResidualStateView:
    """One residual-stream snapshot: which node owns which columns."""

    key: str  # "input", "L{i}.post", or "output"
    index: int
    columns: Mapping[str, tuple[int, ...]]

    def nodes(self) -> tuple[str, ...]:
        """Node references materialized in this state."""
        return tuple(self.columns)

    def columns_for(self, node: str) -> tuple[int, ...]:
        """The node's residual columns, in component order."""
        try:
            return self.columns[node]
        except KeyError as err:
            raise KeyError(
                f"node {node!r} is not materialized in state {self.key!r}"
            ) from err

    @cached_property
    def _owner_by_column(self) -> dict[int, str]:
        return {
            column: node for node, columns in self.columns.items() for column in columns
        }

    def node_at(self, column: int) -> str | None:
        """The node owning ``column`` here, or None (free/reserved column)."""
        return self._owner_by_column.get(column)


@dataclass(frozen=True)
class OperationView:
    """One schedule operation (attention or MLP) with decoded coordinates."""

    id: str
    layer: int
    kind: str  # "attention" | "mlp"
    type: str
    role: str
    node: str | None
    heads: Span | None
    hidden_slots: tuple[int, ...] | None
    target_columns: tuple[int, ...] | None
    source_columns: tuple[int, ...] | None
    source_columns_b: tuple[int, ...] | None
    query_source_columns: tuple[int, ...] | None
    key_source_columns: tuple[int, ...] | None


def _operation_view(record: Mapping[str, Any], layer: int, kind: str) -> OperationView:
    return OperationView(
        id=record["id"],
        layer=layer,
        kind=kind,
        type=record["type"],
        role=record["role"],
        node=record.get("node"),
        heads=_span(record.get("heads")) if kind == "attention" else None,
        hidden_slots=_decoded(record.get("hidden_slots")) if kind == "mlp" else None,
        target_columns=_decoded(record.get("target_columns")),
        source_columns=_decoded(record.get("source_columns")),
        source_columns_b=_decoded(record.get("source_columns_b")),
        query_source_columns=_decoded(record.get("query_source_columns")),
        key_source_columns=_decoded(record.get("key_source_columns")),
    )


@dataclass(frozen=True)
class Schematic:
    """Query surface over a validated schematic payload.

    Read-only by construction: views are frozen, collections are tuples,
    and ``payload`` must not be mutated after loading (the loaders own
    the dict they just parsed; no defensive copy is taken).
    """

    payload: Mapping[str, Any]

    # -- meaning ---------------------------------------------------------

    @cached_property
    def _source_nodes(self) -> dict[str, SourceNodeView]:
        return {
            record["id"]: SourceNodeView(
                id=record["id"],
                op=record["op"],
                width=record["width"],
                inputs=tuple(record["inputs"]),
                name=record.get("name"),
                annotation=record.get("annotation"),
                region=record.get("region"),
                value_contract=record["value_contract"],
                semantics=record.get("semantics"),
                raw=record,
            )
            for record in self.payload["graphs"]["source"]["nodes"]
        }

    @cached_property
    def _lowered_nodes(self) -> dict[str, LoweredNodeView]:
        return {
            record["id"]: LoweredNodeView(
                id=record["id"],
                op=record["op"],
                width=record["width"],
                inputs=tuple(record["inputs"]),
                source_nodes=tuple(record.get("source_nodes", ())),
                raw=record,
            )
            for record in self.payload["graphs"]["lowered"]["nodes"]
        }

    @cached_property
    def _regions(self) -> dict[str, RegionView]:
        return {
            record["id"]: RegionView(
                id=record["id"],
                op=record["op"],
                parent=record.get("parent"),
                params=record.get("params", {}),
                operands=tuple(record.get("operands", ())),
                results=tuple(record.get("results", ())),
                annotation=record.get("annotation"),
            )
            for record in self.payload["graphs"]["source"]["semantic_regions"]
        }

    def source_node(self, node_id: str) -> SourceNodeView:
        """The source node ``node_id`` (``"s:<n>"``)."""
        return self._source_nodes[node_id]

    def source_nodes(self) -> tuple[SourceNodeView, ...]:
        """Every source node, in canonical order."""
        return tuple(self._source_nodes.values())

    def lowered_node(self, node_id: str) -> LoweredNodeView:
        """The lowered node ``node_id`` (``"l:<n>"``)."""
        return self._lowered_nodes[node_id]

    def region(self, region_id: str) -> RegionView:
        """The semantic region ``region_id`` (``"r:<n>"``)."""
        return self._regions[region_id]

    def regions(self) -> tuple[RegionView, ...]:
        """Every region, in manifest (outermost-first registration) order."""
        return tuple(self._regions.values())

    def region_chain(self, node_id: str) -> tuple[RegionView, ...]:
        """The node's creating-call chain, innermost first (empty if none)."""
        region_id = self.source_node(node_id).region
        chain: list[RegionView] = []
        while region_id is not None:
            view = self._regions[region_id]
            chain.append(view)
            region_id = view.parent
        return tuple(chain)

    @cached_property
    def _region_children(self) -> dict[str, tuple[str, ...]]:
        children: dict[str, list[str]] = {}
        for view in self._regions.values():
            if view.parent is not None:
                children.setdefault(view.parent, []).append(view.id)
        return {parent: tuple(ids) for parent, ids in children.items()}

    def region_children(self, region_id: str) -> tuple[RegionView, ...]:
        """Regions whose parent is ``region_id``."""
        self.region(region_id)  # raise KeyError on unknown ids
        return tuple(
            self._regions[child] for child in self._region_children.get(region_id, ())
        )

    def region_ancestors(self, region_id: str) -> tuple[RegionView, ...]:
        """The region's parent chain, nearest first."""
        parent = self.region(region_id).parent
        chain: list[RegionView] = []
        while parent is not None:
            view = self._regions[parent]
            chain.append(view)
            parent = view.parent
        return tuple(chain)

    @cached_property
    def _direct_members(self) -> dict[str, tuple[str, ...]]:
        members: dict[str, list[str]] = {}
        for node in self._source_nodes.values():
            if node.region is not None:
                members.setdefault(node.region, []).append(node.id)
        return {region: tuple(ids) for region, ids in members.items()}

    def region_members(
        self, region_id: str, *, direct: bool = False
    ) -> tuple[SourceNodeView, ...]:
        """Nodes belonging to the region.

        ``direct=True`` returns creator-only membership — empty for
        composition ops, which create no node themselves.  The default
        unions the region's whole subtree, which is how "the nodes of
        this op call" reads for compositions.
        """
        self.region(region_id)
        region_ids = [region_id]
        if not direct:
            frontier = [region_id]
            while frontier:
                frontier = [
                    child
                    for parent in frontier
                    for child in self._region_children.get(parent, ())
                ]
                region_ids.extend(frontier)
        return tuple(
            self._source_nodes[node_id]
            for rid in region_ids
            for node_id in self._direct_members.get(rid, ())
        )

    # -- physical location ----------------------------------------------

    @cached_property
    def _states(self) -> dict[str, ResidualStateView]:
        states = {}
        for index, record in enumerate(self.payload["residual_stream"]["states"]):
            columns = {
                node: _decoded(runs) or ()
                for node, runs in record.get("nodes", {}).items()
            }
            states[record["key"]] = ResidualStateView(
                key=record["key"], index=index, columns=columns
            )
        return states

    def residual_states(self) -> tuple[ResidualStateView, ...]:
        """Every stream snapshot: input, each ``L{i}.post``, output."""
        return tuple(self._states.values())

    def residual_state(self, key: str) -> ResidualStateView:
        """One stream snapshot by key."""
        return self._states[key]

    def column_owner(self, state_key: str, column: int) -> str | None:
        """The node owning ``column`` in the named state, or None."""
        return self.residual_state(state_key).node_at(column)

    @cached_property
    def _placements(self) -> dict[str, tuple[PlacementRect, ...]]:
        return {
            owner: tuple(
                PlacementRect(
                    matrix=rect["matrix"],
                    axis0=Span(*rect["axis0"]),
                    axis1=Span(*rect["axis1"]),
                    diagonal=bool(rect.get("diagonal", False)),
                )
                for rect in rects
            )
            for owner, rects in self.payload["physical_layout"]["placements"].items()
        }

    def placements(self, owner: str) -> tuple[PlacementRect, ...]:
        """Weight rectangles for a node ref or ``"physical:<op_type>"``."""
        return self._placements[owner]

    def placement_owners(self) -> tuple[str, ...]:
        """Every placement owner in the manifest."""
        return tuple(self._placements)

    def matrix_shape(self, matrix: str) -> tuple[int, int]:
        """Logical shape of ``"L{i}.<kind>"`` in compiler coordinates."""
        shape = self.payload["physical_layout"]["matrices"][matrix]["shape"]
        return (int(shape[0]), int(shape[1]))

    @cached_property
    def _checkpoint_slices(self) -> dict[str, CheckpointSlice]:
        slices = {}
        for record in self.payload["physical_layout"]["checkpoint_parameter_map"]:
            logical = record.get("logical_matrix")
            if logical is None:
                continue
            slices[logical] = CheckpointSlice(
                logical_matrix=logical,
                checkpoint_tensor=record["checkpoint_tensor"],
                transform=record["transform"],
                rows=_span(record.get("checkpoint_rows")),
                columns=_span(record.get("checkpoint_columns")),
                indices=_span(record.get("checkpoint_indices")),
                scale=float(record.get("scale", 1.0)),
            )
        return slices

    def checkpoint_slice(self, logical_matrix: str) -> CheckpointSlice:
        """Where the logical matrix lives in the checkpoint tensors."""
        return self._checkpoint_slices[logical_matrix]

    def to_checkpoint(
        self, matrix: str, row: int, col: int
    ) -> tuple[str, tuple[int, ...]]:
        """Translate a logical matrix cell to its checkpoint coordinate.

        The stored value is ``scale * logical[row, col]`` under the
        slice's transpose, so the returned coordinate swaps the axes into
        the checkpoint region.
        """
        cp = self.checkpoint_slice(matrix)
        if cp.rows is None or cp.columns is None:
            raise ValueError(f"{matrix} has no 2-D checkpoint mapping")
        shape = self.matrix_shape(matrix)
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            raise ValueError(f"({row}, {col}) outside {matrix} shape {shape}")
        return cp.checkpoint_tensor, (cp.rows.start + col, cp.columns.start + row)

    @cached_property
    def _slices_by_tensor(self) -> dict[str, tuple[CheckpointSlice, ...]]:
        by_tensor: dict[str, list[CheckpointSlice]] = {}
        for cp in self._checkpoint_slices.values():
            by_tensor.setdefault(cp.checkpoint_tensor, []).append(cp)
        return {tensor: tuple(cps) for tensor, cps in by_tensor.items()}

    @cached_property
    def _rects_by_matrix(self) -> dict[str, tuple[tuple[str, PlacementRect], ...]]:
        by_matrix: dict[str, list[tuple[str, PlacementRect]]] = {}
        for owner, rects in self._placements.items():
            for rect in rects:
                by_matrix.setdefault(rect.matrix, []).append((owner, rect))
        return {matrix: tuple(entries) for matrix, entries in by_matrix.items()}

    def checkpoint_owner(
        self, tensor: str, coord: tuple[int, ...]
    ) -> tuple[str, PlacementRect] | None:
        """Invert a 2-D checkpoint coordinate to its owning placement.

        Returns ``(owner, rect)`` — owner is a node ref or
        ``"physical:<op_type>"`` — or None when no logical matrix or
        placement covers the coordinate.  1-D (bias) tensors have no
        placement rectangles and always return None.
        """
        if len(coord) != _MATRIX_NDIM:
            return None
        for cp in self._slices_by_tensor.get(tensor, ()):
            if cp.rows is None or cp.columns is None:
                continue
            if coord[0] not in cp.rows or coord[1] not in cp.columns:
                continue
            row = coord[1] - cp.columns.start
            col = coord[0] - cp.rows.start
            assert cp.logical_matrix is not None
            for owner, rect in self._rects_by_matrix.get(cp.logical_matrix, ()):
                if rect.contains(row, col):
                    return owner, rect
        return None

    # -- realization and schedule ---------------------------------------

    @cached_property
    def _realizations(self) -> dict[str, RealizationView]:
        views = {}
        for record in self.payload["graphs"]["realization_map"]:
            sliced = record.get("slice")
            views[record["source"]] = RealizationView(
                source=record["source"],
                status=record["status"],
                lowered=record.get("lowered"),
                slice=None
                if sliced is None
                else (int(sliced["offset"]), int(sliced["width"])),
            )
        return views

    def realization(self, source_id: str) -> RealizationView:
        """How the source node realizes: whole, slice, or not at all."""
        return self._realizations[source_id]

    @cached_property
    def _assignments(self) -> dict[str, Mapping[str, Any]]:
        return {
            record["node"]: record for record in self.payload["schedule"]["assignment"]
        }

    def assignment(self, node: str) -> Mapping[str, Any] | None:
        """The schedule assignment record for a physical node, if any."""
        return self._assignments.get(node)

    @cached_property
    def _realization_candidates(self) -> dict[str, Mapping[str, Any]]:
        return {
            record["node"]: record
            for record in self.payload["schedule"]["realizations"]
        }

    def realization_candidates(self, node: str) -> Mapping[str, Any] | None:
        """The realization-choice record for a physical node, if any."""
        return self._realization_candidates.get(node)

    @cached_property
    def _operations(self) -> dict[str, tuple[OperationView, ...]]:
        by_node: dict[str, list[OperationView]] = {}
        for layer in self.payload["schedule"]["layers"]:
            index = layer["index"]
            for kind, section in (
                ("attention", "attention_operations"),
                ("mlp", "mlp_operations"),
            ):
                for record in layer[section]:
                    if record.get("node") is None:
                        continue
                    view = _operation_view(record, index, kind)
                    by_node.setdefault(record["node"], []).append(view)
        return {node: tuple(views) for node, views in by_node.items()}

    def operations_for(self, node: str) -> tuple[OperationView, ...]:
        """Every schedule operation attributed to a physical node."""
        return self._operations.get(node, ())

    # -- token I/O -------------------------------------------------------

    def embedding_columns(self) -> tuple[int, ...]:
        """The embedding's residual columns in the input state."""
        return _decoded(self.payload["token_io"]["embedding_residual_columns"]) or ()

    def output_columns(self) -> tuple[int, ...]:
        """The output node's residual columns in the output state."""
        return _decoded(self.payload["token_io"]["output_residual_columns"]) or ()


@dataclass(frozen=True)
class SchematicBundle:
    """A schematic bound to the bundle directory it describes."""

    directory: Path
    manifest: Schematic

    def file_record(self, name: str) -> Mapping[str, Any]:
        """The ``artifact.files`` entry (sha256, bytes) for one file."""
        return self.manifest.payload["artifact"]["files"][name]

    def verify_files(self) -> None:
        """Re-hash every bound file against the manifest (multi-GB cost)."""
        validate_bound_files(self.directory, self.manifest.payload["artifact"]["files"])

    @cached_property
    def _support(self) -> SupportArchive:
        return SupportArchive.load(
            self.directory, self.manifest.payload["parameter_support"]
        )

    def support(self) -> SupportArchive:
        """The decoded nonzero-coordinate archive (loads the npz once)."""
        return self._support

    def support_coordinates(self, tensor: str) -> set[tuple[int, ...]]:
        """Nonzero coordinates for one checkpoint tensor."""
        return self._support.coordinates(tensor)


def load_schematic(path: str | os.PathLike[str]) -> Schematic:
    """Load and validate a schematic from a file (or its directory).

    Always runs the manifest-internal checks — format string, section
    inventory, integrity hash, per-graph content hashes, reference
    resolution — and raises :class:`SchematicValidationError` on any
    failure.  No artifact files are touched: this is the
    downloaded-one-file use case.
    """
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / SCHEMATIC_FILENAME
    if not manifest_path.is_file():
        raise SchematicValidationError(f"no schematic at {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(payload)
    return Schematic(payload=payload)


def load_schematic_bundle(
    path: str | os.PathLike[str], *, verify_files: bool = False
) -> SchematicBundle:
    """Load a schematic bound to its bundle directory.

    Everything :func:`load_schematic` checks, plus: ``config.json``
    carries the exact schematic pointer, every bound file exists with
    the recorded byte size, and the support npz is structurally valid.
    ``verify_files=True`` additionally re-hashes every bound file —
    multi-GB shards included — which is also available later via
    :meth:`SchematicBundle.verify_files`.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise SchematicValidationError(f"{directory} is not a bundle directory")
    manifest = load_schematic(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("torchwright_schematic") != expected_config_pointer():
        raise SchematicValidationError(
            "config.json schematic pointer does not match the bundle"
        )
    validate_bound_files(
        directory,
        manifest.payload["artifact"]["files"],
        sizes_only=not verify_files,
    )
    validate_support_archive(directory, manifest.payload["parameter_support"])
    return SchematicBundle(directory=directory, manifest=manifest)
