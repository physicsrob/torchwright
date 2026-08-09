"""Moved validator core on hand-built payloads — no compile, no torch."""

import pytest

from torchwright.schematic.format import sha256_json
from torchwright.schematic.validate import (
    SchematicValidationError,
    _validate_graph_record,
    _validate_lowering_map,
    _validate_runs,
    _validate_semantic_regions,
    expected_config_pointer,
)


def _graph(nodes, root="s:0"):
    graph = {"root": root, "nodes": nodes}
    graph["sha256"] = sha256_json(graph)
    return graph


def _node(node_id, width=1, inputs=()):
    return {"id": node_id, "width": width, "inputs": list(inputs)}


# ---------------------------------------------------------------------------
# _validate_runs
# ---------------------------------------------------------------------------


def test_runs_accepts_zero_length_only_when_allowed():
    _validate_runs([[3, 0]], 8, "heads", min_length=0)
    with pytest.raises(SchematicValidationError, match="exceeds width"):
        _validate_runs([[3, 0]], 8, "columns")


def test_runs_rejects_out_of_bounds_and_malformed():
    with pytest.raises(SchematicValidationError, match="exceeds width"):
        _validate_runs([[6, 3]], 8, "columns")
    with pytest.raises(SchematicValidationError, match="malformed run"):
        _validate_runs([[1, 2, 3]], 8, "columns")
    with pytest.raises(SchematicValidationError, match="must be a list"):
        _validate_runs(None, 8, "columns")


# ---------------------------------------------------------------------------
# _validate_graph_record
# ---------------------------------------------------------------------------


def test_graph_record_roundtrips_and_returns_widths():
    graph = _graph([_node("s:0", width=2, inputs=["s:1"]), _node("s:1")])
    assert _validate_graph_record(graph, "s") == {"s:0": 2, "s:1": 1}


def test_graph_record_rejections():
    duplicate = _graph([_node("s:0"), _node("s:0")])
    with pytest.raises(SchematicValidationError, match="duplicate or invalid"):
        _validate_graph_record(duplicate, "s")

    wrong_prefix = _graph([_node("l:0")])
    with pytest.raises(SchematicValidationError, match="duplicate or invalid"):
        _validate_graph_record(wrong_prefix, "s")

    dangling_root = _graph([_node("s:0")], root="s:9")
    with pytest.raises(SchematicValidationError, match="root does not resolve"):
        _validate_graph_record(dangling_root, "s")

    unresolved_input = _graph([_node("s:0", inputs=["s:9"])])
    with pytest.raises(SchematicValidationError, match="unresolved input"):
        _validate_graph_record(unresolved_input, "s")

    tampered = _graph([_node("s:0")])
    tampered["nodes"][0]["width"] = 5
    with pytest.raises(SchematicValidationError, match="graph hash mismatch"):
        _validate_graph_record(tampered, "s")


# ---------------------------------------------------------------------------
# _validate_semantic_regions
# ---------------------------------------------------------------------------


def _region_graph(regions, node_region="r:0"):
    return {
        "nodes": [{"id": "s:0", "region": node_region}],
        "semantic_regions": regions,
    }


def _region(region_id, parent=None, operands=(), results=()):
    return {
        "id": region_id,
        "parent": parent,
        "operands": list(operands),
        "results": list(results),
    }


def test_semantic_regions_accepts_resolving_table():
    graph = _region_graph([_region("r:0", operands=["s:0"], results=["s:0"])])
    _validate_semantic_regions(graph, {"s:0": 1})


def test_semantic_regions_rejections():
    with pytest.raises(SchematicValidationError, match="parent does not resolve"):
        _validate_semantic_regions(
            _region_graph([_region("r:0", parent="r:9")]), {"s:0": 1}
        )
    with pytest.raises(SchematicValidationError, match="unresolved operands node"):
        _validate_semantic_regions(
            _region_graph([_region("r:0", operands=["s:9"])]), {"s:0": 1}
        )
    with pytest.raises(SchematicValidationError, match="unknown semantic region"):
        _validate_semantic_regions(
            _region_graph([_region("r:0")], node_region="r:7"), {"s:0": 1}
        )
    with pytest.raises(SchematicValidationError, match="duplicate or invalid region"):
        _validate_semantic_regions(
            _region_graph([_region("r:0"), _region("r:0")]), {"s:0": 1}
        )


# ---------------------------------------------------------------------------
# _validate_lowering_map
# ---------------------------------------------------------------------------


def test_lowering_map_covers_and_bounds():
    source = {"s:0": 2, "s:1": 3}
    lowered = {"l:0": 5}
    _validate_lowering_map(
        [
            {"source": "s:0", "status": "whole", "lowered": "l:0"},
            {
                "source": "s:1",
                "status": "slice",
                "lowered": "l:0",
                "slice": {"offset": 2, "width": 3},
            },
        ],
        source,
        lowered,
    )
    with pytest.raises(SchematicValidationError, match="does not cover"):
        _validate_lowering_map(
            [{"source": "s:0", "status": "not_materialized"}], source, lowered
        )
    with pytest.raises(SchematicValidationError, match="exceeds its holder"):
        _validate_lowering_map(
            [
                {"source": "s:0", "status": "not_materialized"},
                {
                    "source": "s:1",
                    "status": "slice",
                    "lowered": "l:0",
                    "slice": {"offset": 3, "width": 3},
                },
            ],
            source,
            lowered,
        )


def test_expected_config_pointer_names_the_schematic_files():
    pointer = expected_config_pointer()
    assert pointer["file"] == "torchwright_schematic.json"
    assert pointer["format"] == "torchwright.schematic.v1"
