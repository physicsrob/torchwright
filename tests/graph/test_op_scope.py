"""``op_scope`` — construction-time op-call provenance records.

Pattern follows ``tests/graph/test_annotate.py``: exercise the ContextVar
machinery directly, then the capture-side translation in
``torchwright.compiler.truth._stamp_semantic_regions``.
"""

import json

import pytest
import torch

from torchwright.compiler.truth import _graph_record, _stamp_semantic_regions
from torchwright.graph import Node, RopeConfig, annotate, op_scope
from torchwright.graph.misc import InputNode
from torchwright.graph.node import _current_op_scope, _sanitize_op_params
from torchwright.ops.linear import add, negate, subtract


def _inputs(n=2, width=1):
    return [
        InputNode(width, name=f"in{i}", value_range=(-10.0, 10.0)) for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Stamping and record contents
# ---------------------------------------------------------------------------


def test_stamps_record_on_created_nodes():
    a, b = _inputs()
    out = add(a, b)
    record = out.op_region
    assert record is not None
    assert record.op == "linear.add"
    assert record.parent is None
    assert record.operand_ids == (a.node_id, b.node_id)
    assert record.result_ids == (out.node_id,)
    assert record.params == {}
    # Direct InputNode construction happened outside any decorated call.
    assert a.op_region is None


def test_composition_nests_and_stamps_zero_nodes_directly():
    a, b = _inputs()
    out = subtract(a, b)
    # subtract = add ∘ negate: the Add node's membership is the innermost
    # creator; subtract's own record is reachable only as a parent.
    assert out.op_region.op == "linear.add"
    outer = out.op_region.parent
    assert outer.op == "linear.subtract"
    assert outer.parent is None
    assert outer.operand_ids == (a.node_id, b.node_id)
    assert outer.result_ids == (out.node_id,)
    neg = out.inputs[1]
    assert neg.op_region.op == "linear.negate"
    assert neg.op_region.parent is outer
    # subtract created no node of its own: nothing has it as membership.
    assert all(node.op_region is not outer for node in (out, neg, a, b))


def test_every_call_is_a_new_record():
    a, b = _inputs()
    first = add(a, b).op_region
    second = add(a, b).op_region
    assert first is not second


# ---------------------------------------------------------------------------
# Operand / param / result collection
# ---------------------------------------------------------------------------


@op_scope
def _combine(nodes: list[Node], gain: float = 2.0) -> Node:
    return add(nodes[0], nodes[1])


@op_scope
def _pair(first: Node, *, other: Node) -> tuple[Node, Node]:
    return first, negate(other)


class _Builder:
    @op_scope
    def __init__(self, seed: Node, gain: float) -> None:
        self.gain = gain
        self.node = negate(seed)


def test_list_operands_and_default_params():
    a, b = _inputs()
    record = _combine([a, b]).op_region.parent
    assert record.op.endswith("._combine")
    assert record.operand_ids == (a.node_id, b.node_id)
    assert record.params == {"gain": 2.0}


def test_keyword_operands_and_tuple_results():
    a, b = _inputs()
    first, neg = _pair(a, other=b)
    assert first is a
    record = neg.op_region.parent
    assert record.operand_ids == (a.node_id, b.node_id)
    assert record.result_ids == (a.node_id, neg.node_id)
    assert record.params == {}


def test_method_skips_self_and_none_result():
    (seed,) = _inputs(1)
    builder = _Builder(seed, 3.0)
    record = builder.node.op_region.parent
    assert record.op.endswith("._Builder.__init__")
    assert record.operand_ids == (seed.node_id,)
    assert record.params == {"gain": 3.0}
    assert record.result_ids == ()


def test_passthrough_keeps_creator_membership():
    a, b = _inputs()
    node = add(a, b)
    returned, neg = _pair(node, other=b)
    assert returned is node
    # The pass-through appears in the outer record's results...
    assert node.node_id in neg.op_region.parent.result_ids
    # ...but its membership is still its creator's.
    assert node.op_region.op == "linear.add"
    assert node.op_region.parent is None


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------


def test_sanitizer_encodings():
    cases = {
        "scalars": _sanitize_op_params([True, 3, "x", None, 1.5]),
        "nan": _sanitize_op_params(float("nan")),
        "pos_inf": _sanitize_op_params(float("inf")),
        "neg_inf": _sanitize_op_params(float("-inf")),
        "tensor": _sanitize_op_params(torch.zeros(3, 4)),
        "rope": _sanitize_op_params(RopeConfig(d_head=64, max_positions=512)),
        "tensor_keyed": _sanitize_op_params({torch.zeros(1): 1}),
        "oversize": _sanitize_op_params(list(range(100))),
        "unknown": _sanitize_op_params(object()),
    }
    assert cases["scalars"] == [True, 3, "x", None, 1.5]
    assert cases["nan"] == "nan"
    assert cases["pos_inf"] == "+inf"
    assert cases["neg_inf"] == "-inf"
    assert cases["tensor"] == {"tensor": {"shape": [3, 4], "dtype": "float32"}}
    rope = RopeConfig(d_head=64, max_positions=512)
    assert cases["rope"] == {
        "d_head": 64,
        "max_positions": 512,
        "base": rope.base,
        "d_rot": rope.d_rot,  # __post_init__ resolves None to d_head
    }
    assert cases["tensor_keyed"] == "<dict[1]>"
    assert cases["oversize"] == "<list[100]>"
    assert cases["unknown"] == "<object>"
    # Totality contract: every encoding must survive the manifest dump.
    json.dumps(cases, allow_nan=False)


def test_sanitizer_depth_cap():
    nested = [[[["deep"]]]]
    assert _sanitize_op_params(nested) == [[["<list[1]>"]]]


# ---------------------------------------------------------------------------
# ContextVar discipline and annotate independence
# ---------------------------------------------------------------------------


@op_scope
def _boom(node: Node) -> Node:
    raise ValueError("boom")


def test_scope_reset_on_exception():
    (a,) = _inputs(1)
    with pytest.raises(ValueError, match="boom"):
        _boom(a)
    assert _current_op_scope.get() is None
    assert add(a, a).op_region.parent is None


def test_annotation_snapshot_and_independence():
    a, b = _inputs()
    with annotate("subsystem"):
        out = add(a, b)
    # The record snapshots the ambient path; the node's own annotation
    # channel is untouched by op_scope.
    assert out.op_region.annotation == "subsystem"
    assert out.annotation == "subsystem"
    assert add(a, b).op_region.annotation is None


# ---------------------------------------------------------------------------
# Capture-side region stamping
# ---------------------------------------------------------------------------


def _capture(output):
    graph, refs = _graph_record(output, "s", {}, stamp_hash=False)
    return _stamp_semantic_regions(output, graph["nodes"], refs), graph


def test_capture_regions_deterministic_across_rebuilds():
    def build():
        a, b = _inputs()
        return subtract(a, b)

    regions_one, graph_one = _capture(build())
    regions_two, graph_two = _capture(build())
    assert regions_one == regions_two
    assert [record["op"] for record in regions_one] == [
        "linear.subtract",
        "linear.add",
        "linear.negate",
    ]
    # Ancestors register before children, so parents have lower ids.
    assert [record["parent"] for record in regions_one] == [None, "r:0", "r:0"]
    assert [node["region"] for node in graph_one["nodes"]] == [
        node["region"] for node in graph_two["nodes"]
    ]


def test_capture_membership_and_dataflow_refs():
    a, b = _inputs()
    out = subtract(a, b)
    regions, graph = _capture(out)
    by_id = {record["id"]: record for record in regions}
    node_regions = {node["id"]: node["region"] for node in graph["nodes"]}
    # Canonical order: Add root (s:0), a (s:1), negate Linear (s:2), b (s:3).
    assert node_regions == {"s:0": "r:1", "s:1": None, "s:2": "r:2", "s:3": None}
    assert by_id["r:0"]["operands"] == ["s:1", "s:3"]
    assert by_id["r:0"]["results"] == ["s:0"]
    assert by_id["r:2"]["operands"] == ["s:3"]


@op_scope
def _pick_first(nodes: list[Node]) -> Node:
    return negate(nodes[0])


def test_capture_drops_unreachable_operand_refs():
    a, b = _inputs()
    out = _pick_first([a, b])  # b never reaches the compiled output
    regions, graph = _capture(out)
    picker = regions[0]
    assert picker["op"].endswith("._pick_first")
    assert picker["operands"] == ["s:1"]  # a only; b's id dropped
    assert all(
        ref in {node["id"] for node in graph["nodes"]}
        for record in regions
        for field in ("operands", "results")
        for ref in record[field]
    )
