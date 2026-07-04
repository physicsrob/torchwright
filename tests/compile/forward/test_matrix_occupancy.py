"""2-D weight-matrix occupancy instrumentation (floor-plan viz).

Covers the ``matrices`` table + ``placements`` map emitted in the debug
sidecar (see ``_build_matrix_occupancy`` in compiler/export.py) and the
``PlacementRecorder`` that feeds it from weight writing.

The load-bearing invariant is **completeness**: the union of an op's
recorded placements must cover every nonzero entry the compile actually
wrote into that weight matrix.  If instrumentation misses a write site,
this fails.
"""

import json
import os
import tempfile

import pytest
import torch

from torchwright.compiler.export import (
    DEBUG_META_FORMAT,
    _build_matrix_occupancy,
    _dense_rects,
    _diag_rects,
    compile_to_onnx,
    debug_meta_path_for,
)
from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.graph_identity import canonical_ids, unwrap_debug
from torchwright.ops.relu.arithmetic_ops import multiply_2d
from torchwright.ops.linear import add
from torchwright.ops.inout_nodes import create_input

D = 256
D_HEAD = 16
N_HEADS = D // D_HEAD


def _build_graph():
    """A small graph that needs multiple layers (attention + MLP chains)."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    c = create_input("c", 1)
    prod = multiply_2d(a, b, max_abs1=10, max_abs2=10)
    out = add(prod, c)
    return out


def _compile():
    out = _build_graph()
    net = forward_compile(D, D_HEAD, out, verbose=False, device="cpu")
    canon = canonical_ids(unwrap_debug(out))
    return net, canon, out


# ---------------------------------------------------------------------------
# Pure rectangle encoding — pins dense (cross product) vs diagonal (paired)
# ---------------------------------------------------------------------------


def _expand(rects):
    """Expand placement rectangles to the set of (row, col) cells they fill."""
    cells = set()
    for r in rects:
        a0s, a0l = r["a0"]
        a1s, a1l = r["a1"]
        if r.get("diag"):
            assert a0l == a1l, "diagonal rect must have equal-length axes"
            cells.update((a0s + k, a1s + k) for k in range(a0l))
        else:
            for i in range(a0l):
                for j in range(a1l):
                    cells.add((a0s + i, a1s + j))
    return cells


def test_dense_rects_are_full_cross_product():
    rows = [0, 1, 2, 5, 6]  # runs (0,3) and (5,2)
    cols = [10, 11]  # run (10,2)
    rects = _dense_rects("M", rows, cols)
    # One rectangle per row-run x col-run pair; together the full product.
    assert len(rects) == 2
    assert all("diag" not in r for r in rects)
    assert _expand(rects) == {(r, c) for r in rows for c in cols}


def test_dense_rects_order_independent():
    # Unsorted input still yields the same cell set (dense = set semantics).
    assert _expand(_dense_rects("M", [6, 0, 1], [3, 2])) == {
        (r, c) for r in (0, 1, 6) for c in (2, 3)
    }


def test_diag_rects_pair_in_lockstep():
    rows = [0, 1, 2, 5]
    cols = [3, 4, 5, 9]
    rects = _diag_rects("M", rows, cols)
    # (0,3)(1,4)(2,5) coalesce into one len-3 segment; (5,9) is its own.
    assert len(rects) == 2
    assert all(r["diag"] for r in rects)
    assert _expand(rects) == {(0, 3), (1, 4), (2, 5), (5, 9)}


def test_diag_rects_break_when_only_one_axis_advances():
    # rows advance by 1 but cols jump -> no coalescing.
    rects = _diag_rects("M", [0, 1], [0, 5])
    assert len(rects) == 2
    assert _expand(rects) == {(0, 0), (1, 5)}


# ---------------------------------------------------------------------------
# Matrices table — shapes and axis labels
# ---------------------------------------------------------------------------


def test_matrices_table_shapes_and_axes():
    net, canon, _ = _compile()
    matrices, placements, n_heads, d_hidden = _build_matrix_occupancy(
        net, canon, D, D_HEAD
    )

    assert n_heads == N_HEADS
    assert isinstance(d_hidden, list)
    assert len(d_hidden) == len(net.layers)
    assert all(isinstance(h, int) and h > 0 for h in d_hidden)

    for i, dh in enumerate(d_hidden):
        expect = {
            f"L{i}.attn.W_Q": ([D, D], "residual_in", "head"),
            f"L{i}.attn.W_K": ([D, D], "residual_in", "head"),
            f"L{i}.attn.W_V": ([D, D], "residual_in", "head"),
            f"L{i}.attn.W_O": ([D, D], "head", "residual_out"),
            f"L{i}.mlp.W_in": ([D, dh], "residual_in", "hidden"),
            f"L{i}.mlp.W_out": ([dh, D], "hidden", "residual_out"),
        }
        for mid, (shape, a0, a1) in expect.items():
            rec = matrices[mid]
            assert rec["layer"] == i
            assert rec["shape"] == shape, mid
            assert rec["axis0"] == a0 and rec["axis1"] == a1, mid


# ---------------------------------------------------------------------------
# Completeness — placements cover every nonzero weight the compile wrote
# ---------------------------------------------------------------------------


def _flatten_weight(net, layer_i, kind):
    """The compiled weight tensor for one matrix kind, in flat 2-D coords."""
    layer = net.layers[layer_i]
    attn = layer.attn.attn
    if kind == "attn.W_Q":
        return attn.query_matrix.permute(1, 0, 2).reshape(D, N_HEADS * D_HEAD)
    if kind == "attn.W_K":
        return attn.key_matrix.permute(1, 0, 2).reshape(D, N_HEADS * D_HEAD)
    if kind == "attn.W_V":
        return attn.value_matrix.permute(1, 0, 2).reshape(D, N_HEADS * D_HEAD)
    if kind == "attn.W_O":
        return attn.output_matrix.reshape(N_HEADS * D_HEAD, D)
    if kind == "mlp.W_in":
        return layer.mlp.linear1.output_matrix
    if kind == "mlp.W_out":
        return layer.mlp.linear2.output_matrix
    raise ValueError(kind)


def test_placements_cover_all_nonzero_weights():
    # trim_heads=False keeps the full (n_heads, d, d_head) tensors so the flat
    # layout matches the sidecar's full-matrix description.
    out = _build_graph()
    net = forward_compile(D, D_HEAD, out, verbose=False, device="cpu", trim_heads=False)
    canon = canonical_ids(unwrap_debug(out))
    matrices, placements, _, _ = _build_matrix_occupancy(net, canon, D, D_HEAD)

    # Gather every rectangle per matrix id (across node keys AND reserved
    # "_<op_type>" buckets — completeness needs the node-less ops too).
    rects_by_matrix: dict[str, list] = {}
    for rects in placements.values():
        for r in rects:
            rects_by_matrix.setdefault(r["matrix"], []).append(r)

    n_checked = 0
    for mid, rec in matrices.items():
        layer_i, kind = rec["layer"], rec["kind"]
        weight = _flatten_weight(net, layer_i, kind)
        rows_n, cols_n = rec["shape"]
        assert tuple(weight.shape) == (rows_n, cols_n), mid

        covered = _expand(rects_by_matrix.get(mid, []))
        # All claimed cells are inside the matrix.
        for r, c in covered:
            assert 0 <= r < rows_n and 0 <= c < cols_n, f"{mid} cell ({r},{c}) OOB"

        nz = torch.nonzero(weight, as_tuple=False)
        for r, c in nz.tolist():
            assert (r, c) in covered, (
                f"{mid}: nonzero weight at ({r},{c}) not covered by any "
                f"placement — instrumentation missed a write site."
            )
            n_checked += 1

    assert n_checked > 0, "expected at least some nonzero weights to verify"


# ---------------------------------------------------------------------------
# Keying — every placement key is a canonical id or a reserved bucket
# ---------------------------------------------------------------------------


def test_placement_keys_are_canonical_or_bucket():
    net, canon, _ = _compile()
    _, placements, _, _ = _build_matrix_occupancy(net, canon, D, D_HEAD)

    valid_ids = set(canon.values())
    real_node_keys = 0
    for key in placements:
        if key.startswith("_"):
            continue  # reserved bucket (e.g. "_cancel", "_unreachable")
        assert int(key) in valid_ids, f"placement key {key!r} is not a canonical id"
        real_node_keys += 1
    assert real_node_keys > 0, "expected some placements keyed by real graph nodes"


# ---------------------------------------------------------------------------
# Sidecar integration — the ONNX export carries the occupancy data
# ---------------------------------------------------------------------------


def _token_parts():
    """A small (1-digit adder) token graph for the sidecar-occupancy tests."""
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        from examples.adder import create_network_parts

        return create_network_parts()
    finally:
        adder_module.max_digits = original


def _compile_sidecar(parts, tmpdir, name="model.onnx"):
    """Compile a token graph and return its debug sidecar dict.

    The debug sidecar (and its occupancy table) is written identically by the
    token exporter; a real token graph just gives compile_to_onnx an embedding
    to unembed against.
    """
    output_node, embedding = parts
    onnx_path = os.path.join(tmpdir, name)
    compile_to_onnx(
        output_node,
        embedding,
        onnx_path,
        d=1024,
        d_head=D_HEAD,
        max_seq_len=32,
        verbose=False,
    )
    with open(debug_meta_path_for(onnx_path)) as f:
        return json.load(f)


def _read_sidecar(tmpdir, name="model.onnx"):
    return _compile_sidecar(_token_parts(), tmpdir, name)


def test_sidecar_carries_occupancy():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _read_sidecar(tmpdir)

    assert data["format"] == DEBUG_META_FORMAT
    assert data["n_heads"] == data["d"] // data["d_head"]
    assert isinstance(data["d_hidden"], list)
    assert len(data["d_hidden"]) == data["n_layers"]
    assert data["matrices"], "matrices table should be non-empty"
    assert data["placements"], "placements should be non-empty"
    # Every placement references a declared matrix id.
    matrix_ids = set(data["matrices"])
    for rects in data["placements"].values():
        for r in rects:
            assert r["matrix"] in matrix_ids


def test_occupancy_stable_across_cache_hit(monkeypatch):
    # The schedule cache replays the CP-SAT solve, but weight writing (and thus
    # placement recording) always runs — a cache hit must still yield identical
    # occupancy data.
    # Two FRESH rebuilds, deliberately: the determinism sweep (node_id
    # tie-breaks + sorted node-set iteration, docs/lowering_copy_plan.md
    # decision 4) makes the schedule a function of topology and *relative*
    # id order, so a rebuild with different absolute ids must produce
    # identical occupancy between the cold solve and the warm replay.
    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TW_SCHEDULE_CACHE_DIR", cache_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            # cold: solves + caches
            first = _compile_sidecar(_token_parts(), tmpdir, "a.onnx")
            # warm: cache hit, fresh graph
            second = _compile_sidecar(_token_parts(), tmpdir, "b.onnx")

    for key in ("n_heads", "d_hidden", "matrices", "placements"):
        assert first[key] == second[key], f"{key} differs across cache hit"
