"""The ONNX embed-table zero-init contract.

The scheduler skips BIRTH-layer dirty-column cancels because it assumes the
runtime hands it a residual stream that is zero in every column a fresh
allocation lands on (``assume_zero_init`` was retired in favour of this being
universal — see ``compile.py`` and ``forward/cpsat_scheduler.py``).  For the
ONNX runtime that guarantee is provided by the ``embed_table`` construction in
``compile_to_onnx``: the table starts all-zeros and only the embedding,
pinned-RMSNorm, and literal-seed (const-1 self-match) columns are written, so a
per-token ``Gather`` produces a residual stream that is exactly zero in every
free-pool column.  If that scatter ever seeds a scheduler-allocated column, a
compiled artifact would compute ``seed + value`` instead of ``value`` in that
column with no dirty cancel to catch it — this test is the guard.

Reads the ONNX file directly (no onnxruntime), so it runs locally.
"""

import os
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper

from torchwright.compiler.export import compile_to_onnx
from torchwright.compiler.forward.compile import forward_compile

D = 256
D_HEAD = 16


def _adder_token_parts():
    """A small (1-digit adder) token graph: ``(output_node, embedding)``."""
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        from examples.adder import create_network_parts

        return create_network_parts()
    finally:
        adder_module.max_digits = original


def _embed_table_nonzero_columns(onnx_path: str, d: int) -> set:
    """Columns where the ONNX ``embed_table`` is nonzero for any vocab row.

    The table is stored COO-sparse (mostly zeros); the sparse indices name the
    nonzero cells directly, so no dense reconstruction is needed.
    """
    model = onnx.load(onnx_path)
    for si in model.graph.sparse_initializer:
        if si.values.name != "embed_table":
            continue
        idx = numpy_helper.to_array(si.indices)
        if idx.ndim == 2:  # (nnz, 2) row/col pairs
            return {int(c) for c in idx[:, 1]}
        return {int(i % d) for i in idx}  # linearized row-major
    raise AssertionError("embed_table sparse initializer not found in ONNX file")


def _layer0_seeded_columns(output_node) -> set:
    """The residual columns allocated at layer-0 entry (input nodes + the
    const-1 self-match column) — the only columns the embed-table may seed.

    Taken from the same deterministic ``forward_compile`` the ONNX exporter
    runs; the layer-0 allocation happens before any scheduling, so it is
    identical to the exporter's regardless of the schedule.
    """
    net = forward_compile(
        output_node=output_node,
        d=D,
        d_head=D_HEAD,
        optimize=0,
        rms_norm=False,
        verbose=False,
    )
    in_state = net.layers[0].attn.in_state
    ra = net.residual_assignment
    assert ra is not None
    seeded = set()
    for node in ra.get_nodes(in_state):
        seeded |= {int(i) for i in ra.get_node_indices(in_state, node)}
    return seeded


def test_token_onnx_embed_table_zeroes_free_pool_columns():
    """Every free-pool column (the ones the scheduler allocates for
    intermediate / output nodes assuming zero-init) is zero in the ONNX
    embed-table.  ``rms_norm=False`` so there are no reserved constant columns
    to complicate the seedable set (embedding + const-1 only)."""
    output_node, embedding = _adder_token_parts()
    seeded = _layer0_seeded_columns(output_node)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        compile_to_onnx(
            output_node,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            max_seq_len=32,
            rms_norm=False,
            verbose=False,
        )
        nonzero = _embed_table_nonzero_columns(onnx_path, D)

    free_pool = set(range(D)) - seeded
    # The table only ever seeds layer-0 columns — never a free-pool column.
    assert nonzero <= seeded, (
        "embed_table seeds free-pool columns "
        f"{sorted(nonzero - seeded)}, breaking the zero-init contract the "
        "scheduler relies on when it skips BIRTH-layer dirty cancels"
    )
    # And concretely: the free pool is large and provably zero on entry.
    assert free_pool, "expected a non-empty free pool for this graph"
    assert nonzero.isdisjoint(free_pool)


def test_token_onnx_embed_table_is_not_trivially_empty():
    """Guard the guard: the embedding + const-1 columns really are seeded, so
    the zero-check above is meaningful and not passing on an all-zero table."""
    output_node, embedding = _adder_token_parts()
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        compile_to_onnx(
            output_node,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            max_seq_len=32,
            rms_norm=False,
            verbose=False,
        )
        nonzero = _embed_table_nonzero_columns(onnx_path, D)
    assert len(nonzero) > 1, "expected embedding + const-1 columns to be seeded"
