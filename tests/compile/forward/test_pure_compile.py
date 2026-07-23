"""Compile is a pure function of the source graph.

This is D6 for L2 of ``docs/lowering_copy_plan.md``, the committed form
of the parked ``scripts/repro_double_compile.py`` / ``scripts/cert_diff.py``.

The motivating bug: compiling the same graph object twice silently
loosened every value bound on the second compile — the first compile's
in-place wrapper strip parked the claims where the second compile's
refresh loop clobbered them.  On the swish machine the loosened bounds
blew the RMSNorm energy certification; on relu the loosening was silent.
With ``lower()`` returning a compiler-private copy, recompiling is
re-lowering the same pristine source: the second compile must succeed
and produce a byte-identical debug sidecar.
"""

import json
import tempfile
from pathlib import Path

from torchwright.compiler.export import compile_to_onnx, debug_meta_path_for
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph.asserts import collect_asserts

D_HEAD = 16


def _token_parts():
    """The 1-digit adder token graph (the double-compile repro's graph)."""
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        from examples.adder import create_network_parts

        return create_network_parts()
    finally:
        adder_module.max_digits = original


def _compile_sidecar(parts, tmpdir, name):
    output_node, embedding = parts
    onnx_path = str(Path(tmpdir) / name)
    compile_to_onnx(
        output_node,
        embedding,
        onnx_path,
        d=1024,
        d_head=D_HEAD,
        max_seq_len=32,
        verbose=False,
    )
    with Path(debug_meta_path_for(onnx_path)).open() as f:
        return json.load(f)


def _graph_snapshot(output_node):
    return {
        n.node_id: (
            type(n).__name__,
            [i.node_id for i in n.inputs],
            (n.value_type.value_range.lo, n.value_type.value_range.hi),
        )
        for n in get_ancestor_nodes({output_node})
    }


def test_double_compile_same_graph_object_identical_sidecars():
    """Second compile of the same graph object succeeds and matches the first.

    The sidecar is identical to the first's: no bound loosening, no
    schedule drift.
    """
    parts = _token_parts()
    output_node = parts[0]

    before = _graph_snapshot(output_node)
    n_asserts = len(collect_asserts(output_node))

    with tempfile.TemporaryDirectory() as tmpdir:
        first = _compile_sidecar(parts, tmpdir, "a.onnx")
        # The motivating failure fired HERE: the swish machine's second
        # compile crossed the rms_norm energy budget.
        second = _compile_sidecar(parts, tmpdir, "b.onnx")

    assert first == second, "debug sidecars differ across a recompile"

    # The source graph is untouched by both compiles: same node set, same
    # wiring, bit-identical value ranges, checks intact.
    assert _graph_snapshot(output_node) == before
    assert len(collect_asserts(output_node)) == n_asserts
