"""Tests for :mod:`torchwright.debug.probe`.

The probe compares a compiled ``HeadlessTransformer`` against its
source graph's ``node.compute`` output using the per-sublayer snapshots
the compiler writes into ``residual_assignment``.  A healthy graph
must produce an empty divergence report.
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import (
    probe_attention,
    probe_layer_diff,
    probe_residual,
    reference_eval,
)
from torchwright.graph import Attn
from torchwright.ops.linear import add_const, multiply_const
from torchwright.ops.inout_nodes import create_input


def test_reference_eval_matches_direct_compute_tiny():
    """Build a tiny graph (small enough that unmemoised direct compute
    is fast) and check that ``reference_eval`` returns exactly the same
    top-level value as ``output_node.compute``.  Also checks that the
    class-level ``compute`` monkey-patches are restored cleanly: a
    second unmemoised compute after reference_eval must still produce
    the identical tensor.
    """
    from torchwright.graph.linear import Linear
    from torchwright.graph.misc import InputNode
    from torchwright.ops.linear import add, add_const, multiply_const

    x = InputNode("x", 1, value_range=(-100.0, 100.0))
    y = InputNode("y", 1, value_range=(-100.0, 100.0))
    # Graph: output = 2*x + (y + 3)  -- ~4 Linear nodes, easy to reason about.
    scaled = multiply_const(x, 2.0)
    shifted = add_const(y, 3.0)
    output = add(scaled, shifted)

    input_values = {
        "x": torch.tensor([[5.0], [0.5], [-1.0]]),
        "y": torch.tensor([[1.0], [2.0], [7.0]]),
    }
    n_pos = 3
    expected = torch.tensor([[14.0], [6.0], [8.0]])

    cache = reference_eval(output, input_values, n_pos)

    assert output in cache
    assert torch.allclose(cache[output], expected, atol=1e-6)
    # The un-memoised direct call must still work — i.e. the class-level
    # monkey-patches were restored.
    direct = output.compute(n_pos, input_values)
    assert torch.allclose(direct, expected, atol=1e-6)
    # Every intermediate is also cached.
    assert x in cache and y in cache
    assert torch.allclose(cache[x], input_values["x"])
    assert torch.allclose(cache[y], input_values["y"])


def test_probe_residual_reads_intermediate_node():
    """Compile a tiny chain and confirm ``probe_residual`` reports the
    intermediate node's value at the layer that materialises it.

    Graph: y = 3*x, z = y + 1.  Compile with output [z, y], probe y.
    (y is also an output leaf so it has two consumers — otherwise the
    lowering boundary's linear fusion would absorb it into z and there
    would be no intermediate to probe.)  Every layer that has y live
    must hold 3*x; empty per_layer would indicate the probe failed to
    surface the node.
    """
    from torchwright.graph.misc import Concatenate

    x = create_input(2)
    y = multiply_const(x, 3.0)
    z = add_const(y, 1.0)
    module = compile_headless(
        Concatenate([z, y]),
        d=64,
        verbose=False,
    )

    inp = torch.tensor([[2.0, 4.0]])
    report = probe_residual(module, inp, y)

    assert report.layers, "probe_residual surfaced no layers for y"
    expected_y = torch.tensor([[6.0, 12.0]])
    for layer_i in report.layers:
        v = report.at(layer_i)
        assert v is not None
        assert torch.allclose(v, expected_y, atol=0.1), (
            f"layer {layer_i}: expected {expected_y.tolist()} got {v.tolist()}"
        )
    # at_layer filter: restrict to the last layer that holds y.
    one = probe_residual(module, inp, y, at_layer=report.layers[-1])
    assert one.layers == [report.layers[-1]]


def test_probe_attention_captures_softmax_weights():
    """Build a tiny single-Attn graph, compile, and confirm
    ``probe_attention`` returns per-head softmax weights that sum to
    one, with at least one non-trivial top contributor at the queried
    row.  This exercises the monkey-patch + forward path end-to-end on
    a graph small enough that layer-hosting lookup is unambiguous.
    """
    x = create_input("x", 4)
    torch.manual_seed(0)
    attn = Attn(
        query_in=x,
        key_in=x,
        value_in=x,
        # Q/K are full-width d_head=16 (every head is rotary on the global grid).
        query_matrix=torch.randn(4, 16),
        key_matrix=torch.randn(4, 16),
        value_matrix=torch.randn(4, 4),
        output_matrix=torch.randn(4, 4),
    )
    module = compile_headless(
        attn,
        d=256,
        d_head=16,
        verbose=False,
    )

    inp = torch.randn(3, 4)
    report = probe_attention(module, inp, attn, query_pos=2)

    assert report.layer_index >= 0
    # Softmax sums to 1 per head (within fp tolerance).
    sums = report.weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
        f"weights don't sum to 1 per head: {sums.tolist()}"
    )
    # Causal mask: query row 2 can attend to positions 0, 1, 2 only.
    # Positions 3..n_keys-1 (if any) must have negligible weight.
    if report.weights.shape[1] > 3:
        future_mass = report.weights[:, 3:].sum().item()
        assert future_mass < 1e-3, (
            f"causal mask leak — future positions got {future_mass:.3g}"
        )
    # top() returns a ranked list.
    top = report.top(k=3, head=0)
    assert len(top) == 3
    # Sorted descending by weight.
    assert top[0][1] >= top[1][1] >= top[2][1]


def test_probe_layer_diff_drift_and_sentinel():
    """Compile y = 3*x, sample at multiple positions, and verify
    ``probe_layer_diff``:

    * reports ``first_drift_layer is None`` when given the correct
      reference (no drift beyond tolerance),
    * reports a populated ``first_drift_layer`` when given a wrong
      reference,
    * reports ``first_sentinel_layer`` at the earliest layer where
      the node equals a caller-supplied sentinel (set to the first
      per-layer value).
    """
    x = create_input("x", 2)
    y = multiply_const(x, 3.0)
    module = compile_headless(
        y,
        d=64,
        verbose=False,
    )

    inp = torch.tensor([[2.0, 4.0], [1.0, 5.0]])
    positions = [0, 1]
    true_ref = torch.tensor([[6.0, 12.0], [3.0, 15.0]])

    # Correct reference → no drift.
    ok = probe_layer_diff(
        module,
        inp,
        y,
        reference=true_ref,
        positions=positions,
        drift_threshold=0.5,
    )
    assert ok.records, "no layers traced"
    assert ok.first_drift_layer is None, (
        f"unexpected drift at layer {ok.first_drift_layer}: "
        f"max_abs_delta={ok.records[0].max_abs_delta}"
    )

    # Wrong reference → drift flagged at the first layer.
    bad_ref = torch.zeros_like(true_ref)
    bad = probe_layer_diff(
        module,
        inp,
        y,
        reference=bad_ref,
        positions=positions,
        drift_threshold=0.5,
    )
    assert bad.first_drift_layer is not None
    assert bad.first_drift_layer == bad.records[0].layer_index

    # Sentinel detection: the first layer's value should contain 6.0
    # (the top-left element).  Using that as the sentinel must flag the
    # first layer; a value that never surfaces must not flag anything.
    s_hit = probe_layer_diff(
        module,
        inp,
        y,
        reference=true_ref,
        positions=positions,
        sentinel=6.0,
        sentinel_tol=0.1,
    )
    assert s_hit.first_sentinel_layer == s_hit.records[0].layer_index
    assert s_hit.sentinel_value == 6.0

    s_miss = probe_layer_diff(
        module,
        inp,
        y,
        reference=true_ref,
        positions=positions,
        sentinel=-999.0,
        sentinel_tol=0.01,
    )
    assert s_miss.first_sentinel_layer is None
