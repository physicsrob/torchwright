"""Tests for DebugWatch nodes and debug=True compiled forward.

Covers:

* DebugWatch.compute() prints when the predicate fires, does not raise.
* DebugWatch.compute() is silent when the predicate passes.
* DebugWatch nodes are stripped during compilation — output is identical.
* collect_watches / collect_debug_nodes find reachable nodes.
* compiled(inputs, debug=True) checks asserts (raises) and watches (prints).
* compiled.step(inputs, past, debug=True) exercises both.
* compile_headless auto-collects asserts and watches onto the module.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import reference_eval
from torchwright.graph import Assert, DebugWatch, Concatenate
from torchwright.graph.asserts import (
    assert_in_range,
    collect_asserts,
    collect_watches,
    collect_debug_nodes,
    debug_watch,
)
from torchwright.ops.relu.arithmetic_ops import clamp
from torchwright.ops.inout_nodes import create_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval(node, inputs, n_pos=1):
    values = reference_eval(node, inputs, n_pos)
    return values[node]


def _build_graph(with_watch=False, with_assert=False):
    x = create_input("x", 1)
    y = clamp(x, -1.0, 1.0)
    if with_watch:

        def watch_pred(t):
            bad = t.abs() > 0.5
            if bad.any():
                return False, f"max={t.abs().max().item():.4f}"
            return True, ""

        y = debug_watch(y, watch_pred, message="big value")
    if with_assert:
        y = assert_in_range(y, -1.0, 1.0)
    return x, y


def _compile(output, **kwargs):
    defaults = dict(d=256, d_head=16, max_layers=40, verbose=False)
    defaults.update(kwargs)
    return compile_headless(output, **defaults)


def _make_input(mod, x_val):
    d_input = max(s + w for _, s, w in mod._input_specs)
    inp = torch.zeros(1, d_input)
    inp[:, mod._input_specs[0][1]] = x_val
    return inp


# ---------------------------------------------------------------------------
# DebugWatch.compute() behavior
# ---------------------------------------------------------------------------


def test_debug_watch_compute_prints(capsys):
    x, y = _build_graph(with_watch=True)
    inputs = {"x": torch.tensor([[0.8]])}
    out = _eval(y, inputs)
    captured = capsys.readouterr()
    assert "DebugWatch" in captured.out
    assert "big value" in captured.out
    assert torch.allclose(out, torch.tensor([[0.8]]))


def test_debug_watch_compute_silent(capsys):
    x, y = _build_graph(with_watch=True)
    inputs = {"x": torch.tensor([[0.3]])}
    out = _eval(y, inputs)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert torch.allclose(out, torch.tensor([[0.3]]))


def test_debug_watch_does_not_raise():
    x, y = _build_graph(with_watch=True)
    inputs = {"x": torch.tensor([[0.99]])}
    _eval(y, inputs)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def test_collect_watches_finds_reachable():
    x, y = _build_graph(with_watch=True)
    watches = collect_watches(y)
    assert len(watches) == 1
    assert isinstance(watches[0], DebugWatch)


def test_collect_watches_returns_empty():
    x = create_input("x", 1)
    assert collect_watches(x) == []


def test_collect_debug_nodes_finds_both():
    x, y = _build_graph(with_watch=True, with_assert=True)
    asserts, watches = collect_debug_nodes(y)
    assert len(asserts) >= 1
    assert len(watches) == 1


# ---------------------------------------------------------------------------
# Compile stripping — output identical with or without DebugWatch
# ---------------------------------------------------------------------------


def test_compile_strips_watches_and_preserves_output():
    _, out_a = _build_graph(with_watch=False)
    mod_a = _compile(out_a)

    _, out_b = _build_graph(with_watch=True)
    mod_b = _compile(out_b)

    assert mod_a._n_layers == mod_b._n_layers

    inp_a = _make_input(mod_a, 0.7)
    inp_b = _make_input(mod_b, 0.7)

    out_a = mod_a(inp_a)
    out_b = mod_b(inp_b)
    assert torch.allclose(out_a, out_b, atol=1e-6)


# ---------------------------------------------------------------------------
# Auto-collection on CompiledHeadless
# ---------------------------------------------------------------------------


def test_auto_collection_asserts():
    _, out = _build_graph(with_assert=True)
    mod = _compile(out)
    assert len(mod._asserts) >= 1
    assert all(isinstance(a, Assert) for a in mod._asserts)


def test_auto_collection_watches():
    _, out = _build_graph(with_watch=True)
    mod = _compile(out)
    assert len(mod._watches) == 1
    assert all(isinstance(w, DebugWatch) for w in mod._watches)


def test_auto_collection_both():
    _, out = _build_graph(with_watch=True, with_assert=True)
    mod = _compile(out)
    assert len(mod._asserts) >= 1
    assert len(mod._watches) == 1


# ---------------------------------------------------------------------------
# debug=True on __call__
# ---------------------------------------------------------------------------


def test_debug_forward_runs_watches(capsys):
    _, out = _build_graph(with_watch=True)
    mod = _compile(out)
    inp = _make_input(mod, 0.8)
    result = mod(inp, debug=True)
    captured = capsys.readouterr()
    assert "DebugWatch" in captured.out
    assert result.shape == (1, 1)


def test_debug_forward_silent_when_ok(capsys):
    _, out = _build_graph(with_watch=True)
    mod = _compile(out)
    inp = _make_input(mod, 0.3)
    mod(inp, debug=True)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_debug_forward_runs_asserts():
    _, out = _build_graph(with_assert=True)
    mod = _compile(out)
    inp = _make_input(mod, 0.5)
    mod(inp, debug=True)


# ---------------------------------------------------------------------------
# debug=True on step
# ---------------------------------------------------------------------------


def test_debug_step_runs_watches(capsys):
    _, out = _build_graph(with_watch=True)
    mod = _compile(out)
    inp = _make_input(mod, 0.8)
    past = mod.empty_past()
    result, new_past = mod.step(inp, past, debug=True)
    captured = capsys.readouterr()
    assert "DebugWatch" in captured.out
    assert result.shape == (1, 1)
    assert len(new_past) == 2


def test_debug_step_silent_when_ok(capsys):
    _, out = _build_graph(with_watch=True)
    mod = _compile(out)
    inp = _make_input(mod, 0.3)
    past = mod.empty_past()
    mod.step(inp, past, debug=True)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_debug_false_does_not_check(capsys):
    """Normal forward (debug=False) should not trigger watches."""
    _, out = _build_graph(with_watch=True)
    mod = _compile(out)
    inp = _make_input(mod, 0.8)
    mod(inp, debug=False)
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Verify extracted value correctness
# ---------------------------------------------------------------------------


def test_debug_watch_sees_correct_compiled_value():
    """The watch predicate receives the actual compiled value, not garbage.

    Build a graph: y = clamp(x, -1, 1).  Attach a watch that captures
    the tensor it receives.  Run compiled forward with debug=True and
    verify the captured tensor matches the compiled output (which for
    clamp(0.7, -1, 1) should be ~0.7).
    """
    captured_values = []

    x = create_input("x", 1)
    y = clamp(x, -1.0, 1.0)

    def capture_pred(t):
        captured_values.append(t.detach().clone())
        return True, ""

    y = debug_watch(y, capture_pred, message="capture")
    mod = _compile(y)

    inp = _make_input(mod, 0.7)
    output = mod(inp, debug=True)

    assert len(captured_values) == 1
    watched = captured_values[0]
    assert watched.shape[1] == 1
    assert abs(watched[0, 0].item() - output[0, 0].item()) < 1e-4


def test_debug_watch_fires_on_correct_threshold():
    """A watch with threshold 0.5 fires for x=0.8 but not x=0.3.

    Verifies the predicate is evaluating the real compiled value,
    not a default or zero tensor.
    """
    fired_values = []

    def build_with_recording_watch():
        x = create_input("x", 1)
        y = clamp(x, -1.0, 1.0)

        def threshold_pred(t):
            if t.abs().max() > 0.5:
                fired_values.append(t.detach().clone())
                return False, f"val={t.abs().max().item():.4f}"
            return True, ""

        y = debug_watch(y, threshold_pred, message="threshold")
        return y

    y = build_with_recording_watch()
    mod = _compile(y)

    # x=0.3: clamp(0.3) = 0.3, below threshold, should not fire
    fired_values.clear()
    inp = _make_input(mod, 0.3)
    mod(inp, debug=True)
    assert len(fired_values) == 0

    # x=0.8: clamp(0.8) = 0.8, above threshold, should fire
    inp = _make_input(mod, 0.8)
    mod(inp, debug=True)
    assert len(fired_values) == 1
    assert abs(fired_values[0][0, 0].item() - 0.8) < 0.05


# ---------------------------------------------------------------------------
# Self-consistency check
# ---------------------------------------------------------------------------


def test_debug_forward_runs_consistency_check():
    """debug=True should not raise on a healthy graph (consistency passes)."""
    _, out = _build_graph(with_watch=True)
    mod = _compile(out)
    inp = _make_input(mod, 0.5)
    mod(inp, debug=True)


def test_debug_step_runs_consistency_check():
    """debug=True step should not raise on a healthy graph."""
    _, out = _build_graph(with_watch=True)
    mod = _compile(out)
    inp = _make_input(mod, 0.5)
    past = mod.empty_past()
    mod.step(inp, past, debug=True)


# ---------------------------------------------------------------------------
# debug_value()
# ---------------------------------------------------------------------------


def test_debug_value_returns_compiled_value():
    """debug_value(node) returns the node's compiled value after debug forward."""
    x = create_input("x", 1)
    y = clamp(x, -1.0, 1.0)
    mod = _compile(y)

    inp = _make_input(mod, 0.7)
    output = mod(inp, debug=True)

    val = mod.debug_value(y)
    assert val is not None
    assert val.shape[1] == 1
    assert abs(val[0, 0].item() - output[0, 0].item()) < 1e-4


def test_debug_value_on_intermediate():
    """debug_value works on intermediate nodes, not just the output."""
    x = create_input("x", 1)
    y = clamp(x, -1.0, 1.0)
    from torchwright.ops.linear import add_const

    z = add_const(y, 1.0)
    mod = _compile(z)

    inp = _make_input(mod, 0.7)
    mod(inp, debug=True)

    val_y = mod.debug_value(y)
    assert val_y is not None
    assert abs(val_y[0, 0].item() - 0.7) < 0.05

    val_z = mod.debug_value(z)
    assert val_z is not None
    assert abs(val_z[0, 0].item() - 1.7) < 0.05


def test_debug_value_unwraps_assert():
    """debug_value unwraps Assert/DebugWatch wrappers to find the target."""
    _, out = _build_graph(with_assert=True)
    mod = _compile(out)
    inp = _make_input(mod, 0.5)
    mod(inp, debug=True)

    val = mod.debug_value(out)
    assert val is not None
    assert abs(val[0, 0].item() - 0.5) < 0.05


def test_debug_value_raises_without_debug_forward():
    """debug_value raises RuntimeError if no debug forward has been run."""
    _, out = _build_graph()
    mod = _compile(out)
    with pytest.raises(RuntimeError, match="requires a prior debug=True"):
        mod.debug_value(out)


def test_debug_value_after_step():
    """debug_value works after a debug=True step call."""
    x = create_input("x", 1)
    y = clamp(x, -1.0, 1.0)
    mod = _compile(y)

    inp = _make_input(mod, 0.4)
    past = mod.empty_past()
    output, _ = mod.step(inp, past, debug=True)

    val = mod.debug_value(y)
    assert val is not None
    assert abs(val[0, 0].item() - output[0, 0].item()) < 1e-4
