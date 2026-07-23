"""Tests for the range_printer example — two-level autoregressive loop.

Test 1 (oracle):        reference_eval with pre-computed state at every
                        PRINT position.  Validates graph logic without
                        compilation.
Test 2 (autoregressive): compile_headless + .step() loop with host-driven
                        feedback.  Validates the full round-trip.
Test 3 (probe):         probe_graph parity — compiled transformer matches
                        oracle at every node.
"""

import numpy as np
import pytest
import torch

from examples.range_printer import (
    D_HEAD,
    D_TOKEN_TYPE,
    E8_ITEM,
    E8_PRINT,
    build_range_printer_graph,
    out_active_col,
    out_done_flag,
    out_feedback_slice,
)
from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_graph, reference_eval

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _input_dict(n_pos: int, max_items: int):
    """Create a zeroed input dict for *n_pos* positions."""
    return {
        "col": torch.zeros(n_pos, 1),
        "is_new_item": torch.full((n_pos, 1), -1.0),
        "item_index": torch.zeros(n_pos, 1),
        "print_mask": torch.zeros(n_pos, max_items),
        "range_hi": torch.zeros(n_pos, 1),
        "range_lo": torch.zeros(n_pos, 1),
        "token_type": torch.zeros(n_pos, D_TOKEN_TYPE),
    }


def _set_item(inputs, pos, idx, lo, hi):
    inputs["token_type"][pos] = E8_ITEM
    inputs["item_index"][pos, 0] = float(idx)
    inputs["range_lo"][pos, 0] = float(lo)
    inputs["range_hi"][pos, 0] = float(hi)


def _set_print(inputs, pos, mask, col_val, is_new):
    inputs["token_type"][pos] = E8_PRINT
    inputs["print_mask"][pos, : len(mask)] = torch.tensor(mask, dtype=torch.float32)
    inputs["col"][pos, 0] = float(col_val)
    inputs["is_new_item"][pos, 0] = 1.0 if is_new else -1.0


def _expected_sequence(items):
    """Return the flat list of column values for items iterated in order."""
    seq = []
    for lo, hi in items:
        seq.extend(range(lo, hi))
    return seq


def _build_full_batch(items, max_items):
    """Build a full (prefill + print) batch with pre-computed oracle state.

    Returns ``(inputs, n_pos, expected_cols)``.
    """
    N = len(items)
    expected_cols = _expected_sequence(items)
    total_print = len(expected_cols)
    n_pos = N + total_print
    inputs = _input_dict(n_pos, max_items)

    # ITEM positions
    for i, (lo, hi) in enumerate(items):
        _set_item(inputs, i, idx=i, lo=lo, hi=hi)

    # PRINT positions with pre-computed state
    mask = np.zeros(max_items)
    pos = N
    for item_idx, (lo, hi) in enumerate(items):
        for c in range(lo, hi):
            is_new = c == lo
            _set_print(inputs, pos, mask, col_val=float(c), is_new=is_new)
            pos += 1
        mask[item_idx] = 1.0

    return inputs, n_pos, expected_cols


# ---------------------------------------------------------------------------
# Row builder for compiled autoregressive rollout
# ---------------------------------------------------------------------------


def _build_step_row(compiled, token_type_vec, max_items, **kwargs):
    """Build a ``(1, d_input)`` row from keyword fields."""
    mask = kwargs.get("mask", np.zeros(max_items))
    vals = {
        "col": torch.tensor([[kwargs.get("col_val", 0.0)]]),
        "is_new_item": torch.tensor([[kwargs.get("is_new", -1.0)]]),
        "item_index": torch.tensor([[kwargs.get("item_idx", 0.0)]]),
        "print_mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
        "range_hi": torch.tensor([[kwargs.get("hi", 0.0)]]),
        "range_lo": torch.tensor([[kwargs.get("lo", 0.0)]]),
        "token_type": token_type_vec.unsqueeze(0),
    }
    d_input = max(start + width for _, start, width in compiled._input_specs)
    row = torch.zeros(1, d_input)
    for name, start, width in compiled._input_specs:
        row[:, start : start + width] = vals[name]
    return row


# ---------------------------------------------------------------------------
# Test cases: (items, max_items)
# max_items must equal len(items), same as DOOM's max_walls == actual walls.
# ---------------------------------------------------------------------------

_THREE_ITEMS = [(2, 5), (7, 9), (0, 3)]  # 3 + 2 + 3 = 8 cols
_SINGLE_ITEM = [(5, 8)]  # 3 cols
_SINGLE_COL = [(3, 4)]  # 1 col
_MIXED = [(0, 1), (5, 8)]  # 1 + 3 = 4 cols

_CASES = [
    pytest.param(_THREE_ITEMS, id="three_items"),
    pytest.param(_SINGLE_ITEM, id="single_item"),
    pytest.param(_SINGLE_COL, id="single_col"),
    pytest.param(_MIXED, id="mixed"),
]


# ---------------------------------------------------------------------------
# Test 1: oracle (reference_eval)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("items", _CASES)
def test_oracle(items):
    max_items = len(items)
    output_node = build_range_printer_graph(max_items)
    inputs, n_pos, expected_cols = _build_full_batch(items, max_items)

    cache = reference_eval(output_node, inputs, n_pos)
    out = cache[output_node]

    N = len(items)
    idx_col = out_active_col()
    idx_done = out_done_flag()

    for k, expected in enumerate(expected_cols):
        got = out[N + k, idx_col].item()
        assert abs(got - expected) < 0.5, (
            f"step {k}: active_col={got:.2f}, expected {expected}"
        )

    # Last print position should have done_flag > 0
    last_pos = N + len(expected_cols) - 1
    assert out[last_pos, idx_done].item() > 0.0, (
        f"expected done_flag > 0 at last step, got {out[last_pos, idx_done].item():.2f}"
    )


# ---------------------------------------------------------------------------
# Test 2: compiled autoregressive rollout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("items", _CASES)
def test_autoregressive(items):
    max_items = len(items)
    output_node = build_range_printer_graph(max_items)

    compiled = compile_headless(
        output_node,
        d=256,
        d_head=D_HEAD,
        max_layers=200,
        verbose=False,
    )

    expected_cols = _expected_sequence(items)
    N = len(items)

    past = compiled.empty_past()
    step_idx = 0

    # Prefill: ITEM tokens
    for i, (lo, hi) in enumerate(items):
        row = _build_step_row(
            compiled,
            E8_ITEM,
            max_items,
            item_idx=float(i),
            lo=float(lo),
            hi=float(hi),
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step_idx)
        step_idx += 1

    # Autoregressive print loop
    mask = np.zeros(max_items)
    is_new = 1.0
    col_val = 0.0
    emitted = []

    fb_sl = out_feedback_slice(max_items)
    idx_col = out_active_col()
    idx_done = out_done_flag()

    max_steps = len(expected_cols) + 5  # generous upper bound
    for k in range(max_steps):
        row = _build_step_row(
            compiled,
            E8_PRINT,
            max_items,
            mask=mask,
            col_val=col_val,
            is_new=is_new,
        )
        with torch.no_grad():
            out, past = compiled.step(row, past, past_len=step_idx)
        step_idx += 1

        active_col = out[0, idx_col].item()
        done = out[0, idx_done].item()
        emitted.append(round(active_col))

        # Read feedback
        fb = out[0, fb_sl].detach().cpu().numpy()
        mask = np.round(fb[:max_items]).clip(0, 1)
        col_val = float(fb[max_items])
        is_new = float(fb[max_items + 1])

        if done > 0.0:
            break

    assert emitted == expected_cols, f"Got {emitted}, expected {expected_cols}"


# ---------------------------------------------------------------------------
# Test 3: probe_graph parity (compiled vs oracle)
# ---------------------------------------------------------------------------


def test_probe_parity():
    items = [(2, 4), (7, 9)]
    max_items = len(items)
    output_node = build_range_printer_graph(max_items)
    inputs, n_pos, _ = _build_full_batch(items, max_items)

    report = probe_graph(
        output_node,
        input_values=inputs,
        n_pos=n_pos,
        d=256,
        d_head=D_HEAD,
        max_layers=200,
        verbose=False,
        atol=1.0,
    )
    assert report.first_divergent is None, (
        f"Probe diverged at node: {report.first_divergent}"
    )
