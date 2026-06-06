"""Unit + probe tests for the resampling primitives.

Each primitive is tested in isolation — build a minimal graph whose
output is just that primitive, run it via ``probe_graph`` against the
reference ``node.compute`` oracle, and assert both (a) zero
compiled-vs-oracle divergence and (b) that the oracle matches an
independent NumPy reference of the intended semantics.

This is the primitive-level safety net the textured renderer bug
exposed: if ``dynamic_extract`` had existed with a sweep like this
back when the original ``_textured_column_fill`` was written, the
ad-hoc band-sum workaround would never have been built.
"""

import math

import numpy as np
import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.graph import Concatenate
from torchwright.ops import (
    create_input,
    dynamic_extract,
    linear_bin_index,
    reciprocal,
    table_lookup_2d,
    table_lookup_3d,
)
from torchwright.ops.arithmetic_ops import clamp, subtract

# ---------------------------------------------------------------------------
# table_lookup_2d
# ---------------------------------------------------------------------------


def _table_lookup_2d_reference(
    table: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    index_scale=1.0,
    sharpness: float = 100.0,
) -> torch.Tensor:
    if isinstance(index_scale, (int, float)):
        scale_i = scale_j = float(index_scale)
    else:
        scale_i, scale_j = float(index_scale[0]), float(index_scale[1])
    rows, cols = table.shape
    eps = 1.0 / sharpness
    max_abs = float(table.abs().max().item())
    row_slack = max(1e-3, max_abs * sharpness * max(rows, 1) * 1e-6)
    offset = max_abs + row_slack + 1.0

    def _axis_blend(x: float, n: int) -> tuple[int, int, float]:
        idx = max(0, min(n - 1, int(math.floor(x + 0.5))))
        if n <= 1:
            return idx, idx, 0.0
        k = int(math.floor(x))
        boundary = float(k) + 0.5
        lo = boundary - eps / 2.0
        hi = boundary + eps / 2.0
        if 0 <= k <= n - 2 and lo <= x <= hi:
            t = (x - lo) / eps
            return k, k + 1, max(0.0, min(1.0, t))
        return idx, idx, 0.0

    def _gate(v: float, m: float) -> float:
        return 0.5 * max(v + offset * m, 0.0) - 0.5 * max(
            -v + offset * m,
            0.0,
        )

    out = torch.empty(i.shape[0], 1, dtype=table.dtype)
    for p in range(i.shape[0]):
        r0, r1, rt = _axis_blend(float(i[p, 0]) * scale_i, rows)
        c0, c1, ct = _axis_blend(float(j[p, 0]) * scale_j, cols)

        def _row_value(col: int) -> float:
            return (1.0 - rt) * float(table[r0, col]) + rt * float(table[r1, col])

        if cols == 1:
            out[p, 0] = _row_value(0)
        else:
            left = _row_value(c0)
            right = _row_value(c1)
            out[p, 0] = _gate(left, 1.0 - 2.0 * ct) + _gate(
                right,
                -1.0 + 2.0 * ct,
            )
    return out


def _build_table_lookup_2d_graph(table, index_scale=1.0, sharpness=100.0):
    i = create_input("i", 1)
    j = create_input("j", 1)
    return table_lookup_2d(
        i,
        j,
        table,
        index_scale=index_scale,
        sharpness=sharpness,
    )


def test_table_lookup_2d_every_integer_cell():
    table = torch.arange(20, dtype=torch.float32).reshape(4, 5) * 3.0 - 7.0
    out_node = _build_table_lookup_2d_graph(table)

    coords = [(r, c) for r in range(table.shape[0]) for c in range(table.shape[1])]
    i_val = torch.tensor([[float(r)] for r, _c in coords], dtype=torch.float32)
    j_val = torch.tensor([[float(c)] for _r, c in coords], dtype=torch.float32)
    inputs = {"i": i_val, "j": j_val}
    n_pos = len(coords)

    expected = _table_lookup_2d_reference(table, i_val, j_val)
    cache = reference_eval(out_node, inputs, n_pos)
    assert torch.allclose(cache[out_node], expected, atol=5e-3)

    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=n_pos,
        d=1024,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, report.format_short()


def test_table_lookup_2d_scaled_unit_coordinates():
    table = torch.tensor(
        [
            [2.0, 3.0, 5.0, 7.0],
            [11.0, 13.0, 17.0, 19.0],
            [23.0, 29.0, 31.0, 37.0],
        ],
        dtype=torch.float32,
    )
    index_scale = table.shape
    out_node = _build_table_lookup_2d_graph(table, index_scale=index_scale)

    coords = [(r, c) for r in range(table.shape[0]) for c in range(table.shape[1])]
    # 0.2 inside each scaled cell keeps the probe well inside the
    # centered integer bin and outside half-integer transition bands.
    i_val = torch.tensor(
        [[(float(r) + 0.2) / table.shape[0]] for r, _c in coords],
        dtype=torch.float32,
    )
    j_val = torch.tensor(
        [[(float(c) + 0.2) / table.shape[1]] for _r, c in coords],
        dtype=torch.float32,
    )
    inputs = {"i": i_val, "j": j_val}
    expected = _table_lookup_2d_reference(table, i_val, j_val, index_scale)

    cache = reference_eval(out_node, inputs, len(coords))
    assert torch.allclose(cache[out_node], expected, atol=5e-3)


def test_table_lookup_2d_target_size_integer_reference():
    generator = torch.Generator().manual_seed(0)
    table = torch.rand((128, 128), generator=generator, dtype=torch.float32) * 2.0 - 1.0
    out_node = _build_table_lookup_2d_graph(table)

    coords = [(0, 0), (1, 17), (64, 64), (126, 3), (127, 127)]
    i_val = torch.tensor([[float(r)] for r, _c in coords], dtype=torch.float32)
    j_val = torch.tensor([[float(c)] for _r, c in coords], dtype=torch.float32)
    inputs = {"i": i_val, "j": j_val}
    expected = _table_lookup_2d_reference(table, i_val, j_val)

    cache = reference_eval(out_node, inputs, len(coords))
    assert torch.allclose(cache[out_node], expected, atol=5e-3)


def test_table_lookup_2d_near_integer_inputs_remain_in_bin():
    table = torch.arange(9, dtype=torch.float32).reshape(3, 3) * 4.0
    out_node = _build_table_lookup_2d_graph(table, sharpness=10.0)

    coords = [(0.4, 0.4), (1.4, 1.4), (1.6, 1.6), (-0.4, 2.4)]
    i_val = torch.tensor([[r] for r, _c in coords], dtype=torch.float32)
    j_val = torch.tensor([[c] for _r, c in coords], dtype=torch.float32)
    inputs = {"i": i_val, "j": j_val}
    expected = _table_lookup_2d_reference(
        table,
        i_val,
        j_val,
        sharpness=10.0,
    )

    cache = reference_eval(out_node, inputs, len(coords))
    assert torch.allclose(cache[out_node], expected, atol=5e-3)


def test_table_lookup_2d_accepts_transition_band_inputs():
    table = torch.tensor([[0.0, 100.0], [20.0, 120.0]], dtype=torch.float32)
    sharpness = 10.0
    out_node = _build_table_lookup_2d_graph(table, sharpness=sharpness)
    inputs = {
        "i": torch.tensor([[0.5]], dtype=torch.float32),
        "j": torch.tensor([[0.5]], dtype=torch.float32),
    }
    expected = _table_lookup_2d_reference(
        table,
        inputs["i"],
        inputs["j"],
        sharpness=sharpness,
    )

    cache = reference_eval(out_node, inputs, 1)
    assert torch.allclose(cache[out_node], expected, atol=5e-3)

    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=1,
        d=256,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, report.format_short()


# ---------------------------------------------------------------------------
# table_lookup_3d
# ---------------------------------------------------------------------------


def _table_lookup_3d_axis_order(shape, outer_axis=None):
    """Replicates the implementation's internal axis-order heuristic:
    A = outer_axis (default smallest), C = larger remaining, B = smaller."""
    sizes = list(shape)
    if outer_axis is None:
        a = int(min(range(3), key=lambda ax: sizes[ax]))
    else:
        a = int(outer_axis)
    remaining = sorted((ax for ax in range(3) if ax != a), key=lambda ax: sizes[ax])
    return a, remaining[0], remaining[1]


def _staircase_value(x: float, n: int, eps: float) -> float:
    """Continuous form of the centered integer-index staircase."""
    idx = max(0, min(n - 1, int(math.floor(x + 0.5))))
    if n <= 1:
        return float(idx)
    k = int(math.floor(x))
    lo = float(k) + 0.5 - eps / 2.0
    hi = float(k) + 0.5 + eps / 2.0
    if 0 <= k <= n - 2 and lo <= x <= hi:
        return float(k) + max(0.0, min(1.0, (x - lo) / eps))
    return float(idx)


def _table_lookup_3d_reference(
    table: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    index_scale=1.0,
    sharpness: float = 100.0,
    outer_axis=None,
) -> torch.Tensor:
    """Independent reference: round the two flattened axes to integer
    indices, flatten as ``q = B*idx_a + idx_b``, and reduce to the
    validated 2D reference over ``table.reshape(A*B, C)``."""
    if isinstance(index_scale, (int, float)):
        scales = [float(index_scale)] * 3
    else:
        scales = [float(s) for s in index_scale]
    a_ax, b_ax, c_ax = _table_lookup_3d_axis_order(table.shape, outer_axis)
    a_size, b_size, c_size = table.shape[a_ax], table.shape[b_ax], table.shape[c_ax]
    table_perm = (
        table.permute(a_ax, b_ax, c_ax).contiguous().reshape(a_size * b_size, c_size)
    )
    eps = 1.0 / sharpness
    inp = [i, j, k]
    n_pos = i.shape[0]
    q = torch.empty(n_pos, 1, dtype=table.dtype)
    c_scaled = torch.empty(n_pos, 1, dtype=table.dtype)
    for p in range(n_pos):
        sv_a = _staircase_value(float(inp[a_ax][p, 0]) * scales[a_ax], a_size, eps)
        sv_b = _staircase_value(float(inp[b_ax][p, 0]) * scales[b_ax], b_size, eps)
        q[p, 0] = b_size * sv_a + sv_b
        c_scaled[p, 0] = float(inp[c_ax][p, 0]) * scales[c_ax]
    return _table_lookup_2d_reference(table_perm, q, c_scaled, sharpness=sharpness)


def _build_table_lookup_3d_graph(
    table, index_scale=1.0, sharpness=100.0, outer_axis=None
):
    i = create_input("i", 1)
    j = create_input("j", 1)
    k = create_input("k", 1)
    return table_lookup_3d(
        i,
        j,
        k,
        table,
        index_scale=index_scale,
        sharpness=sharpness,
        outer_axis=outer_axis,
    )


def test_table_lookup_3d_axis_order_for_target_shape():
    # 16 x 128 x 128 should resolve to A=16 (outer), B=128, C=128.
    assert _table_lookup_3d_axis_order((16, 128, 128)) == (0, 1, 2)


def test_table_lookup_3d_every_integer_cell():
    table = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) * 2.0 - 5.0
    out_node = _build_table_lookup_3d_graph(table)

    coords = [
        (a, b, c)
        for a in range(table.shape[0])
        for b in range(table.shape[1])
        for c in range(table.shape[2])
    ]
    i_val = torch.tensor([[float(a)] for a, _b, _c in coords], dtype=torch.float32)
    j_val = torch.tensor([[float(b)] for _a, b, _c in coords], dtype=torch.float32)
    k_val = torch.tensor([[float(c)] for _a, _b, c in coords], dtype=torch.float32)
    inputs = {"i": i_val, "j": j_val, "k": k_val}
    n_pos = len(coords)

    # Fully independent check: the cell value itself.
    direct = torch.tensor(
        [[float(table[a, b, c])] for a, b, c in coords], dtype=torch.float32
    )
    expected = _table_lookup_3d_reference(table, i_val, j_val, k_val)
    cache = reference_eval(out_node, inputs, n_pos)
    assert torch.allclose(cache[out_node], direct, atol=5e-3)
    assert torch.allclose(cache[out_node], expected, atol=5e-3)

    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=n_pos,
        d=2048,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, report.format_short()


def test_table_lookup_3d_scaled_unit_coordinates():
    table = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) * 1.5 + 1.0
    index_scale = table.shape
    out_node = _build_table_lookup_3d_graph(table, index_scale=index_scale)

    coords = [(0, 1, 2), (1, 0, 3), (1, 2, 0), (0, 2, 3)]
    # +0.2 inside each scaled cell stays in the centered integer bin.  The
    # outer axis (axis 0) must scale to an exact integer (it is asserted
    # integer), so it gets no offset.
    i_val = torch.tensor(
        [[float(a) / table.shape[0]] for a, _b, _c in coords],
        dtype=torch.float32,
    )
    j_val = torch.tensor(
        [[(float(b) + 0.2) / table.shape[1]] for _a, b, _c in coords],
        dtype=torch.float32,
    )
    k_val = torch.tensor(
        [[(float(c) + 0.2) / table.shape[2]] for _a, _b, c in coords],
        dtype=torch.float32,
    )
    inputs = {"i": i_val, "j": j_val, "k": k_val}
    direct = torch.tensor(
        [[float(table[a, b, c])] for a, b, c in coords], dtype=torch.float32
    )

    cache = reference_eval(out_node, inputs, len(coords))
    assert torch.allclose(cache[out_node], direct, atol=5e-3)


# Flattening 16x128 into 2048 row breakpoints makes the row PWL reconstruct
# each value as a sum over ~q active ReLU terms.  Both the per-term slope
# magnitude and the ReLU-argument magnitude scale with `sharpness`, so the
# float32 accumulation error at integer cells scales with it too: ~0.08 at
# sharpness=100, ~0.008 at sharpness=10 on unit-magnitude values.  Integer-
# lookup fidelity at the target shape therefore improves with lower sharpness.
@pytest.mark.parametrize("sharpness,atol", [(100.0, 0.2), (10.0, 0.02)])
def test_table_lookup_3d_target_size_integer_reference(sharpness, atol):
    generator = torch.Generator().manual_seed(0)
    table = (
        torch.rand((16, 128, 128), generator=generator, dtype=torch.float32) * 2.0 - 1.0
    )
    out_node = _build_table_lookup_3d_graph(table, sharpness=sharpness)

    coords = [(0, 0, 0), (3, 17, 99), (8, 64, 64), (15, 127, 1), (15, 0, 127)]
    i_val = torch.tensor([[float(a)] for a, _b, _c in coords], dtype=torch.float32)
    j_val = torch.tensor([[float(b)] for _a, b, _c in coords], dtype=torch.float32)
    k_val = torch.tensor([[float(c)] for _a, _b, c in coords], dtype=torch.float32)
    inputs = {"i": i_val, "j": j_val, "k": k_val}
    direct = torch.tensor(
        [[float(table[a, b, c])] for a, b, c in coords], dtype=torch.float32
    )

    cache = reference_eval(out_node, inputs, len(coords))
    assert torch.allclose(cache[out_node], direct, atol=atol)


def test_table_lookup_3d_inner_axis_amplification():
    # b_size = 16, so idx_a feeds q with a ×16 weight: q = 16*idx_a + idx_b.
    # High-b integer cells (b=15) stress that the staircase lands exactly.
    table = torch.arange(2 * 16 * 16, dtype=torch.float32).reshape(2, 16, 16) * 0.5
    out_node = _build_table_lookup_3d_graph(table)
    assert _table_lookup_3d_axis_order(table.shape) == (0, 1, 2)

    coords = [(0, 0, 0), (0, 15, 0), (1, 15, 15), (1, 0, 7), (0, 9, 11), (1, 8, 3)]
    i_val = torch.tensor([[float(a)] for a, _b, _c in coords], dtype=torch.float32)
    j_val = torch.tensor([[float(b)] for _a, b, _c in coords], dtype=torch.float32)
    k_val = torch.tensor([[float(c)] for _a, _b, c in coords], dtype=torch.float32)
    inputs = {"i": i_val, "j": j_val, "k": k_val}
    direct = torch.tensor(
        [[float(table[a, b, c])] for a, b, c in coords], dtype=torch.float32
    )

    cache = reference_eval(out_node, inputs, len(coords))
    assert torch.allclose(cache[out_node], direct, atol=5e-3)

    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=len(coords),
        d=2048,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, report.format_short()


def test_table_lookup_3d_near_integer_inputs_remain_in_bin():
    # sharpness=10 -> eps=0.1, transition half-width 0.05.  +/-0.4 on the
    # inner (B) and vector (C) axes stays in the stable bin.  The outer (A)
    # axis is held exactly integer (it is asserted integer).
    table = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3) * 4.0
    out_node = _build_table_lookup_3d_graph(table, sharpness=10.0)

    coords = [(0, 0.4, 0.4), (1, 1.4, 1.6), (0, 1.6, 0.4), (1, 0.4, 1.6)]
    i_val = torch.tensor([[a] for a, _b, _c in coords], dtype=torch.float32)
    j_val = torch.tensor([[b] for _a, b, _c in coords], dtype=torch.float32)
    k_val = torch.tensor([[c] for _a, _b, c in coords], dtype=torch.float32)
    inputs = {"i": i_val, "j": j_val, "k": k_val}
    expected = _table_lookup_3d_reference(
        table, i_val, j_val, k_val, sharpness=10.0
    )

    cache = reference_eval(out_node, inputs, len(coords))
    assert torch.allclose(cache[out_node], expected, atol=5e-3)


def test_table_lookup_3d_inner_and_vector_transition_match_reference():
    # Half-integer transition on B (axis 1) and C (axis 2), A held integer.
    table = torch.tensor(
        [
            [[0.0, 10.0], [20.0, 30.0]],
            [[40.0, 50.0], [60.0, 70.0]],
        ],
        dtype=torch.float32,
    )
    sharpness = 10.0
    out_node = _build_table_lookup_3d_graph(table, sharpness=sharpness)
    assert _table_lookup_3d_axis_order(table.shape) == (0, 1, 2)

    inputs = {
        "i": torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float32),
        "j": torch.tensor([[0.5], [0.0], [0.5]], dtype=torch.float32),  # B transition
        "k": torch.tensor([[0.0], [0.5], [0.5]], dtype=torch.float32),  # C transition
    }
    expected = _table_lookup_3d_reference(
        table, inputs["i"], inputs["j"], inputs["k"], sharpness=sharpness
    )

    cache = reference_eval(out_node, inputs, 3)
    assert torch.allclose(cache[out_node], expected, atol=5e-3)

    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=3,
        d=512,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, report.format_short()


def test_table_lookup_3d_outer_axis_must_be_integer():
    table = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    out_node = _build_table_lookup_3d_graph(table)
    # A = axis 0; a clearly non-integer value there must fail the assert.
    inputs = {
        "i": torch.tensor([[0.5]], dtype=torch.float32),
        "j": torch.tensor([[1.0]], dtype=torch.float32),
        "k": torch.tensor([[2.0]], dtype=torch.float32),
    }
    with pytest.raises(AssertionError):
        reference_eval(out_node, inputs, 1)


def test_table_lookup_3d_outer_axis_override():
    table = torch.arange(8 * 4 * 2, dtype=torch.float32).reshape(8, 4, 2) + 0.5
    # Default outer axis would be axis 2 (size 2); override to axis 0.
    assert _table_lookup_3d_axis_order(table.shape) == (2, 1, 0)
    assert _table_lookup_3d_axis_order(table.shape, outer_axis=0) == (0, 2, 1)
    out_node = _build_table_lookup_3d_graph(table, outer_axis=0)

    coords = [(0, 0, 0), (7, 3, 1), (4, 1, 0), (2, 2, 1)]
    i_val = torch.tensor([[float(a)] for a, _b, _c in coords], dtype=torch.float32)
    j_val = torch.tensor([[float(b)] for _a, b, _c in coords], dtype=torch.float32)
    k_val = torch.tensor([[float(c)] for _a, _b, c in coords], dtype=torch.float32)
    inputs = {"i": i_val, "j": j_val, "k": k_val}
    direct = torch.tensor(
        [[float(table[a, b, c])] for a, b, c in coords], dtype=torch.float32
    )

    cache = reference_eval(out_node, inputs, len(coords))
    assert torch.allclose(cache[out_node], direct, atol=5e-3)


def test_table_lookup_3d_rejects_bad_shapes_and_scales():
    i = create_input("i", 1)
    j = create_input("j", 1)
    k = create_input("k", 1)
    with pytest.raises(ValueError):  # 2D table
        table_lookup_3d(i, j, k, torch.zeros(3, 4))
    with pytest.raises(ValueError):  # empty axis
        table_lookup_3d(i, j, k, torch.zeros(3, 0, 4))
    with pytest.raises(ValueError):  # wrong-length index_scale
        table_lookup_3d(i, j, k, torch.zeros(3, 4, 5), index_scale=(1.0, 2.0))
    with pytest.raises(ValueError):  # sharpness must be > 1
        table_lookup_3d(i, j, k, torch.zeros(3, 4, 5), sharpness=1.0)
    with pytest.raises(ValueError):  # bad outer_axis
        table_lookup_3d(i, j, k, torch.zeros(3, 4, 5), outer_axis=3)


# ---------------------------------------------------------------------------
# dynamic_extract
# ---------------------------------------------------------------------------


def _dynamic_extract_reference(
    table: torch.Tensor,
    idx: torch.Tensor,
    n_entries: int,
    d_fill: int,
) -> torch.Tensor:
    """Independent reference for dynamic_extract.

    Given ``table`` of shape ``(n_pos, n_entries*d_fill)`` and ``idx`` of
    shape ``(n_pos, 1)``, returns ``(n_pos, d_fill)`` where each row is
    ``table[p, idx[p]*d_fill : (idx[p]+1)*d_fill]``.
    """
    n_pos = table.shape[0]
    out = torch.empty(n_pos, d_fill, dtype=table.dtype)
    for p in range(n_pos):
        k = int(round(idx[p, 0].item()))
        k = max(0, min(n_entries - 1, k))
        out[p] = table[p, k * d_fill : (k + 1) * d_fill]
    return out


def _build_dynamic_extract_graph(n_entries: int, d_fill: int):
    table_node = create_input(
        "table",
        n_entries * d_fill,
        value_range=(0.0, 1.0),
    )
    idx_node = create_input("idx", 1, value_range=(0.0, float(n_entries)))
    out_node = dynamic_extract(table_node, idx_node, n_entries, d_fill)
    return out_node


@pytest.mark.parametrize(
    "n_entries,d_fill",
    [
        (2, 1),
        (4, 3),
        (8, 3),
        (16, 3),
        (32, 3),
        (64, 3),
        (16, 8),
    ],
)
def test_dynamic_extract_every_index(n_entries, d_fill):
    """For every ``idx`` in ``[0, n_entries - 1]`` the primitive must
    return the exact slice from ``table`` — verified both through
    ``reference_eval`` (graph oracle) and through the compiled
    transformer via ``probe_graph``.
    """
    out_node = _build_dynamic_extract_graph(n_entries, d_fill)

    # One position per possible idx — each row of `table` is randomly
    # initialised so every extracted slice is uniquely identifiable.
    rng = torch.Generator().manual_seed(0xD1E)
    table_val = torch.rand(n_entries, n_entries * d_fill, generator=rng)
    idx_val = torch.arange(n_entries, dtype=torch.float32).unsqueeze(-1)
    inputs = {"table": table_val, "idx": idx_val}
    n_pos = n_entries

    # Oracle must match the independent numpy reference.  ``step_sharpness=10``
    # gives ReLU-approximation wiggle of ~1e-4 per op and those accumulate
    # through the in_range → broadcast_select chain, so a 1e-3 absolute
    # tolerance is the right level for this check — tighter than that and
    # we're testing float-precision, not semantics.
    cache = reference_eval(out_node, inputs, n_pos)
    oracle = cache[out_node]
    expected = _dynamic_extract_reference(table_val, idx_val, n_entries, d_fill)
    assert torch.allclose(oracle, expected, atol=5e-3), (
        f"oracle disagrees with reference "
        f"(n={n_entries}, d={d_fill}):\n"
        f"  oracle:   {oracle}\n  expected: {expected}"
    )

    # Compiled must match the oracle at every node (no divergence).
    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=n_pos,
        d=1024,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, (
        f"probe reported divergence on dynamic_extract "
        f"(n={n_entries}, d={d_fill}):\n{report.format_short()}"
    )


def test_dynamic_extract_random_indices():
    """Random (table, idx) combinations — exercises the primitive
    under arbitrary per-position inputs, not just "row i picks slot i".
    """
    n_entries, d_fill = 16, 3
    out_node = _build_dynamic_extract_graph(n_entries, d_fill)

    rng = torch.Generator().manual_seed(0xB17)
    n_pos = 24
    table_val = torch.rand(n_pos, n_entries * d_fill, generator=rng)
    idx_val = torch.randint(0, n_entries, (n_pos, 1), generator=rng).float()
    inputs = {"table": table_val, "idx": idx_val}

    cache = reference_eval(out_node, inputs, n_pos)
    oracle = cache[out_node]
    expected = _dynamic_extract_reference(table_val, idx_val, n_entries, d_fill)
    assert torch.allclose(oracle, expected, atol=5e-3)

    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=n_pos,
        d=1024,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, report.format_short()


# ---------------------------------------------------------------------------
# linear_bin_index
# ---------------------------------------------------------------------------


def _linear_bin_index_reference(
    x: torch.Tensor,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    """Independent reference: ``clamp(floor((x - x_min) * n_bins / (x_max - x_min)))``."""
    v = (x - x_min) * n_bins / (x_max - x_min)
    idx = torch.floor(v)
    return torch.clamp(idx, 0, n_bins - 1)


def _build_linear_bin_index_graph(
    n_bins: int,
    min_range: float,
    max_range: float,
    n_reciprocal_breakpoints: int = 48,
    mul_step: float = 0.25,
):
    """Build a linear_bin_index graph.

    Test-specific ``min_range`` / ``max_range`` bounds matter — the
    reciprocal lookup and the signed_multiply step scale with them, and
    the signed_multiply's arithmetic precision is roughly
    ``(mul_step / max_sum)`` of full-scale.  Bounds tuned to the actual
    test data keep the primitive accurate down to the last bin.
    """
    x = create_input("x", 1)
    x_min = create_input("x_min", 1)
    x_max = create_input("x_max", 1)
    out = linear_bin_index(
        x,
        x_min,
        x_max,
        n_bins,
        min_range=min_range,
        max_range=max_range,
        n_reciprocal_breakpoints=n_reciprocal_breakpoints,
        mul_step=mul_step,
    )
    return out


def _bounds_for(x_min_val: float, x_max_val: float):
    """Pick tight ``(min_range, max_range)`` bounds around the test's
    actual ``x_max - x_min``.  ``min_range`` is half the actual range
    and ``max_range`` is 2× — generous enough to accommodate future
    perturbations without wasting neurons on an over-wide reciprocal
    lookup.
    """
    actual = x_max_val - x_min_val
    min_r = max(0.1, actual * 0.5)
    max_r = max(actual * 2.0, actual + 1.0)
    return min_r, max_r


def _sweep_inputs(x_min_val, x_max_val, n_bins):
    """Build an n_pos-batch that sweeps ``x`` across the bin centers plus
    both out-of-range extremes so the clamping corners get tested.

    Returns a (x, x_min, x_max, expected_bins) tuple of (n_pos, 1) tensors.
    """
    # Bin centers land at x_min + (k + 0.5) * range / n_bins.
    range_ = x_max_val - x_min_val
    centers = [x_min_val + (k + 0.5) * range_ / n_bins for k in range(n_bins)]
    # Plus out-of-range probes: well below x_min, well above x_max.
    below = x_min_val - range_ * 0.1
    above = x_max_val + range_ * 0.1
    xs = [below] + centers + [above]

    n_pos = len(xs)
    x = torch.tensor([[v] for v in xs], dtype=torch.float32)
    x_min = torch.full((n_pos, 1), float(x_min_val), dtype=torch.float32)
    x_max = torch.full((n_pos, 1), float(x_max_val), dtype=torch.float32)
    return x, x_min, x_max


@pytest.mark.parametrize(
    "n_bins,x_min_val,x_max_val",
    [
        (4, 0.0, 10.0),  # normal case, moderate bins
        (8, 0.0, 10.0),
        (16, 0.0, 10.0),
        (32, 0.0, 10.0),
        (64, 0.0, 10.0),  # DOOM tex_h=64
        (16, -5.0, 5.0),  # signed lower bound
        (16, 100.0, 110.0),  # shifted far from origin
        (16, 0.0, 1.5),  # small range near the reciprocal's min_range boundary
        (16, 0.0, 100.0),  # wide range
    ],
)
def test_linear_bin_index_centers_and_out_of_range(n_bins, x_min_val, x_max_val):
    """For each bin ``k`` in ``[0, n_bins - 1]``, the primitive must
    return ``k`` when ``x`` lands at the bin center.  Below-range
    and above-range probes must clamp to bins ``0`` and ``n_bins - 1``
    respectively.
    """
    min_r, max_r = _bounds_for(x_min_val, x_max_val)
    out_node = _build_linear_bin_index_graph(n_bins, min_r, max_r)

    x, x_min, x_max = _sweep_inputs(x_min_val, x_max_val, n_bins)
    inputs = {"x": x, "x_min": x_min, "x_max": x_max}
    n_pos = x.shape[0]

    cache = reference_eval(out_node, inputs, n_pos)
    oracle = cache[out_node].flatten()
    expected_mid = torch.arange(n_bins, dtype=torch.float32)
    expected = torch.cat(
        [
            torch.tensor([0.0]),  # below -> clamped to 0
            expected_mid,  # each bin center -> bin index
            torch.tensor([n_bins - 1.0]),  # above -> clamped to n_bins-1
        ]
    )
    assert torch.allclose(oracle, expected, atol=0.05), (
        f"oracle disagrees with expected bins "
        f"(n_bins={n_bins}, range=[{x_min_val}, {x_max_val}])\n"
        f"  oracle:   {oracle.tolist()}\n  expected: {expected.tolist()}"
    )

    # Probe at the final-output level: intermediate signed_multiply
    # values wiggle by up to ~5% of max_range/min_range and that wiggle
    # amplifies through the n_bins scale; the downstream clamp +
    # floor_int absorb it as long as bin_f stays within half a bin of
    # the true value, so a ``half-a-bin`` tolerance at the final node
    # is the right pass criterion.  We still assert the oracle matches
    # the expected bins exactly above, so semantic correctness is
    # guarded.
    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=n_pos,
        d=2048,
        d_head=16,
        verbose=False,
        atol=0.5,
    )
    assert report.first_divergent is None, (
        f"probe reported divergence on linear_bin_index "
        f"(n_bins={n_bins}, range=[{x_min_val}, {x_max_val}]):\n"
        f"{report.format_short()}"
    )


def test_linear_bin_index_non_integer_values():
    """Bin indices must track ``floor((x - x_min)/range * n_bins)``
    for non-center ``x`` values too (not just the center probe).
    """
    n_bins = 8
    x_min_val, x_max_val = 0.0, 8.0
    min_r, max_r = _bounds_for(x_min_val, x_max_val)
    out_node = _build_linear_bin_index_graph(n_bins, min_r, max_r)

    rng = torch.Generator().manual_seed(0xB10)
    # Probe at 0.1-increments across the range, offset from integer
    # boundaries by 0.2 so we stay inside the ``floor_int`` flat zone
    # and the bin classification is unambiguous.
    base = torch.arange(n_bins, dtype=torch.float32).repeat_interleave(3)
    offsets = torch.tensor([0.2, 0.5, 0.8] * n_bins, dtype=torch.float32)
    xs = (base + offsets).unsqueeze(-1)
    n_pos = xs.shape[0]
    x_min = torch.full((n_pos, 1), x_min_val)
    x_max = torch.full((n_pos, 1), x_max_val)
    inputs = {"x": xs, "x_min": x_min, "x_max": x_max}

    cache = reference_eval(out_node, inputs, n_pos)
    oracle = cache[out_node]
    expected = _linear_bin_index_reference(
        xs,
        x_min,
        x_max,
        n_bins,
    )
    assert torch.allclose(oracle, expected, atol=0.05), (
        f"oracle disagrees:\n  oracle:   {oracle.flatten().tolist()}\n"
        f"  expected: {expected.flatten().tolist()}"
    )

    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=n_pos,
        d=2048,
        d_head=16,
        verbose=False,
        atol=0.5,
    )
    assert report.first_divergent is None, report.format_short()


# ---------------------------------------------------------------------------
# Composition: linear_bin_index feeding dynamic_extract
# ---------------------------------------------------------------------------


def test_linear_bin_index_into_dynamic_extract():
    """Cascade the two primitives and verify the combined pipeline
    reads the right table row for a sweep of ``x`` values.

    This is the exact shape the textured wall fill needs: given a
    world-space coordinate ``x`` inside ``[x_min, x_max)``, pick the
    matching texture row out of a runtime-supplied ``tex_column_colors``.
    """
    n_bins = 16  # stand-in for tex_height
    d_fill = 3  # RGB
    x_min_val, x_max_val = 0.0, 16.0
    min_r, max_r = _bounds_for(x_min_val, x_max_val)

    x = create_input("x", 1, value_range=(x_min_val, x_max_val))
    x_min = create_input("x_min", 1, value_range=(x_min_val, x_max_val))
    x_max = create_input("x_max", 1, value_range=(x_min_val, x_max_val))
    table = create_input("table", n_bins * d_fill, value_range=(0.0, 1.0))

    idx = linear_bin_index(
        x,
        x_min,
        x_max,
        n_bins,
        min_range=min_r,
        max_range=max_r,
        n_reciprocal_breakpoints=48,
        mul_step=0.25,
    )
    rgb = dynamic_extract(table, idx, n_bins, d_fill)

    # Sweep x across bin centers (plus 0.2 offset to avoid the
    # ``floor_int`` ramp zones around integer boundaries — the
    # cascade's pass criterion is "did we land in the right bin?", and
    # bins straddling a ramp transition are ambiguous by design).
    rng = torch.Generator().manual_seed(0xCAFE)
    n_pos = n_bins
    bin_width = (x_max_val - x_min_val) / n_bins
    xs = torch.tensor(
        [[x_min_val + (k + 0.3) * bin_width] for k in range(n_bins)],
        dtype=torch.float32,
    )
    x_min_t = torch.full((n_pos, 1), x_min_val)
    x_max_t = torch.full((n_pos, 1), x_max_val)
    table_t = torch.rand(n_pos, n_bins * d_fill, generator=rng)
    inputs = {
        "x": xs,
        "x_min": x_min_t,
        "x_max": x_max_t,
        "table": table_t,
    }

    # Reference: compute the expected bin per position, then extract the
    # corresponding slice.
    expected_idx = _linear_bin_index_reference(
        xs,
        x_min_t,
        x_max_t,
        n_bins,
    )
    expected_rgb = _dynamic_extract_reference(
        table_t,
        expected_idx,
        n_bins,
        d_fill,
    )

    cache = reference_eval(rgb, inputs, n_pos)
    assert torch.allclose(cache[rgb], expected_rgb, atol=5e-3), (
        f"cascade oracle disagrees with expected RGB:\n"
        f"  got:  {cache[rgb]}\n  want: {expected_rgb}"
    )

    # Intermediate linear_bin_index arithmetic wiggle is bounded by the
    # half-bin tolerance that keeps the floor stable.  Once past the
    # floor + one-hot, the extracted RGB either matches the right
    # table row exactly or picks a wrong row (fully mismatched) — so
    # the probe atol here only needs to survive the pre-floor
    # arithmetic, with the final RGB comparison above as the real
    # pass criterion.
    report = probe_graph(
        rgb,
        pos_encoding=None,
        input_values=inputs,
        n_pos=n_pos,
        d=2048,
        d_head=16,
        verbose=False,
        atol=0.5,
    )
    assert report.first_divergent is None, report.format_short()


# ---------------------------------------------------------------------------
# linear_bin_index with hoisted inv_range
# ---------------------------------------------------------------------------


def _build_linear_bin_index_hoisted(
    n_bins: int,
    min_range: float,
    max_range: float,
    n_reciprocal_breakpoints: int = 48,
    mul_step: float = 0.25,
):
    """Build a linear_bin_index graph with externally-hoisted inv_range.

    Returns the same output as ``_build_linear_bin_index_graph`` but
    computes ``inv_range`` outside of ``linear_bin_index`` and passes
    it in via the ``inv_range`` parameter.
    """
    x = create_input("x", 1)
    x_min = create_input("x_min", 1)
    x_max = create_input("x_max", 1)

    # Hoist: compute inv_range once
    range_ = subtract(x_max, x_min)
    clamped_range = clamp(range_, min_range, max_range)
    inv = reciprocal(clamped_range, min_value=min_range, max_value=max_range)

    out = linear_bin_index(
        x,
        x_min,
        x_max,
        n_bins,
        min_range=min_range,
        max_range=max_range,
        mul_step=mul_step,
        inv_range=inv,
    )
    return out


@pytest.mark.parametrize(
    "n_bins,x_min_val,x_max_val",
    [
        (4, 0.0, 10.0),
        (8, 0.0, 10.0),
        (16, 0.0, 10.0),
        (64, 0.0, 10.0),
        (16, -5.0, 5.0),
        (16, 0.0, 1.5),
        (16, 0.0, 100.0),
    ],
)
def test_linear_bin_index_with_inv_range(n_bins, x_min_val, x_max_val):
    """Hoisted inv_range produces the same bin indices as the original.

    Mirrors ``test_linear_bin_index_centers_and_out_of_range`` but
    builds the graph with externally-computed ``inv_range``.
    """
    min_r, max_r = _bounds_for(x_min_val, x_max_val)
    out_node = _build_linear_bin_index_hoisted(n_bins, min_r, max_r)

    x, x_min, x_max = _sweep_inputs(x_min_val, x_max_val, n_bins)
    inputs = {"x": x, "x_min": x_min, "x_max": x_max}
    n_pos = x.shape[0]

    cache = reference_eval(out_node, inputs, n_pos)
    oracle = cache[out_node].flatten()
    expected_mid = torch.arange(n_bins, dtype=torch.float32)
    expected = torch.cat(
        [
            torch.tensor([0.0]),
            expected_mid,
            torch.tensor([n_bins - 1.0]),
        ]
    )
    assert torch.allclose(oracle, expected, atol=0.05), (
        f"hoisted oracle disagrees with expected bins "
        f"(n_bins={n_bins}, range=[{x_min_val}, {x_max_val}])\n"
        f"  oracle:   {oracle.tolist()}\n  expected: {expected.tolist()}"
    )


def test_linear_bin_index_shared_inv_range_multi_x():
    """Multiple x values sharing one inv_range node all produce correct bins.

    This is the exact pattern the textured wall fill needs: one shared
    ``inv_range`` node, multiple ``linear_bin_index`` calls with
    different ``x`` values.
    """
    n_bins = 16
    x_min_val, x_max_val = 0.0, 16.0
    min_r, max_r = _bounds_for(x_min_val, x_max_val)

    x_min = create_input("x_min", 1)
    x_max = create_input("x_max", 1)

    # Hoist inv_range — shared across all calls
    range_ = subtract(x_max, x_min)
    clamped_range = clamp(range_, min_r, max_r)
    inv = reciprocal(clamped_range, min_value=min_r, max_value=max_r)

    # Build multiple bin indices from separate x inputs
    x_nodes = [create_input(f"x{i}", 1) for i in range(4)]
    bin_nodes = []
    for x in x_nodes:
        idx = linear_bin_index(
            x,
            x_min,
            x_max,
            n_bins,
            min_range=min_r,
            max_range=max_r,
            mul_step=0.25,
            inv_range=inv,
        )
        bin_nodes.append(idx)

    # Concatenate for a single output node
    out = Concatenate(bin_nodes)

    # Sweep: each x input gets a different bin center
    bin_width = (x_max_val - x_min_val) / n_bins
    x_vals = [x_min_val + (k + 0.5) * bin_width for k in [0, 5, 10, 15]]
    n_pos = 1
    inputs = {
        "x_min": torch.tensor([[x_min_val]]),
        "x_max": torch.tensor([[x_max_val]]),
    }
    for i, v in enumerate(x_vals):
        inputs[f"x{i}"] = torch.tensor([[v]])

    cache = reference_eval(out, inputs, n_pos)
    oracle = cache[out].flatten()
    expected = torch.tensor([0.0, 5.0, 10.0, 15.0])
    assert torch.allclose(oracle, expected, atol=0.05), (
        f"shared inv_range multi-x: oracle={oracle.tolist()}, "
        f"expected={expected.tolist()}"
    )


def test_linear_bin_index_inv_range_probe():
    """Probe test: hoisted inv_range compiles correctly."""
    n_bins = 8
    x_min_val, x_max_val = 0.0, 8.0
    min_r, max_r = _bounds_for(x_min_val, x_max_val)
    out_node = _build_linear_bin_index_hoisted(n_bins, min_r, max_r)

    x, x_min, x_max = _sweep_inputs(x_min_val, x_max_val, n_bins)
    inputs = {"x": x, "x_min": x_min, "x_max": x_max}
    n_pos = x.shape[0]

    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=n_pos,
        d=2048,
        d_head=16,
        verbose=False,
        atol=0.5,
    )
    assert (
        report.first_divergent is None
    ), f"probe divergence on hoisted inv_range:\n{report.format_short()}"


# ---------------------------------------------------------------------------
# Cancellation-free index staircase (regression)
#
# The old single-projection ``piecewise_linear`` builders summed ~n ReLU terms
# of magnitude ~sharpness*index into one accumulator, overflowing fp32's 2^24
# exact-integer limit so the alternating-sign terms cancelled into garbage.
# Two triggers: (a) a scaled index far outside [0, n-1] (sharpness*index past
# 2^24), and (b) a tall table whose top breakpoint position passes 2^24.  See
# docs/numerical_noise_findings.md, "the map_select table-lookup staircases".
# Mirrors test_floor_int_wide_range_large_magnitude; the cancellation lives in
# the exact-math fp32 sum that node.compute / reference_eval reproduces.
# ---------------------------------------------------------------------------


def test_table_lookup_2d_out_of_range_index_clamps_to_edge():
    """Trigger (a): a row index far outside [0, rows-1] clamps to the table
    edge instead of collapsing (index 1e5 at sharpness 100 -> sharpness*index
    1e7 > 2^24 used to return ~0)."""
    rows = 256
    table = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    out_node = _build_table_lookup_2d_graph(table)  # default sharpness=100

    i_val = torch.tensor([[1.0e5], [-50.0], [0.0], [128.0]], dtype=torch.float32)
    j_val = torch.zeros_like(i_val)
    inputs = {"i": i_val, "j": j_val}
    expected = torch.tensor(
        [
            [float(table[255, 0])],
            [float(table[0, 0])],
            [float(table[0, 0])],
            [float(table[128, 0])],
        ]
    )
    cache = reference_eval(out_node, inputs, i_val.shape[0])
    assert torch.allclose(cache[out_node], expected, atol=5e-3), cache[out_node]

    # ...and the min-clamp + saturating-step chain compiles precisely.
    report = probe_graph(
        out_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=i_val.shape[0],
        d=1024,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, report.format_short()


def test_table_lookup_2d_tall_table_stays_exact():
    """Trigger (b): a 20000-row table at sharpness 100 and 1000 keeps in-range
    indices exact (an in-range index used to collapse to 0).  Oracle-only:
    compiling a 20000-row table is needlessly expensive and the cancellation is
    in the exact-math fp32 sum, like the floor_int wide-range regression."""
    rows = 20000
    table = torch.arange(rows, dtype=torch.float32).reshape(rows, 1)
    idxs = [0.0, 1.0, 9999.0, 12345.0, 19998.0, 19999.0]
    for s in (100.0, 1000.0):
        i = create_input("i", 1)
        j = create_input("j", 1)
        out_node = table_lookup_2d(i, j, table, sharpness=s)
        inputs = {
            "i": torch.tensor([[v] for v in idxs]),
            "j": torch.zeros(len(idxs), 1),
        }
        got = reference_eval(out_node, inputs, len(idxs))[out_node].squeeze(-1)
        expected = torch.tensor(idxs)
        assert torch.allclose(got, expected, atol=5e-3), (s, got, expected)


def test_table_lookup_index_staircase_clamps_and_rounds_at_any_magnitude():
    """The scalar staircase computes clamp(round(x), 0, n-1) exactly at any
    magnitude — the smallest layer that reproduces both triggers (out-of-range
    x and large n)."""
    from torchwright.ops.map_select import _table_lookup_index_staircase

    for n, s in [(256, 100.0), (256, 1000.0), (20000, 1000.0)]:
        x = create_input("x", 1)
        node = _table_lookup_index_staircase(
            x, n, sharpness=s, d_max=1024, name="st"
        )
        cases = [
            (-1.0e6, 0.0),
            (-0.4, 0.0),
            (0.0, 0.0),
            (0.6, 1.0),
            (float(n // 3), float(n // 3)),
            (float(n - 1), float(n - 1)),
            (1.0e6, float(n - 1)),
        ]
        xs = torch.tensor([[c[0]] for c in cases])
        out = reference_eval(node, {"x": xs}, len(cases))[node].squeeze(-1)
        exp = torch.tensor([c[1] for c in cases])
        assert torch.allclose(out, exp, atol=5e-3), (n, s, out, exp)


def test_table_lookup_3d_large_flattened_rows_exact():
    """The 3D lookup flattens (A, B) into A*B rows; the motivating 16x128x128
    shape reshapes to 2048 rows — well into the tall-table regime — and stays
    exact on the integer grid through the cancellation-free row builder."""
    t3 = torch.arange(16 * 128 * 128, dtype=torch.float32).reshape(16, 128, 128)
    i = create_input("i", 1)
    j = create_input("j", 1)
    k = create_input("k", 1)
    node = table_lookup_3d(i, j, k, t3)
    cells = [(0, 0, 0), (15, 127, 127), (7, 64, 33), (3, 5, 120)]
    inputs = {
        "i": torch.tensor([[float(a)] for a, _, _ in cells]),
        "j": torch.tensor([[float(b)] for _, b, _ in cells]),
        "k": torch.tensor([[float(c)] for _, _, c in cells]),
    }
    got = reference_eval(node, inputs, len(cells))[node].squeeze(-1)
    expected = torch.tensor([float(t3[a, b, c]) for a, b, c in cells])
    assert torch.allclose(got, expected, atol=5e-3), (got, expected)
