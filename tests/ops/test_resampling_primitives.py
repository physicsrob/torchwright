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

import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.map_select import dynamic_extract, table_lookup_2d

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
        input_values=inputs,
        n_pos=1,
        d=256,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, report.format_short()


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
        input_values=inputs,
        n_pos=n_pos,
        d=1024,
        d_head=16,
        verbose=False,
        atol=5e-3,
    )
    assert report.first_divergent is None, report.format_short()


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
    1e7 > 2^24 used to return ~0).
    """
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
    in the exact-math fp32 sum, like the floor_int wide-range regression.
    """
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
