"""swiglu table_lookup_2d: the gated column stage.

Spec: docs/ops_plain_english.md (table_lookup_2d entry); the telescoping
identity and the in-band blend claims are pinned in
tests/docs/test_swish_constants.py (test_table_lookup_2d_telescoping).
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph.ffn import FFN
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.swiglu import table_lookup_2d


def _count_ffns(node):
    """Number of distinct FFN nodes reachable from ``node``."""
    seen, stack, n = set(), [node], 0
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, FFN):
            n += 1
        stack.extend(cur.inputs)
    return n


D = 64
D_HEAD = 8

_TABLE = torch.tensor(
    [
        [1.0, -2.0, 3.0, 40.0],
        [5.0, 6.0, -7.0, 8.0],
        [-9.0, 10.0, 11.0, 12.0],
    ]
)


def _build(table=_TABLE, **kw):
    i = create_input("i", 1, value_range=(-1.0, 10.0))
    j = create_input("j", 1, value_range=(-1.0, 10.0))
    return table_lookup_2d(i, j, table, **kw), table


def test_integer_grid_exact():
    out, table = _build()
    cases = [(r, c) for r in range(3) for c in range(4)]
    ii = torch.tensor([[float(r)] for r, _ in cases])
    jj = torch.tensor([[float(c)] for _, c in cases])
    val = out.compute(len(cases), {"i": ii, "j": jj})
    ref = torch.tensor([[table[r, c]] for r, c in cases])
    assert torch.allclose(val, ref, rtol=0.0, atol=1e-3), (val - ref).flatten()


def test_out_of_range_clamps_to_edges():
    out, table = _build()
    ii = torch.tensor([[-1.0], [9.0], [1.0]])
    jj = torch.tensor([[2.0], [1.0], [9.0]])
    val = out.compute(3, {"i": ii, "j": jj})
    ref = torch.tensor([[table[0, 2]], [table[2, 1]], [table[1, 3]]])
    assert torch.allclose(val, ref, rtol=0.0, atol=1e-3)


def test_in_band_linear_blend_is_contract():
    """A band on one axis gives the clean two-entry linear blend — the
    coefficient is the blend fraction itself (hinge(α) = α exactly for
    α ≥ 17/scale); this was a disclaimer on relu and is a contract now."""
    out, table = _build()
    # j halfway across the 1.5 boundary ramp (sharpness 100 → ramp
    # [1.495, 1.505]); α = 0.5 blend of columns 1 and 2 at row 0.
    ii = torch.tensor([[0.0], [0.0]])
    jj = torch.tensor([[1.5], [1.4975]])
    val = out.compute(2, {"i": ii, "j": jj})
    assert val[0].item() == pytest.approx(
        0.5 * (table[0, 1] + table[0, 2]).item(), abs=1e-3
    )
    assert val[1].item() == pytest.approx(
        (0.75 * table[0, 1] + 0.25 * table[0, 2]).item(), abs=1e-3
    )


def test_index_scale_folds_into_gates():
    out, table = _build(index_scale=(0.5, 2.0))
    # i raw 4.0 → scaled 2.0; j raw 1.0 → scaled 2.0
    val = out.compute(1, {"i": torch.tensor([[4.0]]), "j": torch.tensor([[1.0]])})
    assert val.item() == pytest.approx(table[2, 2].item(), abs=1e-3)


def test_single_column_and_single_row():
    t_col = torch.tensor([[3.0], [7.0], [-1.0]])
    out, _ = _build(table=t_col)
    val = out.compute(2, {"i": torch.tensor([[1.0], [2.0]]), "j": torch.zeros(2, 1)})
    assert torch.allclose(val, torch.tensor([[7.0], [-1.0]]), atol=1e-3)

    t_row = torch.tensor([[3.0, 7.0, -1.0]])
    out2, _ = _build(table=t_row)
    val2 = out2.compute(2, {"i": torch.zeros(2, 1), "j": torch.tensor([[0.0], [2.0]])})
    assert torch.allclose(val2, torch.tensor([[3.0], [-1.0]]), atol=1e-3)


def test_chunked_axes_match_table():
    """Axes longer than the min_d_hidden//2 boundary cap split into
    summed chunks on both stages; values stay exact on the grid."""
    g = torch.Generator().manual_seed(83)
    tall = torch.rand(600, 3, generator=g) * 20.0 - 10.0
    i = create_input("i", 1, value_range=(0.0, 599.0))
    j = create_input("j", 1, value_range=(0.0, 2.0))
    out = table_lookup_2d(i, j, tall)
    ii = torch.tensor([[0.0], [511.0], [512.0], [599.0]])
    jj = torch.tensor([[0.0], [1.0], [2.0], [1.0]])
    val = out.compute(4, {"i": ii, "j": jj})
    ref = torch.tensor([[tall[0, 0]], [tall[511, 1]], [tall[512, 2]], [tall[599, 1]]])
    assert torch.allclose(val, ref, rtol=0.0, atol=1e-2), (val - ref).flatten()

    wide = torch.rand(3, 600, generator=g) * 20.0 - 10.0
    i2 = create_input("i", 1, value_range=(0.0, 2.0))
    j2 = create_input("j", 1, value_range=(0.0, 599.0))
    out2 = table_lookup_2d(i2, j2, wide)
    val2 = out2.compute(4, {"i": jj.clamp(max=2.0), "j": ii})
    ref2 = torch.tensor([[wide[0, 0]], [wide[1, 511]], [wide[2, 512]], [wide[1, 599]]])
    assert torch.allclose(val2, ref2, rtol=0.0, atol=1e-2), (val2 - ref2).flatten()


def test_compiles_clean():
    out, table = _build()
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    inputs = {
        "i": torch.tensor([[0.0], [1.0], [2.0]]),
        "j": torch.tensor([[3.0], [0.0], [2.0]]),
    }
    report = probe_compiled(compiled, out, inputs, 3, atol=1e-2)
    assert report.first_divergent is None, report.format_short()


# --- Range-aware clamp skip -------------------------------------------------
#
# ``_upper_clamp`` caps ``gate_mult·index`` at ``rows-1`` / ``cols-1`` in one
# FFN sublayer.  When the input's static range proves ``0 ≤ gate_mult·index ≤
# top`` the cap is the identity, so the clamp FFN is skipped and the scaling
# is folded into the staircase gate instead.


def _build_ranged(i_range, j_range, table=_TABLE, **kw):
    i = create_input("i", 1, value_range=i_range)
    j = create_input("j", 1, value_range=j_range)
    return table_lookup_2d(i, j, table, **kw), table


def test_clamp_skip_matches_clamped_form():
    """In-range: the skipped build equals the clamped build value-for-value
    on a dense grid — clamping an in-bounds index is the identity."""
    # _TABLE is 3x4 -> top_i=2, top_j=3.
    skip, table = _build_ranged((0.0, 2.0), (0.0, 3.0))
    keep, _ = _build_ranged((-1e4, 1e4), (-1e4, 1e4))
    ii = torch.linspace(0.0, 2.0, 37)
    jj = torch.linspace(0.0, 3.0, 41)
    grid_i, grid_j = torch.meshgrid(ii, jj, indexing="ij")
    gi = grid_i.reshape(-1, 1)
    gj = grid_j.reshape(-1, 1)
    n = gi.shape[0]
    vs = skip.compute(n, {"i": gi, "j": gj})
    vk = keep.compute(n, {"i": gi, "j": gj})
    assert torch.equal(vs, vk), (vs - vk).abs().max().item()


def test_clamp_skip_drops_one_ffn_per_axis():
    """Structural: a range that proves safety on both axes builds two fewer
    FFNs (the two clamps) than the range-free build of the same lookup."""
    keep, _ = _build_ranged((-1e4, 1e4), (-1e4, 1e4))
    skip, _ = _build_ranged((0.0, 2.0), (0.0, 3.0))
    assert _count_ffns(keep) - _count_ffns(skip) == 2


def test_clamp_kept_when_range_violates_bound():
    """Negative: a range crossing either bound (or the wide default, which
    stands in for "no useful range") keeps that axis's clamp.  The untested
    axis is left wide so the FFN count isolates the tested one."""
    keep, _ = _build_ranged((-1e4, 1e4), (-1e4, 1e4))
    base = _count_ffns(keep)
    # i upper bound exceeded (2.5 > top_i=2); j left wide.
    over_i, _ = _build_ranged((0.0, 2.5), (-1e4, 1e4))
    # j lower bound below 0; i left wide.
    under_j, _ = _build_ranged((-1e4, 1e4), (-0.5, 3.0))
    assert _count_ffns(over_i) == base
    assert _count_ffns(under_j) == base


def test_clamp_skip_is_per_axis():
    """One axis in-range, the other not: exactly one clamp is skipped."""
    keep, _ = _build_ranged((-1e4, 1e4), (-1e4, 1e4))
    base = _count_ffns(keep)
    i_only, _ = _build_ranged((0.0, 2.0), (-0.5, 3.0))  # skip i, keep j
    j_only, _ = _build_ranged((-0.5, 2.0), (0.0, 3.0))  # keep i, skip j
    assert _count_ffns(i_only) == base - 1
    assert _count_ffns(j_only) == base - 1
