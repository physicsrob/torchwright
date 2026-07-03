"""swiglu compare, bool compositions, and equals_vector.

Spec: docs/ops_plain_english.md (compare, equals_vector entries; the bool
ops inherit compare's).  The pinned facts these tests lean on — contract
points bit-exact at scale=100, bend overshoot ≤ swish_dip/scale·|T−F|,
equals_vector's low-side dip 2·swish_dip·speed/scale — live in
tests/docs/test_swish_constants.py.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN
from torchwright.graph.affine_rules import _compare_semantic_bound
from torchwright.ops.const import scale, step_sharpness, swish_dip
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.swiglu import (
    bool_all_true,
    bool_any_true,
    bool_not,
    compare,
    equals_vector,
)

D = 64
D_HEAD = 8


def _unwrap(node):
    """Peel Assert/DebugWatch wrappers to reach the FFN."""
    while not isinstance(node, FFN):
        node = node.inputs[0]
    return node


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_structure():
    x = create_input("x", 1, value_range=(-80.0, 80.0))
    out = compare(x, thresh=0.0)
    ffn = _unwrap(out)
    assert ffn.activation == "swish"
    assert ffn.is_degenerate  # 2 degenerate lanes — no up projection
    assert ffn.n_lanes == 2


def test_compare_contract_points_and_false_side_bit_exact():
    """The robustly bit-exact cases: at thresh and on the whole false
    side beyond the fillet, every lane contribution is exactly ±0.0
    (Swish(0) = 0; σ underflows to exactly 0), so the output IS the
    out_bias; at the true contract point the live lane is exactly scale
    and `scale·fl((T−F)/scale)` rounds to T−F (fl(0.02)'s relative
    representation error, 2.2e-8, is under fp32's half-ulp 6e-8)."""
    x = create_input("x", 1, value_range=(-80.0, 80.0))
    out = compare(x, thresh=0.0)  # ramp width 1/step_sharpness = 0.1
    s = step_sharpness
    xs = torch.tensor([0.0, 1.0 / s, -0.5, -80.0]).unsqueeze(1)
    val = out.compute(4, {"x": xs})
    expected = torch.tensor([-1.0, 1.0, -1.0, -1.0]).unsqueeze(1)
    assert torch.equal(val, expected)


def test_compare_true_side_far_field_ulp_class():
    """True-side outputs past the fillet carry fp32 product rounding at
    the lane-contribution magnitude s·(x−thresh)·|T−F| — the same
    far-field class as the ReLU machine (kernel-dependent: FMA vs
    per-product rounding), NOT bit-exactness. Bounded by a few ulps of
    the contribution."""
    x = create_input("x", 1, value_range=(-80.0, 80.0))
    out = compare(x, thresh=0.0)
    s = step_sharpness
    xs = torch.tensor([0.5, 1.0, 7.3, 80.0, (1.0 + 17.0 / scale) / s]).unsqueeze(1)
    val = out.compute(5, {"x": xs})
    # Worst kernel (gate-FMA + per-product out-proj) leaves 4 half-ulp
    # roundings at the contribution magnitude C = s·|x−thresh|·|T−F|,
    # i.e. ≤ 2 ulps of C; the ×6 is ~3x safety over that (binade
    # position + GPU FP variation), floored at 6 output ulps.
    budget = torch.clamp(6 * 1.2e-7 * s * xs.abs() * 2.0, min=6 * 1.2e-7)
    assert ((val - 1.0).abs() <= budget).all(), (val - 1.0).flatten()


def test_compare_bend_overshoot_bounded():
    """Inputs inside the fillets overshoot the levels by at most
    swish_dip/scale·|T−F|, and the value-range assert carries exactly
    that slack."""
    x = create_input("x", 1, value_range=(-80.0, 80.0))
    out = compare(x, thresh=0.0)
    s = step_sharpness
    # Sweep both fillet zones densely.
    z = torch.cat([torch.linspace(-0.2, 0.0, 2001), torch.linspace(1.0, 1.2, 2001)])
    xs = (z / s).unsqueeze(1)
    val = out.compute(len(xs), {"x": xs})
    slack = swish_dip / scale * 2.0  # |T−F| = 2
    assert val.min() >= -1.0 - slack - 1e-7
    assert val.max() <= 1.0 + slack + 1e-7
    # The overshoot is real (the bound is tight, not vacuous):
    assert val.max() > 1.0 + 0.9 * slack
    assert val.min() < -1.0 - 0.9 * slack
    # And the claimed value range carries the slack.
    r = out.value_type.value_range
    assert r.lo == pytest.approx(-1.0 - slack)
    assert r.hi == pytest.approx(1.0 + slack)


def test_compare_custom_levels_and_sharpness():
    x = create_input("x", 1, value_range=(0.0, 100.0))
    out = compare(x, thresh=50.0, true_level=3.0, false_level=0.5, sharpness=100.0)
    xs = torch.tensor([[0.0], [50.0], [50.01], [60.0]])
    val = out.compute(4, {"x": xs})
    ref = torch.tensor([[0.5], [0.5], [3.0], [3.0]])
    # False side / thresh exact (±0 contributions); true side carries the
    # far-field ulp class — at x=60, s=100 the contribution magnitude is
    # 100·10·2.5 = 2500, so a few ulps is ~1e-3.
    assert torch.equal(val[:2], ref[:2])
    assert torch.allclose(val[2:], ref[2:], atol=2e-3, rtol=0.0)


def test_compare_semantic_bound_collapse_carries_slack():
    """An input interval clearing thresh collapses the semantic bound to
    the level ± the dip slack (the swish constant collapse is an
    interval, not a constant)."""
    x = create_input("x", 1, value_range=(1.0, 5.0))
    out = compare(x, thresh=0.0)
    slack = swish_dip / scale * 2.0
    iv = out.affine_bound.to_interval()
    assert len(iv) == 1
    assert iv[0].lo == pytest.approx(1.0 - slack)
    assert iv[0].hi == pytest.approx(1.0 + slack)

    # Direct unit test of the slack parameter on the shared helper.
    ab = _compare_semantic_bound(
        x._affine_bound, thresh=10.0, true_level=1.0, false_level=-1.0, slack=0.01
    )
    iv = ab.to_interval()
    assert iv[0].lo == pytest.approx(-1.01)
    assert iv[0].hi == pytest.approx(-0.99)


def test_compare_compiles_clean():
    x = create_input("x", 1, value_range=(-80.0, 80.0))
    out = compare(x, thresh=0.0)
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    g = torch.Generator().manual_seed(41)
    xs = torch.rand(32, 1, generator=g) * 160.0 - 80.0
    report = probe_compiled(compiled, out, {"x": xs}, 32, atol=1e-3)
    assert report.first_divergent is None, report.format_short()
    # debug=True runs the value-range assert (with dip slack) on the
    # compiled values.
    compiled(xs, debug=True)


# ---------------------------------------------------------------------------
# bool compositions
# ---------------------------------------------------------------------------


def _bools(*vals):
    return torch.tensor(vals).reshape(1, -1)


def test_bool_not_truth_table():
    x = create_input("x", 1, value_range=(-1.0, 1.0))
    out = bool_not(x)
    val = out.compute(2, {"x": torch.tensor([[1.0], [-1.0]])})
    # x=-1 is compare's false side: both lanes exactly 0.0, the output IS
    # the out_bias literal — bit-exact on every kernel.
    assert val[1].item() == 1.0
    # x=+1 is the far field: two saturated lanes differenced through
    # ±fl(0.02), the kernel-dependent (FMA vs per-product) ulp class —
    # same budget as test_compare_true_side_far_field_ulp_class.
    assert val[0].item() == pytest.approx(-1.0, abs=1e-5)


@pytest.mark.parametrize(
    "op,ref",
    [
        (bool_any_true, lambda a, b, c: max(a, b, c)),
        (bool_all_true, lambda a, b, c: min(a, b, c)),
    ],
)
def test_bool_any_all_truth_tables(op, ref):
    a = create_input("a", 1, value_range=(-1.0, 1.0))
    b = create_input("b", 1, value_range=(-1.0, 1.0))
    c = create_input("c", 1, value_range=(-1.0, 1.0))
    out = op([a, b, c])
    for av in (-1.0, 1.0):
        for bv in (-1.0, 1.0):
            for cv in (-1.0, 1.0):
                val = out.compute(
                    1,
                    {"a": _bools(av), "b": _bools(bv), "c": _bools(cv)},
                )
                # Stacked compares accumulate the far-field ulp class
                # (~1e-6 here), not bit-exactness — far inside every
                # downstream ±1-cond budget.
                assert val.item() == pytest.approx(ref(av, bv, cv), abs=1e-5), (
                    av,
                    bv,
                    cv,
                )


def test_bool_composition_compiles_clean():
    a = create_input("a", 1, value_range=(-1.0, 1.0))
    b = create_input("b", 1, value_range=(-1.0, 1.0))
    out = bool_all_true([bool_not(a), b])
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    inputs = {
        "a": torch.tensor([[1.0], [-1.0], [1.0], [-1.0]]),
        "b": torch.tensor([[1.0], [1.0], [-1.0], [-1.0]]),
    }
    report = probe_compiled(compiled, out, inputs, 4, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


# ---------------------------------------------------------------------------
# equals_vector
# ---------------------------------------------------------------------------

_KEY = torch.tensor([1.0, 2.0, 3.0])


def test_equals_vector_structure():
    x = create_input("x", 3, value_range=(-5.0, 5.0))
    out = equals_vector(x, _KEY)
    ffn = _unwrap(out)
    assert ffn.activation == "swish"
    assert ffn.is_degenerate
    assert ffn.n_lanes == 1


def test_equals_vector_match_and_margin_exact():
    """A match is bit-exact +1 (hinge argument scale/speed, saturated);
    a non-match exactly at the 1/speed margin is exact -1 (argument 0,
    Swish(0)=0); a deep non-match is bit-exact -1 (sigmoid underflow)."""
    x = create_input("x", 3, value_range=(-5.0, 5.0))
    out = equals_vector(x, _KEY)
    speed = 1.0  # embedding_step_sharpness
    key_sq = float(_KEY @ _KEY)
    # Row 1: the key. Row 2: dot = key² − 1/speed (at the margin).
    # Row 3: dot far below (zeros).
    at_margin = _KEY * ((key_sq - 1.0 / speed) / key_sq)
    xs = torch.stack([_KEY, at_margin, torch.zeros(3)])
    val = out.compute(3, {"x": xs})
    assert torch.equal(val, torch.tensor([[1.0], [-1.0], [-1.0]]))


def test_equals_vector_dip_and_range_slack():
    """A non-match engineered just past the margin lands in the hinge dip
    and reads below -1 by up to 2·swish_dip·speed/scale; the value-range
    assert's low side carries exactly that slack."""
    x = create_input("x", 3, value_range=(-5.0, 5.0))
    out = equals_vector(x, _KEY)
    speed = 1.0
    low_slack = 2.0 * swish_dip * speed / scale
    key_sq = float(_KEY @ _KEY)
    # Sweep m + 1/speed in (-17/scale, 0) — the dip window.
    dots = torch.linspace(
        key_sq - 1.0 / speed - 17.0 / scale, key_sq - 1.0 / speed, 2001
    )
    xs = _KEY.unsqueeze(0) * (dots / key_sq).unsqueeze(1)
    val = out.compute(len(xs), {"x": xs})
    assert val.min() >= -1.0 - low_slack - 1e-7
    assert val.min() < -1.0 - 0.9 * low_slack  # the dip is really there
    assert val.max() <= 1.0 + 1e-7

    r = out.value_type.value_range
    assert r.lo == pytest.approx(-1.0 - low_slack)
    assert r.hi == 1.0


def test_equals_vector_compiles_clean():
    x = create_input("x", 3, value_range=(-5.0, 5.0))
    out = equals_vector(x, _KEY)
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    xs = torch.stack([_KEY, torch.zeros(3), torch.tensor([1.0, 2.0, 2.0])])
    report = probe_compiled(compiled, out, {"x": xs}, 3, atol=1e-3)
    assert report.first_divergent is None, report.format_short()
    compiled(xs, debug=True)
