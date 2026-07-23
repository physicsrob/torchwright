"""Pinned-constant RMSNorm: the reserved-column layout and the out-energy bound.

These are the CPU/math-level guards for the identity-RMSNorm feature
(``compiler/export.py`` ``rms_norm``); the full ONNX/HF bit-exactness through
onnxruntime is checked in ``tests/hf/test_rms_norm_identity.py``.

The energy-bound test permanently encodes the finding that broke the shipping
graph: q=30 was validated only on ``calculator_simple`` (data energy ~1e8) and
silently broke the identity on ``calculator_simple`` (squaring path reaches
Sigma data^2 ~ 2.6e13).  The identity holds iff Sigma data^2 stays under the
constant's reduction half-ULP ~ 2^(2q-24).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torchwright.compiler.forward.compile import (
    _RMS_NORM_CONST_EXP,
    RmsNormSpec,
    _certify_rms_norm_energy,
    _reserve_rms_norm_columns,
    _rms_norm_pinned_layout,
    rms_norm_width_supported,
)
from torchwright.compiler.forward.cpsat_scheduler import (
    build_cpsat_model,
    solve_schedule,
)
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.graph_identity import graph_fingerprint
from torchwright.graph.value_type import Range
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear


class _FakeVT:
    def __init__(self, lo, hi):
        self.value_range = Range(lo, hi)


class _FakeNode:
    def __init__(self, lo, hi):
        self.value_type = _FakeVT(lo, hi)


class _FakeRA:
    """Minimal stand-in for ResidualAssignment: .mapping = {state: {node: cols}}."""

    def __init__(self, mapping):
        self.mapping = mapping


def _spec(q=_RMS_NORM_CONST_EXP, reserved=(1023,)):
    b = 10
    m = (2 * q - b) // 2
    return RmsNormSpec(
        reserved_cols=reserved,
        const_values=tuple(float(2**q) for _ in reserved),
        gain=float(2**m),
        eps=1e-5,
    )


@pytest.mark.parametrize(
    ("d", "n_const_expected"),
    [
        (256, 1),  # 2^8, even exponent  -> one column
        (1024, 1),  # 2^10, even          -> one column (calculator)
        (2048, 2),  # 2^11, odd           -> two columns
        (8192, 2),  # 2^13, odd           -> two columns (DOOM)
        (384, 3),  # 3·2^7               -> three columns
        (3072, 3),  # 3·2^10              -> three columns
        (5120, 2),  # 5·2^10              -> two columns (2^q and 2^(q+1))
        (7168, 4),  # 7·2^10              -> four columns
        (15360, 6),  # 15·2^10             -> six columns (contract worst case)
    ],
)
def test_column_count_from_width_parity(d, n_const_expected):
    rmap = ResidualStreamMap(d)
    spec = _reserve_rms_norm_columns(rmap, d, 1e-5, _RMS_NORM_CONST_EXP)
    assert len(spec.reserved_cols) == n_const_expected
    assert len(spec.const_values) == n_const_expected


@pytest.mark.parametrize("d", [256, 1024, 2048, 8192, 3072, 5120, 15360])
def test_gain_exactly_cancels_forced_rms(d):
    """Gain == forced rms == 2^m, so x/rms*gain == x — the identity."""
    rmap = ResidualStreamMap(d)
    spec = _reserve_rms_norm_columns(rmap, d, 0.0, _RMS_NORM_CONST_EXP)
    import math

    energy = sum(v * v for v in spec.const_values)
    forced_rms = (energy / d) ** 0.5
    assert forced_rms == spec.gain
    # gain is an exact power of two (what makes dividing by rms and
    # multiplying by gain bit-exact shifts)
    assert math.log2(spec.gain) == int(math.log2(spec.gain))


def test_distinct_column_values_at_5120():
    """d=5120 = 5*2^10 pins two different constants (2^q and 2^(q+1)).

    5*2^(2q) = 2^(2q) + 2^(2q+2). Guards the per-column value plumbing
    that equal-constant widths cannot distinguish from a single shared
    value.
    """
    q = _RMS_NORM_CONST_EXP
    spec = _reserve_rms_norm_columns(ResidualStreamMap(5120), 5120, 1e-5, q)
    assert spec.const_values == (2.0**q, 2.0 ** (q + 1))


@pytest.mark.parametrize("d", [256, 1024, 2048, 8192, 5120])
def test_reserved_columns_are_freed_from_pool_and_not_allocated(d):
    rmap = ResidualStreamMap(d)
    spec = _reserve_rms_norm_columns(rmap, d, 1e-5, _RMS_NORM_CONST_EXP)
    for c in spec.reserved_cols:
        assert c not in rmap._free, "reserved column must leave the free pool"
        assert c in rmap._reserved, "reserved column must be tracked as reserved"
    # I1 (allocator self-consistency) must still hold after the reservation.
    rmap._check_invariants("post-reserve test")


def test_unbuildable_width_raises():
    """Odd factor 41 is the smallest whose fp32 mean arithmetic misses a power of two.

    The reciprocal-multiply path rounds off it, so the reservation must
    refuse rather than ship a near-identity norm.
    """
    d = 41 * 32
    rmap = ResidualStreamMap(d)
    with pytest.raises(ValueError, match="no bit-exact pinned layout"):
        _reserve_rms_norm_columns(rmap, d, 1e-5, _RMS_NORM_CONST_EXP)


def test_width_contract_predicate():
    """The public promise: any multiple of 1024 up to 16384, or any power of two.

    17408 (a multiple of 1024 past the cap) and 1280 (odd*2^8, buildable
    by the mechanism, but unpromised) are outside the contract.
    """
    assert all(rms_norm_width_supported(n * 1024) for n in range(1, 17))
    assert all(rms_norm_width_supported(2**k) for k in range(6, 16))
    for bad in (0, -1024, 1000, 1280, 17408):
        assert not rms_norm_width_supported(bad)


def test_compile_to_onnx_rejects_unsupported_width_before_compiling():
    """The front door fails fast: an unsupported d raises at entry.

    With the norm on (the default), the raise happens before the graph is
    even touched (graph=None is never dereferenced), so a bad width can't
    waste a long streaming compile.
    """
    from torchwright.compiler.export import compile_to_onnx

    with pytest.raises(ValueError, match="supported width"):
        compile_to_onnx(None, None, "/nonexistent/never_written.onnx", d=1000)


def test_const_exp_overflow_raises():
    """A q so large the pinned energy overflows fp32 fails loudly.

    It does not silently go to inf inside the norm's square.
    """
    rmap = ResidualStreamMap(1024)
    with pytest.raises(ValueError, match="overflows fp32"):
        _reserve_rms_norm_columns(rmap, 1024, 1e-5, 64)  # 2^128 -> inf


def test_overflow_boundary_is_inclusive_doom_q63_at_8192():
    """The fp32 ceiling admits a pinned energy whose top bit is exactly 2^127.

    That value is representable (fp32 max is ~2^128). The DOOM production
    config (d=8192, rms_norm_const_exp=63, forced energy 2^127) sits
    exactly on this boundary and must be accepted; q=64 must overflow.
    Regression for the off-by-one guard that rejected the boundary and
    blocked the production compile.
    """
    spec = _reserve_rms_norm_columns(ResidualStreamMap(8192), 8192, 1e-5, 63)
    assert spec.const_values == (2.0**63, 2.0**63)  # 2·2^126 = 2^127
    assert spec.gain == 2.0**57
    with pytest.raises(ValueError, match="overflows fp32"):
        _reserve_rms_norm_columns(ResidualStreamMap(8192), 8192, 1e-5, 64)


def test_eps_above_rms_lsb_raises_but_zero_is_fine():
    """Eps large enough to perturb the forced mean-square breaks the identity.

    That case must raise; eps=0.0 (falsy but valid, below the LSB) must
    be accepted.
    """
    rmap = ResidualStreamMap(1024)
    with pytest.raises(ValueError, match="too large for the forced RMS"):
        _reserve_rms_norm_columns(rmap, 1024, 1e20, _RMS_NORM_CONST_EXP)
    # 0.0 is below the LSB and legitimate — must not be coerced or rejected.
    spec = _reserve_rms_norm_columns(
        ResidualStreamMap(1024), 1024, 0.0, _RMS_NORM_CONST_EXP
    )
    assert spec.eps == 0.0


@pytest.mark.parametrize("d", [n * 1024 for n in range(1, 17)] + [64, 128, 256, 512])
def test_contract_width_fp32_mean_exactness(d):
    """Every contract width has a pinned layout landing exactly on 2^(2m).

    The compile_to_onnx contract widths are multiples of 1024 up to
    16384, plus small powers of two. The forced mean-of-squares lands
    exactly on 2^(2m) under BOTH mean strategies a runtime may use
    (sum/d, or sum*(1/d)): the property the reservation guard enforces,
    swept explicitly over the whole promised set.
    """
    import numpy as np

    assert rms_norm_width_supported(d)
    col_exps, m = _rms_norm_pinned_layout(d, _RMS_NORM_CONST_EXP)
    energy = np.float32(0.0)
    for ce in sorted(col_exps):  # worst order: smallest pinned values first
        energy = energy + np.float32(2.0**ce) * np.float32(2.0**ce)
    assert float(energy) == d * 2.0 ** (2 * m), "pinned fp32 sum must be exact"
    target = np.float32(2.0 ** (2 * m))
    assert energy / np.float32(d) == target
    assert energy * (np.float32(1.0) / np.float32(d)) == target
    assert np.sqrt(target) == np.float32(2.0**m)


def _rmsnorm_identity_holds_general(d, const_values, gain, data_energy, eps=1e-5):
    """Like :func:`_rmsnorm_identity_holds` but takes the per-column pinned values.

    Taking them directly lets it cover layouts with unequal constants.
    """
    n_const = len(const_values)
    d_data = d - n_const
    torch.manual_seed(0)
    data = torch.randn(8, d_data, dtype=torch.float32)
    cur = (data * data).sum(-1, keepdim=True).clamp_min(1e-30)
    data = data * (data_energy / cur).sqrt()
    const = torch.tensor(const_values, dtype=torch.float32).expand(8, n_const)
    res = torch.cat([data, const], dim=1)
    ms = (res * res).mean(-1, keepdim=True)
    normed = res / torch.sqrt(ms + eps) * gain
    return bool((normed[:, :d_data] == data).all())


@pytest.mark.parametrize("d", [3072, 5120, 15360])
def test_identity_holds_at_non_power_of_two_widths(d):
    """Torch-level RMSNorm identity at odd-factor widths.

    Data under the certified budget comes back bit-for-bit; far over it,
    the mean drifts off the power of two and the identity breaks. (The
    certified budget 2^(2q-24) is deliberately conservative, sound in
    every fp32 summation order, so the observable break point sits above
    it, never below.)
    """
    q = _RMS_NORM_CONST_EXP
    spec = _reserve_rms_norm_columns(ResidualStreamMap(d), d, 1e-5, q)
    assert _rmsnorm_identity_holds_general(
        d, spec.const_values, spec.gain, data_energy=2.6e13
    )
    assert not _rmsnorm_identity_holds_general(
        d, spec.const_values, spec.gain, data_energy=1e21
    )


def _rmsnorm_identity_holds(d, n_const, q, data_energy, eps=1e-5):
    """Build a residual, run a real RMSNorm, and report whether the identity holds.

    The residual is the pinned constant plus random data scaled to a
    given Sigma data^2; the RMSNorm uses gain=2^m; the identity check is
    whether the data columns come back bit-for-bit.
    """
    b = d.bit_length() - 1
    e_exp = 2 * q + (0 if n_const == 1 else 1)
    m = (e_exp - b) // 2
    gain = 2.0**m
    d_data = d - n_const
    torch.manual_seed(0)
    data = torch.randn(8, d_data, dtype=torch.float32)
    # scale so the worst-row Sigma data^2 ~= data_energy
    cur = (data * data).sum(-1, keepdim=True).clamp_min(1e-30)
    data = data * (data_energy / cur).sqrt()
    const = torch.full((8, n_const), float(2**q))
    res = torch.cat([data, const], dim=1)
    ms = (res * res).mean(-1, keepdim=True)
    normed = res / torch.sqrt(ms + eps) * gain
    return bool((normed[:, :d_data] == data).all())


def test_certify_passes_under_budget():
    """A graph whose per-column energy stays under const^2 * 2^-24 certifies."""
    spec = _spec()  # q=44 -> budget 2^64
    # one node spanning many columns, magnitude well under the budget
    ra = _FakeRA({"s0": {_FakeNode(-1e6, 1e6): list(range(900))}})
    _certify_rms_norm_energy(ra, spec)  # must not raise (~900 * 1e12 = 9e14 < 2^64)


def test_certify_raises_over_budget_with_required_q():
    """Energy above the budget raises and names a larger sufficient q."""
    spec = _spec()  # q=44 -> budget 2^64 ~ 1.8e19
    # 1000 columns each at magnitude 1e10 -> 1e20 energy, over budget
    ra = _FakeRA({"s0": {_FakeNode(-1e10, 1e10): list(range(1000))}})
    with pytest.raises(ValueError, match=r"not certified.*rms_norm_const_exp>="):
        _certify_rms_norm_energy(ra, spec)


def test_certify_raises_on_non_finite_range():
    """A node the compiler can't bound cannot be certified — fail loud."""
    spec = _spec()
    ra = _FakeRA({"s0": {_FakeNode(float("-inf"), float("inf")): [0]}})
    with pytest.raises(ValueError, match="non-finite value range"):
        _certify_rms_norm_energy(ra, spec)


def test_certify_ignores_reserved_columns():
    """The pinned constant columns are the pin, not data.

    They are excluded from the energy sum, so a huge value there does
    not trip the budget.
    """
    spec = _spec(reserved=(1023,))
    # the only node sits on the reserved column at the constant magnitude
    ra = _FakeRA({"s0": {_FakeNode(-(2.0**44), 2.0**44): [1023]}})
    _certify_rms_norm_energy(ra, spec)  # must not raise (reserved col skipped)


def test_energy_bound_identity_holds_below_and_breaks_above():
    """The Open-Q5 finding, at the math layer: identity below the half-ULP bound.

    The bound is 2^(2q-24); the identity breaks above it. This is why the
    default q must clear the deepest-layer energy of the shipping graph,
    not just calculator_simple.
    """
    d, n_const, q = 1024, 1, 30  # the original (too-small) calculator setting
    2.0 ** (2 * q - 24)  # ~6.9e10
    # calculator_simple-scale energy: well under the bound -> identity holds
    assert _rmsnorm_identity_holds(d, n_const, q, data_energy=1e8)
    # calculator_simple squaring-path energy: over the bound -> identity breaks
    assert not _rmsnorm_identity_holds(d, n_const, q, data_energy=2.6e13)
    # the default q clears the same high energy with margin
    assert _rmsnorm_identity_holds(d, n_const, _RMS_NORM_CONST_EXP, data_energy=2.6e13)
    assert 2.0 ** (2 * _RMS_NORM_CONST_EXP - 24) > 2.6e13  # the margin is real


# ===========================================================================
# The CP-SAT scheduler reserves the norm column(s) BEFORE scheduling, but the
# solver never sees ``residual_map`` — so its residual-capacity model and its
# schedule-cache key must both account for the reservation independently, or it
# over-counts capacity by ``n_const`` and can emit a schedule that is infeasible
# on replay against the reservation-reduced pool (a loud out-of-columns /
# liveness failure under width pressure — exactly where DOOM-class graphs run).
# These pin that wiring; they run on CPU sub-second (tiny graph, no certify).
# ===========================================================================


def _tiny_graph():
    """X -> FFN (a degenerate-ReLU FFN); small enough to solve sub-second."""
    torch.manual_seed(0)
    x = create_input("x", 8)
    return linear_relu_linear(
        x,
        torch.randn(16, 8),
        torch.zeros(16),
        torch.randn(16, 4),
        torch.zeros(4),
        name="ffn",
    )


_CPSAT_KW = {"d": 64, "d_head": 8, "d_hidden": 128}


def test_cpsat_available_residual_excludes_reserved():
    """build_cpsat_model subtracts reserve_residual from the residual budget.

    That matches how the reservation shrinks the replay pool (GAP 1 root
    cause).
    """
    out = _tiny_graph()
    # Position is rotary post-RoPE, so build_cpsat_model takes no pos node (its
    # ``pos_encoding`` param is vestigial); the self-match column is the fixed
    # base reservation that reserve_residual adds on top of.
    base = build_cpsat_model(out, **_CPSAT_KW).available_residual
    for k in (1, 2):
        got = build_cpsat_model(out, reserve_residual=k, **_CPSAT_KW)
        assert got.available_residual == base - k


def test_cpsat_residual_oversubscription_raises():
    """Reserving every data column fails loud at model build.

    It does not fail as a confusing downstream solve failure, and the
    message names the reserved columns. With the RoPE self-match base of
    1, reserving d-1 leaves zero room.
    """
    out = _tiny_graph()
    with pytest.raises(RuntimeError, match="reserved columns"):
        build_cpsat_model(out, reserve_residual=64 - 1, **_CPSAT_KW)


def test_solve_schedule_threads_reserve_residual():
    """reserve_residual flows end-to-end through solve_schedule.

    The solver still produces a feasible schedule under the reduced
    budget.
    """
    out = _tiny_graph()
    assignment, _ = solve_schedule(
        out, reserve_residual=2, time_budget_s=10.0, max_layers=20, **_CPSAT_KW
    )
    assert assignment is not None


def test_schedule_fingerprint_keys_on_reserved_residual():
    """The schedule cache must distinguish a norm-on (reserved) compile from norm-off.

    This is GAP 2, while keeping the no-reservation hash byte-identical
    so existing cache entries still hit (the conditional payload field).
    """
    out = _tiny_graph()
    fp_kw = {
        "d": 64,
        "d_head": 8,
        "d_hidden": 64,
        "flex_routing": True,
        "cancel_slack": 2,
        "policy": None,
    }
    fp_none = graph_fingerprint(out, **fp_kw)
    fp_zero = graph_fingerprint(out, reserve_residual=0, **fp_kw)
    fp_one = graph_fingerprint(out, reserve_residual=1, **fp_kw)
    fp_two = graph_fingerprint(out, reserve_residual=2, **fp_kw)
    # reserve_residual=0 must hash exactly as if the arg were never passed.
    assert fp_zero == fp_none
    # A reservation must change the key, and different counts must differ.
    assert fp_one != fp_none
    assert fp_two != fp_one


def test_certify_uses_per_column_max_across_states():
    """The certification bound is the per-column max over ALL sublayer snapshots.

    Not one state: a column small in one snapshot and large in another
    is bounded by the large one. Here neither state alone exceeds the
    budget, but the per-column max over both does, so the cross-snapshot
    reduction must catch it (the soundness-bearing path, untested before).
    """
    spec = _spec(reserved=(9999,))  # q=44 -> budget 2^64; reserved col disjoint
    big = _FakeNode(-(2.0**27), 2.0**27)  # energy 2^54 per column
    small = _FakeNode(-1.0, 1.0)
    lo = list(range(768))  # 768 * 2^54 ~ 1.4e19 < budget 1.8e19
    hi = list(range(768, 1536))
    s0 = {big: lo, small: hi}  # big on the low half
    s1 = {small: lo, big: hi}  # big on the high half — mirror image
    # Each state ALONE is under budget...
    _certify_rms_norm_energy(_FakeRA({"s0": s0}), spec)
    _certify_rms_norm_energy(_FakeRA({"s1": s1}), spec)
    # ...but the per-column max over BOTH puts all 1536 columns at 2^54;
    # 1536 * 2^54 ~ 2.8e19 > budget, so the combined certify must raise.
    with pytest.raises(ValueError, match="not certified"):
        _certify_rms_norm_energy(_FakeRA({"s0": s0, "s1": s1}), spec)
