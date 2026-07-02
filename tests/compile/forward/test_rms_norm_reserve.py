"""Pinned-constant RMSNorm: the reserved-column layout and the out-energy bound.

These are the CPU/math-level guards for the identity-RMSNorm feature
(``compiler/export.py`` ``rms_norm``); the full ONNX/HF bit-exactness through
onnxruntime is checked in ``tests/hf/test_rms_norm_identity.py``.

The energy-bound test permanently encodes the finding that broke the shipping
graph: q=30 was validated only on ``calculator_simple`` (data energy ~1e8) and
silently broke the identity on ``calculator_v2`` (squaring path reaches
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
)
from torchwright.compiler.forward.cpsat_scheduler import (
    build_cpsat_model,
    solve_schedule,
)
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.graph_identity import graph_fingerprint
from torchwright.graph import Linear
from torchwright.ops.linear_relu_linear import linear_relu_linear
from torchwright.graph.value_type import Range
from torchwright.ops.inout_nodes import create_input


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
        reserved_cols=reserved, const_value=float(2**q), gain=float(2**m), eps=1e-5
    )


@pytest.mark.parametrize(
    "d, n_const_expected",
    [
        (256, 1),  # b=8 even  -> one column
        (1024, 1),  # b=10 even -> one column (calculator)
        (2048, 2),  # b=11 odd  -> two columns
        (8192, 2),  # b=13 odd  -> two columns (DOOM)
    ],
)
def test_column_count_from_width_parity(d, n_const_expected):
    rmap = ResidualStreamMap(d)
    spec = _reserve_rms_norm_columns(rmap, d, 1e-5, _RMS_NORM_CONST_EXP)
    assert len(spec.reserved_cols) == n_const_expected


@pytest.mark.parametrize("d", [256, 1024, 2048, 8192])
def test_gain_exactly_cancels_forced_rms(d):
    """gain == forced rms == 2^m, so x/rms*gain == x — the identity."""
    rmap = ResidualStreamMap(d)
    spec = _reserve_rms_norm_columns(rmap, d, 0.0, _RMS_NORM_CONST_EXP)
    import math

    energy = len(spec.reserved_cols) * spec.const_value**2
    forced_rms = (energy / d) ** 0.5
    assert forced_rms == spec.gain
    # gain is an exact power of two (what makes ÷rms and ×gain bit-exact shifts)
    assert math.log2(spec.gain) == int(math.log2(spec.gain))


@pytest.mark.parametrize("d", [256, 1024, 2048, 8192])
def test_reserved_columns_are_freed_from_pool_and_not_allocated(d):
    rmap = ResidualStreamMap(d)
    spec = _reserve_rms_norm_columns(rmap, d, 1e-5, _RMS_NORM_CONST_EXP)
    for c in spec.reserved_cols:
        assert c not in rmap._free, "reserved column must leave the free pool"
        assert c in rmap._reserved, "reserved column must be tracked as reserved"
    # I1 (allocator self-consistency) must still hold after the reservation.
    rmap._check_invariants("post-reserve test")


def test_non_power_of_two_width_raises():
    rmap = ResidualStreamMap(384)  # not a power of two
    with pytest.raises(ValueError, match="power-of-two"):
        _reserve_rms_norm_columns(rmap, 384, 1e-5, _RMS_NORM_CONST_EXP)


def test_const_exp_overflow_raises():
    """A q so large the pinned energy overflows fp32 fails loudly, not silently
    to inf inside the norm's square."""
    rmap = ResidualStreamMap(1024)
    with pytest.raises(ValueError, match="overflows fp32"):
        _reserve_rms_norm_columns(rmap, 1024, 1e-5, 64)  # 2^128 -> inf


def test_eps_above_rms_lsb_raises_but_zero_is_fine():
    """eps large enough to perturb the forced mean-square breaks the identity and
    must raise; eps=0.0 (falsy but valid, below the LSB) must be accepted."""
    rmap = ResidualStreamMap(1024)
    with pytest.raises(ValueError, match="too large for the forced RMS"):
        _reserve_rms_norm_columns(rmap, 1024, 1e20, _RMS_NORM_CONST_EXP)
    # 0.0 is below the LSB and legitimate — must not be coerced or rejected.
    spec = _reserve_rms_norm_columns(
        ResidualStreamMap(1024), 1024, 0.0, _RMS_NORM_CONST_EXP
    )
    assert spec.eps == 0.0


def _rmsnorm_identity_holds(d, n_const, q, data_energy, eps=1e-5):
    """Build a residual with the pinned constant + random data scaled to a given
    Sigma data^2, run a real RMSNorm with gain=2^m, and report whether the data
    columns come back bit-for-bit (the identity)."""
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
    """A graph whose per-column energy stays under const²·2⁻²⁴ certifies."""
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
    """The pinned constant columns are the pin, not data — excluded from the
    energy sum (so a huge value there does not trip the budget)."""
    spec = _spec(reserved=(1023,))
    # the only node sits on the reserved column at the constant magnitude
    ra = _FakeRA({"s0": {_FakeNode(-(2.0**44), 2.0**44): [1023]}})
    _certify_rms_norm_energy(ra, spec)  # must not raise (reserved col skipped)


def test_energy_bound_identity_holds_below_and_breaks_above():
    """The Open-Q5 finding, at the math layer: identity below the half-ULP bound
    2^(2q-24), broken above it.  This is why the default q must clear the
    deepest-layer energy of the *shipping* graph, not just calculator_simple."""
    d, n_const, q = 1024, 1, 30  # the original (too-small) calculator setting
    bound = 2.0 ** (2 * q - 24)  # ~6.9e10
    # calculator_simple-scale energy: well under the bound -> identity holds
    assert _rmsnorm_identity_holds(d, n_const, q, data_energy=1e8)
    # calculator_v2 squaring-path energy: over the bound -> identity breaks
    assert not _rmsnorm_identity_holds(d, n_const, q, data_energy=2.6e13)
    # the default q clears the same high energy with margin
    assert _rmsnorm_identity_holds(d, n_const, _RMS_NORM_CONST_EXP, data_energy=2.6e13)
    assert 2.6e13 < 2.0 ** (2 * _RMS_NORM_CONST_EXP - 24)  # the margin is real


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
    """x -> FFN (a degenerate-ReLU FFN); small enough to solve
    sub-second."""
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


_CPSAT_KW = dict(d=64, d_head=8, d_hidden=128)


def test_cpsat_available_residual_excludes_reserved():
    """build_cpsat_model subtracts reserve_residual from the residual budget,
    matching how the reservation shrinks the replay pool (GAP 1 root cause)."""
    out = _tiny_graph()
    # Position is rotary post-RoPE, so build_cpsat_model takes no pos node (its
    # ``pos_encoding`` param is vestigial); the self-match column is the fixed
    # base reservation that reserve_residual adds on top of.
    base = build_cpsat_model(out, **_CPSAT_KW).available_residual
    for k in (1, 2):
        got = build_cpsat_model(out, reserve_residual=k, **_CPSAT_KW)
        assert got.available_residual == base - k


def test_cpsat_residual_oversubscription_raises():
    """Reserving every data column fails loud at model build, not as a confusing
    downstream solve failure, and the message names the reserved columns. With
    the RoPE self-match base of 1, reserving d-1 leaves zero room."""
    out = _tiny_graph()
    with pytest.raises(RuntimeError, match="reserved columns"):
        build_cpsat_model(out, reserve_residual=64 - 1, **_CPSAT_KW)


def test_solve_schedule_threads_reserve_residual():
    """reserve_residual flows end-to-end through solve_schedule and the solver
    still produces a feasible schedule under the reduced budget."""
    out = _tiny_graph()
    assignment, _ = solve_schedule(
        out, reserve_residual=2, time_budget_s=10.0, max_layers=20, **_CPSAT_KW
    )
    assert assignment is not None


def test_schedule_fingerprint_keys_on_reserved_residual():
    """The schedule cache must distinguish a norm-on (reserved) compile from a
    norm-off one (GAP 2), while keeping the no-reservation hash byte-identical
    so existing cache entries still hit (the conditional payload field)."""
    out = _tiny_graph()
    fp_kw = dict(
        d=64,
        d_head=8,
        d_hidden=64,
        flex_routing=True,
        assume_zero_init=True,
        cancel_slack=2,
        policy=None,
    )
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
    """The certification bound is the per-column max over ALL sublayer snapshots,
    not one state: a column small in one snapshot and large in another is bounded
    by the large one.  Here neither state alone exceeds the budget, but the
    per-column max over both does — so the cross-snapshot reduction must catch
    it (the soundness-bearing path, untested before)."""
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
