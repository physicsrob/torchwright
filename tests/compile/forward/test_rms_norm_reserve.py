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
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.graph.value_type import Range


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
