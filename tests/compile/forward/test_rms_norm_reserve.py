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
    _reserve_rms_norm_columns,
)
from torchwright.compiler.forward.residual_map import ResidualStreamMap


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
