"""Consistency checks for ``docs/rms_norm_dmodel.md``.

The doc promises a width contract ("any multiple of 1024 up to 16384, or any
power of two") and tabulates, per width, its factorization and how many
residual columns the pinned-constant RMSNorm reserves.  These tests parse the
table and assert every row against the compiler's actual layout, so the doc
cannot silently drift from the code.  They also pin the doc's prose claims
("the first odd factor that fails at all is 41"; powers of two reserve 1 or 2
columns by exponent parity).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("torch")

from torchwright.compiler.forward.compile import (
    _RMS_NORM_CONST_EXP,
    _rms_norm_pinned_layout,
    rms_norm_width_supported,
)

_DOC = Path(__file__).resolve().parents[2] / "docs" / "rms_norm_dmodel.md"

# | 5120 | 5·2^10 | 2 |
_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)·2\^(\d+)\s*\|\s*(\d+)\s*\|\s*$")


def _table_rows():
    rows = []
    for line in _DOC.read_text().splitlines():
        m = _ROW.match(line.strip())
        if m:
            rows.append(tuple(int(g) for g in m.groups()))
    return rows


def test_doc_exists_and_table_is_the_full_contract():
    rows = _table_rows()
    assert [r[0] for r in rows] == [
        n * 1024 for n in range(1, 17)
    ], "the table must list exactly the sixteen multiples of 1024, in order"


def test_table_rows_match_compiler_layout():
    for d, c, k, n_cols in _table_rows():
        assert c * 2**k == d, f"row {d}: factorization {c}·2^{k} is wrong"
        assert c % 2 == 1, f"row {d}: {c} is not the odd factor"
        assert rms_norm_width_supported(d), f"row {d}: outside the contract"
        col_exps, _m = _rms_norm_pinned_layout(d, _RMS_NORM_CONST_EXP)
        assert len(col_exps) == n_cols, (
            f"row {d}: doc says {n_cols} reserved columns, "
            f"layout has {len(col_exps)}"
        )


def test_power_of_two_column_count_claim():
    """The doc's prose: powers of two reserve 1 column at an even exponent,
    2 at an odd one."""
    for k in range(6, 15):
        col_exps, _m = _rms_norm_pinned_layout(2**k, _RMS_NORM_CONST_EXP)
        assert len(col_exps) == (1 if k % 2 == 0 else 2)


def test_first_failing_odd_factor_claim_is_41():
    """The doc's prose: every odd factor below 41 passes the fp32 mean
    arithmetic both ways (sum/d and sum·(1/d)); 41 fails.  Mirrors the
    reservation guard's arithmetic exactly."""
    import numpy as np

    def mean_exact(c: int) -> bool:
        # Scale-invariant: pinned energy and d share the mantissa c, so the
        # power-of-two parts cancel in both mean strategies.
        div = np.float32(c) / np.float32(c)
        recip = np.float32(c) * (np.float32(1.0) / np.float32(c))
        return div == np.float32(1.0) and recip == np.float32(1.0)

    for c in range(1, 41, 2):
        assert mean_exact(c), f"odd factor {c} unexpectedly fails"
    assert not mean_exact(41)
