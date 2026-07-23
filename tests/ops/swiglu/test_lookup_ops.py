"""swiglu map_to_table and onehot_lookup: the indicator-lane banks.

Spec: docs/ops_plain_english.md (map_to_table, onehot_lookup entries);
the counting margin and dip-leak magnitudes are pinned in
tests/docs/test_swish_constants.py (test_onehot_lookup_counting_margin).
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN, Linear
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.swiglu import map_to_table, onehot_lookup

D = 64
D_HEAD = 8

_KEYS = [
    torch.tensor([2.0, 0.0, 1.0]),
    torch.tensor([0.0, 3.0, -1.0]),
    torch.tensor([-1.0, -1.0, 2.0]),
]
_VALUES = [
    torch.tensor([10.0, -1.0]),
    torch.tensor([20.0, -2.0]),
    torch.tensor([30.0, -3.0]),
]
_DEFAULT = torch.tensor([5.0, 0.5])


def _table():
    return dict(zip(_KEYS, _VALUES, strict=False))


def _unwrap(node):
    while not isinstance(node, (FFN, Linear)):
        node = node.inputs[0]
    return node


# ---------------------------------------------------------------------------
# map_to_table
# ---------------------------------------------------------------------------


def test_map_to_table_structure():
    x = create_input("x", 3, value_range=(-5.0, 5.0))
    out = map_to_table(x, _table(), _DEFAULT)
    ffn = _unwrap(out)
    assert isinstance(ffn, FFN)
    assert ffn.activation == "swish"
    assert ffn.is_degenerate
    assert ffn.n_lanes == 3  # one per entry


def test_map_to_table_match_and_default():
    x = create_input("x", 3, value_range=(-5.0, 5.0))
    out = map_to_table(x, _table(), _DEFAULT)
    xs = torch.stack([*_KEYS, torch.zeros(3)])
    val = out.compute(4, {"x": xs})
    # Matches: value_i to ~1 ulp (×scale/÷scale round trip); other
    # entries' leakage underflows.
    for i, v in enumerate(_VALUES):
        assert torch.allclose(val[i], v, rtol=1e-6, atol=1e-6), (i, val[i])
    # No-match: every indicator saturated to exactly zero → out_bias.
    assert torch.equal(val[3], _DEFAULT)


def test_map_to_table_value_range_claim_ports_unchanged():
    x = create_input("x", 3, value_range=(-5.0, 5.0))
    out = map_to_table(x, _table(), _DEFAULT)
    diff_abs_sum = sum((v - _DEFAULT).abs() for v in _VALUES)
    r = out.value_type.value_range
    assert r.lo == pytest.approx(float((_DEFAULT - diff_abs_sum).min()))
    assert r.hi == pytest.approx(float((_DEFAULT + diff_abs_sum).max()))


def test_map_to_table_compiles_clean():
    x = create_input("x", 3, value_range=(-5.0, 5.0))
    out = map_to_table(x, _table(), _DEFAULT)
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    xs = torch.stack([_KEYS[0], _KEYS[2], torch.zeros(3)])
    report = probe_compiled(compiled, out, {"x": xs}, 3, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


# ---------------------------------------------------------------------------
# onehot_lookup
# ---------------------------------------------------------------------------


def _onehot(i, n):
    v = torch.zeros(n)
    v[i] = 1.0
    return v


def test_onehot_lookup_single_block_is_plain_linear():
    """n_blocks = 1 stays a selection-matrix Linear — no FFN at all."""
    x = create_input("x", 4, value_range=(0.0, 1.0))
    table = {_onehot(0, 4): torch.tensor([7.0]), _onehot(2, 4): torch.tensor([9.0])}
    out = onehot_lookup(x, table, torch.tensor([-1.0]))
    assert isinstance(_unwrap(out), Linear)
    xs = torch.stack([_onehot(0, 4), _onehot(1, 4), _onehot(2, 4)])
    val = out.compute(3, {"x": xs})
    assert torch.equal(val, torch.tensor([[7.0], [-1.0], [9.0]]))


def _two_block_table():
    # keys: digit(3) ⊕ carry(2)
    def key(d, c):
        return torch.cat([_onehot(d, 3), _onehot(c, 2)])

    return {
        key(0, 0): torch.tensor([100.0]),
        key(1, 0): torch.tensor([200.0]),
        key(2, 1): torch.tensor([300.0]),
    }, key


def test_onehot_lookup_multi_block_counting():
    table, key = _two_block_table()
    x = create_input("x", 5, value_range=(0.0, 1.0))
    out = onehot_lookup(x, table, torch.tensor([-7.0]))
    ffn = _unwrap(out)
    assert isinstance(ffn, FFN)
    assert ffn.is_degenerate
    assert ffn.n_lanes == 3
    xs = torch.stack([key(0, 0), key(1, 0), key(2, 1), key(0, 1), key(2, 0)])
    val = out.compute(5, {"x": xs})
    # Matches to ~1 ulp of the value; misses return default plus a
    # ~1e-28-class dip leak (hinge(-0.5) — representable, not exactly
    # zero, nearly thirty orders below visibility).
    ref = torch.tensor([[100.0], [200.0], [300.0], [-7.0], [-7.0]])
    assert torch.allclose(val, ref, rtol=1e-6, atol=1e-9)


def test_onehot_lookup_tight_range_claim():
    table, _ = _two_block_table()
    x = create_input("x", 5, value_range=(0.0, 1.0))
    out = onehot_lookup(x, table, torch.tensor([-7.0]))
    r = out.value_type.value_range
    assert r.lo == -7.0
    assert r.hi == 300.0


def test_onehot_lookup_noise_passthrough_linear_in_epsilon():
    """An input one-hot off by ε shifts the winner's indicator by
    exactly ε: output error 2ε·|value − default| — today's coefficient.
    """
    table, key = _two_block_table()
    x = create_input("x", 5, value_range=(0.0, 1.0))
    default = torch.tensor([-7.0])
    out = onehot_lookup(x, table, default)
    eps = 0.01
    noisy = key(0, 0)
    noisy[0] -= eps  # winner block count drops by eps
    val = out.compute(1, {"x": noisy.unsqueeze(0)})
    expected = 100.0 - 2.0 * eps * (100.0 - (-7.0))
    assert val.item() == pytest.approx(expected, rel=1e-5)


def test_onehot_lookup_compiles_clean():
    table, key = _two_block_table()
    x = create_input("x", 5, value_range=(0.0, 1.0))
    out = onehot_lookup(x, table, torch.tensor([-7.0]))
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    xs = torch.stack([key(0, 0), key(2, 1), key(1, 1)])
    report = probe_compiled(compiled, out, {"x": xs}, 3, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_onehot_lookup_wide_key_accumulated_leak_within_guard():
    """D6 repro (Phase C, calculator_simple 123*456): a machine-built
    one-hot carries ~1e-5 per-element round-trip leak, and a wide key
    (d_key = 61 on the digit pipeline) sums d_key of them, each weighted
    by up to the largest table magnitude — past the old fixed 1e-3
    closing-assert slack in exact math.  The guard is now sized by
    _lookup_numeric_slack(max_abs, 1.0, d_key).
    """
    d_key = 61
    keys = [torch.zeros(d_key) for _ in range(d_key)]
    for i, k in enumerate(keys):
        k[i] = 1.0
    # half the rows carry the max-magnitude value, like the digit table
    table = {keys[i]: torch.tensor([6.0 if i % 2 == 0 else 0.0]) for i in range(d_key)}
    x = create_input("x", d_key, value_range=(0.0, 1.0))
    out = onehot_lookup(x, table, torch.tensor([0.0]))

    noisy = keys[0].clone()
    noisy += 1.5e-5  # per-element leak, ~the measured machine magnitude
    val = out.compute(1, {"x": noisy.unsqueeze(0)})  # asserts run here
    # accumulated error is real (past the old 1e-3 fixed slack) but the
    # derived guard absorbs it
    assert abs(val.item() - 6.0) > 1e-3


def test_onehot_lookup_small_table_guard_stays_tight():
    """The derived slack stays small for small tables (max_abs=300,
    d_key=5 → 0.015): a gross input error that pushes the output past
    the claimed [min, max] still fires the closing assert.
    """
    table, key = _two_block_table()
    x = create_input("x", 5, value_range=(0.0, 1.0))
    out = onehot_lookup(x, table, torch.tensor([-7.0]))
    noisy = key(2, 1)  # the 300-valued row
    noisy[2] += 0.05  # count 1.05 → output ≈ 315, past hi=300 + 0.015
    with pytest.raises(AssertionError, match="range"):
        out.compute(1, {"x": noisy.unsqueeze(0)})
