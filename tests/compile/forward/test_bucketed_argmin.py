"""Compiled (forward-compile) round-trip tests for
``attend_argmin_above_in_bucket``.

The op's exact-math (``node.compute``) behaviour and matrix structure are
covered in ``tests/ops/test_attention_ops.py``.  These tests confirm the
compiled fp32 transformer reproduces the exact-math selection through the
structural paths exact-math cannot reach: the residual-stream layout,
constant materialization, and — for a wide value — the V/O split across
physical heads.  (Softmax hardness itself is not a compiled-only risk here:
the SDPA MATH backend reproduces the oracle ``torch.softmax`` in fp32 to
~1e-5, so hardness is already pinned by the exact-math tests; the compiled
parity check at a tight ``atol`` is what guards the structural paths.)

All fixtures compile with ``d_head >= d_qk`` (the single hard constraint the
weight writer enforces).
"""

import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.graph import InputNode
from torchwright.ops.attention_ops import attend_argmin_above_in_bucket
from torchwright.ops.inout_nodes import create_rope_config


def _rope(d_head):
    # The op places its content (d_qk = 2 + nb + nt cols) on the slowest
    # d_head/2 rotary planes, so rope.d_head must match the compile d_head.
    return create_rope_config(d_head=d_head, max_positions=512)


def _onehot_rows(indices, width):
    out = torch.zeros(len(indices), width)
    for i, idx in enumerate(indices):
        out[i, idx] = 1.0
    return out


def _above_table(scores, thresholds):
    out = torch.zeros(len(scores), len(thresholds))
    for i, s in enumerate(scores):
        for c, t in enumerate(thresholds):
            out[i, c] = 1.0 if s > t else 0.0
    return out


def _build(nb, nt, value_width, *, prefix, d_head):
    """Build the op plus its input nodes, with per-test-unique input names."""
    score = InputNode(f"{prefix}_score", 1, value_range=(-100.0, 100.0))
    validity = InputNode(f"{prefix}_validity", 1, value_range=(-2.0, 2.0))
    kb = InputNode(f"{prefix}_kb", nb, value_range=(-2.0, 2.0))
    above = InputNode(f"{prefix}_above", nt, value_range=(-2.0, 2.0))
    qb = InputNode(f"{prefix}_qb", nb, value_range=(-2.0, 2.0))
    th = InputNode(f"{prefix}_th", nt, value_range=(-2.0, 2.0))
    value = InputNode(f"{prefix}_value", value_width, value_range=(-100.0, 100.0))
    out = attend_argmin_above_in_bucket(
        _rope(d_head), score, validity, kb, above, qb, th, value
    )
    return out


def test_baib_adjacent_score_recovered_compiled():
    """Two matching rows whose scores differ by 1, at the full predicate-bonus
    stack: the compiled fp32 head recovers the lower-score row, matching exact
    math everywhere.
    """
    # d_qk = 2 + 3 + 5 = 10; under RoPE the content rides the slowest d_head/2
    # planes, so d_head >= 2*10 = 20, AND d_head must divide d (512) for an
    # integer head count — so the smallest valid even d_head is 32 (was d_head=16
    # pre-RoPE, when the content fit in fewer planes).
    nb, nt, vw = 3, 5, 4
    out = _build(nb, nt, vw, prefix="adj", d_head=32)
    n_pos = 2
    scores = [6, 5]  # adjacent; row 1 is the lower score
    thresholds = [0, 1, 2, 3, 4]
    inputs = dict(
        adj_score=torch.tensor([[float(s)] for s in scores]),
        adj_validity=torch.tensor([[1.0], [1.0]]),
        adj_kb=_onehot_rows([1, 1], nb),
        adj_above=_above_table(scores, thresholds),
        adj_qb=_onehot_rows([1, 1], nb),
        adj_th=_onehot_rows([2, 2], nt),  # threshold > 2; both 5 and 6 above
        adj_value=torch.eye(n_pos, vw),
    )
    # Oracle: at q1 both rows match; smallest score is row 1 (5).
    oracle = reference_eval(out, inputs, n_pos)[out]
    assert torch.allclose(oracle[1], inputs["adj_value"][1], atol=1e-2), oracle[1]
    # Compiled must reproduce the oracle at every node.
    report = probe_graph(
        out,
        input_values=inputs,
        n_pos=n_pos,
        d=512,
        d_head=32,
        atol=1e-3,
    )
    assert report.first_divergent is None, report.format_short()


def test_baib_wide_value_splits_over_heads_compiled():
    """A value wider than d_head forces the V/O split across physical heads;
    the compiled output still matches exact math.
    """
    nb, nt, vw = 2, 3, 20  # d_qk = 7 <= 16/2; d_v = 20 > 16 -> 2 V/O heads
    out = _build(nb, nt, vw, prefix="wide", d_head=16)
    n_pos = 2
    scores = [5, 6]
    thresholds = [0, 1, 2]
    value_in = torch.zeros(n_pos, vw)
    value_in[0, 7] = 9.0  # winner payload in head 0 (cols 0..15)
    value_in[0, 18] = 7.0  # winner ALSO carries a payload in a head-1 column
    value_in[1, 18] = 4.0  # loser's head-1 payload must be suppressed
    inputs = dict(
        wide_score=torch.tensor([[float(s)] for s in scores]),
        wide_validity=torch.tensor([[1.0], [1.0]]),
        wide_kb=_onehot_rows([1, 1], nb),
        wide_above=_above_table(scores, thresholds),
        wide_qb=_onehot_rows([1, 1], nb),
        wide_th=_onehot_rows([0, 0], nt),  # threshold > 0
        wide_value=value_in,
    )
    # q1: smallest score is row 0 (5) -> its payload (9.0 at col 7 in head 0,
    # 7.0 at col 18 in head 1); head 1 must pass 7.0 through AND suppress the
    # loser's 4.0 at the same col 18.
    oracle = reference_eval(out, inputs, n_pos)[out]
    assert torch.allclose(oracle[1], value_in[0], atol=1e-2), oracle[1]
    report = probe_graph(
        out,
        input_values=inputs,
        n_pos=n_pos,
        d=512,
        d_head=16,
        atol=1e-3,
    )
    assert report.first_divergent is None, report.format_short()


def test_baib_all_invalid_compiled_is_finite():
    """All rows invalid (no match): the compiled head returns a finite,
    defined blend — never NaN, never raises — and matches the oracle blend.
    """
    nb, nt, vw = 2, 3, 4
    out = _build(nb, nt, vw, prefix="inv", d_head=16)
    n_pos = 3
    scores = [2, 3, 4]
    thresholds = [0, 1, 2]
    inputs = dict(
        inv_score=torch.tensor([[float(s)] for s in scores]),
        inv_validity=torch.tensor([[-1.0], [-1.0], [-1.0]]),
        inv_kb=_onehot_rows([1, 1, 1], nb),
        inv_above=_above_table(scores, thresholds),
        inv_qb=_onehot_rows([1, 1, 1], nb),
        inv_th=_onehot_rows([0, 0, 0], nt),
        inv_value=torch.eye(n_pos, vw),
    )
    oracle = reference_eval(out, inputs, n_pos)[out]
    assert torch.isfinite(oracle).all(), oracle
    report = probe_graph(
        out,
        input_values=inputs,
        n_pos=n_pos,
        d=512,
        d_head=16,
        atol=1e-3,
    )
    assert report.first_divergent is None, report.format_short()


def test_baib_self_row_only_compiled_is_finite():
    """Minimal causal window (n_pos=1, only the self row visible): the compiled
    head returns a finite value and matches the oracle.
    """
    nb, nt, vw = 2, 3, 4
    out = _build(nb, nt, vw, prefix="self", d_head=16)
    n_pos = 1
    scores = [3]
    thresholds = [0, 1, 2]
    inputs = dict(
        self_score=torch.tensor([[3.0]]),
        self_validity=torch.tensor([[1.0]]),
        self_kb=_onehot_rows([1], nb),
        self_above=_above_table(scores, thresholds),
        self_qb=_onehot_rows([1], nb),
        self_th=_onehot_rows([0], nt),
        self_value=torch.eye(n_pos, vw),
    )
    oracle = reference_eval(out, inputs, n_pos)[out]
    assert torch.isfinite(oracle).all(), oracle
    report = probe_graph(
        out,
        input_values=inputs,
        n_pos=n_pos,
        d=512,
        d_head=16,
        atol=1e-3,
    )
    assert report.first_divergent is None, report.format_short()
