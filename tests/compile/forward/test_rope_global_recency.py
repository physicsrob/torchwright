"""Phase 7 — global recency on the compiled path.

docs/rope_port_plan.md Phase 7.  The oracle-level selection / position recovery
live in ``tests/ops/test_global_recency.py``; this file proves the same mechanism
on the **compiled** transformer:

- ``attend_most_recent_globally`` picks the most recent match over a sparse
  prefill (stride-10 matches, logit gap=10 → >99.99% one-hot);
- ``probe_compiled`` agrees with the graph oracle everywhere (no divergent node);
- the recency selection is identical between a prefill and an unbounded cached
  decode (BOS/K-rotation cache invariant).

Cost: the global recency head is full-width d_head=256 rotary (needs the
slowest plane for the BOS-weight mechanism).  n is kept modest to stay fast.
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.attention_ops import attend_most_recent_globally
from torchwright.ops.relu.global_recency import global_position_from_bos
from torchwright.ops.inout_nodes import create_input, create_rope_config

D_HEAD = 256
CAP = 61440

# Partial-rotary variant: content rides the NoPE tail [d_rot:d_head], the position
# tiebreak rides the slowest rotated plane d_rot/2-1.  d_head halved vs the
# full-rotary head (the tail dissolves the slow-plane d_head budget).
D_HEAD_PARTIAL = 128
D_ROT_PARTIAL = 64


def _pack(module, named, n):
    total = sum(w for _, _, w in module._input_specs)
    out = torch.zeros(n, total)
    for name, start, w in module._input_specs:
        if name in named:
            out[:, start : start + w] = named[name]
    return out


def _global_recency_graph(d_head=D_HEAD, d_rot=None):
    rope = create_rope_config(d_head=d_head, max_positions=CAP, d_rot=d_rot)
    bos_ind = create_input("bos", 1)
    qv = create_input("qv", 8)
    kv = create_input("kv", 8)
    val = create_input("val", 1)
    gpos = global_position_from_bos(rope, bos_ind)
    sel = attend_most_recent_globally(rope, qv, kv, gpos, val)
    return sel


def _e8_inputs(n, match_positions):
    target, other = index_to_vector(3), index_to_vector(0)
    bos = torch.zeros(n, 1)
    bos[0, 0] = 1.0
    kv = torch.stack([target if p in match_positions else other for p in range(n)])
    qv = target.unsqueeze(0).expand(n, -1).contiguous()
    val = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    return {"bos": bos, "qv": qv, "kv": kv, "val": val}


def test_compiled_picks_most_recent_globally():
    """Sparse-match prefill, compiled path: most recent match selected.

    Stride-10 matches in a 60-position prefill.  Adjacent match logit gap is
    recency_scale × 10 = 10, giving >99.99% weight to the most recent key.
    """
    sel = _global_recency_graph()
    stride = 10
    n = 60
    matches = set(range(0, n, stride))
    named = _e8_inputs(n, matches)
    m = compile_headless(sel, d=2048, d_head=D_HEAD, verbose=False)
    out = m(_pack(m, named, n).to(m._net.device)).reshape(-1).cpu()

    for p in range(1, n):
        most_recent = max(mp for mp in matches if mp <= p)
        assert abs(out[p].item() - float(most_recent)) < 0.5, (
            f"pos {p}: expected {most_recent}, got {out[p].item():.2f}"
        )


def test_probe_compiled_parity():
    """Compiled global-recency head matches the graph oracle everywhere."""
    sel = _global_recency_graph()
    n = 48
    stride = 8
    matches = set(range(0, n, stride))
    named = _e8_inputs(n, matches)
    compiled = compile_headless(sel, d=2048, d_head=D_HEAD, verbose=False)
    report = probe_compiled(compiled, sel, named, n, atol=1.0)
    assert report.first_divergent is None, report.format_short()


def test_prefill_equals_cached_decode():
    """Global recency selection is identical between a prefill and an unbounded
    cached decode — the BOS/K-already-rotated cache invariant (§5)."""
    sel = _global_recency_graph()
    stride = 8
    n = 40
    matches = set(range(0, n, stride))
    named = _e8_inputs(n, matches)
    m = compile_headless(sel, d=2048, d_head=D_HEAD, verbose=False)
    device = m._net.device

    full = m(_pack(m, named, n).to(device)).reshape(-1).cpu()

    past = m.empty_past()
    decoded = []
    for t in range(n):
        row = _pack(m, {k: v[t : t + 1] for k, v in named.items()}, 1).to(device)
        out_t, past = m.step(row, past)
        decoded.append(out_t.reshape(-1).item())

    assert torch.allclose(full, torch.tensor(decoded), atol=0.5), (
        full[:8],
        torch.tensor(decoded)[:8],
    )


def test_compiled_picks_most_recent_globally_larger_n():
    """Compiled path at n=500 — extends coverage beyond the n=60 basic test.

    Full-cap validation (n=61440) is analytic-only
    (``scripts/rope_global_recency_validate.py``).  This test exercises the
    compiled transformer over a longer prefill so the BOS-weight PWL inversion
    is exercised at positions where fp32 accumulation error is non-trivial.
    """
    sel = _global_recency_graph()
    stride = 50
    n = 500
    matches = set(range(0, n, stride))
    named = _e8_inputs(n, matches)
    m = compile_headless(sel, d=2048, d_head=D_HEAD, verbose=False)
    out = m(_pack(m, named, n).to(m._net.device)).reshape(-1).cpu()

    for p in range(1, n):
        most_recent = max(mp for mp in matches if mp <= p)
        assert abs(out[p].item() - float(most_recent)) < 0.5, (
            f"pos {p}: expected {most_recent}, got {out[p].item():.2f}"
        )


def test_partial_global_recency_placement():
    """Under partial rotary the global-recency head splits placement: the 8-wide
    content rides the NoPE tail [d_rot:d_rot+8] and the position tiebreak rides the
    slowest rotated plane d_rot/2-1; the head carries rope_d_rot == d_rot."""
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.graph.attn import Attn

    sel = _global_recency_graph(d_head=D_HEAD_PARTIAL, d_rot=D_ROT_PARTIAL)
    attns = [n for n in get_ancestor_nodes({sel}) if isinstance(n, Attn)]
    # The recency head is the one with the widest content query (W+1 input rows).
    rec = max(attns, key=lambda a: a.d_query_in)
    assert rec.rope_d_rot == D_ROT_PARTIAL
    qm = rec.query_matrix
    # Content on the tail: columns [d_rot:d_rot+8] carry the match_gain rows.
    assert qm[:, D_ROT_PARTIAL : D_ROT_PARTIAL + 8].abs().sum() > 0.0
    # Position tiebreak on the slowest rotated plane d_rot/2-1.
    assert qm[:, D_ROT_PARTIAL // 2 - 1].abs().sum() > 0.0


def test_partial_compiled_picks_most_recent_globally():
    """Partial-rotary global recency (d_head=128, d_rot=64): most recent match
    selected over a stride-10 prefill, compiled path."""
    sel = _global_recency_graph(d_head=D_HEAD_PARTIAL, d_rot=D_ROT_PARTIAL)
    stride = 10
    n = 60
    matches = set(range(0, n, stride))
    named = _e8_inputs(n, matches)
    m = compile_headless(sel, d=2048, d_head=D_HEAD_PARTIAL, verbose=False)
    out = m(_pack(m, named, n).to(m._net.device)).reshape(-1).cpu()

    for p in range(1, n):
        most_recent = max(mp for mp in matches if mp <= p)
        assert abs(out[p].item() - float(most_recent)) < 0.5, (
            f"pos {p}: expected {most_recent}, got {out[p].item():.2f}"
        )


def test_partial_probe_compiled_parity():
    """Partial-rotary global-recency head matches the graph oracle everywhere."""
    sel = _global_recency_graph(d_head=D_HEAD_PARTIAL, d_rot=D_ROT_PARTIAL)
    n = 48
    matches = set(range(0, n, 8))
    named = _e8_inputs(n, matches)
    compiled = compile_headless(sel, d=2048, d_head=D_HEAD_PARTIAL, verbose=False)
    report = probe_compiled(compiled, sel, named, n, atol=1.0)
    assert report.first_divergent is None, report.format_short()


def test_partial_prefill_equals_cached_decode():
    """Partial-rotary global recency is identical between prefill and unbounded
    cached decode (the NoPE tail and rotated position column both cache cleanly)."""
    sel = _global_recency_graph(d_head=D_HEAD_PARTIAL, d_rot=D_ROT_PARTIAL)
    n = 40
    matches = set(range(0, n, 8))
    named = _e8_inputs(n, matches)
    m = compile_headless(sel, d=2048, d_head=D_HEAD_PARTIAL, verbose=False)
    device = m._net.device

    full = m(_pack(m, named, n).to(device)).reshape(-1).cpu()
    past = m.empty_past()
    decoded = []
    for t in range(n):
        row = _pack(m, {k: v[t : t + 1] for k, v in named.items()}, 1).to(device)
        out_t, past = m.step(row, past)
        decoded.append(out_t.reshape(-1).item())

    assert torch.allclose(full, torch.tensor(decoded), atol=0.5), (
        full[:8],
        torch.tensor(decoded)[:8],
    )


def test_smoothed_position_compiled_partial_rotary():
    """``global_position_from_bos(smoothed=True)`` on the compiled path.

    The smoothed position is 2 × an exact uniform causal mean of the raw
    recovery (zero-Q/K head, ×2 folded into O).  It must track t and keep
    its adjacent steps near 1 — the property the recency tiebreak's
    softmax hardness rides on (the raw recovery's compiled fp32 wander
    breaks this at long rollouts; see the op docstring).
    """
    rope = create_rope_config(
        d_head=D_HEAD_PARTIAL, max_positions=CAP, d_rot=D_ROT_PARTIAL
    )
    bos_ind = create_input("bos", 1)
    pos = global_position_from_bos(rope, bos_ind, smoothed=True)

    n = 192
    bos = torch.zeros(n, 1)
    bos[0, 0] = 1.0
    m = compile_headless(pos, d=2048, d_head=D_HEAD_PARTIAL, verbose=False)
    out = m(_pack(m, {"bos": bos}, n).to(m._net.device)).reshape(-1).cpu()

    t = torch.arange(n, dtype=out.dtype)
    assert float((out - t).abs().max()) < 0.5, (
        f"max |pos - t| = {float((out - t).abs().max()):.4f}"
    )
    steps = out[1:] - out[:-1]
    assert float(steps.min()) > 0.8, f"min step {float(steps.min()):.4f}"
    assert float(steps.max()) < 1.2, f"max step {float(steps.max()):.4f}"


def test_smoothed_position_prefill_equals_cached_decode():
    """Smoothed position parity between prefill and unbounded cached decode."""
    rope = create_rope_config(
        d_head=D_HEAD_PARTIAL, max_positions=CAP, d_rot=D_ROT_PARTIAL
    )
    bos_ind = create_input("bos", 1)
    pos = global_position_from_bos(rope, bos_ind, smoothed=True)

    n = 96
    bos = torch.zeros(n, 1)
    bos[0, 0] = 1.0
    m = compile_headless(pos, d=2048, d_head=D_HEAD_PARTIAL, verbose=False)
    device = m._net.device

    full = m(_pack(m, {"bos": bos}, n).to(device)).reshape(-1).cpu()
    past = m.empty_past()
    decoded = []
    for t in range(n):
        row = _pack(m, {"bos": bos[t : t + 1]}, 1).to(device)
        out_t, past = m.step(row, past)
        decoded.append(out_t.reshape(-1).item())

    assert torch.allclose(full, torch.tensor(decoded), atol=0.25), (
        full[:8],
        torch.tensor(decoded)[:8],
    )
