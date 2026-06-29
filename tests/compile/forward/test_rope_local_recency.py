"""Phase 6 — local recency on the compiled path.

docs/rope_port_plan.md Phase 6.  The op-level lobe shape / breakdown / oracle
selection live in ``tests/ops/test_local_recency.py``; this file proves the same
intrinsic rotary-lobe recency on the **compiled** transformer:

- ``attend_most_recent_matching`` picks the immediate predecessor over a prefill
  (the nearest of many in-window candidates) with hard softmax concentration;
- ``probe_compiled`` agrees with the graph oracle everywhere (no divergent node);
- the recency selection is identical between a prefill and an unbounded cached
  decode (BOS/K-rotation cache invariant);
- ``get_prev_value`` latches the most-recent true position.

Cost: the recency lobe rides real slow frequencies, so the head is full-width
``d_head=256`` rotary (the recency band 12..96 must be disjoint from the content
slow-planes).  N is kept modest.
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.attention_ops import attend_most_recent_matching, get_prev_value
from torchwright.ops.inout_nodes import create_input, create_rope_config

D_HEAD = 256
CAP = 61440


def _pack(module, named, n):
    total = sum(w for _, _, w in module._input_specs)
    out = torch.zeros(n, total)
    for name, start, w in module._input_specs:
        if name in named:
            out[:, start : start + w] = named[name]
    return out


def _matcher():
    rope = create_rope_config(d_head=D_HEAD, max_positions=CAP)
    qv = create_input("qv", 8)
    kv = create_input("kv", 8)
    val = create_input("val", 1)
    sel = attend_most_recent_matching(rope, qv, kv, val, exclude_self=True)
    return sel


def _e8_inputs(n, match_positions):
    target, other = index_to_vector(3), index_to_vector(0)
    kv = torch.stack([target if p in match_positions else other for p in range(n)])
    qv = target.unsqueeze(0).expand(n, -1).contiguous()
    val = (100.0 + torch.arange(n, dtype=torch.float32)).unsqueeze(1)
    return {"qv": qv, "kv": kv, "val": val}


def test_compiled_picks_immediate_predecessor():
    """All-matching prefill, exclude_self: each query selects its immediate
    predecessor (Δ=1), the nearest of up to N-1 in-window candidates, hard."""
    sel = _matcher()
    m = compile_headless(sel, d=1024, d_head=D_HEAD, verbose=False)
    n = 96
    named = _e8_inputs(n, set(range(n)))
    out = m(_pack(m, named, n).to(m._net.device)).reshape(-1).cpu()
    for j in range(1, n):
        assert abs(out[j].item() - named["val"][j - 1, 0].item()) < 1e-2, (
            j,
            out[j].item(),
        )


def test_probe_compiled_parity():
    """The compiled recency head matches its graph oracle everywhere."""
    sel = _matcher()
    n = 64
    compiled = compile_headless(sel, d=1024, d_head=D_HEAD, verbose=False)
    report = probe_compiled(compiled, sel, _e8_inputs(n, set(range(n))), n, atol=1e-2)
    assert report.first_divergent is None, report.format_short()


def test_prefill_equals_cached_decode():
    """Recency selection is identical between a prefill and an unbounded cached
    decode — the BOS/K-already-rotated cache invariant (§5)."""
    sel = _matcher()
    m = compile_headless(sel, d=1024, d_head=D_HEAD, verbose=False)
    n = 48
    named = _e8_inputs(n, set(range(n)))
    device = m._net.device
    full = m(_pack(m, named, n).to(device)).reshape(-1).cpu()

    past = m.empty_past()
    decoded = []
    for t in range(n):
        row = _pack(m, {k: v[t : t + 1] for k, v in named.items()}, 1).to(device)
        out_t, past = m.step(row, past)
        decoded.append(out_t.reshape(-1).item())
    assert torch.allclose(full, torch.tensor(decoded), atol=1e-2), (
        full[:6],
        torch.tensor(decoded)[:6],
    )


def test_get_prev_value_latch_compiled():
    """get_prev_value latches the most-recent true position on the compiled path;
    a single far trigger still latches (content gate dominates the lobe)."""
    rope = create_rope_config(d_head=D_HEAD, max_positions=CAP)
    value = create_input("value", 1)
    cond = create_input("cond", 1)
    out = get_prev_value(rope, value, cond)
    m = compile_headless(out, d=1024, d_head=D_HEAD, verbose=False)

    n = 64
    value_in = (10.0 + torch.arange(n, dtype=torch.float32)).unsqueeze(1)
    cond_in = -torch.ones(n, 1)
    cond_in[3, 0] = 1.0
    packed = _pack(m, {"value": value_in, "cond": cond_in}, n).to(m._net.device)
    res = m(packed).reshape(-1).cpu()
    for p in range(3, n):
        assert abs(res[p].item() - 13.0) < 1e-2, (p, res[p].item())
