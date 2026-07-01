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
from torchwright.ops.attention_ops import get_prev_value
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
