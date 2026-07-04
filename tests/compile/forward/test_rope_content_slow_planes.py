"""Phase 3: content selection survives the global rotation on slow planes.

``docs/rope_port_plan.md`` §3/§8.  The content-selection heads
(``attend_argmin``/``argmax``/``_where``/``dot``/``_bucket`` …) are
position-independent today — constant query, content in K.  In the RoPE
end state every head's Q/K is rotated by absolute position, so a content
logit ``Σ_c q_c·k_c`` becomes ``Σ_c q_c·k_c·cos((i−j)·θ_{p_c})``.  Putting
each content column on a **slow** plane (tiny θ) keeps ``cos((i−j)·θ) ≈ 1``
over the rollout, so the match stays effectively position-free.  This is the
capability test for that claim, calibrated (per §9) to the real worst-case
difficulty: the ~42k production distance and each head's own documented gap.

Two layers of validation:
  * **Selection-at-distance (the attenuation gate).**  A query at a large
    absolute position vs keys spanning 0..42k, computed as a single logit row
    (a full n_pos×n_pos attention matrix at 42k would OOM).  ``_rotary_logits``
    mirrors ``Attn.compute``'s rotary path exactly and is anchored to it on a
    small case, so the row is a faithful 42k proxy.
  * **Compiled parity.**  A rotary content head built via ``rotary_content_head``
    compiles and matches its exact-math oracle (``probe_compiled``) — the
    rotation is correctly wired through the compiler at full-width ``d_head``.

Base is locked at LLaMA3's 5e5; the slowest-plane attenuation at 42k (~0.9965)
is set by *base*, not d_head, so d_head only buys plane count.  d_head=256
(the recency analyses' grid) gives ample slow planes for the widest content.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import Attn, LiteralValue
from torchwright.graph.rope import (
    ROPE_BASE,
    apply_rope,
    place_on_slow_planes,
    rope_cos_sin,
    rotary_content_head,
)
from torchwright.ops.inout_nodes import create_input

D = 512
D_HEAD = 256
BASE = ROPE_BASE
PROD = 42000  # production rollout distance (§9)

# Head constants (mirrors torchwright/ops/attention_ops.py).
_QUERY_GAIN = 8.0
_MAX_SCORE_ABS = 120.0
_VALIDITY_BONUS = 256.0


def _rotary_logits(query_matrix, key_matrix, q_in, k_in, q_pos, k_positions):
    """Logit row of a content head made rotary-on-slow-planes: a query at
    absolute ``q_pos`` against keys at ``k_positions``.  Mirrors the rotary path
    of ``Attn.compute`` (place content on slow planes, project, rotate by
    absolute position, dot) without materialising an n_pos×n_pos matrix."""
    qm = place_on_slow_planes(query_matrix, D_HEAD)
    km = place_on_slow_planes(key_matrix, D_HEAD)
    Q = q_in @ qm  # (1, d_head)
    K = k_in @ km  # (nkeys, d_head)
    cq, sq = rope_cos_sin(torch.tensor([q_pos]), D_HEAD, BASE)
    ck, sk = rope_cos_sin(torch.tensor(k_positions), D_HEAD, BASE)
    Qr = apply_rope(Q, cq.to(Q.dtype), sq.to(Q.dtype))
    Kr = apply_rope(K, ck.to(K.dtype), sk.to(K.dtype))
    return (Qr @ Kr.t()).squeeze(0)


def test_rotary_logits_anchored_to_attn_compute():
    """``_rotary_logits`` equals ``Attn.compute`` on a small case, so using it
    as the 42k-distance proxy is faithful (it can't be exercised at 42k via the
    full matrix without OOM)."""
    W = 4
    query_matrix = _QUERY_GAIN * torch.eye(1, W)  # score in col 0
    key_matrix = torch.zeros(1, W)
    key_matrix[0, 0] = 1.0
    n_pos = 12
    score = create_input("score", 1)
    one = LiteralValue(torch.tensor([1.0]))
    head = rotary_content_head(
        one, score, score, query_matrix, key_matrix, d_head=D_HEAD
    )

    # Random distinct scores; the head's argmax-of-score over the causal window.
    svals = torch.tensor([[float(v)] for v in [3, 9, 1, 7, 5, 2, 8, 4, 6, 0, 11, 10]])
    out = head.compute(n_pos, {"score": svals}).squeeze(1)  # (n_pos,) selected score

    # Reconstruct the last query's logit row directly and check argmax agrees.
    q_pos = n_pos - 1
    row = _rotary_logits(
        query_matrix,
        key_matrix,
        torch.ones(1, 1),
        svals,
        q_pos,
        list(range(n_pos)),
    )
    assert int(row.argmax()) == int(svals.squeeze(1).argmax())
    # The compute() output at the last position is that winning score.
    assert torch.allclose(out[-1], svals.squeeze(1).max(), atol=1e-3)


def test_dot_match_selects_at_production_distance():
    """attend_argmax_dot shape (W=8 one-hot, match_gain=200): a matching key at
    the far end of a 42k rollout beats non-matching keys crowded at the near
    end — content match survives the rotation with a wide margin."""
    W, MG = 8, 200.0
    query_matrix = MG * torch.eye(W)
    key_matrix = torch.eye(W)

    q_in = torch.zeros(1, W)
    q_in[0, 3] = 1.0  # query class 3
    nd = 16
    k_in = torch.zeros(nd, W)
    k_in[0, 3] = 1.0  # winner matches, placed FAR (key pos 0)
    nonmatch = [c for c in range(W) if c != 3]
    for k in range(1, nd):
        k_in[k, nonmatch[k % len(nonmatch)]] = 1.0  # distractors, never class 3
    k_pos = [0] + [PROD - k for k in range(1, nd)]  # distractors near the query

    row = _rotary_logits(query_matrix, key_matrix, q_in, k_in, PROD, k_pos)
    margin = row[0] - row[1:].max()
    # On-match dot is MG=200; off-match is 0.  Attenuation barely dents it.
    assert margin > 150.0, f"dot-match margin too small at 42k: {margin:.2f}"


def test_fine_score_resolution_at_production_distance():
    """The binding gate: a unit score delta (gain=8) must still resolve when the
    winner is at the far end (most attenuated) and the runner-up is adjacent to
    the query (un-attenuated) — the worst case for the slow-plane cosine.
    Tested at the max score magnitude (``_MAX_SCORE_ABS``)."""
    query_matrix = torch.tensor([[_QUERY_GAIN]])  # (1,1)
    key_matrix = torch.tensor([[1.0]])  # argmax: K col0 = +score
    q_in = torch.ones(1, 1)

    s_win = _MAX_SCORE_ABS  # 120
    s_run = _MAX_SCORE_ABS - 1.0  # unit delta
    k_in = torch.tensor([[s_win], [s_run]])
    k_pos = [0, PROD - 1]  # winner FAR (pos 0), runner adjacent to query

    row = _rotary_logits(query_matrix, key_matrix, q_in, k_in, PROD, k_pos)
    margin = row[0] - row[1]
    # Nominal margin is _QUERY_GAIN=8; base-set attenuation at 42k on the max
    # score range erodes it to ~4-5 but it stays decisively positive.
    assert margin > 2.0, f"unit-delta score margin collapsed at 42k: {margin:.3f}"
    # exp(margin) concentration: still a clean pick.
    assert torch.exp(margin) > 7.0


def test_bucket_rendezvous_selects_at_production_distance():
    """Bucket-equality rendezvous (one-hot, bonus=256): a key whose bucket
    matches the query — at the far end — beats non-matching keys at the near end.
    Matching is what gates validity; the per-key score (separate slow plane)
    breaks ties among matches."""
    nb = 12
    query_matrix = _VALIDITY_BONUS * torch.eye(nb)
    key_matrix = torch.eye(nb)

    q_in = torch.zeros(1, nb)
    q_in[0, 5] = 1.0  # query bucket 5
    nd = 14
    k_in = torch.zeros(nd, nb)
    k_in[0, 5] = 1.0  # matching bucket, FAR
    nonmatch = [b for b in range(nb) if b != 5]
    for k in range(1, nd):
        k_in[k, nonmatch[k % len(nonmatch)]] = 1.0  # non-matching buckets, near
    k_pos = [0] + [PROD - k for k in range(1, nd)]

    row = _rotary_logits(query_matrix, key_matrix, q_in, k_in, PROD, k_pos)
    margin = row[0] - row[1:].max()
    assert margin > 200.0, f"bucket rendezvous margin too small at 42k: {margin:.2f}"


def test_rotary_content_head_compiled_matches_oracle():
    """A rotary content head built via ``rotary_content_head`` compiles and
    matches its exact-math oracle — the rotation is correctly wired through the
    compiler at full-width d_head (the ONNX/HF full-width rotary contract)."""
    W = 4
    query_matrix = _QUERY_GAIN * torch.eye(1, W)
    key_matrix = torch.zeros(1, W)
    key_matrix[0, 0] = 1.0  # argmax of score (K col0 = +score)

    score = create_input("score", 1)
    one = LiteralValue(torch.tensor([1.0]))
    head = rotary_content_head(
        one, score, score, query_matrix, key_matrix, d_head=D_HEAD
    )

    n_pos = 16
    svals = torch.tensor([[float(v)] for v in range(n_pos)])  # strictly increasing
    compiled = compile_headless(head, d=D, d_head=D_HEAD, verbose=False)
    report = probe_compiled(compiled, head, {"score": svals}, n_pos, atol=1e-2)
    assert report.first_divergent is None, report.format_short()

    # argmax over the causal window of an increasing score is the current pos.
    # CompiledHeadless.__call__ takes the flat input tensor (one input, width 1).
    with torch.no_grad():
        out = compiled(svals)
    assert torch.allclose(out.squeeze(1), svals.squeeze(1), atol=1e-2)
