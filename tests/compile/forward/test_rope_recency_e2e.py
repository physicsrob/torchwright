"""Phase 4 — recency end-to-end: the bucket-2 octant ramp, wired.

docs/rope_port_plan.md §8 Phase 4 (validates Phase 1b).  Phase 1b proved the
octant ramp is monotone on synthetic ``(u, v)`` inputs
(``test_recency_ramp_compiled.py``).  Phase 4 builds the **heads** that produce
``(u, v)`` from the rotation and wires the resulting rank into the recency
*selection*, then validates the chain at production difficulty.

Gates proved here (the §8 Phase-1b gate letters):

- **(c) graded head** — the two ``{BOS, REF}`` rotary heads produce
  ``u = sigmoid(M·cos φ) − 0.5`` / ``v = sigmoid(M·sin φ) − 0.5`` to fp32 on the
  compiled path (``test_phase_heads_track_sigmoid``).
- **(d) leakage budget** — the ``{BOS, REF}`` softmax stays effectively 2-key as
  the background key count grows: the head output does not drift from the ideal
  as ``N`` increases (``test_phase_heads_leakage_does_not_grow_with_N``).
- **(a) BOS attendability** — the recency rank is identical between a prefill and
  unbounded cached decode (``test_recency_rank_prefill_decode_identical``).
- **plane sizing** — the chosen recency plane never wraps over the full cache cap
  (``test_recency_plane_seam_safe_to_cap``).
- **selection** — ``attend_most_recent_matching_via_ramp`` returns the
  most-recent content match, resolving adjacent (gap-1) matches
  (``test_selection_picks_most_recent``,
  ``test_selection_gap1_resolves_at_cap_density``).

Cost note: the heads are **full-width** ``d_head`` rotary (the recency plane must
be a real slow frequency on the grid), so those tests compile at ``d_head=256``.
The selection-logic tests feed a synthetic monotone rank input and stay small.
"""

import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph.rope import ROPE_BASE, recency_plane_index
from torchwright.ops.attention_ops import attend_most_recent_matching_via_ramp
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.ops.recency_heads import recency_phase_heads, recency_rank

CAP = 61440  # ONNX cache cap / e1m1 max_positions — size the plane to THIS.
D_HEAD = 256
GAIN = 2.0
SEAM_FRAC = 0.05


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def _pack(module, named, n):
    """Pack a {name: (n, w)} dict into the module's input-tensor column layout."""
    total = sum(w for _, _, w in module._input_specs)
    out = torch.zeros(n, total)
    for name, start, w in module._input_specs:
        if name in named:
            out[:, start : start + w] = named[name]
    return out


def _markers(n):
    """BOS at slot 0, REF at slot 1 (the two always-visible marked tokens)."""
    bos = torch.zeros(n, 1)
    ref = torch.zeros(n, 1)
    bos[0, 0] = 1.0
    ref[1, 0] = 1.0
    return {"bos_marker": bos, "ref_marker": ref}


def _phase(plane, j):
    theta = ROPE_BASE ** (-2.0 * plane / D_HEAD)
    return j * theta + SEAM_FRAC * 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Plane sizing (pure math — no compile)                                        #
# --------------------------------------------------------------------------- #
def test_recency_plane_seam_safe_to_cap():
    """The chosen plane's phase stays clear of the seam over the whole cache cap.

    Past the seam (``φ = 0 mod 2π``) the octant ramp wraps and the recency order
    silently inverts, so the plane must be sized to the cap, not the typical
    frame.
    """
    plane = recency_plane_index(D_HEAD, ROPE_BASE, CAP, seam_frac=SEAM_FRAC)
    lo = _phase(plane, 0)
    hi = _phase(plane, CAP)
    assert lo >= SEAM_FRAC * 2 * math.pi - 1e-9
    assert hi <= (1.0 - SEAM_FRAC) * 2 * math.pi, (
        f"plane {plane} wraps: φ(cap)={hi/(2*math.pi):.4f} turns exceeds the "
        f"{1 - SEAM_FRAC:.2f}-turn seam margin"
    )
    # Sizing to a longer rollout than the grid can support must raise, not
    # silently return a wrapping plane.
    with pytest.raises(ValueError):
        recency_plane_index(D_HEAD, ROPE_BASE, 10 * CAP, seam_frac=0.49)


# --------------------------------------------------------------------------- #
# Graded heads (gate c) + leakage (gate d) — full-width d_head rotary          #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def heads_module():
    pos = create_pos_encoding()
    bos = create_input("bos_marker", 1)
    ref = create_input("ref_marker", 1)
    u, v = recency_phase_heads(bos, ref, d_head=D_HEAD, max_positions=CAP)
    # Concatenate u,v into a 2-wide output so one compile yields both.
    from torchwright.graph import Concatenate

    out = Concatenate([u, v])
    return compile_headless(out, pos, d=1024, d_head=D_HEAD, verbose=False), out


def _eval_heads(module, n):
    m, _ = module
    packed = _pack(m, _markers(n), n).to(m._net.device)
    uv = m(packed).cpu()
    return uv[:, 0], uv[:, 1]


def test_phase_heads_track_sigmoid(heads_module):
    """Gate (c): u,v == sigmoid(M·cos/sin φ) − 0.5 to fp32 on the compiled path.

    Skips j=0 (BOS attending only to itself — the degenerate reference position
    whose own rank is never consumed) and j=1 (REF).
    """
    n = 4096
    u, v = _eval_heads(heads_module, n)
    plane = recency_plane_index(D_HEAD, ROPE_BASE, CAP, seam_frac=SEAM_FRAC)
    js = torch.arange(n, dtype=torch.float64)
    phis = js * (ROPE_BASE ** (-2.0 * plane / D_HEAD)) + SEAM_FRAC * 2 * math.pi
    u_ref = torch.sigmoid(GAIN * torch.cos(phis)) - 0.5
    v_ref = torch.sigmoid(GAIN * torch.sin(phis)) - 0.5
    eu = (u.double() - u_ref).abs()[2:].max().item()
    ev = (v.double() - v_ref).abs()[2:].max().item()
    assert eu < 1e-5, f"cos-head drift {eu:.2e}"
    assert ev < 1e-5, f"sin-head drift {ev:.2e}"


def test_phase_heads_leakage_does_not_grow_with_N(heads_module):
    """Gate (d): the {BOS, REF} softmax stays effectively 2-key — the head's
    deviation from the ideal does NOT grow as the background key count grows.

    Leakage drift is ~N·exp(−L); with the default L≈25 it is far below the
    gap-1 weight signal (~3e-5) even at the cap, so the max deviation at N=4096
    is the same fp32 floor as at N=256.
    """
    plane = recency_plane_index(D_HEAD, ROPE_BASE, CAP, seam_frac=SEAM_FRAC)
    theta = ROPE_BASE ** (-2.0 * plane / D_HEAD)

    def max_dev(n):
        u, _ = _eval_heads(heads_module, n)
        js = torch.arange(n, dtype=torch.float64)
        u_ref = (
            torch.sigmoid(GAIN * torch.cos(js * theta + SEAM_FRAC * 2 * math.pi)) - 0.5
        )
        return (u.double() - u_ref).abs()[2:].max().item()

    small, large = max_dev(256), max_dev(4096)
    assert large < 1e-5, f"leakage at N=4096 is {large:.2e}, above the fp32 floor"
    # 16× more background keys must not blow up the deviation.
    assert (
        large < small * 10 + 1e-6
    ), f"leakage grew with N: {small:.2e} (N=256) -> {large:.2e} (N=4096)"


# --------------------------------------------------------------------------- #
# Full chain: heads -> octant ramp                                             #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rank_module():
    pos = create_pos_encoding()
    bos = create_input("bos_marker", 1)
    ref = create_input("ref_marker", 1)
    rank = recency_rank(bos, ref, d_head=D_HEAD, max_positions=CAP)
    return compile_headless(rank, pos, d=1024, d_head=D_HEAD, verbose=False), rank


def test_recency_rank_monotone_compiled(rank_module):
    """The chained rank (heads -> octant ramp) is strictly increasing in
    absolute position over a contiguous prefill (consumed positions j>=2)."""
    m, _ = rank_module
    n = 4096
    packed = _pack(m, _markers(n), n).to(m._net.device)
    r = m(packed).reshape(-1).cpu()[2:]
    steps = r[1:] - r[:-1]
    assert bool(
        (steps > 0).all()
    ), f"{int((steps <= 0).sum())} non-increasing steps; min {steps.min():.3e}"


def test_recency_rank_prefill_decode_identical(rank_module):
    """Gate (a): the recency rank is identical between prefill and unbounded
    cached decode — BOS stays attendable, K is stored already-rotated."""
    m, _ = rank_module
    n = 64
    device = m._net.device
    named = _markers(n)
    packed = _pack(m, named, n).to(device)
    full = m(packed).reshape(-1).cpu()

    past = m.empty_past()
    decoded = []
    for t in range(n):
        row = _pack(m, {k: v[t : t + 1] for k, v in named.items()}, 1).to(device)
        out_t, past = m.step(row, past)
        decoded.append(out_t.reshape(-1).item())
    assert torch.allclose(full, torch.tensor(decoded), atol=1e-4), (
        full[:6],
        torch.tensor(decoded)[:6],
    )


def test_recency_rank_probe_compiled(rank_module):
    """probe_compiled: the compiled chain matches its graph oracle everywhere
    (the ramp's PL ops included) — no divergent node."""
    _, rank = rank_module
    pos = create_pos_encoding()
    n = 512
    compiled = compile_headless(rank, pos, d=1024, d_head=D_HEAD, verbose=False)
    report = probe_compiled(compiled, rank, _markers(n), n, atol=1e-2)
    assert report.first_divergent is None, report.format_short()


# --------------------------------------------------------------------------- #
# Selection logic (synthetic monotone rank — cheap, small d_head)             #
# --------------------------------------------------------------------------- #
def _selection_module(W, d_v, match_gain, rank_gain):
    pos = create_pos_encoding()
    qv = create_input("qv", W)
    kv = create_input("kv", W)
    val = create_input("val", d_v)
    rank = create_input("rank", 1)
    sel = attend_most_recent_matching_via_ramp(
        qv, kv, val, rank, match_gain=match_gain, rank_gain=rank_gain
    )
    return compile_headless(sel, pos, d=256, verbose=False), sel


def test_selection_picks_most_recent():
    """attend_most_recent_matching_via_ramp selects the most-recent content
    match.  Well-separated matches concentrate hard; an adjacent (gap-1) pair at
    the worst-case (cap-min) ramp step is still picked correctly — but only
    ~0.98-hard, the octant-boundary concentration dip that ``G=2e5`` accepts
    (8-logit gap at the typical step, ~4 at a boundary; see
    test_selection_gap1_resolves_at_cap_density).
    """
    W, d_v, n = 2, 1, 16
    match_gain, rank_gain = 200.0, 2.0e5
    m, _ = _selection_module(W, d_v, match_gain, rank_gain)
    device = m._net.device

    # Type A = [1,0], type B = [0,1].  Query everywhere = A.
    qv = torch.tensor([[1.0, 0.0]]).repeat(n, 1)
    kv = torch.tensor([[0.0, 1.0]]).repeat(n, 1)  # default: type B (no match)
    val = (100.0 + torch.arange(n, dtype=torch.float32)).unsqueeze(1)
    # Matching keys (type A) at positions 5, 10, 11 (10 & 11 are gap-1).
    for p in (5, 10, 11):
        kv[p] = torch.tensor([1.0, 0.0])
    # Stress the WORST case: a uniform rank step at the cap-density minimum.
    rank = (torch.arange(n, dtype=torch.float32) * 2.0e-5).unsqueeze(1)

    packed = _pack(m, {"qv": qv, "kv": kv, "val": val, "rank": rank}, n).to(device)
    out = m(packed).reshape(-1).cpu()

    # Well-separated: position 10 sees matches at 5 and 10 (5-step recency gap ->
    # ~20 logits) and selects 10 hard.
    assert abs(out[10].item() - val[10, 0].item()) < 1e-2, out[10]
    # Gap-1 at the worst step: the last query picks 11 over the adjacent 10
    # (output unambiguously on the newer side of the midpoint), ~0.98-hard.
    midpoint = 0.5 * (val[10, 0].item() + val[11, 0].item())
    assert out[-1].item() > midpoint, (out[-1].item(), midpoint)
    assert abs(out[-1].item() - val[11, 0].item()) < 0.05, out[-1]


def test_selection_gap1_resolves_at_cap_density():
    """The rank-gain ``G`` resolves adjacent positions at the cap's φ-density and
    content still dominates recency — the two ``attend_most_recent_matching``
    invariants, transposed from the counter to the ramp.

    Uses the analytic octant ramp at the production per-token step (the compiled
    ramp's monotonicity at this density is ``test_recency_ramp_compiled.py``).
    """
    from scripts.rope_octant_assembly import ramp as analytic_ramp

    plane = recency_plane_index(D_HEAD, ROPE_BASE, CAP, seam_frac=SEAM_FRAC)
    theta = ROPE_BASE ** (-2.0 * plane / D_HEAD)
    js = torch.arange(0, CAP + 1)
    phis = (js.double() * theta + SEAM_FRAC * 2 * math.pi).numpy()
    r = torch.from_numpy(analytic_ramp(phis))
    steps = r[1:] - r[:-1]
    rank_range = (r.max() - r.min()).item()
    min_step = steps.min().item()

    G = 2.0e5
    # (a) adjacent-position recency gap resolvable (clean softmax concentration).
    adj_gap = G * min_step
    assert adj_gap > 3.0, f"adjacent recency gap {adj_gap:.2f} logits too small"
    # (b) content dominates: match_gain·dot_gap must exceed G·rank_range.
    needed_match_gain = G * rank_range  # one-hot dot gap = 1
    assert needed_match_gain < 6.0e5, (
        f"content-dominance needs match_gain > {needed_match_gain:.0f}; "
        f"out of the documented band"
    )


def test_selection_end_to_end_with_real_rank():
    """Integration: heads -> ramp -> selection, all compiled together, picks the
    most-recent match.  Small N (the recency machinery is full-width rotary)."""
    W, d_v, n = 2, 1, 16
    pos = create_pos_encoding()
    bos = create_input("bos_marker", 1)
    ref = create_input("ref_marker", 1)
    qv = create_input("qv", W)
    kv = create_input("kv", W)
    val = create_input("val", d_v)
    rank = recency_rank(bos, ref, d_head=D_HEAD, max_positions=CAP)
    # Content-dominance bound: match_gain·dot_gap > rank_gain·rank_range.  The
    # degenerate reference positions (BOS/REF attend only to themselves -> rank
    # 0.5, a high outlier vs the ~0.15 interior) make the real rank range ~0.35
    # here, so match_gain must clear rank_gain·0.35 = 7e4 — this is exactly the
    # bound that keeps a content-mismatched BOS from winning on its outlier rank.
    sel = attend_most_recent_matching_via_ramp(
        qv, kv, val, rank, match_gain=1.0e6, rank_gain=2.0e5
    )
    m = compile_headless(sel, pos, d=1024, d_head=D_HEAD, verbose=False)
    device = m._net.device

    named = _markers(n)
    named["qv"] = torch.tensor([[1.0, 0.0]]).repeat(n, 1)
    named["kv"] = torch.tensor([[0.0, 1.0]]).repeat(n, 1)
    named["val"] = (100.0 + torch.arange(n, dtype=torch.float32)).unsqueeze(1)
    for p in (5, 10, 11):
        named["kv"][p] = torch.tensor([1.0, 0.0])

    packed = _pack(m, named, n).to(device)
    out = m(packed).reshape(-1).cpu()
    # Last query selects the most-recent type-A key (position 11), beating the
    # adjacent older match at 10 and the content-mismatched BOS (outlier rank).
    assert abs(out[-1].item() - named["val"][11, 0].item()) < 1e-2, out[-1]
