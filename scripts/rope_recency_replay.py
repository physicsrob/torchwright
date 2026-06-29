"""Replay candidate recency readouts against the real renderer selections.

Stage 2 of the RoPE-port recency derisk (docs/rope_port_plan.md). The
position-attention instrumentation log
(torchwright_doom/scripts/position_attention_log_full.jsonl) records every
real `attend_most_recent_matching` selection, split into a frozen *content*
logit per candidate and the replaceable *position* logit. The current scheme
adds `8 * key_pos`; this script asks, for a candidate position readout
`R(key_pos)`: does `argmax_k (content_logit_k + G * R(k))` reproduce the real
selection? Pass = 0 flips (every flip is a wrong pixel).

KEY MODELING FACT (post-2026-06 softmax-is-exact finding): the compiled
transformer computes softmax EXACTLY (SDPA MATH backend, fp32). So a readout
read straight from a softmax weight carries only ~fp32 noise (~1e-7), not the
~1.5e-4 PL-softmax noise the earlier R12 estimate assumed. The dominant risk
is therefore NOT weight noise but the readout's *shape* — does its per-token
resolution (slope) stay above the noise floor everywhere a tight gap occurs?
A single cosine collapses to zero slope at the ends of its half-turn; a
uniform (atan2-class) readout does not.

Each position's recency signal is computed once (when that token was current)
and stored, so noise is ONE draw per position, fixed across all later reads.

Run:  python3 scripts/rope_recency_replay.py [path-to-compact.bin]
The compact.bin is produced by the extractor (see the RoPE plan); each record:
  query_pos(i32) n_pos(i32) selected(i32) decisive(i32) n_cand(i32)
  then n_cand * (key_pos(i32), content_logit(f32))
"""

from __future__ import annotations

import math
import struct
import sys

import numpy as np

# Production sizing: the rotary plane is sized so it never wraps over the
# whole rollout. max_positions in e1m1 is 61440.
MAX_POSITIONS = 61440


def load(path: str):
    """Yield (query_pos, selected, decisive, cand_pos[], cand_clogit[])."""
    with open(path, "rb") as fh:
        buf = fh.read()
    off = 0
    rows = []
    n = len(buf)
    while off < n:
        qp, npos, sel, dec, nc = struct.unpack_from("<5i", buf, off)
        off += 20
        pos = np.empty(nc, dtype=np.int64)
        clog = np.empty(nc, dtype=np.float64)
        for i in range(nc):
            kp, cl = struct.unpack_from("<if", buf, off)
            off += 8
            pos[i] = kp
            clog[i] = cl
        rows.append((qp, sel, dec, pos, clog))
    return rows


# fp32 absolute noise on a softmax weight near 0.5 (exact-softmax floor):
# weight ~0.5, fp32 rel ~1.2e-7 -> ~6e-8 abs.
SIGMA_W = 6e-8
G_M = 2.0  # gain*amplitude; mid-sigmoid operating point


def _sigmoid_prime(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 - s)


def build_readouts(max_pos: int, seed: int):
    """Per-position stored recency signal R[k] under a SLOPE-AWARE noise model.

    Each scheme reads softmax weight(s) with fp32 noise SIGMA_W. The error in
    the recovered POSITION is that weight noise divided by the local slope
    d(weight)/d(position) of the readout. Single-cosine's slope collapses to
    zero at the ends of its turn (sin(phase)->0); the octant-select readout
    always uses whichever of cos/sin head is steep, so its slope never drops
    below ~sin(45deg) and its position error stays uniform.

    Returns dict name -> (R array, gain G). One noise draw per position
    (the signal is computed once when the token is current, then stored).
    """
    rng = np.random.default_rng(seed)
    k = np.arange(max_pos + 1, dtype=np.float64)
    out = {}

    # 0. Sanity: linear counter (the current scheme). Must give 0 flips.
    out["linear_8k"] = (k.copy(), 8.0)

    # All readouts are ranked on their ACTUAL value (a softmax weight, or an
    # assembly of weights) with BOUNDED fp32 noise SIGMA_W added per weight
    # read. A flip happens where the value-space gap between adjacent positions
    # is smaller than the noise -- i.e. where the readout's slope (in value
    # units) collapses. Gain 1.0: only the ordering matters for the tie-break.
    def noisy(v, n_reads=1):
        e = sum(rng.normal(0.0, SIGMA_W, v.shape) for _ in range(n_reads))
        return v + e

    # 1. Single cosine: read one weight w = sigmoid(gM cos(phase)); use (1-w)
    #    so it rises with k. Slope (value units) ~ |sin(phase)| -> collapses at
    #    phase 0 and pi. Two sizings: production-max and this-frame stress.
    for period, label in (
        (MAX_POSITIONS, "cos_halfturn_maxpos"),
        (max_pos, "cos_halfturn_framestress"),
    ):
        phase = (math.pi / period) * k
        w = 1.0 / (1.0 + np.exp(-G_M * np.cos(phase)))
        out[label] = (noisy(1.0 - w), 1.0)

    # 2. Octant-select two-head: pick whichever of the cos/sin weight is steep,
    #    chain octants -> a monotone value whose slope never drops below the
    #    pi/4 worst case (|cos|=|sin|=sin45). Modeled as a uniform-slope ramp
    #    in value units with that slope floor, read from TWO weights (2x noise).
    theta_full = 2.0 * math.pi / MAX_POSITIONS
    c_min = _sigmoid_prime(G_M * math.cos(math.pi / 4)) * G_M * math.sin(math.pi / 4)
    ramp = (c_min * theta_full) * k  # value-space, uniform slope
    out[f"octant_twohead(slope={c_min*theta_full:.1e}/tok)"] = (
        noisy(ramp, n_reads=2),
        1.0,
    )

    return out


def replay(rows, R, G):
    flips = 0
    dec_flips = 0
    flip_positions = []
    for qp, sel, dec, pos, clog in rows:
        score = clog + G * R[pos]
        pick = pos[int(np.argmax(score))]
        if pick != sel:
            flips += 1
            flip_positions.append(qp)
            if dec:
                dec_flips += 1
    return flips, dec_flips, np.array(flip_positions)


def margin_diagnostic(rows, R):
    """Deterministic, noiseless budget the readout R must preserve.

    For each selection the renderer chose `selected` (the most-recent key in
    the top content tier). For a monotone R to reproduce it, among candidates
    whose content >= the selected key's content, R[selected] must be the
    largest. The margin is R[selected] minus the next-highest such R. The
    MINIMUM margin over all selections is the per-position noise the readout
    can absorb before a flip — in R units, seed-independent.
    """
    worst = math.inf
    worst_at = None
    n_neg = 0  # selections where R is already non-monotone (margin <= 0)
    for qp, sel, dec, pos, clog in rows:
        csel = clog[pos == sel]
        if len(csel) == 0:
            continue
        csel = csel[0]
        contenders = R[pos][clog >= csel - 1e-9]
        rsel = R[sel]
        others = contenders[contenders != rsel]
        if len(others) == 0:
            continue
        margin = rsel - others.max()
        if margin <= 0:
            n_neg += 1
        if margin < worst:
            worst = margin
            worst_at = qp
    return worst, worst_at, n_neg


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/recency_compact.bin"
    rows = load(path)
    max_pos = max(qp for qp, *_ in rows)
    n_dec = sum(
        1 for *_, dec, _, _ in [(r[0], r[1], r[2], r[3], r[4]) for r in rows] if dec
    )
    n_dec = sum(1 for r in rows if r[2])
    print(f"rows={len(rows)} decisive={n_dec} max_query_pos={max_pos}\n")

    names = list(build_readouts(max_pos, seed=0).keys())

    # (a) Deterministic headroom: min value-space margin the readout must
    #     preserve, vs the fp32 read noise SIGMA_W. headroom = margin / SIGMA_W
    #     is how many fp32-noise units of slack exist at the worst position.
    print("Deterministic margin (value units) and fp32 headroom.\n")
    print(f"{'scheme':>40} {'min_margin':>12} {'headroom(x fp32)':>17}")
    print("-" * 72)
    margins = {}
    for name in names:
        R, _ = build_readouts(max_pos, seed=0)[name]
        m, _, _ = margin_diagnostic(rows, R)
        margins[name] = m
        hr = "-" if name == "linear_8k" else f"{m / SIGMA_W:.0f}"
        print(f"{name:>40} {m:>12.3e} {hr:>17}")

    # (b) Stress: scale the read noise above the pure-fp32 floor (TF32 /
    #     accumulation-order / 1-LSB excursions are documented to occur).
    #     Find where each weight-read scheme starts to flip.
    print("\nNoise-multiplier stress (x SIGMA_W); flips max over 8 seeds.")
    print(f"{'scheme':>40} " + " ".join(f"{m:>5}x" for m in (1, 3, 10, 30, 100)))
    print("-" * 80)
    SEEDS = list(range(8))
    for name in names:
        if name == "linear_8k":
            continue
        cells = []
        for mult in (1, 3, 10, 30, 100):
            worst = 0
            for s in SEEDS:
                R, G = build_readouts(max_pos, seed=s)[name]
                # re-add (mult-1)x extra noise on top of the 1x already baked in
                rng = np.random.default_rng(1000 + s)
                R2 = R + rng.normal(0.0, SIGMA_W, R.shape) * math.sqrt(
                    max(mult**2 - 1, 0)
                )
                f, _, _ = replay(rows, R2, G)
                worst = max(worst, f)
            cells.append(worst)
        print(f"{name:>40} " + " ".join(f"{c:>6}" for c in cells))


if __name__ == "__main__":
    main()
