"""Octant recency ramp — oracle-level validation (RoPE port Phase 1b).

These tests evaluate the graph ramp through the **memoised** oracle
(``reference_eval``) and compare against the analytic reference
``scripts/rope_octant_assembly.py``.  They confirm the *construction* (offset
table, octant tree, soft_blend wiring) is correct and strictly monotone away
from the seam.  The *compiled*, sub-token-dense, fp32-nondeterminism gate-b
sweep is a separate heavier test.

(Raw ``node.compute`` is un-memoised and blows up on the shared soft_blend DAG;
``reference_eval`` collapses it to one O(graph) pass — see probe.py.)
"""

import math

import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.recency_ramp import octant_recency_ramp, _octant_offsets, _SPEC

from scripts.rope_octant_assembly import GAIN, ramp as ref_ramp


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def _uv(phi, gain=GAIN):
    return _sig(gain * math.cos(phi)) - 0.5, _sig(gain * math.sin(phi)) - 0.5


def _build():
    u = create_input("u", 1)
    v = create_input("v", 1)
    return u, v, octant_recency_ramp(u, v, gain=GAIN)


def _run(out, phis, gain=GAIN):
    us, vs = zip(*[_uv(p, gain) for p in phis])
    u_t = torch.tensor(us, dtype=torch.float32).reshape(-1, 1)
    v_t = torch.tensor(vs, dtype=torch.float32).reshape(-1, 1)
    n = len(phis)
    cache = reference_eval(out, {"u": u_t, "v": v_t}, n)
    return cache[out].reshape(-1)


def test_offset_table_matches_assembly():
    """The graph builder's offset table equals the analytic assembly's."""
    from scripts.rope_octant_assembly import octant_offsets

    ours = _octant_offsets(GAIN)
    theirs = octant_offsets(GAIN).tolist()
    for o in range(8):
        assert abs(ours[o] - theirs[o]) < 1e-12


def test_ramp_matches_reference_mid_octant():
    """Mid-octant (crisp conds): graph ramp == analytic reference, tightly."""
    u, v, out = _build()
    # 8 mid-octant angles, each pi/8 past a boundary.
    phis = [(k + 0.5) * (math.pi / 4.0) for k in range(8)]
    got = _run(out, phis).tolist()
    ref = ref_ramp(torch.tensor(phis).numpy()).tolist()
    for g, r in zip(got, ref):
        assert abs(g - r) < 1e-3


# Seam exclusion: at the default compare sharpness the top-of-tree sign(v)
# soft_blend stays soft within ~0.03 of phi/2pi=0 (zone width ~4/(g*sharpness)).
# 0.05 clears it with margin. In production this is the plane's phase offset
# (phi_0), placing the rollout's phi range away from the phi=0/2pi seam.
_SEAM_MARGIN = 0.05


def _seam_excluded_phis(n):
    return [
        (_SEAM_MARGIN + (1.0 - 2.0 * _SEAM_MARGIN) * i / (n - 1)) * 2.0 * math.pi
        for i in range(n)
    ]


def test_ramp_strictly_monotone_off_seam():
    """Dense phi sweep avoiding the seam: strictly increasing on the oracle."""
    u, v, out = _build()
    n = 6000
    got = _run(out, _seam_excluded_phis(n))
    diffs = got[1:] - got[:-1]
    assert torch.all(diffs > 0), (
        f"non-monotone: {int((diffs <= 0).sum())} of {n - 1} steps "
        f"non-increasing; min step {diffs.min().item():.3e}"
    )


def test_ramp_min_slope_near_reference():
    """The oracle ramp's min slope per token matches the analytic model."""
    u, v, out = _build()
    n = 6000
    phis = _seam_excluded_phis(n)
    got = _run(out, phis)
    dphi = phis[1] - phis[0]
    theta = 2.0 * math.pi / 61440.0
    min_step_per_token = (got[1:] - got[:-1]).min().item() / dphi * theta
    # Analytic min step ~2.275e-5/token (g=2.0); the oracle is slightly lower
    # (~1.7e-5) because interior boundary soft-zones flatten it. The compiled,
    # production-density confirmation is the separate gate-b sweep.
    assert (
        min_step_per_token > 1.3e-5
    ), f"min step/token {min_step_per_token:.3e} too small"


def test_branch_equality_assertion_catches_bad_offsets():
    """The construction-time boundary check rejects a corrupted offset table."""
    import torchwright.ops.recency_ramp as rr

    bad = _octant_offsets(GAIN)
    bad[3] += 0.1  # break continuity at one boundary
    try:
        rr._assert_branches_meet_at_boundaries(GAIN, bad)
    except AssertionError:
        return
    raise AssertionError("expected the boundary check to reject bad offsets")
