"""Gate-b: the octant recency ramp is strictly monotone on the COMPILED path.

docs/rope_port_plan.md Phase 1b gate (b). The oracle tests
(tests/ops/test_recency_ramp.py) prove the construction; this proves the
compiled transformer reproduces it with no boundary dip, at the resolution
real tokens actually see.

**Why token-density across phase offsets, not sub-token-dense.** The plan
flagged that the soft window around an octant boundary can be narrower than one
production token (theta = 2*pi/61440), so a fixed per-token grid could step over
a flat spot. But sampling *sub-token* is a trap of its own: the ramp output is
O(1) in fp32, so a sub-token step (~few * 1e-7 rad -> ~few ULP of the output)
hits fp32 *output quantization* and reports spurious zero steps that are not a
real flat spot. The physically correct test is at **token spacing** (steps
~theta -> ~2e-5 in ramp units, ~180x the fp32 floor, cleanly resolvable), swept
over many **sub-token phase offsets** delta in [0, theta): real tokens are theta
apart, and delta covers wherever the token grid falls relative to each boundary.
If every offset is monotone, no real token grid can land in a dip. (Sub-token
sampling at m=120 reports min step 0 purely from output quantization; at token
density across offsets the worst boundary step is a clean ~2.19e-5/token.)
"""

import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.ops.recency_ramp import octant_recency_ramp

GAIN = 2.0
THETA = 2.0 * math.pi / 61440.0  # one rotary turn over max_positions
SEAM_MARGIN = 0.05  # exclude the phi=0/2pi seam (caller's phase offset)
# Strictly-positive floor on the per-token step. Observed compiled worst case is
# ~1.7e-5/token (global, interior); boundaries run ~2.2e-5. 1e-5 keeps margin for
# fp32 run-to-run variation while still ~80x the fp32 weight-noise floor (1.2e-7).
MIN_STEP_PER_TOKEN = 1.0e-5


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


@pytest.fixture(scope="module")
def ramp_module():
    pos = create_pos_encoding()
    u = create_input("u", 1)
    v = create_input("v", 1)
    out = octant_recency_ramp(u, v, gain=GAIN)
    return compile_headless(out, pos, d=256, verbose=False)


def _eval(module, phis):
    rows = torch.tensor(
        [
            [_sig(GAIN * math.cos(p)) - 0.5, _sig(GAIN * math.sin(p)) - 0.5]
            for p in phis
        ],
        dtype=torch.float32,
    )
    return module(rows).reshape(-1)


def _assert_monotone(phis, vals, label):
    diffs = vals[1:] - vals[:-1]
    n_bad = int((diffs <= 0).sum())
    assert n_bad == 0, (
        f"{label}: {n_bad}/{len(diffs)} non-increasing steps; "
        f"min step {diffs.min().item():.3e}"
    )


def test_compiled_ramp_monotone_coarse(ramp_module):
    """Coarse sweep over the whole seam-excluded range is monotone.

    Catches any gross construction/compile error across the full ramp.
    """
    n = 4000  # ~14 tokens/sample; memory-safe single forward
    phis = [
        (SEAM_MARGIN + (1 - 2 * SEAM_MARGIN) * i / (n - 1)) * 2 * math.pi
        for i in range(n)
    ]
    vals = _eval(ramp_module, phis)
    _assert_monotone(phis, vals, "coarse")
    dphi = phis[1] - phis[0]
    min_step_per_token = (vals[1:] - vals[:-1]).min().item() / dphi * THETA
    assert min_step_per_token > MIN_STEP_PER_TOKEN, (
        f"coarse: min step/token {min_step_per_token:.3e} "
        f"below floor {MIN_STEP_PER_TOKEN:.0e}"
    )


def test_compiled_ramp_monotone_at_boundaries_token_density(ramp_module):
    """Token-density windows straddling each in-range boundary, swept over
    sub-token phase offsets, are strictly monotone with a resolvable step."""
    K = 4  # tokens each side of the boundary
    n_offsets = 40  # sub-token phase offsets covering [0, theta)
    worst = math.inf
    for k in range(1, 8):  # k=0 is the seam
        center = k * (math.pi / 4.0)
        for j in range(n_offsets):
            delta = THETA * j / n_offsets
            phis = [center - K * THETA + delta + i * THETA for i in range(2 * K + 1)]
            vals = _eval(ramp_module, phis)
            _assert_monotone(phis, vals, f"boundary k={k} offset {j}")
            worst = min(worst, (vals[1:] - vals[:-1]).min().item())
    # steps here are theta-spaced -> compare directly against the per-token floor
    assert (
        worst > MIN_STEP_PER_TOKEN
    ), f"boundary token step {worst:.3e} below floor {MIN_STEP_PER_TOKEN:.0e}"
