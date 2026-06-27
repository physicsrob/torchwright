"""Unit tests for the ``soft_blend`` op (docs/rope_port_plan.md Phase 1b).

``soft_blend(cond, t, f)`` is a bounded crisp-handoff switch:
- crisp cond (±1) → returns ``t`` / ``f`` exactly;
- soft cond (≈0) → returns a value inside the ``[min(t,f), max(t,f)]`` box,
  never ``select``'s ``−M`` dip;
- same-sign ``t,f`` → the ``min``/``max`` clamp restores in-box-ness even
  though the unclamped carrier core overshoots (the D6 monotonicity repro).
"""

import torch

from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.map_select import soft_blend


def _bounded(name, lo, hi):
    return assert_matches_value_type(
        create_input(name, 1), NodeValueType(value_range=Range(lo, hi))
    )


def _run(out, cond, t, f):
    return out.compute(
        n_pos=1,
        input_values={
            "cond": torch.tensor([[cond]]),
            "t": torch.tensor([[t]]),
            "f": torch.tensor([[f]]),
        },
    ).item()


def test_soft_blend_crisp_picks_branch_exactly():
    """cond=±1 returns t / f to within the clamp's PL noise."""
    out = soft_blend(
        _bounded("cond", -1, 1), _bounded("t", -1, 1), _bounded("f", -1, 1)
    )
    assert abs(_run(out, 1.0, 0.8, -0.5) - 0.8) < 1e-4
    assert abs(_run(out, -1.0, 0.8, -0.5) - (-0.5)) < 1e-4
    # small-magnitude branch values survive (the case select's cancellation loses)
    assert abs(_run(out, 1.0, 1e-4, -1e-4) - 1e-4) < 1e-5
    assert abs(_run(out, -1.0, 1e-4, -1e-4) - (-1e-4)) < 1e-5


def test_soft_blend_soft_stays_in_box():
    """cond=0 with t≈f returns a value inside [min(t,f), max(t,f)] (no −M dip)."""
    out = soft_blend(
        _bounded("cond", -1, 1), _bounded("t", -1, 1), _bounded("f", -1, 1)
    )
    # t≈f: the octant-boundary case. Output must sit between them, not at −M.
    v = _run(out, 0.0, 0.3001, 0.3000)
    assert 0.3000 - 1e-4 <= v <= 0.3001 + 1e-4
    # crossing the boundary the other way is symmetric
    v = _run(out, 0.0, -0.3000, -0.3001)
    assert -0.3001 - 1e-4 <= v <= -0.3000 + 1e-4


def test_soft_blend_same_sign_overshoot_clamped():
    """D6 repro: same-sign t,f make the carrier core overshoot; the clamp saves it.

    At cond=0 the raw core computes ReLU(t)+ReLU(f) = t+f (both positive),
    which is outside [min,max]=[t,f]. The median clamp pulls it back in-box,
    which is what keeps the octant ramp monotone.
    """
    out = soft_blend(_bounded("cond", 0, 1), _bounded("t", 0, 1), _bounded("f", 0, 1))
    # t=f=0.3: unclamped raw would be 0.6; clamped must be 0.3.
    assert abs(_run(out, 0.0, 0.3, 0.3) - 0.3) < 1e-4
    # t≠f same sign: stays inside [0.2, 0.4] rather than overshooting to 0.6.
    v = _run(out, 0.0, 0.4, 0.2)
    assert 0.2 - 1e-4 <= v <= 0.4 + 1e-4


def test_soft_blend_output_value_type_is_union_box():
    """Output static value-type is the union of the t/f boxes."""
    out = soft_blend(
        _bounded("cond", -1, 1), _bounded("t", -2, 3), _bounded("f", -5, 1)
    )
    r = out.value_type.value_range
    assert r.lo == -5.0 and r.hi == 3.0
