"""Octant recency ramp — the bucket-2 monotone position signal (RoPE port).

docs/rope_port_plan.md §3 / Phase 1b.  One rotary plane turns once over the
rollout: ``phi(pos) = pos*theta``, ``theta ~= 2*pi/max_positions``.  Two
position-only attention heads read it against BOS as graded 2-key softmax
weights, centered to ``u = sigmoid(gain*cos phi) - 0.5`` (tracks ``cos phi``)
and ``v = sigmoid(gain*sin phi) - 0.5`` (tracks ``sin phi``).

This module turns ``(u, v)`` into a **strictly monotone** ramp over ``phi``.
At every ``phi`` at least one of ``|cos|, |sin|`` is steep, so in each of the 8
octants the steep (nearer-0) centered weight is used with a per-octant sign and
a chaining offset that makes the ramp continuous.  The analytic reference is
``scripts/rope_octant_assembly.py`` (proven monotone, min step
~2.275e-5/token at ``gain=2.0``).

Why ``soft_blend`` and not ``select``: a token's ``phi`` can land arbitrarily
close to a ``k*pi/4`` octant boundary, where the selecting ``compare`` cannot
saturate.  ``select``'s ``(M+v)-M`` core then dips to ``~-M`` (non-monotone);
``soft_blend`` clamps into the ``[min,max]`` box of its two (equal-at-boundary)
operands instead.  See ``soft_blend`` in ``map_select.py``.

**Seam constraint (caller's responsibility).** The ramp has one discontinuity,
the wrap between octant 7 (``phi -> 2*pi``) and octant 0 (``phi -> 0``), which
lives on the ``v = 0`` boundary at ``u > 0``.  The top-of-tree ``sign(v)``
``soft_blend`` is soft there, so the queried ``phi`` range must **exclude a
neighborhood of the seam** (``phi = 0 mod 2*pi``) — i.e. choose the plane phase
/ ``max_positions`` so the rollout's ``phi`` stays inside ``(0, 2*pi)`` away
from the ends.  The other ``v = 0`` crossing (``phi = pi``) is a continuous
interior boundary and is handled normally.
"""

import math

from torchwright.graph import Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.arithmetic_ops import (
    abs as abs_op,
    add_const,
    compare,
    multiply_const,
    subtract,
)
from torchwright.ops.const import step_sharpness
from torchwright.ops.map_select import soft_blend

# Per octant (increasing phi): (which centered weight is steep, sign that makes
# the term increase with phi).  u tracks cos, v tracks sin.  Mirrors
# scripts/rope_octant_assembly.py::_SPEC.
_SPEC = {
    0: ("v", +1.0),
    1: ("u", -1.0),
    2: ("u", -1.0),
    3: ("v", -1.0),
    4: ("v", -1.0),
    5: ("u", +1.0),
    6: ("u", +1.0),
    7: ("v", +1.0),
}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _term(o: int, u: float, v: float) -> float:
    coord, sign = _SPEC[o]
    return sign * (u if coord == "u" else v)


def _octant_offsets(gain: float):
    """Chaining offset per octant so ``offset[o] + term`` is continuous.

    Identical formula to ``scripts/rope_octant_assembly.py::octant_offsets``."""
    edges = [k * (math.pi / 4.0) for k in range(9)]
    offset = [0.0] * 8
    running = 0.0
    for o in range(8):
        cs, ss = math.cos(edges[o]), math.sin(edges[o])
        offset[o] = running - _term(o, _sig(gain * cs) - 0.5, _sig(gain * ss) - 0.5)
        ce, se = math.cos(edges[o + 1]), math.sin(edges[o + 1])
        running = offset[o] + _term(o, _sig(gain * ce) - 0.5, _sig(gain * se) - 0.5)
    return offset


def _assert_branches_meet_at_boundaries(gain: float, offset, atol: float = 1e-9):
    """Construction-time check: adjacent octant branch values are *exactly*
    equal at each shared ``k*pi/4`` boundary.

    This is the precondition ``soft_blend`` relies on (its ``cond`` is soft only
    where its two operands are equal).  A wrong offset table would silently make
    the ramp non-monotone; this catches it at build time on the analytic
    ``(u, v)`` of each boundary.  Checks the 7 in-range boundaries
    ``phi = pi/4 .. 7pi/4``; the 8th (``phi = 0``, the seam) is intentionally a
    discontinuity (see module docstring)."""
    for k in range(1, 8):
        phi = k * (math.pi / 4.0)
        u = _sig(gain * math.cos(phi)) - 0.5
        v = _sig(gain * math.sin(phi)) - 0.5
        o_lo, o_hi = k - 1, k  # octants meeting at this boundary
        b_lo = offset[o_lo] + _term(o_lo, u, v)
        b_hi = offset[o_hi] + _term(o_hi, u, v)
        if abs(b_lo - b_hi) > atol:
            raise AssertionError(
                f"octant branches disagree at phi={phi:.6f} (k={k}): "
                f"o{o_lo}={b_lo:.9f} vs o{o_hi}={b_hi:.9f} "
                f"(diff {abs(b_lo - b_hi):.2e} > {atol:.0e}) — offset table is wrong."
            )


def octant_recency_ramp(
    u: Node,
    v: Node,
    *,
    gain: float = 2.0,
    sharpness: float | None = None,
) -> Node:
    """Monotone recency ramp from the two centered rotary-plane weights.

    Args:
        u: length-1 node, ``sigmoid(gain*cos phi) - 0.5`` (cos-head, centered).
        v: length-1 node, ``sigmoid(gain*sin phi) - 0.5`` (sin-head, centered).
        gain: head gain ``M`` baked into the offset table (default 2.0, the
            value proven in ``rope_octant_assembly.py``).  Must match the gain
            the heads actually apply.
        sharpness: ``compare`` ramp sharpness for the three octant tests.  A
            larger value narrows the soft zone at each boundary; the right
            value is fixed by the Phase-1b gate-b sweep.  ``None`` uses the
            module default.

    Returns:
        length-1 node: the recency ramp (multiply by the rank gain ``G`` at the
        call site; ``G`` is argmax-invariant).
    """
    assert len(u) == 1 and len(v) == 1

    offset = _octant_offsets(gain)
    _assert_branches_meet_at_boundaries(gain, offset)

    # Centered sigmoid weights live strictly in (-0.5, 0.5); pin that so the
    # branch values (and soft_blend's M) carry finite bounds.
    box = NodeValueType(value_range=Range(-0.5, 0.5))
    u = assert_matches_value_type(u, box, atol=1e-3)
    v = assert_matches_value_type(v, box, atol=1e-3)

    # The 8 octant branch values b_o = offset[o] + sign_o * (u or v), each a
    # bounded affine function of one centered weight.
    def branch(o: int) -> Node:
        coord, sign = _SPEC[o]
        node = u if coord == "u" else v
        return add_const(multiply_const(node, sign), offset[o])

    b = [branch(o) for o in range(8)]

    # Three octant tests (torchwright booleans: +1 = true).  compare's ramp is
    # one-sided: cond=0 lands at inp = thresh + 1/(2s), and cond=-1 (crisp) at
    # inp = thresh.  With thresh=0 the whole soft zone sits on the true side of
    # the boundary, so soft_blend would be soft where the branches already
    # differ — its precondition (soft only where t==f) holds only approximately.
    # Shift each threshold by -1/(2s) so cond=0 lands exactly on the boundary
    # (inp=0) and the soft zone straddles it symmetrically: now soft_blend is
    # fully soft precisely where the adjacent branches are equal.
    s = step_sharpness if sharpness is None else sharpness
    half = 1.0 / (2.0 * s)
    cond_u = compare(u, -half, sharpness=s)  # +1 iff u >~ 0 (soft zone centered on 0)
    cond_v = compare(v, -half, sharpness=s)  # +1 iff v >~ 0
    cond_c = compare(
        subtract(abs_op(u), abs_op(v)), -half, sharpness=s
    )  # +1 iff |u| >~ |v|

    # Binary tree on (sign v, sign u, |u|>|v|).  With the centered thresholds
    # above, each node's two operands are equal exactly on its cond=0 surface
    # (the matching in-range boundary), so every select is a soft_blend.
    # Octant assignments (sign_v, sign_u, |u|>|v|):
    #   o0:(+,+,T) o1:(+,+,F) o2:(+,-,F) o3:(+,-,T)
    #   o4:(-,-,T) o5:(-,-,F) o6:(-,+,F) o7:(-,+,T)
    # cond_c soft at |u|=|v|:
    n_pp = soft_blend(cond_c, b[0], b[1])  # v>=0, u>=0  (phi=pi/4: o0|o1)
    n_pn = soft_blend(cond_c, b[3], b[2])  # v>=0, u<0   (phi=3pi/4: o2|o3)
    n_np = soft_blend(cond_c, b[7], b[6])  # v<0,  u>=0  (phi=7pi/4: o6|o7)
    n_nn = soft_blend(cond_c, b[4], b[5])  # v<0,  u<0   (phi=5pi/4: o4|o5)
    # cond_u soft at u=0:
    n_p = soft_blend(cond_u, n_pp, n_pn)  # v>=0 (phi=pi/2: o1|o2)
    n_n = soft_blend(cond_u, n_np, n_nn)  # v<0  (phi=3pi/2: o5|o6)
    # cond_v soft at v=0 (phi=pi interior continuous; phi=0 seam out of range):
    return soft_blend(cond_v, n_p, n_n)
