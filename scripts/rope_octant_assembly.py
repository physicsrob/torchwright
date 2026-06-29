"""Octant recency-ramp assembly — the analytic (noise-free) reference.

Phase-1b gate (b) step 1 (docs/rope_port_plan.md §8): confirm the REAL octant
assembly — not the replay's idealized uniform-slope line — produces a strictly
monotone ramp with the modeled min slope, before building it as a graph.

Signal: one rotary plane turning once over the rollout, ``phi(pos) = pos*theta``,
``theta = 2*pi/max_positions``.  Two position-only heads read it against BOS as
graded 2-key softmax weights ``a = sigmoid(g*cos phi)``, ``s = sigmoid(g*sin phi)``.
Centered ``u = a-0.5`` (tracks cos phi), ``v = s-0.5`` (tracks sin phi).

The 8 octants are the ``(sign cos, sign sin, |cos|>|sin|)`` combos — derivable
in-graph from ``sign(u)``, ``sign(v)``, ``|u|>|v|`` (compare ops).  In each octant
the STEEP (mid-sigmoid) coordinate is the one nearer 0 (smaller ``|.|``); use it
with a per-octant sign and a chaining offset so the ramp is continuous and
strictly increasing.  This is the function the graph assembly must reproduce; the
rank fed to the recency argmax is ``G * ramp`` (``G ~ 2e5``, argmax-invariant).

Result (g=2.0, max_positions=61440): strictly monotone over all 8 boundaries,
min step 2.275e-5/token (~190x the fp32 weight-noise floor) — matching the
replay model's slope/headroom assumption.  ``G = 2.0`` is the chosen gain ``M``.
"""

import numpy as np

GAIN = 2.0  # the head gain M: mid-sigmoid at the octant boundaries, max headroom.

_SIG = lambda x: 1.0 / (1.0 + np.exp(-x))

# Per octant (increasing phi): which centered weight is steep, and the sign that
# makes the term increase with phi.  u = cos-weight, v = sin-weight.
_SPEC = {0: ("v", +1), 1: ("u", -1), 2: ("u", -1), 3: ("v", -1),
         4: ("v", -1), 5: ("u", +1), 6: ("u", +1), 7: ("v", +1)}


def _octant(c, s):
    table = {
        (True, True, True): 0, (True, True, False): 1,
        (False, True, False): 2, (False, True, True): 3,
        (False, False, True): 4, (False, False, False): 5,
        (True, False, False): 6, (True, False, True): 7,
    }
    return table[(c >= 0, s >= 0, abs(c) >= abs(s))]


def _term(o, u, v):
    coord, sign = _SPEC[o]
    return sign * (u if coord == "u" else v)


def octant_offsets(g):
    """Chaining offset per octant so ``ramp = offset[o] + term`` is continuous."""
    edges = np.arange(9) * (np.pi / 4)
    offset, running = np.zeros(8), 0.0
    for o in range(8):
        cs, ss = np.cos(edges[o]), np.sin(edges[o])
        offset[o] = running - _term(o, _SIG(g * cs) - 0.5, _SIG(g * ss) - 0.5)
        ce, se = np.cos(edges[o + 1]), np.sin(edges[o + 1])
        running = offset[o] + _term(o, _SIG(g * ce) - 0.5, _SIG(g * se) - 0.5)
    return offset


def ramp(phis, g=GAIN):
    """The monotone recency ramp at angles ``phis`` (the analytic reference)."""
    u = _SIG(g * np.cos(phis)) - 0.5
    v = _SIG(g * np.sin(phis)) - 0.5
    offset = octant_offsets(g)
    out = np.empty_like(phis)
    for i, phi in enumerate(phis):
        o = _octant(np.cos(phi), np.sin(phi))
        out[i] = offset[o] + _term(o, u[i], v[i])
    return out


def analyze(g=GAIN, max_positions=61440, n=200000):
    phis = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = ramp(phis, g)
    dr = np.diff(r)
    theta = 2 * np.pi / max_positions
    min_step = (dr / np.diff(phis)).min() * theta
    floor = 2 * 6e-8  # two fp32-exact weight reads
    return {"g": g, "monotone": bool(np.all(dr > 0)),
            "min_step_per_token": float(min_step),
            "headroom_vs_floor": float(min_step / floor)}


if __name__ == "__main__":
    for g in (1.0, 2.0, 3.0, 4.0):
        r = analyze(g)
        print(f"g={g:>4}: monotone={r['monotone']}  "
              f"min_step/tok={r['min_step_per_token']:.3e}  "
              f"headroom={r['headroom_vs_floor']:.1f}x")
