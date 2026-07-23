"""radix_floor_int: floor(x) as hi-floor -> integer snap -> lo-floor.

Spec: docs/ops_plain_english.md (radix_floor_int entry, inheriting
floor_int's build).  The properties under test, in contract order:

1. Exactness parity with floor_int on legal inputs (out of the
   1/sharpness ramp below each integer), including exact integers.
2. **The divisor-boundary sliver reconstructs exactly** — an input in
   the HI floor's ramp (just below a multiple of D) but out of the LO
   floor's ramp gives the exact floor, because the snapped hi and the
   extended-range lo compensate whichever side the snap lands on.
   This is the D-amplification hazard of the two-digit emit split
   (the digit-quad's find #3), and the reason the snap stage exists.
3. An input inside the LO ramp degrades to the same ±1-step window as
   flat floor_int — no D-amplification.
4. Lane cost stays ~8.5*sqrt(N) (vs 3N flat) and the composed graph
   compiles clean against the oracle.
"""

import math

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.swiglu.arithmetic_ops import radix_floor_int

# The production call site this op was built for: the DOOM native
# texture-coordinate floor (render_ops.FLOOR_NATIVE), N=2046, s=10^4.
NATIVE_LO, NATIVE_HI, NATIVE_S = -1023, 1023, 10_000.0


def _native():
    x = create_input("x", 1, value_range=(float(NATIVE_LO), float(NATIVE_HI)))
    return x, radix_floor_int(x, NATIVE_LO, NATIVE_HI, sharpness=NATIVE_S)


def _eval(node, v: float) -> float:
    return node.compute(n_pos=1, input_values={"x": torch.tensor([[v]])}).item()


def test_exact_on_integers_and_flat_zones():
    _x, f = _native()
    for v in (-1023.0, -1022.0, -513.0, -1.0, 0.0, 1.0, 512.0, 1023.0):
        assert abs(_eval(f, v) - v) < 1e-3, f"floor({v}) drifted: {_eval(f, v)}"
    for v in (-1022.6, -1000.4, -0.5, 0.7, 127.9, 500.25, 1022.4):
        assert abs(_eval(f, v) - math.floor(v)) < 1e-3


def test_divisor_boundary_sliver_reconstructs_exactly():
    """Inputs in the HI ramp but out of the LO ramp reconstruct exactly.

    Just below a multiple of D=64: the flat form is exact here, and the
    radix form must be too — a fractional hi would be amplified x64 by
    the recombine, so this pins the snap + extended-lo compensation.
    """
    _x, f = _native()
    for m in (-512, -64, 64, 512, 896):
        for delta in (1e-4 + 1e-6, 5e-4, 1e-3, 5e-3):
            v = float(m) - delta
            got = _eval(f, v)
            assert abs(got - (m - 1)) < 1e-3, (
                f"floor({v}) at the D-boundary sliver: expected {m - 1}, "
                f"got {got} (fractional hi leaked through the snap?)"
            )


def test_lo_ramp_stays_within_one_step():
    """Inside the LO ramp, the output stays within one step, never D-amplified.

    Within 1/s below an integer the output is fractional — the same
    tolerated window as flat floor_int.  The pin: it stays inside
    (true, true+1), never D-amplified.
    """
    _x, f = _native()
    for k in (-512, -100, 0, 64, 1000):
        for frac in (0.2, 0.5, 0.9):
            v = float(k) - frac / NATIVE_S
            got = _eval(f, v)
            assert (k - 1) - 1e-3 < got < k + 1e-3, (
                f"floor({v}) in the lo ramp: {got} outside ({k - 1}, {k})"
            )


def test_negative_range_and_default_divisor():
    x = create_input("x", 1, value_range=(-3.0, 3.0))
    # n=6, default divisor=4 (n > d) -> radix path over a tiny range.
    f = radix_floor_int(x, -3, 3)
    for v, expected in [(-2.5, -3.0), (-1.0, -1.0), (0.0, 0.0), (1.5, 1.0)]:
        assert abs(_eval(f, v) - expected) < 0.01


def test_falls_back_to_flat_when_range_within_divisor():
    """N <= divisor takes the flat-floor_int early return and still floors correctly.

    Nothing to split; the op must still floor correctly on that branch.
    """
    x = create_input("x", 1, value_range=(-2.0, 2.0))
    # n=4 <= d=8 -> fallback branch.
    f = radix_floor_int(x, -2, 2, divisor=8)
    for v, expected in [(-1.5, -2.0), (-1.0, -1.0), (0.0, 0.0), (1.7, 1.0)]:
        assert abs(_eval(f, v) - expected) < 0.01


def test_lane_cost_is_sqrtn_class():
    """The point of the op: lane cost is in the sqrt(N) class, not linear in N.

    ~8.5*sqrt(N) hidden lanes vs 3N flat, and no residual intermediate
    wider than 2*max(ceil(N/D), D+1).
    """
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.graph import FFN

    _x, f = _native()
    n = NATIVE_HI - NATIVE_LO
    ffns = [nd for nd in get_ancestor_nodes({f}) if isinstance(nd, FFN)]
    lanes = sum(nd.n_lanes for nd in ffns)
    assert lanes <= 12 * math.isqrt(n) + 24, f"{lanes} lanes: radix split missing?"
    assert lanes < 3 * n / 4, f"{lanes} lanes: not meaningfully below flat (3N={3 * n})"
    widest = max(len(nd) for nd in ffns)
    assert widest <= 2 * max(-(-n // 64), 65), f"widest FFN intermediate {widest}"


def test_compiles_clean():
    """Compiled swish graph matches the exact-math oracle on a sweep of legal inputs.

    The sweep spans the native range.
    """
    x = create_input("x", 1, value_range=(-127.0, 127.0))
    f = radix_floor_int(x, -127, 127, sharpness=1000.0)
    g = torch.Generator().manual_seed(41)
    n_pos = 64
    vals = torch.rand(n_pos, 1, generator=g) * 254.0 - 127.0
    # Nudge anything inside the 1/s ramp out of it so the sweep tests
    # only the legal-input contract.
    gap = torch.ceil(vals) - vals
    vals = torch.where(gap < 2e-3, vals - 2e-3, vals)
    compiled = compile_headless(f, d=256, d_head=8)
    report = probe_compiled(compiled, f, {"x": vals}, n_pos, atol=1e-2)
    assert report.first_divergent is None, report.format_short()
