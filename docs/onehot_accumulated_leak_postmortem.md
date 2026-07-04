# Postmortem: the intermittent `123×456` failure (Phase C)

2026-07-04. Fixed in `8ff8618`. The one-sentence cause (D3): a
machine-built one-hot carries a ~1e-5 fp32 round-trip leak *per
element*, and `onehot_lookup`'s closing assert budgeted only a fixed
1e-3 slack — sound for any one element, but the calculator digit
pipeline's key is 61 elements wide and the leaks sum, each weighted by
up to the largest table value, landing right at the slack boundary
where run-to-run GPU float variation flips the verdict.

## Symptom

`tests/examples/test_calculator_arithmetic.py::test_multiply_three_digit
[calculator_simple]` failed on one full-suite Modal run, passed on the
next identical-code run, and always passed locally under a `-k` filter:

    AssertionError: Assert failed at node_117: matches
    NodeValueType(value_range=Range(lo=0.0, hi=6.0))
    (range Range(lo=0.0, hi=6.0) (atol=0.001); bad at [0]=-0.0011)

Note the failing check ran in **exact-math reference eval**
(`node.compute`) — no compiled transformer involved. The graph's own
math, evaluated faithfully, produced −0.0011 against a slack of 0.001.
This appeared during the Phase C examples cutover, when
`calculator_simple` started building the swish machine.

## Mechanism

The multiply digit pipeline feeds `onehot_lookup` a one-hot key built
from `in_range` indicators mapped ±1 → 0/1. On the swish machine those
indicators are never *exactly* 0 or 1 in exact math: each element ends
up within ~1e-5–3e-5 of its ideal value (the `×scale/÷scale` round trip
of the sharpened hinge — "winner rounding" in the op's docstring).
Individually that is two orders of magnitude inside the closing
assert's 1e-3 slack, which is exactly what the docstring argued.

What the argument missed is width. The digit pipeline's key
concatenates indicator blocks into a 61-element vector, and the lookup
is a weighted sum over all of them: the output error is
`Σ_k leak_k · |value_k|`, not `max_k leak_k`. Measured on the failing
input: per-element leak ≤ 3.4e-5, summed leak ~3–6e-4, and with table
values up to 6 the output lands 0.9–1.6e-3 outside the claimed range —
*straddling* the 1e-3 slack.

The straddle is why it was intermittent: exact-math eval on GPU still
has run-to-run float variation (cuBLAS algorithm selection, TF32,
reduction order — the *FP nondeterminism at tolerance boundaries*
section of `CLAUDE.md`), worth ~1e-4 at these magnitudes. One run
landed at −0.0009 (pass), another at −0.0011 (fail). Same code, same
inputs. The full-suite-only signature is the same section's: allocator
state biases the kernel choice; a `-k` run reproducibly picks the
lucky path.

## Why the ReLU machine never saw it

ReLU indicators saturate *exactly* in exact math — a losing lane is
`relu(negative) = 0.0`, bit-exact, and a winner is exactly 1. The
accumulated leak is identically zero, so a fixed 1e-3 slack was never
exercised. The swish machine's hinges saturate exactly too
(`σ(z) = 1.0` for `z ≥ 17`, the A0 probes), but the folded
`×scale/÷scale` arithmetic around them leaves the ~1e-5 ulp-scale
residue per element. This is precisely the class the Phase B checklist
called "re-budget every ±1 check on in_range-fed masks" — this one was
missed because the per-element analysis looked complete.

## Fix

`onehot_lookup`'s closing assert now derives its slack:

    atol = _lookup_numeric_slack(max_abs, 1.0, d_key)
         = max(1e-3, max_abs · d_key · 1e-5)

`_lookup_numeric_slack` (`ops/_math.py`) is the Phase B helper built
for exactly this shape — guard slack over a wide construction, sized
above accumulated fp32 noise *and* GPU cross-test variation. For the
failing case (`max_abs = 6`, `d_key = 61`) that is 3.7e-3 against an
observed worst of ~1.6e-3.

Two properties worth keeping in mind:

- **The slack is check-only.** `assert_matches_value_type`'s `atol`
  widens the runtime predicate, not the claimed range — downstream
  static analysis still sees the tight `[min, max]`, which is the
  reason `onehot_lookup` exists (`map_to_table`'s pessimistic
  `default ± Σ|Δ|` range blew up chained interval arithmetic).
- **Small tables stay tight.** The floor keeps 1e-3; the derived term
  only grows with real width and magnitude.

D6 repros live in `tests/ops/swiglu/test_lookup_ops.py`
(`test_onehot_lookup_wide_key_accumulated_leak_within_guard`,
`test_onehot_lookup_small_table_guard_stays_tight`).

## Is anything else exposed?

Surveyed the swiglu closing asserts (2026-07-04): every other one
either derives its slack in actual-value terms already
(`select`/`broadcast_select`/`cond_gate` gate slacks, `equals_vector`,
`table_lookup_2d` via the same `_lookup_numeric_slack`) or claims a
range that is pessimistically wide by construction (`map_to_table`'s
`default ± Σ|Δ|`, `in_range`'s dip-widened ±1). `onehot_lookup` was the
one op combining a *tight* range claim, a *fixed* slack, and a *wide*
machine-built input.

The lesson that generalizes: a per-element noise argument does not
survive summation — any guard over an output that sums many
machine-built approximate values must budget `Σ leak·|weight|`, not
`max leak`. When adding an op whose input is a wide indicator vector,
size the closing assert with `_lookup_numeric_slack` from the start.

## Addendum (2026-07-04, follow-up investigation)

`scripts/investigate_onehot_leak.py` swept the full mechanism space —
the carry lookup's input is completely described by (integer total
0..60, bounded upstream noise), so 61 totals × a dense noise grid
covers every reachable case — on two hosts (local, Modal container).
One sharpening and two corrections to the account above.

**The bit-level source of the per-element leak.** The sharpened
hinges saturate to exact integers; the leak enters in `in_range`'s
output projection, whose folded weight `2/scale = 0.02` is **not
representable in fp32** (`fl(0.02)` is off by −4.5e-10). Each
saturated lane's product `hinge_value × fl(0.02)` rounds at the
product's magnitude — up to ~1220 for the 61-slot staircase, where
one fp32 ulp is 1.2e-4 — and the near-cancellation of each ramp's two
hinges preserves those product roundings as the residue. Measured
structure: leak exactly 0.0 on every slot below the winner (those
lanes underflow to signed zero), ≤ 2.4e-7 at the winner, ≤ 4.6e-5 per
slot above it (the 3.4e-5 above was one input's max, not the
mechanism's bound). Proof of attribution: patching `scale` to 128
(`2/scale = 2⁻⁶`, exactly representable) makes every product exact
and the measured leak is identically 0.0 across all 61×61
total/slot combinations, on both hosts.

**The failing eval ran on CPU, not GPU.** Reference eval computes on
whatever device the graph's tensors were built on — plain CPU for the
calculator tests; the conftest `device` fixture steers only the
compiler. The run-to-run variation that flipped the verdict is CPU
matmul reduction order, which varies with thread count and host
microarchitecture: measured 3.9e-4 shift in a carry value between
torch thread settings on one machine, and the whole leak pattern
shifts host-to-host (Σ|leak| max 1.339e-3 local vs 1.327e-3 Modal,
identical code and inputs) — same order as the observed
−0.0009/−0.0011 flake pair. The "GPU float variation" story above was
right that an environment-dependent reduction order flips a
boundary-straddling value, wrong about the device.

**Measured guard margin is 1.5×, not ~2.3×.** Worst *range violation*
(what the assert actually checks) of the carry lookup over the swept
space: 2.41e-3 local / 2.35e-3 Modal, at t=0 under ≈−2e-3 upstream
noise, against the derived slack 3.66e-3. The "observed worst
~1.6e-3" above was the flaking input, not the worst case. A table
with the D6 repro's shape (max-magnitude values on half the rows)
violates the old fixed 1e-3 deterministically at zero upstream noise
(1.19e-3 at δ=0). The derived slack holds, with thinner margin than
the fix commit implied.

**Root-fix option, not taken here.** `scale = 128` eliminates the
class rather than budgeting it: integer-fed
`in_range`/`bool_to_01`/`onehot_lookup` chains become bit-exact (leak
0.0, host- and thread-invariant), and injected-noise pass-through
drops to ≤ 2.4e-5 against the same slacks (150–960× headroom on the
swept tables). This is the same power-of-two-exactness trick the
no-bias constant lane already uses deliberately
(`bias_lane_gate`/`bias_lane_up` in `ops/const.py`). Cost: `scale` is
a settled design constant (docs/swiglu_step2_plan.md, decision 5)
baked into every swiglu op's weights — changing it means a full
`make measure-noise` cycle plus `tests/docs/test_swish_constants.py`.
Taken the same day: `scale = 128` landed in the follow-up commit,
with the probe script as its verification harness. The sizing lesson
above still stands for genuinely-approximate wide inputs (non-integer
bounds, embedding-fed keys) — their per-element deviations are real
and still sum.
