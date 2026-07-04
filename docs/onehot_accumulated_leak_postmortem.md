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
