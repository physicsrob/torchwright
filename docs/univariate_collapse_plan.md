# Univariate-subgraph collapse — design

Status: DESIGN (no implementation yet).
Measurement basis: `scripts/measure_fusion_opportunities.py` at commits
`4e4b59c`..`f482c2d`, run on all eight examples and the production doom
graph (2026-07-05).  Revised 2026-07-05 after design review: integer
contract added to the gate, lane budget restated in emitted lanes,
lane-budget tiers collapsed to one rule, noise measurement required
before the default flips.

## The idea in one paragraph

A **univariate subgraph** is a set of per-position nodes (FFN, Linear,
Add, Concatenate; literals allowed as extra inputs) whose only
non-literal source is a single 1-D node.  Every member computes a
function of that one scalar, so however deep the subgraph's internal
chain, its externally-consumed values are each *one univariate
piecewise-linear function* of the source — and a single FFN can compute
any such function in one MLP sublayer (`piecewise_linear`, which both
machines already ship).  The collapse pass finds these subgraphs on the
compiler-private copy inside `lower()`, re-synthesizes each
externally-consumed member as one `piecewise_linear` of the source, and
orphans the interior.  Depth drops from the subgraph's chain length to 1.

Attention ends a univariate subgraph (it mixes values across positions);
a node that combines two different sources ends it too (a function of
two variables has no 1-D breakpoint grid).  Both enders can *seed* new
subgraphs downstream when their own output is 1-D.

## Why: the measurements

Modeled critical path in sublayers (Attn/FFN/Linear/Add cost 1,
Concatenate virtual), on the lowered graph — so every number below is
an opportunity the current optimizer does **not** already take:

| graph                  | baseline | collapse (upper bound) | collapse (feasible only) |
|------------------------|----------|------------------------|--------------------------|
| adder_v2               | 25       | −9                     | −0                       |
| calculator_v2          | 38       | −17                    | −0                       |
| sort_digits_v1         | 63       | −26                    | −26                      |
| calculator_scratchpad  | 16       | −2                     | −2                       |
| **doom (production)**  | **64**   | **−15**                | **−10**                  |

"Feasible only" keeps a subgraph un-collapsed when its source range
exceeds a 4096-lane breakpoint budget — integer-grained structure needs
~range-width breakpoints, which is exactly why the radix decomposition
exists.  (That filter is the script's proxy: source range ≤ 4096.  The
pass's real gate is on emitted lanes — see the feasibility gate; all
four doom winners emit well under the cap, so the −10 stands.)  The
filter kills the calculator wins (their deep subgraphs hang off
full-number scalars, range 2·10⁴–1.25·10⁶) and keeps doom's.

Doom's −10 is concentrated in four narrow-range subgraphs on the
critical path:

| source          | range              | members | chain → 1 | existing lanes |
|-----------------|--------------------|---------|-----------|----------------|
| `split_7`       | [0, 10.15]         | 22      | 8 → 1     | 147            |
| thermometer Attn| [0, 322]           | 13      | 6 → 1     | 152            |
| `sub`           | [−160, 324]        | 3       | 3 → 1     | 7              |
| `table_lookup`  | [−0.6, 658]        | 3       | 3 → 1     | 1152           |

"Existing lanes" is the member FFNs' summed hidden lanes as compiled
today — the width the collapse frees by orphaning the interior.

The doom schedule is chain-bound, so these translate to real layers:
modeled critical path 64 sublayers vs the actual 51-layer compile, and
aggregate FFN lane demand is 146,655 of 835,533 (51 × 16,383) — **17.6%
utilization**.  Width is not the constraint; depth is.

## Constraints, stated upfront

- **The pass changes exact-math values on the swish machine.**  A
  swiglu `piecewise_linear` is the exact PL function with each corner
  rounded in a fillet; the collapsed FFN has *one* fillet set where the
  original chain had one per stage.  At in-contract inputs (the plateau
  contract below) outputs are bit-exact — which holds because the
  sharpened gate rows put the fillet's exponential tail ≈ K·0.45 input
  units from every plateau center (K = scale·input_scale), far below
  fp32 resolution; the unit sweep in the rollout verifies this rather
  than assuming it.  Between plateaus values move within the documented
  fillet magnitudes.  The relu machine composes exactly everywhere.
- **v1 only collapses staircase subgraphs** (composed function constant
  on plateaus around integers — see *Feasibility gate*), and only for
  sources carrying an explicit integer contract (`assert_integer` on
  the source — gate condition 1).  Continuous-source subgraphs (e.g.
  the table-lookup interpolation chain) are v2.
- **The source's `value_range` must be trusted** — it sizes the
  tabulation domain.  It already is: the same cached bounds drive
  compilation.
- **Requires the lowering-copy machinery**: the pass creates new nodes
  and orphans old ones, which only works because `lower()` operates on
  a throwaway clone and tracks values through `FoldLog` / `node_map`.
  The source graph is never touched.  The pass must also be a
  deterministic function of the copy — stable iteration order over
  subgraphs and boundary members, synthesized-node names derived from
  source/member names — because the CP-SAT schedule-cache key hashes
  the lowered copy's topology; a nondeterministic pass turns every
  compile into a spurious cache miss.
- **`test_lowering_parity` must learn the pass** (its in-place twin
  applies `lower()`'s pass list — the maintenance rule in that module's
  docstring).  The debug sidecar needs nothing: `OnnxDebugSession`
  never re-lowers, and `debug_fingerprint` deliberately excludes
  compile-side knobs because the sidecar records the residual
  assignment explicitly (keyed by source-graph canonical ids,
  value-tracked through the same `record_move` fusion already uses).
  Orphaned interior members simply have no entry, exactly like fusion
  orphans today.  Recording `collapse_univariate` in the sidecar
  metadata is worthwhile provenance, not a correctness need.

## The pass

### Placement

Inside `lower()`, **after** linear fusion.  (2026-07, post
assert-metadata migration: checks and range claims are node metadata —
docs/assert_metadata_plan.md — so there is no wrapper strip; claims are
refresh-proof at the `refresh_node_caches` choke point and a claim
never ends a subgraph.)  A follow-up **second `fuse_consecutive_linears`
round** after the collapse remains desirable: a collapse
produces `FFN(source)`, and when the source is a Linear whose only
remaining consumers are the synthesized FFNs, the existing
linear-into-gate fold absorbs it — a free extra sublayer (doom's
`split_7` source is exactly this shape).

With claims refresh-proof by construction, that second round is
bounds-safe unconditionally — the strip bookkeeping this section used
to describe (and the claim-loss trap that made an unconditional
post-strip round unsafe) no longer exists.  The remaining question —
which folds may delete runtime checkability — is the explicit
fold-decline policy in `graph/optimize.py`, revisitable on
measurement.

Gated by a `collapse_univariate: bool = False` keyword threaded from
the compile entry points (`compile_headless`, `compile_to_onnx`).
Default flips to on after the doom parity gate passes (rollout below).

### Finding subgraphs

Walk the copy in topological order, computing for each node the single
1-D node it is a pointwise function of (the algorithm already lives in
`scripts/measure_fusion_opportunities.py:_scalar_sources`): literals
map to "no source", non-pointwise or input-less nodes seed a new
subgraph iff 1-D, pointwise nodes inherit their inputs' common source
or end the subgraph when two sources meet.  A subgraph is the set of
nodes sharing a source.  Note the shape is a DAG, not a chain — the
finder flood-fills branches that fork off the source and reconverge.

**Members that must survive** (the *boundary*): any member with a
consumer outside the subgraph, or that is the graph output.  Interior
members are orphaned wholesale.

### Feasibility gate (v1: the plateau contract)

Collapse a subgraph only when all four conditions hold, for `w` the
plateau slack (default 0.05 — justified below):

1. **Integer contract on the source**: the source carries an
   `assert_integer` claim in the user graph (`node.integer_claim` —
   metadata that rides the compiler-private copy; `NodeValueType`
   stays range-only; no type-system change).  This
   is the half of the certificate the composed-function check cannot
   supply: a source whose legitimate values land on half-integers has
   a composed function that is constant near every *integer* — the
   staircase check below passes — yet is not a function of
   `round(source)`, and the synthesized staircase would return
   mid-ramp values at inputs the source actually takes.  Requiring
   the assert closes that hole, and buys the runtime backstop for
   free: `debug=True` re-checks the predicate on compiled values, so
   a contract violation on real inputs fires an assert instead of
   silently rendering garbage.  If a doom source lacks the assert
   today, adding it is part of landing the pass (and independently
   useful).
2. **Plateau pre-screen**: `round(hi) − round(lo) + 1 ≤` the lane cap
   (below).  Cheap; runs before tabulation.
3. **Lane gate, on emitted lanes**: after tabulating the composed
   function on the integer grid, the synthesized staircase costs one
   hidden neuron per slope change — two per step — so
   `lanes = 2 × #(adjacent plateau pairs whose values differ in any
   output dimension)`; steps where no output dimension changes are
   free.  Decline when `lanes >` the cap.  The pre-screen alone is
   not enough: it counts plateaus, and N plateaus can emit ~2N lanes
   — past `d_max` the op would chunk into parallel FFNs plus an Add,
   re-adding exactly the depth the pass exists to remove.  `d_max` is
   set at the cap, so a collapse that passes this gate never chunks.
4. **Staircase check**: for every integer `k` in `[round(lo),
   round(hi)]` and every boundary member `m`, the exact oracle
   satisfies `f_m(k + δ) == f_m(k)` for sampled offsets `δ` spanning
   `[−w, +w]` (half a dozen offsets per plateau; the whole sweep is
   still one batched `compute` call) — the composed function is
   constant on plateaus around integers, i.e. a function of
   `round(source)`.

The certificate, stated honestly: the synthesized staircase passes
through every knot exactly by construction, so at tabulated integers
the collapse is bit-exact against the oracle — no sampled curve-fit
risk there.  Plateau *constancy* of the original composed function,
however, is a sampled check: a function could in principle match at
every sampled offset and still wiggle between them (two canceling
transitions inside the band; interior gain stages can compress feature
widths in source coordinates below any fixed sampling density).  No
graph shape we build produces that — step constructions put their
transitions at half-integers — but what the gate certifies is
equality at the sampled offsets, no more.  The realistic near-miss (a
compare threshold sitting exactly at an integer) lands inside the
sampled band and is caught.  A subgraph that fails any condition is
declined, never approximated — with condition 1 in place the worst
case is a lost opportunity, not a wrong compile (D2).

**Why w = 0.05.**  Not arbitrary: it is `1/(2·step_sharpness)` — half
the transition-band width of the machine's own step constructions
(`step_sharpness = 10.0` in `ops/const.py`; step pairs ramp over
`1/step_sharpness = 0.1`).  The check must also cover the source's
real noise band (e.g. softmax leakage on attention-output sources);
if a source's contract allows more than `w`, widen `w` for that
subgraph's check.

All four measured doom winners are integer-grained and pass this shape
of gate (instance indices, thermometer counts, compare/abs on integer
gaps) — pending the condition-1 audit that each source actually
carries its `assert_integer`.  The table-lookup chain's interpolation
segment may fail the plateau check; it stays multi-stage until v2.

### Synthesis

Boundary members are treated individually, and only where collapse
buys depth: a member sitting at depth 1 above the source (its inputs
are the source and literals only, so no interior member must survive
on its behalf) is **kept as-is** — re-synthesizing it would trade
nothing for new lanes and, on the swish machine, a changed fillet
set.  Members at depth ≥ 2 are synthesized; a subgraph with no
depth-≥ 2 boundary member is skipped entirely (the pass-side analogue
of the measurement script's `crit_chain == 0` skip).

For each synthesized boundary member `m` (possibly vector-valued —
`emit_*` members are width-8):

1. **Re-root the oracle**: clone the subgraph members with the source
   replaced by a fresh 1-D `InputNode`.  All members are per-position
   ops, so the entire integer grid evaluates in a single
   `m_clone.compute(n_pos=len(grid), ...)` call.
2. **Emit** `piecewise_linear(source, breakpoints, fn)` with staircase
   breakpoints (step pairs at half-integers, `step_sharpness` input
   scale) and `fn` reading the tabulated grid.  Vector-valued `fn`
   keeps multi-D members at one FFN.  `d_max` = the lane cap; the
   gate's emitted-lane condition guarantees a passing subgraph never
   chunks.
3. **Rewire** every external consumer of `m` to the synthesized node;
   `fold_log.record_move(orphan=m, survivor=new_ffn)`.

Synthesis iterates subgraphs and boundary members in the copy's
topological order and names synthesized nodes from the source and
member names — the determinism constraint above, made concrete.

One synthesized FFN **per boundary member**, not one shared multi-output
FFN: consumers read whole nodes, so a shared FFN would need slice
Linears behind it (depth 2, defeating the point).  Boundary members
duplicate the breakpoint grid's lanes; with 82% lane headroom and
typically 1–2 boundary members per subgraph, that trade is taken
silently in favor of depth.

Interior members are dropped from `node_map` (their values cease to
exist — the same handling fusion's `value_changed`/orphan cases already
get).  A semantic affine override on an interior member dies with it
(safe: the synthesized node's bounds come from the tabulated min/max,
which is tighter than anything propagation derived); an override on a
boundary member describes a value the collapse preserves and transfers
to the synthesized node.

### Lane cap

Two decline rules and one assertion, on **emitted lanes** (gate
condition 3's count, not plateau count) and on the **predicted
accumulation error**:

- **Decline** when emitted lanes exceed **`d_hidden / 4`** of the
  target compile (production: 4,096 of 16,384).  Expressed as a
  fraction, not an absolute, so a `d=4096`-class compile shrinks it in
  step.  This keeps a single FFN from monopolizing a layer's MLP pool.
  (An earlier draft had a second threshold — decline above
  `d_hidden / 2` when saving a single sublayer — which is unreachable
  behind this cap and is dropped.)
- **Decline on the error model** (2026-07-05 measurement — the
  staircase entries in `docs/op_noise_data.json` and their finding in
  `docs/numerical_noise_findings.md`): a staircase's fp32 error is
  governed by `ulp_fp32(step_sharpness · R · max|Δv|)` — R the source
  range width, |Δv| the largest adjacent-plateau value change — **not
  by lane count**; saturated ramp lanes carry intermediates of that
  magnitude and the lane sum quantizes to its ulps (measured maxima
  are 1–4 ulps, bit-identical on both machines).  Decline when
  `4 × ulp_fp32(step_sharpness · R · max|Δv|)` exceeds the synthesized
  claim's tolerance (the `assert_matches_value_type` default atol,
  1e-3 — v1 declines rather than widen it; sizing the claim from the
  bound is a v2 refinement).  Both R and max|Δv| are known exactly at
  tabulation time, and the measured maxima sit at or under the bound
  at every scale (exactly at it for the 2048-lane point).  A v2
  mitigation if a real collapse declines here: re-center the source
  (breakpoints relative to the range midpoint), halving R.
- **Assert** emitted lanes ≤ `d_hidden − 1` (`bias=False` reserves
  hidden slot 0; `scheduler.py` packs to `d_hidden`).  Unreachable
  given the decline rule; it stays as a structural guard.

Net lane delta (synthesized lanes − lanes freed by orphaning the
interior) is **logged, not gated** — depth wins are taken by default;
the log keeps width creep visible.  Declines are logged too, with the
failing condition (for staircase failures: the member and the integer).
Doom headroom check (2026-07-05): 17.6% aggregate lane utilization —
width is not binding today; revisit the always-collapse default if
utilization passes ~60%.

## What this pass is not (out of scope)

- **Attention affine folds** (a Linear feeding only Attn Q/K/V, or
  reading an Attn output, composed into the head matrices): measured at
  only −3 doom sublayers but 1,040 foldable Linears — a *width/head-
  budget* play, separate design when width pressure appears.
- **Re-seeding declined subgraphs at narrow interior scalars** (v1.5
  candidate): the finder attributes every pointwise node to the
  deepest seed, so a narrow-range interior 1-D node — a mod-10 output
  with range [0, 9] inside a chain hanging off a range-10⁶ scalar —
  never seeds its own subgraph, and declining the wide subgraph
  forfeits the collapsible chains downstream of it.  Promoting a
  declined subgraph's narrow-range interior scalars to seeds and
  re-running the finder would recover part of the calculator upper
  bounds (−9, −17) without radix machinery; measurable today with a
  small extension to the measurement script.
- **Partial collapse of infeasible subgraphs** (radix-split a wide
  source, then collapse per-digit): the rest of the calculator upper
  bounds live here.  v2+.
- **Add-of-Linears normalization**: measured at zero instances on doom.
  Dropped.
- **Multi-variable regions** (functions of two scalars via 2-D grids):
  `multiply_2d`-style machinery exists but the measurement gives no
  evidence of critical-path demand.

## Noise accounting (D4)

`piecewise_linear` is an existing op and the pass adds no new one, but
the shipped measurement does not cover the operating point: the relu
entry in `docs/op_noise_data.json` was measured on the canonical x²
test function with 11 breakpoints, while a collapsed staircase emits
hundreds to thousands of lanes (doom's `sub` → ~970, thermometer →
~646; the cap allows 4,096).  fp32 accumulation across the lane sum
grows with lane count, and past ~1,024 lanes — the largest single FFN
in production — nothing is measured.  This is not D7's trigger (no op
implementation or breakpoint grid changes); it is D4: flipping the
default on an unmeasured noise foundation.

So, reversing this design's earlier "no static noise entry" position
(its premise — `piecewise_linear` is "existing, measured hardware" —
fails at these lane counts): `scripts/measure_op_noise.py` gains a
staircase distribution at representative lane counts (64 / 512 /
2048), regenerated via `make measure-noise`, **before the default
flips**.  What remains per-collapse is the *composition*, and the
sampled plateau certificate covers it: verified at build time against
the exact oracle for every collapse, every compile.  A paragraph in
`docs/numerical_noise_findings.md` records this reasoning when the
pass lands.

## Verification and rollout

1. **Unit layer (D6)**: build small staircase subgraphs
   (compare→gate→select chains on both machines), collapse, sweep the
   full integer grid at the sampled plateau offsets against the
   original oracle — exact match required on relu; on swish, exact
   match at plateau centers (this is what verifies the
   fillet-tail-underflow claim in *Constraints*).  Negative tests:
   two-source region declined; over-budget range declined;
   emitted-lane overflow declined (pre-screen passes, lane gate
   declines); non-staircase (interpolating) member declined; source
   without `assert_integer` declined; half-integer-grained source
   declined (the gate-condition-1 counterexample); wrapper/override
   handling; depth-1 boundary member kept as-is.
2. **Pass-level invariants**: the unconditional post-collapse fusion
   round lands as its own change with `test_lowering_parity`'s
   in-place twin updated; the twin then gains the collapse pass (same
   round order).  Compiler invariants I1–I4 are untouched (the pass
   runs before scheduling) but the full suite runs with the flag
   forced on in a one-off sweep before the default flips.
3. **Example parity**: every example compiles with the flag on; token
   outputs identical on the existing example tests.
4. **Doom gate**: `probe_compiled` parity on the lowres config, then
   the pixel-oracle gates (`test_flat_pixel_oracle`,
   `test_forward_ar_rollout`).
5. **Noise measurement — REQUIRED before the default flips.**  The
   staircase distribution lands in `scripts/measure_op_noise.py`;
   `docs/op_noise_data.json`, the generated markdown, and
   `docs/numerical_noise_findings.md` are regenerated/updated via the
   standard `make measure-noise` workflow (see *Noise accounting*).
6. **Depth report — REQUIRED before the default flips.**  The point of
   the pass is fewer layers, so measure it directly, not by proxy: for
   **every example graph and the production doom graph**, compile with
   the flag off and on and report the **compiled layer count**
   (`n_layers` on the artifact / `CompiledHeadless`) side by side —
   before, after, delta — in the landing PR/commit message, together
   with the pass's own collapse/decline log (per subgraph: source,
   members, chain → 1, emitted lanes; or the decline reason).  Also
   re-run `scripts/measure_fusion_opportunities.py` on both settings so
   the modeled critical path (expect ~54 from doom's 64, plus whatever
   the post-collapse fusion round finds) can be checked against the
   realized layer delta; a modeled saving that does not show up in real
   layers means the schedule is not chain-bound where we thought, and
   that discrepancy is a finding to explain (D4), not to skip.
7. **Flip the default** (`collapse_univariate=True`) only after 1–6
   are green.  The flag stays as an escape hatch until the doom
   pixel-oracle gates have run green with the default on; then it is
   deleted — no dormant knobs.
