# Univariate-subgraph collapse — design

Status: DESIGN (no implementation yet).
Measurement basis: `scripts/measure_fusion_opportunities.py` at commits
`4e4b59c`..`f482c2d`, run on all eight examples and the production doom
graph (2026-07-05).

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
exists.  The filter kills the calculator wins (their deep subgraphs hang
off full-number scalars, range 2·10⁴–1.25·10⁶) and keeps doom's.

Doom's −10 is concentrated in four narrow-range subgraphs on the
critical path:

| source          | range              | members | chain → 1 | existing lanes |
|-----------------|--------------------|---------|-----------|----------------|
| `split_7`       | [0, 10.15]         | 22      | 8 → 1     | 147            |
| thermometer Attn| [0, 322]           | 13      | 6 → 1     | 152            |
| `sub`           | [−160, 324]        | 3       | 3 → 1     | 7              |
| `table_lookup`  | [−0.6, 658]        | 3       | 3 → 1     | 1152           |

The doom schedule is chain-bound, so these translate to real layers:
modeled critical path 64 sublayers vs the actual 51-layer compile, and
aggregate FFN lane demand is 146,655 of 835,533 (51 × 16,383) — **17.6%
utilization**.  Width is not the constraint; depth is.

## Constraints, stated upfront

- **The pass changes exact-math values on the swish machine.**  A
  swiglu `piecewise_linear` is the exact PL function with each corner
  rounded in a fillet; the collapsed FFN has *one* fillet set where the
  original chain had one per stage.  At in-contract inputs (the plateau
  contract below) outputs are bit-exact; between plateaus values move
  within the documented fillet magnitudes.  The relu machine composes
  exactly everywhere.
- **v1 only collapses staircase subgraphs** (composed function constant
  on plateaus around integers — see *Feasibility gate*).  Continuous-
  source subgraphs (e.g. the table-lookup interpolation chain) are v2.
- **The source's `value_range` must be trusted** — it sizes the
  tabulation domain.  It already is: the same cached bounds drive
  compilation.
- **Requires the lowering-copy machinery**: the pass creates new nodes
  and orphans old ones, which only works because `lower()` operates on
  a throwaway clone and tracks values through `FoldLog` / `node_map`.
  The source graph is never touched.
- **`test_lowering_parity` must learn the pass** (its in-place twin
  applies `lower()`'s pass list — the maintenance rule in that module's
  docstring), and the debug sidecar keeps working through the same
  `record_move` value-tracking fusion already uses.

## The pass

### Placement

Inside `lower()`, **after** `_strip_debug_wrappers` (wrappers are not
per-position ops and would end every subgraph at each assert; the
strip has already tightened their claims onto the wrapped nodes), and
followed by **one more `fuse_consecutive_linears` round**: a collapse
produces `FFN(source)`, and when the source is a Linear whose only
remaining consumers are the synthesized FFNs, the existing
linear-into-gate fold absorbs it — a free extra sublayer (doom's
`split_7` source is exactly this shape).

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

Collapse a subgraph only when, for `w` the plateau slack (default 0.05):

1. **Range budget**: `round(hi) − round(lo) + 1 ≤ lane budget` (below).
2. **Staircase check**: for every integer `k` in `[round(lo),
   round(hi)]` and every boundary member `m`, the exact oracle
   satisfies `f_m(k − w) == f_m(k) == f_m(k + w)` — the composed
   function is constant on plateaus around integers, i.e. a function of
   `round(source)`.

The staircase check is the correctness certificate: the original chain
was only ever consumed with the source at integer ± noise (that is what
its own first-stage ramps assume), and the synthesized staircase is
bit-exact on those plateaus **by construction** — `piecewise_linear`
passes through every knot exactly.  There is no missed-feature risk of
the kind a sampled curve-fit would have: we do not claim the function
between plateaus, and neither did the original chain.  A subgraph that
fails the check is declined, never approximated — worst case is a lost
opportunity, not a wrong compile (D2).

All four measured doom winners are integer-grained and pass this shape
of gate (instance indices, thermometer counts, compare/abs on integer
gaps).  The table-lookup chain's interpolation segment may fail the
plateau check; it stays multi-stage until v2.

### Synthesis

For each boundary member `m` (possibly vector-valued — `emit_*` members
are width-8):

1. **Re-root the oracle**: clone the subgraph members with the source
   replaced by a fresh 1-D `InputNode`.  All members are per-position
   ops, so the entire integer grid evaluates in a single
   `m_clone.compute(n_pos=len(grid), ...)` call.
2. **Emit** `piecewise_linear(source, breakpoints, fn)` with staircase
   breakpoints (step pairs at half-integers, `step_sharpness` input
   scale) and `fn` reading the tabulated grid.  Vector-valued `fn`
   keeps multi-D members at one FFN.  `d_max` = the lane budget, so a
   feasible subgraph never chunks (chunking would re-add depth).
3. **Rewire** every external consumer of `m` to the synthesized node;
   `fold_log.record_move(orphan=m, survivor=new_ffn)`.

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

### Lane budget (the three tiers)

1. **Hard cap** — composed lanes ≤ `d_hidden` of the target compile.
   Above this the FFN cannot be placed in any MLP sublayer
   (`scheduler.py` packs to `d_hidden`; `bias=False` reserves one lane).
2. **Default: always collapse.**  Net lane delta (synthesized lanes −
   lanes freed by orphaning the interior) is **logged, not gated** —
   depth wins are taken by default; the log keeps width creep visible.
   The only decline: composed lanes > `d_hidden / 2` to save a single
   sublayer (one FFN monopolizing a layer's MLP pool for a marginal
   win).
3. **Screening cap `d_hidden / 4`** — expressed as a *fraction* of the
   compile's `d_hidden`, not an absolute (the production 16,384 makes
   4,096 comfortable; a `d=4096`-class compile shrinks it in step).
   Its real job is the noise axis: fp32 accumulation across a PWL's
   lane sum grows with lane count, and past a few thousand lanes we
   exceed every grid in production today (largest single FFN: 1,024).

Doom headroom check (2026-07-05): 17.6% aggregate lane utilization —
tier 2 protects a non-binding constraint today; revisit as a real gate
if utilization passes ~60%.

## What this pass is not (out of scope)

- **Attention affine folds** (a Linear feeding only Attn Q/K/V, or
  reading an Attn output, composed into the head matrices): measured at
  only −3 doom sublayers but 1,040 foldable Linears — a *width/head-
  budget* play, separate design when width pressure appears.
- **Partial collapse of infeasible subgraphs** (radix-split a wide
  source, then collapse per-digit): the calculator examples' upper
  bounds (−9, −17) live here.  v2+.
- **Add-of-Linears normalization**: measured at zero instances on doom.
  Dropped.
- **Multi-variable regions** (functions of two scalars via 2-D grids):
  `multiply_2d`-style machinery exists but the measurement gives no
  evidence of critical-path demand.

## Noise accounting (D7)

`piecewise_linear` is existing, measured hardware — its entries in
`docs/op_noise_data.json` cover the mechanism, and the pass adds no new
op.  What is per-collapse is the *composition*, and the plateau
certificate covers it: bit-exact at every in-contract input, verified
at build time against the exact oracle for every collapse, every
compile.  No static noise entry is added; a paragraph in
`docs/numerical_noise_findings.md` records this reasoning when the pass
lands.

## Verification and rollout

1. **Unit layer (D6)**: build small staircase subgraphs
   (compare→gate→select chains on both machines), collapse, sweep the
   full integer grid ± plateau slack against the original oracle —
   exact match required.  Negative tests: two-source region declined,
   over-budget range declined, non-staircase (interpolating) member
   declined, wrapper/override handling.
2. **Pass-level invariants**: `test_lowering_parity`'s in-place twin
   gains the pass (same round order); compiler invariants I1–I4 are
   untouched (the pass runs before scheduling) but the full suite runs
   with the flag forced on in a one-off sweep before the default flips.
3. **Example parity**: every example compiles with the flag on; token
   outputs identical on the existing example tests.
4. **Doom gate**: `probe_compiled` parity on the lowres config, then
   the pixel-oracle gates (`test_flat_pixel_oracle`,
   `test_forward_ar_rollout`).  Re-run
   `scripts/measure_fusion_opportunities.py` to confirm the realized
   critical path (expect ~54 from 64, plus whatever the post-collapse
   fusion round finds) and re-measure compiled layer count on the doom
   side.
5. **Flip the default** (`collapse_univariate=True`) only after 1–4 are
   green; the flag stays as an escape hatch for one release.
