# Constants as just-in-time nodes — implementation plan

**Status:** proposed (not started)
**Author:** Rob Porter
**Date:** 2026-06-03
**Scope:** `torchwright` compiler — `LiteralValue` lifecycle.

---

## 1. Problem

A `LiteralValue` (a compile-time constant node, `torchwright/graph/misc.py:114`)
is currently classified as an **input node** by
`GraphAnalyzer.is_input_node` (`torchwright/compiler/forward/graph_analysis.py:203-204`,
which lumps it with `Embedding`, `PosEncoding`, `InputNode`). The consequences,
traced through the forward-compile pipeline:

1. **Pre-allocated at layer 0.** Every constant is handed a residual-stream
   column before scheduling begins (`compile.py:517-521`) and its value is
   baked into the initial residual stream at forward time
   (`transformer.py:91-93`).
2. **Pre-seeded as computed.** All input nodes — constants included — are put
   into the scheduler's `computed` set up front (`compile.py:534`), so a
   `LiteralValue` never enters the `ready` queue.
3. **Held until its last consumer.** The *eager* free path
   (`scheduler._freshly_dead_inputs`, `scheduler.py:988-1022`) explicitly
   refuses to reclaim input nodes (`scheduler.py:1015-1022`). The constant's
   column is reclaimed only later, by the non-eager dead-node cancellation path
   (`_find_dead_nodes` → `cancel_candidates` → `residual_map.free`,
   `scheduler.py:151, 415, 449`), once every consumer has run.

**Net effect.** A constant first needed deep in the network — the motivating
case `select(cond, x, LiteralValue([0.0]))` at layer *N* — needlessly reserves
a scarce residual column across the *entire* span layers `0…N`, even though the
value is read once, at the end.

By **residual pressure** I mean the number of residual-stream columns
simultaneously occupied by live nodes, and its peak across layers. The stream
has fixed width `d`; whenever the peak would exceed `d`, the scheduler must
spread work across more layers (or `d` must grow). So a column reserved from
layer 0 for a late-consumed constant raises occupancy across the whole `0…N`
span, which can force a higher layer count or a larger `d`. For a graph with
many such constants this is a real and avoidable tax.

### 1.1 The fix already exists but is unreachable

The compiler *already* contains the machinery to materialize a constant mid-network:

- the scheduler op `compute_literal_value` (`scheduler.py:787-814`), and
- its weight writer `_write_compute_literal_value` (`weight_writer.py:661-673`),
  which writes the constant into the MLP's `linear2.output_bias`.

This path is **dead code** today: because constants are pre-seeded as computed
(point 2 above), a `LiteralValue` never becomes `ready`, so the op is never
emitted. The investigation confirmed the write path itself is correct (the
output bias is zero-initialised per layer, the target columns are disjoint, the
skip-connection column is zeroed by the *dirty-cancel* before the bias write).

A **dirty-cancel** is the existing compiler mechanism that zeroes a column whose
current contents are stale — either a column recycled from a just-freed node, or
a never-written column from the initially-"dirty" free pool — by adding the
negative of its current value via an attention head, so a subsequent *additive*
write (here the bias) lands on a clean zero. It is not specific to constants;
every fresh allocation uses it. The same mechanism, when it fires in the layer a
node is first allocated ("born"), is what §4.3/Phase 2 calls the **birth
dirty-cancel**.

---

## 2. Decision

Make `LiteralValue` a **first-class schedulable node everywhere** — remove it
from `is_input_node` and let it be materialized just-in-time via the existing
`compute_literal_value` op, then freed by the normal dead-node path. One mental
model ("a constant is a node"), two ways of choosing *when* to materialize it:

- **Heuristic scheduler (`optimize=0`, the default/fast path).** The greedy
  scheduler is eager, so it needs an explicit **just-in-time gate**: a constant
  is materialized in the layer where a consumer's *other* (non-constant) inputs
  land, which adds no latency to that consumer's readiness (argued in Phase 1,
  task 3 below), and is freed right after.

- **CP-SAT scheduler (`optimize>0`).** No gate needed. A constant has no inputs,
  so its birth layer is free to float in `[0, min_consumer_layer − 1]`, and its
  residual cost is its live interval `[birth, cancel]` × width (Phase 2, task 1).
  A *later* birth shortens that interval, so it **weakly dominates**: it never
  raises the objective, and it strictly lowers residual pressure *when the
  schedule is residual-bound*; because a constant has no inputs, nothing
  downstream benefits from an earlier birth. So an optimal schedule with
  just-in-time placement always exists, and under residual pressure it is the
  *unique* optimum. **Caveat (verified in implementation):** absent residual
  pressure, later birth only *weakly* dominates — the solver is indifferent
  among equal-objective placements and may materialize a constant early (even at
  layer 0). That is harmless: with slack columns the early placement is
  uncontended and the layer count is unchanged. So the residual win is captured
  exactly where it matters (pressure-bound graphs) and is a no-op where it
  doesn't — unlike the heuristic, whose explicit gate defers *unconditionally*.
  The cost/timing machinery already generalises (Phase 2); the cost/routing
  helpers already have `LiteralValue` branches written
  (`cpsat_scheduler.py:378, 414, 467, 482`), currently dead for the same reason.

- **Replay (`DirectedLayerScheduler`).** Once CP-SAT assigns the constant a
  layer (Phase 2), the constant replays like any other scheduled node.

Both paths reduce to the same `compute_literal_value` op and the same dead-node
free path.

### 2.1 Why not "fold constants into the consumer's bias" first

A constant feeding a bias-carrying `Linear` *could* be absorbed into that
linear's bias at graph-rewrite time (`output_bias += const @ output_matrix[const_rows]`),
needing **zero** residual columns. But foldability is **partial**: `Attn` has no
bias (`attn.py:73-83`) and `Add` has no bias, so the many attention-fed
constants (≈17 sites in `attention_ops.py` alone) cannot fold and *must* be
materialized. Just-in-time materialization is therefore necessary regardless and
fully solves the stated problem; folding only shaves the *brief* (one column for
~1–2 layers) footprint of the linear-fed subset to exactly zero, at the cost of
a real graph-rewrite pass (`Concatenate`+`Linear` surgery, plus wiring up the
currently-uncalled `optimize_graph` hook in `optimize.py`). Folding is deferred
to a **measured** Phase 3 (§5.4).

### 2.2 Why not bolt JIT onto CP-SAT's output ("overlay"), or gate `optimize>0` off

- **Overlay** (keep constants as full-span sources in the solver, inject
  materialization only at replay): CP-SAT would plan against the old pessimistic
  residual budget, never *see* the freed columns, and so capture **none** of the
  layer-count benefit on exactly the literal-heavy graphs we care about — and
  could deem the heuristic warm-start (which does free those columns) infeasible
  and fall back to greedy (`compile.py:678-704`). A "headroom margin" variant is
  an unprincipled fudge: too small → replay overflows `d`; too large → wastes the
  benefit.
- **Gate `optimize>0` off:** regresses existing CP-SAT compiles that use
  constants and leaves the better scheduler unable to handle a basic node type
  (a D4 foundation crack).

The full reasoning is recorded in the session that produced this plan.

---

## 3. Atomicity and the phased seam

`is_input_node` is a **single switch** shared by the compile-time setup
(`compile.py` pre-allocation/pre-seeding) and *both* schedulers. Flipping it for
constants is therefore naturally one atomic change: the moment constants leave
`is_input_node`, the heuristic, the CP-SAT model, and the replay path all see
the change together. The phases below are still genuine go/no-go gates — they
gate *whether to build the next mechanism*, not whether main compiles.

**Recommended delivery: one branch, internal gate, land together.** Develop
Phase 1 (heuristic just-in-time) and Phase 2 (CP-SAT modeling) on a single
branch. `optimize>0` is temporarily broken *on the branch* during Phase 1
development (constants left `is_input_node` but CP-SAT doesn't model them yet) —
acceptable because it is never merged in that state. Gate 1 → 2 is a real
decision point: verify the heuristic path delivers the residual win and is fully
correct before investing in the CP-SAT modeling. Land Phases 1 and 2 together so
**main never has a broken `optimize>0`** (D4). This is the default precisely
because it never reintroduces a rejected shape (see next paragraph).

**Optional alternative: a temporary source shim, only if Phase 1 must merge
independently.** If there's a hard requirement to ship the heuristic path to main
before Phase 2 exists, keep `optimize>0` correct in the interim with a throwaway
shim: in `cpsat_scheduler.build_graph_model`, compute the source set as
`is_input_node(n) or isinstance(n, LiteralValue)` so the solver keeps treating
constants as full-span sources, while the heuristic gate materializes them
just-in-time in *both* `LayerScheduler` and `DirectedLayerScheduler`.

> **Be explicit about what this is.** Mechanically, this interim state *is* the
> "overlay" approach §2.2 rejected (solver plans constants as full-span sources;
> replay materializes them just-in-time). We rejected overlay as a *permanent*
> design because it forgoes the residual win forever; here it is only a
> *transitional scaffold*, removed in Phase 2, accepted solely to keep
> `optimize>0` correct between merges. It is not a silent reuse of the rejected
> mechanism — it is the rejected mechanism knowingly used as a one-release prop.
>
> **Safety is by-construction, to be confirmed at the gate, not yet proven.** The
> argument: the solver reserves each constant a full-span column `[0, end]`,
> which is the maximal possible window, so any actual materialization window is a
> sub-interval and replay cannot need more columns than the solver reserved. That
> argument has *not* been validated by implementation — in particular, that the
> `DirectedLayerScheduler` constant pass-through allocates every constant from the
> reserved headroom within its window, for *every* constant and not just the
> common case, must be **confirmed by the Phase-1 gate's `optimize>0` tests**
> (§5.6: feasibility + `probe_compiled` + I1–I4 on literal graphs). The heuristic
> fallback (`compile.py:678`) is the backstop if a feasibility edge case slips
> through. Until that gate passes, treat the shim's safety as argued, not
> guaranteed.

Because the shim revives a rejected shape and carries an unproven integration,
**land-together is preferred** unless an independent Phase-1 merge is genuinely
required.

---

## 4. Phases

Each phase ends with an explicit **GATE**: proceed only if every listed
condition holds.

### Phase 0 — Baseline, reproducer, measurement harness

No production code changes. Establish the metric the later gates are judged on.

**Tasks**
- Add a measurement probe (a test, not a `/tmp` script — D8) that compiles a
  graph with a constant consumed deep in the network (a `select(cond, x, lit)`
  at a late layer) and reports, per layer, how many residual columns are held by
  `LiteralValue` nodes, plus the total layer count. Use `probe_residual` /
  `residual_map` introspection (`torchwright/debug/probe.py`).
- Capture the same numbers for a representative real graph if one is reachable
  from `torchwright` tests; otherwise note that the DOOM-graph baseline is taken
  in `torchwright_doom` as a follow-up.
- Write the **regression test** that encodes the bug: assert that, *today*, the
  constant's column is occupied during early layers (this test is rewritten in
  Phase 1 to assert the opposite — it is the test that would have caught the
  problem and guards against its return).

**GATE 0 → 1**
- Baseline residual-occupancy and layer-count numbers recorded in this doc.
- The reproducer test exists and demonstrates the `0…N` occupancy span.

---

### Phase 1 — Heuristic just-in-time (`optimize=0`) + safe `optimize>0`

The substance of the fix for the default path.

**Tasks**
1. **Reclassify.** Remove `LiteralValue` from `is_input_node`
   (`graph_analysis.py:204`), leaving `(Embedding, PosEncoding, InputNode)`.
2. **Stop pre-allocating / pre-seeding constants.** `compile.py:477` then
   excludes constants from `input_nodes`, so the pre-allocate loop
   (`517-521`), the `computed` seed (`534`), and the `input_indices` capture
   (`737-742`) skip them automatically. Verify no `get_indices`/`KeyError` for a
   now-unallocated constant (blast-radius risk).
3. **Just-in-time gate (heuristic).** In the MLP literal branch
   (`scheduler.py:787-814`), replace "schedule every ready constant" with:
   materialize a constant **iff** some consumer `C` has all of its *non-constant*
   effective inputs already in `computed_nodes` (including nodes scheduled
   earlier this same layer). Build a `LiteralValue → consumers` map once in
   `LayerScheduler.__init__` (via `graph.get_consumers`). Constants not yet
   eligible stay in `ready` and are retried next layer. A constant with several
   consumers becomes eligible at the **earliest** layer any one consumer's
   non-constant inputs are satisfied, and is held until its **last** consumer
   (matching the shared-constant test, §5.2).
   - *Zero added latency to the consumer (argued).* Premise: with the constant
     pre-placed at layer 0 (today's behavior), it is available from the start, so
     it is *never* the last-arriving input — a consumer `C`'s ready-layer is
     determined entirely by `C`'s latest *non-constant* input. Now: a constant
     written in layer `L`'s MLP is visible from `L+1`; gating on "`C`'s
     non-constant inputs satisfied incl. this layer" materializes the constant in
     the same layer `L` that `C`'s last non-constant input lands. So both the
     constant and that last sibling are in `computed` after `L`, and `C` is ready
     at `L+1` — the **same** layer as in the pre-placed baseline (where the same
     sibling was the binding input). The gate therefore adds no latency to `C`'s
     readiness. *Caveat:* the materialization itself adds a bias write — and, if
     the column is dirty, one cancel head — to layer `L`; under a tight per-layer
     head/slot budget that is a small second-order cost which could perturb
     packing. The Phase-1 measurement (layer count vs baseline) is the guard.
   - *No deadlock:* a pure-constant consumer (e.g. `Add(lit, lit)`) has an empty
     non-constant input set ⊆ `computed`, so it is eligible immediately.
4. **Freeing.** With `LiteralValue` no longer an input node, the eager path
   (`_freshly_dead_inputs`) and the dead-node path free it automatically; no
   change to those methods. Confirm via I4 (column liveness) that no constant is
   freed before its last consumer.
5. **Remove dead prefill branches.** Drop (or guard with an
   `assert node in in_state`) the now-unreachable `LiteralValue` branches in
   `transformer.get_input_res_stream` (`transformer.py:91-93`) and the in-state
   handlers in `export.py:632, 825`. The `assert False, "Unsupported node type"`
   (`transformer.py:111`) must not be reachable by a constant.
6. **`optimize>0` during Phase 1.** Under the recommended land-together delivery
   (§3), no action here — Phase 2's CP-SAT modeling lands in the same merge, so
   `optimize>0` is broken only on the unmerged branch. *Only* if Phase 1 must
   merge independently, apply the temporary §3 source shim and give
   `DirectedLayerScheduler._get_ready_nodes` (`scheduler.py:1298-1306`) a
   pass-through for constants (outside the CP-SAT assignment, handled by the
   shared heuristic gate) — with the shim's "argued, not yet proven" safety
   confirmed by this gate's `optimize>0` tests (§5.6).
7. **Update tests that pre-allocate constants** to the new model
   (`test_scheduler.py:904-954`, `test_residual_map.py:93`,
   `test_weight_writer.py`; see §6).
8. **`probe.py`.** Update `probe.py:370` so a `LiteralValue` is checked as a
   computed node (it now materializes mid-network) rather than skipped as a
   source — strengthens the D2 oracle.

**GATE 1 → 2** (proceed to CP-SAT modeling only if all hold)
- **Correctness:** full suite green under `make test`; `compiled(inputs,
  debug=True)` self-consistency clean on literal-containing graphs; invariants
  I1–I4 do not fire; `probe_compiled` reports `first_divergent is None` (within
  a justified `atol`) on the literal test graphs under `optimize=0` *and*
  `optimize=1`.
- **Benefit:** the Phase-0 regression test now shows the constant's column **free
  during early layers** (span `0…N` collapses to a window around the consumer),
  with layer count no worse than baseline.
- **Decision (whether to build Phase 2 now):** estimate the residual win CP-SAT
  modeling would capture for representative `optimize>0` compiles — i.e. how much
  the freed constant columns would let the solver tighten the schedule beyond
  what it plans while still treating constants as full-span. If that win is
  material (or `optimize>0` is on the critical path for any target graph), build
  Phase 2 and land it together with Phase 1 (default). If negligible and an
  independent Phase-1 merge is wanted, ship Phase 1 with the temporary shim and
  schedule Phase 2 as a follow-up (the shim is throwaway, so closing it out
  remains the eventual default).

---

### Phase 2 — CP-SAT first-class constants

Capture the `optimize>0` residual benefit so both schedulers share one model.
(If Phase 1 shipped with the temporary §3 shim, this removes it.)

**Tasks**
1. **Schedulable, not source.** In `build_graph_model`
   (`cpsat_scheduler.py:272-283`) classify `LiteralValue` into `schedulable`,
   out of `input_nodes`/`pinned_nodes` (`339-340`), and out of the
   `input_residual` reservation (`595-598`) — removing the §3 shim if it was
   applied. `available_residual` grows; the constant's residual demand becomes
   the interval `[layer_var, cancel_layer]`
   like any node.
2. **Dependency edges auto-enable.** Edges `constant → consumer` are already
   built (`295-310`); the constraint is skipped today only because the source is
   an input node (`638-641`). Once the constant is schedulable, the constraint
   switches on, and because the constant routes to MLP the same-layer rule
   forces `layer[consumer] ≥ layer[constant] + 1` — the correct
   MLP-visible-next-layer timing (`642-657`).
3. **Birth dirty-cancel.** Confirm the fresh-allocation birth cancel
   (`846-874`, gated by `assume_zero_init`) generalises to constants without
   special-casing. **This is the one seam to read carefully** before declaring
   the CP-SAT delta "small"; if it does not generalise cleanly, model the
   constant's birth cancel explicitly here.
4. **Warm-start hints.** Feed the heuristic's just-in-time constant layers into
   `hint_layers` so CP-SAT solves fast. This is a solve-*speed* optimization,
   not a correctness requirement (the solver finds the placement without hints),
   and may be split into a follow-up.
5. **Routing/cost helpers** already cover `LiteralValue`
   (`routing`→MLP, `slots_for`→0, `heads_for`→0, `uses_residual`→True). Confirm,
   don't rewrite.

**GATE 2 → 3** (gate on whether folding is even worth pursuing)
- **Correctness:** full suite green under `make test`; under `optimize ∈
  {1,2,3}`, CP-SAT produces feasible schedules on literal graphs; replay via
  `DirectedLayerScheduler` matches; `probe_compiled` clean; the heuristic
  fallback (`compile.py:678`) still works when the solver finds nothing.
- **Benefit:** `optimize>0` now shows the residual win (fewer reserved columns;
  layer count ≤ Phase-1 on literal-heavy graphs).
- **Measurement for Phase 3:** with JIT in place, measure how many constant
  columns are simultaneously live per layer on the DOOM graph and whether those
  brief footprints push the layer count up. **Proceed to Phase 3 only if** that
  cost exceeds an agreed threshold (e.g. ≥1–2 layers attributable to
  simultaneously-live constant columns). Otherwise **stop** — JIT is sufficient.

---

### Phase 3 — (gated, optional) Constant folding into consumer bias

Only if Gate 2 → 3's measurement justifies it.

**Tasks (sketch — detailed design deferred until the gate opens)**
- Add a graph-rewrite pass in `optimize.py` (and wire `optimize_graph` into the
  compile pipeline — it is currently never called): for a `Linear` whose input
  is a `Concatenate` containing `LiteralValue` leaves (or a direct
  `LiteralValue` input), precompute `output_bias += const @ output_matrix[rows]`,
  delete those rows from `output_matrix`, and remove the leaf from the
  `Concatenate`.
- Handle the seams: `Concatenate`-transparency in the scheduler; the
  *chain detector* — a "chain" is the `L1 → ReLU → L2` pattern (a `Linear`
  feeding a `ReLU` feeding a `Linear`) that the compiler fuses into one MLP
  sublayer via `linear_relu_linear`; folding a constant out of L1's
  `Concatenate` input reshapes L1, so the detector must still match the chain;
  and shared-leaf cases (a constant feeding both a `Linear` and an `Attn` folds
  only for the linear path and still needs JIT for the rest).
- JIT (Phases 1–2) remains the fallback for every non-foldable constant.

**GATE 3 → done**
- Folding reduces simultaneously-live constant columns / layer count by the
  measured-and-agreed amount, with full suite green and `probe_compiled` clean.

---

## 5. Testing strategy

The user's explicit requirement: **this must be well tested.** Tests are layered
from smallest reproducer upward (D6), each runs via `make test` /
`make test-local FILE=…` (never raw pytest — see `torchwright/CLAUDE.md`), and
the existing compiler invariants do the structural policing for free.

> **Note — no op-noise update needed (scoped).** `compute_literal_value` is an
> *exact* bias write (assignment onto a per-layer zero-initialised output bias),
> not a piecewise-linear approximation, and has no entry in
> `docs/op_noise_data.json`. D7 governs *piecewise* ops' breakpoint grids, which
> this is not. The exemption also holds *downstream*: the constant value reaching
> each consumer is identical to today's (an exact prefill value becomes an exact
> bias-written value), so no consumer's piecewise op sees a changed input
> distribution and no measured noise figure moves. The one new mechanism on the
> value's path — the dirty-cancel that zeroes the column before the bias write —
> is the *pre-existing* fresh-allocation cancel (an exact `a + (−a)` via an
> attention head), not a new approximate op. So changing this path does **not**
> trigger `make measure-noise`. The §5.5 `probe_compiled` oracle comparison is
> the guard: if any of this reasoning is wrong, it surfaces as a divergence.
>
> *Verified during implementation:* the noise harness (`measure_op_isolated`)
> measures each op via the graph **oracle** (`output_node.compute()`), **never**
> `forward_compile` — so a compiler-only change like this one cannot move any
> measured op-noise figure, by construction. (`reciprocal`, which has zero
> `LiteralValue` nodes, measures byte-identically before and after; an
> intermittent Modal-only drift observed for it is pre-existing
> cross-microarchitecture fp variance at the drift test's 30% tolerance, not a
> consequence of this change.)

### 5.1 Op-level (`tests/compile/forward/test_weight_writer.py`)
- **Dirty-reuse case (currently untested — the investigation's main gap).**
  Allocate a node, compute it, free it, then allocate a `LiteralValue` to those
  *same* (now dirty) columns; initialise the residual stream non-zero; assert
  the dirty-cancel zeroes the columns and the bias write yields exactly the
  constant. This is the path Phase 1 makes load-bearing.
- Width/truncation: keep the existing I2 assertion test
  (`len(target_cols) == value.numel()`).
- Optional hardening: assert `output_matrix[:, target_cols]` is zero in
  `_write_compute_literal_value` (catches a future overlapping-allocation bug,
  since the bias write is assignment, not `+=`). **If added, this is a new
  compiler assertion and must ship with a matching negative test in
  `tests/compile/forward/test_compiler_assertions.py`** (the project rule: no new
  assertion without a negative test) — so prefer adding it only if §5.5's oracle
  comparison proves insufficient to catch the overlap it guards against.

### 5.2 Scheduler-level (`tests/compile/forward/test_scheduler.py`)
- **Deferral:** a constant whose only consumer is a late node is *not* scheduled
  at layer 0; it is materialized at/just before the consumer's layer.
- **Zero added latency:** the consumer's scheduled layer equals that of an
  otherwise-identical graph whose constant input is replaced by a pre-computed
  node (proves the gate doesn't delay consumers).
- **Free after use:** the constant's column is reclaimed once its last consumer
  runs (cross-check with I4).
- **Shared constant:** a constant consumed at two different layers stays live
  across `[first, last]` consumer and is then freed.
- **No deadlock / pure-constant subgraph:** `Add(lit, lit)` and
  `Linear(Concatenate([lit, lit]))` materialize immediately and compile.
- **Chain interaction:** a constant feeding a `linear_relu_linear` (the `select`
  shape) schedules and frees correctly without breaking chain detection or
  `chain_protected`.

### 5.3 Residual-occupancy regression (`tests/compile/forward/`)
- The Phase-0 test, rewritten: compile `select(cond, x, lit)` deep in the
  network and assert the constant's residual column is **free** during the early
  layers. This is the canonical guard against the original bug returning.

### 5.4 Invariants (`tests/compile/forward/test_compiler_assertions.py`)
- I1–I4 must continue to pass on literal graphs (no aliasing, no truncated
  write, correct Q/K/V widths, no premature free). Run with
  `TW_COMPILER_VERIFY=1` for the end-of-layer liveness walk.
- **If Phase 1 or 3 introduces any new compiler assertion**, it must ship with a
  matching negative test here (the project rule: no new invariant without a
  negative test).

### 5.5 End-to-end oracle correctness (`tests/compile/forward/test_forward_compile.py`)
- `probe_compiled` (compiled vs recursive oracle) on a matrix of placements:
  constant feeding a `Linear`; feeding an `Attn` value path; a shared constant;
  a pure-constant subgraph; a deep `select`. Run under `optimize=0` (Phase 1)
  and `optimize ∈ {1,2,3}` (Phase 2). Require `first_divergent is None` within a
  justified `atol`.
- `compiled(inputs, debug=True)` self-consistency on the same graphs (the new
  allocate/free timing must not corrupt the residual stream).

### 5.6 CP-SAT-specific (Phase 2)
- Constants receive `layer_var`s and are placed late (just-in-time), not at
  layer 0.
- Schedule feasibility and `DirectedLayerScheduler` replay parity on literal
  graphs; warm-start path exercised; heuristic-fallback path exercised.
- Residual benefit assertion: reserved-column count / layer count improves vs
  the Phase-1 shim on a literal-heavy fixture.

### 5.7 Export (`tests/` covering `export.py`)
- ONNX/step export of a literal-containing graph still computes the constant
  correctly now that it is materialized mid-network rather than prefilled — the
  exported model's output matches the compiled model.

### 5.8 Whole-suite discipline
- All existing tests green under `make test` (Modal-sharded). Pay attention to
  full-suite-only flakes at tolerance boundaries (see *FP nondeterminism* in
  `CLAUDE.md`): the constant path is exact, so any such flake is unrelated and
  must not be "fixed" by touching this code.

---

## 6. Known existing tests to update (from the blast-radius audit)

- `tests/compile/forward/test_scheduler.py:904-954`
  (`test_add_into_shared_addend_not_reassigned`) — manually pre-allocates a
  shared `LiteralValue`; rework to scheduler-driven allocation.
- `tests/compile/forward/test_residual_map.py:93` — constructs the residual map
  with a constant in the input set.
- `tests/compile/forward/test_weight_writer.py` (the `compute_literal_value`
  tests, ~804-969) — currently use pre-allocated/zero-init columns; extend to
  the dirty-reuse case (§5.1).
- `tests/compile/forward/test_graph_analysis.py:119-122` — asserts every
  `is_input_node` has positive critical path; a `LiteralValue` no longer
  qualifies, update accordingly.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Deferral gate keeps a constant unmaterialized when its consumer is scheduled (consumer reads a stale/empty column). | Gate on the *consumer's non-constant inputs* (incl. same-layer), which guarantees the constant is written the layer before the consumer reads. I4 + `debug=True` self-consistency catch any miss. |
| Constant's column freed before its last consumer. | Normal dead-node liveness (`_is_dead` = effective consumers ⊆ computed); I4 end-of-layer liveness walk under `TW_COMPILER_VERIFY=1`. |
| `optimize>0` broken between Phase 1 and Phase 2. | Default: land Phases 1+2 together so main never sees the broken state (§3, D4). If an independent Phase-1 merge is required, the §3 source shim keeps the solver conservative — overflow-safe *by construction* (full-span reservation ⊇ any actual window), to be **confirmed** by the gate's `optimize>0` tests, with `compile.py:678` heuristic fallback as backstop. |
| CP-SAT birth dirty-cancel does not generalise to constants. | Phase 2 task 3 reads that seam explicitly; model it directly if needed before claiming "small delta". |
| A constant reaches `get_input_res_stream` unhandled → `assert False`. | Remove/guard the dead prefill branch (Phase 1 task 5); covered by §5.7. |
| Full-suite-only tolerance flake blamed on this change. | The path is exact (no PL approximation); investigate per CLAUDE.md, do not weaken tolerances. |

---

## 8. Out of scope / future

- **Re-materialization of widely-shared constants** (splitting one `LiteralValue`
  into per-consumer copies so a constant used at layers 5 and 50 doesn't hold a
  column across `[5,50]`). A graph rewrite; revisit only if measurement shows
  shared constants dominate residual pressure.
- **Folding** beyond the linear-input case (Phase 3 covers the linear case if
  the gate opens).

---

## 9. File-by-file change map (reference)

| File | Change | Phase |
|---|---|---|
| `compiler/forward/graph_analysis.py:204` | drop `LiteralValue` from `is_input_node` | 1 |
| `compiler/forward/compile.py:477,517-521,534,737-742` | constants excluded from input pre-alloc/seed/index (automatic once `is_input_node` changes; verify) | 1 |
| `compiler/forward/scheduler.py:787-814` | add just-in-time eligibility gate to literal branch | 1 |
| `compiler/forward/scheduler.py` (`__init__`) | build `LiteralValue → consumers` map | 1 |
| `compiler/forward/scheduler.py:1298-1306` | `DirectedLayerScheduler` constant pass-through | 1 |
| `compiler/transformer.py:91-93,111` | remove/guard dead prefill `LiteralValue` branch | 1 |
| `compiler/export.py:632,825` | guard dead in-state `LiteralValue` handlers | 1 |
| `debug/probe.py:370` | check constants as computed nodes | 1 |
| `compiler/forward/cpsat_scheduler.py` (`build_graph_model`) | Phase-1 source shim; Phase-2 reclassify to schedulable | 1, 2 |
| `compiler/forward/cpsat_scheduler.py:339-340,595-598` | drop constants from pinned/input_residual | 2 |
| `compiler/forward/cpsat_scheduler.py:846-874` | confirm/model birth dirty-cancel for constants | 2 |
| `compile.py` warm-start | hint constant layers | 2 |
| `graph/optimize.py` + compile wiring | folding pass (if gated in) | 3 |
| tests (see §5, §6) | new + updated coverage | all |
