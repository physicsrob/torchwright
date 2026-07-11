# Compile as a pure function — lower() returns a copy

**LANDED 2026-07-04** (L0 e610988, determinism sweep 807a754, L1+L2
94b32ed, L3 338ed91, wrapper-transparent lookups 00c1784; full suite
green; Phase C resumed at 049d9ca). Decision 2 settled as **(a)**
wrapper-cloning — the parity gate (`tests/compile/test_lowering_parity.py`)
measured bit-identical bounds against the old pipeline. Two additions
the plan did not anticipate, both caught by the suite: scheduling
predecessors are scheduling-only edges (a *sibling* may not be cloned
yet when its dependent is — remapped in a second pass), and node-keyed
lookup surfaces (`ResidualAssignment`, `flatten_concat_nodes`, the
extraction helpers) must resolve through the wrappers the source now
keeps — the old in-place strip used to rewire user Concatenate children
away from them. The end-of-compile translation is a single re-key of
the net's node-keyed artifacts (residual assignment, placements,
realization table) onto source nodes — "the net speaks source" — rather
than per-surface map threading. Copy overhead measured ~1.6 KB/node
retained on the 1-digit adder. Handed off to the next doom compile:
render-walkthrough re-verification, flagship schedule-cache-hit check,
and the flagship-scale memory measurement.

Execution plan for making compilation never mutate the source graph:
`lower()` builds a compiler-private copy of the graph, every downstream
pass consumes the copy, and the user's graph objects are untouched by
compilation. Written 2026-07-03, after the Phase C examples cutover
(`docs/swiglu_step2_plan.md`) surfaced the recompile bug described
below. **Phase C is parked until this plan lands.** Revised 2026-07-03
(same day, post-review): the fusion premise corrected (fusion is not in
the compile pipeline), decision 2's mechanism reopened with a
wrapper-cloning variant, the scheduling-determinism sweep added as L2
work, and the L4 cache expectation corrected from miss to hit.

## The bug that motivated this

Compiling the same graph object twice silently loosens every value
bound on the second compile, on both machines. Mechanism, one sentence:
the first compile's `GraphAnalyzer._strip_asserts`
(`compiler/forward/graph_analysis.py:49`) removes — by in-place
rewiring — the Assert wrapper nodes that carry the op library's tight
declared ranges, parking each claim in the wrapped node's
`_structural_type`; the second compile's `lower()` refresh loop
(`compiler/lower.py:299`) recomputes `_structural_type` from graph
structure, which is exactly the write that clobbers the parked claim,
and with the wrappers gone the tightening cannot be re-derived.

Measured on the 1-digit adder (double `compile_to_onnx`, no schedule
cache involved):

| machine | worst bound, compile 1 | worst bound, compile 2 |
|---|---|---|
| swish | ±4,830 | ±7.8e9 — rms_norm energy cert **fails** (9.7e20 vs budget 1.8e19) |
| relu | ±4,889 | ±3.8e7 — cert passes, loosening **silent** |

The relu row is why this was never seen before Phase C: the bug is
machine-independent, but only the swish machine's raw envelopes cross a
threshold that raises. The failing test is
`tests/compile/forward/test_occupancy_stable_across_cache_hit`, which
compiles the same graph object twice deliberately.

The deeper wart the bug exposes: compilation mutates its input. The
strip rewires the user's nodes, and `lower()`'s refresh loop rewrites
their cached bounds in place; the `refresh_node_caches` machinery
exists to chase the staleness such mutations create. (This paragraph
originally also blamed the fusion pass at `optimize>=1` — wrong:
`optimize` selects the CP-SAT time budget, `compile.py:868-876`, and
`fuse_consecutive_linears` has no caller in the compile pipeline at
all; only two test files invoke it manually, pre-compile, on their own
graphs.) `lower()`'s own docstring (`compiler/lower.py:25`)
says its purpose is to make the stale-bounds bug class "structurally
impossible at this boundary instead of being guarded by per-pass
discipline" — while the Assert-strip sits outside the boundary, guarded
by precisely such discipline (the ordering contract at
`compiler/lower.py:29-33`).

## The invariant being installed

**Compilation is a pure function of the source graph.** `lower()`
returns a `LoweredGraph` holding a compiler-private *copy*; the
analyzer, fusion pass, scheduler, weight writer, and certifications all
operate on the copy; no compile pass writes to a source node, ever.
Recompiling is re-lowering the same pristine source — identical output
by construction, nothing to converge, no fixed-point argument needed.

## Settled decisions

1. **`lower()` copies; nothing else changes the source.** The copy is
   built in one deterministic canonical-order walk. Rejected
   alternatives, recorded so they aren't relitigated:
   - *In-place strip moved into `lower()` + persistent claims*: fixes
     recompile convergence but keeps one-time mutation of the user's
     graph. (The rejection originally also cited in-pipeline fusion
     mutating user nodes at `optimize>=1`; that premise was wrong —
     fusion has no pipeline caller — but the rejection stands on the
     mutation half alone.)
   - *Assert-transparency in every pass* (the `Concatenate` treatment):
     no mutation, but smears wrapper special-casing across scheduler,
     liveness, and weight writer permanently.
   - *Asserts become node annotations instead of wrapper nodes*: clean
     but a wide user-facing refactor — and unnecessary once the copy
     exists (see decision 2).

2. **The source keeps its wrappers forever** — that much is settled:
   `collect_asserts` works after compile, and the "collect asserts
   *before* compiling" caveats (`graph/asserts.py:54`,
   `debug/probe.py:926`) are deleted, not worked around.

   **REOPENED (2026-07-03, post-review): how claims reach the copy.**
   The originally settled mechanism (wrappers not copied; each claim
   applied to the copied node's bounds at copy time; persisted in a
   `_claimed_type` field re-applied by `refresh_node_caches`, the
   `_semantic_affine_override` pattern) rested on two premises the
   review falsified:

   - *The persistence field has no reader.* Its stated reason —
     "fusion folds refresh copied nodes and must not clobber the
     claim" — describes a pass that is not in the pipeline. Nothing
     refreshes a copy after claims are applied, so `_claimed_type`
     would be written once and never read. Don't build it.
   - *Copy-time claim application cannot reproduce today's bounds
     exactly.* A claim reaches bounds through two channels in the
     Assert affine rule (`affine_rules.py:439-491`): for leaf targets
     (InputNode/Embedding, `:452`) it narrows the leaf's per-input
     range — transferable and reproducible. For general targets with
     a finite claim it *replaces* the wrapper's affine bound with a
     constant box (coefficients zeroed, endpoints claim∩propagated),
     and that box lives only in the wrapper and in the cached bounds
     of consumers built on the wrapper. Post-strip, three bound
     states coexist for such an assert — wrapper-consumers:
     box-derived; direct consumers of the target: claim-free; the
     target itself: structural tightening only. A wrapper-free copy
     applies the claim once, so one of the three moves relative to
     today whichever way it's applied (a box also destroys
     correlation structure: `x − x` through an affine bound is 0,
     through a box `[−2M, 2M]`). See *Numerical implications*.

   Two candidate mechanisms; decide at L1 start:

   - **(a) Copy the wrappers too; run the existing strip on the
     copy** (review's recommendation). Clone Assert/DebugWatch
     (predicate closures shared by reference, like weight payloads);
     the existing, tested strip runs unchanged on compiler-private
     nodes — relocated into `lower()` post-copy so the analyzer stays
     pure analysis (decision 6). Bounds bit-identical to today *by
     construction*; the claim-application and persistence machinery
     is never built. Cost: two trivial clone paths.
   - **(b) Original mechanism minus persistence**: wrappers not
     copied, claims applied at copy time. Requires solving the
     two-channel reproduction problem and passing the L1 old-vs-new
     parity gate before proceeding.

   Either way the settled outer decision is unchanged: the user's
   graph keeps its wrappers; compilation never strips the source.

3. **Weight tensors are shared by reference.** Copies duplicate the
   thin node objects, never the `LiteralValue` payloads, `Linear`
   matrices, or embedding tables. Nothing in the compiler writes to a
   node's weight tensor (weight *writing* assembles new transformer
   matrices); sharing is safe and keeps the copy O(nodes), not
   O(parameters). Verified: the weight writer's in-place writes all
   target the runtime matrices under assembly — 3-D per-head stacks
   (`weight_writer.py:261-272`) and `(d, d_hidden)` MLP projections
   (`:718-724`, `:749-751`, `:769-774`), shapes no graph node has —
   and the fusion fold *reassigns* `output_matrix`/`output_bias` to
   fresh matmul results (`graph/optimize.py:31-35`), never writing
   into the originals.

4. **Copy order is the canonical walk; scheduling determinism is a
   separate, required sweep.** `graph_identity.py` walks are already
   Assert/DebugWatch-transparent and creation-order-independent, so the
   copy's canonical ids equal the source's — the source↔copy map can be
   built directly on them.

   The stability half was under-modeled (review, 2026-07-03). The
   occupancy test's "ordered by node id()" is not one tie-break site:
   `Node.__eq__`/`__hash__` key on `node_id` (`graph/node.py:276-283`),
   so every *set* of nodes iterates in hash-table order driven by the
   absolute id values, and `_critical_path_key` has **no tie-break**
   (`scheduler.py:985`), so every stable sort over it inherits that set
   order. Same-object recompiles are deterministic today only because
   the ids are identical both times. The copy *breaks* that accident —
   every compile mints clones with offset ids — so the copy alone makes
   recompiles less stable, not more. The fix is the determinism sweep
   (L2): clones are constructed inputs-first in canonical-walk-derived
   order, so *relative* clone-id order is process-independent — sort
   node collections by `node_id` at the order-sensitive sites and
   tie-break the `_critical_path_key` sorts on `node_id`, and schedules
   become identical across recompiles and across processes. Do NOT
   normalize clone ids to 0..n-1: equality keys on `node_id`, so a copy
   sharing an id with a source node would compare equal to it and
   corrupt any set/dict holding both; clone ids keep coming from the
   global counter.

   Corrected one-time cost: the flagship schedule-cache entry **hits**
   (the fingerprint hashes canonical topology + geometry + knobs,
   `graph_identity.py:93-130`, all preserved by the copy); artifact
   bytes may still shift once because the per-layer walk — which runs
   on every compile, cache hit or not (the cache stores only the CP-SAT
   assignment, `compile.py:907-940`) — changes order under the sweep.
   Column-pinned test expectations may need one-time updates. An actual
   cache miss at L4 is a bug to investigate, not the expected effect.

5. **Every user-facing surface keyed by node objects translates through
   the source↔copy map** carried on `LoweredGraph`: `debug_value`,
   `probe_compiled` / `probe_residual` / `probe_attention` /
   `probe_layer_diff`, the debug sidecar writer, `OnnxDebugSession`.
   (Input binding needs nothing: it is name-keyed — `CompiledHeadless`
   slices the packed input tensor by `(name, start_col, width)` specs,
   `export.py:1886-1907`.) `debug=True`
   assert re-checking evaluates the *source* graph's predicates against
   the copy's compiled values via the map — simpler than today's
   stripped-assert bookkeeping, which dies.

6. **`GraphAnalyzer` becomes pure analysis.** `_strip_asserts` and its
   claim-transfer code are deleted; the analyzer receives a
   wrapper-free copy and may assert it never sees a wrapper. The
   ordering contract docstring (`lower.py:29-33`) is deleted with it.

7. **The copy pass replaces the refresh loop.** Each copy's
   `_structural_type` / `_affine_bound` are computed once, in
   topological order, with claims applied — this *is* the "fresh
   bounds" guarantee `lower()`'s refresh loop provides today
   (`lower.py:299-301`), so that loop is subsumed, not duplicated.
   `refresh_node_caches` itself stays for its one remaining caller, the
   standalone fusion pass — which users run on *source* graphs, where
   it is claim-safe without any persistence mechanism: the source keeps
   its wrappers, so the refresh re-derives claims through the Assert
   affine rule.

## Ground truth (where things live today)

- Strip + claim transfer: `compiler/forward/graph_analysis.py:34, 49-100`.
- `forward_compile` pipeline order: `lower()` at
  `compiler/forward/compile.py:711`, `GraphAnalyzer` at `:715`.
- Refresh loop: `compiler/lower.py:299-301`; ordering contract
  docstring `:25-33` (cites the commit 0570af1 stale-bounds bug).
- Fusion-pass refresh after in-place folds: `graph/optimize.py:264-270`
  — no compile-pipeline caller; invoked only by two test files,
  pre-compile, on their own graphs (`tests/compile/forward/
  test_cpsat_knobs.py:36`, `tests/graph/test_optimize.py:8`).
- `value_type` = intersection of `_affine_bound` scalar range with
  `_structural_type`: `graph/node.py:196`.
- The persistence precedent: `_semantic_affine_override` stored at
  `graph/node.py:186`, re-applied by `refresh_node_caches` at
  `graph/affine_rules.py:578-592`.
- Canonical walks (wrapper-transparent, creation-order-independent):
  `compiler/graph_identity.py` (`unwrap_debug` at `:32`).
- RMS-norm energy certification (the bound consumer that fired):
  `compiler/forward/compile.py:227`, called at `:1435`.
- `CompiledHeadless`: `compiler/export.py:1861`.
- Current strip behavior pinned by `tests/graph/test_claimed_type_strip.py`
  (seven tests: two `tightened_with` units, four `InputNode` leaf strip
  cases, one degenerate `LiteralValue` general target whose
  `assert_integer` claim equals its own range — no effectual box case).
- Node equality/hash key on `node_id`: `graph/node.py:276-283`.
- Realization table is wrapper-indifferent: `is_schedulable` is a
  positive allowlist (`Attn, FFN, LiteralValue, Add, Linear`),
  `compiler/realization.py:87-89` — wrappers never get entries, so the
  table is identical built pre-strip, post-strip, or from a copy.
- Schedule cache stores canonical-id-keyed *assignments*, not node
  references: `graph_identity.py:1-18`, `compile.py:907-940` (a hit
  skips the solver; the per-layer walk always reruns).
- Repro scripts: parked untracked at `scripts/repro_double_compile.py`
  and `scripts/cert_diff.py` (copied 2026-07-03 from the planning
  session's volatile scratchpad, strictly as a stopgap) — they become
  the committed D6 tests in L1/L2 and are deleted then.

## Phases

### L0 — clone infrastructure + verification checklist

A `clone` path for every type in `lower.VOCABULARY` — generate the
dispatch from that tuple, not a hand list, and make an unhandled type a
loud error. Two members are easy to forget: `Placeholder`, and
`ValueLogger` — which is *not* stripped by the analyzer and *not*
transparent to the canonical walk (`graph_identity.py:32-36`), so it
cannot be silently dropped from the copy without breaking decision 4's
id-equality premise; clone it or reject it at lower(), decide
explicitly. Under decision 2(a), `Assert`/`DebugWatch` get clone paths
too (predicate closures shared by reference). Each clone: new node with
rewired inputs, every semantic field carried, weight tensors shared —
and `scheduling_predecessors` **remapped through the source→copy map**,
not copied verbatim (it holds node references outside `inputs`,
`graph/node.py:178-182`, honored by the scheduler's readiness check; a
verbatim copy points at source nodes and silently drops the hints, and
`topology_entries` cannot see the loss). Decide the mechanism here —
re-running `__init__` (simple, recomputes eager caches at clone time,
which decision 7 wants anyway) vs. structural clone with explicit cache
computation; bias toward whichever makes "a field the clone forgot" a
loud failure rather than a silent one.

Completeness guard — not "public attributes", which misses exactly the
dangerous fields: clone representative *variant* instances (Attn with
and without partial rotary; FFN with and without `up_proj`) of every
vocabulary type and compare (i) `topology_entries`, (ii) all semantic
fields including the underscored ones (`_semantic_affine_override`,
`_structural_type`), with node-reference-valued fields compared through
the map, and (iii) *behavior*: `value_type`, the affine scalar range,
and `compute()` output on random inputs, source vs. copy.

Unit tests (D6, smallest layer):

- Copy of a mixed graph is structurally equal (`topology_entries`
  identical — the walk is wrapper-transparent under either decision-2
  mechanism), with the source's asserts intact.
- Weight tensors are the same objects (`is`), not equal copies.
- Copied bounds equal the source's assert-tightened bounds.
- Source node set, inputs, and `value_type`s are bit-identical before
  and after copying.

Verification checklist (updated 2026-07-03 post-review):

- [x] Input binding is name-keyed (`(name, start_col, width)` specs,
      `export.py:1886-1907`) — nothing to translate.
- [x] The "tie-break site" is not one site: `_critical_path_key` has
      no tie-break at all; ordering comes from node-*set* iteration
      keyed on absolute `node_id` hashes. Converted from a checklist
      item into the L2 determinism sweep (decision 4).
- [x] No pass caches node references across compile calls: the
      schedule cache stores canonical-id-keyed assignments
      (`graph_identity.py:1-18`, `compile.py:907-940`).
- [ ] Decide `ValueLogger`'s copy treatment (clone vs. reject at
      lower()).

### L1 — lower() returns the copy

Opens by settling reopened decision 2 — (a) wrapper-cloning vs. (b)
copy-time claim application. `LoweredGraph` gains the copied
`output_node`, the source output reference, and the source↔copy node
map (canonical-id based). Under (a) the existing strip relocates into
`lower()` and runs on the copy; under (b) claims are applied at copy
time (no persistence field — nothing refreshes copies) and the
two-channel reproduction problem must be solved first. Either way
`GraphAnalyzer` stops stripping (decision 6).
`tests/graph/test_claimed_type_strip.py` is rewritten against the
chosen semantics, renamed accordingly, and gains the case it has
always lacked: a general target whose finite claim is strictly tighter
than its propagated bound, with a consumer built on the wrapper.

D6 tests: lower the same source twice → per-node `value_type`
bit-identical across the two copies; source untouched (node set,
asserts, bounds). Gate regardless of the 2(a)/2(b) choice: a one-time
old-vs-new parity diff on first-compile bounds (per-node `value_type`
and sidecar bounds, on the 1-digit adder and one swish graph) —
trivially clean under (a), load-bearing under (b). No other planned
test compares new against old.

### L2 — the pipeline consumes the copy

Opens with the **scheduling-determinism sweep** (decision 4) — without
it this phase's own tests cannot pass reliably: `node_id` tie-breaks on
the `_critical_path_key` sorts (`scheduler.py:329, 617, 661, 717`) and
`node_id`-sorted iteration at the order-sensitive set sites
(`_build_topo_order` seeding and consumer drains,
`graph_analysis.py:157-173`; the per-layer ready filter,
`scheduler.py:1203`). These sites run on every compile, cache hit or
miss — the cache stores only the CP-SAT assignment. The sweep is
independent of the copy and may be pulled earlier as a standalone
commit with its own D6 test (two fresh rebuilds in one process →
identical occupancy — exactly the strengthening below).
`weight_writer.py` was not exhaustively audited for further
order-sensitive iteration; the identity tests below are the
completeness gate for any missed site.

Then `forward_compile` threads `lowered.output_node` (the copy) into
`GraphAnalyzer`, scheduling, weight writing, and the rms_norm
certification (fusion is not in the pipeline — nothing to thread). The
CP-SAT path (`cpsat_scheduler.py:289` builds its own analyzer) follows.
Schedule-cache keys and canonical ids verified stable across recompiles
by test.

D6 tests: the committed form of the parked repro scripts — double
`compile_to_onnx` on the 1-digit adder (both machines) → second compile
succeeds and both sidecars are identical. The occupancy test drops its
same-object workaround and *strengthens*: two fresh rebuilds now must
produce identical occupancy too. (Cross-process stability, if enforced
rather than assumed, needs a subprocess test — in-process double
compiles cannot observe it.)

### L3 — debug surfaces translate

`debug_value`, the four probes, the sidecar writer, and
`OnnxDebugSession` accept source nodes and translate via the map.
`debug=True` re-checks predicates from the intact source graph against
copy values. The collect-before-compile caveats
(`graph/asserts.py:54`, `debug/probe.py:926`) and the stripped-assert
bookkeeping are deleted. `CLAUDE.md`'s *Debugging compiled graphs*
wording ("stripped at compile time") and `graph_identity.py`'s module
docstring (lines 11-18, which explains wrapper transparency by
reference to the in-place strip) update to match the chosen decision-2
mechanism — "stripped from the compiler-private copy" under (a), "not
copied into the lowered graph" under (b).

### L4 — suite, flagship, and handoffs

Full suite on Modal. Expected one-time effects, called out so they are
not misread as regressions: a flagship schedule-cache **hit** with
artifact byte-diffs from the determinism sweep's ordering (decision 4;
an actual cache miss here is a bug to investigate, not the expected
effect); doom-side render
walkthrough re-verification happens at the next doom compile (handoff
note, not blocking here). Then **Phase C resumes**: its parked state
(examples flipped to swiglu, three mechanical fixture-expectation fixes
in `test_module.py` / `test_onnx_debug_session.py`, `calculator_simple` and
`binary_increment` staying relu for the HF path) is recorded in
`docs/swiglu_step2_plan.md`.

## Numerical implications

Decision-2-dependent. Under 2(a) (wrapper-cloning): none, by
construction — bounds derive through the same wrappers, refresh, and
strip as today. Under 2(b) (copy-time claim application): first-compile
parity is NOT established — a general-target finite claim reaches
downstream affine bounds only through the wrapper's constant-box bound,
and a wrapper-free copy applying the claim once cannot reproduce
today's consumer-dependent split (reopened decision 2); the L1
old-vs-new parity diff is the gate, and without it a shift would
surface only as an L4 cert surprise pre-authorized as an "expected
one-time effect". Either way: no op math changes; no noise
re-measurement needed (D7 does not trigger); *second* compiles now
match first compiles — the silent relu recompile loosening in the table
above is fixed as a side effect.

## What dies

- Under decision 2(b): `GraphAnalyzer._strip_asserts` and the
  claim-transfer code. Under 2(a) they survive, relocated to run on the
  compiler-private copy inside `lower()` — what dies either way is
  their mutation of *user* nodes.
- The `_claimed_type` persistence field — never built (its only stated
  reader, in-pipeline fusion, does not exist).
- The `lower()` ordering-contract docstring and the refresh loop in
  `lower()` (subsumed by the copy pass).
- The occupancy test's same-object workaround and its comment.
- The collect-asserts-before-compile caveats in `graph/asserts.py` and
  `debug/probe.py`.
- The "`optimize>=1` fusion on copies" risk item (it tested a
  non-interaction).
- The latent both-machines recompile loosening.

## Risks / parked

- **Clone completeness** is the main correctness risk: a semantic field
  the clone forgets (an `Attn` rope config, an FFN packing field) is a
  silent miscompile. The L0 completeness guard is the mitigation;
  invest there.
- **Memory**: copies double the node-object and bound-cache footprint
  transiently during compile (weights shared, so no parameter
  duplication). The exporters' streaming memory bound is untouched —
  the copy exists before weight streaming begins. Measure on the
  flagship compile at L4 before declaring done.
- **Fusion is not in the compile pipeline** (`optimize` selects the
  CP-SAT budget; `fuse_consecutive_linears` has no pipeline caller).
  The standalone pass remains available to users on *source* graphs,
  where it stays claim-safe because the source keeps its wrappers and
  the refresh re-derives claims through the Assert affine rule. If
  fusion is ever meant to join the pipeline (running on copies), that
  is a new decision with its own plan — and the moment anything
  refreshes copies post-claim, a persistence mechanism becomes
  necessary again.
- Parked: whether `LoweredGraph` should be reusable across multiple
  `compile_*` calls as a public workflow ("lower once, compile many").
  The machinery makes it possible; deciding to *expose* it is separate.
