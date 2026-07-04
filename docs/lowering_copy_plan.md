# Compile as a pure function — lower() returns a copy

Execution plan for making compilation never mutate the source graph:
`lower()` builds a compiler-private copy of the graph, every downstream
pass consumes the copy, and the user's graph objects are untouched by
compilation. Written 2026-07-03, after the Phase C examples cutover
(`docs/swiglu_step2_plan.md`) surfaced the recompile bug described
below. **Phase C is parked until this plan lands.**

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
strip rewires the user's nodes; the fusion pass (`graph/optimize.py:264`)
folds the user's nodes in place at `optimize>=1`; and the whole
`refresh_node_caches` machinery exists to chase the staleness those
mutations create. `lower()`'s own docstring (`compiler/lower.py:25`)
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
     graph and leaves the fusion pass mutating user nodes at
     `optimize>=1`.
   - *Assert-transparency in every pass* (the `Concatenate` treatment):
     no mutation, but smears wrapper special-casing across scheduler,
     liveness, and weight writer permanently.
   - *Asserts become node annotations instead of wrapper nodes*: clean
     but a wide user-facing refactor — and unnecessary once the copy
     exists (see decision 2).

2. **Assert/DebugWatch wrappers are not copied.** They stay intact in
   the source graph forever: `collect_asserts` works after compile, and
   the "collect asserts *before* compiling" caveats
   (`graph/asserts.py:54`, `debug/probe.py:926`) are deleted, not
   worked around. Each Assert's claimed range is applied to the copied
   node's bounds as the copy is built — and *persisted* on the copy (a
   `_claimed_type` field re-applied by `refresh_node_caches`, exactly
   the `_semantic_affine_override` pattern at `graph/node.py:186` /
   `graph/affine_rules.py:578`), because fusion folds refresh copied
   nodes and must not clobber the claim. The persistence mechanism from
   the rejected small fix survives here, applied to compiler-private
   nodes only.

3. **Weight tensors are shared by reference.** Copies duplicate the
   thin node objects, never the `LiteralValue` payloads, `Linear`
   matrices, or embedding tables. Nothing in the compiler writes to a
   weight tensor (weight *writing* assembles new transformer matrices);
   sharing is safe and keeps the copy O(nodes), not O(parameters).

4. **Copy order is the canonical walk, and creation-order tie-breaks
   become rebuild-stable.** `graph_identity.py` walks are already
   Assert/DebugWatch-transparent and creation-order-independent, so the
   copy's canonical ids equal the source's — the source↔copy map can be
   built directly on them. Scheduling tie-breaks that today depend on
   node creation order (the occupancy test's comment: "head-column
   assignment is ordered by node id(), so two FRESH rebuilds would
   permute heads") get deterministic relative order from the copy pass,
   making head assignment identical across recompiles *and* across
   process-separated rebuilds — strictly better than today. Accepted
   one-time cost: head/layout order may shift once relative to current
   artifacts (one flagship schedule-cache miss, possible column-pinned
   test expectation updates).

5. **Every user-facing surface keyed by node objects translates through
   the source↔copy map** carried on `LoweredGraph`: `debug_value`,
   `probe_compiled` / `probe_residual` / `probe_attention` /
   `probe_layer_diff`, the debug sidecar writer, `OnnxDebugSession`,
   and input binding if it is node-keyed (L0 verifies). `debug=True`
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
   `refresh_node_caches` itself stays, for the fusion pass's use on
   copies.

## Ground truth (where things live today)

- Strip + claim transfer: `compiler/forward/graph_analysis.py:34, 49-100`.
- `forward_compile` pipeline order: `lower()` at
  `compiler/forward/compile.py:711`, `GraphAnalyzer` at `:715`.
- Refresh loop: `compiler/lower.py:299-301`; ordering contract
  docstring `:25-33` (cites the commit 0570af1 stale-bounds bug).
- Fusion-pass refresh after in-place folds: `graph/optimize.py:264-270`.
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
- Current strip behavior pinned by `tests/graph/test_claimed_type_strip.py`.
- Repro scripts: scratchpad-only (`repro_double_compile.py`,
  `cert_diff.py` in the session scratchpad) — they become the committed
  D6 tests in L1/L2 and must not be lost before that.

## Phases

### L0 — clone infrastructure + verification checklist

A `clone` path for every node type in the compilable vocabulary
(`Linear`, `Add`, `Concatenate`, `Attn`, `FFN`, `LiteralValue`,
`InputNode`, `Embedding`, …): new node with rewired inputs, every
semantic field carried, weight tensors shared. Decide the mechanism
here — re-running `__init__` (simple, recomputes eager caches at clone
time, which decision 7 wants anyway) vs. structural clone with explicit
cache computation; bias toward whichever makes "a field the clone
forgot" a loud failure rather than a silent one. Add a completeness
guard: a test that clones one instance of every vocabulary type and
diffs `topology_entries` plus every public attribute.

Unit tests (D6, smallest layer):

- Copy of a mixed graph is structurally equal (`topology_entries`
  identical) with asserts absent from the copy, present in the source.
- Weight tensors are the same objects (`is`), not equal copies.
- Copied bounds equal the source's assert-tightened bounds.
- Source node set, inputs, and `value_type`s are bit-identical before
  and after copying.

Verification checklist (facts the plan needs but this session did not
pin down — resolve before L2):

- [ ] How `CompiledHeadless.__call__` binds input values (node-keyed →
      needs the map; name-keyed → nothing to do).
- [ ] The exact site of the creation-order head-column tie-break the
      occupancy test's comment refers to (likely a
      `_critical_path_key` tie-break in `compiler/forward/scheduler.py`).
- [ ] Whether any pass caches node references across compile calls
      (schedule cache stores schedules, not nodes — confirm).

### L1 — lower() returns the copy

`LoweredGraph` gains the copied `output_node`, the source output
reference, and the source↔copy node map (canonical-id based).
Claims applied at copy time and persisted per decision 2.
`GraphAnalyzer` stops stripping (decision 6);
`tests/graph/test_claimed_type_strip.py` is rewritten against
`lower()`'s copy semantics and renamed accordingly.

D6 tests: lower the same source twice → per-node `value_type`
bit-identical across the two copies; source untouched (node set,
asserts, bounds).

### L2 — the pipeline consumes the copy

`forward_compile` threads `lowered.output_node` (the copy) into
`GraphAnalyzer`, fusion, scheduling, weight writing, and the rms_norm
certification. The CP-SAT path (`cpsat_scheduler.py:289` builds its own
analyzer) follows. Schedule-cache keys and canonical ids verified
stable across recompiles by test.

D6 tests: the committed form of the scratchpad repro — double
`compile_to_onnx` on the 1-digit adder (both machines) → second compile
succeeds and both sidecars are identical. The occupancy test drops its
same-object workaround and *strengthens*: two fresh rebuilds now must
produce identical occupancy too.

### L3 — debug surfaces translate

`debug_value`, the four probes, the sidecar writer, and
`OnnxDebugSession` accept source nodes and translate via the map.
`debug=True` re-checks predicates from the intact source graph against
copy values. The collect-before-compile caveats
(`graph/asserts.py:54`, `debug/probe.py:926`) and the stripped-assert
bookkeeping are deleted. `CLAUDE.md`'s *Debugging compiled graphs*
wording ("stripped at compile time") updates to "not copied into the
lowered graph".

### L4 — suite, flagship, and handoffs

Full suite on Modal. Expected one-time effects, called out so they are
not misread as regressions: a flagship schedule-cache miss (decision 4)
and possible artifact byte-diffs from tie-break order; doom-side render
walkthrough re-verification happens at the next doom compile (handoff
note, not blocking here). Then **Phase C resumes**: its parked state
(examples flipped to swiglu, three mechanical fixture-expectation fixes
in `test_module.py` / `test_onnx_debug_session.py`, `calculator_v2` and
`binary_increment` staying relu for the HF path) is recorded in
`docs/swiglu_step2_plan.md`.

## Numerical implications

None. First-compile bounds are computed exactly as today (with asserts
visible); no op math changes; no noise re-measurement needed (D7 does
not trigger). What changes is that *second* compiles now match first
compiles — the silent relu recompile loosening in the table above is
fixed as a side effect.

## What dies

- `GraphAnalyzer._strip_asserts` and the claim-transfer code.
- The `lower()` ordering-contract docstring and the refresh loop in
  `lower()` (subsumed by the copy pass).
- The occupancy test's same-object workaround and its comment.
- The collect-asserts-before-compile caveats in `graph/asserts.py` and
  `debug/probe.py`.
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
- **`optimize>=1` fusion on copies** should be behavior-identical, but
  fusion folds were written against user graphs; L2 runs the
  `optimize=2` suite paths explicitly.
- Parked: whether `LoweredGraph` should be reusable across multiple
  `compile_*` calls as a public workflow ("lower once, compile many").
  The machinery makes it possible; deciding to *expose* it is separate.
