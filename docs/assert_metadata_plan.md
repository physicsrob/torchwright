# Asserts as node metadata — migration design

Status: DESIGN (no implementation yet).
Motivation: the univariate-collapse work (2026-07-05) exposed that
Assert claims have no durable home — see *Why*, below.  Study basis:
`_assert_rule` (affine_rules.py), `_canonical_walk` /
`topology_entries` (graph_identity.py), and a full census of
Assert/DebugWatch touch points (this doc's *Census*).

## The idea in one paragraph

An Assert is a *fact about a node's value* (a runtime predicate, plus
optionally a claimed range).  Today that fact is represented as a
**wrapper node interposed in the dataflow**, which forces every
topology-walking mechanism to know about wrappers: the lowering strip
and its one-shot claim transfer, unwrap helpers in seven modules,
wrapper-transparent canonical ids, wrapper special cases in the
node_map, clone dispatch, and the vocabulary.  The wrapper also
*blocks* linear folds by accident (the fold's consumer pattern doesn't
match through an interposed node), and its claim evaporates if any
pass recomputes bounds after the strip — the trap that blocked the
collapse plan's post-strip fusion round.  This migration moves the
fact onto the node itself: `Node` gains a `checks` list (predicate,
message, kind) and a persistent `claimed_type` (plus the
`integer_claim` flag), applied to bounds at the two existing cache
choke points.  Wrappers stop existing; a dozen mechanisms delete; the
claim-loss bug class becomes unrepresentable.

## Why: the evidence

- **Claims are the only bound-affecting information without a durable
  home.**  Semantic overrides got persistence long ago —
  `node._semantic_affine_override` exists precisely so
  `refresh_node_caches` can re-apply it ("a recompute that dropped it
  would silently loosen every downstream bound", node.py).  Assert
  claims never got the same treatment: the strip transfers them into
  the caches once, then deletes the wrappers that were their home.
  Any post-strip refresh silently widens claim-tightened bounds, and
  `test_lowering_parity` cannot see it (both pipelines lose the claims
  identically).  This is what made the collapse plan's "unconditional
  post-strip fusion round" unsafe as designed.
- **Wrapping buys nothing semantic.**  A wrapper asserts the wrapped
  node's value; every consumer sees the same value whether it reads
  the wrapper or the raw node.  There is no per-edge assert semantics
  to preserve.  Attachment-without-mutation and
  predicates-run-during-compute both survive the move to metadata.
- **Fold-blocking is an accident, not a policy.**  `optimize.py`
  contains no wrapper `isinstance` at all — folds skip wrapped values
  only because the pattern `producer's sole consumer is the fold
  target` doesn't match through an interposed node.  A real decision
  (asserted values stay materialized so `debug=True` can check them)
  is hiding inside a pattern-match coincidence.

## The design

**New state on `Node`** (all default-empty; carried onto clones by the
existing `__dict__`-by-reference copy):

- `checks: list[Check]` — `Check(predicate, message, kind)` where
  `kind` is `"assert"` (raises) or `"watch"` (prints), plus the
  `annotation` captured at attach time.  Predicates are the same
  closures wrappers hold today, shared by reference.
- `claimed_type: Optional[NodeValueType]` — the running intersection
  of all attached claims (claims commute; intersection order is
  irrelevant).
- `integer_claim: bool` — moves from the Assert wrapper (where the
  collapse work just put it) onto the node.

**Claim application at the two existing choke points** — nowhere else:

- `Node.__init__` and `refresh_node_caches` end by intersecting
  `claimed_type` into `_structural_type` (the `tightened_with` the
  strip uses today) and applying the affine channel (below).  Claims
  are refresh-proof *by construction*: there is no code path that
  recomputes bounds without re-applying them.

**The two `_assert_rule` channels, relocated to the node:**

- *General channel*: a finite claim degenerates the node's affine
  bound to the claim-intersected constant box — the same math
  `_assert_rule` applies at the wrapper today, one node earlier
  (identical, since the wrapper is a pass-through directly above the
  node).
- *Leaf channel*: a claim on an `InputNode`/`Embedding` tightens that
  leaf's own `input_ranges` entry at the leaf's rule, so every
  downstream bound inherits it through normal topological
  recomputation — replacing the wrapper-position-scoped tightening the
  strip does today.

**The attach API keeps its shape.**  `assert_in_range(x, lo, hi)` and
friends append a check, intersect the claim, refresh `x`'s caches, and
return `x`.  Every existing call site (`x = assert_integer(x)`)
compiles unchanged.

**Fold policy becomes explicit.**  `fuse_consecutive_linears` gains
one written rule: a fold that would absorb a node carrying checks is
declined.  This reproduces today's accidental blocking exactly — and
turns "which depth wins are worth deleting runtime checkability for?"
into a revisitable, measurable policy instead of an emergent property.

**Multi-input checks attach to their composite node.**  Five helpers
(`assert_strictly_less`, `assert_distinct_across`,
`assert_score_gap_at_least`, `assert_picked_from`,
`assert_softmax_hardness`) need a predicate that sees several nodes'
values at once, so today they build a synthetic
`Concatenate([...])`, wrap *that*, and project the caller's value back
out through a Linear.  Under metadata the check attaches to the
Concatenate; the composite structure (Concatenate + projecting Linear)
survives unchanged, and the explicit fold-decline rule above does
double duty — it is exactly what keeps fusion from absorbing the
checked Concatenate, preserving today's compiled shape.  This is the
subtlest constructor migration; it gets its own tests.

**ValueLogger stays a node.**  The canonical walk does not step
through it (only Assert/DebugWatch), so migrating it would change
fingerprints for every graph that uses one; it also has genuinely
positional semantics (print during compute).  The census's
counter-argument — that keeping it fragments the wrapper-transparency
idiom — mostly dissolves after the migration: every unwrap helper is
deleted, and ValueLogger remains as one ordinary vocabulary node with
one affine pass-through rule.  Out of scope; revisit separately if it
ever bothers anyone.

## What changes semantically (stated upfront)

1. **Bounds get equal-or-tighter, never wider.**  Where every consumer
   read the wrapper (the near-universal case), bounds are bit-identical.
   A consumer that read the *raw* node past a wrapper today missed the
   claim; under metadata it sees it.  Sound — the claim is a
   runtime-checked fact about the shared value.  Fingerprints don't
   hash bounds, so no cache effect.
2. **Attach-before-consumers becomes a convention, not a dataflow
   fact.**  Ops bake gating offsets from input ranges at graph-build
   time; today the wrapper forces claim visibility on whoever consumes
   the returned handle.  Under metadata, attaching a claim after
   consumers were built leaves those consumers' *build-time* reads
   un-tightened (compile-time bounds are always right — `lower()`
   recomputes fresh in topological order).  Existing call sites all
   attach immediately after construction; the helpers' docstrings state
   the convention.
3. **The `Assert.compute_value_type` quirk disappears.**  Today a
   claimed Assert reports an *unbounded* structural type (the claim
   reaches downstream only through the affine channel until the strip
   lands it structurally).  Under metadata the claim is on the node
   from the start — another equal-or-tighter case.

## Fingerprint and artifact stability (verified)

`_canonical_walk` steps through Assert/DebugWatch transparently and
`topology_entries` unwraps inputs — the docstrings state, and the code
confirms, that the encoding "is identical to walking the stripped
compiler-private copy".  Removing wrappers from topology therefore
leaves `topology_entries` **byte-identical**: schedule-cache keys keep
hitting, committed debug sidecars keep loading, and the sidecar's
"rebuild carries fewer checks than the compile did" warning survives as
a count of node checks.  A pinned-golden-hash test (below) keeps this
honest through the migration.

## Census: what the wrapper representation touches today

Verdicts: **D** deleted, **S** simplified, **R** rewritten (same job,
less machinery).

- graph/misc.py — `Assert`, `DebugWatch` classes → **R** (one `Check`
  dataclass; classes deleted).
- graph/asserts.py — constructor helpers → **S** (attach-and-return;
  the five multi-input helpers are the **R** exception, above);
  `collect_debug_nodes` → **R** (iterate nodes with checks);
  `assert_softmax_hardness`'s unwrap loop → **D**.
- ops layer — compare/select/map_select/logic/attention ops return
  Assert-wrapped results today (`assert_matches_value_type` at op
  tails; raw `Assert(cond, ...)` claims in `cond_gate` and two
  map_select/logic sites) → **S** (attach to the result node and
  return it).  High blast radius but mechanical: callers already treat
  the return as an opaque Node; only tests that unwrap op results
  change.
- graph/affine_rules.py — `_assert_rule` + dispatch + internal unwrap
  → **D** (channels move to choke points); DebugWatch pass-through →
  **D**.
- graph/optimize.py — no wrapper code today; gains the one explicit
  decline rule → **R** (policy made visible).
- compiler/lower.py — `_unwrap`, `_strip_debug_wrappers` (claim
  transfer + rewire + `integer_claimed` return), wrapper cases in
  node_map (two sites), chain-miner transparency, vocabulary entries →
  **D** (the strip has no job left).
- compiler/graph_clone.py — Assert/DebugWatch dispatch entries + notes
  → **D** (checks ride the generic `__dict__` copy; the stray-node-ref
  guard is unaffected — `Check` holds no node references).
- compiler/graph_identity.py — `unwrap_debug`, walk/encoding
  transparency → **D**/**S** (plain DFS).
- compiler/residual_assignment.py — alias-key unwrap and
  `flatten_concat_nodes`'s wrapper handling → **D**/**S**; `add_alias`
  may serve non-wrapper aliasing — verify its remaining callers before
  deleting.
- compiler/forward/graph_analysis.py — wrapper-free guard → **D**
  (unrepresentable).
- compiler/forward/compile.py — wrapper filtering for the node_map
  re-key, plus the whole `assert_aliases` mechanism
  (`HeadlessTransformer.assert_aliases` and the `compute()` loop that
  copies `result[wrapper] = result[target]`, transformer.py) — it
  exists only so result dicts can answer wrapper-keyed lookups → **D**.
- compiler/export.py — `_unwrap_output_node` (exists solely because
  compare/select/logic ops return Assert-wrapped results, so the graph
  *output* is routinely a wrapper) and a second unwrap loop → **D**;
  sidecar `assert_targets` (canonical ids of wrapped nodes) → **S**
  (ids of checked nodes, no `.inputs[0]` indirection);
  `CompiledHeadless(asserts=, watches=)` → **R** (lists of checked
  nodes; predicate evaluation against residual values loses its unwrap
  step).
- compiler/collapse.py — survivor unwrap, the post-synthesis re-strip
  in lower(), and the `integer_claimed` plumbing → **D** (the gate
  reads `source.integer_claim`; `piecewise_linear` attaches its clamp
  claim directly).
- debug/probe.py — wrapper skip + unwrap → **D**; `reference_eval`
  runs checks after each node's value lands (once per node, vs. once
  per wrapper-compute today) → **S**.  An existing inconsistency dies
  with the migration: probe's assert runner unwraps only Assert while
  extraction.py unwraps both wrapper types.
- debug/extraction.py — four unwrap loops → **D**;
  `check_debug_predicates` (the shared debug heart both
  `CompiledHeadless` and `OnnxDebugSession` route through) → **R**
  (iterate checked nodes, run each node's checks on its own compiled
  value).
- scripts/ — two investigation scripts strip wrappers by hand → **S**.
- debug/onnx_debug.py — `unwrap_debug` uses + wrapper-exemption
  docs → **S**.
- Tests — structural wrapper tests (`test_wrapper_transparent_lookup`
  entirely; the strip/claim-transfer tests in `test_lower`,
  `test_claimed_type_lowering`, `test_graph_clone`'s wrapper cases,
  `test_graph_analysis`'s guard; `test_debug_watch`'s and
  `test_onnx_debug_session`'s wrapper-shaped assertions) → **R** as
  claim-application tests pinning the same *behaviors*; the parity
  twin drops its strip step; the ~20 test files that merely *use*
  assert helpers compile unchanged (API preserved).
  `test_compiler_assertions.py` is a false positive — its 13 hits are
  the I1–I4 `AssertionError`s, not the Assert node; leave it alone.

Nesting (`Assert(Assert(x))`) and assert-on-output are handled today
by unwrap loops in seven places; both become unrepresentable.  Nested
checks merge order-independently into one node's list (claims
intersect commutatively — the same property the strip's node_id-order
processing already relies on).

## Migration plan

Single cutover (no compatibility shims — helpers keep their names and
signatures, so the diff is concentrated in the library):

1. **Pin the invariant first**: a test that hashes
   `topology_entries` for a fixed wrapper-using example graph and
   compares against a committed golden value, added *before* the
   migration commit.  It must pass unchanged after the cutover — that
   is the byte-identity claim, enforced.
2. **Core commit**: `Check` + node fields + choke-point claim
   application; helpers attach-and-return; delete wrapper classes and
   every **D** item above; explicit fold-decline rule; rewrite
   `collect_debug_nodes` / sidecar coverage / `CompiledHeadless`
   debug path; collapse pass cleanup (re-strip and `integer_claimed`
   plumbing removed).
3. **Test migration commit**: rewrite the structural wrapper tests as
   claim-application tests; keep every negative behavior (claim
   intersection, leaf channel, fold-decline, "rebuild has fewer
   checks" warning) pinned.
4. **Gates**: full suite; the golden-hash test; load an existing
   committed debug sidecar against a rebuilt graph (proves artifact
   compatibility); one doom-flagship compile with the schedule cache
   enabled, confirming a cache HIT (proves key stability end to end).

## What this unblocks (in order)

1. The collapse plan's post-strip fusion question dissolves: with
   claims refresh-proof, the general round is bounds-safe, and the
   remaining question — which folds may delete runtime checkability —
   is the explicit policy rule, decided on measurement.
2. The `split_7` source-Linear fold becomes a trivially safe ~15-line
   step in the collapse pass.
3. `docs/univariate_collapse_plan.md`'s Placement section and gate
   condition 1 get simpler (no strip bookkeeping to describe).
