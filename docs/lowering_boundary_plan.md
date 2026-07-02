# The lowering boundary and realization choice: design + plan (rev 2)

> **Historical note (2026-07):** the `Block` node described in this document
> has been renamed `FFN` (`torchwright/graph/ffn.py`), and
> `scripts/block_equivalence.py` is now `scripts/ffn_equivalence.py`.  Names
> below reflect the pre-rename vocabulary.

*Companion to `ir_semantic_vs_structural.md` (the diagnosis). Rev 1 of this
document was an end-state design with no sequencing. This revision follows a
code-verified review of rev 1: it corrects what the review found, scopes the
design to what the proposed machinery actually fits, and adds sequencing.
Written against the codebase as of the block refactor (`block-ir-2a`).*

## STATUS: B1–B3 COMPLETE (2026-07-02) — closeout record

All three scheduled steps landed on `block-ir-2a` (B1 `342751e` + follow-up,
B2 `4849873`, B3 `c23a46c`); item D remains deferred as designed.

**Gates, verified:**
- Full `make test` green (10 shards) at each step boundary.
- Flagship schedule metrics (e1m1, hud-off, eager heuristic):
  **(57, 781, 16384, 8192)** — byte-identical to the step-1 closeout's
  post-refactor tuple.  10,954 nodes certified at the boundary; build
  95.9s, schedule 17.0s.
- No `blockify` symbol remains in either repo (one extra call site was
  found beyond the plan's two — torchwright's own
  `scripts/block_equivalence.py` harness, missed because the B1 sweep
  grepped the package dir, not repo-root `scripts/`).
- Routing decisions live only in the two resolvers (`resolve_static`;
  the solve via `resolve_from_assignment`); `_route_linear_to_attn` and
  the directed override are deleted.
- `cost_summary()` reconciles with the solve's accounting on suite
  graphs (bypass slots equal; total heads within the Add bounds); the
  flagship demand line reads: attn heads 398..464 (attn_heads=332,
  adds 66..132), 10,176 bypass slots, 185,256 lanes.  Note the metrics
  tuple's 781 heads also counts schedule-emergent cancel heads — the
  summary is compute demand, by design.

**One deviation from the B1 text:** `forward_compile` keeps its
`output_node: Node` signature (57 direct call sites in tests) and calls
`lower()` unbypassably as step 0, ahead of `GraphAnalyzer`; the
`LoweredGraph` handoff is internal (its table feeds the resolvers and the
walk).  The boundary guarantee is by construction rather than by caller
signature — every compile certifies; nothing reaches the scheduler
uncertified.

**Implementation notes for future readers:**
- Semantic affine overrides (`_apply_semantic_override` — cond_gate,
  compare, select) are persisted on the node and re-applied by
  `refresh_node_caches`; a recompute that dropped them would silently
  loosen every downstream bound.  Fusion's Block→Linear fold voids the
  survivor's override (that fold changes the survivor's value; the other
  two folds preserve it).
- `LayerScheduler` constructed without a table resolves statically from
  its policy through the same resolver code path (standalone/test
  convenience); compile passes the table explicitly.
- `ScheduleAssignment.node_to_routing` and its schedule-cache
  serialization are unchanged; the table is the shared in-memory
  artifact, resolved fresh per compile.

**Remaining (cleanup/decisions, no open engineering):**
1. torchwright_doom: `block_equivalence_flagship.py` switched to
   `lower()` + demand line (committed on `block-ir-2a-r8`); umbrella
   pointer bumps happen at merge time as usual.
2. `scripts/block_equivalence.py`'s two-path comparison is vestigial —
   since Phase 2b/3 both "paths" build the same Block-native graph.
   Candidate deletion (keep `schedule_metrics`, drop the chain/block
   compare); left for an explicit decision since the harness is
   committed Gate-C tooling.

## What changed from rev 1

1. **Two kinds of realization choice are now distinguished** — a *write-path
   choice* (one fixed node, two possible hardware emissions) and an *op-site
   choice* (two different subgraphs for one op call site). Rev 1 treated the
   second as a smooth generalization of the first; it is not — it changes
   which nodes exist and who consumes whom. The realization table is scoped
   to write-path choices; op-site candidates move to a deferred section with
   the open representation question stated instead of papered over.
2. **The tripwire claim was understated.** Rev 1 said `lower()` "absorbs
   blockify's tripwire." In fact no production compile entry point runs any
   vocabulary check today — `forward_compile`, `compile_headless`, and
   `compile_to_onnx` contain no `blockify()` call; only tests and the
   manually-run `block_equivalence_flagship.py` invoke it. `lower()` is the
   first time the check would run on ordinary compiles — and once it does,
   the standalone pass has no reason to exist: **`blockify` is deleted in
   B1** (the miner moves into the boundary validator; only the `ReLU` node
   class survives, as detector/affine-rule internal).
3. **`node_to_routing` is not a private dict.** It is a documented
   `ScheduleAssignment` field already serialized in the schedule cache
   (`schedule_cache.py:82,125`). Promoting it into the shared table makes
   the cache format a named compatibility surface: keep the serialized
   shape, or bump the cache key.
4. **Constraints added** (stated up front rather than discovered later):
   the freshness recompute must use a true topological sort, not node-id
   order; `lower()` must run before GraphAnalyzer's Assert-strip tightening;
   the `ReLU` node *class* stays (it is blockify's detector internal, per
   the step-1 disposition) while `ReLU` remains rejected as graph
   vocabulary.
5. **Sequencing added** — steps B1–B3 with gates, plus an explicitly
   unscheduled deferred item D.

## The position (unchanged from rev 1)

**The graph is the structural IR. The semantic level lives in the ops layer**
(the op functions plus `ops_plain_english.md`). There is one `Node`
substrate — `Block` and `Attn` are structural nodes, and `Swish`/`Mul` never
become node types; the nonlinearity and the gate exist only inside a Block
lane.

Why one substrate: every piece of verification machinery works uniformly
because everything is a Node with `compute()` — the recursive oracle,
`probe_compiled`, affine bounds, Assert/DebugWatch, `debug_value`. A second
node hierarchy would fork that debug surface, and the debug surface is the
most valuable property the codebase has. This is the lesson of the
successful multi-level compilers (MLIR): levels share one operation
substrate, and a "level" is a *vocabulary plus a validator enforced at a
stage boundary*, not a separate class hierarchy.

What the codebase lacks is therefore not a second IR but a **boundary**: a
moment where a graph is certified ready for the scheduler, and a declared
place where realization decisions land.

## Vocabulary

- **Realization** — how a piece of graph math becomes concrete transformer
  hardware (attention heads vs MLP hidden slots).
- **Realization class** — a *write-path choice*: which hardware computes a
  fixed, already-built node. A standalone Linear is `attn_transport` (Δ=0
  self-match head) or `mlp_bypass` (`act(z) − act(−z) = z`, two hidden
  slots per output column — an identity that holds exactly for swish as
  well as relu, so the class survives the step-2 gated-MLP move); an Add is
  `residual_reuse` (add_into, needs a dead addend) or `attn_copy`
  (compute_add); a Block is always the MLP composite; an Attn is always
  attention heads; a LiteralValue is always MLP-written. In every case the
  node, its weights, and its consumers are unchanged — only the emission
  differs.
- **Op-site candidates** — a *subgraph choice*: one op call site with more
  than one buildable form, where the forms differ in node count, weights,
  and liveness. Exemplar: `select(cond, a, b)` (returns `a` when cond is
  true, else `b`) as one Block with two gate lanes that only *reads* `a`
  and `b`, versus `b + indicator·(a−b)` — one lane computing the
  correction, written onto the residual columns already holding `b`,
  consuming them. **The realization table below does not represent this
  kind of choice.** See *Deferred: op-site candidates*.
- **Placement** — *which layer, which columns and slots*. The layer walk's
  job.

The class/placement split formalizes behavior the scheduler already has:
realization class never flips during the layer walk. A bypass-routed Linear
that doesn't fit a layer's hidden slots is deferred to a later layer, never
re-routed to attention; same for attention-routed nodes under head pressure
(both verified in `scheduler.py` — the head-fit check skips and retries,
and routing is a pure function of node + policy or a table read). Placement
flexes; class does not.

Today, realization classes get resolved in two unrelated forms: a policy
constant consulted mid-walk (`LayerScheduler._route_linear_to_attn`) in the
eager path, and the CP-SAT solve's per-Linear flex variables landing in
`ScheduleAssignment.node_to_routing` in the directed path. The option set
itself is hard-coded twice — `_route_linear_to_attn` on one side, `is_flex`
(cpsat_scheduler) on the other — and nothing checks that the two agree.
The end state gives that resolution one declared stage and one shared
artifact; it deliberately does not coin a component name for it — the
resolvers are the concrete things they already are (the static policy; the
solve).

## The pipeline

```
ops build graph
   → fusion (may delete a Linear by folding it into a Block;
             stays a driver-invoked pass, as today)
   → lower()          — certifies vocabulary; recomputes derived caches;
                        produces LoweredGraph with an UNRESOLVED
                        realization table:
                          linear_47 → {attn_transport, mlp_bypass}
   → resolve the      — every entry resolved to one class
     realization         optimize=0: static policy (microseconds)
     table               optimize>0: the CP-SAT solve itself, jointly with
                         layer assignment (the entanglement is the point;
                         the design separates the INTERFACE, not the solve)
   → layer walk       — pure reader; places classes, defers under pressure
   → weight writer    — emits the class's write path
```

The table is resolved between `lower()` and the layer walk, and nowhere
else. `lower()` only establishes what the options are (it cannot choose
well — the choice is resource-coupled); the walker only executes the
choice. Swapping what resolves the table (static policy ↔ solve ↔ a future
per-op cost model) changes nothing upstream or downstream of it.

**Ordering constraint.** `lower()` runs *before* `GraphAnalyzer`'s
Assert-strip, which transfers claimed ranges onto wrapped nodes and
tightens their structural types and bounds. The freshness recompute must
happen first so the tightening lands on top of fresh bounds — recomputing
after the strip would wipe the assert-derived tightenings, and the
stale-bounds bug this boundary exists to prevent (0570af1) was precisely an
affine/structural disagreement surfacing in the RMSNorm energy
certification.

## The typed handoff: `LoweredGraph`

`lower(output_node, ...) -> LoweredGraph`, called internally by
`compile_headless` / `compile_to_onnx` (their signatures don't change) and
callable standalone for tests and inspection. Constructing a `LoweredGraph`
validates:

1. **Closed vocabulary** — only Block, Attn, Linear, Add, bookkeeping
   (InputNode, LiteralValue, Embedding, Concatenate), and debug nodes
   (Assert, DebugWatch, ValueLogger) remain. No `ReLU` *in the graph* and
   no unclaimed L→R→L chain shapes. The chain miner moves here from
   `graph/blockify.py`, and **the standalone `blockify()` pass is deleted
   in the same step**: an unarmed tripwire (today the check runs only in
   tests and one manual flagship script; no compile entry point calls it)
   is cruft the moment the boundary runs the same detection on every
   compile. The `ReLU` node class itself stays, per the step-1
   disposition — it is the miner's and the affine rules' internal.
2. **Fresh derived data** — `_affine_bound` / `_structural_type` are
   recomputed over the reachable set in **true topological order** (not
   node-id order: the "inputs always have smaller ids" invariant is a
   property of fusion's folds, not of graphs in general, and the boundary
   must not inherit it). The stale-bounds class of bug becomes structurally
   impossible at this boundary instead of being guarded by per-pass
   discipline inside each mutating pass.

The scheduler entry points accept only `LoweredGraph`. That is the
type-level boundary: an unlowered graph *cannot* reach the scheduler, by
signature rather than by convention. Pre-lowering, a graph is whatever the
ops layer builds; post-lowering, it is certified compilable vocabulary.

## The realization table

`LoweredGraph` carries one artifact recording every schedulable node's
realization class — initially unresolved for nodes with more than one
candidate class. Both resolvers — the static policy and the CP-SAT solve —
write this one table; the layer walk reads it (`_route_linear_to_attn`
becomes a table read; the directed path's `node_to_routing` *is* this
table, promoted from solver output to shared artifact).

Three constraints rev 1 left implicit:

- **One declared option set.** Which node types have which candidate
  classes is declared in exactly one place, read by both the static
  resolver and the solve's flex-variable construction. This is what
  actually kills the `_route_linear_to_attn` / `is_flex` parallel
  hard-coding — the table alone would not.
- **Cache compatibility.** `node_to_routing` is serialized in the schedule
  cache. The promoted table keeps that serialized shape, or the cache key
  is bumped in the same commit.
- **Conditional ≠ unresolved.** Add's entry is an explicit conditional —
  "`residual_reuse` if the addend is dead at schedule time, else
  `attn_copy`". Deadness depends on layer assignment, which is exactly what
  the solve co-decides (via reified consumer-ordering booleans) and what
  the eager walk discovers as it goes. A conditional entry encodes a
  *predicate on schedule state*; it can represent a choice the schedule
  determines, but it cannot resolve a *free* choice — which is one reason
  op-site candidates don't fit this table (see below).

A completeness check runs before the walk: every entry resolved or
explicitly conditional. And because the table plus each class's resource
signature determine hardware demand, `LoweredGraph.cost_summary()` — heads
by class, bypass slot demand, lane counts — is readable *before*
scheduling: the compile-metrics tuple stops being something only a finished
compile can report.

Why one table: it is the difference between realization being *decided* and
being *taken*. With one artifact there is one vocabulary of options, one
place to validate completeness, one surface to inspect and diff, and one
plug point for anything smarter.

## Solver-chosen write-path realizations

In the directed path, each multi-class node's entry becomes a choice
variable in the CP-SAT model, with resource demands conditional on the
choice, co-optimized with layer assignment. For **write-path choices this
genuinely is not new machinery in kind**: `is_flex` already does exactly
this for the one hard-coded case (standalone Linear: heads vs slots), and
the generalization is reading the option set from the shared declaration
instead of an isinstance check.

The entanglement of realization choice with placement stays *inside* the
solve, on purpose: the right realization depends on which layer a node
lands in and what else competes for that layer's heads and slots. What the
design separates is the interface — where the answer lands — not the solve.

The same sentence is **not true of op-site candidates**: those change which
nodes exist and who consumes whom, so the model would need per-candidate
conditional *liveness* (in the select example, `b`'s cancel window — the
span of layers during which `b`'s residual columns stay allocated — depends
on the choice), a materially bigger model than a demand swap. That is the
second reason they are deferred.

## Deferred: op-site candidates

The end-state idea stands: an op registers **candidates** — each a builder
emitting ordinary graph vocabulary, plus a resource signature (heads,
hidden slots, residual columns, which inputs it consumes vs keeps live) —
and a resolver picks per call site. An op with a single candidate is the
legitimate degenerate case. Candidates emit existing graph vocabulary, so
everything downstream of the choice is the unchanged pipeline.

It is deferred because it contains an **unresolved representation
question** rev 1 did not acknowledge. The pipeline pins resolution after
`lower()` (and it cannot move earlier without forfeiting the
co-optimization with layer assignment). But executing a resolved subgraph
choice requires the chosen form's nodes and weights to exist, and each
obvious mechanism carries real costs — the list is open, not a dichotomy:

- **Rewrite after certification** — the resolver runs the chosen builder on
  the lowered graph. Bounds freshness, the structural fingerprint, and
  schedule-cache keying must all re-run post-rewrite; "certified" stops
  meaning "final".
- **Placeholder node expanded post-resolution** — strains "everything is an
  ordinary Node with `compute()`" unless the placeholder carries a default
  expansion for the oracle and probes.
- **Dual parameterization on one node** — build both forms' weights up
  front, emit the chosen one. Add gets this for free only because its two
  classes share one node and weight-free semantics; select's forms differ
  in lane count and weight tensors.

**Trigger to build it, and the measurement that justifies it:** when the
step-2 gated `select` op is written, hand-build both forms and diff the
compile-metrics tuple on the flagship graph. If the delta is material,
choose a representation from the list above *first*, then build the
registry. Until then, hand-authoring the best form per op — what
Block-native ops already are — is the diagnosis doc's own "cheaper hedge",
and it stays the operating mode.

## What the design deliberately excludes

- **A second node hierarchy.** Rejected above; the boundary is a stage
  property, not a type split.
- **Schedule-time class re-routing** (flipping a Linear to attention
  because a layer's MLP is full). The table makes it *representable* — it
  would be a visible policy change rather than an accident — but the design
  does not include it: class stability during the walk is what keeps
  placement simple and schedules explainable.
- **Fusion as a solver choice.** Fusion remains a greedy graph pass
  upstream of `lower()`. Its decisions are visible in the lowered graph (a
  folded Linear simply isn't there), which is enough.
- **Hoisting Add's deadness decision out of the scheduler.** Impossible
  without also hoisting layer assignment; the conditional entry is the
  honest representation.
- **Op-site candidate machinery, until the trigger above fires.**

## Sequencing

All three steps are numerics-frozen (no op math changes, no weight
changes); each gate includes the full suite green and the flagship schedule
metrics unchanged — (57, 781, 16384, 8192) for the non-HUD e1m1 build,
(61-layer figure) for hud=1, measured with the screen env set *before*
torchwright_doom imports (see the step-1 closeout trap).

### B1 — the boundary

Deliverables: the `LoweredGraph` type; `lower()` performing both
validations (chain-miner vocabulary check; topological-order cache
recompute); wired into `forward_compile` ahead of `GraphAnalyzer`;
scheduler entry points typed to `LoweredGraph`; **`graph/blockify.py`
deleted** — the miner relocates into the boundary validator, the
standalone `blockify()` entry point is removed, and its call sites (the
`tests/graph/test_blockify.py` suite; `block_equivalence_flagship.py` in
torchwright_doom) switch to `lower()`.

Tests: blockify's negative tests move to the boundary (raw chain →
`lower()` raises); a stale-bound reproducer at the boundary layer (mutate a
node's weights in place, assert `lower()` restores bound = fresh recompute
— the boundary-level twin of `test_fusion_refreshes_stale_bounds`); an
unlowered-graph-cannot-reach-the-scheduler negative test.

Gate B1: suite green; flagship metrics identical; no `blockify` symbol
remains in either repo (the torchwright_doom call site lands with the
pointer bump).

### B2 — the table

Deliverables: the table artifact on `LoweredGraph`; the option set declared
once and read by both resolvers; the optimize=0 static resolver writing the
table before the walk; the solve writing the same table (`node_to_routing`
promoted, serialized shape kept or cache key bumped); the walker reading
only the table (`_route_linear_to_attn` deleted in both scheduler classes);
the completeness check.

Tests: eager and directed paths produce identical tables under a pinned
policy on a fixed graph; unresolved entry reaching the walk → error;
schedule-cache round-trip of the table.

Gate B2: suite green; flagship metrics identical; grep shows no routing
decision outside the resolution stage.

### B3 — cost_summary()

Deliverables: `LoweredGraph.cost_summary()` (heads by class, bypass slot
demand, lane counts) computed from the resolved table + resource
signatures; the flagship metrics script gains a pre-schedule summary line.

Gate B3: on the flagship graph, the summary's totals reconcile with the
finished compile's metrics tuple.

### D — deferred (not scheduled)

Op-site candidates + solver choice over them. Blocked on: the step-2
`select` measurement (both forms hand-built, metrics diffed) showing a
material delta, then a decided representation from the list in *Deferred:
op-site candidates*.

## Relation to step 2 (SwiGLU)

B1–B3 change no numerics and don't touch the op layer, so they can land
before the step-2 op rebuild. That ordering is the point of doing them now:
the gated-MLP ops then land on a certified boundary and a declared
resolution surface, instead of adding a third ad-hoc place where routing is
decided — and the step-2 `select` op is itself the measurement that decides
whether D ever gets built.
