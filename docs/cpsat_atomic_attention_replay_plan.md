# Atomic attention replay plan

Status: IMPLEMENTED, 2026-07-13. Revised the same day to fold in the
adversarial review (`cpsat_atomic_attention_replay_plan_review.md`,
findings 1–7), then implemented per §6's sequence.  The batch lives in
`_schedule_attn_sublayer` + the `_assigned_attention_releases` hook
(`scheduler.py`); fixtures and contract negative tests in
`test_cpsat_intralayer.py`; held-batch pin in `test_tied_embeddings.py`;
cache-commit tests in `test_cpsat_knobs.py`.

This plan fixes the CP-SAT directed-replay deadlock exposed by
`test_solver_same_layer_handoff_replays_correctly`. The fix must preserve the
solver's feasible set and objective, emit exactly the solver-assigned depth,
and avoid introducing a second scheduler or a more detailed CP-SAT time axis.

The design decision is:

> Replay the solver-assigned attention sublayer as the atomic read/cancel/write
> event that the residual cumulative already models.

This is an executor correction, not a scheduling-model change. The existing
attention weight writer already realizes all heads against the same
pre-attention residual and sums their deltas. The replay allocator must stop
requiring an artificial per-output bootstrap order before emitting those
heads.

## 1. Outcome and constraints

The completed change must provide all of the following:

1. Every sound CP-SAT assignment remains feasible. No CP variable, constraint,
   objective term, or layer bound changes.
2. Directed replay emits `assignment.n_layers`, including assignments with a
   collective same-layer attention handoff.
3. `optimize=0` heuristic scheduling is unchanged, including existing golden
   layer counts.
4. There is no scratch-column reservation, no extra transformer layer, no
   re-solve, and no deterministic-seed workaround.
5. The existing `AttnHeadOp`, weight-writer algebra, and runtime format remain
   unchanged.
6. Held tied-embedding columns keep the current `allocated -> held -> target`
   ownership transition. Atomic replay generalizes when that transition can
   occur; it does not add an MLP handoff or expand the tied-embedding scope.
7. The implementation should delete the directed retry/unary-reuse machinery
   that the batch transition supersedes, keeping the net conceptual change
   small.

## 2. Failure being fixed

The failing fixture has eight branches of the form:

```text
x -> Li[12] -> Ma_i[2]
             -> Mb_i[2]
```

At the observed failing layer, replay has one free physical residual column.
`Ma7` and `Mb7` are both attention-routed, both read `L7`, and both require two
fresh target columns. `L7` has twelve columns and is assigned an attention
cancel in that layer. `Mb5` is also assigned to the MLP and requires two
columns.

The model admits the layer because its aggregate transition is:

```text
free on attention entry                  1
attention-cancel L7                    +12
allocate Ma7 and Mb7                    -4
allocate MLP-routed Mb5                 -2
-------------------------------------------
free after the layer                     7
```

Attention compute capacity is exact rather than violated:

```text
Ma7 transport                            2 heads
Mb7 transport                            2 heads
coalesced cancellation of L7[12]         2 heads
-----------------------------------------------
total                                    6 heads
available at d=48, d_head=8              6 heads
```

The current replay attempts one output at a time. Neither `Ma7` nor `Mb7` is
individually the last consumer of `L7` while the other remains uncomputed, so
neither can trigger `_dying_input_to_reuse`. One free column is insufficient
for either two-column output. Retrying the same candidates cannot create a
first operation.

This is the smallest instance of a general mismatch:

- the model requires a feasible aggregate attention transition; while
- replay currently requires an ordering in which each individual output can
  allocate after releasing at most one input.

The aggregate condition does not imply that such an ordering exists. It does
imply that a simultaneous attention transition exists.

## 3. Semantics and invariants

Let, for attention layer `t`:

- `F_t` be ordinary columns free on entry;
- `B_t` be fresh attention outputs assigned to `t`;
- `M_t` be fresh MLP outputs assigned to `t`;
- `R_t` be allocated values assigned an attention-mechanism cancel at `t`;
- `w(v)` be the physical residual width of value `v`.

Ignoring free Adds, which transfer an existing allocation, the residual model
establishes:

```text
sum(w(v) for v in B_t) + sum(w(v) for v in M_t)
    <= F_t + sum(w(u) for u in R_t)
```

An attention cancellation at `t` is legal only when every uncomputed consumer
of the cancelled value reads in that attention sublayer. A same-layer
MLP-routed consumer forces a later attention cancel or an MLP-mechanism cancel,
so it cannot appear in `R_t`. A value born at `t` cannot also be cancelled at
`t` because `cancel_layer >= birth_layer + 1`.

Two further model bounds pin down what `R_t` can contain, and A2's
validation relies on both. First, a free Add's addends — the dead one and
the live one — are bounded `cancel >= layer[A] + is_free[A]`, one layer
after a free Add at `t`, so a value read by this layer's `add_into` can
never be in `R_t` and the batch can never collide with the free-Add
`reassign`. Second, ordinary freeable inputs carry the uniform gap-1 bound
(`cancel >= layer[c] + 1` for every consumer, regardless of routing), so
any input appearing in `R_t` is already dead on entry to `t`; the held
source is the only input the model permits to die mid-attention. An A2
implementation that "defensively" handles either forbidden case is not
defensive — it is dead code that would hide a model regression.

The executor must preserve these invariants:

### A1. Capture before release

Every source-column list for every attention compute operation is captured
while all of its inputs are still allocated. No cancellation or allocator
ownership change occurs first.

### A2. Whole-batch last-reader validation

For each `u in R_t`, every effective consumer of `u` is either already
computed or a member of the attention batch being captured. If not, the
assignment and replay contract disagree and compilation fails before mutation.

### A3. Exact attention charge

Before mutation, replay checks:

```text
free-Add heads
+ sum(compute heads for B_t)
+ ceil(number of distinct cancel columns / d_head)
<= n_heads
```

The cancel term is charged once over the coalesced union, matching the CP-SAT
attention cumulative and the existing writer.

This check is not merely consistent with the model; it is equivalent to it.
The attention cumulative pools compute-head units and raw cancel columns at
capacity `(n_heads - reserve_heads) * d_head`, and every compute charge is
a whole number of heads, so

```text
union_cols <= (n_heads - sum(compute heads)) * d_head
    <=>  ceil(union_cols / d_head) <= n_heads - sum(compute heads)
```

The per-op compute charges agree exactly on both sides (`linear_attn_heads`
is shared by the model, the scheduler, and the emitter; an `Attn` charges
`ceil(d_v / d_head)` on both sides; a free Add charges the same head count
on both sides), except that a compute-Add's replay charge is at most the
model's `2 * ceil(d_out / d_head)`. A model-feasible assignment therefore
cannot fail this preflight. That is the license for the assertion policy
below: an A3 failure is a model/replay contract violation, never a
legitimate deferral.

### A4. Aggregate residual preflight

Before mutation, replay checks that ordinary target width fits after ordinary
releases. Held columns and their exact target are excluded from both sides of
the ordinary-free calculation: `hold(source)` does not make the bank generally
free, and `allocate_at(target, bank)` claims that held bank without consuming
ordinary columns.

Because `ResidualStreamMap` permits arbitrary noncontiguous columns, aggregate
free width is sufficient; there is no fragmentation condition.

Order-of-operations caveat: the preflight runs before the release commits,
so the bank is not yet in the held state and `can_allocate_at` would return
a false negative. The preflight must check the intended transition directly
— the held target's width equals the source bank's width, and the held
source is in this batch's release set (or the bank is already held from an
earlier layer) — not call the allocator predicate.

### A5. One physical attention event

All captured compute heads and the coalesced cancel head read the pre-attention
residual. The allocator transitions are compile-time ownership bookkeeping;
they do not impose runtime head order.

### A6. MLP boundary remains unchanged

MLP-cancelled values remain live through their cancel layer and are released
only after MLP compute. This plan adds no atomic MLP handoff. Once the attention
batch has been committed, existing sequential MLP allocation is sufficient:
the residual cumulative already counted every MLP birth and retained every
MLP-cancelled source through the layer.

## 4. Minimal implementation shape

The change belongs in the existing `_schedule_attn_sublayer`; do not create a
second directed scheduler implementation and do not duplicate weight-writing
logic.

### 4.1 Factor source capture once

Extract the existing `compute_linear`, `compute_attn`, and `compute_add` source
capture block into a private helper with the conceptual signature:

```python
_capture_attn_sources(op_type, node, residual_map) -> dict[str, list[int]]
```

The helper performs the current `_require_live` checks and returns the exact
attributes later copied to `AttnHeadOp`. The heuristic calls it lazily, exactly
where it captures sources today. Directed replay calls it for its complete
attention batch before releasing anything.

This relocates compiler invariant I4's schedule-time check with the capture
site; it does not bypass it. `_require_live` runs once per batch member
while every input is still allocated and before any release commits — that
is A1, and it is I4's content (sources are captured while live). The
placement loop then consumes the saved capture and must not re-run the
liveness lookup: after a legitimate release, a placement-time re-check
would be a wrong duplicate, not a safeguard. `_require_live` itself is
unchanged, so the existing I4 negative test in
`test_compiler_assertions.py` continues to pin the invariant.

This is a refactor, not a new representation. A local dictionary keyed by node
is sufficient; no public `AttentionBatch` class is needed.

### 4.2 Add one polymorphic batch-selection hook

Add one protected hook used by the shared attention scheduler:

```python
_assigned_attention_releases(
    batch_nodes, residual_map, computed_nodes
) -> Optional[list[Node]]
```

- The base `LayerScheduler` returns `None`. `None` means retain the existing
  greedy heuristic behavior.
- `DirectedLayerScheduler` returns the complete set of allocated values whose
  assignment says `cancel_layer == current_layer` and cancel mechanism is
  attention.

The directed override validates A2. It must include both values already dead
at layer entry and values that become dead only after all batch readers have
run. The held source is allowed through the existing registered
`HeldOutputLayout`; ordinary keep-forever values and the constant column remain
ineligible.

Use equality with the current layer, scoped to this hook only. Immediately
after collecting the equality matches, the hook also scans the allocated
values for an attention-mechanism `cancel_layer < current_layer` and raises,
naming the layer, the node, and its assigned cancel. An overdue directed
cancellation must be an assertion failure, not silently rescheduled with
`<=`; an assignment that fit only because a prior cancel occurred on time is
not soundly replayed by delaying it — and without the explicit scan, an
equality-only match would implement "assert" as "silently leak the columns."

Do not change the `<=` in the directed `_find_dead_nodes` override. After
this plan it serves only the MLP-cancel path, whose `cancel_bypass` batch
still defers silently when a layer's hidden slots run out and relies on the
`<=` to resurface the deferred node. Converting that deferral into an
assertion is justified by the same contract argument as A3 (the model
charges the `2 * len` hidden slots at the assigned cancel layer), but it is
a second behavioral change with its own red test — out of scope; see §8.

### 4.3 Preflight and commit in the shared method

After free Adds have been handled and attention compute candidates have been
built, the shared method does the following only when the hook returns a list:

1. Form `batch_nodes` from every attention compute candidate assigned here.
2. Capture sources for every batch node using the helper from §4.1.
3. Resolve the physical columns of every release candidate.
4. Validate the complete attention-head charge (A3).
5. Validate ordinary residual width and any held exact-bank claim (A4).
6. Add the release columns to the existing coalesced cancel batch.
7. Call the existing `_release_cancelled` for every release candidate:
   ordinary values become free; a held source becomes held.
8. Run the existing candidate placement loop. It consumes the saved source
   dictionaries rather than looking up released inputs again.

After the preflight, every directed attention target must allocate. A failure
at that point is an internal model/replay assertion with layer, free width,
released width, requested width, and held-bank state—not a normal deferral.

When the hook returns a list, the legacy dead-list machinery is bypassed
for that attention sublayer: `cancel_candidates` is not built from the
layer-entry dead list, the placement-time cancel promotion does not run,
and the end-of-sublayer leftover-cancel loop does not run. Every directed
attention-mechanism cancel flows through the batch — the batch is the
single definition of the attention transition. Without this bypass, values
dead at layer entry appear both in the batch's release set and in the stale
dead list, and the leftover loop calls `get_indices` on already-released
nodes (a `KeyError`) or charges and releases the same value twice. The
attention-side eager-freeing surface (`_freshly_dead_inputs` feeding cancel
candidates mid-placement) becomes inert under the batch — released values
are no longer allocated — but it stays untouched for the heuristic and for
the MLP sublayer, whose gap-0 MLP-mechanism cancels still surface through
it.

The emitted `AttnHeadOp` objects and final coalesced `cancel` operation are
unchanged. The writer therefore needs no new operation type or execution path.

### 4.4 Delete superseded directed machinery

Once the batch path is green:

- remove `_retry_within_layer` from both scheduler classes;
- replace the retry-to-fixpoint loop with the heuristic's existing single
  candidate pass;
- remove `DirectedLayerScheduler._dying_input_to_reuse` for ordinary values;
- retain only the base scheduler's narrow held source-to-target handoff needed
  by `optimize=0` heuristic replay;
- update comments that claim unary self-consumer reuse makes the residual model
  exact; and
- update the directed `_find_dead_nodes` comment: its `<=` no longer serves
  attention cancels (the batch hook owns those, with equality plus the
  overdue assertion) — it remains only for MLP-cancel deferral re-surfacing.

Directed placement no longer needs retries or unary reuse: all assigned
same-attention releases have already occurred after source capture. Removing
these mechanisms is part of the fix, not optional cleanup; it prevents two
competing definitions of same-layer handoff from surviving.

### 4.5 Commit schedule cache entries only after replay

Move `store_assignment` from immediately after solve to the successful end of
the compile, after directed replay, replay-depth validation, and output-layout
validation. A cache hit is not rewritten.

This is defense in depth. Atomic replay is the correctness fix, but no future
model/replay defect should turn a transient failed solve result into a
persistent cached failure.

No transaction or cache format change is required. Keep the assignment and its
existing metadata in local variables until compilation succeeds.

One deliberate behavior change rides along: today the store precedes the
`_solve_only` early return, so a sound solve-only measurement run populates
the cache; after the move it no longer does. That is intended — a
measurement seam should not write production cache state — and §5.4's tests
pin it.

## 5. Deterministic regression fixtures

Tests must not depend on which equal-depth optimum a parallel CP-SAT run
returns. The current width-starved solver test can remain as a smoke test, but
the incident is pinned by small hard-fixed assignments.

### 5.1 One source, two simultaneous readers

Add a minimal fixture to `test_cpsat_intralayer.py`:

```text
inputs:
    x[4]
    blocker[10]

S[8]   = Linear(x)
A[2]   = Linear(S)
B[2]   = Linear(S)
out[4] = Linear(Concat(A, x, B, blocker))

d=24, d_head=4, d_hidden=24
```

Keep `A` and `B` separated in the terminal `Concatenate` so lowering does not
fold adjacent sibling Linears.

Hard-fix this assignment by lowered node name:

```text
layer 0: S via attention
layer 1: A and B via attention; out via MLP
cancel:  S at layer 1 via attention
depth:   2
```

Two lowering gates keep this fixture intact, and the fixture comment must
name both. `A` and `B` are sole-consumer Linear leaves of a Concatenate
whose sole consumer is a Linear, so `fuse_consecutive_linears`'
fold-through-Concatenate case applies structurally and is declined only by
its never-grow-parameter-count gate (folding `A[8->2]` into `out`'s 2x4
block would cost 8*4 = 32 parameters against the current 8*2 + 2*4 = 24) —
that gate is the property a future fold change would silently break. The
width-four `x` additionally keeps the graph out of the univariate collapse
pass. This exact source and lowered model have both been checked feasible
at depth two, and the current replay fails on the fixed assignment with the
one-free-column no-progress error.

On entry to layer 1, the constant column plus `x`, `blocker`, and `S` occupy 23
of 24 columns, leaving exactly one free. After releasing `S`, writing `A` and
`B`, and writing the four-column MLP output, one column remains free. The layer
uses exactly six of six attention heads:

```text
A reads S[8]        2
B reads S[8]        2
cancel S[8]         2
```

Add two tests over this fixture:

1. `test_collective_readers_assignment_is_cpsat_feasible` hard-fixes every
   relevant layer, route, cancel, and mechanism variable and asserts the model
   is feasible.
2. `test_collective_readers_replay_atomically` feeds that exact assignment to
   `forward_compile` by monkeypatching `compile.solve_schedule` at the existing
   private test seam. The replacement receives the lowered graph, maps names to
   its node ids, and returns the fixed assignment. Assert:
   - compilation succeeds;
   - emitted depth is exactly two;
   - the handoff layer uses exactly six of six attention heads, read from
     the compile's recorded per-layer head counts
     (`per_layer_head_counts`), not re-derived by hand;
   - output matches graph evaluation; and
   - the solver assignment is not modified or re-solved.

Before the fix, test 1 passes and test 2 raises the one-free-column no-progress
error. That pair directly proves the model/replay mismatch and its repair.

### 5.2 One reader, multiple dying inputs

Add the dual collective-handoff fixture:

```text
inputs:
    x[4]
    y[4]
    blocker[6]

Sx[4]  = Linear(x)
Sy[4]  = Linear(y)
C[6]   = Linear(Concat(Sx, Sy))
out[2] = Linear(Concat(C, x, blocker, y))

d=24, d_head=4, d_hidden=24
```

Hard-fix `Sx` and `Sy` to layer 0, `C` to attention in layer 1, both source
cancels to attention in layer 1, and `out` to the layer-1 MLP. Entry has one
free column. Releasing only one four-column source leaves five columns, still
short of `C[6]`; releasing both leaves nine and succeeds. Compute plus the
coalesced eight-column cancel uses four attention heads.

This catches the other incompleteness of the unary API: even a single consumer
may need the combined release of more than one dying input.

The general linear lowering pass intentionally collapses this algebraically
linear graph, so test the replay half directly through `DirectedLayerScheduler`
rather than disabling or mocking lowering. Pair a hard-fixed CP feasibility
assertion on the source graph with a scheduler test that:

- allocates the constant and three inputs;
- replays the fixed layer-0 births;
- observes exactly one free column on layer-1 entry;
- verifies both `Sx` and `Sy` are in the one coalesced cancel batch;
- verifies `C.source_cols` is the captured pre-release concatenation; and
- verifies `C` and the layer-1 MLP output both allocate without an extra layer.

The first fixture supplies the real-lowering, real-writer, numerical-parity
coverage. This fixture isolates the dual allocator case without adding a test
escape hatch to production lowering.

### 5.3 Source-capture and held-bank assertions

Extend the tied-embedding tests with one focused assertion over a direct held
handoff:

- every operation reading the held source has its source columns captured
  before `hold(source)`;
- the held bank is never reported as ordinary free;
- the target claims exactly that bank; and
- ordinary attention outputs in the same batch cannot claim held columns.

The existing end-to-end tied parity tests remain the primary artifact check.
No separate held-batch implementation should be added.

### 5.4 Cache failure test

Add a cache test that monkeypatches directed replay to raise after a successful
solve and records calls to `store_assignment`. Assert it is not called. Add the
positive mirror: a successful non-cached solve stores exactly once after replay
and a cache hit does not store again. Also assert a `_solve_only` run stores
nothing — the deliberate behavior change stated in §4.5.

### 5.5 Repair the nondeterministic smoke test

Keep `_width_starved_graph` as a broad solver/replay smoke test, but remove the
claim that a second independent `solve_schedule` reproduces the compile's
assignment. Parallel CP-SAT intentionally does not guarantee that.

The smoke test should assert only properties of the compile it actually ran:
successful optimized compilation, replay-depth equality, and numerical
parity. Gap-zero and collective semantics are pinned by §§5.1–5.2, where the
assignment is deterministic.

Repeated solver seeds may be used as a non-gating stress check, never as the
only regression.

### 5.6 Contract-assertion negative tests

Every fail-loud check this plan adds — the A2 whole-batch last-reader
validation, the A3 head preflight, the A4 width/held preflight, the
post-preflight must-allocate assertion, and the overdue-cancel assertion —
gets one negative test pinning its error shape, in the same file as the
§5.2 scheduler-level tests.

Status of these checks: they are directed-replay model/replay contract
assertions, not extensions of the canonical compiler invariants I1–I4.
They get no CLAUDE.md Compiler Invariants entry and no test in
`test_compiler_assertions.py`; the assertion-plus-negative-test pairing
discipline is borrowed because it is what keeps an assertion honest across
refactors. Canonizing any of them later is a separate, flagged decision
requiring both pieces of I1–I4 bookkeeping.

Each test drives `DirectedLayerScheduler` directly with a hand-built
assignment corrupted in exactly one way:

- **A2**: move a value's cancel to a layer where one uncomputed consumer is
  not in the attention batch (e.g. an MLP-routed consumer at that layer).
- **A3**: move one extra attention-routed node into the handoff layer so
  the batch's exact head charge exceeds `n_heads`.
- **A4**: remove one release from an exactly-fitting assignment so the
  ordinary width preflight fails.
- **Overdue**: assign a cancel one layer earlier than the value goes dead,
  so it is still allocated when its assigned layer has passed.
- **Must-allocate**: reachable only through a preflight bug, so no
  single-field corruption triggers it honestly; pin it with a direct unit
  call (force `_try_allocate` to return `None` past a passing preflight)
  rather than weakening one of the four checks above to reach it.

### 5.7 Entry-dead and mid-batch releases share one batch

§§5.1–5.2 pin the deadlock, but in both fixtures every released value dies
mid-batch (its last readers are in the batch). The other release class —
values already dead on entry to the layer, which today flow through the
promotion/leftover paths that §4.3 bypasses — appears in neither. A botched
bypass (entry-dead values dropped instead of batched) would pass both
fixtures and surface only as diffuse full-suite failures. Pin it with one
scheduler-level fixture in the §5.2 style:

```text
inputs:
    x[4]
    blocker[10]

W[2]   = Linear(x)                     layer 0, attention
P[2]   = Linear(W)                     layer 0, MLP (same-layer attn->mlp)
S[4]   = Linear(x)                     layer 0, attention
A[2]   = Linear(S)                     layer 1, attention
B[2]   = Linear(S)                     layer 1, attention
out[2] = Linear(Concat(A, B, P, x, blocker))   layer 1, MLP

d=24, d_head=4, d_hidden=24
```

`W`'s only consumer runs in layer 0's MLP, so its attention-mechanism
cancel pins to layer 1 and `W` is dead on entry to layer 1; `S` dies
mid-batch under its two layer-1 readers. Entry to layer 1 holds
`const + x + blocker + W + P + S = 23` of 24 columns; the batch releases
`W + S = 6` columns in one coalesced cancel (two heads; layer total four of
six heads), then `A`, `B`, and the MLP `out` all place at depth two. This
exact assignment has been checked hard-fixed feasible, and a free solve
independently returns it (both cancels at layer 1, attention mechanism).

The test asserts: one free column on layer-1 entry; `W` and `S` both in
the single coalesced cancel op; the overdue scan stays silent; depth two.
It is green once the fix lands (the old promotion path also handles `W`),
so it is a bypass regression pin, not a red incident test.

## 6. Implementation sequence

### Step 1: Land the red deterministic tests

1. Add the two fixtures from §§5.1–5.2.
2. Add hard-fixed CP feasibility checks.
3. Add exact-assignment directed replay checks and verify they fail with the
   expected residual no-progress error before production code changes.
4. Record the expected two-layer depth for each fixture.

This step establishes that the assignments are model-feasible and isolates
replay as the failing component.

### Step 2: Refactor source capture without behavior change

1. Extract `_capture_attn_sources`.
2. Use it in the current lazy placement path.
3. Run the scheduler, writer, and compiler tests before adding batch release.

### Step 3: Add directed atomic pre-release

1. Add the single batch-selection hook (equality matching plus the overdue
   scan, §4.2).
2. Capture the complete directed attention batch.
3. Preflight consumers, heads, ordinary width, and held-bank state.
4. Commit coalesced cancels and ownership releases, and bypass the legacy
   dead-list cancel paths for the sublayer (§4.3).
5. Place targets using the saved source maps.
6. Make any post-preflight allocation failure an internal assertion.
7. Add the contract-assertion negative tests (§5.6) and the entry-dead
   batch-unification pin (§5.7).

The red replay tests must now pass at their fixed depth and match the graph
reference.

### Step 4: Delete unary directed handoff machinery

Remove the retry flag, fixpoint loop, and ordinary directed
`_dying_input_to_reuse` override. Re-run the deterministic fixtures after each
deletion so no hidden dependency on the old path remains.

### Step 5: Delay cache commit

Move cache storage after successful replay and add the negative/positive cache
tests from §5.4, including the pin on `_solve_only` no longer storing (§4.5).

### Step 6: Update contracts and run the full suite

Update:

- `docs/cpsat_scheduler.md`: attention residual occupancy is realized by an
  atomic directed batch, not unary self-consumer reuse;
- `docs/tied_embeddings_plan.md` §6.4: direct held handoff participates in the
  same batch transition;
- `test_cpsat_intralayer.py` module and test docstrings: remove deterministic
  solve claims; and
- scheduler comments describing retry, cancellation deferral, and source
  capture.

Run tests through `make test`. The acceptance run is the complete suite; the
new deterministic cases are not a substitute for it.

## 7. Acceptance gates

The work is complete only if all of these are true:

- [ ] Both hard-fixed collective assignments are CP-SAT feasible before and
      after the change.
- [ ] Both assignments replay successfully after the change.
- [ ] Each emits exactly two layers; no slack layer is introduced.
- [ ] Attention use at the handoff layer equals, rather than exceeds,
      capacity, as read from the compile's per-layer head counts.
- [ ] Under directed replay, attention-mechanism cancels flow exclusively
      through the batch; the promotion and leftover-cancel paths are
      unreachable there and unchanged for the heuristic. Entry-dead and
      mid-batch releases share one coalesced batch (§5.7).
- [ ] Each contract assertion (A2, A3, A4, must-allocate, overdue) has a
      scheduler-layer negative test pinning its error shape.
- [ ] A `_solve_only` run stores no cache entry.
- [ ] Numerical outputs match the source graph at existing compiler tolerance.
- [ ] The CP-SAT model proto/objective behavior is unchanged by production
      code; no scheduler constraint is added.
- [ ] `optimize=0` layer-count goldens are unchanged.
- [ ] The existing width-starved optimized compile no longer depends on which
      equal optimum parallel CP-SAT returns.
- [ ] Held tied-output placement still uses exactly the embedding bank.
- [ ] No cache entry is written before successful replay.
- [ ] The retry flag/fixpoint and ordinary directed unary-reuse override are
      gone.
- [ ] `make test` passes, modulo separately characterized baseline failures.

## 8. Complexity budget and non-goals

The implementation may add one private source-capture helper and one protected
batch-selection hook. It should remove a comparable amount of retry and unary
handoff code. It must not add:

- a CP-SAT sublayer or ordering dimension;
- per-node ordering variables or prefix-capacity constraints;
- reserved scratch columns;
- a second directed attention scheduler;
- a new runtime op or artifact field;
- an MLP atomic handoff;
- a schedule-rejection/re-solve loop;
- a compatibility flag; or
- seed/worker pinning as a correctness mechanism.

If the implementation begins to require any of those, stop and revisit the
shared attention-batch boundary. They solve a harder problem than the machine
actually has.

One known asymmetry is deliberately left in place: MLP `cancel_bypass`
still defers silently when a layer's hidden slots run out, tolerated by the
`<=` in the directed `_find_dead_nodes`. The same contract argument that
makes A3 an assertion (the model charges the `2 * len` hidden slots at the
assigned cancel layer, so a sound assignment cannot overflow them) would
justify asserting there too, but that is a separate behavioral change with
its own red test — a flagged follow-up, not part of this plan.

## 9. Why this is the minimum solution

The CP model already describes the physical attention machine correctly: all
heads read one input state, all cancellation and compute deltas are summed, and
only the aggregate post-sublayer residual occupancy matters. The weight writer
already records source columns independently of later allocator ownership.

The extra complexity is the current replay fiction that each head must obtain
its output columns before the next head may read. Removing that fiction both
fixes the failure and deletes its incomplete retry/unary-reuse workaround. No
depth, compute, model, or artifact change is needed.
