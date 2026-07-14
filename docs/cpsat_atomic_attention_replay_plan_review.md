# Review: atomic attention replay plan

Status: adversarial review of `docs/cpsat_atomic_attention_replay_plan.md`
(the 2026-07-13 "implementation-ready" revision), 2026-07-13.

Method: every load-bearing claim in the plan was checked against the code in
this worktree (`scheduler.py`, `cpsat_scheduler.py`, `compile.py`,
`residual_map.py`, `weight_writer.py`, `graph/optimize.py`, the referenced
tests and docs), and the two §5 fixtures were built and run against the
current compiler (lowering survival, hard-fixed CP feasibility, free-solve
depth, and current replay behavior). Line numbers below refer to the current
worktree state.

> **Post-review note (2026-07-13):** the plan was revised the same day to
> fold in findings 1–7 below. The findings are preserved here as written
> against the original revision; the "not implementation-ready as written"
> verdict applies to that original, not to the revised plan.

## Verdict

The design is correct and the plan's premise is real: I reproduced the
failure mechanism deterministically with the plan's own §5.1 fixture — the
hard-fixed assignment is CP-SAT feasible (OPTIMAL), the free solve returns
depth 2, and `forward_compile(optimize=1)` today dies with exactly the
predicted `No progress: 3 nodes remaining, 1 free columns`. The central
argument — the model prices attention as one aggregate transition while the
replay demands a per-output ordering that need not exist — matches the code
on both sides, and the "no writer change needed" claim holds.

The plan is **not implementation-ready as written**. Two interactions with
existing scheduler machinery are unspecified and the more serious one
produces a guaranteed crash if an implementer follows the plan's steps
literally (finding 1). Both are plan-text fixes, not design changes. Two
smaller gaps (missing negative tests, an unstated cache-behavior change)
should also be addressed before implementation starts.

## 1. Claims verified as correct

### 1.1 The failure mechanism (§2)

- `DirectedLayerScheduler._dying_input_to_reuse`
  (`scheduler.py:1753-1789`) requires the dying input's *only* uncomputed
  consumer to be the node being placed. With two uncomputed same-layer
  readers, neither qualifies — exactly the plan's chicken-and-egg.
- The retry loop (`scheduler.py:781-792`) breaks on a no-progress pass, so
  retrying identical candidates indeed "cannot create a first operation."
- Empirical: the §5.1 fixture reproduces this today (see §3 below). The §2
  narrative's specific numbers (Ma7/Mb7/L7, 6-of-6 heads, 7 free after) are
  arithmetically self-consistent under the current head-charge rules but
  are a snapshot of one nondeterministic solver draw; they cannot be
  re-verified post hoc. That is fine — §5.1 pins the same mechanism
  deterministically, which is the point of §5.

  Consistent with the plan's nondeterminism claim: one fresh run of the
  width-starved compile on this machine **succeeded** (the solver happened
  to return a replayable optimum). The incident test is flaky-red, not
  deterministically red — §5.5's repair of the smoke test is necessary, not
  optional. `solve_schedule` defaults to 16 workers
  (`cpsat_scheduler.py:2002-2003`), so the current test-file comment
  "solve_schedule reproduces the assignment the compile used
  (deterministic)" (`test_cpsat_intralayer.py:99`) is wrong today, as the
  plan says.

### 1.2 The residual model matches §3's inequality term for term

The residual cumulative (`cpsat_scheduler.py:1553-1622`) gives an
attention-cancelled value the interval `[layer, cancel)`, an MLP-cancelled
value `[layer, cancel + 1)` (`rend = cancel + cim`), a free Add a start
shifted by `is_free` (allocation transfer, no fresh columns), and freeable
inputs `[0, cancel)`. Evaluating the cumulative at layer `t` yields exactly

```text
sum(w(B_t)) + sum(w(M_t)) <= F_t + sum(w(R_t))
```

as the plan states. The A6 argument then goes through: after the atomic
attention transition, free width is `F_t + R_t - B_t >= M_t`, and since
`ResidualStreamMap.allocate` takes arbitrary free columns
(`residual_map.py:53-79`), sequential MLP allocation cannot fail on a sound
assignment. MLP-cancelled sources are released only at end of layer
(`scheduler.py:1059-1082`), matching the `[layer, cancel+1)` accounting.

### 1.3 The A3 head charge is exactly the model's charge (this is stronger than the plan claims)

The model pools compute heads and cancel *columns* in one cumulative with
capacity `(n_heads - reserve_heads) * d_head` column-units: compute demand
is `heads * d_head`, cancel demand is `len(node)` raw columns
(`cpsat_scheduler.py:1379-1522`). Because every compute charge is an
integer number of heads, for a single coalesced cancel batch:

```text
cancel_cols <= (n_heads - sum(compute_heads)) * d_head
    <=>  ceil(cancel_cols / d_head) <= n_heads - sum(compute_heads)
```

so the replay's rounded-up coalesced-cancel head charge is *equivalent* to
the model's column-unit charge, not merely bounded by it. Per-op compute
charges agree exactly for Linear (`linear_attn_heads` is shared by model,
scheduler, and emitter — `cpsat_scheduler.py:492-493`,
`scheduler.py:1370-1378`, `weight_writer.py:494`), for Attn
(`ceil(d_v/d_head)` both sides), and for free Adds; for compute-Adds the
replay's combined-head optimization (`scheduler.py:1356-1368`) charges *at
most* the model's `2·ceil(d_out/d_head)`.

That case list is complete, not sampled: `AttnHeadOp.op_type` is a closed
`Literal` of exactly five values — `compute_attn`, `compute_linear`,
`compute_add`, `add_into`, `cancel` (`weight_writer.py:99-105`) — which
map one-to-one onto the discussion above: the first three are the Attn /
Linear / compute-Add charge cases, `add_into` *is* the free-Add case, and
`cancel` is the coalesced term the derivation already handles. The
attention sublayer can emit nothing else (all other ops are `MLPOp`s,
their own closed `Literal`, and charge hidden slots, not heads); and on
the model side `heads_for` (`cpsat_scheduler.py:473-497`) is the single
head-charging function, with branches for exactly Attn / Linear / Add
(the Add branch covering both regimes via `is_free` gating,
`cpsat_scheduler.py:1385-1408`) and `return 0` otherwise. Net: **a
model-feasible layer can never fail the A3 preflight**, so treating an A3
failure as an internal assertion (not a deferral) is exactly right. The
plan should state this equivalence explicitly — it is the license for
deleting the deferral tolerance in finding 2.

### 1.4 One physical attention event (A5, §9)

`_write_cancel` realizes the coalesced cancel as Δ=0 self-match heads (a
head whose query/key construction makes each position attend to itself, so
it reads and rewrites a row's own values in place) with
`V = I, O = -I` over `op.target_cols`, splitting at `d_head` granularity
(`weight_writer.py:648-694`) — the emitted head count is exactly the charged
`ceil(cols/d_head)`. All heads in a layer read the pre-attention residual
and sum their deltas; `_write_compute_linear` *requires* captured
`op.source_cols` and never resolves from the residual map at write time
(`weight_writer.py:489`). So `pre - pre + delta` lands correctly when a
batch output reuses a released value's columns, with no writer change. The
capture-before-release invariant (A1) is the same one the existing unary
reuse already relies on (`scheduler.py:655-680`).

### 1.5 Structural claims about the code

- **Capture block to factor (§4.1)**: exists as described, one block
  covering `compute_linear` / `compute_attn` / `compute_add` inside
  `_try_place` (`scheduler.py:659-680`), with `_require_live` run at
  capture. The proposed helper signature matches what the block produces.
- **Machinery to delete (§4.4)**: `_retry_within_layer`
  (`scheduler.py:152`, set at `1658`), the fixpoint loop
  (`scheduler.py:781-792`), and the directed `_dying_input_to_reuse`
  override (`scheduler.py:1753-1789`) are referenced **only inside
  `scheduler.py`** — no test or other module pins them. Deletion blast
  radius is as small as the plan hopes. The base scheduler's held handoff
  (`_held_handoff_dying_source`, `scheduler.py:1420-1442`) is genuinely
  needed by `optimize=0` and survives, as §4.4 says.
- **Held machinery (A4, §5.3)**: `hold()` moves columns to a transient
  `_held` set that is neither free nor allocated; `allocate_at` claims
  exactly the complete held bank; ordinary `allocate` draws only from
  `_free` (`residual_map.py:92-154`) — so "ordinary attention outputs
  cannot claim held columns" is enforced by the allocator, and the held
  bank is correctly outside both sides of the ordinary-width calculation.
- **Cache timing (§4.5)**: `store_assignment` runs immediately after the
  solve, before replay (`compile.py:1460-1483`); a cache hit takes a
  different branch and never stores (`compile.py:1278-1302`). Replay-depth
  validation (`compile.py:1821-1822`) and held-output-layout validation
  (`compile.py:1824-1840`) exist at the proposed new commit point.
- **Test seam (§5.1)**: `monkeypatch.setattr(compile_mod, "solve_schedule",
  ...)` is an established pattern used by at least four existing test files
  (e.g. `test_replay_depth_tripwire.py:98`). The fake receives the
  *lowered* output node (`compile.py:986, 1396`), as the plan assumes.
- **Admission control is a non-issue**: `optimize>0` with
  `admission_control=True` raises at entry (`compile.py:1240-1244`), so
  `clusters` is always `None` on the directed path and `_is_admissible`
  can never defer a batch member. The plan doesn't mention admission; it
  doesn't need to.
- **What the release set can contain is fixed by the model's variable
  structure** (relevant to A2; "gap-N" throughout means the cancel sits N
  layers after its last consumer's read). The model has exactly two
  cancel-variable families: `cancel_layer`, one per schedulable node
  (`cpsat_scheduler.py:1000-1140`), and `input_cancel_layer`, one per
  freeable input (`cpsat_scheduler.py:1147-1229`); nothing else can carry
  a cancel at any layer. Within those two families: ordinary freeable
  inputs get the uniform gap-1 bound `cancel >= layer[c] + 1` for *every*
  consumer regardless of routing, and the pin uses the same form
  (`cpsat_scheduler.py:1177, 1200-1208`) — only the held source gets the
  routing-aware gap-0 form. Free-add addends (dead and live) are bounded
  `cancel >= layer[A] + is_free[A] = t + 1`
  (`cpsat_scheduler.py:1321-1332`). And a schedulable value with
  `cancel == t` has, by the routing-aware bound, every not-yet-computed
  consumer attention-routed at `t` — i.e. an attention compute candidate,
  which the plan's §4.3 step 1 defines into the batch. (The residual case
  — a node assigned to `t` that is not graph-ready at `t` — is a
  replay-depth violation the existing tripwire catches independently of
  A2.) So every ordinary input and every free-Add addend in the release
  set is already dead at layer entry, and mid-attention deaths are exactly
  the schedulable values (plus the held source) read by batch members. The
  plan's A2 wording silently depends on these bounds; see finding 5.

### 1.6 Fixture claims (§5) — verified by running them

See §3 for the runs. Summary: §5.1 survives lowering with `S/A/B/out`
intact; the hard-fixed assignment is feasible; the free solve returns depth
2; the current compile deadlocks with the predicted error. §5.2 collapses
under lowering to a single Linear, so testing it through
`DirectedLayerScheduler` directly is the right call. Both fixtures' column
and head arithmetic check out exactly (including the constant column, which
`forward_compile` allocates unconditionally, `compile.py:1140-1141`).

One correction to §5.1's explanation. Two distinct lowering mechanisms are
in play: the *fold-through-Concatenate* case of `fuse_consecutive_linears`
(a weight-algebra rewrite that absorbs a sole-consumer Linear leaf into
its consumer's weight matrix, gated to never grow the parameter count —
`graph/optimize.py:856-876`), and the *univariate collapse pass* (a
separate pass, `torchwright/compiler/collapse.py`, that replaces a whole
subgraph computed from a single input with a synthesized piecewise-linear
form). The fold-through-Concatenate case applies to the fixture
structurally — `A` and `B` are sole-consumer Linear leaves of a
Concatenate whose sole consumer is a Linear — and is declined only by the
parameter-count gate (folding `A[8→2]` into `out`'s 2×4 block costs
8·4 = 32 params against the current 8·2 + 2·4 = 24). The width-4 `x`
additionally keeps the graph out of the univariate collapse pass. Both
gates are needed; the plan names only the second. The conclusion stands
(verified empirically), but the fixture comment written in step 1 should
name the parameter-count gate, because that is the property a future fold
change would silently break.

## 2. Findings

### Finding 1 (blocking): the batch commit collides with the legacy dead-node cancel paths

§4.3 says the shared method should "add the release columns to the existing
coalesced cancel batch," call `_release_cancelled` for every release
candidate, then "run the existing candidate placement loop." The plan never
says what happens to the *existing* dead-node machinery in the same pass,
and following the steps literally crashes:

- `_schedule_layer_inner` computes `dead` before the attention sublayer
  (`scheduler.py:357`), and `_schedule_attn_sublayer` builds
  `cancel_candidates` from it (`scheduler.py:613-620`). Under the directed
  scheduler, `_find_dead_nodes` returns exactly the entry-dead values with
  `cancel <= current` — the same values the new hook returns.
- After the batch commit releases them, they are no longer allocated. The
  promotion path (the code's own label for cancelling pending dead nodes
  mid-placement to free columns for a compute op that cannot otherwise
  allocate, `scheduler.py:686-702`) and, unconditionally, the
  leftover-cancel loop (`scheduler.py:794-803`) then call
  `residual_map.get_indices(cn)` on freed nodes → `KeyError`. Where the
  lookups happen to survive, the same value would be charged and released
  twice.

The design intent is clearly that the hook's set is *exhaustive* for the
directed path (the plan's own words: it "prevents two competing definitions
of same-layer handoff from surviving"), but §4.3 must state the mechanism:
**when the hook returns a list, the legacy dead-list paths — promotion
during placement and the end-of-sublayer leftover loop — are skipped
entirely; all directed attention-mechanism cancels flow through the batch.**
The heuristic (hook returns `None`) keeps both paths unchanged. Also note
the mid-placement eager-freeing surface (`scheduler.py:753-763`) becomes
inert for directed (batch-released inputs are no longer allocated, so
`_freshly_dead_inputs` skips them), but it must remain for the heuristic
and for the directed *MLP* sublayer (`_surface_mlp_freshly_dead`,
`scheduler.py:852-869`), which still executes gap-0 MLP-mechanism cancels
(gap-0: the cancel sits in the same layer as the value's last read; gap-1:
forced one layer later).

### Finding 2 (blocking): the equality prescription conflicts with the shared `_find_dead_nodes`, and the overdue assertion has no specified site

§4.2 prescribes equality (`cancel_layer == current_layer`) and demands that
an overdue directed cancel be an assertion failure, replacing today's
deliberate `<=` re-surfacing (`scheduler.py:1719-1727`). Two problems:

1. **`_find_dead_nodes` serves two masters.** The directed override's `<=`
   feeds both the attention dead list *and* the MLP cancel candidates
   (`scheduler.py:430-438`). MLP `cancel_bypass` today defers silently when
   hidden slots run out (`scheduler.py:1069-1070`, "defer the free to a
   later layer") and relies on the `<=` to resurface. If the equality
   change lands in `_find_dead_nodes`, a deferred MLP cancel never
   resurfaces — the columns leak for the rest of the compile and the
   failure appears layers later as an unrelated allocation failure. The
   plan must scope equality to the new hook and either (a) leave
   `_find_dead_nodes`'s `<=` in place for the MLP path, or (b) also
   convert the MLP slot-exhaustion deferral into an assertion. Note (b) is
   actually justified by the same argument as A3: the model charges
   `2·len` hidden slots at the assigned cancel layer in the MLP cumulative
   (`cpsat_scheduler.py:1541-1551`), so a sound assignment cannot overflow
   it — but that is a second behavioral change the plan doesn't currently
   claim, and it needs its own red test if taken. Pick one and say so.
2. **Where does the overdue assert live?** An equality-matched hook simply
   never returns an overdue value; nothing fails. The natural site is the
   hook itself: after collecting the equality matches, scan allocated
   values for `cancel_layer < current_layer` with attention mechanism and
   raise with the layer, node, and assigned cancel. Without a named check
   site, "should be an assertion failure" implements as "silently leaks."

### Finding 3: the new assertions have no negative tests

The plan adds at least four new fail-loudly checks (A2 whole-batch
last-reader validation, A3 head preflight, A4 width/held preflight, and
the post-preflight must-allocate assertion), plus the overdue-cancel
assertion from finding 2. To be precise about their status: these are
**directed-replay model/replay-contract assertions, not extensions of the
canonical I1–I4 invariant list** — they do not belong in the CLAUDE.md
Compiler Invariants section or in
`tests/compile/forward/test_compiler_assertions.py`. But the *discipline*
that keeps I1–I4 honest ("the pair — assertion + negative test — is what
keeps the invariant honest across refactors") applies with equal force
here, and §5 currently contains no negative tests at all: every proposed
test exercises the happy path or the cache. Step 1 or step 3 should add
negative tests at the scheduler layer — alongside §5.2's
`DirectedLayerScheduler`-driven tests — that feed a deliberately corrupted
assignment (a consumer missing from the batch for A2; an over-capacity
batch for A3; an under-width release set for A4) and pin the error shape.
If the maintainer later wants any of these checks canonized as I5+, that
is a separate, flagged decision requiring both pieces of I1–I4
bookkeeping; nothing in this review presumes it.

### Finding 4: moving `store_assignment` changes `_solve_only` behavior — unstated

Today the store at `compile.py:1460` runs *before* the
`_disabled_families or _solve_only` early return at `compile.py:1493`, so a
sound `_solve_only` measurement populates the schedule cache. After §4.5
moves the store to the end of the compile, solve-only runs no longer
cache. That is probably acceptable (measurement runs arguably shouldn't
write production cache state), but it is a behavior change to a
measurement seam and the plan should state it — or explicitly keep a store
at the solve-only early return. The `_disabled_families` case already
never stores and is unaffected.

### Finding 5 (minor): A2's correctness silently depends on two model bounds — state them

A2 ("every uncomputed consumer of a released value is in the batch") is
only sound because of the model-structure facts derived in §1.5 above: the
release set is confined by construction to the model's two cancel-variable
families, and within them same-layer MLP readers push an attention cancel
to `t+1` (`cancel >= layer[c] + 1 - is_attn[c]`), free-Add addends (both
of them) are bounded to `t+1` (`cpsat_scheduler.py:1321-1332`), and
ordinary inputs get the uniform gap-1 bound (`cpsat_scheduler.py:1177`).
The plan's §3 states the first bound; it nowhere states the second and
third, and both are exactly the kind of constraint an A2 implementer needs
(an implementer who doesn't know free-Add addends are excluded by the
model will write dead code — or worse, "defensive" code that hides a model
regression). One sentence each in the plan's §3 fixes this.

### Finding 6 (minor): the A4 preflight cannot literally call `can_allocate_at`

`can_allocate_at` requires the bank to already be in the held state
(`residual_map.py:112-121`), but the preflight runs *before* the release
that holds it (§4.3 steps 5 vs 7). The preflight's held check must be its
own logic — "the held target's bank equals the held source's current
columns, and the source is in the release set (or the bank is already
held from an earlier layer)" — not a call to the existing predicate. Worth
one sentence in §4.3 so an implementer doesn't chase a false-negative.

### Finding 7 (minor): acceptance-gate measurability

Gate "attention use at the handoff layer equals, rather than exceeds,
capacity" needs a measurement surface; `per_layer_head_counts` /
`_count_heads_by_type` (`compile.py:1776`) provides it. The §5.1 test
should assert against that rather than re-deriving head counts.

## 3. Empirical verification record

Run against this worktree (CPU, cache disabled):

**§5.1 fixture** (`d=24, d_head=4, d_hidden=24`):

- Lowering survival: lowered graph is `x[4], blocker[10], S[8], A[2],
  B[2], Concat[18], out[4]` — no fold fired. ✓
- Hard-fixed assignment (S@0 attn; A,B@1 attn; out@1 MLP; cancel S@1,
  mech attn): **OPTIMAL**. ✓
- Free `solve_schedule`: **n_layers=2, OPTIMAL** (that draw routed S via
  MLP at layer 0 and cancelled it at 1 — the collective handoff is not
  merely admitted, it is what the optimum uses). ✓
- `forward_compile(optimize=1)` today: **RuntimeError: "No progress: 3
  nodes remaining, 1 free columns."** — the plan's predicted red, with the
  predicted one-free-column signature. ✓

**§5.2 fixture**: lowering collapses it to `x, y, blocker, Concat[14],
out[2]` — `Sx/Sy/C` all folded away (the through-Concatenate fold's
parameter gate passes here), confirming that the replay half must be tested
at the scheduler layer. ✓

**Width-starved incident fixture** (`d=48, d_head=8, optimize=1`): compiled
successfully on this run (status FEASIBLE) — the failure depends on which
equal-depth optimum the 16-worker solve returns, exactly the
nondeterminism §5.5 exists to remove. ✓

## 4. Smaller observations

- §4.2's "ordinary keep-forever values and the constant column remain
  ineligible" is enforced structurally: keep-forever values carry
  `cancel == max_layers` (never equal to a real layer), and the compile's
  `const_one` is minted after lowering, appears in no assignment, and is
  priced out of the model via `available_residual = d - 1 - reserve`
  (`cpsat_scheduler.py:843-844`). True as stated; no action.
- §4.3's requirement that the placement loop "consumes the saved source
  dictionaries rather than looking up released inputs again" means I4's
  schedule-time check **relocates with the capture site — it is not
  bypassed**. Today `_require_live` runs at the moment `_try_place`
  captures sources; under the batch design that same check runs inside the
  §4.1 helper, once per batch member, while every input is still allocated
  and before any release commits (that is exactly A1, and exactly I4's
  content: sources are captured while live). The placement loop then
  consumes the saved capture and must not re-run the liveness lookup —
  after a legitimate release, a placement-time re-check would be a *wrong*
  duplicate, not a safeguard. The plan should state this as a relocation
  so nobody "helpfully" re-adds the check at placement time; `_require_live`
  itself is unchanged, so the existing I4 negative test in
  `test_compiler_assertions.py` still pins the invariant.
- §4.5's claim "keep the assignment and its existing metadata in local
  variables" is workable: everything the store needs
  (`net.cpsat_solve_stats`, `schedule_fp`, geometry) is still in scope at
  the end of `forward_compile`.
- Step 6's doc targets are accurate: `docs/cpsat_scheduler.md` (~line 431)
  does attribute the exactness of the residual model to directed
  self-consumer reuse, and `docs/tied_embeddings_plan.md` §6.4 is the held
  input-cancellation section the batch transition generalizes.
- The `hint` capture path (`_run_heuristic_warm_start`) uses the base
  scheduler and is untouched by a `None`-returning hook — consistent with
  the "optimize=0 unchanged" gate, and the warm-start's recorded cancel
  mechanisms flow through `_release_cancelled` exactly as before.

## 5. Conclusion

Design: sound, minimal, and — on the A3 head-charge question — provably
exact against the current model, which is stronger than the plan claims for
itself. Premise, fixtures, and "red today" state: verified by execution.
Writer- and cache-related claims: verified against the code.

Before implementation, the plan text needs:

1. Finding 1 — an explicit statement that the hook's release set supersedes
   the legacy dead-list cancel paths (promotion + leftover loop) whenever
   the hook returns a list, with the heuristic untouched.
2. Finding 2 — scope the equality change to the hook, decide the MLP-side
   deferral question explicitly, and name the overdue-assert site.
3. Finding 3 — add negative tests for A2/A3/A4/must-allocate/overdue to §5.
4. Finding 4 — state (or avoid) the `_solve_only` cache-behavior change.

Findings 5–7 are one-sentence plan-text additions. With those, the plan
merits its "implementation-ready" status.
