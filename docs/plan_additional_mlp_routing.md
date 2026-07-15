# Additional MLP routing for Add

## Status

Proposed implementation plan. No code in this plan has been implemented yet.

Revised 2026-07-14 against the worktree-tied landing (tied embeddings / the
held output bank, atomic attention replay, and the ungated `optimize=0`
directed replay). The revision adds the tied-output constraints, rewrites
the directed-replay requirements against the atomic attention batch,
resequences the implementation steps, and descopes the same-layer
live-addend cancel loosening to the `add_live_addend_gap` ledger entry
(known optimality gap #2 in `docs/cpsat_scheduler.md`).

This document uses `add_into`, the compiler's existing name for adding into
the residual columns of a dead addend. The earlier name `add_input` is treated
as a reference to `add_into`.

## Goal

Allow every position-local `Add` node to use either attention heads or MLP
hidden slots — except the tied held target, which stays pinned to attention
(see *Tied output* under *Scope and constraints*). Keep the existing choice
between reusing a dead addend's residual columns and allocating fresh output
columns.

The resulting operation matrix is:

| Residual-column placement | Attention route | MLP route |
| --- | --- | --- |
| Reuse a dead addend's columns | `add_into` | `add_into_bypass` |
| Allocate fresh output columns | `compute_add` | `compute_add_bypass` |

The two axes are orthogonal, but placement is route-aware because the two
sublayers have different read times:

1. **Routing** chooses attention or MLP before the layer walk begins.
2. **Placement** derives fresh versus reused residual columns from the
   liveness state at the Add's assigned layer.
3. **Reuse target selection** identifies the exact input occurrence whose
   columns are reassigned. If both addends are reusable, input 0 wins. This is
   the current heuristic's deterministic tie-break and must be shared by the
   heuristic, CP-SAT, and directed replay.

The route must not change because one layer happens to be full. An Add routed
to MLP waits for a later layer if that layer lacks MLP slots; it does not spill
to attention. This matches the existing rule for standalone `Linear` nodes.

## Why this is useful

`add_into` and `compute_add` are position-local operations. They currently use
rotary delta-zero self-match attention heads only because that was the first
available linear transport mechanism. The MLP activation-bypass identity can
perform the same arithmetic:

- ReLU: `ReLU(z) - ReLU(-z) = z`.
- Swish: `Swish(z) - Swish(-z) = z`, with the compiler's existing scale
  folding.

Moving Adds to MLP can make schedules feasible when `n_heads` is deliberately
smaller than `d_model / d_head`, and can improve packing when a layer has spare
MLP slots but no spare heads.

## Scope and constraints

This plan adds atomic MLP realizations of `Add`. It does not split one Add
across layers.

For an Add of width `w`, either MLP operation uses `2 * w` hidden slots. The
positive half and negative half recover one linear output column through the
activation-bypass identity. Therefore:

```text
2 * w <= usable_hidden_slots
```

is a structural requirement for the MLP route. If it is false, the Add is
pinned to attention even when the static policy otherwise prefers MLP.

Pinning to attention is a routing fallback, not a guarantee that every
geometry is feasible. The selected attention operation must still fit the
configured `n_heads`: reused placement needs `ceil(w / d_head)` heads and the
initial conservative fresh-placement model may charge
`2 * ceil(w / d_head)`. With decoupled `n_heads`, an Add that fits neither
route is structurally unschedulable and must produce a resource diagnostic;
this change does not split it across layers.

### Tied output (held bank)

Tied compiles (token.v6) withhold the tied `Embedding`'s residual columns as
a single transient held bank until the schedulable graph output — the held
target — claims the entire bank, in order, via `allocate_at`, so one
`(vocab, d)` table serves as both embedding and unembedding. In the
canonical tied shape the held target is exactly an `Add`
(`logits = transported_embedding + correction`). `compile_to_onnx` and the
direct-HF path always compile tied, so the flagship's output Add is a held
target on every production compile.

The held-target Add is pinned to `ATTN_ADD` with fresh placement,
unconditionally — regardless of `local_in_attention` and `flex_routing`:

- The direct held handoff (the target directly consumes the tied Embedding)
  is one atomic attention event — read the old embedding value, cancel it,
  write the target into the reclaimed bank. There is intentionally no MLP
  equivalent, and CP-SAT already pins `is_attn[held_target] == 1`. The
  generalized static resolver must force a held-target Add to `ATTN_ADD`
  through the forced-classes mechanism, exactly parallel to the existing
  direct-handoff Linear forcing to `ATTN_TRANSPORT` — that forcing branch
  currently lets an `Add` target through untouched, which is safe only
  while Add is attention-only.
- The held target's `is_free` stays pinned to 0 without the addend-deadness
  biconditional (the 32983b0 fix; posting both went hard-INFEASIBLE for the
  canonical tied shape). The per-occurrence selector construction below
  must skip the held target entirely, as `_hinted_add_is_free` already does
  on the hint-validation side.
- An MLP-side bank claim for an *indirect* handoff (the source cancelled by
  an earlier consumer, the bank claimed later) is explicitly out of scope:
  no MLP-phase capture/cancel/claim executor exists, and the occupancy
  timing would need new modeling.

The held source (the tied Embedding) is never a reuse target and never a
same-layer cancel candidate on either route: its columns pass through a
physical cancel into the held state (`_is_dead_for_add` returns false for
it), and it has no MLP cancel mechanism. The graph-input reuse modeling
below must carve it out explicitly.

Other non-goals:

- Do not route a genuine `Attn` node to MLP. It communicates across positions.
- Do not change the existing MLP routing for standalone `Linear` nodes.
- Do not change the existing attention/MLP mechanism choice for dead-value
  cancellation, except where its timing constraints must understand an
  MLP-routed Add consumer.
- Do not broaden ordinary graph-input cancellation. `InputNode` and `Embedding`
  values remain attention-cancelled only and remain ineligible for same-layer
  freshly-dead cancellation. Their reuse-target path still ends ownership
  through `reassign`, not through a cancel operation.
- Do not opportunistically change an Add's route during directed replay.
- Do not add an MLP-side held-bank claim; the held-target Add is pinned to
  `ATTN_ADD` with fresh placement (see *Tied output* above).
- Do not close known optimality gap #2 (the live-addend same-layer cancel
  conservatism) as part of this feature; see *Cancellation timing* below.
- Do not add a new public compile flag unless compatibility review rejects the
  existing documented policy behavior described below.

## Current state

The compiler currently mixes the two Add decisions:

- `RESIDUAL_REUSE` means `add_into`, and is fixed to attention.
- `ATTN_COPY` means `compute_add`, and is fixed to attention.
- `RealizationTable` calls Add conditional because the layer walk chooses
  between those two forms from addend deadness.
- Only standalone `Linear` is treated as a free attention-versus-MLP routing
  choice.
- The heuristic scheduler constructs every Add as an attention operation.
- The CP-SAT model pins every Add to attention and charges either the free-Add
  or compute-Add head demand.
- The heuristic may reuse a preallocated graph input as an Add target, while
  the CP-SAT `is_free` construction currently rejects inputs because they are
  pinned and have no layer variable. The new replay invariant cannot tolerate
  that existing disagreement.
- CP-SAT records only the aggregate `is_free` result, not which input occurrence
  is the deterministic reuse target, and currently charges a phantom death
  cancel for a value whose ownership actually ends through `reassign`.
- Existing same-layer MLP consumers fold deferred biases from attention-routed
  Linears. The new Add bypass writers must join that protocol.
- Since the worktree-tied landing, `optimize=0` builds the heuristic trace
  and assignment, then emits through a `DirectedLayerScheduler` replay of
  that assignment (`clusters=None`). The base heuristic walker is a
  trace/assignment producer at every optimize level, never the final
  emitter.
- The directed replay executes each layer's attention sublayer as one
  atomic transition: capture every batch member's sources while all inputs
  are live; preflight the exact head charge (contract check A3) and the
  aggregate ordinary-column width with held-bank carve-outs (A4); release
  every value whose assigned attention-mechanism cancel equals the current
  layer in a single coalesced cancel; place outputs from the saved
  captures. Contract check A2 requires every uncomputed consumer of a
  released value to be inside the attention batch. (A2/A3/A4 are replay
  contract checks, deliberately not canonized as compiler invariants.) The
  MLP sublayer replay remains incremental; an assigned MLP op that cannot
  fit records a structural skip and the replay-depth tripwire raises at end
  of replay.
- `_mlp_cancel_defers_live_addends` is already `False` on the directed
  path; only the base heuristic keeps the conservative `True`.
- Tied compiles pin the held-target Add (`is_attn == 1`, `is_free == 0`)
  and exclude the held Embedding from reuse and MLP cancellation — see
  *Tied output* above.

This disagrees with `SchedulingPolicy`'s existing documentation, which says
that standalone Linears, `add_into`, and `compute_add` are position-local and
controlled by `local_in_attention`.

## Layer-order semantics

A transformer layer executes its attention sublayer before its MLP sublayer.
All operations within one sublayer execute in parallel. Add liveness must
respect that order.

The rules below deliberately use phase-start snapshots. This is a conservative
scheduler contract, not a hardware limitation: same-sublayer readers can all
read the common pre-sublayer residual before the boundary writes are combined.
The current walker, however, mutates residual ownership as it constructs a
phase. Relaxing this contract safely requires pre-capturing every operation's
sources and globally arbitrating cases where several same-phase Adds want the
same target. That broader change is out of scope here.

An addend is reusable by an attention-routed Add only when every other
consumer completed in an earlier layer. A same-layer attention consumer is
intentionally excluded from the attention phase-start snapshot.

An addend is reusable by an MLP-routed Add when every other consumer either:

- completed in an earlier layer; or
- is attention-routed and completes in the Add's layer.

A same-layer MLP consumer is intentionally excluded from the MLP phase-start
snapshot. A same-layer attention consumer counts as complete because the
attention phase has already been constructed before the MLP snapshot.

Reusable placement also has a deterministic target rule. For input
occurrences `E0` and `E1`:

```text
reuse_0[A] = reusable(E0, A)
reuse_1[A] = NOT reuse_0[A] AND reusable(E1, A)
is_free[A] = reuse_0[A] OR reuse_1[A]
```

The first-input priority is observable scheduler behavior when both addends
are dead, and it determines residual ownership and cancellation. It is not an
arbitrary solver decision. For `add(x, x)`, occurrence 0 is the reuse target
and occurrence 1 is the source read; both refer to the same captured columns.

For an addend `E`, another consumer `C`, and Add `A`, the CP-SAT predicate for
"`C` counts complete in `A`'s placement snapshot" is:

```text
layer[C] < layer[A]
or
(
    layer[C] == layer[A]
    and C is attention-routed
    and A is MLP-routed
)
```

`E` is dead for `A` when this predicate holds for every other effective
consumer of `E`, provided `E` owns reassignable residual columns. A
`Concatenate`, positional/reserved compiler value, or value retained by a
terminal output consumer is not reassignable. An effective consumer without a
layer variable prevents reuse because its read cannot be ordered.

The predicate deliberately has no term for `E`'s own birth layer. The
existing route-aware dependency bounds on the edge `E -> A` (a zero-layer
gap when an attention producer feeds an MLP consumer, one layer for every
other route pair) already force `E` to be materialized by the time `A`'s
placement snapshot is taken, on both routes. The assignment-level
derivation and the physical-versus-derived tripwires may therefore rely on
the layer and route maps alone.

Preallocated graph inputs (`InputNode` and `Embedding`) are an intentional
exception to the old CP-SAT shortcut: they own residual columns and the
heuristic already permits those columns to become an `add_into` target. The
input itself has no layer variable, but its schedulable consumers do, which is
enough to evaluate the predicate. CP-SAT must model input-target reassignment
rather than rejecting an addend merely because the addend is in
`gm.pinned_nodes` or lacks `layer_var[E]`. This preserves historical heuristic
placement for direct input Adds. The held source is the exception to this
exception: a tied compile's Embedding is never reusable on either route
(`_is_dead_for_add` is false for it); its columns end through the held-bank
cancel/hold transition only.

That target exception does not turn graph inputs into ordinary same-layer
cancel candidates. When a graph input is an ordinary source occurrence, retain
the existing input lifetime contract: it can use only an attention cancel and
its earliest cancel layer is the layer after the Add, on either Add route. The
directed scheduler must not surface it through the same-layer freshly-dead
paths. When it is the selected reuse target, ownership instead transfers by
`reassign`, no cancel is emitted, and its virtual lifetime ends at
`layer[A] + 1`.

The heuristic must use the same rule. In particular, it must snapshot the
computed set at the start of the MLP phase before placing any MLP operation,
not merely before the first MLP Add. The iteration order of parallel MLP
operations must not change which addends are considered dead. Any future
relaxation must change heuristic placement, assignment derivation, and CP-SAT
together, and add shared-target arbitration rather than relying on walker
iteration order.

## Deferred-bias semantics

An attention-routed standalone `Linear` is not always fully materialized at
the attention/MLP boundary. Its attention head writes `W x`; a separate
`compute_bias` operation writes `b` in the MLP sublayer. An MLP Add in the same
layer reads the pre-MLP residual and therefore does not see that bias unless
the compiler folds it into the Add's MLP input lanes, just as the existing FFN
and standalone Linear-bypass writers do.

The required rules are:

- Both MLP Add writers receive the layer's `biased_linears` set.
- For every input occurrence read through the bypass lane pair, fold a
  same-layer attention Linear's `output_bias` into the positive/negative
  hidden-slot biases with the same `in_gain` and `BiasFold` handling as
  `_write_compute_linear_bypass`. If an Add input is a `Concatenate`, walk its
  flattened leaves and offsets exactly as the existing FFN and Linear-bypass
  folds do.
- `compute_add_bypass` folds both source occurrences. Duplicate sources fold
  twice, so `add(x, x)` includes `2 * bias(x)`.
- `add_into_bypass` folds only the live/source *occurrence*. If the reused
  target occurrence is a same-layer biased Linear, its own `compute_bias`
  writes that occurrence's bias directly to the target columns in parallel and
  must be applied exactly once, not folded into the Add delta as well. The
  distinction is occurrence-based, not node-based: for biased `add(x, x)`,
  occurrence 0 is the target and receives one direct `compute_bias`, while
  occurrence 1 is the source and folds one bias into the Add delta, producing
  `2 * bias(x)` in total.
- Capture every `compute_bias` target column list at the start of the MLP
  phase, before any MLP Add can reassign a newly born Linear's columns. Do not
  call `residual_map.get_indices(linear)` after a possible reassignment.

Alternatively forbidding biased attention-Linear -> MLP-Add same-layer edges
would be correct but would contradict the general same-layer route promised by
this plan. Bias folding and early target capture are therefore part of this
feature, not follow-up work.

## Proposed representation

### Resolve route, not residual placement

Replace Add's two route-conflating realization classes with two route
families, tentatively named:

- `ATTN_ADD`
- `MLP_ADD`

`candidate_classes(Add)` returns both families. The static resolver or CP-SAT
resolves one family before the layer walk. The layer walk then chooses the
operation within that family:

```text
ATTN_ADD + reusable addend -> add_into
ATTN_ADD + no reusable addend -> compute_add
MLP_ADD  + reusable addend -> add_into_bypass
MLP_ADD  + no reusable addend -> compute_add_bypass
```

The current `Entry.conditional` representation exists only for Add. Once Add
has a resolved route family, remove that special unresolved-but-complete
state. Keep residual placement as scheduler state rather than pretending it
is an unresolved hardware route.

Resource helpers must make their required context explicit:

- Attention demand for `ATTN_ADD` depends on reused versus fresh placement.
- MLP demand for `MLP_ADD` is always `2 * len(add)`.
- Cost summaries may continue to report a range when placement is not known.

Do not encode all four cells of the operation matrix as independent free
realization choices. Only two are valid at a given liveness state, and making
all four candidates would let a resolver select an impossible reuse form
before placement is known.

### Static routing

Generalize the existing static flex resolver so it does not assume every
flexible node is a `Linear`:

1. Pin a held-target Add to attention (the tied direct-handoff contract;
   see *Tied output*). This flows through the same forced-classes mechanism
   that pins a direct-handoff Linear to `ATTN_TRANSPORT`.
2. Compute the candidate MLP slot demand.
3. Pin to attention if the complete MLP operation cannot fit in a layer.
4. Otherwise apply `policy.local_in_attention`.

Under `local_in_attention="always"`, fitting Adds remain on attention. Under
`local_in_attention="never"`, fitting Adds use MLP. `LEGACY_POLICY` therefore
preserves the existing attention-only Add behavior on the heuristic/static
path. The CP-SAT compatibility contract is stated separately below because
`flex_routing=True` intentionally overrides static policy choices.

### CP-SAT routing

Under `flex_routing=True`, a structurally fitting Add receives the same
`is_attn` decision variable as a flexible standalone Linear. Static
`local_in_attention` policy does not pin that variable. If `2 * w` exceeds
usable MLP capacity, constrain `is_attn == 1` instead of presenting an
infeasible MLP mode to the solver. A held-target Add is not flex-eligible:
its `is_attn == 1` pin stays posted under `flex_routing=True`, and its
`is_free` stays pinned to 0 without the deadness biconditional.

Under `flex_routing=False`, use the generalized static resolver and the
existing policy.

Consequently, the legacy CP-SAT configuration is
`policy=LEGACY_POLICY, cpsat_flex_routing=False`. `LEGACY_POLICY` alone is
sufficient for `optimize=0`, but it cannot promise attention-only Adds while
the caller simultaneously asks CP-SAT to choose flexible routes. Compile API
documentation and tests must state this pair explicitly.

## MLP operation emission

### `add_into_bypass`

Inputs:

- `source_cols`: the live addend's columns.
- `target_cols`: the dead addend's columns, reassigned to the Add.
- `mlp_slots`: exactly `2 * width` slots.
- `reuse_input_index`: the target occurrence, 0 or 1; the source occurrence is
  the other index. This is required even when both occurrences name the same
  node.

Emit the existing activation-bypass lane pair with `W = I`:

```text
residual target entering MLP = dead_addend
MLP delta written to target = live_addend
residual target leaving MLP = dead_addend + live_addend
```

### `compute_add_bypass`

Inputs:

- `source_cols`: the first addend's columns.
- `source_cols_b`: the second addend's columns.
- `target_cols`: fresh, zeroed output columns.
- `mlp_slots`: exactly `2 * width` slots.

Emit the activation-bypass lane pair over the concatenated source rows with:

```text
W = [I]
    [I]
```

The existing source-row coalescing must be used so `add(x, x)` sums duplicate
source columns instead of allowing a last-write-wins scatter to drop one copy.

Both implementations should call `_write_bypass_lane_pair`; do not duplicate
the ReLU/Swish scaling and placement-recording logic. The caller-specific code
around that helper must still perform the deferred-bias folding specified
above; `_write_bypass_lane_pair` intentionally handles only the value path.

## Scheduler work

### Mutable and replay operation records

Extend `_MlpOp` and `PlannedMlpOp` with:

- `add_into_bypass`
- `compute_add_bypass`
- `source_cols_b`, needed by `compute_add_bypass`
- `reuse_input_index: Optional[Literal[0, 1]]`, required for
  `add_into_bypass` and absent for every non-reuse MLP operation. The live
  source occurrence is `1 - reuse_input_index`.

Also add `reuse_input_index` to `_AttentionOp` and `PlannedAttentionOp`. It is
required for `add_into` and absent for `compute_add` and every other attention
operation. The attention writer does not need the index to emit weights, but
the physical trace and replay-plan validator need it to preserve which input
occurrence was actually selected.

The reuse index is transient physical-plan metadata, not a
`ScheduleAssignment` or schedule-cache field. The scheduler sets it before
`reassign`; the corresponding `Planned*Op.from_scheduler_op` freezes it; and
replay-plan validation requires exactly one valid index for each reuse
operation and `None` for every fresh or unrelated operation. The MLP writer
must use this occurrence index to choose the input/flattened leaves whose
deferred bias is folded. It must not try to infer the source from node identity
or from post-reassignment residual ownership: neither distinguishes the two
semantic occurrences of `add(x, x)`, and the writer does not receive a
residual map.

Extend the heuristic trace with an unambiguous physical observation for every
Add:

```text
observed_add_placement[add_id] = (is_reused, reuse_input_index)
```

Populate it from the operation actually emitted on either route. Fresh
operations record `(False, None)`; `add_into` and `add_into_bypass` record
`(True, 0|1)`. This trace-only field is not serialized into
`ScheduleAssignment`. It exists so assignment completion can compare the
physical walk against the independently derived placement before the residual
map has erased the old ownership, including for `add(x, x)`.

Treat both operations as bypass-slot consumers in parameter counts,
dominance/resource diagnostics, replay-plan validation, and placement
recording.

Build or capture the zero-slot `compute_bias` records before placing any MLP
Add. Keep `biased_linears` available to the writer so the Add writers can fold
biases for source occurrences that were born in attention in the same layer.

### Heuristic layer walk

The base walker is a trace/assignment producer at every optimize level; the
physical emission of every schedule, including `optimize=0`, happens in the
directed replay of the completed assignment. The walk's job is to make
placement decisions and record cancel layers, mechanisms, and routes that
the directed replay (and its contract checks) will accept.

Change Add handling as follows:

1. Classify ready Adds by their already-resolved route.
2. Keep attention-routed Adds on the existing writer and placement path.
3. Carry MLP-routed Adds into the MLP phase.
4. After the attention phase, add newly-ready MLP-routed Adds to the MLP
   candidates. This permits an attention result to feed an MLP Add in the same
   layer.
5. At the start of the MLP phase, snapshot computed nodes, capture all
   `compute_bias` target columns, and classify each MLP Add as fresh or reused
   with the exact `reuse_0`/`reuse_1` target selector above.
6. Pack `2 * width` slots. If they do not fit in the remaining pool, record a
   skip and defer the Add without changing its route.
7. For reusable placement, capture the live source, reassign the dead
   addend's columns to the Add, record the selected occurrence in
   `reuse_input_index`, and emit `add_into_bypass`.
8. For fresh placement, capture both sources, allocate new columns, and emit
   `compute_add_bypass`.
9. Surface inputs made dead by the placement to the existing cancellation
   machinery, subject to the same no-cancel-at-birth rule and the route-aware
   truth table below. Never surface the reused target: `reassign` ended its old
   ownership without a physical cancel operation.
10. Pass `biased_linears` to both Add bypass writers and fold deferred biases
    for the source occurrences exactly as specified above.
11. Record the actual fresh/reused form and selected target occurrence in the
    heuristic trace from the emitted physical operation.

Keep MLP Add ordering deterministic and use the existing residual-pressure
and critical-path priorities. Reuse candidates should remain higher priority
than fresh candidates because they require no residual allocation.

Live addends keep the existing conservative treatment on both routes: the
live/source occurrence of an `add_into` or `add_into_bypass` is excluded
from same-layer cancellation (the `add_into_live_addends` filters, extended
to cover `add_into_bypass`), and its earliest cancel stays at the layer
after the Add. Loosening this is known optimality gap #2 and is out of
scope (see *Cancellation timing* below). The selected target is never
surfaced through any cancel path — its old ownership ends through
`reassign`. An ordinary graph-input source never enters same-layer paths;
its gap-one rule is defined below.

### Directed replay

The directed scheduler must read Add routing from the resolved realization
table, just as it does for standalone Linears. It must reproduce the solver's
layer, route, placement predicate, residual reassignment, and cancel
mechanism without falling back to the heuristic policy.

If directed replay derives a different reusable/fresh result or a different
reuse target occurrence from the solver's derived predicates, raise an
invariant error that names the Add, its route, its layer, both per-addend
reusability results, the selected target, and the other consumers that caused
the disagreement. Do not silently emit the other form.

The atomic attention batch already delivers the read-before-cancel
guarantee for any same-layer attention cancel the model assigns: every
batch member's sources are captured before the single coalesced cancel
releases anything, so no incremental cancel insertion exists or is needed.
Contract check A2 — every uncomputed consumer of a released value must be
inside the attention batch — is the runtime enforcement of the cancellation
table below: the model's bounds must make it impossible for an MLP-routed
Add to be the final consumer of a source assigned a same-layer attention
cancel, and a corrupted assignment that violates this must raise A2 at
replay rather than mis-execute.

An MLP-routed Add replays in the incremental MLP phase; there is no atomic
MLP transition and no MLP analogue of the A3/A4 preflights. A hidden-slot
shortfall at the Add's assigned layer follows the existing assigned-MLP-op
contract: the op records a structural skip, defers, and the replay-depth
tripwire (`_check_replay_depth`) raises at end of replay naming the node,
its assigned layer, and its realized layer. No new inline MLP preflight is
required by this feature.

The reused target's canonical virtual cancel layer `layer[A] + 1`
(bookkeeping only — no operation is emitted; ownership ends through
`reassign`) is compatible with the atomic batch: by layer `A + 1` the
target's old columns have been reassigned, the old node is no longer
allocated, and the release walk — which visits allocated nodes only —
never sees it. Pin this with an explicit test rather than relying on the
walk's current shape.

Persist neither `is_free` nor the reuse-target selector in
`ScheduleAssignment` unless replay cannot derive and verify them cheaply from
the existing layer and route assignment. Derivation is preferred because it
avoids a cache-format addition and keeps one definition of addend deadness.
If persistence proves necessary, persist the target occurrence, not only
`is_free`; the latter is insufficient to replay cancellation and ownership.

Not persisting the selector does not mean discarding it unchecked. Retain the
per-occurrence reusable and selected-target literals on `BuiltModel` until
solution extraction. Before constructing `ScheduleAssignment`:

1. Extract the node layers and routes.
2. Recompute each Add's two reusable predicates and deterministic target from
   that assignment with the same assignment-level derivation replay uses.
3. Assert that the recomputed values equal `solver.Value(reusable_i)`,
   `solver.Value(reuse_i)`, and `solver.Value(is_free)` for both occurrences.

This extraction-time tripwire verifies what CP-SAT actually charged while the
literal values still exist. Directed replay then compares its physical
residual-map state with the same assignment-derived expectation. Together the
two checks can distinguish a bad CP-SAT reification from a bad replay liveness
calculation without adding a serialized selector field.

Give heuristic schedules the symmetric tripwire. Before
`ScheduleAssignment.from_heuristic_trace` canonicalizes target metadata,
derive each Add's expected placement from the completed layer/route maps and
compare it with `observed_add_placement`. A mismatch must raise an invariant
that names the Add, route, layer, observed and derived occurrence, and the
consumers that determined both reusable predicates. Do not silently replace
the physical observation with the derived value.

## CP-SAT model work

### Add routing variables

Include Add in flex-routing eligibility. Update mode-aware earliest/latest
layer bounds so an attention producer may feed an MLP-routed Add in the same
layer, while every other route pair keeps the existing one-layer dependency
gap. The held-target Add is excluded from flex eligibility: its
`is_attn == 1` pin stays posted (the single authoritative held-handoff pin
in the routing loop must cover the Add case, not only Linears).

### Route-aware reusable placement

Replace the current strict-prior-layer `is_free` construction with the
layer-order predicate specified above. Retain the two per-occurrence reusable
predicates long enough to build the deterministic selector:

```text
reuse_0[A] = reusable_0[A]
reuse_1[A] = NOT reusable_0[A] AND reusable_1[A]
is_free[A] = reuse_0[A] OR reuse_1[A]
```

Do not make `reuse_0` versus `reuse_1` a solver choice. For graph inputs,
derive `reusable_i` from their effective consumers even though the input has
no node-layer variable. Only the consumers need layer variables. Reject reuse
for a physically non-reassignable value or an unordered effective consumer,
not merely for membership in `gm.pinned_nodes`. The held source is always
rejected: it is not reassignable — its columns end through the held-bank
cancel/hold transition. The held target skips this construction entirely:
its `is_free` stays pinned to 0 without the deadness biconditional, exactly
as today.

Expose the per-occurrence maps on `BuiltModel` (for example,
`add_reusable[(add_id, i)]` and `add_reuse[(add_id, i)]`) alongside `is_free`.
They are diagnostic/extraction state only and are not copied into
`ScheduleAssignment`.

Also derive `reused_as_target[E]` from all `reuse_i[A]` literals that select
`E`. Graph structure and the deadness predicates should make these selectors
mutually exclusive for one old value; assert or constrain that invariant so a
single residual owner cannot be reassigned twice.

### Attention capacity

Gate both Add attention intervals on `is_attn[A]`:

- Reused Add: `is_attn[A] AND is_free[A]`.
- Fresh Add: `is_attn[A] AND NOT is_free[A]`.

Retain the current conservative compute-Add head charge initially. Improving
that approximation to match combined small chunks is useful but separate
work; it must not be mixed into this routing change.

### MLP capacity

Add one optional MLP interval with demand `2 * width`, present when
`NOT is_attn[A]`. The demand is the same for reused and fresh placement.
Include it in the same cumulative pool as FFN lanes, standalone Linear bypass
slots, and MLP cancellation slots.

### Residual occupancy

Preserve the existing rule that a reused Add allocates no fresh columns in
its birth layer. Validate that the current shifted interval start,
`layer[A] + is_free[A]`, remains sound for both attention- and MLP-routed
Adds. Add focused model-versus-replay tests before retaining the formula; do
not assume the old attention-only proof automatically covers MLP timing.

Model the selected target's ownership handoff explicitly:

- the old target owns its columns through the Add's layer;
- the Add owns the same columns starting at the next layer boundary;
- no death-cancel head or MLP bypass slots are charged for the old target,
  because `reassign` is the physical death mechanism;
- this rule also applies when the target is a preallocated graph input.

It is acceptable to keep a virtual lifetime-end/cancel-layer value of
`layer[A] + 1` for interval bookkeeping, but the target selector must gate the
physical cancel interval absent and make its cancel mechanism canonical or
irrelevant. Concretely, for a schedulable selected target constrain
`cancel_in_mlp[E] == 0` and `cancel_layer[E] == layer[A] + 1` under the
selecting `reuse_i[A]` literal; for a graph-input target constrain its input
cancel layer to the same value under that literal. Gate the ordinary cancel
pin/window/bump equations on `reused_as_target[E].Not()`, including forcing the
pinned model's one-layer cancel bump to zero when the target is selected. Do not
retain the current phantom cancel resource charge or MLP-cancel occupancy
extension for a reassigned target.

A selected target is an ownership handoff, not a parked value, even when its
virtual end is `max_layers`. Make the unpinned parked representation
target-aware: under `reused_as_target[E]`, force the parked-presence literal
false and gate the ordinary parked window, converse, and canonicalization
constraints off. This includes the usual implication from "not parked" to a
cancel layer inside the executable horizon, which is incompatible with a
last-layer handoff. Scope note: the parked machinery exists only in the
unpinned cancel model (`_pin_cancels=False`); the production pinned model
builds none of it. Within the unpinned model the always-on constraint to
gate is the cancel window bound posted under `parked.Not()`; the parked
converse and the explicit not-parked implication to a cancel layer at most
`max_layers - 1` are additionally gated behind the measurement-only
`_canonical_cancel_reps` knob (default off) and must become target-aware
wherever they are posted. Exclude selected schedulable and graph-input
targets from `SolveStats.parked_count` and from diagnostics that say a
value was left forever. The successor Add may remain live; the old target
did not.

### Cancellation timing

Make the Add-consumer cancel lower bounds route-aware where the new MLP
route requires it, and keep the existing conservative live-addend bound.
The physical facts:

- An attention cancel happens in the attention sublayer.
- An MLP-routed Add reads after that cancel, so its source cannot be
  attention-cancelled in the same layer.
- An MLP cancel happens after both sublayers' reads and may cancel an ordinary
  source in the same layer.
- A reused target addend must remain allocated through the Add's layer so its
  columns can be reassigned; it cannot be treated like an ordinary copied
  source.

Two separable concerns follow, and this plan takes only the first:

- **Correctness tightening (this feature).** An ordinary source of an
  MLP-routed Add cannot be attention-cancelled in the Add's layer; that
  bound must rise to `L + 1`. This slots into the pinned-cancel model's
  per-term maxima, which are already route-aware for non-Add consumers
  (`layer[c] + 1 - is_attn[c]`).
- **Optimality loosening (descoped).** Allowing the live source of a reused
  Add to be cancelled in the Add's own layer is known optimality gap #2
  (`docs/cpsat_scheduler.md`, diagnostic family `add_live_addend_gap`),
  deliberately open, with an all-or-none closure rule — model, heuristic,
  and directed replay must change together — and a bindingness-measurement
  recipe. This plan keeps the conservative `layer[A] + is_free[A]` bound
  for live addends on both routes and extends the `add_into_live_addends`
  exclusion to `add_into_bypass` live sources. Run the ledger's measurement
  before closing the gap as its own follow-up; the atomic attention batch
  has made the replay side of that closure cheap, but nothing lands until
  all three sides land together.

For an Add at layer `L`, before applying the global no-cancel-at-birth lower
bound, the truth table for ordinary schedulable values that have an
attention/MLP cancel-mechanism choice is:

| Add route | Placement / operand role | Earliest attention cancel | Earliest MLP cancel | Physical action |
| --- | --- | --- | --- | --- |
| Attention | Fresh, ordinary source | `L` | `L` | Source read in attention |
| Attention | Reused, ordinary live/source occurrence | `L + 1` | `L + 1` | Conservative (gap #2) |
| Attention | Reused target occurrence | none | none | Reassign; retain through `L` |
| MLP | Fresh, ordinary source | `L + 1` | `L` | MLP reads after attention |
| MLP | Reused, ordinary live/source occurrence | `L + 1` | `L + 1` | Conservative (gap #2) |
| MLP | Reused target occurrence | none | none | Reassign; retain through `L` |

“None” means no cancel op is emitted or charged; the old value's lifetime ends
through the reassignment handoff. For ordinary sources, combine the table with
`cancel_layer >= birth_layer + 1`. In particular, a newly born value is not
cancelled at birth even when sublayer order would otherwise permit it.
When the target and source occurrences name the same node, as in `add(x, x)`,
the selected-target override dominates: ownership transfers once and the node
has no physical cancel despite also being read through the source occurrence.

Graph inputs keep a separate, deliberately conservative contract:

| Graph-input role | Add route / placement | Earliest physical cancel | Mechanism | Lifetime action |
| --- | --- | --- | --- | --- |
| Ordinary source | Any route, fresh or reused | `L + 1` | Attention only | Physical cancel; never same-layer freshly dead |
| Selected reuse target | Either route | none | none | Reassign; virtual end `L + 1` |

Thus an ordinary graph-input consumer contributes the existing
`max(1, layer[C] + 1)` input-cancel lower bound, including when `C` is an Add.
The selected-target literal overrides that ordinary bound with the virtual
`layer[A] + 1` handoff and gates the physical cancel interval absent. Do not
derive input bounds from the schedulable-value truth table.

The lower bound for an ordinary source of Add `A` can be expressed from the
route and cancel mechanism. The reused-target override must use `reuse_0` or
`reuse_1`; `is_free` alone cannot distinguish the target from the live source.
Use the same expressions for schedulable-source lower bounds,
`_pin_cancels` equality pins, hint validation, and residual interval ends. Use
the explicit gap-one exception above for input cancel bounds and input hint
validation, with the selected-target override applied separately. The pinned
and unpinned models must describe the same earliest legal physical action.

No new heuristic cancel-surfacing machinery is needed for these bounds.
Live/source occurrences of a reuse operation stay excluded from same-layer
cancellation on both routes (the `add_into_live_addends` filters, extended
to `add_into_bypass`), the selected target is protected by the reassign
handoff, and graph inputs keep the gap-one rule. The one new same-layer
case — an MLP-routed Add's fresh ordinary source dying at `L` with an
MLP-mechanism cancel — flows through the existing mid-MLP-phase
freshly-dead surfacing that every other MLP consumer already uses.

### Assignment derivation and warm-start cancellation metadata

Add one assignment-level placement derivation that accepts graph effective
consumers plus complete `node_to_layer` and `node_to_routing` maps and returns,
for every Add, `(reusable_0, reusable_1, reuse_input_index)`. It implements the
physical-reassignability checks, sublayer-order predicate, and input-0 priority
defined above. The held target short-circuits to `(False, False, None)`,
matching its pinned `is_free == 0`; the held source is never reusable. Use
it in three places:

- CP-SAT solution extraction, to check the still-live solver literals before
  discarding them.
- Completion of a heuristic trace into `ScheduleAssignment`.
- Directed replay, to establish the expected placement before comparing it
  with the residual map.

This derivation is built in Step 2, with direct unit tests (including the
held-target and held-source short-circuits), before the replay placement
work consumes it. Coverage differs by call site: the directed-replay
comparison runs on every compile, because all emission — including
`optimize=0` — replays through the directed scheduler; the heuristic
observed-versus-derived check runs whenever a heuristic trace is completed
into an assignment; the extraction-time check needs live solver literals,
so it runs only when CP-SAT actually solves — not at `optimize=0` and not
on a schedule-cache hit. The replay-side tripwire is therefore a
production check on every compile, not a test-only one.

The heuristic tracking residual map must continue to omit a physical cancel
when `reassign` ends an old value's ownership. After the heuristic trace has
complete layer and route maps, `ScheduleAssignment.from_heuristic_trace` (or a
helper immediately before it) must run the assignment-level derivation, verify
it against the physical observation described above, and only then
canonicalize every selected target `E` of Add `A` as follows:

```text
node_to_cancel_layer[E] = layer[A] + 1
node_to_cancel_mech[E] = "attn"  # schedulable targets only; canonical, not run
```

Graph-input targets have no cancel-mechanism entry, matching the existing
assignment contract. In every case the target selector gates the physical
cancel interval absent, so the canonical mechanism is bookkeeping only. This
virtual cancel layer is nevertheless required: it makes the old target's
occupancy end exactly where the Add's shifted ownership begins and makes the
full heuristic hint feasible under both `_pin_cancels=True` and
`_pin_cancels=False`. Do not leave a reassigned target at the generic
`n_layers` default merely because `_TrackingResidualStreamMap.free` was never
called. Update the `_TrackingResidualStreamMap` docstring in the same
change: it currently says reassign-consumed nodes are "correctly omitted
from the cancel hint," which stops being true once the canonical virtual
layer is assigned.

Make hint validation selector-aware. A complete layer/route hint determines
the expected reuse target, so validate a target's virtual cancel layer against
`layer[A] + 1` and reject an MLP cancel mechanism for that target. Under the
pinned-cancel model the cancel-layer values may still be dropped before
solving, but the canonical mechanism hint remains consistent; under the
unpinned model the complete hint must be accepted without CP-SAT silently
discarding it.

### Objective and diagnostics

Update:

- `total_attn_heads` so an MLP-routed Add contributes zero heads.
- `total_mlp_bypass_slots` so an MLP-routed Add contributes `2 * width`.
- Cancel resource totals so a reused target contributes no phantom death
  cancel; its old ownership ends through `reassign`.
- Parked-value counts and diagnostics so selected targets are ownership
  handoffs, including when their virtual end equals `max_layers`.
- Cost-summary ranges and per-class diagnostics.
- Hint validation and decision-strategy inputs.

No new objective coefficient is needed: Add slots belong to the existing MLP
bypass-slot cost.

Candidate count is not route feasibility. Once Add has two candidate families,
do not continue to set `SkipReason.rerouteable` from `has_flex_choice(node)`:
an MLP candidate may already have been eliminated because `2 * width` exceeds
usable hidden capacity, and a static policy may intentionally fix a remaining
route. Replace that boolean with resolution- and geometry-aware diagnostic
metadata containing:

- the selected route and operation/placement;
- whether that route satisfied its structural eligibility check when resolved;
- every alternative route's whole-layer demand (or route-aware placement
  range), capacity, and elimination or policy-fixed reason; and
- the selected attention placement's head demand once fresh versus reused is
  known.

`SkipReason` now also carries the tied machinery's `"held output bank"`
resource string; keep the Add-routing diagnostics distinguishable from a
held-bank deadlock skip. The MLP-only held-handoff rejection (whose error
names the identity-Linear workaround) must keep firing for MLP-only target
*types*; a held-target Add is instead silently forced to `ATTN_ADD`.

Classify a structural skip as a compiler misroute only when the selected route
violates an eligibility constraint that its resolver was required to enforce,
for example an MLP Add assigned despite `2 * width > usable_hidden_slots`.
Merely having another candidate family is not enough. If MLP was eliminated and
the selected attention placement also exceeds `n_heads`, report one
geometry-unschedulable diagnostic naming both demands and capacities. If policy
fixed attention while a fitting MLP route exists, report a policy-fixed route
failure and the controlling policy setting; do not call it a compiler bug and
do not opportunistically reroute it during the walk.

## Policy and compatibility decision

The existing policy documentation says `local_in_attention` applies to Adds
and defaults to `"never"`, but the implementation currently ignores that part
for Adds. Making the implementation match the documentation changes default
heuristic schedules: every fitting Add will move to MLP unless
`LEGACY_POLICY` is selected.

`flex_routing=True` has a different and already-established meaning: CP-SAT,
not the static policy, chooses the route of eligible flexible nodes. Extending
that behavior to Add means `LEGACY_POLICY` alone cannot promise attention-only
Adds on the flexible solver path. The supported compatibility configurations
are therefore:

| Compile path | Historical attention-only Add routing |
| --- | --- |
| `optimize=0` heuristic | `policy=LEGACY_POLICY` |
| CP-SAT / directed replay | `policy=LEGACY_POLICY, cpsat_flex_routing=False` |

This uses existing public controls and does not add an Add-specific flag.

The tied held-target pin is orthogonal to policy: under every configuration
above, a held-target Add stays `ATTN_ADD` with fresh placement.

Recommended decision:

- Honor the existing documented policy instead of adding another flag.
- Measure the flip before it ships (a Step 5 gate): compile the flagship
  and representative example graphs under the historical configuration
  (`LEGACY_POLICY`, plus `cpsat_flex_routing=False` on CP-SAT paths) and
  under the new defaults, and compare layer counts, structural skips, and
  CP-SAT solve wall time. A material regression is a stop-and-discuss
  before the default lands, not a silent acceptance. If the flip is
  rejected, the fallback — keep the static default attention-preferring
  for Adds while retaining CP-SAT flex eligibility — keeps Add a policy
  exception on the static path and therefore obligates changing the
  `SchedulingPolicy` documentation in the same change; a silent
  documentation-versus-implementation mismatch is not an option.
- Document the two legacy configurations above wherever `LEGACY_POLICY` is
  advertised for Add routing.
- Update schedule-count tests that intend to test the default policy.
- Change tests that intend to pin historical placement to pass
  `LEGACY_POLICY` explicitly and, on CP-SAT paths, disable flex routing.

This compatibility decision must be called out in the implementation change
and release notes. `LEGACY_POLICY` preserves the historical attention Add
writers, deterministic first-reusable target selection, and graph-input reuse.
It does not promise byte-identical whole schedules: CP-SAT may choose
different layers from the heuristic, as it already can today. Cancellation
micro-placement, however, is unchanged under `LEGACY_POLICY`: the
conservative live-addend treatment stays (gap #2 remains open), and the new
`L + 1` attention-cancel tightening applies only to MLP-routed Adds, which
`LEGACY_POLICY` never produces.

If byte-identical whole schedules are required, stop before implementation and
introduce a separate compatibility mode that pins routing, placement, and
cancellation timing together; do not claim that `LEGACY_POLICY` alone provides
that stronger guarantee.

Schedule cache and graph fingerprints already include `flex_routing`, the
scheduling policy, a content hash of the compiler's Python sources, and the
held source/target identities. The implementation changes therefore
invalidate old attention-only Add schedules without a manual generation
bump; add a regression test for that assumption. The CP-SAT snapshot schema
is `FORMAT_VERSION = 2` (bumped for the held-bank contract); bump it to 3
only if the final design persists a new target field in
`ScheduleAssignment` or otherwise changes serialized data.

## Implementation sequence

Each step should leave its internal APIs tested, but the feature should merge
only when heuristic and CP-SAT paths agree.

### Step 1: Representation and resource declarations

- Add `ATTN_ADD` and `MLP_ADD` route families.
- Remove Add's unresolved-but-complete conditional table entry.
- Generalize static capacity-aware routing, including the held-target Add
  forcing to `ATTN_ADD` through the forced-classes mechanism. Order matters
  inside this step: `resolve_static` rejects a forced class for a
  conditional entry, so the conditional-entry removal must land before the
  forcing can flow through `forced_classes`; the compile-side change is the
  direct-handoff branch that currently lets `Attn` and `Add` targets
  through untouched.
- Retain per-route structural eligibility/elimination reasons for diagnostics;
  do not equate multiple candidates with a currently usable alternative.
- Add Add demands to cost summaries.
- Update realization-table and cost-summary unit tests, including tied
  fixtures.

Gate: realization tests prove that policy, geometry, and solver assignment
admit the same Add route set; the static resolver and `flex_routing=False`
model resolve the same route, while a flexible solve may choose either fitting
candidate.

### Step 2: MLP writers, replay records, and directed replay placement

- Build the assignment-level placement derivation (effective consumers plus
  layer/route maps in, `(reusable_0, reusable_1, reuse_input_index)` out)
  with direct unit tests, including the held-target and held-source
  short-circuits. The replay placement below and the Step 3/4 tripwires all
  consume it.
- Add the two planned MLP operation types, second source field, and required
  occurrence-level `reuse_input_index` for `add_into_bypass`.
- Add the same occurrence-level index to attention `add_into` records; require
  it on every reuse record and reject it on every fresh or unrelated record.
- Implement both writers through `_write_bypass_lane_pair`.
- Fold same-layer deferred Linear biases for every source occurrence, including
  duplicate sources, while leaving a reused target occurrence's direct
  `compute_bias` write single-counted. For biased `add(x, x)`, use the reuse
  index to apply one direct target bias and one folded source bias.
- Teach the directed replay to place both MLP Add operations in the
  incremental MLP phase (slot packing, structural-skip recording); a
  shortfall at the assigned layer surfaces through the replay-depth
  tripwire. Because `optimize=0` replays every schedule, nothing in Step 3
  ships without this.
- Update slot counting, placement recording, and replay validation.
- Test ReLU, Swish, self-add, noncontiguous source columns, biased sources,
  both reuse-index orientations, biased reused targets, and `bias=False`.

Gate: direct writer tests match `a + b`, including deferred-bias contributions,
without invoking the scheduler.

### Step 3: Heuristic scheduling (lands with Step 2's replay support)

- Split ready Adds by resolved route.
- Add MLP-phase placement and post-attention readiness.
- Implement phase-start deadness snapshots and deterministic target selection.
- Store the chosen occurrence on every attention and MLP reuse physical op,
  record every Add's observed placement in the heuristic trace, and assert
  that it matches assignment-level derivation before canonicalization.
- Capture `compute_bias` targets before any MLP reassignment.
- Extend the `add_into_live_addends` same-layer cancel exclusion to
  `add_into_bypass` live sources; never surface the selected target through
  any cancel path.
- Preserve graph-input reuse while retaining gap-one, attention-only
  cancellation for an ordinary graph-input source. Apply only the selected
  target's no-cancel/virtual-`L + 1` exception. Keep the held source fully
  excluded.
- Execute the route-aware cancellation table for schedulable sources.
- Preserve fixed-route deferral and structural fallback.
- Add heuristic schedule and end-to-end parity tests.

Gate: `optimize=0` compiles representative reused and fresh Adds with zero Add
heads under the MLP policy, while `LEGACY_POLICY` retains attention Add ops.
The gate exercises the directed replay implicitly — `optimize=0` emission
runs through it, so the observed-versus-derived tripwire and the replay
contract checks fire on every gate compile.

### Step 4: CP-SAT model and directed replay

- Add route variables and capacity pins.
- Preserve the held-target pins (`is_attn == 1`; `is_free == 0` without the
  deadness biconditional) and the `_hinted_add_is_free` special case;
  exclude the held source from every `reusable_i`.
- Add per-addend reusable/target literals; make head capacity, MLP capacity,
  residual occupancy, physical cancel presence, and cancel timing route- and
  target-aware.
- Make parked literals, their converse/canonicalization constraints, parked
  statistics, and parked diagnostics target-aware at the final horizon.
- Model reassignment of preallocated graph inputs and remove phantom cancel
  charges for selected targets.
- Keep an ordinary graph-input Add source on the input-specific gap-one cancel
  bound on both routes; do not derive it from the schedulable-source table.
- Canonicalize a heuristic target's virtual cancel to `layer[A] + 1`, with no
  physical cancel and an attention mechanism only as schedulable-target
  bookkeeping; make both pinned and unpinned hints accept that contract.
- Retain reusable/selector literals through solution extraction, recompute the
  expected selectors from extracted layers/routes, and assert equality before
  discarding the literals.
- Update warm-start hints, selector-aware validation, assignment extraction,
  snapshots, and replay.
- Add pinned/unpinned model and replay-equivalence tests.

Gate: the solver can choose either route, every returned assignment replays,
and a low-`n_heads` graph becomes feasible by routing Adds to MLP.

### Step 5: Documentation, diagnostics, and regression closeout

- Update `docs/cpsat_scheduler.md`, `docs/lowering_boundary_plan.md`, policy
  docstrings, and compile API descriptions.
- Update the gap-#2 ledger entry in `docs/cpsat_scheduler.md` to note the
  `add_into_live_addends` exclusion now also covers `add_into_bypass` live
  sources, and update `docs/tied_embeddings_plan.md` for the held-target
  Add forcing.
- Update schedule diagnostics to distinguish Add heads and Add bypass slots,
  retain route-elimination reasons, and distinguish an actual resolver misroute
  from a fixed route that fits neither the selected placement nor its eliminated
  alternative.
- Review schedule-count changes and label intentional default-policy changes.
- Run the default-flip measurement from *Policy and compatibility decision*
  (flagship and example graphs, historical versus new configuration; layer
  counts, structural skips, solve wall time) and record the results in the
  implementation change.
- Run formatting and diff checks.
- Run focused tests, then the full suite through `make test`.

Gate: no stale statement says Adds always run in attention, all focused tests
pass, the full suite passes, and old behavior remains available through
the documented legacy configurations.

## Test matrix

### Writer and replay records

- `add_into_bypass` on ReLU.
- `add_into_bypass` on Swish.
- `compute_add_bypass` on ReLU.
- `compute_add_bypass` on Swish.
- `add(x, x)` duplicate-source coalescing.
- Noncontiguous and concatenated source columns.
- `bias=False` constant-lane reservation.
- Same-layer biased attention Linear as a fresh Add source.
- Same-layer biased attention Linear as an `add_into_bypass` live source.
- Same-layer biased attention Linear as the reused target; `compute_bias`
  target capture survives reassignment and applies the bias once.
- `add_into_bypass` with only input 0 reusable and with only input 1 reusable;
  the frozen `reuse_input_index` selects the opposite live source and its
  flattened bias leaves in both orientations.
- Two biased sources and biased `add(x, x)` apply each source occurrence's bias
  exactly once. In the reused biased self-add case, one occurrence is direct
  target bias and the other is folded source bias even though both occurrences
  name the same node.
- Attention `add_into` and MLP `add_into_bypass` records require a valid
  occurrence index; fresh and unrelated records reject one.
- Replay-plan serialization and bypass-slot counts.

### Heuristic scheduler

- Reusable MLP Add emits `add_into_bypass` and reassigns columns.
- Fresh MLP Add emits `compute_add_bypass` and allocates columns.
- Unbiased and biased attention producers feed MLP Add in the same layer.
- MLP producer forces its Add consumer to a later layer.
- Same-layer MLP consumer does not make an addend reusable.
- When both addends are reusable, input 0 is the target on both routes.
- A direct graph-input Add preserves historical target reuse.
- A tied compile under the default MLP-preferring policy keeps the
  held-target Add on `ATTN_ADD` with fresh placement into the bank.
- The held Embedding is never selected as a reuse target on either route,
  even when otherwise dead.
- MLP slot exhaustion defers without spilling to attention.
- `2 * width > usable_hidden_slots` resolves to attention.
- An Add that fits neither MLP slots nor attention heads reports both structural
  resource failures.
- `LEGACY_POLICY` emits the existing attention Add operations under
  `optimize=0`.
- Live/source occurrences of `add_into` and `add_into_bypass` are excluded
  from same-layer cancellation on both routes; the selected target is never
  cancelled.
- A fresh MLP-routed Add's ordinary source assigned a same-layer MLP cancel
  flows through the existing mid-MLP freshly-dead surfacing.
- Ordinary sources surface to attention or MLP cancellation without being
  freed before their final read.
- A graph input used as an ordinary source of either attention- or MLP-routed
  Add retains its `L + 1` attention cancel and never enters same-layer
  freshly-dead cancellation; a selected graph-input target still reassigns
  with no physical cancel.
- Completing a heuristic trace assigns every reused target the virtual cancel
  layer `layer[A] + 1`, emits no physical cancel, and never leaves an earlier
  reassignment target at a generic `n_layers` default.
- Deliberately corrupt each route's observed fresh/reused form or reuse
  occurrence, including `add(x, x)`, and verify assignment completion raises
  the named physical-versus-derived invariant before returning an assignment.
- A structurally over-wide MLP candidate followed by an over-head attention
  placement reports both demands as an unschedulable geometry rather than a
  `has_flex_choice`/rerouteable compiler bug. A policy-fixed attention failure
  is identified separately and does not spill to MLP.

### CP-SAT

- Force each of the four operation-matrix cells through route and liveness
  hints.
- Verify optional attention and MLP intervals are mutually exclusive.
- Verify reused/fresh Add head demand.
- Verify MLP Add demand is exactly `2 * width`.
- Verify the sublayer-aware per-addend reusable predicates, deterministic
  `reuse_0`/`reuse_1` selector, and `is_free` projection.
- Verify preallocated graph inputs can be selected as reuse targets without an
  addend layer variable.
- Verify cancel timing for every Add route, placement, and cancel mechanism.
- Verify an ordinary graph-input source has input cancel bound `L + 1` on both
  Add routes and in fresh and non-target-reused placements, while a selected
  graph-input target has no physical cancel and a virtual end at `L + 1`.
- A corrupted assignment that gives an MLP-routed Add's source a same-layer
  attention cancel trips contract check A2 at replay.
- An assigned MLP Add facing a hidden-slot shortfall at its assigned layer
  records a structural skip and trips the replay-depth tripwire.
- The held-target Add stays `is_attn == 1`, `is_free == 0` under
  `flex_routing=True`; the extraction tripwire and `_hinted_add_is_free`
  both special-case it; the held source is rejected from every
  `reusable_i`.
- Verify a selected target has no physical cancel interval or cancel resource
  charge, has canonical non-MLP cancel metadata, has cancel bump zero, and its
  ownership interval hands off to the Add exactly at `layer[A] + 1`.
- Put a selected schedulable target and a selected graph-input target at the
  final layer: each parked literal is false, each virtual end may equal
  `max_layers`, and neither contributes to `SolveStats.parked_count` or a
  parked-value diagnostic.
- Route an attention `add_into` result immediately into an MLP
  `add_into_bypass` in the same layer. Verify the intermediate Add has a
  zero-length shifted residual interval, the original target covers the
  layer, the final Add starts at the next boundary, and directed replay
  performs both ownership handoffs without a physical cancel.
- Verify pinned and unpinned cancel models agree.
- Verify a complete heuristic hint containing an early reassignment target is
  accepted under both pinned and unpinned cancel models.
- Verify static and flexible routing capacity behavior.
- Verify `LEGACY_POLICY` plus `flex_routing=False` pins CP-SAT Adds to
  attention; verify `LEGACY_POLICY` alone does not override an explicitly
  flexible solve.
- Verify hint capture, selector-aware validation, snapshot round trip, and
  directed replay.
- At solution extraction, verify every solver `reusable_i`, `reuse_i`, and
  `is_free` value equals the selector recomputed from extracted layers/routes;
  a deliberately inconsistent test model must trip the named invariant before
  `ScheduleAssignment` is returned.
- Verify objective counters and cost summaries.

### End to end

- Compile a graph with `n_heads` too small for its attention-only Add schedule
  and show that MLP Add routing makes it feasible.
- Compare recursive-oracle, heuristic, and directed-compile outputs.
- Cover both ReLU and gated-Swish transformer machines.
- Cover ONNX and HF export through the ordinary compile entry points — those
  are always tied, so the output Add is the held target; route other Adds in
  the graph to MLP.
- Preserve historical attention Add writers, first-reusable target selection,
  and graph-input reuse under `LEGACY_POLICY` for `optimize=0`.
- Preserve attention-only Add routing on CP-SAT with
  `LEGACY_POLICY, cpsat_flex_routing=False`.

## Definition of done

The feature is complete when:

1. Every fitting Add has attention and MLP route candidates, except the tied
   held target, which stays pinned to `ATTN_ADD` with fresh placement under
   every policy and flex configuration.
2. Route resolution happens before the layer walk and never changes during
   placement.
3. Both MLP Add operations emit correct weights on ReLU and Swish machines.
4. Same-layer MLP Adds preserve deferred biases from attention-routed Linears,
   including both reuse-target orientations, occurrence-distinct biased
   `add(x, x)`, reused targets, and `bias=False`.
5. Heuristic and CP-SAT schedules use the same deliberately conservative
   phase-snapshot, addend-deadness, graph-input reuse, and deterministic
   target-selection rules; every heuristic physical observation matches the
   independent assignment-level derivation.
6. Before selector literals are discarded, every CP-SAT solution's
   `reusable_i`, `reuse_i`, and `is_free` values match the result derived from
   its extracted layers/routes; every returned assignment then replays without
   route, fresh/reused, target, or cancellation disagreement.
7. Resource accounting includes Add heads or Add MLP slots, never both, and
   charges no phantom cancel for a reassigned target. Live addends of either
   route's reuse operation keep the conservative next-layer cancellation
   contract (gap #2 stays open, its exclusion extended to
   `add_into_bypass`), and ordinary graph-input sources retain their
   gap-one, attention-only cancellation contract.
8. Low-head configurations can use spare MLP capacity for Adds.
9. An Add too wide for MLP routes to attention when that attention operation
   fits; a geometry where neither route fits fails with an explicit structural
   diagnostic that names both demands and is not mislabeled rerouteable merely
   because two candidate families exist.
10. `LEGACY_POLICY` preserves attention-only Add routing for `optimize=0`; the
    CP-SAT legacy configuration additionally sets `cpsat_flex_routing=False`.
11. Heuristic assignment completion gives every reused target the canonical
    virtual cancel layer `layer[A] + 1`; pinned and unpinned warm starts accept
    it while replay emits no physical cancel for the target. A final-layer
    target is not represented, counted, or diagnosed as parked.
12. Every attention `add_into` and MLP `add_into_bypass` physical record
    carries a validated `reuse_input_index`, and no fresh or unrelated record
    carries one.
13. An attention reuse result can feed an MLP reuse in the same layer with a
    zero-length intermediate residual interval and two verified ownership
    handoffs.
14. Tied compiles are untouched by Add routing: the held-target Add still
    lands in the bank via `allocate_at` on the attention route with fresh
    placement, and the held Embedding is never reassigned or MLP-cancelled.
15. Focused tests and the full `make test` suite pass.
