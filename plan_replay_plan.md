# ReplayPlan follow-up plan

**Status:** implemented; validation status is recorded in the completion notes
below. Large-fixture performance measurement remains tracked separately in
`todo.md`.

## Goal

Replace the current two lightweight directed scheduler walks with one concrete,
backend-neutral `ReplayPlan` that is constructed before dense weight emission
and then consumed by every emitter.

The change should:

- preserve the existing `ScheduleAssignment` as the semantic scheduling type;
- record each considered assignment's concrete allocator and placement
  decisions exactly once;
- announce exact post-trim layer dimensions before layer 0 is allocated;
- keep dense emission streaming at approximately one layer of peak memory;
- preserve bitwise artifact parity on deterministic fixtures;
- enable exact comparison of resource-weighted CP-SAT objectives.

It should not change graph lowering, the default pure-depth scheduling policy,
CP-SAT feasibility, HF architecture selection, or runtime model semantics.
For non-default weighted objectives, replacing modeled candidate dominance with
realized physical dominance is an intentional correctness change.

Internal backwards compatibility is explicitly not a goal. Prefer the smallest
coherent final design over adapters, deprecated aliases, dual callback
protocols, or preserving scheduler/writer signatures. Artifact and runtime
semantics remain compatibility boundaries; compiler-private APIs do not.

## Current state

The compiler currently has three distinct stages conceptually:

```text
ScheduleAssignment
    -> shape-only directed preflight on cloned allocator state
    -> exact LayerShape tuple announced to streaming sinks
    -> directed replay on production allocator state
    -> dense weight emission
```

This already solves the HF spool problem. Each canonical in-memory layer is
transformed directly into its final shard, and no layer file is reread.

The remaining duplication is two deterministic allocator/scheduler walks over
the same assignment. The first retains only `LayerShape`; the second immediately
turns operations into dense weights.

## Non-goals

- Do not redesign `ScheduleAssignment` or merge concrete placement into it.
- Do not change the default pure-depth CP-SAT objective.
- Do not redesign HF configuration or safetensors sharding.
- Do not persist `ReplayPlan` in the schedule cache.
- Do not retain dense tensors in the plan.
- Do not make node objects or mutable allocator structures part of serialized
  artifact metadata.
- Do not combine this work with legacy hint API removal, cache-policy changes,
  or exported-provenance schema changes.
- Do not add compatibility wrappers for the old preflight, mutable operation
  records, allocator-aware writers, or scheduling callbacks.

## Core types

Introduce backend-neutral immutable planning records. Exact names may follow
existing conventions, but responsibilities should remain separated.

```python
@dataclass(frozen=True)
class PlannedLayer:
    attention_ops: tuple[PlannedAttentionOp, ...]
    mlp_ops: tuple[PlannedMlpOp, ...]
    biased_linear_ids: frozenset[int]
    shape: LayerShape
    residual_snapshot: ResidualSnapshot
    newly_computed_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReplayPlan:
    assignment: ScheduleAssignment
    layers: tuple[PlannedLayer, ...]
    input_indices: tuple[tuple[int, tuple[int, ...]], ...]
    final_indices: tuple[tuple[int, tuple[int, ...]], ...]
    nodes_by_id: tuple[tuple[int, Node], ...]
```

Use dedicated frozen planned-operation records with tuples for every column and
slot collection. The scheduler may use temporary mutable structures internally,
but `build_replay_plan` converts them at the boundary and does not expose them.
Do not retain two public operation families merely for compatibility: once the
new boundary is established, make writers consume only planned operations and
remove obsolete mutable records where possible.

`ResidualSnapshot` should contain only immutable node-ID-to-column mappings
needed for `ResidualAssignment` and diagnostics. It must not retain a mutable
`ResidualStreamMap` per layer. Represent it as sorted tuples, not a frozen
dataclass containing a mutable dictionary.

`nodes_by_id` is an immutable, ephemeral compiler-private resolver needed by weight
writing, callbacks, `ResidualAssignment`, placement diagnostics, and final
copy-to-source re-keying. It may retain node references because the plan is
never serialized or cached. Compiler-created nodes such as the constant lane
remain compiler-keyed when they have no source counterpart.

## Locked invariants

1. `ScheduleAssignment` describes semantic decisions; `ReplayPlan` describes
   their deterministic physical realization.
2. Each assignment considered for emission is concretely planned at most once.
   Dense emission reuses the selected plan and never replays it. Weighted
   dominance may require one incumbent plan and one candidate plan before
   selection.
3. Dense weight emission does not call `schedule_layer`, allocate, free, or
   reassign residual columns.
4. Every planned operation owns immutable source/target column and slot data.
5. `PlannedLayer.shape` is derived from that layer's concrete operations, not
   from a separate assignment-level estimator.
6. The emitted trimmed dimensions must equal the announced `LayerShape`.
7. Replay planning never allocates dense transformer weights.
8. A plan contains no backend-specific tensor names, padding, fusion, or shard
   decisions.
9. Solver-only and deliberately relaxed diagnostic modes do not build a plan.
10. The plan is ephemeral and is not stored in the schedule cache.
11. No plan field contains a tensor, NumPy array, dense layer,
    `ResidualStreamMap`, or backend sink object.
12. Compiler-private API compatibility must not add indirection or duplicate
    paths to the final design.

## Design questions to resolve first

### Operation and collection immutability

Introduce `PlannedAttentionOp` and `PlannedMlpOp`. All source columns, target
columns, and MLP slots are tuples. All ID-to-column collections in the plan are
sorted tuples or another structurally immutable representation. Construction
must defensively copy scheduler-owned collections.

### Node references

Planned operations identify nodes by stable `node_id`; `ReplayPlan.nodes_by_id`
resolves IDs to compiler-private nodes when weights or callbacks need them.
This keeps snapshots and comparisons structural while preserving simple access
to graph weights during ephemeral emission.

### ResidualAssignment construction

The current compiler records per-layer residual snapshots while emitting.
Move snapshot capture into planning, then associate each immutable snapshot
with the corresponding emitted layer state when building `ResidualAssignment`.
Capture the snapshot after `schedule_layer` and synthetic `Concatenate`
completion. It represents the post-MLP residual state even though dense writes
have not happened yet. Define the zero-layer placeholder behavior explicitly:
it has no planned operations, but still supplies valid input/output states.

### Verification callbacks

`on_node_scheduled` should fire during planning, because that is when scheduling
actually occurs. `on_layer_compiled` remains an emission callback after dense
weights are written. Replace callback signatures directly if a plan-oriented
protocol is simpler; do not maintain the old protocol in parallel.

Compiler verification should distinguish:

- planning/allocator invariants; and
- emission/shape/weight-write invariants.

## Implementation phases

### Phase 0 — focused baseline

- Capture current layer counts, post-trim shapes, placement sidecars, and
  canonical weights on a small deterministic fixture set before refactoring.
- The mandatory fast set includes:
  - a small ReLU custom model;
  - a small SwiGLU Phi-3 model;
  - a width-pressure CP-SAT graph;
  - a graph using free Add, copied Add, cancellation heads, MLP cancellation,
    attention nodes, standalone Linears, literals, and FFNs.
- Reserve calculator_simple, biasless/RMSNorm integration, compile-time, and
  peak-RSS measurements for the pre-merge suite rather than blocking the first
  structural edit.

Gate: reproducible parity artifacts exist for the fast deterministic set.

### Phase 1 — immutable planned-operation records

- Define dedicated immutable planned-operation records.
- Add conversion helpers from scheduler operations.
- Add unambiguous physical resource accounting directly on planned operations:
  - actual emitted attention heads after packing;
  - trimmed hidden width (`highest occupied slot + 1`);
  - count of MLP bypass slots for the weighted objective;
  - cancellation mechanism;
  - residual source and target columns.
- Replace `_planned_layer_shape`'s independent branching with
  `LayerShape.from_planned_ops(...)`.
- Change `write_attn_sublayer` and `write_mlp_sublayer` to consume only planned
  operations, the dense destination layer, placement recording, and immutable
  global emission constants such as `const_one_col`. Remove
  `ResidualStreamMap` from writer signatures.

Gate: every attention/MLP mechanism has targeted shape-accounting tests, and
planned dimensions equal dimensions obtained after actual trimming. Mutating
all scheduler-originated lists after conversion cannot change the plan.

### Phase 2 — build `ReplayPlan`

- Extract a `build_replay_plan(...)` function from the current shape-preflight
  loop.
- Run `DirectedLayerScheduler` against cloned initial allocator/computed state.
- Store immutable planned operations, biased-linear IDs, newly-computed IDs,
  and residual snapshots per layer.
- Store input and final output column mappings plus the ephemeral node resolver.
- Run replay-depth, assignment-coverage, liveness, and allocator invariants at
  the planning boundary.
- Announce `tuple(layer.shape for layer in replay_plan.layers)` through
  `CompileHeader`.

Gate: the plan completely describes every layer needed by emission and
diagnostics; no emitter or writer needs allocator state.

### Phase 3 — emit from the plan

- Replace the production `scheduler.schedule_layer(...)` loop with iteration
  over `ReplayPlan.layers`.
- Feed planned operations directly into `write_attn_sublayer` and
  `write_mlp_sublayer`.
- Build debug placement records and `ResidualAssignment` from plan snapshots.
- Preserve streaming ownership transfer, simplifying the callback protocol if
  useful.
- Delete the shape-only preflight and second directed scheduler construction.
- Delete obsolete mutable operation types, adapters, and callback branches once
  their consumers have moved.

Gate: production dense emission never invokes scheduling or allocator mutation.

### Phase 4 — exact weighted-objective dominance

- Define one authoritative objective evaluator and lexicographic-scaling helper
  shared by replay-plan comparison, CP-SAT configuration, tests, and diagnostic
  reporting. Do not duplicate equivalent-looking formulas.
- Evaluate both incumbent and solver-candidate plans under the complete `Costs`
  configuration, including:
  - layer count;
  - attention heads;
  - MLP bypass slots;
  - earliness;
  - residual occupancy/waste;
  - the same lexicographic scaling used by CP-SAT.
- Apply the canonical assignment tie-break only after exact objective equality.
- Retain rejected candidate objective/plan diagnostics without emitting it.
- Avoid planning a candidate twice: selection should return the already-built
  winning plan.

The selection flow is:

```text
build incumbent plan if an incumbent exists and comparison needs it
build solver-candidate plan
compare realized objectives
discard the loser
emit the winner without replanning
```

The realized plan objective is authoritative for production dominance even if
the solver's modeled resource estimate differs. Document this explicitly as a
candidate-selection correction, while leaving the default pure-depth objective
unchanged.

Gate: non-default weighted objectives have independent dominance tests, and a
worse feasible solver candidate can never replace its incumbent.

### Phase 5 — cleanup and measurements

- Remove `_planned_layer_shape` and obsolete preflight comments/helpers.
- Remove superseded internal APIs instead of deprecating them.
- Update compiler architecture documentation and `todo.md`.
- Record before/after planning and total compile times.
- Measure at least three warm runs of the representative integration fixtures.
  Record median planning/total time and peak RSS. Require no all-layer dense
  accumulation, no canonical spool, and no plan-owned tensor/array objects;
  investigate any peak-RSS regression above 10% or one dense layer, whichever
  is larger.

Gate: no duplicate allocator walk remains, documentation matches production,
and performance does not regress.

## Test matrix

### Assignment and planning

- Heuristic assignment produces a complete plan.
- Solver assignment produces a complete plan.
- Cache-loaded assignment produces the same plan as the original assignment.
- Solver fallback produces the same plan as its exact incumbent.
- Cold CP-SAT solve works when heuristic planning fails.

### Operation coverage

- Real attention with multi-head V/O splitting.
- Sparse and zero-support standalone Linear transport.
- MLP-bypass Linear.
- Free Add / residual reuse.
- Copied Add, including combined-input head packing and self-add.
- Attention cancellation heads.
- MLP cancellation bypass.
- Literal and bias writes.
- ReLU FFN and SwiGLU FFN.
- Biasless constant lane.

### Artifact parity

- Exact canonical layer arrays on deterministic fixtures.
- Exact final Phi-3 state dict and shard index.
- Exact custom-model state dict.
- Exact ONNX initializers and per-layer shape metadata.
- Numerical parity for in-process, Hugging Face generate, and ONNX Runtime.

### Diagnostics

- Replay-depth mismatch fails before dense emission.
- Planned/emitted shape mismatch fails at the first layer.
- Allocator liveness and write-coverage violations name the layer and node.
- Node scheduling callbacks fire once and at assigned layers.
- Plan construction rejects tensors, mutable column collections, allocator
  objects, and missing node-ID resolutions.
- Trivial graphs follow the documented zero-layer placeholder behavior.

### Performance

- Median planning plus emission time is compared with the former shape-only
  preflight plus replay over at least three warm runs.
- No all-layer dense accumulation.
- No canonical disk spool or emitted-shard reread.

## Rollout strategy

Land this as a focused compiler follow-up after the current HF/scheduling work.
Keep the first pull request limited to Phases 0–3 if exact weighted objectives
are not immediately needed. Phase 4 can follow independently once the physical
plan is stable and tested.

Recommended commit sequence:

1. Baselines and immutable planned-operation records.
2. `ReplayPlan` construction with temporary test-only structural comparison
   against the existing replay: operation kinds, node IDs, every source/target
   column, slots, biased-linear IDs, newly-computed IDs, and snapshots.
3. Dense emission switched to consume `ReplayPlan`; immediately remove the
   duplicate walk, old writer boundary, and temporary comparison path.
4. Exact weighted-objective dominance.
5. Documentation, performance results, and cleanup.

The comparison in commit 2 is a short-lived verification tool, not a supported
dual implementation or compatibility layer.

## Completion criteria

- Each assignment considered for emission is planned at most once; the selected
  plan is emitted without another allocator/scheduler walk.
- Every dense emitter consumes the resulting immutable `ReplayPlan`.
- Exact layer shapes are available before layer 0 without a second replay.
- HF final shards remain direct-written with no canonical spool.
- Default and weighted CP-SAT candidate dominance are exact.
- Deterministic artifacts remain unchanged unless an intentional difference is
  documented and approved.
- Full tests pass, including ONNX Runtime validation whenever that dependency is
  available.
- No backwards-compatibility adapters or obsolete internal paths remain.

## Completion notes

- The forward compiler suite passes: **495 passed, 9 skipped**. The three
  existing keep-forever CP-SAT hint warnings remain tracked in `todo.md`.
- Focused HF direct-emission/parity coverage passes: **8 passed, 1 skipped**.
- ReplayPlan-specific tests cover defensive immutability, physical shape/cost
  accounting, writer type enforcement, one directed walk, zero-layer graphs,
  exact weighted dominance, and weighted cache ratcheting.
- Follow-up review made the zero-layer runtime placeholder a planned layer, so
  streaming sinks receive its `(1 head, 1 hidden slot)` shape before allocation.
  It also added in-process planned/emitted shape assertions, including
  post-compaction removal of zero-output attention heads.
- Weighted cache ratcheting now compares scale-independent `(primary,
  secondary)` objective blocks; the scaled total and scale remain metadata for
  diagnostics only. ReplayPlan construction now deeply freezes column
  collections and rejects malformed operations, unresolved node IDs, tensors,
  and duplicate resolver entries.
- A three-run warm micro-fixture measured a 5.8 ms median compile in this
  environment. This is a smoke measurement, not the representative
  before/after performance study still tracked in `todo.md`.
- ONNX Runtime validation was not run because `onnxruntime` is not installed in
  this environment.
