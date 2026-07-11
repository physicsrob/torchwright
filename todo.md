# Remaining work

The compile-to-Hugging-Face, Phi-3-default, unified-scheduling, and no-spool
workstreams are functionally complete. The full test suite passes. The
remaining work is architectural cleanup and integration hygiene.

## Compiler architecture

- [x] Materialize a concrete, lightweight `ReplayPlan` containing the planned
  operations, residual placements, cancellation batches, and per-layer
  dimensions.
  - Replaced the shape-only directed preflight plus emission replay with one
    allocator walk whose recorded operations are consumed by dense weight
    emission.
  - Preserve approximately one dense layer of peak memory.
  - Measure compile-time improvement after removing the duplicate lightweight
    scheduler walk.

- [x] Evaluate CP-SAT candidate dominance exactly for every `Costs`
  configuration.
  - Pure-depth dominance is already exact.
  - Resource-weighted dominance now compares immutable incumbent/candidate
    plans using the physical accounting that also derives emitted shapes.
  - Weighted cache keys include `Costs`, and the cache ratchets on realized
    objective rather than layer count alone.

- [ ] Reduce or isolate the legacy parallel-hint API.
  - Production compilation passes one complete incumbent assignment.
  - `solve_schedule` still accepts `hint_layers`, `hint_routing`,
    `hint_cancel`, and `hint_cancel_mech` for snapshots, experiments, and older
    tests.
  - Move those parameters behind an explicitly diagnostic/internal interface,
    or migrate remaining callers to serialized assignments.

## Provenance and cache semantics

- [ ] Bring exported schedule metadata up to the richer internal provenance
  model.
  - Preserve assignment origin (`heuristic` or `solver`).
  - Preserve delivery (`fresh` or `cache`).
  - Preserve solver-attempt statistics independently from the chosen
    assignment.
  - Version artifact metadata if changing the serialized schema.

- [ ] Decide whether ordinary `optimize=0` heuristic assignments should be
  persisted in the schedule cache.
  - Solver-validated heuristic incumbents can already be cached and tagged.
  - Document cache eligibility, optimize-level gating, and provenance rules.
  - Version the cache format if the policy or serialized assignment changes.

## Diagnostics and tests

- [ ] Normalize keep-forever cancellation hints so the test suite is
  warning-clean.
  - Three current warnings report keep-forever cancel hints below the solver
    horizon in pinned-cancel/symmetry tests.
  - Confirm whether completion should always rewrite those cancels to the
    active horizon or whether hint validation should recognize their sentinel
    semantics.

- [ ] Add generated small-DAG replay fuzzing.
  - Compare heuristic assignments with directed replay decisions.
  - Check node layers, routing, cancellation, concrete residual placement, and
    allocator invariants—not only final numerical output.
  - Cover deterministic seeds and width-pressure cases where the heuristic
    fails but a cold CP-SAT solve succeeds.

- [ ] Add explicit performance regression measurements.
  - Separate heuristic planning, CP-SAT, ReplayPlan construction, dense weight
    emission, and backend shard-writing time.
  - Record peak RAM and peak disk usage for representative Phi-3 compiles.
  - Assert that HF bundle compilation never creates canonical layer spool files
    or rereads emitted layer shards.

## Integration hygiene

- [ ] Perform a final diff review across the combined workstreams.
- [ ] Polish `RELEASE_NOTES.md` and user-facing documentation.
- [ ] Decide how to split the large change set into reviewable commits or pull
  requests.
- [ ] Run optional ONNX Runtime validation after the final cleanup changes.

## Current validation baseline

- Full suite: **1307 passed, 24 skipped**.
- Known warnings: **3 CP-SAT keep-forever hint warnings**.
- HF bundle emission writes final safetensors shards directly; no canonical
  temporary spool or reread pass remains.
