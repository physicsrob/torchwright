# Step 1 plan: block-first ReLU machine (numerics frozen)

## STATUS: COMPLETE (2026-07-01) — closeout record

All phases (0, 1, 2a, 2b, 2c, 3) are done and verified; every gate passed.
This section is the state a fresh session needs; the phase sections below are
the original plan, kept for reference (Gate A ruling section included).

**Branches (committed, NOT pushed, umbrella pointers untouched):**
- torchwright `block-ir-2a` @ `0570af1` (2a: 84178ac..77ea7e9; 2b: 726f349;
  2c: e50eb7a; 3: e5a6481; fix: 0570af1). Net production delta
  77ea7e9..e5a6481: +343/−1003 in torchwright/.
- torchwright_doom `block-ir-2a-r8` @ `b2d1973` (branched from rope-phase8
  because doom main predates torchwright's RoPE Phase 5 and cannot build the
  flagship against torchwright main — pre-existing pointer skew, fix
  separately). Five untracked scripts in scripts/ (see cleanup below).

**Verified results:**
- Full `make test` green (10 shards) at each phase boundary and after the fix.
- Flagship metrics (e1m1, non-HUD build): (layers, heads, peak_hidden,
  residual_peak) = **(57, 781, 16384, 8192)** vs pre-refactor
  (64, 813, 16384, 8157). −7 layers / −32 heads from block-aware fusion
  firing where the old relu-ejection gate declined (~536 Gate-A candidates).
  Residual at exactly d is feasible (57 layers even with the 2-column
  RMSNorm reserve).
- Forward divergence, original chain-64 model vs final block-57 model, same
  prefill: max 1.220703e-04, mean 8.06e-05, argmax 9/9 — fp-accumulation
  floor. (Caveat: 9 fallback tokens; does not exercise HUD emission.)
- `docs/op_noise_data.json` byte-identical throughout (numerics frozen).
- Production `make compile` (e1m1, hud=1, 64-CPU Modal): completes, 61
  layers, artifact in cache volume.

**Bug found & fixed on the way (0570af1 + reproducer
`test_fusion_refreshes_stale_bounds`):** the 2c Block→Linear fold inverts
survivorship (the surviving Block absorbs the downstream Linear's transform,
changing the survivor's VALUE), so its construction-time-cached
`_affine_bound`/`_structural_type` went stale-unsound; GraphAnalyzer's
Assert-strip then tightened the structural type from the assert's claimed
range ([0,255] pixel) and the RMSNorm-certification soundness check fired
(affine −255 vs structural [0,255], `pspr/R_DrawPlayerSprites/emit`).
Fix: after folds settle, recompute bounds for every mutated survivor and all
downstream nodes in node_id (topological) order. Weights were always folded
correctly — artifacts and the divergence result were never affected.

**Trap for any future flagship verification:** the graph depends on env read
at import time (`TORCHWRIGHT_DOOM_HUD`, `_DETAIL`, `_RENDER_SCALE`, screen
dims). e1m1 production = hud=1/detail=low/scale=1/320x200, set via
`apply_screen_env` BEFORE torchwright_doom imports. Hud-off builds fuse 774
pairs / 57 layers; hud-on fuses 839 / 61 layers — verifying the wrong one
looks clean and proves nothing about the other.

**Confirmed out-of-band finding (separate thread):** cold `optimize=2`
CP-SAT at flagship scale returns UNKNOWN at its 180s budget even on the
production 64-CPU container and silently falls back to the eager heuristic
(RuntimeWarning only). Recommend recording solver status in the artifact
meta and/or failing loudly. Until then, `optimize=2` is only effective via a
warm schedule cache.

**Remaining work (decisions/cleanup, no open engineering):**
1. Merge/push both branches; bump umbrella pointers.
2. Pre-merge cleanup: `git worktree remove` the two `.divergence_baseline`
   worktrees; revert the TEMPORARY baseline-mount block in doom
   `modal_image.py` (committed in b2d1973, marked with a revert note); in
   doom scripts/ keep `find_valuetype_soundness.py` (general graph-soundness
   scanner) + `count_chain_flexibilities.py` (Phase-0 instrument) as
   committed tooling, delete `trace_block_bound.py`,
   `compare_pspr_constant_chain.py`, `baseline_soundness_scan.py` (one-off
   traces; durable value is the committed reproducer test).
3. Disposition (decided): KEEP `blockify` (verification tripwire: asserts no
   raw L→R→L chains return) and the `ReLU` node type (blockify's detector +
   affine-rule internals); ReLU is no longer op-facing, per Gate A.
4. Step 2 (SwiGLU) is unblocked: the Block node already carries the
   up-projection fields and per-block activation the gated machine needs —
   see `block_lane_spec.md` and `ops_plain_english.md`.

*Companion to `ir_semantic_vs_structural.md` (the diagnosis) and
`ops_plain_english.md` (the target SwiGLU op formulas). This is the derisking
step for the ReLU → SwiGLU migration: reshape the current ReLU machine into
the block-first shape the gated machine needs, while changing no math. Step 2
(the actual nonlinearity/gating change) is out of scope here.*

## Goal

Introduce a first-class MLP-block node — lanes + output projection, the shape
the SwiGLU machine needs — and make the scheduler consume declared blocks
instead of mining `Linear → ReLU → Linear` chains. Zero math change: every op
computes the identical function with identical weights.

**The invariant that makes this safe:** at every phase boundary,

1. the compiled artifact for any existing graph is output-equivalent within
   existing tolerances;
2. `docs/op_noise_data.json` is byte-identical — the drift test
   (`tests/docs/test_numerical_noise_drift.py`) enforces this for free, so any
   drift is a refactor bug by definition;
3. the flagship DOOM graph shows no cost regression (layers / heads / hidden
   width).

## Phase 0 — Measure what's load-bearing

The chain machinery has ~6 flexibilities; each gets ported as a block feature
or deleted, decided by data, not caution. On a real DOOM-graph compile, count
how often each fires:

| Path | Where it fires today |
|---|---|
| plain chains | `_detect_chains` hits |
| non-exclusive L1 (chain input linear with outside consumers → dual realization) | `exclusive=False` chains |
| chain splitting (L1 early / ReLU standalone / L2 standalone) | `scheduler.py:384-395` path + `compute_standalone_relu` emissions |
| MLP-bypass linears | `compute_linear_bypass` emissions |
| cross-op fusion + relu ejection | `fuse_consecutive_linears` verbose output — and first, whether the flagship build invokes it at all (fusion is caller-invoked; `torchwright_doom/scripts/analyze_forward_cost.py` defaults `run_optimize_graph=False`) |
| add_into vs compute_add | attn-op counts (context for the cost baseline) |

Mechanism: a small counting hook on scheduled-op emission (committed under
`scripts/`, run via `make modal-run` if the compile needs GPU). Deliverable: a
table; every zero-count path is deleted, not ported. **This phase gates the
scope of everything below.**

## Gate A ruling (2026-07-01, measured on production e1m1: d=8192, optimize=2 config)

Counts from the Phase-0 report (`torchwright_doom/scripts/count_chain_flexibilities.py`):

- **PORT:** plain exclusive chains (2293 — the Block itself); MLP-bypass
  linears (2422); width-safe fusion (351 pairs — made block-aware per 2c);
  `add_into` (66).
- **DELETE:** non-exclusive-L1 dual realization (0 fired); chain splitting,
  both paths (0); the relu-ejection machinery — `_ejected_relu`,
  `eject_budget`, `skip_relu_ejecting` (0 fired; all 536 ejecting candidates
  declined by the production width-safe gate). Blockify asserts L1
  exclusivity instead.
- **ReLU becomes block-internal:** removed from the op-facing IR vocabulary;
  `compute_standalone_relu` deleted with it. This matches the SwiGLU
  end-state, where the nonlinearity exists only inside a lane.
- **KEEP `compute_add`** despite its zero count: an Add whose both addends
  have later consumers has no other lowering (`add_into` needs a dead
  addend) — it's a correctness fallback, not chain machinery.
- **Assert transparency:** safe default confirmed — blockify does not see
  through Asserts; it asserts none exist on chain internals (0 today).
- Consequence for the block spec: the **export-raw-lanes attribute is not
  needed** (it existed to give splitting/ejection an explicit home; both are
  deleted).
- **Out-of-band finding, tracked separately from this refactor:** cold
  `optimize=2` CP-SAT timed out at flagship scale (13k nodes / 79k vars /
  180s) and silently fell back to the eager heuristic. Verify on the
  production compile host / warm schedule cache whether shipped artifacts are
  actually CP-SAT-scheduled.

## Phase 1 — Lane/block spec (short design doc, written against SwiGLU)

Define the block node's semantics *from the gated case first*, then
instantiate ReLU as the degenerate form:

- **Lane** = `activation(gate_in) * up_in`, where `gate_in` is a row of the
  gate projection and `up_in` is a row of the up projection **or the constant
  1**. ReLU lane = `ReLU(gate_in) · 1`. SwiGLU lane (step 2) =
  `Swish(gate_in) · up_in`. One interface, no re-design later.
- **Block node** = inputs, lanes, output projection + bias; `compute()` gives
  exact math (for the recursive oracle / `reference_eval`);
  `compute_value_type`; Assert/DebugWatch wrap the block output like any
  node. Lane internals are *not* graph nodes.
- **Required integrations for any new node type** (easy to miss; all have
  per-node-type dispatch today):
  - `graph/affine_rules.py`: a `_block_rule` for `compute_affine_bound` —
    compose the linear rule, the ReLU envelope case analysis, and the output
    projection per lane (today this happens implicitly across three nodes,
    each with its own rule).
  - Assert transparency: ops may wrap a chain-internal value (e.g. the ReLU
    output) in `Assert`. Decide: blockify sees through Asserts (predicates
    re-attach to lane outputs — requires lane values be extractable) or a
    chain with an internal Assert is simply not blockified in 2a (safe
    default; count occurrences in phase 0).
  - `graph/scheduling_hints.py`: `sequential_scope`'s entry-node walk and
    `scheduling_predecessors` must treat a Block as one node (should be
    automatic, but add a test).
  - Streaming/export: the exporters' streaming memory bound is a hard
    invariant; the Block writer must not materialize anything the chain
    writer didn't.
- **Block ≠ sublayer** — a block is a packable unit; the scheduler bins many
  blocks' lanes into one sublayer's hidden pool. This constraint goes in the
  node's docstring so the `linear_relu_linear` "one call = one MLP sublayer"
  mistake isn't rebuilt.
- Per Gate A, no export-raw-lanes / deferred-output-projection attribute:
  the flexibilities it would have hosted (chain splitting, relu ejection)
  are deleted, not ported. A Block is always realized whole.

## Phase 2 — Graph layer, via a `blockify` stepping stone

Lowest-risk landing order:

**2a.** Add the `Block` node, plus a graph-level `blockify` pass: literally
`_detect_chains_static`'s logic relocated to run **once, after any fusion,
before compile**, converting each mined L→R→L into a Block. Ops don't change
at all in 2a; the mining still exists but now runs once at graph level with an
inspectable, assertable result instead of transiently inside three schedulers.

**2b.** Make `linear_relu_linear` construct a Block natively (signature
unchanged — ops diff ≈ zero), migrate the handful of hand-built L/R/L
constructions (`map_select`'s direct Linears — phase 0 says which are
chain-shaped), and shrink `blockify` to an assertion that no unclaimed chain
shapes remain.

**2c** *(fusion IS live in production — 351 pairs, per Gate A)*: make
`fuse_consecutive_linears` block-aware — fold a Linear into a block's input
projection, fold a block's output projection into a downstream Linear. The
relu-ejection machinery (`_ejected_relu`, `eject_budget`,
`skip_relu_ejecting`) is deleted outright per Gate A: with blocks declared,
an ejecting fusion is simply not a legal rewrite (there is no standalone-ReLU
lowering to eject into), which is exactly the behavior production already
has via the width-safe gate.

## Phase 3 — Scheduler, CP-SAT, writer

The payoff phase: today the same structural inference is mirrored in **three
places** — `scheduler.py:_detect_chains` (~71 chain refs),
`cpsat_scheduler.py:_detect_chains_static` + `Chain` (~77),
`sibling_clusters.py:_is_chain_internal_relu` (~42). All three become direct
reads of declared blocks.

- `scheduler.py`: delete `_detect_chains`; `compute_relu` becomes
  `compute_block` (the emission logic barely changes — it already treats the
  chain as a unit); surviving flexibilities become block-attribute branches;
  `chain_protected` liveness logic keys on blocks.
- `cpsat_scheduler.py`: delete `Chain` / `_detect_chains_static`;
  `GraphModel` reads blocks; `is_flex` unchanged for standalone Linears.
- `sibling_clusters.py`: `ChainInfo` → block-keyed (mostly renames; it
  already treats chains as units).
- `weight_writer.py`: `_write_compute_relu` reads lanes from the Block node
  instead of three nodes.
- I4's schedule-time liveness checks move with the code; per the compiler
  invariants rule, every touched assertion keeps a matching negative test in
  `tests/compile/forward/test_compiler_assertions.py`.
- Expect the CP-SAT schedule cache to invalidate wholesale (fingerprints
  include node structure) — budget for re-solves on first compiles.

## Phase 4 — Debug-surface parity

`debug_value` / `probe_compiled` / `probe_residual` on a Block output work
like any node. Lane internals have exact parity with today (a chain ReLU
already has no residual assignment — `debug_value` returns `None`).
Checklist items: ONNX debug-sidecar canonical IDs and the rebuild fingerprint
handle the new node type; `OnnxDebugSession` round-trips a blockified graph.

## Phase 5 — Verification gates (each lands with its phase; this is the final sweep)

1. `make test` green.
2. Noise drift test: `op_noise_data.json` unchanged. No `make measure-noise`
   should be needed — needing it means math changed, which is a bug in this
   step.
3. Flagship equivalence: DOOM graph compiled before/after — outputs within
   existing tolerances **and** a compile-metrics diff (n_layers, heads/layer,
   peak hidden usage) with no regression. This crosses into
   `torchwright_doom`; `analyze_forward_cost.py` is already most of the
   instrument.
4. Any bug found on the way gets its smallest-layer reproducer test (D6).

## Phase 6 — Cleanup

Delete phase-0-condemned paths and the `blockify` shim; update the CLAUDE.md
compiler sections that describe chains / `compute_relu`; and finally fix the
`linear_relu_linear` docstring — or retire the helper's old name in favor of
the block builder.

## Risks, stated upfront

- **Scope concentration in phase 3** (~190 chain references across three
  files). The 2a stepping stone is the mitigation: blocks are definitionally
  identical to mined chains at that point, so schedules should match almost
  exactly and any divergence is immediately attributable.
- **Cost regressions from deleted flexibilities** — gated by phase-0 counts,
  and caught by gate 3 if the counts mislead.
- **Un-rehearsable residue**: this step cannot rehearse `Mul`'s
  no-standalone-lowering property or the ±-lane pairing conventions — those
  stay in step 2 no matter what. What it does buy: representation, three
  schedulers, writer, fusion, and tooling all pre-shaken on frozen numerics.

## Decisions made vs. reserved

Pre-decided (implement without asking):

- Node name: `Block` in `torchwright/graph/block.py` (rename is cheap later;
  do not bikeshed).
- Multi-input handling: a Block takes a single input node, using
  `Concatenate` for multi-input cases — same convention as `Linear` today.
- Lane interface: `activation(gate_in) * up_in` with `up_in` defaulting to
  constant 1; activation is a lane field even though step 1 only uses ReLU.
- 2a ordering: `blockify` runs after any caller-side `optimize_graph`, before
  `compile_headless` / `compile_to_onnx`.

Reserved for the user (STOP and report at these gates; do not proceed on a
guess):

- **Gate A (after phase 0):** the port-vs-delete call for each flexibility.
  Present the counts table; the user decides. Zero-count ⇒ propose delete,
  but the user confirms.
- **Gate B (after phase 1):** the lane/block spec doc itself — one page, user
  sign-off before any scheduler code changes.
- **Gate C (after 2a + flagship equivalence run):** metrics diff for the
  DOOM graph. Any cost regression (layers/heads/hidden) is a stop-and-report,
  not a tolerance to negotiate unilaterally.

## Equivalence harness (build in 2a, reuse at every gate)

A committed script (`scripts/`, run via `make modal-run` if GPU-bound) that,
for a given graph builder: compiles on both code paths (chain-mined vs
blockified — during the transition both exist, selected by a flag), runs
identical inputs, reports max output divergence and the compile-metrics
tuple (n_layers, heads/layer, peak hidden usage, residual peak). "Both paths
in one tree" is what makes each commit's equivalence checkable without
juggling worktrees.

## Suggested landing structure

Land 2a → 3 as one reviewable arc on a branch (bisectable commits,
equivalence checked at each), with 2b/2c and 6 as follow-ups. The natural
first action is the phase-0 counting script.

Execution note for an implementing agent: phases are separate work sessions,
not one continuous run — the gates above are hard stops. Doctrine D1–D8 and
the testing rules in `CLAUDE.md` apply throughout; in particular, a firing
compiler invariant (I1–I4) during this refactor is a stop-and-report, never
something to weaken to get the suite green.
