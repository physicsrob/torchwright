# SwiGLU step 2 — execution plan

Step 2 of the ReLU→SwiGLU migration: gated compiler support and the
swiglu op library. Step 1 is on main (the FFN node, the lowering
boundary B1–B3, cost_summary, and the gated affine-bound rules — swish
sandwich + McCormick lanes). The op-by-op designs live in
`docs/ops_plain_english.md` (the spec — 16 entries, every numeric claim
pinned by `tests/docs/test_swish_constants.py`). This doc records the
settled decisions, the execution order, and the parked questions.

## Settled decisions

1. **No mixed networks.** Every compiled graph is uniformly one machine
   — all-ReLU or all-swish. Enforced by a compile-time check: every FFN
   node in the graph must carry the same `activation`; the compiler
   selects the physical MLP kind from it; a mixed graph is a compile
   error, not a warning.

2. **Two op packages, machine selection by import path.**
   `ops/relu/` is today's op library, moved and frozen. `ops/swiglu/`
   is the new library, built to the spec. No mode flags anywhere — the
   import *is* the machine choice, and the uniformity check is the
   backstop against accidental mixing. This makes incremental main-line
   landing possible under the no-mixing rule: each swiglu op lands with
   tests that build pure-swish graphs while everything existing keeps
   importing relu untouched.

3. **The compiler supports both activations indefinitely.** This is
   cheap — the FFN IR already carries `activation` and optional
   up-projections, and the scheduler/allocator/invariants (I1–I4) are
   lane-count-based and activation-blind. `ops/relu` retirement is a
   separate, deferred decision; default policy: delete it when it first
   costs real refactor effort to drag along. Until then it is the
   regression baseline and the A/B diagnostic ("is this divergence the
   machine or my graph?" — flip an import).

4. **One builder, named for the compiled substrate:**

       # ops/swiglu — hardcodes activation="swish"
       swiglu_ffn(input_node, gate_proj, gate_bias,
                  output_proj, output_bias,
                  *, up_proj=None, up_bias=None, name="")

   Degenerate vs gated is expressed by `up_proj` presence — one concept,
   one name. Named `swiglu_ffn` by the `linear_relu_linear` precedent:
   the name describes the compiled structure the call becomes a piece
   of, not the per-call math — a degenerate bank still occupies gate
   *and* up rows of the physical SwiGLU sublayer (up = (0, 1)). No
   `activation` parameter exists in any op code; `ops/relu` keeps
   `linear_relu_linear`. The builder is the seam that absorbs future
   authoring changes without touching call sites (as
   `linear_relu_linear` absorbed the B1 lowering change).

5. **`scale = 100`** — the hinge-sharpening module constant, in
   `ops/const.py` next to `step_sharpness`. Self-normalizing only
   (`Swish(scale·z)/scale`); no op carries a push constant. Recorded
   cost: fp16 export is foreclosed (hidden slots saturate at ~10⁵; the
   artifact is fp32 everywhere today). See the spec preamble.

6. **No semantic-layer builder.** Ops fold `scale` into gate rows and
   `/scale` into out_proj explicitly, matching the spec's Build lines.
   A bank-authoring helper that folds automatically was considered and
   rejected: the constructions vary too much (bypass pairs, exact ±
   multiply pairs that use no scale at all, the snap-multiply gate),
   and the pinning tests + noise measurement catch a forgotten fold
   immediately.

## Phase A — compiler gated support (lands first, on main)

Tested with directly-authored FFN fixtures (the spec's constructions
serve as fixtures); no swiglu ops exist yet.

- **A0 — runtime saturation probe. DONE (2026-07-02).** Gated every
  bit-exactness claim in the spec: verify on torch-CUDA and
  onnxruntime-CUDA that fp32 `σ(z) = 1.0` exactly for `z ≥ 17`,
  `Swish(0) = 0`, and `σ(−scale) = 0.0` (CPU-pinned in
  `test_swish_constants.py`). **Verdict: the spec's claims hold
  unchanged on both deployed kernels.** Three permanent probe tests
  landed, all green on Modal A100s:
  - *torch-CUDA* — `tests/docs/test_swish_saturation_cuda.py` (skips
    without CUDA; re-verifies on every `make test`). Saturation at 17,
    compare contract points, abs integer grid, dead-branch zero: all
    bit-exact, matching torch-CPU.
  - *onnxruntime-CUDA 1.26.0* — torchwright_doom
    `tests/inference/test_ort_cuda_saturation.py`, run on the deployed
    pair its Modal image pins. Placed doom-side as planned; the
    alternative (onnxruntime-gpu on the torchwright image) turned out
    to be a swap, not an addition — the CPU and GPU ORT builds collide
    on one import path (see the `test-onnx` comment in
    `pyproject.toml`). All claims hold, matching torch.
  - *CPU onnxruntime — the one divergent kernel.* The `OnnxTokenModule`
    parity oracle reaches exact 1.0 only from **z ≥ 18** (up to ~1.8e-7
    below 1.0 on [17, 18)) and is exactly 0.0 for every z ≤ −18 (torch
    keeps denormals down to ~−103). Every claim the spec leans on still
    holds there (`σ(−100) = 0`, `Swish(0) = 0`, `Swish(100) = 100`, the
    ±50 onehot indicator — whose leak is even exactly zero); what moves
    is the fillet radius on that kernel alone, 17/scale → 18/scale, so
    exact-equality tests against the CPU-ORT oracle need hinge
    arguments ≥ 18/scale past the bend (and the piecewise grid-spacing
    audit constant reads 36/K there, not 34/K). Pinned in
    `tests/docs/test_ort_cpu_saturation.py`; spec preamble updated.
  All other Modal use in this plan is the existing plumbing: `make
  test` for suites (full-suite runs are also what surface
  `c_tol`-boundary FP flakes — `-k` filters never do), `make modal-run
  MODULE=scripts.…` for committed investigation scripts (D8), and
  nothing for `make measure-noise`, which is deliberately local CPU
  (`measure_op_noise.py` pins `torch.set_default_device("cpu")`).
- **A1 — physical module. DONE (2026-07-02).** `GatedMLPSubLayer`
  (`compiler/groups/mlp_sublayer.py`) alongside the ReLU sublayer:
  `down_proj(swish(gate_proj(x)) · up_proj(x)) + x`, HF-Llama component
  naming, swish computed exactly as `g * sigmoid(g)` (the expression the
  A0 probes pinned — not a fused silu). Machine kind threads
  `HeadlessTransformer(activation=…)` → `TransformerLayer` → sublayer
  choice; both sublayer classes carry an `activation` attribute. The
  gated trim counts *biases* toward slot usedness (a written degenerate
  lane's only up-side signature is its bias 1). Unit tests:
  `tests/compile/forward/test_gated_mlp_sublayer.py`.
- **A2 — weight writer. DONE (2026-07-02).** `compute_ffn`'s step-1
  assert became a machine-mismatch assert (node activation must equal
  the sublayer's; unreachable in a real compile once A3 selects the
  machine) plus the gated path: gate rows as today, up rows per lane
  (degenerate lanes get up-row 0, up-bias 1). Deferred biased-Linear
  folding lands in **both** hidden biases — the up matmul reads the
  biasless columns too. `compute_linear_bypass` gained the swish bypass
  pair (`Swish(scale·z)/scale − Swish(−scale·z)/scale = z`), sharpened
  by the module `scale`. The `scale = 100` constant landed early in
  `ops/const.py` (Phase B item 1 consumes it); the writer imports it
  sideways (a leaf constants module — no cycle), and
  `test_swish_constants.py` now pins the doc constant to the shipped
  one. Writer-level tests appended to `test_weight_writer.py`.
- **A3 — uniformity check. DONE (2026-07-02).** In `forward_compile`,
  next to the rope_d_rot global check: all FFNs must share one
  `activation` (mixed → `ValueError`), the machine is selected from it
  (no FFNs → relu), and relu FFNs carrying `up_proj` are rejected at
  the boundary. (The FFN *node* still allows the relu-gated combo — the
  McCormick bound rule is deliberately tested in that generality;
  rejection is the compiler's job.) Negative + end-to-end tests in
  `test_ffn_compile.py` (probe_compiled clean on pure-swish graphs).
- **A4 — ONNX export + debug. DONE (2026-07-02).** Gated emission:
  `l{i}_Wgate/bgate/Wup/bup/Wdown/bdown` inits; MLP as
  MatMul→Add→Sigmoid→Mul (the pinned swish pattern) →Mul with the up
  affine→MatMul→Add→skip; residual tensor names unchanged, so the debug
  session and every probe work as-is (confirmed activation-agnostic).
  Machine kind recorded as `"activation"` in `OnnxArtifact`, the token
  meta, and the debug sidecar — deliberately **outside** the frozen
  topology fingerprint; `OnnxDebugSession` cross-checks it explicitly
  (a same-shape relu rebuild trips it; negative test). The HF converter
  refuses swish artifacts loudly (the native module is relu-only).
  Tests: `tests/debug/test_swish_onnx_debug.py`.
- **A5 — cost model. DONE (2026-07-02).** CP-SAT and `cost_summary`
  confirmed lane/slot-based with no per-slot param math — no change
  needed; schedules are machine-blind (so a schedule-cache hit across a
  machine flip of the same topology is correct, not a bug). The three
  param-accounting spots in `compile.py` (layer capacity,
  `_count_layer_params`, trim savings) now use the machine's matrix
  count (per-slot `2d+2` → `3d+3`; capacity gains the up matrix and up
  bias).

## Phase B — `ops/swiglu`, incrementally on main

**COMPLETE (2026-07-03).** All eleven items landed on main, one commit
per item, full suite green on Modal after each batch. What a fresh
session should know beyond the per-op entries:

- **The chunk-cap question resolved as declared-minimum.**
  `min_d_hidden = 1024` in `ops/const.py`; piecewise_linear's `d_max`,
  floor_int's per-chunk boundary count, and table_lookup_2d's axis
  chunks all derive from it. Flagship geometry confirmed:
  `d_hidden = 16384` in both e1m1 configs (16x headroom).
- **The grid-spacing audit (34/K) caught two real call sites**:
  reciprocal's geometric grid (stacked fillets ~2e-2 at input_scale=1)
  and global_position_from_bos's inversion table (~10 positions of
  error at position 0). Both derive `input_scale` from the grid's
  smallest gap now; measured numbers land at relu parity (reciprocal
  byte-identical). Dense smooth-target grids do NOT self-absorb once
  fillets overlap the grid pitch.
- **The noise pipeline gained the machine axis** (`(machine, name)`
  keying, schema_version 2); relu numbers frozen with zero drift. The
  drift test's `_ATOL` is 5e-4 — the swiglu ulp floor is
  kernel-dependent (FMA vs per-product; 0.0 locally vs 5.8e-5 on
  Modal's EPYC for the same seed).
- **`_MASK_TOL = 4·swish_dip/scale` (swiglu/map_select.py)** encodes
  the in_range → broadcast_select interlock; broadcast_select carries
  no ±1 mask assert (junk-mask contract is unit-test-pinned).
- **Semantic-override widenings re-derived in actual-value terms**:
  `_compare_semantic_bound` gained `slack`, `_select_semantic_bound` /
  `_broadcast_select_semantic_bound` gained `rel_tolerance` (per-side
  δ·|hull side|), `_cond_gate_semantic_bound` gained `rel_tol`
  ((1+δ) envelope scaling). relu callers pass none of them —
  byte-identical behavior.
- **Machine-neutral seams**: swiglu modules import purely-linear ops
  (sum_nodes, concat, add_const, …), attention hardware, and pure-math
  helpers from the frozen relu package, each site commented; these
  relocate at relu retirement.
- Swiglu op tests live in `tests/ops/swiglu/` (the frozen relu
  baseline files in `tests/ops/` are untouched). `swish_dip` and
  `min_d_hidden` live in `ops/const.py`, pinned by
  `test_swish_constants.py`.

The original sequencing:

`git mv` today's ops to `ops/relu/` first (plus import shims for
existing callers), then land swiglu ops in dependency order, roughly:

1. `swiglu_ffn` builder + `scale` constant.
2. `multiply`, `square` — exact ± pairs, no scale, smallest surface.
3. `compare` (+ bool compositions), `equals_vector`.
4. `select`, `cond_gate` (+ `switch`).
5. `map_to_table`, `onehot_lookup`.
6. `abs`, `min`.
7. `piecewise_linear` (+ clamp/reciprocal/thermometer_floor_div/
   mod_const/global_position_from_bos), grid-spacing audit per site
   (`34/K`).
8. `floor_int` (+ `ceil_int`), `scalar_to_embedding`.
9. `in_range`, `broadcast_select`, `dynamic_extract`.
10. `table_lookup_2d`.
11. Remaining compositions (digit pipeline, `output_sequence`, …).

Per-op landing checklist (every commit):

- Unit tests at the smallest reproducing layer (D6); reuse the spec's
  pinned constants.
- Noise measurement TargetOp added and measured in the same commit
  (D7); findings doc updated.
- Semantic overrides where the spec requires them: compare (required —
  chord relaxation explodes on sharpened degenerate lanes),
  select/cond_gate/broadcast_select (re-derived actual-value widenings
  replacing `c_tol·M`; provisional — measure at flagship scale whether
  the structural rule made them dead weight).
- Assert slacks per the spec's checklist item 5 (dip slack on indicator
  outputs; the lookup guard drops its `offset·0.005` term).

**What `ops/swiglu` does not contain** (deletions, per the spec):
`multiply_2d`/quarter-square, `relu_add`, `multiply_integers`,
`square`'s `max_value`/`step`, the select/cond_gate/broadcast_select
offset apparatus (`per_column_offsets`, `scalar_M`, finite-range
requirement), `broadcast_select`'s `approximate` flag and two-sublayer
path, `table_lookup_2d`'s column-mask staircase and offset gate.
`ops/relu` keeps all of it, untouched, until retirement.

**Noise pipeline machine axis:** `op_noise_data.json`,
`measure_op_noise.py`, and the drift/consistency tests gain a machine
dimension (swiglu entries added as ops land). The frozen relu numbers
never drift — the drift test keeps validating them at zero maintenance.

## Phase C — examples cutover

Flip the calculator/adder/fibonacci examples to `ops/swiglu` (they
exercise the digit pipeline end to end: `onehot_lookup`,
`scalar_to_embedding`, `thermometer_floor_div`). Full suite green with
both packages. After this, `ops/relu`'s only consumers are its own
tests.

## Phase D — flagship cutover and follow-ups (handoffs)

Tracked here until torchwright_doom picks them up; the detailed
doom-side plan lands there when cutover approaches. Pointers are to
spec entries.

**Cutover mechanics:** flip the `std.py` import block; call-site
signature changes (`approximate=False` args disappear, offset-related
kwargs die); re-derive graph tolerances (`c_tol`-style budgets move
from `δ·M` to `δ·|actual value|` semantics — mostly relaxations).
The flagship configs also flip `bias=False` (no-bias emission,
`docs/no_bias_plan.md`), gated on re-measuring op noise under the
flag (D7).

**Blocking decisions at cutover:**

- `pick_by_one_hot` / `_collapse_scalar_emits`: the "byte-identical"
  emission claim weakens to "equal within ~10⁻⁷ relative" (spec:
  broadcast_select). Expected invisible under the ~10⁻³
  recovered-state noise feeding those picks — flagship must sign off.
- Mask/cond tolerance budgets: in_range's ported slack
  (`4·0.2785/scale` ≈ 0.011) exceeds the ReLU-era 0.005; every ±1
  check on in_range-fed masks re-budgets (spec: broadcast_select,
  in_range).
- Acceptance criterion: the walkthrough passes with re-derived
  tolerances. Renders are *not* expected byte-identical — the
  bit-exactness profile changed (some ops got exact, one regressed).

**Opportunities (optional, after cutover):**

- Lookup axis rebalance: swiglu table_lookup_2d costs 3 lanes/boundary
  on *both* axes, so balanced fusion minimizes lanes — the flat bank as
  512×512 is ~3.1k lanes vs ~6.5k at 2048×128, paying ~100 lanes of
  div/mod index arithmetic and ~2 sublayers of depth on the split path
  (spec: table_lookup_2d, dimension-ordering note). Per-table Pareto
  for flats / wall banks / HUD to be costed doom-side.
- Dead pre-clamp choreography: `clamp_to_slot` and the ±3072 dispatch
  clamps existed only to keep `broadcast_select`'s offset `M` sane;
  with `M` gone they lose their reason — keep or remove on their own
  merits (spec: broadcast_select, what-dies).
- `pick_by_one_hot` lane halving: a literal-zero false branch drops its
  lanes at build time (spec: broadcast_select, Build).

## Parked / open

- `ops/relu` retirement timing (default: first real refactor cost).
- ~~Chunk caps vs pool size~~ — resolved 2026-07-03 as the
  declared-minimum option: `min_d_hidden = 1024` in `ops/const.py`,
  swiglu chunk caps derive from it, flagship geometry confirmed at
  `d_hidden = 16384`.
- fp16 export: foreclosed by scale=100 (recorded, not planned around).
