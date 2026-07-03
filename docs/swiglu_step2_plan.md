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

- **A0 — runtime saturation probe.** Gates every bit-exactness claim in
  the spec: verify on torch-CUDA and onnxruntime-CUDA that fp32
  `σ(z) = 1.0` exactly for `z ≥ 17`, `Swish(0) = 0`, and
  `σ(−scale) = 0.0` (CPU-pinned in `test_swish_constants.py`). If a
  deployed kernel misses by an ulp, budgets survive but the spec's
  "bit-exact" claims and any exact-equality tests must be softened
  before ops land.
- **A1 — physical module.** A gated MLP component alongside
  `linear1→ReLU→linear2`: gate matmul, up matmul, Swish, elementwise
  mul, down matmul. Per-network kind, chosen from the graph's uniform
  activation.
- **A2 — weight writer.** `compute_ffn` drops its step-1
  degenerate-ReLU assert and gains the gated path: gate rows as today,
  up rows written per lane (degenerate lanes get up-row 0, up-bias 1).
  `compute_linear_bypass` gets the swish bypass pair
  (`Swish(scale·z)/scale − Swish(−scale·z)/scale = z`, exact; sharpened
  by convention — 100× tighter sandwich slack for free, see the spec's
  `min` entry).
- **A3 — uniformity check.** The compile-time all-FFNs-one-activation
  check, with a negative test. (Not a numbered invariant — it guards a
  policy, not allocator soundness.)
- **A4 — ONNX export + debug.** The gated emission pattern; machine
  kind in `OnnxArtifact` metadata and the debug sidecar;
  `OnnxDebugSession`/probes verified activation-agnostic (residual
  capture doesn't inspect the MLP internals, but confirm).
- **A5 — cost model.** CP-SAT / cost_summary already count lanes;
  confirm nothing assumes 2 matrices per MLP (per-lane weights go
  ~2d → ~3d; layer capacity formula in `compile.py` gains the up
  matrix).

## Phase B — `ops/swiglu`, incrementally on main

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
- Chunk caps vs pool size: `_LOOKUP_D_MAX = 1024` and floor_int's
  512-boundary chunks require `d_hidden ≥ 1024` (an FFN must fit one
  sublayer's pool). Confirm flagship geometry before porting the
  lookups, or derive the caps from a declared minimum `d_hidden`.
- fp16 export: foreclosed by scale=100 (recorded, not planned around).
