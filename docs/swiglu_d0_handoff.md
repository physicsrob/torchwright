# Flagship cutover — torchwright-side punch list (doom Phase D0)

torchwright_doom is about to execute its SwiGLU cutover
(`torchwright_doom/swiglu_cutover_plan.md`, drafted 2026-07-04): the
DOOM graph flips from the ReLU op machine to the SwiGLU machine, then
the compiled artifact flips to `bias=False`. A **machine** here means
which of the two mirrored op libraries a graph is built from —
`torchwright.ops.relu` or `torchwright.ops.swiglu`; the import path is
the choice, and a compiled graph must be uniformly one machine.

This document is the complete torchwright-side worklist. When every
item lands, nothing in the doom cutover waits on torchwright, and the
doom side can treat torchwright as frozen (see the freeze section).

Drafted 2026-07-04 against HEAD `b12bac3` (working tree carrying one
uncommitted edit to `docs/swiglu_step2_plan.md` — see Task 4). Every
file/line claim below was verified against that tree, not copied from
older docs.

## Verified already done — no action needed

Listed so nobody re-does or re-audits them:

- **The swiglu machine is complete** (step-2 Phases A–C). The three
  ops the doom rewrites are authored against exist at HEAD:
  `multiply` (`ops/swiglu/arithmetic_ops.py:118`), `broadcast_select`
  (`ops/swiglu/map_select.py:243`), and `swiglu_ffn` with the exact
  signature the doom plan quotes
  (`ops/swiglu/swiglu_ffn.py:9-19`: `(input_node, gate_proj,
  gate_bias, output_proj, output_bias, *, up_proj=None, up_bias=None,
  name="")`).
- **The shared constants doom will import are pinned in
  `ops/const.py`**: `scale = 128.0` (:29), `swish_dip = 0.2784645`
  (:48), `bias_lane_gate = 32.0` (:61), `bias_lane_up = 0.03125`
  (:62). Doom's kernel-pin update (their D1) imports these by name.
- **Both torchwright-side saturation probe files already track
  scale=128**: `tests/docs/test_ort_cpu_saturation.py` pins
  `SCALE = 128.0` (:45); `tests/docs/test_swish_saturation_cuda.py`
  imports `SCALE` from `test_swish_constants.py`, which reads
  `ops/const.py`. (The one stale saturation file is doom's ORT-CUDA
  probe — doom-side work, their D1.)
- **The torch-CPU lane-constant pin exists**:
  `test_bias_lane_constants_exact_unit_lane`
  (`tests/docs/test_swish_constants.py:404`).

## Task 1 — move `attend_most_recent_globally` out of `ops/relu/`

**Why.** Doom's `past.py` imports `attend_most_recent_globally` from
`ops/relu/global_recency.py`. After the machine flip, doom must import
zero relu modules. The op is stranded in `relu/` only because it
shares a module with `global_position_from_bos` — its full body
(`relu/global_recency.py:185-382`) builds an `Attn` /
`rotary_content_head` head from graph-core and machine-neutral pieces
only. It is attention hardware, not a machine op.

**What.** Move `attend_most_recent_globally` and `_RECENCY_SCALE = 1.0`
(`relu/global_recency.py:73` — consumed only as the default of the
function's `recency_scale` kwarg, :193) to `ops/attention_ops.py`,
the machine-neutral home established by the 2026-07-03 relocation.
`global_position_from_bos` (:81) stays behind, along with the
module-level `piecewise_linear` import (:61) that only it consumes.
The other module-level imports (`torchwright.graph.rope` helpers,
`ops/_math` helpers) split by which function uses them — let the
moved body's needs decide, and the linter/tests confirm.

**Callers to update** (verified exhaustive by grepping all `*.py`):

- `tests/ops/test_global_recency.py:27` (import)
- `tests/compile/forward/test_rope_global_recency.py:23` (import)
- `torchwright/graph/rope.py:286` — a docstring `:func:` cross-reference
  spelling the old `ops.relu.global_recency` path
- doom's `past.py` — **doom-side, updated in their flip commit; do
  not touch it from here**

**Constraints.** No aliases, no re-exports, matching the 2026-07-03
relocation convention. No `__init__.py` edits are needed: nothing
re-exports `attend_most_recent_globally` today (`swiglu/__init__.py:33`
re-exports only `global_position_from_bos`).

**Why it's safe.** The op has no entry in `docs/op_noise_data.json`
(it is not a measured piecewise op), so the move is numerically inert
and the frozen relu baseline is untouched.

**Gate.** `make test`.

## Task 2 — resolve the `bias=False` noise obligation (decision: Rob)

`docs/numerical_noise_findings.md:27-29` records:

> Before the flagship flips `bias=False`, re-run `make measure-noise`
> under the flag and re-derive any budget that assumed the
> biased-machine numbers (D7).

**That knob does not exist, and cannot be added trivially**: the
harness (`measure_op_isolated`, `torchwright/debug/noise.py`)
evaluates its "compiled" leg via `node.compute()` in-process
(`noise.py:137`) and never invokes the compiler, while the bias fold
is an export-time transform in `export.py`. The flag is structurally
invisible to the measurement pipeline as built. Two resolutions;
Rob picks:

**Option A (recommended) — amend the obligation.** Rewrite the
paragraph at `numerical_noise_findings.md:27-29` to record:

- the knob is structurally absent (the harness never compiles);
- the fold's end-to-end cost is already measured
  (`tests/debug/test_no_bias_onnx.py`: logits move ≤ ~4e-4 absolute
  at ~700 magnitude, worst ~8e-5 relative on small cancelling logits
  — same error class as `bias=True`, shifted accumulation order);
- doom holds no per-op budgets that folded per-op numbers would
  re-derive (verified during the doom plan's drafting inventory:
  doom passes no `c_tol` or assert tolerances explicitly);
- the operative gate is doom's D3 on the real `bias=False` artifact:
  `debug=True` asserts, the `probe_compiled` oracle, and two scored
  walkthrough renders.

Keep a fallback trigger in the amended text: if a doom D3 gate
surfaces a divergence the end-to-end bound doesn't explain, doom
stops (foundation rule) and Option B becomes mandatory before their
D3 proceeds.

**Option B (heavyweight) — honor the letter.** Extend the harness to
optionally measure through `compile_headless(..., bias=False)`. Under
this option the extension must land and run **before** doom's D3, and
their D3 gate list grows a review of the folded per-op numbers plus
re-derivation of any budget those numbers move.

Under Option A this task is a documentation edit and nothing else.

## Task 3 — lane-constant pins on the two remaining torchwright kernels

`docs/no_bias_plan.md:268-273` wants the no-bias lane constants pinned
**on every deployed kernel** — σ(32) saturating to exactly 1.0, and
the full lane expression `32 · σ(32) · (1/32)` computing exactly 1.0
in fp32 (this is the arithmetic every folded bias rides on in a
`bias=False` artifact). Ownership: torchwright holds torch-CPU,
torch-CUDA, and ORT-CPU; the ORT-CUDA member lives doom-side (their
D1 adds it — don't duplicate it here).

Current coverage is torch-CPU only. Add the same two assertions,
mirroring `test_bias_lane_constants_exact_unit_lane` and importing
`bias_lane_gate` / `bias_lane_up` from `ops/const.py`, to:

- `tests/docs/test_swish_saturation_cuda.py` (torch-CUDA kernel)
- `tests/docs/test_ort_cpu_saturation.py` (ORT-CPU kernel; note this
  kernel's late-saturation quirk — exact 1.0 from input 18, not 17 —
  doesn't matter at input 32, comfortably past either bend)

## Task 4 — land, push, stabilize

- **Commit the dirty `docs/swiglu_step2_plan.md` hunk.** It is the
  Phase D handoff note pointing at the doom plan (verified — nothing
  else is mixed into the diff).
- **Push.** `main` is ahead 3 of `origin/main` before this punch list
  starts. Doom's compile cache keys on both repos' HEAD SHAs plus
  working-tree digests, and the doom cutover wants to start from a
  pushed, clean torchwright tree so its cache keys are stable and
  reproducible.
- **Bump the umbrella pointer** (torchdoom tracks a pinned submodule
  SHA) after the push.
- **Final gate:** `make test` green on the finished tree.

## Freeze until the doom cutover lands

The doom plan's migration inventory — which of its imports flip
untouched versus need rewrites — was verified mechanically (ast-level
signature diffs) at `bb3af2e` for the 18 ops doom imports:

`compare, clamp, piecewise_linear, mod_const, thermometer_floor_div,
floor_int, ceil_int, in_range, select, switch, dynamic_extract,
table_lookup_2d, cond_gate, bool_all_true, bool_any_true, bool_not,
global_position_from_bos, broadcast_select`

Between this punch list landing and doom's D2+D3 landing, treat as
frozen:

- public signatures of those 18 swiglu ops, plus `multiply` and
  `swiglu_ffn`;
- the `ops/const.py` values (`scale`, `swish_dip`, `bias_lane_gate`,
  `bias_lane_up`, `min_d_hidden`);
- `broadcast_select`'s documented mask contract (fractional masks
  blend within the hull of zero and the branch ranges; zero must lie
  in the branch-range union);
- `multiply`'s measured-noise docstring footer (2.24e-7 relative) —
  doom's per-call-site sizing arguments cite it.

If any of these must change anyway, flag it to the doom side before
their D2 so the inventory gets re-verified rather than silently
invalidated.

## Done when

- [ ] `attend_most_recent_globally` + `_RECENCY_SCALE` live in
      `ops/attention_ops.py`; both torchwright test files and the
      `rope.py:286` docstring reference updated; no aliases left.
- [ ] D0b resolved: Option A's amendment landed in
      `numerical_noise_findings.md`, **or** Option B's harness
      extension landed and run.
- [ ] σ(32) lane pins present in the torch-CUDA and ORT-CPU probe
      files.
- [ ] `swiglu_step2_plan.md` handoff note committed; everything
      pushed; umbrella pointer bumped; `make test` green.
