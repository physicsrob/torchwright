# No-bias emission — execution plan

Standard SwiGLU transformers (the HF-Llama reference the gated MLP
sublayer is named after) carry no bias parameters anywhere. Today every
compiled torchwright transformer carries biases on the MLP sublayer's
projections (attention is already bias-free). This plan makes bias
emission a compiler option: when disabled, the compiler folds every
bias into the weight matrices against a residual column pinned to the
constant 1.0.

Settled in discussion 2026-07-03; adversarially reviewed (Codex) the
same day and revised — the review's real findings (placement-recorder
gap, `const_one` terminology, trim ambiguity, fingerprint
compatibility, kernel-complete pins) are folded in below. Status:
**N0–N3 landed 2026-07-03** (three commits: N0 seed unification /
token.v5, N1+N2 folds + constant lane + scheduling, N3 emission +
debug + metadata), full Modal suite green after each.  What remains is
the follow-ups section: the doom-side ORT-CUDA lane pin and the noise
re-measure land at the flagship's Phase D `bias=False` flip; the
Llama conversion is unscheduled.  Measured biased/no-bias parity and
the D7 obligation are recorded in `numerical_noise_findings.md`.

## Settled decisions

1. **One independent flag: `bias: bool = True`** on `forward_compile`,
   threaded through `compile_headless` and `compile_to_onnx`. Fully
   orthogonal to the machine choice (ReLU/swish) — either machine can
   compile with or without biases; the compiler never infers the flag
   from the activation. `True` is today's behavior, byte-for-byte.
   `False` emits no bias parameters anywhere. Naming follows the
   `nn.Linear(bias=False)` convention. Call sites choose: relu-era
   graphs keep the default; the swiglu flagship passes `bias=False`
   at the Phase D cutover.

2. **The folds reuse the existing pinned constant-1 column.**
   `const_one` — the constant-1 `LiteralValue` column that the Δ=0
   rotary self-match heads already read — is the 1.0 source for every
   fold. Precisely: it is *allocated once and never freed*
   (`forward_compile`), not a `ResidualStreamMap.reserve()`
   reservation (that API is the pinned-constant RMSNorm's mechanism);
   the writer reads it via `get_indices`, exactly as the self-match
   heads do. CP-SAT already subtracts this column from its modeled
   residual capacity, alongside `reserve_residual`. No new residual
   column is added.

3. **Seed unification lands first, unconditionally.** The const-1
   column starts being seeded the way the pinned-constant RMSNorm
   columns already are: folded into `embed_table` rows. This is a
   value-identical prep commit that applies to all artifacts, not just
   `bias=False` ones (see Phase N0).

4. **Constant-lane constants are powers of two, pinned in
   `ops/const.py`.** Swish machine: gate row 32, up row 1/32 — exact
   fp32 arithmetic end to end (see Phase N2). ReLU machine: gate row
   1.0.

5. **The HF converter refuses `bias=False` artifacts loudly** (as it
   already refuses swish ones). A true `LlamaForCausalLM` emission is
   a follow-up, not part of this plan.

## Where biases live today (ground truth)

Attention has no biases — `AttnLayerComponent` is Q/K/V/O matrices
only. Every physical bias is a `LinearLayerComponent.output_bias` in
the MLP sublayer: `linear1`/`linear2` (ReLU machine) or
`gate_proj`/`up_proj`/`down_proj` (swish machine). Exactly four ops in
`weight_writer.py` write them:

- `compute_ffn` — `gate_bias` into the gate projection's bias vector;
  `up_bias` (or the constant 1 for degenerate lanes) into the up
  projection's; `out_bias` into the down projection's; plus deferred
  biased-Linear input contributions folded into the hidden biases.
- `compute_literal_value` — a `LiteralValue`'s constant vector written
  as down-projection bias on its target columns. This is how graph
  constants materialize.
- `compute_bias` — the deferred bias of a `Linear` whose matmul ran on
  an attention head, added into the down-projection bias.
- `compute_linear_bypass` — degenerate up-bias 1 on both slot halves;
  folded input biases into the hidden slot biases; the Linear's own
  output bias into the down-projection bias.

One bias-like construct exists outside the layers: the ONNX seed
`res_0 = Gather(embed_table, token_ids) + constant_values`, a
positionwise Add carrying the const-1 column (and any input-state
literal seeds). The HF module mirrors it with a `constant_values`
buffer. The pinned-constant RMSNorm columns, by contrast, are already
folded into `embed_table` rows directly — two mechanisms for the same
job, split for historical reasons only (the `constant_values` vector
predates the RMSNorm feature).

## Phase N0 — prep: unify constant seeding (value-identical)

Fold `constant_values` into `embed_table` exactly the way the RMSNorm
columns fold: every vocab row gets the constant at the literal's
columns. Delete the Add node and the `constant_values` initializer
from the ONNX graph; drop the HF module's buffer and update the
converter. Bump `TOKEN_META_FORMAT` v4 → v5 (the format header is
explicit that a changed initializer set requires a version bump; old
artifacts re-export).

Bit-identical by construction: the const column is disjoint from the
embedding's columns (residual allocation is pairwise disjoint, I1), so
the folded table cells are exactly 0.0 today, and gathering a row that
contains 1.0 equals gathering 0.0 then adding 1.0. Every numeric and
parity test passes untouched; only structure-pinning expectations and
the HF converter move.

After this commit the `bias=False` feature has nothing seed-related
left to do.

## Phase N1 — fold 1: hidden-side biases → const-column rows

Under `bias=False`, every write into a gate/up bias vector becomes a
matrix write at `[const_col, slot]` — the always-1.0 residual column
makes the matmul pick up the bias term. Affected sites in
`weight_writer.py`:

- `compute_ffn`: `gate_bias`, `up_bias`; a degenerate lane's
  up-bias-1 becomes up-row 1.0 at the const column.
- The deferred biased-Linear folds in `compute_ffn` and
  `compute_linear_bypass` target the const-column row instead of the
  bias vectors.

No capacity cost — the row already exists in every matrix. The writer
asserts that no real input row aliases `const_col` (it cannot —
`const_one` is compiler-internal and no graph node resolves to its
column — but the assert makes the invariant explicit).

## Phase N2 — fold 2: output-side constants → the constant lane

The down projection reads hidden slots, not the residual stream, so a
constant entering at an output column needs a hidden slot whose value
is exactly 1.0 — the **constant lane**, one per layer:

- ReLU machine: gate row 1.0 at the const column → ReLU(1) = 1.
- Swish machine: gate row **32** at the const column, up row **1/32**.
  σ(32) = 1.0 bit-exactly on all three deployed kernels (the A0
  saturation probes pinned the thresholds: z ≥ 17 on torch-CUDA and
  ORT-CUDA, z ≥ 18 on CPU-ORT), and powers of two make
  `32 · 1.0 · (1/32) = 1.0` exact fp32 arithmetic with no rounding
  anywhere.

Every former down-projection bias write redirects to
`down_matrix[lane, cols] += value`:

- `compute_literal_value` — literals stay **bit-exact**: a literal's
  columns receive only the lane's contribution, and 1.0 × value =
  value.
- `compute_bias` — same layer, same ordering as today (matmul in the
  attention sublayer, constant in that layer's MLP sublayer).
- `compute_ffn`'s `out_bias` and `compute_linear_bypass`'s output
  bias.

Each node owns its output columns, so lane down-row cells never
collide across ops.

Lane constants land in `ops/const.py` next to `scale`, pinned by a
`test_swish_constants.py`-style test.

**Capacity and scheduling.** Hidden slot 0 is reserved when the flag
is off; the lane is written lazily, only in layers that actually use
it. Both schedulers see `d_hidden − 1`: the heuristic packs from slot
1 and CP-SAT's slot capacity drops by one — the same template CP-SAT
already uses on the residual side, where it subtracts the const-one
column and `reserve_residual` from modeled capacity before solving.
Reserving unconditionally (rather than on demand) keeps literals at
their current zero-marginal-slot-cost in both schedulers and avoids
feasibility surprises. A capacity-edge test must cover the
exactly-full layer on *both* schedulers (heuristic, and CP-SAT
replay).

The flag keys the schedule-cache fingerprint following
`reserve_residual`'s compatibility pattern: the payload gains the
field only when `bias=False`, so every existing biased-compile cache
entry keeps hashing byte-identically and still hits. Consequence,
stated plainly: the first `bias=False` compile of any topology is a
guaranteed cache miss and pays a fresh CP-SAT solve — expected, not a
regression, but real wall time at flagship scale.

**Trim.** The trailing-slice trim logic needs no change, but be
precise about what that means: the reserved slot 0 *survives* trim in
every layer that has any later used slot — that surviving idle slot is
the accepted cost, at most one per layer against `d_hidden ≥ 1024`.
Separately, the gated trim's bias-counting usedness term goes
vestigial under the flag (all bias vectors are zero); a degenerate
lane's usedness signature moves to its up-row 1.0 at the const column,
which the matrix-based check already catches.

The three param-accounting spots in `compile.py` (layer capacity,
`_count_layer_params`, trim savings) adjust for the missing bias
vectors and the lane's rows.

**Placement recording.** `_write_compute_literal_value` and
`_write_compute_bias` currently take no `PlacementRecorder` — bias
vectors are not part of the debug sidecar's 2-D weight-matrix
floorplan, so today's bias writes are correctly invisible to it.
Under `bias=False` every one of these writes becomes a real matrix
cell and must be recorded: the const-column gate/up rows from fold 1
(`mlp.W_in` / `mlp.W_up`), the lane's own gate/up cells, and the lane
down-rows from fold 2 (`mlp.W_out`). The `compute_ffn` rule "record
`W_up` only when real rows were written" also changes — a degenerate
lane now writes a real up-row cell at the const column. Skipping this
would silently under-report matrix occupancy in the sidecar.

## Phase N3 — emission, debug, metadata

- ONNX: skip all bias initializers (`l{i}_b1/b2`,
  `l{i}_bgate/bup/bdown`) and their Add nodes. The exporter asserts
  the in-process bias tensors really are all zero under the flag
  (defense in depth against a missed writer site).
- In-process `HeadlessTransformer` keeps its bias tensors, all zero —
  `x + 0.0` is exact, so probes and debug work unchanged. The
  "standardness" lives in the artifact.
- The flag is recorded in `OnnxArtifact`, the token meta, and the
  debug sidecar **outside** the frozen topology fingerprint
  (`debug_fingerprint` stays topology + geometry by contract; the flag
  rides the sidecar payload), with `OnnxDebugSession` cross-checking
  it explicitly — the same treatment `activation` got in swiglu Phase
  A4.
- HF converter: refuse `bias=False` artifacts loudly.

With `bias=False` + `rms_norm=True` on the swish machine, the ONNX
artifact is structurally a standard Llama-style decoder — Gather →
[norm → attn → norm → gated MLP] × L → norm → lm_head — with no bias
tensors anywhere.

## Numerical implications (stated upfront)

- **Not bit-identical to biased compiles in general.** A bias
  previously added after the matmul now enters the matmul's
  accumulation (as 1.0 × bias), so accumulation order shifts —
  ulp-level differences on biased hidden pre-activations and on output
  columns shared between real lanes and the constant lane. Same error
  class as before.
- The committed per-op noise numbers were measured on the biased
  machine. A flagship flip to `bias=False` requires re-measuring under
  it (D7); note this in `numerical_noise_findings.md` when the feature
  lands.
- Literals and the lane itself are bit-exact by construction
  (power-of-two arithmetic + pinned saturation).

## Graph-level ops need no changes

`bias=False` is purely about physical emission; the graph IR keeps its
biases. `add_const`, for example, builds a biased `Linear` (identity
matrix + scalar bias) and keeps working unchanged through all three
realization routes: attention matmul + deferred `compute_bias` (→
constant lane), MLP bypass (→ constant lane), or folded into a
consumer's hidden bias (→ const-column row). Routing decisions
(`_has_zero_bias`, the `biased_linears` deferral set, scheduling) are
untouched — only the final write destination moves.

## Test plan

- Writer-level units (`test_weight_writer.py` style): gate-bias
  const-row fold, degenerate up-row, lane value exactly 1.0 on both
  machines, literal bit-exactness, `compute_bias` and bypass output
  bias through the lane.
- Compile-level: `probe_compiled` end-to-end on both machines across
  all four MLP op types under `bias=False`; `add_const` as the
  smallest fixture exercising the attention-deferred and
  consumer-folded routes.
- Capacity edge: a graph that fills `d_hidden` exactly, verifying the
  reserved slot spills scheduling correctly.
- ONNX: structural checks (no bias initializers, no seed Add),
  `OnnxDebugSession` over a `bias=False` artifact, token-parity
  against a `bias=True` compile of the same graph.
- Lane constants pinned (σ(32) saturation, exact-1.0 lane) alongside
  the existing swish constant pins — on every deployed kernel: the
  torch-CPU/CUDA and ORT-CPU pins live here; the ORT-CUDA case goes in
  the doom-side probe (`tests/inference/test_ort_cuda_saturation.py`
  in torchwright_doom, where ORT-CUDA pins live because the CPU and
  GPU ORT builds collide on one import path).
- Sidecar floorplan: placement-recorder entries present for the lane
  and const-column writes (the occupancy map accounts for every matrix
  cell the folds touch).

## Follow-ups / parked

- ~~True `LlamaForCausalLM` conversion for `bias=False` + `rms_norm` +
  swish artifacts~~ — superseded 2026-07-04: stock Llama cannot express
  the flagship's partial rotary (`LlamaForCausalLM` accepts
  `partial_rotary_factor` in config but the modeling code silently
  ignores it — measured on transformers 5.12.1). The audited stock
  target is `Phi3ForCausalLM`; plan: `docs/phi3_conversion_plan.md`.
- Flagship cutover flips `bias=False` in the e1m1 configs at swiglu
  Phase D, gated on re-measured noise and a green walkthrough.
- Default flip (`bias=False` becoming the default) — not planned;
  revisit only if biased emission loses its last consumer.
