# Ops in plain English

Each op as a one-line description, the math, and how it lands on the gated
FFN hardware. `scale` is the hinge-sharpening constant, pinned at **100** —
one module constant in `ops/const.py` alongside `step_sharpness`, not a
per-call knob. It only ever appears in the self-normalizing pattern
`Swish(scale·z)/scale`, which is why big is free: the `/scale` cancels it
out of every value path; and once the sigmoid's input exceeds ~17, fp32
computes σ as exactly 1.0 (`e^{−17}` is below fp32's resolution next to 1),
so a hinge whose input sits ≥ `17/scale` past its bend is evaluated with
zero error — at scale=128 that's nearly the whole input range, and the fp
cancellation error of hinge pairs is scale-neutral besides. (Measured
per-kernel, plan A0: torch-CPU, torch-CUDA, and onnxruntime-CUDA — the
deployed inference pair — all saturate by 17. CPU onnxruntime, the
parity-test oracle, reaches exact 1.0 only from **18**, sits up to
~1.8e-7 below 1.0 on [17, 18), and is exactly 0.0 for every input
≤ −18; so a claim of *exactness on every kernel* needs its hinge
argument ≥ `18/scale` past the bend, while every error *budget* is
unaffected — the shortfall is five orders below the 0.2785/scale
sandwich. Pinned in `tests/docs/test_ort_cpu_saturation.py` and the two
CUDA probe files.) An op whose own
sharpness multiplies a *noisy value* without the matching `/scale` must
keep its own, separately budgeted constant — but no current op does: the
table lookups port as self-normalizing hinges too (see `map_to_table`,
which supersedes the C-lift bump sketched in earlier drafts of this doc).
(`embedding_step_sharpness` is a different thing: the embedding-space
*margin* knob, a `sharpness` analog — see `equals_vector`.) One recorded cost:
scale=128 saturates hidden slots at ~1.3·10⁵-magnitude values, closing the
door on any future fp16 export (the artifact is fp32 everywhere today). One
recorded gain (the reason the constant moved off the original 100 on
2026-07-04): 128 is a power of two, so the folded `k/scale` out-proj weights
are exactly representable and a saturated integer hinge value times such a
weight does not round — integer-fed indicator chains (`in_range` →
`bool_to_01` → `onehot_lookup`) are bit-exact end to end, where scale=100
leaked ~1e-5 per element and the leaks summed across wide keys (see
docs/onehot_accumulated_leak_postmortem.md).

## The FFN, and how to read the formulas

The buildable unit is the **FFN** (`torchwright/graph/ffn.py`): a set of
lanes summed by an output projection. For an input row `x`:

```
lane[j] = Swish(gate[j] · x + gate_bias[j]) · up[j](x)
FFN(x)  = Σ_j out_proj[j] · lane[j] + out_bias
```

where `up[j](x)` is either an affine `up[j] · x + up_bias[j]` (a **gated**
lane) or the constant 1 (a **degenerate** lane). `Swish` and the lane
multiply are not node types — they exist only inside an FFN lane (one node
substrate; see `docs/lowering_boundary_plan.md`). An FFN is a **packable
unit, not a sublayer**: the scheduler bins many FFNs' lanes into one MLP
sublayer's hidden pool.

How to read the formulas below:

- `Swish(g) · u` is one gated lane; a bare `Swish(g)` is one degenerate
  lane (up ≡ 1).
- `hinge(z) = Swish(scale·z)/scale` — the sharpened ReLU, within
  `0.2785/scale` of `ReLU(z)` everywhere. Every entry means exactly this
  by `hinge`, with the module `scale`, unless it states a larger
  amplification (`piecewise_linear` uses `K = scale·input_scale`).
- Every numeric constant in this doc (0.2785 and the 1.278 argmin, the
  0.557 abs peak, the ~17 fp32 saturation threshold, the
  naive-transliteration failures, the exactness identities) is
  re-derived by `tests/docs/test_swish_constants.py` — pinned by CI, not
  folklore.
- A sum of lane terms is the out_proj — free, inside the FFN, **not** an
  `Add` node. A constant added to the result is the out_bias.
- Anything one FFN cannot express — adding two FFN outputs as live values,
  an attention move — is called out explicitly, because it costs separate
  hardware (an Add node, an attention head).
- Each entry ends with its **Build** — lane count, gated or degenerate,
  what folds into which projection. (Not called "realization": that word
  already means a node's write-path class in
  `docs/lowering_boundary_plan.md`.)
- Lane counts are bookkeeping, not a design axis, for scalar ops: the
  compiled budget is dominated by the big-N constructions (one-hot keys,
  piecewise-linear grids, lookup tables — the flagship spends its ~185k
  lanes there). A one-or-two-lane difference between candidate forms of a
  scalar op never drives the choice; error properties and graph
  simplicity do.
- Where an op carries machinery the formula doesn't show — a semantic
  affine override, a surviving signature parameter — its notes say so,
  with the swish-specific slack the port must add.

## Inventory and status

Every public op has exactly one disposition below — "no entry" is never
ambiguous.

**Formula entries (this doc) — complete.** compare, select, cond_gate,
broadcast_select, equals_vector, map_to_table, onehot_lookup, in_range,
multiply (new op), square, abs, min, piecewise_linear,
scalar_to_embedding, floor_int, table_lookup_2d.

**Compositions — migrate automatically when their ingredients do:**

| Op(s) | Built from |
|---|---|
| bool_not, bool_any_true, bool_all_true | compare (+ sum_nodes) |
| clamp, reciprocal, thermometer_floor_div, mod_const, global_position_from_bos | piecewise_linear |
| ceil_int | floor_int |
| radix_floor_int | floor_int ×3 (hi floor → integer snap → lo floor over [−1, D]) + Linears; ~8.5·√N lanes vs 3N flat, ~√N-wide intermediates — see the op docstring for the snap/compensation contract |
| switch | cond_gate + sum_nodes |
| dynamic_extract | in_range + broadcast_select |
| digit_to_scaled_scalar, digits_to_number, number_to_digit_scalars | map_to_table, thermometer_floor_div |
| sum_digits, sum_digit_seqs | map_to_table |
| check_is_digit, output_sequence | select / cond_gate / map_to_table / equals_vector |
| remove_leading_0s | equals_vector + compare + in_range + broadcast_select (constant depth in max_removals) |
| count_since_marker | reciprocal + attention |
| attend_* family, get_prev_value, attend_most_recent_globally | attention hardware — untouched; MLP ingredients (min, compare) inherit those entries |

**Deletions:**

- `multiply_2d` / `_product_2d_quarter_square` — subsumed by exact
  `multiply` (4 DOOM `render_ops` call sites migrate or keep a thin
  wrapper; decide at port time).
- `relu_add` — only production caller was `abs`; the identity is
  ReLU-specific.
- `multiply_integers` — becomes plain `multiply`; the integer
  restriction and its abs/square machinery evaporate.
- `square`'s `max_value`/`step` params — exact now; signature shrinks.
- The select/cond_gate/broadcast_select offset apparatus
  (`per_column_offsets`, `scalar_M`, the finite-range requirement) —
  dies with the gated forms (see those entries' what-dies notes).
- `broadcast_select`'s `approximate` flag and its two-sublayer exact
  path — one gated form serves both callers (see the entry: the
  junk-mask and bit-exact-winner needs that motivated the split).
- `table_lookup_2d`'s column-mask staircase
  (`_table_lookup_column_mask`) and offset-cancellation gate — the
  gated column stage replaces both (see the entry).
- The C-lift bump form — superseded; recorded in `map_to_table`.

**Untouched (purely linear / attention-side):** add, subtract, negate,
add_const, multiply_const, add_scaled_nodes, sum_nodes, concat,
bool_to_01, attend_mean_where, attend_to_offset.

## The gate, and how to build a real product

The only multiply the hardware has is the lane's gate: `Swish(g) · u` — the
gate-side operand always passes through Swish, the up-side operand goes
through untouched. So a *clean* product of two live values is not a single
lane — you **build** it from a ± pair of lanes, pairing `+a`/`+b` with
`−a`/`−b` so the Swish's sigmoid factors cancel:

```
multiply(a, b) = Swish(a)·b + Swish(-a)·(-b)  =  a·b   (exact)
```

This is exact because `Swish(z) = z·σ(z)` and `σ(a) + σ(-a) = 1`, so the two
lanes sum to `a·b·(σ(a) + σ(-a)) = a·b`. This `±` identity is the foundation
of every arithmetic op below; `select` uses the same gate to apply an
*indicator* to a value instead.

---

### compare(inp, thresh)

Return `true_level` (+1 by default) if `inp` is above `thresh`, or `false_level`
(−1) if it's below.

```
z        = sharpness · (inp − thresh)      # ramp coordinate: 0 at thresh, 1 at full-true
hinge(z) = Swish(scale·z) / scale          # ≈ ReLU(z), within 0.2785/scale everywhere
compare  = false_level + (true_level − false_level) · ( hinge(z) − hinge(z−1) )
```

Notes:
- Same construction as the ReLU version — a saturating ramp built from two
  hinges, `ReLU(z) − ReLU(z−1)` — with each ReLU replaced by a *sharpened*
  Swish: `Swish(scale·z)/scale` approaches ReLU as `scale` grows, and is
  within `0.2785/scale` of it for every input. Both constants fold into the
  projections, so sharpening is free.
- Two knobs, two jobs — in the ReLU machine they were one thing. `sharpness`
  is the existing caller-facing contract: the ramp is `1/sharpness` wide in
  input units; inputs at least `1/sharpness` above `thresh` read true, inputs
  at or below `thresh` read false. `scale` is new and internal: how closely
  the smooth hinge imitates the exact ReLU hinge.
- Error at the contract points is `~e^{−scale}` in exact math (each hinge
  sits a full `scale` away from the *other* hinge's bend); at scale=128 in
  fp32 it vanishes outright — the sigmoid's input at a contract point is
  ±scale, far past the ~17 where fp32 computes σ as exactly 1.0 — so
  contract-point outputs are bit-exact. The worst case
  anywhere is `0.2785/scale · |true_level − false_level|` (0.0022 at
  scale=128), paid only by inputs landing within ~`1.3/(scale·sharpness)`
  of one of the two bend points (`thresh` and `thresh + 1/sharpness`) —
  that's Swish's negative dip. The two bends are `scale` apart in Swish
  units, so the two dips never stack.
- Unlike the ReLU form, the output is not exactly confined to
  `[false_level, true_level]` — it can overshoot either level by up to
  `0.2785/scale · |true_level − false_level|` near the bends. The output's
  value-range assert and every downstream `c_tol` budget need that slack.
- Sharpening is mandatory, not a tuning nicety: the naive transliteration
  (`scale = 1`, i.e. `Swish(z) − Swish(z−1)`) misses its saturation levels by
  `0.27·|true−false|` at the contract points and still by ~`0.10` two full
  ramp-widths past them.
- This is the op that manufactures every ±1 cond in the graph: the bool ops
  are compare compositions, and `select` / `cond_gate` consume compare
  outputs — their "cond is ±1, enforced by an assert" precondition is this
  entry's `e^{−scale}` tail. `select`'s indicator (`Swish(scale·cond)/scale`)
  is this same sharpened hinge with its bend at 0, evaluated at cond = ±1.
- Bound story: the structural FFN rule cannot bound this construction. A
  degenerate swish lane's envelope is the ReLU chord envelope minus a
  constant (the swish sandwich), and the chord relaxation of a two-hinge
  ramp with `scale·sharpness` folded into the gate rows is
  ~`sharpness·|input range|/2` wide — ~10⁶ at production sharpness. The
  semantic override (`_compare_semantic_bound`) is therefore **required**.
  It encodes the caller contract (inputs never land inside the ramp),
  collapsing to a constant when the input interval clears `thresh`. The
  port must widen it: the constant collapse is now exact only to
  `~e^{−scale}`, and the hull needs `0.2785/scale · |true−false|` slack
  for the bend overshoot.

**Build:** one FFN, 2 degenerate lanes — gate rows `scale·sharpness`
with biases putting the bends at `thresh` and `thresh + 1/sharpness`,
out_proj `±(true_level − false_level)/scale`, out_bias `false_level`.

---

### select(cond, a, b)

Choose `a` if `cond` is true, or `b` if `cond` is false.

```
select(cond, a, b) = Swish(scale·cond)·a/scale + Swish(-scale·cond)·b/scale
```

Notes:
- `cond` is ±1 (true = +1, false = −1), enforced by an assert.
- `Swish(scale·cond)/scale ≈ ReLU(cond)`: it's 1 at `cond=+1` and 0 at
  `cond=−1`. So the two gates are complementary on/off indicators — one
  passes `a`, the other passes `b`.
- The `/scale` matters: Swish is not a 0/1 gate. It grows like the identity on
  the right, so raw `Swish(scale·cond) ≈ scale` for `cond=+1`. Dividing by
  `scale` turns it back into a clean indicator. The `/scale` folds into
  out_proj, so it's free.
- Error is `~e^{-scale}` (from `Swish` not being exactly the identity at
  `scale`). It relies on `cond` sitting safely at ±1, away from Swish's bend at
  0 — the assert guarantees that.
- **What dies:** today's select is *not* a gate — it is a ReLU
  additive-cancellation construction with an offset apparatus
  (`per_column_offsets`, `scalar_M`, a finite-range requirement on both
  branches) whose cond-noise term scales as `c_tol·M` — the *range
  maximum* of the branch values. The gated form deletes all of it: no
  offsets, no `M`, no finite-range precondition from the op itself, and
  cond noise lands proportionally to the *actual* branch values. Do not
  port the offset machinery.
- Bound story: the structural rule's McCormick lane bounds get within
  ~1.10× of the true range on this construction and retain
  linear-in-input structure. The existing semantic override
  (`_select_semantic_bound`) is interval-only — the hull of the two
  branch ranges — so it is **provisional**: port it with a re-derived
  widening in actual-value terms (the old `c_tol·M` parameterization
  dies with the offset apparatus above), then measure at flagship scale
  whether the structural rule has made it dead weight.

**Build:** one FFN, `2·w` gated lanes for branch width `w`: `w` lanes
with gate row `+scale·cond` and up rows picking each component of `a`, and
`w` mirrored lanes with gate row `−scale·cond` and up rows picking `b`; the
`/scale` folds into out_proj.

---

### cond_gate(cond, inp)

Output `inp` if `cond` is true, else `0`.

```
cond_gate(cond, inp) = Swish(scale·cond) · inp / scale
```

Notes:
- `select` with the false-branch pinned to zero — just the `a` term.
- `Swish(scale·cond)/scale ≈ ReLU(cond)`: 1 when `cond=+1`, 0 when `cond=−1`.
  Multiplying `inp` by it passes the value through or zeroes it.
- `cond` is ±1, enforced by an assert; same `~e^{-scale}` error as `select`.
- Because the multiply is direct, error scales with the *actual* value of
  `inp`, not the range maximum — so small gated values stay accurate.
- **What dies:** the same ReLU-era offset/cancellation apparatus as
  `select` (cond_gate is its one-branch case). The `c_tol·M` widening in
  `_cond_gate_semantic_bound` re-derives in actual-value terms.
- Bound story: the structural rule gets within ~1.05× on this
  construction, but the semantic override (`_cond_gate_semantic_bound`)
  is stronger in kind, not just degree — in the sign-determined cases it
  passes the input's own affine rows through (output lies between 0 and
  `inp`, correlation kept). **Keep it**; re-derive its `c_tol·M` widening
  from the swish gate error.

**Build:** one FFN, `w` gated lanes (`w` = width of `inp`): every gate
row is `scale·cond`, up rows pick the components of `inp`, `/scale` folds
into out_proj.

---

### broadcast_select(masks, true_value, false_value, n_slots, d_fill)

Per slot `i`: output that slot's true value if `mask_i` is true, its false
value if false. `select`, vectorized over `n_slots` independent conds.

```
t_ij = true[j]  if true is d_fill wide (broadcast), else true[i·d_fill + j]
f_ij = the same for false
out[i·d_fill + j] = Swish(scale·mask_i)·t_ij/scale + Swish(−scale·mask_i)·f_ij/scale
```

Notes:
- Each slot is exactly `select`'s two-gate form with `mask_i` as the cond;
  broadcasting is free (the up row picks the shared column instead of the
  per-slot one). See `select` for the gate mechanics.
- **The `approximate` flag dies.** Today the op is two constructions
  behind one signature. The default mode is the additive-cancellation
  trick — four ReLU units per output column computing
  `ReLU(M·mask + t) − ReLU(M·mask) + ReLU(−M·mask + f) − ReLU(−M·mask)` —
  where a mask off ±1 by δ leaks up to δ·M of the losing branch.
  `approximate=False` is a two-sublayer cancellation-free variant (compute
  `ReLU(±mask)` first, then ReLU-clip each branch against `M·`those),
  added because the flagship's dispatch picks need two things the default
  refuses: tolerance for junk masks (no ±1 assert — see the mask-contract
  note) and a bit-exact winning row. One gated FFN serves both callers, so
  the mode split loses its reason.
- Exactness at clean masks, fp32 at scale=128: the losing branch
  contributes **exactly zero** — `σ(−128)` computes as 0.0 (pinned on CPU;
  `e^{−128}` sits below fp32's subnormal floor, so even a kernel that
  materialized the exponential would underflow to 0). The winning branch is
  **bit-exact**: the value rides through `×scale` then `÷scale`, and both
  factors are powers of two, so neither product rounds. (At the original
  scale=100 this carried up to 1 ulp relative rounding.)
- Versus today's default mode that is strictly better: the `(M+t)−M`
  cancellation recovers the winner with absolute error up to half an ulp
  *of M* (3·10⁻⁵ at M = 1000) — error at the offset's magnitude even when
  the value is tiny. The gated form's rounding is relative to the actual
  value.
- **Versus the exact mode it is the one regression, and a production
  caller leans on it:** `approximate=False` passes the winner bit-for-bit,
  and torchwright_doom's `pick_by_one_hot` uses that in the dispatch
  scalar collapse (`_collapse_scalar_emits`), whose docstring argues
  "byte-identical" head emission from it. After the port that claim
  weakens to "equal within ~10⁻⁷ relative". The picked scalars feed a
  clamp-and-quantize whose inputs already carry ~10⁻³-class
  recovered-state noise, so an ulp of relative error should be invisible —
  but that is the flagship's call to make at port time, recorded here, not
  silently absorbed.
- Mask contract: ±1 where the output is consumed — but unlike `select`,
  this op must tolerate junk masks. The flagship builds picks eagerly at
  every position and discards the output where the token type doesn't
  match, so discarded rows carry fractional masks (that's why
  `pick_by_one_hot` picks the assert-free mode today; note the current
  docstring claims a per-element mask assert in the default mode that was
  never in the code — only `select` has one). The gated form is junk-safe
  by construction: once `|mask| ≥ 0.17` the gate saturates to the mask
  value itself, so a fractional mask blends `≈ ReLU(m)·t + ReLU(−m)·f` —
  bounded by the branch hull plus the dip term `0.2785/scale·|branch|`, no
  sentinels, no offset to leak. Any mask check the port adds should be a
  debug watch scoped to trusted-mask callers, not an op-level assert.
- Noise interlock with `in_range`: `dynamic_extract` feeds this op
  in_range outputs, whose ported in-contract deviation reaches
  `4·0.2785/scale` ≈ 0.011. A saturated gate is linear in the mask, so a
  mask off by δ mis-scales the winner by exactly δ·|value| — at δ = 0.011
  that exceeds the ReLU-era cond tolerance (0.005), but it lands
  proportional to the actual value, not `M`. Consumer budgets re-derive
  from δ·|value|.
- **What dies:** `M` and everything downstream of it — `_select_offset`'s
  finite-range requirement, `_broadcast_select_per_column_offsets`, the
  `δ·M` bound widening — and the *caller-side* choreography that existed
  only to keep `M` small: the flagship pre-clamps every candidate to its
  slot range or the ±3072 atan square before picking (`clamp_to_slot`, the
  dispatch clamps in `render_main`) purely because `M` derives from the
  union of candidate ranges. Those clamps lose their reason; keep or
  remove them on their own merits at port time.
- Bound story: per slot this is `select`'s construction, and its bound
  story carries over — structural McCormick within ~1.10×,
  linear-in-input. The semantic override
  (`_broadcast_select_semantic_bound`) stays what it is — the per-channel
  hull of the two branches through the broadcast index mapping — and is
  **provisional** exactly like `select`'s: port it with the widening
  re-derived per channel as mask-tolerance·|channel hull| (replacing the
  global `c_tol·M`), then measure at flagship scale whether the structural
  rule has made it dead weight.

**Build:** one FFN, 2 gated lanes per output column (`2·n_slots·d_fill`
total): gate rows `±scale·mask_i`, up rows pick the broadcast-resolved
true/false source column, `/scale` folds into out_proj. A branch that is a
known zero literal contributes nothing — its lanes drop at build time
(`dynamic_extract`'s false branch: the op degenerates to a per-slot
cond_gate, `n_slots·d_fill` lanes).

---

### equals_vector(inp, vector)

+1 if `inp` equals the fixed key `vector`, −1 otherwise.

```
m             = vector·inp − vector·vector      # match margin: 0 at match, ≤ −1/speed at non-match
equals_vector = 2·speed · Swish(scale·(m + 1/speed))/scale − 1
```

Notes:
- A one-sided compare on a dot product: one sharpened hinge whose bend
  sits at `m = −1/speed`, with the match a full `1/speed` above it.
  `speed` is `embedding_step_sharpness` (= 1): the margin, in dot-product
  units, that distinct keys must clear — sized to absorb embedding noise
  (embedding norms are ~40, so tiny Euclidean errors become large dot
  errors). Unchanged by the port.
- Self-normalizing, so the module `scale` applies: the hinge is
  `Swish(scale·z)/scale`, and the output's sensitivity to dot-product
  noise near a match is `2·speed` — identical to the ReLU form. This op
  does *not* have `map_to_table`'s push amplification: the distinction is
  the hinge pattern (normalized by the sharpener) versus the bump pattern
  (normalized by `C`), not scalar-space versus embedding-space.
- Exactness: matches are bit-exact (+1 — hinge argument `scale/speed`,
  fully saturated). A non-match sitting exactly at the margin is exact
  (−1 — argument 0, on the bend). Non-matches within `~17/scale` *past*
  the margin land in the dip and read as low as
  `−1 − 0.557·speed/scale` (−1.0044 at scale=128); anything deeper is
  bit-exact −1.
- The value-range assert (`[−1, 1]` today) needs `0.557·speed/scale`
  slack on the low side. The top stays unclamped, as today: `m > 0` (a
  vector out-dotting the key) exceeds +1 — contract-excluded, caught by
  the assert.
- `bool_not` / `bool_any_true` / `bool_all_true` are pure `compare`
  compositions — they inherit compare's entry, not this one.
- Bound story: no semantic override; the value-range assert carries the
  claim, with the new low-side slack.

**Build:** one FFN, 1 degenerate lane: gate row `scale·vector`, bias
`scale·(1/speed − vector·vector)`, out_proj `2·speed/scale`, out_bias `−1`.

---

### map_to_table(inp, table, default)

Return the value whose key matches `inp`, else `default`.

```
m_i     = key_i·inp − key_i·key_i                     # per entry: 0 at match, ≤ −1/speed at non-match
match_i = speed · Swish(scale·(m_i + 1/speed))/scale  # ≈1 at match, ≈0 at non-match
result  = default + Σ_i match_i · (value_i − default)
```

Notes:
- A bank of `equals_vector` hinges — one per table entry, rescaled to 0/1
  instead of ±1 — with each entry's value-delta folded into out_proj. The
  deltas are constants, so the lanes stay degenerate: no live multiply
  anywhere. This is today's ReLU construction ported hinge-for-hinge.
- **Supersedes the C-lift bump form sketched in earlier drafts of this
  entry** (`Swish(push·m + C)/C`). That form existed to keep the winner's
  value un-shrunk in Swish's curved region at scale ~15, and it carried a
  real cost: the winner's sensitivity to dot-product noise was `push/C` —
  a second constant to budget. At scale=128 the plain hinge is bit-exact
  at matches, so there is no second constant and no push amplification.
- Exactness: a matching entry's indicator is bit-exact 1 (hinge argument
  `scale/speed`, fully saturated), so the result is exactly `value_i`
  plus leakage from the other entries — and in real vocabularies that
  leakage is zero: a non-matching key's margin sits at `−|key_j|²`-ish,
  hundreds of dot-units below the bend for ~norm-40 embeddings, deep in
  fp32 underflow. The dip (an indicator reading `−0.2785·speed/scale`)
  only exists for keys engineered inside the narrow window
  `m ∈ (−1/speed − 17/scale, −1/speed)`.
- Sensitivity to embedding noise at a match is `speed` per dot-product
  unit — identical to the ReLU form (self-normalizing hinge; the module
  `scale` applies). The margin contract and its rationale (norms ~40, so
  the `1/speed` margin absorbs dot noise) are unchanged —
  `embedding_step_sharpness` keeps its meaning.
- Between-keys inputs can partially fire several indicators, as today:
  the op's contract has always been approximate match, not exact
  selection. The un-clamped top also survives: `m > 0` over-drives an
  indicator past 1 — excluded by vocabulary construction, as today.
- Bound story: the hand-written value-range claim in the op (per channel,
  `default ± Σ_i |value_i − default|`) ports unchanged and stays
  essential for the same reason — without it, interval arithmetic blows
  up after a few chained lookups. Dips sit comfortably inside it (each
  entry's worst contribution is `0.0028·|Δ_i|` against a claim of
  `|Δ_i|`), so no new slack.

**Build:** one FFN, `N` degenerate lanes (one per table entry): entry
`i`'s gate row is `scale·key_i` with bias `scale·(1/speed − key_i·key_i)`,
its out_proj row is `speed·(value_i − default)/scale`, out_bias is
`default`.

---

### onehot_lookup(inp, key_to_value, default)

Map a one-hot input — or a concatenation of one-hot blocks, e.g.
`digit ⊕ digit ⊕ carry` — to a table value by exact block counting;
`default` for any input matching no key.

```
match_i = hinge(inp·key_i − (n_blocks − 0.5))     # 0.5 at the exact key; ≤ hinge(−0.5) ≈ 0 elsewhere
result  = default + Σ_i match_i · 2·(value_i − default)
```

Notes:
- Two shapes; only one migrates. With a single one-hot block
  (`n_blocks = 1`) the op is a plain selection-matrix `Linear` — no
  activation, linear hardware, untouched by the port. The hinge bank
  above is the multi-block shape.
- The counting trick ports verbatim. For one-hot blocks, `inp·key_i`
  counts agreeing blocks — an integer in `0..n_blocks`, equal to
  `n_blocks` only at the exact key — so the `−(n_blocks − 0.5)` bias
  parks every lane's argument at exactly `+0.5` (the winner) or
  `≤ −0.5` (everyone else): a structural half-count margin with no
  tuning constant. Sharpened, those arguments are `±scale/2` — deep in
  saturation on both sides.
- vs `map_to_table`: the same bank-of-indicator-lanes shape, but this op
  carries **no margin knob at all** — no `speed`, no sharpness of its
  own; the margin is the integer count structure, and the module `scale`
  is the only constant. That was the op's appeal in the ReLU machine and
  it survives unchanged.
- Exactness, fp32 at scale=100: the winner's indicator is exactly 0.5
  (`σ(50) = 1.0`, and `50/100` is exact), so the result is `value_i` to
  ~1 ulp of the value (the `×scale/÷scale` round trip — 6·10⁻⁵ at values
  near 1000). A non-matching lane leaks `hinge(−0.5) ≈ −10⁻²²` — *not*
  the exact zero of `σ(−scale)` (`e^{−50}` is still representable in
  fp32), but twenty orders of magnitude below visibility: a no-match
  input returns `default` bit-exactly in practice, the leaks vanishing
  under fp32 addition. Pinned in `tests/docs/test_swish_constants.py`.
- Noise pass-through is unchanged from the ReLU form: an input one-hot
  off by ε (recovered-state softmax-noise class) shifts the count by ε
  and the winner's indicator by exactly ε — the saturated gate is linear
  down to a count deviation of 0.33 (`17/scale` short of the margin) —
  so the output error is `2ε·|value_i − default|`, today's coefficient.
- Bound story: the tight `[min, max]` claim over the values and the
  default — the reason this op exists (`map_to_table`'s pessimistic
  `default ± Σ|Δ|` widening blew up chained interval arithmetic) —
  survives with **no new slack**: winner rounding (~1 ulp) and dip leaks
  (~10⁻²²) sit far inside the closing assert's existing 1e-3 tolerance.
  No semantic override, as today.
- Callers: the calculator examples (both shapes); the flagship does not
  use it.

**Build:** one FFN, `N` degenerate lanes (one per table row): gate row
`scale·key_i`, bias `−scale·(n_blocks − 0.5)`, out_proj row
`2·(value_i − default)/scale`, out_bias `default`. The `n_blocks = 1`
shape stays a plain `Linear` — no FFN at all.

---

### in_range(lower, upper, n_slots)

Per integer slot `i ∈ {0..n_slots−1}`: +1 if `lower ≤ i + 0.5 < upper`,
else −1.

```
center_i     = i + 0.5
past_lower_i = hinge(S·(center_i − lower)) − hinge(S·(center_i − lower) − 1)   # unit step: 1 once lower ≤ center_i
past_upper_i = hinge(S·(center_i − upper)) − hinge(S·(center_i − upper) − 1)   # unit step: 1 once upper ≤ center_i
out_i        = 2·(past_lower_i − past_upper_i) − 1
```

Notes:
- Per slot, exactly two `compare`-shaped saturating ramps (unit steps with
  ramp width `1/S`, `S = step_sharpness`) combined in out_proj: a slot is
  in range when the lower edge has passed its center and the upper edge
  hasn't. Compare's entry covers each ramp; this entry is about the
  composition.
- Integer-valued bounds are **bit-exact** across the whole slot vector:
  the `+0.5` center offset keeps every hinge argument at least 4 units
  from its bend (`S·|center − bound| ≥ 5`, minus 1 for the ramp's second
  hinge) — ≥ 400 in Swish units at scale=100, fully
  saturated/underflowed.
- Continuous bounds inherit compare's contract per boundary: a bound
  inside a center's ramp zone `(center − 1/S, center)` makes that slot an
  interpolated intermediate (exactly as today), and a bound within
  `~17/(scale·S)` of a ramp edge adds a fillet dip. At most two hinges
  per slot can sit in fillets at once (one per bound), so the worst
  in-contract error is `4·0.2785/scale` ≈ 0.011 — the `[−1, 1]`
  value-range assert needs that slack.
- The outputs are consumed as ±1 conds (`dynamic_extract` chains this
  into `broadcast_select`), so the slack above is what lands on
  broadcast_select's mask contract — see that entry's noise interlock.
- Bound story: no semantic override; the value-range assert carries the
  claim, with the new slack.

**Build:** one FFN, 4 degenerate lanes per slot (`4·n_slots` total): two
sharpened two-hinge ramps reading `−lower` / `−upper` (gate entries
`−scale·S` on the respective input column, biases `scale·S·center_i` and
`scale·(S·center_i − 1)`), out_proj `[+2, −2, −2, +2]/scale` into slot
`i`, out_bias `−1`.

---

### multiply(a, b)

Multiply two live values.

```
multiply(a, b) = Swish(a)·b + Swish(-a)·(-b)
```

Notes:
- Exact (`a·b`), for all `a`, `b` — no range limit, no grid. See *The gate*
  above for why: the `±` pair makes the Swish sigmoids sum to 1.
- Both terms share the sign of `a·b`, so they add constructively — no
  catastrophic cancellation.
- This replaces the ReLU-era workarounds for multiplication (the quarter-square
  construction in `multiply_2d`, the `signed_multiply` chain).

**Build:** one FFN, 2 gated lanes: gate rows `+a`/`−a`, up rows
`+b`/`−b`, out_proj `[1, 1]`.

---

### square(inp)

Compute `inp²`.

```
square(inp) = Swish(inp)·inp + Swish(-inp)·(-inp)
```

Notes:
- `multiply(a, b)` with `a = b = inp`. Exact (`x²`) for all `inp`.
- Both terms are `x²·σ(±inp)` — non-negative, so they add cleanly.
- Drops the current `[0, max_value]` restriction, the `step`/grid, and the huge
  near-zero relative error of the piecewise-linear version (which approximates
  `x²` by straight segments and is worst exactly where `x²` is smallest).

**Build:** one FFN, 2 gated lanes: gate rows `±inp`, up rows `±inp`,
out_proj `[1, 1]`.

---

### abs(inp)

Element-wise absolute value.

```
abs(x) = Swish(scale·x)/scale + Swish(-scale·x)/scale   =   x·tanh(scale·x/2)
```

Notes:
- The ReLU identity (`|x| = ReLU(x) + ReLU(−x)`, exact) with each ReLU
  replaced by the sharpened hinge from `compare`. Unlike `multiply`'s ±
  pair, the two sigmoids here *add* instead of cancelling, giving
  `x·tanh(scale·x/2)` — an approximation. This is the rare op that
  regresses under swish: exact today (its noise entry is literally 0),
  approximate after the migration. There is no exact swish form: `|x|`
  has a corner, and every finite sum of Swish lanes is smooth.
- The error is one-sided and bounded: the output always lies in
  `[0, |x|]` — never negative, never above the true value. Worst
  underestimate is `0.557/scale` (0.0056 at scale=100), hit at
  `|x| = 1.278/scale` (Swish's argmin: one hinge sits in its dip while the
  other hasn't reached its straight region). Exact at `x = 0`, and
  `~2|x|·e^{−scale·|x|}` beyond the bend region — at scale=100 in fp32,
  tanh saturates to exactly 1.0, so abs is bit-exact for every
  `|x| ≳ 0.2`: the entire integer grid.
- Because the output stays in `[0, |x|]` exactly, the `≥ 0` range claim
  survives with no slack. Only consumers that need `abs` to *not
  under-read* near the origin — anything dividing by it, or comparing it
  against a small threshold — must budget the `0.557/scale`.
- Consumers: `min` is the main graph-op caller (`(a+b−|a−b|)/2` — its
  entry decides whether it keeps the abs route or switches to a direct
  hinge). DOOM's coordinate abs goes through `piecewise_linear`
  breakpoint tables instead and inherits that entry's story.
- `relu_add` — today's substrate for abs (`ReLU(a)+ReLU(b)`) — has no
  other production caller and does not survive the migration.
- Bound story: no semantic override today and none needed after the port.
  The swish sandwich over the ± hinge pair concretizes to about
  `[−0.557/scale, max|x|]` on a symmetric input range, and the true
  output range `[0, max|x|]` sits inside it.

**Build:** one FFN, 2 degenerate lanes: gate rows `+scale·x` / `−scale·x`,
out_proj `[1/scale, 1/scale]`.

---

### min(inp1, inp2)

Element-wise minimum of two nodes.

```
min(a, b) = a − Swish(scale·(a−b))/scale         # a − hinge(a−b), hinge ≈ ReLU(a−b)
```

Notes:
- Replaces today's abs route (`(a+b−|a−b|)/2`). Under swish the two forms
  have *identical* error — the abs route's `/2` halves abs's `0.557/scale`
  right back to the single hinge's `0.2785/scale`, same direction — so the
  choice falls to graph simplicity: the hinge form is self-contained, with
  no dependency on the abs op's budget. (Hardware cost does not enter:
  scalar-op lane counts never drive design, per the preamble.)
- Error is one-sided: min is *over*-estimated by at most `0.2785/scale`
  (0.0028 at scale=100), and only when `|a−b| ≲ 0.2`. Ties are exact
  (`a=b` puts the hinge at its bend, where `Swish(0)=0`), and
  `|a−b| ≳ 0.2` is bit-exact in fp32 — so on the integer grid, min is
  exact everywhere.
- The construction is asymmetric (`a` is the pass-through) but the error
  is not: the hinge's gap to ReLU is an even function of `a−b`, so
  `min(a,b)` and `min(b,a)` compute the same value.
- The `+a` pass-through is a bypass pair: `Swish(scale·a)/scale −
  Swish(−scale·a)/scale = a` — exact at any sharpening (`σ(z)+σ(−z)=1`
  again; this is the identity the `mlp_bypass` realization class relies
  on). Convention: bypass pairs are sharpened too — the *value* is exact
  either way, but the affine-bound sandwich slack on the pair is
  `±0.2785/scale` sharpened versus `±0.2785` raw, a 100× tighter bound
  for free.
- fp note: min of far-apart magnitudes inherits the larger operand's
  relative fp error (the `a − (a−b)` cancellation) — unchanged from the
  ReLU machine.
- A cheaper correction form exists (one hinge lane written onto `a`'s
  residual columns when they're dead — an op-site candidate in the
  lowering plan's sense). Not built: the lane delta is immaterial by the
  budget calibration above, which is one more sign item D never fires.
- Bound story: no semantic override today, none needed. With all three
  lanes sharpened, the sandwich concretizes to within `~1/scale` of the
  true hull `[min(lo_a,lo_b), min(hi_a,hi_b)]`.

**Build:** one FFN, 3 degenerate lanes per component: hinge lane with gate
row `scale·(a−b)` and out_proj `−1/scale`; bypass pair with gate rows
`±scale·a` and out_proj `±1/scale`.

---

### piecewise_linear(inp, breakpoints, fn)

Evaluate a piecewise-linear function: interpolate `fn`'s values at the
breakpoints, clamped (by default) outside the range.

```
hinge(z) = Swish(K·z) / K                     # K = scale · input_scale
f(x)     = y_0 + Σ_i Δm_i · hinge(x − x_i)    # Δm_i = slope change at breakpoint x_i
```

Notes:
- The ReLU construction ports verbatim: one lane per slope *change*
  (equal-slope segments stay free), the clamp trick (a slope-cancelling
  hinge at the last breakpoint), `d_max` chunking, and vector-valued `fn`
  sharing lanes with per-dim out_proj — all unchanged. `input_scale`'s
  argument amplification is the same trick as sharpening, so the two
  knobs simply multiply into `K = scale·input_scale` in the gate rows.
- Error structure: the swish PL is the exact PL function with each corner
  rounded in a radius-`~17/K` fillet. Outside the fillets it is
  **bit-exact in fp32**: segment interiors compute the exact
  interpolation, and the function passes through every knot exactly (the
  local hinge contributes `Swish(0) = 0`; all others are saturated).
  Inside a fillet the error is ≤ `0.2785·|Δm_i|/K` — it scales with the
  slope change, not the value magnitudes.
- Staircases are the steep case (`Δm ≈ ramp sharpness`), and the
  docstring's existing advice — `input_scale = step_sharpness` — makes
  the two cancel: bend error ≈ `0.2785/scale · step_height` (0.0028 at
  scale=100).
- Spacing condition: two breakpoints closer than `~34/K` have overlapping
  fillets and their errors stack (additively). Migration audit, per call
  site: check each committed grid's minimum spacing against `34/K`. The
  staircase ops place hinge pairs `1/sharpness` apart — fine when
  `input_scale ≈ sharpness`, but check, don't assume.
- Monotonicity is no longer exact: Swish's derivative dips to −0.0998, so
  a monotone target acquires a dip of up to `0.2785·|Δm|/K` just before
  each rising ramp. Consumers thresholding a staircase with 0.5-ish
  margins don't care; anything with sub-`0.003·|Δm|` margins does.
- Non-changes: the chord error *between* breakpoints — approximating a
  curved `fn` by segments, which dominates today's measured noise — is
  untouched; and smooth-target grids (`reciprocal`) need no redesign,
  since the fillet bends toward the curve the corner was approximating.
  `clamp`, `reciprocal`, `thermometer_floor_div`, `mod_const`, and
  `global_position_from_bos` inherit this entry.
- Bound story: no semantic override today, none added. The structural
  rule's chord looseness is unchanged from the ReLU machine; the swish
  sandwich adds `Σ_i 0.2785·|Δm_i|/K` of lower slack — additive across
  lanes, so big grids pay it N times in the *bound* (never the value).

**Build:** one FFN per `d_max` chunk of slope changes: lane `i` degenerate
with gate row `±K` and bias `−K·x_i`, out_proj row `Δm_i/K` (one column
per output dim), out_bias `y_0` on the first chunk. Grids beyond `d_max`
split into multiple FFNs joined by `sum_nodes` — real Add nodes, the one
place this op leaves the FFN.

---

### scalar_to_embedding(inp, embedding)

Convert a scalar digit in `[0, 9]` back to its token embedding — the tail
of the digit pipeline (`digits_to_number` → scalar arithmetic →
`number_to_digit_scalars` → this).

```
step_k = hinge(z_k) − hinge(z_k − 1),  z_k = S·(x − (k+0.5))   # unit step at threshold k+0.5, k = 0..8
result = embed(0) + Σ_k step_k · (embed(k+1) − embed(k))       # telescoping embedding deltas
```

(`S = step_sharpness` = 10.)

Notes:
- A `piecewise_linear` special case, ported hinge-for-hinge: nine unit
  steps at half-integer thresholds, vector-valued (the embedding deltas
  fold into out_proj, so all lanes stay degenerate — no live multiply).
  It gets its own entry only because it is the one op whose *output*
  lives in embedding space, so its error budget is judged against
  embedding margins, not scalar tolerances.
- The single-FFN form stays safe — no `floor_int`-style two-stage split
  needed. That split exists to bound partial sums when boundaries number
  in the thousands and arguments reach fp32's 2²⁴ window; here there are
  9 boundaries and the sharpened arguments top out near
  `scale·S·9 ≈ 10⁴`. Two different regimes of the same construction;
  this one is the small-and-safe end.
- Exactness on the contract inputs: an integer digit `d` puts every
  hinge argument on an exact integer (`z = 10·d − 10·k − 5`, never zero,
  always `|z| ≥ 5` — so every sharpened argument is ≥ 500, saturated or
  underflowed exactly). The indicators are exact 0/1 and the only error
  is out_proj rounding: a few ulps of the embedding components, ~2·10⁻⁶
  at norm-40 embeddings — the same class as today's matmul rounding.
  Against the embedding-space margin (`1/speed` = 1 dot-product unit;
  see `equals_vector`) that is invisible.
- Input-noise headroom unchanged: a digit scalar off by up to ±0.4
  reconstructs the same embedding (the nearest threshold is 0.5 away and
  the ramp is `1/S` = 0.1 wide; saturation holds until `17/(scale·S)` =
  0.017 from a ramp edge). Mid-ramp inputs blend the two adjacent
  embeddings linearly, as today.
- The `piecewise_linear` spacing audit closes in closed form here: the
  unit-step pair's two hinges sit `1/S` apart and each fillet has radius
  `17/(scale·S)`, so the fillets never overlap exactly when
  `scale > 34` — independent of `S`. The module scale = 100 clears it
  3×. (The same fact, in Swish units, is compare's "the two dips never
  stack".)
- Bound story: no semantic override and no closing value-range assert
  today; none added. The FFN's structural sandwich carries
  `piecewise_linear`'s additive slack (`Σ_k 0.2785·|Δ_k|/K` per
  component, `K = scale·S`).
- Callers: the adder/calculator/fibonacci examples; the flagship emits
  tokens through its own head machinery and does not use this op.

**Build:** one FFN, 18 degenerate lanes (2 per threshold): gate rows
`scale·S·x` with biases `−scale·S·(k+0.5)` and `−scale·(S·(k+0.5) + 1)` —
the pair's bends at `z_k = 0` and `z_k = 1`; out_proj rows
`±(embed(k+1) − embed(k))/scale`, out_bias `embed(0)`.

---

### floor_int(inp, min_value, max_value)

Floor of a continuous scalar known to lie in `[min_value, max_value]`.

```
t_k    = sharpness·(x − k) + 1                    # per boundary k = min+1 .. max
step_k = hinge(t_k) − hinge(t_k − W)              # FFN 1: clamped step ∈ [0, W], W ≈ 2
floor  = min + n − Σ_k hinge(1 − step_k)          # FFN 2: count the not-yet-ON steps
```

Notes:
- **Not a flat staircase, and the depth is load-bearing.** The obvious
  single-FFN form (`Σ ±sharpness·hinge(x − k)`, what `piecewise_linear`
  would emit) was this op's original implementation and was abandoned:
  it sums ~n terms of magnitude `sharpness·x` in one projection, whose
  partial sums overflow fp32's 2²⁴ exact-integer window at production
  magnitudes and collapse. The two-stage form keeps every accumulated
  term bounded (`[0, W]`, then `[0, 1]`). That constraint is about fp32
  accumulation, not about the activation — do not "simplify" this back
  to one layer. A mod-1 route (`x − frac(x)`) buys nothing either:
  `frac` is a sawtooth with a slope change at every integer — same lane
  count, same overflow problem, plus a live subtract.
- Contract unchanged from today: inputs stay out of the
  `1/sharpness`-wide ramp zone just below each boundary; the flat zone
  `[k, k+1−1/sharpness]` is the home of legal inputs.
- Exactness: flat-zone interiors and exact integer inputs are
  **bit-exact** (at `x = k`, the critical hinges sit either exactly on a
  bend, where `Swish(0) = 0`, or fully saturated; every other boundary's
  step is saturated ON or OFF). The port adds fillet zones of width
  `~17/(scale·sharpness)` at each ramp edge contributing
  ≤ `0.2785/scale` apiece — at most a couple are ever live at once, so
  in-contract error stays ≲ 0.006 at scale=100.
- **The W-slack now absorbs fillets too.** An ON step parks stage 2's
  hinge argument at `1 − W = −1`, hundreds of times past the `17/scale`
  saturation margin — so stage-1 fillet noise (±0.0028) on an ON step
  still reads exactly 0 in stage 2, the same mechanism that absorbs fp
  ulps today. The existing sizing `W = max(2, 8·ulp(sharpness·n))`
  already dominates the swish requirement `W ≥ 1 + 17/scale`.
- `ceil_int` inherits this entry (`−floor_int(−x)`).
  `thermometer_floor_div` is *not* this op (different thresholds,
  integer-input contract) — it inherits `piecewise_linear`.
- Bound story: no semantic override; the closing value-range assert
  (`[min_value, max_value]`) carries the tight claim, exactly as today.
  Structural chord looseness is unchanged from the ReLU machine, with
  the same additive sandwich slack as `piecewise_linear`.

**Build:** two chained FFNs per 512-boundary chunk. FFN 1: 2 degenerate
lanes per boundary — gate rows `scale·sharpness·x` with biases
`scale·(1 − sharpness·k)` and `scale·(1 − W − sharpness·k)`, out_proj
`±1/scale` into that boundary's step column. FFN 2: 1 degenerate lane per
boundary — gate row `−scale` on the step column, bias `scale`, out_proj
`−1/scale`, single output column. Chunks and the `min + n` constant join
via `sum_nodes`/`add_const` — Add hardware, as today.

---

### table_lookup_2d(i, j, table)

Read one scalar out of a compile-time constant `A×B` table at integer-ish
indices `(i, j)`, clamped to the table edges when out of range.

```
step_k(x) = hinge(t_k) − hinge(t_k − W),  t_k = s·(x − (k−0.5)) + 0.5   # bounded step ∈ [0, W] at boundary k−0.5
row_c = T[A−1, c] − Σ_k hinge(1 − step_k(i)) · (T[k,c] − T[k−1,c])      # constant deltas → degenerate lanes
out   = row_{B−1} + Σ_k hinge(1 − step_k(j)) · (row_{k−1} − row_k)      # live deltas → gated lanes
```

(`s = sharpness`; both indices min-clamped to their axis first.)

Notes:
- Both axes are the same machine: `floor_int`'s two-stage saturating
  staircase — a bounded per-boundary step, then a "boundary not yet
  reached" indicator `hinge(1 − step)` ∈ [0, 1] weighting that boundary's
  table delta. That entry's rationale (the depth is load-bearing;
  single-projection forms overflow fp32's 2²⁴ window; the `W ≈ 2` cap
  makes every accumulated term bounded and its slack absorbs fp error at
  saturated steps) carries over unchanged, deltas now vector-valued.
- **The two axes differ only in whether the deltas are live, and that is
  the whole redesign.** Row-axis deltas are table constants — they fold
  into out_proj, lanes stay degenerate, the port is hinge-for-hinge.
  Column-axis deltas are differences of the just-built row vector — a
  *live* multiply, impossible on ReLU hardware, which is why today's
  column axis takes a detour: build a ±1 one-hot mask with a *second full
  staircase*, then apply it with an offset-cancellation gate
  (`0.5·[ReLU(offset·mask + v) − ReLU(offset·mask − v)]`,
  `offset = max|table| + slack + 1`). The gated hardware collapses the
  detour into one gated lane per column boundary — gate row
  `scale·(1 − step_k(j))`, up row `row_{k−1} − row_k` — the fp-snap and
  the multiply in a single lane. The mask staircase and the offset gate
  both die.
- Why the gate is a hinge and not the exact ± multiply pair: the
  indicator must be *snapped*. A deeply-on step carries fp noise at
  `ulp(scale·s·x)` scale; the hinge's saturated ends read it as exactly 0
  or exactly 1 (`σ(−scale) = 0`, `σ(scale) = 1` in fp32 — the same
  mechanism as `broadcast_select`'s losing branch), where the exact
  multiply pair would faithfully reproduce the noise times the delta.
- Cost: **3 lanes per boundary on both axes** — 2 for the bounded step,
  1 for the indicator-times-delta — so `3(A+B)` plus two 3-lane
  min-clamps and one always-on pass-through lane, versus today's
  `3A + 5B` (the column axis pays 3 for the mask staircase + 2 for the
  offset gate). Depth unchanged at 4 chained MLP sublayers (clamp → step
  → row deltas → column gate). Flagship flat table (2048×128,
  sharpness 1000): ~6.8k hidden units today → ~6.5k.
- **Dimension ordering, revisited.** Today's per-boundary cost is
  asymmetric (3 on rows, 5 on cols), so the fused axis goes on rows —
  the flagship fuses `(id, v)` into the row index by affine arithmetic
  (`row = base_rows[id] + v`) and puts screen-`u` on columns. The swish
  form is symmetric, which changes the optimum: for a fixed entry count
  `A·B`, lanes `3(A+B)` are minimized by *balancing* the axes. The
  16×128×128 flat bank as 512×512 instead of 2048×128 would cost ~3.1k
  lanes instead of ~6.5k — but the rebalanced row/column indices need
  divide-and-remainder arithmetic (`floor_int` + multiply-add) instead
  of one free Linear. That is a flagship call-site decision with its own
  noise budget, recorded here because this op is one of the flagship's
  biggest lane consumers — not something the op does by itself.
- Exactness: on the integer grid and outside boundary bands, every
  indicator is exactly 0 or 1 (off steps are exactly 0; on steps land in
  the `W`-slack where `1 − step ≤ −1` saturates the hinge to zero), so
  the value path is the delta telescoping's fp32 accumulation — the same
  sum today's row stage performs. What disappears is the offset gate's
  *additional* absolute error at `ulp(offset)` scale (6·10⁻⁵ at
  max|table| = 1000 — the `(M+v)−M` class pinned in the
  broadcast_select what-dies test).
- **In-band behavior upgrades from disclaimer to contract.** Today's
  docstring says in-band output is "defined and local... not a bilinear
  interpolation guarantee". The swish form's band coefficient is the
  blend fraction itself (`hinge(α) = α` exactly in fp32 for
  `α ≥ 17/scale`, since σ saturates), so a band on one axis gives the
  clean two-entry linear blend, and a corner band (both indices mid-ramp)
  gives genuine bilinear interpolation — pinned in
  `tests/docs/test_swish_constants.py`. Below `α = 0.17` the coefficient
  rolls through Swish's dip: error ≤ `0.2785/scale · |Δ|` per band edge,
  `Δ` the local adjacent-entry difference.
- `W` re-derivation: keep today's sizing
  (`W = max(2, 8·ulp(s·(n−1)))`) plus the swish saturation floor
  `W ≥ 1 + 17/scale` (as in `floor_int`). The step pair is now computed
  at `scale·s·x` magnitudes, but the `/scale` in out_proj brings the fp
  error back to today's `ulp(s·x)` class — the existing 8× margin
  covers it at flagship sizes; re-derive per site only if `scale·s·n`
  approaches 2³¹.
- What survives: `sharpness` and its contract (ramp width `1/s` centered
  on half-integer boundaries; flagship uses 1000), `index_scale` (can now
  fold into the step FFN's gate rows, deleting the separate scaling
  Linear), the `d_max = 1024` chunking, the min-clamps, and the
  output-range guard with `_lookup_numeric_slack` (its GPU
  reduced-precision-matmul rationale is untouched).
- What dies: `_table_lookup_column_mask` and the final offset gate —
  including the `offset·0.005` term in the closing guard's tolerance, the
  op family's last range-coupled offset. `_saturating_step_select`
  survives as the shared staircase engine (it *is* the row stage).
- Bound story: no semantic override today, none added. The closing
  value-range assert (table min/max) carries the claim; its tolerance
  drops the `offset·0.005` term and gains the band-edge dip term
  `0.2785/scale · max|Δ|`.

**Build:** per axis, `floor_int`'s step FFN (2 degenerate lanes per
boundary, chunked at 512 boundaries). Row axis third stage: 1 degenerate
lane per boundary — gate row `scale·(1 − step_k(i))` (coefficient
`−scale` on the step column, bias `scale`), out_proj row
`−(T[k,:] − T[k−1,:])/scale`, out_bias `T[A−1,:]` on the first chunk.
Column axis third stage: 1 gated lane per boundary — same gate row shape
on `step_k(j)`, up row `row_{k−1} − row_k`, out_proj `1/scale` — plus one
always-on lane (gate = bias `scale`, up row `row_{B−1}`) carrying the
live top entry that out_bias cannot. Min-clamps: swish `min`, 3 lanes per
index. Chunks join via `sum_nodes`, as today.

---

## Migration checklist (collected from the entries)

The first item gates everything else; the rest land with their ops.

1. **Runtime saturation probe, before any op ports.** The bit-exactness
   claims assume the deployed kernels compute `σ(z) = 1.0` exactly for
   `z ≥ 17`, `Swish(0) = 0`, and `σ(−scale) = 0.0` (the losing-branch
   exact zero in broadcast_select/select; a kernel returning the
   denormal `e^{−scale}` instead is fine for budgets but softens the
   "exactly zero" claims). `tests/docs/test_swish_constants.py`
   pins these on CPU; verify once on torch-CUDA and onnxruntime-CUDA. If
   a kernel misses by an ulp, budgets survive (~1e-7-class errors) but
   every "bit-exact" claim in this doc — and any test asserting exact
   equality — must be softened.
2. **Builder API — pinned** (see `docs/swiglu_step2_plan.md`):
   `swiglu_ffn(input_node, gate_proj, gate_bias, output_proj,
   output_bias, *, up_proj=None, up_bias=None, name="")` in
   `ops/swiglu`, hardcoding `activation="swish"`; degenerate vs gated
   is `up_proj` presence.
3. **piecewise_linear grid audit.** Per call site: minimum breakpoint
   spacing vs `34/K`. Pre-cleared: `scalar_to_embedding` (its pair
   spacing reduces to `scale > 34` in closed form — see the entry).
4. **Semantic overrides.** Re-derive widenings: compare (hull slack
   `0.2785/scale·|T−F|`; the constant collapse keeps its caller-contract
   soundness argument), select / cond_gate (actual-value-term widenings
   replacing `c_tol·M` — see the what-dies notes), broadcast_select
   (per-channel mask-tolerance·|channel hull|; the mask tolerance itself
   must absorb in_range's `4·0.2785/scale` ≈ 0.011 — the ReLU-era 0.005
   cannot survive for in_range-fed masks).
5. **Assert-slack sweep.** Every value-range assert on an indicator
   output gains its entry's dip slack: compare `0.2785/scale·|T−F|`,
   equals_vector `0.557·speed/scale` (low side), in_range
   `4·0.2785/scale`. table_lookup_2d's closing guard drops its
   `offset·0.005` term and gains the band-edge dip term
   `0.2785/scale·max|Δ|`.
6. **Deletions land with their replacements, D7 in force.** `relu_add`
   with abs; the offset apparatus with select/cond_gate; `multiply_2d`
   with multiply. Re-measure noise in the same commits; update
   `scripts/measure_op_noise.py` TargetOps for new, changed, and dead
   ops.
