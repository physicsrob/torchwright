# Plan: RMSNorm as a compiled-transformer identity

**Status:** design exploration, hardened against post-merge `main` (the merges
that added the HuggingFace export layer `torchwright/compiler/hf/`, removed the
windowed KV-cache, and switched the token export to a vanilla untied
llama3-style embedding — `embed_table` is now `(vocab, d)`, gathered straight
into the seed, with no `embedding_proj` / `d_embed`). Decisions below are
**leanings, not locked**, except the four marked **[LOCKED]**. Prototyped on
`calculator_simple`; not yet wired into any emitter. Branch:
`worktree-rmsnorm-prototype`.

## Goal & motivation

Give torchwright-compiled transformers a *real* RMSNorm, so the artifacts they
emit are architecturally standard (a stock decoder-only transformer that runs
on HuggingFace / standard inference engines). The two driving users are the
**calculator** (calculator blog post) and **DOOM** (DOOM blog post).

The norm computes **nothing** — by construction it is the identity. Its entire
value is faithfulness/credibility: today the compiler emits "no normalization
anywhere," and a skeptical reader sees that as skipping the hard parts.

**This gap is already shipped and explicit.** `examples/calculator_hf_export.py`
publishes the compiled calculator as a HuggingFace model whose card claims it is
"a *bona fide* standard transformer, not a bespoke runtime" — while the shipped
model code (`torchwright/compiler/hf/modeling_torchwright.py:26`) and config
(`configuration_torchwright.py:21`) both document **"No normalization anywhere;
no final norm"** as a correctness invariant. The norm closes exactly that gap.

**The honest framing (and the actual insight):** real transformers need
normalization to keep activations in range *during training*. We don't train —
we compile exact values — so our norm doesn't need to *do* anything. We include
a genuine RMSNorm and arrange the residual stream so it acts as the identity,
preserving our exact values. That trained-vs-compiled distinction is the blog
point; if we're not willing to make it, there's no reason to add the norm.

## Mechanism

RMSNorm computes, per position, `rms = sqrt(mean(x^2) + eps)` then outputs
`x_i / rms * g_i` (g = per-channel gain).

1. **Pin the RMS.** Reserve a residual column (two at odd power-of-two
   widths — see *Two gain variants*) holding a large constant. Its
   squared magnitude dominates the sum-of-squares, so `rms` becomes a
   position-independent constant `C`. In fp32 the data energy falls below the
   constant's LSB, so `rms` is **bit-exactly** constant (measured:
   `rms_spread = 0`).
2. **Cancel it with the gain.** Set the gain uniformly to `C`. Then
   `x_i / rms * C = x_i` — the norm is the identity, and **every existing
   compiled weight is reused unchanged**. The constant column(s) are seeded once
   (folded into the embedding table — see *the one unavoidable core hook*), pass
   through each norm unchanged, and are never written by any sublayer, so they
   re-pin the RMS at every layer (including the final norm).

**Insertion points (pre-norm).** Each emitter currently produces, per layer,
`res = res + attn(res)` then `res = res + mlp(res)` with no norm
(`export.py:_emit_cached_layer_nodes` lines ~822–938; `modeling_torchwright.py`
`TorchwrightDecoderLayer.forward`). Pre-norm inserts the norm on the *input* to
each sublayer only: `res = res + attn(norm(res))`, `res = res + mlp(norm(res))`.
The skip term stays the un-normed `res`, so the residual stream itself is never
overwritten with normed values. A **final norm** is applied to the last hidden
state before the unembed (an untied `lm_head` over the full residual stream,
`export.py:1265–1266` — `res @ lm_head.T`; no output-column gather post-merge),
matching a standard Llama-style decoder. **[LOCKED: include the final norm.]**

### Why the gain is necessarily large (a credibility caveat)

Making RMSNorm an identity *by pinning* forces `g_i = C` for every channel,
because cancelling a position-constant RMS with the per-channel gain leaves no
freedom: `x_i / C * g_i = x_i` only when `g_i = C`. `C` is the large pinned
constant (e.g. `2^25 ≈ 3.3e7`), so **every gain in every layer is that same
large value** — unavoidable in this design, and visibly atypical (trained
RMSNorm gains cluster near 1). This does **not** affect "runs on standard
engines" (RMSNorm runs identically regardless of gain magnitude), but a skeptic
who inspects the *weights* (rather than reading the blog's framing) sees an
immediate tell. Lean: accept it and name it in the model card, consistent with
the honest-framing philosophy above. It is the one thing that partially
undercuts the credibility goal, so it is called out here rather than left
implicit.

### Two gain variants

The committed design **[LOCKED: pin the constant energy so the forced RMS is an
exact power of two]** (see Constraints), which makes the gain bit-exact for
*every* graph at any power-of-two width. The formula gain is recorded only as
the fallback this commitment lets us avoid.

- **Power-of-two-RMS gain (committed)** Pin the constant column(s) so the total
  pinned energy is `d · 2^(2m)` for integer `m`; the forced
  `rms = sqrt(d·2^(2m)/d) = 2^m` is then an exact power of two. Set the gain
  `= 2^m`. Then `÷rms` and `×gain` are pure exponent shifts — **bit-exact for
  any float** (modulo a denormal-scale floor for near-zero data). Required for
  DOOM (see Evidence).

  **How many constant columns** depends on the width `d = 2^b`. One column
  `= 2^q` contributes energy `2^(2q)` (an *even* power of two), so its RMS
  `2^(q - b/2)` is an exact power of two **only when `b` is even** (`d=1024=2^10`
  ✓). When `b` is **odd** — DOOM runs at `d=8192=2^13` — a single column lands
  the RMS on `2^(q-7)·√2`, an irrational multiple of a power of two. The fp32
  `sqrt` returns a representable, correctly-rounded value, but it is **not a
  power of two**, so `÷rms` and `×gain` become genuine fp32 roundings rather than
  exact exponent shifts (the `1±ε` scaling the cancel-head analysis below warns
  about).
  The fix: use **two equal columns** `= 2^q`. Their combined energy `2^(2q+1)`
  is an *odd* power of two, so `rms = sqrt(2^(2q+1)/2^13) = 2^(q-6)` is again an
  exact power of two. **General rule: one constant column for even-power-of-two
  widths, two equal columns for odd-power-of-two widths** (the minimal
  power-of-two-RMS construction — 8 or 32 equal columns also work but waste
  columns; unequal or non-power-of-two-RMS layouts are *not* bit-exact). Reserve
  the column(s)
  *inside* the existing `d` — their value folds into the embedding table
  (`embed_table[:, j] = 2^q`, replicated across every vocab row so the per-token
  lookup reproduces it at every position; see *the one unavoidable core hook*),
  so the RMS denominator stays `d`, **there is no width blowup**, and matmul
  shapes are unchanged. The two-column layout at 8192 is **confirmed bit-exact**
  by the float-roundtrip prototype (see Evidence).
- **Formula gain (fallback, not used)** `C = sqrt(K/d)` (K = constant energy).
  Bit-exact for exactly-representable values (the calculator's one-hots and
  integers), but ~`6e-8` relative error per norm on *arbitrary* floats (two
  roundings: `÷C` then `×C`). Bounded, not compounding — but it would require a
  per-graph cancel-heads re-validation (see below), which the power-of-two-RMS
  commitment removes.

### Cancel-heads: the reason DOOM needs bit-exactness

The scheduler emits **cancel heads** (`V=identity, O=-identity`, so
`attn_out + skip == 0` algebraically) to zero residual columns
(`torchwright/compiler/forward/scheduler.py:269`,
`AttnHeadOp("cancel", ...)`). These are **not** DOOM-specific — any compiled
graph that zeroes columns uses them, the calculator included. The sensitivity is
documented at `torchwright/compiler/components/attn.py:18–22`: a single fp32-LSB
perturbation leaves a ~`1/512` residual leak that flips Gray-code bits.

**Mechanism, bit-level (why a non-exact norm breaks it):** in pre-norm a cancel
head computes `attn_out ≈ -norm(res)` against the skip `res`, so the layer
output is `res - norm(res)`. A bit-exact norm makes this exactly `0`; a
formula-gain norm scales by `1±1LSB`, making it `-res·ε ≈ -res·1e-7` — exactly
the leak `attn.py` warns about. The calculator survives the formula gain only
because its values are exactly representable (`ε = 0` for one-hots/integers);
DOOM's genuine floats (reciprocals of distance, trig) make `ε ≠ 0`. The
power-of-two-RMS gain makes `ε = 0` for *all* floats, so cancel-heads survive
untouched everywhere.

## Evidence (prototypes under `scripts/`)

> **Reproducibility note:** both prototypes have been **re-run against post-merge
> `main`** (current `compile_headless` / `HeadlessTransformer` surface, `<bos>`
> token spelling). Numbers below are from those runs and reflect the committed
> power-of-two-RMS, reserve-inside design.

`scripts/proto_rmsnorm_float_roundtrip.py` — DOOM-like float stream (distances
~±2000, reciprocals ~1e-3..1e-4, trig, mid-range, zeros), pure fp32 (the pow2
claim is platform-independent), **power-of-two-RMS gain only**, **reserve-inside**
(constant column(s) inside a power-of-two `d`, no width pad):
- `d=1024` (even `b`, one column) **and `d=8192` (odd `b`, two equal columns —
  DOOM's shipping width)**: `rms_spread = 0`, `max|Δ| = 0`, `drift@200 = 0` —
  **bit-exact, with `eps=0` and `eps=1e-6`.**
- Energy sweep at `d=8192`: bit-exact while `Σ data²` stays under the out-energy
  bound (and ~15× past it, on the `sqrt`-rounding margin); the identity breaks
  once the data energy is well over the bound — confirming the constraint is real
  and where it bites.

`scripts/proto_rmsnorm_identity.py` — full `calculator_simple` compile, baseline
vs. pre-norm forward with the pow2 gain seeded into free columns **inside** `d`,
through the scheduler's real cancel heads:
- `d=1024` (even `b`, one column) and `d=2048` (odd `b`, two equal columns):
  `rms_spread = 0`, decode identical, `max|Δ|` only at the fp32 denormal floor
  (~1e-38..1e-45) — the documented near-zero-data caveat, not drift; cancel-heads
  survive. (`d=8192` itself is too memory-heavy for the dense in-process backend's
  `O(d²)` attention matrices, so the real odd-`b`/two-column graph runs at
  `d=2048` here and at `d=8192` in the roundtrip.)

**Evidence-gap status (the three hardening actions are now done):**
1. **Reserve-inside layout — closed.** Both prototypes reserve the constant
   *inside* a power-of-two `d` (no pad to a larger width); bit-exact above.
2. **Deepest-layer energy — exercised; one piece deferred.** The roundtrip energy
   sweep validates the out-energy bound with margin, and the compiler proto runs
   the calculator's real (deep-layer) residual energy bit-exactly. DOOM's *actual*
   deepest-layer energy is exercised only by the real DOOM graph (Open questions,
   DOOM validation).
3. **Odd-power-of-two width — closed.** The two-equal-column rule is bit-exact at
   `d=8192` (roundtrip) and survives real cancel heads at `d=2048` (compiler).

## Design decisions (LEANINGS unless marked)

1. **Emission feature, ONNX-first, with a hard converter gate.** The norm is an
   identity that reuses existing weights, so adding it is a *rendering* choice.
   But after the merge the emitters are **not** parallel siblings — there is a
   dependency order:
   - `compile_to_onnx` (`export.py`) is the **source of truth**.
   - The HF model is **converted from the ONNX artifact** by
     `convert.py:convert_onnx_to_hf` (it reads ONNX initializers → HF state
     dict), and `modeling_torchwright.py` **transcribes** the ONNX forward
     one-for-one.
   - `convert.py:_assert_all_mapped` (lines ~223–236) **raises** if any ONNX
     initializer is unmapped. So if the norm adds a per-layer initializer (a
     gain vector), the converter *must* map it and the shipped model *must* have
     a matching parameter — or HF conversion fails loudly.

   Order of work: ONNX emission → teach `convert.py` → shipped HF model + config.
   The **core** (`compiler/forward/`: allocator, scheduler, weight-writer) stays
   "the math" save for the one conditional hook below.
2. **Default-on for the standard-transformer artifacts (ONNX → HF); off for
   in-process/debug.** The value of the norm is "looks like a real transformer,"
   which is the ONNX/HF surface. The in-process `HeadlessTransformer.forward`
   (`compiler/transformer.py`) and the debug paths **skip** it — it's a no-op, so
   debugging the math loses nothing, and this insulates the tolerance-sensitive
   test suite. Because the feature is a toggle, existing compiles (norm off)
   reserve no column and change in no way.
3. **[LOCKED] The gain is a real saved `weight` parameter, named the Llama3 way.**
   Emit one per-norm (per-layer + final) RMSNorm `weight` initializer of width
   `d`, uniformly `C`, mapped by `convert.py` to the standard Llama3 names —
   `model.layers.{i}.input_layernorm.weight` (pre-attention),
   `…post_attention_layernorm.weight` (pre-MLP), and `model.norm.weight` (final).
   This is exactly how a stock Llama3 stores its RMSNorm gains; the alternative
   (synthesize the uniform gain from config at load, no initializer) is smaller
   but non-standard. Chosen because the shipped model must be as Llama3-like as
   possible — the *magnitude* of `C` stays the unavoidable tell (the credibility
   caveat above), but the *representation* is identical to Llama3's.
4. **The constant's value folds into the embedding table; only its column
   reservation is compiler-internal.** The value is written directly into
   `embed_table` (replicated across vocab rows — see decision 6); it is the only
   seed constant. What is *not* a graph value is the never-freed column
   reservation: a `LiteralValue` with no consumer would be reclaimed
   (`graph_analysis.py:202`; see *the one unavoidable core hook*).
5. **Config impact.** `TorchwrightConfig` currently lists normalization among
   "invariants this config does NOT carry a knob for" (lines ~18–28). Adding the
   norm means new fields (at least `rms_norm_eps`; possibly a norm-on flag) and
   rewriting the "no normalization" docstrings in **both** shipped files. The
   shipped files are **hermetic** (torch + transformers only, enforced by
   `tests/hf/test_shipped_model.py`), so the RMSNorm must be a standalone
   `nn.Module` there — no torchwright import.
6. **[LOCKED] Seed the norm constant via `embed_table`; delete the vestigial
   `constant_values` buffer.** Post-merge the token seed is a stock-llama
   `res = embed_tokens(ids) + pos` with the placement scatters folded into the
   weights. There is also a `constant_values` (d,) buffer added at the seed — but
   it is **provably all-zeros**: graph `LiteralValue` constants are *not* seeded.
   The merged "constants as JIT nodes" design materializes each one just-in-time
   into the per-layer MLP `linear2` output bias near its consumer and frees it
   (`graph_analysis.py:202` excludes `LiteralValue` from input nodes;
   `scheduler.py:824` → `weight_writer.py:693`; `test_literal_jit.py:96` asserts a
   literal is absent from `in_state`). So the export fill loop's `LiteralValue`
   branch (`export.py:1163–1165`) is dead code, and the buffer is zero
   (empirically confirmed: nonzero count 0). The change is therefore **two
   independent pieces**:
   - **Delete `constant_values`** — a norm-independent cleanup of an always-zero
     dead buffer; dropping an all-zero `Add` is bit-exact *today*.
   - **Seed the norm's pinned constant** into `embed_table[:, j] = 2^q`
     (replicated across every vocab row so the per-token gather reproduces it at
     every position). This is the *only* genuine new seed constant; the graph's
     real constants live in the MLP bias and are untouched.

   Bit-exact: the constant lands in a disjoint reserved column, so `gather + pos`
   equals today's `gather + pos + (zero) constant_values` (HF parity
   `max_logit_diff == 0` is the gate). `lm_head` is built from the compact
   real-feature table and is zero off the output columns, so the finite `2^q`
   multiplies to 0 and never reaches the logits. Storage: HF
   `embed_tokens.weight` is already dense `(vocab, d)`, so seeding adds nothing
   and the buffer is *removed*; the ONNX COO init grows by `vocab × (1 or 2)`
   nonzeros. **Atomicity:** exporter (`export.py`), converter (`convert.py`
   mapping + `_assert_all_mapped`), HF buffer (`modeling_torchwright.py`), and the
   meta format must change together, or conversion/load fails before the norm is
   reached. Caveat: deleting `constant_values` changes the just-landed `fa49576`
   seed for *all* token models (norm-off included) — worth a glance from its
   author.

### The one unavoidable core hook

**Seeding the norm constant is folding it into the embedding table** (decision
6): write `2^q` into `embed_table[:, j]` for every vocab row, so the per-token
gather reproduces it at every position. `embed_table` is already built
`(vocab, d)` (`export.py:1199–1200`) and gathered straight into the seed
(`export.py:1244–1245`); the constant is one more populated column — no bespoke
buffer, no new initializer, and the converter gate (`convert.py:_assert_all_mapped`,
~223–236) stays satisfied because nothing new is introduced on the constant path.
(The graph's *own* constants are not seeded — they live in the per-layer MLP
bias; see decision 6.)

What is **new** is **not** the reservation primitive — `ResidualStreamMap.reserve()`
already exists (`residual_map.py:114`), removes columns from the free pool
permanently, and the never-freed/never-allocated guarantee is enforced by
`_check_invariants`. It currently has **no callers** (the overlay user it was
built for is gone on this branch). What is missing is everything *around* it: a
caller from the norm path; **recording the reserved `(column, value)`** so the
exporter can fold it (today a reserved column is in neither `_free` nor any
node's residual assignment, so the export loop's `get_nodes(in_state)` cannot see
it); and routing the in-process/debug path around it. Folding changes *seeding*;
this is *lifetime + metadata*. The split:
- **Core (small, conditional hook):** when the norm is requested, call
  `reserve()` for the column(s); compute `C`; record the pinned `(column, value)`
  on the compiled artifact. Off by default, so existing compiles are unaffected.
- **Emission (the bulk):** seed the pinned norm constant into `embed_table`,
  delete the `constant_values` buffer and bump the meta format, insert the norm
  op in the ONNX emitter, transcribe to the shipped HF model, teach `convert.py`.

## Constraints (state upfront)

- **[LOCKED] Pin the constant energy for an exact power-of-two RMS.**
  Bit-exactness needs the forced `rms` to be an exact power of two. Pin the total
  constant energy to `d · 2^(2m)`: **one** constant column `= 2^q` for
  even-power-of-two widths, **two equal** columns for odd-power-of-two widths
  (DOOM's `d=8192=2^13` needs two — see *Two gain variants*). The width must be a
  power of two (so `E/d` is a power of two); the compiler does not require this
  today (`d` only needs `d % d_head == 0`, `compiler/components/attn.py`), so the
  norm path must assert `d` is a power of two and pick the column count from the
  parity of `b = log2(d)`. A non-power-of-two width is unsupported and must raise.
- **The constant must out-energy all live data, quantified.** For the mean of
  squares to round to exactly `2^(2m)`, the total data energy must satisfy
  `Σ data_i^2 < E · 2^-24`, where `E = d·2^(2m)` is the pinned energy (the fp32
  mantissa is 24 bits; this is roughly "pinned energy exceeds total data energy
  by a factor of `2^24 ≈ 1.7e7`"). Pick the constant with margin against the
  **deepest-layer** energy. Example at `d=8192` with two columns `= 2^30`:
  `E = 2^61`, `rms = 2^24` (gain ≈ `1.7e7`), tolerates `Σ data² < ~1.4e11`; well
  clear of fp32 overflow at `~3.4e38`. The `2^-24` half-ULP is against the
  *final* `E`; a reduction passing through a partial sum of one constant column
  (`2^60`) has a tighter half-ULP (`2^36 ≈ 6.8e10`), so the safe ceiling is
  reduction-order-dependent and up to ~2× below the headline figure — the
  runtime's RMSNorm reduction order is implementation-defined, so budget against
  the tighter bound. Raise the constant if a deep layer's energy approaches it.
- **RMSNorm, not LayerNorm** — LayerNorm subtracts the mean, which the large
  constants would dominate and which would shift every data column.
- **The final norm reads every column, so scratch must stay finite.** Unlike the
  unembed (`res @ lm_head.T`, which contributes only the output columns), the
  final norm computes `mean(x²)` over **all** `d` columns — a single Inf/NaN in
  any freed/scratch column would poison `rms` and therefore *every* logit (a
  strictly broader blast radius than today; the exporter already flags the finite
  requirement at `export.py:1262–1264`). The pinned constant is finite, but the
  norm path must not let non-finite values reach reserved/scratch columns.
- **Shipped-file hermeticity + converter gate** — restated from decisions 1/5:
  the shipped RMSNorm is torch-only; the gain vectors are the only new ONNX
  initializers and must be mapped in `convert.py` or conversion fails. The
  constant introduces no initializer (it folds into `embed_table`, decision 6),
  and `constant_values` is removed.

## Open questions

1. **Gain representation — RESOLVED** (decision 3, [LOCKED]): a saved Llama3-style
   `weight` parameter (`input_layernorm` / `post_attention_layernorm` /
   `model.norm`), emitted as an ONNX initializer that `convert.py` maps.
2. **Allocator fork:** `ResidualStreamMap.reserve()` already exists
   (`residual_map.py:114`) but is unused; the norm constant is the only thing
   needing a never-freed reserved column (graph constants are JIT-materialized
   into the MLP bias, not reserved). Does that reservation stay a norm-special
   case or become a general "pinned column" primitive — and either way it needs a
   value-surfacing path so the exporter can fold the reserved column. Decides
   whether this is real allocator machinery or an emission-time one-off.
3. **ONNX scope / debug interaction:** ONNX gets the norm (it must, since HF is
   converted from it). Verify the added ops don't perturb `OnnxDebugSession`'s
   structural fingerprint or its per-layer self-consistency check — pre-norm
   should leave the residual snapshots un-normed, but confirm rather than assume.
4. **DOOM validation:** confirm bit-exactness on the *actual* DOOM graph (in the
   `torchwright_doom` submodule, not tested here) by diffing a rendered frame
   against baseline — especially the cancel-heads paths and the deepest-layer
   energy bound.
5. **Validate on the shipping graph.** The prototypes are now re-run post-merge in
   the reserve-inside layout at even and odd widths (Evidence above), but they
   compile `calculator_simple`; production HF export uses `calculator_v2`
   (`examples/calculator_hf_export.py`). Re-confirm bit-exactness on the graph
   that actually ships.

## Out of scope / not done

- Wiring the norm into the ONNX emitter, `convert.py`, and the shipped HF model.
- Seeding the norm constant via `embed_table` and deleting the vestigial
  `constant_values` buffer (decision 6), with the HF parity test as the gate.
- The allocator "reserved column" support and its conditional core hook.
- New `TorchwrightConfig` fields and the "no normalization" docstring rewrites.
- Validation against a real DOOM frame.
- Only the two prototype scripts exist; no production code changed.
