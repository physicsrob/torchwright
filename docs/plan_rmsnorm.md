# Plan: RMSNorm as a compiled-transformer identity

**Status:** design exploration, hardened against post-merge `main` (the merge
that added the HuggingFace export layer `torchwright/compiler/hf/` and removed
the windowed KV-cache). Decisions below are **leanings, not locked**, except the
two marked **[LOCKED]**. Prototyped on `calculator_simple`; not yet wired into
any emitter. Branch: `worktree-rmsnorm-prototype`.

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
model code (`torchwright/compiler/hf/modeling_torchwright.py:29`) and config
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

1. **Pin the RMS.** Reserve a residual column holding a large constant. Its
   squared magnitude dominates the sum-of-squares, so `rms` becomes a
   position-independent constant `C`. In fp32 the data energy falls below the
   constant's LSB, so `rms` is **bit-exactly** constant (measured:
   `rms_spread = 0`).
2. **Cancel it with the gain.** Set the gain uniformly to `C`. Then
   `x_i / rms * C = x_i` — the norm is the identity, and **every existing
   compiled weight is reused unchanged**. The constant column is seeded once,
   passes through each norm unchanged, and is never written by any sublayer, so
   it re-pins the RMS at every layer (including the final norm).

**Insertion points (pre-norm).** Each emitter currently produces, per layer,
`res = res + attn(res)` then `res = res + mlp(res)` with no norm
(`export.py:_emit_cached_layer_nodes` lines ~822–938; `modeling_torchwright.py`
`TorchwrightDecoderLayer.forward`). Pre-norm inserts the norm on the *input* to
each sublayer only: `res = res + attn(norm(res))`, `res = res + mlp(norm(res))`.
The skip term stays the un-normed `res`, so the residual stream itself is never
overwritten with normed values. A **final norm** is applied to the last hidden
state before the output gather/unembed (`export.py` lines ~1249–1257;
`modeling_torchwright.py` lines ~397–398), matching a standard Llama-style
decoder. **[LOCKED: include the final norm.]**

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

We **[LOCKED: require an even power-of-two residual width]** (see Constraints),
which makes the power-of-two-RMS gain bit-exact for *every* graph. The formula
gain is recorded only as the fallback this commitment lets us avoid.

- **Power-of-two-RMS gain (committed)** `C = 2^p`. With residual width
  `d = 2^b` and `b` **even**, set the constant column `= 2^q`; the forced RMS is
  `rms = sqrt(2^(2q)/2^b) = 2^(q - b/2)`, an exact power of two precisely
  *because `b` is even*. Set the gain `= 2^(q - b/2)`. Then `÷C` and `×C` are
  pure exponent shifts — **bit-exact for any float** (modulo a denormal-scale
  floor for near-zero data). Required for DOOM (see Evidence). Reserve the
  constant *inside* the existing even-power-of-two `d` (one of the `d=1024=2^10`
  columns), so the RMS denominator stays `d`, **there is no width blowup**, and
  matmul shapes are unchanged.
- **Formula gain (fallback, not used)** `C = sqrt(K/d)` (K = constant energy).
  Bit-exact for exactly-representable values (the calculator's one-hots and
  integers), but ~`6e-8` relative error per norm on *arbitrary* floats (two
  roundings: `÷C` then `×C`). Bounded, not compounding — but it would require a
  per-graph cancel-heads re-validation (see below), which the even-power-of-two
  commitment removes.

### Cancel-heads: the reason DOOM needs bit-exactness

The scheduler emits **cancel heads** (`V=identity, O=-identity`, so
`attn_out + skip == 0` algebraically) to zero residual columns
(`torchwright/compiler/forward/scheduler.py:277`,
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

> **Reproducibility note:** both prototypes predate the post-merge `main`. The
> `<bos` → `<bos>` token rename (`graph/embedding.py:10`) has been applied to
> `proto_rmsnorm_identity.py`; re-running it against the current
> `compile_headless` / `HeadlessTransformer` surface is the first implementation
> step (see Open questions). Numbers below are from the pre-merge runs.

`scripts/proto_rmsnorm_identity.py` — full `calculator_simple` compile, baseline
vs. normed forward:
- 1- and 3-digit add/sub/mul: `rms_spread = 0`, `max|Δ|` at the denormal floor
  (~1e-38..1e-45), decode identical. **Bit-exact.**
- `M` sweep (999*999): bit-exact for `M ≥ 1e6`; still decodes correctly down to
  `M=1e3`; **breaks at `M≈1e2`**, when `C` drops to the data's own RMS. Takeaway:
  the constant must out-energy *all* live data.

`scripts/proto_rmsnorm_float_roundtrip.py` — DOOM-like float stream (distances
~±2000, reciprocals ~1e-3..1e-4, trig, signs, zeros):
- Formula gain: `max_rel ≈ 6e-8` per norm; `drift@200 = 1.2e-4` (bounded — does
  not compound).
- Power-of-two gain: `max|Δ| = 0`, **bit-exact**, even with `eps=1e-6`.

**Two gaps in the current evidence (hardening actions, not yet done):**
1. The pow2 prototype measured bit-exactness at width **4096**, not the
   committed reserve-inside-`1024` layout: it *appends* the constant to a full
   1024-wide data block (width 1025) and pads up to the next even power of two
   (4096). The reserve-inside-`1024` scheme is mechanically sound (1024 is an
   even power of two, denominator stays 1024), but it was never the thing
   measured. Re-run the prototype in the reserve-inside layout and re-measure.
2. Both prototypes stress *input* energy only. RMS is re-pinned each layer only
   while data energy stays swamped at *every* layer, and residual energy grows
   with depth. Stress the constant against the **deepest-layer** residual energy,
   not just the input — this is the real risk for DOOM's depth.

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
   - `convert.py:_assert_all_mapped` (lines ~231–244) **raises** if any ONNX
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
3. **The gain representation is a real saved parameter** (lean): emit one
   per-layer (+ final) RMSNorm `weight` initializer of width `d`, uniformly `C`,
   mapped by `convert.py` to `model.layers.{i}.<norm>.weight`. The alternative —
   synthesize the uniform gain from config at load time and emit no initializer —
   keeps the ONNX graph smaller but makes the shipped model less literally a
   "standard RMSNorm." Decide before implementing (Open questions).
4. **Constant column is compiler-internal**, not a graph/embedding value. (See
   the allocator note below for *why* — it can't honestly be a graph node.)
5. **Config impact.** `TorchwrightConfig` currently lists normalization among
   "invariants this config does NOT carry a knob for" (lines ~18–28). Adding the
   norm means new fields (at least `rms_norm_eps`; possibly a norm-on flag) and
   rewriting the "no normalization" docstrings in **both** shipped files. The
   shipped files are **hermetic** (torch + transformers only, enforced by
   `tests/hf/test_shipped_model.py`), so the RMSNorm must be a standalone
   `nn.Module` there — no torchwright import.

### The one unavoidable core hook

Seeding the constant is **free and already exists**: the preamble builds a dense
`constant_values` (d,) vector (`export.py` lines ~1158–1168, added to the seed
once at ~1235), populated from graph `LiteralValue` nodes; the HF model already
carries it as a saved buffer (`modeling_torchwright.py:203–205`, mapped in
`convert.py:209`). A constant column is just another entry there.

What is **new** machinery: the column must be **permanently reserved,
never-freed, never-written** across the whole network depth. Normal nodes
(including `LiteralValue`s) are freed when their consumers finish; a constant
with no graph consumer would be reclaimed immediately and never survive to
layer 1. So the reservation is inherently a core/allocator concept — this is
why the constant cannot honestly live in the graph/embedding. The split:
- **Core (small, conditional hook):** when the norm is requested, reserve +
  write-protect one column; compute `C`. Off by default, so existing compiles
  are unaffected.
- **Emission (the bulk):** seed via `constant_values`; insert the norm op in
  the ONNX emitter; transcribe to the shipped HF model; teach `convert.py`.

## Constraints (state upfront)

- **[LOCKED] Even-power-of-two residual width.** Bit-exactness needs
  `rms = 2^(q - b/2)` to be an exact power of two, where width `= 2^b`; that
  requires `b` **even**. `d=1024=2^10` qualifies (the default);
  `d=2048=2^11` does **not** (it forces a non-power-of-two RMS and a width pad).
  The compiler does not enforce this today (`d` only needs `d % d_head == 0`,
  `compiler/components/attn.py`), so the norm path must assert it.
- **The constant must out-energy all live data, quantified.** For the mean of
  squares to round to exactly `2^(2q-b)`, the total data energy must satisfy
  `Σ data_i^2 < 2^(2q-24)` (the fp32 mantissa is 23 bits; this is roughly
  "constant² exceeds total data energy by a factor of `2^24 ≈ 1.7e7`"). Pick `q`
  with margin against the **deepest-layer** energy. Example: `q=30` →
  `const² = 2^60`, tolerates `Σ data² < ~7e10`; well clear of fp32 overflow at
  `~3.4e38`. Raise `q` if a deep layer's energy approaches the bound.
- **RMSNorm, not LayerNorm** — LayerNorm subtracts the mean, which the large
  constants would dominate and which would shift every data column.
- **Shipped-file hermeticity + converter gate** — restated from decisions 1/5:
  the shipped RMSNorm is torch-only; every ONNX initializer the norm introduces
  must be mapped in `convert.py` or conversion fails.

## Open questions

1. **Gain representation:** saved per-layer `weight` initializer (decision 3
   lean) vs. synthesized-from-config at load. Decides how much `convert.py` and
   the ONNX emitter grow.
2. **Allocator fork:** does "permanently-reserved column" stay a norm-specific
   special case, or become a general "pinned column" primitive? Decides whether
   this is real allocator machinery or an emission-time one-off.
3. **ONNX scope / debug interaction:** ONNX gets the norm (it must, since HF is
   converted from it). Verify the added ops don't perturb `OnnxDebugSession`'s
   structural fingerprint or its per-layer self-consistency check — pre-norm
   should leave the residual snapshots un-normed, but confirm rather than assume.
4. **DOOM validation:** confirm bit-exactness on the *actual* DOOM graph (in the
   `torchwright_doom` submodule, not tested here) by diffing a rendered frame
   against baseline — especially the cancel-heads paths and the deepest-layer
   energy bound.
5. **Re-run/re-measure the prototypes** against post-merge APIs and in the
   committed reserve-inside-`1024` layout (Evidence gaps 1–2). Production HF
   export uses `calculator_v2` (`examples/calculator_hf_export.py`), not the
   `calculator_simple` the prototype compiles — validate on the graph that ships.

## Out of scope / not done

- Wiring the norm into the ONNX emitter, `convert.py`, and the shipped HF model.
- The allocator "reserved column" support and its conditional core hook.
- New `TorchwrightConfig` fields and the "no normalization" docstring rewrites.
- Validation against a real DOOM frame.
- Only the two prototype scripts exist; no production code changed.
