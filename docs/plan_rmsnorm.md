# Plan: RMSNorm as a compiled-transformer identity

**Status:** design exploration. Decisions below are **leanings, not locked**
(except where noted). Prototyped on `calculator_simple`; not yet wired into any
emitter. Branch: `worktree-rmsnorm-prototype`.

## Goal & motivation

Give torchwright-compiled transformers a *real* RMSNorm, so the artifacts they
emit are architecturally standard (a stock decoder-only transformer that runs
on HuggingFace / standard inference engines). The two driving users are the
**calculator** (calculator blog post) and **DOOM** (DOOM blog post).

The norm computes **nothing** — by construction it is the identity. Its entire
value is faithfulness/credibility: today the compiler emits "no normalization
anywhere," and a skeptical reader sees that as skipping the hard parts.

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
   compiled weight is reused unchanged**. The constant column passes through
   each norm unchanged and is never written by any sublayer, so it re-pins the
   RMS at every layer.

### Two gain variants

- **Formula gain** `C = sqrt(K/(d+k))` (K = constant energy). Bit-exact for
  exactly-representable values (the calculator's one-hots and integers), but
  ~`6e-8` relative error per norm on *arbitrary* floats (two roundings:
  `÷C` then `×C`). The error is **bounded, not compounding**.
- **Power-of-two-RMS gain** `C = 2^p`. Choose the constant `= 2^q` and a
  power-of-two total width so the forced RMS is exactly `2^p`. Then `÷C` and
  `×C` are pure exponent shifts — **bit-exact for any float**. Required for
  DOOM (see Evidence). Reserve the constant *inside* the existing power-of-two
  `d` (e.g. one column of `d=1024=2^10`) so there is **no width blowup** and
  matmul shapes are unchanged.

## Evidence (prototypes under `scripts/`)

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

### Why DOOM specifically needs power-of-two

DOOM has genuine floating-point arithmetic (reciprocal of distance, etc.), so
the calculator's free bit-exactness does not transfer. The formula gain's
~`6e-8`/norm perturbation is exactly the fp32-LSB scale that the HF modeling
notes say breaks the **cancel-heads trick** (`attn_out + skip == 0`
algebraically). Power-of-two-RMS keeps the norm bit-exact, so cancel-heads
survives untouched.

## Design decisions (LEANINGS — not locked)

1. **Emission-layer, not core.** The norm is an identity that reuses existing
   weights, so adding it is a *rendering* choice — it belongs with the emitters
   (`HeadlessTransformer.forward`, `compile_to_onnx` in `export.py`,
   `modeling_torchwright.py`), alongside existing emission-time variants
   (ONNX-vs-HF, windowed-vs-unbounded cache, `scale=1.0`, cancel-heads). The
   **core** (`compiler/forward/`: allocator, scheduler, weight-writer) stays
   "the math" and ignorant of the norm. Rationale: an always-on identity baked
   into the core is permanent dead-weight surface area touching every future
   compiler change, every debug probe, and every test.
2. **Default-on** for the artifacts whose value is "looks like a real
   transformer" (HF; probably ONNX for parity). The in-process / debug path can
   **skip** it — it's a no-op, so debugging the math loses nothing, and this
   insulates the existing tolerance-sensitive test suite.
3. **Constant column is compiler-internal**, not a graph/embedding value. (See
   the allocator note below for *why* — it can't honestly be a graph node.)
4. **Power-of-two `d` as a commitment** if we want bit-exactness always.
   `d=1024` already qualifies; DOOM picks a power-of-two `d`. Non-power-of-two
   `d` forces the formula gain (~1e-7/norm) and a cancel-heads re-validation.

### The one unavoidable core hook

Seeding the constant is **free and already exists**: the preamble adds a dense
`constant_values` (d,) vector (`export.py` ~1694–1773), populated from graph
`LiteralValue` nodes. A constant column is just another entry there.

What is **new** machinery: the column must be **permanently reserved,
never-freed, never-written** across the whole network depth. Normal nodes
(including `LiteralValue`s) are freed when their consumers finish; a constant
with no graph consumer would be reclaimed immediately and never survive to
layer 1. So the reservation is inherently a core/allocator concept — this is
why the constant cannot honestly live in the graph/embedding. The split:
- **Core (small hook):** reserve + write-protect one column; compute `C`.
- **Emission (the bulk):** seed via `constant_values`; insert the norm op in
  each emitter.

## Constraints (state upfront)

- Bit-exactness requires **power-of-two residual width**.
- The constant must **out-energy all live data** (graph-dependent; size `2^q`
  above the largest residual magnitude squared — `2^30 ≈ 1e9` covers values to
  ~1e5 with margin; well clear of fp32 overflow at `~3.4e38`).
- **RMSNorm, not LayerNorm** — LayerNorm subtracts the mean, which the large
  constants would dominate and which would shift every data column.

## Open questions

1. **Allocator fork:** does "permanently-reserved column" stay a norm-specific
   special case, or become a general "pinned column" primitive? Decides whether
   this is real allocator machinery or an emission-time one-off.
2. **ONNX scope:** does the ONNX export get the norm too (for HF↔ONNX parity),
   or only HF?
3. **DOOM validation:** confirm bit-exactness on the *actual* DOOM graph (in the
   `torchwright_doom` submodule, not tested here) by diffing a rendered frame
   against baseline — especially the cancel-heads paths.

## Out of scope / not done

- Wiring the norm into the ONNX and HF emitters.
- The allocator "reserved column" support.
- Validation against a real DOOM frame.
- Only the two prototype scripts exist; no production code changed.
