# RoPE port — plan

Status: **Phases 0–7 done; Phase 8 remaining** (branch `worktree-rope`). Phase 0
(rotary `Attn`, in-process/ONNX/HF), Phase 1 (bucket-1 near-marker count), Phase 1b (recency ramp:
`soft_blend` + octant ramp, confirm-compile green), Phase 2 Part 1 (relative-offset all-Δ + sign
lock), Phase 2 Part 2 (compiler self-match → rotary, all three surfaces), Phase 3 (content-selection
capability on slow planes — proven; builder rewrite landed in Phase 5), Phase 4 (recency end-to-end:
the two graded `{BOS, REF}` rotary heads → octant ramp → ramp-based selection, proven; in-place
builder rewrite landed in Phase 5).

**Design change (2026-06-28, post-Phase-5): the global `{BOS, REF}` octant-ramp recency above is being
superseded** (see Phases 6–8 in §8). It proved the capability but requires an artificial `<ref>` token
and is far heavier than bounded consumers need: new **Phase 6** moves every consumer to a **local**
rotary-lobe recency (no `<ref>`, no ramp, monotone within the lobe), new **Phase 7** redesigns the
unbounded/global case without an injected token (and may live outside the compiler), and the
`torchwright_doom` port becomes **Phase 8**.

**Phase 5 done (2026-06-28, committed on `worktree-rope` as `d9f4272` + `7a92c72`).** All strands complete and the full
`make test` suite is **green** on Modal: Strand A (RoPE-native library) + B1 (PosEncoding deletion),
Strand C (tests/examples migrated, `prefix_*` retired), Strand D (ONNX/HF gate validated end-to-end —
a save→load NaN bug found and fixed), and Strand B2 (**universal full-width rotary, no escape hatch** —
the `Attn.rotary` flag removed, `d_qk == d_head` enforced, wall attention migrated to a slow-plane
rotary content head to keep the DOOM-port derisk valid). Two real bugs were found and fixed with D6
repros along the way: an I1 `const_one` node-id collision and the HF save→load NaN. See the Phase-5
status block in §8. `torchwright_doom` is untouched (Phase 8).

**Phase 6 done (2026-06-29, `worktree-rope`).** Local rotary-lobe recency (`rope_lobe_band` +
`rotary_recency_head`, `W≈414`) replaced the octant ramp; `<ref>`, `recency_heads.py`,
`recency_ramp.py`, `soft_blend`, and `recency_plane_index` deleted; consumers (`attend_most_recent_matching`,
`get_prev_value`, `NumericSequence`, `output_sequence`) dropped `recency_rank`; examples migrated. Full
`make test` green on Modal (all 10 shards). Three bit-exact cross-execution-path parity tests (onnx-vs-torch,
onnxruntime-vs-headless, batched-vs-sequential) were relaxed to bounded fp floors — execution-path rounding
from the graph reshape, generation token-identical, independently reviewed (Claude + Codex); see the Phase-6
status block in §8 and `docs/numerical_noise_findings.md`.

**Phase 7 complete** (commit `d583133`): `global_position_from_bos` + `attend_most_recent_globally`
in `torchwright/ops/global_recency.py`; 7 oracle tests + 3 compiled tests; full suite green.
Candidate (d) — boosted-BOS weight inversion — validated: monotone over [0, 61440], min gap 4.1e-6
(69× fp32 floor), PWL error ≤ 0.09 in float32 (well below the 0.5 rounding threshold),
recency_scale=1.0 gives 26× float32 representability margin at E8 content score scale.

Remaining: **Phase 8** (`torchwright_doom` — rewire the
DOOM call sites, full render parity, and the 42k real-log recency replay). **`main` merged in
(`f281ee8`)**: ONNX/HF now
run in CI (the Modal image carries `onnxruntime`/`transformers` — the runtime-parity gap is closed),
the float-I/O headless ONNX export and the delta-transfer compile mode were removed, and the token
export uses a vanilla untied embedding.

> **RoPE** = rotary position embeddings: instead of adding a position vector to the
> residual stream, attention rotates each query and key by an angle proportional to its
> absolute position, so the pre-softmax logit between two tokens depends on their
> *relative* offset. A **plane** is the 2-D `(cos, sin)` coordinate pair that one rotary
> frequency `θ_i` rotates; a head's `d_head` dimensions are `d_head/2` such planes. `R_N`
> denotes the rotary rotation for a shift of `N` positions (block-diagonal; each plane
> rotated by `N·θ_i`).

## 1. Goal and constraint

Replace torchwright's position mechanism — the sinusoidal trig block **plus** integer
counter column in `torchwright/graph/pos_encoding.py` — with RoPE applied inside attention.

**Motivation (scoped):** architectural fidelity for the DOOM-as-transformer blog, *for the
attention mechanism*. Doing relative-position attention the way a modern trained transformer
does (rotary) is a better story than the bespoke sinusoidal scheme. This is **not** a perf or
correctness fix — today's scheme works.

**Caveat:** the compiler needs position-dependent behavior a language model does not (DOOM
pixel arithmetic, recency). This is **not** a single absolute index — it splits into two
buckets (§3), both graph-derived from in-context tokens, never host-supplied.

**Hard constraint — provenance, not shape.** Position may enter only via rotary attention and
signals the graph *derives* from it. No position-*encoding input* (host-supplied or statically
injected residual feature carrying position) may remain. **The host may not seed position at
any point** — not a first-token literal, not a decode-start seed from the prefill length. A
monotone position scalar the graph computes from attention is compliant; a host counter is
not. (NoPE — dimensions left deliberately un-rotated — also fails this, as the forbidden
middle ground.) During migration some heads are vanilla and some rotary; that scaffold is
removed at Phase 5, when all heads are rotary on one global grid.

## 2. What position does today (the replacement target)

Two independent mechanisms live in `PosEncoding` (`torchwright/graph/pos_encoding.py`):

- **Sin/cos trig block** (cols `0..d_pos-2`): read by attention Q/K for relative-offset
  matching. `trig_shift_matrix(k)` (pos_encoding.py:215) is a static per-frequency rotation
  `[[c,-s],[s,c]]` — this *is* RoPE's 2-D rotation form, applied at compile time to one side.
  The current grid uses the interleaved `(2i,2i+1)` layout at base `10000`; the rotary path moves to
  the LLaMA3 `rotate_half` layout at a LLaMA3-family base (§6 RoPE-convention block), so the port is
  in part a layout + base change.
- **Counter column** (col `d_pos-1`): the raw integer position, read as a *number* for exact
  arithmetic and as a monotone recency rank.

Attention is used in three structurally different ways:

| Class | Heads | Position role |
|---|---|---|
| **Positional** | `attend_to_offset`; compiler-internal self-match via `_current_pos_attn_matrices` (Linear/Add/Cancel/add_into — 4 callers; delta_transfer removed on `main`) | fixed offset / self (Δ=0), trig dot product |
| **Content selection** | `attend_argmin/argmax/_where/_above/_bucket/_unmasked/_dot` family; DOOM scene-fact lookups | none — winner chosen on content, any distance |
| **Content + recency** | `attend_most_recent_matching` / `get_prev_value` family (clip-memory lookup is the hard case) | content gate **plus** monotone counter |

(Head census is sub-agent-sourced; re-confirm exact per-class counts by committed grep before
per-site work — see §9.)

## 3. Target architecture

A single **global rotary frequency grid** `θ_i = base^(-2i/d)` applied to every head's Q/K by
absolute position. A head controls its behavior by *which planes it places energy in*:

- **Offset / self-match (positional class):** `W_K = R_N · W_Q` on a constant feature peaks
  attention exactly N positions back, uniformly over query position. The existing
  `trig_shift_matrix` is this with the rotation pre-baked; the rotary version moves it into the
  kernel. Δ=0 (self-match for the 5 compiler-internal callers) is the trivial case.
- **Content selection:** place content keys on the **slow planes**. A plane applies `R(Δ·θ_i)`
  to its Q/K pair before their dot product; if `θ_i` is tiny and Δ bounded, `R(Δθ_i) ≈
  identity`, so the plane is effectively un-rotated — a content-matching dimension. With base
  `1e6` the slowest planes stay quasi-static out to frame length (§6). This is a property of
  standard RoPE on the global grid, not NoPE.
- **Position-dependent behavior — two buckets, not an absolute integer.** No consumer reads the
  absolute *value* of position, so we never recover a 64k absolute integer.

  - **Bucket 1 — bounded local differences (`1/(gap+1)` from a near marker).** All pixel/row/
    prefix arithmetic is `pos − a recent reference` (wall/flat pixel index = `pos − span_start`;
    weapon/HUD row = `sy_value + (pos − sy_pos)`; `prefix_*` gate on `pos ≥ 2^k`). The
    differences are **bounded by a screen dimension** (< ~350: column height ~100, span width
    ~320, sprite/patch ~50). Mechanism: a `W_Q=0` uniform-attention count over the keys between
    the relevant **recent, graph-identified marker** (the same `span_start` / `sy_pos` the
    consumer already subtracts from) and now gives share `1/(gap+1)`; invert to the small
    integer gap. Resolvable because the gap is bounded; non-recurrent; no rotation, no decode.

  - **Bucket 2 — recency ordering (the octant two-head readout).** `attend_most_recent_matching`
    / `get_prev_value` rank "most recent matching key" via `8 × counter` in the logit. This
    needs a signal **globally monotone over the rollout** with **uniform** resolution: recency
    is load-bearing in 95% of real selections and must split gaps as small as **1 token at any
    absolute position** (§9), so a signal whose resolution degrades or collapses anywhere fails.
    It does **not** need an exact integer (order is all the argmax uses).

    > **⚠ Superseded (2026-06-28) — see Phases 6–7 in §8.** The bucket-2 design below (the global
    > `{BOS, REF}` octant-ramp readout, the `<ref>` token, the seam/leakage/gain calibration) is **no
    > longer the plan of record.** Bounded consumers move to a **local** rotary-lobe recency (Phase 6 —
    > no `<ref>`, monotone within the lobe width `W`); the unbounded/global case is redesigned without an
    > injected token in Phase 7 (and may live outside the compiler). The bucket-2 text from here through
    > §6's recency-signal bullet and §11's recency gates is retained as the **Phase-7 design reference**,
    > not a build commitment.

    *Premise — the softmax is a true (non-PL) softmax on every path.* The recency phase is read
    out of an attention *weight* (RoPE rotates Q/K not V, so the phase lives only in the score).
    That weight is computed by an exact float softmax everywhere — `F.scaled_dot_product_attention`
    on the MATH backend in the in-process component (`components/attn.py`) and the production HF
    model (`modeling_torchwright.py`), and an ONNX `Softmax` op in the export (`export.py`) — never
    the torchwright `exp`+`reciprocal` PL chain. So the weight is ~fp32-exact (~6e-8), and the
    readout's risk is its *shape*, not weight noise.

    **Mechanism:** one rotary plane sized to turn once over the rollout (`θ ≈ 2π/max_positions`,
    seam out of range), read against BOS by **two** position-only heads — a cos-head
    (`score = M·cos φ`) and a sin-head (`M·sin φ`, 90°-rotated query feature) — each as a
    **graded** 2-key softmax weight (`{BOS, REF}`, operated mid-sigmoid so the weight tracks the
    phase). *(The two keys are BOS — the phase carrier — and a second always-visible marked
    reference token REF carrying a constant logit; **not** `{BOS, self}`. Phase 4 found `self` rides
    the same recency plane as BOS and injects a constant `−M·cos ψ` shift that breaks the octant
    cos/sin symmetry — see §8 Phase 4 status.)* At every `φ` at least one of `|sin φ|, |cos φ|` is
    steep, so **select the steep head
    per octant** (via `compare`/`cond_gate` on the two weights), apply a fixed sign, and chain
    octants with constant `add` offsets into one monotone ramp. Slope never drops below the
    `φ = π/4` worst case (`sin45°`), giving **uniform** resolution — ~370× the fp32 noise floor
    (measured, §9: 0 replay flips over 8 seeds, and at 30× the floor). The rank is `G × ramp`,
    `G ~ 2e5` to sit in the same content-dominance band as today's `8 × counter` (`G` is for the
    content balance, **not** resolution — a constant multiplier is argmax-invariant). No
    un-sigmoid, no `atan2`, no decode, no new op.

Net: no sin/cos block, no host-injected position encoding, no recovered absolute integer.
Position survives as (1) bounded near-marker counts and (2) the octant two-head BOS-relative
ramp. The `d_head ≥ trig_width` compile constraint disappears.

## 4. Rejected structures (forward design constraints)

These are dead ends with mechanical reasons; honoring them keeps the design out of known traps.

- **No recurrent `prev+1` carry.** A carried counter (`position(t) = position(t−1) + 1` via a
  look-one-back read) is structurally impossible in a fixed-depth transformer: the KV cache
  exposes a past token's residual only as it stood at the *input* to each layer, but the
  incremented counter is that layer's *output*, one layer higher — so the read layer must be ≥
  the previous token's write layer (visibility) yet < the current token's write layer
  (causality), forcing the write layer up by one per token and exhausting the layers after ~57
  tokens. Every position signal must be derived **non-recurrently** (each token computes its own
  from in-context tokens). This is *why* today's scheme injects the counter from the host.
- **Recency rank must be globally monotone — no wrapped/mod-N counter.** A `pos mod N` counter
  inverts the ranking whenever a comparison straddles a wrap (an older key just below the wrap
  outranks a newer key just above it). The octant ramp turns once over the rollout, no wrap.
- **No recency via the rotary lobe alone.** A direct rotary similarity `Σ_k cos(Δ·θ_k)` is
  non-monotone past a main lobe (`W≈414` measured for base 1e6 / d_head 256 / Hann taper);
  DOOM clip look-backs exceed that. Recency needs a *materialized* monotone signal.
- **Recency readout must have uniform slope — no single cosine, no PL `atan2`.** A single cosine
  weight's slope `∝ |sin φ|` collapses to zero at the turning points, giving only ~6× the fp32
  floor where gap-1 cases live. A continuous `atan2` as a `piecewise_linear_2d` op has PL trig
  noise ~0.1 rad — ~10⁴× the ~1e-5 rad budget. Hence the octant two-head scheme (§3).
- **Bounded-difference counts measure from a near marker, not BOS.** An unbounded `1/(m+1)`
  count to BOS dies to the `1/m²` resolution collapse at large `m`. The near-marker count works
  *because* the gap is bounded.
- **Off-path but validated:** an exact-integer absolute position is recoverable from BOS-relative
  phases via a mixed-radix decode (`scripts/rope_position_decode.py`, proven over 0..64k). No
  consumer needs it, so it is not on the build path — kept only as a pointer should one ever.

## 5. Cache and runtime semantics

- **The cache is slot == position with no sliding eviction** (the windowed/attention-sink
  variant was removed). Production runs the native HF model (`HfTokenRuntime` over an unbounded
  `transformers.DynamicCache`); the ONNX export (`compiler/onnx_load.py` loaders, the
  contract-correctness harness) uses a fixed `S`-slot static buffer where `S = cache_stride`
  (default `max_seq_len`). On both, BOS at slot 0 is retained for the whole rollout (HF: cache
  grows; ONNX: `S = max_seq_len = 61440` covers the run), so the recency read of BOS needs no
  pinning or non-expirable set.
- **K is rotated at write time** by `cache_position` and stored already-phased (slot == position,
  so rotating by slot index = rotating by absolute position; no re-rotation of cached K per
  step). Sites: `export.py` `_emit_cached_preamble`/`_emit_cached_layer_nodes`; the HF model's
  attention; the unbounded `inference/kv_cache.py`.

## 6. Key design parameters (settle during Phase 0–3)

### RoPE convention — LLaMA3-aligned (lock in Phase 0, before `_write_compute_attn`)

The rotary path adopts the **HF LLaMA3 RoPE convention**, not the conventions baked into the
sinusoidal block it replaces. "Compatible direction" means *convention-faithful* (the rotation is
the one a modern trained transformer uses), **not** checkpoint-loadable — torchwright compiles
weights, it does not load LLaMA. Locking these in Phase 0 because the layout and rotation set are
baked into `Attn.compute`, `_write_compute_attn`, the ONNX emission, and the HF model
simultaneously; changing them later re-touches all four.

- **Rotation layout: `rotate_half` (half-split), not interleaved.** HF LLaMA3 splits `d_head` into
  halves `[0:d/2]` and `[d/2:d]` and rotates dim `i` against dim `i+d/2`. The legacy
  `trig_shift_matrix` (`pos_encoding.py:240-243`) instead pairs `(2i, 2i+1)` — the **interleaved /
  GPT-J** layout. The two are equivalent only up to a fixed permutation of head dims; they are not
  the same convention. **Drop the interleaved layout; the rotary path uses `rotate_half`.** The
  bespoke constructions (offset head `R_N`, the recency plane's `(cos, sin)` read) reference whichever
  two dims form a plane, so they port cleanly to half-split — but every weight construction and the
  runtime rotation must agree on the layout. Pin it with the Phase-0 offset-head test.
- **Rotate both Q and K by absolute position; V unrotated.** Q is rotated by the query's current
  absolute position at compute time; K is rotated at write time by `cache_position` and the cache
  stores already-rotated K (this *matches* HF, which applies rotary to `key_states` before the cache
  update — it is not an optimization peculiar to us). The logit then depends on `(m−n)`. The plan
  must spell out the Q rotation, not only the K rotation — omitting it makes the logit depend on
  absolute `n`, the easiest silent error, and exactly what the Phase-0 offset-head test catches.
- **`base` — adopt a LLaMA3-family value, and reconcile with the analyses.** LLaMA3 uses
  `base = 500000`. The recency/lobe analyses (`W≈414`, the octant headroom, the slow-plane
  quasi-static bound) were run at `base = 1e6`. Choosing `base` is now a LLaMA3-alignment decision,
  not a free knob: if `base` moves to `5e5`, re-confirm the slow-plane `Δθ` bound (§6 `base` bullet),
  the recency-lobe `W`, and that the global grid still contains a sub-one-turn recency plane (below).
  The 372× octant headroom is large enough to likely survive, but it must be re-measured, not assumed.
- **Long-context frequency scaling: out of scope, stated.** LLaMA3.1 adds NTK/frequency scaling for
  long context. The port does **not** include it (one fixed grid). This is a deliberate simplification
  from full LLaMA3.1, noted so it is not mistaken for an omission.
- **Grid must contain a sub-one-turn recency plane (torchwright-specific, on the standard grid).**
  The octant readout needs a plane with `θ ≈ 2π/max_positions` (one turn over the rollout, seam out
  of range). This must be an actual frequency on the standard `θ_i = base^(−2i/d)` grid — a joint
  constraint on `base` and `d_head`, not a separately-injected plane. (At `base 1e6`, `d_head 256`
  it is ~plane 85 of 128; re-solve for the chosen `base`.) This is the one place the *usage* extends
  beyond LLaMA — but it rides the standard grid, it does not bend the geometry.

**Old-scheme baggage to drop (do not carry forward):** the interleaved `(2i,2i+1)` rotation layout;
`base = 10000`; the integer counter column and any notion of position as a *gathered/host-injected*
feature (§10 host position table, the `[0,100000]` counter affine bound, the `d_head ≥ trig_width`
constraint). Position under RoPE is a rotation applied inside attention, never a residual feature.

- **`base`** — large enough that the slowest planes are quasi-static for content over the real
  rollout, **and** a LLaMA3-family value (see the RoPE-convention block above — LLaMA3 is `5e5`).
  At base `1e6`, `θ_min ≈ 1.5e-6`, so over `N≈61440` `Δθ ≈ 0.09 rad` (`cos ≈ 0.996`). Confirm
  against the production frame length (~42k tokens); if `base` moves to `5e5`, re-run this bound and
  the recency analyses (they were measured at `1e6`).
- **`d_head`** — number of rotary planes: content discrimination (slow planes), offset matching,
  and the recency signal (one plane, two heads). Quantify the minimum that gives each content
  head its required gap.
- **Plane partition** — which dims carry content (slow) vs offset energy vs the recency plane.
  One global grid; the partition is per-head placement.
- **Recency signal** — the gain·amplitude `M` that keeps both heads graded (mid-sigmoid, no
  saturation), the octant-boundary handoff continuity, and the rank gain `G` (`~2e5`, content
  balance: `match_gain · content_gap > G · span`). **✅ Settled (Phase 4):** `M = 2.0`; the recency
  plane is the fastest grid plane that never wraps over the 61440 cap (`recency_plane_index` → plane
  91 at base `5e5`/`d_head 256`, `θ ≈ 8.88e-5`); the constant reference is a marked **REF** token (DC
  `L=25`) on the slowest plane (127); `G = 2e5`.
- **Marker gate (bucket 1)** — per consumer, the recent marker token, the post-marker attention
  gate for its `1/(gap+1)` count, and the worst-case gap bound.

## 7. Migration principle

Keep the public helper *call semantics* stable so DOOM and the ops layer change minimally, but
**`pos_encoding` drops out of the signatures** — there is no `PosEncoding` node to pass.

- **Symmetry-only callers** — `attend_argmin/argmax/_where/_above_*/_unmasked/_valid_unmasked/
  _mean_where` accept `pos_encoding` today but never read it (explicit "API symmetry"). Drop the
  parameter.
- **Position-aware callers** — `attend_to_offset` (→ pure rotary head), `get_position_scalar` /
  `prefix_*` (→ a bucket-1 near-marker count), and the recency family (→ the bucket-2 octant
  ramp). These reference the new substrate instead of a `pos_encoding` node.

DOOM (`torchwright_doom`) builds no raw `Attn` (re-confirm by grep), so a semantics-preserving
migration leaves the renderer untouched until final validation.

**Working split — `torchwright_doom` untouched through Phase 5.** All Phase 0–5 work happens in
torchwright and is validated by the §9 confidence suite + the ONNX/HF export path (capabilities
calibrated to real DOOM difficulty). `torchwright_doom` is only *read* (the consumer census), never
edited, from a torchwright worktree. Through Phases 1–4 each helper's signature stays stable and the
mechanism swaps *internally* (no call-site churn); **Phase 5 makes the `pos_encoding`-drops-from-
signatures change** — still entirely within torchwright (its own tests and `examples/` migrate in the
same phase). *(Phase-5 status: the signature change — `pos_encoding` → `RopeConfig` — and the
`PosEncoding` deletion are ✅ done on `worktree-rope`; the tests/`examples/` migration is ✅ done.
See §8 Phase-5 status.)* **DOOM consumer edits + render parity are Phase 8** — a separate
`torchwright_doom` branch, coordinated with the Phase-5 torchwright branch via the umbrella pointer
bump.

## 8. Implementation phases (ordered, each independently testable)

Each phase ends with a probe/oracle validation (`probe_compiled` in-process,
`OnnxDebugSession` for the artifact) and a smallest-layer reproducer test (D6). **Rotation is a
per-head capability, not a global switch** — a global flip would rotate today's content
selectors (whose correctness depends on position-free logits) before Phase 3 migrates them. The
flag is removed at Phase 5.

> **Status: DONE (2026-06-27, committed on `worktree-rope`).** Implemented across all three
> runtime surfaces. New `torchwright/graph/rope.py` is the single source of truth (`rotate_half`,
> half-split, `ROPE_BASE=500000`, `apply_rope`, `rotary_offset_head`) imported by both the oracle and
> the in-process component so they can't drift. `Attn(rotary=, rope_base=)` + oracle row-index
> rotation; component rotates in `forward`/`forward_cached` (cache stores rotated K); compiler threads
> it via `_write_compute_attn` → `component.rotary_width`. ONNX export emits `rotate_half` in-graph
> (cos/sin from `cache_position` in the cached preamble, per-layer enable consts, `rope_freq`/
> `rope_split` inits, `rope_base` in the sidecar meta) — the loaders need no change. HF:
> `TorchwrightConfig.rope_base`/`rotary_enable_per_layer`, converter derives them from inits+meta,
> shipped `modeling_torchwright.py` carries inline `rotate_half`.
>
> **Decision the plan didn't anticipate — rotary grid width.** The oracle/in-process path rotates
> over the node's `d_qk` (so *partial-width* rotary works), but **ONNX and HF require full-width
> `d_qk == d_head`** — one uniform `rotate_half` over `d_head`; partial width raises
> `NotImplementedError`. This is the §6 LLaMA3 end-state and is forward-compatible, but "rotary grid
> width" is now an explicit parameter, not the single `θ_i = base^(-2i/d)` the plan assumed.
>
> **Validation.** `tests/compile/forward/test_rope_offset.py` (in-process: oracle == trig-shift,
> `probe_compiled` clean, prefill == unbounded decode) and `tests/hf/test_rope_token.py` (a
> "predict-previous-token" model: config carried, predicts previous, prefill == cached decode
> bit-exact). **Post-`main`-merge (`f281ee8`):** the float-I/O headless ONNX export was removed, so the
> Phase-0 headless-ONNX offset tests are dropped — the token-path ONNX RoPE emission (cos/sin from
> `cache_position` in `compile_to_onnx`, full-width `d_qk==d_head` guard) is now covered by
> `test_rope_token.py`. The Modal image now carries `onnxruntime`/`transformers`, so **all three
> runtime surfaces are CI-tested** and green (full suite 927 passed, 0 errors). The earlier
> "validated locally only / CI-blind" caveat is resolved.

**Phase 0 — Rotary `Attn` capability, end to end.** Add an opt-in rotary mode across the graph
node (`graph/attn.py`, **including its `compute()` oracle path**), compiler (`_write_compute_attn`,
`forward/weight_writer.py`), ONNX emission (`export.py`), the onnxruntime loaders, and the
production HF model (`modeling_torchwright.py`). **Teach the oracle RoPE first:** `Attn.compute`
(attn.py:124) does plain attention with the shift baked into static weights; RoPE rotation is
per-token (by row index) and cannot be a static weight, so `compute()` needs a rotary path
keyed on row index — otherwise `probe_compiled` compares against a wrong reference and every
rotary head looks broken. Rotate **both Q and K** by absolute position (Q at compute time, K at write time by
`cache_position`, cache holds rotated K — §5, §6 RoPE-convention block). **Lock the RoPE convention
first** (§6): `rotate_half` layout (not the legacy interleaved `(2i,2i+1)` pairing), the Q+K
rotation set, and the `base` value — all are baked into `_write_compute_attn` and cannot change
cheaply later. Validation: reimplement `attend_to_offset(-1)` as a rotary head, prove token-identical
selection vs the current trig-shift via `probe_compiled`, and add an invariant test that prefill and
unbounded decode produce identical offset-head logits. The offset-head test pins both the **sign
convention** (`j+N` vs `j−N`) and the **rotation layout** (`rotate_half` vs interleaved would each
peak at a different key).

**Phase 1 — Bounded local differences (bucket 1).** Reimplement `get_position_scalar` and
`prefix_*` on the near-marker `1/(gap+1)` count. Gates: (a) each consumer's marker is a
graph-recognizable in-context token and the post-marker gate is constructible; (b) the
worst-case `gap` bound per consumer holds and `1/(gap+1)` resolves at that bound (`1/350` vs
`1/351` ≫ noise). Validation: the §9 bucket-1 confidence test — a torchwright-only graph with an
in-stream marker and value `= pos − marker_pos` recovered via the count, vs oracle, pushed to the
~350 gap bound over a deep rollout (no `torchwright_doom`).

> **Status (2026-06-27): core capability built + proven; two follow-ups remain.**
> - ✅ **The near-marker count capability** (`torchwright/ops/marker_count.py`,
>   `count_since_marker`). `attend_mean_where` over the window `[marker, now]` reading a marker
>   one-hot gives the uniform mean `1/(gap+1)`; `reciprocal` inverts to `gap+1`; subtract 1. It is
>   **RoPE-clean** — `attend_mean_where` uses a literal query + validity-driven keys with no position
>   term, so position enters only via the caller's `window_validity`/`marker_onehot` features.
> - ✅ **Gate (b) — resolution at the bound.** The §9 confidence test
>   (`tests/ops/test_marker_count.py`) pushes gap to **350** (mean `1/351`, adjacent gaps ~8e-6
>   apart). Worst compiled error **0.06** (geometric-breakpoint reciprocal, 239 breakpoints,
>   `_RECIP_REL_SAFETY=16`) → rounds to the right integer with **~8× margin**; compiled matches the
>   oracle. Empty-window queries degrade to bounded garbage (`~max_gap`), not an error — the caller
>   must only consume the gap where the marker is visible (documented contract).
> - **Census (committed grep, 2026-06-27) — settles scope.** `prefix_sum`/`prefix_and` are used
>   **only in `examples/`** (balanced_parens, sort_digits, token_balance) — **not** in
>   `torchwright_doom` and not in core torchwright. `torchwright_doom` reads position at exactly **one**
>   site: `render_main.py:262` `pos.get_position_scalar()`, used for `pixel_index = pos −
>   span_v0.pos − 1` — i.e. *exactly* the bucket-1 marker-relative count (marker = the span's vertex-0
>   publish), which `count_since_marker` serves directly. So the port-relevant bucket-1 deliverable is
>   **done**.
> - **`prefix_*` is out-of-scope for the port (examples-only) — ✅ RETIRED in Phase 5.** It gates on
>   an *absolute* `pos ≥ 2^k` threshold (k up to ~16), which is **not** a bounded near-marker
>   difference and would need its own RoPE-native OOB detector. Since no DOOM consumer uses it, that
>   detector is **not** built and the demos are **retired** rather than migrated. Phase 5 deletes
>   `ops/prefix_ops.py` and the four `prefix_sum` users — `balanced_parens`, `token_balance`,
>   **`sort_digits_v2`, `sort_digits_v4`** (the actual call sites; wider than this note originally
>   implied) — plus their tests. (A count-to-BOS boolean `compare(1/(pos+1), 1/(2^k+1))` resolves
>   small k but hits the `1/m²` collapse for large k — confirming this needs a different mechanism,
>   deferred had we kept them.)
> - **DOOM consumer rewiring is Phase 8.** The one DOOM `get_position_scalar` site rewires to
>   `count_since_marker(span_v0_marker, …)`; per §7 (DOOM untouched through Phase 5) that lands at
>   Phase 8, where gate (a) — the marker is graph-recognizable + the post-marker window is
>   constructible — is verified against the real span-v0 publish. (Phase 5 lands the torchwright-side
>   `get_position_scalar` → `count_since_marker` signature change; Phase 8 is the DOOM call site.)

**Phase 1b — Recency signal (bucket 2): the octant two-head readout.** Build the §3 mechanism.

> **Status (2026-06-27): confirm-compile DONE — the unproven-mechanism crack is closed.** The
> octant recency ramp is built as a graph (`torchwright/ops/recency_ramp.py`, commit `c2ad29d`) on the
> new `soft_blend` op (`102dedb`) and is **strictly monotone on the compiled transformer** (gate-b,
> `tests/compile/forward/test_recency_ramp_compiled.py`, `65089c8`). What remains in Phase 1b is
> wiring the ramp to the *actual* two graded `Attn` heads and gates (a)/(c)/(d) below — that overlaps
> Phase 4 (recency end-to-end); the ramp builder currently takes the two centered weights `u`,`v` as
> abstract inputs.
> - **Codex-review refinement (post-`65089c8`):** `compare`'s ramp is one-sided — `cond=0` lands at
>   `inp = thresh + 1/(2s)`, so with `thresh=0` the soft zone sits entirely on the true side and
>   `soft_blend` is soft where the branches already *differ* (precondition only approximately true).
>   Fixed by shifting each octant test's threshold to `-1/(2s)` so `cond=0` lands exactly on the
>   boundary and the soft zone straddles it symmetrically — `soft_blend` is now fully soft precisely
>   where the adjacent branches are equal. Tests still green (oracle + compiled gate-b).
> - **Assembly math proven.** `scripts/rope_octant_assembly.py` builds the *real* (not idealized)
>   octant ramp from the two centered weights `u=σ(g·cosφ)−0.5`, `v=σ(g·sinφ)−0.5`: 8 octants from
>   `sign(u)`, `sign(v)`, `|u|>|v|`; in each, the steep (nearer-0) weight with a per-octant sign +
>   chaining offset. **Strictly monotone, min step 2.275e-5/token at `g=2.0`, ~190× the fp32
>   weight-noise floor — matches the replay model.** `GAIN = g = 2.0` is the chosen head gain `M`.
> - **Naive `select` build fails at boundaries (the real gate-b finding).** Building the 8-octant
>   tree with the existing `select` is exact at mid-octant (verified 4.6e-7) but breaks near each
>   boundary: a token's `φ = pos·θ` can land arbitrarily close to a `kπ/4` boundary, where the
>   selection signal (`|u|−|v|`, `u`, or `v`) passes through 0, so the selector `compare` can't
>   saturate to ±1 → `select`'s shared `output_bias=−M` cancellation drives the output to ≈ `−M`
>   (`map_select.py:698`). Compiled dense sweep: ~13% non-monotone. No `compare` sharpness fixes it
>   (a token can land *on* a boundary). This is the genuine unproven-mechanism crack the
>   confirm-compile existed to find.
> - **Fix: a new `soft_blend(cond, t, f)` op (dual-reviewed by Claude + Codex, agreed).** A
>   *bounded crisp handoff switch*, **not** a cond-blender — it passes ~zero resolution through
>   `cond` (the ramp's gap-1 resolution lives in `t`/`f`, read during crisp-cond octant interiors);
>   `cond` is soft only where `t≈f` (proven by construction — the offset chaining makes adjacent
>   octants' branch values *exactly* equal at each shared edge, and each boundary makes exactly one
>   of the three tests soft). Build: `out = median(raw, min(t,f), max(t,f))` = `min(max(raw,
>   min(t,f)), max(t,f))` via the elementwise `min`/`max` ops (`arithmetic_ops.py:359/383`; **not**
>   the static-bounds `clamp`). `raw` reuses **`broadcast_select`'s per-unit-carrier core**
>   (`map_select.py:1043-1067`, 4 units/col, `output_bias=0` → `raw(0)=ReLU(t)+ReLU(f)`, bounded by
>   `|t|+|f|`, **no `−M` dip**) instantiated `n_slots=1` — **not** a helper shared with `select`
>   (whose `−M` core *is* the failure), and **dropping** broadcast_select's crisp-mask tail (the
>   `atol=M·c_tol` union assert at `:1082` and c_tol override at `:1090`, which would fire on soft
>   cond). The `min`/`max` clamp is load-bearing: for same-sign `t,f`, `raw` overshoots the box and
>   is non-monotone; the clamp restores both in-box *and* monotonicity. No activation×activation
>   multiply. Output value-type = `union(t,f)` box (explicit override, applied post-clamp);
>   precondition: `t,f` carry finite value-types (else `_max_abs_or_raise` raises). Consider whether
>   a single-sublayer median-of-three PL construction is cheaper than min(max()).
> - **Remaining build steps:** (1) ✅ **DONE** — `soft_blend` added (`map_select.py`,
>   commit `102dedb`): median-of-three on broadcast_select's carrier core (output_bias=0, no
>   crisp-mask assert tail), clamped by the elementwise `min`/`max`. Op tests
>   (`tests/ops/test_soft_blend.py`) cover crisp→t/f, soft→in-box, `t≈f`→≈t, and the
>   same-sign-overshoot D6 repro; `TargetOp` + `make measure-noise` + findings landed. Measured at
>   **fp32 round-off (1.19e-07 abs)** even at the octant-boundary worst input — exact-by-construction,
>   no PL floor of its own. (The op tests are oracle-level via `.compute()`; the *compiled*
>   monotonicity check is the gate-b sweep below, step 4.) (2) add a checked graph assertion that
>   adjacent-octant branches are equal at all three boundary types (`u=0`, `v=0`, `|u|=|v|`) so a
>   violation is caught at construction — ✅ **DONE** (`_assert_branches_meet_at_boundaries`, a
>   build-time Python check on the offset table at the 7 in-range boundaries); (3) ✅ **DONE** — the
>   7-node octant tree is built with `soft_blend` (`recency_ramp.py`); (4) ✅ **DONE** — gate green.
> - **Gate-b sweep — DONE, with a correction to the plan's wording** (`65089c8`). The compiled ramp is
>   strictly monotone, worst boundary step **~2.19e-5/token** (~183× the fp32 floor), matching the
>   assembly model. **The "sub-token-dense" wording was wrong:** the ramp output is `O(1)` in fp32, so
>   a sub-token step (~few ULP of the output) hits fp32 **output quantization** and reports spurious
>   zero steps that are *not* a real flat spot (m=120 sub-token sampling shows min step 0 for this
>   reason, independent of `compare` sharpness). The physically correct gate samples at **token
>   spacing** (steps ~`θ` are cleanly resolvable) and **sweeps the token-grid phase offset** over
>   `[0,θ)` to cover every alignment of the real token grid relative to each boundary. All offsets
>   monotone ⇒ no real token grid lands in a dip. There was **no real `soft_blend` φ-dip** — the
>   fallback below stays unused.
> - **Fallback (documented only):** convex-multiply `f + ½(cond+1)(t−f)` inside `soft_blend`. Not
>   promoted — it is an activation×activation multiply, and at a boundary `t−f→0` so
>   `dC/dφ = ½(cond+1)(t′−f′)` leans on head derivatives at the crossover too: no guaranteed-smoother
>   φ-slope for the multiply's cost. Use only if the sweep shows a real `soft_blend` φ-dip.

Gates (a)/(c)/(d) are **✅ satisfied in Phase 4** (§8 Phase 4 status), gate (b) in the Phase-1b
confirm-compile above; the gate text is kept for the rationale, with the one correction that the
2-key set is `{BOS, REF}`, not `{BOS, self}` (Phase 4 finding). (a) **BOS attendability** — trivially
satisfied (§5); confirm the recency signal is
identical between prefill and unbounded decode out to the full rollout. (b) **Build the real
two-weight assembly and confirm monotone with margin** — the replay derisk (§9) modeled the ramp
as an *idealized uniform-slope line* (`ramp = c_min·θ·k`, no `compare`/`cond_gate`/`add` ops); this
phase builds the actual assembly and verifies (i) the octant-boundary handoff is *continuous* (no
flat spot / jump). Note the head *slopes* match by construction — the octant switch sits at
`φ = π/4 + kπ/4`, exactly where `|sin φ| = |cos φ|`, so the two heads are equally steep there and a
single constant `add` offset bridges only the *level*. The residual risk is the selection itself: a
`compare`/`cond_gate` on the *difference* of the two weights, which passes through zero at the
boundary — a tolerance-boundary cond (root CLAUDE.md, *FP nondeterminism at tolerance boundaries*)
where PL `compare` noise (~1e-3) and fp32 run-to-run variation dominate, and a misfire selects the
flatter head for a few tokens (a local flat spot / jump). So this gate must specifically: (1) sample
the **boundary-straddling tokens** — those whose `φ` lands within the `compare` `c_tol` band of each
`π/4` switch (the worst case, not random tokens); (2) measure slope *through* each boundary at
**production `φ`-density** (θ set by the production `max_positions`, not the 11.6k frame), confirming
no flat spot exceeds the gap-1 step; (3) repeat the boundary tokens across runs to clear fp32
nondeterminism. (ii) the compiled min slope matches the modeled ~2.2e-5/token, (iii) 0 flips on the
full-frame replay. *Fallback if the hard select misfires at the crossover: a continuous blend of the
two heads near each boundary (weight by closeness to `π/4`) removes the `compare`-at-zero entirely at
some slope cost.* (c) **Graded
head** — confirm a mid-sigmoid 2-key `{BOS, REF}` head is expressible (`Attn` takes arbitrary
Q/K/V, no saturation constraint) and `M` keeps both heads graded. (d) **Leakage budget** — the
readout must be an *effectively 2-key* `{BOS, REF}` softmax, so all other causal keys are suppressed
below the gap-1 signal. With `N` other keys each leaking weight `ε`, total leakage `N·ε` must stay
well below the ~2e-5 gap-1 weight difference (the octant ramp's worst-case per-token step at the
cache cap — 1.98e-5 on the real plane-91 grid); at `N ~ 42k` that forces an exclusion score gap of
≈ `log(N / resolution)` ≈ 22+ logit units (×scale). Pin that number against the fp32 score dynamic
range, and confirm the leakage — *and* its growth with key count (a position-dependent drift riding
on the ramp) — stays below the gap-1 signal. *(Phase 4: the DC gap is `L=25`; REF is the constant
reference, and the deviation from the 2-key ideal does not grow with `N`.)*

**Phase 2 — Relative-offset attention + compiler self-match.** Reimplement `attend_to_offset`
(all Δ) on `W_K = R_N W_Q` (lock the sign convention — `j+N` vs `j−N` is an easy silent flip).
Move the compiler-internal self-match (`_current_pos_attn_matrices`, Δ=0) for **all four callers
— Linear/Add/Cancel/add_into** (delta_transfer was removed on `main`) — to rotary. **High blast radius: every
Linear/Add compiles through this** (these paths are arithmetic transport; 1-LSB attention
perturbations have caused real regressions). Migrate behind the smallest compiler-invariant
tests *before* broad suite runs; compiler invariants I1–I4 must keep passing — any firing is a
D1 stop.

> **Status (2026-06-27).**
> - ✅ **Part 1 — relative-offset, all Δ + sign lock** (`feffb20`,
>   `tests/compile/forward/test_rope_offset_all_deltas.py`). The rotary offset head (`W_K = R_N W_Q`,
>   `rotary_offset_head`) is token-identical to the trig `attend_to_offset` at every *in-bounds*
>   position across the real backward-Δ range (committed grep: −1, −2, −3; plus −5, −8). The
>   out-of-bounds region (target before BOS) is a don't-care fallback the two schemes handle
>   differently and harmlessly, so it's excluded. Sign locked by a directional test + `probe_compiled`
>   on backward/wider/forward Δ. Forward offsets (`+N`) are causally degenerate (`j+N` masked → self),
>   so not claimed trig-identical.
> - ✅ **Part 2 — compiler self-match → rotary (all three surfaces).** The four self-match callers
>   (`_write_compute_linear`/`_add`/`_cancel`/`_add_into` via `_current_pos_attn_matrices`) now build,
>   behind the opt-in `rotary_self_match` flag, a rotary Δ=0 head: query/key read a **reserved
>   constant-1 residual column** projected to `hardness·ones` / `ones` across `d_head`, marked rotary
>   (full-width, `rope_base=5e5`), so the runtime rotation makes `logit(j,i) ∝ Σ_p cos((i−j)·θ_p)`
>   peak at `i=j`. This is the offset head at Δ=0 — the trig self-match already *was*
>   rotary-on-a-constant (trig cols = `R(i)·[1,0,…]`); the migration reads the un-rotated constant and
>   lets the runtime rotate it.
>   - **Reduced blast radius vs the plan's fear.** The const-1 feature is a `LiteralValue([1.0])`
>     input node, and literal input columns are **already** initialised on all three surfaces
>     (in-process `get_input_res_stream`; the ONNX/HF `constant_values` seed). So the "ONNX + HF
>     mirrors" needed **no new emission code** — the rotation propagates via per-head `rotary_width`
>     (Phase-0 infra), and the constant rides the existing seed. New code is confined to
>     `weight_writer.py` (rotary branch + `_mark_rotary`) and `forward_compile` (the flag + the
>     reserved column added to `input_indices`); the flag threads through `compile_headless` /
>     `compile_to_onnx`.
>   - **Concentration proven (D2).** The rotary self-match softmax puts weight **exactly 1.0** on the
>     diagonal to fp32 out to position 61440 (sampled, `d_head` 16/32/256) — translation-invariant
>     (depends only on `i−j`), so transport is bit-identical to a perfect Δ=0 selection. Matches the
>     trig self-match's "diagonal at 1.0 past any realistic length" claim.
>   - **Validation.** `tests/compile/forward/test_rope_self_match.py` (in-process: rotary == trig ==
>     oracle on `add`/`signed_multiply`, prefill==decode, the path is actually taken, odd-`d_head`
>     guard) and `tests/hf/test_rope_self_match_token.py` (ONNX→HF: predict-previous survives, the
>     const-1 column reaches HF `constant_values`, prefill==cached-decode). **Full suite green; I1–I4
>     held** (no D1 stop). `d_head` must be even (rotate_half). **(Phase-5 update:** the in-process
>     `test_rope_self_match.py` was **removed** when Phase 5 deleted the trig self-match path — a
>     `rotary == trig` assertion is no longer constructible. Direct in-process compiled self-match
>     transport is now covered only indirectly (any compiled model; the `const_one` mint in
>     `test_recompile_node_id_collision.py`); cross-surface coverage via `test_rope_self_match_token.py`
>     is retained. **Open decision:** re-add a direct in-process compiled self-match assertion, or
>     accept the indirect + cross-surface coverage.)**
>   - **`assume_zero_init` note.** `compile_to_onnx` defaults `assume_zero_init=True`, which elides
>     some BIRTH-layer cancels, so the rotary-head count on the ONNX/HF path is lower than the
>     `compile_headless` (`assume_zero_init=False`) count for the same graph — the HF test asserts
>     "more than the lone offset head", not a literal count.
>
>   The reserved column is allocated once and never freed (no graph node owns it, no op targets it),
>   so it holds 1.0 unchanged through every layer — the `ResidualStreamMap` "computed-once-then-
>   reserved" pattern §10 anticipated. `pos_encoding` is untouched (other position ops migrate in
>   Phases 3–4); the trig self-match remains the default until Phase 5 makes rotary the only path.

**Phase 3 — Content selection.** Migrate the `attend_*` family to place content on slow planes;
fix `base` and `d_head` (§6). Validation: each selection head against **its own documented score
and validity bounds** (e.g. `_QUERY_GAIN=8`, the `attend_argmin_above_in_bucket` margins) — RoPE
slow planes add distance-dependent cosine attenuation and possible cross-plane mixing that must
stay inside each head's gap.

> **Status (2026-06-27): capability proven; builder rewrite ✅ landed in Phase 5 (this branch, see §8
> status), DOOM wiring to Phase 8.** Like Phases 1 and 1b, this builds and proves the *capability* with
> a torchwright helper + confidence tests; the in-place rewrite of the production `attend_*` builders
> (it needs `d_head` at graph-construction time — see "the architectural finding" below — which arrives
> with the §7 `RopeConfig` signature change) is **done** as of Phase 5: every builder takes `RopeConfig`
> and routes through `rotary_content_head`.
> - **Mechanism.** A content head's logit `Σ_c q_c·k_c` becomes, under the global rotation,
>   `Σ_c q_c·k_c·cos((i−j)·θ_{p_c})`. Placing each content column on a **slow** plane (tiny θ) keeps
>   `cos((i−j)θ)≈1` over the rollout, so the match is effectively position-free — standard RoPE on the
>   global grid, **not** NoPE. New `torchwright/graph/rope.py` helpers: `place_on_slow_planes` (relocate
>   a `(rows, W)` content projection onto the slowest `W` planes — col `d_head/2−1−c`, rotate_half
>   partner left zero so planes don't mix) and `rotary_content_head` (build the full-width rotary `Attn`).
> - **The architectural finding — content heads must be full-width `d_head` rotary, so they need the
>   grid.** Because ONNX/HF require `d_qk == d_head`, a content head is a full-width rotary `Attn` with
>   content relocated onto the slowest planes — it must know `d_head`/`base` at construction. The
>   current `attend_*` signatures carry only `pos_encoding` (no `d_head`), so the in-place mechanism
>   swap can't happen "internally" the way §7 envisioned for Phases 1–4; it lands with the Phase-5
>   signature change (`pos_encoding` → a RoPE config carrying `d_head`/`base`). Phase 3 therefore
>   proves the capability head-by-head; Phase 5 rewires the 12 builders (torchwright); Phase 8 wires DOOM.
> - **`d_head`/`base` settled.** `base = 5e5` (locked, LLaMA3). Key result: the slowest-plane
>   attenuation at 42k (~0.9965) is set by **base, not `d_head`** (θ_min → 1/base as `d_head`→∞), so
>   `d_head` only buys *plane count* for wider content. **`d_head = 256`** (the recency-analysis grid)
>   gives 128 planes — ample for the widest content head (`attend_argmin_above_in_bucket`,
>   `d_qk = 2+n_buckets+n_thresholds`; `attend_argmax_dot`, `d_qk = W`).
> - **Validation — selection survives rotation to the 42k production distance** (calibrated per §9, via
>   single logit rows since a 42k×42k matrix would OOM; `_rotary_logits` is anchored to `Attn.compute`).
>   `tests/compile/forward/test_rope_content_slow_planes.py`: dot-match (W=8, `match_gain=200`) margin
>   **198**; bucket rendezvous (`bonus=256`) margin **253**; the binding fine-score gate (unit Δ at the
>   max `_MAX_SCORE_ABS=120` range, winner far/runner adjacent — worst case) margin **3.8 → 47× softmax
>   concentration**, still a clean pick; typical score ranges give near-nominal `_QUERY_GAIN=8`. Plus
>   `probe_compiled` parity on a compiled rotary content head (rotation wired through the compiler at
>   full-width `d_head`). The recency family (`attend_most_recent_matching`, the one position-dependent
>   head) is Phase 4, not here.

**Phase 4 — Recency end-to-end (validates Phase 1b).** Wire the `attend_most_recent_matching` / `get_prev_value` family onto
the octant ramp and validate the clip-memory lookup against the full-frame log (§9): the rank
stays resolvable against the content gate (winner beats runner-up at gap-1, out to the max
position); the ramp does not wrap; all keys are retained. Re-run the replay on a ~42k
production-length log to confirm the headroom survives the tighter per-token step (it halves;
octant stays ~180×).

> **Status (2026-06-27): capability proven end-to-end; the in-place builder rewrite is Phase 5
> (torchwright), the DOOM wiring + 42k real-log replay are Phase 8 (cross-repo).** Like Phases 1/1b/3,
> this builds and proves the
> *capability* with torchwright-only graphs + confidence tests (`tests/compile/forward/
> test_rope_recency_e2e.py`, 9 tests green on Modal). New code: `recency_plane_index` (`graph/rope.py`),
> `recency_phase_heads` / `recency_rank` (`ops/recency_heads.py`), `attend_most_recent_matching_via_ramp`
> (`ops/attention_ops.py`).
>
> - **The gate-(c) construction the plan left open — and the {BOS, self} dead end it corrects.** A
>   naive 2-key `{BOS, self}` softmax does **not** give the clean `u = σ(M·cosφ)−0.5` the octant ramp
>   requires: `self` rides the *same* recency plane as BOS, so its phase-0 contribution injects a
>   constant `−M·cos ψ` shift that breaks the ramp's cos/sin symmetry (the
>   `_assert_branches_meet_at_boundaries` precondition). The working head reads **two marked tokens**:
>   **BOS** carries recency-plane energy (logit `L + M·cos(jθ+φ0)`) and **REF** (a second always-visible
>   marked token) carries **no** recency energy (constant logit `L`). With `L ≫ ln N` the softmax is
>   effectively 2-key and the BOS weight — read out by giving BOS value 1, everything else 0 — is
>   `σ(M·cos φ)` exactly. **Validated on the compiled path to fp32** (~3e-7) at the real plane out to
>   4096 positions; sin-head is the same with a 90°-rotated query.
> - **Plane sized to the cache *cap*, not the frame.** `recency_plane_index` picks the **fastest** grid
>   plane that still never wraps over `max_positions` with a seam margin: at base `5e5`, `d_head 256`,
>   cap `61440` that is **plane 91** (`θ ≈ 8.88e-5`, one turn ≈ 70761 positions, rollout phase ≈0.87
>   turns). Sizing to the ~42k frame instead (plane 87) **wraps past ~47k** and silently inverts the
>   order — so the cap is the sizing target. DC marker on the slowest plane (127), quasi-static; leakage
>   `L=25` (gate d's "≈22+"), ~23× margin over the gap-1 weight signal **at the 61440 cap** (`61440·exp(−25)
>   ≈ 8.5e-7` vs the ~2e-5 ramp step; ~350× at the ~4k typical length — the cap is the worst case).
> - **Gates.** (a) BOS attendability ✅ — recency rank bit-identical prefill vs unbounded cached decode.
>   (c) graded head ✅ (above). (d) leakage ✅ — head output matches the 2-key ideal to fp32 and does
>   **not** grow with the background key count (`N=256 → 4096`). Chain (`heads → octant ramp`) strictly
>   monotone on the compiled path with `probe_compiled` parity.
> - **Selection.** `attend_most_recent_matching_via_ramp` is the RoPE-native twin of
>   `attend_most_recent_matching` — `pos_encoding`/counter drops out; a `recency_rank` node (the ramp)
>   replaces the counter column. Picks the most-recent content match; **content-dominance bound**
>   `match_gain·dot_gap > rank_gain·rank_range` (≈`2e5·2.06 ≈ 4.1e5` at the default `G=2e5`) — the same
>   invariant the counter op documents, now against the ramp's value swing. **Degenerate finding:** the
>   reference positions (BOS/REF attend only to themselves) carry an **outlier** rank (`0.5` vs the
>   ~0.15 interior at small N), so the content-dominance bound is load-bearing precisely to keep a
>   content-mismatched BOS from winning on its outlier rank. gap-1 concentration: `G·min_step ≈ 4`
>   logits at the cap-density octant-boundary worst case (0.98-hard), ~8 at the typical step — the
>   octant trade-off `G~2e5` accepts (argmax always correct; the *ordering*, hence "0 flips," is
>   `G`-invariant).
> - **Still to do.** (1, **Phase 8**) The **42k real-log replay** with the real ramp needs a
>   ~42k `torchwright_doom` instrumentation log; the committed log is the ~11.6k frame, and DOOM is
>   untouched through Phase 5 (§7), so this lands on the Phase-8 `torchwright_doom` branch. The ramp's
>   monotonicity/resolvability *at cap φ-density* is already proven (`test_recency_ramp_compiled.py` +
>   the analytic gap-1 band test here). (2, **Phase 5 — ✅ DONE on this branch, see §8 status**) The
>   **in-place rewrite** of `attend_most_recent_matching` / `get_prev_value` onto `recency_rank` landed
>   (oracle-validated); `get_prev_value(rope, value, cond, recency_rank)` and the unified
>   `attend_most_recent_matching(rope, …, recency_rank)` are the shipped forms.

**Phase 5 — Reach the global end state (torchwright only; `torchwright_doom` untouched).** All
changes land in torchwright and are validated by the §9 confidence suite + the ONNX/HF export path
— **no DOOM render in this phase.** Three strands:

> **Status (2026-06-28, committed on `worktree-rope` as `d9f4272` + `7a92c72`): ✅ ALL STRANDS DONE — full `make test` green
> on Modal.** Library + PosEncoding-deletion, tests/examples migration, ONNX/HF end-to-end validation,
> and the universal-rotary flag removal are all complete. Tracked as four working strands (A/B map onto
> the plan's strand 1+2; C/D onto strand 3 + the test/example migration):
>
> - **Strand A — RoPE-native library. ✅ DONE, oracle-validated.** New
>   `RopeConfig(d_head, max_positions, base=ROPE_BASE)` in `graph/rope.py` (exported; the concrete
>   form of the §7 "RoPE config carrying d_head/base") replaces the `pos_encoding` parameter.
>   Every `attend_*` builder in `ops/attention_ops.py` now takes `rope` and builds a **full-width
>   rotary-on-slow-planes** head via `rotary_content_head` (the three packed-V/O builders —
>   `_above_integer`/`_unmasked`/`_valid_unmasked` — were decoupled to identity V/O). The
>   counter-based `attend_most_recent_matching` is **deleted**; the `_via_ramp` form is now the single
>   `attend_most_recent_matching(rope, …, recency_rank, exclude_self=…)`. New module-level
>   `attend_to_offset` (wraps `rotary_offset_head`) and `get_prev_value(rope, value, cond,
>   recency_rank)`. `ops/marker_count.py` (`count_since_marker`) and `ops/sequence_ops.py`
>   (`NumericSequence`/`output_sequence`) migrated to `rope` + a threaded `recency_rank`.
>   `ops/prefix_ops.py` **retired**; `create_pos_encoding` → `create_rope_config`. Validated at the
>   oracle level (`reference_eval`): argmax / argmin_where / count_since_marker (gap exact to ±0.02) /
>   recency_rank (strictly monotone) / get_prev_value (multi-fire latch) / most_recent_matching.
> - **Strand B1 — delete `PosEncoding`. ✅ DONE, validated in-process.** Removed across
>   `compile.py`, `transformer.py` (host position table), `weight_writer.py` (the rotary self-match is
>   now the only path — `_current_pos_attn_matrices` and the trig branch deleted; the reserved const-1
>   column is **unconditional**), `components/attn.py` + the two layer groups, `scheduler.py`,
>   `cpsat_scheduler.py`, `graph_analysis.py` (input-node class), `graph_identity.py` (`pos_width`
>   fingerprint term dropped), `affine_rules.py` (`_pos_encoding_rule` + the `[0,100000]` counter
>   bound deleted), `probe.py`, `onnx_debug.py`, `export.py` (the `pos_encoding_full` initializer +
>   `Gather` + `_compute_pos_encoding` + the additive `pos` residual term deleted; const-1 rides the
>   existing `constant_values` seed), `hf/modeling_torchwright.py` + `convert.py` (the
>   `pos_encoding_full` buffer removed). `graph/pos_encoding.py` and the `d_head ≥ trig_width`
>   constraint are **gone**. The whole compiler imports clean and a `compile_headless` +
>   `probe_compiled` run shows **oracle parity** on a compiled rotary content head.
> - **ONNX/HF emission — ✅ VALIDATED.** The `export.py` / `modeling_torchwright.py` / `convert.py`
>   changes are exercised end-to-end on Modal (token model through `compile_to_onnx` → HF; prefill ==
>   cached decode; save→load round trip). A real save→load **NaN bug** was found and fixed: the rotary
>   `rope_freq`/`rope_enable` buffers were `persistent=False` and computed in `__init__`, so
>   `from_pretrained`'s meta-device (`low_cpu_mem_usage`) load left them as uninitialised memory (NaN on
>   the A100 fp32 path). They are now recomputed in `forward` from the serialized config — no buffers to
>   mis-materialise. Pinned by `tests/hf/test_hf_save_load_rope.py` (D6).
> - **Strand C — migrate tests + examples. ✅ DONE.** Tests + `examples/` migrated off `pos_encoding`
>   to `RopeConfig`; the four `prefix_sum` examples (`balanced_parens`, `token_balance`,
>   `sort_digits_v2`, `sort_digits_v4`) and their tests are retired; obsolete `test_pos_encoding.py` /
>   `test_position_scalar.py` deleted. Tail fixes landed: `test_baib_adjacent` d_head 20→32 (must divide
>   d); `test_attn_compute_multiposition` dropped a vestigial PosEncoding `pos` key leaf; the
>   ONNX-meta key set now includes `rope_base`. Also fixed an **I1 regression** the unconditional
>   `const_one` exposed under pytest's node-id-reset + module-scoped re-compile: the compile-minted
>   `const_one` could collide with a graph node id; `forward_compile` now advances the global node-id
>   counter past the graph before minting (`reserve_node_id_above`), pinned by
>   `test_recompile_node_id_collision.py` (D6).
> - **Strand D — validate. ✅ DONE.** Full `make test` **green** on Modal (all shards). ONNX/HF gate
>   green (above). `make measure-noise` not needed — no PL op implementation or breakpoint grid changed,
>   and the committed-noise drift test passes. **I1–I4 held** throughout (no D1 stop).
> - **Strand B2 — universal full-width rotary (no escape hatch). ✅ DONE.** The per-head `rotary` flag
>   is removed from `Attn` (every head rotates) and the in-process path now asserts `d_qk == d_head`
>   (matching the existing ONNX/HF contract), so a NoPE head **and** partial-width rotary are both
>   structurally impossible — universal LLaMA3 rotation, no exceptions. The lone partial-width survivor
>   (`rotary_offset_head`'s `d_qk=8` default) and the V/O-split tests' small/odd `d_qk` fixtures moved to
>   full-width. **Wall attention migrated, not deleted** (`tests/graph/test_wall_attention.py`): the raw
>   geometric content head (was `d_head=3`, content on fast planes — incompatible with rotation) is now a
>   full-width `rotary_content_head` at `d_head=256` with its angular/distance/role content on the three
>   slowest planes (rotation ≈ identity over the small token offsets, so the analytical reference holds
>   unchanged). This **derisks the Phase-8 `torchwright_doom` port**; the prototype's docstring records
>   the open production-difficulty questions (distance normalisation, real angular resolution, Δ-regime,
>   interval-containment vs midpoint-cos) that need the not-yet-existing walls-as-tokens spec.
> - **Strand B2 tail — emission dead-code cleanup + cross-backend parity. ✅ DONE.** With the
>   `Attn.rotary` flag gone, the ONNX/HF emission's now-dead non-rotary plumbing was deleted: the
>   always-1 per-head `rope_enable_q/k` inits + their blend nodes (`export.py`), `rotary_enable_per_layer`
>   (config/convert), the `rotary_width` list (component/weight-writer self-match), and the additive
>   `pos_encoding_full` table — every head rotates unconditionally over `d_head`. This surfaced a parity
>   subtlety: **ONNX-oracle (onnxruntime) ≡ native-HF (torch) bit-exactness is not algebraic.** It holds
>   for every normal-magnitude logit, but rows where the cancel-heads cancel to denormal magnitude
>   (~1e-40) differ by one denormal ULP (~1e-43) — a sub-ULP rounding difference (one backend contracts a
>   multiply-add into an FMA, the other doesn't) that the topology change made onnxruntime's optimizer
>   apply. **Not a compiler bug** (no I1–I4 invariant governs onnxruntime≡torch; meaningful logits and all
>   tokens are bit-identical). The parity tests now compare **teacher-forced** (identical inputs to both
>   backends) and tolerate the denormal floor (`tests/hf/test_convert.py`, mirroring the already-robust
>   `test_calculator_parity`). The earlier "the always-1 enable `Mul` blocked the FMA fusion" story was
>   **wrong** — multiply-by-1.0 is the IEEE identity (`fma(x,1,y)==add(x,y)`); the cause is onnxruntime's
>   topology-sensitive fusion, not the enable node. Follow-up: the rotation is now emitted **directly as
>   `rot`** in both backends (the `src + (rot - src)` reconstruction — once thought load-bearing for
>   cross-backend exactness — was removed after the full suite, including every strict cross-backend
>   `== 0.0` parity assertion, stayed green with the direct form; it bought nothing and only added a
>   `Sub`+`Add` per rotation).
> - **Finding — gain calibration is load-bearing for recency selection.** `match_gain` must be large
>   enough to exclude the BOS/REF **degenerate outlier rank** (≈0.5 vs the ~0.15 interior — a
>   content-mismatched reference token will otherwise win on its rank) *and* `rank_gain` must be large
>   enough to concentrate the tiebreak softmax. The two pull opposite ways; each consumer (examples and
>   DOOM) must size them per-graph, not trust defaults. Smallest-layer repro lives in
>   `tests/compile/forward/test_rope_recency_e2e.py` (the content-dominance bound + the degenerate
>   BOS/REF outlier-rank cases).

- **Rewire the production builders onto the proven capabilities** (each needs `d_head` at
  construction — the §7 signature change). The ~12 `attend_*` content builders → full-width rotary on
  slow planes (Phase 3); `attend_most_recent_matching` / `get_prev_value` → `recency_rank` (Phase 4,
  the `_via_ramp` twin is the shape); `get_position_scalar` → `count_since_marker` (Phase 1). Decide
  the `prefix_*` demos: migrate to a RoPE-native OOB detector or retire (examples-only — §8 Phase 1).
- **Delete `PosEncoding`** (the sin/cos block + counter): the host position table (`transformer.py`,
  the `export.py` `pos_encoding_full` initializer + `Gather`, the HF `modeling_torchwright.py` gather),
  the input-node classification/reservation, and the self-match gatekeeper (`_current_pos_attn_matrices`,
  which simplifies once Δ=0 is rotary identity). Remove the per-head rotary flag (rotation becomes
  global), the `d_head ≥ trig_width` constraint, the counter affine bound, and the `len(pos_encoding)`
  fingerprint term. Larger than a node deletion: the scheduler, residual reservation /
  `ResidualStreamMap`, `write_attn_sublayer`, and runtime input construction all assume a
  `pos_encoding` node (§10). The signature change drops `pos_encoding` from the `attend_*` family,
  `sequence_ops`, `prefix_ops`, `create_pos_encoding`, the `export.py` entry points, the layer
  constructors, and the probe/onnx_debug/graph_identity fingerprint refs; **all torchwright tests and
  `examples/` that pass `pos_encoding` migrate in the same phase.**
- **Validate the migrated chain on the ONNX/HF export path, not just in-process.** Every Phase 1b/3/4
  capability is proven only under `compile_headless` (in-process). The full-width rotary content/recency
  heads — and especially `soft_blend` + the octant tree — have **never been exercised through ONNX/HF
  emission**. Phase 0 proved rotary works on all three surfaces, but the novel recency ops carry more
  emission risk than the content heads; this is the early Phase-5 gate (a token-model export through
  `compile_to_onnx` → HF, prefill == cached decode), *before* any DOOM dependency.

**~~Define the BOS/REF interface here…~~ — superseded by Phases 6–7.** Phase 5 built the `{BOS, REF}`
two-token readout (synthetic `bos_marker`/`ref_marker` in Phase 4; the builder interface in Phase 5).
The `<ref>` token and that global readout are **removed in Phase 6**; the unbounded/global case is
redesigned without an injected token in Phase 7. The note is kept only to mark what Phases 6–7 unwind.

**Phase 6 — Local recency; delete `<ref>` and the global octant mechanism (torchwright only).**
Replace the bucket-2 recency machinery with a **local** mechanism and remove the `<ref>` token. "Most
recent matching key" becomes a content match plus the **natural rotary distance-decay**: `Σ_p
cos(Δ·θ_p)` (the rotary self-similarity between a query and a key `Δ` positions back) is maximal at
`Δ=0` and decreases monotonically over the main lobe, so a nearer key of equal content outranks a
farther one — no BOS-relative phase, no two-token `{BOS, REF}` readout, no octant ramp.

> **Status: DONE (2026-06-29, `worktree-rope`). Full `make test` green on Modal (all 10 shards).**
> - **Mechanism.** New `rope_lobe_band(d_head, base, max_positions, *, theta_max=0.3)` +
>   `rotary_recency_head` in `graph/rope.py`: a constant feature on a Hann-tapered mid-band of planes
>   (disjoint from the content slow planes) gives logit `recency_gain·Σ_p amp_p·cos(Δθ_p)`, peak at
>   `Δ=0`, strictly decreasing over the window `W`. Measured at base `5e5` / `d_head 256` / cap `61440`,
>   `theta_max 0.3`: band = planes 12–96, peak `≈42.1`, **`W = 414`** (breakdown pair Δ=419 loses to
>   Δ=429) — the plan's `~415` holds. Default `_LOCAL_RECENCY_GAIN = 600` (sized to the lobe's flat-peak
>   `Δ=0→1` step so the `exclude_self` predecessor case is hard; `get_prev_value` auto-sets
>   `gate = 4·recency_gain·peak`, always content-dominant).
> - **Deleted.** `<ref>` token (+ `ref_token` from `onnx_load.generate` / the example harness),
>   `ops/recency_heads.py` (phase heads / `recency_rank` / `recency_rank_from_tokens`),
>   `ops/recency_ramp.py` (octant ramp), `soft_blend` (`map_select.py` — its only consumer was the ramp;
>   removed from the noise data + `measure_op_noise.py`), `recency_plane_index` (`rope.py`),
>   `scripts/rope_octant_assembly.py`. `<bos>` stays.
> - **Consumers rewired** (`recency_rank` dropped from signatures): `attend_most_recent_matching`,
>   `get_prev_value`, `NumericSequence`, `output_sequence`. Examples migrated (calculator family, adders,
>   cipher, fibonacci, sort_digits). The scratchpad's column gather moved to content-only
>   `attend_argmax_dot` (each scratch column is written once → unique match, recency-irrelevant, and its
>   width-`N` content would otherwise collide with the lobe band) — this also restored its flat-depth
>   invariant. `scripts/rope_window_frontier.py` / `rope_recency.py` parameterized off the live `base`.
> - **Validation.** New confidence tests: `tests/ops/test_local_recency.py` (lobe monotone-to-`W` +
>   breakdown past `W` + nearest-match selection + the disjointness guard) and
>   `tests/compile/forward/test_rope_local_recency.py` (compiled: immediate-predecessor selection,
>   `probe_compiled` parity, prefill == cached decode). I1–I4 held (no D1 stop). The octant/ramp/soft_blend
>   and `<ref>` tests were deleted; the synthetic-rank `attend_most_recent_matching` block and the Tier-3/4
>   `ref_token` tests were migrated.
> - **One numerical finding (independently reviewed — Claude + Codex).** The graph reshape broke three
>   tests that asserted **bit-exactness across two execution paths** (onnxruntime-vs-torch ~2–5 logits;
>   onnxruntime-vs-in-process-headless ~1.2e-3; batched-vs-sequential ~0.016). All are execution-path fp
>   rounding, **not** logic/compiler bugs — generation stays token-identical and correct, I1–I4 held. Key
>   facts (verified, not assumed): the onnx-vs-torch divergence is **amplitude-independent** (not the lobe
>   magnitude — it's the documented topology-sensitive onnxruntime FMA fusion, now on a normal-magnitude
>   logit); the headless compiled circuit matches the oracle to ~2e-6 (only the real onnxruntime artifact
>   rounds); the batched floor is **gain-independent** and emerges only at full-graph depth (a lone
>   recency *or* content head is bit-exact batched-vs-sequential), from the different SDPA mask/shape of
>   the `n_new=1` vs `n_new>1` paths. Resolved by relaxing those three to bounded fp floors while keeping
>   the token-identical + correct-answer assertions strict; documented in
>   `docs/numerical_noise_findings.md` ("Local-recency cross-execution-path fp floor"). `make measure-noise`
>   showed only SHA + sub-ULP wiggle (no op changed).

- **Delete:** the `<ref>` token and the two-token interface; `recency_phase_heads` / `recency_rank` /
  `recency_rank_from_tokens` (`ops/recency_heads.py`); the octant ramp (`ops/recency_ramp.py`); and
  `soft_blend` (`ops/map_select.py`) if the ramp was its only consumer. `<bos>` **stays** (a natural
  sequence start, not an artificial scaffold); only `<ref>` and the global readout go.
- **Rewire every consumer to local recency.** `attend_most_recent_matching` / `get_prev_value` take a
  local rotary-lobe recency bias instead of a `recency_rank` node (the `recency_rank` parameter drops
  from their signatures and from `sequence_ops` / `NumericSequence`); the calculator examples migrate.
  This is what removes `<ref>` from everything that lands.
- **The breakdown is measured and comfortable — document it precisely (load-bearing).** Local recency
  is monotone only within the lobe width `W` (the largest distance at which a nearer key always beats
  *every* farther key); past `W` the similarity is non-monotone and "most recent" can *silently* pick a
  wrong key. **Target: ~100 tokens.** Measured (`scripts/rope_window_frontier.py`, base `5e5`, `d_head
  256`, Hann taper): **`W ≈ 415`** at the natural plane set — **~4× the target** — with per-step
  resolution at distance 100 of `~2.4e-3` (≈ 4×10⁴ the fp32 weight floor `~6e-8`), so a 100-token
  window is trivially resolvable. **`W` is set by the plane cutoff, not the base** (≈ unchanged from the
  base-`1e6` `W ≈ 414`); restricting to slower planes trades up to `W ≈ 6800` for coarser near-Δ
  resolution. Put the chosen `W` + breakdown in the op docstring; a confidence test exhibits the
  non-monotone region past `W` (so the limit is tested, not a footnote). *(Parameterize `BASE` in
  `rope_window_frontier.py` when Phase 6 lands — it currently hard-codes `1e6`.)*
- **Validation.** The example suite (now `<ref>`-free, green) + a local-recency confidence test that
  proves monotone "most recent" to ~100 tokens **and exhibits the breakdown past `W`**; `probe_compiled`
  parity; I1–I4 hold; D6 repro at the op layer.
- DOOM's *bounded* reads use this (Phase 8); DOOM's clip-memory (look-back `> W`) needs Phase 7.

**Phase 7 — Global most-recent: DONE (commit `d583133`).**
Solved **unbounded** "most recent matching key" — the one place the local lobe is insufficient (DOOM
clip look-backs exceed `W`) — **without an injected `<ref>` token.** Candidate (d) won: boosted-BOS
weight inversion with a 1024-breakpoint PWL inverse and a direct per-position logit tiebreak. Candidates, roughly in order of how clean they
are if they hold:

- **(a) `self` carrying DC-only.** `{BOS, self}` was rejected in Phase 4 because `self` carried
  recency-plane energy, so its logit *moved* with the query position (the `−M·cos ψ` shift) — a moving
  reference. But if `self`'s key is projected to the **near-static DC plane only** (no recency energy)
  it is a true constant `L`, and it is already always-visible via the existing const-1 self-match
  column — **no token at all.** The cross-terms (does the single-plane DC self-match stay clean against
  BOS's recency feature?) need verifying; this is the first thing to test, not a proven answer.
- **(b) A real always-present token** in the stream designated as the reference and projected DC-only
  — REF's *role* without REF the *token*. Clean and certain if the stream guarantees such a token at a
  fixed early position (a DOOM frame header / fixed first tile).
- **(c) BOS-only mixed-radix position decode** (`scripts/rope_position_decode.py`, validated 0..64k) —
  a perfect integer rank from BOS-relative phases alone, no second key. Heavier than the ramp but needs
  no reference token by construction.
- **(d) Boosted-BOS weight inversion.** Place a one-hot feature on the slowest RoPE plane at BOS with
  value `sqrt(ln(MAX_LEN))` and use a matching constant query. All other keys carry zero on this
  dimension. The Q·K score to BOS is then `ln(MAX_LEN)·cos(m·θ_slow)` and all other keys score 0,
  giving softmax weight `w = MAX_LEN^cos(m·θ_slow) / (MAX_LEN^cos(m·θ_slow) + m)`. The minimum
  gap between adjacent-m weights is `≥ 4.9e-6` — about 81× the fp32 weight floor
  (~6e-8). The function is strictly monotone over [0, MAX_LEN] because `θ·m·sin(m·θ)+cos(m·θ) > 0`
  holds whenever `m·θ_slow < π/2`; at the slowest natural plane (θ_slow ≈ 2.22e-6) the maximum
  `m·θ_slow ≈ 0.142`, well inside that bound. The inverse `m = g(w)` is a smooth monotone function
  on [1/2, 1] that can be precomputed and fit as a PWL table. The cosine attenuation factor bakes
  into the PWL — there is no separate correction step. **Shipped:** 1024 log-uniform breakpoints
  (PWL error ~0.009); fp32 softmax noise ~0.006; fp32 ReLU accumulation in the PWL sum dominates
  at ~0.09 at small positions; empirical ceiling ~0.15 — 3.3× below the 0.5 rounding threshold.
  Full-cap monotonicity validated analytically via `scripts/rope_global_recency_validate.py`;
  compiled tests exercise shorter contexts (n ≤ 117).

- **It may live outside the compiler.** It is plausible the global recency index is a host-side /
  post-processing mechanism (clip-memory served by an external index) rather than a compiled per-token
  attention weight. Phase 7 decides this; the compiler-internal octant-ramp work (Phases 1b/4, now in
  git history + the §3/§4 design notes) is the reference design, **not** a build commitment.
- **Deliverable: a targeted demo** proving the capability on whichever mechanism wins — monotone,
  gap-1-resolvable recency out to the 61440 cap, 0-flips on a long replay — decoupled from DOOM, in the
  style of the Phase 0–5 confidence suite.

**Phase 8 — `torchwright_doom`: consumer rewiring + render parity (cross-repo).** A separate
`torchwright_doom` branch, coordinated with the Phase-5 torchwright branch via the umbrella pointer
bump (§7). Rewire the DOOM call sites onto the new signatures (the one `render_main.py`
`get_position_scalar` site → `count_since_marker` against the span-v0 marker; the bounded
recency-family and content call sites onto the **local** mechanism (Phase 6); the clip-memory
look-backs that exceed the lobe onto the **Phase-7 global mechanism**), and validate: **(1)** full DOOM
render matches the reference renderer within the documented op-noise budget; **(2)** the recency
0-flips replay with the Phase-7 global mechanism on a freshly captured ~42k instrumentation log (the
committed log is the ~11.6k frame; this is the only place a 42k real-log replay can run, since it needs
the DOOM renderer). This is the final integration gate for the whole port.

## 9. Validation and evidence

**Strategy — torchwright-internal confidence suite (decoupled from `torchwright_doom`).**
Phases 0–5 are validated entirely inside torchwright, with **no `torchwright_doom` edits**. The
move: treat each DOOM consumer *class* as a torchwright *capability*, and prove that capability
with a torchwright-only test graph that reproduces the class's essential shape **and its real
worst-case difficulty**. If the suite passes, the eventual DOOM migration is mechanical (swap each
consumer onto an already-proven capability), not a question of whether RoPE can do it. The census
of which DOOM call sites map to which capability is **read-only** analysis of `torchwright_doom`
(no edits from a torchwright worktree); consumer edits and full DOOM render parity are **Phase 8** — a
separate `torchwright_doom` branch, coordinated via the umbrella pointer-bump, after the torchwright
branch (through Phase 5) is mergeable (§7). The three capabilities and their calibration targets:

| Capability (DOOM consumer class) | torchwright confidence test | Calibrate to |
|---|---|---|
| **Offset / self-match** (`attend_to_offset`, the 4 compiler self-match callers) | rotary offset head; all-Δ + Δ=0 self-match (Phases 0/2) | token-identical vs trig-shift; I1–I4 hold |
| **Bucket 1 — bounded near-marker count** (`get_position_scalar`, `prefix_*`) | marker token in-stream + value `= pos − marker_pos` via the `1/(gap+1)` count, vs oracle | gap **< ~350**; `1/350` vs `1/351` resolves |
| **Bucket 2 — recency ordering** (`attend_most_recent_matching`, `get_prev_value`) | **local** rotary-lobe "most recent" (Phase 6); global octant ramp superseded → Phase 7 | local: monotone to **~100 tokens** (lobe `W`, measured) + breakdown past `W` exhibited; global: Phase-7 demo to the cap |
| **Content selection** (`attend_argmin/argmax/_where/...`) | each selector under RoPE slow planes, against its own score/validity bounds | real head gaps (`_QUERY_GAIN=8`, the bucket margins) |

**Calibration discipline (the rule that makes this real, not theater):** every confidence test
must encode the *actual worst-case difficulty*, not a toy version. A bucket-1 test at gap≤10 or a
recency test at 11.6k proves little; they must push to ~350 and ~42k respectively. A test easier
than the real consumer is false confidence. The numbers exist (this §9, the analysis scripts), so
faithful tests are buildable now without DOOM.

**Testing strategy.** Per-phase `probe_compiled` + `OnnxDebugSession` (root-CLAUDE triage
sequence). D6 reproducers at the smallest layer: offset → op test; bucket-1 count → op test over
the bounded gap range; recency → op test over the gap distribution. Oracle parity first:
`Attn.compute` rotary path tested against the compiled rotary head before any head migrates.
Cache invariant test (Phase 0): prefill and unbounded decode produce identical offset-head
logits. Compiler-invariant tests for the self-match migration before broad suite runs.
`make measure-noise` + update `numerical_noise_findings.md` for any new PL op (D7). Per-phase
validation is the torchwright-internal confidence suite above (deep-rollout parity for Phases 1/4
via representative graphs, not DOOM). Full **DOOM render parity is Phase 8 — the final integration
step on a separate `torchwright_doom` branch**, not a per-phase gate in this repo.

**Load-bearing numbers and their sources.**
- **Recency requirement + octant headroom:** the full-frame instrumentation log
  (`torchwright_doom/scripts/position_attention_log_full.jsonl`, 276,831 real selections) replayed
  by `scripts/rope_recency_replay.py`. Recency decisive in 95%; binding gap min/median/max =
  1/19/8071; gap-1 cases out to the max absolute position (median ~5,900, max 11,621) ⇒ uniform
  resolution required. Octant ramp: ~370× the fp32 floor, 0 flips over 8 seeds (and at 30× the
  floor); single-cosine ~6× (rejected); flip budget ~1e-5 rad clean. Re-run on a ~42k log
  (Phase 4) — this frame reaches ~11.6k.
- **Recency lobe `W≈414`:** `scripts/rope_recency.py`, `scripts/rope_window_frontier.py`
  (config-specific: base 1e6, d_head 256, Hann taper).
- **Clip look-back distribution:** `scripts/clip_gaps.py` / `scripts/clip_sweep.py` (proper home
  `torchwright_doom/scripts/`).
- **base-`1e6` `Δθ`:** analytic / `scripts/rope_recency.py` `theta_min` — confirm against the
  production frame length (§6).
- **Per-class DOOM call-site counts:** sub-agent census, **not** re-confirmed — re-grep before
  per-site work.

## 10. Code change sites

> **Phase-5 status (this branch): all torchwright-side sites below are ✅ done** (the deletions and the
> `pos_encoding` → `RopeConfig` signature change landed and import clean). Line
> numbers predate the edits. The ONNX/HF emission edits among them are **validated end-to-end**
> (Strand D — `compile_to_onnx` → HF, prefill == cached decode, save→load round trip), and the
> per-head rotary-flag removal (Strand B2) is **done** (universal full-width rotary, `d_qk == d_head`
> enforced). See §8 Phase-5 status and §11.

- **Delete the host position table (three sites).** `transformer.py:103-106` (writes
  `get_pos_encoding(...)` rows into reserved residual columns each forward); the `pos_encoding_full`
  ONNX initializer + `Gather(pos_encoding_full, cache_position)` in `export.py`; and the HF model
  `compiler/hf/modeling_torchwright.py`, which gathers `pos_encoding_full[cache_position]`.
- **The compiler self-match gatekeeper.** `_current_pos_attn_matrices` (`weight_writer.py:338`) —
  every Linear/Add/Cancel/add_into self-match flows through it (Phase-2 blast radius; delta_transfer
  removed on `main`). Today it reads `counter_col` only to zero it out of the logit; under RoPE the Δ=0
  self-match is identity, so this simplifies.
- **Input-node classification & reservation.** `PosEncoding` is an input node
  (`graph_analysis.py:208`), pre-allocated and never freed (`compile.py:570/573`). The recency
  signal's reserved residual column inherits the reservation pattern but is **computed**, not
  host-supplied (a "computed-once-then-reserved" category).
- **Constraints that vanish.** `d_head ≥ trig_width` (`compile.py:551`), the counter affine bound
  `[0, 100000]` (`affine_rules.py`), the `len(pos_encoding)` debug-fingerprint term
  (`graph_identity.py:116`).
- **Signatures.** `pos_encoding` drops from the `attend_*` family (`attention_ops.py`),
  `sequence_ops.py`, `prefix_ops.py`, `inout_nodes.create_pos_encoding`, the `export.py` entry
  points, and the layer constructors (`transformer_layer.py`, `attn_sublayer.py`,
  `components/attn.py`); `probe.py`/`onnx_debug.py`/`graph_identity.py` fingerprint references
  update.

## 11. Open gates (live)

- **Content plane budget.** ✅ DONE (Phase 3). The slowest-plane attenuation at 42k (~0.9965) is set
  by `base` (5e5), not `d_head`; `d_head = 256` (128 planes) covers the widest content head with
  margin. Per-head selection-at-distance gates green (dot 198, bucket 253, worst-case fine-score 3.8 →
  47× concentration). **The builder rewrite (the `attend_*` funcs → full-width rotary on slow planes)
  ✅ DONE (Phase 5, this branch)** — every builder takes `RopeConfig` and routes through
  `rotary_content_head`; oracle-validated. **DOOM call-site wiring is Phase 8**.
- **Re-grep per-class call sites.** Replace the sub-agent census with committed greps before
  per-site work (a direct grep found fewer literal `attend_most_recent_matching` sites than the
  census implied). Confirmed 2026-06-27: `pick_most_recent` does **not** exist in current code —
  the recency family is `attend_most_recent_matching` (`attention_ops.py:1383`) and
  `get_prev_value` (`pos_encoding.py:129`). The compiler self-match callers (now **four** —
  delta_transfer removed on `main`) flow through `_current_pos_attn_matrices` (`weight_writer.py`),
  and the host-position-table sites are in `transformer.py`, `export.py`, `modeling_torchwright.py`.
  **All line numbers here predate the `main` merge — re-grep before per-site work** (the merge
  refactored `export.py`/`modeling_torchwright.py`, e.g. the untied embedding and removed headless
  export shifted those line numbers). ✅ **Executed in Phase 5 (this branch):** the per-class rewrite
  and the host-table deletion landed; `get_prev_value` is now a module function in `attention_ops.py`,
  and `_current_pos_attn_matrices` / `graph/pos_encoding.py` no longer exist.
- **RoPE convention (LLaMA3-aligned).** ✅ DONE (Phase 0). `rotate_half` layout (dropped the legacy
  interleaved `(2i,2i+1)` pairing), both Q and K rotated by absolute position (cache holds rotated K,
  matching HF), base = `5e5`, no long-context frequency scaling. ✅ Resolved (Phase 4): the global
  grid contains a sub-one-turn recency plane — plane 91 at base `5e5`/`d_head 256` (`θ ≈ 8.88e-5`,
  one turn ≈ 70761 positions, never wrapping over the 61440 cap); the `1e6`-measured analyses were
  re-confirmed at `5e5` (the octant headroom survives).
- **Offset sign convention.** ✅ DONE (Phase 0). `W_K = R_N W_Q` pinned by the offset-head test
  (which also pins the rotation layout).
- **Compiler blast radius.** ✅ DONE (Phase 2 Part 2). Linear/Add/Cancel/add_into self-match
  recompiles through rotary self-match behind the opt-in `rotary_self_match` flag; the reserved
  constant-1 residual feature is a `LiteralValue([1.0])` that rides the existing all-surface
  `constant_values` seed (no new ONNX/HF emission), and the rotation propagates via per-head
  `rotary_width`. Full suite green, **I1–I4 held** (no D1 stop). Part 1 (relative-offset, all Δ) ✅
  `feffb20`. ✅ **Phase 5 (this branch) made the rotary self-match the only path** — the
  `rotary_self_match` flag, the trig self-match, and `_current_pos_attn_matrices` are removed; the
  reserved const-1 column is allocated unconditionally. (B2, below, still has to make the *rotation
  itself* unconditional — drop the per-head `rotary_width` flag.)
- **Runtime rotation parity.** ✅ DONE (Phase 0) **and now CI-tested** — `main` (`f281ee8` merge) added
  `onnxruntime`/`transformers` to the Modal image, so the ONNX/HF rotation tests run in CI; the full
  suite is green (`test_rope_token.py` passes the token-ONNX/HF RoPE path through the untied
  embedding). The "add onnxruntime to the Modal image" blocking gate is resolved.
- **Oracle-first.** ✅ DONE (Phase 0). `Attn.compute` rotary path landed before any head migration;
  `probe_compiled` parity confirmed.
- **Recency confirm-compile.** ✅ **DONE.** The unproven-mechanism crack is closed. `soft_blend`
  (`102dedb`), the graph-level octant ramp builder with a build-time branch-equality check
  (`recency_ramp.py`, `c2ad29d`), and the compiled gate-b monotonicity sweep
  (`test_recency_ramp_compiled.py`, `65089c8`) are all landed and green. The compiled ramp is strictly
  monotone, worst boundary step ~2.19e-5/tok (~183× fp32 floor), matching the assembly model. The
  gate's correct form is **token-density across phase offsets**, not sub-token-dense (which hits fp32
  output quantization — see Phase 1b status). ✅ **DONE (Phase 4):** the ramp is wired to the real two
  graded `{BOS, REF}` rotary heads (`recency_phase_heads` / `recency_rank`, `ops/recency_heads.py`),
  `probe_compiled` parity holds on the `heads → ramp` chain, and the heads produce `σ(M·cos/sin φ)−0.5`
  to fp32. (The 0-flips replay with the real ramp moves to the 42k cross-repo follow-up below.)
- **Recency leakage budget.** ✅ DONE (Phase 4, gate d). The reference is **REF**, not `self` (a
  `{BOS, self}` softmax shifts the weight by a constant `−M·cos ψ` — see Phase 4 status). The DC
  exclusion gap is `L=25` (≈ `log(N/resolution)` at the cap); the compiled head output matches the
  2-key ideal to fp32 and does **not** grow with key count (`N=256 → 4096`), ~23× under the gap-1
  weight signal at the 61440 cap (~350× at N=4096; cap bound pinned by
  `test_leakage_below_gap1_signal_at_cap`).
- **Recency at production length.** ⏳ Phase 8 (cross-repo). The ramp's resolvability *at the
  cap φ-density* is proven (`test_recency_ramp_compiled.py` + the analytic gap-1 band in
  `test_rope_recency_e2e.py`); the **0-flips replay against real selections at ~42k** needs a ~42k
  `torchwright_doom` instrumentation log (committed log is the ~11.6k frame), so it lands on the
  Phase-8 `torchwright_doom` branch (DOOM untouched through Phase 5, §7).
- **Bucket-1 marker gates.** Per consumer, confirm the marker is graph-recognizable and the gap
  bound holds at production config (the marker check is the Phase-8 DOOM call-site rewire).
- **ONNX/HF emission validation (Phase 5, Strand D).** ✅ **DONE.** Exercised end-to-end on Modal
  (`compile_to_onnx` → HF; prefill == cached decode; save→load round trip). A real save→load NaN bug
  (non-persistent rotary buffers not materialised under `from_pretrained`'s meta-device load) was found
  and fixed by recomputing the rotary constants in `forward`; pinned by `test_hf_save_load_rope.py`.
- **tests + examples migration (Phase 5, Strand C).** ✅ **DONE.** Tests + `examples/` migrated to
  `RopeConfig`; the four `prefix_sum` examples and their tests retired; `test_pos_encoding.py` /
  `test_position_scalar.py` and the `_current_pos_attn_matrices` tests deleted. Tail green-up fixes
  landed (baib d_head, multiposition dead-leaf, ONNX-meta `rope_base`) plus an I1 `const_one` node-id
  collision fix (`reserve_node_id_above`, `test_recompile_node_id_collision.py`). Full `make test` green.
- **Per-head rotary flag removal (Phase 5, Strand B2).** ✅ **DONE.** The `Attn.rotary` flag is gone
  (every head rotates) and the in-process path asserts `d_qk == d_head`, so NoPE **and** partial-width
  rotary are both structurally impossible — universal full-width LLaMA3 rotation, no escape hatch. Wall
  attention was migrated (not deleted) to a slow-plane `rotary_content_head` to keep the Phase-8
  DOOM-port derisk valid (`test_wall_attention.py`); its docstring records the production-difficulty
  open questions for Phase 8.
