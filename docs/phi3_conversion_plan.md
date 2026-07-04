# Phi-3 conversion — execution plan

Goal: a torchwright swish artifact (`bias=False`, `rms_norm`) becomes a
checkpoint that loads as **stock `Phi3ForCausalLM`** — no custom
modeling file, no `trust_remote_code`, no custom tokenizer code.
Written 2026-07-04, after the partial-rotary audit killed the
`LlamaForCausalLM` target that `docs/no_bias_plan.md` had recorded as
the follow-up (that bullet now points here). Unscheduled. Example-scale
work is unblocked today — Phase C gives swish example graphs, the
N-series gives `bias=False` emission, rms_norm is landed. The flagship
checkpoint additionally waits on the Phase D cutover
(`docs/swiglu_step2_plan.md`).

## Why Phi-3 and not Llama

The flagship needs partial rotary (`d_head: 128, d_rot: 64` in both
e1m1 configs) and that is structural, not a knob: the unrotated NoPE
tail is where position-independent content matching lives — the
engineered content-equality heads cannot work in rotating planes.

Stock Llama cannot express that, and it fails in the worst way:
`LlamaConfig` **accepts** `partial_rotary_factor` into its
`rope_parameters` dict without complaint, but `LlamaForCausalLM`'s
modeling code **ignores it** — measured on the pinned transformers
(5.12.1): requesting factor 0.5 on a 16-dim head still builds a
full-width frequency table (8 pairs where 4 were requested). A
converted checkpoint would silently rotate the 64 dims the compiled
attention relies on being position-independent. Silent wrong model,
not an error.

Architecture sweep (transformers 5.12.1, tiny-model instantiation,
`inv_freq` width inspected — not config fields, which lie):

| architecture | partial rotary honored | SwiGLU | RMSNorm | biasless |
|---|---|---|---|---|
| Llama, Mistral, Qwen2/3, Gemma2, Granite, Olmo2, Helium, SmolLM3 | no (ignored) | yes | yes | mostly |
| GPT-NeoX, Persimmon, Nemotron | yes | no | no | — |
| StableLm | yes | yes | no (LayerNorm) | yes |
| GLM / GLM4 | yes | yes | yes | no (qkv bias default) |
| **Phi3** | **yes** | **yes** | **yes** | **yes** |

**Phi-3's rotation is semantically identical to ours.** Its
`apply_rotary_pos_emb` rotates the first `rotary_dim` dims with
half-split pairing (dim `i` with `i + rotary_dim/2`) and passes the
tail through — character-for-character `graph/rope.py::apply_rope`.
Measured at flagship geometry (`head_dim=128`, factor 0.5, base 10⁴):
same 32 planes, `inv_freq` equal to `rope_inv_freq(64, base)` within
≤ 1.2e-7 relative (their table is fp32 storage of a float64
intermediate — one rounding), rotated q/k agree to ~4e-7 absolute, and
the NoPE tail passes through **bit-exactly**. No weight permutation
needed.

Rejected alternatives, recorded so they aren't relitigated:

- **GLM / GLM4** — the runner-up and the designated fallback if a
  transformers upgrade breaks Phi-3's partial-rotary path. Honors the
  factor with the right body (RMSNorm, SwiGLU, `head_dim=128` is even
  its default), and rotates the same first-`rotary_dim` split — but
  pairs dims interleaved (`2p, 2p+1`), which costs an exact Q/K row
  permutation at conversion, and ships attention biases by default.
  Workable; strictly more moving parts than Phi-3.
- **StableLm** — honors the factor but normalizes with LayerNorm, and
  mean subtraction breaks the pinned-constant identity trick: the
  giant constant shifts every data column through the mean, and the
  variance stops being an exact power of two.
- **Full-rotary emulation inside stock Llama** — no frequency choice
  freezes the tail planes (the grid is geometric in the base; making
  the front 32 planes match exactly via `base' = base²` leaves tail
  planes rotating up to ~0.8 rad over the context window), and
  quadrature tricks make content scores distance-dependent, which is
  exactly what the NoPE tail exists to avoid.
- **Upstreaming a Llama fix** — parked (see *Risks*). The
  config-accepts/modeling-ignores asymmetry suggests upstream would
  treat it as unsupported-by-design for Llama checkpoints.

## Ground truth: what the artifact provides (token.v5)

From `compile_to_onnx` on a swish, `bias=False`, `rms_norm` graph:

- **Attention**: `l{i}_WQ / WK / WV / WO`, emitted as MatMul
  right-operands (`(d, n_heads·d_head)` orientation — HF `Linear`
  weights are the transpose). Scores are **raw** `Q·Kᵀ` — no
  `1/√d_head` anywhere; every gain is folded into the weights.
- **Causal mask**: an **overwrite** (`Where(mask, SENTINEL, logits)`),
  not additive — chosen because real logits can be very negative.
  Phi-3's eager path masks *additively* with `finfo.min`. Delta to
  audit, not to port.
- **MLP**: `l{i}_Wgate / Wup / Wdown` (no `b*` under `bias=False`).
  Runtime pattern is `down(silu(gate(x)) · up(x))` — exactly Phi-3's,
  whose fused `qkv_proj` / `gate_up_proj` are plain concatenations.
- **Norms**: `l{i}_input_layernorm`, `l{i}_post_attention_layernorm`,
  `final_norm` — already Llama/Phi3-named, each a uniform `(d,)`
  vector `2^m` cancelling the forced `rms = 2^m` exactly, plus a
  shared `_rms_eps` scalar. This maps 1:1.
- **Embedding / head**: `embed_table` (over-allocated: row count is
  the logit width, which the compiler pads past `len(vocab)`) and an
  untied `lm_head`.
- **Meta**: `activation`, `bias`, `rms_norm(_eps)`, `rope_base`,
  `d_rot`, `d`, `d_head`, per-layer head counts, `cache_stride`,
  `vocab`, `max_seq_len`. `cache_stride` and the strided-decode slot
  machinery are torchwright-runtime concerns — HF uses its own cache;
  the conversion consumes initializers + meta only.
- **Non-uniformity**: `trim_heads` / slot trim make per-layer head
  counts and MLP hidden widths vary. Stock configs are uniform.

## The weight mapping

| artifact | Phi-3 parameter | note |
|---|---|---|
| `embed_table` | `model.embed_tokens.weight` | keep padded rows; `vocab_size` = logit width |
| `l{i}_WQᵀ · √d_head` | rows `[0, H·dh)` of `qkv_proj.weight` | the scaling fold — Q only |
| `l{i}_WKᵀ` | rows `[H·dh, 2H·dh)` | |
| `l{i}_WVᵀ` | rows `[2H·dh, 3H·dh)` | |
| `l{i}_WOᵀ` | `o_proj.weight` | |
| `l{i}_Wgateᵀ` | rows `[0, I)` of `gate_up_proj.weight` | |
| `l{i}_Wupᵀ` | rows `[I, 2I)` | |
| `l{i}_Wdownᵀ` | `down_proj.weight` | |
| `l{i}_input_layernorm` | `layers[i].input_layernorm.weight` | 1:1 |
| `l{i}_post_attention_layernorm` | `layers[i].post_attention_layernorm.weight` | 1:1 |
| `final_norm` | `model.norm.weight` | 1:1 |
| `_rms_eps` | `config.rms_norm_eps` | |
| `lm_head` | `lm_head.weight` | `tie_word_embeddings=False` |

Config: `hidden_size = d`, heads padded to uniform (see below),
`rope_parameters = {rope_type: "default", rope_theta: rope_base,
partial_rotary_factor: d_rot/d_head}`, `max_position_embeddings =
max_seq_len`, `sliding_window = None`, `hidden_act = "silu"`,
`attention_dropout = 0`.

**Padding to uniform.** Per-layer trimmed heads pad back to the
uniform count with all-zero Q/K/V rows and zero `o_proj` columns; MLP
widths pad with all-zero gate/up rows and zero `down_proj` columns.
Both are *bit-exact no-ops*: a zero gate lane contributes
`silu(0)·up = 0` and a zero down-column reads nothing; a zero head
produces uniform softmax over V-rows that are exactly zero, so the
head output is exactly zero into zero o-columns. Cost is parameters
and KV cache only (measure at P3). Default: pad heads to `d / d_head`
(so `hidden_size / num_heads` matches `head_dim` with no explicit
knob); if `Phi3Config` accepts an explicit `head_dim`, padding to the
per-layer max is the cheaper alternative — settle at P0(c).

## Numerical implications (stated upfront)

The native-module converter's claim is bit-exactness. The Phi-3 claim
is **bounded-relative parity + token-exact decode**, from four rounding
sources, none removable:

1. **Rope tables**: fp32-rounding-equal, not bit-equal (≤ 1.2e-7
   relative). Reaches logits as ~relative·|logit|.
2. **The `√d_head` fold**: `√128` is irrational — one fp32 rounding
   per WQ element, and Phi-3 multiplies logits by `fl(d_head^-1/2)` at
   runtime — a second rounding. Net relative perturbation ~1e-7.
3. **Mask semantics**: additive `finfo.min` vs our sentinel overwrite.
   With engineered logit magnitudes (~1e5–1e6), `finfo.min` dominance
   should still give masked weights exactly 0.0 in fp32 — verify with
   our magnitudes at P0, don't assume.
4. **Kernel choice**: parity gates pin `attn_implementation="eager"`;
   SDPA/flash consumers get token-level claims only.

Sign-off: worst-case logit perturbation from (1)+(2) against measured
score-gap margins (`probe_attention` / `assert_score_gap_at_least`) —
the same class as Phase D's accepted "~1e-7 relative" weakening for
the pick emissions. The rms eps is a non-issue by the same argument
the artifact itself already relies on (pinned energy `2^{2m}` swamps
eps below fp32 resolution). No op implementations change, so D7
requires no noise re-measurement.

## Phases

- **P0 — probes and audits.**
  - *(a) Rotary parity probe* (`tests/hf/test_phi3_rope_parity.py`):
    pins the split, the pairing, NoPE-tail bit-exactness, and
    `inv_freq` 1-ulp agreement against `graph/rope.py`, at flagship
    geometry. The A0 pattern: a transformers upgrade that breaks the
    partial-rotary path fails loudly here, and the GLM fallback
    decision triggers.
  - *(b) Score-path audit*: read Phi-3's eager scaling site and mask
    application; bound the logit perturbation from rounding sources
    (1)+(2); compare against measured score gaps on a swish example
    artifact (flagship at P3). Gate: recorded margin factor.
  - *(c) Config constraint check*: explicit `head_dim` support vs
    pad-to-`d/d_head`.
- **P1 — the converter.** A Phi-3 path in `compiler/hf/convert.py`:
  routing on meta (`swish` + `bias=False` + `rms_norm` → Phi-3; relu →
  the existing native module, unchanged; anything else refuses
  loudly — today's two `NotImplementedError`s become this routing).
  Weight mapping + padding per the table; config build; tokenizer
  emitted as a `tokenizer.json` (WordLevel over the vocab,
  char-level pre-tokenization, specials registered) loaded by stock
  `PreTrainedTokenizerFast` — the current character-level
  `TorchwrightTokenizer` semantics, zero custom code in the bundle.
  Unit scale: adder-size swish `bias=False` exports.
- **P2 — parity gate.** A Phi-3 mirror of `tests/hf/test_convert.py`:
  teacher-forced max-logit-diff under the derived bound, greedy decode
  token-exact against the `load_onnx` oracle, eager attention pinned;
  D6-scale repros for each mapping piece (fold, padding, norms,
  tokenizer round-trip).
- **P3 — flagship checkpoint (after Phase D).** Export e1m1
  (`bias=False` flips there), convert, measure padding overhead
  (params + KV), walkthrough frames through HF `generate`
  (token-level equivalence), and bundle the HF repo — which now reads
  as *maximally* ordinary: a stock-architecture checkpoint with no
  custom code at all.

## What dies / what stays

- The custom trio (`modeling_torchwright.py`,
  `configuration_torchwright.py`, `tokenization_torchwright.py`)
  stays serving relu artifacts and `tests/hf` until relu retirement,
  then dies with it — the HF surface becomes Phi-3-only.
- `build_config`'s swish and `bias=False` refusals die at P1 (they
  become the routing).
- The `no_bias_plan.md` follow-up bullet ("True `LlamaForCausalLM`
  conversion") is superseded by this document.

## Risks / parked

- **Version drift**: `partial_rotary_factor` is a code path official
  Phi-3 checkpoints don't exercise; upstream could break or remove it
  without noticing. The P0 probe is the tripwire; GLM (+ exact
  interleave permutation, biases zeroed) is the recorded fallback.
- **SDPA/flash numerics** for stock consumers who don't pin eager:
  token-level equivalence is the only claim made; whether to document
  a recommended `attn_implementation` in the model card is a P3 call.
- **Padded-head KV cost at flagship scale**: HF is the contract/demo
  path, not the perf runtime (that stays doom-side CUDA-graph), but
  measure before publishing.
- **Upstreaming Llama partial rotary**: would reopen the nicer-named
  target; low leverage, unclear upstream appetite. Not planned.
- Multi-machine bundles (relu native + swish Phi-3 in one repo): no
  use case; one artifact, one checkpoint.
