# torchwright ops cut spec

*Phase-1 deliverable. Code-verified against the working tree on `main`
(2026-06-30). Supersedes the reachability claims in `ops_audit.md` where they
conflict — see [Corrections to the audit](#corrections-to-the-audit).*

**Goal (from the user):** a *minimal coherent* op surface to carry through the
upcoming op rewrite. Rule agreed:

> **Keep = actually used today** by an example (`torchwright/examples/`) or by
> the DOOM **renderer** package (`torchwright_doom/torchwright_doom/`, incl.
> `std.py`). Everything else is cut — including dead keyword arguments and
> unexercised code paths — unless keeping it makes the *surviving* surface more
> coherent. Speculation ("a future consumer might want it") is **not** a keep
> reason. Tests are **not** a keep reason.

**Method.** Four agents built a call graph over all 95 public ops annotated
with the **branch condition guarding every internal edge**; two agents
enumerated every consumer call site with argument values. The keep-set is the
transitive closure of the live-consumer seed over **edges that are actually
taken** — an edge behind `if strategy == "shallow"` is pruned when no live
caller passes `"shallow"`. This is the step a static call graph gets wrong, and
it is why this spec differs from `ops_audit.md` in three material places.

**Consumer-set convention.** Live = examples + renderer package. **Not** live =
all tests, and the DOOM `scripts/` (diagnostic, not product). One cut op
(`attend_most_recent_matching`) is kept alive *only* by a script — it is called
out in [Decisions](#decisions-for-you) rather than cut silently.

---

## Summary

| | count |
|---|---|
| public ops total | **95** |
| **keep** | **64** |
| **cut** | **31** (30 unconditional + `piecewise_linear_2d`, a flagged decision) |

Two whole modules cut (`quantization.py`, `loop_ops.py`); `embedding_arithmetic.py`
loses 6 of 8; `attention_ops.py` loses 8 of 15.

---

## Corrections to the audit

The branch-guard verification found two ops the audit marks live but that no
live caller actually reaches, and one swapped pair. These are the payoff of
verifying against code rather than a static call graph:

1. **`square_signed` — DEAD (audit implied live).** `ops_audit.md` line 69
   draws `multiply_integers → square_signed`. That edge is behind
   `if strategy == "shallow"`; `multiply_integers` defaults to `"deep"` and the
   one live caller (`calculator_simple.py`) passes no strategy. `square_signed` is
   reached **only** via the shallow branch of `multiply_integers` /
   `signed_multiply`, and `"shallow"` is passed nowhere. (Audit line 182 had
   this right; line 69 contradicts it.)

2. **`piecewise_linear_2d` — effectively DEAD (audit: "doom-live, indirect").**
   The audit draws `multiply_2d → piecewise_linear_2d`. That edge is behind
   `if range1 < 1e-12 or range2 < 1e-12` — the degenerate zero-width-axis
   fallback. Every live `multiply_2d` call passes real ranges (`max_abs1=512`,
   etc.), so the quarter-square path runs and `piecewise_linear_2d` is never
   reached. Not imported directly by any example or renderer file. Cutting it
   has a subtlety (the fallback branch) — see [Decisions](#decisions-for-you).

3. **The recency pair is swapped.** Audit: `attend_most_recent_globally` dead,
   `attend_most_recent_matching` doom-live. **Reality:** the renderer's
   `past.py` imports and calls **`attend_most_recent_globally`** (live);
   **`attend_most_recent_matching`** appears only in a DOOM *script* and tests
   (cut). Likely the graph moved to the global-recency head since the audit's
   commit.

Everything else in the audit's dead-29 reproduced.

---

## The cut list (31)

`reason` codes: **U** = unused by any live consumer, directly or through a
taken edge · **branch** = the only path to it is a never-taken conditional ·
**dup** = duplicated by a local copy in the consumer.

| op | file | reason | tests that die |
|---|---|---|---|
| `relu` (op fn) | arithmetic_ops | U — every ReLU comes from the `ReLU` node class / `linear_relu_linear`, never this op | none dedicated (name collides; no op-specific test) |
| `max` | arithmetic_ops | U — `min` is live (`calculator_scratchpad`), `max` is not | `test_arithmetic_ops.py` (surgical) |
| `low_rank_2d` | arithmetic_ops | U — removed doom build-cost stopgap (`render_ops.py:484` note) | `test_low_rank_2d.py`, `test_low_rank_2d_compile.py` (whole-file) |
| `square_signed` | arithmetic_ops | branch — only via never-passed `strategy="shallow"` | none |
| `signed_multiply` | arithmetic_ops | U — only in-package caller is `linear_bin_index` (also cut) | `test_arithmetic_ops.py`, `test_resampling_primitives.py`, +compile fixtures (surgical) |
| `linear_bin_index` | arithmetic_ops | U | `test_resampling_primitives.py` (surgical) |
| `log` | arithmetic_ops | U | `test_arithmetic_ops.py` (surgical) |
| `exp` | arithmetic_ops | U | `test_arithmetic_ops.py` (surgical) |
| `log_abs` | arithmetic_ops | U | `test_arithmetic_ops.py` (surgical) |
| `reduce_min` | arithmetic_ops | U | `test_arithmetic_ops.py` (surgical) |
| `reduce_max` | arithmetic_ops | U | `test_arithmetic_ops.py` (surgical) |
| `piecewise_linear_2d` | arithmetic_ops | branch — see Decisions | `test_piecewise_linear_2d.py` (whole-file), `test_multiply_2d.py`, `test_low_rank_2d.py` (surgical) |
| `attend_argmin` | attention_ops | U | `test_attention_ops.py`, `test_forward_compile.py` (surgical) |
| `attend_argmax` | attention_ops | U | `test_attention_ops.py`, `test_forward_compile.py` (surgical) |
| `attend_argmin_where` | attention_ops | U | `test_attention_ops.py`, `test_forward_compile.py`, `test_rope_partial.py` (surgical) |
| `attend_argmax_where` | attention_ops | U | `test_attention_ops.py`, `test_forward_compile.py` (surgical) |
| `attend_argmin_valid_unmasked` | attention_ops | U | `test_attention_ops.py` (surgical) |
| `attend_argmax_dot_where` | attention_ops | U | `test_attention_ops.py` (surgical) |
| `attend_argmin_dot_where` | attention_ops | U | `test_attention_ops.py` (surgical) |
| `attend_most_recent_matching` | attention_ops | U by renderer — script+tests only (see Decisions) | `test_local_recency.py`, `test_rope_local_recency.py`, `test_global_recency.py`, `test_attention_ops.py`, `test_forward_compile.py` (surgical) |
| `table_lookup_3d` | map_select | U — only caller was itself→`table_lookup_2d` | `test_resampling_primitives.py` (surgical) |
| `cond_add_vector` | logic_ops | U — imported into `map_select` but never called | `test_logic_ops.py`, `test_weight_writer.py`, `test_forward_compile.py` (surgical) |
| `subtract_digits` | embedding_arithmetic | dup — calculator examples shadow it locally | `test_calculator_arithmetic.py` (surgical) |
| `subtract_digit_seqs` | embedding_arithmetic | dup | `test_calculator_arithmetic.py` (surgical) |
| `compare_digit_pair` | embedding_arithmetic | dup | `test_calculator_arithmetic.py` (surgical) |
| `compare_digit_seqs` | embedding_arithmetic | dup | `test_calculator_arithmetic.py` (surgical) |
| `multiply_digit_pair` | embedding_arithmetic | dup | `test_calculator_arithmetic.py` (surgical) |
| `multiply_digit_seqs` | embedding_arithmetic | dup | `test_calculator_arithmetic.py` (surgical) |
| `quantize_to_range` | quantization | U — whole module dead | `test_quantization.py` (whole-file) |
| `dequantize_from_range` | quantization | U — whole module dead | `test_quantization.py` (whole-file) |
| `unrolled_loop` | loop_ops | U — whole module dead | `test_loop_ops.py` (whole-file) |

**Private helpers that cascade-die** (delete with their public callers, no
separate decision): `_build_selection_attn`, `_build_where_attn`,
`_build_dot_where_attn` (attention); `_log_single`, `_log_compare_01`,
`_log_abs_single` (arithmetic). `_product_2d_quarter_square` **stays** (it is
the live path of the kept `multiply_2d`).

---

## The keep-set (64)

- **arithmetic_ops (23):** add, subtract, negate, add_const, multiply_const,
  bool_to_01, add_scaled_nodes, sum_nodes, concat, relu_add, abs, compare, min,
  piecewise_linear, multiply_2d, square, thermometer_floor_div, mod_const,
  clamp, reciprocal, floor_int, ceil_int, multiply_integers
- **attention_ops (7):** attend_argmin_above_integer, attend_argmin_above_in_bucket,
  attend_argmin_unmasked, attend_mean_where, attend_argmax_dot, attend_to_offset,
  get_prev_value
- **global_recency (2):** attend_most_recent_globally, global_position_from_bos
- **map_select (7):** map_to_table, table_lookup_2d, switch, select, in_range,
  dynamic_extract, broadcast_select
- **logic_ops (6):** bool_any_true, bool_all_true, bool_not, equals_vector,
  cond_gate, per_column_offsets *(pure numeric helper used by `cond_gate`; keep
  but candidate to de-publicize)*
- **embedding_arithmetic (2):** sum_digits, sum_digit_seqs
- **sequence_ops (4):** check_is_digit, NumericSequence, output_sequence, remove_leading_0s
- **scalar_encoding (4):** digit_to_scaled_scalar, digits_to_number, number_to_digit_scalars, scalar_to_embedding
- **onehot_table (1):** onehot_lookup
- **inout_nodes (6):** create_input, create_literal_value, create_embedding, create_onehot_embedding, create_unembedding, create_rope_config
- **marker_count (1):** count_since_marker
- **linear_relu_linear (1):** linear_relu_linear

Ops that look cuttable but are **kept via a taken edge**, for the record:
`multiply_const` (←`bool_to_01`,`mod_const`), `abs` (←`min`,`multiply_integers`
deep), `square` (←`multiply_integers`), `reciprocal` (←`count_since_marker`),
`sum_digits` (←`sum_digit_seqs`).

---

## Dead code paths inside KEPT ops

Strip these when the op is rewritten — they are unreachable given live callers:

| kept op | dead path | note |
|---|---|---|
| `multiply_integers` | `strategy` param + `if strategy=="shallow"` branch | drops the `square_signed` edge; keep the `"deep"` body only |
| `multiply_2d` | `if range<1e-12` fallback → `piecewise_linear_2d` | tied to the `piecewise_linear_2d` decision below |
| `select` | `approximate=False` branch (`select_cond_gates`) | only tests pass `approximate=False`; **asymmetric** — see below |
| `cond_gate` | `approximate=False` branch (`cond_gate_c_off`) | only tests pass it |
| `attend_most_recent_globally` | `exclude_self=True` branch | renderer never passes `exclude_self`; kwarg escape hatch |

**Asymmetry to note:** `broadcast_select`'s `approximate=False` branch **is
live** — `std.py` calls `_broadcast_select(..., approximate=False)`. So of the
three `approximate`-carriers, only `select` and `cond_gate` have a dead False
branch; `broadcast_select` does not. Cutting the `approximate` split from
`select`/`cond_gate` but not `broadcast_select` is defensible (match reality)
but leaves the three siblings inconsistent — flagging for the rewrite.

---

## Dead knobs on KEPT ops (never overridden by any live caller)

| kept op | dead kwarg (default) |
|---|---|
| `square` | `d_max` (1024) |
| `reciprocal` | `d_max` (1024) |
| `table_lookup_2d` | `d_max` (1024) |
| `select` | `c_tol` (0.005) |
| `cond_gate` | `c_tol` (0.005) |
| `broadcast_select` | `c_tol` (0.005) |
| `attend_argmin_above_integer` | `assert_hardness_gt` (None) |
| `attend_argmin_unmasked` | `assert_hardness_gt` (None) |
| `attend_argmax_dot` | `assert_hardness_gt` (None) |
| `attend_most_recent_globally` | `assert_hardness_gt` (None) |
| `get_prev_value` | `recency_gain` (kw-only) |
| `global_position_from_bos` | `n_breakpoints` (kw-only) |
| `multiply_2d` | `min1`, `min2`, `max_abs_output` (None) |

**Live** knobs to keep (verified passed by a live caller):
`compare.sharpness` (doom `32000.0`), `floor_int/ceil_int.sharpness`,
`multiply_2d.step1/step2/breakpoints1/breakpoints2/name`,
`table_lookup_2d.index_scale/sharpness`, `attend_argmax_dot.match_gain`,
`attend_most_recent_globally.match_gain/recency_scale`,
`attend_argmin_above_in_bucket.assert_hardness_gt` (doom passes it — the **one**
op that exercises the hardness assert), `broadcast_select.approximate`,
`sum_nodes.max_fanout`, `piecewise_linear.clamp` (`global_position_from_bos`
passes `clamp=False`), `multiply_integers`… (none — its only knob `strategy` is
dead), `create_input.value_range`, `create_rope_config.d_head/max_positions`,
`remove_leading_0s.max_removals`.

**`assert_hardness_gt` is dead on 4 of the 5 kept ops that carry it** and live
on 1. Either make it a uniform, always-present debug knob or drop it from the
4 — a coherence call for the rewrite.

---

## Decisions for you

1. **`piecewise_linear_2d`** — cut it? It's unreachable via live inputs, but it
   *is* `multiply_2d`'s defensive fallback for a zero-width input axis (a
   legitimate if degenerate case). Cutting the op means also removing that
   `if range<1e-12` branch, so a future zero-range `multiply_2d` call would need
   its own guard. **My lean: cut**, and have `multiply_2d` raise on a
   zero-range axis instead of silently routing to a general 2-D PWL. Flagging
   because it's the one cut that changes a kept op's behavior on an edge input.

2. **`attend_most_recent_matching`** — no renderer/example use; kept alive only
   by `scripts/position_attention_log.py` (a diagnostic) and tests. Under the
   agreed rule (scripts not live) it's a **cut**. Confirm you're fine losing the
   script's use, or promote the script to a keeper.

3. **`select`/`cond_gate` `approximate=False`** — dead in production, live only
   in tests. Cut the branch (and the `approximate`/`c_tol` knobs) from these two
   while leaving `broadcast_select`'s live? Or keep all three uniform for
   coherence? My lean: cut the dead branches; note the asymmetry in the rewrite.

---

## Phase-2 execution notes

- **Whole-file test deletes:** `test_quantization.py`, `test_loop_ops.py`,
  `test_low_rank_2d.py`, `test_low_rank_2d_compile.py`,
  `test_piecewise_linear_2d.py` (if #1 = cut).
- **Whole-module deletes:** `ops/quantization.py`, `ops/loop_ops.py` (+ remove
  `unrolled_loop`, `cond_add_vector`, and the cut `arithmetic`/`map_select`/
  `embedding_arithmetic`/`attention` names from `ops/__init__.py`).
- **Surgical test edits** concentrate in three shared files —
  `tests/ops/test_arithmetic_ops.py`, `tests/ops/test_attention_ops.py`,
  `tests/ops/test_resampling_primitives.py` — plus
  `tests/examples/test_calculator_arithmetic.py` (the digit-seq families).
- **Caution — compile fixtures:** `test_forward_compile.py` and
  `test_weight_writer.py` use some dead ops (`attend_argmin*`, `cond_add_vector`)
  as *generic graph fixtures* for structural/compiler coverage, not to test the
  op itself. Cutting the op means **rewiring those fixtures to a kept op**, not
  just deleting lines — real work, and a place a careless delete would drop
  compiler coverage. Per D6, any behavior a fixture was pinning should be
  re-pinned on a kept op.
