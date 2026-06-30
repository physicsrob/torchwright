# torchwright ops audit

*Generated 2026-06-30 against commit `8e6b686`. Reflects `torchwright/` and its
sibling consumer `torchwright_doom/` at that point.*

Two questions, per op:

1. **Is it used by any example** (directly or indirectly)? — i.e. would any
   example break if the op were deleted?
2. **Are any of its keyword arguments never set to a non-default value** anywhere?

## Scope & method

**What counts as an "op":** all **95 public callables** (non-underscore
`def`/`class`) in `torchwright/ops/` — 14 files (`const.py` and `__init__.py`
define none). `ops/__init__.py` re-exports only ~70 of them; examples and the
doom flagship import the rest **directly from submodules** (e.g.
`from torchwright.ops.attention_ops import get_prev_value`), so this audit
covers the full public surface, not just the `__init__` list. Graph primitives
(`Linear`, `Attn`, `ReLU`, `Concatenate`, `Embedding`, …) live in
`torchwright/graph/` and are out of scope.

**"Used by an example"** = reachable from `examples/` either by a direct
reference *or transitively* through another op's implementation. A call graph
over the ops package was built and reachability taken from the set of ops the
examples reference directly. So `abs` counts as used because `min` (used
directly) calls it.

**"Used by DOOM"** = the same transitive reachability, seeded from every op
`torchwright_doom` imports from `torchwright.ops` — across its renderer files
and the `std.py` wrapper surface the renderers build on. Tests are deliberately
**ignored** on both axes: "dead" below means unused by examples *and* by doom,
regardless of unit-test coverage.

**Kwarg scope:** "never set to a non-default" was checked across **both**
`torchwright/` and `torchwright_doom/` (so it means never, in either repo),
excluding the stale `.claude/worktrees/` branch copies.

**Confidence:** a multi-agent sweep did the inventory + call-site analysis; the
example and doom reachability was then recomputed from an AST-parsed call graph
and re-verified by hand against the source. Care was needed where op names
collide with Python builtins (`max`, `min`, `abs`, `log`): a naive call graph
falsely marks op `max` as reached because other ops call the *builtin* `max()`
on plain numbers — those edges were excluded after reading context. The hand-check
also caught one kwarg error (`select.approximate`, corrected below). The doom
classification was cross-checked by parsing every `torchwright_doom` import of
`torchwright.ops` and sweeping the doom package for stray references.

---

## Q1 — Which ops would break an example if deleted?

**54 of 95 ops are reachable from an example; 41 are not.** Since you care
about DOOM rather than examples, each example-unused op is further split by
whether `torchwright_doom` uses it (directly, or transitively through an op it
calls). Result: **of the 41 example-unused ops, 12 are used by doom and 29 are
used by neither.** Combined, **66 ops are live (example and/or doom) and 29 are
dead** to both — see the [doom breakdown](#of-the-41-example-unused-ops-which-does-doom-use-12) below.

`D` = directly referenced in an example · `I` = only reached indirectly
(witness shown) · **NO** = not reachable from any example.

| file | op | example use |
|---|---|---|
| **arithmetic_ops** | add, subtract, negate, add_const, bool_to_01, add_scaled_nodes, sum_nodes, concat, relu_add, compare, min, multiply_integers | **D** |
| | multiply_const | **I** `bool_to_01 → multiply_const` |
| | abs | **I** `min → abs` |
| | square | **I** `multiply_integers → square` |
| | square_signed *(not in `__init__`)* | **I** `multiply_integers → square_signed` |
| | thermometer_floor_div | **I** `number_to_digit_scalars → thermometer_floor_div` |
| | piecewise_linear | **I** `… → thermometer_floor_div → piecewise_linear` |
| | reciprocal | **I** `count_since_marker → reciprocal` |
| | relu, max, piecewise_linear_2d, multiply_2d, low_rank_2d, mod_const, linear_bin_index, clamp\*, log, exp, log_abs, floor_int, ceil_int, signed_multiply, reduce_min, reduce_max | **NO** |
| **attention_ops** *(none in `__init__`)* | attend_argmin_above_integer, attend_argmin_unmasked, attend_argmax_dot, attend_to_offset, get_prev_value | **D** |
| | attend_mean_where | **I** `count_since_marker → attend_mean_where` |
| | attend_argmin, attend_argmax, attend_argmin_where, attend_argmax_where, attend_argmin_above_in_bucket, attend_argmin_valid_unmasked, attend_argmax_dot_where, attend_argmin_dot_where, attend_most_recent_matching | **NO** |
| **embedding_arithmetic** | sum_digit_seqs | **D** (adder.py) |
| | sum_digits | **I** `sum_digit_seqs → sum_digits` |
| | subtract_digits, subtract_digit_seqs, compare_digit_pair, compare_digit_seqs, multiply_digit_pair, multiply_digit_seqs | **NO** ⚠️ dead |
| **global_recency** *(none in `__init__`)* | global_position_from_bos, attend_most_recent_globally | **NO** |
| **inout_nodes** | create_input, create_literal_value, create_embedding, create_onehot_embedding\*, create_unembedding, create_rope_config | **D** |
| **linear_relu_linear** | linear_relu_linear | **I** `equals_vector → linear_relu_linear` |
| **logic_ops** | bool_any_true, bool_all_true, bool_not, equals_vector, cond_gate | **D** |
| | per_column_offsets *(not in `__init__`)* | **I** `cond_gate → per_column_offsets` |
| | cond_add_vector | **NO** |
| **loop_ops** | unrolled_loop | **NO** |
| **map_select** | map_to_table, switch, select, in_range | **D** |
| | table_lookup_2d, table_lookup_3d, dynamic_extract, broadcast_select | **NO** |
| **marker_count** | count_since_marker *(not in `__init__`)* | **D** |
| **onehot_table** | onehot_lookup *(not in `__init__`)* | **D** |
| **quantization** *(none in `__init__`)* | quantize_to_range, dequantize_from_range | **NO** |
| **scalar_encoding** | digit_to_scaled_scalar, digits_to_number, number_to_digit_scalars, scalar_to_embedding | **D** |
| **sequence_ops** | check_is_digit, NumericSequence, output_sequence, remove_leading_0s | **D** |

\* not re-exported by `ops/__init__.py`.

### Of the 41 example-unused ops, which does DOOM use? (12)

These have no example but **are load-bearing for doom** — deleting any would
break the renderer:

| op | file | how doom reaches it |
|---|---|---|
| multiply_2d | arithmetic_ops | direct import |
| piecewise_linear_2d | arithmetic_ops | **indirect** — `multiply_2d → piecewise_linear_2d` |
| mod_const | arithmetic_ops | direct import |
| clamp | arithmetic_ops | direct import |
| floor_int | arithmetic_ops | direct import |
| ceil_int | arithmetic_ops | direct import |
| attend_argmin_above_in_bucket | attention_ops | direct import |
| attend_most_recent_matching | attention_ops | direct import |
| global_position_from_bos | global_recency | direct import |
| table_lookup_2d | map_select | direct import |
| dynamic_extract | map_select | direct import |
| broadcast_select | map_select | direct import |

### Dead — used by neither examples nor DOOM (29)

Nothing in `examples/` or `torchwright_doom/` reaches these, directly or
transitively. From your standpoint they are dead weight (most still carry unit
tests, so deletion would trip `tests/` — but no example or renderer depends on
them):

| file | dead ops |
|---|---|
| **arithmetic_ops** (10) | relu, max, low_rank_2d, linear_bin_index, log, exp, log_abs, signed_multiply, reduce_min, reduce_max |
| **attention_ops** (7) | attend_argmin, attend_argmax, attend_argmin_where, attend_argmax_where, attend_argmin_valid_unmasked, attend_argmax_dot_where, attend_argmin_dot_where |
| **embedding_arithmetic** (6) | subtract_digits, subtract_digit_seqs, compare_digit_pair, compare_digit_seqs, multiply_digit_pair, multiply_digit_seqs |
| **global_recency** (1) | attend_most_recent_globally |
| **logic_ops** (1) | cond_add_vector |
| **loop_ops** (1) | unrolled_loop |
| **map_select** (1) | table_lookup_3d |
| **quantization** (2) | quantize_to_range, dequantize_from_range |

Notable whole-module / whole-family dead zones:

- **`quantization` is entirely dead** — neither `quantize_to_range` nor
  `dequantize_from_range` is reached by any example or doom.
- **The `embedding_arithmetic` subtract/compare/multiply families are dead.**
  The calculator examples re-implement digit-sequence subtract/compare/multiply
  **locally** instead of importing these ops (and `scripts/calculator_stats.py`
  imports the *example* versions, not the ops). Only the `sum_*` family is live,
  via `adder.py`. These six have no caller at all except each other's internal
  chain and the `__init__.py` re-export.
- **The plain argmin/argmax/`…_where`/`…_dot_where` attention variants are dead**
  — doom selects with the *dot* and *above-in-bucket* variants instead, so 7 of
  the 15 attention ops are orphaned.
- **The `log`/`exp`/`log_abs` family, `reduce_min`/`reduce_max`,
  `signed_multiply`, `low_rank_2d`, `linear_bin_index` are all dead.**
  (`low_rank_2d` was a doom build-cost stopgap that has since been removed —
  see the comment at `torchwright_doom/render_ops.py:484`.)
- The op-function **`relu` is dead** — every real ReLU in the graphs comes from
  the `ReLU` *node class* and `linear_relu_linear`, never the `relu` op.

---

## Q2 — Keyword arguments never set to a non-default

Roughly half the ops have **no keyword arguments at all** (pure `node → node`,
e.g. `add`, `abs`, `concat`, all the digit ops, the `bool_*` family). **29 ops
have at least one keyword argument that is never set to a non-default value**
anywhere in torchwright or doom. Two parameter names dominate the dead list:

- **`d_max` (=1024)** — the per-sublayer neuron cap — is dead on **11 ops**
  (`square`, `square_signed`, `reciprocal`, `log`, `exp`, `log_abs`,
  `signed_multiply`, `piecewise_linear_2d`, `low_rank_2d`, `table_lookup_2d`,
  `table_lookup_3d`). It is only ever forwarded internally at its default; no
  caller tunes it. (It *is* live on `piecewise_linear` and `multiply_2d`, where
  tests pass `d_max=4`.)
- **`assert_hardness_gt` (=None)** — the softmax-hardness debug assertion — is
  dead on **11 attention ops**. The only op that exercises it is
  `attend_argmin_above_in_bucket` (a test helper passes `assert_hardness_gt=0.99`);
  every sibling leaves it `None`.

### Full never-overridden list (one row per op; ✅ = corrected after re-verification)

| op | never-overridden kwarg(s) (default) | note |
|---|---|---|
| piecewise_linear_2d | `d_max`(1024) | only internal forward at default |
| low_rank_2d | `multiply_steps_per_axis`(20), `max_abs_output`(None), `d_max`(1024) | op is doom-only; none of its knobs are tuned |
| square | `d_max`(1024) | `step` is live |
| square_signed | `step`(1.0), `d_max`(1024) | only reached via `signed_multiply(strategy="shallow")` — **no `"shallow"` call exists anywhere** |
| reciprocal | `d_max`(1024) | `step`, ranges live |
| log | `section_factor`(10.0), `d_max`(1024) | `n_breakpoints` live |
| exp | `d_max`(1024) | `n_breakpoints` live |
| log_abs | `n_breakpoints`(256), `section_factor`(10.0), `d_max`(1024) | `min_abs`/`max_abs` live |
| signed_multiply | `d_max`(1024) | `step`, `max_abs_output`, `strategy` live |
| linear_bin_index | `name`("linear_bin_index") | all numeric knobs live |
| attend_argmin | `assert_hardness_gt`(None) | always called with leading positional args only |
| attend_argmax | `assert_hardness_gt`(None) | " |
| attend_argmin_where | `assert_hardness_gt`(None) | " |
| attend_argmax_where | `assert_hardness_gt`(None) | " |
| attend_argmin_above_integer | `assert_hardness_gt`(None) | " |
| attend_argmin_unmasked | `assert_hardness_gt`(None) | " |
| attend_argmin_valid_unmasked | `assert_hardness_gt`(None) | " |
| attend_argmax_dot | `assert_hardness_gt`(None) | `match_gain` live |
| attend_argmax_dot_where | `match_gain`(200.0), `assert_hardness_gt`(None) | single all-positional test call |
| attend_argmin_dot_where | `match_gain`(200.0), `assert_hardness_gt`(None) | two all-positional test calls |
| get_prev_value | `recency_gain`(_LOCAL_RECENCY_GAIN) | kw-only; never passed |
| attend_most_recent_matching | `recency_gain`(_LOCAL_RECENCY_GAIN), `assert_hardness_gt`(None) | `match_gain`, `exclude_self` live |
| global_position_from_bos | `n_breakpoints`(_N_BPS) | kw-only; never passed |
| attend_most_recent_globally | `match_gain`(200.0), `recency_scale`(_RECENCY_SCALE), `assert_hardness_gt`(None) | only `exclude_self` is ever set |
| cond_gate | `c_tol`(0.005) | kw-only; `approximate` is live |
| table_lookup_2d | `d_max`(1024) | `name` is set only by an internal forward (`"table_lookup_3d_2d"`) |
| table_lookup_3d | `d_max`(1024), `name`("table_lookup_3d") | `index_scale`/`sharpness`/`outer_axis` live |
| select | `c_tol`(0.005) | ✅ **`approximate` is LIVE** — see correction below |
| broadcast_select | `c_tol`(0.005) | `approximate` is live |

**✅ Correction.** The automated pass reported `select.approximate` as
never-overridden. That is **wrong** — `tests/ops/test_map_select.py:40` and
`:58` call `select(…, approximate=False)` (default is `True`). The agent
misattributed those lines to `broadcast_select`/`cond_gate`. So
`select.approximate` is **live**; only `select.c_tol` is dead. All other
never-overridden claims reproduced under hand re-grep, including positional
checks for the attention ops.

### Ops whose every keyword argument is exercised somewhere

`sum_nodes`(max_fanout), `compare`(true_level/false_level/sharpness),
`piecewise_linear`(clamp/d_max/input_scale/name), `multiply_2d`(all 9),
`floor_int`/`ceil_int`(sharpness), `multiply_integers`(strategy),
`attend_argmin_above_in_bucket`(assert_hardness_gt), `attend_to_offset`(delta_pos),
`create_input`(width/value_range), `create_rope_config`(d_head/max_positions),
`quantize_to_range`/`dequantize_from_range`(n_levels), plus name-only
`create_literal_value`/`linear_relu_linear`.

---

## The two things worth acting on

1. **Dead ops (29):** unused by every example *and* by doom — the full list is
   in the table above. The fattest targets: the entire `quantization` module,
   the six `embedding_arithmetic` subtract/compare/multiply functions (the
   calculator examples shadow them with local copies), the seven orphaned
   attention variants, and the `log`/`exp`/`log_abs` family. Either delete them
   or, where the duplication is the cause (digit-seq ops), wire the consumer to
   the op instead of its local copy.
2. **Dead knobs:** `d_max` (11 ops) and `assert_hardness_gt` (11 attention ops)
   are parameters no caller ever touches. If they're intended as escape
   hatches, keep them; otherwise they're surface area that could be dropped.
