# Task 3 - torchwright bucketed argmin primitive design

**Status:** design for implementation; not implemented.

This is the torchwright-owned half of the Task 3 split. It defines a
general attention primitive for "pick the minimum local score among
valid rows whose bucket matches a runtime query bucket and whose local
score is strictly above a runtime threshold."

It deliberately does **not** define DOOM screen-column sentinels,
`SolidIntervals`, radix decomposition, or fallback successor logic. The
downstream caller owns those.

## Goal

Add one reusable op in `torchwright/torchwright/ops/attention_ops.py`:

```python
def attend_argmin_above_in_bucket(
    pos_encoding: PosEncoding,
    score: Node,
    validity: Node,
    key_bucket_onehot: Node,
    score_above_each_threshold: Node,
    query_bucket_onehot: Node,
    threshold_onehot: Node,
    value: Node,
    assert_hardness_gt: Optional[float] = None,
) -> Node:
    ...
```

The op returns `value` at the selected past row. It is a vanilla
attention head with explicit Q/K/V/O matrices, like the existing
selection primitives.

`pos_encoding` is accepted for API symmetry with the other attention
ops. The proposed construction does not read position columns: this op is
purely content-based (no recency term), and causal masking -- attend only
to past rows -- is applied by the attention framework, not encoded in Q/K.

## Input contract

All inputs are per-position graph nodes -- one value per token-stream
position.

Terminology: a **row** below means one token-stream position (one past
token = one row). The head looks back over earlier rows, and the two tables
introduced below carry one row per position -- so "row" is the natural
word. Read it as "position" if you prefer; they are the same thing.

Two of the filters -- bucket and threshold -- depend on the row AND the
query at the same time: the same row passes one query and fails another, so
neither can be a plain per-row flag. Both use the SAME mechanism: a small
table of pre-answered yes/no questions, one row per position and one column
per choice, where the query names a column and the attention dot product
reads out that one cell. The two tables differ only in how they are filled.

Read straight off each row:

- `score`: width-1 scalar. Lower score wins among rows that pass all three
  filters. For Task 3 this is a local radix digit -- integer-valued, unit
  gaps.
- `validity`: width-1, torchwright's convention `+1.0` = valid /
  `-1.0` = invalid. An invalid row must never beat a valid row that passes
  the query's filters.

Bucket table -- equality, "is this row in the group the query asked for?"
One column per group; each row has a single `1`, in its own group's column:

```text
               grp0  grp1  grp2  grp3
  row A (g2):    0     0     1     0
  row B (g0):    1     0     0     0
```

- `key_bucket_onehot`: width `n_buckets`. One row of this table -- the row's
  own group, as a 0/1 one-hot.
- `query_bucket_onehot`: width `n_buckets`. The column picker -- a 0/1
  one-hot naming the group this query wants.
  `dot(query_bucket_onehot, key_bucket_onehot)` is `1` exactly when the
  groups match, else `0`.

Threshold table -- greater-than, "is this row's score above the threshold
the query asked for?" One column per possible threshold; each row is a run
of `1`s that stops at its score (a ruler filled up to the score):

```text
                       >t0   >t1   >t2   >t3   >t4   >t5  ...
  row A (score 5):       1     1     1     1     1     0   ...
  row B (score 2):       1     1     0     0     0     0   ...
```

- `score_above_each_threshold`: width `n_thresholds`. One row of this table
  -- slot `c` is `1` iff this row's `score` is strictly above threshold
  `t_c`. The threshold values `t_0, t_1, …` live only in how the caller
  fills this row; the op never sees them, which is what keeps it reusable.
- `threshold_onehot`: width `n_thresholds`. The column picker -- a 0/1
  one-hot naming which threshold this query wants.
  `dot(threshold_onehot, score_above_each_threshold)` reads out that one
  precomputed answer.

The only difference between the two tables: a bucket row has a single `1`
(you are in exactly one group), a threshold row is a run of `1`s (you are
above every threshold up to your score). Same machine otherwise.

Payload:

- `value`: arbitrary-width payload, read from the selected row.

Required shape checks:

- `len(score) == 1`
- `len(validity) == 1`
- `len(key_bucket_onehot) == len(query_bucket_onehot)`
- `len(score_above_each_threshold) == len(threshold_onehot)`
- `len(value) >= 1`
- `n_buckets >= 1` and `n_thresholds >= 1` (a zero-width table would
  silently degrade the op to a plain validity+score selection)

Expected semantic checks. Validate these in EXACT-MATH tests
(`node.compute`), NOT as runtime asserts on compiled values: compiled table
vectors are only NEAR-clean. The softening comes from the encoding step and
the softmax value-recovery that reads them back through an attention head
-- NOT from `thermometer_floor_div`, which is exact on integer inputs. Pin
the actual degraded value with a probe rather than assuming a constant. An
"exactly one-hot" runtime assert would fire on this benign compiled noise
-- the FP-nondeterminism-at-tolerance trap. The op TOLERATES near-clean
inputs by design: the predicate-bonus margin absorbs a degraded dot.

- both bucket vectors are 0/1 and one-hot for ordinary matched queries;
- `threshold_onehot` is 0/1 and one-hot; `score_above_each_threshold` is
  0/1 and monotone (a prefix run of `1`s -- NOT a one-hot);
- `validity` is near `+1` or `-1`;
- `score` has a bounded DIAMETER (max - min across the rows a query
  compares) small enough that the predicate bonuses dominate the gained
  score term: `_QUERY_GAIN * diameter < min(2*_VALIDITY_BONUS,
  _BUCKET_BONUS, _ABOVE_MATCH_BONUS)`, i.e. diameter `< 32` at the shipped
  constants (see "Constant sizing and precision"). Past that, a
  filter-failing low-score row can silently outscore a match.

The op should not depend on DOOM's `SCREEN_WIDTH` or radix base.

## Logit formula

For query position `j` and key position `i`, the desired logit is:

```text
_QUERY_GAIN * (-score_i)
    + _VALIDITY_BONUS * validity_i
    + _BUCKET_BONUS * dot(query_bucket_onehot_j, key_bucket_onehot_i)
    + _ABOVE_MATCH_BONUS * dot(threshold_onehot_j, score_above_each_threshold_i)
```

`_VALIDITY_BONUS`, `_BUCKET_BONUS`, and `_ABOVE_MATCH_BONUS` are three new
OP-LOCAL constants for this primitive. Do NOT reuse the IDENTIFIERS or the
values of the existing `_VALIDITY_DIRECT` / `_ABOVE_BONUS` (both `1000`)
globals -- in particular do not name a constant `_ABOVE_BONUS`: that name
already exists in `attention_ops.py` and is read by
`attend_argmin_above_integer`, so a second module-level `_ABOVE_BONUS`
would silently rebind it and undersize the sibling op. Size each one
MINIMALLY -- just above this op's worst-case gained score swing
`_QUERY_GAIN * S`, so it dominates a single-predicate miss. For the Task 3
range (`S <= 12`, so `_QUERY_GAIN * S = 96`), ~256 each is a sound choice.
That is deliberately an order of magnitude below the inherited `1000`, and
that is correct, not a precision compromise -- see "Constant sizing and
precision". Do not route bucket or above through the gained score column.

The validity term is `+/- _VALIDITY_BONUS`, giving a `2 * _VALIDITY_BONUS`
valid-vs-invalid swing.

## Matrix layout

Use decoupled identity V/O, unlike the current
`attend_argmin_above_integer` layout.

Logical Q/K width:

```text
d_qk = 2 + n_buckets + n_thresholds
```

Column layout:

```text
col 0:
    score term.
    Q reads LiteralValue([1.0]) with coefficient _QUERY_GAIN.
    K reads score with coefficient -1.0.

col 1:
    static validity term.
    Q reads LiteralValue([1.0]) with coefficient 1.0.
    K reads validity with coefficient _VALIDITY_BONUS.

cols 2 .. 2 + n_buckets - 1:
    bucket equality rendezvous.
    Q reads query_bucket_onehot[c] with coefficient _BUCKET_BONUS.
    K reads key_bucket_onehot[c] with coefficient 1.0.

cols 2 + n_buckets .. 2 + n_buckets + n_thresholds - 1:
    strict-above rendezvous.
    Q reads threshold_onehot[c] with coefficient _ABOVE_MATCH_BONUS.
    K reads score_above_each_threshold[c] with coefficient 1.0.
```

`query_in` should be:

```text
concat(LiteralValue([1.0]), query_bucket_onehot, threshold_onehot)
```

`key_in` should be:

```text
concat(score, validity, key_bucket_onehot, score_above_each_threshold)
```

`value_matrix = torch.eye(len(value))`
`output_matrix = torch.eye(len(value))`

This is the important width distinction from
`attend_argmin_above_integer`: `value` may be wide, but it must not
increase logical `d_qk`. The compiler can split wide V/O over physical
heads. That is not free in physical head count, but it is free for the
`d_head >= d_qk` constraint.

Head demand is implicit from the matrix shapes -- there is no head-demand
API to call or "report". The op just builds identity V/O of width
`len(value)` and a `d_qk`-wide Q/K (`d_qk = 2 + n_buckets + n_thresholds`);
the compiler reads `node.d_qk = query_matrix.shape[1]` and
`node.d_v = value_matrix.shape[1]` and derives the physical head count
itself as `ceil(d_v / d_head)`, padding one Q/K to `d_head` and sharing it
across the V/O chunk heads -- exactly as the identity-V/O sibling ops do.

The single hard constraint is `d_qk <= d_head`. It is enforced LATE, as an
assertion in the weight writer (`assert layer_d_head >= node.d_qk`, message
"Pass d_head>=... to compile_headless()") -- NOT at graph construction and
NOT in either scheduler, both of which size head reservations from `d_v`
alone and never look at `d_qk`. So at real scale
`d_qk = 2 + n_buckets + n_thresholds` can exceed a small default `d_head`
(`probe_graph` defaults to `d_head = 16`, but `B = 13` alone already pushes
`d_qk` well above 16). Compiled/probe tests MUST pass an explicit
`d_head >= d_qk`, or they fail to compile with that assertion rather than
with a selection error.

## Correctness argument

Let a "matching" row mean:

```text
validity_i == +1
bucket_i == query_bucket_j
score_i > threshold_j
```

For any two matching rows, all predicate bonuses are equal, so the
smaller `score` has the larger logit by `_QUERY_GAIN * score_gap`.
With integer local scores and unit gaps, `_QUERY_GAIN = 8` gives a
large enough softmax gap for hard selection.

A non-matching row misses at least one predicate:

- invalid row: loses `2 * _VALIDITY_BONUS` logit versus a valid row,
  because validity reads `- _VALIDITY_BONUS` instead of `+ _VALIDITY_BONUS`;
- wrong bucket: loses `_BUCKET_BONUS`;
- not above threshold: loses `_ABOVE_MATCH_BONUS`.

For a score range with diameter `S`, a matching row with the worst score
must still beat a non-matching row with the best score. The required
condition is:

```text
min(2 * _VALIDITY_BONUS, _BUCKET_BONUS, _ABOVE_MATCH_BONUS)
    > _QUERY_GAIN * S
```

For the Task 3 downstream use, `S <= B - 1`. At real scale `B = 13`, so
`_QUERY_GAIN * S <= 96`. With op-local bonuses at ~256 the margin is
comfortable (`min(512, 256, 256) = 256 > 96`). In the fp32 compiled path
this exact-arithmetic condition is essentially the whole story -- see
"Constant sizing and precision" for why finite precision does not erode it.

## Constant sizing and precision

Sizing. Each bonus only has to dominate a single-predicate miss against the
full score swing. With integer scores of diameter `S`, the worst-case
gained score swing is `_QUERY_GAIN * S`, so `min(2 * _VALIDITY_BONUS,
_BUCKET_BONUS, _ABOVE_MATCH_BONUS) > _QUERY_GAIN * S` is the whole sizing
requirement. For the Task 3 range (`S <= 12`, so `_QUERY_GAIN * S = 96`),
~256 each gives a ~2.7x margin.

That ~256 is an order of magnitude below the inherited `1000`, and that is
the CORRECT analog of the sibling convention, not a deviation from it: the
single-bonus siblings set their `1000` just above THEIR score swing
(`_VALIDITY_DIRECT` / `_ABOVE_BONUS` dominate `_QUERY_GAIN * _MAX_SCORE_ABS
= 8 * 120 = 960`). This op's score range is an order of magnitude smaller
(`S <= 12`, swing `96`), so the faithful minimal bonus is an order of
magnitude smaller too. Inheriting `1000` here would just be oversized.

Precision. The binding constraint is the exact-arithmetic separation above,
NOT a low-precision rounding budget. The compiled attention runs the K·Q
logit in fp32: every attention op compiles through the same component,
which routes `F.scaled_dot_product_attention` through the `SDPBackend.MATH`
kernel on fp32 tensors, and that kernel matches manual `softmax + matmul`
exactly. PyTorch's CUDA TF32 matmul flag is off in the supported
environment (a regression test pins both `float32_matmul_precision ==
"highest"` and `allow_tf32 == False`). The selection among matching rows is
decided by the SMALLEST gap in the logit -- the gained unit-score gap
`_QUERY_GAIN * 1 = 8`. In fp32 that gap sits thousands of ULP above the
rounding floor at any total logit in the low thousands, so it never blends,
at either `256` or `1000`.

The one documented precision risk is a FUTURE global switch to TF32, whose
coarser mantissa can erase an 8-logit tiebreak -- but only when the total
logit reaches the hundreds of thousands (see the precision-policy note on
`attend_most_recent_matching` in `attention_ops.py`, which carries a
`match_gain = 300_000` for exactly that reason). With ~256 bonuses the max
total logit is ~768 (all three bonuses present, score 0); even at `1000` it
is ~3000. Both are orders of magnitude below that threshold, so keeping the
bonuses small is mild future-proofing, not a present requirement.

(The module-header and `_QUERY_GAIN` comments in `attention_ops.py` still
say "bf16"; those are stale and contradict the fp32 precision-policy
docstring in the same file. Do not take them as the operative regime -- an
earlier draft of this doc did, and built a now-removed bf16 logit budget on
them.)

Corollary for `assert_hardness_gt`: the achievable selection hardness
between two ADJACENT-score matching rows is bounded by the score-gap term,
`softmax(_QUERY_GAIN * 1) = 1 / (1 + exp(-8)) ~= 0.9997` (it is ~1.0 when
the runner-up is a non-matching row, which loses a whole bonus). Do not set
`assert_hardness_gt` above that ~0.9997 ceiling, or it false-fails on
legitimate adjacent-score cases.

## Tie semantics

Tied matching rows produce the usual soft average of their `value`
payloads. This is acceptable only when the caller can tolerate that
blend.

For Task 3, duplicate matching starts are harmless if every duplicated
row carries the same selected start and predicate features. A caller
that carries row-specific payload beside the selected key must test
ties explicitly.

## No-match semantics

When no row satisfies all predicates, the output is undefined. The op
must not claim to return a present/missing boolean.

This is not a defect; it is the same contract as existing hard
selection primitives whose masks can be empty. The caller has two safe
choices:

1. Prove a matching row exists before consuming the output.
2. Carry enough key-side predicate features in `value` to recompute
   whether the selected row actually matched.

Important: recomputing no-match presence from averaged scalar fields is
not safe in general. If two wrong-bucket rows tie, their scalar bucket
ids can average to the query bucket id. A robust caller should carry
the selected `key_bucket_onehot` and selected `score_above_each_threshold`, then
test dot-products against the query one-hots. A true hard match gives a
dot near 1; a blend of non-matching one-hots does not.

## Degenerate key sets

The op must return a finite, defined value -- never NaN, never raise --
for the structural degenerate cases a reusable primitive will hit:

- a minimal causal window (the query's earliest positions, where only the
  causal self-row is visible);
- an all-invalid key set (every `validity == -1`).

In both the softmax is over equal/absent logits and the output is a blend
of whatever values are present. The caller separates these from a real
match via the presence recomputation, so the only contract on the op is
"finite and total." Do not special-case them in the op; a test that
confirms no NaN and no crash is enough.

## Downstream presence pattern

If a caller needs a reliable `present` bit, set:

```text
value = concat(payload, validity, key_bucket_onehot, score_above_each_threshold)
```

After attention:

```text
selected_valid = snap_bool(selected_validity)
selected_bucket_match =
    dot(selected_key_bucket_onehot, query_bucket_onehot) > 0.9
selected_above =
    dot(selected_score_above_each_threshold, threshold_onehot) > 0.9

present = selected_valid && selected_bucket_match && selected_above
```

The threshold should be comfortably above 0.5 and below the expected
compiled value for a true one-hot match. Use tests/probes to pin the
exact compare tolerance.

`snap_bool` and `dot` above are illustrative, not existing ops: build the
bool snap from `compare` / `bool_to_01` and the one-hot dot from a
per-component `Linear`. This presence recomputation is caller-owned.

## Tests

Add focused op tests in `torchwright/tests/ops/test_attention_ops.py`.

Required `node.compute` tests:

- picks the lowest score among rows that are valid, in bucket, and
  above threshold;
- ignores rows that are valid and above threshold but in the wrong
  bucket;
- ignores rows that are valid and in bucket but not above threshold;
- ignores rows that are invalid but otherwise match;
- handles query bucket varying per position;
- handles query threshold varying per position;
- accepts arbitrary `value` width without increasing `Attn.d_qk`;
- duplicate matching rows with identical payload blend harmlessly.

Required edge tests:

- bucket boundary: query bucket `k`, candidates in `k - 1`, `k`, and
  `k + 1`;
- threshold boundary: score equal to threshold is not above;
- threshold at maximum local digit has no match;
- all rows invalid has no correctness assertion on payload, but should
  not crash.

Required negative/presence tests:

- demonstrate why scalar bucket recomputation is unsafe under no-match
  blends;
- demonstrate the one-hot predicate-feature recomputation stays false
  in the same case.

Required compile/probe tests:

- Adjacent-score selection probe (compiled, load-bearing): a COMPILED
  fixture -- compiled with an explicit `d_head >= d_qk` -- with two
  matching rows whose local scores differ by 1, at the FULL predicate-bonus
  stack (validity + bucket + above all present), asserting the lower-score
  row is recovered cleanly. This is the worst case for hardness: all three
  bonuses are equal between the two rows, so only the gained 8-unit score
  gap separates them, and the test confirms that gap still concentrates
  softmax in the real fp32 compiled path. It must run compiled, not just
  `node.compute`. (It does NOT pin the bonus magnitudes -- in fp32 it
  passes identically at 256 and at 1000; use the matrix-inspection test
  below for that.)
- Bonus-magnitude inspection (the test that actually pins the constants):
  construct the `Attn` and read the validity / bucket / above coefficients
  out of `query_matrix` / `key_matrix`; assert they equal the chosen
  op-local constants and are NOT `1000.0`. The compiled probe cannot
  enforce this in fp32, so this matrix-inspection test is the only guard
  against an implementor accidentally inheriting the 1000-unit globals.
- `assert_hardness_gt=0.99` on representative matched cases in EXACT-MATH
  (note the ~0.9997 score-gap hardness ceiling above -- do not set it
  higher). A separate COMPILED hardness probe (`probe_attention` on the
  winning key) is redundant for this op: the SDPA MATH backend reproduces
  the oracle softmax in fp32 to ~1e-5, so compiled hardness equals the
  exact-math hardness already checked, and the tight-`atol` compiled parity
  probe catches any compiled divergence. Skip it unless debugging.
- near-one-hot tolerance: feed bucket/threshold one-hots perturbed to a
  representative near-one-hot value (e.g. ~0.97) and confirm selection is
  unchanged;
- degenerate inputs: all-invalid and minimal (self-row-only) key sets
  return a finite value and do not raise;
- width regression: inspect the constructed `Attn` and assert
  `attn.d_qk == 2 + n_buckets + n_thresholds`;
- V/O regression: `attn.d_v == len(value)` and value identity matrices
  are decoupled from `d_qk`.

## Implementation checklist

- Add the op to `attention_ops.py`.
- Export it from `torchwright/torchwright/ops/__init__.py` if local
  convention requires.
- Add tests in `torchwright/tests/ops/test_attention_ops.py`.
- Do not add DOOM-specific constants, sentinels, or radix assumptions.
- Do not use a value-tail layout as a template; use identity V/O. Three
  existing ops ride value inside `d_qk` -- `attend_argmin_above_integer`,
  `attend_argmin_unmasked`, `attend_argmin_valid_unmasked` -- none of them
  is the template here.
- Use distinct OP-LOCAL bonus constants (`_VALIDITY_BONUS`, `_BUCKET_BONUS`,
  `_ABOVE_MATCH_BONUS`) sized just above this op's score swing
  (`_QUERY_GAIN * S = 96`) -- ~256. Do NOT reuse the identifiers OR the
  values of the existing `_VALIDITY_DIRECT` / `_ABOVE_BONUS` (= 1000)
  globals; naming a new `_ABOVE_BONUS` would rebind the one
  `attend_argmin_above_integer` reads. Pin the magnitudes with the
  matrix-inspection test; confirm hard selection with the compiled
  adjacent-score probe (compiled at `d_head >= d_qk`).
- Validate one-hot / validity / score-range semantics in exact-math tests
  only; do NOT runtime-assert exact one-hot on compiled values.
- Guarantee finite, non-raising output for all-invalid and minimal key
  sets.
- Do not add a `TargetOp` for this attention primitive unless the
  measurement system already has attention-op targets. The downstream
  arithmetic used to build bucket digits is measured separately.
- Optional cleanup while you are in `attention_ops.py`: the module-header
  comment and the `_QUERY_GAIN` comment still reference "bf16", which
  contradicts the live fp32 precision-policy docstring on
  `attend_most_recent_matching` in the same file. Those stale comments are
  what an earlier draft of this doc inherited; fixing them prevents the
  next reader from repeating it.
