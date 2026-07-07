# Supported `d_model` widths with RMSNorm on

ONNX exports carry a real RMSNorm by default (`compile_to_onnx`'s
`rms_norm=None` resolves to on). With the norm on, the residual width
`d` must be a **supported width**:

> **Any multiple of 1024 up to 16384, or any power of two.**

That covers the familiar LLM widths — 4096 (Llama-8B), 5120
(Llama-13B / Qwen-14B), 8192 (Llama-70B) — and the small powers of two
(64, 128, 256, 512) used by test artifacts. Anything else raises a
`ValueError` at the top of `compile_to_onnx`, before the streaming
compile starts; pass `rms_norm=False` to deliberately export without
the norm.

| `d_model` | factors as | reserved columns |
|---:|---:|---:|
|  1024 |  1·2^10 | 1 |
|  2048 |  1·2^11 | 2 |
|  3072 |  3·2^10 | 3 |
|  4096 |  1·2^12 | 1 |
|  5120 |  5·2^10 | 2 |
|  6144 |  3·2^11 | 3 |
|  7168 |  7·2^10 | 4 |
|  8192 |  1·2^13 | 2 |
|  9216 |  9·2^10 | 3 |
| 10240 |  5·2^11 | 4 |
| 11264 | 11·2^10 | 5 |
| 12288 |  3·2^12 | 3 |
| 13312 | 13·2^10 | 4 |
| 14336 |  7·2^11 | 5 |
| 15360 | 15·2^10 | 6 |
| 16384 |  1·2^14 | 1 |

Powers of two outside this table (any `2^k`) reserve 1 column when `k`
is even, 2 when `k` is odd.

`tests/docs/test_rms_norm_dmodel_doc.py` asserts every row of the
table above against the compiler's actual layout — if this file and
the code disagree, that test fails.

## Why widths are constrained at all

The compiled RMSNorm must be a **bit-exact identity**: it is a real
norm in the artifact (mean of squares, sqrt, divide, gain), but the
graph's values must pass through it unchanged, bit for bit.

The trick: a few residual columns are reserved — never allocated to
any node — and hold large pinned constants, seeded through the
embedding table so they are present at every position. Their combined
energy dominates the norm's mean-of-squares and forces it to land
exactly on an even power of two `2^(2m)`. From there every step is
exact in fp32: `sqrt(2^(2m)) = 2^m` exactly, and dividing by `2^m`
then multiplying by the uniform gain `2^m` are pure exponent shifts
that return every input bit unchanged.

Forcing the mean onto a power of two requires pinned energy
`E = d·2^(2m)`. Factor `d = c·2^k` with `c` odd: the layout writes
`E = c·2^(k+2m)` as a sum of powers of two, one or two reserved
columns per set bit of `c` (`_rms_norm_pinned_layout` in
`compiler/forward/compile.py`). Small odd factors therefore cost only
a handful of columns — that is the entire table above.

Two further conditions are checked at compile time and raise loudly
if violated:

- **fp32 mean arithmetic.** Whether the runtime computes the mean by
  dividing the sum by `d` or by multiplying with a rounded reciprocal
  `1/d` (ONNX `ReduceMean`'s internal strategy is not contractual),
  the result must land exactly on `2^(2m)`. Every width in the
  contract passes both ways; the first odd factor that fails at all
  is 41.
- **Data energy budget.** The graph's residual energy must stay below
  the smallest pinned column's rounding threshold
  (`Σ data² < 2^(2q−24)`, `q = rms_norm_const_exp`), or the fp32 sum
  drifts off the pinned energy and the identity silently breaks.
  `_certify_rms_norm_energy` proves this from the compiler's static
  value ranges on every norm-on compile and names the smallest
  sufficient `q` when it fails.

The mechanism generalizes past the contract (any odd factor whose
fp32 arithmetic checks out is buildable — `forward_compile` accepts
such widths directly), but `compile_to_onnx` deliberately promises
only the small, memorable set above.
