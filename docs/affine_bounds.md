# Affine bound propagation

Design of the affine-bound system that replaced scalar `Range`
propagation as the source of truth for value ranges. This
document describes the completed system — not the steps to get
there.

---

## Why this exists

The scheduler compiles a computation graph into transformer weights.
Several graph-level ops derive numerical-stability constants from
their inputs' declared bounds:

- `cond_gate` picks its cancellation offset `M` from `max |input|`.
- `assert_in_range` attaches a runtime predicate and a static range
  claim to the node itself.
- `attend_mean_where` re-claims the value input's range on its
  output (a mean is a convex combination, so the range survives).

(`floor_int`, by contrast, takes its breakpoint range as explicit
`min_value` / `max_value` arguments from the caller and reads no
bounds — it is unaffected by the bound system.)

`NodeValueType.value_range` is a single `Range(lo, hi)` that
every component of a node's output shares. In the pre-affine
system, each op's `compute_value_type()` propagated it by scalar
interval arithmetic — if `u.value_range` was `[a, b]` and
`v.value_range` was `[c, d]`, then `(u + v).value_range` was
`[a + c, b + d]`. (In the shipped system most primitives'
`compute_value_type()` return `unknown()` and the affine channel
carries the range — see *Bound computation lifecycle*.) That
propagation was sound, but it treated every pair of inputs to
every op as independent. In a torchwright graph, inputs are rarely
independent.
The residual stream carries the same node through many downstream
consumers; `cond_gate` adds `M` and a later `Add` subtracts it;
sum chains are frequently dominated by correlated or identical
operands.

The primary workaround in the pre-affine codebase was
`linear_output_range` (since removed), a helper called by
`Linear.compute_value_type()`. It performed per-column interval
arithmetic over the weight matrix (each output component's
contribution computed independently, then the componentwise
min/max aggregated into a single `Range`). This recovered some
tightness inside each `Linear`, but the per-component information
was lost at the aggregate `Range` boundary — downstream `Linear`s
consuming that output saw only the scalar union, so the recovered
tightness didn't compose through the graph.

The motivating cancellation pattern is `cond_gate`: add a large
offset `M`, do gated work, subtract `M`. The two `M` constants
should cancel exactly, but under scalar interval arithmetic each
side contributes `±M` independently, widening the output range by
`2M`. The compiler picks `M = 2 · max|input|` (a safety factor
baked into `_max_abs_or_raise`) — so an IBP-loose bound forces a
wider `M`, which in turn forces more numerical precision loss in
the gate's MLP. Tightening the bound tightens the constant.

Worked-example sketch. Say an input node `x` has declared range
`[-10, 10]`. The compiler picks `M = 20`. Inside `cond_gate` a
subgraph computes `y = ReLU(x + M) = ReLU(x + 20)` (always
non-negative since `x + 20 ∈ [10, 30]`), then `z = y - M = y - 20`.
The true value of `z` equals `x`. Under scalar interval arithmetic:
`y.value_range = [10, 30]`; `z.value_range = [10 - 20, 30 - 20] =
[-10, 10]` — correct in this simple case because the straddling
case is avoided, but the `±M` contributes in other `cond_gate`
patterns that do cross zero. Under affine bounds: `y.bounds` is the
affine expression `x + 20`; `z.bounds` subtracts the constant `20`
from the offset, yielding the affine expression `x`. Downstream
`Linear`s that consume `z` see the same `A` matrix as they would
for `x` directly — no accumulated looseness from the offset round
trip. This is the structural pattern that extends to every `Add`
whose operands share upstream structure.

This is the failure mode: scalar interval arithmetic can't see
cancellation between correlated operands.
Affine bound propagation fixes that at the abstraction layer —
`Linear`, `Add`, `ReLU`, and `Concatenate` all become exact (modulo
`ReLU`'s linear envelope — a pair of affine functions that
sandwich the true `ReLU` output when the input straddles zero; see
the `ReLU` rule below), and downstream constants come out tighter
as a consequence. The per-component min/max logic that
`linear_output_range` provided is now redundant and has been
removed.

### Why forward propagation

The bound-propagation literature distinguishes forward-mode (each
node's affine bound is computed from its parents' affine bounds,
cached, and reused by descendants) from backward-mode (given a
specific query node, walk back through its ancestors composing
relaxations to derive that node's bound). Backward-mode is tighter
per query — because it picks a relaxation optimized for the
specific output — but it re-walks the graph per query.

Torchwright's per-node `compute_value_type()` contract fits
forward-mode natively: bounds are computed once, eagerly at
construction time, and stored on each node where every
bound-consumer op picks them up by reading `node.value_type`.
Backward queries would pay re-propagation cost per consumer,
which there are many of (one per `cond_gate`, `assert_in_range`,
`attend_mean_where` call in the compiled graph).
Tighter-per-query backward-mode is a real option to consider if
forward bounds prove insufficient in practice, but it would
require restructuring bound computation around query nodes and is
deliberately not part of v1.

---

## The basis

Every affine expression in the system is written over a set of
variables called **the basis**. The term is local to this system;
it is unrelated to the "residual-stream column basis" used
elsewhere in the codebase. The basis consists of the components
of every **source node** in the graph — `InputNode` and `Embedding`
nodes. Position contributes no basis variable: it is not a residual
feature but a rotation applied to Q and K inside attention (the
RoPE substrate in `torchwright/graph/rope.py`, which replaced the
old `PosEncoding` node).

Each `AffineBound` carries a **self-keyed** column map: a
`columns` dict mapping each source node's `node_id` to its
`(start_col, width)` in the A matrices, and an `input_ranges`
dict mapping each source node's `node_id` to its `(lo, hi)`
declared range. There is no shared `Basis` object; column layouts
are local to each bound and are merged on demand when binary ops
combine bounds from different source paths.

A given bound's `n_cols` is typically much smaller than the total
number of source-node components in the graph, because each bound
only tracks the source nodes that are reachable upstream. As
bounds merge through `Add` or `Concatenate`, `n_cols` grows and
the A matrices are scattered into a wider common layout via
`AffineBound.align()`.

### Position semantics

Torchwright's `compute_value_type()` is position-agnostic: it
produces one `NodeValueType` per node that describes every
component of the node's output at every token position. The affine
bound system inherits this. A basis variable represents the
per-position value of its corresponding source-node component,
with an interval wide enough to hold every possible value at every
position. An affine expression `A[i, :] · x + b[i]` describes the
node's value at component `i` at *whatever position it is evaluated
at*, in terms of that same position's source-node-component values.

Soundness follows pointwise: for any concrete position `p` and any
concrete sample of source-node values at that position drawn from
the basis box, the node's per-position value at component `i` lies
in the interval derived from the affine expression at those input
values. Because every position draws from the same basis box, the
derived component interval covers every position simultaneously.

The `Attn` rule leverages this: the attention output at position
`p` is a function of source-node values at positions
`0, 1, ..., p` — not just position `p`. The rule handles this by
observing that softmax produces convex-combination weights, so
per-component affine bounds on `value @ V` carry through unchanged
(a convex combination of values that each satisfy the same
per-component affine bound also satisfies that bound). The Q/K
logits, which determine which positions contribute, are not
tracked — only the value input's coefficients propagate.

### Sizing

For a given bound, `n_cols` is on the order of tens to low
hundreds — game-state inputs, cursor positions, flags,
embedding components from upstream. Per-node storage is
`[d_output × n_cols]` floats per bound, so typical cost per node
stays modest. Bound coefficients are held in `torch.float64`
regardless of the transformer's forward-pass dtype;
scalar-interval error compounds over many `Linear` / `Add`
steps and the extra precision buys headroom for free. This
assumes weight matrices with bounded spectral norm (condition number
roughly O(1) per `Linear`), which is the regime compiled torchwright
graphs are built to stay in. Graphs that stack many `Linear`s with
large weight norms would see bound error accumulate geometrically
regardless of dtype; the compiler's numerical-stability discipline
— which is part of why this tightening effort matters — keeps
that regime out of scope.

---

## `AffineBound`

```python
@dataclass
class AffineBound:
    """Per-component affine lower/upper bounds with self-keyed basis.

    For a node with width ``d_output``:

        lower(x)[i] = A_lo[i, :] · x + b_lo[i]
        upper(x)[i] = A_hi[i, :] · x + b_hi[i]

    with the invariant, for every x in the basis box:

        lower(x)[i] ≤ node_output[i] ≤ upper(x)[i]
    """
    A_lo: torch.Tensor          # shape [d_output, n_cols]
    A_hi: torch.Tensor          # shape [d_output, n_cols]
    b_lo: torch.Tensor          # shape [d_output]
    b_hi: torch.Tensor          # shape [d_output]
    columns: Dict[int, Tuple[int, int]]
        # Maps source node_id → (start_col, width)
    input_ranges: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
        # Maps source node_id → (lo, hi) declared range
```

Each `AffineBound` carries its own column layout and input ranges
— there is no shared `Basis` object. The `columns` dict says
which columns of the A matrices correspond to which source node's
components; `input_ranges` stores the declared range for each
source node. Binary ops (`Add`, `Concatenate`) merge their
operands' column maps via `AffineBound.align()`, which scatters
both operands' A matrices into a common wider layout.

### Deriving per-component intervals

```python
# for each component i
lo[i] = b_lo[i] + Σ_j term_lo(A_lo[i, j], x_lo[j], x_hi[j])
hi[i] = b_hi[i] + Σ_j term_hi(A_hi[i, j], x_lo[j], x_hi[j])

# where
term_lo(a, lo, hi) = 0            if a == 0
                     a * lo       if a > 0
                     a * hi       if a < 0

term_hi(a, lo, hi) = 0            if a == 0
                     a * hi       if a > 0
                     a * lo       if a < 0
```

Two sign-split reductions. `AffineBound.to_interval()` returns
a list of per-component `Range` objects.
`AffineBound.to_scalar_range()` returns the hull (union) of all
per-component intervals as a single `Range`.

The explicit `a == 0` branch is not cosmetic: the basis may contain
unbounded variables (`x_lo[j] = -inf` or `x_hi[j] = +inf`) for
inputs without declared ranges, and IEEE `0 * inf = NaN` would
corrupt every downstream bound. The short-circuit keeps zeroed
coefficients out of the sum.

### API surface

The public API is intentionally small:

- **Factory methods.**
  `AffineBound.identity(input_node)` — identity A-matrix with
  columns keyed by `input_node.node_id`; offsets zero; ranges
  from the input's declared range.
  `AffineBound.constant(values)` — zero `A`; `b_lo = b_hi =
  values`; empty column map.
  `AffineBound.degenerate(d_output, lo, hi)` — zero `A`;
  scalar `lo` and `hi` broadcast to `[d_output]`, used as the
  escape hatch when an op cannot compute affine bounds.
- **`align(a, b)`** — reindex two bounds to share the same column
  layout, merging column maps and intersecting input ranges.
- **`to_interval()`** — per-component list of `Range` objects,
  computed via the sign-split formula above.
- **`to_scalar_range()`** — hull of all per-component intervals.
- **`__repr__`** — summary only (`d_output`, `n_cols`, coarse
  interval display).
- **Serialization.** `AffineBound`s are not persisted to the
  compiled artifact — they are compile-time metadata.

Arithmetic (addition, matrix multiply, scalar scale) is not
exposed as operator overloads; the per-op rules in
*Per-op propagation rules* do the required math inline.

---

## Integration with `Node` and `NodeValueType`

`AffineBound` is stored on `Node` directly, not inside
`NodeValueType`. The two are combined at access time:

```python
class Node:
    @property
    def value_type(self) -> NodeValueType:
        r = self._affine_bound.to_scalar_range()
        sr = self._structural_type.value_range
        if sr.lo > r.lo or sr.hi < r.hi:
            lo = max(r.lo, sr.lo)
            hi = min(r.hi, sr.hi)
            if lo > hi:
                scale = max(1.0, abs(sr.lo), abs(sr.hi),
                            abs(r.lo), abs(r.hi))
                if lo - hi > 1e-6 * scale:
                    raise ValueError(...)  # disjoint beyond fp noise
                # fp-noise crossing: collapse onto the structural
                # boundary the affine bound rounded past.
                lo = hi = min(max(hi, sr.lo), sr.hi)
            r = Range(lo, hi)
        return NodeValueType(value_range=r)
```

The intersection carries an fp-noise clamp. The affine bound is
float64 interval arithmetic whose rounding can push a bound a few
ULPs past the structural range — e.g. a value structurally in
`[0, 1]` whose affine bound collapses to the point `1 + 2^-26`.
That makes the two *fp-disjoint* even though both soundly bound the
same scalar, and a plain `Range.intersect` (which treats any
crossing as a bug) would raise. Instead the property clamps the
affine bound into the structural range, guarded by a
magnitude-scaled `1e-6` tolerance so a *gross* disjointness — a
genuinely unsound bound, not rounding — still raises `ValueError`.

- `_affine_bound` is the source of truth for numeric bounds.
  `to_scalar_range()` returns the hull of all per-component
  intervals.
- `_structural_type` is the scalar-interval type from the node's
  `compute_value_type()` rule, with any attached range claim
  (`claimed_type`) intersected in via `tightened_with`. When it is
  tighter than the affine hull on either side, `value_type`
  intersects the two.
- `NodeValueType` itself carries only `value_range`; its only
  constructors are `unknown()` and `bounded(lo, hi)`. It does not
  hold an `AffineBound`.
- Every existing consumer that reads `node.value_type.value_range`
  keeps working; the returned `Range` is strictly inside (or equal
  to) the old scalar-interval-based one.
- Structural properties (integer-ness, binary-ness, one-hot-ness)
  are not part of `NodeValueType`. They live as attached-check
  metadata on the node: the helpers in `graph/asserts.py`
  (`assert_integer`, `assert_bool`, `assert_01`, `assert_onehot`)
  append runtime predicates to `node.checks` and intersect a range
  claim into `node.claimed_type`. `assert_integer` additionally
  sets `Node.integer_claim`, the boolean the univariate collapse
  pass gates on. The range claim is the only part the static bound
  system consumes — the semantic content (exactly-integer,
  exactly-one-hot) is enforced by the runtime predicates, which run
  on every `compute` and against compiled values on `debug=True`
  forwards.
- The `Guarantee` enum is removed. Its transition-zone
  (`APPROXIMATE`) semantics went with it; each predicate helper
  carries an explicit `atol` parameter where the old level implied
  a tolerance.

Nodes whose rules can't (or won't) compute affine bounds carry a
degenerate `AffineBound` — zero `A_lo` and `A_hi`, and `b_lo` /
`b_hi` set to whatever scalar interval the rule decides on (or
±∞). This preserves the "unknown range" escape hatch and keeps
every op total.

### Removal of the `Guarantee` machinery

The `Guarantee` enum removal is a complete deletion, not a shim. In
the finished system:

- **`Guarantee`, `GuaranteeLevel`, `_min_guarantee`, `_max_guarantee`
  are gone.** Nothing in the codebase imports or mentions them.
- **The structural-flag factories went with them.** `NodeValueType`
  has exactly two constructors — `unknown()` and `bounded(lo, hi)`
  — plus the `with_range` combinator. There is no `integer(...)` /
  `binary(...)` / `one_hot(...)` factory because there are no flags
  to construct: the type is range-only.
- **`tightened_with`** combines two claims by intersecting their
  `value_range`s — nothing else. It is called from
  `refresh_node_caches` to fold a node's `claimed_type` into its
  structural type, and from `attach_assert` to intersect a new
  claim into the node's running claim (claims commute, so attach
  order is irrelevant).
- **Claims with structural content became runtime predicates.**
  `assert_integer` / `assert_bool` / `assert_01` / `assert_onehot`
  (`graph/asserts.py`) attach predicates that run on every
  `compute` and against compiled values on `debug=True` forwards;
  each takes an explicit `atol` where the old `APPROXIMATE` level
  implied a tolerance. Their static footprint is a range claim
  (`claimed_type`) and, for `assert_integer`, the
  `Node.integer_claim` boolean.
- **`assert_matches_value_type`** survives as one of those helpers:
  it checks the observed tensor against a `NodeValueType`'s
  `value_range` within `atol`, and promotes that type to the node's
  range claim so downstream static analysis sees it.

### How `assert_in_range` interacts with bounds

An assertion is node *metadata*, not a wrapper node: there is no
`Assert` node class. `assert_in_range` appends a runtime predicate
to `node.checks`, intersects `NodeValueType.bounded(lo, hi)` into
`node.claimed_type`, and returns the same node. The claim reaches
the bound system through `refresh_node_caches` — the single choke
point every bounds recompute goes through — which applies it on two
channels (`_apply_claim_to_bound` in `graph/affine_rules.py`)
depending on what carries the claim:

- **Claims on a source node** (`InputNode` or `Embedding`) tighten
  that node's own `input_ranges` entry in its `AffineBound`. No
  coefficients change; every downstream `to_interval()` that
  evaluates against the basis box automatically sees the tighter
  ranges.

- **Finite claims on a non-source node** use **degenerate
  collapse**: the node's `AffineBound` is replaced with zero A
  matrices and per-component intervals from the intersection of
  the propagated affine evaluation with the claimed range.
  Upstream coefficients are discarded. The appendix (*Why Assert on
  non-InputNode uses degenerate collapse*) documents the four
  coefficient-preservation alternatives that were explored and why
  each was rejected.

A non-finite claim on a non-source node leaves the affine bound
alone; it still tightens `_structural_type`, so `value_type`
intersects it. A semantic override (see *Composite ops and semantic
overrides*) wins over the claim on the affine channel — ops install
overrides after attaching their claims, and the override's
correlation structure is worth more downstream than the claim's
constant box; the claim still tightens `_structural_type`, so
`value_type` stays claim-tight.

The runtime predicate runs on every `compute` and catches cases
where the actual tensor exceeds the claimed range.

---

## Bound computation lifecycle

Affine bounds are computed **eagerly** during graph construction.
`Node.__init__` ends by calling `refresh_node_caches(self)`
(`graph/affine_rules.py`) — the same choke point every later bounds
recompute goes through, so claim application can never be skipped.
The refresh runs, in order:

```python
node._structural_type = node.compute_value_type()
# ...claimed_type folded in via tightened_with, if attached...
node._affine_bound = compute_affine_bound(node)
# ...then the semantic override is re-applied, or else the claim...
```

`compute_value_type()` is each op's scalar-interval rule; it
returns a range-only `NodeValueType`. Most primitives return
`unknown()` and let the affine channel carry the tight range —
`LiteralValue` is the notable exception, reporting its exact
min/max. `compute_affine_bound()` dispatches to the appropriate
per-op rule (see *Per-op propagation rules*) and returns the node's
`AffineBound`. The two channels meet at read time in the
`value_type` property.

Because each `AffineBound` carries its own column map and input
ranges (see *The basis*), there is no need for a global basis or
finalization pass — each rule reads its inputs' already-computed
`_affine_bound` attributes, which are available because inputs are
constructed before the nodes that consume them.

Bound-consumer ops (`cond_gate`, `attend_mean_where`) read their
inputs' `value_type.value_range` — which is available eagerly — and
build their full subgraphs immediately at construction time.
(`floor_int` builds its subgraph eagerly too, but its breakpoint
range comes from explicit `min_value` / `max_value` arguments, not
from bounds; `assert_in_range` reads no bounds and builds no
subgraph — it attaches metadata to an existing node, and its claim
is applied to that node's bound via `refresh_node_caches`.)

There is no separate `finalize()` pass, no placeholder nodes, and
no deferred materialization. An earlier design attempted a
two-phase model (construction, then finalization) but it was
found to cause silent M-value explosions; see the appendix
(*Why eager bounds replaced the finalize pass*) for details.

---

## Per-op propagation rules

Each rule below consumes its inputs' `AffineBound`s and produces
this node's `AffineBound`. Column maps are merged as needed via
`AffineBound.align()`.

### `Linear` — `y = W · v + c`

Exact. Sign-split `W`:

```
W_plus  = clamp(W, min=0)
W_minus = clamp(W, max=0)

A_lo(y) = W_plus · A_lo(v) + W_minus · A_hi(v)
b_lo(y) = W_plus · b_lo(v) + W_minus · b_hi(v) + c

A_hi(y) = W_plus · A_hi(v) + W_minus · A_lo(v)
b_hi(y) = W_plus · b_hi(v) + W_minus · b_lo(v) + c
```

Four GEMMs (two per bound), `[d_out × d_in] · [d_in × n]`. This is
the dominant compile-time cost of the bound-propagation pass
itself; it is separate from the compiled transformer's runtime
forward-pass compute.

The per-component min/max aggregation that the old
`linear_output_range` helper performed is no longer needed and
has been removed — per-component tightness is already present in
the input's `A`.

### `Add` — `z = u + v`

Preserves affine structure exactly. The two operands' column maps
are first merged via `AffineBound.align()`, then their A matrices
and offsets are added element-wise:

```
u, v = AffineBound.align(u, v)
A_lo(z) = A_lo(u) + A_lo(v)
b_lo(z) = b_lo(u) + b_lo(v)

A_hi(z) = A_hi(u) + A_hi(v)
b_hi(z) = b_hi(u) + b_hi(v)
```

No sign split — addition preserves monotonicity in both
directions. This is the rule that makes correlation cancellation
automatic. `cond_gate`'s `add M → ... → sub M` pattern cancels
exactly: the `M` constants have zero `A` coefficients and their
`b` offsets are equal-and-opposite, so the sum's `b_lo` / `b_hi`
contributions from the two `M`s cancel cleanly.

Two important caveats on "exactly":

1. *Degenerate operands do not cancel against each other.* If
   either operand has a degenerate `AffineBound` (zero `A`
   matrices — e.g. the output of a semantic override on
   `select` or `compare`), `Add` correctly sums their intervals
   but no coefficient cancellation is possible because there are
   no coefficients.

2. *Straddling `ReLU` breaks exact cancellation through the ReLU.*
   When a node passes through `ReLU` in the straddling case, its
   output carries the linear-envelope coefficients (chord on the
   upper side, `α · v` on the lower), not the original function's
   coefficients. Two operands whose *true* values cancel exactly
   can re-emerge from `ReLU` with envelopes that differ and no
   longer cancel. Cancellation via `Add` is exact only when both
   operands reach `Add` via paths through exact rules (`Linear`,
   `Add`, `Concatenate`, non-straddling `ReLU`); a straddling
   `ReLU` on the path inserts envelope gaps that survive the sum.
   In practice the scheduler does not construct patterns that
   depend on cancellation through a straddling `ReLU`.

### `ReLU` — element-wise

Element-wise case analysis on each component's derived interval
`[l[i], h[i]]`. Three cases, using non-strict inequalities on the
boundary so `l[i] == 0` or `h[i] == 0` exactly are deterministic:

- `l[i] >= 0` (component always non-negative, including `l[i] == 0`):
  identity. Copy the input's row of `A_lo`, `A_hi`, `b_lo`, `b_hi`
  unchanged.
- `h[i] <= 0` (component always non-positive, including
  `h[i] == 0`): zero the row — `A_lo[i, :] = A_hi[i, :] = 0`,
  `b_lo[i] = b_hi[i] = 0`.
- `l[i] < 0 < h[i]` (strict straddling): apply linear envelope.

The boundary cases (`l[i] == 0` alone, `h[i] == 0` alone) are
handled by the first two branches above; the all-zero interval
`l[i] == h[i] == 0` takes the first branch (the `l[i] >= 0` check
runs first) and copies the input's rows — sound because `ReLU` is
the identity on `{0}`. This matters because `slope = h / (h - l)`
in the straddling branch is only defined when `h - l > 0`, which
strict straddling guarantees.

Straddling components:

```
# Upper bound: chord from (l, 0) to (h, h).
slope = h[i] / (h[i] - l[i])
A_hi(z)[i, :] = slope * A_hi(v)[i, :]
b_hi(z)[i]    = slope * (b_hi(v)[i] - l[i])

# Lower bound: α · v, with continuous chord α = h / (h - l).
alpha = h[i] / (h[i] - l[i])
A_lo(z)[i, :] = alpha * A_lo(v)[i, :]
b_lo(z)[i]    = alpha * b_lo(v)[i]
```

The continuous chord `α = h / (h - l)` is strictly tighter than
the simpler `α ∈ {0, 1}` heuristic. It tracks correlation through
straddling ReLU: for example, `ReLU(x) + (-ReLU(x))` with
`x ∈ [-2, 4]` gives `[-4/3, 4/3]` instead of `[-4, 4]`.

Soundness sketch for the lower bound. Pointwise, `ReLU(v) ≥ α · v`
holds for every `v` and every `α ∈ [0, 1]` (at `v ≥ 0`,
`ReLU(v) = v ≥ α · v`; at `v < 0`, `ReLU(v) = 0 ≥ α · v`).
Since `h / (h - l) ∈ (0, 1)` for strictly straddling intervals,
this is always sound. Substituting `v`'s lower affine envelope
`v_lower(x)` and using `α ≥ 0`:
`ReLU(v(x)) ≥ α · v(x) ≥ α · v_lower(x)`. When
`α · v_lower(x)` happens to be negative on part of the box, this
is trivially true (`ReLU` outputs are non-negative); the bound is
loose there but still sound.

When either endpoint is infinite (unbounded straddling), the
chord computation `h / (h - l)` would produce `inf/inf = NaN`.
A guard falls back to degenerate `[0, h]` (or `[0, +inf]`) for
those components.

`ReLU` and the `FFN` rule's activation/product relaxations (next
section) are the only primitive rules that introduce looseness.
Every other primitive rule is exact; any slack in a downstream
interval traces back to one of them. (Composite ops introduce
additional looseness via semantic overrides; see *Composite ops
and semantic overrides*.)

A tighter choice — learning α per component via gradient descent
against a specific output bound — is known in the bound-propagation
literature as α-CROWN. It is *not* a drop-in swap of this rule:
α-CROWN is formulated around a specific query node (the thing
whose bound you're trying to tighten) and requires
backward-propagation of gradients from that query through every
intermediate `AffineBound`. Adopting it would mean adding autograd
through the bound computation and a per-query optimization pass —
a meaningful architectural addition, not a local rule upgrade. If
the continuous-chord bounds prove too loose in practice, that work
is well-understood in the literature; it just isn't small.

### `FFN` — gate projection → activation → (× up projection) → output projection

One node, composed from the rules above (`_ffn_rule`). The
degenerate-ReLU form (`up_proj is None`, `activation="relu"`) is
exactly the three rules the equivalent `Linear → ReLU → Linear`
three-node chain applied: sign-split GEMM through `gate_proj`,
the ReLU envelope per lane, sign-split GEMM through `out_proj`.

**Swish activation.** Swish is sandwiched by ReLU globally:

```
relu(z) - C <= swish(z) <= relu(z),   C = 0.278465
```

(`C` is the magnitude of swish's global minimum at
`z ≈ -1.2785`, rounded up; for `z > 0` the gap
`z - swish(z) = z·σ(-z)` has the same maximum by symmetry.) So the
swish envelope is the ReLU envelope with the lower offsets shifted
down by `C` (`_swish_affine`). The slack is the *constant* `C` per
lane — it does not grow with value magnitude, and for
`Swish(scale·cond)/scale` gates the fold-through divides it by
`scale`.

**Gated lanes.** A gated lane `w = act(gate(x)) · up(x)` is a
product of two affine-bounded quantities, which is not affine.
Given per-lane interval endpoints `s ∈ [sl, sh]`, `u ∈ [ul, uh]`
that hold pointwise, the four standard linear product relaxations
(the McCormick inequalities) are affine in `(s, u)`:

```
w >= ul·s + sl·u - sl·ul      w <= ul·s + sh·u - sh·ul
w >= uh·s + sh·u - sh·uh      w <= uh·s + sl·u - sl·uh
```

Substituting each factor's affine bound (routing every coefficient
to the factor's lower or upper expression by sign, as in the
sign-split GEMM) keeps the lane bound affine in the graph inputs
(`_gated_lane_affine`). Per side, both candidates are built and
the tighter one by concretized interval is kept per lane.

Two details that matter for tightness and soundness:

- The activation's interval endpoints `[sl, sh]` come from the
  **exact 1-D activation range over the gate's concretized
  interval** (`_swish_interval`: endpoints plus the interior
  minimum when the interval contains the argmin) — *not* from
  concretizing the activation envelope, whose lower line
  evaluated at a wide gate interval is far below the true
  minimum and would poison the corner coefficients.
- A lane with a non-finite factor interval falls back to an
  unbounded row (zero coefficients, `±inf` offsets) — sound, and
  it degenerates only that lane.

Measured on the multiply / select / cond_gate constructions
(regression-pinned in `tests/graph/test_affine_bounds.py`): the
±-pair `multiply` FFN over `[-10, 10]²` concretizes to within
1.06× of the true `[-100, 100]`; `select` and `cond_gate` land
within 1.10× and 1.05×. The remaining slack is the cross-lane
correlation (the `σ(g) + σ(-g) = 1` cancellation) that no
per-lane relaxation can see; semantic overrides remain the tool
when an op needs the exact range.

### `Concatenate` — row-stacking

Exact. The inputs' column maps are first merged via
`_merge_layouts`, then each input's A matrices are scattered into
the common layout. The result stacks the scattered A matrices
row-wise and their offset vectors element-wise:

```
merged_columns, merged_ranges, n = _merge_layouts(*inputs)
A_lo(z) = vstack([scatter(A_lo(inp)) for inp in inputs])
b_lo(z) = concat([b_lo(inp) for inp in inputs])
# same for A_hi, b_hi
```

Downstream `Linear`s read the stacked `A` directly, which is
where per-component tightness comes from.

### `LiteralValue`

```
A_lo(z) = A_hi(z) = zeros([d_output, 0])
b_lo(z) = b_hi(z) = value
columns = {}; input_ranges = {}
```

A constant has no dependence on any source node.

### `InputNode`

Identity A-matrix with columns keyed by the `InputNode`'s own
`node_id`:

```
A_lo(z) = A_hi(z) = eye(d_output)
b_lo(z) = b_hi(z) = 0
columns = {node_id: (0, d_output)}
input_ranges = {node_id: (declared_lo, declared_hi)}
```

### Range claims (`assert_in_range` and friends)

There is no `Assert` node and no claim branch in the dispatch — a
range claim is metadata on the claimed node itself
(`node.claimed_type`), applied by `_apply_claim_to_bound` after the
node's own rule runs. The two channels are described under *How
`assert_in_range` interacts with bounds*: a claim on a source node
(`InputNode` or `Embedding`) tightens that leaf's `input_ranges`
entry with coefficients untouched; a finite claim on any other node
collapses the bound to the claim-intersected constant box (zero A
matrices); a non-finite claim on a non-source node leaves the
affine bound unchanged.

### `Embedding`

An `Embedding` node is an integer-indexed lookup into a constant
table whose values are known at construction time. It is a
**basis source**: its `AffineBound` has an identity A-matrix
keyed by the `Embedding`'s own `node_id`, with per-component
input ranges from the table's column-wise min and max:

```
A_lo = A_hi = eye(d_output)
b_lo = b_hi = zeros(d_output)
columns = {node_id: (0, d_output)}
input_ranges = {node_id: (col_min, col_max)}
```

where `col_min[k]` and `col_max[k]` are the min and max of
column `k` over all rows of the embedding table. This is
structurally identical to an `InputNode`'s identity bound, just
with ranges derived from the table rather than from a user
declaration.

### `Placeholder`

`Placeholder` is a zero-width sentinel used where a real node is
not yet available. Its `AffineBound` is degenerate with shape
`[0, 0]` — technically valid, contributes nothing to downstream
consumers.

### `ValueLogger`

`ValueLogger` is a pass-through wrapper used for debugging. Its
affine rule returns the wrapped input's `AffineBound` itself — same
coefficients, same offsets — and its `compute_value_type()` returns
the wrapped input's `value_type`. It is part of the compilable
vocabulary (`VOCABULARY` in `compiler/lower.py`): the lowering
boundary clones it rather than rejecting it, so it may appear in
graphs that are compiled.

---

### Composite ops and semantic overrides

Composite ops are Python functions, not `Node` subclasses. They
split into two groups depending on whether they read their inputs'
bounds.

**Pure subgraph constructors** — `sum_nodes`, `thermometer_*`, and
similar ops that never read `.value_type` — build their subgraphs
immediately from primitive nodes and return one of those
primitives. They need no affine rule; their return node is a
`Linear` / `Add` / `ReLU` / `Concatenate`, each of which has a
rule.

**Bound-consumer ops** — `cond_gate`, `attend_mean_where`, and any
other op that reads its input's `value_type.value_range` to pick a
constant — read bounds eagerly and build their full subgraphs at
construction time. Because bounds are computed eagerly in
`Node.__init__`, the input's `value_type` is already available when
the consumer op runs. (`floor_int` belongs to neither group: it
builds its subgraph eagerly but takes its breakpoint range as
explicit `min_value` / `max_value` arguments and reads no bounds.
`assert_in_range` reads no bounds and builds no subgraph — it
attaches metadata to an existing node.)

Four composite ops apply a **semantic override** after building
their subgraph. The override replaces the propagated
`AffineBound` on the output node with a bound that captures the
op's mathematical output semantics rather than the internal MLP's
interval arithmetic:

- **`cond_gate`**: per-component `[min(0, inp), max(0, inp)]`
  envelope, with non-zero coefficients. When the input's interval
  is entirely non-negative, the upper bound keeps the input's
  `A_hi` / `b_hi` row and the lower bound is the constant 0
  (mirrored for entirely non-positive components). In the
  straddling case, it uses a linear envelope: slope `h / (h - l)`
  on the upper side, slope `-l / (h - l)` with offset
  `l·h / (h - l)` on the lower side.
- **`select`**: per-component hull of both branches' intervals.
  Zero A matrices.
- **`compare`**: constant bound (`true_level` or `false_level`)
  when the input interval is fully above or below the threshold;
  degenerate interval `[min(levels), max(levels)]` otherwise.
- **`broadcast_select`**: per-component hull of the corresponding
  true/false channels, computed per output slot (a broadcast
  branch's single-slot interval applies to every slot). Zero A
  matrices.

These overrides are a source of looseness in the system: the
`select`, `compare`, and `broadcast_select` overrides kill
upstream coefficient tracking through the gate entirely, and
`cond_gate`'s — though it keeps the input's coefficients —
replaces the true gate output with its envelope. This is
deliberate: the alternative (propagating coefficients through the
gate's internal `linear_relu_linear` subgraph) produces correct
but extremely wide bounds because the cancellation offset `M`
appears in the coefficients.

---

## `Attn`

Torchwright's `Attn` computes
`softmax(rope(Q) · rope(K)^T, causal-masked) · V · output_matrix`,
where `Q = query_in · query_matrix`, `K = key_in · key_matrix`, and
`V = value_in · value_matrix` — each projection is a `Linear`-shape
transform on one of the three input nodes, and `rope(·)` rotates Q
and K by absolute position on the `rotate_half` grid
(`torchwright/graph/rope.py`) before the dot product (see
`torchwright/graph/attn.py`). V is never rotated, so the rotation
lives entirely on the Q/K path — which the rule below does not
model anyway — and the value-path soundness argument is unaffected.

Sound linear relaxations for softmax and for the bilinear `Q·K^T`
do exist in the bound-propagation literature (Shi et al., ICLR
2020, "Robustness Verification for Transformers," and subsequent
work). We do not adopt them here. The reasons are design-local: the
published relaxations introduce coupling between token positions
that a single-position basis cannot represent, and the compile-time
benefit at torchwright's specific bound-consumer sites is
unmeasured.

The Attn rule propagates value-input coefficients through V and O,
but does not model Q/K logits. Softmax produces convex-combination
weights (non-negative, summing to 1), so per-component affine
bounds on `value @ V` carry through unchanged — a convex
combination of values that each satisfy the same affine bound
also satisfies that bound. This preserves the value input's
relationship to the basis, while Q/K correlation is lost.

Concretely, the rule performs two sign-split GEMMs:

1. **Propagate `value_in`'s `AffineBound` through `value_matrix`**
   using the same sign-split rule as `Linear`:

   ```
   V_plus  = clamp(value_matrix, min=0)
   V_minus = clamp(value_matrix, max=0)

   proj_A_lo = V_plus^T · A_lo(value) + V_minus^T · A_hi(value)
   proj_A_hi = V_plus^T · A_hi(value) + V_minus^T · A_lo(value)
   proj_b_lo = V_plus^T · b_lo(value) + V_minus^T · b_hi(value)
   proj_b_hi = V_plus^T · b_hi(value) + V_minus^T · b_lo(value)
   ```

2. **Propagate through `output_matrix`** with another sign-split:

   ```
   O_plus  = clamp(output_matrix, min=0)
   O_minus = clamp(output_matrix, max=0)

   A_lo = O_plus^T · proj_A_lo + O_minus^T · proj_A_hi
   A_hi = O_plus^T · proj_A_hi + O_minus^T · proj_A_lo
   b_lo = O_plus^T · proj_b_lo + O_minus^T · proj_b_hi
   b_hi = O_plus^T · proj_b_hi + O_minus^T · proj_b_lo
   ```

The returned `AffineBound` carries non-zero coefficients — the
value input's columns and input_ranges pass through. Downstream
ops that skip-connect around the attention sublayer retain
cancellation with the Attn output's basis variables. Query and
key matrices never appear in the bound — the convex-combination
relaxation makes them irrelevant to the output's relationship to
the basis.

### What is lost

Cancellation that depends on Q/K correlation — i.e. knowing
*which* softmax weights are large — cannot be recovered, because
the rule assumes any probability vector is possible. In practice
the scheduler does not build graphs that rely on Q/K-dependent
cancellation; when tight post-attention bounds are needed, an
explicit `assert_in_range` on the post-attention quantity gives
downstream `Add` something tight to cancel against.

### Related prior work

Forward-mode linear-relaxation bounds for transformer verification
are developed in Shi, Zhang, Chang, Huang, and Hsieh, "Robustness
Verification for Transformers" (ICLR 2020), which includes
relaxations for softmax and the bilinear `Q·K^T`. Bonaert et al.,
"Fast and Precise Certification of Transformers" (PLDI 2021), and
subsequent work refine the softmax rules. The auto_LiRPA library
(Xu et al., NeurIPS 2020) provides both forward- and backward-mode
LiRPA/CROWN implementations with these relaxations. This design
deliberately stops short of adopting those rules — see the
rationale at the top of the `Attn` section — but the literature is
the natural reference if we decide to push further on attention
tightness later.

---

## Effect on existing bound consumers

Every op that today reads `node.value_type.value_range` keeps
reading it. The aggregate returned by the derived property is
strictly inside (or equal to) the old one, so the only observable
change at these call sites is tighter constants:

| Consumer | Where it reads bounds | Effect of tighter bounds |
|---|---|---|
| `cond_gate` | `_max_abs_or_raise(input)` to pick `M` | Smaller `M` offset; less precision loss in the gate's MLP |
| `assert_in_range` | declared `[lo, hi]` vs inferred | Runtime predicate unchanged; inferred static claim often meets or exceeds what the user asserted |
| `attend_mean_where` | `value` input's range, re-claimed on the output | Tighter claimed range on the mean output |

Consumers that want per-component precision (instead of the
node-wide aggregate) can call `node.affine_bound.to_interval()`
and index into individual `Range` objects. In most cases they
don't need that granularity, because downstream `Linear`s already
see per-component tightness through the input's `A`.

---

## Runtime verification

`TW_VERIFY_VALUE_TYPES` wraps each `Node.compute` and checks the
observed tensor against the declared `value_type`:

- The check is range-only (`_verify_tensor_against_value_type` in
  `graph/node.py`): `t.min()` and `t.max()` are compared against
  the aggregate `value_type.value_range` with tolerance `1e-4`, and
  any violation raises `AssertionError`. Because `value_range` is
  now derived from the affine bound (via `to_scalar_range()`), the
  aggregate range is tighter than it was under the old
  scalar-interval system, so the check catches more violations than
  before even though it operates on the aggregate.
- Structural properties are verified by the attached-check
  predicates (`node.checks`), which run on every `compute`
  independently of `TW_VERIFY_VALUE_TYPES` (suppressible via
  `suppress_checks()`), and against compiled values on `debug=True`
  forwards.

---

## Deliberately out of scope

- **Full softmax / bilinear relaxation for attention.** Published
  linear relaxations for softmax and `Q·K^T` exist but are not
  adopted here (see the `Attn` section for the rationale). The
  current rule propagates value-input coefficients through V and
  O but does not model Q/K correlation.
- **Cancellation through the Q/K path of attention.** The Attn
  rule propagates coefficients from the value input only. Any
  cancellation that requires tracking which Q/K logits produced
  the softmax weights is not recovered.
- **Per-component learnable α on `ReLU` (α-CROWN).** The current
  continuous-chord `α = h / (h - l)` is tighter than a discrete
  `{0, 1}` choice but still fixed per component.  Per-component
  learnable α (α-CROWN) is a real tightness upgrade, but adopting
  it requires backward-mode infrastructure (autograd through
  bounds, per-query optimization) — not a local change to the
  `ReLU` rule alone. See the `ReLU` section for details.
- **Piecewise-affine joint reasoning.** Each component's bound is
  a single pair of affine expressions, not a union over cases.

---

## Data flow summary

```
Source nodes ──► Linear ─► Add ─► ReLU ─► Linear ─► Add ─► ... ─► Attn ─► Linear ─► ...
            │                                                          │
            │   all AffineBounds with self-keyed column maps           │
            │   (InputNode / Embedding components)                     │
            │                                                          │
            │ exact propagation for Linear/Add/Concatenate             │
            │ linear-envelope relaxation for straddling ReLU           │
            │ value-coefficient propagation through Attn (V → O)      │
```

Every bit of bound tightness comes from exact affine algebra on
`Linear` / `Add` / `Concatenate`, and from `Attn`'s
value-coefficient propagation through V and O.

Three sources of looseness:

1. **Straddling-`ReLU` envelopes.** The only primitive rule that
   introduces gap between the affine bounds and the true function.
2. **Semantic overrides on composite ops.** `cond_gate`, `select`,
   `compare`, and `broadcast_select` replace the propagated
   `AffineBound` with a semantic bound that captures the op's
   mathematical output range. For `select`, `compare`, and
   `broadcast_select` the semantic bound has zero coefficients,
   which kills upstream coefficient tracking through the gate;
   `cond_gate`'s keeps the input's coefficients (copied on
   sign-definite components, slope-scaled when straddling), and
   its looseness is the envelope gap.
3. **Claim degenerate collapse.** A finite range claim
   (`assert_in_range`) on a non-source node collapses that node's
   bound to zero coefficients (see *Range claims* and the appendix
   for why).

---

## Appendix: development history

This appendix records alternatives that were explored during
development and why they were rejected. The main document above
describes the system as built; this appendix preserves the
reasoning behind key design choices for future reference.

---

### Why eager bounds replaced the finalize pass

The original design called for a `finalize(root)` pass that
built a global `Basis` from all `InputNode`s, then propagated
bounds in topological order. Consumer ops (`cond_gate`,
`floor_int`, etc.) would return placeholder nodes during
construction and materialize their subgraphs during finalization.

This was implemented and found to be actively harmful: placeholder
nodes' `compute_value_type()` returned pessimistic types that
downstream nodes cached during construction and never recomputed
after finalization. The result was silent M-value explosions
(M = 3.3e21 observed in an early balanced-parentheses test graph).

The replacement — eager bounds computed in `Node.__init__` with a
self-keyed column map per `AffineBound` — eliminates the global
`Basis`, the `finalize` pass, `ConsumerPlaceholder`, and all their
helpers. `cond_gate` and `select` build their subgraphs eagerly.

---

### Why Attn propagates value coefficients

The original design had `Attn` emit a zero-coefficient (degenerate)
`AffineBound`. The implementation propagates the value input's
affine bound through V and O matrices via sign-split, producing
non-zero coefficients. This is tighter: downstream ops that
skip-connect around the attention sublayer retain cancellation
with the Attn output's basis variables. Soundness holds because
softmax produces convex-combination weights, so per-component
affine bounds on `value @ V` carry through.

---

### Why continuous chord replaced α ∈ {0, 1}

The original design specified `α = 1 if h >= -l else 0` for the
straddling ReLU lower bound. The implementation uses
`α = h / (h - l)` for both upper and lower envelopes. This
is strictly tighter: the continuous chord tracks correlation
through straddling ReLU. For example, `ReLU(x) + (-ReLU(x))`
with `x ∈ [-2, 4]` gives `[-4/3, 4/3]` instead of `[-4, 4]`.

When either endpoint is infinite (unbounded straddling), the
chord computation would produce `inf/inf = NaN`. A guard falls
back to degenerate `[0, h]` (or `[0, +inf]`) for those
components.

---

### Why Assert on non-InputNode uses degenerate collapse

Four approaches to preserving the wrapped node's affine
coefficients through an Assert boundary were explored and rejected:

**1. Full passthrough with `claimed_range` clipping.** The
Assert's `AffineBound` keeps the input's A/b coefficients and
adds a `claimed_range` field that `to_interval()` clips against.
Problem: downstream `Linear` nodes multiply the A matrices by
their weight matrices, producing new `AffineBound`s without
`claimed_range`. The unclamped coefficients, evaluated against
the full input basis box, produce ranges in the millions. In the
3-digit adder, this caused M-offset values of 1.1e7 (sanity bound
is 1e6).

**2. Assert as a new basis variable.** The Assert creates a fresh
identity A-matrix keyed by the Assert's own `node_id`, with the
claimed range as the basis variable's range. This gives tight
ranges for one downstream Linear (identity × claimed range =
claimed range). But when the Assert basis variable merges with
original `InputNode` basis variables through `Add`/`Concatenate`
and feeds into a second Linear, cross-terms between the two
basis families widen the result. In the DOOM graph (a large
downstream game-rendering graph used for scale testing),
worst-case bound width went from 42M to 189M, and gate M from
43K to 75K, regressing its end-to-end tests.

**3. Back-projection of claim onto input_ranges.** Given
`A · x + b ∈ [C_lo, C_hi]`, derive tighter ranges on each basis
variable `x_j`. For each variable, the tightest bound assumes all
other variables take their most favorable values (minimizing their
contribution for upper-bound constraints, maximizing for
lower-bound constraints — NOT worst-case, which over-constrains).
This is mathematically sound but was shown to be unnecessary:
forward IBP propagation of the claim through downstream weight
matrices is always at least as tight as affine evaluation against
the back-projected box, because the back-projected box is a
conservative (axis-aligned) approximation of the true feasible
polytope.

**4. IBP fields inside AffineBound.** Add per-component interval
bounds (`ibp_lo`, `ibp_hi`) that propagate via interval arithmetic
through each rule alongside the A/b coefficients. `to_interval()`
returns the tighter of (coefficient evaluation, IBP) per component.
This is the most principled approach — two complementary systems
in one object. Problem: it fails for the same reason as approach 1,
just less severely. The degenerate collapse kills upstream columns;
with 2 basis columns through a 434-unit ReLU hidden layer, the
coefficient evaluation stays tight (two terms per component). With
coefficient passthrough, the Assert preserves 3 additional columns
from the Attn upstream, each tracking InputNode components with
range ±10000. Through the same ReLU layer, the coefficient
evaluation explodes to 366K. IBP recovers partially (to 16K via
`clamp(ibp, min=0)` through ReLU then interval arithmetic through
the output matrix), but the old 2-column bound gave 5.7K. In the
DOOM graph, 83 of 1722 nodes regressed (worst case: 3x wider).

**Why degenerate collapse is the right choice.** The column-count
effect is structural, not accidental. At an Assert boundary, the
upstream chain has mapped many source-node components through
many weights into a value the Assert claims is in a narrow range.
Downstream ops don't need to track which combination of upstream
inputs produced that value — they need to know it's in the
narrow range. Degenerate collapse expresses this: "the upstream
relationship is consumed; downstream, this is a fresh bounded
value." Preserving the upstream coefficients forces downstream
ReLU layers to evaluate those coefficients against the full
upstream input ranges, accumulating width through every hidden
unit.

No current graph pattern exercises cancellation through Assert
boundaries. Gate outputs (cond_gate, select, compare) have
semantic overrides that replace the Assert's bound. Digit
extractions, score assertions, and similar claims are consumed by
a single downstream path with no fan-out-and-recombine. If a
future pattern requires cancellation through Assert, the design
space is well-understood; the constraint is finding a mechanism
that preserves coefficients without adding columns that widen
downstream ReLU evaluations.

---

### NaN guards for unbounded inputs

Three guard sites prevent `0 × ±inf = NaN` and `inf / inf = NaN`
when source nodes have unbounded declared ranges:

1. `_eval_lower` / `_eval_upper`: separate `pos == 0` and
   `neg == 0` guards instead of a single `a == 0` check, because
   `clamp(a, min=0) * x_lo` can produce `0 × (-inf) = NaN` even
   when `a > 0` (the negative clamp is zero but IEEE computes both
   branches of `pos * x_lo + neg * x_hi` before the where-select).
2. `_safe_matvec(W, b)`: treats `0 × ±inf → 0` in bias
   propagation through Linear and Attn rules.
3. ReLU and cond_gate straddling: when either endpoint is
   infinite, falls back to degenerate bounds instead of computing
   `h / (h - l)`.
