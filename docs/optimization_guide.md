# Optimization Guide

A conceptual guide to reducing layer count and parameter cost when
compiling a torchwright graph into a transformer. The scheduler and
compiler are small and readable (`torchwright/compiler/forward/`);
read the source when a detail matters.

This guide is deliberately light on specific numbers — the graph and
compiler evolve, and quantitative snapshots go stale. Several sections
refer to a **graph-stats diagnostic**: a per-annotation stats pass
over the compiled graph — node and param counts per annotation, the
compiled layer count, and the longest critical-path chains — which
you can replicate from `compile_headless(..., verbose=True)` output
plus a walk over the graph's annotations. The DOOM-anchored worked
examples in this doc are illustrative — they preserve the original
numbers that informed each rule of thumb.

---

## 1. Mental model: three cost tiers

Compiling a graph produces a stack of transformer layers. Every layer
has the same capacity regardless of what's scheduled into it:

    layer_capacity = 4 · d · d             (attention Q/K/V/O)
                   + 2 · d · d_hidden      (MLP linear1 + linear2)
                   + d_hidden + d          (MLP biases)

That cost is paid whether the layer is full or empty.

Three quantities describe the workload:

| Quantity | Definition | What to optimize for |
|---|---|---|
| **Graph params** | Non-zero entries in the graph's weight matrices. Per node: `d_in × d_out + d_out` for Linear, QKVO for Attn. | The irreducible information content. Usually small. |
| **Allocated params** | Heads × `4·d·d_head` + MLP slots × `2d + 2`. What the compiler reserves. | The actual cost — what you're paying. |
| **Total capacity** | `n_layers × layer_capacity`. | Dominated by `n_layers` once you've fixed `d`. |

The two highest-leverage optimizations, in order:

1. **Reduce layer count** — each layer is a substantial fixed cost
   shared across every token position.
2. **Reduce distinct Linear/Attn nodes** in hot annotations — each
   `Attn` consumes a whole head-block regardless of how tiny its
   matrix is, and every node is another schedulable unit competing
   for layer capacity.

Density (`graph / allocated`) is a diagnostic for wasted head-width,
not a target. Low density in a big annotation is a compression
opportunity; low density in a small one is noise.

---

## 2. Anatomy of one layer

Each `TransformerLayer` has two sublayers
(`torchwright/compiler/groups/transformer_layer.py`):

    attn_sublayer:  out = attn(x) + x       # n_heads parallel heads
    mlp_sublayer:   out = W2 · ReLU(W1·x) + x

(The MLP activation is ReLU or swish, per the graph's op library;
this guide's cost model uses the ReLU form throughout.)

The scheduler (`scheduler.py:schedule_layer`) processes a layer in
phases:

1. **Attention sublayer** — packs up to `n_heads = d / d_head` heads.
   Candidates: `Attn` nodes, `Add`s (deferred adds and free adds via
   `add_into`), cancellations of dead nodes, and any standalone
   `Linear` the scheduling policy routes here (not the default — see
   §3). All compete for the same head budget.

2. **Attention→MLP handoff** — after the attention sublayer adds its
   result into the residual stream, the MLP sublayer reads
   `x + attn(x)`. So **nodes that became ready because of attention
   outputs can still schedule into the same layer's MLP sublayer**.
   This is load-bearing: a `Attn → linear_relu_linear` pattern fits in
   one layer, not two.

3. **MLP sublayer** — packs FFNs (the `L → ReLU → L` composite,
   which `linear_relu_linear` builds as a single first-class `FFN`
   node), standalone `Linear`s via the MLP bypass, constants, and
   bias writes into `d_hidden` slots. Each slot costs `2d + 2`
   params — **orders of magnitude cheaper per unit of work than an
   attention head**.

Key consequence: **ops in the same sublayer run in parallel with each
other**. Two ops where one reads the other's output cannot share a
sublayer — the second waits for the next layer. The attention→MLP
handoff is the only within-layer sequencing: an op scheduled into a
layer's MLP sublayer can read that layer's attention outputs, never
the other way around.

---

## 3. Per-node cost reference

What each graph node compiles to:

| Graph node | Sublayer | Cost model |
|---|---|---|
| `Attn` | attention | `ceil(d_v / d_head)` heads. |
| Standalone `Linear` | MLP (default) | `2 × d_output` hidden slots via the MLP bypass. Under the eager scheduler, routes to attention (`ceil(d_input / d_head)` heads) only when `2 × d_output` exceeds the layer's usable hidden pool, or under the legacy `local_in_attention="always"` policy; under CP-SAT scheduling the default `cpsat_flex_routing=True` lets the solver pick attention vs MLP per node. |
| `FFN` (`linear_relu_linear`) | MLP | One hidden slot per lane, `2d + 2` params each. A packable unit — many FFNs share one MLP sublayer's hidden pool. |
| `Add` (one addend dead) | attention | 1 head (`add_into`). |
| `Add` (neither dead) | attention | 1 head when `2 × d_out ≤ d_head` (both addends share a combined head); otherwise 2 heads per `d_head`-wide output chunk. |
| `Concatenate` | — | 0. Never allocated; compiler resolves through it. Children still need simultaneous residency. |
| `LiteralValue` | MLP | Bias entries only — effectively free. |
| `InputNode` / `Embedding` | — | 0 cost; sits in residual stream for its lifetime. (Position is not a node at all — it's a rotation applied inside attention, configured by `RopeConfig`.) |

Confirmed at source: `_allocate_head` in
`compiler/forward/weight_writer.py` is a bump-allocator — each
attention-routed op gets its own head-block, no cross-op head
sharing. Routing itself lives in `SchedulingPolicy`
(`compiler/forward/scheduling_policy.py`): `local_in_attention`
defaults to `"never"`, so standalone Linears go to the MLP bypass.

### Op-level shapes

The library ops compose primitives above. Principles:

- **Every `piecewise_linear` / `clamp` / `reciprocal` / `floor_int` /
  `compare` / `select` / `cond_gate` is one MLP-sublayer step** —
  regardless of output width. Their hidden-slot usage scales with the
  number of breakpoints / cases.
- **Every affine Linear (`negate`, `add_const`, `multiply_const`,
  `add_scaled_nodes`) is a standalone Linear** — `2 × d_output` MLP
  hidden slots via the bypass under the default policy.
- **`subtract` is `add(a, negate(b))`** — one standalone Linear plus
  an Add; the Add is free when the negation has no other consumers.
- **`multiply_2d` is one MLP-sublayer step; `multiply_integers` is a
  three-sublayer chain.** `multiply_2d` builds the product as a
  single ReLU bank (O(n) neurons in the breakpoint count) with grid
  precision `step1 · step2 / 4`; `multiply_integers` is exact on
  integer inputs but three MLP sublayers deep. See their docstrings
  for the precision tradeoffs.
- **Attention primitives (`attend_mean_where`, `attend_argmin_*`,
  etc.) are one attention head** when the value fits in `d_head`.

---

## 4. What drives layer depth

### What a critical path is

A **critical path** is a chain of ops in the DAG where each op reads
the previous op's output, traced from an input to an output node. Each
edge in such a chain forces "consumer layer ≥ producer layer + 1," so
the length of the longest chain is a **hard lower bound** on N. No
amount of packing, sharding, or capacity tuning can violate it.

Two things to keep straight:

1. **There may be multiple chains tied at the maximum depth.**
   Shortening one tied chain does not reduce N unless every chain of
   max depth shortens — another chain of equal length still binds the
   lower bound. Before celebrating a DAG-depth win, check that no
   other chain is about to become the binding constraint.

2. **DAG depth is a lower bound, not the compiled depth.** The
   scheduler inflates beyond this bound when per-layer capacity
   (heads/slots) or residual-stream pressure forces ops into separate
   layers. On the DOOM graph the compiled layer count was roughly 2×
   the DAG critical-path depth, so DAG-depth work and packing/capacity
   work are both worth doing — a 1-layer DAG-depth win is a 1-layer
   floor reduction, but actual N only drops if scheduling slack exists
   at that depth.

### Every output imposes the same depth constraint

Overlaid outputs (bit-copied back into the next step's input buffer)
and overflow outputs (read directly by the host, e.g., pixels) are
**identical from the depth-lower-bound perspective**. Both must be
computable by layer N of the current forward pass. A chain of DAG
depth D ending at an overlaid output imposes N ≥ D just as strictly
as a chain ending at an overflow output.

The difference that autoregression introduces is covered in §6 — it's
about splitting a *logical* computation across multiple forward
passes, not about giving any single output slack within a pass.

### Rules of thumb

Rules of thumb for counting layers along a path:

- `Attn` node: **+1 layer** (attention sublayer).
- Standalone `Linear`: **+1 layer** (MLP bypass). Two standalone
  Linears in sequence = 2 layers.
- `L1 → ReLU → L2` chain: **+1 layer** (MLP sublayer).
- `Attn → L1 → ReLU → L2`: **+1 layer** (attn-sublayer + same-layer
  MLP).
- Two sequential L→R→L chains: **+2 layers**.
- `Concatenate`, `Add`, `LiteralValue`, `InputNode`: **+0 layers**.

The graph-stats diagnostic (see intro) reports the actual compiled
layer count and lists the longest contiguous annotation-runs on the
critical path — these are the ops whose depth most directly drives
layer count.

### How to shorten the critical path

1. **Hoist loop-invariant work out of unrolled loops.** Any
   computation whose inputs don't vary across loop iterations should
   be computed once upstream and shared. The per-iteration code then
   collapses to cheap affine Linears — which, after the fusion pass
   (see §8), become free.

2. **Replace nested `select` trees with a table lookup.** A
   depth-`k` select tree is `k` chain layers. When the function is a
   compile-time constant table over two integer indices,
   `table_lookup_2d` (`ops/relu/map_select.py`) computes it as a
   shallow row-select-then-column-gate pipeline instead of `k`
   chained selects.

3. **Avoid expensive multipliers when a coarse grid suffices.** A
   `multiply_2d` on a small breakpoint grid is one MLP-sublayer step
   with grid precision `step1 · step2 / 4`; the exact
   `multiply_integers` chain is three sublayers. Trade precision for
   depth deliberately.

4. **Pack independent chains into one layer.** The scheduler packs
   chains into the MLP sublayer up to `d_hidden` slots. If two chains
   are truly independent and ready simultaneously, they share a layer;
   if one feeds the other, they don't.

5. **Prefer `bool_all_true` over `bool_any_true`** when you already
   hold positive-polarity booleans. `bool_all_true` is a single
   compare; `bool_any_true` is N compares + a sum + a compare.

---

## 5. Attention vs MLP: where should work live?

Per unit of work, the MLP sublayer is **orders of magnitude cheaper**
than the attention sublayer. At typical `d` and `d_head`, one MLP
slot is comparable to thousands of attention-head bytes. So:

**Prefer chain-based expressions (anything built on
`linear_relu_linear`) over standalone Linear nodes whenever you're
doing per-position work.**

### When to use attention

Cross-position communication. This is the only way to move
information between token positions — MLPs operate per-position.

- `attend_mean_where`, `attend_argmin_*`, `attend_argmax_dot` — read
  a value from another position based on content / validity / mask.
- Any KV-cache-backed read in autoregressive generation.

Use attention for what it's uniquely good at (cross-position
content-addressable reads), not for work it's merely capable of
(acting as a 1-to-1 projection).

### When a Linear still lands on attention

Standalone Linears default to the MLP bypass
(`SchedulingPolicy.local_in_attention="never"`), so `negate`,
`add_const`, `multiply_const`, and the small helper Linears ops emit
internally (the base and normalization Linears inside `multiply_2d`,
the sum-collapse `Linear` at the tail of `dynamic_extract`) cost
`2 × d_output` hidden slots, not a head — when the fusion pass (§8)
hasn't folded them away first. Under the eager scheduler, two cases
still route a Linear to an attention head:

- The legacy policy (`local_in_attention="always"`).
- A Linear whose MLP demand (`2 × d_output`) exceeds the layer's
  usable hidden pool — it goes to attention regardless of policy.

Under CP-SAT scheduling the default `cpsat_flex_routing=True` lets
the solver pick attention vs MLP per node; the policy is consulted
only with `cpsat_flex_routing=False`.

Long chains of scalar Linears are still worth hunting: cheap in
params now, but each un-fused link on the critical path is a layer
of depth (§4).

---

## 6. Autoregression: earlier positions precompute for later ones

Multi-phase graphs (e.g. `WALL → EOS → SORTED → RENDER` in DOOM)
exploit the causal KV cache: position `j > i` can attend to `i`'s
values from any prior layer where `i` already held them.

### How autoregression interacts with the critical path

Autoregression reduces N by **splitting a logically long computation
across multiple forward passes**, not by giving overlaid outputs
within-pass slack. The two mechanisms:

- **Overlaid output emitted at step T → input at step T+1.** The
  chain from inputs to the overlaid output must fit in N layers of
  step T. At step T+1, the consumer reads the emitted value as a
  regular input at layer 0 — no DAG depth carries across the step
  boundary. This is how a computation that would be N=200 deep in
  one pass can be split into, say, four passes of N=50 each.

- **Same-pass cross-position attention read.** If position i produces
  a value at layer L and position j > i attends to it within the
  same forward pass, j's attn consumer sits at layer ≥ L+1. The
  chain crosses positions but stays within one pass, so it **does**
  extend the critical path for that pass.

Common confusion worth flushing: an overlaid output does *not* have
"extra slack" relative to an overflow output within a pass. Both must
be computable by layer N. What's special about an overlaid output is
that the *next* step's read of that value starts at layer 0 fresh —
i.e., the chain terminates at the output, it doesn't extend into the
next pass's DAG.

Two consequences for graph design:

### (a) Precompute at an earlier token type

Values needed by many later tokens should be computed at the earlier
token type, packed into a value vector, and read via a single
attention head at the consumers. The downstream stack starts from the
attn output rather than redoing the upstream work.

### (b) Batch cross-position reads

`attend_mean_where` / `attend_argmin_*` can return values up to
`d_head` wide — so 10 scalars bundled into one attention read cost
the same as 1 scalar. If two reads share source positions and
validity/mask, concatenate the values and fuse to one read.

### Constraints

- **Causal mask.** Position `j` can only attend to `i ≤ j`. Token
  ordering is your tool for staging computation.
- **Residual occupancy.** A value produced at WALL layer L and read
  at RENDER layer K occupies residual columns for K−L layers at *every
  WALL-and-later position*. This can be a real cost for wide
  intermediates; narrow what you cache.

---

## 7. Residual stream pressure

Width `d` holds everything "live" (needed by a future consumer). Two
pressure-driven behaviours matter:

1. **Cancellation.** When free columns drop below a threshold, the
   scheduler aggressively runs `cancel` ops to reclaim dead columns.
   Cancels themselves cost heads.
2. **Priority flip.** Under pressure, column-freeing ops are
   prioritised over critical-path progress. Under no pressure,
   critical path wins.

Lifetime matters:

- **Wide intermediates with one far-away consumer** occupy residual
  columns for the distance between producer and consumer. Shortening
  that distance frees column-layer bandwidth.
- **Concatenate is free but non-recombinable.** Concatenating values
  with different natural lifetimes pins all of them until the concat
  is consumed.

### When pressure becomes a plateau

The most damaging shape: **N parallel chains feeding a common
Concatenate**, where each chain has a wide intermediate that's much
wider than the chain's terminal output. Classic example: an unrolled
loop where each iteration computes a one-hot select and produces a
narrow result (DOOM's tex_sample loop produced a 192-col masked-table
intermediate per row inside `dynamic_extract`, then narrowed to
3 cols).

The scheduler is greedy. With N independent chains all simultaneously
ready, it admits as many as fit. Each in-flight chain pins its wide
intermediate until the chain's terminal places. If `K` chains are
in flight, residual occupancy hits `K × peak_intermediate_width`. If
that exceeds the pressure threshold, the scheduler enters a long
plateau: 95–99% occupancy, low ops/layer, MLP packing collapses, and
compiled N inflates well beyond DAG critical path.

**How to recognise this pattern:**

- The graph-stats diagnostic (see intro) shows DAG critical path much
  shorter than compiled `N`.
- Verbose compile log shows a long stretch of high-occupancy layers
  with low op counts.
- A per-annotation occupancy probe (see §11) breaks per-layer
  occupancy down by annotation; one annotation will dominate the
  plateau (e.g., `render/column_fill/tex_sample` was 63% of the
  plateau for DOOM at d=2048).

### Lever: `sequential_scope` for parallel chains

`torchwright.graph.scheduling_hints.sequential_scope(factories,
batch_size=K)` calls each factory in order, identifies per-iteration
node sets via creation-order ID ranges, and wires synthetic scheduling
predecessors: iteration `i`'s entry nodes wait until iteration
`i - K`'s terminal is in `computed_nodes`. The scheduler honours these
via `GraphAnalyzer.is_ready` — they're not data inputs, so compute
semantics are unchanged, only ordering.

Effect: at most `K` chains are in flight concurrently. Tune `K` so
peak residual occupancy from in-flight chains stays well below the
pressure threshold.

```python
from torchwright.graph.scheduling_hints import sequential_scope

row_rgbs = sequential_scope(
    [lambda y_idx=y_idx: _build_tex_row(y_idx)
     for y_idx in range(rows_per_patch)],
    batch_size=8,
)
```

**Tuning K — empirical scaling on DOOM:**

| Setup                       | Optimal `K` | Compiled N |
|-----------------------------|-------------|------------|
| `d=2048`, `chunk_size=20`   | 8           | 51         |
| `d=4096`, `chunk_size=100`  | 16          | 63         |

`K` scales roughly linearly with `d`, since the binding constraint is
fitting `K × peak_intermediate_width` into available residual budget.
A reasonable default heuristic: `K ≈ d / (4 × peak_intermediate_width)`,
but always sweep — the optimum has a sharp basin.

**Knobs that matter for tuning:**

- **`d`** — sets total residual budget. Larger `d` ⇒ optimal `K` rises
  linearly.
- **`peak_intermediate_width`** — the largest live width per chain.
  This is graph-structure-dependent; for tex_sample it was 192 (the
  width-`n_entries × d_fill` masked table `broadcast_select` produces
  inside `dynamic_extract`).
- **`chunk_size` / number of chains** — the loop unroll count. More
  chains means the plateau lasts longer if not gated, but the optimal
  `K` is determined by peak width, not chain count.
- **Other plateau contributors** — any cols pinned by *non-cluster*
  work during the same layers narrows the budget available for
  in-flight chains. Use the per-annotation occupancy breakdown to
  estimate this.

**Footgun: `K` too close to the natural in-flight count.** A hint
that's too loose disables the scheduler's organic backpressure
(greedy-admit-with-cancel) without adding effective gating. The
scheduler trusts the constraint, admits up to `K` chains in parallel,
and can deadlock when the wide intermediates won't fit. Concretely on
DOOM at `d=4096`, `chunk_size=100`: `K ≥ 50` raised
`RuntimeError: No progress`. Without `sequential_scope`, the same
graph compiles (slowly) because greedy admission only commits as many
as fit. Rule: pick `K` well below the count the scheduler would
naturally settle at — the sweet spot is in the
"prevent-plateau-but-keep-some-parallelism" middle, not near the
unconstrained ceiling.

**Footgun: `K = 1` (fully serial).** Forces every chain through one at
a time, multiplying the chain's depth by the number of iterations.
For DOOM this nearly tripled compiled N (130 layers vs 81 unbatched).

**When `sequential_scope` is the right lever:**

- The graph has ≥4 parallel chains feeding a Concatenate (or similar
  N-way join).
- Chain peak width × N exceeds residual budget.
- Per-annotation instrumentation confirms the cluster is the dominant
  plateau pinner (≥50% of pinned cols).

If only one of these holds, `sequential_scope` may not help or may
hurt — do the measurement first.

---

## 8. Graph-level fusion pass

There is a fusion pass
(`torchwright/graph/optimize.py:fuse_consecutive_linears`) that
fuses adjacent linear maps in place, FFN-aware. Five folds, each
gated so it never grows the parameter count (the sibling merge needs
no gate — it is parameter-neutral):

- **Linear → Linear** — L1's sole consumer is L2; L2 absorbs L1
  (product matrix, combined bias).
- **Linear → FFN** — the Linear's sole consumer is the FFN; the
  Linear folds into the FFN's gate (and up) projection.
- **FFN → Linear** — the FFN's sole consumer is the Linear; the
  FFN's output projection absorbs it (declined when the Linear is a
  caller-held output).
- **Linear leaves through Concatenate** — when a Concatenate's sole
  consumer is a Linear, single-consumer Linear leaves are absorbed
  into the downstream matrix and LiteralValue leaves fold into its
  bias.
- **Sibling Linear leaves of a Concatenate** — a contiguous run of
  same-input, sole-consumer Linear leaves merges into one wide
  Linear.

When it runs, chains of `multiply_const`, `add_const`, `negate`, and
other scalar affine Linears collapse into one Linear — including
through Concatenates — saving ops and layers automatically. It runs
automatically: `lower()`, the compiler's lowering boundary, applies
it to the compiler-private copy of the graph before scheduling, so
every compile entry point gets it without caller action. A fold that
would erase a checked value (`node.checks`) is declined, so asserted
values stay materialized.

Manual fusion (writing `Linear(x, combined_matrix, combined_bias)`
directly) remains worthwhile when:

- The intermediate has fanout (every fold requires a sole consumer,
  and the duplicate computation dominates).
- The fused matrix would grow the parameter count (the pass declines
  bottleneck-inflating fusions you might still want for depth).

---

## 9. Optimization techniques

The graph-stats diagnostic (see intro) gives a prioritised list of
critical-path annotations and their contiguous chain lengths — start
there. For each hot annotation, the levers are:

### Reduce depth (highest leverage)

- Hoist loop-invariant work out of unrolled loops.
- Replace `select` trees with `table_lookup_2d` lookups (§4).
- Collapse sequences of standalone affine Linears into one
  `Linear(input, combined_matrix, combined_bias)` — the fusion pass
  handles some of this automatically; the rest is manual.
- Merge cross-position reads with shared validity/mask into a single
  bundled attention call.
- Prefer `multiply_2d` (one MLP-sublayer step) over the
  `multiply_integers` chain (three) when its grid precision
  `step1 · step2 / 4` is acceptable and `d_hidden` has slots to
  spare.

### Reduce node count (medium leverage)

- Vectorise scalar ops across parallel lanes. Many primitives
  currently assume `len(input) == 1`; per-scalar operations that run
  in parallel on disjoint data are good candidates for a wider
  variant — but this usually requires extending the op library, not
  just the caller.
- Combine bool expressions: one `bool_all_true` over the whole list
  beats a tree of pairwise combines; flip negations to use
  `bool_all_true` in place of `bool_any_true` when possible.

### Tighten bounds (low leverage but pays off)

- `multiply_2d`, `reciprocal`, and `piecewise_linear` all scale
  hidden-slot count linearly with their bounds (the breakpoint grid
  spans the declared range at a fixed step). Loose bounds waste
  precision AND width.

### `d_head` (limited)

Layer count is critical-path bound, so changing `d_head` mostly
shifts param cost per head (smaller `d_head` → less waste per head,
more heads per layer). It doesn't typically buy layer reduction.

---

## 10. Anti-patterns

- **Long sequences of scalar standalone Linears** (`negate`,
  `add_const`, `multiply_const`) on the critical path. Fuse by hand
  if the optimization pass doesn't (fanout-bearing intermediates).
- **`bool_any_true([a, b])` when the negations already exist.**
  `bool_any_true` costs one more chain than `bool_all_true`.
- **Computing a value per-consumer that could be computed once
  upstream and read via attention.**
- **Loose `max_abs1` / `max_abs2` bounds on `multiply_2d`.** The breakpoint grid
  spans the declared range, so slack bounds burn hidden slots (at a
  fixed `step`) or precision (at a fixed slot count).
- **Concatenating values with different natural lifetimes.** Pins
  both until the concat is consumed.

---

## 11. Debugging strategies

### Start with per-annotation stats

The graph-stats diagnostic (see intro) is the primary measurement.
What to measure:

- Per-annotation node counts, graph params, allocated params, and
  density.
- Actual compiled layer count (this requires running the compiler).
- Critical path length and annotation breakdown.
- Longest contiguous annotation-runs on the critical path, ordered
  by length — these are the biggest depth-reduction targets.

Two caveats when reading the critical-path output:

- A critical-path trace surfaces **one example chain** of maximum DAG
  depth. If multiple chains are tied at that depth (common in
  non-trivial graphs), shortening only that one may not reduce N
  because another tied chain still binds the lower bound.
- The **DAG depth reported is a lower bound**; the compiled layer
  count may be substantially larger (roughly 2× in DOOM) because the
  scheduler inflates N when per-layer capacity or residual-stream
  pressure forces ops apart. A DAG-depth win of K layers only
  translates to a compiled-N win of K if there's scheduling slack at
  that depth. Check the layer spans in the per-annotation table to
  sanity-check: if the targeted chain's layer span is much wider
  than its op count, scheduling, not DAG depth, is the binding
  constraint.

Add `with annotate("subsystem"):` blocks liberally in your graph
construction code; annotations are free at runtime and make the
per-annotation breakdowns meaningful.

### Isolate a subsystem

Temporarily return an intermediate node as the graph output and
re-run the stats pass. Ancestors collapse to just what feeds that
node, so you can measure a subsystem in isolation.

### Read the verbose compile log

`compile_headless(..., verbose=True)` prints per-layer ops,
fill percentages, and residual-stream occupancy. Layers with very
low fill but high critical-path priority were forced by sequencing,
not capacity — those are the ones you'd reduce by restructuring
dependencies. Spikes in residual occupancy that persist across many
layers indicate a wide intermediate living too long.

### Correctness checks after structural changes

`torchwright/debug/probe.py` runs the compiled module side-by-side
with a recursive oracle evaluator over all `n_pos` prefill positions
and reports the first divergence. Run it after any graph
restructuring. For autoregressive decode behaviour, the test suite
(`make test`) is the authoritative check.

### Attribute layer count to a subsystem

Stub out a subsystem (return literal zeros for its output, or replace
with a constant) and recompile. The delta in compiled layer count
tells you how much depth the subsystem actually contributed — often
more than its allocated-params share suggests.

### Per-layer per-annotation occupancy probe

When the verbose compile log shows a residual-occupancy plateau (many
consecutive layers at 90%+), figure out *which subsystem* is pinning
columns before reaching for any heuristic tweak. No instrumentation
needed: the compiler already materializes a per-layer snapshot of
node → residual columns (each planned layer's `residual_snapshot` in
the replay plan, carried onto the compiled module as
`residual_assignment` — one mapping per post-MLP sublayer state).
Group each layer's columns by `node.annotation` and report avg cols
per annotation across plateau layers.

A plateau dominated (≥50%) by one annotation means
`sequential_scope` on that subsystem is the right lever (see §7). A
plateau spread across many annotations means the lever is elsewhere
— likely critical-path shortening or graph restructuring of the
biggest contributor.

---

## 12. Summary principles

1. **Layer count is critical-path-bound**, not capacity-bound. Saves
   come from shortening the critical path, not from shaving heads
   inside a layer.
2. **Each `Attn` node consumes a whole head-block; standalone
   `Linear`s default to the MLP bypass** (`2 × d_output` hidden
   slots). Node count in an annotation is still often the real cost
   — every node is a schedulable unit with depth implications.
3. **MLP slots are orders of magnitude cheaper than attention
   heads** per unit of work — push per-position work into
   `linear_relu_linear` chains.
4. **Attention's unique value is cross-position.** Use it for that;
   don't use it as a 1-to-1 projection.
5. **Autoregression lets earlier tokens precompute for later tokens.**
   Upstream work read via a bundled attention head often beats
   duplicating work at the consumer.
6. **The compiler fuses most adjacent linear maps — including through
   Concatenates and into/out of FFNs.** Bottleneck-inflating fusions,
   fanout-bearing intermediates, checked values, and caller-held
   outputs are declined. Fuse manually where the pass doesn't.
7. **`Concatenate` is free; a non-dead `Add` costs 1 head when
   `2 × d_out ≤ d_head`, 2 per chunk beyond that.** Fused
   `Linear(Concatenate([a, b]), [[1],[-1]])` is one op and one layer;
   `subtract(a, b)` as `negate + add` is typically a negate Linear
   plus 1 free-add head.
8. **Bound everything as tightly as possible.** `multiply_2d`,
   `reciprocal`, and the piecewise ops scale width AND precision with
   their input bounds.
9. **N parallel chains feeding a join can plateau the residual stream.**
   When per-annotation occupancy probes confirm one cluster pins
   ≥50% of plateau cols, `sequential_scope(factories, batch_size=K)`
   gates concurrency. Tune `K` so `K × peak_intermediate_width` stays
   well under residual budget; expect the optimum to scale linearly
   with `d`. Avoid `K` near the unconstrained in-flight count — it
   disables the scheduler's organic backpressure and can deadlock.

If a cost decision isn't obvious: open `compiler/forward/scheduler.py`
and read it. Zero hidden state, every placement decision is local.
