# Optimization Guide

A conceptual guide to reducing layer count and parameter cost when
compiling a torchwright graph into a transformer. The scheduler and
compiler are small and readable (`torchwright/compiler/forward/`);
read the source when a detail matters.

This guide is deliberately light on specific numbers — the graph and
compiler evolve, and quantitative snapshots go stale. References below
to `make graph-stats` describe a per-annotation/per-stage diagnostic
that was removed from this repo in Step E (it shipped together with
the DOOM renderer and now lives in the sibling `torchwright_doom`
repo). The principles still apply to any compiled graph; replicate
the diagnostic from the verbose `compile_headless` output, or pull the
original implementation from the sibling repo if you need it. The
DOOM-anchored worked examples in this doc are illustrative — they
preserve the original numbers that informed each rule of thumb.

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
2. **Reduce distinct Linear/Attn nodes** in hot annotations — each one
   consumes a whole head-block regardless of how tiny its matrix is.

Density (`graph / allocated`) is a diagnostic for wasted head-width,
not a target. Low density in a big annotation is a compression
opportunity; low density in a small one is noise.

---

## 2. Anatomy of one layer

Each `TransformerLayer` has two sublayers
(`torchwright/compiler/groups/transformer_layer.py`):

    attn_sublayer:  out = attn(x) + x       # n_heads parallel heads
    mlp_sublayer:   out = W2 · ReLU(W1·x) + x

The scheduler (`scheduler.py:schedule_layer`) processes a layer in
phases:

1. **Attention sublayer** — packs up to `n_heads = d / d_head` heads.
   Candidates: `Attn` nodes, standalone `Linear` nodes (input isn't a
   ReLU), deferred `Add`s, free adds (`add_into`), and cancellations.
   All compete for the same head budget.

2. **Attention→MLP handoff** — after the attention sublayer adds its
   result into the residual stream, the MLP sublayer reads
   `x + attn(x)`. So **nodes that became ready because of attention
   outputs can still schedule into the same layer's MLP sublayer**.
   This is load-bearing: a `Attn → linear_relu_linear` pattern fits in
   one layer, not two.

3. **MLP sublayer** — packs chains (`L1 → ReLU → L2`), standalone
   ReLUs, constants, and bias writes into `d_hidden` slots. Each slot
   costs `2d + 2` params — **orders of magnitude cheaper per unit of
   work than an attention head**.

Key consequence: **attention-sublayer ops in the same layer run in
parallel with each other**. Two standalone Linears where one reads
the other's output cannot share a layer. The second has to wait for
the next layer — unless it's the L1 of an L→R→L chain, in which case
it goes into the same layer's MLP.

---

## 3. Per-node cost reference

What each graph node compiles to:

| Graph node | Sublayer | Cost model |
|---|---|---|
| `Attn` | attention | `ceil(d_v / d_head)` heads. |
| Standalone `Linear` (input ≠ ReLU) | attention | `ceil(d_input / d_head)` heads. Even scalar (`d_input = 1`) ops take a full head. |
| L1 of an L→ReLU→L chain | MLP | Shared with L2 as one `MLPOp`. |
| L2 of chain | MLP | `d_hidden × (2d + 2)` slot params. |
| Chain ReLU | MLP | Absorbed in L2. |
| Standalone ReLU | MLP | `d_relu × (2d + 2)` slot params. Rare. |
| `Add` (one addend dead) | attention | 1 head (`add_into`). |
| `Add` (neither dead) | attention | `2 × ceil(d_out / d_head)` heads — copies both inputs. |
| `Concatenate` | — | 0. Never allocated; compiler resolves through it. Children still need simultaneous residency. |
| `LiteralValue` | MLP | Bias entries only — effectively free. |
| `InputNode` / `PosEncoding` / `Embedding` | — | 0 cost; sits in residual stream for its lifetime. |

Confirmed at source: `_allocate_head` in
`compiler/forward/weight_writer.py` is a bump-allocator — each Linear
and each Attn node gets its own head-block, no cross-op head sharing.

### Op-level shapes

The library ops compose primitives above. Principles:

- **Every `piecewise_linear` / `piecewise_linear_2d` / `clamp` /
  `reciprocal` / `floor_int` / `compare` / `select` / `cond_gate`
  is one MLP chain** — i.e. one layer of MLP-sublayer work, regardless
  of output width. Their hidden-slot usage scales with the number of
  breakpoints / cases.
- **Every affine Linear (`negate`, `add_const`, `multiply_const`,
  `add_scaled_nodes`) is one attention head**, scaled by
  `ceil(d_input / d_head)`.
- **`subtract` is `add(a, negate(b))`** — one Linear head plus an Add;
  the Add is free when `negate_b` has no other consumers.
- **`signed_multiply` / `multiply_integers` / `multiply_2d` compose
  several chains**. See their docstrings for the current layer cost
  and precision tradeoffs — the `shallow` / `deep` choices really
  matter. Don't take the name "deep" as "better"; read the docstring.
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
   layers. In DOOM today the compiled layer count is roughly 2× the
   DAG critical-path depth, so DAG-depth work and packing/capacity
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
- Standalone `Linear` (input not a ReLU): **+1 layer** (attention
  sublayer). Two standalone Linears in sequence = 2 layers.
- `L1 → ReLU → L2` chain: **+1 layer** (MLP sublayer).
- `Attn → L1 → ReLU → L2`: **+1 layer** (attn-sublayer + same-layer
  MLP).
- Two sequential L→R→L chains: **+2 layers**.
- `Concatenate`, `Add`, `LiteralValue`, `InputNode`: **+0 layers**.

`make graph-stats` reports the actual compiled layer count and lists
the longest contiguous annotation-runs on the critical path — these
are the ops whose depth most directly drives layer count.

### How to shorten the critical path

1. **Hoist loop-invariant work out of unrolled loops.** Any
   computation whose inputs don't vary across loop iterations should
   be computed once upstream and shared. The per-iteration code then
   collapses to cheap affine Linears — which, after the optional
   fusion pass (see §8), become free.

2. **Replace nested `select` trees with `piecewise_linear_2d`.** A
   depth-`k` select tree is `k` chain layers. A 2-input
   `piecewise_linear_2d` over a dense function is 1 chain layer.

3. **Avoid expensive multipliers when a coarse grid suffices.** A
   `piecewise_linear_2d` on a small breakpoint grid is one chain; a
   full `signed_multiply` is several. Trade precision for depth
   deliberately.

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

### Hidden "uses attention" costs

These are ops that silently compile to attention heads because their
input isn't a ReLU:

- `negate`, `add_const`, `multiply_const`, chained scalar affine
  transforms.
- The base-term `Linear` that `piecewise_linear_2d` emits when the
  fit's linear coefficients are non-zero.
- The sum-collapse `Linear` at the tail of `dynamic_extract`.

Each costs a full attention head, even at `d_input = 1`. Long chains
of these are the biggest single-node-type waste to look for.

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
narrow result (DOOM's tex_sample loop produced a 192-col `masked_i`
intermediate per row, then narrowed to 3 cols).

The scheduler is greedy. With N independent chains all simultaneously
ready, it admits as many as fit. Each in-flight chain pins its wide
intermediate until the chain's terminal places. If `K` chains are
in flight, residual occupancy hits `K × peak_intermediate_width`. If
that exceeds the pressure threshold, the scheduler enters a long
plateau: 95–99% occupancy, low ops/layer, MLP packing collapses, and
compiled N inflates well beyond DAG critical path.

**How to recognise this pattern:**

- `make graph-stats` shows DAG critical path much shorter than
  compiled `N`.
- Verbose compile log shows a long stretch of high-occupancy layers
  with low op counts.
- `modal_inspect_residual.py` (or its local variant) breaks per-layer
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
  `masked_i` intermediate inside `dynamic_extract`).
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

There is an optional pre-compile optimization pass
(`torchwright/graph/optimize.py:fuse_consecutive_linears`) that
merges `Linear → Linear` pairs in-place, computing the product matrix
and combined bias. It fires when:

- L1's only consumer is L2.
- L1's input is not a `Concatenate` (the pass skips these).
- The fused matrix has ≤ the params of the separate pair (no
  bottleneck-inflation fusions).

When it runs, chains of `multiply_const`, `add_const`, `negate`, and
other scalar affine Linears collapse into one Linear — saving heads
and layers automatically. Whether it's wired into your compile
entrypoint is worth checking; DOOM's `compile_game` calls it before
`compile_headless`.

Manual fusion (writing `Linear(x, combined_matrix, combined_bias)`
directly) remains worthwhile when:

- The input is a `Concatenate` (pass skips these).
- The intermediate has fanout (pass skips these, and the duplicate
  computation dominates).
- You're using a raw `compile_headless` call that doesn't invoke the
  optimization pass.

---

## 9. Optimization techniques

`make graph-stats` gives a prioritised list of critical-path
annotations and their contiguous chain lengths — start there. For
each hot annotation, the levers are:

### Reduce depth (highest leverage)

- Hoist loop-invariant work out of unrolled loops.
- Replace `select` trees with table-valued `piecewise_linear_2d`.
- Collapse sequences of standalone affine Linears into one
  `Linear(input, combined_matrix, combined_bias)` — the fusion pass
  handles some of this automatically; the rest is manual.
- Merge cross-position reads with shared validity/mask into a single
  bundled attention call.
- Choose the shallower variant of composite multiplier ops when
  `d_hidden` permits.

### Reduce node count (medium leverage)

- Vectorise scalar ops across parallel lanes. Many primitives
  currently assume `len(input) == 1`; per-scalar operations that run
  in parallel on disjoint data are good candidates for a wider
  variant — but this usually requires extending the op library, not
  just the caller.
- Combine bool expressions: prefer `bool_all_true` to chains of
  `bool_and`; flip negations to use `bool_all_true` in place of
  `bool_any_true` when possible.

### Tighten bounds (low leverage but pays off)

- `signed_multiply`, `reciprocal`, `piecewise_linear*` all scale
  hidden-slot count linearly with their bounds. Loose bounds waste
  precision AND width.

### `d_head` (limited)

Layer count is critical-path bound, so changing `d_head` mostly
shifts param cost per head (smaller `d_head` → less waste per head,
more heads per layer). It doesn't typically buy layer reduction.

---

## 10. Anti-patterns

- **Long sequences of scalar standalone Linears** (`negate`,
  `add_const`, `multiply_const`) on the critical path. Fuse by hand
  if the optimization pass doesn't (Concatenate inputs, fanout).
- **`bool_any_true([a, b])` when the negations already exist.**
  `bool_any_true` costs one more chain than `bool_all_true`.
- **Computing a value per-consumer that could be computed once
  upstream and read via attention.**
- **Unbounded `max_abs` on `signed_multiply`.** Burns precision and
  neurons simultaneously.
- **Concatenating values with different natural lifetimes.** Pins
  both until the concat is consumed.

---

## 11. Debugging strategies

### Start with graph-stats

`make graph-stats` is the primary diagnostic. It reports:

- Per-annotation node counts, graph params, allocated params, and
  density.
- Actual compiled layer count (it runs the compiler).
- Critical path length and annotation breakdown.
- Longest contiguous annotation-runs on the critical path, ordered
  by length — these are the biggest depth-reduction targets.

Two caveats when reading the critical-path output:

- The tool prints **one example chain** of maximum DAG depth. If
  multiple chains are tied at that depth (common in non-trivial
  graphs), shortening only the displayed one may not reduce N
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
construction code; annotations are free at runtime and make
`graph-stats` output meaningful.

### Isolate a subsystem

Temporarily return an intermediate node as the graph output and
re-run `graph-stats`. Ancestors collapse to just what feeds that
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
with a recursive oracle evaluator for a single position and reports
the first divergence. Run it after any graph restructuring. For
multi-position / autoregressive behaviour, the test suite (`make
test`) is the authoritative check.

### Attribute layer count to a subsystem

Stub out a subsystem (return literal zeros for its output, or replace
with a constant) and recompile. The delta in compiled layer count
tells you how much depth the subsystem actually contributed — often
more than its allocated-params share suggests.

### Per-layer per-annotation occupancy probe

When the verbose compile log shows a residual-occupancy plateau (many
consecutive layers at 90%+), figure out *which subsystem* is pinning
columns before reaching for any heuristic tweak. The pattern: monkey-
patch `write_mlp_sublayer` to snapshot
`residual_map._node_to_indices` after each layer, then group by
`node.annotation` and report avg cols per annotation across plateau
layers. See `modal_inspect_residual.py` for a working template.

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
2. **Each `Linear` / `Attn` node consumes a whole head-block**, so
   node count in an annotation is often the real cost.
3. **MLP slots are orders of magnitude cheaper than attention
   heads** per unit of work — push per-position work into
   `linear_relu_linear` chains.
4. **Attention's unique value is cross-position.** Use it for that;
   don't use it as a 1-to-1 projection.
5. **Autoregression lets earlier tokens precompute for later tokens.**
   Upstream work read via a bundled attention head often beats
   duplicating work at the consumer.
6. **The compiler fuses some but not all adjacent Linears.**
   Bottleneck-inflating fusions, Concatenate-fed Linears, and
   fanout-bearing Linears are skipped. Fuse manually where the pass
   doesn't.
7. **`Concatenate` is free; non-dead `Add` costs 2 heads.** Fused
   `Linear(Concatenate([a, b]), [[1],[-1]])` is 1 head and 1 layer;
   `subtract(a, b)` as `negate + add` is typically 1 negate head plus
   1 free-add head.
8. **Bound everything as tightly as possible.** `signed_multiply`,
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
