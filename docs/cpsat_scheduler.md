# cpsat_scheduler

Optimization-driven scheduler for `forward_compile`. Produces an optimal
placement of every graph node into transformer layers, an optimal
cancellation timing for every node's residual columns, and (optionally)
an optimal attention-versus-MLP routing for every standalone `Linear`.

## 1. Overview

`forward_compile` turns a computation graph into a
`HeadlessTransformer` (the compiler's output type — a transformer with
no embedding layer and no LM head, just a residual-stream and stacked
attention-plus-MLP layers) by walking the graph layer by layer,
placing nodes into either the attention sublayer (which costs attention
heads) or the MLP sublayer (which costs MLP hidden slots), and
cancelling residual columns when nodes become dead.

Two scheduler implementations exist:

- **`LayerScheduler`** (heuristic). Greedy, layer by layer, with no
  lookahead. Picks the next layer's contents based on local pressure
  (residual occupancy, critical-path length, ready set). Fast — no
  solver overhead — and good enough for many graphs.

- **`DirectedLayerScheduler`** (this subsystem). A subclass of
  `LayerScheduler` that takes a precomputed `ScheduleAssignment` from
  the CP-SAT solver in `cpsat_scheduler.py` and replays it. The solver
  considers all layers simultaneously and proves an optimal schedule
  under a configurable cost objective. Because `DirectedLayerScheduler`
  is a subclass that overrides only the macro decisions (which node
  goes in which layer, in which sublayer, when each is cancelled) and
  inherits the parent's allocator and op-emission code, every
  micro-decision and every runtime invariant the parent enforces
  (cancel batching, source column capture, dirty-bit tracking, the
  four allocator invariants I1–I4 documented in `CLAUDE.md`) holds
  unchanged.

The CP-SAT scheduler exists because the heuristic's local decisions —
which `Linear` goes to attention versus MLP, when to cancel a dead
node — are globally suboptimal on the `(layer count, attention head
count)` Pareto front. The solver enumerates that front under a
configurable cost objective. The user navigates it via
`Costs(alpha, beta, gamma)` (see §4); two notable points on it are
"layer-min" (matches or beats every heuristic policy on layer count)
and "heads-min" (matches the lowest-attention heuristic policy with
fewer layers).

## 2. Architecture

### Code map

- `torchwright/compiler/forward/cpsat_scheduler.py` — the solver.
  Exports `solve_schedule()`, `ScheduleAssignment`, `Costs`, and
  `SolveStats`.
- `torchwright/compiler/forward/scheduler.py` — adds
  `DirectedLayerScheduler` next to the existing `LayerScheduler`.
- `torchwright/compiler/forward/compile.py` — adds the `optimize`
  level kwarg to `forward_compile` and the warm-start probe that
  feeds CP-SAT a complete heuristic hint.

### Data flow

```
   computation graph
   (output_node, pos_encoding)
              │
              ▼
   ┌───────────────────────────────┐
   │  cpsat_scheduler.py           │
   │    solve_schedule(            │
   │      graph, d, d_head,        │
   │      d_hidden, costs,         │
   │      flex_routing,            │
   │      time_budget,             │
   │    )                          │
   └───────────────────────────────┘
              │
              ▼
   ScheduleAssignment
     node_to_layer
     node_to_cancel_layer
     node_to_routing
              │
              ▼
   ┌───────────────────────────────┐
   │  forward_compile(use_cpsat=…) │
   │    │                          │
   │    ▼                          │
   │  DirectedLayerScheduler       │
   │    .schedule_layer(L)         │
   │    for L in 0..n_layers       │
   └───────────────────────────────┘
              │
              ▼
   HeadlessTransformer
```

### `ScheduleAssignment`

The contract between the solver and the replay. A frozen dataclass:

```python
@dataclass(frozen=True)
class ScheduleAssignment:
    node_to_layer: Dict[int, int]
    node_to_cancel_layer: Dict[int, int]
    node_to_routing: Dict[int, str]   # "attn" or "mlp"
    n_layers: int
```

Every schedulable node — every non-`Concatenate`, non-input node in the
ancestor cone of `output_node` — appears in all three dicts.
(`Concatenate` is the graph's "view" op: it has no value of its own and
is never placed in the residual stream; consumers reference its leaves
directly.) When the solver returns `OPTIMAL` or `FEASIBLE`, the
assignment is fully populated and respects every constraint in §3. On
`INFEASIBLE` or unrecoverable time-out, no assignment is returned and
`forward_compile` raises (see §5).

The solver guarantees:

- `node_to_layer[n]` is the transformer layer where `n` executes.
- `node_to_cancel_layer[n]` is the layer where `n`'s residual columns
  are reclaimed. Set to `n_layers` for nodes that stay alive forever
  (inputs, output, output-cone leaves).
- `node_to_routing[n]` is either `"attn"` or `"mlp"` — which sublayer
  of `node_to_layer[n]` runs the op.

The MLP composite is a single **`FFN`** node (`torchwright/graph/ffn.py`)
— a feed-forward unit with `n_lanes` hidden lanes: a gate projection, an
optional up projection, and an output projection, with `act` either
`"relu"` or `"swish"`. Graph lowering builds the `FFN` before scheduling,
so the solver sees one schedulable node, not a triple. It always runs in
the MLP sublayer (`is_attn` pinned to `0`) and carries routing `"mlp"`;
its `n_lanes` hidden lanes live in the MLP hidden-slot pool
(`demand_hidden_slots(FFN) == n_lanes`) while its output uses residual
columns like any other node.

> **Historical note.** Earlier revisions detected a `(L1, R, L2)` chain
> triple (`Linear → ReLU → Linear`) and scheduled it atomically, with
> "chain-internal" nodes and an "exclusive `L1`" that got no residual
> columns. That model is gone: the composite is now the single `FFN`
> node above, `uses_residual` is unconditionally `True` for every
> schedulable node (`cpsat_scheduler.py:uses_residual`), and the word
> "chain" no longer appears in the scheduler. Passages below that still
> say "chain" are describing the retired model.

### `DirectedLayerScheduler`

Subclass of `LayerScheduler`. Three things change relative to the
heuristic; everything else inherits from the parent and runs unchanged.

What it overrides:

- **Ready filter.** Only nodes with `assignment.node_to_layer[n] ==
  current_layer` are eligible to schedule this layer. Nodes whose
  layer has not arrived yet stay deferred.
- **Routing.** Each `Linear` is forced into the attention sublayer or
  the MLP bypass per `assignment.node_to_routing[n]`. The
  `policy.local_in_attention` setting is ignored.
- **Cancellation.** At each layer `L`, cancels are queued for every
  node where `assignment.node_to_cancel_layer[n] == L`. The
  heuristic's eager freeing of dead nodes is suppressed.

What it preserves (by inheriting the parent's per-layer code path):

- Cancel coalescing into a single batched
  `AttnHeadOp("cancel", None, cancel_cols)`. Cancels queued by the
  override flow into the parent's existing batching machinery, which
  emits one cancel op per layer.
- Dirty-bit tracking and same-batch cancellation of dirty target
  columns from fresh allocations.
- Source column capture (`q_source_cols`, `k_source_cols`, etc.) via
  `_require_live` at schedule time.
- All four allocator invariants I1–I4 (see `CLAUDE.md`). These are
  runtime assertions inside `ResidualStreamMap` and the
  weight-writer, which `DirectedLayerScheduler` doesn't touch.

## 3. The optimization model

The solver builds a CP-SAT model from the graph and minimizes the
configured objective.

### Variables

Per schedulable node `n`:

- `layer_var[n]` ∈ [0, max_layers−1] — the transformer layer where
  `n` executes.
- `cancel_layer[n]` ∈ [layer_var[n]+1, max_layers] — the layer
  where `n`'s residual columns are reclaimed. The value `max_layers`
  is the sentinel for "never freed."
- `cancel_in_mlp[n]` — boolean, per non-keep-forever schedulable node
  (never freeable inputs — they stay attention-cancelled for the
  snapshot-based value lookup). 1 routes `n`'s death cancel through the
  MLP sublayer (a `cancel_bypass` op, cost on the MLP-slot budget) with
  the uniform gap-0 consumer bound; 0 uses a batched attention cancel
  head with the routing-aware bound. See *Resource cumulatives →
  Within-layer reuse* for the mechanism and the `[layer, cancel + 1)`
  occupancy it implies.
- `is_attn[n]` — boolean. Pinned to 1 for `Attn` and `Add`. Pinned to
  0 for `LiteralValue` and `FFN` (the `FFN` composite always runs in the
  MLP sublayer). For standalone `Linear` nodes the value is free under
  `flex_routing=True` and pinned per `policy` under `flex_routing=False`
  — the standalone `Linear` is the only flex-routed node. (The retired
  chain model also had a flex "non-exclusive `L1`"; see the §2 historical
  note.)

For each `Add` node `A`:

- `is_free[A]` — derived boolean (not a free decision variable).
  `is_free[A]` is the OR over `A`'s addends `E` of "every other
  consumer of `E` finishes strictly before `layer_var[A]`," each
  consumer comparison itself reified into a `before[E, C, A]`
  boolean. An addend that is a `Concatenate`, an input/`LiteralValue`
  pinned node, or has any consumer outside the schedulable set
  (terminal `Concatenate`) is forced not-dead, matching
  `LayerScheduler._is_dead_for_add`.

The `FFN` composite (defined in §2) is a single node with one
`layer_var`, scheduled in the MLP sublayer of its layer; there is no
multi-node atomicity constraint to impose (the retired chain model
constrained the three `(L1, R, L2)` `layer_var`s equal — see the §2
historical note).

### Dependency constraints

For each directed edge `u → v` in the graph (after walking
`Concatenate` inputs to their leaves):

```
same_layer_ok = is_attn[u] ∧ ¬is_attn[v]

if  same_layer_ok:  layer_var[v] ≥ layer_var[u]
else:               layer_var[v] ≥ layer_var[u] + 1
```

This encodes the within-layer sublayer ordering: attention writes
happen first within a layer, then MLP reads. The only same-layer
producer/consumer pair that fits is `u` writing in attention and `v`
reading in MLP.

### Resource cumulatives

Three `AddCumulative` constraints, one per resource pool. The cost
function `heads_for(n)` returns the integer number of attention heads
the op consumes if scheduled in the attention sublayer:
`⌈d_v/d_head⌉` for `Attn`, `⌈d_input/d_head⌉` for `Linear`,
`⌈d_output/d_head⌉` for `Add` (the unit free-add cost — `Add`'s
actual cost is `1 ·` or `2 ·` this depending on `is_free[A]`).

**Attention budget — heads plus cancel columns plus dirty-allocation
columns, all combined.** Capacity `n_heads_per_layer · d_head` per
layer.

- For each non-`Add` node `n` with attention cost `heads_for(n) > 0`:
  an optional unit-width interval at `layer_var[n]` gated by
  `is_attn[n]`. Demand `heads_for(n) · d_head` (the column footprint
  of those heads).
- For each `Add` node `A` (always attention-routed): two optional
  unit-width intervals at `layer_var[A]`. The free-add interval is
  gated by `is_free[A]` with demand `heads_for(A) · d_head`; the
  compute-add interval is gated by `is_free[A].Not()` with demand
  `2 · heads_for(A) · d_head` (compute-add copies both addends into
  fresh columns, so it costs roughly twice the free-add heads). The
  two intervals are mutually exclusive — exactly one is active at
  `layer_var[A]`.
- For each non-pinned residual-using schedulable node `n`: a
  unit-width DEATH-layer cancel interval at `cancel_layer[n]`,
  demand `len(n)` columns.
- For each non-pinned residual-using non-`Add` schedulable node `n`:
  a unit-width BIRTH-layer dirty interval at `layer_var[n]`, demand
  `len(n)` columns. For `Add` nodes the BIRTH-layer dirty interval
  is gated by `is_free[A].Not()` — free-add reuses the dead addend's
  already-clean cols (no fresh allocation, no dirty bits to clear),
  while compute-add allocates fresh cols and pays the dirty cancel.
  The heuristic combines DEATH-layer dead-node cancels and BIRTH-
  layer dirty-allocation cancels into one batched
  `AttnHeadOp("cancel", ...)` per layer, so they share the
  attention head budget.

The combined-column form `H_a · d_head + cancel_cols + dirty_cols
≤ n_heads · d_head` is mathematically equivalent to the heuristic's
per-layer `H_a + ⌈(cancel_cols + dirty_cols)/d_head⌉ ≤ n_heads`. Both
`H_a` and `n_heads` are non-negative integers, and `d_head > 0`.
Forward: from `H_a · d_head + (cancel_cols + dirty_cols) ≤ n_heads
· d_head`, divide by `d_head` and use the fact that
`(n_heads − H_a)` is an integer to conclude
`⌈(cancel_cols + dirty_cols)/d_head⌉ ≤ n_heads − H_a`. Reverse: from
`H_a + ⌈(cancel_cols + dirty_cols)/d_head⌉ ≤ n_heads`, multiply by
`d_head` and use `cancel_cols + dirty_cols ≤
⌈(cancel_cols + dirty_cols)/d_head⌉ · d_head`.

The DEATH and BIRTH terms run for every residual-using node — which,
since `uses_residual` is now unconditionally `True`, is every
schedulable node. (Under the retired chain model an exclusive `L1` and
the chain-internal `ReLU` lived only in MLP hidden slots and had no
residual columns to cancel; those node kinds no longer exist.)

**BIRTH-dirty is over-conservative under `assume_zero_init=False`.**
The model charges a full-width BIRTH-dirty cancel to *every* fresh
allocation, but the heuristic only clears the *dirty subset* of the
allocated columns — columns recycled from a previously-cancelled node
are already clean (`ResidualStreamMap.dirty_subset`).  Under width
pressure columns recycle constantly, so the model charges hundreds of
columns of phantom cancels the replay never pays, and the pooled
attention cumulative rejects schedules the heuristic compiles.  The
exact dirty cost is allocation-order-dependent (a property of replay,
not of the layer assignment), so a sound tight per-layer bound is not
easily expressible.  The practical resolution: `compile_to_onnx`
defaults `assume_zero_init=True`, because the ONNX runtime always
builds the residual stream from zeros (`get_input_res_stream`), making
every BIRTH-dirty cancel unnecessary — both the model and the replay
then skip them and agree.  `assume_zero_init=False` stays the
conservative default of `forward_compile` for callers that may pass a
non-zero residual stream to `forward()` directly.

**MLP slot budget.** Capacity `d_hidden` per layer.

- For each MLP-routed standalone `Linear` `n`: an optional unit-width
  interval at `layer_var[n]` gated by `¬is_attn[n]`. Demand
  `2 · n.d_output` (MLP bypass slots).
- For each `FFN` `n`: a unit-width interval at `layer_var[n]`, demand
  `n.n_lanes` (its hidden lanes). The `FFN` is pinned to MLP
  (`is_attn = 0`), so the interval is unconditional. Its per-node hidden
  demand is `demand_hidden_slots(FFN) = n_lanes`
  (`cpsat_scheduler.py`).

**Residual column budget.** Capacity `d − len(pos_encoding)`.  Only
`pos_encoding` is reserved for the whole schedule (the attention
sublayer reads it at nearly every layer); every other input node is
*freeable*.

- For each residual-using scheduled node `n`: a regular interval
  `[layer_var[n], cancel_layer[n] + cancel_in_mlp[n])`, demand `len(n)`.
  The `+ cancel_in_mlp[n]` extends the occupancy one layer for an
  MLP-cancelled node (freed at the end of its cancel layer, so live
  through it); an attention-cancelled node (`cancel_in_mlp = 0`) ends at
  `cancel_layer` exactly (freed mid-attention-sublayer).  See the
  mechanism section above — leaving MLP-cancel at `[layer, cancel)`
  would let the model free columns during the cancel layer's attention
  sublayer where the replay still holds them (an unreplayable schedule).
- For each freeable input node `n` (every input except `pos_encoding`):
  a regular interval `[0, input_cancel_layer[n])`, demand `len(n)`.
  Inputs are pre-allocated at layer 0, so their birth is fixed at 0;
  `input_cancel_layer[n]` obeys the same consumer lower bound and
  cancel-slack window as a scheduled node's `cancel_layer`, and is kept
  at `max_layers` for an input feeding a terminal `Concatenate`
  (output cone).  `solve_schedule` writes these cancels into
  `ScheduleAssignment.node_to_cancel_layer`, and
  `DirectedLayerScheduler._find_dead_nodes` frees the input at replay —
  matching the heuristic, which frees consumed inputs and recycles their
  columns (the wide token `Embedding` is freed early and its 600+
  columns carry the geometry-stage intermediates).  Reserving every
  input forever (the pre-fix model) starved intermediates under width
  pressure and made the residual cumulative reject schedules the
  heuristic compiles fine.  Each freeable input also contributes a
  DEATH-layer cancel interval to the attention-head cumulative (its
  columns must be zeroed before reuse).

`uses_residual(n)` is now unconditionally `True`
(`cpsat_scheduler.py:uses_residual`): every schedulable node — `FFN`,
`Linear`, `Add`, `Attn`, `LiteralValue` — writes its output to the
residual stream and gets its own column allocation. (The `FFN`'s
internal hidden lanes live in MLP hidden slots, not the residual, but
that is the slot budget above, not a residual allocation.) The retired
chain model returned `False` here for chain-internal `ReLU` and
exclusive `L1`; those node kinds no longer exist as separate
schedulable nodes.

**Within-layer (intra-layer) reuse IS modelled — the two cancel
mechanisms (Units 1–2).**  Freeing a dead node's columns and reusing
them in the *same* layer a consumer read the node (gap-0 reuse) is a
density the model now represents, via a per-node choice of *which
sublayer the cancel runs in*:

- **attention-cancel** — a batched attention head (`AttnHeadOp("cancel")`)
  that reads the column's pre-sublayer value and adds its negation.
  Every attention head reads the same layer-entry residual, so a reader
  of X, the cancel of X, and a new writer into X's columns all compose
  within one attention sublayer (`x − x + new = new`).  The cancel may
  therefore fire at the consumer's *own* layer when that consumer reads
  in attention (gap 0); an MLP-routed consumer reads post-attention
  state, so its cancel keeps the layer-after bound (gap 1).  Encoded by
  the routing-aware bound `cancel ≥ layer[c] + 1 − is_attn[c]`.
- **MLP-cancel** — a `cancel_bypass` MLP op (the activation bypass
  lane-pair with `W = −I`, 2 hidden slots per column).  It fires in the
  MLP sublayer, *after both sublayers' reads*, so it permits the uniform
  gap-0 bound `cancel ≥ layer[c]` for **every** consumer (attention or
  MLP).  Its cost lands on the MLP-slot budget instead of the
  attention-head budget — head-budget decoupling, which lets a cancel
  fire on a head-saturated layer instead of deferring.  The per-node
  boolean `cancel_in_mlp[n]` selects the mechanism (`node_to_cancel_mech`
  in the assignment).

**The sublayer-resolution axis: half-disproven, half-resurrected.**  The
older claim that this density "needs a sublayer-resolution residual
axis" is wrong for cancel-*timing* legality — cancels fire per-layer in
a batch, and only the *consumer's read-sublayer* matters, which the
routing-aware / uniform bounds above capture at layer granularity.  But
it is exactly right for *occupancy*: because the MLP-cancel frees at the
END of its layer, an MLP-cancelled node stays live through the whole
cancel layer, so its residual interval is `[layer, cancel + 1)`, not
`[layer, cancel)` (encoded as `end = cancel_layer + cancel_in_mlp`, in
both the free-add and normal residual branches).  That is sound but
conservative: it forbids a physically-realizable *same-layer MLP→MLP
column handoff* (an MLP-born successor taking the cancelled columns in
the same MLP sublayer — composition `x + (−x) + new = new`).  Closing
that would need the sublayer-resolution axis in the model **plus** an
MLP-phase capture→cancel→free→allocate executor in the directed replay
— both or neither.  This is recorded as **known optimality gap #1** in
the derisk doc's 2026-07-08 correction (with a bindingness-measurement
recipe: re-solve with the diagnostic-only
`_disabled_families={"mlp_cancel_occupancy"}` relaxation, which reverts
the end to `[layer, cancel)` for a lower bound, and compare layer
counts); `scripts/intralayer_example_sweep.py` runs that measurement.

**Self-consumer subtlety (trap #2).**  The `[layer, cancel)` accounting
for attention-cancel is exact when a *distinct* node takes the dying
node's columns in that attention sublayer.  It is silent on a
width-starved graph forcing a node's own last consumer to be the only
reuser (`Ma0 = Linear(L0)` reusing L0's columns).  The model stays
exact; the `DirectedLayerScheduler` gains **self-consumer reuse**
(capture→cancel→free→allocate, preserving I1/I4) to realize it, scoped
to the directed replay only — the eager heuristic never learns it (that
would shift every golden layer count with no production need).

### Objective

```
minimize  alpha · n_layers
        + beta  · total_attn_heads
        + gamma · total_mlp_bypass_slots
```

where:

- `n_layers = max(layer_var) + 1`.
- `total_attn_heads = Σ heads_for(n)` over attention-routed nodes.
  For pinned-attention nodes this is a constant; for flex Linears
  it depends on `is_attn[n]`.
- `total_mlp_bypass_slots = Σ 2 · n.d_output` over MLP-routed flex
  Linears. Chain composite slots and standalone ReLU slots are
  constants and therefore don't appear in the objective.

### Model preconditions

The heuristic `LayerScheduler` distinguishes two `Add` scheduling
modes: **free_add** (one input is dead — the `Add` reuses the dead
input's residual columns and only needs to copy the live input,
costing `⌈len(n)/d_head⌉` heads) and **compute_add** (both inputs
still alive, or one input is a `Concatenate` — the `Add` allocates
fresh columns and copies both inputs, costing roughly twice that
plus a dirty cancel for the fresh cols). The model encodes both
regimes via a per-Add `is_free[A]` boolean derived from reified
consumer-ordering booleans (see *Variables*), so the cumulative
budget reflects the regime the heuristic will actually use at
replay.

The model is sound for graphs and configurations satisfying:

- `admission_control=False` in `forward_compile`. The model does not
  represent the sibling-cluster admission constraint described in
  `torchwright/compiler/forward/sibling_clusters.py`. With admission
  control on, the solver may produce schedules the replay cannot
  honor; `forward_compile` raises if you combine `optimize > 0`
  with `admission_control=True`.
- The `FFN` composite (defined in §2) is a single MLP-sublayer node;
  the model does not consider splitting it. (The retired chain model
  scheduled a `(L1, R, L2)` triple atomically — see the §2 historical
  note.)
- Standalone `Linear` is the only flex-routing-eligible node type.
  `Attn`, `Add`, `FFN`, and `LiteralValue` have routing fixed by their
  type (`FFN` and `LiteralValue` to MLP, `Attn`/`Add` to attention).

## 4. API

### `Costs`

```python
@dataclass(frozen=True)
class Costs:
    alpha: int = 1
    beta: int = 0
    gamma: int = 0
```

The objective is
`alpha · n_layers + beta · total_attn_heads + gamma · total_mlp_bypass_slots`.
All three are non-negative integers.

`alpha` weights the layer count. The default `alpha=1` always
penalizes adding a layer.

`beta` weights the total attention head count. Set this above zero
when sequence length makes attention compute expensive: per-token
attention compute scales as `O(L · d_head)` per head for sequence
length `L`, while per-layer compute (the full
`d × d_hidden` MLP matmul plus the `4 · d²` attention QKVO matmuls)
is independent of `L`. For long autoregressive sequences this means
attention dominates total compute and a small reduction in head
count pays back larger than a small reduction in layer count. As a
starting point, `beta ≈ L` makes one attention head equivalent to
one extra layer.

`gamma` weights MLP bypass slot usage. The per-layer MLP matmul
costs the full `d × d_hidden` regardless of how many slots are used
— zero-padded slots still pay — so `gamma = 0` is the normal case.
Provided so that deployments with deployment-time slot pruning can
express a non-trivial slot cost.

### `flex_routing`

Short for "flexible routing" — whether the standalone-`Linear`
sublayer choice (attention versus MLP-bypass) is a CP-SAT decision
variable rather than fixed by the policy.

When `True` (the default), each standalone `Linear` (a `Linear`
outside any chain) gets its own `is_attn` decision variable and the
solver picks attention versus MLP per node. When `False`, standalone
Linears are pinned per `policy.local_in_attention` and only the
placement and cancellation decisions are optimized.

`flex_routing=True` weakly dominates `flex_routing=False` on the
solver objective: anything `flex_routing=False` can produce is also
producible under `flex_routing=True`, and the larger search space
admits strictly better optima for objectives where the routing choice
matters. The `flex_routing=False` mode exists to support comparing
CP-SAT against a specific heuristic policy's routing choice.

### `forward_compile` integration

```python
forward_compile(
    d, d_head, output_node, pos_encoding,
    ...,
    optimize: int = 0,                # 0=heuristic,1=60s,2=330s,3=600s descent
    cpsat_costs: Costs = Costs(),     # advanced: Pareto navigator
    cpsat_flex_routing: bool = True,  # advanced: routing decision
)
```

`optimize` is the user-facing knob:

| level | scheduler | budget |
|------:|-----------|--------|
|     0 | heuristic `LayerScheduler` (default) | — |
|     1 | CP-SAT, single warm-start solve, best-feasible | 60s |
|     2 | CP-SAT, single warm-start solve, best-feasible | 330s (180 + 150 folded floor-probe) |
|     3 | CP-SAT, in-compile iterated descent, best-feasible | 600s (~180s/rung; rung k>0 hinted with best-so-far at horizon best+1, until budget or proven optimal) |

At `optimize=0` the compiler skips CP-SAT entirely — same code path
as before this subsystem existed.  Use it for fast iteration where
compile latency matters more than schedule quality.

At `optimize > 0` the flow is:

1. **Warm-start probe.** A schedule-only run of the heuristic
   produces a complete known-feasible schedule:
   `(layer, routing, cancel_layer)` per node.
2. **CP-SAT solve.** `solve_schedule` runs with the full hint and the
   heuristic's layer count as the search horizon.  Returns
   `(assignment, stats)`.
3. **Replay or fall back.** If `assignment is not None`, the
   `DirectedLayerScheduler` replays it.  If `None` (no feasible
   incumbent within budget), the compile falls back to a fresh
   heuristic `LayerScheduler` against the same residual map — users
   always get a schedule, never a bare exception from a budget
   timeout.
4. **Compile.** The chosen scheduler runs the per-layer loop and
   produces a `HeadlessTransformer`.  Token semantics are identical
   regardless of which scheduler ran — the schedule is a placement
   decision, not a value-changing transformation.

The `policy` argument is honored only when `cpsat_flex_routing=False`,
where it pins the routing of standalone Linears.  With
`cpsat_flex_routing=True` (the default), `policy` is ignored.

`cpsat_costs` is the Pareto navigator (see *Costs* above); ignored
when `optimize=0`.

## 5. Runtime behavior

### When the solver runs

At `optimize > 0`, `solve_schedule` runs once at the start of
`forward_compile`, before the first layer is allocated.  It does not
run during the layer loop.  The layer loop is then deterministic:
`DirectedLayerScheduler` reads the assignment and emits ops.

### Warm-start hints

Before invoking CP-SAT, `forward_compile` runs the heuristic
`LayerScheduler` in schedule-only mode (no weight writes) on a
clone of the residual map.  The probe captures three things per
node:

- `hint_layers[n]` — the layer where the heuristic placed `n`.
- `hint_routing[n]` — `"attn"` or `"mlp"`, recovered from whether
  the heuristic emitted `compute_linear` (attention) or
  `compute_linear_bypass` (MLP) for the node.
- `hint_cancel[n]` — the layer where the heuristic freed `n`'s
  residual columns.  Captured by a small `_TrackingResidualStreamMap`
  subclass that records the current layer when `free()` is called;
  nodes consumed via `reassign` (the free-add path) don't go
  through `free` and are correctly omitted.
- `hint_cancel_mech[n]` — `"attn"` or `"mlp"`, which sublayer the
  heuristic's cancel ran in (the same tracking-map `free(node, mech)`
  records it).  Hinted to `cancel_in_mlp[n]`.

All four are passed to `solve_schedule` as `AddHint` calls.  The
heuristic's layer count also tightens the search horizon
(`max_layers = min(user_max, hint_n_layers + 1)`), which shrinks
each `layer_var`'s domain.

**The warm start emits the eager schedule (the only mode).**  The
heuristic frees a node's columns *within* its consumer's layer and
reuses them the same layer (within-layer reuse).  Historically this
eager density was not model-representable, so the warm start ran with
freeing disabled (`eager_free=False`) and handed CP-SAT the deeper but
feasible no-eager schedule.  Units 1–2 taught the model that density —
the gap-0 attention cancels and the MLP-cancel mechanism (see *Resource
cumulatives → Within-layer reuse*) — so the eager schedule is now a
**feasible** hint, and `eager_free` has been removed entirely
(`LayerScheduler` is always eager).  CP-SAT receives the shallower eager
incumbent directly and improves from it.  The heuristic **fallback**
(used only when CP-SAT finds nothing within budget) is the same eager
schedule, so a timeout never regresses below the eager depth.  Because
the eager schedule now exercises MLP-cancel, `hint_cancel_mech[n]`
(`"attn"`/`"mlp"`) is captured alongside `hint_cancel[n]` and hinted to
`cancel_in_mlp` — without it CP-SAT would cold-choose the mechanism and
could reject an otherwise-feasible incumbent.

**The deferred-cancel infeasibility (2026-07).**  The eager-free fix
above was the first time a silently-infeasible hint caused the
optimize=2 fallback; this was the second.  The heuristic defers a
node's free to the next layer when the current layer's attention
heads are exhausted (`try_add_cancel` returns `None`), so its
captured `hint_cancel` can land past the model's uniform
`last_consumer + 1 + K` cancel window.  Since the block-IR refactor
this happened for ~385 nodes on the production e1m1 hud-on graph —
every violation exactly one layer over — so CP-SAT silently dropped
the hint, cold-searched a model too hard for its 180 s budget,
returned UNKNOWN, and every cold `optimize=2` compile shipped the
61-layer heuristic fallback instead of the ~51-layer CP-SAT schedule.
Diagnosed with `torchwright_doom/scripts/cpsat_hint_audit.py`
(hard-fix the hint → INFEASIBLE in presolve; family bisect →
`cancel_slack` is the sole rejector).  Fixed by the hint-aware
per-node window widening described under *Cancel-domain restriction*.

**The hint-validation tripwire.**  Both incidents were invisible: a
bad hint either fails the `if nid in vars` guard at the `AddHint`
site (vanishes) or is silently discarded by CP-SAT at solve time.
`solve_schedule` now runs `_validate_hint` before applying hints —
it mirrors every hint-checkable model constraint (guard drops for
non-`Concatenate` nodes, tightened layer domains, `cancel ≥ birth+1`,
`cancel ≥ consumer_layer+1`, keep-forever pins, and the *widened*
cancel windows) and reports violations.  Default behavior is one
`RuntimeWarning` (production keeps its fall-back-don't-fail
contract); `strict_hint=True` raises `ValueError` naming the first
violations (tests use strict).  Post-widening, the cancel-window
check should never fire — if it does, a third class of hint
infeasibility has appeared and should be investigated before
trusting any `optimize>0` result.

**Rollback-cancel correction.**  `LayerScheduler` speculatively
allocates a node, then rolls the allocation back with `free(node)`
when the op can't be committed under the dirty-cancel / head budget.
That rollback is not a real death — the node is re-allocated (reborn)
in a later layer, and may then die by `reassign` (free-add, never
`free`).  Recording the rollback layer as the cancel produced hints
with `cancel < birth` (e.g. freed at layer 6, actually born at 7),
which is hard-infeasible in the model — exactly the 3 records that kept
the otherwise-feasible no-eager hint from being accepted as a complete
incumbent.  `_TrackingResidualStreamMap` therefore clears a node's
stale cancel whenever it is re-allocated or reborn via `reassign`, so
the emitted `hint_cancel` is internally consistent (`cancel ≥ birth+1`,
or omitted when the node dies by reassign).

### Cancel-domain restriction

The cancel decision space is the dominant LB-search cost when the
attention/residual cumulatives are tight.  By default, each
non-pinned node's cancel layer is restricted to a small window
above its earliest dead layer:

```
last_consumer = max(layer_var[c]) over consumers c
cancel_layer[n] in [layer_var[n]+1, last_consumer + 1 + cancel_slack]
```

with `cancel_slack=2`.  The heuristic almost always cancels within
1–2 layers of the last consumer, so K=2 is generous enough to
preserve optimality while cutting the cancel-decision space ~30×.
The kwarg is on `solve_schedule` for users who want to widen or
disable it; `forward_compile` doesn't expose it (the default is
correct for every tested geometry).

**Hint-aware per-node widening (2026-07).**  "Almost always" was the
bug: freeing a node's columns costs attention-head work charged
against the same per-layer head budget as compute, and the heuristic
*defers* a free when a layer's heads are full (`try_add_cancel`
returns `None` — the free retries next layer).  On the production
DOOM graph ~385 nodes' frees land exactly one layer past the K=2
window, making the entire warm-start hint infeasible (see the
warm-start section below).  When `build_cpsat_model` receives
`hint_layers`/`hint_cancel`, each hinted node's window is therefore
widened by exactly the amount its hint needs:

```
delta_n = max(0, hint_cancel[n] - (hint_last_consumer + 1 + K))
cancel_layer[n] <= last_consumer + 1 + K + delta_n
```

where `hint_last_consumer` is the max hinted layer over the same
consumers the model's `last_consumer` var ranges over (the node's own
hinted birth layer when it has no layer-bound consumers; layer 0 for
freeable inputs).  `delta_n = 0` whenever the node or any needed
consumer lacks a hint entry, so an unhinted build is byte-identical
to the pre-widening model.  The widening is a pure relaxation — the
cancel still pays its attention-head charge at whatever layer it
lands and the residual interval still spans `[birth, cancel)` — so no
invalid schedule becomes expressible; the window keeps its moving
near-last-consumer shape instead of being anchored to a constant.  (The
residual interval it charges is still `[birth, cancel + cancel_in_mlp)`
— the window relaxes only the cancel-layer upper bound, never the
mechanism-dependent occupancy.)

### Determinism

CP-SAT runs with `num_search_workers=16` by default (override with the
`TW_CPSAT_WORKERS` env var — e.g. the 64-CPU Modal compile container
sets it to 64) and uses parallel worker strategies. Different runs may produce different `ScheduleAssignment`
values for the same model — different worker discovery orders find
different optima of equal objective value. The compiled
`HeadlessTransformer` differs across runs only in scheduling: token
outputs are bitwise identical (modulo float-point ordering effects
that already affect every compile, see *FP nondeterminism at
tolerance boundaries* in `CLAUDE.md`).

### Failure modes

**Precondition violation.** `solve_schedule` raises `RuntimeError`
on structural problems (no residual columns left after pre-allocated
inputs).  `forward_compile` itself raises if `admission_control=True`
is combined with `optimize > 0`.

**`INFEASIBLE`.** CP-SAT proves no schedule fits — typically
because `max_layers` is too small for the graph.  Returns
`(None, stats)` with `stats.status_name == "INFEASIBLE"`.
`forward_compile` falls back to the heuristic; the heuristic
respects the same `max_layers` and may itself fail with a
deadlock error.

**Time limit exceeded with no feasible solution** (CP-SAT status
`UNKNOWN`).  Returns `(None, stats)`.  `forward_compile` falls
back to the heuristic schedule.

**Time limit exceeded with feasible solution** (CP-SAT status
`FEASIBLE`).  The solver returns a `ScheduleAssignment` that is
feasible but possibly non-optimal.  `forward_compile` accepts it —
`optimize > 0` semantics treat any feasible schedule as success;
`stats.is_optimal` reports whether optimality was proven.

**Hint validation failure** (`strict_hint=True` raises
`ValueError`).  The warm-start hint contains something the model
would drop or reject — see *The hint-validation tripwire* above.  A
strict-mode raise means the hint capture and the model have drifted
apart again (the June eager-free and July deferred-cancel incidents
are the two known instances): the compile would still *work* via the
silent-fallback path, but `optimize>0` would silently have no
effect.  Fix the capture or the model — do not just switch strict
off.  In default (non-strict) mode the same condition surfaces as a
`RuntimeWarning` and the solve proceeds, usually ending in the
heuristic fallback.

### Geometry sensitivity

The win-size from CP-SAT versus heuristic depends on residual-stream
slack.  The first three rows were measured on an earlier headless DOOM
graph (~4.4K nodes); the last row is the post-J graph (~13K nodes,
the flat pass roughly doubled depth).  Each row records the
``optimize`` level (and corresponding budget) used:

| geometry                       | -O / budget | heuristic | CP-SAT | Δ    | first incumbent | OPTIMAL at |
|--------------------------------|-------------|----------:|-------:|-----:|----------------:|-----------:|
| d=2048, d_h=8192               | -O 1 (60s)  | 61        | (none) | n/a  | not in 60s      | not in 60s |
| d=3072, d_h=8192               | -O 2 (180s) | 58        | 46     | -21% | ~27s            | ~80s       |
| d=4096, d_h=4096               | -O 1 (60s)  | 59        | 46     | -22% | ~17s            | ~31s       |
| post-J d=4096, d_h=4096, dh=32 | -O 2 (180s) | 85        | 81–82  | -4%  | ~70–80s         | never (LB 50) |

These rows predate Units 1–2 and the eager warm start; the
"feasible (no-eager) hint vs eager hint" distinction below is
historical — the eager schedule is now the feasible hint (see *Warm-start
hints*), and the current expected DOOM d=4096 figure is re-measured by
the step-9 acceptance run.  At d=2048 the residual cumulative is the
binding constraint and CP-SAT struggles to close the LB gap within
budget.  At d=3072+ CP-SAT converges optimally inside the budget.  On
the much larger post-J graph CP-SAT no longer proves optimality (the
lower bound sticks at 50) but still beats the heuristic — historically
only once the warm-start fed a *feasible* (then no-eager) hint.  The
heuristic-fallback behavior (when CP-SAT can't find an incumbent) is the
right answer for d=2048 — users always get a schedule, just not
the CP-SAT one.

### Experiments tried that didn't pan out

- **Symmetry breaking on equivalent sibling chains.** Detected
  parallel chains feeding common `Concatenate` joins via
  `SiblingClusterAnalyzer`, grouped them by structural
  fingerprint, and added chained lex-min constraints between
  layer assignments.  At DOOM scale the constraints tightened
  the LB but starved incumbent-finding workers — CP-SAT never
  found a feasible solution within reasonable budgets even with
  a feasible warm-start hint.  Removed; if revisited, would
  need a different solver-parameter mix (LNS-heavy or fixed
  search strategy that respects the hint).
- **Feasibility-first stop-on-first mode.** Installed a callback
  that called `StopSearch()` at the first complete feasible
  solution.  CP-SAT's first feasible reproduces the heuristic
  warm-start; stopping there returned no improvement over
  `optimize=0` while paying the model-build cost.  Removed.
- **`repair_hint=True`** would have let CP-SAT actively complete the
  partial hint into a feasible solution, but it is **unusable in this
  OR-Tools version** (v9.15): it aborts the process with
  `Check failed: heuristics.fixed_search != nullptr` — *with and
  without* `AddDecisionStrategy`, and even with
  `search_branching=FIXED_SEARCH` (where the crash just moves from
  setup into a parallel worker mid-search).  (An earlier note here
  claimed it merely "conflicts with `AddDecisionStrategy`"; that was
  wrong — removing the strategy does not avoid the crash.)  And it
  would not have helped regardless: measured on the post-J DOOM graph,
  the ~70 s to first incumbent is **genuine model-size search**
  (≈23 s presolve + ≈45 s LB/feasibility search on a ~80 k-variable
  model with three coupled `AddCumulative` constraints), not a blocked
  seed.  The raw hint (with the 3 infeasible `cancel < birth` records)
  and the clean hint reach their first incumbent at the *same* time —
  the soft `AddHint` silently drops the infeasible values either way —
  so the cancel artifacts were a correctness curiosity, not the
  performance lever.  The lever is feeding a *feasible* hint at all
  (historically the no-eager schedule; now the eager schedule, which
  Units 1–2 made feasible — see *Warm-start hints*) plus an `optimize≥2`
  budget; CP-SAT then improves from it.
