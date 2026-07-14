# Tied embeddings: held-bank implementation plan

Status: implemented and feature-verified in this worktree. The literal merge
gate is not yet green because the repository-wide run retains two unrelated
baseline failures, and production-geometry measurements remain pending; see
§15.

This plan implements one deliberately narrow design:

1. the embedding owns an ordered residual-column bank `B` at model input;
2. the compiler cancels the embedding when its last reader has finished;
3. the now-zero bank is held out of the free pool;
4. the final output is freshly written into that same ordered bank; and
5. one full-width `embed_table` is used by both the token `Gather` and the
   final unembed `MatMul`.

There is no no-hold mode, scratch window, no-late-survivor constraint,
schedule-aware bank coloring, transferred output lease, atomic MLP handoff,
unembed mask, untied token layout, or compatibility switch in this plan.
`docs/tied_allocation_problem.md` remains the broader formal analysis; this
plan instantiates only its held case, `D = P`.

The other fixed requirements are:

- token artifacts have one new layout, `torchwright.token.v6`;
- old token artifacts must be recompiled rather than accepted through a
  compatibility path;
- the literal const-one seed is cancelled in the final transformer layer;
- pinned-RMS seed columns are zeroed by the existing final normalization;
- both norm-on and norm-off logits have zero seed contribution; and
- ONNX and Hugging Face artifacts are genuinely tied, not merely initialized
  from equal values.

## 1. End state and success conditions

For compact embedding table `E` with shape `(vocab, b)`, choose an ordered
tuple of distinct residual columns

```
B = (B[0], ..., B[b-1]).
```

The completed compiler must establish all of the following:

```
input embedding coordinate k  -> B[k]
final output coordinate k     -> B[k]
embed_table[:, B[k]]          == E[:, k]
```

At runtime, the single exported table has shape `(vocab, d)` and contains:

- `E` in `B`;
- the pinned RMS constants in their reserved columns, when RMSNorm is on;
- the literal const-one seed in its compiler-reserved column; and
- exact zero everywhere else.

The graph is then exactly:

```
res_0  = Gather(embed_table, token_ids)
...
logits = final_residual @ Transpose(embed_table)
```

Immediately before the unembed:

- the learned output value occupies `B` in the same coordinate order as `E`;
- every pinned RMS seed column is exactly zero after final normalization;
- the literal const-one column is exactly zero because the final MLP added
  `-1`; and
- columns outside `B` and the seed columns have zero table weights and cannot
  contribute to logits.

The feature is complete only when:

- `output_indices == tied_embedding_indices` element by element;
- ONNX contains `embed_table` and no `lm_head` initializer;
- the same ONNX initializer feeds both `Gather` and unembed;
- native and Phi-3 HF models report `tie_word_embeddings=True`;
- `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()` before and
  after `save_pretrained` / `from_pretrained`;
- norm-on and norm-off exports have no uniform seed offset; and
- heuristic scheduling, CP-SAT replay, cache replay, and CP-SAT fallback all
  obey the same held-bank contract.

## 2. Scope boundaries

### In scope

- One tied `Embedding` input and one non-`Concatenate` output of equal width.
- Arbitrary noncontiguous bank columns; ordering is significant.
- An output written through attention or MLP, provided the embedding has
  already been cancelled before the write.
- The special same-attention-event case in which the output writer itself is
  the embedding's last reader.
- A terminal `Add`, forced to its fresh `compute_add` realization.
- Both compiler scheduling paths and all optimize levels.
- ReLU and swish machines, `bias=True` and `bias=False`, and RMSNorm on and
  off.
- Native and stock Phi-3 HF conversion.

### Explicitly out of scope

- Releasing `B` as ordinary scratch at any point.
- Choosing between hold and no-hold in CP-SAT.
- NLS or any other scratch-owner deadline.
- Assigning physical columns to ordinary nodes after scheduling.
- Letting a free `Add` transfer ownership of `B`.
- Preassigning an earlier root of a transferred output chain.
- An MLP operation that simultaneously reads, cancels, and overwrites the
  embedding bank.
- Terminal `Concatenate` outputs or slice-wise tied outputs.
- Multiple tied embeddings or multiple tied output heads.
- An unembed mask, pre-unembed gather, logit subtraction, or accepted `+1`
  offset.
- Reading `torchwright.token.v5` through a legacy branch.
- A `tie_embeddings` flag on token export.

If the held design later proves too expensive at a production geometry, that
is a new design decision. This implementation must not quietly grow the
no-hold machinery described in the formal note.

## 3. Precise lifecycle of the bank

Use the following events:

- `C_E`: the attention event that cancels the embedding value in `B`;
- `P`: the sublayer event at which the final output first claims `B`; and
- `W`: the end of the transformer layer at which the final output value is
  complete, including an MLP-side bias when applicable.

The held design enforces:

```
C_E <= P <= W.
```

The physical states are:

| interval/event | owner of `B` | runtime contents | allocator state |
|---|---|---|---|
| model input through `C_E` | tied embedding | embedding value | allocated |
| immediately after `C_E` through `P` | none | exact zero | held |
| `P` onward | final output | final output value | allocated |

The interval between `C_E` and `P` may be zero or many layers long. In either
case, held columns are not free and cannot be used by intermediates.

At a same-event attention handoff (`C_E = P`), the writer first captures its
source-column lists from the pre-attention residual. The layer then contains
both additive writes:

```
cancel:  B += -embedding
writer:  B += desired_output
skip:    B starts as embedding
```

so the post-attention value is exactly:

```
embedding - embedding + desired_output = desired_output.
```

This is already the execution model of the attention sublayer: heads read the
same pre-sublayer residual and their output deltas are summed. No sequential
runtime operation or copy is added.

An MLP writer that directly reads the tied embedding cannot use this rule.
The attention cancel would occur before the MLP read, while the existing MLP
cancel becomes available only after MLP target allocation. Such a graph is
rejected unless the writer is a flex `Linear` that can be forced onto
attention. This plan does not add atomic MLP handoff machinery.

## 4. Compiler-facing contract

`compile_to_onnx(output_node, embedding, ...)` already receives the exact
source `Embedding` whose compact table is exported. It must unconditionally
request a held output layout from `forward_compile`; there is no public
boolean and no untied token-export branch.

`forward_compile` is also the generic, non-token compiler, so it needs an
internal optional source-node parameter such as:

```
output_layout_source: Optional[Node] = None
```

`None` means that a generic headless compile has no forced input/output column
identity. It does not select a legacy token artifact: every token caller must
pass its embedding, and omission by `compile_to_onnx` is a compiler error.

After `lower()` returns, `forward_compile` resolves both endpoints in the
compiler-private graph:

```
source = lowered.copy_of(output_layout_source)
target = lowered.output_node
```

This mapping is important. The scheduler, CP-SAT model, allocator, and weight
writer operate only on lowered copy nodes; the residual assignment is re-keyed
to source nodes at the existing boundary after compilation.

Validate before allocating or solving:

- `source` is an `Embedding` and an input node of the lowered graph;
- it is the only reachable `Embedding` in this single-table token artifact;
- `source` is the exact embedding passed by token export, not merely the first
  `Embedding` found in a residual assignment;
- `target` is not `source`;
- `target` is not a `Concatenate` or other structural-only output;
- `target` is a schedulable writer node with a real allocation/emission path;
- `len(source) == len(target)` and the width is nonzero;
- both endpoints survived lowering as whole values, not slice records; and
- a direct source-to-target dependency is executable under the attention
  handoff rule.

For a direct dependency:

- `Attn` and `Add` targets are already attention operations;
- a flex `Linear` target is forced to `ATTN_TRANSPORT` in both the static
  resolver and CP-SAT;
- an MLP-only target such as an `FFN` fails early with a message explaining
  that held-bank tying has no direct MLP handoff; and
- the direct-read check must flatten `Concatenate` inputs so a hidden embedding
  leaf is not missed.

The compiler should construct one immutable handoff description after initial
input allocation:

```
HeldOutputLayout(
    source=source,
    target=target,
    bank=tuple(residual_map.get_indices(source)),
)
```

`bank` is compiler-local physical layout data. It is not a Phi-3 vocabulary
mapping and does not become an HF config field.

## 5. Residual allocator changes

Add one transient state to `ResidualStreamMap`:

```
_held: Set[int]
```

The allocator partition becomes:

```
free U allocated U held U reserved = {0, ..., d-1}
```

with all four sets pairwise disjoint. Held columns differ from reserved
columns:

- reserved columns are unavailable for the whole compile and hold a runtime
  constant;
- held columns were zeroed by a scheduled cancel, are temporarily unavailable,
  and will be claimed by a named output.

Add two narrow primitives:

```
hold(node, mech="attn")
allocate_at(node, ordered_cols)
```

`hold`:

1. requires `node` to own every listed column currently;
2. removes the node from `_node_to_indices`;
3. moves its columns to `_held`, not `_free`;
4. preserves the ordered list in the immutable `HeldOutputLayout`; and
5. runs the full allocator invariant check.

`allocate_at` is intentionally not a general precolored allocator in this
plan. Because this design has exactly one held bank, it requires
`set(ordered_cols) == _held`; it then removes those columns from `_held` and
assigns the exact ordered list to the target. It must reject duplicate
columns, a width mismatch, a partial bank, free columns, allocated columns,
reserved columns, or a second claimant.

Ordinary `allocate`, `free`, `reserve`, and `reassign` retain their current
meaning. No ordinary allocation may inspect or consume `_held`.

Add a non-mutating query used by the scheduler, for example:

```
can_allocate_at(node, ordered_cols) -> bool
```

`get_free_count()` continues to count only `_free`. As a result, the eager
scheduler naturally sees the real residual pressure: holding `b` zero columns
costs the same capacity as keeping `b` live columns.

Update `_TrackingResidualStreamMap` to copy `_held`, record a `hold` at the
current cancel layer/mechanism exactly as it records `free`, and preserve the
ordinary rollback behavior. Warm-start cancellation hints must describe the
physical cancel event `C_E`, not the later claim `P`.

At successful compile completion `_held` must be empty, `source` must no
longer own columns, and `target` must own exactly `bank`.

## 6. Scheduler and replay changes

Thread the same `HeldOutputLayout` through:

- `LayerScheduler`;
- `DirectedLayerScheduler`;
- `_run_heuristic_warm_start`;
- the optimize>0 fallback scheduler; and
- cache replay.

There must be one executor contract, not a CP-only special case.

### 6.1 Centralize physical release

The scheduler currently calls `residual_map.free` at several real cancellation
sites. Route those calls through one helper:

```
_release_cancelled(node, residual_map, mech)
```

It calls `hold` for the layout source and ordinary `free` for every other
node. Call it only after the cancellation has been accepted into the
attention batch or the MLP cancel op list. A logical death alone must never
move a column into `_held`.

Use the helper in:

- attention cancellation promotion;
- same-consumer cancellation/reuse;
- the final attention cancellation pass; and
- MLP cancellation, even though graph inputs are currently ineligible there,
  so future call-path changes cannot silently bypass the invariant.

### 6.2 Force the target to use the bank

Change `_try_allocate` as follows:

- for the layout target, return `allocate_at(target, bank)` only when the full
  bank is held;
- never allocate the target from ordinary free columns; and
- for every other node, retain ordinary allocation.

Thus a ready output waits if the embedding has not actually been cancelled,
even when unrelated columns are free. Add a specific skip/deadlock diagnostic
for “held output bank not yet available” so the failure is not reported as a
misleading generic free-column shortage.

### 6.3 Prevent ownership transfer through `Add`

`Add` normally uses `reassign(dead_addend, add_node)` when an addend is dead.
The layout source must never be selected as that dead addend. Override its
dead-for-Add predicate to false. If the other addend is dead, `add_into` may
reuse the other addend; otherwise the Add uses fresh `compute_add` columns.

Independently, if the Add itself is the layout target, force it to
`compute_add` even when either addend is dead. The final output must be a fresh
claim of `B`; this plan does not trace or preassign a transferred lease root.

The CP-SAT `is_free[Add]` encoding must enforce the same two facts:

- the held source is never an inheritable dead addend; and
- `is_free[target] == 0` when the target is an `Add`.

### 6.4 Permit cancellation of the special graph input

Ordinary graph inputs keep their existing conservative behavior. The tied
source alone may be surfaced as freshly dead after its last attention reader,
because its purpose is to be cancelled and held rather than kept for later
snapshot lookup.

The target handoff participates in the atomic attention replay specified by
`cpsat_atomic_attention_replay_plan.md`:

1. capture the source columns for every attention reader in the assigned
   batch, including every target input;
2. verify that every remaining source consumer is in that captured batch;
3. add all of `B` to the coalesced cancel and charge its head demand;
4. call `hold(source)`; and
5. call `allocate_at(target, B)` while ordinary batch outputs allocate only
   from ordinary free columns.

The base heuristic retains the narrow registered `(source, target)` handoff it
needs for `optimize=0`. Directed replay uses the general atomic batch and does
not carry a separate ordinary self-consumer-reuse path.

If the target's attention work plus the required cancellation does not fit in
one attention sublayer, the handoff is genuinely infeasible at that geometry.
The compiler must report that resource conflict; it must not route a direct
reader to MLP, allocate the output elsewhere, or emit an uncharged extra
operation.

### 6.5 Route a direct flex target consistently

When a flex `Linear` target directly reads the source, resolve it to
`ATTN_TRANSPORT` for:

- optimize=0 static scheduling;
- the heuristic warm start;
- the CP-SAT model; and
- directed replay.

Implement this as a narrow forced-class input to the realization resolver,
not as an ad hoc scheduler override after the table has supposedly become
complete. The table remains the single source of truth for the writer class.

## 7. CP-SAT contract for held columns

CP-SAT does not need physical-column identity variables. It needs to account
for the fact that the `b` bank columns remain unavailable continuously even
after their value has been cancelled.

Pass the lowered source and target ids through `solve_schedule`, the model
builders, snapshot-backed builders, and their tests. Keep this separate from
`reserve_residual`: the bank is not permanently reserved, and its
cancellation still consumes attention capacity at `C_E`.

### 7.1 Source cancellation

The source remains an attention-cancelled input. For this source only, use
executor-correct last-reader bounds:

```
non-Add attention consumer c:  C_E >= layer[c]
non-Add MLP consumer c:        C_E >= layer[c] + 1
Add consumer A:                C_E >= layer[A] + is_free[A]
input floor:                   C_E >= 0
```

Equivalently, the non-Add term is:

```
layer[c] + 1 - is_attn[c].
```

The Add term preserves current executor behavior: a fresh `compute_add` may
be cancelled after its pre-attention read in the same layer, while an
`add_into` live addend remains until the following layer. Because `is_free`
is created later in the current builder, collect the source's Add terms and
post both its lower bound and pinned `AddMaxEquality` after the `is_free`
booleans exist.

The tied source is already present before transformer layer 0 begins, so its
scoped same-attention-event rule must allow cancellation in layer 0. The
ordinary-input `cancel >= 1` floor does not apply to this source; retaining it
would make a valid layer-0 `read -> cancel -> claim` handoff artificially
infeasible.

Under production pinned cancels:

```
C_E = max(0,
          layer[c] + 1 - is_attn[c]  for non-Add consumers,
          layer[A] + is_free[A]      for Add consumers).
```

Post the claim precedence:

```
C_E <= layer[target].
```

This admits equality for a valid attention handoff and rejects a direct
MLP-only handoff. The source cancellation interval remains in the attention
cumulative at `C_E` with demand `b`; the held period consumes no attention or
MLP compute.

### 7.2 Residual occupancy

Replace only the tied source's ordinary input residual interval:

```
ordinary input: [0, input_cancel_layer)
tied source:    [0, layer[target])
```

The target's existing keep-forever interval begins at `layer[target]`. With
half-open endpoint semantics, the model therefore counts exactly `b` bank
columns at every modeled layer and never counts `2b` at the handoff:

```
[0, P) U [P, horizon).
```

This is the entire held-capacity model. Do not add a release variable, scratch
window, owner deadline, or bank-coloring constraint.

If the objective's residual-waste term is enabled, use `layer[target]` rather
than `C_E` as the tied source's occupancy end. Diagnostics and objectives must
describe the same physical occupancy as the cumulative.

### 7.3 Fresh output realization and routing

Post:

- `is_free[target] == 0` for an `Add` target; and
- `is_attn[target] == 1` for a flex `Linear` target that directly reads the
  source.

No other output route is forced. An MLP target may claim the held bank in its
MLP sublayer when the source was cancelled in an earlier attention event (or
in the same layer's attention sublayer without directly reading it).

### 7.4 Hint, replay, and snapshot parity

Update hint validation for the tied source's discounted attention-reader
bound. Otherwise a valid held warm start will be silently discarded as an
invalid ordinary-input hint.

The live-graph builder and snapshot-backed builder must create equivalent
protos for the held configuration. Thread the canonical source identity into
the snapshot/build call rather than inferring “the first embedding.”

Add the canonical held-source identity to the schedule fingerprint. The
target is already the fingerprint root, but including both endpoints in the
payload makes the constraint input explicit. This is required because the
generic compiler can schedule the same topology without a token handoff.

The replay-depth tripwire remains unchanged and must pass. A cancel deferred
past its assigned layer prevents the target claim and is an executor/model
parity bug, not permission to allocate a different column set.

## 8. Final literal-seed cleanup

The full-width tied table contains seed values as well as learned embedding
coordinates. Those seed residuals must be zero before the same table is used
as the unembed weight.

### 8.1 Literal const-one

The compiler-internal `const_one` must remain `1.0` through the final
attention sublayer because rotary self-match heads in that layer read it. It
must be zero in the final post-MLP residual.

Add one compiler-internal MLP op type, for example:

```
MLPOp("clear_literal_seed", const_one, [const_col])
```

After `schedule_layer` reveals that the logical target was computed in this
layer, append the clear op before `write_mlp_sublayer` runs. The writer adds
`-const_one.value` to the same residual column:

- with `bias=True`, accumulate into the existing MLP output-bias vector;
- with `bias=False`, use `BiasFold.out_bias` and the already-reserved constant
  lane; and
- for both ReLU and swish, read the pre-MLP const-one value to produce the
  additive `-1` exactly.

This uses no ordinary residual column, attention head, or additional MLP
slot. Under `bias=False`, hidden slot 0 is already removed from usable
capacity for the whole compile. The op must be included in placement and
parameter-count diagnostics, but it adds no CP scheduling decision.

All final-layer MLP operations read the pre-sublayer residual, so clearing the
seed does not break a bias fold or any other final-layer computation. After
weight writing, retire `const_one` from the final live map before recording
the final snapshot. Keep its saved input-state assignment so runtime seeding
still works.

### 8.2 Pinned RMS columns

Pinned RMS constants are still needed when the final RMS denominator is
computed, so they cannot be cancelled in the transformer layer. Instead:

- every per-layer pre-attention and pre-MLP norm keeps the uniform gain;
- the final norm starts from the same uniform gain vector;
- only `rms_spec.reserved_cols` are set to exact `0.0` in `final_norm`; and
- learned columns, including `B`, keep the identity gain.

The final RMS first reads the pinned energy, then its per-column multiply
zeros the pinned coordinates. This is existing normalization machinery, not
an unembed mask. No standalone `Mul`, gather, or mask initializer is added.

With `rms_norm=False`, there are no pinned RMS columns and the final-layer
literal clear is still emitted. Therefore norm-off logits do not carry a
uniform `+1` offset.

## 9. ONNX artifact changes

In `compile_to_onnx`:

1. pass the explicit source `embedding` into the held compiler contract;
2. retrieve that exact source node's input-state columns;
3. retrieve the output's final-state columns;
4. assert ordered equality, not set equality;
5. build the one full-width `embed_table` exactly as today;
6. stop constructing or emitting `lm_head`;
7. transpose `embed_table` directly for the final `MatMul`; and
8. bump `TOKEN_META_FORMAT` to `torchwright.token.v6`.

The ordered assertion is:

```
output_indices == tied_bank
```

not merely equal width or equal sets. Coordinate `k` of the compact table must
mean the same feature on both sides.

The zero-initialization contract remains unchanged. `embed_table` is still
zero off the learned and seed columns, so every initially free residual
column starts at exact zero.

Update the metadata reader and HF converter to accept only v6. Do not retain a
v5 branch. A v5 artifact must fail with the existing “re-export/recompile”
style message.

## 10. Hugging Face conversion

Both converter paths derive the full `(vocab, d)` embedding weight from the
single `embed_table` initializer.

Set `tie_word_embeddings=True` in:

- `TorchwrightConfig` construction; and
- stock `Phi3Config` construction.

For conversion, consume `embed_table` once. If `load_state_dict` requires both
parameter names, place the same tensor object under both keys in the transient
state dict:

```
weight = tensor_from(embed_table)
state_dict["model.embed_tokens.weight"] = weight
state_dict["lm_head.weight"] = weight
```

This is not a second artifact weight. It satisfies model loading while the
model's tying contract makes the two parameters aliases.

For the native model, add the transformers tying declaration mapping
`lm_head.weight` to `model.embed_tokens.weight`. Keep the standard
`get_input_embeddings`, `get_output_embeddings`, and setters. For Phi-3, use
its stock tying support.

After loading, call or verify the model's normal `tie_weights()` path and
assert storage identity. Repeat the assertion after save/reload; equality of
values is insufficient.

Remove all converter reads of an `lm_head` initializer and update the
all-initializers-consumed assertion accordingly. Update native modeling and
configuration prose so it no longer describes an untied head.

## 11. Implementation sequence

These are development phases inside one feature branch. There is no public
half-enabled token format and no final compatibility flag.

### H0: Contract and allocator

Files:

- `torchwright/compiler/forward/residual_map.py`
- `torchwright/compiler/forward/compile.py`
- allocator/compiler assertion tests

Work:

- add `HeldOutputLayout`;
- map the explicit source through lowering and validate the supported shape;
- capture the ordered bank after input allocation;
- add `_held`, `hold`, `allocate_at`, and held-aware invariants;
- copy/track held state in `_TrackingResidualStreamMap`; and
- add end-of-compile assertions for a completed handoff.

Exit tests:

- every valid state transition preserves the four-way partition;
- held columns never appear in the free count;
- ordinary allocation cannot consume held columns;
- only the full ordered bank can be claimed;
- wrong width/order/state/double claim fail loudly; and
- source and target retain element-wise bank order.

### H1: Heuristic executor and weight replay

Files:

- `torchwright/compiler/forward/scheduler.py`
- `torchwright/compiler/realization.py`
- `torchwright/compiler/forward/compile.py`
- scheduler, writer, and replay tests

Work:

- thread `HeldOutputLayout` through every scheduler construction;
- centralize real cancellation release;
- hold the source and force target `allocate_at`;
- prevent free-Add inheritance and force a fresh target Add;
- implement the scoped same-attention-event source handoff;
- force a direct flex target to attention;
- add targeted deadlock/resource diagnostics; and
- verify captured source columns remain usable after the map ownership change.

Exit tests:

- an early-cancel graph has several held layers and no intermediate uses `B`;
- attention direct-read handoff emits cancel and writer into `B` in one layer;
- direct MLP-only handoff is rejected before the layer loop;
- a final Add emits `compute_add`, never `add_into`;
- an intermediate Add never inherits the source bank;
- optimize=0 output columns equal input embedding columns; and
- attention-head exhaustion for direct handoff fails as a modeled capacity
  error rather than allocating elsewhere.

### H2: CP-SAT and cache parity

Files:

- `torchwright/compiler/forward/cpsat_scheduler.py`
- `torchwright/compiler/forward/cpsat_snapshot.py`
- `torchwright/compiler/graph_identity.py`
- `torchwright/compiler/forward/compile.py`
- CP-SAT, snapshot, cache, and replay tests

Work:

- thread source/target identity through live and snapshot builders;
- post the tied-input reader bounds and pinned equality;
- post `C_E <= layer[target]`;
- extend tied residual occupancy to target birth;
- force fresh output Add and direct flex attention route;
- update waste accounting and hint validation;
- key the cache on the canonical held source; and
- run directed replay through the same allocator contract.

Exit tests:

- a capacity-tight fixture proves the bank is counted during the clean-held
  interval;
- the handoff layer counts `b`, not `0` or `2b`, bank columns;
- cancellation compute is charged at `C_E`, not at `P`;
- equality `C_E = P` works for an attention reader;
- direct MLP-only use is infeasible or preflight-rejected;
- an output Add has `is_free == 0`;
- held warm-start hints pass strict validation;
- live and snapshot model protos agree;
- cache replay preserves the handoff; and
- `_check_replay_depth` sees no cancellation or output deferral.

### H3: Final seed cleanup

Files:

- `torchwright/compiler/forward/weight_writer.py`
- `torchwright/compiler/forward/compile.py`
- `torchwright/compiler/export.py`
- writer, no-bias, RMSNorm, and end-to-end tests

Work:

- add and write `clear_literal_seed`;
- append it only to the final logical-output layer;
- retire the literal from the final live snapshot;
- zero only pinned RMS columns in `final_norm`; and
- update placement and parameter diagnostics.

Exit tests cover the Cartesian product:

```
activation = relu, swish
bias       = true, false
rms_norm   = true, false
```

For every supported combination, assert the literal seed is exactly zero in
the final transformer state. With RMSNorm on, also assert pinned columns are
nonzero before final norm and exactly zero after it. Learned output columns
must retain the expected values.

### H4: v6 ONNX layout

Files:

- `torchwright/compiler/export.py`
- `torchwright/compiler/onnx_load.py`
- token export/load/debug tests

Work:

- make token export unconditionally request held tying;
- assert ordered bank equality;
- remove `lm_head` construction and emission;
- feed `embed_table` to both ends;
- bump the format to v6 and update only the exact-match gates; and
- update comments and debug expectations.

Exit tests:

- ONNX has one table initializer and no `lm_head`;
- graph inputs show the one table feeding `Gather` and `Transpose`;
- the table is zero in every initially free column;
- direct reference logits equal `output_value @ E.T` within the established
  fp32 tolerance;
- norm-off logits have no `+1` delta;
- generation demos remain correct; and
- v5 fails rather than taking a compatibility branch.

### H5: HF tying

Files:

- `torchwright/compiler/hf/convert.py`
- `torchwright/compiler/hf/configuration_torchwright.py`
- `torchwright/compiler/hf/modeling_torchwright.py`
- native and Phi-3 HF tests

Work:

- consume one table;
- set both configs to tied embeddings;
- declare native tied-weight keys;
- remove untied converter assumptions;
- verify load, save, reload, generation, and cache parity; and
- update shipped trust-remote-code files in the same change.

Exit tests:

- both model families have storage-identical embedding/head weights;
- aliasing survives save/reload;
- ONNX and HF logits agree for norm-on and norm-off fixtures;
- no missing/unexpected parameter is ignored; and
- the saved bundle contains one tied tensor according to the framework's
  serialization convention.

### H6: Repository-wide verification and documentation

Run all tests through the repository's required `make test` entry point.
Exercise every token-compilable example at optimize=0 and its production
optimize level. Record:

- layer count;
- solver status and fallback provenance;
- source cancel layer `C_E`;
- output claim layer `P`;
- held duration `P - C_E`;
- held width `b` and peak residual occupancy; and
- end-to-end expected output.

Use existing layer ceilings and recorded pre-change layer counts as regression
baselines; do not retain an untied runtime mode merely to produce an A/B axis.
For the external DOOM geometry, measure the held 600+ column bank explicitly
before release. A layer-count or width-floor regression is a release decision
about this held design, not authorization to add no-hold machinery to the
same implementation.

Update stale tied-layout prose in at least:

- `docs/plan_rmsnorm.md`;
- `docs/no_bias_plan.md`;
- `docs/phi3_conversion_plan.md`;
- `docs/rope_port_plan.md`;
- compiler/HF module docstrings; and
- any README or roadmap text that calls the head untied or names token.v5 as
  current.

## 12. Required adversarial tests

Happy-path examples are insufficient. Add small focused graphs for these
failure modes:

1. **Held capacity is real.** After `C_E`, ordinary free capacity remains
   `d - b - permanent`, and a width-`b` intermediate cannot occupy `B`.
2. **No double count.** A schedule that exactly fills residual capacity at
   `P` remains feasible because source `[0,P)` and target `[P,H)` meet at a
   half-open boundary.
3. **Cancel capacity is real.** A direct attention writer plus `b` cancel
   columns exceeding the attention pool is rejected or moved only in a way
   that preserves direct-read semantics.
4. **No early claim.** Even with many unrelated free columns, target
   allocation fails until all of `B` is held.
5. **No Add theft.** A dead tied embedding offered to an intermediate free Add
   is not reassigned; the bank becomes held only after a real cancel.
6. **Fresh terminal Add.** A terminal Add with a dead addend still uses
   `compute_add` and lands in `B`.
7. **Direct attention handoff.** The writer reads the original embedding while
   cancel and output deltas overlap on `B`, producing the exact expected
   result.
8. **Direct MLP rejection.** An MLP-only target reading the embedding fails
   with the intended diagnostic, not a no-progress loop.
9. **Ordered coordinates.** A deliberately noncontiguous bank preserves the
   compact table's coordinate order; set equality alone would fail the test.
10. **Final literal zero.** The last attention layer still sees one, while the
    final post-MLP state sees exact zero.
11. **No-bias clear.** The constant lane emits `-1` without using a scheduled
    hidden slot or a physical bias initializer.
12. **Pinned RMS zero.** Final norm uses the pinned value in its denominator
    and emits exact zero at its reserved coordinates.
13. **No mask.** The ONNX graph contains no new masking/gather/subtraction op
    between final residual and tied `MatMul`.
14. **Cache identity.** Held and generic headless schedules of the same graph
    cannot share a fingerprinted assignment.
15. **Fallback parity.** A forced CP-SAT no-incumbent path still produces a
    tied held-bank artifact through the heuristic fallback.

## 13. Invariants to assert in production code

Keep these assertions close to the mutations they guard:

- `bank` has exactly `len(source) == len(target)` distinct columns;
- `bank` is ordered and never normalized through a set or sort after capture;
- before `C_E`, `source` owns exactly `bank`;
- after `C_E` and before `P`, every bank column is held and no node owns it;
- at `P`, `target` receives exactly `bank` in one transition;
- no ordinary allocation or `reassign` touches a held column;
- the source is cancelled exactly once;
- the target is never allocated outside `bank`;
- the completed compile has no held columns;
- the final literal seed is not present in the final live map;
- exported output and embedding index lists are element-wise equal;
- `lm_head` is absent from v6 initializers; and
- HF input/output embeddings share storage.

## 14. Acceptance gate

The implementation is ready to merge only when all of the following are true:

- the allocator, heuristic, CP-SAT, directed replay, fallback, and cache paths
  pass their held-bank tests;
- `make test` passes;
- every committed token example compiles and produces its expected output;
- norm-on/off and bias-on/off seed-zero tests pass exactly;
- native and Phi-3 save/reload alias tests pass;
- v6 exports contain one table and no `lm_head`;
- no compatibility flag, v5 reader, unembed mask, or accepted logit offset is
  present; and
- the production-geometry held-capacity and layer-count measurements have
  been reviewed.

If the final measurement rejects the held design, stop and write a new plan.
Do not extend this plan with scratch-window allocation as an implementation
detail.

## 15. Implementation and verification record (2026-07-13)

The held-only design in this document is implemented in this worktree. The
implementation includes allocator state, heuristic and directed scheduling,
CP-SAT live/snapshot parity, cache identity, final seed cleanup, token.v6 ONNX
export, native/Phi-3 HF tying, and the associated documentation updates.

Feature-specific verification is green:

- `tests/compile/forward/test_tied_embeddings.py`: 11 passed;
- `tests/compile/forward/test_residual_map.py`: 12 passed;
- `tests/hf/test_convert.py`: 10 passed;
- `tests/hf/test_phi3_convert.py`: 20 passed;
- `tests/hf/test_rms_norm_identity.py`: 8 passed;
- `tests/compile/forward/test_module.py`: 17 passed; and
- the tied tests also passed inside the repository-wide run.

The final `make test` run completed all ten shards. Shards 0–8 passed. Shard 9
reported 1,266 passed and two failures unrelated to tied allocation:

1. `test_solver_same_layer_handoff_replays_correctly` selected a CP-SAT
   assignment that deadlocked during generic replay. The same test passed in
   an immediate isolated `make test` run, and it had passed in the preceding
   full run; this is schedule-choice nondeterminism in a non-held compile.
2. `test_committed_measurements_match_current_code` reproduced the existing
   Modal staircase GEMM reduction-order mismatch already documented in
   `docs/numerical_noise_findings.md`. The prescribed local
   `make measure-noise` produced no metric changes, only generated
   date/commit churn, which was discarded as that finding instructs.

These results establish the implementation behavior but do not waive §14's
literal `make test` requirement. Before merge, the repository owners must
either make those two baseline gates deterministic or explicitly disposition
them. The external DOOM-width held-capacity/layer-count measurement required
by H6 also remains a release measurement; it is not authorization to add a
no-hold mode.

## 16. Addendum: direct-HF builder retargeting (2026-07-14)

Section 10 targets `torchwright/compiler/hf/convert.py`, and section 15's
record reflects that path: at implementation time the Phi-3 tie lived in the
converter and `tests/hf/test_convert.py` verified it. The subsequent merge of
main replaced the converter with the direct streaming builder
(`torchwright/compiler/hf/build.py`, commit 1706c8b), which had been written
untied for stock Phi-3 before the v6 contract existed. The merge deleted
`convert.py` and its tests, silently dropping the Phi-3 half of section 10's
acceptance criteria; the post-merge reconciliation (d006732) carried the
held-bank scheduling contract into the direct path but kept the stock target
untied and rewrote the parity tests to assert that state.

This addendum records the correction. The direct builder is now the sole HF
path and satisfies section 1's success conditions for both targets:

- stock `Phi3Config` is constructed with `tie_word_embeddings=True`;
- both targets serialize only `model.embed_tokens.weight`; stock Phi-3's
  normal `tie_weights()` path reconstructs the `lm_head.weight` alias at
  load;
- `TokenModelWeights` no longer carries a separate `lm_head` projection
  (`build_token_weights` still validates the ordered-bank equality that
  makes the tied readout exact); and
- `tests/hf/test_direct_phi3_parity.py` and `tests/hf/test_phi3_compile.py`
  assert storage identity after compile and after save/reload, and the
  absence of `lm_head.weight` from every serialized weight map.

Where section 10 says `convert.py`, read `build.py`; the contract is
unchanged.
