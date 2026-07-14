# Tied embedding column allocation: formal problem and lemmas

Status: working formal note, 2026-07-12. The abstract allocation theorem has
been reviewed adversarially; the mapping to the current CP-SAT model and replay
is still conditional on the explicit gaps in §13. This document formalizes P1
(physical column identity) and compares hold-and-handoff with a no-hold
alternative. The current implementation plan has selected hold-and-handoff;
the no-hold material here is retained as analysis, not as implementation scope.

## Reader's map

The document has three layers:

1. §§1–8 define the physical allocation problem and prove a sufficient
   condition. These sections are independent of CP-SAT implementation detail.
2. §§9–12 map that condition onto scheduling, cancellation capacity, output
   routing, and the current pinned-cancel model.
3. §§13–16 list implementation gaps, decision gates, and the separate
   seed-column cleanup.

The central result is Theorem 1: ordinary aggregate residual capacity is
sufficient for tied allocation if no new scratch lease can outlive the tied
bank's claim by the output lease. The theorem does **not** say the current
compiler already enforces that condition.

Normative requirements carried into this note from the design discussion are:
one new artifact layout with no backwards-compatibility branch; no unembed
mask; and compiler cleanup of literal seed columns in the final transformer
layer. `tied_embeddings_plan.md` now instantiates only the held-bank case
(`D = P`). The no-hold results below remain formal analysis, not part of that
implementation plan.

The broader formal design space is:

- There is one tied artifact layout. Backwards compatibility is not a goal.
- There is no unembed mask.
- The tied bank may be released as ordinary residual scratch after the
  embedding is cancelled, or kept clean and held until the output claims it.
- Before the output lease claims the bank, every bank column must be clean:
  either never released after the embedding cancellation, or zeroed by the
  scheduled cancellation of its current scratch owner.
- Actual columns are assigned after scheduling, using the birth and physical
  release time of every scheduled value.

The folded seed columns and their final-layer cleanup are a separate problem.
They are mentioned only at the boundary of this document; they are not part
of the allocation proof below.

## 1. The question

Let the residual stream have a set `C` of allocatable physical columns. Let
the embedding and output each have width `b`. We must select an ordered bank

```
B = (B[0], ..., B[b-1])
```

of distinct columns in `C` such that:

1. embedding coordinate `k` occupies `B[k]` at model input;
2. output coordinate `k` occupies `B[k]` when the output is produced; and
3. between those two uses, columns in `B` may be allocated to arbitrary
   intermediate values, subject to the same zero-before-write invariant as
   every other recycled residual column.

The compiler already chooses when each graph operation executes and when
each dead value is cancelled. The question is:

> Given such a schedule, when does there exist an assignment of logical
> residual values to physical columns that realizes the schedule and the two
> forced placements on `B`?

The follow-on scheduling question is:

> What small set of CP-SAT constraints guarantees that every admitted
> schedule has such a physical assignment, without reserving `B` for the
> whole computation?

These are different questions. CP-SAT currently models aggregate residual
width and compute resources, but not physical column identities. A schedule
can therefore satisfy its aggregate cumulative constraint yet fail a later
precolored-column allocation unless we either model the identities or prove
that additional schedule constraints make identity feasibility automatic.

## 2. Fixed scope and assumptions

This note considers one non-`Concatenate` output whose width equals the tied
embedding width. Multiple output leaves would require a slice-wise version of
the same problem and are out of scope here.

Permanent residual columns, such as pinned-RMS columns, are removed from `C`
before this problem begins. The compile-internal constant-one column is also
not available as scratch. Thus, if the physical residual width is `d_phys`
and there are `r` permanent columns, this note uses

```
d = |C| = d_phys - r.
```

We assume:

- Intermediate values have no preferred physical columns.
- A value may occupy arbitrary noncontiguous columns. This matches
  `ResidualStreamMap` and the weight writer's indexed scatters.
- The `k`th coordinate of a value keeps one physical column for its whole
  physical lifetime. There is no implicit migration or copy.
- A scheduled cancellation really zeroes the released column. Merely becoming
  logically dead is not a release.
- Every read happens before the cancellation that is allowed to follow that
  read at the same execution event.
- Any same-event cancel-and-reuse must be explicitly supported by the
  executor and weight writer. It is not inferred just from equal integer
  layer numbers.

The last point matters in the current implementation. Attention cancellation
can participate in an in-layer handoff. An MLP `cancel_bypass` is currently
emitted after MLP compute allocations and its columns become allocatable only
after that MLP sublayer. Although `x + (-x) + y = y` makes an atomic MLP
handoff algebraically possible, that is not the allocator behavior today.
The formalism below therefore uses ordered execution events, not layer numbers
alone.

## 3. Logical death, physical release, and execution time

Let `T` be a totally ordered set of execution points. It is fine to implement
`T` as ordered sublayer phases, for example

```
A(0) < M(0) < A(1) < M(1) < ...
```

provided the implementation refines a point further wherever allocation,
reads, cancellation, and writes have different availability. We use half-open
intervals: a physical lifetime `[s, e)` occupies its column at times `t` with
`s <= t < e` and does not conflict with a new lifetime beginning at `e`.

For a logical value `v`:

- `birth(v)` is the event at which its result is written;
- `last_read(v)` is its final scheduled read;
- `death(v)` is the first event after which no semantic consumer needs it;
- `release(v)` is the event at which its columns are physically zero and
  available to a later allocation.

Always,

```
last_read(v) <= death(v) <= release(v).
```

The inequalities may collapse at a supported same-event handoff. They need
not collapse in general. In particular, a dead but uncancelled value still
occupies residual capacity and still dirties its columns.

This distinction is central: tied allocation is constrained by `release`, not
by semantic `death` and not just by the last consumer's layer.

Four global events must also remain distinct:

| event | meaning |
|---|---|
| `C_E` | the tied embedding value is cancelled, leaving `B` clean |
| `D` | the clean tied bank is released to the ordinary scratch allocator |
| `P` | the final output ownership lease first claims `B` |
| `W` | the logical output node is computed |

They satisfy

```
C_E <= D <= P <= W.
```

`C_E` and `D` differ under hold-and-handoff: the embedding is cancelled early
at `C_E`, but its clean columns remain withheld until `D`. `P` and `W` differ
when the output is a free `Add` or another exact-column transfer: a predecessor
lease must claim `B` at `P`, then ownership transfers to the logical output at
`W` without a fresh allocation.

The important special cases are:

- **no hold, fresh output:** `C_E = D` and `P = W`;
- **hold-and-handoff, fresh output:** `C_E < D = P = W`;
- **released scratch plus transferred output:** `C_E <= D < P < W`.

This separation repairs a misleading shortcut in an earlier draft. Delaying
the embedding cancellation until `W` is not equivalent to hold-and-handoff:
it moves cancellation compute into the output layer. A design that truly
contains both strategies must keep `C_E` early and make `D` the bank-release
choice.

## 4. Coordinate strands and transfer contraction

A width-`w` node is treated as `w` coordinate strands. This is sound because
the allocator permits arbitrary noncontiguous columns. A strand has width one
and occupies a single physical column throughout its physical lifetime.

Most node births begin new strands. A free `Add` is different:
`ResidualStreamMap.reassign(dead_addend, add_node)` transfers ownership of the
dead addend's existing columns to the `Add` output without zeroing or freeing
them. The physical column lifetime is continuous across that logical-node
boundary.

Define a **transfer edge** from strand `x` to strand `y` when the executor
requires `y` to inherit `x`'s exact physical column without an intervening
release. Contract every maximal chain of transfer edges into one **lease**.
A lease `q` has:

- a start `s(q)`, the birth of the first strand in the chain;
- an end `e(q)`, the physical release of the last strand in the chain; and
- one physical column, unchanged throughout `[s(q), e(q))`.

For a width-`w` ownership chain, this produces `w` unit leases. We can group
unit leases with identical constraints back into a weighted interval for
efficiency, but the unit formulation is the semantic ground truth.

Why contraction is necessary: the original addend can be logically cancelled
at the `Add`, yet its physical column remains occupied by the `Add` output. A
test based only on the original node's cancel layer would incorrectly declare
that column available for the tied output.

Transfer contraction applies through the output node too. If the output is a
free `Add`, the final output lease begins at the birth of the first inherited
owner (`P < W`), not when the `Add` is computed. That whole lease must be
preassigned to `B`. A final `allocate_at(output, B)` cannot realize this case,
because replay calls `reassign(dead_addend, output)` and performs no allocation
at `W`.

## 5. The fixed-schedule allocation problem

After transfer contraction, let `Q` be the ordinary intermediate leases. Each
`q in Q` has interval `[s(q), e(q))`.

Let `I = {I_0, ..., I_(b-1)}` be the initial tied-bank tenures. `I_k` contains
the embedding value until `C_E`; if `C_E < D`, it then contains zero but remains
withheld. Let `O = {O_0, ..., O_(b-1)}` be the final transfer-contracted output
leases. Then

```
I_k occupies B[k] on [T_start, D)
O_k occupies B[k] on [P, T_end)
```

The scratch window is `[D, P)`. A zero-width window (`D = P`) is ordinary
hold-and-handoff.

The current implementation makes `C_E` common across embedding coordinates:
there is one cancel-layer/mechanism decision per node and one whole-index-list
`ResidualStreamMap.free`. `D` is a bank-level compiler decision. `P` is common
for the supported single-node output because fresh allocation and `reassign`
both operate on the output's whole ordered index list. A future coordinate-wise
transfer would require a slice-wise generalization.

An allocation is a function `col` from leases to `C` satisfying:

1. **Forced endpoints:** `col(I_k) = col(O_k) = B[k]`.
2. **No overlap:** if two distinct leases' intervals overlap, their columns
   differ.
3. **No migration:** `col(q)` is constant on all of `[s(q), e(q))`.
4. **Transfer preservation:** already enforced by contraction; all strands in
   one transfer chain are one lease.
5. **Ordered coordinates:** embedding/output coordinate `k` uses the same
   `B[k]`, not merely the same unordered set of columns.

The bank `B` itself may either be selected by the allocator or preselected
from the embedding's initial allocation. Those formulations are equivalent
for intermediate-allocation feasibility because all columns in `C` are
otherwise symmetric. In the compiler, the natural implementation is to save
the embedding's ordered initial columns and call that tuple `B`.

Define the general bank

```
G = C \\ set(B),       g = |G| = d - b.
```

A non-endpoint lease may use a tied-bank column only if its entire lifetime
fits in the scratch window:

```
s(q) >= D  and  e(q) <= P.
```

Endpoint equality is interpreted using the executor's real availability
semantics. For example, a cancellation at attention event `A(L)` may count as
`e(q) = A(L)` for a supported atomic handoff, while a current MLP cancellation
in layer `L` is not available to another MLP allocation in that layer. When
`P < W`, claiming `B` means allocating the root of the output ownership chain
at `P`; there is no new physical allocation at logical output time `W`.

Call a lease **bank-eligible** when it satisfies the window condition. Call it
**forced-general** otherwise. Forced-general leases must use `G`.

The exact fixed-schedule decision problem is therefore:

> Can the interval leases be colored by the `d` physical columns so that
> forced-general leases use only the `g` general columns and the embedding and
> output receive their prescribed tied-bank columns?

## 6. What aggregate residual capacity does and does not prove

Let

```
A(t) = number of all active physical demands at time t,
       including the initial bank tenure and final output leases
F(t) = number of active forced-general unit leases at time t.
```

The current CP-SAT residual cumulative enforces the analogue of

```
A(t) <= d                                      (1)
```

at its modeled time resolution. Tied allocation additionally requires

```
F(t) <= g = d - b.                             (2)
```

Condition (2) does not follow from (1).

### Counterexample 1: aggregate-feasible but tied-infeasible

Let `d = 2`, `b = 1`, and therefore `g = 1`. Let the initial tenure own `B`
until `D = 1`, and let a fresh output claim `B` at `P = W = 4`. Add two unit
leases:

```
L = [0, 3)     # overlaps the embedding era, so L is forced into G
R = [2, 5)     # overlaps the output era, so R is forced into G
```

The maximum total occupancy is two, so (1) holds. But on `[2, 3)`, both `L`
and `R` are active and both require the sole column in `G`. Thus `F(t) = 2 > 1`
and no tied assignment exists.

This is the precise kind of schedule CP-SAT can currently admit: it sees two
active values fitting in two residual columns, but does not know that both are
forbidden from the tied column for different boundary reasons.

The example also identifies the dangerous shape:

- a **left-crosser**, already live while the embedding owns `B`, persists into
  the scratch window; and
- a **right-crosser**, born during the scratch window, persists after the
  output needs `B`.

When enough left- and right-crossers overlap, they exceed `G` even though
total occupancy does not exceed `C`.

Pointwise conditions (1) and (2) are plainly necessary, but they are not
sufficient either.

### Counterexample 2: even both pointwise bounds are insufficient

Let `d = 3`, `b = 1`, and `g = 2`, with `D = 0` and a fresh output claim at
`P = W = 10`. Consider:

```
L1 = L2 = [-1, 2)    # two left-crossers; both forced into G
R1 = R2 = [8, 11)    # two right-crossers; both forced into G
X       = [0, 6)     # bank-eligible
Y       = [4, 10)    # bank-eligible
```

At the left end, `L1` and `L2` fill `G`, so `X` is forced into the sole
column in `B`. At the right end, `R1` and `R2` fill `G`, so `Y` is also forced
into `B`. But `X` and `Y` overlap on `[4, 6)`, so they cannot both use that
column. Nevertheless, total active demand never exceeds three and active
forced-general demand never exceeds two. Thus both (1) and (2) hold.

This rules out a tempting but incorrect shortcut: adding only a cumulative
for forced-general width is not a complete solution to the general
precolored allocation problem. We instead use a structural schedule rule
that eliminates new right-crossers.

## 7. No late survivors

Define the **no-late-survivor rule**:

> Every physical lease that starts inside the scratch window is itself
> physically released no later than the output-bank claim `P`.

Formally,

```
for every q in Q:
    D <= s(q) < P  implies  e(q) <= P.              (NLS)
```

Equivalently, no lease born while `B` is available as scratch becomes a
right-crosser. Leases born at or after `P` are not covered: the final output
lease already occupies `B`, so they are born directly into `G`.

This is weaker than **terminal cleanliness**,

```
for every q in Q: e(q) <= P.                        (TC)
```

which cancels every intermediate lease by `P`. A lease that began before `D`
was necessarily allocated in `G`, so it may survive beyond `P` without
obstructing `B`. Requiring its cancellation would spend compute capacity for
no tied-allocation benefit.

NLS is also a statement about transfer-contracted physical leases, not merely
logical node births. If a free `Add` born in `[D, P)` inherits a lease that
started before `D`, that continuing lease remains safely in `G` and is not a
late survivor in the sense above.

The direct requirement proposed for the allocator is weaker still: only a
lease actually assigned to `B` must end by `P`. That condition is necessary
but circular when scheduling and allocation are separate—the allocator does
not know which leases can safely take `B` until it solves the constrained
coloring problem. NLS removes the circularity by making every genuinely new
lease in the scratch window safe for either bank.

## 8. Lemmas

### Lemma 1: coordinate decomposition

If values may occupy arbitrary noncontiguous columns and every operation's
weight writer records an ordered source/target index per coordinate, then a
width-`w` residual allocation is equivalent to `w` unit-column allocations
with common birth/release constraints.

**Proof.** Any width-`w` allocation yields one unit allocation per ordered
coordinate. Conversely, the ordered list of `w` distinct unit allocations is
exactly the index list consumed by the weight writer. Physical adjacency is
never consulted. QED.

This lemma is what lets the proof reason about unit intervals rather than
contiguous blocks. It would be false for an allocator requiring contiguous
ranges.

### Lemma 2: transfer contraction

Contracting every exact-column ownership transfer into one lease preserves
physical allocation feasibility.

**Proof.** In any valid allocation, both sides of a transfer edge must use the
same column and their lifetimes meet without a free boundary, so assigning the
contracted lease that column preserves the allocation. Conversely, expanding
a contracted lease assigns the same column to each member of the chain, which
satisfies every transfer edge. Non-transfer overlaps are unchanged. QED.

### Lemma 3: clean release is sufficient for overwrite

Suppose a lease occupying column `c` ends at event `t`, its scheduled release
emits the additive update `-x_c`, and a new value beginning at `t` emits
`+y_c`. If the executor defines both operations in the same state transition
and has captured every required read before that transition, then the column
contains `y_c` afterward:

```
x_c + (-x_c) + y_c = y_c.
```

**Proof.** Directly by the additive residual update. QED.

This lemma establishes semantic safety, not resource feasibility. The cancel
and writer still consume attention-head or MLP-slot capacity and must both be
charged by the schedule. It also does not grant same-event reuse to an
executor that allocates/frees at a different phase. The implementation's
availability relation decides whether the intervals may meet at `t`.

### Lemma 4: pre-release leases fit in the general bank

Assume aggregate capacity (1), including the width-`b` initial bank tenure
through `D`. While that tenure owns or withholds `B`, all other leases can be
allocated in `G`. In particular, every left-crosser is already in `G` when the
bank is released at `D`.

**Proof.** At every `t < D`, the initial tenure occupies or withholds `b`
columns. Thus at most `d - b = g` other unit leases are active. Process their
births in time order and assign any free column in `G`. If none were free at
a birth, the new lease plus the bank tenure and the `g` existing general
leases would make total demand exceed `d`, contradicting (1). Leases crossing
`D` retain the general columns they were assigned. QED.

### Lemma 5: under NLS, no forced-general lease begins inside the scratch
window

Assume (NLS). Every lease beginning in `[D, P)` is bank-eligible.

**Proof.** Its start satisfies `D <= s(q) < P` by hypothesis. NLS gives
`e(q) <= P`. Those are exactly the two bank-eligibility conditions. QED.

Thus, during `[D, P)`, the only forced-general leases are left-crossers that
were already allocated before `D`; that set can only shrink.

### Lemma 6: greedy extension through the scratch window

Assume:

1. all leases beginning before `D` have been validly allocated;
2. aggregate capacity (1) holds at every allocation event; and
3. no forced-general lease begins inside `[D, P)`.

Then the allocation can be extended through `[D, P)` by assigning every newly
beginning unit lease any currently free column.

**Proof.** At a new lease's start `t`, all previously assigned active leases
occupy distinct columns. If no column were free, they would already occupy all
`d` columns, and adding the new lease would make `A(t) > d`, contradicting
(1). Thus a free column exists. Because no newly beginning lease is
forced-general, either a free `B` column or a free `G` column is legal. Assign
one and continue in start-time order. Ending leases only increase the free
set. QED.

For a weighted interval, apply the same argument one coordinate at a time.
Aggregate capacity guarantees at least its width in total free columns at its
birth.

### Lemma 7: NLS frees the entire tied bank at `P`

Under (NLS), no intermediate lease occupies a tied-bank column at `P`.

**Proof.** A lease beginning before `D` could not have been assigned to `B`,
because the initial tenure occupied `B` then. Every lease beginning inside
`[D, P)` ends at or before `P` by NLS. A lease beginning at or after `P` is not
active before the output claim. Thus no intermediate lease occupies `B` at
`P`. QED.

### Theorem 1: aggregate feasibility plus NLS is sufficient
for tied allocation

Assume:

1. the schedule's residual occupancy, including the withheld width `b` on
   `[C_E, D)`, satisfies `A(t) <= d` at every relevant execution point;
2. transfer edges have been contracted into physical leases;
3. every lease satisfies the no-late-survivor rule (NLS); and
4. interval endpoints use the executor's actual release/placement semantics.

Then there exists a physical-column assignment realizing the schedule with
both the embedding and the output in the ordered bank `B`.

**Proof.** Lemma 4 constructs the allocation through `D`, with every lease
crossing `D` in `G`. At `D`, the initial tenure releases `B`. By Lemma 5, every
lease beginning inside the scratch window is eligible for either bank. Extend
the allocation greedily to `P` using Lemma 6. Lemma 7 says all of `B` is free
at `P`, so assign the root of output coordinate `k`'s ownership lease to
`B[k]`. The lease may pass through exact transfers until the logical output is
born at `W`; contraction preserves the ordered column. For any ordinary lease
born at or after `P`, allocate in `G`: the active width-`b` output lease
occupies `B`, so aggregate capacity guarantees a free general column at each
such birth by the same greedy argument as Lemma 4. QED.

### Corollary 1: terminal cleanliness is sufficient, but unnecessary

Terminal cleanliness (TC) implies NLS and therefore also implies tied
allocation feasibility under Theorem 1. The converse is false: a pre-`D`
lease may safely remain in `G` after `P`.

### Corollary 2: the full bank is usable after `D`

Under Theorem 1's assumptions, tied allocation does not reduce residual
capacity during the scratch window. The `b` tied-bank columns enter the
ordinary free pool at `D` and may be used by intermediates until `P`.

This is the substantive difference from hold-and-handoff, which removes `b`
columns from useful capacity until `P`. Choosing `D = C_E` eliminates the held
capacity tax after the embedding cancellation; a hybrid `D > C_E` retains that
tax only on `[C_E, D)`.

### Lemma 8: column assignment adds no cancellation compute demand

Suppose every lease end in the fixed schedule corresponds to an already
scheduled physical cancellation (or an exact transfer that was contracted),
and the schedule's resource cumulatives charge that cancellation. Choosing a
physical column for the lease does not add attention-head or MLP-slot demand.

**Proof.** Cancellation cost is a function of cancellation mechanism and
number of columns, not the numeric identities of those columns. Recoloring a
lease changes only the rows/columns into which the same cancel weights are
written. QED.

This is why the proposed allocator should not invent a terminal overwrite
operation. The current owner's scheduled death cancellation is the cleanup.
The tied constraint is a deadline and placement constraint on that existing
work, not a second cancellation.

### Lemma 9: a correctly modeled compute-capacity violation cannot hide in
column allocation

Suppose CP-SAT:

1. creates a cancellation decision for every scratch-window lease that must
   be released under NLS;
2. constrains that release to be available no later than `P`;
3. charges attention cancellation against the attention cumulative and MLP
   cancellation against the MLP cumulative at the chosen event; and
4. charges the operation that starts the output ownership lease against the
   same cumulative when it shares the bank-claim event.

Then a shortage of cancellation compute capacity before the bank claim
cannot appear only during physical column assignment. It makes the CP-SAT
schedule infeasible at that depth (or forces cancellation/output to different
events).

**Proof.** The allocator performs no compute; by Lemma 8 it only assigns
column identities to work already represented in the resource cumulatives.
Any excess simultaneous demand is therefore a violation of one of those
cumulatives, independent of the selected column numbers. QED.

This lemma has a strong premise. If an input or keep-forever node has no
cancellation decision, if a deadline is expressed at integer-layer precision
that disagrees with executor availability, or if a cancellation mechanism is
not charged, then CP-SAT can still admit an unreplayable schedule. Those are
model/replay bugs, not an inherent cost of tying.

## 9. When can the overall problem really be unsatisfiable?

Under Theorem 1, physical tied-column assignment itself cannot be the source
of unsatisfiability. Failure must come from at least one violated premise.
Concretely, one of the following must be true:

1. **Ordinary residual pressure is too high.** At some event, active width
   exceeds `d`. This is already the ordinary residual cumulative failure.
2. **Required cancellation work does not fit.** Late-starting values whose
   last reads occur late cannot all be cancelled by `P` within the
   attention-head and MLP-slot budgets. A correct CP-SAT model sees this and
   either cancels other values earlier, moves the output later, chooses
   different mechanisms, or declares the chosen depth infeasible.
3. **A scratch-window intermediate is required after `P`.** Then NLS rejects
   that schedule even if the value could safely remain in `G`. For fresh
   output placement (`P = W`), no non-output value is externally observable
   afterward. For transferred output placement (`P < W`), another operand may
   legitimately remain live until `W`; this is one reason the fresh-output
   policy is simpler.
4. **A physical ownership chain crosses `P`.** A free-`Add` transfer or another
   exact-column inheritance was analyzed node-by-node rather than as a
   contracted lease, so the apparent cancellation did not actually free the
   column.
5. **The executor cannot realize the modeled endpoint order.** For example,
   the model treats an MLP-cancelled column as free for an output allocation
   in the same MLP sublayer, but replay frees it only after MLP allocations.
6. **Some required cancellation is absent from the model or replay.** Current
   graph inputs are a concrete audit point: they are excluded from MLP
   cancellation and from same-layer eager reuse, and input cancellations have
   no mechanism choice. That policy must either suffice for the NLS deadline
   or be deliberately changed.
7. **`B` intersects a permanent reservation.** This must be forbidden when the
   embedding's initial columns are selected.

Items 4--7 are correctness obligations. They should cause assertions or model
changes, not be accepted as mysterious allocation failures.

## 10. Precise compute-capacity accounting

NLS does not mean "cancel everything in the last layer." A pre-`D` lease may
remain in `G` after `P`, and a scratch-window value should be released as soon
as it is dead and resource capacity permits. The deadline merely prevents a
lease that could have entered `B` from surviving past the bank claim.

For a transformer layer `L`, current cancellation costs are:

- **Attention cancellation:** cancelled columns join the attention
  cumulative. The implementation batches columns into cancel heads; the
  model charges their column width alongside the padded demands of attention
  compute operations against the effective `n_heads * d_head` capacity.
- **MLP cancellation:** `cancel_bypass` costs two hidden slots per cancelled
  column and shares the `d_hidden` cumulative with FFNs and bypass Linears.
- **Bank-claiming computation:** the operation that starts the final output
  ownership lease at `P` pays its ordinary attention or MLP demand. For a
  fresh output this is the output operation itself (`P = W`); for a transferred
  output it is an earlier owner in the final chain.

A bank-claim bottleneck is therefore possible, but its shape must follow the
endpoint semantics above. In the common fresh-output case (`P = W`), an MLP
cancellation in the output's own layer frees only at the next layer and misses
the deadline. An MLP release required by `P` must sit in the previous MLP
sublayer or earlier, after the value's last read. The two relevant collisions
are therefore:

- `m` columns attention-cancelled in the bank-claim layer add `m` to the
  attention cumulative that may also carry the claimant's attention demand;
- `m` columns MLP-cancelled one layer earlier add `2m` hidden slots beside
  that layer's FFNs and bypass Linears.

If either sum exceeds capacity, those exact choices cannot coexist. CP-SAT
must move last consumers/cancels earlier, switch mechanisms, move `P` or `W`,
or reject the depth.

Hold-and-handoff avoids NLS cleanup pressure by choosing `D = P`; no scratch
lease can start in an empty window. It does **not** delay the embedding cancel:
`C_E` stays at its ordinary early event and pays its cancellation demand there.
The cost is instead residual capacity—the clean bank is withheld on
`[C_E, P)`.

The key separation is:

```
schedule feasibility = timing + semantic precedence + compute resources
allocation feasibility = physical column identities for that schedule
```

The theorem removes the second source of failure once NLS is enforced. It
does not manufacture compute capacity for the first.

## 11. Candidate CP-SAT contract

The proof suggests the following solver/executor contract:

1. The tied embedding is born on the ordered bank `B`.
2. It receives an ordinary scheduled cancellation at `C_E`; that cancellation
   is charged at `C_E` whether or not the bank is immediately released.
3. If `D > C_E`, the clean bank is held on `[C_E, D)` and its width `b` remains in
   the residual cumulative. At `D`, `B` enters the ordinary free pool.
4. Transfer contraction identifies the final output ownership lease and its
   root birth `P`. That root—not necessarily the logical output node—is
   preassigned to the ordered bank `B`.
5. The logical output is the sole keep-forever graph value and is reached from
   that root by zero or more exact transfers, ending at `W`.
6. Every ordinary lease starting inside `[D, P)` has a real release available
   no later than `P`.
7. Every such release has exactly one cancellation mechanism, unless its end
   is an exact transfer into a continuing lease.
8. All cancellation and computation demands participate in their existing
   resource cumulatives.
9. Residual intervals end at physical availability, not merely logical death.
10. Directed replay implements the same endpoint order used by the model.

There are two possible encodings of item 6:

- **Exact lease rule:** follow the selected free-`Add` transfers and constrain
  only leases whose first owner is born inside `[D, P)`. This matches NLS
  exactly but makes the solver reason about ownership chains.
- **Conservative node rule:** after identifying and excluding every owner in
  the final output transfer chain, require every other logical owner born at
  a time in `[D, P)` to end (by cancellation or a transfer to a successor
  subject to the same rule) no later than `P`. A free `Add` that inherits a pre-`D` lease
  is unnecessarily constrained, but the rule is local to node
  birth/cancellation variables and remains sufficient.

The conservative rule may be the better implementation if measurements show
no depth cost. The proof does not require the solver to recover every safe
exception; it only requires that a physical chain capable of entering `B`
cannot survive the bank claim.

The solver need not assign actual columns. After solve, a schedule-aware
allocator can:

1. identify the final output transfer chain and its root event `P`;
2. preassign the embedding tenure and output-chain root to ordered bank `B`;
3. process births/releases in execution order, withholding `B` until `D` and
   then returning it to the ordinary free set;
4. allocate ordinary leases from all currently legal free columns;
5. while processing births in `[D, P)`, assert the lease ends by `P` before
   allowing it into `B`; pre-`D` left-crossers and post-`P` births stay in `G`;
6. immediately before `P`, execute every scheduled cancellation needed to
   clean the current owners of `B`, even if unrelated free columns exist;
7. at `P`, assert all of `B` is clean/free and place the output-chain root
   there; and
8. preserve those ordered columns through every transfer to the logical
   output at `W`.

Steps 5–7 are defensive replay obligations, not additional cancel work. They
catch mismatches between schedule metadata, selected columns, and replay.
Existing generic allocation is insufficient here: it triggers self-consumer
reuse only when it runs out of arbitrary columns and frees at most one chosen
input. A tied handoff must target every current owner of `B` regardless of the
free count.

Section 12 works out the common fresh-output case (`P = W`) under the
production pinned-cancel model, then lists the exceptions that prevent the
result from being treated as automatic.

## 12. What the production pinned-cancel model actually implies

This section maps the formal events onto the current code. It deliberately
does not say NLS is automatic.

Under the production model (`_pin_cancels`, the default for `optimize > 0`),
every ordinary schedulable node's cancel layer is equality-pinned to its
earliest legal value for the selected mechanism
(`cpsat_scheduler.py:1247–1279`):

```
pin_attn = max(layer[n] + 1,
               layer[c] + 1 - is_attn[c] for non-Add consumers,
               layer[A] + is_free[A]      for Add consumers)

pin_mlp  = max(layer[n] + 1,
               layer[c]                   for non-Add consumers,
               layer[A] + is_free[A]      for Add consumers)
```

An attention cancel is physically available at its cancel layer. An MLP
cancel is available one layer later. The mechanism choice remains free, so an
earliest cancel *within a chosen mechanism* is not automatically the earliest
physical release across mechanisms.

Ignoring the node-birth floor when it is earlier, the important last-consumer
cases are:

| owner / final consumer | attention release | MLP release |
|---|---:|---:|
| ordinary node, attention-routed non-`Add` consumer at `L` | `L` | `L+1` |
| ordinary node, MLP-routed non-`Add` consumer at `L` | `L+1` | `L+1` |
| addend of fresh `compute_add` at `L` | `L` | `L+1` |
| non-inherited addend of free `add_into` at `L` | `L+1` | `L+2` |
| graph input consumed at `L` | `L+1` | unavailable |

For the inherited addend of a free `Add`, there is no physical release at the
`Add`: ownership transfers into the `Add` output. That path must be handled as
one contracted lease, irrespective of the model's existing per-node phantom
cancel charge.

The table also has a birth-floor qualification. A node cannot cancel in its
own birth layer (`cancel >= layer[n] + 1`), so a value born in the layer just
before `P` may require an attention mechanism to be physically free by `P`
even if its last semantic read is earlier.

### 12.1 Fresh output placement (`P = W`)

For a fresh output, the output operation allocates `B` at `W`. The cases are:

- **Attention-routed, ordinary non-input output.** An ordinary operand whose
  only binding final consumer is the output can be released at `W`, but only
  with an attention cancellation. CP-SAT must constrain the mechanism or post
  the physical-release deadline; the pin alone may select MLP and miss it.
- **Graph-input operand.** Inputs have no same-layer discount or MLP choice;
  their release is consumer layer + 1. Ordinary inputs began before `D` in
  `G`, so this does not constrain `B`. The tied embedding is different: if the
  bank-claiming operation directly consumes it, the current pin gives
  `C_E > P` no matter which layer contains that operation. Moving the operation
  later does not change the gap. The compiler needs a special same-event input
  handoff, or an earlier copied value that becomes the later operand.
- **MLP-routed output.** A direct operand read by the MLP cannot be physically
  released before that MLP allocation with current machinery. Attention
  cancellation would happen before the read; MLP cancellation becomes
  available afterward. The choices are: place the output through attention,
  keep `D = P` so ordinary operands remain in `G`, otherwise ensure each
  binding operand lease began before `D`, or implement and charge an atomic
  MLP handoff. A direct tied-embedding operand still needs the input-specific
  solution above.
- **Fresh `Add` output.** `Add` normally becomes free `add_into` whenever an
  addend is dead, so fresh tied placement is not automatic. A simple policy is
  to force the tied output to the `compute_add` realization, model its larger
  attention demand, and allocate it into `B`.

Even in the first case, `allocate_at` is not the only replay change. `B` may be
split among several owners, and the existing self-consumer reuse path selects
at most one dying input and runs only after arbitrary allocation fails. Replay
must instead collect and execute every deadline-assigned cancellation covering
`B`, then place the output there regardless of unrelated free columns.

### 12.2 Transferred output placement (`P < W`)

For a free-`Add` output, replay performs
`reassign(dead_addend, output)` at `W`. The final ownership lease therefore
starts at an earlier root `P`; that root must be allocated into `B` when it is
born. The other addend is still read at `W` and may legitimately remain in
`G` after `P`, which can violate the deliberately strong NLS rule if that
addend lease started inside `[D, P)` (a lease born at or after `P` is already
forced into `G` and is not constrained by NLS).

There are two straightforward policies:

1. support transferred output leases explicitly—identify their roots before
   allocation, preassign those roots to `B`, and accept the additional NLS
   restrictions; or
2. force fresh output realization and use the simpler `P = W` contract.

The second is likely simpler unless measurements show its extra `compute_add`
demand binds.

### 12.3 Embedding cancellation versus bank release

The tied embedding is a graph input. Its pinned cancellation `C_E` is the
uniform gap-1 value `max(layer[consumer] + 1)`, with attention as its only
mechanism. That is independent of the compiler's bank-release choice `D`.

- `D = C_E` gives the no-hold design and maximizes the scratch window.
- `C_E < D = P` gives true hold-and-handoff: the bank is zeroed early, charged
  at `C_E`, then withheld cleanly until the output-chain claim.
- Allowing `C_E <= D <= P` creates a hybrid choice. The residual cumulative must
  count the held width `b` on `[C_E, D)`, and NLS applies only to leases born
  inside `[D, P)`.

This variable-`D` model genuinely contains hold and no-hold schedules. Merely
unpinning and delaying the embedding cancellation does not: it moves cancel
compute to a later layer and can be worse than early-cancel hold-and-handoff.

### 12.4 Historical staging analysis (not current implementation scope)

The theorem describes a broad design space. The current
`tied_embeddings_plan.md` stops at step 1 below regardless of the measurement;
steps 3–6 would require a separate design decision and plan. The earlier
staging analysis was:

1. implement and measure early-cancel hold-and-handoff (`C_E < D = P`);
2. stop there if obligation 0 shows no depth or width-floor cost;
3. if no-hold is needed, initially require fresh output placement (`P = W`),
   force a final `Add` to `compute_add`, and reject transferred output roots;
4. expose only two bank-release choices—`D = C_E` (full scratch) or `D = P`
   (hold)—rather than a continuous hybrid;
5. use no-hold only when the physical-release constraints, mechanism choices,
   and targeted replay handoff are satisfiable; otherwise use the hold branch;
6. leave atomic MLP handoff and transferred-output support for evidence-backed
   follow-up work.

This policy keeps the proof useful without making the first implementation
solve every precolored-interval variant. It still needs an explicit decision
for a bank claimant that directly consumes the tied embedding; neither branch
fixes the current input gap-one rule by itself.

## 13. Current implementation gaps to audit

The formal argument is conditional until these are resolved against the code:

### G1. Define `P` at sublayer precision

For fresh output placement, `node_to_layer` plus `node_to_routing` identifies
`P = W`. For a transferred output, the compiler must first identify the root
of the final ownership chain and use that root's birth as `P`. In both cases,
the deadline must use executor allocation-availability order; equal integer
layer numbers are insufficient.

We need one helper used by model validation and replay that answers:

```
is_release_available_before_claim(cancel_layer, cancel_mech,
                                  claim_layer, claim_route,
                                  owner_class, consumer_class)
```

Its truth table should be documented and tested directly. It must distinguish
ordinary nodes, graph inputs, fresh `compute_add`, free `add_into`, and exact
transfers. §12 gives the expected rows.

### G2. Make the embedding and every scratch-window owner cancellable

The tied embedding itself is an input node. Current inputs never receive an
MLP cancel (no mechanism-choice variable is built for them), are excluded
from same-layer dying-input reuse, and are not cancelled at all when they
feed a terminal `Concatenate` (keep-forever). A freeable input's pin is the
uniform gap-1 bound — cancel at max consumer layer + 1, with no attention
same-layer discount (cpsat_scheduler.py:1128–1139) — which is what fixes
the earliest cancellation `C_E`. The compiler separately chooses `D >= C_E`.
Other ordinary inputs begin before `D` and are already confined to `G`, so
NLS does not require cancelling them by `P`.

For a bank claimant that directly consumes the tied embedding, the existing
pin gives `C_E = layer[claimant] + 1`. A constraint `C_E <= P` is therefore
infeasible when `P` is that claimant's event. Moving the claimant one layer
later moves both sides and does not repair the inequality. The current
execution plan's suggestion that the solver can always add a layer must not be
applied to this direct-dependency case.

Every logical owner born inside `[D, P)` that is covered by the chosen NLS
encoding must have a modeled cancellation or a modeled transfer successor.
The rule cannot silently omit an owner class that can receive scratch columns.

The const-one seed is separate: it is a compiler-owned lease (allocated
once at compile start and never freed — held via `allocate`, not
`reserve`, though the model folds it into the permanent reservation) and it
never receives an ordinary scheduler cancellation. Under the agreed P2 design,
the compiler appends an internal final-layer write that adds `-1` to this
column, using the output bias when present or the reserved constant lane when
`bias=False`. It is not an unembed/final-norm mask and it must also work when
`rms_norm=False`. This write must not be confused with freeing tied data-bank
columns for P1.

### G3. Identify transfer leases from schedule decisions

For every free `Add`, replay chooses a dead addend whose columns are
`reassign`ed. The NLS check must follow the selected ownership chain to
its final physical release. Checking each logical node independently is
insufficient.

If the final output is such a transfer, the compiler must either identify and
preassign the chain root at `P`, or force a fresh output realization. A final
`allocate_at(output, B)` cannot repair a chain that was born in `G`.

### G4. Remove accidental keep-forever values

The compile loop stops as soon as the output is computed. A value that is
semantically dead at that point may remain allocated simply because there is
no later scheduling iteration. That is harmless for a pre-`D` lease already
in `G`, but not for a scratch-window lease subject to NLS. Any required cleanup must
be part of producing the final scheduled layer(s), not an after-the-loop
allocator wish. For fresh placement, a cancel assigned to layer `L_W + 1`
sits in a layer the loop never runs. For transferred placement (`P < W`), a
cancel may execute before loop termination yet still be too late because `B`
was already claimed at `P`.

Terminal `Concatenate` leaves are currently modeled as keep-forever. They are
outside this note's single-output scope and should remain rejected for tied
allocation until given a slice-wise formalization.

### G5. Prove model/replay parity for cancellation cost

The CP-SAT model already charges ordinary scheduled attention cancellations
and MLP `cancel_bypass` operations. Tests must still establish that every
NLS-deadline cancellation appears exactly once in replay, uses the
modeled sublayer, and consumes the modeled capacity. No special "tied cleanup"
demand should be added on top.

### G6. Replay cancel-deferral must respect the deadline

Replay today silently defers a cancel to a later layer when the attention
head budget or the MLP slot budget is exhausted at the assigned layer (the
`<=` in `_find_dead_nodes` re-surfacing the node, scheduler.py:1610–1643;
the whole-node `continue` in `cancel_bypass` emission,
scheduler.py:1059–1060). It never splits a node's cancel, but it can slide
one past `P`. For an NLS-deadline cancel this silently produces exactly the
unreplayable schedule Lemma 9's caveat names. Deferral of an NLS-deadline
cancel must become a hard replay error, or the model must provably reserve
enough capacity that deferral never triggers on one.

### G7. Replay must target every owner of `B`

At `P`, `B` may be split across multiple node allocations. Existing promotion
and self-consumer reuse are driven by global free-column pressure, not by a
required destination set, and self-consumer reuse returns only one input.
Tied replay needs a targeted handoff:

1. enumerate every current owner covering a column of `B`;
2. verify its scheduled physical release is available by `P`;
3. emit/collect every required cancellation within the modeled capacity;
4. free exactly those columns; and
5. place the output-chain root in ordered `B`.

This is still existing scheduled cancellation work, not a terminal overwrite.

### G8. Heuristic and fallback parity

The eager scheduler supplies warm-start hints and is also the fallback when
CP-SAT finds no solution. A no-hold artifact cannot silently fall back to an
allocator that ignores NLS, `D`, `P`, or targeted bank cleanup. Either the
heuristic implements the same contract, or tied compilation must fail loudly
instead of emitting an untied or physically invalid artifact.

## 14. Stronger versus weaker rules

NLS is a sufficient rule chosen for proof simplicity. It is not the
mathematically weakest rule. Terminal cleanliness is a still stronger rule
and is not required.

A weaker allocator could allow intermediate leases to survive beyond `P` as
long as they are placed in `G`. It would then need to prevent all such
right-crossers from entering `B` and establish that the remaining
forced-general intervals fit in `g` columns. That may preserve schedules that
NLS rejects, but it reintroduces a second constrained allocation problem and
a possible post-solve failure.

The tradeoff is therefore:

- **NLS:** a simple theorem; only scratch-window leases receive the bank-claim
  deadline, and its capacity consequences are visible to CP-SAT as ordinary
  cancellation scheduling.
- **General precolored allocation:** potentially admits more schedules; needs
  stronger allocation machinery or more column-aware solver constraints.

For fresh output placement (`P = W`), NLS is semantically natural because only
the output is externally required afterward. For transferred placement
(`P < W`), NLS can constrain ordinary values still needed between `P` and `W`;
that cost is another reason to prefer a fresh-output policy unless measurement
justifies transfer support. Any depth increase from NLS is an honest, visible
compute-scheduling bottleneck, not a reason to add an unmodeled overwrite.
Terminal cleanliness should not be substituted unless some separate invariant
requires every final scratch column to be zero.

## 15. Proof obligations for designs in the broader formal space

The held-only execution plan has its own concrete acceptance checklist. The
following broader list is retained so a future design that exposes the bank as
scratch cannot accidentally rely on the aggregate theorem alone. NLS-specific
items are not instructions to implement NLS in the current plan.

0. **Bindingness gate:** implement/solve the minimal hold baseline and measure
   whether its capacity tax changes layer count or width floor. Peak occupancy
   over `[C_E, P)` is a useful diagnostic but is not, by itself, proof that no
   alternative schedule is lost. The DOOM flagship (600+ of 8192 columns) is
   the real exposure. If hold does not bind, its smaller implementation wins.
1. **Lease completeness:** enumerate every way physical column ownership can
   continue without a cancel/free boundary; today, free `Add.reassign` is the
   known case.
2. **Event contract:** represent `C_E`, `D`, `P`, and `W` separately and enforce
   `C_E <= D <= P <= W`. Count the held width on `[C_E, D)`.
3. **Endpoint table:** test the §12 availability table, including ordinary
   nodes, graph inputs, fresh Adds, free Adds, birth-floor cases, and both
   cancellation mechanisms.
4. **Output-realization policy:** decide whether tied output must be fresh. If
   transferred output is supported, identify and preassign its lease root at
   `P`; otherwise force fresh `compute_add` where needed and charge it.
5. **Output-route policy:** for fresh MLP-routed output, choose among attention
   routing, `D = P` hold, operands proven pre-`D`, or an implemented and
   charged atomic MLP handoff.
6. **Input handoff policy:** handle a graph input—especially the tied
   embedding—used by the bank-claiming operation. Current gap-1 input pins and
   replay exclusions do not permit the same-event attention handoff.
7. **NLS semantic safety:** prove every ordinary lease beginning inside
   `[D, P)` may be released by `P` after its last read.
8. **NLS model completeness:** give every such lease a cancellation or
   transfer end, constrain its physical availability by `P`, and constrain
   the mechanism when only one mechanism meets the deadline.
9. **Resource parity:** show the cancellation is charged once, and only once,
   in the same capacity pool used by replay.
10. **Residual parity:** show CP-SAT's half-open residual interval matches the
   allocator's actual free point for both cancellation mechanisms.
11. **Constructive allocator:** implement the greedy allocation from Theorem
   1, with the initial tenure and output-chain root preassigned to `B`.
12. **Targeted replay:** cancel every scheduled owner of `B` by `P`, independent
   of unrelated free capacity; forbid deadline deferral.
13. **Heuristic/fallback parity:** enforce the same contract outside directed
   CP-SAT replay or fail loudly.
14. **Ordered tie assertion:** at export, assert element-wise equality of the
   embedding and output index lists.
15. **Adversarial allocation tests:** encode both counterexamples from §6; the
   second must demonstrate why aggregate plus forced-general cumulative is
   still insufficient.
16. **Capacity test:** construct a graph where NLS-required cancellation plus
    the bank-claiming writer exceeds a sublayer's capacity and verify CP-SAT
    moves work/claim or reports infeasibility rather than failing in replay.
17. **Seed cleanup:** clear compiler literal-seed columns in the final layer in
    both norm-on and norm-off modes; do not mask or accept a uniform offset.

If these obligations hold, P1 requires no column-identity variables in CP-SAT
and no second cancellation. The current execution plan fixes `D = P` and
exposes none of `[C_E, P)` as scratch. `D = C_E` and a modeled hybrid remain
formal alternatives only.

## 16. Boundary with the seed-column problem

The tied ONNX table also contains compiler seed constants outside the compact
embedding bank `B`. They must be zero at unembed time because there can be no
unembed mask. This is P2, not a counterexample to the allocation theorem:

- pinned RMS columns remain reserved until final normalization and can be
  zeroed at the final norm by setting only their per-column final gain to zero;
  they cannot be cancelled earlier because the final norm still needs them;
- the compile-internal const-one/literal-seed columns must instead be cleared
  by a compiler-internal write in the final transformer layer. With physical
  bias, append `-seed_value` to that column through the existing MLP output
  bias. Without physical bias, use the already-reserved constant lane and
  `BiasFold` output path. This consumes no new ordinary hidden slot and works
  with both `rms_norm=True` and `rms_norm=False`;
- no `+1` norm-off offset is accepted, and neither an unembed mask nor a
  final-norm mask is used for the literal seed;
- neither operation changes where the `b` learned embedding/output coordinates
  are allocated.

After scheduling a layer reveals that it computes the logical output, append
the literal clear to that layer's MLP op list before weight writing. Runtime
MLP reads still see the pre-sublayer constant, and the additive update leaves
it zero in the final state. The const-one owner can then be omitted from the
final live map. This P2 work should not be mixed into the proof that learned
tied columns can be leased as ordinary scratch between `D` and `P`.
