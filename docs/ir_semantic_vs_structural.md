# The IR is semantic, not structural

*A design observation about torchwright's intermediate representation, surfaced
while working out how the ops would be rebuilt on a gated (SwiGLU) MLP. This is
a diagnosis and an open question, not a decided plan.*

## Background: what the compiler maps, and onto what

torchwright takes a **computation graph** — a DAG of primitive nodes (`Linear`,
`ReLU`, `Add`, `Attn`, `Concatenate`, …; the planned gated-MLP rework adds
`Swish` and `Mul`, which do not exist as node types today) — and compiles it into a
transformer's weights, so the transformer run autoregressively reproduces the
graph's output. The graph is the compiler's **intermediate representation
(IR)**: the thing every op is built from and the thing the compiler lowers.

The target hardware — a transformer — has two kinds of compute per layer:

- an **attention sublayer** (query/key/value projections, softmax over
  positions, output projection), and
- an **MLP sublayer** (also called the **FFN**): a `Linear`, a nonlinearity, a
  `Linear`.

Both write their output by **adding it into the residual stream** — the running
per-position vector that flows down the layers (`x = x + sublayer(x)`).

The cost of a compiled graph is measured in this hardware: how many attention
heads, how much MLP width, how many layers deep. The compiler's job is to place
the graph's computation into as few of these as possible.

## The diagnosis

**The graph IR is purely *semantic*: it says what value to compute, not what
hardware computes it. Because all the cost lives in the hardware realization,
the IR can express any computation but can neither see nor control its cost.**

A single semantic node under-determines its hardware. The same math can be
realized by physically different, differently-priced pieces of the transformer,
and the IR gives no way to say — or even see — which one you get.

## The exemplar: `Add`

`Add(a, b)` is one node type whose cost depends **entirely on how it's
realized**, spanning the full range:

1. **Absorbed into a sublayer's output — cost zero.** Every sublayer adds its
   output into the residual stream. So an `Add` is free whenever a sublayer that
   was running anyway lands one addend on the columns already holding the other —
   either by summing two of its own FFN hidden lanes in the output projection, or
   by writing its output onto columns an earlier (now-dead) node left a value in.
   No *extra* hardware is spent.
2. **A standalone attention transport head — cost: a head.** When the addends
   are not co-produced by a convenient sublayer, torchwright emits the `Add` as a
   *rotary Δ=0 self-match attention head*: the head matches the current position
   against itself and copies the addends onto shared columns, and the sublayer's
   additive output does the summing (`compiler/forward/weight_writer.py:_write_compute_add`
   / `_write_add_into`). An attention head is a scarce per-layer resource. The
   cheapest standalone form, `add_into`, reuses a *dead* addend's columns —
   permitted only because invariant I1 forbids two *simultaneously-live* nodes
   from sharing columns — but still spends one head to bring the live addend in;
   `compute_add` spends head(s) for both.

Same math, cost from nothing to a whole head. **Nothing in the IR distinguishes
these** — the choice is made downstream, inside the scheduler, implicitly.

## Worked example: `select`

`select(cond, a, b)` returns `a` when `cond` is true (+1) and `b` when false
(−1). On a gated MLP it can be written:

```
select(cond, a, b) = Add( Mul(Swish(scale·cond)/scale,  a),
                          Mul(Swish(-scale·cond)/scale, b) )
```

This is expressible as **one FFN**: two gate lanes (`cond` into the gate
projection with ±scale, `a`/`b` into the up projection), and the `Add` is the
**output projection** summing the lanes — not an attention head. There's even a
second single-FFN realization, `select = b + indicator·(a − b)`: one gate lane
computes `indicator·(a − b)` and `b` is supplied by the FFN's residual-add,
halving the width but consuming `b`'s columns.

So the identical op is either a tidy one-FFN block *or*, if its two `Mul`s and
`Add` are left as loose graph nodes and scheduled apart, a fragmented thing —
two gate lanes plus a between-sublayer `Add` that now costs an attention head.
The difference is **purely how it was lowered**, and the semantic IR records
none of it.

## Why "there are many ways to compile a graph, and we pick one"

This follows directly. Because the semantic IR under-determines the hardware,
the map from IR to transformer is **one-to-many**. torchwright navigates that
map by fiat: each op function directly emits primitive nodes that happen to
lower a certain way, plus fuser heuristics in the scheduler (the
`Linear → ReLU → Linear` block recognizer, the dead-operand `add_into` reuse).
There is no explicit representation of the branch points, so there is no place
to *choose* a branch against a cost model. The branches are taken, not decided.

Any node type that spans multiple realizations is a place the IR hides cost:

| Node | Cheap realization | Expensive realization |
|---|---|---|
| `Add` | absorbed into a running sublayer's output (FFN projection / residual-add onto a dead node's columns) | standalone attention transport head |
| `Linear` | absorbed into a block's projection | standalone transport head |
| `Concatenate` | a free view over columns | a real data move |
| `Mul` | one gate lane of an FFN | (only realizable inside a gate) |

For each, the hidden question is identical: **does this node get absorbed into a
neighboring block, or stand alone as its own hardware?** That absorption
question *is* instruction selection — the choice of which hardware realizes each
op — and the IR does not represent its answer.

## What's actually missing: a structural level

There are two levels of description trying to be the same object:

- a **semantic level** — `Linear`, `ReLU`, `Add`, `Attn` (and the planned
  `Swish`/`Mul`); what ops are written in and what numerics are reasoned about;
  and
- a **structural level** — nodes that *are* transformer constructs: "a gated FFN
  block with these lanes and this output projection," "a self-match transport
  head," "a residual-add"; where cost is directly readable and realization is a
  choice.

torchwright has the first and not the second. The pipeline leaps straight from
the semantic graph to scheduled sublayers, and the structural facts (*these
three math nodes are one FFN*) are **reconstructed transiently inside the
scheduler** by pattern-matching — never made first-class. So "this is one FFN,
and its `Add` is free" is not something the IR states; it's something the fuser
rediscovers, and silently fails to when the pattern doesn't match.

## What one could do about it

The textbook answer is a **second IR** — a structural, lower-level one whose
nodes are transformer blocks and heads — with an explicit **instruction-selection
pass** between the semantic graph and it. Then:

- `select`'s two realizations become two explicit lowerings a cost model picks
  between;
- `Add`-in-output-projection vs. `Add`-as-head becomes a decision, not an
  accident;
- the scheduler's ordering problem — realization frozen *before* the scheduler
  sees each layer's resource pressure, so it can't move an `Add` from an
  overloaded attention sublayer into spare MLP width — dissolves, because
  realization and packing would share one representation.

## The honest tradeoff

A full second IR plus cost-driven instruction selection is a large piece of
machinery, and the search space it opens (multiple registered lowerings per op,
a cost model, co-optimized with scheduling) is real complexity to flag before
committing.

The cheaper hedge — and likely the right first move for a fixed target like
DOOM — is to **author the ops that matter as explicit block-builders**: a unit
that *is* "one gated FFN," the way today's `linear_relu_linear` is one MLP
sublayer. That buys the structural guarantee (single FFN, free `Add`, known
cost) *by construction*, at the price of committing to one realization per op
with no search. It is **manual** instruction selection instead of automated.

The `docs/ops_plain_english.md` reference is already doing exactly this by hand:
when it writes `select = Add(Mul(…), Mul(…))` and then annotates "one FFN, width
2d, `Add` in the output projection," that annotation *is* the structural IR,
written in prose because there is no formal place for it. The formulas look
semantic but are quietly asserting structure. That ambiguity is the symptom of
the missing level.

## The open question

Whether to **formalize the structural level** (a real second IR, searchable,
cost-driven) or keep **patching it by hand** with curated block-builders
(cheaper, controllable, but no search). Both are defensible; the first pays off
only if the implicit lowering leaves enough on the table to justify the
machinery, and the `square` case — where a new instruction collapsed a wide,
approximate lowering to an exact width-2 one — is evidence that the gap between
"a working lowering" and "the best lowering" can be large.
