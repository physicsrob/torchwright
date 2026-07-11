# TorchWright

TorchWright is a compiler that transforms computation graphs into transformer
neural network weights. You define what you want the transformer to compute
using high-level operations (arithmetic, logic, attention, table lookups), and
the compiler produces actual weight matrices that run in a standard transformer
architecture.

The key insight is that transformers are not just learnable architectures --
they are a fixed computational substrate that can be *programmed* by setting
weights directly, without any training.


## Getting Started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync          # install dependencies
make test        # run tests
make lint        # run black + mypy
```


## Examples

TorchWright has been used to compile several non-trivial programs into
transformer weights:

**Multi-digit adder** (`examples/adder.py`) -- Parses "123+456=" and outputs
"579" autoregressively. Digit-by-digit addition with carry propagation, exactly
like pencil-and-paper arithmetic. The 3-digit version compiles to a 29-layer
transformer with d=1024.

**Calculator** (`examples/calculator_simple.py`) -- Supports +, -, * on positive
integers up to 3 digits. Subtraction handles negative results; multiplication
uses long multiplication with partial product rows. Compiles to 38 layers with
d=2048.

Both examples can be compiled to ONNX for portable inference:

```bash
make compile     # produces adder.onnx and calculator_simple.onnx
```

Hugging Face is a sibling compile target and does not use ONNX as an
intermediate artifact:

```python
from torchwright.compiler import compile_to_hf, compile_hf_bundle

model = compile_to_hf(output_node, embedding, d=1024)
compile_hf_bundle(output_node, embedding, "hf_bundle", d=1024)
```

The default is a stock `Phi3ForCausalLM` contract: SwiGLU graph, biasless
projections, RMSNorm, no custom code, and no `trust_remote_code`. A legacy
ReLU graph must opt in explicitly with `architecture="custom"`, which produces
`TorchwrightCustomForCausalLM`. `compile_hf_bundle` writes sharded safetensors
one layer at a time for large-model publishing. Publication is transactional:
the destination is replaced only after the staged config, tensor manifest, and
tokenizer validate. Token artifacts currently require one graph `Embedding`,
and the supplied `embedding` argument must be that exact node.


## Key Concepts

### Nodes

A node represents a computed value in the computation graph. It *is* its
output -- a `Linear` node that produces 10 floats *is* those 10 floats.
Nodes know their size (number of output floats) and their inputs (other
nodes they depend on).

For example:

```
input_node         →  10 floats  (no inputs -- comes from the user)
linear_result      →  10 floats  (input: input_node)
relu_result        →  10 floats  (input: linear_result)
```

The computation graph is just nodes pointing to their inputs. The compiler's
job is to figure out how to pack all these values into a transformer's residual
stream and set the weights so the transformer computes them correctly.

### States

A state is a label for a point in the transformer's forward pass. It names a
specific moment -- "the residual stream right after attention finishes" or "the
residual stream right before the MLP sublayer." It is not the vector itself, and
it is not a slice of the vector. It's a label for *when*.

A simple 1-layer transformer has states at each boundary:

```
→ attn_in_state    (residual stream before attention)
    [attention runs]
→ mlp_in_state     (residual stream after attention, before MLP)
    [MLP runs]
→ mlp_out_state    (residual stream after MLP)
```

At each state, some set of nodes are **alive** -- their values are sitting in
the residual stream, either because they were just computed or because they're
being passed through for later use. The compiler tracks which nodes are alive at
each state, and assigns each alive node to specific columns in the residual
stream at that state.


## Glossary

TorchWright is closely related to Anthropic's mechanistic-interpretability
"circuits" work, so we try to use the same vocabulary where it exists.
A few terms are TorchWright-specific (`column`, `state`, `chain`, `slot`)
because they describe the compiler's internals.

**Architecture (transformer-side):**

- **residual stream** — The fixed-width vector that carries all information
  between transformer layers. Same meaning as in the circuits literature.
- **column** — One dimension of the residual stream. The "whiteboard with
  numbered columns" metaphor below. (Anthropic would just say "dimension";
  we use *column* because it pairs naturally with allocation.)
- **`d`** (in code) / **`d_model`** (in prose) — The width of the residual
  stream. We use the short name in code for brevity.
- **layer** — One full transformer block: attention sublayer + MLP sublayer.
- **attention sublayer** — Multi-head attention + skip connection.
- **MLP sublayer** — `Linear → ReLU → Linear` + skip connection. We
  previously called this an "FFN sublayer"; we now use *MLP* to align with
  the circuits literature. Both names refer to the same thing.
- **attention head** — One head within multi-head attention, parameterised
  by `W_Q`, `W_K`, `W_V`, `W_O`. Same meaning as in the circuits work.
- **neuron** — One dimension of the MLP's hidden layer (the intermediate
  representation between `linear1` and `linear2`). Anthropic uses both
  *MLP neuron* and *ReLU unit* for this concept; we use **neuron**.
- **`d_hidden`** — Width of the MLP's hidden layer (number of neurons per
  MLP sublayer).

**Computation graph (TorchWright-specific):**

- **node** — A value in the computation graph. A `Linear` node *is* its
  output vector; nodes know their width and their input nodes.
- **chain** — A `Linear → ReLU → Linear` pattern in the graph. The
  compiler maps each chain to one MLP sublayer.
- **state** (`ResidualStreamState`) — A label for a specific point in the
  forward pass — e.g., "the residual stream right after attention." A
  state is *not* the vector itself; it is a name for *when* the vector
  exists.
- **residual assignment** (`ResidualAssignment`) — The mapping of graph
  nodes to residual-stream columns at each state. We previously called
  this `FeatureAssignment`; renamed to avoid colliding with the
  circuits-literature meaning of *feature* (an interpretable direction in
  activation space).
- **slot** — A scheduler-internal term for an allocated neuron in the
  MLP hidden layer.
- **dead node** — A node whose downstream consumers are all already
  computed; its columns can be reclaimed.

**A word on "feature":** in the circuits literature *feature* means an
interpretable direction in activation space, the central object of
mechanistic interpretability. We don't use the word in that sense, and
we deliberately avoid it as a synonym for "node," "column," or
"assignment" so that anyone coming from the circuits work isn't
constantly translating.


## The Residual Stream Allocation Problem

A transformer's residual stream is a fixed-width vector (say, 1024 floats) that
carries all information between layers. Think of it as a shared whiteboard with
numbered columns. At every point between layers, everything the transformer
"knows" must be written somewhere on this whiteboard.

Each value in the computation graph occupies a set of columns. For example, if
your computation involves an input (10 floats), a linear transformation result
(10 floats), a ReLU result (10 floats), and a final output (10 floats), each
of these needs its own non-overlapping columns in the residual stream.

### The rules

**No overlaps.** Two values that are alive at the same point in the network
can't share columns. If both `input` and `linear_result` exist in the residual
stream at the same time, they need different columns.

**Skip connections preserve positions.** A transformer's skip connection is just
addition: `x + f(x)`. It doesn't rearrange anything. So if `input` is at
columns [0-9] before the MLP sublayer, it's still at columns [0-9] after the skip
connection. Any value that flows through a skip connection must keep its column
assignment.

**Concatenation is logical, not physical.** When the computation graph
concatenates two values (e.g., to feed them into a Linear node), they do *not*
need to be adjacent in the residual stream. The compiler scatters the Linear
node's weight matrix to whatever columns the inputs occupy. This is a key
simplification -- it means concatenation adds no allocation constraints.

**Columns need not be contiguous.** A node's columns can be scattered anywhere
in the residual stream. The weight writer gathers and scatters via index lists,
so physical adjacency is never required. This eliminates fragmentation entirely.


## How Compilation Works

The compiler works **forward** from inputs to outputs, building the transformer
one layer at a time. At each layer, a scheduler decides what to compute using
the attention sublayer and MLP sublayer, then a weight writer sets the
corresponding weight matrices.

### The skip connection

Every sublayer in a transformer computes `out = in + f(in)`. The sublayer
output gets *added* to whatever is already in the residual stream. This single
fact determines how values enter, persist, and leave:

- **Values persist automatically.** They are the `in` that passes through
  unchanged. A value written at layer 3 is still there at layer 30 unless
  something explicitly removes it.

- **A sublayer can only add.** It cannot overwrite or rearrange columns. What
  happens depends on what is already in the target columns:
  - If the columns are **empty** (zeroed out), the new value appears:
    `0 + new = new`. This is how new computations enter the residual stream.
  - If the columns hold a **dead value** (nothing downstream needs it anymore),
    write its negation: `v + (-v) = 0`. This frees those columns for reuse.
  - If the columns hold a value you want to **add to**, write the other addend
    and the skip connection produces the sum. This is how the compiler
    implements Add nodes without using any extra columns.

### What each sublayer computes

**Attention.** Each attention head applies a weight matrix and writes to
d_head columns -- so anything that is a matrix multiply or a targeted write
to specific columns maps here. This includes:
- Attention nodes (cross-position lookups -- the thing only attention can do)
- Linear nodes without bias (a head applies the weight matrix at the
  current position)
- Column management: cancellations and additions into existing columns are
  just targeted writes with the right values

**MLP.** The MLP sublayer's internal structure is Linear → ReLU → Linear,
so it directly handles graph operations with the same shape:
- Linear → ReLU → Linear chains
- Standalone ReLU nodes
- Constants (via the output bias)
- Bias terms for Linear nodes that were computed in the attention sublayer

### Scheduling

The compiler picks what to compute at each layer based on what is ready (all
inputs alive in the residual stream) and what is most urgent (longest
dependency chain to an output). When free columns run low, the scheduler
shifts to prioritizing operations that free space -- cancellations and
additions into existing columns -- over new computations. Column assignment
is greedy: nodes get whatever columns are free when they are computed.


## The Ops Layer

The `torchwright/ops/` package provides high-level operations built
on top of the raw graph nodes. These are what examples typically use:

- **Arithmetic**: `add_const`, `negate`, `subtract`, `multiply_const`
- **Logic**: `equals_vector`, `bool_not`, `bool_all_true`, `bool_any_true`
- **Table lookups**: `map_to_table` -- maps an embedding-valued input to an
  embedding-valued output via an MLP lookup table
- **Selection**: `select` (if/else), `switch` (multi-way dispatch)
- **Sequence operations**: `output_sequence`, `remove_leading_0s`
- **Embedding arithmetic**: `sum_digits`, `sum_digit_seqs` -- digit-by-digit
  addition with carry propagation in embedding space


## Architecture

```
torchwright/
├── graph/              # Computation graph: nodes, embeddings, attention, etc.
├── ops/                # High-level operations (arithmetic, logic, tables)
└── compiler/
    ├── forward/        # The forward compiler
    │   ├── compile.py        # Main entry point (forward_compile)
    │   ├── graph_analysis.py # Topological order, consumers, critical paths
    │   ├── residual_map.py   # Greedy column allocator
    │   ├── scheduler.py      # Layer-by-layer operation scheduling
    │   └── weight_writer.py  # Writes weight matrices into transformer layers
    ├── components/     # Transformer component abstractions (attn, linear, relu)
    ├── groups/         # Sublayer and layer groupings
    ├── export.py       # ONNX export
    └── transformer.py  # HeadlessTransformer (the compiled artifact)
```
