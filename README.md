# torchwright

torchwright is a compiler that transforms computation graphs into the weights of a
transformer. It treats a transformer as a fixed computational substrate that can
be programmed: the compiler sets the weights directly, without any training, so
that a standard decoder-only transformer — causal softmax attention, rotary
position embeddings, RMSNorm, a KV cache — executes the source graph. Compiled
models are packaged, by default, as ordinary `transformers` checkpoints in the
Phi-3 architecture: `AutoModelForCausalLM` loads them with no custom code and no
`trust_remote_code`.

## Example

The graph in `examples/binary_increment.py` parses a binary string and increments
it, carry propagation included. Compiling it produces an ordinary Hugging Face
model:

```python
from examples.binary_increment import create_network_parts, D_MODEL, D_HEAD
from torchwright import compile_hf_bundle

output_node, embedding = create_network_parts()
compile_hf_bundle(output_node, embedding, "binary_increment_hf_bundle",
                  d=D_MODEL, d_head=D_HEAD)
```

The result — safetensors, config, tokenizer — loads like any `transformers`
checkpoint:

```python
from transformers import pipeline

generate = pipeline("text-generation", model="binary_increment_hf_bundle")
print(generate("1011\n", return_full_text=False)[0]["generated_text"])
# 1100
```

The compiler scheduled this graph into a 20-layer decoder at hidden size 512 (the
`d=D_MODEL` argument). Every weight was computed from the source graph; nothing
was trained. Compiling this example took under ten seconds on a laptop CPU.

## How it works

A torchwright program is a computation graph: ordinary Python wiring op calls into
a DAG of nodes, with token embeddings at the leaves and one output node at the
root. `examples/adder_1digit.py` is a 99-line worked example.

Ops come in three groups: linear ops (`add`, `subtract`, `concat`, …), attention
ops (latch a value at a marker position, read a fixed offset back,
argmax/argmin/mean over positions), and nonlinear ops (`compare`, `select`,
`multiply`, table lookups, `floor_int`, `mod_const`, …).

The nonlinear ops are built from the transformer's MLP activation function, so
choosing an activation means choosing an op library. `torchwright.ops.relu`
constructs every nonlinear op from ReLU sublayers; `torchwright.ops.swiglu`
constructs the same ops from gated SwiGLU sublayers, and is what the Phi-3
output uses. A graph imports from exactly one — the compiler rejects a graph
that mixes them.

The compiled computation lives in the residual stream — the fixed-width vector a
transformer carries between layers. Every value the graph computes owns a set of
columns there for as long as anything downstream still needs it; columns need not
be contiguous, and concatenation is logical rather than physical. A sublayer can
only *add* to the stream, and that one fact drives the compilation scheme: a new
value lands in zeroed columns (0 + new = new), a value nothing needs anymore is
cancelled by writing its negation (v + (−v) = 0), which frees its columns for
reuse, and the graph's additions are implemented on the skip connection itself,
costing no extra columns.

Every op compiles down to concrete weights — attention heads, rows of MLP
sublayers. A scheduler assigns every node to a layer: `optimize=0`, the default,
places nodes with a heuristic, and `optimize=1`–`3` hand placement to a
constraint-programming solver (CP-SAT, from Google OR-Tools) that minimizes layer
count under growing time budgets (60 to 600 seconds). Weights stream into the
artifact layer by layer, so peak memory during
compilation stays near one layer's worth regardless of depth. The same compile
can also target ONNX: `compile_to_onnx` emits the transformer as a KV-cached
ONNX artifact runnable outside `transformers`.

## Install

Python 3.10 or later.

```
pip install "torchwright[hf]"
```

Or from source:

```
git clone https://github.com/physicsrob/torchwright
cd torchwright
uv sync --extra hf
```

Core dependencies are torch, onnx, ortools, and pydantic. The `hf` extra adds
`transformers` and `safetensors` for the Hugging Face output path. Running the
ONNX output additionally needs an onnxruntime: install the `onnxruntime` package
for CPU, or the `gpu` extra for `onnxruntime-gpu`.

## Verification

torchwright checks that a compiled transformer faithfully executes its source
graph. Many ops are implemented as piecewise-linear approximations, so
correctness is measured and enforced rather than assumed, at four levels. Every
approximate op is measured against its exact-math reference, and the per-op
error bounds are committed to the repo (`docs/op_noise_data.json`); the test
suite fails if the committed numbers drift from what the code measures. The
compiler asserts its own structural invariants while compiling, each pinned by a
negative test. Assert predicates can be attached to graph nodes; they run on
exact values during reference evaluation and again on compiled values during
debug forward passes, so an assert that passes in exact math but fires compiled
pinpoints where approximation error exceeded its budget. Finally,
`torchwright.debug.probe.probe_compiled` diffs a compiled transformer
node-by-node against direct evaluation of the source graph.

Per-op bounds are measured on each op's intended input ranges and are not
additive through chains of ops; chain-level questions go to `probe_compiled`.

## Examples

Twelve example graphs live in `examples/`:

- `adder_1digit` — parses `"A+B\n"`, single-digit addition; the smallest
  complete program.
- `adder` — 3-digit addition: each digit pair goes through a lookup table and
  carries propagate right-to-left, pencil-and-paper style.
- `adder_v2` — the same adder a different way: digits become numbers, one add,
  numbers become digits.
- `binary_increment` — the example above.
- `caesar_cipher` — shift cipher; the shift amount is a runtime input digit.
- `sort_digits_v1` — sorts a digit string ascending, one digit per
  autoregressive step.
- `range_printer` — a two-level loop: iterate items, and for each item iterate a
  range of values.
- `fibonacci` — autoregressive: each number is computed from the model's own
  previously emitted tokens.
- `calculator_simple` — `+`, `-`, `*` on multi-digit integers, one lookup table
  and one fold at a time.
- `calculator_advanced` — the same functions at logarithmic depth, the way a
  hardware multiplier does it.
- `calculator_memorize` — computes nothing: every answer is a memorized fact.
- `calculator_scratchpad` — streams the serial carry/borrow work out as visible
  tokens and reads it back, so compiled depth stays flat as operand length grows.

Examples that define `create_network_parts()` compile to a bundle with
`uv run python -m examples.compile <name>`.

## Limitations

- Only fp32 is supported.
- The KV cache is static: sequence length is fixed at export (`max_seq_len`,
  default 512), and overrunning it raises.
- With RMSNorm on (the default), supported hidden sizes (`d`) are any multiple
  of 1024 up to 16384, or any power of two; other widths raise. See
  `docs/rms_norm_dmodel.md`.

## Further reading

- `docs/optimization_guide.md` — reducing layer count and parameter cost when
  compiling a graph.
- `docs/cpsat_scheduler.md` — the constraint-programming scheduler: model,
  objective, warm starts.
- `docs/affine_bounds.md` — how value bounds propagate through the graph.
- `docs/numerical_noise.md` — measured per-op approximation error (generated
  from `docs/op_noise_data.json`).

## Development

`torchwright/` holds the package (`graph`, `ops`, `compiler`, `debug`);
`examples/`, `tests/`, and `docs/` are what they say. `make lint` (ruff + mypy)
and `make test-local FILE=tests/...` (single-file pytest) run anywhere. The full
suite, `make test`, shards across GPU containers on Modal and needs a Modal
account. `make measure-noise` regenerates the committed per-op noise data; two
tests pin it to the code.

CI runs on every push and pull request: `make lint`, plus a smoke that compiles
an example on CPU and checks its generated output. The full suite additionally
runs on CPU weekly. Releases are tagged (`v*`) and published to PyPI from CI;
`docs/releasing.md` has the procedure.

## License

Apache-2.0.
