# torchwright

torchwright is a compiler that transforms computation graphs into the weights of
a transformer. You define a computation graph in ordinary Python, and
torchwright produces the weights of a transformer that executes it. There is no
training anywhere in the pipeline. Compiled models are packaged, by default, as
ordinary `transformers` checkpoints in the Phi-3 architecture:
`AutoModelForCausalLM` loads them with no custom code and no
`trust_remote_code`.

This README covers the tool; the [intro
post](https://ood.dev/posts/torchwright-intro/) covers how it came to exist and
how the constructions were derived, from the ReLU primitives to a stock Phi-3
checkpoint.


## Example

The graph in `examples/binary_increment.py` parses a binary string and increments
it, carry propagation included. Compiling it produces an ordinary Hugging Face
checkpoint:

```python
from examples.binary_increment import create_network_parts, D_MODEL, D_HEAD
from torchwright import compile_hf_bundle

output_node, embedding = create_network_parts()
compile_hf_bundle(output_node, embedding, "binary_increment_hf_bundle",
                  d=D_MODEL, d_head=D_HEAD, optimize=1)
```

The result is a directory with safetensors, config, and tokenizer which loads like any `transformers`
checkpoint:

```python
from transformers import pipeline

generate = pipeline("text-generation", model="binary_increment_hf_bundle")
print(generate("1011\n", return_full_text=False)[0]["generated_text"])
# 1100
```

The compiler schedules this graph into a 16-layer decoder at hidden size 512 (the
`d=D_MODEL` argument). Every weight is computed from the source graph. Nothing
was trained. Compiling this example takes under ten seconds on a laptop CPU.

## How it works

A torchwright program is a computation graph: ordinary Python wiring op calls into
a DAG of nodes, with token embeddings at the leaves and one output node at the
root. 

Ops come in three groups: linear ops (`add`, `subtract`, `concat`, etc), attention
ops (latch a value at a marker position, read a fixed offset back,
argmax/argmin/mean over positions), and nonlinear ops (`compare`, `select`,
`multiply`, etc).

The nonlinear ops are built from the transformer's FFN activation function, so
choosing an activation means choosing an op library. `torchwright.ops.relu`
constructs every nonlinear op from ReLU sublayers; `torchwright.ops.swiglu`
constructs the same ops from gated SwiGLU sublayers, and is what the Phi-3
output uses. A graph imports from exactly one, and the compiler rejects a graph
that mixes them.

The compiled computation lives in the residual stream. Every value the graph
computes owns a set of columns there for as long as anything downstream still
needs it; columns need not be contiguous, and concatenation is logical rather
than physical. A sublayer can only *add* to the stream, and that one fact drives
the compilation scheme: a new value lands in zeroed columns (0 + new = new), a
value nothing needs anymore is cancelled by writing its negation (v + (−v) = 0),
which frees its columns for reuse, and the graph's additions are implemented on
the skip connection itself, costing no extra columns.

Every op compiles down to concrete weights living in attention heads or rows of FFN 
sublayers. A scheduler assigns every node to a layer:
- `optimize=0`, the default, places nodes with a heuristic
- `optimize=1`–`3` uses a constraint-programming solver (CP-SAT, from Google OR-Tools) that minimizes layer
count 


## Prior work

The idea of hand-built transformer weights is not new.
[RASP](https://arxiv.org/abs/2106.06981) defines a language whose primitives map
onto transformer sublayers, and [Tracr](https://arxiv.org/abs/2301.05062)
compiles RASP programs into actual weights. torchwright differs at both ends:
programs are arbitrary computation graphs written in ordinary Python instead of 
RASP, and the output targets a stock Phi-3 architecture, rather than a custom
transformer model (with no norm).

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

Some of the constructions are exact, but plenty are piecewise-linear
approximations. Correctness gets measured rather than assumed. Every
approximated operation is checked against its exact-math reference, and the
resulting error bounds are committed to the repository
(`docs/op_noise_data.json`); the test suite fails if the committed numbers drift
from what the code measures. Graph nodes can carry assert predicates. During
development each assert is checked against the exact reference evaluation and
again during debug forward passes of the compiled model. One that passes in
exact math but fires when compiled points at exactly the operation where
approximation error exceeded its budget. Finally,
`torchwright.debug.probe.probe_compiled` diffs a compiled transformer
node-by-node against direct evaluation of the source graph.

Per-op bounds are measured on each op's intended input ranges and do not compose
through chains of ops; chain-level questions go to `probe_compiled`.

## Examples

Twelve example graphs live in `examples/`:

- `adder_1digit` parses `"A+B\n"`, single-digit addition; the smallest
  complete program.
- `adder` 3-digit addition: each digit pair goes through a lookup table and
  carries propagate right-to-left, pencil-and-paper style.
- `adder_v2` the same adder a different way: digits become numbers, one add,
  numbers become digits.
- `binary_increment` the example above.
- `caesar_cipher` shift cipher; the shift amount is a runtime input digit.
- `sort_digits_v1` sorts a digit string ascending, one digit per
  autoregressive step.
- `range_printer` a two-level loop: iterate items, and for each item iterate a
  range of values.
- `fibonacci` autoregressive: each number is computed from the model's own
  previously emitted tokens.
- `calculator_simple` `+`, `-`, `*` on multi-digit integers, one lookup table
  and one fold at a time.
- `calculator_advanced` the same functions at logarithmic depth, the way a
  hardware multiplier does it.
- `calculator_memorize` computes nothing: every answer is a memorized fact.
- `calculator_scratchpad` streams the serial carry/borrow work out as visible
  tokens and reads it back, so compiled depth stays flat as operand length grows.

Examples that define `create_network_parts()` compile to a bundle with
`uv run python -m examples.compile <name>`.

## Limitations

- Only fp32 is supported.
- With RMSNorm on (the default), supported hidden sizes (`d`) are any multiple
  of 1024 up to 16384, or any power of two; other widths raise. See
  `docs/rms_norm_dmodel.md`.
- The KV cache is static: sequence length is fixed at export (`max_seq_len`,
  default 512), and overrunning it raises.

## Further reading
- [Introducing torchwright](https://ood.dev/posts/torchwright-intro/) -- the story and the constructions, from ReLU
  derivations to stock Phi-3.
- `docs/optimization_guide.md` -- reducing layer count and parameter cost when
  compiling a graph.
- `docs/cpsat_scheduler.md` -- the constraint-programming scheduler: model,
  objective, warm starts.
- `docs/affine_bounds.md` -- how value bounds propagate through the graph.
- `docs/numerical_noise.md` -- measured per-op approximation error (generated
  from `docs/op_noise_data.json`).

## Development

`make lint` (ruff + mypy) and `make test-local FILE=tests/...` (single-file
pytest) run anywhere.

Currently the full suite, `make test`, is designed to run on Modal, sharded across GPU containers (and thus needs a Modal account).

`make measure-noise` regenerates the committed per-op noise data.

CI runs on every push and pull request: `make lint`, plus a smoke test that
compiles an example on CPU and checks its generated output.

## License

Apache-2.0.
