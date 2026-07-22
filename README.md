# torchwright

torchwright is a compiler that transforms computation graphs into the weights of a
transformer. The output is a standard decoder-only transformer — causal softmax
attention, rotary position embeddings, RMSNorm, a KV cache — and its weights are
emitted by the compiler, not trained. Compiled models are fp32-only, decode
greedily, and need no GPU.

## Example

The graph in `examples/binary_increment.py` parses a binary string and increments
it, carry propagation included. Compiling it produces an ordinary Hugging Face
model directory:

```python
from examples.binary_increment import create_network_parts, D_MODEL, D_HEAD
from torchwright import compile_hf_bundle

output_node, embedding = create_network_parts()
compile_hf_bundle(output_node, embedding, "binary_increment_hf_bundle",
                  d=D_MODEL, d_head=D_HEAD)
```

The bundle is a stock Phi-3 `transformers` checkpoint — safetensors, config,
tokenizer — and loads with `AutoModelForCausalLM`, no custom code, no
`trust_remote_code`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("binary_increment_hf_bundle").eval()
tok = AutoTokenizer.from_pretrained("binary_increment_hf_bundle")

enc = tok("1011\n", return_tensors="pt")
with torch.no_grad():
    out = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                         max_new_tokens=16, do_sample=False,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
# 1100
```

The compiler scheduled this graph into a 22-layer decoder at hidden size 256 (the
`d=D_MODEL` argument). Every weight was computed from the source graph; nothing
was trained. Compiling this example took under ten seconds on a laptop CPU.

## How it works

A torchwright program is a computation graph: ordinary Python wiring op calls into
a DAG of nodes, with token embeddings at the leaves and one output node at the
root. `examples/adder_1digit.py` is a 99-line worked example.

Ops come in three groups: linear ops (`add`, `subtract`, `concat`, …), attention
ops (latch a value at a marker position, read a fixed offset back,
argmax/argmin/mean over positions), and nonlinear ops (`compare`, `select`,
`multiply`, table lookups, `floor_int`, `mod_const`, …). The nonlinear ops are
built from the transformer's MLP activation function and exist in two parallel
libraries, `torchwright.ops.relu` and `torchwright.ops.swiglu`: the import path
decides which activation the compiled model uses, a graph uses exactly one, and
the compiler rejects a mix.

Every op compiles down to concrete weights — attention heads, rows of MLP
sublayers. A constraint-programming scheduler (CP-SAT, from Google OR-Tools) assigns every
node to a layer, minimizing layer count; `optimize=0`, the default, uses a
heuristic instead, and `optimize=1`–`3` buy increasing solver budgets (60 to 600
seconds). Weights stream into the artifact layer by layer, so peak memory during
compilation stays near one layer's worth regardless of depth.

## Output formats

`compile_hf_bundle`, shown above, is the primary output: a stock Phi-3 checkpoint
any `transformers` code can load. Two alternates cover other consumers:

- `compile_to_onnx` emits a KV-cached ONNX decoder (opset 14) plus two companion
  files: a metadata file the loader reads to run the artifact as a token LM, and
  a debug file that maps compiled tensors back to graph nodes. `load_onnx` wraps
  the artifact in a tokenizer and a greedy generate loop under onnxruntime.
- `compile_headless` builds an in-process torch module, used for tests and
  debugging.

## Install

Not yet on PyPI — install from source. Python 3.10 or later.

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

Correctness is checked at four levels. Every approximate op is measured against
its exact-math reference, and the per-op error bounds are committed to the repo
(`docs/op_noise_data.json`); the test suite fails if the committed numbers drift
from what the code measures. The compiler asserts its own structural invariants
while compiling — no two live values share a residual column, writes never
truncate, attention head widths match their declarations, values stay allocated
until their last consumer — and each invariant is pinned by a negative test.
Assert predicates can be attached to graph nodes; they run on exact values during
reference evaluation and again on compiled values during debug forward passes, so
an assert that passes in exact math but fires compiled pinpoints where
approximation error exceeded its budget. Finally,
`torchwright.debug.probe.probe_compiled` diffs a compiled transformer
node-by-node against direct evaluation of the source graph — on the in-process
backend or on the exported ONNX artifact.

Per-op bounds are measured on each op's intended input ranges and are not
additive through chains of ops; chain-level questions go to `probe_compiled`.

## Examples

Twelve example graphs live in `examples/`:

- `adder_1digit` — parses `"A+B\n"`, single-digit addition; the smallest
  complete program.
- `adder` — 3-digit addition, computed digit-by-digit in embedding space with
  carry propagation.
- `adder_v2` — the same adder in scalar space: digits become numbers, one add,
  digits again.
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

- fp32 only: the compiled attention relies on exact cancellations that a
  fp16/bf16 downcast breaks.
- Greedy decoding only (`do_sample=False`).
- One activation per graph; the ReLU and SwiGLU op libraries cannot mix.
- The KV cache is static: sequence length is fixed at export, and overrunning it
  raises.
- With RMSNorm on (the default for the Hugging Face and ONNX paths), supported
  hidden sizes (`d`) are any multiple of 1024 up to 16384, or any power of two;
  other widths raise. See `docs/rms_norm_dmodel.md`.
- `optimize>0` compiles run for minutes by design; the budgets are ceilings, not
  proofs of optimality.

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
`examples/`, `tests/`, and `docs/` are what they say. `make lint` (black + mypy)
and `make test-local FILE=tests/...` (single-file pytest) run anywhere. The full
suite, `make test`, shards across GPU containers on Modal and needs a Modal
account. `make measure-noise` regenerates the committed per-op noise data; two
tests pin it to the code.

## License

Apache-2.0.
