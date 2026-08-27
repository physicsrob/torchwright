# Continuous Hugging Face bundles

Torchwright can compile a graph with named tensor inputs and outputs into a
Hugging Face-style checkpoint. This is a parallel path to token compilation:
token bundles still use an `Embedding`, tokenizer, language-model head, and
`generate()`, while continuous bundles call the compiled transformer as a
numerical function.

For a state transition, one invocation represents

\[
S_{k+1} = T_\theta(S_k, u_k).
\]

Values stay in the transformer's residual stream throughout. They are never
converted to token IDs or logits.

## Compile a bundle

Build a graph from named `InputNode` objects and pass a mapping of public output
names to graph nodes:

```python
from torchwright import compile_continuous_hf_bundle
from torchwright.graph import Add, InputNode, LiteralValue
import torch

state = InputNode("state", 2, value_range=(-100.0, 100.0))
update = InputNode("update", 2, value_range=(-10.0, 10.0))
offset = LiteralValue(torch.tensor([1.0, -2.0]))
next_state = Add(Add(state, update), offset)

compile_continuous_hf_bundle(
    outputs={"state": next_state},
    output_dir="state_transition",
    n_positions=8,
    d=256,
    d_head=16,
)
```

`n_positions` fixes the sequence dimension of the interface. Each unbatched
runtime input has shape `(n_positions, InputNode.d_output)`. A batched call adds
one leading batch dimension. All inputs in a call must either be unbatched or
have the same batch size.

Continuous bundles support both Torchwright operation libraries. The compiler
infers a biased-ReLU or biased-SwiGLU decoder from the graph's FFN nodes. A
mixed-machine graph is rejected, as it is by the other compiler targets. A graph
with no FFN nodes falls back to ReLU. Pass `machine="relu"` or
`machine="swish"` to pin the physical machine and reject a graph built from the
other library.

Use `torchwright.ops.swiglu` for continuous algorithms that multiply two live
values. Its `multiply(a, b)` uses the complementary gated pair
`Swish(a)·b + Swish(-a)·(-b) = a·b`: it has no product grid or declared range
limit. The identity is exact in real arithmetic; execution still has ordinary
fp32 rounding. ReLU's `multiply_2d` is instead a range-bounded piecewise-linear
approximation.

The graph must contain at least one named `InputNode`, and it cannot contain a
token `Embedding`. Output names must be non-empty; input names must also be
non-empty and unique.

The other compiler controls have their usual meanings: `max_layers`,
`optimize`, `d_hidden`, `trim_heads`, and `n_heads`. `rms_norm=False` is the
default for both machines. Enabling it preserves the compiler's pinned
normalization constants and uses the same supported-width rules as other
RMSNorm compilation.

Compilation returns `ContinuousHFBundleReport`, containing the published path,
layer count, fixed position count, and selected-schedule provenance.

## Run named inputs and outputs

`ContinuousRunner` hides the residual coordinates:

```python
from torchwright import ContinuousRunner
import torch

runner = ContinuousRunner.from_pretrained("state_transition")

result = runner(
    state=torch.zeros(8, 2),
    update=torch.ones(8, 2),
)
next_state = result["state"]
```

Inputs are converted to fp32 on the model's device. Outputs are fp32 tensors
with the same unbatched or batched convention as the inputs. Missing names,
unexpected names, inconsistent batching, and incorrect sequence or value widths
raise before model execution.

Pass `device="cuda"` or another PyTorch device to
`ContinuousRunner.from_pretrained` to move both the model and stored initial
residual. A local directory and a Hugging Face Hub repository ID are accepted.

## What the bundle stores

A continuous bundle contains normal `config.json` and sharded model
safetensors, plus:

- `continuous_io.json`: the versioned named interface;
- `continuous_io.safetensors`: the compiler-generated initial residual stream;
- Torchwright's custom configuration and modeling Python files, so the model is
  a loadable Hugging Face artifact.

The JSON format starts with
`"format": "torchwright_continuous_io_v1"`. It records the fixed number of
positions, residual width, fp32 dtype, each input and output width and shape, and
the exact residual columns assigned to every named value. `config.json` carries
a `torchwright_continuous_io` pointer to both sidecars.

The stored base residual has shape `(n_positions, d_model)`. It is created by
`HeadlessTransformer.get_input_res_stream()` with zero runtime inputs, so it
contains the same compiler-created literals and pinned RMSNorm values as
headless execution. At runtime the runner expands and clones this tensor, then
overwrites only the recorded input columns. There is no separately maintained
implementation of residual initialization.

The checkpoint has no tokenizer. Its one-row embedding parameter is structural
storage required by the shared Hugging Face causal-model class; neither the
source graph nor `ContinuousRunner` performs an embedding lookup or uses the
language-model head.

## Raw residual execution

The shipped custom Hugging Face model accepts the standard `inputs_embeds`
argument. `forward_residual()` returns the decoder output after all compiled
layers but before the final token-oriented RMSNorm and language-model head:

```python
raw_residual = runner.model.forward_residual(
    inputs_embeds=complete_initial_residual,
    use_cache=False,
)
```

Output columns in `continuous_io.json` refer to this raw tensor. Do not extract
continuous outputs from `last_hidden_state`: when final RMSNorm is enabled,
`last_hidden_state` is in a different numerical representation. Constructing
`complete_initial_residual` manually is also unnecessary for normal use; call
the runner instead.

Existing token models can use `inputs_embeds` too, and their ordinary
`input_ids`, logits, tied language-model head, and `generate()` behavior remain
unchanged.

## Recurrent execution

`run_until` repeatedly feeds one named output into one named input. Each step is
a fresh transformer invocation with `use_cache=False`; no autoregressive token
serialization or key/value-cache history is involved.

```python
result = runner.run_until(
    initial_state,
    state_input="state",
    state_output="state",
    stop_output="converged",
    max_steps=50,
    update=static_update,
)
```

The graph computes `converged`. The Python loop only decides whether another
invocation is needed. By default it stops when every element of the named stop
output is greater than zero, which matches Torchwright's positive/negative
boolean convention. Supply `stop_when=lambda value: ...` to choose a different
generic reduction. The returned mapping is the final invocation's complete
named output.

`state_input`, `state_output`, and `stop_output` are configurable; the helper
contains no application-specific update, convergence, matrix, or SCF logic.
