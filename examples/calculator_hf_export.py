"""Export the compiled calculator as a standard HuggingFace causal LM.

The calculator (``examples/calculator_simple.py``) is compiled by torchwright into a
transformer; this script compiles it directly to a stock Phi-3
``transformers`` bundle — no custom model code or ``trust_remote_code``.

What it does:

1. Compiles the calculator graph directly into sharded safetensors.
2. Writes ``config.json`` + the shipped
   modeling/config/tokenizer files, and writes a model card.
3. Demos a clean-room reload via ordinary ``AutoModelForCausalLM`` and greedy
   generation.
4. Optionally ``push_to_hub``.

Usage:
    uv run python -m examples.calculator_hf_export                 # export + demo
    uv run python -m examples.calculator_hf_export --out DIR
    uv run python -m examples.calculator_hf_export --push user/calculator-tw

fp32, greedy-only, CPU-fine. The graph uses the maintained one-hot/SwiGLU
calculator implementation and the saved model is ordinary stock Phi-3.
"""

from __future__ import annotations

import argparse
import os

BOS = "<bos>"
EOS = "<eos>"

_MODEL_CARD = """\
---
license: mit
library_name: transformers
tags:
  - torchwright
  - compiled-transformer
  - calculator
pipeline_tag: text-generation
---

# Compiled Calculator (torchwright)

This is a **compiled** transformer: a computation graph for integer arithmetic
(`A op B` with `op` in `+ - *`) compiled by
[torchwright](https://github.com/) into transformer weights, then reimplemented
here as a stock Phi-3 `transformers` causal LM. The weights are not trained — they
are *emitted* by a compiler from the arithmetic graph.

* **Task**: greedy text generation. Prompt `"<bos>" + "12*34\\n"`, read back the
  digits of the answer until `"<eos>"`.
* **Precision**: **fp32 only**. The compiled attention relies on exact algebraic
  cancellation; a fp16/bf16 downcast breaks correctness.
* **Decoding**: greedy (`do_sample=False`) only.
* **Hardware**: CPU is fine (tiny model).

The same source graph may be compiled independently to ONNX for sibling-backend
validation; ONNX is not an intermediate artifact for this bundle.

## Normalization: a genuine RMSNorm that computes the identity

Like a stock Llama-style decoder, every block applies an `RMSNorm` before
attention and before the MLP, with a final `RMSNorm` before the unembedding —
the standard `input_layernorm` / `post_attention_layernorm` / `model.norm`
weights. They are real ops and run on any standard engine.

Because the weights are **compiled, not trained**, the norm does not need to
*do* anything. Training needs normalization to keep activations in range; this
model emits exact values and must preserve them. So the residual stream is
arranged so the norm is the **identity**: one residual column is pinned to a
large constant whose energy fixes the per-position RMS to an exact power of two,
and the gain is set to cancel that RMS exactly — `x / rms * gain == x`, bit for
bit. The norm runs for real; it just returns its input.

The one honest tell that this was compiled rather than trained: every gain is
the same large constant (`2^39 ≈ 5.5e11` in this model), where a trained
RMSNorm's gains cluster near 1. We keep the real norm — rather than dropping it
and claiming "no normalization" — so the architecture is a faithful standard
transformer; the atypical gain magnitude is the price of making the norm an
exact identity, and we name it rather than hide it.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(REPO).eval()
tok = AutoTokenizer.from_pretrained(REPO)

enc = tok("12*34\\n", return_tensors="pt")
out = model.generate(enc["input_ids"], max_new_tokens=10, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
# 408
```
"""


def export(out_dir: str) -> None:
    from examples import calculator_simple
    from torchwright.compiler.hf import compile_hf_bundle

    output_node, embedding = calculator_simple.create_network_parts()
    compile_hf_bundle(output_node, embedding, out_dir,
        d=calculator_simple.D_MODEL, bos_token=BOS, eos_token=EOS)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(_MODEL_CARD)
    print(f"Wrote bundle to {out_dir}: {sorted(os.listdir(out_dir))}")


def demo(out_dir: str) -> None:
    """Clean-room reload + greedy generation, as a user would consume it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(out_dir).eval()
    tok = AutoTokenizer.from_pretrained(out_dir)

    print("\n=== clean-room stock Phi-3 demo ===")
    for expr in ["12*34\n", "7+8\n", "100-99\n", "123*4\n"]:
        enc = tok(expr, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(
                enc["input_ids"],
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
        out = tok.decode(g[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"  {expr.strip():10s} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="calculator_hf_bundle", help="output dir")
    ap.add_argument("--push", default=None, help="Hub repo id to push to")
    ap.add_argument("--no-demo", action="store_true")
    args = ap.parse_args()

    export(args.out)
    if not args.no_demo:
        demo(args.out)

    if args.push:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(args.out)
        tok = AutoTokenizer.from_pretrained(args.out)
        model.push_to_hub(args.push)
        tok.push_to_hub(args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
