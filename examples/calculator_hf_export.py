"""Export the compiled calculator as a standard HuggingFace causal LM.

The calculator (``examples/calculator_v2.py``) is compiled by torchwright into a
transformer; this script reimplements that artifact as a native
``transformers`` model (``compiler/hf``) and writes a self-contained,
``trust_remote_code`` bundle — proving the compiled artifact is a *bona fide*
standard transformer, not a bespoke runtime.

What it does:

1. Ensures ``calculator_v2.onnx`` exists (compiles it in-process if missing —
   the artifact is gitignored, never committed).
2. Converts ONNX → safetensors + ``config.json`` + the shipped
   modeling/config/tokenizer files, and writes a model card.
3. Demos a clean-room reload via ``AutoModelForCausalLM.from_pretrained(...,
   trust_remote_code=True)`` and greedy generation.
4. Optionally ``push_to_hub``.

Usage:
    uv run python -m examples.calculator_hf_export                 # export + demo
    uv run python -m examples.calculator_hf_export --out DIR
    uv run python -m examples.calculator_hf_export --push user/calculator-tw

fp32, greedy-only, CPU-fine. The model is token-identical (in fact bit-exact) to
the ONNX runtime across the calculator's reliable range; at the extreme top of
its multiply range the compiled circuit runs at its numerical-noise budget and
the two fp32 backends can round a borderline digit differently (see
``tests/hf/test_calculator_parity.py``).
"""

from __future__ import annotations

import argparse
import os

ONNX_PATH = "calculator_v2.onnx"
BOS = "<bos"
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
here as a native `transformers` causal LM. The weights are not trained — they
are *emitted* by a compiler from the arithmetic graph.

* **Task**: greedy text generation. Prompt `"<bos" + "12*34\\n"`, read back the
  digits of the answer until `"<eos>"`.
* **Precision**: **fp32 only**. The compiled attention relies on exact algebraic
  cancellation; a fp16/bf16 downcast breaks correctness.
* **Decoding**: greedy (`do_sample=False`) only.
* **Hardware**: CPU is fine (tiny model).

It is token-identical — in fact bit-exact — to the source ONNX runtime across
the calculator's reliable range. At the very top of its multiply range the
compiled piecewise-linear arithmetic runs at its numerical-noise budget, where
two faithful fp32 backends can round one borderline output digit differently.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True).eval()
tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)

enc = tok("12*34\\n", return_tensors="pt")
out = model.generate(enc["input_ids"], max_new_tokens=10, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
# 408
```
"""


def ensure_artifact() -> str:
    if os.path.exists(ONNX_PATH):
        return ONNX_PATH
    import importlib

    from torchwright.compiler.export import compile_to_onnx

    module = importlib.import_module("examples.calculator_v2")
    output_node, pos_encoding, embedding = module.create_network_parts()
    compile_to_onnx(
        output_node, pos_encoding, embedding, ONNX_PATH, d=module.D_MODEL
    )
    return ONNX_PATH


def export(out_dir: str) -> None:
    from torchwright.compiler.hf.convert import save_bundle

    onnx_path = ensure_artifact()
    save_bundle(onnx_path, out_dir, bos_token=BOS, eos_token=EOS)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(_MODEL_CARD)
    print(f"Wrote bundle to {out_dir}: {sorted(os.listdir(out_dir))}")


def demo(out_dir: str) -> None:
    """Clean-room reload + greedy generation, as a user would consume it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        out_dir, trust_remote_code=True
    ).eval()
    tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)

    print("\n=== clean-room trust_remote_code demo ===")
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
        out = tok.decode(
            g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
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

        model = AutoModelForCausalLM.from_pretrained(args.out, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(args.out, trust_remote_code=True)
        model.push_to_hub(args.push)
        tok.push_to_hub(args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
