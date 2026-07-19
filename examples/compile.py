"""Compile an example to a published Hugging Face bundle.

Any module under ``examples/`` that defines ``create_network_parts()``
compiles into a stock Phi-3 ``transformers`` bundle — sharded safetensors,
``config.json``, tokenizer files, and a generated model card — loadable by
ordinary ``AutoModelForCausalLM`` with no custom model code and no
``trust_remote_code``.

Usage:
    uv run python -m examples.compile <name>                  # export + demo
    uv run python -m examples.compile <name> --out DIR
    uv run python -m examples.compile <name> --push USER/REPO
    uv run python -m examples.compile <name> --no-demo

Optional attributes on the example module:

* ``D_MODEL`` / ``D_HEAD`` / ``N_HEADS`` — compile dimensions
  (defaults 1024 / 16 / None).
* ``DEMO_PROMPTS`` — list of prompt strings.  Drives the clean-room demo
  and the model card's usage snippet; without it the demo is skipped and
  the card carries no usage section.
* ``CARD_TASK`` — noun phrase describing the source graph, slotted into
  the model card's opening sentence (e.g. "a computation graph for
  integer arithmetic").

fp32, greedy-only, CPU-fine.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os

from torchwright.compiler.hf import compile_hf_bundle

_CARD_HEADER = """\
---
license: mit
library_name: transformers
tags:
  - torchwright
  - compiled-transformer
  - {name}
pipeline_tag: text-generation
---

# Compiled `{name}` (torchwright)

This is a **compiled** transformer: {task} compiled by torchwright into
transformer weights, shipped as a stock Phi-3 `transformers` causal LM. The
weights are not trained — they are *emitted* by a compiler from the source
graph.

* **Precision**: **fp32 only**. The compiled attention relies on exact
  algebraic cancellation; a fp16/bf16 downcast breaks correctness.
* **Decoding**: greedy (`do_sample=False`) only.
* **Hardware**: CPU is fine.

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
the same large constant ({gain} in this model), where a trained
RMSNorm's gains cluster near 1. We keep the real norm — rather than dropping it
and claiming "no normalization" — so the architecture is a faithful standard
transformer; the atypical gain magnitude is the price of making the norm an
exact identity, and we name it rather than hide it.
"""

_CARD_USAGE = """\

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(REPO).eval()
tok = AutoTokenizer.from_pretrained(REPO)

enc = tok({prompt!r}, return_tensors="pt")
out = model.generate(enc["input_ids"], max_new_tokens=32, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```
"""


def _norm_gain(out_dir: str) -> str:
    """The bundle's actual RMSNorm gain, formatted for the model card."""
    import math

    from safetensors import safe_open

    with open(os.path.join(out_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    key = "model.layers.0.input_layernorm.weight"
    with safe_open(os.path.join(out_dir, weight_map[key]), framework="pt") as f:
        gain = float(f.get_tensor(key).max())
    exp = round(math.log2(gain))
    approx = f"{gain:.1e}".replace("e+", "e")
    if 2.0**exp == gain:
        return f"`2^{exp} ≈ {approx}`"
    return f"`{approx}`"


def model_card(name: str, task: str | None, gain: str, prompts) -> str:
    card = _CARD_HEADER.format(name=name, task=task or "a computation graph", gain=gain)
    if prompts:
        card += _CARD_USAGE.format(prompt=prompts[0])
    return card


def demo(out_dir: str, prompts) -> None:
    """Clean-room reload + greedy generation, as a user would consume it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(out_dir).eval()
    tok = AutoTokenizer.from_pretrained(out_dir)

    print("\n=== clean-room stock Phi-3 demo ===")
    for expr in prompts:
        enc = tok(expr, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(
                enc["input_ids"],
                max_new_tokens=32,
                do_sample=False,
                use_cache=True,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
        out = tok.decode(g[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"  {expr.strip():10s} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="example module name under examples/")
    ap.add_argument("--out", default=None, help="output dir (default <name>_hf_bundle)")
    ap.add_argument("--push", default=None, help="Hub repo id to push the bundle to")
    ap.add_argument("--no-demo", action="store_true")
    args = ap.parse_args()

    module = importlib.import_module(f"examples.{args.name}")
    if not hasattr(module, "create_network_parts"):
        raise SystemExit(f"examples.{args.name} has no create_network_parts()")

    out_dir = args.out or f"{args.name}_hf_bundle"
    output_node, embedding = module.create_network_parts()
    compile_hf_bundle(
        output_node,
        embedding,
        out_dir,
        d=getattr(module, "D_MODEL", 1024),
        d_head=getattr(module, "D_HEAD", 16),
        n_heads=getattr(module, "N_HEADS", None),
    )
    prompts = getattr(module, "DEMO_PROMPTS", None)
    card = model_card(
        args.name, getattr(module, "CARD_TASK", None), _norm_gain(out_dir), prompts
    )
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(card)
    print(f"Wrote bundle to {out_dir}: {sorted(os.listdir(out_dir))}")

    if prompts and not args.no_demo:
        demo(out_dir, prompts)

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
