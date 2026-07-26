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
    uv run python -m examples.compile <name> --max-digits 5   # sized variants
    uv run python -m examples.compile <name> --optimize 2

``--max-digits`` threads through to ``create_network_parts(max_digits=...)``
for the examples that take it (the calculator family); the default bundle
directory then becomes ``<name>_n<max_digits>_hf_bundle`` so differently
sized variants don't overwrite each other.

Optional attributes on the example module:

* ``D_MODEL`` / ``D_HEAD`` / ``N_HEADS`` / ``D_HIDDEN`` — compile dimensions
  (defaults 1024 / 16 / None / None = d).
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
import inspect
import json
from pathlib import Path
from typing import cast

from torchwright.compiler.hf import compile_hf_bundle

_CARD_HEADER = """\
---
license: apache-2.0
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
out = model.generate(enc["input_ids"], max_new_tokens={max_new_tokens}, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```
"""


def _norm_gain(out_dir: str) -> str:
    """The bundle's actual RMSNorm gain, formatted for the model card."""
    import math

    from safetensors import safe_open

    with Path(out_dir, "model.safetensors.index.json").open() as f:
        weight_map = json.load(f)["weight_map"]
    key = "model.layers.0.input_layernorm.weight"
    with safe_open(str(Path(out_dir, weight_map[key])), framework="pt") as f:
        gain = float(f.get_tensor(key).max())
    exp = round(math.log2(gain))
    approx = f"{gain:.1e}".replace("e+", "e")
    if 2.0**exp == gain:
        return f"`2^{exp} ≈ {approx}`"
    return f"`{approx}`"


def model_card(
    name: str,
    task: str | None,
    gain: str,
    prompts: list[str] | None,
    max_new_tokens: int = 32,
) -> str:
    card = _CARD_HEADER.format(name=name, task=task or "a computation graph", gain=gain)
    if prompts:
        card += _CARD_USAGE.format(prompt=prompts[0], max_new_tokens=max_new_tokens)
    return card


def run_prompts(
    out_dir: str, prompts: list[str], max_new_tokens: int = 32
) -> list[str]:
    """Clean-room reload + greedy generation, as a user would consume it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(out_dir).eval()
    tok = AutoTokenizer.from_pretrained(out_dir)

    outputs: list[str] = []
    for expr in prompts:
        enc = tok(expr, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(
                enc["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
        outputs.append(
            cast(
                "str",
                tok.decode(g[0, enc["input_ids"].shape[1] :], skip_special_tokens=True),
            )
        )
    return outputs


def demo(out_dir: str, prompts: list[str], max_new_tokens: int = 32) -> None:
    print("\n=== clean-room stock Phi-3 demo ===")
    for expr, out in zip(
        prompts, run_prompts(out_dir, prompts, max_new_tokens), strict=True
    ):
        print(f"  {expr.strip():10s} -> {out}")


def _full_width_prompts(max_digits: int) -> list[str]:
    """Demo prompts that exercise the full operand width.

    The 2026-07 truncation shipped because every variant's demo used
    3-digit operands — the one region where all widths agree.  A sized
    bundle's demo (and the pre-push verification built on it) must
    prove arithmetic at its own width limit, carries included.
    """
    nines = "9" * max_digits
    dense = "".join(str(i % 9 + 1) for i in range(max_digits))
    return [f"{nines}*{nines}\n", f"{nines}+1\n", f"{nines}-{dense}\n", f"{dense}*2\n"]


def bundle_dirname(name: str, max_digits: int | None) -> str:
    """Default bundle directory name for an example (sized or not)."""
    suffix = f"_n{max_digits}" if max_digits is not None else ""
    return f"{name}{suffix}_hf_bundle"


def build_bundle(
    name: str,
    out_dir: str | None = None,
    *,
    max_digits: int | None = None,
    optimize: int = 0,
) -> tuple[str, list[str] | None, int]:
    """Compile ``examples.<name>`` into an HF bundle with its model card.

    ``max_digits`` threads through to ``create_network_parts`` for the
    examples that take it (the calculator family); passing it for one
    that doesn't raises.  Returns ``(out_dir, demo prompts or None,
    generation budget)`` — the demo itself is the caller's decision,
    not part of the build.
    """
    module = importlib.import_module(f"examples.{name}")
    if not hasattr(module, "create_network_parts"):
        raise ValueError(f"examples.{name} has no create_network_parts()")

    build_kwargs = {}
    if max_digits is not None:
        if (
            "max_digits"
            not in inspect.signature(module.create_network_parts).parameters
        ):
            raise ValueError(
                f"examples.{name}.create_network_parts takes no max_digits"
            )
        build_kwargs["max_digits"] = max_digits

    out_dir = out_dir or bundle_dirname(name, max_digits)
    output_node, embedding = module.create_network_parts(**build_kwargs)
    compile_hf_bundle(
        output_node,
        embedding,
        out_dir,
        d=getattr(module, "D_MODEL", 1024),
        d_head=getattr(module, "D_HEAD", 16),
        n_heads=getattr(module, "N_HEADS", None),
        d_hidden=getattr(module, "D_HIDDEN", None),
        optimize=optimize,
    )
    prompts = getattr(module, "DEMO_PROMPTS", None)
    if prompts is not None and max_digits is not None:
        prompts = [*prompts, *_full_width_prompts(max_digits)]
    max_new_tokens = getattr(module, "DEMO_MAX_NEW_TOKENS", 32)
    task = getattr(module, "CARD_TASK", None)
    if task is not None and max_digits is not None:
        task = f"{task}, sized for operands up to {max_digits} digits"
    card = model_card(name, task, _norm_gain(out_dir), prompts, max_new_tokens)
    with Path(out_dir, "README.md").open("w") as f:
        f.write(card)
    return out_dir, prompts, max_new_tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="example module name under examples/")
    ap.add_argument("--out", default=None, help="output dir (default <name>_hf_bundle)")
    ap.add_argument("--push", default=None, help="Hub repo id to push the bundle to")
    ap.add_argument("--no-demo", action="store_true")
    ap.add_argument(
        "--max-digits",
        type=int,
        default=None,
        help="operand digit width, for examples whose create_network_parts "
        "takes max_digits (default: the example's own default)",
    )
    ap.add_argument(
        "--optimize",
        type=int,
        default=0,
        help="compiler optimization level passed to compile_hf_bundle",
    )
    args = ap.parse_args()

    try:
        out_dir, prompts, max_new_tokens = build_bundle(
            args.name, args.out, max_digits=args.max_digits, optimize=args.optimize
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    bundle_files = sorted(p.name for p in Path(out_dir).iterdir())
    print(f"Wrote bundle to {out_dir}: {bundle_files}")

    if prompts and not args.no_demo:
        demo(out_dir, prompts, max_new_tokens)

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
