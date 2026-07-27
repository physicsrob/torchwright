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

_TWO_DIGIT_EXAMPLE_MIN_WIDTH = 2

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

# `{name}`{title_config} (torchwright)

A **compiled** transformer: the
[torchwright](https://github.com/physicsrob/torchwright) compiler emitted
these weights directly from a computation graph — nothing was trained.  This
bundle is the `{name}` example{built_with}: {task}.

The bundle uses the stock Phi-3 architecture and loads through `transformers`
without custom model code or `trust_remote_code`.

Run it in **fp32** with **greedy decoding** (`do_sample=False`).  Other
precisions and decoding modes are outside the supported contract.
"""

_CARD_IO = """\

## Input and output

Prompts are `A op B` terminated by a newline: two non-negative decimal
operands of up to {max_digits} digits, with `op` one of `+`, `-`, `*`.
Subtraction may produce a negative result.  Wider operands, or any character
outside the model's small vocabulary, are outside the contract — the output
is undefined.

| prompt | output |
|---|---|
{rows}{output_note}"""

_CARD_SIZE = """\

## Size

The checkpoint stores {size_gb} GB of dense fp32 weights ({total} entries){width}.
{zero_pct}% of those entries are
exactly zero: the vast majority of the model is unused canvas, so size reflects
the compile geometry rather than stored knowledge.

The zero entries are not compressed, and dense `transformers` execution still
pays their memory and compute cost.  CPU execution is supported; allow
additional RAM beyond the checkpoint size.
"""

_CARD_LIMITS = """\

## Intended use and limitations

This model is a demonstration of a computation graph compiled into transformer
weights.  It is not a general language model or a general-purpose calculator;
only the input contract above is supported.
"""

_CARD_VERIFICATION = """\

## Verification

The examples above are exact reference outputs.  The Modal publishing path
reloads the emitted checkpoint through stock `transformers`, checks those
examples plus additional width-limit cases against Python integer arithmetic,
and refuses to upload on a mismatch.  This is a functional smoke test, not
exhaustive verification of every allowed expression.
"""

_CARD_FAMILY = """\

## Family

One example of many compiled with torchwright.  Calculator siblings —
`calculator-simple` (serial arithmetic, depth grows with the digit count),
`calculator-advanced` (carry-lookahead, near-flat depth), and
`calculator-scratchpad` (flat depth; the serial work streams out as visible
thinking tokens) — are published at several digit widths.  Browse the
[torchwright calculator models](https://huggingface.co/models?search=physicsrob%2Ftorchwright-calculator)
on Hugging Face.
"""

_CARD_FAMILY_GENERIC = """\

## Family

One example of many compiled with torchwright; the compiler and the full
example set live at [physicsrob/torchwright](https://github.com/physicsrob/torchwright).
"""

_CARD_USAGE = """\

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = {repo_id!r}
model = AutoModelForCausalLM.from_pretrained(repo_id).eval()
tok = AutoTokenizer.from_pretrained(repo_id)

enc = tok({prompt!r}, return_tensors="pt")
out = model.generate(enc["input_ids"], max_new_tokens={max_new_tokens}, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```
"""


def _weight_stats(out_dir: str) -> tuple[int, int, int]:
    """Return ``(nonzero entries, total entries, serialized bytes)``."""
    from safetensors import safe_open

    with Path(out_dir, "model.safetensors.index.json").open() as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    size_bytes = int(index["metadata"]["total_size"])
    nonzero = total = 0
    for filename in sorted(set(weight_map.values())):
        with safe_open(str(Path(out_dir, filename)), framework="pt") as f:
            for key in list(f.keys()):
                tensor = f.get_tensor(key)
                nonzero += int((tensor != 0).sum())
                total += tensor.numel()
    return nonzero, total, size_bytes


def _exact_answer(expr: str) -> str:
    """Exact integer answer for an ``A op B`` expression."""
    for op in ("*", "+", "-"):
        a, found, b = expr.partition(op)
        if found:
            f = {"*": int.__mul__, "+": int.__add__, "-": int.__sub__}[op]
            return str(f(int(a), int(b)))
    raise ValueError(f"not an 'A op B' expression: {expr!r}")


def _io_section(module: object, max_digits: int) -> str:
    """The input-contract section with exact example rows."""
    exprs = _card_expressions(max_digits)
    output_note = getattr(module, "CARD_OUTPUT_NOTE", None)
    rows = ""
    for expr in exprs:
        answer = _exact_answer(expr)
        shown = f"<THINKING>…</THINKING>{answer}" if output_note else answer
        rows += f"| `{expr}` | `{shown}` |\n"
    note = (
        "\n"
        + output_note.format(
            nines="9" * max_digits,
            power_of_ten="1" + "0" * max_digits,
        )
        if output_note
        else ""
    )
    return _CARD_IO.format(max_digits=max_digits, rows=rows, output_note=note)


def write_card(
    name: str,
    out_dir: str,
    max_digits: int | None,
    *,
    repo_id: str | None = None,
) -> None:
    """Generate the bundle's README from the module and the written shards.

    Standalone on purpose: it reads everything it needs from the bundle
    directory, so a card can be regenerated (locally or on a Modal
    worker holding the volume) without recompiling.  ``repo_id`` replaces
    the local bundle path in the usage example when the card is about to
    be uploaded.
    """
    module = importlib.import_module(f"examples.{name}")
    prompts = getattr(module, "DEMO_PROMPTS", None)
    max_new_tokens = getattr(module, "DEMO_MAX_NEW_TOKENS", 32)
    config = f", max_digits={max_digits}" if max_digits is not None else ""
    built = f" built with `max_digits={max_digits}`" if max_digits is not None else ""
    card = _CARD_HEADER.format(
        name=name,
        title_config=config,
        built_with=built,
        task=getattr(module, "CARD_TASK", None) or "a computation graph",
    )
    if prompts:
        card += _CARD_USAGE.format(
            repo_id=repo_id or out_dir,
            prompt=prompts[0],
            max_new_tokens=max_new_tokens,
        )
    if max_digits is not None and name.startswith("calculator"):
        card += _io_section(module, max_digits)
        card += _CARD_LIMITS
        card += _CARD_VERIFICATION
    nonzero, total, size_bytes = _weight_stats(out_dir)
    config_path = Path(out_dir, "config.json")
    width = ""
    if config_path.is_file():
        with config_path.open() as f:
            hidden = json.load(f).get("hidden_size")
        if hidden:
            width = f" at compile width d={hidden}"
    card += _CARD_SIZE.format(
        width=width,
        zero_pct=f"{100.0 * (1.0 - nonzero / total):.2f}",
        total=f"{total:,}",
        size_gb=f"{size_bytes / 1e9:.2f}",
    )
    card += _CARD_FAMILY if name.startswith("calculator") else _CARD_FAMILY_GENERIC
    with Path(out_dir, "README.md").open("w") as f:
        f.write(card)


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
    return [
        f"{nines}*{nines}\n",
        f"{nines}+1\n",
        f"{nines}-{dense}\n",
        f"{dense}*2\n",
        f"{dense}-{nines}\n",
    ]


def _card_expressions(max_digits: int) -> list[str]:
    """Representative in-contract expressions for a sized calculator card."""
    exprs = ["7+8"]
    if max_digits >= _TWO_DIGIT_EXAMPLE_MIN_WIDTH:
        exprs.insert(0, "12*34")
    full_width = _full_width_prompts(max_digits)
    exprs.extend(p.strip() for p in [*full_width[:3], full_width[-1]])
    return list(dict.fromkeys(exprs))


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
    d: int | None = None,
    d_hidden: int | None = None,
) -> tuple[str, list[str] | None, int]:
    """Compile ``examples.<name>`` into an HF bundle with its model card.

    ``max_digits`` threads through to ``create_network_parts`` for the
    examples that take it (the calculator family); passing it for one
    that doesn't raises.  ``d`` / ``d_hidden`` override the module's
    family geometry — for right-sized small-n bundles whose free-shrink
    region is measured (scripts/measure_calculator_compiled_layers);
    ``d_head`` is never overridable (it is baked into the graph at
    build time).  Returns ``(out_dir, demo prompts or None, generation
    budget)`` — the demo itself is the caller's decision, not part of
    the build.
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
        d=d if d is not None else int(getattr(module, "D_MODEL", 1024)),
        d_head=getattr(module, "D_HEAD", 16),
        n_heads=getattr(module, "N_HEADS", None),
        d_hidden=d_hidden
        if d_hidden is not None
        else getattr(module, "D_HIDDEN", None),
        optimize=optimize,
    )
    write_card(name, out_dir, max_digits)
    prompts = getattr(module, "DEMO_PROMPTS", None)
    if prompts is not None and max_digits is not None:
        prompts = [*prompts, *_full_width_prompts(max_digits)]
    return out_dir, prompts, getattr(module, "DEMO_MAX_NEW_TOKENS", 32)


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
        "--d",
        type=int,
        default=None,
        help="override the module's D_MODEL (right-sized small-n bundles)",
    )
    ap.add_argument(
        "--d-hidden",
        type=int,
        default=None,
        help="override the module's D_HIDDEN",
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
            args.name,
            args.out,
            max_digits=args.max_digits,
            optimize=args.optimize,
            d=args.d,
            d_hidden=args.d_hidden,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    bundle_files = sorted(p.name for p in Path(out_dir).iterdir())
    print(f"Wrote bundle to {out_dir}: {bundle_files}")

    if prompts and not args.no_demo:
        demo(out_dir, prompts, max_new_tokens)

    if args.push:
        from huggingface_hub import HfApi

        write_card(
            args.name,
            out_dir,
            args.max_digits,
            repo_id=args.push,
        )
        api = HfApi()
        api.create_repo(args.push, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
