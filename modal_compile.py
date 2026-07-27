"""Compile example HF bundles on Modal and optionally push them to the Hub.

A d=8192 calculator bundle is 6-25GB of dense fp32: compiling a family
of them is pure-CPU but heavy, and the artifacts are pointless to sync
back to the local machine when the push to the Hugging Face Hub can run
from Modal too.  This is one of the purpose-built entrypoints CLAUDE.md
carves out for artifact-producing runs: :func:`examples.compile.build_bundle`
runs remotely — one worker per (example, max_digits) cell, fanned out
with ``.map`` — and every bundle lands in the persistent Modal Volume
``torchwright-hf-bundles``.

Each worker then replays the example's demo prompts through the actual
bundle (the same clean-room reload a Hub user gets).  For the calculator
family the answers are checked against exact integer arithmetic; a
mismatch marks the cell as an error, and error cells are never pushed.

Pushing (``--push-prefix``) runs from the volume on Modal workers and
needs the ``huggingface`` Modal secret (an ``HF_TOKEN`` with write
scope).  Repos are named ``<prefix><example>-max-digits-<n>`` with
underscores hyphenated, e.g. ``physicsrob/torchwright-`` +
``calculator_simple:3`` -> ``physicsrob/torchwright-calculator-simple-max-digits-3``.

Inspect the volume with ``uv run modal volume ls torchwright-hf-bundles``.

Usage::

    uv run modal run modal_compile.py --specs calculator_simple:3
    uv run modal run modal_compile.py \
        --specs calculator_simple:3,calculator_advanced:10 --optimize 2 \
        --push-prefix physicsrob/torchwright-
    uv run modal run modal_compile.py --specs binary_increment  # no max_digits
"""

import json
import re
import time
from pathlib import Path

import modal

from modal_image import IMAGE

BUNDLE_ROOT = "/bundles"

app = modal.App("torchwright-compile", image=IMAGE)
bundles = modal.Volume.from_name("torchwright-hf-bundles", create_if_missing=True)

_EXPR = re.compile(r"^(\d+)([+\-*])(\d+)\n$")


def _expected_answer(prompt: str) -> str | None:
    """Exact answer for a calculator demo prompt; None for other examples."""
    m = _EXPR.match(prompt)
    if m is None:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return str(a + b if op == "+" else a - b if op == "-" else a * b)


def repo_name(name: str, max_digits: int | None) -> str:
    base = name.replace("_", "-")
    return base if max_digits is None else f"{base}-max-digits-{max_digits}"


@app.function(cpu=8, memory=65536, timeout=7200, volumes={BUNDLE_ROOT: bundles})
def compile_remote(spec: tuple) -> dict:
    from examples.compile import build_bundle, bundle_dirname, run_prompts

    name, max_digits, optimize, d, d_hidden = (*spec, None, None)[:5]
    dirname = bundle_dirname(name, max_digits)
    t0 = time.time()
    try:
        out_dir, prompts, max_new_tokens = build_bundle(
            name,
            str(Path(BUNDLE_ROOT) / dirname),
            max_digits=max_digits,
            optimize=optimize,
            d=d,
            d_hidden=d_hidden,
        )
    except (ValueError, RuntimeError) as exc:
        # A cell can refuse to build (calculator_memorize past n=2) or to
        # schedule (calculator_scratchpad at n>=9); report and keep the sweep.
        return {"bundle": dirname, "error": f"{type(exc).__name__}: {exc}"}
    bundles.commit()

    with (Path(out_dir) / "config.json").open() as f:
        config = json.load(f)
    with (Path(out_dir) / "model.safetensors.index.json").open() as f:
        size_gb = json.load(f)["metadata"]["total_size"] / 1e9
    row = {
        "bundle": dirname,
        "n_layers": config.get("num_hidden_layers", config.get("n_layers")),
        "size_gb": round(size_gb, 2),
        "seconds": round(time.time() - t0),
    }

    if prompts:
        failures = []
        for prompt, out in zip(
            prompts, run_prompts(out_dir, prompts, max_new_tokens), strict=True
        ):
            expected = _expected_answer(prompt)
            print(
                f"[{dirname}] {prompt.strip()!r} -> {out.strip()!r}"
                f" (expected {expected})",
                flush=True,
            )
            # endswith, not ==: the scratchpad streams thinking glyphs
            # before the answer digits.
            if expected is not None and not out.strip().endswith(expected):
                failures.append(f"{prompt.strip()} -> {out.strip()!r} != {expected}")
        if failures:
            row["error"] = "demo mismatch: " + "; ".join(failures)
    return row


@app.function(cpu=8, memory=65536, timeout=7200, volumes={BUNDLE_ROOT: bundles})
def verify_remote(dirname: str, exprs: str, max_new_tokens: int = 32) -> list[str]:
    """Run comma-separated ``A op B`` expressions through a volume bundle.

    Each answer is checked against exact integer arithmetic; the returned
    report lines carry OK / MISMATCH verdicts.
    """
    from examples.compile import run_prompts

    prompts = [e.strip() + "\n" for e in exprs.split(",")]
    outputs = run_prompts(str(Path(BUNDLE_ROOT) / dirname), prompts, max_new_tokens)
    lines = []
    for prompt, out in zip(prompts, outputs, strict=True):
        expected = _expected_answer(prompt)
        ok = expected is not None and out.strip().endswith(expected)
        lines.append(
            f"{prompt.strip()} -> {out.strip()!r}"
            f" (expected {expected}) {'OK' if ok else 'MISMATCH'}"
        )
        print(f"[{dirname}] {lines[-1]}", flush=True)
    return lines


@app.function(
    gpu="B200", cpu=8, memory=262144, timeout=7200, volumes={BUNDLE_ROOT: bundles}
)
def accuracy_remote(
    dirname: str, n: int = 500, seed: int = 0, batch_size: int = 250
) -> dict:
    """Randomized accuracy sweep of a volume bundle on one B200.

    A single B200's 192GB holds even the n=12 bundles in full fp32 —
    no sharding, no downcast (TF32 disabled in the script; the compiled
    attention's exact cancellation does not survive it).  Operand digit
    lengths are drawn independently in [1, max_digits] per side.
    """
    from scripts.verify_bundle_accuracy import make_examples, report, run

    m = re.fullmatch(r"(.+?)_n(\d+)_hf_bundle", dirname)
    if m is None:
        raise ValueError(f"not a sized bundle dirname: {dirname!r}")
    examples = make_examples(n, int(m.group(2)), seed)
    results = run(str(Path(BUNDLE_ROOT) / dirname), examples, batch_size=batch_size)
    n_bad = report(results)
    return {"bundle": dirname, "n": n, "seed": seed, "mismatches": n_bad}


@app.function(gpu="B200", cpu=8, memory=65536, timeout=7200)
def exhaustive_remote(
    repo_id: str,
    shard: int,
    n_shards: int,
    max_value: int = 999,
    batch_size: int = 8192,
) -> dict:
    """One shard of the exhaustive grid, straight from the published Hub repo.

    Every ``A op B`` with both operands in ``[0, max_value]`` and each
    op, split op-major across ``n_shards`` workers.  A mismatch is
    immediately re-run unbatched to separate a model error from a
    batching/padding artifact before it lands in the report.
    """
    from scripts.verify_bundle_accuracy import (
        exhaustive_examples,
        expected,
        generate_answers,
        load,
    )

    total = 3 * (max_value + 1) ** 2
    start = shard * total // n_shards
    stop = (shard + 1) * total // n_shards
    model, tok = load(repo_id)
    results = generate_answers(
        model,
        tok,
        exhaustive_examples(start, stop, max_value=max_value),
        batch_size=batch_size,
        max_new_tokens=10,
    )
    mismatches = []
    for a, op, b, out in results:
        want = expected(a, op, b)
        if out != want:
            single = generate_answers(
                model, tok, [(a, op, b)], batch_size=1, max_new_tokens=10
            )[0][3]
            mismatches.append(
                {"expr": f"{a}{op}{b}", "batched": out, "single": single, "want": want}
            )
            print(
                f"[shard {shard}] MISMATCH {a}{op}{b}: batched={out!r} "
                f"single={single!r} expected={want}",
                flush=True,
            )
    print(
        f"[shard {shard}] SUMMARY {len(results) - len(mismatches)}/{len(results)} "
        f"correct ({start}..{stop})",
        flush=True,
    )
    return {"shard": shard, "n": len(results), "mismatches": mismatches}


@app.function(cpu=4, memory=8192, timeout=7200, volumes={BUNDLE_ROOT: bundles})
def card_remote(dirname: str) -> str:
    """Regenerate a volume bundle's README without recompiling.

    ``write_card`` reads everything it needs from the bundle directory
    (density from the shards, geometry from config.json), so card
    iterations don't cost a compile.
    """
    from examples.compile import write_card

    m = re.fullmatch(r"(.+?)(?:_n(\d+))?_hf_bundle", dirname)
    if m is None:
        raise ValueError(f"not a bundle dirname: {dirname!r}")
    write_card(
        m.group(1),
        str(Path(BUNDLE_ROOT) / dirname),
        int(m.group(2)) if m.group(2) else None,
    )
    bundles.commit()
    return dirname


@app.function(
    cpu=4,
    memory=8192,
    timeout=7200,
    volumes={BUNDLE_ROOT: bundles},
    secrets=[modal.Secret.from_name("huggingface")],
)
def push_remote(dirname: str, repo_id: str) -> str:
    from huggingface_hub import HfApi

    from examples.compile import write_card

    m = re.fullmatch(r"(.+?)(?:_n(\d+))?_hf_bundle", dirname)
    if m is None:
        raise ValueError(f"not a bundle dirname: {dirname!r}")
    write_card(
        m.group(1),
        str(Path(BUNDLE_ROOT) / dirname),
        int(m.group(2)) if m.group(2) else None,
        repo_id=repo_id,
    )
    bundles.commit()

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=str(Path(BUNDLE_ROOT) / dirname),
        repo_id=repo_id,
        # A replacement bundle may have fewer shards than its
        # predecessor; stale remote shards would linger (the index
        # ignores them, but they bloat and confuse the repo).
        delete_patterns=["model-*.safetensors"],
    )
    return f"https://huggingface.co/{repo_id}"


@app.local_entrypoint()
def main(specs: str, optimize: int = 0, push_prefix: str = "") -> None:
    cells = []
    for cell_spec in specs.split(","):
        # name[:max_digits[:d[:d_hidden]]] — trailing fields optional.
        fields = [f or None for f in cell_spec.strip().split(":")]
        fields += [None] * (4 - len(fields))
        name_part, digits, d, d_hidden = fields[:4]
        if not name_part:
            raise ValueError(f"bad spec: {cell_spec!r}")
        cells.append(
            (
                name_part,
                int(digits) if digits else None,
                optimize,
                int(d) if d else None,
                int(d_hidden) if d_hidden else None,
            )
        )

    to_push = []
    for (name, n, *_rest), row in zip(cells, compile_remote.map(cells), strict=True):
        print(row, flush=True)
        if push_prefix and "error" not in row:
            to_push.append((row["bundle"], f"{push_prefix}{repo_name(name, n)}"))
    for url in push_remote.starmap(to_push):
        print(f"pushed: {url}", flush=True)
