"""Randomized accuracy sweep for a published calculator bundle.

Draws ``A op B`` examples with independently random operand digit
lengths in ``[1, max_digits]``, runs them through the bundle with
batched greedy decoding, and checks every answer against exact integer
arithmetic.  Batching is the speed lever: each decode step streams the
full weight set regardless of batch size, so one batch of hundreds
costs barely more than one prompt.

fp32 end to end; TF32 is explicitly disabled — the compiled attention's
exact cancellation does not survive TF32's mantissa truncation.

Run on Modal against a volume bundle (B200 worker)::

    uv run modal run modal_compile.py::accuracy_remote \
        --dirname calculator_simple_n12_hf_bundle --n 500

or locally against any bundle directory::

    uv run python -m scripts.verify_bundle_accuracy --path DIR --n 50 --device cpu
"""

import argparse
from collections import Counter

import numpy as np

Example = tuple[int, str, int]


def make_examples(
    n: int, max_digits: int, seed: int, ops: str = "+-*"
) -> list[Example]:
    """``n`` random examples; digit lengths independent-uniform in [1, max]."""
    rng = np.random.default_rng(seed)
    out: list[Example] = []
    for _ in range(n):
        operands = []
        for _side in range(2):
            length = int(rng.integers(1, max_digits + 1))
            lo = 0 if length == 1 else 10 ** (length - 1)
            operands.append(int(rng.integers(lo, 10**length)))
        out.append((operands[0], ops[int(rng.integers(len(ops)))], operands[1]))
    return out


def expected(a: int, op: str, b: int) -> str:
    f = {"+": int.__add__, "-": int.__sub__, "*": int.__mul__}[op]
    return str(f(a, b))


def run(
    path: str,
    examples: list[Example],
    *,
    batch_size: int = 250,
    device: str = "cuda",
    max_new_tokens: int = 32,
) -> list[tuple[int, str, int, str]]:
    """Batched greedy decode; returns ``(a, op, b, model_answer)`` rows."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model = model.eval().to(device)

    results: list[tuple[int, str, int, str]] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            prompts = [f"{a}{op}{b}\n" for a, op, b in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True).to(device)
            g = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
            outs = tok.batch_decode(
                g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for (a, op, b), out in zip(chunk, outs, strict=True):
                results.append((a, op, b, out.strip()))
            print(f"  {len(results)}/{len(examples)} generated", flush=True)
    return results


def report(results: list[tuple[int, str, int, str]]) -> int:
    """Print the verdict breakdown; returns the mismatch count."""
    by_op: Counter[str] = Counter()
    bad_by_op: Counter[str] = Counter()
    by_width: Counter[int] = Counter()
    bad_by_width: Counter[int] = Counter()
    mismatches = []
    for a, op, b, out in results:
        width = max(len(str(abs(a))), len(str(abs(b))))
        by_op[op] += 1
        by_width[width] += 1
        if out != expected(a, op, b):
            bad_by_op[op] += 1
            bad_by_width[width] += 1
            mismatches.append((a, op, b, out))

    print(f"\n{len(results) - len(mismatches)}/{len(results)} correct", flush=True)
    print("per op:  ", flush=True)
    for op in sorted(by_op):
        print(f"    {op}: {by_op[op] - bad_by_op[op]}/{by_op[op]}", flush=True)
    print("per max operand width:", flush=True)
    for w in sorted(by_width):
        print(f"   {w:2d}: {by_width[w] - bad_by_width[w]}/{by_width[w]}", flush=True)
    for a, op, b, out in mismatches:
        print(f"  MISMATCH {a}{op}{b} -> {out!r} != {expected(a, op, b)}", flush=True)
    return len(mismatches)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", required=True, help="bundle directory")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-digits", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    examples = make_examples(args.n, args.max_digits, args.seed)
    results = run(args.path, examples, batch_size=args.batch_size, device=args.device)
    n_bad = report(results)
    raise SystemExit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
