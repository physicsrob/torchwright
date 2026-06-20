"""Depth-vs-digit-count scaling for the one-hot arithmetic ops.

For each registered arithmetic implementation, each operation
(``add`` / ``subtract`` / ``multiply``), and a sweep of digit counts ``n``,
this records two structural metrics of the standalone ``op(a, b)`` graph:

* **depth** — the *critical-path* length: the longest chain of neuron-producing
  nodes (ReLU lookups / attention) from the inputs to the outputs.  This is the
  number of transformer layers the algorithm needs *given enough width per
  layer* — i.e. its true depth complexity.
* **size** — the total neuron count: the sum of every ReLU node's width.  This
  is the algorithm's total compute width and is proportional to the compiled
  parameter count (``params ≈ 2·d·neurons``), so it is the "width cost" axis.

Both are read directly from the graph — **no compile** — which is what makes
the figure trustworthy.  The compiled layer count is *not* used: it is
residual-width-serialized (when the residual width ``d`` is tight the scheduler
frees and recomputes columns, multiplying the layer count by a factor that
grows with ``n``), so it measures the width budget, not the algorithm's depth.
De-serializing it would need a ``d`` large enough to exceed memory.  The
critical-path depth is ``d``-independent, instant, and OOM-free, so the full
sweep — multiply included — runs with no width caps.

The two implementations tell the asymptotic story the blog's payoff figure
makes visible:

* ``simple`` (``onehot_arithmetic``) — legible serial carry/borrow folds and a
  column-sum multiply.  Depth is **linear** in ``n`` (add ``n``, multiply
  ``~2n``).
* ``advanced`` (``onehot_arithmetic_fast``) — carry-lookahead add/subtract and
  carry-save (Wallace) multiply.  Depth is **logarithmic** in ``n``, at a higher
  neuron (width) cost.

Each operand is fed at its declared ``n`` digits with no extra padding, so
``n`` means the same thing across all three ops — directly comparable curves.

Output: a JSON file (the data the blog/vizkit consumes), a printed table, and
an optional matplotlib PNG (depth-vs-n and neurons-vs-n).

Run locally (CPU is fine, no GPU needed)::

    uv run python -m scripts.arithmetic_scaling
    uv run python -m scripts.arithmetic_scaling --plot
    uv run python -m scripts.arithmetic_scaling --digits 1,2,3,4
"""

import argparse
import json

from torchwright.ops import onehot_arithmetic, onehot_arithmetic_fast
from torchwright.ops.inout_nodes import create_literal_value, create_onehot_embedding

# Registered implementations, in figure order: the legible serial folds vs the
# depth-optimized carry-lookahead / carry-save versions.
IMPLEMENTATIONS = {
    "simple": onehot_arithmetic,
    "advanced": onehot_arithmetic_fast,
}

# Each op -> the digit-sequence function name it exposes.
OPS = {
    "add": "add_digit_seqs",
    "subtract": "subtract_digit_seqs",
    "multiply": "multiply_digit_seqs",
}

DEFAULT_DIGIT_SWEEP = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
# The multiply graph build is ~O(n^2) nodes (and seconds of construction at the
# top end); the log-vs-linear divergence is already unmistakable by n=16
# (advanced 16 layers vs simple 33), so cap it there to keep the harness ~2 min.
DEFAULT_MULTIPLY_CAP = 16

# Operands are one-hot digits 0..9; the ops only ever look up digit rows.
VOCAB = [str(d) for d in range(10)]


def log(message: str) -> None:
    """Emit a progress line."""
    print(f"[scaling] {message}")


def _is_neuron(node) -> bool:
    """A node that lowers to a hidden ReLU layer (or an attention sublayer)."""
    t = type(node).__name__
    return "ReLU" in t or "Attn" in t


def _walk(outputs):
    """Yield every unique node reachable from ``outputs`` (post-order ids)."""
    seen = set()
    stack = list(outputs)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(getattr(node, "inputs", None) or [])


def critical_path_depth(outputs) -> int:
    """Longest chain of neuron-producing nodes from any input to ``outputs``.

    This is the transformer-layer depth the algorithm needs with enough width
    per layer — its true depth complexity, independent of the residual width.
    Computed with an explicit stack (iterative post-order) so deep graphs do
    not overflow Python's recursion limit.
    """
    memo = {}
    for root in outputs:
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                ins = getattr(node, "inputs", None) or []
                best = max((memo[id(i)] for i in ins), default=0)
                memo[id(node)] = best + (1 if _is_neuron(node) else 0)
                continue
            if id(node) in memo:
                continue
            stack.append((node, True))
            for i in getattr(node, "inputs", None) or []:
                if id(i) not in memo:
                    stack.append((i, False))
    return max(memo[id(o)] for o in outputs)


def total_neurons(outputs) -> int:
    """Sum of every neuron-producing node's width (total compute, ~params/2d)."""
    return sum(len(node) for node in _walk(outputs) if _is_neuron(node))


def _operand(embedding, n):
    return [create_literal_value(embedding.get_embedding("1")) for _ in range(n)]


def measure_op(arith, fn_name, n, embedding):
    """Build ``op(a, b)`` over two n-digit operands; return (depth, neurons)."""
    fn = getattr(arith, fn_name)
    outputs = fn(embedding, _operand(embedding, n), _operand(embedding, n))
    return critical_path_depth(outputs), total_neurons(outputs)


def _sweep_for(op_name, digit_sweep, multiply_cap):
    if op_name == "multiply":
        capped = [n for n in digit_sweep if n <= multiply_cap]
        if capped != digit_sweep:
            log(
                f"multiply: capping digit sweep at n<={multiply_cap} "
                f"(O(n^2) graph build); full sweep is {digit_sweep}, running {capped}"
            )
        return capped
    return digit_sweep


def run(digit_sweep, multiply_cap=DEFAULT_MULTIPLY_CAP):
    """Build every (impl, op, n) graph and collect depth/size records."""
    embedding = create_onehot_embedding(VOCAB)
    results = {}
    for impl_name, arith in IMPLEMENTATIONS.items():
        results[impl_name] = {}
        for op_name, fn_name in OPS.items():
            records = []
            for n in _sweep_for(op_name, digit_sweep, multiply_cap):
                depth, neurons = measure_op(arith, fn_name, n, embedding)
                records.append({"n": n, "depth": depth, "neurons": neurons})
                log(f"{impl_name}/{op_name} n={n}: depth={depth}, neurons={neurons:,}")
            results[impl_name][op_name] = records
    return results


def print_table(results) -> None:
    print("\n== one-hot arithmetic scaling (critical-path depth / total neurons) ==")
    for impl_name, ops in results.items():
        print(f"\nimpl = {impl_name}")
        for op_name, records in ops.items():
            print(f"  {op_name}")
            print(f"    {'n':>4}  {'depth':>6}  {'neurons':>12}")
            for r in records:
                print(f"    {r['n']:>4}  {r['depth']:>6}  {r['neurons']:>12,}")


def build_payload(results, digit_sweep, multiply_cap):
    return {
        "config": {
            "digit_sweep": digit_sweep,
            "multiply_cap": multiply_cap,
            "depth_metric": "critical-path length over neuron-producing nodes",
            "size_metric": "total neuron count (~ params / 2d)",
        },
        "results": results,
    }


def write_json(payload, path) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote scaling data -> {path}")


def _plot_series(impl_name, ops):
    """(label, records) per impl, collapsing add+subtract when they are equal.

    Add and subtract are the same structure over same-shape tables, so they have
    identical depth and size — plotting them as two coincident lines just hides
    one under the other.  When the records match exactly, emit a single
    ``add+subtract`` series; otherwise keep them apart.
    """
    add, sub = ops.get("add"), ops.get("subtract")
    collapse = add is not None and sub is not None and add == sub
    for op_name, records in ops.items():
        if collapse and op_name == "subtract":
            continue
        name = "add+subtract" if collapse and op_name == "add" else op_name
        yield f"{impl_name}/{name}", records


def write_plot(results, path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed; skipping PNG (JSON is still written)")
        return

    fig, (ax_depth, ax_size) = plt.subplots(1, 2, figsize=(12, 5))
    for impl_name, ops in results.items():
        for label, records in _plot_series(impl_name, ops):
            ns = [r["n"] for r in records]
            ax_depth.plot(ns, [r["depth"] for r in records], marker="o", label=label)
            ax_size.plot(ns, [r["neurons"] for r in records], marker="o", label=label)
    ax_depth.set_ylabel("critical-path depth (layers)")
    ax_size.set_ylabel("neurons (~ params / 2d)")
    ax_size.set_yscale("log")
    for ax in (ax_depth, ax_size):
        ax.set_xlabel("digit count n")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("One-hot arithmetic: depth and size vs digit count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    log(f"wrote scaling plot -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default="docs/arithmetic_scaling.json", help="JSON output path"
    )
    parser.add_argument("--plot", action="store_true", help="also write a PNG")
    parser.add_argument(
        "--plot-out", default="docs/arithmetic_scaling.png", help="PNG output path"
    )
    parser.add_argument(
        "--digits",
        default=None,
        help="comma-separated digit sweep (default: %s)" % DEFAULT_DIGIT_SWEEP,
    )
    parser.add_argument(
        "--multiply-cap",
        type=int,
        default=DEFAULT_MULTIPLY_CAP,
        help="cap the multiply sweep at this n (default: %d)" % DEFAULT_MULTIPLY_CAP,
    )
    args = parser.parse_args()

    digit_sweep = (
        [int(x) for x in args.digits.split(",")] if args.digits else DEFAULT_DIGIT_SWEEP
    )

    results = run(digit_sweep, args.multiply_cap)
    print_table(results)
    write_json(build_payload(results, digit_sweep, args.multiply_cap), args.out)
    if args.plot:
        write_plot(results, args.plot_out)


if __name__ == "__main__":
    main()
