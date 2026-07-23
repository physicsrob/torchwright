"""Depth-vs-digit-count scaling for the one-hot arithmetic ops.

For each registered arithmetic implementation, each operation
(``add`` / ``subtract`` / ``multiply``), and a sweep of digit counts ``n``,
this records two structural metrics of the standalone ``op(a, b)`` graph:

* **depth** — the *critical-path* length: the longest chain of neuron-producing
  nodes (FFN units / ReLU lookups / attention) from the inputs to the outputs.
  This is the number of transformer layers the algorithm needs *given enough
  width per layer* — i.e. its true depth complexity.
* **size** — the total neuron count: the sum of every FFN's lane count (plus
  any ReLU node's width).  This is the algorithm's total compute width and is
  roughly proportional to the compiled parameter count (``params ≈ 2·d·neurons``
  for degenerate/relu lanes, ``≈ 3·d`` per gated swish lane), so it is the
  "width cost" axis.

Both are read directly from the graph — **no compile** — which is what makes
the figure trustworthy.  The compiled layer count is *not* used: it is
residual-width-serialized (when the residual width ``d`` is tight the scheduler
frees and recomputes columns, multiplying the layer count by a factor that
grows with ``n``), so it measures the width budget, not the algorithm's depth.
De-serializing it would need a ``d`` large enough to exceed memory.  The
critical-path depth is ``d``-independent, instant, and OOM-free, so the full
sweep — multiply included — runs with no width caps.

The two implementations tell the asymptotic story the scaling figure
makes visible:

* ``simple`` (``examples.calculator_simple``) — legible serial carry/borrow
  folds and a column-sum multiply.  Depth is **linear** in ``n`` (add ``n``,
  multiply ``~2n``).
* ``advanced`` (``examples.calculator_advanced``) — carry-lookahead add/subtract
  and carry-save (Wallace) multiply.  Depth is **logarithmic** in ``n``, at a
  higher neuron (width) cost.

Each operand is fed at its declared ``n`` digits with no extra padding, so
``n`` means the same thing across all three ops — directly comparable curves.

On top of those per-op *kernel* metrics, a second pass measures **end-to-end
model depth** — ``critical_path_depth`` of the whole ``create_network_parts(n)``
graph (parse + all three ops + dispatch + emit), one number per implementation.
This is the figure that makes the payoff visible: ``simple`` and ``advanced``
grow with their in-graph arithmetic alone (the shared parse, comparison, and
leading-zero trim are all constant-depth — ``simple``'s serial folds are
linear, ``advanced``'s carry-lookahead / carry-save ~log), while
``examples.calculator_scratchpad`` — which streams the
serial carry/borrow/comparison work out as "thinking" tokens — stays **flat**
in ``n`` and pays the cost in decode *steps* (worst case ``8n+3``, the
multiply transcript) instead.

Output: a JSON file (docs/arithmetic_scaling.json), printed tables, and an
optional matplotlib PNG (per-op kernel depth and end-to-end model depth, with a
colour per implementation shared across both panels).  Decode steps stay in the
JSON (``decode_steps``) but are no longer a plotted panel.

Run locally (CPU is fine, no GPU needed)::

    uv run python -m scripts.arithmetic_scaling
    uv run python -m scripts.arithmetic_scaling --plot
    uv run python -m scripts.arithmetic_scaling --digits 1,2,3,4
"""

import argparse
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

from examples import calculator_advanced, calculator_scratchpad, calculator_simple
from torchwright.graph import Embedding, Node, RopeConfig
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_onehot_embedding,
    create_rope_config,
)

if TYPE_CHECKING:
    from torchwright.graph.ffn import FFN

ScratchpadOp = Callable[
    [RopeConfig, Embedding, list[Node], list[Node], int, Node],
    tuple[list[Node], list[Node]],
]
Records = list[dict[str, int]]

# Registered implementations, in figure order: the legible serial folds vs the
# depth-optimized carry-lookahead / carry-save versions.  Each calculator module
# exposes its own ``add_digit_seqs`` / ``subtract_digit_seqs`` /
# ``multiply_digit_seqs`` at module level.
IMPLEMENTATIONS = {
    "simple": calculator_simple,
    "advanced": calculator_advanced,
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

# End-to-end *model* depth (parse + all three ops + dispatch + emit), one number
# per implementation per digit count — the figure that makes the scratchpad's
# payoff visible.  All three modules expose ``create_network_parts(max_digits)``.
# ``simple`` / ``advanced`` grow with their arithmetic alone (the shared
# parse, comparison, and leading-zero trim are constant-depth);
# ``scratchpad`` stays flat and pays in decode *steps* instead.
MODEL_IMPLEMENTATIONS = {
    "simple": calculator_simple,
    "advanced": calculator_advanced,
    "scratchpad": calculator_scratchpad,
}

# Each model build includes the O(n^2) multiply graph, so the end-to-end sweep
# is capped low; the flat-vs-linear divergence is already unmistakable by n=6.
DEFAULT_MODEL_DIGIT_SWEEP = [1, 2, 3, 4, 5, 6]


def log(message: str) -> None:
    """Emit a progress line."""
    print(f"[scaling] {message}")


def _is_neuron(node: Node) -> bool:
    """A node that lowers to hidden compute.

    Includes an FFN (the packable lane unit the swiglu ops build), a
    legacy ReLU layer, or an attention sublayer.
    """
    t = type(node).__name__
    return t == "FFN" or "ReLU" in t or "Attn" in t


def _neuron_width(node: Node) -> int:
    """The hidden width a neuron node contributes.

    An FFN's lane count (``gate_proj`` rows), otherwise the node's
    output width.
    """
    if type(node).__name__ == "FFN":
        return cast("FFN", node).gate_proj.shape[0]
    return len(node)


def _walk(outputs: Sequence[Node]) -> Iterator[Node]:
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


def critical_path_depth(outputs: Sequence[Node]) -> int:
    """Longest chain of neuron-producing nodes from any input to ``outputs``.

    This is the transformer-layer depth the algorithm needs with enough width
    per layer — its true depth complexity, independent of the residual width.
    Computed with an explicit stack (iterative post-order) so deep graphs do
    not overflow Python's recursion limit.
    """
    memo: dict = {}
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
            stack.extend(
                (i, False)
                for i in getattr(node, "inputs", None) or []
                if id(i) not in memo
            )
    return max(memo[id(o)] for o in outputs)


def total_neurons(outputs: Sequence[Node]) -> int:
    """Sum of every neuron-producing node's width (total compute, ~params/2d)."""
    return sum(_neuron_width(node) for node in _walk(outputs) if _is_neuron(node))


def _operand(embedding: Embedding, n: int) -> list[Node]:
    return [create_literal_value(embedding.get_embedding("1")) for _ in range(n)]


def measure_op(
    arith: ModuleType, fn_name: str, n: int, embedding: Embedding
) -> tuple[int, int]:
    """Build ``op(a, b)`` over two n-digit operands; return (depth, neurons)."""
    fn = getattr(arith, fn_name)
    outputs = fn(embedding, _operand(embedding, n), _operand(embedding, n))
    return critical_path_depth(outputs), total_neurons(outputs)


# The scratchpad's per-op "kernel" is its streamed sweep, not a digit-seq
# transform: it needs the scratch vocab, a rope config, and the decode-step
# counter, and returns emitted-token nodes, so it can't go through
# ``measure_op``.  These are its private op builders ``(rope, embedding, A, B,
# n, steps_since) -> (thinking, answer)``; reaching for them keeps the kernel
# panel able to show all three implementations without widening the
# scratchpad's public surface for a viz.
_SCRATCHPAD_OPS = {
    "add": calculator_scratchpad._add_op,
    "subtract": calculator_scratchpad._sub_op,
    "multiply": calculator_scratchpad._mul_op,
}

# The streamed kernel's *graph build* is ~O(n^3): n^2 partial products plus the
# pointer-gather trim, which derives each answer digit once into a scratch region
# and reads it back with a single one-hot attention per slot (no per-slot
# materialization of all N digits — the old ~O(n^4) term).  Its depth is flat in
# n, so a short sweep shows the whole story; the cap is comfortably affordable now.
SCRATCHPAD_OP_CAP = 12

# Under the RoPE end-state every head rotates, so content-matching heads route
# their content columns onto the slowest rotary planes — capping content width
# at d_head/2 (graph/rope.py::place_on_slow_planes).  The scratchpad's answer
# gather matches a one-hot over all answer digits at once, and an n-digit
# multiply has 2n of them, so its kernel only builds while 2n <= D_HEAD/2.
# Add/subtract answers have n+1 digits and clear the budget up to the op cap.
SCRATCHPAD_MULTIPLY_PLANE_CAP = calculator_scratchpad.D_HEAD // 4


def measure_scratchpad_op(op_fn: ScratchpadOp, n: int) -> tuple[int, int]:
    """Build the streamed scratchpad ``op(a, b)`` kernel; return (depth, neurons).

    Depth is flat in ``n`` (the streaming moves the serial recurrence onto the
    decode-step axis); only the neuron count grows.
    """
    embedding = create_onehot_embedding(calculator_scratchpad.scratch_vocab(n))
    rope = create_rope_config(
        d_head=calculator_scratchpad.D_HEAD,
        max_positions=calculator_scratchpad.MAX_POSITIONS,
    )
    steps_since = calculator_scratchpad._steps_since_newline(
        rope, embedding, max_gap=calculator_scratchpad.decode_steps(n) + 2
    )
    a = _operand(embedding, n)
    b = _operand(embedding, n)
    thinking, answer = op_fn(rope, embedding, a, b, n, steps_since)
    outputs = thinking + answer
    return critical_path_depth(outputs), total_neurons(outputs)


def _sweep_for(op_name: str, digit_sweep: list[int], multiply_cap: int) -> list[int]:
    if op_name == "multiply":
        capped = [n for n in digit_sweep if n <= multiply_cap]
        if capped != digit_sweep:
            log(
                f"multiply: capping digit sweep at n<={multiply_cap} "
                f"(O(n^2) graph build); full sweep is {digit_sweep}, running {capped}"
            )
        return capped
    return digit_sweep


def run(
    digit_sweep: list[int], multiply_cap: int = DEFAULT_MULTIPLY_CAP
) -> dict[str, dict[str, Records]]:
    """Build every (impl, op, n) graph and collect depth/size records."""
    embedding = create_onehot_embedding(VOCAB)
    results: dict[str, dict[str, Records]] = {}
    for impl_name, arith in IMPLEMENTATIONS.items():
        results[impl_name] = {}
        for op_name, fn_name in OPS.items():
            records: Records = []
            for n in _sweep_for(op_name, digit_sweep, multiply_cap):
                depth, neurons = measure_op(arith, fn_name, n, embedding)
                records.append({"n": n, "depth": depth, "neurons": neurons})
                log(f"{impl_name}/{op_name} n={n}: depth={depth}, neurons={neurons:,}")
            results[impl_name][op_name] = records

    # The scratchpad's per-op kernels (flat depth) alongside the legible serial /
    # depth-optimized ones, so all three implementations appear in the kernel
    # panel too — not only the end-to-end model panel.
    results["scratchpad"] = {}
    for op_name, op_fn in _SCRATCHPAD_OPS.items():
        records = []
        cap = SCRATCHPAD_OP_CAP
        if op_name == "multiply" and cap > SCRATCHPAD_MULTIPLY_PLANE_CAP:
            cap = SCRATCHPAD_MULTIPLY_PLANE_CAP
            log(
                f"scratchpad/multiply: capping at n<={cap} — the answer gather's "
                f"content width (2n digits) is limited to D_HEAD/2 = "
                f"{calculator_scratchpad.D_HEAD // 2} slow rotary planes"
            )
        sweep = [n for n in _sweep_for(op_name, digit_sweep, multiply_cap) if n <= cap]
        for n in sweep:
            depth, neurons = measure_scratchpad_op(op_fn, n)
            records.append({"n": n, "depth": depth, "neurons": neurons})
            log(f"scratchpad/{op_name} n={n}: depth={depth}, neurons={neurons:,}")
        results["scratchpad"][op_name] = records
    return results


def measure_model(impl: ModuleType, n: int) -> tuple[int, int]:
    """Build the whole model ``create_network_parts(n)``.

    Returns its end-to-end (depth, neurons): parse + arithmetic +
    dispatch + emit, not one op.
    """
    output_node, _ = impl.create_network_parts(max_digits=n)
    return critical_path_depth([output_node]), total_neurons([output_node])


def run_models(digit_sweep: list[int]) -> dict[str, Records]:
    """End-to-end model depth/size for every implementation.

    Also includes the scratchpad worst-case decode-step count (its O(n)
    cost, the axis that grows instead of depth — the multiply
    transcript's length; add/sub stop at their own <eos> earlier).
    """
    results: dict[str, Records] = {}
    for name, impl in MODEL_IMPLEMENTATIONS.items():
        records: Records = []
        for n in digit_sweep:
            depth, neurons = measure_model(impl, n)
            record = {"n": n, "depth": depth, "neurons": neurons}
            if name == "scratchpad":
                record["decode_steps"] = impl.decode_steps(n)
            records.append(record)
            steps = (
                f", steps={record['decode_steps']}" if "decode_steps" in record else ""
            )
            log(f"model {name} n={n}: depth={depth}, neurons={neurons:,}{steps}")
        results[name] = records
    return results


def print_models_table(model_results: dict[str, Records]) -> None:
    print("\n== end-to-end model scaling (whole-model critical-path depth) ==")
    for name, records in model_results.items():
        print(f"\n  model = {name}")
        has_steps = any("decode_steps" in r for r in records)
        header = f"    {'n':>4}  {'depth':>6}  {'neurons':>12}"
        if has_steps:
            header += f"  {'steps':>6}"
        print(header)
        for r in records:
            row = f"    {r['n']:>4}  {r['depth']:>6}  {r['neurons']:>12,}"
            if has_steps:
                row += f"  {r['decode_steps']:>6}"
            print(row)


def print_table(results: dict[str, dict[str, Records]]) -> None:
    print("\n== one-hot arithmetic scaling (critical-path depth / total neurons) ==")
    for impl_name, ops in results.items():
        print(f"\nimpl = {impl_name}")
        for op_name, records in ops.items():
            print(f"  {op_name}")
            print(f"    {'n':>4}  {'depth':>6}  {'neurons':>12}")
            for r in records:
                print(f"    {r['n']:>4}  {r['depth']:>6}  {r['neurons']:>12,}")


def build_payload(
    results: dict[str, dict[str, Records]],
    model_results: dict[str, Records],
    digit_sweep: list[int],
    model_digit_sweep: list[int],
    multiply_cap: int,
) -> dict[str, object]:
    return {
        "config": {
            "digit_sweep": digit_sweep,
            "model_digit_sweep": model_digit_sweep,
            "multiply_cap": multiply_cap,
            "scratchpad_multiply_plane_cap": SCRATCHPAD_MULTIPLY_PLANE_CAP,
            "depth_metric": "critical-path length over neuron-producing nodes",
            "size_metric": (
                "total neuron count (FFN lanes + ReLU widths; "
                "~ params / 2d, gated lanes ~ params / 3d)"
            ),
            "model_depth_metric": (
                "whole-model critical-path depth (parse + ops + dispatch + emit)"
            ),
            "decode_steps_metric": (
                "worst-case decode steps per query (scratchpad: the multiply "
                "transcript, 8n+3; add/sub transcripts hit their own <eos> "
                "earlier and generation stops there)"
            ),
        },
        "results": results,
        "model_results": model_results,
    }


def write_json(payload: dict[str, object], path: str) -> None:
    with Path(path).open("w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote scaling data -> {path}")


def write_plot(
    results: dict[str, dict[str, Records]],
    model_results: dict[str, Records],
    path: str,
) -> None:
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed; skipping PNG (JSON is still written)")
        return

    # One colour per implementation, shared across both panels so the eye maps
    # the same algorithm between the kernel-depth and the model-depth view, and
    # both legends read the same three names.
    impl_names = list(dict.fromkeys(list(results) + list(model_results)))
    cmap = plt.get_cmap("tab10")
    impl_color = {name: cmap(i) for i, name in enumerate(impl_names)}

    fig, (ax_depth, ax_model) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: the multiply kernel's depth — the headline op, one line per
    # implementation (simple linear, advanced log, scratchpad flat), so the
    # legend matches the right panel exactly.
    for impl_name, color in impl_color.items():
        ops = results.get(impl_name)
        if not ops or "multiply" not in ops:
            continue
        records = ops["multiply"]
        ns = [r["n"] for r in records]
        ax_depth.plot(
            ns,
            [r["depth"] for r in records],
            marker="o",
            color=color,
            label=impl_name,
        )
    ax_depth.set_title("arithmetic-kernel depth (multiply)")
    ax_depth.set_ylabel("critical-path depth (layers)")

    # Right (the headline): whole-model depth — scratchpad flat, the legible
    # calculators linear.
    for name, records in model_results.items():
        ns = [r["n"] for r in records]
        ax_model.plot(
            ns,
            [r["depth"] for r in records],
            marker="o",
            color=impl_color[name],
            label=name,
        )
    ax_model.set_title("end-to-end model depth (flat ⇔ scratchpad)")
    ax_model.set_ylabel("model critical-path depth (layers)")

    for ax in (ax_depth, ax_model):
        ax.set_xlabel("digit count n")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("One-hot arithmetic: kernel depth and model depth")
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
        help=f"comma-separated digit sweep (default: {DEFAULT_DIGIT_SWEEP})",
    )
    parser.add_argument(
        "--multiply-cap",
        type=int,
        default=DEFAULT_MULTIPLY_CAP,
        help=f"cap the multiply sweep at this n (default: {DEFAULT_MULTIPLY_CAP})",
    )
    parser.add_argument(
        "--model-digits",
        default=None,
        help=(
            "comma-separated end-to-end model sweep "
            f"(default: {DEFAULT_MODEL_DIGIT_SWEEP})"
        ),
    )
    args = parser.parse_args()

    digit_sweep = (
        [int(x) for x in args.digits.split(",")] if args.digits else DEFAULT_DIGIT_SWEEP
    )
    model_digit_sweep = (
        [int(x) for x in args.model_digits.split(",")]
        if args.model_digits
        else DEFAULT_MODEL_DIGIT_SWEEP
    )

    results = run(digit_sweep, args.multiply_cap)
    model_results = run_models(model_digit_sweep)
    print_table(results)
    print_models_table(model_results)
    write_json(
        build_payload(
            results, model_results, digit_sweep, model_digit_sweep, args.multiply_cap
        ),
        args.out,
    )
    if args.plot:
        write_plot(results, model_results, args.plot_out)


if __name__ == "__main__":
    main()
