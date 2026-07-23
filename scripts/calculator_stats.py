"""Compiled-calculator statistics (read, don't edit).

Prints the headline numbers for the compiled calculator, all derived from a
real compile of ``examples.calculator_simple``:

* the whole calculator's depth / width / residual occupancy (from the ONNX
  debug sidecar);
* each algorithm's standalone layer count (add / subtract / compare /
  multiply compiled on their own);
* a compiled one-hot lookup printed back as the arithmetic table it *is* —
  the single-digit addition grid (reconstructed by evaluating the compiled
  op) and the raw carry→digit selection matrix.

Run locally (CPU is fine, no GPU needed)::

    uv run python -m scripts.calculator_stats
"""

import json
import tempfile
from pathlib import Path
from typing import cast

import torch

import examples.calculator_simple as calc
from examples._calculator_common import _CARRY_W, _NO, _state
from examples.calculator_simple import (
    add_digit_seqs,
    compare_digit_seqs,
    multiply_digit_seqs,
    subtract_digit_seqs,
)
from torchwright.compiler.export import compile_headless, compile_to_onnx
from torchwright.graph import Concatenate, Node
from torchwright.ops.inout_nodes import create_input, create_onehot_embedding
from torchwright.ops.relu.onehot_table import onehot_lookup

D = 1024
D_HEAD = 16
MAX_DIGITS = 3


def _peak_residual_occupancy(sidecar: dict) -> int:
    """Max residual columns simultaneously live across all sublayer snapshots."""
    peak = 0
    for state in sidecar["states"]:
        used = sum(width for ranges in state["nodes"].values() for _, width in ranges)
        peak = max(peak, used)
    return peak


def whole_calculator_stats() -> None:
    print(f"== whole calculator (max_digits={MAX_DIGITS}) ==")
    output_node, embedding = calc.create_network_parts(max_digits=MAX_DIGITS)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "calculator.onnx")
        artifact = compile_to_onnx(
            output_node,
            embedding,
            path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )
        with Path(cast("str", artifact.debug_path)).open() as f:
            sidecar = json.load(f)

    peak = _peak_residual_occupancy(sidecar)
    print(f"  layers              : {artifact.n_layers}")
    print(f"  d (residual width)  : {sidecar['d']}")
    print(f"  d_head              : {sidecar['d_head']}")
    print(f"  peak MLP hidden     : {max(sidecar['d_hidden'])}")
    print(f"  graph nodes         : {len(sidecar['nodes'])}")
    print(f"  d_embed / vocab     : {len(calc.CALC_VOCAB)}")
    print(
        f"  peak residual cols  : {peak} / {sidecar['d']} "
        f"({100.0 * peak / sidecar['d']:.1f}% occupancy)"
    )


def per_operation_layers() -> None:
    print("\n== per-operation depth (each algorithm compiled alone) ==")
    embedding = create_onehot_embedding(calc.CALC_VOCAB)
    d_embed = embedding.d_embed

    def operand(prefix: str, n: int) -> list[Node]:
        return [
            create_input(f"{prefix}{i}", d_embed, value_range=(0.0, 1.0))
            for i in range(n)
        ]

    n = MAX_DIGITS
    a, b = operand("a", n), operand("b", n)
    a1, b1 = operand("c", n + 1), operand("d", n + 1)  # padded operands for add
    ops = {
        "add": Concatenate(add_digit_seqs(embedding, a1, b1)),
        "subtract": Concatenate(subtract_digit_seqs(embedding, a, b)),
        "compare": compare_digit_seqs(embedding, a, b),
        "multiply": Concatenate(multiply_digit_seqs(embedding, a, b)),
    }
    for name, out in ops.items():
        compiled = compile_headless(out, d=D, d_head=D_HEAD)
        print(f"  {name:10s}: {compiled._n_layers} layers")


def addition_table_figure() -> None:
    """The compiled single-digit add lookup, reconstructed as a 10x10 grid."""
    print("\n== compiled single-digit addition lookup (carry-in = 0) ==")
    embedding = create_onehot_embedding(calc.CALC_VOCAB)
    d_embed = embedding.d_embed

    digit_table = {}
    for x in range(10):
        for y in range(10):
            for carry in range(2):
                key = torch.cat(
                    [
                        embedding.get_embedding(str(x)),
                        embedding.get_embedding(str(y)),
                        _state(carry, _CARRY_W),
                    ]
                )
                digit_table[key] = embedding.get_embedding(str((x + y + carry) % 10))

    key_node = create_input("key", 2 * d_embed + _CARRY_W, value_range=(0.0, 1.0))
    lookup = onehot_lookup(key_node, digit_table, embedding.get_embedding("0"))

    # Evaluate all 100 (x, y) pairs at carry-in 0 in one batched forward.
    no_carry = _state(_NO, _CARRY_W)
    rows = [
        torch.cat(
            [
                embedding.get_embedding(str(x)),
                embedding.get_embedding(str(y)),
                no_carry,
            ]
        )
        for x in range(10)
        for y in range(10)
    ]
    out = lookup.compute(n_pos=100, input_values={"key": torch.stack(rows)})
    decoded = [embedding.tokenizer.vocab[int(v.argmax())] for v in out]

    print("     " + " ".join(str(y) for y in range(10)))
    for x in range(10):
        cells = " ".join(decoded[x * 10 + y] for y in range(10))
        print(f"  {x}: {cells}")
    print("  (cell = ones digit of x + y; the tens digit rides the carry fold)")


def times_table_figure() -> None:
    """The compiled digit-products lookup, reconstructed as the 10x10 times table.

    Multiply now sums these products per column and carries once.
    """
    print("\n== compiled times-table lookup (digit x digit) ==")
    embedding = create_onehot_embedding(calc.CALC_VOCAB)
    d_embed = embedding.d_embed

    product_table = {}
    for x in range(10):
        for y in range(10):
            key = torch.cat(
                [embedding.get_embedding(str(x)), embedding.get_embedding(str(y))]
            )
            # value = (tens, ones) as plain numbers, exactly as multiply uses it.
            product_table[key] = torch.tensor([float(x * y // 10), float(x * y % 10)])

    key_node = create_input("key", 2 * d_embed, value_range=(0.0, 1.0))
    lookup = onehot_lookup(key_node, product_table, torch.tensor([0.0, 0.0]))

    rows = [
        torch.cat([embedding.get_embedding(str(x)), embedding.get_embedding(str(y))])
        for x in range(10)
        for y in range(10)
    ]
    out = lookup.compute(n_pos=100, input_values={"key": torch.stack(rows)})

    print("      " + "  ".join(str(y) for y in range(10)))
    for x in range(10):
        cell_values = (
            round(out[x * 10 + y, 0].item()) * 10 + round(out[x * 10 + y, 1].item())
            for y in range(10)
        )
        cells = "  ".join(f"{v:2d}" for v in cell_values)
        print(f"  {x}: {cells}")
    print("  (each cell = 10*tens + ones, the compiled (tens, ones) product pair)")


def main() -> None:
    whole_calculator_stats()
    per_operation_layers()
    addition_table_figure()
    times_table_figure()


if __name__ == "__main__":
    main()
