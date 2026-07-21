"""Layer / step / parameter table for the four calculators across digit counts.

One record per (implementation, ``max_digits``), three separate cost
dimensions — never conflated:

* ``layers`` — the compiled layer count at saturated width.  For the
  computing variants that equals ``critical_path_layers`` of the graph
  after ``lower()`` with both collapse passes — the exact DAG minimum,
  which ``optimize=2`` compiles land on once width saturates (verified
  for calculator_simple at n=3 and n=6 across d=1024..8192).  A module
  that is capacity-bound at every geometry instead declares its own
  measured ``compiled_layers(n)`` law (calculator_memorize: ``14 +
  ceil(facts / d_hidden)`` at the d_hidden=16384 flagship reference —
  its 13-layer dependency floor is never attainable), and refused rows
  extend it as ``layers_extrapolated``.  This is THE depth number; the
  nonlinear-op count it replaced both under- and over-counted
  (attention→MLP pairing on one side, surviving linear glue on the
  other).
* ``steps`` — worst-case decode steps to finish the answer.  Direct
  emitters pay up to 2n product digits plus the terminating <eos>
  (``2n + 1``); the scratchpad additionally streams its serial work as
  thinking tokens first (``decode_steps(n)`` = 8n + 3).
* ``params`` — weight count of the lowered graph: every FFN / Attn /
  Linear / literal tensor the compiled model actually carries, before
  geometry zero-padding.  A build refused for a structural cap records
  its ``build_error`` instead, plus ``params_extrapolated`` where the
  module provides a validated closed form
  (``calculator_memorize.n_params``).

CPU only, and CPU-hungry — collection belongs on Modal.  Regenerate the
committed ``docs/calculator_layer_table.json`` with::

    uv run modal run modal_layer_table.py

which runs :func:`collect` remotely and only writes the JSON locally.
Running this module directly computes everything on the local machine —
fine for a single --ns cell, unkind for the full sweep.  It can also
read differently: calculator_advanced's n=2,3 cells sit on a borderline
collapse certification (a compress cascade whose measured deviation is
within ~13% of the 1e-3 budget), and the oracle sweep's fp32 reduction
order differs across BLAS environments — the local sweep collapses it
(23/25 layers) where Modal declines it (25/27).  The committed numbers
are the Modal environment's; the instability class is the same one
``docs/numerical_noise_findings.md`` records for the staircase noise
measurements.
"""

import argparse
import importlib
import json
import time

from torchwright.compiler.forward.cpsat_scheduler import critical_path_layers
from torchwright.compiler.lower import lower
from torchwright.compiler.utils import get_ancestor_nodes

IMPLS = [
    "calculator_simple",
    "calculator_advanced",
    "calculator_scratchpad",
    "calculator_memorize",
]
DEFAULT_NS = "2,3,4,5,6,7,8,9,10"
# Flagship-geometry collapse cap (d_hidden 8192 // 4); the synthesized
# staircases top out well below it, so wider caps give the same floor.
LANE_CAP = 2048

# Weight-bearing tensor attributes per graph node type name.
_WEIGHT_ATTRS = {
    "FFN": ("gate_proj", "gate_bias", "up_proj", "up_bias", "out_proj", "out_bias"),
    "Attn": ("query_matrix", "key_matrix", "value_matrix", "output_matrix"),
    "Linear": ("output_matrix", "output_bias"),
    "LiteralValue": ("value",),
    "Embedding": ("table",),
}


def graph_params(output_node) -> int:
    """Total weight count of the graph reachable from ``output_node``."""
    total = 0
    for node in get_ancestor_nodes({output_node}):
        for attr in _WEIGHT_ATTRS.get(type(node).__name__, ()):
            t = getattr(node, attr, None)
            if t is not None:
                total += t.numel()
    return total


def collect(ns) -> dict:
    """Measure every (impl, n) cell; pure CPU, no artifacts."""
    results: dict = {impl: [] for impl in IMPLS}
    for impl_name in IMPLS:
        impl = importlib.import_module(f"examples.{impl_name}")
        for n in ns:
            t0 = time.time()
            try:
                out, _embedding = impl.create_network_parts(max_digits=n)
            except Exception as exc:  # structural caps (slow planes, fact table)
                row = {"n": n, "build_error": str(exc).splitlines()[0]}
                if hasattr(impl, "n_params"):
                    row["params_extrapolated"] = impl.n_params(n)
                if hasattr(impl, "compiled_layers"):
                    row["layers_extrapolated"] = impl.compiled_layers(n)
                results[impl_name].append(row)
                print(
                    f"[{impl_name} n={n}] build refused: " f"{row['build_error'][:80]}",
                    flush=True,
                )
                continue
            lowered = lower(
                out,
                collapse_univariate=True,
                collapse_pl=True,
                collapse_lane_cap=LANE_CAP,
            )
            row = {
                "n": n,
                # A capacity-bound module's own measured law wins over the
                # dependency floor (see the docstring's layers entry).
                "layers": (
                    impl.compiled_layers(n)
                    if hasattr(impl, "compiled_layers")
                    else critical_path_layers(lowered.output_node)
                ),
                "steps": (
                    impl.decode_steps(n) if hasattr(impl, "decode_steps") else 2 * n + 1
                ),
                "params": graph_params(lowered.output_node),
            }
            results[impl_name].append(row)
            print(
                f"[{impl_name} n={n}] layers={row['layers']} "
                f"steps={row['steps']} params={row['params']:,} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return {"config": {"ns": list(ns), "lane_cap": LANE_CAP}, "results": results}


def render(payload: dict) -> str:
    """Three tables — one per cost dimension."""
    ns = payload["config"]["ns"]
    results = payload["results"]
    short = {impl: impl.split("_")[1] for impl in IMPLS}

    def cell(impl, n, key, fmt=str):
        row = next((r for r in results[impl] if r["n"] == n), None)
        if row is None:
            return "—"
        if key in row:
            return fmt(row[key])
        if f"{key}_extrapolated" in row:
            return f"({fmt(row[f'{key}_extrapolated'])})"
        return "—"

    lines = []
    for key, title, fmt, width in (
        ("layers", "compiled layers (() = extrapolated)", "{:,}".format, 26),
        ("steps", "worst-case decode steps", str, 12),
        ("params", "lowered-graph parameters (() = extrapolated)", "{:,}".format, 26),
    ):
        lines.append(f"\n=== {title} ===")
        lines.append(f"{'n':>3} " + "".join(f"{short[i]:>{width}}" for i in IMPLS))
        for n in ns:
            lines.append(
                f"{n:>3} " + "".join(f"{cell(i, n, key, fmt):>{width}}" for i in IMPLS)
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", default=DEFAULT_NS)
    ap.add_argument(
        "--out",
        default="docs/calculator_layer_table.json",
        help="JSON output path (empty string to skip writing)",
    )
    args = ap.parse_args()
    payload = collect([int(x) for x in args.ns.split(",")])
    print(render(payload))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
