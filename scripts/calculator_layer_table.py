"""Layer / step / parameter table for the four calculators across digit counts.

One record per (implementation, ``max_digits``), three separate cost
dimensions — never conflated:

* ``layers`` — the layer count of a real ``optimize=2`` compile of the
  cell at the family's one canonical geometry (``D_MODEL`` / ``D_HEAD`` /
  ``D_HIDDEN`` in ``examples/_calculator_common.py``) — witnessed per
  cell, never derived.  Each built row also records ``floor``, the
  ``critical_path_layers`` of the graph after ``lower()`` with both
  collapse passes (the exact width-independent DAG minimum), as a
  cross-check: ``layers == floor`` certifies the schedule optimal for
  the graph the scheduler was given, and ``layers != floor`` is a
  finding to report, not an error — expected for calculator_memorize
  (its FFN fact bank is capacity-bound at every geometry, so its
  13-layer dependency floor is never attainable) and possible within ±2
  for calculator_advanced's borderline-collapse cells (below).  A row
  whose build is refused extrapolates the module's measured
  ``compiled_layers(n)`` law as ``layers_extrapolated``
  (calculator_memorize: ``14 + ceil(facts / d_hidden)``); a row that
  builds but has NO schedule at the canonical width records
  ``compile_error`` instead — witnessed refusal, not extrapolation
  (calculator_scratchpad's ~n³ live width outgrows d=8192 at large n,
  so its flat 16-layer floor is unattainable there and the table says
  so rather than quoting it).  This is THE
  depth number; the nonlinear-op count it replaced both under- and
  over-counted (attention→MLP pairing on one side, surviving linear
  glue on the other).
* ``steps`` — worst-case decode steps to finish the answer.  Direct
  emitters pay up to 2n product digits plus the terminating <eos>
  (``2n + 1``); the scratchpad additionally streams its serial work as
  thinking tokens first (``decode_steps(n)`` = 8n + 3).
* ``params`` — weight count of the lowered graph: every FFN / Attn /
  Linear / literal tensor the compiled model actually carries, before
  geometry zero-padding.  Not fully geometry-free: attention Q/K rows
  are ``d_head`` wide, so the number is tied to the family ``D_HEAD``
  (the 2026-07-20 d_head 32→64 rebuild moved each direct emitter's
  count by a few thousand).  A build refused for a structural cap records
  its ``build_error`` instead, plus ``params_extrapolated`` where the
  module provides a validated closed form
  (``calculator_memorize.n_params``).

CPU only, and CPU-hungry — collection belongs on Modal.  Regenerate the
committed ``docs/calculator_layer_table.json`` with::

    uv run modal run modal_layer_table.py

which fans the (impl, n) cells out across Modal workers (one witnessed
compile per cell) and only writes the JSON locally.  Running this module
directly computes everything on the local machine — fine for a single
--ns cell, unkind for the full sweep.  It can also read differently:
calculator_advanced's n=2,3 cells sit on a borderline collapse
certification (a compress cascade whose measured deviation is within
~13% of the 1e-3 budget), and the oracle sweep's fp32 reduction order
differs across BLAS environments — the local sweep collapses it where
Modal declines it (a ±2-layer read on those cells).  The same flips can
move ``params``: a cascade that certifies on one environment is replaced
by a leaner synthesized staircase (advanced n=5 reads 5,136,237 on Modal
— two 6-member cascades collapse — vs 5,228,969 locally, where their
emission checks miss the 1e-3 budget by 1.05–1.51e-3).  The committed
numbers are the Modal environment's; the instability class is the same
one ``docs/numerical_noise_findings.md`` records for the staircase noise
measurements.
"""

import argparse
import importlib
import json
import os
import time
import warnings

from examples._calculator_common import D_HEAD, D_HIDDEN, D_MODEL
from torchwright.compiler.export import compile_headless
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
# Canonical-geometry collapse cap; the synthesized staircases top out well
# below it, so wider caps give the same floor.
LANE_CAP = D_HIDDEN // 4

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


def table_config(ns) -> dict:
    """The committed JSON's config block: sweep + the witnessed geometry."""
    return {
        "ns": list(ns),
        "geometry": {
            "d": D_MODEL,
            "d_hidden": D_HIDDEN,
            "d_head": D_HEAD,
            "optimize": 2,
        },
        "lane_cap": LANE_CAP,
    }


def collect_cell(impl_name: str, n: int) -> dict:
    """Measure one (impl, n) cell — build, floor, and the witnessed compile.

    Pure CPU, no artifacts.  A replayed schedule would not witness anything
    (the cache fingerprint ignores weights), so the schedule cache is
    disabled for the compile.
    """
    os.environ.pop("TW_SCHEDULE_CACHE_DIR", None)
    impl = importlib.import_module(f"examples.{impl_name}")
    t0 = time.time()
    try:
        out, embedding = impl.create_network_parts(max_digits=n)
    except Exception as exc:  # structural caps (slow planes, fact table)
        row = {"n": n, "build_error": str(exc).splitlines()[0]}
        if hasattr(impl, "n_params"):
            row["params_extrapolated"] = impl.n_params(n)
        if hasattr(impl, "compiled_layers"):
            row["layers_extrapolated"] = impl.compiled_layers(n, d_hidden=D_HIDDEN)
        print(
            f"[{impl_name} n={n}] build refused: " f"{row['build_error'][:80]}",
            flush=True,
        )
        return row
    lowered = lower(
        out,
        collapse_univariate=True,
        collapse_pl=True,
        collapse_lane_cap=LANE_CAP,
    )
    row = {
        "n": n,
        "steps": impl.decode_steps(n) if hasattr(impl, "decode_steps") else 2 * n + 1,
        "params": graph_params(lowered.output_node),
        "floor": critical_path_layers(lowered.output_node),
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compiled = compile_headless(
                out,
                d=D_MODEL,
                d_head=D_HEAD,
                d_hidden=D_HIDDEN,
                optimize=2,
                bias=False,
                output_layout_source=embedding,
            )
        # A witnessed layer count must say when CP-SAT gave up within its
        # optimize=2 budget and the heuristic incumbent shipped instead —
        # that cell's number is a valid compile but not a solver optimum.
        fallbacks = [
            str(w.message).splitlines()[0] for w in caught if "CP-SAT" in str(w.message)
        ]
    except RuntimeError as exc:
        # A geometry with no findable schedule is a finding, not a crash —
        # record it; the floor stays as the cell's only depth information.
        row["compile_error"] = str(exc).splitlines()[0]
        print(
            f"[{impl_name} n={n}] NO SCHEDULE ({time.time() - t0:.0f}s): "
            f"{row['compile_error'][:80]}",
            flush=True,
        )
        return row
    row["layers"] = compiled.n_layers
    # Reorder for the committed JSON: layers first (THE depth number).
    row = {k: row[k] for k in ("n", "layers", "steps", "params", "floor")}
    if fallbacks:
        row["compile_warning"] = fallbacks[0]
    print(
        f"[{impl_name} n={n}] layers={row['layers']} (floor {row['floor']}) "
        f"steps={row['steps']} params={row['params']:,} "
        f"({time.time() - t0:.0f}s)" + (" [CP-SAT fallback]" if fallbacks else ""),
        flush=True,
    )
    return row


def collect(ns) -> dict:
    """Measure every (impl, n) cell serially (the local, single-cell path;
    the Modal entrypoint fans cells out instead)."""
    results: dict = {impl: [] for impl in IMPLS}
    for impl_name in IMPLS:
        for n in ns:
            results[impl_name].append(collect_cell(impl_name, n))
    return {"config": table_config(ns), "results": results}


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
        if key == "layers" and "compile_error" in row:
            return "✗ no schedule"
        return "—"

    lines = []
    for key, title, fmt, width in (
        ("layers", "compiled layers (witnessed; () = extrapolated)", "{:,}".format, 26),
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
