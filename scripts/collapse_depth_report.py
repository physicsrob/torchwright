"""Realized layer-count report for the univariate collapse pass.

Rollout item 6 of docs/univariate_collapse_plan.md: the point of the
pass is fewer layers, so measure it directly.  For each example graph
(the same list :mod:`scripts.measure_fusion_opportunities` measures),
compile with ``collapse_univariate`` off and on at the same ``d`` /
``d_head`` the token tests use and report the compiled layer count
before / after / delta, together with the pass's own collapse/decline
log (per subgraph: source, members, chain depth, emitted lanes — or the
decline reason).

The modeled counterpart (critical-path sublayers under the idealized
"any univariate subgraph collapses to one FFN" cost model) is
``scripts/measure_fusion_opportunities.py --collapse``; this script
measures the realized quantity on the actual compile.

Usage::

    # all examples: collapse report + off/on compile each
    uv run python -m scripts.collapse_depth_report

    # a subset, by substring match on the module name
    uv run python -m scripts.collapse_depth_report --only sort_digits

    # any graph builder without compiling (collapse report only) — the
    # cheap way to iterate on assert placement for a big graph (doom):
    uv run python -m scripts.collapse_depth_report \
        --spec torchwright_doom.inference.compiled_model:build_graph \
        --kwargs '{"d_head": 128, "d_rot": 64, "max_positions": 61440, \
                   "wad_path": "doom1.wad"}' \
        --lower-only --lane-cap 4096
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from typing import Optional, Tuple

from torchwright.compiler.export import compile_headless
from torchwright.compiler.lower import lower
from torchwright.graph import Node

from scripts.measure_fusion_opportunities import _EXAMPLES


def _build_example(mod_name: str) -> Tuple[Node, int, int]:
    """Build one example graph; return ``(output, d, d_head)``.

    ``d`` / ``d_head`` follow the token-test convention
    (tests/compile/forward): the ``_example_onnx`` defaults (1024 / 16)
    unless the example's own test compiles at the module's ``D_MODEL``
    / ``D_HEAD`` (sort_digits_v1, calculator_scratchpad — graphs whose
    residual peak or rotary-head width needs the module value).
    """
    mod = importlib.import_module(mod_name)
    if hasattr(mod, "create_network_parts"):
        output, _embedding = mod.create_network_parts()
    else:
        output = mod.create_network().inp  # Unembedding wraps the node
    if mod_name in (
        "examples.sort_digits_v1",
        "examples.calculator_scratchpad",
    ):
        return output, mod.D_MODEL, mod.D_HEAD
    return output, 1024, 16


def _load_spec(spec: str, kwargs_json: str) -> Node:
    """``module:callable`` loader (same contract as
    measure_fusion_opportunities): the output node is the result itself
    or the first element of a result tuple."""
    mod_name, fn_name = spec.split(":")
    fn = getattr(importlib.import_module(mod_name), fn_name)
    result = fn(**json.loads(kwargs_json))
    return result[0] if isinstance(result, tuple) else result


def _print_collapse_report(output: Node, lane_cap: int) -> None:
    lowered = lower(output, collapse_univariate=True, collapse_lane_cap=lane_cap)
    report = lowered.collapse_report
    print(f"collapse report (lane cap {lane_cap}):")
    if report is None or not report.outcomes:
        print("  (no univariate subgraphs found)")
        return
    for line in report.format().splitlines():
        print(f"  {line}")
    print(
        f"  -> {report.n_collapsed} collapsed, "
        f"{len(report.outcomes) - report.n_collapsed} declined"
    )


def _compile_layers(
    output: Node, *, d: int, d_head: int, device: str, collapse: bool, optimize: int
) -> int:
    t0 = time.perf_counter()
    compiled = compile_headless(
        output,
        d=d,
        d_head=d_head,
        device=device,
        collapse_univariate=collapse,
        optimize=optimize,
    )
    dt = time.perf_counter() - t0
    print(
        f"  compile collapse={'on ' if collapse else 'off'} "
        f"optimize={optimize}: {compiled.n_layers} layers ({dt:.1f}s)"
    )
    return compiled.n_layers


def report_graph(
    output: Node,
    name: str,
    *,
    d: int,
    d_head: int,
    device: str,
    lane_cap: Optional[int] = None,
    lower_only: bool = False,
    optimize: int = 0,
) -> None:
    # forward_compile's own cap formula (d_hidden defaults to d) unless
    # overridden — so the report matches what a compile would do.
    cap = lane_cap if lane_cap is not None else d // 4
    dims = "" if lower_only else f" (d={d}, d_head={d_head})"
    print(f"\n=== {name}{dims} ===")
    _print_collapse_report(output, cap)
    if lower_only:
        return
    n_off = _compile_layers(
        output, d=d, d_head=d_head, device=device, collapse=False, optimize=optimize
    )
    n_on = _compile_layers(
        output, d=d, d_head=d_head, device=device, collapse=True, optimize=optimize
    )
    print(f"n_layers: off={n_off} on={n_on} delta={n_on - n_off:+d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", help="module:callable building the graph")
    ap.add_argument("--kwargs", default="{}", help="JSON kwargs for the callable")
    ap.add_argument(
        "--lower-only",
        action="store_true",
        help="skip the off/on compiles; print only the collapse report",
    )
    ap.add_argument(
        "--lane-cap",
        type=int,
        default=None,
        help="collapse lane cap (default: d//4, forward_compile's formula; "
        "--lower-only with no --d defaults to 4096)",
    )
    ap.add_argument("--d", type=int, default=None, help="residual width for --spec")
    ap.add_argument("--d-head", type=int, default=64, help="head width for --spec")
    ap.add_argument(
        "--device", default="cpu", help="compile device (cpu default, cuda on Modal)"
    )
    ap.add_argument(
        "--only",
        default=None,
        help="substring filter on the example module names",
    )
    ap.add_argument(
        "--optimize",
        type=int,
        default=0,
        help="forward_compile optimize level for the off/on compiles "
        "(0 greedy; 2 = the production CP-SAT schedule)",
    )
    args = ap.parse_args()

    if args.spec:
        output = _load_spec(args.spec, args.kwargs)
        if args.lower_only:
            report_graph(
                output,
                args.spec,
                d=args.d or 0,
                d_head=args.d_head,
                device=args.device,
                lane_cap=args.lane_cap if args.lane_cap is not None else 4096,
                lower_only=True,
            )
            return
        if args.d is None:
            ap.error("--spec without --lower-only requires --d")
        report_graph(
            output,
            args.spec,
            d=args.d,
            d_head=args.d_head,
            device=args.device,
            lane_cap=args.lane_cap,
            optimize=args.optimize,
        )
        return

    for mod_name in _EXAMPLES:
        if args.only and args.only not in mod_name:
            continue
        output, d, d_head = _build_example(mod_name)
        report_graph(
            output,
            mod_name,
            d=d,
            d_head=d_head,
            device=args.device,
            lane_cap=args.lane_cap,
            lower_only=args.lower_only,
            optimize=args.optimize,
        )


if __name__ == "__main__":
    main()
