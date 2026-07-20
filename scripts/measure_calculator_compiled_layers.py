"""Compiled layer count vs compile widths for the calculator examples.

``scripts.arithmetic_scaling`` measures the *critical-path* depth — the
width-independent floor a graph needs given enough width per layer.  The
*compiled* layer count sits above that floor by whatever the scheduler
must serialize: a tight residual width ``d`` forces column frees and
recomputes, and a tight ``d_hidden`` (which defaults to ``d``) forces
parallel FFN units into separate sublayers.  This script measures that
gap directly: it compiles a calculator at several width configs and
prints ``n_layers`` next to the critical-path floor.

Compiles mirror the publish path (``examples.compile``): ``optimize=0``,
``bias=False`` (the phi3 target), tied output layout.  Pure CPU — run it
on Modal to spare the local machine::

    make modal-run MODULE=scripts.measure_calculator_compiled_layers CPU_ONLY=1
"""

import argparse
import importlib
import time

from scripts.arithmetic_scaling import critical_path_depth
from torchwright.compiler.export import compile_headless
from torchwright.compiler.forward.cpsat_scheduler import critical_path_layers
from torchwright.compiler.lower import lower

# (d, d_hidden): the current publish default, then progressively doom-like
# widths (the flagship runs d=8192, d_hidden=16384).  d_head is NOT swept:
# it is baked into the graph at build time (every Attn's d_qk must equal
# the compile d_head), so changing it means rebuilding with a new module
# D_HEAD, not passing a different compile knob.
DEFAULT_CONFIGS = [
    (1024, 1024),
    (2048, 4096),
    (4096, 8192),
    (8192, 16384),
]
DEFAULT_NS = [3, 6]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--impl", default="calculator_simple")
    ap.add_argument("--ns", default=None, help="comma-separated digit counts")
    ap.add_argument("--optimize", type=int, default=0, help="publish default is 0")
    args = ap.parse_args()

    impl = importlib.import_module(f"examples.{args.impl}")
    ns = [int(x) for x in args.ns.split(",")] if args.ns else DEFAULT_NS

    for n in ns:
        output_node, embedding = impl.create_network_parts(max_digits=n)
        floor = critical_path_depth([output_node])
        print(f"[{args.impl} n={n}] critical-path floor: {floor}", flush=True)
        d_head = impl.D_HEAD
        for d, d_hidden in DEFAULT_CONFIGS:
            # The width-independent DAG layer floor of the *lowered* graph at
            # this config's collapse lane cap: no schedule can compile below
            # it, so n_layers == layer_floor certifies the schedule optimal
            # for the graph the scheduler was given.  (It exceeds the op-depth
            # floor by whatever linear glue survives fusion + collapse.)
            lowered = lower(
                output_node,
                collapse_univariate=True,
                collapse_pl=True,
                collapse_lane_cap=d_hidden // 4,
            )
            layer_floor = critical_path_layers(lowered.output_node)
            t0 = time.time()
            try:
                compiled = compile_headless(
                    output_node,
                    d=d,
                    d_head=d_head,
                    d_hidden=d_hidden,
                    optimize=args.optimize,
                    bias=False,
                    output_layout_source=embedding,
                )
            except RuntimeError as exc:
                # A width-starved geometry can have no schedule the eager
                # walk (or CP-SAT within budget) can find.  Report and keep
                # sweeping the wider configs.
                print(
                    f"[{args.impl} n={n}] d={d} d_head={d_head} "
                    f"d_hidden={d_hidden}: NO SCHEDULE "
                    f"({time.time() - t0:.0f}s): {str(exc).splitlines()[0]}",
                    flush=True,
                )
                continue
            dt = time.time() - t0
            print(
                f"[{args.impl} n={n}] d={d} d_head={d_head} d_hidden={d_hidden}: "
                f"n_layers={compiled.n_layers} (op-depth floor {floor}, "
                f"lowered layer floor {layer_floor}, "
                f"overhead {compiled.n_layers - floor}, {dt:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
