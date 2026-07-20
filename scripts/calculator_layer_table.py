"""Layer-count table for the three calculators across digit counts.

For each implementation and ``max_digits`` this prints the nonlinear-op
depth (``scripts.arithmetic_scaling.critical_path_depth``) and the
**lowered layer floor** — ``critical_path_layers`` of the graph after
``lower()`` with both collapse passes, the exact width-independent DAG
minimum.  At saturated width the compiled ``n_layers`` equals this floor
(confirmed by ``optimize=2`` compiles at n=3 and n=6 across d=1024..8192),
so the floor column IS the actual layer count without paying a CP-SAT
solve per cell.

The scratchpad rows also print decode steps (its depth is flat because
the serial work moves onto the decode axis), and a build that violates a
structural cap (the multiply pointer gather's slow-plane budget bounds
its ``max_digits``) is reported instead of crashing the sweep.

CPU only; run on Modal to spare the local machine::

    make modal-run MODULE=scripts.calculator_layer_table CPU_ONLY=1
"""

import argparse
import importlib
import time

from scripts.arithmetic_scaling import critical_path_depth
from torchwright.compiler.forward.cpsat_scheduler import critical_path_layers
from torchwright.compiler.lower import lower

IMPLS = ["calculator_simple", "calculator_advanced", "calculator_scratchpad"]
# Flagship-geometry collapse cap (d_hidden 8192 // 4); the synthesized
# staircases top out well below it, so wider caps give the same floor.
LANE_CAP = 2048


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", default="2,3,4,5,6,7,8,9,10")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]

    rows = {}
    for impl_name in IMPLS:
        impl = importlib.import_module(f"examples.{impl_name}")
        for n in ns:
            t0 = time.time()
            try:
                out, _embedding = impl.create_network_parts(max_digits=n)
            except Exception as exc:  # structural caps (slow-plane budget)
                rows[(impl_name, n)] = (None, None, None, str(exc).splitlines()[0])
                print(
                    f"[{impl_name} n={n}] build failed: " f"{str(exc).splitlines()[0]}",
                    flush=True,
                )
                continue
            depth = critical_path_depth([out])
            lowered = lower(
                out,
                collapse_univariate=True,
                collapse_pl=True,
                collapse_lane_cap=LANE_CAP,
            )
            floor = critical_path_layers(lowered.output_node)
            steps = None
            if hasattr(impl, "decode_steps"):
                steps = impl.decode_steps(n)
            rows[(impl_name, n)] = (depth, floor, steps, None)
            print(
                f"[{impl_name} n={n}] depth={depth} floor={floor}"
                + (f" steps={steps}" if steps is not None else "")
                + f" ({time.time() - t0:.0f}s)",
                flush=True,
            )

    print(
        "\n=== layer count (lowered floor = compiled n_layers at "
        "saturated width); op depth in parens ==="
    )
    header = f"{'n':>3} " + "".join(f"{name.split('_')[1]:>22}" for name in IMPLS)
    print(header)
    for n in ns:
        cells = []
        for impl_name in IMPLS:
            depth, floor, steps, err = rows[(impl_name, n)]
            if err is not None:
                cells.append(f"{'—':>22}")
                continue
            cell = f"{floor} ({depth})"
            if steps is not None:
                cell += f" +{steps} steps"
            cells.append(f"{cell:>22}")
        print(f"{n:>3} " + "".join(cells))


if __name__ == "__main__":
    main()
