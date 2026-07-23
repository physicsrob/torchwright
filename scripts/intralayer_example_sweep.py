"""Example-graph width sweep for CP-SAT intra-layer reuse (step-8 phase gate).

For each example graph it sweeps the residual width ``d`` downward from the
graph's natural width to the scheduling floor and records, per width:

- ``opt0`` — layer count of the eager heuristic compile (``optimize=0``).
- ``opt1`` — layer count of the CP-SAT compile (``optimize=1``), plus whether
  the solver actually produced the schedule (vs a heuristic fallback).
- ``sound`` / ``relaxed`` — the CP-SAT optimum under the sound
  ``[layer, cancel+1)`` MLP-cancel occupancy vs the diagnostic-only
  ``[layer, cancel)`` relaxation (``_disabled_families={"mlp_cancel_occupancy"}``,
  never compiled — its count is a valid lower bound).  ``sound - relaxed`` is
  the bindingness of known optimality gap #1 (the forgone same-layer MLP->MLP
  column handoff; derisk doc 2026-07-08 correction).

Acceptance (the phase gate): ``opt1 <= opt0`` at every width, strictly less at
some width, and the two compiles share the same minimum feasible width (the
solver does not over-estimate width).

Run on Modal GPU (compiles need it):

    make modal-run MODULE=scripts.intralayer_example_sweep

Options via ARGS, e.g.:

    make modal-run MODULE=scripts.intralayer_example_sweep ARGS="--graphs calculator caesar --budget 20"

Gap #2 (the free-add phantom cancel charge) is NOT relaxed here: whether a node
dies by ``reassign`` is a per-node solver outcome (``is_free`` of the consuming
Add and which addend), so no *static* relaxation isolates it without also
dropping genuinely-charged cancels.  Recorded, not measured.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    critical_path_layers,
    solve_schedule,
)
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.compiler.lower import lower

_RELAX = frozenset({"mlp_cancel_occupancy"})

# Full per-family relaxation set for the CP-SAT expressiveness plan §0 gap
# attribution.  Each is disabled ALONE (never combined) in a solve-only
# re-solve; ``sound - relaxed`` attributes that family's bindingness on this
# graph.  ``cancel_consumer_lb`` and ``dependency`` are sanity-only (unsound,
# expected to drop far — the LB collapses / the schedule lands below the
# critical-path floor); they confirm the harness reacts, they are not gaps to
# close.  Example graphs solve to proven-optimal, so their family drops are
# authoritative (DOOM's are directional — plan §0 bound-directions rule).
_FAMILY_SWEEP = [
    "residual_cumulative",
    "attn_cumulative",
    "mlp_cumulative",
    "cancel_slack",
    "mlp_cancel_occupancy",
    "cancel_consumer_lb",
    "dependency",
]


def _build_bucketed_argmin():
    from torchwright.graph import InputNode
    from torchwright.ops.attention_ops import attend_argmin_above_in_bucket
    from torchwright.ops.inout_nodes import create_rope_config

    nb, nt, vw, d_head = 3, 5, 4, 32
    score = InputNode("baib_score", 1, value_range=(-100.0, 100.0))
    validity = InputNode("baib_validity", 1, value_range=(-2.0, 2.0))
    kb = InputNode("baib_kb", nb, value_range=(-2.0, 2.0))
    above = InputNode("baib_above", nt, value_range=(-2.0, 2.0))
    qb = InputNode("baib_qb", nb, value_range=(-2.0, 2.0))
    th = InputNode("baib_th", nt, value_range=(-2.0, 2.0))
    value = InputNode("baib_value", vw, value_range=(-100.0, 100.0))
    return attend_argmin_above_in_bucket(
        create_rope_config(d_head=d_head, max_positions=512),
        score,
        validity,
        kb,
        above,
        qb,
        th,
        value,
    )


def _example_specs() -> dict[str, tuple[Callable, int, int]]:
    """Name -> (build_output_node, d_natural, d_head).  Built lazily so a
    ``--graphs`` subset never imports the others.
    """
    from examples import (
        binary_increment,
        caesar_cipher,
        calculator_simple,
        fibonacci,
        sort_digits_v1,
    )

    return {
        # calculator: d pinned at the geometry the committed measurements ran
        # at (the canonical publish D_MODEL=8192 would change the measurement
        # context); d_head follows the module — baked into the graph.
        "calculator": (
            lambda: calculator_simple.create_network_parts()[0],
            1024,
            calculator_simple.D_HEAD,
        ),
        "caesar": (
            lambda: caesar_cipher.create_network_parts()[0],
            caesar_cipher.D_MODEL,
            caesar_cipher.D_HEAD,
        ),
        "sort_digits": (
            lambda: sort_digits_v1.create_network_parts()[0],
            sort_digits_v1.D_MODEL,
            sort_digits_v1.D_HEAD,
        ),
        "fibonacci": (
            lambda: fibonacci.create_network_parts()[0],
            fibonacci.D_MODEL,
            fibonacci.D_HEAD,
        ),
        "binary_increment": (
            lambda: binary_increment.create_network_parts()[0],
            binary_increment.D_MODEL,
            binary_increment.D_HEAD,
        ),
        "bucketed_argmin": (_build_bucketed_argmin, 512, 32),
    }


def _compile_layers(out, d, d_head, optimize) -> tuple[int | None, str]:
    """Compile ``out`` and return (n_layers, status).  ``status`` is
    'heuristic'/'solver'/'fallback' for a successful compile, or 'FAIL:<reason>'
    when the compile raised (the width is below this optimize level's floor).
    The optimize=1 solve uses forward_compile's own 60s budget.
    """
    try:
        net = forward_compile(
            d=d,
            d_head=d_head,
            output_node=out,
            device="cpu",
            verbose=False,
            optimize=optimize,
        )
    except Exception as exc:  # noqa: BLE001 - a failed compile marks the floor
        return None, f"FAIL:{type(exc).__name__}"
    n = len(net.layers)
    if optimize == 0:
        return n, "heuristic"
    stats = net.cpsat_solve_stats
    solved = stats is not None and stats.status_name in (
        "OPTIMAL",
        "FEASIBLE",
        "CACHED",
    )
    return n, ("solver" if solved else "fallback")


def _solve_layers(
    low_out, d, d_head, budget, disabled
) -> tuple[int | None, bool, float | None]:
    """Solve-only (never compiled): (n_layers, is_optimal, proven_bound).

    Returns ``(None, False, None)`` on no-solution / raise.  The proven bound
    (``SolveStats.best_objective_bound``) is a certified lower bound on the
    sound optimum — recorded per family per the plan §0 honest-ledger rule.

    ``low_out`` is the LOWERED output node — forward_compile schedules the
    lowered graph (collapse passes), so a raw-graph solve would over-count.
    Uses ``SchedulingPolicy()`` and ``tighten_domains`` to match the compile;
    the only intended difference between calls is ``disabled`` (the relaxed
    constraint family).
    """
    try:
        asg, stats = solve_schedule(
            low_out,
            d=d,
            d_head=d_head,
            d_hidden=d,
            max_layers=100,
            time_budget_s=budget,
            policy=SchedulingPolicy(),
            tighten_domains=True,
            _disabled_families=disabled,
        )
    except Exception:  # noqa: BLE001
        return None, False, None
    if asg is None:
        return None, False, stats.best_objective_bound
    return asg.n_layers, stats.is_optimal, stats.best_objective_bound


def _lower(out, d):
    """Lower ``out`` the way forward_compile does before scheduling."""
    return lower(
        out,
        verbose=False,
        collapse_univariate=True,
        collapse_pl=True,
        collapse_lane_cap=d // 4,
    ).output_node


def _width_grid(d_natural: int, d_head: int, n_points: int) -> list[int]:
    """Widths from d_natural down toward a pressured floor, on the d_head grid.
    Floor target is a quarter of the natural width (>= two heads); compiles
    below the true scheduling floor are recorded as FAIL, which locates it.
    """
    lo = max(2 * d_head, (d_natural // 4 // d_head) * d_head)
    lo = min(lo, d_natural)
    if n_points <= 1 or d_natural <= lo:
        return sorted({d_natural, lo}, reverse=True)
    step = (d_natural - lo) / (n_points - 1)
    grid = {
        int(round((d_natural - i * step) / d_head)) * d_head for i in range(n_points)
    } | {d_natural, lo}
    return sorted((d for d in grid if d >= d_head), reverse=True)


def sweep(graphs: list[str], budget: float, n_points: int):
    specs = _example_specs()
    results: dict[str, list[dict]] = {}
    for name in graphs:
        build, d_nat, d_head = specs[name]
        cp_lb = critical_path_layers(build())
        grid = _width_grid(d_nat, d_head, n_points)
        print(
            f"\n=== {name}: d_natural={d_nat} d_head={d_head} "
            f"critical-path floor={cp_lb} layers; widths={grid} ===",
            flush=True,
        )
        rows: list[dict] = []
        for d in grid:
            t0 = time.perf_counter()
            # Layer counts depend only on topology, so one build serves all
            # four (compilation never mutates the source graph).  The compiles
            # lower internally; the solves get the pre-lowered graph so both
            # schedule the same node set.
            out = build()
            n0, s0 = _compile_layers(out, d, d_head, 0)
            n1, s1 = _compile_layers(out, d, d_head, 1)
            try:
                low_out = _lower(build(), d)
            except Exception:  # noqa: BLE001
                low_out = None
            if low_out is not None:
                sound_n, sound_opt, sound_bound = _solve_layers(
                    low_out, d, d_head, budget, frozenset()
                )
                # Full per-family sweep: each family disabled alone (§0).
                fam: dict[str, dict] = {}
                for family in _FAMILY_SWEEP:
                    fn, fopt, fbound = _solve_layers(
                        low_out, d, d_head, budget, frozenset({family})
                    )
                    fam[family] = dict(n=fn, opt=fopt, bound=fbound)
            else:
                sound_n, sound_opt, sound_bound = None, False, None
                fam = {f: dict(n=None, opt=False, bound=None) for f in _FAMILY_SWEEP}
            dt = time.perf_counter() - t0
            row = dict(
                d=d,
                floor=cp_lb,
                opt0=n0,
                opt0_status=s0,
                opt1=n1,
                opt1_status=s1,
                sound=sound_n,
                sound_opt=sound_opt,
                sound_bound=sound_bound,
                families=fam,
                # ``relaxed`` kept for the legacy phase-gate verdict (gap #1).
                relaxed=fam["mlp_cancel_occupancy"]["n"],
                relaxed_opt=fam["mlp_cancel_occupancy"]["opt"],
                secs=round(dt, 1),
            )
            rows.append(row)

            def _drop(family: str) -> str:
                fn = fam[family]["n"]
                if fn is None or sound_n is None:
                    return "  -"
                return f"{sound_n - fn:>3}"

            print(
                f"  d={d:>5} floor={cp_lb:>2}  opt0={_fmt(n0)}({s0:<9}) "
                f"opt1={_fmt(n1)}({s1:<8}) sound={_fmt(sound_n)}"
                f"{'*' if sound_opt else '~'}  drop: "
                f"resid={_drop('residual_cumulative')} "
                f"attn={_drop('attn_cumulative')} "
                f"mlp={_drop('mlp_cumulative')} "
                f"cslack={_drop('cancel_slack')} "
                f"mlpcanc={_drop('mlp_cancel_occupancy')} "
                f"[clb={_drop('cancel_consumer_lb')} "
                f"dep={_drop('dependency')}]  [{dt:.0f}s]",
                flush=True,
            )
        results[name] = rows
    _verdict(results)
    return results


def _fmt(v) -> str:
    return "  -" if v is None else f"{v:>3}"


def _verdict(results: dict[str, list[dict]]):
    print("\n" + "=" * 72)
    print("PHASE-GATE VERDICT")
    print("=" * 72)
    any_strictly_better = False
    all_ge = True  # opt1 <= opt0 wherever both compiled
    floors_match = True
    gap1_binds_anywhere = False
    for name, rows in results.items():
        opt0_floor = min((r["d"] for r in rows if r["opt0"] is not None), default=None)
        opt1_floor = min((r["d"] for r in rows if r["opt1"] is not None), default=None)
        solver_floor = min(
            (r["d"] for r in rows if r["opt1_status"] == "solver"), default=None
        )
        for r in rows:
            if r["opt0"] is not None and r["opt1"] is not None:
                if r["opt1"] > r["opt0"]:
                    all_ge = False
                if r["opt1"] < r["opt0"]:
                    any_strictly_better = True
            if (
                r["sound"] is not None
                and r["relaxed"] is not None
                and r["sound"] > r["relaxed"]
            ):
                gap1_binds_anywhere = True
        heuristic_floor = opt0_floor
        this_match = solver_floor == heuristic_floor
        if not this_match:
            floors_match = False
        print(
            f"  {name:<16} heuristic_floor={heuristic_floor} "
            f"solver_floor={solver_floor} "
            f"opt1_compile_floor={opt1_floor} floors_match={this_match}"
        )
    print("-" * 72)
    print(f"  opt1 <= opt0 everywhere:        {all_ge}")
    print(f"  opt1 strictly better somewhere: {any_strictly_better}")
    print(f"  solver floor == heuristic floor: {floors_match}")
    print(f"  [gap #1 binds somewhere (sound>relaxed): {gap1_binds_anywhere}]")
    gate = all_ge and any_strictly_better and floors_match
    print(f"\n  PHASE GATE: {'PASS' if gate else 'FAIL'}")
    print(
        "  (gap #2 free-add phantom charge: no clean static relaxation exists — "
        "dies-by-reassign is a per-node solver outcome; recorded, not measured.)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="*", default=None)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--points", type=int, default=3)
    ap.add_argument(
        "--json", default=None, help="write the full per-family results as JSON"
    )
    args = ap.parse_args()
    all_names = list(_example_specs().keys())
    graphs = args.graphs or all_names
    unknown = set(graphs) - set(all_names)
    if unknown:
        raise SystemExit(f"unknown graphs {sorted(unknown)}; valid: {all_names}")
    torch.manual_seed(0)
    results = sweep(graphs, args.budget, args.points)
    if args.json:
        import json

        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
