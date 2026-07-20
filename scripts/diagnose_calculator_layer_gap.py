"""Name the glue: why the calculator's lowered layer floor exceeds its op depth.

``scripts.arithmetic_scaling.critical_path_depth`` counts the longest chain
of *nonlinear* ops (FFN / Attn) — the depth the algorithm needs with enough
width.  The compiled layer count at saturated width instead equals
``critical_path_layers`` of the **lowered** graph, which also pays a layer
for every standalone Linear / Add hop that survives fusion (minus the free
attention→MLP same-layer pairings).  This script attributes that gap:

1. build a calculator, lower it exactly as the compile entry points do
   (fusion + both collapse passes);
2. compute the mode-aware per-node earliest-layer bounds
   (``cpsat_scheduler._compute_layer_bounds`` — the exact floor);
3. backtrack one longest chain and print every op on it with its type,
   name (op provenance), and width;
4. histogram the chain's standalone Linears / Adds by name, so the gap
   decomposes into "which builder emitted this glue".

Pure CPU, seconds per config::

    python -m scripts.diagnose_calculator_layer_gap --impl calculator_simple --n 6
"""

import argparse
import importlib
from collections import Counter
from typing import Dict, List, Tuple

from torchwright.compiler.lower import lower
from torchwright.compiler.forward.cpsat_scheduler import (
    ATTN,
    MLP,
    _compute_layer_bounds,
    build_graph_model,
    is_flex,
)
from torchwright.compiler.forward.scheduling_policy import LEGACY_POLICY


def _describe(node) -> str:
    name = getattr(node, "name", None) or "<unnamed>"
    return f"{type(node).__name__:12s} w={len(node):4d}  {name}"


def longest_chain(gm, es: Dict[int, int]) -> List:
    """Backtrack one longest chain through the earliest-start bounds.

    Heuristic reconstruction (the bounds are mode-aware; this walk is not):
    start from a node achieving the max earliest layer, and repeatedly step
    to a predecessor whose earliest layer is >= this node's minus one —
    i.e. a predecessor that could be what forced this node's start.  Prefer
    the tightest predecessor (max es) so the walk follows the binding chain.
    """
    by_id = {n.node_id: n for n in gm.schedulable}
    preds: Dict[int, List[int]] = {n.node_id: [] for n in gm.schedulable}
    for u, v in gm.edges:
        if u.node_id in preds and v.node_id in preds:
            preds[v.node_id].append(u.node_id)

    cur = max(es, key=es.get)
    chain = [cur]
    while True:
        candidates = [u for u in preds[cur] if es[u] >= es[cur] - 1]
        if not candidates:
            break
        cur = max(candidates, key=es.get)
        chain.append(cur)
    return [by_id[i] for i in reversed(chain)]


def forensic_carry_sweep(output_node, lane_cap: int) -> None:
    """Re-certify the first declined carry-sweep subgraph with keep_raw and
    dissect the worst chargeable sample: member, oracle vs chord, and why the
    sample was not excused as fillet/band.

    Expects a lowered copy with the v1 pass only — the exact graph state
    collapse_pl saw its subgraphs in.
    """
    import torch

    from torchwright.compiler.collapse import _seeded_oracle, scalar_sources
    from torchwright.compiler.graph_clone import topological_order
    from torchwright.compiler.pl_function import (
        _HINGE_EXACT_Z,
        _in_intervals,
        certify_subgraph,
    )
    from torchwright.graph.misc import Add

    order = topological_order(output_node)
    src = scalar_sources(order)
    by_src: Dict = {}
    for n in order:
        s = src[n]
        if s is not None and s is not n:
            by_src.setdefault(s, []).append(n)

    # The carry-sweep subgraphs: rooted at an Add, chain through in_range.
    for source, members in by_src.items():
        if not isinstance(source, Add):
            continue
        if not any(
            (getattr(m, "name", None) or "").startswith("in_range") for m in members
        ):
            continue
        topo_index = {n: i for i, n in enumerate(order)}
        print(f"\n=== forensic: Add#{topo_index[source]} ({len(members)} members) ===")
        vr = source.value_type.value_range
        print(f"source range: [{vr.lo}, {vr.hi}]")
        cert = certify_subgraph(
            source,
            members,
            max_kinks=4 * lane_cap,
            hinge_exact=_HINGE_EXACT_Z,
            keep_raw=True,
        )
        if cert.declined:
            print(f"certify declined: {cert.declined}")
            return
        for m, c in cert.members.items():
            name = getattr(m, "name", None) or type(m).__name__
            print(
                f"  member {name:32s} kinks={c.n_kinks:5d} "
                f"dev={c.deviation:.3e} at {c.deviation_at:.6g}  "
                f"banded={c.banded_deviation:.3e} at {c.banded_deviation_at:.6g}  "
                f"fillet={c.fillet_deviation:.3e} at {c.fillet_deviation_at:.6g}"
            )
        # Dissect the worst member's deviation point against its oracle.
        worst = max(cert.members.values(), key=lambda c: c.deviation)
        m = worst.node
        name = getattr(m, "name", None) or type(m).__name__
        x = worst.deviation_at
        print(f"\nworst member: {name}, x={x!r}")

        xs = torch.tensor(
            [x - 0.01, x - 0.001, x, x + 0.001, x + 0.01], dtype=torch.float64
        )
        grid = xs.to(torch.float32).reshape(-1, 1)
        vals = _seeded_oracle(list(members), source, grid)[m]
        chord = worst.fn.eval(xs)
        for xi, v, cval in zip(xs.tolist(), vals, chord):
            e = (v.to(torch.float64) - cval).abs().max()
            print(
                f"  x={xi:12.8f} max|oracle-chord|={e:.3e}  "
                f"oracle={[round(float(t), 6) for t in v[:6]]}  "
                f"chord={[round(float(t), 6) for t in cval[:6]]}"
            )
        if worst.bands is not None and worst.bands.numel():
            inb = _in_intervals(worst.bands, torch.tensor([x], dtype=torch.float64))
            near = worst.bands[(worst.bands[:, 0] - x).abs().argmin()]
            print(
                f"  in analytic band: {bool(inb[0])}; nearest band "
                f"[{float(near[0]):.6g}, {float(near[1]):.6g}]"
            )
        kn = worst.fn_raw.x if worst.fn_raw is not None else worst.fn.x
        j = int((kn - x).abs().argmin())
        print(f"  nearest knots: {[float(v) for v in kn[max(0, j - 2) : j + 3]]}")
        return
    print("no carry-sweep subgraph found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--impl", default="calculator_simple")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--d-hidden", type=int, default=8192)
    ap.add_argument(
        "--no-collapse",
        action="store_true",
        help="lower with fusion only (skip both collapse passes)",
    )
    ap.add_argument(
        "--forensic",
        action="store_true",
        help="dissect the first declined carry-sweep subgraph "
        "(lowers with the v1 pass only, then exits)",
    )
    ap.add_argument(
        "--pl-budget",
        type=float,
        default=None,
        help="what-if: rerun the collapse_pl pass with this budget instead "
        "of the production 1e-3 (replicates lower()'s pipeline manually; "
        "measures the floor a budget change would buy — not a fix)",
    )
    args = ap.parse_args()

    impl = importlib.import_module(f"examples.{args.impl}")
    out, _embedding = impl.create_network_parts(max_digits=args.n)

    from scripts.arithmetic_scaling import critical_path_depth

    depth = critical_path_depth([out])
    print(f"[{args.impl} n={args.n}] nonlinear-op depth: {depth}")

    if args.forensic:
        lowered = lower(
            out,
            collapse_univariate=True,
            collapse_pl=False,
            collapse_lane_cap=args.d_hidden // 4,
        )
        forensic_carry_sweep(lowered.output_node, args.d_hidden // 4)
        return

    collapse = not args.no_collapse
    if args.pl_budget is not None:
        # Replicate lower()'s pipeline with the collapse_pl budget knob
        # turned: v1 via lower(), then the v2 pass by hand.
        from torchwright.compiler.collapse_pl import collapse_pl_subgraphs
        from torchwright.graph.optimize import FoldLog

        lowered = lower(
            out,
            collapse_univariate=True,
            collapse_pl=False,
            collapse_lane_cap=args.d_hidden // 4,
        )
        pl_out, pl_report = collapse_pl_subgraphs(
            lowered.output_node,
            lane_cap=args.d_hidden // 4,
            fold_log=FoldLog(),
            budget=args.pl_budget,
        )
        reports = [
            ("collapse_univariate", lowered.collapse_report),
            (f"collapse_pl (budget={args.pl_budget:g})", pl_report),
        ]

        class _L:  # minimal stand-in for the LoweredGraph fields read below
            output_node = pl_out

        lowered = _L()
    else:
        lowered = lower(
            out,
            collapse_univariate=collapse,
            collapse_pl=collapse,
            collapse_lane_cap=args.d_hidden // 4 if collapse else None,
        )
        reports = [
            ("collapse_univariate", lowered.collapse_report),
            ("collapse_pl", lowered.collapse_pl_report),
        ]
    for label, report in reports:
        if report is None:
            continue
        print(f"\n=== {label} outcomes ===")
        print(report.format())

    gm = build_graph_model(lowered.output_node)
    es, _ls = _compute_layer_bounds(
        gm, LEGACY_POLICY, flex_routing=True, max_layers=1 << 20
    )
    floor = max(es.values()) + 1
    print(
        f"[{args.impl} n={args.n}] lowered layer floor: {floor} "
        f"(gap {floor - depth}); {len(gm.schedulable)} schedulable nodes"
    )

    chain = longest_chain(gm, es)
    print(f"\n=== one longest chain ({len(chain)} ops) ===")
    comp = Counter(type(n).__name__ for n in chain)
    print(f"composition: {dict(comp)}")
    for n in chain:
        mode = "flex" if is_flex(n, gm) else "-"
        print(f"  L{es[n.node_id]:3d} [{mode:4s}] {_describe(n)}")

    glue = [n for n in chain if type(n).__name__ in ("Linear", "Add")]
    print(f"\n=== chain glue by name ({len(glue)} Linear/Add on chain) ===")
    for (t, name), c in Counter(
        (type(n).__name__, getattr(n, "name", None) or "<unnamed>") for n in glue
    ).most_common():
        print(f"  {c:3d}  {t:8s} {name}")

    all_glue = [n for n in gm.schedulable if type(n).__name__ in ("Linear", "Add")]
    print(f"\n=== whole-graph glue by name ({len(all_glue)} Linear/Add total) ===")
    for (t, name), c in Counter(
        (type(n).__name__, getattr(n, "name", None) or "<unnamed>") for n in all_glue
    ).most_common(30):
        print(f"  {c:3d}  {t:8s} {name}")


if __name__ == "__main__":
    main()
