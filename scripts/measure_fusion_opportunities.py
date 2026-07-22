"""Measure depth-fusion opportunities on lowered graphs.

Answers "which fusion families would actually shorten the critical path?"
before any of them is built.  Three candidate families are measured
(numbering from the depth-fusion discussion, 2026-07):

1. **Attention affine folds** — a Linear whose every effective consumer
   (through Concatenates) is an Attn could compose into that Attn's
   Q/K/V matrices; a Linear reading an Attn directly could compose into
   its output matrix.  Either removes the Linear from the chain.
2. **Univariate subgraphs** — a subgraph of per-position ops
   (FFN/Linear/Add/Concatenate, literals allowed) whose only non-literal
   source is a single 1-D node computes a piecewise-linear function of
   that one scalar, so the whole subgraph could in principle be
   re-synthesized as a single FFN (one MLP sublayer) of the source,
   whatever its internal chain depth.  Attention ends a univariate
   subgraph (it mixes across positions).
3. **Add-of-Linears** — ``Add(Linear(x), Linear(y))`` could normalize to
   ``Linear(Concatenate(x, y))``, feeding family-1/concat folds.

Depth is modeled in **sublayers** (the scheduler's currency): Attn, FFN,
Linear, and Add each cost one sublayer on a dependency chain; Concatenate
is virtual and LiteralValues/inputs are level 0.  This ignores the
attn/MLP parity alternation, so absolute numbers are a proxy — the
*deltas* between cost models are the signal.

Run on the lowered graph (``lower()``: wrappers stripped, existing linear
fusion applied), so every count is an opportunity the current optimizer
does NOT already take.

Usage::

    # the torchwright examples
    uv run python -m scripts.measure_fusion_opportunities

    # any graph builder: module:callable, kwargs as JSON; the output node
    # is the result itself or the first element of a result tuple
    uv run python -m scripts.measure_fusion_opportunities \
        --spec torchwright_doom.inference.compiled_model:build_graph \
        --kwargs '{"d_head": 128, "d_rot": 64, "max_positions": 61440, "wad_path": "doom1.wad"}'
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, cast

from torchwright.compiler.collapse import scalar_sources
from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.lower import lower
from torchwright.graph import Add, Attn, Concatenate, Linear, Node
from torchwright.graph.ffn import FFN

# Node types that cost one sublayer on a dependency chain.
_COSTLY = (Attn, FFN, Linear, Add)


def _cost(n: Node) -> int:
    return 1 if isinstance(n, _COSTLY) else 0


def _effective_inputs(n: Node) -> List[Node]:
    """Inputs with Concatenates flattened away (they are virtual)."""
    out: List[Node] = []
    stack = list(n.inputs)
    while stack:
        u = stack.pop()
        if isinstance(u, Concatenate):
            stack.extend(u.inputs)
        else:
            out.append(u)
    return out


def _effective_consumers(order: List[Node]) -> Dict[Node, List[Node]]:
    """node -> the non-Concatenate nodes that read it (through concats)."""
    direct: Dict[Node, List[Node]] = {n: [] for n in order}
    for n in order:
        for u in n.inputs:
            direct[u].append(n)
    eff: Dict[Node, List[Node]] = {}
    # order is inputs-first; walk it reversed so a Concatenate's own
    # effective consumers are resolved before the nodes feeding it ask.
    for n in reversed(order):
        acc: List[Node] = []
        for c in direct[n]:
            if isinstance(c, Concatenate):
                acc.extend(eff[c])
            else:
                acc.append(c)
        eff[n] = acc
    return eff


def _levels(order: List[Node], zero_cost: frozenset = frozenset()) -> Dict[Node, int]:
    lv: Dict[Node, int] = {}
    for n in order:
        base = max((lv[u] for u in n.inputs), default=0)
        lv[n] = base + (0 if n in zero_cost else _cost(n))
    return lv


def _collapsed_levels(
    order: List[Node], src: Dict[Node, Optional[Node]]
) -> Dict[Node, int]:
    """Levels under family 2: any member of a univariate subgraph lands one
    sublayer above its source (one FFN computes any function of it)."""
    lv: Dict[Node, int] = {}
    for n in order:
        s = src[n]
        if s is not None and s is not n:
            lv[n] = lv[s] + 1
        else:
            base = max((lv[u] for u in n.inputs), default=0)
            lv[n] = base + _cost(n)
    return lv


def _critical_edges(output: Node, lv: Dict[Node, int]) -> Tuple[set, Counter]:
    """Nodes on some longest path, and a bigram histogram of the costly
    producer->consumer pairs along those paths (concats skipped)."""
    critical = {output}
    bigrams: Counter = Counter()
    stack = [output]
    seen = set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        need = lv[n] - _cost(n)
        for u in n.inputs:
            if lv[u] == need:
                critical.add(u)
                stack.append(u)
                if _cost(n):
                    # attribute through virtual concats
                    producers = (
                        [u]
                        if not isinstance(u, Concatenate)
                        else [w for w in _effective_inputs(u) if lv[w] == need]
                    )
                    for w in producers:
                        bigrams[(type(w).__name__, type(n).__name__)] += 1
    return critical, bigrams


def _subgraph_details(
    order: List[Node],
    src: Dict[Node, Optional[Node]],
    critical: set,
    top: int = 15,
) -> None:
    """Per-source report for the univariate subgraphs touching the
    critical path.

    ``chain`` is the subgraph's internal sublayer depth (what collapses
    to 1); ``crit chain`` the same restricted to critical members — the
    depth the collapse actually removes from the longest path through
    this subgraph.  ``lanes`` sums member FFNs' hidden lanes (max single
    FFN in parens): the breakpoint-cost proxy for re-tabulating the
    composed function.
    """
    by_src: Dict[Node, List[Node]] = {}
    for n in order:
        s = src[n]
        if s is not None and s is not n:
            by_src.setdefault(s, []).append(n)

    reports: List[Dict[str, Any]] = []
    for s, members in by_src.items():
        mset = set(members)
        # Longest chain inside the subgraph, overall and critical-only.
        depth_in: Dict[Node, int] = {}
        depth_crit: Dict[Node, int] = {}
        for n in order:  # order is topological; members appear after s
            if n not in mset:
                continue
            base = max((depth_in.get(u, 0) for u in n.inputs), default=0)
            depth_in[n] = base + _cost(n)
            if n in critical:
                base_c = max((depth_crit.get(u, 0) for u in n.inputs), default=0)
                depth_crit[n] = base_c + _cost(n)
        crit_chain = max(depth_crit.values(), default=0)
        if crit_chain == 0:
            continue
        lanes = [m.gate_proj.shape[0] for m in members if isinstance(m, FFN)]
        vr = getattr(getattr(s, "value_type", None), "value_range", None)
        rng = f"[{vr.lo:.6g}, {vr.hi:.6g}]" if vr is not None else "?"
        ops = Counter(m.name or type(m).__name__ for m in members)
        reports.append(
            {
                "source": f"{s.name or type(s).__name__} ({type(s).__name__})",
                "range": rng,
                "members": len(members),
                "critical": sum(1 for m in members if m in critical),
                "chain": max(depth_in.values(), default=0),
                "crit_chain": crit_chain,
                "lanes": sum(lanes),
                "max_lanes": max(lanes, default=0),
                "ops": ops,
            }
        )

    reports.sort(key=lambda r: r["crit_chain"], reverse=True)
    print(
        f"critical-path univariate subgraphs "
        f"(top {top} of {len(reports)} by critical chain):"
    )
    for r in reports[:top]:
        print(
            f"  {r['source']}: range {r['range']}, {r['members']} nodes "
            f"({r['critical']} critical), chain {r['chain']} "
            f"(critical {r['crit_chain']}) -> 1, "
            f"lanes {r['lanes']} (max {r['max_lanes']})"
        )
        top_ops = ", ".join(f"{k}x{v}" for k, v in r["ops"].most_common(8))
        print(f"      ops: {top_ops}")


def analyze(
    output: Node,
    name: str,
    detail_subgraphs: bool = False,
    collapse: bool = False,
) -> None:
    # With ``collapse=True`` the univariate collapse pass runs in the
    # lowering (lane cap 4096, the modeled feasibility budget below), so
    # every number measures the *post-collapse* graph — the modeled
    # remaining opportunity after the pass took what it could.
    lowered = lower(
        output,
        collapse_univariate=collapse,
        collapse_lane_cap=4096 if collapse else None,
    )
    if collapse and lowered.collapse_report is not None:
        print(f"\n=== {name}: collapse pass log ===")
        print(lowered.collapse_report.format())
    out = lowered.output_node
    order = topological_order(out)
    eff_consumers = _effective_consumers(order)

    counts = Counter(type(n).__name__ for n in order)
    lv0 = _levels(order)
    depth0 = lv0[out]
    critical, bigrams = _critical_edges(out, lv0)

    # Family 1: attention affine folds.
    lin_into_attn = {
        n
        for n in order
        if isinstance(n, Linear)
        and eff_consumers[n]
        and all(isinstance(c, Attn) for c in eff_consumers[n])
    }
    attn_into_lin = {
        n
        for n in order
        if isinstance(n, Linear)
        and len(n.inputs) == 1
        and isinstance(n.inputs[0], Attn)
    }
    fam1 = lin_into_attn | attn_into_lin
    depth1 = _levels(order, zero_cost=frozenset(fam1))[out]

    # Family 2: univariate subgraphs.
    src = scalar_sources(order)
    members = [n for n in order if src[n] is not None and src[n] is not n]
    lv2 = _collapsed_levels(order, src)
    depth2 = lv2[out]
    subgraph_sizes = Counter(
        cast(Node, src[n]).name or type(src[n]).__name__ for n in members
    )

    # Feasibility-filtered variant: a collapsed univariate subgraph is one
    # FFN whose breakpoint grid must resolve the composed function over
    # the source's range.  Integer-grained structure (floor steps, table
    # rows) needs ~range-width breakpoints, so a source range wider than a
    # generous lane budget (4096 — the largest single-FFN lane count in
    # production) cannot be re-tabulated; its subgraph keeps its current
    # chain.  This is a coarse proxy: it keeps a wide-range subgraph whose
    # composed function happens to be simple (over-conservative) and keeps
    # a narrow-range subgraph whose function needs sub-integer resolution
    # (over-optimistic).
    _LANE_BUDGET = 4096.0

    def _feasible(s: Node) -> bool:
        vr = getattr(getattr(s, "value_type", None), "value_range", None)
        if vr is None:
            return False
        return (vr.hi - vr.lo) <= _LANE_BUDGET

    src_feasible = {
        n: (s if s is not None and _feasible(s) else None) for n, s in src.items()
    }
    depth2f = _collapsed_levels(order, src_feasible)[out]

    # Family 3: Add of two Linears.
    fam3 = [
        n
        for n in order
        if isinstance(n, Add) and all(isinstance(u, Linear) for u in n.inputs)
    ]

    # Families 1+2 combined.
    lv12: Dict[Node, int] = {}
    for n in order:
        s = src[n]
        if s is not None and s is not n:
            lv12[n] = lv12[s] + 1
        else:
            base = max((lv12[u] for u in n.inputs), default=0)
            lv12[n] = base + (0 if n in fam1 else _cost(n))
    depth12 = lv12[out]

    print(f"\n=== {name} ===")
    print(f"nodes: {sum(counts.values())}  {dict(counts)}")
    print(f"critical path: {depth0} sublayers  (critical nodes: {len(critical)})")
    print("critical bigrams (producer -> consumer, top 12):")
    for (a, b), k in bigrams.most_common(12):
        print(f"    {a:>12} -> {b:<12} {k}")
    print(
        f"family 1 (attn affine folds): {len(lin_into_attn)} Linear->Attn, "
        f"{len(attn_into_lin)} Attn->Linear, "
        f"{len(fam1 & critical)} on critical path"
        f" -> depth {depth1} ({depth0 - depth1} saved)"
    )
    print(
        f"family 2 (univariate subgraphs): {len(members)} member nodes over "
        f"{len(subgraph_sizes)} scalar sources, "
        f"{sum(1 for n in members if n in critical)} on critical path"
        f" -> depth {depth2} ({depth0 - depth2} saved)"
    )
    for s_name, k in subgraph_sizes.most_common(6):
        print(f"    subgraph[{s_name}]: {k} nodes")
    print(
        f"family 2, feasible only (source range <= {_LANE_BUDGET:g})"
        f" -> depth {depth2f} ({depth0 - depth2f} saved)"
    )
    print(
        f"family 3 (Add of Linears): {len(fam3)} "
        f"({sum(1 for n in fam3 if n in critical)} on critical path)"
    )
    print(f"families 1+2 combined -> depth {depth12} ({depth0 - depth12} saved)")
    if detail_subgraphs:
        _subgraph_details(order, src, critical)


_EXAMPLES = [
    "examples.binary_increment",
    "examples.adder",
    "examples.adder_v2",
    "examples.calculator_simple",
    "examples.caesar_cipher",
    "examples.fibonacci",
    "examples.sort_digits_v1",
    "examples.calculator_scratchpad",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", help="module:callable building the graph")
    ap.add_argument("--kwargs", default="{}", help="JSON kwargs for the callable")
    ap.add_argument(
        "--subgraphs",
        action="store_true",
        help="per-subgraph detail for univariate subgraphs on the critical path",
    )
    ap.add_argument(
        "--collapse",
        action="store_true",
        help="run the univariate collapse pass (lane cap 4096) in the "
        "lowering, so the analysis measures the post-collapse graph",
    )
    args = ap.parse_args()

    if args.spec:
        mod_name, fn_name = args.spec.split(":")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        result = fn(**json.loads(args.kwargs))
        output = result[0] if isinstance(result, tuple) else result
        analyze(
            output, args.spec, detail_subgraphs=args.subgraphs, collapse=args.collapse
        )
        return

    for mod_name in _EXAMPLES:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "create_network_parts"):
            output, _embedding = mod.create_network_parts()
        else:
            output = mod.create_network().inp  # Unembedding wraps the node
        analyze(
            output, mod_name, detail_subgraphs=args.subgraphs, collapse=args.collapse
        )


if __name__ == "__main__":
    main()
