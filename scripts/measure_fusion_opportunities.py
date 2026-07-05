"""Measure depth-fusion opportunities on lowered graphs.

Answers "which fusion families would actually shorten the critical path?"
before any of them is built.  Three candidate families are measured
(numbering from the depth-fusion discussion, 2026-07):

1. **Attention affine folds** — a Linear whose every effective consumer
   (through Concatenates) is an Attn could compose into that Attn's
   Q/K/V matrices; a Linear reading an Attn directly could compose into
   its output matrix.  Either removes the Linear from the chain.
2. **Scalar piecewise-linear cones** — a subgraph of per-position ops
   (FFN/Linear/Add/Concatenate, literals allowed) whose only non-literal
   source is a single 1-D node computes a piecewise function of that one
   scalar, so the whole cone could in principle be re-synthesized as a
   single FFN (one MLP sublayer) of the source, whatever its internal
   chain depth.  Attention breaks a cone (it mixes across positions).
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
from typing import Dict, List, Optional, Tuple

from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.lower import lower
from torchwright.graph import Add, Attn, Concatenate, Linear, Node
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import LiteralValue

# Node types that cost one sublayer on a dependency chain.
_COSTLY = (Attn, FFN, Linear, Add)
# Per-position ops a scalar cone may pass through (Attn mixes across
# positions and is excluded; anything else unknown is excluded too).
_POINTWISE = (FFN, Linear, Add, Concatenate)


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


_TOP = object()  # "not a function of a single scalar"


def _scalar_sources(order: List[Node]) -> Dict[Node, Optional[Node]]:
    """For each node, the single 1-D node it is a pointwise function of.

    Returns ``src[n] = s`` when n's only non-literal ancestry, walked
    through pointwise ops, is the 1-D node ``s`` (``s`` itself maps to
    ``s``).  ``None`` means literal-only ancestry; ``_TOP`` (filtered to
    ``None`` in the result) means no single scalar source.
    """
    raw: Dict[Node, object] = {}
    for n in order:
        if isinstance(n, LiteralValue):
            raw[n] = None
            continue
        if not n.inputs or not isinstance(n, _POINTWISE):
            # A source candidate: any node computed "elsewhere" (input,
            # embedding, attention, ...) starts a cone iff it is 1-D.
            raw[n] = n if len(n) == 1 else _TOP
            continue
        acc: object = None
        for u in n.inputs:
            s = raw[u]
            if s is None:
                continue
            if acc is None:
                acc = s
            elif acc is not s:
                acc = _TOP
                break
        if acc is _TOP:
            # The cone dies here, but n itself can seed a new one.
            raw[n] = n if len(n) == 1 else _TOP
        else:
            raw[n] = acc
    return {
        n: (s if (s is not None and s is not _TOP) else None) for n, s in raw.items()
    }


def _cone_levels(order: List[Node], src: Dict[Node, Optional[Node]]) -> Dict[Node, int]:
    """Levels under family 2: any member of a scalar cone lands one
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


def _cone_details(
    order: List[Node],
    src: Dict[Node, Optional[Node]],
    critical: set,
    top: int = 15,
) -> None:
    """Per-source report for the cones that touch the critical path.

    ``chain`` is the cone's internal sublayer depth (what collapses to 1);
    ``crit chain`` the same restricted to critical members — the depth the
    collapse actually removes from the longest path through this cone.
    ``lanes`` sums member FFNs' hidden lanes (max single FFN in parens):
    the breakpoint-cost proxy for re-tabulating the composed function.
    """
    by_src: Dict[Node, List[Node]] = {}
    for n in order:
        s = src[n]
        if s is not None and s is not n:
            by_src.setdefault(s, []).append(n)

    reports = []
    for s, members in by_src.items():
        mset = set(members)
        # Longest chain inside the cone, overall and critical-only.
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
    print(f"critical-path cones (top {top} of {len(reports)} by critical chain):")
    for r in reports[:top]:
        print(
            f"  {r['source']}: range {r['range']}, {r['members']} nodes "
            f"({r['critical']} critical), chain {r['chain']} "
            f"(critical {r['crit_chain']}) -> 1, "
            f"lanes {r['lanes']} (max {r['max_lanes']})"
        )
        top_ops = ", ".join(f"{k}x{v}" for k, v in r["ops"].most_common(8))
        print(f"      ops: {top_ops}")


def analyze(output: Node, name: str, detail_cones: bool = False) -> None:
    lowered = lower(output)
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

    # Family 2: scalar piecewise cones.
    src = _scalar_sources(order)
    cone_members = [n for n in order if src[n] is not None and src[n] is not n]
    lv2 = _cone_levels(order, src)
    depth2 = lv2[out]
    cone_sizes = Counter(src[n].name or type(src[n]).__name__ for n in cone_members)

    # Feasibility-filtered variant: a collapsed cone is one FFN whose
    # breakpoint grid must resolve the composed function over the source's
    # range.  Integer-grained structure (floor steps, table rows) needs
    # ~range-width breakpoints, so a source range wider than a generous
    # lane budget (4096 — the largest single-FFN lane count in production)
    # cannot be re-tabulated; its cone keeps its current chain.  This is a
    # coarse proxy: it keeps a wide-range cone whose composed function
    # happens to be simple (over-conservative) and keeps a narrow-range
    # cone whose function needs sub-integer resolution (over-optimistic).
    _LANE_BUDGET = 4096.0

    def _feasible(s: Node) -> bool:
        vr = getattr(getattr(s, "value_type", None), "value_range", None)
        if vr is None:
            return False
        return (vr.hi - vr.lo) <= _LANE_BUDGET

    src_feasible = {
        n: (s if s is not None and _feasible(s) else None) for n, s in src.items()
    }
    depth2f = _cone_levels(order, src_feasible)[out]

    # Family 3: Add of two Linears.
    fam3 = [
        n
        for n in order
        if isinstance(n, Add) and all(isinstance(u, Linear) for u in n.inputs)
    ]

    # Families 1+2 combined.
    lv12 = {}
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
        f"family 2 (scalar PWL cones): {len(cone_members)} cone nodes over "
        f"{len(cone_sizes)} scalar sources, "
        f"{sum(1 for n in cone_members if n in critical)} on critical path"
        f" -> depth {depth2} ({depth0 - depth2} saved)"
    )
    for s_name, k in cone_sizes.most_common(6):
        print(f"    cone[{s_name}]: {k} nodes")
    print(
        f"family 2, feasible only (source range <= {_LANE_BUDGET:g})"
        f" -> depth {depth2f} ({depth0 - depth2f} saved)"
    )
    print(
        f"family 3 (Add of Linears): {len(fam3)} "
        f"({sum(1 for n in fam3 if n in critical)} on critical path)"
    )
    print(f"families 1+2 combined -> depth {depth12} ({depth0 - depth12} saved)")
    if detail_cones:
        _cone_details(order, src, critical)


_EXAMPLES = [
    "examples.binary_increment",
    "examples.adder",
    "examples.adder_v2",
    "examples.calculator_v2",
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
        "--cones",
        action="store_true",
        help="per-cone detail for cones on the critical path",
    )
    args = ap.parse_args()

    if args.spec:
        mod_name, fn_name = args.spec.split(":")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        result = fn(**json.loads(args.kwargs))
        output = result[0] if isinstance(result, tuple) else result
        analyze(output, args.spec, detail_cones=args.cones)
        return

    for mod_name in _EXAMPLES:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "create_network_parts"):
            output, _embedding = mod.create_network_parts()
        else:
            output = mod.create_network().inp  # Unembedding wraps the node
        analyze(output, mod_name, detail_cones=args.cones)


if __name__ == "__main__":
    main()
