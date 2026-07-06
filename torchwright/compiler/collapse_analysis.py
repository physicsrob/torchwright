"""Report-only analysis for univariate collapse v2 (Phase A).

Runs on a lowered copy — with the shipping v1 staircase collapse
already applied, so everything reported here is opportunity v1 does
NOT take.  For every univariate subgraph it:

1. certifies the composed piecewise-linear function of each boundary
   member with the PLFunction walk (candidate kinks from weights,
   values from the seeded oracle, midpoint-linearity certificate —
   :mod:`torchwright.compiler.pl_function`);
2. models both emission shapes — S1 (one interpolating FFN, chain → 1)
   and S2 (two-sublayer bounded-step emission, chain → 2) — with their
   lane counts and error bounds, and picks the cheapest admissible one;
3. estimates the S2 **liveness cost**: stage-1 bounded-step values
   occupy one residual column per step between the two sublayers, and
   the columns of simultaneously-live chains add up — the trap that
   sank the reverted 36-layer folded-floor variant at d=4096;
4. models the layer floor with and without the collapse by grafting
   1- or 2-sublayer stand-in FFNs onto the throwaway copy and asking
   ``critical_path_layers`` (the mode-aware dependency floor the
   production CP-SAT schedule provably reaches on the doom graph) —
   the realized quantity, not the measure_fusion proxy.

Nothing here mutates a source graph or is called by ``lower()`` /
compile: the copy it rewires is the caller's throwaway
``LoweredGraph.output_node``, consumed only by the floor model.
Phase A instrument; synthesis is Phase B, gated on Rob's GO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from torchwright.compiler.collapse import (
    _SYNTH_CLAIM_ATOL,
    _machine,
    _member_depths,
    scalar_sources,
)
from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.pl_function import (
    MemberCertificate,
    S1Model,
    S2Model,
    _snap_f32,
    certify_subgraph,
    model_s1,
    model_s2,
    transition_runs,
)
from torchwright.graph import Node
from torchwright.graph.ffn import FFN


@dataclass(frozen=True)
class MemberAnalysis:
    """One synthesized-boundary member's certificate + shape models."""

    name: str
    d_output: int
    depth: int
    n_kinks: int
    deviation: float
    deviation_at: float
    fillet_deviation: float
    s1: S1Model
    s2: S2Model
    linear_ok: bool
    s1_ok: bool
    s2_ok: bool


@dataclass
class SubgraphAnalysis:
    """One univariate subgraph's v2 verdict."""

    source: str
    annotation: str
    domain: Tuple[float, float]
    n_members: int
    chain_depth: int
    n_boundary: int
    n_synthesized: int
    verdict: str  # 'S1' | 'S2' | 'no depth gain' | 'declined: <reason>'
    depth_after: Optional[int] = None  # modeled chain depth if taken
    members: List[MemberAnalysis] = field(default_factory=list)
    stage1_cols: int = 0  # union bounded-step residual columns (S2 shape)
    n_oracle_points: int = 0

    @property
    def taken(self) -> bool:
        return self.verdict in ("S1", "S2")

    def format_line(self) -> str:
        head = (
            f"[{self.source}] chain {self.chain_depth}, "
            f"{self.n_members} members, {self.n_synthesized} synth"
        )
        if self.taken:
            worst = max(
                (m.s1.total_bound if self.verdict == "S1" else m.s2.total_bound)
                for m in self.members
            )
            kinks = max(m.n_kinks for m in self.members)
            line = (
                f"{self.verdict:8s}{head} -> {self.depth_after}: "
                f"kinks {kinks}, bound {worst:.2e}"
            )
            if self.verdict == "S2":
                line += f", stage-1 cols {self.stage1_cols}"
            return line
        return f"{'--':8s}{head} — {self.verdict}"

    def format_detail(self) -> str:
        """The subgraph line plus one line per synthesized member."""
        lines = [self.format_line()]
        for m in self.members:
            lines.append(
                f"    {m.name} (d={m.d_output}, depth {m.depth}): "
                f"kinks {m.n_kinks}, dev {m.deviation:.2e} "
                f"(in-band {m.fillet_deviation:.2e}); "
                f"S1 {m.s1.lanes} lanes, bound {m.s1.total_bound:.2e} "
                f"[{'ok' if m.s1_ok else 'no'}]; "
                f"S2 {m.s2.n_steps} steps, bound {m.s2.total_bound:.2e}, "
                f"fillet class {m.s2.fillet_bound:.2e} "
                f"[{'ok' if m.s2_ok else 'no'}]"
            )
        return "\n".join(lines)


@dataclass
class V2Report:
    """All subgraph verdicts + the modeled floor and liveness totals."""

    subgraphs: List[SubgraphAnalysis]
    floor_off: Optional[int] = None
    floor_on: Optional[int] = None
    machine: Optional[str] = None

    @property
    def total_stage1_cols(self) -> int:
        """Simultaneous S2 residual columns, summed across taken
        subgraphs — the conservative liveness estimate (all chains
        live at once)."""
        return sum(s.stage1_cols for s in self.subgraphs if s.verdict == "S2")

    def format(self) -> str:
        lines = [s.format_line() for s in self.subgraphs]
        taken = [s for s in self.subgraphs if s.taken]
        lines.append(
            f"-> {len(taken)} taken "
            f"(S1: {sum(1 for s in taken if s.verdict == 'S1')}, "
            f"S2: {sum(1 for s in taken if s.verdict == 'S2')}), "
            f"{len(self.subgraphs) - len(taken)} not taken"
        )
        if self.floor_off is not None:
            lines.append(
                f"modeled layer floor: {self.floor_off} -> {self.floor_on} "
                f"(delta {self.floor_on - self.floor_off:+d})"
            )
        lines.append(
            f"S2 stage-1 residual columns, simultaneous: "
            f"{self.total_stage1_cols} (adds to the existing residual "
            f"peak — the reverted-36-layer liveness trap check)"
        )
        return "\n".join(lines)


def _stage1_col_union(members: List[MemberAnalysis], certs, plateau_tol) -> int:
    """Bounded-step residual columns for one source's S2 emission:
    steps shared across members where their transitions coincide."""
    mids: List[float] = []
    for cert in certs:
        _, runs = transition_runs(cert.fn, plateau_tol)
        for i, j in runs:
            mids.append(float((cert.fn.x[i] + cert.fn.x[j]) / 2.0))
    if not mids:
        return 0
    return int(torch.unique(_snap_f32(torch.tensor(mids))).numel())


def analyze_collapse_v2(
    output_node: Node,
    *,
    lane_cap: int,
    budget: float = _SYNTH_CLAIM_ATOL,
    max_kinks: int = 100_000,
    model_floor: bool = True,
    verbose: bool = False,
) -> V2Report:
    """Analyze every univariate subgraph of a lowered copy for v2.

    Args:
        output_node: a throwaway lowered copy's output (v1 collapse
            already applied by the caller's ``lower()``).  When
            ``model_floor`` is on, this graph IS REWIRED in place for
            the floor-on measurement — never pass a graph you keep.
        lane_cap: per-FFN emitted-lane decline threshold
            (``d_hidden // 4`` of the target compile).
        budget: the synthesized claim's tolerance (v1's 1e-3).
        max_kinks: candidate-kink backstop per member.
        model_floor: model the layer floor off/on via stand-in nodes +
            ``critical_path_layers``.
        verbose: print one line per subgraph as verdicts land.
    """
    order = topological_order(output_node)
    src = scalar_sources(order)
    machine = _machine(order)

    by_src: Dict[Node, List[Node]] = {}
    for n in order:
        s = src[n]
        if s is not None and s is not n:
            by_src.setdefault(s, []).append(n)

    consumers: Dict[Node, List[Node]] = {n: [] for n in order}
    for n in order:
        for u in n.inputs:
            if u in consumers:
                consumers[u].append(n)

    topo_index = {n: i for i, n in enumerate(order)}
    subgraphs: List[SubgraphAnalysis] = []
    # (source, synthesized member, its shape, its certificate) for the
    # floor model's rewiring pass.
    rewiring: List[Tuple[Node, Node, str]] = []

    for source in sorted(by_src, key=topo_index.__getitem__):
        members = by_src[source]
        member_set = set(members)
        depth = _member_depths(source, members)
        chain_depth = max(depth[m] for m in members)
        src_name = source.name or f"{type(source).__name__}#{topo_index[source]}"
        ann = source.annotation or "-"

        boundary = [
            m
            for m in members
            if m is output_node or any(c not in member_set for c in consumers[m])
        ]
        synthesized = [m for m in boundary if depth[m] >= 2]

        def outcome(verdict: str, **kw) -> SubgraphAnalysis:
            return SubgraphAnalysis(
                source=src_name,
                annotation=ann,
                domain=kw.pop("domain", (float("nan"), float("nan"))),
                n_members=len(members),
                chain_depth=chain_depth,
                n_boundary=len(boundary),
                n_synthesized=len(synthesized),
                verdict=verdict,
                **kw,
            )

        if not synthesized:
            subgraphs.append(outcome("no depth gain"))
            continue
        if machine == "mixed":
            subgraphs.append(outcome("declined: graph mixes relu and swish FFNs"))
            continue

        cert = certify_subgraph(source, members, max_kinks=max_kinks)
        if cert.declined is not None:
            subgraphs.append(outcome(f"declined: {cert.declined}", domain=cert.domain))
            continue

        analyses: List[MemberAnalysis] = []
        for m in synthesized:
            c: MemberCertificate = cert.members[m]
            s1 = model_s1(c.fn, c.deviation)
            s2 = model_s2(c.fn, c.deviation, machine=machine, plateau_tol=budget)
            analyses.append(
                MemberAnalysis(
                    name=m.name or f"{type(m).__name__}#{topo_index[m]}",
                    d_output=m.d_output,
                    depth=depth[m],
                    n_kinks=c.n_kinks,
                    deviation=c.deviation,
                    deviation_at=c.deviation_at,
                    fillet_deviation=c.fillet_deviation,
                    s1=s1,
                    s2=s2,
                    linear_ok=c.linear(budget),
                    s1_ok=c.linear(budget) and s1.admissible(lane_cap, budget),
                    s2_ok=c.linear(budget) and s2.admissible(lane_cap, budget),
                )
            )

        not_linear = [a for a in analyses if not a.linear_ok]
        if not_linear:
            a = max(not_linear, key=lambda a: a.deviation)
            subgraphs.append(
                outcome(
                    f"declined: not PL within budget (member {a.name} "
                    f"deviates {a.deviation:.2e} at x={a.deviation_at:.6g})",
                    domain=cert.domain,
                    members=analyses,
                    n_oracle_points=cert.n_oracle_points,
                )
            )
            continue

        if all(a.s1_ok for a in analyses):
            verdict, depth_after = "S1", 1
        elif all(a.s1_ok or a.s2_ok for a in analyses):
            verdict, depth_after = "S2", 2
        else:
            bad = next(a for a in analyses if not (a.s1_ok or a.s2_ok))
            subgraphs.append(
                outcome(
                    f"declined: member {bad.name} fits neither shape "
                    f"(S1: {bad.s1.lanes} lanes, bound {bad.s1.total_bound:.2e}; "
                    f"S2: {bad.s2.stage1_lanes} stage-1 lanes, "
                    f"bound {bad.s2.total_bound:.2e}; cap {lane_cap}, "
                    f"budget {budget:g})",
                    domain=cert.domain,
                    members=analyses,
                    n_oracle_points=cert.n_oracle_points,
                )
            )
            continue

        if depth_after >= chain_depth:
            subgraphs.append(
                outcome(
                    f"no depth gain ({verdict} shape needs {depth_after} "
                    f"sublayers, chain is {chain_depth})",
                    domain=cert.domain,
                    members=analyses,
                    n_oracle_points=cert.n_oracle_points,
                )
            )
            continue

        stage1_cols = 0
        if verdict == "S2":
            stage1_cols = _stage1_col_union(
                analyses, [cert.members[m] for m in synthesized], budget
            )
        sg = outcome(
            verdict,
            domain=cert.domain,
            depth_after=depth_after,
            members=analyses,
            stage1_cols=stage1_cols,
            n_oracle_points=cert.n_oracle_points,
        )
        subgraphs.append(sg)
        for m in synthesized:
            rewiring.append((source, m, verdict))
        if verbose:
            print(f"  {sg.format_line()}")

    report = V2Report(subgraphs=subgraphs, machine=machine)

    if model_floor:
        from torchwright.compiler.forward.cpsat_scheduler import critical_path_layers

        report.floor_off = critical_path_layers(output_node)
        output_node = _graft_standins(output_node, rewiring, consumers, by_src)
        report.floor_on = critical_path_layers(output_node)

    if verbose:
        print(report.format())
    return report


def _standin_ffn(inp: Node, d_out: int, name: str) -> FFN:
    return FFN(
        inp,
        gate_proj=torch.zeros(1, len(inp)),
        gate_bias=torch.zeros(1),
        out_proj=torch.zeros(1, d_out),
        out_bias=torch.zeros(d_out),
        name=name,
    )


def _graft_standins(
    output_node: Node,
    rewiring: List[Tuple[Node, Node, str]],
    consumers: Dict[Node, List[Node]],
    by_src: Dict[Node, List[Node]],
) -> Node:
    """Replace each taken member's external consumers with 1- (S1) or
    2-sublayer (S2) stand-in FFNs fed by the source, for the floor
    model.  Stage-1 stand-ins are shared per source, mirroring the S2
    emission's shared bounded-step stage."""
    stage1: Dict[Node, Node] = {}
    for source, m, verdict in rewiring:
        member_set = set(by_src[source])
        if verdict == "S1":
            new = _standin_ffn(source, m.d_output, f"v2_s1_{m.name or m.node_id}")
        else:
            s1 = stage1.get(source)
            if s1 is None:
                s1 = _standin_ffn(
                    source, 1, f"v2_s2_steps_{source.name or source.node_id}"
                )
                stage1[source] = s1
            new = _standin_ffn(s1, m.d_output, f"v2_s2_{m.name or m.node_id}")
        for c in list(consumers[m]):
            if c not in member_set:
                c.replace_input(m, new)
        if m is output_node:
            output_node = new
    return output_node
