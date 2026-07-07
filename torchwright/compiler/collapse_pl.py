"""Univariate collapse v2 — the S1 emitter (general PL synthesis).

Runs AFTER the shipping v1 staircase pass and takes what v1 leaves:
any univariate subgraph whose boundary members certify as
piecewise-linear under the composed-PL certificate
(:mod:`torchwright.compiler.pl_function`) and fit the **S1 shape** —
one interpolating ``piecewise_linear`` FFN per synthesized member,
chain → 1.

Descoped per Rob's 2026-07-06 decision (the floor-vs-budget sweep):

* **strict policy only** — chargeable chord deviation ≤ the production
  ``_SYNTH_CLAIM_ATOL`` (1e-3); no banded excusals, no budget knob;
* **S1 only** — the two-sublayer S2 shape is decided separately, on
  the measured S1-only floor;
* the emitted shape is the certificate's **band skeleton** (in-band
  knots trace the machine's own dip/tail; the emission's own hinges
  recreate their own in-band shape — the inherited-ramp clause).

The emitted node reuses the ops-level ``piecewise_linear`` builder
verbatim (vector-valued ``fn``, clamp semantics matching the
certificate's domain-clamped tails, one lane per slope change — the
exact geometry ``model_s1`` priced).  ``input_scale`` is chosen so the
swish fillet dip ``swish_dip·|Δm|/(scale·input_scale)`` sits at or
below half the claim budget.

Every emission is verified empirically before rewiring: the new
node's exact compute is compared against the certified skeleton at
every skeleton knot and segment midpoint outside the analytic hinge
bands; a violation declines the subgraph (safety net — the modeled
admissibility should make it unreachable).

Checks attached to replaced members are **orphaned and counted**
(``SubgraphOutcome.n_checks_orphaned``): predicates are opaque
closures, so they cannot be re-targeted at the synthesized value with
a widened tolerance.  v1 orphaned silently; the count keeps the
coverage loss visible.  Range claims are NOT lost — the emitted
``piecewise_linear`` carries its own clamp-range claim as metadata.

Determinism: the certificate walk and the emission are pure float64
functions of the copy in topological order — safe next to the CP-SAT
schedule-cache key.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from torchwright.compiler.collapse import (
    _SYNTH_CLAIM_ATOL,
    CollapseReport,
    SubgraphOutcome,
    _machine,
    _member_depths,
    _piecewise_linear_fn,
    _seeded_oracle,
    scalar_sources,
)
from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.pl_function import (
    SIMPLIFY_TOL,
    _HINGE_EXACT_Z,
    PLFunction,
    _in_intervals,
    band_skeleton,
    certify_subgraph,
    model_s1,
)
from torchwright.graph import Node
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import LiteralValue
from torchwright.graph.optimize import FoldLog

_F64 = torch.float64

# Cap on the emission's gate-row amplification.  input_scale narrows
# the swish fillets (dip = swish_dip·|Δm|/(scale·input_scale)); the
# cap keeps gate biases (-input_scale·knot) well inside fp32 exponent
# range for any certified knot position.
_MAX_INPUT_SCALE = 2.0**20


def _emit_s1(
    source: Node,
    skel: PLFunction,
    machine: Optional[str],
    lane_cap: int,
    budget: float,
    name: str,
) -> Node:
    """One interpolating ``piecewise_linear`` (or constant) computing
    the certified skeleton's values from the source."""
    deltas = skel.slope_deltas().abs()
    if not bool((deltas > 1e-12).any()):
        return LiteralValue(skel.y[0].to(torch.float32).clone().reshape(-1), name=name)

    breakpoints = [float(v) for v in skel.x.tolist()]
    rows = {b: [float(v) for v in row] for b, row in zip(breakpoints, skel.y.tolist())}

    input_scale = 1.0
    if machine == "swish":
        from torchwright.ops.const import scale, swish_dip

        max_dm = float(deltas.max())
        # Dip at or below half the claim budget; the emission-time
        # verification measures the realized value.
        input_scale = min(
            max(1.0, 2.0 * swish_dip * max_dm / (scale * budget)),
            _MAX_INPUT_SCALE,
        )

    piecewise_linear = _piecewise_linear_fn(machine)
    return piecewise_linear(
        source,
        breakpoints,
        lambda x: rows[x],
        clamp=True,
        d_max=lane_cap,
        input_scale=input_scale,
        name=name,
    )


def _verify_emission(
    new: Node,
    member: Node,
    members: List[Node],
    source: Node,
    skel: PLFunction,
    bands: Optional[torch.Tensor],
    budget: float,
) -> Tuple[bool, float, float]:
    """Compare the emitted node's exact compute against the ORIGINAL
    member's oracle at every skeleton knot and segment midpoint outside
    the analytic hinge bands.  The original — not the skeleton — is the
    claim's reference: comparing against the skeleton would be circular
    (a degenerate skeleton trivially matches its own emission).
    Returns ``(ok, worst, at)``."""
    xs = torch.unique(torch.cat([skel.x, (skel.x[1:] + skel.x[:-1]) / 2.0]))
    if bands is not None and bands.numel():
        xs = xs[~_in_intervals(bands, xs, strict=True)]
    if not xs.numel():
        return True, 0.0, 0.0
    grid = xs.to(torch.float32).reshape(-1, 1)
    got = _seeded_oracle([new], source, grid)[new]
    want = _seeded_oracle(members, source, grid)[member]
    err = (got.to(_F64) - want.to(_F64)).abs().amax(dim=1)
    i = int(err.argmax())
    worst = float(err[i])
    return worst <= budget, worst, float(xs[i])


def collapse_pl_subgraphs(
    output_node: Node,
    *,
    lane_cap: int,
    fold_log: FoldLog,
    budget: float = _SYNTH_CLAIM_ATOL,
    max_kinks: int = 100_000,
    verbose: bool = False,
) -> Tuple[Node, CollapseReport]:
    """Run the v2 S1 pass on the compiler-private copy (after v1).

    Args:
        output_node: the copy's output after fusion and the v1 pass.
        lane_cap: decline threshold on emitted lanes per synthesized
            FFN (``d_hidden / 4`` of the target compile).
        fold_log: the lowering run's fold log; boundary-member value
            moves are recorded so ``lower()``'s node_map stays
            value-correct.
        budget: the synthesized claim's tolerance (the production
            1e-3 — Rob's descope pins it; the knob exists for tests).
        max_kinks: candidate-kink backstop per member.
        verbose: print one line per subgraph outcome.

    Returns:
        ``(output_node, report)`` — the (possibly replaced) output and
        the per-subgraph log.
    """
    order = topological_order(output_node)
    src = scalar_sources(order)
    machine = _machine(order)

    by_src: Dict[Node, List[Node]] = {}
    for n in order:  # topological, so member lists stay topological
        s = src[n]
        if s is not None and s is not n:
            by_src.setdefault(s, []).append(n)

    consumers: Dict[Node, List[Node]] = {n: [] for n in order}
    for n in order:
        for u in n.inputs:
            if u in consumers:
                consumers[u].append(n)

    topo_index = {n: i for i, n in enumerate(order)}
    outcomes: List[SubgraphOutcome] = []

    for source in sorted(by_src, key=topo_index.__getitem__):
        members = by_src[source]
        member_set = set(members)
        depth = _member_depths(source, members)
        chain_depth = max(depth[m] for m in members)
        src_name = source.name or f"{type(source).__name__}#{topo_index[source]}"

        def declined(reason: str) -> SubgraphOutcome:
            return SubgraphOutcome(
                source=src_name,
                n_members=len(members),
                chain_depth=chain_depth,
                collapsed=False,
                reason=reason,
            )

        boundary = [
            m
            for m in members
            if m is output_node or any(c not in member_set for c in consumers[m])
        ]
        synthesized = [m for m in boundary if depth[m] >= 2]
        if not synthesized:
            outcomes.append(
                declined("no depth gain (no boundary member at depth >= 2)")
            )
            continue
        if machine == "mixed":
            outcomes.append(declined("graph mixes relu and swish FFNs"))
            continue

        cert = certify_subgraph(
            source,
            members,
            max_kinks=max_kinks,
            hinge_exact=_HINGE_EXACT_Z if machine == "swish" else 0.0,
        )
        if cert.declined is not None:
            outcomes.append(declined(cert.declined))
            continue

        # Strict chord certificate + S1 admissibility, all synthesized
        # members (the emitted shape is the band skeleton).
        skels: Dict[Node, PLFunction] = {}
        fail = None
        for m in synthesized:
            c = cert.members[m]
            m_name = m.name or f"{type(m).__name__}#{topo_index[m]}"
            if not c.linear(budget):
                fail = (
                    f"not PL within budget (member {m_name} deviates "
                    f"{c.deviation:.2e} at x={c.deviation_at:.6g})"
                )
                break
            skel = band_skeleton(c.fn, c.bands)
            s1 = model_s1(skel, c.deviation + SIMPLIFY_TOL)
            if not s1.admissible(lane_cap, budget):
                fail = (
                    f"S1 inadmissible for member {m_name} "
                    f"({s1.lanes} lanes, bound {s1.total_bound:.2e}; "
                    f"cap {lane_cap}, budget {budget:g})"
                )
                break
            skels[m] = skel
        if fail is not None:
            outcomes.append(declined(fail))
            continue

        # Emit + verify every synthesized member BEFORE any rewiring —
        # a verification failure must leave the graph untouched.
        emitted: Dict[Node, Node] = {}
        for m in synthesized:
            m_name = m.name or f"m{topo_index[m]}"
            new = _emit_s1(
                source,
                skels[m],
                machine,
                lane_cap,
                budget,
                name=f"collapse_pl_{src_name}_{m_name}",
            )
            ok, worst, at = _verify_emission(
                new, m, members, source, skels[m], cert.members[m].bands, budget
            )
            if not ok:
                fail = (
                    f"emission verification failed for member {m_name} "
                    f"(|emitted - original| = {worst:.2e} at x={at:.6g}, "
                    f"budget {budget:g})"
                )
                break
            emitted[m] = new
        if fail is not None:
            outcomes.append(declined(fail))
            continue

        # --- Rewire and orphan (v1's skeleton verbatim). ---------------
        kept = {m for m in boundary if depth[m] < 2}
        freed = sum(
            m.gate_proj.shape[0]
            for m in members
            if isinstance(m, FFN) and m not in kept
        )
        n_orphaned = sum(len(m.checks) for m in members if m not in kept)
        emitted_lanes = 0
        for m in synthesized:
            new = emitted[m]
            if isinstance(new, FFN):
                emitted_lanes += new.gate_proj.shape[0]
            for c in list(consumers[m]):
                if c not in member_set:
                    c.replace_input(m, new)
            if m is output_node:
                output_node = new
            fold_log.record_move(orphan=m, survivor=new)

        outcomes.append(
            SubgraphOutcome(
                source=src_name,
                n_members=len(members),
                chain_depth=chain_depth,
                collapsed=True,
                n_synthesized=len(synthesized),
                n_kept=len(boundary) - len(synthesized),
                emitted_lanes=emitted_lanes,
                freed_lanes=freed,
                n_checks_orphaned=n_orphaned,
            )
        )

    report = CollapseReport(outcomes)
    if verbose and outcomes:
        print("univariate collapse v2 (S1):")
        for line in report.format().splitlines():
            print(f"  {line}")
    return output_node, report
