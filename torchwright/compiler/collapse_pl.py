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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from torchwright.compiler.collapse import (
    _MIN_SYNTH_DEPTH,
    _SYNTH_CLAIM_ATOL,
    CollapseReport,
    SubgraphOutcome,
    _index_subgraphs,
    _machine,
    _member_depths,
    _node_label,
    _piecewise_linear_fn,
    _seeded_oracle,
)
from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.pl_function import (
    _HINGE_EXACT_Z,
    SIMPLIFY_TOL,
    PLFunction,
    SubgraphCertificate,
    _in_intervals,
    band_skeleton,
    certify_subgraph,
    model_s1,
)
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import LiteralValue

if TYPE_CHECKING:
    from torchwright.graph import Node
    from torchwright.graph.optimize import FoldLog

_F64 = torch.float64

# Cap on the emission's gate-row amplification.  input_scale narrows
# the swish fillets (dip = swish_dip·|Δm|/(scale·input_scale)); the
# cap keeps gate biases (-input_scale·knot) well inside fp32 exponent
# range for any certified knot position.
_MAX_INPUT_SCALE = 2.0**20

# Below this, a skeleton's slope deltas are float64 noise around a flat
# function — synthesize a constant instead of a degenerate one-lane PWL.
_FLAT_SLOPE_ATOL = 1e-12

# Kink pre-screen (2026-07-06, the default-flip cost gate): a member
# whose candidate-kink population exceeds this multiple of the lane
# cap declines before its oracle sweep — the sweep is what made the
# certify walk double suite wall time.  S1 lanes track skeleton knots
# (bounded by candidates), so a population far past the cap cannot
# emit admissibly; the factor is set from the measured margins — every
# realized take peaks at 0.4x its lane cap (scratchpad, 306 @ cap
# 768) while the expensive declines run 45x-800x over (adder_v2
# 29,609 @ cap 256; calculator_simple 213,918 @ cap 256).  A subgraph
# whose candidates exceed the screen but would simplify below the cap
# is a lost take by construction; nothing measured is in that class.
_KINK_PRESCREEN_FACTOR = 4


def _emit_s1(source: Node, skel: PLFunction, ctx: _PLPassCtx, *, name: str) -> Node:
    """One interpolating ``piecewise_linear`` (or constant) from the source.

    Computes the certified skeleton's values from the source.
    """
    deltas = skel.slope_deltas().abs()
    if not bool((deltas > _FLAT_SLOPE_ATOL).any()):
        return LiteralValue(skel.y[0].to(torch.float32).clone().reshape(-1), name=name)

    breakpoints = [float(v) for v in skel.x.tolist()]
    rows = {
        b: [float(v) for v in row]
        for b, row in zip(breakpoints, skel.y.tolist(), strict=False)
    }

    input_scale = 1.0
    if ctx.machine == "swish":
        from torchwright.ops.const import scale, swish_dip

        max_dm = float(deltas.max())
        # Dip at or below half the claim budget; the emission-time
        # verification measures the realized value.
        input_scale = min(
            max(1.0, 2.0 * swish_dip * max_dm / (scale * ctx.budget)),
            _MAX_INPUT_SCALE,
        )

    piecewise_linear = _piecewise_linear_fn(ctx.machine)
    return piecewise_linear(
        source,
        breakpoints,
        lambda x: rows[x],
        clamp=True,
        d_max=ctx.lane_cap,
        input_scale=input_scale,
        name=name,
    )


def _verify_emission(
    new: Node,
    member: Node,
    members: list[Node],
    source: Node,
    skel: PLFunction,
    bands: torch.Tensor | None,
    budget: float,
) -> tuple[bool, float, float]:
    """Compare the emitted node's exact compute against the ORIGINAL member's oracle.

    Compared at every skeleton knot and segment midpoint outside the
    analytic hinge bands.  The original — not the skeleton — is the
    claim's reference: comparing against the skeleton would be circular
    (a degenerate skeleton trivially matches its own emission).
    Returns ``(ok, worst, at)``.
    """
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


@dataclass(frozen=True)
class _PLPassCtx:
    """Loop-invariant context shared by the per-subgraph S1 steps."""

    consumers: dict[Node, list[Node]]
    topo_index: dict[Node, int]
    machine: str | None
    lane_cap: int
    budget: float


@dataclass
class _PLSubgraph:
    """One univariate subgraph's derived structure (pre-gating)."""

    source: Node
    src_name: str
    members: list[Node]
    member_set: set[Node]
    boundary: list[Node]
    depth: dict[Node, int]
    chain_depth: int
    synthesized: list[Node]


def _pl_subgraph(
    source: Node, members: list[Node], output_node: Node, ctx: _PLPassCtx
) -> _PLSubgraph:
    """Derive one subgraph's boundary/synthesized structure."""
    member_set = set(members)
    depth = _member_depths(source, members)
    boundary = [
        m
        for m in members
        if m is output_node or any(c not in member_set for c in ctx.consumers[m])
    ]
    return _PLSubgraph(
        source=source,
        src_name=_node_label(source, ctx.topo_index),
        members=members,
        member_set=member_set,
        boundary=boundary,
        depth=depth,
        chain_depth=max(depth[m] for m in members),
        synthesized=[m for m in boundary if depth[m] >= _MIN_SYNTH_DEPTH],
    )


def _declined_pl(sub: _PLSubgraph, reason: str) -> SubgraphOutcome:
    return SubgraphOutcome(
        source=sub.src_name,
        n_members=len(sub.members),
        chain_depth=sub.chain_depth,
        collapsed=False,
        reason=reason,
    )


def _s1_gate(
    sub: _PLSubgraph, cert: SubgraphCertificate, ctx: _PLPassCtx
) -> tuple[dict[Node, PLFunction], str | None]:
    """Strict chord certificate + S1 admissibility, all synthesized members.

    The emitted shape is the band skeleton.  Returns (skeletons, fail).
    """
    skels: dict[Node, PLFunction] = {}
    for m in sub.synthesized:
        c: Any = cert.members[m]
        m_name = _node_label(m, ctx.topo_index)
        if not c.linear(ctx.budget):
            return skels, (
                f"not PL within budget (member {m_name} deviates "
                f"{c.deviation:.2e} at x={c.deviation_at:.6g})"
            )
        skel = band_skeleton(c.fn, c.bands)
        s1 = model_s1(skel, c.deviation + SIMPLIFY_TOL)
        if not s1.admissible(ctx.lane_cap, ctx.budget):
            return skels, (
                f"S1 inadmissible for member {m_name} "
                f"({s1.lanes} lanes, bound {s1.total_bound:.2e}; "
                f"cap {ctx.lane_cap}, budget {ctx.budget:g})"
            )
        skels[m] = skel
    return skels, None


def _emit_gate(
    sub: _PLSubgraph,
    cert: SubgraphCertificate,
    skels: dict[Node, PLFunction],
    ctx: _PLPassCtx,
) -> tuple[dict[Node, Node], str | None]:
    """Emit + verify every synthesized member BEFORE any rewiring.

    A verification failure must leave the graph untouched.  Returns
    (emitted, fail).
    """
    emitted: dict[Node, Node] = {}
    for m in sub.synthesized:
        m_name = m.name or f"m{ctx.topo_index[m]}"
        new = _emit_s1(
            sub.source, skels[m], ctx, name=f"collapse_pl_{sub.src_name}_{m_name}"
        )
        ok, worst, at = _verify_emission(
            new, m, sub.members, sub.source, skels[m], cert.members[m].bands, ctx.budget
        )
        if not ok:
            return emitted, (
                f"emission verification failed for member {m_name} "
                f"(|emitted - original| = {worst:.2e} at x={at:.6g}, "
                f"budget {ctx.budget:g})"
            )
        emitted[m] = new
    return emitted, None


def _apply_pl_collapse(
    sub: _PLSubgraph,
    emitted: dict[Node, Node],
    ctx: _PLPassCtx,
    output_node: Node,
    fold_log: FoldLog,
) -> tuple[Node, SubgraphOutcome]:
    """Rewire and orphan (v1's skeleton verbatim)."""
    kept = {m for m in sub.boundary if sub.depth[m] < _MIN_SYNTH_DEPTH}
    freed = sum(
        m.gate_proj.shape[0]
        for m in sub.members
        if isinstance(m, FFN) and m not in kept
    )
    n_orphaned = sum(len(m.checks) for m in sub.members if m not in kept)
    emitted_lanes = 0
    for m in sub.synthesized:
        new = emitted[m]
        if isinstance(new, FFN):
            emitted_lanes += new.gate_proj.shape[0]
        for c in list(ctx.consumers[m]):
            if c not in sub.member_set:
                c.replace_input(m, new)
        if m is output_node:
            output_node = new
        fold_log.record_move(orphan=m, survivor=new)

    outcome = SubgraphOutcome(
        source=sub.src_name,
        n_members=len(sub.members),
        chain_depth=sub.chain_depth,
        collapsed=True,
        n_synthesized=len(sub.synthesized),
        n_kept=len(sub.boundary) - len(sub.synthesized),
        emitted_lanes=emitted_lanes,
        freed_lanes=freed,
        n_checks_orphaned=n_orphaned,
    )
    return output_node, outcome


def collapse_pl_subgraphs(
    output_node: Node,
    *,
    lane_cap: int,
    fold_log: FoldLog,
    budget: float = _SYNTH_CLAIM_ATOL,
    max_kinks: int = 100_000,
    verbose: bool = False,
) -> tuple[Node, CollapseReport]:
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
        max_kinks: candidate-kink backstop per member.  The effective
            per-member ceiling is ``min(max_kinks,
            _KINK_PRESCREEN_FACTOR * lane_cap)`` — the pre-screen that
            keeps the certify walk off graphs that could never emit
            within the cap.
        verbose: print one line per subgraph outcome.

    Returns:
        ``(output_node, report)`` — the (possibly replaced) output and
        the per-subgraph log.
    """
    order = topological_order(output_node)
    machine = _machine(order)
    by_src, consumers, topo_index = _index_subgraphs(order)
    ctx = _PLPassCtx(
        consumers=consumers,
        topo_index=topo_index,
        machine=machine,
        lane_cap=lane_cap,
        budget=budget,
    )
    outcomes: list[SubgraphOutcome] = []

    for source in sorted(by_src, key=topo_index.__getitem__):
        sub = _pl_subgraph(source, by_src[source], output_node, ctx)
        if not sub.synthesized:
            outcomes.append(
                _declined_pl(sub, "no depth gain (no boundary member at depth >= 2)")
            )
            continue
        if machine == "mixed":
            outcomes.append(_declined_pl(sub, "graph mixes relu and swish FFNs"))
            continue

        cert = certify_subgraph(
            source,
            sub.members,
            # The pre-screen: the lane-cap-derived candidate ceiling
            # (see _KINK_PRESCREEN_FACTOR); max_kinks stays the
            # absolute backstop.
            max_kinks=min(max_kinks, _KINK_PRESCREEN_FACTOR * lane_cap),
            hinge_exact=_HINGE_EXACT_Z if machine == "swish" else 0.0,
        )
        if cert.declined is not None:
            outcomes.append(_declined_pl(sub, cert.declined))
            continue

        skels, fail = _s1_gate(sub, cert, ctx)
        if fail is not None:
            outcomes.append(_declined_pl(sub, fail))
            continue

        emitted, fail = _emit_gate(sub, cert, skels, ctx)
        if fail is not None:
            outcomes.append(_declined_pl(sub, fail))
            continue

        output_node, outcome = _apply_pl_collapse(
            sub, emitted, ctx, output_node, fold_log
        )
        outcomes.append(outcome)

    report = CollapseReport(outcomes)
    if verbose and outcomes:
        print("univariate collapse v2 (S1):")
        for line in report.format().splitlines():
            print(f"  {line}")
    return output_node, report
