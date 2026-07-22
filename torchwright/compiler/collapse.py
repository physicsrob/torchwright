"""Univariate-subgraph collapse — the depth pass of
``docs/univariate_collapse_plan.md``.

A **univariate subgraph** is a set of per-position nodes (FFN, Linear,
Add, Concatenate; literals allowed as extra inputs) whose only
non-literal source is a single 1-D node.  Every member computes a
function of that one scalar, so each externally-consumed member is one
univariate piecewise-linear function of the source — computable by a
single staircase ``piecewise_linear`` FFN in one MLP sublayer.  This
pass finds those subgraphs on the compiler-private copy inside
``lower()`` (after linear fusion), re-synthesizes each boundary
member that sits at depth >= 2 above the source, rewires its external
consumers, and orphans the interior.  Depth drops from the subgraph's
chain length to 1.

The pass never touches a source graph: it runs only on the throwaway
clone ``lower()`` built, and reports value moves through the same
``FoldLog`` linear fusion uses so the source→copy ``node_map`` stays
value-correct.  It must be a deterministic function of the copy —
subgraphs and members are visited in topological order and synthesized
nodes are named from source/member names — because the CP-SAT schedule
cache keys on the lowered copy's topology.

Feasibility gate (v1, the plateau contract — all four must hold):

1. **Integer contract**: the source carries an ``assert_integer``
   claim (``node.integer_claim`` — metadata attached in the user graph
   rides the copy).  Constancy near integers alone cannot
   rule out a half-integer-grained source, for which the synthesized
   staircase would return mid-ramp values at inputs the source
   actually takes.
2. **Plateau pre-screen**: the integer grid implied by the source's
   trusted ``value_range`` has at most ``lane_cap`` plateaus.
3. **Lane gate, on emitted lanes**: the staircase costs one hidden
   neuron per slope change — two per value-changing step — so
   ``lanes = 2 x #(adjacent plateau pairs whose values differ in any
   output dimension)``.  Decline past ``lane_cap``; ``d_max`` is the
   cap, so a passing member never chunks.
4. **Staircase check, composite error budget** (2026-07-06, decided
   with Rob): the exact oracle (each member's own ``compute``, seeded
   with the grid at the source) is evaluated at sampled offsets
   spanning ``[-w, +w]`` around every integer, ``w =
   1/(2*step_sharpness)`` — half the transition band of the machine's
   own step constructions.  The member's max deviation from its
   plateau-center value, **plus** the modeled fp32 lane-accumulation
   error of the synthesized staircase, must fit inside the tolerance
   the synthesized node's own range claim carries
   (``_SYNTH_CLAIM_ATOL``).  Relu compositions measure deviation
   exactly 0 (they compose exactly), so for them this is still the
   bit-identical check; swish compositions may carry an ulp-scale
   fillet-tail residue at the band edge, which is measured and charged
   against the budget rather than declined outright.

A subgraph failing any condition is declined — the collapse never
introduces more error than the synthesized claim tolerates — and the
decline is recorded with its reason in the :class:`CollapseReport`.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

import torch

from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import Node
from torchwright.graph.ffn import FFN
from torchwright.graph.linear import Linear
from torchwright.graph.misc import Add, Concatenate, LiteralValue
from torchwright.graph.optimize import FoldLog
from torchwright.ops.const import step_sharpness

# Per-position ops a univariate subgraph may contain.  Kept in step with
# scripts/measure_fusion_opportunities.py, which imports these from here.
POINTWISE = (FFN, Linear, Add, Concatenate)

_TOP = object()  # "not a function of a single scalar"

# Plateau slack: half the transition-band width of the machine's own
# step constructions (step pairs ramp over 1/step_sharpness).
_PLATEAU_SLACK = 1.0 / (2.0 * step_sharpness)

# Sampled offsets (in units of the slack) at which the composed
# function must equal its plateau-center value.  Index 0 is the center;
# the tabulation reads those rows.
_OFFSET_FRACTIONS = (0.0, -1.0, 1.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0)

# The synthesized staircase carries piecewise_linear's clamp-range claim
# at assert_matches_value_type's default tolerance.  This is the whole
# error budget of a collapse: the member's measured plateau-band
# deviation plus the predicted fp32 lane accumulation must fit inside
# it (the composite gate below) — decline instead of widening the
# claim.
_SYNTH_CLAIM_ATOL = 1e-3


def _ulp32(magnitude: float) -> float:
    """fp32 ulp at ``magnitude`` — the accumulation-error quantum of a
    lane sum whose intermediates reach that magnitude."""
    import math

    if magnitude <= 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(magnitude)) - 23)


def scalar_sources(order: List[Node]) -> Dict[Node, Optional[Node]]:
    """For each node, the single 1-D node it is a pointwise function of.

    Returns ``src[n] = s`` when n's only non-literal ancestry, walked
    through pointwise ops, is the 1-D node ``s`` (``s`` itself maps to
    ``s``).  ``None`` means literal-only ancestry or no single scalar
    source.

    Checks and range claims are node metadata, so an asserted value no
    longer interrupts the walk — a subgraph extends straight through a
    claim (the wrapper-node representation used to end it there).
    """
    raw: Dict[Node, object] = {}
    for n in order:
        if isinstance(n, LiteralValue):
            raw[n] = None
            continue
        if not n.inputs or not isinstance(n, POINTWISE):
            # A source candidate: any node computed "elsewhere" (input,
            # embedding, attention, ...) starts a univariate subgraph iff
            # it is 1-D.
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
            # The univariate subgraph ends here, but n itself can seed a
            # new one.
            raw[n] = n if len(n) == 1 else _TOP
        else:
            raw[n] = acc
    return {
        n: (cast(Node, s) if (s is not None and s is not _TOP) else None)
        for n, s in raw.items()
    }


@dataclass(frozen=True)
class SubgraphOutcome:
    """One subgraph's collapse-or-decline record (the pass's log line)."""

    source: str
    n_members: int
    chain_depth: int
    collapsed: bool
    reason: str = ""  # decline reason when not collapsed
    n_synthesized: int = 0
    n_kept: int = 0
    emitted_lanes: int = 0
    freed_lanes: int = 0
    # Checks on members orphaned by the rewiring (predicates are opaque
    # closures — they cannot be re-targeted at the synthesized value
    # with a widened tolerance, so they die with the subgraph and the
    # count keeps the coverage loss visible).  v1 orphaned silently;
    # the v2 pass fills this in.
    n_checks_orphaned: int = 0

    def format_line(self) -> str:
        if self.collapsed:
            orphaned = (
                f", {self.n_checks_orphaned} checks orphaned"
                if self.n_checks_orphaned
                else ""
            )
            return (
                f"collapsed [{self.source}]: {self.n_members} members, "
                f"chain {self.chain_depth} -> 1, "
                f"{self.n_synthesized} synthesized (+{self.emitted_lanes} lanes), "
                f"{self.n_kept} kept, {self.freed_lanes} lanes freed{orphaned}"
            )
        return (
            f"declined  [{self.source}]: {self.n_members} members, "
            f"chain {self.chain_depth} — {self.reason}"
        )


@dataclass
class CollapseReport:
    """All subgraph outcomes of one pass run, collapses and declines."""

    outcomes: List[SubgraphOutcome]

    @property
    def n_collapsed(self) -> int:
        return sum(1 for o in self.outcomes if o.collapsed)

    def format(self) -> str:
        return "\n".join(o.format_line() for o in self.outcomes)


def _seeded_oracle(
    members: List[Node], source: Node, grid: torch.Tensor
) -> Dict[Node, torch.Tensor]:
    """Evaluate every member's exact-math value with ``source`` pinned to
    ``grid`` — the re-rooted oracle of the plan's synthesis step.

    Uses :func:`torchwright.debug.probe.reference_eval` seeded at the
    source: each member's own ``compute`` runs (the same semantic
    definition the compiler trusts), the recursion short-circuits at the
    source, and one batched call covers the whole grid — nothing
    upstream of the source is ever evaluated.

    Attached checks are suppressed for the walk: the grid's "positions"
    are synthetic samples (every plateau × offset), so cross-position
    predicates (distinct-across, score gaps) would fire spuriously.
    The same predicates still run on real data through the source
    graph's checks.
    """
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.node import suppress_checks

    root = members[0] if len(members) == 1 else Concatenate(list(members))
    with suppress_checks():
        cache = reference_eval(root, {}, grid.shape[0], seed={source: grid})
    return {m: cache[m] for m in members}


def _member_depths(source: Node, members: List[Node]) -> Dict[Node, int]:
    """Depth in costly sublayers of each member above the source.

    Concatenate is virtual (cost 0); FFN/Linear/Add cost 1.  Inputs
    outside the subgraph (literals) contribute depth 0.
    """
    depth: Dict[Node, int] = {source: 0}
    for m in members:  # members arrive in topological order
        base = max((depth.get(u, 0) for u in m.inputs), default=0)
        depth[m] = base + (0 if isinstance(m, Concatenate) else 1)
    return depth


def _machine(all_nodes) -> Optional[str]:
    """The graph's FFN machine: 'relu', 'swish', None (no FFNs), or
    'mixed'."""
    activations = {n.activation for n in all_nodes if isinstance(n, FFN)}
    if not activations:
        return None
    if len(activations) > 1:
        return "mixed"
    return next(iter(activations))


def _piecewise_linear_fn(machine: Optional[str]):
    """The machine-matching staircase builder.  ``None`` (no FFNs in the
    graph) can only ever synthesize constants, which take the
    LiteralValue path before this builder is called — relu is a safe
    default for that unreachable case."""
    if machine == "swish":
        from torchwright.ops.swiglu.arithmetic_ops import piecewise_linear
    else:
        from torchwright.ops.relu.arithmetic_ops import piecewise_linear
    return piecewise_linear


def _synthesize_member(
    source: Node,
    table: torch.Tensor,
    lo_int: int,
    lane_cap: int,
    machine: Optional[str],
    name: str,
) -> Node:
    """One staircase ``piecewise_linear`` (or constant) computing
    ``member``'s tabulated values from the source.

    ``table`` is ``(n_plateaus, d)`` — the oracle at plateau centers.
    Breakpoints are step pairs at half-integers between value-changing
    plateaus, ``1/step_sharpness`` apart (the machine's step
    convention), with ``input_scale=step_sharpness`` so biases stay
    fp32-exact (relu) / hinges stay sharp past the spacing audit
    (swish).
    """
    changed = (table[1:] != table[:-1]).any(dim=1)
    if not bool(changed.any()):
        return LiteralValue(table[0].clone(), name=name)

    h = _PLATEAU_SLACK
    breakpoints: List[float] = []
    for i in torch.nonzero(changed, as_tuple=False).flatten().tolist():
        mid = lo_int + i + 0.5
        breakpoints.extend((mid - h, mid + h))

    values = table.tolist()
    n_max = len(values) - 1

    def fn(x: float) -> List[float]:
        idx = min(max(round(x) - lo_int, 0), n_max)
        return values[idx]

    piecewise_linear = _piecewise_linear_fn(machine)
    return piecewise_linear(
        source,
        breakpoints,
        fn,
        clamp=True,
        d_max=lane_cap,
        input_scale=step_sharpness,
        name=name,
    )


def collapse_univariate_subgraphs(
    output_node: Node,
    *,
    lane_cap: int,
    fold_log: FoldLog,
    verbose: bool = False,
) -> Tuple[Node, CollapseReport]:
    """Run the collapse pass on the compiler-private copy.

    Args:
        output_node: the copy's output after linear fusion.
        lane_cap: decline threshold on emitted lanes per synthesized
            FFN (``d_hidden / 4`` of the target compile).
        fold_log: the lowering run's fold log; boundary-member value
            moves are recorded here so ``lower()``'s node_map stays
            value-correct.
        verbose: print one line per subgraph outcome.

    Returns:
        ``(output_node, report)`` — the (possibly replaced) output and
        the per-subgraph collapse/decline log.  Synthesized values
        carry ``piecewise_linear``'s clamp-range claim as node
        metadata.
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

        # Gate 1 — integer contract on the source (metadata rode the copy).
        if not source.integer_claim:
            outcomes.append(declined("source carries no assert_integer"))
            continue

        # Gate 2 — plateau pre-screen on the trusted value_range.
        vr = source.value_type.value_range
        if not vr.is_finite():
            outcomes.append(declined("source value_range is unbounded"))
            continue
        import math

        lo_int = math.ceil(vr.lo)
        hi_int = math.floor(vr.hi)
        if hi_int < lo_int:
            outcomes.append(declined("value_range contains no integer"))
            continue
        n_plateaus = hi_int - lo_int + 1
        if n_plateaus > lane_cap:
            outcomes.append(
                declined(f"{n_plateaus} plateaus exceed the {lane_cap}-lane cap")
            )
            continue

        # Re-rooted oracle over the full sampled grid, one batched call.
        n_offsets = len(_OFFSET_FRACTIONS)
        ks = torch.arange(lo_int, hi_int + 1, dtype=torch.float32)
        offsets = torch.tensor(
            [f * _PLATEAU_SLACK for f in _OFFSET_FRACTIONS], dtype=torch.float32
        )
        grid = (ks.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 1)
        values = _seeded_oracle(synthesized, source, grid)

        # Gate 4 — plateau constancy, measured: each member's max
        # deviation from its plateau-center value across the sampled
        # band.  Relu compositions compose exactly (deviation 0); swish
        # fillet tails may leave an ulp-scale residue at the band edge.
        # A deviation past the whole budget means "not a staircase" —
        # decline with the location; a sub-budget deviation is charged
        # against the composite budget in gate 3 below.
        member_dev: Dict[Node, float] = {}
        stair_fail = None
        for m in synthesized:
            v = values[m].reshape(n_plateaus, n_offsets, -1)
            dev = float((v - v[:, :1, :]).abs().max())
            member_dev[m] = dev
            if dev > _SYNTH_CLAIM_ATOL:
                bad_k = (
                    int(torch.nonzero((v != v[:, :1, :]).any(dim=2).any(dim=1))[0])
                    + lo_int
                )
                m_name = m.name or f"{type(m).__name__}#{topo_index[m]}"
                stair_fail = (
                    f"not constant on the plateau at k={bad_k} "
                    f"(member {m_name}, max deviation {dev:.3g} > "
                    f"budget {_SYNTH_CLAIM_ATOL:g})"
                )
                break
        if stair_fail is not None:
            outcomes.append(declined(stair_fail))
            continue

        # Gate 3 — emitted lanes per synthesized member (2 per
        # value-changing step; equal-value steps are free), and the
        # composite error budget: a staircase's saturated ramp lanes
        # carry intermediates of magnitude ~ step_sharpness · R ·
        # max|adjacent value change|, and the fp32 lane sum quantizes
        # the recovered plateau to ulps of that magnitude (measured at
        # 1-4 ulps — the staircase entries in docs/op_noise_data.json).
        # The measured band deviation plus this predicted accumulation
        # must fit the synthesized claim's tolerance — decline rather
        # than exceed it.
        tables = {
            m: values[m].reshape(n_plateaus, n_offsets, -1)[:, 0, :]
            for m in synthesized
        }
        range_width = float(hi_int - lo_int)
        gate_fail = None
        member_lanes: Dict[Node, int] = {}
        for m in synthesized:
            t = tables[m]
            lanes = 2 * int((t[1:] != t[:-1]).any(dim=1).sum())
            member_lanes[m] = lanes
            m_name = m.name or f"{type(m).__name__}#{topo_index[m]}"
            if lanes > lane_cap:
                gate_fail = f"member {m_name} needs {lanes} lanes (cap {lane_cap})"
                break
            err_bound = 0.0
            if lanes:
                max_dv = float((t[1:] - t[:-1]).abs().max())
                err_bound = 4.0 * _ulp32(step_sharpness * range_width * max_dv)
            if member_dev[m] + err_bound > _SYNTH_CLAIM_ATOL:
                gate_fail = (
                    f"member {m_name}: band deviation {member_dev[m]:.2e} "
                    f"+ predicted fp32 accumulation {err_bound:.2e} "
                    f"exceeds the synthesized claim tolerance "
                    f"{_SYNTH_CLAIM_ATOL:g}"
                )
                break
        if gate_fail is not None:
            outcomes.append(declined(gate_fail))
            continue

        # --- All gates passed: synthesize, rewire, orphan. -------------
        # Kept boundary members (depth < 2) survive with their whole
        # in-subgraph input cone (which is depth-0 Concatenates over the
        # source and literals only, so no FFN survives through them);
        # every other member becomes unreachable once the synthesized
        # nodes take over, freeing its lanes.
        kept = {m for m in boundary if depth[m] < 2}
        freed = sum(
            m.gate_proj.shape[0]
            for m in members
            if isinstance(m, FFN) and m not in kept
        )
        for m in synthesized:
            m_name = m.name or f"m{topo_index[m]}"
            new = _synthesize_member(
                source,
                tables[m],
                lo_int,
                lane_cap,
                machine,
                name=f"collapse_{src_name}_{m_name}",
            )
            for c in list(consumers[m]):
                if c not in member_set:
                    c.replace_input(m, new)
            if m is output_node:
                output_node = new
            # The synthesized value IS m's value: record the move so the
            # source->copy node_map follows it.  (piecewise_linear's
            # clamp-range claim is metadata on ``new`` itself.)
            fold_log.record_move(orphan=m, survivor=new)

        outcomes.append(
            SubgraphOutcome(
                source=src_name,
                n_members=len(members),
                chain_depth=chain_depth,
                collapsed=True,
                n_synthesized=len(synthesized),
                n_kept=len(boundary) - len(synthesized),
                emitted_lanes=sum(member_lanes.values()),
                freed_lanes=freed,
            )
        )

    report = CollapseReport(outcomes)
    if verbose and outcomes:
        print("univariate collapse:")
        for line in report.format().splitlines():
            print(f"  {line}")
    return output_node, report
