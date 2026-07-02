"""Realization classes and the realization table.

A **realization class** is a write-path choice: which transformer hardware
computes a fixed, already-built graph node (docs/lowering_boundary_plan.md).
The node, its weights, and its consumers are unchanged across classes —
only the weight-writer emission differs.

This module is the **single declaration of the option set**
(:func:`candidate_classes`): it is read by ``lower()`` (which builds the
unresolved table), by the static resolver (``optimize=0``), and by the
CP-SAT model builder (``is_flex`` / static ``routing``).  Before this
module existed the option set was hard-coded twice — a policy read in the
eager scheduler and an ``isinstance`` check in the CP-SAT model — with
nothing keeping the two in agreement.

The **realization table** is the one artifact recording every schedulable
node's class.  Both resolvers write it; the layer walk only reads it.
``Add`` gets an explicitly *conditional* entry — its class is determined
by a predicate on schedule state (is an addend's column set dead when the
walker reaches it?), which is discovered during placement (eager walk) or
co-decided with layer assignment (the solve).  A conditional entry can
represent that; it is not an unresolved entry.
"""

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple

from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.graph import Add, Attn, Linear, Node
from torchwright.graph.block import Block
from torchwright.graph.misc import LiteralValue

# ---------------------------------------------------------------------------
# Realization classes (write-path choices)
# ---------------------------------------------------------------------------

#: Standalone Linear as rotary Δ=0 self-match attention head(s).
ATTN_TRANSPORT = "attn_transport"
#: Standalone Linear in the MLP via the bypass identity act(z) − act(−z) = z
#: (two hidden slots per output column; exact for relu and for swish).
MLP_BYPASS = "mlp_bypass"
#: An Attn node on attention heads (the only hardware that attends).
ATTN_HEADS = "attn_heads"
#: A Block on the MLP sublayer (the L→act→L composite; always MLP-locked).
MLP_COMPOSITE = "mlp_composite"
#: A LiteralValue written by the MLP constant path.
MLP_LITERAL = "mlp_literal"
#: An Add reusing a dead addend's residual columns (add_into; one head for
#: the live addend).
RESIDUAL_REUSE = "residual_reuse"
#: An Add copying both addends to fresh columns (compute_add; head(s) each).
ATTN_COPY = "attn_copy"

#: Which sublayer each class occupies.  Both Add classes are attention ops
#: (add_into and compute_add are AttnHeadOps).
CLASS_SUBLAYER: Dict[str, str] = {
    ATTN_TRANSPORT: "attn",
    ATTN_HEADS: "attn",
    RESIDUAL_REUSE: "attn",
    ATTN_COPY: "attn",
    MLP_BYPASS: "mlp",
    MLP_COMPOSITE: "mlp",
    MLP_LITERAL: "mlp",
}


def candidate_classes(node: Node) -> Tuple[str, ...]:
    """The candidate realization classes for a schedulable node.

    THE single declaration of the option set.  A single-candidate result is
    the legitimate degenerate case (most node types have one obviously-right
    hardware form); only the standalone ``Linear`` has a free choice today.
    """
    if isinstance(node, Attn):
        return (ATTN_HEADS,)
    if isinstance(node, Block):
        return (MLP_COMPOSITE,)
    if isinstance(node, LiteralValue):
        return (MLP_LITERAL,)
    if isinstance(node, Add):
        return (RESIDUAL_REUSE, ATTN_COPY)
    if isinstance(node, Linear):
        return (ATTN_TRANSPORT, MLP_BYPASS)
    raise TypeError(f"No realization classes for {type(node).__name__}")


def is_schedulable(node: Node) -> bool:
    """True for node types that occupy hardware (get a table entry)."""
    return isinstance(node, (Attn, Block, LiteralValue, Add, Linear))


def is_conditional(node: Node) -> bool:
    """True when the node's class is a predicate on schedule state, not a
    free choice — exactly ``Add`` (residual_reuse iff an addend is dead at
    schedule time)."""
    return isinstance(node, Add)


def has_flex_choice(node: Node) -> bool:
    """True when the node's class is a *free* choice a resolver must make
    (a CP-SAT decision variable in the directed path) — exactly the
    standalone ``Linear`` today."""
    return not is_conditional(node) and len(candidate_classes(node)) > 1


def static_flex_class(candidates: Tuple[str, ...], policy: SchedulingPolicy) -> str:
    """The static policy's pick for a free choice — the one rule both the
    optimize=0 resolver and the CP-SAT model's pinned (``flex_routing=
    False``) routing apply: attention transport unless
    ``local_in_attention == "never"``."""
    choice = ATTN_TRANSPORT if policy.local_in_attention != "never" else MLP_BYPASS
    assert choice in candidates
    return choice


# ---------------------------------------------------------------------------
# The realization table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One node's realization record.

    ``resolved`` is the chosen class (None while unresolved).
    ``conditional`` marks entries whose class is schedule-state-dependent
    (Add); they are complete without being resolved.
    """

    candidates: Tuple[str, ...]
    resolved: Optional[str] = None
    conditional: bool = False


class UnresolvedRealizationError(RuntimeError):
    """A realization entry was read (or reached the walk) unresolved."""


class RealizationTable:
    """node_id → :class:`Entry` for every schedulable node in a graph.

    Built unresolved by ``lower()``; resolved by exactly one of the two
    resolvers (:meth:`resolve_static` for the heuristic path,
    :meth:`resolve_from_assignment` for the CP-SAT directed path); read by
    the layer walk.  Resolvers return a new table — the unresolved table
    on the ``LoweredGraph`` is never mutated, so several resolutions of one
    lowering (e.g. the CP-SAT warm-start hint pass and the directed replay)
    cannot alias.
    """

    def __init__(self, entries: Dict[int, Entry]):
        self.entries = entries

    @classmethod
    def build(cls, nodes: Iterable[Node]) -> "RealizationTable":
        """Unresolved table over the schedulable nodes of a graph.

        Single-candidate entries are born resolved (the degenerate case);
        ``Add`` entries are born conditional; multi-candidate entries
        (standalone Linears) are born unresolved.
        """
        entries: Dict[int, Entry] = {}
        for node in nodes:
            if not is_schedulable(node):
                continue
            candidates = candidate_classes(node)
            if is_conditional(node):
                entries[node.node_id] = Entry(candidates, conditional=True)
            elif len(candidates) == 1:
                entries[node.node_id] = Entry(candidates, resolved=candidates[0])
            else:
                entries[node.node_id] = Entry(candidates)
        return cls(entries)

    # -- resolvers ---------------------------------------------------------

    def resolve_static(self, policy: SchedulingPolicy) -> "RealizationTable":
        """The optimize=0 resolver: the static policy picks every free choice.

        ``policy.local_in_attention == "never"`` routes standalone Linears
        to the MLP bypass; anything else to attention transport.
        """
        resolved: Dict[int, Entry] = {}
        for node_id, entry in self.entries.items():
            if entry.resolved is None and not entry.conditional:
                choice = static_flex_class(entry.candidates, policy)
                resolved[node_id] = replace(entry, resolved=choice)
            else:
                resolved[node_id] = entry
        return RealizationTable(resolved)

    def resolve_from_assignment(
        self, node_to_routing: Dict[int, str]
    ) -> "RealizationTable":
        """The directed-path resolver: the CP-SAT solve's per-node sublayer
        decisions (``ScheduleAssignment.node_to_routing``) pick every free
        choice.  Raises if the assignment contradicts a locked entry's
        sublayer — the tripwire for the two option sets drifting apart.
        """
        resolved: Dict[int, Entry] = {}
        for node_id, entry in self.entries.items():
            routing = node_to_routing.get(node_id)
            if entry.conditional:
                resolved[node_id] = entry
                continue
            if entry.resolved is not None:
                if routing is not None and CLASS_SUBLAYER[entry.resolved] != routing:
                    raise UnresolvedRealizationError(
                        f"solve routed node {node_id} to {routing!r} but its "
                        f"only realization class {entry.resolved!r} lives on "
                        f"{CLASS_SUBLAYER[entry.resolved]!r}"
                    )
                resolved[node_id] = entry
                continue
            if routing is None:
                # Not placed by the solve (e.g. a freeable input literal
                # precomputed at layer 0) — leave unresolved; the
                # completeness check only covers nodes the walk schedules.
                resolved[node_id] = entry
                continue
            matches = [c for c in entry.candidates if CLASS_SUBLAYER[c] == routing]
            if len(matches) != 1:
                raise UnresolvedRealizationError(
                    f"solve routing {routing!r} for node {node_id} does not "
                    f"select exactly one of its candidates {entry.candidates}"
                )
            resolved[node_id] = replace(entry, resolved=matches[0])
        return RealizationTable(resolved)

    # -- readers -----------------------------------------------------------

    def resolved_class(self, node: Node) -> str:
        entry = self.entries.get(node.node_id)
        if entry is None:
            raise UnresolvedRealizationError(
                f"node {node.node_id} ({type(node).__name__}) has no "
                f"realization entry"
            )
        if entry.resolved is None:
            raise UnresolvedRealizationError(
                f"realization for node {node.node_id} read before resolution "
                f"(candidates {entry.candidates})"
            )
        return entry.resolved

    def is_attention_routed(self, node: Node) -> bool:
        """The layer walk's read: does this node's class live on attention?"""
        return CLASS_SUBLAYER[self.resolved_class(node)] == "attn"

    def check_complete(self, nodes: Iterable[Node]) -> None:
        """Every schedulable node has an entry that is resolved or
        explicitly conditional.  Runs before the layer walk."""
        problems = []
        for node in nodes:
            if not is_schedulable(node):
                continue
            entry = self.entries.get(node.node_id)
            if entry is None:
                problems.append(f"{type(node).__name__} {node.node_id}: no entry")
            elif entry.resolved is None and not entry.conditional:
                problems.append(
                    f"{type(node).__name__} {node.node_id}: unresolved "
                    f"{entry.candidates}"
                )
        if problems:
            raise UnresolvedRealizationError(
                "realization table incomplete before the layer walk: "
                + "; ".join(problems)
            )
