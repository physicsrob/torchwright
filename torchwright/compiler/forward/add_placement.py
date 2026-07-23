"""Assignment-level Add placement derivation.

ONE definition of addend deadness (docs/plan_additional_mlp_routing.md,
*Assignment derivation*), computed from a complete layer/route assignment
rather than from live scheduler state.  Three checks call it:

- CP-SAT solution extraction verifies the solver's per-occurrence
  ``reusable_i`` / ``reuse_i`` / ``is_free`` literals against it before
  discarding them.
- Heuristic-trace completion verifies the walk's observed physical
  placement against it before canonicalizing target metadata.
- Directed replay derives the expected placement from it before comparing
  with the residual map.

The sublayer-order predicate: an occurrence ``E`` of Add ``A`` is reusable
when every *other* effective consumer ``C`` of ``E`` satisfies

    layer[C] < layer[A]
    or (layer[C] == layer[A] and C is attention-routed and A is MLP-routed)

— a same-layer attention consumer counts complete for an MLP-routed Add
because the attention phase is constructed before the MLP phase-start
snapshot; every other same-layer consumer does not.  A consumer without a
layer entry (e.g. a terminal ``Concatenate`` retaining an output leaf)
cannot be ordered, so it blocks reuse.  The predicate needs no term for
``E``'s own birth layer: the route-aware dependency bounds on the edge
``E -> A`` (zero-layer gap only for attention producer -> MLP consumer)
already guarantee ``E`` is materialized by ``A``'s placement snapshot in
any dependency-consistent assignment, which is what this module is handed.

Target selection is deterministic, never a solver choice: occurrence 0
wins when both are reusable (``reuse_0 = reusable_0``,
``reuse_1 = not reusable_0 and reusable_1``).  For ``add(x, x)`` both
occurrences name one node and share the consumer set, so occurrence 0 is
the reuse target and occurrence 1 the source read.

Tied compiles: the held target short-circuits to fresh placement
(``(False, False, None)``, matching its pinned ``is_free == 0``), and the
held source is never reusable — its columns end through the held-bank
cancel/hold transition, not through ``reassign``.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from torchwright.graph import Concatenate, Node

#: Route strings as they appear in ``ScheduleAssignment.node_to_routing``.
ATTN = "attn"
MLP = "mlp"


@dataclass(frozen=True)
class AddPlacement:
    """One Add's derived residual placement.

    ``reusable_0`` / ``reusable_1`` are the per-occurrence deadness
    predicates; ``reuse_input_index`` is the deterministically selected
    target occurrence (0 or 1), or ``None`` for fresh placement.
    """

    reusable_0: bool
    reusable_1: bool
    reuse_input_index: int | None

    @property
    def is_free(self) -> bool:
        return self.reuse_input_index is not None


def derive_add_placement(
    add: Node,
    *,
    effective_consumers: Callable[[Node], Iterable[Node]],
    node_to_layer: Mapping[int, int],
    node_to_routing: Mapping[int, str],
    held_source_id: int | None = None,
    held_target_id: int | None = None,
) -> AddPlacement:
    """Derive one Add's ``(reusable_0, reusable_1, reuse_input_index)``.

    ``effective_consumers`` must walk ``Concatenate`` consumers
    transparently and keep terminal Concatenates (output retention) — the
    same convention as the scheduler's ``_get_effective_consumers`` and
    compile's ``_effective_consumers``.  ``node_to_layer`` /
    ``node_to_routing`` must cover the Add and every orderable consumer;
    the Add itself missing from either map is a caller bug and raises.
    """
    if add.node_id == held_target_id:
        # The held target claims the whole held bank via allocate_at with
        # fresh placement, unconditionally (its is_free is pinned 0 without
        # the deadness biconditional).
        return AddPlacement(False, False, None)
    layer_a = node_to_layer.get(add.node_id)
    route_a = node_to_routing.get(add.node_id)
    if layer_a is None or route_a is None:
        raise ValueError(
            f"derive_add_placement needs the Add's own layer and route; "
            f"node {add.node_id} has layer={layer_a!r}, route={route_a!r}"
        )

    def occurrence_reusable(occ: Node) -> bool:
        if occ.node_id == held_source_id:
            # The tied bank passes through a physical cancel into the held
            # state; a reuse may never inherit it through reassign().
            return False
        if isinstance(occ, Concatenate):
            # Not residual-allocated; its columns cannot be reassigned.
            return False
        for consumer in effective_consumers(occ):
            if consumer.node_id == add.node_id:
                continue
            layer_c = node_to_layer.get(consumer.node_id)
            if layer_c is None:
                return False  # unordered read (e.g. terminal Concatenate)
            if layer_c < layer_a:
                continue
            if (
                layer_c == layer_a
                and node_to_routing.get(consumer.node_id) == ATTN
                and route_a == MLP
            ):
                continue
            return False
        return True

    a0, a1 = add.inputs
    reusable_0 = occurrence_reusable(a0)
    reusable_1 = occurrence_reusable(a1)
    if reusable_0:
        index: int | None = 0
    elif reusable_1:
        index = 1
    else:
        index = None
    return AddPlacement(reusable_0, reusable_1, index)
