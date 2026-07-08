"""Unit coverage for the CP-SAT warm-start capture map.

``_run_heuristic_warm_start`` runs the eager heuristic on a *clone* of the
real allocator, wrapped in ``_TrackingResidualStreamMap``, to capture
per-node layer / routing / cancel-layer hints for the CP-SAT solve.  The
coverage inventory flagged this capture path as untested; these unit tests
pin the two behaviors the hint quality depends on:

1. the tracking map carries the base map's dirty-state forward (so the
   probe doesn't pay phantom cancel heads on already-clean columns), and
2. the rollback-free correction (a rolled-back speculative free does not
   leave a stale ``cancel < birth`` record).
"""

from torchwright.compiler.forward.compile import _TrackingResidualStreamMap
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.graph.misc import InputNode


def test_tracking_map_carries_base_dirty_state():
    """The tracking wrapper must inherit the base map's cleaned columns.

    Regression for the warm-start dirty-state bug: ``super().__init__`` seeds
    ``_dirty`` with *every* column, and the wrapper copies ``_free`` /
    ``_node_to_indices`` / ``_reserved`` from the base map — but before the
    fix it did **not** copy ``_dirty``.  So the probe saw every free column as
    dirty, charged a phantom cancel head per column, and emitted a schedule
    deeper than the eager fallback it exists to improve on.  After the fix the
    wrapper's ``dirty_subset`` agrees with the base map's on every column.
    """
    base = ResidualStreamMap(64)
    # The real allocator marks the pos-encoding / input columns clean up front
    # (get_input_res_stream) and cancelled columns clean inline.  Model that by
    # cleaning an arbitrary subset here.
    cleaned = [0, 1, 2, 3, 10, 11, 40]
    base.mark_clean(cleaned)

    tmap = _TrackingResidualStreamMap(base)

    all_cols = list(range(64))
    # The wrapper's view of what is still dirty must match the base map exactly.
    assert tmap.dirty_subset(all_cols) == base.dirty_subset(all_cols)
    # And concretely: the cleaned columns are dirty in neither map.
    assert tmap.dirty_subset(cleaned) == []
    # A column the base never cleaned is still dirty in the wrapper.
    assert 63 in tmap.dirty_subset(all_cols)


def test_tracking_map_dirty_state_is_an_independent_copy():
    """Cleaning columns on the wrapper must not mutate the base map's set.

    The probe mutates its clone freely; the caller's allocator state must be
    untouched (``_run_heuristic_warm_start`` relies on this isolation).
    """
    base = ResidualStreamMap(32)
    base.mark_clean([0, 1])

    tmap = _TrackingResidualStreamMap(base)
    tmap.mark_clean([5, 6, 7])

    # The wrapper cleaned three more columns; the base map did not follow.
    assert tmap.dirty_subset(list(range(32))) == base.dirty_subset(
        [c for c in range(32) if c not in (5, 6, 7)]
    )
    assert base.dirty_subset([5, 6, 7]) == [5, 6, 7]


def test_tracking_map_rollback_free_correction():
    """A rolled-back speculative free must not leave a ``cancel < birth`` hint.

    Pins the tracking map's documented rollback-free correction: the
    heuristic speculatively allocates a node, fails its budget, and rolls the
    allocation back with ``free`` at an early layer — but the node is really
    born (and dies) later.  (Re)allocation clears the stale cancel so the
    emitted ``hint_cancel`` stays internally consistent.
    """
    base = ResidualStreamMap(64)
    tmap = _TrackingResidualStreamMap(base)
    x = InputNode("x", 1, value_range=(-100.0, 100.0))

    tmap.current_layer = 6
    tmap.allocate(x)
    tmap.free(x)  # speculative rollback -> records cancel=6
    assert tmap.cancel_layer[x.node_id] == 6

    tmap.current_layer = 7
    tmap.allocate(x)  # genuine birth -> stale cancel cleared
    assert x.node_id not in tmap.cancel_layer

    tmap.current_layer = 9
    tmap.free(x)  # genuine death -> real cancel recorded
    assert tmap.cancel_layer[x.node_id] == 9
