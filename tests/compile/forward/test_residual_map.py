import pytest
import torch

from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.residual_assignment import ResidualStreamState
from torchwright.graph import Concatenate
from torchwright.graph.misc import InputNode, LiteralValue


def test_allocate_and_free():
    """Allocate a node, verify indices, free it, verify columns recovered."""
    rmap = ResidualStreamMap(64)
    node = InputNode("x", 8, value_range=(-100.0, 100.0))

    indices = rmap.allocate(node)
    assert len(indices) == 8
    assert len(set(indices)) == 8  # all unique
    assert all(0 <= i < 64 for i in indices)
    assert rmap.get_free_count() == 56
    assert rmap.is_allocated(node)
    assert rmap.get_indices(node) == indices

    rmap.free(node)
    assert rmap.get_free_count() == 64
    assert not rmap.is_allocated(node)


def test_multiple_allocations():
    """Three nodes get non-overlapping index sets."""
    rmap = ResidualStreamMap(64)
    a = InputNode("a", 8, value_range=(-100.0, 100.0))
    b = InputNode("b", 16, value_range=(-100.0, 100.0))
    c = InputNode("c", 4, value_range=(-100.0, 100.0))

    idx_a = rmap.allocate(a)
    idx_b = rmap.allocate(b)
    idx_c = rmap.allocate(c)

    # No overlap
    all_indices = set(idx_a) | set(idx_b) | set(idx_c)
    assert len(all_indices) == 8 + 16 + 4
    assert rmap.get_free_count() == 64 - 28
    assert rmap.get_allocated_nodes() == {a, b, c}


def test_full_stream():
    """Fill the stream exactly, then verify next allocation raises."""
    rmap = ResidualStreamMap(16)
    a = InputNode("a", 8, value_range=(-100.0, 100.0))
    b = InputNode("b", 8, value_range=(-100.0, 100.0))
    c = InputNode("c", 1, value_range=(-100.0, 100.0))

    rmap.allocate(a)
    rmap.allocate(b)
    assert rmap.get_free_count() == 0

    with pytest.raises(ValueError):
        rmap.allocate(c)


def test_reassign():
    """Reassign columns from one node to another."""
    rmap = ResidualStreamMap(64)
    old = InputNode("old", 8, value_range=(-100.0, 100.0))
    new = InputNode("new", 8, value_range=(-100.0, 100.0))

    indices = rmap.allocate(old)
    rmap.reassign(old, new)

    assert not rmap.is_allocated(old)
    assert rmap.is_allocated(new)
    assert rmap.get_indices(new) == indices
    assert rmap.get_free_count() == 56  # unchanged


def test_hold_bank_is_not_free_and_only_full_ordered_claim_succeeds():
    rmap = ResidualStreamMap(16)
    source = InputNode("source", 4, value_range=(-10.0, 10.0))
    target = InputNode("target", 4, value_range=(-10.0, 10.0))
    bank = rmap.allocate(source)
    free_before = rmap.get_free_count()

    assert rmap.hold(source) == bank
    assert not rmap.is_allocated(source)
    assert rmap.get_free_count() == free_before

    other = InputNode("other", free_before, value_range=(-10.0, 10.0))
    assert set(rmap.allocate(other)).isdisjoint(bank)
    with pytest.raises(AssertionError, match="complete held bank"):
        rmap.allocate_at(target, [*bank[:-1], rmap.get_indices(other)[0]])

    assert rmap.allocate_at(target, bank) == bank
    assert rmap.get_indices(target) == bank
    with pytest.raises(AssertionError, match="already allocated"):
        rmap.allocate_at(target, bank)


def test_hold_rejects_a_second_bank():
    rmap = ResidualStreamMap(16)
    a = InputNode("a", 3, value_range=(-10.0, 10.0))
    b = InputNode("b", 3, value_range=(-10.0, 10.0))
    rmap.allocate(a)
    rmap.allocate(b)
    rmap.hold(a)
    with pytest.raises(AssertionError, match="another held bank"):
        rmap.hold(b)


def test_build_residual_assignment():
    """Build a ResidualAssignment and verify get_node_indices works."""
    rmap = ResidualStreamMap(64)
    inp = InputNode("x", 8, value_range=(-100.0, 100.0))
    const = LiteralValue(torch.ones(4))
    out = InputNode("out", 3, value_range=(-100.0, 100.0))

    idx_inp = rmap.allocate(inp)
    idx_const = rmap.allocate(const)
    idx_out = rmap.allocate(out)

    in_state = ResidualStreamState(name="in")
    out_state = ResidualStreamState(name="out")

    ra = rmap.build_residual_assignment(
        in_state=in_state,
        out_state=out_state,
        input_nodes=[inp, const],
        output_node=out,
    )

    assert ra.get_node_indices(in_state, inp) == idx_inp
    assert ra.get_node_indices(in_state, const) == idx_const
    assert ra.get_node_indices(out_state, out) == idx_out


def test_no_fragmentation():
    """Non-contiguous free space is usable because contiguity is not required."""
    rmap = ResidualStreamMap(32)
    a = InputNode("a", 8, value_range=(-100.0, 100.0))
    b = InputNode("b", 8, value_range=(-100.0, 100.0))
    c = InputNode("c", 8, value_range=(-100.0, 100.0))

    rmap.allocate(a)
    rmap.allocate(b)
    rmap.allocate(c)
    assert rmap.get_free_count() == 8

    # Free the middle node — creates a gap
    rmap.free(b)
    assert rmap.get_free_count() == 16

    # Allocate a node larger than the gap — succeeds because columns
    # don't need to be contiguous
    d = InputNode("d", 16, value_range=(-100.0, 100.0))
    indices = rmap.allocate(d)
    assert len(indices) == 16
    assert len(set(indices)) == 16

    # No overlap with a or c
    assert not (set(indices) & set(rmap.get_indices(a)))
    assert not (set(indices) & set(rmap.get_indices(c)))


def test_resolve_indices_concatenate():
    """resolve_indices resolves Concatenate to its children's indices in order."""
    rmap = ResidualStreamMap(64)
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 3, value_range=(-100.0, 100.0))

    idx_a = rmap.allocate(a)
    idx_b = rmap.allocate(b)

    cat = Concatenate([a, b])
    # Concatenate is NOT allocated — resolve_indices resolves through it
    result = rmap.resolve_indices(cat)
    assert result == idx_a + idx_b


def test_tracking_map_clears_rollback_cancel_on_reallocate():
    """A node freed by a *rolled-back* speculative allocation, then re-allocated
    (reborn) at a later layer, must not keep the rollback layer as its cancel.

    Regression for the CP-SAT warm-start hint producing ``cancel < birth``:
    ``LayerScheduler`` speculatively allocates a node, fails its dirty-cancel/
    head budget, and rolls the allocation back with ``free(node)`` at layer 6 —
    but the node is really born at layer 7 and dies by ``reassign`` (free-add,
    never ``free``).  The recorded ``cancel = 6 < birth = 7`` was hard-infeasible
    in the CP-SAT model.  The tracking map clears the stale cancel on
    (re)allocate / reassign so the emitted hint is internally consistent.
    """
    from torchwright.compiler.forward.compile import _TrackingResidualStreamMap

    base = ResidualStreamMap(64)
    tmap = _TrackingResidualStreamMap(base)
    x = InputNode("x", 1, value_range=(-100.0, 100.0))

    # Layer 6: speculative allocate then rollback free -> records cancel=6.
    tmap.current_layer = 6
    tmap.allocate(x)
    tmap.free(x)
    assert tmap.cancel_layer[x.node_id] == 6

    # Layer 7: the node is really born here -> the stale cancel must be cleared.
    tmap.current_layer = 7
    tmap.allocate(x)
    assert x.node_id not in tmap.cancel_layer

    # A genuine later free re-records the real cancel.
    tmap.current_layer = 9
    tmap.free(x)
    assert tmap.cancel_layer[x.node_id] == 9


def test_node_deepcopy_preserves_identity():
    """`copy.deepcopy` of a container that *references* nodes must keep node
    identity (nodes are graph singletons keyed on node_id), while still
    deep-copying the surrounding container.

    Regression for the warm-start: `_run_heuristic_warm_start` does
    `copy.deepcopy(residual_map)` to isolate its scheduler mutations.  Without
    `Node.__deepcopy__` returning self, that cloned the `pos` node, breaking the
    `n is self.pos_encoding` identity check so the warm-start freed `pos` early
    and produced a hint the CP-SAT model (which reserves pos) rejected by exactly
    len(pos) columns.
    """
    import copy

    n = InputNode("x", 4, value_range=(-100.0, 100.0))
    container = {"node": n, "list": [n]}
    dup = copy.deepcopy(container)

    assert dup["node"] is n  # node identity preserved (not cloned)
    assert dup["list"][0] is n
    assert dup is not container  # the container itself is still deep-copied
    assert dup["list"] is not container["list"]


def test_tracking_map_clears_stale_cancel_on_reassign():
    """A node reborn via ``reassign`` (the free-add path) must not keep a stale
    cancel from an earlier rolled-back allocation.
    """
    from torchwright.compiler.forward.compile import _TrackingResidualStreamMap

    base = ResidualStreamMap(64)
    tmap = _TrackingResidualStreamMap(base)
    y = InputNode("y", 1, value_range=(-100.0, 100.0))
    w = InputNode("w", 1, value_range=(-100.0, 100.0))

    # Earlier rolled-back allocation of y records a stale cancel.
    tmap.current_layer = 6
    tmap.allocate(y)
    tmap.free(y)
    assert tmap.cancel_layer[y.node_id] == 6

    # Later, y is reborn by reassigning w's columns to it (free-add reuse).
    tmap.current_layer = 8
    tmap.allocate(w)
    tmap.reassign(w, y)
    assert y.node_id not in tmap.cancel_layer
    assert tmap.is_allocated(y)
