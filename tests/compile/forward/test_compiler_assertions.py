"""Negative tests for the compiler's invariant assertions.

Each test deliberately constructs bad state and asserts the matching
invariant fires with a useful message. Constructing the bad state
directly (rather than hoping the compiler gets there) is intentional —
the assertion is supposed to make the bad state unreachable through
normal paths, so the only way to exercise it is to force it.
"""

import pytest
import torch

import torchwright.compiler.device as device_mod
from torchwright.compiler.forward.compile import (
    _verify_end_of_layer_liveness,
    forward_compile,
)
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.replay_plan import (
    PlannedAttentionOp,
    PlannedMlpOp,
)
from torchwright.compiler.forward.weight_writer import (
    _write_compute_attn,
    _write_compute_literal_value,
)
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.graph import Attn, Linear
from torchwright.graph.misc import InputNode, LiteralValue

D = 64
D_HEAD = 16


# ---------------------------------------------------------------------------
# D — allocator self-consistency
# ---------------------------------------------------------------------------


def test_D_allocate_rejects_overlap_with_live_node():
    """If _free somehow contained a column already owned by a live node,
    allocate's pre-commit check fires with the overlap detail."""
    rmap = ResidualStreamMap(32)
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    rmap.allocate(a)

    # Poison: re-add a's columns to _free while leaving a allocated.
    a_cols = rmap.get_indices(a)
    rmap._free |= set(a_cols)

    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    with pytest.raises(
        AssertionError, match=r"free set proposed columns already owned"
    ):
        rmap.allocate(b)


def test_D_check_invariants_catches_missing_column():
    """After mutation, if a column is neither in _free nor in any node's
    indices, _check_invariants names the missing column(s)."""
    rmap = ResidualStreamMap(16)
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    rmap.allocate(a)

    # Poison: remove one column from _free without adding it anywhere.
    orphan = next(iter(rmap._free))
    rmap._free.discard(orphan)

    with pytest.raises(AssertionError, match=r"neither free, allocated"):
        rmap._check_invariants("poisoned")


def test_D_check_invariants_catches_pairwise_overlap():
    """If two nodes' index lists share a column, disjointness check fires."""
    rmap = ResidualStreamMap(16)
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    rmap.allocate(a)
    rmap.allocate(b)

    # Poison: overwrite b's indices to overlap with a's.
    rmap._node_to_indices[b] = list(rmap._node_to_indices[a])

    with pytest.raises(AssertionError, match=r"assigned to both"):
        rmap._check_invariants("poisoned")


# ---------------------------------------------------------------------------
# C — literal stability
# ---------------------------------------------------------------------------


def test_C_literal_write_rejects_truncation():
    """_write_compute_literal_value refuses when target_cols is shorter
    than the literal's value tensor."""
    layer = TransformerLayer(D, D_HEAD)
    lit = LiteralValue(torch.tensor([1.0, 2.0, 3.0, 4.0]), name="lit")
    # Only 2 target cols, but value has 4 entries — would silently drop.
    bogus = PlannedMlpOp(
        op_type="compute_literal_value", node=lit, target_cols=[0, 1], mlp_slots=[]
    )
    with pytest.raises(AssertionError, match=r"Literal truncation would drop values"):
        _write_compute_literal_value(layer.mlp, bogus)


def test_C_literal_write_rejects_extra_target_cols():
    """Target cols wider than the literal would write uninitialized tail."""
    layer = TransformerLayer(D, D_HEAD)
    lit = LiteralValue(torch.tensor([1.0, 2.0]), name="lit")
    bogus = PlannedMlpOp(
        op_type="compute_literal_value",
        node=lit,
        target_cols=[0, 1, 2, 3],
        mlp_slots=[],
    )
    with pytest.raises(AssertionError, match=r"Literal truncation"):
        _write_compute_literal_value(layer.mlp, bogus)


# ---------------------------------------------------------------------------
# B — Q/K/V row-width correctness
# ---------------------------------------------------------------------------


def test_B_attn_rejects_v_source_cols_wrong_length():
    """_write_compute_attn raises when V source_cols length doesn't match
    value_in width."""
    # A plain residual node feeding both Q and K (it stands in for whatever
    # the head reads; position is now a rotation inside attention, not an input).
    qk = InputNode("qk", 17, value_range=(-100.0, 100.0))
    v_in = InputNode("v", 4, value_range=(-100.0, 100.0))
    node = Attn(
        query_in=qk,
        key_in=qk,
        value_in=v_in,
        query_matrix=torch.eye(len(qk), D_HEAD),
        key_matrix=torch.eye(len(qk), D_HEAD),
        value_matrix=torch.eye(len(v_in), D_HEAD),
        output_matrix=torch.eye(D_HEAD, len(v_in)),
    )
    rmap = ResidualStreamMap(D)
    rmap.allocate(qk)
    rmap.allocate(v_in)
    out_cols = rmap.allocate(node)

    layer = TransformerLayer(D, D_HEAD)
    bogus = PlannedAttentionOp(
        op_type="compute_attn",
        node=node,
        target_cols=out_cols,
        q_source_cols=rmap.resolve_indices(qk),
        k_source_cols=rmap.resolve_indices(qk),
        source_cols=rmap.resolve_indices(v_in)[:2],  # too short — truncated
    )
    with pytest.raises(AssertionError, match=r"V row-width mismatch"):
        _write_compute_attn(layer.attn.attn, bogus, rmap)


def test_B_attn_rejects_q_source_cols_wrong_length():
    qk = InputNode("qk", 17, value_range=(-100.0, 100.0))
    v_in = InputNode("v", 4, value_range=(-100.0, 100.0))
    node = Attn(
        query_in=qk,
        key_in=qk,
        value_in=v_in,
        query_matrix=torch.eye(len(qk), D_HEAD),
        key_matrix=torch.eye(len(qk), D_HEAD),
        value_matrix=torch.eye(len(v_in), D_HEAD),
        output_matrix=torch.eye(D_HEAD, len(v_in)),
    )
    rmap = ResidualStreamMap(D)
    rmap.allocate(qk)
    rmap.allocate(v_in)
    out_cols = rmap.allocate(node)

    layer = TransformerLayer(D, D_HEAD)
    bogus = PlannedAttentionOp(
        op_type="compute_attn",
        node=node,
        target_cols=out_cols,
        q_source_cols=rmap.resolve_indices(qk)[:4],  # too short
        k_source_cols=rmap.resolve_indices(qk),
        source_cols=rmap.resolve_indices(v_in),
    )
    with pytest.raises(AssertionError, match=r"Q row-width mismatch"):
        _write_compute_attn(layer.attn.attn, bogus, rmap)


# ---------------------------------------------------------------------------
# Full-width rotary — d_qk == d_head (an I3-sibling; NOT one of the canonical
# I1–I4).  Every head is full-width rotary on the global grid; export.py always
# rotates the full d_head with no width guard of its own, so the in-process
# assert in _scatter_compute_attn is the only barrier against a partial-width
# head (silently NoPE on the un-filled planes).  Guard BOTH directions: the
# pre-existing test_forward_compile coverage only exercised d_qk > d_head, but
# the dangerous regression is relaxing the `==` to `<=`, which would let
# d_qk < d_head through and zero-pad/rotate dead planes with no divergence
# signal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d_qk", [8, 32])  # below and above d_head=16
def test_attn_rejects_d_qk_not_equal_d_head(d_qk):
    """_scatter_compute_attn requires d_qk == d_head (full-width rotary)."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    out = Attn(
        query_in=x,
        key_in=x,
        value_in=x,
        query_matrix=torch.randn(len(x), d_qk),
        key_matrix=torch.randn(len(x), d_qk),
        value_matrix=torch.randn(len(x), 4),
        output_matrix=torch.randn(4, 4),
    )
    with pytest.raises(AssertionError, match=r"d_qk == d_head"):
        forward_compile(d=256, d_head=16, output_node=out, verbose=False)


# ---------------------------------------------------------------------------
# A — end-of-layer liveness (gated)
# ---------------------------------------------------------------------------


def test_A_end_of_layer_catches_freed_too_early(monkeypatch):
    """If a node is freed while still having uncomputed consumers, the
    gated end-of-layer check fires with the consumer names."""
    # Two-node linear chain: x -> l (Linear). Compile expects l to read
    # x's columns, so x must stay allocated while l is uncomputed.
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    weight = torch.eye(4, 4)
    bias = torch.zeros(4)
    l = Linear(x, weight, bias)

    graph = GraphAnalyzer(l)
    rmap = ResidualStreamMap(D)
    rmap.allocate(x)

    # x is allocated, l is uncomputed — invariant holds here.
    computed = {x}
    _verify_end_of_layer_liveness(graph, rmap, computed, layer_idx=0)

    # Poison: free x while l still needs it. The check must fire.
    rmap.free(x)
    with pytest.raises(AssertionError, match=r"liveness violation"):
        _verify_end_of_layer_liveness(graph, rmap, computed, layer_idx=1)


# ---------------------------------------------------------------------------
# A — schedule-time liveness
# ---------------------------------------------------------------------------


def test_A_require_live_raises_for_unallocated_input():
    """LayerScheduler._require_live surfaces unallocated source nodes with
    op context before the downstream KeyError."""
    from torchwright.compiler.forward.scheduler import LayerScheduler

    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    weight = torch.eye(4, 4)
    bias = torch.zeros(4)
    l = Linear(x, weight, bias)

    graph = GraphAnalyzer(l)
    scheduler = LayerScheduler(graph, D, D_HEAD)
    rmap = ResidualStreamMap(D)
    # Deliberately do NOT allocate x — this should be caught by _require_live.

    with pytest.raises(AssertionError, match=r"Live-column invariant violated"):
        scheduler._require_live(x, rmap, f"compute_linear input for {l!r}")


# ---------------------------------------------------------------------------
# Overlay target-column reservation
# ---------------------------------------------------------------------------


def test_reserve_rejects_allocated_col():
    """reserve() must fail if any requested column is already allocated —
    the caller would otherwise silently lose that column from the free pool
    without any effect (it was already not free)."""
    rmap = ResidualStreamMap(16)
    a = InputNode("a", 4, value_range=(-1.0, 1.0))
    rmap.allocate(a)  # takes [0,1,2,3]

    with pytest.raises(AssertionError, match=r"already allocated"):
        rmap.reserve([0, 1])


def test_reserve_blocks_subsequent_allocation():
    """Reserved columns cannot be handed out by allocate() — the
    allocator-level guarantee that a reserved column is never reused."""
    rmap = ResidualStreamMap(8)
    rmap.reserve([0, 1, 2, 3])

    a = InputNode("a", 4, value_range=(-1.0, 1.0))
    rmap.allocate(a)
    # a must get the unreserved columns [4,5,6,7], not the reserved ones.
    assert rmap.get_indices(a) == [4, 5, 6, 7]


# ---------------------------------------------------------------------------
# bias=False writer contracts (docs/no_bias_plan.md)
# ---------------------------------------------------------------------------


def _no_bias_setup():
    """Residual map with the pinned const-1 column allocated."""
    rmap = ResidualStreamMap(D)
    const_one = LiteralValue(torch.ones(1), name="const_one")
    rmap.allocate(const_one)
    return rmap, const_one


def test_no_bias_slot0_claim_fires():
    """An PlannedMlpOp claiming hidden slot 0 under bias=False is a scheduler bug:
    slot 0 is the constant lane, packing must start at slot 1."""
    from torchwright.compiler.forward.weight_writer import write_mlp_sublayer
    from torchwright.graph import FFN

    rmap, const_one = _no_bias_setup()
    x = InputNode("x", 4, value_range=(-1.0, 1.0))
    rmap.allocate(x)
    ffn = FFN(
        x,
        gate_proj=torch.randn(4, 4),
        gate_bias=torch.zeros(4),
        out_proj=torch.randn(4, 4),
        out_bias=torch.zeros(4),
        activation="relu",
    )
    cols = rmap.allocate(ffn)
    op = PlannedMlpOp(
        "compute_ffn",
        ffn,
        cols,
        mlp_slots=[0, 1, 2, 3],
        source_cols=rmap.get_indices(x),
    )
    layer = TransformerLayer(D, D_HEAD)
    with pytest.raises(AssertionError, match="constant lane"):
        write_mlp_sublayer(layer, [op], rmap.get_indices(const_one)[0], bias=False)


def test_no_bias_const_column_aliasing_fires():
    """An FFN whose captured input columns include the pinned constant-1
    column would collide with the bias-fold row — the writer must refuse."""
    from torchwright.compiler.forward.weight_writer import write_mlp_sublayer
    from torchwright.graph import FFN

    rmap, const_one = _no_bias_setup()
    x = InputNode("x", 4, value_range=(-1.0, 1.0))
    rmap.allocate(x)
    ffn = FFN(
        x,
        gate_proj=torch.randn(4, 4),
        gate_bias=torch.zeros(4),
        out_proj=torch.randn(4, 4),
        out_bias=torch.zeros(4),
        activation="relu",
    )
    cols = rmap.allocate(ffn)
    # Forge source columns that alias the const-1 column (never producible
    # by the allocator; the assert is the backstop).
    bad_source = list(rmap.get_indices(x))
    bad_source[0] = rmap.get_indices(const_one)[0]
    op = PlannedMlpOp(
        "compute_ffn", ffn, cols, mlp_slots=[1, 2, 3, 4], source_cols=bad_source
    )
    layer = TransformerLayer(D, D_HEAD)
    with pytest.raises(AssertionError, match="constant-1 column"):
        write_mlp_sublayer(layer, [op], rmap.get_indices(const_one)[0], bias=False)
