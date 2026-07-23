"""TDD tests for the forward compiler's LayerScheduler.

Tests verify observable behavior: ops returned, state changes, errors raised.
No tests for internal ordering heuristics or priority strategies — efficiency
is tested at the integration level (Phase 4/5).

Conventions:
- D=64, D_HEAD=16 → n_heads=4 attention heads per layer.
- A 17-wide reserved block stands in for the always-allocated columns the
  scheduler keeps pinned (under RoPE there is no PosEncoding; the production
  reserved region is the 1-wide const-1 self-match column, but these tests use
  a 17-wide block so the column-pressure arithmetic below is unchanged). It is
  passed as the scheduler's never-free sentinel, exactly as pos_encoding was.
- "dead-for-add" means all consumers except the pending Add are computed.
"""

import pytest
import torch

from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.scheduler import LayerScheduler
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)
from torchwright.graph import Add, Attn, Concatenate, Linear
from torchwright.graph.misc import InputNode, LiteralValue
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

D = 64
D_HEAD = 16
N_HEADS = D // D_HEAD  # 4

# The shipping default keeps Adds on attention (SchedulingPolicy.
# add_in_attention); the MLP-Add tests opt in to the MLP route explicitly.
MLP_ADD_POLICY = SchedulingPolicy(add_in_attention="never")


def _make_reserved_block():
    # A generic always-allocated stand-in for the scheduler's pinned reserved
    # columns (17 wide so the column-pressure tests' arithmetic is unchanged).
    return InputNode("reserved", 17, value_range=(-1.0, 1.0))


def _make_attn(v):
    # A minimal self-reading Attn node — ready as soon as ``v`` is computed —
    # standing in for the old ``pos.attend_to_offset(v)`` (a rotary offset head
    # has a different input shape; the scheduler only cares that this is an Attn
    # node that produces a ``compute_attn`` op).
    d = len(v)
    return Attn(
        query_in=v,
        key_in=v,
        value_in=v,
        query_matrix=torch.eye(d),
        key_matrix=torch.eye(d),
        value_matrix=torch.eye(d),
        output_matrix=torch.eye(d),
    )


def _make_linear(inp, d_out, name=""):
    """Zero-bias Linear."""
    return Linear(inp, torch.randn(len(inp), d_out), torch.zeros(d_out), name=name)


def _make_biased_linear(inp, d_out, name=""):
    """Linear with non-zero bias."""
    return Linear(inp, torch.randn(len(inp), d_out), torch.randn(d_out), name=name)


def _make_ffn(inp, d_hidden, d_out, name=""):
    """A degenerate-ReLU FFN (the former L1 -> ReLU -> L2 chain, now one
    node).  Returns the FFN.
    """
    return linear_relu_linear(
        inp,
        torch.randn(d_hidden, len(inp)),
        torch.randn(d_hidden),
        torch.randn(d_hidden, d_out),
        torch.randn(d_out),
        name=name,
    )


# ---------------------------------------------------------------------------
# 1. Basic op routing — correct node type produces correct op type
# ---------------------------------------------------------------------------


def test_schedule_attn_node():
    """Attn node produces a compute_attn scheduler operation."""
    pos = _make_reserved_block()
    v = InputNode("v", 4, value_range=(-100.0, 100.0))
    attn_node = _make_attn(v)

    graph = GraphAnalyzer(attn_node)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(v)
    computed = {pos, v}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    compute_attn = [op for op in attn_ops if op.op_type == "compute_attn"]
    assert len(compute_attn) == 1
    assert compute_attn[0].node is attn_node
    assert attn_node in computed


def test_schedule_block():
    """An FFN produces compute_ffn; the FFN is marked computed."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    ffn = _make_ffn(x, 8, 3, "chain")

    graph = GraphAnalyzer(ffn)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    ffn_ops = [op for op in mlp_ops if op.op_type == "compute_ffn"]
    assert len(ffn_ops) == 1
    assert ffn_ops[0].node is ffn
    assert len(ffn_ops[0].mlp_slots) == 8  # hidden width (n_lanes)

    assert ffn in computed


def test_schedule_constant():
    """LiteralValue produces compute_literal_value with no MLP slots.

    Note: in the compile loop, Constants are typically pre-populated as input nodes.
    This tests the scheduler's capability to handle Constants that aren't pre-populated.
    """
    pos = _make_reserved_block()
    const = LiteralValue(torch.tensor([1.0, -2.0, 3.5]))

    graph = GraphAnalyzer(const)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    computed = {pos}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    const_ops = [op for op in mlp_ops if op.op_type == "compute_literal_value"]
    assert len(const_ops) == 1
    assert const_ops[0].node is const
    assert const_ops[0].mlp_slots == []
    assert const in computed


def test_schedule_zero_bias_linear():
    """Zero-bias Linear: legacy policy uses attention, default uses MLP bypass."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    linear = _make_linear(x, 3, "lin")

    graph = GraphAnalyzer(linear)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=LEGACY_POLICY)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    linear_ops = [op for op in attn_ops if op.op_type == "compute_linear"]
    assert len(linear_ops) == 1
    assert linear_ops[0].node is linear
    assert linear in computed


def test_schedule_zero_bias_linear_bypass():
    """Zero-bias Linear with default policy produces compute_linear_bypass."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    linear = _make_linear(x, 3, "lin")

    graph = GraphAnalyzer(linear)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    bypass_ops = [op for op in mlp_ops if op.op_type == "compute_linear_bypass"]
    assert len(bypass_ops) == 1
    assert bypass_ops[0].node is linear
    assert len(bypass_ops[0].mlp_slots) == 2 * linear.d_output
    attn_linears = [op for op in attn_ops if op.op_type == "compute_linear"]
    assert len(attn_linears) == 0
    assert linear in computed


def test_saturated_head_death_layer_frees_via_mlp_slots():
    """When the attention head budget is full at a dead node's death layer, the
    eager heuristic frees it via an MLP ``cancel_bypass`` the same layer instead
    of deferring the cancel to a later attention batch.

    Geometry: d_head == d gives one attention head per layer.  A single Attn op
    consumes it, so the dead node ``D`` cannot join an attention cancel batch;
    the MLP sublayer zeroes it via ``cancel_bypass`` (2 hidden slots per
    column) and frees its columns.
    """
    d = 64
    d_head = 64  # one attention head per layer
    pos = InputNode("reserved", 1, value_range=(-1.0, 1.0))
    x = InputNode("x", 4, value_range=(-1.0, 1.0))
    attn = _make_attn(x)  # d_v == 4, one head — saturates the layer's budget
    dead = _make_linear(x, 8, "D")
    dead_consumer = _make_linear(dead, 4, "Dc")  # already computed → D is dead
    out = _make_linear(Concatenate([attn, dead_consumer]), 4, "out")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(d)
    for n in (pos, x, dead, dead_consumer):
        rmap.allocate(n)
    dead_cols = set(rmap.get_indices(dead))
    computed = {pos, x, dead, dead_consumer}

    scheduler = LayerScheduler(graph, d, d_head, pos, d_hidden=d)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    # The Attn op took the single head; the dead node was cancelled in the MLP
    # sublayer rather than the attention batch.
    assert any(op.op_type == "compute_attn" for op in attn_ops)
    assert not any(op.op_type == "cancel" for op in attn_ops), (
        "the dead node should not have been attention-cancelled — the head "
        "budget was full"
    )
    cancel_bypass = [op for op in mlp_ops if op.op_type == "cancel_bypass"]
    d_cancel = [op for op in cancel_bypass if set(op.target_cols) == dead_cols]
    assert d_cancel, f"expected a cancel_bypass zeroing D's columns {dead_cols}"
    op = d_cancel[0]
    assert op.node is None
    assert op.source_cols == op.target_cols
    assert len(op.mlp_slots) == 2 * len(op.target_cols)
    assert not rmap.is_allocated(dead), "D's columns should be freed"


def test_schedule_large_input_linear():
    """Zero-bias Linear with input dim > d_head: legacy uses attention, default uses bypass."""
    pos = _make_reserved_block()
    inputs = [InputNode(f"x{i}", 8, value_range=(-100.0, 100.0)) for i in range(4)]
    cat = Concatenate(inputs)
    d_out = 8
    W = torch.zeros(32, d_out)
    for i in range(32):
        W[i, i % d_out] = 1.0
    linear = Linear(cat, W, torch.zeros(d_out), name="sum")

    graph = GraphAnalyzer(linear)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    for inp in inputs:
        rmap.allocate(inp)
    computed = {pos} | set(inputs)

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=LEGACY_POLICY)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    linear_ops = [op for op in attn_ops if op.op_type == "compute_linear"]
    assert any(op.node is linear for op in linear_ops)
    assert linear in computed


def test_schedule_biased_linear():
    """Biased Linear: legacy uses attn Wx + MLP b, default uses bypass Wx+b."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    linear = _make_biased_linear(x, 3, "biased")

    # Legacy: attention Wx + MLP bias
    graph = GraphAnalyzer(linear)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=LEGACY_POLICY)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    linear_ops = [op for op in attn_ops if op.op_type == "compute_linear"]
    assert any(op.node is linear for op in linear_ops)
    bias_ops = [op for op in mlp_ops if op.op_type == "compute_bias"]
    assert any(op.node is linear for op in bias_ops)
    assert linear in computed


def test_schedule_biased_linear_bypass():
    """Biased Linear with default policy: bypass handles full Wx+b, no compute_bias."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    linear = _make_biased_linear(x, 3, "biased")

    graph = GraphAnalyzer(linear)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    bypass_ops = [op for op in mlp_ops if op.op_type == "compute_linear_bypass"]
    assert len(bypass_ops) == 1
    assert bypass_ops[0].node is linear
    bias_ops = [op for op in mlp_ops if op.op_type == "compute_bias"]
    assert len(bias_ops) == 0
    assert linear in computed


def test_schedule_cancellation():
    """Dead node (all consumers computed) produces a cancel operation."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    a = _make_linear(x, 4, "a")
    ffn = _make_ffn(a, 8, 3, "out")
    # Graph: x -> a -> ffn (output)
    # After computing a: x is dead (x's only consumer is a, which is computed)

    graph = GraphAnalyzer(ffn)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    rmap.allocate(a)
    computed = {pos, x, a}
    x_cols = set(rmap.get_indices(x))

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    cancel_ops = [op for op in attn_ops if op.op_type == "cancel"]
    # Cancels are coalesced into a single operation whose target_cols
    # is the union of all cols to clear in this layer — check that x's
    # cols are present in that batch.
    cancel_targets = {c for op in cancel_ops for c in op.target_cols}
    assert x_cols <= cancel_targets, (
        f"expected x's cols {x_cols} within cancel targets {cancel_targets}"
    )
    assert not rmap.is_allocated(x)  # columns freed


# ---------------------------------------------------------------------------
# 2. Add node behavior
# ---------------------------------------------------------------------------


def test_schedule_free_add():
    """Add with one dead-for-add addend produces add_into.

    dead_node's only consumer (besides add_node) is nothing — dead for add.
    live_node has another consumer (other) that isn't computed — NOT dead for add.
    """
    pos = _make_reserved_block()
    dead_node = InputNode("dead", 4, value_range=(-100.0, 100.0))
    live_node = InputNode("live", 4, value_range=(-100.0, 100.0))
    add_node = Add(dead_node, live_node)
    # Give live_node another consumer so it's NOT dead-for-add
    other = _make_linear(live_node, 4, "other")
    out = Add(add_node, other)

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(dead_node)
    rmap.allocate(live_node)
    computed = {pos, dead_node, live_node}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    add_ops = [op for op in attn_ops if op.op_type == "add_into"]
    assert len(add_ops) == 1
    assert add_ops[0].node is add_node
    assert add_node in computed


def test_schedule_deferred_add_via_compute():
    """Add where neither addend is dead-for-add is scheduled via compute_add.

    Both a and b have non-Add consumers that aren't computed yet, so add_into
    can't reuse columns. Instead, compute_add copies both inputs to fresh
    columns via attention.
    """
    # Use a larger d so the attention-head budget has room for both the
    # three compute ops (a_other, b_other, add_node) and the per-op
    # dirty-col cancellation emitted by the scheduler.  With the default
    # D=64 there are only 4 heads per layer, which is a tight fit that
    # cancellation can push over.
    d_test = 128
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    add_node = Add(a, b)
    a_other = _make_linear(a, 2, "a_other")
    b_other = _make_linear(b, 2, "b_other")
    out_cat = Concatenate([add_node, a_other, b_other])
    out = _make_linear(out_cat, 1, "out")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(d_test)
    rmap.allocate(pos)
    rmap.allocate(a)
    rmap.allocate(b)
    computed = {pos, a, b}

    scheduler = LayerScheduler(graph, d_test, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    add_ops = [op for op in attn_ops if op.node is add_node and op.op_type != "cancel"]
    assert len(add_ops) == 1
    assert add_ops[0].op_type == "compute_add"
    assert add_node in computed


def test_schedule_add_into_preferred_over_compute_add():
    """When one input is dead, add_into is used (not compute_add).

    add_into reuses existing columns (free), while compute_add allocates new
    ones. The scheduler should prefer the cheaper option.
    """
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    add_node = Add(a, b)
    # Give b another consumer so it's NOT dead-for-add, but a IS dead-for-add
    b_other = _make_linear(b, 4, "b_other")
    out = Add(add_node, b_other)

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(a)
    rmap.allocate(b)
    computed = {pos, a, b}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    add_ops = [op for op in attn_ops if op.node is add_node]
    assert len(add_ops) == 1
    assert add_ops[0].op_type == "add_into"  # NOT compute_add


def test_schedule_add_both_addends_dead():
    """Add where both addends are dead-for-add: produces add_into without error.

    Both a and b have no consumers besides add_node → both dead-for-add.
    """
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    add_node = Add(a, b)

    graph = GraphAnalyzer(add_node)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(a)
    rmap.allocate(b)
    computed = {pos, a, b}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    add_ops = [op for op in attn_ops if op.op_type == "add_into"]
    assert len(add_ops) == 1
    assert add_ops[0].node is add_node
    assert add_node in computed


# ---------------------------------------------------------------------------
# 3. Resource limits — scheduler respects budgets
# ---------------------------------------------------------------------------


def test_head_budget_exhaustion():
    """More ready attn ops than available heads: respects budget, defers excess."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    # 6 zero-bias Linears, but only N_HEADS=4 attention heads available
    linears = [_make_linear(x, 2, f"lin{i}") for i in range(6)]
    out_cat = Concatenate(linears)
    out = _make_linear(out_cat, 1, "out")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=LEGACY_POLICY)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    # Should not exceed head budget
    assert len(attn_ops) <= N_HEADS

    scheduled = [op.node for op in attn_ops if op.op_type == "compute_linear"]
    assert 0 < len(scheduled) <= N_HEADS


def test_mlp_slot_exhaustion():
    """More FFNs than MLP slots: respects slot budget."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    # 4 FFNs × 20 slots each = 80 > D=64
    ffns = []
    for i in range(4):
        ffns.append(_make_ffn(x, 20, 2, f"chain{i}"))
    out_cat = Concatenate(ffns)
    out = _make_linear(out_cat, 1, "out")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    total_slots = sum(
        len(op.mlp_slots) for op in mlp_ops if op.op_type == "compute_ffn"
    )
    assert total_slots <= D

    ffn_ops = [op for op in mlp_ops if op.op_type == "compute_ffn"]
    assert 0 < len(ffn_ops) < 4


# ---------------------------------------------------------------------------
# 4. Column pressure
# ---------------------------------------------------------------------------


def test_schedule_under_column_pressure():
    """Stream full with dead nodes: scheduler cancels to make room for new computes.

    Setup: D=64, pos=17, filler=39, x=4, a=4 → 0 free.
    x is dead (consumer a is computed). Relu chain needs 3 output cols.
    Scheduler must cancel x to free space, then schedule the chain.
    """
    pos = _make_reserved_block()
    filler = InputNode(
        "filler", D - len(pos) - 8, value_range=(-100.0, 100.0)
    )  # fills the stream to 0 free alongside pos + x + a
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    a = _make_linear(x, 4, "a")
    ffn = _make_ffn(a, 8, 3, "out")

    graph = GraphAnalyzer(ffn)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(filler)
    rmap.allocate(x)
    rmap.allocate(a)
    assert rmap.get_free_count() == 0
    computed = {pos, filler, x, a}
    x_cols = set(rmap.get_indices(x))

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    # Must cancel x (dead) to free columns; cancels are coalesced so
    # check the merged target_cols for x's columns.
    cancel_ops = [op for op in attn_ops if op.op_type == "cancel"]
    cancel_targets = {c for op in cancel_ops for c in op.target_cols}
    assert x_cols <= cancel_targets, (
        f"expected x's cols {x_cols} within cancel targets {cancel_targets}"
    )

    # FFN should still be scheduled
    ffn_ops = [op for op in mlp_ops if op.op_type == "compute_ffn"]
    assert len(ffn_ops) == 1
    assert ffn in computed


# ---------------------------------------------------------------------------
# 5. Multi-layer state progression
# ---------------------------------------------------------------------------


def test_multi_layer_progression():
    """Chained computation across multiple schedule_layer calls.

    Chain B depends on Chain A's output — must be scheduled in a later layer.
    """
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    block_a = _make_ffn(x, 8, 4, "a")
    block_b = _make_ffn(block_a, 6, 3, "b")

    graph = GraphAnalyzer(block_b)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)

    # Layer 1: first FFN scheduled
    scheduler.schedule_layer(rmap, computed)
    assert block_a in computed
    assert block_b not in computed

    # Layer 2: second FFN scheduled (depends on first)
    scheduler.schedule_layer(rmap, computed)
    assert block_b in computed


def test_deferred_add_fires_via_compute_add():
    """Add where neither addend is dead fires via compute_add in the first layer.

    Both addends have non-Add consumers (relu chains), so add_into can't be used.
    The compute_add path copies both inputs to fresh columns alongside the chains.
    """
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    a_chain = _make_ffn(a, 8, 2, "ac")
    b_chain = _make_ffn(b, 8, 2, "bc")
    add_node = Add(a, b)
    out_cat = Concatenate([add_node, a_chain, b_chain])
    out = _make_linear(out_cat, 1, "out")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(a)
    rmap.allocate(b)
    computed = {pos, a, b}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)

    # Layer 1: chains AND the Add are scheduled (Add via compute_add)
    scheduler.schedule_layer(rmap, computed)
    assert add_node in computed


# ---------------------------------------------------------------------------
# 6. Concatenate interaction
# ---------------------------------------------------------------------------


def test_scheduling_with_concatenate_input():
    """Node with Concatenate input: ready only when all Concatenate children computed.

    cat = Concatenate([a, b, c]). The relu chain depending on cat is not ready
    until all three children are computed.
    """
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    c = InputNode("c", 4, value_range=(-100.0, 100.0))
    cat = Concatenate([a, b, c])
    ffn = _make_ffn(cat, 8, 3, "out")

    graph = GraphAnalyzer(ffn)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(a)
    rmap.allocate(b)
    rmap.allocate(c)
    computed = {pos, a, b}  # c NOT computed

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    scheduler.schedule_layer(rmap, computed)

    # FFN not ready: c is missing
    assert ffn not in computed

    # Now add c to computed
    computed.add(c)
    scheduler.schedule_layer(rmap, computed)

    # FFN should fire
    assert ffn in computed


# ---------------------------------------------------------------------------
# 7. Mixed operations
# ---------------------------------------------------------------------------


def test_mixed_attn_and_mlp():
    """Both an Attn node and an FFN ready: both scheduled in same layer."""
    pos = _make_reserved_block()
    v = InputNode("v", 4, value_range=(-100.0, 100.0))
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    attn_node = _make_attn(v)
    ffn = _make_ffn(x, 8, 3, "chain")
    out_cat = Concatenate([attn_node, ffn])
    out = _make_linear(out_cat, 1, "out")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(v)
    rmap.allocate(x)
    computed = {pos, v, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    assert any(op.op_type == "compute_attn" and op.node is attn_node for op in attn_ops)
    assert any(op.op_type == "compute_ffn" and op.node is ffn for op in mlp_ops)
    assert attn_node in computed
    assert ffn in computed


# ---------------------------------------------------------------------------
# 9. Error cases
# ---------------------------------------------------------------------------


def test_no_progress_raises_error():
    """Raises error when no progress can be made.

    Residual stream is full (0 free cols). Ready nodes need columns.
    No dead nodes to cancel. → deadlock → error.
    """
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    a_consumer = _make_linear(a, 2, "ac")
    b_consumer = _make_linear(b, 2, "bc")
    add_node = Add(a, b)
    out_cat = Concatenate([add_node, a_consumer, b_consumer])
    out = _make_linear(out_cat, 1, "out")

    graph = GraphAnalyzer(out)
    # Tiny d: just enough for pos + a + b, nothing spare
    small_d = len(pos) + 4 + 4  # 17 + 4 + 4
    rmap = ResidualStreamMap(small_d)
    rmap.allocate(pos)  # len(pos) cols
    rmap.allocate(a)  # 4 cols
    rmap.allocate(b)  # 4 cols
    assert rmap.get_free_count() == 0
    computed = {pos, a, b}
    # a: consumers={add_node, a_consumer}, neither computed → not dead
    # b: consumers={add_node, b_consumer}, neither computed → not dead
    # Ready nodes (a_consumer, b_consumer) need 2 cols each, 0 available

    scheduler = LayerScheduler(graph, small_d, D_HEAD, pos)
    with pytest.raises(Exception):
        scheduler.schedule_layer(rmap, computed)


def test_add_into_shared_addend_not_reassigned():
    """Shared node used as live addend must not be reassigned as dead later.

    Bug pattern: A shared LiteralValue is an input to multiple Add nodes. All Adds
    become free_adds in the same layer. The step 2a loop processes them
    sequentially, adding each Add to computed_nodes. On the last Add, the shared
    LiteralValue's other consumers (the earlier Adds) are now computed, making the
    LiteralValue dead-for-add. The scheduler reassigns the LiteralValue's columns to the
    last Add — but the earlier Adds' ops still reference the LiteralValue as their
    live addend, and the weight writer needs its columns.

    This is the exact bug from the calculator's switch() pattern.
    """
    pos = _make_reserved_block()
    shared = LiteralValue(torch.randn(4))

    # 3 Add nodes sharing the same LiteralValue, each with a unique dead addend
    dead_nodes = [
        InputNode(f"dead{i}", 4, value_range=(-100.0, 100.0)) for i in range(3)
    ]
    adds = [Add(shared, dn) for dn in dead_nodes]
    # Wire into output so graph includes everything
    out_cat = Concatenate(adds)
    out = _make_linear(out_cat, 2, "out")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(shared)
    for dn in dead_nodes:
        rmap.allocate(dn)
    computed = {pos, shared} | set(dead_nodes)

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    add_into_ops = [op for op in attn_ops if op.op_type == "add_into"]
    assert len(add_into_ops) == 3, f"Expected 3 add_into ops, got {len(add_into_ops)}"

    # Replicate the weight writer's live-addend resolution:
    # if a0 is allocated, live = a0; else live = a1.
    # Then resolve_indices(live) must succeed.
    for op in add_into_ops:
        a0, a1 = op.node.inputs
        live = a0 if rmap.is_allocated(a0) else a1
        try:
            rmap.resolve_indices(live)
        except KeyError:
            pytest.fail(
                f"add_into live addend not in residual map — shared node "
                f"was reassigned as dead addend in the same batch. "
                f"a0={type(a0).__name__}(alloc={rmap.is_allocated(a0)}) "
                f"a1={type(a1).__name__}(alloc={rmap.is_allocated(a1)})"
            )


def test_output_already_computed():
    """When all graph nodes are already computed, schedule_layer doesn't error."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    out = _make_linear(x, 2, "out")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    rmap.allocate(out)
    computed = {pos, x, out}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    # Should not raise — nothing to do
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)


def test_within_layer_freeing_surfaces_freshly_dead_intermediate():
    """Within-layer (eager) freeing is always on: placing ``b`` makes its input
    ``a`` freshly dead, and ``_freshly_dead_inputs`` surfaces it so ``a``'s
    column can be reclaimed in the same layer.  Units 1 and 2 made this density
    CP-SAT-representable, so the warm start and the production heuristic share
    it (the ``eager_free=False`` mode that used to gate this off is gone).
    """
    pos = _make_reserved_block()
    x = InputNode("x", D, value_range=(-100.0, 100.0))
    a = _make_linear(x, D, "a")  # intermediate: a's only consumer is b
    b = _make_linear(a, D, "b")
    graph = GraphAnalyzer(b)

    rmap = ResidualStreamMap(D * 4)
    rmap.allocate(a)  # a is live; b has just been placed -> a is now dead

    computed = {x, a, b}
    scheduler = LayerScheduler(graph, D, D_HEAD, pos)

    assert a in scheduler._freshly_dead_inputs(b, computed, rmap)


# ---------------------------------------------------------------------------
# MLP-routed Adds (docs/plan_additional_mlp_routing.md)
#
# Under an MLP-preferring Add policy (add_in_attention="never"; the
# shipping default keeps Adds on attention) a fitting Add resolves to
# MLP_ADD; the walk carries it into the MLP phase and derives fresh/reused
# placement from the MLP phase-start snapshot.
# ---------------------------------------------------------------------------


def test_mlp_add_reuses_dead_addend():
    """A reusable MLP Add emits add_into_bypass and reassigns the dead
    addend's columns; no attention Add op is emitted.
    """
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    add_node = Add(a, b, name="add")
    b_other = _make_linear(b, 4, "b_other")  # b stays live (pending consumer)
    out = Concatenate([add_node, b_other])

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    a_cols = rmap.allocate(a)
    rmap.allocate(b)
    computed = {pos, a, b}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=MLP_ADD_POLICY)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    assert not [op for op in attn_ops if op.op_type in ("add_into", "compute_add")]
    add_ops = [op for op in mlp_ops if op.op_type == "add_into_bypass"]
    assert len(add_ops) == 1
    op = add_ops[0]
    assert op.node is add_node
    assert op.reuse_input_index == 0
    assert op.target_cols == a_cols
    assert op.mlp_slots and len(op.mlp_slots) == 2 * 4
    # The dead addend's columns now belong to the Add.
    assert rmap.get_indices(add_node) == a_cols
    assert not rmap.is_allocated(a)


def test_mlp_add_fresh_allocates_columns():
    """A fresh MLP Add (both addends still live) emits compute_add_bypass
    into freshly allocated columns.
    """
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    add_node = Add(a, b, name="add")
    a_other = _make_linear(a, 4, "a_other")
    b_other = _make_linear(b, 4, "b_other")
    out = Concatenate([add_node, a_other, b_other])

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    a_cols = rmap.allocate(a)
    b_cols = rmap.allocate(b)
    computed = {pos, a, b}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=MLP_ADD_POLICY)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    add_ops = [op for op in mlp_ops if op.op_type == "compute_add_bypass"]
    assert len(add_ops) == 1
    op = add_ops[0]
    assert op.reuse_input_index is None
    assert op.source_cols == a_cols and op.source_cols_b == b_cols
    assert set(op.target_cols).isdisjoint(set(a_cols) | set(b_cols))
    assert rmap.is_allocated(a) and rmap.is_allocated(b)


def test_mlp_add_both_dead_selects_input0():
    """When both addends are reusable, occurrence 0 is the target on the
    MLP route (same deterministic tie-break as attention).
    """
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    add_node = Add(a, b, name="add")

    graph = GraphAnalyzer(add_node)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    a_cols = rmap.allocate(a)
    rmap.allocate(b)
    computed = {pos, a, b}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=MLP_ADD_POLICY)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    add_ops = [op for op in mlp_ops if op.op_type == "add_into_bypass"]
    assert len(add_ops) == 1
    assert add_ops[0].reuse_input_index == 0
    assert add_ops[0].target_cols == a_cols
    assert rmap.is_allocated(b)  # occurrence 1 is the source read


def test_attention_producer_feeds_mlp_add_same_layer():
    """An attention result (Attn node) may feed an MLP-routed Add in the
    same layer: the Add joins the MLP candidates after the attention pass,
    and the attention-born value is a legal reuse target at the MLP
    phase-start snapshot.
    """
    pos = _make_reserved_block()
    v = InputNode("v", 4, value_range=(-100.0, 100.0))
    attn_node = _make_attn(v)
    other = InputNode("other", 4, value_range=(-100.0, 100.0))
    other_keeper = _make_linear(other, 4, "keeper")  # other stays live
    add_node = Add(attn_node, other, name="add")
    out = Concatenate([add_node, other_keeper])

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(v)
    rmap.allocate(other)
    computed = {pos, v, other}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=MLP_ADD_POLICY)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    assert [op for op in attn_ops if op.op_type == "compute_attn"]
    add_ops = [
        op for op in mlp_ops if op.op_type in ("add_into_bypass", "compute_add_bypass")
    ]
    assert len(add_ops) == 1, "the Add must place in the same layer's MLP phase"
    # The attention output's only consumer is the Add, so it is dead at the
    # MLP snapshot and its columns are reused.
    assert add_ops[0].op_type == "add_into_bypass"
    assert add_ops[0].reuse_input_index == 0
    assert rmap.get_indices(add_node)


def test_mlp_producer_defers_mlp_add_to_next_layer():
    """An MLP-routed producer (FFN) cannot feed an MLP Add in the same
    layer — the Add waits (one-layer dependency gap).
    """
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    ffn = _make_ffn(x, 6, 4, "ffn")
    other = InputNode("other", 4, value_range=(-100.0, 100.0))
    add_node = Add(ffn, other, name="add")

    graph = GraphAnalyzer(add_node)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    rmap.allocate(other)
    computed = {pos, x, other}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=MLP_ADD_POLICY)
    attn_ops0, mlp_ops0, _ = scheduler.schedule_layer(rmap, computed)
    assert [op for op in mlp_ops0 if op.op_type == "compute_ffn"]
    assert not [
        op for op in mlp_ops0 if op.op_type in ("add_into_bypass", "compute_add_bypass")
    ]

    attn_ops1, mlp_ops1, _ = scheduler.schedule_layer(rmap, computed)
    assert [
        op for op in mlp_ops1 if op.op_type in ("add_into_bypass", "compute_add_bypass")
    ]


def test_same_layer_consumer_reusability_is_sublayer_aware():
    """A same-layer attention consumer of an addend counts complete for an
    MLP Add (the attention phase precedes the MLP snapshot); a same-layer
    MLP consumer does not.
    """
    # Attention peer: a is reusable.
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    add_node = Add(a, b, name="add")
    attn_peer = _make_attn(a)
    b_keeper = _make_linear(b, 4, "b_keeper")
    out = Concatenate([add_node, attn_peer, b_keeper])

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    a_cols = rmap.allocate(a)
    rmap.allocate(b)
    computed = {pos, a, b}
    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=MLP_ADD_POLICY)
    attn_ops, mlp_ops, _ = scheduler.schedule_layer(rmap, computed)
    add_ops = [op for op in mlp_ops if op.op_type == "add_into_bypass"]
    assert len(add_ops) == 1 and add_ops[0].reuse_input_index == 0
    assert add_ops[0].target_cols == a_cols

    # MLP peer: a is NOT reusable (and b is kept live), so the Add is fresh.
    pos2 = _make_reserved_block()
    a2 = InputNode("a2", 4, value_range=(-100.0, 100.0))
    b2 = InputNode("b2", 4, value_range=(-100.0, 100.0))
    add2 = Add(a2, b2, name="add2")
    mlp_peer = _make_linear(a2, 4, "mlp_peer")  # default policy: MLP bypass
    b2_keeper = _make_linear(b2, 4, "b2_keeper")
    out2 = Concatenate([add2, mlp_peer, b2_keeper])

    graph2 = GraphAnalyzer(out2)
    rmap2 = ResidualStreamMap(D)
    rmap2.allocate(pos2)
    rmap2.allocate(a2)
    rmap2.allocate(b2)
    computed2 = {pos2, a2, b2}
    scheduler2 = LayerScheduler(graph2, D, D_HEAD, pos2, policy=MLP_ADD_POLICY)
    attn_ops2, mlp_ops2, _ = scheduler2.schedule_layer(rmap2, computed2)
    add_ops2 = [
        op for op in mlp_ops2 if op.op_type in ("add_into_bypass", "compute_add_bypass")
    ]
    assert len(add_ops2) == 1
    assert add_ops2[0].op_type == "compute_add_bypass"
    assert rmap2.is_allocated(a2) and rmap2.is_allocated(b2)


def test_mlp_add_slot_exhaustion_defers_without_spilling():
    """When a layer's hidden slots cannot hold a second MLP Add, it defers
    to the next layer and never spills to attention.
    """
    pos = _make_reserved_block()
    inputs = [InputNode(f"i{k}", 4, value_range=(-100.0, 100.0)) for k in range(4)]
    add_a = Add(inputs[0], inputs[1], name="add_a")
    add_b = Add(inputs[2], inputs[3], name="add_b")
    out = Concatenate([add_a, add_b])

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    for i in inputs:
        rmap.allocate(i)
    computed = {pos, *inputs}

    # Each reused Add needs 2*4 = 8 slots; d_hidden=12 holds only one.
    scheduler = LayerScheduler(
        graph, D, D_HEAD, pos, d_hidden=12, policy=MLP_ADD_POLICY
    )
    attn_ops0, mlp_ops0, _ = scheduler.schedule_layer(rmap, computed)
    placed0 = [op for op in mlp_ops0 if op.op_type == "add_into_bypass"]
    assert len(placed0) == 1
    assert not [op for op in attn_ops0 if op.op_type in ("add_into", "compute_add")]

    attn_ops1, mlp_ops1, _ = scheduler.schedule_layer(rmap, computed)
    placed1 = [op for op in mlp_ops1 if op.op_type == "add_into_bypass"]
    assert len(placed1) == 1
    assert not [op for op in attn_ops1 if op.op_type in ("add_into", "compute_add")]


def test_mlp_add_live_source_not_cancelled_same_layer():
    """The live/source occurrence of an add_into_bypass is excluded from
    same-layer cancellation (gap #2 conservatism, extended to the MLP
    route): even when the Add is its final consumer, it stays allocated
    through the Add's layer.
    """
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    lin = _make_linear(x, 4, "lin")
    add_node = Add(a, lin, name="add")  # a: reuse target; lin: live source

    graph = GraphAnalyzer(add_node)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    rmap.allocate(a)
    computed = {pos, x, a}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=MLP_ADD_POLICY)
    # Layer 0: lin places (MLP bypass under the default policy).
    scheduler.schedule_layer(rmap, computed)
    assert lin in computed and rmap.is_allocated(lin)
    lin_cols = set(rmap.get_indices(lin))

    # Layer 1: the Add reuses a's columns and reads lin as the live source.
    attn_ops, mlp_ops, _ = scheduler.schedule_layer(rmap, computed)
    add_ops = [op for op in mlp_ops if op.op_type == "add_into_bypass"]
    assert len(add_ops) == 1 and add_ops[0].reuse_input_index == 0
    cancel_cols = {
        c for op in mlp_ops if op.op_type == "cancel_bypass" for c in op.target_cols
    }
    assert not (cancel_cols & lin_cols), (
        "the live source of an add_into_bypass must not be cancelled in the "
        "Add's own layer"
    )
    assert rmap.is_allocated(lin)
