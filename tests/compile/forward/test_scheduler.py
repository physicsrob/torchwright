"""TDD tests for the forward compiler's LayerScheduler.

Tests verify observable behavior: ops returned, state changes, errors raised.
No tests for internal ordering heuristics or priority strategies — efficiency
is tested at the integration level (Phase 4/5).

Conventions:
- D=64, D_HEAD=16 → n_heads=4 attention heads per layer.
- pos_encoding is always allocated but may not be a graph node.
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
from torchwright.compiler.forward.weight_writer import AttnHeadOp, MLPOp
from torchwright.graph import Linear, ReLU, Attn, Add, Concatenate
from torchwright.graph.misc import InputNode, LiteralValue
from torchwright.graph.pos_encoding import PosEncoding

D = 64
D_HEAD = 16
N_HEADS = D // D_HEAD  # 4


def _make_pos_encoding():
    return PosEncoding(17)  # trig_width 16 == D_HEAD


def _make_linear(inp, d_out, name=""):
    """Zero-bias Linear."""
    return Linear(inp, torch.randn(len(inp), d_out), torch.zeros(d_out), name=name)


def _make_biased_linear(inp, d_out, name=""):
    """Linear with non-zero bias."""
    return Linear(inp, torch.randn(len(inp), d_out), torch.randn(d_out), name=name)


def _make_relu_chain(inp, d_hidden, d_out, name=""):
    """L1 -> ReLU -> L2 chain. Returns (l2, relu, l1)."""
    l1 = Linear(
        inp, torch.randn(len(inp), d_hidden), torch.randn(d_hidden), name=f"{name}_l1"
    )
    r = ReLU(l1, name=f"{name}_r")
    l2 = Linear(r, torch.randn(d_hidden, d_out), torch.randn(d_out), name=f"{name}_l2")
    return l2, r, l1


# ---------------------------------------------------------------------------
# 1. Basic op routing — correct node type produces correct op type
# ---------------------------------------------------------------------------


def test_schedule_attn_node():
    """Attn node produces AttnHeadOp('compute_attn')."""
    pos = _make_pos_encoding()
    v = InputNode("v", 4, value_range=(-100.0, 100.0))
    attn_node = pos.attend_to_offset(v, delta_pos=-1)

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


def test_schedule_relu_chain():
    """L->R->L chain produces MLPOp('compute_relu'); all 3 nodes marked computed."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    l2, r, l1 = _make_relu_chain(x, 8, 3, "chain")

    graph = GraphAnalyzer(l2)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    relu_ops = [op for op in mlp_ops if op.op_type == "compute_relu"]
    assert len(relu_ops) == 1
    assert relu_ops[0].node is l2
    assert len(relu_ops[0].mlp_slots) == 8  # intermediate dim

    assert l1 in computed
    assert r in computed
    assert l2 in computed


def test_schedule_constant():
    """LiteralValue node produces MLPOp('compute_literal_value') with no MLP slots.

    Note: in the compile loop, Constants are typically pre-populated as input nodes.
    This tests the scheduler's capability to handle Constants that aren't pre-populated.
    """
    pos = _make_pos_encoding()
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
    pos = _make_pos_encoding()
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
    """Zero-bias Linear with default policy produces MLPOp('compute_linear_bypass')."""
    pos = _make_pos_encoding()
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


def test_schedule_large_input_linear():
    """Zero-bias Linear with input dim > d_head: legacy uses attention, default uses bypass."""
    pos = _make_pos_encoding()
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
    pos = _make_pos_encoding()
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
    pos = _make_pos_encoding()
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
    """Dead node (all consumers computed) produces cancel AttnHeadOp."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    a = _make_linear(x, 4, "a")
    l2, r, l1 = _make_relu_chain(a, 8, 3, "out")
    # Graph: x -> a -> l1 -> r -> l2 (output)
    # After computing a: x is dead (x's only consumer is a, which is computed)

    graph = GraphAnalyzer(l2)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    rmap.allocate(a)
    computed = {pos, x, a}
    x_cols = set(rmap.get_indices(x))

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    cancel_ops = [op for op in attn_ops if op.op_type == "cancel"]
    # Cancels are coalesced into a single AttnHeadOp whose target_cols
    # is the union of all cols to clear in this layer — check that x's
    # cols are present in that batch.
    cancel_targets = {c for op in cancel_ops for c in op.target_cols}
    assert (
        x_cols <= cancel_targets
    ), f"expected x's cols {x_cols} within cancel targets {cancel_targets}"
    assert not rmap.is_allocated(x)  # columns freed


# ---------------------------------------------------------------------------
# 2. Add node behavior
# ---------------------------------------------------------------------------


def test_schedule_free_add():
    """Add with one dead-for-add addend produces add_into AttnHeadOp.

    dead_node's only consumer (besides add_node) is nothing — dead for add.
    live_node has another consumer (other) that isn't computed — NOT dead for add.
    """
    pos = _make_pos_encoding()
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
    pos = _make_pos_encoding()
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
    rmap.mark_clean(rmap.get_indices(pos))
    rmap.mark_clean(rmap.get_indices(a))
    rmap.mark_clean(rmap.get_indices(b))
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
    pos = _make_pos_encoding()
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
    pos = _make_pos_encoding()
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
    pos = _make_pos_encoding()
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
    """More L->R->L chains than MLP slots: respects slot budget."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    # 4 chains × 20 slots each = 80 > D=64
    chains = []
    for i in range(4):
        l2, _, _ = _make_relu_chain(x, 20, 2, f"chain{i}")
        chains.append(l2)
    out_cat = Concatenate(chains)
    out = _make_linear(out_cat, 1, "out")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    total_slots = sum(
        len(op.mlp_slots) for op in mlp_ops if op.op_type == "compute_relu"
    )
    assert total_slots <= D

    relu_ops = [op for op in mlp_ops if op.op_type == "compute_relu"]
    assert 0 < len(relu_ops) < 4


# ---------------------------------------------------------------------------
# 4. Column pressure
# ---------------------------------------------------------------------------


def test_schedule_under_column_pressure():
    """Stream full with dead nodes: scheduler cancels to make room for new computes.

    Setup: D=64, pos=17, filler=39, x=4, a=4 → 0 free.
    x is dead (consumer a is computed). Relu chain needs 3 output cols.
    Scheduler must cancel x to free space, then schedule the chain.
    """
    pos = _make_pos_encoding()
    filler = InputNode(
        "filler", D - len(pos) - 8, value_range=(-100.0, 100.0)
    )  # fills the stream to 0 free alongside pos + x + a
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    a = _make_linear(x, 4, "a")
    l2, r, l1 = _make_relu_chain(a, 8, 3, "out")

    graph = GraphAnalyzer(l2)
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
    assert (
        x_cols <= cancel_targets
    ), f"expected x's cols {x_cols} within cancel targets {cancel_targets}"

    # Relu chain should still be scheduled
    relu_ops = [op for op in mlp_ops if op.op_type == "compute_relu"]
    assert len(relu_ops) == 1
    assert l2 in computed


# ---------------------------------------------------------------------------
# 5. Multi-layer state progression
# ---------------------------------------------------------------------------


def test_multi_layer_progression():
    """Chained computation across multiple schedule_layer calls.

    Chain B depends on Chain A's output — must be scheduled in a later layer.
    """
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    l2a, ra, l1a = _make_relu_chain(x, 8, 4, "a")
    l2b, rb, l1b = _make_relu_chain(l2a, 6, 3, "b")

    graph = GraphAnalyzer(l2b)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)

    # Layer 1: first chain scheduled
    scheduler.schedule_layer(rmap, computed)
    assert l2a in computed
    assert l2b not in computed

    # Layer 2: second chain scheduled (depends on first)
    scheduler.schedule_layer(rmap, computed)
    assert l2b in computed


def test_deferred_add_fires_via_compute_add():
    """Add where neither addend is dead fires via compute_add in the first layer.

    Both addends have non-Add consumers (relu chains), so add_into can't be used.
    The compute_add path copies both inputs to fresh columns alongside the chains.
    """
    pos = _make_pos_encoding()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    a_chain, _, _ = _make_relu_chain(a, 8, 2, "ac")
    b_chain, _, _ = _make_relu_chain(b, 8, 2, "bc")
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
    pos = _make_pos_encoding()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    c = InputNode("c", 4, value_range=(-100.0, 100.0))
    cat = Concatenate([a, b, c])
    l2, r, l1 = _make_relu_chain(cat, 8, 3, "out")

    graph = GraphAnalyzer(l2)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(a)
    rmap.allocate(b)
    rmap.allocate(c)
    computed = {pos, a, b}  # c NOT computed

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    scheduler.schedule_layer(rmap, computed)

    # Chain not ready: c is missing
    assert l2 not in computed

    # Now add c to computed
    computed.add(c)
    scheduler.schedule_layer(rmap, computed)

    # Chain should fire
    assert l2 in computed


# ---------------------------------------------------------------------------
# 7. Mixed operations
# ---------------------------------------------------------------------------


def test_mixed_attn_and_mlp():
    """Both Attn node and L->R->L chain ready: both scheduled in same layer."""
    pos = _make_pos_encoding()
    v = InputNode("v", 4, value_range=(-100.0, 100.0))
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    attn_node = pos.attend_to_offset(v, delta_pos=-1)
    l2, r, l1 = _make_relu_chain(x, 8, 3, "chain")
    out_cat = Concatenate([attn_node, l2])
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
    assert any(op.op_type == "compute_relu" and op.node is l2 for op in mlp_ops)
    assert attn_node in computed
    assert l2 in computed


# ---------------------------------------------------------------------------
# 8. L->R->L chain edge cases and standalone ReLU
# ---------------------------------------------------------------------------


def test_schedule_standalone_relu():
    """Standalone ReLU (not part of L->R->L chain) produces MLPOp('compute_standalone_relu').

    This pattern occurs in cond_gate: ReLU(Add(...)) where the ReLU's consumer
    is an Add, not a Linear — so it's not part of any chain.
    """
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    relu_node = ReLU(x, name="standalone")
    # ReLU feeds into an Add, not a Linear → not part of a chain
    other = InputNode("other", 4, value_range=(-100.0, 100.0))
    add_node = Add(relu_node, other)

    graph = GraphAnalyzer(add_node)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    rmap.allocate(other)
    computed = {pos, x, other}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, _biased = scheduler.schedule_layer(rmap, computed)

    relu_ops = [op for op in mlp_ops if op.op_type == "compute_standalone_relu"]
    assert len(relu_ops) == 1
    assert relu_ops[0].node is relu_node
    assert len(relu_ops[0].mlp_slots) == len(relu_node)  # 4 slots for 4-dim
    assert relu_node in computed


def test_relu_chain_broken_by_fanout():
    """L1 has consumers besides its ReLU — scheduler still makes progress.

    The chain can't be claimed as a unit because L1 also feeds 'other'.
    The scheduler must handle this (e.g., schedule L1 separately, use alternative
    strategy, or compute the chain while also placing L1 in the stream).
    """
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(x, torch.randn(4, 8), torch.randn(8), name="l1")
    r = ReLU(l1, name="r")
    l2 = Linear(r, torch.randn(8, 3), torch.randn(3), name="l2")
    # l1 also feeds another node — breaks the exclusive chain pattern
    other = _make_linear(l1, 2, "other")
    out_cat = Concatenate([l2, other])
    out = _make_linear(out_cat, 1, "final")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    # Use legacy policy so L1 goes to attention (non-exclusive chain + attention L1)
    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=LEGACY_POLICY)
    before = len(computed)
    scheduler.schedule_layer(rmap, computed)

    # Must make progress — at least one new node computed
    assert len(computed) > before


def test_relu_chain_broken_by_fanout_bypass():
    """Same as above but with default bypass policy — still makes progress."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(x, torch.randn(4, 8), torch.randn(8), name="l1")
    r = ReLU(l1, name="r")
    l2 = Linear(r, torch.randn(8, 3), torch.randn(3), name="l2")
    other = _make_linear(l1, 2, "other")
    out_cat = Concatenate([l2, other])
    out = _make_linear(out_cat, 1, "final")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    before = len(computed)
    scheduler.schedule_layer(rmap, computed)

    assert len(computed) > before


def test_relu_chain_broken_by_fanout_bypass_emits_l1_standalone():
    """Non-exclusive L1 + bypass policy: the bypass loop emits a
    standalone ``compute_linear_bypass(L1)`` alongside the chain
    composite's ``compute_relu(L2)``, in the same MLP sublayer.

    Regression test for the smallest reproducing layer of the
    silent value-corruption bug where the chain block marked L1 as
    ``computed`` and pre-allocated its residual cols but emitted no
    op that wrote a value into them — leaving consumers of L1 to
    read uninitialized data.
    """
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(x, torch.randn(4, 8), torch.randn(8), name="l1")
    r = ReLU(l1, name="r")
    l2 = Linear(r, torch.randn(8, 3), torch.randn(3), name="l2")
    other = _make_linear(l1, 2, "other")
    out_cat = Concatenate([l2, other])
    out = _make_linear(out_cat, 1, "final")

    graph = GraphAnalyzer(out)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    computed = {pos, x}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, biased = scheduler.schedule_layer(rmap, computed)

    # The chain composite must be emitted (writes L2).
    relu_ops = [op for op in mlp_ops if op.op_type == "compute_relu"]
    assert len(relu_ops) == 1
    assert relu_ops[0].node is l2

    # AND a separate compute_linear_bypass for L1 must be emitted in
    # the same MLP sublayer — without it, L1's residual cols are
    # allocated but never written and consumers read garbage.
    bypass_ops = [
        op for op in mlp_ops
        if op.op_type == "compute_linear_bypass" and op.node is l1
    ]
    assert len(bypass_ops) == 1, (
        f"Expected a compute_linear_bypass(L1) op in the same MLP "
        f"sublayer as the chain composite, but mlp_ops were: "
        f"{[(op.op_type, getattr(op.node, 'name', op.node)) for op in mlp_ops]}. "
        f"L1 must be written when it has consumers besides the "
        f"chain's ReLU."
    )

    # Both the chain composite and the bypass realization should
    # have actually completed: L1, R, and L2 in computed_nodes.
    assert l1 in computed
    assert r in computed
    assert l2 in computed
    # L1's residual cols are allocated and were just written by the
    # bypass op.
    assert rmap.is_allocated(l1)
    assert bypass_ops[0].target_cols == rmap.get_indices(l1)


def test_linear_with_computed_relu_input():
    """Linear whose input ReLU is already computed can be scheduled standalone.

    After Linear fusion or certain scheduling patterns, L1 and ReLU may be
    computed in earlier layers while L2 is deferred. L2's input is a ReLU node
    that's already in the residual stream, so L2 should be schedulable as a
    standalone Linear rather than as part of a chain.
    """
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(x, torch.randn(4, 8), torch.randn(8), name="l1")
    r = ReLU(l1, name="r")
    l2 = Linear(r, torch.randn(8, 3), torch.randn(3), name="l2")

    graph = GraphAnalyzer(l2)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    rmap.allocate(l1)
    rmap.allocate(r)
    computed = {pos, x, l1, r}

    # Legacy: scheduled via attention
    scheduler = LayerScheduler(graph, D, D_HEAD, pos, policy=LEGACY_POLICY)
    attn_ops, mlp_ops, biased = scheduler.schedule_layer(rmap, computed)

    compute_linears = [op for op in attn_ops if op.op_type == "compute_linear"]
    assert len(compute_linears) == 1
    assert compute_linears[0].node is l2
    assert l2 in computed


def test_linear_with_computed_relu_input_bypass():
    """Same scenario with default policy — scheduled via MLP bypass."""
    pos = _make_pos_encoding()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    l1 = Linear(x, torch.randn(4, 8), torch.randn(8), name="l1")
    r = ReLU(l1, name="r")
    l2 = Linear(r, torch.randn(8, 3), torch.randn(3), name="l2")

    graph = GraphAnalyzer(l2)
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    rmap.allocate(l1)
    rmap.allocate(r)
    computed = {pos, x, l1, r}

    scheduler = LayerScheduler(graph, D, D_HEAD, pos)
    attn_ops, mlp_ops, biased = scheduler.schedule_layer(rmap, computed)

    bypass_ops = [op for op in mlp_ops if op.op_type == "compute_linear_bypass"]
    assert len(bypass_ops) == 1
    assert bypass_ops[0].node is l2
    assert l2 in computed

    # Should NOT be scheduled via compute_relu (that's for full chains)
    relu_ops = [op for op in mlp_ops if op.op_type == "compute_relu"]
    assert len(relu_ops) == 0


# ---------------------------------------------------------------------------
# 9. Error cases
# ---------------------------------------------------------------------------


def test_no_progress_raises_error():
    """Raises error when no progress can be made.

    Residual stream is full (0 free cols). Ready nodes need columns.
    No dead nodes to cancel. → deadlock → error.
    """
    pos = _make_pos_encoding()
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
    pos = _make_pos_encoding()
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
    pos = _make_pos_encoding()
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
