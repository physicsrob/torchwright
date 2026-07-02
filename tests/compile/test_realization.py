"""Tests for the realization table (:mod:`torchwright.compiler.realization`).

One option-set declaration (candidate_classes) read by both resolvers; the
table is the single artifact recording which hardware runs each node.  The
static policy resolves it at optimize=0; the CP-SAT solve's node_to_routing
resolves it on the directed path; the layer walk only reads it.
"""

import pytest
import torch

from torchwright.compiler.lower import lower
from torchwright.compiler.realization import (
    ATTN_COPY,
    ATTN_HEADS,
    ATTN_TRANSPORT,
    MLP_BYPASS,
    MLP_COMPOSITE,
    MLP_LITERAL,
    RESIDUAL_REUSE,
    RealizationTable,
    UnresolvedRealizationError,
    candidate_classes,
    has_flex_choice,
    is_conditional,
)
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)
from torchwright.graph import Add, Linear
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import Concatenate, LiteralValue
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear_relu_linear import linear_relu_linear


def _block(x, d_input, n_lanes, d_output, seed=0):
    g = torch.Generator().manual_seed(seed)
    return linear_relu_linear(
        x,
        torch.randn(n_lanes, d_input, generator=g),
        torch.randn(n_lanes, generator=g),
        torch.randn(n_lanes, d_output, generator=g),
        torch.randn(d_output, generator=g),
    )


def _test_graph():
    """x -> {a, b} Linears -> Add; plus an FFN and a LiteralValue."""
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    a = Linear(x, torch.randn(4, 3) * 0.2, torch.randn(3) * 0.1, name="a")
    b = Linear(x, torch.randn(4, 3) * 0.2, torch.randn(3) * 0.1, name="b")
    add = Add(a, b, name="add")
    blk = _block(x, 4, 6, 3, seed=2)
    lit = LiteralValue(torch.ones(2), name="lit")
    out = Concatenate([add, blk, lit])
    return out, a, b, add, blk, lit


# ---------------------------------------------------------------------------
# The option set
# ---------------------------------------------------------------------------


def test_candidate_classes_declaration():
    out, a, b, add, blk, lit = _test_graph()
    assert candidate_classes(a) == (ATTN_TRANSPORT, MLP_BYPASS)
    assert candidate_classes(add) == (RESIDUAL_REUSE, ATTN_COPY)
    assert candidate_classes(blk) == (MLP_COMPOSITE,)
    assert candidate_classes(lit) == (MLP_LITERAL,)
    assert isinstance(blk, FFN)
    with pytest.raises(TypeError):
        candidate_classes(out)  # Concatenate is not schedulable

    assert has_flex_choice(a) and not is_conditional(a)
    assert is_conditional(add) and not has_flex_choice(add)
    assert not has_flex_choice(blk)


def test_attn_candidate_class():
    # Attn nodes are heavier to construct; check the declaration directly
    # via an instance-free type probe using the dispatch on a real graph in
    # the compile-level tests below — here just pin the class constant.
    assert ATTN_HEADS == "attn_heads"


# ---------------------------------------------------------------------------
# Table construction and resolution
# ---------------------------------------------------------------------------


def test_lower_builds_unresolved_table():
    out, a, b, add, blk, lit = _test_graph()
    table = lower(out).realization_table

    ea = table.entries[a.node_id]
    assert ea.resolved is None and not ea.conditional
    assert ea.candidates == (ATTN_TRANSPORT, MLP_BYPASS)

    eadd = table.entries[add.node_id]
    assert eadd.conditional and eadd.resolved is None

    assert table.entries[blk.node_id].resolved == MLP_COMPOSITE
    assert table.entries[lit.node_id].resolved == MLP_LITERAL
    # Non-schedulable nodes get no entry.
    assert out.node_id not in table.entries


def test_resolve_static_both_policies():
    out, a, b, add, blk, lit = _test_graph()
    table = lower(out).realization_table

    bypass = table.resolve_static(SchedulingPolicy())  # local_in_attention="never"
    assert bypass.entries[a.node_id].resolved == MLP_BYPASS
    assert bypass.entries[b.node_id].resolved == MLP_BYPASS
    assert not bypass.is_attention_routed(a)

    attn = table.resolve_static(LEGACY_POLICY)  # "always"
    assert attn.entries[a.node_id].resolved == ATTN_TRANSPORT
    assert attn.is_attention_routed(a)

    # Conditional and single-candidate entries are untouched by resolution.
    for t in (bypass, attn):
        assert t.entries[add.node_id].conditional
        assert t.entries[blk.node_id].resolved == MLP_COMPOSITE

    # The unresolved source table is not mutated (resolvers return new).
    assert table.entries[a.node_id].resolved is None


def test_resolve_from_assignment():
    out, a, b, add, blk, lit = _test_graph()
    table = lower(out).realization_table

    routing = {
        a.node_id: "attn",
        b.node_id: "mlp",
        add.node_id: "attn",
        blk.node_id: "mlp",
        lit.node_id: "mlp",
    }
    resolved = table.resolve_from_assignment(routing)
    assert resolved.entries[a.node_id].resolved == ATTN_TRANSPORT
    assert resolved.entries[b.node_id].resolved == MLP_BYPASS
    assert resolved.entries[add.node_id].conditional
    assert resolved.entries[blk.node_id].resolved == MLP_COMPOSITE


def test_resolve_from_assignment_rejects_contradiction():
    out, a, b, add, blk, lit = _test_graph()
    table = lower(out).realization_table
    # The solve routing an FFN to attention contradicts its only class.
    with pytest.raises(UnresolvedRealizationError, match="only realization"):
        table.resolve_from_assignment({blk.node_id: "attn"})


def test_check_complete_and_unresolved_read():
    out, a, b, add, blk, lit = _test_graph()
    from torchwright.compiler.utils import get_ancestor_nodes

    nodes = get_ancestor_nodes({out})
    table = lower(out).realization_table

    with pytest.raises(UnresolvedRealizationError, match="incomplete"):
        table.check_complete(nodes)
    with pytest.raises(UnresolvedRealizationError, match="before resolution"):
        table.resolved_class(a)

    resolved = table.resolve_static(SchedulingPolicy())
    resolved.check_complete(nodes)  # no raise


# ---------------------------------------------------------------------------
# Compile integration: both paths record the table; they agree when the
# solve's routing freedom is pinned off.
# ---------------------------------------------------------------------------


def _chain_graph():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    l1 = Linear(x, torch.randn(4, 4) * 0.2, torch.randn(4) * 0.1, name="l1")
    l2 = Linear(l1, torch.randn(4, 4) * 0.2, torch.randn(4) * 0.1, name="l2")
    return l2, l1


def test_compile_records_resolved_table():
    from torchwright.compiler.forward.compile import forward_compile

    out, l1 = _chain_graph()
    net = forward_compile(d=64, d_head=8, output_node=out, verbose=False)
    table = net.realization_table
    # Default policy routes standalone Linears to the MLP bypass.
    assert table.resolved_class(l1) == MLP_BYPASS
    assert table.resolved_class(out) == MLP_BYPASS


def test_cost_summary_static_hand_computed():
    """d_head=8; a,b: Linear 4->3 (bypass 2*3=6 slots each; transport
    ceil(4/8)=1 head each); blk: 6 lanes; add width 3: reuse ceil(3/8)=1,
    copy 2; literal: no slots."""
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)

    cs = lowered.cost_summary(d_head=8)  # default policy: bypass
    assert cs.mlp_bypass_slots == 12
    assert cs.mlp_lanes == 6
    assert cs.heads_by_class == {}  # nothing attention-routed
    assert (cs.add_heads_if_all_reuse, cs.add_heads_if_all_copy) == (1, 2)
    assert (cs.attn_heads_min, cs.attn_heads_max) == (1, 2)

    cs_attn = lowered.cost_summary(d_head=8, policy=LEGACY_POLICY)
    assert cs_attn.mlp_bypass_slots == 0
    assert cs_attn.heads_by_class == {ATTN_TRANSPORT: 2}
    assert (cs_attn.attn_heads_min, cs_attn.attn_heads_max) == (3, 4)
    assert "bypass" in cs_attn.format_short()


def test_cost_summary_requires_resolved_table():
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    with pytest.raises(UnresolvedRealizationError, match="incomplete"):
        lowered.cost_summary(d_head=8, realization_table=lowered.realization_table)


def test_cost_summary_reconciles_with_solver_totals():
    """Gate B3: the pre-schedule summary's totals agree with the finished
    solve's accounting (flex pinned so routing is the static table)."""
    from torchwright.compiler.forward.compile import forward_compile

    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    cs = lowered.cost_summary(d_head=8)

    net = forward_compile(
        d=64,
        d_head=8,
        output_node=out,
        verbose=False,
        optimize=1,
        cpsat_flex_routing=False,
        require_solver=True,
    )
    stats = net.cpsat_solve_stats
    assert stats.total_mlp_bypass_slots == cs.mlp_bypass_slots
    assert cs.attn_heads_min <= stats.total_attn_heads <= cs.attn_heads_max


def test_eager_and_directed_tables_agree_when_flex_pinned():
    """With cpsat_flex_routing=False the solve uses the shared static
    routing, so the directed path's table must equal the eager path's —
    the one-option-set property, exercised end to end."""
    from torchwright.compiler.forward.compile import forward_compile

    out, l1 = _chain_graph()
    net0 = forward_compile(d=64, d_head=8, output_node=out, verbose=False)
    net1 = forward_compile(
        d=64,
        d_head=8,
        output_node=out,
        verbose=False,
        optimize=1,
        cpsat_flex_routing=False,
        require_solver=True,
    )
    t0 = {nid: e.resolved for nid, e in net0.realization_table.entries.items()}
    t1 = {nid: e.resolved for nid, e in net1.realization_table.entries.items()}
    assert t0 == t1
