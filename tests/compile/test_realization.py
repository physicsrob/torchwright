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
    first_hidden_slot,
    fits_mlp_bypass,
    has_flex_choice,
    is_conditional,
    static_flex_class,
    usable_hidden_slots,
)
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import Add, Linear
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import Concatenate, LiteralValue
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear


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
    # The lowered table is keyed by the compiler-private copy's node ids
    # (the scheduler consumes it); translate source nodes via copy_of.
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    table = lowered.realization_table
    cid = lambda n: lowered.copy_of(n).node_id

    ea = table.entries[cid(a)]
    assert ea.resolved is None and not ea.conditional
    assert ea.candidates == (ATTN_TRANSPORT, MLP_BYPASS)

    eadd = table.entries[cid(add)]
    assert eadd.conditional and eadd.resolved is None

    assert table.entries[cid(blk)].resolved == MLP_COMPOSITE
    assert table.entries[cid(lit)].resolved == MLP_LITERAL
    # Non-schedulable nodes get no entry.
    assert cid(out) not in table.entries


def test_resolve_static_both_policies():
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    table = lowered.realization_table
    nodes = get_ancestor_nodes({lowered.output_node})
    ca, cb, cadd, cblk = (lowered.copy_of(n) for n in (a, b, add, blk))

    # a and b are 4->3 Linears: 2*3 = 6 bypass slots, well inside 64.
    bypass = table.resolve_static(nodes, SchedulingPolicy(), 64)  # "never"
    assert bypass.entries[ca.node_id].resolved == MLP_BYPASS
    assert bypass.entries[cb.node_id].resolved == MLP_BYPASS
    assert not bypass.is_attention_routed(ca)

    attn = table.resolve_static(nodes, LEGACY_POLICY, 64)  # "always"
    assert attn.entries[ca.node_id].resolved == ATTN_TRANSPORT
    assert attn.is_attention_routed(ca)

    # Conditional and single-candidate entries are untouched by resolution.
    for t in (bypass, attn):
        assert t.entries[cadd.node_id].conditional
        assert t.entries[cblk.node_id].resolved == MLP_COMPOSITE

    # The unresolved source table is not mutated (resolvers return new).
    assert table.entries[ca.node_id].resolved is None


def test_resolve_from_assignment():
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    table = lowered.realization_table
    cid = lambda n: lowered.copy_of(n).node_id

    routing = {
        cid(a): "attn",
        cid(b): "mlp",
        cid(add): "attn",
        cid(blk): "mlp",
        cid(lit): "mlp",
    }
    resolved = table.resolve_from_assignment(routing)
    assert resolved.entries[cid(a)].resolved == ATTN_TRANSPORT
    assert resolved.entries[cid(b)].resolved == MLP_BYPASS
    assert resolved.entries[cid(add)].conditional
    assert resolved.entries[cid(blk)].resolved == MLP_COMPOSITE


def test_resolve_from_assignment_rejects_contradiction():
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    # The solve routing an FFN to attention contradicts its only class.
    with pytest.raises(UnresolvedRealizationError, match="only realization"):
        lowered.realization_table.resolve_from_assignment(
            {lowered.copy_of(blk).node_id: "attn"}
        )


def test_check_complete_and_unresolved_read():
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    nodes = get_ancestor_nodes({lowered.output_node})
    table = lowered.realization_table

    with pytest.raises(UnresolvedRealizationError, match="incomplete"):
        table.check_complete(nodes)
    with pytest.raises(UnresolvedRealizationError, match="before resolution"):
        table.resolved_class(lowered.copy_of(a))

    resolved = table.resolve_static(nodes, SchedulingPolicy(), 64)
    resolved.check_complete(nodes)  # no raise


def test_resolve_static_rejects_nodes_missing_the_flex_entry():
    """The capacity check reads widths off the nodes; a caller that passes a
    node set not covering the table's unresolved entries gets a named error,
    not a KeyError."""
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    with pytest.raises(UnresolvedRealizationError, match="absent from the nodes"):
        lowered.realization_table.resolve_static([], SchedulingPolicy(), 64)


# ---------------------------------------------------------------------------
# Compile integration: both paths record the table; they agree when the
# solve's routing freedom is pinned off.
# ---------------------------------------------------------------------------


def _chain_graph():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    l1 = Linear(x, torch.randn(4, 4) * 0.2, torch.randn(4) * 0.1, name="l1")
    l2 = Linear(l1, torch.randn(4, 4) * 0.2, torch.randn(4) * 0.1, name="l2")
    # l1 is also an output leaf: two consumers, so the lowering boundary's
    # fusion declines the Linear->Linear fold and both nodes get entries.
    out = Concatenate([l2, l1])
    return out, l1, l2


def test_compile_records_resolved_table():
    from torchwright.compiler.forward.compile import forward_compile

    out, l1, l2 = _chain_graph()
    net = forward_compile(d=64, d_head=8, output_node=out, verbose=False)
    table = net.realization_table
    # Default policy routes standalone Linears to the MLP bypass.
    assert table.resolved_class(l1) == MLP_BYPASS
    assert table.resolved_class(l2) == MLP_BYPASS


def test_cost_summary_static_hand_computed():
    """d_head=8; a,b: Linear 4->3 (bypass 2*3=6 slots each; transport
    ceil(4/8)=1 head each); blk: 6 lanes; add width 3: reuse ceil(3/8)=1,
    copy 2; literal: no slots."""
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)

    cs = lowered.cost_summary(d_head=8, usable_slots=64)  # default policy: bypass
    assert cs.mlp_bypass_slots == 12
    assert cs.mlp_lanes == 6
    assert cs.heads_by_class == {}  # nothing attention-routed
    assert (cs.add_heads_if_all_reuse, cs.add_heads_if_all_copy) == (1, 2)
    assert (cs.attn_heads_min, cs.attn_heads_max) == (1, 2)

    cs_attn = lowered.cost_summary(d_head=8, policy=LEGACY_POLICY, usable_slots=64)
    assert cs_attn.mlp_bypass_slots == 0
    assert cs_attn.heads_by_class == {ATTN_TRANSPORT: 2}
    assert (cs_attn.attn_heads_min, cs_attn.attn_heads_max) == (3, 4)
    assert "bypass" in cs_attn.format_short()


def test_cost_summary_bypass_demand_beyond_capacity_routes_to_attention():
    """The bypass-slot column is not a policy readout: a Linear whose
    ``2 * d_output`` exceeds the layer's usable pool has no MLP realization,
    so the summary must charge it heads even under the bypass policy."""
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)

    # a and b each want 6 bypass slots.  At usable_slots=5 neither fits.
    cs = lowered.cost_summary(d_head=8, usable_slots=5)
    assert cs.mlp_bypass_slots == 0
    assert cs.heads_by_class == {ATTN_TRANSPORT: 2}


def test_cost_summary_requires_usable_slots_to_resolve():
    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    with pytest.raises(ValueError, match="usable_slots"):
        lowered.cost_summary(d_head=8)


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
    cs = lowered.cost_summary(d_head=8, usable_slots=64)

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

    out, _, _ = _chain_graph()
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


# ---------------------------------------------------------------------------
# Capacity-aware static routing
#
# The MLP bypass spends two hidden slots per output column.  A Linear whose
# ``2 * d_output`` exceeds a layer's whole usable hidden pool cannot be placed
# in *any* layer, so the static rule must route it to attention transport even
# under the bypass policy.  Before this, such a node was skipped by the MLP
# placer every layer forever and the eager walk raised ``No progress``, while
# the pinned-routing CP-SAT model pinned it into a cumulative it could not
# satisfy and returned INFEASIBLE.
#
# ``usable_hidden_slots(d_hidden, bias)`` is the pool: ``bias=False`` reserves
# hidden slot 0 for the constant lane, so the boundary moves down by one.
# ---------------------------------------------------------------------------


def _linear_of_width(d_output: int) -> Linear:
    x = create_input("wide_x", 4, value_range=(-1.0, 1.0))
    return Linear(x, torch.randn(4, d_output) * 0.2, name="wide")


def _wide_graph(d_output: int):
    """``x -> wide -> tap``, with ``wide`` also an output leaf.

    The second consumer is load-bearing: with ``tap`` as its only consumer,
    ``lower()``'s ``_fuse_linear_into_linear`` absorbs ``wide`` into ``tap``'s
    weights and the node under test stops existing.  Mirrors ``_chain_graph``.
    """
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    torch.manual_seed(0)
    wide = Linear(x, torch.randn(4, d_output) * 0.2, name="wide")
    tap = Linear(wide, torch.randn(d_output, 2) * 0.2, name="tap")
    return Concatenate([tap, wide]), wide


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("policy", [SchedulingPolicy(), LEGACY_POLICY])
def test_static_flex_class_capacity_boundary(bias, policy):
    """d_hidden=64: the widest placeable bypass Linear has 2*d_output equal to
    the usable pool.  One column wider and only attention transport remains,
    whatever ``local_in_attention`` says."""
    usable = usable_hidden_slots(64, bias)
    assert usable == (64 if bias else 63)

    widest = usable // 2  # 32 under bias, 31 without (2*32 = 64 > 63)
    fits = _linear_of_width(widest)
    too_wide = _linear_of_width(widest + 1)

    assert fits_mlp_bypass(fits, usable)
    assert not fits_mlp_bypass(too_wide, usable)

    # Under the bypass policy the boundary is where the class flips; under
    # the attention policy both sides are attention anyway.
    expected_fitting = (
        MLP_BYPASS if policy.local_in_attention == "never" else ATTN_TRANSPORT
    )
    assert static_flex_class(fits, policy, usable) == expected_fitting
    assert static_flex_class(too_wide, policy, usable) == ATTN_TRANSPORT


def test_first_hidden_slot_is_the_only_bias_arithmetic():
    """The scheduler's packing base and the routing rule's pool are two reads
    of one fact: bias=False reserves slot 0 for the constant lane."""
    assert first_hidden_slot(True) == 0
    assert first_hidden_slot(False) == 1
    for d_hidden in (2, 7, 64):
        for bias in (True, False):
            assert usable_hidden_slots(d_hidden, bias) == d_hidden - first_hidden_slot(
                bias
            )


@pytest.mark.parametrize("bias", [True, False])
def test_resolve_static_routes_unplaceable_bypass_to_attention(bias):
    """The resolver, not just the rule: an over-wide Linear comes back
    attention-routed even though the default policy asks for the bypass, while
    its narrow consumer still takes the bypass."""
    usable = usable_hidden_slots(64, bias)
    out, wide = _wide_graph(usable // 2 + 1)

    lowered = lower(out)
    nodes = get_ancestor_nodes({lowered.output_node})
    table = lowered.realization_table.resolve_static(nodes, SchedulingPolicy(), usable)

    resolved = {n.name: table.resolved_class(n) for n in nodes if isinstance(n, Linear)}
    assert resolved == {"wide": ATTN_TRANSPORT, "tap": MLP_BYPASS}


@pytest.mark.parametrize("bias", [True, False])
def test_cpsat_routing_agrees_with_resolve_static(bias):
    """The drift tripwire.  ``routing()`` (CP-SAT's pinned path) and
    ``resolve_static`` (the eager path) must classify the same node the same
    way; when they disagree, ``resolve_from_assignment`` raises
    ``UnresolvedRealizationError`` on a graph that used to compile."""
    from torchwright.compiler.forward.cpsat_scheduler import (
        build_graph_model,
        routing,
    )
    from torchwright.compiler.realization import CLASS_SUBLAYER

    usable = usable_hidden_slots(64, bias)
    out, _ = _wide_graph(usable // 2 + 1)

    lowered = lower(out)
    nodes = get_ancestor_nodes({lowered.output_node})
    policy = SchedulingPolicy()
    table = lowered.realization_table.resolve_static(nodes, policy, usable)

    gm = build_graph_model(lowered.output_node)
    for n in gm.schedulable:
        if table.entries[n.node_id].conditional:
            continue  # Add: schedule-state conditional, both classes attention
        assert routing(n, gm, policy, usable) == CLASS_SUBLAYER[table.resolved_class(n)]


def test_routing_without_capacity_raises_rather_than_guessing():
    """``critical_path_layers`` passes no geometry; under flex_routing=True it
    never routes a Linear.  If a caller reaches this path anyway, it must say
    so rather than pick a sublayer that may not hold the node."""
    from torchwright.compiler.forward.cpsat_scheduler import (
        build_graph_model,
        routing,
    )

    out, a, b, add, blk, lit = _test_graph()
    lowered = lower(out)
    gm = build_graph_model(lowered.output_node)
    ca = lowered.copy_of(a)

    with pytest.raises(ValueError, match="usable hidden-slot count"):
        routing(ca, gm, SchedulingPolicy(), None)

    # Non-flex nodes never consult it.
    cblk = lowered.copy_of(blk)
    assert routing(cblk, gm, SchedulingPolicy(), None) == "mlp"


@pytest.mark.parametrize("bias", [True, False])
def test_wide_bypass_linear_compiles_and_matches_compute(bias):
    """End to end at optimize=0: the rescued Linear runs as attention
    transport and computes what ``node.compute`` computes."""
    from torchwright.compiler.export import compile_headless
    from torchwright.debug.probe import probe_compiled

    d_hidden = 64
    usable = usable_hidden_slots(d_hidden, bias)
    out, _ = _wide_graph(usable // 2 + 1)  # one column past the bypass boundary

    compiled = compile_headless(out, d=64, d_head=8, d_hidden=d_hidden, bias=bias)
    torch.manual_seed(1)
    report = probe_compiled(
        compiled, out, {"x": torch.rand(3, 4) * 2 - 1}, n_pos=3, atol=1e-4
    )
    assert report.first_divergent is None, report.format_short()


@pytest.mark.parametrize("bias", [True, False])
def test_wide_bypass_linear_solves_under_pinned_cpsat_routing(bias):
    """``cpsat_flex_routing=False`` pins every Linear via ``routing()``.  With
    the capacity rule absent it pinned the wide Linear into the MLP and the
    hidden-slot cumulative made the model INFEASIBLE; ``require_solver=True``
    turns the silent heuristic fallback into a raise, so this test sees it."""
    from torchwright.compiler.forward.compile import forward_compile

    d_hidden = 64
    out, wide = _wide_graph(usable_hidden_slots(d_hidden, bias) // 2 + 1)

    net = forward_compile(
        d=64,
        d_head=8,
        output_node=out,
        d_hidden=d_hidden,
        bias=bias,
        verbose=False,
        optimize=1,
        cpsat_flex_routing=False,
        require_solver=True,
    )
    assert net.realization_table.resolved_class(wide) == ATTN_TRANSPORT


def test_eager_scheduler_places_the_wide_linear_in_attention():
    """The eager walk is what deadlocked, and it is what CP-SAT's warm start
    runs to produce its hint: while the wide Linear had no reachable
    realization, the MLP placer skipped it every layer and ``schedule_layer``
    raised ``No progress`` with an empty hint behind it.  Now it comes out as
    a ``compute_linear`` attention op."""
    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
    from torchwright.compiler.forward.residual_map import ResidualStreamMap
    from torchwright.compiler.forward.scheduler import LayerScheduler

    d, d_hidden = 64, 64
    out, _ = _wide_graph(33)  # 2*33 = 66 > 64

    graph = GraphAnalyzer(lower(out).output_node)
    rmap = ResidualStreamMap(d)
    computed = set()
    for n in graph.get_all_nodes():
        if graph.is_input_node(n):
            rmap.allocate(n)
            computed.add(n)

    sched = LayerScheduler(graph, d, 8, None, d_hidden=d_hidden)
    assert sched.usable_hidden_slots == 64

    # schedule_layer raises `No progress` the moment a ready node can be
    # placed nowhere; an empty layer means the walk has drained the graph.
    placed_attn = []
    for _ in range(20):
        attn_ops, mlp_ops, _ = sched.schedule_layer(rmap, computed)
        if not attn_ops and not mlp_ops:
            break
        placed_attn += [
            op.node.name for op in attn_ops if op.op_type == "compute_linear"
        ]
    else:
        pytest.fail("eager walk did not drain the wide graph in 20 layers")

    assert "wide" in placed_attn
    assert "tap" not in placed_attn  # still bypass-routed: 2*2 = 4 slots
