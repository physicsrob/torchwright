"""Support-aware attention-transport head charge (spans_plan W6).

A standalone ``Linear`` routed through attention is emitted as one Δ=0
self-match head per ``d_head``-wide input chunk.  The charge used to read the
*declared* input width — ``ceil(d_input / d_head)`` — so a chunk whose weight
rows are all zero was charged, allocated, and emitted as a head whose O block
is zero (contributing exactly nothing).  On the production DOOM graph that
wasted 3,749 of 5,680 charged heads (measured 2026-07-09,
``torchwright_doom/scripts/count_standalone_extracts.py``), 708 of them on
standalone ``extract_derived`` Linears alone.

The charge now counts live chunks only (``realization.linear_attn_chunks``,
floor 1), and every accounting site — the CP-SAT model's ``heads_for``, the
eager scheduler, ``realization.class_heads``, the compile diagnostics — and
the emitter iterate that one list, so the budget and the emission cannot
desync.  Dense weights charge exactly the old formula, which is why the rest
of the suite (built on ``randn`` weights) does not move.
"""

import json

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    build_cpsat_model,
    build_graph_model,
    build_model_from_snapshot,
    heads_for,
)
from torchwright.compiler.forward.cpsat_snapshot import (
    SchedulingProblem,
    snapshot_from_graph_model,
)
from torchwright.compiler.forward.scheduling_policy import LEGACY_POLICY
from torchwright.compiler.graph_identity import graph_fingerprint
from torchwright.compiler.realization import (
    ATTN_TRANSPORT,
    class_heads,
    linear_attn_chunks,
    linear_attn_heads,
    live_weight_row_ranges,
)
from torchwright.graph import Concatenate, Linear, fresh_graph_session
from torchwright.graph.optimize import fuse_consecutive_linears
from torchwright.ops.inout_nodes import create_input

D_HEAD = 8


def _sparse_w(d_in=40, d_out=2, rows=((3, 0), (25, 1))):
    w = torch.zeros(d_in, d_out)
    for r, c in rows:
        w[r, c] = 0.5 + 0.1 * r
    return w


# ---------------------------------------------------------------------------
# The chunk rule (smallest layer)
# ---------------------------------------------------------------------------


def test_live_ranges_and_chunks():
    x = create_input("x", 40, value_range=(-1.0, 1.0))
    lin = Linear(x, _sparse_w(), name="sparse")  # live rows 3 and 25
    assert live_weight_row_ranges(lin) == ((3, 1), (25, 1))
    assert linear_attn_chunks(lin, D_HEAD) == (0, 3)
    assert linear_attn_heads(lin, D_HEAD) == 2


def test_contiguous_run_spanning_chunk_boundary():
    x = create_input("x", 40, value_range=(-1.0, 1.0))
    w = torch.zeros(40, 1)
    w[6:11] = 1.0  # rows 6..10 straddle chunks 0 and 1
    lin = Linear(x, w, name="run")
    assert live_weight_row_ranges(lin) == ((6, 5),)
    assert linear_attn_chunks(lin, D_HEAD) == (0, 1)


def test_dense_weights_charge_the_old_formula():
    torch.manual_seed(0)
    for d_in in (4, 8, 40, 43):
        x = create_input(f"x{d_in}", d_in, value_range=(-1.0, 1.0))
        lin = Linear(x, torch.randn(d_in, 3), name="dense")
        assert linear_attn_heads(lin, D_HEAD) == (d_in + D_HEAD - 1) // D_HEAD


def test_zero_support_floors_at_one_head():
    """A zero-weight Linear keeps chunk 0: h == 0 would drop the node from
    the CP-SAT attention cumulative and — for a flex Linear — its routing
    variable from the objective (the builder skips h == 0 nodes).
    """
    x = create_input("x", 40, value_range=(-1.0, 1.0))
    lin = Linear(x, torch.zeros(40, 2), name="zero")
    assert linear_attn_chunks(lin, D_HEAD) == (0,)
    assert linear_attn_heads(lin, D_HEAD) == 1


def test_bias_never_creates_support():
    """A bias needs no input read (it is written by the separate
    ``compute_bias`` MLP op), so it must not make a weight row live.
    """
    x = create_input("x", 40, value_range=(-1.0, 1.0))
    lin = Linear(x, torch.zeros(40, 2), torch.ones(2), name="bias_only")
    assert live_weight_row_ranges(lin) == ()
    assert linear_attn_heads(lin, D_HEAD) == 1  # the floor, not the bias


def test_fold_invalidates_the_support_cache():
    """The charge is memoized on the node; a fold that rewrites the weights
    must drop the cache or a pre-fusion query would freeze the pre-fold
    support — and the emitter would skip chunks that are live post-fold.
    """
    x = create_input("x", 40, value_range=(-1.0, 1.0))
    l1 = Linear(x, _sparse_w(), name="l1")  # support rows 3, 25
    torch.manual_seed(0)
    l2 = Linear(l1, torch.randn(2, 2), name="l2")  # dense on l1's 2 cols

    assert linear_attn_heads(l2, D_HEAD) == 1  # cache it pre-fusion
    assert fuse_consecutive_linears({l2}) == 1  # l1 absorbed into l2
    # l2 now reads x through l1's sparse rows: support = rows {3, 25}.
    assert live_weight_row_ranges(l2) == ((3, 1), (25, 1))
    assert linear_attn_heads(l2, D_HEAD) == 2


# ---------------------------------------------------------------------------
# Lockstep: every charge site and the emitter agree
# ---------------------------------------------------------------------------


def _three_kinds():
    """Sparse (2 live chunks of 5), dense (5), zero (floor 1) — each on its
    own input so the sibling merge cannot pool their support.
    """
    x1 = create_input("x1", 40, value_range=(-1.0, 1.0))
    x2 = create_input("x2", 40, value_range=(-1.0, 1.0))
    x3 = create_input("x3", 40, value_range=(-1.0, 1.0))
    torch.manual_seed(0)
    sparse = Linear(x1, _sparse_w(), name="sparse")
    dense = Linear(x2, torch.randn(40, 2) * 0.1, name="dense")
    zero = Linear(x3, torch.zeros(40, 1), name="zero")
    return sparse, dense, zero


def test_all_charge_sites_agree():
    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
    from torchwright.compiler.forward.scheduler import LayerScheduler

    sparse, dense, zero = _three_kinds()
    out = Concatenate([sparse, dense, zero])
    sched = LayerScheduler(GraphAnalyzer(out), 192, D_HEAD, None)
    for node, expect in ((sparse, 2), (dense, 5), (zero, 1)):
        assert heads_for(node, D_HEAD) == expect
        assert class_heads(node, ATTN_TRANSPORT, D_HEAD) == expect
        assert sched._heads_for_linear(node) == expect


def test_emitter_allocates_exactly_the_charged_heads_and_values_match():
    with fresh_graph_session():
        sparse, dense, zero = _three_kinds()
        out = Concatenate([sparse, dense, zero])
        net = forward_compile(
            d=192,
            d_head=D_HEAD,
            output_node=out,
            verbose=False,
            device="cpu",
            policy=LEGACY_POLICY,  # route everything through attention
            # This test measures the EMITTER's allocation against the charge;
            # the post-compile compaction (trim_unused_heads) would separately
            # remove the zero Linear's floor head (its O block is zero) and
            # reset used_heads — that mechanism has its own tests
            # (test_head_trim.py).
            trim_heads=False,
        )
        used = sum(layer.attn.attn.used_heads for layer in net.layers)
        assert used == 2 + 5 + 1  # was 5 + 5 + 5 under the width charge

        torch.manual_seed(1)
        vals = {
            "x1": torch.rand(3, 40) * 2 - 1,
            "x2": torch.rand(3, 40) * 2 - 1,
            "x3": torch.rand(3, 40) * 2 - 1,
        }
        got = net.compute(3, vals)
        for leaf in (sparse, dense, zero):
            torch.testing.assert_close(
                got[leaf].cpu(), leaf.compute(3, vals), rtol=0, atol=1e-5
            )


def test_zero_support_flex_linear_solves_and_matches():
    """The h == 0 edge case, end to end: with the floor, a zero-weight flex
    Linear stays in the CP-SAT model, solves under require_solver, and its
    (all-zero) value plus bias compiles correctly.
    """
    with fresh_graph_session():
        x = create_input("x", 40, value_range=(-1.0, 1.0))
        zero = Linear(x, torch.zeros(40, 2), torch.tensor([1.5, -0.5]), name="z")
        y = create_input("y", 4, value_range=(-1.0, 1.0))
        torch.manual_seed(0)
        other = Linear(y, torch.randn(4, 2) * 0.1, name="other")
        out = Concatenate([zero, other])

        net = forward_compile(
            d=192,
            d_head=D_HEAD,
            output_node=out,
            verbose=False,
            device="cpu",
            optimize=1,
            require_solver=True,
        )
        torch.manual_seed(1)
        vals = {"x": torch.rand(3, 40) * 2 - 1, "y": torch.rand(3, 4) * 2 - 1}
        got = net.compute(3, vals)
        torch.testing.assert_close(
            got[zero].cpu(), zero.compute(3, vals), rtol=0, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Snapshot: geometry-free support runs roundtrip; legacy records fall back
# ---------------------------------------------------------------------------

_CFG = {"d": 192, "d_head": D_HEAD, "d_hidden": 192, "max_layers": 12}


def _sparse_graph():
    sparse, dense, zero = _three_kinds()
    return Concatenate([sparse, dense, zero])


def test_snapshot_roundtrip_proto_identity_with_sparse_weights():
    """The stand-in nodes carry no weights; the persisted live-row runs must
    reproduce the identical charge, hence the identical proto.
    """
    with fresh_graph_session():
        out = _sparse_graph()
        gm = build_graph_model(out)
        live = str(build_cpsat_model(out, **_CFG).model.Proto())
        problem = SchedulingProblem.loads(snapshot_from_graph_model(gm).dumps())
        snap = str(build_model_from_snapshot(problem, **_CFG).model.Proto())
    assert live == snap


def test_legacy_snapshot_without_ranges_builds_with_dense_charge():
    """A pre-support-charge snapshot has no ``live_row_ranges``; the rebuild
    falls back to the dense-equivalent run (the old width charge) instead of
    crashing.  Such snapshots also fail the identity fingerprint gate, so
    this is determinism hygiene, not a supported measurement path.
    """
    with fresh_graph_session():
        out = _sparse_graph()
        gm = build_graph_model(out)
        live = str(build_cpsat_model(out, **_CFG).model.Proto())
        d = json.loads(snapshot_from_graph_model(gm).dumps())
        recs = d["nodes"] if isinstance(d["nodes"], list) else list(d["nodes"].values())
        for rec in recs:
            rec.pop("live_row_ranges", None)
        legacy = SchedulingProblem.loads(json.dumps(d))
        legacy_built = build_model_from_snapshot(legacy, **_CFG)
    # Builds — and charges the old dense heads, so the proto differs.
    assert str(legacy_built.model.Proto()) != live


# ---------------------------------------------------------------------------
# Cache key: sparsity now schedules, so it must fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_sees_sparsity():
    """Two graphs with identical topology but different weight support now
    schedule differently; replaying a cached schedule across them would
    under-book heads in one direction.  The fingerprint must separate them
    — and must stay stable across rebuilds of the same graph.
    """
    kw = {
        "d": 192,
        "d_head": D_HEAD,
        "d_hidden": 192,
        "flex_routing": True,
        "cancel_slack": 2,
        "policy": None,
    }

    def build(sparse: bool):
        x = create_input("x", 40, value_range=(-1.0, 1.0))
        w = _sparse_w() if sparse else torch.ones(40, 2)
        return Concatenate([Linear(x, w, name="l")])

    with fresh_graph_session():
        fp_sparse = graph_fingerprint(build(sparse=True), **kw)
    with fresh_graph_session():
        fp_sparse2 = graph_fingerprint(build(sparse=True), **kw)
    with fresh_graph_session():
        fp_dense = graph_fingerprint(build(sparse=False), **kw)

    assert fp_sparse == fp_sparse2  # deterministic across rebuilds
    assert fp_sparse != fp_dense  # sparsity participates
