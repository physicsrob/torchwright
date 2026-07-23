"""``concat(Linear(x, W_0), ..., Linear(x, W_n))`` is one ``Linear(x, hstack(W_i))``.

``Concatenate`` never occupies the residual stream — it is resolved at read time
into its leaves' columns — so ``n`` one-column Linears reading the same input
*are* the single ``n``-column Linear holding the same weights, laid side by side.

``fuse_consecutive_linears`` now merges them (``_merge_sibling_linear_leaves``).
It previously did not: its only Concatenate fold,
``_fold_through_concatenate(l, c, ...)``, absorbs Linear leaves into a
*downstream Linear's* row blocks and takes that Linear as an argument, so it
cannot fire when the concat feeds an ``FFN`` or an ``Attn``.  The sibling merge
needs no downstream node at all, is exact, and is weight-neutral by construction
(``n`` matrices of ``d_in x 1`` hold the same entries as one ``d_in x n``), so it
needs no parameter-count gate.

Why it pays: ``heads_for(Linear) = ceil(d_input / d_head)``, so a Linear's
*output* width is free under attention routing and the split form paid ``n`` times
the attention heads of the merge for identical values.  MLP-bypass slots
(``2 * d_output``) are the same either way, so whether the split cost a *layer*
depended on which resource bound.  ``span_finding.md`` at the torchdoom root has
the measured table; do not generalize the layer cost beyond the regime each test
below states.

Unconditionally, the split left ``n`` redundant nodes in the graph that ``lower()``
handed to the scheduler.  ``torchwright_doom`` hits this with three families of
``COLUMN_COUNT``-many one-column ``extract_derived_*`` Linears, concatenated and
fed to ``Attn``s — 480 nodes at the production render scale
(``COLUMN_COUNT=160``), 180 at the bare-import default (``COLUMN_COUNT=60``).

The second half of this file is the precondition matrix: each case is a graph the
fold must *decline*, or a shape it must handle exactly.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.forward.cpsat_scheduler import build_graph_model, heads_for
from torchwright.graph import Attn, Concatenate, Linear, fresh_graph_session
from torchwright.graph.affine_rules import _apply_semantic_override
from torchwright.graph.asserts import assert_in_range, attach_assert
from torchwright.graph.optimize import fuse_consecutive_linears
from torchwright.graph.scheduling_hints import (
    add_scheduling_dependency,
    sequential_scope,
)
from torchwright.graph.value_type import NodeValueType
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear import concat
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

D_EMBED = 96
D_HEAD = 16
HEADS_PER_LINEAR = -(-D_EMBED // D_HEAD)  # ceil(96 / 16) == 6

# 6 lanes x 6 heads = 36, over the d=512 per-layer budget of 512 // 16 == 32.
N = 6
D = 512


def _weight_matrix(n: int) -> torch.Tensor:
    """A ``(D_EMBED, n)`` **dense** weight matrix.

    Deliberately dense, not the identity block ``extract_derived`` actually
    builds.  ``heads_for`` charges ``ceil(d_input / d_head)`` on the *declared*
    input width rather than on the weight matrix's nonzero row support, which is
    a **separate, independent defect** (see ``span_finding.md``, "A second
    defect").  With an input-sparse weight the two defects are confounded: a
    sparse-aware head charge alone would drop each lane from 6 heads to 1 and
    erase the layer delta below without anyone having added the sibling fold.

    Dense weights make the ``ceil(d_input/d_head)`` charge *legitimate* for every
    lane, so a sparse-aware head fix alone cannot close the gap.
    """
    torch.manual_seed(0)
    return torch.randn(D_EMBED, n)


def _lanes(x, w: torch.Tensor, tag: str = "lane") -> list:
    """One width-1 ``Linear`` per column of ``w``."""
    return [
        Linear(x, w[:, i : i + 1].contiguous(), name=f"{tag}_{i:03d}")
        for i in range(w.shape[1])
    ]


def _consumer(node, n: int):
    """An ``FFN`` consumer.

    (DOOM's real consumer is an ``Attn`` — see ``_build_split_attn``,
    which isolates the sibling merge without the FFN's confounding
    ``Linear -> FFN`` gate fold.)

    It must not be a ``Linear``: then ``_fold_through_concatenate`` also fires
    and there is nothing to attribute to the sibling merge.  A hand-built
    ``Linear -> ReLU -> Linear`` is rejected by ``lower()``'s vocabulary check,
    so build the FFN natively.
    """
    return linear_relu_linear(
        node,
        torch.eye(n),
        torch.zeros(n),
        torch.ones((n, 1)),
        torch.zeros(1),
        name="consumer",
    )


def _build_split(n: int):
    """``n`` width-1 Linears joined by a free ``concat``."""
    x = create_input("x", D_EMBED)
    return _consumer(concat(_lanes(x, _weight_matrix(n))), n)


def _build_merged(n: int):
    """One width-``n`` Linear with the identical weight matrix."""
    x = create_input("x", D_EMBED)
    return _consumer(Linear(x, _weight_matrix(n), name="span"), n)


def _attn_over(x, value_in, n: int):
    """An ``Attn`` reading ``value_in`` as V — the shape DOOM's lanes feed."""
    torch.manual_seed(0)
    return Attn(
        query_in=x,
        key_in=x,
        value_in=value_in,
        query_matrix=0.1 * torch.randn(D_EMBED, 16),
        key_matrix=0.1 * torch.randn(D_EMBED, 16),
        value_matrix=0.1 * torch.randn(n, 16),
        output_matrix=0.1 * torch.randn(16, D_EMBED),
    )


def _build_split_attn(n: int):
    x = create_input("x", D_EMBED, value_range=(-1.0, 1.0))
    return _attn_over(x, concat(_lanes(x, _weight_matrix(n))), n)


def _build_merged_attn(n: int):
    x = create_input("x", D_EMBED, value_range=(-1.0, 1.0))
    return _attn_over(x, Linear(x, _weight_matrix(n), name="span"), n)


def _census(*outs) -> dict:
    """Count nodes reachable from ``outs`` by class name."""
    kinds: dict
    seen, kinds = set(), {}

    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        kinds[type(node).__name__] = kinds.get(type(node).__name__, 0) + 1
        for inp in node.inputs:
            visit(inp)

    for out in outs:
        visit(out)
    return kinds


def _fused_census(build) -> dict:
    with fresh_graph_session():
        out = build(N)
        fuse_consecutive_linears({out})
        return _census(out)


def _compiled_layers(build) -> int:
    with fresh_graph_session():
        return compile_headless(build(N), d=D, d_head=D_HEAD, verbose=False).n_layers


def _probe(width: int = D_EMBED) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(1, width)


# --- The defect, and its consequences ------------------------------------


def test_split_and_merged_are_the_same_function():
    """Both hold the same weight matrix, so merging the lanes is exact."""
    probe = _probe()
    with fresh_graph_session():
        split_out = compile_headless(
            _build_split(N), d=D, d_head=D_HEAD, verbose=False
        )(probe)
    with fresh_graph_session():
        merged_out = compile_headless(
            _build_merged(N), d=D, d_head=D_HEAD, verbose=False
        )(probe)
    torch.testing.assert_close(split_out, merged_out, rtol=0, atol=1e-5)


@pytest.mark.parametrize("n", [1, 6, 32])
def test_a_linears_head_demand_ignores_its_output_width(n):
    """Why the merge pays: ``heads_for`` reads ``d_input`` only.

    A Linear produces one output column or ``n`` for the same heads — so
    splitting one on its output columns pays ``n`` times for the same read.
    """
    with fresh_graph_session():
        gm = build_graph_model(_build_merged(n))
        (span,) = [node for node in gm.schedulable if isinstance(node, Linear)]
        assert heads_for(span, D_HEAD) == HEADS_PER_LINEAR


def test_sibling_linears_fuse_like_one_wide_linear():
    """Fusion renders the two graphs identical — they compute the same values."""
    assert _fused_census(_build_split) == _fused_census(_build_merged)


def test_sibling_linears_cost_no_extra_compiled_layer():
    """The consequence, *in this regime*: the split no longer compiles larger.

    Scope carefully.  The extra layer was never universal — it depends on which
    resource binds.  MLP-bypass slot demand is ``2 * d_output`` summed either
    way, so the two forms are identical on that route.  Measured with an ``Attn``
    consumer and ``n=160`` lanes: with hidden slots plentiful both compiled to 2
    layers even before the fold; with hidden slots scarce the merged form escapes
    to a 6-head attention route and the unmerged split (960 heads) could not.

    See ``span_finding.md`` at the torchdoom root, "What it costs".
    """
    assert _compiled_layers(_build_split) == _compiled_layers(_build_merged)


def test_sibling_linears_fuse_under_an_attn_consumer():
    """DOOM's shape: the lanes feed an ``Attn``, not an ``FFN``.

    Asserted separately because the FFN case is confounded — there the merged
    form also collapses via the ``Linear -> FFN`` gate fold, which has nothing to
    do with sibling merging.  Under an ``Attn`` there is no such fold, so this
    isolates the new rewrite: the split graph presents one ``Linear``, exactly as
    the merged graph does.
    """
    with fresh_graph_session():
        split = _build_split_attn(N)
        fuse_consecutive_linears({split})
        split_linears = _census(split).get("Linear", 0)
    with fresh_graph_session():
        merged = _build_merged_attn(N)
        fuse_consecutive_linears({merged})
        merged_linears = _census(merged).get("Linear", 0)
    assert split_linears == merged_linears == 1


def test_merged_leaves_wider_than_the_mlp_still_compile():
    """The merge is the only fold that grows a node's ``d_output``.

    That's what lets it turn ``n`` individually-placeable Linears into
    one Linear with no MLP-bypass realization: the bypass costs
    ``2 * d_output`` hidden slots, and ``2 * 40 = 80`` exceeds this
    ``d_hidden=64`` layer's whole pool.

    Static routing is capacity-aware (``realization.static_flex_class``), so the
    merged Linear goes to attention transport instead of being skipped by the
    MLP placer in every layer forever.  Before that, this graph raised
    ``No progress`` while the same graph with the merge *declined* — by
    attaching a debug assert to a leaf — compiled fine, which made a debug
    annotation decide schedulability.
    """
    n, d_hidden = 40, 64
    assert 2 * n > d_hidden

    with fresh_graph_session():
        merged = compile_headless(
            _build_split_attn(n), d=D, d_head=D_HEAD, d_hidden=d_hidden, verbose=False
        )
    with fresh_graph_session():
        x = create_input("x", D_EMBED, value_range=(-1.0, 1.0))
        lanes = _lanes(x, _weight_matrix(n))
        for lane in lanes:
            assert_in_range(lane, -1e9, 1e9)  # every leaf checked -> no merge
        declined = compile_headless(
            _attn_over(x, concat(lanes), n),
            d=D,
            d_head=D_HEAD,
            d_hidden=d_hidden,
            verbose=False,
        )

    # Both compile.  The merged form is no deeper — it was the *unschedulable*
    # one, not the expensive one.
    assert merged.n_layers <= declined.n_layers


def test_merge_is_exact_on_the_graph():
    """The fold preserves the concat's value exactly, not just to tolerance."""
    probe = _probe()
    with fresh_graph_session():
        out = _build_split_attn(N)
        before = out.compute(1, {"x": probe})
        assert fuse_consecutive_linears({out}) > 0
        after = out.compute(1, {"x": probe})
    torch.testing.assert_close(before, after, rtol=0, atol=1e-5)


def test_merge_is_parameter_neutral():
    """``n`` matrices of ``d_in x 1`` hold the entries of one ``d_in x n``."""
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        before = sum(lane.num_params() for lane in lanes)
        out = concat(lanes)
        fuse_consecutive_linears({out})
        (span,) = [n for n in out.inputs if isinstance(n, Linear)]
    assert span.num_params() == before


def test_merged_split_compiles_under_compiler_verify(monkeypatch):
    """Exercise I1/I4 (column liveness, allocator consistency) on the new node."""
    monkeypatch.setenv("TW_COMPILER_VERIFY", "1")
    probe = _probe()
    with fresh_graph_session():
        split_out = compile_headless(
            _build_split_attn(N), d=D, d_head=D_HEAD, verbose=False
        )(probe)
    with fresh_graph_session():
        merged_out = compile_headless(
            _build_merged_attn(N), d=D, d_head=D_HEAD, verbose=False
        )(probe)
    torch.testing.assert_close(split_out, merged_out, rtol=0, atol=1e-4)


# --- The precondition matrix ---------------------------------------------
#
# Each case below builds a bare ``concat`` as the output node: with no
# consumer at all, only the sibling merge can fire, so a Linear census reads
# straight off the fold.


def _merge_only(*outs) -> int:
    return fuse_consecutive_linears(set(outs))


def test_duplicate_leaf_merges_and_duplicates_its_column():
    """``concat([l, l])`` merges, duplicating l's column rather than coalescing.

    ``_consumer_map`` appends once per *occurrence*, so ``len(consumers[l]) == 2``
    here while the distinct-consumer count is 1.  A precondition-3 check written
    as ``len(consumers[leaf]) == 1`` would wrongly decline this.
    """
    probe = _probe()
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        (lane,) = _lanes(x, _weight_matrix(1))
        out = Concatenate([lane, lane])
        before = out.compute(1, {"x": probe})
        assert _merge_only(out) == 1
        after = out.compute(1, {"x": probe})

    assert _census(out).get("Linear", 0) == 1
    assert len(out) == 2
    (span,) = [n for n in out.inputs if isinstance(n, Linear)]
    assert span.output_matrix.shape == (D_EMBED, 2)
    # The duplicated column is duplicated, not summed away.
    torch.testing.assert_close(span.output_matrix[:, 0], span.output_matrix[:, 1])
    torch.testing.assert_close(before, after, rtol=0, atol=1e-5)


def test_checked_leaf_declines_merge():
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        assert_in_range(lanes[1], -1e9, 1e9)
        out = concat(lanes)
        _merge_only(out)
    # lanes[1] blocks; the runs either side of it still merge.
    assert _census(out).get("Linear", 0) == 3
    assert lanes[1] in out.inputs


def test_claimed_type_leaf_declines_merge():
    """A range claim carries no predicate, so ``_blocks_absorb`` misses it.

    Set directly: no public helper attaches a claim without also attaching a
    check, and the point of this test is the claim in isolation.
    """
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        lanes[1].claimed_type = NodeValueType.bounded(-1e9, 1e9)
        out = concat(lanes)
        _merge_only(out)
    assert _census(out).get("Linear", 0) == 3
    assert lanes[1] in out.inputs


def test_integer_claim_leaf_declines_merge():
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        lanes[1].integer_claim = True
        out = concat(lanes)
        _merge_only(out)
    assert _census(out).get("Linear", 0) == 3
    assert lanes[1] in out.inputs


def test_semantic_override_leaf_declines_merge():
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        _apply_semantic_override(lanes[1], lanes[1]._affine_bound)
        out = concat(lanes)
        _merge_only(out)
    assert _census(out).get("Linear", 0) == 3
    assert lanes[1] in out.inputs


def test_attached_assert_leaf_declines_merge():
    """The public path — a predicate plus a claim — declines too."""
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        attach_assert(
            lanes[1],
            lambda v: torch.ones_like(v[..., 0], dtype=torch.bool),
            claimed_type=NodeValueType.bounded(-1e9, 1e9),
            integer_claim=True,
        )
        out = concat(lanes)
        _merge_only(out)
    assert _census(out).get("Linear", 0) == 3


def test_non_contiguous_leaves_merge_only_maximal_runs():
    """A leaf on a different input breaks the run; each side merges separately."""
    probe_x, probe_y = _probe(), _probe()
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        y = create_input("y", D_EMBED)
        lx = _lanes(x, _weight_matrix(4), tag="lx")
        (ly,) = _lanes(y, _weight_matrix(1), tag="ly")
        out = Concatenate([lx[0], lx[1], ly, lx[2], lx[3]])
        before = out.compute(1, {"x": probe_x, "y": probe_y})
        assert _merge_only(out) == 2
        after = out.compute(1, {"x": probe_x, "y": probe_y})

    # Two merged runs plus the untouched y-lane.
    assert _census(out).get("Linear", 0) == 3
    assert len(out.inputs) == 3
    assert out.inputs[1] is ly
    assert len(out) == 5
    torch.testing.assert_close(before, after, rtol=0, atol=1e-5)


def test_leaf_occurring_outside_its_run_declines_merge():
    """``concat([l0, l1, ly, l0])`` must not merge ``[l0, l1]``.

    Widening ``l0`` in place would leave the trailing slot reading the widened
    value.  The run's occurrence count must equal the leaf's total count.
    """
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        y = create_input("y", D_EMBED)
        lx = _lanes(x, _weight_matrix(2), tag="lx")
        (ly,) = _lanes(y, _weight_matrix(1), tag="ly")
        out = Concatenate([lx[0], lx[1], ly, lx[0]])
        assert _merge_only(out) == 0

    assert _census(out).get("Linear", 0) == 3
    assert len(out) == 4


def test_multi_consumer_concat_still_merges():
    """DOOM's ``x_oh`` case: the concat feeds two ``Attn``s.

    The leaves are still sole-consumer, so the merge is legal — only
    the *leaves* are gated.
    """
    with fresh_graph_session():
        x = create_input("x", D_EMBED, value_range=(-1.0, 1.0))
        c = concat(_lanes(x, _weight_matrix(N)))
        a1, a2 = _attn_over(x, c, N), _attn_over(x, c, N)
        assert _merge_only(a1, a2) > 0
        assert _census(a1, a2).get("Linear", 0) == 1


def test_multi_consumer_leaf_declines_merge():
    """A leaf read from outside the concat cannot be merged away.

    Its reader has no way to address a sub-range of the merged node.
    """
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        escapee = Linear(lanes[1], torch.ones(1, 1), name="escapee")
        out = concat(lanes)
        _merge_only(out, escapee)
    # lanes[0] (a run of one) and lanes[1] (two consumers) survive; lanes[2:]
    # merge.  The escapee is not reachable from `out`, hence the second census.
    assert _census(out).get("Linear", 0) == 3
    assert _census(out, escapee).get("Linear", 0) == 4
    assert lanes[1] in out.inputs
    assert escapee.inputs[0] is lanes[1]
    assert len(lanes[1]) == 1


def test_leaf_carrying_a_scheduling_hint_declines_merge():
    """A leaf that waits on another node keeps its ordering constraint.

    The merge orphans every leaf but one, and the survivor keeps only its own
    ``scheduling_predecessors``.  Merging a hinted leaf would silently discard
    the serialization ``sequential_scope`` exists to impose.
    """
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        other = Linear(x, _weight_matrix(1), name="other")
        add_scheduling_dependency(lanes[1], other)
        out = Concatenate([*lanes, other])
        _merge_only(out)

    assert lanes[1] in out.inputs
    assert other in lanes[1].scheduling_predecessors


def test_leaf_named_as_a_scheduling_predecessor_declines_merge():
    """Orphaning a hint's *target* would leave its waiter blocked forever.

    ``GraphAnalyzer.is_ready`` holds a node until every scheduling predecessor
    is in ``computed_nodes``; a predecessor that no longer exists in the graph
    never gets there, and the scheduler dies with ``No progress``.
    """
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        waiter = Linear(x, _weight_matrix(1), name="waiter")
        add_scheduling_dependency(waiter, lanes[1])
        out = Concatenate([*lanes, waiter])
        _merge_only(out)

    assert lanes[1] in out.inputs
    assert lanes[1] in waiter.scheduling_predecessors


def test_sequential_scope_survives_fusion():
    """The integration shape: ``sequential_scope`` over sibling Linears.

    Every row is a same-input Linear leaf of one concat — the merge's exact
    target — and every row after the first batch carries a hint.  The fold must
    leave the serialized rows alone.
    """
    with fresh_graph_session():
        x = create_input("x", 4)
        rows = sequential_scope(
            [lambda i=i: Linear(x, torch.zeros(4, 3), name=f"r_{i}") for i in range(4)],
            batch_size=2,
        )
        out = Concatenate(rows)
        hinted = [r for r in rows if r.scheduling_predecessors]
        assert hinted, "fixture no longer exercises scheduling hints"
        _merge_only(out)

    for row in rows:
        assert row in out.inputs
    assert all(r.scheduling_predecessors for r in hinted)


def test_output_node_leaf_declines_merge():
    """A caller-held output node's identity and width must survive."""
    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(N))
        out = concat(lanes)
        _merge_only(out, lanes[1])
    assert _census(out).get("Linear", 0) == 3
    assert lanes[1] in out.inputs
    assert len(lanes[1]) == 1


# --- Slice records: merged leaves stay addressable --------------------------
#
# The merge orphans every leaf but one, yet each leaf's value survives as a
# column slice of the widened survivor.  ``FoldLog.slices`` records where, the
# lowering boundary translates the records to source nodes, and the compile's
# source re-key turns them into residual-column slices.  Without this chain an
# output ``Concatenate`` of merged leaves has nothing to gather: compiling
# ``Concatenate([Linear(x, Wa), Linear(x, Wb)])`` raised ``KeyError`` from
# ``ResidualAssignment.get_node_indices`` — while attaching a debug assert to
# either leaf (declining the merge) made it compile, so a debug annotation
# decided compilability.


def test_output_concat_of_merged_siblings_compiles_and_matches_compute():
    """The regression reproducer, pinned at the compile level."""
    from torchwright.compiler.export import compile_headless

    with fresh_graph_session():
        x = create_input("x", D_EMBED, value_range=(-1.0, 1.0))
        lanes = _lanes(x, _weight_matrix(N))
        out = concat(lanes)
        compiled = compile_headless(out, d=D, d_head=D_HEAD, verbose=False)
        probe = _probe()
        got = compiled(probe).cpu()
        ref = out.compute(1, {"x": probe})
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-5)


def test_merged_leaves_are_probeable_as_survivor_slices():
    """Each merged source leaf keeps an addressable residual assignment.

    It's the leaf's slice of the survivor's columns, so probe_compiled
    checks it against the oracle instead of skipping it.
    """
    from torchwright.compiler.export import compile_headless
    from torchwright.debug.probe import probe_compiled

    with fresh_graph_session():
        x = create_input("x", D_EMBED, value_range=(-1.0, 1.0))
        lanes = _lanes(x, _weight_matrix(N))
        out = concat(lanes)
        compiled = compile_headless(out, d=D, d_head=D_HEAD, verbose=False)
        report = probe_compiled(compiled, out, {"x": _probe()}, n_pos=1, atol=1e-5)
    assert report.first_divergent is None, report.format_short()
    checked = {n.name for n in report.per_node}
    assert {lane.name for lane in lanes} <= checked


def test_fold_log_records_merge_slices():
    """The fold-log layer keeps one record per run member.

    Offsets follow run order, and the survivor's self-record sits at
    offset 0.
    """
    from torchwright.graph.optimize import FoldLog

    with fresh_graph_session():
        x = create_input("x", D_EMBED)
        lanes = _lanes(x, _weight_matrix(3))
        out = concat(lanes)
        log = FoldLog()
        assert fuse_consecutive_linears({out}, fold_log=log) > 0

    survivor = lanes[0]  # _merge_run widens run[0] in place
    assert log.slices[lanes[0]] == (survivor, 0, 1)
    assert log.slices[lanes[1]] == (survivor, 1, 1)
    assert log.slices[lanes[2]] == (survivor, 2, 1)


def _four_nodes():
    """Four distinct real nodes for FoldLog identity bookkeeping tests.

    FoldLog keys and targets by node object; the tests below never read
    weights or widths off the nodes, only identities, so minimal one-column
    Linears over one input serve.
    """
    x = create_input("x", 2)
    return [Linear(x, torch.zeros(2, 1), name=t) for t in ("a", "b", "c", "d")]


def test_fold_log_merge_composition():
    """A member that was itself a merge survivor carries its history forward.

    Its members' records follow it into the new survivor in one
    retarget step.
    """
    from torchwright.graph.optimize import FoldLog

    with fresh_graph_session():
        a, b, c, _ = _four_nodes()
        log = FoldLog()
        log.record_merge([a, b], [3, 2])  # a widened to 5: a=[0:3], b=[3:5]
        log.record_merge([c, a], [4, 5])  # c widened to 9, a's block at offset 4
        assert log.slices[c] == (c, 0, 4)
        assert log.slices[a] == (c, 4, 3)  # a's ORIGINAL value, not its widened one
        assert log.slices[b] == (c, 4 + 3, 2)


def test_fold_log_duplicate_survivor_occurrence_keeps_offset_zero():
    """``concat([l, l])`` duplicates the survivor without disturbing it.

    The survivor's own columns stay at the front; the second
    occurrence duplicates them and must not retarget the self-record.
    """
    from torchwright.graph.optimize import FoldLog

    with fresh_graph_session():
        a, _, _, _ = _four_nodes()
        log = FoldLog()
        log.record_merge([a, a], [3, 3])
        assert log.slices[a] == (a, 0, 3)


def test_fold_log_move_retargets_and_invalidates_slices():
    """record_move follows the orphan's slices onto the survivor.

    Slices of the survivor's replaced value are dropped.
    """
    from torchwright.graph.optimize import FoldLog

    with fresh_graph_session():
        a, b, s, ffn = _four_nodes()
        log = FoldLog()
        log.record_merge([s, a], [3, 2])  # s holds s=[0:3], a=[3:5]
        log.record_merge([ffn, b], [1, 1])  # unrelated records pointing at ffn
        # The FFN fold moves s's (widened) value onto ffn: records that read
        # s's columns follow; records that read ffn's OLD columns are
        # invalidated.
        log.record_move(orphan=s, survivor=ffn)
        assert log.slices[s] == (ffn, 0, 3)
        assert log.slices[a] == (ffn, 3, 2)
        assert ffn not in log.slices
        assert b not in log.slices
