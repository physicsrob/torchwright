"""Just-in-time materialization of ``LiteralValue`` constants.

Before this change a constant was an *input node*: it was allocated a
residual-stream column at layer 0 and held across the whole network, even
when consumed only deep in the graph (the ``select(cond, x, literal)``
case).  Now a constant is a first-class *schedulable* node, materialized via
``compute_literal_value`` only in the layer(s) around its consumer and freed
after use.  See ``constants_plan.md``.

The end-to-end *correctness* of literal-bearing graphs is covered by
``test_forward_compile.py`` (``test_compile_constant``,
``test_compile_select``, ``test_compile_multi_switch_shared_constants``,
…).  The tests here assert the JIT-specific properties those don't:

- a deep-consumed constant is **not** prefilled at layer 0;
- it is materialized in the interior and **freed** after its consumer;
- compiled values still match the oracle under both the heuristic
  (``optimize=0``) and CP-SAT (``optimize=1``) schedulers;
- pure-constant subgraphs and shared constants compile correctly.
"""

import pytest
import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.graph import Linear
from torchwright.ops.linear import add, concat
from torchwright.ops.attention_ops import attend_to_offset
from torchwright.ops.inout_nodes import (
    create_input,
    create_literal_value,
    create_rope_config,
)
from torchwright.ops.relu.map_select import select

D = 256
D_HEAD = 16


def _idchain(x, depth, name):
    """A chain of ``depth`` identity Linears — forces ``depth`` layers of
    dependency without changing the value (each ``Linear(h, I, 0) == h``)."""
    h = x
    for i in range(depth):
        h = Linear(h, torch.eye(len(x)), torch.zeros(len(x)), name=f"{name}{i}")
    return h


def _live_layers(net, node):
    """Layer indices whose post-MLP residual snapshot holds ``node``."""
    ra = net.residual_assignment
    return [
        i
        for i, layer in enumerate(net.layers)
        if ra.has_node(layer.mlp.out_state, node)
    ]


def _first_layer(net, node):
    """First post-MLP layer that materializes ``node`` (None if never)."""
    live = _live_layers(net, node)
    return live[0] if live else None


# ---------------------------------------------------------------------------
# is_input_node no longer claims constants
# ---------------------------------------------------------------------------


def test_literal_is_not_an_input_node():
    lit = create_literal_value(torch.tensor([1.0, 2.0]))
    assert not GraphAnalyzer(lit).is_input_node(lit)


# ---------------------------------------------------------------------------
# Deep-consumed constant: not prefilled, materialized in the interior, freed
# ---------------------------------------------------------------------------


def test_deep_literal_not_prefilled_and_materialized_in_interior():
    x = create_input("x", 2, value_range=(-5.0, 5.0))
    h = _idchain(x, 3, "pre")  # consumer lands several layers in
    lit = create_literal_value(torch.tensor([7.0, -3.0]))
    consumed = add(h, lit)  # the constant's only consumer — deep
    out = _idchain(consumed, 3, "post")  # downstream work after the consumer

    net = forward_compile(
        d=D, d_head=D_HEAD, output_node=out, verbose=False
    )
    ra = net.residual_assignment

    # (1) The constant is not baked into the layer-0 input residual stream.
    in0 = ra.get_nodes(net.layers[0].attn.in_state)
    assert lit not in in0, "constant was prefilled at layer 0 (still an input node)"

    # (2) It is materialized strictly in the interior: born after layer 0
    #     (just-in-time, not eagerly) and freed before the final layer (its
    #     consumer is mid-network, downstream layers don't need it).
    live = _live_layers(net, lit)
    assert live, "constant was never materialized"
    assert min(live) > 0, f"constant materialized at layer 0, not JIT: live={live}"
    assert max(live) < len(net.layers) - 1, (
        f"constant held to the final layer instead of freed after its "
        f"consumer: live={live}, n_layers={len(net.layers)}"
    )

    # (3) Output is still correct.
    xv = torch.randn(4, 2)
    result = net.compute(4, {"x": xv})
    assert torch.allclose(result[out].cpu(), out.compute(4, {"x": xv}), atol=1e-3)


# ---------------------------------------------------------------------------
# Oracle agreement under both schedulers (the motivating select case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("optimize", [0, 1])
def test_deep_select_literal_matches_oracle(optimize):
    """``select(cond, <deep>, literal)`` — the constant is the false branch,
    consumed deep in the network.  Compiled output must match exact math
    under the heuristic (optimize=0) and CP-SAT (optimize=1) schedulers, and
    the false-branch positions must equal the constant."""
    cond = create_input("cond", 1, value_range=(-1.0, 1.0))
    base = create_input("t", 2, value_range=(-4.0, 4.0))
    deep_true = _idchain(base, 3, "t")
    lit = create_literal_value(torch.tensor([2.0, -1.0]))
    out = select(cond, deep_true, lit)

    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=optimize,
    )

    n_pos = 3
    inputs = {
        "cond": torch.tensor([[1.0], [-1.0], [1.0]]),  # position 1 picks false
        "t": torch.randn(n_pos, 2),
    }
    actual = net.compute(n_pos, inputs)[out].cpu()
    expected = out.compute(n_pos, inputs)
    assert torch.allclose(
        actual, expected, atol=1e-3
    ), f"optimize={optimize} max diff {(actual - expected).abs().max():.5f}"
    # The false-branch row must equal the just-in-time-materialized constant.
    assert torch.allclose(actual[1], lit.value, atol=1e-2)


# ---------------------------------------------------------------------------
# No deadlock: a pure-constant computation has nothing to wait for
# ---------------------------------------------------------------------------


def test_pure_constant_subgraph_compiles():
    a = create_literal_value(torch.tensor([1.0, 2.0]))
    b = create_literal_value(torch.tensor([3.0, -4.0]))
    out = add(a, b)  # Add(literal, literal) — eligible immediately, no stall

    net = forward_compile(d=D, d_head=D_HEAD, output_node=out, verbose=False)
    result = net.compute(2, {})
    assert torch.allclose(result[out].cpu(), out.compute(2, {}), atol=1e-4)


# ---------------------------------------------------------------------------
# Shared constant: two consumers at different depths, still correct
# ---------------------------------------------------------------------------


def test_shared_literal_two_consumers_correct():
    x = create_input("x", 2, value_range=(-5.0, 5.0))
    lit = create_literal_value(torch.tensor([1.5, -2.5]))
    shallow = add(x, lit)  # needs the constant near layer 0
    deep = add(_idchain(x, 3, "d"), lit)  # needs it deep
    # Wrap the join in an identity Linear so the output is a single
    # materialized node (a bare Concatenate isn't keyed in compute()'s result).
    out = Linear(concat([shallow, deep]), torch.eye(4), torch.zeros(4), name="out")

    net = forward_compile(
        d=D, d_head=D_HEAD, output_node=out, verbose=False
    )
    xv = torch.randn(3, 2)
    result = net.compute(3, {"x": xv})
    assert torch.allclose(result[out].cpu(), out.compute(3, {"x": xv}), atol=1e-3)
    # The constant is materialized at least once and lives long enough to
    # serve both consumers.
    assert _live_layers(net, lit), "shared constant was never materialized"


# ---------------------------------------------------------------------------
# Non-foldable path: a constant feeding an attention value (Attn has no bias)
# ---------------------------------------------------------------------------


def test_deep_attention_fed_constant_matches_oracle():
    """A constant that feeds an attention *value* path must be materialized
    just-in-time and produce correct output.  ``Attn`` has no bias, so this
    constant could never be folded — JIT is the only mechanism that handles
    it.  The constant is part of the attention value via a Concatenate, needed
    only when the (deep) attention op runs."""
    rope = create_rope_config(d_head=D_HEAD, max_positions=512)
    x = create_input("x", 2, value_range=(-5.0, 5.0))
    h = _idchain(x, 3, "d")
    lit = create_literal_value(torch.tensor([4.0, -2.0]))
    v = concat([h, lit])  # the constant is part of the attention value
    out = attend_to_offset(rope, v, delta_pos=-1)

    net = forward_compile(
        d=D, d_head=D_HEAD, output_node=out, verbose=False
    )
    # Not prefilled — materialized as a schedulable node.
    assert lit not in net.residual_assignment.get_nodes(net.layers[0].attn.in_state)
    n_pos = 5
    xv = torch.randn(n_pos, 2)
    actual = net.compute(n_pos, {"x": xv})[out].cpu()
    expected = out.compute(n_pos, {"x": xv})
    assert torch.allclose(
        actual, expected, atol=1e-2
    ), f"max diff {(actual - expected).abs().max():.5f}"


# ---------------------------------------------------------------------------
# Dirty-column reuse: a late constant lands on a recycled column
# ---------------------------------------------------------------------------


def test_literal_into_recycled_column_is_clean():
    """A constant materialized deep in the network, under residual pressure,
    lands on a column recycled from an earlier dead node.  The birth
    dirty-cancel must zero that column before the bias write — otherwise the
    dead node's stale value contaminates the constant.  The investigation
    flagged this recycled-column path as previously untested, since the old
    code prefilled every constant into its own layer-0 column.

    A small ``d`` forces recycling; correctness of the output (the constant
    is a saturated value the stale data would visibly corrupt) validates the
    cancel."""
    x = create_input("x", 8, value_range=(-3.0, 3.0))
    # ``early`` is consumed immediately and then dies, freeing its (now
    # dirty) columns back into the pool.
    early = _idchain(x, 1, "early")
    sink = add(x, early)  # consumes ``early``; it dies here
    deep = _idchain(sink, 4, "deep")  # push the constant's consumer late
    lit = create_literal_value(torch.full((8,), 5.0))
    out = add(deep, lit)

    net = forward_compile(
        d=64, d_head=16, output_node=out, verbose=False
    )
    xv = torch.randn(5, 8)
    result = net.compute(5, {"x": xv})
    assert torch.allclose(result[out].cpu(), out.compute(5, {"x": xv}), atol=1e-3)


# ---------------------------------------------------------------------------
# Zero added latency: the JIT gate must not delay the consumer
# ---------------------------------------------------------------------------


def _consumer_layer(other_kind):
    """Compile ``Add(deep_chain(x), other)`` and return the layer the Add is
    materialized at.  ``other`` is either a constant (JIT-materialized) or a
    pre-seeded input (available from layer 0).  Structurally identical
    otherwise, so the Add's layer reveals whether the JIT gate delayed it."""
    x = create_input("x", 2, value_range=(-5.0, 5.0))
    deep = _idchain(x, 3, "d")
    if other_kind == "literal":
        other = create_literal_value(torch.tensor([1.0, 2.0]))
    else:
        other = create_input("c", 2, value_range=(-5.0, 5.0))
    out = add(deep, other)
    net = forward_compile(
        d=D, d_head=D_HEAD, output_node=out, verbose=False
    )
    return _first_layer(net, out)


def test_jit_constant_adds_no_latency_to_consumer():
    """The consumer of a just-in-time constant is scheduled at the same layer
    as it would be if that constant were a pre-seeded input (available from
    layer 0).  Materializing the constant alongside the consumer's last
    non-constant input — rather than eagerly at layer 0 — costs the consumer
    nothing.  This is the load-bearing scheduling claim from the design."""
    with_literal = _consumer_layer("literal")
    with_input = _consumer_layer("input")
    assert with_literal is not None and with_input is not None
    assert with_literal <= with_input, (
        f"JIT gate delayed the consumer: Add at layer {with_literal} with a "
        f"constant vs {with_input} with a pre-seeded input"
    )


# ---------------------------------------------------------------------------
# I4 (column liveness): a constant must not be freed before its consumer
# ---------------------------------------------------------------------------


def test_jit_graph_passes_end_of_layer_liveness(monkeypatch):
    """Compile a deep-constant graph with the gated end-of-layer liveness
    walk on (``TW_COMPILER_VERIFY=1``).  It raises if any node — here a
    just-in-time constant — is freed while an effective consumer is still
    uncomputed.  Exercising it directly validates the new free-after-use
    behavior for constants."""
    monkeypatch.setenv("TW_COMPILER_VERIFY", "1")
    x = create_input("x", 2, value_range=(-5.0, 5.0))
    deep = _idchain(x, 3, "d")
    lit = create_literal_value(torch.tensor([7.0, -3.0]))
    out = add(deep, lit)

    net = forward_compile(
        d=D, d_head=D_HEAD, output_node=out, verbose=False
    )
    xv = torch.randn(3, 2)
    assert torch.allclose(
        net.compute(3, {"x": xv})[out].cpu(), out.compute(3, {"x": xv}), atol=1e-3
    )


# ---------------------------------------------------------------------------
# CP-SAT discovers just-in-time placement as the residual-pressure optimum
# ---------------------------------------------------------------------------


def test_cpsat_treats_constant_as_schedulable_not_prefilled():
    """Phase 2: under CP-SAT (optimize=2) the constant is a *schedulable* node
    materialized via ``compute_literal_value`` — not a residual column
    prefilled at layer 0 — and the output is correct.

    Note on placement: CP-SAT defers a constant (just-in-time) only when
    residual pressure makes a later birth strictly better.  In an
    unconstrained graph like this one (large ``d``, few nodes) the solver is
    *indifferent* to the constant's birth layer and may place it early — which
    is harmless, since the column is uncontended and the layer count is
    unchanged.  The pressure-bound "places late" behavior is exercised on a
    real, residual-bound graph (the DOOM graph), not here."""
    x = create_input("x", 2, value_range=(-5.0, 5.0))
    deep = _idchain(x, 4, "d")
    lit = create_literal_value(torch.tensor([1.0, 2.0]))
    out = add(deep, lit)

    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=2,
    )
    ra = net.residual_assignment
    # Schedulable, not a prefilled source.
    assert lit not in ra.get_nodes(
        net.layers[0].attn.in_state
    ), "CP-SAT prefilled the constant into layer-0 in_state (still a source)"
    assert _live_layers(net, lit), "CP-SAT never materialized the constant"
    # Correct.
    xv = torch.randn(3, 2)
    assert torch.allclose(
        net.compute(3, {"x": xv})[out].cpu(), out.compute(3, {"x": xv}), atol=1e-3
    )
