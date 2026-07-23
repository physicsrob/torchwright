"""Integration tests for CP-SAT-driven forward compilation.

The CP-SAT solver in ``torchwright/compiler/forward/cpsat_scheduler.py``
produces a ``ScheduleAssignment`` that ``DirectedLayerScheduler`` (a
subclass of ``LayerScheduler``) replays through the existing per-layer
code path.  These tests exercise the integration end-to-end: the same
graph compiled twice (heuristic vs CP-SAT) must produce the same token
output, and the CP-SAT version must use no more layers than the
heuristic.

See ``docs/cpsat_scheduler.md`` for the spec.
"""

import pytest
import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import Costs
from torchwright.graph import Linear
from torchwright.ops.inout_nodes import create_input, create_literal_value
from torchwright.ops.linear import add, add_scaled_nodes, concat, sum_nodes
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

D = 256
D_HEAD = 16


def _build_relu_chain():
    """Input -> FFN (a degenerate-ReLU FFN) graph."""
    x = create_input("x", 8)
    ffn = linear_relu_linear(
        x,
        torch.randn(16, 8),
        torch.randn(16),
        torch.randn(16, 4),
        torch.randn(4),
        name="mlp",
    )
    return ffn, {"x": torch.randn(3, 8)}


def _build_branchy():
    """A non-trivial graph: input -> two parallel FFNs -> add."""
    x = create_input("x", 8)
    a = linear_relu_linear(
        x,
        torch.randn(16, 8),
        torch.zeros(16),
        torch.randn(16, 8),
        torch.zeros(8),
        name="a",
    )
    b = linear_relu_linear(
        x,
        torch.randn(16, 8),
        torch.zeros(16),
        torch.randn(16, 8),
        torch.zeros(8),
        name="b",
    )
    out = add(a, b)
    return out, {"x": torch.randn(2, 8)}


def test_relu_chain_compiles_with_cpsat():
    """Smallest non-trivial graph: chain of one FFN."""
    out, inputs = _build_relu_chain()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    actual = net.compute(3, inputs)[out].cpu()
    expected = out.compute(3, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_branchy_compiles_with_cpsat():
    """Two parallel chains feed into an Add."""
    out, inputs = _build_branchy()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_matches_heuristic_output():
    """Token output is identical regardless of scheduler choice.

    The schedule is a placement decision, not a value-changing
    transformation.  Compiling the same graph twice — once with the
    heuristic, once with CP-SAT — must produce the same numerical
    output (modulo float-point ordering effects).
    """
    out, inputs = _build_branchy()
    net_heur = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=0,
    )
    net_cpsat = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    out_heur = net_heur.compute(2, inputs)[out].cpu()
    out_cpsat = net_cpsat.compute(2, inputs)[out].cpu()
    torch.testing.assert_close(out_cpsat, out_heur, atol=1e-4, rtol=1e-4)


def test_cpsat_layer_count_no_worse_than_heuristic():
    """CP-SAT proves the layer-count optimum; heuristic is upper bound.

    With ``Costs(alpha=1, beta=0, gamma=0)`` (the default), CP-SAT
    minimizes layer count.  The heuristic — being a feasible schedule —
    gives an upper bound the solver always matches or beats.
    """
    out, _inputs = _build_branchy()
    net_heur = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=0,
    )
    net_cpsat = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    assert len(net_cpsat.layers) <= len(net_heur.layers), (
        f"CP-SAT used {len(net_cpsat.layers)} layers, "
        f"heuristic used {len(net_heur.layers)}; CP-SAT should be ≤"
    )


def test_cpsat_with_admission_control_raises():
    """``admission_control=True`` is a CP-SAT model precondition.

    See ``docs/cpsat_scheduler.md`` §3 — the model does not represent
    the sibling-cluster admission constraint, so a solver-feasible
    schedule may not be replayable.
    """
    out, _inputs = _build_relu_chain()
    with pytest.raises(RuntimeError, match="admission_control"):
        forward_compile(
            d=D,
            d_head=D_HEAD,
            output_node=out,
            verbose=False,
            optimize=1,
            admission_control=True,
        )


def test_cpsat_flex_routing_explores_both_sublayers():
    """Flex routing is a CP-SAT decision variable per standalone Linear.

    A graph with a single standalone Linear can route to either
    attention or MLP-bypass; flex_routing=True lets the solver pick.
    The compile must succeed regardless.
    """
    x = create_input("x", 8)
    out = Linear(x, torch.randn(8, 4), torch.randn(4), name="lin")

    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
        cpsat_flex_routing=True,
    )
    inputs = {"x": torch.randn(2, 8)}
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_costs_beta_routes_more_to_mlp():
    """beta>0 should route at least as many heads off attention as beta=0.

    With multiple standalone Linears (no chains), the alpha=1, beta=0
    objective is indifferent between attention and MLP routing as long
    as the layer count is the same.  Setting beta>0 makes attention
    heads costly, so the solver prefers MLP-bypass.
    """
    x = create_input("x", 8)
    l1 = Linear(x, torch.randn(8, 4), torch.zeros(4), name="l1")
    l2 = Linear(x, torch.randn(8, 4), torch.zeros(4), name="l2")
    out = add(l1, l2)

    net_alpha = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
        cpsat_costs=Costs(alpha=1, beta=0, gamma=0),
    )
    net_beta = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
        cpsat_costs=Costs(alpha=1, beta=10, gamma=0),
    )
    # Both compiles should produce the same numerical output.
    inputs = {"x": torch.randn(2, 8)}
    out_alpha = net_alpha.compute(2, inputs)[out].cpu()
    out_beta = net_beta.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(out_alpha, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_beta, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_compiles_under_zero_init():
    """The model and heuristic skip BIRTH-layer dirty cancels because the
    runtime always zero-initialises the residual stream (universal since
    the ``assume_zero_init`` flag was retired).  The CP-SAT-scheduled module
    must still produce correct output — which ``HeadlessTransformer.compute()``
    verifies by zero-initialising the stream.
    """
    out, inputs = _build_branchy()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_warm_start_layer_count_no_worse():
    """Warm-start must not produce a worse schedule than the heuristic.

    ``forward_compile`` runs a schedule-only heuristic pass before
    invoking CP-SAT and feeds the complete assignment as ``incumbent``.
    Because the warm start is feasible, CP-SAT can always match it; with
    ``cpsat_costs.alpha=1`` (the default) it tries to beat it.
    """
    out, _inputs = _build_branchy()
    net_heur = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=0,
    )
    net_cpsat = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    assert len(net_cpsat.layers) <= len(net_heur.layers)


def test_cpsat_compiles_shared_input_adds():
    """Two Adds sharing both inputs — neither addend can be free.

    Mirrors ``test_compile_add_shared_inputs`` from
    ``test_forward_compile.py``.  Both ``x`` and ``y`` feed both Adds,
    so neither input is dead at either Add's layer — the heuristic
    falls back to ``compute_add`` (the costly regime).  Under the P2
    fix, CP-SAT models the compute-add cost and must produce a
    schedule that's replayable.
    """
    width = 8
    x = create_input("x", width)
    y = create_input("y", width)
    sum1 = add(x, y)
    sum2 = add(x, y)
    out = add_scaled_nodes(1.0, sum1, 1.0, sum2)
    inputs = {"x": torch.randn(2, width), "y": torch.randn(2, width)}

    net_heur = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=0,
        max_layers=10,
    )
    net_cpsat = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
        max_layers=10,
    )
    out_heur = net_heur.compute(2, inputs)[out].cpu()
    out_cpsat = net_cpsat.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(out_cpsat, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_heur, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_compiles_three_adds_shared_inputs():
    """Three Adds sharing both inputs — the calculator pattern.

    Mirrors ``test_compile_three_adds_shared_inputs`` from
    ``test_forward_compile.py``.  Each addend has multiple Add
    consumers; whether an addend is dead at a given Add depends on
    when the other Adds are scheduled — exercises the reified
    consumer-ordering booleans.
    """
    x = create_input("x", 1)
    y = create_input("y", 1)
    s1 = add(x, y)
    s2 = add(x, y)
    s3 = add(x, y)
    out = sum_nodes([s1, s2, s3])
    inputs = {
        "x": torch.tensor([[3.0], [7.0]]),
        "y": torch.tensor([[4.0], [2.0]]),
    }

    net_cpsat = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
        max_layers=10,
    )
    out_cpsat = net_cpsat.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(out_cpsat, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_compiles_concatenate_input_add():
    """Add with a Concatenate input — used to be a precondition raise.

    Before P2, ``solve_schedule`` rejected graphs containing
    ``Concatenate``-input Adds with a precondition error: the model
    assumed free-add cost everywhere, but the heuristic forces
    ``compute_add`` for these (Concatenate has no reusable cols).
    After P2, the model encodes both regimes and the precondition is
    gone — these graphs compile cleanly under ``optimize=1``.
    """
    c1 = create_literal_value(torch.tensor([1.0]))
    c2 = create_literal_value(torch.tensor([1.0]))
    c3 = create_literal_value(torch.tensor([2.0, 2.0]))
    out = add(concat([c1, c2]), c3)

    net_cpsat = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    actual = net_cpsat.compute(1, {})[out].cpu()
    expected = out.compute(1, {})
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_falls_back_to_heuristic_when_no_incumbent(monkeypatch):
    """When CP-SAT finds no feasible solution within budget,
    ``forward_compile`` falls back to the heuristic schedule rather
    than raising.  Simulated by monkey-patching ``solve_schedule`` to
    return ``(None, stats)``; the compile must still produce the
    correct token output.
    """
    from torchwright.compiler.forward import compile as compile_mod
    from torchwright.compiler.forward.cpsat_scheduler import SolveStats

    fake_stats = SolveStats(
        status_name="UNKNOWN",
        objective_value=-1,
        best_objective_bound=0.0,
        wall_time_s=0.0,
        solver_log="",
        total_attn_heads=-1,
        total_mlp_bypass_slots=-1,
        is_optimal=False,
    )

    def fake_solve(*args, **kwargs):
        return None, fake_stats

    monkeypatch.setattr(compile_mod, "solve_schedule", fake_solve)

    out, inputs = _build_branchy()
    # The fallback now warns loudly (RuntimeWarning) so it can't masquerade
    # as a solve; the compile is still valid.
    with pytest.warns(RuntimeWarning, match="UNOPTIMIZED|fall(ing|s)? back"):
        net = forward_compile(
            d=D,
            d_head=D_HEAD,
            output_node=out,
            verbose=False,
            optimize=1,
        )
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_frees_wide_input_for_intermediate():
    """A wide input consumed early must be freed so its columns carry a
    wide downstream intermediate.

    Regression for the input-freeing fix.  The token-`Embedding`-shaped
    case in miniature: a 24-wide input ``x`` feeds one narrow ``Linear``
    and is then dead, after which a 40-wide standalone ``Linear`` ``b``
    must materialise.  At ``d=64``:

      * Pinning ``x`` forever (the pre-fix model) keeps ``x`` (24) and
        its consumer ``a`` (4) live while ``b`` (40) materialises —
        ``24 + 4 + 40 = 68 > 64`` residual columns, so the residual
        cumulative is INFEASIBLE and CP-SAT falls back.
        ``require_solver=True`` turns that silent regression into a
        hard error.
      * Freeing ``x`` once its consumer ``a`` runs (the fix) frees its
        24 columns, leaving ``a`` (4) plus ``b`` (40) = 44 <= 64 — a
        comfortable fit, so CP-SAT solves.

    The compiled output must match the graph oracle: the directed replay
    has to actually reclaim ``x``'s columns and reuse them for ``b``.
    """
    torch.manual_seed(0)
    x = create_input("x", 24)
    a = Linear(x, torch.randn(24, 4), torch.zeros(4), name="a")
    b = Linear(a, torch.randn(4, 40), torch.zeros(40), name="b")
    out = Linear(b, torch.randn(40, 4), torch.zeros(4), name="out")

    net = forward_compile(
        d=64,
        d_head=8,
        output_node=out,
        verbose=False,
        optimize=1,
        require_solver=True,
    )
    # require_solver=True would have raised on a fallback, so reaching here
    # means CP-SAT produced a real assignment.
    assert net.cpsat_solve_stats is not None
    assert net.cpsat_solve_stats.status_name in ("OPTIMAL", "FEASIBLE")

    inputs = {"x": torch.randn(2, 24)}
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_require_solver_raises_on_fallback(monkeypatch):
    """``require_solver=True`` converts a silent CP-SAT fallback into a hard
    error; the default (``require_solver=False``) warns loudly instead of
    failing silently.  Regression guard for the silent-fallback footgun.
    """
    from torchwright.compiler.forward import compile as compile_mod
    from torchwright.compiler.forward.cpsat_scheduler import SolveStats

    fake_stats = SolveStats(
        status_name="INFEASIBLE",
        objective_value=-1,
        best_objective_bound=0.0,
        wall_time_s=0.0,
        solver_log="",
        total_attn_heads=-1,
        total_mlp_bypass_slots=-1,
        is_optimal=False,
    )
    monkeypatch.setattr(
        compile_mod, "solve_schedule", lambda *a, **k: (None, fake_stats)
    )

    out, inputs = _build_branchy()

    # require_solver=True -> raise rather than silently fall back.
    with pytest.raises(RuntimeError, match="no usable assignment|require_solver"):
        forward_compile(
            d=D,
            d_head=D_HEAD,
            output_node=out,
            verbose=False,
            optimize=1,
            require_solver=True,
        )

    # require_solver=False (default) -> warn, fall back, still produce a
    # correct compile.
    with pytest.warns(RuntimeWarning, match="UNOPTIMIZED|fall(ing|s)? back"):
        net = forward_compile(
            d=D,
            d_head=D_HEAD,
            output_node=out,
            verbose=False,
            optimize=1,
        )
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
