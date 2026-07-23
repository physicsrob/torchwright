"""Compiled-correctness tests for bias=False (no-bias emission).

docs/no_bias_plan.md: under ``bias=False`` every bias folds into the weight
matrices against the pinned constant-1 column — hidden-side biases as
const-column rows, output-side constants (literals, deferred Linear biases,
FFN out biases) through the constant lane at hidden slot 0, which both
schedulers reserve.  These tests pin the end-to-end behavior on both
machines: ``probe_compiled`` agrees with the oracle, the physical bias
vectors stay zero, literals land bit-exactly, the slot-0 reservation costs
exactly one slot of per-layer capacity on both schedulers, and the schedule
cache key distinguishes the modes while keeping bias=True hashes stable.

``add_const`` is the smallest op exercising the deferred-bias route (an
identity matmul on an attention head plus a ``compute_bias`` constant); it
is used here as the canonical biased-Linear fixture.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.graph_identity import graph_fingerprint
from torchwright.debug.probe import probe_compiled
from torchwright.graph.misc import LiteralValue
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear import add, add_const
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn

N_POS = 4
D = 64
D_HEAD = 8


def _assert_no_physical_bias(compiled):
    """Every layer's bias vectors must be exactly zero under bias=False."""
    for i, layer in enumerate(compiled._net.layers):
        mlp = layer.mlp
        if mlp.activation == "swish":
            vecs = [
                mlp.gate_proj.output_bias,
                mlp.up_proj.output_bias,
                mlp.down_proj.output_bias,
            ]
        else:
            vecs = [mlp.linear1.output_bias, mlp.linear2.output_bias]
        for v in vecs:
            assert (v == 0.0).all(), f"layer {i} has a nonzero bias vector"


def _relu_graph(seed=11):
    """add_const (deferred-bias route) -> relu FFN, plus a literal Add —
    all four bias-writing op types in one graph.
    """
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(seed)
    shifted = add_const(x, 3.25)
    h = linear_relu_linear(
        x,
        torch.randn(10, 4, generator=g) * 0.3,
        torch.randn(10, generator=g) * 0.1,
        torch.randn(10, 4, generator=g) * 0.3,
        torch.randn(4, generator=g) * 0.1,
        name="ffn",
    )
    lit = LiteralValue(torch.tensor([2.5, -1.25, 0.5, 0.125]), name="lit")
    out = add(add(shifted, h), lit)
    return x, lit, out


def _swish_graph(seed=13):
    """Same shape on the swish machine: add_const -> gated swiglu FFN +
    literal (add_const/add are the machine-neutral linear seams).
    """
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(seed)
    shifted = add_const(x, 3.25)
    h = swiglu_ffn(
        x,
        torch.randn(10, 4, generator=g) * 0.3,
        torch.randn(10, generator=g) * 0.1,
        torch.randn(10, 4, generator=g) * 0.3,
        torch.randn(4, generator=g) * 0.1,
        up_proj=torch.randn(10, 4, generator=g) * 0.3,
        up_bias=torch.randn(10, generator=g) * 0.1,
        name="ffn",
    )
    lit = LiteralValue(torch.tensor([2.5, -1.25, 0.5, 0.125]), name="lit")
    out = add(add(shifted, h), lit)
    return x, lit, out


@pytest.fixture
def xt():
    g = torch.Generator().manual_seed(99)
    return torch.randn(N_POS, 4, generator=g)


@pytest.mark.parametrize("build", [_relu_graph, _swish_graph], ids=["relu", "swish"])
def test_no_bias_probe_clean(build, xt):
    """probe_compiled agrees with the oracle everywhere under bias=False,
    on both machines, and no physical bias vector is written.
    """
    _, lit, out = build()
    compiled = compile_headless(out, d=D, d_head=D_HEAD, bias=False)
    _assert_no_physical_bias(compiled)
    report = probe_compiled(compiled, out, {"x": xt}, N_POS, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


@pytest.mark.parametrize("build", [_relu_graph, _swish_graph], ids=["relu", "swish"])
def test_no_bias_literal_bit_exact_end_to_end(build, xt):
    """A literal compiled under bias=False lands bitwise-equal (the
    constant lane computes exactly 1.0; only the lane writes its columns).
    """
    from torchwright.graph.asserts import assert_in_range

    _, lit, out = build()
    # debug=True's snapshot capture (what debug_value reads) only engages
    # when the graph carries checked nodes — attach a check to the output.
    wrapped = assert_in_range(out, -1000.0, 1000.0)
    compiled = compile_headless(wrapped, d=D, d_head=D_HEAD, bias=False)
    compiled(xt, debug=True)
    val = compiled.debug_value(lit)
    assert val is not None
    expected = lit.compute(N_POS, {}).to(val.device)
    assert torch.equal(val, expected), f"literal drifted: {val} vs {expected}"


@pytest.mark.parametrize("build", [_relu_graph, _swish_graph], ids=["relu", "swish"])
def test_no_bias_matches_biased_compile(build, xt):
    """The same graph compiled with bias=True and bias=False agrees to
    fp32 accumulation noise (the folds move bias adds into the matmul).
    """
    _, _, out_a = build()
    ca = compile_headless(out_a, d=D, d_head=D_HEAD, bias=True)
    _, _, out_b = build()
    cb = compile_headless(out_b, d=D, d_head=D_HEAD, bias=False)
    assert torch.allclose(ca(xt), cb(xt), atol=1e-5)


def _two_ffn_graph(n_lanes=4, seed=7):
    """Two independent FFNs whose lane demand sums to exactly 2*n_lanes."""
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    g = torch.Generator().manual_seed(seed)

    def ffn(name):
        return linear_relu_linear(
            x,
            torch.randn(n_lanes, 4, generator=g) * 0.3,
            torch.randn(n_lanes, generator=g) * 0.1,
            torch.randn(n_lanes, 4, generator=g) * 0.3,
            torch.randn(4, generator=g) * 0.1,
            name=name,
        )

    return x, add(ffn("a"), ffn("b"))


def test_no_bias_capacity_edge_heuristic(xt):
    """d_hidden exactly equal to the layer's lane demand: bias=True packs
    both FFNs into one MLP sublayer; bias=False reserves slot 0, so the
    same demand must spill — and the compiled values stay oracle-clean.
    """
    _, out_a = _two_ffn_graph()
    ca = compile_headless(out_a, d=D, d_head=D_HEAD, d_hidden=8, bias=True)
    _, out_b = _two_ffn_graph()
    cb = compile_headless(out_b, d=D, d_head=D_HEAD, d_hidden=8, bias=False)
    assert len(cb._net.layers) > len(ca._net.layers), (
        f"slot-0 reservation did not cost capacity: "
        f"{len(ca._net.layers)} vs {len(cb._net.layers)} layers"
    )
    report = probe_compiled(cb, out_b, {"x": xt}, N_POS, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_no_bias_capacity_edge_cpsat(xt):
    """The CP-SAT path models the reserved slot too (capacity d_hidden-1):
    a real solve under bias=False replays cleanly at the capacity edge.
    """
    _, out = _two_ffn_graph()
    compiled = compile_headless(
        out, d=D, d_head=D_HEAD, d_hidden=8, bias=False, optimize=1
    )
    _assert_no_physical_bias(compiled)
    report = probe_compiled(compiled, out, {"x": xt}, N_POS, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


def test_schedule_fingerprint_keys_on_bias():
    """bias=False changes the schedule-cache key; bias=True hashes
    byte-identically to the pre-feature payload (compatibility: existing
    cache entries keep hitting).
    """
    _, _, out = _relu_graph()
    common = dict(
        d=D,
        d_head=D_HEAD,
        d_hidden=D,
        flex_routing=True,
        cancel_slack=2,
        policy=None,
    )
    fp_default = graph_fingerprint(out, **common)
    fp_true = graph_fingerprint(out, bias=True, **common)
    fp_false = graph_fingerprint(out, bias=False, **common)
    assert fp_default == fp_true
    assert fp_false != fp_true


def test_no_bias_rejects_d_hidden_1():
    """bias=False needs d_hidden >= 2 (slot 0 is the constant lane)."""
    _, _, out = _relu_graph()
    with pytest.raises(ValueError, match="constant lane"):
        compile_headless(out, d=D, d_head=D_HEAD, d_hidden=1, bias=False)
