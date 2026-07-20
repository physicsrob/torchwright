"""The simple multiply's carry sweep must stay collapsible to its op depth.

Regression guard for the calculator layer-gap investigation (2026-07-20).
``multiply_digit_seqs`` threads a scalar column total through a serial
carry sweep; each per-column ``total -> digit / carry`` step is a
univariate staircase that the compiler's collapse passes flatten to one
FFN — one compiled layer per column.  That collapse is a *certified*
rewrite: the chain's exact fp32 compute must sit within the 1e-3 claim
budget of a piecewise-linear fit.

The regression this pins: a saturated ``in_range`` slot at index ``i``
carries fp32 rounding jitter of ``ulp(scale·sharpness·i)/scale`` at
off-integer inputs (its hinge pair straddles a binade boundary), and
reading the carry off the wide one-hot with a selection row weighted
``0..2n`` amplified two adjacent slots' jitter to 1.22e-3 — past the
budget — so every carry step declined and compiled at three layers
instead of one (a 22-layer gap on the full calculator at
``max_digits=6``).  The carry is now a unit-weight threshold-count sum,
which keeps the composed jitter an order of magnitude under budget.

The multiply is built over InputNode digits (not literals): a literal
operand makes the whole sweep constant-ancestry, and the collapse
passes only seed univariate subgraphs at non-constant scalar sources.
"""

from examples.calculator_simple import multiply_digit_seqs
from scripts.arithmetic_scaling import critical_path_depth
from torchwright.compiler.forward.cpsat_scheduler import critical_path_layers
from torchwright.compiler.lower import lower
from torchwright.graph.misc import Concatenate
from torchwright.ops.inout_nodes import create_input, create_onehot_embedding

VOCAB = [str(d) for d in range(10)]


def test_multiply_carry_sweep_collapses_to_op_depth():
    n = 6  # the digit count where the amplified jitter first broke the budget
    embedding = create_onehot_embedding(vocab=VOCAB)
    a = [create_input(f"a{i}", len(VOCAB), value_range=(0.0, 1.0)) for i in range(n)]
    b = [create_input(f"b{i}", len(VOCAB), value_range=(0.0, 1.0)) for i in range(n)]
    out = Concatenate(multiply_digit_seqs(embedding, a, b))

    depth = critical_path_depth([out])
    lowered = lower(
        out,
        collapse_univariate=True,
        collapse_pl=True,
        collapse_lane_cap=2048,  # d_hidden 8192 // 4, the flagship geometry
    )
    floor = critical_path_layers(lowered.output_node)
    declines = [
        o.format_line()
        for o in lowered.collapse_pl_report.outcomes
        if not o.collapsed and "no depth gain" not in o.reason
    ]
    assert floor <= depth, (
        f"lowered layer floor {floor} exceeds the nonlinear-op depth {depth}: "
        f"the carry-sweep collapse regressed.  collapse_pl declines:\n"
        + ("\n".join(declines) or "  (none — the gap is elsewhere)")
    )
