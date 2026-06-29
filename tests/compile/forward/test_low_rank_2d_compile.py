"""Regression tests for the scheduler deadlock in ``low_rank_2d``.

``low_rank_2d`` emits, for rank K, a fanout from shared width-K 1D PL
nodes into K parallel ``multiply_2d`` chains, summed at the end.  When
this graph is compiled via ``compile_headless`` the scheduler aborts
with ``RuntimeError: No progress: N nodes remaining, M free columns``
— not out of residual-stream space (plenty of columns free) but unable
to schedule a remaining dependency pattern.

The op's unit tests in ``tests/ops/test_low_rank_2d.py`` pass — but
only because they use a small 5×5 uniform grid and rank=1.  The tests
here use the realistic 15×9 non-uniform grid from the TODO, and at that
scale the compile deadlocks across every rank / depth combination.

**What the failures show diagnostically:**

- ``K=1`` with clamped inputs fails (~4 nodes stuck) — the bug is *not*
  rank-dependent.  Even a single ``multiply_2d`` subtree inside the
  fanout-sum pattern is enough.
- ``K=3`` at ``max_layers=200`` fails the same way as at
  ``max_layers=20`` — the bug is *not* depth-budget-dependent.
- ``K=3`` with *no* preceding clamp also fails — the bug isn't an
  interaction with surrounding structure; it's internal to
  ``low_rank_2d``'s emitted graph shape.
- Fully 487/512 residual columns are still free when the scheduler
  gives up, so this isn't a resource-exhaustion problem either.

The stuck nodes are all downstream of nodes the scheduler has already
placed; there's no cyclic dependency.  Something in admission control
is refusing to place them under the fanout-then-sum pattern the op
generates.

**Fix candidates** (either closes these tests):

1. Teach the scheduler to resolve this fanout-then-sum dependency
   pattern.  There's no cyclic dependency; the stuck nodes are all
   downstream of nodes that were already computed — something in
   admission control is refusing to place them.
2. Refactor ``low_rank_2d`` to emit a single fused
   ``linear_relu_linear`` for all K multiplies, removing the fanout
   structure from the compiled graph.  Also delivers the 2-sublayer
   cost I originally advertised (currently it's 1 + K sublayers).
"""

import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.arithmetic_ops import clamp, low_rank_2d
from torchwright.ops.inout_nodes import create_input

# Non-uniform grid that motivated low_rank_2d (TODO.md pathological case).
_BP_X = [
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    -0.1,
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
]
_BP_Y = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]


def _build_atan_graph(rank: int, clamp_inputs: bool):
    """Build a minimal graph: optionally clamp inputs, then ``low_rank_2d``.

    The clamp-or-not split is what shows the bug is about the
    interaction between surrounding structure and the fanout, not
    ``low_rank_2d`` alone.
    """
    x = create_input("x", 1)
    y = create_input("y", 1)

    if clamp_inputs:
        x_in = clamp(x, _BP_X[0], _BP_X[-1])
        y_in = clamp(y, _BP_Y[0], _BP_Y[-1])
    else:
        x_in, y_in = x, y

    result = low_rank_2d(
        x_in,
        y_in,
        _BP_X,
        _BP_Y,
        lambda a, b: math.atan(a / b),
        rank=rank,
        name="atan_lr",
    )
    return Concatenate([result])


def test_low_rank_2d_bare_rank3_compiles():
    """``low_rank_2d(rank=3)`` with no surrounding structure must compile.

    Currently fails: ``No progress: 13 nodes remaining, 487 free
    columns``.  The only node upstream of ``low_rank_2d`` is a raw
    ``InputNode``, so the deadlock is coming from ``low_rank_2d``'s own
    internal graph shape — not from interaction with preceding ops.
    """
    out = _build_atan_graph(rank=3, clamp_inputs=False)
    compile_headless(out, d=512, d_head=32, max_layers=200, verbose=False)


def test_low_rank_2d_rank1_with_clamped_inputs_compiles():
    """K=1 with input clamps must compile — showing the bug isn't K-gated.

    Currently fails: ``RuntimeError: No progress: 4 nodes remaining,
    493 free columns``.
    """
    out = _build_atan_graph(rank=1, clamp_inputs=True)
    compile_headless(out, d=512, d_head=32, max_layers=20, verbose=False)


def test_low_rank_2d_rank3_with_clamped_inputs_compiles_at_shallow_depth():
    """K=3 with input clamps at ``max_layers=20`` must compile.

    This is the exact environment of the SORTED stage's
    ``_endpoint_to_column`` fixture — migrating it to ``low_rank_2d``
    triggers the deadlock at compile-time before any precision test
    fires.  Currently fails: ``No progress: 13 nodes remaining, 487
    free columns``.
    """
    out = _build_atan_graph(rank=3, clamp_inputs=True)
    compile_headless(out, d=512, d_head=32, max_layers=20, verbose=False)


def test_low_rank_2d_rank3_with_clamped_inputs_compiles_at_generous_depth():
    """K=3 with input clamps at ``max_layers=200`` must compile.

    Failing here proves the deadlock isn't a depth-budget issue — the
    scheduler gets stuck regardless of how much room it has.  Currently
    fails with the same ``13 nodes remaining`` signature.
    """
    out = _build_atan_graph(rank=3, clamp_inputs=True)
    compile_headless(out, d=512, d_head=32, max_layers=200, verbose=False)
