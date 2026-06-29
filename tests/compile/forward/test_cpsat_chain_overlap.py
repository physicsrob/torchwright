"""Regression tests for fusion-induced chain overlap.

`fuse_consecutive_linears` mutates a downstream Linear's input to
point at an upstream ReLU.  Before the de-dup fix, a single Linear
could then satisfy both the L2 condition for one chain and the L1
condition for the next:

    ... -> R -> L_a -> L_b -> R' -> L_c -> ...

After fusing the consecutive Linears `(L_a, L_b)` (which become a
single fused Linear, identity-preserving on `L_b`):

    ... -> R -> L_b -> R' -> L_c -> ...

The chain detector then emitted two chains:

    chain 0: (L_pre, R,  L_b)   # L_b is L2
    chain 1: (L_b,  R', L_c)    # L_b is L1

The CP-SAT chain-coupling equalities (`layer[L1]==layer[R]==layer[L2]`)
combined with the strict dependency `R < R'` made the model
`INFEASIBLE` in presolve, silently masked by the heuristic-fallback
path.
"""

import pytest
import torch

from torchwright.compiler.forward.cpsat_scheduler import (
    build_graph_model,
    solve_schedule,
)
from torchwright.compiler.forward.scheduler import LayerScheduler
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.graph import Linear
from torchwright.graph.optimize import fuse_consecutive_linears
from torchwright.graph.relu import ReLU
from torchwright.ops.inout_nodes import create_input


def _build_post_fusion_repro():
    """Build the minimal post-fusion topology that triggered the bug.

    Pre-fusion:  x -> L_pre -> R -> L_a -> L_b -> R' -> L_c
    Post-fusion: x -> L_pre -> R -> L_b' -> R' -> L_c
                 (where L_b' = fused(L_a, L_b), input rewired to R)
    """
    torch.manual_seed(0)
    x = create_input("x", 8)
    L_pre = Linear(x, torch.randn(8, 16), torch.zeros(16), name="L_pre")
    R = ReLU(L_pre, name="R")
    L_a = Linear(R, torch.randn(16, 12), torch.zeros(12), name="L_a")
    L_b = Linear(L_a, torch.randn(12, 16), torch.zeros(16), name="L_b")
    R_prime = ReLU(L_b, name="R_prime")
    L_c = Linear(R_prime, torch.randn(16, 4), torch.zeros(4), name="L_c")
    return x, L_pre, R, L_a, L_b, R_prime, L_c


def test_chain_detector_no_overlap_after_fusion():
    """No node appears in more than one chain after `fuse_consecutive_linears`."""
    x, L_pre, R, L_a, L_b, R_prime, L_c = _build_post_fusion_repro()

    n_fused = fuse_consecutive_linears({L_c})
    assert n_fused == 1, f"expected 1 fusion, got {n_fused}"
    # Sanity: post-fusion, L_b's input is now R (not L_a).
    assert L_b.inputs[0] is R, (
        f"expected L_b.inputs[0] to be R after fusion, got " f"{L_b.inputs[0]!r}"
    )

    gm = build_graph_model(L_c)

    # The structural assertion in build_graph_model would have raised
    # if a node showed up in two chains.  Spot-check it ourselves so
    # the test still says something meaningful if the assertion is
    # ever weakened.
    seen = {}
    for c in gm.chains:
        for role, node in (("L1", c.l1), ("R", c.relu), ("L2", c.l2)):
            assert node not in seen, (
                f"{node!r} is {seen[node]!r} of one chain and "
                f"{(role, c.chain_id)!r} of another"
            )
            seen[node] = (role, c.chain_id)


def test_chain_detector_picks_upstream_chain():
    """When two chains overlap on a Linear, the upstream chain wins.

    Iteration order in the detector is `node_id`, which corresponds
    to graph build order — the upstream Linear has the smaller id,
    so its chain is detected first and the shared Linear gets locked
    into its L2 role.  The downstream chain doesn't form, and its
    Linears fall back to standalone scheduling.
    """
    x, L_pre, R, L_a, L_b, R_prime, L_c = _build_post_fusion_repro()
    fuse_consecutive_linears({L_c})

    gm = build_graph_model(L_c)

    # Exactly one chain — the upstream one.  The downstream chain
    # would have been (L_b, R', L_c); since L_b is locked as L2 of
    # the upstream chain, that detection is suppressed.
    assert len(gm.chains) == 1
    chain = gm.chains[0]
    assert chain.l1 is L_pre
    assert chain.relu is R
    assert chain.l2 is L_b
    # R' and L_c fall outside any chain.
    assert R_prime not in gm.node_to_chain
    assert L_c not in gm.node_to_chain


def test_cpsat_solves_post_fusion_repro():
    """The post-fusion repro is CP-SAT-feasible (was INFEASIBLE pre-fix)."""
    x, L_pre, R, L_a, L_b, R_prime, L_c = _build_post_fusion_repro()
    fuse_consecutive_linears({L_c})

    assignment, stats = solve_schedule(
        L_c,
        d=64,
        d_head=8,
        d_hidden=128,
        time_budget_s=10.0,
        max_layers=20,
    )
    assert stats.status_name in ("OPTIMAL", "FEASIBLE"), (
        f"expected OPTIMAL or FEASIBLE, got {stats.status_name}.  "
        f"INFEASIBLE here would mean the chain de-dup regressed and "
        f"the chain-coupling equalities are again contradicting the "
        f"R < R' dependency."
    )
    assert assignment is not None


def test_heuristic_chain_detector_no_overlap_after_fusion():
    """Same de-dup applies to the heuristic detector for parity.

    The heuristic dodges the bug in practice via per-layer
    ``ready.discard`` after a chain is scheduled, but the underlying
    detector has the same gap.  Apply the de-dup so that any future
    code path that calls ``_detect_chains`` on a `ready` set
    containing both upstream and downstream Linears simultaneously
    doesn't reintroduce the overlap.
    """
    x, L_pre, R, L_a, L_b, R_prime, L_c = _build_post_fusion_repro()
    fuse_consecutive_linears({L_c})

    graph = GraphAnalyzer(L_c)
    scheduler = LayerScheduler(graph, d=64, d_head=8)
    # Build a ready set that includes both L_pre and L_b — the case
    # where the bug would fire if the de-dup were missing.
    ready = {L_pre, L_b, L_c}
    chains = scheduler._detect_chains(ready)

    # Only the upstream chain should form.
    assert len(chains) == 1
    l1, relu, l2, d_hidden, exclusive = chains[0]
    assert l1 is L_pre and relu is R and l2 is L_b
