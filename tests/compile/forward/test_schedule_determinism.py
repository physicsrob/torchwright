"""Scheduling determinism across fresh graph rebuilds (D6 for the sweep).

Two builds of the same construction code in one process mint different
absolute ``node_id`` values (the counter keeps running), which used to
change node-set iteration order and critical-path tie resolution — the
schedule was only stable when the *same graph object* was recompiled.
The determinism sweep (docs/lowering_copy_plan.md, decision 4) sorts
node collections by ``node_id`` at the order-sensitive sites and
tie-breaks the critical-path sorts, so any two graphs with the same
topology and the same *relative* id order schedule identically.
"""

import torch

import torchwright.graph.node as _node_module
from torchwright.compiler.export import _build_matrix_occupancy
from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.graph_identity import canonical_ids
from torchwright.graph import Linear
from torchwright.graph.misc import Add, Concatenate, LiteralValue
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

D = 64
D_HEAD = 8


def _w(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * 0.3


def _build_graph():
    """Parallel same-depth branches → critical-path ties the sweep must break
    identically across rebuilds.  Weights are seeded per-site so both builds
    carry identical parameters (schedule inputs must match exactly)."""
    a = create_input("a", 2, value_range=(-1.0, 1.0))
    b = create_input("b", 2, value_range=(-1.0, 1.0))
    c = create_input("c", 2, value_range=(-1.0, 1.0))
    l1 = Linear(a, _w(2, 3, seed=1), _w(3, seed=2), name="l1")
    l2 = Linear(b, _w(2, 3, seed=3), _w(3, seed=4), name="l2")
    l3 = Linear(c, _w(2, 3, seed=5), name="l3")  # zero bias variant
    lit = LiteralValue(torch.tensor([0.25, -0.5, 1.0]), name="lit")
    add1 = Add(l1, l2, name="add1")
    add2 = Add(l3, lit, name="add2")
    blk = linear_relu_linear(
        add1,
        _w(4, 3, seed=6),
        _w(4, seed=7),
        _w(4, 3, seed=8),
        _w(3, seed=9),
        name="blk",
    )
    return Linear(Concatenate([blk, add2]), _w(6, 4, seed=10), _w(4, seed=11))


def _compile_fingerprint(out):
    net = forward_compile(
        d=D, d_head=D_HEAD, output_node=out, verbose=False, device="cpu"
    )
    canon = canonical_ids(out)

    matrices, placements, n_heads, d_hidden = _build_matrix_occupancy(
        net, canon, D, D_HEAD
    )

    ra = net.residual_assignment
    states = [("input", net.layers[0].attn.in_state)]
    for i, layer in enumerate(net.layers):
        states.append((f"L{i}.attn", layer.attn.out_state))
        states.append((f"L{i}.mlp", layer.mlp.out_state))
    ra_by_canon = {}
    for key, st in states:
        table = ra.mapping.get(st)
        if table is None:
            continue
        ra_by_canon[key] = {
            canon[n.node_id]: list(cols)
            for n, cols in table.items()
            if n.node_id in canon
        }

    return {
        "n_layers": len(net.layers),
        "matrices": matrices,
        "placements": placements,
        "n_heads": n_heads,
        "d_hidden": d_hidden,
        "ra": ra_by_canon,
    }


def test_two_fresh_rebuilds_schedule_identically():
    first = _compile_fingerprint(_build_graph())

    # Shift the id counter by an odd offset so the second build's absolute
    # ids differ AND land in different hash-table slots — the exact
    # perturbation that used to reorder node sets.
    _node_module.global_node_id += 13

    second = _compile_fingerprint(_build_graph())

    for key in ("n_layers", "n_heads", "d_hidden", "matrices", "placements", "ra"):
        assert first[key] == second[key], f"{key} differs across fresh rebuilds"
