"""``trim_unused_heads`` compaction: interior dead heads are reclaimed.

A head whose output projection is entirely zero contributes exactly zero to
the residual stream whatever its Q/K/V hold, so removing it is sound.  Dead
heads arise interior, not just trailing: a zero-support ``Linear`` keeps one
floor head whose O block is zero (the support-aware charge skips zero-weight
chunks but floors at one head so the node stays inside the CP-SAT model —
``realization.linear_attn_chunks``), and heads from different ops interleave
within a sublayer.  The previous trim sliced ``[:used_heads]`` — a counter,
never the matrices — so interior dead heads survived, each costing
``4 * d * d_head`` parameters and ``d_head`` of KV-cache width per layer.

Compaction changes no schedule and recovers no layers (heads were charged at
schedule time); it is not bit-identical (reindexing perturbs the einsum
accumulation order at the ~1e-6 level).
"""

import torch

from torchwright.compiler.components.attn import AttnLayerComponent
from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.scheduling_policy import LEGACY_POLICY
from torchwright.graph import Concatenate, Linear
from torchwright.ops.inout_nodes import create_input

D, D_HEAD = 32, 8  # 4 head slots


def _component_with_interior_dead_head() -> AttnLayerComponent:
    """Heads 0 and 2 live, head 1 allocated but dead (O == 0), head 3 unused."""
    attn = AttnLayerComponent(D, D_HEAD)
    g = torch.Generator().manual_seed(0)
    for h in (0, 1, 2):
        attn.query_matrix[h] = torch.randn(D, D_HEAD, generator=g) * 0.1
        attn.key_matrix[h] = torch.randn(D, D_HEAD, generator=g) * 0.1
        attn.value_matrix[h] = torch.randn(D, D_HEAD, generator=g) * 0.1
        if h != 1:
            attn.output_matrix[h] = torch.randn(D_HEAD, D, generator=g) * 0.1
    attn.used_heads = 3
    return attn


def test_interior_dead_head_is_compacted():
    attn = _component_with_interior_dead_head()
    live0 = attn.output_matrix[0].clone()
    live2 = attn.output_matrix[2].clone()

    torch.manual_seed(1)
    inp = torch.randn(5, D)
    before = attn.forward(inp)

    attn.trim_unused_heads()

    assert attn.n_heads == 2  # head 1 (dead) and head 3 (unallocated) gone
    assert attn.used_heads == 2
    # Relative order of live heads preserved.
    assert torch.equal(attn.output_matrix[0], live0)
    assert torch.equal(attn.output_matrix[1], live2)

    after = attn.forward(inp)
    torch.testing.assert_close(after, before, rtol=0, atol=1e-5)


def test_all_dead_heads_keeps_one():
    """Degenerate case: every allocated head is dead — keep one zero head
    rather than emitting empty-tensor shapes.
    """
    attn = AttnLayerComponent(D, D_HEAD)
    attn.used_heads = 2  # allocated, but O left zero
    attn.trim_unused_heads()
    assert attn.n_heads == 1
    assert (attn.output_matrix == 0).all()


def test_trailing_trim_behavior_unchanged():
    """The old contract: no dead interior heads -> compaction equals the
    trailing slice.
    """
    attn = _component_with_interior_dead_head()
    attn.output_matrix[1] = torch.randn(D_HEAD, D) * 0.1  # make head 1 live too
    attn.trim_unused_heads()
    assert attn.n_heads == 3  # heads 0..2 live, head 3 unallocated


def test_compiled_zero_support_floor_head_is_compacted():
    """Compile-level: a zero-support Linear keeps one floor head whose O
    block is zero — the interior dead head the support-aware emitter still
    produces (a merely *sparse* Linear no longer emits dead heads at all:
    ``linear_attn_chunks`` skips its zero chunks).  Compaction reclaims it
    and the compiled values still match node.compute.
    """
    torch.manual_seed(0)
    x = create_input("x", 2 * D_HEAD, value_range=(-1.0, 1.0))
    # `other` reads its own input: same-input siblings would merge
    # (test_sibling_linear_fusion), pooling the zero rows with live ones and
    # erasing the dead head this test needs.
    y = create_input("y", D_HEAD, value_range=(-1.0, 1.0))
    # Zero weights, nonzero bias: the value is written by compute_bias (MLP),
    # so the value check stays nontrivial while the floor head's O == 0.
    lin = Linear(
        x, torch.zeros(2 * D_HEAD, 3), torch.tensor([1.5, -0.5, 0.25]), name="allzero"
    )
    other = Linear(y, torch.randn(D_HEAD, 2) * 0.2, name="other")
    out = Concatenate([lin, other])

    def compile_(trim):
        return forward_compile(
            d=64,
            d_head=D_HEAD,
            output_node=out,
            verbose=False,
            device="cpu",
            policy=LEGACY_POLICY,
            trim_heads=trim,
        )

    untrimmed = compile_(False)

    def dead_allocated_heads(net):
        return sum(
            int(net_layer.attn.attn.output_matrix[h].eq(0).all())
            for net_layer in net.layers
            for h in range(net_layer.attn.attn.used_heads)
        )

    # The premise the support-aware emitter did NOT remove: the floor head
    # is allocated and dead.
    assert dead_allocated_heads(untrimmed) >= 1

    trimmed = compile_(True)
    heads = lambda net: sum(layer.attn.attn.n_heads for layer in net.layers)
    assert heads(trimmed) < heads(untrimmed)

    vals = {
        "x": torch.rand(3, 2 * D_HEAD) * 2 - 1,
        "y": torch.rand(3, D_HEAD) * 2 - 1,
    }
    got = trimmed.compute(3, vals)
    for leaf in (lin, other):
        torch.testing.assert_close(
            got[leaf].cpu(), leaf.compute(3, vals), rtol=0, atol=1e-4
        )
