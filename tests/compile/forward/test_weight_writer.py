"""TDD tests for the forward compiler's WeightWriter.

Each test builds a small graph, sets up a residual stream with known values,
writes weights into one TransformerLayer, runs the forward pass, and verifies
the output matches node.compute().

Conventions:
- n_pos > 1 for all attention tests to exercise causal mask behavior.
- For add_into: the Add node must be constructed as Add(dead_addend, live_addend).
  inputs[0] is the dead addend whose columns (target_cols) are being reused.
  inputs[1] is the live addend whose values are copied via attention.
  The scheduler (Phase 3) enforces this ordering.
"""

from typing import cast, Literal

import torch
import torch.nn.functional as F

import torchwright.compiler.device as device_mod
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.weight_writer import (
    AttnHeadOp,
    MLPOp,
    write_attn_sublayer,
    write_mlp_sublayer,
)
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.graph import Linear, Attn, Add, Concatenate
from torchwright.ops.linear_relu_linear import linear_relu_linear
from torchwright.graph.misc import InputNode, LiteralValue

D = 64
D_HEAD = 16
N_POS = 4
# Old PosEncoding constant — the self-match query gain.  Kept locally now that
# graph/pos_encoding.py is gone (RoPE: position is a rotation, not a node).
attention_hardness = 100.0


def _make_reserved_block():
    # Generic 17-wide always-allocated stand-in for the reserved columns a
    # layer keeps (under RoPE there is no PosEncoding; the attention tests read
    # this as a plain Q/K source, the transport tests use it only as filler).
    return InputNode("reserved", 17, value_range=(-1.0, 1.0))


def _const_one(residual_map: ResidualStreamMap) -> LiteralValue:
    """Allocate the reserved constant-1 self-match column the transport ops
    (compute_linear/compute_add/cancel/add_into) read.  ``write_attn_sublayer``
    requires it; :func:`_build_residual_stream` fills its column with 1.0 so the
    rotary Δ=0 self-match concentrates on the diagonal."""
    const_one = LiteralValue(torch.ones(1), name="const_one")
    residual_map.allocate(const_one)
    return const_one


def _build_residual_stream(
    residual_map: ResidualStreamMap, node_values: dict
) -> torch.Tensor:
    """Build a residual stream tensor with known values at each node's columns."""
    device = device_mod.get_device(verbose=False)
    res = torch.zeros(N_POS, D, device=device)
    # The reserved const-1 self-match column must carry 1.0 (the transport ops'
    # rotary self-match reads it); fill it automatically wherever allocated.
    for node in residual_map.get_allocated_nodes():
        if getattr(node, "name", "") == "const_one":
            for idx in residual_map.get_indices(node):
                res[:, idx] = 1.0
    for node, values in node_values.items():
        indices = residual_map.get_indices(node)
        values = values.to(res.device)
        for i, idx in enumerate(indices):
            res[:, idx] = values[:, i]
    return res


def _make_op(rmap: ResidualStreamMap, op_type: str, node, target_cols, **kwargs):
    """Construct an AttnHeadOp with source_cols captured from ``rmap``.

    The weight-writer requires source_cols to be populated at op-construction
    time (see weight_writer.AttnHeadOp docstring).  This helper resolves
    the right source indices for each op type so tests can be terse.
    """
    if op_type == "compute_attn":
        q_in, k_in, v_in = node.inputs
        kwargs.setdefault("q_source_cols", rmap.resolve_indices(q_in))
        kwargs.setdefault("k_source_cols", rmap.resolve_indices(k_in))
        kwargs.setdefault("source_cols", rmap.resolve_indices(v_in))
    elif op_type == "compute_linear":
        kwargs.setdefault("source_cols", rmap.resolve_indices(node.inputs[0]))
    elif op_type == "compute_add":
        a0, a1 = node.inputs
        kwargs.setdefault("source_cols", rmap.resolve_indices(a0))
        kwargs.setdefault("source_cols_b", rmap.resolve_indices(a1))
    elif op_type == "add_into":
        # Caller must specify which input is live via kwargs['source_cols']
        # or we infer: whichever is currently allocated.
        if "source_cols" not in kwargs:
            a0, a1 = node.inputs
            if rmap.is_allocated(a0) or isinstance(a0, Concatenate):
                kwargs["source_cols"] = rmap.resolve_indices(a0)
            else:
                kwargs["source_cols"] = rmap.resolve_indices(a1)
    return AttnHeadOp(
        op_type=cast(
            Literal[
                "compute_attn",
                "compute_linear",
                "compute_add",
                "cancel",
                "add_into",
            ],
            op_type,
        ),
        node=node,
        target_cols=target_cols,
        **kwargs,
    )


def _make_mlp_op(
    rmap: ResidualStreamMap, op_type: str, node, target_cols, mlp_slots=None, **kwargs
):
    """Construct an MLPOp with source_cols captured from ``rmap``."""
    if mlp_slots is None:
        mlp_slots = []
    if op_type == "compute_ffn":
        # node is the FFN; its input is the actual source
        kwargs.setdefault("source_cols", rmap.resolve_indices(node.inputs[0]))
    elif op_type == "compute_linear_bypass":
        kwargs.setdefault("source_cols", rmap.resolve_indices(node.inputs[0]))
    return MLPOp(
        op_type=cast(
            Literal[
                "compute_ffn",
                "compute_literal_value",
                "compute_bias",
                "compute_linear_bypass",
            ],
            op_type,
        ),
        node=node,
        target_cols=target_cols,
        mlp_slots=mlp_slots,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def test_identity_layer():
    """A layer with no ops written is pure identity via skip connections."""
    layer = TransformerLayer(D, D_HEAD)
    device = device_mod.get_device(verbose=False)
    layer.to(device)
    inp = torch.randn(N_POS, D, device=device)
    out = layer.attn.forward(inp)
    out = layer.mlp.forward(out)
    assert torch.allclose(inp, out, atol=1e-6)


# ---------------------------------------------------------------------------
# Attention — compute_attn
# ---------------------------------------------------------------------------


def test_attn_compute():
    """Compile a basic Attn node into one attention head."""
    pos = _make_reserved_block()
    value_in = InputNode("v", 4, value_range=(-100.0, 100.0))
    # Build an Attn node that does current-position attention and passes through value
    d_head = D_HEAD
    attn_node = Attn(
        query_in=pos,
        key_in=pos,
        value_in=value_in,
        query_matrix=attention_hardness * torch.eye(len(pos), d_head),
        key_matrix=torch.eye(len(pos), d_head),
        value_matrix=torch.eye(len(value_in), d_head),
        output_matrix=torch.eye(d_head, len(value_in)),
    )

    rmap = ResidualStreamMap(D)
    pos_cols = rmap.allocate(pos)
    v_cols = rmap.allocate(value_in)
    out_cols = rmap.allocate(attn_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "compute_attn", attn_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    # Build input residual stream
    v_values = torch.randn(N_POS, len(value_in))
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, value_in: v_values})

    # Run attention sublayer only (skip adds input, so output cols get 0 + attn_output)
    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = attn_node.compute(N_POS, {"v": v_values, "reserved": pe_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_attn_compute_small_d_v():
    """Attn node whose value width d_v is smaller than the layer d_head — the
    V/O projection is padded up to d_head.  (Q/K are full-width d_head: every
    head is rotary on the one global grid, so partial-width Q/K no longer
    exists.)"""
    pos = _make_reserved_block()
    value_in = InputNode("v", 4, value_range=(-100.0, 100.0))
    small_d_v = 8  # smaller than D_HEAD=16 -> V/O padded up to d_head

    attn_node = Attn(
        query_in=pos,
        key_in=pos,
        value_in=value_in,
        query_matrix=attention_hardness * torch.eye(len(pos), D_HEAD),
        key_matrix=torch.eye(len(pos), D_HEAD),
        value_matrix=torch.eye(len(value_in), small_d_v),
        output_matrix=torch.eye(small_d_v, len(value_in)),
    )

    rmap = ResidualStreamMap(D)
    pos_cols = rmap.allocate(pos)
    v_cols = rmap.allocate(value_in)
    out_cols = rmap.allocate(attn_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "compute_attn", attn_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    v_values = torch.randn(N_POS, len(value_in))
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, value_in: v_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = attn_node.compute(N_POS, {"v": v_values, "reserved": pe_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_attn_compute_shared_inputs():
    """Attn node where query_in == key_in (like attend_to_offset)."""
    pos = _make_reserved_block()
    value_in = InputNode("v", 4, value_range=(-100.0, 100.0))

    # Shared Q/K source: query_in and key_in are the SAME node (the weight
    # writer must capture one source for both sides — the structural property
    # the old attend_to_offset exercised).
    attn_node = Attn(
        query_in=pos,
        key_in=pos,
        value_in=value_in,
        query_matrix=attention_hardness * torch.eye(len(pos), D_HEAD),
        key_matrix=torch.eye(len(pos), D_HEAD),
        value_matrix=torch.eye(len(value_in), D_HEAD),
        output_matrix=torch.eye(D_HEAD, len(value_in)),
    )

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(value_in)
    out_cols = rmap.allocate(attn_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "compute_attn", attn_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    v_values = torch.randn(N_POS, len(value_in))
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, value_in: v_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = attn_node.compute(N_POS, {"v": v_values, "reserved": pe_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_attn_compute_multiposition():
    """Attn node with cross-position attention (get_prev_value pattern)."""
    value_in = InputNode("v", 4, value_range=(-100.0, 100.0))
    cond_in = InputNode("c", 1, value_range=(-100.0, 100.0))

    # Cross-position attention with a Concatenate key_in and a literal query
    # (the structural property the old get_prev_value exercised): the query is
    # a constant 1.0, the key is Concatenate([cond]), value passes through.
    # Under RoPE there is no position block in the key — position is a rotation,
    # not a key feature — so the old vestigial `pos` leaf is dropped.
    query_one = LiteralValue(torch.tensor([1.0]))
    key_in = Concatenate([cond_in])
    d_qk = D_HEAD
    query_matrix = torch.zeros(1, d_qk)
    query_matrix[0, 0] = attention_hardness
    key_matrix = torch.zeros(len(key_in), d_qk)
    key_matrix[0, 0] = 1.0  # gate on cond
    attn_node = Attn(
        query_in=query_one,
        key_in=key_in,
        value_in=value_in,
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=torch.eye(len(value_in), d_qk),
        output_matrix=torch.eye(d_qk, len(value_in)),
    )

    rmap = ResidualStreamMap(D)
    rmap.allocate(value_in)
    rmap.allocate(cond_in)
    rmap.allocate(query_one)
    out_cols = rmap.allocate(attn_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "compute_attn", attn_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    v_values = torch.tensor(
        [
            [10.0, 20.0, 30.0, 40.0],
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
            [13.0, 23.0, 33.0, 43.0],
        ]
    )
    c_values = torch.tensor([[1.0], [0.0], [0.0], [1.0]])

    # get_prev_value's key_in is Concatenate([cond]); the query is an exact 1.0
    # literal.  Place every leaf/constant the head reads.
    res = _build_residual_stream(
        rmap,
        {
            value_in: v_values,
            cond_in: c_values,
            query_one: query_one.compute(N_POS, {}),
        },
    )

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = attn_node.compute(N_POS, {"v": v_values, "c": c_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Attention — compute_linear (zero-bias)
# ---------------------------------------------------------------------------


def test_linear_zero_bias():
    """Zero-bias Linear compiled via current-position attention."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    W = torch.randn(4, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "compute_linear", linear_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_linear_large_input():
    """Zero-bias Linear with input dim > d_head — requires multiple attention heads.

    This is the sum_nodes pattern from the adder: Concatenate(4 × 8-dim) → Linear.
    With d_input=32 and d_head=16, needs ceil(32/16) = 2 heads.
    """
    pos = _make_reserved_block()
    # 4 inputs of 8 dims each, concatenated → 32-dim input
    inputs = [InputNode(f"x{i}", 8, value_range=(-100.0, 100.0)) for i in range(4)]
    cat = Concatenate(inputs)
    # Summing matrix: each output dim accumulates from all 4 inputs
    d_out = 8
    W = torch.zeros(32, d_out)
    for i in range(32):
        W[i, i % d_out] = 1.0
    linear_node = Linear(cat, W, torch.zeros(d_out), name="sum")

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    for inp in inputs:
        rmap.allocate(inp)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "compute_linear", linear_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    input_values = {f"x{i}": torch.randn(N_POS, 8) for i in range(4)}
    pe_values = torch.randn(N_POS, len(pos))
    node_values = {pos: pe_values}
    for inp in inputs:
        node_values[inp] = input_values[inp.name]
    res = _build_residual_stream(rmap, node_values)

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, input_values)
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_linear_different_dims():
    """Zero-bias Linear where input dim != output dim."""
    pos = _make_reserved_block()
    x = InputNode("x", 8, value_range=(-100.0, 100.0))
    W = torch.randn(8, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "compute_linear", linear_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 8)
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Attention — cancel
# ---------------------------------------------------------------------------


def test_cancel():
    """Cancel a node: columns should become zero after attn sublayer."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    x_cols = rmap.allocate(x)

    layer = TransformerLayer(D, D_HEAD)
    op = AttnHeadOp(op_type="cancel", node=x, target_cols=x_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    # After attn sublayer (includes skip): x + (-x) = 0
    out = layer.attn.forward(res)
    result = out[:, x_cols]

    assert torch.allclose(result.cpu(), torch.zeros_like(result.cpu()), atol=1e-4)


def test_cancel_multiple():
    """Cancel two nodes in the same layer using different heads."""
    pos = _make_reserved_block()
    a = InputNode("a", 3, value_range=(-100.0, 100.0))
    b = InputNode("b", 5, value_range=(-100.0, 100.0))

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    a_cols = rmap.allocate(a)
    b_cols = rmap.allocate(b)

    layer = TransformerLayer(D, D_HEAD)
    ops = [
        AttnHeadOp(op_type="cancel", node=a, target_cols=a_cols),
        AttnHeadOp(op_type="cancel", node=b, target_cols=b_cols),
    ]
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, ops, rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    a_values = torch.randn(N_POS, 3)
    b_values = torch.randn(N_POS, 5)
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, a: a_values, b: b_values})

    out = layer.attn.forward(res)
    assert torch.allclose(out[:, a_cols].cpu(), torch.zeros(N_POS, 3), atol=1e-4)
    assert torch.allclose(out[:, b_cols].cpu(), torch.zeros(N_POS, 5), atol=1e-4)


# ---------------------------------------------------------------------------
# Attention — add_into
# ---------------------------------------------------------------------------


def test_add_into():
    """Add(A, B) where A is dead — write B into A's columns via skip."""
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    # inputs[0]=a is dead (at target_cols), inputs[1]=b is live (copied via attention)
    add_node = Add(a, b)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    a_cols = rmap.allocate(a)  # A occupies these columns (will become Add result)
    rmap.allocate(b)

    # Simulate scheduler: reassign dead addend's columns to the Add node
    rmap.reassign(a, add_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "add_into", add_node, a_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    a_values = torch.randn(N_POS, 4)
    b_values = torch.randn(N_POS, 4)
    pe_values = torch.randn(N_POS, len(pos))
    # a's columns now belong to add_node, but still hold a's values
    res = _build_residual_stream(
        rmap, {pos: pe_values, add_node: a_values, b: b_values}
    )

    # After attn sublayer: A's columns get A + B (skip adds A, attn writes B)
    out = layer.attn.forward(res)
    result = out[:, a_cols]

    expected = add_node.compute(N_POS, {"a": a_values, "b": b_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_add_into_dead_at_inputs1():
    """Add(live, dead) where dead is inputs[1] — still works correctly.

    This matches the adder's cond_add_vector pattern: Add(inp, chain_output)
    where chain_output is dead but lives at inputs[1].
    """
    pos = _make_reserved_block()
    live = InputNode("live", 4, value_range=(-100.0, 100.0))
    dead = InputNode("dead", 4, value_range=(-100.0, 100.0))
    # Dead addend is inputs[1], not inputs[0]
    add_node = Add(live, dead)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(live)
    dead_cols = rmap.allocate(dead)

    # Simulate what the scheduler does: reassign dead's columns to the Add node
    rmap.reassign(dead, add_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "add_into", add_node, dead_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    live_values = torch.randn(N_POS, 4)
    dead_values = torch.randn(N_POS, 4)
    pe_values = torch.randn(N_POS, len(pos))
    # dead's columns now belong to add_node, but still hold dead's values
    res = _build_residual_stream(
        rmap, {pos: pe_values, live: live_values, add_node: dead_values}
    )

    out = layer.attn.forward(res)
    result = out[:, dead_cols]

    expected = add_node.compute(N_POS, {"live": live_values, "dead": dead_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Attention — compute_add
# ---------------------------------------------------------------------------


def test_compute_add():
    """Add(a, b) with neither input dead — copies both via separate heads."""
    pos = _make_reserved_block()
    a = InputNode("a", 4, value_range=(-100.0, 100.0))
    b = InputNode("b", 4, value_range=(-100.0, 100.0))
    add_node = Add(a, b)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(a)
    rmap.allocate(b)
    out_cols = rmap.allocate(add_node)

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "compute_add", add_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    a_values = torch.randn(N_POS, 4)
    b_values = torch.randn(N_POS, 4)
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, a: a_values, b: b_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = add_node.compute(N_POS, {"a": a_values, "b": b_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_compute_add_wide():
    """compute_add with vectors wider than d_head — requires multiple head groups."""
    pos = _make_reserved_block()
    # 20 > D_HEAD=16, so needs 2 heads per input (4 heads total)
    a = InputNode("a", 20, value_range=(-100.0, 100.0))
    b = InputNode("b", 20, value_range=(-100.0, 100.0))
    add_node = Add(a, b)

    d_wide = 128  # Need room for pos(16) + a(20) + b(20) + out(20)
    rmap = ResidualStreamMap(d_wide)
    rmap.allocate(pos)
    rmap.allocate(a)
    rmap.allocate(b)
    out_cols = rmap.allocate(add_node)

    layer = TransformerLayer(d_wide, D_HEAD)
    op = _make_op(rmap, "compute_add", add_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    a_values = torch.randn(N_POS, 20)
    b_values = torch.randn(N_POS, 20)
    pe_values = torch.randn(N_POS, len(pos))
    device = device_mod.get_device(verbose=False)
    res = torch.zeros(N_POS, d_wide, device=device)
    # const_one carries 1.0 so the transport's rotary self-match concentrates
    # (this test builds the stream by hand rather than via _build_residual_stream).
    for node, values in {
        pos: pe_values,
        a: a_values,
        b: b_values,
        const_one: torch.ones(N_POS, 1),
    }.items():
        indices = rmap.get_indices(node)
        values = values.to(res.device)
        for i, idx in enumerate(indices):
            res[:, idx] = values[:, i]

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = add_node.compute(N_POS, {"a": a_values, "b": b_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# MLP — compute_ffn
# ---------------------------------------------------------------------------


def test_mlp_block():
    """A degenerate-ReLU FFN compiled via MLP."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    ffn = linear_relu_linear(
        x,
        torch.randn(8, 4),
        torch.randn(8),
        torch.randn(8, 3),
        torch.randn(3),
        name="ffn",
    )

    rmap = ResidualStreamMap(D)
    x_cols = rmap.allocate(x)
    out_cols = rmap.allocate(ffn)

    mlp_slots = list(range(0, 8))  # 8 hidden slots for the 8-lane FFN

    layer = TransformerLayer(D, D_HEAD)
    op = _make_mlp_op(rmap, "compute_ffn", ffn, out_cols, mlp_slots=mlp_slots)
    write_mlp_sublayer(layer, [op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    res = _build_residual_stream(rmap, {x: x_values})

    # Run MLP sublayer only
    out = layer.mlp.forward(res)
    result = out[:, out_cols]

    expected = ffn.compute(N_POS, {"x": x_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_mlp_block_multiple():
    """Two FFNs in the same MLP, using different slot ranges."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    y = InputNode("y", 3, value_range=(-100.0, 100.0))

    ffn1 = linear_relu_linear(
        x,
        torch.randn(6, 4),
        torch.randn(6),
        torch.randn(6, 2),
        torch.randn(2),
        name="ffn1",
    )
    ffn2 = linear_relu_linear(
        y,
        torch.randn(5, 3),
        torch.randn(5),
        torch.randn(5, 2),
        torch.randn(2),
        name="ffn2",
    )

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    rmap.allocate(y)
    out1_cols = rmap.allocate(ffn1)
    out2_cols = rmap.allocate(ffn2)

    layer = TransformerLayer(D, D_HEAD)
    ops = [
        _make_mlp_op(
            rmap, "compute_ffn", ffn1, out1_cols, mlp_slots=list(range(0, 6))
        ),
        _make_mlp_op(
            rmap, "compute_ffn", ffn2, out2_cols, mlp_slots=list(range(6, 11))
        ),
    ]
    write_mlp_sublayer(layer, ops, rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    y_values = torch.randn(N_POS, 3)
    res = _build_residual_stream(rmap, {x: x_values, y: y_values})

    out = layer.mlp.forward(res)

    expected1 = ffn1.compute(N_POS, {"x": x_values})
    expected2 = ffn2.compute(N_POS, {"y": y_values})
    assert torch.allclose(out[:, out1_cols].cpu(), expected1, atol=1e-4)
    assert torch.allclose(out[:, out2_cols].cpu(), expected2, atol=1e-4)


# ---------------------------------------------------------------------------
# MLP — compute_literal_value
# ---------------------------------------------------------------------------


def test_mlp_constant():
    """LiteralValue written via MLP output bias."""
    const_value = torch.tensor([1.0, -2.0, 3.5])
    const = LiteralValue(const_value)

    rmap = ResidualStreamMap(D)
    out_cols = rmap.allocate(const)

    layer = TransformerLayer(D, D_HEAD)
    op = MLPOp(
        op_type="compute_literal_value", node=const, target_cols=out_cols, mlp_slots=[]
    )
    write_mlp_sublayer(layer, [op], rmap)
    device = device_mod.get_device(verbose=False)
    layer.to(device)

    res = torch.zeros(N_POS, D, device=device)

    out = layer.mlp.forward(res)
    result = out[:, out_cols]

    expected = const.compute(N_POS, {})
    assert torch.allclose(result.cpu(), expected, atol=1e-6)


def test_compute_literal_value_clears_dirty_column():
    """A constant materialized into a column that still holds a dead node's
    leftover value must come out equal to the constant — not constant+leftover.

    Just-in-time materialization makes constants reuse *recycled* residual
    columns (previously every constant owned a fresh layer-0 column, so this
    path was never exercised — the investigation flagged it).  The birth
    dirty-cancel (attn sublayer, ``x + (-x) = 0``) must zero the column before
    the MLP output-bias write adds the constant.  Reproduced deterministically:
    seed the target column with non-zero garbage, schedule cancel +
    compute_literal_value on it, and assert the result is exactly the constant.
    """
    const_value = torch.tensor([5.0, -7.0, 3.0])
    const = LiteralValue(const_value)

    pos = _make_reserved_block()
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    const_cols = rmap.allocate(const)

    layer = TransformerLayer(D, D_HEAD)
    const_one = _const_one(rmap)
    # Attn sublayer: dirty-cancel the recycled target columns.
    write_attn_sublayer(
        layer,
        [AttnHeadOp(op_type="cancel", node=const, target_cols=const_cols)],
        rmap,
        const_one=const_one,
    )
    # MLP sublayer: write the constant via output bias.
    write_mlp_sublayer(
        layer,
        [
            MLPOp(
                op_type="compute_literal_value",
                node=const,
                target_cols=const_cols,
                mlp_slots=[],
            )
        ],
        rmap,
    )
    layer.to(device_mod.get_device(verbose=False))

    # Seed the target columns with a dead node's non-zero leftover value.
    garbage = torch.tensor([[111.0, -222.0, 333.0]]).repeat(N_POS, 1)
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, const: garbage})

    out = layer.attn.forward(res)  # cancel zeroes the dirty column
    out = layer.mlp.forward(out)  # bias write adds the constant onto zero

    result = out[:, const_cols].cpu()
    expected = const_value.unsqueeze(0).repeat(N_POS, 1)
    assert torch.allclose(result, expected, atol=1e-4), (
        f"dirty column not cleared before bias write: got {result[0].tolist()}, "
        f"expected {const_value.tolist()}"
    )


# ---------------------------------------------------------------------------
# Biased Linear split (attention Wx + MLP b)
# ---------------------------------------------------------------------------


def test_biased_linear_split():
    """Linear with non-zero bias: attention computes Wx, MLP adds b."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    W = torch.randn(4, 3)
    b = torch.randn(3)
    linear_node = Linear(x, W, b, name="biased")

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD)

    # Attention writes Wx (zero-bias part)
    attn_op = _make_op(rmap, "compute_linear", linear_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [attn_op], rmap, const_one=const_one)

    # MLP adds bias
    mlp_op = MLPOp(
        op_type="compute_bias", node=linear_node, target_cols=out_cols, mlp_slots=[]
    )
    write_mlp_sublayer(layer, [mlp_op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    # Run full layer: attn sublayer then mlp sublayer
    out = layer.attn.forward(res)
    out = layer.mlp.forward(out)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Non-contiguous columns
# ---------------------------------------------------------------------------


def test_non_contiguous_columns():
    """Operations work with scattered (non-contiguous) column indices."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    W = torch.randn(4, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    # Manually create a residual map and force non-contiguous allocation
    # by allocating and freeing intermediate nodes
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)  # takes first 16 cols
    dummy1 = InputNode("d1", 2, value_range=(-100.0, 100.0))
    rmap.allocate(x)  # takes next 4
    d1_cols = rmap.allocate(dummy1)  # takes next 2
    out_cols = rmap.allocate(linear_node)  # takes next 3
    rmap.free(dummy1)  # frees 2 cols in the middle

    # Verify output cols are non-contiguous with input cols
    # (they're in different regions of the stream)
    assert set(rmap.get_indices(x)) & set(out_cols) == set()

    layer = TransformerLayer(D, D_HEAD)
    op = _make_op(rmap, "compute_linear", linear_node, out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [op], rmap, const_one=const_one)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    out = layer.attn.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Mixed layer (attn + MLP together)
# ---------------------------------------------------------------------------


def test_mixed_layer():
    """One layer with both attention ops and MLP ops, verifying composition."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))

    # Attention: zero-bias linear
    W_attn = torch.randn(4, 3)
    lin_attn = Linear(x, W_attn, torch.zeros(3), name="lin_attn")

    # MLP: constant
    const_value = torch.tensor([7.0, -3.0])
    const = LiteralValue(const_value)

    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.allocate(x)
    attn_out_cols = rmap.allocate(lin_attn)
    const_cols = rmap.allocate(const)

    layer = TransformerLayer(D, D_HEAD)

    # Write attention ops
    attn_op = _make_op(rmap, "compute_linear", lin_attn, attn_out_cols)
    const_one = _const_one(rmap)
    write_attn_sublayer(layer, [attn_op], rmap, const_one=const_one)

    # Write MLP ops
    mlp_op = MLPOp(
        op_type="compute_literal_value",
        node=const,
        target_cols=const_cols,
        mlp_slots=[],
    )
    write_mlp_sublayer(layer, [mlp_op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    pe_values = torch.randn(N_POS, len(pos))
    res = _build_residual_stream(rmap, {pos: pe_values, x: x_values})

    # Run full layer
    out = layer.attn.forward(res)
    out = layer.mlp.forward(out)

    # Check attention result
    expected_attn = lin_attn.compute(N_POS, {"x": x_values})
    assert torch.allclose(out[:, attn_out_cols].cpu(), expected_attn, atol=1e-4)

    # Check MLP result
    expected_const = const.compute(N_POS, {})
    assert torch.allclose(out[:, const_cols].cpu(), expected_const, atol=1e-4)


# ---------------------------------------------------------------------------
# MLP — compute_linear_bypass
# ---------------------------------------------------------------------------


def test_mlp_linear_bypass_zero_bias():
    """Zero-bias Linear compiled via MLP bypass: ReLU(Wx) - ReLU(-Wx) = Wx."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    W = torch.randn(4, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    mlp_slots = list(range(0, 2 * 3))  # 2 slots per output column

    layer = TransformerLayer(D, D_HEAD)
    op = _make_mlp_op(
        rmap, "compute_linear_bypass", linear_node, out_cols, mlp_slots=mlp_slots
    )
    write_mlp_sublayer(layer, [op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.mlp.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_mlp_linear_bypass_with_bias():
    """Linear with non-zero bias compiled via MLP bypass."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    W = torch.randn(4, 3)
    b = torch.randn(3)
    linear_node = Linear(x, W, b, name="biased_lin")

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    mlp_slots = list(range(0, 2 * 3))

    layer = TransformerLayer(D, D_HEAD)
    op = _make_mlp_op(
        rmap, "compute_linear_bypass", linear_node, out_cols, mlp_slots=mlp_slots
    )
    write_mlp_sublayer(layer, [op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.mlp.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_mlp_linear_bypass_different_dims():
    """Linear where d_input != d_output, compiled via MLP bypass."""
    x = InputNode("x", 8, value_range=(-100.0, 100.0))
    W = torch.randn(8, 3)
    b = torch.randn(3)
    linear_node = Linear(x, W, b, name="narrow")

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    mlp_slots = list(range(0, 2 * 3))

    layer = TransformerLayer(D, D_HEAD)
    op = _make_mlp_op(
        rmap, "compute_linear_bypass", linear_node, out_cols, mlp_slots=mlp_slots
    )
    write_mlp_sublayer(layer, [op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 8)
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.mlp.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_mlp_linear_bypass_preserves_input():
    """MLP bypass doesn't corrupt the input node's columns."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    W = torch.randn(4, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    rmap = ResidualStreamMap(D)
    x_cols = rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    mlp_slots = list(range(0, 2 * 3))

    layer = TransformerLayer(D, D_HEAD)
    op = _make_mlp_op(
        rmap, "compute_linear_bypass", linear_node, out_cols, mlp_slots=mlp_slots
    )
    write_mlp_sublayer(layer, [op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.mlp.forward(res)

    assert torch.allclose(out[:, x_cols].cpu(), x_values, atol=1e-4)


def test_mlp_linear_bypass_wide_output():
    """Linear with output wider than input, compiled via MLP bypass."""
    x = InputNode("x", 3, value_range=(-100.0, 100.0))
    W = torch.randn(3, 8)
    b = torch.randn(8)
    linear_node = Linear(x, W, b, name="wide")

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    mlp_slots = list(range(0, 2 * 8))

    layer = TransformerLayer(D, D_HEAD)
    op = _make_mlp_op(
        rmap, "compute_linear_bypass", linear_node, out_cols, mlp_slots=mlp_slots
    )
    write_mlp_sublayer(layer, [op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 3)
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.mlp.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


def test_mlp_linear_bypass_matches_attention():
    """Same Linear produces identical result via MLP bypass vs attention path."""
    pos = _make_reserved_block()
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    W = torch.randn(4, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    x_values = torch.randn(N_POS, 4)
    pe_values = torch.randn(N_POS, len(pos))

    # Attention path
    rmap_attn = ResidualStreamMap(D)
    rmap_attn.allocate(pos)
    rmap_attn.allocate(x)
    attn_out_cols = rmap_attn.allocate(linear_node)

    layer_attn = TransformerLayer(D, D_HEAD)
    attn_op = _make_op(rmap_attn, "compute_linear", linear_node, attn_out_cols)
    const_one = _const_one(rmap_attn)
    write_attn_sublayer(layer_attn, [attn_op], rmap_attn, const_one=const_one)
    layer_attn.to(device_mod.get_device(verbose=False))

    res_attn = _build_residual_stream(rmap_attn, {pos: pe_values, x: x_values})
    out_attn = layer_attn.attn.forward(res_attn)
    result_attn = out_attn[:, attn_out_cols]

    # MLP bypass path
    rmap_mlp = ResidualStreamMap(D)
    rmap_mlp.allocate(x)
    mlp_out_cols = rmap_mlp.allocate(linear_node)

    layer_mlp = TransformerLayer(D, D_HEAD)
    mlp_slots = list(range(0, 2 * 3))
    mlp_op = _make_mlp_op(
        rmap_mlp,
        "compute_linear_bypass",
        linear_node,
        mlp_out_cols,
        mlp_slots=mlp_slots,
    )
    write_mlp_sublayer(layer_mlp, [mlp_op], rmap_mlp)
    layer_mlp.to(device_mod.get_device(verbose=False))

    res_mlp = _build_residual_stream(rmap_mlp, {x: x_values})
    out_mlp = layer_mlp.mlp.forward(res_mlp)
    result_mlp = out_mlp[:, mlp_out_cols]

    assert torch.allclose(result_attn.cpu(), result_mlp.cpu(), atol=1e-4)


def test_mlp_linear_bypass_concatenated_input():
    """Linear with Concatenate input compiled via MLP bypass."""
    a = InputNode("a", 3, value_range=(-100.0, 100.0))
    b = InputNode("b", 5, value_range=(-100.0, 100.0))
    cat = Concatenate([a, b])
    W = torch.randn(8, 4)
    bias = torch.randn(4)
    linear_node = Linear(cat, W, bias, name="cat_lin")

    rmap = ResidualStreamMap(D)
    rmap.allocate(a)
    rmap.allocate(b)
    out_cols = rmap.allocate(linear_node)

    mlp_slots = list(range(0, 2 * 4))

    layer = TransformerLayer(D, D_HEAD)
    op = _make_mlp_op(
        rmap, "compute_linear_bypass", linear_node, out_cols, mlp_slots=mlp_slots
    )
    write_mlp_sublayer(layer, [op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    a_values = torch.randn(N_POS, 3)
    b_values = torch.randn(N_POS, 5)
    res = _build_residual_stream(rmap, {a: a_values, b: b_values})

    out = layer.mlp.forward(res)
    result = out[:, out_cols]

    expected = linear_node.compute(N_POS, {"a": a_values, "b": b_values})
    assert torch.allclose(result.cpu(), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Swish machine (gated MLP sublayer) — swiglu plan A2
# ---------------------------------------------------------------------------


def _swish_ffn(x, n_lanes=6, d_out=3, gated=True, seed=0):
    """A directly-authored swish FFN fixture (the spec's constructions are
    op-level; writer tests need only the node shape)."""
    from torchwright.graph import FFN

    g = torch.Generator().manual_seed(seed)
    kwargs = {}
    if gated:
        kwargs["up_proj"] = torch.randn(n_lanes, len(x), generator=g)
        kwargs["up_bias"] = torch.randn(n_lanes, generator=g)
    return FFN(
        x,
        gate_proj=torch.randn(n_lanes, len(x), generator=g),
        gate_bias=torch.randn(n_lanes, generator=g),
        out_proj=torch.randn(n_lanes, d_out, generator=g),
        out_bias=torch.randn(d_out, generator=g),
        activation="swish",
        name="swish_ffn",
        **kwargs,
    )


def test_swish_mlp_ffn_gated():
    """A gated swish FFN compiled through the gated MLP sublayer."""
    x = InputNode("x", 4, value_range=(-10.0, 10.0))
    ffn = _swish_ffn(x, gated=True)

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    out_cols = rmap.allocate(ffn)

    layer = TransformerLayer(D, D_HEAD, activation="swish")
    op = _make_mlp_op(rmap, "compute_ffn", ffn, out_cols, mlp_slots=list(range(6)))
    write_mlp_sublayer(layer, [op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.mlp.forward(res)
    expected = ffn.compute(N_POS, {"x": x_values})
    assert torch.allclose(out[:, out_cols].cpu(), expected, atol=1e-4)


def test_swish_mlp_ffn_degenerate():
    """A degenerate swish FFN (up ≡ 1): the writer emits up-row 0 / up-bias 1
    and the compiled value matches the node's bare-swish math."""
    x = InputNode("x", 4, value_range=(-10.0, 10.0))
    ffn = _swish_ffn(x, gated=False)
    slots = list(range(3, 9))  # off-origin slots to catch indexing slips

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    out_cols = rmap.allocate(ffn)

    layer = TransformerLayer(D, D_HEAD, activation="swish")
    op = _make_mlp_op(rmap, "compute_ffn", ffn, out_cols, mlp_slots=slots)
    write_mlp_sublayer(layer, [op], rmap)

    # The degenerate up factor: bias 1, matrix column untouched (zero).
    assert (layer.mlp.up_proj.output_bias[slots] == 1.0).all()
    assert (layer.mlp.up_proj.output_matrix[:, slots] == 0.0).all()

    layer.to(device_mod.get_device(verbose=False))
    x_values = torch.randn(N_POS, 4)
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.mlp.forward(res)
    expected = ffn.compute(N_POS, {"x": x_values})
    assert torch.allclose(out[:, out_cols].cpu(), expected, atol=1e-4)


def test_swish_mlp_ffn_deferred_bias_fold():
    """A gated FFN reading a deferred-bias Linear folds the leaf bias into
    BOTH hidden biases (gate and up) — the up matmul reads the biasless
    columns too."""
    x = InputNode("x", 4, value_range=(-10.0, 10.0))
    W_leaf = torch.randn(4, 5)
    b_leaf = torch.randn(5)
    leaf = Linear(x, W_leaf, b_leaf, name="biased_leaf")
    ffn = _swish_ffn(leaf, n_lanes=6, gated=True, seed=1)

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    rmap.allocate(leaf)
    out_cols = rmap.allocate(ffn)

    layer = TransformerLayer(D, D_HEAD, activation="swish")
    op = _make_mlp_op(rmap, "compute_ffn", ffn, out_cols, mlp_slots=list(range(6)))
    write_mlp_sublayer(layer, [op], rmap, biased_linears={leaf})
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    # The deferred-bias contract: the leaf's columns hold Wx WITHOUT the bias
    # when the FFN reads them in the same layer.
    leaf_biasless = x_values @ W_leaf
    res = _build_residual_stream(rmap, {x: x_values, leaf: leaf_biasless})

    out = layer.mlp.forward(res)
    expected = ffn.compute(N_POS, {"x": x_values})  # oracle sees the full bias
    assert torch.allclose(out[:, out_cols].cpu(), expected, atol=1e-4)


def test_swish_mlp_linear_bypass():
    """The swish bypass pair: Swish(scale·z)/scale − Swish(−scale·z)/scale = z."""
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    W = torch.randn(4, 3)
    linear_node = Linear(x, W, torch.zeros(3), name="lin")

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD, activation="swish")
    op = _make_mlp_op(
        rmap, "compute_linear_bypass", linear_node, out_cols, mlp_slots=list(range(6))
    )
    write_mlp_sublayer(layer, [op], rmap)

    # Both slot halves are degenerate lanes on the swish machine.
    assert (layer.mlp.up_proj.output_bias[list(range(6))] == 1.0).all()

    layer.to(device_mod.get_device(verbose=False))
    x_values = torch.randn(N_POS, 4)
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.mlp.forward(res)
    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(out[:, out_cols].cpu(), expected, atol=1e-4)


def test_swish_mlp_linear_bypass_with_bias():
    x = InputNode("x", 4, value_range=(-100.0, 100.0))
    W = torch.randn(4, 3)
    b = torch.randn(3)
    linear_node = Linear(x, W, b, name="biased_lin")

    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    out_cols = rmap.allocate(linear_node)

    layer = TransformerLayer(D, D_HEAD, activation="swish")
    op = _make_mlp_op(
        rmap, "compute_linear_bypass", linear_node, out_cols, mlp_slots=list(range(6))
    )
    write_mlp_sublayer(layer, [op], rmap)
    layer.to(device_mod.get_device(verbose=False))

    x_values = torch.randn(N_POS, 4)
    res = _build_residual_stream(rmap, {x: x_values})

    out = layer.mlp.forward(res)
    expected = linear_node.compute(N_POS, {"x": x_values})
    assert torch.allclose(out[:, out_cols].cpu(), expected, atol=1e-4)


def test_swish_mlp_constant_literal():
    """LiteralValue lands in the down projection's output bias."""
    const = LiteralValue(torch.tensor([1.0, -2.0, 3.5]))

    rmap = ResidualStreamMap(D)
    out_cols = rmap.allocate(const)

    layer = TransformerLayer(D, D_HEAD, activation="swish")
    op = MLPOp(
        op_type="compute_literal_value", node=const, target_cols=out_cols, mlp_slots=[]
    )
    write_mlp_sublayer(layer, [op], rmap)
    device = device_mod.get_device(verbose=False)
    layer.to(device)

    out = layer.mlp.forward(torch.zeros(N_POS, D, device=device))
    expected = const.compute(N_POS, {})
    assert torch.allclose(out[:, out_cols].cpu(), expected, atol=1e-6)


def test_machine_mismatch_asserts():
    """A node/machine activation mismatch is a compiler bug and must fire the
    writer's assert (the uniformity check makes it unreachable in a real
    compile)."""
    import pytest

    x = InputNode("x", 4, value_range=(-10.0, 10.0))

    # swish FFN on the ReLU machine
    ffn_swish = _swish_ffn(x, gated=False)
    rmap = ResidualStreamMap(D)
    rmap.allocate(x)
    out_cols = rmap.allocate(ffn_swish)
    op = _make_mlp_op(
        rmap, "compute_ffn", ffn_swish, out_cols, mlp_slots=list(range(6))
    )
    with pytest.raises(AssertionError, match="machine mismatch"):
        write_mlp_sublayer(TransformerLayer(D, D_HEAD), [op], rmap)

    # relu FFN on the swish machine
    relu_ffn = linear_relu_linear(
        x, torch.randn(6, 4), torch.randn(6), torch.randn(6, 3), torch.randn(3)
    )
    rmap2 = ResidualStreamMap(D)
    rmap2.allocate(x)
    out_cols2 = rmap2.allocate(relu_ffn)
    op2 = _make_mlp_op(
        rmap2, "compute_ffn", relu_ffn, out_cols2, mlp_slots=list(range(6))
    )
    with pytest.raises(AssertionError, match="machine mismatch"):
        write_mlp_sublayer(
            TransformerLayer(D, D_HEAD, activation="swish"), [op2], rmap2
        )

    # gated relu FFN on the ReLU machine: activation matches but the machine
    # has no up projection to realize the gated lanes.
    from torchwright.graph import FFN

    gated_relu = FFN(
        x,
        gate_proj=torch.randn(6, 4),
        gate_bias=torch.randn(6),
        out_proj=torch.randn(6, 3),
        out_bias=torch.randn(3),
        up_proj=torch.randn(6, 4),
        up_bias=torch.randn(6),
        activation="relu",
    )
    rmap3 = ResidualStreamMap(D)
    rmap3.allocate(x)
    out_cols3 = rmap3.allocate(gated_relu)
    op3 = _make_mlp_op(
        rmap3, "compute_ffn", gated_relu, out_cols3, mlp_slots=list(range(6))
    )
    with pytest.raises(AssertionError, match="no up projection"):
        write_mlp_sublayer(TransformerLayer(D, D_HEAD), [op3], rmap3)
