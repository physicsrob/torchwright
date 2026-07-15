"""Tests for the assignment-level Add placement derivation.

One definition of addend deadness (add_placement.derive_add_placement),
computed from complete layer/route maps.  These tests hand-build tiny
consumer relations instead of running a scheduler: the derivation must be
usable from CP-SAT extraction, heuristic-trace completion, and directed
replay alike, so it depends only on the maps and an effective-consumer
callable.
"""

import pytest
import torch

from torchwright.compiler.forward.add_placement import (
    AddPlacement,
    derive_add_placement,
)
from torchwright.graph import Add, Concatenate, Linear
from torchwright.ops.inout_nodes import create_input


def _linear(x, w_out, name):
    return Linear(x, torch.randn(len(x), w_out) * 0.2, name=name)


def _consumers_from(pairs):
    """Build an effective-consumer callable from {node_id: [nodes]}."""

    def effective_consumers(node):
        return pairs.get(node.node_id, [])

    return effective_consumers


def _derive(add, pairs, layers, routes, **kwargs):
    return derive_add_placement(
        add,
        effective_consumers=_consumers_from(pairs),
        node_to_layer=layers,
        node_to_routing=routes,
        **kwargs,
    )


def _graph():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    a = _linear(x, 3, "a")
    b = _linear(x, 3, "b")
    add = Add(a, b, name="add")
    return x, a, b, add


def test_both_addends_dead_earlier_selects_occurrence_0():
    x, a, b, add = _graph()
    pairs = {a.node_id: [add], b.node_id: [add]}
    layers = {add.node_id: 3, a.node_id: 1, b.node_id: 1}
    routes = {add.node_id: "attn", a.node_id: "attn", b.node_id: "attn"}
    p = _derive(add, pairs, layers, routes)
    assert p == AddPlacement(True, True, 0)
    assert p.is_free


def test_only_occurrence_1_reusable():
    x, a, b, add = _graph()
    later = _linear(a, 2, "later")  # a has a later reader; b does not
    pairs = {a.node_id: [add, later], b.node_id: [add]}
    layers = {add.node_id: 3, later.node_id: 5}
    routes = {add.node_id: "attn", later.node_id: "attn"}
    assert _derive(add, pairs, layers, routes) == AddPlacement(False, True, 1)


def test_same_layer_attention_consumer_counts_for_mlp_add_only():
    x, a, b, add = _graph()
    peer = _linear(a, 2, "peer")  # reads a in the same layer as the add
    pairs = {a.node_id: [add, peer], b.node_id: [add, peer]}
    layers = {add.node_id: 3, peer.node_id: 3}
    routes_attn_peer = {add.node_id: "mlp", peer.node_id: "attn"}
    # MLP-routed Add: a same-layer attention consumer is complete by the
    # MLP phase-start snapshot.
    assert _derive(add, pairs, layers, routes_attn_peer) == AddPlacement(True, True, 0)
    # Attention-routed Add: the same consumer is NOT complete (attention
    # phase-start snapshot).
    routes_attn_add = {add.node_id: "attn", peer.node_id: "attn"}
    assert _derive(add, pairs, layers, routes_attn_add) == AddPlacement(
        False, False, None
    )


def test_same_layer_mlp_consumer_never_counts():
    x, a, b, add = _graph()
    peer = _linear(a, 2, "peer")
    pairs = {a.node_id: [add, peer], b.node_id: [add]}
    layers = {add.node_id: 3, peer.node_id: 3}
    routes = {add.node_id: "mlp", peer.node_id: "mlp"}
    # a blocked by the same-layer MLP peer; b is free.
    assert _derive(add, pairs, layers, routes) == AddPlacement(False, True, 1)


def test_unordered_consumer_blocks_reuse():
    x, a, b, add = _graph()
    terminal = Concatenate([a])  # output retention: no layer entry
    pairs = {a.node_id: [add, terminal], b.node_id: [add]}
    layers = {add.node_id: 3}
    routes = {add.node_id: "attn"}
    assert _derive(add, pairs, layers, routes) == AddPlacement(False, True, 1)


def test_concatenate_occurrence_is_not_reusable():
    x, a, b, add0 = _graph()
    cat = Concatenate([a, b])
    tail = create_input("tail", 6, value_range=(-1.0, 1.0))
    add = Add(cat, tail, name="cat_add")
    pairs = {cat.node_id: [add], tail.node_id: [add]}
    layers = {add.node_id: 2}
    routes = {add.node_id: "attn"}
    # Occurrence 0 is a Concatenate (never residual-allocated); occurrence 1
    # is a graph input with no other consumers.
    assert _derive(add, pairs, layers, routes) == AddPlacement(False, True, 1)


def test_graph_input_target_needs_no_own_layer():
    x, a, b, add0 = _graph()
    y = create_input("y", 3, value_range=(-1.0, 1.0))
    add = Add(y, a, name="input_add")
    reader = _linear(y, 2, "reader")
    pairs = {y.node_id: [add, reader], a.node_id: [add]}
    layers = {add.node_id: 4, reader.node_id: 2}
    routes = {add.node_id: "attn", reader.node_id: "attn"}
    # y has no layer entry of its own — only its consumers need one.
    assert _derive(add, pairs, layers, routes) == AddPlacement(True, True, 0)


def test_self_add_selects_occurrence_0():
    x = create_input("x", 4, value_range=(-1.0, 1.0))
    a = _linear(x, 3, "a")
    add = Add(a, a, name="self_add")
    pairs = {a.node_id: [add]}
    layers = {add.node_id: 2}
    routes = {add.node_id: "mlp"}
    assert _derive(add, pairs, layers, routes) == AddPlacement(True, True, 0)


def test_held_target_short_circuits_fresh():
    x, a, b, add = _graph()
    pairs = {a.node_id: [add], b.node_id: [add]}
    layers = {add.node_id: 3}
    routes = {add.node_id: "attn"}
    p = _derive(add, pairs, layers, routes, held_target_id=add.node_id)
    assert p == AddPlacement(False, False, None)
    assert not p.is_free


def test_held_source_occurrence_never_reusable():
    x, a, b, add0 = _graph()
    emb = create_input("emb", 3, value_range=(-1.0, 1.0))
    add = Add(emb, a, name="handoff")
    pairs = {emb.node_id: [add], a.node_id: [add]}
    layers = {add.node_id: 3}
    routes = {add.node_id: "attn"}
    p = _derive(add, pairs, layers, routes, held_source_id=emb.node_id)
    # The held source's columns end through the held-bank cancel/hold
    # transition, never through reassign; occurrence 1 stays eligible.
    assert p == AddPlacement(False, True, 1)


def test_missing_add_layer_or_route_raises():
    x, a, b, add = _graph()
    pairs = {a.node_id: [add], b.node_id: [add]}
    with pytest.raises(ValueError, match="layer and route"):
        _derive(add, pairs, {}, {add.node_id: "attn"})
    with pytest.raises(ValueError, match="layer and route"):
        _derive(add, pairs, {add.node_id: 3}, {})
