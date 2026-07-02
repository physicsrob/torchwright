"""Exactness and structure of the one-hot lookup op.

These run as reference evaluation (``node.compute``), which executes the same
Linear/ReLU matmuls the compiler emits — so "exact on every key" here is the
op's exact arithmetic, and the weight-matrix readback is the literal compiled
block (D6: the smallest layer that reproduces "a lookup is a selection
matrix").
"""

import torch

from torchwright.graph import Block, Linear
from torchwright.graph.misc import Assert
from torchwright.ops.arithmetic_ops import concat
from torchwright.ops.inout_nodes import create_input, create_onehot_embedding
from torchwright.ops.onehot_table import onehot_lookup


def _onehot(index, width):
    v = torch.zeros(width)
    v[index] = 1.0
    return v


def _unwrap(node):
    """Strip the trailing value-range Assert that onehot_lookup adds."""
    return node.inputs[0] if isinstance(node, Assert) else node


# ---------------------------------------------------------------------------
# Single one-hot input -> plain Linear selection matrix
# ---------------------------------------------------------------------------


def test_single_input_exact_on_every_key():
    table = {
        "a": torch.tensor([10.0, 1.0]),
        "b": torch.tensor([20.0, 2.0]),
        "c": torch.tensor([30.0, 3.0]),
    }
    default = torch.tensor([-1.0, -1.0])
    embedding = create_onehot_embedding(vocab=["a", "b", "c", "d"])
    out = onehot_lookup(
        embedding,
        {embedding.get_embedding(t): v for t, v in table.items()},
        default,
    )
    for token, expected in table.items():
        got = out.compute(n_pos=1, input_values={"embedding_input": [token]})[0]
        assert torch.allclose(got, expected, atol=1e-5), (token, got)
    # "d" is absent from the table -> default.
    got = out.compute(n_pos=1, input_values={"embedding_input": ["d"]})[0]
    assert torch.allclose(got, default, atol=1e-5)


def test_single_input_is_a_plain_selection_matrix():
    """No ReLU; the Linear's rows are literally the table values."""
    table = {
        "a": torch.tensor([10.0, 1.0]),
        "b": torch.tensor([20.0, 2.0]),
    }
    default = torch.tensor([-1.0, -1.0])
    embedding = create_onehot_embedding(vocab=["a", "b", "c"])
    out = onehot_lookup(
        embedding,
        {embedding.get_embedding(t): v for t, v in table.items()},
        default,
    )
    linear = _unwrap(out)
    assert isinstance(linear, Linear)
    assert linear.inputs[0] is embedding  # reads the one-hot directly, no ReLU
    weight = linear.output_matrix  # (d_key, d_value)
    for token, value in table.items():
        row = embedding.tokenizer.get_token_id(token)
        assert torch.allclose(weight[row], value, atol=1e-6)
    # The absent token's row is the default.
    absent = embedding.tokenizer.get_token_id("c")
    assert torch.allclose(weight[absent], default, atol=1e-6)


# ---------------------------------------------------------------------------
# Several one-hot inputs -> AND-of-matches
# ---------------------------------------------------------------------------


def _two_block_table():
    a0, a1 = _onehot(0, 2), _onehot(1, 2)
    return {
        torch.cat([a0, a0]): torch.tensor([1.0]),
        torch.cat([a0, a1]): torch.tensor([2.0]),
        torch.cat([a1, a0]): torch.tensor([3.0]),
        torch.cat([a1, a1]): torch.tensor([4.0]),
    }


def test_multi_input_exact_on_every_key():
    a = create_input("a", 2)
    b = create_input("b", 2)
    out = onehot_lookup(
        concat([a, b]), _two_block_table(), default=torch.tensor([-1.0])
    )
    cases = {
        (0, 0): 1.0,
        (0, 1): 2.0,
        (1, 0): 3.0,
        (1, 1): 4.0,
    }
    for (ai, bi), expected in cases.items():
        got = out.compute(
            n_pos=1,
            input_values={
                "a": _onehot(ai, 2).unsqueeze(0),
                "b": _onehot(bi, 2).unsqueeze(0),
            },
        )
        assert torch.allclose(got, torch.tensor([[expected]]), atol=1e-5), (ai, bi, got)


def test_multi_input_default_when_unmatched():
    a = create_input("a", 2)
    b = create_input("b", 2)
    # Omit the (1, 1) combination.
    table = _two_block_table()
    del table[next(k for k in table if torch.equal(k, torch.cat([_onehot(1, 2)] * 2)))]
    out = onehot_lookup(concat([a, b]), table, default=torch.tensor([-1.0]))
    got = out.compute(
        n_pos=1,
        input_values={"a": _onehot(1, 2).unsqueeze(0), "b": _onehot(1, 2).unsqueeze(0)},
    )
    assert torch.allclose(got, torch.tensor([[-1.0]]), atol=1e-5)


def test_multi_input_uses_block():
    a = create_input("a", 2)
    b = create_input("b", 2)
    out = onehot_lookup(
        concat([a, b]), _two_block_table(), default=torch.tensor([-1.0])
    )
    # The multi-block path needs the nonlinear ReLU-AND, so it builds a Block
    # (via linear_relu_linear) rather than the pure-Linear single-block lookup.
    node = _unwrap(out)
    assert isinstance(node, Block)


# ---------------------------------------------------------------------------
# Tight value range
# ---------------------------------------------------------------------------


def test_tight_value_range_single():
    table = {"a": torch.tensor([10.0]), "b": torch.tensor([20.0])}
    default = torch.tensor([-1.0])
    # "c" is absent, so the default is a reachable row and lands in the range.
    embedding = create_onehot_embedding(vocab=["a", "b", "c"])
    out = onehot_lookup(
        embedding,
        {embedding.get_embedding(t): v for t, v in table.items()},
        default,
    )
    r = out.value_type.value_range
    # Tight [min, max] over {values, default}; not map_to_table's wider
    # default ± Σ|value − default| = -1 ± 32.
    assert r.lo == -1.0 and r.hi == 20.0


def test_tight_value_range_multi():
    a = create_input("a", 2)
    b = create_input("b", 2)
    out = onehot_lookup(
        concat([a, b]), _two_block_table(), default=torch.tensor([-1.0])
    )
    r = out.value_type.value_range
    # Values are {1, 2, 3, 4}, default -1: tight [-1, 4], not a widened sum.
    assert r.lo == -1.0 and r.hi == 4.0
