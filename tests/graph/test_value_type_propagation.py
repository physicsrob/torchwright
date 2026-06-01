"""Propagation-rule tests for per-op ``compute_value_type()`` implementations.

These exercise representative subgraphs (not the full test suite). The
runtime verifier (``TW_VERIFY_VALUE_TYPES=1``) on the full suite is the
primary guard against soundness bugs.
"""

import math

import torch

from torchwright.graph import (
    Add,
    Attn,
    Concatenate,
    Embedding,
    InputNode,
    Linear,
    LiteralValue,
    NodeValueType,
    PosEncoding,
    Range,
    ReLU,
    ValueLogger,
)
from torchwright.graph.misc import Placeholder

# --- Leaf nodes -----------------------------------------------------


def test_input_node_has_declared_range():
    n = InputNode("x", 4, value_range=(-100.0, 100.0))
    assert n.value_type.value_range == Range(-100.0, 100.0)


def test_placeholder_is_unknown():
    n = Placeholder(d=3)
    assert n.value_type == NodeValueType.unknown()


def test_pos_encoding_bounded():
    n = PosEncoding(d_pos=9)
    vt = n.value_type
    # Sinusoidal columns are [-1, 1]; the counter column (the last one)
    # has range [0, 100000].  The scalar range is the envelope.
    assert vt.value_range == Range(-1.0, 100000.0)


def test_literal_range():
    lit = LiteralValue(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    vt = lit.value_type
    assert vt.value_range == Range(0.0, 3.0)


def test_literal_binary_range():
    lit = LiteralValue(torch.tensor([0.0, 1.0, 0.0, 1.0]))
    vt = lit.value_type
    assert vt.value_range == Range(0.0, 1.0)


def test_literal_one_hot_range():
    lit = LiteralValue(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    vt = lit.value_type
    assert vt.value_range == Range(0.0, 1.0)


def test_literal_sign_range():
    lit = LiteralValue(torch.tensor([-1.0, 1.0, -1.0]))
    vt = lit.value_type
    assert vt.value_range == Range(-1.0, 1.0)


def test_literal_non_integer():
    lit = LiteralValue(torch.tensor([0.3, 0.7]))
    vt = lit.value_type
    assert abs(vt.value_range.lo - 0.3) < 1e-6
    assert abs(vt.value_range.hi - 0.7) < 1e-6


def test_embedding_has_finite_range():
    emb = Embedding(vocab=["1", "2", "3"])
    vt = emb.value_type
    assert math.isfinite(vt.value_range.lo)
    assert math.isfinite(vt.value_range.hi)
    assert vt.value_range.lo <= vt.value_range.hi


# --- Add / Concatenate ---------------------------------------------


def test_add_sums_ranges():
    a = LiteralValue(torch.tensor([1.0, 2.0]))
    b = LiteralValue(torch.tensor([3.0, 4.0]))
    s = Add(a, b)
    vt = s.value_type
    assert vt.value_range == Range(1.0 + 3.0, 2.0 + 4.0)


def test_concatenate_has_range():
    a = LiteralValue(torch.tensor([0.0, 1.0]))
    b = LiteralValue(torch.tensor([2.0, 3.0]))
    c = Concatenate([a, b])
    vt = c.value_type
    assert vt.value_range == Range(0.0, 3.0)


# --- Linear ---------------------------------------------------------


def test_linear_with_integer_weights_and_bounded_input():
    inp = LiteralValue(torch.tensor([1.0, 2.0, 3.0]))
    W = torch.tensor([[1.0], [0.0], [-1.0]])
    b = torch.tensor([0.0])
    lin = Linear(inp, W, b)
    vt = lin.value_type
    # Affine bound propagates exact constant: 1*1 + 2*0 + 3*(-1) = -2
    assert vt.value_range == Range(-2.0, -2.0)


def test_linear_bounded_input_propagates_range():
    inp = InputNode("x", 2, value_range=(-100.0, 100.0))
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    lin = Linear(inp, W)
    vt = lin.value_type
    assert vt.value_range == Range(-100.0, 100.0)


# --- ReLU -----------------------------------------------------------


def test_relu_clamps_range():
    inp = LiteralValue(torch.tensor([-2.0, 1.0, 3.0]))
    r = ReLU(inp)
    vt = r.value_type
    assert vt.value_range == Range(0.0, 3.0)


# --- ValueLogger (pass-through) ------------------------------------


def test_value_logger_passes_through():
    inp = LiteralValue(torch.tensor([0.0, 1.0]))
    vl = ValueLogger(inp, name="debug")
    assert vl.value_type == inp.value_type


# --- Attn default is unknown (claims come via Assert wrappers) -----


def test_attn_propagates_value_range_from_value_input():
    from torchwright.graph.value_type import Range

    pe = PosEncoding(d_pos=9)
    value = LiteralValue(torch.tensor([2.0, 3.0]))
    attn = Attn(
        query_in=pe,
        key_in=pe,
        value_in=value,
        query_matrix=torch.eye(9, 2),
        key_matrix=torch.eye(9, 2),
        value_matrix=torch.eye(2),
        output_matrix=torch.eye(2),
    )
    assert attn.value_type.value_range == Range(2.0, 3.0)


# --- Op propagation rules ----------------------------------------


def test_compare_has_bounded_range():
    from torchwright.ops.arithmetic_ops import compare

    inp = LiteralValue(torch.tensor([3.0]))
    out = compare(inp, 2.0)
    assert out.value_type.value_range.lo >= -1.0
    assert out.value_type.value_range.hi <= 1.0


def test_compare_01_levels():
    from torchwright.ops.arithmetic_ops import compare

    inp = LiteralValue(torch.tensor([3.0]))
    out = compare(inp, 2.0, true_level=1.0, false_level=0.0)
    assert out.value_type.value_range.lo >= 0.0
    assert out.value_type.value_range.hi <= 1.0


def test_compare_arbitrary_levels():
    from torchwright.ops.arithmetic_ops import compare

    inp = LiteralValue(torch.tensor([3.0]))
    out = compare(inp, 2.0, true_level=5.0, false_level=-3.0)
    vt = out.value_type
    assert vt.value_range.lo >= -3.0
    assert vt.value_range.hi <= 5.0


def test_compare_float_levels():
    from torchwright.ops.arithmetic_ops import compare

    inp = LiteralValue(torch.tensor([3.0]))
    out = compare(inp, 2.0, true_level=0.5, false_level=-0.5)
    assert out.value_type.value_range.lo >= -0.5
    assert out.value_type.value_range.hi <= 0.5


def test_equals_vector_has_bounded_range():
    from torchwright.ops.logic_ops import equals_vector

    inp = LiteralValue(torch.tensor([1.0, 0.0, 0.0]))
    out = equals_vector(inp, torch.tensor([1.0, 0.0, 0.0]))
    assert out.value_type.value_range.lo >= -1.0
    assert out.value_type.value_range.hi <= 1.0


def test_select_bounded_range():
    from torchwright.ops.map_select import select
    from torchwright.graph.asserts import assert_bool

    cond = assert_bool(LiteralValue(torch.tensor([1.0])))
    a = LiteralValue(torch.tensor([2.0, 3.0]))
    b = LiteralValue(torch.tensor([5.0, 7.0]))
    out = select(cond, a, b)
    vt = out.value_type
    assert vt.value_range.lo >= 2.0 - 0.1
    assert vt.value_range.hi <= 7.0 + 0.1


def test_select_binary_branches():
    from torchwright.ops.map_select import select
    from torchwright.graph.asserts import assert_bool

    cond = assert_bool(LiteralValue(torch.tensor([1.0])))
    a = LiteralValue(torch.tensor([0.0, 1.0]))
    b = LiteralValue(torch.tensor([1.0, 0.0]))
    out = select(cond, a, b)
    assert out.value_type.value_range.lo >= -0.02
    assert out.value_type.value_range.hi <= 1.02


def test_select_unknown_cond():
    from torchwright.ops.map_select import select

    cond = InputNode("cond", 1, value_range=(-100.0, 100.0))
    a = LiteralValue(torch.tensor([2.0, 3.0]))
    b = LiteralValue(torch.tensor([5.0, 7.0]))
    out = select(cond, a, b)
    vt = out.value_type
    assert vt.value_range.lo >= 2.0 - 0.1
    assert vt.value_range.hi <= 7.0 + 0.1


def test_cond_gate_bounded_range():
    from torchwright.ops.logic_ops import cond_gate
    from torchwright.graph.asserts import assert_bool

    cond = assert_bool(LiteralValue(torch.tensor([1.0])))
    inp = LiteralValue(torch.tensor([3.0, 5.0]))
    out = cond_gate(cond, inp)
    vt = out.value_type
    assert vt.value_range.lo >= -0.1
    assert vt.value_range.hi <= 5.1


def test_cond_gate_binary_inp():
    from torchwright.ops.logic_ops import cond_gate
    from torchwright.graph.asserts import assert_bool

    cond = assert_bool(LiteralValue(torch.tensor([1.0])))
    inp = LiteralValue(torch.tensor([0.0, 1.0]))
    out = cond_gate(cond, inp)
    assert out.value_type.value_range.lo >= -0.02
    assert out.value_type.value_range.hi <= 1.02


def test_cond_gate_unknown_cond():
    from torchwright.ops.logic_ops import cond_gate

    cond = InputNode("cond", 1, value_range=(-100.0, 100.0))
    inp = LiteralValue(torch.tensor([3.0, 5.0]))
    out = cond_gate(cond, inp)
    vt = out.value_type
    assert vt.value_range.lo >= -0.1
    assert vt.value_range.hi <= 5.1


# --- Phase 3: in_range, map_to_table, floor/ceil ----------------


def test_in_range_bounded():
    from torchwright.ops.map_select import in_range

    lower = LiteralValue(torch.tensor([1.0]))
    upper = LiteralValue(torch.tensor([3.0]))
    out = in_range(lower, upper, 5)
    assert out.value_type.value_range.lo >= -1.0
    assert out.value_type.value_range.hi <= 1.0


def test_floor_int_bounded():
    from torchwright.ops.arithmetic_ops import floor_int
    from torchwright.graph.asserts import assert_in_range

    inp = assert_in_range(LiteralValue(torch.tensor([2.5])), 0.0, 10.0)
    out = floor_int(inp, 0, 10)
    vt = out.value_type
    assert vt.value_range.lo >= 0.0
    assert vt.value_range.hi <= 10.0


def test_ceil_int_bounded():
    from torchwright.ops.arithmetic_ops import ceil_int
    from torchwright.graph.asserts import assert_in_range

    inp = assert_in_range(LiteralValue(torch.tensor([2.5])), 0.0, 10.0)
    out = ceil_int(inp, 0, 10)
    vt = out.value_type
    assert vt.value_range.lo >= 0.0
    assert vt.value_range.hi <= 10.0


def test_thermometer_floor_div_bounded():
    from torchwright.ops.arithmetic_ops import thermometer_floor_div

    inp = LiteralValue(torch.tensor([35.0]))
    out = thermometer_floor_div(inp, 10, 100)
    vt = out.value_type
    assert vt.value_range.lo >= 0.0
    assert vt.value_range.hi <= 10.0
