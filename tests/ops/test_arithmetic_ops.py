import torch
from torchwright.ops.relu.arithmetic_ops import (
    abs as abs_node,
    clamp,
    relu_add,
    compare,
    min as min_node,
    mod_const,
    piecewise_linear,
    square,
    multiply_integers,
    reciprocal,
    floor_int,
    ceil_int,
)
from torchwright.ops.linear import (
    add_const,
    bool_to_01,
    multiply_const,
    negate,
    subtract,
)
from torchwright.ops.inout_nodes import create_input


def test_bool_to_01():
    x = create_input("x", 1)
    out = bool_to_01(x)
    for x_val, expected in [(-1.0, 0.0), (1.0, 1.0)]:
        result = out.compute(n_pos=1, input_values={"x": torch.tensor([[x_val]])})
        assert result.item() == expected


def test_bool_to_01_wide():
    x = create_input("x", 3)
    out = bool_to_01(x)
    result = out.compute(
        n_pos=1,
        input_values={"x": torch.tensor([[1.0, -1.0, 1.0]])},
    )
    assert result.tolist() == [[1.0, 0.0, 1.0]]


def test_add_const():
    offset = 100.0
    value_input = create_input("value", 1)
    n = add_const(value_input, offset)
    output = n.compute(n_pos=1, input_values={"value": torch.tensor([[1.0]])})
    assert output.tolist() == [[101.0]]


def test_relu_add():
    input_val1 = create_input("val1", 3)
    input_val2 = create_input("val2", 3)
    x = relu_add(input_val1, input_val2)
    for i in range(10):
        val1 = 100.0 * (torch.rand(3) - 0.5)
        val2 = 100.0 * (torch.rand(3) - 0.5)
        output = x.compute(
            n_pos=1, input_values={"val1": val1.unsqueeze(0), "val2": val2.unsqueeze(0)}
        )
        expected_value = torch.clamp(val1, min=0) + torch.clamp(val2, min=0)
        assert (output == expected_value).all()


def test_negate():
    x = create_input("x", 3)
    out = negate(x)
    vals = torch.tensor([[1.0, -2.0, 3.0]])
    result = out.compute(n_pos=1, input_values={"x": vals})
    assert torch.allclose(result, -vals)


def test_subtract():
    a = create_input("a", 3)
    b = create_input("b", 3)
    out = subtract(a, b)
    va = torch.tensor([[5.0, 10.0, 15.0]])
    vb = torch.tensor([[1.0, 3.0, 5.0]])
    result = out.compute(n_pos=1, input_values={"a": va, "b": vb})
    assert torch.allclose(result, va - vb)


def test_multiply_const():
    x = create_input("x", 2)
    out = multiply_const(x, 3.0)
    vals = torch.tensor([[2.0, -4.0]])
    result = out.compute(n_pos=1, input_values={"x": vals})
    assert torch.allclose(result, 3.0 * vals)


def test_compare():
    thresh = 100.0
    for delta in [-10.0, -5.0, 5.0, 10.0]:
        for true_level in [-10.0, 0.0, 1.0, 10.0]:
            for false_level in [-5.0, -1.0]:
                value_input = create_input("value", 1)
                n = compare(value_input, thresh, true_level, false_level)
                test_val = thresh + delta
                output = n.compute(
                    n_pos=1, input_values={"value": torch.tensor([[test_val]])}
                )
                assert output.item() == true_level if delta > 0 else false_level


def test_square():
    x = create_input("x", 1)
    sq = square(x, max_value=9)
    for val in range(10):
        result = sq.compute(n_pos=1, input_values={"x": torch.tensor([[float(val)]])})
        expected = val * val
        assert (
            abs(result.item() - expected) < 0.01
        ), f"{val}² = {expected}, got {result.item()}"


def test_square_large():
    """Test with max_value=18 (needed for digit a+b range in multiply_integers)."""
    x = create_input("x", 1)
    sq = square(x, max_value=18)
    for val in [0, 1, 9, 10, 17, 18]:
        result = sq.compute(n_pos=1, input_values={"x": torch.tensor([[float(val)]])})
        expected = val * val
        assert (
            abs(result.item() - expected) < 0.01
        ), f"{val}² = {expected}, got {result.item()}"


def test_multiply_integers_all_digit_pairs():
    """Verify a*b for all 100 single-digit pairs."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    prod = multiply_integers(a, b, max_value=9)
    for i in range(10):
        for j in range(10):
            result = prod.compute(
                n_pos=1,
                input_values={
                    "a": torch.tensor([[float(i)]]),
                    "b": torch.tensor([[float(j)]]),
                },
            )
            expected = i * j
            assert (
                abs(result.item() - expected) < 0.5
            ), f"{i}*{j} = {expected}, got {result.item()}"


def test_multiply_integers_zero():
    """0 * anything = 0."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    prod = multiply_integers(a, b, max_value=9)
    for val in [0, 5, 9]:
        result = prod.compute(
            n_pos=1,
            input_values={
                "a": torch.tensor([[0.0]]),
                "b": torch.tensor([[float(val)]]),
            },
        )
        assert abs(result.item()) < 0.1, f"0*{val} should be 0, got {result.item()}"


def test_mod_const():
    x = create_input("x", 1)
    cases = [
        # (value, divisor, max_value, expected)
        (7, 3, 10, 1),
        (10, 5, 10, 0),
        (13, 4, 15, 1),
        (9, 3, 10, 0),
        (2, 5, 10, 2),
        (0, 3, 10, 0),
    ]
    for val, divisor, max_val, expected in cases:
        m = mod_const(x, divisor, max_val)
        result = m.compute(n_pos=1, input_values={"x": torch.tensor([[float(val)]])})
        assert (
            abs(result.item() - expected) < 0.5
        ), f"{val} % {divisor} = {expected}, got {result.item()}"


def _eval_pw(node, val):
    """Helper: evaluate a 1D piecewise_linear node at a scalar value."""
    return node.compute(n_pos=1, input_values={"x": torch.tensor([[val]])}).item()


def test_piecewise_linear_identity():
    """f(x) = x on [0, 10], clamped outside."""
    x = create_input("x", 1)
    f = piecewise_linear(x, [0.0, 10.0], lambda x: x)
    assert abs(_eval_pw(f, 0.0) - 0.0) < 0.01
    assert abs(_eval_pw(f, 5.0) - 5.0) < 0.01
    assert abs(_eval_pw(f, 10.0) - 10.0) < 0.01
    # Clamped outside range
    assert abs(_eval_pw(f, -3.0) - 0.0) < 0.01
    assert abs(_eval_pw(f, 15.0) - 10.0) < 0.01


def test_piecewise_linear_vshape():
    """Absolute value: f(x) = |x| on [-5, 5]."""
    x = create_input("x", 1)
    f = piecewise_linear(x, [-5.0, 0.0, 5.0], abs)
    assert abs(_eval_pw(f, -5.0) - 5.0) < 0.01
    assert abs(_eval_pw(f, -2.5) - 2.5) < 0.01
    assert abs(_eval_pw(f, 0.0) - 0.0) < 0.01
    assert abs(_eval_pw(f, 3.0) - 3.0) < 0.01
    assert abs(_eval_pw(f, 5.0) - 5.0) < 0.01


def test_piecewise_linear_square():
    """Approximate x^2 via breakpoints at integers 0-9."""
    x = create_input("x", 1)
    bp = [float(i) for i in range(10)]
    f = piecewise_linear(x, bp, lambda x: x * x)
    for i in range(10):
        result = _eval_pw(f, float(i))
        assert abs(result - i * i) < 0.01, f"f({i}) = {result}, expected {i*i}"


def test_piecewise_linear_constant_segment():
    """Flat section between x=5 and x=10."""
    x = create_input("x", 1)
    f = piecewise_linear(
        x,
        [0.0, 5.0, 10.0, 15.0],
        lambda x: 2 * x if x <= 5 else (10 if x <= 10 else 2 * x - 10),
    )
    assert abs(_eval_pw(f, 2.5) - 5.0) < 0.01
    assert abs(_eval_pw(f, 5.0) - 10.0) < 0.01
    assert abs(_eval_pw(f, 7.5) - 10.0) < 0.01
    assert abs(_eval_pw(f, 10.0) - 10.0) < 0.01
    assert abs(_eval_pw(f, 12.5) - 15.0) < 0.01


def test_piecewise_linear_extrapolate():
    """clamp=False: linear extrapolation beyond breakpoint range."""
    x = create_input("x", 1)
    f = piecewise_linear(x, [0.0, 10.0], lambda x: 10 * x, clamp=False)
    # Interior
    assert abs(_eval_pw(f, 5.0) - 50.0) < 0.01
    # Extrapolation (slope = 10)
    assert abs(_eval_pw(f, -5.0) - (-50.0)) < 0.01
    assert abs(_eval_pw(f, 15.0) - 150.0) < 0.01


def test_piecewise_linear_chunking():
    """d_max=4 forces multiple MLP sublayers with 10 breakpoints."""
    x = create_input("x", 1)
    bp = [float(i) for i in range(10)]
    f = piecewise_linear(x, bp, lambda x: x * x, d_max=4)
    for i in range(10):
        result = _eval_pw(f, float(i))
        assert abs(result - i * i) < 0.01, f"f({i}) = {result}, expected {i*i}"


def test_abs():
    """abs on a 3-wide node with positive, negative, and zero."""
    x = create_input("x", 3)
    out = abs_node(x)
    vals = torch.tensor([[5.0, -3.0, 0.0]])
    result = out.compute(n_pos=1, input_values={"x": vals})
    assert torch.allclose(result, torch.tensor([[5.0, 3.0, 0.0]]))


def test_abs_scalar():
    """abs on scalar inputs."""
    x = create_input("x", 1)
    out = abs_node(x)
    for v in [-7.0, 0.0, 4.5]:
        result = out.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        assert abs(result.item() - abs(v)) < 0.01


def test_min():
    a = create_input("a", 3)
    b = create_input("b", 3)
    out = min_node(a, b)
    va = torch.tensor([[5.0, -2.0, 7.0]])
    vb = torch.tensor([[3.0, 1.0, 7.0]])
    result = out.compute(n_pos=1, input_values={"a": va, "b": vb})
    expected = torch.min(va, vb)
    assert torch.allclose(result, expected, atol=0.01)


def test_reciprocal():
    """Exact at integer multiples of step in [1, 10]."""
    x = create_input("x", 1)
    r = reciprocal(x, min_value=1.0, max_value=10.0)
    for v in range(1, 11):
        result = r.compute(n_pos=1, input_values={"x": torch.tensor([[float(v)]])})
        expected = 1.0 / v
        assert (
            abs(result.item() - expected) < 0.01
        ), f"1/{v} = {expected}, got {result.item()}"


def test_reciprocal_interpolation():
    """Between grid points the result should be close to 1/x."""
    x = create_input("x", 1)
    r = reciprocal(x, min_value=1.0, max_value=10.0)
    # Halfway between grid points — linear interpolation error is bounded
    for v in [1.5, 2.5, 5.5]:
        result = r.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        exact = 1.0 / v
        assert (
            abs(result.item() - exact) < 0.1
        ), f"1/{v} = {exact:.4f}, got {result.item():.4f}"


def test_reciprocal_small_min_value():
    """reciprocal must be accurate even when min_value << 1.

    The game graph calls reciprocal(x, min_value=0.01, max_value=20)
    and reciprocal(x, min_value=0.1, max_value=20).  With uniform
    breakpoint spacing (old bug), the first segment [0.01, 1.01]
    maps [100, 0.99] linearly — giving 51.5 at x=0.5 (true: 2.0)
    and 1.98 at x=1.0 (true: 1.0).
    """
    x = create_input("x", 1)

    # min_value=0.01: used by sort_inv_den and render inv_abs_den
    r_001 = reciprocal(x, min_value=0.01, max_value=20.0)
    for v in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        result = r_001.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = 1.0 / v
        rel_err = abs(result.item() - expected) / expected
        assert rel_err < 0.05, (
            f"reciprocal(min=0.01): 1/{v} = {expected:.4f}, "
            f"got {result.item():.4f} (rel_err={rel_err:.1%})"
        )

    # min_value=0.1: used by inv_dot_a/b in visibility mask
    r_01 = reciprocal(x, min_value=0.1, max_value=20.0)
    for v in [0.1, 0.3, 0.5, 1.0, 2.0, 10.0]:
        result = r_01.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = 1.0 / v
        rel_err = abs(result.item() - expected) / expected
        assert rel_err < 0.05, (
            f"reciprocal(min=0.1): 1/{v} = {expected:.4f}, "
            f"got {result.item():.4f} (rel_err={rel_err:.1%})"
        )


def test_floor_int():
    x = create_input("x", 1)
    f = floor_int(x, min_value=0, max_value=5)
    # Exact at integers
    for v in range(6):
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[float(v)]])})
        assert abs(result.item() - v) < 0.01, f"floor({v}) = {v}, got {result.item()}"
    # Between integers
    for v, expected in [(2.3, 2.0), (2.7, 2.0), (4.9, 4.0)]:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        assert (
            abs(result.item() - expected) < 0.01
        ), f"floor({v}) = {expected}, got {result.item()}"


def test_floor_int_negative():
    x = create_input("x", 1)
    f = floor_int(x, min_value=-3, max_value=3)
    for v, expected in [(-2.5, -3.0), (-1.0, -1.0), (0.0, 0.0), (1.5, 1.0)]:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        assert (
            abs(result.item() - expected) < 0.01
        ), f"floor({v}) = {expected}, got {result.item()}"


def test_ceil_int():
    x = create_input("x", 1)
    c = ceil_int(x, min_value=0, max_value=5)
    # Exact at integers
    for v in range(6):
        result = c.compute(n_pos=1, input_values={"x": torch.tensor([[float(v)]])})
        assert abs(result.item() - v) < 0.01, f"ceil({v}) = {v}, got {result.item()}"
    # Between integers
    for v, expected in [(2.3, 3.0), (2.7, 3.0), (0.1, 1.0)]:
        result = c.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        assert (
            abs(result.item() - expected) < 0.01
        ), f"ceil({v}) = {expected}, got {result.item()}"


def test_floor_int_wide_range_large_magnitude():
    """Regression: the old slope-change ReLU staircase summed ~n terms of
    magnitude ~s*x in one projection, overflowing fp32's 2^24 exact-integer
    limit and collapsing to ~0 above x ~ 2^24/s (e.g. floor_int(40000.4,
    [0,65535], s=1000) returned 0.0). The saturating-step staircase keeps every
    partial sum in [0, n] so it stays exact at any magnitude/sharpness."""
    import math

    # Mid-bin values (0.4 above an integer, clear of any ramp for s >= 10) so the
    # only thing under test is the staircase summation, not near-boundary ramps.
    x = create_input("x", 1)
    for s in (10.0, 1000.0):
        f = floor_int(x, min_value=0, max_value=65535, sharpness=s)
        for v in (10.4, 1000.4, 16800.4, 40000.4, 62195.4, 65534.4):
            r = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])}).item()
            assert abs(r - math.floor(v)) < 0.01, (
                f"floor_int({v}, [0,65535], s={s}) should be {math.floor(v)}, "
                f"got {r}"
            )


def test_floor_int_byte_range_exact_high_sharpness():
    """Exactness guard over the [0,255] byte range at high sharpness — the DOOM
    digit-quad high byte floor(q/256). The saturating-step form must stay exact
    across the whole range (the old form's partial sums grew with the step count
    and position and could lose ~1-2 units near the top)."""
    import math

    x = create_input("x", 1)
    f = floor_int(x, min_value=0, max_value=255, sharpness=1000.0)
    for v in (0.4, 1.4, 100.6, 127.95, 200.3, 242.953, 254.6, 255.0):
        r = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])}).item()
        assert abs(r - math.floor(v)) < 0.01, f"floor_int({v}, [0,255]) got {r}"


def test_ceil_int_wide_range_large_magnitude():
    """ceil_int = -floor_int(-x); inherits the saturating-step fix."""
    import math

    x = create_input("x", 1)
    c = ceil_int(x, min_value=0, max_value=65535)
    for v in (10.4, 1000.4, 40000.4, 62195.98):
        r = c.compute(n_pos=1, input_values={"x": torch.tensor([[v]])}).item()
        assert (
            abs(r - math.ceil(v)) < 0.01
        ), f"ceil_int({v}) should be {math.ceil(v)}, got {r}"


def test_ceil_int_high_sharpness_screen_y():
    """ceil_int must forward sharpness to its inner floor_int (the DOOM screen-y
    CEIL_Y staircase). At sharpness=10000 the ramp is 1e-4 wide, so a raw screen-y
    carrying ~0.077 of piecewise-linear multiply noise stays in the flat zone and
    ceils to an exact integer instead of interpolating across a scanline boundary.
    Covers the narrow [0,49] and wide [0,128] / [-128,49] ranges H actually uses."""
    import math

    x = create_input("x", 1)
    # Mid-bin offsets (well outside the 1e-4 ramp) over the screen-y ranges.
    ranges = [(0, 49), (0, 128), (-128, 49)]
    offsets = (0.077, 0.4, -0.077, -0.4, 0.953)
    for lo, hi in ranges:
        c = ceil_int(x, min_value=lo, max_value=hi, sharpness=10000.0)
        for k in range(lo, hi):
            for off in offsets:
                v = k + off
                if not (lo <= v <= hi):
                    continue
                r = c.compute(n_pos=1, input_values={"x": torch.tensor([[v]])}).item()
                assert abs(r - math.ceil(v)) < 0.01, (
                    f"ceil_int({v}, [{lo},{hi}], s=10000) should be "
                    f"{math.ceil(v)}, got {r}"
                )


def test_clamp():
    """clamp(x, lo, hi) clamps to [lo, hi] and passes through in between."""
    x = create_input("x", 1)
    out = clamp(x, 2.0, 8.0)

    cases = [
        (-5.0, 2.0),  # below lo → lo
        (0.0, 2.0),  # below lo → lo
        (2.0, 2.0),  # at lo → lo
        (5.0, 5.0),  # in range → identity
        (8.0, 8.0),  # at hi → hi
        (15.0, 8.0),  # above hi → hi
        (100.0, 8.0),  # far above → hi
    ]
    for x_val, expected in cases:
        result = out.compute(
            n_pos=1, input_values={"x": torch.tensor([[x_val]])}
        ).item()
        assert (
            abs(result - expected) < 0.15
        ), f"clamp({x_val}, 2, 8): expected {expected}, got {result:.4f}"


def test_piecewise_linear_vector():
    """piecewise_linear with vector-valued fn looks up vector values from a scalar key."""
    import math

    x = create_input("x", 1)

    # 4 breakpoints, 3-dimensional output (cos, sin, linear ramp)
    breakpoints = [0.0, 1.0, 2.0, 3.0]
    out = piecewise_linear(
        x,
        breakpoints,
        lambda t: [
            math.cos(t * math.pi / 2),
            math.sin(t * math.pi / 2),
            10.0 * (t + 1),
        ],
    )
    assert len(out) == 3

    cases = [
        (0.0, [1.0, 0.0, 10.0]),
        (1.0, [0.0, 1.0, 20.0]),
        (2.0, [-1.0, 0.0, 30.0]),
        (3.0, [0.0, -1.0, 40.0]),
        (0.5, [0.5, 0.5, 15.0]),  # interpolated
        (1.5, [-0.5, 0.5, 25.0]),  # interpolated
    ]
    for x_val, expected in cases:
        result = (
            out.compute(n_pos=1, input_values={"x": torch.tensor([[x_val]])})
            .squeeze(0)
            .tolist()
        )
        for j, (r, e) in enumerate(zip(result, expected)):
            assert (
                abs(r - e) < 0.01
            ), f"piecewise_linear({x_val})[{j}]: expected {e}, got {r:.4f}"
