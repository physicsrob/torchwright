import torch
from torchwright import ops
from torchwright.ops.arithmetic_ops import (
    add_const,
    bool_to_01,
    clamp,
    relu_add,
    compare,
    negate,
    subtract,
    multiply_const,
    mod_const,
    piecewise_linear,
    square,
    multiply_integers,
    reciprocal,
    log,
    log_abs,
    exp,
    floor_int,
    ceil_int,
    signed_multiply,
    reduce_min,
    reduce_max,
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
    """ops.abs on a 3-wide node with positive, negative, and zero."""
    x = create_input("x", 3)
    out = ops.abs(x)
    vals = torch.tensor([[5.0, -3.0, 0.0]])
    result = out.compute(n_pos=1, input_values={"x": vals})
    assert torch.allclose(result, torch.tensor([[5.0, 3.0, 0.0]]))


def test_abs_scalar():
    """ops.abs on scalar inputs."""
    x = create_input("x", 1)
    out = ops.abs(x)
    for v in [-7.0, 0.0, 4.5]:
        result = out.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        assert abs(result.item() - abs(v)) < 0.01


def test_min():
    a = create_input("a", 3)
    b = create_input("b", 3)
    out = ops.min(a, b)
    va = torch.tensor([[5.0, -2.0, 7.0]])
    vb = torch.tensor([[3.0, 1.0, 7.0]])
    result = out.compute(n_pos=1, input_values={"a": va, "b": vb})
    expected = torch.min(va, vb)
    assert torch.allclose(result, expected, atol=0.01)


def test_max():
    a = create_input("a", 3)
    b = create_input("b", 3)
    out = ops.max(a, b)
    va = torch.tensor([[5.0, -2.0, 7.0]])
    vb = torch.tensor([[3.0, 1.0, 7.0]])
    result = out.compute(n_pos=1, input_values={"a": va, "b": vb})
    expected = torch.max(va, vb)
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


def test_log_endpoints_near_exact():
    """log(min) and log(max) match closed-form values within FP slack.

    The endpoints are pinned via ``breakpoints[0] = min_value`` and
    ``breakpoints[-1] = max_value``, so there's no interpolation error
    at the ends — only float32 representation of the breakpoint and
    accumulated matmul rounding.
    """
    import math

    x = create_input("x", 1)
    lo, hi = 0.01, 100.0
    f = log(x, min_value=lo, max_value=hi, n_breakpoints=256)

    for v in [lo, hi]:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = math.log(v)
        # Accumulated matmul rounding across ~256 active ReLUs at the
        # high end is the dominant term here, not interpolation.
        assert abs(result.item() - expected) < 3e-3, (
            f"log({v}) = {expected:.6f}, got {result.item():.6f}"
        )


def test_log_accuracy_4_decades():
    """log over [0.01, 100] (4 decades) with 256 BPs has bounded abs error.

    The geometric BP grid has ``ratio ≈ 1.0367``, so the per-cell
    linear-interpolation bound is ``(ratio-1)²/8 ≈ 1.7e-4``.  The
    observed end-to-end error is dominated by float32 matmul
    accumulation across ~256 active ReLUs at the high end of the
    range, which empirically tops out near 2e-3.
    """
    import math

    x = create_input("x", 1)
    f = log(x, min_value=0.01, max_value=100.0, n_breakpoints=256)

    # Sample a mix of near-breakpoint and interior values across decades.
    test_values = [0.012, 0.05, 0.1, 0.317, 1.0, 2.71828, 10.0, 33.3, 99.0]
    worst = 0.0
    for v in test_values:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = math.log(v)
        err = abs(result.item() - expected)
        worst = max(worst, err)
        assert err < 3e-3, (
            f"log({v}) = {expected:.6f}, got {result.item():.6f} (abs_err={err:.2e})"
        )


def test_log_wide_range_6_decades():
    """log over [0.01, 30000] (6 decades) — the case that fails for naïve PWL.

    Without sectioning, partial-sum cancellation in the slope-delta
    representation hits float32 ULP at ``(x_max/x_min) · 2⁻²³ ≈ 0.36``,
    multiplied by accumulation factor ~3 giving ~1 absolute. With
    per-decade sectioning, each section's pre-cancellation magnitude
    is bounded by its own ``B_{i+1}/B_i = 10``, so the floor drops to
    ``10 · 2⁻²³ ≈ 1.2e-6`` — six decades of headroom.

    Includes both clean inputs and inputs near section boundaries.
    """
    import math

    x = create_input("x", 1)
    f = log(x, min_value=0.01, max_value=30000.0, n_breakpoints=256)

    # Spread test inputs across all decades, including some near
    # interior boundaries (0.1, 1, 10, 100, 1000, 10000) to exercise
    # the multiply_2d blending in ramp zones.
    test_values = [
        0.012, 0.05, 0.099, 0.105,  # section 0 + boundary 0.1
        0.3, 0.95, 1.005,           # section 1 + boundary 1
        3.0, 9.5, 10.05,            # section 2 + boundary 10
        50.0, 99.0, 100.5,          # section 3 + boundary 100
        500.0, 999.0, 1001.0,       # section 4 + boundary 1000
        5000.0, 9999.0, 10010.0,    # section 5 + boundary 10000
        20000.0, 29999.0,           # section 6
    ]
    worst = 0.0
    worst_v = None
    for v in test_values:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = math.log(v)
        err = abs(result.item() - expected)
        if err > worst:
            worst = err
            worst_v = v
        # 1e-2 absolute is generous — the design floor is ~1e-5;
        # blending in narrow ramp zones contributes the dominant noise.
        assert err < 1e-2, (
            f"log({v}) = {expected:.6f}, got {result.item():.6f} (abs_err={err:.2e})"
        )


def test_log_extreme_range_7_decades():
    """log over [0.001, 30000] (7+ decades) — naïve PWL fails by ~9 absolute.

    Demonstrates that the precision floor is now controlled by section
    width, not overall range.
    """
    import math

    x = create_input("x", 1)
    f = log(x, min_value=0.001, max_value=30000.0, n_breakpoints=256)

    test_values = [0.002, 0.05, 1.0, 100.0, 5000.0, 28000.0]
    for v in test_values:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = math.log(v)
        err = abs(result.item() - expected)
        assert err < 1e-2, (
            f"log({v}) = {expected:.6f}, got {result.item():.6f} (abs_err={err:.2e})"
        )


def _log_abs_ref(x: float, min_abs: float, max_abs: float) -> float:
    """Math reference: log(clamp(|x|, min_abs, max_abs))."""
    import math

    return math.log(min(max(abs(x), min_abs), max_abs))


def test_log_abs_v_bottom_at_zero():
    """x = 0 lands in the flat V-bottom: output = log(min_abs)."""
    import math

    x = create_input("x", 1)
    f = log_abs(x, min_abs=0.1, max_abs=100.0)
    result = f.compute(n_pos=1, input_values={"x": torch.tensor([[0.0]])})
    expected = math.log(0.1)
    assert abs(result.item() - expected) < 1e-3, (
        f"log_abs(0) = {expected:.6f}, got {result.item():.6f}"
    )


def test_log_abs_flat_zone():
    """|x| <= min_abs is flat at log(min_abs)."""
    import math

    x = create_input("x", 1)
    min_abs, max_abs = 0.1, 100.0
    f = log_abs(x, min_abs=min_abs, max_abs=max_abs)
    expected = math.log(min_abs)

    # Inside the flat zone (|x| < min_abs).
    for v in [-min_abs / 2, -min_abs / 4, 0.0, min_abs / 4, min_abs / 2]:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        assert abs(result.item() - expected) < 1e-3, (
            f"log_abs({v}) flat-zone: expected {expected:.6f}, "
            f"got {result.item():.6f}"
        )


def test_log_abs_at_min_abs_boundary():
    """At |x| = min_abs (the V-bottom edge), output = log(min_abs)."""
    import math

    x = create_input("x", 1)
    min_abs, max_abs = 0.1, 100.0
    f = log_abs(x, min_abs=min_abs, max_abs=max_abs)
    expected = math.log(min_abs)

    for v in [-min_abs, min_abs]:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        assert abs(result.item() - expected) < 1e-3, (
            f"log_abs({v}) boundary: expected {expected:.6f}, "
            f"got {result.item():.6f}"
        )


def test_log_abs_just_past_flat_zone():
    """At |x| = 2*min_abs, output ≈ log(2*min_abs) — out of flat zone."""
    import math

    x = create_input("x", 1)
    min_abs, max_abs = 0.1, 100.0
    f = log_abs(x, min_abs=min_abs, max_abs=max_abs)
    expected = math.log(2 * min_abs)

    for v in [-2 * min_abs, 2 * min_abs]:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        # Tighter tolerance — inside the active log-curve region.
        assert abs(result.item() - expected) < 1e-3, (
            f"log_abs({v}) past-flat: expected {expected:.6f}, "
            f"got {result.item():.6f}"
        )


def test_log_abs_at_max_abs_boundary():
    """At |x| = max_abs, output = log(max_abs)."""
    import math

    x = create_input("x", 1)
    min_abs, max_abs = 0.1, 100.0
    f = log_abs(x, min_abs=min_abs, max_abs=max_abs)
    expected = math.log(max_abs)

    for v in [-max_abs, max_abs]:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        assert abs(result.item() - expected) < 1e-3, (
            f"log_abs({v}) max boundary: expected {expected:.6f}, "
            f"got {result.item():.6f}"
        )


def test_log_abs_clamps_outside_max():
    """|x| > max_abs clamps to log(max_abs)."""
    import math

    x = create_input("x", 1)
    min_abs, max_abs = 0.1, 100.0
    f = log_abs(x, min_abs=min_abs, max_abs=max_abs)
    expected = math.log(max_abs)

    for v in [-2 * max_abs, -1.5 * max_abs, 1.5 * max_abs, 2 * max_abs]:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        assert abs(result.item() - expected) < 1e-3, (
            f"log_abs({v}) clamp: expected {expected:.6f}, "
            f"got {result.item():.6f}"
        )


def test_log_abs_symmetry():
    """log_abs(x) ≈ log_abs(-x) by construction (mirrored breakpoints).

    The breakpoint set and per-vertex values are exactly symmetric
    around 0, but the matmul accumulates ReLU contributions in a fixed
    order, so the FP output isn't bit-exact symmetric — left and right
    arm ReLUs sum into the same accumulator with different cancellation
    patterns. The residual asymmetry is at the float32 ULP scale of the
    partial sum (the V-spike contributions reach ~10³ in magnitude, so
    accumulator ULP is ~10⁻⁴ but mostly correlates between ±x; observed
    asymmetry is ~10⁻⁵). Well below the 1e-3 absolute precision target.
    """
    x = create_input("x", 1)
    f = log_abs(x, min_abs=0.1, max_abs=100.0)

    for v in [0.05, 0.1, 0.5, 1.0, 7.3, 50.0, 99.0, 150.0]:
        pos = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        neg = f.compute(n_pos=1, input_values={"x": torch.tensor([[-v]])})
        assert abs(pos.item() - neg.item()) < 5e-4, (
            f"log_abs({v}) = {pos.item():.7f} vs log_abs(-{v}) = "
            f"{neg.item():.7f} (asymmetry {abs(pos.item() - neg.item()):.2e})"
        )


def test_log_abs_precision_signed_distribution():
    """Worst-case error across signed log-uniform |x| distribution.

    Mirrors the spec's measurement protocol: |x| log-uniform on
    [min_abs, max_abs], sign uniform ±, plus structured grid points
    near 0 and near ±min_abs to stress the V-bottom transition.
    Targets:
      - max abs error < 1e-3 (soft) / < 5e-3 (hard ceiling)
      - mean abs error < 1e-4 (soft) / < 5e-4 (hard ceiling)
    """
    import math

    x = create_input("x", 1)
    min_abs, max_abs = 0.1, 100.0
    f = log_abs(x, min_abs=min_abs, max_abs=max_abs)

    gen = torch.Generator().manual_seed(0)
    n_random = 1024
    # Log-uniform |x| in [min_abs, max_abs].
    log_lo = math.log(min_abs)
    log_hi = math.log(max_abs)
    u = torch.rand(n_random, generator=gen)
    abs_x = torch.exp(log_lo + u * (log_hi - log_lo))
    signs = (
        torch.randint(0, 2, (n_random,), generator=gen).to(torch.float32) * 2.0
        - 1.0
    )
    random_xs = (signs * abs_x).tolist()

    # Structured grid concentrating near the V-bottom.
    structured = [0.0]
    for v in [
        min_abs * 0.1,
        min_abs * 0.5,
        min_abs * 0.9,
        min_abs,
        min_abs * 1.1,
        min_abs * 1.5,
        min_abs * 2.0,
    ]:
        structured.extend([-v, v])
    structured.extend([-max_abs, max_abs, -max_abs * 0.5, max_abs * 0.5])

    test_xs = random_xs + structured

    abs_errs = []
    worst = 0.0
    worst_v = None
    for v in test_xs:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = _log_abs_ref(v, min_abs, max_abs)
        err = abs(result.item() - expected)
        abs_errs.append(err)
        if err > worst:
            worst = err
            worst_v = v

    mean_err = sum(abs_errs) / len(abs_errs)
    # Hard-ceiling assertions — fail loud if precision regresses.
    assert worst < 5e-3, (
        f"max abs error {worst:.2e} (worst at x={worst_v}) exceeds 5e-3"
    )
    assert mean_err < 5e-4, f"mean abs error {mean_err:.2e} exceeds 5e-4"


def test_log_abs_wide_range_fallback():
    """For ratios > 10⁴, log_abs falls back to abs+sectioned-log.

    Verifies the fallback path produces correct values across the wider
    range. Precision target: 1e-2 absolute (looser than the single path,
    matching the sectioned log floor at 6+ decades).
    """
    import math

    x = create_input("x", 1)
    # Triggers the wide-range fallback (ratio = 6e6 > 10⁴).
    min_abs, max_abs = 0.005, 30000.0
    f = log_abs(x, min_abs=min_abs, max_abs=max_abs)

    test_values = [-29000.0, -100.0, -1.0, -0.01, 0.0, 0.01, 1.0, 100.0, 29000.0]
    for v in test_values:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = _log_abs_ref(v, min_abs, max_abs)
        err = abs(result.item() - expected)
        assert err < 1e-2, (
            f"log_abs({v}) wide: expected {expected:.6f}, "
            f"got {result.item():.6f} (abs_err={err:.2e})"
        )


def test_log_accuracy_2_decades():
    """log over [0.1, 10] (2 decades) with 256 BPs is much tighter.

    With a 2-decade range, ``ratio ≈ 1.0182`` — interpolation error
    drops 4× and accumulator depth halves, so we expect <5e-4 absolute
    error on log values that themselves span [-2.3, 2.3].
    """
    import math

    x = create_input("x", 1)
    f = log(x, min_value=0.1, max_value=10.0, n_breakpoints=256)

    test_values = [0.11, 0.3, 0.7, 1.0, 1.5, 3.0, 5.5, 9.5]
    for v in test_values:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = math.log(v)
        err = abs(result.item() - expected)
        assert err < 5e-4, (
            f"log({v}) = {expected:.6f}, got {result.item():.6f} (abs_err={err:.2e})"
        )


def test_exp_exact_at_breakpoints():
    """exp(x) is interpolation-free at the breakpoint values."""
    import math

    x = create_input("x", 1)
    lo, hi, n = -5.0, 5.0, 256
    f = exp(x, min_value=lo, max_value=hi, n_breakpoints=n)

    step = (hi - lo) / (n - 1)
    for k in [0, 1, n // 4, n // 2, n - 2, n - 1]:
        v = lo + k * step if k != n - 1 else hi
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = math.exp(v)
        # Relative tolerance because exp spans many orders of magnitude.
        rel_err = abs(result.item() - expected) / expected
        assert rel_err < 1e-4, (
            f"exp({v:.4f}) = {expected:.6f}, got {result.item():.6f} "
            f"(rel_err={rel_err:.2e})"
        )


def test_exp_relative_error():
    """exp over [-5, 5] with 256 BPs has bounded relative error.

    Theoretical bound for uniform BPs is ~(Δx)²/8 per cell.  With
    Δx ≈ 0.0392, that's ~1.9e-4.  Allow 1e-3 for FP slack.
    """
    import math

    x = create_input("x", 1)
    f = exp(x, min_value=-5.0, max_value=5.0, n_breakpoints=256)

    test_values = [-4.7, -2.3, -1.0, 0.0, 0.5, 1.7, 3.14, 4.9]
    for v in test_values:
        result = f.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        expected = math.exp(v)
        rel_err = abs(result.item() - expected) / expected
        assert rel_err < 1e-3, (
            f"exp({v}) = {expected:.6f}, got {result.item():.6f} "
            f"(rel_err={rel_err:.2e})"
        )


def test_exp_log_roundtrip():
    """exp(log(x)) ≈ x — the foundational identity for log-space chains."""
    from torchwright.ops.inout_nodes import create_input as _ci

    x = _ci("x", 1)
    log_x = log(x, min_value=0.01, max_value=100.0, n_breakpoints=256)
    # exp's input range covers log([0.01, 100]) = [-4.605, 4.605].
    out = exp(log_x, min_value=-4.7, max_value=4.7, n_breakpoints=256)

    test_values = [0.02, 0.1, 0.5, 1.0, 2.0, 7.5, 30.0, 80.0]
    for v in test_values:
        result = out.compute(n_pos=1, input_values={"x": torch.tensor([[v]])})
        rel_err = abs(result.item() - v) / v
        # Two PWL stages compound — bound at 1% (each contributes <0.1%).
        assert rel_err < 1e-2, (
            f"exp(log({v})) = {v}, got {result.item():.6f} "
            f"(rel_err={rel_err:.2e})"
        )


def test_log_exp_multiplication():
    """exp(log(a) + log(b)) ≈ a*b — the headline use case for log-space ops.

    Demonstrates that an end-to-end multiply via log/exp lands within
    a couple percent of the true product across operands spanning two
    decades.
    """
    a = create_input("a", 1)
    b = create_input("b", 1)
    log_a = log(a, min_value=0.01, max_value=100.0, n_breakpoints=256)
    log_b = log(b, min_value=0.01, max_value=100.0, n_breakpoints=256)
    # log(a) + log(b) ranges over [2·log(0.01), 2·log(100)] = [-9.21, 9.21].
    from torchwright.ops.arithmetic_ops import add

    log_ab = add(log_a, log_b)
    out = exp(log_ab, min_value=-9.3, max_value=9.3, n_breakpoints=256)

    test_pairs = [
        (0.05, 0.5),
        (0.5, 2.0),
        (1.0, 1.0),
        (3.0, 7.0),
        (10.0, 0.1),
        (50.0, 1.5),
        (80.0, 0.8),
    ]
    for av, bv in test_pairs:
        result = out.compute(
            n_pos=1,
            input_values={
                "a": torch.tensor([[av]]),
                "b": torch.tensor([[bv]]),
            },
        )
        expected = av * bv
        rel_err = abs(result.item() - expected) / expected
        # Three PWL stages (log, log, exp) compound — bound at 2%.
        assert rel_err < 2e-2, (
            f"exp(log({av})+log({bv})) = {expected}, "
            f"got {result.item():.6f} (rel_err={rel_err:.2e})"
        )


def test_log_clamps_outside_range():
    """log clamps to endpoint values outside [min_value, max_value]."""
    import math

    x = create_input("x", 1)
    f = log(x, min_value=0.1, max_value=10.0, n_breakpoints=128)

    # Below min: clamp to log(min)
    result = f.compute(n_pos=1, input_values={"x": torch.tensor([[0.05]])})
    assert abs(result.item() - math.log(0.1)) < 1e-3
    # Above max: clamp to log(max)
    result = f.compute(n_pos=1, input_values={"x": torch.tensor([[20.0]])})
    assert abs(result.item() - math.log(10.0)) < 1e-3


def test_exp_clamps_outside_range():
    """exp clamps to endpoint values outside [min_value, max_value]."""
    import math

    x = create_input("x", 1)
    f = exp(x, min_value=-2.0, max_value=2.0, n_breakpoints=128)

    # Below min: clamp to exp(min)
    result = f.compute(n_pos=1, input_values={"x": torch.tensor([[-5.0]])})
    rel_err = abs(result.item() - math.exp(-2.0)) / math.exp(-2.0)
    assert rel_err < 1e-3
    # Above max: clamp to exp(max)
    result = f.compute(n_pos=1, input_values={"x": torch.tensor([[5.0]])})
    rel_err = abs(result.item() - math.exp(2.0)) / math.exp(2.0)
    assert rel_err < 1e-3


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
        assert abs(r - math.ceil(v)) < 0.01, f"ceil_int({v}) should be {math.ceil(v)}, got {r}"


def test_signed_multiply():
    """Test all sign combinations."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    prod = signed_multiply(a, b, max_abs1=10.0, max_abs2=10.0, step=1.0)
    cases = [
        (3.0, 4.0, 12.0),
        (-3.0, 4.0, -12.0),
        (3.0, -4.0, -12.0),
        (-3.0, -4.0, 12.0),
        (0.0, 5.0, 0.0),
        (5.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    for va, vb, expected in cases:
        result = prod.compute(
            n_pos=1,
            input_values={
                "a": torch.tensor([[va]]),
                "b": torch.tensor([[vb]]),
            },
        )
        assert (
            abs(result.item() - expected) < 0.5
        ), f"{va}*{vb} = {expected}, got {result.item()}"


def test_signed_multiply_with_clamp():
    """max_abs_output clamps the result."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    prod = signed_multiply(
        a, b, max_abs1=10.0, max_abs2=10.0, step=1.0, max_abs_output=20.0
    )
    result = prod.compute(
        n_pos=1,
        input_values={"a": torch.tensor([[5.0]]), "b": torch.tensor([[5.0]])},
    )
    assert abs(result.item() - 20.0) < 0.5  # 5*5=25, clamped to 20


def test_signed_multiply_strategies():
    """Both deep and shallow strategies produce correct results."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    cases = [
        (3.0, 4.0, 12.0),
        (-3.0, 4.0, -12.0),
        (3.0, -4.0, -12.0),
        (-3.0, -4.0, 12.0),
        (5.0, -2.0, -10.0),
        (0.0, 7.0, 0.0),
    ]
    for strategy in ("deep", "shallow"):
        prod = signed_multiply(
            a,
            b,
            max_abs1=10.0,
            max_abs2=10.0,
            step=1.0,
            strategy=strategy,
        )
        for va, vb, expected in cases:
            result = prod.compute(
                n_pos=1,
                input_values={
                    "a": torch.tensor([[va]]),
                    "b": torch.tensor([[vb]]),
                },
            )
            assert abs(result.item() - expected) < 0.5, (
                f"strategy={strategy} {va}*{vb}: expected {expected}, "
                f"got {result.item()}"
            )


def test_multiply_integers_strategies():
    """Both strategies produce correct integer products."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    for strategy in ("deep", "shallow"):
        prod = multiply_integers(a, b, max_value=9, strategy=strategy)
        for va in range(10):
            for vb in range(10):
                result = prod.compute(
                    n_pos=1,
                    input_values={
                        "a": torch.tensor([[float(va)]]),
                        "b": torch.tensor([[float(vb)]]),
                    },
                )
                expected = va * vb
                assert abs(result.item() - expected) < 0.5, (
                    f"strategy={strategy} {va}*{vb}: expected {expected}, "
                    f"got {result.item()}"
                )


def test_reduce_min():
    """Find minimum key and its associated value."""
    keys = [create_input(f"k{i}", 1, value_range=(0.0, 100.0)) for i in range(4)]
    vals = [create_input(f"v{i}", 2, value_range=(0.0, 1000.0)) for i in range(4)]
    win_k, win_v = reduce_min(keys, vals)

    input_values = {
        "k0": torch.tensor([[10.0]]),
        "k1": torch.tensor([[3.0]]),
        "k2": torch.tensor([[7.0]]),
        "k3": torch.tensor([[5.0]]),
        "v0": torch.tensor([[100.0, 200.0]]),
        "v1": torch.tensor([[300.0, 400.0]]),
        "v2": torch.tensor([[500.0, 600.0]]),
        "v3": torch.tensor([[700.0, 800.0]]),
    }

    result_k = win_k.compute(n_pos=1, input_values=input_values)
    result_v = win_v.compute(n_pos=1, input_values=input_values)
    assert abs(result_k.item() - 3.0) < 0.5
    assert torch.allclose(result_v, torch.tensor([[300.0, 400.0]]), atol=1.0)


def test_reduce_max():
    """Find maximum key and its associated value."""
    keys = [create_input(f"k{i}", 1, value_range=(0.0, 100.0)) for i in range(4)]
    vals = [create_input(f"v{i}", 2, value_range=(0.0, 1000.0)) for i in range(4)]
    win_k, win_v = reduce_max(keys, vals)

    input_values = {
        "k0": torch.tensor([[10.0]]),
        "k1": torch.tensor([[3.0]]),
        "k2": torch.tensor([[7.0]]),
        "k3": torch.tensor([[5.0]]),
        "v0": torch.tensor([[100.0, 200.0]]),
        "v1": torch.tensor([[300.0, 400.0]]),
        "v2": torch.tensor([[500.0, 600.0]]),
        "v3": torch.tensor([[700.0, 800.0]]),
    }

    result_k = win_k.compute(n_pos=1, input_values=input_values)
    result_v = win_v.compute(n_pos=1, input_values=input_values)
    assert abs(result_k.item() - 10.0) < 0.5
    assert torch.allclose(result_v, torch.tensor([[100.0, 200.0]]), atol=1.0)


def test_reduce_min_single():
    """N=1: pass-through."""
    k = create_input("k", 1)
    v = create_input("v", 3)
    win_k, win_v = reduce_min([k], [v])
    input_values = {
        "k": torch.tensor([[42.0]]),
        "v": torch.tensor([[1.0, 2.0, 3.0]]),
    }
    assert abs(win_k.compute(n_pos=1, input_values=input_values).item() - 42.0) < 0.01
    assert torch.allclose(
        win_v.compute(n_pos=1, input_values=input_values),
        torch.tensor([[1.0, 2.0, 3.0]]),
    )


def test_reduce_min_odd():
    """N=3: odd element passes through."""
    keys = [create_input(f"k{i}", 1, value_range=(0.0, 100.0)) for i in range(3)]
    vals = [create_input(f"v{i}", 1, value_range=(0.0, 1000.0)) for i in range(3)]
    win_k, win_v = reduce_min(keys, vals)

    input_values = {
        "k0": torch.tensor([[5.0]]),
        "k1": torch.tensor([[2.0]]),
        "k2": torch.tensor([[8.0]]),
        "v0": torch.tensor([[50.0]]),
        "v1": torch.tensor([[20.0]]),
        "v2": torch.tensor([[80.0]]),
    }
    result_k = win_k.compute(n_pos=1, input_values=input_values)
    result_v = win_v.compute(n_pos=1, input_values=input_values)
    assert abs(result_k.item() - 2.0) < 0.5
    assert abs(result_v.item() - 20.0) < 1.0


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
