"""Tests for the attached-check predicate helpers (graph/asserts.py).

Covers:

* Each predicate helper accepts valid values and rejects invalid ones
  (at reference-eval time via ``node.compute``).
* Failure messages include the site's annotation and the helper's
  detail string.
* Checks cost nothing compiled — a graph compiles to the same forward
  output with or without them.
* ``check_asserts_on_compiled`` fires a predicate when the compiled
  transformer's value violates an invariant that reference math
  satisfies (synthetically constructed, to prove the probe path
  works).
"""

import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import check_asserts_on_compiled, reference_eval
from torchwright.graph import Concatenate, NodeValueType, annotate
from torchwright.graph.asserts import (
    assert_01,
    assert_bool,
    assert_distinct_across,
    assert_in_range,
    assert_matches_value_type,
    assert_onehot,
    assert_picked_from,
    assert_score_gap_at_least,
    assert_strictly_less,
    assert_unique_values,
    collect_asserts,
)
from torchwright.ops.inout_nodes import create_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval(node, inputs: dict, n_pos: int = 1) -> torch.Tensor:
    """Shortcut for reference-eval that returns the root value."""
    values = reference_eval(node, inputs, n_pos)
    return values[node]


# ---------------------------------------------------------------------------
# assert_in_range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values,should_raise",
    [
        ([-0.5, 0.0, 0.9], False),
        ([-0.5, 0.0, 2.5], True),
    ],
    ids=["accepts_within_bounds", "rejects_outside_bounds"],
)
def test_assert_in_range(values, should_raise):
    x = create_input("x", 3)
    wrapped = assert_in_range(x, -1.0, 1.0)
    inputs = {"x": torch.tensor([values])}
    if should_raise:
        with pytest.raises(AssertionError, match=r"\[-1\.0, 1\.0\]"):
            _eval(wrapped, inputs)
    else:
        out = _eval(wrapped, inputs)
        assert torch.allclose(out, torch.tensor([values]))


# ---------------------------------------------------------------------------
# assert_bool (±1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values,should_raise",
    [
        ([1.0, -1.0, 1.0, -1.0], False),
        ([1.0, 0.0, -1.0], True),
    ],
    ids=["accepts_pm1", "rejects_non_pm1"],
)
def test_assert_bool(values, should_raise):
    x = create_input("x", len(values))
    wrapped = assert_bool(x)
    inputs = {"x": torch.tensor([values])}
    if should_raise:
        with pytest.raises(AssertionError, match=r"±1"):
            _eval(wrapped, inputs)
    else:
        _eval(wrapped, inputs)


# ---------------------------------------------------------------------------
# assert_01 ({0,1})
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values,should_raise",
    [
        ([0.0, 1.0, 0.0, 1.0], False),
        ([0.0, 0.5, 1.0], True),
    ],
    ids=["accepts_zero_one", "rejects_other"],
)
def test_assert_01(values, should_raise):
    x = create_input("x", len(values))
    wrapped = assert_01(x)
    inputs = {"x": torch.tensor([values])}
    if should_raise:
        with pytest.raises(AssertionError, match=r"\{0,1\}"):
            _eval(wrapped, inputs)
    else:
        _eval(wrapped, inputs)


# ---------------------------------------------------------------------------
# assert_onehot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values,should_raise,match",
    [
        ([0.0, 1.0, 0.0, 0.0], False, None),
        ([1.0, 1.0, 0.0, 0.0], True, r"not one-hot"),
        ([0.0, 0.0, 0.0, 0.0], True, None),
    ],
    ids=["accepts_valid", "rejects_double_one", "rejects_all_zero"],
)
def test_assert_onehot(values, should_raise, match):
    x = create_input("x", 4)
    wrapped = assert_onehot(x)
    inputs = {"x": torch.tensor([values])}
    if should_raise:
        kwargs = {"match": match} if match else {}
        with pytest.raises(AssertionError, **kwargs):
            _eval(wrapped, inputs)
    else:
        _eval(wrapped, inputs)


# ---------------------------------------------------------------------------
# assert_strictly_less
# ---------------------------------------------------------------------------


def test_assert_strictly_less_accepts_a_lt_b():
    a = create_input("a", 2)
    b = create_input("b", 2)
    # Returns a wrapper of b; thread it downstream.
    wrapped_b = assert_strictly_less(a, b)
    # Evaluate the wrapped b — should equal b's input.
    out = _eval(
        wrapped_b,
        {
            "a": torch.tensor([[1.0, 5.0]]),
            "b": torch.tensor([[2.0, 10.0]]),
        },
    )
    assert torch.allclose(out, torch.tensor([[2.0, 10.0]]))


def test_assert_strictly_less_rejects_a_ge_b():
    a = create_input("a", 2)
    b = create_input("b", 2)
    wrapped_b = assert_strictly_less(a, b)
    with pytest.raises(AssertionError, match=r"a < b"):
        _eval(
            wrapped_b,
            {
                "a": torch.tensor([[1.0, 20.0]]),
                "b": torch.tensor([[2.0, 10.0]]),  # b[1]=10 < a[1]=20
            },
        )


def test_assert_strictly_less_rejects_equal_at_zero_margin():
    a = create_input("a", 1)
    b = create_input("b", 1)
    wrapped_b = assert_strictly_less(a, b, margin=0.0)
    with pytest.raises(AssertionError):
        _eval(
            wrapped_b,
            {
                "a": torch.tensor([[5.0]]),
                "b": torch.tensor([[5.0]]),
            },
        )


# ---------------------------------------------------------------------------
# assert_unique_values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values,should_raise",
    [
        ([0.0, 1.0, 2.0, 3.0], False),
        ([0.0, 1.0, 1.2], True),  # 1.0 ≈ 1.2 at margin 0.5
    ],
    ids=["accepts_distinct", "rejects_close_pair"],
)
def test_assert_unique_values(values, should_raise):
    x = create_input("x", len(values))
    wrapped = assert_unique_values(x, margin=0.5)
    inputs = {"x": torch.tensor([values])}
    if should_raise:
        with pytest.raises(AssertionError, match=r"duplicate values"):
            _eval(wrapped, inputs)
    else:
        _eval(wrapped, inputs)


# ---------------------------------------------------------------------------
# Message formatting — annotation tagging
# ---------------------------------------------------------------------------


def test_failure_message_includes_annotation():
    x = create_input("x", 1)
    with annotate("bsp/rank"):
        wrapped = assert_in_range(x, 0.0, 100.0)
    with pytest.raises(AssertionError, match=r"bsp/rank"):
        _eval(wrapped, {"x": torch.tensor([[999.0]])})


# ---------------------------------------------------------------------------
# collect_asserts
# ---------------------------------------------------------------------------


def test_collect_asserts_finds_reachable():
    x = create_input("x", 1)
    y = create_input("y", 1)
    a = assert_in_range(x, 0.0, 1.0)
    b = assert_bool(y)
    root = Concatenate([a, b])
    asserts = collect_asserts(root)
    assert set(asserts) == {x, y}  # the checks live on the nodes themselves


def test_collect_asserts_returns_empty_for_no_asserts():
    x = create_input("x", 1)
    assert collect_asserts(x) == []


# ---------------------------------------------------------------------------
# Checks cost nothing compiled — forward identical with or without
# ---------------------------------------------------------------------------


def _build_simple_graph(with_assert: bool):
    """Simple compilable graph: ``y = clamp(x, -1, 1)`` optionally asserted."""
    from torchwright.ops.relu.arithmetic_ops import clamp

    x = create_input("x", 1)
    clamped = clamp(x, -1.0, 1.0)
    if with_assert:
        clamped = assert_in_range(clamped, -1.0, 1.0)
    return x, clamped


def test_checks_cost_nothing_compiled():
    _, out_a = _build_simple_graph(with_assert=False)
    mod_a = compile_headless(
        out_a,
        d=256,
        d_head=16,
        max_layers=40,
        verbose=False,
    )

    _, out_b = _build_simple_graph(with_assert=True)
    mod_b = compile_headless(
        out_b,
        d=256,
        d_head=16,
        max_layers=40,
        verbose=False,
    )

    # Same layer count (checks don't compile into layers).
    assert mod_a._n_layers == mod_b._n_layers

    # Same output on matching inputs.  ``_build_simple_graph`` uses a
    # width-1 ``x`` (clamp requires a scalar input).
    x_val = torch.tensor([[0.7]], dtype=torch.float32)
    d_input_a = max(s + w for _, s, w in mod_a._input_specs)
    d_input_b = max(s + w for _, s, w in mod_b._input_specs)
    inp_a = torch.zeros(1, d_input_a)
    inp_b = torch.zeros(1, d_input_b)
    for specs, inp in ((mod_a._input_specs, inp_a), (mod_b._input_specs, inp_b)):
        for name, start, width in specs:
            if name == "x":
                inp[:, start : start + width] = x_val
    with torch.no_grad():
        y_a = mod_a(inp_a)
        y_b = mod_b(inp_b)
    assert torch.allclose(y_a, y_b, atol=1e-4), f"Strip changed output: {y_a} vs {y_b}"


# ---------------------------------------------------------------------------
# Compiled-side predicate check
# ---------------------------------------------------------------------------


def test_check_asserts_on_compiled_passes_when_invariant_holds():
    """A predicate that the compiled value satisfies should not raise."""
    from torchwright.ops.relu.arithmetic_ops import clamp

    x = create_input("x", 1)
    clamped = clamp(x, -1.0, 1.0)
    # Assert that the clamped value is in [-1, 1] — it always is.
    out = assert_in_range(clamped, -1.0, 1.0)
    asserts = collect_asserts(out)
    mod = compile_headless(
        out,
        d=256,
        d_head=16,
        max_layers=40,
        verbose=False,
    )

    # Build a valid input.
    d_input = max(s + w for _, s, w in mod._input_specs)
    inp = torch.zeros(1, d_input)
    for name, start, width in mod._input_specs:
        if name == "x":
            inp[:, start : start + width] = 0.3
    # Should not raise.
    check_asserts_on_compiled(
        mod,
        asserts,
        input_values={"x": torch.tensor([[0.3]])},
        n_pos=1,
    )


def test_check_asserts_on_compiled_raises_when_compiled_violates():
    """Predicate passes at reference-eval time, fails at compiled time.

    Construct this by asserting with an atol tighter than the compiled
    approximation's actual error on a bumpy function.  ``piecewise_linear``
    on a jagged lambda gives us a reference value that's exact at
    breakpoints and approximate between them.  We pick an input between
    breakpoints where the compiled approximation will drift beyond our
    atol but reference math is still exact (interpolating).
    """
    from torchwright.ops.relu.arithmetic_ops import piecewise_linear

    x = create_input("x", 1)
    # Jagged function with sharp slope changes — PL approximation will
    # differ meaningfully from the exact cubic between breakpoints.
    bp = [0.0, 0.5, 1.0]
    y = piecewise_linear(x, bp, lambda t: t * t * t)  # cubic
    # Assert |y| <= 0.01.  Reference eval of the assert node sees the
    # *same* PL-interpolated y that the compiled path sees (modulo
    # compile-time fuzz), so this test exercises the compiled path
    # firing on a value that exceeds the bound.
    out = assert_in_range(y, -0.01, 0.01)
    asserts = collect_asserts(out)

    # At x=0.8, reference y = piecewise_linear_interpolation ≈ 0.65.
    # That exceeds bound.  We expect both reference-eval and compiled
    # checks to fire — verify the compiled one.
    inp_val = torch.tensor([[0.8]])

    # Skip reference eval (it'd raise first); just compile and check
    # the compiled side.  Compilation never runs predicates, so the
    # compile succeeds.
    mod = compile_headless(
        out,
        d=256,
        d_head=16,
        max_layers=40,
        verbose=False,
    )
    with pytest.raises(AssertionError):
        check_asserts_on_compiled(
            mod,
            asserts,
            input_values={"x": inp_val},
            n_pos=1,
        )


# ---------------------------------------------------------------------------
# assert_distinct_across — cross-position uniqueness with validity gate
# ---------------------------------------------------------------------------


def test_distinct_across_accepts_distinct_valid_rows():
    """Three valid rows with ranks 0, 1, 2 at margin 0.5 → passes."""
    value = create_input("value", 1)
    where = create_input("where", 1)
    wrapped = assert_distinct_across(value, where, margin=0.5)
    _eval(
        wrapped,
        {
            "value": torch.tensor([[0.0], [1.0], [2.0], [99.0]]),
            "where": torch.tensor([[1.0], [1.0], [1.0], [-1.0]]),  # last row invalid
        },
        n_pos=4,
    )


def test_distinct_across_rejects_tied_valid_rows():
    """Two valid rows with values 1.0 and 1.1 at margin 0.5 → fails."""
    value = create_input("value", 1)
    where = create_input("where", 1)
    wrapped = assert_distinct_across(value, where, margin=0.5)
    with pytest.raises(AssertionError, match=r"valid-subset rows"):
        _eval(
            wrapped,
            {
                "value": torch.tensor([[1.0], [1.1], [99.0]]),
                "where": torch.tensor([[1.0], [1.0], [-1.0]]),
            },
            n_pos=3,
        )


def test_distinct_across_ignores_invalid_rows():
    """Ties among where<0.5 rows are not a violation."""
    value = create_input("value", 1)
    where = create_input("where", 1)
    wrapped = assert_distinct_across(value, where, margin=0.5)
    _eval(
        wrapped,
        {
            "value": torch.tensor([[5.0], [5.0], [5.0], [1.0]]),
            "where": torch.tensor([[0.0], [0.0], [0.0], [1.0]]),  # only row 3 valid
        },
        n_pos=4,
    )


def test_distinct_across_zero_or_one_valid_row_passes():
    """Trivially distinct with < 2 valid rows."""
    value = create_input("value", 1)
    where = create_input("where", 1)
    wrapped = assert_distinct_across(value, where, margin=0.5)
    # Zero valid
    _eval(
        wrapped,
        {
            "value": torch.tensor([[5.0], [5.0]]),
            "where": torch.tensor([[0.0], [0.0]]),
        },
        n_pos=2,
    )
    # One valid
    _eval(
        wrapped,
        {
            "value": torch.tensor([[5.0], [5.0]]),
            "where": torch.tensor([[1.0], [0.0]]),
        },
        n_pos=2,
    )


def test_distinct_across_multi_dim_uses_l_infinity():
    """With d=2 values, the tightest component dominates the margin check."""
    value = create_input("value", 2)
    where = create_input("where", 1)
    wrapped = assert_distinct_across(value, where, margin=0.5)
    # Rows [1, 0] and [1, 0.1] — L∞ distance 0.1 < margin 0.5 → fail.
    with pytest.raises(AssertionError):
        _eval(
            wrapped,
            {
                "value": torch.tensor([[1.0, 0.0], [1.0, 0.1]]),
                "where": torch.tensor([[1.0], [1.0]]),
            },
            n_pos=2,
        )
    # Rows [1, 0] and [1, 2] — L∞ distance 2.0 → pass.
    _eval(
        wrapped,
        {
            "value": torch.tensor([[1.0, 0.0], [1.0, 2.0]]),
            "where": torch.tensor([[1.0], [1.0]]),
        },
        n_pos=2,
    )


def test_distinct_across_returns_value_width():
    """The wrapped result must have the same width as the input value."""
    value = create_input("value", 3)
    where = create_input("where", 1)
    wrapped = assert_distinct_across(value, where, margin=0.5)
    assert len(wrapped) == 3


# ---------------------------------------------------------------------------
# assert_picked_from — attention-concentration
# ---------------------------------------------------------------------------


def test_picked_from_accepts_clean_pick():
    """Result matches one of the valid value rows exactly → passes."""
    result = create_input("result", 2)
    values = create_input("values", 2)
    keys = create_input("keys", 1)
    wrapped = assert_picked_from(result, values, keys, atol=1e-3)
    # At every query row, result = [1.0, 2.0] — which matches value row 1 (valid key).
    _eval(
        wrapped,
        {
            "result": torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
            "values": torch.tensor([[9.9, 9.9], [1.0, 2.0], [3.0, 4.0]]),
            "keys": torch.tensor([[0.0], [1.0], [0.0]]),  # only row 1 valid
        },
        n_pos=3,
    )


def test_picked_from_rejects_blend():
    """Synthetic blend — result is midpoint of two value rows, matches neither."""
    result = create_input("result", 1)
    values = create_input("values", 1)
    keys = create_input("keys", 1)
    wrapped = assert_picked_from(result, values, keys, atol=1e-2)
    # Valid values are 0.0 and 1.0; result = 0.5 matches neither within atol=0.01.
    with pytest.raises(AssertionError, match=r"doesn't match any value row"):
        _eval(
            wrapped,
            {
                "result": torch.tensor([[0.5]]),
                "values": torch.tensor([[0.0]]),
                "keys": torch.tensor([[1.0]]),
            },
            n_pos=1,
        )


def test_picked_from_skips_no_valid_keys():
    """Zero valid key positions → skip (caller's type-boolean convention
    means every query is itself inactive, so there's nothing to check).

    Previously this case raised AssertionError; commit ``ee3047c``
    changed it to a silent skip with a stdout note, because the skip
    fires in decode mode at step positions whose type boolean is 0
    (e.g. a PLAYER_X step for a sort-stage assert keyed on is_wall)
    and asserting against values the consumer never reads was noise.
    """
    result = create_input("result", 1)
    values = create_input("values", 1)
    keys = create_input("keys", 1)
    wrapped = assert_picked_from(result, values, keys, atol=1e-3)
    # Does not raise — the assert is skipped when no key rows are
    # valid, per the "every query inactive" semantic.
    _eval(
        wrapped,
        {
            "result": torch.tensor([[0.5], [0.3]]),
            "values": torch.tensor([[1.0], [2.0]]),
            "keys": torch.tensor([[0.0], [0.0]]),
        },
        n_pos=2,
    )


def test_picked_from_accepts_duplicate_valid_values():
    """Multiple identical valid value rows → passes (result matches one of them)."""
    result = create_input("result", 1)
    values = create_input("values", 1)
    keys = create_input("keys", 1)
    wrapped = assert_picked_from(result, values, keys, atol=1e-3)
    _eval(
        wrapped,
        {
            "result": torch.tensor([[3.0], [3.0]]),
            "values": torch.tensor([[3.0], [3.0]]),  # two rows, same value
            "keys": torch.tensor([[1.0], [1.0]]),  # both valid
        },
        n_pos=2,
    )


def test_picked_from_width_mismatch_raises():
    """Result/values width mismatch is caught at construction time."""
    result = create_input("result", 2)
    values = create_input("values", 3)
    keys = create_input("keys", 1)
    with pytest.raises(ValueError, match=r"width mismatch"):
        assert_picked_from(result, values, keys)


def test_picked_from_returns_result_width():
    """The wrapped result preserves ``result``'s width for downstream use."""
    result = create_input("result", 4)
    values = create_input("values", 4)
    keys = create_input("keys", 1)
    wrapped = assert_picked_from(result, values, keys)
    assert len(wrapped) == 4


# NOTE: an end-to-end ``test_picked_from_compiled_fires_on_blend`` used
# to live here.  It deliberately constructed sub-unit-gap fractional
# scores (0.0, 0.01, 0.02, 0.03) on attend_argmin_unmasked to force a
# compile-side softmax blend that assert_picked_from would catch.  With
# attend_*'s integer-score contract enforced at graph construction,
# that scenario is now rejected upstream and can't be built — the
# blending it tested is no longer reachable through normal API usage.
#
# assert_picked_from's predicate is still covered by the unit tests
# above (accepts_clean_pick, rejects_blend, rejects_no_valid_keys,
# accepts_duplicate_valid_values).


# ---------------------------------------------------------------------------
# assert_score_gap_at_least — minimum pairwise gap among valid rows
# ---------------------------------------------------------------------------


def test_score_gap_fires_at_reference_eval():
    """Two WALL scores within the softmax-resolvability margin should fire."""
    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)
    with pytest.raises(AssertionError, match=r"min-gap"):
        _eval(
            checked,
            {
                "sort_score": torch.tensor([[1.0], [1.04], [99.0]]),
                "is_wall": torch.tensor([[1.0], [1.0], [0.0]]),
            },
            n_pos=3,
        )


def test_score_gap_passes_with_comfortable_margin():
    """Scores spaced well above the gap margin must pass reference eval."""
    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)
    _eval(
        checked,
        {
            "sort_score": torch.tensor([[1.0], [1.5], [99.0]]),
            "is_wall": torch.tensor([[1.0], [1.0], [0.0]]),
        },
        n_pos=3,
    )


def test_score_gap_vacuous_zero_valid():
    """Zero valid rows -> vacuous pass."""
    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)
    _eval(
        checked,
        {
            "sort_score": torch.tensor([[1.0], [1.0], [1.0]]),
            "is_wall": torch.tensor([[0.0], [0.0], [0.0]]),
        },
        n_pos=3,
    )


def test_score_gap_vacuous_one_valid():
    """Single valid row -> vacuous pass (no pair to check)."""
    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)
    _eval(
        checked,
        {
            "sort_score": torch.tensor([[1.0], [1.0], [1.0]]),
            "is_wall": torch.tensor([[1.0], [0.0], [0.0]]),
        },
        n_pos=3,
    )


def test_score_gap_ignores_non_wall_ties():
    """Ties between a valid and an invalid row are ignored."""
    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)
    _eval(
        checked,
        {
            "sort_score": torch.tensor([[1.0], [1.0], [99.0]]),
            "is_wall": torch.tensor([[1.0], [0.0], [0.0]]),
        },
        n_pos=3,
    )


# --- All violations are hard errors now -----------------------------------


def test_range_violation_raises():
    inp = create_input("v", 1)
    node = assert_matches_value_type(inp, NodeValueType.bounded(0, 9))
    with pytest.raises(AssertionError, match="range"):
        node.compute(1, {"v": torch.tensor([[15.0]])})  # outside [0, 9]


def test_contradictory_claim_rejected_at_attach():
    """A claim disjoint from the node's structural type is a bug in one
    of the two; it fails loudly at attach time (the old wrapper design
    only caught it at the compile-time strip)."""
    from torchwright.graph import LiteralValue

    lit = LiteralValue(torch.tensor([15.0]))  # structural type [15, 15]
    with pytest.raises(ValueError, match="intersection"):
        assert_matches_value_type(lit, NodeValueType.bounded(0, 9))


def test_format_bad_multidim_reports_true_positions():
    """_format_bad on a multi-dim tensor: indices are into the flattened
    tensor and pair with the actual offending values. (Regression: a
    multi-dim nonzero returns coordinate rows; flattening those produced
    coordinates misread as flat indices, pairing wrong values with wrong
    positions.)"""
    from torchwright.graph.asserts import _format_bad

    x = torch.tensor([[1.0, 2.0], [-3.0, -4.0], [5.0, 6.0], [-7.0, -8.0]])
    mask = x < -4.5  # true at flat positions 6 (-7.0) and 7 (-8.0)
    msg = _format_bad(x, mask)
    assert "[6]=-7.0000" in msg
    assert "[7]=-8.0000" in msg
