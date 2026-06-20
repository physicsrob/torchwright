"""End-to-end ONNX tests for the sort_digits example variants.

Each variant exports its ``create_network_parts`` through
:func:`compile_to_onnx`, runs argmax autoregressive decode on a battery of
inputs, and asserts the decoded output matches the ascending sort.

V4 is the primary variant and supports duplicates. V1 only supports
distinct-digit inputs. V2 is the rank-lookup variant (MLP selection brain,
attention is a lookup).  Converted from ``forward_compile``; see
``_example_onnx``.

The sort examples emit their sorted output starting at the trigger position
(the ``"\\n"``), with no guaranteed ``<eos>``, so each rollout is capped at
``len(input)`` decoded tokens.
"""

import pytest

from ._example_onnx import load_example, run

D_HEAD = 32


def _check_case(model, input_str: str, expected: str):
    result = run(
        model,
        input_str + "\n",
        bos_token="<bos>",
        max_new_tokens=len(input_str),
    )
    assert (
        result == expected
    ), f"input={input_str!r} expected={expected!r} got={result!r}"


# Cases shared by all variants that support only distinct digits.
_DISTINCT_CASES = [
    ("9583", "3589"),
    ("1", "1"),
    ("5432", "2345"),
    ("1234", "1234"),
    ("9876543210", "0123456789"),
]

# Additional cases with duplicates, for variants that handle them.
_DUPLICATE_CASES = [
    ("1111", "1111"),
    ("1121", "1112"),
    ("3131", "1133"),
    ("2211", "1122"),
    ("223331", "122333"),
]


# ---------------------------------------------------------------------------
# Fixtures — export once per variant, share across tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sort_v4(tmp_path_factory):
    from examples.sort_digits_v4 import D_MODEL, create_network_parts

    return load_example(
        create_network_parts,
        tmp_path_factory.mktemp("sortv4"),
        d=D_MODEL,
        d_head=D_HEAD,
        name="sortv4",
    )


@pytest.fixture(scope="module")
def sort_v1(tmp_path_factory):
    from examples.sort_digits_v1 import D_MODEL, create_network_parts

    return load_example(
        create_network_parts,
        tmp_path_factory.mktemp("sortv1"),
        d=D_MODEL,
        d_head=D_HEAD,
        name="sortv1",
    )


@pytest.fixture(scope="module")
def sort_v2(tmp_path_factory):
    from examples.sort_digits_v2 import D_MODEL, create_network_parts

    return load_example(
        create_network_parts,
        tmp_path_factory.mktemp("sortv2"),
        d=D_MODEL,
        d_head=D_HEAD,
        name="sortv2",
    )


# ---------------------------------------------------------------------------
# V4 — primary variant, handles duplicates.
# ---------------------------------------------------------------------------


def test_sort_digits_v4_distinct_battery(sort_v4):
    model, _ = sort_v4
    for inp, expected in _DISTINCT_CASES:
        _check_case(model, inp, expected)


def test_sort_digits_v4_duplicate_battery(sort_v4):
    model, _ = sort_v4
    for inp, expected in _DUPLICATE_CASES:
        _check_case(model, inp, expected)


# ---------------------------------------------------------------------------
# V1 — distinct digits only. Does not support duplicates by design.
# ---------------------------------------------------------------------------


def test_sort_digits_v1_distinct_battery(sort_v1):
    model, _ = sort_v1
    for inp, expected in _DISTINCT_CASES:
        _check_case(model, inp, expected)


# ---------------------------------------------------------------------------
# V2 — rank-lookup (MLP selection brain, attention is a lookup).
# ---------------------------------------------------------------------------


_V2_DISTINCT_CASES = [
    ("9583", "3589"),
    ("1", "1"),
    ("5432", "2345"),
    ("1234", "1234"),
]

_V2_DUPLICATE_CASES = [
    ("1111", "1111"),
    ("1121", "1112"),
    ("3131", "1133"),
    ("2211", "1122"),
]


def test_sort_digits_v2_distinct_battery(sort_v2):
    model, _ = sort_v2
    for inp, expected in _V2_DISTINCT_CASES:
        _check_case(model, inp, expected)


def test_sort_digits_v2_duplicate_battery(sort_v2):
    model, _ = sort_v2
    for inp, expected in _V2_DUPLICATE_CASES:
        _check_case(model, inp, expected)
