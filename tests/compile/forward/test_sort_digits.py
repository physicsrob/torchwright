"""End-to-end ONNX test for the sort_digits example.

The variant exports its ``create_network_parts`` through
:func:`compile_to_onnx`, runs argmax autoregressive decode on a battery of
inputs, and asserts the decoded output matches the ascending sort.

V1 supports distinct-digit inputs only. (The prefix-sum variants V2/V4 were
retired in the RoPE port — they gated on an absolute ``pos >= 2**k`` threshold,
which is not a bounded near-marker difference and has no DOOM consumer;
see ``docs/rope_port_plan.md`` §8 Phase 1 / §11.)

The sort example emits its sorted output starting at the trigger position
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


# Cases supported by the distinct-digit sorter.
_DISTINCT_CASES = [
    ("9583", "3589"),
    ("1", "1"),
    ("5432", "2345"),
    ("1234", "1234"),
    ("9876543210", "0123456789"),
]


@pytest.fixture(scope="module")
def sort_v1(tmp_path_factory):
    from examples.sort_digits_v1 import D_MODEL, create_network_parts

    return load_example(
        create_network_parts,
        tmp_path_factory.mktemp("sortv1"),
        d=D_MODEL,
        d_head=D_HEAD,
        name="sortv1",
        # D_MODEL (384) is outside the norm's supported-width contract
        # (docs/rms_norm_dmodel.md), and the norm isn't what this exercises.
        rms_norm=False,
    )


# ---------------------------------------------------------------------------
# V1 — distinct digits only. Does not support duplicates by design.
# ---------------------------------------------------------------------------


def test_sort_digits_v1_distinct_battery(sort_v1):
    model, _ = sort_v1
    for inp, expected in _DISTINCT_CASES:
        _check_case(model, inp, expected)
