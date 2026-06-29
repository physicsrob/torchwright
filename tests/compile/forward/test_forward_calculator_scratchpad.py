"""Scratchpad calculator: export to ONNX and verify +, -, * via argmax decode.

The variant's claim is a flat *compiled* depth in ``max_digits`` while arbitrary
operand sizes are handled by more decode steps; this checks the compiled
artifact actually decodes the streamed scratchpad transcript correctly — carry
ripple, borrow ripple, a negative difference, a tall multiply column whose carry
exceeds 1, and — new with the leading-zero trim — that the post-``</THINKING>``
region is the *exact* MSB-first answer (no zero pad).  Inputs are fixed-width
zero-padded to ``max_digits`` (the protocol the ``O(1)`` parse requires).  The
flat-depth invariant itself is guarded without a compile in
``tests/examples/test_calculator_scratchpad.py``.
"""

import pytest

from examples.calculator_scratchpad import D_HEAD, D_MODEL, RESULT, create_network_parts

from ._example_onnx import load_example, run


@pytest.fixture(scope="module")
def calc3(tmp_path_factory):
    # D_MODEL (not the _example_onnx default 1024): the dispatched graph's peak
    # live width exceeds 1024 — the three streamed ops run in parallel and each
    # carries wide one-hot column totals (see calculator_scratchpad.D_MODEL).
    # d_head=D_HEAD (32): the family's rope width (the multiply pointer gather is
    # the widest content head); the compile d_head must match the rope.
    return load_example(
        lambda: create_network_parts(max_digits=3),
        tmp_path_factory.mktemp("scratch3"),
        name="scratch3",
        d=D_MODEL,
        d_head=D_HEAD,
    )


def _answer(model, prompt):
    """The answer region: everything after ``</THINKING>`` up to ``<eos>``.

    With the leading-zero trim this is the *exact* answer (MSB-first, sign kept,
    no zero pad), so we assert it verbatim — no normalization — which is what
    proves the trim works.
    """
    # Full transcript is at most 8n+3 tokens (multiply, incl. the scratch-digit
    # region); 32 clears it for n=3 with margin.  ref_token="<ref>": recency
    # example — <ref> must land at position 1 for the RoPE recency rank.
    text = run(model, prompt, max_new_tokens=32, ref_token="<ref>")
    assert RESULT in text, f"no {RESULT!r} in decoded output {text!r}"
    return text.split(RESULT, 1)[1]


def _check(model, prompt, expected):
    got = _answer(model, prompt)
    assert got == expected, f"for {prompt!r}: expected {expected!r}, got {got!r}"


def test_scratchpad_addition(calc3):
    model, _ = calc3
    _check(model, "001+001\n", "2")
    _check(model, "123+456\n", "579")
    _check(model, "999+001\n", "1000")  # carry ripples the whole width
    _check(model, "999+999\n", "1998")
    _check(model, "000+000\n", "0")


def test_scratchpad_subtraction(calc3):
    model, _ = calc3
    _check(model, "005-003\n", "2")
    _check(model, "456-123\n", "333")
    _check(model, "010-001\n", "9")  # borrow ripples up from the ones column
    _check(model, "100-100\n", "0")


def test_scratchpad_subtraction_negative(calc3):
    model, _ = calc3
    _check(model, "001-005\n", "-4")
    _check(model, "100-999\n", "-899")


def test_scratchpad_multiplication(calc3):
    model, _ = calc3
    _check(model, "002*003\n", "6")
    _check(model, "012*034\n", "408")
    _check(model, "013*039\n", "507")  # a column carry exceeds 1
    _check(model, "123*456\n", "56088")
    _check(model, "100*100\n", "10000")
    _check(model, "000*055\n", "0")
