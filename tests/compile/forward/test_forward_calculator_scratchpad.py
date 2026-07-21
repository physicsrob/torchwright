"""Scratchpad calculator: export to ONNX and verify +, -, * via argmax decode.

The variant's claim is a flat *compiled* depth in ``max_digits`` while arbitrary
operand sizes are handled by more decode steps; this checks the compiled
artifact actually decodes the streamed scratchpad transcript correctly — carry
ripple, borrow ripple, a negative difference, a tall multiply column whose carry
exceeds 1, and — with the leading-zero trim — that the post-``</THINKING>``
region is the *exact* MSB-first answer (no zero pad).  Inputs are the same
variable-width prompts the other two calculators accept (the shared
constant-depth pointer-gather parse zero-pads internally); one zero-padded
prompt per op stays pinned, since padded input remains valid as the
full-width special case.  The flat-depth invariant itself is guarded without
a compile in ``tests/examples/test_calculator_scratchpad.py``.
"""

import pytest

from examples.calculator_scratchpad import D_HEAD, RESULT, create_network_parts

from ._example_onnx import load_example, run

# The smallest width that schedules the n=3 dispatched graph with margin (it
# needs ~2560 live columns; see the D_MODEL geometry note in the module).  The
# family's canonical publish width (D_MODEL=8192) would make this ONNX fixture
# a multi-GB artifact for nothing; the canonical-geometry compile is witnessed
# by the layer table instead.
_EXPORT_D = 3072


@pytest.fixture(scope="module")
def calc3(tmp_path_factory):
    # d_head=D_HEAD: the family's canonical head width, baked into the graph
    # (this module's multiply pointer gather is what sets it — see the
    # geometry note above).  d=3072 is inside the norm's supported-width
    # contract (3·2^10; docs/rms_norm_dmodel.md), so the norm stays on,
    # matching the publish path.
    return load_example(
        lambda: create_network_parts(max_digits=3),
        tmp_path_factory.mktemp("scratch3"),
        name="scratch3",
        d=_EXPORT_D,
        d_head=D_HEAD,
    )


def _answer(model, prompt):
    """The answer region: everything after ``</THINKING>`` up to ``<eos>``.

    With the leading-zero trim this is the *exact* answer (MSB-first, sign kept,
    no zero pad), so we assert it verbatim — no normalization — which is what
    proves the trim works.
    """
    # Full transcript is at most 8n+3 tokens (multiply, incl. the scratch-digit
    # region); 32 clears it for n=3 with margin.
    text = run(model, prompt, max_new_tokens=32)
    assert RESULT in text, f"no {RESULT!r} in decoded output {text!r}"
    return text.split(RESULT, 1)[1]


def _check(model, prompt, expected):
    got = _answer(model, prompt)
    assert got == expected, f"for {prompt!r}: expected {expected!r}, got {got!r}"


def test_scratchpad_addition(calc3):
    model, _ = calc3
    _check(model, "1+1\n", "2")
    _check(model, "123+456\n", "579")
    _check(model, "999+1\n", "1000")  # carry ripples the whole width
    _check(model, "999+999\n", "1998")
    _check(model, "0+0\n", "0")
    _check(model, "001+001\n", "2")  # zero-padded input stays valid


def test_scratchpad_subtraction(calc3):
    model, _ = calc3
    _check(model, "5-3\n", "2")
    _check(model, "456-123\n", "333")
    _check(model, "10-1\n", "9")  # borrow ripples up from the ones column
    _check(model, "100-100\n", "0")
    _check(model, "005-003\n", "2")  # zero-padded input stays valid


def test_scratchpad_subtraction_negative(calc3):
    model, _ = calc3
    _check(model, "1-5\n", "-4")
    _check(model, "100-999\n", "-899")


def test_scratchpad_multiplication(calc3):
    model, _ = calc3
    _check(model, "2*3\n", "6")
    _check(model, "12*34\n", "408")
    _check(model, "13*39\n", "507")  # a column carry exceeds 1
    _check(model, "123*456\n", "56088")
    _check(model, "100*100\n", "10000")
    _check(model, "0*55\n", "0")
    _check(model, "012*034\n", "408")  # zero-padded input stays valid
