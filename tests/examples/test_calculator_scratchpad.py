"""Scratchpad calculator: the flat-depth invariant + the streamed column steps.

These need neither a GPU nor a compile, so they live as fast
reference-evaluation tests:

* **flat depth** — ``critical_path_depth`` of the whole model
  (``create_network_parts``) is constant across ``max_digits``.  This is the
  entire point of the variant, so it gets a guarding test (per the plan).
* **the streamed carry/borrow column step** — the smallest layer that reproduces
  a column emission: a literal column total resolves to the right carry/borrow
  glyph and the right answer digit (the two halves of the same total).
* **the streamed normalization step** — the smallest layer that reproduces one
  leading-zero count update: the count sticky-increments while still in the
  leading-zero run and freezes the moment a non-zero digit (or an already-ended
  run) is seen.  This is the unit the normalization sweep is built from (D6:
  smallest reproducing layer).

The end-to-end decode of the *compiled* artifact (carry ripple, the trimmed
MSB-first answer, negative differences, a tall multiply column) lives in
``tests/compile/forward/test_forward_calculator_scratchpad.py``.
"""

import torch

from examples import calculator_scratchpad as cs
from scripts.arithmetic_scaling import critical_path_depth


def test_depth_is_flat_in_max_digits():
    depths = {
        n: critical_path_depth([cs.create_network_parts(max_digits=n)[0]])
        for n in range(2, 7)
    }
    assert len(set(depths.values())) == 1, (
        f"model depth must be constant in max_digits (the whole point of the "
        f"scratchpad variant); got {depths}"
    )


def _emit(n, total_value, kind):
    """Reference-eval one streamed carry/borrow column step: a literal column
    total → the emitted token, through the same ``_emit_from_total`` the sweeps
    use."""
    embedding = cs.create_onehot_embedding(cs.scratch_vocab(n))
    embed = embedding.get_embedding
    W = 20 * n + 1
    if kind == "carry":
        table = {cs._state(t, W): embed(cs._thinking_text(t // 10)) for t in range(W)}
        default = embed(cs._thinking_text(0))
    else:
        table = {cs._state(t, W): embed(str(t % 10)) for t in range(W)}
        default = embed("0")
    node = cs._emit_from_total(
        embedding, cs._scalar(float(total_value)), table, default, W
    )
    value = node.compute(n_pos=1, input_values={})[0]
    return embedding.tokenizer.vocab[int(value.argmax())]


def test_streamed_column_step_add_range():
    # Every reachable addition column total a+b+carry resolves to the right
    # carry-out glyph and the right answer digit.
    n = 3
    for a in range(10):
        for b in range(10):
            for carry in range(2):
                total = a + b + carry
                assert _emit(n, total, "carry") == cs._thinking_text(total // 10)
                assert _emit(n, total, "digit") == str(total % 10)


def test_streamed_column_step_wide_carry():
    # A multiply column can carry more than 1 (total up to 20n); the wide carry
    # glyph must still resolve from the one-hot index.
    n = 3
    for total in [0, 9, 10, 19, 25, 37, 20 * n]:
        assert _emit(n, total, "carry") == cs._thinking_text(total // 10)
        assert _emit(n, total, "digit") == str(total % 10)


def _norm_step(n, m, prev_count, digit_value):
    """Reference-eval one normalization column step → the emitted count glyph.

    Mirrors the body of ``_stream_normalize`` for column ``m`` with the previous
    running count ``prev_count`` and this column's answer digit ``digit_value``:
    the count increments iff the run is still alive (``prev_count == m``, since
    the count can never exceed ``m``) *and* the digit is zero, then is emitted as
    a subscript count glyph.  Returns the decoded glyph string.
    """
    embedding = cs.create_onehot_embedding(cs.scratch_vocab(n))
    embed = embedding.get_embedding
    n_state = cs._n_thinking(n)
    count_table = {
        cs._state(k, n_state): embed(cs._count_text(k)) for k in range(n_state)
    }
    count_default = embed(cs._count_text(0))

    prev = cs._scalar(float(prev_count))
    digit_emb = cs.create_literal_value(embed(str(digit_value)))
    is_zero = cs.equals_vector(inp=digit_emb, vector=embed("0"))
    in_run = cs.compare(prev, thresh=m - 0.5)
    still_zero = cs.bool_all_true([in_run, is_zero])
    this_count = cs.add(prev, cs.bool_to_01(still_zero))
    node = cs._emit_from_total(
        embedding, this_count, count_table, count_default, n_state
    )
    value = node.compute(n_pos=1, input_values={})[0]
    return embedding.tokenizer.vocab[int(value.argmax())]


def test_normalization_step_increments_in_run():
    # Still scanning leading zeros (prev_count == m) and this digit is zero:
    # the count ticks up by one.
    n = 3
    for m in range(2 * n):
        assert _norm_step(n, m, prev_count=m, digit_value=0) == cs._count_text(m + 1)


def test_normalization_step_freezes_after_run():
    n = 3
    # The run ends the moment a non-zero digit appears (prev_count == m but the
    # digit is non-zero): the count freezes at m.
    for m in range(2 * n):
        for d in (1, 5, 9):
            assert _norm_step(n, m, prev_count=m, digit_value=d) == cs._count_text(m)
    # Once the run has already ended (prev_count < m), the count stays put even
    # for a later zero digit — a zero after the first significant digit is an
    # interior zero, not a leading one.
    for m in range(1, 2 * n):
        for prev in range(m):  # prev_count < m: run already ended
            assert _norm_step(n, m, prev_count=prev, digit_value=0) == cs._count_text(
                prev
            )
            assert _norm_step(n, m, prev_count=prev, digit_value=7) == cs._count_text(
                prev
            )
