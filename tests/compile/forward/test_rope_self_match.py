"""Phase-2 part 2: the compiler-internal self-match goes rotary.

``docs/rope_port_plan.md`` Phase 2 Part 2.  Every ``Linear``/``Add``/``Cancel``/
``add_into`` the compiler emits transports its operands through a
*current-position* (Δ=0) attention head — built by
``weight_writer._current_pos_attn_matrices``.  Today that head matches on the
``PosEncoding`` trig block; the migration replaces it with a **rotary** head
that reads a reserved constant-1 residual column and lets the runtime rotation
(rotate_half, by absolute position) make the softmax peak on the diagonal
``i == j``.

The rotary self-match is the Phase-0/Part-1 offset head at Δ=0: query reads the
constant-1 column → ``hardness · ones(d_head)``, key → ``ones(d_head)``, both
rotated by absolute position, so ``logit(j, i) ∝ Σ_p cos((i − j)·θ_p)`` is
uniquely maximal at ``i == j``.  The diagonal softmax weight is 1.0 to fp32 out
past any real sequence length, so transport is bit-identical to the trig
self-match.

These tests pin three things:
  * **Equivalence** — the rotary self-match produces the same compiled output
    as the trig self-match (and as the exact-math oracle via ``probe_compiled``).
  * **The path is actually taken** — with the flag on, self-match heads are
    marked rotary (``rotary_width != 0``); with it off, no head is rotary.
    (These graphs build no rotary ``Attn`` node, so every rotary head is a
    self-match head.)
  * **Cache parity** — prefill and decode agree, so the rotation is applied at
    absolute position on both paths.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.ops.arithmetic_ops import add, signed_multiply
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

D = 256
D_HEAD = 16


def _rotary_head_count(compiled) -> int:
    """Number of attention heads marked rotary across all compiled layers."""
    return sum(
        1
        for layer in compiled._net.layers
        for w in layer.attn.attn.rotary_width
        if w != 0
    )


def _add_graph():
    """``a + b`` — exercises ``compute_add`` (self-match transport of both
    addends into fresh columns)."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    return add(a, b), create_pos_encoding()


def _multiply_graph():
    """``signed_multiply`` — a richer graph whose lowering emits
    ``compute_linear`` / ``cancel`` / ``add_into`` self-match transport."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    return signed_multiply(a, b, max_abs1=10, max_abs2=10), create_pos_encoding()


_ADD_INPUTS = torch.tensor(
    [[3.0, 4.0], [5.0, -1.0], [-2.0, 3.0], [0.0, 7.0], [4.0, 2.0]]
)
_MUL_INPUTS = _ADD_INPUTS  # same (a, b) columns


def test_rotary_self_match_matches_trig_add():
    """compute_add: rotary self-match == trig self-match == oracle."""
    out, pos = _add_graph()
    trig = compile_headless(out, pos, d=D, d_head=D_HEAD, verbose=False)

    out2, pos2 = _add_graph()
    rotary = compile_headless(
        out2, pos2, d=D, d_head=D_HEAD, verbose=False, rotary_self_match=True
    )

    with torch.no_grad():
        trig_out = trig(_ADD_INPUTS)
        rotary_out = rotary(_ADD_INPUTS)

    assert torch.allclose(trig_out, rotary_out, atol=1e-4), (
        f"rotary vs trig self-match differ: "
        f"{(trig_out - rotary_out).abs().max().item():.6f}"
    )
    # The flag must actually have routed the self-match through rotary heads,
    # and the trig compile must have none (no rotary Attn node in this graph).
    assert _rotary_head_count(rotary) > 0, "rotary self-match did not mark any head"
    assert _rotary_head_count(trig) == 0, "trig self-match unexpectedly rotary"

    report = probe_compiled(
        rotary, out2, {"a": _ADD_INPUTS[:, :1], "b": _ADD_INPUTS[:, 1:]}, 5, atol=1e-4
    )
    assert report.first_divergent is None, report.format_short()


def test_rotary_self_match_matches_trig_multiply():
    """signed_multiply: rotary self-match == trig self-match == oracle."""
    out, pos = _multiply_graph()
    trig = compile_headless(out, pos, d=D, d_head=D_HEAD, verbose=False)

    out2, pos2 = _multiply_graph()
    rotary = compile_headless(
        out2, pos2, d=D, d_head=D_HEAD, verbose=False, rotary_self_match=True
    )

    with torch.no_grad():
        trig_out = trig(_MUL_INPUTS)
        rotary_out = rotary(_MUL_INPUTS)

    assert torch.allclose(trig_out, rotary_out, atol=1e-3), (
        f"rotary vs trig self-match differ: "
        f"{(trig_out - rotary_out).abs().max().item():.6f}"
    )
    assert _rotary_head_count(rotary) > 0, "rotary self-match did not mark any head"

    report = probe_compiled(
        rotary, out2, {"a": _MUL_INPUTS[:, :1], "b": _MUL_INPUTS[:, 1:]}, 5, atol=1e-3
    )
    assert report.first_divergent is None, report.format_short()


def test_rotary_self_match_prefill_decode():
    """Cached parity: prefill-then-decode == a single full-sequence forward,
    so the self-match rotation is applied at absolute position on both paths."""
    out, pos = _multiply_graph()
    rotary = compile_headless(
        out, pos, d=D, d_head=D_HEAD, verbose=False, rotary_self_match=True
    )

    with torch.no_grad():
        full = rotary(_MUL_INPUTS)
        past = rotary.empty_past()
        prefill_out, past = rotary.step(_MUL_INPUTS[:4], past)
        decode_out, past = rotary.step(_MUL_INPUTS[4:5], past)

    assert torch.allclose(
        full[:4], prefill_out, atol=1e-3
    ), f"prefill diff: {(full[:4] - prefill_out).abs().max().item():.6f}"
    assert torch.allclose(
        full[4], decode_out[0], atol=1e-3
    ), f"decode diff: {(full[4] - decode_out[0]).abs().max().item():.6f}"


def test_rotary_self_match_rejects_odd_d_head():
    """rotate_half pairs dim p with p+d_head/2, so an odd d_head is rejected
    up front with an actionable message (not deep in the weight writer)."""
    out, pos = _add_graph()
    with pytest.raises(ValueError, match="even d_head"):
        compile_headless(out, pos, d=255, d_head=17, rotary_self_match=True)
