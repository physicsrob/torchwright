"""Bucket-1 near-marker count — confidence test (RoPE port Phase 1, §9).

Proves the torchwright capability "recover the bounded gap ``pos - marker_pos``
from in-context marker features, no position counter" at its real worst-case
difficulty: gap pushed to ~350 (the DOOM screen-dimension bound), where the
uniform-count mean is ``1/351`` and adjacent gaps differ by only ~8e-6.

Oracle parity (memoised ``reference_eval``) confirms the construction; the
compiled check confirms the inversion resolves to the right integer through the
real transformer.
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import reference_eval
from torchwright.graph.asserts import assert_in_range
from torchwright.ops.inout_nodes import create_input, create_rope_config
from torchwright.ops.relu.marker_count import count_since_marker

MAX_GAP = 350

# count_since_marker rides the slowest rotary plane.  Over a gap-350 window the
# residual window non-uniformity goes as ~333·theta_slow^2·gap^3, and theta_slow
# = base^(-(d_head-2)/d_head) shrinks toward base^-1 as d_head grows.  d_head=16
# leaves the slowest plane too fast (worst gap error ~1.5); d_head=64 pushes it
# to ~0.19, comfortably under the +/-0.5 round-to-integer budget with margin for
# the compiled fp32 path.
D_HEAD = 64


def _build(max_gap=MAX_GAP):
    rope = create_rope_config(d_head=D_HEAD, max_positions=512)
    marker_onehot = create_input("marker_onehot", 1)
    window_validity = create_input("window_validity", 1)
    marker = assert_in_range(marker_onehot, 0.0, 1.0)
    gap = count_since_marker(rope, window_validity, marker, max_gap=max_gap)
    return rope, marker_onehot, window_validity, gap


def _inputs(n_pos, marker_pos):
    """marker_onehot = 1 at marker_pos; window_validity = +1 at/after marker."""
    marker = torch.zeros(n_pos, 1)
    marker[marker_pos, 0] = 1.0
    valid = torch.full((n_pos, 1), -1.0)
    valid[marker_pos:, 0] = 1.0
    return marker, valid


def test_oracle_recovers_gap_to_bound():
    """Oracle: gap[n] == n - marker_pos out to the ~350 bound."""
    _, _marker_node, _valid_node, gap = _build()
    n_pos = MAX_GAP + 1  # positions 0..350, marker at 0 -> gaps 0..350
    marker, valid = _inputs(n_pos, marker_pos=0)
    cache = reference_eval(
        gap, {"marker_onehot": marker, "window_validity": valid}, n_pos
    )
    got = cache[gap].reshape(-1)
    expected = torch.arange(n_pos, dtype=torch.float32)
    err = (got - expected).abs()
    # accurate to well under +/-0.5 so it rounds to the right integer
    assert err.max().item() < 0.5, (
        f"worst gap error {err.max().item():.3f} at gap "
        f"{int(err.argmax())} (expected {int(expected[err.argmax()])}, "
        f"got {got[err.argmax()].item():.3f})"
    )


def test_oracle_marker_not_at_zero():
    """Oracle: gap measured from a mid-stream marker."""
    _, _marker_node, _valid_node, gap = _build()
    n_pos = 200
    marker_pos = 37
    marker, valid = _inputs(n_pos, marker_pos)
    cache = reference_eval(
        gap, {"marker_onehot": marker, "window_validity": valid}, n_pos
    )
    got = cache[gap].reshape(-1)
    # only positions at/after the marker are meaningful
    for n in range(marker_pos, n_pos):
        assert abs(got[n].item() - (n - marker_pos)) < 0.5, (
            f"pos {n}: expected gap {n - marker_pos}, got {got[n].item():.3f}"
        )


def test_unsafe_dhead_raises():
    """count_since_marker raises ValueError for d_head too small to give <0.5
    gap error at the given max_gap (the quasi-static approximation breaks down).
    d_head=16 with max_gap=350 gives analytic error ~1.5 >> 0.45 threshold.
    """
    import pytest

    rope = create_rope_config(d_head=16, max_positions=512)
    with pytest.raises(ValueError, match="estimated gap error"):
        count_since_marker(
            rope,
            create_input("w", 1),
            create_input("m", 1),
            max_gap=350,
        )


def test_compiled_recovers_gap_to_bound():
    """Compiled transformer: gap[n] rounds to n out to the ~350 bound."""
    _rope, _marker_in, _valid_in, gap = _build()
    module = compile_headless(gap, d=512, d_head=D_HEAD, verbose=False)

    n_pos = MAX_GAP + 1
    marker, valid = _inputs(n_pos, marker_pos=0)
    # alphabetical input order: marker_onehot, window_validity
    flat = torch.cat([marker, valid], dim=1)
    out = module(flat).reshape(-1)
    expected = torch.arange(n_pos, dtype=torch.float32)
    err = (out - expected).abs()
    assert err.max().item() < 0.5, (
        f"compiled worst gap error {err.max().item():.3f} at gap "
        f"{int(err.argmax())} (got {out[err.argmax()].item():.3f})"
    )
