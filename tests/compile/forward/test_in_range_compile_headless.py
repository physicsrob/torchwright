"""Test that in_range compiled via compile_headless matches forward_compile.

compile_headless sorts inputs alphabetically, so the flat input tensor
must be ordered [hi, lo] not [lo, hi].  Both paths should produce ±1.
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.forward.compile import forward_compile
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.relu.map_select import in_range


def test_in_range_headless_values_are_pm1():
    """compile_headless in_range output must be ±1."""
    lo = create_input("lo", 1)
    hi = create_input("hi", 1)

    N = 32
    masks = in_range(lo, hi, N)

    module = compile_headless(masks, d=256, verbose=False)

    # Alphabetical order: hi=8, lo=4 → in_range(4, 8, 32)
    # Slots 4,5,6,7 should be +1, rest -1
    result = module(torch.tensor([[8.0, 4.0]])).squeeze(0)

    for i in range(N):
        expected = 1.0 if 4 <= i < 8 else -1.0
        actual = result[i].item()
        assert (
            abs(actual - expected) < 0.5
        ), f"slot {i}: expected {expected}, got {actual:.1f}"


def test_in_range_headless_matches_forward_compile():
    """compile_headless and forward_compile must agree on in_range output."""
    lo = create_input("lo", 1)
    hi = create_input("hi", 1)

    N = 32
    masks = in_range(lo, hi, N)

    # forward_compile path
    net = forward_compile(d=256, d_head=16, output_node=masks, verbose=False)
    vals = {"lo": torch.tensor([[4.0]]), "hi": torch.tensor([[8.0]])}
    fc_out = net.compute(1, vals)[masks].squeeze(0)

    # compile_headless path (alphabetical: hi first)
    module = compile_headless(masks, d=256, verbose=False)
    ch_out = module(torch.tensor([[8.0, 4.0]])).squeeze(0)

    for i in range(N):
        fc_val = fc_out[i].item()
        ch_val = ch_out[i].item()
        assert abs(fc_val - ch_val) < 0.5, (
            f"slot {i}: forward_compile={fc_val:.1f}, " f"compile_headless={ch_val:.1f}"
        )
