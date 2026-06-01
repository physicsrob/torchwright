"""Tests for PosEncoding.get_position_scalar().

The position scalar is consumed by every graph that wants to know "which
token am I" (DOOM sharding, prefix ops, etc.).  It is computed as the
slowest-frequency sin component of the positional encoding, scaled back
to position space.  That approximation is only accurate for a bounded
range of positions — these tests pin down the accuracy envelope across
a range of sequence lengths the graph code has been asked to handle.

Both an uncompiled (graph.compute) and a compiled (compile_headless)
path are tested.  The uncompiled path checks the pure-math contract;
the compiled path mirrors how downstream ops actually consume the value
inside a real transformer forward pass.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import PosEncoding
from torchwright.ops.arithmetic_ops import add, multiply_const
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

# Representative compiled sizes: smallest DOOM uses, old default, exhaustive limit.
# The uncompiled math is covered by test_position_scalar_exhaustive below.
SEQ_LENS = [32, 640, 2048]


def test_position_scalar_exhaustive():
    """Every position 0-2047 must be accurate to within 0.5."""
    pos_encoding = PosEncoding(17)
    scalar_node = pos_encoding.get_position_scalar()
    out = scalar_node.compute(n_pos=2048, input_values={}).squeeze(-1)
    expected = torch.arange(2048, dtype=out.dtype)
    errors = (out - expected).abs()
    max_err = errors.max().item()
    worst_pos = errors.argmax().item()
    assert max_err < 1e-6, (
        f"position_scalar max error {max_err:.3e} at pos={worst_pos} "
        f"(got {out[worst_pos].item():.6f}, expected {worst_pos})"
    )


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_position_scalar_compiled(seq_len):
    """Compiled transformer — tests the value as downstream ops see it."""
    pos_encoding = create_pos_encoding()
    # compile_headless needs at least one input; fold a zero-weighted
    # copy of it into the output so the graph is well-formed without
    # distorting what we measure.
    dummy = create_input("dummy", 1)
    scalar = pos_encoding.get_position_scalar()
    output = add(scalar, multiply_const(dummy, 0.0))

    module = compile_headless(
        output,
        pos_encoding,
        d=1024,
        d_head=16,
        max_layers=20,
        verbose=False,
    )

    inputs = torch.zeros(seq_len, 1)
    with torch.no_grad():
        out = module(inputs).squeeze(-1)
    expected = torch.arange(seq_len, dtype=out.dtype)
    max_err = (out - expected).abs().max().item()
    assert max_err < 0.5, (
        f"compiled position_scalar max error {max_err:.3f} at " f"seq_len={seq_len}"
    )
