"""swiglu floor_int / ceil_int / scalar_to_embedding.

Spec: docs/ops_plain_english.md (floor_int, scalar_to_embedding entries).
The load-bearing structure (two-stage depth, W-slack absorbing fillets)
is the same as relu's; these tests pin the contract behavior plus the
swish-specific claims.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph import FFN
from torchwright.ops.const import scale, swish_dip
from torchwright.ops.inout_nodes import create_embedding, create_input
from torchwright.ops.swiglu import ceil_int, floor_int, scalar_to_embedding

D = 64
D_HEAD = 8


def test_floor_int_flat_zone_and_integers_exact():
    x = create_input("x", 1, value_range=(-5.0, 10.0))
    out = floor_int(x, min_value=-5, max_value=10)
    # Integers and flat-zone interiors (contract inputs).
    xs = torch.tensor([[-5.0], [-4.3], [0.0], [0.5], [3.0], [7.7], [10.0]])
    val = out.compute(7, {"x": xs})
    ref = torch.floor(xs)
    assert torch.allclose(val, ref, rtol=0.0, atol=1e-4), (val - ref).flatten()


def test_floor_int_range_claim_carries_fillet_slack():
    x = create_input("x", 1, value_range=(-5.0, 10.0))
    out = floor_int(x, min_value=-5, max_value=10)
    slack = 2.0 * swish_dip / scale
    r = out.value_type.value_range
    assert r.lo == pytest.approx(-5.0 - slack)
    assert r.hi == pytest.approx(10.0 + slack)


def test_floor_int_chunking_matches_unchunked():
    """A range wide enough to split into multiple 512-boundary chunks
    computes the same floor as exact math (the W-slack keeps saturated
    chunks exact)."""
    x = create_input("x", 1, value_range=(0.0, 1300.0))
    out = floor_int(x, min_value=0, max_value=1300)  # 1300 boundaries > 512
    xs = torch.tensor([[0.5], [511.3], [512.5], [1024.2], [1299.5], [1300.0]])
    val = out.compute(6, {"x": xs})
    ref = torch.floor(xs)
    assert torch.allclose(val, ref, rtol=0.0, atol=1e-3), (val - ref).flatten()


def test_ceil_int():
    x = create_input("x", 1, value_range=(-5.0, 10.0))
    out = ceil_int(x, min_value=-5, max_value=10)
    xs = torch.tensor([[-4.5], [0.0], [2.3], [9.5]])
    val = out.compute(4, {"x": xs})
    assert torch.allclose(val, torch.ceil(xs), rtol=0.0, atol=1e-4)


def test_floor_int_compiles_clean():
    x = create_input("x", 1, value_range=(-5.0, 10.0))
    out = floor_int(x, min_value=-5, max_value=10)
    compiled = compile_headless(out, d=D, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    xs = torch.tensor([[-4.3], [0.5], [7.7]])
    report = probe_compiled(compiled, out, {"x": xs}, 3, atol=1e-3)
    assert report.first_divergent is None, report.format_short()


# ---------------------------------------------------------------------------
# scalar_to_embedding
# ---------------------------------------------------------------------------

_VOCAB = [str(d) for d in range(10)] + ["+", "="]


def test_scalar_to_embedding_reconstructs_digit_embeddings():
    emb = create_embedding(vocab=_VOCAB)
    x = create_input("x", 1, value_range=(0.0, 9.0))
    out = scalar_to_embedding(x, emb)
    ffn = out
    assert isinstance(ffn, FFN)
    assert ffn.is_degenerate
    assert ffn.n_lanes == 18
    xs = torch.arange(10.0).unsqueeze(1)
    val = out.compute(10, {"x": xs})
    for d in range(10):
        ref = emb.get_embedding(str(d))
        # Exact 0/1 indicators; the only error is out_proj rounding
        # (~ulps of the embedding components at norm ~40).
        assert torch.allclose(val[d], ref, rtol=0.0, atol=1e-4), d


def test_scalar_to_embedding_noise_headroom():
    """A digit scalar off by ±0.4 reconstructs the same embedding (the
    nearest threshold is 0.5 away; saturation holds to 17/(scale·S) of
    a ramp edge)."""
    emb = create_embedding(vocab=_VOCAB)
    x = create_input("x", 1, value_range=(0.0, 9.0))
    out = scalar_to_embedding(x, emb)
    xs = torch.tensor([[3.0 - 0.4], [3.0], [3.0 + 0.4]])
    val = out.compute(3, {"x": xs})
    ref = emb.get_embedding("3")
    for i in range(3):
        assert torch.allclose(val[i], ref, rtol=0.0, atol=1e-3), i


def test_scalar_to_embedding_compiles_clean():
    emb = create_embedding(vocab=_VOCAB)
    x = create_input("x", 1, value_range=(0.0, 9.0))
    out = scalar_to_embedding(x, emb)
    compiled = compile_headless(out, d=128, d_head=D_HEAD)
    assert compiled._net.activation == "swish"
    xs = torch.tensor([[0.0], [4.0], [9.0]])
    report = probe_compiled(compiled, out, {"x": xs}, 3, atol=1e-2)
    assert report.first_divergent is None, report.format_short()
