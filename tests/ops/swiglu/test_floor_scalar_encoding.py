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


def test_floor_int_intermediates_carry_tight_range_pins():
    """The two residual-resident intermediates are Assert-pinned to hinge bounds.

    The intermediates are the per-boundary step stage and each chunk's
    saturate sum: step in [-dip, W+dip]; chunk sum in [-c(1+dip), c·dip],
    which hold for ANY input since hinge(z) <= relu(z) and >= relu(z)-dip
    pointwise.  Unpinned, the affine relaxation declares ~sharpness·range
    on the step stage (~1e16 on a production-magnitude floor), and the
    flagship's rms_norm residual-energy certifier — which reads every
    residual-resident value's claim-tightened type, not just the op's
    asserted output — blows its fp32-feasible budget.  Found by the doom
    D2 cutover compile gate.
    """
    from collections import deque

    x = create_input("x", 1, value_range=(-1023.0, 1023.0))
    out = floor_int(x, -1023, 1023, sharpness=10_000.0)

    dip = swish_dip / scale
    seen, q = set(), deque([out])
    pins = {"floor_int_step": [], "floor_int_saturate": []}
    while q:
        n = q.popleft()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if n.claimed_type is not None:
            name = getattr(n, "name", "")
            if name in pins:
                pins[name].append(n.claimed_type.value_range)
        q.extend(getattr(n, "inputs", []))

    # 2046 boundaries at _CHUNK = min_d_hidden//2 = 512 -> 4 chunks.
    assert len(pins["floor_int_step"]) == 4, pins
    assert len(pins["floor_int_saturate"]) == 4, pins
    for r in pins["floor_int_step"]:
        # W = max(2, 8·ulp(s·n)) is 16 here (plus the W/4 fp slack); the
        # failure mode this guards against is the ~1e16 class, so assert
        # the magnitude class only.
        assert r.lo >= -dip - 16.0, r
        assert r.hi <= 100.0, r
    for r in pins["floor_int_saturate"]:
        assert r.lo >= -512.0 * (1.0 + dip) - 1.0 - 1e-6, r
        assert r.hi <= 3.0, r


def test_floor_int_chunking_matches_unchunked():
    """A wide range that splits into multiple chunks still floors exactly.

    A range wide enough to split into multiple 512-boundary chunks computes
    the same floor as exact math (the W-slack keeps saturated chunks exact).
    """
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
# floor_int output_map (fold a following piecewise-constant op into floor)
# ---------------------------------------------------------------------------


def _saturate_ffns(node):
    """Collect every stage-2 ``floor_int_saturate`` FFN feeding ``node``."""
    from collections import deque

    seen, q, acc = set(), deque([node]), []
    while q:
        n = q.popleft()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, FFN) and n.name == "floor_int_saturate":
            acc.append(n)
        q.extend(getattr(n, "inputs", []))
    return acc


def test_floor_int_output_map_default_byte_identical():
    """Omitting ``output_map`` leaves behavior unchanged.

    No new argument means no behavior change: the stage-2 output weights
    are exactly the pre-output_map ``-ones/scale``, and the op still floors.
    """
    x = create_input("x", 1, value_range=(-5.0, 10.0))
    out = floor_int(x, min_value=-5, max_value=10)  # output_map absent
    (sat,) = _saturate_ffns(out)
    assert torch.equal(sat.out_proj, -torch.ones_like(sat.out_proj) / scale)
    xs = torch.tensor([[-5.0], [-4.3], [0.0], [3.0], [7.7], [10.0]])
    val = out.compute(6, {"x": xs})
    assert torch.allclose(val, torch.floor(xs), rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("H", [3, 256])
def test_floor_int_output_map_sawtooth_matches_mod(H):
    """``output_map = k % H`` reproduces ``floor(x) % H`` exactly on flat-zone inputs.

    This includes inputs just past multiples of H (the ``-(H-1)``
    boundaries, where the largest |delta| lives).
    """
    lo, hi = 0, 3 * H + 2
    x = create_input("x", 1, value_range=(float(lo), float(hi)))
    out = floor_int(x, min_value=lo, max_value=hi, output_map=lambda k: float(k % H))
    pts = [b + 0.3 for b in range(lo, hi)]  # flat-zone interiors
    pts += [float(m) + 0.05 for m in range(H, hi, H)]  # just past each -(H-1) drop
    xs = torch.tensor([[p] for p in pts])
    val = out.compute(len(pts), {"x": xs})
    ref = torch.tensor([[float(int(p) % H)] for p in pts])
    assert torch.allclose(val, ref, rtol=0.0, atol=1e-3), (val - ref).abs().max()


def test_floor_int_output_map_sawtooth_compiles_clean():
    H, lo, hi = 8, 0, 40
    x = create_input("x", 1, value_range=(float(lo), float(hi)))
    out = floor_int(x, min_value=lo, max_value=hi, output_map=lambda k: float(k % H))
    compiled = compile_headless(out, d=256, d_head=D_HEAD)
    xs = torch.tensor([[0.5], [7.3], [8.05], [15.4], [39.5]])
    report = probe_compiled(compiled, out, {"x": xs}, xs.shape[0], atol=1e-2)
    assert report.first_divergent is None, report.format_short()
    ref = torch.tensor([[float(int(p.item()) % H)] for p in xs])
    assert torch.allclose(compiled(xs), ref, rtol=0.0, atol=1e-2)


def test_floor_int_output_map_lookup_step_function():
    """A non-sawtooth piecewise-constant map pins the generality of output_map.

    An arbitrary per-integer lookup shows any g(floor(x)) folds in, not
    just modular ones.
    """
    lut = {-4: 2.0, -3: 2.0, -2: -7.5, -1: 0.0, 0: 9.0, 1: 9.0, 2: -3.25, 3: 1.0}
    lo, hi = -4, 3
    x = create_input("x", 1, value_range=(float(lo), float(hi)))
    out = floor_int(x, min_value=lo, max_value=hi, output_map=lambda k: lut[k])
    pts = [k + 0.4 for k in range(lo, hi)]
    xs = torch.tensor([[p] for p in pts])
    val = out.compute(len(pts), {"x": xs})
    ref = torch.tensor([[lut[int(p // 1)]] for p in pts])
    assert torch.allclose(val, ref, rtol=0.0, atol=1e-3), (val - ref).abs().max()


def test_floor_int_output_map_integers_and_flat_zones_exact():
    """Contract inputs stay exact under output_map, same as the plain floor path.

    Contract inputs are exact integers and flat-zone interiors.
    """
    H, lo, hi = 5, 0, 14
    x = create_input("x", 1, value_range=(float(lo), float(hi)))
    out = floor_int(x, min_value=lo, max_value=hi, output_map=lambda k: float(k % H))
    xs = torch.tensor(
        [[0.0], [4.0], [5.0], [9.0], [10.0], [0.6], [4.9], [10.05], [13.3]]
    )
    val = out.compute(xs.shape[0], {"x": xs})
    ref = torch.floor(xs).remainder(H)
    assert torch.allclose(val, ref, rtol=0.0, atol=1e-3), (val - ref).abs().max()


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
    """A digit scalar off by +/-0.4 still reconstructs the same embedding.

    The nearest threshold is 0.5 away; saturation holds to 17/(scale*S) of
    a ramp edge.
    """
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
