"""Pins every numeric constant stated in ``docs/ops_plain_english.md``.

The SwiGLU op designs in that doc make quantitative claims — hinge
deviation bounds, error peaks, fp32 saturation thresholds, exactness
identities.  Each test here re-derives one of them, so the doc's numbers
are pinned by CI rather than trusted from the design conversation.  If
one of these fails (e.g. a torch upgrade changes sigmoid rounding), the
corresponding doc claim needs revisiting — the doc cites this file.

All tests are CPU-only: exact-math checks in float64, plus explicit
float32 checks for the saturation-dependent claims.  The doc's
bit-exactness claims additionally assume the *deployed* kernels
(torch-CUDA, onnxruntime-CUDA) saturate identically — probed
permanently in ``test_swish_saturation_cuda.py`` (torch-CUDA),
torchwright_doom's ``tests/inference/test_ort_cuda_saturation.py``
(the deployed ORT pair), and ``test_ort_cpu_saturation.py`` (the CPU
onnxruntime parity oracle, the one kernel with a different profile).
"""

import torch

#: The module hinge-sharpening constant the doc pins (`scale = 100`).
SCALE = 100.0

#: Swish's global-minimum magnitude and location:
#: min_z z*sigmoid(z) = -0.2784645... at z = -1.2784645...
SWISH_PEAK = 0.2784645
SWISH_ARGMIN = 1.2784645


def test_scale_is_the_module_constant():
    """The value every claim below is derived at IS the shipped module
    constant (ops/const.py) — the one the compiler's swish bypass pair and
    the swiglu ops fold into weights."""
    from torchwright.ops.const import scale

    assert scale == SCALE


def _swish(z: torch.Tensor) -> torch.Tensor:
    return z * torch.sigmoid(z)


def _hinge(z: torch.Tensor, scale: float = SCALE) -> torch.Tensor:
    """The doc's `hinge(z) = Swish(scale*z)/scale`."""
    return _swish(scale * z) / scale


def test_sharpened_hinge_uniform_bound():
    """Preamble/compare: |hinge(z) - relu(z)| <= 0.2785/scale everywhere,
    peaking at z = -1.278/scale (Swish's argmin)."""
    z = torch.linspace(-8, 8, 400_001, dtype=torch.float64)
    dev = (_hinge(z, 1.0) - torch.relu(z)).abs()  # scale=1: peak is the raw constant
    peak = dev.max().item()
    argmax = z[dev.argmax()].item()
    assert abs(peak - SWISH_PEAK) < 1e-6
    # |swish(z) - relu(z)| = |z|*sigmoid(-|z|) is even: both z = ±1.278
    # achieve the peak (the dip on the left, the identity gap on the right).
    assert abs(abs(argmax) - SWISH_ARGMIN) < 1e-3

    # The affine sandwich constant is this same peak, rounded UP for
    # soundness (docs/affine_bounds.md).
    from torchwright.graph.affine_rules import _SWISH_SANDWICH_C

    assert _SWISH_SANDWICH_C >= peak
    assert _SWISH_SANDWICH_C - peak < 1e-5


def test_naive_transliteration_fails():
    """compare: the unsharpened ramp Swish(z) - Swish(z-1) misses its
    levels by 0.27 at the contract points and ~0.10 two ramp-widths past
    saturation — sharpening is mandatory, not tuning."""

    def ramp(z: float) -> float:
        t = torch.tensor([z, z - 1.0], dtype=torch.float64)
        s = _swish(t)
        return (s[0] - s[1]).item()

    assert abs(ramp(0.0) - 0.2689) < 1e-3  # false contract point (target 0)
    assert abs((ramp(3.0) - 1.0) - 0.0961) < 1e-3  # two ramp-widths past true


def test_fp32_sigmoid_saturation_threshold():
    """Preamble: fp32 sigmoid computes exactly 1.0 once its input exceeds
    ~17 (e^-17 is below fp32's resolution next to 1); 16 is not enough."""
    assert torch.sigmoid(torch.tensor(17.0, dtype=torch.float32)).item() == 1.0
    assert torch.sigmoid(torch.tensor(16.0, dtype=torch.float32)).item() < 1.0


def test_compare_contract_points_bit_exact_fp32():
    """compare: at scale=100 in fp32, the contract-point outputs are
    bit-exact (+1/-1), because both hinges are saturated or on-bend."""

    def compare_out(z: float) -> float:
        # y = F + (T-F) * (hinge(z) - hinge(z-1)), T=+1, F=-1, fp32 throughout
        t = torch.tensor([z, z - 1.0], dtype=torch.float32)
        h = _swish(SCALE * t) / SCALE
        return (-1.0 + 2.0 * (h[0] - h[1])).item()

    assert compare_out(0.0) == -1.0  # x == thresh
    assert compare_out(1.0) == 1.0  # x == thresh + 1/sharpness


def test_multiply_identity_exact():
    """multiply: Swish(a)*b + Swish(-a)*(-b) = a*b exactly, all a, b —
    the sigma(a) + sigma(-a) = 1 cancellation."""
    g = torch.linspace(-50, 50, 501, dtype=torch.float64)
    a, b = torch.meshgrid(g, g, indexing="ij")
    lhs = _swish(a) * b + _swish(-a) * (-b)
    assert torch.allclose(lhs, a * b, rtol=0.0, atol=1e-9)


def test_bypass_identity_exact_at_any_sharpening():
    """min (and mlp_bypass): Swish(s*a)/s - Swish(-s*a)/s = a exactly,
    sharpened or not."""
    a = torch.linspace(-1000, 1000, 20_001, dtype=torch.float64)
    for s in (1.0, SCALE):
        lhs = _swish(s * a) / s - _swish(-s * a) / s
        assert torch.allclose(lhs, a, rtol=1e-12, atol=1e-9)


def test_abs_error_peak_and_one_sidedness():
    """abs: hinge(x) + hinge(-x) = x*tanh(scale*x/2) lies in [0, |x|],
    worst underestimate 0.557/scale (= 2 * hinge peak) at
    |x| = 1.278/scale."""
    u = torch.linspace(0, 8, 400_001, dtype=torch.float64)  # scale=1 units
    f = u * torch.tanh(u / 2)
    err = u - f
    assert (f >= -1e-15).all()  # never negative
    assert (err >= -1e-15).all()  # never above |x|
    peak = err.max().item()
    argp = u[err.argmax()].item()
    assert abs(peak - 2 * SWISH_PEAK) < 1e-6
    assert abs(argp - SWISH_ARGMIN) < 1e-3


def test_abs_integer_grid_bit_exact_fp32():
    """abs: at scale=100 in fp32, bit-exact on the whole integer grid
    (tanh/sigmoid saturation)."""
    x = torch.arange(-1000.0, 1001.0, dtype=torch.float32)
    f = _swish(SCALE * x) / SCALE + _swish(-SCALE * x) / SCALE
    assert torch.equal(f, x.abs())


def test_min_hinge_form():
    """min: a - hinge(a-b) over-estimates min by at most 0.2785/scale,
    ties are exact, and the error is symmetric in the arguments despite
    the asymmetric construction."""
    g = torch.linspace(-5, 5, 401, dtype=torch.float64)
    a, b = torch.meshgrid(g, g, indexing="ij")
    m = a - _hinge(a - b)
    err = m - torch.minimum(a, b)
    assert (err >= -1e-12).all()  # one-sided: never under-reads
    assert err.max().item() <= SWISH_PEAK / SCALE + 1e-9
    # ties exact: hinge(0) = 0
    assert _hinge(torch.zeros(1, dtype=torch.float64)).item() == 0.0
    # symmetry: min(a,b) and min(b,a) compute the same value
    m_swapped = b - _hinge(b - a)
    assert torch.allclose(m, m_swapped, atol=1e-12)


def test_gated_select_off_branch_exact_zero_fp32():
    """broadcast_select/select: at mask=-1, scale=100 in fp32, the losing
    branch contributes exactly zero — sigmoid(-100) computes as 0.0 on the
    CPU kernel.  (A kernel returning the denormal e^-100 instead would leak
    <= ~1e-42 * |branch| — the deployed-kernel probe is a migration-
    checklist item.)"""
    assert torch.sigmoid(torch.tensor(-100.0, dtype=torch.float32)).item() == 0.0
    f = torch.linspace(-4096, 4096, 1001, dtype=torch.float32)
    leak = _swish(torch.tensor(-SCALE, dtype=torch.float32)) * f / SCALE
    assert (leak == 0.0).all()


def test_gated_select_winning_branch_one_ulp():
    """broadcast_select: at mask=+1 the winning branch passes with at most
    1 fp32 ulp *relative* rounding (the value rides through x scale, then
    / scale) — often bit-exact, but not always: the recorded regression
    versus today's approximate=False path, which is bit-exact."""
    t = torch.linspace(-5000, 5000, 100_001, dtype=torch.float32)
    t = t[t != 0]
    gate = _swish(torch.tensor(SCALE, dtype=torch.float32))  # Swish(100) = 100.0
    out = gate * t * torch.tensor(1.0 / SCALE, dtype=torch.float32)
    rel = ((out - t) / t).abs()
    assert rel.max().item() <= 1.3e-7  # ~1 ulp (eps = 1.19e-7)
    assert (out != t).any()  # the bit-exactness regression is real


def test_gated_select_mask_deviation_passthrough():
    """broadcast_select: a saturated gate IS the mask (sigma == 1.0 for
    scale*m >= 17, i.e. m >= 0.17), so a mask off +1 by delta mis-scales
    the winner by exactly delta * value — actual value, not the range
    maximum M."""
    m = torch.linspace(0.17, 2.0, 4001, dtype=torch.float32)
    gate = _swish(SCALE * m) / SCALE
    assert ((gate - m) / m).abs().max().item() <= 1.3e-7  # gate == mask, 1 ulp
    # exact math: the deviation passes through linearly
    delta, t, f = 0.011, 1000.0, -777.0
    m1 = torch.tensor(1.0 - delta, dtype=torch.float64)
    out = _swish(SCALE * m1) * t / SCALE + _swish(-SCALE * m1) * f / SCALE
    assert abs(out.item() - (1 - delta) * t) < 1e-9


def test_relu_cancellation_absolute_error_at_M():
    """broadcast_select what-dies note: today's approximate=True recovers
    the winner via (M + t) - M in fp32, absolute error up to half an ulp
    of M — 3e-5 at M=1000 — error at the offset's magnitude even for tiny
    values, versus the gated form's relative-to-the-value ulp."""
    M = torch.tensor(1000.0, dtype=torch.float32)
    t = torch.linspace(-1, 1, 100_001, dtype=torch.float32)
    err = ((M + t) - M) - t
    half_ulp = ((torch.nextafter(M, torch.tensor(float("inf"))) - M) / 2).item()
    assert abs(half_ulp - 3.0517578125e-05) < 1e-12
    assert err.abs().max().item() <= half_ulp
    assert err.abs().max().item() >= half_ulp / 2  # genuinely at M's scale


def test_table_lookup_2d_telescoping():
    """table_lookup_2d: the swish form — per-axis two-stage staircases, the
    column axis consuming its steps through gated lanes (gate =
    hinge(1 - step), up = adjacent-column difference) — recovers integer-
    grid entries and edge clamps exactly, and yields genuine bilinear
    interpolation inside boundary bands (today's op disclaims bilinear)."""
    s, W = 100.0, 2.0
    dt = torch.float64

    def lookup(T, i, j):
        A, B = T.shape
        i = torch.as_tensor(i, dtype=dt)
        j = torch.as_tensor(j, dtype=dt)
        i = i - _hinge(i - (A - 1))  # min-clamp (swish min)
        j = j - _hinge(j - (B - 1))

        def steps(x, n):  # bounded step in [0, W] per boundary k-0.5
            k = torch.arange(1, n, dtype=dt)
            t = s * (x - (k - 0.5)) + 0.5
            return _hinge(t) - _hinge(t - W)

        # row axis: constant deltas -> degenerate lanes
        ind_i = _hinge(1.0 - steps(i, A))
        row = T[A - 1] - (ind_i.unsqueeze(1) * (T[1:] - T[:-1])).sum(0)
        # column axis: live deltas -> one gated lane per boundary
        gates = _hinge(1.0 - steps(j, B))
        return (row[B - 1] + (gates * (row[:-1] - row[1:])).sum()).item()

    g = torch.Generator().manual_seed(0)
    T = torch.randn(5, 4, generator=g, dtype=dt) * 100

    # integer grid exact
    for a in range(5):
        for b in range(4):
            assert abs(lookup(T, a, b) - T[a, b].item()) < 1e-9
    # out-of-range clamps to the edge entries
    assert abs(lookup(T, 9.0, -3.0) - T[4, 0].item()) < 1e-12
    assert abs(lookup(T, -7.0, 11.0) - T[0, 3].item()) < 1e-12
    # mid-band j: clean two-column blend
    assert abs(lookup(T, 2, 1.5) - 0.5 * (T[2, 1] + T[2, 2]).item()) < 1e-9
    # corner band: genuine bilinear interpolation
    want = 0.25 * (T[1, 1] + T[1, 2] + T[2, 1] + T[2, 2]).item()
    assert abs(lookup(T, 1.5, 1.5) - want) < 1e-9
    # off-center blend: coefficient alpha = 0.8; exact-math tail is
    # e^-(scale*alpha) ~ 4e-10 (in fp32 it vanishes: sigma(20) == 1.0)
    want = (0.2 * T[2, 1] + 0.8 * T[2, 2]).item()
    assert abs(lookup(T, 2, 1.5 + 0.3 / s) - want) < 1e-6


def test_onehot_lookup_counting_margin():
    """onehot_lookup: the -(n_blocks - 0.5) counting trick parks every lane
    at argument +0.5 (winner) or <= -0.5 (rest).  Sharpened: hinge(0.5) is
    exactly 0.5 in fp32; hinge(-0.5) leaks ~1e-22 per row (e^-50 is
    representable, unlike sigma(-100) = 0); a slightly-off one-hot shifts
    the indicator by exactly its count deviation (saturated gate)."""
    half = torch.tensor(SCALE * 0.5, dtype=torch.float32)
    assert (_swish(half) / SCALE).item() == 0.5  # winner indicator exact
    leak = (_swish(-half) / SCALE).abs().item()
    assert 0.0 < leak <= 1e-21  # non-match: negligible but not bit-zero

    # noise pass-through: linear in the count deviation while saturated
    eps = torch.linspace(0.0, 0.33, 1000, dtype=torch.float32)
    g = _swish(SCALE * (0.5 - eps)) / SCALE
    assert (g - (0.5 - eps)).abs().max().item() <= 1e-7

    # mini 2-block table, fp32 end to end: winner ~1 ulp, no-match exact
    g_ = torch.Generator().manual_seed(1)
    vals = torch.randn(6, 3, generator=g_, dtype=torch.float32) * 1000
    default = torch.randn(3, generator=g_, dtype=torch.float32) * 100
    keys = []
    for i in range(6):
        k = torch.zeros(8, dtype=torch.float32)
        k[i % 4] = 1.0
        k[4 + i % 3] = 1.0
        keys.append(k)

    def lookup(x):
        out = default.clone()
        for k, v in zip(keys, vals):
            lane = _swish(SCALE * (x @ k - 1.5))  # bias -(n_blocks - 0.5)
            out = out + lane * (2.0 * (v - default) / SCALE)
        return out

    for k, v in zip(keys, vals):
        assert (lookup(k) - v).abs().max().item() <= 1e-3  # ~ulp(1000)
    miss = torch.zeros(8)
    miss[3] = 1.0
    miss[5] = 1.0  # (3, 1): not a table key
    assert torch.equal(lookup(miss), default)  # leaks vanish under fp32 add


def test_scalar_to_embedding_staircase():
    """scalar_to_embedding: the 9-unit-step staircase (piecewise_linear
    special case) ports hinge-for-hinge.  Integer digits put every hinge
    argument on an exact saturated integer, so component error is out_proj
    rounding only (~few ulps at norm-40 embeddings); mid-ramp blends the
    adjacent embeddings; the pair-spacing audit reduces to scale > 34."""
    S = 10.0  # step_sharpness
    g = torch.Generator().manual_seed(2)
    E = torch.randn(10, 32, generator=g, dtype=torch.float32) * (40.0 / 32**0.5)

    def s2e(x, dtype=torch.float32):
        x = torch.as_tensor(x, dtype=dtype)
        Ed = E.to(dtype)
        out = Ed[0].clone()
        for k in range(9):
            z = S * (x - (k + 0.5))
            step = _swish(SCALE * z) / SCALE - _swish(SCALE * (z - 1.0)) / SCALE
            out = out + step * (Ed[k + 1] - Ed[k])
        return out

    # integer digits: exact indicators, few-ulp out_proj rounding
    for d in range(10):
        assert (s2e(float(d)) - E[d]).abs().max().item() <= 1e-5
    # ±0.4 input noise absorbed by saturation (contract headroom)
    for d in range(10):
        for eps in (-0.4, 0.4):
            x = min(max(d + eps, 0.0), 9.0)
            assert (s2e(x) - E[d]).abs().max().item() <= 1e-4
    # mid-ramp (x = 0.55): clean blend of the adjacent embeddings
    want = 0.5 * (E[0].double() + E[1].double())
    assert (s2e(0.55, dtype=torch.float64) - want).abs().max().item() <= 1e-9
    # spacing: pair hinges 1/S apart, fillet radius 17/(scale*S) --
    # fillets never overlap iff 34/(scale*S) < 1/S, i.e. scale > 34
    assert 34.0 / (SCALE * S) < 1.0 / S and SCALE > 34.0


def test_hinge_fillet_width():
    """Entries' fillet radius: beyond |z| ~ 17/scale the hinge deviation
    from ReLU is below 1e-8 — the '17/(scale*sharpness)' error zones."""
    for z in (0.17, 0.5, 1.0):
        zt = torch.tensor(z, dtype=torch.float64)
        assert abs(_hinge(zt).item() - z) < 1e-8  # positive side -> identity
        assert abs(_hinge(-zt).item()) < 1e-8  # negative side -> zero
