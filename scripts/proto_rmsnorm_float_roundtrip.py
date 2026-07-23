"""Does the constant-column RMSNorm-identity survive GENUINE floats (DOOM-style:
reciprocals, distances, wide dynamic range), in the COMMITTED reserve-inside
layout, at the widths that actually ship?

Committed **power-of-two-RMS** gain only (the formula-gain fallback is dropped —
it is not bit-exact on arbitrary floats; see docs/plan_rmsnorm.md):

  - Reserve the constant column(s) INSIDE a power-of-two width ``d`` (no padding):
    ONE column ``= 2^q`` when ``b = log2(d)`` is even; TWO EQUAL columns ``= 2^q``
    when ``b`` is odd (DOOM ships at ``d=8192=2^13``).  The pinned energy
    ``E = n_const * 2^(2q)`` is then ``2^(even)`` (1 col) or ``2^(odd)`` (2 cols),
    chosen so ``rms = sqrt(E/d)`` is an exact power of two ``2^m``.
  - gain ``= 2^m``  ->  ``÷rms`` and ``×gain`` are pure fp32 exponent shifts  ->
    **bit-exact identity for ALL floats** (modulo a denormal floor for near-zero
    data, not exercised by DOOM magnitudes).

Per ``(d, energy_scale)`` we check:
  - ``rms_spread == 0``   (rms bit-exactly constant across positions)
  - ``max|Δ|     == 0``   (bit-exact identity on a single norm)
  - ``drift@200  == 0``   (no accumulation over 200 pure-identity sublayers)
and we stress the out-energy bound by scaling the data toward ``E·2^-24`` — a
proxy for the deepest-layer residual energy (the real per-graph deepest-layer
energy is exercised by proto_rmsnorm_identity.py / the DOOM validation).

Pure torch on CPU: the pow2 claim is a platform-independent IEEE-754 fp32
property, so CPU is a faithful test of what onnxruntime/GPU will do.
"""

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

N_POS = 64
Q = 30  # constant magnitude 2^Q


def doom_like_stream(d_data, n, scale=1.0):
    """A residual stream spanning DOOM-ish magnitudes (distances ~1e3, their
    reciprocals ~1e-3..1e-4, unit-ish trig, mid-range), scaled by ``scale``
    (``>1`` simulates the larger residual energy of a deep layer)."""
    x = torch.zeros(n, d_data)
    w = d_data // 5
    x[:, 0 * w : 1 * w] = torch.empty(n, w).uniform_(-2000, 2000)  # coords/dist
    x[:, 1 * w : 2 * w] = 1.0 / torch.empty(n, w).uniform_(1, 4000)  # reciprocals
    x[:, 2 * w : 3 * w] = torch.empty(n, w).uniform_(-1, 1)  # trig-ish
    x[:, 3 * w : 4 * w] = torch.empty(n, w).uniform_(-50, 50)  # mid range
    # last fifth left zero (free columns)
    return x * scale


def pow2_layout(d):
    """Reserve-inside power-of-two-RMS layout for width ``d = 2^b``.

    Returns ``(n_const, m)``: ``n_const`` columns each ``= 2^Q``, forcing an
    exact ``rms = 2^m``.  One column for even ``b``, two equal for odd ``b``.
    """
    b = d.bit_length() - 1
    assert 1 << b == d, f"d={d} is not a power of two"
    n_const = 1 if b % 2 == 0 else 2  # even b -> 1 col; odd b -> 2 equal cols
    e_exp = 2 * Q + (0 if n_const == 1 else 1)  # log2(E), E = n_const * 2^(2Q)
    assert (e_exp - b) % 2 == 0, "rms exponent not integer — layout bug"
    return n_const, (e_exp - b) // 2


def rmsnorm(res, gain, eps):
    ms = (res * res).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(ms + eps)
    return res / rms * gain, rms


def run(d, eps=0.0, energy_scale=1.0, n_iter=200):
    n_const, m = pow2_layout(d)
    d_data = d - n_const
    data = doom_like_stream(d_data, N_POS, scale=energy_scale)

    const = torch.full((N_POS, n_const), float(2**Q))
    res = torch.cat([data, const], dim=1)  # width is exactly d (a power of two)
    gain = 2.0**m

    normed, rms = rmsnorm(res, gain, eps)
    err = (normed[:, :d_data] - data).abs()
    rms_spread = (rms.max() - rms.min()).item()

    # iterate (proxy for n_iter sublayers of pure identity); a real sublayer
    # rewrites the data columns and leaves the reserved const columns intact.
    cur = res.clone()
    for _ in range(n_iter):
        cur, _ = rmsnorm(cur, gain, eps)
        cur[:, d_data:] = const
    drift = (cur[:, :d_data] - data).abs().max().item()

    data_energy = (data * data).sum(dim=-1).max().item()  # worst-position Σdata²
    E = float(n_const) * (2.0 ** (2 * Q))
    bound = E * 2**-24  # half-ULP of E; the data energy must stay under this
    ok = "OK" if (err.max() == 0 and rms_spread == 0 and drift == 0) else "BIT-DRIFT"
    print(
        f"  d={d:<5} b={d.bit_length() - 1} cols={n_const} rms=2^{m} gain={gain:.2e} "
        f"eps={eps:g} scale={energy_scale:<5g} "
        f"Σdata²={data_energy:.2e}(<{bound:.1e}? {'y' if data_energy < bound else 'N'}) "
        f"rms_spread={rms_spread:.0e} max|Δ|={err.max():.0e} "
        f"drift@{n_iter}={drift:.0e}  [{ok}]"
    )
    return err.max().item(), rms_spread, drift


def main():
    print("Reserve-inside power-of-two-RMS identity, DOOM-like floats:\n")
    print("d=1024 (even b -> ONE constant column):")
    run(1024, eps=0.0)
    run(1024, eps=1e-6)

    print("\nd=8192 (odd b -> TWO equal constant columns; DOOM's shipping width):")
    run(8192, eps=0.0)
    run(8192, eps=1e-6)

    print("\nDeepest-layer energy stress at d=8192 (scale data toward the bound):")
    for scale in [1, 3, 10, 30, 100, 300]:
        run(8192, eps=1e-6, energy_scale=scale)
    print(
        "\n(Expect bit-exact identity while Σdata² stays under the bound, and the\n"
        " identity to break once the data energy crosses it — confirming the\n"
        " out-energy constraint is real and where it bites.)"
    )


if __name__ == "__main__":
    main()
