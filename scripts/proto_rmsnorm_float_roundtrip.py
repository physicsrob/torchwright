"""Does the constant-column RMSNorm-identity survive GENUINE floats (DOOM-style:
reciprocals, distances, wide dynamic range)?  Two gain choices:

  (A) formula gain C = sqrt(K/(d+k))            -> what the calculator used
  (B) power-of-two gain C = 2^p with width and K chosen so rms == 2^p exactly

Round-trip error per value = |norm(x) - x|.  We also iterate the norm N times to
see how per-layer error accumulates.
"""

import math
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

D = 1024  # power of two on purpose
N_POS = 64


def doom_like_stream(d, n):
    """A residual stream spanning DOOM-ish magnitudes: distances ~1e3, their
    reciprocals ~1e-3..1e-4, unit-ish trig values, signs, and zeros."""
    x = torch.zeros(n, d)
    x[:, 0:200] = torch.empty(n, 200).uniform_(-2000, 2000)  # coords/dist
    x[:, 200:400] = 1.0 / torch.empty(n, 200).uniform_(1, 4000)  # reciprocals
    x[:, 400:600] = torch.empty(n, 200).uniform_(-1, 1)  # trig-ish
    x[:, 600:800] = torch.empty(n, 200).uniform_(-50, 50)  # mid range
    # rest left zero (free columns)
    return x


def rmsnorm(res_wide, gain, eps):
    ms = (res_wide * res_wide).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(ms + eps)
    return res_wide / rms * gain, rms


def run(mode, k_cols, M, p=None, eps=0.0):
    d = D
    dk = d + k_cols
    data = doom_like_stream(d, N_POS)

    if mode == "formula":
        const = torch.full((N_POS, k_cols), float(M))
        const[:, k_cols // 2 :] *= -1
        K = k_cols * (M * M)
        C = math.sqrt(K / dk + eps)
    elif mode == "pow2":
        # Choose ONE constant column = 2^q so K = 2^(2q); width dk is a power of
        # two; gain = 2^p with rms = sqrt(2^(2q)/dk) forced to 2^p exactly.
        assert k_cols == 1
        # dk = d+1 is NOT a power of two -> pad data width so dk is.  Use d s.t.
        # d+1 == 2^b: pick b with 2^b > d, put the const col + zero-pad columns.
        b = (dk - 1).bit_length()  # smallest power of two >= dk
        if b % 2:  # force EVEN exponent so 2^(b/2) is exact
            b += 1
        dk_pad = 1 << b
        pad = dk_pad - d - 1
        data = torch.cat([data, torch.zeros(N_POS, pad)], dim=1)
        q = 30
        const = torch.full((N_POS, 1), float(2**q))
        C = 2.0 ** (q - b // 2)  # sqrt(2^(2q)/2^b) = 2^(q - b/2)
    else:
        raise ValueError(mode)

    res = torch.cat([data, const], dim=1)
    data_cols = data.shape[1]

    # single round-trip
    normed, rms = rmsnorm(res, C, eps)
    err = (normed[:, :data_cols] - data).abs()
    rel = err / data.abs().clamp_min(1e-30)
    rms_spread = (rms.max() - rms.min()).item()

    # iterate N times (proxy for N sublayers of pure identity)
    cur = res.clone()
    for _ in range(200):
        cur, _ = rmsnorm(cur, C, eps)
        # restore const cols exactly (a real layer rewrites data, not consts)
        cur[:, data_cols:] = const
    drift = (cur[:, :data_cols] - data).abs().max().item()

    label = f"{mode:8} k={k_cols} M/2^q C={C:.4e}"
    print(
        f"{label}  rms_spread={rms_spread:.2e}  "
        f"max|Δ|={err.max():.3e}  max_rel={rel.max():.3e}  "
        f"drift@200={drift:.3e}"
    )


print("DOOM-like float stream, d=%d, eps=0:" % D)
run("formula", k_cols=4, M=1e9)
run("formula", k_cols=4, M=1e12)
run("pow2", k_cols=1, M=None)

print("\nwith eps=1e-6 (typical RMSNorm eps):")
run("formula", k_cols=4, M=1e9, eps=1e-6)
run("pow2", k_cols=1, M=None, eps=1e-6)
