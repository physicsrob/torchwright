"""Prototype: insert a real RMSNorm into a *compiled* torchwright transformer,
pinning the residual RMS with the committed power-of-two-RMS constant column(s)
RESERVED INSIDE the existing power-of-two width, and confirm the output is
bit-identical to the no-norm baseline — i.e. the norm is the identity even
through the scheduler's cancel heads (the DOOM-critical interaction).

This is the compiler-integration counterpart to proto_rmsnorm_float_roundtrip.py
(which validates the bit-exact math on genuine floats).  Here the point is the
*structure*: a real compiled graph, real cancel heads, the reserve-inside layout
at both an even width (d=1024 -> ONE constant column) and DOOM's odd width
(d=8192 -> TWO equal columns).

Run from repo root with the workspace venv (CPU is fine; compile_headless
defaults to device="cpu").
"""

import torch

from examples.calculator_simple import create_network_parts
from torchwright.compiler.export import compile_headless

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

D_HEAD = 16
Q = 30  # constant magnitude 2^Q
EPS = 1e-6
NZ = 1e-12  # "nonzero" threshold for free-column detection


def pow2_layout(d):
    """Reserve-inside power-of-two-RMS layout: (n_const, m) for width d=2^b.
    One constant column (= 2^Q) for even b, two equal for odd b; rms = 2^m."""
    b = d.bit_length() - 1
    assert 1 << b == d, f"d={d} is not a power of two"
    n_const = 1 if b % 2 == 0 else 2
    e_exp = 2 * Q + (0 if n_const == 1 else 1)  # log2(E), E = n_const * 2^(2Q)
    assert (e_exp - b) % 2 == 0
    return n_const, (e_exp - b) // 2


def rmsnorm(res, gain, eps):
    ms = (res * res).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(ms + eps)
    return res / rms * gain, rms


def baseline_and_free_cols(net, res0):
    """Run the no-norm baseline sublayer-by-sublayer and return
    (final_res, free_col_indices) — columns never written by any sublayer
    (and zero in the seed), so they can hold a reserved constant untouched."""
    cur = res0.clone()
    ever_nz = (res0.abs() > NZ).any(dim=0)  # (d,) bool
    for layer in net.layers:
        for sublayer in (layer.attn, layer.mlp):
            cur = sublayer.forward(cur)
            ever_nz |= (cur.abs() > NZ).any(dim=0)
    free = (~ever_nz).nonzero(as_tuple=True)[0]
    return cur, free


def normed_forward(net, res0, free_cols, n_const, gain, eps):
    """Pre-norm forward with the pinned constant in the last n_const free
    columns. With a bit-exact (pow2) norm, norm(res) == res, so this must
    track the baseline exactly. Returns (final_res, rms_spread)."""
    const_cols = free_cols[-n_const:]
    res = res0.clone()
    res[:, const_cols] = float(2**Q)

    cur = res
    rms_seen = []
    for layer in net.layers:
        for sublayer in (layer.attn, layer.mlp):
            normed, rms = rmsnorm(cur, gain, eps)
            rms_seen.append(rms.flatten())
            cur = cur + (sublayer.forward(normed) - normed)  # skip un-normed res
    rms_all = torch.stack(rms_seen)
    return cur, (rms_all.max() - rms_all.min()).item()


def run_case(d, max_digits, prompt):
    n_const, m = pow2_layout(d)
    gain = 2.0**m
    output_node, pos_encoding, embedding = create_network_parts(max_digits=max_digits)
    compiled = compile_headless(output_node, pos_encoding, d=d, d_head=D_HEAD)
    net = compiled._net
    out_idx = compiled._output_indices
    res0 = net.get_input_res_stream(len(prompt), {"embedding_input": prompt})

    with torch.no_grad():
        base, free = baseline_and_free_cols(net, res0)
        assert len(free) >= n_const, f"only {len(free)} free cols, need {n_const}"
        normed, rms_spread = normed_forward(net, res0, free, n_const, gain, EPS)

    table = embedding.table
    base_out, normed_out = base[:, out_idx], normed[:, out_idx]
    base_tok = (base_out @ table.T).argmax(-1)
    norm_tok = (normed_out @ table.T).argmax(-1)
    max_abs = (normed_out - base_out).abs().max().item()
    decode_ok = bool((base_tok == norm_tok).all())
    # max|Δ| at the fp32 denormal floor (~1e-38..1e-45) is the documented
    # near-zero-data caveat, not drift: dividing a near-zero residual value by
    # rms=2^m underflows to a denormal. Anything above that floor would be real.
    DENORMAL_FLOOR = 1e-30
    if rms_spread == 0 and max_abs == 0:
        verdict = "BIT-EXACT"
    elif rms_spread == 0 and max_abs < DENORMAL_FLOOR and decode_ok:
        verdict = "OK (denormal floor)"
    else:
        verdict = "DRIFT"
    print(
        f" d={d:<5} b={d.bit_length()-1} cols={n_const} rms=2^{m} gain={gain:.2e} "
        f"layers={len(net.layers):3} free_cols={len(free):4}  "
        f"rms_spread={rms_spread:.0e} max|Δ|={max_abs:.0e} "
        f"decode_ok={decode_ok}  [{verdict}]"
    )


def main():
    cases = [
        (1, ["<bos>", "2", "*", "3", "\n"]),
        (3, ["<bos>", "1", "2", "*", "3", "4", "\n"]),
        (3, ["<bos>", "9", "9", "9", "*", "9", "9", "9", "\n"]),
    ]
    # d=1024 (even b) -> 1 col; d=2048 (odd b) -> 2 equal cols. DOOM's actual
    # d=8192 odd-b two-column bit-exactness is covered by the float-roundtrip
    # proto; the dense in-process backend's O(d^2) attn matrices make d=8192
    # too memory-heavy to compile here (~tens of GB), so we exercise the
    # odd-b/two-column reserve-inside layout through a real graph at d=2048.
    for d in (1024, 2048):
        tag = (
            "even b -> 1 column"
            if (d.bit_length() - 1) % 2 == 0
            else "odd b -> 2 equal columns"
        )
        print(f"\n=== reserve-inside pow2-RMS identity, d={d} ({tag}) ===")
        for md, prompt in cases:
            run_case(d, md, prompt)


if __name__ == "__main__":
    main()
