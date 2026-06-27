"""Prototype: insert a real RMSNorm into a compiled torchwright transformer,
pinning the residual RMS with large constant columns so the norm is the
identity, and confirm output is unchanged vs the no-norm baseline.

Run from repo root:  python /path/to/proto_rmsnorm.py
"""

import math
import torch

from examples.calculator_simple import create_network_parts
from torchwright.compiler.export import compile_headless

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

MAX_DIGITS = 1
D = 1024
D_HEAD = 16

# Constant-column knobs.
K_COLS = 4  # how many reserved constant columns
M = 1e9  # magnitude of each constant column
EPS = 1e-6


def build_prefill(net, embedding, prompt_tokens):
    n = len(prompt_tokens)
    return net.get_input_res_stream(n, {"embedding_input": prompt_tokens}), n


def rmsnorm_identity(net, res_data):
    """Run the layer stack as a pre-norm transformer with a REAL RMSNorm,
    where K_COLS constant columns pin the RMS to a constant C and the gain is
    set to C (so the norm is algebraically the identity).

    res_data: (n, D) initial residual stream (data columns only).
    Returns (n, D) final residual stream over data columns.
    """
    n = res_data.shape[0]
    d = res_data.shape[1]
    dk = d + K_COLS

    # Append the constant columns: half +M, half -M (sign is irrelevant to RMS;
    # the ± keeps the stream mean near zero, harmless for RMSNorm).
    const = torch.full((n, K_COLS), M)
    const[:, K_COLS // 2 :] = -M
    res = torch.cat([res_data, const], dim=1)

    # Fixed gain = the constant RMS the constant columns force.
    K = K_COLS * (M * M)
    C = math.sqrt(K / dk + EPS)

    rms_seen = []
    for layer in net.layers:
        for sublayer in (layer.attn, layer.mlp):
            ms = (res * res).mean(dim=-1, keepdim=True)  # (n,1)
            rms = torch.sqrt(ms + EPS)
            rms_seen.append(rms.flatten())
            normed = res / rms * C  # gain = C
            data_in = normed[:, :d]
            sub_full = sublayer.forward(data_in)  # (n, d) = x + op(x)
            delta = sub_full - data_in  # = op(normed_x)
            new = res.clone()
            new[:, :d] = res[:, :d] + delta
            res = new  # const cols untouched
    return res[:, :d], C, torch.stack(rms_seen)


def run_case(max_digits, prompt, m, k_cols=K_COLS):
    global M, K_COLS
    M, K_COLS = m, k_cols
    output_node, pos_encoding, embedding = create_network_parts(max_digits=max_digits)
    compiled = compile_headless(output_node, pos_encoding, d=D, d_head=D_HEAD)
    net = compiled._net
    out_idx = compiled._output_indices
    res0, n = build_prefill(net, embedding, prompt)

    with torch.no_grad():
        base = net.forward(res0.clone())
        normed_full, C, rms_seen = rmsnorm_identity(net, res0.clone())
    base_out = base[:, out_idx]
    normed_out = normed_full[:, out_idx]

    table = embedding.table
    base_tok = (base_out @ table.T).argmax(-1)
    norm_tok = (normed_out @ table.T).argmax(-1)
    return {
        "C": C,
        "rms_spread": (rms_seen.max() - rms_seen.min()).item(),
        "max_abs_diff": (normed_out - base_out).abs().max().item(),
        "decode_ok": bool((base_tok == norm_tok).all()),
        "n_layers": len(net.layers),
    }


def main():
    cases = [
        (1, ["<bos", "2", "*", "3", "\n"], 1e9),
        (3, ["<bos", "1", "2", "*", "3", "4", "\n"], 1e9),
        (3, ["<bos", "9", "9", "9", "*", "9", "9", "9", "\n"], 1e9),
    ]
    print("=== identity check across cases (M=1e9, K_COLS=4) ===")
    for md, prompt, m in cases:
        r = run_case(md, prompt, m)
        p = "".join(prompt[1:])
        print(
            f" digits={md} prompt={p!r:18} layers={r['n_layers']:3}  "
            f"rms_spread={r['rms_spread']:.2e}  max|d|={r['max_abs_diff']:.2e}  "
            f"decode_ok={r['decode_ok']}"
        )

    print("\n=== sweep M (3-digit 999*999, the largest data energy) ===")
    for m in [1e9, 1e6, 1e4, 1e3, 1e2, 1e1]:
        r = run_case(3, ["<bos", "9", "9", "9", "*", "9", "9", "9", "\n"], m)
        print(
            f" M={m:>7.0e}  C={r['C']:.3e}  rms_spread={r['rms_spread']:.3e}  "
            f"max|d|={r['max_abs_diff']:.3e}  decode_ok={r['decode_ok']}"
        )


if __name__ == "__main__":
    main()
