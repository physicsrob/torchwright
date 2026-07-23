"""Measure the fp32 residue left by an MLP-side cancel on the swish machine.

Scope gate for step 3 of the CP-SAT intra-layer reuse plan.  An MLP-side cancel
reuses the activation bypass pair with ``W = -I``: it emits

    Swish(s·(-x))/s - Swish(-s·(-x))/s  =  -x        (exact in real arithmetic)

and adds it to the column already holding ``x``, so ``x + cancel`` should be 0.
The two swish lanes round separately in fp32, so a ~ulp residue can survive.
The ReLU pair is bit-exact (one lane is always exactly 0: ReLU(z) - ReLU(-z) = z
picks the nonzero lane and the other is 0), so this only bites the swish
machine.  DOOM is swiglu, so this measurement decides whether DOOM gets
MLP-cancel at all.

Because IEEE-754 round-to-nearest is symmetric about zero, fp32 subtraction is
exactly anti-symmetric — ``fl(a-b) == -fl(b-a)`` — so the bypass pair is exactly
odd: ``bypass(-x) == -bypass(x)`` bit-for-bit.  The cancel residue therefore
reduces to

    residue(x) = x + bypass(-x) = x - bypass(x)

i.e. the fp32 error of the bypass identity ``bypass(x) == x`` itself.

Decisive gate criterion (per the plan): the residue accumulated over the worst
reuse chain — a column cancelled and reused across the ~85 DOOM layers — must
stay at least one order of magnitude below the 1e-3-class ``c_tol`` budgets.

Norm-path note (Codex finding #5, resolved): the MLP read path goes through the
pre-MLP norm on the HF surface (``h = h + mlp(norm(h))``), and that norm is a
bit-exact identity (pinned-constant RMSNorm or ``nn.Identity``), so ``norm(x) ==
x`` bitwise and measuring the primitive on raw ``x`` measures the right thing.

Run:  make modal-run MODULE=scripts.measure_mlp_cancel_residue
"""

import torch

from torchwright.graph.rope import ROPE_BASE
from torchwright.ops.const import scale as _swish_scale

# Self-match attention hardness (weight_writer._SELF_MATCH_HARDNESS): scales the
# diagonal self-match logit so the diagonal softmax weight is 1.0 to fp32.
SELF_MATCH_HARDNESS = 100.0
N_LAYERS = 85  # DOOM's compiled layer count — the worst reuse-chain length.
N_SAMPLES = 1 << 22  # 4.2M samples per range.


def swish(u: torch.Tensor) -> torch.Tensor:
    """SiLU, matching graph/ffn.py: gate * sigmoid(gate)."""
    return u * torch.sigmoid(u)


def swish_bypass(z: torch.Tensor, s: float) -> torch.Tensor:
    """The runtime's two-lane recombination: Swish(s·z)/s - Swish(-s·z)/s."""
    return swish(s * z) / s - swish(-s * z) / s


def relu_bypass(z: torch.Tensor) -> torch.Tensor:
    """ReLU pair: ReLU(z) - ReLU(-z) = z (bit-exact)."""
    return torch.relu(z) - torch.relu(-z)


def _stats(residue: torch.Tensor) -> dict:
    a = residue.abs()
    return {
        "max": a.max().item(),
        "mean": a.mean().item(),
        "p99": torch.quantile(a.float(), 0.99).item(),
    }


def measure_swish_cancel(mag: float, device: str) -> dict:
    """Residue of a swish MLP-cancel over x ~ Uniform[-mag, mag]."""
    x = (torch.rand(N_SAMPLES, device=device, dtype=torch.float32) * 2 - 1) * mag
    # residue = x + bypass(-x); computed exactly as the runtime would (cancel
    # feeds -x through the pair, result added to the column holding x).
    residue = x + swish_bypass(-x, _swish_scale)
    return _stats(residue)


def measure_relu_cancel(mag: float, device: str) -> dict:
    x = (torch.rand(N_SAMPLES, device=device, dtype=torch.float32) * 2 - 1) * mag
    residue = x + relu_bypass(-x)
    return _stats(residue)


def _self_match_logits(n_pos: int, d_head: int, device: str) -> torch.Tensor:
    """Self-match logit(i, j) = hardness · Σ_p cos((i-j)·θ_p).

    The RoPE Δ=0 transport the attention-cancel head uses (peaks on the
    diagonal).
    """
    p = torch.arange(d_head // 2, device=device, dtype=torch.float32)
    theta = ROPE_BASE ** (-2.0 * p / d_head)  # (d_head/2,)
    idx = torch.arange(n_pos, device=device, dtype=torch.float32)
    delta = idx[:, None] - idx[None, :]  # (n_pos, n_pos)
    angles = delta[:, :, None] * theta[None, None, :]  # (n_pos, n_pos, d_head/2)
    return SELF_MATCH_HARDNESS * torch.cos(angles).sum(dim=-1)


def measure_attention_cancel(
    mag: float, device: str, n_pos: int = 128, d_head: int = 32
) -> dict:
    """Baseline: residue of the attention-cancel head transporting a column.

    Transported via the self-match softmax (weight ≈ 1.0 on the diagonal).
    The cancel adds -(softmax @ v) to the column, so the residue at position i
    is v_i - (softmax @ v)_i.  Off-diagonal leakage (weight not exactly 0 for
    j≠i) is what production already tolerates.
    """
    logits = _self_match_logits(n_pos, d_head, device)  # (n_pos, n_pos)
    weights = torch.softmax(logits, dim=-1)  # rows sum to 1
    # A batch of independent value columns, each a full (n_pos,) vector of
    # DOOM-range values; transport every position and collect all residues.
    batch = 4096
    v = (torch.rand(batch, n_pos, device=device, dtype=torch.float32) * 2 - 1) * mag
    transported = v @ weights.T  # (batch, n_pos): (softmax @ v) per position
    residue = v - transported
    return _stats(residue)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "This measurement must run on GPU (production fp32). "
            "Run: make modal-run MODULE=scripts.measure_mlp_cancel_residue"
        )
    device = "cuda"
    torch.manual_seed(0)
    print(
        f"device={torch.cuda.get_device_name()}  _swish_scale={_swish_scale}  "
        f"SELF_MATCH_HARDNESS={SELF_MATCH_HARDNESS}  N_LAYERS={N_LAYERS}"
    )

    # DOOM residual columns hold values from order ~0.01 up to order 1e4
    # (op_noise_data.json distributions top out near ±1023; residual columns
    # accumulate further).  Sweep the transition region and the DOOM ranges.
    mags = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0, 10000.0]

    print("\n=== swish MLP-cancel residue (primary — the gate) ===")
    print(f"{'|x| range':>12}  {'max_abs':>12}  {'mean_abs':>12}  {'p99_abs':>12}")
    swish_worst_max = 0.0
    swish_worst_mag = None
    for mag in mags:
        s = measure_swish_cancel(mag, device)
        print(f"  ±{mag:<10g}  {s['max']:12.3e}  {s['mean']:12.3e}  {s['p99']:12.3e}")
        if s["max"] > swish_worst_max:
            swish_worst_max = s["max"]
            swish_worst_mag = mag

    print("\n=== ReLU MLP-cancel residue (contrast — expected bit-exact 0) ===")
    print(f"{'|x| range':>12}  {'max_abs':>12}")
    relu_worst_max = 0.0
    for mag in mags:
        r = measure_relu_cancel(mag, device)
        print(f"  ±{mag:<10g}  {r['max']:12.3e}")
        relu_worst_max = max(relu_worst_max, r["max"])

    print("\n=== attention-cancel residue (baseline — production tolerates this) ===")
    print(f"{'|x| range':>12}  {'max_abs':>12}  {'mean_abs':>12}  {'p99_abs':>12}")
    attn_worst_max = 0.0
    for mag in mags:
        a = measure_attention_cancel(mag, device)
        print(f"  ±{mag:<10g}  {a['max']:12.3e}  {a['mean']:12.3e}  {a['p99']:12.3e}")
        attn_worst_max = max(attn_worst_max, a["max"])

    # Compounding: a column can be cancelled/reused once per layer.  Worst case
    # is every reuse landing the worst-magnitude residue with the same sign
    # (a strict upper bound); the random-walk estimate is the realistic figure.
    worst_chain_linear = N_LAYERS * swish_worst_max
    worst_chain_rms = (N_LAYERS**0.5) * swish_worst_max
    c_tol = 1e-3  # the 1e-3-class postcondition budget
    gate_threshold = c_tol / 10.0  # "at least one order of magnitude below"

    print("\n=== compounding over the worst reuse chain ===")
    print(
        f"  worst per-cancel swish residue: {swish_worst_max:.3e} "
        f"(at |x| <= {swish_worst_mag:g})"
    )
    print(
        f"  worst-chain (linear, {N_LAYERS}x same-sign upper bound): "
        f"{worst_chain_linear:.3e}"
    )
    print(f"  worst-chain (random-walk, sqrt({N_LAYERS})x): {worst_chain_rms:.3e}")
    print(f"  attention-cancel worst residue (baseline): {attn_worst_max:.3e}")
    print(f"  ReLU-cancel worst residue: {relu_worst_max:.3e}")

    print("\n=== GATE ===")
    print(
        f"  c_tol (1e-3 class) = {c_tol:.1e};  ships iff worst-chain < "
        f"{gate_threshold:.1e} (one order below)"
    )
    passed = worst_chain_linear < gate_threshold
    print(
        f"  worst-chain (linear upper bound) = {worst_chain_linear:.3e}  "
        f"->  {'PASS' if passed else 'FAIL'}"
    )
    if not passed and worst_chain_rms < gate_threshold:
        print(
            f"  NOTE: linear bound fails but random-walk estimate "
            f"{worst_chain_rms:.3e} clears — inspect the residue distribution."
        )
    rel = "<=" if swish_worst_max <= attn_worst_max else ">"
    print(
        f"  MLP-cancel residue {rel} attention-cancel baseline "
        f"({swish_worst_max:.3e} vs {attn_worst_max:.3e})"
    )
    verdict = (
        "SHIP MLP-cancel for the swish machine"
        if passed
        else ("swish MLP-cancel DECLINED — pin cancel_in_mlp=0 on swish compiles")
    )
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
