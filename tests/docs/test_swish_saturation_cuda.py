"""The torch-CUDA half of the deployed-kernel saturation probe.

``test_swish_constants.py`` pins every numeric claim in
``docs/ops_plain_english.md`` on the CPU kernels.  The doc's
bit-exactness claims additionally assume the *deployed* kernels saturate
identically: fp32 sigmoid computes exactly 1.0 once its input exceeds
~17, exactly 0.0 at -128 (e^-128 sits below fp32's subnormal floor), and
the compositions built on those saturations (compare's contract points,
abs on the integer grid, a gated select's dead branch) are therefore
bit-exact end to end.  This file re-verifies those claims on the
torch-CUDA kernel on every suite run — ``make test`` runs on Modal
A100s, and kernel behavior can shift under torch upgrades.  If one of
these fails, the doc's "bit-exact" claims and any exact-equality op
tests must be softened before swiglu ops land (see
``docs/swiglu_step2_plan.md``, A0 — runtime saturation probe).

The onnxruntime-CUDA half lives in torchwright_doom's suite
(``tests/inference/test_ort_cuda_saturation.py`` there), next to the
runtime environment that pins the deployed ORT version + CUDA execution
provider.

Skipped when CUDA is unavailable (local CPU-only runs): the CPU claims
are already pinned by ``test_swish_constants.py``.
"""

import pytest
import torch

from tests.docs.test_swish_constants import SCALE, _swish

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-kernel saturation probe; CPU claims are pinned in "
    "test_swish_constants.py",
)


def _cuda(vals) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float32, device="cuda")


def test_sigmoid_saturates_to_one_from_17():
    """Preamble: fp32 sigmoid is exactly 1.0 for every input >= 17.

    16 is not enough — the same threshold the CPU kernel pins.
    """
    z = torch.linspace(17.0, 200.0, 100_001, dtype=torch.float32, device="cuda")
    sig = torch.sigmoid(z)
    bad = z[sig != 1.0]
    assert bad.numel() == 0, (
        f"sigmoid != 1.0 at {bad.numel()} points in [17, 200]; "
        f"largest offender z={bad.max().item():.6f}, "
        f"worst 1-sig={(1.0 - sig[sig != 1.0]).max().item():.3e}"
    )
    assert torch.sigmoid(_cuda(17.0)).item() == 1.0
    assert torch.sigmoid(_cuda(16.0)).item() < 1.0


def test_sigmoid_saturates_to_zero_at_minus_scale():
    """broadcast_select/select: sigma(-128) computes as exactly 0.0.

    The losing branch of a gated select contributes bit-zero, no
    denormal leak into the output.
    """
    assert torch.sigmoid(_cuda(-SCALE)).item() == 0.0
    f = torch.linspace(-4096, 4096, 1001, dtype=torch.float32, device="cuda")
    leak = _swish(_cuda(-SCALE)) * f / SCALE
    assert (leak == 0.0).all()


def test_swish_fixed_points():
    """Swish(0) = 0, and Swish(scale) = scale exactly (saturated gate).

    The winning-branch gate of a select is exactly the mask value.
    """
    assert _swish(_cuda(0.0)).item() == 0.0
    assert _swish(_cuda(SCALE)).item() == SCALE


def test_compare_contract_points_bit_exact():
    """compare: the contract-point outputs are bit-exact +1/-1.

    Both hinges saturated or on-bend — mirrors the CPU pin.
    """

    def compare_out(z: float) -> float:
        # y = F + (T-F) * (hinge(z) - hinge(z-1)), T=+1, F=-1, fp32 throughout
        t = _cuda([z, z - 1.0])
        h = _swish(SCALE * t) / SCALE
        return (-1.0 + 2.0 * (h[0] - h[1])).item()

    assert compare_out(0.0) == -1.0  # x == thresh
    assert compare_out(1.0) == 1.0  # x == thresh + 1/sharpness


def test_abs_integer_grid_bit_exact():
    """abs: hinge(x) + hinge(-x) is bit-exact |x| on the whole integer grid.

    Sigmoid saturation on every point.
    """
    x = torch.arange(-1000.0, 1001.0, dtype=torch.float32, device="cuda")
    f = _swish(SCALE * x) / SCALE + _swish(-SCALE * x) / SCALE
    assert torch.equal(f, x.abs())


def test_onehot_winner_indicator_exact():
    """onehot_lookup: hinge(0.5) is exactly 0.5 (the winner indicator).

    hinge(-0.5) leaks at most ~1e-27 per row (e^-64 is representable,
    unlike sigma(-128); a kernel that flushes it to zero is fine too —
    the budget-relevant direction is the upper bound).
    """
    half = _cuda(SCALE * 0.5)
    assert (_swish(half) / SCALE).item() == 0.5
    assert (_swish(-half) / SCALE).abs().item() <= 1e-27


def test_bias_lane_constants_exact_unit_lane():
    """The no-bias constant lane (docs/no_bias_plan.md) on this kernel.

    The gate value saturates sigma bit-exactly, and the full lane
    expression — the GatedMLPSubLayer's ``g * sigmoid(g) * u`` —
    computes exactly 1.0 in fp32, so a constant routed through the
    lane's down-projection row lands verbatim in a ``bias=False``
    artifact. Mirrors the torch-CPU pin in ``test_swish_constants.py``.
    """
    from torchwright.ops.const import bias_lane_gate, bias_lane_up

    g = _cuda(bias_lane_gate)
    u = _cuda(bias_lane_up)
    assert torch.sigmoid(g).item() == 1.0
    assert (g * torch.sigmoid(g) * u).item() == 1.0
