"""Pins the onnxruntime-CPU sigmoid saturation profile — the oracle kernel.

``OnnxTokenModule`` (``compiler/onnx_load.py``) is the contract-
correctness harness: artifact parity tests run the exported model under
CPU onnxruntime.  A0 of ``docs/swiglu_step2_plan.md`` measured that
kernel and found it does NOT match torch's saturation profile (pinned in
``test_swish_constants.py``):

- torch fp32 sigmoid is exactly 1.0 from z >= 17; ORT-CPU reaches exact
  1.0 only from z >= 18, sitting up to ~1.8e-7 BELOW 1.0 on parts of
  [17, 18).
- torch returns representable denormals down to z ~ -103; ORT-CPU is
  exactly 0.0 for every z <= -18 (and intermittently already in
  (-18, -15.8]).

Consequences the swiglu op designs must respect: any bit-exactness
claim that needs sigmoid saturation must place its hinge argument >= 18
past the bend (not 17) to be exact on BOTH kernels, and near-bend tail
values differ between kernels by up to ~2e-7 absolute — noise-level,
inside every budget, but fatal to exact-equality comparisons between a
torch oracle and an ORT artifact in the [15.8, 18] band.  The claims
the spec actually leans on (sigma(-100) = 0 dead branch, Swish(0) = 0,
Swish(100) = 100 winning gate, Swish(+-50) onehot indicator) hold on
this kernel — the onehot leak is even exactly zero here, where torch
leaks ~1e-22.

This file pins the MEASURED ORT-CPU profile so a runtime upgrade that
shifts it fails loudly.  The torch-CUDA and ORT-CUDA halves of the A0
probe live in ``test_swish_saturation_cuda.py`` and torchwright_doom's
``tests/inference/test_ort_cuda_saturation.py``.

Runs wherever CPU onnxruntime is importable: the Modal test image
(test-onnx group) and the local workspace venv (the GPU build's CPU
execution provider — same kernel source).
"""

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

#: The module hinge-sharpening constant (test_swish_constants.SCALE).
SCALE = 100.0


def _run_sigmoid_swish(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sig = Sigmoid(z), swish = Mul(z, sig) under CPU onnxruntime,
    default session options.  Opset 14, matching the exporter."""
    n = len(z)
    graph = helper.make_graph(
        [
            helper.make_node("Sigmoid", ["z"], ["sig"]),
            helper.make_node("Mul", ["z", "sig"], ["swish"]),
        ],
        "ort_cpu_saturation_probe",
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [n])],
        [
            helper.make_tensor_value_info("sig", TensorProto.FLOAT, [n]),
            helper.make_tensor_value_info("swish", TensorProto.FLOAT, [n]),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=10
    )
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sig, swish = sess.run(["sig", "swish"], {"z": z.astype(np.float32)})
    return sig, swish


def test_positive_saturation_threshold_is_18_not_17():
    """The divergence from torch: sigmoid(17) sits one ulp below 1.0;
    exact 1.0 holds from 18 up, with at most ~1.8e-7 shortfall on
    [17, 18)."""
    z = np.linspace(17.0, 200.0, 100_001, dtype=np.float32)
    sig, _ = _run_sigmoid_swish(z)

    shortfall = 1.0 - float(sig[0])  # z = 17.0
    assert 0.0 < shortfall <= 3e-7, f"sigmoid(17): 1-sig = {shortfall:.3e}"

    band = sig[z < 18.0]
    worst = (1.0 - band).max()
    assert worst <= 3e-7, f"worst 1-sig on [17, 18) = {worst:.3e}"

    above = sig[z >= 18.0]
    bad = z[z >= 18.0][above != 1.0]
    assert bad.size == 0, (
        f"sigmoid != 1.0 at {bad.size} points in [18, 200]; "
        f"largest offender z={bad.max():.6f}"
    )


def test_negative_saturation_exact_zero_from_minus_18():
    """ORT-CPU sigmoid is exactly 0.0 for every z <= -18 (torch keeps
    denormals down to ~-103).  Strictly safer for the spec's leak
    claims: sigma(-100) = 0 (dead select branch) is included."""
    z = np.linspace(-200.0, -18.0, 100_001, dtype=np.float32)
    sig, swish = _run_sigmoid_swish(z)
    bad = z[sig != 0.0]
    assert bad.size == 0, (
        f"sigmoid != 0.0 at {bad.size} points in [-200, -18]; "
        f"largest offender z={bad.max():.6f}, value={sig[sig != 0.0].max():.3e}"
    )
    assert (swish == 0.0).all()  # dead-branch gate contributes bit-zero

    # ... and the cutoff is genuinely near -18, not far below it.
    sig15, _ = _run_sigmoid_swish(np.full(4, -15.0, dtype=np.float32))
    assert sig15[0] > 0.0


def test_swish_fixed_points_and_onehot_indicator():
    """The claims the spec leans on hold on this kernel: Swish(0) = 0,
    Swish(100) = 100 (saturated winning gate), Swish(50) = 50 (onehot
    winner indicator = exactly 0.5 after /scale), and the onehot leak
    at -0.5 is exactly zero (sigma(-50) = 0 here, unlike torch)."""
    _, swish = _run_sigmoid_swish(np.array([0.0, SCALE, 50.0, -50.0], dtype=np.float32))
    assert swish[0] == 0.0
    assert swish[1] == SCALE
    assert swish[2] == 50.0
    assert swish[3] == 0.0
