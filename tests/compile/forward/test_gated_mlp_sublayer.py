"""Unit tests for the gated (SwiGLU) physical MLP sublayer — swiglu plan A1.

The gated sublayer is the swish machine's MLP:

    out = down_proj(swish(gate_proj(x)) * up_proj(x)) + x

with swish computed exactly as ``g * sigmoid(g)`` — the expression whose
fp32 saturation profile the A0 probes pinned.  These tests exercise the
module directly with hand-written weights; compiled-path coverage (weight
writer gated path, uniformity check) lands with plan steps A2/A3.
"""

import pytest
import torch

from torchwright.compiler.groups.mlp_sublayer import GatedMLPSubLayer, MLPSubLayer
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.compiler.transformer import HeadlessTransformer

D = 8
D_HIDDEN = 6
N_POS = 5


def _rand_gated(seed=0):
    """A gated sublayer with dense random weights in every tensor."""
    g = torch.Generator().manual_seed(seed)
    mlp = GatedMLPSubLayer(D, D_HIDDEN)
    mlp.gate_proj.output_matrix = torch.randn(D, D_HIDDEN, generator=g)
    mlp.gate_proj.output_bias = torch.randn(D_HIDDEN, generator=g)
    mlp.up_proj.output_matrix = torch.randn(D, D_HIDDEN, generator=g)
    mlp.up_proj.output_bias = torch.randn(D_HIDDEN, generator=g)
    mlp.down_proj.output_matrix = torch.randn(D_HIDDEN, D, generator=g)
    mlp.down_proj.output_bias = torch.randn(D, generator=g)
    return mlp


@pytest.fixture
def x():
    g = torch.Generator().manual_seed(42)
    return torch.randn(N_POS, D, generator=g)


def test_forward_matches_hand_swiglu(x):
    mlp = _rand_gated()
    got = mlp.forward(x)

    gate = x @ mlp.gate_proj.output_matrix + mlp.gate_proj.output_bias
    up = x @ mlp.up_proj.output_matrix + mlp.up_proj.output_bias
    hidden = gate * torch.sigmoid(gate) * up
    want = hidden @ mlp.down_proj.output_matrix + mlp.down_proj.output_bias + x

    assert torch.equal(got, want)


def test_degenerate_lane_reduces_to_bare_swish(x):
    """A lane written the degenerate way (up-row 0, up-bias 1) computes
    swish(gate) exactly — the up factor is the constant 1.
    """
    mlp = GatedMLPSubLayer(D, D_HIDDEN)
    g = torch.Generator().manual_seed(7)
    w = torch.randn(D, generator=g)
    mlp.gate_proj.output_matrix[:, 0] = w
    mlp.gate_proj.output_bias[0] = 0.25
    mlp.up_proj.output_bias[0] = 1.0  # degenerate lane: up ≡ 1
    mlp.down_proj.output_matrix[0, 3] = 1.0

    out = mlp.forward(x)
    # Same-shaped matmul as the module's own gate computation (a matvec
    # `x @ w` rounds differently at the bit level than the (n, d)@(d, dh)
    # matmul kernel).
    gate = (x @ mlp.gate_proj.output_matrix + mlp.gate_proj.output_bias)[:, 0]
    want = gate * torch.sigmoid(gate) + x[:, 3]
    assert torch.equal(out[:, 3], want)
    # Untouched columns pass through on the skip.
    other = [c for c in range(D) if c != 3]
    assert torch.equal(out[:, other], x[:, other])


def test_activation_attributes():
    assert MLPSubLayer(D).activation == "relu"
    assert GatedMLPSubLayer(D).activation == "swish"


def test_transformer_layer_selects_machine():
    assert isinstance(TransformerLayer(D, 2).mlp, MLPSubLayer)
    assert isinstance(TransformerLayer(D, 2, activation="relu").mlp, MLPSubLayer)
    assert isinstance(TransformerLayer(D, 2, activation="swish").mlp, GatedMLPSubLayer)
    with pytest.raises(ValueError, match="activation"):
        TransformerLayer(D, 2, activation="gelu")


def test_headless_transformer_threads_activation():
    net = HeadlessTransformer(D, 2, activation="swish")
    layer = net.add_layer(append=True)
    assert isinstance(layer.mlp, GatedMLPSubLayer)
    assert net.activation == "swish"

    default_net = HeadlessTransformer(D, 2)
    assert isinstance(default_net.add_layer(append=True).mlp, MLPSubLayer)
    assert default_net.activation == "relu"

    with pytest.raises(ValueError, match="activation"):
        HeadlessTransformer(D, 2, activation="gelu")


def test_trim_unused_slots_gated(x):
    """Trailing all-zero slots are removed; a bias-only slot (the written-
    degenerate-lane signature: up-bias 1, everything else zero) counts as
    used; the forward value is unchanged by the trim.
    """
    mlp = GatedMLPSubLayer(D, D_HIDDEN)
    g = torch.Generator().manual_seed(3)
    # Slot 0: fully-populated gated lane.
    mlp.gate_proj.output_matrix[:, 0] = torch.randn(D, generator=g)
    mlp.up_proj.output_matrix[:, 0] = torch.randn(D, generator=g)
    mlp.down_proj.output_matrix[0, :] = torch.randn(D, generator=g)
    # Slot 1: degenerate lane (gate row + up-bias 1 + down row).
    mlp.gate_proj.output_matrix[:, 1] = torch.randn(D, generator=g)
    mlp.up_proj.output_bias[1] = 1.0
    mlp.down_proj.output_matrix[1, :] = torch.randn(D, generator=g)
    # Slot 2: up-bias only.  Its lane value is swish(0)·1 = 0 and its down
    # row is zero, but trimming must still treat it as used (trailing-trim
    # correctness relies on usedness being write-coverage, not effect).
    mlp.up_proj.output_bias[2] = 1.0
    # Slots 3..5: never written.

    before = mlp.forward(x)
    mlp.trim_unused_slots()
    assert mlp.d_hidden == 3
    assert mlp.gate_proj.output_matrix.shape == (D, 3)
    assert mlp.gate_proj.output_bias.shape == (3,)
    assert mlp.up_proj.output_matrix.shape == (D, 3)
    assert mlp.up_proj.output_bias.shape == (3,)
    assert mlp.down_proj.output_matrix.shape == (3, D)
    # Not torch.equal: trimming changes the matmul shapes, and GPU kernel
    # selection rounds different-shaped matmuls differently at ~1 ulp even
    # though the removed slots are exactly zero (same reason the ONNX trim
    # parity test uses a tolerance).
    assert torch.allclose(mlp.forward(x), before, atol=1e-5)


def test_trim_keeps_one_slot_when_all_unused():
    mlp = GatedMLPSubLayer(D, D_HIDDEN)
    mlp.trim_unused_slots()
    assert mlp.d_hidden == 1
    assert mlp.gate_proj.output_matrix.shape == (D, 1)
    assert mlp.down_proj.output_matrix.shape == (1, D)


def test_num_params():
    mlp = GatedMLPSubLayer(D, D_HIDDEN)
    per_proj_up = D * D_HIDDEN + D_HIDDEN  # matrix + bias
    down = D_HIDDEN * D + D
    assert mlp.num_params() == 2 * per_proj_up + down
