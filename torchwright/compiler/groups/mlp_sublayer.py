import torch

from torchwright.compiler.components.linear import LinearLayerComponent
from torchwright.compiler.components.relu import ReLULayerComponent
from torchwright.compiler.residual_assignment import ResidualStreamState


class MLPSubLayer:
    """MLP sublayer: linear1 -> relu -> linear2 + residual skip.

    Forward: out = linear2(relu(linear1(inp))) + inp

    ``d`` is the residual stream width; ``d_hidden`` is the MLP hidden
    width (the number of neurons / packed slots per layer).  They are
    independent — passing ``d_hidden=None`` defaults it to ``d``.
    """

    #: Machine kind (uniform per compiled network; see GatedMLPSubLayer).
    activation = "relu"

    def __init__(self, d: int, d_hidden: int | None = None) -> None:
        if d_hidden is None:
            d_hidden = d
        self.d = d
        self.d_hidden = d_hidden
        self.in_state = ResidualStreamState(name="MLPSubLayer In State")
        self.out_state = ResidualStreamState(name="MLPSubLayer Out State")
        self.linear1 = LinearLayerComponent(d, d_hidden, name="linear1")
        self.relu = ReLULayerComponent(d_hidden, name="relu")
        self.linear2 = LinearLayerComponent(d_hidden, d, name="linear2")

    def forward(self, inp: torch.Tensor, return_states=False):
        states = {}

        x = self.linear1.forward(inp)
        states["linear1_out_state"] = (self.linear1.out_state, x)
        x = self.relu.forward(x)
        states["relu_out_state"] = (self.relu.out_state, x)
        x = self.linear2.forward(x)
        states["linear2_out_state"] = (self.linear2.out_state, x)
        x = x + inp
        states["skip"] = (self.out_state, x)
        if return_states:
            return x, states
        return x

    def trim_unused_slots(self) -> None:
        """Remove trailing unused (all-zero) hidden slots after compilation.

        Slots are allocated contiguously from index 0, so slicing
        [:used] is safe.  Keeps at least 1 slot to avoid degenerate
        empty-tensor shapes.
        """
        # linear1 output columns and linear2 input rows correspond to slots.
        # A slot is unused if its linear1 column and linear2 row are both zero.
        l1_used = (self.linear1.output_matrix != 0).any(dim=0)  # (d_hidden,)
        l2_used = (self.linear2.output_matrix != 0).any(dim=1)  # (d_hidden,)
        used = l1_used | l2_used
        last_used = int(used.nonzero()[-1].item()) + 1 if used.any() else 1
        n = max(last_used, 1)
        if n < self.d_hidden:
            self.linear1.output_matrix = self.linear1.output_matrix[:, :n].contiguous()
            self.linear1.output_bias = self.linear1.output_bias[:n].contiguous()
            self.linear2.output_matrix = self.linear2.output_matrix[:n, :].contiguous()
            self.d_hidden = n
            self.relu.d = n

    def num_params(self):
        return self.linear1.num_params() + self.linear2.num_params()

    def to(self, device):
        self.linear1.to(device)
        self.relu.to(device)
        self.linear2.to(device)
        return self


class GatedMLPSubLayer:
    """Gated (SwiGLU) MLP sublayer: the swish machine's physical MLP.

    Forward: out = down_proj(swish(gate_proj(inp)) * up_proj(inp)) + inp

    where ``swish(g) = g * sigmoid(g)`` — written exactly that way (not a
    fused silu kernel) to match the expression whose fp32 saturation
    profile the A0 probes pinned (``tests/docs/test_swish_constants.py``,
    ``test_swish_saturation_cuda.py``) and the :class:`~torchwright.graph.ffn.FFN`
    oracle's ``compute``.

    Component naming follows the canonical SwiGLU reference (HF Llama):
    ``gate_proj`` / ``up_proj`` are ``d -> d_hidden``, ``down_proj`` is
    ``d_hidden -> d``.  Hidden-slot semantics mirror :class:`MLPSubLayer`:
    the scheduler packs FFN lanes into contiguous slots from index 0, one
    lane per slot; a slot's lane value is ``swish(gate) * up``.  Degenerate
    lanes (up ≡ 1) are written by the weight writer as up-row 0 / up-bias 1.
    """

    #: Machine kind (uniform per compiled network; see MLPSubLayer).
    activation = "swish"

    def __init__(self, d: int, d_hidden: int | None = None) -> None:
        if d_hidden is None:
            d_hidden = d
        self.d = d
        self.d_hidden = d_hidden
        self.in_state = ResidualStreamState(name="GatedMLPSubLayer In State")
        self.out_state = ResidualStreamState(name="GatedMLPSubLayer Out State")
        self.gate_proj = LinearLayerComponent(d, d_hidden, name="gate_proj")
        self.up_proj = LinearLayerComponent(d, d_hidden, name="up_proj")
        self.down_proj = LinearLayerComponent(d_hidden, d, name="down_proj")

    def forward(self, inp: torch.Tensor, return_states=False):
        states = {}

        g = self.gate_proj.forward(inp)
        states["gate_proj_out_state"] = (self.gate_proj.out_state, g)
        u = self.up_proj.forward(inp)
        states["up_proj_out_state"] = (self.up_proj.out_state, u)
        h = g * torch.sigmoid(g) * u
        x = self.down_proj.forward(h)
        states["down_proj_out_state"] = (self.down_proj.out_state, x)
        x = x + inp
        states["skip"] = (self.out_state, x)
        if return_states:
            return x, states
        return x

    def trim_unused_slots(self) -> None:
        """Remove trailing unused (all-zero) hidden slots after compilation.

        Slots are allocated contiguously from index 0, so slicing
        [:used] is safe.  Keeps at least 1 slot to avoid degenerate
        empty-tensor shapes.

        Unlike the ReLU trim, biases count toward usedness: a written
        degenerate lane carries up-bias 1 (its only up-side signature),
        and a constant lane's gate can be pure bias.  An unwritten slot
        is all-zero across all five tensors, so this never keeps a
        genuinely unused slot.
        """
        g_used = (self.gate_proj.output_matrix != 0).any(dim=0) | (
            self.gate_proj.output_bias != 0
        )
        u_used = (self.up_proj.output_matrix != 0).any(dim=0) | (
            self.up_proj.output_bias != 0
        )
        d_used = (self.down_proj.output_matrix != 0).any(dim=1)
        used = g_used | u_used | d_used
        last_used = int(used.nonzero()[-1].item()) + 1 if used.any() else 1
        n = max(last_used, 1)
        if n < self.d_hidden:
            self.gate_proj.output_matrix = self.gate_proj.output_matrix[
                :, :n
            ].contiguous()
            self.gate_proj.output_bias = self.gate_proj.output_bias[:n].contiguous()
            self.up_proj.output_matrix = self.up_proj.output_matrix[:, :n].contiguous()
            self.up_proj.output_bias = self.up_proj.output_bias[:n].contiguous()
            self.down_proj.output_matrix = self.down_proj.output_matrix[
                :n, :
            ].contiguous()
            self.d_hidden = n

    def num_params(self):
        return (
            self.gate_proj.num_params()
            + self.up_proj.num_params()
            + self.down_proj.num_params()
        )

    def to(self, device):
        self.gate_proj.to(device)
        self.up_proj.to(device)
        self.down_proj.to(device)
        return self
