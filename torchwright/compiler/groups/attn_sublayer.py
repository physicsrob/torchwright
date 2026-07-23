import torch

from torchwright.compiler.components.attn import AttnLayerComponent
from torchwright.compiler.residual_assignment import ResidualStreamState


class AttnSubLayer:
    """Attention sublayer: multi-head attention + residual skip connection.

    Forward: out = attn(inp) + inp
    """

    def __init__(
        self,
        d: int,
        d_head: int,
        n_heads: int | None = None,
    ) -> None:
        self.d = d
        self.in_state = ResidualStreamState(name="AttnSubLayer In State")
        self.out_state = ResidualStreamState(name="AttnSubLayer Out State")
        self.attn = AttnLayerComponent(d, d_head, name="attn", n_heads=n_heads)

    def forward(self, inp: torch.Tensor, return_states=False):
        states = {}

        x = self.attn.forward(inp)
        states["attn_out_state"] = (self.attn.out_state, x)
        x = x + inp
        states["skip_out_state"] = (self.out_state, x)
        if return_states:
            return x, states
        return x

    def forward_cached(self, inp, past_kv=None):
        x, new_kv = self.attn.forward_cached(inp, past_kv)
        x = x + inp
        return x, new_kv

    def num_params(self):
        return self.attn.num_params()

    def to(self, device):
        self.attn.to(device)
        return self

    def resize(self, new_d) -> None:
        self.d = new_d
        self.attn.resize(new_d)
