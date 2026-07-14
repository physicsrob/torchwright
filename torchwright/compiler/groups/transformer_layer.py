from typing import Optional

from torchwright.compiler.groups.attn_sublayer import AttnSubLayer
from torchwright.compiler.groups.mlp_sublayer import GatedMLPSubLayer, MLPSubLayer


class TransformerLayer:
    attn: AttnSubLayer

    def __init__(
        self,
        d: int,
        d_head: int,
        d_hidden: Optional[int] = None,
        activation: str = "relu",
        n_heads: Optional[int] = None,
    ):
        # The machine kind is uniform per network (all-ReLU or all-swish);
        # the compiler selects it from the graph's FFN nodes and threads it
        # here via HeadlessTransformer.add_layer.
        if activation == "relu":
            self.mlp = MLPSubLayer(d, d_hidden)
        elif activation == "swish":
            self.mlp = GatedMLPSubLayer(d, d_hidden)
        else:
            raise ValueError(
                f"TransformerLayer activation must be 'relu' or 'swish', "
                f"got {activation!r}"
            )
        self.attn = AttnSubLayer(d, d_head, n_heads=n_heads)

    def to(self, device):
        self.attn.to(device)
        self.mlp.to(device)
        return self

    def num_params(self):
        return self.attn.num_params() + self.mlp.num_params()
