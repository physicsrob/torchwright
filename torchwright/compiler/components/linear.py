import torch

from torchwright.compiler.components.component import Component


class LinearLayerComponent(Component):
    """Linear projection component.

    Weight matrices:
        output_matrix: (d_in, d_out)
        output_bias:   (d_out,)

    Forward: out = inp @ output_matrix + output_bias
    """

    def __init__(self, d_in: int, d_out: int, name: str = "") -> None:
        super().__init__(d_out, name)
        self.d_in = d_in
        self.d_out = d_out
        self.output_matrix = torch.zeros(d_in, d_out)
        self.output_bias = torch.zeros(d_out)

    def __repr__(self) -> str:
        return f"LinearLayerComponent(name='{self.name}')"

    def forward(self, inp: torch.Tensor):
        # inp shape (n_pos, d_in) -> (n_pos, d_out)
        x = inp @ self.output_matrix
        return x + self.output_bias

    def num_params(self) -> int:
        return self.output_matrix.numel() + self.output_bias.numel()
