import torch

from torchwright.compiler.components.component import Component


class ReLULayerComponent(Component):
    """ReLU activation component. No parameters."""

    def __init__(self, d_hidden: int, name: str = "") -> None:
        super().__init__(d_hidden, name)
        self.d_hidden = d_hidden

    def __repr__(self) -> str:
        return f"ReLULayerComponent(name='{self.name}')"

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.clamp(inp, min=0)

    def num_params(self) -> int:
        return 0
