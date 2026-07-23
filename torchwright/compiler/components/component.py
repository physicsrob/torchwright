from abc import ABC, abstractmethod

import torch

from torchwright.compiler.residual_assignment import ResidualStreamState


class Component(ABC):
    """Base class for transformer layer components.

    Holds weight tensors and implements the forward pass.
    """

    d: int
    in_state: ResidualStreamState
    out_state: ResidualStreamState
    name: str

    def __init__(self, d: int, name: str = "") -> None:
        self.d = d
        self.name = name
        self.in_state = ResidualStreamState(name=f"{self} in_state")
        self.out_state = ResidualStreamState(name=f"{self} out_state")

    @abstractmethod
    def forward(self, inp: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def num_params(self) -> int: ...

    def resize(self, new_d: int) -> None:
        self.d = new_d

    def to(self, device: torch.device | str) -> "Component":
        """Move all tensor attributes to the given device."""
        for attr_name in list(vars(self)):
            val = getattr(self, attr_name)
            if isinstance(val, torch.Tensor):
                setattr(self, attr_name, val.to(device))
        return self
