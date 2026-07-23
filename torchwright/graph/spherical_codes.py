import torch

from torchwright.graph.e8_codes import CODES

#: The E8 lattice's dimension — the only ``d`` this spherical code table supports.
_E8_DIM = 8

#: Number of codewords in the E8 spherical code table.
_E8_CODE_COUNT = 1024

spherical_codes = 10.0 * torch.tensor(CODES, dtype=torch.float32)


def get_spherical_codes(d: int, max_index: int = _E8_CODE_COUNT) -> torch.Tensor:
    assert d == _E8_DIM
    assert max_index == _E8_CODE_COUNT, f"Only support max_index = {_E8_CODE_COUNT}"
    return spherical_codes


def index_to_vector(index: int, max_index: int = _E8_CODE_COUNT) -> torch.Tensor:
    assert max_index == _E8_CODE_COUNT, f"Only support {_E8_CODE_COUNT} currently"
    return spherical_codes[index]
