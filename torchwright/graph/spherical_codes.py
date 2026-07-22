import torch

from torchwright.graph.e8_codes import CODES

spherical_codes = 10.0 * torch.tensor(CODES, dtype=torch.float32)


def get_spherical_codes(d: int, max_index: int = 1024):
    assert d == 8
    assert max_index == 1024, "Only support max_index = 1024"
    return spherical_codes


def index_to_vector(index: int, max_index: int = 1024):
    assert max_index == 1024, "Only support 1024 currently"
    return spherical_codes[index]
