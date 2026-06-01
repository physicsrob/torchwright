"""Long-span regression for ``attend_most_recent_matching``.

Plan C guards the DOOM renderer's long recency reads.  The important
behavior is the compiled attention backend, not the small oracle
``Attn.compute`` path: at 32,768 positions the oracle would build a
large CPU attention matrix, while the compiled path uses the same SDPA
backend the real graph uses.
"""

import pytest
import torch

from torchwright.compiler.device import get_device
from torchwright.compiler.forward.compile import forward_compile
from torchwright.ops.attention_ops import attend_most_recent_matching
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

MATCH_GAIN_LONG = 300_000.0
SPANS = (8_500, 32_768)
D = 64
D_HEAD = 16


@pytest.fixture(scope="module")
def compiled_long_span_pick():
    device = get_device(verbose=False)
    if device.type != "cuda":
        pytest.skip("long-span recency regression requires the CUDA backend")

    assert torch.get_default_dtype() == torch.float32
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cuda.matmul.allow_tf32

    pos = create_pos_encoding()
    query = create_input("query", 1, value_range=(0.0, 1.0))
    key = create_input("key", 1, value_range=(0.0, 1.0))
    value = create_input("value", 1, value_range=(0.0, float(max(SPANS))))
    out = attend_most_recent_matching(
        pos,
        query,
        key,
        value,
        match_gain=MATCH_GAIN_LONG,
    )

    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        pos_encoding=pos,
        verbose=False,
        max_layers=10,
    )
    assert net.device.type == "cuda"
    return net, out


def _run_case(compiled_long_span_pick, n_pos: int, key: torch.Tensor) -> torch.Tensor:
    net, out = compiled_long_span_pick
    inputs = {
        "query": torch.ones(n_pos, 1),
        "key": key,
        "value": torch.arange(float(n_pos)).unsqueeze(1),
    }
    return net.compute(n_pos, inputs)[out].detach().cpu().squeeze(1)


@pytest.mark.parametrize("n_pos", SPANS)
def test_long_span_dense_adjacent_matches_pick_most_recent(
    compiled_long_span_pick,
    n_pos: int,
) -> None:
    """Dense adjacent matches still resolve the one-position recency gap."""
    result = _run_case(compiled_long_span_pick, n_pos, torch.ones(n_pos, 1))

    for probe in (n_pos // 2, n_pos - 1):
        assert result[probe].item() == pytest.approx(float(probe), abs=0.05)


@pytest.mark.parametrize("n_pos", SPANS)
def test_long_span_sparse_match_beats_recent_non_matches(
    compiled_long_span_pick,
    n_pos: int,
) -> None:
    """At ``300_000`` gain, one old unit match beats newer non-matches."""
    match_pos = n_pos // 3
    key = torch.zeros(n_pos, 1)
    key[match_pos, 0] = 1.0

    result = _run_case(compiled_long_span_pick, n_pos, key)

    for probe in (match_pos, match_pos + 1, n_pos // 2, n_pos - 1):
        assert result[probe].item() == pytest.approx(float(match_pos), abs=0.05)
