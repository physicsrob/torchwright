"""Long-span regression for ``attend_most_recent_matching``.

Plan C guards the DOOM renderer's long recency reads.  The important
behavior is the compiled attention backend, not the small oracle
``Attn.compute`` path: at 32,768 positions the oracle would build a
large CPU attention matrix, while the compiled path uses the same SDPA
backend the real graph uses.

RoPE port: recency is no longer a host position counter — it is the
graph-derived octant ramp (``recency_rank``) built from two always-visible
marked tokens, ``<bos>`` at position 0 and ``<ref>`` at position 1 (see
``torchwright/ops/recency_heads.py``).  Those two positions are reserved
recency markers, so the content tests hold their content key at 0 there
(a non-match) and probe positions ``>= 2``.

Sizing notes (the recency machinery, not the old counter):
- ``D_HEAD = 32``.  At ``max_positions = 32_768`` the slowest usable recency
  plane at ``d_head = 16`` gives a per-token ramp step so small that the
  adjacent (gap-1) recency separation is only ~2.4 logits (~0.91 softmax
  concentration → up to ~0.086 blend of the neighbour's value), which would
  bust the 0.05 tolerance.  ``d_head = 32`` lands on a faster plane: ~5.4-logit
  adjacent gap (~0.995 concentration → ~0.005 blend), comfortably inside it.
- ``MATCH_GAIN_LONG = 1e6``.  Content must dominate recency: a content-matched
  but older key must beat a newer non-match, i.e.
  ``match_gain · dot_gap > rank_gain · rank_range``.  With the default
  ``rank_gain = 2e5`` and the ramp's ~1.5 interior range at this d_head
  (plus the degenerate BOS/REF outlier ranks), the bound is ~3e5; 1e6 buys
  ~3x margin.  (The old counter value 300_000 was below this bound and is
  retired.)
- ``MAX_LAYERS = 64``.  The octant ramp + two graded recency heads compile
  ~50 layers, far deeper than the old counter column's ~10.
"""

import pytest
import torch

from torchwright.compiler.device import get_device
from torchwright.compiler.forward.compile import forward_compile
from torchwright.ops.attention_ops import attend_most_recent_matching
from torchwright.ops.inout_nodes import create_input, create_rope_config
from torchwright.ops.recency_heads import recency_rank

MATCH_GAIN_LONG = 1_000_000.0
SPANS = (8_500, 32_768)
MAX_POSITIONS = max(SPANS)  # the recency plane is sized seam-safe to this span
D = 64
D_HEAD = 32
MAX_LAYERS = 64


@pytest.fixture(scope="module")
def compiled_long_span_pick():
    device = get_device(verbose=False)
    if device.type != "cuda":
        pytest.skip("long-span recency regression requires the CUDA backend")

    assert torch.get_default_dtype() == torch.float32
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cuda.matmul.allow_tf32

    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)
    query = create_input("query", 1, value_range=(0.0, 1.0))
    key = create_input("key", 1, value_range=(0.0, 1.0))
    value = create_input("value", 1, value_range=(0.0, float(max(SPANS))))
    # The two always-visible recency markers: <bos> at position 0, <ref> at 1.
    bos = create_input("bos_marker", 1, value_range=(0.0, 1.0))
    ref = create_input("ref_marker", 1, value_range=(0.0, 1.0))
    rank = recency_rank(bos, ref, d_head=D_HEAD, max_positions=MAX_POSITIONS)
    out = attend_most_recent_matching(
        rope,
        query,
        key,
        value,
        rank,
        match_gain=MATCH_GAIN_LONG,
    )

    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        max_layers=MAX_LAYERS,
    )
    assert net.device.type == "cuda"
    return net, out


def _run_case(compiled_long_span_pick, n_pos: int, key: torch.Tensor) -> torch.Tensor:
    net, out = compiled_long_span_pick
    bos = torch.zeros(n_pos, 1)
    ref = torch.zeros(n_pos, 1)
    bos[0, 0] = 1.0  # <bos> marks position 0
    ref[1, 0] = 1.0  # <ref> marks position 1
    inputs = {
        "query": torch.ones(n_pos, 1),
        "key": key,
        "value": torch.arange(float(n_pos)).unsqueeze(1),
        "bos_marker": bos,
        "ref_marker": ref,
    }
    return net.compute(n_pos, inputs)[out].detach().cpu().squeeze(1)


@pytest.mark.parametrize("n_pos", SPANS)
def test_long_span_dense_adjacent_matches_pick_most_recent(
    compiled_long_span_pick,
    n_pos: int,
) -> None:
    """Dense adjacent matches still resolve the one-position recency gap.

    Positions 0 and 1 are the reserved ``<bos>`` / ``<ref>`` recency markers,
    so their content key is held at 0 (a non-match); every position ``>= 2`` is
    a unit match and the head must pick the most recent (self) at each probe.
    """
    key = torch.ones(n_pos, 1)
    key[0, 0] = 0.0  # BOS: reserved recency marker, not a content match
    key[1, 0] = 0.0  # REF: reserved recency marker, not a content match
    result = _run_case(compiled_long_span_pick, n_pos, key)

    for probe in (n_pos // 2, n_pos - 1):
        assert result[probe].item() == pytest.approx(float(probe), abs=0.05)


@pytest.mark.parametrize("n_pos", SPANS)
def test_long_span_sparse_match_beats_recent_non_matches(
    compiled_long_span_pick,
    n_pos: int,
) -> None:
    """One old unit match beats newer non-matches — content dominates recency.

    ``match_pos = n_pos // 3`` sits well past the position-0/1 markers, which
    carry a zero content key (non-match) and so never win.
    """
    match_pos = n_pos // 3
    key = torch.zeros(n_pos, 1)
    key[match_pos, 0] = 1.0

    result = _run_case(compiled_long_span_pick, n_pos, key)

    for probe in (match_pos, match_pos + 1, n_pos // 2, n_pos - 1):
        assert result[probe].item() == pytest.approx(float(match_pos), abs=0.05)
