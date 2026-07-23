"""IndexedRegion: constant-depth random access into a marker-delimited token run.

Reference-evaluation tests (no compile, no GPU) over a real token stream —
``<bos>`` as the marker, digits before ``+`` as the region, mirroring the
calculator parse this primitive was built for.  The cases pin the three
contract-critical behaviors:

* an in-range index reads exactly the region member at that index;
* an out-of-range index (negative, or at/past the region's actual length)
  reads the region default — including the trap where the *stream* has a
  token at the queried marker distance but the region does not (the position
  right after the region, or a lookalike digit from a later run, must not
  leak through);
* ``length_at`` latches the region length at the event position and holds it
  for the rest of the stream.

The rope is built at the production ``max_positions=512``: at a degenerate
recency-lobe band (e.g. 64) ``get_prev_value`` refuses to build the latch
(``tests/ops/test_local_recency.py`` pins that guard) — see the note on
``IndexedRegion``'s rope requirement.
"""

import pytest
import torch

from torchwright.graph.embedding import bos_token
from torchwright.ops.attention_ops import get_prev_value
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_onehot_embedding,
    create_rope_config,
)
from torchwright.ops.linear import add_const
from torchwright.ops.swiglu import IndexedRegion, check_is_digit
from torchwright.ops.swiglu.logic_ops import bool_all_true, bool_not, equals_vector

VOCAB = [str(d) for d in range(10)] + ["+", "\n", bos_token, "<eos>"]
MAX_LEN = 3


@pytest.fixture
def setup():
    embedding = create_onehot_embedding(vocab=VOCAB)
    rope = create_rope_config(d_head=32, max_positions=512)
    embed = embedding.get_embedding
    saw_bos = equals_vector(embedding, embed(bos_token))
    is_plus = equals_vector(embedding, embed("+"))
    seen_plus = get_prev_value(rope, is_plus, is_plus)
    in_region = bool_all_true([check_is_digit(embedding), bool_not(seen_plus)])
    region = IndexedRegion(
        rope,
        embedding,
        marker=saw_bos,
        in_region=in_region,
        max_len=MAX_LEN,
        max_read_distance=16,
        default=embed("0"),
    )
    return embedding, rope, region, is_plus


def _decode_last(embedding, node, tokens):
    """Evaluate ``node`` over the stream and argmax-decode its last position."""
    value = node.compute(n_pos=len(tokens), input_values={"embedding_input": tokens})
    return embedding.tokenizer.vocab[int(value[-1].argmax())]


def _idx(i):
    return create_literal_value(torch.tensor([float(i)]))


def test_in_range_reads_each_member(setup):
    embedding, _, region, _ = setup
    tokens = [bos_token, "7", "4", "2", "+", "9", "\n"]
    for i, expected in enumerate(["7", "4", "2"]):
        assert _decode_last(embedding, region.token_at(_idx(i)), tokens) == expected, i


def test_negative_index_reads_default(setup):
    embedding, _, region, _ = setup
    tokens = [bos_token, "7", "4", "2", "+", "9", "\n"]
    assert _decode_last(embedding, region.token_at(_idx(-1)), tokens) == "0"


def test_index_past_length_reads_default(setup):
    # The stream HAS tokens at marker distances 2 and 3 (the "+" and a digit
    # right after it), but the region is only one member long: index 1 and 2
    # must fall to the default via the sentinel, not leak the lookalike
    # digits whose marker distance lands in [0, max_len).
    embedding, _, region, _ = setup
    tokens = [bos_token, "1", "+", "2", "3", "\n"]
    assert _decode_last(embedding, region.token_at(_idx(0)), tokens) == "1"
    for i in (1, 2):
        assert _decode_last(embedding, region.token_at(_idx(i)), tokens) == "0", i


def test_length_at_latches_and_holds(setup):
    _embedding, _, region, is_plus = setup
    length = region.length_at(is_plus)
    for tokens, expected in [
        ([bos_token, "1", "+", "2", "3", "\n"], 1.0),
        ([bos_token, "7", "4", "+", "9", "\n"], 2.0),
        ([bos_token, "7", "4", "2", "+", "9", "\n"], 3.0),
    ]:
        value = length.compute(
            n_pos=len(tokens), input_values={"embedding_input": tokens}
        )
        # Held at every position from the event to the end of the stream.
        event_pos = tokens.index("+")
        for p in range(event_pos, len(tokens)):
            assert value[p, 0].item() == pytest.approx(expected, abs=0.3), (
                tokens,
                p,
            )


def test_msb_padded_window(setup):
    # The calculator-parse composition: MSB-first fixed-width window with
    # implicit zero pads, index = length - width + i.
    embedding, _, region, is_plus = setup
    length = region.length_at(is_plus)
    tokens = [bos_token, "4", "2", "+", "9", "\n"]
    window = [
        region.token_at(add_const(length, float(i - MAX_LEN))) for i in range(MAX_LEN)
    ]
    got = "".join(_decode_last(embedding, node, tokens) for node in window)
    assert got == "042"


def test_wide_region_raises_past_dominance_bound():
    # Codex finding (2026-07): a wide region's content lanes reach fast rotary
    # planes, whose cos(Δ·θ) attenuation breaks the member/sentinel/zero
    # dominance ordering — a 14-digit operand mis-parsed at d_head=32 even
    # though max_len "fit the slow planes" per the old docstring.  The
    # constructor must refuse the geometry, not build a silently-wrong gather.
    embedding = create_onehot_embedding(vocab=VOCAB)
    rope = create_rope_config(d_head=32, max_positions=512)
    embed = embedding.get_embedding
    saw_bos = equals_vector(embedding, embed(bos_token))
    in_region = check_is_digit(embedding)
    with pytest.raises(ValueError, match="dominance ordering"):
        IndexedRegion(
            rope,
            embedding,
            marker=saw_bos,
            in_region=in_region,
            max_len=14,
            max_read_distance=2 * 14 + 4,
            default=embed("0"),
        )


def test_wide_region_builds_at_wider_d_head():
    # The same 14-lane region is healthy at d_head=64: every content lane
    # stays on a quasi-static plane over the consumed read distance.
    embedding = create_onehot_embedding(vocab=VOCAB)
    rope = create_rope_config(d_head=64, max_positions=512)
    embed = embedding.get_embedding
    saw_bos = equals_vector(embedding, embed(bos_token))
    in_region = check_is_digit(embedding)
    IndexedRegion(  # must not raise
        rope,
        embedding,
        marker=saw_bos,
        in_region=in_region,
        max_len=14,
        max_read_distance=2 * 14 + 4,
        default=embed("0"),
    )
