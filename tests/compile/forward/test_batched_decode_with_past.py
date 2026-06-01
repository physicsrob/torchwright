"""Batched decode with non-empty past — the speculative-decoding shape.

When ``forward_cached`` is called with ``n_new > 1`` and a non-empty past
(e.g. K+1 batched rows in a spec-decode step), each new row must:

1. Attend unconditionally to every past key — those are verified history.
2. Attend causally among new rows — row ``i`` sees new rows ``0..i`` only.

A bug in either direction silently corrupts the rollout. This test pins
both: drive a sequential per-row rollout to length L, then run one
batched ``K+1``-row decode from past_len=L and check row-by-row that each
batched-row output equals the corresponding sequential-step output.
"""

import pytest
import torch

from torchwright.compiler.forward.compile import forward_compile

D = 1024
D_HEAD = 16


@pytest.fixture(scope="module")
def compiled_calc():
    from examples.calculator_v2 import create_network_parts

    output_node, pos_encoding, embedding = create_network_parts(1)
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )
    return net, output_node, embedding


def _row_input(net, tokens, t):
    """Build the t-th row of the input residual stream for ``tokens``."""
    full_res = net.get_input_res_stream(t + 1, {"embedding_input": tokens[: t + 1]})
    return full_res[t : t + 1].to(net.device)


def test_batched_decode_with_past_matches_sequential(compiled_calc):
    """K+1 batched rows from past_len=L equal L+1..L+K+1 sequential rows."""
    net, _, _ = compiled_calc
    tokens = ["<bos"] + list("3+5+9\n")
    seed_len = 3  # past_len at the start of the batched step
    batch_size = len(tokens) - seed_len  # K+1 rows in the batched step
    assert batch_size >= 2, "test requires n_new >= 2 to exercise the new mask"

    # Stage 1: sequential rollout up to seed_len, snapshot the past.
    kvs = None
    for t in range(seed_len):
        kvs, _ = _step_one(net, tokens, t, kvs)
    past_kvs_seed = [(K.clone(), V.clone()) for (K, V) in kvs]

    # Stage 2a: continue sequentially to capture per-row reference outputs.
    seq_outputs = []
    kvs_seq = [(K.clone(), V.clone()) for (K, V) in past_kvs_seed]
    for t in range(seed_len, len(tokens)):
        kvs_seq, out = _step_one(net, tokens, t, kvs_seq)
        seq_outputs.append(out.detach().clone())

    # Stage 2b: run the same suffix as one batched call from the snapshot.
    suffix_inp = torch.cat(
        [_row_input(net, tokens, t) for t in range(seed_len, len(tokens))],
        dim=0,
    )
    batched_out, _ = net.forward_cached(suffix_inp, past_kvs_seed)

    # Bit-exact row-by-row equality. SDPA forces the MATH backend (see
    # _SDPA_BACKEND in components/attn.py), and the matmul shapes for a
    # given row are identical in the two paths once the mask is correct.
    for i, expected in enumerate(seq_outputs):
        actual = batched_out[i : i + 1]
        diff = (actual - expected).abs().max().item()
        assert diff == 0.0, (
            f"row {i}: sequential vs batched diverged by {diff}; "
            f"expected bit-exact equality"
        )


def _step_one(net, tokens, t, kvs):
    """Run one cached single-row forward and return (new_kvs, output)."""
    new_inp = _row_input(net, tokens, t)
    out, new_kvs = net.forward_cached(new_inp, kvs)
    return new_kvs, out
