"""Tests for KV cache correctness.

Verifies that forward_cached produces identical results to forward,
both in single-pass (prefill) and token-by-token (autoregressive) modes.
"""

import pytest
import torch

from examples.calculator_simple import D_HEAD
from torchwright.compiler.forward.compile import forward_compile

D = 1024


@pytest.fixture(scope="module")
def compiled_calc():
    from examples.calculator_simple import create_network_parts

    output_node, embedding = create_network_parts(1)
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        verbose=False,
    )
    return net, output_node, embedding


def test_prefill_matches_forward(compiled_calc):
    """forward_cached with no past KV should match forward exactly."""
    net, _output_node, _embedding = compiled_calc
    tokens = ["<bos>", *list("3+5\n")]
    res_stream = net.get_input_res_stream(len(tokens), {"embedding_input": tokens})
    inp = res_stream.to(net.device)

    expected = net.forward(inp)
    actual, kvs = net.forward_cached(inp)

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"max diff: {(expected - actual).abs().max().item()}"
    )
    assert len(kvs) == len(net.layers)


def test_token_by_token_matches_full(compiled_calc):
    """Token-by-token cached generation should match single-pass forward."""
    net, _output_node, _embedding = compiled_calc
    tokens = ["<bos>", *list("7+8\n")]
    n_pos = len(tokens)

    # Full forward pass on entire sequence
    full_res_stream = net.get_input_res_stream(n_pos, {"embedding_input": tokens})
    expected = net.forward(full_res_stream.to(net.device))

    # Token-by-token with KV cache
    kvs = None
    for t in range(n_pos):
        full_res = net.get_input_res_stream(t + 1, {"embedding_input": tokens[: t + 1]})
        new_inp = full_res[t : t + 1].to(net.device)
        res, kvs = net.forward_cached(new_inp, kvs)

    # The last token's output should match
    assert torch.allclose(expected[-1:], res, atol=1e-4), (
        f"max diff: {(expected[-1:] - res).abs().max().item()}"
    )
