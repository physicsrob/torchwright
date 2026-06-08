"""Tests for the token streaming cached ONNX exporter.

Parity oracle: HeadlessTransformer.compute() (the reference per-node
evaluator) against compile_to_onnx + onnxruntime.  These tests catch
bugs in layer emission, causal mask construction, cached protocol
wiring, and the token embedding / unembed paths.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from examples.adder import create_network_parts
from torchwright.compiler.export import (
    TOKEN_META_FORMAT,
    meta_path_for,
    compile_to_onnx,
)
from torchwright.compiler.forward.compile import forward_compile

onnxruntime = pytest.importorskip("onnxruntime")

D = 1024
D_HEAD = 16


def _build_1digit():
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        output_node, pos_encoding, embedding = create_network_parts()
    finally:
        adder_module.max_digits = original
    return output_node, pos_encoding, embedding


def _empty_past_feeds(per_layer_n_heads: list, d_head: int) -> dict:
    feeds = {"past_len": np.array(0, dtype=np.int64)}
    for i, nh in enumerate(per_layer_n_heads):
        feeds[f"past_K_{i}"] = np.zeros((0, nh, d_head), dtype=np.float32)
        feeds[f"past_V_{i}"] = np.zeros((0, nh, d_head), dtype=np.float32)
    return feeds


def _discover_meta(session):
    # past_K_i is sequence-major (n_past, n_heads, d_head): heads on axis 1.
    inputs = {inp.name: inp for inp in session.get_inputs()}
    n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
    per_layer_n_heads = [int(inputs[f"past_K_{i}"].shape[1]) for i in range(n_layers)]
    d_head = int(inputs["past_K_0"].shape[2])
    return n_layers, per_layer_n_heads, d_head


def _reference_logits(output_node, pos_encoding, embedding, tokens):
    """Run compute() and return full logits (seq_len, vocab_size).

    Applies the same ``out_emb @ embedding.table.T`` unembed as the ONNX
    graph, so the result is numerically comparable to the ONNX output
    via ``np.allclose``.
    """
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )
    result = net.compute(
        n_pos=len(tokens),
        input_values={"embedding_input": tokens},
    )
    out_emb = result[output_node].cpu()  # (seq_len, d_embed)
    return (out_emb @ embedding.table.T).numpy()


# ---------------------------------------------------------------------------
# Test 1: Prefill argmax matches compute() reference
# ---------------------------------------------------------------------------


def test_token_onnx_prefill_matches_compute():
    output_node, pos_encoding, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
            pos_encoding,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )

        tokens = ["<bos", "1", "+", "2", "\n"]
        ref_logits = _reference_logits(output_node, pos_encoding, embedding, tokens)

        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head = _discover_meta(session)
        token_ids = np.array(
            [embedding.tokenizer.get_token_id(t) for t in tokens],
            dtype=np.int64,
        )
        feeds = {"token_ids": token_ids}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        onnx_logits = session.run(["logits"], feeds)[0]

        assert np.allclose(
            ref_logits, onnx_logits, atol=1e-4
        ), f"logits max diff: {np.abs(ref_logits - onnx_logits).max():.6f}"


# ---------------------------------------------------------------------------
# Test 2: Decode step matches full prefill (catches dynamic-mask seam bugs)
# ---------------------------------------------------------------------------


def test_token_onnx_decode_step_matches_full_prefill():
    output_node, pos_encoding, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
            pos_encoding,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )

        tokens = ["<bos", "1", "+", "2", "\n"]
        token_ids = np.array(
            [embedding.tokenizer.get_token_id(t) for t in tokens],
            dtype=np.int64,
        )

        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head = _discover_meta(session)
        out_names = ["logits"]
        for i in range(n_layers):
            out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        feeds = {"token_ids": token_ids}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        full_logits = session.run(["logits"], feeds)[0]

        feeds = {"token_ids": token_ids[:-1]}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        outputs = session.run(out_names, feeds)
        past_K = [outputs[1 + 2 * i] for i in range(n_layers)]
        past_V = [outputs[1 + 2 * i + 1] for i in range(n_layers)]

        feeds = {
            "token_ids": token_ids[-1:],
            "past_len": np.array(len(tokens) - 1, dtype=np.int64),
        }
        for i in range(n_layers):
            feeds[f"past_K_{i}"] = past_K[i]
            feeds[f"past_V_{i}"] = past_V[i]
        decode_logits = session.run(["logits"], feeds)[0]

        assert np.allclose(full_logits[-1], decode_logits[0], atol=1e-4), (
            f"decode vs full max diff: "
            f"{np.abs(full_logits[-1] - decode_logits[0]).max():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Autoregressive generation via REPL (1-digit adder)
# ---------------------------------------------------------------------------


def test_token_onnx_autoregressive_1digit():
    from torchwright.compiler.repl import _load, generate

    output_node, pos_encoding, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
            pos_encoding,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )

        model = _load(onnx_path)
        test_cases = [("1+1\n", "2"), ("2+3\n", "5"), ("4+5\n", "9")]
        for input_str, expected in test_cases:
            result = "".join(generate(model, input_str))
            assert (
                result == expected
            ), f"{input_str}: expected {expected!r}, got {result!r}"


# ---------------------------------------------------------------------------
# Test 4: Autoregressive generation via REPL (3-digit adder)
# ---------------------------------------------------------------------------


def test_token_onnx_autoregressive_3digit():
    from torchwright.compiler.repl import _load, generate

    output_node, pos_encoding, embedding = create_network_parts()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder3.onnx")
        compile_to_onnx(
            output_node,
            pos_encoding,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )

        model = _load(onnx_path)
        test_cases = [
            ("1+2\n", "3"),
            ("12+34\n", "46"),
            ("99+1\n", "100"),
            ("100+200\n", "300"),
            ("456+123\n", "579"),
        ]
        for input_str, expected in test_cases:
            result = "".join(generate(model, input_str))
            assert (
                result == expected
            ), f"{input_str}: expected {expected!r}, got {result!r}"


# ---------------------------------------------------------------------------
# Test 5: Sidecar schema + repl metadata discovery
# ---------------------------------------------------------------------------


def test_token_onnx_sidecar_schema_and_metadata():
    from torchwright.compiler.repl import _load

    output_node, pos_encoding, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
            pos_encoding,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )

        meta_path = meta_path_for(onnx_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["format"] == TOKEN_META_FORMAT
        assert meta["vocab"] == embedding.tokenizer.vocab

        model = _load(onnx_path)
        assert model.n_layers > 0
        assert len(model.per_layer_n_heads) == model.n_layers
        assert all(nh <= D // D_HEAD for nh in model.per_layer_n_heads)
        assert model.d_head == D_HEAD
