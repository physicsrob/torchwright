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


def _discover_meta(session, onnx_path):
    # past_K_i is sequence-major (cache_slots, n_heads, d_head) with a
    # SYMBOLIC slot dim (stride bucketing); the full stride S comes from
    # the sidecar meta.
    import json

    from torchwright.compiler.export import meta_path_for
    from torchwright.compiler.onnx_load import discover_cache_stride

    inputs = {inp.name: inp for inp in session.get_inputs()}
    n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
    per_layer_n_heads = [int(inputs[f"past_K_{i}"].shape[1]) for i in range(n_layers)]
    d_head = int(inputs["past_K_0"].shape[2])
    slot_dim = inputs["past_K_0"].shape[0]
    assert not isinstance(
        slot_dim, int
    ), f"past_K_0 first dim must be the symbolic cache_slots, got {slot_dim!r}"
    with open(meta_path_for(onnx_path)) as f:
        sidecar = json.load(f)
    cache_stride = discover_cache_stride(inputs, sidecar.get("cache_stride"), onnx_path)
    return n_layers, per_layer_n_heads, d_head, cache_stride


def _zero_past(per_layer_n_heads: list, d_head: int, S: int):
    """Full static-S zero-filled cache buffers, one (k, v) list pair."""
    k = [np.zeros((S, nh, d_head), dtype=np.float32) for nh in per_layer_n_heads]
    v = [np.zeros((S, nh, d_head), dtype=np.float32) for nh in per_layer_n_heads]
    return k, v


def _feeds(token_ids: np.ndarray, past_k, past_v, base: int) -> dict:
    """Static-cache feeds: full-S past buffers + cache_position for the rows."""
    n_new = int(token_ids.shape[0])
    feeds = {
        "token_ids": token_ids,
        "cache_position": np.arange(base, base + n_new, dtype=np.int64),
    }
    for i, (k, v) in enumerate(zip(past_k, past_v)):
        feeds[f"past_K_{i}"] = k
        feeds[f"past_V_{i}"] = v
    return feeds


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

        tokens = ["<bos>", "1", "+", "2", "\n"]
        ref_logits = _reference_logits(output_node, pos_encoding, embedding, tokens)

        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)
        token_ids = np.array(
            [embedding.tokenizer.get_token_id(t) for t in tokens],
            dtype=np.int64,
        )
        past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
        onnx_logits = session.run(["logits"], _feeds(token_ids, past_k, past_v, 0))[0]

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

        tokens = ["<bos>", "1", "+", "2", "\n"]
        token_ids = np.array(
            [embedding.tokenizer.get_token_id(t) for t in tokens],
            dtype=np.int64,
        )

        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)
        out_names = ["logits"]
        for i in range(n_layers):
            out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        pk_full, pv_full = _zero_past(per_layer_n_heads, d_head, S)
        full_logits = session.run(
            ["logits"], _feeds(token_ids, pk_full, pv_full, 0)
        )[0]

        # Prefill all-but-last, persisting deltas into slots [0 : n-1).
        past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
        outputs = session.run(out_names, _feeds(token_ids[:-1], past_k, past_v, 0))
        n_prefill = len(tokens) - 1
        for i in range(n_layers):
            past_k[i][:n_prefill] = outputs[1 + 2 * i]
            past_v[i][:n_prefill] = outputs[1 + 2 * i + 1]

        decode_logits = session.run(
            ["logits"], _feeds(token_ids[-1:], past_k, past_v, n_prefill)
        )[0]

        assert np.allclose(full_logits[-1], decode_logits[0], atol=1e-4), (
            f"decode vs full max diff: "
            f"{np.abs(full_logits[-1] - decode_logits[0]).max():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Autoregressive generation via REPL (1-digit adder)
# ---------------------------------------------------------------------------


def test_token_onnx_autoregressive_1digit():
    from torchwright.compiler.onnx_load import load_onnx

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

        model = load_onnx(onnx_path)
        test_cases = [("1+1\n", "2"), ("2+3\n", "5"), ("4+5\n", "9")]
        for input_str, expected in test_cases:
            result = "".join(model.generate(input_str))
            assert (
                result == expected
            ), f"{input_str}: expected {expected!r}, got {result!r}"


# ---------------------------------------------------------------------------
# Test 4: Autoregressive generation via REPL (3-digit adder)
# ---------------------------------------------------------------------------


def test_token_onnx_autoregressive_3digit():
    from torchwright.compiler.onnx_load import load_onnx

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

        model = load_onnx(onnx_path)
        test_cases = [
            ("1+2\n", "3"),
            ("12+34\n", "46"),
            ("99+1\n", "100"),
            ("100+200\n", "300"),
            ("456+123\n", "579"),
        ]
        for input_str, expected in test_cases:
            result = "".join(model.generate(input_str))
            assert (
                result == expected
            ), f"{input_str}: expected {expected!r}, got {result!r}"


# ---------------------------------------------------------------------------
# Test 5: Sidecar schema + repl metadata discovery
# ---------------------------------------------------------------------------


def test_token_onnx_sidecar_schema_and_metadata():
    from torchwright.compiler.onnx_load import OnnxTokenModule, load_onnx

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

        model = load_onnx(onnx_path)
        assert isinstance(model, OnnxTokenModule)
        assert model.n_layers > 0
        assert len(model.per_layer_n_heads) == model.n_layers
        assert all(nh <= D // D_HEAD for nh in model.per_layer_n_heads)
        assert model.d_head == D_HEAD


# ---------------------------------------------------------------------------
# Test 6: OnnxArtifact return handle + extra_metadata (token exporter)
# ---------------------------------------------------------------------------


def test_token_onnx_artifact_fields_load_and_generate():
    from torchwright.compiler.export import debug_meta_path_for
    from torchwright.compiler.onnx_load import OnnxTokenModule

    output_node, pos_encoding, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        artifact = compile_to_onnx(
            output_node,
            pos_encoding,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )

        assert artifact.kind == "token"
        assert artifact.path == onnx_path
        assert artifact.meta_path == meta_path_for(onnx_path)
        assert artifact.debug_path == debug_meta_path_for(onnx_path)
        assert artifact.d == D
        assert artifact.d_head == D_HEAD
        assert artifact.cache_stride == 512  # max_seq_len default
        assert artifact.cache_window is None
        # vocab_size is the embedding TABLE's row count (the logits
        # width) — the table is padded past the tokenizer's vocab list.
        assert artifact.vocab_size == embedding.table.shape[0]
        assert artifact.d_embed == embedding.table.shape[1]
        assert artifact.n_layers > 0
        assert artifact.per_layer_n_heads == tuple(
            OnnxTokenModule(onnx_path).per_layer_n_heads
        )

        model = artifact.load()
        assert isinstance(model, OnnxTokenModule)
        assert "".join(model.generate("2+3\n")) == "5"


def test_token_onnx_extra_metadata_roundtrip():
    from torchwright.compiler.onnx_load import load_onnx
    from torchwright.debug.onnx_debug import OnnxDebugSession

    output_node, pos_encoding, embedding = _build_1digit()
    extra = {"rows_per_patch": 7, "note": "hello"}

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
            extra_metadata=extra,
        )

        # Nested under "extra"; top-level keys unchanged.
        with open(meta_path_for(onnx_path)) as f:
            meta = json.load(f)
        assert meta["extra"] == extra
        assert meta["format"] == TOKEN_META_FORMAT
        assert meta["vocab"] == embedding.tokenizer.vocab
        assert meta["cache_stride"] == 512

        # Surfaced by the loader...
        model = load_onnx(onnx_path)
        assert model.metadata == extra

        # ...and by the debug session (full sidecar dict there).
        out2, pos2, _emb2 = _build_1digit()
        sess = OnnxDebugSession(onnx_path, out2, pos2)
        assert sess.metadata["extra"] == extra


def test_token_onnx_meta_has_no_extra_key_when_omitted():
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
        with open(meta_path_for(onnx_path)) as f:
            meta = json.load(f)
        assert "extra" not in meta
        assert set(meta) == {"format", "vocab", "cache_stride"}
