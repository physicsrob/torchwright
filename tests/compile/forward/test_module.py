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
        output_node, embedding = create_network_parts()
    finally:
        adder_module.max_digits = original
    return output_node, embedding


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


def _reference_logits(output_node, embedding, tokens):
    """Run compute() and return full logits (seq_len, vocab_size).

    Applies the same ``out_emb @ embedding.table.T`` unembed as the ONNX
    graph, so the result is numerically comparable to the ONNX output
    via ``np.allclose``.
    """
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
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
    output_node, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )

        tokens = ["<bos>", "1", "+", "2", "\n"]
        ref_logits = _reference_logits(output_node, embedding, tokens)

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
    output_node, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
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
        full_logits = session.run(["logits"], _feeds(token_ids, pk_full, pv_full, 0))[0]

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

    output_node, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
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

    output_node, embedding = create_network_parts()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder3.onnx")
        compile_to_onnx(
            output_node,
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

    output_node, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
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

    output_node, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        artifact = compile_to_onnx(
            output_node,
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
        # vocab_size is the embedding TABLE's row count (the logits
        # width) — the table is padded past the tokenizer's vocab list.
        assert artifact.vocab_size == embedding.table.shape[0]
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

    output_node, embedding = _build_1digit()
    extra = {"rows_per_patch": 7, "note": "hello"}

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
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
        out2, _emb2 = _build_1digit()
        sess = OnnxDebugSession(onnx_path, out2)
        assert sess.metadata["extra"] == extra


def test_token_onnx_meta_has_no_extra_key_when_omitted():
    output_node, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )
        with open(meta_path_for(onnx_path)) as f:
            meta = json.load(f)
        assert "extra" not in meta
        # rope_base and d_rot are present for every rotary model (every model
        # post-RoPE-port; d_rot == d_head here, full rotary); rms_norm/rms_norm_eps
        # for every model post-RMSNorm; "schedule" (solver provenance) for
        # every model post-2026-07; "activation" (machine kind) for every
        # model post-swiglu-step-2; "extra" (asserted absent above) is the
        # optional key this guards against.
        assert set(meta) == {
            "format",
            "vocab",
            "cache_stride",
            "rope_base",
            "d_rot",
            "rms_norm",
            "rms_norm_eps",
            "activation",
            "bias",
            "collapse_univariate",
            "collapse_pl",
            "schedule",
        }
        # The adder builds the swish machine (Phase C examples cutover) —
        # the machine kind records that.
        assert meta["activation"] == "swish"
        # Emission mode ("bias") is recorded for every model post-no-bias
        # (docs/no_bias_plan.md); default compiles are biased.
        assert meta["bias"] is True
        # optimize defaults to 0 here, so the selected heuristic schedule has
        # no solver-attempt fields.
        assert meta["schedule"] == {
            "optimize": 0,
            "selected_origin": "heuristic",
            "delivery": "fresh",
            "selected_is_optimal": False,
        }


# ===========================================================================
# KV-cache protocol + head/slot-trim coverage.
#
# Ported from the retired float-I/O ("headless") ONNX exporter's tests: the
# cached preamble (in-graph causal mask), the ScatterND static-tail, the
# stride-bucket prefix-window binding, and the head/MLP trimming are all
# SHARED machinery (compile_to_onnx and the removed compile_headless_to_onnx
# emitted the identical graph), so they are exercised here on the surviving
# token vehicle.  The single-row decode seam + prefill-vs-compute already live
# above; these add the combinations those do not cover.
# ===========================================================================

_PROTO_TOKENS = ["<bos>", "1", "+", "2", "\n"]


def _ids(embedding, tokens):
    return np.array(
        [embedding.tokenizer.get_token_id(t) for t in tokens], dtype=np.int64
    )


@pytest.fixture(scope="module")
def adder_artifact(tmp_path_factory):
    """A compiled 1-digit adder token artifact (max_seq_len=32) shared by the
    cached-protocol self-consistency tests below."""
    output_node, embedding = _build_1digit()
    tmpdir = str(tmp_path_factory.mktemp("token_onnx_proto"))
    onnx_path = os.path.join(tmpdir, "adder.onnx")
    compile_to_onnx(
        output_node,
        embedding,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        max_seq_len=32,
        verbose=False,
    )
    return onnx_path, embedding


def test_token_onnx_chunked_decode_matches_full_prefill(adder_artifact):
    """Prefill 2 rows, then decode a 3-row chunk (n_new>1 at base>0) — the
    dynamic-mask seam the single-row decode test does not cover."""
    onnx_path, embedding = adder_artifact
    token_ids = _ids(embedding, _PROTO_TOKENS)
    session = onnxruntime.InferenceSession(onnx_path)
    n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)
    out_names = ["logits"]
    for i in range(n_layers):
        out_names += [f"delta_K_{i}", f"delta_V_{i}"]

    pk_full, pv_full = _zero_past(per_layer_n_heads, d_head, S)
    full = session.run(["logits"], _feeds(token_ids, pk_full, pv_full, 0))[0]

    past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
    results = session.run(out_names, _feeds(token_ids[:2], past_k, past_v, 0))
    for i in range(n_layers):
        past_k[i][0:2] = results[1 + 2 * i]
        past_v[i][0:2] = results[1 + 2 * i + 1]

    chunk = session.run(["logits"], _feeds(token_ids[2:5], past_k, past_v, 2))[0]
    assert np.allclose(
        full[2:5], chunk, atol=1e-4
    ), f"chunked decode diff: {np.abs(full[2:5] - chunk).max():.6f}"


def test_token_onnx_static_tail_is_inert(adder_artifact):
    """Finite garbage in masked tail slots (positions > cache_position) must
    not change the decode output — the in-graph mask zeroes those weights
    exactly.  (Slot n itself is overwritten by the ScatterND diagonal.)"""
    onnx_path, embedding = adder_artifact
    token_ids = _ids(embedding, _PROTO_TOKENS)
    n = 4
    session = onnxruntime.InferenceSession(onnx_path)
    n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)
    out_names = ["logits"]
    for i in range(n_layers):
        out_names += [f"delta_K_{i}", f"delta_V_{i}"]

    pk_full, pv_full = _zero_past(per_layer_n_heads, d_head, S)
    full = session.run(["logits"], _feeds(token_ids, pk_full, pv_full, 0))[0]

    past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
    results = session.run(out_names, _feeds(token_ids[:n], past_k, past_v, 0))
    for i in range(n_layers):
        past_k[i][:n] = results[1 + 2 * i]
        past_v[i][:n] = results[1 + 2 * i + 1]

    def decode(pk, pv):
        return session.run(["logits"], _feeds(token_ids[n : n + 1], pk, pv, n))[0]

    exact = decode(past_k, past_v)
    assert np.allclose(full[n], exact[0], atol=1e-4)

    garbage_k = [pk.copy() for pk in past_k]
    garbage_v = [pv.copy() for pv in past_v]
    for i in range(n_layers):
        garbage_k[i][n:] = 7.0
        garbage_v[i][n:] = -3.0
    dirty = decode(garbage_k, garbage_v)
    assert np.allclose(exact[0], dirty[0], atol=1e-6), (
        "garbage in masked static-cache slots changed the output "
        f"(max diff {np.abs(exact[0] - dirty[0]).max():.6e}) — the mask is "
        "not zeroing the tail weights exactly"
    )


def test_token_onnx_prefix_window_binding(adder_artifact):
    """Every covering prefix window cache[:S_eff] (base+n_new <= S_eff <= S)
    must produce bit-identical outputs — the symbolic cache_slots dim derives
    the mask width from Shape(past_K_0)."""
    onnx_path, embedding = adder_artifact
    token_ids = _ids(embedding, _PROTO_TOKENS)
    n = 4
    session = onnxruntime.InferenceSession(onnx_path)
    n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)
    out_names = ["logits"]
    for i in range(n_layers):
        out_names += [f"delta_K_{i}", f"delta_V_{i}"]

    past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
    results = session.run(out_names, _feeds(token_ids[:n], past_k, past_v, 0))
    for i in range(n_layers):
        past_k[i][:n] = results[1 + 2 * i]
        past_v[i][:n] = results[1 + 2 * i + 1]

    def decode_window(s_eff):
        pk = [k[:s_eff] for k in past_k]
        pv = [v[:s_eff] for v in past_v]
        return session.run(out_names, _feeds(token_ids[n : n + 1], pk, pv, n))

    ref = decode_window(S)
    for s_eff in (n + 1, n + 3, S - 1):
        got = decode_window(s_eff)
        for r, g, name in zip(ref, got, out_names):
            assert np.array_equal(r, g), (
                f"S_eff={s_eff}: {name} differs from the full-S binding "
                f"(max diff {np.abs(r - g).max():.6e})"
            )


def test_token_onnx_module_step_matches_full_call(adder_artifact):
    """OnnxTokenModule.step threads the cache: prefill 4 + decode 1 == one
    full-sequence call."""
    from torchwright.compiler.onnx_load import OnnxTokenModule

    onnx_path, embedding = adder_artifact
    token_ids = torch.tensor(_ids(embedding, _PROTO_TOKENS), dtype=torch.int64)
    module = OnnxTokenModule(onnx_path)

    full = module(token_ids)
    past = module.empty_past()
    prefill_out, past = module.step(token_ids[:4], past)
    decode_out, past = module.step(token_ids[4:5], past)

    assert torch.allclose(full[:4], prefill_out, atol=1e-4)
    assert torch.allclose(full[4], decode_out[0], atol=1e-4)


def test_token_onnx_module_empty_past_shape(adder_artifact):
    from torchwright.compiler.onnx_load import OnnxTokenModule

    onnx_path, _ = adder_artifact
    module = OnnxTokenModule(onnx_path)
    past = module.empty_past()
    S = module.cache_stride
    assert S == 32  # max_seq_len passed to the fixture; cache_stride defaults to it
    assert past.length == 0
    assert len(past.k) == module.n_layers
    for i, K in enumerate(past.k):
        assert K.shape == (S, module.per_layer_n_heads[i], module.d_head)
        assert (K == 0).all(), "static cache must be zero-initialized"


# ---- Head / MLP-slot trimming (compile_to_onnx trim_heads) ----------------


def _export_token(tmpdir, name="model.onnx", trim_heads=True):
    output_node, embedding = _build_1digit()
    onnx_path = os.path.join(tmpdir, name)
    compile_to_onnx(
        output_node,
        embedding,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        max_seq_len=32,
        verbose=False,
        trim_heads=trim_heads,
    )
    return onnx_path, embedding


def _l_w1_d_hidden(onnx_path):
    """Per-layer MLP hidden widths read off the l{i}_W1 initializers
    (shape (d, d_hidden_i)); densify-aware (sparse protos carry full dims)."""
    import onnx

    model = onnx.load(onnx_path)
    dims_by_name = {init.name: list(init.dims) for init in model.graph.initializer}
    for sp in model.graph.sparse_initializer:
        dims_by_name[sp.values.name] = list(sp.dims)
    widths = []
    i = 0
    # The hidden-width-bearing initializer is l{i}_W1 on the relu machine
    # and l{i}_Wgate on the gated machine; accept either so the helper
    # follows the fixture's machine.
    while f"l{i}_W1" in dims_by_name or f"l{i}_Wgate" in dims_by_name:
        name = f"l{i}_W1" if f"l{i}_W1" in dims_by_name else f"l{i}_Wgate"
        dims = dims_by_name[name]
        assert dims[0] == D, f"{name} first dim {dims[0]} != d {D}"
        widths.append(int(dims[1]))
        i += 1
    assert widths, "no l{i}_W1 / l{i}_Wgate initializers found"
    return widths


def test_token_onnx_trim_heads_shrinks_kv_cache():
    """trim_heads=True drops per-layer past_K widths below the full head count
    and leaves them non-uniform across layers."""
    max_heads = D // D_HEAD
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path, _ = _export_token(tmpdir, trim_heads=True)
        session = onnxruntime.InferenceSession(onnx_path)
        _, per_layer_n_heads, _, _ = _discover_meta(session, onnx_path)
    assert per_layer_n_heads, "no layers discovered"
    assert all(1 <= nh <= max_heads for nh in per_layer_n_heads)
    assert (
        min(per_layer_n_heads) < max_heads
    ), f"no layer trimmed below full width {max_heads}: {per_layer_n_heads}"
    assert (
        len(set(per_layer_n_heads)) > 1
    ), f"head counts are uniform across layers: {per_layer_n_heads}"


def test_token_onnx_no_trim_preserves_full_width():
    """trim_heads=False keeps every layer at the full d/d_head head count."""
    max_heads = D // D_HEAD
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path, _ = _export_token(tmpdir, trim_heads=False)
        session = onnxruntime.InferenceSession(onnx_path)
        _, per_layer_n_heads, _, _ = _discover_meta(session, onnx_path)
    assert per_layer_n_heads, "no layers discovered"
    assert all(
        nh == max_heads for nh in per_layer_n_heads
    ), f"expected all layers at full width {max_heads}: {per_layer_n_heads}"


def test_token_onnx_trim_shrinks_mlp_slots():
    """trim_heads=True shrinks some hidden widths below full d_hidden; False keeps all."""
    full_d_hidden = D  # d_hidden defaults to d
    with tempfile.TemporaryDirectory() as tmpdir:
        trim_path, _ = _export_token(tmpdir, "trim.onnx", trim_heads=True)
        notrim_path, _ = _export_token(tmpdir, "notrim.onnx", trim_heads=False)
        trim_widths = _l_w1_d_hidden(trim_path)
        notrim_widths = _l_w1_d_hidden(notrim_path)
    assert all(1 <= w <= full_d_hidden for w in trim_widths)
    assert (
        min(trim_widths) < full_d_hidden
    ), f"no layer's MLP trimmed below full d_hidden {full_d_hidden}: {trim_widths}"
    assert all(w == full_d_hidden for w in notrim_widths)


def test_token_onnx_trim_is_numerical_noop():
    """Trimmed and full-width ONNX produce the same logits on the same prefill
    (trimming removes only all-zero heads/slots)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trim_path, emb = _export_token(tmpdir, "trim.onnx", trim_heads=True)
        notrim_path, _ = _export_token(tmpdir, "notrim.onnx", trim_heads=False)
        token_ids = _ids(emb, _PROTO_TOKENS)
        sess_trim = onnxruntime.InferenceSession(trim_path)
        sess_notrim = onnxruntime.InferenceSession(notrim_path)
        _, ht, d_head, St = _discover_meta(sess_trim, trim_path)
        _, hn, _, Sn = _discover_meta(sess_notrim, notrim_path)
        pk, pv = _zero_past(ht, d_head, St)
        out_trim = sess_trim.run(["logits"], _feeds(token_ids, pk, pv, 0))[0]
        pk, pv = _zero_past(hn, d_head, Sn)
        out_notrim = sess_notrim.run(["logits"], _feeds(token_ids, pk, pv, 0))[0]
    assert np.allclose(
        out_trim, out_notrim, atol=1e-4
    ), f"trim changed the output: max diff {np.abs(out_trim - out_notrim).max():.6e}"
