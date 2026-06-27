"""Tests for :class:`torchwright.debug.onnx_debug.OnnxDebugSession`.

The contract under test: a compiled ONNX artifact plus its
``<stem>.debug.json`` sidecar plus a *freshly rebuilt* graph (different
``node_id``s — every test here rebuilds, exercising the canonical-id
remap) gives the same debug surface as ``CompiledHeadless`` — the same
``probe_compiled`` verdict, the same ``debug_value``s, working
``debug=True`` assert checks — while running the real artifact under
onnxruntime.

Includes the emission-divergence reproducer (corrupt one initializer in
the saved model → the probe must flag it): that is the bug class the
ONNX backend exists to catch and the in-process backend structurally
cannot.
"""

import json
import os
import shutil

import pytest
import torch

from torchwright.compiler.export import (
    DEBUG_META_FORMAT,
    compile_headless,
    compile_headless_to_onnx,
    compile_to_onnx,
    debug_meta_path_for,
)
from torchwright.compiler.graph_identity import (
    debug_fingerprint,
    decode_cols,
    encode_cols,
    graph_fingerprint,
)
from torchwright.debug.probe import (
    probe_attention,
    probe_compiled,
    probe_residual,
)
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.attn import Attn
from torchwright.ops.arithmetic_ops import add, multiply_const, signed_multiply
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

onnxruntime = pytest.importorskip("onnxruntime")

from torchwright.debug.onnx_debug import OnnxDebugSession  # noqa: E402

D = 1024
D_HEAD = 16
TOKENS = ["<bos>", "1", "+", "2", "\n"]


def _build_adder():
    """Fresh 1-digit adder graph — new node ids every call."""
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        from examples.adder import create_network_parts

        return create_network_parts()
    finally:
        adder_module.max_digits = original


def _token_ids(embedding) -> torch.Tensor:
    return torch.tensor(
        [embedding.tokenizer.get_token_id(t) for t in TOKENS], dtype=torch.long
    ).reshape(-1, 1)


@pytest.fixture(scope="module")
def token_artifact(tmp_path_factory):
    """A compiled token-I/O artifact (1-digit adder) with debug sidecar."""
    tmpdir = tmp_path_factory.mktemp("onnx_debug")
    onnx_path = str(tmpdir / "adder.onnx")
    output_node, pos_encoding, embedding = _build_adder()
    compile_to_onnx(
        output_node,
        pos_encoding,
        embedding,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        verbose=False,
    )
    return onnx_path


# ---------------------------------------------------------------------------
# Sidecar contents
# ---------------------------------------------------------------------------


def test_encode_decode_cols_roundtrip():
    for cols in (
        [],
        [3],
        [0, 1, 2, 3],
        [5, 6, 7, 100, 101, 9],  # order-preserving: 9 after the 100-run
        [2, 1, 0],  # descending never merges
    ):
        assert decode_cols(encode_cols(cols)) == cols


def test_sidecar_written_with_expected_schema(token_artifact):
    path = debug_meta_path_for(token_artifact)
    assert os.path.exists(path)
    with open(path) as f:
        sidecar = json.load(f)
    assert sidecar["format"] == DEBUG_META_FORMAT
    assert sidecar["kind"] == "token"
    assert sidecar["d"] == D
    n_layers = sidecar["n_layers"]
    keys = [e["key"] for e in sidecar["states"]]
    assert keys[0] == "input"
    assert keys[1:] == [
        f"L{i}.{s}" for i in range(n_layers) for s in ("attn", "mlp")
    ]
    # The adder graph carries internal asserts; coverage must see them.
    assert sidecar["assert_coverage"]["n_asserts"] > 0


def _build_annotated_graph():
    """Small graph whose nodes carry nested ``annotate()`` label paths."""
    from torchwright.graph.node import annotate

    a = create_input("a", 1)
    b = create_input("b", 1)
    with annotate("scene"):
        prod = signed_multiply(a, b, max_abs1=10, max_abs2=10)
        with annotate("scene/sum"):
            out = add(prod, multiply_const(a, 2.0))
    return out, create_pos_encoding(), prod, out


def test_sidecar_carries_annotations(tmp_path):
    """The ``annotate()`` label path round-trips through the debug sidecar
    and onto OnnxDebugSession.annotation() against a fresh rebuild."""
    out, pos, prod, top = _build_annotated_graph()
    onnx_path = str(tmp_path / "annotated.onnx")
    compile_headless_to_onnx(
        out, pos, onnx_path, d=D, d_head=D_HEAD, max_seq_len=16, verbose=False
    )

    with open(debug_meta_path_for(onnx_path)) as f:
        sidecar = json.load(f)
    # Annotations now ride in the per-node table; "scene/sum" is deduped,
    # not "scene/scene/sum".
    labels = {
        m["annotation"]
        for m in sidecar["nodes"].values()
        if m["annotation"] is not None
    }
    assert "scene" in labels
    assert "scene/sum" in labels

    # Round-trip onto a fresh rebuild (new node ids) via the session.
    out2, pos2, prod2, top2 = _build_annotated_graph()
    sess = OnnxDebugSession(onnx_path, out2, pos2)
    assert sess.annotation(prod2) == "scene"
    assert sess.annotation(top2) == "scene/sum"


def test_sidecar_nodes_table_schema(tmp_path):
    """The per-node table keys by canonical id (same space as placements /
    states) and carries op/width/weights/inputs/layer/sublayer per node."""
    out, pos, prod, top = _build_annotated_graph()
    onnx_path = str(tmp_path / "nodes.onnx")
    compile_headless_to_onnx(
        out, pos, onnx_path, d=D, d_head=D_HEAD, max_seq_len=16, verbose=False,
        optimize=1, extra_metadata={"screen": {"width": 80, "height": 50}, "scale": 4},
    )
    with open(debug_meta_path_for(onnx_path)) as f:
        sidecar = json.load(f)

    # Compile knob is first-class; caller metadata rides through "extra"
    # verbatim (torchwright does not interpret its keys).
    assert sidecar["optimize"] == 1
    assert sidecar["extra"] == {"screen": {"width": 80, "height": 50}, "scale": 4}

    nodes = sidecar["nodes"]
    assert nodes, "nodes table should be non-empty"
    # Every placement / state key is a node in the table.
    for key in sidecar["placements"]:
        if key.startswith("_"):  # reserved op/unreachable buckets
            continue
        assert key in nodes, f"placement key {key} missing from nodes"
    for entry in sidecar["states"]:
        for key in entry.get("nodes", {}):
            assert key in nodes, f"state node {key} missing from nodes"

    # Required per-entry shape.
    for cid, m in nodes.items():
        assert isinstance(m["op"], str)
        assert isinstance(m["width"], int)
        assert isinstance(m["weight_params"], int)
        assert isinstance(m["weight_shapes"], list)
        assert isinstance(m["inputs"], list)
        assert all(i in nodes for i in m["inputs"])
        assert m["layer"] is None or isinstance(m["layer"], int)
        assert m["sublayer"] in (None, "attn", "mlp", "embed")

    # The two InputNodes carry weight_params == 0; some weight-bearing node
    # (the signed-multiply / add chain compiles to Linears) has > 0.
    assert any(m["weight_params"] > 0 for m in nodes.values())
    # A computed node lands in a real transformer layer/sublayer.
    assert any(
        m["layer"] is not None and m["sublayer"] in ("attn", "mlp")
        for m in nodes.values()
    )


# ---------------------------------------------------------------------------
# Parity with CompiledHeadless
# ---------------------------------------------------------------------------


def test_probe_and_debug_value_parity_with_headless(token_artifact):
    # ONNX backend over a fresh rebuild.
    out_onnx, pos_onnx, emb_onnx = _build_adder()
    session = OnnxDebugSession(token_artifact, out_onnx, pos_onnx)
    ids = _token_ids(emb_onnx)
    iv = {"embedding_input": ids}
    report_onnx = probe_compiled(session, out_onnx, iv, n_pos=len(TOKENS), atol=1e-3)
    assert report_onnx.first_divergent is None, report_onnx.format_short()

    # In-process backend over another fresh rebuild.
    out_h, pos_h, emb_h = _build_adder()
    headless = compile_headless(out_h, pos_h, d=D, d_head=D_HEAD, verbose=False)
    report_h = probe_compiled(headless, out_h, iv, n_pos=len(TOKENS), atol=1e-3)
    assert report_h.first_divergent is None, report_h.format_short()

    # Same probe coverage on both backends.
    assert len(report_onnx.nodes_checked) == len(report_h.nodes_checked)

    # debug_value parity at the output node.
    session(ids, debug=True)
    headless(headless.build_prefill(iv, len(TOKENS)), debug=True)
    v_onnx = session.debug_value(out_onnx)
    v_h = headless.debug_value(out_h)
    assert v_onnx is not None and v_h is not None
    assert float((v_onnx - v_h).abs().max()) < 1e-4


def test_decode_with_past_matches_prefill_and_debug_passes(token_artifact):
    out, pos, emb = _build_adder()
    session = OnnxDebugSession(token_artifact, out, pos)
    ids = _token_ids(emb)

    full = session(ids, debug=True)

    past = session.empty_past()
    _, past = session.step(ids[:-1], past)
    last, past = session.step(ids[-1:], past, debug=True)
    assert past[0][0].shape[0] == len(TOKENS)
    assert float((full[-1] - last[0]).abs().max()) < 1e-5


def test_probe_attention_fetches_artifact_weights(token_artifact):
    out, pos, emb = _build_adder()
    session = OnnxDebugSession(token_artifact, out, pos)
    ids = _token_ids(emb)

    stack, seen, attn_node = [out], set(), None
    while stack:
        n = stack.pop()
        if n.node_id in seen:
            continue
        seen.add(n.node_id)
        if isinstance(n, Attn):
            attn_node = n
            break
        stack.extend(getattr(n, "inputs", None) or [])
    assert attn_node is not None

    ap = probe_attention(session, ids, attn_node, query_pos=len(TOKENS) - 1)
    assert ap.weights.shape == ap.logits.shape
    assert ap.weights.shape[1] == len(TOKENS)  # n_keys = past + new
    # Each head's weights are a softmax row.
    assert torch.allclose(ap.weights.sum(dim=-1), torch.ones(ap.weights.shape[0]))


# ---------------------------------------------------------------------------
# Fingerprint and assert semantics
# ---------------------------------------------------------------------------


def test_assert_wrappers_are_fingerprint_transparent_and_fire(token_artifact):
    out, pos, emb = _build_adder()
    wrapped = assert_in_range(out, lo=1e6, hi=2e6)  # impossible claim
    # Wrapping must NOT change the fingerprint — the session loads fine...
    session = OnnxDebugSession(token_artifact, wrapped, pos)
    # ...and the rebuilt graph's assert fires on the artifact's values.
    with pytest.raises(AssertionError):
        session(_token_ids(emb), debug=True)


def test_fingerprint_mismatch_raises(token_artifact):
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 2
        from examples.adder import create_network_parts

        out_big, pos_big, _ = create_network_parts()
    finally:
        adder_module.max_digits = original
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        OnnxDebugSession(token_artifact, out_big, pos_big)


def test_assert_coverage_warning_on_weaker_rebuild(tmp_path, capsys):
    onnx_path = str(tmp_path / "adder_asserted.onnx")
    out, pos, emb = _build_adder()
    compile_to_onnx(
        assert_in_range(out, lo=-1e9, hi=1e9),
        pos,
        emb,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        verbose=False,
    )
    out2, pos2, _ = _build_adder()  # rebuilt WITHOUT the extra assert
    OnnxDebugSession(onnx_path, out2, pos2)
    assert "checking fewer invariants" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Emission divergence — the bug class only this backend can catch
# ---------------------------------------------------------------------------


def test_corrupted_initializer_is_detected(token_artifact, tmp_path):
    """D6 reproducer: an artifact whose weights differ from what the
    compiler computed must show up as oracle divergence on the ONNX
    backend.  (The in-process backend never reads the artifact, so it
    is structurally blind to this class.)"""
    import numpy as np
    import onnx

    model = onnx.load(token_artifact)
    target = next(i for i in model.graph.initializer if i.name == "l0_b1")
    arr = onnx.numpy_helper.to_array(target).copy()
    arr = arr * 1.5
    arr.flat[0] += 7.0
    target.CopyFrom(onnx.numpy_helper.from_array(arr.astype(np.float32), "l0_b1"))

    corrupt = str(tmp_path / "corrupt.onnx")
    onnx.save_model(model, corrupt)
    for ext in (".meta.json", ".debug.json"):
        shutil.copy(token_artifact[:-5] + ext, corrupt[:-5] + ext)

    out, pos, emb = _build_adder()
    session = OnnxDebugSession(corrupt, out, pos)
    report = probe_compiled(
        session,
        out,
        {"embedding_input": _token_ids(emb)},
        n_pos=len(TOKENS),
        atol=1e-3,
    )
    assert report.first_divergent is not None


# ---------------------------------------------------------------------------
# Headless (float-I/O) artifact kind
# ---------------------------------------------------------------------------


def _build_headless_graph():
    a = create_input("a", 1)
    b = create_input("b", 1)
    return signed_multiply(a, b, max_abs1=10, max_abs2=10), create_pos_encoding()


def test_headless_kind_probe_and_residual(tmp_path):
    onnx_path = str(tmp_path / "model.onnx")
    out1, pos1 = _build_headless_graph()
    compile_headless_to_onnx(
        out1, pos1, onnx_path, d=256, d_head=D_HEAD, max_seq_len=32, verbose=False
    )

    out2, pos2 = _build_headless_graph()
    session = OnnxDebugSession(onnx_path, out2, pos2)
    assert session._input_specs == [("a", 0, 1), ("b", 1, 1)]

    iv = {
        "a": torch.tensor([[1.0], [2.0], [3.0], [-4.0]]),
        "b": torch.tensor([[5.0], [6.0], [-7.0], [8.0]]),
    }
    report = probe_compiled(session, out2, iv, n_pos=4, atol=1e-2)
    assert report.first_divergent is None, report.format_short()

    prefill = session.build_prefill(iv, 4)
    outputs = session(prefill, debug=True)
    expected = torch.tensor([5.0, 12.0, -21.0, -32.0])
    assert float((outputs[:, 0] - expected).abs().max()) < 1e-2

    rp = probe_residual(session, prefill, out2)
    assert rp.layers, "output node never materialised in any snapshot"


# ---------------------------------------------------------------------------
# Fingerprint stability guards
# ---------------------------------------------------------------------------


def _tiny_graph():
    a = create_input("a", 1)
    b = create_input("b", 1)
    return add(multiply_const(a, 2.0), b), create_pos_encoding()


def test_fingerprints_are_pinned():
    """The fingerprint encodings are persistence formats.

    ``graph_fingerprint`` keys every ``TW_SCHEDULE_CACHE_DIR`` entry and
    ``debug_fingerprint`` keys every ``.debug.json`` sidecar — changing
    either encoding silently invalidates all of them.  If this test
    fails, you changed the encoding: confirm that is intentional and
    update the pins (existing schedule caches will re-solve and existing
    debug sidecars will need re-export).
    """
    out, pos = _tiny_graph()
    assert (
        debug_fingerprint(out, pos, d=256, d_head=16)
        == "8164b45dfdb3cc78093c6dba7e57e3158aa1b7d275707e6bd32534e1350e03d0"
    )
    assert (
        graph_fingerprint(
            out,
            pos,
            d=256,
            d_head=16,
            d_hidden=256,
            flex_routing=True,
            assume_zero_init=True,
            cancel_slack=2,
            policy=None,
        )
        == "9aa233367f7f4889d5e105439ee14a95b90216b251f9c913f9da1cb2280a254c"
    )


def test_fingerprint_stable_across_rebuilds_and_wrappers():
    out1, pos1 = _tiny_graph()
    out2, pos2 = _tiny_graph()  # fresh node ids
    fp1 = debug_fingerprint(out1, pos1, d=256, d_head=16)
    assert fp1 == debug_fingerprint(out2, pos2, d=256, d_head=16)
    # Assert wrappers are transparent — including at the root.
    wrapped = assert_in_range(out2, lo=-1e9, hi=1e9)
    assert fp1 == debug_fingerprint(wrapped, pos2, d=256, d_head=16)


# ---------------------------------------------------------------------------
# OnnxArtifact.debug_session — handle-built session matches direct
# ---------------------------------------------------------------------------


def test_artifact_debug_session_matches_direct(tmp_path):
    onnx_path = str(tmp_path / "model.onnx")
    out1, pos1 = _build_headless_graph()
    artifact = compile_headless_to_onnx(
        out1, pos1, onnx_path, d=256, d_head=D_HEAD, max_seq_len=32, verbose=False
    )

    iv = {
        "a": torch.tensor([[1.0], [2.0], [3.0], [-4.0]]),
        "b": torch.tensor([[5.0], [6.0], [-7.0], [8.0]]),
    }

    out2, pos2 = _build_headless_graph()
    sess_handle = artifact.debug_session(out2, pos2)
    assert isinstance(sess_handle, OnnxDebugSession)
    report = probe_compiled(sess_handle, out2, iv, n_pos=4, atol=1e-2)
    assert report.first_divergent is None, report.format_short()

    out3, pos3 = _build_headless_graph()
    sess_direct = OnnxDebugSession(onnx_path, out3, pos3)
    a = sess_handle(sess_handle.build_prefill(iv, 4))
    b = sess_direct(sess_direct.build_prefill(iv, 4))
    assert torch.allclose(a, b, atol=0)
