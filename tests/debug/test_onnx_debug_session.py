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
from torchwright.ops.linear import add, multiply_const
from torchwright.ops.inout_nodes import create_input

onnxruntime = pytest.importorskip("onnxruntime")

from torchwright.debug.onnx_debug import OnnxDebugSession  # noqa: E402

D = 1024
D_HEAD = 16
TOKENS = ["<bos>", "1", "+", "2", "\n"]

# onnxruntime-execution probe tolerance. The in-process headless backend matches
# the graph oracle to fp32 round-off (~2e-6, kept at 1e-3 below), but the real
# onnxruntime artifact rounds ~4e-5 relative on the large-magnitude attention the
# Phase-6 local-recency head introduces (~1.2e-3 absolute on this adder) — an
# execution-provider floor, not a compiled-circuit error. See
# docs/numerical_noise_findings.md.
_ONNX_PROBE_ATOL = 2.5e-3


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
    output_node, embedding = _build_adder()
    compile_to_onnx(
        output_node,
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
    assert keys[1:] == [f"L{i}.{s}" for i in range(n_layers) for s in ("attn", "mlp")]
    # The adder graph carries internal asserts; coverage must see them.
    assert sidecar["assert_coverage"]["n_asserts"] > 0


def _build_annotated_graph():
    """Small TOKEN graph whose nodes carry nested ``annotate()`` label paths.

    The output is a d_embed-wide embedding pick (a valid 'next token') so
    ``compile_to_onnx`` can unembed it.
    """
    from torchwright.graph.node import annotate
    from torchwright.ops.inout_nodes import create_embedding, create_literal_value
    from torchwright.ops.relu.logic_ops import equals_vector
    from torchwright.ops.relu.map_select import select

    vocab = list("0123456789+") + ["\n", "<bos>", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    with annotate("scene"):
        cond = equals_vector(inp=embedding, vector=embedding.get_embedding("+"))
        with annotate("scene/sum"):
            out = select(
                cond,
                create_literal_value(embedding.get_embedding("1")),
                create_literal_value(embedding.get_embedding("0")),
            )
    return out, embedding, cond, out


def test_sidecar_carries_annotations(tmp_path):
    """The ``annotate()`` label path round-trips through the debug sidecar
    and onto OnnxDebugSession.annotation() against a fresh rebuild."""
    out, embedding, prod, top = _build_annotated_graph()
    onnx_path = str(tmp_path / "annotated.onnx")
    compile_to_onnx(
        out,
        embedding,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        max_seq_len=16,
        verbose=False,
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
    out2, _emb2, prod2, top2 = _build_annotated_graph()
    sess = OnnxDebugSession(onnx_path, out2)
    assert sess.annotation(prod2) == "scene"
    assert sess.annotation(top2) == "scene/sum"


def test_sidecar_nodes_table_schema(tmp_path):
    """The per-node table keys by canonical id (same space as placements /
    states) and carries op/width/weights/inputs/layer/sublayer per node."""
    out, embedding, prod, top = _build_annotated_graph()
    onnx_path = str(tmp_path / "nodes.onnx")
    compile_to_onnx(
        out,
        embedding,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        max_seq_len=16,
        verbose=False,
        optimize=1,
        extra_metadata={"screen": {"width": 80, "height": 50}, "scale": 4},
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
    out_onnx, emb_onnx = _build_adder()
    session = OnnxDebugSession(token_artifact, out_onnx)
    ids = _token_ids(emb_onnx)
    iv = {"embedding_input": ids}
    report_onnx = probe_compiled(
        session, out_onnx, iv, n_pos=len(TOKENS), atol=_ONNX_PROBE_ATOL
    )
    assert report_onnx.first_divergent is None, report_onnx.format_short()

    # In-process backend over another fresh rebuild — the compiled circuit matches
    # the oracle to fp32 round-off, so this stays tight (no onnxruntime rounding).
    out_h, emb_h = _build_adder()
    headless = compile_headless(out_h, d=D, d_head=D_HEAD, verbose=False)
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
    # v_onnx carries the onnxruntime fp32 execution floor (the same one this
    # test budgets _ONNX_PROBE_ATOL=2.5e-3 against above), while v_h is
    # oracle-tight; so their cross-backend difference must allow that floor, not
    # the 1e-4 that assumed both sides were oracle-tight. The old 1e-4 sat below
    # the onnxruntime floor and flipped on the EP's rounding (a single 2^-10 ULP
    # at the adder's value magnitude tripped it deterministically on some
    # workers). probe_compiled(session, atol=2.5e-3) above already proves the
    # ONNX backend matches the oracle at every node, so this is purely the
    # cross-backend round-off budget.
    assert float((v_onnx - v_h).abs().max()) < _ONNX_PROBE_ATOL


def test_decode_with_past_matches_prefill_and_debug_passes(token_artifact):
    out, emb = _build_adder()
    session = OnnxDebugSession(token_artifact, out)
    ids = _token_ids(emb)

    full = session(ids, debug=True)

    past = session.empty_past()
    _, past = session.step(ids[:-1], past)
    last, past = session.step(ids[-1:], past, debug=True)
    assert past[0][0].shape[0] == len(TOKENS)
    assert float((full[-1] - last[0]).abs().max()) < 1e-5


def test_probe_attention_fetches_artifact_weights(token_artifact):
    out, emb = _build_adder()
    session = OnnxDebugSession(token_artifact, out)
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


def test_attached_checks_are_fingerprint_transparent_and_fire(token_artifact):
    out, emb = _build_adder()
    # A claim the artifact's actual values violate (kept inside the
    # output's existing op-tail claim so the attach-time intersection is
    # non-empty — a fully disjoint claim raises at attach, by design).
    checked = assert_in_range(out, lo=1000.0, hi=2000.0)
    # Attaching must NOT change the fingerprint — the session loads fine...
    session = OnnxDebugSession(token_artifact, checked)
    # ...and the rebuilt graph's assert fires on the artifact's values.
    with pytest.raises(AssertionError):
        session(_token_ids(emb), debug=True)


def test_fingerprint_mismatch_raises(token_artifact):
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 2
        from examples.adder import create_network_parts

        out_big, _ = create_network_parts()
    finally:
        adder_module.max_digits = original
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        OnnxDebugSession(token_artifact, out_big)


def test_assert_coverage_warning_on_weaker_rebuild(tmp_path, capsys):
    onnx_path = str(tmp_path / "adder_asserted.onnx")
    out, emb = _build_adder()
    compile_to_onnx(
        assert_in_range(out, lo=-1e9, hi=1e9),
        emb,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        verbose=False,
    )
    out2, _ = _build_adder()  # rebuilt WITHOUT the extra assert
    OnnxDebugSession(onnx_path, out2)
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
    target = next(i for i in model.graph.initializer if i.name == "l0_bgate")
    arr = onnx.numpy_helper.to_array(target).copy()
    arr = arr * 1.5
    arr.flat[0] += 7.0
    target.CopyFrom(onnx.numpy_helper.from_array(arr.astype(np.float32), "l0_bgate"))

    corrupt = str(tmp_path / "corrupt.onnx")
    onnx.save_model(model, corrupt)
    for ext in (".meta.json", ".debug.json"):
        shutil.copy(token_artifact[:-5] + ext, corrupt[:-5] + ext)

    out, emb = _build_adder()
    session = OnnxDebugSession(corrupt, out)
    report = probe_compiled(
        session,
        out,
        {"embedding_input": _token_ids(emb)},
        n_pos=len(TOKENS),
        atol=1e-3,
    )
    assert report.first_divergent is not None


# ---------------------------------------------------------------------------
# Fingerprint stability guards
# ---------------------------------------------------------------------------


def _tiny_graph():
    a = create_input("a", 1)
    b = create_input("b", 1)
    return add(multiply_const(a, 2.0), b)


def test_fingerprints_are_pinned():
    """The fingerprint encodings are persistence formats.

    ``graph_fingerprint`` keys every ``TW_SCHEDULE_CACHE_DIR`` entry and
    ``debug_fingerprint`` keys every ``.debug.json`` sidecar — changing
    either encoding silently invalidates all of them.  If this test
    fails, you changed the encoding: confirm that is intentional and
    update the pins (existing schedule caches will re-solve and existing
    debug sidecars will need re-export).
    """
    out = _tiny_graph()
    assert (
        debug_fingerprint(out, d=256, d_head=16)
        == "50bcb97a0852df4782908a8418bb1a34d2674af2555f1c4f3183b1e65d0e0e29"
    )
    assert (
        graph_fingerprint(
            out,
            d=256,
            d_head=16,
            d_hidden=256,
            flex_routing=True,
            cancel_slack=2,
            policy=None,
        )
        # Pin updated 2026-07 (support-aware head charge): the payload gained
        # ``linear_support`` — every Linear's live weight-row runs — because
        # the head charge became a function of weight sparsity, the first
        # schedule input the topology hash cannot see.  The layout change
        # doubles as the generation bump: every pre-support-charge
        # schedule-cache entry re-solves once under the new charge.
        == "4c72af10fd692b94a45c457e30f32393033a065b6d9ebd715f59774f8906c880"
    )


def test_fingerprint_stable_across_rebuilds_and_wrappers():
    out1 = _tiny_graph()
    out2 = _tiny_graph()  # fresh node ids
    fp1 = debug_fingerprint(out1, d=256, d_head=16)
    assert fp1 == debug_fingerprint(out2, d=256, d_head=16)
    # Assert wrappers are transparent — including at the root.
    wrapped = assert_in_range(out2, lo=-1e9, hi=1e9)
    assert fp1 == debug_fingerprint(wrapped, d=256, d_head=16)


# ---------------------------------------------------------------------------
# OnnxArtifact.debug_session — handle-built session matches direct
# ---------------------------------------------------------------------------


def test_artifact_debug_session_matches_direct(tmp_path):
    onnx_path = str(tmp_path / "model.onnx")
    out1, emb1 = _build_adder()
    artifact = compile_to_onnx(out1, emb1, onnx_path, d=D, d_head=D_HEAD, verbose=False)
    ids = _token_ids(emb1)
    iv = {"embedding_input": ids}

    out2, _ = _build_adder()
    sess_handle = artifact.debug_session(out2)
    assert isinstance(sess_handle, OnnxDebugSession)
    # onnxruntime execution floor (see _ONNX_PROBE_ATOL) — this probes the real
    # artifact, not the in-process compiled circuit.
    report = probe_compiled(
        sess_handle, out2, iv, n_pos=len(TOKENS), atol=_ONNX_PROBE_ATOL
    )
    assert report.first_divergent is None, report.format_short()

    out3, _ = _build_adder()
    sess_direct = OnnxDebugSession(onnx_path, out3)
    a = sess_handle(ids)
    b = sess_direct(ids)
    assert torch.allclose(a, b, atol=0)


# ---------------------------------------------------------------------------
# Oversized sparse initializers (the ORT >= 1.26 embedded-data ceiling)
# ---------------------------------------------------------------------------


def test_oversized_sparse_initializer_densifies_and_session_matches(
    token_artifact, monkeypatch
):
    """ORT >= 1.26 refuses any initializer whose dense size exceeds its
    embedded-data ceiling unless the data is external — and ONNX sparse
    initializers have no external form, so a production-width sparse
    ``embed_table`` (3.3 GB declared at d=8192) cannot load at all.  The
    session densifies such tensors to external data and loads by path.

    Exercised by shrinking the module ceiling so the small adder artifact's
    own sparse initializers cross it; the converted session must behave
    byte-for-byte like the normal in-memory one.
    """
    import onnx

    from torchwright.debug import onnx_debug as od

    model = onnx.load(token_artifact)
    assert len(model.graph.sparse_initializer) > 0, (
        "fixture regression: the adder artifact no longer carries sparse "
        "initializers, so this test no longer exercises the conversion"
    )

    out_ref, emb = _build_adder()
    ref = OnnxDebugSession(token_artifact, out_ref)
    ids = _token_ids(emb)
    ref_out, _ = ref.step(ids, ref.empty_past())

    monkeypatch.setattr(od, "_ORT_EMBEDDED_INITIALIZER_LIMIT", 1)
    out2, _ = _build_adder()
    converted = OnnxDebugSession(token_artifact, out2)
    assert converted._external_data_dir is not None  # conversion actually ran
    got, _ = converted.step(ids, converted.empty_past())
    torch.testing.assert_close(got, ref_out, rtol=0, atol=0)
