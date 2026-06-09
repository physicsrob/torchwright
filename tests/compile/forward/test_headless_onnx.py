"""KV-cache protocol tests for the headless streaming cached ONNX exporter.

Covers prefill, decode, the OnnxHeadlessModule.step API, the dynamic
causal mask seam, and the meta sidecar schema.  Basic compute()<->ONNX
parity lives in test_headless_module.py.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from torchwright.compiler.export import (
    HEADLESS_META_FORMAT,
    meta_path_for,
    compile_headless_to_onnx,
)
from torchwright.compiler.forward.compile import forward_compile
from torchwright.ops.arithmetic_ops import add, signed_multiply
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

onnxruntime = pytest.importorskip("onnxruntime")

D = 256
D_HEAD = 16


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


def _build_sample_graph():
    """A simple multi-input graph large enough to need multiple layers."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    out = signed_multiply(a, b, max_abs1=10, max_abs2=10)
    return out, create_pos_encoding()


def _export(output_node, pos_encoding, tmpdir, name="model.onnx", trim_heads=True):
    onnx_path = os.path.join(tmpdir, name)
    compile_headless_to_onnx(
        output_node,
        pos_encoding,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        max_seq_len=32,
        verbose=False,
        trim_heads=trim_heads,
    )
    return onnx_path


def _l_w1_d_hidden(onnx_path):
    """Per-layer MLP hidden widths read straight off the ONNX initializers.

    ``l{i}_W1`` is the first MLP weight, emitted with shape ``(d, d_hidden_i)``
    (it is the LHS of ``MatMul(res, W1)`` where ``res`` is ``(t, d)``), so its
    second dim is that layer's hidden width after trimming.  A mostly-zero W1 is
    emitted as a SparseTensorProto, so look in both the dense and sparse init
    lists (the sparse proto's ``dims`` still carries the full dense shape and its
    name lives on ``values.name``).
    """
    import onnx

    model = onnx.load(onnx_path)
    dims_by_name = {init.name: list(init.dims) for init in model.graph.initializer}
    for sp in model.graph.sparse_initializer:
        dims_by_name[sp.values.name] = list(sp.dims)

    widths = []
    i = 0
    while f"l{i}_W1" in dims_by_name:
        dims = dims_by_name[f"l{i}_W1"]
        assert dims[0] == D, f"l{i}_W1 first dim {dims[0]} != d {D}"
        widths.append(int(dims[1]))
        i += 1
    assert widths, "no l{i}_W1 initializers found"
    return widths


# ---------------------------------------------------------------------------
# Test 1: Prefill on full sequence matches compute() reference
# ---------------------------------------------------------------------------


def test_headless_onnx_prefill_matches_compute():
    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0]])

    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        pos_encoding=pos,
        verbose=False,
    )
    expected = (
        net.compute(n_pos=4, input_values={"a": a_vals, "b": b_vals})[out].cpu().numpy()
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir)
        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head = _discover_meta(session)

        inputs_np = torch.cat([a_vals, b_vals], dim=1).numpy().astype(np.float32)
        feeds = {"inputs": inputs_np}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        onnx_out = session.run(["outputs"], feeds)[0]

    assert np.allclose(
        onnx_out, expected, atol=1e-3
    ), f"prefill diff: {np.abs(onnx_out - expected).max():.6f}"


# ---------------------------------------------------------------------------
# Test 2: Decode step matches full prefill (dynamic-mask seam)
# ---------------------------------------------------------------------------


def test_headless_onnx_chunked_decode_matches_full_prefill():
    """Prefill 2 rows, then decode a 3-row chunk.  Exercises the dynamic
    mask at past_len>0 and n_new>1, a combination the single-row decode
    test does not cover.
    """
    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs_np = torch.cat([a_vals, b_vals], dim=1).numpy().astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir)
        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head = _discover_meta(session)
        out_names = ["outputs"]
        for i in range(n_layers):
            out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        # Full prefill (ground truth)
        feeds = {"inputs": inputs_np}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        full_outputs = session.run(["outputs"], feeds)[0]

        # Prefill 2 rows
        feeds = {"inputs": inputs_np[:2]}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        results = session.run(out_names, feeds)
        past_K = [results[1 + 2 * i] for i in range(n_layers)]
        past_V = [results[1 + 2 * i + 1] for i in range(n_layers)]

        # Decode a chunk of 3 rows (past_len=2, n_new=3)
        feeds = {
            "inputs": inputs_np[2:5],
            "past_len": np.array(2, dtype=np.int64),
        }
        for i in range(n_layers):
            feeds[f"past_K_{i}"] = past_K[i]
            feeds[f"past_V_{i}"] = past_V[i]
        chunk_out = session.run(["outputs"], feeds)[0]

    assert np.allclose(full_outputs[2:5], chunk_out, atol=1e-3), (
        f"chunked decode diff: " f"{np.abs(full_outputs[2:5] - chunk_out).max():.6f}"
    )


def test_headless_onnx_decode_step_matches_full_prefill():
    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs_np = torch.cat([a_vals, b_vals], dim=1).numpy().astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir)
        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head = _discover_meta(session)
        out_names = ["outputs"]
        for i in range(n_layers):
            out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        # Full prefill
        feeds = {"inputs": inputs_np}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        full_outputs = session.run(["outputs"], feeds)[0]

        # Prefill 4 rows + decode 1 row
        feeds = {"inputs": inputs_np[:4]}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        results = session.run(out_names, feeds)
        past_K = [results[1 + 2 * i] for i in range(n_layers)]
        past_V = [results[1 + 2 * i + 1] for i in range(n_layers)]

        feeds = {
            "inputs": inputs_np[4:5],
            "past_len": np.array(4, dtype=np.int64),
        }
        for i in range(n_layers):
            feeds[f"past_K_{i}"] = past_K[i]
            feeds[f"past_V_{i}"] = past_V[i]
        decode_out = session.run(["outputs"], feeds)[0]

    assert np.allclose(
        full_outputs[-1], decode_out[0], atol=1e-3
    ), f"decode seam diff: {np.abs(full_outputs[-1] - decode_out[0]).max():.6f}"


# ---------------------------------------------------------------------------
# Test 2b: the causal mask follows the BOUND past_K shape, not past_len.
# This is the load-bearing invariant of the in-place (delta) cache: the
# runtime binds past_K = cache[:cache_len] (a contiguous prefix of a larger
# owned buffer), so the mask geometry must come from the bound seq-dim, and
# binding the whole allocation would attend to the dead tail.
# ---------------------------------------------------------------------------


def test_headless_onnx_mask_follows_bound_cache_shape():
    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs_np = torch.cat([a_vals, b_vals], dim=1).numpy().astype(np.float32)
    n = 4  # rows committed to the cache before the decode step

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir)
        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head = _discover_meta(session)
        out_names = ["outputs"]
        for i in range(n_layers):
            out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        feeds = {"inputs": inputs_np}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        full_outputs = session.run(["outputs"], feeds)[0]

        # Prefill n rows from empty: the deltas ARE the n-row cache (seq-major).
        feeds = {"inputs": inputs_np[:n]}
        feeds.update(_empty_past_feeds(per_layer_n_heads, d_head))
        results = session.run(out_names, feeds)
        cache_k = [results[1 + 2 * i] for i in range(n_layers)]
        cache_v = [results[1 + 2 * i + 1] for i in range(n_layers)]

        def decode(past_k, past_v, past_len):
            feeds = {
                "inputs": inputs_np[n : n + 1],
                "past_len": np.array(past_len, dtype=np.int64),
            }
            for i in range(n_layers):
                feeds[f"past_K_{i}"] = past_k[i]
                feeds[f"past_V_{i}"] = past_v[i]
            return session.run(["outputs"], feeds)[0]

        # (1) Exact n-row past reproduces row n of the full prefill.
        exact = decode(cache_k, cache_v, n)
        assert np.allclose(full_outputs[n], exact[0], atol=1e-3)

        # Allocate a larger buffer; [:n] is the real cache, the tail is garbage.
        m = n + 3
        big_k, big_v = [], []
        for i in range(n_layers):
            nh = per_layer_n_heads[i]
            bk = np.full((m, nh, d_head), 7.0, dtype=np.float32)
            bv = np.full((m, nh, d_head), -3.0, dtype=np.float32)
            bk[:n] = cache_k[i]
            bv[:n] = cache_v[i]
            big_k.append(bk)
            big_v.append(bv)

        # (1') Binding the contiguous prefix VIEW big[:n] is bit-equal — the
        # owned-cache in-place pattern is sound.
        sliced = decode([bk[:n] for bk in big_k], [bv[:n] for bv in big_v], n)
        assert np.allclose(
            exact[0], sliced[0], atol=1e-6
        ), f"prefix-slice bind diverged: {np.abs(exact[0] - sliced[0]).max():.6e}"

        # (2) Binding the WHOLE buffer (past_len=n) attends to the dead tail —
        # the mask spans m, not n — so it must differ. This is why the runtime
        # binds cache[:cache_len], never the full allocation.
        full_buf = decode(big_k, big_v, n)
        assert not np.allclose(exact[0], full_buf[0], atol=1e-3), (
            "binding the full allocation matched the prefix bind — the mask is "
            "not following the bound past_K shape"
        )


# ---------------------------------------------------------------------------
# Test 3: OnnxHeadlessModule.step API threads the cache correctly
# ---------------------------------------------------------------------------


def test_onnx_headless_module_step_matches_full_call():
    from torchwright.compiler.onnx_load import OnnxHeadlessModule

    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs = torch.cat([a_vals, b_vals], dim=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir)
        module = OnnxHeadlessModule(onnx_path)

        # Independent call: prefill full sequence, drop cache
        full = module(inputs)

        # Stateful call: prefill 4, then decode 1
        past = module.empty_past()
        prefill_out, past = module.step(inputs[:4], past)
        decode_out, past = module.step(inputs[4:5], past)

    assert torch.allclose(
        full[:4], prefill_out, atol=1e-3
    ), f"prefill portion diff: {(full[:4] - prefill_out).abs().max().item():.6f}"
    assert torch.allclose(
        full[4], decode_out[0], atol=1e-3
    ), f"decode row diff: {(full[4] - decode_out[0]).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# Test 4: empty_past has the right shape
# ---------------------------------------------------------------------------


def test_onnx_headless_module_empty_past_shape():
    from torchwright.compiler.onnx_load import OnnxHeadlessModule

    out, pos = _build_sample_graph()
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir)
        module = OnnxHeadlessModule(onnx_path)
        past_K, past_V = module.empty_past()
        assert len(past_K) == module._n_layers
        assert len(past_V) == module._n_layers
        for i, K in enumerate(past_K):
            assert K.shape == (0, module._per_layer_n_heads[i], module._d_head)
        for i, V in enumerate(past_V):
            assert V.shape == (0, module._per_layer_n_heads[i], module._d_head)


# ---------------------------------------------------------------------------
# Test 5: Meta sidecar schema + input name ordering
# ---------------------------------------------------------------------------


def test_headless_onnx_sidecar_schema():
    zebra = create_input("zebra", 1)
    alpha = create_input("alpha", 1)
    middle = create_input("middle", 1)
    out = add(add(zebra, alpha), middle)
    pos = create_pos_encoding()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir)
        meta_path = meta_path_for(onnx_path)
        with open(meta_path) as f:
            data = json.load(f)

    assert data["format"] == HEADLESS_META_FORMAT
    assert data["input_names"] == ["alpha", "middle", "zebra"]
    # The sidecar should not carry any "cached" discriminator — there's
    # only one protocol now.
    assert "cached" not in data


# ---------------------------------------------------------------------------
# Test 6: CompiledHeadless.step (in-memory cached path)
# ---------------------------------------------------------------------------


def test_compiled_headless_step_matches_call():
    """step(inputs, empty_past) on a full sequence == module(inputs)."""
    from torchwright.compiler.export import compile_headless

    out, pos = _build_sample_graph()
    module = compile_headless(out, pos, d=D, d_head=D_HEAD, verbose=False)

    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs = torch.cat([a_vals, b_vals], dim=1)

    with torch.no_grad():
        full = module(inputs)
        step_out, _ = module.step(inputs, module.empty_past())

    assert torch.allclose(
        full, step_out, atol=1e-4
    ), f"step diff: {(full - step_out).abs().max().item():.6f}"


def test_compiled_headless_step_prefill_decode_matches_full():
    """Prefill 4 + decode 1 matches a single full-sequence forward."""
    from torchwright.compiler.export import compile_headless

    out, pos = _build_sample_graph()
    module = compile_headless(out, pos, d=D, d_head=D_HEAD, verbose=False)

    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs = torch.cat([a_vals, b_vals], dim=1)

    with torch.no_grad():
        full = module(inputs)
        past = module.empty_past()
        prefill_out, past = module.step(inputs[:4], past)
        decode_out, past = module.step(inputs[4:5], past)

    assert torch.allclose(
        full[:4], prefill_out, atol=1e-4
    ), f"prefill diff: {(full[:4] - prefill_out).abs().max().item():.6f}"
    assert torch.allclose(
        full[4], decode_out[0], atol=1e-4
    ), f"decode diff: {(full[4] - decode_out[0]).abs().max().item():.6f}"
    # past_K should have grown to n_total = 5
    past_K, _ = past
    assert past_K[0].shape[1] == 5


# ---------------------------------------------------------------------------
# Test 7: trim_heads actually trims the exported ONNX (heads + MLP slots)
# ---------------------------------------------------------------------------
#
# The streaming exporter used to leave every layer full-width (d/d_head heads,
# full d_hidden MLP), merely sparsifying the unused heads/slots to zero.  These
# tests pin that ``trim_heads=True`` genuinely shrinks the per-layer KV cache
# (past_K_i widths) and MLP MatMuls (l{i}_W1 widths), that ``trim_heads=False``
# preserves the full width, and that trimming is a numerical no-op (only all-zero
# heads/slots are removed).


def test_headless_onnx_trim_heads_shrinks_kv_cache():
    """trim_heads=True: per-layer past_K widths are below the full head count."""
    out, pos = _build_sample_graph()
    max_heads = D // D_HEAD  # 16
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir, trim_heads=True)
        session = onnxruntime.InferenceSession(onnx_path)
        _, per_layer_n_heads, _ = _discover_meta(session)

    assert per_layer_n_heads, "no layers discovered"
    assert all(1 <= nh <= max_heads for nh in per_layer_n_heads)
    assert min(per_layer_n_heads) < max_heads, (
        f"no layer trimmed below full width {max_heads}: {per_layer_n_heads}"
    )
    # A real per-layer trim leaves the layers non-uniform (different layers use
    # different head counts); a coincidental uniform value would still satisfy
    # the bound above, so assert the per-layer variation explicitly.
    assert len(set(per_layer_n_heads)) > 1, (
        f"head counts are uniform across layers: {per_layer_n_heads}"
    )


def test_headless_onnx_no_trim_preserves_full_width():
    """trim_heads=False: every layer keeps the full d/d_head head count."""
    out, pos = _build_sample_graph()
    max_heads = D // D_HEAD  # 16
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir, trim_heads=False)
        session = onnxruntime.InferenceSession(onnx_path)
        _, per_layer_n_heads, _ = _discover_meta(session)

    assert per_layer_n_heads, "no layers discovered"
    assert all(nh == max_heads for nh in per_layer_n_heads), (
        f"expected all layers at full width {max_heads}: {per_layer_n_heads}"
    )


def test_headless_onnx_trim_shrinks_mlp_slots():
    """trim_heads=True shrinks some l{i}_W1 below full d_hidden; False keeps all."""
    out, pos = _build_sample_graph()
    full_d_hidden = D  # d_hidden defaults to d when omitted
    with tempfile.TemporaryDirectory() as tmpdir:
        trim_path = _export(out, pos, tmpdir, name="trim.onnx", trim_heads=True)
        notrim_path = _export(out, pos, tmpdir, name="notrim.onnx", trim_heads=False)
        trim_widths = _l_w1_d_hidden(trim_path)
        notrim_widths = _l_w1_d_hidden(notrim_path)

    assert all(1 <= w <= full_d_hidden for w in trim_widths)
    assert min(trim_widths) < full_d_hidden, (
        f"no layer's MLP trimmed below full d_hidden {full_d_hidden}: {trim_widths}"
    )
    assert all(w == full_d_hidden for w in notrim_widths), (
        f"expected all layers at full d_hidden {full_d_hidden}: {notrim_widths}"
    )


def test_headless_onnx_trim_is_numerical_noop():
    """Trimmed and full-width ONNX produce identical output on the same prefill.

    Trimming removes only all-zero heads/slots, so prefilling the same rows
    through both sessions must agree — any divergence means a non-zero head or
    slot was trimmed (a real bug).
    """
    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs_np = torch.cat([a_vals, b_vals], dim=1).numpy().astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        trim_path = _export(out, pos, tmpdir, name="trim.onnx", trim_heads=True)
        notrim_path = _export(out, pos, tmpdir, name="notrim.onnx", trim_heads=False)

        sess_trim = onnxruntime.InferenceSession(trim_path)
        sess_notrim = onnxruntime.InferenceSession(notrim_path)

        _, heads_trim, d_head = _discover_meta(sess_trim)
        _, heads_notrim, _ = _discover_meta(sess_notrim)

        feeds_trim = {"inputs": inputs_np}
        feeds_trim.update(_empty_past_feeds(heads_trim, d_head))
        out_trim = sess_trim.run(["outputs"], feeds_trim)[0]

        feeds_notrim = {"inputs": inputs_np}
        feeds_notrim.update(_empty_past_feeds(heads_notrim, d_head))
        out_notrim = sess_notrim.run(["outputs"], feeds_notrim)[0]

    assert np.allclose(out_trim, out_notrim, atol=1e-4), (
        f"trim changed the output: max diff "
        f"{np.abs(out_trim - out_notrim).max():.6e}"
    )
