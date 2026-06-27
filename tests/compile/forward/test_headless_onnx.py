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


def _discover_meta(session, onnx_path):
    # past_K_i is sequence-major (cache_slots, n_heads, d_head) with a
    # SYMBOLIC slot dim — the bound prefix length S_eff (stride bucketing).
    # The full stride S comes from the sidecar meta.
    from torchwright.compiler.onnx_load import discover_cache_stride

    inputs = {inp.name: inp for inp in session.get_inputs()}
    n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
    per_layer_n_heads = [int(inputs[f"past_K_{i}"].shape[1]) for i in range(n_layers)]
    d_head = int(inputs["past_K_0"].shape[2])
    slot_dim = inputs["past_K_0"].shape[0]
    assert not isinstance(slot_dim, int), (
        f"past_K_0 first dim must be the symbolic cache_slots, got {slot_dim!r}"
    )
    with open(meta_path_for(onnx_path)) as f:
        sidecar = json.load(f)
    cache_stride = discover_cache_stride(inputs, sidecar.get("cache_stride"), onnx_path)
    return n_layers, per_layer_n_heads, d_head, cache_stride


def _zero_past(per_layer_n_heads: list, d_head: int, S: int):
    """Full static-S zero-filled cache buffers, one (k, v) pair per layer."""
    k = [np.zeros((S, nh, d_head), dtype=np.float32) for nh in per_layer_n_heads]
    v = [np.zeros((S, nh, d_head), dtype=np.float32) for nh in per_layer_n_heads]
    return k, v


def _feeds(inputs_np, past_k, past_v, base: int) -> dict:
    """Static-cache feeds: full-S past buffers + cache_position for the rows."""
    n_new = int(inputs_np.shape[0])
    feeds = {
        "inputs": inputs_np,
        "cache_position": np.arange(base, base + n_new, dtype=np.int64),
    }
    for i, (k, v) in enumerate(zip(past_k, past_v)):
        feeds[f"past_K_{i}"] = k
        feeds[f"past_V_{i}"] = v
    return feeds


def _write_deltas(past_k, past_v, results, base: int, n_new: int):
    """Persist a run's delta outputs into the cache slots [base : base+n_new)."""
    n_layers = len(past_k)
    for i in range(n_layers):
        past_k[i][base : base + n_new] = results[1 + 2 * i]
        past_v[i][base : base + n_new] = results[1 + 2 * i + 1]


def _build_sample_graph():
    """A simple multi-input graph large enough to need multiple layers."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    out = signed_multiply(a, b, max_abs1=10, max_abs2=10)
    return out, create_pos_encoding()


def _export(output_node, pos_encoding, tmpdir, name="model.onnx", trim_heads=True):
    """Export and return the OnnxArtifact (model path at ``.path``)."""
    onnx_path = os.path.join(tmpdir, name)
    return compile_headless_to_onnx(
        output_node,
        pos_encoding,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        max_seq_len=32,
        verbose=False,
        trim_heads=trim_heads,
    )


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
        onnx_path = _export(out, pos, tmpdir).path
        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)

        inputs_np = torch.cat([a_vals, b_vals], dim=1).numpy().astype(np.float32)
        past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
        onnx_out = session.run(["outputs"], _feeds(inputs_np, past_k, past_v, 0))[0]

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
        onnx_path = _export(out, pos, tmpdir).path
        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)
        out_names = ["outputs"]
        for i in range(n_layers):
            out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        # Full prefill (ground truth)
        pk_full, pv_full = _zero_past(per_layer_n_heads, d_head, S)
        full_outputs = session.run(
            ["outputs"], _feeds(inputs_np, pk_full, pv_full, 0)
        )[0]

        # Prefill 2 rows, persisting the deltas into slots [0:2)
        past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
        results = session.run(out_names, _feeds(inputs_np[:2], past_k, past_v, 0))
        _write_deltas(past_k, past_v, results, 0, 2)

        # Decode a chunk of 3 rows (base=2, n_new=3)
        chunk_out = session.run(
            ["outputs"], _feeds(inputs_np[2:5], past_k, past_v, 2)
        )[0]

    assert np.allclose(full_outputs[2:5], chunk_out, atol=1e-3), (
        f"chunked decode diff: " f"{np.abs(full_outputs[2:5] - chunk_out).max():.6f}"
    )


def test_headless_onnx_decode_step_matches_full_prefill():
    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs_np = torch.cat([a_vals, b_vals], dim=1).numpy().astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir).path
        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)
        out_names = ["outputs"]
        for i in range(n_layers):
            out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        # Full prefill
        pk_full, pv_full = _zero_past(per_layer_n_heads, d_head, S)
        full_outputs = session.run(
            ["outputs"], _feeds(inputs_np, pk_full, pv_full, 0)
        )[0]

        # Prefill 4 rows + decode 1 row
        past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
        results = session.run(out_names, _feeds(inputs_np[:4], past_k, past_v, 0))
        _write_deltas(past_k, past_v, results, 0, 4)

        decode_out = session.run(
            ["outputs"], _feeds(inputs_np[4:5], past_k, past_v, 4)
        )[0]

    assert np.allclose(
        full_outputs[-1], decode_out[0], atol=1e-3
    ), f"decode seam diff: {np.abs(full_outputs[-1] - decode_out[0]).max():.6f}"


# ---------------------------------------------------------------------------
# Test 2b: slots at positions > cache_position are INERT.
# This is the load-bearing invariant of the static cache (the INVERSE of the
# old dynamic-cache test that lived here): the runtime always binds the full
# (S, nh, d_head) allocation, and the in-graph mask must give the
# not-yet-committed tail slots softmax weight exactly 0.0 — finite garbage
# in those slots cannot change the output.  (Zero-init remains required in
# production because 0 * NaN = NaN; this test uses finite garbage.)
# ---------------------------------------------------------------------------


def test_headless_onnx_static_tail_is_inert():
    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs_np = torch.cat([a_vals, b_vals], dim=1).numpy().astype(np.float32)
    n = 4  # rows committed to the cache before the decode step

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir).path
        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)
        out_names = ["outputs"]
        for i in range(n_layers):
            out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        pk_full, pv_full = _zero_past(per_layer_n_heads, d_head, S)
        full_outputs = session.run(
            ["outputs"], _feeds(inputs_np, pk_full, pv_full, 0)
        )[0]

        # Prefill n rows into a zero cache.
        past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
        results = session.run(out_names, _feeds(inputs_np[:n], past_k, past_v, 0))
        _write_deltas(past_k, past_v, results, 0, n)

        def decode(pk, pv):
            return session.run(["outputs"], _feeds(inputs_np[n : n + 1], pk, pv, n))[0]

        # (1) Zero-tail cache reproduces row n of the full prefill (the
        # prefill/decode seam under the static mask).
        exact = decode(past_k, past_v)
        assert np.allclose(full_outputs[n], exact[0], atol=1e-3)

        # (2) Fill every slot at positions > n with finite garbage.  Slot n
        # itself is overwritten by the in-graph ScatterND (the decode row's
        # own key — the causal diagonal), so garbage there is inert too.
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


# ---------------------------------------------------------------------------
# Test 2c: prefix-window (stride-bucket) bindings are equivalent.
# The symbolic cache_slots dim lets a feeder bind any prefix cache[:S_eff]
# with base + n_new <= S_eff <= S; the in-graph mask derives its width from
# Shape(past_K_0), so every covering window must produce the same outputs.
# On CPU with the same n_new the kernels are identical, so the comparison
# is bit-level, not just allclose.
# ---------------------------------------------------------------------------


def test_headless_onnx_prefix_window_binding():
    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0], [0.0], [4.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0], [7.0], [2.0]])
    inputs_np = torch.cat([a_vals, b_vals], dim=1).numpy().astype(np.float32)
    n = 4  # committed rows before the decode step

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export(out, pos, tmpdir).path
        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, per_layer_n_heads, d_head, S = _discover_meta(session, onnx_path)
        out_names = ["outputs"]
        for i in range(n_layers):
            out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        # Prefill n rows into the full-S cache.
        past_k, past_v = _zero_past(per_layer_n_heads, d_head, S)
        results = session.run(out_names, _feeds(inputs_np[:n], past_k, past_v, 0))
        _write_deltas(past_k, past_v, results, 0, n)

        def decode_window(s_eff):
            pk = [k[:s_eff] for k in past_k]  # contiguous prefix views
            pv = [v[:s_eff] for v in past_v]
            return session.run(
                out_names, _feeds(inputs_np[n : n + 1], pk, pv, n)
            )

        # Full-S binding is the reference; every covering prefix window
        # (smallest legal = base + n_new = n + 1) must match bit-for-bit.
        ref = decode_window(S)
        for s_eff in (n + 1, n + 3, S - 1):
            got = decode_window(s_eff)
            for r, g, name in zip(ref, got, out_names):
                assert np.array_equal(r, g), (
                    f"S_eff={s_eff}: {name} differs from the full-S binding "
                    f"(max diff {np.abs(r - g).max():.6e})"
                )

        # Prefill itself also rides a window: prefill at the smallest
        # covering bucket equals prefill at full S.
        pk_a, pv_a = _zero_past(per_layer_n_heads, d_head, S)
        full = session.run(out_names, _feeds(inputs_np, pk_a, pv_a, 0))
        pk_b, pv_b = _zero_past(per_layer_n_heads, d_head, S)
        win = session.run(
            out_names,
            _feeds(
                inputs_np,
                [k[: len(inputs_np)] for k in pk_b],
                [v[: len(inputs_np)] for v in pv_b],
                0,
            ),
        )
        for r, g, name in zip(full, win, out_names):
            assert np.array_equal(r, g), f"prefill window: {name} differs"


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
        onnx_path = _export(out, pos, tmpdir).path
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
        onnx_path = _export(out, pos, tmpdir).path
        module = OnnxHeadlessModule(onnx_path)
        past = module.empty_past()
        S = module.cache_stride
        assert S == 32  # _export passes max_seq_len=32; cache_stride defaults to it
        assert past.length == 0
        assert len(past.k) == module._n_layers
        assert len(past.v) == module._n_layers
        for i, K in enumerate(past.k):
            assert K.shape == (S, module._per_layer_n_heads[i], module._d_head)
            assert (K == 0).all(), "static cache must be zero-initialized"
        for i, V in enumerate(past.v):
            assert V.shape == (S, module._per_layer_n_heads[i], module._d_head)
            assert (V == 0).all(), "static cache must be zero-initialized"


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
        onnx_path = _export(out, pos, tmpdir).path
        meta_path = meta_path_for(onnx_path)
        with open(meta_path) as f:
            data = json.load(f)

    assert data["format"] == HEADLESS_META_FORMAT
    assert data["input_names"] == ["alpha", "middle", "zebra"]
    # The full stride S travels in the sidecar (the symbolic cache_slots
    # input dim is not readable as an int).
    assert data["cache_stride"] == 32  # _export passes max_seq_len=32
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
        onnx_path = _export(out, pos, tmpdir, trim_heads=True).path
        session = onnxruntime.InferenceSession(onnx_path)
        _, per_layer_n_heads, _, _ = _discover_meta(session, onnx_path)

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
        onnx_path = _export(out, pos, tmpdir, trim_heads=False).path
        session = onnxruntime.InferenceSession(onnx_path)
        _, per_layer_n_heads, _, _ = _discover_meta(session, onnx_path)

    assert per_layer_n_heads, "no layers discovered"
    assert all(nh == max_heads for nh in per_layer_n_heads), (
        f"expected all layers at full width {max_heads}: {per_layer_n_heads}"
    )


def test_headless_onnx_trim_shrinks_mlp_slots():
    """trim_heads=True shrinks some l{i}_W1 below full d_hidden; False keeps all."""
    out, pos = _build_sample_graph()
    full_d_hidden = D  # d_hidden defaults to d when omitted
    with tempfile.TemporaryDirectory() as tmpdir:
        trim_path = _export(out, pos, tmpdir, name="trim.onnx", trim_heads=True).path
        notrim_path = _export(out, pos, tmpdir, name="notrim.onnx", trim_heads=False).path
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
        trim_path = _export(out, pos, tmpdir, name="trim.onnx", trim_heads=True).path
        notrim_path = _export(out, pos, tmpdir, name="notrim.onnx", trim_heads=False).path

        sess_trim = onnxruntime.InferenceSession(trim_path)
        sess_notrim = onnxruntime.InferenceSession(notrim_path)

        _, heads_trim, d_head, S_trim = _discover_meta(sess_trim, trim_path)
        _, heads_notrim, _, S_notrim = _discover_meta(sess_notrim, notrim_path)

        pk, pv = _zero_past(heads_trim, d_head, S_trim)
        out_trim = sess_trim.run(["outputs"], _feeds(inputs_np, pk, pv, 0))[0]

        pk, pv = _zero_past(heads_notrim, d_head, S_notrim)
        out_notrim = sess_notrim.run(["outputs"], _feeds(inputs_np, pk, pv, 0))[0]

    assert np.allclose(out_trim, out_notrim, atol=1e-4), (
        f"trim changed the output: max diff "
        f"{np.abs(out_trim - out_notrim).max():.6e}"
    )


# ---------------------------------------------------------------------------
# OnnxArtifact return handle
# ---------------------------------------------------------------------------


def test_headless_onnx_artifact_fields_and_load():
    """The exporter's OnnxArtifact matches the export params, and
    artifact.load() behaves identically to a directly constructed
    OnnxHeadlessModule."""
    from torchwright.compiler.export import debug_meta_path_for
    from torchwright.compiler.onnx_load import OnnxHeadlessModule

    out, pos = _build_sample_graph()
    a_vals = torch.tensor([[3.0], [5.0], [-2.0]])
    b_vals = torch.tensor([[4.0], [-1.0], [3.0]])
    inp = torch.cat([a_vals, b_vals], dim=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _export(out, pos, tmpdir)

        assert artifact.kind == "headless"
        assert artifact.path == os.path.join(tmpdir, "model.onnx")
        assert artifact.meta_path == meta_path_for(artifact.path)
        assert artifact.debug_path == debug_meta_path_for(artifact.path)
        assert os.path.exists(artifact.path)
        assert os.path.exists(artifact.meta_path)
        assert os.path.exists(artifact.debug_path)
        assert artifact.d == D
        assert artifact.d_head == D_HEAD
        assert artifact.cache_stride == 32  # = max_seq_len default in _export
        assert artifact.d_embed is None and artifact.vocab_size is None
        assert artifact.n_layers > 0
        assert isinstance(artifact.per_layer_n_heads, tuple)
        assert len(artifact.per_layer_n_heads) == artifact.n_layers

        loaded = artifact.load()
        assert isinstance(loaded, OnnxHeadlessModule)
        direct = OnnxHeadlessModule(artifact.path)
        assert torch.allclose(loaded(inp), direct(inp), atol=0)


def test_headless_onnx_artifact_no_debug_sidecar():
    """debug_sidecar=False -> artifact.debug_path is None, no file written."""
    from torchwright.compiler.export import debug_meta_path_for

    out, pos = _build_sample_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        artifact = compile_headless_to_onnx(
            out,
            pos,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            max_seq_len=32,
            verbose=False,
            debug_sidecar=False,
        )
        assert artifact.debug_path is None
        assert not os.path.exists(debug_meta_path_for(onnx_path))


def test_headless_onnx_optimize_kwarg_accepted():
    """compile_headless_to_onnx accepts optimize=0 (schedule parity kwarg)."""
    out, pos = _build_sample_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        artifact = compile_headless_to_onnx(
            out,
            pos,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            max_seq_len=32,
            verbose=False,
            optimize=0,
        )
        assert os.path.exists(artifact.path)


def test_load_onnx_dispatches_headless():
    """load_onnx on a headless export returns an OnnxHeadlessModule;
    a doctored sidecar format raises a loud ValueError."""
    from torchwright.compiler.onnx_load import OnnxHeadlessModule, load_onnx

    out, pos = _build_sample_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = _export(out, pos, tmpdir)
        model = load_onnx(artifact.path)
        assert isinstance(model, OnnxHeadlessModule)

        # Doctor the format key -> loud error naming both expected values.
        with open(artifact.meta_path) as f:
            meta = json.load(f)
        meta["format"] = "torchwright.bogus.v9"
        with open(artifact.meta_path, "w") as f:
            json.dump(meta, f)
        with pytest.raises(ValueError, match="bogus"):
            load_onnx(artifact.path)

        # Missing sidecar -> FileNotFoundError with re-export hint.
        os.remove(artifact.meta_path)
        with pytest.raises(FileNotFoundError, match="Re-export"):
            load_onnx(artifact.path)
