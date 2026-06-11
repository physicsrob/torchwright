"""Windowed-cache (attention-sink + sliding-window) protocol tests.

``cache_window=C`` exports a model whose committed KV cache is a fixed
C-slot host-managed window: every pass binds ``past_K_i`` exactly
``(C + n_new, nh, d_head)``, new rows scatter in-graph to the staging
tail ``[C, C+n_new)``, committed slot ``j`` is visible iff
``j < cache_position[0]``, and eviction is physical overwrite by the
host (no masking, no position metadata per slot).  See the module
docstring of ``torchwright/compiler/export.py``.

The toy graph is ``attend_most_recent_matching`` over a 3-class one-hot
key space — the smallest graph whose attention reads actually span
positions, so the window either preserves or breaks them:

  - class SINK is published only by the prefill rows (the host pins
    those in the sink slots ``[0, P)`` — a read of SINK late in the
    rollout is an arbitrarily-long-span read that must keep working);
  - classes A/B alternate during the rollout (a read of "the other
    class" has span <= 2 — always inside the ring);
  - the eviction test publishes A only early, then queries it after
    the ring has wrapped past every A row — the windowed output MUST
    diverge from the unbounded reference there (proving slots really
    evaporate; without this the equivalence tests could pass with a
    mask that never blocks anything).

Host policy under test (the DOOM ring plan): sink prefix ``[0, P)``
written by prefill at identity slots, ring ``[P, P+W)`` written at
``P + (pos - P) % W``.  This fills committed slots in slot order until
all C are written once — the contract the ``j < base`` writtenness
comparison requires.

Equivalence to the unbounded export is CONDITIONAL by design: it holds
iff every read's span fits what the host keeps resident.  The tests pin
both directions of that conditional.
"""

import json
import os

import numpy as np
import pytest

from torchwright.compiler.export import (
    _resolve_cache_stride,
    compile_headless_to_onnx,
    meta_path_for,
)
from torchwright.ops.attention_ops import attend_most_recent_matching
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

onnxruntime = pytest.importorskip("onnxruntime")
import onnx  # noqa: E402  (after the importorskip gate, like onnxruntime)

D = 256
D_HEAD = 16
MAXSEQ = 64

P = 4  # sink slots (the prefill length)
W = 6  # ring slots
C = P + W  # cache_window: committed slots
T = 28  # total rows incl. prefill -> (T - P) / W = 4 ring wraps

SINK, A, B = 0, 1, 2  # key classes (one-hot over 3 dims)


def _onehot(i: int) -> list:
    return [1.0 if j == i else 0.0 for j in range(3)]


def _build_toy_graph():
    """out[t] = value at the most recent position whose key matches q[t]."""
    q = create_input("q", 3)
    k = create_input("k", 3)
    v = create_input("v", 1)
    pos = create_pos_encoding()
    # match_gain * 1 (one-hot dot) >> _QUERY_GAIN * MAXSEQ = 512: hard
    # selection at every span this file uses.
    out = attend_most_recent_matching(pos, q, k, v, match_gain=50000.0)
    return out, pos


def _pack_rows(rows) -> np.ndarray:
    """(q, k, v) rows -> the model's ``inputs`` array.

    input_names are sorted alphabetically by the exporter: k, q, v.
    """
    k = np.array([r[1] for r in rows], dtype=np.float32)
    q = np.array([r[0] for r in rows], dtype=np.float32)
    v = np.array([[r[2]] for r in rows], dtype=np.float32)
    return np.concatenate([k, q, v], axis=1)


def _standard_sequence() -> np.ndarray:
    """Sink rows publish SINK; rollout alternates A/B keys; every 5th row
    queries SINK (long span, sink-resident), the rest query the other
    class (span <= 2, ring-resident)."""
    rows = [(_onehot(SINK), _onehot(SINK), 100.0 + i) for i in range(P)]
    for t in range(P, T):
        key = A if t % 2 == 0 else B
        qry = SINK if t % 5 == 0 else (A if key == B else B)
        rows.append((_onehot(qry), _onehot(key), 200.0 + t))
    return _pack_rows(rows)


def _eviction_sequence() -> np.ndarray:
    """A is published ONLY at rows 4..7; rows >= 20 query A — by then the
    ring (last W=6 rows) holds no A row, so the windowed model cannot see
    one while the unbounded reference still can."""
    rows = [(_onehot(SINK), _onehot(SINK), 100.0 + i) for i in range(P)]
    for t in range(P, T):
        key = A if t < 8 else B
        qry = A if t >= 20 else SINK
        rows.append((_onehot(qry), _onehot(key), 200.0 + t))
    return _pack_rows(rows)


# ---------------------------------------------------------------------------
# One export per protocol, shared by every test in the module (the toy
# compiles in seconds, but twice is enough).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    out, pos = _build_toy_graph()
    tmpdir = str(tmp_path_factory.mktemp("windowed_cache"))
    unbounded = os.path.join(tmpdir, "unbounded.onnx")
    windowed = os.path.join(tmpdir, "windowed.onnx")
    compile_headless_to_onnx(
        out, pos, unbounded, d=D, d_head=D_HEAD, max_seq_len=MAXSEQ, verbose=False
    )
    compile_headless_to_onnx(
        out,
        pos,
        windowed,
        d=D,
        d_head=D_HEAD,
        max_seq_len=MAXSEQ,
        verbose=False,
        cache_window=C,
    )
    sess_u = onnxruntime.InferenceSession(unbounded)
    sess_w = onnxruntime.InferenceSession(windowed)
    inputs = {i.name: i for i in sess_u.get_inputs()}
    n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
    heads = [int(inputs[f"past_K_{i}"].shape[1]) for i in range(n_layers)]
    d_head = int(inputs["past_K_0"].shape[2])
    out_names = ["outputs"]
    for i in range(n_layers):
        out_names += [f"delta_K_{i}", f"delta_V_{i}"]
    return {
        "unbounded_path": unbounded,
        "windowed_path": windowed,
        "sess_u": sess_u,
        "sess_w": sess_w,
        "n_layers": n_layers,
        "heads": heads,
        "d_head": d_head,
        "out_names": out_names,
    }


# ---------------------------------------------------------------------------
# Protocol drivers (the host side of each protocol, in miniature)
# ---------------------------------------------------------------------------


def _run_unbounded(m, inputs_np: np.ndarray) -> np.ndarray:
    """Full prefill + width-1 decodes under the standard static protocol."""
    nl, heads, dh = m["n_layers"], m["heads"], m["d_head"]
    pk = [np.zeros((MAXSEQ, nh, dh), np.float32) for nh in heads]
    pv = [np.zeros((MAXSEQ, nh, dh), np.float32) for nh in heads]

    def feeds(x, base):
        f = {
            "inputs": x,
            "cache_position": np.arange(base, base + len(x), dtype=np.int64),
        }
        for i in range(nl):
            f[f"past_K_{i}"], f[f"past_V_{i}"] = pk[i], pv[i]
        return f

    total = len(inputs_np)
    got = np.zeros((total, 1), np.float32)
    r = m["sess_u"].run(m["out_names"], feeds(inputs_np[:P], 0))
    got[:P] = r[0]
    for i in range(nl):
        pk[i][0:P], pv[i][0:P] = r[1 + 2 * i], r[2 + 2 * i]
    for t in range(P, total):
        r = m["sess_u"].run(m["out_names"], feeds(inputs_np[t : t + 1], t))
        got[t] = r[0]
        for i in range(nl):
            pk[i][t : t + 1], pv[i][t : t + 1] = r[1 + 2 * i], r[2 + 2 * i]
    return got


class _WindowedHost:
    """The windowed host in miniature: a (C, nh, dh) committed region per
    layer, sink+ring persistence, exact ``C + n_new`` bindings."""

    def __init__(self, m):
        self.m = m
        nl, heads, dh = m["n_layers"], m["heads"], m["d_head"]
        self.ck = [np.zeros((C, nh, dh), np.float32) for nh in heads]
        self.cv = [np.zeros((C, nh, dh), np.float32) for nh in heads]

    def feeds(self, x, base, staging_fill: float = 0.0):
        nl, heads, dh = self.m["n_layers"], self.m["heads"], self.m["d_head"]
        n = len(x)
        f = {
            "inputs": x,
            "cache_position": np.arange(base, base + n, dtype=np.int64),
        }
        for i in range(nl):
            stage_k = np.full((n, heads[i], dh), staging_fill, np.float32)
            stage_v = np.full((n, heads[i], dh), staging_fill, np.float32)
            f[f"past_K_{i}"] = np.concatenate([self.ck[i], stage_k])
            f[f"past_V_{i}"] = np.concatenate([self.cv[i], stage_v])
        return f

    @staticmethod
    def slot_for(pos: int) -> int:
        return pos if pos < P else P + (pos - P) % W

    def persist(self, results, base: int, n_new: int):
        for row in range(n_new):
            slot = self.slot_for(base + row)
            for i in range(self.m["n_layers"]):
                self.ck[i][slot] = results[1 + 2 * i][row]
                self.cv[i][slot] = results[2 + 2 * i][row]

    def run_pass(self, x, base, **feed_kw):
        r = self.m["sess_w"].run(self.m["out_names"], self.feeds(x, base, **feed_kw))
        self.persist(r, base, len(x))
        return r[0]


def _run_windowed(m, inputs_np: np.ndarray, schedule) -> np.ndarray:
    host = _WindowedHost(m)
    got = np.zeros((len(inputs_np), 1), np.float32)
    for base, n in schedule:
        got[base : base + n] = host.run_pass(inputs_np[base : base + n], base)
    return got


def _mixed_schedule(total: int) -> list:
    """Chunked prefill (2+2) + decode in mixed widths 3 and 1 — multi-row
    staging passes repeatedly straddle the ring-wrap boundary."""
    schedule = [(0, 2), (2, 2)]
    t = P
    while t < total:
        n = 3 if (t - P) % 4 == 0 and t + 3 <= total else 1
        schedule.append((t, n))
        t += n
    return schedule


# ---------------------------------------------------------------------------
# Test 1: windowed == unbounded while every read span fits (sink reads +
# span-2 ring reads), across 4 ring wraps, chunked prefill, mixed widths.
# ---------------------------------------------------------------------------


def test_windowed_matches_unbounded_across_wraps(models):
    inputs_np = _standard_sequence()
    ref = _run_unbounded(models, inputs_np)
    got = _run_windowed(models, inputs_np, _mixed_schedule(T))
    assert np.allclose(ref, got, atol=1e-3), (
        f"windowed diverged from unbounded with all spans in-window: "
        f"max diff {np.abs(ref - got).max():.6f} at row "
        f"{int(np.abs(ref - got).argmax())}"
    )


# ---------------------------------------------------------------------------
# Test 2: eviction is real — a read whose span exceeds the window MUST
# diverge from the unbounded reference (and the rows before it must not).
# ---------------------------------------------------------------------------


def test_windowed_eviction_is_real(models):
    inputs_np = _eviction_sequence()
    ref = _run_unbounded(models, inputs_np)
    schedule = [(0, P)] + [(t, 1) for t in range(P, T)]
    got = _run_windowed(models, inputs_np, schedule)

    # Rows before any out-of-window read agree (sink reads only).
    assert np.allclose(ref[:20], got[:20], atol=1e-3), (
        f"pre-eviction rows diverged: max diff "
        f"{np.abs(ref[:20] - got[:20]).max():.6f}"
    )
    # Rows >= 20 query class A; every A row (4..7) left the ring long ago.
    # Unbounded still returns an A value (200+4 .. 200+7); windowed cannot.
    tail_diff = np.abs(ref[20:] - got[20:]).max()
    assert tail_diff > 1.0, (
        f"out-of-window read did NOT diverge (max tail diff {tail_diff:.6f}) "
        f"— eviction is not actually happening, the window is a no-op"
    )
    # And the unbounded reference really did find the stale A row.
    ref_val = float(ref[20, 0])
    assert 203.0 < ref_val < 208.5, f"reference sanity: {ref_val}"


# ---------------------------------------------------------------------------
# Test 3: unwritten committed slots and the staging region of the binding
# are inert — finite garbage there cannot change the output.  (The windowed
# analog of the static-tail inertness test in test_headless_onnx.py.)
# ---------------------------------------------------------------------------


def test_windowed_unwritten_and_staging_slots_inert(models):
    inputs_np = _standard_sequence()

    # Drive to base=7: committed slots [0,7) written, [7,10) never written.
    host = _WindowedHost(models)
    host.run_pass(inputs_np[:P], 0)
    for t in range(P, 7):
        host.run_pass(inputs_np[t : t + 1], t)

    clean = models["sess_w"].run(
        models["out_names"], host.feeds(inputs_np[7:8], 7)
    )[0]

    # Garbage in the never-written committed slots...
    dirty_host = _WindowedHost(models)
    for i in range(models["n_layers"]):
        dirty_host.ck[i][:] = host.ck[i]
        dirty_host.cv[i][:] = host.cv[i]
        dirty_host.ck[i][7:] = 7.0
        dirty_host.cv[i][7:] = -3.0
    # ...and garbage in the binding's staging tail (the in-graph ScatterND
    # fully overwrites it in the transient before attention reads it).
    dirty = models["sess_w"].run(
        models["out_names"], dirty_host.feeds(inputs_np[7:8], 7, staging_fill=9.0)
    )[0]

    assert np.allclose(clean, dirty, atol=1e-6), (
        f"garbage in masked windowed-cache slots changed the output "
        f"(max diff {np.abs(clean - dirty).max():.6e}) — the writtenness "
        f"mask or the staging scatter is leaking"
    )


# ---------------------------------------------------------------------------
# Test 4: a binding that is not exactly C + n_new wide fails loudly.
# (The windowed mask is built from constants + cache_position, so a wrong
# width must crash at the mask broadcast or the scatter — never silently
# mis-attend.)
# ---------------------------------------------------------------------------


def test_windowed_wrong_binding_width_fails_loudly(models):
    host = _WindowedHost(models)
    inputs_np = _standard_sequence()
    x = inputs_np[:P]
    base_feeds = host.feeds(x, 0)

    for extra in (1, P):  # too wide by 1; and a "prefix-style" C-only bind
        bad = dict(base_feeds)
        for i in range(models["n_layers"]):
            k = base_feeds[f"past_K_{i}"]
            v = base_feeds[f"past_V_{i}"]
            if extra == P:
                bad[f"past_K_{i}"], bad[f"past_V_{i}"] = k[:C], v[:C]
            else:
                pad = np.zeros((extra,) + k.shape[1:], np.float32)
                bad[f"past_K_{i}"] = np.concatenate([k, pad])
                bad[f"past_V_{i}"] = np.concatenate([v, pad])
        with pytest.raises(Exception):
            models["sess_w"].run(models["out_names"], bad)


# ---------------------------------------------------------------------------
# Test 5: sidecar + graph-shape pins, and config validation.
# ---------------------------------------------------------------------------


def test_windowed_sidecar_and_graph_pins(models):
    with open(meta_path_for(models["windowed_path"])) as f:
        meta_w = json.load(f)
    assert meta_w["cache_window"] == C
    assert meta_w["cache_stride"] == C  # arange_S is baked at length C

    with open(meta_path_for(models["unbounded_path"])) as f:
        meta_u = json.load(f)
    assert "cache_window" not in meta_u  # default protocol: no discriminator

    gw = onnx.load(models["windowed_path"]).graph
    gu = onnx.load(models["unbounded_path"]).graph
    w_ops = {n.op_type for n in gw.node}
    w_outs = {o for n in gw.node for o in n.output}
    u_outs = {o for n in gu.node for o in n.output}

    # Windowed mode: no Shape chain at all (capture cleanliness), and the
    # layers scatter via the staging indices.
    assert "Shape" not in w_ops
    assert "_write_slot_col" in w_outs
    w_scatter_idx = {n.input[1] for n in gw.node if n.op_type == "ScatterND"}
    assert w_scatter_idx == {"_write_slot_col"}

    # Default emission is unchanged: Shape-derived mask width, scatter at
    # cache_position, no windowed tensors anywhere.
    assert any(n.op_type == "Shape" for n in gu.node)
    assert {"_pastK0_shape", "_s_eff_1d", "_slots"} <= u_outs
    u_scatter_idx = {n.input[1] for n in gu.node if n.op_type == "ScatterND"}
    assert u_scatter_idx == {"_cache_pos_col"}
    assert not any(name.startswith("_write_slot") for name in u_outs)
    assert not any(name.startswith("_slot_pos") for name in u_outs)


def test_windowed_config_validation():
    # Mutually exclusive with cache_stride.
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_cache_stride(16, MAXSEQ, cache_window=C)
    # A window wider than the position space can never fill.
    with pytest.raises(ValueError, match="cache_window"):
        _resolve_cache_stride(None, MAXSEQ, cache_window=MAXSEQ + 1)
    with pytest.raises(ValueError, match="cache_window"):
        _resolve_cache_stride(None, MAXSEQ, cache_window=0)
    # Valid configs resolve to C itself (arange_S length == C).
    assert _resolve_cache_stride(None, MAXSEQ, cache_window=C) == C
