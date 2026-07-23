"""Debug session over a compiled ONNX artifact.

:class:`OnnxDebugSession` gives the ONNX export the same debug surface
as the in-process :class:`~torchwright.compiler.export.CompiledHeadless`
— ``step(..., debug=True)`` (residual self-consistency + attached-check
predicates), ``debug_value(node)``, and the probe suite in
:mod:`torchwright.debug.probe` — **without recompiling**.  It runs the
real artifact: the per-layer residual streams already exist as named
internal tensors in the emitted ONNX graph (``res_0``,
``l{i}_res_attn``, ``l{i}_res_next``, plus ``l{i}_weights`` /
``l{i}_logits_masked`` for attention probing), so the session promotes
them to graph outputs and fetches them from an ordinary onnxruntime
run.  Because the artifact itself executes, this path also catches
ONNX-emission and execution-provider bugs that an in-process recompile
is structurally blind to.

What it needs besides the ``.onnx`` file:

* the ``<stem>.debug.json`` sidecar written by
  :func:`~torchwright.compiler.export.compile_to_onnx`
  (``debug_sidecar=True``, the default) — the residual assignment keyed
  by canonical node id, plus a structural fingerprint;
* the **rebuilt graph** (``output_node`` + ``pos_encoding``) — graph
  reconstruction is seconds where the compile it replaces is minutes.
  The graph must be rebuilt by the same deterministic construction code
  that produced the compiled artifact (the same property the CP-SAT
  schedule cache already relies on); the fingerprint check turns a
  mismatch into a loud error.  Attached checks are exempt: they are
  node metadata outside the fingerprint, so the rebuild may carry
  more, fewer, or different checks than the compiled graph —
  predicates always come from the rebuilt graph (they are Python
  callables and are never serialized).

Constraints (stated upfront):

* This is a **separate session from production** — the promoted debug
  outputs defeat onnxruntime's memory-reuse planning.  Never put it on
  a hot path.
* Fetching every residual snapshot costs
  ``n_pos × d × 2·n_layers`` floats on host per run — fine for
  debug-sized runs; probe very long prefills in slices.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

from torchwright.compiler.export import (
    DEBUG_META_FORMAT,
    _DebugState,
    debug_meta_path_for,
    meta_path_for,
)
from torchwright.compiler.graph_identity import (
    debug_fingerprint,
    decode_cols,
    nodes_by_canonical_id,
)
from torchwright.compiler.residual_assignment import (
    ResidualAssignment,
    ResidualStreamState,
)
from torchwright.graph import Node
from torchwright.graph.embedding import Embedding

#: Appended to the self-consistency failure preamble on this backend —
#: unlike the in-process backend, a violation here has three candidate
#: causes, and naming only the first would send the investigation to
#: the wrong place.
_ONNX_CONSISTENCY_CAUSES = (
    "\n  Candidate causes on the ONNX debug backend:"
    "\n    (1) scheduler/allocator bug — D1: stop and report;"
    "\n    (2) ONNX emission bug in torchwright/compiler/export.py — also D1;"
    "\n    (3) debug-sidecar / canonical-id remap bug"
    " (torchwright/compiler/graph_identity.py)."
    "\n  Recompiling via compile_headless and re-running debug=True"
    " discriminates (1) from (2)+(3) — with TW_SCHEDULE_CACHE_DIR unset"
    " for that run: a cache-replayed schedule bug reproduces on both"
    " backends and would masquerade as cause 2/3."
)


def _sidecar_or_raise(onnx_path: str) -> dict:
    path = debug_meta_path_for(onnx_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing debug sidecar {path}. Re-export with "
            f"compile_to_onnx (debug_sidecar=True, the default) to produce it."
        )
    with open(path) as f:
        sidecar = json.load(f)
    fmt = sidecar.get("format")
    if fmt != DEBUG_META_FORMAT:
        raise ValueError(
            f"{path}: unexpected format {fmt!r}, expected {DEBUG_META_FORMAT!r}"
        )
    return sidecar


#: onnxruntime's embedded-initializer ceiling (bytes).  ORT >= 1.26 refuses
#: to load ANY initializer whose dense size exceeds this unless the data is
#: external — and a *sparse* initializer (ONNX has no external form for
#: them) can therefore never load once its declared dense size crosses the
#: line.  The token vocab's ``embed_table`` does exactly that at production
#: widths (e.g. 101,617 rows x d=8192 x 4 B = 3.3 GB declared, stored
#: sparse).  Module-level so the regression test can shrink it and exercise
#: the conversion on a small artifact.
_ORT_EMBEDDED_INITIALIZER_LIMIT = 2**31

#: Flat elements per densification block in _make_session (256 MiB of fp32).
#: Bounds peak conversion memory to one block + the COO index/value arrays,
#: independent of the tensor's declared size.  Module-level so the chunk-seam
#: regression test can shrink it and force multi-block writes on a small
#: tensor.
_DENSIFY_BLOCK_ELEMS = 64 * 1024 * 1024


def _make_session(model, providers, owner):
    """Build the onnxruntime session for the (output-promoted) debug model.

    Small models load from in-memory bytes, as before.  When any sparse
    initializer declares a dense size over the ORT embedded ceiling, it is
    densified here and the model is saved to a session-lifetime temp
    directory with large tensors as external data, then loaded by path —
    the only load form ORT accepts for >2 GiB tensors.  Debug-session-only
    cost (one dense materialization + disk copy); the production runtime
    never takes this path.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper as onnx_helper
    from onnx import numpy_helper

    providers = providers or ["CPUExecutionProvider"]

    def _declared_bytes(sp) -> int:
        n = 1
        for d in sp.dims:
            n *= int(d)
        np_dtype = onnx_helper.tensor_dtype_to_np_dtype(sp.values.data_type)
        return n * np.dtype(np_dtype).itemsize

    oversized = [
        sp
        for sp in model.graph.sparse_initializer
        if _declared_bytes(sp) > _ORT_EMBEDDED_INITIALIZER_LIMIT
    ]
    if not oversized:
        return ort.InferenceSession(model.SerializeToString(), providers=providers)

    import tempfile

    from onnx import TensorProto

    # The temp dir must outlive the session (ORT reads the external file at
    # init; keeping it for the session's lifetime is the safe contract).
    owner._external_data_dir = tempfile.TemporaryDirectory(prefix="tw_onnx_debug_")
    data_name = "debug_dense.bin"
    data_path = os.path.join(owner._external_data_dir.name, data_name)

    # The dense bytes go straight into a side file, never into a
    # TensorProto: a protobuf message caps at 2 GiB, so a >2 GiB dense
    # tensor cannot even be constructed in-graph — only an EXTERNAL-stub
    # proto (dims + dtype + file/offset/length) can represent it.
    #
    # Densification is CHUNKED (flat blocks of _DENSIFY_BLOCK_ELEMS,
    # scatter the block's COO entries, ndarray.tofile) so peak memory is
    # one block plus the COO arrays — never the full dense tensor, and
    # never a second `tobytes()` copy of it (2x 3.3 GB at production
    # width).
    offset = 0
    with open(data_path, "wb") as f:
        for sp in oversized:
            dims = tuple(int(d) for d in sp.dims)
            numel = int(np.prod(dims))
            np_dtype = np.dtype(
                onnx_helper.tensor_dtype_to_np_dtype(sp.values.data_type)
            )
            idx = numpy_helper.to_array(sp.indices)
            val = numpy_helper.to_array(sp.values)
            # The exporter emits np.flatnonzero order (ascending), but
            # sort anyway — searchsorted below silently mis-slices on
            # unsorted input, and one O(nnz log nnz) pass is cheap.
            order = np.argsort(idx, kind="stable")
            idx = idx[order]
            val = val[order]

            length = numel * np_dtype.itemsize
            for lo in range(0, numel, _DENSIFY_BLOCK_ELEMS):
                hi = min(lo + _DENSIFY_BLOCK_ELEMS, numel)
                block = np.zeros(hi - lo, dtype=np_dtype)
                a, b = np.searchsorted(idx, [lo, hi])
                block[idx[a:b] - lo] = val[a:b]
                block.tofile(f)

            stub = TensorProto()
            stub.name = sp.values.name
            stub.data_type = sp.values.data_type
            stub.dims.extend(dims)
            # Entries set by hand: onnx.external_data_helper.set_external_data
            # only converts tensors that already carry raw_data, and the whole
            # point of the stub is to never materialize the bytes in-proto.
            for key, value in (
                ("location", data_name),
                ("offset", str(offset)),
                ("length", str(length)),
            ):
                entry = stub.external_data.add()
                entry.key = key
                entry.value = value
            stub.data_location = TensorProto.EXTERNAL
            model.graph.initializer.append(stub)
            model.graph.sparse_initializer.remove(sp)
            offset += length

    # Everything still embedded fits comfortably (the artifact held it all
    # in one file), so a plain save alongside the side file suffices; ORT
    # resolves external_data locations relative to the model's directory.
    tmp_model = os.path.join(owner._external_data_dir.name, "debug_model.onnx")
    onnx.save(model, tmp_model)
    return ort.InferenceSession(tmp_model, providers=providers)


class OnnxDebugSession:
    """Debuggable runtime over a compiled ONNX artifact.

    Implements the same DebugRuntime surface as ``CompiledHeadless``
    (``run_with_states`` / ``debug_layout`` / ``build_prefill`` /
    ``capture_attention`` / ``step(debug=True)`` / ``debug_value``), so
    ``probe_compiled``, ``probe_residual``, ``probe_attention``,
    ``probe_layer_diff``, and ``check_asserts_on_compiled`` accept it
    directly.

    Args:
        onnx_path: the ``.onnx`` artifact; ``<stem>.debug.json`` (and
            ``<stem>.meta.json``) must sit alongside it.
        output_node: the rebuilt graph's output node.
        providers: onnxruntime execution providers (default CPU).
    """

    def __init__(
        self,
        onnx_path: str,
        output_node: Node,
        providers=None,
    ) -> None:
        import onnx

        sidecar = _sidecar_or_raise(onnx_path)

        self._onnx_path = onnx_path
        self._kind: str = sidecar["kind"]  # "token" | "headless"
        self._d: int = int(sidecar["d"])
        self._n_layers: int = int(sidecar["n_layers"])
        self._input_specs: list[tuple] = [
            (str(n), int(s), int(w)) for n, s, w in sidecar["input_specs"]
        ]
        self.input_names: list[str] = [n for n, _, _ in self._input_specs]

        # --- Fingerprint: the rebuilt graph must be the compiled graph.
        out = output_node
        fp = debug_fingerprint(out, d=self._d, d_head=int(sidecar["d_head"]))
        if fp != sidecar["fingerprint"]:
            raise ValueError(
                f"{debug_meta_path_for(onnx_path)}: graph fingerprint mismatch — "
                f"the rebuilt graph's topology differs from the graph this "
                f"artifact was compiled from (sidecar "
                f"{sidecar['fingerprint'][:12]}..., rebuilt {fp[:12]}...).  "
                f"Rebuild with the same construction code/parameters as the "
                f"compile.  (Attached checks do NOT affect the "
                f"fingerprint; anything else does.)"
            )

        # --- Check predicates come from the rebuilt graph.
        from torchwright.graph.asserts import collect_debug_nodes

        self._checked_nodes = collect_debug_nodes(output_node)
        n_asserts = sum(
            1 for n in self._checked_nodes for c in n.checks if c.kind == "assert"
        )
        cov = sidecar.get("assert_coverage") or {}
        if n_asserts < int(cov.get("n_asserts", 0)):
            print(
                f"OnnxDebugSession WARNING: rebuilt graph carries "
                f"{n_asserts} assert check(s) but the compiled graph "
                f"had {cov['n_asserts']} — debug=True is checking fewer "
                f"invariants than the original compile would have."
            )

        # --- Remap the sidecar's canonical-id residual assignment onto
        # the rebuilt graph's live nodes.
        by_canon = nodes_by_canonical_id(out)

        # --- Machine kind: the topology fingerprint predates the swish
        # machine and cannot see FFN activation (its encoding is frozen), so
        # a relu-vs-swish rebuild of the same shapes would slip through it.
        # Cross-check explicitly against the sidecar's recorded machine
        # (pre-swish sidecars lack the key: they are all ReLU artifacts).
        from torchwright.graph.ffn import FFN as _FFN

        rebuilt_acts = {n.activation for n in by_canon.values() if isinstance(n, _FFN)}
        rebuilt_act = (
            next(iter(rebuilt_acts))
            if len(rebuilt_acts) == 1
            else ("relu" if not rebuilt_acts else "mixed")
        )
        sidecar_act = sidecar.get("activation", "relu")
        if rebuilt_act != sidecar_act:
            raise ValueError(
                f"{debug_meta_path_for(onnx_path)}: machine mismatch — the "
                f"artifact was compiled as the {sidecar_act!r} machine but the "
                f"rebuilt graph's FFNs give {rebuilt_act!r}.  Rebuild with the "
                f"same op package the compile used."
            )
        states: list[ResidualStreamState] = []
        key_to_state: dict[str, ResidualStreamState] = {}
        for entry in sidecar["states"]:
            st = ResidualStreamState(name=entry["key"])
            key_to_state[entry["key"]] = st
            states.append(st)
        ra = ResidualAssignment(set(states))
        for entry in sidecar["states"]:
            st = key_to_state[entry["key"]]
            same_as = entry.get("same_as")
            if same_as is not None:
                ra.duplicate_state(key_to_state[same_as], st)
                continue
            for cid_str, runs in entry["nodes"].items():
                node = by_canon.get(int(cid_str))
                if node is None:
                    continue
                ra.assign(st, node, decode_cols(runs))
        self._ra = ra
        self._key_to_state = key_to_state

        # --- Annotation paths from the sidecar's per-node table, keyed by
        # canonical id and mapped onto the rebuilt graph's live node
        # objects so callers can look one up by node (mirroring
        # node.annotation on the in-process backend).  Nodes that carried
        # no annotation are simply absent.
        self._annotation_by_node_id: dict[int, str] = {}
        for cid_str, meta in (sidecar.get("nodes") or {}).items():
            label = meta.get("annotation")
            if label is None:
                continue
            node = by_canon.get(int(cid_str))
            if node is not None:
                self._annotation_by_node_id[node.node_id] = label
        # Post-MLP triples — the ordered states every check/probe scans,
        # mirroring the in-process backend.
        self._ordered: list[tuple] = [
            (i, f"L{i}.mlp", key_to_state[f"L{i}.mlp"]) for i in range(self._n_layers)
        ]
        # state -> ONNX tensor name, every capturable state.
        self._state_fetch: list[tuple[ResidualStreamState, str, str]] = []
        if "input" in key_to_state:
            self._state_fetch.append((key_to_state["input"], "res_0", "input_res_0"))
        for i in range(self._n_layers):
            self._state_fetch.append(
                (
                    key_to_state[f"L{i}.attn"],
                    f"l{i}_res_attn",
                    f"layer_{i}_attn_skip_out_state",
                )
            )
            self._state_fetch.append(
                (
                    key_to_state[f"L{i}.mlp"],
                    f"l{i}_res_next",
                    f"layer_{i}_mlp_out_state",
                )
            )

        # --- Token graphs: the Embedding node translates token strings.
        self._embedding: Embedding | None = None
        if self._kind == "token":
            stack = [out]
            seen: set = set()
            while stack:
                n = stack.pop()
                if n.node_id in seen:
                    continue
                seen.add(n.node_id)
                if isinstance(n, Embedding):
                    self._embedding = n
                    break
                stack.extend(getattr(n, "inputs", None) or [])

        # --- Production meta sidecar (vocab etc.) — optional but normal.
        self.metadata: dict = {}
        meta_path = meta_path_for(onnx_path)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

        # --- Build the debug ORT session: promote the per-layer residual
        # and attention tensors to graph outputs.  Outputs are computed
        # only when fetched, so production-shaped runs through this
        # session fetch the normal outputs and pay nothing extra beyond
        # the disabled memory-reuse planning.
        model = onnx.load(onnx_path)
        graph_inputs = {vi.name: vi for vi in model.graph.input}
        if "past_K_0" not in graph_inputs:
            raise ValueError(
                f"{onnx_path}: no past_K_0 input — not a cached-protocol model"
            )

        # Emission-mode cross-check: the sidecar's recorded bias flag (a
        # compile option — the rebuilt graph cannot reveal it, so unlike
        # the machine check this compares sidecar against the artifact's
        # own initializer set).  Pre-feature sidecars lack the key: they
        # are all biased artifacts.
        sidecar_bias = bool(sidecar.get("bias", True))
        init_names = {t.name for t in model.graph.initializer} | {
            s.values.name for s in model.graph.sparse_initializer
        }
        artifact_bias = any(
            name in init_names
            for name in ("l0_b1", "l0_b2", "l0_bgate", "l0_bup", "l0_bdown")
        )
        if sidecar_bias != artifact_bias:
            raise ValueError(
                f"{debug_meta_path_for(onnx_path)}: emission-mode mismatch — "
                f"the sidecar records bias={sidecar_bias} but the artifact "
                f"{'carries' if artifact_bias else 'carries no'} bias "
                f"initializers.  The sidecar does not belong to this model; "
                f"re-export the pair together."
            )

        def _dim(vi, k):
            d = vi.type.tensor_type.shape.dim[k]
            return d.dim_value if d.HasField("dim_value") else None

        self._per_layer_n_heads = [
            int(_dim(graph_inputs[f"past_K_{i}"], 1)) for i in range(self._n_layers)
        ]
        self._d_head = int(_dim(graph_inputs["past_K_0"], 2))
        # Symbolic slot dim => prefix bindings allowed (stride
        # bucketing); a static dim (old export) forces full-width feeds.
        self._static_slot_dim: int | None = _dim(graph_inputs["past_K_0"], 0)
        self._cache_stride: int = int(self._static_slot_dim or sidecar["cache_stride"])

        from onnx import TensorProto, helper

        existing_outputs = {vo.name for vo in model.graph.output}
        for _state, tensor_name, _label in self._state_fetch:
            if tensor_name not in existing_outputs:
                model.graph.output.append(
                    helper.make_tensor_value_info(
                        tensor_name, TensorProto.FLOAT, ["n_new", self._d]
                    )
                )
        for i in range(self._n_layers):
            for suffix in ("weights", "logits_masked"):
                name = f"l{i}_{suffix}"
                if name not in existing_outputs:
                    model.graph.output.append(
                        helper.make_tensor_value_info(
                            name,
                            TensorProto.FLOAT,
                            [self._per_layer_n_heads[i], "n_new", "n_keys"],
                        )
                    )

        # Session-lifetime home of externalized initializer data; set by
        # _make_session only when an oversized sparse tensor was converted.
        self._external_data_dir = None
        self._session = _make_session(model, providers, self)
        self._primary_output = "logits" if self._kind == "token" else "outputs"
        self._debug_state: _DebugState | None = None

    # ---- feed/run plumbing ----------------------------------------------

    def empty_past(self) -> tuple:
        """Zero-length sequence-major KV tuples, mirroring
        ``CompiledHeadless.empty_past``'s grow-per-step representation
        (each entry ``(n_committed, n_heads_i, d_head)``).
        """
        k = tuple(torch.zeros(0, nh, self._d_head) for nh in self._per_layer_n_heads)
        v = tuple(torch.zeros(0, nh, self._d_head) for nh in self._per_layer_n_heads)
        return (k, v)

    def _feeds(self, prefill: torch.Tensor, base: int, past: tuple | None) -> dict:
        """Build ORT feeds for ``n_new`` rows at absolute positions
        ``base..base+n_new``.

        The KV binding is the smallest covering prefix ``base + n_new``
        (valid under stride bucketing) so debug runs never materialize
        the full stride buffer; old static-dim exports get full-width
        zero-padded feeds instead.
        """
        n_new = int(prefill.shape[0])
        if base + n_new > self._cache_stride:
            raise RuntimeError(
                f"static cache overrun: base {base} + n_new {n_new} exceeds "
                f"cache_stride {self._cache_stride}"
            )
        feeds: dict = {"cache_position": np.arange(base, base + n_new, dtype=np.int64)}
        if self._kind == "token":
            ids = prefill.reshape(-1).detach().cpu()
            feeds["token_ids"] = ids.numpy().astype(np.int64, copy=False)
        else:
            feeds["inputs"] = (
                prefill.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        if self._static_slot_dim is not None:
            bind_width = self._static_slot_dim
        else:
            bind_width = base + n_new
        for i, nh in enumerate(self._per_layer_n_heads):
            buf = np.zeros((bind_width, nh, self._d_head), dtype=np.float32)
            if past is not None and base > 0:
                buf[:base] = (
                    past[0][i].detach().cpu().numpy().astype(np.float32, copy=False)
                )
            feeds[f"past_K_{i}"] = buf
            buf_v = np.zeros((bind_width, nh, self._d_head), dtype=np.float32)
            if past is not None and base > 0:
                buf_v[:base] = (
                    past[1][i].detach().cpu().numpy().astype(np.float32, copy=False)
                )
            feeds[f"past_V_{i}"] = buf_v
        return feeds

    def _resolve_base(self, past: tuple | None, past_len: int | None) -> int:
        base = 0 if past is None else int(past[0][0].shape[0])
        if past_len is not None and int(past_len) != base:
            raise ValueError(
                f"past_len {past_len} != committed length {base}: the static "
                f"cache derives mask AND pos from cache_position; a trimmed "
                f"cache with a larger absolute position is not expressible"
            )
        return base

    # ---- inference surface ------------------------------------------------

    def step(
        self,
        inputs: torch.Tensor,
        past: tuple,
        past_len: int | None = None,
        debug: bool = False,
        debug_atol: float = 1e-7,
    ) -> tuple:
        """One cached-protocol call; returns ``(outputs, new_past)``.

        ``inputs`` is ``(n_new, 1)`` token ids (token models — a flat
        ``(n_new,)`` int tensor is also accepted) or ``(n_new, d_input)``
        floats (headless models).  With ``debug=True``, additionally
        fetches every per-layer residual snapshot from the same run and
        performs the same three checks as
        ``CompiledHeadless.step(debug=True)``: residual self-consistency,
        assert-kind checks (raise), watch-kind checks (print).
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1) if self._kind == "token" else inputs
        base = self._resolve_base(past, past_len)
        n_new = int(inputs.shape[0])
        feeds = self._feeds(inputs, base, past)

        fetch = [self._primary_output]
        for i in range(self._n_layers):
            fetch += [f"delta_K_{i}", f"delta_V_{i}"]
        n_production = len(fetch)
        if debug:
            fetch += [tensor_name for _s, tensor_name, _l in self._state_fetch]

        results = self._session.run(fetch, feeds)
        outputs = torch.from_numpy(results[0])
        new_k = tuple(
            torch.cat([past[0][i], torch.from_numpy(results[1 + 2 * i])])
            for i in range(self._n_layers)
        )
        new_v = tuple(
            torch.cat([past[1][i], torch.from_numpy(results[1 + 2 * i + 1])])
            for i in range(self._n_layers)
        )

        if debug:
            state_tensor = {
                state: (torch.from_numpy(results[n_production + j]), label)
                for j, (state, _name, label) in enumerate(self._state_fetch)
            }
            self._run_debug_checks(state_tensor, atol=debug_atol)

        return outputs, (new_k, new_v)

    def __call__(
        self,
        inputs: torch.Tensor,
        debug: bool = False,
        debug_atol: float = 1e-7,
    ) -> torch.Tensor:
        """Stateless prefill that discards the cache."""
        outputs, _ = self.step(
            inputs, self.empty_past(), debug=debug, debug_atol=debug_atol
        )
        return outputs

    def eval(self) -> OnnxDebugSession:
        return self

    # ---- debug surface ------------------------------------------------------

    def _run_debug_checks(self, state_tensor: dict, atol: float) -> None:
        from torchwright.debug.extraction import (
            check_debug_predicates,
            run_consistency_check,
        )

        ordered_states = [s for _, _, s in self._ordered]
        self._debug_state = _DebugState(
            state_tensor=state_tensor,
            ordered_states=ordered_states,
            ra=self._ra,
        )
        run_consistency_check(
            ordered_states,
            state_tensor,
            self._ra,
            atol,
            extra_cause=_ONNX_CONSISTENCY_CAUSES,
        )
        check_debug_predicates(
            self._checked_nodes, self._ra, ordered_states, state_tensor
        )

    def debug_value(self, node: Node) -> torch.Tensor | None:
        """Compiled value of ``node`` from the last debug=True run, or None."""
        if self._debug_state is None:
            raise RuntimeError("debug_value() requires a prior debug=True run")
        from torchwright.debug.extraction import (
            extract_compiled_value,
            first_state_with,
        )

        ds = self._debug_state
        state = first_state_with(node, ds.ra, ds.ordered_states)
        if state is None:
            return None
        tensor_pair = ds.state_tensor.get(state)
        if tensor_pair is None:
            return None
        res_tensor, _ = tensor_pair
        return extract_compiled_value(node, ds.ra, state, res_tensor)

    def annotation(self, node: Node) -> str | None:
        """The ``annotate()`` path recorded for ``node`` at compile time.

        Looks the rebuilt ``node`` up against the annotations the sidecar
        captured (keyed by canonical id).  Returns ``None`` for nodes that
        carried no annotation when compiled, or that aren't reachable from
        the output.
        """
        return self._annotation_by_node_id.get(node.node_id)

    # ---- DebugRuntime protocol surface (probe entry points) ----------------

    def debug_layout(self) -> tuple:
        """``(residual_assignment, ordered)`` without running the model."""
        return self._ra, list(self._ordered)

    def run_with_states(
        self,
        prefill: torch.Tensor,
        past_len: int = 0,
        past_kvs: tuple | None = None,
    ) -> tuple:
        """Run once fetching every residual snapshot.

        Returns ``(ra, ordered, state_tensor)`` with the same shapes the
        in-process backend produces, so every probe consumes it
        unchanged.
        """
        if prefill.ndim == 1:
            prefill = prefill.reshape(-1, 1)
        base = self._resolve_base(past_kvs, past_len or None)
        feeds = self._feeds(prefill, base, past_kvs)
        fetch = [tensor_name for _s, tensor_name, _l in self._state_fetch]
        results = self._session.run(fetch, feeds)
        state_tensor = {
            state: (torch.from_numpy(results[j]), label)
            for j, (state, _name, label) in enumerate(self._state_fetch)
        }
        return self._ra, list(self._ordered), state_tensor

    def build_prefill(
        self,
        input_values: dict[str, Any],
        n_pos: int,
    ) -> torch.Tensor:
        """Pack an input-name → value dict into the flat prefill layout.

        Token models additionally accept a list of token *strings*,
        translated through the rebuilt graph's Embedding tokenizer.
        """
        if self._kind == "token":
            name = self._input_specs[0][0]
            if name not in input_values:
                raise ValueError(f"missing input '{name}'")
            raw = input_values[name]
            if isinstance(raw, torch.Tensor):
                ids = raw.reshape(-1)
            else:
                assert self._embedding is not None, (
                    "token-string inputs need an Embedding node in the rebuilt graph"
                )
                ids = torch.tensor(
                    [self._embedding.tokenizer.get_token_id(t) for t in raw]
                )
            assert ids.shape[0] == n_pos
            return ids.reshape(-1, 1).float()
        d_input = max(start + width for _, start, width in self._input_specs)
        out = torch.zeros(n_pos, d_input)
        for name, start, width in self._input_specs:
            if name not in input_values:
                raise ValueError(f"missing input '{name}'")
            out[:, start : start + width] = input_values[name]
        return out

    def capture_attention(
        self,
        layer_index: int,
        prefill: torch.Tensor,
        past_len: int = 0,
        past_kvs: tuple | None = None,
    ) -> tuple:
        """Softmax ``(weights, logits)`` at one attention layer.

        Fetched directly from the artifact's ``l{i}_weights`` /
        ``l{i}_logits_masked`` tensors; each is
        ``(n_heads, n_queries, n_keys)`` with ``n_keys = past + new`` —
        the same shape the in-process ``attention_capture`` hook
        produces.
        """
        if prefill.ndim == 1:
            prefill = prefill.reshape(-1, 1)
        base = self._resolve_base(past_kvs, past_len or None)
        feeds = self._feeds(prefill, base, past_kvs)
        weights_np, logits_np = self._session.run(
            [f"l{layer_index}_weights", f"l{layer_index}_logits_masked"], feeds
        )
        return torch.from_numpy(weights_np), torch.from_numpy(logits_np)
