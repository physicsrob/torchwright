"""Debug session over a compiled ONNX artifact.

:class:`OnnxDebugSession` gives the ONNX export the same debug surface
as the in-process :class:`~torchwright.compiler.export.CompiledHeadless`
— ``step(..., debug=True)`` (residual self-consistency + Assert/
DebugWatch predicates), ``debug_value(node)``, and the probe suite in
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
  mismatch into a loud error.  Assert/DebugWatch wrappers are exempt:
  the fingerprint is wrapper-transparent, so the rebuild may carry
  more, fewer, or different debug wrappers than the compiled graph —
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
from typing import Any, Dict, List, Optional, Tuple

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
    unwrap_debug,
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
        output_node: the rebuilt graph's output node.  May be
            Assert-wrapped; wrappers are handled transparently.
        providers: onnxruntime execution providers (default CPU).
    """

    def __init__(
        self,
        onnx_path: str,
        output_node: Node,
        providers=None,
    ) -> None:
        import onnx
        import onnxruntime as ort

        sidecar = _sidecar_or_raise(onnx_path)

        self._onnx_path = onnx_path
        self._kind: str = sidecar["kind"]  # "token" | "headless"
        self._d: int = int(sidecar["d"])
        self._n_layers: int = int(sidecar["n_layers"])
        self._input_specs: List[tuple] = [
            (str(n), int(s), int(w)) for n, s, w in sidecar["input_specs"]
        ]
        self.input_names: List[str] = [n for n, _, _ in self._input_specs]

        # --- Fingerprint: the rebuilt graph must be the compiled graph.
        out = unwrap_debug(output_node)
        fp = debug_fingerprint(out, d=self._d, d_head=int(sidecar["d_head"]))
        if fp != sidecar["fingerprint"]:
            raise ValueError(
                f"{debug_meta_path_for(onnx_path)}: graph fingerprint mismatch — "
                f"the rebuilt graph's topology differs from the graph this "
                f"artifact was compiled from (sidecar "
                f"{sidecar['fingerprint'][:12]}..., rebuilt {fp[:12]}...).  "
                f"Rebuild with the same construction code/parameters as the "
                f"compile.  (Assert/DebugWatch wrappers do NOT affect the "
                f"fingerprint; anything else does.)"
            )

        # --- Assert/DebugWatch predicates come from the rebuilt graph.
        from torchwright.graph.asserts import collect_debug_nodes

        self._asserts, self._watches = collect_debug_nodes(output_node)
        cov = sidecar.get("assert_coverage") or {}
        if len(self._asserts) < int(cov.get("n_asserts", 0)):
            print(
                f"OnnxDebugSession WARNING: rebuilt graph carries "
                f"{len(self._asserts)} Assert node(s) but the compiled graph "
                f"had {cov['n_asserts']} — debug=True is checking fewer "
                f"invariants than the original compile would have."
            )

        # --- Remap the sidecar's canonical-id residual assignment onto
        # the rebuilt graph's live nodes.
        by_canon = nodes_by_canonical_id(out)
        states: List[ResidualStreamState] = []
        key_to_state: Dict[str, ResidualStreamState] = {}
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
        self._annotation_by_node_id: Dict[int, str] = {}
        for cid_str, meta in (sidecar.get("nodes") or {}).items():
            label = meta.get("annotation")
            if label is None:
                continue
            node = by_canon.get(int(cid_str))
            if node is not None:
                self._annotation_by_node_id[node.node_id] = label
        # Post-MLP triples — the ordered states every check/probe scans,
        # mirroring the in-process backend.
        self._ordered: List[tuple] = [
            (i, f"L{i}.mlp", key_to_state[f"L{i}.mlp"]) for i in range(self._n_layers)
        ]
        # state -> ONNX tensor name, every capturable state.
        self._state_fetch: List[Tuple[ResidualStreamState, str, str]] = []
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
        self._embedding: Optional[Embedding] = None
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

        def _dim(vi, k):
            d = vi.type.tensor_type.shape.dim[k]
            return d.dim_value if d.HasField("dim_value") else None

        self._per_layer_n_heads = [
            int(_dim(graph_inputs[f"past_K_{i}"], 1)) for i in range(self._n_layers)
        ]
        self._d_head = int(_dim(graph_inputs["past_K_0"], 2))
        # Symbolic slot dim => prefix bindings allowed (stride
        # bucketing); a static dim (old export) forces full-width feeds.
        self._static_slot_dim: Optional[int] = _dim(graph_inputs["past_K_0"], 0)
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

        self._session = ort.InferenceSession(
            model.SerializeToString(),
            providers=providers or ["CPUExecutionProvider"],
        )
        self._primary_output = "logits" if self._kind == "token" else "outputs"
        self._debug_state: Optional[_DebugState] = None

    # ---- feed/run plumbing ----------------------------------------------

    def empty_past(self) -> tuple:
        """Zero-length sequence-major KV tuples, mirroring
        ``CompiledHeadless.empty_past``'s grow-per-step representation
        (each entry ``(n_committed, n_heads_i, d_head)``)."""
        k = tuple(torch.zeros(0, nh, self._d_head) for nh in self._per_layer_n_heads)
        v = tuple(torch.zeros(0, nh, self._d_head) for nh in self._per_layer_n_heads)
        return (k, v)

    def _feeds(self, prefill: torch.Tensor, base: int, past: Optional[tuple]) -> dict:
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

    def _resolve_base(self, past: Optional[tuple], past_len: Optional[int]) -> int:
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
        past_len: Optional[int] = None,
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
        Assert predicates (raise), DebugWatch predicates (print).
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

    def eval(self) -> "OnnxDebugSession":
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
            self._asserts, self._watches, self._ra, ordered_states, state_tensor
        )

    def debug_value(self, node: Node) -> Optional[torch.Tensor]:
        """Compiled value of ``node`` from the last debug=True run, or None."""
        if self._debug_state is None:
            raise RuntimeError("debug_value() requires a prior debug=True run")
        from torchwright.debug.extraction import (
            extract_compiled_value,
            first_state_with,
        )

        node = unwrap_debug(node)
        ds = self._debug_state
        state = first_state_with(node, ds.ra, ds.ordered_states)
        if state is None:
            return None
        tensor_pair = ds.state_tensor.get(state)
        if tensor_pair is None:
            return None
        res_tensor, _ = tensor_pair
        return extract_compiled_value(node, ds.ra, state, res_tensor)

    def annotation(self, node: Node) -> Optional[str]:
        """The ``annotate()`` path recorded for ``node`` at compile time.

        Looks the rebuilt ``node`` up against the annotations the sidecar
        captured (keyed by canonical id).  Returns ``None`` for nodes that
        carried no annotation when compiled, or that aren't reachable from
        the output.  Assert/DebugWatch wrappers are unwrapped first.
        """
        return self._annotation_by_node_id.get(unwrap_debug(node).node_id)

    # ---- DebugRuntime protocol surface (probe entry points) ----------------

    def debug_layout(self) -> tuple:
        """``(residual_assignment, ordered)`` without running the model."""
        return self._ra, list(self._ordered)

    def run_with_states(
        self,
        prefill: torch.Tensor,
        past_len: int = 0,
        past_kvs: Optional[tuple] = None,
    ) -> tuple:
        """Run once fetching every residual snapshot.

        Returns ``(ra, ordered, state_tensor)`` with the same shapes the
        in-process backend produces, so every probe consumes it
        unchanged.
        """
        if prefill.ndim == 1:
            prefill = prefill.reshape(-1, 1)
        base = self._resolve_base(past_kvs, past_len if past_len else None)
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
        input_values: Dict[str, Any],
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
                    "token-string inputs need an Embedding node in the " "rebuilt graph"
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
        past_kvs: Optional[tuple] = None,
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
        base = self._resolve_base(past_kvs, past_len if past_len else None)
        feeds = self._feeds(prefill, base, past_kvs)
        weights_np, logits_np = self._session.run(
            [f"l{layer_index}_weights", f"l{layer_index}_logits_masked"], feeds
        )
        return torch.from_numpy(weights_np), torch.from_numpy(logits_np)
