"""Graph-vs-compiled divergence probe.

Runs a compiled ``HeadlessTransformer`` side-by-side with a direct,
recursive evaluation of its source graph (the oracle: ``node.compute``)
and reports the first graph node in topological order whose compiled
value disagrees with the reference beyond a numeric tolerance.

The probe relies on the per-sublayer column snapshots that
:func:`torchwright.compiler.forward.compile.forward_compile` writes into
``HeadlessTransformer.residual_assignment`` — one snapshot per
post-MLP sublayer state.  For each graph node the probe picks the
earliest state where the node is materialised and extracts its
compiled value from that sublayer's residual-stream tensor.

Scope and limits:

* Single-position (non-autoregressive) only.  Cross-position attention
  is evaluated by the oracle (``Attn.compute`` runs the full softmax
  matmul), so multi-position graphs do produce a correct oracle value,
  but a stateful decode-protocol bug — KV cache trimming, ``past_len``
  drift, etc. — would still hide behind the compiled module's
  ``forward()`` path used here.
* The oracle uses class-level monkey-patching of ``Node.compute`` to
  memoise each node's value.  The patches are restored in a ``finally``
  block; the probe is not thread-safe.
* Nodes whose columns never survive to the final ``out_state`` can
  still be checked as long as they appeared in one of the per-sublayer
  snapshots — this is what lets us localise a bug to the exact layer
  that broke it rather than only the top output.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import torch

from torchwright.compiler.graph_clone import topological_order
from torchwright.compiler.residual_assignment import (
    ResidualAssignment,
    ResidualStreamState,
)
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.debug.extraction import (
    StateTensors,
    extract_compiled_value,
    first_state_with,
)
from torchwright.graph import Concatenate, Node
from torchwright.graph.attn import CAUSAL_MASK_SENTINEL, Attn
from torchwright.graph.misc import (
    InputNode,
    LiteralValue,
    Placeholder,
)

# Backward-compatible aliases — the implementations moved to
# torchwright.debug.extraction so the ONNX debug backend can share them.
_extract_compiled_value = extract_compiled_value
_first_state_with = first_state_with

# KV cache in either backend's own representation: the in-process backend
# uses a per-layer list of (K, V) tensor pairs; the ONNX debug backend uses
# a plain tuple (see torchwright.debug.onnx_debug).
PastKVs = list[tuple[torch.Tensor, torch.Tensor]] | tuple


class DebugRuntime(Protocol):
    """Structural type for a debuggable compiled backend.

    Satisfied by both :class:`~torchwright.compiler.export.CompiledHeadless`
    (in-process torch forward with state capture) and
    :class:`~torchwright.debug.onnx_debug.OnnxDebugSession` (the real
    ONNX artifact re-run with its residual tensors promoted to
    outputs).  Every probe in this module accepts either.

    ``ordered`` is always ``[(layer_index, state_name, state), ...]`` —
    the post-MLP sublayer states in execution order; ``state_tensor``
    maps each captured state to ``(residual_tensor, label)``.
    """

    def debug_layout(self) -> tuple[ResidualAssignment, list[tuple]]: ...

    def run_with_states(
        self,
        prefill: torch.Tensor,
        past_len: int = 0,
        # Any, not PastKVs: the two concrete backends this Protocol is
        # matched against structurally (CompiledHeadless, OnnxDebugSession)
        # declare genuinely different concrete parameter types for
        # past_kvs (list[tuple[Tensor, Tensor]] | None vs. tuple | None,
        # in export.py and onnx_debug.py respectively) — a real, load-
        # bearing type difference across backends, not a laziness gap.
        past_kvs: Any = None,
    ) -> tuple[ResidualAssignment, list[tuple], StateTensors]: ...

    def build_prefill(
        self, input_values: dict[str, Any], n_pos: int
    ) -> torch.Tensor: ...

    def capture_attention(
        self,
        layer_index: int,
        prefill: torch.Tensor,
        past_len: int = 0,
        past_kvs: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


# ---------------------------------------------------------------------------
# Oracle: memoised recursive evaluator
# ---------------------------------------------------------------------------


def reference_eval(
    output_node: Node,
    input_values: dict[str, torch.Tensor],
    n_pos: int,
    seed: dict[Node, torch.Tensor] | None = None,
) -> dict[Node, torch.Tensor]:
    """Recursively evaluate the graph and return ``{Node: tensor}``.

    Walks the graph via ``node.compute`` — the same method the compiler
    already trusts as the semantic definition of each node.  Each node
    is computed exactly once via class-level ``compute`` monkey-patches
    that consult a shared cache: the patches intercept every recursive
    ``self.inputs[i].compute`` call, so the recursion collapses to
    O(n) torch ops over the graph's n nodes instead of the O(n²) the
    un-memoised recursion would pay.

    Args:
        output_node: the root of the subgraph to evaluate.
        input_values: ``{input_name: (n_pos, d_input) tensor}`` — one
            entry per :class:`InputNode` reachable from ``output_node``.
        n_pos: number of positions to evaluate at.
        seed: optional ``{node: (n_pos, d) tensor}`` values to pin
            mid-graph — a seeded node's ``compute`` never runs and
            nothing upstream of it is evaluated (its inputs are only
            reached through it).  This is how the univariate collapse
            pass re-roots a subgraph's oracle at its scalar source.

    Returns:
        A dict mapping every reachable node (including ``output_node``
        itself) to its oracle value as an ``(n_pos, node.d_output)``
        tensor.  Seeded entries are included verbatim.
    """
    cache: dict[Node, torch.Tensor] = dict(seed) if seed else {}

    # Collect all node subclasses reachable from the output graph so we
    # only patch classes actually in use.  Walking by class lets us
    # restore every patch in a tight finally block even if compute()
    # raises mid-run.  Attached checks run as part of each node's
    # ``compute`` (once per node — the memoisation prevents recomputes),
    # so the oracle pass exercises every assert and watch.
    all_nodes = get_ancestor_nodes({output_node})
    classes_in_graph = {type(n) for n in all_nodes}

    def _make_cached(
        orig_compute: Callable[[Node, int, dict], torch.Tensor],
    ) -> Callable[[Node, int, dict], torch.Tensor]:
        def wrapped(self: Node, n_pos_arg: int, input_values_arg: dict) -> torch.Tensor:
            hit = cache.get(self)
            if hit is not None:
                return hit
            val = orig_compute(self, n_pos_arg, input_values_arg)
            cache[self] = val
            return val

        return wrapped

    patched: list[tuple[type, Any]] = []
    try:
        for cls in classes_in_graph:
            if "compute" in cls.__dict__:
                orig = cls.__dict__["compute"]
                patched.append((cls, orig))
                cls.compute = cast("Any", _make_cached(orig))  # type: ignore[method-assign]
        output_node.compute(n_pos, input_values)
    finally:
        for cls, orig in patched:
            cls.compute = orig  # type: ignore[method-assign]

    return cache


# ---------------------------------------------------------------------------
# Probe report
# ---------------------------------------------------------------------------


@dataclass
class NodeDivergence:
    """Per-node divergence record."""

    node: Node
    state: ResidualStreamState
    max_abs_error: float
    compiled_mean: float
    oracle_mean: float
    compiled_min: float
    compiled_max: float
    oracle_min: float
    oracle_max: float

    def summary(self) -> str:
        return (
            f"{type(self.node).__name__}(id={self.node.node_id}, "
            f"name='{self.node.name}', d={self.node.d_output}) "
            f"at {self.state.name or f'state_{self.state.state_id}'}: "
            f"max_abs_err={self.max_abs_error:.4g} "
            f"(compiled mean={self.compiled_mean:.4g} "
            f"range=[{self.compiled_min:.4g}, {self.compiled_max:.4g}]; "
            f"oracle mean={self.oracle_mean:.4g} "
            f"range=[{self.oracle_min:.4g}, {self.oracle_max:.4g}])"
        )


@dataclass
class ProbeReport:
    """Structured result of a probe run."""

    #: Graph nodes ordered by topological rank, excluding nodes the
    #: probe cannot check (Concatenate groupings, nodes with no column
    #: assignment in any snapshot).
    nodes_checked: list[Node] = field(default_factory=list)

    #: Per-node divergence records, keyed by the node's topological
    #: index.  Every entry in ``nodes_checked`` has a record; the record
    #: is considered "divergent" if ``max_abs_error`` exceeds the
    #: probe's ``atol``.
    per_node: dict[Node, NodeDivergence] = field(default_factory=dict)

    #: The first node in topological order whose compiled value
    #: exceeds ``atol``, or ``None`` if the probe found no divergence.
    first_divergent: NodeDivergence | None = None

    #: Graph nodes the probe deliberately skipped, with a reason.
    skipped: dict[Node, str] = field(default_factory=dict)

    #: Tolerance used for the "divergent" classification.
    atol: float = 1e-3

    def format_short(self, show_top_k: int = 10) -> str:
        lines = [
            f"ProbeReport: checked {len(self.nodes_checked)} nodes, "
            f"skipped {len(self.skipped)} (atol={self.atol:.2g})"
        ]
        if self.first_divergent is None:
            lines.append("  no divergence found")
            return "\n".join(lines)
        lines.append(f"  first divergent: {self.first_divergent.summary()}")
        ranked = sorted(
            self.per_node.values(),
            key=lambda r: -r.max_abs_error,
        )
        lines.append(f"  top-{show_top_k} by error magnitude:")
        lines.extend(f"    {r.summary()}" for r in ranked[:show_top_k])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared forward / state-capture helpers
# ---------------------------------------------------------------------------


def _ordered_mlp_states(
    net: HeadlessTransformer,
    ra: ResidualAssignment,
) -> list[tuple[int, str, ResidualStreamState]]:
    """Post-MLP sublayer states in execution order.

    Returns ``(layer_index, state_name, state)`` triples, one per
    transformer layer whose ``mlp.out_state`` is recorded in
    ``ra.mapping``.  The final layer's ``mlp.out_state`` is always
    appended (even if missing from ``ra.mapping``) so the top-level
    output is reachable when the last layer happens to receive no new
    assignments.
    """
    ordered: list[tuple[int, str, ResidualStreamState]] = []
    for i, layer in enumerate(net.layers):
        st = layer.mlp.out_state
        if st in ra.mapping:
            ordered.append((i, f"L{i}.mlp_out", st))
    last_i = len(net.layers) - 1
    last_st = net.layers[-1].mlp.out_state
    if not any(s is last_st for _, _, s in ordered):
        ordered.append((last_i, f"L{last_i}.mlp_out", last_st))
    return ordered


def _run_with_states(
    compiled: DebugRuntime,
    prefill: torch.Tensor,
    past_len: int = 0,
    past_kvs: PastKVs | None = None,
) -> tuple[
    ResidualAssignment,
    list[tuple],
    dict[ResidualStreamState, tuple[torch.Tensor, str]],
]:
    """Run the compiled backend once with per-sublayer state capture.

    Thin dispatcher over the backend's ``run_with_states`` — the
    in-process backend forwards the torch module with
    ``return_states=True`` (or a manual cached layer walk when
    ``past_kvs`` is supplied); the ONNX backend re-runs the artifact
    fetching its promoted residual-tensor outputs.

    Returns ``(ra, ordered, state_tensor)`` where ``ordered`` is the
    post-MLP ``(layer_index, state_name, state)`` triple list.
    """
    return compiled.run_with_states(prefill, past_len=past_len, past_kvs=past_kvs)


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------


def probe_compiled(
    compiled: DebugRuntime,
    output_node: Node,
    input_values: dict[str, torch.Tensor],
    n_pos: int,
    atol: float = 1e-3,
) -> ProbeReport:
    """Run a divergence probe against an already-compiled module.

    Args:
        compiled: any :class:`DebugRuntime` — a :class:`CompiledHeadless`
            or an :class:`~torchwright.debug.onnx_debug.OnnxDebugSession`
            over the production artifact.
        output_node: the same graph node that was passed to the
            compiler (or its deterministic rebuild).  Used both as the
            oracle root and as the topological-order anchor.
        input_values: ``{input_name: (n_pos, d_input) tensor}``.
        n_pos: number of positions to run the probe on.
        atol: max absolute error below which a per-node comparison is
            treated as clean.  Raise for noisier op realisations.

    Returns:
        A populated :class:`ProbeReport`.
    """
    # Oracle first — cheap and deterministic.
    oracle = reference_eval(output_node, input_values, n_pos)

    ra, ordered, state_tensor = _run_with_states(
        compiled,
        compiled.build_prefill(input_values, n_pos),
        past_len=0,
    )
    ordered_states = [s for _, _, s in ordered]

    report = ProbeReport(atol=atol)

    for node in topological_order(output_node):
        if isinstance(node, (InputNode, LiteralValue, Placeholder)):
            report.skipped[node] = "input/literal/placeholder"
            continue
        if isinstance(node, Concatenate):
            report.skipped[node] = "concat grouping — leaves are checked individually"
            continue

        state = _first_state_with(node, ra, ordered_states)
        if state is None:
            report.skipped[node] = "no residual assignment found in any captured state"
            continue

        tensor_pair = state_tensor.get(state)
        if tensor_pair is None:
            report.skipped[node] = (
                f"state {state.name or state.state_id} was assigned in "
                f"residual_assignment but not produced by forward(return_states=True)"
            )
            continue
        res_tensor, _ = tensor_pair

        compiled_val = _extract_compiled_value(node, ra, state, res_tensor)
        if compiled_val is None:
            report.skipped[node] = "no columns allocated at chosen state"
            continue

        oracle_val = oracle.get(node)
        if oracle_val is None:
            report.skipped[node] = "oracle did not reach this node"
            continue

        if compiled_val.shape != oracle_val.shape:
            report.skipped[node] = (
                f"shape mismatch compiled={tuple(compiled_val.shape)} "
                f"oracle={tuple(oracle_val.shape)}"
            )
            continue

        diff = (compiled_val.detach().cpu() - oracle_val.detach().cpu()).abs()
        max_err = float(diff.max().item())
        rec = NodeDivergence(
            node=node,
            state=state,
            max_abs_error=max_err,
            compiled_mean=float(compiled_val.mean().item()),
            oracle_mean=float(oracle_val.mean().item()),
            compiled_min=float(compiled_val.min().item()),
            compiled_max=float(compiled_val.max().item()),
            oracle_min=float(oracle_val.min().item()),
            oracle_max=float(oracle_val.max().item()),
        )
        report.nodes_checked.append(node)
        report.per_node[node] = rec
        if report.first_divergent is None and max_err > atol:
            report.first_divergent = rec

    return report


def _inputs_from_dict(
    compiled: DebugRuntime,
    input_values: dict[str, torch.Tensor],
    n_pos: int,
) -> torch.Tensor:
    """Pack an input-name → tensor dict into the flat row-tensor layout.

    Delegates to the backend's ``build_prefill``.
    """
    return compiled.build_prefill(input_values, n_pos)


def probe_graph(
    output_node: Node,
    input_values: dict[str, torch.Tensor],
    n_pos: int,
    *,
    d: int = 1024,
    d_head: int = 16,
    d_hidden: int | None = None,
    max_layers: int = 200,
    atol: float = 1e-3,
    verbose: bool = False,
) -> ProbeReport:
    """Compile a graph and run the probe in one call.

    Convenience wrapper around ``compile_headless`` + ``probe_compiled``
    for the usual "I have a graph and I want to know where it breaks"
    workflow.  The compilation uses the same defaults as
    :func:`torchwright.compiler.forward.compile.forward_compile`.
    """
    # Keep the ONNX-backed export module out of compiler-only/HF imports.
    from torchwright.compiler.export import compile_headless

    compiled = compile_headless(
        output_node,
        d=d,
        d_head=d_head,
        max_layers=max_layers,
        verbose=verbose,
        d_hidden=d_hidden,
    )
    return probe_compiled(
        compiled,
        output_node,
        input_values,
        n_pos,
        atol=atol,
    )


# ---------------------------------------------------------------------------
# Direct-inspection harness — residual / attention / layer-diff probes
# ---------------------------------------------------------------------------
#
# The probes above (probe_compiled / probe_graph) check a compiled module
# against its oracle and report *divergence* — useful for confirming
# correctness.  The harness below answers a different question: "what is
# this node's value right now, at this layer, at these positions, in
# this compiled module?"  Callers that already have a failing scene and
# want to localise it reach for these.


def build_prefill_from_input_values(
    compiled: DebugRuntime,
    input_values: dict[str, torch.Tensor],
    n_pos: int,
) -> torch.Tensor:
    """Pack an ``{input_name: tensor}`` dict into the flat prefill layout.

    Thin public wrapper over :func:`_inputs_from_dict` for callers that
    already speak the dict convention from the oracle-probe API but
    want to feed the resulting tensor into the direct-inspection probes
    (which accept a flat prefill so callers can also build one via the
    DOOM pipeline's ``_build_row`` machinery).
    """
    return _inputs_from_dict(compiled, input_values, n_pos)


@dataclass
class ResidualProbe:
    """A graph node's compiled residual values, indexed by layer.

    ``per_layer`` maps layer index to a ``(n_pos, node.d_output)``
    tensor extracted from that layer's post-MLP snapshot.  Only layers
    where the node is materialised appear; an empty dict means the node
    never surfaced in the captured states (either never materialised,
    or the caller restricted ``at_layer`` to a layer that does not
    hold it).
    """

    node: Node
    per_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    layers: list[int] = field(default_factory=list)
    # Shape of any per-layer tensor (all layers share the same shape by
    # construction); empty tuple if ``per_layer`` is empty.
    shape: tuple[int, ...] = ()

    def at(self, layer: int) -> torch.Tensor | None:
        """Value at ``layer``, or ``None`` if the node is not materialised there."""
        return self.per_layer.get(layer)

    def positions(self, positions: "Sequence[int]") -> "ResidualProbe":
        """Return a copy restricted to the given token positions.

        Indexes each per-layer tensor along axis 0.  Useful for zooming
        in on e.g. just the WALL rows of a long prefill.
        """
        pos = list(positions)
        new_per_layer = {layer: tensor[pos] for layer, tensor in self.per_layer.items()}
        new_shape = (len(pos), *self.shape[1:]) if self.shape else ()
        return ResidualProbe(
            node=self.node,
            per_layer=new_per_layer,
            layers=list(self.layers),
            shape=new_shape,
        )


def probe_residual(
    compiled: DebugRuntime,
    prefill: torch.Tensor,
    node: Node,
    *,
    at_layer: int | None = None,
    past_len: int = 0,
    past_kvs: PastKVs | None = None,
) -> ResidualProbe:
    """Extract a node's residual value from each post-MLP layer snapshot.

    Runs the compiled module once with ``return_states=True`` and pulls
    ``node``'s columns out of every captured post-MLP residual-stream
    tensor.  The result is a layer → ``(n_pos, node.d_output)`` tensor
    mapping; callers slice to the positions they care about with
    :meth:`ResidualProbe.positions` or indexed access.

    Args:
        compiled: the module to probe.  Must have a populated
            ``residual_assignment`` (post ``forward_compile``).
        prefill: the flat ``(n_pos, d_input)`` input the compiled
            module expects.  Build with :func:`build_prefill_from_input_values`
            or with the pipeline-specific helpers (e.g. the DOOM
            ``_build_row`` family).
        node: any graph :class:`Node` — including :class:`Concatenate`
            groupings, which resolve to the concat of their leaves'
            columns via :meth:`ResidualAssignment.get_node_indices`.
        at_layer: optional single-layer filter.  When set, only that
            layer's snapshot is scanned; other layers are skipped
            without reading.  When ``None`` (the default), every
            post-MLP snapshot where the node is materialised is
            returned.
        past_len: forwarded to ``compiled._build_res_stream`` for
            KV-cache-aware prefills (default 0 = fresh forward).
        past_kvs: optional per-layer ``(K, V)`` cache — when supplied,
            the probe drives the module through the cached decode path
            instead of a full prefill forward.  Pair with ``past_len``
            to match the cache length.

    Returns:
        A :class:`ResidualProbe` with per-layer values.
    """
    ra, ordered, state_tensor = _run_with_states(
        compiled,
        prefill,
        past_len,
        past_kvs=past_kvs,
    )
    per_layer: dict[int, torch.Tensor] = {}
    shape: tuple[int, ...] = ()
    for layer_i, _name, state in ordered:
        if at_layer is not None and layer_i != at_layer:
            continue
        tensor_pair = state_tensor.get(state)
        if tensor_pair is None:
            continue
        res_tensor, _ = tensor_pair
        value = _extract_compiled_value(node, ra, state, res_tensor)
        if value is None:
            continue
        per_layer[layer_i] = value
        if not shape:
            shape = tuple(value.shape)

    layers = sorted(per_layer.keys())
    return ResidualProbe(
        node=node,
        per_layer=per_layer,
        layers=layers,
        shape=shape,
    )


@contextmanager
def attention_capture(
    net: HeadlessTransformer,
    layer_index: int,
) -> Iterator[dict[str, torch.Tensor | None]]:
    """Monkey-patch ``net.layers[layer_index].attn.attn.forward_cached``.

    Captures the explicit softmax weights and raw logits produced on
    each call.

    On ``__enter__`` the attention module's ``forward_cached`` is
    replaced with a version that reproduces its numerical contract (Q /
    K / V projections, causal mask, softmax, output projection) while
    recording the per-head ``logits`` and ``weights`` tensors into the
    yielded dict.  On ``__exit__`` the original method is restored, even
    on exception.

    Only the *final* call to ``forward_cached`` at this layer is
    retained — if you drive the compiled module through multiple steps,
    the captured tensors reflect the last step.  For multi-step capture
    wrap each step in its own ``attention_capture`` block.

    Yields a dict with keys ``"logits"`` and ``"weights"`` (initially
    ``None``; populated when the hook fires), each of shape
    ``(n_heads, n_queries, n_keys)``.
    """
    captured: dict[str, torch.Tensor | None] = {
        "logits": None,
        "weights": None,
    }
    attn_module = net.layers[layer_index].attn.attn
    orig_fwd_cached = attn_module.forward_cached

    def patched_fwd_cached(
        inp: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        Q = torch.einsum("pd,hdk->hpk", inp, attn_module.query_matrix)
        K_new = torch.einsum("pd,hdk->hpk", inp, attn_module.key_matrix)
        V_new = torch.einsum("pd,hdk->hpk", inp, attn_module.value_matrix)
        if past_kv is not None:
            K = torch.cat([past_kv[0], K_new], dim=1)
            V = torch.cat([past_kv[1], V_new], dim=1)
        else:
            K, V = K_new, V_new
        n_new = inp.shape[0]
        n_total = K.shape[1]
        attn_logits = torch.bmm(Q, K.transpose(1, 2))
        mask = torch.triu(
            torch.ones(n_new, n_total, device=inp.device),
            diagonal=n_total - n_new + 1,
        ).bool()
        attn_logits.masked_fill_(mask.unsqueeze(0), CAUSAL_MASK_SENTINEL)
        weights = torch.softmax(attn_logits, dim=2)
        captured["logits"] = attn_logits.detach().cpu()
        captured["weights"] = weights.detach().cpu()
        weighted = torch.bmm(weights, V)
        output = torch.einsum(
            "hpk,hkd->pd",
            weighted,
            attn_module.output_matrix,
        )
        return output, (K, V)

    attn_module.forward_cached = patched_fwd_cached  # type: ignore[method-assign]
    try:
        yield captured
    finally:
        attn_module.forward_cached = orig_fwd_cached  # type: ignore[method-assign]


@dataclass
class AttentionProbe:
    """Per-head softmax weights and logits at a single query position."""

    attn_node: Attn
    #: Transformer layer index whose attention sublayer hosts ``attn_node``.
    layer_index: int
    #: Query row we extracted.
    query_pos: int
    #: ``(n_heads, n_keys)`` softmax weights at the query row.
    weights: torch.Tensor
    #: ``(n_heads, n_keys)`` pre-softmax logits at the query row.
    logits: torch.Tensor
    #: Optional per-key labels (len == n_keys), supplied by the caller.
    #: Empty list if the caller did not pass ``position_labels``.
    position_labels: list[str] = field(default_factory=list)

    def top(
        self,
        k: int = 8,
        head: int = 0,
    ) -> list[tuple[int, float, str]]:
        """Return ``(key_pos, weight, label)`` for the ``k`` largest weights.

        Weights are taken from head ``head``.  ``label`` is empty if
        ``position_labels`` was not supplied.
        """
        w = self.weights[head]
        n_keys = int(w.shape[0])
        k_eff = min(k, n_keys)
        topk = torch.topk(w, k=k_eff)
        out: list[tuple[int, float, str]] = []
        for val, idx in zip(topk.values.tolist(), topk.indices.tolist(), strict=False):
            label = self.position_labels[idx] if idx < len(self.position_labels) else ""
            out.append((int(idx), float(val), label))
        return out


def probe_attention(
    compiled: DebugRuntime,
    prefill: torch.Tensor,
    attn_node: Attn,
    *,
    query_pos: int,
    past_len: int = 0,
    past_kvs: PastKVs | None = None,
    position_labels: Sequence[str] | None = None,
) -> AttentionProbe:
    """Capture softmax weights and logits at a specific query position.

    Locates the transformer layer whose post-MLP state first surfaces
    ``attn_node`` in the residual assignment, then asks the backend for
    that layer's attention distribution: the in-process backend installs
    the :func:`attention_capture` hook and re-runs the torch forward;
    the ONNX backend fetches the artifact's own ``l{i}_weights`` /
    ``l{i}_logits_masked`` tensors from a run of the real model.

    Args:
        compiled: any :class:`DebugRuntime`.
        prefill: flat ``(n_pos, d_input)`` input for the new rows —
            the full prefill on a fresh run, or the single decode step
            when ``past_kvs`` is supplied.
        attn_node: the graph :class:`Attn` whose attention distribution
            you want.  Must be materialised at some layer — raises
            :class:`ValueError` if no hosting layer is found.
        query_pos: row in the attention output to extract.  Indexes
            into the *new* rows axis: for a fresh prefill this matches
            the row in ``prefill``; for a single-row decode step it's
            always ``0``.
        past_len: absolute base position of the new rows.  Usually
            matches the cache length when ``past_kvs`` is supplied.
        past_kvs: optional KV cache in the backend's own representation
            (``CompiledHeadless``: per-layer ``(K, V)`` list;
            ``OnnxDebugSession``: the ``(K_tuple, V_tuple)`` pair its
            ``step`` returns).  ``None`` means fresh prefill.
        position_labels: optional per-key labels, length equal to the
            total number of key positions (i.e. ``past_len +
            prefill.shape[0]``).  Populates
            :attr:`AttentionProbe.position_labels`.

    Returns:
        :class:`AttentionProbe` with one-query-row slices of the
        captured tensors.
    """
    ra, ordered = compiled.debug_layout()

    layer_index: int | None = None
    for i, _name, state in ordered:
        if ra.has_node(state, attn_node):
            layer_index = i
            break
    if layer_index is None:
        raise ValueError(
            f"Attn node {attn_node!r} not materialised in any layer's "
            f"residual assignment — nothing to hook"
        )

    weights, logits = compiled.capture_attention(
        layer_index, prefill, past_len=past_len, past_kvs=past_kvs
    )

    return AttentionProbe(
        attn_node=attn_node,
        layer_index=layer_index,
        query_pos=query_pos,
        weights=weights[:, query_pos, :],
        logits=logits[:, query_pos, :],
        position_labels=list(position_labels) if position_labels else [],
    )


@dataclass
class LayerDiffRecord:
    """A node's value + delta-vs-reference at a single post-MLP layer."""

    layer_index: int
    state_name: str
    value: torch.Tensor  # (len(positions), node.d_output)
    delta: torch.Tensor  # abs(value - reference)
    max_abs_delta: float


@dataclass
class LayerDiffReport:
    """Layer-by-layer trace of a node's value against a reference."""

    node: Node
    #: One record per layer where ``node`` is materialised within
    #: ``layer_range``, in ascending layer order.
    records: list[LayerDiffRecord] = field(default_factory=list)
    #: Earliest layer with ``max_abs_delta > drift_threshold``; ``None``
    #: if the delta stayed within threshold across every observed layer.
    first_drift_layer: int | None = None
    #: Earliest layer where ``|value - sentinel| < sentinel_tol`` holds
    #: for at least one element; ``None`` either because the caller
    #: did not set ``sentinel`` or because the sentinel never surfaced.
    first_sentinel_layer: int | None = None
    #: Echoed back from the call for reference; ``None`` when sentinel
    #: detection was not requested.
    sentinel_value: float | None = None


def probe_layer_diff(
    compiled: DebugRuntime,
    prefill: torch.Tensor,
    node: Node,
    *,
    reference: torch.Tensor,
    positions: Sequence[int],
    layer_range: tuple[int, int] | None = None,
    drift_threshold: float = 1e-3,
    sentinel: float | None = None,
    sentinel_tol: float = 1e-4,
    past_len: int = 0,
    past_kvs: PastKVs | None = None,
) -> LayerDiffReport:
    """Track a node's value + delta-vs-reference across consecutive layers.

    For every post-MLP snapshot in ``layer_range`` where ``node`` is
    materialised, record the extracted value at ``positions``, the
    absolute delta against ``reference``, and the maximum of that
    delta.  The first layer whose max delta exceeds ``drift_threshold``
    is flagged in :attr:`LayerDiffReport.first_drift_layer`.  If
    ``sentinel`` is set, the first layer whose value equals
    ``sentinel`` within ``sentinel_tol`` (elementwise min) is flagged
    in :attr:`LayerDiffReport.first_sentinel_layer`.

    ``reference`` is a caller-supplied "ground truth" tensor of shape
    ``(len(positions), node.d_output)``.  This function does *not*
    compute it — callers who want oracle-based reference should feed
    the output of ``reference_eval(...)[node][positions]`` themselves.
    Callers who want sentinel-only detection can pass a zero tensor and
    ignore ``first_drift_layer``.

    Args:
        compiled: post-``forward_compile`` module.
        prefill: flat ``(n_pos, d_input)`` input to drive the compiled
            forward pass.
        node: graph :class:`Node` to trace.
        reference: host-known truth; shape
            ``(len(positions), node.d_output)``.
        positions: token-position indices to extract (e.g. the WALL
            rows of a long prefill).
        layer_range: optional ``(start, end)`` filter on layer index;
            both ends inclusive.  ``None`` means "every layer".
        drift_threshold: max-abs-delta above which a layer is flagged
            as the first drift.
        sentinel: optional sentinel value.  When set, the first layer
            whose extracted value contains an element within
            ``sentinel_tol`` of this number is recorded in
            ``first_sentinel_layer``.
        sentinel_tol: tolerance for sentinel match (elementwise).
        past_len: forwarded to ``_build_res_stream``.
        past_kvs: optional KV cache for decode-path probes (e.g.
            inspecting the residual on a single autoregressive step).
            ``None`` means fresh-prefill forward.

    Returns:
        A populated :class:`LayerDiffReport`.
    """
    ra, ordered, state_tensor = _run_with_states(
        compiled,
        prefill,
        past_len,
        past_kvs=past_kvs,
    )

    pos_list = list(positions)
    ref_cpu = reference.detach().cpu()

    lo = layer_range[0] if layer_range is not None else -1
    hi = layer_range[1] if layer_range is not None else 10**9

    report = LayerDiffReport(node=node, sentinel_value=sentinel)
    for layer_i, state_name, state in ordered:
        if not (lo <= layer_i <= hi):
            continue
        tensor_pair = state_tensor.get(state)
        if tensor_pair is None:
            continue
        res_tensor, _ = tensor_pair
        value_all = _extract_compiled_value(node, ra, state, res_tensor)
        if value_all is None:
            continue
        value = value_all[pos_list].detach().cpu()
        if value.shape != ref_cpu.shape:
            raise ValueError(
                f"reference shape {tuple(ref_cpu.shape)} does not match "
                f"value shape {tuple(value.shape)} at layer {layer_i}"
            )
        delta = (value - ref_cpu).abs()
        max_abs = float(delta.max().item())
        report.records.append(
            LayerDiffRecord(
                layer_index=layer_i,
                state_name=state_name,
                value=value,
                delta=delta,
                max_abs_delta=max_abs,
            )
        )
        if report.first_drift_layer is None and max_abs > drift_threshold:
            report.first_drift_layer = layer_i
        if (
            sentinel is not None
            and report.first_sentinel_layer is None
            and float((value - sentinel).abs().min().item()) < sentinel_tol
        ):
            report.first_sentinel_layer = layer_i

    return report


# ---------------------------------------------------------------------------
# Compiled-side Assert checks
# ---------------------------------------------------------------------------


def check_asserts_on_compiled(
    compiled: DebugRuntime,
    asserts: list[Node],
    input_values: dict[str, torch.Tensor],
    n_pos: int,
) -> None:
    """Run each checked node's assert predicates against the compiled residual stream.

    Complements the reference-eval check (which runs predicates as each
    checked node's ``compute`` is called during the oracle walk).  Here
    we run the same predicates against the node's *compiled* value —
    catching invariants that reference math satisfies but compiled
    approximations violate.

    Collect ``asserts`` via
    ``torchwright.graph.asserts.collect_asserts(output_node)`` — before
    or after compiling; compilation never mutates the source graph.

    Raises ``AssertionError`` on the first violation, with the same
    annotation-tagged message format as the reference-eval path.  Nodes
    with no residual assignment (e.g. pure-literal
    subgraphs) are silently skipped — they have no compiled value to
    check.
    """
    if not asserts:
        return

    ra, ordered, state_tensor = _run_with_states(
        compiled,
        compiled.build_prefill(input_values, n_pos),
        past_len=0,
    )
    ordered_states = [s for _, _, s in ordered]

    for node in asserts:
        state = _first_state_with(node, ra, ordered_states)
        if state is None:
            continue  # no residual assignment — can't check on compiled.
        tensor_pair = state_tensor.get(state)
        if tensor_pair is None:
            continue
        res_tensor, _ = tensor_pair
        compiled_val = _extract_compiled_value(node, ra, state, res_tensor)
        if compiled_val is None:
            continue
        for check in node.checks:
            if check.kind == "assert":
                check.run(compiled_val, node)
