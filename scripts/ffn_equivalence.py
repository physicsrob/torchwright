"""Equivalence harness for the FFN-IR (formerly block-IR) step-1 refactor
(``docs/block_ir_step1_plan.md``, "Equivalence harness").

For a given graph builder it compiles **both** code paths — historically the
chain-mined path vs the blockified one; since Phase 2b/3 the ops layer builds
FFNs natively, so both builds are the same graph and the "FFN path" is
the build certified at the lowering boundary (``compiler.lower``, blockify's
successor) — runs identical inputs, and reports the max output divergence
plus a compile-metrics tuple for each path.

Two collection modes, so this scales from a unit-test graph to the d=8192
flagship:

- **Full** (``run_output=True``): ``compile_headless`` both paths with weights,
  run a random (or supplied) input through each, and report the max abs output
  divergence.  Use on graphs small enough to materialize weights.
- **Schedule-only metrics** (always): run the heuristic ``LayerScheduler`` in
  schedule-only mode (no weight tensors — memory-safe at d=8192) and capture
  ``n_layers``, total and per-layer attention heads, peak MLP hidden usage, and
  residual peak.  Both paths use identical seeding, so the tuples are directly
  comparable; any cost regression (layers/heads/hidden) is the Gate-C
  stop-and-report signal.

The scheduler at flagship scale is the eager heuristic (production's
optimize=2 CP-SAT times out cold and falls back to it), so both paths are
compared under the same scheduler mode by construction here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from torchwright.compiler.forward.compile import _count_heads_by_type
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.scheduler import LayerScheduler
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.lower import lower
from torchwright.graph import Concatenate, Node
from torchwright.graph.misc import LiteralValue
from torchwright.graph.node import reserve_node_id_above


@dataclass
class ScheduleMetrics:
    n_layers: int
    total_heads: int
    per_layer_heads: List[int]
    peak_hidden: int
    residual_peak: int

    def as_tuple(self):
        return (self.n_layers, self.total_heads, self.peak_hidden, self.residual_peak)


@dataclass
class EquivalenceReport:
    chain_metrics: ScheduleMetrics
    ffn_metrics: ScheduleMetrics
    max_output_divergence: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def regressions(self) -> List[str]:
        """Metrics where the FFN path is worse than the chain path."""
        out = []
        c, b = self.chain_metrics, self.ffn_metrics
        if b.n_layers > c.n_layers:
            out.append(f"n_layers {c.n_layers} -> {b.n_layers}")
        if b.total_heads > c.total_heads:
            out.append(f"total_heads {c.total_heads} -> {b.total_heads}")
        if b.peak_hidden > c.peak_hidden:
            out.append(f"peak_hidden {c.peak_hidden} -> {b.peak_hidden}")
        if b.residual_peak > c.residual_peak:
            out.append(f"residual_peak {c.residual_peak} -> {b.residual_peak}")
        return out

    def format(self) -> str:
        lines = ["FFN-IR equivalence report", "=" * 40]
        lines.append(f"  metric          {'chain':>12} {'ffn':>12}")
        for label, cval, bval in (
            ("n_layers", self.chain_metrics.n_layers, self.ffn_metrics.n_layers),
            (
                "total_heads",
                self.chain_metrics.total_heads,
                self.ffn_metrics.total_heads,
            ),
            (
                "peak_hidden",
                self.chain_metrics.peak_hidden,
                self.ffn_metrics.peak_hidden,
            ),
            (
                "residual_peak",
                self.chain_metrics.residual_peak,
                self.ffn_metrics.residual_peak,
            ),
        ):
            lines.append(f"  {label:<14} {cval:>12} {bval:>12}")
        if self.max_output_divergence is not None:
            lines.append(f"  max output divergence: {self.max_output_divergence:.3e}")
        regs = self.regressions()
        if regs:
            lines.append("  COST REGRESSIONS (stop-and-report): " + ", ".join(regs))
        else:
            lines.append("  no cost regression")
        for n in self.notes:
            lines.append("  note: " + n)
        return "\n".join(lines)


def _seed_residual_map(graph: GraphAnalyzer, d: int):
    """Replicate forward_compile's residual-stream seed for schedule-only runs:
    the const-1 self-match column and every input node (no RMSNorm reservation
    — off by default, and irrelevant to the chain-vs-FFN comparison as long as
    both paths use the same seed).  The runtime always zero-initialises, so the
    whole free pool starts clean and no cancel bookkeeping is needed."""
    input_nodes = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]
    rmap = ResidualStreamMap(d)
    reserve_node_id_above(graph.get_all_nodes())
    const_one = LiteralValue(torch.ones(1), name="rope_self_match_const_one")
    rmap.allocate(const_one)
    for n in input_nodes:
        rmap.allocate(n)
    return rmap, set(input_nodes)


def schedule_metrics(
    output_node: Node,
    *,
    d: int,
    d_head: int,
    d_hidden: Optional[int] = None,
    max_layers: int = 800,
) -> ScheduleMetrics:
    """Run the heuristic scheduler schedule-only (no weight tensors) and capture
    the compile-metrics tuple.  Memory-safe at flagship scale."""
    d_hidden = d if d_hidden is None else d_hidden
    graph = GraphAnalyzer(output_node)
    output_node = graph.get_output_node()
    rmap, computed = _seed_residual_map(graph, d)
    sched = LayerScheduler(graph, d, d_head, pos_encoding=None, d_hidden=d_hidden)

    per_layer_heads: List[int] = []
    peak_hidden = 0
    residual_peak = d - rmap.get_free_count()
    n_layers = 0
    for i in range(max_layers):
        if output_node in computed:
            break
        attn_ops, mlp_ops, _biased = sched.schedule_layer(rmap, computed)
        heads = sum(_count_heads_by_type(attn_ops, d_head).values())
        per_layer_heads.append(heads)
        slots = sum(len(op.mlp_slots) for op in mlp_ops if op.mlp_slots)
        peak_hidden = max(peak_hidden, slots)
        residual_peak = max(residual_peak, d - rmap.get_free_count())
        for node in graph.get_all_nodes():
            if isinstance(node, Concatenate) and node not in computed:
                if all(leaf in computed for leaf in flatten_concat_nodes([node])):
                    computed.add(node)
        if not attn_ops and not mlp_ops:
            break
        n_layers = i + 1

    return ScheduleMetrics(
        n_layers=n_layers,
        total_heads=sum(per_layer_heads),
        per_layer_heads=per_layer_heads,
        peak_hidden=peak_hidden,
        residual_peak=residual_peak,
    )


def schedule_trace(
    output_node: Node,
    *,
    d: int,
    d_head: int,
    d_hidden: Optional[int] = None,
    max_layers: int = 800,
) -> List[dict]:
    """Schedule-only, but return per-layer detail for tracing occupancy diffs.

    Each list entry is a dict for one layer: ``hidden`` (total MLP slots used),
    ``composites`` (a list of ``(annotation, width, slots, node_id)`` for every
    compute_ffn op that layer), and ``layer`` (index).  The MLP composite is
    the FFN node, matchable across schedules by ``(annotation, width)``.
    """
    d_hidden = d if d_hidden is None else d_hidden
    graph = GraphAnalyzer(output_node)
    output_node = graph.get_output_node()
    rmap, computed = _seed_residual_map(graph, d)
    sched = LayerScheduler(graph, d, d_head, pos_encoding=None, d_hidden=d_hidden)

    trace: List[dict] = []
    for i in range(max_layers):
        if output_node in computed:
            break
        attn_ops, mlp_ops, _biased = sched.schedule_layer(rmap, computed)
        composites = []
        hidden = 0
        for op in mlp_ops:
            if op.mlp_slots:
                hidden += len(op.mlp_slots)
            if op.op_type == "compute_ffn":
                composites.append(
                    (
                        op.node.annotation,
                        op.node.d_output,
                        len(op.mlp_slots),
                        op.node.node_id,
                    )
                )
        trace.append({"layer": i, "hidden": hidden, "composites": composites})
        for node in graph.get_all_nodes():
            if isinstance(node, Concatenate) and node not in computed:
                if all(leaf in computed for leaf in flatten_concat_nodes([node])):
                    computed.add(node)
        if not attn_ops and not mlp_ops:
            break
    return trace


def equivalence_report(
    build_fn: Callable[[], Node],
    *,
    d: int,
    d_head: int,
    d_hidden: Optional[int] = None,
    run_output: bool = False,
    input_tensor: Optional[torch.Tensor] = None,
    n_pos: int = 4,
) -> EquivalenceReport:
    """Compile both paths from ``build_fn`` and report metrics (+ optional
    output divergence).

    ``build_fn`` must build the graph deterministically and return the output
    node; it is called once per path so each path gets a fresh graph.
    """
    # Chain path.
    chain_out = build_fn()
    chain_metrics = schedule_metrics(
        chain_out,
        d=d,
        d_head=d_head,
        d_hidden=d_hidden,
    )

    # FFN path (certified at the lowering boundary; blockify's successor).
    block_out = lower(build_fn()).output_node
    ffn_metrics = schedule_metrics(
        block_out,
        d=d,
        d_head=d_head,
        d_hidden=d_hidden,
    )

    report = EquivalenceReport(chain_metrics=chain_metrics, ffn_metrics=ffn_metrics)

    if run_output:
        from torchwright.compiler.export import compile_headless

        c_chain = compile_headless(
            build_fn(),
            d=d,
            d_head=d_head,
            d_hidden=d_hidden,
        )
        c_block = compile_headless(
            lower(build_fn()).output_node,
            d=d,
            d_head=d_head,
            d_hidden=d_hidden,
        )
        n_in = sum(w for _, _, w in c_chain._input_specs)
        if input_tensor is None:
            input_tensor = torch.randn(n_pos, n_in)
        oc = c_chain(input_tensor)
        ob = c_block(input_tensor)
        report.max_output_divergence = float((oc - ob).abs().max())

    return report


if __name__ == "__main__":
    # Self-check on a small two-MLP graph.
    from torchwright.ops.inout_nodes import create_input
    from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

    def build():
        x = create_input("x", 8, value_range=(-1.0, 1.0))
        g = torch.Generator().manual_seed(11)
        h = linear_relu_linear(
            x,
            torch.randn(16, 8, generator=g) * 0.3,
            torch.randn(16, generator=g) * 0.1,
            torch.randn(16, 8, generator=g) * 0.3,
            torch.randn(8, generator=g) * 0.1,
            name="mlp1",
        )
        return linear_relu_linear(
            h,
            torch.randn(12, 8, generator=g) * 0.3,
            torch.randn(12, generator=g) * 0.1,
            torch.randn(12, 4, generator=g) * 0.3,
            torch.randn(4, generator=g) * 0.1,
            name="mlp2",
        )

    rep = equivalence_report(build, d=64, d_head=8, run_output=True)
    print(rep.format())
