"""Backend-agnostic residual-snapshot extraction and debug checks.

The debug machinery has two backends — the in-process
:class:`~torchwright.compiler.export.CompiledHeadless` (torch forward
with per-sublayer state capture) and the
:class:`~torchwright.debug.onnx_debug.OnnxDebugSession` (the real ONNX
artifact re-run with its internal per-layer residual tensors promoted
to outputs).  Both reduce a debug run to the same three things:

* a :class:`ResidualAssignment` mapping nodes to columns per state,
* an ordered list of post-MLP states,
* ``state_tensor``: ``{state: (residual_tensor, label)}`` snapshots.

Everything in this module operates on that triple only, so the
self-consistency walk, Assert/DebugWatch evaluation, and per-node value
extraction are guaranteed to behave identically on both backends.

This module must stay import-light (graph + residual_assignment only):
it is imported by ``compiler/export.py``, ``debug/probe.py``, and
``debug/onnx_debug.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from torchwright.compiler.residual_assignment import (
    ResidualAssignment,
    ResidualStreamState,
)
from torchwright.graph import Concatenate, Node
from torchwright.graph.misc import Assert, DebugWatch

#: ``{state: (residual_tensor, label)}`` — the label embeds the layer
#: index (e.g. ``"layer_5_mlp_out_state"``) because ``state.name`` alone
#: is ambiguous (every layer's MLP out state shares the same name).
StateTensors = Dict[ResidualStreamState, Tuple[torch.Tensor, str]]


def extract_compiled_value(
    node: Node,
    ra: ResidualAssignment,
    state: ResidualStreamState,
    res_tensor: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Pull a graph node's compiled value out of a post-sublayer residual stream.

    Returns ``None`` if the node is not materialised at ``state``.
    Concatenate nodes resolve to the concatenation of their children's
    columns via ``ResidualAssignment.get_node_indices``.
    """
    nodes_here = ra.get_nodes(state)
    if node not in nodes_here and not isinstance(node, Concatenate):
        return None
    try:
        cols = ra.get_node_indices(state, node)
    except KeyError:
        return None
    if not cols:
        return None
    return res_tensor[:, cols]


def first_state_with(
    node: Node,
    ra: ResidualAssignment,
    ordered_states: List[ResidualStreamState],
) -> Optional[ResidualStreamState]:
    """Earliest sublayer state in which ``node`` is materialised."""
    if isinstance(node, Concatenate):
        # Concatenate is resolved transparently — pick the earliest
        # state where *all* of its leaves are present.
        children = [i for i in node.inputs]
        best: Optional[ResidualStreamState] = None
        for st in ordered_states:
            if all(first_state_with(c, ra, [st]) is not None for c in children):
                best = st
                break
        return best
    for st in ordered_states:
        if ra.has_node(st, node):
            return st
    return None


def run_consistency_check(
    ordered_states: List[ResidualStreamState],
    state_tensor: StateTensors,
    ra: ResidualAssignment,
    atol: float,
    extra_cause: str = "",
) -> None:
    """Residual-stream self-consistency: raise if a live node's columns changed.

    For every node that appears in multiple captured states, verifies the
    value at its assigned columns is identical across all of them (up to
    ``atol`` per column, to tolerate fp-rounding noise).  A diff exceeding
    ``atol`` means something overwrote the node's columns before all
    consumers read them.

    ALL violating nodes are collected (one entry per node, the first
    state where its value diverged from the initial reading) and a single
    error lists every one.  Aborting on the first failure used to hide
    secondary aliasings behind the first, which made it easy to
    misdiagnose a whole affected class of nodes as a single isolated
    issue.

    ``extra_cause`` lets a backend append backend-specific candidate
    causes to the error preamble (e.g. the ONNX backend adds emission
    and sidecar-remap bugs to the in-process backend's scheduler/
    allocator cause).
    """
    captured_states = [s for s in ordered_states if s in state_tensor]
    # Each entry: (value_tensor, label_from_state_tensor, cols_list).
    node_first_value: Dict[Node, Tuple[torch.Tensor, str, list]] = {}
    violations: List[tuple] = []
    violated_nodes: set = set()
    for state in captured_states:
        state_label = state_tensor[state][1]
        for node in ra.get_nodes(state):
            if node in violated_nodes:
                continue
            res_tensor, _ = state_tensor[state]
            val = extract_compiled_value(node, ra, state, res_tensor)
            if val is None:
                continue
            cols = list(ra.get_node_indices(state, node))
            if node in node_first_value:
                first_val, first_label, first_cols = node_first_value[node]
                if first_val.shape == val.shape:
                    diff = (first_val - val).abs()
                    max_diff = diff.max().item()
                    if max_diff > atol:
                        violations.append(
                            (
                                node,
                                first_label,
                                state_label,
                                cols,
                                first_val.detach().clone(),
                                val.detach().clone(),
                                max_diff,
                            )
                        )
                        violated_nodes.add(node)
            else:
                node_first_value[node] = (
                    val.detach().clone(),
                    state_label,
                    cols,
                )

    if not violations:
        return

    # Sort by max_abs_diff descending so the biggest offenders are at
    # the top.
    violations.sort(key=lambda v: v[6], reverse=True)
    msg_lines = [
        f"Residual-stream self-consistency failure (atol={atol:g}) — "
        f"{len(violations)} node(s):{extra_cause}"
    ]
    for i, (
        node,
        first_label,
        later_label,
        cols,
        first_val,
        val,
        max_diff,
    ) in enumerate(violations, 1):
        diff = (first_val - val).abs()
        if diff.ndim >= 2:
            per_col_max = diff.max(dim=0).values
        else:
            per_col_max = diff
        worst_col_idx = int(per_col_max.argmax().item())
        worst_col = cols[worst_col_idx] if worst_col_idx < len(cols) else None
        node_name = node.annotation or node.name or f"node_{node.node_id}"
        # Summarise value tensors at the worst col if they span many
        # positions: show min/max across positions rather than dumping
        # the whole list so the error stays readable when the prefill
        # is long.
        a_slice = first_val[..., worst_col_idx]
        b_slice = val[..., worst_col_idx]
        if a_slice.ndim == 0 or a_slice.numel() <= 8:
            a_fmt = a_slice.tolist()
            b_fmt = b_slice.tolist()
        else:
            a_fmt = (
                f"n={a_slice.numel()} min={float(a_slice.min()):g} "
                f"max={float(a_slice.max()):g}"
            )
            b_fmt = (
                f"n={b_slice.numel()} min={float(b_slice.min()):g} "
                f"max={float(b_slice.max()):g}"
            )
        msg_lines.append(
            f"\n  [{i}] {node_name} (id={node.node_id}, "
            f"type={type(node).__name__}, width={len(cols)})\n"
            f"      first:     {first_label}\n"
            f"      later:     {later_label}\n"
            f"      cols:      {cols if len(cols) <= 16 else str(cols[:16]) + '...'}\n"
            f"      worst col: residual[{worst_col}] (node index {worst_col_idx})\n"
            f"      first val: {a_fmt}\n"
            f"      later val: {b_fmt}\n"
            f"      max_abs_diff: {max_diff:.6g}"
        )
    raise RuntimeError("".join(msg_lines))


def check_debug_predicates(
    asserts: List[Assert],
    watches: List[DebugWatch],
    ra: ResidualAssignment,
    ordered_states: List[ResidualStreamState],
    state_tensor: StateTensors,
) -> None:
    """Run Assert (raises) and DebugWatch (prints) predicates on captured values.

    Each wrapper's innermost non-wrapper target is looked up at the
    earliest state where it is materialised; wrappers whose targets have
    no residual assignment are silently skipped — there is no compiled
    value to check.
    """
    for assert_node in asserts:
        target = assert_node.inputs[0]
        while isinstance(target, (Assert, DebugWatch)):
            target = target.inputs[0]
        state = first_state_with(target, ra, ordered_states)
        if state is None:
            continue
        tensor_pair = state_tensor.get(state)
        if tensor_pair is None:
            continue
        res_tensor, _ = tensor_pair
        compiled_val = extract_compiled_value(target, ra, state, res_tensor)
        if compiled_val is None:
            continue
        assert_node._check(compiled_val)

    for watch_node in watches:
        target = watch_node.inputs[0]
        while isinstance(target, (Assert, DebugWatch)):
            target = target.inputs[0]
        state = first_state_with(target, ra, ordered_states)
        if state is None:
            continue
        tensor_pair = state_tensor.get(state)
        if tensor_pair is None:
            continue
        res_tensor, _ = tensor_pair
        compiled_val = extract_compiled_value(target, ra, state, res_tensor)
        if compiled_val is None:
            continue
        watch_node._check(compiled_val)
