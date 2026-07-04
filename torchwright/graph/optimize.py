"""Graph optimization passes.

These passes transform the computation graph before compilation to reduce
layer count and parameter overhead.
"""

from typing import Dict, List, Set

from torchwright.graph import Concatenate, Node
from torchwright.graph.ffn import FFN
from torchwright.graph.linear import Linear


def _consumer_map(all_nodes: Set[Node]) -> Dict[Node, List[Node]]:
    consumers: Dict[Node, List[Node]] = {n: [] for n in all_nodes}
    for node in all_nodes:
        for inp in node.inputs:
            if inp in consumers:
                consumers[inp].append(node)
    return consumers


def _fuse_linear_into_linear(l1: Linear, l2: Linear) -> None:
    """Fuse ``l1 -> l2`` (both Linear) by mutating ``l2`` in place.

    ``l2`` survives (its identity is preserved so consumers and output nodes
    stay valid); ``l1`` is orphaned::

        y2 = (x @ M1 + b1) @ M2 + b2 = x @ (M1 @ M2) + (b1 @ M2 + b2)
    """
    fused_matrix = l1.output_matrix @ l2.output_matrix
    fused_bias = l1.output_bias @ l2.output_matrix + l2.output_bias
    l2.inputs = [l1.inputs[0]]
    l2.output_matrix = fused_matrix
    l2.output_bias = fused_bias
    l2.d_input = l1.output_matrix.shape[0]
    if l1.name and l2.name:
        l2.name = f"fused_{l1.name}_{l2.name}"


def _fold_linear_into_ffn_gate(u: Linear, b: FFN) -> None:
    """Fold an upstream Linear ``u`` into FFN ``b``'s gate (and up) projection.

    ``b`` survives (mutated in place); ``u`` is orphaned.  With
    ``gate_in = (x @ M_u + b_u) @ gate_proj.T + gate_bias``::

        gate_proj' = gate_proj @ M_u.T
        gate_bias' = b_u @ gate_proj.T + gate_bias

    and the same for ``up_proj`` when the FFN is gated.  This is width-safe by
    construction — the FFN is realized whole, so folding into its gate never
    ejects the ReLU (the property the deleted ejection gate used to police in
    the chain world).
    """
    m_u = u.output_matrix  # (d_x, d_u)
    b.gate_bias = u.output_bias @ b.gate_proj.t() + b.gate_bias
    b.gate_proj = b.gate_proj @ m_u.t()
    if b.up_proj is not None:
        b.up_bias = u.output_bias @ b.up_proj.t() + b.up_bias
        b.up_proj = b.up_proj @ m_u.t()
    b.inputs = [u.inputs[0]]
    b.d_input = m_u.shape[0]
    if u.name and b.name:
        b.name = f"fused_{u.name}_{b.name}"


def _fold_ffn_into_linear(b: FFN, consumers: Dict[Node, List[Node]]) -> None:
    """Fold FFN ``b``'s output projection into its sole downstream Linear.

    ``b``'s only consumer is a Linear ``l``.  The fused node is still an FFN
    (lanes + output projection), so ``b`` survives and ``l`` is orphaned; every
    consumer of ``l`` is rewired to ``b``.  With ``z = (lane @ out_proj +
    out_bias) @ M_l + b_l``::

        out_proj' = out_proj @ M_l
        out_bias' = out_bias @ M_l + b_l
    """
    (l,) = consumers[b]
    assert isinstance(l, Linear)
    b.out_bias = b.out_bias @ l.output_matrix + l.output_bias
    b.out_proj = b.out_proj @ l.output_matrix
    b.d_output = l.output_matrix.shape[1]
    # This fold inverts survivorship: b's VALUE becomes what l's was, so a
    # semantic affine override installed on b describes the pre-fold value
    # and must not be re-applied by the bounds refresh.  (The other two
    # folds preserve the survivor's value, so their overrides stay valid.)
    b._semantic_affine_override = None
    if b.name and l.name:
        b.name = f"fused_{b.name}_{l.name}"
    for consumer in list(consumers.get(l, [])):
        consumer.replace_input(l, b)


def _fuse_one_pass(output_nodes: Set[Node], verbose: bool, mutated: Set[Node]) -> int:
    """One fusion round: apply a set of non-overlapping folds, return the count.

    Each node participates in at most one fold per pass (a node touched as a
    fold's survivor or orphaned input is skipped for the rest of the pass);
    chains resolve across successive passes.  Candidates are gathered and
    applied in ``node_id`` order for determinism (the schedule cache keys on
    node structure).  Every surviving (mutated) node is added to ``mutated`` so
    the caller can refresh the affine bounds a fold invalidates.
    """
    from torchwright.compiler.utils import get_ancestor_nodes

    all_nodes = get_ancestor_nodes(output_nodes)
    consumers = _consumer_map(all_nodes)

    touched: Set[Node] = set()
    applied = 0

    def param_delta_ok(new_params: int, old_params: int) -> bool:
        # Skip folds that would grow the parameter count (bottleneck patterns).
        return new_params <= old_params

    # Linear -> Linear, and FFN -> Linear (out-proj fold): keyed on the
    # downstream Linear.  Linear -> FFN (gate fold): keyed on the FFN.
    for node in sorted(all_nodes, key=lambda n: n.node_id):
        if node in touched:
            continue

        if isinstance(node, Linear):
            inp = node.inputs[0]
            if isinstance(inp, Concatenate) or inp in touched:
                continue
            if len(consumers.get(inp, [])) != 1:
                continue

            if isinstance(inp, Linear):
                l1, l2 = inp, node
                d_in = l1.output_matrix.shape[0]
                d_mid = l1.output_matrix.shape[1]
                d_out = l2.output_matrix.shape[1]
                old = d_in * d_mid + d_mid + d_mid * d_out + d_out
                new = d_in * d_out + d_out
                if not param_delta_ok(new, old):
                    continue
                _fuse_linear_into_linear(l1, l2)
                touched.add(l1)
                touched.add(l2)
                mutated.add(l2)
                applied += 1

            elif isinstance(inp, FFN):
                # Fold the FFN's out_proj into this downstream Linear.  The
                # FFN becomes the survivor, so the Linear must not be a
                # caller-held output node (its identity would be lost).
                b, l = inp, node
                if l in output_nodes:
                    continue
                n_lanes = b.n_lanes
                d_b = b.d_output
                d_z = l.output_matrix.shape[1]
                old = (n_lanes * d_b + d_b) + (d_b * d_z + d_z)
                new = n_lanes * d_z + d_z
                if not param_delta_ok(new, old):
                    continue
                _fold_ffn_into_linear(b, consumers)
                touched.add(b)
                touched.add(l)
                mutated.add(b)
                applied += 1

        elif isinstance(node, FFN):
            inp = node.inputs[0]
            if not isinstance(inp, Linear) or inp in touched:
                continue
            if len(consumers.get(inp, [])) != 1:
                continue
            u, b = inp, node
            d_x = u.output_matrix.shape[0]
            d_u = u.output_matrix.shape[1]
            n_lanes = b.n_lanes
            gated = b.up_proj is not None
            u_params = d_x * d_u + d_u
            per_proj_old = n_lanes * d_u
            per_proj_new = n_lanes * d_x
            n_proj = 2 if gated else 1
            old = u_params + n_proj * per_proj_old
            new = n_proj * per_proj_new
            if not param_delta_ok(new, old):
                continue
            _fold_linear_into_ffn_gate(u, b)
            touched.add(u)
            touched.add(b)
            mutated.add(b)
            applied += 1

    if verbose and applied:
        print(f"  fused {applied} pair(s) this pass")
    return applied


def fuse_consecutive_linears(
    output_nodes: Set[Node],
    verbose: bool = False,
) -> int:
    """Fuse adjacent linear maps, FFN-aware, until no more folds apply.

    Three folds, each width-safe and gated so it never grows the parameter
    count:

    - **Linear -> Linear** (``l1``'s sole consumer is ``l2``): ``l2`` absorbs
      ``l1`` (``M1 @ M2``, ``b1 @ M2 + b2``).  Saves one compiled layer.
    - **Linear -> FFN** (the Linear's sole consumer is the FFN): the Linear
      folds into the FFN's gate (and up) projection.  In the chain world this
      was the "eject the ReLU" case the width-safe gate had to decline; with a
      first-class FFN there is nothing to eject, so it is always legal.
    - **FFN -> Linear** (the FFN's sole consumer is the Linear, and the
      Linear is not a caller-held output): the FFN's output projection folds
      into the downstream Linear (``out_proj @ M_l``), the FFN absorbing it.

    All folds mutate a surviving node in place; ``output_nodes`` stay valid
    references (an FFN-into-output-Linear fold is declined to preserve the
    caller's output identity).

    Because each fold rewrites a surviving node's weights and inputs in place,
    that node's eagerly-cached ``_affine_bound`` / ``_structural_type`` (and
    every downstream node's, which was derived from it) go stale.  A stale
    bound can be unsound once a downstream ``Assert``'s claim later tightens
    a node's structural type (the lowering strip, applied to the compile's
    private copy of this graph) — the affine and structural ranges then
    disagree and the RMSNorm certification's soundness check fires.  So
    after all folds settle, refresh the bounds of every mutated node and
    everything downstream of one (see
    :func:`_recompute_bounds_after_fusion`).  The refresh re-derives claims
    through the Assert affine rule because this pass runs on *source*
    graphs, which keep their wrappers.

    Args:
        output_nodes: The graph's output nodes (used to find all ancestors and
            to protect caller-held output identity).
        verbose: Print per-pass fold counts.

    Returns:
        Total number of folds performed.
    """
    total = 0
    mutated: Set[Node] = set()
    while True:
        n = _fuse_one_pass(output_nodes, verbose, mutated)
        total += n
        if n == 0:
            break
    if mutated:
        _recompute_bounds_after_fusion(output_nodes, mutated)
    if verbose and total:
        print(f"Fused {total} pair(s) total")
    return total


def _recompute_bounds_after_fusion(output_nodes: Set[Node], mutated: Set[Node]) -> None:
    """Refresh ``_structural_type`` / ``_affine_bound`` on every node a fold
    made stale — the mutated survivors and everything downstream of one.

    Both are computed eagerly in ``Node.__init__`` from the node's inputs, so
    an in-place fold (new weights, new input) leaves them describing the
    pre-fold graph.  Recomputing only the mutated node is not enough: its own
    recompute reads its inputs' bounds, which are correct, but a node *further*
    downstream was derived from the mutated node's pre-fold bound, so it must
    recompute too — and only after its inputs have.  Walking in ``node_id``
    order is a valid topological order (a node's inputs always have smaller
    ids, and every fold rewires inputs to strictly-smaller-id nodes), so one
    forward sweep recomputes each dirty node after its inputs.
    """
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.graph.affine_rules import refresh_node_caches

    reachable = get_ancestor_nodes(output_nodes)
    dirty: Set[Node] = set(mutated)
    for node in sorted(reachable, key=lambda n: n.node_id):
        if node in mutated or any(inp in dirty for inp in node.inputs):
            refresh_node_caches(node)
            dirty.add(node)


def optimize_graph(
    output_nodes: Set[Node],
    verbose: bool = False,
) -> None:
    """Apply all graph optimization passes.

    Modifies the graph in-place by redirecting node inputs to optimized
    versions.

    Args:
        output_nodes: The graph's output nodes.
        verbose: Print optimization details.
    """
    fused = fuse_consecutive_linears(output_nodes, verbose=verbose)
    if verbose:
        print(f"Graph optimization: fused {fused} pairs")
