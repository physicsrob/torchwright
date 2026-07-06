"""Graph optimization passes.

These passes transform the computation graph before compilation to reduce
layer count and parameter overhead.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from torchwright.graph import Concatenate, LiteralValue, Node
from torchwright.graph.ffn import FFN
from torchwright.graph.linear import Linear


@dataclass
class FoldLog:
    """Value-identity bookkeeping for a fusion run.

    Folds mutate nodes in place, and two of them change which *value* a
    surviving node computes.  A caller that keys per-node data by graph
    node (the lowering boundary's source→copy ``node_map``) needs to
    know, after fusion:

    - ``moves``: orphaned node → survivor that now computes its value
      (the FFN→Linear fold: the FFN's post-fold output IS what the
      orphaned downstream Linear used to compute).
    - ``value_changed``: nodes that survive in the graph but no longer
      compute their pre-fusion value (the move survivors themselves,
      and Concatenates whose leaves were absorbed).

    Every other orphan's value simply ceases to exist; reachability
    from the output identifies those without bookkeeping here.
    """

    moves: Dict[Node, Node] = field(default_factory=dict)
    value_changed: Set[Node] = field(default_factory=set)

    def record_move(self, orphan: Node, survivor: Node) -> None:
        # A survivor absorbing a second downstream Linear stops holding
        # the first orphan's value — that value is then simply lost.
        for prev, holder in list(self.moves.items()):
            if holder is survivor:
                del self.moves[prev]
        self.moves[orphan] = survivor
        self.value_changed.add(survivor)

    def record_value_changed(self, node: Node) -> None:
        self.value_changed.add(node)


# --- Fold policy for checked values (docs/assert_metadata_plan.md) --------
#
# A node carrying checks (``node.checks``) stays materialized: a fold
# that would absorb it — erase the value its predicates and range claim
# describe — is DECLINED.  Rationale: the runtime checkability of an
# asserted value (``debug=True`` re-checks it against the residual
# stream) is worth more than the layer a fold would save; which folds
# are worth deleting checkability for is a measurable policy question,
# and until measured the answer is "none".  This reproduces exactly the
# blocking the old Assert wrapper nodes caused by accident (an
# interposed wrapper broke the ``producer's sole consumer`` pattern
# match).  One deliberate widening: a check whose returned handle the
# caller discarded used to detach from the graph (dead wrapper) and
# blocked nothing; as metadata it stays live and declines folds like
# any other check.
#
# The FFN->Linear fold is the one case where the orphaned node's VALUE
# survives (it moves onto the FFN): there the orphan's checks and claim
# MIGRATE to the survivor instead of blocking the fold — mirroring how
# a wrapper above the orphaned Linear used to be rewired onto the FFN.


def _blocks_absorb(node: Node) -> bool:
    """True when *node*'s value may not be erased by a fold."""
    return bool(node.checks)


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


def _fold_ffn_into_linear(
    b: FFN, consumers: Dict[Node, List[Node]], fold_log: FoldLog
) -> None:
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
    # b's value becomes what l's was: l's value moves onto b, and b's
    # own pre-fold value ceases to exist.
    fold_log.record_move(orphan=l, survivor=b)
    b.out_bias = b.out_bias @ l.output_matrix + l.output_bias
    b.out_proj = b.out_proj @ l.output_matrix
    b.d_output = l.output_matrix.shape[1]
    # l's checks and range claim describe l's value, which now lives on
    # b — migrate them so the value stays runtime-checkable.  b's own
    # metadata is guaranteed empty here (a checked b declines the fold),
    # so this is a plain transfer, not a merge.
    if l.checks:
        b.checks = list(l.checks)
        b.claimed_type = l.claimed_type
        b.integer_claim = l.integer_claim
    # This fold inverts survivorship: b's VALUE becomes what l's was, so a
    # semantic affine override installed on b describes the pre-fold value
    # and must not be re-applied by the bounds refresh.  (The other two
    # folds preserve the survivor's value, so their overrides stay valid.)
    b._semantic_affine_override = None
    if b.name and l.name:
        b.name = f"fused_{b.name}_{l.name}"
    for consumer in list(consumers.get(l, [])):
        consumer.replace_input(l, b)


def _fold_through_concatenate(
    l: Linear,
    c: Concatenate,
    output_nodes: Set[Node],
    consumers: Dict[Node, List[Node]],
    touched: Set[Node],
    mutated: Set[Node],
    fold_log: FoldLog,
) -> int:
    """Fold foldable leaves of single-consumer ``c`` into ``l``'s row blocks.

    ``Linear(Concatenate(..., leaf_i, ...))`` reads leaf ``i`` through the
    row block ``M_i`` of ``l.output_matrix`` at the leaf's column offset::

        y = sum_i (leaf_i @ M_i) + b

    Three leaf rewrites, all exact, mutating only the survivors ``l`` and
    ``c`` (leaves are orphaned, never mutated):

    - a single-consumer ``Concatenate`` leaf is **spliced** inline (its
      inputs replace it in ``c.inputs``) — value-identical, and exposes
      its own leaves to the folds below on a later pass;
    - a ``LiteralValue`` leaf's constant contribution ``v @ M_i`` moves
      into ``l.output_bias`` and the leaf is dropped with its block rows
      — skipped when every leaf is a literal (an input-less Linear is
      not representable);
    - a single-consumer ``Linear`` leaf is **absorbed**: its block
      becomes ``leaf.output_matrix @ M_i``, its bias contribution joins
      ``l.output_bias``, and ``c`` reads the leaf's input directly.
      Guarded per leaf so the fold never grows the parameter count.

    When exactly one leaf remains, the concat itself is bypassed
    (``l`` reads that leaf directly) so the plain Linear→Linear /
    FFN→Linear folds can continue in later passes.

    Declined outright (except splices, which are value-identical) when
    ``c`` carries a semantic affine override — the folds change ``c``'s
    value/width and would leave the override describing the pre-fold
    node.

    Returns the number of leaf rewrites applied.
    """
    # A checked concat stays whole: every fold below changes c's value
    # or width, invalidating what its predicates read (the composite
    # two-input asserts attach here).  Splices are value-identical but
    # declined too — exact parity with the old interposed-wrapper block.
    if _blocks_absorb(c):
        return 0

    # Splice nested single-consumer concat leaves inline first, so the
    # folds below see a flat leaf list.
    applied = 0
    leaves: List[Node] = []
    for leaf in c.inputs:
        if (
            isinstance(leaf, Concatenate)
            and not _blocks_absorb(leaf)
            and leaf not in touched
            and leaf not in output_nodes
            and len(consumers.get(leaf, [])) == 1
        ):
            leaves.extend(leaf.inputs)
            touched.add(leaf)
            applied += 1
        else:
            leaves.append(leaf)

    allow_value_folds = c._semantic_affine_override is None
    # Literal folds drop leaves; never drop them all.
    has_non_literal = any(not isinstance(x, LiteralValue) for x in leaves)

    d_out = l.d_output
    blocks: List[torch.Tensor] = []
    new_leaves: List[Node] = []
    bias = l.output_bias
    offset = 0
    value_folds = 0  # leaf folds that change c's value (splices don't)
    for leaf in leaves:
        w = len(leaf)
        m = l.output_matrix[offset : offset + w]
        offset += w

        if (
            allow_value_folds
            and isinstance(leaf, LiteralValue)
            and not _blocks_absorb(leaf)
            and has_non_literal
        ):
            bias = bias + leaf.value @ m
            applied += 1
            value_folds += 1
            continue

        if (
            allow_value_folds
            and isinstance(leaf, Linear)
            and not _blocks_absorb(leaf)
            and leaf not in touched
            and leaf not in output_nodes
            and len(consumers.get(leaf, [])) == 1
        ):
            d_x = leaf.output_matrix.shape[0]
            old = (d_x * w + w) + w * d_out
            new = d_x * d_out
            if new <= old:
                blocks.append(leaf.output_matrix @ m)
                bias = bias + leaf.output_bias @ m
                new_leaves.append(leaf.inputs[0])
                touched.add(leaf)
                applied += 1
                value_folds += 1
                continue

        blocks.append(m)
        new_leaves.append(leaf)

    # Coalesce duplicate leaves: two absorbed selectors over the same input
    # (or a hand-built duplicate) leave the same node in two slots.  A value
    # read through blocks B1 and B2 contributes x @ (B1 + B2), so the blocks
    # merge exactly and the concat narrows — which also keeps duplicate
    # source columns out of the weight-writer scatters.
    if allow_value_folds and len({id(x) for x in new_leaves}) != len(new_leaves):
        slot: Dict[int, int] = {}
        merged_leaves: List[Node] = []
        merged_blocks: List[torch.Tensor] = []
        for leaf, block in zip(new_leaves, blocks):
            j = slot.get(id(leaf))
            if j is None:
                slot[id(leaf)] = len(merged_leaves)
                merged_leaves.append(leaf)
                merged_blocks.append(block)
            else:
                merged_blocks[j] = merged_blocks[j] + block
                applied += 1
                value_folds += 1
        new_leaves, blocks = merged_leaves, merged_blocks

    if applied == 0:
        return 0

    l.output_matrix = torch.cat(blocks, dim=0)
    l.output_bias = bias
    l.d_input = l.output_matrix.shape[0]
    if len(new_leaves) == 1:
        l.inputs = [new_leaves[0]]  # bypass the concat entirely
    else:
        c.inputs = new_leaves
        c.d_output = sum(len(x) for x in new_leaves)
        mutated.add(c)
        if value_folds:
            fold_log.record_value_changed(c)
    touched.add(c)
    touched.add(l)
    mutated.add(l)
    return applied


def _fuse_one_pass(
    output_nodes: Set[Node], verbose: bool, mutated: Set[Node], fold_log: FoldLog
) -> int:
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
            if inp in touched:
                continue
            if isinstance(inp, Concatenate):
                if len(consumers.get(inp, [])) == 1:
                    applied += _fold_through_concatenate(
                        node, inp, output_nodes, consumers, touched, mutated, fold_log
                    )
                continue
            if len(consumers.get(inp, [])) != 1:
                continue

            if isinstance(inp, Linear):
                l1, l2 = inp, node
                if _blocks_absorb(l1):  # l1's checked value would be erased
                    continue
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
                if _blocks_absorb(b):  # b's checked pre-fold value would be erased
                    continue  # (l's checks migrate — see the fold)
                n_lanes = b.n_lanes
                d_b = b.d_output
                d_z = l.output_matrix.shape[1]
                old = (n_lanes * d_b + d_b) + (d_b * d_z + d_z)
                new = n_lanes * d_z + d_z
                if not param_delta_ok(new, old):
                    continue
                _fold_ffn_into_linear(b, consumers, fold_log)
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
            if _blocks_absorb(u):  # u's checked value would be erased
                continue
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
    fold_log: Optional[FoldLog] = None,
) -> int:
    """Fuse adjacent linear maps, FFN-aware, until no more folds apply.

    Four folds, each width-safe and gated so it never grows the parameter
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
    - **Linear leaves through Concatenate** (the concat's sole consumer is a
      Linear): each single-consumer Linear leaf is absorbed into its row
      block of the downstream Linear's matrix, LiteralValue leaves fold into
      its bias, and nested single-consumer concat leaves are spliced inline
      — the downstream Linear survives as the summing hardware, so no Add
      node (which would cost a compiled layer) is ever introduced.  Saves
      one compiled layer per absorbed leaf stage; collapses the
      fanout-limited accumulator chains ``sum_nodes`` builds.  See
      :func:`_fold_through_concatenate`.

    All folds mutate a surviving node in place; ``output_nodes`` stay valid
    references (an FFN-into-output-Linear fold is declined to preserve the
    caller's output identity).  A fold that would erase a *checked* value
    (``node.checks``) is declined — see the fold-policy note at the top
    of this module.

    Because each fold rewrites a surviving node's weights and inputs in place,
    that node's eagerly-cached ``_affine_bound`` / ``_structural_type`` (and
    every downstream node's, which was derived from it) go stale — the affine
    and structural ranges can then disagree and the RMSNorm certification's
    soundness check fires.  So after all folds settle, refresh the bounds of
    every mutated node and everything downstream of one (see
    :func:`_recompute_bounds_after_fusion`).  The refresh re-applies every
    node's range claim at the ``refresh_node_caches`` choke point, so
    claim-tightened bounds survive the recompute by construction.

    Args:
        output_nodes: The graph's output nodes (used to find all ancestors and
            to protect caller-held output identity).
        verbose: Print per-pass fold counts.
        fold_log: When given, records which nodes' values moved to a
            different holder and which surviving nodes no longer compute
            their pre-fusion value (see :class:`FoldLog`) — the lowering
            boundary uses this to keep its source→copy node map
            value-correct.

    Returns:
        Total number of folds performed.
    """
    total = 0
    mutated: Set[Node] = set()
    if fold_log is None:
        fold_log = FoldLog()
    while True:
        n = _fuse_one_pass(output_nodes, verbose, mutated, fold_log)
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
