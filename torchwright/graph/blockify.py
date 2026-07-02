"""The ``blockify`` graph pass: turn mined ``Linear -> ReLU -> Linear`` chains
into first-class :class:`~torchwright.graph.block.Block` nodes.

This is the Phase-2a stepping stone (``docs/block_ir_step1_plan.md``): the same
structural inference the scheduler's ``_detect_chains_static`` does transiently
inside the compiler, relocated to run **once, at graph level, before compile**,
with an inspectable and assertable result.  It changes no math — each Block
computes the identical function with the identical weights the mined chain
would have compiled — and is **opt-in**: nothing calls it in the default
pipeline, so the chain-mined path and the block path coexist and are directly
comparable (that is the whole point of the stepping stone).

Gate-A rulings baked in here (``docs/block_lane_spec.md``):

- **L1 exclusivity is asserted, not handled.**  Non-exclusive-L1 dual
  realization fired 0 times on the flagship, so it is deleted: a mined chain
  whose first Linear has consumers beyond the chain's ReLU raises rather than
  being blockified.
- **No chain-internal Assert/DebugWatch.**  Both measured 0 on the flagship.
  An Assert or DebugWatch wrapping a chain's L1 output or ReLU output raises —
  a stop-and-report, not a case to silently skip.  (Asserts/watches wrapping
  the chain *input* or the chain *output* are fine — those are external.)
"""

from typing import Dict, List, Optional, Set, Tuple

from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import Node
from torchwright.graph.block import Block
from torchwright.graph.linear import Linear
from torchwright.graph.misc import Assert, Concatenate, DebugWatch
from torchwright.graph.relu import ReLU


def _unwrap(node: Node) -> Node:
    """Step through Assert/DebugWatch wrappers to the wrapped node."""
    while isinstance(node, (Assert, DebugWatch)):
        node = node.inputs[0]
    return node


class _ChainMiner:
    """Assert/Concatenate-transparent ``L1 -> ReLU -> L2`` chain miner.

    Mirrors the semantics of
    ``cpsat_scheduler._detect_chains_static`` — each ReLU and each Linear
    participates in at most one chain, iteration is in node-id order for
    determinism — but resolves effective consumers transparently through
    ``Concatenate`` **and** ``Assert``/``DebugWatch``.  The transparency
    matters because the compiler strips Assert/DebugWatch before it mines, so
    the chains it compiles are exactly the ones that survive that stripping;
    mining transparently here reproduces that set, and lets the pass see (and
    reject) a chain that carries an internal Assert instead of silently missing
    it.
    """

    def __init__(self, all_nodes: Set[Node]):
        self._direct_consumers: Dict[Node, List[Node]] = {n: [] for n in all_nodes}
        for node in all_nodes:
            for inp in node.inputs:
                if inp in self._direct_consumers:
                    self._direct_consumers[inp].append(node)
        self._eff_cache: Dict[Node, Set[Node]] = {}

    def effective_consumers(self, node: Node) -> Set[Node]:
        """Consumers resolving through Concatenate/Assert/DebugWatch.

        A terminal transparent wrapper (an output Concatenate/Assert with no
        further consumers) is kept as the effective consumer so the node it
        wraps is never treated as dead — the same terminal-Concatenate rule
        ``_detect_chains_static`` uses."""
        if node in self._eff_cache:
            return self._eff_cache[node]
        result: Set[Node] = set()
        for consumer in self._direct_consumers.get(node, []):
            if isinstance(consumer, (Concatenate, Assert, DebugWatch)):
                downstream = self.effective_consumers(consumer)
                if downstream:
                    result |= downstream
                else:
                    result.add(consumer)
            else:
                result.add(consumer)
        self._eff_cache[node] = result
        return result

    def direct_consumers(self, node: Node) -> List[Node]:
        return self._direct_consumers.get(node, [])

    def mine(self) -> List[Tuple[Linear, ReLU, Linear]]:
        chains: List[Tuple[Linear, ReLU, Linear]] = []
        seen_relus: Set[Node] = set()
        seen_linears: Set[Node] = set()

        linears = sorted(
            (n for n in self._direct_consumers if isinstance(n, Linear)),
            key=lambda n: n.node_id,
        )
        for l1 in linears:
            if l1 in seen_linears:
                continue
            l1_eff = self.effective_consumers(l1)
            relu_candidates = [c for c in l1_eff if isinstance(c, ReLU)]
            if len(relu_candidates) != 1:
                continue
            relu = relu_candidates[0]
            if relu in seen_relus:
                continue
            relu_eff = self.effective_consumers(relu)
            l2_candidates = [c for c in relu_eff if isinstance(c, Linear)]
            if len(relu_eff) != 1 or len(l2_candidates) != 1:
                continue
            l2 = l2_candidates[0]
            if _unwrap(l2.inputs[0]) is not relu:
                continue
            if l2 in seen_linears:
                continue
            chains.append((l1, relu, l2))
            seen_relus.add(relu)
            seen_linears.add(l1)
            seen_linears.add(l2)

        return chains


def _assert_no_internal_wrappers(l1: Linear, relu: ReLU, miner: _ChainMiner) -> None:
    """Raise if an Assert/DebugWatch wraps the chain's L1 or ReLU output.

    Chain-internal values (the L1 output and the ReLU output) are not graph
    nodes once blockified — they live in MLP hidden slots — so an Assert or
    DebugWatch on them has nowhere to re-attach.  Gate A measured 0 of these
    on the flagship, so this is a tripwire: firing means the graph has a shape
    the pass was told does not occur.  An assert/watch on the chain *input* or
    the chain *output* is external and allowed; only a wrapper whose direct
    producer is L1 or the ReLU is internal."""
    for internal, role in ((l1, "L1 output"), (relu, "ReLU output")):
        for consumer in miner.direct_consumers(internal):
            if isinstance(consumer, (Assert, DebugWatch)):
                raise AssertionError(
                    f"blockify: chain-internal {role} of {internal!r} is wrapped "
                    f"in {type(consumer).__name__} {consumer!r}. Chain-internal "
                    f"values live in MLP hidden slots after blockify and cannot "
                    f"carry an assert/watch (Gate A measured 0 of these). This is "
                    f"a stop-and-report: the graph has a shape blockify was told "
                    f"does not occur."
                )


def blockify(output_node: Node, *, verbose: bool = False) -> Node:
    """Replace every mined ``L1 -> ReLU -> L2`` chain with a :class:`Block`.

    Runs over the graph reachable from ``output_node`` (Asserts/DebugWatches
    still present — blockify sees the raw graph the way the caller built it,
    before ``GraphAnalyzer`` strips wrappers at compile).  Mutates consumers in
    place — each direct consumer of a chain's L2 is rewired to the Block, the
    way ``fuse_consecutive_linears`` mutates L2 in place — and returns the
    (possibly new) output node: when the output *is* a chain's L2, the returned
    node is that chain's Block.

    Assertions (Gate A):

    - Every mined chain's L1 is **exclusive** (its only effective consumer is
      the chain's ReLU).  A non-exclusive L1 raises.
    - No chain-internal Assert/DebugWatch (see
      :func:`_assert_no_internal_wrappers`).

    Args:
        output_node: The graph's output node.
        verbose: Print the number of chains blockified.

    Returns:
        The output node after blockify (the same object unless the output was
        itself a blockified chain's L2).
    """
    all_nodes = get_ancestor_nodes({output_node})
    miner = _ChainMiner(all_nodes)
    chains = miner.mine()

    new_output = output_node
    n_blockified = 0
    for l1, relu, l2 in chains:
        l1_eff = miner.effective_consumers(l1)
        if l1_eff != {relu}:
            raise AssertionError(
                f"blockify: chain L1 {l1!r} is not exclusive — its effective "
                f"consumers are {sorted(c.node_id for c in l1_eff)}, expected only "
                f"the chain ReLU {relu.node_id}. Non-exclusive-L1 dual realization "
                f"was deleted per Gate A; a shared upstream value must stay a "
                f"separate Linear feeding the Block. Stop-and-report."
            )
        _assert_no_internal_wrappers(l1, relu, miner)

        # Degenerate ReLU block: gate_proj rows are L1's neurons (L1 stores its
        # matrix as (d_input, n_lanes), the Block wants (n_lanes, d_input));
        # out_proj is L2's matrix (n_lanes, d_output) unchanged.
        block = Block(
            l1.inputs[0],
            gate_proj=l1.output_matrix.t().contiguous(),
            gate_bias=l1.output_bias,
            out_proj=l2.output_matrix,
            out_bias=l2.output_bias,
            activation="relu",
            name=(f"block_{l2.name}" if l2.name else ""),
        )
        # Inherit the chain output's annotation so a blockified Block still
        # reports which render stage it came from (debug/trace parity).  The
        # annotation is metadata only — not part of the canonical fingerprint
        # or any scheduling decision — so this does not perturb the compile.
        block.annotation = l2.annotation

        # Rewire every direct consumer of L2 onto the Block.  A consumer that
        # is a Concatenate or an external Assert/DebugWatch is rewired in place
        # too, so wrappers on the chain output stay attached to the Block.
        for consumer in list(miner.direct_consumers(l2)):
            consumer.replace_input(l2, block)
        if l2 is new_output:
            new_output = block

        n_blockified += 1

    if verbose:
        print(f"blockify: {n_blockified} chain(s) -> Block")

    return new_output
