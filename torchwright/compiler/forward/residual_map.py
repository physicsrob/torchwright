from typing import Dict, List, Set

from torchwright.compiler.residual_assignment import (
    ResidualAssignment,
    ResidualStreamState,
    flatten_concat_nodes,
)
from torchwright.graph import Node, Concatenate


class ResidualStreamMap:
    """Set-based column allocator for the residual stream.

    Assigns arbitrary free columns to nodes — no contiguity required.
    Weight matrices scatter/gather via index lists, so physical adjacency
    is never needed.
    """

    def __init__(self, d: int):
        self.d = d
        self._free: Set[int] = set(range(d))
        self._node_to_indices: Dict[Node, List[int]] = {}
        # Columns permanently withheld from the free pool: never allocated to
        # any node, never freed, for the whole compile.  The sole current user
        # is the pinned-constant RMSNorm (``_reserve_rms_norm_columns``), which
        # reserves 1–2 columns up front to hold the constant that forces the
        # RMS to a power of two — they are seeded once (into ``embed_table``) and
        # read but never written.  (The primitive was first built for an
        # end-of-compile delta-transfer layer, since removed.)
        self._reserved: Set[int] = set()

    def allocate(self, node: Node) -> List[int]:
        n = len(node)
        if n > len(self._free):
            raise ValueError(
                f"Cannot allocate {n} columns for {node}: "
                f"only {len(self._free)} free of {self.d} "
                f"({len(self._reserved)} reserved, e.g. by the rms_norm "
                f"pinned constant)"
            )
        indices = sorted(list(self._free)[:n])
        already_owned = {
            c: owner
            for owner, cols in self._node_to_indices.items()
            for c in cols
            if c in set(indices)
        }
        if already_owned:
            raise AssertionError(
                f"ResidualStreamMap.allocate({node!r}, {n} cols): "
                f"free set proposed columns already owned: "
                f"{ {c: repr(o) for c, o in list(already_owned.items())[:4]} }. "
                f"d={self.d}."
            )
        self._free -= set(indices)
        self._node_to_indices[node] = indices
        self._check_invariants(f"allocate({node!r}, {n} cols)")
        return indices

    def free(self, node: Node, mech: str = "attn"):
        # ``mech`` ("attn"/"mlp") records which sublayer the freeing cancel ran
        # in; the base map ignores it (the free is identical either way).  The
        # warm-start tracking subclass records it so the CP-SAT mechanism hint
        # reflects the heuristic's choice.
        if node not in self._node_to_indices:
            raise KeyError(f"Node {node} is not allocated")
        self._free |= set(self._node_to_indices[node])
        del self._node_to_indices[node]
        self._check_invariants(f"free({node!r})")

    def reassign(self, old_node: Node, new_node: Node):
        if old_node not in self._node_to_indices:
            raise KeyError(f"Node {old_node} is not allocated")
        indices = self._node_to_indices.pop(old_node)
        self._node_to_indices[new_node] = indices
        self._check_invariants(f"reassign({old_node!r} -> {new_node!r})")

    def get_indices(self, node: Node) -> List[int]:
        return self._node_to_indices[node]

    def resolve_indices(self, node: Node) -> List[int]:
        """Get column indices for a node, resolving through Concatenate.

        Unlike get_indices(), this handles Concatenate nodes by gathering
        the indices of their children in order. Needed by the weight writer
        when an Attn node's input is a Concatenate (e.g. get_prev_value).
        """
        if isinstance(node, Concatenate):
            indices = []
            for child in flatten_concat_nodes([node]):
                indices += self.get_indices(child)
            return indices
        return self.get_indices(node)

    def is_allocated(self, node: Node) -> bool:
        return node in self._node_to_indices

    def get_free_count(self) -> int:
        return len(self._free)

    def get_allocated_nodes(self) -> Set[Node]:
        return set(self._node_to_indices.keys())

    def reserve(self, cols) -> None:
        """Remove ``cols`` from the free pool without assigning them to any node.

        Protects columns that must hold a fixed value for the whole compile from
        being reused by intermediate allocations — currently the pinned-constant
        RMSNorm columns (see ``_reserve_rms_norm_columns``).  Reserved columns are
        disjoint from both ``_free`` and ``_node_to_indices`` and stay that way
        for the rest of the compile.  Callers that withhold columns from the
        scheduler must also tell the CP-SAT model (``reserve_residual``), which
        does not see this map.
        """
        cols = set(cols)
        unowned = cols - self._free
        if unowned:
            owners = {
                c: owner
                for owner, indices in self._node_to_indices.items()
                for c in indices
                if c in unowned
            }
            raise AssertionError(
                f"ResidualStreamMap.reserve({sorted(cols)[:8]}...): "
                f"columns {sorted(unowned)[:8]} are already allocated "
                f"(e.g. {{c: repr(o) for c, o in list(owners.items())[:3]}})."
            )
        self._free -= cols
        self._reserved |= cols
        self._check_invariants(f"reserve({sorted(cols)[:4]}...)")

    def _check_invariants(self, where: str) -> None:
        """Assert allocator state is self-consistent.

        Invariants (all must hold after any successful mutation):
          1. Pairwise disjointness: no two nodes share a column.
          2. free ∩ allocated == ∅.
          3. free ∪ allocated == {0 .. d-1}.

        Called at the end of every mutator (allocate/free/reassign) so a
        corrupted state is surfaced at the *source* rather than the next
        unrelated get_indices lookup.
        """
        seen: Dict[int, Node] = {}
        for node, cols in self._node_to_indices.items():
            for c in cols:
                if c in seen:
                    other = seen[c]
                    raise AssertionError(
                        f"ResidualStreamMap invariant violated after {where}: "
                        f"column {c} assigned to both {node!r} "
                        f"(cols={cols}) and {other!r} "
                        f"(cols={self._node_to_indices[other]}). d={self.d}."
                    )
                seen[c] = node
        overlap = self._free & seen.keys()
        if overlap:
            ov = sorted(overlap)
            raise AssertionError(
                f"ResidualStreamMap invariant violated after {where}: "
                f"columns {ov[:8]} are both free and allocated (e.g. "
                f"{ {c: repr(seen[c]) for c in ov[:3]} }). d={self.d}."
            )
        reserved_overlap_alloc = self._reserved & seen.keys()
        if reserved_overlap_alloc:
            ov = sorted(reserved_overlap_alloc)
            raise AssertionError(
                f"ResidualStreamMap invariant violated after {where}: "
                f"reserved columns {ov[:8]} are also allocated "
                f"(e.g. { {c: repr(seen[c]) for c in ov[:3]} }). d={self.d}."
            )
        reserved_overlap_free = self._reserved & self._free
        if reserved_overlap_free:
            ov = sorted(reserved_overlap_free)
            raise AssertionError(
                f"ResidualStreamMap invariant violated after {where}: "
                f"reserved columns {ov[:8]} are also in the free pool. "
                f"d={self.d}."
            )
        total = self._free | seen.keys() | self._reserved
        if total != set(range(self.d)):
            missing = sorted(set(range(self.d)) - total)
            raise AssertionError(
                f"ResidualStreamMap invariant violated after {where}: "
                f"columns {missing[:8]} are neither free, allocated, "
                f"nor reserved. d={self.d}, free={len(self._free)}, "
                f"allocated={len(seen)}, reserved={len(self._reserved)}."
            )

    def build_residual_assignment(
        self,
        in_state: ResidualStreamState,
        out_state: ResidualStreamState,
        input_nodes: List[Node],
        output_node: Node,
    ) -> ResidualAssignment:
        """Build a ResidualAssignment bridge for HeadlessTransformer.compute().

        Populates exactly two states:
        - in_state with all input_nodes at their allocated columns
          (read by get_input_res_stream)
        - out_state with output_node at its allocated columns
          (read by compute)
        """
        ra = ResidualAssignment({in_state, out_state})
        for node in input_nodes:
            ra.assign(in_state, node, self.get_indices(node))
        ra.assign(out_state, output_node, self.get_indices(output_node))
        return ra
