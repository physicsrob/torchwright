"""Golden-hash pins for the topology encoding of an assert-using graph.

``topology_entries`` is the persistence encoding shared by the CP-SAT
schedule-cache fingerprint and the ONNX debug-sidecar fingerprint.  The
asserts-as-node-metadata migration (docs/assert_metadata_plan.md)
removes Assert/DebugWatch wrapper nodes from the graph representation;
its byte-identity claim is that the encoding of an assert-using graph —
where the traversal steps through wrappers today and simply finds no
wrappers afterwards — does not change.  These pins enforce that claim:
they were computed BEFORE the migration and must pass unchanged after
it.

Two pins:

* the SOURCE graph's encoding — keys every committed ``.debug.json``
  sidecar (via ``debug_fingerprint``), so a change strands existing
  artifacts;
* the LOWERED copy's encoding — keys every ``TW_SCHEDULE_CACHE_DIR``
  entry (``graph_fingerprint`` runs on ``lowered.output_node``), so a
  change silently re-solves every cached schedule.  This pin also locks
  the linear-fusion fold/decline behavior around asserted values: a
  fold that fires (or stops firing) where it didn't before changes the
  lowered topology.

If either pin fails, the encoding or the lowering behavior changed for
assert-using graphs.  Confirm that is intentional, accept that existing
schedule caches re-solve and existing debug sidecars need re-export,
and update the pin.
"""

import hashlib
import json

import torch

from torchwright.compiler.graph_identity import topology_entries
from torchwright.compiler.lower import lower
from torchwright.graph import Concatenate, Linear
from torchwright.graph.asserts import (
    assert_in_range,
    assert_integer,
    assert_strictly_less,
    debug_watch,
)
from torchwright.graph.misc import Add
from torchwright.ops.inout_nodes import create_input

# Computed 2026-07-05 on the pre-migration (wrapper-node) representation.
SOURCE_GOLDEN = "3826c09194ec46f5d87c0aefeb27005db182bd378857ec7c655e8b3262655071"
LOWERED_GOLDEN = "2b954a7c805f5703ed36f6162d2c3063ad44825a97a38c452bb510bf4b142e80"


def _entries_hash(entries) -> str:
    encoded = json.dumps(entries, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_asserted_graph():
    """Deterministic graph exercising every assert-attachment shape.

    * nested claims on an input leaf (``assert_integer`` over
      ``assert_in_range`` — the leaf claim channel);
    * a claim between two Linears (``guarded`` — blocks the l1->l2
      linear fold);
    * an unasserted Linear pair (``m1 -> m2`` — the fold must fire);
    * a composite two-input helper (``assert_strictly_less`` — builds a
      Concatenate the predicate reads plus a projecting Linear);
    * a DebugWatch on an interior value.
    """
    a = create_input("a", 1, value_range=(0.0, 9.0))
    b = create_input("b", 1, value_range=(0.0, 20.0))

    a_checked = assert_integer(assert_in_range(a, 0.0, 9.0))
    ordered_b = assert_strictly_less(a_checked, b)

    l1 = Linear(a_checked, torch.tensor([[2.0]]), name="l1")
    guarded = assert_in_range(l1, 0.0, 18.0)
    l2 = Linear(guarded, torch.tensor([[0.5]]), name="l2")

    m1 = Linear(b, torch.tensor([[3.0]]), name="m1")
    m2 = Linear(m1, torch.tensor([[1.0]]), name="m2")

    watched = debug_watch(l2, lambda _x: (True, ""), "golden watch")
    return Concatenate([watched, ordered_b, m2, Add(l2, m2)])


def test_source_topology_encoding_is_pinned():
    out = _build_asserted_graph()
    assert _entries_hash(topology_entries(out)) == SOURCE_GOLDEN


def test_source_encoding_stable_across_rebuilds():
    h1 = _entries_hash(topology_entries(_build_asserted_graph()))
    h2 = _entries_hash(topology_entries(_build_asserted_graph()))
    assert h1 == h2


def test_lowered_topology_encoding_is_pinned():
    out = _build_asserted_graph()
    lowered = lower(out)
    assert _entries_hash(topology_entries(lowered.output_node)) == LOWERED_GOLDEN
