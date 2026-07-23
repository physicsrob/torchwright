"""Static value-range types attached to every Node.

Each Node has a ``value_type: NodeValueType`` that summarises what we
statically know about its output tensor's element range. Propagation
rules defined per-op populate this eagerly at graph-build time, and
primitives can enforce contracts on their inputs via the helpers in
``torchwright.graph.asserts``.

``Range`` uses ±inf for unbounded sides so arithmetic is total; no
``Optional[Range]`` handling is needed at every rule site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

_INF = math.inf


@dataclass(frozen=True)
class Range:
    """Closed interval ``[lo, hi]`` with ±inf endpoints allowed.

    A ``Range`` always satisfies ``lo <= hi``. Use ``Range.unbounded()``
    for "no information"; use the arithmetic helpers (``__add__`` etc.)
    rather than reimplementing interval math at each op.
    """

    lo: float = -_INF
    hi: float = _INF

    def __post_init__(self) -> None:
        if not (self.lo <= self.hi):
            raise ValueError(f"Range lo must be <= hi, got lo={self.lo}, hi={self.hi}")

    @staticmethod
    def unbounded() -> Range:
        return Range(-_INF, _INF)

    @staticmethod
    def point(v: float) -> Range:
        return Range(v, v)

    def contains(self, other: Range) -> bool:
        return self.lo <= other.lo and other.hi <= self.hi

    def is_finite(self) -> bool:
        return math.isfinite(self.lo) and math.isfinite(self.hi)

    def __add__(self, other: Range) -> Range:
        return Range(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self) -> Range:
        return Range(-self.hi, -self.lo)

    def __sub__(self, other: Range) -> Range:
        return self + (-other)

    def union(self, other: Range) -> Range:
        return Range(min(self.lo, other.lo), max(self.hi, other.hi))

    def intersect(self, other: Range) -> Range:
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            raise ValueError(
                f"Empty range intersection: {self} \u2229 {other}. Both should be "
                f"sound over-approximations; disjointness indicates a bug."
            )
        return Range(lo, hi)

    def relu(self) -> Range:
        return Range(max(0.0, self.lo), max(0.0, self.hi))


@dataclass(frozen=True)
class NodeValueType:
    """Static properties of a node's output tensor.

    Tracks the value_range: a closed interval bounding every scalar
    element of the output.
    """

    value_range: Range = field(default_factory=Range.unbounded)

    # --- Factory helpers ------------------------------------------------

    @staticmethod
    def unknown() -> NodeValueType:
        return NodeValueType()

    @staticmethod
    def bounded(lo: float, hi: float) -> NodeValueType:
        return NodeValueType(value_range=Range(float(lo), float(hi)))

    # --- Combinators ----------------------------------------------------

    def with_range(self, r: Range) -> NodeValueType:
        return replace(self, value_range=r)


def tightened_with(a: NodeValueType, b: NodeValueType) -> NodeValueType:
    """Combine two claims into the strictest type both admit.

    Ranges are intersected.

    Used by ``refresh_node_caches`` to fold a node's ``claimed_type``
    into its structural type, and by the attach helpers to intersect a
    new claim into a node's running claim (claims commute).
    """
    r = a.value_range.intersect(b.value_range)
    return NodeValueType(value_range=r)
