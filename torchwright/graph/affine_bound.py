"""Affine bound propagation for the computation graph.

An ``AffineBound`` represents per-component lower and upper affine
expressions over a self-keyed basis of ``InputNode`` components::

    lower(x)[i] = A_lo[i, :] · x + b_lo[i]
    upper(x)[i] = A_hi[i, :] · x + b_hi[i]

where ``x`` is the concatenation of tracked ``InputNode`` values.
Each bound carries its own column map keyed by ``InputNode.node_id``.

All coefficient tensors are stored as ``torch.float64``, CPU-only.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Dict, Tuple

from torchwright.graph.value_type import Range


@dataclass
class AffineBound:
    """Per-component affine lower/upper bounds with self-keyed basis.

    Attributes:
        A_lo: Lower bound coefficient matrix, shape ``(d_output, n_cols)``.
        A_hi: Upper bound coefficient matrix, shape ``(d_output, n_cols)``.
        b_lo: Lower bound offset vector, shape ``(d_output,)``.
        b_hi: Upper bound offset vector, shape ``(d_output,)``.
        columns: Maps InputNode.node_id -> (start_col, width).
        input_ranges: Maps InputNode.node_id -> (lo, hi) declared range.
    """

    A_lo: torch.Tensor
    A_hi: torch.Tensor
    b_lo: torch.Tensor
    b_hi: torch.Tensor
    columns: Dict[int, Tuple[int, int]]
    input_ranges: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    def __post_init__(self):
        d = self.A_lo.shape[0]
        n = self.n_cols
        assert self.A_lo.shape == (d, n), f"A_lo shape {self.A_lo.shape} != ({d}, {n})"
        assert self.A_hi.shape == (d, n), f"A_hi shape {self.A_hi.shape} != ({d}, {n})"
        assert self.b_lo.shape == (d,), f"b_lo shape {self.b_lo.shape} != ({d},)"
        assert self.b_hi.shape == (d,), f"b_hi shape {self.b_hi.shape} != ({d},)"
        assert (
            self.A_lo.dtype == torch.float64
        ), f"A_lo must be float64, got {self.A_lo.dtype}"
        total_width = sum(w for _, w in self.columns.values())
        assert total_width == n, f"columns total width {total_width} != n_cols {n}"
        for nid in self.columns:
            assert (
                nid in self.input_ranges
            ), f"node_id {nid} in columns but missing from input_ranges"

    @property
    def n_cols(self) -> int:
        return self.A_lo.shape[1]

    @property
    def d_output(self) -> int:
        return self.A_lo.shape[0]

    @classmethod
    def identity(cls, input_node) -> "AffineBound":
        """One-hot rows for *input_node*'s columns, zero offsets."""
        from torchwright.graph.misc import InputNode

        assert isinstance(input_node, InputNode)
        d = input_node.d_output
        A = torch.eye(d, dtype=torch.float64)
        b = torch.zeros(d, dtype=torch.float64)
        lo_val, hi_val = input_node._declared_value_range
        lo = torch.full((d,), float(lo_val), dtype=torch.float64)
        hi = torch.full((d,), float(hi_val), dtype=torch.float64)
        return cls(
            A_lo=A.clone(),
            A_hi=A.clone(),
            b_lo=b.clone(),
            b_hi=b.clone(),
            columns={input_node.node_id: (0, d)},
            input_ranges={input_node.node_id: (lo, hi)},
        )

    @classmethod
    def constant(cls, values: torch.Tensor) -> "AffineBound":
        """Zero ``A``, ``b_lo = b_hi = values``."""
        d = values.shape[0]
        A = torch.zeros(d, 0, dtype=torch.float64)
        b = values.to(torch.float64)
        return cls(
            A_lo=A.clone(),
            A_hi=A.clone(),
            b_lo=b.clone(),
            b_hi=b.clone(),
            columns={},
            input_ranges={},
        )

    @classmethod
    def degenerate(
        cls,
        d_output: int,
        *,
        lo: float = float("-inf"),
        hi: float = float("inf"),
    ) -> "AffineBound":
        """Zero ``A``, broadcast scalar offsets."""
        A = torch.zeros(d_output, 0, dtype=torch.float64)
        b_lo = torch.full((d_output,), lo, dtype=torch.float64)
        b_hi = torch.full((d_output,), hi, dtype=torch.float64)
        return cls(
            A_lo=A.clone(),
            A_hi=A.clone(),
            b_lo=b_lo,
            b_hi=b_hi,
            columns={},
            input_ranges={},
        )

    @staticmethod
    def align(
        a: "AffineBound", b: "AffineBound"
    ) -> tuple["AffineBound", "AffineBound"]:
        """Reindex *a* and *b* to share the same column layout."""
        if a.columns == b.columns:
            merged_ranges = _merge_ranges(a.input_ranges, b.input_ranges)
            if _ranges_equal(merged_ranges, a.input_ranges) and _ranges_equal(
                merged_ranges, b.input_ranges
            ):
                return a, b
            a = AffineBound(a.A_lo, a.A_hi, a.b_lo, a.b_hi, a.columns, merged_ranges)
            b = AffineBound(b.A_lo, b.A_hi, b.b_lo, b.b_hi, b.columns, merged_ranges)
            return a, b

        merged_columns, merged_ranges, n = _merge_layouts(a, b)
        return _scatter(a, merged_columns, merged_ranges, n), _scatter(
            b, merged_columns, merged_ranges, n
        )

    def to_interval(self) -> list[Range]:
        """Concretize to per-component scalar intervals."""
        import math

        x_lo, x_hi = self._basis_box()
        ranges = []
        for i in range(self.d_output):
            lo = self._eval_lower(i, x_lo, x_hi)
            hi = self._eval_upper(i, x_lo, x_hi)
            assert not math.isnan(lo), (
                f"NaN in lower bound at component {i}; likely a 0*inf case "
                f"not caught by the eval guard"
            )
            assert not math.isnan(hi), (
                f"NaN in upper bound at component {i}; likely a 0*inf case "
                f"not caught by the eval guard"
            )
            if lo > hi:
                # A sound affine bound has lower <= upper for every component;
                # a crossing is fp accumulation noise on a near-degenerate
                # (point-valued) component, e.g. an embedding column at exactly
                # 0.5 whose lower eval lands 1 ULP above 0.5. Snap when the gap
                # is within fp tolerance; a gross crossing falls through to the
                # strict ``Range`` check below as a genuine-bug signal.
                if lo - hi <= 1e-6 * (1.0 + max(abs(lo), abs(hi))):
                    lo = hi
            ranges.append(Range(lo, hi))
        return ranges

    def to_scalar_range(self) -> Range:
        """Union of all per-component intervals into one scalar Range."""
        intervals = self.to_interval()
        if not intervals:
            return Range.unbounded()
        lo = min(r.lo for r in intervals)
        hi = max(r.hi for r in intervals)
        return Range(lo, hi)

    def _basis_box(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_lo, x_hi) vectors for the input basis box."""
        n = self.n_cols
        x_lo = torch.full((n,), float("-inf"), dtype=torch.float64)
        x_hi = torch.full((n,), float("inf"), dtype=torch.float64)

        for node_id, (start, width) in self.columns.items():
            assert (
                node_id in self.input_ranges
            ), f"node_id {node_id} in columns but not in input_ranges"
            lo_vec, hi_vec = self.input_ranges[node_id]
            x_lo[start : start + width] = lo_vec
            x_hi[start : start + width] = hi_vec
        return x_lo, x_hi

    def _eval_lower(self, i: int, x_lo: torch.Tensor, x_hi: torch.Tensor) -> float:
        """Evaluate lower bound for component i: min of A_lo[i] . x + b_lo[i]."""
        a = self.A_lo[i]
        pos = torch.clamp(a, min=0)
        neg = torch.clamp(a, max=0)
        term_pos = torch.where(pos == 0, torch.zeros_like(pos), pos * x_lo)
        term_neg = torch.where(neg == 0, torch.zeros_like(neg), neg * x_hi)
        val = (term_pos + term_neg).sum() + self.b_lo[i]
        return float(val.item())

    def _eval_upper(self, i: int, x_lo: torch.Tensor, x_hi: torch.Tensor) -> float:
        """Evaluate upper bound for component i: max of A_hi[i] . x + b_hi[i]."""
        a = self.A_hi[i]
        pos = torch.clamp(a, min=0)
        neg = torch.clamp(a, max=0)
        term_pos = torch.where(pos == 0, torch.zeros_like(pos), pos * x_hi)
        term_neg = torch.where(neg == 0, torch.zeros_like(neg), neg * x_lo)
        val = (term_pos + term_neg).sum() + self.b_hi[i]
        return float(val.item())

    def __repr__(self) -> str:
        intervals = self.to_interval()
        if len(intervals) <= 4:
            ivs = ", ".join(f"[{r.lo:.3g}, {r.hi:.3g}]" for r in intervals)
        else:
            ivs = ", ".join(f"[{r.lo:.3g}, {r.hi:.3g}]" for r in intervals[:3])
            ivs += f", ... ({len(intervals)} total)"
        return (
            f"AffineBound(d={self.d_output}, n_cols={self.n_cols}, intervals=[{ivs}])"
        )


def _ranges_equal(
    a: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    b: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> bool:
    if a.keys() != b.keys():
        return False
    return all(
        torch.equal(a[k][0], b[k][0]) and torch.equal(a[k][1], b[k][1]) for k in a
    )


def _merge_ranges(
    a: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    b: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Intersect input ranges from two bounds."""
    merged = dict(a)
    for k, (blo, bhi) in b.items():
        if k in merged:
            alo, ahi = merged[k]
            merged[k] = (torch.maximum(alo, blo), torch.minimum(ahi, bhi))
        else:
            merged[k] = (blo, bhi)
    return merged


def _merge_layouts(
    *bounds: AffineBound,
) -> tuple[
    Dict[int, Tuple[int, int]], Dict[int, Tuple[torch.Tensor, torch.Tensor]], int
]:
    """Build a merged column layout from multiple bounds."""
    merged_columns: Dict[int, Tuple[int, int]] = {}
    merged_ranges: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    offset = 0
    all_keys = sorted(set().union(*(b.columns for b in bounds)))
    for key in all_keys:
        width = None
        for b in bounds:
            if key in b.columns:
                _, width = b.columns[key]
                break
        assert width is not None
        merged_columns[key] = (offset, width)
        for b in bounds:
            if key in b.input_ranges:
                if key not in merged_ranges:
                    merged_ranges[key] = b.input_ranges[key]
                else:
                    old_lo, old_hi = merged_ranges[key]
                    new_lo, new_hi = b.input_ranges[key]
                    merged_ranges[key] = (
                        torch.maximum(old_lo, new_lo),
                        torch.minimum(old_hi, new_hi),
                    )
        offset += width
    return merged_columns, merged_ranges, offset


def _scatter(
    ab: AffineBound,
    merged_columns: Dict[int, Tuple[int, int]],
    merged_ranges: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    n: int,
) -> AffineBound:
    """Scatter *ab*'s columns into a merged layout."""
    d = ab.d_output
    A_lo = torch.zeros(d, n, dtype=torch.float64)
    A_hi = torch.zeros(d, n, dtype=torch.float64)
    for key, (old_start, width) in ab.columns.items():
        new_start, _ = merged_columns[key]
        A_lo[:, new_start : new_start + width] = ab.A_lo[
            :, old_start : old_start + width
        ]
        A_hi[:, new_start : new_start + width] = ab.A_hi[
            :, old_start : old_start + width
        ]
    return AffineBound(
        A_lo=A_lo,
        A_hi=A_hi,
        b_lo=ab.b_lo.clone(),
        b_hi=ab.b_hi.clone(),
        columns=merged_columns,
        input_ranges=merged_ranges,
    )
