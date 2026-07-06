"""Piecewise-linear function algebra for univariate collapse v2 (Phase A).

Any composition of piecewise-linear ops of one scalar is itself one
piecewise-linear function of that scalar.  This module builds that
composed function *symbolically enough to know where its kinks are*,
and *empirically enough to trust its values*:

- **Candidate kink positions come from weights.**  Affine members
  (Linear, Add with a literal) create no kinks.  An FFN member creates
  candidate kinks where a gate row's argument crosses zero — each
  crossing is found by pulling the row's zero level back through the
  composed upstream function (one crossing per segment that straddles
  zero; this is where kinks multiply through sawtooth-shaped upstream
  functions, and why summing per-lane counts is not a valid bound).
  Same-source Add unions the two kink sets; Concatenate concatenates
  values and unions kinks.

- **Values come from the sampled oracle, never from weights.**  The
  member's own ``compute`` (seeded at the source, the same
  ``_seeded_oracle`` the v1 collapse pass trusts) is evaluated at every
  candidate kink and at a ladder of offsets inside every segment.  The
  **midpoint-linearity certificate** — the oracle at each sampled
  interior offset lies on the chord between the segment's endpoint
  values, within budget — is what certifies "this member really is
  piecewise-linear".  Genuinely curved members (a same-source
  ``multiply`` is piecewise-quadratic; an unsaturated gated product is
  curved) decline with a measured deviation and its location.  The
  ladder includes near-knot offsets so curvature that hides next to a
  kink (the swish machine's corner fillets, ``abs``'s rounding near
  zero) is sampled, not assumed away — but the certificate remains a
  *sampled* check: it certifies deviation at the sampled offsets, no
  more (the same honesty clause as v1's plateau gate).

Every position (knot or sample) is snapped to the fp32 grid before
use: the oracle — like the compiled transformer — computes in fp32,
so a position fp32 cannot represent is not a real input either, and
snapping merges candidate kinks closer than fp32 resolution.

All arithmetic on positions and chords is float64 and the walk visits
members in topological order, so the analysis is a deterministic
function of the graph — safe next to the CP-SAT schedule-cache key.

Emission-shape models (S1/S2) for the analysis report live here too:

- **S1** — one interpolating ``piecewise_linear`` FFN (chain → 1).
  Lanes: one per slope change (the emitter's own rule).  Error model:
  a saturated lane's accumulated term reaches ``|Δm_i|·(hi − x_i)``,
  and the fp32 lane sum quantizes to ulps of the worst term.
- **S2** — two-sublayer bounded-step emission (chain → 2),
  generalizing ``floor_int``'s ``output_map`` fold: stage 1 emits one
  bounded step per value transition (2 lanes and **one residual
  column** per step — the residual columns are the liveness trap the
  reverted folded-floor variant hit), stage 2 folds the composed
  post-map into per-step δ-weights (1 lane per step, shared across
  output dims).  Error model: measured deviation + plateau drift +
  the swish fillet term ``2·swish_dip/scale·max|δ|`` + fp32
  accumulation at total-variation magnitude.

Phase A ships this as analysis only — nothing here mutates a graph or
is called by ``lower()``; see ``torchwright/compiler/collapse_analysis.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from torchwright.compiler.collapse import _SYNTH_CLAIM_ATOL, _seeded_oracle, _ulp32
from torchwright.graph import Node
from torchwright.graph.ffn import FFN
from torchwright.graph.misc import Concatenate, LiteralValue

_F64 = torch.float64

# Per-segment sample offsets (fractions of the segment width) for the
# linearity certificate.  0.5 is the core midpoint check; the geometric
# near-knot rungs catch curvature that concentrates next to a kink
# (fillets, |x|'s rounding) which a lone midpoint would never see.
_LADDER = (
    2.0**-10,
    2.0**-7,
    2.0**-4,
    0.125,
    0.5,
    0.875,
    1.0 - 2.0**-4,
    1.0 - 2.0**-7,
    1.0 - 2.0**-10,
)


def _snap_f32(xs: torch.Tensor) -> torch.Tensor:
    """Round positions to the fp32 grid, kept as float64 values."""
    return xs.to(torch.float32).to(_F64)


@dataclass(frozen=True)
class PLFunction:
    """Exact vector-valued piecewise-linear function of one scalar.

    ``x`` is ``(n,)`` float64 strictly ascending knots; ``y`` is
    ``(n, d)`` float64 values at the knots.  Between adjacent knots the
    function is the chord; outside ``[x[0], x[-1]]`` it extends with
    the per-dim tail slopes (0 = clamped — the convention of the
    domain-restricted walk, whose knots always include the domain
    endpoints).
    """

    x: torch.Tensor
    y: torch.Tensor
    slope_lo: torch.Tensor
    slope_hi: torch.Tensor

    def __post_init__(self):
        object.__setattr__(self, "x", torch.as_tensor(self.x, dtype=_F64).reshape(-1))
        y = torch.as_tensor(self.y, dtype=_F64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        object.__setattr__(self, "y", y)
        d = y.shape[1]
        for attr in ("slope_lo", "slope_hi"):
            t = torch.as_tensor(getattr(self, attr), dtype=_F64).reshape(-1)
            if t.numel() == 1 and d > 1:
                t = t.expand(d).clone()
            object.__setattr__(self, attr, t)
        n = self.x.shape[0]
        assert n >= 1, "need at least one knot"
        assert self.y.shape == (n, d)
        assert self.slope_lo.shape == (d,) and self.slope_hi.shape == (d,)
        if n > 1:
            assert bool(
                (self.x[1:] > self.x[:-1]).all()
            ), "knots must be strictly ascending"

    # --- shape ----------------------------------------------------------

    @property
    def d(self) -> int:
        return self.y.shape[1]

    @property
    def n_knots(self) -> int:
        return self.x.shape[0]

    # --- constructors ---------------------------------------------------

    @staticmethod
    def line(
        slope: torch.Tensor, intercept: torch.Tensor, lo: float, hi: float
    ) -> "PLFunction":
        """The affine function ``slope·x + intercept`` on ``[lo, hi]``,
        clamped outside (zero tail slopes)."""
        slope = torch.as_tensor(slope, dtype=_F64).reshape(-1)
        intercept = torch.as_tensor(intercept, dtype=_F64).reshape(-1)
        x = torch.tensor([lo, hi], dtype=_F64)
        y = torch.stack([slope * lo + intercept, slope * hi + intercept])
        zero = torch.zeros_like(slope)
        return PLFunction(x, y, zero, zero)

    @staticmethod
    def constant(values: torch.Tensor, lo: float, hi: float) -> "PLFunction":
        values = torch.as_tensor(values, dtype=_F64).reshape(-1)
        x = torch.tensor([lo, hi], dtype=_F64)
        y = torch.stack([values, values])
        zero = torch.zeros_like(values)
        return PLFunction(x, y, zero, zero)

    # --- evaluation -----------------------------------------------------

    def eval(self, xs: torch.Tensor) -> torch.Tensor:
        """Evaluate at ``xs`` — returns ``(len(xs), d)`` float64."""
        xs = torch.as_tensor(xs, dtype=_F64).reshape(-1)
        n = self.n_knots
        if n == 1:
            dx = xs - self.x[0]
            tail = torch.where(
                (dx < 0).unsqueeze(1),
                torch.outer(dx, self.slope_lo),
                torch.outer(dx, self.slope_hi),
            )
            return self.y[0].unsqueeze(0) + tail
        idx = torch.searchsorted(self.x, xs, right=True) - 1
        idx = idx.clamp(0, n - 2)
        x0, x1 = self.x[idx], self.x[idx + 1]
        y0, y1 = self.y[idx], self.y[idx + 1]
        t = ((xs - x0) / (x1 - x0)).unsqueeze(1)
        out = y0 + t * (y1 - y0)
        below = xs < self.x[0]
        above = xs > self.x[-1]
        if bool(below.any()):
            out[below] = self.y[0] + torch.outer(xs[below] - self.x[0], self.slope_lo)
        if bool(above.any()):
            out[above] = self.y[-1] + torch.outer(xs[above] - self.x[-1], self.slope_hi)
        return out

    # --- algebra --------------------------------------------------------

    def map_affine(
        self, matrix: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> "PLFunction":
        """Post-compose with an affine map: ``f'(x) = f(x) @ matrix + bias``.

        ``matrix`` is ``(d, d_out)``.  Creates no kinks (an affine image
        of a PL function keeps the same knots).
        """
        matrix = torch.as_tensor(matrix, dtype=_F64)
        y = self.y @ matrix
        if bias is not None:
            y = y + torch.as_tensor(bias, dtype=_F64).reshape(1, -1)
        return PLFunction(self.x, y, self.slope_lo @ matrix, self.slope_hi @ matrix)

    def add(self, other: "PLFunction") -> "PLFunction":
        """Pointwise sum — knot union (each PL is exact at the union)."""
        assert self.d == other.d
        xs = torch.unique(torch.cat([self.x, other.x]))
        return PLFunction(
            xs,
            self.eval(xs) + other.eval(xs),
            self.slope_lo + other.slope_lo,
            self.slope_hi + other.slope_hi,
        )

    @staticmethod
    def concat(fns: Sequence["PLFunction"]) -> "PLFunction":
        """Vector concatenation — knot union, values side by side."""
        assert fns
        xs = torch.unique(torch.cat([f.x for f in fns]))
        return PLFunction(
            xs,
            torch.cat([f.eval(xs) for f in fns], dim=1),
            torch.cat([f.slope_lo for f in fns]),
            torch.cat([f.slope_hi for f in fns]),
        )

    def refined(self, extra: torch.Tensor) -> "PLFunction":
        """The same function with additional knots inserted."""
        extra = torch.as_tensor(extra, dtype=_F64).reshape(-1)
        xs = torch.unique(torch.cat([self.x, extra]))
        return PLFunction(xs, self.eval(xs), self.slope_lo, self.slope_hi)

    def simplified(self, tol: float) -> "PLFunction":
        """Drop interior knots that are chord-consistent — a sequential
        sleeve scan: knot ``i`` is dropped when it sits within ``tol``
        of the chord from the last *kept* knot to knot ``i+1``.  Keeps
        real bends (to within one dropped neighbor) and mid-step
        anchors; sheds oracle-noise knots and the ±1-ulp candidate
        brackets on straight stretches.  Simultaneous removal would be
        wrong here: a bend hugged by its ulp brackets looks consistent
        with them even though it is the structure.  Callers measuring
        deviation against the simplified function stay honest — any
        error a removal introduces shows up in the measurement."""
        n = self.n_knots
        if n <= 2:
            return self
        keep = [0]
        for i in range(1, n - 1):
            a = keep[-1]
            xa, xb = self.x[a], self.x[i + 1]
            t = (self.x[i] - xa) / (xb - xa)
            pred = self.y[a] + t * (self.y[i + 1] - self.y[a])
            if float((self.y[i] - pred).abs().max()) > tol:
                keep.append(i)
        keep.append(n - 1)
        idx = torch.tensor(keep)
        return PLFunction(self.x[idx], self.y[idx], self.slope_lo, self.slope_hi)

    # --- kink structure ---------------------------------------------------

    def segment_slopes(self) -> torch.Tensor:
        """Per-segment slopes, ``(n-1, d)``."""
        if self.n_knots == 1:
            return torch.zeros(0, self.d, dtype=_F64)
        dx = (self.x[1:] - self.x[:-1]).unsqueeze(1)
        return (self.y[1:] - self.y[:-1]) / dx

    def slope_deltas(self) -> torch.Tensor:
        """Slope change at each knot (after minus before), ``(n, d)``.

        The tails count as the outermost slopes, so with clamped tails
        the first and last knots register the entry into / exit out of
        the outermost segments — exactly the lanes ``piecewise_linear``
        with ``clamp=True`` emits.
        """
        segs = self.segment_slopes()
        before = torch.cat([self.slope_lo.unsqueeze(0), segs])
        after = torch.cat([segs, self.slope_hi.unsqueeze(0)])
        return after - before

    def lanes(self, tol: Optional[float] = None) -> int:
        """Emitted-lane count of an S1 ``piecewise_linear`` of this
        function: one lane per knot whose slope changes in any dim."""
        deltas = self.slope_deltas().abs()
        if tol is None:
            segs = self.segment_slopes().abs()
            peak = float(segs.max()) if segs.numel() else 0.0
            tol = max(1e-9, 1e-6 * peak)
        return int((deltas.amax(dim=1) > tol).sum())

    def zero_crossings(self) -> torch.Tensor:
        """Interior positions where any output dim crosses zero.

        One crossing per (segment, dim) whose endpoint values strictly
        straddle zero.  A dim that touches zero exactly at a knot needs
        no entry — the knot is already a knot.  Tail crossings are not
        searched (the walk's functions are domain-clamped).
        """
        if self.n_knots < 2:
            return torch.zeros(0, dtype=_F64)
        y0, y1 = self.y[:-1], self.y[1:]
        strad = (y0 * y1) < 0
        if not bool(strad.any()):
            return torch.zeros(0, dtype=_F64)
        t = y0 / (y0 - y1)
        cross = self.x[:-1].unsqueeze(1) + t * (self.x[1:] - self.x[:-1]).unsqueeze(1)
        return cross[strad]


# ---------------------------------------------------------------------------
# The certificate walk
# ---------------------------------------------------------------------------


def transition_runs(
    fn: PLFunction, plateau_tol: float = _SYNTH_CLAIM_ATOL
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Segment ``fn`` into plateaus and value transitions.

    A segment is a **plateau** when no output dim's value changes
    across it by more than ``plateau_tol``; a maximal run of
    non-plateau segments is one **transition** (one bounded step of the
    S2 emission, ramp geometry inherited from the run's own extent).

    Returns ``(is_plateau, runs)``: a per-segment boolean tensor and
    the transition runs as ``(start_knot_index, end_knot_index)`` pairs.
    """
    dy = (fn.y[1:] - fn.y[:-1]).abs().amax(dim=1)
    n_seg = int(dy.shape[0])
    is_plateau = dy <= plateau_tol
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n_seg:
        j = i
        while j < n_seg and bool(is_plateau[j]) == bool(is_plateau[i]):
            j += 1
        if not bool(is_plateau[i]):
            runs.append((i, j))
        i = j
    return is_plateau, runs


# Ladder fractions strictly between 1/8 and 7/8 are "mid-segment"; the
# rest are the near-knot rungs.  Next to an interior knot — a candidate
# kink, i.e. somewhere the chain's own machinery has a hinge bend — a
# near-knot rung sits inside the chain's transition band, where the
# swish machine's fillet residue lives (a corner's rounding, or a
# canceling pair whose composed value is flat while an interior stage
# still transitions — a min of two compares measures ~swish_dip/scale
# there).  In-band behavior is the original chain's, per the v2
# certificate's inherited-ramp clause, so deviation there is reported
# separately (``fillet_deviation``), not charged against the chord
# certificate.
_MID_FRACTION = 0.125


# A chargeable sample is excused when its deviation sits under the
# fp32 resolution floor ``_RES_FLOOR_K · |chord slope| · eps32(|x|)``:
# a transition's position is representable only to one fp32 spacing of
# its own magnitude, so chord deviation up to ~slope·eps32(x) is the
# machine's own quantization — shared by any construction, the
# original chain included — not disagreement the collapse introduces.
_RES_FLOOR_K = 8.0

# Sleeve tolerance for the certificate's reported function
# (PLFunction.simplified): well under the budget, and charged into the
# emission-shape bounds by the analysis layer — the emitted shape
# reproduces the simplified function, which sits within SIMPLIFY_TOL
# of the measured one.
SIMPLIFY_TOL = _SYNTH_CLAIM_ATOL / 8.0

# Banded policy: a transition run is a *band* when it is narrow
# relative to the plateaus on both sides.  Samples strictly inside a
# band measure where the emission re-creates the original's own
# transition (position-uncertain at the upstream-noise scale) — the
# inherited-ramp clause.  The strict policy charges them; the banded
# policy bins them with the in-band deviations.  Both are measured and
# reported; adopting banded is the Phase-A checkpoint's policy call.
_BAND_PLATEAU_RATIO = 0.25


@dataclass(frozen=True)
class MemberCertificate:
    """One member's composed-PL certificate."""

    node: Node
    fn: PLFunction
    n_kinks: int  # interior candidate kinks (domain endpoints excluded)
    deviation: (
        float  # strict-chargeable: max |oracle - chord|, resolution floor excused
    )
    deviation_at: float  # x position of the worst strict deviation
    banded_deviation: float  # strict minus samples inside narrow bands
    banded_deviation_at: float
    fillet_deviation: float  # in-band / quantization-class deviation (reported)
    fillet_deviation_at: float
    n_samples: int

    def linear(self, budget: float = _SYNTH_CLAIM_ATOL) -> bool:
        return self.deviation <= budget

    def linear_banded(self, budget: float = _SYNTH_CLAIM_ATOL) -> bool:
        return self.banded_deviation <= budget


@dataclass
class SubgraphCertificate:
    """All member certificates of one univariate subgraph, or the
    reason the walk stopped."""

    source: Node
    domain: Tuple[float, float]
    members: Dict[Node, MemberCertificate]
    declined: Optional[str] = None
    n_oracle_points: int = 0


class _OracleCache:
    """Growing cache of oracle values at fp32-snapped positions.

    Each miss evaluates *every* member at the new positions in one
    seeded batch (``reference_eval`` computes the whole member cone
    anyway), so the walk pays one oracle call per member that
    introduces new positions, not per point.
    """

    def __init__(self, oracle: Callable[[torch.Tensor], Dict[Node, torch.Tensor]]):
        self._oracle = oracle
        self._index: Dict[float, int] = {}
        self._vals: Dict[Node, List[torch.Tensor]] = {}
        self.n_points = 0

    def ensure(self, xs: torch.Tensor) -> None:
        new = [float(v) for v in xs.tolist() if float(v) not in self._index]
        if not new:
            return
        out = self._oracle(torch.tensor(new, dtype=_F64))
        for node, val in out.items():
            self._vals.setdefault(node, []).append(val)
        for v in new:
            self._index[v] = self.n_points
            self.n_points += 1

    def get(self, node: Node, xs: torch.Tensor) -> torch.Tensor:
        rows = torch.tensor([self._index[float(v)] for v in xs.tolist()])
        return torch.cat(self._vals[node])[rows].to(_F64)


def certify_subgraph(
    source: Node,
    members: Sequence[Node],
    *,
    domain: Optional[Tuple[float, float]] = None,
    max_kinks: int = 100_000,
    oracle: Optional[Callable[[torch.Tensor], Dict[Node, torch.Tensor]]] = None,
) -> SubgraphCertificate:
    """Walk a univariate subgraph and certify each member's composed PL.

    Args:
        source: the subgraph's single 1-D source node.
        members: the subgraph members in topological order (the
            grouping ``scalar_sources`` produces).
        domain: source interval to certify over; defaults to the
            source's trusted ``value_range`` (must be finite).
        max_kinks: decline threshold on a member's candidate kink
            count (a runaway-pullback backstop).
        oracle: ``f(positions_float64) -> {member: (n, d) values}``;
            defaults to the seeded exact oracle (``_seeded_oracle``).

    Returns:
        A :class:`SubgraphCertificate`.  ``declined`` is set (and the
        walk stops) only on kink explosion or a non-finite domain;
        a *curved* member is not a decline here — it gets a certificate
        with its measured deviation, and the caller applies the budget.
    """
    if domain is None:
        vr = source.value_type.value_range
        if not vr.is_finite():
            return SubgraphCertificate(
                source,
                (float("-inf"), float("inf")),
                {},
                declined="source value_range is unbounded",
            )
        domain = (vr.lo, vr.hi)
    lo = float(_snap_f32(torch.tensor([domain[0]]))[0])
    hi = float(_snap_f32(torch.tensor([domain[1]]))[0])
    if not lo < hi:
        return SubgraphCertificate(
            source, (lo, hi), {}, declined="source domain is a single point"
        )

    members = list(members)
    if oracle is None:

        def oracle(xs: torch.Tensor) -> Dict[Node, torch.Tensor]:
            grid = xs.to(torch.float32).reshape(-1, 1)
            return _seeded_oracle(members, source, grid)

    cache = _OracleCache(oracle)
    plmap: Dict[Node, PLFunction] = {}
    kinks: Dict[Node, torch.Tensor] = {}
    certs: Dict[Node, MemberCertificate] = {}
    # Whether a hinge bend coincides with a domain endpoint — the
    # endpoint knot is then band-bearing for the fillet-zone split.
    edge_bends: Dict[Node, Tuple[bool, bool]] = {}

    consts: Dict[Node, PLFunction] = {}

    def input_pl(u: Node) -> PLFunction:
        if u is source:
            return PLFunction.line(
                torch.ones(1, dtype=_F64), torch.zeros(1, dtype=_F64), lo, hi
            )
        if isinstance(u, LiteralValue):
            return PLFunction.constant(u.value, lo, hi)
        pl = plmap.get(u)
        if pl is not None:
            return pl
        # Neither the source nor a member: the finder attributes it to
        # no subgraph, so its ancestry is literal-only — a constant
        # subexpression.  Evaluate it once.
        pl = consts.get(u)
        if pl is None:
            from torchwright.debug.probe import reference_eval
            from torchwright.graph.node import suppress_checks

            with suppress_checks():
                val = reference_eval(u, {}, 1)[u][0]
            pl = PLFunction.constant(val, lo, hi)
            consts[u] = pl
        return pl

    def input_kinks(u: Node) -> torch.Tensor:
        if u is source or u not in kinks:
            return torch.zeros(0, dtype=_F64)
        return kinks[u]

    ladder = torch.tensor(_LADDER, dtype=_F64)
    declined: Optional[str] = None

    for m in members:
        ks = [input_kinks(u) for u in m.inputs]
        bend_lo = any(edge_bends.get(u, (False, False))[0] for u in m.inputs)
        bend_hi = any(edge_bends.get(u, (False, False))[1] for u in m.inputs)
        if isinstance(m, FFN):
            phi = input_pl(m.inputs[0]).map_affine(
                m.gate_proj.t().to(_F64), m.gate_bias.to(_F64)
            )
            cross = _snap_f32(phi.zero_crossings())
            ks.append(cross[(cross > lo) & (cross < hi)])
            # A bend can coincide with a domain endpoint: a crossing
            # snapped onto it, or a gate argument exactly zero there
            # (zero_crossings records only strict straddles).
            bend_lo = (
                bend_lo or bool((cross == lo).any()) or bool((phi.y[0] == 0).any())
            )
            bend_hi = (
                bend_hi or bool((cross == hi).any()) or bool((phi.y[-1] == 0).any())
            )
        edge_bends[m] = (bend_lo, bend_hi)
        kset = torch.unique(torch.cat(ks)) if ks else torch.zeros(0, dtype=_F64)
        m_name = m.name or type(m).__name__
        if kset.numel() > max_kinks:
            declined = (
                f"kink explosion at member {m_name}: "
                f"{kset.numel()} candidates > cap {max_kinks}"
            )
            break
        kinks[m] = kset

        if isinstance(m, Concatenate):
            plmap[m] = PLFunction.concat([input_pl(u) for u in m.inputs])
            child = [certs[u] for u in m.inputs if u in certs]
            dev, dev_at = 0.0, 0.0
            bdev, bdev_at = 0.0, 0.0
            fdev, fdev_at = 0.0, 0.0
            for c in child:
                if c.deviation >= dev:
                    dev, dev_at = c.deviation, c.deviation_at
                if c.banded_deviation >= bdev:
                    bdev, bdev_at = c.banded_deviation, c.banded_deviation_at
                if c.fillet_deviation >= fdev:
                    fdev, fdev_at = c.fillet_deviation, c.fillet_deviation_at
            certs[m] = MemberCertificate(
                node=m,
                fn=plmap[m],
                n_kinks=int(kset.numel()),
                deviation=dev,
                deviation_at=dev_at,
                banded_deviation=bdev,
                banded_deviation_at=bdev_at,
                fillet_deviation=fdev,
                fillet_deviation_at=fdev_at,
                n_samples=0,
            )
            continue

        # Bracket every candidate with its fp32 neighbors: a transition
        # sharper than fp32 resolution collapses both ramp-edge
        # candidates onto one representable position whose oracle value
        # is mid-step — without the ±1-ulp knots that mid-step value
        # poisons the chords of both adjacent (wide) segments.
        k32 = kset.to(torch.float32)
        inf32 = torch.full_like(k32, float("inf"))
        expanded = torch.cat(
            [
                kset,
                torch.nextafter(k32, inf32).to(_F64),
                torch.nextafter(k32, -inf32).to(_F64),
            ]
        )
        expanded = expanded[(expanded > lo) & (expanded < hi)]
        knots = torch.unique(torch.cat([torch.tensor([lo, hi], dtype=_F64), expanded]))
        # Ladder samples inside every segment, snapped; drop any that
        # land on a knot (adjacent-fp32 knots leave no interior).
        x0 = knots[:-1].unsqueeze(1)
        x1 = knots[1:].unsqueeze(1)
        raw = _snap_f32(x0 + ladder.unsqueeze(0) * (x1 - x0)).reshape(-1)
        knot_set = {float(v) for v in knots.tolist()}
        samples = torch.unique(
            torch.tensor(
                [v for v in raw.tolist() if v not in knot_set] or [lo],
                dtype=_F64,
            )
        )
        cache.ensure(knots)
        cache.ensure(samples)
        # ``fn`` keeps every candidate (and bracket) as a knot — the
        # measurement/classification frame: a candidate marks a band
        # even where the composed value is flat (a canceling pair, a
        # min whose fillet is the only feature).  The certificate's
        # *reported* function sheds chord-consistent knots (below) so
        # the shape models see real bends and mid-step anchors only.
        fn = PLFunction(
            knots,
            cache.get(m, knots),
            torch.zeros(m.d_output, dtype=_F64),
            torch.zeros(m.d_output, dtype=_F64),
        )
        actual = cache.get(m, samples)
        err = (actual - fn.eval(samples)).abs().amax(dim=1)

        # Classify samples: a near-knot rung beside an interior knot
        # (a candidate kink — a hinge bend of the chain's own
        # machinery) sits in the chain's transition band — the fillet
        # zone (see _MID_FRACTION).  Rungs beside a domain endpoint are
        # chargeable unless a bend coincides with that endpoint.
        n = fn.n_knots
        is_plateau, _runs = transition_runs(fn)
        seg = (torch.searchsorted(fn.x, samples, right=True) - 1).clamp(0, n - 2)
        frac = (samples - fn.x[seg]) / (fn.x[seg + 1] - fn.x[seg])
        interior = torch.ones(n, dtype=torch.bool)
        interior[0], interior[-1] = edge_bends[m]
        near_left = frac <= _MID_FRACTION
        near_right = frac >= 1.0 - _MID_FRACTION
        fillet_zone = (near_left & interior[seg]) | (near_right & interior[seg + 1])
        # Quantization-limited transitions: a ramp narrower than a few
        # fp32 spacings at its own magnitude has no resolvable interior
        # — any construction (the original chain included) quantizes it
        # the same way, so its interior samples are reproduction-class,
        # not chord-certificate evidence.
        seg_w = fn.x[seg + 1] - fn.x[seg]
        eps32 = 2.0 ** (torch.floor(torch.log2(samples.abs().clamp(min=1.0))) - 23)
        fillet_zone |= (~is_plateau[seg]) & (seg_w <= 16.0 * eps32)
        # Resolution floor (see _RES_FLOOR_K): deviation under
        # slope·eps32(|x|) is the machine's own position quantization.
        # The neighboring segments' slopes count too: a knot whose
        # value is position-quantized (sampled mid-ramp because the
        # true corner is within one fp32 spacing) tilts the chord of
        # the segment NEXT to the ramp by up to ramp-slope·eps32.
        seg_slopes = fn.segment_slopes().abs().amax(dim=1)
        n_seg = int(seg_slopes.shape[0])
        prev_s = seg_slopes[(seg - 1).clamp(0, n_seg - 1)]
        next_s = seg_slopes[(seg + 1).clamp(0, n_seg - 1)]
        local_slope = torch.maximum(seg_slopes[seg], torch.maximum(prev_s, next_s))
        fillet_zone |= err <= _RES_FLOOR_K * local_slope * eps32

        # Banded policy (see _BAND_PLATEAU_RATIO): samples strictly
        # inside a transition run that is narrow relative to both
        # neighboring plateau runs.  Plateau runs narrower than a few
        # fp32 spacings do not separate (or anchor) runs — a
        # sub-resolution flat between two transitions is part of one
        # band.
        n_seg = int(is_plateau.shape[0])

        def _plateau_is_separator(a: int, b: int) -> bool:
            width = float(fn.x[b] - fn.x[a])
            mag = max(1.0, float(fn.x[a].abs()), float(fn.x[b].abs()))
            import math as _math

            return width > 64.0 * 2.0 ** (_math.floor(_math.log2(mag)) - 23)

        # Coalesced transition runs: (start_knot, end_knot) spans.
        coalesced: List[Tuple[int, int]] = []
        for i, j in _runs:
            if coalesced and not _plateau_is_separator(coalesced[-1][1], i):
                coalesced[-1] = (coalesced[-1][0], j)
            else:
                coalesced.append((i, j))
        seg_in_narrow_run = torch.zeros(n_seg, dtype=torch.bool)
        for i, j in coalesced:
            width = float(fn.x[j] - fn.x[i])
            left = 0.0
            k = i
            while k > 0 and bool(is_plateau[k - 1]):
                k -= 1
            if k < i:
                left = float(fn.x[i] - fn.x[k])
            right = 0.0
            k = j
            while k < n_seg and bool(is_plateau[k]):
                k += 1
            if k > j:
                right = float(fn.x[k] - fn.x[j])
            if (
                left > 0.0
                and right > 0.0
                and width <= _BAND_PLATEAU_RATIO * min(left, right)
            ):
                seg_in_narrow_run[i:j] = True
        band_zone = seg_in_narrow_run[seg]

        def _worst(mask: torch.Tensor) -> Tuple[float, float]:
            if not bool(mask.any()):
                return 0.0, 0.0
            e = torch.where(mask, err, torch.zeros_like(err))
            i = int(e.argmax())
            return float(e[i]), float(samples[i])

        dev, dev_at = _worst(~fillet_zone)
        bdev, bdev_at = _worst(~fillet_zone & ~band_zone)
        fdev, fdev_at = _worst(fillet_zone | band_zone)
        # The reported/propagated function: chord-consistent knots (the
        # ±1-ulp brackets on straight stretches, oracle-noise wiggles)
        # shed, real bends and mid-step anchors kept.  The sleeve
        # tolerance is charged by the analysis layer (SIMPLIFY_TOL).
        fn = fn.simplified(SIMPLIFY_TOL)
        plmap[m] = fn
        certs[m] = MemberCertificate(
            node=m,
            fn=fn,
            n_kinks=int(kset.numel()),
            deviation=dev,
            deviation_at=dev_at,
            banded_deviation=bdev,
            banded_deviation_at=bdev_at,
            fillet_deviation=fdev,
            fillet_deviation_at=fdev_at,
            n_samples=int(samples.numel()),
        )

    return SubgraphCertificate(
        source=source,
        domain=(lo, hi),
        members=certs,
        declined=declined,
        n_oracle_points=cache.n_points,
    )


# ---------------------------------------------------------------------------
# Emission-shape models (S1 / S2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class S1Model:
    """Single-FFN interpolating emission (chain → 1)."""

    lanes: int
    max_abs_dm: float  # largest slope change at any knot
    worst_term: float  # max_i |Δm_i|·(hi − x_i): the worst accumulated lane term
    fp_bound: float  # 4·ulp32(worst_term)
    total_bound: float  # fp_bound + measured deviation

    def admissible(self, lane_cap: int, budget: float = _SYNTH_CLAIM_ATOL) -> bool:
        return self.lanes <= lane_cap and self.total_bound <= budget


def model_s1(fn: PLFunction, measured_dev: float) -> S1Model:
    """The S1 lane count and fp32 error bound of emitting ``fn``.

    No swish fillet term: an interpolating ``piecewise_linear`` can
    raise ``input_scale`` until each corner's dip
    ``swish_dip·|Δm|/(scale·input_scale)`` sits below any target — the
    knob is value-path-neutral, so the fillet is an emission-time
    choice, not a bound.  The fp32 story is the lane sum, modeled here.
    """
    deltas = fn.slope_deltas()
    mags = deltas.abs().amax(dim=1)
    segs = fn.segment_slopes().abs()
    peak = float(segs.max()) if segs.numel() else 0.0
    tol = max(1e-9, 1e-6 * peak)
    live = mags > tol
    lanes = int(live.sum())
    if lanes:
        reach = (fn.x[-1] - fn.x).clamp(min=0.0)
        worst = float((mags * reach)[live].max())
        max_dm = float(mags[live].max())
    else:
        worst = 0.0
        max_dm = 0.0
    fp = 4.0 * _ulp32(worst)
    return S1Model(
        lanes=lanes,
        max_abs_dm=max_dm,
        worst_term=worst,
        fp_bound=fp,
        total_bound=fp + measured_dev,
    )


@dataclass(frozen=True)
class S2Model:
    """Two-sublayer bounded-step emission (chain → 2), the
    ``floor_int`` ``output_map`` shape generalized."""

    n_steps: int  # value-changing transitions
    stage1_lanes: int  # 2 per step
    stage1_cols: int  # 1 residual column per step, live between the stages
    stage2_lanes: int  # 1 per step (shared across output dims)
    max_abs_delta: float  # largest per-step value jump
    total_variation: float  # max over dims of Σ|δ|
    max_ramp_width: float  # widest transition run
    plateau_drift: float  # worst net value change across a plateau run
    # The shape's own worst in-band fillet excursion,
    # 2·swish_dip/scale·max|δ| on the swish machine.  REPORTED, NOT
    # CHARGED: the emission inherits the transition geometry, so this
    # is the same excursion class the original chain already carries
    # (the shipped ops' documented range-claim slack) — not new error
    # introduced by the collapse.  Phase B's synthesis certificate
    # must still verify the original-vs-emitted mismatch empirically
    # at fp32-representable in-band points.
    fillet_bound: float
    fp_bound: float
    total_bound: float  # measured deviation + plateau drift + fp

    def admissible(self, lane_cap: int, budget: float = _SYNTH_CLAIM_ATOL) -> bool:
        return (
            self.n_steps > 0
            and self.stage1_lanes <= lane_cap
            and self.stage2_lanes <= lane_cap
            and self.total_bound <= budget
        )


def model_s2(
    fn: PLFunction,
    measured_dev: float,
    *,
    machine: Optional[str],
    plateau_tol: float = _SYNTH_CLAIM_ATOL,
) -> S2Model:
    """Segment ``fn`` into plateaus and transitions and model the
    bounded-step emission.

    Segmentation is :func:`transition_runs`'s.  ``plateau_drift``
    reports the worst accumulated value change across a plateau run —
    a true staircase keeps it at fp noise, and it is charged into the
    bound.
    """
    is_plateau, runs = transition_runs(fn, plateau_tol)
    n_seg = int(is_plateau.shape[0])

    deltas = [fn.y[j] - fn.y[i] for i, j in runs]
    widths = [float(fn.x[j] - fn.x[i]) for i, j in runs]
    drift = 0.0
    i = 0
    while i < n_seg:
        j = i
        while j < n_seg and bool(is_plateau[j]) == bool(is_plateau[i]):
            j += 1
        if bool(is_plateau[i]):
            drift = max(drift, float((fn.y[j] - fn.y[i]).abs().max()))
        i = j

    n_steps = len(deltas)
    if n_steps:
        dmat = torch.stack(deltas)  # (n_steps, d)
        max_delta = float(dmat.abs().max())
        tv = float(dmat.abs().sum(dim=0).max())
        max_w = max(widths)
    else:
        max_delta = tv = max_w = 0.0
    if machine == "swish" and n_steps:
        from torchwright.ops.const import scale, swish_dip

        fillet = 2.0 * swish_dip / scale * max_delta
    else:
        fillet = 0.0
    fp = 4.0 * _ulp32(max(tv, float(fn.y.abs().max()))) if n_steps else 0.0
    return S2Model(
        n_steps=n_steps,
        stage1_lanes=2 * n_steps,
        stage1_cols=n_steps,
        stage2_lanes=n_steps,
        max_abs_delta=max_delta,
        total_variation=tv,
        max_ramp_width=max_w,
        plateau_drift=drift,
        fillet_bound=fillet,
        fp_bound=fp,
        total_bound=measured_dev + drift + fp,
    )
