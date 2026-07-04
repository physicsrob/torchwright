"""RoPE rotation primitives — the single source of truth for the rotary path.

Both the graph oracle (:meth:`torchwright.graph.attn.Attn.compute`) and the
compiled component (:class:`torchwright.compiler.components.attn.AttnLayerComponent`)
import these helpers, so the rotation applied during ``reference_eval`` is the
same one the compiled transformer applies bit-for-bit.  Keeping a second copy
anywhere is the exact drift R15 warns about (the oracle silently disagreeing
with the compiled head).

Convention — **HF LLaMA3** (see ``docs/rope_port_plan.md`` §6 "RoPE convention"):

- ``rotate_half`` (half-split) layout: a rotary width ``w`` splits into halves
  ``[0:w/2]`` and ``[w/2:w]``; dim ``p`` and dim ``p+w/2`` form one rotary
  plane, both rotated by the same angle ``pos · θ_p``.  This is *not* the
  legacy interleaved ``(2i, 2i+1)`` pairing of ``trig_shift_matrix``.
- Both Q and K are rotated by *absolute* position; V is never rotated.
- ``θ_p = base^(-2p/w)`` for ``p = 0 .. w/2-1``.

The grid is defined over the rotary width ``w`` (the ``Attn`` node's ``d_qk``).
In the Phase-5 end state every head uses ``w = d_head`` on one global ``base``,
recovering the single global grid; this same formula covers that case.
"""

import math
from dataclasses import dataclass

import torch

# LLaMA3-family base.  See docs/rope_port_plan.md §6 — reconcile downstream
# analyses (measured at 1e6) if this changes.
ROPE_BASE = 500000.0


@dataclass(frozen=True)
class RopeConfig:
    """The RoPE substrate a graph is built against — replaces ``PosEncoding``.

    Under the Phase-5 end state position is *not* a residual feature; it is a
    rotation applied inside attention (``docs/rope_port_plan.md`` §1, §3).  The
    ``attend_*`` builders that used to take a ``PosEncoding`` node now take this
    config because they must lay their content out on the **slow planes** of the
    full ``d_head`` ``rotate_half`` grid at construction time (the §7 finding:
    a content head is a full-width rotary ``Attn``, so it needs ``d_head`` and
    ``base`` before compile — see :func:`place_on_slow_planes`).

    Fields:
        d_head: the head width (= the compiled ``d_head``).  The compile entry
            points assert the ``d_head`` passed to them matches this.  Must be even.
        max_positions: the rollout length (the cache cap, not the typical
            frame).  Used by :func:`rope_lobe_band` to set the local-recency
            band's near-DC floor (``θ > π/max_positions``).
        base: the rotary base (LLaMA3-family ``5e5`` by default).
        d_rot: the partial-rotary width (vanilla HF ``partial_rotary_factor``).
            The first ``d_rot`` dims of every head rotate; the last
            ``d_head - d_rot`` are the unrotated NoPE tail.  ``None`` (default)
            resolves to ``d_head`` (full rotary, identical to the LLaMA3 end
            state).  Must be even and in ``(0, d_head]``.  Global: the ONNX
            exporter asserts a single ``d_rot`` across all heads.
    """

    d_head: int
    max_positions: int
    base: float = ROPE_BASE
    d_rot: int | None = None

    def __post_init__(self) -> None:
        if self.d_head % 2 != 0:
            raise ValueError(
                f"RopeConfig.d_head must be even (rotate_half pairs dim p with "
                f"p+d_head/2); got {self.d_head}."
            )
        # Resolve d_rot (None → full rotary) to a concrete even width in
        # (0, d_head]; frozen dataclass, so commit it via object.__setattr__.
        d_rot = self.d_head if self.d_rot is None else self.d_rot
        if d_rot % 2 != 0 or not (0 < d_rot <= self.d_head):
            raise ValueError(
                f"RopeConfig.d_rot must be even and in (0, d_head={self.d_head}]; "
                f"got {self.d_rot}."
            )
        object.__setattr__(self, "d_rot", d_rot)


def require_full_rotary(d_rot: int, d_head: int, feature: str) -> None:
    """Raise unless the head is full rotary (``d_rot == d_head``).

    The **local-recency lobe** (:func:`rotary_recency_head` /
    :func:`~torchwright.ops.attention_ops.attend_most_recent_matching`) lays a
    constant feature across a *band* of planes of the full ``d_head``
    ``rotate_half`` grid (:func:`rope_lobe_band`).  Under partial rotary a band
    plane's ``rotate_half`` partner (paired by ``d_rot/2`` once the rotation runs
    over the front, not by ``d_head/2``) can fall in the unrotated NoPE tail, which
    silently breaks the pairing — so this builder rejects a partial ``d_rot``
    loudly rather than miscompute.  Callers pass ``rope.d_rot`` / ``rope.d_head``.

    The content heads (:func:`rotary_content_head`) and the offset head
    (:func:`rotary_offset_head`) now *support* partial rotary — content rides the
    NoPE tail (:func:`place_on_nope_tail`); the local lobe is left full-rotary only
    because DOOM uses the global recency mechanism past the lobe window."""
    if d_rot != d_head:
        raise NotImplementedError(
            f"{feature} assumes full rotary (d_rot == d_head) but got "
            f"d_rot={d_rot}, d_head={d_head}.  Partial rotary is supported only "
            f"by the rotary offset head; the plane-based content/recency heads "
            f"are full-rotary only."
        )


def rope_inv_freq(width: int, base: float) -> torch.Tensor:
    """Per-plane angular frequencies ``θ_p = base^(-2p/width)``.

    Returns a ``(width // 2,)`` float32 tensor.  ``width`` must be even.
    """
    if width % 2 != 0:
        raise ValueError(f"RoPE width must be even, got {width}")
    p = torch.arange(0, width, 2, dtype=torch.float64)
    return (base ** (-p / width)).to(torch.float32)


def rope_cos_sin(
    positions: torch.Tensor, width: int, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(cos, sin)`` rotation tables for ``rotate_half`` RoPE.

    ``positions`` is a ``(P,)`` tensor of absolute positions.  Returns ``cos``
    and ``sin`` each ``(P, width)`` in the LLaMA3 half-split layout: the first
    and second halves carry the same per-plane angles, so dim ``p`` and dim
    ``p+width/2`` share one angle ``pos · θ_p``.
    """
    inv = rope_inv_freq(width, base).to(positions.device)  # (width/2,)
    ang = positions.to(torch.float32)[:, None] * inv[None, :]  # (P, width/2)
    cos_half = torch.cos(ang)
    sin_half = torch.sin(ang)
    cos = torch.cat([cos_half, cos_half], dim=-1)  # (P, width)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``rotate_half([a, b]) = [-b, a]`` over the last-dim halves (LLaMA3)."""
    w = x.shape[-1]
    x1 = x[..., : w // 2]
    x2 = x[..., w // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply ``rotate_half`` RoPE, vanilla partial-rotary (HF ``partial_rotary_factor``).

    ``x`` is ``(..., P, d_head)``; ``cos``/``sin`` are ``(P, d_rot)`` and broadcast
    over the leading dims.  The rotary width ``d_rot`` is read from ``cos.shape[-1]``:
    the first ``d_rot`` dims of ``x`` are rotated (``rotate_half`` over ``d_rot``,
    pairing dim ``i`` with ``i + d_rot/2``) and the last ``d_head - d_rot`` dims pass
    through unrotated (the NoPE tail).  When ``d_rot == d_head`` (``cos`` as wide as
    ``x``) this is full rotary and takes the exact pre-partial expression, so existing
    full-width callers are byte-identical."""
    d_rot = cos.shape[-1]
    assert d_rot <= x.shape[-1], (
        f"RoPE cos width {d_rot} exceeds x width {x.shape[-1]}; cos/sin must be the "
        f"rotary width d_rot ≤ d_head."
    )
    if d_rot == x.shape[-1]:
        return x * cos + rotate_half(x) * sin
    x_front = x[..., :d_rot]
    x_pass = x[..., d_rot:]
    rotated = x_front * cos + rotate_half(x_front) * sin
    return torch.cat([rotated, x_pass], dim=-1)


def place_on_slow_planes(mat: torch.Tensor, d_head: int) -> torch.Tensor:
    """Relocate a content head's ``(rows, W)`` Q/K projection onto the **slowest**
    ``W`` planes of the full ``d_head`` ``rotate_half`` grid.

    A content-selection head (``attend_argmin``/``argmax``/``_where``/``dot``/
    ``_bucket`` …) matches on content, not position: its logit is
    ``Σ_c q_c · k_c`` with the content in a few small columns ``c = 0 .. W-1``.
    Under the end-state global rotation every head's Q/K is rotated by absolute
    position, turning that logit into ``Σ_c q_c · k_c · cos((i − j)·θ_{p_c})`` —
    so a content column on a *fast* plane (large ``θ``) is scrambled by the
    relative offset.  Placing each content column on a **slow** plane (tiny
    ``θ``) keeps ``cos((i − j)·θ) ≈ 1`` over the rollout, so the match is
    effectively position-free (``docs/rope_port_plan.md`` §3 — standard RoPE on
    the global grid, not NoPE).

    Layout: content column ``c`` goes to the first-half dim ``d_head/2 − 1 − c``
    (the slowest plane first), with its ``rotate_half`` partner ``d_head − 1 − c``
    left zero, so each plane carries exactly one content scalar and the planes do
    not mix under rotation.  Requires ``W ≤ d_head/2``.

    Returns a ``(rows, d_head)`` matrix; pass it as the Q/K projection of an
    ``Attn``.  This is the **full-rotary** content placement; under partial rotary
    :func:`rotary_content_head` routes content to the NoPE tail
    (:func:`place_on_nope_tail`) instead, because partial rotary would move the
    slow planes into the unrotated tail and break the slow-plane assumption.
    """
    rows, w = mat.shape
    half = d_head // 2
    if w > half:
        raise ValueError(
            f"content width {w} exceeds the {half} planes available at "
            f"d_head={d_head}; raise d_head or narrow the content."
        )
    full = mat.new_zeros((rows, d_head))
    for c in range(w):
        full[:, half - 1 - c] = mat[:, c]
    return full


def place_on_nope_tail(mat: torch.Tensor, d_head: int, d_rot: int) -> torch.Tensor:
    """Relocate a content head's ``(rows, W)`` Q/K projection onto the **unrotated
    NoPE tail** ``[d_rot:d_head]`` of the partial-rotary grid.

    The partial-rotary dual of :func:`place_on_slow_planes`.  Under vanilla partial
    rotary (``d_rot < d_head``) :func:`apply_rope` rotates only the first ``d_rot``
    dims and passes the last ``d_head - d_rot`` (the tail) through *exactly*
    unmodified at every position (proven by
    ``test_apply_rope_tail_is_position_invariant``).  A content scalar placed on a
    tail dim is therefore read by the Q·K dot product with **no** position
    dependence at all — exactly position-free at any distance, with none of the
    slow-plane cosine attenuation :func:`place_on_slow_planes` leaves behind.  This
    is what dissolves the full-rotary ``d_head`` budget: content rides the tail,
    position/recency rides the disjoint rotary front.

    Layout: content column ``c`` goes to tail dim ``d_rot + c`` (the ``W`` content
    columns fill the front of the tail; the rotary front ``[0:d_rot]`` stays all
    zero, so this head's logit is the bare content dot product, no rotation).
    Requires ``W <= d_head - d_rot``.

    Returns a ``(rows, d_head)`` matrix; pass it as the Q/K projection of an
    ``Attn`` built with ``rope_d_rot=d_rot`` so the runtime leaves the tail dims
    unrotated.
    """
    rows, w = mat.shape
    tail = d_head - d_rot
    if w > tail:
        raise ValueError(
            f"content width {w} exceeds the {tail}-wide NoPE tail at "
            f"d_head={d_head}, d_rot={d_rot}; raise d_head, lower d_rot, or "
            f"narrow the content."
        )
    full = mat.new_zeros((rows, d_head))
    for c in range(w):
        full[:, d_rot + c] = mat[:, c]
    return full


def rotary_content_head(
    query_in,
    key_in,
    value,
    query_matrix: torch.Tensor,
    key_matrix: torch.Tensor,
    *,
    d_head: int,
    base: float = ROPE_BASE,
    d_rot: int | None = None,
):
    """A content-selection ``Attn`` made rotary, with the content placed so the
    match survives the global rotation — **routed by ``d_rot``**.

    Takes a content head's compact ``(·, W)`` ``query_matrix`` / ``key_matrix``
    (the same layout the ``attend_*`` builders construct — score in col 0,
    validity / bucket / dot dims in later cols) and rebuilds it as a full-width
    ``d_head`` head.  Two placements, picked by the config's ``d_rot``:

    - **Full rotary** (``d_rot is None`` or ``== d_head``): content relocated onto
      the slowest ``W`` planes of the ``d_head`` ``rotate_half`` grid via
      :func:`place_on_slow_planes`.  The slow planes are quasi-static under the
      global rotation (``cos((i−j)·θ_slow) ≈ 1``), so the match is *approximately*
      position-free (``docs/rope_port_plan.md`` §3).

    - **Partial rotary** (``d_rot < d_head``): content relocated onto the unrotated
      NoPE tail ``[d_rot:d_head]`` via :func:`place_on_nope_tail`, and the head is
      built with ``rope_d_rot=d_rot`` so the runtime leaves the tail dims
      unrotated.  The match is then *exactly* position-free at any distance (no
      slow-plane attenuation, no slow-plane budget).  This is the deliberate
      relaxation of §1's "NoPE is the forbidden middle ground" stance for the
      ``d_rot`` knob — content on disjoint dims from the rotated position/recency
      planes.

    V/O pass the payload through unchanged in both cases.  The selection-only
    ``attend_*`` builders need no per-builder change: passing ``rope.d_rot`` here
    routes them automatically.

    Note this routes only the *content* placement.  Heads that also need a
    position/recency signal on a *rotated* plane (the BOS-weight head and
    :func:`~torchwright.ops.relu.global_recency.attend_most_recent_globally`) cannot
    delegate here under partial rotary — they place the rotated column explicitly.
    The local-recency lobe (:func:`rotary_recency_head`) stays full-rotary only.
    """
    from torchwright.graph import Attn

    d_rot = d_head if d_rot is None else d_rot
    d_v = len(value)
    if d_rot < d_head:
        q_full = place_on_nope_tail(query_matrix, d_head, d_rot)
        k_full = place_on_nope_tail(key_matrix, d_head, d_rot)
        rope_d_rot = d_rot
    else:
        q_full = place_on_slow_planes(query_matrix, d_head)
        k_full = place_on_slow_planes(key_matrix, d_head)
        rope_d_rot = None  # full rotary — byte-identical to the pre-partial head
    return Attn(
        query_in=query_in,
        key_in=key_in,
        value_in=value,
        query_matrix=q_full,
        key_matrix=k_full,
        value_matrix=torch.eye(d_v),
        output_matrix=torch.eye(d_v),
        rope_base=base,
        rope_d_rot=rope_d_rot,
    )


# Upper θ cutoff for the local-recency lobe band.  The default 0.3 is the
# "natural" mid-band that gives W ≈ 415 at base 5e5 / d_head 256 (measured by
# scripts/rope_window_frontier.py).  Lowering it drops more fast planes and
# widens W at the cost of coarser near-Δ resolution (docs/rope_port_plan.md
# Phase 6); raising it shrinks W.
LOBE_THETA_MAX = 0.3


def rope_lobe_band(
    d_head: int,
    base: float,
    max_positions: int,
    *,
    theta_max: float = LOBE_THETA_MAX,
) -> tuple[list[int], torch.Tensor]:
    """Planes + Hann amplitudes for the **local-recency distance-decay lobe**.

    The recency dual of :func:`place_on_slow_planes`.  A constant feature placed
    identically on Q and K across a *band* of planes, rotated by absolute
    position, gives the self-similarity logit ``Σ_p amp_p · cos(Δ·θ_p)`` for two
    tokens ``Δ`` positions apart (the per-plane identity :func:`apply_rope`
    realises: one scalar on plane ``p`` with its ``rotate_half`` partner zeroed
    becomes ``(cos pos·θ_p, sin pos·θ_p)``, and two such dot to ``cos(Δ·θ_p)``).
    At ``Δ=0`` every cosine is 1 (the peak); as ``Δ`` grows the planes dephase
    and the sum decays — the main lobe.  Within the lobe width ``W`` a nearer key
    always outscores a farther one, so "most recent matching" falls out of the
    global rotation with **no value path and no precomputed rank**
    (``docs/rope_port_plan.md`` Phase 6 — the local recency that supersedes the
    octant ramp).

    Band: planes with ``π/max_positions < θ_p < theta_max``.  Drop the *fastest*
    planes (``θ ≥ theta_max``): they alias within a few positions and are what
    sets where decay begins, i.e. the window ``W``.  Drop the *near-DC* planes
    (``θ ≤ π/max_positions``): they barely decay over the rollout and are left
    for content (:func:`place_on_slow_planes` puts content on the very slowest
    planes, so excluding the near-DC band keeps the lobe disjoint from content).
    A **Hann taper** over the kept band suppresses the Dirichlet-kernel sidelobes
    that would let a farther key tie/beat a nearer one, extending the clean
    monotone window (the ``+1e-3`` floor keeps the endpoint planes nonzero).
    Lowering ``theta_max`` widens ``W`` at the cost of coarser near-Δ resolution.

    At ``base 5e5``, ``d_head 256``, ``max_positions 61440``, ``theta_max 0.3``
    the band is planes 12..96 (85 planes), peak ``Σ amp_p ≈ 42.6``, with a
    measured monotone window ``W ≈ 415`` — ~4× the ~100-token target.  Returns
    ``(planes, amps)`` where ``amps`` is a ``(len(planes),)`` float32 tensor.

    Raises ``ValueError`` if fewer than two planes survive the band (raise
    ``d_head`` / ``base`` or widen ``theta_max``).
    """
    inv = rope_inv_freq(d_head, base)  # (d_head/2,), θ_p decreasing in p
    floor = math.pi / max_positions
    planes = [p for p in range(d_head // 2) if floor < float(inv[p]) < theta_max]
    if len(planes) < 2:
        raise ValueError(
            f"local-recency lobe band has <2 planes at base={base} "
            f"d_head={d_head} max_positions={max_positions} theta_max={theta_max}; "
            f"raise d_head/base or widen theta_max."
        )
    n = len(planes)
    k = torch.arange(n, dtype=torch.float32)
    amps = 0.5 - 0.5 * torch.cos(2.0 * math.pi * k / (n - 1)) + 1e-3
    return planes, amps


def rotary_recency_head(
    query_in,
    key_in,
    value,
    query_matrix: torch.Tensor,
    key_matrix: torch.Tensor,
    *,
    d_head: int,
    max_positions: int,
    base: float = ROPE_BASE,
    recency_gain: float,
    theta_max: float = LOBE_THETA_MAX,
    d_rot: int | None = None,
):
    """A "most recent matching" content head with **intrinsic rotary recency**.

    Like :func:`rotary_content_head` — the compact ``(·, W)`` content
    ``query_matrix`` / ``key_matrix`` is relocated onto the slowest ``W`` planes
    (position-free match) — but it *also* appends a constant-``1`` feature placed
    on the Hann-tapered mid-band from :func:`rope_lobe_band`, identical on Q and
    K, with ``recency_gain`` riding the query side.  Under the global rotation the
    logit gains the local distance-decay lobe ::

        recency_gain · Σ_{p∈band} amp_p · cos(Δ·θ_p)

    so among keys whose content matches, the **nearest (most recent)** wins —
    monotone within the window ``W`` (``docs/rope_port_plan.md`` Phase 6).

    **Recency is local.**  Past ``W`` the lobe is non-monotone and "most recent"
    can *silently* pick a wrong key; consumers whose look-back can exceed ``W``
    need the Phase-7 global mechanism.  ``W`` and the per-step resolution are
    properties of :func:`rope_lobe_band` (≈415 / ~2.4e-3 at distance 100 for the
    default config).

    The recency band is **disjoint** from the content slow planes; this raises if
    the content width would reach into the band (content occupies planes
    ``d_head/2−W .. d_head/2−1`` — keep ``W`` small, ≲30 at the default config).
    The head is full-width rotary on the ``d_head`` grid, so it carries **no new
    ONNX/HF emission** — it rides the Phase-0 rotary path exactly like
    :func:`rotary_content_head`.
    """
    from torchwright.graph import Attn, Concatenate, LiteralValue

    require_full_rotary(
        d_head if d_rot is None else d_rot, d_head, "rotary_recency_head"
    )
    half = d_head // 2
    w_content = max(query_matrix.shape[1], key_matrix.shape[1])
    planes, amps = rope_lobe_band(d_head, base, max_positions, theta_max=theta_max)
    content_planes = set(range(half - w_content, half))
    overlap = content_planes & set(planes)
    if overlap:
        raise ValueError(
            f"content width {w_content} overlaps the recency lobe band "
            f"(planes {min(planes)}..{max(planes)}); raise d_head or narrow "
            f"the content (overlapping planes: {sorted(overlap)})."
        )

    q_full = place_on_slow_planes(query_matrix, d_head)  # (rows_q, d_head)
    k_full = place_on_slow_planes(key_matrix, d_head)  # (rows_k, d_head)

    # Constant-1 recency feature on the lobe band: one scalar per plane, the
    # rotate_half partner (p + half) left zero so the planes do not mix.
    q_rec = torch.zeros((1, d_head))
    k_rec = torch.zeros((1, d_head))
    for p, a in zip(planes, amps):
        q_rec[0, p] = recency_gain * float(a)
        k_rec[0, p] = 1.0

    one_q = LiteralValue(torch.tensor([1.0]), name="recency_lobe_query_one")
    one_k = LiteralValue(torch.tensor([1.0]), name="recency_lobe_key_one")
    query_in = Concatenate([query_in, one_q])
    key_in = Concatenate([key_in, one_k])
    q_full = torch.cat([q_full, q_rec], dim=0)
    k_full = torch.cat([k_full, k_rec], dim=0)

    d_v = len(value)
    return Attn(
        query_in=query_in,
        key_in=key_in,
        value_in=value,
        query_matrix=q_full,
        key_matrix=k_full,
        value_matrix=torch.eye(d_v),
        output_matrix=torch.eye(d_v),
        rope_base=base,
    )


def rotary_offset_head(
    value,
    delta_pos: int = -1,
    *,
    d_qk: int,
    base: float = ROPE_BASE,
    hardness: float = 100.0,
    d_rot: int | None = None,
):
    """A pure-rotary "attend to position ``j + delta_pos``" head.

    The rotary analogue of :meth:`PosEncoding.attend_to_offset`, used as the
    Phase-0 validation head (``docs/rope_port_plan.md`` §8): it carries **no**
    positional input — position enters only through the runtime rotation.

    Construction.  Query and key both read a constant ``1.0`` feature.  The
    query projects it to ``hardness · 1`` across all ``d_qk`` dims; the key
    projects the **unit** vector ``1`` (no ``hardness``) **pre-rotated by
    ``-delta_pos``** — i.e. ``W_K = R_{-delta_pos} W_Q / hardness``, so
    ``hardness`` rides the query alone and sets the softmax sharpness (the static
    rotation of §3).  With the runtime rotation
    ``R(j)`` on Q and ``R(i)`` on K, the logit becomes

        logit(j, i) ∝ Σ_p cos((i - j - delta_pos) · θ_p),

    uniquely maximal at ``i - j - delta_pos = 0`` (every plane at ``cos 0 = 1``),
    i.e. key ``i = j + delta_pos``.  ``delta_pos = -1`` ⇒ the previous position.
    V/O transport the payload unchanged.

    ``d_qk`` **must equal the compile ``d_head``** (the head fills the grid; the
    compiler asserts ``d_qk == d_head``).  ``d_rot`` (default ``d_qk``) is the
    rotary width: the first ``d_rot`` dims rotate and the rest are the unrotated
    NoPE tail (vanilla partial rotary).  Production callers pass
    ``d_qk=rope.d_head, d_rot=rope.d_rot``.

    Partial rotary is harmless here: the query tail is ``hardness·1`` and the key
    tail is ``1`` (both unrotated), so they contribute a position-independent
    constant ``hardness·(d_qk - d_rot)`` to every logit, which the softmax's
    row-max subtraction cancels; the rotated front still peaks uniquely at
    ``i - j - delta_pos = 0``.
    """
    from torchwright.graph import Attn, LiteralValue

    if delta_pos == 0:
        return value

    d_rot = d_qk if d_rot is None else d_rot
    d_v = len(value)
    one = LiteralValue(torch.tensor([1.0]), name="rotary_offset_query_one")

    query_matrix = hardness * torch.ones((1, d_qk))  # W_Q · 1 = hardness · 1

    # W_K · 1 = R_{-delta_pos} (1-vector) over the rotary front d_rot: the static
    # key pre-rotation.  apply_rope rotates only the first d_rot dims, so the tail
    # [d_rot:d_qk] stays 1 (the NoPE constant the softmax cancels — see docstring).
    cos, sin = rope_cos_sin(torch.tensor([-delta_pos]), d_rot, base)
    key_matrix = apply_rope(torch.ones((1, d_qk)), cos, sin)  # (1, d_qk)

    return Attn(
        query_in=one,
        key_in=one,
        value_in=value,
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=torch.eye(d_v),
        output_matrix=torch.eye(d_v),
        rope_base=base,
        rope_d_rot=d_rot,
    )
