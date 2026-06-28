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

import torch

# LLaMA3-family base.  See docs/rope_port_plan.md §6 — reconcile downstream
# analyses (measured at 1e6) if this changes.
ROPE_BASE = 500000.0


def rope_inv_freq(width: int, base: float) -> torch.Tensor:
    """Per-plane angular frequencies ``θ_p = base^(-2p/width)``.

    Returns a ``(width // 2,)`` float32 tensor.  ``width`` must be even.
    """
    if width % 2 != 0:
        raise ValueError(f"RoPE width must be even, got {width}")
    p = torch.arange(0, width, 2, dtype=torch.float64)
    return (base ** (-p / width)).to(torch.float32)


def recency_plane_index(
    d_head: int,
    base: float,
    max_positions: int,
    *,
    seam_frac: float = 0.05,
) -> int:
    """Pick the recency plane: the **fastest** grid plane whose phase still never
    wraps over the whole rollout (``docs/rope_port_plan.md`` §3 bucket 2, §6).

    The bucket-2 recency readout reads the BOS-relative phase ``phi(pos) =
    pos · θ_p`` of one rotary plane.  The octant ramp built on it has a single
    discontinuity at the seam ``phi = 0 mod 2π`` (the wrap), so the whole
    rollout's phase must stay inside ``(seam_frac·2π, (1−seam_frac)·2π)``.  That
    bounds the plane's angular frequency: one turn (``2π/θ_p``) must cover at
    least ``max_positions / (1 − 2·seam_frac)`` positions.  Among the planes
    that satisfy this, the **fastest** (largest ``θ_p``, smallest index) gives
    the steepest per-token phase step, i.e. the most recency resolution.

    ``θ_p = base^(−2p/d_head)`` decreases as ``p`` grows, so turn grows with
    ``p``; the first ``p`` whose turn clears the bound is the answer.

    Raises ``ValueError`` if no plane on the grid is slow enough (raise
    ``d_head`` or ``base`` — the slowest plane ``θ_{d_head/2−1} ≈ 1/base`` still
    wraps within the rollout).
    """
    need_turn = max_positions / (1.0 - 2.0 * seam_frac)
    for p in range(d_head // 2):
        theta_p = base ** (-2.0 * p / d_head)
        turn = 2.0 * math.pi / theta_p
        if turn >= need_turn:
            return p
    raise ValueError(
        f"no rotary plane on the base={base} d_head={d_head} grid turns slowly "
        f"enough for a {max_positions}-position rollout (need turn ≥ "
        f"{need_turn:.0f}); raise d_head or base."
    )


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
    """Apply ``rotate_half`` RoPE.  ``x`` is ``(..., P, width)``; ``cos``/``sin``
    are ``(P, width)`` and broadcast over the leading dims."""
    return x * cos + rotate_half(x) * sin


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

    Returns a ``(rows, d_head)`` matrix; pass it to a ``rotary=True`` ``Attn``.
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


def rotary_content_head(
    query_in,
    key_in,
    value,
    query_matrix: torch.Tensor,
    key_matrix: torch.Tensor,
    *,
    d_head: int,
    base: float = ROPE_BASE,
):
    """A content-selection ``Attn`` made **rotary on slow planes**.

    Takes a content head's compact ``(·, W)`` ``query_matrix`` / ``key_matrix``
    (the same layout the ``attend_*`` builders construct — score in col 0,
    validity / bucket / dot dims in later cols) and rebuilds it as a full-width
    ``d_head`` rotary head with the content relocated onto the slowest ``W``
    planes via :func:`place_on_slow_planes`.  V/O pass the payload through
    unchanged.  This is the Phase-3 content capability (``docs/rope_port_plan.md``
    §3, §8): selection by content survives the global rotation because the match
    rides quasi-static planes.
    """
    from torchwright.graph import Attn

    d_v = len(value)
    return Attn(
        query_in=query_in,
        key_in=key_in,
        value_in=value,
        query_matrix=place_on_slow_planes(query_matrix, d_head),
        key_matrix=place_on_slow_planes(key_matrix, d_head),
        value_matrix=torch.eye(d_v),
        output_matrix=torch.eye(d_v),
        rotary=True,
        rope_base=base,
    )


def rotary_offset_head(
    value,
    delta_pos: int = -1,
    *,
    d_qk: int = 8,
    base: float = ROPE_BASE,
    hardness: float = 100.0,
):
    """A pure-rotary "attend to position ``j + delta_pos``" head.

    The rotary analogue of :meth:`PosEncoding.attend_to_offset`, used as the
    Phase-0 validation head (``docs/rope_port_plan.md`` §8): it carries **no**
    positional input — position enters only through the runtime rotation.

    Construction.  Query and key both read a constant ``1.0`` feature.  The
    query projects it to ``hardness · 1`` across all ``d_qk`` dims; the key
    projects it to that same vector **pre-rotated by ``-delta_pos``** (the
    static ``W_K = R_{-delta_pos} W_Q`` of §3).  With the runtime rotation
    ``R(j)`` on Q and ``R(i)`` on K, the logit becomes

        logit(j, i) ∝ Σ_p cos((i - j - delta_pos) · θ_p),

    uniquely maximal at ``i - j - delta_pos = 0`` (every plane at ``cos 0 = 1``),
    i.e. key ``i = j + delta_pos``.  ``delta_pos = -1`` ⇒ the previous position.
    V/O transport the payload unchanged.
    """
    from torchwright.graph import Attn, LiteralValue

    if delta_pos == 0:
        return value

    d_v = len(value)
    one = LiteralValue(torch.tensor([1.0]), name="rotary_offset_query_one")

    query_matrix = hardness * torch.ones((1, d_qk))  # W_Q · 1 = hardness · 1

    # W_K · 1 = R_{-delta_pos} (1-vector): the static key pre-rotation.
    cos, sin = rope_cos_sin(torch.tensor([-delta_pos]), d_qk, base)
    key_matrix = apply_rope(torch.ones((1, d_qk)), cos, sin)  # (1, d_qk)

    return Attn(
        query_in=one,
        key_in=one,
        value_in=value,
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=torch.eye(d_v),
        output_matrix=torch.eye(d_v),
        rotary=True,
        rope_base=base,
    )
