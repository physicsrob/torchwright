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
