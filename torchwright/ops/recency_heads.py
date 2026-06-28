"""Bucket-2 recency rank — the two graded rotary heads that feed the octant ramp.

docs/rope_port_plan.md §3 bucket 2 / Phase 1b / Phase 4.  The octant ramp
(:func:`torchwright.ops.recency_ramp.octant_recency_ramp`) is a monotone
position signal built from two centered weights ``u = sigmoid(M·cos φ) − 0.5``
and ``v = sigmoid(M·sin φ) − 0.5``, where ``φ = pos·θ`` is the BOS-relative
phase of one rotary plane.  This module builds the **attention heads** that
materialise ``u`` and ``v`` from the rotation, and the full ``heads → ramp``
chain.

The construction (the gate-(c)/(d) expression Phase 1b deferred).  A single
2-key softmax over ``{BOS, self}`` does **not** give a clean ``sigmoid(M·cos φ)``
— ``self`` rides the same rotary plane as BOS and injects a constant
``−M·cos ψ`` shift that breaks the octant ramp's cos/sin symmetry.  Instead each
head reads two **marked** tokens that are always causally visible:

- **BOS** (position 0): its key carries energy on the recency plane, so seen
  from query ``j`` its logit is ``L + M·cos(j·θ + φ0)`` (cos-head; ``M·sin`` for
  the sin-head).  ``L`` is a position-independent DC term on a near-static slow
  plane; ``φ0`` shifts the whole rollout's phase off the seam.
- **REF** (a second marked token): **no** recency energy, so its logit is the
  bare DC ``L`` — a constant reference.

Every other (unmarked) key sees logit ``≈ 0``.  With ``L ≫ ln N`` the softmax is
effectively 2-key ``{BOS, REF}``, so the weight on BOS — read out by giving BOS
value ``1`` and everything else value ``0`` — is

    w_BOS = sigmoid((L + M·cos φ) − L) = sigmoid(M·cos φ),     u = w_BOS − 0.5.

This needs **full-width** rotary heads (``d_qk == d_head``): the recency plane
must be an actual slow frequency on the ``θ_p = base^(−2p/d_head)`` grid (a
width-2 head only has ``θ_0 = 1``, one turn per token).  Like the Phase-3
content heads, it therefore needs ``d_head`` at construction — so DOOM wiring
lands in Phase 5; this module proves the capability.
"""

import math

import torch

from torchwright.graph import Attn, Concatenate, LiteralValue, Node
from torchwright.graph.rope import ROPE_BASE, recency_plane_index
from torchwright.ops.arithmetic_ops import add_const
from torchwright.ops.recency_ramp import octant_recency_ramp

# Leakage DC.  The {BOS, REF} pair must dominate the N−2 unmarked background
# keys: residual drift is ~N·exp(−L), so L must clear ln(N/resolution).  At the
# 61440 cache cap and the octant gap-1 weight signal (~3e-5) this is ≈25 logits
# (docs/rope_port_plan.md §8 gate d, "≈22+").  25 leaves ~150× margin.
_DC_GAIN = 25.0


def recency_phase_heads(
    bos_marker: Node,
    ref_marker: Node,
    *,
    d_head: int,
    max_positions: int,
    base: float = ROPE_BASE,
    gain: float = 2.0,
    seam_frac: float = 0.05,
    dc_gain: float = _DC_GAIN,
) -> tuple[Node, Node]:
    """The two graded rotary heads — returns ``(u, v)`` for the octant ramp.

    Args:
        bos_marker: length-1 node, ``1`` at the BOS position and ``0`` elsewhere
            (the phase carrier; also reused as the value so the head output is
            the softmax weight on BOS).
        ref_marker: length-1 node, ``1`` at a second always-visible reference
            token and ``0`` elsewhere (the constant DC reference).
        d_head: rotary width.  Must be even; the recency and DC planes must be
            distinct, which holds whenever the rollout does not need the very
            slowest plane (raise ``d_head``/``base`` otherwise).
        max_positions: rollout length the plane is sized never to wrap over
            (size to the cache cap, not the typical frame — past the wrap the
            recency order silently inverts).
        base: rotary base (LLaMA3 ``5e5`` by default).
        gain: head gain ``M`` — must match the gain baked into the ramp's
            octant offset table (``octant_recency_ramp(gain=…)``).
        seam_frac: fraction of ``2π`` kept clear of the seam at each end.
        dc_gain: the leakage DC ``L`` (see ``_DC_GAIN``).

    Returns:
        ``(u, v)`` length-1 nodes, ``sigmoid(M·cos φ) − 0.5`` and
        ``sigmoid(M·sin φ) − 0.5``; feed straight to ``octant_recency_ramp``.
    """
    if d_head % 2 != 0:
        raise ValueError(f"recency heads need an even d_head, got {d_head}")
    assert len(bos_marker) == 1 and len(ref_marker) == 1
    half = d_head // 2
    plane = recency_plane_index(d_head, base, max_positions, seam_frac=seam_frac)
    dc_plane = half - 1  # slowest plane -> most static DC
    if plane >= dc_plane:
        raise ValueError(
            f"recency plane {plane} collides with the DC plane {dc_plane} at "
            f"d_head={d_head}; raise d_head so the recency plane stays well "
            f"below the slowest plane."
        )
    phi0 = seam_frac * 2.0 * math.pi

    # query reads a constant 1.0 (no positional input); key reads the markers.
    one = LiteralValue(torch.tensor([1.0]), name="recency_phase_query_one")
    key_in = Concatenate([bos_marker, ref_marker])

    def build(which: str) -> Node:
        # Query feature on the recency plane (dims plane, plane+half): the logit
        # against BOS's key [1, 0] is M·cos(j·θ+φ0) (cos) / M·sin(j·θ+φ0) (sin).
        qm = torch.zeros((1, d_head))
        if which == "cos":
            qm[0, plane] = gain * math.cos(phi0)
            qm[0, plane + half] = gain * math.sin(phi0)
        else:  # sin: query rotated 90° -> reads sin φ
            qm[0, plane] = gain * math.sin(phi0)
            qm[0, plane + half] = -gain * math.cos(phi0)
        qm[0, dc_plane] = dc_gain  # DC: query side carries the gain L

        # Key: BOS carries the recency-plane unit feature + DC; REF carries DC
        # only (no recency energy -> constant reference logit).
        km = torch.zeros((2, d_head))  # rows: [bos_marker, ref_marker]
        km[0, plane] = 1.0  # BOS recency key [1, 0]
        km[0, dc_plane] = 1.0  # BOS DC (unit; gain on the query side)
        km[1, dc_plane] = 1.0  # REF DC

        head = Attn(
            query_in=one,
            key_in=key_in,
            value_in=bos_marker,  # value 1 at BOS, 0 elsewhere -> output = w_BOS
            query_matrix=qm,
            key_matrix=km,
            value_matrix=torch.eye(1),
            output_matrix=torch.eye(1),
            rotary=True,
            rope_base=base,
        )
        return add_const(head, -0.5)  # center the weight -> u / v

    return build("cos"), build("sin")


def recency_rank(
    bos_marker: Node,
    ref_marker: Node,
    *,
    d_head: int,
    max_positions: int,
    base: float = ROPE_BASE,
    gain: float = 2.0,
    seam_frac: float = 0.05,
    dc_gain: float = _DC_GAIN,
    ramp_sharpness: float | None = None,
) -> Node:
    """Full bucket-2 chain: two graded heads → the monotone octant recency ramp.

    Returns a length-1 node, strictly increasing in absolute position over the
    seam-safe rollout (multiply by the rank gain ``G`` at the selection site;
    ``G`` is argmax-invariant).  Args mirror :func:`recency_phase_heads`.

    The rank is valid (monotone) for positions ``j ≥ 2``.  BOS (``j=0``) and REF
    (``j=1``) attend only to themselves, so their rank is **degenerate** — an
    outlier inside the ramp's value box, not part of the monotone sequence.  A
    consumer must keep these reference tokens content-excluded (the
    content-dominance bound on the selection head does this); their own rank is
    never meaningfully consumed.
    """
    u, v = recency_phase_heads(
        bos_marker,
        ref_marker,
        d_head=d_head,
        max_positions=max_positions,
        base=base,
        gain=gain,
        seam_frac=seam_frac,
        dc_gain=dc_gain,
    )
    return octant_recency_ramp(u, v, gain=gain, sharpness=ramp_sharpness)
