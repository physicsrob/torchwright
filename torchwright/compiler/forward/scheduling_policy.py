from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SchedulingPolicy:
    """Controls how the forward compiler routes ops to attention vs MLP.

    The compiler maps graph nodes to two sublayer types: attention heads
    (cross-position communication via Q/K/V/O matrices) and MLP slots
    (position-local transforms via Linear1/ReLU/Linear2).  Many ops —
    standalone Linears, add_into, compute_add — are position-local
    and can use either mechanism.  This policy controls which they use.
    """

    # Whether standalone Linears (attn_transport on attention, mlp_bypass
    # on MLP) use attention heads or MLP bypass.
    #   "always": use attention heads (legacy behavior)
    #   "never":  use MLP bypass, defer to next layer if MLP full
    # A node whose MLP demand (2*d_output) exceeds a layer's usable hidden
    # pool goes to attention regardless.  On CP-SAT paths the policy is
    # consulted only under cpsat_flex_routing=False.
    local_in_attention: Literal["always", "never"] = "never"

    # The same choice for Adds (add_into/compute_add on attention,
    # add_into_bypass/compute_add_bypass on MLP), with the OPPOSITE
    # default.  A static MLP default costs one layer for every Add wedged
    # between two MLP-sublayer ops: an attention-routed Add rides the
    # attention sublayer that sits idle between consecutive MLP sublayers
    # (its output is readable by the same layer's MLP), while an
    # MLP-routed Add pays the mlp->mlp one-layer gap on both sides.
    # Measured 2026-07-14 (scripts/measure_add_routing_flip.py): a static
    # MLP default cost calculator +5 and fibonacci +1 layers at
    # optimize=0 while winning nothing.  Adds still reach MLP wherever a
    # per-node chooser exists — CP-SAT flex routing decides regardless of
    # this knob (docs/plan_additional_mlp_routing.md, *Policy and
    # compatibility decision*).  Same carve-outs as local_in_attention:
    # an Add too wide for the hidden pool goes to attention, and a tied
    # compile's held-target Add is pinned to attention under every
    # setting.
    add_in_attention: Literal["always", "never"] = "always"

    # Whether dead nodes may be zeroed by the MLP ``cancel_bypass``
    # mechanism when the layer's batched attention cancel is full.
    #   "auto":   attention batch first, MLP fall-back (the default walk)
    #   "always": attention only — a cancel that does not fit waits for a
    #             later layer's batch
    # An MLP bypass cancel on the swish machine deposits a
    # schedule-dependent fp32 residue (~2^-41) on near-zero lanes of the
    # reused columns, so two compiles of the same graph whose layouts
    # differ (e.g. the RMSNorm pinned-column reservation) can disagree at
    # that magnitude on lanes an MLP cancel touched.  Pin to "always"
    # where a bit-exact cross-compile comparison is the point (the norm
    # identity certification); production compiles keep "auto".  Eager
    # path only — CP-SAT's per-node cancel mechanism is a solver choice.
    cancel_in_attention: Literal["always", "auto"] = "auto"

    # Residual stream occupancy fraction above which the scheduler switches
    # its sort order from "longest critical path first" to "free columns
    # first."  Under pressure, freeing residual columns (cancels and ops
    # whose output reclaims more columns than it allocates) gets prioritized
    # over the deepest-chain-first heuristic, to keep the residual stream
    # from filling up.
    pressure_threshold: float = 0.75


#: The historical attention-preferring routing.  ``add_in_attention`` is
#: set explicitly (not left to its default) so LEGACY keeps meaning
#: "everything on attention" even if a default moves.  For attention-only
#: Adds on CP-SAT paths, pair it with ``cpsat_flex_routing=False`` — the
#: policy alone cannot pin routes the solver is asked to choose.
LEGACY_POLICY = SchedulingPolicy(
    local_in_attention="always",
    add_in_attention="always",
)
