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

    # Whether position-local compute ops (standalone Linears and Adds —
    # add_into/compute_add on attention, add_into_bypass/compute_add_bypass
    # on MLP) use attention heads or MLP bypass.
    #   "always": use attention heads (legacy behavior)
    #   "never":  use MLP bypass, defer to next layer if MLP full
    # A node whose MLP demand (2*d_output) exceeds a layer's usable hidden
    # pool goes to attention regardless; a tied compile's held-target Add
    # is pinned to attention under every setting.  On CP-SAT paths the
    # policy is consulted only under cpsat_flex_routing=False — the legacy
    # attention-only Add configuration is LEGACY_POLICY *plus*
    # cpsat_flex_routing=False (docs/plan_additional_mlp_routing.md).
    local_in_attention: Literal["always", "never"] = "never"

    # Residual stream occupancy fraction above which the scheduler switches
    # its sort order from "longest critical path first" to "free columns
    # first."  Under pressure, freeing residual columns (cancels and ops
    # whose output reclaims more columns than it allocates) gets prioritized
    # over the deepest-chain-first heuristic, to keep the residual stream
    # from filling up.
    pressure_threshold: float = 0.75


#: The historical attention-preferring routing.  For attention-only Adds on
#: CP-SAT paths, pair it with ``cpsat_flex_routing=False`` — the policy alone
#: cannot pin routes the solver is asked to choose.
LEGACY_POLICY = SchedulingPolicy(
    local_in_attention="always",
)
