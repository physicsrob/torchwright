"""One-hot-native lookup table on the swish machine.

``onehot_lookup`` is the one-hot counterpart of
:func:`torchwright.ops.swiglu.map_select.map_to_table` — same
bank-of-indicator-lanes shape, but with **no margin knob at all**: the
margin is the integer count structure (the counting trick ports
verbatim), and the module ``scale`` is the only constant.  Two shapes;
only one is machine-specific:

* **Single one-hot input** (``n_blocks = 1``): a plain
  :class:`~torchwright.graph.Linear` whose row ``k`` holds the value for
  token ``k`` — no activation, linear hardware, untouched by the port.

* **Several one-hot blocks concatenated**: one degenerate swish lane per
  table row.  ``inp·key_i`` counts agreeing blocks — an integer in
  ``0..n_blocks``, equal to ``n_blocks`` only at the exact key — so the
  ``-(n_blocks - 0.5)`` bias parks every lane's hinge argument at
  exactly ``+0.5`` (the winner) or ``≤ -0.5`` (everyone else).
  Sharpened, those are ``±scale/2`` — deep in saturation on both sides:
  the winner's indicator is exactly 0.5 (``σ(50) = 1.0`` and ``50/100``
  is exact), a losing lane leaks ``hinge(-0.5) ≈ -1e-22`` — not the
  exact zero of ``σ(-scale)`` (``e^{-50}`` is representable) but twenty
  orders below visibility, vanishing under fp32 addition.

Because exactly one row fires (or none → ``default``), the claimed value
range is the *tight* ``[min, max]`` over the values and the default —
the reason this op exists (``map_to_table``'s pessimistic
``default ± Σ|Δ|`` widening blew up chained interval arithmetic).  The
closing assert's slack must budget the *accumulated* input leak: a
machine-built one-hot (in_range indicators through ``bool_to_01``)
carries a per-element fp32 round-trip leak of order 1e-5 — individually
invisible, but a ``d_key``-wide key sums ``d_key`` of them, each
weighted by up to the largest table magnitude (measured on the
calculator digit pipeline: ``d_key = 61``, values ≤ 6, output error up
to ~2e-3 in exact math — past the 1e-3 default).  The assert atol is
therefore sized by ``_lookup_numeric_slack(max_abs, 1.0, d_key)`` —
check-only slack; the claimed range downstream analysis sees stays
tight.  Winner rounding itself (~1 ulp of the value, the
``×scale/÷scale`` round trip) and dip leaks are inside that budget by
construction.  Noise pass-through is unchanged from the ReLU form: an
input one-hot off by ε shifts the winner's indicator by exactly ε (the
saturated gate is linear down to a count deviation of 0.33), so output
error is ``2ε·|value_i − default|``.
"""

from typing import Dict

import torch

from torchwright.graph import Linear, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops._math import _lookup_numeric_slack
from torchwright.ops.const import scale
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn

_ONEHOT_ATOL = 1e-6


def _count_one_hot_blocks(key: torch.Tensor) -> int:
    """Number of ``1`` entries in ``key``, after checking it is all 0/1."""
    near_zero = key.abs() <= _ONEHOT_ATOL
    near_one = (key - 1.0).abs() <= _ONEHOT_ATOL
    if not bool((near_zero | near_one).all()):
        raise ValueError(
            "onehot_lookup keys must be concatenations of one-hot blocks "
            "(every entry 0 or 1); got a key with an entry that is neither"
        )
    return int(near_one.sum().item())


def onehot_lookup(
    inp: Node,
    key_to_value: Dict[torch.Tensor, torch.Tensor],
    default: torch.Tensor,
) -> Node:
    """Map a one-hot (or concatenation of one-hots) input to a lookup table.

    Args:
        inp: Node whose value is a one-hot block, or several one-hot blocks
            concatenated (e.g. ``concat([digit_a, digit_b, carry])``).
        key_to_value: The table.  Every key must be a 0/1 vector of width
            ``len(inp)`` with the same number of ones (one per block); every
            value must have width ``len(default)``.
        default: Value returned for any input that matches no key.

    Returns:
        Node holding the looked-up value, carrying a tight ``[min, max]``
        value range over the table's values and the default.

    .. noise-footer::

       Max error: 7.629e-06 abs, 7.629e-08 rel over 4096 samples;
       measured at commit 6694f64. See docs/numerical_noise.md.
    """
    if not key_to_value:
        raise ValueError("onehot_lookup requires a non-empty table")

    d_key = len(inp)
    d_value = len(default)
    for key in key_to_value:
        if len(key) != d_key:
            raise ValueError(
                f"onehot_lookup key width {len(key)} != input width {d_key}"
            )
    for value in key_to_value.values():
        if len(value) != d_value:
            raise ValueError(
                f"onehot_lookup value width {len(value)} != default width {d_value}"
            )

    block_counts = {_count_one_hot_blocks(key) for key in key_to_value}
    if len(block_counts) != 1:
        raise ValueError(
            f"onehot_lookup keys must all have the same number of one-hot "
            f"blocks; got {sorted(block_counts)}"
        )
    n_blocks = block_counts.pop()
    if n_blocks < 1:
        raise ValueError("onehot_lookup keys must have at least one one-hot block")

    # Tight output range: the output is always exactly one value row (or the
    # default), so [min, max] over them all bounds every element.
    stacked = torch.stack(
        [v.to(torch.float32) for v in key_to_value.values()]
        + [default.to(torch.float32)]
    )
    out_range = Range(float(stacked.min().item()), float(stacked.max().item()))

    if n_blocks == 1:
        # A single one-hot selects one row: the lookup *is* a selection
        # matrix — linear hardware, identical on both machines.
        weight = default.to(torch.float32).unsqueeze(0).repeat(d_key, 1).clone()
        for key, value in key_to_value.items():
            row = int((key > 0.5).nonzero(as_tuple=False)[0].item())
            weight[row] = value.to(torch.float32)
        result = Linear(inp, weight, name="onehot_lookup_select")
    else:
        # One degenerate lane per row: gate row scale·key_i, bias
        # -scale·(n_blocks - 0.5); the winner's hinge argument is exactly
        # +scale/2, everyone else ≤ -scale/2.  out_proj scales the 0.5
        # spike back to (value - default), /scale folded in.
        n_lanes = len(key_to_value)
        gate_proj = torch.zeros(n_lanes, d_key)
        gate_bias = torch.full((n_lanes,), -scale * (n_blocks - 0.5))
        output_proj = torch.zeros(n_lanes, d_value)
        default_f = default.to(torch.float32)
        for i, (key, value) in enumerate(key_to_value.items()):
            gate_proj[i] = scale * key.to(torch.float32)
            output_proj[i] = 2.0 * (value.to(torch.float32) - default_f) / scale
        result = swiglu_ffn(
            inp,
            gate_proj,
            gate_bias,
            output_proj,
            default_f,
            name="onehot_lookup",
        )

    # Check-only slack for the accumulated input leak (see module
    # docstring): d_key summed per-element ~1e-5 round-trip leaks, each
    # weighted by up to the largest table magnitude.  The claimed range
    # itself stays tight — atol never reaches static analysis.
    max_abs = max(abs(out_range.lo), abs(out_range.hi))
    return assert_matches_value_type(
        result,
        NodeValueType(value_range=out_range),
        atol=_lookup_numeric_slack(max_abs, 1.0, d_key),
    )
