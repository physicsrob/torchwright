"""The FFN node: a first-class feed-forward unit (lanes + output projection).

An FFN is the *structural* unit the compiled transformer's MLP sublayer
realizes directly, as opposed to the *semantic* ``Linear -> ReLU -> Linear``
chain the scheduler mines today.  It is the shape the gated (SwiGLU) machine
needs; step 1 (this phase) instantiates only the degenerate ReLU form (each
lane's "up" factor is the constant 1), so an FFN compiles to exactly the same
``linear1 -> ReLU -> linear2`` MLP module a mined chain does — identical math,
identical weights.

See ``docs/block_lane_spec.md`` for the full spec.  Two facts worth pinning
here because they are easy to get wrong:

- **An FFN is a packable unit, not a sublayer.**  The scheduler bins many
  FFNs' lanes into one MLP sublayer's hidden pool.  Nothing here promises
  sublayer ownership — do not rebuild the ``linear_relu_linear`` "one call =
  one MLP sublayer" assumption on top of it.
- **An FFN owns its input projection.**  The gate rows are the FFN's own
  weights; a shared upstream value stays a separate ``Linear`` feeding the
  FFN.
"""

from typing import cast

import torch

from torchwright.graph.node import Node
from torchwright.graph.value_type import NodeValueType


class FFN(Node):
    """One feed-forward unit: lanes computed from a gate (and optional up)
    projection, combined by an output projection.

    For an input row ``x`` (width ``d_input``)::

        lane_out[j] = act(gate_proj[j] · x + gate_bias[j]) * up[j](x)
        output      = lane_out @ out_proj + out_bias

    where ``up[j](x)`` is the constant 1 (degenerate lane, ``up_proj is None``
    — step 1's only form) or ``up_proj[j] · x + up_bias[j]`` (gated lane, step
    2).  ``act`` is ``"relu"`` (step 1) or ``"swish"`` (step 2).  Lane kind is
    uniform per FFN.

    Attributes:
        gate_proj: ``(n_lanes, d_input)`` gate projection (row per lane).
        gate_bias: ``(n_lanes,)`` gate bias.
        up_proj: ``(n_lanes, d_input)`` up projection, or ``None`` (degenerate).
        up_bias: ``(n_lanes,)`` up bias, or ``None`` (degenerate).
        out_proj: ``(n_lanes, d_output)`` output projection.
        out_bias: ``(d_output,)`` output bias.
        activation: ``"relu"`` or ``"swish"``.
    """

    gate_proj: torch.Tensor
    gate_bias: torch.Tensor
    up_proj: torch.Tensor | None
    up_bias: torch.Tensor | None
    out_proj: torch.Tensor
    out_bias: torch.Tensor
    activation: str

    def __init__(
        self,
        input_node: Node,
        gate_proj: torch.Tensor,
        gate_bias: torch.Tensor,
        out_proj: torch.Tensor,
        out_bias: torch.Tensor,
        up_proj: torch.Tensor | None = None,
        up_bias: torch.Tensor | None = None,
        activation: str = "relu",
        name: str = "",
    ) -> None:
        if activation not in ("relu", "swish"):
            raise ValueError(
                f"FFN activation must be 'relu' or 'swish', got {activation!r}"
            )
        n_lanes, d_input = gate_proj.shape
        assert len(input_node) == d_input, (
            f"FFN input width {len(input_node)} != gate_proj d_input {d_input}"
        )
        assert gate_bias.shape == (n_lanes,), (
            f"gate_bias shape {tuple(gate_bias.shape)} != ({n_lanes},)"
        )
        assert out_proj.shape[0] == n_lanes, (
            f"out_proj rows {out_proj.shape[0]} != n_lanes {n_lanes}"
        )
        d_output = out_proj.shape[1]
        assert out_bias.shape == (d_output,), (
            f"out_bias shape {tuple(out_bias.shape)} != ({d_output},)"
        )
        if (up_proj is None) != (up_bias is None):
            raise ValueError(
                "FFN up_proj and up_bias must both be set (gated lane) or "
                "both None (degenerate lane)."
            )
        if up_proj is not None:
            assert up_proj.shape == (
                n_lanes,
                d_input,
            ), f"up_proj shape {tuple(up_proj.shape)} != ({n_lanes}, {d_input})"
            assert cast("torch.Tensor", up_bias).shape == (n_lanes,), (
                f"up_bias shape {tuple(cast('torch.Tensor', up_bias).shape)} != ({n_lanes},)"
            )

        self.gate_proj = gate_proj
        self.gate_bias = gate_bias
        self.up_proj = up_proj
        self.up_bias = up_bias
        self.out_proj = out_proj
        self.out_bias = out_bias
        self.activation = activation
        self.d_input = d_input

        super().__init__(d_output, [input_node], name=name)

    @property
    def n_lanes(self) -> int:
        return self.gate_proj.shape[0]

    @property
    def is_degenerate(self) -> bool:
        """True when every lane's up-factor is the constant 1 (step-1 form)."""
        return self.up_proj is None

    def _activate(self, gate: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return torch.clamp(gate, min=0.0)
        # swish(z) = z * sigmoid(z)
        return gate * torch.sigmoid(gate)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.inputs[0].compute(n_pos, input_values)
        assert x.shape == (n_pos, self.d_input)

        gate = torch.matmul(x, self.gate_proj.t()) + self.gate_bias
        lane = self._activate(gate)
        if self.up_proj is not None:
            up = torch.matmul(x, self.up_proj.t()) + cast("torch.Tensor", self.up_bias)
            lane = lane * up
        return torch.matmul(lane, self.out_proj) + self.out_bias

    def compute_value_type(self) -> NodeValueType:
        # Default; the real per-component range comes from the affine bound
        # (_ffn_rule).  Semantic overrides stay op-level, as for Linear/ReLU.
        return NodeValueType()

    def num_params(self):
        n_lanes = self.n_lanes
        gate = n_lanes * self.d_input + n_lanes
        up = 0 if self.up_proj is None else n_lanes * self.d_input + n_lanes
        out = n_lanes * self.d_output + self.d_output
        return gate + up + out
