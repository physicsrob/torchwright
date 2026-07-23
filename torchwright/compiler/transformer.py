from typing import TYPE_CHECKING, Any, Optional, cast

import torch

from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.compiler.residual_assignment import (
    ResidualAssignment,
    ResidualStreamState,
)
from torchwright.graph import (
    Concatenate,
    Embedding,
    InputNode,
    LiteralValue,
    Node,
)

if TYPE_CHECKING:
    from torchwright.compiler.forward.compile import RmsNormSpec
    from torchwright.compiler.forward.cpsat_scheduler import (
        ScheduleAssignment,
        ScheduleResult,
        SolveStats,
    )
    from torchwright.compiler.groups.attn_sublayer import AttnSubLayer
    from torchwright.compiler.groups.mlp_sublayer import GatedMLPSubLayer, MLPSubLayer
    from torchwright.compiler.realization import RealizationTable

# A sublayer's ``forward(..., return_states=True)`` result: the output
# tensor plus one (ResidualStreamState, tensor) snapshot per named state.
_SublayerStates = dict[str, tuple[ResidualStreamState, torch.Tensor]]
_SublayerForwardStatesResult = tuple[torch.Tensor, _SublayerStates]


class HeadlessTransformer:
    """Stack of transformer layers without embedding or unembedding heads.

    Produced by the forward compiler. Use ``compute()`` to run the
    transformer on graph-level inputs and retrieve output node values.
    """

    layers: list[TransformerLayer]
    d: int
    d_hidden: int
    d_head: int
    n_heads: int
    residual_assignment: ResidualAssignment | None
    # 2-D weight-matrix occupancy recorded during weight writing (see
    # forward.weight_writer.PlacementRecorder).  ``None`` until compile sets it.
    placements: Any | None
    # --- Attributes attached by ``forward_compile`` (forward/compile.py) ---
    # Emission mode: False means no physical bias parameters.
    bias: bool
    # Solver provenance, populated only when CP-SAT runs (optimize>0); stays
    # None for the heuristic path.
    cpsat_solve_stats: Optional["SolveStats"]
    schedule_result: Optional["ScheduleResult"]
    # Solve-only measurement output (None when no feasible incumbent).
    cpsat_assignment: Optional["ScheduleAssignment"]
    # Diagnostics from the dominating-replay-plan choice.
    replay_candidate_diagnostics: dict[str, object]
    # Pinned-constant RMSNorm layout; None when the norm is off.
    rms_norm_spec: Optional["RmsNormSpec"]
    # Resolved realization table the compile scheduled against.
    realization_table: "RealizationTable"
    # Per-layer attention-head usage by op type (observability only).
    per_layer_head_counts: list[dict[str, int]]

    def __init__(
        self,
        d: int,
        d_head: int,
        d_hidden: int | None = None,
        activation: str = "relu",
        n_heads: int | None = None,
    ) -> None:
        if activation not in ("relu", "swish"):
            raise ValueError(
                f"HeadlessTransformer activation must be 'relu' or 'swish', "
                f"got {activation!r}"
            )
        from torchwright.compiler.utils import resolve_n_heads

        self.d = d
        self.d_hidden = d if d_hidden is None else d_hidden
        self.d_head = d_head
        self.n_heads = resolve_n_heads(d, d_head, n_heads)
        # Machine kind, uniform across all layers: "relu" compiles MLP
        # sublayers as linear1->ReLU->linear2, "swish" as the gated
        # (SwiGLU) sublayer.  Chosen by the compiler from the graph's
        # FFN nodes' uniform activation.
        self.activation = activation
        self.layers = []
        self.residual_assignment = None
        self.placements = None

    @property
    def device(self) -> torch.device:
        if self.layers:
            return self.layers[0].attn.attn.query_matrix.device
        return torch.device("cpu")

    def to(self, device: str | torch.device) -> "HeadlessTransformer":
        for layer in self.layers:
            layer.to(device)
        return self

    def add_layer(self, *, append: bool = False) -> TransformerLayer:
        layer = TransformerLayer(
            self.d,
            self.d_head,
            d_hidden=self.d_hidden,
            activation=self.activation,
            n_heads=self.n_heads,
        )
        if append:
            self.layers.append(layer)
        else:
            self.layers = [layer, *self.layers]
        return layer

    def get_input_res_stream(
        self,
        n_pos: int,
        input_values: dict[str, Any],
        past_len: int = 0,
    ) -> torch.Tensor:
        """Build the initial residual stream for ``n_pos`` positions.

        ``past_len`` is **vestigial** under RoPE: position now enters only
        through the rotary attention rotation (applied inside the layers from
        ``cache_position``), not through any residual feature built here.
        Literals, InputNode values, and Embedding lookups are all
        position-independent, so the residual stream this builds does not
        depend on ``past_len`` — the parameter is accepted for call-site
        compatibility but no longer used (a candidate for removal; left as-is
        to avoid a signature change).
        """
        assert self.residual_assignment
        in_state = self.layers[0].attn.in_state
        res_stream = torch.zeros((n_pos, self.d))

        for node in self.residual_assignment.get_nodes(in_state):
            indices = self.residual_assignment.get_node_indices(in_state, node)
            if isinstance(node, LiteralValue):
                for i, idx in enumerate(indices):
                    res_stream[:, idx] = node.value[i]
            elif isinstance(node, InputNode):
                assert node.name in input_values
                value = input_values[node.name]
                for i, idx in enumerate(indices):
                    res_stream[:, idx] = value[:, i]
            elif isinstance(node, Concatenate):
                # Noop — children are guaranteed to be in the state individually.
                pass
            elif isinstance(node, Embedding):
                embedding_output = node.compute(n_pos, input_values)
                for i, idx in enumerate(indices):
                    res_stream[:, idx] = embedding_output[:, i]
            else:
                raise TypeError("Unsupported node type")
        return res_stream

    def forward(
        self, inp: torch.Tensor, *, return_states: bool = False
    ) -> torch.Tensor | _SublayerForwardStatesResult:
        res = inp
        all_states: _SublayerStates = {}
        for i, layer in enumerate(self.layers):
            sublayer_pairs: list[
                tuple[AttnSubLayer | MLPSubLayer | GatedMLPSubLayer, str]
            ] = [
                (layer.attn, "attn"),
                (layer.mlp, "mlp"),
            ]
            for sublayer, sublayer_name in sublayer_pairs:
                if return_states:
                    res, states = cast(
                        "_SublayerForwardStatesResult",
                        sublayer.forward(res, return_states=True),
                    )
                    prefixed_states = {
                        f"layer_{i}_{sublayer_name}_{key}": value
                        for key, value in states.items()
                    }
                    all_states.update(prefixed_states)
                else:
                    res = cast("torch.Tensor", sublayer.forward(res))
        if return_states:
            return res, all_states
        return res

    def forward_cached(
        self,
        inp: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with KV cache.

        Args:
            inp: (n_new, d) — new positions only
            past_kvs: None or list of (K, V) per layer

        Returns:
            (output, new_kvs) where output is (n_new, d)
        """
        if past_kvs is None:
            past_kvs = [None] * len(self.layers)

        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        res = inp
        for i, layer in enumerate(self.layers):
            res, kv = layer.attn.forward_cached(res, past_kvs[i])
            new_kvs.append(kv)
            res = cast("torch.Tensor", layer.mlp.forward(res))

        return res, new_kvs

    def compute(
        self, n_pos: int, input_values: dict[str, Any]
    ) -> dict[Node, torch.Tensor]:
        """Run the transformer on graph-level inputs.

        Returns a dict mapping each output Node to its value tensor
        of shape ``(n_pos, node.d_output)``.
        """
        assert self.residual_assignment

        in_stream = self.get_input_res_stream(n_pos, input_values).to(self.device)
        res = cast("torch.Tensor", self.forward(in_stream))
        result: dict[Node, torch.Tensor] = {}
        out_state = self.layers[-1].mlp.out_state

        for node in self.residual_assignment.get_nodes(out_state):
            indices = self.residual_assignment.get_node_indices(out_state, node)
            result[node] = res[:, indices]
        return result
