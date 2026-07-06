from typing import Any, List, Dict, Optional, Tuple, Union

import torch

from torchwright.compiler.residual_assignment import ResidualAssignment
from torchwright.compiler.groups.attn_sublayer import AttnSubLayer
from torchwright.compiler.groups.mlp_sublayer import MLPSubLayer
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.graph import (
    Node,
    LiteralValue,
    InputNode,
    Concatenate,
    Embedding,
)


class HeadlessTransformer:
    """Stack of transformer layers without embedding or unembedding heads.

    Produced by the forward compiler. Use ``compute()`` to run the
    transformer on graph-level inputs and retrieve output node values.
    """

    layers: List[TransformerLayer]
    d: int
    d_hidden: int
    d_head: int
    residual_assignment: Optional[ResidualAssignment]
    # 2-D weight-matrix occupancy recorded during weight writing (see
    # forward.weight_writer.PlacementRecorder).  ``None`` until compile sets it.
    placements: Optional[Any]

    def __init__(
        self,
        d: int,
        d_head: int,
        d_hidden: Optional[int] = None,
        activation: str = "relu",
    ):
        if activation not in ("relu", "swish"):
            raise ValueError(
                f"HeadlessTransformer activation must be 'relu' or 'swish', "
                f"got {activation!r}"
            )
        self.d = d
        self.d_hidden = d if d_hidden is None else d_hidden
        self.d_head = d_head
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

    def to(self, device) -> "HeadlessTransformer":
        for layer in self.layers:
            layer.to(device)
        return self

    def add_layer(self, append: bool = False) -> TransformerLayer:
        layer = TransformerLayer(
            self.d, self.d_head, d_hidden=self.d_hidden, activation=self.activation
        )
        if append:
            self.layers.append(layer)
        else:
            self.layers = [layer] + self.layers
        return layer

    def get_input_res_stream(
        self,
        n_pos: int,
        input_values: Dict[str, Any],
        past_len: int = 0,
    ):
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
                assert False, "Unsupported node type"
        return res_stream

    def forward(self, inp: torch.Tensor, return_states=False):
        res = inp
        all_states = {}
        for i, layer in enumerate(self.layers):
            sublayer_pairs: List[tuple[Union[AttnSubLayer, MLPSubLayer], str]] = [
                (layer.attn, "attn"),
                (layer.mlp, "mlp"),
            ]
            for sublayer, sublayer_name in sublayer_pairs:
                if return_states:
                    res, states = sublayer.forward(res, return_states=True)
                    prefixed_states = {
                        f"layer_{i}_{sublayer_name}_{key}": value
                        for key, value in states.items()
                    }
                    all_states.update(prefixed_states)
                else:
                    res = sublayer.forward(res)
        if return_states:
            return res, all_states
        else:
            return res

    def forward_cached(self, inp: torch.Tensor, past_kvs=None):
        """Forward pass with KV cache.

        Args:
            inp: (n_new, d) — new positions only
            past_kvs: None or list of (K, V) per layer

        Returns:
            (output, new_kvs) where output is (n_new, d)
        """
        if past_kvs is None:
            past_kvs = [None] * len(self.layers)

        new_kvs = []
        res = inp
        for i, layer in enumerate(self.layers):
            res, kv = layer.attn.forward_cached(res, past_kvs[i])
            new_kvs.append(kv)
            res = layer.mlp.forward(res)

        return res, new_kvs

    def compute(
        self, n_pos: int, input_values: Dict[str, Any]
    ) -> Dict[Node, torch.Tensor]:
        """Run the transformer on graph-level inputs.

        Returns a dict mapping each output Node to its value tensor
        of shape ``(n_pos, node.d_output)``.
        """
        assert self.residual_assignment

        res = self.forward(
            self.get_input_res_stream(n_pos, input_values).to(self.device)
        )
        result = {}
        out_state = self.layers[-1].mlp.out_state

        for node in self.residual_assignment.get_nodes(out_state):
            indices = self.residual_assignment.get_node_indices(out_state, node)
            result[node] = res[:, indices]
        return result
