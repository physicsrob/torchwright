"""Backend-neutral streaming records for compiled token transformers.

This module deliberately imports neither ONNX nor transformers.  It owns the
canonical ``x @ W`` compiler layout and the token embedding/output folds used
by every artifact sink.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Callable, Literal, Mapping, Optional, Protocol, TypeAlias, Union

import numpy as np

from torchwright.graph import Concatenate, Embedding, LiteralValue, Node
from torchwright.graph.misc import InputNode
from torchwright.graph.rope import ROPE_BASE

JSONScalar: TypeAlias = Union[None, bool, int, float, str]
JSONValue: TypeAlias = Union[JSONScalar, list["JSONValue"], dict[str, "JSONValue"]]


class CompileProfile(str, Enum):
    """Named physical transformer contracts shared by every backend."""

    PHI3 = "phi3"
    CUSTOM = "custom"

    @property
    def machine(self) -> Literal["swish", "relu"]:
        return "swish" if self is CompileProfile.PHI3 else "relu"

    @property
    def bias(self) -> bool:
        return self is CompileProfile.CUSTOM

    @property
    def rms_norm(self) -> bool:
        return True


@dataclass(frozen=True)
class ScheduleProvenance:
    optimize: int
    selected_origin: Optional[str] = None
    delivery: Optional[str] = None
    selected_objective: Optional[int] = None
    selected_objective_blocks: Optional[tuple[int, int]] = None
    selected_is_optimal: Optional[bool] = None
    solver_status: Optional[str] = None
    solver_objective: Optional[float] = None
    solver_best_bound: Optional[float] = None
    solver_is_optimal: Optional[bool] = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class LayerShape:
    n_heads: int
    d_hidden: int


@dataclass(frozen=True)
class CompileHeader:
    d: int
    d_head: int
    trim_heads: bool
    bias: bool
    layer_shapes: tuple[LayerShape, ...] = ()
    n_heads: Optional[int] = None


@dataclass
class AttentionWeights:
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    n_heads: int
    rope_base: Optional[float]
    d_rot: Optional[int]


@dataclass
class ReluLayerWeights:
    attention: AttentionWeights
    w1: np.ndarray
    b1: Optional[np.ndarray]
    w2: np.ndarray
    b2: Optional[np.ndarray]

    @property
    def d_hidden(self) -> int:
        return self.w1.shape[1]


@dataclass
class SwishLayerWeights:
    attention: AttentionWeights
    wgate: np.ndarray
    bgate: Optional[np.ndarray]
    wup: np.ndarray
    bup: Optional[np.ndarray]
    wdown: np.ndarray
    bdown: Optional[np.ndarray]

    @property
    def d_hidden(self) -> int:
        return self.wgate.shape[1]


CompiledLayerWeights = Union[ReluLayerWeights, SwishLayerWeights]


@dataclass(frozen=True)
class TokenModelSpec:
    d: int
    d_head: int
    max_seq_len: int
    vocab: tuple[str, ...]
    vocab_size: int
    activation: Literal["relu", "swish"]
    bias: bool
    rms_norm: bool
    rms_norm_eps: float
    rope_base: float
    d_rot: int
    n_layers: int
    per_layer_n_heads: tuple[int, ...]
    per_layer_d_hidden: tuple[int, ...]
    schedule_provenance: Optional[ScheduleProvenance] = None
    extra_metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenModelWeights:
    embed_table: np.ndarray
    lm_head: np.ndarray
    norm_gain: Optional[np.ndarray]


class TokenModelSink(Protocol):
    def begin(self, header: CompileHeader) -> None: ...
    def write_layer(self, index: int, weights: CompiledLayerWeights) -> None: ...
    def finalize(self, spec: TokenModelSpec, weights: TokenModelWeights) -> None: ...


def make_layer_callback(header: CompileHeader, sink: TokenModelSink) -> Callable:
    """Return a ``forward_compile`` callback which transfers layer ownership."""
    begun = False

    def on_replay_plan(plan) -> None:
        nonlocal header, begun
        header = replace(
            header, layer_shapes=tuple(layer.shape for layer in plan.layers)
        )
        sink.begin(header)
        begun = True

    def callback(index, layer):
        nonlocal begun
        if not begun:
            sink.begin(header)
            begun = True
        attn, mlp = layer.attn.attn, layer.mlp
        if header.trim_heads:
            attn.trim_unused_heads()
            mlp.trim_unused_slots()
        nh, hd = attn.n_heads, attn.n_heads * header.d_head
        if header.layer_shapes:
            actual = LayerShape(nh, mlp.d_hidden)
            expected = header.layer_shapes[index]
            assert actual == expected, (
                f"layer {index} shape diverged from schedule plan: "
                f"announced {expected}, emitted {actual}"
            )

        def array(t):
            return t.detach().contiguous().cpu().numpy().astype(np.float32, copy=False)

        aw = AttentionWeights(
            array(attn.query_matrix.permute(1, 0, 2).reshape(header.d, hd)),
            array(attn.key_matrix.permute(1, 0, 2).reshape(header.d, hd)),
            array(attn.value_matrix.permute(1, 0, 2).reshape(header.d, hd)),
            array(attn.output_matrix.reshape(hd, header.d)),
            nh,
            getattr(attn, "rope_base", None),
            getattr(attn, "rope_d_rot", None),
        )
        attn.query_matrix = attn.key_matrix = attn.value_matrix = None
        attn.output_matrix = None

        def linear(lin):
            matrix = array(lin.output_matrix)
            if header.bias:
                bias_value = array(lin.output_bias)
            else:
                assert (lin.output_bias == 0.0).all(), "nonzero folded bias"
                bias_value = None
            lin.output_matrix = lin.output_bias = None
            return matrix, bias_value

        if getattr(mlp, "activation", "relu") == "swish":
            wg, bg = linear(mlp.gate_proj)
            wu, bu = linear(mlp.up_proj)
            wd, bd = linear(mlp.down_proj)
            record = SwishLayerWeights(aw, wg, bg, wu, bu, wd, bd)
        else:
            w1, b1 = linear(mlp.linear1)
            w2, b2 = linear(mlp.linear2)
            record = ReluLayerWeights(aw, w1, b1, w2, b2)
        sink.write_layer(index, record)

    callback.token_model_sink = sink
    callback.on_replay_plan = on_replay_plan
    return callback


def resolve_rope(layers: list[CompiledLayerWeights], d_head: int) -> tuple[float, int]:
    bases = {w.attention.rope_base for w in layers if w.attention.rope_base is not None}
    widths = {w.attention.d_rot for w in layers if w.attention.d_rot is not None}
    if len(bases) > 1 or len(widths) > 1:
        raise NotImplementedError("token compilation requires one global RoPE grid")
    return float(next(iter(bases), ROPE_BASE)), int(next(iter(widths), d_head))


def schedule_provenance(compiled, optimize: int) -> ScheduleProvenance:
    result = getattr(compiled, "schedule_result", None)
    if result is None:
        return ScheduleProvenance(optimize=int(optimize))
    selected = result.provenance
    stats = selected.solver_attempt
    return ScheduleProvenance(
        optimize=int(optimize),
        selected_origin=selected.origin,
        delivery=selected.delivery,
        selected_objective=selected.selected_objective,
        selected_objective_blocks=selected.selected_objective_blocks,
        selected_is_optimal=selected.selected_is_optimal,
        solver_status=stats.status_name if stats is not None else None,
        solver_objective=stats.objective_value if stats is not None else None,
        solver_best_bound=(stats.best_objective_bound if stats is not None else None),
        solver_is_optimal=stats.is_optimal if stats is not None else None,
    )


def build_token_weights(compiled, output_node: Node, embedding: Embedding, d: int):
    """Fold token placement, literals and RMS constants into full-width weights."""
    assignment = compiled.residual_assignment
    if assignment is None or not compiled.layers:
        raise ValueError("compiled model has no residual assignment or layers")
    in_state = compiled.layers[0].attn.in_state
    out_state = compiled.layers[-1].mlp.out_state
    embedding_indices = None
    literal_seeds = []
    for node in assignment.get_nodes(in_state):
        indices = assignment.get_node_indices(in_state, node)
        if isinstance(node, Embedding):
            if node is embedding:
                embedding_indices = indices
            else:
                raise ValueError(
                    "compiled residual assignment contains an Embedding other than "
                    "the supplied embedding"
                )
        elif isinstance(node, LiteralValue):
            literal_seeds.extend(zip(indices, map(float, node.value)))
        elif not isinstance(node, (Concatenate, InputNode)):
            raise AssertionError(f"unexpected input-state node {type(node).__name__}")
    if embedding_indices is None:
        raise ValueError("supplied embedding is absent from the residual assignment")
    compact = embedding.table.detach().cpu().numpy().astype(np.float32, copy=False)
    vocab_size = compact.shape[0]
    if compact.shape[1] != len(embedding_indices):
        raise ValueError(
            "embedding width does not match its compiled residual placement"
        )
    embed = np.zeros((vocab_size, d), dtype=np.float32)
    embed[:, embedding_indices] = compact
    rms = compiled.rms_norm_spec
    if rms is not None:
        for col, value in zip(rms.reserved_cols, rms.const_values):
            embed[:, col] = value
    for col, value in literal_seeds:
        embed[:, col] = value
    output_indices = assignment.get_node_indices(out_state, output_node)
    if list(output_indices) != list(embedding_indices):
        raise ValueError(
            "tied token layout violated: the output must occupy the "
            "embedding's exact ordered residual bank (token.v6 held handoff); "
            f"embedding={list(embedding_indices)[:8]}..., "
            f"output={list(output_indices)[:8]}..."
        )
    # The compact untied projection over the bank — the stock-architecture HF
    # target's genuine lm_head.  The tied surfaces (v6 ONNX readout, the
    # custom HF model) transpose/share embed_table itself and ignore this.
    lm_head = np.zeros((vocab_size, d), dtype=np.float32)
    lm_head[:, output_indices] = compact
    gain = np.full(d, rms.gain, dtype=np.float32) if rms is not None else None
    return TokenModelWeights(embed, lm_head, gain)
