"""Continuous-input Hugging Face export and runtime support.

The exporter compiles the same residual-stream transformer used by
``HeadlessTransformer``.  A sidecar safetensors file stores the compiler-built
initial residual, including literal values and pinned normalization constants.
The runtime clones that tensor, overwrites named ``InputNode`` columns, runs the
decoder blocks, and reads named graph outputs from the raw final residual.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from safetensors.torch import load_file, save_file

from torchwright.compiler.forward.compile import (
    forward_compile,
    rms_norm_width_supported,
)
from torchwright.compiler.token_model import (
    CompileHeader,
    CompileProfile,
    ScheduleProvenance,
    make_layer_callback,
    resolve_rope,
    schedule_provenance,
)
from torchwright.compiler.utils import get_ancestor_nodes, resolve_n_heads
from torchwright.graph import Concatenate, Embedding, InputNode, Node

from .build import (
    _copy_custom_code,
    _DirectShardSink,
    _rope_proxy_layers,
    _staged_bundle_directory,
    _validate_staged_bundle,
)
from .configuration_torchwright_custom import TorchwrightCustomConfig
from .modeling_torchwright_custom import TorchwrightCustomForCausalLM

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

CONTINUOUS_IO_FORMAT = "torchwright_continuous_io_v1"
CONTINUOUS_IO_FILENAME = "continuous_io.json"
CONTINUOUS_BASE_FILENAME = "continuous_io.safetensors"
_BASE_RESIDUAL_KEY = "base_residual"
_UNBATCHED_NDIM = 2
_BATCHED_NDIM = 3


@dataclass(frozen=True)
class ContinuousValueSpec:
    """One named continuous value's shape and residual placement."""

    width: int
    shape: list[int]
    batched_shape: list[int | str]
    residual_columns: list[int]
    dtype: str = "float32"


@dataclass(frozen=True)
class ContinuousIOSpec:
    """Versioned continuous interface persisted beside an HF checkpoint."""

    format: str
    n_positions: int
    d_model: int
    inputs: dict[str, ContinuousValueSpec]
    outputs: dict[str, ContinuousValueSpec]
    base_residual: dict[str, str | list[int]]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON representation written into a bundle."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContinuousIOSpec:
        """Parse and validate a persisted interface specification."""
        if value.get("format") != CONTINUOUS_IO_FORMAT:
            raise ValueError(
                f"unsupported continuous I/O format: {value.get('format')!r}"
            )

        def values(section: str) -> dict[str, ContinuousValueSpec]:
            raw = value.get(section)
            if not isinstance(raw, dict):
                raise TypeError(f"continuous I/O {section} must be an object")
            parsed = {}
            for name, item in raw.items():
                if not isinstance(name, str) or not isinstance(item, dict):
                    raise TypeError(f"invalid continuous I/O {section} entry")
                parsed[name] = ContinuousValueSpec(
                    width=int(item["width"]),
                    shape=[int(dim) for dim in item["shape"]],
                    batched_shape=[
                        dim if isinstance(dim, str) else int(dim)
                        for dim in item["batched_shape"]
                    ],
                    residual_columns=[int(col) for col in item["residual_columns"]],
                    dtype=str(item["dtype"]),
                )
            return parsed

        base = value.get("base_residual")
        if not isinstance(base, dict):
            raise TypeError("continuous I/O base_residual must be an object")
        spec = cls(
            format=CONTINUOUS_IO_FORMAT,
            n_positions=int(value["n_positions"]),
            d_model=int(value["d_model"]),
            inputs=values("inputs"),
            outputs=values("outputs"),
            base_residual=cast("dict[str, str | list[int]]", dict(base)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        """Validate dimensions, dtypes, shapes, and residual columns."""
        if self.n_positions <= 0 or self.d_model <= 0:
            raise ValueError("continuous I/O dimensions must be positive")
        if not self.inputs or not self.outputs:
            raise ValueError("continuous I/O requires at least one input and output")
        for section in (self.inputs, self.outputs):
            for name, item in section.items():
                self._validate_value(name, item)

    def _validate_value(self, name: str, item: ContinuousValueSpec) -> None:
        if not name:
            raise ValueError("continuous I/O names must not be empty")
        if item.dtype != "float32":
            raise ValueError(f"unsupported continuous dtype {item.dtype!r}")
        if item.shape != [self.n_positions, item.width]:
            raise ValueError(f"invalid shape metadata for {name!r}")
        if item.batched_shape != ["batch", self.n_positions, item.width]:
            raise ValueError(f"invalid batched shape metadata for {name!r}")
        if len(item.residual_columns) != item.width:
            raise ValueError(f"invalid residual columns for {name!r}")
        if any(col < 0 or col >= self.d_model for col in item.residual_columns):
            raise ValueError(f"residual column out of range for {name!r}")


@dataclass(frozen=True)
class ContinuousHFBundleReport:
    """Small result describing a published continuous HF bundle."""

    output_dir: str | os.PathLike[str]
    n_layers: int
    n_positions: int
    schedule_provenance: ScheduleProvenance


def _named_inputs(outputs: Mapping[str, Node]) -> dict[str, InputNode]:
    found = [
        node
        for node in get_ancestor_nodes(set(outputs.values()))
        if isinstance(node, InputNode)
    ]
    named: dict[str, InputNode] = {}
    for node in sorted(found, key=lambda item: item.node_id):
        if not node.name:
            raise ValueError("continuous HF inputs must have non-empty names")
        if node.name in named and named[node.name] is not node:
            raise ValueError(f"continuous HF input name is duplicated: {node.name!r}")
        named[node.name] = node
    if not named:
        raise ValueError("continuous HF compilation requires at least one InputNode")
    return named


def _validate_outputs(outputs: Mapping[str, Node]) -> dict[str, Node]:
    if not outputs:
        raise ValueError("outputs must contain at least one named Node")
    validated = {}
    for name, node in outputs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("continuous HF output names must be non-empty strings")
        if not isinstance(node, Node):
            raise TypeError(f"continuous HF output {name!r} is not a Node")
        validated[name] = node
    ancestors = get_ancestor_nodes(set(validated.values()))
    if any(isinstance(node, Embedding) for node in ancestors):
        raise ValueError("continuous HF graphs cannot contain token Embedding nodes")
    return validated


def _value_spec(
    node: Node, columns: list[int], n_positions: int
) -> ContinuousValueSpec:
    return ContinuousValueSpec(
        width=node.d_output,
        shape=[n_positions, node.d_output],
        batched_shape=["batch", n_positions, node.d_output],
        residual_columns=list(columns),
    )


def _write_continuous_final_shard(
    output_dir: str | os.PathLike[str],
    *,
    d: int,
    n_layers: int,
    rms_gain: float | None,
    sink: _DirectShardSink,
) -> None:
    """Write structural embedding/norm weights and the shard index."""
    shard_count = n_layers + 1
    state_dict = {"model.embed_tokens.weight": torch.zeros((1, d))}
    if rms_gain is not None:
        gain = torch.full((d,), rms_gain)
        state_dict["model.norm.weight"] = gain
        for index in range(n_layers):
            prefix = f"model.layers.{index}"
            state_dict[f"{prefix}.input_layernorm.weight"] = gain.clone()
            state_dict[f"{prefix}.post_attention_layernorm.weight"] = gain.clone()
    filename = f"model-{shard_count:05d}-of-{shard_count:05d}.safetensors"
    save_file(state_dict, str(Path(output_dir) / filename))
    weight_map = dict(sink.weight_map)
    total_size = sink.total_size
    for name, tensor in state_dict.items():
        weight_map[name] = filename
        total_size += tensor.numel() * tensor.element_size()
    manifest = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (Path(output_dir) / "model.safetensors.index.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _validate_continuous_bundle(directory: str | os.PathLike[str]) -> None:
    path = Path(directory)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    expected_pointer = {
        "format": CONTINUOUS_IO_FORMAT,
        "spec": CONTINUOUS_IO_FILENAME,
        "base_residual": CONTINUOUS_BASE_FILENAME,
    }
    if config.get("torchwright_continuous_io") != expected_pointer:
        raise RuntimeError("config.json continuous I/O pointer is missing or invalid")
    spec = ContinuousIOSpec.from_dict(
        json.loads((path / CONTINUOUS_IO_FILENAME).read_text(encoding="utf-8"))
    )
    tensors = load_file(path / CONTINUOUS_BASE_FILENAME, device="cpu")
    base = tensors.get(_BASE_RESIDUAL_KEY)
    if base is None or list(base.shape) != [spec.n_positions, spec.d_model]:
        raise RuntimeError("continuous base residual shape does not match its spec")
    if base.dtype != torch.float32:
        raise RuntimeError("continuous base residual must be float32")


def compile_continuous_hf_bundle(
    outputs: Mapping[str, Node],
    output_dir: str | os.PathLike[str],
    *,
    n_positions: int,
    d: int = 1024,
    d_head: int = 16,
    n_heads: int | None = None,
    max_layers: int = 400,
    optimize: int = 0,
    d_hidden: int | None = None,
    trim_heads: bool = True,
    rms_norm: bool = False,
    rms_norm_eps: float = 1e-5,
    rms_norm_const_exp: int | None = None,
    machine: Literal["relu", "swish"] | None = None,
    verbose: bool = False,
    _solver_seed: int | None = None,
    _force_resolve: bool = False,
) -> ContinuousHFBundleReport:
    """Compile named tensor outputs into a continuous Hugging Face bundle.

    ``n_positions`` fixes the sequence dimension stored in the bundle. Runtime
    values may be unbatched ``(n_positions, width)`` tensors or consistently
    batched ``(batch, n_positions, width)`` tensors. ``machine`` may pin the
    custom biased-ReLU or biased-SwiGLU decoder and must then match the graph's
    op library. By default the compiler infers it from the graph (with ReLU as
    the no-FFN fallback). The source graph never requires a token embedding.
    """
    if n_positions <= 0:
        raise ValueError("n_positions must be positive")
    if rms_norm and not rms_norm_width_supported(d):
        raise ValueError(f"rms_norm is on but d={d} is unsupported")
    if machine not in (None, "relu", "swish"):
        raise ValueError(f"machine must be 'relu', 'swish', or None; got {machine!r}")
    outputs = _validate_outputs(outputs)
    inputs = _named_inputs(outputs)
    output_node = (
        next(iter(outputs.values()))
        if len(outputs) == 1
        else Concatenate(list(outputs.values()))
    )
    n_heads = resolve_n_heads(d, d_head, n_heads)

    with _staged_bundle_directory(output_dir) as staging:
        sink = _DirectShardSink(
            CompileProfile.CUSTOM, d_head, staging, capture_support=False
        )
        with torch.no_grad():
            compiled = forward_compile(
                d=d,
                d_head=d_head,
                n_heads=n_heads,
                output_node=output_node,
                verbose=verbose,
                max_layers=max_layers,
                device=None,
                on_layer_compiled=make_layer_callback(
                    CompileHeader(
                        d=d,
                        d_head=d_head,
                        trim_heads=trim_heads,
                        bias=True,
                        n_heads=n_heads,
                    ),
                    sink,
                ),
                trim_heads=trim_heads,
                optimize=optimize,
                bias=True,
                d_hidden=d_hidden,
                rms_norm=rms_norm,
                rms_norm_eps=rms_norm_eps,
                machine=machine,
                _solver_seed=_solver_seed,
                _force_resolve=_force_resolve,
                **cast(
                    "dict[str, Any]",
                    {}
                    if rms_norm_const_exp is None
                    else {"rms_norm_const_exp": rms_norm_const_exp},
                ),
            )
            assignment = compiled.residual_assignment
            if assignment is None or not compiled.layers:
                raise RuntimeError("continuous compilation produced no residual states")
            input_state = compiled.layers[0].attn.in_state
            output_state = compiled.layers[-1].mlp.out_state
            zero_inputs = {
                name: torch.zeros((n_positions, node.d_output))
                for name, node in inputs.items()
            }
            base_residual = compiled.get_input_res_stream(n_positions, zero_inputs)
            input_specs = {
                name: _value_spec(
                    node, assignment.get_node_indices(input_state, node), n_positions
                )
                for name, node in inputs.items()
            }
            output_specs = {
                name: _value_spec(
                    node, assignment.get_node_indices(output_state, node), n_positions
                )
                for name, node in outputs.items()
            }
            spec = ContinuousIOSpec(
                format=CONTINUOUS_IO_FORMAT,
                n_positions=n_positions,
                d_model=d,
                inputs=input_specs,
                outputs=output_specs,
                base_residual={
                    "file": CONTINUOUS_BASE_FILENAME,
                    "tensor": _BASE_RESIDUAL_KEY,
                    "dtype": "float32",
                    "shape": [n_positions, d],
                },
            )
            spec.validate()
            heads = [item[1] for item in sink.meta]
            hidden = [item[2] for item in sink.meta]
            rope_base, d_rot = resolve_rope(_rope_proxy_layers(sink.meta), d_head)
            rms_spec = compiled.rms_norm_spec
            config = TorchwrightCustomConfig(
                d=d,
                d_head=d_head,
                vocab_size=1,
                n_layers=len(sink.meta),
                n_heads_per_layer=heads,
                d_hidden_per_layer=hidden,
                max_position_embeddings=n_positions,
                rope_base=rope_base,
                d_rot=d_rot,
                rms_norm=rms_spec is not None,
                rms_norm_eps=rms_spec.eps if rms_spec is not None else rms_norm_eps,
                activation=cast("Literal['relu', 'swish']", compiled.activation),
                bos_token_id=None,
                eos_token_id=None,
            )
            config.architectures = ["TorchwrightCustomForCausalLM"]
            config.auto_map = {
                "AutoConfig": (
                    "configuration_torchwright_custom.TorchwrightCustomConfig"
                ),
                "AutoModelForCausalLM": (
                    "modeling_torchwright_custom.TorchwrightCustomForCausalLM"
                ),
            }
            config.torchwright_continuous_io = {
                "format": CONTINUOUS_IO_FORMAT,
                "spec": CONTINUOUS_IO_FILENAME,
                "base_residual": CONTINUOUS_BASE_FILENAME,
            }
            config.save_pretrained(staging)
            _write_continuous_final_shard(
                staging,
                d=d,
                n_layers=len(sink.meta),
                rms_gain=None if rms_spec is None else rms_spec.gain,
                sink=sink,
            )
            save_file(
                {_BASE_RESIDUAL_KEY: base_residual},
                Path(staging) / CONTINUOUS_BASE_FILENAME,
            )
            (Path(staging) / CONTINUOUS_IO_FILENAME).write_text(
                json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            provenance = schedule_provenance(compiled, optimize)
        _copy_custom_code(staging)
        _validate_staged_bundle(staging, expect_tokenizer=False)
        _validate_continuous_bundle(staging)

    return ContinuousHFBundleReport(
        output_dir=output_dir,
        n_layers=len(sink.meta),
        n_positions=n_positions,
        schedule_provenance=provenance,
    )


def _resolve_pretrained_path(path_or_repo_id: str | os.PathLike[str]) -> Path:
    path = Path(path_or_repo_id)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=os.fspath(path_or_repo_id)))


class ContinuousRunner:
    """Named tensor runtime for a compiled continuous HF bundle."""

    def __init__(
        self,
        model: TorchwrightCustomForCausalLM,
        spec: ContinuousIOSpec,
        base_residual: torch.Tensor,
    ) -> None:
        self.model = model
        self.spec = spec
        self.base_residual = base_residual

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | os.PathLike[str],
        *,
        device: str | torch.device | None = None,
    ) -> ContinuousRunner:
        """Load a local bundle or Hugging Face Hub repository."""
        path = _resolve_pretrained_path(path_or_repo_id)
        _validate_continuous_bundle(path)
        spec = ContinuousIOSpec.from_dict(
            json.loads((path / CONTINUOUS_IO_FILENAME).read_text(encoding="utf-8"))
        )
        base = load_file(path / CONTINUOUS_BASE_FILENAME, device="cpu")[
            _BASE_RESIDUAL_KEY
        ]
        model = TorchwrightCustomForCausalLM.from_pretrained(path).float().eval()
        if device is not None:
            model = model.to(device)
            base = base.to(device)
        return cls(model, spec, base)

    @property
    def device(self) -> torch.device:
        """Device holding the model and constructed residual streams."""
        return next(self.model.parameters()).device

    def _inputs(
        self, values: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], int, bool]:
        expected, actual = set(self.spec.inputs), set(values)
        if missing := sorted(expected - actual):
            raise ValueError(f"missing continuous inputs: {missing}")
        if unexpected := sorted(actual - expected):
            raise ValueError(f"unexpected continuous inputs: {unexpected}")
        normalized = {}
        batched: bool | None = None
        batch_size: int | None = None
        for name, item in self.spec.inputs.items():
            value = values[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"continuous input {name!r} must be a torch.Tensor")
            if value.ndim not in (_UNBATCHED_NDIM, _BATCHED_NDIM):
                raise ValueError(
                    f"continuous input {name!r} must have 2 or 3 dimensions"
                )
            this_batched = value.ndim == _BATCHED_NDIM
            if batched is not None and this_batched != batched:
                raise ValueError("continuous inputs must use consistent batching")
            batched = this_batched
            expected_tail = (self.spec.n_positions, item.width)
            if tuple(value.shape[-2:]) != expected_tail:
                raise ValueError(
                    f"continuous input {name!r} must end in shape {expected_tail}; "
                    f"got {tuple(value.shape)}"
                )
            if this_batched:
                if batch_size is not None and value.shape[0] != batch_size:
                    raise ValueError("continuous inputs must have one batch size")
                batch_size = value.shape[0]
            normalized[name] = value.to(device=self.device, dtype=torch.float32)
        return normalized, 1 if batch_size is None else batch_size, bool(batched)

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one fresh continuous state transition with no KV-cache history."""
        values, batch_size, batched = self._inputs(inputs)
        residual = (
            self.base_residual.to(self.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        for name, item in self.spec.inputs.items():
            value = values[name] if batched else values[name].unsqueeze(0)
            residual[..., item.residual_columns] = value
        with torch.no_grad():
            raw = self.model.forward_residual(inputs_embeds=residual, use_cache=False)
        outputs = {
            name: raw[..., item.residual_columns]
            for name, item in self.spec.outputs.items()
        }
        if batched:
            return outputs
        return {name: value.squeeze(0) for name, value in outputs.items()}

    def run_until(
        self,
        initial_state: torch.Tensor,
        *,
        state_input: str = "state",
        state_output: str = "state",
        stop_output: str = "converged",
        max_steps: int,
        stop_when: Callable[[torch.Tensor], bool] | None = None,
        **static_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply fresh transitions until the model's stop output is true.

        The default considers the model finished when every value in
        ``stop_output`` is greater than zero. ``stop_when`` can replace that
        reduction without moving application-specific numerical work into the
        recurrent loop. The returned mapping is the final invocation's output.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        state = initial_state
        latest: dict[str, torch.Tensor] | None = None
        predicate = stop_when or (lambda value: bool(torch.all(value > 0).item()))
        for _step in range(max_steps):
            latest = self(**{**static_inputs, state_input: state})
            state = latest[state_output]
            if predicate(latest[stop_output]):
                break
        assert latest is not None
        return latest


__all__ = [
    "CONTINUOUS_BASE_FILENAME",
    "CONTINUOUS_IO_FILENAME",
    "CONTINUOUS_IO_FORMAT",
    "ContinuousHFBundleReport",
    "ContinuousIOSpec",
    "ContinuousRunner",
    "ContinuousValueSpec",
    "compile_continuous_hf_bundle",
]
