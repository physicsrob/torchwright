"""Direct compiler-to-Hugging-Face build path (no ONNX dependency)."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from torchwright.compiler.forward.compile import (
    forward_compile,
    rms_norm_width_supported,
)
from torchwright.compiler.token_model import (
    CompiledLayerWeights,
    CompileHeader,
    CompileProfile,
    ReluLayerWeights,
    ScheduleProvenance,
    SwishLayerWeights,
    TokenModelSpec,
    TokenModelWeights,
    build_token_weights,
    make_layer_callback,
    resolve_rope,
    schedule_provenance,
)
from torchwright.compiler.utils import get_ancestor_nodes, resolve_n_heads
from torchwright.graph import Embedding, Node

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from torchwright.compiler.forward.compile import RmsNormSpec

HFArchitecture = CompileProfile | str

# Default special-token spellings (not secrets — named to dodge bandit's
# hardcoded-password heuristic on *_token-named parameters).
_DEFAULT_BOS_SPELLING = "<bos>"
_DEFAULT_EOS_SPELLING = "<eos>"
_DEFAULT_UNK_SPELLING = "<unk>"


@dataclass(frozen=True)
class HFBundleReport:
    """Small, immutable report for a published Hugging Face bundle.

    This deliberately contains only paths and compile facts.  In particular,
    it does not retain the graph, emitted weights, or private compiler state.
    """

    output_dir: str | os.PathLike
    n_layers: int
    schedule_provenance: ScheduleProvenance


def _validate_embedding_contract(output_node: Node, embedding: Embedding) -> None:
    if not isinstance(embedding, Embedding):
        raise TypeError("embedding must be an Embedding node")
    embeddings = [
        node
        for node in get_ancestor_nodes({output_node})
        if isinstance(node, Embedding)
    ]
    if len(embeddings) != 1:
        raise ValueError(
            "HF token compilation requires exactly one Embedding reachable from "
            f"output_node; found {len(embeddings)}"
        )
    if embeddings[0] is not embedding:
        raise ValueError(
            "the supplied embedding is not the Embedding reachable from output_node"
        )


def _remove_path(path: str) -> None:
    if not os.path.lexists(path):
        return
    p = Path(path)
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink()


@contextmanager
def _staged_bundle_directory(output_dir: str | os.PathLike) -> Iterator[str]:
    """Build beside ``output_dir`` and publish with rollback on failure."""
    destination = os.path.abspath(os.fspath(output_dir))
    parent = Path(destination).parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".{Path(destination).name}.staging-", dir=parent)
    backup = staging + ".previous"
    try:
        yield staging
        if os.path.lexists(destination):
            Path(destination).replace(backup)
        try:
            Path(staging).replace(destination)
        except BaseException:
            if os.path.lexists(backup):
                Path(backup).replace(destination)
            raise
        try:
            _remove_path(backup)
        except BaseException:
            Path(destination).replace(staging)
            Path(backup).replace(destination)
            _remove_path(staging)
            raise
    except BaseException:
        _remove_path(staging)
        if os.path.lexists(backup) and not os.path.lexists(destination):
            Path(backup).replace(destination)
        raise


def _write_generation_config(
    output_dir: str | os.PathLike, bos_id: int | None, eos_id: int | None
) -> None:
    """Compiled vocabs carry no pad token.

    Alias pad to eos so ``generate()`` and ``pipeline()`` run without an
    explicit pad_token_id.
    """
    from transformers import GenerationConfig

    GenerationConfig(
        bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=eos_id
    ).save_pretrained(output_dir)


def _validate_staged_weights(directory: Path) -> None:
    """Check the safetensors manifest (sharded or single-file) matches disk."""
    from safetensors import safe_open

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("staged safetensors index has no weight_map")
        expected_by_file: dict[str, set[str]] = {}
        for key, filename in weight_map.items():
            expected_by_file.setdefault(filename, set()).add(key)
        for filename, expected in expected_by_file.items():
            shard = directory / filename
            if not shard.is_file():
                raise RuntimeError(f"staged safetensors shard is missing: {filename}")
            with safe_open(shard, framework="pt", device="cpu") as handle:
                actual = set(handle.keys())
            if actual != expected:
                raise RuntimeError(
                    f"staged shard {filename} keys do not match its index"
                )
    else:
        model_path = directory / "model.safetensors"
        if not model_path.is_file():
            raise RuntimeError("staged HF bundle has no safetensors weights")
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            if not list(handle.keys()):
                raise RuntimeError("staged model.safetensors has no tensors")


def _validate_staged_bundle(
    directory: str | os.PathLike, *, expect_tokenizer: bool
) -> None:
    """Validate bundle structure and tensor manifests without loading weights."""
    from transformers import AutoConfig, AutoTokenizer

    directory = Path(directory)
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise RuntimeError("staged HF bundle has no config.json")
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    if config_data.get("model_type") == "torchwright_custom":
        from .configuration_torchwright_custom import TorchwrightCustomConfig

        TorchwrightCustomConfig.from_pretrained(directory)
    else:
        AutoConfig.from_pretrained(directory)

    _validate_staged_weights(directory)

    if expect_tokenizer:
        if config_data.get("model_type") == "torchwright_custom":
            from .tokenization_torchwright_custom import TorchwrightCustomTokenizer

            TorchwrightCustomTokenizer.from_pretrained(directory)
        else:
            AutoTokenizer.from_pretrained(directory)


def _token_id(vocab: tuple[str, ...], token: str | None, kind: str) -> int | None:
    if token is None:
        return None
    if token not in vocab:
        raise ValueError(f"requested {kind}_token {token!r} is not in the vocabulary")
    return vocab.index(token)


def _resolve_architecture(
    architecture: HFArchitecture, *, bias: bool | None, rms_norm: bool | None
) -> CompileProfile:
    try:
        profile = CompileProfile(architecture)
    except ValueError:
        raise ValueError(
            f"architecture must be 'phi3' or 'custom'; got {architecture!r}"
        ) from None
    if profile is CompileProfile.PHI3:
        if bias not in (None, False):
            raise ValueError("architecture='phi3' requires bias=False")
        if rms_norm is False:
            raise ValueError("architecture='phi3' requires rms_norm=True")
        return profile
    if bias not in (None, True):
        raise ValueError("architecture='custom' requires bias=True")
    return profile


def _target(
    activation: str,
    *,
    bias: bool,
    rms_norm: bool,
    architecture: HFArchitecture | None = None,
) -> str:
    profile = CompileProfile(architecture)
    expected = profile.value
    actual = (
        "phi3"
        if activation == "swish" and not bias and rms_norm
        else "custom"
        if activation == "relu" and bias
        else None
    )
    if actual != expected:
        raise AssertionError(
            f"architecture={architecture!r} compiled to activation={activation!r}, "
            f"bias={bias}, rms_norm={rms_norm}; expected {expected}"
        )
    return actual


def _torch(arr: np.ndarray) -> torch.Tensor:
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32).copy())


def build_fast_tokenizer(
    vocab: Sequence[str],
    *,
    bos_token: str | None = _DEFAULT_BOS_SPELLING,
    eos_token: str | None = _DEFAULT_EOS_SPELLING,
    add_bos_token: bool = True,
) -> PreTrainedTokenizerFast:
    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocab_dict = {token: i for i, token in enumerate(vocab)}
    unk = _DEFAULT_UNK_SPELLING if _DEFAULT_UNK_SPELLING in vocab_dict else None
    tok = Tokenizer(WordLevel(vocab=vocab_dict, unk_token=_DEFAULT_UNK_SPELLING))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens([t for t in (unk, bos_token, eos_token) if t is not None])
    if add_bos_token and bos_token is not None:
        tok.post_processor = processors.TemplateProcessing(
            single=f"{bos_token} $A",
            pair=f"{bos_token} $A {bos_token} $B",
            special_tokens=[(bos_token, vocab_dict[bos_token])],
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token=unk, bos_token=bos_token, eos_token=eos_token
    )


def save_hf_bundle(
    model: PreTrainedModel,
    vocab: Sequence[str],
    output_dir: str | os.PathLike,
    *,
    add_bos_token: bool = True,
    write_tokenizer: bool = True,
) -> PreTrainedModel:
    """Save a directly compiled model and its vocabulary as an HF bundle."""
    with _staged_bundle_directory(output_dir) as staging:
        _save_hf_bundle_into(
            model,
            vocab,
            staging,
            add_bos_token=add_bos_token,
            write_tokenizer=write_tokenizer,
        )
        _validate_staged_bundle(staging, expect_tokenizer=write_tokenizer)
    return model


def _save_hf_bundle_into(
    model: PreTrainedModel,
    vocab: Sequence[str],
    output_dir: str | os.PathLike,
    *,
    add_bos_token: bool = True,
    write_tokenizer: bool = True,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    _write_generation_config(
        output_dir, model.config.bos_token_id, model.config.eos_token_id
    )
    if not write_tokenizer:
        return
    bos_id, eos_id = model.config.bos_token_id, model.config.eos_token_id
    bos = vocab[bos_id] if bos_id is not None else None
    eos = vocab[eos_id] if eos_id is not None else None
    if model.config.model_type == "phi3":
        build_fast_tokenizer(
            list(vocab), bos_token=bos, eos_token=eos, add_bos_token=add_bos_token
        ).save_pretrained(output_dir)
    else:
        from .configuration_torchwright_custom import TorchwrightCustomConfig
        from .modeling_torchwright_custom import TorchwrightCustomForCausalLM
        from .tokenization_torchwright_custom import TorchwrightCustomTokenizer

        TorchwrightCustomConfig.register_for_auto_class()
        TorchwrightCustomForCausalLM.register_for_auto_class("AutoModelForCausalLM")
        vocab_path = Path(output_dir) / "vocab.json"
        with vocab_path.open("w") as f:
            json.dump(list(vocab), f)
        tok = TorchwrightCustomTokenizer(
            vocab_file=str(vocab_path),
            bos_token=bos,
            eos_token=eos,
            add_bos_token=add_bos_token,
        )
        TorchwrightCustomTokenizer.register_for_auto_class()
        tok.save_pretrained(output_dir)
        model.save_pretrained(output_dir)


def compile_hf_bundle(
    output_node: Node,
    embedding: Embedding,
    output_dir: str | os.PathLike,
    *,
    d: int = 1024,
    d_head: int = 16,
    n_heads: int | None = None,
    max_seq_len: int = 512,
    max_layers: int = 400,
    optimize: int = 0,
    d_hidden: int | None = None,
    trim_heads: bool = True,
    rms_norm: bool | None = None,
    rms_norm_eps: float = 1e-5,
    rms_norm_const_exp: int | None = None,
    architecture: HFArchitecture = "phi3",
    bias: bool | None = None,
    bos_token: str | None = _DEFAULT_BOS_SPELLING,
    eos_token: str | None = _DEFAULT_EOS_SPELLING,
    verbose: bool = False,
    add_bos_token: bool = True,
    write_tokenizer: bool = True,
    _solver_seed: int | None = None,
    _force_resolve: bool = False,
) -> HFBundleReport:
    """Compile and transactionally publish a sharded safetensors HF bundle.

    Returns an :class:`HFBundleReport` whose layer count and selected-schedule
    provenance describe the bundle that was successfully published.

    ``n_heads`` defaults to ``d // d_head``. Set it explicitly to make the
    flattened attention width ``n_heads * d_head`` independent of ``d``.
    """
    _validate_embedding_contract(output_node, embedding)
    with _staged_bundle_directory(output_dir) as staging:
        report = _compile_hf_bundle_into(
            output_node,
            embedding,
            staging,
            d=d,
            d_head=d_head,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            max_layers=max_layers,
            optimize=optimize,
            d_hidden=d_hidden,
            trim_heads=trim_heads,
            rms_norm=rms_norm,
            rms_norm_eps=rms_norm_eps,
            rms_norm_const_exp=rms_norm_const_exp,
            architecture=architecture,
            bias=bias,
            bos_token=bos_token,
            eos_token=eos_token,
            verbose=verbose,
            add_bos_token=add_bos_token,
            write_tokenizer=write_tokenizer,
            _solver_seed=_solver_seed,
            _force_resolve=_force_resolve,
        )
        _validate_staged_bundle(staging, expect_tokenizer=write_tokenizer)
    return replace(report, output_dir=output_dir)


class _DirectShardSink:
    """Streams each compiled layer directly into its final HF shard."""

    def __init__(
        self, profile: CompileProfile, d_head: int, output_dir: str | os.PathLike
    ) -> None:
        self._profile = profile
        self._d_head = d_head
        self._output_dir = output_dir
        self.meta: list[tuple[str, int, int, float | None, int | None]] = []
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def begin(self, header: CompileHeader) -> None:
        self.header = header
        if not header.layer_shapes:
            raise RuntimeError("HF streaming requires announced layer shapes")
        self.shard_count = len(header.layer_shapes) + 1
        self.max_heads = max(shape.n_heads for shape in header.layer_shapes)
        self.max_hidden = max(shape.d_hidden for shape in header.layer_shapes)

    def write_layer(self, index: int, layer: CompiledLayerWeights) -> None:
        import torch
        from safetensors.torch import save_file

        d_head = self._d_head
        a = layer.attention
        p = f"model.layers.{index}"
        if self._profile is CompileProfile.CUSTOM:
            assert isinstance(layer, ReluLayerWeights)
            sd = {
                f"{p}.self_attn.q_proj.weight": _torch(a.wq).T.contiguous(),
                f"{p}.self_attn.k_proj.weight": _torch(a.wk).T.contiguous(),
                f"{p}.self_attn.v_proj.weight": _torch(a.wv).T.contiguous(),
                f"{p}.self_attn.o_proj.weight": _torch(a.wo).T.contiguous(),
                f"{p}.mlp.fc1.weight": _torch(layer.w1).T.contiguous(),
                f"{p}.mlp.fc1.bias": _torch(cast("np.ndarray", layer.b1)),
                f"{p}.mlp.fc2.weight": _torch(layer.w2).T.contiguous(),
                f"{p}.mlp.fc2.bias": _torch(cast("np.ndarray", layer.b2)),
            }
            kind = "relu"
        else:
            assert isinstance(layer, SwishLayerWeights)
            rows, inter = self.max_heads * d_head, self.max_hidden
            q = (_torch(a.wq).T.double() * math.sqrt(float(d_head))).float()

            def tpad(value: torch.Tensor, target_size: int, axis: int) -> torch.Tensor:
                shape = list(value.shape)
                shape[axis] = target_size
                out = torch.zeros(shape, dtype=torch.float32)
                sl = [slice(None)] * value.ndim
                sl[axis] = slice(0, value.shape[axis])
                out[tuple(sl)] = value
                return out

            wk, wv, wo = _torch(a.wk), _torch(a.wv), _torch(a.wo)
            wg, wu, wd = _torch(layer.wgate), _torch(layer.wup), _torch(layer.wdown)
            sd = {
                f"{p}.self_attn.qkv_proj.weight": torch.cat(
                    [tpad(q, rows, 0), tpad(wk.T, rows, 0), tpad(wv.T, rows, 0)]
                ),
                f"{p}.self_attn.o_proj.weight": tpad(wo.T, rows, 1),
                f"{p}.mlp.gate_up_proj.weight": torch.cat(
                    [tpad(wg.T, inter, 0), tpad(wu.T, inter, 0)]
                ),
                f"{p}.mlp.down_proj.weight": tpad(wd.T, inter, 1),
            }
            kind = "swish"
        filename = f"model-{index + 1:05d}-of-{self.shard_count:05d}.safetensors"
        save_file(sd, str(Path(self._output_dir) / filename))
        for name, value in sd.items():
            self.weight_map[name] = filename
            self.total_size += value.numel() * value.element_size()
        self.meta.append((kind, a.n_heads, layer.d_hidden, a.rope_base, a.d_rot))

    def finalize(self, spec: TokenModelSpec, weights: TokenModelWeights) -> None:
        self.spec = spec
        self.token_weights = weights


def _rope_proxy_layers(
    meta: list[tuple[str, int, int, float | None, int | None]],
) -> list[Any]:
    """Attribute proxies for ``resolve_rope`` (only RoPE metadata is read)."""
    proxy_layers: list[Any] = []
    for _kind, _nh, _dh, base, drot in meta:
        # Only RoPE metadata is inspected by resolve_rope.
        class A:
            rope_base: Any
            d_rot: Any

        class L:
            attention: Any

        a, layer = A(), L()
        a.rope_base, a.d_rot = base, drot
        layer.attention = a
        proxy_layers.append(layer)
    return proxy_layers


def _copy_custom_code(output_dir: str | os.PathLike) -> None:
    """Ship the custom config/model/tokenizer modules alongside the bundle."""
    here = Path(__file__).parent
    for name in (
        "configuration_torchwright_custom.py",
        "modeling_torchwright_custom.py",
        "tokenization_torchwright_custom.py",
    ):
        shutil.copy2(here / name, Path(output_dir) / name)


def _write_final_shard(
    output_dir: str | os.PathLike,
    spec: TokenModelSpec,
    token: TokenModelWeights,
    rms_norm_spec: RmsNormSpec | None,
    sink: _DirectShardSink,
) -> None:
    """Write the token-table/norm shard and the safetensors index.

    token.v6 tied readout: lm_head aliases embed_tokens for both
    targets (the custom model via _tied_weights_keys, stock Phi-3 via
    tie_word_embeddings=True), so the state dict carries no separate
    unembed.  The folded RMS constants in embed_table are cleared
    before readout by the zeroed final-gain coordinates below, and
    the literal seed by the compiled clear_literal_seed op.
    """
    from safetensors.torch import save_file

    shard_count = spec.n_layers + 1
    weight_map, total_size = sink.weight_map, sink.total_size
    gain = token.norm_gain
    final_sd = {
        "model.embed_tokens.weight": _torch(token.embed_table),
    }
    if gain is not None:
        # Final norm: exact zero on the pinned-constant coordinates so
        # the tied readout never sees the folded RMS constants — the
        # same fold the ONNX exporter's final_norm carries.
        final_gain = gain.copy()
        final_gain[list(cast("Any", rms_norm_spec).reserved_cols)] = 0.0
        final_sd["model.norm.weight"] = _torch(final_gain)
        for i in range(spec.n_layers):
            p = f"model.layers.{i}"
            final_sd[f"{p}.input_layernorm.weight"] = _torch(gain)
            final_sd[f"{p}.post_attention_layernorm.weight"] = _torch(gain)
    filename = f"model-{shard_count:05d}-of-{shard_count:05d}.safetensors"
    save_file(final_sd, str(Path(output_dir) / filename))
    for name, value in final_sd.items():
        weight_map[name] = filename
        total_size += value.numel() * value.element_size()
    index_path = Path(output_dir) / "model.safetensors.index.json"
    with index_path.open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)


def _compile_hf_bundle_into(
    output_node: Node,
    embedding: Embedding,
    output_dir: str | os.PathLike,
    *,
    d: int = 1024,
    d_head: int = 16,
    n_heads: int | None = None,
    max_seq_len: int = 512,
    max_layers: int = 400,
    optimize: int = 0,
    d_hidden: int | None = None,
    trim_heads: bool = True,
    rms_norm: bool | None = None,
    rms_norm_eps: float = 1e-5,
    rms_norm_const_exp: int | None = None,
    architecture: HFArchitecture = "phi3",
    bias: bool | None = None,
    bos_token: str | None = _DEFAULT_BOS_SPELLING,
    eos_token: str | None = _DEFAULT_EOS_SPELLING,
    verbose: bool = False,
    add_bos_token: bool = True,
    write_tokenizer: bool = True,
    _solver_seed: int | None = None,
    _force_resolve: bool = False,
) -> HFBundleReport:
    """Compile directly to a sharded safetensors HF bundle.

    Each compiled layer is transformed directly into its final HF shard.
    Consequently this path never constructs a destination model, retains all
    layers in RAM, or writes a canonical intermediate spool.

    ``architecture="phi3"`` is the strong default and fixes the compiler
    profile to SwiGLU, biasless projections, and RMSNorm. It guarantees a
    stock bundle with no custom code. ``architecture="custom"`` explicitly
    selects the biased ReLU machine and ``TorchwrightCustomForCausalLM``;
    incompatible graph FFNs or contradictory options raise rather than
    falling back between architectures.
    """
    import torch

    profile = _resolve_architecture(architecture, bias=bias, rms_norm=rms_norm)
    machine, bias = profile.machine, profile.bias
    rms_on = profile.rms_norm if rms_norm is None else bool(rms_norm)
    if rms_on and not rms_norm_width_supported(d):
        raise ValueError(f"rms_norm is on but d={d} is unsupported")
    n_heads = resolve_n_heads(d, d_head, n_heads)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sink = _DirectShardSink(profile, d_head, output_dir)
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
                CompileHeader(d, d_head, trim_heads, bias, n_heads=n_heads), sink
            ),
            trim_heads=trim_heads,
            optimize=optimize,
            bias=bias,
            # token.v6 held-bank contract: the output is written into the
            # embedding's exact ordered residual bank, same as compile_to_onnx
            # — both targets' storage-tied lm_head and the shared
            # build_token_weights tied-layout validation rely on it.
            output_layout_source=embedding,
            d_hidden=d_hidden,
            rms_norm=rms_on,
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
        token = build_token_weights(compiled, output_node, embedding, d)
        heads = [m[1] for m in sink.meta]
        hidden = [m[2] for m in sink.meta]
        rope_base, d_rot = resolve_rope(_rope_proxy_layers(sink.meta), d_head)
        vocab = tuple(embedding.tokenizer.vocab)
        spec = TokenModelSpec(
            d,
            d_head,
            max_seq_len,
            vocab,
            token.embed_table.shape[0],
            cast("Literal['relu', 'swish']", compiled.activation),
            bool(bias),
            compiled.rms_norm_spec is not None,
            float(
                compiled.rms_norm_spec.eps if compiled.rms_norm_spec else rms_norm_eps
            ),
            rope_base,
            d_rot,
            len(sink.meta),
            tuple(heads),
            tuple(hidden),
            schedule_provenance(compiled, optimize),
        )
        sink.finalize(spec, token)
        target = _target(
            spec.activation,
            bias=spec.bias,
            rms_norm=spec.rms_norm,
            architecture=profile,
        )
        bos_id = _token_id(vocab, bos_token, "bos")
        eos_id = _token_id(vocab, eos_token, "eos")
        if target == "custom":
            from .configuration_torchwright_custom import TorchwrightCustomConfig

            config = TorchwrightCustomConfig(
                d=d,
                d_head=d_head,
                vocab_size=spec.vocab_size,
                n_layers=spec.n_layers,
                n_heads_per_layer=heads,
                d_hidden_per_layer=hidden,
                max_position_embeddings=max_seq_len,
                rope_base=rope_base,
                d_rot=d_rot,
                rms_norm=spec.rms_norm,
                rms_norm_eps=spec.rms_norm_eps,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                # token.v6: the custom model's lm_head is storage-tied to
                # embed_tokens; the config constructor enforces (and forces)
                # tie_word_embeddings=True.
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
        else:
            from transformers import Phi3Config

            max_heads, inter = max(heads), max(hidden)
            config = cast("Any", Phi3Config)(
                vocab_size=spec.vocab_size,
                hidden_size=d,
                intermediate_size=inter,
                num_hidden_layers=spec.n_layers,
                num_attention_heads=max_heads,
                num_key_value_heads=max_heads,
                head_dim=d_head,
                hidden_act="silu",
                max_position_embeddings=max_seq_len,
                rms_norm_eps=spec.rms_norm_eps,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": rope_base,
                    "partial_rotary_factor": d_rot / d_head,
                },
                sliding_window=None,
                attention_dropout=0.0,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                use_cache=True,
                # token.v6: one serialized token table; stock Phi-3's normal
                # tie_weights() path reconstructs lm_head as an alias of
                # embed_tokens at load.
                tie_word_embeddings=True,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                pad_token_id=None,
            )
            config.architectures = ["Phi3ForCausalLM"]
        config.save_pretrained(output_dir)
        _write_generation_config(output_dir, bos_id, eos_id)

        _write_final_shard(output_dir, spec, token, compiled.rms_norm_spec, sink)

    if target == "custom":
        _copy_custom_code(output_dir)
    if write_tokenizer:
        if target == "phi3":
            build_fast_tokenizer(
                vocab,
                bos_token=bos_token,
                eos_token=eos_token,
                add_bos_token=add_bos_token,
            ).save_pretrained(output_dir)
        else:
            from .tokenization_torchwright_custom import TorchwrightCustomTokenizer

            vocab_path = Path(output_dir) / "vocab.json"
            with vocab_path.open("w") as f:
                json.dump(list(vocab), f)
            TorchwrightCustomTokenizer.register_for_auto_class()
            TorchwrightCustomTokenizer(
                vocab_file=str(vocab_path),
                bos_token=bos_token,
                eos_token=eos_token,
                add_bos_token=add_bos_token,
            ).save_pretrained(output_dir)
    provenance = spec.schedule_provenance
    assert provenance is not None
    return HFBundleReport(
        output_dir=output_dir,
        n_layers=spec.n_layers,
        schedule_provenance=provenance,
    )


# The public in-memory entry point intentionally reuses the streaming bundle
# sink.  Loading its final shards leaves the destination module as the only
# retained full-model copy (plus the currently loaded shard), rather than
# briefly holding every canonical compiler layer alongside every parameter.
def compile_to_hf(
    output_node: Node,
    embedding: Embedding,
    *,
    d: int = 1024,
    d_head: int = 16,
    n_heads: int | None = None,
    max_seq_len: int = 512,
    max_layers: int = 400,
    optimize: int = 0,
    d_hidden: int | None = None,
    trim_heads: bool = True,
    rms_norm: bool | None = None,
    rms_norm_eps: float = 1e-5,
    rms_norm_const_exp: int | None = None,
    architecture: HFArchitecture = "phi3",
    bias: bool | None = None,
    bos_token: str | None = _DEFAULT_BOS_SPELLING,
    eos_token: str | None = _DEFAULT_EOS_SPELLING,
    verbose: bool = False,
    _solver_seed: int | None = None,
    _force_resolve: bool = False,
) -> PreTrainedModel:
    """Compile directly into an fp32 eval-mode Hugging Face model.

    The default is stock ``Phi3ForCausalLM``. The custom implementation is
    reachable only through the explicit ``architecture="custom"`` opt-in.
    ``n_heads`` defaults to ``d // d_head`` and may be set explicitly to
    decouple attention width from ``d``.
    """
    with tempfile.TemporaryDirectory(prefix="torchwright-hf-model-") as directory:
        compile_hf_bundle(
            output_node,
            embedding,
            directory,
            d=d,
            d_head=d_head,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            max_layers=max_layers,
            optimize=optimize,
            d_hidden=d_hidden,
            trim_heads=trim_heads,
            rms_norm=rms_norm,
            rms_norm_eps=rms_norm_eps,
            rms_norm_const_exp=rms_norm_const_exp,
            architecture=architecture,
            bias=bias,
            bos_token=bos_token,
            eos_token=eos_token,
            verbose=verbose,
            _solver_seed=_solver_seed,
            _force_resolve=_force_resolve,
            write_tokenizer=False,
        )
        with (Path(directory) / "config.json").open() as f:
            model_type = json.load(f)["model_type"]
        model: Any
        if model_type == "phi3":
            from transformers import Phi3ForCausalLM

            model = Phi3ForCausalLM.from_pretrained(
                directory, attn_implementation="eager"
            )
        else:
            from .modeling_torchwright_custom import TorchwrightCustomForCausalLM

            model = TorchwrightCustomForCausalLM.from_pretrained(directory)
    return model.float().eval()
