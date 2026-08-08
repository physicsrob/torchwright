"""Direct compiler-to-Hugging-Face build path (no ONNX dependency)."""

from __future__ import annotations

import hashlib
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
from torchwright.compiler.graph_identity import canonical_ids
from torchwright.compiler.token_model import (
    CompiledLayerWeights,
    CompileHeader,
    CompileProfile,
    JSONValue,
    ReluLayerWeights,
    ScheduleProvenance,
    SwishLayerWeights,
    TokenModelSpec,
    TokenModelWeights,
    build_token_weights,
    make_layer_callback,
    resolve_rope,
    schedule_provenance,
    validate_token_vocab,
)
from torchwright.compiler.truth import (
    TRUTH_FILENAME,
    TRUTH_FORMAT,
    TRUTH_SCHEMA_FILENAME,
    TRUTH_SUPPORT_FILENAME,
    column_runs,
    sha256_json,
)
from torchwright.compiler.utils import get_ancestor_nodes, resolve_n_heads
from torchwright.graph import Embedding, Node

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from torchwright.compiler.forward.compile import RmsNormSpec
    from torchwright.compiler.transformer import HeadlessTransformer

HFArchitecture = CompileProfile | str

# Default special-token spellings (not secrets — named to dodge bandit's
# hardcoded-password heuristic on *_token-named parameters).
_DEFAULT_BOS_SPELLING = "<bos>"
_DEFAULT_EOS_SPELLING = "<eos>"
_DEFAULT_UNK_SPELLING = "<unk>"
_MATRIX_NDIM = 2
_RUN_FIELDS = 2
#: The packaged schema: copied into every bundle and the source of the
#: validator's required-section list, so the shipped schema cannot drift
#: from what validation enforces.
_TRUTH_SCHEMA_SOURCE = Path(__file__).parent.parent / TRUTH_SCHEMA_FILENAME


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
    validate_token_vocab(embedding.tokenizer.vocab)


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
    directory: str | os.PathLike,
    *,
    expect_tokenizer: bool,
    expect_truth: bool = False,
    staged_file_hashes: Mapping[str, dict[str, Any]] | None = None,
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
    if expect_truth:
        _validate_truth_manifest(directory, staged_file_hashes=staged_file_hashes)


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

    vocab = list(vocab)
    validate_token_vocab(vocab)
    vocab_dict = {token: i for i, token in enumerate(vocab)}
    unk = _DEFAULT_UNK_SPELLING
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
    validate_token_vocab(list(vocab))
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
    # This path writes no truth files, so the config must not advertise
    # them: a model loaded from a truth-bearing bundle carries the pointer
    # as a config attribute, and save_pretrained would re-serialize it into
    # a bundle where the referenced files do not exist.
    if hasattr(model.config, "torchwright_truth"):
        del model.config.torchwright_truth
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
    truth_metadata: Mapping[str, JSONValue] | None = None,
    _solver_seed: int | None = None,
    _force_resolve: bool = False,
) -> HFBundleReport:
    """Compile and transactionally publish a sharded safetensors HF bundle.

    Returns an :class:`HFBundleReport` whose layer count and selected-schedule
    provenance describe the bundle that was successfully published.

    ``n_heads`` defaults to ``d // d_head``. Set it explicitly to make the
    flattened attention width ``n_heads * d_head`` independent of ``d``.
    ``truth_metadata`` is optional JSON metadata for source/task facts known by
    the caller; it cannot replace compiler-owned truth sections.
    """
    _validate_embedding_contract(output_node, embedding)
    with _staged_bundle_directory(output_dir) as staging:
        report, truth_files = _compile_hf_bundle_into(
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
            truth_metadata=truth_metadata,
            _solver_seed=_solver_seed,
            _force_resolve=_force_resolve,
        )
        _validate_staged_bundle(
            staging,
            expect_tokenizer=write_tokenizer,
            expect_truth=True,
            staged_file_hashes=truth_files,
        )
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
        self.support_array_paths: dict[str, Path] = {}
        self.support_tensors: dict[str, dict[str, Any]] = {}

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
        self.record_support(sd)
        save_file(sd, str(Path(self._output_dir) / filename))
        for name, value in sd.items():
            self.weight_map[name] = filename
            self.total_size += value.numel() * value.element_size()
        self.meta.append((kind, a.n_heads, layer.d_hidden, a.rope_base, a.d_rot))

    def record_support(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Capture exact nonzero coordinates without materializing a full mask."""
        import torch

        max_chunk_elements = 1 << 20
        for name, value in state_dict.items():
            prefix = f"tensor_{len(self.support_tensors):06d}"
            shape = list(value.shape)
            if value.ndim == _MATRIX_NDIM:
                rows, cols = value.shape
                rows_per_chunk = max(1, max_chunk_elements // max(int(cols), 1))
                nnz = 0
                chunks: list[dict[str, Any]] = []
                for start in range(0, int(rows), rows_per_chunk):
                    end = min(start + rows_per_chunk, int(rows))
                    coords = torch.nonzero(value[start:end], as_tuple=False)
                    counts = (
                        torch.bincount(coords[:, 0], minlength=end - start)
                        .cpu()
                        .numpy()
                    )
                    indptr = np.concatenate(
                        [
                            np.zeros(1, dtype=np.int64),
                            np.cumsum(counts, dtype=np.int64),
                        ]
                    )
                    indices = coords[:, 1].to(dtype=torch.int32).cpu().numpy()
                    chunk_index = len(chunks)
                    indptr_key = f"{prefix}_chunk_{chunk_index:06d}_indptr"
                    indices_key = f"{prefix}_chunk_{chunk_index:06d}_indices"
                    self._stage_support_array(indptr_key, indptr)
                    self._stage_support_array(indices_key, indices)
                    chunk_nnz = len(indices)
                    nnz += chunk_nnz
                    chunks.append(
                        {
                            "row_start": start,
                            "row_count": end - start,
                            "nnz": chunk_nnz,
                            "indptr": indptr_key,
                            "indices": indices_key,
                        }
                    )
                support = {"encoding": "csr_row_chunks", "chunks": chunks}
            else:
                flat = value.reshape(-1)
                indices = (
                    torch.nonzero(flat, as_tuple=False)
                    .reshape(-1)
                    .to(dtype=torch.int64)
                    .cpu()
                    .numpy()
                )
                indices_key = f"{prefix}_indices"
                self._stage_support_array(indices_key, indices)
                nnz = len(indices)
                support = {"encoding": "flat_indices", "indices": indices_key}
            self.support_tensors[name] = {
                "shape": shape,
                "dtype": str(value.dtype).removeprefix("torch."),
                "nnz": int(nnz),
                **support,
            }

    def finalize(self, spec: TokenModelSpec, weights: TokenModelWeights) -> None:
        """Satisfy the backend-neutral sink protocol; HF finalizes explicitly."""
        self.spec = spec
        self.token_weights = weights

    def _stage_support_array(self, key: str, value: np.ndarray) -> None:
        path = Path(self._output_dir) / f".{key}.npy"
        np.save(path, value, allow_pickle=False)
        self.support_array_paths[key] = path

    def write_support(self) -> None:
        arrays = {
            key: np.load(path, mmap_mode="r", allow_pickle=False)
            for key, path in self.support_array_paths.items()
        }
        try:
            np.savez_compressed(
                Path(self._output_dir) / TRUTH_SUPPORT_FILENAME,
                **arrays,
            )
        finally:
            for value in arrays.values():
                mmap = getattr(value, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            for path in self.support_array_paths.values():
                path.unlink()


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
    sink.record_support(final_sd)
    save_file(final_sd, str(Path(output_dir) / filename))
    for name, value in final_sd.items():
        weight_map[name] = filename
        total_size += value.numel() * value.element_size()
    index_path = Path(output_dir) / "model.safetensors.index.json"
    with index_path.open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_parameter_map(
    profile: CompileProfile,
    spec: TokenModelSpec,
    sink: _DirectShardSink,
) -> list[dict[str, Any]]:
    """Map logical compiler matrices onto exact checkpoint tensor slices."""
    mappings = []
    stored_heads = max(spec.per_layer_n_heads)
    stored_hidden = max(spec.per_layer_d_hidden)
    stored_rows = stored_heads * spec.d_head

    def add(
        logical: str,
        tensor: str,
        rows: tuple[int, int],
        columns: tuple[int, int],
        *,
        scale: float | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "logical_matrix": logical,
            "checkpoint_tensor": tensor,
            "checkpoint_rows": list(rows),
            "checkpoint_columns": list(columns),
            "transform": "transpose",
        }
        if scale is not None:
            record["scale"] = scale
        mappings.append(record)

    for index, (active_heads, active_hidden) in enumerate(
        zip(
            spec.per_layer_n_heads,
            spec.per_layer_d_hidden,
            strict=True,
        )
    ):
        prefix = f"model.layers.{index}"
        active_rows = active_heads * spec.d_head
        if profile is CompileProfile.PHI3:
            qkv = f"{prefix}.self_attn.qkv_proj.weight"
            add(
                f"L{index}.attn.W_Q",
                qkv,
                (0, active_rows),
                (0, spec.d),
                scale=math.sqrt(float(spec.d_head)),
            )
            add(
                f"L{index}.attn.W_K",
                qkv,
                (stored_rows, active_rows),
                (0, spec.d),
            )
            add(
                f"L{index}.attn.W_V",
                qkv,
                (2 * stored_rows, active_rows),
                (0, spec.d),
            )
            add(
                f"L{index}.attn.W_O",
                f"{prefix}.self_attn.o_proj.weight",
                (0, spec.d),
                (0, active_rows),
            )
            gate_up = f"{prefix}.mlp.gate_up_proj.weight"
            add(
                f"L{index}.mlp.W_in",
                gate_up,
                (0, active_hidden),
                (0, spec.d),
            )
            add(
                f"L{index}.mlp.W_up",
                gate_up,
                (stored_hidden, active_hidden),
                (0, spec.d),
            )
            add(
                f"L{index}.mlp.W_out",
                f"{prefix}.mlp.down_proj.weight",
                (0, spec.d),
                (0, active_hidden),
            )
        else:
            for logical, target in (
                ("W_Q", "q_proj"),
                ("W_K", "k_proj"),
                ("W_V", "v_proj"),
            ):
                add(
                    f"L{index}.attn.{logical}",
                    f"{prefix}.self_attn.{target}.weight",
                    (0, active_rows),
                    (0, spec.d),
                )
            add(
                f"L{index}.attn.W_O",
                f"{prefix}.self_attn.o_proj.weight",
                (0, spec.d),
                (0, active_rows),
            )
            add(
                f"L{index}.mlp.W_in",
                f"{prefix}.mlp.fc1.weight",
                (0, active_hidden),
                (0, spec.d),
            )
            add(
                f"L{index}.mlp.W_out",
                f"{prefix}.mlp.fc2.weight",
                (0, spec.d),
                (0, active_hidden),
            )
            mappings.extend(
                [
                    {
                        "logical_matrix": f"L{index}.mlp.W_in_bias",
                        "checkpoint_tensor": f"{prefix}.mlp.fc1.bias",
                        "checkpoint_indices": [0, active_hidden],
                        "transform": "identity",
                    },
                    {
                        "logical_matrix": f"L{index}.mlp.W_out_bias",
                        "checkpoint_tensor": f"{prefix}.mlp.fc2.bias",
                        "checkpoint_indices": [0, spec.d],
                        "transform": "identity",
                    },
                ]
            )
    mapped_tensors = {record["checkpoint_tensor"] for record in mappings}
    missing = sorted(set(sink.weight_map) - mapped_tensors)
    mappings.extend(
        [
            {
                "logical_matrix": None,
                "checkpoint_tensor": tensor,
                "transform": "runtime_or_token_parameter",
            }
            for tensor in missing
        ]
    )
    return mappings


def _known_null_components(
    capture: dict[str, Any], spec: TokenModelSpec
) -> dict[str, Any]:
    stored_heads = max(spec.per_layer_n_heads)
    stored_hidden = max(spec.per_layer_d_hidden)
    padded_heads = []
    padded_neurons = []
    for index, (heads, hidden) in enumerate(
        zip(
            spec.per_layer_n_heads,
            spec.per_layer_d_hidden,
            strict=True,
        )
    ):
        if heads < stored_heads:
            padded_heads.append(
                {"layer": index, "head_range": [heads, stored_heads - heads]}
            )
        if hidden < stored_hidden:
            padded_neurons.append(
                {"layer": index, "neuron_range": [hidden, stored_hidden - hidden]}
            )

    used_columns = {
        column
        for state in capture["residual_stream"]["states"]
        for runs in state["nodes"].values()
        for start, length in runs
        for column in range(start, start + length)
    }
    column_fields = (
        "target_columns",
        "source_columns",
        "source_columns_b",
        "query_source_columns",
        "key_source_columns",
    )
    for layer in capture["schedule"]["layers"]:
        for operation in [
            *layer["attention_operations"],
            *layer["mlp_operations"],
        ]:
            used_columns.update(
                column
                for field in column_fields
                if operation.get(field) is not None
                for start, length in operation[field]
                for column in range(start, start + length)
            )
    used_columns.add(capture["residual_stream"]["constant_one_column"])
    used_columns.update(capture["residual_stream"]["rms_norm_reserved_columns"])
    unused = sorted(set(range(spec.d)) - used_columns)
    return {
        "padded_attention_heads": padded_heads,
        "padded_mlp_neurons": padded_neurons,
        "never_referenced_residual_columns": column_runs(unused),
        "meaning": (
            "These components are statically absent or padding. This is not an "
            "input-specific causal-relevance claim."
        ),
    }


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("torchwright")
    except PackageNotFoundError:
        return "unknown"


def _write_truth_manifest(
    output_dir: str | os.PathLike,
    *,
    capture: dict[str, Any],
    profile: CompileProfile,
    spec: TokenModelSpec,
    sink: _DirectShardSink,
    compiled: HeadlessTransformer,
    output_node: Node,
    embedding: Embedding,
    bos_id: int | None,
    eos_id: int | None,
    metadata: Mapping[str, JSONValue] | None,
) -> dict[str, dict[str, Any]]:
    """Write the manifest and return its artifact file-hash records."""
    directory = Path(output_dir)
    shutil.copy2(_TRUTH_SCHEMA_SOURCE, directory / TRUTH_SCHEMA_FILENAME)
    sink.write_support()

    # The capture dict is owned by compiled.truth_capture; the HF-specific
    # sections below must not leak into it, so copy every level this
    # function assigns into.
    capture = {
        **capture,
        "build": {**capture["build"]},
        "physical_layout": {**capture["physical_layout"]},
    }

    assignment = compiled.residual_assignment
    if assignment is None:
        raise RuntimeError("truth emission requires a residual assignment")
    input_state = compiled.layers[0].attn.in_state
    output_state = compiled.layers[-1].mlp.out_state
    embedding_columns = assignment.get_node_indices(input_state, embedding)
    output_columns = assignment.get_node_indices(output_state, output_node)

    # The capture's source refs are canonical ids of output_node's graph;
    # recomputing them here resolves the embedding node exactly, whatever
    # its concrete class (Embedding subclasses are valid entry contracts).
    embedding_ref = f"s:{canonical_ids(output_node)[embedding.node_id]}"
    source_embedding = next(
        (
            node
            for node in capture["graphs"]["source"]["nodes"]
            if node["id"] == embedding_ref
        ),
        None,
    )
    if source_embedding is None:
        raise RuntimeError("truth capture does not contain the embedding node")
    provenance = spec.schedule_provenance
    capture["build"].update(
        {
            "torchwright_version": _package_version(),
            "schedule_provenance": ({} if provenance is None else provenance.to_dict()),
        }
    )
    capture["model"] = {
        "architecture": profile.value,
        "d_model": spec.d,
        "d_head": spec.d_head,
        "n_layers": spec.n_layers,
        "stored_attention_heads": max(spec.per_layer_n_heads),
        "active_attention_heads_per_layer": list(spec.per_layer_n_heads),
        "stored_mlp_neurons": max(spec.per_layer_d_hidden),
        "active_mlp_neurons_per_layer": list(spec.per_layer_d_hidden),
        "activation": spec.activation,
        "bias": spec.bias,
        "rms_norm": spec.rms_norm,
        "rms_norm_eps": spec.rms_norm_eps,
        "rope_base": spec.rope_base,
        "rotary_width": spec.d_rot,
        "max_sequence_length": spec.max_seq_len,
    }
    capture["token_io"] = {
        "source_embedding_node": source_embedding["id"],
        "vocabulary_size": spec.vocab_size,
        "vocabulary_sha256": sha256_json(list(spec.vocab)),
        "unknown_token_id": spec.vocab.index(_DEFAULT_UNK_SPELLING),
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "embedding_residual_columns": column_runs(list(embedding_columns)),
        "output_residual_columns": column_runs(list(output_columns)),
        "readout": {
            "kind": "tied_embedding_dot_product",
            "checkpoint_tensor": "model.embed_tokens.weight",
            "input_state": "output",
            "final_normalization": "rms_norm" if spec.rms_norm else None,
            "formula": ("logits[token] = final_normalize(residual) @ embedding[token]"),
        },
    }
    capture["physical_layout"]["checkpoint_parameter_map"] = _checkpoint_parameter_map(
        profile, spec, sink
    )
    capture["physical_layout"]["known_null_components"] = _known_null_components(
        capture, spec
    )
    capture["parameter_support"] = {
        "file": TRUTH_SUPPORT_FILENAME,
        "format": "torchwright.csr_support.v1",
        "zero_test": "stored fp32 value != 0.0",
        "tensors": sink.support_tensors,
    }
    residual_hooks = [
        {"state": "input", "hook": "blocks.0.hook_resid_pre"},
        *[
            {"state": f"L{index}.post", "hook": f"blocks.{index}.hook_resid_post"}
            for index in range(spec.n_layers)
        ],
    ]
    capture["observability"] = {
        "abstract_residual_states": residual_hooks,
        "transformer_lens": {
            "residual_states": residual_hooks,
            "attention": {
                "q": "blocks.{layer}.attn.hook_q",
                "k": "blocks.{layer}.attn.hook_k",
                "v": "blocks.{layer}.attn.hook_v",
                "pattern": "blocks.{layer}.attn.hook_pattern",
                "result": "blocks.{layer}.attn.hook_result",
                "head_axis": 2,
            },
            "mlp": {
                "pre": "blocks.{layer}.mlp.hook_pre",
                "post": "blocks.{layer}.mlp.hook_post",
                "neuron_axis": -1,
            },
            "note": (
                "Use schedule.layers head and hidden-slot ranges to slice the "
                "standard hooks. Hook availability depends on the adapter version."
            ),
        },
    }
    capture["runtime_contract"] = {
        "dtype": "float32",
        "causal_attention": True,
        "generation": {"do_sample": False, "greedy": True},
        "recommended_attention_implementation_for_analysis": "eager",
        "unsupported_inferences": [
            "Static graph ancestry is not dynamic causal relevance.",
            "Unused-on-one-input activations require an input-specific trace.",
            "Other numerical precisions are outside the exactness contract.",
        ],
    }
    capture["reference_fixtures"] = {
        "included": [],
        "reason": "No input corpus was supplied to compile_hf_bundle.",
        "scope": "Input-specific activations and intervention results.",
    }
    capture["metadata"] = dict(metadata or {})
    json.dumps(capture["metadata"], allow_nan=False)

    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != TRUTH_FILENAME:
            files[path.name] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    capture["artifact"] = {
        "kind": "huggingface_causal_lm_bundle",
        "architecture": profile.value,
        "files": files,
    }
    capture["parameter_support"]["sha256"] = files[TRUTH_SUPPORT_FILENAME]["sha256"]
    # Computed last so it binds every other section: any in-place edit to
    # the manifest fails validation, not just edits to the two graph
    # records with their own content hashes.
    capture["integrity"] = {
        "scope": "every manifest section except this one",
        "sha256": sha256_json(
            {key: value for key, value in capture.items() if key != "integrity"}
        ),
    }
    truth_path = directory / TRUTH_FILENAME
    truth_path.write_text(
        json.dumps(capture, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return files


def _validate_truth_bound_files(
    directory: Path,
    files: dict[str, dict[str, Any]],
    *,
    staged_file_hashes: Mapping[str, dict[str, Any]] | None = None,
) -> None:
    """Check every truth-bound artifact file against its manifest record.

    ``staged_file_hashes`` is the hash map ``_write_truth_manifest`` computed
    moments earlier in the same process; when a file's entry is present
    there, its digest substitutes for re-reading multi-GB shards.  External
    validation passes nothing and always re-hashes from disk.
    """
    for name, record in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise RuntimeError(f"truth manifest has unsafe artifact path {name!r}")
        path = directory / relative
        if not path.is_file():
            raise RuntimeError(f"truth-bound artifact is missing: {name}")
        if path.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"truth-bound artifact size mismatch: {name}")
        if staged_file_hashes is not None and name in staged_file_hashes:
            actual = staged_file_hashes[name]["sha256"]
        else:
            actual = _sha256_file(path)
        if actual != record.get("sha256"):
            raise RuntimeError(f"truth-bound artifact hash mismatch: {name}")


def _validate_csr_support(
    *,
    name: str,
    shape: list[int],
    nnz: int,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> None:
    invalid = (
        len(shape) != _MATRIX_NDIM
        or len(indptr) != shape[0] + 1
        or indptr[0] != 0
        or indptr[-1] != nnz
        or bool((np.diff(indptr) < 0).any())
        or (len(indices) and (indices.min() < 0 or indices.max() >= shape[1]))
    )
    if invalid:
        raise RuntimeError(f"malformed CSR support for {name}")


def _validate_support_tensor(
    name: str,
    record: dict[str, Any],
    arrays: np.lib.npyio.NpzFile,
    available: set[str],
) -> None:
    shape = record.get("shape")
    nnz = record.get("nnz")
    if not isinstance(shape, list) or not isinstance(nnz, int):
        raise TypeError(f"invalid support metadata for {name}")
    if record.get("encoding") == "flat_indices":
        indices_key = record.get("indices")
        if indices_key not in available or len(arrays[indices_key]) != nnz:
            raise RuntimeError(f"invalid support indices for {name}")
        indices = arrays[indices_key]
        size = math.prod(shape)
        if len(indices) and (indices.min() < 0 or indices.max() >= size):
            raise RuntimeError(f"out-of-bounds flat support for {name}")
        return
    chunks = record.get("chunks")
    if record.get("encoding") != "csr_row_chunks" or not isinstance(chunks, list):
        raise RuntimeError(f"unknown support encoding for {name}")
    expected_row = total_nnz = 0
    for chunk in chunks:
        indptr_key = chunk.get("indptr")
        indices_key = chunk.get("indices")
        if indptr_key not in available or indices_key not in available:
            raise RuntimeError(f"missing support chunk for {name}")
        if chunk.get("row_start") != expected_row:
            raise RuntimeError(f"non-contiguous support chunks for {name}")
        chunk_rows, chunk_nnz = chunk["row_count"], chunk["nnz"]
        _validate_csr_support(
            name=name,
            shape=[chunk_rows, shape[1]],
            nnz=chunk_nnz,
            indptr=arrays[indptr_key],
            indices=arrays[indices_key],
        )
        expected_row += chunk_rows
        total_nnz += chunk_nnz
    if expected_row != shape[0] or total_nnz != nnz:
        raise RuntimeError(f"incomplete support chunks for {name}")


def _validate_support_records(directory: Path, support: dict[str, Any]) -> None:
    filename = support.get("file")
    if not isinstance(filename, str) or not filename:
        raise RuntimeError("truth manifest parameter_support names no file")
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise RuntimeError(f"truth manifest has unsafe support path {filename!r}")
    support_path = directory / relative
    if not support_path.is_file():
        raise RuntimeError(f"truth support file is missing: {filename}")
    with np.load(support_path, allow_pickle=False) as arrays:
        available = set(arrays.files)
        for name, record in support.get("tensors", {}).items():
            _validate_support_tensor(name, record, arrays, available)


def _validate_runs(
    value: object, limit: int, label: str, *, min_length: int = 1
) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list of [start, length] runs")
    for run in value:
        if (
            not isinstance(run, list)
            or len(run) != _RUN_FIELDS
            or not all(isinstance(part, int) for part in run)
        ):
            raise TypeError(f"{label} has a malformed run")
        start, length = run
        if start < 0 or length < min_length or start + length > limit:
            raise RuntimeError(f"{label} run {run} exceeds width {limit}")


def _validate_graph_record(graph: dict[str, Any], prefix: str) -> dict[str, int]:
    label = "source" if prefix == "s" else "lowered"
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise TypeError(f"{label} graph nodes must be a list")
    widths: dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise TypeError(f"{label} graph node must be an object")
        node_id, width = node.get("id"), node.get("width")
        if not isinstance(node_id, str) or not isinstance(width, int):
            raise TypeError(f"{label} graph node has an invalid ID or width")
        widths[node_id] = width
    if len(widths) != len(nodes) or any(
        not node.startswith(f"{prefix}:") for node in widths
    ):
        raise RuntimeError(f"{label} graph has duplicate or invalid node IDs")
    if graph.get("root") not in widths:
        raise RuntimeError(f"{label} graph root does not resolve")
    hashed = {key: value for key, value in graph.items() if key != "sha256"}
    if graph.get("sha256") != sha256_json(hashed):
        raise RuntimeError(f"{label} graph hash mismatch")
    for node in nodes:
        if any(input_id not in widths for input_id in node.get("inputs", [])):
            raise RuntimeError(f"{label} graph has an unresolved input reference")
    return widths


def _validate_lowering_map(
    records: object, source_widths: dict[str, int], lowered_widths: dict[str, int]
) -> None:
    if not isinstance(records, list) or len(records) != len(source_widths):
        raise RuntimeError("truth lowering map does not cover the source graph")
    if {record.get("source") for record in records} != set(source_widths):
        raise RuntimeError("truth lowering map has invalid source references")
    for record in records:
        status = record.get("status")
        if status == "not_materialized":
            continue
        lowered_id = record.get("lowered")
        if lowered_id not in lowered_widths:
            raise RuntimeError("truth lowering map has an invalid lowered reference")
        if status == "whole":
            continue
        sliced = record.get("slice")
        if status != "slice" or not isinstance(sliced, dict):
            raise RuntimeError("truth lowering map has an invalid status")
        offset, width = sliced.get("offset"), sliced.get("width")
        if (
            not isinstance(offset, int)
            or not isinstance(width, int)
            or offset < 0
            or width < 1
            or offset + width > lowered_widths[lowered_id]
        ):
            raise RuntimeError("truth lowering slice exceeds its holder")


def _validate_attention_operations(
    layer: dict[str, Any], physical_ids: set[str], d_model: int
) -> None:
    for operation in layer.get("attention_operations", []):
        if operation.get("node") not in physical_ids | {None}:
            raise RuntimeError("attention operation has an unresolved node")
        # A zero-length span is legitimate here: a zero-support
        # attention-routed op emits no post-trim heads (its writer-allocated
        # floor head is trimmed), so its span is [cursor, 0].
        _validate_runs(
            [operation.get("heads")],
            layer["active_attention_heads"],
            "attention heads",
            min_length=0,
        )
        for field in (
            "target_columns",
            "source_columns",
            "source_columns_b",
            "query_source_columns",
            "key_source_columns",
        ):
            if operation.get(field) is not None:
                _validate_runs(operation[field], d_model, field)


def _validate_mlp_operations(
    layer: dict[str, Any], physical_ids: set[str], d_model: int
) -> None:
    for operation in layer.get("mlp_operations", []):
        if operation.get("node") not in physical_ids | {None}:
            raise RuntimeError("MLP operation has an unresolved node")
        _validate_runs(operation.get("target_columns"), d_model, "MLP target")
        _validate_runs(
            operation.get("hidden_slots"),
            layer["active_mlp_neurons"],
            "hidden slots",
        )
        for field in ("source_columns", "source_columns_b"):
            if operation.get(field) is not None:
                _validate_runs(operation[field], d_model, field)


def _validate_schedule_references(
    payload: dict[str, Any], physical_ids: set[str]
) -> None:
    model = payload["model"]
    d_model = model["d_model"]
    layers = payload["schedule"].get("layers", [])
    if len(layers) != model["n_layers"]:
        raise RuntimeError("truth schedule layer count disagrees with the model")
    for index, layer in enumerate(layers):
        if layer.get("index") != index:
            raise RuntimeError("truth schedule layers are not in canonical order")
        _validate_attention_operations(layer, physical_ids, d_model)
        _validate_mlp_operations(layer, physical_ids, d_model)
    for state in payload["residual_stream"].get("states", []):
        for node_id, runs in state.get("nodes", {}).items():
            if node_id not in physical_ids:
                raise RuntimeError("residual state has an unresolved node")
            _validate_runs(runs, d_model, "residual columns")


def _validate_physical_layout(payload: dict[str, Any]) -> None:
    layout = payload["physical_layout"]
    matrices = layout.get("matrices", {})
    for owner, rectangles in layout.get("placements", {}).items():
        for rectangle in rectangles:
            matrix = rectangle.get("matrix")
            if matrix not in matrices:
                raise RuntimeError(f"placement {owner!r} names an unknown matrix")
            shape = matrices[matrix]["shape"]
            _validate_runs([rectangle.get("axis0")], shape[0], "matrix axis0")
            _validate_runs([rectangle.get("axis1")], shape[1], "matrix axis1")

    tensors = payload["parameter_support"].get("tensors", {})
    for record in layout.get("checkpoint_parameter_map", []):
        tensor = record.get("checkpoint_tensor")
        if tensor not in tensors:
            raise RuntimeError("checkpoint parameter map names an unknown tensor")
        shape = tensors[tensor]["shape"]
        if "checkpoint_rows" in record:
            _validate_runs([record["checkpoint_rows"]], shape[0], "checkpoint rows")
            _validate_runs(
                [record["checkpoint_columns"]], shape[1], "checkpoint columns"
            )
        elif "checkpoint_indices" in record:
            _validate_runs(
                [record["checkpoint_indices"]], shape[0], "checkpoint indices"
            )


def _validate_truth_internals(payload: dict[str, Any]) -> None:
    graphs = payload["graphs"]
    source_widths = _validate_graph_record(graphs["source"], "s")
    lowered_widths = _validate_graph_record(graphs["lowered"], "l")
    _validate_lowering_map(graphs.get("realization_map"), source_widths, lowered_widths)
    internal = graphs.get("internal_nodes", [])
    internal_ids = {node.get("id") for node in internal}
    if len(internal_ids) != len(internal) or any(
        not isinstance(node, str) or not node.startswith("i:") for node in internal_ids
    ):
        raise RuntimeError("truth manifest has invalid internal node IDs")
    physical_ids = set(lowered_widths) | internal_ids
    schedule = payload["schedule"]
    for section in ("assignment", "realizations"):
        if any(record.get("node") not in physical_ids for record in schedule[section]):
            raise RuntimeError(f"truth schedule {section} has an unresolved node")
    _validate_schedule_references(payload, physical_ids)
    _validate_physical_layout(payload)
    d_model = payload["model"]["d_model"]
    _validate_runs(
        payload["token_io"]["embedding_residual_columns"],
        d_model,
        "embedding columns",
    )
    _validate_runs(
        payload["token_io"]["output_residual_columns"], d_model, "output columns"
    )


def _validate_truth_manifest(
    directory: Path,
    *,
    staged_file_hashes: Mapping[str, dict[str, Any]] | None = None,
) -> None:
    truth_path = directory / TRUTH_FILENAME
    if not truth_path.is_file():
        raise RuntimeError(f"staged HF bundle has no {TRUTH_FILENAME}")
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    if payload.get("format") != TRUTH_FORMAT:
        raise RuntimeError("staged truth manifest has an unsupported format")
    # The packaged schema's required[] list is the one section inventory;
    # enforcing it here keeps the schema shipped in every bundle honest.
    schema = json.loads(_TRUTH_SCHEMA_SOURCE.read_text(encoding="utf-8"))
    missing = sorted(set(schema["required"]) - set(payload))
    if missing:
        raise RuntimeError(f"staged truth manifest is missing sections: {missing}")
    integrity = payload["integrity"]
    expected_digest = sha256_json(
        {key: value for key, value in payload.items() if key != "integrity"}
    )
    if not isinstance(integrity, dict) or integrity.get("sha256") != expected_digest:
        raise RuntimeError("truth manifest integrity hash mismatch")

    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    expected_pointer = {
        "format": TRUTH_FORMAT,
        "file": TRUTH_FILENAME,
        "schema": TRUTH_SCHEMA_FILENAME,
        "support": TRUTH_SUPPORT_FILENAME,
    }
    if config.get("torchwright_truth") != expected_pointer:
        raise RuntimeError("config.json truth pointer does not match the bundle")

    files = payload["artifact"].get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("truth manifest has no artifact file hashes")
    _validate_truth_bound_files(directory, files, staged_file_hashes=staged_file_hashes)
    _validate_support_records(directory, payload["parameter_support"])
    _validate_truth_internals(payload)


def _emit_truth_from_compile(
    output_dir: str | os.PathLike,
    *,
    compiled: HeadlessTransformer,
    profile: CompileProfile,
    spec: TokenModelSpec,
    sink: _DirectShardSink,
    output_node: Node,
    embedding: Embedding,
    bos_id: int | None,
    eos_id: int | None,
    metadata: Mapping[str, JSONValue] | None,
) -> dict[str, dict[str, Any]]:
    truth_capture = compiled.truth_capture
    if truth_capture is None:
        raise RuntimeError("HF bundle compile did not capture artifact truth")
    return _write_truth_manifest(
        output_dir,
        capture=truth_capture,
        profile=profile,
        spec=spec,
        sink=sink,
        compiled=compiled,
        output_node=output_node,
        embedding=embedding,
        bos_id=bos_id,
        eos_id=eos_id,
        metadata=metadata,
    )


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
    truth_metadata: Mapping[str, JSONValue] | None = None,
    _solver_seed: int | None = None,
    _force_resolve: bool = False,
) -> tuple[HFBundleReport, dict[str, dict[str, Any]]]:
    """Compile directly to a sharded safetensors HF bundle.

    Returns the bundle report plus the truth manifest's artifact file-hash
    records, so the caller's immediate re-validation can skip re-hashing
    the shards it just wrote.

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
            _capture_truth=True,
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
        config.torchwright_truth = {
            "format": TRUTH_FORMAT,
            "file": TRUTH_FILENAME,
            "schema": TRUTH_SCHEMA_FILENAME,
            "support": TRUTH_SUPPORT_FILENAME,
        }
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
    truth_files = _emit_truth_from_compile(
        output_dir,
        compiled=compiled,
        profile=profile,
        spec=spec,
        sink=sink,
        output_node=output_node,
        embedding=embedding,
        bos_id=bos_id,
        eos_id=eos_id,
        metadata=truth_metadata,
    )
    provenance = spec.schedule_provenance
    assert provenance is not None
    return (
        HFBundleReport(
            output_dir=output_dir,
            n_layers=spec.n_layers,
            schedule_provenance=provenance,
        ),
        truth_files,
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
    # The temp bundle's truth files are gone with the directory; the
    # returned in-memory model must not claim them (a later
    # save_pretrained would write a dangling truth pointer).
    if hasattr(model.config, "torchwright_truth"):
        del model.config.torchwright_truth
    return model.float().eval()
