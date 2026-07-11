"""Direct compiler-to-Hugging-Face build path (no ONNX dependency)."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from typing import Optional, Union

import numpy as np

from torchwright.compiler.forward.compile import forward_compile, rms_norm_width_supported
from torchwright.compiler.token_model import (
    CompileHeader, CompileProfile, ReluLayerWeights, SwishLayerWeights, TokenModelSpec,
    build_token_weights, make_layer_callback, resolve_rope,
    schedule_provenance,
)
from torchwright.graph import Embedding, Node

HFArchitecture = Union[CompileProfile, str]


def _token_id(vocab, token, kind):
    if token is None:
        return None
    if token not in vocab:
        raise ValueError(f"requested {kind}_token {token!r} is not in the vocabulary")
    return vocab.index(token)


def _resolve_architecture(architecture, bias, rms_norm):
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


def _target(activation: str, bias: bool, rms_norm: bool, architecture=None) -> str:
    profile = CompileProfile(architecture)
    expected = profile.value
    actual = (
        "phi3" if activation == "swish" and not bias and rms_norm
        else "custom" if activation == "relu" and bias
        else None
    )
    if actual != expected:
        raise AssertionError(
            f"architecture={architecture!r} compiled to activation={activation!r}, "
            f"bias={bias}, rms_norm={rms_norm}; expected {expected}"
        )
    return actual


def _torch(arr):
    import torch
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32).copy())


def build_fast_tokenizer(vocab, *, bos_token="<bos>", eos_token="<eos>", add_bos_token=True):
    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    vocab_dict = {token: i for i, token in enumerate(vocab)}
    unk = "<unk>" if "<unk>" in vocab_dict else None
    tok = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens([t for t in (unk, bos_token, eos_token) if t is not None])
    if add_bos_token and bos_token is not None:
        tok.post_processor = processors.TemplateProcessing(
            single=f"{bos_token} $A", pair=f"{bos_token} $A {bos_token} $B",
            special_tokens=[(bos_token, vocab_dict[bos_token])])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token=unk,
        bos_token=bos_token, eos_token=eos_token)


def save_hf_bundle(model, vocab, output_dir, *, add_bos_token=True, write_tokenizer=True):
    """Save a directly compiled model and its vocabulary as an HF bundle."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    if not write_tokenizer:
        return model
    bos_id, eos_id = model.config.bos_token_id, model.config.eos_token_id
    bos = vocab[bos_id] if bos_id is not None else None
    eos = vocab[eos_id] if eos_id is not None else None
    if model.config.model_type == "phi3":
        build_fast_tokenizer(list(vocab), bos_token=bos, eos_token=eos,
            add_bos_token=add_bos_token).save_pretrained(output_dir)
    else:
        from .configuration_torchwright_custom import TorchwrightCustomConfig
        from .modeling_torchwright_custom import TorchwrightCustomForCausalLM
        from .tokenization_torchwright_custom import TorchwrightCustomTokenizer
        TorchwrightCustomConfig.register_for_auto_class()
        TorchwrightCustomForCausalLM.register_for_auto_class("AutoModelForCausalLM")
        vocab_path = os.path.join(output_dir, "vocab.json")
        with open(vocab_path, "w") as f: json.dump(list(vocab), f)
        tok = TorchwrightCustomTokenizer(vocab_file=vocab_path, bos_token=bos,
            eos_token=eos, add_bos_token=add_bos_token)
        TorchwrightCustomTokenizer.register_for_auto_class()
        tok.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
    return model


def compile_hf_bundle(
    output_node: Node,
    embedding: Embedding,
    output_dir,
    *,
    d=1024,
    d_head=16,
    max_seq_len=512,
    max_layers=400,
    optimize=0,
    d_hidden=None,
    trim_heads=True,
    rms_norm=None,
    rms_norm_eps=1e-5,
    rms_norm_const_exp=None,
    architecture: HFArchitecture = "phi3",
    bias: Optional[bool] = None,
    bos_token="<bos>",
    eos_token="<eos>",
    verbose=False,
    add_bos_token=True,
    write_tokenizer=True,
    _solver_seed=None,
    _force_resolve=False,
):
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
    from safetensors.torch import save_file

    profile = _resolve_architecture(architecture, bias, rms_norm)
    machine, bias = profile.machine, profile.bias
    rms_on = profile.rms_norm if rms_norm is None else bool(rms_norm)
    if rms_on and not rms_norm_width_supported(d):
        raise ValueError(f"rms_norm is on but d={d} is unsupported")

    class DirectShardSink:
        def __init__(self):
            self.meta = []
            self.weight_map = {}
            self.total_size = 0
        def begin(self, header):
            self.header = header
            if not header.layer_shapes:
                raise RuntimeError("HF streaming requires announced layer shapes")
            self.shard_count = len(header.layer_shapes) + 1
            self.max_heads = max(shape.n_heads for shape in header.layer_shapes)
            self.max_hidden = max(shape.d_hidden for shape in header.layer_shapes)
        def write_layer(self, index, layer):
            a = layer.attention
            p = f"model.layers.{index}"
            if profile is CompileProfile.CUSTOM:
                assert isinstance(layer, ReluLayerWeights)
                sd = {
                    f"{p}.self_attn.q_proj.weight": _torch(a.wq).T.contiguous(),
                    f"{p}.self_attn.k_proj.weight": _torch(a.wk).T.contiguous(),
                    f"{p}.self_attn.v_proj.weight": _torch(a.wv).T.contiguous(),
                    f"{p}.self_attn.o_proj.weight": _torch(a.wo).T.contiguous(),
                    f"{p}.mlp.fc1.weight": _torch(layer.w1).T.contiguous(),
                    f"{p}.mlp.fc1.bias": _torch(layer.b1),
                    f"{p}.mlp.fc2.weight": _torch(layer.w2).T.contiguous(),
                    f"{p}.mlp.fc2.bias": _torch(layer.b2),
                }
                kind = "relu"
            else:
                assert isinstance(layer, SwishLayerWeights)
                rows, inter = self.max_heads * d_head, self.max_hidden
                q = (_torch(a.wq).T.double() * math.sqrt(float(d_head))).float()
                def tpad(value, target_size, axis):
                    shape = list(value.shape); shape[axis] = target_size
                    out = torch.zeros(shape, dtype=torch.float32)
                    sl = [slice(None)] * value.ndim
                    sl[axis] = slice(0, value.shape[axis])
                    out[tuple(sl)] = value
                    return out
                wk, wv, wo = _torch(a.wk), _torch(a.wv), _torch(a.wo)
                wg, wu, wd = _torch(layer.wgate), _torch(layer.wup), _torch(layer.wdown)
                sd = {
                    f"{p}.self_attn.qkv_proj.weight": torch.cat([
                        tpad(q, rows, 0), tpad(wk.T, rows, 0), tpad(wv.T, rows, 0)
                    ]),
                    f"{p}.self_attn.o_proj.weight": tpad(wo.T, rows, 1),
                    f"{p}.mlp.gate_up_proj.weight": torch.cat([
                        tpad(wg.T, inter, 0), tpad(wu.T, inter, 0)
                    ]),
                    f"{p}.mlp.down_proj.weight": tpad(wd.T, inter, 1),
                }
                kind = "swish"
            filename = f"model-{index+1:05d}-of-{self.shard_count:05d}.safetensors"
            save_file(sd, os.path.join(output_dir, filename))
            for name, value in sd.items():
                self.weight_map[name] = filename
                self.total_size += value.numel() * value.element_size()
            self.meta.append((kind, a.n_heads, layer.d_hidden, a.rope_base, a.d_rot))
        def finalize(self, spec, weights):
            self.spec = spec
            self.token_weights = weights

    os.makedirs(output_dir, exist_ok=True)
    sink = DirectShardSink()
    with torch.no_grad():
        compiled = forward_compile(
            d=d, d_head=d_head, output_node=output_node, verbose=verbose,
            max_layers=max_layers, device=None,
            on_layer_compiled=make_layer_callback(
                CompileHeader(d, d_head, trim_heads, bias), sink),
            trim_heads=trim_heads, optimize=optimize, bias=bias,
            d_hidden=d_hidden, rms_norm=rms_on, rms_norm_eps=rms_norm_eps,
            machine=machine,
            _solver_seed=_solver_seed, _force_resolve=_force_resolve,
            **({} if rms_norm_const_exp is None else {"rms_norm_const_exp": rms_norm_const_exp}),
        )
        token = build_token_weights(compiled, output_node, embedding, d)
        heads = [m[1] for m in sink.meta]
        hidden = [m[2] for m in sink.meta]
        proxy_layers = []
        for kind, nh, dh, base, drot in sink.meta:
            # Only RoPE metadata is inspected by resolve_rope.
            class A: pass
            class L: pass
            a, layer = A(), L()
            a.rope_base, a.d_rot = base, drot
            layer.attention = a
            proxy_layers.append(layer)
        rope_base, d_rot = resolve_rope(proxy_layers, d_head)
        vocab = tuple(embedding.tokenizer.vocab)
        spec = TokenModelSpec(d, d_head, max_seq_len, vocab,
            token.embed_table.shape[0], compiled.activation, bool(bias),
            compiled.rms_norm_spec is not None,
            float(compiled.rms_norm_spec.eps if compiled.rms_norm_spec else rms_norm_eps),
            rope_base, d_rot, len(sink.meta), tuple(heads), tuple(hidden),
            schedule_provenance(compiled, optimize))
        sink.finalize(spec, token)
        target = _target(
            spec.activation, spec.bias, spec.rms_norm, architecture=profile
        )
        bos_id = _token_id(vocab, bos_token, "bos")
        eos_id = _token_id(vocab, eos_token, "eos")
        if target == "custom":
            from .configuration_torchwright_custom import TorchwrightCustomConfig
            config = TorchwrightCustomConfig(d=d, d_head=d_head, vocab_size=spec.vocab_size,
                n_layers=spec.n_layers, n_heads_per_layer=heads,
                d_hidden_per_layer=hidden, max_position_embeddings=max_seq_len,
                rope_base=rope_base, d_rot=d_rot, rms_norm=spec.rms_norm,
                rms_norm_eps=spec.rms_norm_eps, bos_token_id=bos_id,
                eos_token_id=eos_id, tie_word_embeddings=False)
            config.architectures = ["TorchwrightCustomForCausalLM"]
            config.auto_map = {
                "AutoConfig": "configuration_torchwright_custom.TorchwrightCustomConfig",
                "AutoModelForCausalLM": "modeling_torchwright_custom.TorchwrightCustomForCausalLM",
            }
        else:
            from transformers import Phi3Config
            max_heads, inter = max(heads), max(hidden)
            config = Phi3Config(vocab_size=spec.vocab_size, hidden_size=d,
                intermediate_size=inter, num_hidden_layers=spec.n_layers,
                num_attention_heads=max_heads, num_key_value_heads=max_heads,
                head_dim=d_head, hidden_act="silu", max_position_embeddings=max_seq_len,
                rms_norm_eps=spec.rms_norm_eps, rope_parameters={"rope_type":"default",
                "rope_theta":rope_base,"partial_rotary_factor":d_rot/d_head},
                sliding_window=None, attention_dropout=0.0, resid_pdrop=0.0,
                embd_pdrop=0.0, use_cache=True, tie_word_embeddings=False,
                bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=None)
            config.architectures = ["Phi3ForCausalLM"]
        config.save_pretrained(output_dir)

        shard_count = spec.n_layers + 1
        weight_map, total_size = sink.weight_map, sink.total_size
        gain = token.norm_gain
        final_sd = {
            "model.embed_tokens.weight": _torch(token.embed_table),
            "lm_head.weight": _torch(token.lm_head),
        }
        if gain is not None:
            final_sd["model.norm.weight"] = _torch(gain)
            for i in range(spec.n_layers):
                p = f"model.layers.{i}"
                final_sd[f"{p}.input_layernorm.weight"] = _torch(gain)
                final_sd[f"{p}.post_attention_layernorm.weight"] = _torch(gain)
        filename = f"model-{shard_count:05d}-of-{shard_count:05d}.safetensors"
        save_file(final_sd, os.path.join(output_dir, filename))
        for name, value in final_sd.items():
            weight_map[name] = filename; total_size += value.numel() * value.element_size()
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata":{"total_size":total_size},"weight_map":weight_map}, f)

    if target == "custom":
        here = os.path.dirname(__file__)
        for name in ("configuration_torchwright_custom.py", "modeling_torchwright_custom.py", "tokenization_torchwright_custom.py"):
            shutil.copy2(os.path.join(here, name), os.path.join(output_dir, name))
    if write_tokenizer:
        if target == "phi3":
            build_fast_tokenizer(vocab, bos_token=bos_token, eos_token=eos_token,
                add_bos_token=add_bos_token).save_pretrained(output_dir)
        else:
            from .tokenization_torchwright_custom import TorchwrightCustomTokenizer
            vocab_path = os.path.join(output_dir, "vocab.json")
            with open(vocab_path, "w") as f: json.dump(list(vocab), f)
            TorchwrightCustomTokenizer.register_for_auto_class()
            TorchwrightCustomTokenizer(vocab_file=vocab_path, bos_token=bos_token,
                eos_token=eos_token, add_bos_token=add_bos_token).save_pretrained(output_dir)
    return output_dir


# The public in-memory entry point intentionally reuses the streaming bundle
# sink.  Loading its final shards leaves the destination module as the only
# retained full-model copy (plus the currently loaded shard), rather than
# briefly holding every canonical compiler layer alongside every parameter.
def compile_to_hf(
    output_node: Node,
    embedding: Embedding,
    *,
    d=1024,
    d_head=16,
    max_seq_len=512,
    max_layers=400,
    optimize=0,
    d_hidden=None,
    trim_heads=True,
    rms_norm=None,
    rms_norm_eps=1e-5,
    rms_norm_const_exp=None,
    architecture: HFArchitecture = "phi3",
    bias: Optional[bool] = None,
    bos_token="<bos>",
    eos_token="<eos>",
    verbose=False,
    _solver_seed=None,
    _force_resolve=False,
):
    """Compile directly into an fp32 eval-mode Hugging Face model.

    The default is stock ``Phi3ForCausalLM``. The custom implementation is
    reachable only through the explicit ``architecture="custom"`` opt-in.
    """
    with tempfile.TemporaryDirectory(prefix="torchwright-hf-model-") as directory:
        compile_hf_bundle(
            output_node, embedding, directory, d=d, d_head=d_head,
            max_seq_len=max_seq_len, max_layers=max_layers, optimize=optimize,
            d_hidden=d_hidden, trim_heads=trim_heads, rms_norm=rms_norm,
            rms_norm_eps=rms_norm_eps, rms_norm_const_exp=rms_norm_const_exp,
            architecture=architecture, bias=bias,
            bos_token=bos_token, eos_token=eos_token,
            verbose=verbose, _solver_seed=_solver_seed,
            _force_resolve=_force_resolve, write_tokenizer=False,
        )
        with open(os.path.join(directory, "config.json")) as f:
            model_type = json.load(f)["model_type"]
        if model_type == "phi3":
            from transformers import Phi3ForCausalLM
            model = Phi3ForCausalLM.from_pretrained(
                directory, attn_implementation="eager"
            )
        else:
            from .modeling_torchwright_custom import TorchwrightCustomForCausalLM
            model = TorchwrightCustomForCausalLM.from_pretrained(directory)
    return model.float().eval()
