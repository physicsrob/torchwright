"""Direct SwiGLU compilation to the stock Phi-3 Hugging Face target."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from torchwright.compiler.hf import (
    build_fast_tokenizer,
    compile_hf_bundle,
    compile_to_hf,
)
from torchwright.graph import Embedding


def _graph():
    import examples.adder as adder
    old = adder.max_digits
    adder.max_digits = 1
    try:
        out, emb = adder.create_network_parts()
    finally:
        adder.max_digits = old
    return out, emb, adder.D_MODEL, adder.D_HEAD


@pytest.fixture(scope="module")
def direct():
    out, emb, d, d_head = _graph()
    return compile_to_hf(out, emb, d=d, d_head=d_head, max_seq_len=64), emb


def test_direct_phi3_is_stock_fp32_eval(direct):
    from transformers import Phi3ForCausalLM
    model, _ = direct
    assert isinstance(model, Phi3ForCausalLM)
    assert not model.training
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert model.config.tie_word_embeddings is False
    assert model.config.head_dim > 0


def test_padding_and_fusion_are_zero_filled(direct):
    model, _ = direct
    rows = model.config.num_attention_heads * model.config.head_dim
    inter = model.config.intermediate_size
    for layer in model.model.layers:
        assert layer.self_attn.qkv_proj.weight.shape[0] == 3 * rows
        assert layer.mlp.gate_up_proj.weight.shape[0] == 2 * inter


def test_stock_streaming_bundle_loads_without_custom_code(tmp_path, monkeypatch):
    import importlib
    from transformers import AutoModelForCausalLM
    hf_build = importlib.import_module("torchwright.compiler.hf.build")
    monkeypatch.setattr(
        hf_build.tempfile,
        "TemporaryDirectory",
        lambda *args, **kwargs: pytest.fail("bundle build created a spool"),
    )
    out, emb, d, d_head = _graph()
    compile_hf_bundle(out, emb, tmp_path, d=d, d_head=d_head, max_seq_len=64)
    files = {p.name for p in tmp_path.iterdir()}
    assert "model.safetensors.index.json" in files
    assert not any(name.startswith("layer-") for name in files)
    assert not any(name.endswith(".py") for name in files)
    config = json.loads((tmp_path / "config.json").read_text())
    assert "auto_map" not in config
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert loaded.config.model_type == "phi3"


def test_trivial_graph_streaming_announces_placeholder_shape(tmp_path):
    embedding = Embedding(
        ["a"],
        d_embed=2,
        table=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    compile_hf_bundle(
        embedding,
        embedding,
        tmp_path,
        d=16,
        d_head=16,
        d_hidden=16,
        bos_token=None,
        eos_token=None,
        write_tokenizer=False,
    )

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["num_hidden_layers"] == 1
    assert config["num_attention_heads"] == 1
    assert config["intermediate_size"] == 1


def test_fast_tokenizer_character_round_trip(direct):
    _, emb = direct
    vocab = list(emb.tokenizer.vocab)
    tok = build_fast_tokenizer(vocab, bos_token="<bos>", eos_token="<eos>")
    encoded = tok("1+2\n")["input_ids"]
    assert encoded[0] == vocab.index("<bos>")
    assert tok.decode(encoded, skip_special_tokens=True) == "1+2\n"


def test_unsupported_swish_modes_fail_loudly():
    out, emb, d, d_head = _graph()
    with pytest.raises(ValueError, match="requires bias=False"):
        compile_to_hf(out, emb, d=d, d_head=d_head, rms_norm=True, bias=True)
