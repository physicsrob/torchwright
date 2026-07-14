"""Direct SwiGLU compilation to the stock Phi-3 Hugging Face target."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from torchwright.compiler.hf import (
    HFBundleReport,
    build_fast_tokenizer,
    compile_hf_bundle,
    compile_to_hf,
    save_hf_bundle,
)
from torchwright.graph import Concatenate, Embedding


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
    report = compile_hf_bundle(out, emb, tmp_path, d=d, d_head=d_head, max_seq_len=64)
    files = {p.name for p in tmp_path.iterdir()}
    assert "model.safetensors.index.json" in files
    assert not any(name.startswith("layer-") for name in files)
    assert not any(name.endswith(".py") for name in files)
    config = json.loads((tmp_path / "config.json").read_text())
    assert "auto_map" not in config
    assert isinstance(report, HFBundleReport)
    assert report.output_dir == tmp_path
    assert report.n_layers == config["num_hidden_layers"]
    assert report.schedule_provenance.optimize == 0
    assert report.schedule_provenance.selected_origin == "heuristic"
    assert report.schedule_provenance.delivery == "fresh"
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert loaded.config.model_type == "phi3"


def test_failed_bundle_compile_preserves_existing_destination(tmp_path):
    destination = tmp_path / "published"
    destination.mkdir()
    sentinel = destination / "existing.txt"
    sentinel.write_text("keep me")
    out, emb, d, d_head = _graph()

    with pytest.raises(ValueError, match="bos_token"):
        compile_hf_bundle(
            out,
            emb,
            destination,
            d=d,
            d_head=d_head,
            bos_token="<missing-bos>",
            write_tokenizer=False,
        )

    assert {path.name for path in destination.iterdir()} == {"existing.txt"}
    assert sentinel.read_text() == "keep me"
    assert not list(tmp_path.glob(".published.staging-*"))


def test_publish_swap_failure_rolls_back_existing_destination(tmp_path, monkeypatch):
    import torchwright.compiler.hf.build as hf_build

    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "existing.txt").write_text("old")
    real_replace = hf_build.os.replace

    def fail_staging_publish(source, target):
        if (
            ".published.staging-" in str(source)
            and not str(source).endswith(".previous")
            and Path(target) == destination
        ):
            raise OSError("simulated publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(hf_build.os, "replace", fail_staging_publish)
    with pytest.raises(OSError, match="simulated publish failure"):
        with hf_build._staged_bundle_directory(destination) as staging:
            (Path(staging) / "new.txt").write_text("new")

    assert {path.name for path in destination.iterdir()} == {"existing.txt"}
    assert (destination / "existing.txt").read_text() == "old"
    assert not list(tmp_path.glob(".published.staging-*"))


def test_successful_bundle_publish_replaces_stale_destination(tmp_path):
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "stale.txt").write_text("old")
    out, emb, d, d_head = _graph()

    compile_hf_bundle(
        out,
        emb,
        destination,
        d=d,
        d_head=d_head,
        write_tokenizer=False,
    )

    assert not (destination / "stale.txt").exists()
    assert (destination / "model.safetensors.index.json").is_file()


def test_failed_save_bundle_preserves_existing_destination(tmp_path, direct):
    model, _ = direct
    destination = tmp_path / "published"
    destination.mkdir()
    sentinel = destination / "existing.txt"
    sentinel.write_text("keep me")

    with pytest.raises(IndexError):
        save_hf_bundle(model, ["too-short"], destination)

    assert {path.name for path in destination.iterdir()} == {"existing.txt"}
    assert sentinel.read_text() == "keep me"


def test_bundle_rejects_foreign_same_shape_embedding_before_writing(tmp_path):
    out, emb, d, d_head = _graph()
    foreign = Embedding(
        list(emb.tokenizer.vocab)[1:],
        d_embed=emb.d_embed,
    )

    with pytest.raises(ValueError, match="not the Embedding reachable"):
        compile_hf_bundle(out, foreign, tmp_path / "bundle", d=d, d_head=d_head)

    assert not (tmp_path / "bundle").exists()


def test_bundle_rejects_multiple_reachable_embeddings_before_writing(tmp_path):
    first = Embedding(["a"], d_embed=2, table=torch.eye(2), special_tokens=["<unk>"])
    second = Embedding(["a"], d_embed=2, table=torch.eye(2), special_tokens=["<unk>"])

    with pytest.raises(ValueError, match="exactly one Embedding.*found 2"):
        compile_hf_bundle(
            Concatenate([first, second]),
            first,
            tmp_path / "bundle",
            d=16,
            d_head=16,
        )

    assert not (tmp_path / "bundle").exists()


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


def test_trivial_graph_accepts_explicit_decoupled_head_count(tmp_path):
    embedding = Embedding(
        ["a"],
        d_embed=2,
        table=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    compile_hf_bundle(
        embedding,
        embedding,
        tmp_path,
        d=32,
        d_head=8,
        n_heads=3,
        d_hidden=16,
        trim_heads=False,
        bos_token=None,
        eos_token=None,
        write_tokenizer=False,
    )

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["hidden_size"] == 32
    assert config["head_dim"] == 8
    assert config["num_attention_heads"] == 3


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
