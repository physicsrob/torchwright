from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from safetensors import safe_open
from safetensors.torch import save_file

from scripts.migrate_hf_unk import migrate_bundle


def test_migrate_bundle_appends_zero_semantic_row(tmp_path):
    vocab = ["0", "<bos>", "<eos>"]
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": len(vocab)}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<bos>", "eos_token": "<eos>"})
    )
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "WordLevel",
                    "vocab": {token: i for i, token in enumerate(vocab)},
                    "unk_token": "<unk>",
                }
            }
        )
    )

    embed = torch.zeros(3, 8)
    embed[:, 6] = 4.0  # token-independent compiler bookkeeping coordinate
    embed[0, 0] = embed[1, 1] = embed[2, 2] = 1.0
    shard_name = "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": embed,
            "model.norm.weight": torch.ones(8),
        },
        tmp_path / shard_name,
    )
    old_size = embed.numel() * embed.element_size() + 8 * 4
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": old_size},
                "weight_map": {
                    "model.embed_tokens.weight": shard_name,
                    "model.norm.weight": shard_name,
                },
            }
        )
    )

    result = migrate_bundle(tmp_path, label="fixture")

    assert result.changed is True
    assert result.unk_token_id == 3
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["vocab_size"] == 4
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == old_size + 8 * 4
    with safe_open(tmp_path / shard_name, framework="pt", device="cpu") as handle:
        migrated = handle.get_tensor("model.embed_tokens.weight")
    assert migrated.shape == (4, 8)
    assert torch.equal(migrated[:3], embed)
    assert torch.equal(
        migrated[3], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0])
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tmp_path)
    assert tok.unk_token_id == 3
    assert tok("☃", add_special_tokens=False)["input_ids"] == [3]

    again = migrate_bundle(tmp_path, label="fixture")
    assert again.changed is False
    assert again.unk_token_id == 3
