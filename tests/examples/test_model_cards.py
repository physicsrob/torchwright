"""Focused tests for the generated Hugging Face calculator cards."""

from __future__ import annotations

import json

import torch
from huggingface_hub import ModelCard
from safetensors.torch import save_file

from examples import calculator_scratchpad, calculator_simple
from examples.compile import _io_section, write_card


def test_io_examples_respect_the_bundle_width() -> None:
    card = _io_section(calculator_simple, max_digits=1)

    assert "`12*34`" not in card
    assert "| `9*9` | `81` |" in card
    assert "| `1-9` | `-8` |" in card


def test_scratchpad_note_uses_the_bundle_width() -> None:
    card = _io_section(calculator_scratchpad, max_digits=3)

    assert "`999999+1`" not in card
    assert "`999+1` ends\n`…</THINKING>1000`" in card
    assert "| `123-999` | `<THINKING>…</THINKING>-876` |" in card


def test_written_card_is_runnable_and_reports_serialized_size(tmp_path) -> None:
    shard = "model-00001-of-00001.safetensors"
    save_file({"weight": torch.tensor([0.0, 1.0])}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 2_000_000_000},
                "weight_map": {"weight": shard},
            }
        )
    )

    repo_id = "physicsrob/torchwright-calculator-simple-max-digits-3"
    write_card("calculator_simple", str(tmp_path), 3, repo_id=repo_id)
    card = (tmp_path / "README.md").read_text()

    assert card.startswith("---\n")
    assert ModelCard(card).data.to_dict() == {
        "license": "apache-2.0",
        "library_name": "transformers",
        "tags": ["torchwright", "compiled-transformer", "calculator_simple"],
        "pipeline_tag": "text-generation",
    }
    assert f"repo_id = {repo_id!r}" in card
    assert "from_pretrained(REPO)" not in card
    flattened = " ".join(card.split())
    assert (
        "Other precisions and decoding modes are outside the supported contract."
        in flattened
    )
    assert "exact algebraic cancellation" not in card
    assert "2.00 GB of dense fp32 weights" in card
    assert "50.00% of those entries are exactly zero" in flattened
