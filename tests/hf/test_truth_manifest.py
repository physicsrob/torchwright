"""Artifact-bound mechanistic-interpretability metadata for HF bundles."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors")

from torchwright.compiler.hf import compile_hf_bundle
from torchwright.compiler.hf.build import _validate_staged_bundle
from torchwright.compiler.truth import TRUTH_FORMAT
from torchwright.graph import Embedding, Linear


def _compile_small_bundle(path):
    embedding = Embedding(
        ["a"],
        d_embed=2,
        table=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    output = Linear(embedding, torch.eye(2), name="output")
    compile_hf_bundle(
        output,
        embedding,
        path,
        d=32,
        d_head=16,
        d_hidden=16,
        bos_token=None,
        eos_token=None,
        write_tokenizer=False,
        truth_metadata={"task": {"name": "truth-test"}},
    )


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _support_coordinates(record, arrays) -> set[tuple[int, ...]]:
    if record["encoding"] == "flat_indices":
        return {(int(index),) for index in arrays[record["indices"]]}
    result = set()
    for chunk in record["chunks"]:
        indptr = arrays[chunk["indptr"]]
        indices = arrays[chunk["indices"]]
        for local_row in range(chunk["row_count"]):
            start, end = indptr[local_row : local_row + 2]
            result.update(
                (chunk["row_start"] + local_row, int(column))
                for column in indices[start:end]
            )
    return result


def test_truth_manifest_binds_graph_schedule_layout_and_support(tmp_path):
    _compile_small_bundle(tmp_path)
    manifest = json.loads((tmp_path / "torchwright_truth.json").read_text())
    config = json.loads((tmp_path / "config.json").read_text())

    assert manifest["format"] == TRUTH_FORMAT
    assert config["torchwright_truth"] == {
        "format": TRUTH_FORMAT,
        "file": "torchwright_truth.json",
        "schema": "torchwright_truth_v1.schema.json",
        "support": "torchwright_truth_support.npz",
    }
    assert manifest["metadata"] == {"task": {"name": "truth-test"}}

    source = manifest["graphs"]["source"]
    lowered = manifest["graphs"]["lowered"]
    assert {node["op"] for node in source["nodes"]} == {"Embedding", "Linear"}
    assert all(node["id"].startswith("s:") for node in source["nodes"])
    assert all(node["id"].startswith("l:") for node in lowered["nodes"])
    assert {entry["status"] for entry in manifest["graphs"]["realization_map"]} == {
        "whole"
    }
    assert all("value_contract" in node for node in source["nodes"])

    layer = manifest["schedule"]["layers"][0]
    assert layer["active_attention_heads"] == 2
    assert [op["type"] for op in layer["attention_operations"]] == [
        "compute_linear",
        "cancel",
    ]
    assert [op["heads"] for op in layer["attention_operations"]] == [[0, 1], [1, 1]]
    assert manifest["residual_stream"]["states"][0]["key"] == "input"
    assert manifest["residual_stream"]["states"][-1]["key"] == "output"
    assert manifest["schedule"]["residual_accesses"]
    assert "physical_coordinates" in manifest["intervention_contract"]
    assert manifest["reference_fixtures"]["included"] == []
    assert manifest["physical_layout"]["placements"]

    parameter_map = manifest["physical_layout"]["checkpoint_parameter_map"]
    q_map = next(
        record for record in parameter_map if record["logical_matrix"] == "L0.attn.W_Q"
    )
    assert q_map["checkpoint_tensor"].endswith("qkv_proj.weight")
    assert q_map["checkpoint_rows"] == [0, 32]
    assert q_map["scale"] == pytest.approx(4.0)

    files = manifest["artifact"]["files"]
    assert not any(name.startswith(".") for name in files)
    for name, record in files.items():
        path = tmp_path / name
        assert path.stat().st_size == record["bytes"]
        assert _file_sha256(path) == record["sha256"]

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    support = manifest["parameter_support"]
    with np.load(tmp_path / support["file"], allow_pickle=False) as arrays:
        for tensor_name, record in support["tensors"].items():
            shard = tmp_path / index["weight_map"][tensor_name]
            with safetensors.safe_open(shard, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(tensor_name)
            expected = {
                tuple(int(value) for value in coord)
                for coord in torch.nonzero(tensor, as_tuple=False).tolist()
            }
            assert _support_coordinates(record, arrays) == expected
            assert record["nnz"] == len(expected)


def test_truth_validation_rejects_a_tampered_bound_file(tmp_path):
    _compile_small_bundle(tmp_path)
    schema = tmp_path / "torchwright_truth_v1.schema.json"
    original_schema = schema.read_text()
    schema.write_text(original_schema + "\n")

    with pytest.raises(RuntimeError, match="size mismatch"):
        _validate_staged_bundle(tmp_path, expect_tokenizer=False, expect_truth=True)

    schema.write_text(original_schema)
    truth_path = tmp_path / "torchwright_truth.json"
    manifest = json.loads(truth_path.read_text())
    manifest["graphs"]["source"]["nodes"][0]["name"] = "tampered"
    truth_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="source graph hash mismatch"):
        _validate_staged_bundle(tmp_path, expect_tokenizer=False, expect_truth=True)
