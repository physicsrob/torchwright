"""Artifact-bound mechanistic-interpretability metadata for HF bundles."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors")

from torchwright.compiler.hf import compile_hf_bundle, compile_to_hf, save_hf_bundle
from torchwright.compiler.hf.build import _validate_staged_bundle
from torchwright.compiler.truth import TRUTH_FORMAT, sha256_json
from torchwright.graph import Add, Embedding, Linear
from torchwright.ops.linear import add, subtract


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
    result: set[tuple[int, ...]] = set()
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
    const_col = manifest["residual_stream"]["constant_one_column"]
    for op in layer["attention_operations"]:
        # Both ops are Δ=0 self-matches: Q and K read the constant-1 column.
        assert op["query_source_columns"] == [[const_col, 1]]
        assert op["key_source_columns"] == [[const_col, 1]]
    assert "physical_coordinates" in manifest["intervention_contract"]
    assert manifest["integrity"]["sha256"] == sha256_json(
        {key: value for key, value in manifest.items() if key != "integrity"}
    )
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


def test_truth_manifest_semantic_regions(tmp_path):
    """Regions built through decorated ops land in the source record.

    ``subtract`` is the composition proof: it creates no node directly
    (add ∘ negate), so its record is reachable only through the parent
    chain — and it must still appear in the table with resolving
    operand/result references.
    """
    embedding = Embedding(
        ["a"],
        d_embed=2,
        table=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    total = add(embedding, embedding)
    output = subtract(total, embedding)
    compile_hf_bundle(
        output,
        embedding,
        tmp_path,
        d=32,
        d_head=16,
        d_hidden=16,
        bos_token=None,
        eos_token=None,
        write_tokenizer=False,
    )
    manifest = json.loads((tmp_path / "torchwright_truth.json").read_text())
    source = manifest["graphs"]["source"]
    regions = source["semantic_regions"]
    # Canonical walk: outer Add (subtract's), inner Add, embedding, negate.
    assert [record["op"] for record in regions] == [
        "linear.subtract",
        "linear.add",
        "linear.add",
        "linear.negate",
    ]
    assert [record["parent"] for record in regions] == [None, "r:0", None, "r:0"]
    node_ids = {node["id"] for node in source["nodes"]}
    by_op = {node["op"]: node for node in source["nodes"]}
    for record in regions:
        assert record["params"] == {}
        for field in ("operands", "results"):
            assert record[field], record
            assert set(record[field]) <= node_ids
    # Membership: every node names its innermost creating op call.
    memberships = {node["id"]: node["region"] for node in source["nodes"]}
    assert memberships[source["root"]] == "r:1"  # subtract's internal add
    assert memberships[by_op["Embedding"]["id"]] is None
    # The stamped source hash covers the regions table: an edit that also
    # recomputes the whole-manifest integrity hash still trips it.
    truth_path = tmp_path / "torchwright_truth.json"
    manifest["graphs"]["source"]["semantic_regions"][0]["params"] = {"x": 1}
    manifest["integrity"]["sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "integrity"}
    )
    truth_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="source graph hash mismatch"):
        _validate_staged_bundle(tmp_path, expect_tokenizer=False, expect_truth=True)


def test_truth_validation_rejects_a_tampered_bound_file(tmp_path):
    _compile_small_bundle(tmp_path)
    schema = tmp_path / "torchwright_truth_v1.schema.json"
    original_schema = schema.read_text()
    schema.write_text(original_schema + "\n")

    with pytest.raises(RuntimeError, match="size mismatch"):
        _validate_staged_bundle(tmp_path, expect_tokenizer=False, expect_truth=True)

    schema.write_text(original_schema)
    truth_path = tmp_path / "torchwright_truth.json"
    original_truth = truth_path.read_text()

    # Any in-place manifest edit trips the whole-manifest integrity hash.
    manifest = json.loads(original_truth)
    manifest["token_io"]["unknown_token_id"] = 999
    truth_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="integrity hash mismatch"):
        _validate_staged_bundle(tmp_path, expect_tokenizer=False, expect_truth=True)

    # An edit that also recomputes the integrity hash still trips the
    # tampered section's own content hash.
    manifest = json.loads(original_truth)
    manifest["graphs"]["source"]["nodes"][0]["name"] = "tampered"
    manifest["integrity"]["sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "integrity"}
    )
    truth_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="source graph hash mismatch"):
        _validate_staged_bundle(tmp_path, expect_tokenizer=False, expect_truth=True)


def test_zero_support_attention_linear_builds_with_empty_head_span(tmp_path):
    """A zero-support attention-routed op emits a legitimate [cursor, 0] span.

    The writer allocates its one floor head, trim removes it, and the
    manifest must describe the trimmed checkpoint: an empty head span and
    no placement rectangles beyond the declared trimmed matrix widths.
    Before validation accepted empty spans, this build crashed after the
    full compile.
    """
    embedding = Embedding(
        ["a"],
        d_embed=2,
        table=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    # Width 8 makes each Linear's MLP demand (2 * d_output = 16) exceed the
    # usable hidden pool (16 minus the bias=False constant lane), forcing
    # attention transport; the Add blocks consecutive-Linear fusion.
    zero = Linear(embedding, torch.zeros(2, 8), name="zero")
    wide = Linear(embedding, torch.ones(2, 8), name="wide")
    combined = Add(zero, wide)
    output = Linear(combined, torch.ones(8, 2), name="output")
    compile_hf_bundle(
        output,
        embedding,
        tmp_path,
        d=32,
        d_head=16,
        d_hidden=16,
        bos_token=None,
        eos_token=None,
        write_tokenizer=False,
    )
    manifest = json.loads((tmp_path / "torchwright_truth.json").read_text())
    linear_spans = [
        op["heads"]
        for layer in manifest["schedule"]["layers"]
        for op in layer["attention_operations"]
        if op["type"] == "compute_linear"
    ]
    assert [0, 0] in linear_spans
    matrices = manifest["physical_layout"]["matrices"]
    for rectangles in manifest["physical_layout"]["placements"].values():
        for rectangle in rectangles:
            shape = matrices[rectangle["matrix"]]["shape"]
            assert rectangle["axis0"][0] + rectangle["axis0"][1] <= shape[0]
            assert rectangle["axis1"][0] + rectangle["axis1"][1] <= shape[1]


def test_in_memory_model_carries_no_truth_pointer(tmp_path):
    """compile_to_hf discards the truth files with its temp bundle.

    The returned model and any bundle re-saved from it must not advertise
    a truth manifest they do not carry.
    """
    embedding = Embedding(
        ["a"],
        d_embed=2,
        table=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    output = Linear(embedding, torch.eye(2), name="output")
    model = compile_to_hf(
        output,
        embedding,
        d=32,
        d_head=16,
        d_hidden=16,
        bos_token=None,
        eos_token=None,
    )
    assert not hasattr(model.config, "torchwright_truth")
    save_hf_bundle(
        model,
        list(embedding.tokenizer.vocab),
        tmp_path,
        write_tokenizer=False,
    )
    config = json.loads((tmp_path / "config.json").read_text())
    assert "torchwright_truth" not in config
    _validate_staged_bundle(tmp_path, expect_tokenizer=False)
