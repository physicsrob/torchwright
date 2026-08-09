"""The reader answers the consumer questions against a real bundle."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from safetensors import safe_open

from torchwright.compiler.hf import compile_hf_bundle
from torchwright.graph import Embedding
from torchwright.ops.linear import add, subtract
from torchwright.schematic.format import sha256_json
from torchwright.schematic.reader import load_schematic, load_schematic_bundle
from torchwright.schematic.validate import SchematicValidationError


def _compile_region_bundle(path):
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
        path,
        d=32,
        d_head=16,
        d_hidden=16,
        bos_token=None,
        eos_token=None,
        write_tokenizer=False,
    )


def test_region_navigation(tmp_path):
    _compile_region_bundle(tmp_path)
    schematic = load_schematic(tmp_path)

    root = schematic.payload["graphs"]["source"]["root"]
    chain = schematic.region_chain(root)
    assert [region.op for region in chain] == ["linear.add", "linear.subtract"]

    # Composition-op honesty: subtract creates no node directly.
    subtract_id = chain[1].id
    assert schematic.region_members(subtract_id, direct=True) == ()
    subtree = schematic.region_members(subtract_id)
    assert {node.op for node in subtree} == {"Add", "Linear"}
    assert {child.op for child in schematic.region_children(subtract_id)} == {
        "linear.add",
        "linear.negate",
    }
    assert [a.id for a in schematic.region_ancestors(chain[0].id)] == [subtract_id]

    embedding_id = schematic.payload["token_io"]["source_embedding_node"]
    assert schematic.region_chain(embedding_id) == ()
    assert schematic.source_node(embedding_id).region is None


def test_residual_states_and_reverse_lookup(tmp_path):
    _compile_region_bundle(tmp_path)
    schematic = load_schematic(tmp_path)

    states = schematic.residual_states()
    n_layers = schematic.payload["model"]["n_layers"]
    assert len(states) == n_layers + 2
    assert states[0].key == "input"
    assert states[-1].key == "output"

    # token_io columns are the embedding / output nodes' state columns.
    embedding_src = schematic.payload["token_io"]["source_embedding_node"]
    embedding_lowered = schematic.realization(embedding_src).lowered
    assert (
        schematic.residual_state("input").columns_for(embedding_lowered)
        == schematic.embedding_columns()
    )
    root_src = schematic.payload["graphs"]["source"]["root"]
    root_lowered = schematic.realization(root_src).lowered
    assert (
        schematic.residual_state("output").columns_for(root_lowered)
        == schematic.output_columns()
    )

    # Every owned column round-trips through the reverse lookup.
    for state in states:
        for node in state.nodes():
            for column in state.columns_for(node):
                assert schematic.column_owner(state.key, column) == node

    # The constant-1 column is the compiler's own: where a state
    # materializes it, the owner is a compiler-internal ref ("i:*"),
    # never a graph node.
    const_col = schematic.payload["residual_stream"]["constant_one_column"]
    for state in states:
        owner = state.node_at(const_col)
        assert owner is None or owner.startswith("i:")


def test_checkpoint_translation_roundtrip(tmp_path):
    _compile_region_bundle(tmp_path)
    schematic = load_schematic(tmp_path)

    q_slice = schematic.checkpoint_slice("L0.attn.W_Q")
    assert q_slice.transform == "transpose"
    assert q_slice.checkpoint_tensor.endswith("qkv_proj.weight")
    assert q_slice.scale == pytest.approx(4.0)  # sqrt(d_head=16)

    checked = 0
    for owner in schematic.placement_owners():
        for rect in schematic.placements(owner):
            row, col = rect.axis0.start, rect.axis1.start
            tensor, coord = schematic.to_checkpoint(rect.matrix, row, col)
            inverted = schematic.checkpoint_owner(tensor, coord)
            assert inverted is not None, (owner, rect)
            assert inverted[0] == owner
            checked += 1
    assert checked > 0

    # A coordinate outside every mapped region inverts to nothing.
    assert schematic.checkpoint_owner(q_slice.checkpoint_tensor, (10**6, 0)) is None


def test_realization_and_schedule_views(tmp_path):
    _compile_region_bundle(tmp_path)
    schematic = load_schematic(tmp_path)

    realizations = [schematic.realization(node.id) for node in schematic.source_nodes()]
    assert {view.status for view in realizations} == {"whole"}

    root_lowered = schematic.realization(
        schematic.payload["graphs"]["source"]["root"]
    ).lowered
    operations = schematic.operations_for(root_lowered)
    assert operations
    for operation in operations:
        raw_layer = schematic.payload["schedule"]["layers"][operation.layer]
        section = (
            "attention_operations"
            if operation.kind == "attention"
            else "mlp_operations"
        )
        raw = next(op for op in raw_layer[section] if op["id"] == operation.id)
        if raw.get("target_columns") is not None:
            decoded = tuple(
                column
                for start, length in raw["target_columns"]
                for column in range(start, start + length)
            )
            assert operation.target_columns == decoded


def test_reader_rejects_tampering(tmp_path):
    _compile_region_bundle(tmp_path)
    manifest_path = tmp_path / "torchwright_schematic.json"
    original = manifest_path.read_text()

    # Any in-place edit trips the integrity hash.
    payload = json.loads(original)
    payload["model"]["n_layers"] = 99
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(SchematicValidationError, match="integrity hash mismatch"):
        load_schematic(tmp_path)

    # Recomputing the integrity hash still trips the source content hash.
    payload = json.loads(original)
    payload["graphs"]["source"]["semantic_regions"][0]["params"] = {"x": 1}
    payload["integrity"]["sha256"] = sha256_json(
        {key: value for key, value in payload.items() if key != "integrity"}
    )
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(SchematicValidationError, match="source graph hash mismatch"):
        load_schematic(tmp_path)

    manifest_path.write_text(original)
    load_schematic(tmp_path)  # restored bundle loads clean

    # Bundle tier: a truncated bound file fails the always-on size check.
    schema_path = tmp_path / "torchwright_schematic_v1.schema.json"
    schema_text = schema_path.read_text()
    schema_path.write_text(schema_text + "\n")
    with pytest.raises(SchematicValidationError, match="size mismatch"):
        load_schematic_bundle(tmp_path)
    schema_path.write_text(schema_text)

    # A same-size byte flip passes the default tier and fails verify_files.
    shard_name = max(
        json.loads(original)["artifact"]["files"],
        key=lambda name: json.loads(original)["artifact"]["files"][name]["bytes"],
    )
    shard_path = tmp_path / shard_name
    blob = bytearray(shard_path.read_bytes())
    blob[-1] ^= 0xFF
    shard_path.write_bytes(bytes(blob))
    load_schematic_bundle(tmp_path)  # sizes-only tier does not notice
    with pytest.raises(SchematicValidationError, match="hash mismatch"):
        load_schematic_bundle(tmp_path, verify_files=True)


def test_support_matches_shard_nonzeros(tmp_path):
    _compile_region_bundle(tmp_path)
    bundle = load_schematic_bundle(tmp_path)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    support = bundle.manifest.payload["parameter_support"]
    for tensor_name, record in support["tensors"].items():
        shard = tmp_path / index["weight_map"][tensor_name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(tensor_name)
        expected = {
            tuple(int(value) for value in coord)
            for coord in torch.nonzero(tensor, as_tuple=False).tolist()
        }
        assert bundle.support_coordinates(tensor_name) == expected
        assert record["nnz"] == len(expected)
