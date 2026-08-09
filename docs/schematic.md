# The schematic

Every HF bundle torchwright compiles ships `torchwright_schematic.json`
(format `torchwright.schematic.v1`): a hash-bound record of exactly what
the artifact is — the source graph with its semantic regions, the
lowered graph, the schedule, the residual-stream layout, the physical
weight placements in trimmed checkpoint coordinates, and the token-I/O
contract.  It is captured during the compile, while the compiler still
owns the lowering boundary and the replay plan, and it is bound by hash
to the exact shard files it ships beside.  Two sidecars travel with it:
`torchwright_schematic_v1.schema.json` (the normative section
inventory — its `required` list is what validation enforces) and
`torchwright_schematic_support.npz` (per-tensor nonzero coordinates).

## Getting one

Compile locally (`compile_hf_bundle` writes all three files), or pull
just the manifest from a published bundle — it is one small file even
when the checkpoint is 100 GB.  `huggingface_hub` is not a torchwright
dependency; bring your own:

```python
from huggingface_hub import hf_hub_download, snapshot_download

path = hf_hub_download(repo_id, "torchwright_schematic.json")

from torchwright import load_schematic
schematic = load_schematic(path)

# Bundle-aware (support archive, file binding): download everything.
from torchwright import load_schematic_bundle
bundle = load_schematic_bundle(snapshot_download(repo_id))
```

Importing `torchwright.schematic` (and calling `load_schematic`) never
imports torch; numpy loads only when a support archive is opened.

## Reader quickstart

```python
schematic = load_schematic("torchwright_schematic.json")

root = schematic.payload["graphs"]["source"]["root"]
for region in schematic.region_chain(root):      # innermost first
    print(region.op, region.params)

members = schematic.region_members("r:0")        # whole-subtree membership
state = schematic.residual_state("L3.post")
cols = state.columns_for("l:7")                  # ordered component columns
owner = schematic.column_owner("L3.post", 129)   # reverse lookup

cp = schematic.checkpoint_slice("L0.attn.W_Q")   # tensor, transpose, scale
tensor, coord = schematic.to_checkpoint("L0.attn.W_Q", row=5, col=3)
schematic.checkpoint_owner(tensor, coord)        # -> (owner, PlacementRect)

bundle.support_coordinates(cp.checkpoint_tensor) # exact nonzero set
```

## Format tour

**Sections.**  `format` and `integrity` frame the manifest; `artifact`
binds the bundle files; `build` records compiler provenance and
options; `model` is the transformer's shape; `token_io` the
embedding/readout contract; `graphs` the source (with
`semantic_regions`) and lowered graphs plus the realization map;
`schedule` the per-layer operations; `residual_stream` the per-state
column ownership; `physical_layout` matrices, placements, and the
checkpoint parameter map; `parameter_support` points at the npz;
`observability` maps states to TransformerLens hooks;
`runtime_contract` and `intervention_contract` state what the numbers
guarantee; `reference_fixtures` is reserved for input corpora.

**Id namespaces.**  `s:<n>` source nodes (canonical preorder-DFS order),
`l:<n>` lowered nodes, `i:<n>` compiler-internal nodes (e.g. the
constant-1 column's owner), `r:<n>` semantic regions.  Placement owners
are node refs or `physical:<op_type>`.  Schedule and residual-stream
references resolve against lowered + internal ids; regions and the
realization map resolve against source ids.

**The run encoding is ordered.**  Column lists are run-length encoded
as `[start, length]` runs, and column order is meaningful — column *k*
holds component *k* of the node's value — so runs merge only
consecutive ascending indices and decoding must never sort:
`[[5, 2], [3, 1]]` decodes to `[5, 6, 3]`.  A few fields are a single
bare `[start, length]` span, not a run list: attention operation
`heads` (zero length is legal — a fully trimmed op), placement
rectangle axes (`diagonal: true` marks a diagonal band), and
`checkpoint_rows`/`checkpoint_columns`/`checkpoint_indices`.

**Semantic regions.**  Each region is one op-library call: derived op
name (`linear.subtract`, `swiglu.map_select.select`), sanitized
non-node params, operand/result node refs, parent call.  Attribution
rules: a node's `region` is the innermost call that *created* it;
a node returned unchanged through an outer call appears in that outer
region's `results` but keeps its creator's membership; composition ops
create no node directly, so their members are reachable only through
child regions.  `region: null` marks nodes built outside any decorated
op.

**Hashing model.**  `integrity.sha256` covers every section except
itself; each graph record carries its own content `sha256`;
`artifact.files` binds every bundle file (basename → sha256 + bytes)
except the manifest itself.  All hashes use one canonical JSON
encoding (sorted keys, compact separators, raw unicode) — see
`torchwright.schematic.format.sha256_json`.

**What is deliberately absent.**  Vocabulary strings (only
`vocabulary_sha256`; the tokenizer files carry the vocab), and a
TransformerLens hook for the `output` residual state —
`observability` maps `input` and every `L{i}.post`, but the final
`output` state (the readout's `input_state`) has no hook.

## Validation tiers

| check | `load_schematic` | `load_schematic_bundle` | `verify_files=True` |
|---|---|---|---|
| format string + section inventory | ✓ | ✓ | ✓ |
| integrity + per-graph hashes | ✓ | ✓ | ✓ |
| internal reference resolution | ✓ | ✓ | ✓ |
| config.json pointer | | ✓ | ✓ |
| bound files present, byte sizes | | ✓ | ✓ |
| support npz structure | | ✓ | ✓ |
| full sha256 of every bound file | | | ✓ |

Everything raises `torchwright.schematic.SchematicValidationError`
naming the failing fact.  These are the same checks the builder runs
before a bundle leaves staging.
