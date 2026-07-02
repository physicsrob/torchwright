# Block node spec (Phase 1 of `block_ir_step1_plan.md` — Gate B draft)

> **Historical note (2026-07):** the `Block` node described in this document
> has been renamed `FFN` (`torchwright/graph/ffn.py`), and
> `scripts/block_equivalence.py` is now `scripts/ffn_equivalence.py`.  Names
> below reflect the pre-rename vocabulary.

*One page. Written against the SwiGLU end-state (`ops_plain_english.md`);
step 1 instantiates only the degenerate ReLU form. Gate A rulings apply:
no export-raw-lanes, ReLU is block-internal, and no raw chain survives —
originally asserted by the `blockify` pass, now by the lowering boundary
(`compiler.lower`, which absorbed and deleted it).*

## Semantics

A **Block** is a graph node computing, for input row `x` (width `d_input`):

    lane_out[j] = act(gate_proj[j] · x + gate_bias[j]) * up[j](x)
    output      = lane_out @ out_proj + out_bias

- `up[j](x)` is either the **constant 1** (degenerate lane — step 1's only
  form) or `up_proj[j] · x + up_bias[j]` (gated lane — step 2).
- `act` is a per-block field: `"relu"` (step 1) or `"swish"` (step 2).
- Lane kind is uniform per block (all degenerate or all gated); ops needing
  both build two blocks. Revisit only if a real op needs mixing.

Fields: `input: Node` (single node; multi-input via `Concatenate`, same
convention as `Linear`), `gate_proj (n_lanes, d_input)`, `gate_bias
(n_lanes,)`, `up_proj/up_bias` (same shapes, `None` in step 1), `out_proj
(n_lanes, d_output)`, `out_bias (d_output,)`, `activation: str`.

`d_output = out_proj.shape[1]`; `len(block) = d_output`.

## Node-protocol obligations

- `compute(n_pos, input_values)`: the exact math above (recursive oracle /
  `reference_eval`).
- `compute_value_type()`: default `NodeValueType()` initially; semantic
  overrides stay op-level as today.
- `graph/affine_rules.py`: `_block_rule` — the linear rule through
  `gate_proj`, the existing ReLU envelope case analysis per lane, then the
  linear rule through `out_proj`. Step 1 composes the three existing rules;
  a swish envelope is step 2's problem.
- Assert/DebugWatch wrap the Block **output** like any node. Lane internals
  are not graph nodes, not probeable (`debug_value` parity: a chain ReLU
  already returns `None` today), and carry no Asserts (Gate A).
- Canonical-id / ONNX debug sidecar / rebuild fingerprint: Block
  participates like any node type; fingerprints will change — expected.

## Invariants (stated, enforced at the lowering boundary/construction)

1. **A Block is a packable unit, not a sublayer.** The scheduler bins many
   blocks' lanes into one MLP sublayer's hidden pool (`d_hidden`). Nothing
   in the node promises sublayer ownership.
2. **A Block is realized whole** — all lanes and the output projection in
   one sublayer. No splitting, no raw-lane export (Gate A).
3. **The Block owns its input projection.** Blockify asserts the mined L1
   is exclusive (sole consumer = the chain ReLU); a shared upstream value is
   represented as a separate `Linear` feeding the Block.
4. Width bookkeeping: a Block consumes `n_lanes` hidden slots (degenerate)
   — step 2 gated lanes still consume `n_lanes` slots but two projection
   rows each; the slot cost is a lane-count property, not a projection
   property.

## Builder API

- `linear_relu_linear(input_node, input_proj, input_bias, output_proj,
  output_bias, name="")` keeps its exact signature and returns a Block
  (degenerate lanes, `act="relu"`). Ops don't change.
- New explicit builder for step 2 (name TBD at Gate B for step 2, not now):
  constructs gated blocks from (gate rows, up rows, out projection).

## Compiler contract

- Scheduler MLP op: `compute_block` replaces `compute_relu`; emission logic
  unchanged in shape (allocate `d_output` residual cols, claim `n_lanes`
  hidden slots, capture input source cols).
- Writer: `_write_compute_block` = today's `_write_compute_relu` reading
  `gate_proj`/`out_proj` from one node instead of three. Step 1 targets the
  existing linear→relu→linear MLP module unchanged. (Step 2 changes the
  compiled module/ONNX architecture to a gated FFN — the lane interface
  above is the contract that change plugs into; out of scope here.)
- Blockify (2a): relocated `_detect_chains_static` over the post-fusion
  graph → Block emission + invariant 3 assert + assert-no-internal-Asserts.
- Fusion (2c): fold an upstream Linear into `gate_proj` (and `up_proj` when
  present); fold `out_proj` into a downstream Linear. Ejection machinery
  deleted (Gate A) — an ejecting rewrite is not legal against invariant 2.
- CP-SAT: Block is MLP-locked, never flex; `is_flex` reduces to "standalone
  Linear" exactly (non-exclusive-L1 arm deleted).

## Deletions this spec licenses (per Gate A)

`_detect_chains` (scheduler), `_detect_chains_static` + `Chain` (CP-SAT),
`_is_chain_internal_relu` (sibling_clusters), `compute_standalone_relu`,
the L2-standalone split path, non-exclusive-L1 dual realization,
`_ejected_relu` / `eject_budget` / `skip_relu_ejecting`, and the public
`ReLU` node type once no op emits it (final cleanup, phase 6).
