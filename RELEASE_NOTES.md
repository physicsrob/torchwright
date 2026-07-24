# Unreleased

# 0.1.0 — 2026-07-23

First public release.

## Bundles ship a generation config

- `compile_hf_bundle` and `save_hf_bundle` now write
  `generation_config.json` (pad token aliased to eos), so `generate()`
  and `pipeline()` run without an explicit `pad_token_id`.

## Breaking: Hugging Face compilation is Phi-3-first

- `compile_to_hf` and `compile_hf_bundle` now default to the typed
  `CompileProfile.PHI3` contract: SwiGLU, biasless projections, RMSNorm, and a
  stock `Phi3ForCausalLM` bundle with no remote code.
- The former custom classes were renamed to `TorchwrightCustomConfig`,
  `TorchwrightCustomModel`, and `TorchwrightCustomForCausalLM`; their modules
  are now `configuration_torchwright_custom.py` and
  `modeling_torchwright_custom.py`.
- The custom tokenizer is now `TorchwrightCustomTokenizer` in
  `tokenization_torchwright_custom.py`.
- Custom model generation requires the explicit `architecture="custom"`
  profile and `trust_remote_code=True` when loading a published bundle.
- The ONNX-to-HF converter, `save_bundle(onnx_path, ...)`, and `read_vocab`
  were removed. Recompile source graphs directly to HF.
- The obsolete `calculator_v2` compatibility name was removed; use
  `examples.calculator_simple`.
- HF bundle builders now stage and validate a complete bundle before replacing
  the destination. A failed compile leaves an existing published bundle
  untouched, and successful recompilation removes stale files.
- Token compilation now requires exactly one `Embedding` reachable from the
  output and verifies that it is the same node supplied to the HF API.
- Schedule metadata distinguishes the emitted schedule and its certified
  optimality from the independent CP-SAT attempt that proposed a candidate.
- `compile_hf_bundle` returns an immutable `HFBundleReport` containing the
  published output directory, layer count, and typed selected-schedule
  provenance.

There are intentionally no compatibility aliases. Pin an older release to
load historical Python APIs, or recompile the source graph with the current
compiler.
