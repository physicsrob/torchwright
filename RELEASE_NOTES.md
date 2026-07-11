# Unreleased

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

There are intentionally no compatibility aliases. Pin an older release to
load historical Python APIs, or recompile the source graph with the current
compiler.
