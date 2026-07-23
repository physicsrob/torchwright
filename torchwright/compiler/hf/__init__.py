"""Native HuggingFace ``transformers`` surface for compiled torchwright token
models — the torch-native counterpart to ``compiler/onnx_load.py``'s ONNX
runtime loaders.

A compiled torchwright token graph targets stock ``Phi3ForCausalLM`` by
default: SwiGLU, biasless projections, RMSNorm, and ordinary
``AutoModelForCausalLM`` loading without remote code. The custom ReLU model is
an explicit ``architecture="custom"`` escape hatch.  Either way the compiled
token head is tied (token.v6): one table serves lookup and unembed, with the
output written into the embedding's exact ordered residual bank.

- :class:`TorchwrightCustomConfig` / :class:`TorchwrightCustomForCausalLM` —
  the explicitly requested custom config + model.
- :class:`TorchwrightCustomTokenizer` — a generic character-level tokenizer over a
  model's vocab (DOOM ships its own richer tokenizer).
- :func:`compile_to_hf` and :func:`compile_hf_bundle` — direct build-time
  sinks over the compiler's backend-neutral streaming weight records.
"""

from .build import (
    CompileProfile,
    HFArchitecture,
    HFBundleReport,
    ScheduleProvenance,
    build_fast_tokenizer,
    compile_hf_bundle,
    compile_to_hf,
    save_hf_bundle,
)
from .configuration_torchwright_custom import TorchwrightCustomConfig
from .modeling_torchwright_custom import (
    TorchwrightCustomForCausalLM,
    TorchwrightCustomModel,
    TorchwrightCustomPreTrainedModel,
)
from .tokenization_torchwright_custom import TorchwrightCustomTokenizer

__all__ = [
    "CompileProfile",
    "HFArchitecture",
    "HFBundleReport",
    "ScheduleProvenance",
    "TorchwrightCustomConfig",
    "TorchwrightCustomForCausalLM",
    "TorchwrightCustomModel",
    "TorchwrightCustomPreTrainedModel",
    "TorchwrightCustomTokenizer",
    "build_fast_tokenizer",
    "compile_hf_bundle",
    "compile_to_hf",
    "save_hf_bundle",
]
