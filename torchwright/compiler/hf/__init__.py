"""Native HuggingFace ``transformers`` surface for compiled torchwright token
models — the torch-native counterpart to ``compiler/onnx_load.py``'s ONNX
runtime loaders.

A compiled torchwright token artifact is a bona-fide standard transformer: a
``token`` head (one table tied between lookup and unembed) over a uniform-width residual
stream with causal ``scale=1.0`` attention, ``relu`` MLP blocks, and
(optionally) identity RMSNorms. This package reimplements that exact forward
path as real ``nn.Module``s so the artifact loads as an ordinary
``AutoModelForCausalLM`` and runs under stock ``generate``.

- :class:`TorchwrightConfig` / :class:`TorchwrightForCausalLM` — the shipped
  config + model (torch/transformers only; safe to publish with
  ``trust_remote_code``).
- :class:`TorchwrightTokenizer` — a generic character-level tokenizer over a
  model's vocab (DOOM ships its own richer tokenizer).
- ``convert`` (imported explicitly: ``from torchwright.compiler.hf.convert
  import ...``) — the build-time ONNX → safetensors converter. Kept out of this
  ``__init__`` because it imports ``onnx``; loading a published model never
  needs it.
"""

from .configuration_torchwright import TorchwrightConfig
from .modeling_torchwright import (
    TorchwrightForCausalLM,
    TorchwrightModel,
    TorchwrightPreTrainedModel,
)
from .tokenization_torchwright import TorchwrightTokenizer

__all__ = [
    "TorchwrightConfig",
    "TorchwrightModel",
    "TorchwrightPreTrainedModel",
    "TorchwrightForCausalLM",
    "TorchwrightTokenizer",
]
