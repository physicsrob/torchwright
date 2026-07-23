"""The shipped HF files are import-clean.

``configuration_torchwright_custom.py`` / ``modeling_torchwright_custom.py`` /
``tokenization_torchwright_custom.py`` are copied verbatim into an explicitly
custom saved model directory (by ``transformers``' ``custom_object_save``)
and loaded on a stranger's machine via ``trust_remote_code`` with only
``torch`` + ``transformers`` installed — no ``torchwright``. That holds iff
each file
imports only the standard library, ``transformers`` (and ``torch`` for the
model), and pulls in no ``torchwright`` / ``onnx`` / sibling module except the
one allowed relative import the model makes to its config.

This is the static analogue of the import scan ``transformers`` itself runs at
load time (mirrors ``torchwright_doom``'s ``test_shipped_tokenizer.py``). It
needs no ONNX artifact, no torch run — it just reads the source — so it runs in
cloud CI cheaply. A stray ``import torch`` in the config, or ``import onnx`` /
``from torchwright...`` anywhere, would pass every in-repo test (where those are
present) and fail only on the stranger's machine; this catches it here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

_HF_DIR = Path(__file__).resolve().parents[2] / "torchwright" / "compiler" / "hf"

# (filename, extra non-relative imports allowed beyond stdlib, allowed relative imports)
_SHIPPED = [
    ("configuration_torchwright_custom.py", {"transformers"}, []),
    ("tokenization_torchwright_custom.py", {"transformers"}, []),
    # The model file may import torch and exactly one sibling: its config.
    (
        "modeling_torchwright_custom.py",
        {"transformers", "torch"},
        ["configuration_torchwright_custom"],
    ),
]


@pytest.mark.parametrize(("filename", "allowed_extra", "allowed_rel"), _SHIPPED)
def test_shipped_file_imports_are_clean(filename, allowed_extra, allowed_rel):
    from transformers.dynamic_module_utils import get_imports, get_relative_imports

    source = _HF_DIR / filename
    assert source.exists(), f"missing shipped file {source}"

    rel = set(get_relative_imports(str(source)))
    assert rel <= set(allowed_rel), (
        f"{filename} has disallowed relative imports {sorted(rel - set(allowed_rel))}; "
        f"only {allowed_rel} may be referenced (and only the model->config one)"
    )

    allowed = set(sys.stdlib_module_names) | allowed_extra
    leaked = set(get_imports(str(source))) - allowed
    assert not leaked, (
        f"{filename} imports non-stdlib/non-allowed modules {sorted(leaked)}; "
        f"a shipped file must load with only {sorted(allowed_extra)} present"
    )


def test_direct_builder_does_not_import_onnx():
    """The build-time HF sink consumes compiler records, not ONNX artifacts."""
    from transformers.dynamic_module_utils import get_imports

    assert not (_HF_DIR / "convert.py").exists()
    source = _HF_DIR / "build.py"
    imports = set(get_imports(str(source)))
    assert "onnx" not in imports
    assert "onnxruntime" not in imports


def test_custom_architecture_is_explicit_and_renamed(tmp_path):
    from transformers import AutoModelForCausalLM

    from torchwright.compiler.hf import compile_hf_bundle
    from torchwright.graph.rope import rotary_offset_head
    from torchwright.ops.inout_nodes import create_onehot_embedding

    vocab = ["<bos>", "<eos>", "a"]
    embedding = create_onehot_embedding(vocab)
    output = rotary_offset_head(embedding, delta_pos=-1, d_qk=16)
    compile_hf_bundle(
        output,
        embedding,
        tmp_path,
        d=256,
        d_head=16,
        architecture="custom",
    )
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "torchwright_custom"
    assert config["architectures"] == ["TorchwrightCustomForCausalLM"]
    assert config["auto_map"]["AutoModelForCausalLM"].endswith(
        ".TorchwrightCustomForCausalLM"
    )
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True)
    assert type(loaded).__name__ == "TorchwrightCustomForCausalLM"
