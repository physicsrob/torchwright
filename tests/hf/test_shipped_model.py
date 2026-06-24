"""The shipped HF files are import-clean.

``configuration_torchwright.py`` / ``modeling_torchwright.py`` /
``tokenization_torchwright.py`` are copied verbatim into every saved model
directory (by ``transformers``' ``custom_object_save``) and loaded on a
stranger's machine via ``trust_remote_code`` with only ``torch`` +
``transformers`` installed — no ``torchwright``. That holds iff each file
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

import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

_HF_DIR = (
    Path(__file__).resolve().parents[2]
    / "torchwright"
    / "compiler"
    / "hf"
)

# (filename, extra non-relative imports allowed beyond stdlib, allowed relative imports)
_SHIPPED = [
    ("configuration_torchwright.py", {"transformers"}, []),
    ("tokenization_torchwright.py", {"transformers"}, []),
    # The model file may import torch and exactly one sibling: its config.
    ("modeling_torchwright.py", {"transformers", "torch"}, ["configuration_torchwright"]),
]


@pytest.mark.parametrize("filename,allowed_extra,allowed_rel", _SHIPPED)
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


def test_convert_is_not_shipped_but_imports_onnx():
    """``convert.py`` is a build-time tool, not shipped — it may import onnx.

    This pins the layering: the converter (which reads the ONNX artifact) is
    explicitly the one file in the package allowed to depend on ``onnx``, so a
    future accidental ``import onnx`` in a *shipped* file is caught by the test
    above while the converter's legitimate use stays put.
    """
    from transformers.dynamic_module_utils import get_imports

    source = _HF_DIR / "convert.py"
    imports = set(get_imports(str(source)))
    assert "onnx" in imports, "convert.py is expected to read ONNX artifacts"
