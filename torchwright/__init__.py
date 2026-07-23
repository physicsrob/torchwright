"""torchwright: a compiler from computation graphs to transformer weights.

The four compile/load entry points are re-exported here, loaded lazily so
``import torchwright`` stays light and the base install never pulls the
optional ``transformers`` dependency.  Op libraries are deliberately not
re-exported: the import path (``torchwright.ops.relu`` vs
``torchwright.ops.swiglu``) is the activation choice.
"""

__all__ = [
    "compile_headless",
    "compile_hf_bundle",
    "compile_to_onnx",
    "load_onnx",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from . import compiler

        return getattr(compiler, name)
    raise AttributeError(name)
