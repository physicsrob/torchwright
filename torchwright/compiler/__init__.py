"""Public compiler entry points, loaded lazily by backend."""

__all__ = [
    "CompileProfile",
    "ContinuousHFBundleReport",
    "ContinuousIOSpec",
    "ContinuousRunner",
    "ContinuousValueSpec",
    "HFBundleReport",
    "ScheduleProvenance",
    "compile_continuous_hf_bundle",
    "compile_headless",
    "compile_hf_bundle",
    "compile_to_hf",
    "compile_to_onnx",
    "load_onnx",
    "save_hf_bundle",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        if name in {"CompileProfile", "ScheduleProvenance"}:
            from . import token_model

            return getattr(token_model, name)
        if name in {"compile_headless", "compile_to_onnx"}:
            from . import export

            return getattr(export, name)
        if name == "load_onnx":
            from . import onnx_load

            return onnx_load.load_onnx
        if name.startswith("Continuous") or name == "compile_continuous_hf_bundle":
            from .hf import continuous

            return getattr(continuous, name)
        from .hf import build

        return getattr(build, name)
    raise AttributeError(name)
