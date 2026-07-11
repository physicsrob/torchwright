"""Public compiler entry points, loaded lazily by backend."""

__all__ = [
    "CompileProfile",
    "HFBundleReport",
    "ScheduleProvenance",
    "compile_to_hf",
    "compile_hf_bundle",
    "save_hf_bundle",
]


def __getattr__(name):
    if name in __all__:
        if name in {"CompileProfile", "ScheduleProvenance"}:
            from . import token_model

            return getattr(token_model, name)
        from .hf import build

        return getattr(build, name)
    raise AttributeError(name)
