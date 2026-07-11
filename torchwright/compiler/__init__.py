"""Public compiler entry points, loaded lazily by backend."""

__all__ = [
    "CompileProfile",
    "compile_to_hf",
    "compile_hf_bundle",
    "save_hf_bundle",
]


def __getattr__(name):
    if name in __all__:
        if name == "CompileProfile":
            from .token_model import CompileProfile

            return CompileProfile
        from .hf import build

        return getattr(build, name)
    raise AttributeError(name)
