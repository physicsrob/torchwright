"""Automatic device selection with informational display."""

import torch


class _State:
    displayed = False
    override: torch.device | None = None


_state = _State()


def set_device(device: torch.device | None) -> None:
    """Override the auto-detected device. Pass None to restore auto-detection."""
    _state.override = device
    _state.displayed = False


def get_device(*, verbose: bool = True) -> torch.device:
    """Auto-detect the best available torch device.

    Prints a one-line summary on first call (unless verbose=False).
    """
    if _state.override is not None:
        dev = _state.override
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    if verbose and not _state.displayed:
        if dev.type == "cuda":
            name = torch.cuda.get_device_name(dev)
            mem = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
            print(f"  Using {name} ({mem:.1f} GB)")
        else:
            print("  Using CPU (no CUDA GPU detected)")
        _state.displayed = True

    return dev
