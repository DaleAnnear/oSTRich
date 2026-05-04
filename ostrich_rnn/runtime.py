"""Runtime/device helpers."""

from __future__ import annotations

import torch


def resolve_device(requested: str | torch.device | None = "auto") -> torch.device:
    """Resolve a user device request and fail clearly if CUDA is requested but absent."""

    if isinstance(requested, torch.device):
        return requested
    requested = requested or "auto"
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{requested}'. Use auto, cuda, or cpu.")
    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    total_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
    return f"cuda:{index} {name}, capability={capability[0]}.{capability[1]}, memory={total_gb:.1f} GB"
