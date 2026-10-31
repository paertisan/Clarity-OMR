"""Pick the best available torch device: CUDA, then Apple Silicon (MPS), then CPU."""

from __future__ import annotations


def default_device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
