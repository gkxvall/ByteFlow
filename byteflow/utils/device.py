from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """Return torch.device from config string. ``auto`` picks CUDA when available."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
