from __future__ import annotations

from typing import Any

import torch


def cuda_mem_stats(device: torch.device | None = None) -> dict[str, Any] | None:
    """Return basic CUDA memory stats, or None if CUDA is not available."""
    if not torch.cuda.is_available():
        return None
    dev = device if device is not None and device.type == "cuda" else torch.device("cuda")
    torch.cuda.synchronize(dev)
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    return {"device": str(dev), "allocated_bytes": allocated, "reserved_bytes": reserved}


def format_cuda_mem_line(stats: dict[str, Any] | None) -> str:
    """Human-readable one-liner for logging."""
    if stats is None:
        return "CUDA memory: N/A (CPU-only)"
    mb_alloc = stats["allocated_bytes"] / (1024**2)
    mb_res = stats["reserved_bytes"] / (1024**2)
    return (
        f"CUDA memory ({stats['device']}): "
        f"allocated={mb_alloc:.1f} MiB, reserved={mb_res:.1f} MiB"
    )
