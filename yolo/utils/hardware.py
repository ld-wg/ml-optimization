#!/usr/bin/env python3
"""Minimal hardware helpers (deprecated).
Kept only for backward-compat: clear_gpu_cache() and optimize_mps_memory().
Prefer using the inline helpers in train.py.
"""

from __future__ import annotations


def clear_gpu_cache(device: str) -> None:
    try:
        import torch  # local import to avoid hard dep
        if device == "mps" and getattr(torch.backends, "mps", None):
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass


def optimize_mps_memory() -> bool:
    try:
        import torch  # local import
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            if hasattr(mps, "enable_mem_efficient_attention"):
                mps.enable_mem_efficient_attention(True)
            # if hasattr(mps, "set_memory_fraction"):
            #     mps.set_memory_fraction(0.8)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("medium")
            return True
    except Exception:
        pass
    return False
