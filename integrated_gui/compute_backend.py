"""Compute device selection: auto / GPU / CPU-only friendly mode."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

DeviceMode = Literal["auto", "gpu", "cpu"]
ResolvedDevice = Literal["gpu", "cpu"]

_ENV_MODE = "NASA_COMPUTE_DEVICE"
_ENV_FORCE_CPU = "NASA_FORCE_CPU"


@dataclass(frozen=True)
class ComputeInfo:
    mode: DeviceMode
    device: ResolvedDevice
    cuda_available: bool
    gpu_name: str
    detail: str


def cuda_available() -> bool:
    if os.environ.get(_ENV_FORCE_CPU, "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def gpu_name() -> str:
    try:
        from numba import cuda

        if not cuda.is_available():
            return ""
        return str(cuda.get_current_device().name.decode("utf-8", errors="replace"))
    except Exception:
        try:
            from numba import cuda

            if cuda.is_available() and len(cuda.gpus):
                with cuda.gpus[0]:
                    return str(cuda.get_current_device().name.decode("utf-8", errors="replace"))
        except Exception:
            return ""
    return ""


def normalize_mode(mode: Optional[str]) -> DeviceMode:
    raw = (mode or os.environ.get(_ENV_MODE, "auto") or "auto").strip().lower()
    if raw in {"gpu", "cuda"}:
        return "gpu"
    if raw in {"cpu", "host"}:
        return "cpu"
    return "auto"


def resolve_device(mode: Optional[str] = None) -> ComputeInfo:
    """Resolve requested mode to a concrete device with a human-readable note."""
    m = normalize_mode(mode)
    has_cuda = cuda_available()
    name = gpu_name() if has_cuda else ""

    if m == "cpu":
        return ComputeInfo(
            mode=m,
            device="cpu",
            cuda_available=has_cuda,
            gpu_name=name,
            detail="CPU-only mode (Numba parallel)",
        )
    if m == "gpu":
        if has_cuda:
            return ComputeInfo(
                mode=m,
                device="gpu",
                cuda_available=True,
                gpu_name=name,
                detail=f"GPU mode ({name or 'CUDA'})",
            )
        return ComputeInfo(
            mode=m,
            device="cpu",
            cuda_available=False,
            gpu_name="",
            detail="GPU requested but unavailable — falling back to CPU",
        )
    # auto
    if has_cuda:
        return ComputeInfo(
            mode=m,
            device="gpu",
            cuda_available=True,
            gpu_name=name,
            detail=f"Auto -> GPU ({name or 'CUDA'})",
        )
    return ComputeInfo(
        mode=m,
        device="cpu",
        cuda_available=False,
        gpu_name="",
        detail="Auto -> CPU (no CUDA device)",
    )


def apply_env(info: ComputeInfo) -> None:
    """Propagate resolved device into process env for workers/kernels."""
    os.environ[_ENV_MODE] = info.mode
    os.environ["RAPTOR_COMPUTE_DEVICE"] = info.device
    if info.device == "cpu":
        os.environ[_ENV_FORCE_CPU] = "1"
    else:
        os.environ.pop(_ENV_FORCE_CPU, None)


def numpy_module(prefer_gpu: bool = False):
    """Return cupy if GPU preferred and importable, else numpy."""
    if prefer_gpu:
        try:
            import cupy as cp  # type: ignore

            # Touch a device so import failures surface early
            _ = cp.cuda.runtime.getDeviceCount()
            return cp
        except Exception:
            pass
    import numpy as np

    return np
