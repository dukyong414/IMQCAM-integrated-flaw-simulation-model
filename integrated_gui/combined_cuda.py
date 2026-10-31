"""CUDA / GPU helpers for Combined plate pore placement."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def cuda_sampling_available() -> bool:
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def sample_points_in_builds(
    builds: Sequence,
    counts: np.ndarray,
    seed: int,
    *,
    prefer_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Sample uniform points inside each build AABB.

    Returns (xyz[n,3], build_index[n], backend_label).
    Uses Numba-CUDA when prefer_gpu and CUDA is available; otherwise NumPy.
    """
    counts = np.asarray(counts, dtype=np.int64).ravel()
    if len(builds) == 0 or int(counts.sum()) <= 0:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=np.int64), "NumPy CPU"

    if len(counts) != len(builds):
        raise ValueError("counts length must match builds")

    build_idx = np.repeat(np.arange(len(builds), dtype=np.int64), counts)
    n = int(build_idx.size)

    cx = np.fromiter((float(b.cx_mm) for b in builds), dtype=np.float64, count=len(builds))
    cy = np.fromiter((float(b.cy_mm) for b in builds), dtype=np.float64, count=len(builds))
    hx = 0.5 * np.fromiter((float(b.length_mm) for b in builds), dtype=np.float64, count=len(builds))
    hy = 0.5 * np.fromiter((float(b.width_mm) for b in builds), dtype=np.float64, count=len(builds))
    hz = np.fromiter((float(b.height_mm) for b in builds), dtype=np.float64, count=len(builds))
    x0 = (cx - hx)[build_idx]
    x1 = (cx + hx)[build_idx]
    y0 = (cy - hy)[build_idx]
    y1 = (cy + hy)[build_idx]
    z0 = np.zeros(n, dtype=np.float64)
    z1 = hz[build_idx]

    if prefer_gpu:
        try:
            xyz = _sample_cuda(x0, x1, y0, y1, z0, z1, int(seed))
            return xyz, build_idx, "CUDA GPU"
        except Exception:
            pass
        try:
            xyz = _sample_cupy(x0, x1, y0, y1, z0, z1, int(seed))
            return xyz, build_idx, "CuPy GPU"
        except Exception:
            pass

    return _sample_numpy(x0, x1, y0, y1, z0, z1, int(seed)), build_idx, "NumPy CPU"


def _sample_numpy(x0, x1, y0, y1, z0, z1, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(len(x0))
    u = rng.random((n, 3))
    xyz = np.empty((n, 3), dtype=float)
    xyz[:, 0] = x0 + u[:, 0] * (x1 - x0)
    xyz[:, 1] = y0 + u[:, 1] * (y1 - y0)
    xyz[:, 2] = z0 + u[:, 2] * (z1 - z0)
    return xyz


def _sample_cupy(x0, x1, y0, y1, z0, z1, seed: int) -> np.ndarray:
    import cupy as cp

    rs = cp.random.RandomState(seed)
    n = len(x0)
    u = rs.random_sample((n, 3), dtype=cp.float64)
    x0g, x1g = cp.asarray(x0), cp.asarray(x1)
    y0g, y1g = cp.asarray(y0), cp.asarray(y1)
    z0g, z1g = cp.asarray(z0), cp.asarray(z1)
    xyz = cp.empty((n, 3), dtype=cp.float64)
    xyz[:, 0] = x0g + u[:, 0] * (x1g - x0g)
    xyz[:, 1] = y0g + u[:, 1] * (y1g - y0g)
    xyz[:, 2] = z0g + u[:, 2] * (z1g - z0g)
    return cp.asnumpy(xyz)


_CUDA_KERNEL = None


def _get_cuda_kernel():
    global _CUDA_KERNEL
    if _CUDA_KERNEL is not None:
        return _CUDA_KERNEL

    from numba import cuda, float64, uint64

    @cuda.jit
    def _kernel(out, x0a, x1a, y0a, y1a, z0a, z1a, seed64):
        i = cuda.grid(1)
        if i >= out.shape[0]:
            return
        state = seed64 ^ (uint64(i) * uint64(0x9E3779B97F4A7C15))
        state = state * uint64(6364136223846793005) + uint64(1)
        u0 = float64((state >> uint64(11)) & uint64(0x1FFFFFFFFFFFFF)) / float64(9007199254740992.0)
        state = state * uint64(6364136223846793005) + uint64(1)
        u1 = float64((state >> uint64(11)) & uint64(0x1FFFFFFFFFFFFF)) / float64(9007199254740992.0)
        state = state * uint64(6364136223846793005) + uint64(1)
        u2 = float64((state >> uint64(11)) & uint64(0x1FFFFFFFFFFFFF)) / float64(9007199254740992.0)
        out[i, 0] = x0a[i] + u0 * (x1a[i] - x0a[i])
        out[i, 1] = y0a[i] + u1 * (y1a[i] - y0a[i])
        out[i, 2] = z0a[i] + u2 * (z1a[i] - z0a[i])

    _CUDA_KERNEL = _kernel
    return _kernel


def _sample_cuda(x0, x1, y0, y1, z0, z1, seed: int) -> np.ndarray:
    from numba import cuda, uint64

    if not cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    n = int(len(x0))
    xyz = np.empty((n, 3), dtype=np.float64)
    kernel = _get_cuda_kernel()
    threads = 256
    blocks = max(1, (n + threads - 1) // threads)
    d_out = cuda.to_device(xyz)
    kernel[blocks, threads](
        d_out,
        cuda.to_device(np.ascontiguousarray(x0)),
        cuda.to_device(np.ascontiguousarray(x1)),
        cuda.to_device(np.ascontiguousarray(y0)),
        cuda.to_device(np.ascontiguousarray(y1)),
        cuda.to_device(np.ascontiguousarray(z0)),
        cuda.to_device(np.ascontiguousarray(z1)),
        uint64(int(seed) & 0xFFFFFFFFFFFFFFFF),
    )
    d_out.copy_to_host(xyz)
    return xyz
