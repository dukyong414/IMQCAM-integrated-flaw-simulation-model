"""Lightweight host specs + memory usage for the GUI header."""
from __future__ import annotations

import os
import platform
import time
from typing import Any, Dict, Optional


def _bytes_gb(n: float) -> float:
    return float(n) / (1024.0 ** 3)


def _memory_windows() -> Optional[Dict[str, float]]:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        total = float(stat.ullTotalPhys)
        avail = float(stat.ullAvailPhys)
        used = max(0.0, total - avail)
        pct = float(stat.dwMemoryLoad)
        return {"total_gb": _bytes_gb(total), "used_gb": _bytes_gb(used), "avail_gb": _bytes_gb(avail), "pct": pct}
    except Exception:
        return None


def _memory_psutil() -> Optional[Dict[str, float]]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_gb": _bytes_gb(vm.total),
            "used_gb": _bytes_gb(vm.used),
            "avail_gb": _bytes_gb(vm.available),
            "pct": float(vm.percent),
        }
    except Exception:
        return None


def _cpu_info() -> str:
    try:
        import psutil

        phys = psutil.cpu_count(logical=False) or "?"
        logi = psutil.cpu_count(logical=True) or "?"
        freq = psutil.cpu_freq()
        mhz = f"{freq.current:.0f} MHz" if freq and freq.current else ""
        name = platform.processor() or "CPU"
        if len(name) > 48:
            name = name[:45] + "…"
        bits = f"{phys}P/{logi}L"
        return f"{name} · {bits}" + (f" · {mhz}" if mhz else "")
    except Exception:
        name = platform.processor() or platform.machine() or "CPU"
        if len(name) > 56:
            name = name[:53] + "…"
        n = os.cpu_count() or "?"
        return f"{name} · {n} logical"


def _gpu_info() -> str:
    try:
        from .compute_backend import cuda_available, gpu_name

        if not cuda_available():
            return "GPU off"
        name = gpu_name() or "CUDA"
        if len(name) > 36:
            name = name[:33] + "…"
        return f"GPU {name}"
    except Exception:
        return "GPU ?"


def host_snapshot() -> Dict[str, Any]:
    """Return a dict suitable for the header system chip."""
    mem = _memory_psutil() or (_memory_windows() if os.name == "nt" else None)
    snap = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": _cpu_info(),
        "gpu": _gpu_info(),
        "ts": time.time(),
    }
    if mem:
        snap.update(mem)
    return snap


def format_sys_chip(snap: Optional[Dict[str, Any]] = None) -> str:
    s = snap or host_snapshot()
    mem = ""
    if "used_gb" in s and "total_gb" in s:
        mem = f" · RAM {s['used_gb']:.1f}/{s['total_gb']:.1f} GB ({s.get('pct', 0):.0f}%)"
    gpu = s.get("gpu") or "GPU ?"
    return f"{s.get('os', '?')} · {s.get('cpu', 'CPU')}{mem} · {gpu} · Py {s.get('python', '?')}"
