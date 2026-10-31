"""Spawn child processes without hijacking the parent batch/console (Windows)."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Mapping, Optional, Sequence


def _gui_python() -> str:
    """Prefer pythonw.exe on Windows so the child never owns/signals the console."""
    exe = sys.executable
    if os.name == "nt":
        base, name = os.path.split(exe)
        lower = name.lower()
        if lower == "python.exe":
            candidate = os.path.join(base, "pythonw.exe")
            if os.path.isfile(candidate):
                return candidate
        if lower == "pythonw.exe":
            return exe
    return exe


def spawn_detached(
    args: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    use_gui_python: bool = True,
    gui_window: bool = False,
    log_path: Optional[str] = None,
) -> subprocess.Popen:
    """
    Start a subprocess that cannot deliver Ctrl+C to ``run_gui.bat``.

    For interactive OpenGL viewers (``gui_window=True`` on Windows) use a
    brand-new console with ``python.exe`` — more reliable than DETACHED +
    pythonw, which often leaves the PyVista window invisible/stuck.
    """
    cmd = list(args)

    # Interactive 3D viewers: OpenGL window, no console prompt.
    # Prefer pythonw.exe (no console subsystem). Avoid CREATE_NEW_CONSOLE and
    # DETACHED_PROCESS — the former pops a cmd window; the latter can hide GL.
    if gui_window and os.name == "nt":
        if use_gui_python and cmd:
            if os.path.normcase(cmd[0]) == os.path.normcase(sys.executable):
                cmd[0] = _gui_python()
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        log_f = None
        if log_path:
            parent = os.path.dirname(log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            log_f = open(log_path, "w", encoding="utf-8")
            stdout = log_f
            stderr = subprocess.STDOUT
        flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        kwargs = {
            "cwd": cwd,
            "env": dict(env) if env is not None else None,
            "stdin": subprocess.DEVNULL,
            "stdout": stdout,
            "stderr": stderr,
            "close_fds": log_f is None,
            "creationflags": flags,
        }
        # python.exe still allocates a console unless CREATE_NO_WINDOW is set
        if os.path.basename(cmd[0]).lower() == "python.exe":
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            kwargs["creationflags"] = flags
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0  # SW_HIDE
            kwargs["startupinfo"] = si
        return subprocess.Popen(cmd, **kwargs)

    if use_gui_python and cmd:
        if os.path.normcase(cmd[0]) == os.path.normcase(sys.executable):
            cmd[0] = _gui_python()

    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    log_f = None
    if log_path:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8")
        stdout = log_f
        stderr = subprocess.STDOUT

    kwargs = {
        "cwd": cwd,
        "env": dict(env) if env is not None else None,
        "stdin": subprocess.DEVNULL,
        "stdout": stdout,
        "stderr": stderr,
        "close_fds": log_f is None,
    }
    if os.name == "nt":
        create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | create_no_window
        )
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = si

    return subprocess.Popen(cmd, **kwargs)
