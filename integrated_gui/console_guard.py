"""Keep Dash / workers alive when PyVista/VTK sends console Ctrl+C on Windows."""
from __future__ import annotations

import os
import sys


def install_windows_ctrl_guard(*, detach_console: bool = False) -> bool:
    """
    Swallow CTRL_C / CTRL_BREAK delivered to this process.

    On Windows, interactive VTK/PyVista children often broadcast a console
    control event that makes ``cmd`` show ``Terminate batch job (Y/N)?`` and
    kills workers mid-run. Returning True from the handler blocks that.

    If ``detach_console`` is True (worker processes), also call FreeConsole so
    later Ctrl+C events on the parent console are not delivered here.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        @HandlerRoutine
        def _handler(ctrl_type: int) -> bool:
            # 0=CTRL_C_EVENT, 1=CTRL_BREAK_EVENT
            if ctrl_type in (0, 1):
                try:
                    sys.stderr.write(
                        "\n[gui] Ignored console Ctrl+C/Break "
                        "(likely from a 3D viewer). Process still running.\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                return True
            return False

        # Keep a module-level ref so the callback is not GC'd
        install_windows_ctrl_guard._handler = _handler  # type: ignore[attr-defined]
        if not ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler, True):
            return False
        if detach_console:
            try:
                ctypes.windll.kernel32.FreeConsole()
            except Exception:
                pass
        return True
    except Exception:
        return False
