"""Force non-GUI backends so Dash callbacks never touch Tk/Qt from worker threads."""
from __future__ import annotations

import os


def configure_headless_plotting() -> None:
    """
    Must run before matplotlib.pyplot / pyvista GUI usage.

    Dash serves callbacks on worker threads. TkAgg (matplotlib default on Windows)
    and on-screen PyVista plotters cause:
      RuntimeError: main thread is not in main loop
      Tcl_AsyncDelete: async handler deleted by the wrong thread
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    # Prefer EGL/OSMesa-free Windows path: VTK still works off-screen via Win32
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass

    try:
        import pyvista as pv

        pv.OFF_SCREEN = True
        # Avoid theme/plotter init opening a window
        try:
            pv.global_theme.interactive = False
        except Exception:
            pass
    except Exception:
        pass
