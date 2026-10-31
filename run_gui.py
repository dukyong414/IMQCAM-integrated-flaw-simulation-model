"""Start the Integrated Porosity GUI (replacement for run_gui.bat).

Run from anywhere:
    python run_gui.py            # or press Run in the editor
    python run_gui.py --port 8051 --no-browser
    python run_gui.py --stop     # stop a running GUI server

Always uses the project .venv, and stops any old GUI server still holding the
port so the page you open is the current code.

To stop: type q + Enter in this terminal (Ctrl+C is ignored on purpose so the
3D viewer windows can't kill the server), or run with --stop.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")


def _pids_on_port(port: int) -> set[int]:
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, check=False
        ).stdout
    except Exception:
        return set()
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3].upper() == "LISTENING":
            if re.search(rf":{port}$", parts[1]):
                pids.add(int(parts[4]))
    pids.discard(os.getpid())
    return pids


def _stop_old_servers(port: int) -> None:
    pids = _pids_on_port(port)
    for pid in pids:
        print(f"Stopping old server on port {port} (PID {pid})...")
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    if pids:
        for _ in range(20):
            if not _pids_on_port(port):
                break
            time.sleep(0.25)


def _watch_for_quit() -> None:
    """Background thread: 'q' + Enter on the console stops the server."""
    import threading

    def _loop():
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            if line.strip().lower() in ("q", "quit", "exit", "stop"):
                print("Stopping GUI...", flush=True)
                os._exit(0)

    threading.Thread(target=_loop, daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrated Porosity GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stop", action="store_true", help="stop a running GUI server and exit")
    args = parser.parse_args()

    if args.stop:
        if not _pids_on_port(args.port):
            print(f"No GUI server running on port {args.port}.")
        _stop_old_servers(args.port)
        return 0

    # Re-launch with the project venv if started from another interpreter
    if os.path.isfile(VENV_PY) and os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(VENV_PY):
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        return subprocess.call([VENV_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=HERE)
    if not os.path.isfile(VENV_PY):
        print(f"WARNING: project venv not found at {VENV_PY}; using {sys.executable}")

    os.chdir(HERE)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"

    _stop_old_servers(args.port)

    from integrated_cli.main import cmd_gui

    _watch_for_quit()
    print(f"Starting Integrated Porosity GUI at http://{args.host}:{args.port}/")
    print("To stop: type q and press Enter here (Ctrl+C is ignored).")
    return cmd_gui(args)


if __name__ == "__main__":
    sys.exit(main())
