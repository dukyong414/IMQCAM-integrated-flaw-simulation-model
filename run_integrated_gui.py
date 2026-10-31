"""Launch the integrated CMU + NU porosity GUI."""
from __future__ import annotations

import argparse
import os
import sys


def main():
    # Avoid Windows console UnicodeEncodeError (cp949) from scientific symbols in logs.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Integrated CMU + NU Porosity GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port (default 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()

    from integrated_gui.app import launch

    launch(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
