"""Subprocess worker: fit NU model once and write shared cache."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Status/log directory")
    parser.add_argument("--scaling", default="z_score")
    parser.add_argument("--smoothing", type=float, default=0.02)
    parser.add_argument("--homogenize-x", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    status_path = os.path.join(outdir, "status.json")
    log_path = os.path.join(outdir, "worker.log")
    logs = []

    def log(msg: str):
        logs.append(str(msg))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    def write_status(ok: bool, message: str, **extra):
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"ok": ok, "message": message, "log": "\n".join(logs), **extra}, f, indent=2)

    try:
        from integrated_gui.nu_cache import fit_and_cache, MODEL_PATH

        log("Fitting NU spatial model on default datasets (one-time pre-fit)…")
        model, path = fit_and_cache(
            scaling=args.scaling,
            smoothing=float(args.smoothing),
            homogenize_x=bool(args.homogenize_x),
        )
        log(f"Saved cache -> {path}")
        log(f"MODEL_PATH={MODEL_PATH}")
        write_status(True, "ok", model_path=path)
        print("WORKER_OK")
    except Exception:
        tb = traceback.format_exc()
        log(tb)
        write_status(False, "error", error=tb)
        print("WORKER_FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
