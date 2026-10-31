"""Shared on-disk cache for the fitted NU spatial model.

Combined runs load this pickle instead of re-fitting every time.
Fit once via Welcome / Combined “Pre-fit NU” or the NU tab / CLI.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(os.path.dirname(__file__), "assets", "nu_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "nu_model.pkl")
META_PATH = os.path.join(CACHE_DIR, "nu_model_meta.json")


def ensure_cache_dir() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def cache_exists() -> bool:
    return os.path.isfile(MODEL_PATH)


def cache_status() -> Dict[str, Any]:
    """Human-readable cache status for the UI."""
    if not cache_exists():
        return {
            "ready": False,
            "path": MODEL_PATH,
            "message": "NU model not pre-fitted yet — click Pre-fit NU (once).",
            "mtime": None,
            "mtime_label": None,
        }
    mtime = os.path.getmtime(MODEL_PATH)
    label = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    meta = {}
    if os.path.isfile(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    n_ds = meta.get("n_datasets")
    extra = f" · {n_ds} dataset(s)" if n_ds is not None else ""
    return {
        "ready": True,
        "path": MODEL_PATH,
        "message": f"NU cache ready ({label}{extra})",
        "mtime": mtime,
        "mtime_label": label,
        "meta": meta,
    }


def save_fitted_model(model, *, meta: Optional[Dict[str, Any]] = None) -> str:
    ensure_cache_dir()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    payload = {
        "saved_at": time.time(),
        "saved_at_iso": datetime.now().isoformat(timespec="seconds"),
        **(meta or {}),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return MODEL_PATH


def load_fitted_model(path: Optional[str] = None):
    p = path or MODEL_PATH
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def fit_default_model(
    scaling: str = "z_score",
    smoothing: float = 0.02,
    homogenize_x: bool = False,
    datasets=None,
):
    """Fit on DEFAULT_DATASETS (or provided configs) and return the model."""
    from integrated_gui import nu_panel

    NU_DIR = os.path.join(ROOT, "NU team")
    if NU_DIR not in sys.path:
        sys.path.insert(0, NU_DIR)
    from porosity_field_model import PorosityPatternModel, SizeFilteredPorosityDataset

    configs = list(datasets if datasets is not None else nu_panel.DEFAULT_DATASETS)
    model = PorosityPatternModel(
        scaling=scaling,
        smoothing=float(smoothing),
        homogenize_x=bool(homogenize_x),
    )
    for cfg in configs:
        if not os.path.isdir(cfg["base_dir"]):
            raise FileNotFoundError(f"NU dataset missing: {cfg['base_dir']}")
        ignore = set(
            int(x)
            for x in str(cfg.get("ignore_indices", "")).split(",")
            if x.strip().isdigit()
        )
        ds = SizeFilteredPorosityDataset(
            title=cfg["title"],
            base_dir=cfg["base_dir"],
            folder_prefix=cfg["folder_prefix"],
            ignore_indices=ignore,
            start_idx=int(cfg["start_idx"]),
            cols=int(cfg["cols"]),
            index_pattern=re.compile(cfg["index_pattern"], re.I),
            core_pitch_mm=float(cfg["core_pitch_mm"]),
            plate_width_mm=float(cfg["plate_width_mm"]),
            plate_height_mm=float(cfg["plate_height_mm"]),
        )
        model.add_dataset(ds)
    model.fit()
    return model, configs


def fit_and_cache(
    scaling: str = "z_score",
    smoothing: float = 0.02,
    homogenize_x: bool = False,
    datasets=None,
) -> Tuple[Any, str]:
    model, configs = fit_default_model(
        scaling=scaling,
        smoothing=smoothing,
        homogenize_x=homogenize_x,
        datasets=datasets,
    )
    path = save_fitted_model(
        model,
        meta={
            "n_datasets": len(configs),
            "titles": [c.get("title") for c in configs],
            "scaling": scaling,
            "smoothing": smoothing,
            "homogenize_x": homogenize_x,
        },
    )
    # Keep in-process cache warm for Dash NU tab
    try:
        from integrated_gui import nu_panel

        nu_panel._FITTED_MODEL = model
    except Exception:
        pass
    return model, path


def run_prefit_subprocess(
    *,
    scaling: str = "z_score",
    smoothing: float = 0.02,
    homogenize_x: bool = False,
    timeout_s: int = 900,
) -> Dict[str, Any]:
    """Run NU pre-fit in a child process; returns status dict."""
    import json
    import subprocess
    import sys
    import time

    work = os.path.join(CACHE_DIR, "fit_runs")
    os.makedirs(work, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(work, stamp)
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "integrated_gui.nu_fit_worker",
        "--outdir",
        outdir,
        "--scaling",
        scaling,
        "--smoothing",
        str(smoothing),
    ]
    if homogenize_x:
        cmd.append("--homogenize-x")

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    status_path = os.path.join(outdir, "status.json")
    log_path = os.path.join(outdir, "worker.log")
    log_txt = ""
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            log_txt = f.read()
    if proc.stdout:
        log_txt = (log_txt + "\n" + proc.stdout).strip()
    if proc.stderr:
        log_txt = (log_txt + "\nSTDERR:\n" + proc.stderr).strip()

    status: Dict[str, Any] = {"ok": False, "log": log_txt, "outdir": outdir}
    if os.path.isfile(status_path):
        with open(status_path, "r", encoding="utf-8") as f:
            status.update(json.load(f))
    status["returncode"] = proc.returncode
    status["cache"] = cache_status()
    return status
