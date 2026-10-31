"""CLI / subprocess entry: open last Combined 3D result in a PyVista window."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from integrated_gui.combined_3d import PartGeometry
from integrated_gui.combined_viz import open_desktop_viewer

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "assets", "pyvista")


def _write_status(dir_path: str, stage: str, message: str, **extra) -> None:
    path = os.path.join(dir_path, "viewer_status.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"stage": stage, "message": message, "ts": time.time(), **extra},
                f,
                indent=2,
            )
    except Exception:
        pass


def _load_pores(dir_path: str, max_points: int) -> tuple[pd.DataFrame, int, int]:
    """Prefer pre-sampled view CSV; fall back to full CSV with column pruning."""
    view_csv = os.path.join(dir_path, "latest_pores_view.csv")
    full_csv = os.path.join(dir_path, "latest_pores.csv")
    meta_path = os.path.join(dir_path, "latest_view_meta.json")
    n_total = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                n_total = int(json.load(f).get("n_total") or 0)
        except Exception:
            n_total = None

    if os.path.isfile(view_csv):
        pores = pd.read_csv(view_csv, low_memory=False)
        n_total = n_total or int(len(pores))
        return pores, n_total, int(len(pores))

    if not os.path.isfile(full_csv):
        raise SystemExit(f"No saved Combined result in {dir_path}. Run Generate first.")

    # Only columns needed for the viewer — much faster on large OneDrive CSVs
    header = pd.read_csv(full_csv, nrows=0)
    usecols = [
        c
        for c in ("x_mm", "y_mm", "z_mm", "pore_source", "pore_family", "radius_mm")
        if c in header.columns
    ]
    pores = pd.read_csv(full_csv, usecols=usecols or None, low_memory=False)
    n_total = int(len(pores))
    if n_total > max_points:
        src_col = (
            "pore_source"
            if "pore_source" in pores.columns
            else ("pore_family" if "pore_family" in pores.columns else None)
        )
        if src_col:
            parts = []
            groups = list(pores.groupby(src_col, sort=False))
            per = max(50, max_points // max(1, len(groups)))
            for _, g in groups:
                take = min(len(g), per)
                parts.append(g.sample(take, random_state=0) if len(g) > take else g)
            pores = pd.concat(parts, ignore_index=True)
            if len(pores) > max_points:
                pores = pores.sample(max_points, random_state=0)
        else:
            pores = pores.sample(max_points, random_state=0)
    return pores, n_total, int(len(pores))


def main():
    # Interactive desktop window — undo headless flags inherited from run_gui.bat
    os.environ["PYVISTA_OFF_SCREEN"] = "false"
    os.environ.pop("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", None)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Prefer hardware OpenGL for the interactive Combined viewer
    os.environ.setdefault("__GL_SYNC_TO_VBLANK", "0")
    os.environ.setdefault("VTK_USE_OFFSCREEN_OPENGL", "0")

    parser = argparse.ArgumentParser(description="Open Combined 3D pores in PyVista")
    parser.add_argument(
        "--dir",
        default=DEFAULT_DIR,
        help="Directory with latest_pores.csv and latest_geom.json",
    )
    parser.add_argument("--max-points", type=int, default=4000)
    args = parser.parse_args()

    log_path = os.path.join(args.dir, "viewer.log")
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"view_pores start max_points={args.max_points}\n")

        _write_status(args.dir, "loading", "Loading pore sample for PyVista…")
        t0 = time.time()
        pores, n_total, n_view = _load_pores(args.dir, args.max_points)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"loaded n_total={n_total} n_view={n_view} in {time.time()-t0:.2f}s\n")

        geom_path = os.path.join(args.dir, "latest_geom.json")
        if not os.path.exists(geom_path):
            raise SystemExit(f"No saved Combined result in {args.dir}. Run Generate first.")

        with open(geom_path, "r", encoding="utf-8") as f:
            g = json.load(f)

        cubes_path = os.path.join(args.dir, "latest_cubes.csv")
        cubes = pd.read_csv(cubes_path) if os.path.exists(cubes_path) else None

        _write_status(
            args.dir,
            "showing",
            f"Opening window ({n_view:,} of {n_total:,} pores)…",
            n_total=n_total,
            n_view=n_view,
        )
        if g.get("mode") == "plate" or "n_rows" in g:
            open_desktop_viewer(pores, layout=g, cubes=cubes, max_points=args.max_points)
        else:
            geom = PartGeometry(
                **{k: v for k, v in g.items() if k in PartGeometry.__dataclass_fields__}
            )
            open_desktop_viewer(pores, geom=geom, max_points=args.max_points)
        _write_status(args.dir, "closed", "Viewer closed.", n_total=n_total, n_view=n_view)
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(tb)
        except Exception:
            pass
        _write_status(args.dir, "error", tb.splitlines()[-1] if tb else "Viewer failed")
        raise


if __name__ == "__main__":
    main()
