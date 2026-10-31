"""Persist and overlay Raptor scan paths (meters) on PyVista RVE views."""
from __future__ import annotations

import csv
import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def _clip_segment_aabb(
    p0: np.ndarray,
    p1: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Liang–Barsky clip of a 3D segment to an axis-aligned box."""
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    bmin = np.asarray(bmin, dtype=float).reshape(3)
    bmax = np.asarray(bmax, dtype=float).reshape(3)
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-15:
            if p0[i] < bmin[i] or p0[i] > bmax[i]:
                return None
            continue
        inv = 1.0 / d[i]
        t_near = (bmin[i] - p0[i]) * inv
        t_far = (bmax[i] - p0[i]) * inv
        if t_near > t_far:
            t_near, t_far = t_far, t_near
        t0 = max(t0, t_near)
        t1 = min(t1, t_far)
        if t0 > t1:
            return None
    return p0 + t0 * d, p0 + t1 * d


def write_path_vectors_csv(
    csv_path: str,
    path_vectors: Sequence[Any],
    *,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    layer_height: float,
) -> int:
    """
    Write AABB-clipped path segments to CSV (meters).

    Columns: x0,y0,z0,x1,y1,z1,layer
    Returns number of segments written.
    """
    bmin = np.asarray(bounds_min, dtype=float).reshape(3)
    bmax = np.asarray(bounds_max, dtype=float).reshape(3)
    lh = float(layer_height) if layer_height and layer_height > 0 else 1.0
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    n_written = 0
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x0", "y0", "z0", "x1", "y1", "z1", "layer"])
        for vec in path_vectors:
            clipped = _clip_segment_aabb(vec.start_point, vec.end_point, bmin, bmax)
            if clipped is None:
                continue
            a, b = clipped
            if float(np.linalg.norm(b - a)) < 1e-12:
                continue
            z_ref = 0.5 * (float(a[2]) + float(b[2]))
            layer = int(round((z_ref - float(bmin[2])) / lh))
            w.writerow(
                [
                    f"{a[0]:.8e}",
                    f"{a[1]:.8e}",
                    f"{a[2]:.8e}",
                    f"{b[0]:.8e}",
                    f"{b[1]:.8e}",
                    f"{b[2]:.8e}",
                    layer,
                ]
            )
            n_written += 1
    return n_written


def sibling_path_vectors_csv(vti_path: str) -> Optional[str]:
    """Return path_vectors.csv next to a VTI if it exists."""
    if not vti_path:
        return None
    candidate = os.path.join(os.path.dirname(os.path.abspath(vti_path)), "path_vectors.csv")
    return candidate if os.path.isfile(candidate) else None


def load_path_polydata(csv_path: str):
    """Load path_vectors.csv into a PyVista line PolyData (meters)."""
    import pyvista as pv

    if not csv_path or not os.path.isfile(csv_path):
        return None
    pts: List[List[float]] = []
    cells: List[int] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x0, y0, z0 = float(row["x0"]), float(row["y0"]), float(row["z0"])
                x1, y1, z1 = float(row["x1"]), float(row["y1"]), float(row["z1"])
            except (KeyError, TypeError, ValueError):
                continue
            i = len(pts)
            pts.append([x0, y0, z0])
            pts.append([x1, y1, z1])
            cells.extend([2, i, i + 1])
    if not pts:
        return None
    return pv.PolyData(np.asarray(pts, dtype=float), lines=np.asarray(cells, dtype=np.int64))


def add_scan_paths_to_plotter(
    plotter,
    csv_path: Optional[str],
    *,
    line_width: float = 1.0,
    name: str = "Scan path",
    color: str = "#1565C0",
    opacity: float = 0.4,
) -> bool:
    """
    Overlay scan-path lines on a PyVista plotter.

    Returns True if paths were added.
    """
    mesh = load_path_polydata(csv_path) if csv_path else None
    if mesh is None or mesh.n_points == 0:
        return False
    plotter.add_mesh(
        mesh,
        color=color,
        line_width=float(line_width),
        opacity=float(opacity),
        name=name,
        label="Scan path",
        render_lines_as_tubes=False,
    )
    return True
