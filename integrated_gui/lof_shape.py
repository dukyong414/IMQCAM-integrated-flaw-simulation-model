"""Size and shape statistics of Raptor LoF pores (size distribution, sphericity, elongation)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Pores smaller than this many voxels are too coarse for a meaningful surface area.
MIN_SHAPE_VOXELS = 20


def _label_sparse(ijk: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """26-connected component labels for a sparse set of voxel indices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(ijk)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    nx, ny, nz = shape
    keys = (ijk[:, 0].astype(np.int64) * ny + ijk[:, 1]) * nz + ijk[:, 2]
    order = np.argsort(keys)
    skeys = keys[order]
    rows, cols = [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if (di, dj, dk) <= (0, 0, 0):
                    continue  # each neighbour pair once
                nb = ijk + np.array([di, dj, dk])
                ok = (
                    (nb[:, 0] >= 0) & (nb[:, 0] < nx)
                    & (nb[:, 1] >= 0) & (nb[:, 1] < ny)
                    & (nb[:, 2] >= 0) & (nb[:, 2] < nz)
                )
                nkeys = (nb[:, 0].astype(np.int64) * ny + nb[:, 1]) * nz + nb[:, 2]
                pos = np.searchsorted(skeys, nkeys)
                pos = np.clip(pos, 0, n - 1)
                hit = ok & (skeys[pos] == nkeys)
                rows.append(np.nonzero(hit)[0])
                cols.append(order[pos[hit]])
    r = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


def _surface_area_voxels(pts: np.ndarray) -> float:
    """Marching-cubes surface area (in voxel units²) of one pore given its voxel indices."""
    from skimage.measure import marching_cubes, mesh_surface_area

    lo = pts.min(axis=0) - 1
    dims = pts.max(axis=0) - lo + 2
    vol = np.zeros(tuple(int(d) for d in dims), dtype=np.float32)
    p = pts - lo
    vol[p[:, 0], p[:, 1], p[:, 2]] = 1.0
    verts, faces, _, _ = marching_cubes(vol, level=0.5)
    return float(mesh_surface_area(verts, faces))


def lof_pore_shapes(
    phase: np.ndarray, voxel_m: float, *, min_voxels: int = 2, origin_m=(0.0, 0.0, 0.0)
) -> pd.DataFrame:
    """One row per LoF pore (phase == 0 component) with size and shape descriptors."""
    ijk = np.argwhere(np.asarray(phase) == 0)
    labels = _label_sparse(ijk, phase.shape)
    v_um = float(voxel_m) * 1e6
    o_mm = np.asarray(origin_m, dtype=float) * 1e3
    rows = []
    if len(ijk):
        order = np.argsort(labels, kind="stable")
        bounds = np.flatnonzero(np.diff(labels[order])) + 1
        for grp in np.split(order, bounds):
            n = len(grp)
            if n < min_voxels:
                continue
            pts = ijk[grp]
            vol_um3 = n * v_um**3
            esd = (6.0 * vol_um3 / np.pi) ** (1.0 / 3.0)
            # Principal axes of the voxel cloud (+ voxel extent so tiny pores stay finite)
            cov = np.cov(pts.T.astype(float)) if n > 1 else np.zeros((3, 3))
            ev = np.sort(np.linalg.eigvalsh(cov + np.eye(3) / 12.0))[::-1]
            ev = np.maximum(ev, 1e-12)
            major_um = 2.0 * np.sqrt(5.0 * ev[0]) * v_um  # ellipsoid axis length
            minor_um = 2.0 * np.sqrt(5.0 * ev[2]) * v_um
            sph = np.nan
            area_um2 = np.nan
            if n >= MIN_SHAPE_VOXELS:
                try:
                    area_um2 = _surface_area_voxels(pts) * v_um**2
                    sph = float(np.pi ** (1 / 3) * (6 * vol_um3) ** (2 / 3) / area_um2)
                    sph = min(sph, 1.0)
                except Exception:
                    pass
            c = o_mm + pts.mean(axis=0) * v_um * 1e-3
            rows.append(
                {
                    "n_voxels": n,
                    "volume_um3": vol_um3,
                    "esd_um": esd,
                    "surface_um2": area_um2,
                    "sphericity": sph,
                    "major_axis_um": major_um,
                    "minor_axis_um": minor_um,
                    "aspect_ratio": float(np.sqrt(ev[0] / ev[2])),
                    "x_mm": c[0],
                    "y_mm": c[1],
                    "z_mm": c[2],
                }
            )
    return pd.DataFrame(rows)


def summarize_shapes(df: pd.DataFrame, voxel_m: float) -> Dict[str, Any]:
    """Compact size-distribution / shape summary for summary.json."""
    if df is None or not len(df):
        return {"n_pores": 0}
    esd = df["esd_um"].to_numpy(float)
    shaped = df.dropna(subset=["sphericity"])
    edges = [0, 20, 30, 40, 50, 75, 100, 150, 200, 1e9]
    labels = ["<20", "20-30", "30-40", "40-50", "50-75", "75-100", "100-150", "150-200", ">200"]
    counts, _ = np.histogram(esd, bins=edges)
    out = {
        "n_pores": int(len(df)),
        "voxel_um": float(voxel_m) * 1e6,
        "esd_mean_um": float(esd.mean()),
        "esd_median_um": float(np.median(esd)),
        "esd_std_um": float(esd.std()),
        "esd_d10_um": float(np.percentile(esd, 10)),
        "esd_d90_um": float(np.percentile(esd, 90)),
        "esd_max_um": float(esd.max()),
        "volume_weighted_mean_esd_um": float(np.sum(esd * df["volume_um3"]) / df["volume_um3"].sum()),
        "major_axis_mean_um": float(df["major_axis_um"].mean()),
        "aspect_ratio_mean": float(df["aspect_ratio"].mean()),
        "aspect_ratio_median": float(df["aspect_ratio"].median()),
        "esd_histogram_um": {k: int(v) for k, v in zip(labels, counts) if v},
        "shape_min_voxels": MIN_SHAPE_VOXELS,
        "n_shape_pores": int(len(shaped)),
    }
    if len(shaped):
        s = shaped["sphericity"].to_numpy(float)
        out.update(
            {
                "sphericity_mean": float(s.mean()),
                "sphericity_median": float(np.median(s)),
                "sphericity_d10": float(np.percentile(s, 10)),
                "sphericity_d90": float(np.percentile(s, 90)),
            }
        )
    return out
