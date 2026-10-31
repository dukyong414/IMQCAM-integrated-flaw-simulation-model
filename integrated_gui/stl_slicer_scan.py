"""Layer-by-layer STL slicer with realistic L-PBF hatch strategies.

Every build layer (process layer thickness, e.g. 30 um) is cut from the mesh and
its cross-section is filled with hatch lines (process hatch spacing, e.g. 140 um)
by even-odd ray/polygon intersection. A full part produces tens of millions of
vectors, so only per-layer statistics, a few preview layers, and sampled
turn-point locations (edge pores) are kept in memory.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .scan_vectors import ESD_COL, STRATEGY_ANGLES

REALISTIC_STRATEGIES = ("stripes", "rotate", "rotating", "island")


def _prepare_surface(mesh):
    try:
        surf = mesh.extract_surface(algorithm="dataset_surface")
    except TypeError:
        surf = mesh.extract_surface()
    except Exception:
        surf = mesh
    try:
        surf = surf.triangulate().clean()
    except Exception:
        pass
    return surf


def _slice_segments(surf, z: float, cx: float, cy: float) -> np.ndarray:
    """Cross-section outline at height z as an (E, 4) array of x0, y0, x1, y1."""
    try:
        sl = surf.slice(normal="z", origin=(cx, cy, float(z)))
    except Exception:
        return np.zeros((0, 4))
    if sl is None or sl.n_points == 0 or sl.n_cells == 0:
        return np.zeros((0, 4))
    pts = np.asarray(sl.points, dtype=float)
    lines = np.asarray(sl.lines, dtype=np.int64)
    if len(lines) == 0:
        return np.zeros((0, 4))
    if len(lines) % 3 == 0 and np.all(lines[0::3] == 2):
        a = lines[1::3]
        b = lines[2::3]
    else:
        a_list: List[int] = []
        b_list: List[int] = []
        i = 0
        while i < len(lines):
            n = int(lines[i])
            ids = lines[i + 1 : i + 1 + n]
            a_list.extend(ids[:-1].tolist())
            b_list.extend(ids[1:].tolist())
            i += n + 1
        a = np.asarray(a_list, dtype=np.int64)
        b = np.asarray(b_list, dtype=np.int64)
    return np.column_stack([pts[a, 0], pts[a, 1], pts[b, 0], pts[b, 1]])


def _layer_mid_z(li: int, zmin: float, zmax: float, layer_mm: float) -> float:
    """Mid-height of layer ``li`` (avoids flat faces); the partial top layer uses its own middle."""
    lo = zmin + li * layer_mm
    return 0.5 * (lo + min(lo + layer_mm, zmax))


def _layer_angles(strategy: str, layer: int, base_angle_deg: Optional[float], rotation_deg: float) -> List[float]:
    s = str(strategy or "stripes").lower().strip()
    base = float(base_angle_deg) if base_angle_deg is not None else STRATEGY_ANGLES.get(s, 0.0)
    angle = base + layer * float(rotation_deg)
    if s == "crosshatch":
        return [angle, angle + 90.0]
    return [angle]


def _fill_cross_section(
    seg: np.ndarray,
    *,
    angle_deg: float,
    hatch_mm: float,
    split_w_mm: Optional[float],
    split_shift_mm: float,
    min_len_mm: float,
) -> Dict[str, np.ndarray]:
    """
    Hatch one closed cross-section. Hatch offsets are anchored to the plate
    origin so adjacent layers with the same angle line up.
    """
    empty = {k: np.zeros(0) for k in ("x0", "y0", "x1", "y1")}
    if len(seg) == 0:
        return empty
    th = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    u0 = seg[:, 0] * c + seg[:, 1] * s
    v0 = -seg[:, 0] * s + seg[:, 1] * c
    u1 = seg[:, 2] * c + seg[:, 3] * s
    v1 = -seg[:, 2] * s + seg[:, 3] * c

    lo = np.minimum(v0, v1)
    hi = np.maximum(v0, v1)
    k_lo = int(np.ceil(lo.min() / hatch_mm - 0.5))
    k_hi = int(np.floor(hi.max() / hatch_mm - 0.5))
    if k_hi < k_lo:
        return empty
    offsets = (np.arange(k_lo, k_hi + 1, dtype=float) + 0.5) * hatch_mm

    # Half-open [lo, hi) so a hatch line through a shared vertex is counted once
    i0 = np.searchsorted(offsets, lo, side="left")
    i1 = np.searchsorted(offsets, hi, side="left")
    cnt = i1 - i0
    m = cnt > 0
    if not np.any(m):
        return empty
    e_idx = np.repeat(np.nonzero(m)[0], cnt[m])
    first = np.repeat(np.cumsum(cnt[m]) - cnt[m], cnt[m])
    line = np.repeat(i0[m], cnt[m]) + (np.arange(len(e_idx)) - first)
    o = offsets[line]
    dv = v1[e_idx] - v0[e_idx]
    t = (o - v0[e_idx]) / np.where(np.abs(dv) < 1e-15, 1e-15, dv)
    u = u0[e_idx] + t * (u1[e_idx] - u0[e_idx])

    order = np.lexsort((u, line))
    line = line[order]
    u = u[order]
    idx = np.arange(len(line))
    starts = np.r_[True, line[1:] != line[:-1]]
    rank = idx - np.maximum.accumulate(np.where(starts, idx, 0))
    ev = idx[(rank % 2 == 0)]
    ev = ev[ev + 1 < len(line)]
    ev = ev[line[ev + 1] == line[ev]]
    ua, ub = u[ev], u[ev + 1]
    ln = line[ev]
    keep = (ub - ua) >= min_len_mm
    ua, ub, ln = ua[keep], ub[keep], ln[keep]
    if len(ua) == 0:
        return empty
    vv = offsets[ln]
    gline = ln + k_lo

    if split_w_mm and split_w_mm > 0:
        w = float(split_w_mm)
        m_lo = np.floor((ua - split_shift_mm) / w).astype(np.int64)
        m_hi = np.floor((ub - split_shift_mm) / w - 1e-12).astype(np.int64)
        n_piece = np.maximum(1, m_hi - m_lo + 1)
        rep = np.repeat(np.arange(len(ua)), n_piece)
        first = np.repeat(np.cumsum(n_piece) - n_piece, n_piece)
        stripe = m_lo[rep] + (np.arange(len(rep)) - first)
        pa = np.maximum(ua[rep], split_shift_mm + stripe * w)
        pb = np.minimum(ub[rep], split_shift_mm + (stripe + 1) * w)
        ok = (pb - pa) >= min_len_mm
        ua, ub, vv, gline, stripe = pa[ok], pb[ok], vv[rep][ok], gline[rep][ok], stripe[ok]
    else:
        stripe = np.zeros(len(ua), dtype=np.int64)

    # Serpentine: alternate direction line-to-line, scan stripe by stripe
    flip = (gline % 2) == 1
    sa = np.where(flip, ub, ua)
    sb = np.where(flip, ua, ub)
    order = np.lexsort((gline, stripe))
    sa, sb, vv = sa[order], sb[order], vv[order]
    return {
        "x0": sa * c - vv * s,
        "y0": sa * s + vv * c,
        "x1": sb * c - vv * s,
        "y1": sb * s + vv * c,
    }


def _hatch_layer(
    seg: np.ndarray,
    li: int,
    *,
    strategy: str,
    hatch_mm: float,
    base_angle_deg: Optional[float],
    rotation_deg: float,
    split_w_mm: Optional[float],
    min_len_mm: float,
) -> Tuple[Dict[str, np.ndarray], List[float]]:
    angles = _layer_angles(strategy, li, base_angle_deg, rotation_deg)
    shift = ((li * 0.37) % 1.0) * split_w_mm if split_w_mm else 0.0
    parts = [
        _fill_cross_section(
            seg,
            angle_deg=a,
            hatch_mm=hatch_mm,
            split_w_mm=split_w_mm,
            split_shift_mm=shift,
            min_len_mm=min_len_mm,
        )
        for a in angles
    ]
    if len(parts) == 1:
        return parts[0], angles
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, angles


def _inward_along_vector(px, py, ox, oy, rng: np.random.Generator, jitter_mm: float):
    """Move turn points toward the vector's other end by |N(0, jitter)|.

    Endpoints lie on the part contour, so isotropic jitter pushes about a third
    of edge pores outside the part; stepping along the scan vector keeps them
    inside the scanned region.
    """
    px, py, ox, oy = (np.asarray(a, dtype=float) for a in (px, py, ox, oy))
    dx, dy = ox - px, oy - py
    length = np.hypot(dx, dy)
    step = np.minimum(np.abs(rng.normal(0.0, jitter_mm, len(px))), 0.5 * length)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(length > 0, step / length, 0.0)
    return px + f * dx, py + f * dy


def _edge_pores_from_points(
    pts: np.ndarray,
    size_catalog: pd.DataFrame,
    rng: np.random.Generator,
    *,
    jitter_mm: float = 0.05,
) -> pd.DataFrame:
    cols = [
        ESD_COL, "pore_source", "x_mm", "y_mm", "z_mm", "build_index",
        "radius_mm", "volume_mm3", "pore_family", "vector_id", "endpoint",
    ]
    if len(pts) == 0:
        return pd.DataFrame(columns=cols)
    catalog = size_catalog if ESD_COL in size_catalog.columns else pd.DataFrame({ESD_COL: [20.0]})
    esd = catalog[ESD_COL].to_numpy(dtype=float)
    esd = rng.choice(esd, size=len(pts), replace=True)
    r_mm = esd / 2000.0
    x, y = _inward_along_vector(pts[:, 0], pts[:, 1], pts[:, 5], pts[:, 6], rng, jitter_mm)
    return pd.DataFrame(
        {
            ESD_COL: esd,
            "pore_source": "edge",
            "x_mm": x,
            "y_mm": y,
            "z_mm": pts[:, 2],
            "build_index": 0,
            "radius_mm": r_mm,
            "volume_mm3": (4.0 / 3.0) * np.pi * r_mm**3,
            "pore_family": "edge",
            "vector_id": pts[:, 3].astype(np.int64),
            "endpoint": np.where(pts[:, 4] > 0.5, "end", "start"),
        }
    )


def slice_scan_stl(
    mesh,
    *,
    hatch_mm: float,
    layer_mm: float,
    strategy: str = "stripes",
    stripe_width_mm: float = 10.0,
    layer_rotation_deg: float = 67.0,
    base_angle_deg: Optional[float] = None,
    batch_rows: int = 4,
    batch_cols: int = 4,
    velocity_mm_s: Optional[float] = None,
    n_edge_pores: int = 0,
    size_catalog: Optional[pd.DataFrame] = None,
    seed: int = 42,
    layer_indices: Optional[Sequence[int]] = None,
    n_preview_layers: int = 10,
    max_preview_vectors_per_layer: int = 5000,
    include_contour: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Slice + hatch an STL at process resolution.

    ``layer_indices`` restricts the work to those layers (e.g. first few for a
    preview); ``None`` scans every layer of the part.
    """
    hatch_mm = float(max(hatch_mm, 1e-4))
    layer_mm = float(max(layer_mm, 1e-4))
    strategy = str(strategy or "stripes").lower().strip()
    rng = np.random.default_rng(int(seed))

    surf = _prepare_surface(mesh)
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in surf.bounds]
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    height = max(zmax - zmin, layer_mm)
    n_layers_total = max(1, int(np.ceil(height / layer_mm)))

    if strategy == "stripes":
        split_w = float(stripe_width_mm) if stripe_width_mm and stripe_width_mm > 0 else None
    elif strategy == "batch":
        split_w = max((xmax - xmin) / max(1, int(batch_cols)), (ymax - ymin) / max(1, int(batch_rows)))
    else:
        split_w = None
    min_len = 0.5 * hatch_mm

    if layer_indices is None:
        layers = np.arange(n_layers_total)
    else:
        layers = np.asarray(
            sorted({int(i) for i in layer_indices if 0 <= int(i) < n_layers_total}), dtype=np.int64
        )
    if len(layers) == 0:
        layers = np.array([0], dtype=np.int64)
    z_of = lambda li: _layer_mid_z(li, zmin, zmax, layer_mm)  # noqa: E731

    n_prev = max(1, min(int(n_preview_layers), len(layers)))
    preview_set = set(layers[np.unique(np.linspace(0, len(layers) - 1, n_prev).round().astype(int))].tolist())

    # Estimate total turn points so edge pores are spread over all layers in one pass
    n_edge = int(max(0, n_edge_pores))
    keep_frac = 0.0
    if n_edge > 0:
        probe = layers[np.unique(np.linspace(0, len(layers) - 1, min(24, len(layers))).round().astype(int))]
        probe_counts = []
        for li in probe:
            seg = _slice_segments(surf, z_of(li), cx, cy)
            h, _ = _hatch_layer(
                seg, int(li), strategy=strategy, hatch_mm=hatch_mm, base_angle_deg=base_angle_deg,
                rotation_deg=layer_rotation_deg, split_w_mm=split_w, min_len_mm=min_len,
            )
            probe_counts.append(2 * len(h["x0"]))
        est_points = float(np.mean(probe_counts)) * len(layers) if probe_counts else 0.0
        keep_frac = min(1.0, 1.5 * n_edge / est_points) if est_points > 0 else 1.0

    stats_rows = []
    preview_frames: List[pd.DataFrame] = []
    edge_chunks: List[np.ndarray] = []
    vector_offset = 0
    report_every = max(1, len(layers) // 100)

    for k, li in enumerate(layers):
        li = int(li)
        z = z_of(li)
        seg = _slice_segments(surf, z, cx, cy)
        h, angles = _hatch_layer(
            seg, li, strategy=strategy, hatch_mm=hatch_mm, base_angle_deg=base_angle_deg,
            rotation_deg=layer_rotation_deg, split_w_mm=split_w, min_len_mm=min_len,
        )
        nv = len(h["x0"])
        hatch_len = float(np.hypot(h["x1"] - h["x0"], h["y1"] - h["y0"]).sum()) if nv else 0.0
        contour_len = float(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]).sum()) if len(seg) else 0.0
        stats_rows.append(
            {
                "layer": li,
                "z_mm": z,
                "angle_deg": float(angles[0]),
                "n_hatch_vectors": nv,
                "hatch_length_mm": hatch_len,
                "contour_length_mm": contour_len,
                "scanned_area_mm2": hatch_len * hatch_mm,
            }
        )

        if keep_frac > 0 and nv:
            n_pts = 2 * nv
            take = rng.binomial(n_pts, keep_frac)
            if take:
                pick = rng.choice(n_pts, size=min(take, n_pts), replace=False)
                vid = pick // 2
                is_end = pick % 2
                px = np.where(is_end == 1, h["x1"][vid], h["x0"][vid])
                py = np.where(is_end == 1, h["y1"][vid], h["y0"][vid])
                ox = np.where(is_end == 1, h["x0"][vid], h["x1"][vid])
                oy = np.where(is_end == 1, h["y0"][vid], h["y1"][vid])
                edge_chunks.append(
                    np.column_stack([px, py, np.full(len(pick), z), vid + vector_offset, is_end, ox, oy])
                )

        if li in preview_set and nv:
            step = max(1, int(np.ceil(nv / max(1, max_preview_vectors_per_layer))))
            sl = slice(None, None, step)
            frame = pd.DataFrame(
                {
                    "x0": h["x0"][sl], "y0": h["y0"][sl], "z0": z,
                    "x1": h["x1"][sl], "y1": h["y1"][sl], "z1": z,
                    "layer": li, "build_index": 0, "strategy": strategy,
                    "angle_deg": float(angles[0]), "kind": "hatch",
                }
            )
            preview_frames.append(frame)
            if include_contour and len(seg):
                preview_frames.append(
                    pd.DataFrame(
                        {
                            "x0": seg[:, 0], "y0": seg[:, 1], "z0": z,
                            "x1": seg[:, 2], "y1": seg[:, 3], "z1": z,
                            "layer": li, "build_index": 0, "strategy": "contour",
                            "angle_deg": float(angles[0]), "kind": "contour",
                        }
                    )
                )
        vector_offset += nv
        if progress is not None and (k % report_every == 0 or k == len(layers) - 1):
            progress(k + 1, len(layers))

    per_layer = pd.DataFrame(stats_rows)
    edge_pts = np.vstack(edge_chunks) if edge_chunks else np.zeros((0, 7))
    if len(edge_pts) > n_edge:
        edge_pts = edge_pts[rng.choice(len(edge_pts), size=n_edge, replace=False)]
    catalog = size_catalog if size_catalog is not None and len(size_catalog) else pd.DataFrame({ESD_COL: [20.0]})
    edge_df = _edge_pores_from_points(edge_pts, catalog, rng)

    vectors_df = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame(
        columns=["x0", "y0", "z0", "x1", "y1", "z1", "layer", "build_index", "strategy", "angle_deg", "kind"]
    )
    if len(vectors_df):
        vectors_df.insert(0, "vector_id", np.arange(len(vectors_df)))

    hatch_total = float(per_layer["hatch_length_mm"].sum()) if len(per_layer) else 0.0
    contour_total = float(per_layer["contour_length_mm"].sum()) if len(per_layer) else 0.0
    v = float(velocity_mm_s) if velocity_mm_s and velocity_mm_s > 0 else None
    stats = {
        "strategy": strategy,
        "hatch_mm": hatch_mm,
        "layer_mm": layer_mm,
        "stripe_width_mm": split_w,
        "layer_rotation_deg": float(layer_rotation_deg),
        "n_layers_part": int(n_layers_total),
        "n_layers_scanned": int(len(layers)),
        "full_part": bool(layer_indices is None),
        "n_hatch_vectors": int(per_layer["n_hatch_vectors"].sum()) if len(per_layer) else 0,
        "hatch_length_mm": hatch_total,
        "contour_length_mm": contour_total,
        "scanned_volume_mm3": hatch_total * hatch_mm * layer_mm,
        "velocity_mm_s": v,
        "laser_time_s": (hatch_total + (contour_total if include_contour else 0.0)) / v if v else None,
        "bounds_mm": [xmin, xmax, ymin, ymax, zmin, zmax],
    }
    return {"stats": stats, "per_layer": per_layer, "vectors_df": vectors_df, "edge_df": edge_df}


def box_mesh(xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float):
    """Closed triangulated box so cuboids go through the same slicer as STLs."""
    import pyvista as pv

    return pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax)).triangulate()


def scan_keyframes(
    mesh,
    *,
    hatch_mm: float,
    layer_mm: float,
    strategy: str = "stripes",
    stripe_width_mm: float = 10.0,
    layer_rotation_deg: float = 67.0,
    base_angle_deg: Optional[float] = None,
    batch_rows: int = 4,
    batch_cols: int = 4,
    n_layers_anim: int = 12,
    vectors_per_layer: int = 6,
) -> Dict:
    """
    Scan-ordered keyframes that sweep the whole part: evenly spaced layers from the
    first to the last, and within each layer vectors from the first to the last one
    (which ends on the cross-section boundary).

    Returns {"segments": [dict x0,y0,x1,y1,z,layer], "outlines": {layer: (E,4)},
    "n_layers_part": int, "bounds_mm": [...]}.
    """
    hatch_mm = float(max(hatch_mm, 1e-4))
    layer_mm = float(max(layer_mm, 1e-4))
    strategy = str(strategy or "stripes").lower().strip()
    surf = _prepare_surface(mesh)
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in surf.bounds]
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    n_total = max(1, int(np.ceil(max(zmax - zmin, layer_mm) / layer_mm)))
    if strategy == "stripes":
        split_w = float(stripe_width_mm) if stripe_width_mm and stripe_width_mm > 0 else None
    elif strategy == "batch":
        split_w = max((xmax - xmin) / max(1, int(batch_cols)), (ymax - ymin) / max(1, int(batch_rows)))
    else:
        split_w = None

    layers = np.unique(np.linspace(0, n_total - 1, max(1, int(n_layers_anim))).round().astype(int))
    segments: List[Dict] = []
    outlines: Dict[int, np.ndarray] = {}
    for li in layers:
        z = _layer_mid_z(int(li), zmin, zmax, layer_mm)
        seg = _slice_segments(surf, z, cx, cy)
        h, _ = _hatch_layer(
            seg, int(li), strategy=strategy, hatch_mm=hatch_mm, base_angle_deg=base_angle_deg,
            rotation_deg=layer_rotation_deg, split_w_mm=split_w, min_len_mm=0.5 * hatch_mm,
        )
        nv = len(h["x0"])
        if nv == 0:
            continue
        outlines[int(li)] = seg
        pick = np.unique(np.linspace(0, nv - 1, max(1, int(vectors_per_layer))).round().astype(int))
        for k in pick:
            segments.append(
                {
                    "x0": float(h["x0"][k]), "y0": float(h["y0"][k]),
                    "x1": float(h["x1"][k]), "y1": float(h["y1"][k]),
                    "z": float(z), "layer": int(li), "vec_index": int(k), "n_vec_layer": int(nv),
                }
            )
    return {
        "segments": segments,
        "outlines": outlines,
        "n_layers_part": n_total,
        "bounds_mm": [xmin, xmax, ymin, ymax, zmin, zmax],
    }


def format_scan_stats(stats: Dict) -> str:
    """One-line human summary for logs / result panels."""
    if not stats:
        return ""
    t = stats.get("laser_time_s")
    t_txt = f"  laser-on ~{t / 3600.0:.1f} h" if t else ""
    stripe = stats.get("stripe_width_mm")
    stripe_txt = f"  stripe={stripe:.1f} mm" if stripe else ""
    return (
        f"Scan {stats.get('strategy')}: hatch={stats.get('hatch_mm', 0) * 1e3:.0f} um  "
        f"layer={stats.get('layer_mm', 0) * 1e3:.0f} um  rot={stats.get('layer_rotation_deg', 0):.0f} deg"
        f"{stripe_txt}  layers={stats.get('n_layers_scanned')}/{stats.get('n_layers_part')}  "
        f"vectors={stats.get('n_hatch_vectors', 0):,}  "
        f"hatch path={stats.get('hatch_length_mm', 0) / 1e6:.2f} km{t_txt}"
    )
