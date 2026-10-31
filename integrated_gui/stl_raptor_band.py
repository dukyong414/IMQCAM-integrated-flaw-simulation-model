"""Thin height band of an STL (or STL cake slice) for Raptor at process resolution.

Raptor needs ~10-20 um voxels to resolve lack-of-fusion between 140 um hatches, which
is only affordable for a few tens of layers over a large cross-section. The band keeps
the part's real cross-section: scan paths are the slicer's in-part hatch + contour
vectors, and voxels outside the part are masked out of the porosity statistics.

For cake slices (e.g. 1/16 of a logo), the geometric mid-height often has a thin
cross-section that looks nothing like the part. By default the band is placed at the
height with the largest slice area, and the XY domain uses the full mesh footprint so
the Raptor box matches the cake-slice outline on the plate.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .stl_slicer_scan import _hatch_layer, _layer_mid_z, _prepare_surface, _slice_segments


def _slice_extent_mm(seg: np.ndarray) -> float:
    """Rough cross-section size: sum of contour segment lengths (mm)."""
    if seg is None or len(seg) == 0:
        return 0.0
    return float(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]).sum())


def _best_layer_index(surf, *, zmin: float, zmax: float, layer_mm: float, n_total: int, cx: float, cy: float) -> int:
    """Layer index with the largest contour (fullest cross-section of the part)."""
    # Probe up to ~48 heights so logo / tapering parts land on a representative slice
    probe = np.unique(np.linspace(0, max(0, n_total - 1), min(48, n_total)).round().astype(int))
    best_i, best_e = int(probe[len(probe) // 2]), -1.0
    for li in probe:
        z = _layer_mid_z(int(li), zmin, zmax, layer_mm)
        e = _slice_extent_mm(_slice_segments(surf, z, cx, cy))
        if e > best_e:
            best_e, best_i = e, int(li)
    return best_i


def prepare_stl_band(
    mesh,
    *,
    hatch_mm: float,
    layer_mm: float,
    n_layers: int = 15,
    band_z_mm: Optional[float] = None,
    strategy: str = "stripes",
    stripe_width_mm: float = 10.0,
    layer_rotation_deg: float = 67.0,
    base_angle_deg: Optional[float] = None,
    pad_mm: float = 0.15,
    xy_from_mesh: bool = True,
    band_z_mode: str = "fullest",
) -> Dict[str, Any]:
    """Scan segments (local mm) + domain box for ``n_layers`` process layers of ``mesh``.

    ``band_z_mm`` is the band's bottom height on the plate. When omitted:
      * ``band_z_mode="fullest"`` (default) — height with the largest cross-section
      * ``band_z_mode="middle"`` — geometric mid-height of the part

    ``xy_from_mesh`` (default True) sizes the XY domain from the full mesh AABB so a
    cake slice keeps its plate footprint even when the chosen band height is thinner.

    Segment rows are (x0, y0, x1, y1, z_local, kind) with kind 0 = hatch, 1 = contour and
    z_local = k * layer_mm for band layer k (same convention as the cuboid scan).
    """
    strategy = str(strategy or "stripes").lower().strip()
    surf = _prepare_surface(mesh)
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in surf.bounds]
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    n_total = max(1, int(np.ceil((zmax - zmin) / layer_mm)))
    n = int(max(1, min(int(n_layers), n_total)))
    placement = "user"
    if band_z_mm is None or not np.isfinite(float(band_z_mm)):
        if str(band_z_mode or "fullest").lower() == "middle":
            li0 = max(0, n_total // 2 - n // 2)
            placement = "middle"
        else:
            li_best = _best_layer_index(
                surf, zmin=zmin, zmax=zmax, layer_mm=layer_mm, n_total=n_total, cx=cx, cy=cy
            )
            li0 = int(np.clip(li_best - n // 2, 0, n_total - n))
            placement = "fullest"
    else:
        li0 = int(np.floor((float(band_z_mm) - zmin) / layer_mm))
    li0 = int(np.clip(li0, 0, n_total - n))

    split_w = float(stripe_width_mm) if strategy == "stripes" and stripe_width_mm else None
    rows = []
    outline = None  # absolute-mm contour of the centre band layer (for Raptor view)
    for k in range(n):
        li = li0 + k
        z = _layer_mid_z(li, zmin, zmax, layer_mm)
        seg = _slice_segments(surf, z, cx, cy)
        if len(seg) == 0:
            continue
        if outline is None or k == n // 2:
            outline = np.asarray(seg, dtype=float)
        h, _ = _hatch_layer(
            seg, li, strategy=strategy, hatch_mm=hatch_mm, base_angle_deg=base_angle_deg,
            rotation_deg=layer_rotation_deg, split_w_mm=split_w, min_len_mm=0.5 * hatch_mm,
        )
        zl = k * layer_mm
        if len(h["x0"]):
            rows.append(np.column_stack(
                [h["x0"], h["y0"], h["x1"], h["y1"], np.full(len(h["x0"]), zl), np.zeros(len(h["x0"]))]
            ))
        rows.append(np.column_stack([seg, np.full(len(seg), zl), np.ones(len(seg))]))
    if not rows:
        raise ValueError("The chosen band has no part cross-section; pick another height.")
    segs = np.vstack(rows)
    oz = zmin + li0 * layer_mm
    if xy_from_mesh:
        ox, oy = float(xmin - pad_mm), float(ymin - pad_mm)
        Lx = float((xmax + pad_mm) - ox)
        Ly = float((ymax + pad_mm) - oy)
    else:
        allx = np.concatenate([segs[:, 0], segs[:, 2]])
        ally = np.concatenate([segs[:, 1], segs[:, 3]])
        ox, oy = float(allx.min() - pad_mm), float(ally.min() - pad_mm)
        Lx = float(allx.max() + pad_mm - ox)
        Ly = float(ally.max() + pad_mm - oy)
    local = segs.copy()
    local[:, [0, 2]] -= ox
    local[:, [1, 3]] -= oy
    outline_local = None
    if outline is not None and len(outline):
        outline_local = outline.copy()
        outline_local[:, [0, 2]] -= ox
        outline_local[:, [1, 3]] -= oy
    return {
        "segments_mm": local,
        "origin_mm": [ox, oy, oz],
        "size_mm": [Lx, Ly, n * layer_mm],
        "mesh_aabb_mm": [xmin, xmax, ymin, ymax, zmin, zmax],
        "layer_mm": float(layer_mm),
        "first_layer": int(li0),
        "n_layers": int(n),
        "n_layers_part": int(n_total),
        "z_mid_abs_mm": [float(_layer_mid_z(li0 + k, zmin, zmax, layer_mm)) for k in range(n)],
        "n_hatch": int(np.sum(segs[:, 5] == 0)),
        "n_contour": int(np.sum(segs[:, 5] == 1)),
        "band_placement": placement,
        "xy_from_mesh": bool(xy_from_mesh),
        "outline_mm": outline_local,  # local XY contour for the Raptor 3D silhouette
        "full_height": False,
        "surf": surf,
    }


def prepare_stl_volume(
    mesh,
    *,
    hatch_mm: float,
    layer_mm: float,
    max_hatch_layers: int = 120,
    strategy: str = "stripes",
    stripe_width_mm: float = 10.0,
    layer_rotation_deg: float = 67.0,
    base_angle_deg: Optional[float] = None,
    pad_mm: float = 0.15,
) -> Dict[str, Any]:
    """Full-height cake-slice / STL domain for a 3D Raptor view of the part.

    The XY/Z box is the mesh AABB. Hatch paths use CMU/Raptor ``hatch_mm`` /
    ``layer_mm``. When ``max_hatch_layers >=`` the number of process layers in the
    part, every layer is scanned (same as cube / pillar whole-build). A smaller
    cap still spaces layers evenly through the full height.
    """
    strategy = str(strategy or "stripes").lower().strip()
    surf = _prepare_surface(mesh)
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in surf.bounds]
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    n_total = max(1, int(np.ceil((zmax - zmin) / max(layer_mm, 1e-9))))
    n_hatch = int(max(1, min(int(max_hatch_layers), n_total)))
    if n_hatch >= n_total:
        layer_ids = np.arange(n_total, dtype=int)
    else:
        # Evenly spaced process layers through the whole part (includes base + top)
        layer_ids = np.unique(
            np.linspace(0, n_total - 1, n_hatch).round().astype(int)
        )
    split_w = float(stripe_width_mm) if strategy == "stripes" and stripe_width_mm else None
    rows = []
    z_mids = []
    outline = None
    best_e = -1.0
    for li in layer_ids:
        li = int(li)
        z = _layer_mid_z(li, zmin, zmax, layer_mm)
        seg = _slice_segments(surf, z, cx, cy)
        if len(seg) == 0:
            continue
        e = _slice_extent_mm(seg)
        if e > best_e:
            best_e, outline = e, np.asarray(seg, dtype=float)
        h, _ = _hatch_layer(
            seg, li, strategy=strategy, hatch_mm=hatch_mm, base_angle_deg=base_angle_deg,
            rotation_deg=layer_rotation_deg, split_w_mm=split_w, min_len_mm=0.5 * hatch_mm,
        )
        zl = float(z - zmin)  # local z from domain floor
        z_mids.append(float(z))
        if len(h["x0"]):
            rows.append(np.column_stack(
                [h["x0"], h["y0"], h["x1"], h["y1"], np.full(len(h["x0"]), zl), np.zeros(len(h["x0"]))]
            ))
        rows.append(np.column_stack([seg, np.full(len(seg), zl), np.ones(len(seg))]))
    if not rows:
        raise ValueError("The STL / cake slice has no hatchable cross-section.")
    segs = np.vstack(rows)
    ox, oy, oz = float(xmin - pad_mm), float(ymin - pad_mm), float(zmin)
    Lx = float((xmax + pad_mm) - ox)
    Ly = float((ymax + pad_mm) - oy)
    Lz = float(max(zmax - zmin, layer_mm))
    local = segs.copy()
    local[:, [0, 2]] -= ox
    local[:, [1, 3]] -= oy
    outline_local = None
    if outline is not None and len(outline):
        outline_local = outline.copy()
        outline_local[:, [0, 2]] -= ox
        outline_local[:, [1, 3]] -= oy
    return {
        "segments_mm": local,
        "origin_mm": [ox, oy, oz],
        "size_mm": [Lx, Ly, Lz],
        "mesh_aabb_mm": [xmin, xmax, ymin, ymax, zmin, zmax],
        "layer_mm": float(layer_mm),
        "first_layer": int(layer_ids[0]),
        "n_layers": int(len(z_mids)),
        "n_layers_part": int(n_total),
        "z_mid_abs_mm": z_mids,
        "n_hatch": int(np.sum(segs[:, 5] == 0)),
        "n_contour": int(np.sum(segs[:, 5] == 1)),
        "band_placement": "full_height",
        "xy_from_mesh": True,
        "outline_mm": outline_local,
        "full_height": True,
        "surf": surf,
    }


def band_inside_mask(band: Dict[str, Any], shape, voxel_m: float, *, skin_mm: float = 0.0) -> np.ndarray:
    """(nx, ny, nz) bool: voxel centers inside the part (optionally eroded by ``skin_mm``).

    Thin bands evaluate once per process layer. Full-height cake-slice domains
    evaluate at each voxel's absolute z so the solid follows the part through
    the whole build.
    """
    from scipy.ndimage import minimum_filter

    from .stl_fit import _points_inside_mesh

    nx, ny, nz = [int(v) for v in shape]
    v_mm = float(voxel_m) * 1e3
    ox, oy, oz = [float(v) for v in band["origin_mm"]]
    xs = ox + (np.arange(nx) + 0.5) * v_mm
    ys = oy + (np.arange(ny) + 0.5) * v_mm
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    flat_xy = np.column_stack([X.ravel(), Y.ravel()])
    z_local = (np.arange(nz) + 0.5) * v_mm
    z_abs = oz + z_local
    r = int(round(skin_mm / v_mm)) if skin_mm > 0 else 0
    out = np.zeros((nx, ny, nz), dtype=bool)
    cache: Dict[int, np.ndarray] = {}
    full_height = bool(band.get("full_height"))
    layer_mm = float(band["layer_mm"])
    z_mids = [float(z) for z in band["z_mid_abs_mm"]]
    for iz in range(nz):
        if full_height:
            # Quantize to ~1 voxel so neighbouring z planes share a cache entry
            key = int(round(float(z_abs[iz]) / max(v_mm, 1e-6)))
            z_query = float(z_abs[iz])
        else:
            k = int(np.clip(z_local[iz] / max(layer_mm, 1e-9), 0, len(z_mids) - 1))
            key = k
            z_query = z_mids[k]
        if key not in cache:
            pts = np.column_stack([flat_xy, np.full(len(flat_xy), z_query)])
            m = _points_inside_mesh(band["surf"], pts).reshape(nx, ny)
            if r > 0:
                m = minimum_filter(m, size=2 * r + 1, mode="constant", cval=False)
            cache[key] = m
        out[:, :, iz] = cache[key]
    return out
