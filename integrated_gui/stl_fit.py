"""STL load + auto-fit (scale/center) onto the build plate."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_and_fit_stl(
    stl_path: str,
    plate_width_mm: float = 120.0,
    plate_height_mm: float = 120.0,
    *,
    margin_mm: float = 2.0,
    target_height_mm: Optional[float] = None,
    clip_height_mm: Optional[float] = None,
    z_on_plate: bool = True,
    sector_fraction: float = 1.0,
    sector_start_deg: float = 0.0,
) -> Tuple[Any, Dict[str, float]]:
    """
    Load an STL with PyVista, uniformly scale so XY fits inside the plate
    (with margin), optionally scale/clip height, and sit on z=0 centered on the plate.

    ``clip_height_mm`` keeps only the bottom slab z∈[0, clip] after fit (for a
    short Raptor band later). ``target_height_mm`` uniformly rescales instead.

    ``sector_fraction`` < 1 keeps a "cake slice" wedge about the vertical axis through
    the part's XY center, starting at ``sector_start_deg`` (0 = +x, counter-clockwise).

    Returns (pyvista.PolyData mesh in plate mm coords, meta dict).
    """
    import pyvista as pv

    if not os.path.isfile(stl_path):
        raise FileNotFoundError(stl_path)

    mesh = pv.read(stl_path)
    if mesh.n_points == 0:
        raise ValueError(f"Empty mesh: {stl_path}")

    # Work in float64
    mesh = mesh.copy(deep=True)
    mesh.points = np.asarray(mesh.points, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=float)  # xmin,xmax,ymin,ymax,zmin,zmax
    sx0 = max(bounds[1] - bounds[0], 1e-12)
    sy0 = max(bounds[3] - bounds[2], 1e-12)
    sz0 = max(bounds[5] - bounds[4], 1e-12)

    usable_w = max(plate_width_mm - 2 * margin_mm, 1.0)
    usable_h = max(plate_height_mm - 2 * margin_mm, 1.0)
    scale_xy = min(usable_w / sx0, usable_h / sy0)
    scale = float(scale_xy)
    if target_height_mm is not None and sz0 * scale > float(target_height_mm):
        scale = float(target_height_mm) / sz0

    mesh.scale(scale, inplace=True)
    bounds = np.asarray(mesh.bounds, dtype=float)
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz_min = bounds[4]
    tx = plate_width_mm / 2.0 - cx
    ty = plate_height_mm / 2.0 - cy
    tz = -cz_min if z_on_plate else 0.0
    mesh.translate([tx, ty, tz], inplace=True)

    clipped = False
    clip_h = None
    if clip_height_mm is not None and float(clip_height_mm) > 0:
        clip_h = float(clip_height_mm)
        b0 = np.asarray(mesh.bounds, dtype=float)
        if b0[5] > clip_h + 1e-9:
            # Keep bottom slab on the plate (z=0 … clip_h)
            try:
                mesh = mesh.clip_box(
                    bounds=(
                        b0[0] - 1.0,
                        b0[1] + 1.0,
                        b0[2] - 1.0,
                        b0[3] + 1.0,
                        -1.0,
                        clip_h,
                    ),
                    invert=False,
                    crinkle=False,
                )
                clipped = True
            except Exception:
                try:
                    mesh = mesh.clip(normal=(0, 0, 1), origin=(0, 0, clip_h), invert=True)
                    clipped = True
                except Exception:
                    clipped = False

    frac = float(sector_fraction or 1.0)
    sector_center = None
    full_volume = None
    if 0.0 < frac < 0.999:
        b0 = np.asarray(mesh.bounds, dtype=float)
        sector_center = (0.5 * (b0[0] + b0[1]), 0.5 * (b0[2] + b0[3]))
        try:
            full_volume = float(_prepare_closed_surface(mesh).volume)
        except Exception:
            full_volume = None
        mesh = clip_sector(mesh, sector_center, frac, float(sector_start_deg))

    b = np.asarray(mesh.bounds, dtype=float)
    aabb_vol = float(
        max(b[1] - b[0], 0.0) * max(b[3] - b[2], 0.0) * max(b[5] - b[4], 0.0)
    )
    solid_vol = aabb_vol
    try:
        closed = _prepare_closed_surface(mesh)
        vol = float(getattr(closed, "volume", 0.0) or 0.0)
        if np.isfinite(vol) and vol > 1e-6:
            solid_vol = vol
            mesh = closed  # reuse prepared surface for later enclosure tests
    except Exception:
        pass

    b = np.asarray(mesh.bounds, dtype=float)
    meta = {
        "stl_path": os.path.abspath(stl_path),
        "scale": scale,
        "plate_width_mm": float(plate_width_mm),
        "plate_height_mm": float(plate_height_mm),
        "margin_mm": float(margin_mm),
        "clip_height_mm": float(clip_h) if clip_h is not None else None,
        "height_clipped": bool(clipped),
        "xmin": float(b[0]),
        "xmax": float(b[1]),
        "ymin": float(b[2]),
        "ymax": float(b[3]),
        "zmin": float(b[4]),
        "zmax": float(b[5]),
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "aabb_volume_mm3": aabb_vol,
        "volume_mm3": float(solid_vol),
        "sector_fraction": frac if sector_center is not None else 1.0,
        "sector_start_deg": float(sector_start_deg) if sector_center is not None else 0.0,
        "sector_center_mm": list(sector_center) if sector_center is not None else None,
        "full_part_volume_mm3": full_volume,
    }
    return mesh, meta


def clip_sector(mesh, center_xy, fraction: float, start_deg: float = 0.0):
    """Keep a wedge of angle ``fraction * 360°`` about the vertical axis at ``center_xy``.

    Cut faces are capped so the result stays a closed solid (volume, pore placement).
    """
    frac = float(np.clip(fraction, 1e-3, 1.0))
    if frac >= 0.999:
        return mesh
    a = np.deg2rad(float(start_deg))
    b = a + 2.0 * np.pi * frac
    origin = (float(center_xy[0]), float(center_xy[1]), 0.0)
    # Left of the start ray, right of the end ray
    n_start = (-np.sin(a), np.cos(a), 0.0)
    n_end = (np.sin(b), -np.cos(b), 0.0)

    if frac > 0.5 + 1e-9:
        raise ValueError("Sector fraction must be <= 1/2 (convex wedge) or 1 (whole part).")
    # Convex wedge: intersection of two capped half-space cuts
    try:
        out = mesh.extract_surface(algorithm="dataset_surface")
    except TypeError:
        out = mesh.extract_surface()
    src = out.triangulate().clean()
    if frac >= 0.5 - 1e-9:
        cuts_list = [[n_start]]
    else:
        cuts_list = [[n_start, n_end], [n_end, n_start]]
    last_exc: Optional[Exception] = None
    fallback = None
    # Cap triangulation is sensitive to planes grazing existing vertices (both planes
    # contain the center axis). Retry with swapped order / planes nudged outward until
    # the result is watertight; a leaky cap breaks inside/outside tests for pores.
    for eps in (0.0, 1e-3, 1e-2, 3e-2, 1e-1):
        for cuts in cuts_list:
            try:
                out = src
                for n in cuts:
                    o = tuple(origin[i] - eps * n[i] for i in range(3))
                    out = out.clip_closed_surface(normal=n, origin=o)
                if out.n_cells == 0:
                    raise ValueError("Cake slice is empty — try a different start angle.")
                out = out.triangulate()
                if out.is_manifold:
                    return out
                fallback = fallback or out
            except Exception as exc:
                last_exc = exc
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not cut the cake slice at this angle ({last_exc}); try another start angle.")


def _prepare_closed_surface(mesh):
    """Return a triangulated surface suitable for enclosure tests."""
    import pyvista as pv

    try:
        # Already a closed triangle surface (e.g. capped cake slice): re-extracting can
        # break the cap seams, so keep it as is.
        if isinstance(mesh, pv.PolyData) and mesh.is_all_triangles and mesh.is_manifold:
            return mesh
    except Exception:
        pass
    try:
        surf = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
    except Exception:
        surf = mesh
    try:
        # Helps select_enclosed_points on imperfect CAD meshes
        if hasattr(surf, "is_manifold") and not surf.is_manifold:
            surf = surf.fill_holes(1000)
    except Exception:
        pass
    return surf


_RAY_CACHE: Dict[int, Any] = {}


def _build_ray_index(mesh, target_per_cell: float = 8.0):
    """XY bucket grid of triangles for vertical-ray parity tests."""
    faces = np.asarray(mesh.faces)
    if faces.size == 0 or not mesh.is_all_triangles:
        return None
    tri = faces.reshape(-1, 4)[:, 1:]
    v = np.asarray(mesh.points, dtype=np.float64)
    p0, p1, p2 = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    b = np.asarray(mesh.bounds, dtype=float)
    nt = len(tri)
    n_side = int(np.clip(np.sqrt(nt / target_per_cell), 16, 2048))
    x0, y0 = b[0] - 1e-6, b[2] - 1e-6
    cw = max((b[1] - b[0]) + 2e-6, 1e-9) / n_side
    ch = max((b[3] - b[2]) + 2e-6, 1e-9) / n_side
    xs = np.stack([p0[:, 0], p1[:, 0], p2[:, 0]], 1)
    ys = np.stack([p0[:, 1], p1[:, 1], p2[:, 1]], 1)
    i0 = np.clip(((xs.min(1) - x0) / cw).astype(np.int64), 0, n_side - 1)
    i1 = np.clip(((xs.max(1) - x0) / cw).astype(np.int64), 0, n_side - 1)
    j0 = np.clip(((ys.min(1) - y0) / ch).astype(np.int64), 0, n_side - 1)
    j1 = np.clip(((ys.max(1) - y0) / ch).astype(np.int64), 0, n_side - 1)
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    cnt = ni * nj
    tid = np.repeat(np.arange(nt), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    rep_nj = np.repeat(nj, cnt)
    ci = np.repeat(i0, cnt) + off // rep_nj
    cj = np.repeat(j0, cnt) + off % rep_nj
    cell = ci * n_side + cj
    order = np.argsort(cell, kind="stable")
    cell_sorted = cell[order]
    tri_ids = tid[order].astype(np.int64)
    starts = np.searchsorted(cell_sorted, np.arange(n_side * n_side + 1)).astype(np.int64)
    return {
        "p0": p0, "p1": p1, "p2": p2,
        "x0": x0, "y0": y0, "cw": cw, "ch": ch, "n": n_side,
        "starts": starts, "tri_ids": tri_ids, "bounds": b,
    }


_RAY_KERNEL = None


def _ray_kernel():
    global _RAY_KERNEL
    if _RAY_KERNEL is not None:
        return _RAY_KERNEL
    import numba as nb

    @nb.njit(parallel=True, cache=True)
    def _parity(pts, p0, p1, p2, x0, y0, cw, ch, n, starts, tri_ids, out):
        for k in nb.prange(pts.shape[0]):
            px = pts[k, 0]
            py = pts[k, 1]
            pz = pts[k, 2]
            i = int((px - x0) / cw)
            j = int((py - y0) / ch)
            if i < 0 or j < 0 or i >= n or j >= n:
                out[k] = False
                continue
            c = i * n + j
            hits = 0
            for s in range(starts[c], starts[c + 1]):
                t = tri_ids[s]
                ax = p0[t, 0]; ay = p0[t, 1]; az = p0[t, 2]
                bx = p1[t, 0]; by = p1[t, 1]; bz = p1[t, 2]
                cx = p2[t, 0]; cy = p2[t, 1]; cz = p2[t, 2]
                d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                if d == 0.0:
                    continue
                w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / d
                w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / d
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                if w0 * az + w1 * bz + w2 * cz > pz:
                    hits += 1
            out[k] = (hits % 2) == 1

    _RAY_KERNEL = _parity
    return _parity


def _points_inside_fast(mesh, cand: np.ndarray) -> Optional[np.ndarray]:
    """Vertical-ray parity test (Numba); None if unavailable for this mesh."""
    key = id(mesh)
    sig = (int(mesh.n_cells), int(mesh.n_points), tuple(np.round(mesh.bounds, 9)))
    entry = _RAY_CACHE.get(key)
    if entry is None or entry[0] != sig:
        idx = None
        try:
            # Parity counting needs a closed surface
            if mesh.is_manifold:
                idx = _build_ray_index(mesh)
        except Exception:
            idx = None
        if len(_RAY_CACHE) > 8:
            _RAY_CACHE.clear()
        entry = (sig, idx)
        _RAY_CACHE[key] = entry
    idx = entry[1]
    if not idx:
        return None
    try:
        kern = _ray_kernel()
    except Exception:
        return None
    pts = np.ascontiguousarray(cand, dtype=np.float64)
    out = np.zeros(len(pts), dtype=np.bool_)
    kern(
        pts, idx["p0"], idx["p1"], idx["p2"], idx["x0"], idx["y0"], idx["cw"], idx["ch"],
        idx["n"], idx["starts"], idx["tri_ids"], out,
    )
    return out


def _points_inside_mesh(mesh, cand: np.ndarray) -> np.ndarray:
    """Boolean mask: True where candidates are enclosed by the mesh."""
    import pyvista as pv

    if len(cand) == 0:
        return np.zeros(0, dtype=bool)
    try:
        fast = _points_inside_fast(mesh, cand)
        if fast is not None:
            return fast
    except Exception:
        pass
    cloud = pv.PolyData(np.asarray(cand, dtype=float))
    try:
        if hasattr(cloud, "select_interior_points"):
            sel = cloud.select_interior_points(mesh, tolerance=1e-5, check_surface=False)
            key = "selected_points" if "selected_points" in sel.array_names else "SelectedPoints"
            return np.asarray(sel[key], dtype=bool)
        sel = cloud.select_enclosed_points(mesh, tolerance=1e-5, check_surface=False)
        return np.asarray(sel["SelectedPoints"], dtype=bool)
    except Exception:
        # Last resort: signed distance <= 0 means inside for outward-normal surfaces
        try:
            dist = cloud.compute_implicit_distance(mesh)
            d = np.asarray(dist["implicit_distance"], dtype=float)
            return d <= 0.0
        except Exception:
            return np.zeros(len(cand), dtype=bool)


def sample_points_in_mesh(
    mesh,
    n: int,
    rng: np.random.Generator,
    *,
    gas_flow: str = "+x",
    flow_perturbation: float = 0.0,
    gas_flow_sharpness: float = 2.0,
    flow_dirs: Optional[Dict[str, Tuple[float, float]]] = None,
    prepared: bool = False,
) -> np.ndarray:
    """
    Rejection-sample points that are strictly enclosed by the STL mesh.

    If flow_perturbation > 0, mix uniform + flow-biased proposals, but every
    accepted point must still pass the enclosure test (never leave the CAD body).
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    if not prepared:
        mesh = _prepare_closed_surface(mesh)
    b = np.asarray(mesh.bounds, dtype=float)
    alpha = float(np.clip(flow_perturbation, 0.0, 1.0))
    dirs = flow_dirs or {
        "+x": (1.0, 0.0),
        "-x": (-1.0, 0.0),
        "+y": (0.0, 1.0),
        "-y": (0.0, -1.0),
    }
    dx, dy = dirs.get(gas_flow, dirs["+x"])
    s = float(max(0.0, gas_flow_sharpness))
    max_w = float(np.exp(s)) if s > 0 else 1.0

    out = np.empty((n, 3), dtype=float)
    filled = 0
    attempts = 0
    # Laptop-friendly: fewer oversized batches than the old n×120 loop
    max_attempts = max(4000, int(n * 40))
    while filled < n and attempts < max_attempts:
        need = n - filled
        batch = min(max(need * 4, 256), 1_000_000)
        cand = np.column_stack(
            [
                rng.uniform(b[0], b[1], batch),
                rng.uniform(b[2], b[3], batch),
                rng.uniform(b[4], b[5], batch),
            ]
        )
        inside = _points_inside_mesh(mesh, cand)
        cand = cand[inside]
        if len(cand) == 0:
            attempts += batch
            continue

        if alpha > 0:
            proj = cand[:, 0] * dx + cand[:, 1] * dy
            pmin = b[0] * dx + b[2] * dy
            pmax = b[1] * dx + b[3] * dy
            span = max(1e-12, pmax - pmin)
            t = np.clip((proj - pmin) / span, 0.0, 1.0)
            flow_w = np.exp(s * t) / max_w
            keep_prob = (1.0 - alpha) + alpha * flow_w
            keep = rng.random(len(cand)) < keep_prob
            cand = cand[keep]

        take = min(len(cand), need)
        if take:
            out[filled : filled + take] = cand[:take]
            filled += take
        attempts += batch

    if filled < n:
        # Pad with verified interior points
        while filled < n and attempts < max_attempts * 2:
            cand = np.column_stack(
                [
                    rng.uniform(b[0], b[1], 512),
                    rng.uniform(b[2], b[3], 512),
                    rng.uniform(b[4], b[5], 512),
                ]
            )
            kept = cand[_points_inside_mesh(mesh, cand)]
            take = min(len(kept), n - filled)
            if take:
                out[filled : filled + take] = kept[:take]
                filled += take
            attempts += 512
        if filled < n:
            raise RuntimeError(
                f"Could only place {filled}/{n} pores inside the STL. "
                "The mesh may be open/non-manifold; try repairing the CAD."
            )
    return out
