"""Small Raptor RVE helper for Combined single-cuboid LoF."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MELT = os.path.join(
    ROOT,
    "Raptor",
    "examples",
    "data",
    "meltPoolData",
    "ULI_v1700_theta0_widths.txt",
)

ESD_COL = "Equivalent Spherical Diameter (um)"


def ensure_raptor_on_path() -> str:
    """Make vendored Raptor importable (editable install is often missing)."""
    raptor_src = os.path.join(ROOT, "Raptor", "src")
    if os.path.isdir(raptor_src) and raptor_src not in sys.path:
        sys.path.insert(0, raptor_src)
    return raptor_src


def _sphere_vol_mm3(esd_um: np.ndarray) -> np.ndarray:
    r_mm = np.asarray(esd_um, dtype=float) / 2000.0
    return (4.0 / 3.0) * np.pi * r_mm**3


def _resolve_domain_m(cfg: Dict[str, Any]) -> tuple[float, float, float]:
    """Return (Lx, Ly, Lz) in meters for the Raptor domain."""
    size_mm = cfg.get("size_xyz_mm")
    if size_mm is not None and len(size_mm) >= 3:
        L = max(1e-5, float(size_mm[0]) * 1e-3)
        W = max(1e-5, float(size_mm[1]) * 1e-3)
        H = max(1e-5, float(size_mm[2]) * 1e-3)
        return L, W, H
    edge = float(cfg.get("rve_edge_m", 5.0e-4) or 5.0e-4)
    edge = max(1e-5, edge)
    return edge, edge, edge


def _adaptive_voxel_m(Lx: float, Ly: float, Lz: float, cfg: Dict[str, Any]) -> float:
    """Choose voxel size so the full domain stays under a voxel budget."""
    hatch = float(cfg.get("hatch_m", 140e-6) or 140e-6)
    user = cfg.get("voxel_resolution_m")
    max_voxels = int(cfg.get("raptor_max_voxels", 1_200_000) or 1_200_000)
    target_axis = int(cfg.get("raptor_target_voxels_per_axis", 100) or 100)
    voxel = float(user) if user not in (None, "", 0, 0.0) else max(Lx, Ly, Lz) / max(8, target_axis)
    # Prefer resolving ~2 voxels per hatch when budget allows
    voxel = min(voxel, max(hatch / 2.0, 1e-6))
    for _ in range(40):
        nx = max(8, int(np.ceil(Lx / voxel)))
        ny = max(8, int(np.ceil(Ly / voxel)))
        nz = max(8, int(np.ceil(Lz / voxel)))
        if nx * ny * nz <= max_voxels:
            break
        voxel *= 1.12
    return float(voxel)


def _path_vectors_from_strategy(
    cfg: Dict[str, Any],
    *,
    Lx: float,
    Ly: float,
    Lz: float,
    power: float,
    velocity: float,
    hatch: float,
    layer: float,
    rotation: float,
    _log: Callable[[str], None],
):
    """Build Raptor PathVectors. Cuboids use Raptor's ScanPathBuilder."""
    from raptor.structures import PathVector
    from raptor.utilities import ScanPathBuilder

    band = cfg.get("_stl_band")
    if band is not None:
        # Real STL cross-section: slicer hatch + contour vectors (local mm), layer by layer
        segs = np.asarray(band["segments_mm"], dtype=float)
        path_vectors = []
        t = 0.0
        for x0, y0, x1, y1, z, _kind in segs:
            p0 = np.array([x0, y0, z], dtype=float) * 1e-3
            p1 = np.array([x1, y1, z], dtype=float) * 1e-3
            dist = float(np.linalg.norm(p1 - p0))
            if dist <= 0.0:
                continue
            dt = dist / velocity if velocity > 1e-12 else 0.0
            pv = PathVector(p0, p1, t, t + dt)
            pv.set_coordinate_frame()
            path_vectors.append(pv)
            t += dt
        _log(
            f"Scan paths from the STL cross-section: layers {band['first_layer']}–"
            f"{band['first_layer'] + band['n_layers'] - 1} of {band['n_layers_part']}  "
            f"hatch vectors={band['n_hatch']:,}  contour segments={band['n_contour']:,}"
        )
        return path_vectors

    # Same call as examples/api_example/rve.py: tracks extend by the longest edge.
    bound_box = np.array([[0.0, 0.0, 0.0], [Lx, Ly, Lz]], dtype=float)
    scan_extension = float(max(Lx, Ly, Lz))
    builder = ScanPathBuilder(
        bound_box,
        power,
        velocity,
        hatch,
        layer,
        rotation,
        scan_extension,
        int(cfg.get("extra_layers", 0)),
    )
    builder.generate_layers()
    path_vectors = builder.process_vectors()
    _log(
        f"Scan paths from Raptor ScanPathBuilder  "
        f"hatch={hatch * 1e6:.1f} um  "
        f"layer={layer * 1e6:.1f} um  "
        f"effective layer={builder.effective_layer_thickness * 1e6:.1f} um  "
        f"powder consolidation={builder.powder_consolidation:g}  "
        f"rotation={rotation:g} deg  extension={scan_extension * 1e3:.2f} mm  "
        f"vectors={len(path_vectors)}"
    )
    return path_vectors


def run_raptor_rve(
    cfg: Dict[str, Any],
    *,
    outdir: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run Raptor LoF on the Combined cuboid domain (or a cubic RVE).

    cfg keys (optional):
      size_xyz_mm [L,W,H] — full Combined cuboid (preferred)
      rve_edge_m — cubic fallback when size_xyz_mm omitted
      scan_strategy, max_scan_layers, base_angle_deg, batch_rows/cols
      power_w, velocity_m_s, hatch_m, layer_m, rotation_deg,
      voxel_resolution_m, raptor_max_voxels, n_modes, random_seed, …
    """
    _log = log or (lambda _m: None)
    ensure_raptor_on_path()

    from integrated_gui.compute_backend import apply_env, resolve_device
    from raptor.api import (
        create_grid,
        compute_morphology,
        write_vtk,
        write_morphology,
    )

    compute_info = resolve_device(cfg.get("compute_device", "auto"))
    apply_env(compute_info)
    _log("Raptor melt-mask kernel: Numba (ORNL Raptor)")

    Lx, Ly, Lz = _resolve_domain_m(cfg)
    edge_m = float(max(Lx, Ly, Lz))
    voxel_m = _adaptive_voxel_m(Lx, Ly, Lz, cfg)
    power = float(cfg.get("power_w", 370))
    velocity = float(cfg.get("velocity_m_s", 1.3))
    hatch = float(cfg.get("hatch_m", 140e-6))
    layer = float(cfg.get("layer_m", 30e-6))
    rotation = float(cfg.get("rotation_deg", 67))
    n_modes = int(cfg.get("n_modes", 50))
    # Wider melt on coarse full-geometry voxels → grey melted bulk (red = LoF only)
    width_scale = float(cfg.get("width_scale", 1.0))
    depth_scale = float(cfg.get("depth_scale", 0.9))
    height_scale = float(cfg.get("height_scale", 0.45))
    stl_band_cfg = cfg.get("_stl_band") or {}
    full_height_stl = bool(stl_band_cfg.get("full_height"))
    if voxel_m > hatch * 0.45 and "width_scale" not in cfg:
        width_scale = max(width_scale, min(4.0, 2.2 * voxel_m / max(hatch, 1e-9)))
        depth_scale = max(depth_scale, min(2.5, 1.6 * voxel_m / max(hatch, 1e-9)))
    if full_height_stl and "depth_scale" not in cfg:
        # Bridge sparse hatch planes through the cake-slice height
        depth_scale = max(depth_scale, min(6.0, 3.5 * voxel_m / max(hatch, 1e-9)))
        height_scale = max(height_scale, min(3.0, 2.0 * voxel_m / max(hatch, 1e-9)))
        width_scale = max(width_scale, min(6.0, 3.0 * voxel_m / max(hatch, 1e-9)))
    seed = cfg.get("random_seed", 42)
    seed = None if seed is None or str(seed).strip() == "" else int(seed)
    enable_random = bool(cfg.get("enable_random_phases", True))

    from integrated_gui import meltpool_library as mpl

    depth_shape = float(cfg.get("depth_shape", 1))
    height_shape = float(cfg.get("height_shape", 1))
    lib_cfg = cfg.get("library") or None
    lib_cond = None
    if lib_cfg:
        lib_cond = mpl.resolve_condition(lib_cfg, velocity * 1000.0)
        if lib_cfg.get("use_eta", True):
            # Coarse-voxel inflation below still applies on top of the library ratios
            depth_scale = float(lib_cond.eta_d)
            height_scale = float(lib_cond.eta_c)
        if lib_cfg.get("use_depth_shape", True):
            depth_shape = float(lib_cond.depth_shape)
        width_files = dict(lib_cond.width_files)
        _log(
            f"Melt-pool library: V={lib_cond.velocity_mm_s:g} mm/s"
            + (
                f" (nearest to process {velocity * 1000.0:g} mm/s)"
                if abs(lib_cond.velocity_mm_s - velocity * 1000.0) > 1e-6
                else ""
            )
            + f"  eta={lib_cfg.get('eta_source', 'flucWs')}  D scale={depth_scale:.4g}  "
            f"H scale={height_scale:.4g}  depth shape={depth_shape:.4g}  "
            f"θ files={sorted(width_files)}"
        )
    else:
        melt_path = os.path.abspath(cfg.get("melt_pool_data") or DEFAULT_MELT)
        if not os.path.isfile(melt_path):
            raise FileNotFoundError(f"Melt-pool data not found: {melt_path}")
        width_files = {0.0: melt_path}
    if voxel_m > hatch * 0.45 and "width_scale" not in cfg:
        depth_scale = max(depth_scale, min(2.5, 1.6 * voxel_m / max(hatch, 1e-9)))

    min_point = np.array([0.0, 0.0, 0.0], dtype=float)
    max_point = np.array([Lx, Ly, Lz], dtype=float)
    bound_box = np.array([min_point, max_point])

    nx = max(8, int(np.ceil(Lx / voxel_m)))
    ny = max(8, int(np.ceil(Ly / voxel_m)))
    nz = max(8, int(np.ceil(Lz / voxel_m)))
    _log(
        f"Raptor domain LxWxH={Lx*1e3:.2f}x{Ly*1e3:.2f}x{Lz*1e3:.2f} mm  "
        f"voxel={voxel_m*1e6:.2f} um  grid~{nx}x{ny}x{nz}  "
        f"P={power} W  v={velocity} m/s"
    )
    grid = create_grid(voxel_m, bound_box=bound_box)
    cfg_paths = {**cfg, "_voxel_mm_hint": voxel_m * 1e3}
    path_vectors = _path_vectors_from_strategy(
        cfg_paths,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        power=power,
        velocity=velocity,
        hatch=hatch,
        layer=layer,
        rotation=rotation,
        _log=_log,
    )
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        try:
            from raptor.path_viz import write_path_vectors_csv

            path_csv = os.path.join(outdir, "path_vectors.csv")
            n_path_saved = write_path_vectors_csv(
                path_csv,
                path_vectors,
                bounds_min=min_point,
                bounds_max=max_point,
                layer_height=layer,
            )
            _log(f"Wrote scan paths: {path_csv} ({n_path_saved} segments in domain)")
        except Exception as exc:
            _log(f"path_vectors.csv skipped: {exc}")

    porosity, theta_groups, theta_files = mpl.compute_porosity_grouped(
        grid,
        path_vectors,
        width_files=width_files,
        velocity_mm_s=velocity * 1000.0,
        n_modes=n_modes,
        width_scale=width_scale,
        depth_scale=depth_scale,
        height_scale=height_scale,
        depth_shape=depth_shape,
        height_shape=height_shape,
        enable_random_phases=enable_random,
        seed=seed,
        theta_mode=str((lib_cfg or {}).get("theta_mode", "averaged")) if lib_cfg else "fixed",
        theta_deg=float((lib_cfg or {}).get("theta_deg", 0.0)),
        angle_offset_deg=float((lib_cfg or {}).get("angle_offset_deg", 0.0)),
        lambda_mean=float((lib_cfg or {}).get("lambda_mean", mpl.LAMBDA_MEAN)) if lib_cfg else 1.0,
        log=_log,
        tile_width=cfg.get("tile_width"),
        spectral_error_fraction=float(cfg.get("spectral_error_fraction", 0.25)),
        memory_limit_mb=cfg.get("memory_limit_mb"),
    )
    phase = np.asarray(porosity)
    if not phase.flags.writeable:
        phase = phase.copy()
    # Raptor runs in a local box; the in-page view and pore tables use plate coordinates
    world_origin = np.asarray(grid.origin, dtype=float).copy()
    plate_origin = cfg.get("plate_origin_mm")
    if plate_origin is None and cfg.get("_stl_band") is not None:
        plate_origin = cfg["_stl_band"]["origin_mm"]
    if plate_origin is not None:
        world_origin = world_origin + np.asarray(plate_origin, dtype=float) * 1e-3
    # Whole build: first layer has nothing to fuse to below and the last is never
    # remelted, so those faces are not internal LoF. Measure the interior only.
    interior = phase
    interior_z0 = 0
    n_face_layers = int(cfg.get("exclude_face_layers", 0) or 0)
    band = cfg.get("_stl_band")
    band_meta = None
    if band is not None and phase.ndim == 3:
        from scipy.ndimage import minimum_filter

        from integrated_gui.raptor_view import PHASE_OUTSIDE
        from integrated_gui.stl_raptor_band import band_inside_mask

        part = band_inside_mask(band, phase.shape, voxel_m)
        skin_mm = float(cfg.get("band_skin_mm", 0.1) or 0.0)
        r = int(round(skin_mm / (voxel_m * 1e3)))
        region = minimum_filter(part, size=(2 * r + 1, 2 * r + 1, 1), mode="constant", cval=False) if r else part.copy()
        e = int(np.ceil(n_face_layers * layer / max(voxel_m, 1e-12))) if n_face_layers > 0 else 0
        if e and phase.shape[2] > 3 * e:
            region[:, :, :e] = False
            region[:, :, -e:] = False
        phase[~part] = PHASE_OUTSIDE
        # Statistics only inside the part cross-section, away from the contour skin and
        # the band's own top/bottom layers; everything else counts as dense.
        interior = np.where(region, phase, np.int8(1)).astype(phase.dtype)
        n_total = int(region.sum())
        n_pore = int(np.sum(interior == 0))
        z_span = float(band["size_mm"][2]) if band.get("full_height") else (
            band["n_layers"] * band["layer_mm"]
        )
        band_meta = {
            "origin_mm": band["origin_mm"],
            "first_layer": band["first_layer"],
            "n_layers": band["n_layers"],
            "n_layers_part": band["n_layers_part"],
            "z_bottom_mm": band["origin_mm"][2],
            "z_top_mm": band["origin_mm"][2] + z_span,
            "part_voxel_fraction": float(part.mean()),
            "measured_volume_mm3": n_total * (voxel_m * 1e3) ** 3,
            "skin_mm": skin_mm,
            "band_placement": band.get("band_placement"),
            "xy_from_mesh": band.get("xy_from_mesh"),
            "mesh_aabb_mm": band.get("mesh_aabb_mm"),
            "full_height": bool(band.get("full_height")),
        }
        # Persist cake-slice / part silhouette for the in-page Raptor view
        outline = band.get("outline_mm")
        if outdir and outline is not None and len(np.asarray(outline)):
            try:
                np.savez_compressed(
                    os.path.join(outdir, "stl_outline.npz"),
                    segments_mm=np.asarray(outline, dtype=np.float32),
                    origin_mm=np.asarray(band["origin_mm"], dtype=float),
                    size_mm=np.asarray(band["size_mm"], dtype=float),
                )
                band_meta["outline_npz"] = os.path.join(outdir, "stl_outline.npz")
            except Exception as exc:
                _log(f"STL outline save skipped: {exc}")
        _log(
            f"STL band: {part.mean()*100:.1f}% of the box is part; LoF measured on "
            f"{band_meta['measured_volume_mm3']:.3g} mm³ (contour skin {skin_mm:.2f} mm and "
            f"{n_face_layers} band face layers excluded)"
        )
        del part, region
    elif n_face_layers > 0 and phase.ndim == 3:
        e = int(np.ceil(n_face_layers * layer / max(voxel_m, 1e-12)))
        if phase.shape[2] > 4 * e:
            interior = phase[:, :, e:-e]
            interior_z0 = e
            _log(
                f"LoF measured on interior: excluded {n_face_layers} layers "
                f"({e} voxels) at the bottom and top faces"
            )
        n_total = int(interior.size)
        n_pore = int(np.sum(interior == 0))
    else:
        n_total = int(interior.size)
        n_pore = int(np.sum(interior == 0))
    phi = float(n_pore / n_total) if n_total else 0.0

    esd_um = np.array([], dtype=float)
    morph_path = ""
    z_idx = 0
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        vti_path = os.path.join(outdir, "rve.vti")
        morph_path = os.path.join(outdir, "rve_morphology.csv")
        try:
            # Local origin so the PyVista viewer lines up with path_vectors.csv
            write_vtk(grid.origin, grid.resolution, phase, vti_path)
            # Legacy name used by raptor.api.visualize / Open Raptor RVE
            if band_meta is None:
                try:
                    import shutil

                    shutil.copyfile(vti_path, os.path.join(outdir, "porosity_map.vti"))
                except Exception:
                    pass
        except Exception as exc:
            _log(f"Raptor VTK write skipped: {exc}")
        # Compact 3D view for the rotatable in-page figure: phase on the six outer
        # faces (strided) + every LoF pore position, in mm.
        try:
            if phase.ndim == 3:
                from integrated_gui.raptor_view import save_view3d_npz

                save_view3d_npz(
                    os.path.join(outdir, "view3d.npz"),
                    phase,
                    origin_m=world_origin,
                    voxel_m=voxel_m,
                    layer_m=layer,
                    hatch_m=hatch,
                )
        except Exception as exc:
            _log(f"Raptor view3d.npz skipped: {exc}")
        # LoF-max XY preview for Combined / Raptor hatch-space plots
        try:
            if phase.ndim == 3:
                pore_counts = np.sum(phase == 0, axis=(0, 1)).astype(np.int64)
                if pore_counts.size > 20:
                    # Skip the free top surface / bottom face so the slice shows interior LoF
                    edge = max(2, int(np.ceil(layer / max(voxel_m, 1e-12))) + 1)
                    pore_counts[:1] = -1
                    pore_counts[-edge:] = -1
                z_idx = (
                    int(np.argmax(pore_counts))
                    if int(np.max(pore_counts)) > 0
                    else int(phase.shape[2] // 2)
                )
                slice2d = phase[:, :, z_idx].astype(np.uint8)
            else:
                slice2d = phase.astype(np.uint8)
            # Keep the browser heatmap light on large STL cross-sections
            s2 = max(1, int(np.ceil(max(slice2d.shape) / 1500.0)))
            if s2 > 1:
                slice2d = slice2d[::s2, ::s2]
            np.savez_compressed(
                os.path.join(outdir, "preview.npz"),
                phase_slice=slice2d,
                pore_slice=(slice2d == 0).astype(np.uint8),
                origin=world_origin,
                resolution=float(np.asarray(grid.resolution).ravel()[0]) * s2,
                z_index=int(z_idx),
                n_pore_on_slice=int(np.sum(slice2d == 0)),
            )
            _log(
                f"Raptor preview slice z-index={z_idx} "
                f"(LoF voxels on slice={int(np.sum(slice2d == 0))})"
            )
        except Exception as exc:
            _log(f"Raptor preview.npz skipped: {exc}")
        # Off-screen 3D phase view for Combined panel
        try:
            from integrated_gui.raptor_worker import _write_phase_screenshot

            vti_for_shot = os.path.join(outdir, "rve.vti")
            png_path = os.path.join(outdir, "preview_3d.png")
            if os.path.isfile(vti_for_shot) and band_meta is None and not cfg.get("viz_only"):
                _write_phase_screenshot(vti_for_shot, png_path)
                _log(f"Raptor 3D preview written: {png_path}")
        except Exception as exc:
            _log(f"Raptor 3D preview skipped: {exc}")

        # V-spatter wake animation + accumulated landing field (full domain only)
        if not cfg.get("_skip_fine_lof") and not cfg.get("skip_spatter_anim") and band_meta is None:
            try:
                from integrated_gui.combined_spatter_anim import build_spatter_animation

                n_layers_anim = max(
                    1,
                    min(
                        int(cfg.get("max_scan_layers", 12) or 12),
                        max(1, int(np.ceil((Lz * 1e3) / max(layer * 1e3, 1e-6)))),
                    ),
                )
                spat = build_spatter_animation(
                    outdir=outdir,
                    length_mm=Lx * 1e3,
                    width_mm=Ly * 1e3,
                    height_mm=Lz * 1e3,
                    plate_w_mm=float(cfg.get("plate_w_mm", 250) or 250),
                    plate_h_mm=float(cfg.get("plate_h_mm", 250) or 250),
                    cuboid_cx=cfg.get("cuboid_cx_mm"),
                    cuboid_cy=cfg.get("cuboid_cy_mm"),
                    strategy=str(cfg.get("scan_strategy") or "horizontal"),
                    gas_flow=str(cfg.get("gas_flow") or "+x"),
                    n_layers=n_layers_anim,
                    n_frames=max(24, n_layers_anim * 4),
                    seed=int(seed or 42),
                    path_csv=os.path.join(outdir, "path_vectors.csv"),
                    # Long plume (~30 cm) opposite scan direction
                    wake_len_mm=float(cfg.get("spatter_wake_len_mm", 300.0) or 300.0),
                )
                if spat.get("ok"):
                    _log(f"Spatter build animation: {spat.get('gif')}")
                else:
                    _log(f"Spatter animation skipped: {spat.get('error')}")
            except Exception as exc:
                _log(f"Spatter animation skipped: {exc}")
    try:
        morphology = (
            {}
            if cfg.get("viz_only")
            else compute_morphology(
                np.ascontiguousarray(interior), voxel_m, ["area", "equivalent_diameter_area"]
            )
        )
        if outdir and morph_path:
            try:
                write_morphology(morphology, morph_path)
            except Exception:
                pass
        # equivalent_diameter_area is in meters (Raptor / skimage convention)
        diam = morphology.get("equivalent_diameter_area")
        if diam is not None:
            arr = np.asarray(diam, dtype=float).ravel()
            arr = arr[np.isfinite(arr) & (arr > 0)]
            esd_um = arr * 1.0e6  # m → µm
    except Exception as exc:
        _log(f"Raptor morphology skipped: {exc}")

    if len(esd_um) == 0:
        # Fallback synthetic sizes when RVE has no connected pore objects
        esd_um = np.array([40.0, 50.0, 60.0, 80.0, 100.0], dtype=float)
        _log("Raptor morphology empty — using fallback LoF ESD catalog")

    mean_v = float(np.mean(_sphere_vol_mm3(esd_um)))
    rho = float(phi / mean_v) if mean_v > 0 else 0.0
    size_catalog = pd.DataFrame({ESD_COL: esd_um, "pore_source": "lof"})

    # Fine hatch-scale LoF sample for density when the domain is coarse/large
    fine_meta = None
    if (
        not cfg.get("_skip_fine_lof")
        and band_meta is None
        and bool(cfg.get("fine_lof_for_density", True))
        and edge_m > 1.5e-3
    ):
        try:
            fine_edge = max(5.0e-4, 4.0 * hatch)
            fine_cfg = {
                **cfg,
                "rve_edge_m": fine_edge,
                "voxel_resolution_m": min(5.0e-6, hatch / 4.0),
                "raptor_max_voxels": 800_000,
                "width_scale": 1.0,
                "depth_scale": 0.8,
                "height_scale": 0.4,
                "_skip_fine_lof": True,
            }
            # Drop size_xyz so cubic fine RVE is used
            fine_cfg.pop("size_xyz_mm", None)
            fine_outdir = os.path.join(outdir, "lof_sample") if outdir else None
            _log(
                f"Fine LoF sample RVE edge={fine_edge*1e6:.0f} um for density catalog…"
            )
            fine = run_raptor_rve(fine_cfg, outdir=fine_outdir, log=_log)
            rho = float(fine["rho"])
            phi = float(fine["porosity_fraction"])
            esd_um = np.asarray(fine["esd_um"], dtype=float)
            size_catalog = fine["size_catalog"]
            mean_v = float(np.mean(_sphere_vol_mm3(esd_um))) if len(esd_um) else mean_v
            fine_meta = fine.get("summary")
            _log(
                f"Using fine-sample LoF phi={phi*100:.4g}%  rho~={rho:.6g}/mm3 "
                f"for Combined pore counts"
            )
        except Exception as exc:
            _log(f"Fine LoF sample skipped — using full-domain porosity: {exc}")

    summary = {
        "rve_edge_m": edge_m,
        "size_xyz_m": [Lx, Ly, Lz],
        "size_xyz_mm": [Lx * 1e3, Ly * 1e3, Lz * 1e3],
        "voxel_resolution_m": voxel_m,
        "power_w": power,
        "velocity_m_s": velocity,
        "hatch_m": hatch,
        "layer_m": layer,
        "scan_strategy": str(cfg.get("scan_strategy") or "raptor_default"),
        "width_scale": width_scale,
        "depth_scale": depth_scale,
        "height_scale": height_scale,
        "depth_shape": depth_shape,
        "height_shape": height_shape,
        "theta_groups": theta_groups,
        "melt_pool_files": theta_files,
        "library_condition": lib_cond.to_dict() if lib_cond is not None else None,
        "n_path_vectors": int(len(path_vectors)),
        "n_layers_scanned": int(len({round(float(p.start_point[2]) / layer) for p in path_vectors}))
        if path_vectors and hasattr(path_vectors[0], "start_point")
        else None,
        "excluded_face_layers": n_face_layers,
        "porosity_fraction": phi,
        "porosity_pct": phi * 100.0,
        "n_pore_voxels": n_pore,
        "n_voxels": n_total,
        "mean_esd_um": float(np.mean(esd_um)),
        "mean_sphere_volume_mm3": mean_v,
        "lof_density_per_mm3": rho,
        "n_morphology_pores": int(len(esd_um)),
        "morphology": morph_path,
        "compute_device": compute_info.device,
        "compute_detail": compute_info.detail,
        "z_index": int(z_idx),
        "fine_lof_sample": fine_meta,
        "stl_band": band_meta,
        "plate_origin_mm": (world_origin * 1e3).tolist(),
        "spatter_gif": os.path.join(outdir, "spatter_build.gif")
        if outdir and os.path.isfile(os.path.join(outdir, "spatter_build.gif"))
        else "",
        "spatter_accum_png": os.path.join(outdir, "spatter_accum.png")
        if outdir and os.path.isfile(os.path.join(outdir, "spatter_accum.png"))
        else "",
    }
    if phase.ndim == 3:
        try:
            from integrated_gui.lof_shape import lof_pore_shapes, summarize_shapes

            origin = world_origin + np.array([0.0, 0.0, interior_z0 * voxel_m])
            shapes = lof_pore_shapes(np.ascontiguousarray(interior), voxel_m, origin_m=origin)
            if len(shapes):
                o_mm = np.asarray(origin, dtype=float) * 1e3
                v_mm = float(voxel_m) * 1e3
                ix, iy, iz = interior.shape
                ijk = (shapes[["x_mm", "y_mm", "z_mm"]].to_numpy(float) - o_mm) / v_mm
                skin = max(2, int(np.ceil(0.5 / max(v_mm, 1e-9))))
                keep = (
                    (ijk[:, 0] >= skin) & (ijk[:, 0] < ix - skin)
                    & (ijk[:, 1] >= skin) & (ijk[:, 1] < iy - skin)
                    & (ijk[:, 2] >= skin) & (ijk[:, 2] < iz - skin)
                )
                shapes = shapes.loc[keep].reset_index(drop=True)
            summary["lof_shape"] = summarize_shapes(shapes, voxel_m)
            if outdir:
                shapes_path = os.path.join(outdir, "lof_pore_shapes.csv")
                shapes.to_csv(shapes_path, index=False)
                summary["lof_shape"]["per_pore_csv"] = shapes_path
            ls = summary["lof_shape"]
            if ls.get("n_pores"):
                _log(
                    f"LoF pore sizes: n={ls['n_pores']}  ESD mean={ls['esd_mean_um']:.1f} um  "
                    f"median={ls['esd_median_um']:.1f}  D10/D90={ls['esd_d10_um']:.1f}/"
                    f"{ls['esd_d90_um']:.1f} um  aspect ratio median={ls['aspect_ratio_median']:.2f}"
                    + (
                        f"  sphericity median={ls['sphericity_median']:.2f} "
                        f"(pores >= {ls['shape_min_voxels']} voxels, n={ls['n_shape_pores']})"
                        if "sphericity_median" in ls
                        else ""
                    )
                )
        except Exception as exc:
            _log(f"LoF pore shape stats skipped: {exc}")
    if outdir:
        try:
            import json

            with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass
    _log(
        f"Raptor LoF phi={phi*100:.4g}%  mean ESD={summary['mean_esd_um']:.1f} um  "
        f"rho~={rho:.6g} /mm3  (n_morph={len(esd_um)})"
    )
    return {
        "rho": rho,
        "porosity_fraction": phi,
        "porosity_pct": phi * 100.0,
        "size_catalog": size_catalog,
        "esd_um": esd_um,
        "summary": summary,
    }
