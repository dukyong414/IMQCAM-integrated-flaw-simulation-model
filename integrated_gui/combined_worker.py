"""Subprocess worker for Combined plate-scale pore generation (NU + CMU density)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
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
    # Vendored ORNL Raptor (Combined cuboid LoF RVE)
    raptor_src = os.path.join(root, "Raptor", "src")
    if os.path.isdir(raptor_src) and raptor_src not in sys.path:
        sys.path.insert(0, raptor_src)

    # Ignore console Ctrl+C from PyVista / parent console (Windows)
    try:
        from integrated_gui.console_guard import install_windows_ctrl_guard

        install_windows_ctrl_guard(detach_console=True)
    except Exception:
        pass

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    status_path = os.path.join(outdir, "status.json")
    log_path = os.path.join(outdir, "worker.log")
    progress_path = os.path.join(outdir, "progress.json")
    t0 = time.time()

    def write_status(ok: bool, message: str, **extra):
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"ok": ok, "message": message, **extra}, f, indent=2)

    logs = []

    def log(msg: str):
        logs.append(str(msg))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    def progress(stage: str, frac: float, message: str, *, step: int = 0, n_steps: int = 6):
        """Write live progress for the GUI (fraction 0–1 + ETA)."""
        frac = float(max(0.0, min(1.0, frac)))
        elapsed = time.time() - t0
        eta_s = None
        if frac >= 0.02:
            eta_s = max(0.0, elapsed / frac - elapsed)
        payload = {
            "stage": stage,
            "frac": frac,
            "pct": round(100.0 * frac, 1),
            "message": message,
            "step": int(step),
            "n_steps": int(n_steps),
            "elapsed_s": round(elapsed, 1),
            "eta_s": None if eta_s is None else round(eta_s, 1),
            "t0": t0,
        }
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    try:
        cfg_path = os.path.abspath(args.config)
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)

        progress("start", 0.02, "Starting Combined worker…", step=0, n_steps=6)
        stage_s: dict = {}

        from integrated_gui.combined_3d import (
            build_plate_pore_cloud,
            make_plate_layout,
            nu_field_at_xy_batch,
            porosity_summary,
        )
        from integrated_gui.cmu_labeled import run_cmu_labeled_samples
        from integrated_gui.compute_backend import apply_env, resolve_device
        from integrated_gui.nu_cache import cache_exists, load_fitted_model, MODEL_PATH

        CMU_DIR = os.path.join(root, "CMU team")
        NU_DIR = os.path.join(root, "NU team")

        compute_info = resolve_device(cfg.get("compute_device", "auto"))
        apply_env(compute_info)
        prefer_gpu = compute_info.device == "gpu"
        log(f"Compute: {compute_info.detail}")

        raw_cap = cfg.get("max_pores", cfg.get("totalpores", cfg.get("total_pores", 0)))
        max_pores = int(raw_cap) if raw_cap is not None else 0
        if max_pores <= 0:
            max_pores = None

        seed = int(cfg.get("seed", 42))
        plate_mode = str(cfg.get("plate_mode", "grid_cubes"))
        mean_p = float(cfg.get("mean_porosity", 0.02))
        enable_scan = bool(cfg.get("enable_scan_vectors", False))
        use_raptor_lof = bool(cfg.get("use_raptor_lof", False)) and plate_mode in (
            "single_cuboid",
            "stl",
        )
        gas_flow = str(cfg.get("gas_flow", "+x") or "+x").lower().strip()
        if gas_flow not in ("+x", "-x", "+y", "-y"):
            gas_flow = "+x"
        n_steps = 6
        if use_raptor_lof:
            n_steps += 1
        if enable_scan:
            n_steps += 1

        progress("geometry", 0.08, "Building geometry (cubes / cuboid / STL)…", step=1, n_steps=n_steps)
        cuboid_L = cfg.get("cuboid_length_mm")
        cuboid_W = cfg.get("cuboid_width_mm")
        stl_max_h = cfg.get("stl_max_height_mm")
        layout = make_plate_layout(
            plate_width_mm=float(cfg.get("plate_w", 250)),
            plate_height_mm=float(cfg.get("plate_h", 250)),
            n_rows=int(cfg.get("n_rows", 19)),
            n_cols=int(cfg.get("n_cols", 3)),
            cube_size_mm=float(cfg.get("cube_size", 10.0)),
            cube_height_mm=float(cfg.get("cube_height", cfg.get("cube_size", 10.0))),
            gap_mm=cfg.get("gap_mm", None),
            plate_mode=plate_mode,
            cuboid_margin_mm=float(cfg.get("cuboid_margin_mm", 5.0)),
            cuboid_fill_plate=bool(cfg.get("cuboid_fill_plate", False)),
            cuboid_length_mm=float(cuboid_L) if cuboid_L not in (None, "") else None,
            cuboid_width_mm=float(cuboid_W) if cuboid_W not in (None, "") else None,
            stl_path=cfg.get("stl_path"),
            stl_margin_mm=float(cfg.get("stl_margin_mm", 2.0)),
            stl_max_height_mm=float(stl_max_h) if stl_max_h not in (None, "", 0, 0.0) else None,
            n_strips=int(cfg.get("n_strips", 4)),
            strip_gap_mm=cfg.get("strip_gap_mm", None),
            stl_sector_fraction=float(cfg.get("stl_sector_fraction", 1.0) or 1.0),
            stl_sector_start_deg=float(cfg.get("stl_sector_start_deg", 0.0) or 0.0),
        )
        if layout.plate_mode == "grid_cubes":
            log(
                f"Coupon plate: {layout.n_strips} strips × "
                f"{layout.n_rows}×{layout.n_cols} cubes = {layout.n_builds} builds"
            )
        log(
            f"Plate mode={layout.plate_mode}  "
            f"{layout.plate_width_mm:.0f}x{layout.plate_height_mm:.0f} mm  "
            f"builds={layout.n_builds}  height={layout.cube_height_mm:.2f} mm  "
            f"volume={layout.total_build_volume_mm3:.3g} mm³"
        )
        if layout.plate_mode == "stl" and layout.stl_meta:
            m = layout.stl_meta
            clip_txt = (
                f"  clipped_height={m.get('clip_height_mm'):.2f} mm"
                if m.get("height_clipped")
                else ""
            )
            log(
                f"STL fit: scale={m.get('scale'):.4g}  "
                f"z={m.get('zmin'):.2f}..{m.get('zmax'):.2f} mm{clip_txt}"
            )
            if (m.get("sector_fraction") or 1.0) < 1.0:
                full_v = m.get("full_part_volume_mm3")
                log(
                    f"STL cake slice: 1/{round(1.0 / m['sector_fraction'])} of the part "
                    f"(start {m.get('sector_start_deg', 0):.0f} deg)  "
                    f"volume={m.get('volume_mm3', 0):.4g} mm³"
                    + (f" of {full_v:.4g} mm³" if full_v else "")
                )
        if layout.plate_mode == "single_cuboid" and layout.builds:
            b0 = layout.builds[0]
            log(
                f"Cuboid L×W×H = {b0.length_mm:.2f}×{b0.width_mm:.2f}×{b0.height_mm:.2f} mm "
                f"(shape={b0.shape})"
            )

        progress("nu_cache", 0.15, "Loading NU spatial model…", step=2, n_steps=n_steps)
        if NU_DIR not in sys.path:
            sys.path.insert(0, NU_DIR)
        if not cache_exists():
            raise RuntimeError(
                f"NU cache required for Combined placement but missing: {MODEL_PATH}. "
                "Use “Pre-fit NU model” in the Combined tab first."
            )
        nu_model = load_fitted_model()
        log(f"Loaded NU cache: {MODEL_PATH}")

        param_dict = cfg["param_dict"]
        props = cfg["props"]
        models = cfg.get("pore_models") or ["lof", "keyhole", "gas"]

        progress("cmu", 0.28, "CMU density + size catalog…", step=3, n_steps=n_steps)
        catalog_n = 5000
        prev = os.getcwd()
        os.chdir(CMU_DIR)
        if CMU_DIR not in sys.path:
            sys.path.insert(0, CMU_DIR)
        log(f"Drawing CMU size catalog (N={catalog_n})…")
        catalog, _, cmu_meta = run_cmu_labeled_samples(
            param_dict=param_dict,
            props=props,
            pore_model_names=models,
            totalpores=catalog_n,
        )
        dens = cmu_meta["densities"]
        weights = {k: float(v) for k, v in dens.items() if k in models}
        if sum(weights.values()) <= 0:
            weights = {m: 1.0 for m in models}
        rho = float(sum(weights.values()))
        log(f"CMU densities (pores/mm³): {dens}  total ρ={rho:.6g}")
        est = rho * float(layout.total_build_volume_mm3)
        log(f"Pore count from ρ·V ≈ {est:.1f} (NU relative factors applied next; no fixed cap)")
        if max_pores is not None:
            log(f"Max pore cap = {max_pores}")
        else:
            log("No max pore cap — using full ρ·V·(NU/mean_NU)")
        if layout.plate_mode == "stl" and layout.stl_meta:
            log(
                f"STL volume={layout.stl_meta.get('volume_mm3'):.3g} mm³ "
                f"(AABB={layout.stl_meta.get('aabb_volume_mm3'):.3g} mm³)"
            )
        os.chdir(prev)

        raptor_lof = None
        step_i = 4
        if use_raptor_lof:
            roi = cfg.get("raptor_roi") if isinstance(cfg.get("raptor_roi"), dict) else None
            roi_ok = bool(roi and roi.get("ok") and plate_mode == "stl")
            progress(
                "raptor",
                0.38,
                (
                    "Running Raptor LoF on selected STL region…"
                    if roi_ok
                    else "Running Raptor LoF on part AABB…"
                ),
                step=step_i,
                n_steps=n_steps,
            )
            step_i += 1
            from integrated_gui.combined_raptor import run_raptor_rve

            vel_mm_s = float(param_dict.get("Velocity (mm/s)", 1300))
            # Raptor standalone / CMU process params (micrometers) — NOT plate-scan mm
            hatch_um = float(param_dict.get("Hatch Spacing (um)", 140))
            layer_um = float(param_dict.get("Layer Thickness (um)", 30))
            hatch_m = hatch_um * 1e-6
            layer_m = layer_um * 1e-6
            # Domain: selected STL ROI > STL height band (real cross-section) > cuboid / AABB
            geom_label = "cuboid"
            stl_band = None
            if not roi_ok and plate_mode == "stl" and getattr(layout, "stl_mesh", None) is not None:
                from integrated_gui.stl_raptor_band import prepare_stl_band, prepare_stl_volume

                band_z = cfg.get("raptor_band_z_mm")
                try:
                    band_z = float(band_z) if band_z not in (None, "") else None
                except (TypeError, ValueError):
                    band_z = None
                sector_frac = float((layout.stl_meta or {}).get("sector_fraction") or 1.0)
                # Cake slices default to full-height 3D volume; thin band remains available
                domain_mode = str(cfg.get("raptor_stl_domain") or "").strip().lower()
                if domain_mode in ("", "auto"):
                    domain_mode = "volume" if 0.0 < sector_frac < 0.999 else "band"
                band_mode = str(cfg.get("raptor_band_z_mode") or "fullest")
                try:
                    if domain_mode == "volume":
                        # Same as cube / 14 mm pillar: every process layer at CMU hatch & layer
                        import math as _math

                        m = layout.stl_meta or {}
                        Lz_est = float(m.get("zmax", 1) - m.get("zmin", 0))
                        n_total = max(
                            1,
                            int(_math.ceil(Lz_est / max(layer_um * 1e-3, 1e-9))),
                        )
                        # Optional cap only when explicitly set; default = whole STL height
                        n_hatch = int(cfg.get("raptor_volume_hatch_layers") or n_total)
                        n_hatch = max(1, min(n_hatch, n_total))
                        stl_band = prepare_stl_volume(
                            layout.stl_mesh,
                            hatch_mm=hatch_um * 1e-3,
                            layer_mm=layer_um * 1e-3,
                            max_hatch_layers=n_hatch,
                            strategy=str(cfg.get("scan_strategy", "stripes")),
                            stripe_width_mm=float(cfg.get("stripe_width_mm", 10.0) or 10.0),
                            layer_rotation_deg=float(cfg.get("layer_rotation_deg", 67.0)),
                            base_angle_deg=cfg.get("base_angle_deg", None),
                        )
                    else:
                        stl_band = prepare_stl_band(
                            layout.stl_mesh,
                            hatch_mm=hatch_um * 1e-3,
                            layer_mm=layer_um * 1e-3,
                            n_layers=int(cfg.get("raptor_band_layers", 15) or 15),
                            band_z_mm=band_z,
                            strategy=str(cfg.get("scan_strategy", "stripes")),
                            stripe_width_mm=float(cfg.get("stripe_width_mm", 10.0) or 10.0),
                            layer_rotation_deg=float(cfg.get("layer_rotation_deg", 67.0)),
                            base_angle_deg=cfg.get("base_angle_deg", None),
                            xy_from_mesh=True,
                            band_z_mode=band_mode,
                        )
                except Exception as exc:
                    log(f"STL domain for Raptor failed ({exc}) — using a 1×1×1 mm cube instead.")
                    stl_band = None
            if stl_band is not None:
                size_xyz_mm = list(stl_band["size_mm"])
                part_cx = stl_band["origin_mm"][0] + 0.5 * size_xyz_mm[0]
                part_cy = stl_band["origin_mm"][1] + 0.5 * size_xyz_mm[1]
                geom_label = "STL volume" if stl_band.get("full_height") else "STL band"
                z0b = stl_band["origin_mm"][2]
                sector_frac = float((layout.stl_meta or {}).get("sector_fraction") or 1.0)
                cake_txt = (
                    f"  cake slice 1/{round(1.0 / sector_frac)}"
                    if 0.0 < sector_frac < 0.999
                    else ""
                )
                place = stl_band.get("band_placement") or "user"
                if stl_band.get("full_height"):
                    log(
                        f"Raptor on the full-height STL{cake_txt}: "
                        f"{size_xyz_mm[0]:.2f}×{size_xyz_mm[1]:.2f}×{size_xyz_mm[2]:.2f} mm  "
                        f"z={z0b:.2f}–{z0b + size_xyz_mm[2]:.2f} mm  "
                        f"process layers={stl_band['n_layers']}/{stl_band['n_layers_part']} "
                        f"(every CMU/Raptor layer, same as cube / pillar whole-build)"
                    )
                else:
                    log(
                        f"Raptor on a thin STL cross-section{cake_txt}: layers "
                        f"{stl_band['first_layer']}–"
                        f"{stl_band['first_layer'] + stl_band['n_layers'] - 1} of "
                        f"{stl_band['n_layers_part']}  (z = {z0b:.2f}–"
                        f"{z0b + stl_band['n_layers'] * layer_um * 1e-3:.2f} mm, "
                        f"placement={place}, XY=full mesh footprint "
                        f"{size_xyz_mm[0]:.2f}×{size_xyz_mm[1]:.2f} mm)"
                    )
            elif roi_ok:
                edge = float(roi.get("edge_mm") or 0.0)
                if edge <= 0 and roi.get("fixed_volume_mm3"):
                    edge = 1.0
                if edge > 0:
                    # Fixed cube (e.g. 1×1×1) — ignore any accidental resize in JSON
                    size_xyz_mm = [edge, edge, edge]
                else:
                    size_xyz_mm = [
                        float(roi.get("length_mm") or 0.0),
                        float(roi.get("width_mm") or 0.0),
                        float(roi.get("height_mm") or 0.0),
                    ]
                part_cx = float(roi.get("cx_mm", layout.plate_width_mm * 0.5))
                part_cy = float(roi.get("cy_mm", layout.plate_height_mm * 0.5))
                geom_label = "STL ROI"
            elif plate_mode == "stl":
                # Default Raptor LoF volume on STL: 1×1×1 mm³ at part center
                size_xyz_mm = [1.0, 1.0, 1.0]
                b0 = layout.builds[0] if layout.builds else None
                if b0 is not None:
                    part_cx = float(b0.cx_mm)
                    part_cy = float(b0.cy_mm)
                elif layout.stl_meta:
                    m = layout.stl_meta
                    part_cx = 0.5 * float(m.get("xmin", 0) + m.get("xmax", 0))
                    part_cy = 0.5 * float(m.get("ymin", 0) + m.get("ymax", 0))
                else:
                    part_cx = float(layout.plate_width_mm) * 0.5
                    part_cy = float(layout.plate_height_mm) * 0.5
                geom_label = "STL 1x1x1 mm"
                log("No STL Raptor region selected — using 1×1×1 mm³ at part center.")
            else:
                b0 = layout.builds[0] if layout.builds else None
                if b0 is not None:
                    size_xyz_mm = [
                        float(getattr(b0, "length_mm", None) or layout.cube_size_mm),
                        float(getattr(b0, "width_mm", None) or layout.cube_size_mm),
                        float(getattr(b0, "height_mm", None) or layout.cube_height_mm),
                    ]
                    part_cx = float(b0.cx_mm)
                    part_cy = float(b0.cy_mm)
                    geom_label = "cuboid"
                else:
                    size_xyz_mm = [
                        float(layout.cube_size_mm),
                        float(layout.cube_size_mm),
                        float(layout.cube_height_mm),
                    ]
                    part_cx = float(layout.plate_width_mm) * 0.5
                    part_cy = float(layout.plate_height_mm) * 0.5
            # Guard degenerate domain (ROI can be sub-mm; keep small floor)
            size_xyz_mm = [max(0.2, float(v)) for v in size_xyz_mm]
            # Where the Raptor box sits on the plate (its local grid starts at 0,0,0)
            if stl_band is not None:
                plate_origin_mm = [float(v) for v in stl_band["origin_mm"]]
            else:
                z_org = 0.0
                if roi_ok:
                    if roi.get("zmin_mm") is not None:
                        z_org = float(roi["zmin_mm"])
                    elif roi.get("cz_mm") is not None:
                        z_org = float(roi["cz_mm"]) - 0.5 * size_xyz_mm[2]
                plate_origin_mm = [
                    part_cx - 0.5 * size_xyz_mm[0],
                    part_cy - 0.5 * size_xyz_mm[1],
                    z_org,
                ]
            # Fine-sample / cubic fallback edge: 1 mm for STL, else hatch-based
            if plate_mode == "stl":
                default_edge = float(cfg.get("raptor_rve_edge_m", 1.0e-3) or 1.0e-3)
            else:
                default_edge = max(5.0e-4, 4.0 * hatch_m)
                # Cap process layers for tall parts (true layer_um spacing, not 5 mm plate layers)
            import math

            n_proc_layers = max(
                1, int(math.ceil(size_xyz_mm[2] / max(layer_um * 1e-3, 1e-9)))
            )
            raptor_max_layers = min(80, max(24, n_proc_layers))
            # Whole build: every process layer — cuboid OR full-height STL (same as cube / pillar)
            stl_full_volume = bool(stl_band is not None and stl_band.get("full_height"))
            whole_build = bool(cfg.get("scan_full_part", False)) and (
                plate_mode == "single_cuboid" or stl_full_volume
            )
            raptor_voxel = cfg.get("raptor_voxel_m")
            raptor_max_vox = int(cfg.get("raptor_max_voxels", 1_200_000) or 1_200_000)
            if whole_build:
                raptor_max_layers = (
                    int(stl_band["n_layers"]) if stl_full_volume else n_proc_layers
                )
                raptor_max_vox = int(cfg.get("raptor_whole_build_max_voxels", 400_000_000))
                vol_m3 = size_xyz_mm[0] * size_xyz_mm[1] * size_xyz_mm[2] * 1e-9
                raptor_voxel = float(cfg.get("raptor_voxel_m") or 0.0) or max(
                    5.0e-6, (vol_m3 / raptor_max_vox) ** (1.0 / 3.0)
                )
            if stl_band is not None and not whole_build:
                raptor_max_layers = stl_band["n_layers"]
                # Thin LoF band: fine voxels for hatch-accurate porosity
                raptor_max_vox = int(cfg.get("raptor_band_max_voxels", 800_000_000))
                vol_m3 = size_xyz_mm[0] * size_xyz_mm[1] * size_xyz_mm[2] * 1e-9
                raptor_voxel = float(cfg.get("raptor_voxel_m") or 0.0) or max(
                    10.0e-6, (vol_m3 / raptor_max_vox) ** (1.0 / 3.0)
                )
                if raptor_voxel > 20.0e-6:
                    log(
                        f"Warning: band voxel {raptor_voxel*1e6:.1f} um is coarse for LoF "
                        f"(hatch {hatch_um:.0f} um); use fewer band layers or a smaller slice."
                    )
            elif stl_full_volume and whole_build:
                log(
                    f"Full-height STL Raptor (whole build): {raptor_max_layers} process layers, "
                    f"voxel ≈ {float(raptor_voxel) * 1e6:.1f} µm  "
                    f"(CMU hatch={hatch_um:.0f} µm, layer={layer_um:.0f} µm)"
                )
            raptor_cfg = {
                "power_w": float(param_dict.get("Power (W)", 370)),
                "velocity_m_s": vel_mm_s / 1000.0,
                "hatch_m": hatch_m,
                "layer_m": layer_m,
                "hatch_um": hatch_um,
                "layer_um": layer_um,
                "rotation_deg": float(cfg.get("layer_rotation_deg", 67.0)),
                "size_xyz_mm": size_xyz_mm,
                "rve_edge_m": float(cfg.get("raptor_rve_edge_m", default_edge) or default_edge),
                "voxel_resolution_m": raptor_voxel,
                "raptor_max_voxels": raptor_max_vox,
                "n_modes": int(cfg.get("raptor_n_modes", 50)),
                "random_seed": seed,
                "compute_device": cfg.get("compute_device", "auto"),
                "enable_random_phases": True,
                "scan_strategy": str(cfg.get("scan_strategy", "horizontal")),
                "stripe_width_mm": float(cfg.get("stripe_width_mm", 10.0) or 10.0),
                "skip_spatter_anim": bool(cfg.get("scan_full_part", False))
                or stl_band is not None
                or not cfg.get("make_spatter_anim", True),
                "max_scan_layers": int(raptor_max_layers),
                "base_angle_deg": cfg.get("base_angle_deg", None),
                "batch_rows": int(cfg.get("batch_rows", 4) or 4),
                "batch_cols": int(cfg.get("batch_cols", 4) or 4),
                "gas_flow": gas_flow,
                "plate_w_mm": float(layout.plate_width_mm),
                "plate_h_mm": float(layout.plate_height_mm),
                "cuboid_cx_mm": part_cx,
                "cuboid_cy_mm": part_cy,
                # Fine hatch-scale sample only for thin bands / small RVEs — not whole-build
                "fine_lof_for_density": not whole_build and stl_band is None,
                "exclude_face_layers": 2 if (whole_build or stl_band is not None) else 0,
                "geometry_mode": plate_mode,
                "raptor_roi": roi if roi_ok else None,
                "_stl_band": stl_band,
                "plate_origin_mm": plate_origin_mm,
                "band_skin_mm": float(cfg.get("raptor_band_skin_mm", 0.1) or 0.0),
                "library": cfg.get("raptor_library") or None,
                "width_scale": float(cfg.get("width_scale", 1.0)),
            }
            rap_outdir = os.path.join(outdir, "raptor_rve")
            if roi_ok:
                try:
                    with open(os.path.join(outdir, "raptor_roi.json"), "w", encoding="utf-8") as _rf:
                        json.dump(roi, _rf, indent=2)
                except Exception:
                    pass
            try:
                log(
                    f"Raptor on {geom_label} {size_xyz_mm[0]:.2f}x{size_xyz_mm[1]:.2f}x"
                    f"{size_xyz_mm[2]:.2f} mm  @ ({part_cx:.1f},{part_cy:.1f})  "
                    f"strategy={raptor_cfg['scan_strategy']}  "
                    f"process hatch={hatch_um:.1f} um  layer={layer_um:.1f} um  "
                    f"melt pool width={float(cfg.get('meltpool_width_um') or 0):.0f} um  "
                    f"Raptor width={float(cfg.get('meltpool_width_used_um') or 0):.0f} um "
                    f"(plate-scan mm ignored for Raptor paths)"
                )
                if whole_build:
                    log(
                        f"Raptor whole build: all {n_proc_layers} layers, "
                        f"target voxel ~{raptor_voxel * 1e6:.1f} um"
                    )
                    progress(
                        "raptor",
                        0.40,
                        f"Raptor on the whole build ({n_proc_layers} layers)…",
                        step=step_i - 1,
                        n_steps=n_steps,
                    )
                _t_rap = time.time()
                raptor_lof = run_raptor_rve(raptor_cfg, outdir=rap_outdir, log=log)
                stage_s["raptor"] = time.time() - _t_rap
                log(
                    f"Raptor LoF phi={raptor_lof['porosity_pct']:.4g}% -> "
                    f"rho={raptor_lof['rho']:.6g}/mm3 (replaces CMU LoF for this part)"
                )
                # Estimate LoF count after merge
                est_lof = float(raptor_lof["rho"]) * float(layout.total_build_volume_mm3)
                log(f"Raptor LoF count from ρ·V ≈ {est_lof:.1f}")
            except Exception as exc:
                log(f"Raptor LoF failed — falling back to CMU LoF: {exc}")
                raptor_lof = None

        progress(
            "place",
            0.52 if use_raptor_lof else 0.45,
            f"Allocating & dispersing pores…",
            step=step_i,
            n_steps=n_steps,
        )
        step_i += 1
        # Realistic scan: process hatch/layer (µm) from the CMU/Raptor parameter set
        use_proc_scan = bool(cfg.get("use_process_scan_params", False))
        if use_proc_scan:
            scan_hatch_mm = float(param_dict.get("Hatch Spacing (um)", 140)) * 1e-3
            scan_layer_mm = float(param_dict.get("Layer Thickness (um)", 30)) * 1e-3
        else:
            scan_hatch_mm = float(cfg.get("hatch_spacing_mm", 0.14))
            scan_layer_mm = float(cfg.get("layer_thickness_mm", 0.03))
        scan_full_part = bool(cfg.get("scan_full_part", False)) and plate_mode in (
            "stl",
            "single_cuboid",
        )
        scan_vel = float(param_dict.get("Velocity (mm/s)", 1300) or 1300)
        place_frac = 0.52 if use_raptor_lof else 0.45
        _slice_t0 = [time.time()]

        def _slice_progress(done: int, total: int):
            el = time.time() - _slice_t0[0]
            progress(
                "scan",
                place_frac + (0.78 - place_frac) * done / max(total, 1),
                f"Slicing layer {done:,}/{total:,} ({el:.0f} s)…",
                step=step_i,
                n_steps=n_steps,
            )

        if enable_scan:
            log(
                f"Scan: strategy={cfg.get('scan_strategy')}  hatch={scan_hatch_mm * 1e3:.0f} um  "
                f"layer={scan_layer_mm * 1e3:.0f} um  "
                f"({'process params' if use_proc_scan else 'manual mm'})  "
                f"{'whole part, every layer' if scan_full_part else 'limited layers'}"
            )
        _t_place = time.time()
        combined, meta = build_plate_pore_cloud(
            cmu_size_catalog=catalog,
            cmu_density_weights=weights,
            layout=layout,
            nu_model=nu_model,
            mean_porosity=mean_p,
            seed=seed,
            max_pores=max_pores,
            enable_scan_vectors=enable_scan,
            scan_strategy=str(cfg.get("scan_strategy", "horizontal")),
            hatch_spacing_mm=scan_hatch_mm,
            layer_thickness_mm=scan_layer_mm,
            max_scan_layers=cfg.get("max_scan_layers", 5),
            scan_full_part=scan_full_part,
            stripe_width_mm=float(cfg.get("stripe_width_mm", 10.0) or 10.0),
            scan_velocity_mm_s=scan_vel,
            progress_cb=_slice_progress if scan_full_part else None,
            n_edge_pores=int(cfg.get("n_edge_pores", 0)),
            batch_rows=int(cfg.get("batch_rows", 4)),
            batch_cols=int(cfg.get("batch_cols", 4)),
            base_angle_deg=cfg.get("base_angle_deg", None),
            layer_rotation_deg=float(cfg.get("layer_rotation_deg", 67.0)),
            prefer_gpu=prefer_gpu,
            raptor_lof=raptor_lof,
            gas_flow=gas_flow,
            gas_flow_sharpness=float(cfg.get("gas_flow_sharpness", 2.0) or 2.0),
            flow_perturbation=float(cfg.get("flow_perturbation", 0.65) or 0.65),
        )
        stage_s["scan_and_pores"] = time.time() - _t_place
        summary = porosity_summary(combined, layout)
        summary.update(
            {
                "counts_realized": meta["counts_realized"],
                "cmu_total_density": meta["cmu_total_density"],
                "nu_mean_pct": meta["nu_mean_pct"],
                "part_volume_mm3": meta["part_volume_mm3"],
                "base_count_rho_v": meta["base_count_rho_v"],
                "max_pores": max_pores,
                "layout": meta["layout"],
                "cmu_density_weights": meta["cmu_density_weights"],
                "total_pores": meta["total_pores"],
                "n_bulk_pores": meta["n_bulk_pores"],
                "n_edge_pores": meta["n_edge_pores"],
                "n_spatter_pores": 0,
                "spatter_fraction": 0.0,
                "gas_flow": meta.get("gas_flow") or gas_flow,
                "scan_strategy": meta.get("scan_strategy"),
                "enable_scan_vectors": meta.get("enable_scan_vectors"),
                "compute_device": compute_info.device,
                "place_backend": meta.get("place_backend"),
                "raptor_lof": meta.get("raptor_lof") or {},
                "use_raptor_lof": bool(raptor_lof),
                "raptor_roi": (
                    cfg.get("raptor_roi")
                    if isinstance(cfg.get("raptor_roi"), dict) and cfg["raptor_roi"].get("ok")
                    else None
                ),
                "scan_stats": meta.get("scan_stats") or {},
            }
        )
        if meta.get("scan_stats"):
            from integrated_gui.stl_slicer_scan import format_scan_stats

            log(format_scan_stats(meta["scan_stats"]))
        if meta.get("scan_mesh") is not None and not cfg.get("make_spatter_anim", True):
            log("Spatter animation skipped (unticked in the GUI).")
        elif meta.get("scan_mesh") is not None:
            progress("scan", 0.80, "Whole-part spatter animation…", step=step_i, n_steps=n_steps)
            _t_anim = time.time()
            try:
                from integrated_gui.combined_spatter_anim import build_spatter_animation
                from integrated_gui.stl_slicer_scan import scan_keyframes

                kf = scan_keyframes(
                    meta["scan_mesh"],
                    hatch_mm=scan_hatch_mm,
                    layer_mm=scan_layer_mm,
                    strategy=str(cfg.get("scan_strategy", "stripes")),
                    stripe_width_mm=float(cfg.get("stripe_width_mm", 10.0) or 10.0),
                    layer_rotation_deg=float(cfg.get("layer_rotation_deg", 67.0)),
                    base_angle_deg=cfg.get("base_angle_deg", None),
                    batch_rows=int(cfg.get("batch_rows", 4) or 4),
                    batch_cols=int(cfg.get("batch_cols", 4) or 4),
                    n_layers_anim=int(cfg.get("anim_layers", 12) or 12),
                    vectors_per_layer=int(cfg.get("anim_vectors_per_layer", 6) or 6),
                )
                bx0, bx1, by0, by1, bz0, bz1 = kf["bounds_mm"]
                spat = build_spatter_animation(
                    outdir=os.path.join(outdir, "scan_anim"),
                    length_mm=bx1 - bx0,
                    width_mm=by1 - by0,
                    height_mm=bz1 - bz0,
                    plate_w_mm=float(layout.plate_width_mm),
                    plate_h_mm=float(layout.plate_height_mm),
                    cuboid_cx=0.5 * (bx0 + bx1),
                    cuboid_cy=0.5 * (by0 + by1),
                    strategy=str(cfg.get("scan_strategy", "stripes")),
                    gas_flow=gas_flow,
                    seed=seed,
                    wake_len_mm=float(cfg.get("spatter_wake_len_mm", 300.0) or 300.0),
                    segments=kf["segments"],
                    outlines=kf["outlines"],
                    n_layers_part=kf["n_layers_part"],
                )
                if spat.get("ok"):
                    log(
                        f"Whole-part spatter animation: {len(kf['segments'])} frames over "
                        f"{len(kf['outlines'])} layers (1..{kf['n_layers_part']}) -> {spat.get('gif')}"
                    )
                else:
                    log(f"Whole-part spatter animation skipped: {spat.get('error')}")
            except Exception as exc:
                log(f"Whole-part spatter animation skipped: {exc}")
            stage_s["spatter_animation"] = time.time() - _t_anim
        log(f"Placement backend: {meta.get('place_backend', 'NumPy CPU')}")
        log(
            f"Allocated bulk pores={meta['n_bulk_pores']}  "
            f"(ρ·V base≈{meta['base_count_rho_v']:.1f}, NU mean={meta['nu_mean_pct']:.4g}%)"
        )
        if plate_mode == "single_cuboid":
            log(f"Gas distribution direction={gas_flow} (biases cuboid pore cloud; no separate spatter)")
        log(f"Realized counts: {meta['counts_realized']}")
        if enable_scan:
            progress("scan", 0.72, "Scan vectors + edge pores…", step=step_i, n_steps=n_steps)
            log(
                f"Scan strategy={cfg.get('scan_strategy')}  "
                f"edge pores={meta['n_edge_pores']}  "
                f"vectors={len(meta.get('vectors_df', []))}"
            )

        _t_export = time.time()
        progress(
            "export",
            0.82,
            f"Writing CSV ({len(combined):,} pores) / field maps…",
            step=n_steps - 1,
            n_steps=n_steps,
        )
        pores_path = os.path.join(outdir, "pores.csv")
        # Chunked write keeps memory flatter and lets progress update on huge plates
        try:
            combined.to_csv(pores_path, index=False, float_format="%.6g")
        except Exception:
            # Fallback without float_format if pandas/engine complains
            combined.to_csv(pores_path, index=False)
        rap_s = (raptor_lof or {}).get("summary") or {}
        if rap_s.get("plate_origin_mm") and rap_s.get("size_xyz_mm") and len(combined):
            try:
                from integrated_gui.raptor_view import pores_in_raptor_domain

                sub, (z_lo, z_hi) = pores_in_raptor_domain(
                    combined, rap_s["plate_origin_mm"], rap_s["size_xyz_mm"]
                )
                sub_path = os.path.join(outdir, "raptor_rve", "combined_pores_in_domain.csv")
                sub.to_csv(sub_path, index=False, float_format="%.6g")
                with open(sub_path.replace(".csv", ".json"), "w", encoding="utf-8") as _zf:
                    json.dump({"z_lo_mm": z_lo, "z_hi_mm": z_hi}, _zf)
                counts = sub["pore_source"].astype(str).value_counts().to_dict() if len(sub) else {}
                log(
                    f"Combined-model pores inside the Raptor box (z {z_lo:.2f}–{z_hi:.2f} mm): "
                    f"{len(sub):,} {counts}"
                )
            except Exception as exc:
                log(f"Raptor-box pore subset skipped: {exc}")
        progress("export", 0.90, "Writing geometry + NU field maps…", step=n_steps - 1, n_steps=n_steps)
        meta["cube_df"].to_csv(os.path.join(outdir, "cubes.csv"), index=False, float_format="%.6g")
        if meta.get("vectors_df") is not None and len(meta["vectors_df"]):
            meta["vectors_df"].to_csv(
                os.path.join(outdir, "scan_vectors.csv"), index=False, float_format="%.6g"
            )
        if meta.get("scan_layers_df") is not None and len(meta["scan_layers_df"]):
            meta["scan_layers_df"].to_csv(
                os.path.join(outdir, "scan_layers.csv"), index=False, float_format="%.6g"
            )

        if layout.stl_mesh is not None:
            try:
                stl_out = os.path.join(outdir, "fitted.stl")
                layout.stl_mesh.save(stl_out)
                log(f"Saved fitted STL -> {stl_out}")
            except Exception as exc:
                log(f"Could not save fitted STL: {exc}")

        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(outdir, "geom.json"), "w", encoding="utf-8") as f:
            json.dump(meta["layout"], f, indent=2)

        import numpy as np

        cube_df = meta["cube_df"]
        field_path = os.path.join(outdir, "field.npz")
        pw = float(layout.plate_width_mm)
        ph = float(layout.plate_height_mm)
        nu_bar = float(meta["nu_mean_pct"]) or 1e-15

        # Full-plate NU colormap (background for 2D preview in all geometries)
        res_mm = float(cfg.get("field_resolution_mm", 4.0))
        xs = np.arange(0.0, layout.plate_width_mm + 1e-9, res_mm)
        ys = np.arange(0.0, layout.plate_height_mm + 1e-9, res_mm)
        X, Y = np.meshgrid(xs, ys)
        nu_grid = nu_field_at_xy_batch(
            nu_model, X.ravel(), Y.ravel(), pw, ph, mean_p
        ).reshape(X.shape)
        rel_grid = nu_grid / nu_bar

        # Optional footprint (STL AABB only) — coupon / cuboid keep full-plate background
        mask = np.ones_like(rel_grid, dtype=bool)
        if layout.plate_mode == "stl" and layout.stl_meta:
            m = layout.stl_meta
            mask = (
                (X >= float(m.get("xmin", 0)))
                & (X <= float(m.get("xmax", pw)))
                & (Y >= float(m.get("ymin", 0)))
                & (Y <= float(m.get("ymax", ph)))
            )

        field_kw = dict(
            mode="grid",
            grid_x=X,
            grid_y=Y,
            nu_pct_grid=nu_grid,
            nu_relative_grid=rel_grid,
            placement_weight_grid=rel_grid,
            flow_weight_grid=rel_grid,
            footprint_mask=mask,
            plate_mode=str(layout.plate_mode),
        )
        # Coupon cubes: also store build centers/sizes for outlines on the map
        if layout.plate_mode == "grid_cubes" and cube_df is not None and len(cube_df):
            field_kw.update(
                cx=cube_df["cx_mm"].to_numpy(),
                cy=cube_df["cy_mm"].to_numpy(),
                nu_pct=cube_df["nu_pct"].to_numpy(),
                nu_relative=cube_df["nu_relative"].to_numpy(),
                n_allocated=cube_df["n_allocated"].to_numpy(),
                placement_weight=cube_df["placement_weight"].to_numpy(),
                size=cube_df["size_mm"].to_numpy(),
                row=cube_df["row"].to_numpy(),
                col=cube_df["col"].to_numpy(),
            )
        elif layout.plate_mode == "single_cuboid" and len(layout.builds):
            b = layout.builds[0]
            field_kw.update(
                cx=np.array([b.cx_mm], dtype=float),
                cy=np.array([b.cy_mm], dtype=float),
                size=np.array([b.length_mm], dtype=float),
                length_mm=np.array([b.length_mm], dtype=float),
                width_mm=np.array([b.width_mm], dtype=float),
            )
        np.savez_compressed(field_path, **field_kw)

        # Process parameters used by this model (saved beside pores.csv)
        p_w = float(param_dict.get("Power (W)", 370) or 370)
        v_mm_s = float(param_dict.get("Velocity (mm/s)", 1300) or 1300)
        h_um = float(param_dict.get("Hatch Spacing (um)", 140) or 140)
        t_um = float(param_dict.get("Layer Thickness (um)", 30) or 30)
        ved = p_w / max(v_mm_s * (h_um * 1e-3) * (t_um * 1e-3), 1e-12)
        part_dims = None
        if plate_mode == "single_cuboid" and layout.builds:
            b0 = layout.builds[0]
            part_dims = [float(b0.length_mm), float(b0.width_mm), float(b0.height_mm)]
        elif plate_mode == "stl" and layout.stl_meta:
            m = layout.stl_meta
            part_dims = [
                float(m.get("xmax", 0) - m.get("xmin", 0)),
                float(m.get("ymax", 0) - m.get("ymin", 0)),
                float(m.get("zmax", 0) - m.get("zmin", 0)),
            ]
        rap_sum = summary.get("raptor_lof") or {}
        process = {
            "plate_mode": plate_mode,
            "part_size_mm": part_dims,
            "power_W": p_w,
            "scan_speed_mm_s": v_mm_s,
            "hatch_spacing_um": h_um,
            "layer_thickness_um": t_um,
            "volumetric_energy_density_J_mm3": ved,
            "scan_strategy": str(cfg.get("scan_strategy", "")),
            "stripe_width_mm": float(cfg.get("stripe_width_mm", 10.0) or 10.0),
            "layer_rotation_deg": float(cfg.get("layer_rotation_deg", 67.0)),
            "scan_used_process_params": bool(use_proc_scan),
            "scan_hatch_um": scan_hatch_mm * 1e3,
            "scan_layer_um": scan_layer_mm * 1e3,
            "raptor_used": bool(raptor_lof),
            "raptor_voxel_um": (float(rap_sum.get("voxel_resolution_m") or 0) * 1e6) or None,
            "raptor_layers_scanned": rap_sum.get("n_layers_scanned"),
        }
        stage_s["export"] = time.time() - _t_export
        timing = {k: round(float(v), 2) for k, v in stage_s.items()}
        timing["total"] = round(time.time() - t0, 2)
        summary["process"] = process
        summary["timing_s"] = timing
        with open(os.path.join(outdir, "process_parameters.json"), "w", encoding="utf-8") as f:
            json.dump(process, f, indent=2)
        with open(os.path.join(outdir, "process_parameters.csv"), "w", encoding="utf-8") as f:
            f.write("parameter,value\n")
            for k, v in process.items():
                f.write(f"{k},\"{v}\"\n")
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(
            f"Process: P={p_w:.0f} W  v={v_mm_s:.0f} mm/s  hatch={h_um:.0f} um  "
            f"layer={t_um:.0f} um  VED={ved:.1f} J/mm3"
        )
        log("Timing (s): " + "  ".join(f"{k}={v:.1f}" for k, v in timing.items()))

        progress("done", 1.0, "Worker finished", step=n_steps, n_steps=n_steps)
        write_status(True, "ok", summary=summary, log="\n".join(logs))
        log("Worker finished OK")
        print("WORKER_OK")
    except Exception:
        tb = traceback.format_exc()
        log(tb)
        try:
            progress("error", 1.0, "Worker failed", step=0, n_steps=1)
        except Exception:
            pass
        write_status(False, "error", error=tb, log="\n".join(logs))
        print("WORKER_FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
