"""Subprocess worker for ORNL Raptor LoF porosity RVE simulation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _write_phase_screenshot(vti_path: str, png_path: str) -> None:
    """Two-panel phase render: full melted build | every LoF pore."""
    from integrated_gui.raptor_view import write_phase_screenshot

    write_phase_screenshot(vti_path, png_path)


def _write_phase_screenshot_cutaway(vti_path: str, png_path: str) -> None:
    """Legacy single-panel cut-away render (kept for reference)."""
    import pyvista as pv
    from matplotlib.colors import ListedColormap
    from raptor.path_viz import add_scan_paths_to_plotter, sibling_path_vectors_csv

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv.OFF_SCREEN = True
    import numpy as np

    full = pv.read(vti_path)
    outline = full.outline()
    phase_full = np.asarray(full["Phase"])
    pore_idx = np.flatnonzero(phase_full == 0)
    has_pores = bool(len(pore_idx) > 0)
    # Whole-build volumes (10^8+ voxels) are too slow to clip at full resolution:
    # render a thinned volume and overlay every pore voxel as a point.
    max_render = 8_000_000
    pore_pts = None
    if full.n_points > max_render:
        s = int(np.ceil((full.n_points / max_render) ** (1.0 / 3.0)))
        ext = full.extent
        rve = full.extract_subset(ext, rate=(s, s, s))
        if has_pores:
            # Raptor stores z as the fastest index axis and rotates it into place with
            # the image direction matrix, so map index -> world through that matrix.
            d0, d1, _ = full.dimensions
            i = pore_idx % d0
            j = (pore_idx // d0) % d1
            k = pore_idx // (d0 * d1)
            ijk = np.column_stack([i, j, k]).astype(float) * np.asarray(full.spacing, dtype=float)
            try:
                dmat = np.asarray(full.direction_matrix, dtype=float).reshape(3, 3)
            except Exception:
                dmat = np.eye(3)
            pore_pts = np.asarray(full.origin, dtype=float) + ijk @ dmat.T
    else:
        rve = full
    del phase_full
    cmap = ListedColormap(
        [
            (1.0, 0.0, 0.0),
            (0.7, 0.7, 0.7),
            (0.2, 0.2, 0.2),
            (1.0, 1.0, 0.0),
        ],
        name="raptor_phase",
    )
    annotations = {0.5: "Pore", 1.5: "Melted", 2.5: "Boundary", 3.5: "Intersection"}
    # Clip near the leading face so most of the volume (grey melted bulk) stays visible
    b = rve.bounds
    origin = (b[0] + 0.18 * (b[1] - b[0]), rve.center[1], rve.center[2])
    clipped = rve.clip(normal=(1, 0, 0), origin=origin)
    pl = pv.Plotter(off_screen=True, window_size=(1100, 850))
    pl.set_background("white")
    pl.add_mesh(
        clipped,
        scalars="Phase",
        cmap=cmap,
        clim=(0, 4),
        categories=True,
        n_colors=4,
        annotations=annotations if has_pores else {1.5: "Melted", 2.5: "Boundary", 3.5: "Intersection"},
        interpolate_before_map=False,
        lighting=False,
        opacity=1.0,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Phase", "n_labels": 0},
    )
    if pore_pts is not None and len(pore_pts):
        pore_pts = pore_pts[pore_pts[:, 0] >= origin[0]]
        if len(pore_pts):
            pl.add_mesh(
                pv.PolyData(pore_pts),
                color=(1.0, 0.0, 0.0),
                point_size=3,
                render_points_as_spheres=True,
            )
    pl.add_mesh(outline, color="black", line_width=1)
    # Thin dark hatch lines (not bright cyan) so melted grey stays readable.
    # Whole-build runs have tens of thousands of vectors that would bury the volume.
    path_csv = sibling_path_vectors_csv(vti_path)
    n_paths = 0
    try:
        with open(path_csv, "r", encoding="utf-8") as _f:
            n_paths = sum(1 for _ in _f) - 1
    except Exception:
        pass
    if 0 < n_paths <= 5000:
        add_scan_paths_to_plotter(
            pl,
            path_csv,
            line_width=1.0,
            color="#1565C0",
            opacity=0.35,
        )
    pl.add_axes()
    pl.camera_position = "iso"
    pl.screenshot(png_path)
    pl.close()


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
    # Belt-and-suspenders: editable pip install may be missing; still find vendored Raptor
    raptor_src = os.path.join(root, "Raptor", "src")
    if os.path.isdir(raptor_src) and raptor_src not in sys.path:
        sys.path.insert(0, raptor_src)

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

    # Write progress immediately so the GUI leaves the "queued" state
    def _boot_progress():
        try:
            with open(os.path.join(outdir, "progress.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stage": "boot",
                        "frac": 0.02,
                        "pct": 2.0,
                        "message": "Worker process started…",
                        "step": 0,
                        "n_steps": 6,
                        "elapsed_s": 0.0,
                        "eta_s": None,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    _boot_progress()

    # Persistent Numba cache under project Raptor tree
    cache_dir = os.path.join(root, "Raptor", ".numba-cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", cache_dir)

    def write_status(ok: bool, message: str, **extra):
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"ok": ok, "message": message, **extra}, f, indent=2)

    logs = []

    def log(msg: str):
        logs.append(str(msg))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    def progress(stage: str, frac: float, message: str, *, step: int = 0, n_steps: int = 6):
        frac = float(max(0.0, min(1.0, frac)))
        elapsed = time.time() - t0
        eta_s = None if frac < 0.02 else max(0.0, elapsed / frac - elapsed)
        payload = {
            "stage": stage,
            "frac": frac,
            "pct": round(100.0 * frac, 1),
            "message": message,
            "step": int(step),
            "n_steps": int(n_steps),
            "elapsed_s": round(elapsed, 1),
            "eta_s": None if eta_s is None else round(eta_s, 1),
        }
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    try:
        with open(os.path.abspath(args.config), "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)

        progress("start", 0.03, "Loading Raptor…", step=1, n_steps=6)
        import shutil

        import numpy as np

        from integrated_gui.compute_backend import apply_env, resolve_device

        compute_info = resolve_device(cfg.get("compute_device", "auto"))
        apply_env(compute_info)
        log(f"Compute: {compute_info.detail}")

        from raptor.api import (
            create_grid,
            write_vtk,
            compute_morphology,
            write_morphology,
        )
        from raptor.utilities import ScanPathBuilder
        from integrated_gui import meltpool_library as mpl

        log("Raptor melt-mask kernel: Numba (ORNL Raptor)")

        edge_m = float(cfg.get("rve_edge_m", 5.0e-4))
        voxel_m = float(cfg.get("voxel_resolution_m", 5.0e-6))
        power = float(cfg.get("power_w", 370))
        hatch = float(cfg.get("hatch_m", 140e-6))
        layer = float(cfg.get("layer_m", 30e-6))
        rotation = float(cfg.get("rotation_deg", 67))
        n_modes = int(cfg.get("n_modes", 50))
        width_scale = float(cfg.get("width_scale", 1.0))
        seed = cfg.get("random_seed", 42)
        seed = None if seed is None or str(seed).strip() == "" else int(seed)
        enable_random = bool(cfg.get("enable_random_phases", True))
        height_shape = float(cfg.get("height_shape", 1))
        memory_limit_mb = cfg.get("memory_limit_mb")
        if memory_limit_mb is not None and str(memory_limit_mb).strip() != "":
            memory_limit_mb = float(memory_limit_mb)
        else:
            memory_limit_mb = None

        # ---- Resolve the list of run conditions (one per velocity) ----------
        lib_cfg = cfg.get("library") or None
        runs = []
        if lib_cfg:
            lib_dir = lib_cfg.get("dir") or mpl.DEFAULT_LIBRARY_DIR
            eta_source = lib_cfg.get("eta_source", "flucWs")
            library = mpl.load_library(lib_dir, eta_source)
            velocities = cfg.get("velocities_mm_s") or []
            if not velocities:
                raise ValueError("Melt-pool library enabled but no velocities selected.")
            for v in velocities:
                cond = library.get(mpl._vkey(v))
                if cond is None or not cond.width_files:
                    raise FileNotFoundError(f"No ULI_v{v:g}_theta*_widths.txt in {lib_dir}")
                use_eta = bool(lib_cfg.get("use_eta", True))
                use_shape = bool(lib_cfg.get("use_depth_shape", True))
                if use_eta and (cond.eta_d is None or cond.eta_c is None):
                    raise ValueError(f"V={v:g} mm/s missing from {eta_source}.txt")
                if use_shape and cond.depth_shape is None:
                    raise ValueError(f"V={v:g} mm/s missing from depthShapes.txt")
                runs.append(
                    {
                        "velocity_mm_s": float(v),
                        "depth_scale": cond.eta_d if use_eta else float(cfg.get("depth_scale", 0.8)),
                        "height_scale": cond.eta_c if use_eta else float(cfg.get("height_scale", 0.4)),
                        "depth_shape": cond.depth_shape if use_shape else float(cfg.get("depth_shape", 1)),
                        "width_files": cond.width_files,
                        "condition": cond.to_dict(),
                    }
                )
            theta_mode = lib_cfg.get("theta_mode", "averaged")
            theta_fixed = float(lib_cfg.get("theta_deg", 0.0))
            angle_offset = float(lib_cfg.get("angle_offset_deg", 0.0))
            lambda_mean = float(lib_cfg.get("lambda_mean", mpl.LAMBDA_MEAN))
            theta_note = {
                "averaged": "averaged spectrum",
                "per_layer": "per layer",
            }.get(theta_mode, f"fixed {theta_fixed:g}°")
            log(
                f"Melt-pool library: {lib_dir}  eta={eta_source}  "
                f"theta={theta_note}  "
                f"offset={angle_offset:g}°  V={[r['velocity_mm_s'] for r in runs]}"
            )
        else:
            melt_path = cfg.get("melt_pool_data") or os.path.join(
                root, "Raptor", "examples", "data", "meltPoolData", "ULI_v1700_theta0_widths.txt"
            )
            melt_path = os.path.abspath(melt_path)
            if not os.path.isfile(melt_path):
                raise FileNotFoundError(f"Melt-pool data not found: {melt_path}")
            runs.append(
                {
                    "velocity_mm_s": float(cfg.get("velocity_m_s", 1.7)) * 1000.0,
                    "depth_scale": float(cfg.get("depth_scale", 0.8)),
                    "height_scale": float(cfg.get("height_scale", 0.4)),
                    "depth_shape": float(cfg.get("depth_shape", 1)),
                    "width_files": {0.0: melt_path},
                    "condition": None,
                }
            )
            theta_mode, theta_fixed, angle_offset, lambda_mean = "fixed", 0.0, 0.0, 1.0

        min_point = np.array([0.0, 0.0, 0.0], dtype=float)
        max_point = np.array([edge_m, edge_m, edge_m], dtype=float)
        bound_box = np.array([min_point, max_point])
        n_runs = len(runs)
        width_cache = {}

        def run_one(run: dict, run_dir: str, k: int) -> dict:
            os.makedirs(run_dir, exist_ok=True)
            t_run = time.time()
            velocity = run["velocity_mm_s"] / 1000.0
            tag = f"[V={run['velocity_mm_s']:g} mm/s {k + 1}/{n_runs}] " if n_runs > 1 else ""

            def prog(stage, frac, msg, step):
                progress(stage, (k + frac) / n_runs, tag + msg, step=step, n_steps=6)

            prog("grid", 0.05, "Creating voxel grid…", 2)
            log(
                f"{tag}RVE edge={edge_m*1e6:.1f} µm  voxel={voxel_m*1e6:.2f} µm  "
                f"P={power} W  v={velocity} m/s  hatch={hatch*1e6:.1f} µm  layer={layer*1e6:.1f} µm  "
                f"D scale={run['depth_scale']:.4g}  H scale={run['height_scale']:.4g}  "
                f"depth shape={run['depth_shape']:.4g}  height shape={height_shape:g}"
            )
            grid = create_grid(voxel_m, bound_box=bound_box)
            n_vox = int(np.prod(np.asarray(grid.shape)))

            prog("scan", 0.12, "Building scan paths…", 3)
            builder = ScanPathBuilder(
                bound_box, power, velocity, hatch, layer, rotation,
                float(np.max(max_point - min_point)), int(cfg.get("extra_layers", 0)),
            )
            builder.generate_layers()
            path_vectors = builder.process_vectors()
            log(f"{tag}Path vectors: {len(path_vectors)}")
            path_csv = os.path.join(run_dir, "path_vectors.csv")
            try:
                from raptor.path_viz import write_path_vectors_csv

                n_saved = write_path_vectors_csv(
                    path_csv, path_vectors, bounds_min=min_point, bounds_max=max_point, layer_height=layer,
                )
                log(f"{tag}Wrote scan paths: {path_csv} ({n_saved} segments in RVE)")
            except Exception as exc:
                log(f"{tag}path_vectors.csv skipped: {exc}")
                path_csv = ""

            backend_label = "CUDA GPU" if compute_info.device == "gpu" else "Numba CPU"
            porosity, theta_groups, theta_files = mpl.compute_porosity_grouped(
                grid,
                path_vectors,
                width_files=run["width_files"],
                velocity_mm_s=run["velocity_mm_s"],
                n_modes=n_modes,
                width_scale=width_scale,
                depth_scale=run["depth_scale"],
                height_scale=run["height_scale"],
                depth_shape=run["depth_shape"],
                height_shape=height_shape,
                enable_random_phases=enable_random,
                seed=seed,
                theta_mode=theta_mode,
                theta_deg=theta_fixed,
                angle_offset_deg=angle_offset,
                lambda_mean=lambda_mean,
                log=lambda m: log(tag + m),
                on_group=lambda gi, n, th, _mode=theta_mode, _be=backend_label: prog(
                    "compute",
                    0.2 + 0.6 * gi / n,
                    (
                        f"Porosity averaged spectrum ({gi + 1}/{n}, {_be})…"
                        if _mode == "averaged"
                        else f"Porosity θ={th:g}° ({gi + 1}/{n}, {_be})…"
                    ),
                    5,
                ),
                width_cache=width_cache,
                tile_width=cfg.get("tile_width"),
                spectral_error_fraction=float(cfg.get("spectral_error_fraction", 0.25)),
                memory_limit_mb=memory_limit_mb,
            )

            phase = np.asarray(porosity)
            n_total = int(phase.size)
            n_pore = int(np.sum(phase == 0))
            n_interior = int(np.sum(phase == 1))
            n_boundary = int(np.sum(phase == 2))
            n_intersect = int(np.sum(phase == 3))
            phi = float(n_pore / n_total) if n_total else 0.0
            log(
                f"{tag}Phases: pore={n_pore} interior={n_interior} boundary={n_boundary} "
                f"intersect={n_intersect}  porosity={phi*100:.3f}%"
            )

            prog("export", 0.85, "Writing VTK + morphology…", 6)
            vti_path = os.path.join(run_dir, "rve.vti")
            morph_path = os.path.join(run_dir, "rve_morphology.csv")
            write_vtk(grid.origin, grid.resolution, porosity, vti_path)
            try:
                shutil.copyfile(vti_path, os.path.join(run_dir, "porosity_map.vti"))
            except Exception:
                pass

            n_defects = 0
            fields = cfg.get("morphology_fields") or ["area", "equivalent_diameter_area"]
            try:
                morphology = compute_morphology(porosity, voxel_m, fields)
                write_morphology(morphology, morph_path)
                if os.path.isfile(morph_path):
                    shutil.copyfile(morph_path, os.path.join(run_dir, "morphology.csv"))
                    first = next(iter(morphology.values()), [])
                    n_defects = int(len(first))
                else:
                    morph_path = ""
            except Exception as exc:
                log(f"{tag}Morphology skipped: {exc}")
                morph_path = ""

            if phase.ndim == 3:
                pore_counts = np.sum(phase == 0, axis=(0, 1))
                z_mid = int(np.argmax(pore_counts)) if int(np.max(pore_counts)) > 0 else int(phase.shape[2] // 2)
                slice2d = phase[:, :, z_mid].astype(np.uint8)
            else:
                z_mid = 0
                slice2d = phase.astype(np.uint8)
            np.savez_compressed(
                os.path.join(run_dir, "preview.npz"),
                phase_slice=slice2d,
                pore_slice=(slice2d == 0).astype(np.uint8),
                origin=np.asarray(grid.origin, dtype=float),
                resolution=float(np.asarray(grid.resolution).ravel()[0]),
                z_index=int(z_mid),
                n_pore_on_slice=int(np.sum(slice2d == 0)),
            )

            preview_png = os.path.join(run_dir, "preview_3d.png")
            try:
                _write_phase_screenshot(vti_path, preview_png)
            except Exception as exc:
                log(f"{tag}3D preview screenshot skipped: {exc}")
                preview_png = ""

            return {
                "rve_edge_m": edge_m,
                "voxel_resolution_m": voxel_m,
                "power_w": power,
                "velocity_m_s": velocity,
                "velocity_mm_s": run["velocity_mm_s"],
                "hatch_m": hatch,
                "layer_m": layer,
                "rotation_deg": rotation,
                "n_modes": n_modes,
                "width_scale": width_scale,
                "depth_scale": run["depth_scale"],
                "height_scale": run["height_scale"],
                "depth_shape": run["depth_shape"],
                "height_shape": height_shape,
                "theta_mode": theta_mode,
                "theta_groups": theta_groups,
                "melt_pool_files": theta_files,
                "melt_pool_data": next(iter(theta_files.values())),
                "library_condition": run["condition"],
                "n_path_vectors": int(len(path_vectors)),
                "path_vectors_csv": path_csv,
                "n_voxels": n_total,
                "n_pore_voxels": n_pore,
                "porosity_fraction": phi,
                "porosity_pct": phi * 100.0,
                "n_interior": n_interior,
                "n_boundary": n_boundary,
                "n_intersect": n_intersect,
                "n_defects": n_defects,
                "vti": vti_path,
                "morphology": morph_path,
                "preview_3d": preview_png,
                "elapsed_s": round(time.time() - t_run, 2),
                "compute_device": compute_info.device,
                "compute_mode": compute_info.mode,
                "compute_detail": compute_info.detail,
                "gpu_name": compute_info.gpu_name,
            }

        summaries = []
        for k, run in enumerate(runs):
            run_dir = outdir if n_runs == 1 else os.path.join(outdir, f"v{run['velocity_mm_s']:g}")
            summaries.append(run_one(run, run_dir, k))

        summary = dict(summaries[-1])
        if n_runs > 1:
            # Surface the last condition at top level so existing preview/3D/download hooks work
            last_dir = os.path.dirname(summary["vti"])
            for name in (
                "rve.vti", "porosity_map.vti", "rve_morphology.csv", "morphology.csv",
                "preview.npz", "preview_3d.png", "path_vectors.csv",
            ):
                src = os.path.join(last_dir, name)
                if os.path.isfile(src):
                    shutil.copyfile(src, os.path.join(outdir, name))
            for key, name in (("vti", "rve.vti"), ("morphology", "rve_morphology.csv"),
                              ("preview_3d", "preview_3d.png"), ("path_vectors_csv", "path_vectors.csv")):
                if summary.get(key):
                    summary[key] = os.path.join(outdir, name)
            cols = [
                "velocity_mm_s", "porosity_pct", "n_defects", "n_pore_voxels", "depth_scale",
                "height_scale", "depth_shape", "height_shape", "n_path_vectors", "elapsed_s",
            ]
            with open(os.path.join(outdir, "sweep_summary.csv"), "w", encoding="utf-8") as f:
                f.write(",".join(cols + ["theta_groups"]) + "\n")
                for s in summaries:
                    groups_txt = ";".join(f"{k}:{v}" for k, v in s["theta_groups"].items())
                    f.write(",".join(str(s[c]) for c in cols) + f",{groups_txt}\n")
        summary["sweep"] = [
            {k: s[k] for k in (
                "velocity_mm_s", "porosity_pct", "n_defects", "n_pore_voxels", "depth_scale",
                "height_scale", "depth_shape", "theta_groups", "elapsed_s",
            )}
            for s in summaries
        ]
        summary["elapsed_s"] = round(time.time() - t0, 2)
        log(f"Preview slice z-index from V={summary['velocity_mm_s']:g} mm/s")

        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        progress("done", 1.0, "Raptor finished", step=6, n_steps=6)
        write_status(True, "ok", summary=summary, log="\n".join(logs))
        log("Worker finished OK")
        print("WORKER_OK")
    except Exception:
        tb = traceback.format_exc()
        log(tb)
        try:
            progress("error", 1.0, "Raptor failed", step=0, n_steps=1)
        except Exception:
            pass
        write_status(False, "error", error=tb, log="\n".join(logs))
        print("WORKER_FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
