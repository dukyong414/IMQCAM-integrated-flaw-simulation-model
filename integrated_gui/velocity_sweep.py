"""5×5×14 mm cuboid, velocities 1300–1700 mm/s, flucWsMTs. Share pack on the Desktop.

No Combined scan-vector file and no edge pores. Raptor still hatches the cuboid
internally at the CMU hatch and layer thickness.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CMU_DIR = os.path.join(ROOT, "CMU team")
EXAMPLE_PARAM_PATH = os.path.join(CMU_DIR, "input_parameters", "params.json")
MATERIAL_PATH = os.path.join(CMU_DIR, "material_models", "test_ti64_rosenthal.json")

VELOCITIES_MM_S = list(range(1300, 1701, 50))
HATCH_UM = 140.0
LAYER_UM = 30.0
# Paper λ shifts the mean width, depth, and cap. Fluctuations stay unscaled.
LAMBDA_MEAN = 1.04
POWER_W = 370.0
SPOT_UM = 100.0
PILLAR_VOLUME_MM3 = 5.0 * 5.0 * 14.0
# LoF this close to an outer face is the unmelted rim, not a pore inside the build.
SURFACE_LOF_MARGIN_MM = 0.5


def desktop_dir() -> str:
    home = os.path.expanduser("~")
    for path in (
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "OneDrive - Northwestern University", "Desktop"),
    ):
        if os.path.isdir(path):
            return path
    path = os.path.join(home, "Desktop")
    os.makedirs(path, exist_ok=True)
    return path


def sweep_already_running(outdir: str) -> bool:
    pid_path = os.path.join(outdir, "sweep.pid")
    if not os.path.isfile(pid_path):
        return False
    try:
        pid = int(open(pid_path, encoding="utf-8").read().strip())
    except (OSError, ValueError):
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_config(
    velocity_mm_s: float,
    *,
    theta_mode: str = "per_layer",
    lambda_mean: float = LAMBDA_MEAN,
) -> dict:
    from integrated_gui import meltpool_library as mpl

    param = _load_json(EXAMPLE_PARAM_PATH)
    param["Power (W)"] = float(param.get("Power (W)", 370))
    param["Velocity (mm/s)"] = float(velocity_mm_s)
    param["Hatch Spacing (um)"] = float(param.get("Hatch Spacing (um)", 140))
    param["Layer Thickness (um)"] = float(param.get("Layer Thickness (um)", 30))
    props = _load_json(MATERIAL_PATH)
    for key, value in param.items():
        if isinstance(value, (int, float)):
            props[key] = value
    return {
        "plate_mode": "single_cuboid",
        "n_rows": 19,
        "n_cols": 3,
        "n_strips": 4,
        "cube_size": 10.0,
        "cube_height": 14.0,
        "cuboid_margin_mm": 5.0,
        "cuboid_fill_plate": False,
        "cuboid_length_mm": 5.0,
        "cuboid_width_mm": 5.0,
        "stl_path": None,
        "stl_margin_mm": 2.0,
        "stl_max_height_mm": None,
        "stl_sector_fraction": 1.0,
        "stl_sector_start_deg": 0.0,
        "raptor_band_z_mm": None,
        "raptor_band_layers": 15,
        "raptor_stl_domain": "volume",
        "plate_w": 250.0,
        "plate_h": 250.0,
        "seed": 42,
        "max_pores": 0,
        "totalpores": 0,
        "mean_porosity": 0.02,
        "enable_scan_vectors": False,
        "scan_strategy": "stripes",
        "stripe_width_mm": 10.0,
        "use_process_scan_params": True,
        "make_spatter_anim": False,
        "scan_full_part": True,
        "n_edge_pores": 0,
        "hatch_spacing_mm": float(param["Hatch Spacing (um)"]) * 1e-3,
        "layer_thickness_mm": float(param["Layer Thickness (um)"]) * 1e-3,
        "max_scan_layers": 99999,
        "layer_rotation_deg": 67.0,
        "base_angle_deg": None,
        "batch_rows": 4,
        "batch_cols": 4,
        "pore_models": ["lof", "keyhole", "gas"],
        "width_scale": 1.0,
        "meltpool_width_um": process_point(float(velocity_mm_s), lambda_mean=lambda_mean)["meltpool_width_um"],
        "meltpool_width_used_um": process_point(float(velocity_mm_s), lambda_mean=lambda_mean)["meltpool_width_used_um"],
        "param_dict": param,
        "props": props,
        "compute_device": "auto",
        "use_raptor_lof": True,
        "raptor_roi": None,
        "raptor_library": {
            "dir": mpl.DEFAULT_LIBRARY_DIR,
            "eta_source": "flucWsMTs",
            "theta_mode": theta_mode,
            "angle_offset_deg": 0.0,
            "use_eta": True,
            "use_depth_shape": True,
            "lambda_mean": float(lambda_mean),
        },
        "gas_flow": "+x",
        "gas_flow_sharpness": 2.0,
        "flow_perturbation": 0.65,
        "spatter_fraction": 0.0,
        "raptor_rve_edge_m": 5.0e-4,
        "raptor_voxel_m": 5.0e-6,
        # 5×5×14 mm at 5 µm is 2.8e9 voxels. Keep that size; do not coarsen.
        "raptor_whole_build_max_voxels": 3_000_000_000,
        "raptor_n_modes": 50,
    }


def _read_table(path: str) -> pd.DataFrame:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _pore_tables(run_dir: str):
    """LoF shapes plus CMU keyhole/gas (spheres). Edge pores are not generated."""
    from integrated_gui.raptor_view import pore_type_table

    rap = os.path.join(run_dir, "raptor_rve")
    shapes_path = os.path.join(rap, "lof_pore_shapes.csv")
    domain_path = os.path.join(rap, "combined_pores_in_domain.csv")
    shapes = _read_table(shapes_path)
    domain = _read_table(domain_path)
    types = pore_type_table(
        domain if len(domain) else None,
        shapes if len(shapes) else None,
    )
    rows = []
    if len(shapes):
        for _, r in shapes.iterrows():
            rows.append(
                {
                    "type": "raptor_lof",
                    "x_mm": r.get("x_mm"),
                    "y_mm": r.get("y_mm"),
                    "z_mm": r.get("z_mm"),
                    "esd_um": r.get("esd_um"),
                    "sphericity": r.get("sphericity"),
                    "aspect_ratio": r.get("aspect_ratio"),
                    "major_axis_um": r.get("major_axis_um"),
                    "minor_axis_um": r.get("minor_axis_um"),
                    "n_voxels": r.get("n_voxels"),
                    "volume_um3": r.get("volume_um3"),
                }
            )
    if len(types):
        other = types.loc[types["type"].astype(str) != "raptor_lof"]
        for _, r in other.iterrows():
            esd = r.get("esd_um")
            rows.append(
                {
                    "type": r.get("type"),
                    "x_mm": r.get("x_mm"),
                    "y_mm": r.get("y_mm"),
                    "z_mm": r.get("z_mm"),
                    "esd_um": esd,
                    "sphericity": 1.0,
                    "aspect_ratio": 1.0,
                    "major_axis_um": esd,
                    "minor_axis_um": esd,
                    "n_voxels": np.nan,
                    "volume_um3": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _save_shape_chart(df: pd.DataFrame, png_path: str) -> None:
    import matplotlib.pyplot as plt

    from integrated_gui.raptor_view import PORE_TYPE_STYLE

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    titles = (
        "Size distribution (ESD, µm)",
        "Sphericity",
        "Aspect ratio (longest/shortest)",
    )
    cols = ("esd_um", "sphericity", "aspect_ratio")
    kinds = df["type"].astype(str)
    for key, (label, color) in PORE_TYPE_STYLE.items():
        sub = df.loc[kinds == key]
        if not len(sub):
            continue
        for ax, col in zip(axes, cols):
            vals = sub[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            ax.hist(
                vals,
                bins=40,
                weights=np.full(len(vals), 100.0 / len(vals)),
                color=color,
                alpha=0.55,
                label=f"{label} ({len(sub):,})",
            )
    axes[0].set_ylabel("% of that type")
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    proc = os.path.join(os.path.dirname(png_path), "process.json")
    top = 1.0
    if os.path.isfile(proc):
        with open(proc, encoding="utf-8") as f:
            fig.suptitle(process_caption(json.load(f)), fontsize=11)
        top = 0.88
    fig.tight_layout(rect=(0, 0.08, 1, top))
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


_PHASE_RGB = np.array(
    [
        [255, 0, 0],
        [179, 179, 179],
        [51, 51, 51],
        [255, 255, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)


def _add_phase_face(pl, phase2d: np.ndarray, origin, axis_u, axis_v) -> None:
    """One outer face, one texel per voxel, so hatch boundaries stay sharp."""
    import pyvista as pv

    ph = np.clip(np.asarray(phase2d, dtype=np.int32), 0, 4)
    rgb = np.ascontiguousarray(_PHASE_RGB[ph])
    nu, nv = ph.shape
    u = np.asarray(axis_u, dtype=float).reshape(3)
    v = np.asarray(axis_v, dtype=float).reshape(3)
    o = np.asarray(origin, dtype=float).reshape(3)
    # Texture maps (i, j) across the plane; flip rows so +v is up
    tex = pv.numpy_to_texture(np.ascontiguousarray(np.flipud(rgb)))
    tex.interpolate = False
    corners = np.vstack(
        [
            o,
            o + nu * u,
            o + nu * u + nv * v,
            o + nv * v,
        ]
    )
    face = pv.PolyData(corners, faces=np.array([4, 0, 1, 2, 3], dtype=np.int64))
    face.active_texture_coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    pl.add_mesh(face, texture=tex, lighting=False, show_scalar_bar=False)


def _save_melted_build(vti_path: str, png_path: str) -> None:
    """Outer faces of the Raptor phase grid (pore, melted, boundary, track overlap)."""
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    pv.OFF_SCREEN = True
    img = pv.read(vti_path)
    d0, d1, d2 = img.dimensions  # VTK axes = Raptor (z, y, x)
    phase = np.asarray(img["Phase"]).reshape(d2, d1, d0)
    # The outer skin is the scan running past the part. Draw it as melted metal,
    # the same solid the previous figure showed. Pores stay in the interior view.
    phase = np.where(np.isin(phase, (0, 2, 3)), 1, phase)
    v = float(img.spacing[0]) * 1e3
    o = np.asarray(img.origin, dtype=float) * 1e3
    nx, ny, nz = phase.shape
    pl = pv.Plotter(off_screen=True, window_size=(1200, 980), border=False)
    pl.set_background("white")
    # x faces
    _add_phase_face(pl, phase[0], o, (0, v, 0), (0, 0, v))
    _add_phase_face(pl, phase[-1], o + np.array([(nx - 1) * v, 0, 0]), (0, v, 0), (0, 0, v))
    # y faces
    _add_phase_face(pl, phase[:, 0, :], o, (v, 0, 0), (0, 0, v))
    _add_phase_face(pl, phase[:, -1, :], o + np.array([0, (ny - 1) * v, 0]), (v, 0, 0), (0, 0, v))
    # z faces
    _add_phase_face(pl, phase[:, :, 0], o, (v, 0, 0), (0, v, 0))
    _add_phase_face(pl, phase[:, :, -1], o + np.array([0, 0, (nz - 1) * v]), (v, 0, 0), (0, v, 0))
    pl.add_legend(
        [
            ["Melted", "lightgrey"],
            ["Boundary", "black"],
            ["Track overlap", "yellow"],
        ],
        bcolor="white",
        face="rectangle",
    )
    pl.add_axes()
    pl.camera_position = "iso"
    pl.reset_camera()
    pl.camera.zoom(0.85)
    pl.screenshot(png_path)
    pl.close()


def _save_surface_sweep(npz_path: str, gif_path: str, n_frames: int = 36) -> None:
    """GIF of the camera traveling up one outer face of the Raptor pillar.

    Phase 0 on the face is drawn as melted metal, so the sweep shows hatch
    marks and not red pores.
    """
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    from PIL import Image

    pv.OFF_SCREEN = True
    with np.load(npz_path) as stored:
        d = {key: stored[key] for key in stored.files}
    nx, ny, nz = [int(v) for v in d["shape"]]
    if d["strides"].shape == ():
        sx = sy = sz = int(d["strides"])
    else:
        sx, sy, sz = [int(v) for v in np.asarray(d["strides"]).reshape(-1)[:3]]
    origin = np.asarray(d["origin_mm"], dtype=float)
    voxel = float(d["voxel_mm"])
    x0, y0, z0 = origin
    x1 = x0 + (nx - 1) * voxel
    y1 = y0 + (ny - 1) * voxel
    z1 = z0 + (nz - 1) * voxel

    def _paint(face: np.ndarray) -> np.ndarray:
        phase = np.clip(np.asarray(face), 0, 4).copy()
        phase[phase == 0] = 1
        return phase

    def _scene() -> "pv.Plotter":
        pl = pv.Plotter(off_screen=True, window_size=(820, 980), border=False)
        pl.set_background("white")
        _add_phase_face(pl, _paint(d["face_x0"]), origin, (0, sy * voxel, 0), (0, 0, sz * voxel))
        _add_phase_face(pl, _paint(d["face_x1"]), origin + np.array([(nx - 1) * voxel, 0, 0]), (0, sy * voxel, 0), (0, 0, sz * voxel))
        _add_phase_face(pl, _paint(d["face_y0"]), origin, (sx * voxel, 0, 0), (0, 0, sz * voxel))
        _add_phase_face(pl, _paint(d["face_y1"]), origin + np.array([0, (ny - 1) * voxel, 0]), (sx * voxel, 0, 0), (0, 0, sz * voxel))
        _add_phase_face(pl, _paint(d["face_z0"]), origin, (sx * voxel, 0, 0), (0, sy * voxel, 0))
        _add_phase_face(pl, _paint(d["face_z1"]), origin + np.array([0, 0, (nz - 1) * voxel]), (sx * voxel, 0, 0), (0, sy * voxel, 0))
        return pl

    frames = []
    cx = 0.5 * (x0 + x1)
    for i in range(n_frames):
        z = z0 + (z1 - z0) * (i + 0.5) / n_frames
        pl = _scene()
        pl.add_mesh(pv.Line((x0, y0, z), (x1, y0, z)), color="black", line_width=6)
        pl.camera.position = (cx, y0 - 7.5, z)
        pl.camera.focal_point = (cx, y0, z)
        pl.camera.up = (0, 0, 1)
        pl.camera.zoom(1.35)
        pl.add_text(f"Surface sweep   z = {z - z0:.1f} mm", font_size=12, color="black")
        shot = pl.screenshot(return_img=True)
        pl.close()
        frames.append(Image.fromarray(shot))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )


def _render_volume_sweep(vti_path: str, gif_path: str, n_frames: int = 28) -> None:
    """Isometric view of the Raptor block. A cut surface advances along Y and builds the solid.

    The camera stays fixed. Each frame keeps the material behind the moving plane.
    """
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    from PIL import Image

    pv.OFF_SCREEN = True
    vol = pv.read(vti_path)
    # Raptor (x, y, z) = scan, hatch, height. The direction matrix swaps VTK axes.
    dims = vol.dimensions
    phase = np.clip(np.asarray(vol["Phase"]).reshape(dims[2], dims[1], dims[0]), 0, 3)
    nx, ny, nz = phase.shape
    voxel = float(vol.spacing[0])
    xmin, _, ymin, _, zmin, _ = [float(v) for v in vol.bounds]
    # Stay off the outer skin. The top layer is unmelted and paints red.
    skin_xy = max(8, int(round(0.12e-3 / voxel)))
    skin_z = max(6, int(round(0.10e-3 / voxel)))
    j0, j1 = skin_xy, ny - 1 - skin_xy
    k0, k1 = skin_z, nz - 1 - skin_z
    i0 = max(2, int(0.04 * nx))
    i1 = min(nx - 3, int(0.96 * nx))
    ys = np.linspace(j0, j1, n_frames).astype(int)
    x0 = xmin + i0 * voxel
    x1 = xmin + i1 * voxel
    y_start = ymin + int(ys[0]) * voxel
    y_end = ymin + j1 * voxel
    z0 = zmin + k0 * voxel
    z1 = zmin + k1 * voxel
    cx, cy, cz = 0.5 * (x0 + x1), 0.5 * (y_start + y_end), 0.5 * (z0 + z1)
    dx, dy, dz = x1 - x0, y_end - y_start, z1 - z0
    # Fixed isometric camera: in front of the Y cut, off one end, and above.
    eye = (cx + 1.55 * dx, cy + 1.65 * dy, cz + 1.15 * dz)
    frames = []
    for j in ys:
        j = int(j)
        y = ymin + j * voxel
        cut = phase[i0 : i1 + 1, j, k0 : k1 + 1]
        grown = phase[i0 : i1 + 1, int(ys[0]) : j + 1, k0 : k1 + 1]
        pl = pv.Plotter(off_screen=True, window_size=(1400, 1000), border=False)
        pl.set_background("white")
        outline = pv.Box(bounds=(x0, x1, y_start, y_end, z0, z1)).outline()
        pl.add_mesh(outline, color="black", line_width=2)
        # Leading surface moves along Y.
        _add_phase_face(
            pl, cut, (x0, y, z0),
            (voxel * (cut.shape[0] - 1) / cut.shape[0], 0.0, 0.0),
            (0.0, 0.0, voxel * (cut.shape[1] - 1) / cut.shape[1]),
        )
        if grown.shape[1] > 1:
            top = grown[:, :, -1]
            end = grown[-1, :, :]
            _add_phase_face(
                pl, top, (x0, y_start, z1),
                (voxel * (top.shape[0] - 1) / top.shape[0], 0.0, 0.0),
                (0.0, voxel * (top.shape[1] - 1) / top.shape[1], 0.0),
            )
            _add_phase_face(
                pl, end, (x1, y_start, z0),
                (0.0, voxel * (end.shape[0] - 1) / end.shape[0], 0.0),
                (0.0, 0.0, voxel * (end.shape[1] - 1) / end.shape[1]),
            )
        pl.add_legend(
            [
                ["Pore", "red"],
                ["Melted", "lightgrey"],
                ["Boundary", "black"],
                ["Intersection", "yellow"],
            ],
            bcolor="white",
            face="rectangle",
        )
        pl.add_axes()
        pl.camera_position = [eye, (cx, cy, cz), (0, 0, 1)]
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = 0.46 * float(np.linalg.norm([dx, dy, 1.6 * dz]))
        pl.add_text(
            f"Volume sweep   {(y - ymin) * 1e3:.2f} mm along Y",
            font_size=12,
            color="black",
        )
        shot = pl.screenshot(return_img=True)
        frames.append(Image.fromarray(shot))
        pl.close()
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )


def _save_volume_sweep(
    velocity_mm_s: float,
    gif_path: str,
    n_frames: int = 28,
    *,
    theta_mode: str = "averaged",
    lambda_mean: float = LAMBDA_MEAN,
) -> None:
    """Raptor block at this speed, then an isometric sweep that builds the solid."""
    import tempfile

    from integrated_gui.combined_raptor import run_raptor_rve

    # Short block, 10 µm voxels: enough layers to show the melt-pool cross-section.
    # The full 14 mm grid is too large to keep, and this uses the same hatch,
    # layer thickness, and flucWsMTs melt pool as the pillar.
    cfg = build_config(float(velocity_mm_s), theta_mode=theta_mode, lambda_mean=lambda_mean)
    raptor_cfg = {
        "power_w": POWER_W,
        "velocity_m_s": float(velocity_mm_s) / 1000.0,
        "hatch_m": HATCH_UM * 1e-6,
        "layer_m": LAYER_UM * 1e-6,
        # Parallel tracks so a cut through the volume shows the melt-pool cross-section.
        "rotation_deg": 0.0,
        "size_xyz_mm": [4.0, 3.2, 1.8],
        "voxel_resolution_m": 10.0e-6,
        "raptor_max_voxels": 80_000_000,
        "n_modes": int(cfg.get("raptor_n_modes", 50)),
        "random_seed": int(cfg.get("seed", 42)),
        "compute_device": "auto",
        "enable_random_phases": True,
        "scan_strategy": "stripes",
        "stripe_width_mm": 10.0,
        "skip_spatter_anim": True,
        "viz_only": True,
        "_skip_fine_lof": True,
        "max_scan_layers": 80,
        "width_scale": 1.0,
        "library": dict(cfg["raptor_library"]),
        "batch_rows": 4,
        "batch_cols": 4,
    }
    tmp = tempfile.mkdtemp(prefix="raptor_vol_")
    try:
        run_raptor_rve(raptor_cfg, outdir=tmp, log=print)
        vti = os.path.join(tmp, "rve.vti")
        if not os.path.isfile(vti):
            raise FileNotFoundError(f"Raptor did not write {vti}")
        _render_volume_sweep(vti, gif_path, n_frames=n_frames)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _save_all_pore_sources(npz_path: str, pores_csv: str, png_path: str) -> None:
    """See-through melted build with every pore type, matching the GUI 'All pore sources' view."""
    from integrated_gui.raptor_view import make_plotly_phase_figure

    df = pd.read_csv(pores_csv)
    need = {"type", "x_mm", "y_mm", "z_mm", "esd_um"}
    if not need.issubset(df.columns):
        return
    fig = make_plotly_phase_figure(
        npz_path,
        title="All pore sources",
        pore_types=df[list(need)] if len(df) else None,
        path_csv=None,
    )
    names = [str(getattr(tr, "name", "") or "") for tr in fig.data]
    has_objects = any(n.startswith("LoF (Raptor)") or n.startswith("Keyhole") or n.startswith("Gas") for n in names)
    legend_names = {"Pore", "Melted", "Melt boundary", "Track overlap"}
    for tr in fig.data:
        kind = str(getattr(tr, "type", ""))
        name = str(getattr(tr, "name", "") or "")
        if kind in ("surface", "mesh3d"):
            colors = getattr(tr, "surfacecolor", None)
            if colors is not None:
                arr = np.asarray(colors, dtype=float).copy()
                arr[arr <= 0] = 1
                tr.surfacecolor = arr
            tr.opacity = 0.18
            tr.visible = True
        elif name.startswith("LoF pores"):
            tr.visible = not has_objects
        elif name in legend_names:
            tr.visible = True
        else:
            tr.visible = True
    proc_path = os.path.join(os.path.dirname(pores_csv), "process.json")
    title = "All pore sources"
    if os.path.isfile(proc_path):
        with open(proc_path, encoding="utf-8") as f:
            title = process_caption(json.load(f))
    fig.update_layout(title=title, updatemenus=[], margin=dict(l=0, r=0, t=64, b=0))
    fig.write_image(png_path, width=900, height=1100, scale=2)


def process_point(velocity_mm_s: float, *, lambda_mean: float = LAMBDA_MEAN) -> dict:
    """Hatch spacing and the flucWsMTs melt-pool width at one scan speed."""
    from integrated_gui import meltpool_library as mpl

    cond = mpl.resolve_condition(
        {"dir": mpl.DEFAULT_LIBRARY_DIR, "eta_source": "flucWsMTs"},
        float(velocity_mm_s),
    )
    width_um = float(cond.mean_width_m) * 1e6
    return {
        "velocity_mm_s": float(velocity_mm_s),
        "power_w": POWER_W,
        "spot_um": SPOT_UM,
        "hatch_um": HATCH_UM,
        "layer_um": LAYER_UM,
        "meltpool_width_um": width_um,
        "width_scale": 1.0,
        "lambda_mean": float(lambda_mean),
        "meltpool_width_used_um": width_um * float(lambda_mean),
        "eta_source": "flucWsMTs",
    }


def process_caption(info: dict) -> str:
    return (
        f"Hatch spacing {info['hatch_um']:.0f} µm"
        f"   ·   mean width {info['meltpool_width_um']:.0f} µm"
        f"   ·   λ={float(info.get('lambda_mean', 1)):g} mean width {info['meltpool_width_used_um']:.0f} µm"
    )


_PREDICTION_KEYS = (
    "density_unit",
    "pillar_volume_mm3",
    "cmu_lof_per_mm3",
    "cmu_keyhole_per_mm3",
    "cmu_gas_per_mm3",
    "cmu_lof_count",
    "cmu_keyhole_count",
    "cmu_gas_count",
    "raptor_lof_per_mm3",
    "raptor_lof_count",
    "lof_source",
)


def write_process(folder: str, velocity_mm_s: float, *, lambda_mean: float = LAMBDA_MEAN) -> dict:
    info = process_point(velocity_mm_s, lambda_mean=lambda_mean)
    path = os.path.join(folder, "process.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            for key in _PREDICTION_KEYS:
                if key in old:
                    info[key] = old[key]
        except Exception:
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def cmu_number_densities(velocity_mm_s: float) -> dict:
    """CMU LoF, keyhole, and gas number densities in pores/mm³."""
    from integrated_gui.cmu_labeled import run_cmu_labeled_samples

    params = _load_json(EXAMPLE_PARAM_PATH)
    params["Power (W)"] = POWER_W
    params["Velocity (mm/s)"] = float(velocity_mm_s)
    params["Hatch Spacing (um)"] = HATCH_UM
    params["Layer Thickness (um)"] = LAYER_UM
    params["Spot Size (um)"] = SPOT_UM
    props = _load_json(MATERIAL_PATH)
    for key, value in params.items():
        if isinstance(value, (int, float)):
            props[key] = value
    prev = os.getcwd()
    os.chdir(CMU_DIR)
    try:
        _cat, _vol, meta = run_cmu_labeled_samples(
            params, props, ["lof", "keyhole", "gas"], totalpores=2000
        )
    finally:
        os.chdir(prev)
    dens = {str(k).lower(): float(v) for k, v in (meta.get("densities") or {}).items()}
    return dens


def _distance_to_surface_mm(xyz, origin_mm, voxel_mm: float, shape) -> np.ndarray:
    """Shortest distance, in mm, from each point to an outer face of the grid."""
    origin = np.asarray(origin_mm, dtype=float)
    voxel = float(voxel_mm)
    nx, ny, nz = [int(v) for v in shape]
    ijk = (np.asarray(xyz, dtype=float) - origin) / voxel
    dist = np.minimum(
        np.minimum(np.minimum(ijk[:, 0], (nx - 1) - ijk[:, 0]), np.minimum(ijk[:, 1], (ny - 1) - ijk[:, 1])),
        np.minimum(ijk[:, 2], (nz - 1) - ijk[:, 2]),
    )
    return dist * voxel


def drop_near_surface_lof(df: pd.DataFrame, origin_mm, voxel_mm: float, shape) -> pd.DataFrame:
    """Remove Raptor LoF whose center is within SURFACE_LOF_MARGIN_MM of a face."""
    if df is None or not len(df) or "type" not in df.columns:
        return df
    lof = df["type"].astype(str).str.lower().isin(["raptor_lof", "lof"]).to_numpy()
    if not lof.any():
        return df
    dist_mm = _distance_to_surface_mm(
        df.loc[lof, ["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float),
        origin_mm,
        voxel_mm,
        shape,
    )
    keep = np.ones(len(df), dtype=bool)
    keep[np.flatnonzero(lof)] = dist_mm > SURFACE_LOF_MARGIN_MM
    return df.loc[keep].reset_index(drop=True)


def _raptor_lof_count(csv_path: str, npz_path: str = "") -> int:
    """Raptor LoF objects inside the pillar, in the same count CMU uses.

    Centers within SURFACE_LOF_MARGIN_MM of an outer face are the unmelted rim,
    so they are left out of the number density.
    """
    if not os.path.isfile(csv_path):
        return 0
    df = pd.read_csv(csv_path)
    if "type" not in df.columns or not len(df):
        return 0
    lof = df.loc[df["type"].astype(str).str.lower().isin(["raptor_lof", "lof"])]
    if not len(lof) or not npz_path or not os.path.isfile(npz_path):
        return int(len(lof))
    with np.load(npz_path) as z:
        origin = np.asarray(z["origin_mm"], dtype=float)
        voxel = float(z["voxel_mm"])
        nx, ny, nz = [int(v) for v in z["shape"]]
    dist_mm = _distance_to_surface_mm(
        lof[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float),
        origin,
        voxel,
        (nx, ny, nz),
    )
    return int((dist_mm > SURFACE_LOF_MARGIN_MM).sum())


def write_predictions(outdir: str, *, lambda_mean: float = LAMBDA_MEAN) -> None:
    """CMU and Raptor number densities, both in pores/mm³.

    Raptor LoF replaces the CMU LoF prediction. Keyhole and gas stay CMU.
    """
    rows = []
    for name in sorted(os.listdir(outdir)):
        folder = os.path.join(outdir, name)
        if not (name.startswith("V_") and os.path.isdir(folder)):
            continue
        try:
            velocity = float(name.split("_", 1)[1])
        except ValueError:
            continue
        stored = {}
        path = os.path.join(folder, "process.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                stored = json.load(f)
        if "cmu_lof_per_mm3" in stored:
            dens = {
                "lof": float(stored["cmu_lof_per_mm3"]),
                "keyhole": float(stored.get("cmu_keyhole_per_mm3") or 0.0),
                "gas": float(stored.get("cmu_gas_per_mm3") or 0.0),
            }
        else:
            dens = cmu_number_densities(velocity)
        n_raptor = _raptor_lof_count(
            os.path.join(folder, "pores.csv"),
            os.path.join(folder, "view3d.npz"),
        )
        vol = PILLAR_VOLUME_MM3
        info = process_point(velocity, lambda_mean=float(stored.get("lambda_mean", lambda_mean)))
        info.update(stored)
        info.update(
            {
                "density_unit": "pores/mm3",
                "pillar_volume_mm3": vol,
                "cmu_lof_per_mm3": dens.get("lof", 0.0),
                "cmu_keyhole_per_mm3": dens.get("keyhole", 0.0),
                "cmu_gas_per_mm3": dens.get("gas", 0.0),
                "cmu_lof_count": dens.get("lof", 0.0) * vol,
                "cmu_keyhole_count": dens.get("keyhole", 0.0) * vol,
                "cmu_gas_count": dens.get("gas", 0.0) * vol,
                "raptor_lof_per_mm3": n_raptor / vol,
                "raptor_lof_count": n_raptor,
                "lof_source": "raptor_replaces_cmu",
            }
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        rows.append(info)
        print(
            f"{name}: CMU LoF {info['cmu_lof_per_mm3']:.3f}  "
            f"Raptor LoF {info['raptor_lof_per_mm3']:.3f} pores/mm3  "
            f"keyhole {info['cmu_keyhole_per_mm3']:.3f}  gas {info['cmu_gas_per_mm3']:.3f}",
            flush=True,
        )
    if not rows:
        return
    cols = [
        "velocity_mm_s",
        "density_unit",
        "pillar_volume_mm3",
        "cmu_lof_per_mm3",
        "raptor_lof_per_mm3",
        "cmu_keyhole_per_mm3",
        "cmu_gas_per_mm3",
        "cmu_lof_count",
        "raptor_lof_count",
        "cmu_keyhole_count",
        "cmu_gas_count",
        "lof_source",
    ]
    path = os.path.join(outdir, "predictions.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for info in rows:
            f.write(",".join(str(info.get(c, "")) for c in cols) + "\n")


def write_conditions_table(outdir: str, *, lambda_mean: float = LAMBDA_MEAN) -> None:
    rows = []
    for name in sorted(os.listdir(outdir)):
        folder = os.path.join(outdir, name)
        if not (name.startswith("V_") and os.path.isdir(folder)):
            continue
        try:
            velocity = float(name.split("_", 1)[1])
        except ValueError:
            continue
        stored_lam = lambda_mean
        proc_path = os.path.join(folder, "process.json")
        if os.path.isfile(proc_path):
            try:
                with open(proc_path, encoding="utf-8") as f:
                    stored_lam = float(json.load(f).get("lambda_mean", lambda_mean))
            except (OSError, ValueError, TypeError):
                stored_lam = lambda_mean
        info = write_process(folder, velocity, lambda_mean=stored_lam)
        rows.append(info)
    if not rows:
        return
    path = os.path.join(outdir, "conditions.csv")
    with open(path, "w", encoding="utf-8") as f:
        cols = [
            "velocity_mm_s",
            "hatch_um",
            "layer_um",
            "meltpool_width_um",
            "width_scale",
            "meltpool_width_used_um",
            "power_w",
            "spot_um",
            "eta_source",
        ]
        f.write(",".join(cols) + "\n")
        for info in rows:
            f.write(",".join(str(info[c]) for c in cols) + "\n")


def _stamp_png(path: str, text: str) -> None:
    """Add one caption bar. A second call with the same text does not stack bars."""
    marker = path + ".caption.txt"
    if os.path.isfile(marker) and os.path.getmtime(marker) >= os.path.getmtime(path):
        with open(marker, encoding="utf-8") as f:
            if f.read() == text:
                return
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(path).convert("RGB")
    font_path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf")
    font = ImageFont.truetype(font_path, 28) if os.path.isfile(font_path) else ImageFont.load_default()
    bar_h = 54
    canvas = Image.new("RGB", (im.width, im.height + bar_h), "white")
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, im.height + 12), text, fill="black", font=font)
    canvas.save(path)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(text)


def export_share(
    run_dir: str,
    dest: str,
    velocity_mm_s: float,
    *,
    lambda_mean: float = LAMBDA_MEAN,
    theta_mode: str = "averaged",
) -> None:
    """Write the four share files, then drop the working run directory."""
    os.makedirs(dest, exist_ok=True)
    rap = os.path.join(run_dir, "raptor_rve")
    vti = os.path.join(rap, "rve.vti")
    if not os.path.isfile(vti):
        vti = os.path.join(rap, "porosity_map.vti")
    view3d = os.path.join(rap, "view3d.npz")
    if not os.path.isfile(view3d) and os.path.isfile(vti):
        from integrated_gui.raptor_view import view3d_from_vti

        view3d_from_vti(vti, view3d)
    info = write_process(dest, velocity_mm_s, lambda_mean=lambda_mean)
    caption = process_caption(info)
    if os.path.isfile(vti):
        melted_png = os.path.join(dest, "melted_build.png")
        _save_melted_build(vti, melted_png)
        _stamp_png(melted_png, caption)
    if os.path.isfile(view3d):
        shutil.copy2(view3d, os.path.join(dest, "view3d.npz"))
    pores = _pore_tables(run_dir)
    view_for_margin = os.path.join(dest, "view3d.npz")
    if len(pores) and os.path.isfile(view_for_margin):
        with np.load(view_for_margin) as grid:
            pores = drop_near_surface_lof(
                pores,
                grid["origin_mm"],
                float(grid["voxel_mm"]),
                grid["shape"],
            )
    if len(pores):
        pores.to_csv(os.path.join(dest, "pores.csv"), index=False)
        _save_shape_chart(pores, os.path.join(dest, "pore_size_shape.png"))
    view_out = os.path.join(dest, "view3d.npz")
    pores_out = os.path.join(dest, "pores.csv")
    if os.path.isfile(view_out) and os.path.isfile(pores_out):
        _save_all_pore_sources(view_out, pores_out, os.path.join(dest, "all_pore_sources.png"))
    if os.path.isfile(view_out):
        _save_volume_sweep(
            float(velocity_mm_s),
            os.path.join(dest, "volume_sweep.gif"),
            theta_mode=theta_mode,
            lambda_mean=lambda_mean,
        )
    del velocity_mm_s
    if os.path.abspath(os.path.dirname(run_dir)) == os.path.abspath(dest):
        shutil.rmtree(run_dir, ignore_errors=True)


def _readme(
    path: str,
    outdir: str | None = None,
    *,
    theta_mode: str = "per_layer",
    lambda_mean: float = LAMBDA_MEAN,
) -> None:
    table = ""
    cond = os.path.join(outdir or os.path.dirname(path), "conditions.csv")
    if os.path.isfile(cond):
        lines = ["", "Velocity (mm/s)   Hatch (µm)   Melt pool width (µm)   Width used in Raptor (µm)"]
        with open(cond, encoding="utf-8") as f:
            next(f, None)
            for row in f:
                parts = row.strip().split(",")
                if len(parts) < 6:
                    continue
                lines.append(
                    f"{float(parts[0]):8.0f}         {float(parts[1]):6.0f}        "
                    f"{float(parts[3]):8.1f}                 {float(parts[5]):8.1f}"
                )
        table = "\n".join(lines) + "\n"
    pred_table = ""
    pred = os.path.join(outdir or os.path.dirname(path), "predictions.csv")
    if os.path.isfile(pred):
        plines = [
            "",
            "Pore number density. CMU and Raptor are both in pores/mm³.",
            "Raptor LoF replaces the CMU LoF prediction. Keyhole and gas stay the CMU prediction.",
            "Raptor LoF counts pores more than 0.5 mm inside the pillar. LoF closer to the surface is left out.",
            "Counts are that density times the 5 x 5 x 14 mm pillar (350 mm³).",
            "",
            "Velocity   CMU LoF   Raptor LoF   CMU keyhole   CMU gas    (pores/mm³)",
        ]
        with open(pred, encoding="utf-8") as f:
            next(f, None)
            for row in f:
                parts = row.strip().split(",")
                if len(parts) < 7:
                    continue
                plines.append(
                    f"{float(parts[0]):7.0f}    {float(parts[3]):7.3f}    {float(parts[4]):7.3f}"
                    f"      {float(parts[5]):7.3f}     {float(parts[6]):7.3f}"
                )
        pred_table = "\n".join(plines) + "\n"
    text = f"""Raptor 5 x 5 x 14 mm velocity sweep
====================================

Geometry: 5 mm x 5 mm x 14 mm pillar.
Velocities: 1300 to 1700 mm/s, step 50.
Power 370 W, spot 100 µm.
Hatch spacing: {HATCH_UM:.0f} µm on every velocity.
Layer thickness: {LAYER_UM:.0f} µm.
Melt pool width: flucWsMTs mean width at that velocity.
{('One melt pool per velocity: the cosine spectrum is the average of the eight single-track orientation files. ' if theta_mode == 'averaged' else 'Each scan vector uses the orientation file nearest its heading. ')}λ={float(lambda_mean):g} shifts the mean width, depth, and cap only, matching the 0.5 mm cube. Voxel size is 5 µm. Scan paths come from Raptor's ScanPathBuilder.
No edge pores and no scan-vector file.
{table}{pred_table}

Each V_#### folder has only:
  melted_build.png       outer faces (grey melted, black boundary, yellow overlap)
  volume_sweep.gif       isometric view; a surface sweeps along Y and builds the solid
  all_pore_sources.png  see-through build with every pore type (LoF, keyhole, gas)
  pore_size_shape.png   ESD, sphericity, aspect ratio
  pores.csv             every pore: location (plate mm) and shape
  view3d.npz            3D viewer data
  process.json          hatch, melt-pool width, and the pores/mm³ comparison

The folder also has conditions.csv (hatch and width) and predictions.csv
(CMU and Raptor, both in pores/mm³). Raptor LoF is the value that replaces CMU LoF.

View the results
----------------
Unzip the whole folder, then double-click "Open results GUI.bat".

The python folder next to the bat file is a portable Windows 64-bit Python.
The other computer does not need Python installed.
The GUI opens in the browser at http://127.0.0.1:8055
Leave the black command window open while you use the GUI. Close it to stop the GUI.

If Windows says the file is protected, choose More info, then Run anyway.

X and Y are plate millimeters. Z is height above the plate.
Keyhole and gas pores are spheres (sphericity 1, aspect ratio 1).
LoF sphericity is measured only for pores with enough voxels.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


_VIEWER_PACKAGES = ("numpy", "pandas", "plotly", "dash", "dash-bootstrap-components")
_EMBED_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
_GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _download(url: str, dest: str) -> None:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, dest)


def bundle_portable_python(outdir: str) -> str:
    """Put a relocatable Windows Python, with the viewer packages, in outdir/python.

    A venv cannot be zipped to another PC: its pyvenv.cfg points at this machine.
    The embeddable runtime has no absolute paths, so the zip runs elsewhere.
    """
    runtime = os.path.join(outdir, "python")
    exe = os.path.join(runtime, "python.exe")
    if os.path.isfile(exe):
        probe = subprocess.run(
            [exe, "-c", "import dash, plotly, pandas, numpy"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return exe
    if os.path.isdir(runtime):
        shutil.rmtree(runtime, ignore_errors=True)
    cache = os.path.join(os.environ.get("LOCALAPPDATA", outdir), "NASA-porosity")
    embed_zip = os.path.join(cache, "python-3.11.9-embed-amd64.zip")
    get_pip = os.path.join(cache, "get-pip.py")
    _download(_EMBED_URL, embed_zip)
    _download(_GET_PIP_URL, get_pip)
    os.makedirs(runtime, exist_ok=True)
    with zipfile.ZipFile(embed_zip) as zf:
        zf.extractall(runtime)
    pth = next(
        os.path.join(runtime, name)
        for name in os.listdir(runtime)
        if name.endswith("._pth")
    )
    with open(pth, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if "Lib\\site-packages" not in lines:
        lines.append("Lib\\site-packages")
    lines.append("import site")
    with open(pth, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    shutil.copy2(get_pip, os.path.join(runtime, "get-pip.py"))
    print("Installing viewer packages into the portable Python...", flush=True)
    subprocess.check_call([exe, os.path.join(runtime, "get-pip.py")], cwd=runtime)
    subprocess.check_call([exe, "-m", "pip", "install", *_VIEWER_PACKAGES], cwd=runtime)
    subprocess.check_call([exe, "-c", "import dash, plotly, pandas, numpy"], cwd=runtime)
    return exe


def publish_viewer(
    outdir: str,
    *,
    theta_mode: str = "per_layer",
    lambda_mean: float = LAMBDA_MEAN,
) -> None:
    """Copy the standalone GUI, a portable Python, and a launcher into the share folder."""
    os.makedirs(outdir, exist_ok=True)
    src = os.path.join(os.path.dirname(__file__), "share_viewer.py")
    shutil.copy2(src, os.path.join(outdir, "view_results.py"))
    with open(os.path.join(outdir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_VIEWER_PACKAGES) + "\n")
    bundle_portable_python(outdir)
    bat_name = "Open results GUI.bat"
    old_bat = os.path.join(outdir, "run_viewer.bat")
    if os.path.isfile(old_bat):
        os.remove(old_bat)
    with open(os.path.join(outdir, bat_name), "w", encoding="utf-8") as f:
        f.write(
            "@echo off\r\n"
            "cd /d \"%~dp0\"\r\n"
            "if not exist \"python\\python.exe\" (\r\n"
            "  echo The python folder is missing.\r\n"
            "  echo Unzip the whole results folder, then run this file from inside it.\r\n"
            "  pause\r\n"
            "  exit /b 1\r\n"
            ")\r\n"
            "echo Starting the results GUI at http://127.0.0.1:8055\r\n"
            "echo Leave this window open while you use it. Close the window to stop the GUI.\r\n"
            "start \"\" cmd /c \"ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:8055/\"\r\n"
            "\"python\\python.exe\" \"view_results.py\"\r\n"
            "if errorlevel 1 pause\r\n"
            "pause\r\n"
        )
    write_conditions_table(outdir, lambda_mean=lambda_mean)
    _readme(os.path.join(outdir, "README.txt"), outdir, theta_mode=theta_mode, lambda_mean=lambda_mean)


def simplify_share_tree(outdir: str) -> None:
    """Turn an older export (numbered files + run/) into the four-file layout."""
    keep = {
        "melted_build.png",
        "surface_sweep.gif",
        "volume_sweep.gif",
        "all_pore_sources.png",
        "pore_size_shape.png",
        "pores.csv",
        "view3d.npz",
        "process.json",
        "melted_build.png.caption.txt",
    }
    rename = {
        "01_melted_build.png": "melted_build.png",
        "02_pore_size_shape.png": "pore_size_shape.png",
        "04_all_pores_location_shape.csv": "pores.csv",
        "03_raptor_viewer_view3d.npz": "view3d.npz",
    }
    for name in os.listdir(outdir):
        folder = os.path.join(outdir, name)
        if not (name.startswith("V_") and os.path.isdir(folder)):
            continue
        for old, new in rename.items():
            src = os.path.join(folder, old)
            dst = os.path.join(folder, new)
            if os.path.isfile(src) and not os.path.isfile(dst):
                os.replace(src, dst)
        run_dir = os.path.join(folder, "run")
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
        for item in os.listdir(folder):
            if item not in keep:
                path = os.path.join(folder, item)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
    for extra in ("open_raptor_viewer.py", "sweep.pid", "sweep_log.txt"):
        path = os.path.join(outdir, extra)
        if os.path.isfile(path):
            os.remove(path)
    publish_viewer(outdir)


def run_sweep(outdir: str, *, theta_mode: str = "per_layer", lambda_mean: float = LAMBDA_MEAN) -> None:
    os.makedirs(outdir, exist_ok=True)
    publish_viewer(outdir, theta_mode=theta_mode, lambda_mean=lambda_mean)
    log_path = os.path.join(outdir, "sweep_log.txt")

    def log(msg: str):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"Sweep folder: {outdir}")
    log(f"Melt-pool spectrum: {theta_mode}")
    log(f"Lambda: {float(lambda_mean):g} on the mean width, depth, and cap")
    log(f"Velocities: {VELOCITIES_MM_S[0]}–{VELOCITIES_MM_S[-1]} mm/s step 50 ({len(VELOCITIES_MM_S)} runs)")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYVISTA_OFF_SCREEN": "true"}
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    for i, velocity in enumerate(VELOCITIES_MM_S, start=1):
        label = f"V_{int(velocity)}"
        dest = os.path.join(outdir, label)
        run_dir = os.path.join(dest, "run")
        os.makedirs(run_dir, exist_ok=True)
        cfg = build_config(float(velocity), theta_mode=theta_mode, lambda_mean=lambda_mean)
        cfg_path = os.path.join(run_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        log(f"[{i}/{len(VELOCITIES_MM_S)}] {label} starting")
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "integrated_gui.combined_worker",
                "--config",
                cfg_path,
                "--outdir",
                run_dir,
            ],
            cwd=ROOT,
            env=env,
            creationflags=creationflags,
        )
        status_path = os.path.join(run_dir, "status.json")
        ok = proc.returncode == 0 and os.path.isfile(status_path)
        if ok:
            try:
                ok = bool(_load_json(status_path).get("ok"))
            except Exception:
                ok = False
        if not ok:
            log(f"[{i}/{len(VELOCITIES_MM_S)}] {label} FAILED (exit {proc.returncode}). See {run_dir}")
            continue
        try:
            export_share(
                run_dir,
                dest,
                float(velocity),
                lambda_mean=lambda_mean,
                theta_mode=theta_mode,
            )
            log(f"[{i}/{len(VELOCITIES_MM_S)}] {label} exported")
        except Exception as exc:
            log(f"[{i}/{len(VELOCITIES_MM_S)}] {label} ran but export failed: {exc}")
    write_conditions_table(outdir, lambda_mean=lambda_mean)
    write_predictions(outdir, lambda_mean=lambda_mean)
    _readme(os.path.join(outdir, "README.txt"), outdir, theta_mode=theta_mode, lambda_mean=lambda_mean)
    log("Wrote hatch, melt-pool width, and CMU vs Raptor pores/mm³")
    log("Sweep finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="")
    parser.add_argument(
        "--theta-mode",
        default="per_layer",
        choices=("per_layer", "averaged", "fixed"),
        help="averaged uses one melt pool from the mean spectrum of every orientation",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_mean",
        type=float,
        default=LAMBDA_MEAN,
        help="shift of the mean width, depth, and cap (1.04 or 1.14)",
    )
    args = parser.parse_args()
    outdir = args.outdir or os.path.join(desktop_dir(), "Raptor_5x5x14_V1300-1700_flucWsMTs")
    run_sweep(os.path.abspath(outdir), theta_mode=args.theta_mode, lambda_mean=args.lambda_mean)


if __name__ == "__main__":
    main()
