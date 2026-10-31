"""Interactive PyVista visualization for combined plate pore clouds."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyvista as pv

from .combined_3d import PartGeometry, gas_flow_arrow_geometry, GAS_FLOW_LABELS

FAMILY_COLORS_RGB = {
    "lof": (11, 61, 145),
    "keyhole": (252, 61, 33),
    "gas": (27, 127, 78),
    "spatter": (155, 89, 182),
    "cmu": (90, 106, 122),
}

FAMILY_COLORS_HEX = {
    "pore": "#0B3D91",
    "edge": "#FC3D21",
    "lof": "#0B3D91",
    "keyhole": "#FC3D21",
    "gas": "#1B7F4E",
    "spatter": "#9B59B6",
    "cmu": "#5A6A7A",
}

FAMILY_DISPLAY = {
    "pore": "Pore",
    "edge": "Edge (scan on/off)",
    "lof": "LoF",
    "keyhole": "Keyhole",
    "gas": "Gas",
    "spatter": "Spatter",
    "cmu": "CMU",
}


def _rgb(src: str):
    return FAMILY_COLORS_RGB.get(str(src).lower(), (51, 51, 51))


def _hex(src: str) -> str:
    return FAMILY_COLORS_HEX.get(str(src).lower(), "#333333")


def _display_name(src: str) -> str:
    key = str(src).lower()
    return FAMILY_DISPLAY.get(key, str(src))


def _add_part_geometry(plotter: pv.Plotter, geom: PartGeometry) -> None:
    if geom.shape == "cylinder":
        r = geom.diameter_mm / 2.0
        cyl = pv.Cylinder(
            center=(geom.plate_x_mm, geom.plate_y_mm, geom.height_mm / 2.0),
            direction=(0, 0, 1),
            radius=r,
            height=geom.height_mm,
            resolution=48,
        )
        plotter.add_mesh(cyl, style="wireframe", color="#5A6A7A", line_width=2, opacity=0.95)
    else:
        bounds = [
            geom.plate_x_mm - geom.length_mm / 2,
            geom.plate_x_mm + geom.length_mm / 2,
            geom.plate_y_mm - geom.width_mm / 2,
            geom.plate_y_mm + geom.width_mm / 2,
            0.0,
            geom.height_mm,
        ]
        plotter.add_mesh(pv.Box(bounds=bounds), style="wireframe", color="#5A6A7A", line_width=2)


def _add_plate_geometry(
    plotter: pv.Plotter,
    layout: Dict[str, Any],
    cubes: Optional[pd.DataFrame] = None,
) -> None:
    pw = float(layout.get("plate_width_mm", 120))
    ph = float(layout.get("plate_height_mm", 120))
    height = float(layout.get("cube_height_mm", layout.get("cube_size_mm", 8)))
    # plate outline as thin box on z=0
    plate = pv.Box(bounds=[0, pw, 0, ph, -0.05, 0.05])
    plotter.add_mesh(plate, color="#061F4A", opacity=0.25)

    if cubes is not None and len(cubes):
        shape_default = str(layout.get("build_shape") or layout.get("plate_mode") or "cube").lower()
        for _, row in cubes.iterrows():
            cx, cy = float(row["cx_mm"]), float(row["cy_mm"])
            size = float(row.get("size_mm", layout.get("cube_size_mm", 8)))
            h = float(row.get("height_mm", height))
            length = float(row["length_mm"]) if "length_mm" in row and pd.notna(row.get("length_mm")) else size
            width = float(row["width_mm"]) if "width_mm" in row and pd.notna(row.get("width_mm")) else size
            shape = str(row.get("shape", shape_default) or shape_default).lower()
            if shape == "stl":
                # AABB only as fallback when fitted STL is missing
                half_x, half_y = length / 2.0, width / 2.0
                box = pv.Box(bounds=[cx - half_x, cx + half_x, cy - half_y, cy + half_y, 0.0, h])
                plotter.add_mesh(box, style="wireframe", color="#9AA3AD", line_width=2, opacity=0.35)
            else:
                half_x, half_y = length / 2.0, width / 2.0
                box = pv.Box(bounds=[cx - half_x, cx + half_x, cy - half_y, cy + half_y, 0.0, h])
                plotter.add_mesh(box, style="wireframe", color="#5A6A7A", line_width=3, opacity=0.9)

    # STL mode: solid translucent surface matching the website Plotly Mesh3d style
    if str(layout.get("plate_mode", "")).lower() == "stl":
        stl_path = os.path.join(os.path.dirname(__file__), "assets", "pyvista", "latest_fitted.stl")
        if os.path.exists(stl_path):
            try:
                mesh = pv.read(stl_path)
                try:
                    try:
                        mesh = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
                    except TypeError:
                        mesh = mesh.extract_surface().triangulate().clean()
                except Exception:
                    try:
                        mesh = mesh.triangulate()
                    except Exception:
                        pass
                # Light decimate for interactivity (same spirit as website preview)
                try:
                    n_cells = int(getattr(mesh, "n_cells", 0) or 0)
                    if n_cells > 8000:
                        mesh = mesh.decimate_pro(
                            1.0 - max(0.05, 8000.0 / float(n_cells)),
                            preserve_topology=True,
                        )
                except Exception:
                    pass
                plotter.add_mesh(
                    mesh,
                    color="#9AA3AD",
                    opacity=0.22,
                    smooth_shading=False,
                    show_edges=False,
                    name="STL geometry",
                )
            except Exception:
                pass

    flow = str(layout.get("gas_flow") or "+y")
    _add_gas_flow_arrow(plotter, layout, flow)


def _add_gas_flow_arrow(plotter: pv.Plotter, layout: Dict[str, Any], gas_flow: str) -> None:
    pw = float(layout.get("plate_width_mm", 120))
    ph = float(layout.get("plate_height_mm", 120))
    height = float(layout.get("cube_height_mm", layout.get("cube_size_mm", 8)))
    z = max(height * 1.15, 4.0)
    arrow = gas_flow_arrow_geometry(pw, ph, gas_flow, z_mm=z)
    (x0, y0, z0), (x1, y1, z1) = arrow["shaft"]
    dx, dy = arrow["direction"]
    length = float(np.hypot(x1 - x0, y1 - y0))
    center = ((x0 + x1) / 2, (y0 + y1) / 2, z)
    # shaft as thin cylinder along flow
    shaft = pv.Cylinder(
        center=center,
        direction=(dx, dy, 0.0),
        radius=max(0.6, 0.008 * min(pw, ph)),
        height=length,
        resolution=24,
    )
    tip = pv.Cone(
        center=(x1, y1, z),
        direction=(dx, dy, 0.0),
        height=max(4.0, 0.08 * min(pw, ph)),
        radius=max(1.8, 0.025 * min(pw, ph)),
        resolution=24,
    )
    plotter.add_mesh(shaft, color="#E67E22", opacity=0.95)
    plotter.add_mesh(tip, color="#E67E22", opacity=0.95)
    lx, ly, lz = arrow["label_xyz"]
    plotter.add_point_labels(
        [[lx, ly, lz]],
        [arrow["label"]],
        font_size=14,
        text_color="#E67E22",
        point_size=0,
        shape=None,
        always_visible=True,
    )


def _add_pores(plotter: pv.Plotter, combined: pd.DataFrame, max_points: int) -> list:
    """Add colored pore clouds. Returns legend entries [[label, color], ...]."""
    legend_entries = []
    if combined is None or len(combined) == 0:
        plotter.add_text("No pores", font_size=12, color="black")
        return legend_entries

    df = combined.copy()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)

    sources = (
        df["pore_source"].astype(str).str.lower()
        if "pore_source" in df.columns
        else df["pore_family"].astype(str).str.lower()
    )

    # Prefer a stable order: LoF, keyhole, gas, spatter, then others
    preferred = ["pore", "edge", "lof", "keyhole", "gas", "spatter"]
    ordered = [s for s in preferred if s in set(sources)]
    ordered += sorted(s for s in sources.unique() if s not in preferred)

    for src in ordered:
        sub = df.loc[sources == src]
        pts = sub[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
        if len(pts) == 0:
            continue
        color = _hex(src)
        label = f"{_display_name(src)} (n={len(sub)})"
        cloud = pv.PolyData(pts)
        # Solid color (not RGB scalars) so the PyVista legend matches the spheres
        plotter.add_mesh(
            cloud,
            color=color,
            point_size=12,
            render_points_as_spheres=True,
            label=label,
        )
        legend_entries.append([label, color])
    return legend_entries


def build_plotter(
    combined: pd.DataFrame,
    geom=None,
    *,
    layout: Optional[Dict[str, Any]] = None,
    cubes: Optional[pd.DataFrame] = None,
    off_screen: bool = True,
    max_points: int = 5000,
    window_size=(1100, 850),
) -> pv.Plotter:
    pv.set_plot_theme("document")
    try:
        # Use GPU MSAA when the OpenGL backend supports it
        pv.global_theme.multi_samples = 8
    except Exception:
        pass
    plotter = pv.Plotter(off_screen=off_screen, window_size=list(window_size))
    plotter.set_background("white")
    if layout is not None or (isinstance(geom, dict) and geom.get("mode") == "plate"):
        lay = layout or geom
        _add_plate_geometry(plotter, lay, cubes)
        flow = str(lay.get("gas_flow") or "+y")
        flow_label = GAS_FLOW_LABELS.get(flow, f"Gas flow {flow}")
    elif isinstance(geom, PartGeometry):
        _add_part_geometry(plotter, geom)
        flow_label = None
    else:
        flow_label = None

    legend_entries = _add_pores(plotter, combined, max_points=max_points)
    if flow_label:
        legend_entries.append([flow_label, "#E67E22"])

    if legend_entries:
        # Explicit color list — avoids the all-blue RGB-scalar legend bug
        plotter.add_legend(
            legend_entries,
            bcolor=(1, 1, 1),
            border=True,
            size=(0.28, 0.28),
            loc="upper right",
            face="circle",
        )

    # On-screen color key (always readable)
    key_lines = ["Pore colors:"]
    for label, color in legend_entries:
        if label.startswith("Gas flow"):
            continue
        key_lines.append(f"  {label}")
    if flow_label:
        key_lines.append(f"  {flow_label} = orange arrow")
    plotter.add_text("\n".join(key_lines), font_size=9, color="black", position="upper_left")

    plotter.add_axes(line_width=2)
    plotter.show_bounds(grid="back", location="outer", all_edges=True, font_size=10)
    plotter.reset_camera()
    plotter.camera.zoom(1.05)
    return plotter


def export_interactive_html(
    combined: pd.DataFrame,
    geom=None,
    out_path: str = "",
    max_points: int = 5000,
    layout: Optional[Dict[str, Any]] = None,
    cubes: Optional[pd.DataFrame] = None,
) -> str:
    plotter = build_plotter(
        combined, geom, layout=layout, cubes=cubes, off_screen=True, max_points=max_points
    )
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plotter.export_html(out_path)
    finally:
        plotter.close()
    return out_path


def open_desktop_viewer(
    combined: pd.DataFrame,
    geom=None,
    max_points: int = 5000,
    layout: Optional[Dict[str, Any]] = None,
    cubes: Optional[pd.DataFrame] = None,
) -> None:
    plotter = build_plotter(
        combined,
        geom,
        layout=layout,
        cubes=cubes,
        off_screen=False,
        max_points=max_points,
        window_size=(1200, 900),
    )
    plotter.show(title="Combined plate pore field — drag to rotate, scroll to zoom")
