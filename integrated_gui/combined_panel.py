"""Combined 3D: data-driven plate pores (CMU ρ × V × NU/mean)."""
from __future__ import annotations

import json
import os
import shutil
import threading
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from .headless import configure_headless_plotting

configure_headless_plotting()

import dash
from dash import ClientsideFunction, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import empty_figure_placeholder, fig_to_img
from . import meltpool_library as mpl
from .nu_cache import cache_status, run_prefit_subprocess
from .combined_3d import gas_flow_arrow_geometry, GAS_FLOW_LABELS, make_plate_layout
from .scan_vectors import (
    ESD_COL,
    vectors_for_builds,
    vectors_for_stl_mesh,
    sample_edge_pores,
    make_scan_preview_figure,
)
from .stl_slicer_scan import format_scan_stats as _format_scan_stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CMU_DIR = os.path.join(ROOT, "CMU team")
PYVISTA_DIR = os.path.join(os.path.dirname(__file__), "assets", "pyvista")
RAPTOR_ROI_PATH = os.path.join(PYVISTA_DIR, "raptor_roi_latest.json")
WORK_DIR = os.path.join(os.path.dirname(__file__), "assets", "combined_work")

EXAMPLE_PARAM_PATH = os.path.join(CMU_DIR, "input_parameters", "params.json")
MATERIAL_PATH = os.path.join(CMU_DIR, "material_models", "test_ti64_rosenthal.json")
# P&W challenge STL lives next to the Python code folder under the NASA project root
DEFAULT_STL_PATH = os.path.join(
    os.path.dirname(ROOT),
    "P&W Challenge Part",
    "P&W Challenge Part",
    "Geometry",
    "PW_Generic_Part.stl",
)

_LAST_CSV: Optional[str] = None  # legacy in-memory CSV (small runs only)
_LAST_CSV_PATH: Optional[str] = None  # preferred: path to pores.csv on disk
_LAST_SUMMARY: Optional[dict] = None
_LAST_RAPTOR_DIR: Optional[str] = None  # outdir/raptor_rve when Combined cuboid RVE ran

_PLOT_CSV_COLS = (
    "x_mm",
    "y_mm",
    "z_mm",
    "pore_source",
    "pore_family",
    "radius_mm",
)

FAMILY_COLORS = {
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

GAS_FLOW_OPTIONS = [
    {"label": "+X — along plate width (right)", "value": "+x"},
    {"label": "−X — along plate width (left)", "value": "-x"},
    {"label": "+Y — along strips (up) · coupon default", "value": "+y"},
    {"label": "−Y — along strips (down)", "value": "-y"},
]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_SPATTER_DESC = (
    "Spatter particles follow each hatch (~30 cm); map is zoomed "
    "to the sample so the moving laser path is visible. "
    "Right panel: accumulated landings near the sample."
)


_STAGE_LABELS = {
    "raptor": "Raptor",
    "scan_and_pores": "scan + pore placement",
    "spatter_animation": "spatter animation",
    "export": "export",
}


def _fmt_seconds(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m} min {sec:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def _format_run_time(timing: dict) -> str:
    if not timing or "total" not in timing:
        return ""
    parts = [
        f"{_STAGE_LABELS.get(k, k)} {_fmt_seconds(v)}"
        for k, v in timing.items()
        if k != "total"
    ]
    return f"Run time: {_fmt_seconds(timing['total'])}" + (
        f" ({' · '.join(parts)})" if parts else ""
    )


def _format_expected_pores(summary: dict) -> str:
    n_full = summary.get("base_count_rho_v")
    if not n_full:
        return ""
    rho = float(summary.get("cmu_total_density") or 0)
    vol = float(summary.get("part_volume_mm3") or 0)
    n_edge = int(summary.get("n_edge_pores") or 0)
    n_bulk = int(summary.get("n_bulk_pores") or 0)
    txt = (
        f"Pores for the build volume: ρ×V = {rho:.2f}/mm³ × {vol:,.1f} mm³ "
        f"≈ {float(n_full):,.0f} bulk pores"
        + (f" + {n_edge:,} edge pores" if n_edge else "")
    )
    cap = int(summary.get("max_pores") or 0)
    if cap > 0 and n_bulk < round(float(n_full)):
        txt += (
            f" — only {n_bulk:,} generated because Max pores = {cap:,}; "
            "set Max pores to 0 to generate all of them"
        )
    return txt + "."


def _format_process(proc: dict) -> str:
    bits = []
    size = proc.get("part_size_mm")
    if size:
        bits.append("Part " + "×".join(f"{float(v):g}" for v in size) + " mm")
    bits.append(f"P={float(proc.get('power_W', 0)):g} W")
    bits.append(f"v={float(proc.get('scan_speed_mm_s', 0)):g} mm/s")
    bits.append(f"hatch={float(proc.get('hatch_spacing_um', 0)):g} µm")
    bits.append(f"layer={float(proc.get('layer_thickness_um', 0)):g} µm")
    if proc.get("volumetric_energy_density_J_mm3"):
        bits.append(f"VED={float(proc['volumetric_energy_density_J_mm3']):.1f} J/mm³")
    strat = proc.get("scan_strategy")
    if strat:
        s = f"strategy={strat}"
        if strat == "stripes":
            s += f" ({float(proc.get('stripe_width_mm', 10)):g} mm)"
        s += f", rotation {float(proc.get('layer_rotation_deg', 67)):g}°"
        bits.append(s)
    if proc.get("raptor_used") and proc.get("raptor_voxel_um"):
        r = f"Raptor voxel={float(proc['raptor_voxel_um']):.1f} µm"
        if proc.get("raptor_layers_scanned"):
            r += f", {int(proc['raptor_layers_scanned'])} layers"
        bits.append(r)
    return "Process: " + " · ".join(bits) + " (saved as process_parameters.csv/.json)"


def _lof_shape_children(ls: dict, csv_path: str, pore_types=None) -> list:
    """Size, sphericity, and aspect-ratio histograms for every pore type in the Raptor box.

    Raptor LoF uses the measured voxel shape. Keyhole, gas, and edge pores are placed as
    spheres, so they sit at sphericity 1 and aspect ratio 1 with their equivalent diameter.
    """
    from .raptor_view import PORE_TYPE_STYLE

    if (not ls or not ls.get("n_pores")) and (pore_types is None or not len(pore_types)):
        return []
    lines = []
    if ls and ls.get("n_pores"):
        lines.append(
            f"{ls['n_pores']:,} LoF pores · equivalent sphere diameter (ESD) mean "
            f"{ls['esd_mean_um']:.1f} µm, median {ls['esd_median_um']:.1f} µm, "
            f"D10–D90 {ls['esd_d10_um']:.1f}–{ls['esd_d90_um']:.1f} µm, max {ls['esd_max_um']:.1f} µm "
            f"(volume-weighted mean {ls['volume_weighted_mean_esd_um']:.1f} µm)."
        )
        lines.append(
            f"Elongation: mean longest axis {ls['major_axis_mean_um']:.0f} µm, "
            f"aspect ratio (longest/shortest) median {ls['aspect_ratio_median']:.1f}."
        )
        if "sphericity_median" in ls:
            lines.append(
                f"Sphericity (1 = perfect sphere): mean {ls['sphericity_mean']:.2f}, median "
                f"{ls['sphericity_median']:.2f}, D10–D90 {ls['sphericity_d10']:.2f}–"
                f"{ls['sphericity_d90']:.2f}. Measured on the {ls['n_shape_pores']:,} LoF pores with at "
                f"least {ls['shape_min_voxels']} voxels; smaller LoF pores are too few voxels to have a "
                "meaningful surface."
            )
    lines.append(
        "Keyhole, gas, and edge pores are drawn as spheres (sphericity 1, aspect ratio 1) "
        "using their equivalent diameter. Each type is normalised to its own count."
    )
    children = [
        html.H6("Pore size & shape", className="mt-3 mb-1"),
        *[html.Div(t, className="small") for t in lines],
    ]
    series = []
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if len(df):
                label, color = PORE_TYPE_STYLE["raptor_lof"]
                series.append(
                    (
                        "raptor_lof",
                        f"{label} ({len(df):,})",
                        color,
                        df["esd_um"].to_numpy(float),
                        df["sphericity"].dropna().to_numpy(float),
                        df["aspect_ratio"].to_numpy(float),
                    )
                )
        except Exception:
            series = []
    if pore_types is not None and len(pore_types):
        kinds = pore_types["type"].astype(str)
        for key, (label, color) in PORE_TYPE_STYLE.items():
            if key == "raptor_lof":
                continue
            sub = pore_types.loc[kinds == key]
            if not len(sub):
                continue
            esd = sub["esd_um"].to_numpy(dtype=float)
            esd = esd[np.isfinite(esd)]
            if not len(esd):
                continue
            series.append(
                (
                    key,
                    f"{label} ({len(esd):,})",
                    color,
                    esd,
                    np.ones(len(esd)),
                    np.ones(len(esd)),
                )
            )
    if not series:
        return children
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Size distribution (ESD, µm)",
                "Sphericity",
                "Aspect ratio (longest/shortest)",
            ),
        )
        for key, name, color, esd, sph, aspect in series:
            for col, vals in enumerate((esd, sph, aspect), start=1):
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if not len(vals):
                    continue
                fig.add_trace(
                    go.Histogram(
                        x=vals,
                        nbinsx=40,
                        histnorm="percent",
                        marker_color=color,
                        opacity=0.55,
                        name=name,
                        legendgroup=key,
                        showlegend=col == 1,
                    ),
                    1,
                    col,
                )
        fig.update_yaxes(title_text="% of that type", row=1, col=1)
        fig.update_layout(
            height=380,
            barmode="overlay",
            margin=dict(l=48, r=16, t=48, b=72),
            bargap=0.05,
            legend=dict(orientation="h", y=-0.28),
        )
        children.append(
            html.Div(
                [
                    dbc.Button(
                        "Copy image to clipboard",
                        id="comb-copy-pore-shape",
                        color="secondary",
                        outline=True,
                        size="sm",
                        className="me-2",
                    ),
                    html.Small(id="comb-copy-pore-shape-status", className="text-muted"),
                ],
                className="mb-2 mt-2",
            )
        )
        children.append(
            dcc.Graph(
                id="comb-pore-shape-fig",
                figure=fig,
                config={"displaylogo": False},
            )
        )
    except Exception:
        pass
    return children


def _pore_types_children(pt, z_range=None) -> list:
    """Size + spatial (X, Y, height) histograms per pore type inside the Raptor box."""
    if pt is None or not len(pt):
        return []
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .raptor_view import PORE_TYPE_STYLE

    kinds = pt["type"].astype(str)
    present = [k for k in PORE_TYPE_STYLE if (kinds == k).any()]
    counts = ", ".join(f"{PORE_TYPE_STYLE[k][0]} {int((kinds == k).sum()):,}" for k in present)
    note = (
        f"Pores inside the Raptor box: {counts}. Raptor LoF replaces the CMU LoF; keyhole and "
        "gas come from the CMU model and edge pores from scan on/off points."
    )
    if z_range is not None:
        note += (
            f" The Raptor box is thin, so CMU and edge pores are taken from a 2 mm slab "
            f"(z {z_range[0]:.2f}–{z_range[1]:.2f} mm) around it."
        )
    note += " Each type is normalised to its own count (% of that type), so shapes compare directly."
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "Size distribution (ESD, µm)",
            "X position (mm)",
            "Y position (mm)",
            "Height Z (mm)",
        ),
    )
    for k in present:
        label, color = PORE_TYPE_STYLE[k]
        sub = pt.loc[kinds == k]
        for col, name in enumerate(("esd_um", "x_mm", "y_mm", "z_mm"), start=1):
            vals = sub[name].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            fig.add_trace(
                go.Histogram(
                    x=vals,
                    nbinsx=40,
                    histnorm="percent",
                    marker_color=color,
                    opacity=0.6,
                    name=f"{label} ({len(sub):,})",
                    legendgroup=k,
                    showlegend=col == 1,
                ),
                1,
                col,
            )
    fig.update_yaxes(title_text="% of that type", row=1, col=1)
    fig.update_layout(
        height=320,
        barmode="overlay",
        margin=dict(l=40, r=10, t=40, b=30),
        bargap=0.05,
        legend=dict(orientation="h", y=-0.18),
    )
    return [
        html.H6("Pore types: size and location", className="mt-3 mb-1"),
        html.Div(note, className="small"),
        dcc.Graph(figure=fig, config={"displaylogo": False}),
    ]


def _spatter_children(spat_dir: str, desc: str = _SPATTER_DESC) -> list:
    """Heading + caption + GIF (or PNG fallback) for a spatter animation folder."""
    import base64

    spat_gif = os.path.join(spat_dir, "spatter_build.gif")
    spat_png = os.path.join(spat_dir, "spatter_accum.png")
    src_path, mime, alt = None, "", ""
    if os.path.isfile(spat_gif):
        src_path, mime, alt = spat_gif, "image/gif", "Spatter build animation"
    elif os.path.isfile(spat_png):
        src_path, mime, alt = spat_png, "image/png", "Accumulated spatter"
    if src_path is None:
        return []
    with open(src_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return [
        html.H6("Spatter build animation", className="mt-3 mb-1"),
        html.Small(desc, className="text-muted d-block mb-2", id="comb-spatter-desc"),
        html.Img(
            src=f"data:{mime};base64,{b64}",
            className="result-figure",
            style={"width": "100%", "height": "auto", "objectFit": "contain"},
            alt=alt,
        ),
    ]


# Plotly WebGL needs NaN (not None) to break line segments reliably.
_LINE_BREAK = float("nan")

# In-browser preview caps (full N is in CSV / PyVista).
_PREVIEW_PORES_DEFAULT = 1500
_PREVIEW_PORES_MIN = 200
_PREVIEW_PORES_MAX = 20000


def _cuboid_edge_polyline(
    cx, cy, size, height, length=None, width_mm=None
) -> tuple[list, list, list]:
    """One cuboid's 12 edges as a single polyline with NaN breaks (Plotly)."""
    half_x = (length if length is not None else size) / 2.0
    half_y = (width_mm if width_mm is not None else size) / 2.0
    corners = [
        (cx - half_x, cy - half_y, 0.0),
        (cx + half_x, cy - half_y, 0.0),
        (cx + half_x, cy + half_y, 0.0),
        (cx - half_x, cy + half_y, 0.0),
        (cx - half_x, cy - half_y, height),
        (cx + half_x, cy - half_y, height),
        (cx + half_x, cy + half_y, height),
        (cx - half_x, cy + half_y, height),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs: list = []
    ys: list = []
    zs: list = []
    for a, b in edges:
        xs.extend([corners[a][0], corners[b][0], _LINE_BREAK])
        ys.extend([corners[a][1], corners[b][1], _LINE_BREAK])
        zs.extend([corners[a][2], corners[b][2], _LINE_BREAK])
    return xs, ys, zs


def _strip_outline_polyline(cubes: pd.DataFrame, height: float) -> tuple[list, list, list]:
    """Thick outline of each coupon strip AABB so the 4×(19×3) grid reads clearly."""
    xs: list = []
    ys: list = []
    zs: list = []
    if cubes is None or len(cubes) == 0 or "strip" not in cubes.columns:
        return xs, ys, zs
    for _, g in cubes.groupby("strip", sort=True):
        hx = 0.5 * g["length_mm"].to_numpy(dtype=float)
        hy = 0.5 * g["width_mm"].to_numpy(dtype=float)
        cx = g["cx_mm"].to_numpy(dtype=float)
        cy = g["cy_mm"].to_numpy(dtype=float)
        x0, x1 = float(np.min(cx - hx)), float(np.max(cx + hx))
        y0, y1 = float(np.min(cy - hy)), float(np.max(cy + hy))
        h = float(height)
        # bottom + top rectangle + verticals
        for z in (0.0, h):
            xs.extend([x0, x1, x1, x0, x0, _LINE_BREAK])
            ys.extend([y0, y0, y1, y1, y0, _LINE_BREAK])
            zs.extend([z, z, z, z, z, _LINE_BREAK])
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            xs.extend([x, x, _LINE_BREAK])
            ys.extend([y, y, _LINE_BREAK])
            zs.extend([0.0, h, _LINE_BREAK])
    return xs, ys, zs


def _cube_wireframe(cx, cy, size, height, color="#5A6A7A", width=3.0, opacity=0.85, length=None, width_mm=None):
    """Legacy: one Scatter3d per cuboid (prefer batched path in _make_plotly_3d)."""
    xs, ys, zs = _cuboid_edge_polyline(cx, cy, size, height, length=length, width_mm=width_mm)
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            showlegend=False,
            hoverinfo="skip",
        )
    ]


def _build_wireframe(cx, cy, size, height, shape="cube", length=None, width_mm=None, **kwargs):
    return _cube_wireframe(cx, cy, size, height, length=length, width_mm=width_mm, **kwargs)


def _add_gas_flow_arrow_plotly(fig: go.Figure, layout: dict, gas_flow: str) -> None:
    """Orange arrow + legend entry for chamber gas distribution direction."""
    pw = float(layout.get("plate_width_mm", 120))
    ph = float(layout.get("plate_height_mm", 120))
    height = float(layout.get("cube_height_mm", layout.get("cube_size_mm", 8)))
    # float slightly above the cubes so the arrow stays visible
    z = max(height * 1.15, 4.0)
    arrow = gas_flow_arrow_geometry(pw, ph, gas_flow, z_mm=z)
    color = "#E67E22"
    sx0, sy0, sz0 = arrow["shaft"][0]
    sx1, sy1, sz1 = arrow["shaft"][1]
    hx = [p[0] for p in arrow["head"]]
    hy = [p[1] for p in arrow["head"]]
    hz = [p[2] for p in arrow["head"]]

    fig.add_trace(
        go.Scatter3d(
            x=[sx0, sx1],
            y=[sy0, sy1],
            z=[sz0, sz1],
            mode="lines",
            line=dict(color=color, width=10),
            name=arrow["label"],
            hovertemplate=f"{arrow['label']}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=hx,
            y=hy,
            z=hz,
            mode="lines",
            line=dict(color=color, width=10),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # tip marker for clarity
    fig.add_trace(
        go.Scatter3d(
            x=[sx1],
            y=[sy1],
            z=[sz1],
            mode="markers+text",
            marker=dict(size=6, color=color, symbol="diamond"),
            text=[arrow["label"]],
            textposition="top center",
            textfont=dict(size=12, color=color, family="IBM Plex Sans, sans-serif"),
            showlegend=False,
            hovertemplate=f"{arrow['label']}<extra></extra>",
        )
    )


def _add_stl_mesh_plotly(fig: go.Figure, fitted_stl_path: str, max_faces: int = 6000) -> None:
    """Add a decimated translucent STL surface (fast enough for laptop Plotly)."""
    import pyvista as pv

    mesh = pv.read(fitted_stl_path)
    try:
        mesh = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
    except TypeError:
        # Older PyVista without algorithm=
        try:
            mesh = mesh.extract_surface().triangulate().clean()
        except Exception:
            try:
                mesh = mesh.triangulate()
            except Exception:
                pass
    except Exception:
        try:
            mesh = mesh.triangulate()
        except Exception:
            pass

    # Decimate for browser/CPU responsiveness (physics mesh is unchanged on disk)
    try:
        n_cells = int(getattr(mesh, "n_cells", 0) or 0)
        if n_cells > max_faces:
            target = max(0.02, float(max_faces) / float(n_cells))
            mesh = mesh.decimate_pro(1.0 - target, preserve_topology=True)
    except Exception:
        try:
            if int(getattr(mesh, "n_cells", 0) or 0) > max_faces:
                mesh = mesh.decimate(1.0 - float(max_faces) / float(mesh.n_cells))
        except Exception:
            pass

    pts = np.asarray(mesh.points, dtype=float)
    faces = np.asarray(mesh.faces)
    if len(pts) == 0 or len(faces) == 0:
        return

    # PyVista faces format: [3, i, j, k, 3, i, j, k, ...]
    if faces.size % 4 != 0:
        return
    tri = faces.reshape(-1, 4)
    i = tri[:, 1]
    j = tri[:, 2]
    k = tri[:, 3]

    fig.add_trace(
        go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=i,
            j=j,
            k=k,
            color="#9AA3AD",
            opacity=0.22,
            flatshading=True,
            name="STL geometry",
            hoverinfo="skip",
            showscale=False,
        )
    )


def _make_plotly_3d(
    combined: pd.DataFrame,
    layout: dict,
    cubes: Optional[pd.DataFrame] = None,
    gas_flow: Optional[str] = None,
    fitted_stl_path: Optional[str] = None,
    max_points: int = _PREVIEW_PORES_DEFAULT,
    max_wire_cubes: int = 250,
    n_total: Optional[int] = None,
) -> go.Figure:
    fig = go.Figure()
    pw = float(layout.get("plate_width_mm", 120))
    ph = float(layout.get("plate_height_mm", 120))
    size = float(layout.get("cube_size_mm", 8))
    height = float(layout.get("cube_height_mm", size))
    build_shape = str(layout.get("build_shape") or "cube").lower()

    # Filled plate footprint (easy to see) + dark outline
    fig.add_trace(
        go.Mesh3d(
            x=[0, pw, pw, 0],
            y=[0, 0, ph, ph],
            z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#061F4A",
            opacity=0.12,
            name="plate",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0, pw, pw, 0, 0],
            y=[0, 0, ph, ph, 0],
            z=[0, 0, 0, 0, 0],
            mode="lines",
            line=dict(color="#061F4A", width=6),
            name="plate edge",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if cubes is not None and len(cubes):
        # Coupon plate is 4×(19×3)=228 — draw all by default (old cap of 120 looked "missing")
        if len(cubes) <= max_wire_cubes:
            draw = cubes
        else:
            sort_cols = [c for c in ("strip", "row", "col") if c in cubes.columns]
            ordered = cubes.sort_values(sort_cols) if sort_cols else cubes
            # Keep every strip represented when subsampling
            if "strip" in ordered.columns:
                parts = []
                strips = ordered["strip"].unique()
                per = max(1, max_wire_cubes // max(1, len(strips)))
                for s, g in ordered.groupby("strip", sort=True):
                    parts.append(g.head(per))
                draw = pd.concat(parts, ignore_index=True).head(max_wire_cubes)
            else:
                draw = ordered.head(max_wire_cubes)
        # Strip AABBs first (bold) so the coupon grid reads immediately
        sx, sy, sz = _strip_outline_polyline(draw, height)
        if sx:
            n_strips = int(draw["strip"].nunique()) if "strip" in draw.columns else 1
            fig.add_trace(
                go.Scatter3d(
                    x=sx,
                    y=sy,
                    z=sz,
                    mode="lines",
                    line=dict(color="#FC3D21", width=7),
                    opacity=1.0,
                    name=f"strips (n={n_strips})",
                    hoverinfo="skip",
                )
            )
        # One Scatter3d for all cubes — 228×12 separate traces freezes the browser/Dash
        xs: list = []
        ys: list = []
        zs: list = []
        for _, row in draw.iterrows():
            cx_p, cy_p, cz_p = _cuboid_edge_polyline(
                float(row["cx_mm"]),
                float(row["cy_mm"]),
                float(row["size_mm"]),
                float(row.get("height_mm", height)),
                length=float(row["length_mm"]) if "length_mm" in row and pd.notna(row["length_mm"]) else None,
                width_mm=float(row["width_mm"]) if "width_mm" in row and pd.notna(row["width_mm"]) else None,
            )
            xs.extend(cx_p)
            ys.extend(cy_p)
            zs.extend(cz_p)
        if xs:
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color="#1F2A37", width=4),
                    opacity=0.95,
                    name=f"builds (n={len(draw)})",
                    hoverinfo="skip",
                )
            )
    else:
        for tr in _build_wireframe(pw / 2, ph / 2, size, height, shape=build_shape):
            fig.add_trace(tr)

    if str(layout.get("plate_mode", "")).lower() == "stl" and fitted_stl_path and os.path.exists(fitted_stl_path):
        try:
            _add_stl_mesh_plotly(fig, fitted_stl_path)
        except Exception:
            pass

    df = combined
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)

    total_for_label = int(n_total) if n_total else int(len(combined))
    if len(df):
        sources = (
            df["pore_source"].astype(str).str.lower()
            if "pore_source" in df.columns
            else df.get("pore_family", pd.Series(["unknown"] * len(df))).astype(str).str.lower()
        )
        preferred = ["pore", "lof", "edge", "keyhole", "gas", "spatter"]
        ordered = [s for s in preferred if s in set(sources)]
        ordered += sorted(s for s in sources.unique() if s not in preferred)
        for src in ordered:
            sub = df.loc[sources == src]
            # Plotly 3D marker size is screen pixels (not mm) — medium, readable on a 250 mm plate.
            if "radius_mm" in sub.columns:
                sizes = np.clip(sub["radius_mm"].to_numpy() * 1000.0, 4.0, 14.0)
            else:
                sizes = 7.0
            disp = FAMILY_DISPLAY.get(src, src)
            fig.add_trace(
                go.Scatter3d(
                    x=sub["x_mm"],
                    y=sub["y_mm"],
                    z=sub["z_mm"],
                    mode="markers",
                    name=f"{disp} (showing {len(sub):,})",
                    marker=dict(
                        size=sizes,
                        color=FAMILY_COLORS.get(src, "#333"),
                        opacity=0.85,
                        line=dict(width=0),
                        sizemode="diameter",
                    ),
                    hovertemplate=(
                        f"{disp}<br>x=%{{x:.2f}} mm<br>y=%{{y:.2f}} mm<br>z=%{{z:.2f}} mm"
                        "<extra></extra>"
                    ),
                )
            )

    flow = gas_flow or layout.get("gas_flow")
    if flow:
        try:
            _add_gas_flow_arrow_plotly(fig, layout, str(flow))
        except Exception:
            pass

    n_r = layout.get("n_rows", "?")
    n_c = layout.get("n_cols", "?")
    n_s = layout.get("n_strips", 1)
    mode = layout.get("plate_mode", "grid_cubes")
    mode_label = {
        "grid_cubes": f"{n_s}×({n_r}×{n_c}) cubes",
        "single_cuboid": "cuboid",
        "stl": "STL",
    }.get(mode, mode)
    # Stretch Z a bit so 10 mm cubes read on a 250 mm plate
    z_max = max(height * 4.0, 40.0)
    fig.update_layout(
        title=f"Build plate · {mode_label} · preview {min(len(df), max_points):,}/{total_for_label:,} pores",
        margin=dict(l=0, r=0, t=48, b=0),
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=max(0.2, z_max / max(pw, ph))),
            xaxis=dict(range=[-5, pw + 5], showbackground=True, backgroundcolor="#F4F6F8"),
            yaxis=dict(range=[-5, ph + 5], showbackground=True, backgroundcolor="#F4F6F8"),
            zaxis=dict(range=[0, z_max], showbackground=True, backgroundcolor="#FFFFFF"),
            camera=dict(eye=dict(x=1.45, y=-1.55, z=1.15), center=dict(x=0, y=0, z=-0.05)),
        ),
        paper_bgcolor="white",
    )
    return fig


def _make_field_figure(
    combined,
    field,
    layout: dict,
    gas_flow: Optional[str] = None,
    max_overlay_pores: int = 1200,
):
    """2D map: full-plate NU relative field as background; sparse pore overlay."""
    fig2d, ax2 = plt.subplots(figsize=(8.5, 7))
    pw = float(layout.get("plate_width_mm", 120))
    ph = float(layout.get("plate_height_mm", 120))
    plate_mode = str(
        layout.get("plate_mode")
        or (field["plate_mode"].item() if "plate_mode" in field.files else "")
        or ""
    ).lower()

    field_mode = str(field["mode"]) if "mode" in field.files else "builds"
    drew_grid = False
    # STL: no NU colormap — pores / outlines only on a clean plate map
    if plate_mode == "stl":
        ax2.set_facecolor("#F7F9FC")
    elif field_mode == "grid" and "grid_x" in field.files:
        X = np.asarray(field["grid_x"])
        Y = np.asarray(field["grid_y"])
        if "nu_relative_grid" in field.files:
            W = np.asarray(field["nu_relative_grid"], dtype=float)
            cbar = "NU / mean(NU)"
        else:
            W = np.asarray(field["placement_weight_grid"], dtype=float)
            cbar = "Placement weight"
        # Full-plate NU background for coupon cubes + centered cuboid only
        finite = np.isfinite(W)
        vmin = float(np.nanmin(W)) if np.any(finite) else 0.0
        vmax = float(np.nanmax(W)) if np.any(finite) else 1.0
        if abs(vmax - vmin) < 1e-15:
            vmax = vmin + 1e-6
        pcm = ax2.pcolormesh(
            X, Y, W, shading="auto", cmap="viridis",
            vmin=vmin, vmax=vmax, zorder=1,
        )
        fig2d.colorbar(pcm, ax=ax2, label=cbar)
        drew_grid = True
    elif plate_mode != "stl" and field_mode == "builds":
        # Legacy coupon runs: only cube-center scatter (no full grid)
        cx = np.asarray(field["cx"])
        cy = np.asarray(field["cy"])
        if "nu_relative" in field.files:
            w = np.asarray(field["nu_relative"])
            cbar = "NU / mean(NU)"
        else:
            w = np.asarray(field["spatter_weight"])
            cbar = "Placement weight"
        size = float(np.asarray(field["size"])[0]) if len(field["size"]) else 8.0
        sc = ax2.scatter(
            cx,
            cy,
            c=w,
            s=max(40, (size * 8) ** 2 / 4),
            marker="s",
            cmap="viridis",
            alpha=0.95,
            edgecolors="#333",
            linewidths=0.4,
            zorder=1,
        )
        fig2d.colorbar(sc, ax=ax2, label=cbar)

    # Build outlines on top of the NU background
    if drew_grid and "cx" in field.files and "cy" in field.files:
        cx = np.asarray(field["cx"], dtype=float)
        cy = np.asarray(field["cy"], dtype=float)
        if "length_mm" in field.files and "width_mm" in field.files:
            lens = np.asarray(field["length_mm"], dtype=float)
            wids = np.asarray(field["width_mm"], dtype=float)
        elif "size" in field.files:
            sizes = np.asarray(field["size"], dtype=float)
            lens = sizes
            wids = sizes
        else:
            lens = wids = np.full(len(cx), 8.0)
        for i in range(len(cx)):
            li = float(lens[i] if i < len(lens) else lens[0])
            wi = float(wids[i] if i < len(wids) else wids[0])
            ax2.add_patch(
                plt.Rectangle(
                    (cx[i] - li / 2.0, cy[i] - wi / 2.0),
                    li,
                    wi,
                    fill=False,
                    ec="#FFFFFF",
                    lw=1.1,
                    alpha=0.9,
                    zorder=2,
                )
            )
            ax2.add_patch(
                plt.Rectangle(
                    (cx[i] - li / 2.0, cy[i] - wi / 2.0),
                    li,
                    wi,
                    fill=False,
                    ec="#061F4A",
                    lw=0.7,
                    alpha=0.85,
                    zorder=2,
                )
            )
    elif plate_mode == "single_cuboid" and layout.get("builds"):
        # Fallback outline from geom.json builds list
        try:
            b0 = layout["builds"][0]
            cx = float(b0.get("cx_mm", pw / 2))
            cy = float(b0.get("cy_mm", ph / 2))
            li = float(b0.get("length_mm", b0.get("size_mm", 20)))
            wi = float(b0.get("width_mm", b0.get("size_mm", 20)))
            ax2.add_patch(
                plt.Rectangle(
                    (cx - li / 2.0, cy - wi / 2.0),
                    li,
                    wi,
                    fill=False,
                    ec="#061F4A",
                    lw=1.4,
                    zorder=2,
                )
            )
        except Exception:
            pass
    elif plate_mode == "stl":
        m = layout.get("stl_meta") or {}
        try:
            ax2.add_patch(
                plt.Rectangle(
                    (float(m.get("xmin", 0)), float(m.get("ymin", 0))),
                    float(m.get("xmax", pw)) - float(m.get("xmin", 0)),
                    float(m.get("ymax", ph)) - float(m.get("ymin", 0)),
                    fill=False,
                    ec="#061F4A",
                    lw=1.2,
                    ls="--",
                    zorder=2,
                )
            )
        except Exception:
            pass

    n_total = int(len(combined)) if combined is not None else 0
    n_shown = 0
    if n_total > 0:
        src_all = combined["pore_source"].astype(str).str.lower()
        # Prefer showing a few edge pores; subsample bulk pores heavily so colormap stays visible
        for src in sorted(src_all.unique()):
            sub = combined.loc[src_all == src]
            if src == "edge":
                cap = min(len(sub), max(80, max_overlay_pores // 5))
            elif src == "spatter":
                # STL / Combined: show many spatter landings on the 2D map
                cap = min(len(sub), max(max_overlay_pores, 2500))
            else:
                # leave most of the overlay budget for bulk pores
                remaining = max(50, max_overlay_pores - n_shown)
                cap = min(len(sub), remaining)
            if len(sub) > cap:
                sub = sub.sample(cap, random_state=0)
            n_shown += len(sub)
            is_edge = src == "edge"
            is_spatter = src == "spatter"
            is_poreish = src in ("pore", "lof", "gas", "keyhole", "spatter", "cmu") or is_edge
            ax2.scatter(
                sub["x_mm"],
                sub["y_mm"],
                s=6 if is_spatter else (8 if is_edge else 5),
                c=FAMILY_COLORS.get(src, "#333"),
                alpha=0.45 if is_spatter else (0.55 if is_edge else 0.35),
                label=f"{src} (n={len(sub)}/{int((src_all == src).sum())})",
                edgecolors="none" if is_spatter else ("#111111" if is_poreish else "none"),
                linewidths=0.0 if is_spatter else (0.55 if is_poreish else 0.0),
                zorder=3 if is_edge else (2.5 if is_spatter else 2),
            )

    ax2.add_patch(
        plt.Rectangle((0, 0), pw, ph, fill=False, ec="#061F4A", lw=1.5, zorder=3)
    )
    ax2.set_xlim(-2, pw + 2)
    ax2.set_ylim(-2, ph + 2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X plate (mm)")
    ax2.set_ylabel("Y plate (mm)")
    if plate_mode == "stl":
        title = "STL pore map (XY)"
        if n_total > n_shown:
            title = f"STL pore map (XY) — showing {n_shown}/{n_total}"
        ax2.set_title(title)
    elif n_total > n_shown:
        ax2.set_title(f"NU relative field (full plate) + pore overlay ({n_shown}/{n_total})")
    else:
        ax2.set_title("NU relative field (full plate) + pore XY")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.invert_yaxis()
    fig2d.tight_layout()
    return fig2d


def _persist_for_pyvista(
    combined: pd.DataFrame,
    layout: dict,
    cubes: Optional[pd.DataFrame],
    gas_flow: Optional[str] = None,
    fitted_stl_src: Optional[str] = None,
    view_max_points: int = 4000,
    *,
    full_pores_path: Optional[str] = None,
    n_total: Optional[int] = None,
) -> None:
    """Persist a light view sample + geom. Full CSV is copied once (not rewritten)."""
    os.makedirs(PYVISTA_DIR, exist_ok=True)

    view_cols = [
        c
        for c in (
            "x_mm",
            "y_mm",
            "z_mm",
            "pore_source",
            "pore_family",
            "radius_mm",
        )
        if c in combined.columns
    ]
    view_df = combined[view_cols] if view_cols else combined
    n_view_src = int(n_total if n_total is not None else len(view_df))
    if len(view_df) > view_max_points:
        src_col = (
            "pore_source"
            if "pore_source" in view_df.columns
            else ("pore_family" if "pore_family" in view_df.columns else None)
        )
        if src_col:
            parts = []
            groups = list(view_df.groupby(src_col, sort=False))
            per = max(50, view_max_points // max(1, len(groups)))
            for _, g in groups:
                take = min(len(g), per)
                parts.append(g.sample(take, random_state=0) if len(g) > take else g)
            view_df = pd.concat(parts, ignore_index=True)
            if len(view_df) > view_max_points:
                view_df = view_df.sample(view_max_points, random_state=0)
        else:
            view_df = view_df.sample(view_max_points, random_state=0)
    view_df.to_csv(os.path.join(PYVISTA_DIR, "latest_pores_view.csv"), index=False)
    with open(os.path.join(PYVISTA_DIR, "latest_view_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_total": n_view_src,
                "n_view": int(len(view_df)),
                "view_max_points": view_max_points,
            },
            f,
        )

    # Independent copy swapped in atomically. Never hard-link: writing through a link
    # (overlapping refreshes) truncates the run's own pores.csv.
    dest_full = os.path.join(PYVISTA_DIR, "latest_pores.csv")
    tmp_full = f"{dest_full}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        if full_pores_path and os.path.isfile(full_pores_path) and os.path.getsize(full_pores_path) > 0:
            shutil.copyfile(full_pores_path, tmp_full)
        else:
            view_df.to_csv(tmp_full, index=False)
        os.replace(tmp_full, dest_full)
    except Exception:
        try:
            os.remove(tmp_full)
        except OSError:
            pass

    payload = {"mode": "plate", **layout}
    if gas_flow:
        payload["gas_flow"] = gas_flow
    with open(os.path.join(PYVISTA_DIR, "latest_geom.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if cubes is not None:
        cubes.to_csv(os.path.join(PYVISTA_DIR, "latest_cubes.csv"), index=False)
    if fitted_stl_src and os.path.exists(fitted_stl_src):
        dst = os.path.join(PYVISTA_DIR, "latest_fitted.stl")
        shutil.copyfile(fitted_stl_src, dst)


def _howto_panel():
    return html.Div(
        [
            html.H4("Process overview"),
            html.P(
                "Combined builds a plate-scale pore field from CMU empirical density and the "
                "NU spatial map. Gas-flow weighting is not used for bulk placement.",
                className="mb-3",
            ),
            html.Ol(
                [
                    html.Li(
                        [
                            html.Strong("Pre-fit NU (once). "),
                            "Combined requires the shared NU cache before a run.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Choose geometry. "),
                            "Coupon cubes 4×(19×3), a centered cube/cuboid, or an imported STL "
                            "(auto-fit to the plate).",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("CMU density ρ. "),
                            "Empirical pores/mm³ from LoF / keyhole / gas; pore sizes drawn from CMU catalogs.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Count = ρ × volume. "),
                            "Build (or solid mesh) volume sets the base pore count. Optional max-pores "
                            "cap is off by default (0 = uncapped).",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("NU spatial variation. "),
                            "Each region is scaled by NU_local / mean(NU) so the plate mean still matches CMU.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Random disperse. "),
                            "Pores are placed randomly inside each cube / cuboid / STL volume.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Optional scan + edge pores. "),
                            "When enabled, hatch vectors are generated and edge pores sit at scan start/end.",
                        ]
                    ),
                ],
                className="mb-3",
            ),
            html.H4("Tips for laptop runs"),
            html.Ul(
                [
                    html.Li("3D view shows a subsample of pores so the scene stays interactive."),
                    html.Li("STL meshes are decimated for Plotly only; physics still uses the full CAD."),
                    html.Li("Watch the progress bar / ETA while the background worker runs."),
                ]
            ),
        ],
        className="howto-copy",
    )


def _controls_panel(params: dict):
    return html.Div(
        [
            html.P(
                "Bulk pores are labeled “pore”. Count = CMU density × volume × (NU / mean NU); "
                "placement is random inside builds. Pre-fit the NU model before running. "
                "Optional edge pores sit at scan-vector start/end.",
                className="text-muted mb-3",
            ),
            html.H4("Geometry mode"),
            dbc.RadioItems(
                id="comb-plate-mode",
                options=[
                    {"label": "Coupon cubes — 4×(19×3)", "value": "grid_cubes"},
                    {"label": "Centered cube / cuboid", "value": "single_cuboid"},
                    {"label": "Import STL (auto-fit)", "value": "stl"},
                ],
                value="grid_cubes",
                className="mb-2",
            ),
            dbc.Button(
                "Run 5×5×14 mm sweep (1300–1700 mm/s, hatch 140 µm, flucWsMTs width)",
                id="comb-vsweep-btn",
                color="primary",
                outline=True,
                size="sm",
                className="mb-1",
            ),
            html.Small(
                "Fixed 5×5×14 mm pillar. Hatch spacing 140 µm, layer 30 µm. "
                "Melt-pool width is the single-track mean at each speed. "
                "λ=1.04 shifts the mean width, depth, and cap only, at 5 µm voxels, "
                "matching the 0.5 mm cube. Scan paths use Raptor's ScanPathBuilder. "
                "Nine runs (50 mm/s steps). Writes a folder on the Desktop. This takes a long time.",
                className="text-muted d-block mb-1",
            ),
            html.Div(id="comb-vsweep-status", className="small mb-2"),
            html.Div(
                id="comb-grid-controls",
                children=[
                    html.Small(
                        "Matches the LPBF coupon plate: 4 vertical strips of 19×3 cubes (228 total).",
                        className="text-muted d-block mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Strips", className="control-label"),
                                    dbc.Input(id="comb-n-strips", type="number", value=4, min=1, max=8, step=1),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Rows / strip", className="control-label"),
                                    dbc.Input(id="comb-n-rows", type="number", value=19, min=1, max=40, step=1),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Cols / strip", className="control-label"),
                                    dbc.Input(id="comb-n-cols", type="number", value=3, min=1, max=10, step=1),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Cube edge (mm)", className="control-label"),
                                    dbc.Input(id="comb-cube-size", type="number", value=10.0, step=0.5),
                                ],
                                md=3,
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Height (mm)", className="control-label"),
                                    dbc.Input(id="comb-cube-height", type="number", value=10.0, step=0.5),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                ],
            ),
            html.Div(
                id="comb-cuboid-controls",
                style={"display": "none"},
                children=[
                    html.Small(
                        "Small centered cube/cuboid by default (set L×W×H). "
                        "Gas direction biases the pore cloud. "
                        "Enable Raptor below for melt/scan/spatter on this part.",
                        className="text-muted d-block mb-2",
                    ),
                    dbc.RadioItems(
                        id="comb-cuboid-shape",
                        options=[
                            {"label": "Custom L×W×H (centered)", "value": "custom"},
                            {"label": "Cube (edge = height)", "value": "cube"},
                            {"label": "Fill plate (margin from edge)", "value": "fill"},
                        ],
                        value="custom",
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            html.Span("Presets: ", className="control-label me-2"),
                            dbc.Button(
                                "5 × 5 × 14 mm (H = 14)",
                                id="comb-preset-5x5x14",
                                size="sm",
                                color="primary",
                                outline=True,
                                className="me-2",
                            ),
                            dbc.Button(
                                "5 × 5 × 73 mm (4PB bar)",
                                id="comb-preset-5x5x73",
                                size="sm",
                                color="primary",
                                outline=True,
                            ),
                        ],
                        className="mb-2 d-flex align-items-center flex-wrap",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Length X (mm)", className="control-label"),
                                    dbc.Input(id="comb-cuboid-length", type="number", value=5.0, min=0.5, step=0.5),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Width Y (mm)", className="control-label"),
                                    dbc.Input(id="comb-cuboid-width", type="number", value=5.0, min=0.5, step=0.5),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Height Z (mm)", className="control-label"),
                                    dbc.Input(id="comb-cuboid-height", type="number", value=73.0, min=0.5, step=0.5),
                                ],
                                md=4,
                            ),
                            html.Small(
                                "Use the presets above, or type any size. "
                                "For a bar built lying flat, swap to 73×5×5.",
                                className="text-muted",
                            ),
                        ],
                        className="mb-2",
                        id="comb-cuboid-size-row",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Margin from plate edge (mm)", className="control-label"),
                                    dbc.Input(id="comb-cuboid-margin", type="number", value=5.0, step=0.5),
                                    html.Small("Used for Fill plate; also caps L/W to plate.", className="text-muted"),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Gas distribution direction", className="control-label"),
                                    dbc.Select(
                                        id="comb-gas-flow",
                                        options=[
                                            {"label": "+X", "value": "+x"},
                                            {"label": "−X", "value": "-x"},
                                            {"label": "+Y", "value": "+y"},
                                            {"label": "−Y", "value": "-y"},
                                        ],
                                        value="+x",
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        className="mb-2",
                    ),
                ],
            ),
            html.Div(
                id="comb-stl-controls",
                style={"display": "none"},
                children=[
                    html.Div("STL file path", className="control-label"),
                    dbc.Input(
                        id="comb-stl-path-input",
                        type="text",
                        value=DEFAULT_STL_PATH,
                        className="mb-2",
                    ),
                    html.Div("Or upload a different STL", className="control-label"),
                    dcc.Upload(
                        id="comb-stl-upload",
                        children=html.Div(["Drag & drop or ", html.A("select .stl")]),
                        style={
                            "width": "100%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "8px",
                            "textAlign": "center",
                            "marginBottom": "8px",
                        },
                        multiple=False,
                    ),
                    html.Div(
                        id="comb-stl-name",
                        className="text-muted small mb-2",
                        children=(
                            f"Default: {os.path.basename(DEFAULT_STL_PATH)}"
                            if os.path.isfile(DEFAULT_STL_PATH)
                            else "Default STL path not found on disk."
                        ),
                    ),
                    dcc.Store(id="comb-stl-path", data=DEFAULT_STL_PATH),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Fit margin (mm)", className="control-label"),
                                    dbc.Input(id="comb-stl-margin", type="number", value=2.0, step=0.5),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Max height (mm), 0 = full", className="control-label"),
                                    dbc.Input(
                                        id="comb-stl-max-height",
                                        type="number",
                                        value=0,
                                        min=0,
                                        step=0.5,
                                    ),
                                    html.Small(
                                        "0 = whole STL (pores + every scan layer). "
                                        ">0 keeps only the bottom slab for a quick test.",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Portion of the part (cake slice)", className="control-label"),
                                    dcc.Dropdown(
                                        id="comb-stl-sector",
                                        options=[
                                            {"label": "Whole part", "value": 1.0},
                                            {"label": "1/2 (half)", "value": 0.5},
                                            {"label": "1/4 (quarter)", "value": 0.25},
                                            {"label": "1/8", "value": 0.125},
                                            {"label": "1/16", "value": 0.0625},
                                        ],
                                        value=1.0,
                                        clearable=False,
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Slice start angle (deg)", className="control-label"),
                                    dbc.Input(
                                        id="comb-stl-sector-start",
                                        type="number",
                                        value=0,
                                        step=15,
                                    ),
                                    html.Small(
                                        "0° = +x, counter-clockwise, about the part's vertical "
                                        "center axis. Quarter at 0° = the +x/+y quadrant.",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Raptor domain", className="control-label"),
                                    dcc.Dropdown(
                                        id="comb-raptor-stl-domain",
                                        options=[
                                            {
                                                "label": "Full cake-slice / part height (3D)",
                                                "value": "volume",
                                            },
                                            {
                                                "label": "Thin LoF band (flat, hatch-accurate)",
                                                "value": "band",
                                            },
                                        ],
                                        value="volume",
                                        clearable=False,
                                    ),
                                ],
                                md=12,
                            ),
                            html.Small(
                                "Full height runs Raptor on the whole STL / cake slice like the "
                                "cube and 14 mm pillar: every process layer at CMU Hatch/Layer (µm). "
                                "Thin band is only ~0.5 mm tall for a quick hatch-accurate LoF check.",
                                className="text-muted d-block mb-2",
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Thin-band height (mm)", className="control-label"),
                                    dbc.Input(
                                        id="comb-raptor-band-z",
                                        type="number",
                                        placeholder="auto (fullest cross-section)",
                                        step=0.5,
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Thin-band layers", className="control-label"),
                                    dbc.Input(
                                        id="comb-raptor-band-layers",
                                        type="number",
                                        value=15,
                                        min=4,
                                        max=60,
                                        step=1,
                                    ),
                                ],
                                md=6,
                            ),
                            html.Small(
                                "Only used when Raptor domain = Thin LoF band. Leave height empty "
                                "to auto-pick the fullest cross-section.",
                                className="text-muted",
                            ),
                        ],
                        className="mb-2",
                    ),
                ],
            ),
            html.Div(
                id="comb-raptor-controls",
                style={"display": "none"},
                children=[
                    dbc.Checklist(
                        id="comb-use-raptor-lof",
                        options=[
                            {
                                "label": (
                                    "Use Raptor LoF on this part "
                                    "(cuboid / STL region + scan strategy + spatter animation)"
                                ),
                                "value": "on",
                            }
                        ],
                        value=["on"],
                        className="mb-2",
                    ),
                    html.Small(
                        "Raptor always uses Hatch/Layer (µm) from the CMU process box below. "
                        "STL full-height = whole part volume (same path as cube / pillar).",
                        className="text-muted d-block mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Checklist(
                                    id="comb-raptor-lib-on",
                                    options=[
                                        {
                                            "label": "Vamsi melt-pool library (η_D/η_C, depth shape, θ per layer)",
                                            "value": "on",
                                        }
                                    ],
                                    value=["on"] if mpl.available_velocities() else [],
                                    switch=True,
                                ),
                                md=8,
                            ),
                            dbc.Col(
                                dbc.RadioItems(
                                    id="comb-raptor-lib-eta",
                                    options=[{"label": s, "value": s} for s in mpl.ETA_SOURCES],
                                    value="flucWs",
                                    inline=True,
                                ),
                                md=4,
                            ),
                        ],
                        className="mb-1",
                    ),
                    html.Div(id="comb-raptor-lib-hint", className="text-muted small mb-2"),
                    html.Div(
                        id="comb-raptor-roi-controls",
                        style={"display": "none"},
                        children=[
                            dbc.Button(
                                "Select Raptor region on STL…",
                                id="comb-pick-raptor-roi",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="me-2 mb-2",
                                title="Fixed 1×1×1 mm cube — drag pink handle or click to place, S to save",
                            ),
                            dbc.Button(
                                "Clear region",
                                id="comb-clear-raptor-roi",
                                color="link",
                                size="sm",
                                className="mb-2",
                            ),
                            html.Div(
                                id="comb-raptor-roi-status",
                                className="text-muted small mb-2",
                                children="No region selected — Raptor LoF runs on the real cross-section of the part (or cake slice) over a thin band of layers.",
                            ),
                            dcc.Store(id="comb-raptor-roi", data=None),
                            dcc.Interval(
                                id="comb-raptor-roi-tick",
                                interval=1500,
                                n_intervals=0,
                                disabled=True,
                            ),
                        ],
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Plate W (mm)", className="control-label"),
                            dbc.Input(id="comb-plate-w", type="number", value=250.0),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Plate H (mm)", className="control-label"),
                            dbc.Input(id="comb-plate-h", type="number", value=250.0),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Random seed", className="control-label"),
                            dbc.Input(id="comb-seed", type="number", value=42),
                        ],
                        md=4,
                    ),
                ],
                className="mb-2",
            ),
            html.H4("Pore population"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Max pores (0 = full ρ×V)", className="control-label"),
                            dbc.Input(id="comb-totalpores", type="number", value=0, min=0),
                            html.Small(
                                "0 = uncapped N=ρ·V·(NU/mean). Caps the *generated* count, not the 3D view.",
                                className="text-muted",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("Mean porosity (NU input)", className="control-label"),
                            dbc.Input(
                                id="comb-mean-porosity",
                                type="number",
                                value=0.02,
                                step=0.001,
                            ),
                            html.Small(
                                "Setting Max pores also updates Max pores shown (3D downsample).",
                                className="text-muted",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-2",
            ),
            html.H4("NU field (required)"),
            html.Div(id="comb-nu-cache-status", className="mb-2"),
            dbc.Button(
                "Pre-fit NU model (once)",
                id="comb-prefit-nu",
                color="secondary",
                outline=True,
                size="sm",
                className="mb-2 me-2",
            ),
            html.Span(id="comb-prefit-status", className="text-muted small"),
            html.Small(
                "Placement uses only the NU spatial field (local / mean). Gas-flow weighting is disabled.",
                className="text-muted d-block mb-3",
            ),
            html.H4("Scan vectors + edge pores"),
            html.Small(
                "Hatch / layer for the Combined run and for Raptor always come from the CMU "
                "process Hatch (µm) and Layer (µm) fields below — not from manual plate-mm values. "
                "Cube, pillar, and full-height STL scan every process layer of the part.",
                className="text-muted d-block mb-2",
            ),
            dbc.Checklist(
                id="comb-enable-scan",
                options=[{"label": "Generate scan vectors and edge pores in Combined run", "value": "on"}],
                value=["on"],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Scan strategy", className="control-label"),
                            dbc.Select(
                                id="comb-scan-strategy",
                                options=[
                                    {
                                        "label": "Stripes + 67° rotation (realistic, EOS-style)",
                                        "value": "stripes",
                                    },
                                    {"label": "Horizontal (along +X)", "value": "horizontal"},
                                    {"label": "Vertical (along +Y)", "value": "vertical"},
                                    {"label": "Diagonal (+45°)", "value": "diagonal"},
                                    {"label": "Diagonal (−45°)", "value": "diagonal_neg"},
                                    {"label": "Crosshatch (0° + 90°)", "value": "crosshatch"},
                                    {"label": "Rotating (build-dir rotate)", "value": "rotate"},
                                    {"label": "Batch / tiled (e.g. 4×4)", "value": "batch"},
                                ],
                                value="stripes",
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        [
                            html.Div("Stripe width (mm)", className="control-label"),
                            dbc.Input(
                                id="comb-stripe-mm", type="number", value=10.0, min=0.5, step=0.5
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div("Edge pores (count)", className="control-label"),
                            dbc.Input(id="comb-n-edge", type="number", value=100, min=0, step=10),
                        ],
                        md=4,
                    ),
                ],
                className="mb-2",
            ),
            # Kept for callbacks / store compatibility; always on (process µm params)
            dbc.Checklist(
                id="comb-use-process-scan",
                options=[
                    {
                        "label": "Use CMU/Raptor process Hatch & Layer (µm) and scan every "
                        "layer of the part",
                        "value": "on",
                    }
                ],
                value=["on"],
                style={"display": "none"},
            ),
            dbc.Checklist(
                id="comb-make-spatter-anim",
                options=[
                    {
                        "label": "Make spatter animation (GIF) — slowest step; untick for faster runs",
                        "value": "on",
                    }
                ],
                value=[],
                className="mb-1",
            ),
            html.Small(
                "Spatter GIF is optional and slow; leave off unless you need it. "
                "Raptor / scan paths use Hatch & Layer (µm) from the CMU process box.",
                className="text-muted d-block mb-2",
            ),
            html.Div(
                id="comb-batch-controls",
                style={"display": "none"},
                children=[
                    dbc.Alert(
                        [
                            html.Strong("Batch / tiled scan example (centered cube)"),
                            html.P(
                                "Micro-CT XY slice of a cube scanned in a 4×4 tile pattern "
                                "(Kyle sample). Pore bands mark tile boundaries where scan "
                                "islands meet — the same idea as Batch strategy below.",
                                className="mb-2 mt-1 small",
                            ),
                            html.Img(
                                src="/assets/kyle_batch_sample.png",
                                alt="Kyle sample: cube scanned in 4×4 tiles (XY CT slice)",
                                className="batch-example-img",
                            ),
                            html.Div(
                                "Example: 4×4 tiles on one cube layer (XY, z≈1516 in CT).",
                                className="text-muted small mt-1",
                            ),
                        ],
                        color="light",
                        className="mb-2 border",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Batch rows (tiles)", className="control-label"),
                                    dbc.Input(id="comb-batch-rows", type="number", value=4, min=1, max=16, step=1),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Batch cols (tiles)", className="control-label"),
                                    dbc.Input(id="comb-batch-cols", type="number", value=4, min=1, max=16, step=1),
                                ],
                                md=6,
                            ),
                        ],
                        className="mb-2",
                    ),
                ],
            ),
            # Hidden fallbacks for legacy callbacks (preview / old configs). Real runs
            # always use CMU Hatch/Layer (µm) and scan every process layer.
            html.Div(
                [
                    dbc.Input(id="comb-hatch-mm", type="number", value=0.14),
                    dbc.Input(id="comb-layer-mm", type="number", value=0.03),
                    dbc.Input(id="comb-max-layers", type="number", value=99999),
                ],
                style={"display": "none"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Layer rotation (° / layer)", className="control-label"),
                            dbc.Input(id="comb-layer-rot", type="number", value=67.0, step=1.0),
                            html.Small(
                                "0 = same hatch every layer. Typical LPBF rotate ≈ 67°.",
                                className="text-muted",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("Base angle override (°)", className="control-label"),
                            dbc.Input(
                                id="comb-base-angle",
                                type="number",
                                placeholder="auto from strategy",
                            ),
                            html.Small("Leave blank to use strategy default.", className="text-muted"),
                        ],
                        md=6,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Button(
                "Preview scan vectors + edge pores",
                id="comb-scan-preview-btn",
                color="secondary",
                outline=True,
                className="mb-2 me-2",
            ),
            html.Span(id="comb-scan-preview-status", className="text-muted small"),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Scan vector preview")),
                    dbc.ModalBody(
                        [
                            html.P(
                                "Green circles = laser START · Red diamonds = laser END · "
                                "Orange tip = scan direction · Purple × = edge pores. "
                                "For STL, hatches are clipped to the CAD interior at each slice.",
                                className="small text-muted",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div("Show layer", className="control-label"),
                                            dbc.Select(
                                                id="comb-scan-layer-filter",
                                                options=[{"label": "All layers", "value": "all"}],
                                                value="all",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        html.Div(id="comb-scan-preview-meta", className="small mt-4"),
                                        md=8,
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dcc.Graph(id="comb-scan-preview-fig", style={"height": "620px"}),
                            dcc.Store(id="comb-scan-preview-store"),
                        ]
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="comb-scan-preview-close", className="ms-auto")
                    ),
                ],
                id="comb-scan-preview-modal",
                size="xl",
                is_open=False,
                scrollable=True,
            ),
            html.H4("CMU size catalog (process)"),
            dbc.Checklist(
                id="comb-pore-models",
                options=[
                    {"label": "LoF sizes", "value": "lof"},
                    {"label": "Keyhole sizes", "value": "keyhole"},
                    {"label": "Gas sizes", "value": "gas"},
                ],
                value=["lof", "keyhole", "gas"],
                inline=True,
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Power (W)", className="control-label"),
                            dbc.Input(id="comb-power", type="number", value=params.get("Power (W)", 370)),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("Velocity (mm/s)", className="control-label"),
                            dbc.Input(
                                id="comb-velocity",
                                type="number",
                                value=1400,
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Hatch (um) — Raptor / CMU process", className="control-label"),
                            dbc.Input(
                                id="comb-hatch",
                                type="number",
                                value=params.get("Hatch Spacing (um)", 140),
                            ),
                            html.Small(
                                "Used for Raptor and Combined scan paths (cube, pillar, and STL).",
                                className="text-muted",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("Layer (um) — Raptor / CMU process", className="control-label"),
                            dbc.Input(
                                id="comb-layer",
                                type="number",
                                value=params.get("Layer Thickness (um)", 30),
                            ),
                            html.Small(
                                "Process layer thickness for every layer of the part (not plate mm).",
                                className="text-muted",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Compute device", className="control-label"),
                            dbc.RadioItems(
                                id="comb-compute-device",
                                options=[
                                    {"label": "Auto (prefer GPU)", "value": "auto"},
                                    {"label": "GPU (CUDA)", "value": "gpu"},
                                    {"label": "CPU only", "value": "cpu"},
                                ],
                                value="auto",
                                inline=True,
                            ),
                            html.Div(id="comb-compute-hint", className="text-muted small mt-1"),
                        ],
                        md=12,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Button(
                "Generate plate pore field",
                id="comb-run",
                color="primary",
                className="w-100 btn-nasa",
            ),
            html.Div(
                id="comb-run-status",
                className="mt-2 fw-semibold",
                style={"color": "#0B3D91"},
                children="",
            ),
            dbc.Progress(
                id="comb-run-progress",
                value=0,
                striped=True,
                animated=True,
                className="mt-2",
                style={"height": "0.7rem", "display": "none"},
            ),
        ]
    )

def layout():
    params = _load_json(EXAMPLE_PARAM_PATH) if os.path.exists(EXAMPLE_PARAM_PATH) else {}
    return html.Div(
        [
            dcc.Download(id="comb-csv-download"),
            dcc.Store(id="comb-job-store", data=None),
            dcc.Store(id="comb-viewer-watch", data=None),
            dcc.Interval(id="comb-nu-status-tick", interval=5000, n_intervals=0),
            dcc.Interval(id="comb-job-tick", interval=1000, n_intervals=0, disabled=True),
            dcc.Interval(id="comb-viewer-tick", interval=1000, n_intervals=0, disabled=True),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H3("Combined — 3D Build Plate"),
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            _controls_panel(params),
                                            label="Run controls",
                                            tab_id="comb-tab-controls",
                                        ),
                                        dbc.Tab(
                                            _howto_panel(),
                                            label="How it works",
                                            tab_id="comb-tab-howto",
                                        ),
                                    ],
                                    id="comb-side-tabs",
                                    active_tab="comb-tab-controls",
                                    className="mb-2",
                                ),
                            ],
                            className="panel-card",
                        ),
                        lg=5,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H3("Results"),
                                html.H4("Log"),
                                html.Div(
                                    dbc.Textarea(
                                        id="comb-log-output",
                                        rows=8,
                                        readOnly=True,
                                        value=(
                                            "Click Generate plate pore field.\n"
                                            "Default: 4×(19×3) coupon cubes, N=ρ·V·NU (often ~1e5 pores).\n"
                                            "3D preview shows a subsample; Download CSV has the full set."
                                        ),
                                    ),
                                    className="log-box mb-3",
                                ),
                                dcc.Loading(
                                    dcc.Graph(
                                        id="comb-figure-3d",
                                        figure=go.Figure(
                                            layout=dict(
                                                title="Plate pore cloud will appear here after Generate",
                                                height=620,
                                                paper_bgcolor="white",
                                            )
                                        ),
                                        config={"scrollZoom": True},
                                    ),
                                    type="circle",
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Copy image to clipboard",
                                            id="comb-copy-3d",
                                            color="secondary",
                                            outline=True,
                                            size="sm",
                                            className="me-2",
                                        ),
                                        html.Small(id="comb-copy-3d-status", className="text-muted"),
                                    ],
                                    className="mb-2",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    "Max pores shown (3D downsample)",
                                                    className="control-label",
                                                ),
                                                dbc.Input(
                                                    id="comb-preview-pores",
                                                    type="number",
                                                    value=_PREVIEW_PORES_DEFAULT,
                                                    min=1,
                                                    max=_PREVIEW_PORES_MAX,
                                                    step=100,
                                                ),
                                            ],
                                            md=3,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Refresh 3D preview",
                                                    id="comb-refresh-preview",
                                                    color="primary",
                                                    outline=True,
                                                    size="sm",
                                                    className="mt-4",
                                                    disabled=True,
                                                    title="Re-draw with current max-pores-shown (no re-run)",
                                                ),
                                            ],
                                            md=3,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Open rotatable PyVista window",
                                                    id="comb-open-pyvista",
                                                    color="secondary",
                                                    outline=True,
                                                    size="sm",
                                                    className="mt-4",
                                                    disabled=True,
                                                ),
                                            ],
                                            md=3,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Open Raptor RVE (scan paths)",
                                                    id="comb-open-raptor-pv",
                                                    color="secondary",
                                                    outline=True,
                                                    size="sm",
                                                    className="mt-4",
                                                    disabled=True,
                                                    title=(
                                                        "Interactive Raptor phase cube + cyan hatch "
                                                        "paths for this Combined cuboid"
                                                    ),
                                                ),
                                            ],
                                            md=3,
                                        ),
                                    ],
                                    className="mt-2 mb-1",
                                ),
                                html.Small(
                                    "Cap only affects the pore-cloud 3D view. "
                                    "Cuboid or STL + “Use Raptor LoF” runs Raptor on the whole "
                                    "part volume (or thin band if chosen) with CMU Hatch/Layer (µm) "
                                    "and the scan strategy — same path as the cube / 14 mm pillar.",
                                    className="text-muted d-block mb-3",
                                ),
                                dcc.Loading(
                                    html.Div(
                                        id="comb-figure-2d",
                                        children=empty_figure_placeholder(
                                            "Gas-flow weight map + XY pores will appear here."
                                        ),
                                    ),
                                    type="circle",
                                ),
                                html.Div(id="comb-results-output", className="mt-3"),
                                dbc.Button(
                                    "Download pores + process params",
                                    id="comb-download-btn",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    className="mb-2",
                                    disabled=True,
                                ),
                                html.Div(id="comb-pyvista-status", className="text-muted small"),
                            ],
                            className="panel-card",
                        ),
                        lg=7,
                    ),
                ],
                className="g-3",
            ),
        ]
    )


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    pid = int(pid)
    if os.name == "nt":
        try:
            import ctypes

            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def _format_eta(eta_s: Optional[float]) -> str:
    if eta_s is None:
        return "estimating…"
    s = max(0, int(round(float(eta_s))))
    if s < 60:
        return f"~{s}s remaining"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"~{m}m {sec:02d}s remaining"
    h, m = divmod(m, 60)
    return f"~{h}h {m:02d}m remaining"


def _progress_status_text(prog: Dict[str, Any], elapsed_fallback: float = 0.0) -> str:
    msg = str(prog.get("message") or "Running…")
    step = prog.get("step")
    n_steps = prog.get("n_steps")
    pct = prog.get("pct")
    eta = prog.get("eta_s")
    elapsed = prog.get("elapsed_s", elapsed_fallback)
    parts = []
    if step is not None and n_steps:
        parts.append(f"Step {step}/{n_steps}")
    if pct is not None:
        parts.append(f"{pct:.0f}%")
    parts.append(_format_eta(eta))
    parts.append(f"elapsed {float(elapsed):.0f}s")
    return f"{msg} · " + " · ".join(parts)


def _clamp_preview_pores(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = _PREVIEW_PORES_DEFAULT
    # Allow matching small Max-pores caps (e.g. 500); floor at 1
    return int(max(1, min(_PREVIEW_PORES_MAX, n)))


def _spatial_downsample(df: pd.DataFrame, max_points: int, *, n_bins: int = 8) -> pd.DataFrame:
    """
    Spatially distributed downsample: XY grid bins × pore_source strata.
    Keeps rare classes (edge/spatter) visible while covering the plate.
    """
    if df is None or len(df) <= max_points:
        return df if df is not None else pd.DataFrame(columns=list(_PLOT_CSV_COLS))

    src_col = (
        "pore_source"
        if "pore_source" in df.columns
        else ("pore_family" if "pore_family" in df.columns else None)
    )
    has_xy = "x_mm" in df.columns and "y_mm" in df.columns
    if not has_xy:
        return df.sample(max_points, random_state=0)

    work = df.copy()
    x = work["x_mm"].to_numpy(dtype=float)
    y = work["y_mm"].to_numpy(dtype=float)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    nx = max(2, int(n_bins))
    ny = max(2, int(n_bins))
    # Digitize into [0, nx) × [0, ny)
    ix = np.clip(((x - xmin) / max(1e-9, xmax - xmin) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((y - ymin) / max(1e-9, ymax - ymin) * ny).astype(int), 0, ny - 1)
    work["_bin"] = ix * ny + iy
    if src_col:
        work["_src"] = work[src_col].astype(str).str.lower()
    else:
        work["_src"] = "pore"

    # Reserve a small budget for rare classes
    rare = {"edge", "spatter"}
    parts: list[pd.DataFrame] = []
    rare_budget = max(30, max_points // 15)
    used = 0
    for key in sorted(rare):
        g = work.loc[work["_src"] == key]
        if len(g) == 0:
            continue
        take = min(len(g), max(10, rare_budget // max(1, len(rare))))
        parts.append(g.sample(take, random_state=0) if len(g) > take else g)
        used += take
    bulk = work.loc[~work["_src"].isin(rare)]
    budget = max(0, max_points - used)
    if len(bulk) and budget > 0:
        # Equal-ish share per spatial bin among bulk pores
        bins = bulk.groupby("_bin", sort=False)
        n_bins_present = max(1, bins.ngroups)
        per_bin = max(1, budget // n_bins_present)
        for _, g in bins:
            # Within bin, keep source diversity
            if "_src" in g.columns and g["_src"].nunique() > 1:
                sub_parts = []
                srcs = list(g.groupby("_src", sort=False))
                per_src = max(1, per_bin // max(1, len(srcs)))
                for _, sg in srcs:
                    t = min(len(sg), per_src)
                    sub_parts.append(sg.sample(t, random_state=0) if len(sg) > t else sg)
                picked = pd.concat(sub_parts, ignore_index=True)
                if len(picked) > per_bin:
                    picked = picked.sample(per_bin, random_state=0)
                parts.append(picked)
            else:
                t = min(len(g), per_bin)
                parts.append(g.sample(t, random_state=0) if len(g) > t else g)

    if not parts:
        out = work.sample(min(len(work), max_points), random_state=0)
    else:
        out = pd.concat(parts, ignore_index=True)
        if len(out) > max_points:
            out = out.sample(max_points, random_state=0)
        elif len(out) < min(max_points, len(work)):
            # Top up randomly so we use the full preview budget
            need = min(max_points, len(work)) - len(out)
            if need > 0:
                # Avoid exact row duplicates by sampling from work and dropping overlaps on xyz
                extra = work.sample(min(len(work), need * 3), random_state=1)
                out = pd.concat([out, extra], ignore_index=True)
                subset = [c for c in ("x_mm", "y_mm", "z_mm") if c in out.columns]
                if subset:
                    out = out.drop_duplicates(subset=subset, keep="first")
                if len(out) > max_points:
                    out = out.sample(max_points, random_state=0)
    return out.drop(columns=[c for c in ("_bin", "_src") if c in out.columns], errors="ignore")


def _load_pores_for_plot(
    pores_path: str,
    max_points: int = _PREVIEW_PORES_DEFAULT,
    *,
    n_builds: Optional[int] = None,
    plate_w: Optional[float] = None,
    plate_h: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load viz columns and downsample evenly across the plate.

    Coupon CSVs are written build-by-build, so a naive head/sample clusters in
    one corner. Stratify by build_index (or XY bins), then spatial-downsample.
    Avoids DataFrame.to_dict('records') — that path can break when OneDrive
    dehydrates pandas package files under .venv.
    """
    del plate_w, plate_h  # reserved for callers; XY bounds come from the data
    max_points = _clamp_preview_pores(max_points)
    header = pd.read_csv(pores_path, nrows=0)
    wanted = list(_PLOT_CSV_COLS) + ["build_index"]
    usecols = [c for c in wanted if c in header.columns] or None

    df = pd.read_csv(pores_path, usecols=usecols, low_memory=False)
    if len(df) == 0:
        return pd.DataFrame(columns=list(_PLOT_CSV_COLS))
    if len(df) <= max_points:
        return df

    n_groups = 0
    if "build_index" in df.columns and df["build_index"].notna().any():
        parts: list[pd.DataFrame] = []
        groups = list(df.groupby("build_index", sort=False))
        n_groups = len(groups)
        hint = int(n_builds) if n_builds and int(n_builds) > 0 else n_groups
        per = max(1, int(np.ceil(max_points * 1.4 / max(1, hint))))
        for _, g in groups:
            take = min(len(g), per)
            parts.append(g.sample(take, random_state=0) if len(g) > take else g)
        pool = pd.concat(parts, ignore_index=True)
    else:
        # Fall back: light random oversample, then spatial polish
        over = min(len(df), max(max_points * 3, max_points))
        pool = df.sample(over, random_state=0) if len(df) > over else df
        n_groups = 16

    if len(pool) > max_points * 2:
        pool = pool.sample(max_points * 2, random_state=0)
    return _spatial_downsample(
        pool, max_points, n_bins=max(6, int(np.sqrt(max(4, n_groups or 16))))
    )


def _finalize_combined_outdir(outdir: str, preview_pores: Optional[int] = None) -> Tuple[Any, ...]:
    """Load worker outputs into the Combined panel return tuple."""
    global _LAST_CSV, _LAST_CSV_PATH, _LAST_SUMMARY, _LAST_RAPTOR_DIR
    empty_fig = go.Figure(layout=dict(title="3D plot unavailable", height=620))
    _LAST_RAPTOR_DIR = None
    status_path = os.path.join(outdir, "status.json")
    log_path = os.path.join(outdir, "worker.log")
    worker_log = ""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            worker_log = f.read()

    if not os.path.exists(status_path):
        return (
            worker_log or "Worker finished without status.json",
            dbc.Alert("Combined run failed. See log.", color="danger"),
            empty_fig,
            empty_figure_placeholder("Field plot unavailable."),
            True,
            True,
            True,
            True,
            "Failed — see Log",
            "Combined 3D failed",
            100,
            {"display": "none"},
        )

    status = _load_json(status_path)
    if not status.get("ok"):
        return (
            status.get("log") or status.get("error") or worker_log,
            dbc.Alert("Combined run failed. See log.", color="danger"),
            empty_fig,
            empty_figure_placeholder("Field plot unavailable."),
            True,
            True,
            True,
            True,
            "Failed — see Log",
            "Combined 3D failed",
            100,
            {"display": "none"},
        )

    pores_path = os.path.join(outdir, "pores.csv")
    layout_dict = _load_json(os.path.join(outdir, "geom.json"))
    summary = _load_json(os.path.join(outdir, "summary.json"))
    field = np.load(os.path.join(outdir, "field.npz"))
    cubes_path = os.path.join(outdir, "cubes.csv")
    cubes = pd.read_csv(cubes_path) if os.path.exists(cubes_path) else None

    # Fast path: plot a spatially distributed subsample (full CSV unchanged)
    n_total = int(summary.get("total_pores") or summary.get("n_bulk_pores") or 0)
    preview_n = _clamp_preview_pores(
        preview_pores if preview_pores is not None else _PREVIEW_PORES_DEFAULT
    )
    plot_df = _load_pores_for_plot(
        pores_path,
        max_points=preview_n,
        n_builds=int(layout_dict.get("n_builds") or summary.get("n_builds") or 0) or None,
        plate_w=float(layout_dict.get("plate_width_mm") or 250),
        plate_h=float(layout_dict.get("plate_height_mm") or 250),
    )

    _LAST_CSV = None
    _LAST_CSV_PATH = pores_path if os.path.isfile(pores_path) else None
    _LAST_SUMMARY = summary
    fitted_stl_src = os.path.join(outdir, "fitted.stl")
    flow = layout_dict.get("gas_flow")
    _persist_for_pyvista(
        plot_df,
        layout_dict,
        cubes,
        gas_flow=flow,
        fitted_stl_src=fitted_stl_src if os.path.exists(fitted_stl_src) else None,
        view_max_points=preview_n,
        full_pores_path=_LAST_CSV_PATH,
        n_total=n_total or int(len(plot_df)),
    )

    fig3d = _make_plotly_3d(
        plot_df,
        layout_dict,
        cubes,
        gas_flow=flow,
        fitted_stl_path=fitted_stl_src if os.path.exists(fitted_stl_src) else None,
        max_points=preview_n,
        max_wire_cubes=max(250, int(layout_dict.get("n_builds") or 0) or 250),
        n_total=n_total or int(len(plot_df)),
    )
    fig2d = None
    try:
        fig2d = _make_field_figure(plot_df, field, layout_dict, max_overlay_pores=800)
        field_img = fig_to_img(fig2d, width_px=720, height_px=600, dpi=80)
    finally:
        if fig2d is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig2d)
            except Exception:
                pass

    # Embed Raptor 3D view + hatch-space slice when the cuboid RVE ran
    raptor_children: list = []
    scan_anim_dir = os.path.join(outdir, "scan_anim")
    rap_dir = os.path.join(outdir, "raptor_rve")
    rap_preview = os.path.join(rap_dir, "preview.npz")
    rap_png = os.path.join(rap_dir, "preview_3d.png")
    rap_vti = os.path.join(rap_dir, "porosity_map.vti")
    if not os.path.isfile(rap_vti):
        rap_vti = os.path.join(rap_dir, "rve.vti")
    has_raptor = (
        os.path.isfile(rap_png)
        or os.path.isfile(rap_preview)
        or os.path.isfile(rap_vti)
    )
    if has_raptor:
        _LAST_RAPTOR_DIR = rap_dir
        try:
            from .raptor_panel import _slice_figure

            rap_sum = dict(summary.get("raptor_lof") or {})
            rap_sum_path = os.path.join(rap_dir, "summary.json")
            if os.path.isfile(rap_sum_path):
                rap_sum = {**_load_json(rap_sum_path), **rap_sum}
            if not rap_sum.get("hatch_m"):
                rap_sum["hatch_m"] = float(rap_sum.get("hatch_m") or 140e-6)
            size_mm = rap_sum.get("size_xyz_mm") or []
            if len(size_mm) >= 3:
                size_txt = (
                    f"{float(size_mm[0]):.1f}×{float(size_mm[1]):.1f}×{float(size_mm[2]):.1f} mm"
                )
            else:
                edge_um = float(rap_sum.get("rve_edge_m") or 5.0e-4) * 1.0e6
                size_txt = f"~{edge_um:.0f} µm cube"
            band = rap_sum.get("stl_band") or None
            if band:
                place = band.get("band_placement") or "user"
                place_txt = {
                    "fullest": "fullest cross-section",
                    "middle": "mid-height",
                    "user": "user height",
                    "full_height": "full cake-slice / part height",
                }.get(place, place)
                if place == "full_height":
                    size_txt = (
                        f"3D cake-slice / STL volume @ full height "
                        f"({float(band['z_bottom_mm']):.2f}–{float(band['z_top_mm']):.2f} mm), "
                        f"box {size_txt}, voxel "
                        f"{float(rap_sum.get('voxel_resolution_m') or 0) * 1e6:.1f} µm, "
                        f"{int(band['n_layers'])} hatch planes / {int(band['n_layers_part'])} process layers; "
                        f"LoF measured on {float(band['measured_volume_mm3']):.3g} mm³ inside the part"
                    )
                else:
                    size_txt = (
                        f"cake-slice / STL cross-section @ {place_txt}, layers "
                        f"{int(band['first_layer'])}–"
                        f"{int(band['first_layer']) + int(band['n_layers']) - 1} of "
                        f"{int(band['n_layers_part'])} (z {float(band['z_bottom_mm']):.2f}–"
                        f"{float(band['z_top_mm']):.2f} mm), box {size_txt}, voxel "
                        f"{float(rap_sum.get('voxel_resolution_m') or 0) * 1e6:.1f} µm; LoF measured on "
                        f"{float(band['measured_volume_mm3']):.3g} mm³ inside the part "
                        f"({float(band['skin_mm']):.2f} mm contour skin excluded)"
                    )
            strat = rap_sum.get("scan_strategy") or "—"
            raptor_children = [
                html.Hr(className="my-3"),
                html.H5("Raptor on your part (cuboid / STL region)", className="mb-1"),
                html.Small(
                    f"Domain {size_txt} · scan strategy={strat}"
                    + (
                        f" · {int(rap_sum['n_layers_scanned'])} layers scanned"
                        if rap_sum.get("n_layers_scanned")
                        else ""
                    )
                    + (
                        f" · LoF measured on the interior (top/bottom "
                        f"{int(rap_sum['excluded_face_layers'])} layers excluded)"
                        if rap_sum.get("excluded_face_layers")
                        else ""
                    )
                    + ". "
                    "Drag to rotate, scroll to zoom. For a full-height cake slice the grey solid "
                    "is the real fitted STL. Buttons switch the melted build, pores, and all pore "
                    "sources. Thin-band mode colours Raptor melt phases on a short cross-section. "
                    "“Open Raptor RVE” opens a desktop window with a slice plane.",
                    className="text-muted d-block mb-2",
                ),
            ]
            rap_view3d = os.path.join(rap_dir, "view3d.npz")
            rap_fig3d = None
            pore_types, model_z = None, None
            try:
                from .raptor_view import make_plotly_phase_figure, view3d_from_vti

                if not os.path.isfile(rap_view3d) and os.path.isfile(rap_vti):
                    view3d_from_vti(rap_vti, rap_view3d)
                if os.path.isfile(rap_view3d):
                    model_pores = None
                    sub_csv = os.path.join(rap_dir, "combined_pores_in_domain.csv")
                    if os.path.isfile(sub_csv):
                        try:
                            model_pores = pd.read_csv(sub_csv)
                            zj = _load_json(sub_csv.replace(".csv", ".json"))
                            box_h = float((rap_sum.get("size_xyz_mm") or [0, 0, 0])[2])
                            if zj and float(zj["z_hi_mm"]) - float(zj["z_lo_mm"]) > box_h + 1e-6:
                                model_z = (float(zj["z_lo_mm"]), float(zj["z_hi_mm"]))
                        except Exception:
                            model_pores = None
                    raptor_pores = None
                    shapes_csv = os.path.join(rap_dir, "lof_pore_shapes.csv")
                    if model_pores is not None and os.path.isfile(shapes_csv):
                        try:
                            raptor_pores = pd.read_csv(shapes_csv)
                        except Exception:
                            raptor_pores = None
                    if model_pores is not None:
                        from .raptor_view import pore_type_table

                        pore_types = pore_type_table(model_pores, raptor_pores)
                    outline_npz = (
                        (band or {}).get("outline_npz")
                        or os.path.join(rap_dir, "stl_outline.npz")
                    )
                    # Parent outdir holds fitted.stl — use it for full-height cake slices
                    # so the view is the real 3D part, not AABB face slabs
                    fitted_stl = os.path.join(os.path.dirname(rap_dir), "fitted.stl")
                    use_stl_solid = bool(
                        (band or {}).get("full_height") or (band or {}).get("band_placement") == "full_height"
                    ) and os.path.isfile(fitted_stl)
                    rap_fig3d = make_plotly_phase_figure(
                        rap_view3d,
                        title=(
                            f"Raptor · 1/N cake-slice (drag to rotate)"
                            if use_stl_solid
                            else f"Raptor result · {size_txt} (drag to rotate)"
                        ),
                        pore_types=pore_types,
                        model_z_range=model_z,
                        outline_npz=None if use_stl_solid else (
                            outline_npz if os.path.isfile(outline_npz) else None
                        ),
                        stl_path=fitted_stl if use_stl_solid else None,
                        path_csv=None,
                    )
            except Exception:
                rap_fig3d = None
            if rap_fig3d is not None:
                raptor_children.append(
                    dcc.Graph(
                        id="comb-raptor-fig3d",
                        figure=rap_fig3d,
                        config={"displaylogo": False, "scrollZoom": True},
                        style={"height": "780px"},
                    )
                )
                raptor_children.append(
                    html.Div(
                        [
                            dbc.Button(
                                "Copy image to clipboard",
                                id="comb-copy-raptor3d",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="me-2",
                            ),
                            html.Small(id="comb-copy-raptor3d-status", className="text-muted"),
                        ],
                        className="mb-2",
                    )
                )
                raptor_children.extend(_pore_types_children(pore_types, model_z))
            elif os.path.isfile(rap_png):
                import base64

                with open(rap_png, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                raptor_children.extend(
                    [
                        html.Img(
                            src=f"data:image/png;base64,{b64}",
                            className="result-figure",
                            style={"width": "100%", "height": "auto", "objectFit": "contain"},
                            alt="Raptor 3D phase + scan-path preview",
                        ),
                    ]
                )
            raptor_children.extend(
                _lof_shape_children(
                    rap_sum.get("lof_shape") or {},
                    os.path.join(rap_dir, "lof_pore_shapes.csv"),
                    pore_types=pore_types,
                )
            )
            if not os.path.isdir(scan_anim_dir):
                raptor_children.extend(_spatter_children(rap_dir))
            if os.path.isfile(rap_preview):
                rap_fig = _slice_figure(rap_preview, rap_sum)
                raptor_children.extend(
                    [
                        html.H6("XY slice (hatch space)", className="mt-3 mb-1"),
                        html.Small(
                            "Axes are hatch spacings (not mm). Dotted lines mark hatch pitches.",
                            className="text-muted d-block mb-2",
                        ),
                        dcc.Graph(figure=rap_fig, config={"displayModeBar": False}),
                    ]
                )
        except Exception as exc:
            raptor_children = [
                html.Hr(className="my-3"),
                dbc.Alert(
                    f"Raptor RVE finished but the Combined preview failed to load: {exc}",
                    color="warning",
                    className="mb-0",
                ),
            ]

    whole_part_anim = _spatter_children(
        scan_anim_dir,
        desc=(
            "Laser path through the whole part: evenly spaced layers from the first to the "
            "last, each followed until the scan ends at the part boundary. Spatter particles "
            "fly ~30 cm from the laser. Right panel: accumulated landings near the sample."
        ),
    )
    # Raptor first so cuboid LoF view is visible without scrolling past the gas map
    img2d = html.Div([*raptor_children, *whole_part_anim, field_img])

    realized = summary.get("counts_realized") or {}
    shown = int(len(plot_df))
    total_label = n_total or shown
    rap_note = ""
    if _LAST_RAPTOR_DIR or summary.get("use_raptor_lof"):
        rap_note = " · Raptor RVE below (phase + scan paths)"
    timing = summary.get("timing_s") or {}
    proc = summary.get("process") or {}
    run_time_txt = _format_run_time(timing)
    expected_txt = _format_expected_pores(summary)
    result = dbc.Alert(
        [
            *([html.Div(html.B(run_time_txt), className="mb-1")] if run_time_txt else []),
            *([html.Div(html.B(expected_txt), className="mb-1")] if expected_txt else []),
            *([html.Div(_format_process(proc), className="mb-1")] if proc else []),
            html.Div(
                f"Mode={layout_dict.get('plate_mode')} · "
                f"N_bulk={summary.get('n_bulk_pores', summary.get('total_pores'))} · "
                f"edge={summary.get('n_edge_pores', 0)} · "
                f"ρ={summary.get('cmu_total_density')} · "
                f"V={summary.get('part_volume_mm3')} · "
                f"NU_mean={summary.get('nu_mean_pct')}%{rap_note}"
            ),
            html.Div(f"Counts: {realized}"),
            *(
                [html.Div(_format_scan_stats(summary["scan_stats"]), className="small mt-1")]
                if summary.get("scan_stats")
                else []
            ),
            html.Div(
                f"3D preview shows {shown:,} of {total_label:,} pores "
                f"(spatially downsampled to {preview_n:,}). "
                "Change “3D preview downsample” + Refresh, or Download CSV for the full set.",
                className="small mt-1",
            ),
        ],
        color="success",
    )
    # outs: log, results, fig3d, fig2d, dl, pv, refresh, raptor_pv, status, app, pct, style
    return (
        status.get("log") or worker_log or "Done.",
        result,
        fig3d,
        img2d,
        False,
        False,
        False,
        not bool(_LAST_RAPTOR_DIR),
        f"Done — 3D preview ({shown:,}/{total_label:,} pores). Drag to rotate."
        + (f" {run_time_txt}" if run_time_txt else ""),
        "Combined 3D complete",
        100,
        {"display": "none"},
    )


def register_callbacks(app):
    for btn, graph in (("comb-copy-3d", "comb-figure-3d"), ("comb-copy-raptor3d", "comb-raptor-fig3d")):
        app.clientside_callback(
            ClientsideFunction(namespace="figtools", function_name="copy"),
            Output(f"{btn}-status", "children"),
            Input(btn, "n_clicks"),
            State(graph, "id"),
            prevent_initial_call=True,
        )
    app.clientside_callback(
        ClientsideFunction(namespace="figtools", function_name="copyChart"),
        Output("comb-copy-pore-shape-status", "children"),
        Input("comb-copy-pore-shape", "n_clicks"),
        State("comb-pore-shape-fig", "id"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("comb-vsweep-status", "children"),
        Input("comb-vsweep-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def start_velocity_sweep(n_clicks):
        if not n_clicks:
            return dash.no_update
        from .velocity_sweep import desktop_dir, sweep_already_running

        dest = os.path.join(desktop_dir(), "Raptor_5x5x14_V1300-1700_flucWsMTs")
        if sweep_already_running(dest):
            return f"Sweep already running. Folder: {dest}"
        os.makedirs(dest, exist_ok=True)
        log_path = os.path.join(dest, "sweep_log.txt")
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "integrated_gui.velocity_sweep", "--outdir", dest],
            cwd=ROOT,
            stdout=open(log_path, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        with open(os.path.join(dest, "sweep.pid"), "w", encoding="utf-8") as f:
            f.write(str(proc.pid))
        return (
            f"Sweep started (pid {proc.pid}). Nine velocities, no edge pores, no scan vectors. "
            f"Folder: {dest}"
        )

    @app.callback(
        Output("comb-raptor-lib-hint", "children"),
        Input("comb-raptor-lib-on", "value"),
        Input("comb-raptor-lib-eta", "value"),
        Input("comb-velocity", "value"),
    )
    def raptor_library_hint(lib_on, eta_source, velocity):
        if not (lib_on and "on" in lib_on):
            return "Library off: Raptor uses the bundled ULI_v1700 θ=0 series with fixed scales."
        try:
            v = float(velocity or 1300)
            c = mpl.resolve_condition({"eta_source": eta_source or "flucWs"}, v)
        except Exception as exc:
            return f"Melt-pool library unavailable: {exc}"
        note = "" if abs(c.velocity_mm_s - v) < 1e-6 else f" (nearest to {v:g} mm/s)"
        return (
            f"Raptor melt pool: V={c.velocity_mm_s:g} mm/s{note} · η_D={c.eta_d:.3g} · "
            f"η_C={c.eta_c:.3g} · depth shape={c.depth_shape:.3g} · "
            f"spectrum averaged over {len(c.width_files)} orientations."
        )

    @app.callback(
        Output("comb-grid-controls", "style"),
        Output("comb-cuboid-controls", "style"),
        Output("comb-stl-controls", "style"),
        Output("comb-raptor-controls", "style"),
        Output("comb-raptor-roi-controls", "style"),
        Input("comb-plate-mode", "value"),
    )
    def toggle_geom_mode(mode):
        hide, show = {"display": "none"}, {"display": "block"}
        if mode == "single_cuboid":
            return hide, show, hide, show, hide
        if mode == "stl":
            return hide, hide, show, show, show
        return show, hide, hide, hide, hide

    def _roi_status_text(roi: Optional[dict]) -> str:
        if not roi or not isinstance(roi, dict) or not roi.get("ok"):
            return (
                "No region selected — Raptor LoF runs on the real cross-section of the part "
                "(or cake slice) over a thin band of layers (see band height / layers)."
            )
        return (
            f"Raptor region: {float(roi.get('length_mm', 0)):.2f}×"
            f"{float(roi.get('width_mm', 0)):.2f}×{float(roi.get('height_mm', 0)):.2f} mm "
            f"at ({float(roi.get('cx_mm', 0)):.1f}, {float(roi.get('cy_mm', 0)):.1f}) · "
            f"saved {roi.get('saved_at', '')}"
        )

    @app.callback(
        Output("comb-raptor-roi-status", "children"),
        Output("comb-raptor-roi-tick", "disabled"),
        Output("app-status", "children", allow_duplicate=True),
        Input("comb-pick-raptor-roi", "n_clicks"),
        State("comb-stl-path", "data"),
        State("comb-stl-margin", "value"),
        State("comb-stl-max-height", "value"),
        State("comb-plate-w", "value"),
        State("comb-plate-h", "value"),
        prevent_initial_call=True,
    )
    def open_raptor_roi_picker(_n, stl_path, stl_margin, stl_max_height, plate_w, plate_h):
        if not stl_path or not os.path.isfile(str(stl_path)):
            return (
                "Need a valid STL path first (upload or set path).",
                True,
                "Raptor ROI: missing STL",
            )
        try:
            from .process_spawn import spawn_detached

            os.makedirs(PYVISTA_DIR, exist_ok=True)
            # Remove stale file so the poller notices a fresh save
            try:
                if os.path.isfile(RAPTOR_ROI_PATH):
                    os.remove(RAPTOR_ROI_PATH)
            except OSError:
                pass
            clip = float(stl_max_height) if stl_max_height not in (None, "") else 0.0
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            env.pop("PYVISTA_OFF_SCREEN", None)
            env.pop("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", None)
            env["PYVISTA_OFF_SCREEN"] = "false"
            spawn_detached(
                [
                    sys.executable,
                    "-m",
                    "integrated_gui.stl_raptor_roi",
                    "--stl",
                    str(stl_path),
                    "--out",
                    RAPTOR_ROI_PATH,
                    "--plate-w",
                    str(float(plate_w or 250)),
                    "--plate-h",
                    str(float(plate_h or 250)),
                    "--margin",
                    str(float(stl_margin if stl_margin is not None else 2.0)),
                    "--clip-height",
                    str(clip if clip > 0 else 0.0),
                ],
                cwd=ROOT,
                env=env,
                gui_window=True,
                log_path=os.path.join(PYVISTA_DIR, "raptor_roi_picker.log"),
            )
            return (
                "STL picker open — drag the pink handle (fixed 1×1×1 mm), click to jump, S to save.",
                False,
                "Raptor ROI picker launched",
            )
        except Exception as exc:
            return f"Failed to open STL picker: {exc}", True, "Raptor ROI picker failed"

    @app.callback(
        Output("comb-raptor-roi", "data"),
        Output("comb-raptor-roi-status", "children", allow_duplicate=True),
        Output("comb-raptor-roi-tick", "disabled", allow_duplicate=True),
        Output("app-status", "children", allow_duplicate=True),
        Input("comb-raptor-roi-tick", "n_intervals"),
        State("comb-raptor-roi", "data"),
        prevent_initial_call=True,
    )
    def poll_raptor_roi(_n, current):
        if not os.path.isfile(RAPTOR_ROI_PATH):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        try:
            roi = _load_json(RAPTOR_ROI_PATH)
        except Exception:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        if not roi.get("ok"):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        # Avoid rewriting identical payload every tick
        if isinstance(current, dict) and current.get("saved_at") == roi.get("saved_at"):
            return dash.no_update, dash.no_update, True, dash.no_update
        return (
            roi,
            _roi_status_text(roi),
            True,
            f"Raptor ROI saved ({float(roi.get('length_mm', 0)):.1f}×"
            f"{float(roi.get('width_mm', 0)):.1f}×{float(roi.get('height_mm', 0)):.1f} mm)",
        )

    @app.callback(
        Output("comb-raptor-roi", "data", allow_duplicate=True),
        Output("comb-raptor-roi-status", "children", allow_duplicate=True),
        Output("app-status", "children", allow_duplicate=True),
        Input("comb-clear-raptor-roi", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_raptor_roi(_n):
        try:
            if os.path.isfile(RAPTOR_ROI_PATH):
                os.remove(RAPTOR_ROI_PATH)
        except OSError:
            pass
        return (
            None,
            "No region selected — Raptor LoF runs on the real cross-section of the part "
            "(or cake slice) over a thin band of layers.",
            "Raptor ROI cleared",
        )

    @app.callback(
        Output("comb-preview-pores", "value"),
        Input("comb-totalpores", "value"),
        prevent_initial_call=True,
    )
    def sync_preview_pores_to_max_pores(max_pores):
        """Keep 3D downsample cap aligned with Max pores when a finite cap is set."""
        try:
            n = int(max_pores)
        except (TypeError, ValueError):
            return dash.no_update
        if n <= 0:
            # 0 = full ρ×V — leave the downsample control unchanged
            return dash.no_update
        return int(max(1, min(_PREVIEW_PORES_MAX, n)))

    @app.callback(
        Output("comb-cuboid-shape", "value"),
        Output("comb-cuboid-length", "value"),
        Output("comb-cuboid-width", "value"),
        Output("comb-cuboid-height", "value"),
        Input("comb-preset-5x5x14", "n_clicks"),
        Input("comb-preset-5x5x73", "n_clicks"),
        prevent_initial_call=True,
    )
    def apply_cuboid_preset(_n14, _n73):
        presets = {
            "comb-preset-5x5x14": (5.0, 5.0, 14.0),
            "comb-preset-5x5x73": (5.0, 5.0, 73.0),
        }
        dims = presets.get(dash.callback_context.triggered_id)
        if dims is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        return ("custom", *dims)

    @app.callback(
        Output("comb-batch-controls", "style"),
        Input("comb-scan-strategy", "value"),
    )
    def toggle_batch_controls(strategy):
        if str(strategy or "") == "batch":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("comb-stl-path", "data"),
        Output("comb-stl-path-input", "value"),
        Output("comb-stl-name", "children"),
        Input("comb-stl-upload", "contents"),
        Input("comb-stl-path-input", "value"),
        State("comb-stl-upload", "filename"),
    )
    def update_stl_path(upload_contents, path_text, upload_filename):
        triggered = dash.callback_context.triggered_id
        if triggered == "comb-stl-upload":
            if not upload_contents or not upload_filename:
                return dash.no_update, dash.no_update, dash.no_update
            import base64

            if not str(upload_filename).lower().endswith(".stl"):
                return None, dash.no_update, "Please upload a .stl file."
            _header, b64 = upload_contents.split(",", 1)
            raw = base64.b64decode(b64)
            upload_dir = os.path.join(WORK_DIR, "stl_uploads")
            os.makedirs(upload_dir, exist_ok=True)
            path = os.path.join(upload_dir, os.path.basename(upload_filename))
            with open(path, "wb") as f:
                f.write(raw)
            return path, path, f"Uploaded: {upload_filename} ({len(raw)/1024:.1f} KB)"

        path = (path_text or "").strip() or DEFAULT_STL_PATH
        if os.path.isfile(path):
            label = f"Using: {os.path.basename(path)}"
        else:
            label = f"STL not found: {path}"
        return path, dash.no_update, label

    @app.callback(
        Output("comb-scan-preview-modal", "is_open"),
        Output("comb-scan-preview-fig", "figure"),
        Output("comb-scan-preview-store", "data"),
        Output("comb-scan-layer-filter", "options"),
        Output("comb-scan-layer-filter", "value"),
        Output("comb-scan-preview-meta", "children"),
        Output("comb-scan-preview-status", "children"),
        Input("comb-scan-preview-btn", "n_clicks"),
        Input("comb-scan-preview-close", "n_clicks"),
        State("comb-scan-preview-modal", "is_open"),
        State("comb-plate-mode", "value"),
        State("comb-cuboid-shape", "value"),
        State("comb-cuboid-margin", "value"),
        State("comb-cuboid-height", "value"),
        State("comb-cuboid-length", "value"),
        State("comb-cuboid-width", "value"),
        State("comb-cube-height", "value"),
        State("comb-plate-w", "value"),
        State("comb-plate-h", "value"),
        State("comb-stl-path", "data"),
        State("comb-stl-margin", "value"),
        State("comb-stl-max-height", "value"),
        State("comb-scan-strategy", "value"),
        State("comb-n-edge", "value"),
        State("comb-hatch-mm", "value"),
        State("comb-layer-mm", "value"),
        State("comb-max-layers", "value"),
        State("comb-layer-rot", "value"),
        State("comb-base-angle", "value"),
        State("comb-batch-rows", "value"),
        State("comb-batch-cols", "value"),
        State("comb-seed", "value"),
        State("comb-use-process-scan", "value"),
        State("comb-stripe-mm", "value"),
        State("comb-hatch", "value"),
        State("comb-layer", "value"),
        State("comb-stl-sector", "value"),
        State("comb-stl-sector-start", "value"),
        prevent_initial_call=True,
    )
    def preview_scan_vectors(
        n_open,
        n_close,
        is_open,
        plate_mode,
        cuboid_shape,
        cuboid_margin,
        cuboid_height,
        cuboid_length,
        cuboid_width,
        cube_height,
        plate_w,
        plate_h,
        stl_path,
        stl_margin,
        stl_max_height,
        strategy,
        n_edge,
        hatch_mm,
        layer_mm,
        max_layers,
        layer_rot,
        base_angle,
        batch_rows,
        batch_cols,
        seed,
        use_process_scan,
        stripe_mm,
        proc_hatch_um,
        proc_layer_um,
        stl_sector,
        stl_sector_start,
    ):
        triggered = dash.callback_context.triggered_id
        empty = go.Figure(layout=dict(title="Scan preview", height=560))
        if triggered == "comb-scan-preview-close":
            return False, empty, None, [{"label": "All layers", "value": "all"}], "all", "", ""

        mode = str(plate_mode or "grid_cubes")
        if mode not in ("single_cuboid", "stl"):
            return (
                True,
                empty,
                None,
                [{"label": "All layers", "value": "all"}],
                "all",
                "",
                "Scan preview supports Centered cube/cuboid and Import STL.",
            )

        try:
            base = None
            if base_angle is not None and str(base_angle).strip() != "":
                base = float(base_angle)
            rot = float(layer_rot if layer_rot is not None else 67.0)
            strat = str(strategy or "stripes")
            # Preview always uses CMU/Raptor process hatch & layer (µm)
            use_proc = True
            hatch = float(proc_hatch_um or 140) * 1e-3
            layer_t = float(proc_layer_um or 30) * 1e-3
            stripe_w = float(stripe_mm or 10.0)
            # Thin display only — real Combined/Raptor runs scan every process layer
            n_lay = min(12, int(max_layers or 12))
            scan_note = ""
            br = int(batch_rows or 4)
            bc = int(batch_cols or 4)
            catalog = pd.DataFrame({ESD_COL: [25.0, 40.0, 60.0]})
            stl_mesh = None

            if mode == "stl":
                if not stl_path or not os.path.isfile(stl_path):
                    return (
                        True,
                        empty,
                        None,
                        [{"label": "All layers", "value": "all"}],
                        "all",
                        "",
                        "Set a valid STL path before previewing.",
                    )
                stl_h_cap = float(stl_max_height if stl_max_height is not None else 0.0)
                layout = make_plate_layout(
                    plate_width_mm=float(plate_w or 250),
                    plate_height_mm=float(plate_h or 250),
                    plate_mode="stl",
                    stl_path=stl_path,
                    stl_margin_mm=float(stl_margin if stl_margin is not None else 2.0),
                    stl_max_height_mm=stl_h_cap if stl_h_cap > 0 else None,
                    stl_sector_fraction=float(stl_sector or 1.0),
                    stl_sector_start_deg=float(stl_sector_start or 0.0),
                )
                stl_mesh = layout.stl_mesh
                m = layout.stl_meta
                xmin, xmax = float(m["xmin"]), float(m["xmax"])
                ymin, ymax = float(m["ymin"]), float(m["ymax"])
                height = float(m["zmax"] - m["zmin"])
                z_off = float(m["zmin"])
                if use_proc:
                    from .stl_slicer_scan import slice_scan_stl

                    n_total = max(1, int(np.ceil(height / max(layer_t, 1e-6))))
                    idx = np.unique(
                        np.linspace(0, n_total - 1, max(1, n_lay)).round().astype(int)
                    )
                    res = slice_scan_stl(
                        stl_mesh,
                        hatch_mm=hatch,
                        layer_mm=layer_t,
                        strategy=strat,
                        stripe_width_mm=stripe_w,
                        layer_rotation_deg=rot,
                        base_angle_deg=base,
                        batch_rows=br,
                        batch_cols=bc,
                        n_edge_pores=int(n_edge or 0),
                        size_catalog=catalog,
                        seed=int(seed or 42) + 17,
                        layer_indices=idx.tolist(),
                        n_preview_layers=len(idx),
                        max_preview_vectors_per_layer=1500,
                        include_contour=False,
                    )
                    vectors = None
                    vectors_df = res["vectors_df"]
                    edge_df = res["edge_df"]
                    scan_note = (
                        f" · process hatch {hatch * 1e3:.0f} µm / layer {layer_t * 1e3:.0f} µm"
                        f" · {len(idx)} of {n_total:,} layers shown (thinned for display)"
                    )
                else:
                    vectors = vectors_for_stl_mesh(
                        stl_mesh,
                        strategy=strat,
                        hatch_spacing_mm=hatch,
                        layer_thickness_mm=layer_t,
                        max_layers=n_lay,
                        base_angle_deg=base,
                        layer_rotation_deg=rot,
                        batch_rows=br,
                        batch_cols=bc,
                    )
                shape_label = "STL"
                clip_note = " clipped" if m.get("height_clipped") else ""
                size_txt = f"{xmax-xmin:.1f}×{ymax-ymin:.1f}×{height:.1f} mm{clip_note}"
            else:
                shape_mode = str(cuboid_shape or "custom")
                height = float(cuboid_height if cuboid_height is not None else (cube_height or 20.0))
                layout = make_plate_layout(
                    plate_width_mm=float(plate_w or 250),
                    plate_height_mm=float(plate_h or 250),
                    cube_size_mm=height,
                    cube_height_mm=height,
                    plate_mode="single_cuboid",
                    cuboid_margin_mm=float(cuboid_margin if cuboid_margin is not None else 5.0),
                    cuboid_fill_plate=(shape_mode == "fill"),
                    cuboid_length_mm=(
                        float(cuboid_length if cuboid_length is not None else 20.0)
                        if shape_mode == "custom"
                        else None
                    ),
                    cuboid_width_mm=(
                        float(cuboid_width if cuboid_width is not None else 20.0)
                        if shape_mode == "custom"
                        else None
                    ),
                )
                b = layout.builds[0]
                xmin = b.cx_mm - b.length_mm / 2.0
                xmax = b.cx_mm + b.length_mm / 2.0
                ymin = b.cy_mm - b.width_mm / 2.0
                ymax = b.cy_mm + b.width_mm / 2.0
                height = b.height_mm
                z_off = 0.0
                vectors = vectors_for_builds(
                    layout.builds,
                    strategy=strat,
                    hatch_spacing_mm=hatch,
                    layer_thickness_mm=layer_t,
                    max_layers=n_lay,
                    base_angle_deg=base,
                    layer_rotation_deg=rot,
                    batch_rows=br,
                    batch_cols=bc,
                    stripe_width_mm=stripe_w,
                )
                shape_label = b.shape
                size_txt = f"{b.length_mm:.1f}×{b.width_mm:.1f}×{b.height_mm:.1f} mm"

            if vectors is not None:
                edge_df, vectors_df = sample_edge_pores(
                    vectors,
                    catalog,
                    n_edge_pores=int(n_edge or 0),
                    seed=int(seed or 42) + 17,
                )
            fig = make_scan_preview_figure(
                vectors_df,
                edge_df,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                height_mm=height,
                layer_filter=None,
                title=f"Scan · {strat} · {shape_label}",
                stl_mesh=stl_mesh,
                z_offset=z_off,
            )
            layers = sorted(vectors_df["layer"].unique().tolist()) if len(vectors_df) else []
            opts = [{"label": "All layers", "value": "all"}] + [
                {"label": f"Layer {int(li)} only", "value": str(int(li))} for li in layers
            ]
            angles = (
                vectors_df.groupby("layer")["angle_deg"].first().to_dict()
                if len(vectors_df) and "angle_deg" in vectors_df.columns
                else {}
            )
            angle_txt = ", ".join(f"L{int(k)}={float(v):.0f}°" for k, v in sorted(angles.items()))
            batch_txt = f" · tiles {br}×{bc}" if strat == "batch" else ""
            meta = (
                f"{shape_label}: {size_txt} · {len(vectors_df)} vectors · {len(edge_df)} edge pores"
                + batch_txt
                + scan_note
                + (f" · angles {angle_txt}" if angle_txt else "")
                + " · green=START · red◆=END"
            )
            store = {
                "vectors": vectors_df.to_dict("list"),
                "edges": edge_df.to_dict("list") if len(edge_df) else {},
                "bounds": {
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "height": height,
                    "z_offset": z_off,
                },
                "strategy": strat,
                "shape": shape_label,
                "stl_path": stl_path if mode == "stl" else None,
                "stl_sector": float(stl_sector or 1.0),
                "stl_sector_start": float(stl_sector_start or 0.0),
            }
            return True, fig, store, opts, "all", meta, "Preview ready"
        except Exception as exc:
            return (
                True,
                empty,
                None,
                [{"label": "All layers", "value": "all"}],
                "all",
                "",
                f"Preview failed: {exc}",
            )

    @app.callback(
        Output("comb-scan-preview-fig", "figure", allow_duplicate=True),
        Input("comb-scan-layer-filter", "value"),
        State("comb-scan-preview-store", "data"),
        prevent_initial_call=True,
    )
    def filter_scan_preview_layer(layer_val, store):
        if not store:
            return dash.no_update
        vectors_df = pd.DataFrame(store.get("vectors") or {})
        edges_raw = store.get("edges") or {}
        edge_df = pd.DataFrame(edges_raw) if edges_raw else pd.DataFrame()
        b = store["bounds"]
        layer_filter = None if (not layer_val or layer_val == "all") else int(layer_val)
        stl_mesh = None
        stl_path = store.get("stl_path")
        if stl_path and os.path.isfile(stl_path):
            try:
                layout = make_plate_layout(
                    plate_width_mm=250,
                    plate_height_mm=250,
                    plate_mode="stl",
                    stl_path=stl_path,
                    stl_sector_fraction=float(store.get("stl_sector") or 1.0),
                    stl_sector_start_deg=float(store.get("stl_sector_start") or 0.0),
                )
                stl_mesh = layout.stl_mesh
            except Exception:
                stl_mesh = None
        return make_scan_preview_figure(
            vectors_df,
            edge_df if len(edge_df) else None,
            xmin=b["xmin"],
            xmax=b["xmax"],
            ymin=b["ymin"],
            ymax=b["ymax"],
            height_mm=b["height"],
            layer_filter=layer_filter,
            title=f"Scan · {store.get('strategy')} · {store.get('shape')}",
            stl_mesh=stl_mesh,
            z_offset=float(b.get("z_offset", 0.0)),
        )

    @app.callback(
        Output("comb-nu-cache-status", "children"),
        Input("comb-nu-status-tick", "n_intervals"),
    )
    def refresh_comb_nu_status(_tick):
        st = cache_status()
        color = "success" if st["ready"] else "warning"
        return dbc.Alert(st["message"], color=color, className="mb-0 py-2")

    @app.callback(
        Output("comb-prefit-status", "children"),
        Output("comb-nu-cache-status", "children", allow_duplicate=True),
        Output("app-status", "children", allow_duplicate=True),
        Input("comb-prefit-nu", "n_clicks"),
        running=[
            (Output("comb-prefit-nu", "disabled"), True, False),
            (
                Output("comb-prefit-status", "children"),
                "Fitting NU (1–3 min)…",
                dash.no_update,
            ),
            (
                Output("app-status", "children"),
                "Pre-fitting NU…",
                dash.no_update,
            ),
        ],
        prevent_initial_call=True,
    )
    def run_comb_prefit(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        status = run_prefit_subprocess()
        st = cache_status()
        color = "success" if st["ready"] else "warning"
        alert = dbc.Alert(st["message"], color=color, className="mb-0 py-2")
        if status.get("ok"):
            msg = status.get("cache", {}).get("message") or "NU cache ready."
            return f"Done — {msg}", alert, "NU pre-fit complete"
        err = (status.get("log") or status.get("error") or "Pre-fit failed")[-500:]
        return f"Failed — {err}", alert, "NU pre-fit failed"

    @app.callback(
        Output("comb-compute-hint", "children"),
        Input("comb-compute-device", "value"),
    )
    def update_compute_hint(mode):
        # Avoid blocking the Dash thread on a cold CUDA init; show a light hint first.
        try:
            from .compute_backend import resolve_device

            info = resolve_device(mode or "auto")
            if info.cuda_available:
                return f"{info.detail}. Coupon-cube pore placement uses CUDA when GPU is selected."
            return f"{info.detail}. Install an NVIDIA driver / CUDA-capable GPU for acceleration."
        except Exception:
            return "Compute device: Auto prefers GPU when CUDA is available."

    @app.callback(
        Output("comb-job-store", "data"),
        Output("comb-job-tick", "disabled"),
        Output("comb-run", "disabled"),
        Output("comb-run-status", "children"),
        Output("comb-run-progress", "value"),
        Output("comb-run-progress", "style"),
        Output("comb-log-output", "value"),
        Output("app-status", "children", allow_duplicate=True),
        Output("comb-results-output", "children", allow_duplicate=True),
        Output("comb-figure-3d", "figure", allow_duplicate=True),
        Output("comb-figure-2d", "children", allow_duplicate=True),
        Output("comb-download-btn", "disabled", allow_duplicate=True),
        Output("comb-open-pyvista", "disabled", allow_duplicate=True),
        Output("comb-refresh-preview", "disabled", allow_duplicate=True),
        Output("comb-open-raptor-pv", "disabled", allow_duplicate=True),
        Input("comb-run", "n_clicks"),
        State("comb-plate-mode", "value"),
        State("comb-n-strips", "value"),
        State("comb-n-rows", "value"),
        State("comb-n-cols", "value"),
        State("comb-cube-size", "value"),
        State("comb-cube-height", "value"),
        State("comb-cuboid-margin", "value"),
        State("comb-cuboid-shape", "value"),
        State("comb-cuboid-height", "value"),
        State("comb-cuboid-length", "value"),
        State("comb-cuboid-width", "value"),
        State("comb-stl-path", "data"),
        State("comb-stl-margin", "value"),
        State("comb-stl-max-height", "value"),
        State("comb-plate-w", "value"),
        State("comb-plate-h", "value"),
        State("comb-seed", "value"),
        State("comb-totalpores", "value"),
        State("comb-mean-porosity", "value"),
        State("comb-enable-scan", "value"),
        State("comb-scan-strategy", "value"),
        State("comb-n-edge", "value"),
        State("comb-hatch-mm", "value"),
        State("comb-layer-mm", "value"),
        State("comb-max-layers", "value"),
        State("comb-layer-rot", "value"),
        State("comb-base-angle", "value"),
        State("comb-batch-rows", "value"),
        State("comb-batch-cols", "value"),
        State("comb-pore-models", "value"),
        State("comb-power", "value"),
        State("comb-velocity", "value"),
        State("comb-hatch", "value"),
        State("comb-layer", "value"),
        State("comb-compute-device", "value"),
        State("comb-use-raptor-lof", "value"),
        State("comb-gas-flow", "value"),
        State("comb-raptor-roi", "data"),
        State("comb-stripe-mm", "value"),
        State("comb-use-process-scan", "value"),
        State("comb-make-spatter-anim", "value"),
        State("comb-stl-sector", "value"),
        State("comb-stl-sector-start", "value"),
        State("comb-raptor-band-z", "value"),
        State("comb-raptor-band-layers", "value"),
        State("comb-raptor-stl-domain", "value"),
        State("comb-raptor-lib-on", "value"),
        State("comb-raptor-lib-eta", "value"),
        prevent_initial_call=True,
    )
    def start_combined(
        n_clicks,
        plate_mode,
        n_strips,
        n_rows,
        n_cols,
        cube_size,
        cube_height,
        cuboid_margin,
        cuboid_shape,
        cuboid_height,
        cuboid_length,
        cuboid_width,
        stl_path,
        stl_margin,
        stl_max_height,
        plate_w,
        plate_h,
        seed,
        totalpores,
        mean_porosity,
        enable_scan,
        scan_strategy,
        n_edge,
        hatch_mm,
        layer_mm,
        max_layers,
        layer_rot,
        base_angle,
        batch_rows,
        batch_cols,
        pore_models,
        power,
        velocity,
        hatch,
        layer,
        compute_device,
        use_raptor_lof,
        gas_flow,
        raptor_roi,
        stripe_mm,
        use_process_scan,
        make_spatter_anim,
        stl_sector,
        stl_sector_start,
        raptor_band_z,
        raptor_band_layers,
        raptor_stl_domain,
        raptor_lib_on,
        raptor_lib_eta,
    ):
        if not n_clicks:
            return dash.no_update
        prog_style = {"height": "0.7rem", "display": "block"}
        loading_results = dbc.Alert(
            [
                html.Div(
                    [
                        dbc.Spinner(
                            size="sm",
                            color="light",
                            spinner_class_name="me-2",
                        ),
                        html.Strong("Results are loading…"),
                    ],
                    className="d-flex align-items-center",
                ),
                html.Div(
                    "Generating the plate pore field. The 3D / 2D views will appear when ready.",
                    className="small mt-2 mb-0",
                ),
            ],
            color="info",
            className="mb-0",
        )
        loading_fig = go.Figure(
            layout=dict(
                title="Results loading…",
                height=620,
                paper_bgcolor="white",
                annotations=[
                    dict(
                        text="Generating plate pore field…",
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=18, color="#5A6A7A"),
                    )
                ],
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
        )
        loading_2d = html.Div(
            [
                dbc.Spinner(color="primary", size="md"),
                html.Div("2D map / Raptor view loading…", className="text-muted mt-3"),
            ],
            className="text-center p-5",
        )
        try:
            param_dict = _load_json(EXAMPLE_PARAM_PATH)
            param_dict["Power (W)"] = float(power or 370)
            param_dict["Velocity (mm/s)"] = float(velocity or 1300)
            param_dict["Hatch Spacing (um)"] = float(hatch or 140)
            param_dict["Layer Thickness (um)"] = float(layer or 30)
            props = _load_json(MATERIAL_PATH)
            for k, v in param_dict.items():
                if isinstance(v, (int, float)):
                    props[k] = v

            os.makedirs(WORK_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            outdir = os.path.join(WORK_DIR, stamp)
            os.makedirs(outdir, exist_ok=True)

            mode = str(plate_mode or "grid_cubes")
            if mode == "stl" and not stl_path:
                return (
                    None,
                    True,
                    False,
                    "Need STL file",
                    0,
                    {"height": "0.7rem", "display": "none"},
                    "STL mode selected but no file uploaded.",
                    "Combined: missing STL",
                    dbc.Alert("Need an STL file before Generate.", color="warning"),
                    dash.no_update,
                    dash.no_update,
                    True,
                    True,
                    True,
                    True,
                )

            n_rows_i = int(n_rows or 19)
            n_cols_i = int(n_cols or 3)
            n_strips_i = int(n_strips or 4)
            shape_mode = str(cuboid_shape or "custom")
            cuboid_h = float(
                cuboid_height
                if cuboid_height is not None
                else (cube_height or cube_size or 20.0)
            )
            cuboid_L = float(cuboid_length if cuboid_length is not None else 20.0)
            cuboid_W = float(cuboid_width if cuboid_width is not None else 20.0)
            stl_h_cap = float(stl_max_height if stl_max_height is not None else 0.0)
            base = None
            if base_angle is not None and str(base_angle).strip() != "":
                base = float(base_angle)
            cfg = {
                "plate_mode": mode,
                "n_rows": n_rows_i,
                "n_cols": n_cols_i,
                "n_strips": n_strips_i,
                "cube_size": float(cube_size or 10.0),
                "cube_height": cuboid_h if mode == "single_cuboid" else float(cube_height or cube_size or 10.0),
                "cuboid_margin_mm": float(cuboid_margin if cuboid_margin is not None else 5.0),
                "cuboid_fill_plate": shape_mode == "fill",
                "cuboid_length_mm": (
                    cuboid_L if mode == "single_cuboid" and shape_mode == "custom" else None
                ),
                "cuboid_width_mm": (
                    cuboid_W if mode == "single_cuboid" and shape_mode == "custom" else None
                ),
                "stl_path": stl_path,
                "stl_margin_mm": float(stl_margin if stl_margin is not None else 2.0),
                "stl_max_height_mm": stl_h_cap if mode == "stl" and stl_h_cap > 0 else None,
                "stl_sector_fraction": float(stl_sector or 1.0) if mode == "stl" else 1.0,
                "stl_sector_start_deg": float(stl_sector_start or 0.0),
                "raptor_band_z_mm": raptor_band_z if raptor_band_z not in (None, "") else None,
                "raptor_band_layers": int(raptor_band_layers or 15),
                "raptor_stl_domain": str(raptor_stl_domain or "volume"),
                "plate_w": float(plate_w or 250),
                "plate_h": float(plate_h or 250),
                "seed": int(seed or 42),
                "max_pores": int(totalpores) if totalpores is not None else 0,
                "totalpores": int(totalpores) if totalpores is not None else 0,
                "mean_porosity": float(mean_porosity if mean_porosity is not None else 0.02),
                "enable_scan_vectors": bool(enable_scan and "on" in (enable_scan or [])),
                "scan_strategy": scan_strategy or "stripes",
                "stripe_width_mm": float(stripe_mm or 10.0),
                # Always CMU/Raptor process hatch & layer (µm); whole part for STL / cuboid
                "use_process_scan_params": True,
                "make_spatter_anim": bool(make_spatter_anim and "on" in make_spatter_anim),
                "scan_full_part": mode in ("stl", "single_cuboid"),
                "n_edge_pores": int(n_edge or 0),
                "hatch_spacing_mm": float(param_dict.get("Hatch Spacing (um)", 140)) * 1e-3,
                "layer_thickness_mm": float(param_dict.get("Layer Thickness (um)", 30)) * 1e-3,
                "max_scan_layers": int(max_layers or 99999),
                "layer_rotation_deg": float(layer_rot if layer_rot is not None else 67.0),
                "base_angle_deg": base,
                "batch_rows": int(batch_rows or 4),
                "batch_cols": int(batch_cols or 4),
                "pore_models": pore_models or ["lof", "keyhole", "gas"],
                "param_dict": param_dict,
                "props": props,
                "compute_device": (compute_device or "auto"),
                "use_raptor_lof": bool(
                    mode in ("single_cuboid", "stl")
                    and use_raptor_lof
                    and "on" in (use_raptor_lof or [])
                ),
                "raptor_roi": (
                    raptor_roi
                    if mode == "stl" and isinstance(raptor_roi, dict) and raptor_roi.get("ok")
                    else None
                ),
                "raptor_library": (
                    {
                        "dir": mpl.DEFAULT_LIBRARY_DIR,
                        "eta_source": raptor_lib_eta or "flucWs",
                        "theta_mode": "averaged",
                        "angle_offset_deg": 0.0,
                        "use_eta": True,
                        "use_depth_shape": True,
                        "lambda_mean": mpl.LAMBDA_MEAN,
                    }
                    if raptor_lib_on and "on" in raptor_lib_on
                    else None
                ),
                "gas_flow": str(gas_flow or "+x"),
                "gas_flow_sharpness": 2.0,
                "flow_perturbation": 0.65 if mode == "single_cuboid" else 0.0,
                "spatter_fraction": 0.0,
                "raptor_rve_edge_m": 1.0e-3 if mode == "stl" else 5.0e-4,
                "raptor_voxel_m": 5.0e-6,
                "raptor_n_modes": 50,
            }
            cfg_path = os.path.join(outdir, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            # Seed progress so the first poll has something to show
            with open(os.path.join(outdir, "progress.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stage": "queued",
                        "frac": 0.01,
                        "pct": 1.0,
                        "message": "Queued Combined worker…",
                        "step": 0,
                        "n_steps": 6,
                        "elapsed_s": 0.0,
                        "eta_s": None,
                    },
                    f,
                )

            stdout_path = os.path.join(outdir, "worker.stdout.log")
            stderr_path = os.path.join(outdir, "worker.stderr.log")
            env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "NASA_COMPUTE_DEVICE": str(compute_device or "auto"),
            }
            creationflags = 0
            if os.name == "nt":
                # New process group so console Ctrl+C from PyVista does not kill the worker
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            with open(stdout_path, "w", encoding="utf-8") as out_f, open(
                stderr_path, "w", encoding="utf-8"
            ) as err_f:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "integrated_gui.combined_worker",
                        "--config",
                        cfg_path,
                        "--outdir",
                        outdir,
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=out_f,
                    stderr=err_f,
                    creationflags=creationflags,
                )
            job = {"outdir": outdir, "pid": int(proc.pid), "t0": time.time()}
            return (
                job,
                False,
                True,
                "Starting… · results loading…",
                1,
                prog_style,
                "Combined worker started.\n"
                f"Logs: {stdout_path}\n"
                "1) Geometry\n2) NU cache\n3) CMU ρ × volume\n"
                "4) NU variation + disperse\n5) Optional scan/edge\n6) Export / plots\n"
                "ETA updates every second.",
                "Combined running…",
                loading_results,
                loading_fig,
                loading_2d,
                True,
                True,
                True,
                True,
            )
        except Exception:
            tb = traceback.format_exc()
            return (
                None,
                True,
                False,
                "Failed to start",
                0,
                {"height": "0.7rem", "display": "none"},
                tb,
                "Combined 3D failed",
                dbc.Alert("Failed to start Generate — see Log.", color="danger"),
                go.Figure(layout=dict(title="3D plot unavailable", height=620)),
                empty_figure_placeholder("Field plot unavailable."),
                True,
                True,
                True,
                True,
            )

    @app.callback(
        Output("comb-log-output", "value", allow_duplicate=True),
        Output("comb-results-output", "children"),
        Output("comb-figure-3d", "figure"),
        Output("comb-figure-2d", "children"),
        Output("comb-download-btn", "disabled"),
        Output("comb-open-pyvista", "disabled"),
        Output("comb-refresh-preview", "disabled"),
        Output("comb-open-raptor-pv", "disabled"),
        Output("comb-run-status", "children", allow_duplicate=True),
        Output("app-status", "children", allow_duplicate=True),
        Output("comb-run-progress", "value", allow_duplicate=True),
        Output("comb-run-progress", "style", allow_duplicate=True),
        Output("comb-job-store", "data", allow_duplicate=True),
        Output("comb-job-tick", "disabled", allow_duplicate=True),
        Output("comb-run", "disabled", allow_duplicate=True),
        Input("comb-job-tick", "n_intervals"),
        State("comb-job-store", "data"),
        State("comb-preview-pores", "value"),
        prevent_initial_call=True,
    )
    def poll_combined(_n, job, preview_pores):
        if not job or not isinstance(job, dict):
            return (dash.no_update,) * 15

        outdir = job.get("outdir")
        pid = job.get("pid")
        t0 = float(job.get("t0") or time.time())
        if not outdir:
            return (dash.no_update,) * 15

        status_path = os.path.join(outdir, "status.json")
        progress_path = os.path.join(outdir, "progress.json")
        log_path = os.path.join(outdir, "worker.log")
        alive = _pid_alive(pid)
        preview_n = _clamp_preview_pores(preview_pores)

        if os.path.exists(status_path):
            # Prefer finalizing as soon as status.json is written
            try:
                outs = _finalize_combined_outdir(outdir, preview_pores=preview_n)
            except Exception:
                tb = traceback.format_exc()
                empty_fig = go.Figure(layout=dict(title="3D plot unavailable", height=620))
                outs = (
                    tb,
                    dbc.Alert("Combined run failed while loading results.", color="danger"),
                    empty_fig,
                    empty_figure_placeholder("Field plot unavailable."),
                    True,
                    True,
                    True,
                    True,
                    "Failed — see Log",
                    "Combined 3D failed",
                    100,
                    {"height": "0.7rem", "display": "none"},
                )
            # outs: log, results, fig3d, fig2d, dl, pv, refresh, raptor_pv, status, app, pct, style
            return (*outs, None, True, False)

        # Still running — update ETA / progress
        prog = {}
        if os.path.exists(progress_path):
            try:
                prog = _load_json(progress_path)
            except Exception:
                prog = {}
        elapsed = time.time() - t0
        if not prog:
            prog = {
                "message": "Running Combined worker…",
                "pct": min(95.0, 5.0 + elapsed),
                "eta_s": None,
                "elapsed_s": elapsed,
                "step": 0,
                "n_steps": 6,
            }
        status_txt = _progress_status_text(prog, elapsed_fallback=elapsed)
        pct = float(prog.get("pct") or 0)
        log_tail = ""
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
                log_tail = "\n".join(lines[-40:])
            except Exception:
                log_tail = ""
        if not alive and not os.path.exists(status_path):
            # Process died unexpectedly — surface stderr if present
            err_path = os.path.join(outdir, "worker.stderr.log")
            err_tail = ""
            if os.path.isfile(err_path):
                try:
                    with open(err_path, "r", encoding="utf-8", errors="replace") as f:
                        err_tail = f.read().strip()[-2500:]
                except Exception:
                    err_tail = ""
            detail = log_tail or err_tail or "Worker exited without status.json."
            if err_tail and err_tail not in detail:
                detail = f"{detail}\n\n--- stderr ---\n{err_tail}"
            empty_fig = go.Figure(layout=dict(title="3D plot unavailable", height=620))
            return (
                detail,
                dbc.Alert(
                    "Combined worker stopped unexpectedly (often from a console Ctrl+C "
                    "while opening PyVista). Re-run Generate; logs are in the work folder.",
                    color="danger",
                ),
                empty_fig,
                empty_figure_placeholder("Field plot unavailable."),
                True,
                True,
                True,
                True,
                "Failed — worker exited",
                "Combined 3D failed",
                100,
                {"height": "0.7rem", "display": "none"},
                None,
                True,
                False,
            )

        return (
            log_tail or dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            status_txt,
            f"Combined · {pct:.0f}% · {_format_eta(prog.get('eta_s'))}",
            pct,
            {"height": "0.7rem", "display": "block"},
            job,
            False,
            True,
        )

    @app.callback(
        Output("comb-csv-download", "data"),
        Input("comb-download-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_csv(_n):
        if _LAST_CSV_PATH and os.path.isfile(_LAST_CSV_PATH):
            run_dir = os.path.dirname(_LAST_CSV_PATH)
            extras = [
                os.path.join(run_dir, n)
                for n in (
                    "process_parameters.csv",
                    "process_parameters.json",
                    "summary.json",
                    os.path.join("raptor_rve", "lof_pore_shapes.csv"),
                )
                if os.path.isfile(os.path.join(run_dir, n))
            ]
            if not extras:
                return dcc.send_file(_LAST_CSV_PATH)
            import io
            import zipfile

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(_LAST_CSV_PATH, os.path.basename(_LAST_CSV_PATH))
                for p in extras:
                    zf.write(p, os.path.basename(p))
            return dcc.send_bytes(buf.getvalue(), "combined_model_pores_and_process.zip")
        if _LAST_CSV:
            return dict(content=_LAST_CSV, filename="combined_plate_pores.csv")
        return dash.no_update

    @app.callback(
        Output("comb-log-output", "value", allow_duplicate=True),
        Output("comb-results-output", "children", allow_duplicate=True),
        Output("comb-figure-3d", "figure", allow_duplicate=True),
        Output("comb-figure-2d", "children", allow_duplicate=True),
        Output("comb-run-status", "children", allow_duplicate=True),
        Output("app-status", "children", allow_duplicate=True),
        Input("comb-refresh-preview", "n_clicks"),
        State("comb-preview-pores", "value"),
        prevent_initial_call=True,
    )
    def refresh_preview(_n, preview_pores):
        if not _LAST_CSV_PATH or not os.path.isfile(_LAST_CSV_PATH):
            return (
                dash.no_update,
                dbc.Alert("No Combined result yet — run Generate first.", color="warning"),
                dash.no_update,
                dash.no_update,
                "No result to refresh",
                "Combined: nothing to preview",
            )
        outdir = os.path.dirname(_LAST_CSV_PATH)
        try:
            outs = _finalize_combined_outdir(outdir, preview_pores=_clamp_preview_pores(preview_pores))
            # outs: 0 log, 1 results, 2 fig3d, 3 fig2d, … 8 status, 9 app
            return outs[0], outs[1], outs[2], outs[3], outs[8], outs[9]
        except Exception:
            tb = traceback.format_exc()
            return (
                tb,
                dbc.Alert("Preview refresh failed — see Log.", color="danger"),
                go.Figure(layout=dict(title="3D plot unavailable", height=620)),
                empty_figure_placeholder("Field plot unavailable."),
                "Preview refresh failed",
                "Combined preview failed",
            )

    @app.callback(
        Output("comb-pyvista-status", "children", allow_duplicate=True),
        Input("comb-open-raptor-pv", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_raptor_rve_window(_n):
        rap_dir = _LAST_RAPTOR_DIR
        if not rap_dir and _LAST_CSV_PATH:
            cand = os.path.join(os.path.dirname(_LAST_CSV_PATH), "raptor_rve")
            if os.path.isdir(cand):
                rap_dir = cand
        if not rap_dir or not os.path.isdir(rap_dir):
            return (
                "No Raptor RVE for this run — use Centered cube/cuboid or Import STL "
                "with “Use Raptor LoF”, then Generate."
            )
        vti = os.path.join(rap_dir, "porosity_map.vti")
        if not os.path.isfile(vti):
            vti = os.path.join(rap_dir, "rve.vti")
        if not os.path.isfile(vti):
            return "Raptor VTI not found in raptor_rve/."
        try:
            from .process_spawn import spawn_detached

            raptor_src = os.path.join(ROOT, "Raptor", "src")
            code = (
                "import os\n"
                "import sys\n"
                f"sys.path.insert(0, r'''{raptor_src}''')\n"
                "os.environ['PYVISTA_OFF_SCREEN'] = 'false'\n"
                "os.environ.pop('VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN', None)\n"
                "import pyvista as pv\n"
                "pv.OFF_SCREEN = False\n"
                f"sys.path.insert(0, r'''{ROOT}''')\n"
                "from integrated_gui.raptor_view import show_interactive\n"
                f"show_interactive(r'''{vti}''')\n"
            )
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("PYVISTA_OFF_SCREEN", "VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN")
            }
            env["PYVISTA_OFF_SCREEN"] = "false"
            env["PYTHONIOENCODING"] = "utf-8"
            spawn_detached(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
                gui_window=True,
                log_path=os.path.join(PYVISTA_DIR, "raptor_viewer_spawn.log"),
            )
            return (
                "Opened Raptor whole-build viewer: left = melted phase for the full build "
                "(drag the blue plane to slice), right = every LoF pore."
            )
        except Exception as exc:
            return f"Failed to open Raptor RVE viewer: {exc}"

    @app.callback(
        Output("comb-pyvista-status", "children"),
        Output("comb-viewer-watch", "data"),
        Output("comb-viewer-tick", "disabled"),
        Input("comb-open-pyvista", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_pyvista_window(_n):
        csv_path = os.path.join(PYVISTA_DIR, "latest_pores.csv")
        view_csv = os.path.join(PYVISTA_DIR, "latest_pores_view.csv")
        geom_path = os.path.join(PYVISTA_DIR, "latest_geom.json")
        if (not os.path.exists(csv_path) and not os.path.exists(view_csv)) or not os.path.exists(
            geom_path
        ):
            return "No saved scene yet — run Generate first.", None, True
        try:
            from .process_spawn import spawn_detached

            # Build a quick view sample if Generate ran before this optimization
            if not os.path.exists(view_csv) and os.path.exists(csv_path):
                try:
                    full = pd.read_csv(
                        csv_path,
                        usecols=lambda c: c
                        in {
                            "x_mm",
                            "y_mm",
                            "z_mm",
                            "pore_source",
                            "pore_family",
                            "radius_mm",
                        },
                        low_memory=False,
                    )
                    n_total = len(full)
                    view = full.sample(min(4000, n_total), random_state=0) if n_total else full
                    view.to_csv(view_csv, index=False)
                    with open(
                        os.path.join(PYVISTA_DIR, "latest_view_meta.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(
                            {
                                "n_total": int(n_total),
                                "n_view": int(len(view)),
                                "view_max_points": 4000,
                            },
                            f,
                        )
                except Exception:
                    pass

            n_msg = ""
            meta_path = os.path.join(PYVISTA_DIR, "latest_view_meta.json")
            if os.path.isfile(meta_path):
                try:
                    meta = _load_json(meta_path)
                    n_msg = (
                        f" Showing {meta.get('n_view', '?')} of "
                        f"{meta.get('n_total', '?')} pores."
                    )
                except Exception:
                    n_msg = ""

            # Clear previous status so the poller sees a fresh launch
            for name in ("viewer_status.json", "viewer.log"):
                try:
                    os.remove(os.path.join(PYVISTA_DIR, name))
                except OSError:
                    pass

            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            env.pop("PYVISTA_OFF_SCREEN", None)
            env.pop("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", None)
            env["PYVISTA_OFF_SCREEN"] = "false"
            proc = spawn_detached(
                [
                    sys.executable,
                    "-m",
                    "integrated_gui.view_pores",
                    "--dir",
                    PYVISTA_DIR,
                    "--max-points",
                    "4000",
                ],
                cwd=ROOT,
                env=env,
                gui_window=True,
                log_path=os.path.join(PYVISTA_DIR, "viewer_spawn.log"),
            )
            return (
                "Launching PyVista window (no console)… check the taskbar / Alt+Tab."
                f"{n_msg}",
                {"pid": int(proc.pid), "t0": time.time()},
                False,
            )
        except Exception as exc:
            return f"Failed to open PyVista window: {exc}", None, True

    @app.callback(
        Output("comb-pyvista-status", "children", allow_duplicate=True),
        Output("comb-viewer-tick", "disabled", allow_duplicate=True),
        Input("comb-viewer-tick", "n_intervals"),
        State("comb-viewer-watch", "data"),
        prevent_initial_call=True,
    )
    def poll_pyvista_status(_n, watch):
        status_path = os.path.join(PYVISTA_DIR, "viewer_status.json")
        if os.path.isfile(status_path):
            try:
                st = _load_json(status_path)
                stage = str(st.get("stage") or "")
                msg = str(st.get("message") or "")
                if stage == "error":
                    return f"PyVista failed: {msg}", True
                if stage == "showing":
                    return (
                        f"PyVista window open — {msg} (bring it to the front if hidden).",
                        False,
                    )
                if stage == "closed":
                    return "PyVista window closed.", True
                if stage == "loading":
                    return f"PyVista: {msg}", False
            except Exception:
                pass
        if watch and isinstance(watch, dict):
            elapsed = time.time() - float(watch.get("t0") or time.time())
            if elapsed > 45:
                log_path = os.path.join(PYVISTA_DIR, "viewer.log")
                hint = ""
                if os.path.isfile(log_path):
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                            hint = f.read().strip()[-400:]
                    except Exception:
                        hint = ""
                return (
                    "PyVista is taking too long. "
                    + (f"Log: {hint}" if hint else "Check Task Manager for python.exe."),
                    True,
                )
            return f"Waiting for PyVista… ({elapsed:.0f}s)", False
        return dash.no_update, True

