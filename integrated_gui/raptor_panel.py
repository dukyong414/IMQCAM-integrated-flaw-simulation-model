"""ORNL Raptor tab: lack-of-fusion porosity RVE (embedded in NU Spatial Field)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Optional

import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .headless import configure_headless_plotting
from . import meltpool_library as mpl

configure_headless_plotting()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK_DIR = os.path.join(os.path.dirname(__file__), "assets", "raptor_work")
DEFAULT_MELT = os.path.join(
    ROOT, "Raptor", "examples", "data", "meltPoolData", "ULI_v1700_theta0_widths.txt"
)

_LAST_MORPH_CSV: Optional[str] = None


def _fmt_num(x: float, sig: str = ".6g") -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return format(float(x), sig)
    except Exception:
        return "—"


def _lof_compare_block(cmp: dict) -> list:
    """Dash children: unified CMU vs Raptor metrics."""
    cmu = cmp.get("cmu") or {}
    rap = cmp.get("raptor") or {}
    rows = [
        ("Volume porosity φ (%)", _fmt_num(cmu.get("volume_porosity_pct")), _fmt_num(rap.get("volume_porosity_pct")), _fmt_num(cmp.get("ratio_phi_raptor_over_cmu"), ".3g")),
        ("Number density ρ (pores/mm³)", _fmt_num(cmu.get("lof_density_per_mm3")), _fmt_num(rap.get("density_per_mm3")), _fmt_num(cmp.get("ratio_rho_raptor_over_cmu"), ".3g")),
        (
            f"Pore count in RVE (V={_fmt_num(cmp.get('rve_volume_mm3'), '.4g')} mm³)",
            _fmt_num(cmp.get("cmu_n_expected_in_rve"), ".4g"),
            str(rap.get("n_defects", "—")),
            _fmt_num(cmp.get("ratio_count_raptor_over_cmu"), ".3g"),
        ),
    ]
    header = html.Tr(
        [
            html.Th("Metric"),
            html.Th("CMU LoF"),
            html.Th("Raptor"),
            html.Th("Raptor/CMU"),
        ]
    )
    body = [
        html.Tr([html.Td(a), html.Td(b), html.Td(c), html.Td(d)])
        for a, b, c, d in rows
    ]
    note = html.Small(
        [
            "Primary fair compare: ",
            html.Strong("volume porosity φ (%)"),
            ". CMU φ = ρ·⟨sphere volume from ESD⟩. ",
            f"CMU mean ESD ≈ {_fmt_num(cmu.get('mean_esd_um'), '.3g')} µm. ",
            "ρ and pore-count in a tiny RVE are noisy.",
        ],
        className="text-muted d-block mt-2",
    )
    return [
        dbc.Table(
            [html.Thead(header), html.Tbody(body)],
            bordered=True,
            size="sm",
            className="mb-0 bg-white",
        ),
        note,
    ]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


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
    return f"~{m}m {sec:02d}s remaining"


def _progress_text(prog: Dict[str, Any], elapsed: float = 0.0) -> str:
    msg = str(prog.get("message") or "Running…")
    step = prog.get("step")
    n_steps = prog.get("n_steps")
    pct = prog.get("pct")
    eta = prog.get("eta_s")
    if eta is None and pct and float(pct) >= 2.0 and elapsed > 0.5:
        eta = max(0.0, elapsed / (float(pct) / 100.0) - elapsed)
    parts = []
    if step is not None and n_steps:
        parts.append(f"Step {step}/{n_steps}")
    if pct is not None:
        parts.append(f"{float(pct):.0f}%")
    parts.append(_format_eta(eta))
    parts.append(f"elapsed {float(elapsed):.0f}s")
    return f"{msg} · " + " · ".join(parts)


def _slice_figure(preview_path: str, summary: dict) -> go.Figure:
    """Phase map in hatch-spacing units (not mm), with hatch guides."""
    data = np.load(preview_path)
    if "phase_slice" in data:
        phase = np.asarray(data["phase_slice"], dtype=float)
    else:
        # Older previews stored binary pores only
        phase = np.asarray(data["pore_slice"], dtype=float)
        # Map pore→0, other→1 so colors still make sense
        phase = np.where(phase > 0, 0.0, 1.0)
    phase[phase >= 4] = np.nan  # outside the STL cross-section
    res_m = float(np.asarray(data["resolution"]).ravel()[0])
    n_pore_slice = int(np.sum(phase == 0))
    # Block-min downsample for the browser: pore (0) is the lowest code, so pores survive
    b = int(np.ceil(max(phase.shape) / 500.0))
    if b > 1:
        nx0, ny0 = phase.shape
        pad = np.full((int(np.ceil(nx0 / b)) * b, int(np.ceil(ny0 / b)) * b), np.nan)
        pad[:nx0, :ny0] = phase
        blocks = pad.reshape(pad.shape[0] // b, b, pad.shape[1] // b, b)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                phase = np.nanmin(blocks, axis=(1, 3))
        res_m *= b
    res_um = res_m * 1e6
    hatch_m = float(summary.get("hatch_m") or 140e-6)
    hatch_um = max(1e-6, hatch_m * 1e6)
    # Discrete Raptor colormap: 0 red, 1 gray, 2 black, 3 yellow
    colorscale = [
        [0.00, "rgb(255,0,0)"],
        [0.24, "rgb(255,0,0)"],
        [0.25, "rgb(178,178,178)"],
        [0.49, "rgb(178,178,178)"],
        [0.50, "rgb(40,40,40)"],
        [0.74, "rgb(40,40,40)"],
        [0.75, "rgb(255,255,0)"],
        [1.00, "rgb(255,255,0)"],
    ]
    ny, nx = int(phase.T.shape[0]), int(phase.T.shape[1])
    # Coordinates in hatch spacings (hatching space), not mm
    xs = (np.arange(nx) + 0.5) * res_um / hatch_um
    ys = (np.arange(ny) + 0.5) * res_um / hatch_um
    x_max = float(nx * res_um / hatch_um)
    y_max = float(ny * res_um / hatch_um)
    fig = go.Figure(
        data=go.Heatmap(
            z=phase.T,
            x=xs,
            y=ys,
            zmin=0,
            zmax=3,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Phase",
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["Pore", "Melted", "Boundary", "Intersection"],
                len=0.75,
            ),
            hovertemplate=(
                "x=%{x:.2f} hatch<br>y=%{y:.2f} hatch<br>phase=%{z:.0f}<extra></extra>"
            ),
        )
    )
    # Hatch guides (integer multiples of hatch spacing)
    # One line trace (add_vline per hatch costs ~30 ms each); skipped when too dense to see
    n_hatch_x = int(np.floor(x_max + 1e-9)) + 1
    if n_hatch_x <= 150:
        gx = np.repeat(np.arange(n_hatch_x, dtype=float), 3)
        gy = np.tile([0.0, y_max, np.nan], n_hatch_x)
        fig.add_trace(
            go.Scatter(
                x=gx, y=gy, mode="lines", hoverinfo="skip", showlegend=False,
                line=dict(color="rgba(11,61,145,0.35)", width=1, dash="dot"),
            )
        )
    edge_um = float(summary.get("rve_edge_m", 0)) * 1e6
    edge_h = edge_um / hatch_um if hatch_um else 0.0
    z_idx = int(np.asarray(data["z_index"]).ravel()[0]) if "z_index" in data else -1
    fig.update_layout(
        title=(
            f"LoF XY slice · RVE {edge_h:.2f}×{edge_h:.2f} hatch "
            f"(hatch={hatch_um:.0f} µm · voxel={res_um:.2f} µm)"
            + (f" · z-index {z_idx}" if z_idx >= 0 else "")
            + f" · LoF voxels={n_pore_slice}"
        ),
        xaxis_title="X (hatch spacings)",
        yaxis_title="Y (hatch spacings)",
        height=480,
        margin=dict(l=50, r=20, t=70, b=50),
        yaxis=dict(scaleanchor="x", scaleratio=1, range=[0, y_max]),
        xaxis=dict(range=[0, x_max]),
    )
    return fig


def _rebuild_preview_from_vti(outdir: str, summary: dict) -> Optional[str]:
    """Rebuild preview.npz from VTI using the z-plane with most LoF voxels."""
    vti = os.path.join(outdir, "rve.vti")
    if not os.path.isfile(vti):
        vti = os.path.join(outdir, "porosity_map.vti")
    if not os.path.isfile(vti):
        return None
    try:
        import pyvista as pv

        mesh = pv.read(vti)
        name = "Phase" if "Phase" in mesh.array_names else mesh.active_scalars_name
        dims = np.asarray(mesh.dimensions, dtype=int)
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        arr = np.asarray(mesh[name]).reshape((nx, ny, nz), order="F")
        pore_counts = np.sum(arr == 0, axis=(0, 1))
        z_idx = int(np.argmax(pore_counts)) if int(np.max(pore_counts)) > 0 else nz // 2
        slice2d = arr[:, :, z_idx].astype(np.uint8)
        spacing = np.asarray(mesh.spacing, dtype=float)
        origin = np.asarray(mesh.origin, dtype=float)
        preview = os.path.join(outdir, "preview.npz")
        np.savez_compressed(
            preview,
            phase_slice=slice2d,
            pore_slice=(slice2d == 0).astype(np.uint8),
            origin=origin,
            resolution=float(spacing[0]),
            z_index=z_idx,
            n_pore_on_slice=int(np.sum(slice2d == 0)),
        )
        return preview
    except Exception:
        return None


def _preview_3d_img(outdir: str, summary: dict):
    png = summary.get("preview_3d") or os.path.join(outdir, "preview_3d.png")
    if not png or not os.path.isfile(png):
        return None
    # Serve via Dash assets-relative path under raptor_work
    rel = os.path.relpath(png, os.path.join(os.path.dirname(__file__), "assets")).replace("\\", "/")
    return html.Img(
        src=f"/assets/{rel}?t={int(time.time())}",
        alt="Raptor 3D phase preview (Pore / Melted / Boundary / Intersection)",
        style={
            "width": "100%",
            "maxWidth": "720px",
            "border": "1px solid #c5d0dc",
            "borderRadius": "8px",
            "marginTop": "0.75rem",
            "background": "#fff",
        },
    )


def _library_table(folder: str, eta_source: str):
    """Per-velocity melt-pool parameters the library will feed into Raptor."""
    try:
        lib = mpl.load_library(folder, eta_source)
    except Exception as exc:
        return dbc.Alert(f"Could not read melt-pool library: {exc}", color="danger", className="py-2 mb-0")
    if not lib:
        return dbc.Alert(
            f"No ULI_v*_theta*_widths.txt files found in {folder}", color="warning", className="py-2 mb-0"
        )
    header = html.Tr(
        [html.Th(h) for h in ("V (mm/s)", "Mean W (µm)", "η_D (D scale)", "η_C (H scale)", "Depth shape", "θ files")]
    )
    rows = [
        html.Tr(
            [
                html.Td(f"{v:g}"),
                html.Td(_fmt_num(None if c.mean_width_m is None else c.mean_width_m * 1e6, ".4g")),
                html.Td(_fmt_num(c.eta_d, ".4g")),
                html.Td(_fmt_num(c.eta_c, ".4g")),
                html.Td(f"{_fmt_num(c.depth_shape, '.4g')} (n={c.depth_shape_n})"),
                html.Td(", ".join(f"{t:g}" for t in sorted(c.width_files)) or "—"),
            ]
        )
        for v, c in lib.items()
    ]
    return dbc.Table(
        [html.Thead(header), html.Tbody(rows)], bordered=True, size="sm", className="mb-0 bg-white small"
    )


def _library_section() -> html.Div:
    lib_dir = mpl.DEFAULT_LIBRARY_DIR
    vels = mpl.available_velocities(lib_dir)
    thetas = mpl.available_thetas(lib_dir)
    has_lib = bool(vels)
    return html.Div(
        [
            html.H5("Melt-pool library (V × θ)", className="mt-1"),
            dbc.Checklist(
                id="rap-lib-on",
                options=[{"label": "Use velocity / scan-angle library", "value": "on"}],
                value=["on"] if has_lib else [],
                switch=True,
            ),
            html.Div("Library folder", className="control-label mt-1"),
            dbc.Input(id="rap-lib-dir", type="text", value=lib_dir, debounce=True),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Velocities (mm/s) — one run each", className="control-label"),
                            dcc.Dropdown(
                                id="rap-lib-vels",
                                options=[{"label": f"{v:g}", "value": v} for v in vels],
                                value=vels[:1],
                                multi=True,
                            ),
                            dbc.Button("All V", id="rap-lib-all-v", color="link", size="sm", className="px-0"),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.Div("η_D / η_C source", className="control-label"),
                            dbc.RadioItems(
                                id="rap-lib-eta",
                                options=[{"label": s, "value": s} for s in mpl.ETA_SOURCES],
                                value="flucWs",
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mt-1",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Scan angle θ", className="control-label"),
                            dbc.RadioItems(
                                id="rap-lib-theta-mode",
                                options=[
                                    {"label": "Averaged spectrum (all orientations)", "value": "averaged"},
                                    {"label": "Per layer (nearest θ to scan direction)", "value": "per_layer"},
                                    {"label": "Fixed θ for all layers", "value": "fixed"},
                                ],
                                value="averaged",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("Fixed θ (°)", className="control-label"),
                            dcc.Dropdown(
                                id="rap-lib-theta",
                                options=[{"label": f"{t:g}", "value": t} for t in thetas],
                                value=thetas[0] if thetas else 0.0,
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div("θ offset (°)", className="control-label"),
                            dbc.Input(id="rap-lib-offset", type="number", value=0, step=1),
                        ],
                        md=3,
                    ),
                ],
                className="mt-1",
            ),
            dbc.Checklist(
                id="rap-lib-use",
                options=[
                    {"label": "D / H scales from η_D / η_C", "value": "eta"},
                    {"label": "Depth shape = mean of depthShapes.txt", "value": "shape"},
                ],
                value=["eta", "shape"],
                inline=True,
                className="mt-1",
            ),
            html.Small(
                "When on, the velocity list replaces the Velocity field. θ offset rotates the RVE scan "
                "direction before matching θ files (use it if Vamsi's θ=0 is not the RVE +x axis).",
                className="text-muted d-block mb-1",
            ),
            html.Div(id="rap-lib-table", children=_library_table(lib_dir, "flucWs")),
        ],
        className="mb-2 p-2 border rounded",
    )


def _sweep_block(sweep: list) -> list:
    """Table + porosity-vs-V plot for multi-velocity runs."""
    rows = [
        html.Tr(
            [
                html.Td(f"{s['velocity_mm_s']:g}"),
                html.Td(_fmt_num(s.get("porosity_pct"), ".4g")),
                html.Td(str(s.get("n_defects", "—"))),
                html.Td(_fmt_num(s.get("depth_scale"), ".4g")),
                html.Td(_fmt_num(s.get("height_scale"), ".4g")),
                html.Td(_fmt_num(s.get("depth_shape"), ".4g")),
                html.Td(", ".join(f"{k}°:{v}" for k, v in (s.get("theta_groups") or {}).items())),
            ]
        )
        for s in sweep
    ]
    header = html.Tr(
        [html.Th(h) for h in ("V (mm/s)", "φ (%)", "Defects", "D scale", "H scale", "Depth shape", "θ groups (vectors)")]
    )
    fig = go.Figure(
        go.Scatter(
            x=[s["velocity_mm_s"] for s in sweep],
            y=[s.get("porosity_pct") for s in sweep],
            mode="lines+markers",
            line=dict(color="#0B3D91"),
        )
    )
    fig.update_layout(
        title="Raptor LoF porosity vs scan velocity",
        xaxis_title="V (mm/s)",
        yaxis_title="φ (%)",
        height=320,
        margin=dict(l=50, r=20, t=50, b=45),
    )
    return [
        html.H5("Velocity sweep", className="mt-2"),
        dbc.Table([html.Thead(header), html.Tbody(rows)], bordered=True, size="sm", className="bg-white small"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Small(
            "Phase map / 3D preview / downloads below show the last velocity. "
            "Per-V outputs and sweep_summary.csv are in the run folder.",
            className="text-muted d-block",
        ),
    ]


def embedded_panel() -> html.Div:
    """Raptor block for the right column of the NU Spatial Field tab."""
    melt_ok = os.path.isfile(DEFAULT_MELT)
    return html.Div(
        [
            dcc.Download(id="rap-morph-download"),
            dcc.Store(id="rap-job-store", data=None),
            dcc.Store(id="rap-last-outdir", data=None),
            dcc.Interval(id="rap-job-tick", interval=1000, n_intervals=0, disabled=True),
            html.H3("Raptor — LoF (ORNL)"),
            html.P(
                [
                    "Geometric melt-pool RVE for lack-of-fusion porosity — tracks ",
                    html.Strong("Pore / Melted / Boundary / Intersection"),
                    " like the ORNL Raptor README view. ",
                    html.A(
                        "ORNL-MDF/Raptor",
                        href="https://github.com/ORNL-MDF/Raptor",
                        target="_blank",
                    ),
                ],
                className="text-muted mb-2",
            ),
            dbc.Alert(
                "Raptor ready (bundled Ti64 melt-pool series)."
                if melt_ok
                else "Melt-pool data missing — check Raptor/examples/data.",
                color="success" if melt_ok else "warning",
                className="py-2",
            ),
            html.H4("RVE / process"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Edge (µm)", className="control-label"),
                            # Defaults match Raptor GitHub examples/api_example/rve.py
                            dbc.Input(id="rap-edge-um", type="number", value=500, min=50, step=10),
                            html.Small(
                                "Slice plot uses hatch spacings (not mm).",
                                className="text-muted",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Voxel (µm)", className="control-label"),
                            dbc.Input(id="rap-voxel-um", type="number", value=5, min=2, step=1),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Modes", className="control-label"),
                            dbc.Input(id="rap-nmodes", type="number", value=50, min=5, max=80, step=1),
                        ],
                        md=4,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Power (W)", className="control-label"),
                            dbc.Input(id="rap-power", type="number", value=370),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Velocity (m/s)", className="control-label"),
                            dbc.Input(id="rap-velocity", type="number", value=1.3, step=0.1),
                            html.Small(
                                "1.3 m/s = 1300 mm/s (CMU default)",
                                className="text-muted",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Hatch (µm)", className="control-label"),
                            dbc.Input(id="rap-hatch-um", type="number", value=140),
                        ],
                        md=4,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Button(
                "Use CMU default P–V (370 W, 1.3 m/s)",
                id="rap-use-cmu-pv",
                color="link",
                size="sm",
                className="px-0 mb-2",
            ),
            _library_section(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Layer (µm)", className="control-label"),
                            dbc.Input(id="rap-layer-um", type="number", value=30),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Rotation (°)", className="control-label"),
                            dbc.Input(id="rap-rot", type="number", value=67),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Seed", className="control-label"),
                            dbc.Input(id="rap-seed", type="number", value=42),
                        ],
                        md=4,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("W / D / H scales", className="control-label"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="rap-w-scale", type="number", value=1.0, step=0.1),
                                    dbc.Input(id="rap-d-scale", type="number", value=0.8, step=0.1),
                                    dbc.Input(id="rap-h-scale", type="number", value=0.4, step=0.1),
                                ]
                            ),
                            html.Small(
                                "D / H are overridden by η_D / η_C when the library is on.",
                                className="text-muted",
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        [
                            html.Div("Height shape", className="control-label"),
                            dbc.Input(id="rap-height-shape", type="number", value=1, min=0.1, step=0.1),
                            html.Small("1 parabola · 2 ellipse", className="text-muted"),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div("Random phase", className="control-label"),
                            dbc.Checklist(
                                id="rap-random-phase",
                                options=[{"label": "On", "value": "on"}],
                                value=["on"],
                            ),
                        ],
                        md=4,
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
                                id="rap-compute-device",
                                options=[
                                    {"label": "Auto (prefer GPU)", "value": "auto"},
                                    {"label": "GPU (CUDA)", "value": "gpu"},
                                    {"label": "CPU only", "value": "cpu"},
                                ],
                                value="auto",
                                inline=True,
                            ),
                            html.Div(id="rap-compute-hint", className="text-muted small mt-1"),
                        ],
                        md=12,
                    ),
                ],
                className="mb-2",
            ),
            html.Div(
                [
                    html.H5("Compare to CMU LoF", className="mt-1"),
                    html.Small(
                        "Unifies CMU and Raptor to volume porosity φ (%). "
                        "CMU also reports ρ (pores/mm³); Raptor φ comes from unmelted voxels. "
                        "With the library off, the melt-pool series is ULI_v1700 θ=0.",
                        className="text-muted d-block mb-2",
                    ),
                    dbc.Button(
                        "Preview CMU LoF at current P–V",
                        id="rap-cmu-compare",
                        color="secondary",
                        outline=True,
                        size="sm",
                        className="me-2",
                    ),
                    html.Div(id="rap-cmu-compare-out", className="mt-2"),
                ],
                className="mb-2 p-2 border rounded",
            ),
            dbc.Button("Run Raptor RVE", id="rap-run", color="primary", className="w-100 btn-nasa"),
            html.Div(id="rap-run-status", className="mt-2 fw-semibold", style={"color": "#0B3D91"}),
            dbc.Progress(
                id="rap-run-progress",
                value=0,
                striped=True,
                animated=True,
                className="mt-2",
                style={"height": "0.7rem", "display": "none"},
            ),
            html.Hr(className="my-3"),
            html.Div(id="rap-results"),
            dcc.Graph(
                id="rap-figure",
                figure=go.Figure(
                    layout=dict(
                        title="Run Raptor to preview mid-plane phases (Pore / Melted / Boundary / Intersection)",
                        height=420,
                    )
                ),
                config={"displayModeBar": False},
            ),
            html.Div(id="rap-preview-3d"),
            html.Div(
                [
                    dbc.Button(
                        "Open 3D phase viewer (PyVista)",
                        id="rap-open-pv",
                        color="primary",
                        className="me-2 btn-nasa",
                        disabled=True,
                    ),
                    dbc.Button(
                        "Download morphology CSV",
                        id="rap-download-btn",
                        color="secondary",
                        outline=True,
                        size="sm",
                        disabled=True,
                    ),
                ],
                className="mt-2",
            ),
            html.Div(
                id="rap-pv-status",
                className="text-muted small mt-1",
                children="After a run: interactive PyVista shows clipped RVE with the README colormap.",
            ),
            html.H5("Raptor log", className="mt-3"),
            dcc.Textarea(
                id="rap-log",
                style={
                    "width": "100%",
                    "height": "140px",
                    "fontFamily": "IBM Plex Mono, monospace",
                    "fontSize": "12px",
                },
                readOnly=True,
                value="",
            ),
        ],
        className="panel-card",
    )


def layout():
    """Standalone Raptor tab."""
    return html.Div([embedded_panel()])


def register_callbacks(app):
    @app.callback(
        Output("rap-power", "value"),
        Output("rap-velocity", "value"),
        Input("rap-use-cmu-pv", "n_clicks"),
        prevent_initial_call=True,
    )
    def use_cmu_default_pv(n_clicks):
        if not n_clicks:
            return dash.no_update
        return 370, 1.3

    @app.callback(
        Output("rap-lib-table", "children"),
        Output("rap-lib-vels", "options"),
        Output("rap-lib-theta", "options"),
        Input("rap-lib-dir", "value"),
        Input("rap-lib-eta", "value"),
        prevent_initial_call=True,
    )
    def refresh_library(folder, eta_source):
        folder = (folder or "").strip().strip('"') or mpl.DEFAULT_LIBRARY_DIR
        vels = mpl.available_velocities(folder)
        thetas = mpl.available_thetas(folder)
        return (
            _library_table(folder, eta_source or "flucWs"),
            [{"label": f"{v:g}", "value": v} for v in vels],
            [{"label": f"{t:g}", "value": t} for t in thetas],
        )

    @app.callback(
        Output("rap-lib-vels", "value"),
        Input("rap-lib-all-v", "n_clicks"),
        State("rap-lib-vels", "options"),
        prevent_initial_call=True,
    )
    def select_all_velocities(n_clicks, options):
        if not n_clicks:
            return dash.no_update
        return [o["value"] for o in (options or [])]

    @app.callback(
        Output("rap-cmu-compare-out", "children"),
        Input("rap-cmu-compare", "n_clicks"),
        State("rap-power", "value"),
        State("rap-velocity", "value"),
        State("rap-hatch-um", "value"),
        State("rap-layer-um", "value"),
        prevent_initial_call=True,
    )
    def compare_cmu_lof(n_clicks, power, velocity, hatch_um, layer_um):
        if not n_clicks:
            return dash.no_update
        try:
            from .cmu_lof_density import query_cmu_lof

            v_m_s = float(velocity or 1.3)
            info = query_cmu_lof(
                power_w=float(power or 370),
                velocity_mm_s=v_m_s * 1000.0,
                hatch_um=float(hatch_um or 140),
                layer_um=float(layer_um or 30),
            )
            return dbc.Alert(
                [
                    html.Div(
                        f"CMU at P={info['power_w']:g} W, V={info['velocity_mm_s']:g} mm/s, "
                        f"hatch={info['hatch_um']:g} µm, layer={info['layer_um']:g} µm"
                    ),
                    html.Div(
                        f"φ ≈ {_fmt_num(info.get('volume_porosity_pct'))}% · "
                        f"ρ = {_fmt_num(info.get('lof_density_per_mm3'))} pores/mm³ · "
                        f"mean ESD ≈ {_fmt_num(info.get('mean_esd_um'), '.3g')} µm"
                    ),
                    html.Div(
                        "Run Raptor at the same P–V to fill the comparison table "
                        "(Raptor/CMU on volume porosity φ).",
                        className="small mt-1",
                    ),
                ],
                color="info",
                className="py-2 mb-0",
            )
        except Exception as exc:
            return dbc.Alert(f"CMU LoF query failed: {exc}", color="danger", className="py-2 mb-0")

    @app.callback(
        Output("rap-cmu-compare-out", "children", allow_duplicate=True),
        Input("rap-last-outdir", "data"),
        prevent_initial_call=True,
    )
    def fill_cmu_compare_after_run(outdir):
        """Deferred CMU vs Raptor table — keeps the main results callback fast."""
        if not outdir or not os.path.isdir(outdir):
            return dash.no_update
        try:
            from .cmu_lof_density import compare_raptor_to_cmu

            summary_path = os.path.join(outdir, "summary.json")
            if not os.path.isfile(summary_path):
                return dash.no_update
            summary = _load_json(summary_path)
            morph_path = os.path.join(outdir, "rve_morphology.csv")
            if not os.path.exists(morph_path):
                morph_path = os.path.join(outdir, "morphology.csv")
            n_defects = 0
            if os.path.exists(morph_path):
                try:
                    n_defects = len(pd.read_csv(morph_path))
                except Exception:
                    n_defects = 0
            cmp = compare_raptor_to_cmu(
                power_w=float(summary.get("power_w") or 370),
                velocity_m_s=float(summary.get("velocity_m_s") or 1.3),
                hatch_um=float(summary.get("hatch_m") or 140e-6) * 1e6,
                layer_um=float(summary.get("layer_m") or 30e-6) * 1e6,
                rve_edge_m=float(summary.get("rve_edge_m") or 0.0),
                raptor_porosity_pct=float(summary.get("porosity_pct") or 0.0),
                n_defects=n_defects,
            )
            return dbc.Alert(
                [html.Strong("CMU vs Raptor (unified units)")] + _lof_compare_block(cmp),
                color="info",
                className="py-2 mb-0",
            )
        except Exception as exc:
            return dbc.Alert(
                f"CMU comparison unavailable: {exc}",
                color="warning",
                className="py-2 mb-0",
            )

    @app.callback(
        Output("rap-compute-hint", "children"),
        Input("rap-compute-device", "value"),
    )
    def update_compute_hint(mode):
        from .compute_backend import resolve_device

        info = resolve_device(mode or "auto")
        if info.cuda_available:
            return f"{info.detail}. Melt-mask uses CUDA when GPU is selected."
        return f"{info.detail}. Install an NVIDIA driver / CUDA-capable GPU for acceleration."

    @app.callback(
        Output("rap-job-store", "data"),
        Output("rap-job-tick", "disabled"),
        Output("rap-run", "disabled"),
        Output("rap-run-status", "children"),
        Output("rap-run-progress", "value"),
        Output("rap-run-progress", "style"),
        Output("rap-log", "value"),
        Output("app-status", "children", allow_duplicate=True),
        Input("rap-run", "n_clicks"),
        State("rap-edge-um", "value"),
        State("rap-voxel-um", "value"),
        State("rap-power", "value"),
        State("rap-velocity", "value"),
        State("rap-hatch-um", "value"),
        State("rap-layer-um", "value"),
        State("rap-rot", "value"),
        State("rap-nmodes", "value"),
        State("rap-w-scale", "value"),
        State("rap-d-scale", "value"),
        State("rap-h-scale", "value"),
        State("rap-random-phase", "value"),
        State("rap-seed", "value"),
        State("rap-compute-device", "value"),
        State("rap-height-shape", "value"),
        State("rap-lib-on", "value"),
        State("rap-lib-dir", "value"),
        State("rap-lib-vels", "value"),
        State("rap-lib-eta", "value"),
        State("rap-lib-theta-mode", "value"),
        State("rap-lib-theta", "value"),
        State("rap-lib-offset", "value"),
        State("rap-lib-use", "value"),
        prevent_initial_call=True,
    )
    def start_raptor(
        n_clicks,
        edge_um,
        voxel_um,
        power,
        velocity,
        hatch_um,
        layer_um,
        rot,
        nmodes,
        w_scale,
        d_scale,
        h_scale,
        random_phase,
        seed,
        compute_device,
        height_shape,
        lib_on,
        lib_dir,
        lib_vels,
        lib_eta,
        lib_theta_mode,
        lib_theta,
        lib_offset,
        lib_use,
    ):
        if not n_clicks:
            return dash.no_update
        prog_style = {"height": "0.7rem", "display": "block"}
        try:
            os.makedirs(WORK_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            outdir = os.path.join(WORK_DIR, stamp)
            os.makedirs(outdir, exist_ok=True)

            edge_um = float(edge_um or 500)
            voxel_um = float(voxel_um or 5)
            if voxel_um <= 0 or edge_um / voxel_um > 120:
                return (
                    None,
                    True,
                    False,
                    "RVE too fine — reduce edge or increase voxel size",
                    0,
                    {"height": "0.7rem", "display": "none"},
                    "Guardrail: edge/voxel > 120 would create a very large grid on a laptop.",
                    "Raptor: grid too large",
                )

            cfg = {
                "rve_edge_m": edge_um * 1e-6,
                "voxel_resolution_m": voxel_um * 1e-6,
                "power_w": float(power or 370),
                "velocity_m_s": float(velocity or 1.3),
                "hatch_m": float(hatch_um or 140) * 1e-6,
                "layer_m": float(layer_um or 30) * 1e-6,
                "rotation_deg": float(rot or 67),
                "n_modes": int(nmodes or 50),
                "width_scale": float(w_scale if w_scale is not None else 1.0),
                "depth_scale": float(d_scale if d_scale is not None else 0.8),
                "height_scale": float(h_scale if h_scale is not None else 0.4),
                "enable_random_phases": bool(random_phase and "on" in (random_phase or [])),
                "random_seed": int(seed) if seed is not None else 42,
                "melt_pool_data": DEFAULT_MELT,
                "compute_device": (compute_device or "auto"),
                "height_shape": float(height_shape or 1),
            }
            if lib_on and "on" in lib_on:
                if not lib_vels:
                    return (
                        None,
                        True,
                        False,
                        "Select at least one velocity in the melt-pool library",
                        0,
                        {"height": "0.7rem", "display": "none"},
                        "Melt-pool library is on but no velocities are selected.",
                        "Raptor: no velocity selected",
                    )
                cfg["velocities_mm_s"] = sorted(float(v) for v in lib_vels)
                cfg["velocity_m_s"] = cfg["velocities_mm_s"][0] / 1000.0
                cfg["library"] = {
                    "dir": (lib_dir or "").strip().strip('"') or mpl.DEFAULT_LIBRARY_DIR,
                    "eta_source": lib_eta or "flucWs",
                    "theta_mode": lib_theta_mode or "averaged",
                    "theta_deg": float(lib_theta or 0.0),
                    "angle_offset_deg": float(lib_offset or 0.0),
                    "use_eta": "eta" in (lib_use or []),
                    "use_depth_shape": "shape" in (lib_use or []),
                    "lambda_mean": mpl.LAMBDA_MEAN,
                }
            cfg_path = os.path.join(outdir, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            stdout_path = os.path.join(outdir, "worker.stdout.log")
            stderr_path = os.path.join(outdir, "worker.stderr.log")
            env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "NUMBA_CACHE_DIR": os.path.join(ROOT, "Raptor", ".numba-cache"),
                "NASA_COMPUTE_DEVICE": str(compute_device or "auto"),
            }
            # Detach from the console so Ctrl+C on run_gui.bat does not kill the worker mid-run
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

            with open(stdout_path, "w", encoding="utf-8") as out_f, open(
                stderr_path, "w", encoding="utf-8"
            ) as err_f:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "integrated_gui.raptor_worker",
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

            return (
                {"outdir": outdir, "pid": int(proc.pid), "t0": time.time()},
                False,
                True,
                "Starting Raptor worker… · estimating…",
                2,
                prog_style,
                f"Raptor worker started (pid={proc.pid}).\nConfig: {cfg_path}\n"
                f"Logs: {stdout_path}",
                "Raptor running…",
            )
        except Exception:
            return (
                None,
                True,
                False,
                "Failed to start",
                0,
                {"height": "0.7rem", "display": "none"},
                traceback.format_exc(),
                "Raptor failed",
            )

    @app.callback(
        Output("rap-log", "value", allow_duplicate=True),
        Output("rap-results", "children"),
        Output("rap-figure", "figure"),
        Output("rap-preview-3d", "children"),
        Output("rap-download-btn", "disabled"),
        Output("rap-open-pv", "disabled"),
        Output("rap-run-status", "children", allow_duplicate=True),
        Output("app-status", "children", allow_duplicate=True),
        Output("rap-run-progress", "value", allow_duplicate=True),
        Output("rap-run-progress", "style", allow_duplicate=True),
        Output("rap-job-store", "data", allow_duplicate=True),
        Output("rap-job-tick", "disabled", allow_duplicate=True),
        Output("rap-run", "disabled", allow_duplicate=True),
        Output("rap-last-outdir", "data"),
        Input("rap-job-tick", "n_intervals"),
        State("rap-job-store", "data"),
        prevent_initial_call=True,
    )
    def poll_raptor(_n, job):
        global _LAST_MORPH_CSV
        if not job or not isinstance(job, dict):
            return (dash.no_update,) * 14

        outdir = job.get("outdir")
        pid = job.get("pid")
        t0 = float(job.get("t0") or time.time())
        elapsed = time.time() - t0
        status_path = os.path.join(outdir, "status.json")
        progress_path = os.path.join(outdir, "progress.json")
        log_path = os.path.join(outdir, "worker.log")
        stderr_path = os.path.join(outdir, "worker.stderr.log")
        alive = _pid_alive(pid)
        empty = go.Figure(layout=dict(title="Raptor preview unavailable", height=420))

        def _tail_logs() -> str:
            chunks = []
            for p in (log_path, stderr_path, os.path.join(outdir, "worker.stdout.log")):
                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.read().splitlines()
                        if lines:
                            chunks.append(f"--- {os.path.basename(p)} ---\n" + "\n".join(lines[-50:]))
                    except Exception:
                        pass
            return "\n\n".join(chunks)

        if os.path.exists(status_path):
            status = _load_json(status_path)
            log_txt = _tail_logs() or status.get("log") or ""
            if not status.get("ok"):
                return (
                    log_txt or status.get("error") or "Failed",
                    dbc.Alert("Raptor failed. See log.", color="danger"),
                    empty,
                    None,
                    True,
                    True,
                    "Failed — see Log",
                    "Raptor failed",
                    100,
                    {"height": "0.7rem", "display": "none"},
                    None,
                    True,
                    False,
                    None,
                )

            summary = status.get("summary") or _load_json(os.path.join(outdir, "summary.json"))
            morph_path = os.path.join(outdir, "rve_morphology.csv")
            if not os.path.exists(morph_path):
                morph_path = os.path.join(outdir, "morphology.csv")
            if os.path.exists(morph_path):
                with open(morph_path, "r", encoding="utf-8") as f:
                    _LAST_MORPH_CSV = f.read()
            preview = os.path.join(outdir, "preview.npz")
            # Old runs used mid-plane (often no LoF); rebuild if slice has no pores
            try:
                need_rebuild = True
                if os.path.exists(preview):
                    _prev = np.load(preview)
                    _ph = np.asarray(
                        _prev["phase_slice"] if "phase_slice" in _prev else _prev["pore_slice"]
                    )
                    need_rebuild = (
                        int(np.sum(_ph == 0)) == 0
                        and int(summary.get("n_pore_voxels") or 0) > 0
                    )
                if need_rebuild:
                    rebuilt = _rebuild_preview_from_vti(outdir, summary)
                    if rebuilt:
                        preview = rebuilt
            except Exception:
                pass
            try:
                fig = _slice_figure(preview, summary) if os.path.exists(preview) else empty
            except Exception:
                fig = empty
            preview_3d = _preview_3d_img(outdir, summary)
            n_defects = 0
            if os.path.exists(morph_path):
                try:
                    n_defects = len(pd.read_csv(morph_path))
                except Exception:
                    n_defects = 0

            # Keep poll fast — CMU compare runs in a separate callback via rap-last-outdir
            alert = dbc.Alert(
                [
                    html.Div(
                        f"Porosity φ = {summary.get('porosity_pct', 0):.4g}% · "
                        f"pore voxels={summary.get('n_pore_voxels')} · "
                        f"melted={summary.get('n_interior')} · "
                        f"boundary={summary.get('n_boundary')} · "
                        f"intersection={summary.get('n_intersect')} · "
                        f"defects={n_defects}"
                    ),
                    html.Div(
                        f"Elapsed {summary.get('elapsed_s')} s · "
                        f"P={float(summary.get('power_w') or 370):g} W, "
                        f"V={float(summary.get('velocity_m_s') or 1.3)*1000:g} mm/s · "
                        f"vectors = {summary.get('n_path_vectors')} · "
                        "CMU comparison fills in below shortly.",
                        className="small mt-1",
                    ),
                    html.Div(
                        f"Melt pool: D scale={_fmt_num(summary.get('depth_scale'), '.4g')} · "
                        f"H scale={_fmt_num(summary.get('height_scale'), '.4g')} · "
                        f"depth shape={_fmt_num(summary.get('depth_shape'), '.4g')} · "
                        f"height shape={_fmt_num(summary.get('height_shape'), '.4g')} · "
                        "θ groups: "
                        + (
                            ", ".join(f"{k}°:{v}" for k, v in (summary.get("theta_groups") or {}).items())
                            or "—"
                        ),
                        className="small",
                    ),
                ],
                color="success",
            )
            sweep = summary.get("sweep") or []
            if len(sweep) > 1:
                alert = html.Div([alert] + _sweep_block(sweep))
            return (
                log_txt or "Done.",
                alert,
                fig,
                preview_3d
                or html.P(
                    "3D screenshot unavailable — use Open 3D phase viewer (PyVista).",
                    className="text-muted small mt-2",
                ),
                False,
                False,
                "Done — phase map + 3D preview ready.",
                "Raptor complete",
                100,
                {"height": "0.7rem", "display": "none"},
                None,
                True,
                False,
                outdir,
            )

        prog = {}
        if os.path.exists(progress_path):
            try:
                prog = _load_json(progress_path)
            except Exception:
                prog = {}
        if not prog:
            prog = {
                "message": "Starting Raptor (import / Numba compile may take a minute)…",
                "pct": min(15.0, 3.0 + elapsed),
                "eta_s": None,
                "step": 0,
                "n_steps": 6,
            }
        elif str(prog.get("stage")) == "queued" and elapsed > 2.0:
            prog = {
                **prog,
                "message": "Worker starting (Python import / Numba)…",
                "pct": min(12.0, 2.0 + elapsed),
            }

        log_tail = _tail_logs()
        if not alive and not os.path.exists(status_path):
            return (
                log_tail or "Worker exited without status.json — see worker.stderr.log.",
                dbc.Alert("Raptor worker stopped unexpectedly.", color="danger"),
                empty,
                None,
                True,
                True,
                "Failed — worker exited",
                "Raptor failed",
                100,
                {"height": "0.7rem", "display": "none"},
                None,
                True,
                False,
                None,
            )

        pct = float(prog.get("pct") or 0)
        return (
            log_tail or dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            _progress_text(prog, elapsed),
            f"Raptor · {pct:.0f}% · {_format_eta(prog.get('eta_s'))}",
            pct,
            {"height": "0.7rem", "display": "block"},
            job,
            False,
            True,
            dash.no_update,
        )

    @app.callback(
        Output("rap-morph-download", "data"),
        Input("rap-download-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_morph(_n):
        if not _LAST_MORPH_CSV:
            return dash.no_update
        return dict(content=_LAST_MORPH_CSV, filename="raptor_morphology.csv")

    @app.callback(
        Output("rap-pv-status", "children"),
        Input("rap-open-pv", "n_clicks"),
        State("rap-last-outdir", "data"),
        prevent_initial_call=True,
    )
    def open_pv(_n, outdir):
        if not outdir:
            return "No run yet."
        vti = os.path.join(outdir, "porosity_map.vti")
        if not os.path.exists(vti):
            return "porosity_map.vti not found."
        try:
            # GUI process forces off-screen PyVista; clear that for the interactive viewer child.
            code = (
                "import os\n"
                "import sys\n"
                f"sys.path.insert(0, r'''{os.path.join(ROOT, 'Raptor', 'src')}''')\n"
                "os.environ['PYVISTA_OFF_SCREEN'] = 'false'\n"
                "os.environ.pop('VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN', None)\n"
                "import pyvista as pv\n"
                "pv.OFF_SCREEN = False\n"
                "from raptor.api import visualize\n"
                f"visualize(r'''{vti}''')\n"
            )
            env = {k: v for k, v in os.environ.items() if k not in (
                "PYVISTA_OFF_SCREEN",
                "VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN",
            )}
            env["PYVISTA_OFF_SCREEN"] = "false"
            from .process_spawn import spawn_detached

            spawn_detached(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
            )
            return (
                "Opened interactive PyVista window "
                "(clipped RVE · Pore/Melted/Boundary/Intersection · cyan scan paths). "
                "Use the blue plane widget to slice Phase; hatch lines stay as a fixed reference."
            )
        except Exception as exc:
            return f"Failed to open viewer: {exc}"
