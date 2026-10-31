"""Standalone viewer for a Raptor velocity-sweep folder.

Run from the folder that contains V_1300, V_1350, ... :

    python view_results.py
"""
from __future__ import annotations

import base64
import json
import os

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE_COLORS = ["rgb(179,179,179)", "rgb(51,51,51)", "rgb(255,255,0)"]
PHASE_NAMES = ["Melted", "Melt boundary", "Track overlap"]
# Display values 1, 2, 3 on a 4-step scale (0 is unused so red is reserved for LoF markers).
PORE_TYPES = (
    ("raptor_lof", "LoF", "#E41A1C"),
    ("gas", "Gas", "#1B7F4E"),
    ("keyhole", "Keyhole", "#FF8C00"),
)


def velocity_folders():
    names = []
    for name in sorted(os.listdir(HERE)):
        path = os.path.join(HERE, name)
        if name.startswith("V_") and os.path.isdir(path) and os.path.isfile(os.path.join(path, "view3d.npz")):
            names.append(name)
    return names


def _b64(path: str) -> str:
    mime = "image/gif" if path.lower().endswith(".gif") else "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def _plane(value, shape_like, colors):
    arr = np.full(shape_like.shape, float(value))
    arr[np.asarray(colors) >= 4] = np.nan
    return arr


def _pore_kind(raw: str) -> str:
    s = str(raw).strip().lower()
    if "key" in s:
        return "keyhole"
    if "gas" in s:
        return "gas"
    if "lof" in s or "raptor" in s:
        return "raptor_lof"
    return s


# LoF closer than this to an outer face is the unmelted rim, not a pore inside the build.
SURFACE_LOF_MARGIN_MM = 0.5


def _away_from_skin(df: pd.DataFrame, origin_mm, voxel_mm: float, shape) -> np.ndarray:
    o = np.asarray(origin_mm, dtype=float)
    nx, ny, nz = [int(v) for v in shape]
    ijk = (df[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float) - o) / float(voxel_mm)
    dx = np.minimum(ijk[:, 0], (nx - 1) - ijk[:, 0])
    dy = np.minimum(ijk[:, 1], (ny - 1) - ijk[:, 1])
    dz = np.minimum(ijk[:, 2], (nz - 1) - ijk[:, 2])
    return np.minimum(np.minimum(dx, dy), dz) * float(voxel_mm) > SURFACE_LOF_MARGIN_MM


def _pores_for_view(csv_path: str, npz_path: str):
    """Pore table with the box-face LoF rim removed. Returns (frame, n_skin_lof)."""
    if not csv_path or not os.path.isfile(csv_path):
        return pd.DataFrame(), 0
    df = pd.read_csv(csv_path)
    need = {"type", "x_mm", "y_mm", "z_mm"}
    if not need.issubset(df.columns) or not len(df):
        return df, 0
    df = df.copy()
    df["_kind"] = df["type"].map(_pore_kind)
    n_skin = 0
    if npz_path and os.path.isfile(npz_path):
        grid = np.load(npz_path)
        lof = df["_kind"].to_numpy() == "raptor_lof"
        if lof.any():
            inside = _away_from_skin(df, grid["origin_mm"], float(grid["voxel_mm"]), grid["shape"])
            n_skin = int((lof & ~inside).sum())
            df = df.loc[~lof | inside]
    return df, n_skin


def _pore_traces(csv_path: str, npz_path: str = "") -> list:
    """One labeled marker trace per pore type, using pore centers inside the pillar."""
    grouped = {key: None for key, _label, _color in PORE_TYPES}
    df, _n_skin = _pores_for_view(csv_path, npz_path)
    if len(df) and "_kind" in df.columns:
        for key, _label, _color in PORE_TYPES:
            grouped[key] = df.loc[df["_kind"] == key]
    traces = []
    for key, label, color in PORE_TYPES:
        sub = grouped[key]
        n_src = 0 if sub is None else len(sub)
        if sub is not None and n_src:
            esd = sub["esd_um"].to_numpy(dtype=float) if "esd_um" in sub.columns else np.full(n_src, np.nan)
            size = np.where(np.isfinite(esd), np.clip(esd / 12.0, 4, 12), 5)
            traces.append(
                go.Scatter3d(
                    x=np.round(sub["x_mm"].to_numpy(float), 4),
                    y=np.round(sub["y_mm"].to_numpy(float), 4),
                    z=np.round(sub["z_mm"].to_numpy(float), 4),
                    mode="markers",
                    marker=dict(size=size, color=color, line=dict(width=0)),
                    name=f"{label}: {n_src:,}",
                    customdata=np.round(np.where(np.isfinite(esd), esd, 0), 1),
                    hovertemplate=(
                        f"{label}<br>(%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}}) mm"
                        "<br>ESD %{customdata} µm<extra></extra>"
                    ),
                )
            )
        else:
            traces.append(
                go.Scatter3d(
                    x=[], y=[], z=[], mode="markers",
                    marker=dict(size=8, color=color),
                    name=f"{label}: 0",
                    hoverinfo="skip",
                )
            )
    return traces


def figure_3d(npz_path: str, csv_path: str = "") -> go.Figure:
    d = np.load(npz_path)
    nx, ny, nz = [int(v) for v in d["shape"]]
    if "strides" in d.files:
        sx, sy, sz = [int(v) for v in d["strides"]]
    else:
        sx = sy = sz = int(d["stride"])
    o = np.asarray(d["origin_mm"], dtype=float)
    v = float(d["voxel_mm"])
    xs = o[0] + np.arange(0, nx, sx) * v
    ys = o[1] + np.arange(0, ny, sy) * v
    zs = o[2] + np.arange(0, nz, sz) * v
    x0, x1 = float(o[0]), float(o[0] + (nx - 1) * v)
    y0, y1 = float(o[1]), float(o[1] + (ny - 1) * v)
    z0, z1 = float(o[2]), float(o[2] + (nz - 1) * v)
    # Keep the shell light enough to rotate. Pore centers come from pores.csv.
    def _step(n, limit=120):
        return max(1, int(np.ceil(n / float(limit))))

    kx, ky, kz = _step(len(xs)), _step(len(ys)), _step(len(zs))
    xs, ys, zs = xs[::kx], ys[::ky], zs[::kz]
    # Phase 0 is a pore voxel on the skin. Draw the shell as melted metal so red
    # markers are the LoF pores through the interior, not only the outer face.
    def _shell(key, sy_, sz_):
        colors = np.asarray(d[key])[::sy_, ::sz_].astype(float)
        colors[colors <= 0] = 1
        return colors

    fx = {k: _shell(k, ky, kz) for k in ("face_x0", "face_x1")}
    fy = {k: _shell(k, kx, kz) for k in ("face_y0", "face_y1")}
    fz = {k: _shell(k, kx, ky) for k in ("face_z0", "face_z1")}
    shell_colors = ["rgb(179,179,179)", "rgb(51,51,51)", "rgb(255,255,0)"]
    cscale = []
    for i, c in enumerate(["rgb(179,179,179)"] + shell_colors):
        cscale += [[i / 4.0, c], [(i + 1) / 4.0, c]]
    surf_kw = dict(
        colorscale=cscale,
        cmin=-0.5,
        cmax=3.5,
        showscale=False,
        hoverinfo="skip",
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        opacity=0.22,
    )
    faces = []
    Yx, Zx = np.meshgrid(ys, zs, indexing="ij")
    for xv, key in ((x0, "face_x0"), (x1, "face_x1")):
        colors = fx[key]
        faces.append(go.Surface(x=_plane(xv, Yx, colors), y=Yx, z=Zx, surfacecolor=colors, **surf_kw))
    Xy, Zy = np.meshgrid(xs, zs, indexing="ij")
    for yv, key in ((y0, "face_y0"), (y1, "face_y1")):
        colors = fy[key]
        faces.append(go.Surface(x=Xy, y=_plane(yv, Xy, colors), z=Zy, surfacecolor=colors, **surf_kw))
    Xz, Yz = np.meshgrid(xs, ys, indexing="ij")
    for zv, key in ((z0, "face_z0"), (z1, "face_z1")):
        colors = fz[key]
        faces.append(go.Surface(x=Xz, y=Yz, z=_plane(zv, Xz, colors), surfacecolor=colors, **surf_kw))
    pore_traces = _pore_traces(csv_path, npz_path)
    legend = [
        go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=8, color=c, symbol="square"), name=n,
        )
        for c, n in zip(PHASE_COLORS, PHASE_NAMES)
    ]
    fig = go.Figure(data=faces + pore_traces + legend)
    n_faces = len(faces)
    n_pores = len(pore_traces)

    def _vis(face_opacity, show_pores):
        return [{
            "opacity": [face_opacity] * n_faces + [1.0] * (n_pores + len(legend)),
            "visible": [True] * n_faces + [show_pores] * n_pores + [True] * len(legend),
        }]

    fig.update_layout(
        height=680,
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.9)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.08,
                xanchor="center",
                buttons=[
                    dict(label="Pores inside", method="restyle", args=_vis(0.22, True)),
                    dict(label="Melted build", method="restyle", args=_vis(1.0, False)),
                ],
            )
        ],
    )
    return fig


def _process_line(folder: str) -> str:
    path = os.path.join(folder, "process.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    lines = [
        f"Hatch spacing {float(d['hatch_um']):.0f} µm. "
        f"Melt pool width {float(d['meltpool_width_um']):.0f} µm "
        f"(Raptor uses {float(d['meltpool_width_used_um']):.0f} µm). "
        f"Layer {float(d['layer_um']):.0f} µm."
    ]
    if "cmu_lof_per_mm3" in d and "raptor_lof_per_mm3" in d:
        lines.append(
            "Pores/mm³. "
            f"CMU LoF {float(d['cmu_lof_per_mm3']):.3f}. "
            f"Raptor LoF {float(d['raptor_lof_per_mm3']):.3f} replaces the CMU LoF prediction. "
            f"CMU keyhole {float(d.get('cmu_keyhole_per_mm3') or 0):.3f}. "
            f"CMU gas {float(d.get('cmu_gas_per_mm3') or 0):.3f}."
        )
    return "\n".join(lines)


def _counts(csv_path: str, npz_path: str = "") -> str:
    if not os.path.isfile(csv_path):
        return "No pores.csv"
    df, n_skin = _pores_for_view(csv_path, npz_path)
    lines = [f"{len(df):,} pores inside the pillar"]
    if "_kind" in df.columns and len(df):
        labels = {key: label for key, label, _color in PORE_TYPES}
        for key, n in df["_kind"].astype(str).value_counts().items():
            lines.append(f"  {labels.get(key, key)}: {int(n):,}")
    elif "type" in df.columns:
        for name, n in df["type"].astype(str).value_counts().items():
            lines.append(f"  {name}: {int(n):,}")
    if n_skin:
        lines.append(
            f"{n_skin:,} LoF pores are within {SURFACE_LOF_MARGIN_MM:.1f} mm of the outer surface and are not shown."
        )
    if "esd_um" in df.columns and len(df):
        esd = df["esd_um"].to_numpy(dtype=float)
        esd = esd[np.isfinite(esd)]
        if len(esd):
            lines.append(f"ESD median {np.median(esd):.1f} µm, max {esd.max():.1f} µm")
    return "\n".join(lines)


def main():
    folders = velocity_folders()
    if not folders:
        raise SystemExit(f"No V_#### folders with view3d.npz in {HERE}")
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    app.title = "Raptor 5×5×14 mm"
    app.layout = dbc.Container(
        [
            html.H3("Raptor 5×5×14 mm velocity sweep", className="mt-3"),
            html.P(
                "Drag to rotate. LoF (red), gas (green), and keyhole (orange) are drawn only when they are more than 0.5 mm inside the pillar. "
                "The animation below is an isometric view of the Raptor volume. A surface sweeps along Y and builds the solid. Z is height above the plate."
            ),
            dcc.Dropdown(id="vel", options=[{"label": n.replace("V_", "") + " mm/s", "value": n} for n in folders], value=folders[0], clearable=False),
            html.Pre(id="process", className="mt-2 mb-0"),
            html.Pre(id="counts", className="mt-2"),
            dcc.Graph(id="view3d", style={"height": "680px"}, config={"scrollZoom": True, "displaylogo": False}),
            html.H5("Volume sweep", className="mt-3"),
            html.Img(id="sweep", style={"width": "100%", "maxWidth": "820px"}),
            dbc.Row(
                [
                    dbc.Col(html.Img(id="melted", style={"width": "100%"}), md=4),
                    dbc.Col(html.Img(id="sources", style={"width": "100%"}), md=4),
                    dbc.Col(html.Img(id="shape", style={"width": "100%"}), md=4),
                ],
                className="mt-2",
            ),
        ],
        fluid=True,
    )

    @app.callback(
        Output("melted", "src"),
        Output("sources", "src"),
        Output("shape", "src"),
        Output("sweep", "src"),
        Output("view3d", "figure"),
        Output("counts", "children"),
        Output("process", "children"),
        Input("vel", "value"),
    )
    def show(name):
        folder = os.path.join(HERE, name or folders[0])
        melted = os.path.join(folder, "melted_build.png")
        sources = os.path.join(folder, "all_pore_sources.png")
        shape = os.path.join(folder, "pore_size_shape.png")
        npz = os.path.join(folder, "view3d.npz")
        csv_path = os.path.join(folder, "pores.csv")
        sweep = os.path.join(folder, "volume_sweep.gif")
        return (
            _b64(melted) if os.path.isfile(melted) else "",
            _b64(sources) if os.path.isfile(sources) else "",
            _b64(shape) if os.path.isfile(shape) else "",
            _b64(sweep) if os.path.isfile(sweep) else "",
            figure_3d(npz, csv_path),
            _counts(csv_path, npz),
            _process_line(folder),
        )

    app.run(debug=False, port=8055)


if __name__ == "__main__":
    main()
