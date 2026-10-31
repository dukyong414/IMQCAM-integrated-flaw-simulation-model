"""Whole-build Raptor phase views (results screenshot + interactive window).

Whole-build runs hold 10^8+ voxels, which is too much to clip or render directly.
The melted volume is drawn from a thinned copy (every s-th voxel) and every LoF
pore voxel is overlaid at full resolution as a point.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

PHASE_COLORS = [
    (1.0, 0.0, 0.0),
    (0.7, 0.7, 0.7),
    (0.2, 0.2, 0.2),
    (1.0, 1.0, 0.0),
]
PHASE_ANNOTATIONS = {0.5: "Pore", 1.5: "Melted", 2.5: "Boundary", 3.5: "Intersection"}
# Voxels outside the STL cross-section (STL band runs); not a Raptor phase
PHASE_OUTSIDE = 4


def load_phase_view(vti_path: str, max_render_voxels: int = 8_000_000):
    """Return (volume_for_display, pore_points_world (N,3) or None, thin_rate)."""
    import pyvista as pv

    full = pv.read(vti_path)
    phase = np.asarray(full["Phase"])
    pore_idx = np.flatnonzero(phase == 0)
    del phase
    pore_pts: Optional[np.ndarray] = None
    if len(pore_idx):
        # Raptor stores z as the fastest index axis and rotates it into place with
        # the image direction matrix, so map index -> world through that matrix.
        d0, d1, _ = full.dimensions
        ijk = np.column_stack(
            [pore_idx % d0, (pore_idx // d0) % d1, pore_idx // (d0 * d1)]
        ).astype(float) * np.asarray(full.spacing, dtype=float)
        try:
            dmat = np.asarray(full.direction_matrix, dtype=float).reshape(3, 3)
        except Exception:
            dmat = np.eye(3)
        pore_pts = np.asarray(full.origin, dtype=float) + ijk @ dmat.T
    rate = 1
    view = full
    if full.n_points > max_render_voxels:
        rate = int(np.ceil((full.n_points / max_render_voxels) ** (1.0 / 3.0)))
        view = full.extract_subset(full.extent, rate=(rate, rate, rate))
    if float(np.max(view["Phase"])) > PHASE_OUTSIDE - 0.5:
        # STL band: drop voxels outside the part so the real cross-section shows
        view = view.threshold(PHASE_OUTSIDE - 0.5, scalars="Phase", invert=True)
    return view, pore_pts, rate


def save_view3d_npz(
    path: str,
    phase: np.ndarray,
    *,
    origin_m: np.ndarray,
    voxel_m: float,
    layer_m: float = 0.0,
    hatch_m: float = 0.0,
    max_face_samples: int = 200,
    max_pores: int = 60_000,
    seed: int = 0,
) -> None:
    """Save outer-face phase maps + pore positions (mm) for a Plotly 3D figure.

    ``phase`` is Raptor's (nx, ny, nz) array with physical x, y, z axes.
    """
    nx, ny, nz = phase.shape
    sx, sy, sz = [max(1, int(np.ceil(n / float(max_face_samples)))) for n in (nx, ny, nz)]
    ijk = np.argwhere(phase == 0)
    # Pore voxels within 0.5 mm of the box face are the unmelted rim, not internal LoF.
    skin = max(2, int(np.ceil(0.5 / max(float(voxel_m) * 1e3, 1e-9))))
    if len(ijk) and min(nx, ny, nz) > 2 * skin + 4:
        keep = (
            (ijk[:, 0] >= skin) & (ijk[:, 0] < nx - skin)
            & (ijk[:, 1] >= skin) & (ijk[:, 1] < ny - skin)
            & (ijk[:, 2] >= skin) & (ijk[:, 2] < nz - skin)
        )
        ijk = ijk[keep]
    n_interior_pores = int(len(ijk))
    if len(ijk) > max_pores:
        ijk = ijk[np.random.default_rng(seed).choice(len(ijk), max_pores, replace=False)]
    o_mm = np.asarray(origin_m, dtype=float) * 1e3
    v_mm = float(voxel_m) * 1e3
    np.savez_compressed(
        path,
        face_x0=phase[0, ::sy, ::sz].astype(np.int8),
        face_x1=phase[-1, ::sy, ::sz].astype(np.int8),
        face_y0=phase[::sx, 0, ::sz].astype(np.int8),
        face_y1=phase[::sx, -1, ::sz].astype(np.int8),
        face_z0=phase[::sx, ::sy, 0].astype(np.int8),
        face_z1=phase[::sx, ::sy, -1].astype(np.int8),
        strides=np.array([sx, sy, sz]),
        shape=np.array([nx, ny, nz]),
        origin_mm=o_mm,
        voxel_mm=v_mm,
        pores_mm=(o_mm + ijk.astype(float) * v_mm).astype(np.float32),
        n_pore_voxels=n_interior_pores,
        layer_um=float(layer_m) * 1e6,
        hatch_um=float(hatch_m) * 1e6,
    )


def view3d_from_vti(vti_path: str, npz_path: str) -> None:
    """Build view3d.npz from a Raptor VTI (for runs made before view3d existed)."""
    import pyvista as pv

    img = pv.read(vti_path)
    d0, d1, d2 = img.dimensions  # VTK index axes = Raptor (z, y, x)
    phase = np.asarray(img["Phase"]).reshape(d2, d1, d0)
    save_view3d_npz(
        npz_path,
        phase,
        origin_m=np.asarray(img.origin, dtype=float),
        voxel_m=float(img.spacing[0]),
    )


PORE_TYPE_STYLE = {
    "raptor_lof": ("LoF (Raptor)", "#E41A1C"),
    "keyhole": ("Keyhole (CMU)", "#FF8C00"),
    "gas": ("Gas (CMU)", "#1B7F4E"),
    "cmu": ("Keyhole/gas (CMU)", "#8E44AD"),
    "edge": ("Edge (scan on/off)", "#FF00FF"),
}


def pore_type_table(model_pores=None, raptor_pores=None):
    """One row per pore: type, x_mm, y_mm, z_mm, esd_um.

    Raptor's LoF pore objects replace the combined model's LoF; keyhole / gas come from
    the CMU draw and edge pores from scan on/off points.
    """
    import pandas as pd

    parts = []
    if raptor_pores is not None and len(raptor_pores):
        parts.append(pd.DataFrame({
            "type": "raptor_lof",
            "x_mm": raptor_pores["x_mm"].to_numpy(float),
            "y_mm": raptor_pores["y_mm"].to_numpy(float),
            "z_mm": raptor_pores["z_mm"].to_numpy(float),
            "esd_um": raptor_pores["esd_um"].to_numpy(float),
        }))
    if model_pores is not None and len(model_pores):
        src = model_pores["pore_source"].astype(str).str.lower()
        if "cmu_mechanism" in model_pores.columns:
            mech = model_pores["cmu_mechanism"].astype(str).str.lower()
            kind = np.where(mech.isin(["keyhole", "gas"]), mech, "cmu")
        else:
            kind = np.full(len(src), "cmu", dtype=object)
        kind = np.where(src == "edge", "edge", kind)
        keep = (src != "lof").to_numpy()
        esd_col = next((c for c in model_pores.columns if c.lower().startswith("equivalent")), None)
        mp = model_pores.loc[keep]
        parts.append(pd.DataFrame({
            "type": np.asarray(kind)[keep],
            "x_mm": mp["x_mm"].to_numpy(float),
            "y_mm": mp["y_mm"].to_numpy(float),
            "z_mm": mp["z_mm"].to_numpy(float),
            "esd_um": mp[esd_col].to_numpy(float) if esd_col else np.nan,
        }))
    if not parts:
        return pd.DataFrame(columns=["type", "x_mm", "y_mm", "z_mm", "esd_um"])
    return pd.concat(parts, ignore_index=True)


def pores_in_raptor_domain(df, origin_mm, size_mm, *, min_slab_mm: float = 2.0):
    """Combined-model pores inside the Raptor box (plate mm).

    Thin boxes (e.g. an STL band of a few layers) are widened in z to at least
    ``min_slab_mm`` about their centre so a useful number of pores is shown.
    Returns (subset DataFrame, (z_lo, z_hi)).
    """
    o = np.asarray(origin_mm, dtype=float)
    s = np.asarray(size_mm, dtype=float)
    zc = o[2] + 0.5 * s[2]
    half = 0.5 * max(s[2], float(min_slab_mm))
    z_lo, z_hi = zc - half, zc + half
    x = df["x_mm"].to_numpy(dtype=float)
    y = df["y_mm"].to_numpy(dtype=float)
    z = df["z_mm"].to_numpy(dtype=float)
    m = (x >= o[0]) & (x <= o[0] + s[0]) & (y >= o[1]) & (y <= o[1] + s[1]) & (z >= z_lo) & (z <= z_hi)
    return df.loc[m], (float(z_lo), float(z_hi))


def _outline_trace(outline_npz: str, *, z_mm: float, origin_mm):
    """Cake-slice / part silhouette as a dark line at ``z_mm`` (plate mm).

    ``segments_mm`` in the npz are Raptor-local (domain origin at 0); ``origin_mm``
    is the plate origin of the Raptor box from view3d.npz.
    """
    import plotly.graph_objects as go

    data = np.load(outline_npz)
    segs = np.asarray(data["segments_mm"], dtype=float)
    if segs.size == 0:
        return None
    o = np.asarray(origin_mm, dtype=float)
    xs, ys, zs = [], [], []
    for x0_, y0_, x1_, y1_ in segs[:, :4]:
        xs += [o[0] + float(x0_), o[0] + float(x1_), None]
        ys += [o[1] + float(y0_), o[1] + float(y1_), None]
        zs += [z_mm, z_mm, None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="#0B3D91", width=4),
        name="Cake-slice / part outline",
        hoverinfo="skip",
    )


def _scan_path_trace(
    path_csv: str,
    *,
    origin_mm: Optional[np.ndarray] = None,
    max_segments: int = 10_000,
    max_layers: int = 24,
    color: str = "#1565C0",
):
    """Raptor hatch segments from ``path_vectors.csv`` (meters, RVE-local) → plate mm lines.

    Full-height runs write 10^5+ segments; we keep an even layer subsample so Plotly stays light.
    """
    import csv

    import plotly.graph_objects as go

    if not path_csv or not os.path.isfile(path_csv):
        return None
    rows: list[tuple[float, float, float, float, float, float, int]] = []
    with open(path_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    (
                        float(row["x0"]),
                        float(row["y0"]),
                        float(row["z0"]),
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["z1"]),
                        int(float(row.get("layer") or 0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    # Unit / frame: Raptor path_vectors are meters in the RVE box; Combined scan_vectors are plate mm
    coords = arr[:, :6]
    cmax = float(np.nanmax(np.abs(coords)))
    o = np.asarray(origin_mm if origin_mm is not None else [0.0, 0.0, 0.0], dtype=float).reshape(3)
    if cmax < 2.0:
        coords = coords * 1000.0
        coords[:, 0] += o[0]
        coords[:, 1] += o[1]
        coords[:, 2] += o[2]
        coords[:, 3] += o[0]
        coords[:, 4] += o[1]
        coords[:, 5] += o[2]
    elif cmax < 200.0 and float(o[0]) > 50.0 and float(np.nanmin(coords[:, 0])) < 50.0:
        coords[:, 0] += o[0]
        coords[:, 1] += o[1]
        coords[:, 2] += o[2]
        coords[:, 3] += o[0]
        coords[:, 4] += o[1]
        coords[:, 5] += o[2]
    layers = arr[:, 6].astype(int)
    uniq = np.unique(layers)
    if len(uniq) > max_layers:
        pick = np.unique(np.round(np.linspace(0, len(uniq) - 1, max_layers)).astype(int))
        keep_layers = set(int(uniq[i]) for i in pick)
    else:
        keep_layers = set(int(x) for x in uniq)
    mask = np.array([int(L) in keep_layers for L in layers], dtype=bool)
    sel = coords[mask]
    if len(sel) > max_segments:
        step = int(np.ceil(len(sel) / float(max_segments)))
        sel = sel[::step]
    if len(sel) == 0:
        return None
    # Plotly line breaks via None between segments
    xs: list = []
    ys: list = []
    zs: list = []
    for x0, y0, z0, x1, y1, z1 in sel:
        xs.extend([round(float(x0), 4), round(float(x1), 4), None])
        ys.extend([round(float(y0), 4), round(float(y1), 4), None])
        zs.extend([round(float(z0), 4), round(float(z1), 4), None])
    n_seg = len(sel)
    n_lay = len(keep_layers)
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=2),
        name=f"Scan path ({n_seg:,} segs / {n_lay} layers)",
        hoverinfo="skip",
        visible=True,
    )


def _stl_mesh_trace(stl_path: str, *, max_faces: int = 60_000, opacity: float = 0.55):
    """Solid cake-slice / part from the fitted STL (plate mm)."""
    import plotly.graph_objects as go
    import pyvista as pv

    mesh = pv.read(stl_path)
    try:
        mesh = mesh.triangulate()
    except Exception:
        pass
    n_faces = int(getattr(mesh, "n_cells", 0) or 0)
    if n_faces > max_faces:
        try:
            mesh = mesh.decimate(1.0 - max_faces / float(n_faces))
            mesh = mesh.triangulate()
        except Exception:
            pass
    faces = np.asarray(mesh.faces)
    if faces.size == 0:
        return None
    # PyVista: [3, i, j, k, 3, i, j, k, ...]
    if faces.ndim == 1:
        tri = faces.reshape(-1, 4)[:, 1:4]
    else:
        tri = faces[:, 1:4] if faces.shape[1] == 4 else faces
    pts = np.asarray(mesh.points, dtype=float)
    return go.Mesh3d(
        x=np.round(pts[:, 0], 4),
        y=np.round(pts[:, 1], 4),
        z=np.round(pts[:, 2], 4),
        i=tri[:, 0],
        j=tri[:, 1],
        k=tri[:, 2],
        color="rgb(179,179,179)",
        opacity=float(opacity),
        name="Cake-slice / part (STL)",
        flatshading=True,
        lighting=dict(ambient=0.75, diffuse=0.45, specular=0.05),
        hoverinfo="skip",
        showlegend=True,
    )


def make_plotly_phase_figure(
    npz_path: str,
    *,
    title: str = "Raptor result (drag to rotate)",
    model_pores=None,
    raptor_pores=None,
    pore_types=None,
    model_z_range=None,
    outline_npz: Optional[str] = None,
    stl_path: Optional[str] = None,
    path_csv: Optional[str] = None,
    max_model_points: int = 20_000,
    max_face_axis: int = 120,
    max_raptor_pores: int = 25_000,
):
    """Rotatable Plotly figure: phase on the part's outer faces + LoF pores inside.

    ``pore_types`` (see ``pore_type_table``; built from ``model_pores`` +
    ``raptor_pores`` if not given) adds Raptor LoF, CMU keyhole / gas and edge pores
    for the "All pore sources" view, one legend entry per type.

    ``outline_npz`` (from ``stl_outline.npz``) draws the cake-slice / part contour so a
    thin mid-band still shows the 1/N footprint.

    ``stl_path`` (fitted cake-slice / part) draws the real 3D solid instead of the
    AABB face maps — needed for full-height cake slices, where box faces look like
    floating slabs.

    ``path_csv`` (``path_vectors.csv`` next to the VTI) overlays Raptor hatch scan paths
    in plate mm (subsampled for browser size).
    """
    import plotly.graph_objects as go

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
    x0, x1 = o[0], o[0] + (nx - 1) * v
    y0, y1 = o[1], o[1] + (ny - 1) * v
    z0, z1 = o[2], o[2] + (nz - 1) * v

    # Discrete phase colours: value v -> (v + 0.5) / 4 on the colourscale
    cols = ["rgb(255,0,0)", "rgb(179,179,179)", "rgb(51,51,51)", "rgb(255,255,0)"]
    cscale = []
    for i, c in enumerate(cols):
        cscale += [[i / 4.0, c], [(i + 1) / 4.0, c]]
    surf_kw = dict(
        colorscale=cscale,
        cmin=-0.5,
        cmax=3.5,
        showscale=False,
        hoverinfo="skip",
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        opacity=1.0,
    )

    def _plane(value, shape_like, colors):
        # Voxels outside the STL cross-section become holes (NaN coordinates)
        arr = np.full(shape_like.shape, float(value))
        arr[np.asarray(colors) >= PHASE_OUTSIDE] = np.nan
        return arr

    # Browser payload / WebGL cost scales with face samples; keep each axis <= max_face_axis
    def _step(n):
        return max(1, int(np.ceil(n / float(max_face_axis))))

    kx, ky, kz = _step(len(xs)), _step(len(ys)), _step(len(zs))
    xs, ys, zs = xs[::kx], ys[::ky], zs[::kz]
    def _shell(face: np.ndarray) -> np.ndarray:
        # Outer skin is the scan leaving the part. Draw that solid as melted metal.
        out = np.array(face, copy=True)
        out[(out == 0) | (out == 2) | (out == 3)] = 1
        return out

    fx = {k: _shell(np.asarray(d[k])[::ky, ::kz]) for k in ("face_x0", "face_x1")}
    fy = {k: _shell(np.asarray(d[k])[::kx, ::kz]) for k in ("face_y0", "face_y1")}
    fz = {k: _shell(np.asarray(d[k])[::kx, ::ky]) for k in ("face_z0", "face_z1")}

    def _r(a):
        return np.round(np.asarray(a, dtype=float), 4)

    use_stl = bool(stl_path and os.path.isfile(stl_path))
    faces = []
    if use_stl:
        mesh_tr = _stl_mesh_trace(stl_path)
        if mesh_tr is not None:
            faces = [mesh_tr]
        else:
            use_stl = False
    if not use_stl:
        # x faces: surfacecolor over (y, z)
        Yx, Zx = np.meshgrid(ys, zs, indexing="ij")
        for xv, key in ((x0, "face_x0"), (x1, "face_x1")):
            faces.append(go.Surface(x=_r(_plane(xv, Yx, fx[key])), y=_r(Yx), z=_r(Zx), surfacecolor=fx[key], **surf_kw))
        Xy, Zy = np.meshgrid(xs, zs, indexing="ij")
        for yv, key in ((y0, "face_y0"), (y1, "face_y1")):
            faces.append(go.Surface(x=_r(Xy), y=_r(_plane(yv, Xy, fy[key])), z=_r(Zy), surfacecolor=fy[key], **surf_kw))
        Xz, Yz = np.meshgrid(xs, ys, indexing="ij")
        for zv, key in ((z0, "face_z0"), (z1, "face_z1")):
            faces.append(go.Surface(x=_r(Xz), y=_r(Yz), z=_r(_plane(zv, Xz, fz[key])), surfacecolor=fz[key], **surf_kw))

    pores = np.asarray(d["pores_mm"], dtype=float)
    if len(pores) > max_raptor_pores:
        pores = pores[np.random.default_rng(0).choice(len(pores), max_raptor_pores, replace=False)]
    pores = np.round(pores, 4)
    n_pv = int(d["n_pore_voxels"])
    pore_trace = go.Scatter3d(
        x=pores[:, 0] if len(pores) else [],
        y=pores[:, 1] if len(pores) else [],
        z=pores[:, 2] if len(pores) else [],
        mode="markers",
        marker=dict(size=2, color="red"),
        name=f"LoF pores ({n_pv:,} voxels)",
        hovertemplate="LoF pore<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}) mm<extra></extra>",
        visible=False,
    )
    # Legend swatches for the phase colours (AABB face mode only)
    legend = []
    if not use_stl:
        legend = [
            go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(size=8, color=c, symbol="square"), name=n,
            )
            for c, n in zip(cols[1:], ["Melted", "Melt boundary", "Track overlap"])
        ]
    if pore_types is None and (model_pores is not None or raptor_pores is not None):
        pore_types = pore_type_table(model_pores, raptor_pores)
    model_traces = []
    has_raptor_objects = False
    if pore_types is not None and len(pore_types):
        kinds = pore_types["type"].astype(str).to_numpy()
        has_raptor_objects = bool((kinds == "raptor_lof").any())
        zr = ""
        if model_z_range is not None:
            zr = f" (CMU/edge z {model_z_range[0]:.2f}–{model_z_range[1]:.2f} mm)"
        for key, (label, color) in PORE_TYPE_STYLE.items():
            m = kinds == key
            if not m.any():
                continue
            sub = pore_types.loc[m]
            if key == "raptor_lof" and len(sub):
                ijk = (sub[["x_mm", "y_mm", "z_mm"]].to_numpy(float) - o) / v
                dx = np.minimum(ijk[:, 0], (nx - 1) - ijk[:, 0])
                dy = np.minimum(ijk[:, 1], (ny - 1) - ijk[:, 1])
                dz = np.minimum(ijk[:, 2], (nz - 1) - ijk[:, 2])
                sub = sub.loc[(np.minimum(np.minimum(dx, dy), dz) * v) > 0.5]
            n_src = len(sub)
            if n_src > max_model_points:
                sub = sub.sample(max_model_points, random_state=0)
            shown = "" if n_src == len(sub) else f" (showing {len(sub):,})"
            esd = sub["esd_um"].to_numpy(dtype=float)
            size = np.where(np.isfinite(esd), np.clip(esd / 15.0, 3, 10), 4)
            model_traces.append(
                go.Scatter3d(
                    x=np.round(sub["x_mm"].to_numpy(float), 4),
                    y=np.round(sub["y_mm"].to_numpy(float), 4),
                    z=np.round(sub["z_mm"].to_numpy(float), 4),
                    mode="markers",
                    marker=dict(size=size, color=color, line=dict(width=0)),
                    name=f"{label}: {n_src:,}{shown}" + (zr if key != "raptor_lof" else ""),
                    customdata=np.round(esd, 1),
                    hovertemplate=(
                        f"{label}<br>(%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}}) mm"
                        "<br>ESD %{customdata} µm<extra></extra>"
                    ),
                    visible=True,
                )
            )
    if model_traces:
        for face in faces:
            face.opacity = 0.18

    outline_list = []
    # Outline overlays are for thin-band AABB views; the STL mesh already is the part
    if not use_stl and outline_npz and os.path.isfile(outline_npz):
        try:
            outline_tr = _outline_trace(outline_npz, z_mm=float(z1), origin_mm=o)
            if outline_tr is not None:
                outline_list = [outline_tr]
        except Exception:
            outline_list = []

    path_list = []
    if path_csv and os.path.isfile(path_csv):
        try:
            path_tr = _scan_path_trace(path_csv, origin_mm=o)
            if path_tr is not None:
                path_list = [path_tr]
        except Exception:
            path_list = []
    has_paths = bool(path_list)
    # Default: show hatch paths when available (faded solid so stripes are visible)
    if has_paths and path_list:
        path_list[0].visible = True
        if use_stl and faces:
            faces[0].opacity = 0.28

    fig = go.Figure(
        data=faces + [pore_trace] + model_traces + path_list + legend + outline_list
    )
    n_faces = len(faces)
    n_model = len(model_traces)
    n_paths = len(path_list)
    n_outline = len(outline_list)

    def _vis(
        face_opacity: float,
        show_faces: bool,
        show_pores: bool,
        show_model: bool = False,
        show_paths: bool = False,
    ):
        # Mesh3d opacity is per-trace; Surface mode uses the same restyle keys
        return [
            {
                "opacity": (
                    [face_opacity] * n_faces
                    + [1.0] * (1 + n_model + n_paths + len(legend) + n_outline)
                ),
                "visible": [show_faces] * n_faces
                + [show_pores]
                + [show_model] * n_model
                + [show_paths] * n_paths
                + [show_faces] * len(legend)
                + [True] * n_outline,
            }
        ]

    build_label = "Part (STL)" if use_stl else "Melted build"
    buttons = []
    if has_paths:
        buttons.append(
            dict(
                label="Scan paths",
                method="restyle",
                args=_vis(0.28 if use_stl else 0.35, True, False, False, True),
            )
        )
    buttons.extend(
        [
            dict(
                label=build_label,
                method="restyle",
                args=_vis(0.55 if use_stl else 1.0, True, False, False, False),
            ),
            dict(
                label="See-through + pores",
                method="restyle",
                args=_vis(0.22 if use_stl else 0.18, True, True, False, has_paths),
            ),
            dict(
                label="Pores only",
                method="restyle",
                args=_vis(1.0, False, True, False, False),
            ),
        ]
    )
    if n_model:
        buttons.append(
            dict(
                label="All pore sources",
                method="restyle",
                args=_vis(
                    0.22 if use_stl else 0.18,
                    True,
                    not has_raptor_objects,
                    True,
                    False,
                ),
            )
        )

    # Pull the camera back in proportion to the box so a tall pillar / STL
    # stays fully inside the view (Plotly normalizes the longest side to 1).
    dx = max(float(x1 - x0), 1e-6)
    dy = max(float(y1 - y0), 1e-6)
    dz = max(float(z1 - z0), 1e-6)
    span = max(dx, dy, dz)
    radius = 0.5 * float(np.hypot(np.hypot(dx / span, dy / span), dz / span))
    dist = max(2.4, 2.5 * radius * 1.65)
    ux, uy, uz = 1.2, 1.05, 0.72
    un = float(np.hypot(np.hypot(ux, uy), uz))
    fig.update_layout(
        title=title,
        height=760,
        margin=dict(l=8, r=8, t=56, b=28),
        paper_bgcolor="white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=ux / un * dist, y=uy / un * dist, z=uz / un * dist),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                xanchor="center",
                y=1.0,
                yanchor="bottom",
                buttons=buttons,
            )
        ],
    )
    return fig


def _phase_cmap():
    from matplotlib.colors import ListedColormap

    return ListedColormap(PHASE_COLORS, name="raptor_phase")


def _add_phase_volume(pl, vol, *, show_bar: bool = True):
    return pl.add_mesh(
        vol,
        scalars="Phase",
        cmap=_phase_cmap(),
        clim=(0, 4),
        categories=True,
        n_colors=4,
        annotations=PHASE_ANNOTATIONS,
        interpolate_before_map=False,
        lighting=False,
        show_scalar_bar=show_bar,
        scalar_bar_args={"title": "Phase", "n_labels": 0},
    )


def _add_pores(pl, pore_pts, *, point_size: float = 3.0):
    import pyvista as pv

    if pore_pts is None or not len(pore_pts):
        return None
    return pl.add_mesh(
        pv.PolyData(pore_pts),
        color=PHASE_COLORS[0],
        point_size=point_size,
        render_points_as_spheres=True,
        name="LoF pores",
    )


def _count_paths(path_csv: str) -> int:
    try:
        with open(path_csv, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return 0


def _layout_two_panels(pl, vol, pore_pts, path_csv: Optional[str]):
    del path_csv  # scan-path lines are not drawn on the Raptor result
    n_pores = 0 if pore_pts is None else len(pore_pts)
    outline = vol.outline()

    pl.subplot(0, 0)
    pl.add_text("Whole build: melted phase", font_size=10, color="black")
    vol_actor = _add_phase_volume(pl, vol)
    pl.add_mesh(outline, color="black", line_width=1)
    pl.add_axes()
    pl.camera_position = "iso"

    pl.subplot(0, 1)
    pl.add_text(f"LoF pores in the whole build ({n_pores:,} voxels)", font_size=10, color="black")
    pl.add_mesh(vol.outline(), color="black", line_width=1)
    try:
        shell = vol.extract_surface(algorithm="dataset_surface")
    except TypeError:
        shell = vol.extract_surface()
    pl.add_mesh(shell, color=PHASE_COLORS[1], opacity=0.08, lighting=False)
    _add_pores(pl, pore_pts)
    pl.add_axes()
    pl.camera_position = "iso"
    pl.link_views()
    return vol_actor


def write_phase_screenshot(vti_path: str, png_path: str) -> None:
    """Off-screen two-panel image: full melted build | all LoF pores."""
    import pyvista as pv
    from raptor.path_viz import sibling_path_vectors_csv

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv.OFF_SCREEN = True
    vol, pore_pts, _ = load_phase_view(vti_path)
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1500, 850), border=False)
    pl.set_background("white")
    _layout_two_panels(pl, vol, pore_pts, sibling_path_vectors_csv(vti_path))
    pl.screenshot(png_path)
    pl.close()


def show_interactive(vti_path: str, max_render_voxels: int = 4_000_000) -> None:
    """Interactive window: whole melted build (with slice plane) | all LoF pores."""
    import pyvista as pv
    from raptor.path_viz import sibling_path_vectors_csv

    vol, pore_pts, rate = load_phase_view(vti_path, max_render_voxels=max_render_voxels)
    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 850), title="Raptor whole build")
    pl.set_background("white")
    vol_actor = _layout_two_panels(pl, vol, pore_pts, sibling_path_vectors_csv(vti_path))

    # Slice plane on the melted volume: starts outside the part so the full build shows
    pl.subplot(0, 0)

    def _clip(normal, origin):
        vol_actor.mapper.SetInputData(vol.clip(normal=normal, origin=origin))

    b = vol.bounds
    pl.add_plane_widget(
        _clip,
        normal=(1, 0, 0),
        origin=(b[1], 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])),
        bounds=b,
        color="blue",
        outline_translation=False,
    )
    if rate > 1:
        pl.add_text(
            f"(volume shown every {rate} voxels; pores at full resolution)",
            position="lower_left",
            font_size=8,
            color="gray",
        )
    pl.show()
