"""Laser scan-vector patterns and edge-point pore sampling (cube / cuboid first)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ESD_COL = "Equivalent Spherical Diameter (um)"

# Named strategies → base hatch angle (deg, 0 = lines along +X / spaced in Y)
STRATEGY_ANGLES = {
    "horizontal": 0.0,
    "vertical": 90.0,
    "diagonal": 45.0,
    "diagonal_neg": -45.0,
}


@dataclass
class ScanVector:
    x0: float
    y0: float
    z: float
    x1: float
    y1: float
    layer: int
    build_index: int
    strategy: str
    angle_deg: float = 0.0

    @property
    def start(self) -> Tuple[float, float, float]:
        return (self.x0, self.y0, self.z)

    @property
    def end(self) -> Tuple[float, float, float]:
        return (self.x1, self.y1, self.z)


def _hatch_lines_1d(low: float, high: float, spacing: float) -> np.ndarray:
    if high <= low:
        return np.array([(low + high) / 2.0])
    n = max(1, int(np.floor((high - low) / max(spacing, 1e-9))) + 1)
    return np.linspace(low + spacing / 2.0, high - spacing / 2.0, n)


def _clip_line_to_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Cohen–Sutherland style clip of a segment to an axis-aligned rectangle."""
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x, y):
        c = INSIDE
        if x < xmin:
            c |= LEFT
        elif x > xmax:
            c |= RIGHT
        if y < ymin:
            c |= BOTTOM
        elif y > ymax:
            c |= TOP
        return c

    c0, c1 = code(x0, y0), code(x1, y1)
    for _ in range(20):
        if not (c0 | c1):
            return (x0, y0, x1, y1)
        if c0 & c1:
            return None
        c_out = c0 or c1
        if c_out & TOP:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0 + 1e-18)
            y = ymax
        elif c_out & BOTTOM:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0 + 1e-18)
            y = ymin
        elif c_out & RIGHT:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0 + 1e-18)
            x = xmax
        else:
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0 + 1e-18)
            x = xmin
        if c_out == c0:
            x0, y0 = x, y
            c0 = code(x0, y0)
        else:
            x1, y1 = x, y
            c1 = code(x1, y1)
    return None


def hatch_at_angle(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    z: float,
    *,
    angle_deg: float,
    hatch_spacing_mm: float,
    layer: int,
    build_index: int,
    strategy: str,
    serpentine: bool = True,
) -> List[ScanVector]:
    """
    Fill a rectangle with parallel hatch lines at ``angle_deg``
    (0° = horizontal along +X, 90° = vertical along +Y).
    """
    spacing = float(max(hatch_spacing_mm, 1e-6))
    ang = np.deg2rad(float(angle_deg))
    # Direction along the scan line
    ux, uy = float(np.cos(ang)), float(np.sin(ang))
    # Spacing direction (perpendicular)
    px, py = -uy, ux

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    # Project corners onto perpendicular axis to get offset range
    corners = np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmin, ymax],
            [xmax, ymax],
        ],
        dtype=float,
    )
    offs = (corners[:, 0] - cx) * px + (corners[:, 1] - cy) * py
    o_min, o_max = float(offs.min()), float(offs.max())
    # Length along scan to overshoot the AABB
    diag = float(np.hypot(xmax - xmin, ymax - ymin)) + spacing

    vectors: List[ScanVector] = []
    offsets = _hatch_lines_1d(o_min, o_max, spacing)
    for i, o in enumerate(offsets):
        mx = cx + o * px
        my = cy + o * py
        a = (mx - ux * diag, my - uy * diag)
        b = (mx + ux * diag, my + uy * diag)
        clipped = _clip_line_to_rect(a[0], a[1], b[0], b[1], xmin, xmax, ymin, ymax)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        if serpentine and (i % 2 == 1):
            x0, y0, x1, y1 = x1, y1, x0, y0
        # Skip degenerate
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 < 1e-12:
            continue
        vectors.append(
            ScanVector(
                float(x0),
                float(y0),
                float(z),
                float(x1),
                float(y1),
                int(layer),
                int(build_index),
                str(strategy),
                float(angle_deg),
            )
        )
    return vectors


def _split_into_stripes(
    vectors: List[ScanVector],
    *,
    angle_deg: float,
    stripe_width_mm: float,
    shift_mm: float = 0.0,
) -> List[ScanVector]:
    """Cut hatch lines at stripe boundaries along the scan direction; scan stripe by stripe."""
    w = float(stripe_width_mm)
    if w <= 0 or not vectors:
        return vectors
    ang = np.deg2rad(float(angle_deg))
    ux, uy = float(np.cos(ang)), float(np.sin(ang))
    pieces: List[Tuple[int, int, ScanVector]] = []
    for order, v in enumerate(vectors):
        u0 = v.x0 * ux + v.y0 * uy
        u1 = v.x1 * ux + v.y1 * uy
        if abs(u1 - u0) < 1e-12:
            continue
        lo, hi = min(u0, u1), max(u0, u1)
        k0 = int(np.floor((lo - shift_mm) / w))
        k1 = int(np.floor((hi - shift_mm) / w - 1e-12))
        for k in range(k0, k1 + 1):
            a = max(lo, k * w + shift_mm)
            b = min(hi, (k + 1) * w + shift_mm)
            if b - a < 1e-9:
                continue
            ta = (a - u0) / (u1 - u0)
            tb = (b - u0) / (u1 - u0)
            if ta > tb:
                ta, tb = tb, ta
            dx, dy = v.x1 - v.x0, v.y1 - v.y0
            pieces.append(
                (
                    k,
                    order,
                    ScanVector(
                        v.x0 + ta * dx,
                        v.y0 + ta * dy,
                        v.z,
                        v.x0 + tb * dx,
                        v.y0 + tb * dy,
                        v.layer,
                        v.build_index,
                        v.strategy,
                        v.angle_deg,
                    ),
                )
            )
    pieces.sort(key=lambda p: (p[0], p[1]))
    return [p[2] for p in pieces]


def generate_scan_vectors_rect(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    z_levels: Sequence[float],
    *,
    strategy: str = "horizontal",
    hatch_spacing_mm: float = 0.14,
    build_index: int = 0,
    batch_rows: int = 4,
    batch_cols: int = 4,
    base_angle_deg: Optional[float] = None,
    layer_rotation_deg: float = 67.0,
    serpentine: bool = True,
    stripe_width_mm: float = 10.0,
) -> List[ScanVector]:
    """
    Hatch vectors covering a rectangular XY footprint at each z level.

    Strategies
    ----------
    horizontal / vertical / diagonal / diagonal_neg : fixed named angles
    rotate / rotating : base_angle + layer * layer_rotation_deg (build-direction rotate)
    crosshatch : two passes at angle and angle+90 on each layer
    batch : subdivide into cells, horizontal hatch in each
    stripes : rotate + hatch cut into stripes of ``stripe_width_mm`` (EOS-style)
    """
    strategy = str(strategy or "horizontal").lower().strip()
    spacing = float(max(hatch_spacing_mm, 1e-6))
    vectors: List[ScanVector] = []

    if base_angle_deg is None:
        base_angle_deg = STRATEGY_ANGLES.get(strategy, 0.0)
    base = float(base_angle_deg)
    rot = float(layer_rotation_deg)

    for li, z in enumerate(z_levels):
        zf = float(z)
        if strategy == "stripes":
            angle = base + li * rot
            lines = hatch_at_angle(
                xmin,
                xmax,
                ymin,
                ymax,
                zf,
                angle_deg=angle,
                hatch_spacing_mm=spacing,
                layer=li,
                build_index=build_index,
                strategy=strategy,
                serpentine=serpentine,
            )
            w = float(stripe_width_mm or 0.0)
            vectors.extend(
                _split_into_stripes(
                    lines,
                    angle_deg=angle,
                    stripe_width_mm=w,
                    shift_mm=((li * 0.37) % 1.0) * w,
                )
            )
        elif strategy == "batch":
            br = max(1, int(batch_rows))
            bc = max(1, int(batch_cols))
            dx = (xmax - xmin) / bc
            dy = (ymax - ymin) / br
            angle = base + li * rot
            for r in range(br):
                for c in range(bc):
                    x0 = xmin + c * dx
                    x1 = xmin + (c + 1) * dx
                    y0 = ymin + r * dy
                    y1 = ymin + (r + 1) * dy
                    vectors.extend(
                        hatch_at_angle(
                            x0,
                            x1,
                            y0,
                            y1,
                            zf,
                            angle_deg=angle,
                            hatch_spacing_mm=spacing,
                            layer=li,
                            build_index=build_index,
                            strategy=strategy,
                            serpentine=serpentine,
                        )
                    )
        elif strategy == "crosshatch":
            angle = base + li * rot
            vectors.extend(
                hatch_at_angle(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    zf,
                    angle_deg=angle,
                    hatch_spacing_mm=spacing,
                    layer=li,
                    build_index=build_index,
                    strategy=strategy,
                    serpentine=serpentine,
                )
            )
            vectors.extend(
                hatch_at_angle(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    zf,
                    angle_deg=angle + 90.0,
                    hatch_spacing_mm=spacing,
                    layer=li,
                    build_index=build_index,
                    strategy=strategy,
                    serpentine=serpentine,
                )
            )
        else:
            # Named fixed, or rotating (build-direction pattern rotate)
            if strategy in ("rotate", "rotating", "island"):
                angle = base + li * rot
            elif strategy in STRATEGY_ANGLES:
                # Still allow optional layer rotation on named strategies
                angle = STRATEGY_ANGLES[strategy] + li * rot
            else:
                angle = base + li * rot
            vectors.extend(
                hatch_at_angle(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    zf,
                    angle_deg=angle,
                    hatch_spacing_mm=spacing,
                    layer=li,
                    build_index=build_index,
                    strategy=strategy,
                    serpentine=serpentine,
                )
            )

    return vectors


def layer_z_levels(
    height_mm: float,
    layer_thickness_mm: float = 0.03,
    max_layers: Optional[int] = None,
) -> np.ndarray:
    n_layers = max(1, int(np.ceil(height_mm / max(layer_thickness_mm, 1e-6))))
    if max_layers is not None:
        n_layers = min(n_layers, int(max_layers))
    return np.linspace(
        layer_thickness_mm / 2.0,
        max(height_mm - layer_thickness_mm / 2.0, layer_thickness_mm / 2.0),
        n_layers,
    )


def vectors_for_builds(
    builds,
    *,
    strategy: str = "horizontal",
    hatch_spacing_mm: float = 0.14,
    layer_thickness_mm: float = 0.03,
    max_layers: Optional[int] = None,
    batch_rows: int = 4,
    batch_cols: int = 4,
    base_angle_deg: Optional[float] = None,
    layer_rotation_deg: float = 67.0,
    max_vectors: int = 25000,
    stripe_width_mm: float = 10.0,
) -> List[ScanVector]:
    """Generate scan vectors for every plate build (cube/cuboid footprints)."""
    all_v: List[ScanVector] = []
    for b in builds:
        half = b.size_mm / 2.0
        if hasattr(b, "length_mm") and hasattr(b, "width_mm") and b.length_mm and b.width_mm:
            half_x = b.length_mm / 2.0
            half_y = b.width_mm / 2.0
        else:
            half_x = half_y = half
        xmin, xmax = b.cx_mm - half_x, b.cx_mm + half_x
        ymin, ymax = b.cy_mm - half_y, b.cy_mm + half_y
        z_levels = layer_z_levels(b.height_mm, layer_thickness_mm, max_layers)
        all_v.extend(
            generate_scan_vectors_rect(
                xmin,
                xmax,
                ymin,
                ymax,
                z_levels,
                strategy=strategy,
                hatch_spacing_mm=hatch_spacing_mm,
                build_index=b.index,
                batch_rows=batch_rows,
                batch_cols=batch_cols,
                base_angle_deg=base_angle_deg,
                layer_rotation_deg=layer_rotation_deg,
                stripe_width_mm=stripe_width_mm,
            )
        )
        if len(all_v) >= max_vectors:
            return all_v[:max_vectors]
    return all_v


def vectors_to_dataframe(vectors: Sequence[ScanVector]) -> pd.DataFrame:
    rows = []
    for i, v in enumerate(vectors):
        rows.append(
            {
                "vector_id": i,
                "x0": v.x0,
                "y0": v.y0,
                "z0": v.z,
                "x1": v.x1,
                "y1": v.y1,
                "z1": v.z,
                "layer": v.layer,
                "build_index": v.build_index,
                "strategy": v.strategy,
                "angle_deg": v.angle_deg,
            }
        )
    return pd.DataFrame(rows)


def sample_edge_pores(
    vectors: Sequence[ScanVector],
    size_catalog: pd.DataFrame,
    *,
    n_edge_pores: int,
    seed: int = 42,
    endpoint_jitter_mm: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Place pores at randomly chosen scan-vector start/end points.

    Returns (edge_pores_df, vectors_df for plotting).
    """
    rng = np.random.default_rng(seed)
    vectors_df = vectors_to_dataframe(vectors)

    n_edge = int(max(0, n_edge_pores))
    if n_edge == 0 or len(vectors) == 0:
        empty = pd.DataFrame(
            columns=[
                ESD_COL,
                "pore_source",
                "x_mm",
                "y_mm",
                "z_mm",
                "build_index",
                "radius_mm",
                "volume_mm3",
                "pore_family",
                "vector_id",
                "endpoint",
            ]
        )
        return empty, vectors_df

    from integrated_gui.stl_slicer_scan import _inward_along_vector

    cand = []
    for i, v in enumerate(vectors):
        cand.append((i, "start", v.x0, v.y0, v.z, v.build_index, v.x1, v.y1))
        cand.append((i, "end", v.x1, v.y1, v.z, v.build_index, v.x0, v.y0))
    cand = np.array(cand, dtype=object)
    take_n = min(n_edge, len(cand))
    pick = rng.choice(len(cand), size=take_n, replace=False)

    catalog = size_catalog.copy()
    if ESD_COL not in catalog.columns:
        catalog[ESD_COL] = 20.0
    sizes = catalog.sample(take_n, replace=len(catalog) < take_n, random_state=int(rng.integers(1e9)))

    rows = []
    for k, idx in enumerate(pick):
        vid, endpoint, x, y, z, bi, ox, oy = cand[int(idx)]
        xs, ys = _inward_along_vector([x], [y], [ox], [oy], rng, endpoint_jitter_mm)
        x, y = float(xs[0]), float(ys[0])
        z = float(max(0.0, z))
        esd = float(sizes.iloc[k][ESD_COL])
        r_mm = (esd / 2.0) * 1e-3
        vol = (4.0 / 3.0) * np.pi * r_mm**3
        rows.append(
            {
                ESD_COL: esd,
                "pore_source": "edge",
                "x_mm": x,
                "y_mm": y,
                "z_mm": z,
                "build_index": int(bi),
                "radius_mm": r_mm,
                "volume_mm3": vol,
                "pore_family": "edge",
                "vector_id": int(vid),
                "endpoint": str(endpoint),
            }
        )
    return pd.DataFrame(rows), vectors_df


def _points_inside_mesh(mesh, pts: np.ndarray, *, surface_band_mm: float = 0.75) -> np.ndarray:
    """
    Boolean mask for points that belong to the printable solid at this Z.

    Prefer signed distance (d <= surface_band): captures closed solid interiors and
    thin-wall shells where select_interior_points often fails on hatch midpoints.
    """
    import pyvista as pv

    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    cloud = pv.PolyData(np.asarray(pts, dtype=float))
    try:
        dist = cloud.compute_implicit_distance(mesh)
        d = np.asarray(dist["implicit_distance"], dtype=float)
        # Negative / near-zero = inside or on the surface
        return d <= float(surface_band_mm)
    except Exception:
        pass
    try:
        if hasattr(cloud, "select_interior_points"):
            sel = cloud.select_interior_points(mesh, tolerance=1e-4, check_surface=False)
            key = "selected_points" if "selected_points" in sel.array_names else "SelectedPoints"
            return np.asarray(sel[key], dtype=bool)
        sel = cloud.select_enclosed_points(mesh, tolerance=1e-4, check_surface=False)
        return np.asarray(sel["SelectedPoints"], dtype=bool)
    except Exception:
        return np.zeros(len(pts), dtype=bool)


def _clip_hatch_to_mesh(
    mesh,
    vectors: List[ScanVector],
    *,
    samples_per_mm: float = 2.0,
    surface_band_mm: float = 0.75,
) -> List[ScanVector]:
    """Keep only contiguous interior / near-surface segments of each hatch line."""
    out: List[ScanVector] = []
    for v in vectors:
        length = float(np.hypot(v.x1 - v.x0, v.y1 - v.y0))
        n = max(12, int(np.ceil(length * samples_per_mm)) + 1)
        t = np.linspace(0.0, 1.0, n)
        xs = v.x0 + t * (v.x1 - v.x0)
        ys = v.y0 + t * (v.y1 - v.y0)
        zs = np.full(n, v.z)
        pts = np.column_stack([xs, ys, zs])
        inside = _points_inside_mesh(mesh, pts, surface_band_mm=surface_band_mm)
        if not inside.any():
            continue
        padded = np.concatenate([[False], inside, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        for s, e in zip(starts, ends):
            if e <= s:
                continue
            # Require a minimum length so we don't keep speckles
            seg_len = float(np.hypot(xs[e] - xs[s], ys[e] - ys[s]))
            if seg_len < 0.25:
                continue
            out.append(
                ScanVector(
                    float(xs[s]),
                    float(ys[s]),
                    float(v.z),
                    float(xs[e]),
                    float(ys[e]),
                    int(v.layer),
                    int(v.build_index),
                    str(v.strategy),
                    float(v.angle_deg),
                )
            )
    return out


def vectors_for_stl_mesh(
    mesh,
    *,
    strategy: str = "horizontal",
    hatch_spacing_mm: float = 0.14,
    layer_thickness_mm: float = 0.03,
    max_layers: Optional[int] = None,
    batch_rows: int = 4,
    batch_cols: int = 4,
    base_angle_deg: Optional[float] = None,
    layer_rotation_deg: float = 67.0,
    build_index: int = 0,
    max_vectors: int = 25000,
    samples_per_mm: float = 2.0,
    surface_band_mm: float = 1.0,
) -> List[ScanVector]:
    """
    Hatch an imported STL: generate AABB hatches at each layer Z, then clip
    each vector to the mesh solid / near-surface band so scans follow the CAD.
    """
    try:
        surf = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
    except Exception:
        surf = mesh
    b = np.asarray(surf.bounds, dtype=float)
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, b)
    height = max(zmax - zmin, layer_thickness_mm)
    z_levels = layer_z_levels(height, layer_thickness_mm, max_layers)
    z_levels = z_levels + zmin

    raw = generate_scan_vectors_rect(
        xmin,
        xmax,
        ymin,
        ymax,
        z_levels,
        strategy=strategy,
        hatch_spacing_mm=hatch_spacing_mm,
        build_index=build_index,
        batch_rows=batch_rows,
        batch_cols=batch_cols,
        base_angle_deg=base_angle_deg,
        layer_rotation_deg=layer_rotation_deg,
    )
    clipped = _clip_hatch_to_mesh(
        surf,
        raw,
        samples_per_mm=samples_per_mm,
        surface_band_mm=surface_band_mm,
    )
    if len(clipped) > max_vectors:
        return clipped[:max_vectors]
    return clipped

def make_scan_preview_figure(
    vectors_df: pd.DataFrame,
    edge_df: Optional[pd.DataFrame] = None,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    height_mm: float,
    layer_filter: Optional[int] = None,
    max_arrows: int = 400,
    title: str = "Scan vectors",
    stl_mesh=None,
    z_offset: float = 0.0,
):
    """Plotly 3D figure: scan arrows with clear START/END markers (+ optional edge pores)."""
    import plotly.graph_objects as go

    # Ensure PyVista never opens an interactive window from a Dash thread
    try:
        from .headless import configure_headless_plotting

        configure_headless_plotting()
    except Exception:
        pass

    fig = go.Figure()

    # Optional translucent STL body
    if stl_mesh is not None:
        try:
            import pyvista as pv

            mesh = stl_mesh
            try:
                try:
                    mesh = mesh.extract_surface(algorithm="dataset_surface").triangulate()
                except TypeError:
                    mesh = mesh.extract_surface().triangulate()
            except Exception:
                pass
            pts = np.asarray(mesh.points, dtype=float)
            faces = np.asarray(mesh.faces)
            if len(pts) and len(faces):
                tri = faces.reshape(-1, 4)
                if tri.shape[1] == 4:
                    fig.add_trace(
                        go.Mesh3d(
                            x=pts[:, 0],
                            y=pts[:, 1],
                            z=pts[:, 2],
                            i=tri[:, 1],
                            j=tri[:, 2],
                            k=tri[:, 3],
                            color="#9AA3AD",
                            opacity=0.18,
                            name="STL",
                            hoverinfo="skip",
                            flatshading=True,
                            showscale=False,
                        )
                    )
        except Exception:
            pass

    # Build / AABB outline wireframe
    z0, z1 = float(z_offset), float(z_offset) + float(height_mm)
    for xs, ys, zs in (
        ([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], [z0] * 5),
        ([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], [z1] * 5),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="#061F4A", width=3),
                name="bounds",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for x, y in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)):
        fig.add_trace(
            go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[z0, z1],
                mode="lines",
                line=dict(color="#061F4A", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if vectors_df is None or len(vectors_df) == 0:
        fig.update_layout(title="No scan vectors", height=560)
        return fig

    df = vectors_df.copy()
    if layer_filter is not None and "layer" in df.columns:
        df = df[df["layer"] == int(layer_filter)]

    layers = sorted(df["layer"].unique()) if "layer" in df.columns and len(df) else [0]
    palette = ["#0B3D91", "#1B7F4E", "#8E44AD", "#16A085", "#2980B9", "#27AE60", "#6C3483"]

    shown = 0
    all_start_x, all_start_y, all_start_z = [], [], []
    all_end_x, all_end_y, all_end_z = [], [], []
    all_start_text, all_end_text = [], []

    for li, layer in enumerate(layers):
        sub = df[df["layer"] == layer] if "layer" in df.columns else df
        if len(sub) == 0:
            continue
        step = max(1, int(np.ceil(len(sub) / max(1, max_arrows // max(len(layers), 1)))))
        sub = sub.iloc[::step]
        color = palette[li % len(palette)]
        xs, ys, zs = [], [], []
        # Direction chevrons near the end (last 15% of each vector)
        cx, cy, cz = [], [], []
        for _, row in sub.iterrows():
            x0, y0, z0v = float(row["x0"]), float(row["y0"]), float(row["z0"])
            x1, y1, z1v = float(row["x1"]), float(row["y1"]), float(row["z1"])
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
            zs.extend([z0v, z1v, None])
            # Chevron from 85% → 100%
            xa, ya = x0 + 0.85 * (x1 - x0), y0 + 0.85 * (y1 - y0)
            cx.extend([xa, x1, None])
            cy.extend([ya, y1, None])
            cz.extend([z0v, z1v, None])
            all_start_x.append(x0)
            all_start_y.append(y0)
            all_start_z.append(z0v)
            all_end_x.append(x1)
            all_end_y.append(y1)
            all_end_z.append(z1v)
            all_start_text.append(f"START L{int(layer)}<br>({x0:.2f}, {y0:.2f}, {z0v:.2f})")
            all_end_text.append(f"END L{int(layer)}<br>({x1:.2f}, {y1:.2f}, {z1v:.2f})")
            shown += 1
        ang = float(sub.iloc[0]["angle_deg"]) if "angle_deg" in sub.columns else 0.0
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=3),
                name=f"Layer {int(layer)} ({ang:.0f}°)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=cx,
                y=cy,
                z=cz,
                mode="lines",
                line=dict(color="#E67E22", width=7),
                name=f"Direction L{int(layer)}",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Global start / end markers (clear legend)
    if all_start_x:
        fig.add_trace(
            go.Scatter3d(
                x=all_start_x,
                y=all_start_y,
                z=all_start_z,
                mode="markers",
                marker=dict(size=5, color="#1B7F4E", symbol="circle", line=dict(width=1, color="#0E4D2E")),
                name="Scan START",
                text=all_start_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=all_end_x,
                y=all_end_y,
                z=all_end_z,
                mode="markers",
                marker=dict(size=6, color="#FC3D21", symbol="diamond", line=dict(width=1, color="#8B1A0F")),
                name="Scan END",
                text=all_end_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if edge_df is not None and len(edge_df):
        ed = edge_df
        if layer_filter is not None and "z_mm" in ed.columns and len(df):
            z_ref = float(df["z0"].median()) if len(df) else None
            if z_ref is not None:
                tol = max(0.2, float(np.nanstd(df["z0"])) + 0.2) if len(df) else 0.5
                ed = ed[np.abs(ed["z_mm"] - z_ref) < tol]
        fig.add_trace(
            go.Scatter3d(
                x=ed["x_mm"],
                y=ed["y_mm"],
                z=ed["z_mm"],
                mode="markers",
                marker=dict(size=7, color="#9B59B6", symbol="x", line=dict(width=2, color="#6C3483")),
                name=f"Edge pores ({len(ed)})",
                hovertemplate="edge pore<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{title} · {shown} vectors · green=START · red◆=END",
        height=640,
        margin=dict(l=0, r=0, t=52, b=0),
        legend=dict(x=0.02, y=0.98),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
    )
    return fig

