"""Filled trapezoid spatter wake that follows the laser scan path (Combined cuboid)."""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _unit(vx: float, vy: float) -> Tuple[float, float]:
    n = float(np.hypot(vx, vy))
    if n < 1e-12:
        return (1.0, 0.0)
    return (vx / n, vy / n)


def _trapezoid_vertices(
    tip_xy: np.ndarray,
    travel_dir: Tuple[float, float],
    *,
    wake_len_mm: float,
    tip_half_w_mm: float,
    base_half_w_mm: float,
) -> np.ndarray:
    """
    Filled-V / trapezoid behind the laser tip.

    Narrow end at the tip; wide base in the −travel (wake) direction.
    Returns (4, 2) vertices CCW.
    """
    ux, uy = _unit(*travel_dir)
    wx, wy = -ux, -uy  # wake
    px, py = -uy, ux  # perpendicular
    tip = np.asarray(tip_xy, dtype=float).reshape(2)
    base = tip + np.array([wx, wy]) * float(wake_len_mm)
    t_hw = float(tip_half_w_mm)
    b_hw = float(base_half_w_mm)
    p = np.array([px, py], dtype=float)
    return np.vstack(
        [
            tip - p * t_hw,
            tip + p * t_hw,
            base + p * b_hw,
            base - p * b_hw,
        ]
    )


def _sample_trapezoid(verts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform-ish samples inside a convex quad via two triangles."""
    if n <= 0:
        return np.zeros((0, 2))
    v0, v1, v2, v3 = verts
    # triangles (0,1,2) and (0,2,3)
    a1 = 0.5 * abs(np.cross(v1 - v0, v2 - v0))
    a2 = 0.5 * abs(np.cross(v2 - v0, v3 - v0))
    tot = a1 + a2 + 1e-18
    n1 = int(round(n * a1 / tot))
    n2 = max(0, n - n1)

    def _tri(a, b, c, k):
        if k <= 0:
            return np.zeros((0, 2))
        r1 = np.sqrt(rng.random(k))
        r2 = rng.random(k)
        return (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c

    return np.vstack([_tri(v0, v1, v2, n1), _tri(v0, v2, v3, n2)])


def _sample_plume(
    verts: np.ndarray,
    tip_xy: np.ndarray,
    travel_dir: Tuple[float, float],
    *,
    wake_len_mm: float,
    n: int,
    rng: np.random.Generator,
    tip_bias: float = 2.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dense particle samples inside the spatter plume.

    Returns (xy points, along-axis t) with t=0 at the laser tip and t=1 at the far end.
    Density is biased toward the tip (tip_bias > 1 → more dots near the laser).
    """
    if n <= 0:
        return np.zeros((0, 2)), np.zeros(0)
    over = max(n * 4, n + 50)
    pts = _sample_trapezoid(verts, over, rng)
    tip = np.asarray(tip_xy, dtype=float).reshape(2)
    ux, uy = _unit(*travel_dir)
    wx, wy = -ux, -uy
    along = np.array([wx, wy], dtype=float)
    proj = (pts - tip) @ along
    t = np.clip(proj / max(float(wake_len_mm), 1e-9), 0.0, 1.0)
    keep_p = np.power(np.clip(1.0 - 0.85 * t, 0.05, 1.0), float(tip_bias))
    keep = rng.random(len(pts)) < keep_p
    pts, t = pts[keep], t[keep]
    if len(pts) > n:
        idx = rng.choice(len(pts), size=n, replace=False)
        pts, t = pts[idx], t[idx]
    elif len(pts) < n:
        extra = _sample_trapezoid(verts, n - len(pts), rng)
        tip2 = tip.reshape(1, 2)
        t_extra = np.clip(
            ((extra - tip2) @ along) / max(float(wake_len_mm), 1e-9), 0.0, 1.0
        )
        pts = np.vstack([pts, extra]) if len(pts) else extra
        t = np.concatenate([t, t_extra]) if len(t) else t_extra
    return pts, t


def _load_path_segments_mm(
    path_csv: str,
    *,
    cuboid_cx: float,
    cuboid_cy: float,
    length_mm: float,
    width_mm: float,
) -> List[dict]:
    """Load path_vectors.csv (meters, local) → plate-frame mm segments."""
    if not path_csv or not os.path.isfile(path_csv):
        return []
    x_off = float(cuboid_cx) - 0.5 * float(length_mm)
    y_off = float(cuboid_cy) - 0.5 * float(width_mm)
    segs: List[dict] = []
    with open(path_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x0 = float(row["x0"]) * 1e3 + x_off
                y0 = float(row["y0"]) * 1e3 + y_off
                z0 = float(row["z0"]) * 1e3
                x1 = float(row["x1"]) * 1e3 + x_off
                y1 = float(row["y1"]) * 1e3 + y_off
                z1 = float(row["z1"]) * 1e3
                layer = int(float(row.get("layer", 0)))
            except (KeyError, TypeError, ValueError):
                continue
            if np.hypot(x1 - x0, y1 - y0) < 1e-6:
                continue
            segs.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "z": 0.5 * (z0 + z1),
                    "layer": layer,
                }
            )
    return segs


def build_spatter_animation(
    *,
    outdir: str,
    length_mm: float,
    width_mm: float,
    height_mm: float,
    plate_w_mm: float = 250.0,
    plate_h_mm: float = 250.0,
    cuboid_cx: Optional[float] = None,
    cuboid_cy: Optional[float] = None,
    strategy: str = "horizontal",
    gas_flow: str = "+x",
    n_layers: int = 8,
    n_frames: int = 36,
    seed: int = 42,
    path_csv: Optional[str] = None,
    wake_len_mm: Optional[float] = None,
    tip_half_w_mm: float = 0.15,
    base_half_w_mm: Optional[float] = None,
    segments: Optional[List[dict]] = None,
    outlines: Optional[Dict[int, np.ndarray]] = None,
    n_layers_part: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Write spatter_build.gif + spatter_accum.png/.npz.

    Spatter is a filled trapezoid (very narrow tip at the laser, wide base behind)
    that follows each hatch vector opposite the instantaneous scan direction.
    Default wake length is 300 mm (~30 cm) so the V extends well beyond the part.
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    L = float(length_mm)
    W = float(width_mm)
    H = float(height_mm)
    cx = float(cuboid_cx) if cuboid_cx is not None else float(plate_w_mm) * 0.5
    cy = float(cuboid_cy) if cuboid_cy is not None else float(plate_h_mm) * 0.5
    x0, x1 = cx - L / 2, cx + L / 2
    y0, y1 = cy - W / 2, cy + W / 2
    # ~30 cm flight distance by default (not tied to part size)
    wake_len = float(wake_len_mm) if wake_len_mm is not None else 300.0
    wake_len = max(10.0, wake_len)
    # Scale base width with wake length so a long V still reads as a plume
    if base_half_w_mm is None:
        base_half_w_mm = max(8.0, min(25.0, 0.04 * wake_len))
    else:
        base_half_w_mm = float(base_half_w_mm)
    # Keep tip distinctly sharper than the base (short top width)
    tip_half_w_mm = float(min(tip_half_w_mm, 0.25 * base_half_w_mm))

    if segments:
        # Pre-computed whole-part keyframes (already in plate mm, scan order)
        segs = list(segments)
        frame_segs = segs
        n_frames = len(frame_segs)
    else:
        segs = _load_path_segments_mm(
            path_csv or os.path.join(outdir, "path_vectors.csv"),
            cuboid_cx=cx,
            cuboid_cy=cy,
            length_mm=L,
            width_mm=W,
        )
        if not segs:
            return {"ok": False, "error": "No scan path segments for spatter"}
        # Subsample path into animation keyframes (progress along scan order)
        n_frames = max(12, int(n_frames))
        idx = np.linspace(0, len(segs) - 1, n_frames)
        frame_segs = [segs[int(round(i))] for i in idx]
    z_top = max((float(s.get("z", 0.0)) for s in frame_segs), default=H)

    # View tightly around the sample so hatch motion is readable.
    # Plume can still be long (e.g. 30 cm) but is clipped at the frame edges.
    view_pad = max(2.0, 0.15 * max(L, W, 1.0))
    vx0, vx1 = x0 - view_pad, x1 + view_pad
    vy0, vy1 = y0 - view_pad, y1 + view_pad
    # Accumulation / display grid matches the sample-fitted map
    gx0, gx1, gy0, gy1 = vx0, vx1, vy0, vy1
    nxg, nyg = 160, 160
    xs = np.linspace(gx0, gx1, nxg)
    ys = np.linspace(gy0, gy1, nyg)
    accum = np.zeros((nyg, nxg), dtype=float)

    # Dense tip-biased particles (gradient fill of the plume)
    # Small parts see only a short piece of the long plume; scale up so that piece stays dense
    view_span = max(vx1 - vx0, vy1 - vy0, 1.0)
    n_samples = int(min(30000, max(900, 500 + 1.5 * wake_len, 400.0 * wake_len / view_span)))
    frame_polys: List[np.ndarray] = []
    frame_samples: List[np.ndarray] = []
    frame_t: List[np.ndarray] = []
    for seg in frame_segs:
        tip = np.array([seg["x1"], seg["y1"]], dtype=float)
        travel = (seg["x1"] - seg["x0"], seg["y1"] - seg["y0"])
        poly = _trapezoid_vertices(
            tip,
            travel,
            wake_len_mm=wake_len,
            tip_half_w_mm=tip_half_w_mm,
            base_half_w_mm=base_half_w_mm,
        )
        frame_polys.append(poly)
        pts, tvals = _sample_plume(
            poly,
            tip,
            travel,
            wake_len_mm=wake_len,
            n=n_samples,
            rng=rng,
            tip_bias=2.4,
        )
        in_view = (
            (pts[:, 0] >= gx0)
            & (pts[:, 0] <= gx1)
            & (pts[:, 1] >= gy0)
            & (pts[:, 1] <= gy1)
        )
        if np.any(in_view):
            frame_samples.append(pts[in_view])
            frame_t.append(tvals[in_view])
        else:
            # Keep a tip-local subset so the map is never empty
            tip_n = max(80, min(len(pts), n_samples // 8))
            frame_samples.append(pts[:tip_n])
            frame_t.append(tvals[:tip_n])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.patches import Polygon, Rectangle
    except Exception as exc:
        return {"ok": False, "error": f"matplotlib unavailable: {exc}"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=110)
    ax0, ax1 = axes

    def _draw_cuboid(ax, layer: Optional[int] = None):
        if outlines and layer is not None and layer in outlines:
            ol = np.asarray(outlines[layer], dtype=float)
            if len(ol):
                xs_l = np.column_stack([ol[:, 0], ol[:, 2], np.full(len(ol), np.nan)]).ravel()
                ys_l = np.column_stack([ol[:, 1], ol[:, 3], np.full(len(ol), np.nan)]).ravel()
                ax.plot(xs_l, ys_l, color="#0B3D91", lw=1.6, zorder=3)
                return
        ax.add_patch(
            Rectangle(
                (x0, y0),
                L,
                W,
                fill=False,
                edgecolor="#0B3D91",
                linewidth=2.0,
                zorder=3,
            )
        )

    def _draw_path_ghost(ax, upto: int):
        # Completed hatch segments of the current layer
        cur_layer = frame_segs[min(upto, len(frame_segs) - 1)].get("layer")
        for seg in frame_segs[: max(1, upto)]:
            if segments and seg.get("layer") != cur_layer:
                continue
            ax.plot(
                [seg["x0"], seg["x1"]],
                [seg["y0"], seg["y1"]],
                color="#607D8B",
                lw=1.0,
                alpha=0.55,
                zorder=1,
            )

    def update(frame: int):
        ax0.clear()
        ax1.clear()
        fi = int(np.clip(frame, 0, n_frames - 1))
        seg = frame_segs[fi]
        poly = frame_polys[fi]
        pts = frame_samples[fi]
        tvals = frame_t[fi]

        # Running accumulation through this frame
        accum[:] = 0.0
        for si in range(fi + 1):
            pts_i = frame_samples[si]
            if len(pts_i) == 0:
                continue
            H2, _, _ = np.histogram2d(pts_i[:, 1], pts_i[:, 0], bins=[ys, xs])
            accum[: H2.shape[0], : H2.shape[1]] += H2

        if segments:
            n_lp = int(n_layers_part or 0) or "?"
            ax0.set_title(
                f"Spatter · layer {seg['layer'] + 1}/{n_lp} · z={seg.get('z', 0.0):.2f}/"
                f"{z_top:.1f} mm · vector {seg.get('vec_index', 0) + 1}/{seg.get('n_vec_layer', 1)}",
                fontsize=9,
            )
        else:
            ax0.set_title(
                f"Spatter · layer {seg['layer']} · vector {fi + 1}/{n_frames}"
            )
        ax0.set_xlabel("X (mm)")
        ax0.set_ylabel("Y (mm)")
        ax0.set_aspect("equal")
        _draw_path_ghost(ax0, fi)
        _draw_cuboid(ax0, seg.get("layer"))
        # Active hatch
        ax0.plot(
            [seg["x0"], seg["x1"]],
            [seg["y0"], seg["y1"]],
            color="#0B3D91",
            lw=2.4,
            zorder=4,
            label="Scan path",
        )
        ax0.scatter([seg["x1"]], [seg["y1"]], c="#D32F2F", s=48, zorder=6, label="Laser tip")
        # Light outline only — fill is the particle gradient
        ax0.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor="none",
                edgecolor="#BF360C",
                alpha=0.35,
                lw=0.8,
                zorder=2,
                clip_on=True,
            )
        )
        if len(pts):
            # Gradient: dense / hot near tip (t~0), cooler farther out (t~1)
            ax0.scatter(
                pts[:, 0],
                pts[:, 1],
                c=1.0 - tvals,
                cmap="YlOrRd",
                s=7,
                alpha=0.55,
                linewidths=0,
                vmin=0.0,
                vmax=1.0,
                zorder=3,
                label="Spatter",
            )
        ax0.set_xlim(vx0, vx1)
        ax0.set_ylim(vy0, vy1)
        ax0.legend(loc="upper right", fontsize=8)

        ax1.set_title("Accumulated spatter (near sample)")
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_aspect("equal")
        ax1.imshow(
            accum,
            origin="lower",
            extent=[gx0, gx1, gy0, gy1],
            cmap="YlOrRd",
            aspect="equal",
            interpolation="bilinear",
        )
        _draw_cuboid(ax1, seg.get("layer"))
        ax1.set_xlim(vx0, vx1)
        ax1.set_ylim(vy0, vy1)
        fig.tight_layout()
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=100, blit=False
    )
    gif_path = os.path.join(outdir, "spatter_build.gif")
    png_path = os.path.join(outdir, "spatter_accum.png")
    npz_path = os.path.join(outdir, "spatter_accum.npz")
    try:
        anim.save(gif_path, writer="pillow", fps=8)
    except Exception as exc:
        plt.close(fig)
        return {"ok": False, "error": f"GIF save failed: {exc}"}

    update(n_frames - 1)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    accum[:] = 0.0
    for pts in frame_samples:
        H2, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[ys, xs])
        accum[: H2.shape[0], : H2.shape[1]] += H2
    np.savez_compressed(
        npz_path,
        accum=accum,
        xs=xs,
        ys=ys,
        cuboid=np.array([x0, x1, y0, y1, H]),
        strategy=str(strategy),
        gas_flow=str(gas_flow),
        n_path_segments=len(segs),
    )
    return {
        "ok": True,
        "gif": gif_path,
        "png": png_path,
        "npz": npz_path,
        "n_layers": int(n_layers),
        "n_frames": n_frames,
        "n_path_segments": len(segs),
        "wake_len_mm": float(wake_len),
        "base_half_w_mm": float(base_half_w_mm),
    }
