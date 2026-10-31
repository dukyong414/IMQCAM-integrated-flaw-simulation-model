"""
Interactive PyVista tool: place a fixed 1×1×1 mm Raptor cube on the STL bottom surface.

Usage:
  python -m integrated_gui.stl_raptor_roi --stl PATH --out ROI.json \\
      --plate-w 250 --plate-h 250 --margin 2 --clip-height 5

Controls:
  Drag the pink handle — cube stays exactly 1×1×1 mm (no resize).
  Click the part / bottom plane to jump the cube there.
  Arrow keys / W A D to nudge (S saves).
  Press S — save ROI and close
  Press Q / Esc — quit without saving
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


CUBE_EDGE_MM = 1.0


def _fit_mesh(
    stl_path: str,
    *,
    plate_w: float,
    plate_h: float,
    margin: float,
    clip_height: Optional[float],
):
    from integrated_gui.stl_fit import load_and_fit_stl

    return load_and_fit_stl(
        stl_path,
        plate_width_mm=plate_w,
        plate_height_mm=plate_h,
        margin_mm=margin,
        clip_height_mm=clip_height if clip_height and clip_height > 0 else None,
    )


def _bottom_z_mm(mesh_bounds, meta: Optional[dict] = None) -> float:
    """Part / plate contact plane — cube sits on this and grows upward."""
    zmin = float(mesh_bounds[4])
    if meta is not None and meta.get("zmin") is not None:
        try:
            zmin = float(meta["zmin"])
        except (TypeError, ValueError):
            pass
    return max(0.0, zmin)


def _cube_bounds_at(
    cx: float,
    cy: float,
    *,
    z_bottom: float,
    edge_mm: float = CUBE_EDGE_MM,
    xy_limits: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[float, ...]:
    """Fixed-size cube on the bottom surface, clamped to the part XY footprint."""
    half = 0.5 * float(edge_mm)
    if xy_limits is not None:
        xmin_l, xmax_l, ymin_l, ymax_l = xy_limits
        cx = min(max(cx, xmin_l + half), xmax_l - half)
        cy = min(max(cy, ymin_l + half), ymax_l - half)
    z0 = float(z_bottom)
    return (
        cx - half,
        cx + half,
        cy - half,
        cy + half,
        z0,
        z0 + float(edge_mm),
    )


def _default_box_bounds(mesh_bounds, *, edge_mm: float = CUBE_EDGE_MM, meta=None) -> Tuple[float, ...]:
    """1×1×1 mm cube at part center, sitting on the bottom surface."""
    xmin, xmax, ymin, ymax, _, _ = [float(v) for v in mesh_bounds]
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    z0 = _bottom_z_mm(mesh_bounds, meta)
    return _cube_bounds_at(
        cx,
        cy,
        z_bottom=z0,
        edge_mm=edge_mm,
        xy_limits=(xmin, xmax, ymin, ymax),
    )


def _snap_to_bottom_cube(
    bounds,
    *,
    z_bottom: float,
    edge_mm: float = CUBE_EDGE_MM,
    xy_limits: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[float, ...]:
    """Keep XY center from the widget, lock Z to bottom, force 1×1×1 size."""
    b = [float(v) for v in bounds]
    if len(b) != 6:
        return _cube_bounds_at(0.0, 0.0, z_bottom=z_bottom, edge_mm=edge_mm, xy_limits=xy_limits)
    cx = 0.5 * (b[0] + b[1])
    cy = 0.5 * (b[2] + b[3])
    return _cube_bounds_at(
        cx,
        cy,
        z_bottom=z_bottom,
        edge_mm=edge_mm,
        xy_limits=xy_limits,
    )


def run_picker(
    *,
    stl_path: str,
    out_json: str,
    plate_w: float = 250.0,
    plate_h: float = 250.0,
    margin: float = 2.0,
    clip_height: Optional[float] = 5.0,
    edge_mm: float = CUBE_EDGE_MM,
) -> int:
    os.environ["PYVISTA_OFF_SCREEN"] = "false"
    os.environ.pop("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", None)

    import pyvista as pv

    pv.OFF_SCREEN = False

    mesh, meta = _fit_mesh(
        stl_path,
        plate_w=plate_w,
        plate_h=plate_h,
        margin=margin,
        clip_height=clip_height,
    )
    try:
        surf = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
    except TypeError:
        surf = mesh.extract_surface().triangulate().clean()
    except Exception:
        surf = mesh

    mb = [float(v) for v in surf.bounds]
    z_bottom = _bottom_z_mm(mb, meta)
    xy_limits = (mb[0], mb[1], mb[2], mb[3])
    edge = float(edge_mm) if edge_mm and edge_mm > 0 else CUBE_EDGE_MM
    # Large drag handle (easy to grab) — independent of the fixed 1×1×1 cube
    span = max(mb[1] - mb[0], mb[3] - mb[2], 20.0)
    handle_r = float(max(3.0, min(12.0, 0.025 * span)))
    handle_z = z_bottom + 0.5 * edge
    nudge = max(0.25, 0.25 * edge)

    state: Dict[str, Any] = {
        "bounds": list(_default_box_bounds(mb, edge_mm=edge, meta=meta)),
        "saved": False,
        "suppress": False,
    }

    pl = pv.Plotter(
        window_size=(1280, 900),
        title="Raptor 1×1×1 mm — drag the pink handle (size fixed)",
    )
    pl.set_background("white")
    pl.add_mesh(
        surf,
        color="#9AA3AD",
        opacity=0.35,
        smooth_shading=True,
        show_edges=False,
        name="stl",
        pickable=True,
    )
    pw, ph = float(plate_w), float(plate_h)
    plate = pv.Plane(center=(pw / 2, ph / 2, 0), direction=(0, 0, 1), i_size=pw, j_size=ph)
    pl.add_mesh(plate, color="#E8EEF5", opacity=0.35, name="plate", pickable=False)
    guide = pv.Plane(
        center=(0.5 * (mb[0] + mb[1]), 0.5 * (mb[2] + mb[3]), z_bottom),
        direction=(0, 0, 1),
        i_size=max(1.0, mb[1] - mb[0]),
        j_size=max(1.0, mb[3] - mb[2]),
    )
    pl.add_mesh(guide, color="#90CAF9", opacity=0.22, name="bottom_guide", pickable=True)
    pl.add_axes()
    pl.show_bounds(grid="back", location="outer", color="black")

    def _center_xy(bounds: List[float]) -> Tuple[float, float]:
        return 0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3])

    def _refresh_cube(bounds: List[float]) -> None:
        try:
            pl.add_mesh(
                pv.Box(bounds=bounds),
                color="#29B6F6",
                opacity=0.55,
                name="raptor_cube",
                reset_camera=False,
                pickable=False,
            )
            pl.add_mesh(
                pv.Box(bounds=bounds),
                style="wireframe",
                color="#01579B",
                line_width=3,
                name="raptor_cube_wire",
                reset_camera=False,
                pickable=False,
            )
        except Exception:
            pass

    def _place_xy(cx: float, cy: float) -> None:
        snapped = list(
            _cube_bounds_at(
                float(cx),
                float(cy),
                z_bottom=z_bottom,
                edge_mm=edge,
                xy_limits=xy_limits,
            )
        )
        state["bounds"] = snapped
        _refresh_cube(snapped)

    _refresh_cube(state["bounds"])
    cx0, cy0 = _center_xy(state["bounds"])

    sphere_ref: Dict[str, Any] = {"w": None}

    def _sync_sphere(cx: float, cy: float) -> None:
        w = sphere_ref.get("w")
        if w is None or state["suppress"]:
            return
        state["suppress"] = True
        try:
            if hasattr(w, "SetCenter"):
                w.SetCenter(float(cx), float(cy), float(handle_z))
            elif hasattr(w, "place_widget"):
                w.place_widget((cx - handle_r, cx + handle_r, cy - handle_r, cy + handle_r, handle_z - handle_r, handle_z + handle_r))
        except Exception:
            pass
        finally:
            state["suppress"] = False

    def _on_sphere(point):
        if state["suppress"]:
            return
        try:
            if hasattr(point, "__len__") and len(point) >= 2:
                cx, cy = float(point[0]), float(point[1])
            else:
                return
        except Exception:
            return
        _place_xy(cx, cy)
        cx2, cy2 = _center_xy(state["bounds"])
        _sync_sphere(cx2, cy2)

    sphere = pl.add_sphere_widget(
        _on_sphere,
        center=(cx0, cy0, handle_z),
        radius=handle_r,
        color="deeppink",
        selected_color="magenta",
        style="surface",
        test_callback=False,
        interaction_event="always",
    )
    sphere_ref["w"] = sphere

    def _on_pick(point):
        try:
            if point is None or len(point) < 2:
                return
            _place_xy(float(point[0]), float(point[1]))
            cx2, cy2 = _center_xy(state["bounds"])
            _sync_sphere(cx2, cy2)
        except Exception:
            pass

    # Click the STL / bottom plane to jump the cube there
    try:
        pl.enable_surface_point_picking(
            callback=_on_pick,
            show_message=False,
            show_point=False,
            left_clicking=True,
            tolerance=0.02,
        )
    except TypeError:
        try:
            pl.enable_surface_point_picking(callback=_on_pick, show_message=False)
        except Exception:
            pass
    except Exception:
        pass

    def _nudge(dx: float, dy: float):
        cx, cy = _center_xy(state["bounds"])
        _place_xy(cx + dx, cy + dy)
        cx2, cy2 = _center_xy(state["bounds"])
        _sync_sphere(cx2, cy2)

    pl.add_key_event("Left", lambda: _nudge(-nudge, 0.0))
    pl.add_key_event("Right", lambda: _nudge(nudge, 0.0))
    pl.add_key_event("Up", lambda: _nudge(0.0, nudge))
    pl.add_key_event("Down", lambda: _nudge(0.0, -nudge))
    pl.add_key_event("a", lambda: _nudge(-nudge, 0.0))
    pl.add_key_event("d", lambda: _nudge(nudge, 0.0))
    pl.add_key_event("w", lambda: _nudge(0.0, nudge))

    pl.add_text(
        f"Fixed 1×1×1 mm Raptor cube on bottom (z={z_bottom:.2f} mm) — size cannot change.\n"
        "Drag the PINK handle  ·  click the part to jump  ·  arrows / W A D to nudge\n"
        "Press S to save   |   Q / Esc to cancel",
        font_size=11,
        color="black",
        position="upper_left",
    )

    def _save():
        b = list(
            _snap_to_bottom_cube(
                state["bounds"],
                z_bottom=z_bottom,
                edge_mm=edge,
                xy_limits=xy_limits,
            )
        )
        # Force exact edge (no float drift)
        cx = 0.5 * (b[0] + b[1])
        cy = 0.5 * (b[2] + b[3])
        b = list(
            _cube_bounds_at(cx, cy, z_bottom=z_bottom, edge_mm=edge, xy_limits=xy_limits)
        )
        xmin, xmax, ymin, ymax, zmin, zmax = b
        payload = {
            "ok": True,
            "stl_path": os.path.abspath(stl_path),
            "xmin_mm": xmin,
            "xmax_mm": xmax,
            "ymin_mm": ymin,
            "ymax_mm": ymax,
            "zmin_mm": zmin,
            "zmax_mm": zmax,
            "length_mm": float(edge),
            "width_mm": float(edge),
            "height_mm": float(edge),
            "cx_mm": 0.5 * (xmin + xmax),
            "cy_mm": 0.5 * (ymin + ymax),
            "cz_mm": 0.5 * (zmin + zmax),
            "z_bottom_mm": z_bottom,
            "edge_mm": float(edge),
            "fixed_volume_mm3": float(edge) ** 3,
            "slide_on_bottom": True,
            "plate_w_mm": plate_w,
            "plate_h_mm": plate_h,
            "margin_mm": margin,
            "clip_height_mm": clip_height,
            "stl_meta": {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in (meta or {}).items()
            },
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        state["saved"] = True
        print(f"ROI_SAVED {out_json}")
        print(
            f"ROI fixed {edge:.3f}^3 mm at ({payload['cx_mm']:.2f}, {payload['cy_mm']:.2f}) "
            f"on bottom z={z_bottom:.3f}"
        )
        pl.close()

    def _cancel():
        print("ROI_CANCELLED")
        pl.close()

    # Save on capital/lowercase S — don't bind lowercase s to nudge
    pl.add_key_event("S", _save)
    # Use KeyPress for 's' via vtk — pyvista add_key_event is case-sensitive on some builds
    pl.add_key_event("s", _save)
    pl.add_key_event("q", _cancel)
    pl.add_key_event("Escape", _cancel)

    pl.camera_position = "iso"
    pl.show()
    return 0 if state["saved"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pick 1×1×1 mm Raptor ROI on STL bottom surface")
    p.add_argument("--stl", required=True, help="Path to .stl")
    p.add_argument("--out", required=True, help="Output ROI JSON path")
    p.add_argument("--plate-w", type=float, default=250.0)
    p.add_argument("--plate-h", type=float, default=250.0)
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument(
        "--clip-height",
        type=float,
        default=0.0,
        help="Max height mm after fit (0 = full part height)",
    )
    p.add_argument(
        "--edge-mm",
        type=float,
        default=CUBE_EDGE_MM,
        help="Cube edge length in mm (default 1)",
    )
    args = p.parse_args(argv)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        from integrated_gui.console_guard import install_windows_ctrl_guard

        install_windows_ctrl_guard(detach_console=True)
    except Exception:
        pass

    clip = float(args.clip_height) if args.clip_height and args.clip_height > 0 else None
    try:
        return run_picker(
            stl_path=args.stl,
            out_json=args.out,
            plate_w=float(args.plate_w),
            plate_h=float(args.plate_h),
            margin=float(args.margin),
            clip_height=clip,
            edge_mm=float(args.edge_mm),
        )
    except Exception as exc:
        print(f"ROI_ERROR {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
