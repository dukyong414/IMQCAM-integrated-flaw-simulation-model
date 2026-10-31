"""
Combined plate-scale pore generation (data-driven).

Workflow
--------
1. CMU empirical pore number density ρ (pores/mm³) + size catalogs.
2. Geometry: coupon cubes / centered cube·cuboid / fitted STL.
3. Build volume V → base count ρ·V. NU spatial field scales each region by
   (NU_local / NU_mean) so the plate mean matches CMU while NU adds variation.
4. Randomly disperse pores inside each build (or STL volume).
5. Optional edge pores at laser scan-vector endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .scan_vectors import (
    STRATEGY_ANGLES,
    sample_edge_pores,
    vectors_for_builds,
    vectors_for_stl_mesh,
)
from .stl_fit import load_and_fit_stl, sample_points_in_mesh

ESD_COL = "Equivalent Spherical Diameter (um)"

GAS_FLOW_DIRS = {
    "+x": (1.0, 0.0),
    "-x": (-1.0, 0.0),
    "+y": (0.0, 1.0),
    "-y": (0.0, -1.0),
}

GAS_FLOW_LABELS = {
    "+x": "Gas distribution +X",
    "-x": "Gas distribution −X",
    "+y": "Gas distribution +Y",
    "-y": "Gas distribution −Y",
}


def gas_flow_arrow_geometry(
    plate_width_mm: float,
    plate_height_mm: float,
    gas_flow: str = "+x",
    *,
    z_mm: float = 0.0,
    length_frac: float = 0.55,
    head_frac: float = 0.12,
) -> dict:
    dx, dy = GAS_FLOW_DIRS.get(gas_flow, GAS_FLOW_DIRS["+x"])
    cx = float(plate_width_mm) / 2.0
    cy = float(plate_height_mm) / 2.0
    span = min(float(plate_width_mm), float(plate_height_mm))
    half_len = 0.5 * length_frac * span
    head = head_frac * span

    x0, y0 = cx - dx * half_len, cy - dy * half_len
    x1, y1 = cx + dx * half_len, cy + dy * half_len
    px, py = -dy, dx
    hx0 = x1 - dx * head + px * head * 0.45
    hy0 = y1 - dy * head + py * head * 0.45
    hx1 = x1 - dx * head - px * head * 0.45
    hy1 = y1 - dy * head - py * head * 0.45
    label = GAS_FLOW_LABELS.get(gas_flow, f"Gas flow {gas_flow}")
    lx = x1 - dx * head * 0.2 + px * (0.08 * span)
    ly = y1 - dy * head * 0.2 + py * (0.08 * span)
    lz = float(z_mm) + max(2.0, 0.05 * span)
    return {
        "gas_flow": gas_flow,
        "label": label,
        "direction": (dx, dy),
        "shaft": [(x0, y0, z_mm), (x1, y1, z_mm)],
        "head": [(hx0, hy0, z_mm), (x1, y1, z_mm), (hx1, hy1, z_mm)],
        "label_xyz": (lx, ly, lz),
        "z_mm": float(z_mm),
    }


@dataclass
class PlateBuild:
    """One solid on the plate (cube or cuboid)."""

    index: int
    row: int
    col: int
    cx_mm: float
    cy_mm: float
    size_mm: float  # cube edge; for cuboid = max(L,W) for pitch helpers
    height_mm: float
    shape: str = "cube"  # cube | cuboid
    length_mm: Optional[float] = None  # X extent (cuboid)
    width_mm: Optional[float] = None  # Y extent (cuboid)
    strip: int = 0  # which vertical 19×3 strip (0..n_strips-1) for grid_cubes

    def __post_init__(self):
        if self.length_mm is None:
            self.length_mm = self.size_mm
        if self.width_mm is None:
            self.width_mm = self.size_mm

    @property
    def volume_mm3(self) -> float:
        return float(self.length_mm * self.width_mm * self.height_mm)


CubeBuild = PlateBuild


@dataclass
class PlateLayout:
    plate_width_mm: float
    plate_height_mm: float
    n_rows: int
    n_cols: int
    cube_size_mm: float
    cube_height_mm: float
    gap_mm: float
    builds: List[PlateBuild]
    plate_mode: str = "grid_cubes"  # grid_cubes | single_cuboid | stl
    build_shape: str = "cube"
    stl_meta: Dict[str, Any] = field(default_factory=dict)
    stl_mesh: Any = None  # pyvista mesh when mode=stl
    n_strips: int = 1
    strip_gap_mm: float = 0.0

    @property
    def n_builds(self) -> int:
        return len(self.builds)

    @property
    def total_build_volume_mm3(self) -> float:
        if self.plate_mode == "stl" and self.stl_meta:
            v = self.stl_meta.get("volume_mm3")
            if v is not None and float(v) > 0:
                return float(v)
            # AABB fallback
            m = self.stl_meta
            return float(
                max(m.get("xmax", 0) - m.get("xmin", 0), 0)
                * max(m.get("ymax", 0) - m.get("ymin", 0), 0)
                * max(m.get("zmax", 0) - m.get("zmin", 0), 0)
            )
        return float(sum(b.volume_mm3 for b in self.builds))


def esd_um_to_volume_mm3(esd_um: np.ndarray) -> np.ndarray:
    r_mm = (np.asarray(esd_um, dtype=float) / 2.0) * 1e-3
    return (4.0 / 3.0) * np.pi * r_mm**3


def make_plate_layout(
    plate_width_mm: float = 250.0,
    plate_height_mm: float = 250.0,
    n_rows: int = 19,
    n_cols: int = 3,
    cube_size_mm: float = 10.0,
    cube_height_mm: float = 10.0,
    gap_mm: Optional[float] = None,
    plate_mode: str = "grid_cubes",
    cuboid_margin_mm: float = 2.0,
    cuboid_fill_plate: bool = False,
    cuboid_length_mm: Optional[float] = None,
    cuboid_width_mm: Optional[float] = None,
    stl_path: Optional[str] = None,
    stl_margin_mm: float = 2.0,
    stl_max_height_mm: Optional[float] = None,
    n_strips: int = 4,
    strip_gap_mm: Optional[float] = None,
    stl_sector_fraction: float = 1.0,
    stl_sector_start_deg: float = 0.0,
) -> PlateLayout:
    """
    plate_mode:
      grid_cubes     — coupon plate: n_strips vertical blocks of n_rows×n_cols cubes
                       (default 4 strips of 19×3 = 228 cubes, matching the LPBF coupon plate)
      single_cuboid  — one cuboid nearly filling the plate (height = cube_height)
      stl            — fitted STL (stl_path required)
    """
    mode = str(plate_mode or "grid_cubes").lower().strip()
    if mode in ("cube", "cubes", "grid"):
        mode = "grid_cubes"
    if mode in ("cuboid", "single"):
        mode = "single_cuboid"

    if mode == "stl":
        if not stl_path:
            raise ValueError("stl_path required for plate_mode='stl'")
        max_h = stl_max_height_mm
        if max_h is not None and float(max_h) <= 0:
            max_h = None
        mesh, meta = load_and_fit_stl(
            stl_path,
            plate_width_mm,
            plate_height_mm,
            margin_mm=stl_margin_mm,
            target_height_mm=None,
            clip_height_mm=float(max_h) if max_h is not None else None,
            sector_fraction=float(stl_sector_fraction or 1.0),
            sector_start_deg=float(stl_sector_start_deg or 0.0),
        )
        # proxy build for summaries / scan (AABB)
        length = meta["xmax"] - meta["xmin"]
        width = meta["ymax"] - meta["ymin"]
        height = meta["zmax"] - meta["zmin"]
        builds = [
            PlateBuild(
                index=0,
                row=0,
                col=0,
                cx_mm=0.5 * (meta["xmin"] + meta["xmax"]),
                cy_mm=0.5 * (meta["ymin"] + meta["ymax"]),
                size_mm=max(length, width),
                height_mm=height,
                shape="stl",
                length_mm=length,
                width_mm=width,
            )
        ]
        return PlateLayout(
            plate_width_mm=float(plate_width_mm),
            plate_height_mm=float(plate_height_mm),
            n_rows=1,
            n_cols=1,
            cube_size_mm=float(cube_size_mm),
            cube_height_mm=float(height),
            gap_mm=0.0,
            builds=builds,
            plate_mode="stl",
            build_shape="stl",
            stl_meta=meta,
            stl_mesh=mesh,
        )

    if mode == "single_cuboid":
        margin = float(max(0.0, cuboid_margin_mm))
        usable_w = max(1.0, plate_width_mm - 2 * margin)
        usable_h = max(1.0, plate_height_mm - 2 * margin)
        # fill | cube | custom L×W×H
        if cuboid_fill_plate:
            length = usable_w
            width = usable_h
            height = float(cube_height_mm)
            shape = "cuboid"
        elif cuboid_length_mm is not None and cuboid_width_mm is not None:
            length = max(1.0, min(float(cuboid_length_mm), usable_w))
            width = max(1.0, min(float(cuboid_width_mm), usable_h))
            height = max(1.0, float(cube_height_mm if cube_height_mm else cube_size_mm))
            shape = "cuboid" if abs(length - width) > 1e-9 or abs(length - height) > 1e-9 else "cube"
        else:
            # Cube edge: use cube_height (or cube_size), capped to usable plate
            edge = float(cube_height_mm if cube_height_mm else cube_size_mm)
            edge = max(1.0, min(edge, usable_w, usable_h))
            length = width = height = edge
            shape = "cube"
        builds = [
            PlateBuild(
                index=0,
                row=0,
                col=0,
                cx_mm=plate_width_mm / 2.0,
                cy_mm=plate_height_mm / 2.0,
                size_mm=max(length, width),
                height_mm=height,
                shape=shape,
                length_mm=length,
                width_mm=width,
            )
        ]
        return PlateLayout(
            plate_width_mm=float(plate_width_mm),
            plate_height_mm=float(plate_height_mm),
            n_rows=1,
            n_cols=1,
            cube_size_mm=float(length if shape == "cube" else cube_size_mm),
            cube_height_mm=float(height),
            gap_mm=margin,
            builds=builds,
            plate_mode="single_cuboid",
            build_shape=shape,
        )

    # --- coupon plate: n_strips vertical blocks of n_rows × n_cols cubes ---
    # Matches the LPBF build: four columns of 19×3 numbered cubes (228 total).
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be >= 1")
    n_strips = int(max(1, n_strips))
    # Tight packing inside each strip (cubes nearly touching in the photo)
    within_gap = 1.0 if gap_mm is None else float(max(0.0, gap_mm))

    strip_w = n_cols * cube_size_mm + max(0, n_cols - 1) * within_gap
    strip_h = n_rows * cube_size_mm + max(0, n_rows - 1) * within_gap
    if strip_h > plate_height_mm + 1e-9:
        raise ValueError(
            f"Strip height ({strip_h:.1f} mm) exceeds plate height ({plate_height_mm:.1f} mm). "
            "Increase plate size or reduce cube size."
        )

    if strip_gap_mm is None:
        if n_strips > 1:
            leftover = plate_width_mm - n_strips * strip_w
            strip_gap = max(within_gap, leftover / (n_strips + 1))
        else:
            strip_gap = 0.0
    else:
        strip_gap = float(max(0.0, strip_gap_mm))

    total_w = n_strips * strip_w + max(0, n_strips - 1) * strip_gap
    if total_w > plate_width_mm + 1e-9:
        if n_strips > 1:
            strip_gap = max(0.0, (plate_width_mm - n_strips * strip_w) / (n_strips - 1))
            total_w = n_strips * strip_w + (n_strips - 1) * strip_gap
        if total_w > plate_width_mm + 1e-9:
            raise ValueError(
                f"Strip layout width ({total_w:.1f} mm) exceeds plate width ({plate_width_mm:.1f} mm)."
            )

    x_origin = (plate_width_mm - total_w) / 2.0
    # Row 0 at the top of the plate (high Y), matching coupon numbering in the photo
    y_top = (plate_height_mm + strip_h) / 2.0

    builds = []
    idx = 0
    for s in range(n_strips):
        strip_x0 = x_origin + s * (strip_w + strip_gap)
        for r in range(n_rows):
            for c in range(n_cols):
                cx = strip_x0 + cube_size_mm / 2.0 + c * (cube_size_mm + within_gap)
                cy = y_top - cube_size_mm / 2.0 - r * (cube_size_mm + within_gap)
                builds.append(
                    PlateBuild(
                        index=idx,
                        row=r,
                        col=c,
                        cx_mm=float(cx),
                        cy_mm=float(cy),
                        size_mm=float(cube_size_mm),
                        height_mm=float(cube_height_mm),
                        shape="cube",
                        length_mm=float(cube_size_mm),
                        width_mm=float(cube_size_mm),
                        strip=s,
                    )
                )
                idx += 1

    return PlateLayout(
        plate_width_mm=float(plate_width_mm),
        plate_height_mm=float(plate_height_mm),
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        cube_size_mm=float(cube_size_mm),
        cube_height_mm=float(cube_height_mm),
        gap_mm=float(within_gap),
        builds=builds,
        plate_mode="grid_cubes",
        build_shape="cube",
        n_strips=int(n_strips),
        strip_gap_mm=float(strip_gap),
    )


def _sample_in_build(build: PlateBuild, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    half_x = float(build.length_mm) / 2.0
    half_y = float(build.width_mm) / 2.0
    x = rng.uniform(build.cx_mm - half_x, build.cx_mm + half_x, n)
    y = rng.uniform(build.cy_mm - half_y, build.cy_mm + half_y, n)
    z = rng.uniform(0.0, build.height_mm, n)
    return np.column_stack([x, y, z])


def _sample_in_build_with_flow_bias(
    build: PlateBuild,
    n: int,
    rng: np.random.Generator,
    *,
    gas_flow: str = "+x",
    gas_flow_sharpness: float = 2.0,
    flow_perturbation: float = 0.5,
) -> np.ndarray:
    """
    Sample points inside one build with in-build gas-flow population perturbation.

    For single-cuboid mode, build-level weights are degenerate (only one build), so
    we bias point density along flow direction inside the cuboid footprint.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    alpha = float(np.clip(flow_perturbation, 0.0, 1.0))
    if alpha <= 0:
        return _sample_in_build(build, n, rng)

    # Mixture of uniform + flow-biased proposals
    n_uniform = int(round((1.0 - alpha) * n))
    n_flow = max(0, n - n_uniform)
    pts = []
    if n_uniform > 0:
        pts.append(_sample_in_build(build, n_uniform, rng))

    if n_flow > 0:
        dx, dy = GAS_FLOW_DIRS.get(gas_flow, GAS_FLOW_DIRS["+x"])
        half_x = float(build.length_mm) / 2.0
        half_y = float(build.width_mm) / 2.0
        x_low, x_high = build.cx_mm - half_x, build.cx_mm + half_x
        y_low, y_high = build.cy_mm - half_y, build.cy_mm + half_y

        accepted = []
        # Rejection sampling on normalized downstream coordinate t in [0,1]
        # acceptance ~ exp(sharpness * t), normalized by exp(sharpness)
        s = float(max(0.0, gas_flow_sharpness))
        max_w = float(np.exp(s)) if s > 0 else 1.0
        need = n_flow
        attempts = 0
        max_attempts = max(5000, n_flow * 200)
        while len(accepted) < need and attempts < max_attempts:
            batch = max(64, (need - len(accepted)) * 6)
            x = rng.uniform(x_low, x_high, batch)
            y = rng.uniform(y_low, y_high, batch)
            z = rng.uniform(0.0, build.height_mm, batch)
            proj = x * dx + y * dy
            pmin = x_low * dx + y_low * dy
            pmax = x_high * dx + y_high * dy
            span = max(1e-12, pmax - pmin)
            t = np.clip((proj - pmin) / span, 0.0, 1.0)
            w = np.exp(s * t) / max_w
            keep = rng.random(batch) < w
            kept = np.column_stack([x[keep], y[keep], z[keep]])
            if len(kept):
                accepted.extend(list(kept))
            attempts += batch
        if len(accepted) < need:
            # fallback pad (still deterministic/reproducible by rng stream)
            pad = _sample_in_build(build, need - len(accepted), rng)
            if len(accepted):
                accepted = np.vstack([np.asarray(accepted)[:need], pad])
            else:
                accepted = pad
        else:
            accepted = np.asarray(accepted, dtype=float)[:need]
        pts.append(np.asarray(accepted, dtype=float))

    return np.vstack(pts) if pts else np.zeros((0, 3), dtype=float)


def _sample_in_cube(build: PlateBuild, n: int, rng: np.random.Generator) -> np.ndarray:
    return _sample_in_build(build, n, rng)


def nu_field_at_xy(model, x_mm: float, y_mm: float, plate_w: float, plate_h: float, mean_porosity: float) -> float:
    """NU predicted porosity (%) at one plate point; non-negative."""
    vals = nu_field_at_xy_batch(model, np.array([x_mm]), np.array([y_mm]), plate_w, plate_h, mean_porosity)
    return float(vals[0]) if len(vals) else 0.0


def _nu_denorm_array(model, raw: np.ndarray, mean_porosity: float, std_porosity: Optional[float] = None) -> np.ndarray:
    """Vectorized denormalization matching PorosityPatternModel._denorm."""
    raw = np.asarray(raw, dtype=float)
    s = float(std_porosity if std_porosity is not None else getattr(model, "_pooled_std", 1.0) or 1.0)
    mean = float(mean_porosity)
    scaling = str(getattr(model, "scaling", "relative") or "relative")
    if scaling == "z_score":
        return mean + raw * s
    if scaling == "relative":
        return mean * raw
    if scaling == "min_max":
        return mean + (raw - 0.5) * 2.0 * s
    return raw


def nu_field_at_xy_batch(
    model,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    plate_w: float,
    plate_h: float,
    mean_porosity: float,
) -> np.ndarray:
    """Batch NU porosity (%) — one RBF call for the whole array."""
    x = np.asarray(x_mm, dtype=float).ravel()
    y = np.asarray(y_mm, dtype=float).ravel()
    if len(x) == 0:
        return np.zeros(0, dtype=float)
    if model is None:
        return np.full(len(x), float(mean_porosity) * 100.0, dtype=float)
    try:
        pw = max(float(plate_w), 1e-12)
        ph = max(float(plate_h), 1e-12)
        x_n = x / pw
        y_n = y / ph
        if getattr(model, "homogenize_x", False):
            raw = np.asarray(model._model_1d(y_n[:, None]), dtype=float).ravel()
        else:
            raw = np.asarray(
                model._model_2d(np.column_stack((x_n, y_n))),
                dtype=float,
            ).ravel()
        pct = _nu_denorm_array(model, raw, mean_porosity)
        return np.maximum(pct, 0.0)
    except Exception:
        # Fallback: scalar predict (slow path)
        out = np.empty(len(x), dtype=float)
        for i in range(len(x)):
            try:
                out[i] = float(model.predict(float(x[i]), float(y[i]), plate_w, plate_h, mean_porosity, None))
            except Exception:
                out[i] = float(mean_porosity) * 100.0
        return np.maximum(out, 0.0)


def nu_values_for_builds(
    model,
    builds: List[PlateBuild],
    plate_w: float,
    plate_h: float,
    mean_porosity: float,
) -> np.ndarray:
    if not builds:
        return np.zeros(0, dtype=float)
    xs = np.array([b.cx_mm for b in builds], dtype=float)
    ys = np.array([b.cy_mm for b in builds], dtype=float)
    return nu_field_at_xy_batch(model, xs, ys, plate_w, plate_h, mean_porosity)


def nu_relative_factors(nu_vals: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Variation around the NU mean: factor_i = NU_i / mean(NU).
    Mean factor is 1 → CMU density×volume is preserved on average.
    """
    v = np.asarray(nu_vals, dtype=float)
    if len(v) == 0:
        return v, 1.0
    bar = float(np.mean(v))
    if bar <= 1e-15:
        return np.ones(len(v), dtype=float), 0.0
    return v / bar, bar


def allocate_integer_counts(expected: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn expected real counts into integers (floor + largest-remainder)."""
    exp = np.asarray(expected, dtype=float)
    exp = np.maximum(exp, 0.0)
    if len(exp) == 0:
        return np.asarray([], dtype=int)
    total = int(np.round(exp.sum()))
    floors = np.floor(exp).astype(int)
    rem = total - int(floors.sum())
    if rem > 0:
        frac = exp - floors
        order = np.argsort(-frac)
        for i in order[:rem]:
            floors[i] += 1
    elif rem < 0:
        # trim from largest floors
        order = np.argsort(-floors)
        left = -rem
        for i in order:
            if left <= 0:
                break
            take = min(floors[i], left)
            floors[i] -= take
            left -= take
    return floors


def cmu_total_density(cmu_density_weights: Optional[Dict[str, float]]) -> float:
    if not cmu_density_weights:
        return 0.0
    return float(sum(max(0.0, float(v)) for v in cmu_density_weights.values()))


def _sample_in_build_with_nu(
    build: PlateBuild,
    n: int,
    rng: np.random.Generator,
    nu_model,
    *,
    plate_w: float,
    plate_h: float,
    mean_porosity: float,
    nu_mean: float,
    gas_flow: Optional[str] = None,
    gas_flow_sharpness: float = 2.0,
    flow_perturbation: float = 0.0,
) -> np.ndarray:
    """
    Proposals inside build, accepted with probability ∝ NU/mean (capped).

    When ``flow_perturbation`` > 0 and ``gas_flow`` is set, proposals are
    gas-biased (downstream denser) instead of uniform — used for cuboid mode
    so gas direction reshapes the cloud without a separate spatter population.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    use_flow = bool(gas_flow) and float(flow_perturbation or 0.0) > 0.0
    if nu_model is None or nu_mean <= 1e-15:
        if use_flow:
            return _sample_in_build_with_flow_bias(
                build,
                n,
                rng,
                gas_flow=str(gas_flow),
                gas_flow_sharpness=float(gas_flow_sharpness),
                flow_perturbation=float(flow_perturbation),
            )
        return _sample_in_build(build, n, rng)

    out = np.empty((n, 3), dtype=float)
    filled = 0
    attempts = 0
    max_attempts = max(4000, n * 40)
    probe_xy = np.array(
        [
            [build.cx_mm, build.cy_mm],
            [build.cx_mm - build.length_mm / 2, build.cy_mm - build.width_mm / 2],
            [build.cx_mm + build.length_mm / 2, build.cy_mm - build.width_mm / 2],
            [build.cx_mm - build.length_mm / 2, build.cy_mm + build.width_mm / 2],
            [build.cx_mm + build.length_mm / 2, build.cy_mm + build.width_mm / 2],
        ],
        dtype=float,
    )
    pvals = nu_field_at_xy_batch(nu_model, probe_xy[:, 0], probe_xy[:, 1], plate_w, plate_h, mean_porosity) / nu_mean
    max_f = max(1.0, float(np.max(pvals)) if len(pvals) else 1.0)

    while filled < n and attempts < max_attempts:
        need = n - filled
        batch = min(max(64, need * 4), 20000)
        if use_flow:
            cand = _sample_in_build_with_flow_bias(
                build,
                batch,
                rng,
                gas_flow=str(gas_flow),
                gas_flow_sharpness=float(gas_flow_sharpness),
                flow_perturbation=float(flow_perturbation),
            )
        else:
            cand = _sample_in_build(build, batch, rng)
        factors = nu_field_at_xy_batch(
            nu_model, cand[:, 0], cand[:, 1], plate_w, plate_h, mean_porosity
        ) / nu_mean
        keep_p = np.clip(factors / max_f, 0.0, 1.0)
        kept = cand[rng.random(len(cand)) < keep_p]
        take = min(len(kept), need)
        if take:
            out[filled : filled + take] = kept[:take]
            filled += take
        attempts += batch
    if filled < n:
        if use_flow:
            out[filled:] = _sample_in_build_with_flow_bias(
                build,
                n - filled,
                rng,
                gas_flow=str(gas_flow),
                gas_flow_sharpness=float(gas_flow_sharpness),
                flow_perturbation=float(flow_perturbation),
            )
        else:
            out[filled:] = _sample_in_build(build, n - filled, rng)
    return out


def _sample_stl_with_nu(
    mesh,
    n: int,
    rng: np.random.Generator,
    nu_model,
    *,
    plate_w: float,
    plate_h: float,
    mean_porosity: float,
    nu_mean: float,
) -> np.ndarray:
    """
    Fast STL placement: enclosure-filter AABB proposals, then NU-weighted acceptance.
    Requests ~N interior points (not 6N), with vectorized NU.
    """
    from .stl_fit import _points_inside_mesh

    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    b = np.asarray(mesh.bounds, dtype=float)
    out = np.empty((n, 3), dtype=float)
    filled = 0
    attempts = 0
    max_attempts = max(6000, n * 50)

    # Probe NU max over AABB corners for acceptance scaling
    corners = np.array(
        [
            [b[0], b[2]],
            [b[1], b[2]],
            [b[0], b[3]],
            [b[1], b[3]],
            [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])],
        ],
        dtype=float,
    )
    pvals = nu_field_at_xy_batch(nu_model, corners[:, 0], corners[:, 1], plate_w, plate_h, mean_porosity)
    max_f = max(1.0, float(np.max(pvals / max(nu_mean, 1e-15))))

    while filled < n and attempts < max_attempts:
        need = n - filled
        batch = min(max(need * 5, 512), 25000)
        cand = np.column_stack(
            [
                rng.uniform(b[0], b[1], batch),
                rng.uniform(b[2], b[3], batch),
                rng.uniform(b[4], b[5], batch),
            ]
        )
        inside = _points_inside_mesh(mesh, cand)
        cand = cand[inside]
        if len(cand) == 0:
            attempts += batch
            continue
        factors = nu_field_at_xy_batch(
            nu_model, cand[:, 0], cand[:, 1], plate_w, plate_h, mean_porosity
        ) / max(nu_mean, 1e-15)
        keep_p = np.clip(factors / max_f, 0.0, 1.0)
        kept = cand[rng.random(len(cand)) < keep_p]
        take = min(len(kept), need)
        if take:
            out[filled : filled + take] = kept[:take]
            filled += take
        attempts += batch

    if filled < n:
        # Fallback: uniform interior sample for the remainder
        pad = sample_points_in_mesh(mesh, n - filled, rng, prepared=True)
        out[filled:] = pad
    return out


def _draw_size_catalog(
    cmu_size_catalog: pd.DataFrame,
    n: int,
    cmu_density_weights: Optional[Dict[str, float]],
    rng: np.random.Generator,
    *,
    type_catalogs: Optional[Dict[str, pd.DataFrame]] = None,
    keep_source_labels: bool = False,
) -> pd.DataFrame:
    """Sample N pore sizes from CMU catalog; optionally keep mechanism labels."""
    catalog = cmu_size_catalog
    if n <= 0:
        cols = list(catalog.columns) if len(catalog.columns) else [ESD_COL]
        if "pore_source" not in cols:
            cols = list(cols) + ["pore_source"]
        return pd.DataFrame(columns=cols)

    # Fast path for large N: index into catalog with numpy choice
    if "pore_source" in catalog.columns and cmu_density_weights:
        names = [k for k, v in cmu_density_weights.items() if float(v) > 0]
        if not names:
            names = list(cmu_density_weights.keys())
        w = np.array([max(0.0, float(cmu_density_weights[k])) for k in names], dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(names))
        w = w / w.sum()
        src_choice = rng.choice(len(names), size=n, p=w)
        parts = []
        for i, name in enumerate(names):
            n_need = int(np.sum(src_choice == i))
            if n_need <= 0:
                continue
            pool = None
            if type_catalogs and str(name).lower() in type_catalogs:
                pool = type_catalogs[str(name).lower()]
            if pool is None or len(pool) == 0:
                pool = catalog.loc[
                    catalog["pore_source"].astype(str).str.lower() == str(name).lower()
                ]
            if pool is None or len(pool) == 0:
                pool = catalog
            idx = rng.integers(0, len(pool), size=n_need)
            chunk = pool.iloc[idx].copy()
            if keep_source_labels:
                chunk["pore_source"] = str(name).lower()
            parts.append(chunk)
        out = (
            pd.concat(parts, ignore_index=True)
            if parts
            else catalog.iloc[rng.integers(0, len(catalog), size=n)].copy()
        )
    else:
        idx = rng.integers(0, len(catalog), size=n)
        out = catalog.iloc[idx].copy()
    out = out.copy()
    if not keep_source_labels:
        out["pore_source"] = "pore"
    elif "pore_source" not in out.columns:
        out["pore_source"] = "pore"
    return out.reset_index(drop=True)


def _scan_unit_from_strategy(
    scan_strategy: str = "horizontal",
    base_angle_deg: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Return (ux, uy, angle_deg) for primary scan direction."""
    key = str(scan_strategy or "horizontal").lower().strip()
    if base_angle_deg is not None:
        angle = float(base_angle_deg)
    else:
        angle = float(STRATEGY_ANGLES.get(key, 0.0))
    rad = np.deg2rad(angle)
    return float(np.cos(rad)), float(np.sin(rad)), float(angle)


def _attach_xyz(
    samples: pd.DataFrame,
    xyz: np.ndarray,
    build_indices: Optional[np.ndarray],
    builds: List[PlateBuild],
) -> pd.DataFrame:
    out = samples.copy()
    out["x_mm"] = xyz[:, 0]
    out["y_mm"] = xyz[:, 1]
    out["z_mm"] = xyz[:, 2]
    if build_indices is not None and len(builds):
        bi = np.asarray(build_indices, dtype=int)
        out["build_index"] = bi
        rows = np.array([builds[i].row for i in range(len(builds))], dtype=int)
        cols = np.array([builds[i].col for i in range(len(builds))], dtype=int)
        out["build_row"] = rows[bi]
        out["build_col"] = cols[bi]
    else:
        out["build_index"] = 0
        out["build_row"] = 0
        out["build_col"] = 0
    if ESD_COL in out.columns:
        esd = out[ESD_COL].to_numpy(dtype=float)
        out["radius_mm"] = (esd / 2.0) * 1e-3
        out["volume_mm3"] = esd_um_to_volume_mm3(esd)
    out["pore_family"] = out["pore_source"].astype(str)
    return out




def build_plate_pore_cloud(
    cmu_size_catalog: pd.DataFrame,
    cmu_density_weights: Dict[str, float],
    layout: PlateLayout,
    *,
    nu_model=None,
    mean_porosity: float = 0.02,
    seed: int = 42,
    max_pores: Optional[int] = None,
    # scan / edge pores
    enable_scan_vectors: bool = False,
    scan_strategy: str = "horizontal",
    hatch_spacing_mm: float = 0.14,
    layer_thickness_mm: float = 0.03,
    max_scan_layers: Optional[int] = 5,
    n_edge_pores: int = 0,
    batch_rows: int = 4,
    batch_cols: int = 4,
    base_angle_deg: Optional[float] = None,
    layer_rotation_deg: float = 67.0,
    prefer_gpu: bool = False,
    raptor_lof: Optional[Dict[str, Any]] = None,
    # STL: slice every layer of the part (per-layer stats, preview layers only kept)
    scan_full_part: bool = False,
    stripe_width_mm: float = 10.0,
    scan_velocity_mm_s: Optional[float] = None,
    progress_cb=None,
    # ignored legacy kwargs (kept so old callers don't crash)
    total_pores: Optional[int] = None,
    flow_perturbation: float = 0.0,
    gas_flow: str = "+x",
    gas_flow_sharpness: float = 2.0,
    use_nu_modulation: bool = True,
    spatter_fraction: Optional[float] = None,  # legacy; ignored (no separate spatter class)
) -> Tuple[pd.DataFrame, Dict]:
    """
    Data-driven pore cloud:
      n_i ≈ ρ · V_i · (NU_i / mean_NU), then disperse in each build.

    When ``raptor_lof`` is set (single_cuboid), LoF density/sizes come from Raptor;
    gas/keyhole stay CMU. Cuboid placement is gas-biased (no separate spatter class).
    """
    del spatter_fraction  # no longer an independent pore module
    rng = np.random.default_rng(seed)
    builds = layout.builds
    vectors_df = pd.DataFrame()
    edge_df = pd.DataFrame()
    place_backend = "NumPy CPU"
    flow_key = str(gas_flow or "+x").lower().strip()
    if flow_key not in GAS_FLOW_DIRS:
        flow_key = "+x"
    # Cuboid: reshape cloud along gas direction (replaces old V-spatter add-on)
    cuboid_flow_alpha = float(
        flow_perturbation if flow_perturbation and flow_perturbation > 0 else 0.65
    )

    density_weights = {str(k).lower(): float(v) for k, v in (cmu_density_weights or {}).items()}
    type_catalogs: Dict[str, pd.DataFrame] = {}
    raptor_meta: Dict[str, Any] = {}
    raptor_modes = ("single_cuboid", "stl")
    if raptor_lof and layout.plate_mode in raptor_modes:
        rho_lof = float(raptor_lof.get("rho") or 0.0)
        density_weights["lof"] = max(0.0, rho_lof)
        cat = raptor_lof.get("size_catalog")
        if cat is not None and len(cat):
            type_catalogs["lof"] = cat
        raptor_meta = dict(raptor_lof.get("summary") or {})
        raptor_meta["lof_density_per_mm3"] = density_weights["lof"]

    rho = max(0.0, cmu_total_density(density_weights))

    if nu_model is None:
        raise RuntimeError(
            "NU model is required for Combined placement. Pre-fit the NU cache first."
        )

    pw = float(layout.plate_width_mm)
    ph = float(layout.plate_height_mm)
    nu_vals = nu_values_for_builds(nu_model, builds, pw, ph, mean_porosity)
    rel, nu_bar = nu_relative_factors(nu_vals)

    volumes = np.asarray([float(b.volume_mm3) for b in builds], dtype=float)
    if layout.plate_mode == "stl":
        # Use solid mesh volume (not AABB) for ρ·V
        volumes = np.array([float(layout.total_build_volume_mm3)], dtype=float)
    expected = rho * volumes * rel
    counts = allocate_integer_counts(expected, rng)

    # Optional override only — 0 / None / negative means no cap (full ρ·V·NU)
    cap = max_pores if max_pores is not None else total_pores
    if cap is not None and int(cap) > 0 and int(counts.sum()) > int(cap):
        scale = float(cap) / float(counts.sum())
        counts = allocate_integer_counts(counts.astype(float) * scale, rng)

    n = int(counts.sum())
    keep_labels = bool(raptor_lof and layout.plate_mode in raptor_modes)
    sizes = _draw_size_catalog(
        cmu_size_catalog,
        n,
        density_weights,
        rng,
        type_catalogs=type_catalogs or None,
        keep_source_labels=keep_labels,
    )
    if keep_labels and len(sizes) and "pore_source" in sizes.columns:
        # Collapse bulk mechanisms to "pore" except keep lof tag for viz clarity
        src = sizes["pore_source"].astype(str).str.lower()
        sizes["cmu_mechanism"] = src.to_numpy()
        sizes.loc[~src.isin(["lof"]), "pore_source"] = "pore"
        sizes.loc[src == "lof", "pore_source"] = "lof"

    if n == 0 or len(builds) == 0:
        placed = sizes
        w = np.ones(len(builds), dtype=float) / max(len(builds), 1)
        nu_w = rel / rel.sum() if rel.sum() > 0 else w
    elif layout.plate_mode == "stl" and layout.stl_mesh is not None:
        xyz = _sample_stl_with_nu(
            layout.stl_mesh,
            n,
            rng,
            nu_model,
            plate_w=pw,
            plate_h=ph,
            mean_porosity=mean_porosity,
            nu_mean=nu_bar,
        )
        placed = _attach_xyz(sizes, xyz, None, builds)
        w = np.array([1.0])
        nu_w = w
    elif layout.plate_mode == "single_cuboid" and len(builds) == 1:
        xyz = _sample_in_build_with_nu(
            builds[0],
            n,
            rng,
            nu_model,
            plate_w=pw,
            plate_h=ph,
            mean_porosity=mean_porosity,
            nu_mean=nu_bar,
            gas_flow=flow_key,
            gas_flow_sharpness=float(gas_flow_sharpness),
            flow_perturbation=cuboid_flow_alpha,
        )
        placed = _attach_xyz(sizes, xyz, np.zeros(n, dtype=int), builds)
        w = np.array([1.0])
        nu_w = rel / max(rel.sum(), 1e-15)
        place_backend = f"NumPy CPU + gas bias ({flow_key})"
    else:
        # Coupon plate: allocate per cube, uniform random inside each cube
        from .combined_cuda import sample_points_in_builds

        xyz, build_idx, place_backend = sample_points_in_builds(
            builds, counts, seed=seed + 101, prefer_gpu=bool(prefer_gpu)
        )
        if len(xyz):
            placed = _attach_xyz(sizes, xyz, build_idx, builds)
        else:
            placed = sizes
        nu_w = rel / rel.sum() if rel.sum() > 0 else np.ones(len(builds)) / len(builds)
        w = (volumes * rel)
        w = w / w.sum() if w.sum() > 0 else nu_w

    _scan_ux, _scan_uy, scan_angle = _scan_unit_from_strategy(scan_strategy, base_angle_deg)

    scan_stats: Dict[str, Any] = {}
    scan_layers_df = pd.DataFrame()
    scan_mesh = None
    if enable_scan_vectors and scan_full_part:
        if layout.plate_mode == "stl" and layout.stl_mesh is not None:
            scan_mesh = layout.stl_mesh
        elif layout.plate_mode == "single_cuboid" and len(builds) == 1:
            from .stl_slicer_scan import box_mesh

            b0 = builds[0]
            scan_mesh = box_mesh(
                b0.cx_mm - 0.5 * b0.length_mm,
                b0.cx_mm + 0.5 * b0.length_mm,
                b0.cy_mm - 0.5 * b0.width_mm,
                b0.cy_mm + 0.5 * b0.width_mm,
                0.0,
                float(b0.height_mm),
            )
    if scan_mesh is not None:
        from .stl_slicer_scan import slice_scan_stl

        res = slice_scan_stl(
            scan_mesh,
            hatch_mm=hatch_spacing_mm,
            layer_mm=layer_thickness_mm,
            strategy=scan_strategy,
            stripe_width_mm=stripe_width_mm,
            layer_rotation_deg=layer_rotation_deg,
            base_angle_deg=base_angle_deg,
            batch_rows=batch_rows,
            batch_cols=batch_cols,
            velocity_mm_s=scan_velocity_mm_s,
            n_edge_pores=int(n_edge_pores),
            size_catalog=cmu_size_catalog if len(cmu_size_catalog) else sizes,
            seed=seed + 17,
            progress=progress_cb,
        )
        edge_df = res["edge_df"]
        vectors_df = res["vectors_df"]
        scan_stats = res["stats"]
        scan_layers_df = res["per_layer"]
    elif enable_scan_vectors and layout.plate_mode in ("grid_cubes", "single_cuboid"):
        vectors = vectors_for_builds(
            builds,
            strategy=scan_strategy,
            hatch_spacing_mm=hatch_spacing_mm,
            layer_thickness_mm=layer_thickness_mm,
            max_layers=max_scan_layers,
            batch_rows=batch_rows,
            batch_cols=batch_cols,
            base_angle_deg=base_angle_deg,
            layer_rotation_deg=layer_rotation_deg,
            stripe_width_mm=stripe_width_mm,
        )
        edge_df, vectors_df = sample_edge_pores(
            vectors,
            cmu_size_catalog if len(cmu_size_catalog) else sizes,
            n_edge_pores=int(n_edge_pores),
            seed=seed + 17,
        )
    elif enable_scan_vectors and layout.plate_mode == "stl" and layout.stl_mesh is not None:
        vectors = vectors_for_stl_mesh(
            layout.stl_mesh,
            strategy=scan_strategy,
            hatch_spacing_mm=hatch_spacing_mm,
            layer_thickness_mm=layer_thickness_mm,
            max_layers=max_scan_layers,
            batch_rows=batch_rows,
            batch_cols=batch_cols,
            base_angle_deg=base_angle_deg,
            layer_rotation_deg=layer_rotation_deg,
        )
        edge_df, vectors_df = sample_edge_pores(
            vectors,
            cmu_size_catalog if len(cmu_size_catalog) else sizes,
            n_edge_pores=int(n_edge_pores),
            seed=seed + 17,
        )

    frames = [placed]
    if len(edge_df):
        frames.append(edge_df)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else placed

    cube_rows = []
    for i, b in enumerate(builds):
        wi = float(w[i]) if i < len(w) else 0.0
        ni = float(nu_vals[i]) if i < len(nu_vals) else 0.0
        ri = float(rel[i]) if i < len(rel) else 1.0
        cube_rows.append(
            {
                "build_index": b.index,
                "row": b.row,
                "col": b.col,
                "strip": getattr(b, "strip", 0),
                "cx_mm": b.cx_mm,
                "cy_mm": b.cy_mm,
                "size_mm": b.size_mm,
                "length_mm": b.length_mm,
                "width_mm": b.width_mm,
                "height_mm": b.height_mm,
                "volume_mm3": float(b.volume_mm3),
                "shape": b.shape,
                "nu_pct": ni,
                "nu_relative": ri,
                "n_expected": float(expected[i]) if i < len(expected) else 0.0,
                "n_allocated": int(counts[i]) if i < len(counts) else 0,
                "placement_weight": wi,
                "flow_weight": wi,  # legacy alias for plots
                "spatter_weight": wi,
            }
        )
    cube_df = pd.DataFrame(cube_rows)

    # Cuboid only: gas distribution arrow (not a spatter wake)
    show_gas = layout.plate_mode == "single_cuboid"

    meta = {
        "counts_realized": combined["pore_source"].value_counts().to_dict() if len(combined) else {},
        "total_pores": int(len(combined)),
        "n_bulk_pores": int(n),
        "n_edge_pores": int(len(edge_df)) if len(edge_df) else 0,
        "n_spatter_pores": 0,
        "cmu_total_density": float(rho),
        "nu_mean_pct": float(nu_bar),
        "part_volume_mm3": float(volumes.sum()) if len(volumes) else 0.0,
        "base_count_rho_v": float(rho * volumes.sum()) if len(volumes) else 0.0,
        "scan_strategy": scan_strategy,
        "scan_angle_deg": float(scan_angle),
        "spatter_fraction": 0.0,
        "gas_flow": flow_key,
        "enable_scan_vectors": bool(enable_scan_vectors),
        "layer_rotation_deg": float(layer_rotation_deg),
        "base_angle_deg": None if base_angle_deg is None else float(base_angle_deg),
        "prefer_gpu": bool(prefer_gpu),
        "place_backend": place_backend,
        "raptor_lof": raptor_meta,
        "layout": {
            "plate_width_mm": layout.plate_width_mm,
            "plate_height_mm": layout.plate_height_mm,
            "n_rows": layout.n_rows,
            "n_cols": layout.n_cols,
            "n_strips": getattr(layout, "n_strips", 1),
            "strip_gap_mm": getattr(layout, "strip_gap_mm", 0.0),
            "cube_size_mm": layout.cube_size_mm,
            "cube_height_mm": layout.cube_height_mm,
            "gap_mm": layout.gap_mm,
            "n_builds": layout.n_builds,
            "build_shape": layout.build_shape,
            "plate_mode": layout.plate_mode,
            "stl_meta": layout.stl_meta or {},
            "gas_flow": flow_key if show_gas else None,
            "spatter_wake": None,
            "scan_angle_deg": float(scan_angle),
        },
        "cube_df": cube_df,
        "vectors_df": vectors_df,
        "scan_stats": scan_stats,
        "scan_layers_df": scan_layers_df,
        "scan_mesh": scan_mesh,
        "cmu_density_weights": {k: float(v) for k, v in density_weights.items()},
    }
    return combined, meta


def porosity_summary(combined: pd.DataFrame, layout: PlateLayout) -> Dict[str, float]:
    v_part = layout.total_build_volume_mm3
    out = {"part_volume_mm3": v_part, "n_builds": layout.n_builds, "plate_mode": layout.plate_mode}
    if len(combined) == 0 or "volume_mm3" not in combined.columns:
        out["total_phi"] = 0.0
        return out
    for src, grp in combined.groupby(combined["pore_source"].astype(str).str.lower()):
        v = float(grp["volume_mm3"].sum())
        out[f"n_{src}"] = int(len(grp))
        out[f"phi_{src}"] = v / v_part if v_part else 0.0
    out["total_phi"] = float(combined["volume_mm3"].sum()) / v_part if v_part else 0.0
    return out


@dataclass
class PartGeometry:
    shape: str = "cube"
    length_mm: float = 10.0
    width_mm: float = 10.0
    height_mm: float = 10.0
    diameter_mm: float = 10.0
    plate_x_mm: float = 60.0
    plate_y_mm: float = 60.0

    @property
    def volume_mm3(self) -> float:
        return float(self.length_mm * self.width_mm * self.height_mm)
