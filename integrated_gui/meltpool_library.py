"""Velocity / scan-angle melt-pool library for Raptor (Vamsi's ULI dataset).

Folder layout (``Files from Vamsi``):

* ``ULI_v{V}_theta{θ}_widths.txt`` — width time series (time, width [m]) per
  scan velocity V (mm/s) and scan angle θ (deg).
* ``flucWs.txt`` / ``flucWsMTs.txt`` — per V: mean width [m], ``eta_D`` (depth
  / width) and ``eta_C`` (height / width). These are Raptor's depth and height
  scales relative to the width series.
* ``depthShapes.txt`` — (V, depth shape exponent) samples; averaged per V.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LIBRARY_DIR = os.path.abspath(os.path.join(ROOT, "..", "Files from Vamsi"))
ETA_SOURCES = ("flucWs", "flucWsMTs")
# Subraveti et al.: isotropic scale on the mean width, depth, and cap.
# 1.04 is the bound that sits on the measured mean pool (Fig. 5).
LAMBDA_MEAN = 1.04

_WIDTH_RE = re.compile(r"^ULI_v(\d+(?:\.\d+)?)_theta(\d+(?:\.\d+)?)_widths\.txt$", re.IGNORECASE)


@dataclass
class VelocityCondition:
    velocity_mm_s: float
    mean_width_m: Optional[float]
    eta_d: Optional[float]
    eta_c: Optional[float]
    depth_shape: Optional[float]
    depth_shape_n: int
    width_files: Dict[float, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "velocity_mm_s": self.velocity_mm_s,
            "mean_width_m": self.mean_width_m,
            "eta_d": self.eta_d,
            "eta_c": self.eta_c,
            "depth_shape": self.depth_shape,
            "depth_shape_n": self.depth_shape_n,
            "width_files": {str(k): v for k, v in sorted(self.width_files.items())},
        }


def _vkey(v: float) -> float:
    return round(float(v), 3)


def scan_width_files(folder: str) -> Dict[float, Dict[float, str]]:
    """Return {V: {theta: path}} for every ULI_v*_theta*_widths.txt in folder."""
    out: Dict[float, Dict[float, str]] = {}
    if not folder or not os.path.isdir(folder):
        return out
    for name in os.listdir(folder):
        m = _WIDTH_RE.match(name)
        if not m:
            continue
        v, th = _vkey(m.group(1)), float(m.group(2)) % 360.0
        out.setdefault(v, {})[th] = os.path.join(folder, name)
    return out


def load_eta_table(folder: str, source: str = "flucWs") -> Dict[float, tuple]:
    """Return {V: (mean_width_m, eta_D, eta_C)} from flucWs.txt or flucWsMTs.txt."""
    if source not in ETA_SOURCES:
        raise ValueError(f"eta source must be one of {ETA_SOURCES}")
    path = os.path.join(folder, f"{source}.txt")
    if not os.path.isfile(path):
        return {}
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", comments="#"))
    return {_vkey(r[0]): (float(r[1]), float(r[2]), float(r[3])) for r in data}


def load_depth_shapes(folder: str) -> Dict[float, tuple]:
    """Return {V: (mean depth shape exponent, n samples)} from depthShapes.txt."""
    path = os.path.join(folder, "depthShapes.txt")
    if not os.path.isfile(path):
        return {}
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", comments="#"))
    out = {}
    for v in np.unique(data[:, 0]):
        vals = data[data[:, 0] == v, 1]
        out[_vkey(v)] = (float(np.mean(vals)), int(vals.size))
    return out


def load_library(folder: str = DEFAULT_LIBRARY_DIR, eta_source: str = "flucWs") -> Dict[float, VelocityCondition]:
    widths = scan_width_files(folder)
    etas = load_eta_table(folder, eta_source)
    shapes = load_depth_shapes(folder)
    out: Dict[float, VelocityCondition] = {}
    for v in sorted(set(widths) | set(etas) | set(shapes)):
        mw, ed, ec = etas.get(v, (None, None, None))
        ds, dn = shapes.get(v, (None, 0))
        out[v] = VelocityCondition(v, mw, ed, ec, ds, dn, dict(widths.get(v, {})))
    return out


def available_velocities(folder: str = DEFAULT_LIBRARY_DIR) -> List[float]:
    return sorted(scan_width_files(folder))


def available_thetas(folder: str = DEFAULT_LIBRARY_DIR) -> List[float]:
    thetas = set()
    for per_v in scan_width_files(folder).values():
        thetas.update(per_v)
    return sorted(thetas)


def _raptor_cosine_amplitudes(data: np.ndarray) -> np.ndarray:
    """Phase-free cosine amplitudes, matching ``raptor.api.compute_spectral_components``.

    Index 0 is the mean width. Later entries are ``weight * |rFFT| / n``.
    """
    signal = np.asarray(data[:, 1], dtype=np.float64)
    n = int(signal.size)
    fft_values = np.fft.rfft(signal)
    weights = np.full(fft_values.size, 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    amplitudes = weights * np.abs(fft_values) / float(n)
    amplitudes[0] = float(signal.mean())
    return amplitudes


def _series_from_cosine_amplitudes(time_values: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    """Zero-phase series that inverts ``_raptor_cosine_amplitudes``."""
    time_values = np.asarray(time_values, dtype=np.float64)
    n = int(time_values.size)
    weights = np.full(amplitudes.size, 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    fft_values = np.zeros(amplitudes.shape, dtype=np.complex128)
    fft_values.real = np.asarray(amplitudes, dtype=np.float64) * n / weights
    signal = np.fft.irfft(fft_values, n=n)
    signal = signal - float(signal.mean()) + float(amplitudes[0])
    return np.column_stack([time_values, signal])


def average_orientation_series(width_files: Dict[float, str], velocity_mm_s: float):
    """Width series whose cosine spectrum is the mean across orientations.

    Subraveti et al. use one melt pool at each velocity:
    ``S_bar(xi) = mean over theta of S(xi, theta)``, then a cosine expansion.
    Amplitudes are averaged, not complex FFTs, so uncorrelated phases do not cancel.
    The mean (bin 0) is the average of the track means. Returns ``(series, meta)``.
    """
    if not width_files:
        raise ValueError("No width files to average.")
    loaded = []
    for theta in sorted(width_files):
        data, factor = read_width_series(width_files[theta], velocity_mm_s)
        loaded.append((float(theta), np.asarray(data, dtype=np.float64), float(factor)))
    ref_time = loaded[0][1][:, 0]
    n = int(ref_time.size)
    dt = float(ref_time[1] - ref_time[0]) if n > 1 else 0.0
    amplitudes = []
    for _theta, data, _factor in loaded:
        aligned = data
        same_grid = data.shape[0] == n and (
            n < 2 or abs(float(data[1, 0] - data[0, 0]) - dt) <= 1.0e-6 * max(abs(dt), 1.0e-18)
        )
        if not same_grid:
            aligned = np.column_stack(
                [ref_time, np.interp(ref_time, data[:, 0], data[:, 1])]
            )
        amplitudes.append(_raptor_cosine_amplitudes(aligned))
    averaged = np.mean(np.vstack(amplitudes), axis=0)
    series = _series_from_cosine_amplitudes(ref_time, averaged)
    meta = {
        "n_theta": len(loaded),
        "thetas": [theta for theta, _data, _factor in loaded],
        "mean_width_m": float(averaged[0]),
        "time_factor": loaded[0][2],
    }
    return series, meta


def read_width_series(path: str, velocity_mm_s: float, min_track_m: float = 1.0e-4):
    """Load (time [s], width [m]); fix a time axis that is 1000x too short.

    Vamsi's files carry the same widths as Raptor's bundled ULI_v1700 series
    but with time ending at ~5e-6 instead of ~5e-3 s, i.e. an implied track of
    a few µm. Raptor's float32 spectral error bound scales with frequency, so
    the time axis must be in seconds. Returns (data, time_factor_applied).
    """
    from pathlib import Path

    from raptor.io import read_data

    data = np.array(read_data(Path(path)), dtype=np.float64)
    track_m = float(data[-1, 0] - data[0, 0]) * float(velocity_mm_s) / 1000.0
    factor = 1.0
    while track_m * factor < min_track_m and factor < 1.0e9:
        factor *= 1000.0
    if factor != 1.0:
        data[:, 0] *= factor
    return data, factor


def nearest_theta(angle_deg: float, thetas) -> float:
    """Closest available θ on the circle (e.g. 67° → 45°, 340° → 0°)."""
    thetas = np.asarray(sorted(thetas), dtype=float)
    if thetas.size == 0:
        raise ValueError("No scan angles available.")
    a = float(angle_deg) % 360.0
    diff = np.abs((thetas - a + 180.0) % 360.0 - 180.0)
    return float(thetas[int(np.argmin(diff))])


def vector_angle_deg(vector) -> float:
    """In-plane scan direction of a Raptor PathVector, degrees in [0, 360)."""
    d = np.asarray(vector.end_point, dtype=float) - np.asarray(vector.start_point, dtype=float)
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


def resolve_condition(lib_cfg: dict, velocity_mm_s: float, *, exact: bool = False) -> VelocityCondition:
    """Library condition for a velocity (nearest available V unless ``exact``)."""
    folder = (lib_cfg or {}).get("dir") or DEFAULT_LIBRARY_DIR
    eta_source = (lib_cfg or {}).get("eta_source", "flucWs")
    library = {v: c for v, c in load_library(folder, eta_source).items() if c.width_files}
    if not library:
        raise FileNotFoundError(f"No ULI_v*_theta*_widths.txt files in {folder}")
    v = _vkey(velocity_mm_s)
    if v not in library:
        if exact:
            raise FileNotFoundError(f"No ULI_v{velocity_mm_s:g}_theta*_widths.txt in {folder}")
        v = min(library, key=lambda k: abs(k - float(velocity_mm_s)))
    cond = library[v]
    if (lib_cfg or {}).get("use_eta", True) and (cond.eta_d is None or cond.eta_c is None):
        raise ValueError(f"V={v:g} mm/s missing from {eta_source}.txt")
    if (lib_cfg or {}).get("use_depth_shape", True) and cond.depth_shape is None:
        raise ValueError(f"V={v:g} mm/s missing from depthShapes.txt")
    return cond


def compute_porosity_grouped(
    grid,
    path_vectors,
    *,
    width_files: Dict[float, str],
    velocity_mm_s: float,
    n_modes: int,
    width_scale: float,
    depth_scale: float,
    height_scale: float,
    depth_shape: float,
    height_shape: float,
    enable_random_phases: bool,
    seed: Optional[int],
    theta_mode: str = "per_layer",
    theta_deg: float = 0.0,
    angle_offset_deg: float = 0.0,
    lambda_mean: float = 1.0,
    log=None,
    on_group=None,
    width_cache: Optional[dict] = None,
    **porosity_kwargs,
):
    """Raptor porosity with one melt pool per scan-angle group of vectors.

    ``theta_mode="averaged"`` builds one melt pool from the mean cosine spectrum
    of every orientation file and uses it for all vectors (Subraveti et al.).
    ``per_layer`` assigns each vector the θ file nearest its scan direction.
    ``fixed`` uses one θ. Phase fields are merged. Returns
    (phase_field, {θ: n_vectors}, {θ: width_file}).
    """
    from raptor.api import compute_porosity, create_melt_pool

    log = log or (lambda _m: None)
    width_cache = {} if width_cache is None else width_cache
    thetas = sorted(width_files)
    groups: Dict[float, list] = {}
    averaged_series = None
    if theta_mode == "averaged" and len(thetas) > 1:
        averaged_series, avg_meta = average_orientation_series(width_files, velocity_mm_s)
        groups[0.0] = list(path_vectors)
        log(
            f"Averaged spectrum over {avg_meta['n_theta']} orientations "
            f"{avg_meta['thetas']}: mean width={avg_meta['mean_width_m'] * 1e6:.2f} µm, "
            f"{len(path_vectors)} vectors, one melt pool"
        )
    elif theta_mode == "per_layer" and len(thetas) > 1:
        for vec in path_vectors:
            th = nearest_theta(vector_angle_deg(vec) + angle_offset_deg, thetas)
            groups.setdefault(th, []).append(vec)
    else:
        groups[nearest_theta(theta_deg, thetas)] = list(path_vectors)
    if averaged_series is None:
        log("θ groups: " + ", ".join(f"{th:g}°→{len(v)} vec" for th, v in sorted(groups.items())))

    phase = None
    for gi, (th, vecs) in enumerate(sorted(groups.items())):
        if averaged_series is not None:
            wdata = averaged_series
        else:
            wpath = width_files[th]
            if wpath not in width_cache:
                wdata, tfac = read_width_series(wpath, velocity_mm_s)
                if tfac != 1.0:
                    log(
                        f"{os.path.basename(wpath)}: time axis ×{tfac:g} (implied track length was "
                        f"< 0.1 mm; now {(wdata[-1, 0] - wdata[0, 0]) * velocity_mm_s:.2f} mm)"
                    )
                width_cache[wpath] = wdata
            wdata = width_cache[wpath]
        if on_group is not None:
            on_group(gi, len(groups), th)
        series = np.array(wdata, dtype=float, copy=True)
        width_used = float(width_scale)
        depth_used = float(depth_scale)
        height_used = float(height_scale)
        if float(lambda_mean) != 1.0:
            # Same as the 0.5 mm paper cube: shift the mean only.
            # Depth and cap scales stay η, so fluctuations are not multiplied by λ.
            mu = float(series[:, 1].mean())
            series[:, 1] += (float(lambda_mean) - 1.0) * mu
            if gi == 0:
                log(
                    f"Lambda={float(lambda_mean):g} shifts the mean width, depth, and cap. "
                    f"Mean width {mu * 1e6:.2f} → {mu * float(lambda_mean) * 1e6:.2f} µm. "
                    f"Fluctuations are unchanged."
                )
        melt_pool = create_melt_pool(
            {
                "width": (series, n_modes, width_used, 2.0),
                "depth": (series, n_modes, depth_used, float(depth_shape)),
                "height": (series, n_modes, height_used, float(height_shape)),
            },
            enable_random_phases=enable_random_phases,
        )
        field_ = compute_porosity(
            grid, vecs, melt_pool, random_seed=None if seed is None else seed + gi, **porosity_kwargs
        )
        phase = merge_phase_fields(phase, field_) if len(groups) > 1 else field_
        del field_
    if averaged_series is not None:
        group_map = {"averaged": len(path_vectors)}
        file_map = {f"{th:g}": width_files[th] for th in thetas}
    else:
        group_map = {f"{th:g}": len(v) for th, v in sorted(groups.items())}
        file_map = {f"{th:g}": width_files[th] for th in sorted(groups)}
    return phase, group_map, file_map


def merge_phase_fields(acc: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
    """Union of two Raptor phase fields computed with separate melt pools.

    Pore (0) survives only where neither field melted, so porosity is exact.
    Where both melted: boundary on boundary -> intersection (3); otherwise
    interior wins (approximates Raptor's in-order overwrite).
    """
    if acc is None:
        return np.array(new, dtype=np.int8, copy=True)
    both = (acc > 0) & (new > 0)
    out = np.where(acc == 0, new, acc).astype(np.int8)
    both_bnd = both & (acc >= 2) & (new >= 2)
    out[both & ~both_bnd] = np.minimum(acc, new)[both & ~both_bnd]
    out[both_bnd] = 3
    return out
