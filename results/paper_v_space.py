"""Subraveti et al. velocity-space SVE: 0.5 mm cube, 5 um, mean-only lambda.

ScanPathBuilder uses his powder consolidation of 0.5, so a 30 um nominal
layer is spaced at 60 um.

One ensemble per (velocity, lambda). Tracks extend l/2 past each face.
The width spectrum is the average of the eight single-track orientations.
Lambda shifts the mean width only; depth and cap use the multitrack
aspect ratios on that shifted series. Defects smaller than 20 um ESD
are counted separately, matching the paper's validation filter.
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import time

import numpy as np

from integrated_gui.combined_raptor import ensure_raptor_on_path

ensure_raptor_on_path()
from raptor.api import compute_morphology, compute_porosity, create_grid, create_melt_pool
from raptor.utilities import ScanPathBuilder
from integrated_gui import meltpool_library as mpl

VELOCITIES = list(range(1300, 1701, 50))
LAMBDAS = (1.04, 1.14)
# 125 mm3 below the calibration speed, 31 mm3 above it (defects are common).
N_SVE = {v: (1000 if v <= 1550 else 250) for v in VELOCITIES}
EDGE_M = 0.5e-3
VOXEL_M = 5.0e-6
ESD_MIN_M = 20e-6
OUT_CSV = os.path.join(os.path.dirname(__file__), "paper_v_space.csv")


def _scale_mean(series: np.ndarray, lam: float) -> np.ndarray:
    out = np.array(series, dtype=float, copy=True)
    mu = float(out[:, 1].mean())
    out[:, 1] += (float(lam) - 1.0) * mu
    return out


def _quiet(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def run() -> None:
    box = np.array([[0.0, 0.0, 0.0], [EDGE_M, EDGE_M, EDGE_M]], dtype=float)
    grid = _quiet(create_grid, VOXEL_M, bound_box=box)
    sve_m3 = EDGE_M ** 3
    rows = []
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "velocity_mm_s",
                "lambda",
                "n_sve",
                "volume_mm3",
                "n_ge20",
                "vf_percent_ge20",
                "per_mm3_ge20",
                "max_esd_um",
                "n_all",
                "seconds",
            ],
        )
        writer.writeheader()
        for velocity in VELOCITIES:
            cond = mpl.resolve_condition(
                {"dir": mpl.DEFAULT_LIBRARY_DIR, "eta_source": "flucWsMTs"},
                float(velocity),
                exact=True,
            )
            base, _meta = mpl.average_orientation_series(cond.width_files, float(velocity))
            builder = ScanPathBuilder(
                box,
                370.0,
                float(velocity) / 1000.0,
                140e-6,
                30e-6,
                67.0,
                EDGE_M / 2.0,
                0,
            )
            _quiet(builder.generate_layers)
            paths = _quiet(builder.process_vectors)
            n_sve = N_SVE[velocity]
            for lam in LAMBDAS:
                series = _scale_mean(base, lam)
                vol_ge20 = 0.0
                n_ge20 = 0
                n_all = 0
                max_esd = 0.0
                t0 = time.perf_counter()
                for i in range(n_sve):
                    np.random.seed(velocity * 1000 + int(round(lam * 100)) * 10 + i)
                    pool = _quiet(
                        create_melt_pool,
                        {
                            "width": (series, 50, 1.0, 2.0),
                            "depth": (series, 50, float(cond.eta_d), float(cond.depth_shape)),
                            "height": (series, 50, float(cond.eta_c), 1.0),
                        },
                        enable_random_phases=True,
                    )
                    phase = _quiet(compute_porosity, grid, paths, pool)
                    if int(np.count_nonzero(phase == 0)) < 2:
                        continue
                    morph = _quiet(
                        compute_morphology,
                        phase,
                        VOXEL_M,
                        ["equivalent_diameter_area", "area"],
                    )
                    diam = np.asarray(morph.get("equivalent_diameter_area", []), dtype=float)
                    vol = np.asarray(morph.get("area", []), dtype=float)
                    if diam.size == 0:
                        continue
                    n_all += int(diam.size)
                    max_esd = max(max_esd, float(diam.max()) * 1e6)
                    keep = diam >= ESD_MIN_M
                    n_ge20 += int(keep.sum())
                    if keep.any():
                        vol_ge20 += float(vol[keep].sum())
                elapsed = time.perf_counter() - t0
                volume_mm3 = n_sve * sve_m3 * 1e9
                row = {
                    "velocity_mm_s": velocity,
                    "lambda": lam,
                    "n_sve": n_sve,
                    "volume_mm3": f"{volume_mm3:.4f}",
                    "n_ge20": n_ge20,
                    "vf_percent_ge20": f"{(vol_ge20 / (n_sve * sve_m3)) * 100:.6e}",
                    "per_mm3_ge20": f"{n_ge20 / volume_mm3:.6e}",
                    "max_esd_um": f"{max_esd:.2f}",
                    "n_all": n_all,
                    "seconds": f"{elapsed:.1f}",
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()
                print(
                    f"V={velocity} λ={lam}  {n_sve} SVEs ({volume_mm3:.1f} mm³)  "
                    f">=20 µm: {n_ge20}  vf={row['vf_percent_ge20']} %  "
                    f"{row['per_mm3_ge20']}/mm³  max ESD {max_esd:.1f} µm  "
                    f"all defects {n_all}  {elapsed:.0f}s",
                    flush=True,
                )
    print(f"Wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    run()
