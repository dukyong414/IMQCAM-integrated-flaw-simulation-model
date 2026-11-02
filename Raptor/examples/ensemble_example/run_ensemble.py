# =============================================================================
# Copyright (c) 2025 Oak Ridge National Laboratory
#
# All rights reserved.
#
# This file is part of Raptor.
#
# For details, see the top-level LICENSE file at:
# https://github.com/ORNL-MDF/Raptor/LICENSE
# =============================================================================
"""Ensemble of stochastic lack-of-fusion defects at V = 1300 mm/s vs XCT.

Simulates an ensemble of SVEs whose total volume matches one XCT-scanned
Ti-6Al-4V coupon (CMU02-05-1), repeats it ``--n-mc`` times, and compares the
defect number density (ND) and volume fraction (VF) of defects larger than
20 um with the experiment. Bootstrap confidence intervals are reported for the
ensemble-mean ("true") ND and VF.

usage:
    python run_ensemble.py                # 20 runs x 2362 SVEs (a few minutes)
    python run_ensemble.py --n-mc 4 --n-sve 200   # quick look
    python run_ensemble.py --plot-only    # re-analyze ensemble_results.npz
"""
import argparse
import contextlib
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from raptor.api import (  # noqa: E402
    compute_morphology,
    compute_porosity,
    create_grid,
    create_melt_pool_from_spectra,
    create_path_vectors,
)
from raptor.io import read_data  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "data"
SPECTRUM_FILE = DATA_DIR / "meltPoolData" / "v1300_average_spectrum.csv"
DEFECT_FILE = DATA_DIR / "defectData" / "v1300_CMU02-05-1_eqd.csv"

# Process parameters of the coupon (V = 1300 mm/s).
POWER = 370.0  # W
SCAN_SPEED = 1.3  # m/s
HATCH_SPACING = 140.0e-6  # m
LAYER_HEIGHT = 30.0e-6  # m, nominal; effective thickness = L / 0.5
ROTATION = 67.0  # deg
SCAN_EXTENSION = 500.0e-6  # m

# Melt-pool statistics at V = 1300 mm/s (multitrack cross sections averaged
# over gas-flow angles): mean width and depth / cap aspect ratios.
MEAN_WIDTH = 1.727071074124999811e-04  # m
DEPTH_RATIO = 7.781430808729211668e-01
CAP_RATIO = 1.612419576611167915e-01
INFLATION = 1.04  # calibrated scaling of the mean width (lambda)
N_MODES = 50  # spectral rows used (mean + 49 modes)

# Simulation domain and ensemble.
SVE_EDGE = 500.0e-6  # m
VOXEL_RESOLUTION = 5.0e-6  # m
SEED0 = 13_000_000  # seed = SEED0 + 10000 * run + sve

# Analysis settings.
THRESHOLD_UM = 20.0
EQD_BINS = np.arange(0.0, 100.0, 2.5)  # um
N_BOOTSTRAP = 10_000
SIM_COLOR = "#2a78d6"
EXP_COLOR = "#1f1f1e"


def read_scanned_volume(path):
    """Scanned XCT volume (mm^3) from the data file header."""
    for line in open(path):
        if "scanned_volume_mm3" in line:
            return float(line.split("=")[1])
    raise ValueError(f"No scanned_volume_mm3 entry in {path}")


def build_model():
    """Melt pool, scan vectors and grid for one 500 um SVE."""
    spectrum = read_data(SPECTRUM_FILE)[:N_MODES]
    mean = MEAN_WIDTH * INFLATION
    melt_pool = create_melt_pool_from_spectra(
        {
            "width": (spectrum, 1.0, 2.0),
            "depth": (spectrum, DEPTH_RATIO, 1.0),  # parabolic depth
            "height": (spectrum, CAP_RATIO, 1.0),  # parabolic cap
        },
        enable_random_phases=True,
        means={"width": mean, "depth": mean, "height": mean},
    )

    # Layers sit one effective layer thickness (L / 0.5, the default powder
    # consolidation) apart. Extra layers keep every track whose melt pool can
    # reach the top of the grid.
    pitch = LAYER_HEIGHT / 0.5
    base_layers = int(SVE_EDGE // pitch + 1)
    top = LAYER_HEIGHT + SVE_EDGE + melt_pool.depth_max
    extra_layers = max(0, math.ceil(top / pitch) - base_layers) + 1
    path_vectors = create_path_vectors(
        np.array([[0.0, 0.0, 0.0], [SVE_EDGE] * 3]),
        POWER,
        SCAN_SPEED,
        HATCH_SPACING,
        LAYER_HEIGHT,
        ROTATION,
        SCAN_EXTENSION,
        extra_layers,
    )
    grid = create_grid(
        VOXEL_RESOLUTION,
        bound_box=np.array(
            [
                [0.0, 0.0, LAYER_HEIGHT],
                [SVE_EDGE, SVE_EDGE, LAYER_HEIGHT + SVE_EDGE],
            ]
        ),
    )
    return melt_pool, path_vectors, grid


def simulate(n_mc, n_sve, output):
    melt_pool, path_vectors, grid = build_model()
    eqds, counts = [], []
    for run in range(n_mc):
        run_eqds = []
        for sve in range(n_sve):
            with contextlib.redirect_stdout(io.StringIO()):
                phase = compute_porosity(
                    grid,
                    path_vectors,
                    melt_pool,
                    random_seed=SEED0 + 10000 * run + sve,
                    boundary_tracking=False,
                )
                pores = compute_morphology(
                    phase, VOXEL_RESOLUTION, ["equivalent_diameter_area"]
                )
            run_eqds.append(pores["equivalent_diameter_area"] * 1.0e6)
        run_eqds = np.concatenate(run_eqds)
        eqds.append(run_eqds)
        counts.append(run_eqds.size)
        n_large = int((run_eqds > THRESHOLD_UM).sum())
        print(
            f"run {run + 1}/{n_mc}: {run_eqds.size} pores, "
            f"{n_large} > {THRESHOLD_UM:.0f} um"
        )
    np.savez(
        output,
        eqd_um=np.concatenate(eqds),
        pores_per_run=np.array(counts),
        n_sve=n_sve,
        sve_volume_mm3=(SVE_EDGE * 1.0e3) ** 3,
    )
    print(f"Saved {output}")


def number_density(eqd_um, volume_mm3):
    """Defects > threshold per bin and unit volume (1/mm^3)."""
    hist, _ = np.histogram(eqd_um[eqd_um > THRESHOLD_UM], bins=EQD_BINS)
    return hist / volume_mm3


def volume_fraction(eqd_um, volume_mm3):
    """Volume fraction of defects > threshold (equivalent spheres)."""
    large = eqd_um[eqd_um > THRESHOLD_UM] * 1.0e-3  # mm
    return float(np.sum(np.pi / 6.0 * large**3) / volume_mm3)


def bootstrap_mean(values, rng):
    """Bootstrap 95% CI of the mean over runs (axis 0)."""
    values = np.asarray(values, dtype=np.float64)
    picks = rng.integers(0, values.shape[0], (N_BOOTSTRAP, values.shape[0]))
    means = values[picks].mean(axis=1)
    return values.mean(axis=0), np.percentile(means, [2.5, 97.5], axis=0)


def percentile_of(value, samples):
    """Mid-rank percentile of value within samples."""
    samples = np.asarray(samples)
    below = np.count_nonzero(samples < value)
    ties = np.count_nonzero(samples == value)
    return 100.0 * (below + 0.5 * ties) / samples.size


def analyze(results_path, figure_path):
    results = np.load(results_path)
    splits = np.cumsum(results["pores_per_run"])[:-1]
    runs = np.split(results["eqd_um"], splits)
    run_volume = float(results["n_sve"] * results["sve_volume_mm3"])

    exp_eqd = read_data(DEFECT_FILE)
    exp_volume = read_scanned_volume(DEFECT_FILE)
    exp_nd = number_density(exp_eqd, exp_volume)
    exp_vf = volume_fraction(exp_eqd, exp_volume)
    exp_total = exp_nd.sum()

    nd = np.array([number_density(r, run_volume) for r in runs])
    vf = np.array([volume_fraction(r, run_volume) for r in runs])
    total = nd.sum(axis=1)
    rng = np.random.default_rng(2024)
    nd_mean, nd_ci = bootstrap_mean(nd, rng)
    total_mean, total_ci = bootstrap_mean(total, rng)
    vf_mean, vf_ci = bootstrap_mean(vf, rng)

    def row(name, mean, ci, per_run, exp):
        lo, hi = np.percentile(per_run, [2.5, 97.5])
        inside = lo <= exp <= hi
        print(
            f"{name:<26}{mean:11.3e}  [{ci[0]:.3e}, {ci[1]:.3e}]"
            f"  [{lo:.3e}, {hi:.3e}]  {exp:11.3e}"
            f"  {percentile_of(exp, per_run):5.1f}  {'yes' if inside else 'no'}"
        )

    print(
        f"\nV = 1300 mm/s, lambda = {INFLATION}: {len(runs)} runs x "
        f"{int(results['n_sve'])} SVEs ({run_volume:.2f} mm^3 per run); "
        f"experiment {exp_volume:.2f} mm^3"
    )
    print(
        f"{'quantity (> 20 um)':<26}{'run mean':>11}  "
        f"{'bootstrap 95% CI of mean':<26}  {'per-run 2.5-97.5%':<26}"
        f"  {'experiment':>11}  {'pctl':>5}  in range"
    )
    row("ND total (1/mm^3)", total_mean, total_ci, total, exp_total)
    row("VF (-)", vf_mean, vf_ci, vf, exp_vf)
    counts = np.array([(r > THRESHOLD_UM).sum() for r in runs])
    print(
        f"defects > 20 um per run: {counts.mean():.2f} +/- "
        f"{counts.std(ddof=1) if counts.size > 1 else 0.0:.2f} "
        f"(experiment {int((exp_eqd > THRESHOLD_UM).sum())})"
    )

    plot(nd, nd_mean, nd_ci, exp_nd, vf, vf_mean, vf_ci, exp_vf, figure_path)


def plot(nd, nd_mean, nd_ci, exp_nd, vf, vf_mean, vf_ci, exp_vf, path):
    fig, (ax_nd, ax_vf) = plt.subplots(
        1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # Number density per 2.5 um bin above the threshold. Zero counts cannot
    # be shown on the log axis; they are capped at the lower edge and drawn
    # as downward triangles. Per-run markers are jittered within each bin.
    keep = (EQD_BINS[:-1] >= THRESHOLD_UM) & (EQD_BINS[1:] <= 30.0)
    centers = (EQD_BINS[:-1] + np.diff(EQD_BINS) / 2.0)[keep]
    nd_floor = 1.0e-4
    rng = np.random.default_rng(1)

    run_x = centers + rng.uniform(-0.6, 0.6, nd[:, keep].shape) - 0.3
    run_nd = nd[:, keep]
    zero = run_nd <= 0
    ax_nd.scatter(
        run_x[~zero],
        run_nd[~zero],
        s=22,
        facecolor="none",
        edgecolor=SIM_COLOR,
        alpha=0.6,
        label="Simulation, one marker per run and bin",
    )
    ax_nd.scatter(
        run_x[zero],
        np.full(int(zero.sum()), nd_floor),
        s=26,
        marker="v",
        color=SIM_COLOR,
        alpha=0.35,
        clip_on=False,
        label="Simulation, zero count (capped)",
    )

    mean = nd_mean[keep]
    lower, upper = nd_ci[0][keep], nd_ci[1][keep]
    shown = mean > 0
    ax_nd.errorbar(
        centers[shown] + 0.7,
        mean[shown],
        yerr=[
            mean[shown] - np.maximum(lower[shown], nd_floor),
            upper[shown] - mean[shown],
        ],
        fmt="s",
        ms=6,
        color=SIM_COLOR,
        capsize=4,
        label="Ensemble mean, bootstrap 95% CI",
    )

    exp = exp_nd[keep]
    ax_nd.scatter(
        centers[exp > 0] + 0.7,
        exp[exp > 0],
        s=70,
        marker="x",
        linewidths=2,
        color=EXP_COLOR,
        label="Experiment (XCT)",
    )
    ax_nd.scatter(
        centers[exp <= 0] + 0.7,
        np.full(int((exp <= 0).sum()), nd_floor),
        s=40,
        marker="v",
        color=EXP_COLOR,
        clip_on=False,
        label="Experiment, zero count (capped)",
    )
    ax_nd.set_yscale("log")
    ax_nd.set_ylim(nd_floor, 3.0e-2)
    ax_nd.set_xlim(THRESHOLD_UM, 30.0)
    ax_nd.set_xticks(np.arange(THRESHOLD_UM, 30.01, 2.5))
    ax_nd.set_xlabel("Defect equivalent diameter (µm), 2.5 µm bins")
    ax_nd.set_ylabel("Number density (1/mm$^3$ per bin)")
    ax_nd.set_title("Number density, defects > 20 µm")
    ax_nd.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )

    # Volume fraction in percent on a log axis. Runs without a defect above
    # the threshold have zero volume fraction and are drawn on the lower edge.
    vf_floor, vf_top = 1.0e-7, 1.0e-4
    vf_pct = vf * 100
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, vf.size)
    nonzero = vf_pct > 0
    ax_vf.scatter(
        jitter[nonzero],
        vf_pct[nonzero],
        s=28,
        facecolor="none",
        edgecolor=SIM_COLOR,
        alpha=0.7,
        label="Simulation, one marker per run",
    )
    n_zero = int((~nonzero).sum())
    if n_zero:
        ax_vf.scatter(
            jitter[~nonzero],
            np.full(n_zero, vf_floor),
            s=36,
            marker="v",
            color=SIM_COLOR,
            alpha=0.7,
            clip_on=False,
            label=f"Runs with no defect > 20 µm ({n_zero})",
        )
    ax_vf.errorbar(
        [0.35],
        [vf_mean * 100],
        yerr=[[(vf_mean - vf_ci[0]) * 100], [(vf_ci[1] - vf_mean) * 100]],
        fmt="s",
        color=SIM_COLOR,
        capsize=5,
        label="Ensemble mean, bootstrap 95% CI",
    )
    ax_vf.axhline(exp_vf * 100, color=EXP_COLOR, lw=1.6, label="Experiment")
    ax_vf.set_yscale("log")
    ax_vf.set_ylim(vf_floor, vf_top)
    ax_vf.set_xlim(-0.4, 0.6)
    ax_vf.set_xticks([])
    ax_vf.set_ylabel("Volume fraction (%)")
    ax_vf.set_title("Volume fraction, defects > 20 µm")
    ax_vf.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
    )

    for ax in (ax_nd, ax_vf):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"V = 1300 mm/s, λ = {INFLATION}: {nd.shape[0]} simulated coupons "
        "vs XCT"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-mc", type=int, default=20, help="ensemble runs")
    parser.add_argument(
        "--n-sve",
        type=int,
        default=None,
        help="SVEs per run (default: scanned XCT volume / SVE volume)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="re-analyze ensemble_results.npz without simulating",
    )
    args = parser.parse_args()

    results_path = SCRIPT_DIR / "ensemble_results.npz"
    if not args.plot_only:
        n_sve = args.n_sve or int(
            read_scanned_volume(DEFECT_FILE) / (SVE_EDGE * 1.0e3) ** 3
        )
        simulate(args.n_mc, n_sve, results_path)
    analyze(results_path, SCRIPT_DIR / "ensemble_v1300.png")


if __name__ == "__main__":
    main()
