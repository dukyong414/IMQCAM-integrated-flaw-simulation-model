# Ensemble example: stochastic lack-of-fusion at V = 1300 mm/s vs XCT

Each Monte Carlo run simulates a set of 500 µm SVEs whose total volume matches one XCT-scanned Ti-6Al-4V coupon
(CMU02-05-1, 295.32 mm³, giving 2362 SVEs). The script repeats this for several runs and compares defects
larger than 20 µm with the experiment:

* **number density (ND)**: defects per mm³ in 2.5 µm equivalent-diameter bins;
* **volume fraction (VF)**: Σ(π/6)d³ / volume of those defects, computed the same way for simulation and
  experiment.

The script reports the ensemble mean with a bootstrap 95% confidence interval (resampling the runs). It also gives
the run-to-run spread and the experiment's percentile within that spread. One run is the simulated counterpart of
one coupon, so the spread is what you would expect from scanning coupons of this size.

## Run

```bash
cd examples/ensemble_example
python run_ensemble.py                         # 20 runs x 2362 SVEs, a few minutes
python run_ensemble.py --n-mc 4 --n-sve 200    # quick look
python run_ensemble.py --plot-only             # re-analyze ensemble_results.npz
```

Outputs: `ensemble_results.npz` (pore diameters for every run), `ensemble_v1300.png`, and a summary table printed
to the console.

## Model settings

| Setting | Value |
|---|---|
| Process | P = 370 W, v = 1.3 m/s, H = 140 µm, L = 30 µm, 67° rotation |
| Effective layer thickness | L / 0.5 = 60 µm (default `powder_consolidation`) |
| Melt-pool width | spectrum `data/meltPoolData/v1300_average_spectrum.csv` (first 50 rows), mean W·λ with W = 172.7 µm, λ = 1.04 |
| Depth / cap | η_D = 0.778, η_C = 0.161 times the width, both parabolic (shape factor 1) |
| Phases | random per scan vector, seed = 13000000 + 10000·run + SVE |
| Porosity | `boundary_tracking=False`: every never-melted voxel is a pore |
| Voxels | 5 µm, 26-connected pores, single voxels removed |

## Data

* `data/meltPoolData/v1300_average_spectrum.csv`: width spectrum averaged over 8 gas-flow angles. Columns are
  amplitude (m) and frequency (Hz); row 0 is the mean. Source: Miner & Narra (2024),
  https://doi.org/10.1184/R1/25696293.v1.
* `data/defectData/v1300_CMU02-05-1_eqd.csv`: XCT equivalent spherical diameters (µm) of the defects in coupon
  CMU02-05-1. The header records the scanned volume.

## Expected output (20 runs)

```
quantity (> 20 um)           run mean  bootstrap 95% CI of mean    per-run 2.5-97.5%            experiment   pctl  in range
ND total (1/mm^3)           3.556e-03  [2.202e-03, 5.080e-03]  [0.000e+00, 1.016e-02]    1.016e-02   90.0  yes
VF (-)                      1.655e-08  [1.008e-08, 2.358e-08]  [0.000e+00, 4.779e-08]    5.856e-08  100.0  no
defects > 20 um per run: 1.05 +/- 1.00 (experiment 3)
```

* **Number density.** The coupon's 3 defects > 20 µm sit at the 90th percentile of the per-run distribution
  (1.05 ± 1.00 defects per simulated coupon), inside its 2.5–97.5% range. The experiment is consistent with the
  ensemble, near its upper tail.
* **Volume fraction.** The experiment lies above every simulated run. All simulated defects > 20 µm are smaller
  than 22.5 µm, while the coupon contains one 24.2 µm defect, and the d³ weighting makes that single defect
  dominate. With 3 experimental defects, this rests on one large outlier.
