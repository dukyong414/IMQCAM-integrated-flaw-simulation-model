# Integrated porosity module

This repository joins three porosity models in one GUI and one Python package:

- **CMU** samples lack-of-fusion (LoF), keyhole, and gas pores from process parameters.
- **NU** fits a spatial porosity field on the build plate and uses it to place those pores.
- **Raptor** (ORNL) computes LoF from the scan path and a measured melt-pool spectrum. When Raptor runs, its LoF replaces the CMU LoF. Keyhole and gas stay CMU.

Run commands from the repository root. The Python packages are `integrated_cli` and `integrated_gui`.

## Install

```bat
py -3 -m venv "%LOCALAPPDATA%\NASA-porosity\.venv"
"%LOCALAPPDATA%\NASA-porosity\.venv\Scripts\python.exe" -m pip install -r requirements_integrated.txt
"%LOCALAPPDATA%\NASA-porosity\.venv\Scripts\python.exe" -m pip install -e .\Raptor
```

`run_gui.bat` and `run_cli.bat` use that virtual environment. If it is missing they fall back to `.venv` in this folder. Set `PYTHONIOENCODING=utf-8`. For off-screen figures also set `PYVISTA_OFF_SCREEN=true`.

Raptor uses a Numba melt-mask kernel and picks a GPU when one is available. Set `NASA_FORCE_CPU=1` to force CPU.

The melt-pool library is the folder `Files from Vamsi` next to this repository (`ULI_v*_theta*_widths.txt`, `flucWs.txt`, `flucWsMTs.txt`, `depthShapes.txt`).

## GUI

```bat
run_gui.bat
```

Opens http://127.0.0.1:8050. Leave the window open. Close it to stop the server.

The pages are Welcome, Combined, CMU, NU, and Raptor.

- **Combined** builds a plate of coupon cubes, one cuboid, or an STL. It draws CMU pore sizes, modulates their number with the NU field, and can replace LoF with a Raptor run on that geometry.
- **CMU** runs the CMU pore models alone.
- **NU** fits the spatial field and writes the shared cache that Combined reuses.
- **Raptor** runs LoF on a cuboid or an STL band and shows the phase volume.

From Python, the same server is `integrated_gui.app.launch(host="127.0.0.1", port=8050)`.

## Command line

`python -m integrated_cli` with no arguments starts the GUI. `run_cli.bat` forwards every argument.

| Command | What it does |
|---|---|
| `gui` | Dash GUI on port 8050 |
| `cmu` | CMU pore-size samples, or a full CMU model run |
| `nu fit` | Fit the NU field and save `nu_model.pkl` |
| `nu predict` | Porosity grid from a fitted model |
| `combined` | Plate of pores: CMU density × volume × NU field |
| `raptor-stl` | Raptor LoF on a large STL, one tile at a time |

```bat
run_cli.bat cmu --outdir out\cmu1 --totalpores 5000 --mode sizes
run_cli.bat nu fit --outdir out\nu1
run_cli.bat nu predict --outdir out\nu1 --mean-porosity 0.02 --plate-w 250 --plate-h 250
run_cli.bat combined --outdir out\plate1 --n-strips 4 --n-rows 19 --n-cols 3
run_cli.bat raptor-stl --velocity 1300 --lambda 1.04
run_cli.bat raptor-stl --sector-fraction 0.125
run_cli.bat raptor-stl --assemble-only
```

`combined` writes `config.json`, then `pores.csv`, `cubes.csv`, and `status.json`. Pore count defaults to CMU density times build volume, modulated by the NU field. `--max-pores 0` leaves that count uncapped.

`raptor-stl` defaults to the P&W challenge-part STL, 1300 mm/s, λ = 1.04, 5 µm voxels, and 3 mm tiles. Output goes to `D:\NASA-porosity-runs\<part>_V<speed>_lambda<λ>` when drive D exists. Rerun the same command to resume: finished tiles are skipped. `--assemble-only` joins whatever tiles are already on disk into `pores.csv`, `summary.json`, and the plots. A 3 mm tile at 5 µm uses about 1 GB. Do not voxelize a full part in one array.

## Core functions

Import from the repository root so `integrated_gui` is on `sys.path`.

### CMU pore sizes

`integrated_gui.cmu_labeled.run_cmu_labeled_samples` draws a catalog of pores with a `pore_source` column (`lof`, `keyhole`, or `gas`). Call it with the working directory set to `CMU team`, or pass absolute data directories. It returns `(samples, equivalent_volume_mm3, meta)`. `meta["densities"]` and `meta["counts"]` are per mechanism. The CLI `cmu --mode sizes` is this function.

```python
from integrated_gui.cmu_labeled import run_cmu_labeled_samples

samples, volume_mm3, meta = run_cmu_labeled_samples(
    param_dict, props,
    pore_model_names=["lof", "keyhole", "gas"],
    meltpool_model_name="Rosenthal",
    totalpores=5000,
)
```

`param_dict` is `CMU team/input_parameters/params.json`. `props` is `CMU team/material_models/test_ti64_rosenthal.json` with the numeric process parameters copied onto it.

### NU spatial field

`integrated_cli.main._fit_nu_model` builds `porosity_field_model.PorosityPatternModel` from the datasets listed in `integrated_gui.nu_panel.DEFAULT_DATASETS`, fits it, and returns the model. `nu fit` pickles that model to `nu_model.pkl` and also writes the GUI cache through `integrated_gui.nu_cache.save_fitted_model`, so Combined does not fit again.

Prediction uses the fitted object's grid method. The CLI writes `predicted_porosity.csv` and `predicted_field.png` for a plate at `--resolution` mm. `--mean-porosity` sets the field mean (fraction, so `0.02` is 2%).

### Plate layout and pore cloud

`integrated_gui.combined_3d.make_plate_layout` returns a `PlateLayout`. `plate_mode` is one of:

- `grid_cubes` — `n_strips` blocks of `n_rows` × `n_cols` coupons
- `single_cuboid` — one block on the plate
- `stl` — a mesh fit by `stl_path`

`integrated_gui.combined_3d.build_plate_pore_cloud` places the CMU size catalog into that layout. Each build `i` gets about `ρ · V_i · (NU_i / mean_NU)` pores, then the pores are dispersed inside the build. Pass `raptor_lof` (the dict from `run_raptor_rve`) to replace LoF sizes and density with Raptor. Gas and keyhole stay CMU. It returns `(pores_dataframe, summary_dict)`.

```python
from integrated_gui.combined_3d import make_plate_layout, build_plate_pore_cloud

layout = make_plate_layout(
    plate_width_mm=250, plate_height_mm=250,
    n_strips=4, n_rows=19, n_cols=3, cube_size_mm=10,
)
pores, summary = build_plate_pore_cloud(
    samples, meta["densities"], layout,
    nu_model=model, mean_porosity=0.02, seed=42,
)
```

`integrated_gui.stl_fit.load_and_fit_stl` scales an STL onto the plate, sits it on z = 0, and returns `(mesh, meta)`. `sector_fraction` below 1 keeps a wedge about the vertical axis (a cake slice). `clip_height_mm` keeps only the bottom slab.

`integrated_gui.stl_slicer_scan.slice_scan_stl` slices that mesh and hatches every layer. Hatch spacing, layer thickness, stripe width, and layer rotation are arguments. It returns a dict of segments, contours, and optional edge pores. `layer_indices=None` scans the whole part.

`integrated_gui.scan_vectors.generate_scan_vectors_rect` does the same hatch on a rectangle, without an STL.

### Raptor on one cuboid

`integrated_gui.combined_raptor.run_raptor_rve(cfg)` hatches a cuboid with Raptor's `ScanPathBuilder` and returns a dict of LoF pores, phase-field paths, and densities. Lengths in `cfg` are metres except `size_xyz_mm`.

```python
from integrated_gui.combined_raptor import run_raptor_rve

result = run_raptor_rve({
    "size_xyz_mm": [5.0, 5.0, 14.0],
    "power_w": 370,
    "velocity_m_s": 1.3,
    "hatch_m": 140e-6,
    "layer_m": 30e-6,
    "rotation_deg": 67,
    "voxel_resolution_m": 5e-6,
    "random_seed": 42,
    "library": {"folder": r"..\Files from Vamsi", "eta_source": "flucWsMTs", "theta_mode": "averaged"},
    "lambda_mean": 1.04,
})
```

`library.theta_mode` is `averaged` (one melt pool from the mean of the eight orientation files), `per_layer` (each track uses the angle file nearest its heading), or `fixed`. `lambda_mean` scales the mean width, depth, and cap. Fluctuations are not scaled. `1.04` and `1.14` are the two paper values.

Powder consolidation is Raptor's default `0.5`. A nominal 30 µm layer is placed every 60 µm. That pitch is separate from changing λ.

`integrated_gui.meltpool_library.load_library` reads the width files into `{velocity_mm_s: VelocityCondition}`. `resolve_condition` picks the library velocity nearest the process speed. `compute_porosity_grouped` is the call `run_raptor_rve` uses: one melt pool per scan-angle group, then a merged phase field. It returns `(phase_field, counts_by_theta, files_by_theta)`.

### Raptor on a large STL

A whole part at 5 µm does not fit in memory. `integrated_gui.stl_raptor_tiles.run` slices every layer once, saves the tracks under `layers\`, and runs Raptor on one 3 mm tile at a time. Each tile is one file in `tiles\`. `assemble` joins pores that cross a tile face and writes `pores.csv` and `summary.json`.

```python
from integrated_gui.stl_raptor_tiles import run

summary = run(
    {"stl": r"path\to\part.stl", "velocity_mm_s": 1300, "lambda_mean": 1.04,
     "voxel_um": 5, "tile_mm": 3, "sector_fraction": 0.125},
    max_tiles=None,          # None runs every tile; set a small int to time a few
    assemble_only=False,
)
```

`run` calls `prepare_layers`, `run_tiles`, and `assemble`. A second call with the same output folder and the same settings skips tiles already on disk. Changing velocity, λ, voxel size, or tile size starts a new folder. `default_outdir` builds that folder name.

Pores are counted when their center is at least `surface_margin_mm` (default 0.5 mm) inside the part. `summary["density_per_mm3"]` uses the voxelized part volume.

### 5 × 5 × 14 mm velocity sweep

`integrated_gui.velocity_sweep.run_sweep(outdir, theta_mode="averaged", lambda_mean=1.04)` runs 1300 to 1700 mm/s in steps of 50 on a 5 × 5 × 14 mm pillar (370 W, 140 µm hatch, 30 µm layer). Each speed gets a `V_####` folder. `build_config` is the per-speed config. `drop_near_surface_lof` removes LoF whose center is within 0.5 mm of an outer face. `write_predictions` writes `predictions.csv`.

`integrated_gui.share_viewer.main` (copied into the sweep folder as `view_results.py`) opens the result browser on port 8055.

### Figures

`integrated_gui.raptor_view.save_view3d_npz` stores the six outer phase faces and interior pore coordinates for a Plotly view. `make_plotly_phase_figure(npz_path)` draws those faces plus LoF pores. `write_phase_screenshot(vti_path, png_path)` writes the PyVista two-panel image from a full `.vti`.

`integrated_gui.velocity_sweep._save_melted_build` and `_save_melted_build_from_view` color the outer faces: grey melted, black boundary, yellow track overlap. A pore voxel on the skin is drawn as melted metal so the legend stays those three phases.
