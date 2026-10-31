"""
Unified command-line interface for the integrated porosity project.

Examples
--------
  python -m integrated_cli combined --outdir out/run1 --n-strips 4
  python -m integrated_cli cmu --totalpores 3000 --outdir out/cmu1
  python -m integrated_cli nu fit --outdir out/nu1
  python -m integrated_cli nu predict --outdir out/nu1 --mean-porosity 0.02
  python -m integrated_cli gui
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CMU_DIR = ROOT / "CMU team"
NU_DIR = ROOT / "NU team"
DEFAULT_PARAMS = CMU_DIR / "input_parameters" / "params.json"
DEFAULT_PROPS = CMU_DIR / "material_models" / "test_ti64_rosenthal.json"


def _utf8():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _ensure_outdir(path: str | Path) -> Path:
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def cmd_gui(args: argparse.Namespace) -> int:
    import threading
    import webbrowser

    # Before any matplotlib/pyvista import: prevent Tk crashes in Dash threads
    from integrated_gui.headless import configure_headless_plotting

    configure_headless_plotting()
    from integrated_gui.app import launch

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        def _open():
            import time
            time.sleep(1.2)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()
        print(f"Opening browser at {url}")
    else:
        print(f"Dash GUI available at {url}")

    launch(host=args.host, port=args.port, debug=args.debug)
    return 0


def cmd_cmu(args: argparse.Namespace) -> int:
    if str(CMU_DIR) not in sys.path:
        sys.path.insert(0, str(CMU_DIR))

    outdir = _ensure_outdir(args.outdir)
    params = _load_json(Path(args.params) if args.params else DEFAULT_PARAMS)
    props = _load_json(Path(args.props) if args.props else DEFAULT_PROPS)
    for k, v in params.items():
        if isinstance(v, (int, float)):
            props[k] = v

    models = args.pore_models or ["lof", "keyhole", "gas"]
    prev = os.getcwd()
    os.chdir(CMU_DIR)
    try:
        from integrated_gui.cmu_labeled import run_cmu_labeled_samples
        from integrated_gui.cmu_data import resolve_cmu_data_dir
        from core.execute_porosity_model import execute_porosity_model

        if args.mode == "sizes":
            samples, equiv_vol, meta = run_cmu_labeled_samples(
                param_dict=params,
                props=props,
                pore_model_names=models,
                meltpool_model_name=args.meltpool,
                totalpores=int(args.totalpores),
            )
            samples.to_csv(outdir / "samples.csv", index=False)
            with open(outdir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "equivalent_volume_mm3": equiv_vol,
                        "densities": meta["densities"],
                        "counts": meta["counts"],
                        "total_density": meta["total_density"],
                    },
                    f,
                    indent=2,
                )
            print(f"Wrote {outdir / 'samples.csv'}  (N={len(samples)})")
            print(f"CMU counts: {meta['counts']}")
        else:
            results = execute_porosity_model(
                itemids=args.itemids or [28152209, 26304175],
                data_dir=args.data_dir,
                output_dir=str(outdir),
                pore_model_names=models,
                meltpool_model_name=args.meltpool,
                param_dict=params,
                props=props,
                totalpores=int(args.totalpores),
                merged_data_dir=resolve_cmu_data_dir(args.merged_data_dir, args.itemids),
                return_figure=False,
            )
            print(f"CMU run complete. Output: {results['output_dir']}")
            print(f"Plot: {results['plot_path']}")
    finally:
        os.chdir(prev)
    return 0


def _fit_nu_model(scaling: str, smoothing: float, homogenize_x: bool):
    if str(NU_DIR) not in sys.path:
        sys.path.insert(0, str(NU_DIR))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from porosity_field_model import PorosityPatternModel, SizeFilteredPorosityDataset
    from integrated_gui import nu_panel

    model = PorosityPatternModel(
        scaling=scaling,
        smoothing=smoothing,
        homogenize_x=homogenize_x,
    )
    for cfg in nu_panel.DEFAULT_DATASETS:
        if not os.path.isdir(cfg["base_dir"]):
            raise FileNotFoundError(f"NU dataset path not found: {cfg['base_dir']}")
        ignore = set(
            int(x) for x in str(cfg.get("ignore_indices", "")).split(",") if x.strip().isdigit()
        )
        ds = SizeFilteredPorosityDataset(
            title=cfg["title"],
            base_dir=cfg["base_dir"],
            folder_prefix=cfg["folder_prefix"],
            ignore_indices=ignore,
            start_idx=int(cfg["start_idx"]),
            cols=int(cfg["cols"]),
            index_pattern=re.compile(cfg["index_pattern"], re.I),
            core_pitch_mm=float(cfg["core_pitch_mm"]),
            plate_width_mm=float(cfg["plate_width_mm"]),
            plate_height_mm=float(cfg["plate_height_mm"]),
        )
        model.add_dataset(ds)
    model.fit()
    return model


def cmd_nu_fit(args: argparse.Namespace) -> int:
    outdir = _ensure_outdir(args.outdir)
    model = _fit_nu_model(args.scaling, float(args.smoothing), bool(args.homogenize_x))
    model_path = outdir / "nu_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    # Also write shared GUI cache so Combined does not re-fit
    try:
        from integrated_gui.nu_cache import save_fitted_model, MODEL_PATH

        save_fitted_model(
            model,
            meta={
                "source": "cli_nu_fit",
                "scaling": args.scaling,
                "smoothing": float(args.smoothing),
                "homogenize_x": bool(args.homogenize_x),
            },
        )
        print(f"Shared GUI cache -> {MODEL_PATH}")
    except Exception as exc:
        print(f"Warning: could not write shared NU cache: {exc}", file=sys.stderr)
    fig = model.plot_training_field(show=False, homogenize_x=False)
    fig.savefig(outdir / "training_field.png", dpi=150, bbox_inches="tight")
    print(f"Fitted NU model -> {model_path}")
    print(f"Training plot -> {outdir / 'training_field.png'}")
    return 0


def cmd_nu_predict(args: argparse.Namespace) -> int:
    outdir = _ensure_outdir(args.outdir)
    model_path = Path(args.model) if args.model else outdir / "nu_model.pkl"
    if not model_path.exists():
        raise SystemExit(
            f"No fitted model at {model_path}. Run: python -m integrated_cli nu fit --outdir {outdir}"
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    mean_p = float(args.mean_porosity)
    std_p = float(args.std_porosity) if args.std_porosity is not None else None
    w, h = float(args.plate_w), float(args.plate_h)
    res = float(args.resolution)

    df = model.predict_grid(
        width_mm=w,
        height_mm=h,
        mean_porosity=mean_p,
        std_porosity=std_p,
        resolution_mm=res,
    )
    csv_path = outdir / "predicted_grid.csv"
    df.to_csv(csv_path, index=False)

    fig = model.plot_predicted_field(
        width_mm=w,
        height_mm=h,
        mean_porosity=mean_p,
        std_porosity=std_p,
        resolution_mm=res,
        homogenize_x=bool(args.homogenize_x),
        show=False,
    )
    fig.savefig(outdir / "predicted_field.png", dpi=150, bbox_inches="tight")
    print(f"Grid -> {csv_path}  ({len(df)} points)")
    print(f"Plot -> {outdir / 'predicted_field.png'}")
    print(
        f"min={df.predicted_porosity_pct.min():.4f}%  "
        f"max={df.predicted_porosity_pct.max():.4f}%  "
        f"mean={df.predicted_porosity_pct.mean():.4f}%"
    )
    return 0


def cmd_combined(args: argparse.Namespace) -> int:
    outdir = _ensure_outdir(args.outdir)
    params = _load_json(Path(args.params) if args.params else DEFAULT_PARAMS)
    props = _load_json(Path(args.props) if args.props else DEFAULT_PROPS)

    # CLI overrides for process params
    if args.power is not None:
        params["Power (W)"] = float(args.power)
    if args.velocity is not None:
        params["Velocity (mm/s)"] = float(args.velocity)
    if args.hatch is not None:
        params["Hatch Spacing (um)"] = float(args.hatch)
    if args.layer is not None:
        params["Layer Thickness (um)"] = float(args.layer)
    for k, v in params.items():
        if isinstance(v, (int, float)):
            props[k] = v

    models = args.pore_models or ["lof", "keyhole", "gas"]
    if args.grid is not None:
        # Legacy square grid: one block of n×n
        n_rows = int(args.grid)
        n_cols = int(args.grid)
        n_strips = 1
    else:
        n_rows = int(args.n_rows)
        n_cols = int(args.n_cols)
        n_strips = int(args.n_strips)
    # Default uncapped (0); optional --max-pores / --totalpores for a safety cap
    if args.max_pores is not None:
        max_pores = int(args.max_pores)
    elif args.totalpores is not None:
        max_pores = int(args.totalpores)
    else:
        max_pores = 0
    cfg = {
        "build_shape": str(args.build_shape),
        "plate_mode": str(args.plate_mode),
        "stl_path": args.stl,
        "enable_scan_vectors": bool(args.enable_scan),
        "scan_strategy": str(args.scan_strategy),
        "n_edge_pores": int(args.n_edge_pores),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_strips": n_strips,
        "cube_size": float(args.cube_size),
        "cube_height": float(args.cube_height if args.cube_height is not None else args.cube_size),
        "plate_w": float(args.plate_w),
        "plate_h": float(args.plate_h),
        "seed": int(args.seed),
        "max_pores": max_pores,
        "totalpores": max_pores,
        "mean_porosity": float(args.mean_porosity),
        "pore_models": models,
        "param_dict": params,
        "props": props,
    }
    cfg_path = outdir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Running Combined plate 3D -> {outdir}")
    print(f"Config: {cfg_path}")
    print(
        f"  strips={n_strips} × {n_rows}×{n_cols} cubes  "
        f"N=ρ_CMU×V (cap={max_pores or 'none'})"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "integrated_gui.combined_worker",
            "--config",
            str(cfg_path),
            "--outdir",
            str(outdir),
        ],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        print("Combined run FAILED. See worker.log in outdir.", file=sys.stderr)
        return proc.returncode

    status_path = outdir / "status.json"
    if status_path.exists():
        status = _load_json(status_path)
        if status.get("ok"):
            s = status.get("summary") or {}
            print("Combined OK")
            print(f"  pores.csv: {outdir / 'pores.csv'}")
            print(f"  cubes.csv: {outdir / 'cubes.csv'}")
            print(f"  total pores: {s.get('total_pores')}")
            print(f"  bulk pores: {s.get('n_bulk_pores')}")
            print(f"  CMU ρ: {s.get('cmu_total_density')}")
            print(f"  NU mean %: {s.get('nu_mean_pct')}")
            print(f"  counts realized: {s.get('counts_realized')}")
        else:
            print("Combined reported failure:", status.get("message"), file=sys.stderr)
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="integrated_cli",
        description="Unified CLI for CMU pore models, NU spatial field, Combined 3D, and GUI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- gui ---
    g = sub.add_parser("gui", help="Launch the integrated Dash GUI")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8050)
    g.add_argument("--debug", action="store_true")
    g.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser tab")
    g.set_defaults(func=cmd_gui)

    # --- cmu ---
    c = sub.add_parser("cmu", help="Run CMU pore-source sampling")
    c.add_argument("--outdir", required=True, help="Output directory")
    c.add_argument("--mode", choices=["full", "sizes"], default="full",
                   help="full=execute_porosity_model; sizes=labeled size samples only")
    c.add_argument("--totalpores", type=int, default=5000)
    c.add_argument("--pore-models", nargs="+", default=["lof", "keyhole", "gas"])
    c.add_argument("--meltpool", default="Rosenthal")
    c.add_argument("--params", default=None, help="Process params JSON")
    c.add_argument("--props", default=None, help="Material props JSON")
    c.add_argument("--data-dir", default="rawdata")
    c.add_argument("--merged-data-dir", default="data")
    c.add_argument("--itemids", nargs="+", type=int, default=None)
    c.set_defaults(func=cmd_cmu)

    # --- nu ---
    n = sub.add_parser("nu", help="NU spatial field model")
    nsub = n.add_subparsers(dest="nu_command", required=True)

    nf = nsub.add_parser("fit", help="Fit NU model on default datasets")
    nf.add_argument("--outdir", required=True)
    nf.add_argument("--scaling", default="z_score")
    nf.add_argument("--smoothing", type=float, default=0.02)
    nf.add_argument("--homogenize-x", action="store_true")
    nf.set_defaults(func=cmd_nu_fit)

    np_ = nsub.add_parser("predict", help="Predict porosity grid from fitted model")
    np_.add_argument("--outdir", required=True, help="Dir with nu_model.pkl (or use --model)")
    np_.add_argument("--model", default=None, help="Path to nu_model.pkl")
    np_.add_argument("--plate-w", type=float, default=120.0)
    np_.add_argument("--plate-h", type=float, default=120.0)
    np_.add_argument("--mean-porosity", type=float, default=0.02)
    np_.add_argument("--std-porosity", type=float, default=None)
    np_.add_argument("--resolution", type=float, default=1.0)
    np_.add_argument("--homogenize-x", action="store_true")
    np_.set_defaults(func=cmd_nu_predict)

    # --- combined ---
    m = sub.add_parser(
        "combined",
        help="Combined plate 3D: CMU ρ×V × NU/mean, random disperse (default 4×19×3)",
    )
    m.add_argument("--outdir", required=True, help="Output directory")
    m.add_argument("--n-rows", type=int, default=19, help="Rows per strip (default 19)")
    m.add_argument("--n-cols", type=int, default=3, help="Cols per strip (default 3)")
    m.add_argument("--n-strips", type=int, default=4, help="Number of vertical strips (default 4)")
    m.add_argument(
        "--grid",
        type=int,
        default=None,
        help="Deprecated legacy square n×n; prefer --n-rows/--n-cols/--n-strips",
    )
    m.add_argument(
        "--plate-mode",
        choices=["grid_cubes", "single_cuboid", "stl"],
        default="grid_cubes",
        help="Geometry: coupon cubes, one cuboid, or STL",
    )
    m.add_argument("--stl", default=None, help="STL path when --plate-mode stl")
    m.add_argument("--enable-scan", action="store_true", help="Generate scan vectors + edge pores")
    m.add_argument(
        "--scan-strategy",
        choices=["horizontal", "vertical", "diagonal", "batch"],
        default="horizontal",
    )
    m.add_argument("--n-edge-pores", type=int, default=100)
    m.add_argument(
        "--build-shape",
        choices=["cube", "cylinder"],
        default="cube",
        help="Deprecated; use --plate-mode",
    )
    m.add_argument("--cube-size", type=float, default=10.0, help="Cube edge (mm)")
    m.add_argument("--cube-height", type=float, default=None, help="Build height Z (mm); default=cube-size")
    m.add_argument("--plate-w", type=float, default=250.0)
    m.add_argument("--plate-h", type=float, default=250.0)
    m.add_argument(
        "--max-pores",
        type=int,
        default=None,
        help="Optional safety cap (default uncapped N=ρ×V). 0 also means uncapped.",
    )
    m.add_argument(
        "--totalpores",
        type=int,
        default=None,
        help="Alias for --max-pores; default uncapped empirical ρ·V·NU",
    )
    m.add_argument("--mean-porosity", type=float, default=0.02)
    m.add_argument("--pore-models", nargs="+", default=["lof", "keyhole", "gas"])
    m.add_argument("--power", type=float, default=None)
    m.add_argument("--velocity", type=float, default=None)
    m.add_argument("--hatch", type=float, default=None)
    m.add_argument("--layer", type=float, default=None)
    m.add_argument("--params", default=None)
    m.add_argument("--props", default=None)
    m.add_argument("--seed", type=int, default=42)
    m.set_defaults(func=cmd_combined)

    return p


def main(argv=None) -> int:
    _utf8()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # No args → launch Dash GUI (single-command UX)
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0:
        argv = ["gui"]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
