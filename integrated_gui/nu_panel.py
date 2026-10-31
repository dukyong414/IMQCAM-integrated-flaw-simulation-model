"""NU spatial porosity-field panel for the integrated GUI."""
from __future__ import annotations

import io
import os
import re
import sys
import traceback
from typing import List, Optional

import dash
from dash import ALL, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from .utils import empty_figure_placeholder, fig_to_img, parse_float, parse_int_list

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NASA_PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
NU_DIR = os.path.join(ROOT, "NU team")

if NU_DIR not in sys.path:
    sys.path.insert(0, NU_DIR)

from porosity_field_model import (  # noqa: E402
    PorosityPatternModel,
    SizeFilteredPorosityDataset,
)

# In-memory fitted model for predict callbacks
_FITTED_MODEL: Optional[PorosityPatternModel] = None
_LAST_GRID_CSV: Optional[str] = None

# NU spatial field: Sneha km_ti64 (Narra) only. CAMI/SnT belongs with Raptor, not here.
_SNEHA_KM_TI64_DIR = os.path.join(
    NASA_PROJECT_ROOT,
    "Porosity_analysis_all_datasets",
    "2026-02-Narra-1013031",
    "size_filtered_porosity_pcts",
)

DEFAULT_DATASETS = [
    {
        "title": "2-BM 370W 1200mm/s (Size-Filtered >=50um)",
        "base_dir": _SNEHA_KM_TI64_DIR,
        "folder_prefix": "km_ti64",
        "ignore_indices": "116,170",
        "start_idx": 58,
        "cols": 3,
        "index_pattern": r"KM_Ti64_(\d+)",
        "core_pitch_mm": 6.0,
        "plate_width_mm": 120.0,
        "plate_height_mm": 120.0,
    },
]


def _dataset_row(idx: int, cfg: dict):
    return html.Div(
        [
            html.Div(
                [
                    html.Strong(f"Dataset {idx + 1}"),
                    dbc.Button(
                        "Remove",
                        id={"type": "nu-ds-remove", "index": idx},
                        color="link",
                        size="sm",
                        className="float-end text-danger",
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Title", className="control-label"),
                            dbc.Input(
                                id={"type": "nu-ds-title", "index": idx},
                                value=cfg.get("title", ""),
                            ),
                        ],
                        md=12,
                    ),
                ],
                className="mb-1",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Base directory", className="control-label"),
                            dbc.Input(
                                id={"type": "nu-ds-base-dir", "index": idx},
                                value=cfg.get("base_dir", ""),
                            ),
                        ],
                        md=12,
                    ),
                ],
                className="mb-1",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Folder prefix", className="control-label"),
                            dbc.Input(
                                id={"type": "nu-ds-prefix", "index": idx},
                                value=cfg.get("folder_prefix", ""),
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Index regex", className="control-label"),
                            dbc.Input(
                                id={"type": "nu-ds-pattern", "index": idx},
                                value=cfg.get("index_pattern", r"(\d+)"),
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        [
                            html.Div("Ignore indices", className="control-label"),
                            dbc.Input(
                                id={"type": "nu-ds-ignore", "index": idx},
                                value=cfg.get("ignore_indices", ""),
                                placeholder="e.g. 116,170",
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-1",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Start idx", className="control-label"),
                            dbc.Input(
                                id={"type": "nu-ds-start", "index": idx},
                                type="number",
                                value=cfg.get("start_idx", 0),
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div("Cols", className="control-label"),
                            dbc.Input(
                                id={"type": "nu-ds-cols", "index": idx},
                                type="number",
                                value=cfg.get("cols", 3),
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Div("Pitch (mm)", className="control-label"),
                            dbc.Input(
                                id={"type": "nu-ds-pitch", "index": idx},
                                type="number",
                                value=cfg.get("core_pitch_mm", 6.0),
                                step=0.1,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div("Plate W×H (mm)", className="control-label"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        id={"type": "nu-ds-pw", "index": idx},
                                        type="number",
                                        value=cfg.get("plate_width_mm", 120.0),
                                    ),
                                    dbc.Input(
                                        id={"type": "nu-ds-ph", "index": idx},
                                        type="number",
                                        value=cfg.get("plate_height_mm", 120.0),
                                    ),
                                ]
                            ),
                        ],
                        md=4,
                    ),
                ],
            ),
        ],
        className="dataset-row",
    )


def layout():
    return html.Div(
        [
            dcc.Store(id="nu-dataset-store", data=DEFAULT_DATASETS),
            dcc.Store(id="nu-model-ready", data=False),
            dcc.Download(id="nu-csv-download"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H3("NU — Spatial Field Model"),
                                html.P(
                                    "Train an RBF spatial porosity pattern from size-filtered CT datasets, "
                                    "then predict on a target build plate.",
                                    className="text-muted mb-3",
                                ),
                                html.H4("Model options"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div("Scaling", className="control-label"),
                                                dcc.Dropdown(
                                                    id="nu-scaling",
                                                    options=[
                                                        {"label": s, "value": s}
                                                        for s in PorosityPatternModel.VALID_SCALINGS
                                                    ],
                                                    value="z_score",
                                                    clearable=False,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div("Smoothing", className="control-label"),
                                                dbc.Input(
                                                    id="nu-smoothing",
                                                    type="number",
                                                    value=0.02,
                                                    step=0.001,
                                                    min=0,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div("Homogenize X", className="control-label"),
                                                dbc.Checklist(
                                                    id="nu-homogenize",
                                                    options=[{"label": "X-averaged trend", "value": "on"}],
                                                    value=[],
                                                    switch=True,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                html.H4("Training datasets"),
                                html.Div(id="nu-dataset-rows"),
                                dbc.Button(
                                    "Add dataset",
                                    id="nu-add-dataset",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    className="mb-3",
                                ),
                                dbc.Button(
                                    "Fit model",
                                    id="nu-fit-model",
                                    color="primary",
                                    className="w-100 btn-nasa mb-3",
                                ),
                                html.Hr(className="section-divider"),
                                html.H4("Predict"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div("Plate width (mm)", className="control-label"),
                                                dbc.Input(id="nu-plate-w", type="number", value=120.0),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div("Plate height (mm)", className="control-label"),
                                                dbc.Input(id="nu-plate-h", type="number", value=120.0),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div("Resolution (mm)", className="control-label"),
                                                dbc.Input(id="nu-resolution", type="number", value=0.5, step=0.1),
                                            ],
                                            md=4,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div("Mean porosity (%)", className="control-label"),
                                                dbc.Input(id="nu-mean-porosity", type="number", value=0.02, step=0.001),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div("Std porosity (optional)", className="control-label"),
                                                dbc.Input(
                                                    id="nu-std-porosity",
                                                    type="number",
                                                    placeholder="pooled training σ",
                                                    step=0.001,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div("Process label", className="control-label"),
                                                dbc.Input(id="nu-process-label", type="text", value=""),
                                            ],
                                            md=4,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                html.Div("Sample regions (x,y[,w,h,label] per line)", className="control-label"),
                                dbc.Textarea(
                                    id="nu-sample-regions",
                                    rows=3,
                                    value="60, 30, 6, 6, A\n60, 90, 6, 6, B",
                                    className="mb-2",
                                ),
                                dbc.Checklist(
                                    id="nu-predict-homogenize",
                                    options=[{"label": "Predict with X-homogenized field", "value": "on"}],
                                    value=["on"],
                                    switch=True,
                                    className="mb-2",
                                ),
                                dbc.Button(
                                    "Predict field",
                                    id="nu-predict",
                                    color="danger",
                                    className="w-100 btn-accent",
                                    disabled=True,
                                ),
                            ],
                            className="panel-card",
                        ),
                        lg=5,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H3("Results"),
                                dcc.Loading(
                                    html.Div(
                                        id="nu-figure-display",
                                        children=empty_figure_placeholder(
                                            "Training / prediction heatmaps will appear here."
                                        ),
                                    ),
                                    type="circle",
                                ),
                                html.Div(id="nu-results-output", className="mt-3"),
                                dbc.Button(
                                    "Download grid CSV",
                                    id="nu-download-csv-btn",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    className="mb-2",
                                    disabled=True,
                                ),
                                html.H4("Log"),
                                html.Div(
                                    dbc.Textarea(id="nu-log-output", rows=14, readOnly=True, value=""),
                                    className="log-box",
                                ),
                            ],
                            className="panel-card",
                        ),
                        lg=7,
                    ),
                ],
                className="g-3",
            ),
        ]
    )


def _read_dataset_configs(
    titles,
    base_dirs,
    prefixes,
    patterns,
    ignores,
    starts,
    cols,
    pitches,
    pws,
    phs,
) -> List[dict]:
    n = len(titles or [])
    configs = []
    for i in range(n):
        configs.append(
            {
                "title": (titles[i] or f"Dataset {i + 1}"),
                "base_dir": base_dirs[i] or "",
                "folder_prefix": prefixes[i] or "",
                "ignore_indices": ignores[i] or "",
                "start_idx": int(starts[i] or 0),
                "cols": int(cols[i] or 3),
                "index_pattern": patterns[i] or r"(\d+)",
                "core_pitch_mm": float(pitches[i] or 6.0),
                "plate_width_mm": float(pws[i] or 120.0),
                "plate_height_mm": float(phs[i] or 120.0),
            }
        )
    return configs


def _parse_regions(text: str):
    regions = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        region = [x, y]
        if len(parts) >= 4:
            region.extend([float(parts[2]), float(parts[3])])
        if len(parts) >= 5:
            region.append(parts[4])
        regions.append(tuple(region))
    return regions


def register_callbacks(app):
    @app.callback(
        Output("nu-dataset-store", "data"),
        Input("nu-add-dataset", "n_clicks"),
        Input({"type": "nu-ds-remove", "index": ALL}, "n_clicks"),
        State("nu-dataset-store", "data"),
        State({"type": "nu-ds-title", "index": ALL}, "value"),
        State({"type": "nu-ds-base-dir", "index": ALL}, "value"),
        State({"type": "nu-ds-prefix", "index": ALL}, "value"),
        State({"type": "nu-ds-pattern", "index": ALL}, "value"),
        State({"type": "nu-ds-ignore", "index": ALL}, "value"),
        State({"type": "nu-ds-start", "index": ALL}, "value"),
        State({"type": "nu-ds-cols", "index": ALL}, "value"),
        State({"type": "nu-ds-pitch", "index": ALL}, "value"),
        State({"type": "nu-ds-pw", "index": ALL}, "value"),
        State({"type": "nu-ds-ph", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def update_dataset_store(
        add_clicks,
        remove_clicks,
        store,
        titles,
        base_dirs,
        prefixes,
        patterns,
        ignores,
        starts,
        cols,
        pitches,
        pws,
        phs,
    ):
        configs = _read_dataset_configs(
            titles, base_dirs, prefixes, patterns, ignores, starts, cols, pitches, pws, phs
        )
        if not configs:
            configs = list(store or DEFAULT_DATASETS)

        ctx = dash.callback_context
        if not ctx.triggered:
            return configs

        triggered = ctx.triggered[0]["prop_id"]
        if triggered.startswith("nu-add-dataset"):
            configs.append(
                {
                    "title": f"Dataset {len(configs) + 1}",
                    "base_dir": "",
                    "folder_prefix": "",
                    "ignore_indices": "",
                    "start_idx": 0,
                    "cols": 3,
                    "index_pattern": r"(\d+)",
                    "core_pitch_mm": 6.0,
                    "plate_width_mm": 120.0,
                    "plate_height_mm": 120.0,
                }
            )
            return configs

        # remove button
        try:
            import json as _json

            tid = _json.loads(triggered.split(".")[0])
            if tid.get("type") == "nu-ds-remove":
                idx = tid.get("index")
                if isinstance(idx, int) and 0 <= idx < len(configs):
                    configs.pop(idx)
        except Exception:
            pass
        return configs or list(DEFAULT_DATASETS)

    @app.callback(
        Output("nu-dataset-rows", "children"),
        Input("nu-dataset-store", "data"),
    )
    def render_dataset_rows(store):
        configs = store or DEFAULT_DATASETS
        return [_dataset_row(i, cfg) for i, cfg in enumerate(configs)]

    @app.callback(
        Output("nu-log-output", "value"),
        Output("nu-results-output", "children"),
        Output("nu-figure-display", "children"),
        Output("nu-model-ready", "data"),
        Output("nu-predict", "disabled"),
        Output("nu-download-csv-btn", "disabled"),
        Output("app-status", "children", allow_duplicate=True),
        Input("nu-fit-model", "n_clicks"),
        State("nu-scaling", "value"),
        State("nu-smoothing", "value"),
        State("nu-homogenize", "value"),
        State({"type": "nu-ds-title", "index": ALL}, "value"),
        State({"type": "nu-ds-base-dir", "index": ALL}, "value"),
        State({"type": "nu-ds-prefix", "index": ALL}, "value"),
        State({"type": "nu-ds-pattern", "index": ALL}, "value"),
        State({"type": "nu-ds-ignore", "index": ALL}, "value"),
        State({"type": "nu-ds-start", "index": ALL}, "value"),
        State({"type": "nu-ds-cols", "index": ALL}, "value"),
        State({"type": "nu-ds-pitch", "index": ALL}, "value"),
        State({"type": "nu-ds-pw", "index": ALL}, "value"),
        State({"type": "nu-ds-ph", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def fit_model(
        _n,
        scaling,
        smoothing,
        homogenize,
        titles,
        base_dirs,
        prefixes,
        patterns,
        ignores,
        starts,
        cols,
        pitches,
        pws,
        phs,
    ):
        global _FITTED_MODEL, _LAST_GRID_CSV
        log_msgs = []

        def log(msg):
            log_msgs.append(str(msg))

        try:
            configs = _read_dataset_configs(
                titles, base_dirs, prefixes, patterns, ignores, starts, cols, pitches, pws, phs
            )
            if not configs:
                raise ValueError("Add at least one training dataset.")

            model = PorosityPatternModel(
                scaling=scaling or "z_score",
                smoothing=float(smoothing if smoothing is not None else 0.02),
                homogenize_x="on" in (homogenize or []),
            )

            buf = io.StringIO()
            import contextlib

            with contextlib.redirect_stdout(buf):
                for cfg in configs:
                    if not cfg["base_dir"] or not os.path.isdir(cfg["base_dir"]):
                        raise FileNotFoundError(
                            f"Dataset path not found: {cfg['base_dir']}\n"
                            "Edit the base directory to a folder that exists on this machine."
                        )
                    ignore = set(parse_int_list(cfg["ignore_indices"], []))
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
                fig = model.plot_training_field(show=False, homogenize_x=False)

            captured = buf.getvalue().strip()
            if captured:
                log(captured)

            _FITTED_MODEL = model
            _LAST_GRID_CSV = None
            try:
                from .nu_cache import save_fitted_model

                cache_path = save_fitted_model(
                    model,
                    meta={
                        "n_datasets": len(configs),
                        "titles": [c.get("title") for c in configs],
                        "scaling": scaling or "z_score",
                        "smoothing": float(smoothing if smoothing is not None else 0.02),
                        "homogenize_x": "on" in (homogenize or []),
                        "source": "nu_panel",
                    },
                )
                log(f"Saved shared NU cache -> {cache_path}")
            except Exception as cache_exc:
                log(f"Warning: could not save NU cache: {cache_exc}")
            img = fig_to_img(fig)
            result = dbc.Alert(
                f"Fitted on {len(configs)} dataset(s). Shared cache updated — Combined can reuse it.",
                color="success",
            )
            return "\n".join(log_msgs), result, img, True, False, True, "NU model fitted"
        except Exception:
            _FITTED_MODEL = None
            tb = traceback.format_exc()
            log(tb)
            return (
                "\n".join(log_msgs),
                dbc.Alert("Fit failed. See log for details.", color="danger"),
                empty_figure_placeholder("Fit failed."),
                False,
                True,
                True,
                "NU fit failed",
            )

    @app.callback(
        Output("nu-log-output", "value", allow_duplicate=True),
        Output("nu-results-output", "children", allow_duplicate=True),
        Output("nu-figure-display", "children", allow_duplicate=True),
        Output("nu-download-csv-btn", "disabled", allow_duplicate=True),
        Output("app-status", "children", allow_duplicate=True),
        Input("nu-predict", "n_clicks"),
        State("nu-plate-w", "value"),
        State("nu-plate-h", "value"),
        State("nu-resolution", "value"),
        State("nu-mean-porosity", "value"),
        State("nu-std-porosity", "value"),
        State("nu-process-label", "value"),
        State("nu-sample-regions", "value"),
        State("nu-predict-homogenize", "value"),
        prevent_initial_call=True,
    )
    def predict_field(
        _n,
        plate_w,
        plate_h,
        resolution,
        mean_porosity,
        std_porosity,
        process_label,
        regions_text,
        predict_homogenize,
    ):
        global _LAST_GRID_CSV
        log_msgs = []

        def log(msg):
            log_msgs.append(str(msg))

        if _FITTED_MODEL is None:
            return (
                "Model is not fitted yet.",
                dbc.Alert("Fit the model before predicting.", color="warning"),
                empty_figure_placeholder("No fitted model."),
                True,
                "NU predict blocked",
            )

        try:
            width = float(plate_w or 120.0)
            height = float(plate_h or 120.0)
            res = float(resolution or 1.0)
            mean_p = float(mean_porosity if mean_porosity is not None else 0.02)
            std_p = parse_float(std_porosity, None)
            regions = _parse_regions(regions_text)
            hx = "on" in (predict_homogenize or [])

            buf = io.StringIO()
            import contextlib

            with contextlib.redirect_stdout(buf):
                if regions:
                    r0 = regions[0]
                    point_pred = _FITTED_MODEL.predict(
                        x_mm=r0[0],
                        y_mm=r0[1],
                        plate_width_mm=width,
                        plate_height_mm=height,
                        mean_porosity=mean_p,
                        std_porosity=std_p,
                    )
                    print(
                        f"Point @ ({r0[0]}, {r0[1]}) mm → {point_pred:.4f}% "
                        f"(mean {mean_p:.4f}%, ratio {point_pred / mean_p:.3f}×)"
                    )

                df_grid = _FITTED_MODEL.predict_grid(
                    width_mm=width,
                    height_mm=height,
                    mean_porosity=mean_p,
                    std_porosity=std_p,
                    resolution_mm=res,
                )
                print(
                    f"Grid {width}×{height} mm @ {res} mm → {len(df_grid)} pts; "
                    f"min={df_grid.predicted_porosity_pct.min():.4f}% "
                    f"max={df_grid.predicted_porosity_pct.max():.4f}% "
                    f"mean={df_grid.predicted_porosity_pct.mean():.4f}%"
                )

                fig = _FITTED_MODEL.plot_predicted_field(
                    width_mm=width,
                    height_mm=height,
                    mean_porosity=mean_p,
                    std_porosity=std_p,
                    resolution_mm=res,
                    homogenize_x=hx,
                    sample_regions=regions or None,
                    title_extra=process_label or "",
                    show=False,
                )

            captured = buf.getvalue().strip()
            if captured:
                log(captured)

            _LAST_GRID_CSV = df_grid.to_csv(index=False)
            img = fig_to_img(fig)
            result = dbc.Alert(
                [
                    html.Div(f"Predicted field on {width:.0f}×{height:.0f} mm plate."),
                    html.Div(f"Grid points: {len(df_grid)} — use Download grid CSV to export."),
                ],
                color="success",
            )
            return "\n".join(log_msgs), result, img, False, "NU prediction complete"
        except Exception:
            tb = traceback.format_exc()
            log(tb)
            return (
                "\n".join(log_msgs),
                dbc.Alert("Prediction failed. See log.", color="danger"),
                empty_figure_placeholder("Prediction failed."),
                True,
                "NU predict failed",
            )

    @app.callback(
        Output("nu-csv-download", "data"),
        Input("nu-download-csv-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_csv(_n):
        if not _LAST_GRID_CSV:
            return dash.no_update
        return dict(content=_LAST_GRID_CSV, filename="predicted_porosity_grid.csv")
