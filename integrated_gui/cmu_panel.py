"""CMU pore-source model panel for the integrated GUI."""
from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
import traceback

import dash
from dash import ALL, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from .cmu_data import resolve_cmu_data_dir
from .utils import empty_figure_placeholder, fig_to_img, parse_int_list

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CMU_DIR = os.path.join(ROOT, "CMU team")

if CMU_DIR not in sys.path:
    sys.path.insert(0, CMU_DIR)

from models import get_model as get_pore_model  # noqa: E402
from models import list_models as list_pore_models  # noqa: E402
from meltpool import get_model as get_meltpool_model  # noqa: E402
from meltpool import list_models as list_meltpool_models  # noqa: E402
from core.execute_porosity_model import execute_porosity_model  # noqa: E402

DEFAULT_ITEMIDS = [28152209, 26304175]
EXAMPLE_PARAM_PATH = os.path.join(CMU_DIR, "input_parameters", "params.json")
MATERIAL_LIBRARY_DIR = os.path.join(CMU_DIR, "material_models")

try:
    PORE_MODELS = list_pore_models() or ["lof", "keyhole", "gas"]
except Exception:
    PORE_MODELS = ["lof", "keyhole", "gas"]

try:
    MELTPOOL_MODELS = list_meltpool_models() or ["Rosenthal"]
except Exception:
    MELTPOOL_MODELS = ["Rosenthal"]


def collect_params(selected_pores, meltpool_model):
    param_info = {}
    for pm in selected_pores or []:
        try:
            cls = get_pore_model(pm)
        except Exception:
            cls = None
        if not cls:
            continue
        try:
            params_dict = cls.getparams() if hasattr(cls, "getparams") else {}
            for pname, desc in params_dict.items():
                param_info[pname] = desc
            if hasattr(cls, "getpropnames"):
                for prop in cls.getpropnames():
                    param_info[prop["name"]] = prop.get("description", "")
        except Exception as exc:
            print(f"Pore model {pm} param collection failed: {exc}")

    try:
        cls = get_meltpool_model(meltpool_model)
    except Exception:
        cls = None
    if cls:
        try:
            params_dict = cls.getparams() if hasattr(cls, "getparams") else {}
            for pname, desc in params_dict.items():
                param_info[pname] = desc
            if hasattr(cls, "getpropnames"):
                for prop in cls.getpropnames():
                    param_info[prop["name"]] = prop.get("description", "")
        except Exception as exc:
            print(f"Meltpool model {meltpool_model} param collection failed: {exc}")
    return param_info


def generate_param_inputs(param_info, param_values=None):
    if not param_info:
        return html.P("No parameters required for the selected combination.", className="text-muted")
    rows = []
    for pname, desc in sorted(param_info.items()):
        val = param_values.get(pname) if param_values else ""
        rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(pname, className="control-label"),
                            html.Small(desc or "", className="text-muted"),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        dbc.Input(
                            type="text",
                            id={"type": "cmu-param", "name": pname},
                            placeholder="Enter value",
                            value="" if val is None else str(val),
                        ),
                        md=7,
                    ),
                ],
                className="mb-2",
            )
        )
    return rows


def list_material_files():
    if not os.path.isdir(MATERIAL_LIBRARY_DIR):
        return []
    return sorted(f for f in os.listdir(MATERIAL_LIBRARY_DIR) if f.endswith(".json"))


def _fmt_metric(x, fmt=".6g"):
    try:
        if x is None:
            return "—"
        v = float(x)
        if v != v:  # NaN
            return "—"
        return format(v, fmt)
    except Exception:
        return "—"


def _pore_type_metrics_table(metrics: dict):
    """Table of LoF / keyhole / gas metrics shown under the number-density plot."""
    if not metrics:
        return html.Div()

    dens = metrics.get("densities_per_mm3") or {}
    weights = metrics.get("weights") or {}
    counts = metrics.get("sample_counts") or {}
    mean_esd = metrics.get("mean_esd_um") or {}
    # Prefer selected model order, fall back to density keys
    names = list(dens.keys()) or ["lof", "keyhole", "gas"]

    header = html.Tr(
        [
            html.Th("Pore type"),
            html.Th("ρ (pores/mm³)"),
            html.Th("Share of ρ"),
            html.Th("Sampled N"),
            html.Th("Mean ESD (µm)"),
            html.Th("φ est. (%)"),
        ]
    )
    rows = []
    for name in names:
        rho = dens.get(name, 0.0) or 0.0
        esd = mean_esd.get(name)
        phi_pct = None
        if esd is not None:
            try:
                r_mm = float(esd) / 2000.0
                vol = (4.0 / 3.0) * 3.141592653589793 * (r_mm**3)
                phi_pct = float(rho) * vol * 100.0
            except Exception:
                phi_pct = None
        rows.append(
            html.Tr(
                [
                    html.Td(html.Strong(str(name))),
                    html.Td(_fmt_metric(rho)),
                    html.Td(_fmt_metric((weights.get(name) or 0.0) * 100.0, ".2f") + "%"),
                    html.Td(str(int(counts.get(name) or 0))),
                    html.Td(_fmt_metric(esd, ".3g")),
                    html.Td(_fmt_metric(phi_pct)),
                ]
            )
        )

    total_rho = metrics.get("total_density_per_mm3")
    vol_eq = metrics.get("equivalent_volume_mm3")
    n_tot = metrics.get("totalpores")
    caption = html.Small(
        [
            f"Total ρ = {_fmt_metric(total_rho)} pores/mm³ · "
            f"equivalent catalog volume = {_fmt_metric(vol_eq, '.4g')} mm³ · "
            f"totalpores = {n_tot}. ",
            "φ est. = ρ · ⟨sphere volume from mean ESD⟩ (approximate volume porosity).",
        ],
        className="text-muted d-block mt-2",
    )
    return html.Div(
        [
            html.H5("Pore-type metrics", className="mt-3 mb-2"),
            dbc.Table(
                [html.Thead(header), html.Tbody(rows)],
                bordered=True,
                hover=True,
                size="sm",
                className="mb-0 bg-white",
            ),
            caption,
        ]
    )


def layout():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H3("CMU — Pore Source Model"),
                                html.P(
                                    "Lack-of-fusion, keyhole, and gas porosity sampling with melt-pool thermal models.",
                                    className="text-muted mb-3",
                                ),
                                html.H4("Models"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div("Pore models", className="control-label"),
                                                dbc.Checklist(
                                                    options=[
                                                        {"label": pm.upper() if len(pm) <= 3 else pm.capitalize(), "value": pm}
                                                        for pm in PORE_MODELS
                                                    ],
                                                    value=list(PORE_MODELS),
                                                    id="cmu-pore-models",
                                                    inline=True,
                                                ),
                                            ],
                                            md=7,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div("Melt-pool model", className="control-label"),
                                                dcc.Dropdown(
                                                    options=[{"label": mp, "value": mp} for mp in MELTPOOL_MODELS],
                                                    value=MELTPOOL_MODELS[0],
                                                    id="cmu-meltpool-model",
                                                    clearable=False,
                                                ),
                                            ],
                                            md=5,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                html.H4("Run settings"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div("Total pores", className="control-label"),
                                                dbc.Input(
                                                    id="cmu-totalpores",
                                                    type="number",
                                                    value=None,
                                                    placeholder="catalog",
                                                ),
                                                html.Div(
                                                    "Blank uses every pore measured on the matched CMU coupons.",
                                                    className="small text-muted",
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div("Figshare item IDs", className="control-label"),
                                                dbc.Input(
                                                    id="cmu-itemids",
                                                    type="text",
                                                    value=",".join(map(str, DEFAULT_ITEMIDS)),
                                                ),
                                            ],
                                            md=8,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div("Raw data directory", className="control-label"),
                                                dbc.Input(id="cmu-data-dir", type="text", value="rawdata"),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div("Output folder (optional)", className="control-label"),
                                                dbc.Input(
                                                    id="cmu-output-dir",
                                                    type="text",
                                                    placeholder="auto timestamp under output/",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                html.H4("Parameters"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                "Load example case",
                                                id="cmu-load-example",
                                                color="secondary",
                                                outline=True,
                                                className="w-100",
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="cmu-material-dropdown",
                                                options=[{"label": f, "value": f} for f in list_material_files()],
                                                placeholder="Select material JSON",
                                            ),
                                            md=5,
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                "Import material",
                                                id="cmu-import-material",
                                                color="secondary",
                                                outline=True,
                                                className="w-100",
                                            ),
                                            md=3,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                dcc.Upload(
                                    id="cmu-upload-json",
                                    children=html.Div(["Drop a params JSON, or ", html.A("browse")]),
                                    className="mb-3",
                                    style={
                                        "width": "100%",
                                        "height": "52px",
                                        "lineHeight": "52px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "8px",
                                        "textAlign": "center",
                                        "color": "#5a6a7a",
                                        "background": "#fafbfc",
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="cmu-params-container"),
                                html.Small(
                                    "Fill every parameter before running.",
                                    id="cmu-param-warning",
                                    className="text-danger",
                                    style={"display": "none"},
                                ),
                                dbc.Button(
                                    "Run CMU model",
                                    id="cmu-run-model",
                                    color="primary",
                                    className="w-100 mt-3 btn-nasa",
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
                                        id="cmu-figure-display",
                                        children=empty_figure_placeholder(
                                            "Number-density plot will appear after a successful run."
                                        ),
                                    ),
                                    type="circle",
                                ),
                                html.Div(id="cmu-pore-metrics", className="mt-2"),
                                html.Div(id="cmu-results-output", className="mt-3"),
                                html.H4("Log"),
                                html.Div(
                                    dbc.Textarea(id="cmu-log-output", rows=12, readOnly=True, value=""),
                                    className="log-box",
                                ),
                            ],
                            className="panel-card",
                        ),
                        lg=7,
                    ),
                ],
                className="g-3",
            )
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("cmu-params-container", "children"),
        Input("cmu-pore-models", "value"),
        Input("cmu-meltpool-model", "value"),
        Input("cmu-upload-json", "contents"),
        Input("cmu-load-example", "n_clicks"),
        Input("cmu-import-material", "n_clicks"),
        State("cmu-material-dropdown", "value"),
        State({"type": "cmu-param", "name": ALL}, "value"),
        State({"type": "cmu-param", "name": ALL}, "id"),
    )
    def update_params(
        selected_pores,
        meltpool_model,
        uploaded_contents,
        _load_example,
        _import_material,
        material_file,
        current_values,
        current_ids,
    ):
        param_info = collect_params(selected_pores, meltpool_model)
        param_values = {}
        if current_values and current_ids:
            for val, pid in zip(current_values, current_ids):
                param_values[pid["name"]] = val

        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        if triggered == "cmu-load-example":
            try:
                with open(EXAMPLE_PARAM_PATH, "r", encoding="utf-8") as f:
                    param_values.update(json.load(f))
            except Exception as exc:
                print(f"Failed to load example params: {exc}")

        if triggered == "cmu-import-material" and material_file:
            try:
                mat_path = os.path.join(MATERIAL_LIBRARY_DIR, material_file)
                with open(mat_path, "r", encoding="utf-8") as f:
                    param_values.update(json.load(f))
            except Exception as exc:
                print(f"Failed to import material file: {exc}")

        if uploaded_contents and triggered == "cmu-upload-json":
            try:
                import base64

                parts = uploaded_contents.split(",", 1)
                if len(parts) == 2:
                    decoded = base64.b64decode(parts[1])
                    param_values.update(json.loads(decoded))
            except Exception as exc:
                print(f"Failed to parse uploaded JSON: {exc}")

        return generate_param_inputs(param_info, param_values)

    @app.callback(
        Output("cmu-run-model", "disabled"),
        Output("cmu-param-warning", "style"),
        Input({"type": "cmu-param", "name": ALL}, "value"),
    )
    def enable_run_button(param_values):
        if not param_values or any(v in [None, "", []] for v in param_values):
            return True, {"display": "block"}
        return False, {"display": "none"}

    @app.callback(
        Output("cmu-log-output", "value"),
        Output("cmu-results-output", "children"),
        Output("cmu-figure-display", "children"),
        Output("cmu-pore-metrics", "children"),
        Output("app-status", "children"),
        Input("cmu-run-model", "n_clicks"),
        State("cmu-pore-models", "value"),
        State("cmu-meltpool-model", "value"),
        State("cmu-totalpores", "value"),
        State("cmu-itemids", "value"),
        State("cmu-data-dir", "value"),
        State("cmu-output-dir", "value"),
        State({"type": "cmu-param", "name": ALL}, "value"),
        State({"type": "cmu-param", "name": ALL}, "id"),
        prevent_initial_call=True,
    )
    def run_model(
        _n_clicks,
        pore_models_selected,
        meltpool_model,
        totalpores,
        itemids_str,
        data_dir,
        output_dir,
        param_values,
        param_ids,
    ):
        log_msgs = []

        def log(msg):
            log_msgs.append(str(msg))

        prev_cwd = os.getcwd()
        metrics_ui = html.Div()
        try:
            os.chdir(CMU_DIR)
            itemids = parse_int_list(itemids_str, DEFAULT_ITEMIDS)

            param_dict = {}
            for val, pid in zip(param_values or [], param_ids or []):
                pname = pid["name"]
                if val is None or val == "":
                    param_dict[pname] = None
                else:
                    try:
                        v = float(val)
                        param_dict[pname] = int(v) if v.is_integer() else v
                    except Exception:
                        param_dict[pname] = val

            output_dir = (output_dir or "").strip() or os.path.join(
                "output", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(CMU_DIR, output_dir)
            os.makedirs(output_dir, exist_ok=True)

            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json", encoding="utf-8") as param_file:
                json.dump(param_dict, param_file, indent=4)
                param_file_path = param_file.name

            props_dict = {k: v for k, v in param_dict.items() if isinstance(v, (int, float))}
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json", encoding="utf-8") as props_file:
                json.dump(props_dict, props_file, indent=4)
                props_file_path = props_file.name

            n_req = None
            if totalpores not in (None, ""):
                n_req = int(totalpores)
                if n_req <= 0:
                    n_req = None
            log(f"Running from {CMU_DIR}")
            log(f"Pore models: {pore_models_selected}")
            log(f"Melt pool: {meltpool_model}")
            log("Total pores: catalog" if n_req is None else f"Total pores: {n_req}")

            results = execute_porosity_model(
                itemids=itemids,
                data_dir=data_dir or "rawdata",
                output_dir=output_dir,
                pore_model_names=pore_models_selected or PORE_MODELS,
                meltpool_model_name=meltpool_model or MELTPOOL_MODELS[0],
                param_dict=param_file_path,
                props=props_file_path,
                totalpores=n_req,
                merged_data_dir=resolve_cmu_data_dir(os.path.join(CMU_DIR, "data"), itemids),
                return_figure=True,
            )

            log(f"Model run complete. Output: {results['output_dir']}")
            log(f"Plot saved to: {results['plot_path']}")
            metrics = results.get("metrics") or {}
            dens = metrics.get("densities_per_mm3") or {}
            for name, rho in dens.items():
                log(f"  {name}: ρ = {rho:.6g} pores/mm³")
            log(f"  total ρ = {metrics.get('total_density_per_mm3', float('nan')):.6g} pores/mm³")
            n_used = int(metrics.get("totalpores") or 0)
            if n_req is None:
                log(f"  pores used: {n_used:,} from the matched coupons")

            img_component = fig_to_img(results["figure"])
            metrics_ui = _pore_type_metrics_table(metrics)
            result_msg = dbc.Alert(
                [
                    html.Div(f"Output directory: {results['output_dir']}"),
                    html.Div(f"Plot path: {results['plot_path']}"),
                    html.Div(
                        f"Total ρ = {_fmt_metric(metrics.get('total_density_per_mm3'))} pores/mm³ "
                        f"across {len(dens)} pore type(s)."
                    ),
                ],
                color="success",
            )
            status = "CMU run complete"
        except Exception:
            tb = traceback.format_exc()
            log(f"Runtime error:\n{tb}")
            result_msg = dbc.Alert("Runtime error occurred. See log for details.", color="danger")
            img_component = empty_figure_placeholder("No figure available due to runtime error.")
            metrics_ui = html.Div()
            status = "CMU run failed"
        finally:
            os.chdir(prev_cwd)

        return "\n".join(log_msgs), result_msg, img_component, metrics_ui, status
