import os
import sys
import json
import base64
import tempfile
import datetime
import traceback
import io
import pandas as pd
from flask import send_from_directory
import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# --- Project imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import get_model as get_pore_model
from meltpool import get_model as get_meltpool_model
from download.fetch import getdata
from core.execute_porosity_model import execute_porosity_model

# --- Defaults ---
default_itemids = [28152209, 26304175]
default_data_dir = 'rawdata'
merged_data_dir = 'data'
totalpores_default = None

example_param_path = os.path.join("input_parameters", "params.json")
material_library_dir = os.path.join("material_models")

# --- Available models ---
pore_models = get_pore_model.list_models() if hasattr(get_pore_model, "list_models") else ["lof", "keyhole", "gas"]
meltpool_models = get_meltpool_model.list_models() if hasattr(get_meltpool_model, "list_models") else ["Rosenthal"]

# --- Initialize Dash ---
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Porosity Model V0.1"
# --- Collect parameters statically ---
def collect_params(selected_pores, meltpool_model):
    param_info = {}
    for pm in selected_pores:
        cls = get_pore_model(pm)
        if cls:
            try:
                params_dict = cls.getparams() if hasattr(cls, "getparams") else {}
                for pname, desc in params_dict.items():
                    param_info[pname] = desc
                if hasattr(cls, "getpropnames"):
                    for prop in cls.getpropnames():
                        param_info[prop["name"]] = prop.get("description", "")
            except Exception as e:
                print(f"Pore model {pm} param collection failed: {e}")

    cls = get_meltpool_model(meltpool_model)
    if cls:
        try:
            params_dict = cls.getparams() if hasattr(cls, "getparams") else {}
            for pname, desc in params_dict.items():
                param_info[pname] = desc
            if hasattr(cls, "getpropnames"):
                for prop in cls.getpropnames():
                    param_info[prop["name"]] = prop.get("description", "")
        except Exception as e:
            print(f"Meltpool model {meltpool_model} param collection failed: {e}")

    return param_info

# --- Generate input fields ---
def generate_param_inputs(param_info, param_values=None):
    if not param_info:
        return html.P("No parameters required for the selected combination.")
    components = []
    for pname, desc in sorted(param_info.items()):
        val = param_values.get(pname) if param_values else ""
        components.append(
            dbc.Row([
                dbc.Col(html.Label([pname, html.Br(), html.Small(desc, className="text-muted")]), md=5),
                dbc.Col(
                    dbc.Input(
                        type="text",
                        id={"type": "param", "name": pname},
                        placeholder="Enter value",
                        value=val
                    ),
                    md=5
                ),
            ], className="my-1")
        )
    return components

# --- Load available material files ---
def list_material_files():
    if not os.path.isdir(material_library_dir):
        return []
    return [f for f in os.listdir(material_library_dir) if f.endswith(".json")]

# --- Layout ---
app.layout = dbc.Container([
    html.H2("Porosity Model V0.1", className="my-3 text-center"),
    html.H5("Developed by EMIT Lab (CMU)", className="text-center text-muted"),
    html.H5("Justin P. Miner and Sneha Prabha Narra", className="text-center text-muted"),

    dbc.Row([
        dbc.Col([
            html.H5("Pore Models"),
            dbc.Checklist(
                options=[{"label": pm.capitalize(), "value": pm} for pm in pore_models],
                value=pore_models,
                id="pore-models"
            ),
        ], md=6),

        dbc.Col([
            html.H5("Meltpool Model"),
            dcc.Dropdown(
                options=[{"label": mp, "value": mp} for mp in meltpool_models],
                value=meltpool_models[0],
                id="meltpool-model"
            ),
        ], md=6),
    ], className="my-2"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Total pores (blank = catalog)"),
            dbc.Input(id="totalpores", type="number", value=totalpores_default, placeholder="catalog")
        ], md=3),

        dbc.Col([
            dbc.Label("Item IDs (comma-separated)"),
            dbc.Input(id="itemids", type="text", value=",".join(map(str, default_itemids)))
        ], md=4),

        dbc.Col([
            dbc.Label("Data Directory"),
            dbc.Input(id="data-dir", type="text", value=default_data_dir)
        ], md=3),

        dbc.Col([
            dbc.Label("Output Folder (optional)"),
            dbc.Input(id="output-dir", type="text", placeholder="e.g. output/")
        ], md=2),
    ], className="my-2"),

    html.Hr(),
    html.H5("Upload JSON Parameters"),
    dcc.Upload(
        id="upload-json",
        children=html.Div(["Drag and Drop or ", html.A("Select File")]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin-bottom': '10px'
        },
        multiple=False
    ),

    dbc.Row([
        dbc.Col(dbc.Button("Load Example Case", id="load-example", color="secondary", className="my-1"), md=3),
        dbc.Col([
            dcc.Dropdown(
                id="material-dropdown",
                options=[{"label": f, "value": f} for f in list_material_files()],
                placeholder="Select Material File"
            ),
        ], md=6),
        dbc.Col(dbc.Button("Import Material Library", id="import-material", color="info", className="my-1"), md=3),
    ], className="my-2"),

    html.H5("Model Parameters"),
    html.Div(id="params-container"),
    html.Div(
        dbc.Button("Run Model", id="run-model", color="primary", className="my-3", disabled=True),
        className="text-center"
    ),
    html.Small("Please fill in all model parameters before running.", id="param-warning", className="text-danger", style={"display": "none"}),

    html.Hr(),
    html.H5("Logs"),
    dbc.Textarea(id="log-output", rows=10, style={"width": "100%"}, readOnly=True),
    html.Div(id="results-output", className="mt-3"),
    dcc.Loading(html.Div(id="figure-display", style={"height": "600px"}), type="circle"),
    html.Div(
        html.Small(
            "Disclaimer: This software is provided for research purposes only. "
            "The authors are not responsible for any decisions made based on the outputs.",
            className="text-muted",
        ),
        className="text-center my-3"
    )
], fluid=True)

# --- Callbacks ---
@app.callback(
    Output("params-container", "children"),
    Input("pore-models", "value"),
    Input("meltpool-model", "value"),
    Input("upload-json", "contents"),
    Input("load-example", "n_clicks"),
    Input("import-material", "n_clicks"),
    State("material-dropdown", "value"),
    State({"type": "param", "name": dash.ALL}, "value"),
    State({"type": "param", "name": dash.ALL}, "id"),
)
def update_params(selected_pores, meltpool_model, uploaded_contents, load_example, import_material, material_file, current_values, current_ids):
    param_info = collect_params(selected_pores, meltpool_model)
    param_values = {}

    if current_values and current_ids:
        for val, pid in zip(current_values, current_ids):
            param_values[pid["name"]] = val

    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered == "load-example":
        try:
            with open(example_param_path, "r") as f:
                example_data = json.load(f)
            param_values.update(example_data)
        except Exception as e:
            print(f"Failed to load example params: {e}")

    if triggered == "import-material" and material_file:
        try:
            mat_path = os.path.join(material_library_dir, material_file)
            with open(mat_path, "r") as f:
                mat_data = json.load(f)
            param_values.update(mat_data)
        except Exception as e:
            print(f"Failed to import material file: {e}")

    if uploaded_contents:
        try:
            parts = uploaded_contents.split(',', 1)
            if len(parts) == 2:
                _, content_string = parts
                decoded = base64.b64decode(content_string)
                uploaded_data = json.loads(decoded)
                param_values.update(uploaded_data)
        except Exception as e:
            print(f"Failed to parse uploaded JSON: {e}")

    return generate_param_inputs(param_info, param_values)

@app.callback(
    Output("run-model", "disabled"),
    Output("param-warning", "style"),
    Input({"type": "param", "name": dash.ALL}, "value"),
)
def enable_run_button(param_values):
    if not param_values or any(v in [None, "", []] for v in param_values):
        return True, {"display": "block"}
    return False, {"display": "none"}

# --- Run model ---
@app.callback(
    Output("log-output", "value"),
    Output("results-output", "children"),
    Output("figure-display", "children"),  # use children instead of figure
    Input("run-model", "n_clicks"),
    State("pore-models", "value"),
    State("meltpool-model", "value"),
    State("totalpores", "value"),
    State("itemids", "value"),
    State("data-dir", "value"),
    State("output-dir", "value"),
    State({"type": "param", "name": dash.ALL}, "value"),
    State({"type": "param", "name": dash.ALL}, "id"),
    prevent_initial_call=True
)
def run_model(n_clicks, pore_models_selected, meltpool_model, totalpores, itemids_str,
              data_dir, output_dir, param_values, param_ids):
    log_msgs = []

    def log(msg):
        log_msgs.append(msg)

    try:
        itemids = [int(x.strip()) for x in itemids_str.split(",") if x.strip().isdigit()]
        if not itemids:
            raise ValueError
    except Exception:
        itemids = default_itemids
        log(f"Invalid Item IDs, defaulting to {itemids}")

    param_dict = {}
    for val, pid in zip(param_values, param_ids):
        pname = pid["name"]
        if val is None or val == "":
            param_dict[pname] = None
        else:
            try:
                v = float(val)
                param_dict[pname] = int(v) if v.is_integer() else v
            except Exception:
                param_dict[pname] = val

    output_dir = (output_dir or "").strip() or os.path.join("output", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as param_file:
        json.dump(param_dict, param_file, indent=4)
        param_file_path = param_file.name

    props_dict = {k: v for k, v in param_dict.items() if isinstance(v, (int, float))}
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as props_file:
        json.dump(props_dict, props_file, indent=4)
        props_file_path = props_file.name

    try:
        results = execute_porosity_model(
            itemids=itemids,
            data_dir=data_dir,
            output_dir=output_dir,
            pore_model_names=pore_models_selected,
            meltpool_model_name=meltpool_model,
            param_dict=param_file_path,
            props=props_file_path,
            totalpores=totalpores,
            merged_data_dir=merged_data_dir,
            return_figure=True
        )

        log(f"Model run complete. Output: {results['output_dir']}")
        log(f"Plot saved to: {results['plot_path']}")

        # --- Render figure as PNG ---
        buf = io.BytesIO()
        px_width = 900
        px_height = 600
        dpi = 100

        fig = results["figure"]
        fig.set_size_inches(px_width/dpi, px_height/dpi)
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        img_component = html.Img(
            src=f"data:image/png;base64,{img_b64}",
            style={"width": "100%", "height": "600px", "object-fit": "contain"}
        )

        result_msg = dbc.Alert([
            html.P(f"Output directory: {results['output_dir']}"),
            html.P(f"Plot path: {results['plot_path']}")
        ], color="success")

    except Exception:
        tb = traceback.format_exc()
        log(f"Runtime error:\n{tb}")
        result_msg = dbc.Alert("Runtime error occurred.", color="danger")
        img_component = html.Div("No figure available due to runtime error.", style={"color": "red"})

    return "\n".join(log_msgs), result_msg, img_component

def launch_gui(host="0.0.0.0", port=8050, debug=False):
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    launch_gui()
