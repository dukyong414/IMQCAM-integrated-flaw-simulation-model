"""Welcome / information landing page for first-time users."""
from __future__ import annotations

from dash import html, dcc, Input, Output, State
import dash
import dash_bootstrap_components as dbc

from .nu_cache import cache_status, run_prefit_subprocess


CLI_HELP_MD = """
### How to run the CLI

From the project folder (where `run_cli.bat` lives):

**Open the Dash website (default)**
```
.\\run_cli.bat
```
or
```
.\\run_cli.bat gui
```

**Pre-fit NU model (once, shared cache)**
```
.\\run_cli.bat nu fit --outdir out\\nu1
```

.**Combined 3D plate run (headless)**
```
.\\run_cli.bat combined --outdir out\\run1 --n-strips 4
```

**CMU only**
```
.\\run_cli.bat cmu --outdir out\\cmu1 --totalpores 3000
```

**NU predict**
```
.\\run_cli.bat nu predict --outdir out\\nu1 --mean-porosity 0.02
```

**Help**
```
.\\run_cli.bat --help
.\\run_cli.bat combined --help
```

Use the project virtual environment Python if needed:
```
.\\.venv\\Scripts\\python.exe -m integrated_cli gui
```
"""


def _nu_status_alert():
    st = cache_status()
    color = "success" if st["ready"] else "warning"
    return dbc.Alert(st["message"], color=color, className="mb-0 py-2", id="welcome-nu-status-inner")


def _module_node(mod: str, title: str, subtitle: str, *, hub: bool = False, accent: str = ""):
    classes = "module-node"
    if hub:
        classes += " module-node-hub"
    if accent:
        classes += f" module-node-{accent}"
    return html.Button(
        [
            html.Span(title, className="module-node-title"),
            html.Span(subtitle, className="module-node-sub"),
            html.Span("Open →", className="module-node-cta"),
        ],
        id={"type": "welcome-nav", "mod": mod},
        n_clicks=0,
        className=classes,
        type="button",
    )


def _module_diagram():
    """Clickable flowchart: CMU + NU → Combined; Raptor modulates Combined scan/LoF."""
    return html.Div(
        [
            html.H4("Module map"),
            html.P(
                "Click any box to open that panel. CMU density and NU spatial field feed Combined 3D. "
                "Raptor LoF is not a separate product path — it changes how Combined builds LoF "
                "and scan-related sizing (cuboid mode).",
                className="text-muted mb-3",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            _module_node(
                                "cmu",
                                "CMU — Pore Source",
                                "ρ · LoF / keyhole / gas",
                                accent="cmu",
                            ),
                            _module_node(
                                "nu",
                                "NU — Spatial Field",
                                "RBF plate pattern",
                                accent="nu",
                            ),
                        ],
                        className="module-map-row module-map-sources",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(className="module-arrow-line module-arrow-left"),
                                    html.Div(className="module-arrow-tip"),
                                    html.Div(className="module-arrow-line module-arrow-right"),
                                ],
                                className="module-arrow-fork",
                            ),
                            html.Div(className="module-arrow-down-center", children="▼"),
                        ],
                        className="module-map-merge",
                        **{"aria-hidden": "true"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    _module_node(
                                        "comb",
                                        "Combined — 3D",
                                        "ρ×V × NU · plate / STL pores",
                                        hub=True,
                                        accent="comb",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "▲ modulates LoF / scan in Combined",
                                                className="module-map-side-label",
                                            ),
                                            _module_node(
                                                "rap",
                                                "Raptor — LoF",
                                                "ORNL RVE → Combined LoF φ & sizes",
                                                accent="rap",
                                            ),
                                        ],
                                        className="module-map-raptor-side",
                                    ),
                                ],
                                className="module-map-hub-with-raptor",
                            ),
                        ],
                        className="module-map-row module-map-hub",
                    ),
                ],
                className="module-map",
            ),
        ],
        className="mb-4",
    )


def layout():
    return html.Div(
        [
            dcc.Interval(id="welcome-nu-status-tick", interval=5000, n_intervals=0),
            html.Div(
                [
                    html.H3("Welcome"),
                    html.Img(
                        src="/assets/main_menu.png",
                        alt="Integrated porosity modeling overview",
                        className="welcome-hero-img",
                    ),
                    html.P(
                        "This app integrates CMU pore-source models and the NU spatial porosity field "
                        "into one plate-scale Combined 3D workflow. ORNL Raptor LoF feeds Combined "
                        "(cuboid scan / LoF sizing), not a separate end-to-end path.",
                        className="lead-text",
                    ),
                    _module_diagram(),
                    html.H4("1. Pre-fit NU (do this once)"),
                    html.P(
                        "NU spatial fitting is slow. Run it once here; Combined then loads the saved "
                        "cache instead of re-fitting on every Generate.",
                        className="mb-2",
                    ),
                    html.Div(id="welcome-nu-status", children=_nu_status_alert(), className="mb-2"),
                    dbc.Button(
                        "Pre-fit NU model (once)",
                        id="welcome-prefit-nu",
                        color="primary",
                        className="btn-nasa me-2",
                    ),
                    html.Span(id="welcome-prefit-status", className="text-muted small"),
                    html.Hr(className="my-4"),
                    html.H4("Recommended path"),
                    html.Ol(
                        [
                            html.Li("Pre-fit NU above (required for Combined placement)."),
                            html.Li("Open Combined — 3D and generate the plate pore field (ρ×V × NU)."),
                            html.Li(
                                "Optionally open CMU or NU alone; use Raptor to inspect the LoF RVE "
                                "that Combined can pull into cuboid scan / LoF sizing."
                            ),
                            html.Li("Use CLI help if you need batch / headless runs."),
                        ]
                    ),
                    html.H4("Process overview"),
                    html.Ol(
                        [
                            html.Li(
                                [
                                    html.Strong("1. Pre-fit the NU spatial model "),
                                    "(once) so Combined can load the shared cache.",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("2. Choose geometry: "),
                                    "coupon cubes 4×(19×3), a centered cube/cuboid, or an imported STL.",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("3. CMU density ρ "),
                                    "(pores/mm³ from LoF / keyhole / gas) × build volume → base pore count.",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("4. NU variation: "),
                                    "scale regions by NU_local / mean(NU), then randomly disperse pores "
                                    "inside each build / STL.",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("5. Optional scan vectors + edge pores "),
                                    "at hatch start/end (on by default in Combined).",
                                ]
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Learn about the CLI",
                                id="welcome-cli-btn",
                                color="secondary",
                                outline=True,
                            ),
                        ],
                        className="mt-3 mb-2",
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle("CLI quick reference")),
                            dbc.ModalBody(dcc.Markdown(CLI_HELP_MD)),
                            dbc.ModalFooter(
                                dbc.Button("Close", id="welcome-cli-close", className="ms-auto")
                            ),
                        ],
                        id="welcome-cli-modal",
                        size="lg",
                        is_open=False,
                        scrollable=True,
                    ),
                    html.P(
                        "Research software only. Outputs are for scientific exploration.",
                        className="text-muted mt-4 mb-0",
                    ),
                ],
                className="panel-card welcome-card",
            ),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("welcome-cli-modal", "is_open"),
        Input("welcome-cli-btn", "n_clicks"),
        Input("welcome-cli-close", "n_clicks"),
        State("welcome-cli-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_cli_modal(n_open, n_close, is_open):
        return not is_open

    @app.callback(
        Output("welcome-nu-status", "children"),
        Input("welcome-nu-status-tick", "n_intervals"),
    )
    def refresh_nu_status(_tick):
        return _nu_status_alert()

    @app.callback(
        Output("welcome-prefit-status", "children"),
        Output("welcome-nu-status", "children", allow_duplicate=True),
        Output("app-status", "children", allow_duplicate=True),
        Input("welcome-prefit-nu", "n_clicks"),
        running=[
            (Output("welcome-prefit-nu", "disabled"), True, False),
            (
                Output("welcome-prefit-status", "children"),
                "Fitting NU (1–3 min)… keep this tab open.",
                dash.no_update,
            ),
            (
                Output("app-status", "children"),
                "Pre-fitting NU…",
                dash.no_update,
            ),
        ],
        prevent_initial_call=True,
    )
    def run_welcome_prefit(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        status = run_prefit_subprocess()
        if status.get("ok"):
            msg = status.get("cache", {}).get("message") or "NU cache ready."
            return f"Done — {msg}", _nu_status_alert(), "NU pre-fit complete"
        err = (status.get("log") or status.get("error") or "Pre-fit failed")[-500:]
        return f"Failed — see details: {err}", _nu_status_alert(), "NU pre-fit failed"
