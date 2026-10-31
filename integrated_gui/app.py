"""Integrated Dash application: Welcome → Combined 3D → CMU → NU → Raptor."""
from __future__ import annotations

import os

from .headless import configure_headless_plotting

configure_headless_plotting()

from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc

from . import combined_panel, cmu_panel, nu_panel, raptor_panel, welcome_panel
from .sys_info import format_sys_chip, host_snapshot

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets")
# Authoritative Welcome hero image (synced into Dash assets on launch)
MAIN_MENU_IMAGE_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "Main menu image.png",
    )
)
MAIN_MENU_ASSET = os.path.join(ASSETS_PATH, "main_menu.png")


def _sync_main_menu_image() -> None:
    """Copy ``Main menu image.png`` from the NASA project folder into assets."""
    try:
        if not os.path.isfile(MAIN_MENU_IMAGE_SRC):
            return
        os.makedirs(ASSETS_PATH, exist_ok=True)
        # Refresh when the source is newer or the asset is missing
        if (not os.path.isfile(MAIN_MENU_ASSET)) or (
            os.path.getmtime(MAIN_MENU_IMAGE_SRC) > os.path.getmtime(MAIN_MENU_ASSET)
        ):
            import shutil

            shutil.copy2(MAIN_MENU_IMAGE_SRC, MAIN_MENU_ASSET)
    except Exception:
        pass


def create_app() -> Dash:
    _sync_main_menu_image()
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=ASSETS_PATH,
        suppress_callback_exceptions=True,
        title="Process defect modeling",
    )

    app.layout = html.Div(
        [
            dcc.Interval(id="sys-info-tick", interval=2500, n_intervals=0),
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="/assets/imqcam_logo.png",
                                alt="IMQCAM logo",
                                className="app-logo",
                            ),
                            html.Div(
                                [
                                    html.P("NASA ULI / STRI", className="brand-kicker"),
                                    html.H1("Process porosity modeling"),
                                    html.P(
                                        "Welcome · Combined · CMU · NU · Raptor",
                                        className="subtitle",
                                    ),
                                ]
                            ),
                        ],
                        className="app-header-brand",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "CLI help",
                                id="header-cli-btn",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="me-2",
                            ),
                            html.Div(id="app-status", className="status-chip", children="Ready"),
                        ],
                        className="header-actions",
                    ),
                ],
                className="app-header",
            ),
            html.Div(
                id="sys-info-chip",
                className="sys-info-chip",
                children=format_sys_chip(host_snapshot()),
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("CLI quick reference")),
                    dbc.ModalBody(dcc.Markdown(welcome_panel.CLI_HELP_MD)),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="header-cli-close", className="ms-auto")
                    ),
                ],
                id="header-cli-modal",
                size="lg",
                is_open=False,
                scrollable=True,
            ),
            html.Div(
                [
                    html.Button(
                        [
                            html.Span("Welcome"),
                            html.Span("Process overview", className="tab-desc"),
                        ],
                        id="tab-welcome-btn",
                        className="tab-btn active",
                        n_clicks=0,
                    ),
                    html.Button(
                        [
                            html.Span("Combined — 3D"),
                            html.Span("ρ×V · NU field · scan/edge pores", className="tab-desc"),
                        ],
                        id="tab-comb-btn",
                        className="tab-btn",
                        n_clicks=0,
                    ),
                    html.Button(
                        [
                            html.Span("CMU — Pore Source"),
                            html.Span("LoF · keyhole · gas sampling", className="tab-desc"),
                        ],
                        id="tab-cmu-btn",
                        className="tab-btn",
                        n_clicks=0,
                    ),
                    html.Button(
                        [
                            html.Span("NU — Spatial Field"),
                            html.Span("RBF plate pattern", className="tab-desc"),
                        ],
                        id="tab-nu-btn",
                        className="tab-btn",
                        n_clicks=0,
                    ),
                    html.Button(
                        [
                            html.Span("Raptor — LoF"),
                            html.Span("ORNL melt-pool RVE", className="tab-desc"),
                        ],
                        id="tab-rap-btn",
                        className="tab-btn",
                        n_clicks=0,
                    ),
                ],
                className="tab-bar",
            ),
            html.Div(id="welcome-panel-wrap", children=welcome_panel.layout(), style={"display": "block"}),
            html.Div(id="comb-panel-wrap", children=combined_panel.layout(), style={"display": "none"}),
            html.Div(id="cmu-panel-wrap", children=cmu_panel.layout(), style={"display": "none"}),
            html.Div(id="nu-panel-wrap", children=nu_panel.layout(), style={"display": "none"}),
            html.Div(id="rap-panel-wrap", children=raptor_panel.layout(), style={"display": "none"}),
            html.P(
                "Research software only. Outputs are for scientific exploration and are not certified for flight decisions.",
                className="footer-note",
            ),
        ],
        className="app-shell",
    )

    @app.callback(
        Output("welcome-panel-wrap", "style"),
        Output("comb-panel-wrap", "style"),
        Output("cmu-panel-wrap", "style"),
        Output("nu-panel-wrap", "style"),
        Output("rap-panel-wrap", "style"),
        Output("tab-welcome-btn", "className"),
        Output("tab-comb-btn", "className"),
        Output("tab-cmu-btn", "className"),
        Output("tab-nu-btn", "className"),
        Output("tab-rap-btn", "className"),
        Output("app-status", "children", allow_duplicate=True),
        Input("tab-welcome-btn", "n_clicks"),
        Input("tab-comb-btn", "n_clicks"),
        Input("tab-cmu-btn", "n_clicks"),
        Input("tab-nu-btn", "n_clicks"),
        Input("tab-rap-btn", "n_clicks"),
        Input({"type": "welcome-nav", "mod": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def switch_tab(_w, _comb, _cmu, _nu, _rap, _welcome_nav):
        trig = callback_context.triggered_id
        hide, show = {"display": "none"}, {"display": "block"}
        active, idle = "tab-btn active", "tab-btn"

        mod = None
        if isinstance(trig, dict) and trig.get("type") == "welcome-nav":
            mod = trig.get("mod")
        elif trig == "tab-comb-btn":
            mod = "comb"
        elif trig == "tab-cmu-btn":
            mod = "cmu"
        elif trig == "tab-nu-btn":
            mod = "nu"
        elif trig == "tab-rap-btn":
            mod = "rap"
        elif trig == "tab-welcome-btn":
            mod = "welcome"
        else:
            mod = "welcome"

        # Ignore pattern-matching fires with no real click (all None / 0)
        if isinstance(trig, dict) and trig.get("type") == "welcome-nav":
            if not any(_welcome_nav or []):
                return no_update

        if mod == "comb":
            return (
                hide, show, hide, hide, hide,
                idle, active, idle, idle, idle,
                "Combined 3D workspace",
            )
        if mod == "cmu":
            return (
                hide, hide, show, hide, hide,
                idle, idle, active, idle, idle,
                "CMU workspace",
            )
        if mod == "nu":
            return (
                hide, hide, hide, show, hide,
                idle, idle, idle, active, idle,
                "NU spatial field",
            )
        if mod == "rap":
            return (
                hide, hide, hide, hide, show,
                idle, idle, idle, idle, active,
                "Raptor LoF workspace",
            )
        return (
            show, hide, hide, hide, hide,
            active, idle, idle, idle, idle,
            "Welcome",
        )

    @app.callback(
        Output("header-cli-modal", "is_open"),
        Input("header-cli-btn", "n_clicks"),
        Input("header-cli-close", "n_clicks"),
        State("header-cli-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_header_cli(n_open, n_close, is_open):
        return not is_open

    @app.callback(
        Output("sys-info-chip", "children"),
        Input("sys-info-tick", "n_intervals"),
    )
    def refresh_sys_info(_n):
        return format_sys_chip(host_snapshot())

    welcome_panel.register_callbacks(app)
    combined_panel.register_callbacks(app)
    cmu_panel.register_callbacks(app)
    nu_panel.register_callbacks(app)
    raptor_panel.register_callbacks(app)
    return app


def launch(host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    from .console_guard import install_windows_ctrl_guard

    if install_windows_ctrl_guard():
        print(
            "Console Ctrl+C from 3D viewers is ignored so the server stays up. "
            "Close this window to stop the GUI."
        )
    app = create_app()
    print(f"Opening Process defect modeling GUI at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
