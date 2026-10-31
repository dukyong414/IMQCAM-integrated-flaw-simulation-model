"""Shared helpers for the integrated GUI."""
from __future__ import annotations

import base64
import io
from typing import Optional

from dash import html


def fig_to_img(fig, width_px: int = 900, height_px: int = 600, dpi: int = 100) -> html.Img:
    """Encode a matplotlib Figure as a Dash html.Img (PNG data URI)."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    try:
        fig.set_size_inches(width_px / dpi, height_px / dpi)
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    finally:
        # Always close — repeated Combined finalize/refresh was leaking figures
        try:
            plt.close(fig)
        except Exception:
            pass
        try:
            del fig
        except Exception:
            pass
    return html.Img(
        src=f"data:image/png;base64,{img_b64}",
        className="result-figure",
        style={"width": "100%", "height": "auto", "objectFit": "contain"},
    )


def empty_figure_placeholder(message: str = "Run the model to see results here.") -> html.Div:
    return html.Div(message, className="figure-placeholder")


def parse_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int_list(text: str, default=None):
    if default is None:
        default = []
    if not text:
        return list(default)
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
            out.append(int(part))
    return out or list(default)
