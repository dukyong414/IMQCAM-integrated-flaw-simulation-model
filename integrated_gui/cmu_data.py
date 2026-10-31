"""Locate a readable copy of the CMU pore dataset (poredf.csv / parameters.csv).

The project lives in OneDrive; when a data file is a cloud-only placeholder that
OneDrive cannot hydrate, pandas fails with ``OSError: [Errno 22]``. In that case
the dataset is fetched from Figshare into a per-machine cache outside OneDrive.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CMU_DIR = os.path.join(ROOT, "CMU team")
DEFAULT_ITEMIDS = [28152209, 26304175]
LOCAL_CACHE = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "NASA-porosity", "cmu_data"
)
REQUIRED = ("poredf.csv", "parameters.csv")


def _readable(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.read(1)
            f.seek(-1, os.SEEK_END)
            f.read(1)
        return True
    except OSError:
        return False


def _dir_ok(data_dir: str) -> bool:
    return all(_readable(os.path.join(data_dir, name)) for name in REQUIRED)


def resolve_cmu_data_dir(preferred: str | None = None, itemids=None) -> str:
    """Return a directory whose poredf.csv and parameters.csv can be read."""
    preferred = os.path.abspath(preferred or os.path.join(CMU_DIR, "data"))
    if _dir_ok(preferred):
        return preferred
    cache_data = os.path.join(LOCAL_CACHE, "data")
    if _dir_ok(cache_data):
        return cache_data
    if CMU_DIR not in sys.path:
        sys.path.insert(0, CMU_DIR)
    from download.fetch import getdata

    print(f"CMU data in {preferred} is not readable (OneDrive placeholder?); downloading to {LOCAL_CACHE}")
    getdata(list(itemids or DEFAULT_ITEMIDS), os.path.join(LOCAL_CACHE, "rawdata"), datadir=cache_data)
    if not _dir_ok(cache_data):
        raise OSError(f"Could not obtain a readable CMU dataset (tried {preferred} and {cache_data}).")
    return cache_data
