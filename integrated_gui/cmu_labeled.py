"""Helpers to run CMU sampling with per-pore type labels."""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CMU_DIR = os.path.join(ROOT, "CMU team")
if CMU_DIR not in sys.path:
    sys.path.insert(0, CMU_DIR)


def run_cmu_labeled_samples(
    param_dict: dict,
    props: dict,
    pore_model_names: List[str],
    meltpool_model_name: str = "Rosenthal",
    totalpores: int = 5000,
    itemids: Optional[List[int]] = None,
    data_dir: str = "rawdata",
    merged_data_dir: str = "data",
) -> Tuple[pd.DataFrame, float, Dict]:
    """
    Sample CMU pores with a pore_source column (lof/keyhole/gas).
    Must be called with cwd = CMU team (or absolute merged_data_dir).
    Returns (samples, equivalent_volume_mm3, meta).
    """
    from models import get_model as get_pore_model
    from meltpool import get_model as get_meltpool_model
    from download.fetch import getdata

    from .cmu_data import resolve_cmu_data_dir

    itemids = itemids or [28152209, 26304175]
    os.makedirs(data_dir, exist_ok=True)

    poredata_file = os.path.join(merged_data_dir, "poredf.csv")
    if not os.path.exists(poredata_file):
        getdata(itemids, data_dir, datadir=merged_data_dir)
    merged_data_dir = resolve_cmu_data_dir(merged_data_dir, itemids)
    poredata_file = os.path.join(merged_data_dir, "poredf.csv")
    poredata = pd.read_csv(poredata_file)

    data_file = os.path.join(merged_data_dir, "parameters.csv")
    if os.path.exists(data_file):
        data = pd.read_csv(data_file)
        data = data.rename(
            columns={
                "V": "Velocity (mm/s)",
                "P": "Power (W)",
                "Sample": "Name",
                "SS": "Spot Size (um)",
                "H": "Hatch Spacing (um)",
                "L": "Layer Thickness (um)",
                "T": "Preheat Temperature (°C)",
                "Geo": "Geometry",
            }
        )
    else:
        data = pd.DataFrame()

    if isinstance(param_dict, str):
        with open(param_dict, "r", encoding="utf-8") as f:
            param_dict = json.load(f)
    if isinstance(props, str):
        with open(props, "r", encoding="utf-8") as f:
            props = json.load(f)

    mp_model_class = get_meltpool_model(meltpool_model_name)
    if mp_model_class is None:
        raise ValueError(f"Melt pool model '{meltpool_model_name}' not found.")
    mp_params = {"Model": mp_model_class, "Properties": props}

    selected_models = []
    for pm_name in pore_model_names:
        ModelClass = get_pore_model(pm_name)
        if ModelClass is None:
            raise ValueError(f"Pore model '{pm_name}' not found.")
        model = ModelClass(
            params=param_dict,
            data=data,
            pores=poredata,
            props=mp_params,
            poremodels=selected_models,
        )
        selected_models.append(model)

    densities = np.array([m.getPoreDensity() or 0 for m in selected_models], dtype=float)
    total_density = float(densities.sum())
    if total_density <= 0:
        raise RuntimeError("All pore densities are zero. Cannot sample pores.")

    volume = totalpores / total_density
    weights = densities / total_density

    frames = []
    for weight, model, name in zip(weights, selected_models, pore_model_names):
        n_samples = int(np.round(weight * totalpores))
        if n_samples <= 0:
            continue
        sampled_df = model.sample(n_samples).copy()
        sampled_df["pore_source"] = name
        frames.append(sampled_df)

    if not frames:
        raise RuntimeError("Empty DataFrame of pores")

    samples = pd.concat(frames, ignore_index=True)
    meta = {
        "densities": {n: float(d) for n, d in zip(pore_model_names, densities)},
        "total_density": total_density,
        "volume_mm3_equivalent": volume,
        "counts": samples["pore_source"].value_counts().to_dict(),
    }
    return samples, volume, meta
