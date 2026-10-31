"""Query CMU LoF and compare to Raptor in shared units."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CMU_DIR = os.path.join(ROOT, "CMU team")
DEFAULT_PARAMS = os.path.join(CMU_DIR, "input_parameters", "params.json")
DEFAULT_MATERIAL = os.path.join(
    CMU_DIR, "material_models", "test_ti64_rosenthal.json"
)

_ESD_COLS = (
    "Equivalent Spherical Diameter (um)",
    "ESD",
    "equivalent_spherical_diameter_um",
)


def _sphere_volumes_mm3(esd_um: np.ndarray) -> np.ndarray:
    """ESD in µm → sphere volume in mm³."""
    r_mm = np.asarray(esd_um, dtype=float) / 2000.0
    return (4.0 / 3.0) * np.pi * r_mm**3


def _esd_column(df) -> Optional[str]:
    for c in _ESD_COLS:
        if c in df.columns:
            return c
    return None


def query_cmu_lof(
    power_w: float,
    velocity_mm_s: float,
    hatch_um: float = 140.0,
    layer_um: float = 30.0,
    *,
    n_size_samples: int = 2000,
    params_path: Optional[str] = None,
    material_path: Optional[str] = None,
    meltpool_model_name: str = "Rosenthal",
) -> Dict[str, Any]:
    """
    CMU LoF prediction at a process point.

    Returns number density ρ (pores/mm³) and an estimated volume porosity
    φ = ρ · ⟨V_sphere(ESD)⟩ from sampled LoF sizes.
    """
    if CMU_DIR not in sys.path:
        sys.path.insert(0, CMU_DIR)

    from models import get_model as get_pore_model
    from meltpool import get_model as get_meltpool_model
    import pandas as pd

    params_path = params_path or DEFAULT_PARAMS
    material_path = material_path or DEFAULT_MATERIAL

    with open(params_path, "r", encoding="utf-8") as f:
        param_dict = json.load(f)
    with open(material_path, "r", encoding="utf-8") as f:
        material = json.load(f)

    param_dict = dict(param_dict)
    param_dict["Power (W)"] = float(power_w)
    param_dict["Velocity (mm/s)"] = float(velocity_mm_s)
    param_dict["Hatch Spacing (um)"] = float(hatch_um)
    param_dict["Layer Thickness (um)"] = float(layer_um)

    # Rosenthal needs material props; some paths also read process fields from Properties
    props: Dict[str, Any] = {}
    if isinstance(material, dict):
        props.update({k: v for k, v in material.items() if isinstance(v, (int, float))})
    props.update({k: v for k, v in param_dict.items() if isinstance(v, (int, float))})

    from .cmu_data import resolve_cmu_data_dir

    data_dir = resolve_cmu_data_dir(os.path.join(CMU_DIR, "data"))
    poredata_file = os.path.join(data_dir, "poredf.csv")
    parameters_file = os.path.join(data_dir, "parameters.csv")

    poredata = pd.read_csv(poredata_file)
    if os.path.isfile(parameters_file):
        data = pd.read_csv(parameters_file)
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

    prev = os.getcwd()
    try:
        os.chdir(CMU_DIR)
        mp_model_class = get_meltpool_model(meltpool_model_name)
        if mp_model_class is None:
            raise ValueError(f"Melt pool model '{meltpool_model_name}' not found.")
        mp_params = {"Model": mp_model_class, "Properties": props}

        ModelClass = get_pore_model("lof")
        if ModelClass is None:
            raise ValueError("CMU LoF model not found.")
        model = ModelClass(
            params=param_dict,
            data=data,
            pores=poredata,
            props=mp_params,
            poremodels=[],
        )
        rho = float(model.getPoreDensity() or 0.0)

        mean_esd_um = float("nan")
        mean_vol_mm3 = float("nan")
        phi = float("nan")
        try:
            samples = model.sample(int(n_size_samples))
            col = _esd_column(samples) if samples is not None and len(samples) else None
            if col:
                esd = samples[col].astype(float).to_numpy()
                esd = esd[np.isfinite(esd) & (esd > 0)]
                if esd.size:
                    vols = _sphere_volumes_mm3(esd)
                    mean_esd_um = float(np.mean(esd))
                    mean_vol_mm3 = float(np.mean(vols))
                    phi = float(rho * mean_vol_mm3)
        except Exception:
            pass
    finally:
        os.chdir(prev)

    return {
        "lof_density_per_mm3": rho,
        "mean_esd_um": mean_esd_um,
        "mean_pore_volume_mm3": mean_vol_mm3,
        "volume_porosity_fraction": phi,
        "volume_porosity_pct": (phi * 100.0) if np.isfinite(phi) else float("nan"),
        "power_w": float(power_w),
        "velocity_mm_s": float(velocity_mm_s),
        "hatch_um": float(hatch_um),
        "layer_um": float(layer_um),
    }


# Backward-compatible alias
def query_cmu_lof_density(*args, **kwargs) -> Dict[str, Any]:
    return query_cmu_lof(*args, **kwargs)


def rve_volume_mm3(rve_edge_m: float) -> float:
    edge_mm = float(rve_edge_m) * 1.0e3
    return edge_mm**3


def raptor_pore_number_density(n_defects: int, rve_edge_m: float) -> float:
    vol = rve_volume_mm3(rve_edge_m)
    if vol <= 0:
        return 0.0
    return float(n_defects) / vol


def compare_raptor_to_cmu(
    *,
    power_w: float,
    velocity_m_s: float,
    hatch_um: float,
    layer_um: float,
    rve_edge_m: float,
    raptor_porosity_pct: float,
    n_defects: int,
) -> Dict[str, Any]:
    """
    Unified CMU vs Raptor LoF comparison.

    Primary shared metric: volume porosity φ (%).
    Also reports ρ (pores/mm³) and expected pore count in the same RVE.
    """
    cmu = query_cmu_lof(
        power_w=power_w,
        velocity_mm_s=float(velocity_m_s) * 1000.0,
        hatch_um=hatch_um,
        layer_um=layer_um,
    )
    vol = rve_volume_mm3(rve_edge_m)
    rap_phi_pct = float(raptor_porosity_pct)
    rap_phi = rap_phi_pct / 100.0
    rap_rho = raptor_pore_number_density(n_defects, rve_edge_m)
    cmu_rho = float(cmu["lof_density_per_mm3"])
    cmu_phi_pct = float(cmu.get("volume_porosity_pct") or float("nan"))
    cmu_n_expected = cmu_rho * vol

    def _ratio(a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-30:
            return float("nan")
        return float(a / b)

    return {
        "cmu": cmu,
        "rve_volume_mm3": vol,
        "raptor": {
            "volume_porosity_pct": rap_phi_pct,
            "volume_porosity_fraction": rap_phi,
            "density_per_mm3": rap_rho,
            "n_defects": int(n_defects),
        },
        "cmu_n_expected_in_rve": cmu_n_expected,
        "ratio_phi_raptor_over_cmu": _ratio(rap_phi_pct, cmu_phi_pct),
        "ratio_rho_raptor_over_cmu": _ratio(rap_rho, cmu_rho),
        "ratio_count_raptor_over_cmu": _ratio(float(n_defects), cmu_n_expected),
    }
