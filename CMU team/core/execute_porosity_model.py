import os
import json
import pandas as pd
import numpy as np
import traceback
from models import get_model as get_pore_model
from meltpool import get_model as get_meltpool_model
from download.fetch import getdata
from visualization import plot_number_density

def _catalog_pores(model):
    """Every measured pore on the coupons that enter this model's density.

    Returns (pore rows, scanned volume in mm³). An empty frame means the
    model has no matched coupon.
    """
    part = getattr(model, "partition", None)
    data = getattr(model, "data", None)
    pores = getattr(model, "poredf", None)
    if part is None or data is None or pores is None:
        return pd.DataFrame(), 0.0
    if len(part) == 0 or len(pores) == 0 or "Partition" not in part.columns:
        return pd.DataFrame(), 0.0

    names = []
    volume = 0.0
    seen = set()
    for _, row in part.iterrows():
        if float(row.get("Partition") or 0) == 0:
            continue
        key = (row.get("Power (W)"), row.get("Velocity (mm/s)"))
        if key in seen:
            continue
        seen.add(key)
        scans = data[
            (data["Power (W)"] == key[0]) & (data["Velocity (mm/s)"] == key[1])
        ]
        if scans.empty:
            continue
        if "Volume (mm^3)" in scans.columns:
            volume += float(scans["Volume (mm^3)"].sum())
        names.extend(scans["Name"].dropna().astype(str).tolist())
    if not names:
        return pd.DataFrame(), 0.0
    matched = pores[pores["Name"].astype(str).isin(set(names))].copy()
    return matched, volume


def execute_porosity_model(
    itemids,
    data_dir,
    output_dir,
    pore_model_names,
    meltpool_model_name,
    param_dict,
    props,
    totalpores=None,
    merged_data_dir='data',
    return_figure = False
):
    # Create dirs if missing
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load or download pore data
    poredata_file = os.path.join(merged_data_dir, 'poredf.csv')
    if not os.path.exists(poredata_file):
        print(f"Pore data missing at {poredata_file}. Downloading...")
        getdata(itemids, data_dir, datadir=merged_data_dir)
    poredata = pd.read_csv(poredata_file)

    # Load parameters CSV if exists
    data_file = os.path.join(merged_data_dir, 'parameters.csv')
    if os.path.exists(data_file):
        data = pd.read_csv(data_file)
        data = data.rename(columns={
            'V': 'Velocity (mm/s)', 
            'P': 'Power (W)', 
            'Sample': 'Name', 
            'SS': 'Spot Size (um)',
            'H': "Hatch Spacing (um)",
            'L': "Layer Thickness (um)",
            'T': "Preheat Temperature (°C)",
            'Geo': 'Geometry',
        })
    else:
        data = pd.DataFrame()

    # Ensure param_dict is a dictionary
    if isinstance(param_dict, str):
        with open(param_dict, 'r') as f:
            param_dict = json.load(f)
    elif not isinstance(param_dict, dict):
        raise TypeError("param_dict must be a dict or path to JSON file")

    # Ensure props is a dictionary
    if isinstance(props, str):
        with open(props, 'r') as f:
            props = json.load(f)
    elif not isinstance(props, dict):
        raise TypeError("props must be a dict or path to JSON file")

    # Melt pool model
    mp_model_class = get_meltpool_model(meltpool_model_name)
    if mp_model_class is None:
        raise ValueError(f"Melt pool model '{meltpool_model_name}' not found.")
    mp_params = {'Model': mp_model_class, 'Properties': props}

    # Instantiate pore models
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
            poremodels=selected_models
        )
        selected_models.append(model)

    if not selected_models:
        raise RuntimeError("No pore models instantiated.")

    # Sampling weighted by densities
    densities = np.array([m.getPoreDensity() or 0 for m in selected_models], dtype=float)
    total_density = float(densities.sum())
    if total_density == 0:
        raise RuntimeError("All pore densities are zero. Cannot sample pores.")

    # Blank / non-positive totalpores keeps every measured pore on the matched
    # coupons. A positive value resamples that many sizes, weighted by density.
    use_catalog = totalpores is None or int(totalpores) <= 0
    weights = densities / total_density
    if use_catalog:
        volume = 0.0
    else:
        totalpores = int(totalpores)
        volume = totalpores / total_density

    density_by_type = {}
    weight_by_type = {}
    count_by_type = {}
    mean_esd_by_type = {}
    frames = []
    for weight, model, name, dens in zip(
        weights, selected_models, pore_model_names, densities
    ):
        density_by_type[name] = float(dens)
        weight_by_type[name] = float(weight)
        mean_esd_by_type[name] = None
        if use_catalog:
            if dens <= 0:
                count_by_type[name] = 0
                continue
            sampled_df, scan_volume = _catalog_pores(model)
            volume += scan_volume
        else:
            n_samples = int(np.round(weight * totalpores))
            count_by_type[name] = int(n_samples)
            if n_samples <= 0:
                continue
            sampled_df = model.sample(n_samples).copy()
        if sampled_df is None or len(sampled_df) == 0:
            count_by_type[name] = 0
            continue
        sampled_df = sampled_df.copy()
        sampled_df["pore_source"] = name
        count_by_type[name] = int(len(sampled_df))
        esd_col = None
        for cand in (
            "Equivalent Spherical Diameter (um)",
            "ESD",
        ):
            if cand in sampled_df.columns:
                esd_col = cand
                break
        if esd_col is not None and len(sampled_df):
            mean_esd_by_type[name] = float(
                pd.to_numeric(sampled_df[esd_col], errors="coerce").mean()
            )
        frames.append(sampled_df)

    if use_catalog:
        n_catalog = int(sum(count_by_type.values()))
        if n_catalog <= 0:
            raise RuntimeError("Matched coupons contain no pores.")
        if volume <= 0:
            volume = n_catalog / total_density
        totalpores = n_catalog

    samples = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Save outputs
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(param_dict, f, indent=4)
    with open(os.path.join(output_dir, 'mp_params.json'), 'w') as f:
        json.dump({'Model': str(mp_model_class), 'Properties': props}, f, indent=4)
    samples.to_csv(os.path.join(output_dir, 'samples.csv'), index=False)
    with open(os.path.join(output_dir, 'pore_type_metrics.json'), 'w') as f:
        json.dump(
            {
                "densities_per_mm3": density_by_type,
                "weights": weight_by_type,
                "sample_counts": count_by_type,
                "mean_esd_um": mean_esd_by_type,
                "total_density_per_mm3": total_density,
                "equivalent_volume_mm3": float(volume),
                "totalpores": int(totalpores),
            },
            f,
            indent=2,
        )

    if len(samples) == 0:
        raise RuntimeError("Empty Dataframe of Pores")

    # Plot number density
    fig, ax = plot_number_density(
        samples['Equivalent Spherical Diameter (um)'].values,
        volume=volume,
        ylog=True,
        xlog=False,
        xlabel='Equivalent Spherical Diameter (µm)',
        ylabel='Number Density (1/mm³)',
        figsize=(10, 5),
        color='black'
    )
    plot_path = os.path.join(output_dir, 'number_density.pdf')
    fig.savefig(plot_path)
    # fig.clf()

    metrics = {
        "densities_per_mm3": density_by_type,
        "weights": weight_by_type,
        "sample_counts": count_by_type,
        "mean_esd_um": mean_esd_by_type,
        "total_density_per_mm3": total_density,
        "equivalent_volume_mm3": float(volume),
        "totalpores": int(totalpores),
    }

    if return_figure:
        return {
            'output_dir': output_dir,
            'samples': samples,
            'plot_path': plot_path,
            'volume': volume,
            'figure': fig,
            'metrics': metrics,
        }
    else:
        fig.clf()
        return {
            'output_dir': output_dir,
            'samples': samples,
            'plot_path': plot_path,
            'volume': volume,
            'metrics': metrics,
        }
