import numpy as np
import pandas as pd
from models.factory import PoreModelFactory

def sample_pores(poredf, N=100, method='random'):
    """
    Sample pore descriptors using a chosen strategy.

    Args:
        poredf (pd.DataFrame): pore descriptors
        N (int): number to sample
        method (str): 'random', 'largest', or 'uniform'

    Returns:
        pd.DataFrame
    """
    if poredf is None or len(poredf) == 0:
        raise ValueError("No pore data to sample from.")

    if method == 'random':
        return poredf.sample(n=min(N, len(poredf)), random_state=42)

    elif method == 'largest':
        return poredf.sort_values(by='diameter', ascending=False).head(N)

    elif method == 'uniform':
        bins = np.linspace(poredf['diameter'].min(), poredf['diameter'].max(), N + 1)
        samples = []
        for i in range(N):
            bin_slice = poredf[(poredf['diameter'] >= bins[i]) & (poredf['diameter'] < bins[i+1])]
            if not bin_slice.empty:
                samples.append(bin_slice.sample(1))
        return pd.concat(samples).reset_index(drop=True)

    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
def sample_pores_from_model(model_name, params, n_samples=100, *args, **kwargs):
    """
    Instantiate pore model via factory and sample pores.

    Args:
        model_name (str): e.g. 'lof'
        params (dict): Process parameters
        n_samples (int): Number of pores to sample

    Returns:
        pd.DataFrame of sampled pores
    """
    model = PoreModelFactory.create(model_name, params, *args, **kwargs)
    return model.sample(n_samples)

