from core.poresource import poresource
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger(__name__)

class keyhole(poresource):
    def __init__(self, params, data=None, pores=None, props = None, nsamples=100000, threshold=3217391, poremodels=None,):
        super().__init__(params, 'lof', data, pores, poremodels, nsamples)

        self.threshold = threshold
        

        self.ED = self.EnergyDensity if self.EnergyDensity is not None else self.getED()
        self.Keyhole = self.ED > self.threshold if self.ED is not None else False

        self.isValid = self.valid_process_params()
        if self.isValid:
            self.partition = self.getPartition()
            if self.partition is not None:
                self.partition = self.partition.dropna()
            else:
                self.partition = pd.DataFrame()
        else:
            self.partition = pd.DataFrame()

    def getED(self):
        try:
            return self.Power / (self.Velocity * self.LaserD) * 1000**3 #um * mm to m

        except Exception as e:
            logger.warning(f"Failed to compute ED: {e}")
            return None

    def getED_data(self):
        if self.data is None or self.data.empty:
            logger.warning("Data empty, cannot compute ED column.")
            return
        # Defensive: check columns exist
        required_cols = ['Power (W)', 'Velocity (mm/s)', 'Spot Size (um)']
        if not all(col in self.data.columns for col in required_cols):
            logger.warning(f"Missing columns for ED calculation: {required_cols}")
            return
        with np.errstate(divide='ignore', invalid='ignore'):
            self.data['ED'] = self.data['Power (W)'] / (
                self.data['Velocity (mm/s)'] * self.data['Spot Size (um)']
            ) * 1000**3 #um * mm to m

    def withinHull(self):
        if self.data is None or self.data.empty:
            logger.warning('No data available for convex hull check.')
            return False
        if self.Power is None or self.Velocity is None:
            logger.warning('Power or Velocity parameter missing for hull check.')
            return False

        points = self.data[['Power (W)', 'Velocity (mm/s)']].to_numpy()
        try:
            hull = ConvexHull(points)
        except Exception as e:
            logger.warning(f'ConvexHull computation failed: {e}')
            return False
        point = np.array([self.Power, self.Velocity])
        return np.all(hull.equations[:, :-1].dot(point) <= -hull.equations[:, -1])

    def getPartition(self, npoints=4):
        if self.data is None or self.data.empty:
            return pd.DataFrame()

        subdf = self.data.copy()
        subdf['Partition'] = 0.0
        subdf['Partition'] = subdf['Partition'].astype(float)

        if not self.Keyhole:
            return subdf

        if not self.withinHull():
            logger.warning('Parameters not in convex hull. Extrapolating keyhole model.')

        powers = self.data['Power (W)'].unique()
        pmin = powers[np.max(np.where(powers <= self.Power))]
        pmax = powers[np.min(np.where(powers >= self.Power))]
        # Filter data by power range
        filtered = self.data[(self.data['Power (W)'] <= pmax) & (self.data['Power (W)'] >= pmin)]
        # Group and count samples, losing Partition, so add it below
        subdf = filtered.groupby(['Power (W)', 'Velocity (mm/s)', 'ED']).size().reset_index(name='Count')

        subdf['Partition'] = 0.0  # Add Partition column after groupby

        # Calculate distance in power-velocity space
        subdf['Distance'] = np.sqrt(
            (subdf['Power (W)'] - self.Power)**2 + (subdf['Velocity (mm/s)'] - self.Velocity)**2
        )
        if len(subdf) >= npoints:
            mindist = sorted(subdf['Distance'])[-npoints]
        else:
            mindist = None

        if subdf['Distance'].min() == 0:
            subdf.loc[subdf['Distance'] == 0, 'Partition'] = 1.0
        elif mindist is not None:
            mask = subdf['Distance'] < mindist
            subdf.loc[mask, 'Partition'] = 1.0 / subdf.loc[mask, 'Distance']
            total = subdf['Partition'].sum()
            if total > 0:
                subdf['Partition'] /= total
            else:
                logger.warning("Sum of Partition weights is zero; setting uniform weights.")
                subdf['Partition'] = 1.0 / len(subdf)
        else:
            logger.warning("Mindist is None; partition weights not assigned.")

        subdf = subdf[subdf['Partition'] != 0]

        return subdf


    # --- Sampling logic (unchanged from LoF) ---
    def sample(self, N=None):
        """
        Sample exactly N pores with unique sample numbers.
        Allows duplicate PV draws, but avoids one-to-many merge expansion.
        """
        if self.partition is None or self.partition.empty:
            logger.warning("Partition empty, returning empty samples.")
            return pd.DataFrame()

        if 'Partition' not in self.partition.columns or self.partition['Partition'].max() == 0:
            logger.warning("Partition weights missing or zero, returning empty samples.")
            return pd.DataFrame()

        if self.data is None or self.data.empty:
            logger.warning("No process data available for sampling.")
            return pd.DataFrame()

        if self.poredf is None or self.poredf.empty:
            logger.warning("Pore dataframe empty, returning no samples.")
            return pd.DataFrame()

        N = N or self.nsamples

        weights = self.partition['Partition'].astype(float)
        if weights.sum() <= 0:
            logger.warning("Partition weights sum to zero or less, returning empty samples.")
            return pd.DataFrame()
        weights /= weights.sum()

        rng = np.random.default_rng()

        # --- Step 1: Draw N rows from partition (duplicates allowed)
        sampled_partition = self.partition.sample(
            n=N, weights=weights, replace=True, random_state=rng
        ).reset_index(drop=True)

        # --- Step 2: For each sampled PV, pick exactly one random scan row from data
        pv_tuples = list(zip(sampled_partition['Power (W)'], sampled_partition['Velocity (mm/s)']))
        selected_scans = []
        for p, v in pv_tuples:
            matches = self.data[
                (self.data['Power (W)'] == p) &
                (self.data['Velocity (mm/s)'] == v)
            ]
            if not matches.empty:
                selected_scans.append(matches.sample(n=1, random_state=rng))
        if not selected_scans:
            logger.warning("No matching scans found for sampled partition.")
            return pd.DataFrame()

        merged = pd.concat(selected_scans, ignore_index=True)

        # --- Step 3: For each scan, pick one random pore from poredf
        pore_groups = {n: g.index.to_numpy() for n, g in self.poredf.groupby('Name')}
        chosen_pores = []
        for name in merged['Name']:
            if name in pore_groups:
                idx = rng.choice(pore_groups[name])
                chosen_pores.append(self.poredf.loc[idx])

        if not chosen_pores:
            logger.warning("No pores matched sampled scans.")
            return pd.DataFrame()

        sampled_pores = pd.DataFrame(chosen_pores).reset_index(drop=True)

        # --- Step 4: Guarantee exactly N rows and add unique sample IDs
        sampled_pores = sampled_pores.sample(
            n=min(len(sampled_pores), N), replace=False, random_state=rng
        ).reset_index(drop=True)
        sampled_pores['Sample #'] = np.arange(1, len(sampled_pores) + 1)

        if len(sampled_pores) < N:
            logger.warning(f"Only {len(sampled_pores)}/{N} valid pore samples available.")

        return sampled_pores

    def getPoreDensity(self):
        if self.partition is None or self.partition.empty:
            logger.warning("Keyhole partition empty, returning zero pore density.")
            return 0.0

        if 'Partition' not in self.partition.columns or self.partition['Partition'].max() == 0:
            logger.warning("Partition weights missing or zero, returning zero pore density.")
            return 0.0

        dens = 0.0
        for i, row in self.partition.iterrows():
            scans = self.data[
                (self.data['Power (W)'] == row.get('Power (W)', None)) &
                (self.data['Velocity (mm/s)'] == row.get('Velocity (mm/s)', None))
            ]
            if scans.empty:
                continue

            vol = scans['Volume (mm^3)'].sum() if 'Volume (mm^3)' in scans.columns else 0
            if vol <= 0:
                logger.warning(f"Volume sum is zero or less for scans at index {i}, skipping density contribution.")
                continue

            npores = len(self.poredf[self.poredf['Name'].isin(scans['Name'])]) if self.poredf is not None else 0
            dens += (npores / vol) * row['Partition']

        return dens

    def get_nd(self, minsize=None, maxsize=None, res=2.5):
        if self.partition is None or self.partition.empty:
            logger.warning("Partition empty, returning None for number density.")
            return None, None

        if 'Partition' not in self.partition.columns or self.partition['Partition'].max() == 0:
            logger.warning("Partition weights missing or zero, returning None for number density.")
            return None, None

        samples = self.sample()
        if samples.empty:
            logger.warning("Sampled pores empty, returning None for number density.")
            return None, None

        if minsize is None:
            minsize = samples['ESD'].min() if 'ESD' in samples.columns else 0
        if maxsize is None:
            maxsize = samples['ESD'].max() if 'ESD' in samples.columns else 1

        poredens = self.getPoreDensity()
        bins = max(1, int((maxsize - minsize) / res))
        hist, bin_edges = np.histogram(samples['ESD'], bins=bins, range=(minsize, maxsize), density=True)
        hist *= poredens

        return hist, bin_edges

    def valid_process_params(self):
        self.getED_data()
        return bool(self.withinHull())

    @classmethod
    def getparams(cls):
        """
        Return a dictionary of all process parameters required by the Keyhole pore model,
        including laser parameters, layer info, geometry, and material.
        """
        return {
            'Power (W)': 'Laser power in Watts',
            'Velocity (mm/s)': 'Scan velocity in mm/s',
            'Hatch Spacing (um)': 'Distance between scan lines in micrometers',
            'Layer Thickness (um)': 'Thickness of each layer in micrometers',
            'Spot Size (um)': 'Laser spot diameter in micrometers',
            'Material': 'Material identifier or name',
            'Geometry': 'Part geometry identifier',
            "Preheat Temperature (°C)": 'Preheat Temperature'
        }


    def ispresent(self):
        return self.Keyhole
