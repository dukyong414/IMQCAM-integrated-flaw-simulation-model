import pandas as pd
from abc import ABC, abstractmethod


class poresource(ABC):
    def __init__(self, params, poretype, data=None, pores=None, poremodels=None, nsamples=100000):
        """
        Base class initializer for pore sources.

        Args:
            params (dict): Dictionary of process parameters.
            poretype (str): Type/name of pore model.
            data (optional): Process and pore measurement data (e.g. from CT).
            pores (optional): Pore descriptors DataFrame.

        Raises:
            ValueError: If required parameters are missing.
        """
        required = ["Power (W)", "Velocity (mm/s)", "Hatch Spacing (um)",
                    "Layer Thickness (um)", "Spot Size (um)", "Material", "Geometry"]
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing required process parameters: {missing}")

        self.params = params
        self.Power = params["Power (W)"]
        self.Velocity = params["Velocity (mm/s)"]
        self.H = params["Hatch Spacing (um)"]
        self.L = params["Layer Thickness (um)"]
        self.LaserD = params["Spot Size (um)"]
        self.preheat = params.get("Preheat Temperature (°C)", None)
        self.material = params["Material"]
        self.geometry = params["Geometry"]
        self.poretype = poretype
        self.EnergyDensity = self.Power / self.Velocity / self.LaserD * 1000**3 #um * mm to m

        self.nsamples = nsamples
        self.poremodels = poremodels or []

        # Defensive: Ensure data is filtered by geometry and valid
        if data is not None:
            df = pd.DataFrame(data)
            if 'Geometry' in df.columns:
                self.data = df[df.Geometry == self.geometry].reset_index(drop=True)
            else:
                self.data = df
        else:
            self.data = pd.DataFrame()

        self.poredf = pd.DataFrame(pores) if pores is not None else pd.DataFrame()

    @classmethod
    def getmodels(cls):
        """
        Return all subclasses of this base class.

        Returns:
            list: List of classes inheriting from poresource.
        """
        return cls.__subclasses__()

    def __str__(self):
        return (
            f"{self.poretype} Porosity | Power: {self.Power} W | Velocity: {self.Velocity} mm/s | "
            f"Hatch Spacing: {self.H} um | Layer Thickness: {self.L} um | "
            f"Geometry: {self.geometry} | Material: {self.material}"
        )

    @abstractmethod
    def get_nd(self, *args, **kwargs):
        """
        Generate number density histogram of pores.

        Returns:
            tuple: (histogram_counts, bin_edges)
        """
        pass

    @abstractmethod
    def sample(self, N, *args, **kwargs):
        """
        Sample pore distribution.

        Args:
            N (int): Number of pores to sample.

        Returns:
            pd.DataFrame: Sampled pore data.
        """
        pass

    @abstractmethod
    def getPoreDensity(self, *args, **kwargs):
        """
        Calculate total pore number density.

        Returns:
            float: Pore density (pores per mm³).
        """
        pass

    @classmethod
    def getparams(cls):
        """
        Describe parameters required for the model.

        Returns:
            dict: Dictionary with 'required' and 'description' keys.
        """
        pass

    @abstractmethod
    def ispresent(self, *args, **kwargs):
        """
        Check if this pore source is active under current conditions.

        Returns:
            bool: True if present, False otherwise.
        """
        pass

    @staticmethod
    def from_model_name(model_name, params, *args, **kwargs):
        """
        Create model from string name using factory pattern.

        Args:
            model_name (str): Model identifier.
            params (dict): Process parameters.

        Returns:
            poresource: Initialized pore model instance.

        Raises:
            ValueError: If model creation fails.
        """
        try:
            from models.factory import PoreModelFactory
            return PoreModelFactory.create(model_name, params, *args, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create model '{model_name}': {e}") from e
