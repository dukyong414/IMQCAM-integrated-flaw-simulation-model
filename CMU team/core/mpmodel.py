from abc import ABC, abstractmethod

class mpmodel(ABC):
    """Parent abstract class for generic melt pool model"""

    def __init__(self, params, props):
        self.params = params
        self.P = params['Power (W)']               # Laser Power (W)
        self.V = params['Velocity (mm/s)']         # Scanning Velocity (mm/s)
        self.H = params['Hatch Spacing (um)']      # Hatch spacing (um)
        self.L = params['Layer Thickness (um)']    # Layer thickness (um)
        self.Spot = params['Spot Size (um)']       # Laser spot size (um)
        self.preheat = params['Preheat Temperature (°C)']  # Preheat temp (°C)
        self.props = props                          # Material properties dict

    @classmethod
    def getmodels(cls):
        """Return all melt pool model subclasses for selection"""
        return cls.__subclasses__()

    @abstractmethod
    def getL(self):
        """Get melt pool length"""
        pass

    @abstractmethod
    def getW(self):
        """Get melt pool width"""
        pass

    @abstractmethod
    def getD(self):
        """Get melt pool depth"""
        pass

    def getDims(self):
        """Return melt pool dimensions as a tuple (W, D, L)"""
        return self.getW(), self.getD(), self.getL()

    @abstractmethod
    def getpropnames(self):
        """Return list of material property names required"""
        pass

    @classmethod
    def getparams(cls):
        """
        Return a dictionary of parameters needed by the Lack of Fusion (LoF) pore model,
        including process parameters and melt pool model parameters.
        """
        # Base process parameters common to LoF
        base_params = {
            'Power (W)': 'Laser power in Watts',
            'Velocity (mm/s)': 'Scan velocity in mm/s',
            'Hatch Spacing (um)': 'Distance between scan lines in micrometers',
            'Layer Thickness (um)': 'Thickness of each layer in micrometers',
            'Spot Size (um)': 'Laser spot diameter in micrometers',
            'Material': 'Material identifier or name',
            'Geometry': 'Part geometry identifier',
            "Preheat Temperature (°C)": 'Preheat Temperature'
        }

        # Melt pool model parameters
        mp_params = {}
        propnames = cls.getpropnames()  # <- use cls, not self

        if propnames and isinstance(propnames[0], dict):
            mp_params = {p['name']: p['description'] for p in propnames}
        elif propnames and isinstance(propnames[0], str):
            mp_params = {p: 'Melt pool model parameter' for p in propnames}

        # Combine and return
        combined_params = {**base_params, **mp_params}
        return combined_params
    
    def __str__(self):
        return self.__class__.__name__
