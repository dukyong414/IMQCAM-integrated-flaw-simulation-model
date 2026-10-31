import numpy as np
from core.mpmodel import mpmodel

class Rosenthal(mpmodel):
    def __init__(self, params, props):
        super().__init__(params, props)

        # Normalize keys for parameters and properties
        self.P = self._get_param_value(params, ['P', 'Power (W)'])
        self.V = self._get_param_value(params, ['V', 'Velocity (mm/s)'])
        self.preheat = params.get('preheat', params.get('Preheat Temperature (°C)', 25))

        # Validate parameters with normalized keys
        self._validate_positive({'P': self.P, 'V': self.V}, ['P', 'V'], "parameters")

        # Normalize and validate props
        self.epsilon = self._get_prop_value(props, ['Absorptivity'])
        self.C = self._get_prop_value(props, ['Specific Heat Capacity (J/g°C)']) * 1000  # J/kg°C
        self.rho = self._get_prop_value(props, ['Density (kg/m^3)'])
        self.Tm = self._get_prop_value(props, ['Melting Point (°C)'])
        self.k = self._get_prop_value(props, ['Thermal Conductivity (W/mK)'])

        self._validate_positive(
            {'Absorptivity': self.epsilon, 'Specific Heat Capacity (J/g°C)': self.C, 
             'Density (kg/m^3)': self.rho, 'Melting Point (°C)': self.Tm, 'Thermal Conductivity (W/mK)': self.k},
            ['Absorptivity', 'Specific Heat Capacity (J/g°C)', 'Density (kg/m^3)', 'Melting Point (°C)', 'Thermal Conductivity (W/mK)'],
            "properties"
        )

        self.alpha = self.k / (self.rho * self.C)  # Thermal diffusivity
        self.Q = self.P * self.epsilon  # Absorbed power
        self.dT = self.Tm - self.preheat  # Temperature difference to melt

        self.W = self.getW() * 1e6  # Width in micrometers
        self.Length = self.getL() * 1e6  # Length in micrometers
        self.D = self.getD()  # Depth in micrometers

    def _get_param_value(self, dictionary, possible_keys):
        """Return the first found value from possible_keys in dictionary or raise KeyError."""
        for key in possible_keys:
            if key in dictionary:
                return dictionary[key]
        raise KeyError(f"None of the keys {possible_keys} found in parameters")

    def _get_prop_value(self, dictionary, possible_keys):
        """Return the first found value from possible_keys in dictionary or raise KeyError."""
        for key in possible_keys:
            if key in dictionary:
                return dictionary[key]
        raise KeyError(f"None of the keys {possible_keys} found in properties")

    def _validate_positive(self, dictionary, keys, label):
        for key in keys:
            if key not in dictionary:
                raise ValueError(f"Missing required {label} key: '{key}'")
            if dictionary[key] <= 0:
                raise ValueError(f"{label.capitalize()} value '{key}' must be positive, got {dictionary[key]}")

    def getDims_PV(self, P, V):
        """Get melt pool dimensions for given power and velocity"""
        if P <= 0 or V <= 0:
            raise ValueError("Power and Velocity must be positive for dimension calculation.")
        V = V / 1000  # mm/s to m/s
        Q = P * self.epsilon
        W = 2 * np.sqrt(2 * Q / (np.pi * np.exp(1) * self.rho * self.C * V * self.dT))
        D = W / 2
        L = Q / (2 * np.pi * self.k * self.dT)
        return W * 1e6, D * 1e6, L * 1e6
    
    def getDims_PV_array(self, P, V):
        """Get melt pool dimensions for given power and velocity"""
        if np.any(P <= 0) or np.any(V <= 0):
            raise ValueError("Power and Velocity must be positive for dimension calculation.")
        V = V / 1000  # mm/s to m/s
        Q = P * self.epsilon
        W = 2 * np.sqrt(2 * Q / (np.pi * np.exp(1) * self.rho * self.C * V * self.dT))
        D = W / 2
        L = Q / (2 * np.pi * self.k * self.dT)
        return W * 1e6, D * 1e6, L * 1e6

    def getL(self):
        return self.Q / (2 * np.pi * self.k * self.dT)

    def getW(self):
        return 2 * np.sqrt(2 * self.Q / (np.pi * np.exp(1) * self.rho * self.C * self.V * self.dT))

    def getD(self):
        return self.getW() / 2

    def getDims(self):
        self.W = self.getW() * 1e6
        self.Length = self.getL() * 1e6
        self.D = self.getD() * 1e6
        return self.W, self.D, self.Length
    
    @classmethod
    def getpropnames(cls):
        return [
            {'name': 'Absorptivity', 'description': 'Laser absorptivity of the material'},
            {'name': 'Specific Heat Capacity (J/g°C)', 'description': 'Specific heat capacity of the material in J/g°C'},
            {'name': 'Density (kg/m^3)', 'description': 'Density of the material in kg/m³'},
            {'name': 'Melting Point (°C)', 'description': 'Melting point temperature of the material in °C'},
            {'name': 'Thermal Conductivity (W/mK)', 'description': 'Thermal conductivity of the material in W/m·K'}
        ]

    def __str__(self):
        return "Rosenthal"
