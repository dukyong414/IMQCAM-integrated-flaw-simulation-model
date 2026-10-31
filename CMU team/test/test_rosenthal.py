import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from meltpool import get_model

def test_rosenthal_valid_input():
    Rosenthal = get_model("Rosenthal")

    params = {
        'Power (W)': 200,
        'Velocity (mm/s)': 800,
        'Preheat Temperature (°C)': 100,
        'Hatch Spacing (um)': 100,
        'Layer Thickness (um)': 50,
        'Spot Size (um)': 70,
        'Material': 'Ti6Al4V',
        'Geometry': 'Cube'
    }

    props = {
        'Absorptivity': 0.35,
        'Specific Heat Capacity (J/g°C)': 0.5,
        'Density (kg/m^3)': 7800,
        'Melting Point (°C)': 1500,
        'Thermal Conductivity (W/mK)': 25
    }

    model = Rosenthal(params, props)

    W, D, L = model.getDims()
    assert W > 0 and D > 0 and L > 0

    # Test alternate PV calculation
    W2, D2, L2 = model.getDims_PV(150, 600)
    assert W2 > 0 and D2 > 0 and L2 > 0

    # Test getpropnames
    props_list = model.getpropnames()
    assert isinstance(props_list, list)
    assert all('name' in item and 'description' in item for item in props_list)

def test_rosenthal_invalid_props():
    Rosenthal = get_model("Rosenthal")
    bad_props = {
        'Absorptivity': -0.35,  # Invalid negative absorptivity
        'Specific Heat Capacity (J/g°C)': 0.5,
        'Density (kg/m^3)': 7800,
        'Melting Point (°C)': 1500,
        'Thermal Conductivity (W/mK)': 25
    }
    params = {
        'Power (W)': 200,
        'Velocity (mm/s)': 800,
        'Preheat Temperature (°C)': 100,
        'Hatch Spacing (um)': 100,
        'Layer Thickness (um)': 50,
        'Spot Size (um)': 70,
        'Material': 'Ti6Al4V',
        'Geometry': 'Cube'
    }
    with pytest.raises(ValueError):
        Rosenthal(params, bad_props)

def test_rosenthal_invalid_params():
    Rosenthal = get_model("Rosenthal")
    props = {
        'Absorptivity': 0.35,
        'Specific Heat Capacity (J/g°C)': 0.5,
        'Density (kg/m^3)': 7800,
        'Melting Point (°C)': 1500,
        'Thermal Conductivity (W/mK)': 25
    }
    bad_params = {
        'Power (W)': 0,  # Invalid zero power
        'Velocity (mm/s)': 800,
        'Preheat Temperature (°C)': 100,
        'Hatch Spacing (um)': 100,
        'Layer Thickness (um)': 50,
        'Spot Size (um)': 70,
        'Material': 'Ti6Al4V',
        'Geometry': 'Cube'
    }
    with pytest.raises(ValueError):
        Rosenthal(bad_params, props)

def test_get_model_and_list():
    from meltpool import list_models
    assert 'Rosenthal' in list_models()
    assert get_model('Rosenthal') is not None
