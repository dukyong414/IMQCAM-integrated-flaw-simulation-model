import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd

from models import lof, keyhole, gas, PoreModelFactory

# Common mock parameters and data for tests
mock_params = {
    'Power (W)': 200,
    'Velocity (mm/s)': 800,
    'Hatch Spacing (um)': 100,
    'Layer Thickness (um)': 30,
    'Spot Size (um)': 80,
    'Preheat Temperature (°C)': 100,
    'Material': 'Ti64',
    'Geometry': 'SampleGeo'
}

# Minimal mock melt pool model for lof
class MockMPModel:
    def __init__(self, params, props):
        pass

    def getDims(self):
        return (100, 50, 200)

    def getDims_PV(self, P_arr, V_arr):
        Ws = np.full_like(P_arr, 100)
        Ds = np.full_like(V_arr, 50)
        return Ws, Ds, np.zeros_like(P_arr)

    def getpropNames(self):
        return ['mock_prop']

# Generate mock data frames
def generate_mock_data():
    # Data for P, V, Volume, Geo, Sample columns needed by pore models
    data = pd.DataFrame({
        'Power (W)': [200, 200, 250, 250],
        'Velocity (mm/s)': [800, 850, 800, 850],
        'Volume': [1.0, 1.2, 0.9, 1.1],
        'Geo': ['SampleGeo'] * 4,
        'Sample': ['S1', 'S2', 'S3', 'S4'],
        'Spot Size (um)': [80, 80, 80, 80],  # spot size for keyhole
    })
    # Pore descriptors DataFrame
    pores = pd.DataFrame({
        'Name': ['S1', 'S2', 'S3', 'S4'],
        'ESD': [5.0, 6.0, 7.0, 8.0],
        'Sample': ['S1', 'S2', 'S3', 'S4']
    })
    return data, pores

@pytest.fixture
def lof_instance():
    data, pores = generate_mock_data()
    props = {'Model': MockMPModel, 'Properties': {}}
    model = lof(mock_params, data=data, pores=pores, props=props)
    return model

@pytest.fixture
def keyhole_instance():
    data, pores = generate_mock_data()
    model = keyhole(mock_params, data=data, pores=pores, nsamples=10)
    return model

@pytest.fixture
def gas_instance():
    data, pores = generate_mock_data()
    # gas constructor expects poremodels list - mock empty for simplicity
    model = gas(mock_params, data=data, pores=pores, poremodels=[], nsamples=10)
    return model

def test_lof_init_and_params(lof_instance):
    assert lof_instance.isValid in [True, False]
    params = lof_instance.getparams()
    assert isinstance(params, dict)
    assert "Power (W)" in params['required']

def test_lof_sampling_and_density(lof_instance):
    samples = lof_instance.sample(5)
    assert isinstance(samples, pd.DataFrame)
    # Sample columns should include 'ESD' if present
    if not samples.empty:
        assert 'ESD' in samples.columns or 'Name' in samples.columns

    density = lof_instance.getPoreDensity()
    assert isinstance(density, float)
    assert density >= 0

def test_lof_get_nd(lof_instance):
    hist, bins = lof_instance.get_nd(res=2.5)
    if hist is not None and bins is not None:
        assert len(hist) + 1 == len(bins)

def test_keyhole_init_and_params(keyhole_instance):
    assert isinstance(keyhole_instance.isValid, bool)
    params = keyhole_instance.getparams()
    assert "Power (W)" in params['required']

def test_keyhole_sampling_and_density(keyhole_instance):
    samples = keyhole_instance.sample(5)
    assert isinstance(samples, pd.DataFrame)
    density = keyhole_instance.getPoreDensity()
    assert density >= 0

def test_keyhole_get_nd(keyhole_instance):
    hist, bins = keyhole_instance.get_nd(res=2.5)
    if hist is not None and bins is not None:
        assert len(hist) + 1 == len(bins)

def test_gas_init_and_params(gas_instance):
    # gas has ispresent boolean and getparams dict
    assert hasattr(gas_instance, 'has_gas')
    params = gas_instance.getparams()
    assert "Power (W)" in params['required']

def test_gas_sampling_and_density(gas_instance):
    samples = gas_instance.sample(5)
    assert isinstance(samples, pd.DataFrame)
    density = gas_instance.getPoreDensity()
    assert density >= 0

def test_gas_get_nd(gas_instance):
    hist, bins = gas_instance.get_nd(res=2.5)
    if hist is not None and bins is not None:
        assert len(hist) + 1 == len(bins)

def test_factory_create_and_multiple(lof_instance, keyhole_instance):
    # Test factory create for valid model
    model = PoreModelFactory.create('lof', mock_params, data=lof_instance.data, pores=lof_instance.poredf, props={'Model': MockMPModel, 'Properties': {}})
    assert model.poretype == 'lof'

    # Test factory create for unknown model name raises
    with pytest.raises(ValueError):
        PoreModelFactory.create('unknown_model', mock_params)

    # Test factory create_multiple returns dict of instances
    models = PoreModelFactory.create_multiple(['lof', 'keyhole'], mock_params,
                                             data=lof_instance.data, pores=lof_instance.poredf, props={'Model': MockMPModel, 'Properties': {}})
    assert 'lof' in models and 'keyhole' in models
    assert all(hasattr(m, 'sample') for m in models.values())

def test_factory_combined_sample_and_density(lof_instance, keyhole_instance):
    models = {'lof': lof_instance, 'keyhole': keyhole_instance}

    combined_samples = PoreModelFactory.combined_sample(models, nsamples=20)
    assert isinstance(combined_samples, pd.DataFrame)
    assert 'pore_source' in combined_samples.columns

    combined_density = PoreModelFactory.combined_pore_density(models)
    assert isinstance(combined_density, float)
    assert combined_density >= 0

def test_factory_combined_number_density_histogram(lof_instance, keyhole_instance):
    models = {'lof': lof_instance, 'keyhole': keyhole_instance}
    hist, bins = PoreModelFactory.combined_number_density_histogram(models, nsamples=50)
    assert isinstance(hist, np.ndarray)
    assert isinstance(bins, np.ndarray)
    assert len(hist) + 1 == len(bins)

def test_repr_and_getmodels():
    # Test base poresource __str__ and getmodels classmethod
    from core.poresource import poresource
    models = poresource.getmodels()
    assert any(issubclass(m, poresource) for m in models)
    instance_str = str(lof(mock_params))
    assert isinstance(instance_str, str)
