import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import json
import os

from core import io  # Adjust if `io.py` is located elsewhere


# ---------- Fixtures ----------
@pytest.fixture
def sample_pore_csv(tmp_path):
    path = tmp_path / "pore_data.csv"
    df = pd.DataFrame({
        'x': [1.0, 2.0],
        'y': [3.0, 4.0],
        'size': [5.0, 6.0]
    })
    df.to_csv(path, index=False)
    return path

@pytest.fixture
def sample_parameters_csv(tmp_path):
    path = tmp_path / "params.csv"
    content = "parameter,value\nlaser_power,200\nscan_speed,800"
    path.write_text(content)
    return path

@pytest.fixture
def sample_material_props_csv(tmp_path):
    path = tmp_path / "props.csv"
    content = "property,value\ndensity,4.5\nyield_strength,300"
    path.write_text(content)
    return path

@pytest.fixture
def sample_material_model(tmp_path):
    model_dir = tmp_path / "materialmodels"
    model_dir.mkdir()
    model_file = model_dir / "Ti64.json"
    data = {
        "name": "Ti64",
        "density": 4.43,
        "thermal_conductivity": 6.7,
        "yield_strength": 930
    }
    with open(model_file, 'w') as f:
        json.dump(data, f)
    return model_file, model_dir


# ---------- Tests ----------
def test_load_pore_data(sample_pore_csv):
    df = io.load_pore_data(str(sample_pore_csv))
    assert isinstance(df, pd.DataFrame)
    assert "x" in df.columns
    assert len(df) == 2

def test_load_pore_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        io.load_pore_data("non_existent_file.csv")

def test_load_parameters(sample_parameters_csv):
    params = io.load_parameters(str(sample_parameters_csv))
    assert isinstance(params, dict)
    assert params["laser_power"] == 200
    assert params["scan_speed"] == 800

def test_load_parameters_missing_columns(tmp_path):
    path = tmp_path / "bad_params.csv"
    path.write_text("foo,bar\nx,1\ny,2")
    with pytest.raises(ValueError):
        io.load_parameters(str(path))

def test_load_material_props(sample_material_props_csv):
    props = io.load_material_props(str(sample_material_props_csv))
    assert props["density"] == 4.5
    assert props["yield_strength"] == 300

def test_load_material_props_missing_file():
    with pytest.raises(FileNotFoundError):
        io.load_material_props("bad_props.csv")

def test_load_material_model_success(sample_material_model):
    model_file, model_dir = sample_material_model
    model = io.load_material_model("Ti64", directory=str(model_dir))
    assert isinstance(model, dict)
    assert model["name"] == "Ti64"

def test_load_material_model_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.load_material_model("missing", directory=str(tmp_path))

def test_load_material_model_invalid_json(tmp_path):
    model_dir = tmp_path / "materialmodels"
    model_dir.mkdir()
    bad_json = model_dir / "invalid.json"
    bad_json.write_text("{invalid: true}")  # malformed JSON
    with pytest.raises(ValueError):
        io.load_material_model("invalid", directory=str(model_dir))

def test_list_available_data(tmp_path):
    (tmp_path / "data1.csv").write_text("x,y\n1,2")
    (tmp_path / "data2.csv").write_text("x,y\n3,4")
    (tmp_path / "not_csv.txt").write_text("ignore me")

    files = io.list_available_data(str(tmp_path))
    assert len(files) == 2
    assert all(f.endswith(".csv") for f in files)

def test_list_available_data_not_directory(tmp_path):
    file_path = tmp_path / "somefile.csv"
    file_path.write_text("x,y\n1,2")
    with pytest.raises(NotADirectoryError):
        io.list_available_data(str(file_path))
