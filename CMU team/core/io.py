import os
import json
import pandas as pd

def load_pore_data(path):
    """
    Load pore descriptors from a CSV file.

    Args:
        path (str): Path to CSV file

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If file cannot be read as a DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pore data file not found: {path}")

    try:
        df = pd.read_csv(path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Loaded file is not a valid DataFrame.")
        return df
    except Exception as e:
        raise ValueError(f"Failed to read pore data CSV: {e}")

def load_parameters(path):
    """
    Load process parameters from a CSV into a dictionary.

    Expected format:
        parameter,value

    Args:
        path (str): Path to parameter CSV

    Returns:
        dict

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If required columns are missing or file is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parameter file not found: {path}")

    try:
        df = pd.read_csv(path)
        if 'parameter' not in df.columns or 'value' not in df.columns:
            raise ValueError("Expected CSV columns: 'parameter','value'")
        return dict(zip(df['parameter'], df['value']))
    except Exception as e:
        raise ValueError(f"Failed to load parameters: {e}")

def load_material_props(path):
    """
    Load material property values from a CSV.

    Expected format:
        property,value

    Args:
        path (str): Path to material properties CSV

    Returns:
        dict

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If required columns are missing or file is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Material property file not found: {path}")

    try:
        df = pd.read_csv(path)
        if 'property' not in df.columns or 'value' not in df.columns:
            raise ValueError("Expected CSV columns: 'property','value'")
        return dict(zip(df['property'], df['value']))
    except Exception as e:
        raise ValueError(f"Failed to load material properties: {e}")

def load_material_model(name, directory='materialmodels'):
    """
    Load a material model from a JSON file in the specified directory.

    Args:
        name (str): Name of the material model file (without .json)
        directory (str): Subdirectory containing material model files

    Returns:
        dict: Material model as dictionary

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the JSON is invalid or cannot be parsed
    """
    filename = os.path.join(directory, f"{name}.json")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Material model file not found: {filename}")

    try:
        with open(filename, 'r') as f:
            model = json.load(f)
        if not isinstance(model, dict):
            raise ValueError("Material model must be a dictionary.")
        return model
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {filename}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading material model file: {e}")

def list_available_data(data_dir):
    """
    List all pore data CSV files in a directory.

    Args:
        data_dir (str): Directory to search

    Returns:
        list of str: Paths to pore CSV files

    Raises:
        NotADirectoryError: If the path is not a directory
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Provided path is not a directory: {data_dir}")

    try:
        return [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith('.csv') and os.path.isfile(os.path.join(data_dir, f))
        ]
    except Exception as e:
        raise RuntimeError(f"Failed to list files in directory '{data_dir}': {e}")
