"""
Porosity source models registry.

This module centralizes access to all pore formation models such as
lack-of-fusion (LoF), keyhole, and gas porosity sources.

It supports safe dynamic model lookup and third-party extension through
the `register_model()` function.

"""
from .lof import lof
from .keyhole import keyhole
from .gas import gas
from .factory import PoreModelFactory  # Make sure this line exists

# Internal registry of model name -> class
_available_models = {}

def register_model(name, cls):
    """
    Register a new pore model by name.

    Args:
        name (str): Unique model identifier (case-insensitive).
        cls (type): Model class (must be a subclass of poresource).

    Raises:
        ValueError: If the name is empty or already registered.
        TypeError: If cls is not a subclass of poresource.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model name must be a non-empty string.")
    name = name.lower()
    if name in _available_models:
        raise ValueError(f"Model '{name}' is already registered.")
    
    from core.poresource import poresource
    if not issubclass(cls, poresource):
        raise TypeError(f"Model '{name}' must inherit from poresource.")

    _available_models[name] = cls

def get_model(name):
    """
    Fetch a pore source model class by name.

    Args:
        name (str): Model name (e.g., 'lof', 'keyhole').

    Returns:
        type: The corresponding pore model class.

    Raises:
        KeyError: If the model is not registered.
    """
    name = name.lower()
    if name not in _available_models:
        raise KeyError(f"Pore model '{name}' is not registered.")
    return _available_models[name]

def list_models():
    """
    List all registered pore model names.

    Returns:
        list of str: Sorted list of registered names.
    """
    return sorted(_available_models.keys())

# Register built-in models
register_model('lof', lof)
register_model('keyhole', keyhole)
register_model('gas', gas)

__all__ = ['get_model', 'list_models', 'register_model']
