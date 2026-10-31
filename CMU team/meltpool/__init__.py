from .rosenthal import Rosenthal

# Optional: Register models for reflection
available_models = {
    'Rosenthal': Rosenthal,
    # Add more models here as you implement them
}

def get_model(name):
    """Fetch a melt pool model class by name"""
    return available_models.get(name)

def list_models():
    """Return a list of available melt pool model names"""
    return list(available_models.keys())

__all__ = ['Rosenthal', 'get_model', 'list_models']
