import numpy as np

def set_seed(seed):
    """
    Set NumPy seed for reproducibility.
    """
    np.random.seed(seed)

def get_bounding_box(points):
    """
    Compute bounding box of a point cloud.
    
    Args:
        points (np.ndarray): N x 3 array of x, y, z coords
    
    Returns:
        tuple: (min_xyz, max_xyz)
    """
    return np.min(points, axis=0), np.max(points, axis=0)

def rescale(values, new_min, new_max):
    """
    Rescale array to new [min, max] range.
    
    Args:
        values (array-like)
        new_min, new_max (float): target range
    
    Returns:
        np.ndarray
    """
    vmin, vmax = np.min(values), np.max(values)
    if vmax == vmin:
        return np.full_like(values, new_min)
    return (values - vmin) / (vmax - vmin) * (new_max - new_min) + new_min
