import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def get_number_density(
    column: np.ndarray,
    volume: float,
    minsize: Optional[float] = None,
    maxsize: Optional[float] = None,
    res: float = 2.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute number density histogram for pore sizes.

    Args:
        column (np.ndarray): Array of pore sizes (e.g., Equivalent Spherical Diameter).
        volume (float): Volume in mm^3 for normalization.
        minsize (float, optional): Minimum pore size to consider.
        maxsize (float, optional): Maximum pore size to consider.
        res (float): Bin width for histogram.

    Returns:
        hist (np.ndarray): Histogram values scaled to number density (pores/mm^3).
        bins (np.ndarray): Bin edges.
    """
    if minsize is None:
        minsize = np.min(column)
    if maxsize is None:
        maxsize = np.max(column)

    bins = np.arange(minsize, maxsize + res, res)
    hist, bins = np.histogram(column, bins=bins, density=True)

    # Scale by total pore number density
    poredens = len(column) / volume  # pores per mm³
    hist *= poredens

    return hist, bins

def plot_number_density(
    column: np.ndarray,
    volume: float,
    ylog: bool = True,
    xlog: bool = False,
    xlabel: str = 'Equivalent Spherical Diameter (µm)',
    ylabel: str = 'Number Density (1/mm³)',
    figsize: Tuple[int, int] = (10, 5),
    minsize: Optional[float] = None,
    maxsize: Optional[float] = None,
    res: float = 2.5,
    color: str = 'black'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot number density histogram for pore sizes.

    Args:
        column (np.ndarray): Array of pore sizes.
        volume (float): Volume in mm^3 for normalization.
        ylog (bool): Whether to use logarithmic scale for y-axis.
        xlog (bool): Whether to use logarithmic scale for x-axis.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        figsize (tuple): Figure size.
        minsize (float, optional): Minimum pore size for histogram bins.
        maxsize (float, optional): Maximum pore size for histogram bins.
        res (float): Bin width.
        color (str): Plot color.

    Returns:
        fig (plt.Figure), ax (plt.Axes)
    """
    hist, bins = get_number_density(column, volume, minsize, maxsize, res)
    bincenters = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(bincenters, hist, color=color)

    if ylog:
        ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if minsize is not None and maxsize is not None:
        ax.set_xlim(minsize, maxsize)

    return fig, ax

def plot_cdf(
    column: np.ndarray,
    xlog: bool = False,
    xlabel: str = 'Equivalent Spherical Diameter (µm)',
    ylabel: str = 'Cumulative Fraction',
    figsize: Tuple[int, int] = (10, 5),
    minsize: Optional[float] = None,
    maxsize: Optional[float] = None,
    color: str = 'black'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the cumulative distribution function (CDF) of pore sizes.

    Args:
        column (np.ndarray): Array of pore sizes.
        xlog (bool): Whether to use logarithmic scale for x-axis.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        figsize (tuple): Figure size.
        minsize (float, optional): Minimum pore size for x-axis limit.
        maxsize (float, optional): Maximum pore size for x-axis limit.
        color (str): Plot color.

    Returns:
        fig (plt.Figure), ax (plt.Axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(column, bins=100, cumulative=True, density=True,
            color=color, histtype='step')

    if xlog:
        ax.set_xscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if minsize is not None and maxsize is not None:
        ax.set_xlim(minsize, maxsize)

    return fig, ax
