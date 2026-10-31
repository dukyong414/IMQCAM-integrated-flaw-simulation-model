import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_violin_size_distribution(df, size_column='pore_size', group_column=None,
                                  figsize=(10, 5), color='lightblue', title=None):
    """
    df: pandas DataFrame with pore size data.
    size_column: column name for pore sizes.
    group_column: column name for grouping (optional).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if group_column:
        sns.violinplot(data=df, x=group_column, y=size_column, ax=ax, color=color)
    else:
        sns.violinplot(y=df[size_column], ax=ax, color=color)

    ax.set_xlabel(group_column if group_column else '')
    ax.set_ylabel('Equivalent Spherical Diameter (µm)')
    if title:
        ax.set_title(title)

    return fig, ax

def plot_kde_size_distribution(column, figsize=(10, 5), color='black',
                               xlabel='Equivalent Spherical Diameter (µm)',
                               ylabel='Density'):
    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(column, ax=ax, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax
