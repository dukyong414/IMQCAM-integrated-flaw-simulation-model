import matplotlib.pyplot as plt

def set_global_style():
    plt.style.use('default')  # Base style

    plt.rcParams.update({
        # Figure
        'figure.figsize': (10, 5),
        'figure.dpi': 100,

        # Axes
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'axes.labelcolor': 'black',
        'axes.titlesize': 16,
        'axes.titlecolor': 'black',
        'axes.grid': False,

        # Spines and ticks
        'xtick.bottom': True,
        'xtick.top': False,
        'ytick.left': True,
        'ytick.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': 'black',
        'ytick.color': 'black',

        # Lines / markers
        'lines.markersize': 6,
        'lines.linewidth': 1.5,
        'lines.color': 'black',

        # Font
        'font.size': 12,
        'text.color': 'black',

        # Legend
        'legend.fontsize': 12,
        'legend.edgecolor': 'black',

        # Savefig
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none'
    })

    # Remove top and right spines for all plots by default
    def remove_spines(ax=None):
        if ax is None:
            ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Attach spine cleanup to pyplot
    plt.remove_spines = remove_spines
