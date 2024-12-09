import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(ax: plt.Axes, values, mix_names, _type='p_values', alpha=0.01):
    """ Plot a pairwise heatmap of statistical significance """
    # Reorder values matrix according to sorted mixes
    mask = np.isnan(values)

    # Create a custom colormap that maps values between 0.5-0.95 to viridis
    # and values outside that range to grey
    if _type == 'p_values':
        def custom_colormap(value):
            if np.isnan(value):
                return (0, 0, 0, 0)
            elif value < alpha or value > (1-alpha):
            # elif value < 0.05:
                return (1, 1, 1, 0.05)
            else:
                return plt.cm.viridis(value)
    elif _type == 'power':
        def custom_colormap(value):
            if np.isnan(value) or value < 0:
                return (0, 0, 0, 0)
            elif value > 0.8:
                return (1, 1, 1, 0.05)
            else:
                return plt.cm.viridis(value)

    # Apply custom colors
    colors = [[custom_colormap(val) for val in row] for row in values]
    ax.imshow(colors)

    ax.set_xticks(range(len(mix_names)))
    ax.set_yticks(range(len(mix_names)))
    ax.set_xticklabels(mix_names, rotation=45, ha='right')
    ax.set_yticklabels(mix_names)

    # Add colorbar only for the viridis range
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    plt.colorbar(sm, ax=ax)

    # Add value annotations with smaller font
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not mask[i,j]:
                ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', fontsize=7)

    return ax
