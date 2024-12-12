import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_heatmap(ax: plt.Axes, values, mix_names, mix_scores=None, sig_clusters=None, _type='p_values', alpha=0.01):
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

    if mix_scores is not None:
        mix_names = [f'{name} (score={score:.3f})' for name, score in zip(mix_names, mix_scores)]
    
    if sig_clusters is not None:
        # Find indices where the significance cluster changes, then add a vertical line
        change_indices = np.where(sig_clusters[:-1] != sig_clusters[1:])[0] + 1
        for idx in change_indices:
            ax.axvline(x=idx-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=idx-0.5, xmin=0, xmax=1, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xticks(range(len(mix_names)))
    ax.set_yticks(range(len(mix_names)))
    ax.set_xticklabels(mix_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(mix_names, fontsize=8)

    # Add colorbar only for the viridis range
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)

    # Add value annotations with smaller font
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not mask[i,j]:
                ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', fontsize=7)

    return ax


def plot_training(ax: plt.Axes, x, y, xlabel: str, ylabel: str, label=None, title=None, color=None, fit=None):
    if xlabel == 'step':
        ax.plot(x, y, label=label, color=color, linewidth=0.5, marker='.', markersize=2)
        # ax.plot(df_slice[xlabel], df_slice[ylabel].rolling(window=5).mean(), label=label, color=color, linewidth=0.5, marker='.', markersize=2)
    else:
        ax.scatter(x, y, label=label, color=color, s=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if label is not None: ax.legend()

    ax.tick_params(axis='both', which='major', labelsize=8)

    # add fitted log function
    if fit is not None:
        # ax.set_xscale('log')

        from scipy.optimize import curve_fit
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore", message="invalid value encountered in log")

        def log(x, a, b, c): return a * np.log(b * x) + c
        # def log(x, epsilon, k, gamma): return epsilon - k * np.exp(-gamma * x) # samir err

        # x, y = df_slice[xlabel].values, df_slice[ylabel].values
        x_max = np.max(x)
        x_scaled = x / x_max
        popt, _ = curve_fit(log, x_scaled, y, maxfev=10000)
        
        x_fit = np.linspace(min(x), max(x), 100)
        x_fit_scaled = x_fit / x_max
        y_fit = log(x_fit_scaled, *popt)

        ax.plot(x_fit, y_fit, color=color, alpha=0.5, linestyle='dotted')
    
    return ax