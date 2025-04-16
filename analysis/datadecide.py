from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

FONTSIZE = 8

TASK_KEY_MAP = {
    "arc_challenge": "arc_challenge_test_5shot",
    "arc_easy": "arc_easy_test_5shot",
    "boolq": "boolq_val_5shot",
    "socialiqa": "socialiqa_val_5shot",
    "csqa": "csqa_val_5shot",
    "hellaswag": "hellaswag_val_5shot",
    "openbookqa": "openbookqa_test_5shot",
    "winogrande": "winogrande_val_5shot",
    "piqa": "piqa_val_5shot",
}

SIZE_COLORS = {
    '4M': 'brown',
    '6M': '#7f7f7f',  # gray
    '8M': '#17becf',  # cyan
    '10M': '#bcbd22', # olive
    '14M': '#e377c2', # pink
    '16M': '#8c564b', # brown
    '20M': 'black',
    '60M': 'teal',
    '90M': 'pink',
    '150M': '#1f77b4',
    '300M': '#2ca02c',
    '530M': '#ff7f0e',
    '750M': '#d62728',
    '1B': '#9467bd'
}

FULL_SCHEDULE = {
    '4M': 5725,
    '20M': 14584,
    '60M': 29042,
    '90M': 29901,
    '150M': 38157,
    '300M': 45787,
    '530M': 57786,
    '750M': 63589,
    '1B': 69369,
}

MODEL_TO_BATCH = {
    '4M': 32, # batch_size=32, gpus=8
    '6M': 32,
    '8M': 32,
    '10M': 32,
    '14M': 32,
    '16M': 32,
    '20M': 64,
    '60M': 96,
    '90M': 160,
    '150M': 192,
    '300M': 320,
    '530M': 448,
    '750M': 576,
    '1B': 704
}

MODEL_TO_PARAMETERS = {
    '4M': 3_744_832,
    '6M': 6_010_464,
    '8M': 8_538_240,
    '10M': 9_900_432,
    '12M': 12_066_600,
    '14M': 14_380_224,
    '16M': 16_004_560,
    '20M': 19_101_888,
    '60M': 57_078_144,
    '90M': 97_946_640,
    '150M': 151898880,
    '300M': 319980544,
    '530M': 530074944,
    '750M': 681297408,
    '1B': 1_176_832_000
}


def get_compute(scale):
    return 2048 * 6 * MODEL_TO_BATCH[scale] * MODEL_TO_PARAMETERS[scale] * FULL_SCHEDULE[scale]


def compute_2_class(ranking_a, ranking_b):
    """ Compute 2-class accuracy """
    ranking_a = list(ranking_a)
    ranking_b = list(ranking_b)

    assert len(ranking_b) == len(ranking_b)
    
    n = len(ranking_a)
    same_order_count = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            i_pred = ranking_b.index(ranking_a[i])
            j_pred = ranking_b.index(ranking_a[j])
            
            if (i < j and i_pred < j_pred) or (i > j and i_pred > j_pred):
                same_order_count += 1
            total_pairs += 1
    
    return same_order_count / total_pairs if total_pairs > 0 else 0.0


def get_slice(df, model, task):
    try:
        df = df.loc[(task, model)]
    except KeyError:
        return df.iloc[0:0]
    df = df.reset_index()
    return df


def plot_task_accuracy(ax: plt.Axes, two_class_results, task, sizes, show_legend=False, size_colors=SIZE_COLORS):
    # First plot all scatter points
    all_x = []
    all_y = []
    for size in list(size_colors.keys()):
        if size not in two_class_results.index.tolist():
            continue
        data = two_class_results.loc[size]
        x = np.array(two_class_results.columns, dtype=np.float64)
        y = np.array(data.values, dtype=np.float64)
        
        # Plot scatter points with consistent colors
        ax.scatter(x, y, marker='o', label=f'{size}', s=5, color=size_colors[size])
        
        # Collect valid points for overall spline
        mask = ~np.isnan(y) & ~np.isnan(x) & ~np.isneginf(y) & ~np.isneginf(x)
        all_x.extend(x[mask])
        all_y.extend(y[mask])
    
    # Add interpolating spline, ignoring nans
    mask = ~np.isnan(all_y) & ~np.isnan(all_x)
    if np.sum(mask) >= 3:  # Need at least 4 points for cubic spline
        all_x = np.array(np.array(all_x)[mask]) # exclude compute=0
        all_y = np.array(np.array(all_y)[mask]) # exclude compute=0

        x_nonzero = all_x != 0
        all_x = all_x[x_nonzero] # exclude x=0 values
        all_y = all_y[x_nonzero] # exclude x=0 values
        
        # Sort points by x value
        sort_idx = np.argsort(all_x)
        all_x = all_x[sort_idx]
        all_y = all_y[sort_idx]
        
        # Fit smoothed B-spline with high smoothing parameter
        x_smooth = np.logspace(np.log10(min(all_x)), np.log10(max(all_x)), len(all_x))
        # Use UnivariateSpline with high smoothing for a smoother fit
        spline = UnivariateSpline(np.log10(all_x), all_y, s=len(all_x))
        y_smooth = spline(np.log10(x_smooth))

        ax.plot(x_smooth, y_smooth, color='k', linestyle='--', label='spline', linewidth=1)
    
    # Add random baseline
    ax.axhline(y=0.5, color='r', linestyle='-', label='random', linewidth=0.5)
    
    ax.set_xlabel('Compute')
    ax.set_ylabel('2-class Accuracy')
    ax.set_title(f'{task}')
    ax.set_xscale('log', base=10)
    if show_legend: ax.legend(loc='lower right', fontsize=10, ncols=2)

    # Add vertical lines at specific FLOPS values with matching colors and accuracies
    for size in list(size_colors.keys()):
        if size not in two_class_results.index.tolist():
            continue
        try:
            flops = two_class_results.loc[size].dropna().index[0]
            acc = two_class_results.loc[size].get(np.float64(flops), np.nan)
            if not np.isnan(acc) and not np.isneginf(acc):
                ax.axvline(x=flops, color=size_colors[size], linestyle=':', alpha=0.7)
                ax.text(
                    flops, 0.98, ' ' + ('1.' if acc == 1 else f'{acc:.2f}').lstrip('0'), 
                    rotation=0, color=size_colors[size], ha='left', va='bottom', fontsize=8)
            else:
                # raise FileNotFoundError(f'Not all results found for task={task}, size={size}')
                raise FileNotFoundError(f'Not all results found for task={task}, size={size}')
        except Exception as e:
            # raise RuntimeError(f'Cant graph cheap decisions lines: {e}')
            print(f'Cant graph cheap decisions lines: {e}')