import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.pce import compute_pairwise_p_values
from plot import plot_heatmap
from dataloader import get_nd_array


def calc_total_variation(arr, norm=False, improvement=False):
    """ Total variation """
    if len(arr) == 0: return 0
    arr = np.array(arr)

    tv = np.mean(np.abs(np.diff(arr)))
    if norm: tv /= (max(arr) - min(arr))
    if improvement: tv -= calc_improvement(arr)

    return tv

def calc_improvement(arr):
    return (arr[-1] - arr[0]) / len(arr)


def perc_significant(p_values):
    """ Calculate the % of statistically significant comparisons """
    return ((p_values > 0.95).sum() + (p_values < 0.05).sum()) / (~np.isnan(p_values)).sum()


def compute_significance(df, models, metric, steps=None, tasks=None, do_plot=True):
    models = [models] if not isinstance(models, list) else models
    if tasks is None: 
        tasks = df.index.get_level_values('task').unique()
        tasks = list(set([i.replace(':para', '') for i in tasks]))

    sig_results = pd.DataFrame(index=models, columns=tasks)

    n_tasks = len(tasks)
    if do_plot: fig, axes = plt.subplots(n_tasks, 2, figsize=(20, 8*n_tasks))

    for model in models:
        for i, task in tqdm(enumerate(tasks), desc='Computing pairwise comparisons', total=len(tasks)):
            for j, task_name in enumerate([f'{task}', f'{task}:para']):
                if isinstance(steps, list):
                    raise NotImplementedError('Cannot average across multiple steps!')
                    mixes, scores = get_mix_nd_array_avg(df, model, task_name, metric, steps=steps)
                else:
                    mixes, scores = get_nd_array(df, 'mix', metric, model=model, task=task_name, step='max')

                p_values = compute_pairwise_p_values(scores)

                p_values = np.nan_to_num(compute_pairwise_p_values(scores), nan=0) + np.nan_to_num(compute_pairwise_p_values(scores[::-1]).T, nan=0)
                np.fill_diagonal(p_values, np.nan)

                sig_results.loc[model, task_name] = perc_significant(p_values)

                if do_plot:
                    axes[i, j] = plot_heatmap(axes[i, j], p_values, mixes)
                    axes[i, j].set_title(f'p-values for {task_name} (n={scores.shape[1]}) at {model} across data mixes at step {(steps if steps is not None else "final")} ({metric})', fontsize=10)

    if do_plot:
        fig.tight_layout()
        return sig_results, fig
    return sig_results, None