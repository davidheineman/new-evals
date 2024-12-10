import pandas as pd
import math
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


def convert_sci(n):
    """ Convert to 10^X notation """
    exponent = math.floor(math.log10(n))
    mantissa = n / (10 ** exponent)
    result = f"{mantissa:.2f} * 10^{exponent}"
    return result


def compute_f1(gold_mixes, pred_mixes):
    gold_set = set(gold_mixes)
    pred_set = set(pred_mixes)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1 # we really care about precision (minimize false negatives)


def perc_significant(p_values, alpha=0.05):
    """ Calculate the % of statistically significant comparisons """
    return ((p_values > (1-alpha)).sum() + (p_values < alpha).sum()) / (~np.isnan(p_values)).sum()


def get_bound(arr, idx, alpha):
    """ Get the first index where arr[i] < alpha """
    condition = (arr < alpha) | (arr > (1-alpha))
    indices = np.argwhere(condition)

    if indices.size > 0:
        first_index = tuple(indices[0])
    else:
        return len(arr) + idx

    return first_index[0] + idx


def create_stratified_array(counts):
    """ Convert counts to 1D array of weights: [1172, 2304] => [1172, 1172, ..., 2304, 2304, ...] """
    return np.concatenate([
        np.full(count, count, dtype=np.float64) 
        for value, count in enumerate(counts)
    ])


def get_sig_cluster_bound(p_vals, idx, alpha):
    """ Given an idx and p_vals, compute the boundary of the next significance cluster """
    col = p_vals[:, idx][idx+1:]
    row = p_vals[idx, :][idx+1:]
    return min(get_bound(col, idx, alpha=alpha), get_bound(row, idx, alpha=alpha)) + 1 # PERM-INPUTS clustering
    # return max(get_bound(col, idx, alpha=alpha), get_bound(row, idx, alpha=alpha)) + 1 # conservative clustering (gives less clusters)


def get_sig_clusters(p_vals, alpha=0.01):
    """
    Start with highest scoring mix, assign rank 1 to all mixes until we have 
    encountered a mix statistically significantly different from any mix so far.
    """
    sig_clusters = np.zeros(p_vals.shape[0])

    curr, curr_cluster = 0, 0
    while curr < p_vals.shape[0]:
        idx = curr
        cluster_bound = get_sig_cluster_bound(p_vals, idx, alpha)
        for _ in range(idx, cluster_bound):
            sig_clusters[curr] = curr_cluster
            curr += 1
        curr_cluster += 1

    return sig_clusters


def compute_significance(df, models, metric, last_n=1, tasks=None, alpha=0.05, do_plot=False, quiet=False):
    if tasks is None: 
        tasks = df.index.get_level_values('task').unique()

    sig_results = pd.DataFrame(index=['perc_sig'], columns=tasks)
    all_p_values = {}

    n_tasks = len(tasks)
    if do_plot: 
        fig, axes = plt.subplots(n_tasks, 1, figsize=(10, 8*n_tasks))
        if n_tasks == 1: axes = [axes]

    for i, task in tqdm(enumerate(tasks), desc='Computing pairwise comparisons', total=len(tasks), disable=quiet):
        if last_n > 1:
            mixes, scores = get_nd_array(df, ['mix', 'step'], 'acc_per_char', model=models, task=task)

            scores = scores[:, -last_n:, :] # get last n steps

            scores = scores.mean(axis=1) # approach 1: average over last n ckpts
            # scores = scores.reshape(scores.shape[0], scores.shape[1] * scores.shape[2]) # approach 2: concat last n ckpts -- I'm concerned this wont work with instance weights

            # Recover just the mix names using the first value for step
            first_step = mixes.get_level_values('step').unique()[0]
            mixes = mixes[mixes.get_level_values('step') == first_step].get_level_values('mix')

            # Sort based on new aggregate (average/concat)
            mix_sums = scores.sum(axis=1)
            sorted_indices = mix_sums.argsort()[::-1]
            mixes = mixes[sorted_indices].tolist()
            scores = scores[sorted_indices]
        else:
            mixes, scores = get_nd_array(df, 'mix', metric, model=models, task=task, step='max', sorted=True)

        if isinstance(task, list):
            from dataloader import get_slice
            from utils.pce import compute_weighted_pairwise_p_values

            # Get value counts for each task
            slices = get_slice(df, model=models, task=task)
            unique_counts = slices.groupby('task')['native_id'].nunique()
            weights = create_stratified_array(unique_counts)

            # Compute paired permutation test with instance weights
            p_values, mix_scores, _ = compute_weighted_pairwise_p_values(scores, weights=weights, return_scores=True)

            # Change task name
            task = 'olmes macro average'
        else:
            # p_values = compute_pairwise_p_values(scores)
            p_values = np.nan_to_num(compute_pairwise_p_values(scores), nan=0) + np.nan_to_num(compute_pairwise_p_values(scores[::-1]).T, nan=0)
            np.fill_diagonal(p_values, np.nan)

            mix_scores = None

        sig_clusters = get_sig_clusters(p_values, alpha=alpha)

        sig_results.loc['perc_sig', task] = perc_significant(p_values, alpha=alpha)
        all_p_values[task] = (mixes, scores, p_values)

        if do_plot:
            axes[i] = plot_heatmap(axes[i], p_values, mixes, mix_scores, sig_clusters, alpha=alpha)
            axes[i].set_title(r'$p$' + f'-values for {task} (n={scores.shape[1]}) across data mixes at {("last " + str(last_n) + " steps" if last_n > 1 else "final checkpoint")} ({metric}), ' + r'$\alpha$=' + f'{alpha}', fontsize=10)

    if do_plot:
        fig.tight_layout()
        return sig_results, all_p_values, fig
    return sig_results, all_p_values, None