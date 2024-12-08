import numpy as np
import pandas as pd
from tqdm import tqdm
# from power import run_power_test, run_mcnemar
from stats import compute_pairwise_p_values
import matplotlib.pyplot as plt


def get_slice(df, mix=None, model=None, task=None, metric=None, step=None):
    """ Index to return a df of some (mix, scale, task, metric, step) """
    mixes  = [mix] if isinstance(mix, str) else mix
    models  = [model] if isinstance(model, str) else model
    tasks   = [task] if isinstance(task, str) else task
    metrics = [metric] if isinstance(metric, str) else metric
    steps   = [step] if isinstance(step, int) else step

    # Dynamically create a slicing tuple matching the index levels
    level_slices = {
        'mix': mixes if mixes else slice(None),
        'model': models if models else slice(None),
        'task': tasks if tasks else slice(None),
        'metric': metrics if metrics else slice(None),
        'step': steps if steps else slice(None)
    }
    slicing_tuple = tuple(level_slices.get(level, slice(None)) for level in df.index.names)

    try:
        df = df.loc[slicing_tuple]
    except KeyError:
        return df.iloc[0:0]  # Return an empty DataFrame if no match
    
    # Sort and return
    if 'step' in df.index.names:
        df = df.sort_index(level='step')
    df = df.reset_index()

    return df


def get_max_k_step(_slice, k=1):
    """Filter for only rows with the top 5 steps."""
    top_steps = _slice['step'].nlargest(k).unique()
    step_filter = _slice['step'].isin(top_steps)
    _slice = _slice[step_filter]
    return _slice


def get_mix_nd_array(df, size, task, metric, step=None, sorted=True):
    """ Get an nd array of (mixes, instances), sorted by overall performance """
    slices = get_slice(df, None, size, task, metric, step=(step if step is not None and step >= 10 else None))

    if step is None: 
        slices = get_max_k_step(slices)
    elif step <= 10:
        slices = get_max_k_step(slices, step)
        assert len(set(slices['step'].unique())) == step, f'Did not get the requested number of steps: {step}. Do they exist in the df?'
    
    # Pivot the data to get mixes as columns and question_ids as rows
    pivoted = slices.pivot(index='question_id', columns='mix', values='value')
    mixes = pivoted.columns
    scores = pivoted.to_numpy().T

    if sorted:
        # Sort by overall performance
        mix_sums = scores.sum(axis=1)
        sorted_indices = mix_sums.argsort()[::-1]
        mixes = mixes[sorted_indices].tolist()
        scores = scores[sorted_indices]

    return mixes, scores


def get_mix_nd_array_avg(df, size, task, metric, steps):
    """ Get an nd array averaged over a specified number of training steps """
    mixes, scores = get_mix_nd_array(df, size, task, metric, step=steps[0])
    all_scores = np.zeros((len(steps), scores.shape[0], scores.shape[1]))
    all_scores[0, :, :] = scores
    for i, step in enumerate(steps):
        mixes_i, scores_i = get_mix_nd_array(df, size, task, metric, step=step)

        # reorder scores and mixes according to first array
        mixes_i_index = {mix: i for i, mix in enumerate(mixes_i)}
        mix_i_to_mix_1 = [mixes_i_index[mix] for mix in mixes]
        scores_i = scores_i[mix_i_to_mix_1]

        all_scores[i] = scores_i

    scores = all_scores.mean(axis=0)

    # Sort by overall performance
    mix_sums = scores.sum(axis=1)
    sorted_indices = mix_sums.argsort()[::-1]
    mixes = [mixes[i] for i in sorted_indices]
    scores = scores[sorted_indices]
    return mixes, scores


def perc_significant(p_values):
    """ Calculate the % of statistically significant comparisons """
    return ((p_values > 0.95).sum() + (p_values < 0.05).sum()) / (~np.isnan(p_values)).sum()

def perc_powerful(power):
    """ Calculate the % of adequately powerful comparisons """
    return ((power > 0.8).sum()) / (~np.isnan(power)).sum()


def compute_pairwise_power(scores, α=0.05, r=1000):
    n_models, n_instances = scores.shape
    pairwise_power = np.zeros((n_models, n_models)) - 1

    for i in tqdm(range(n_models), desc="Computing power analysis"):
        for j in range(n_models):
            if i != j:
                preds_i = scores[i]
                preds_j = scores[j]
                
                acc_i = np.mean(preds_i)
                acc_j = np.mean(preds_j)
                agreement = np.mean(preds_i == preds_j)

                # compute (M2 Correct, M1 Incorrect) entry
                acc_better, acc_worse = (preds_i, preds_j) if acc_i > acc_j else (preds_j, preds_i)
                m2_corr_m1_incorr = np.sum(np.logical_and(acc_better, (1-acc_worse))) / len(preds_i)
                
                try:
                    raise NotImplementedError()
                    power, _, _, _ = run_power_test(acc_i, acc_j, agreement, n_instances, α=α, r=r, quiet=True)
                    # _, p_value = run_mcnemar(acc_i, acc_j, agreement, n_instances)
                except RuntimeError:
                    power = -1 # if the true effect size is 0, power is undefined

                # pairwise_power[i, j] = m2_corr_m1_incorr
                # pairwise_power[i, j] = agreement
                # pairwise_power[i, j] = p_value
                pairwise_power[i, j] = power

    return pairwise_power


def plot_heatmap(ax: plt.Axes, values, mix_names, _type='p_values'):
    # Reorder values matrix according to sorted mixes

    mask = np.isnan(values)

    # Create a custom colormap that maps values between 0.5-0.95 to viridis
    # and values outside that range to grey
    if _type == 'p_values':
        def custom_colormap(value):
            if np.isnan(value):
                return (0, 0, 0, 0)
            elif value < 0.05 or value > 0.95:
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


def compute_significance(df, scales, metric, steps=None, tasks=None, _type='p_values', do_plot=True):
    scales = [scales] if not isinstance(scales, list) else scales
    if tasks is None: 
        tasks = df.index.get_level_values('task_suite').unique()
        tasks = list(set([i.replace(':para', '') for i in tasks]))

    sig_results = pd.DataFrame(index=scales, columns=tasks)

    n_tasks = len(tasks)
    if do_plot: fig, axes = plt.subplots(n_tasks, 2, figsize=(20, 8*n_tasks))

    for scale in scales:
        for i, task in tqdm(enumerate(tasks), desc='Computing pairwise comparisons', total=len(tasks)):
            for j, task_name in enumerate([f'{task}', f'{task}:para']):
                if isinstance(steps, list):
                    mixes, scores = get_mix_nd_array_avg(df, scale, task_name, metric, steps=steps)
                else:
                    mixes, scores = get_mix_nd_array(df, scale, task_name, metric, step=steps)

                if _type == 'p_values':
                    p_values = compute_pairwise_p_values(scores)

                    p_values = np.nan_to_num(compute_pairwise_p_values(scores), nan=0) + np.nan_to_num(compute_pairwise_p_values(scores[::-1]).T, nan=0)
                    np.fill_diagonal(p_values, np.nan)

                    sig_results.loc[scale, task_name] = perc_significant(p_values)
                elif _type == 'power':
                    p_values = compute_pairwise_power(scores)
                    sig_results.loc[scale, task_name] = perc_powerful(p_values)
                else: raise ValueError(_type)

                if do_plot:
                    axes[i, j] = plot_heatmap(axes[i, j], p_values, mixes, _type=_type)
                    axes[i, j].set_title(f'{_type} for {task_name} (n={scores.shape[1]}) at {scale} across data mixes at step {(steps if steps is not None else "final")} ({metric})', fontsize=10)

    if do_plot:
        fig.tight_layout()
        return sig_results, fig
    return sig_results, None


def process_instance_stats_df(instance_stats_df):
    # Reshape the nested JSON structure into long format
    instance_stats_df_copy = instance_stats_df.copy()

    # Get first row to extract structure
    first_row = instance_stats_df_copy.iloc[0]
    first_dict = next(iter(first_row.values))
    first_inner_dict = next(iter(first_dict.values()))

    # Get keys from both levels of nested dictionaries
    metrics = []
    for inner_dict in first_dict.values():
        metrics.extend(inner_dict.keys())
    metrics = list(set([m for m in metrics if m != 'checkpoints']))

    # Create rows for each model-metric combination
    rows = []
    for idx in instance_stats_df_copy.index:
        for model in instance_stats_df_copy.columns:
            entry = instance_stats_df_copy.loc[idx, model]
            for instance_id, instance_data in entry.items():
                for metric in metrics:
                    try:
                        checkpoints = instance_data['checkpoints']
                        values = instance_data[metric]
                        if len(checkpoints) != len(values): checkpoints = checkpoints[1:] # sometimes we don't have eval data for the first ckpt
                        for ckpt_idx, checkpoint in enumerate(checkpoints):
                            value = values[ckpt_idx] if isinstance(values, list) else values
                            rows.append({
                                'task_suite': idx,
                                'question_id': instance_id,
                                'model': model,
                                'metric': metric,
                                'checkpoint': checkpoint,
                                'value': value
                            })
                    except (KeyError, TypeError) as e:
                        continue

    # Convert to dataframe and set index
    instance_df = pd.DataFrame(rows)

    # Process specific column names
    import re

    def extract_step(checkpoint):
        match = re.search(r'step(\d+)', checkpoint)
        if match: return int(match.group(1))
        return 0

    instance_df['model_name'] = instance_df['model']
    instance_df['model'] = instance_df['model_name'].apply(lambda x: x.split('-')[-2])
    instance_df['group'] = instance_df['model_name'].apply(lambda x: '-'.join(x.split('-')[:-2]))
    instance_df['step'] = instance_df['checkpoint'].apply(extract_step)

    instance_df = instance_df.drop(columns=['checkpoint', 'model_name'])

    instance_df.set_index(['task_suite', 'model', 'step', 'group', 'metric'], inplace=True)

    return instance_df