import os, sys, itertools
sys.path.append(os.path.dirname(os.getcwd()))
from utils import DATA_DIR, ROOT_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from dataloader import get_slice
from ladder_wrapper import run_ladder
from stats import compute_significance, compute_total_variation
from table import display_task_variants

from ladder_ian import compute_2_class, get_compute, plot_task_accuracy
from utils.constants_models import DDOS_MODEL_NAMES
from download.utils_cheap_decisions import PRIMARY_METRICS_OLMES

DEFAULT_LADDER_CONFIG_PATH = f'{ROOT_DIR}/analysis/utils/ladder_config.json'

ALL_METRICS = ['logits_per_char_corr', 'primary_score']
REVERSED_METRICS = ['margin_per_byte', 'norm_correct_prob_per_byte', 'correct_prob_per_byte', 'correct_logit_per_byte', 'logits_per_char_corr', 'logits_per_byte_corr']

DDOS_SIZES = ['4M', '20M', '60M', '90M', '150M', '300M', '530M', '750M', '1B']
# DDOS_SIZES = ['4M', '20M', '60M', '150M', '300M', '530M', '750M', '1B']
DDOS_COMPUTE_SIZES = tuple(get_compute(size) for size in DDOS_SIZES)

def get_perf_size(df, size, task, metric):
    """ Get performance of all models at a specific size """
    _slice: pd.DataFrame = get_slice(df, task=task)
    _slice = _slice[((_slice['size'] == size)) & (_slice['model'].isin(DDOS_MODEL_NAMES))]
    if isinstance(task, str):
        _slice = _slice[_slice['task'] == task]
    elif isinstance(task, list):
        _slice = _slice[_slice['task'].isin(task)]

    # Only aggregate numerical columns
    numerical_cols = _slice.select_dtypes(include='number').columns.tolist()
    non_numerical_cols = _slice.select_dtypes(exclude='number').columns.tolist()
    _slice = _slice.groupby('model', as_index=False).agg({col: 'mean' for col in numerical_cols} | {col: 'first' for col in non_numerical_cols})
    _slice['task_name'] = 'aggregate'

    _slice = _slice.reset_index().sort_values('step')[['model', 'mix', 'step', 'size', metric]]
    _slice['compute'] = _slice['size'].apply(lambda x: get_compute(x) if '-' in x else x)
    _slice = _slice.sort_values(metric, ignore_index=True)
    return _slice


def construct_2class_table(df, selected_tasks, small_metric=ALL_METRICS, target_metric='primary_metric', model_sizes=DDOS_SIZES):
    """
    Compute 2-class accuracy. We are predicting primary_metric at 1B using the metric at a smaller scale

    There is a TODO to merge with Ian's implmenetation
    """
    if not isinstance(small_metric, list): small_metric = [small_metric]

    combinations = list(itertools.product(small_metric, model_sizes, selected_tasks))
    two_class = pd.DataFrame(columns=['metric', 'size', 'task', 'accuracy'])

    for metric, size, task in tqdm(combinations, desc='Computing two class accuracy', disable=(len(combinations) < 50)):
        _slice = get_slice(df, task=task)
        # _slice = _slice[((_slice['size'] == size)) & (_slice['task'] == task) & (_slice['model'].isin(DDOS_MODEL_NAMES))] # get data for small scale
        _slice = _slice[((_slice['size'] == size)) & (_slice['model'].isin(DDOS_MODEL_NAMES))] # get data for small scale
        steps = [sorted(_slice['step'].unique())[-1]]
        for step in steps:
            # get data at the small scale
            small_scale = get_perf_size(df, size, task, metric)['mix']

            # predict at the target scale (1B) 
            target_scale = get_perf_size(df, '1B', task, target_metric)['mix']

            # display(_slice)
            # # display(target_scale)
            # if size == '150M':
            #     raise RuntimeError()
            
            if metric in REVERSED_METRICS and target_metric not in REVERSED_METRICS: small_scale = reversed(small_scale)
            try:
                accuracy = compute_2_class(small_scale, target_scale)
            except Exception as e:
                print((metric, size, task), e)
                accuracy = float('-inf')

            # Get tokens/compute of small scale
            step_slice = _slice[_slice['step'] == float(step)]
            step_slice = step_slice.reset_index(drop=True)
            # tokens = step_slice['tokens'][0]
            try:
                compute = get_compute(step_slice['size'][0])
            except Exception as e:
                print((metric, size, task), e)
                compute = float('-inf')

            new_entry = pd.DataFrame({
                'metric': [metric],
                'size': [size], 
                'step': [step], 
                'task': [str(task)],
                'accuracy': [accuracy],
                # 'tokens': [tokens],
                'compute': [compute]
            })
            new_entry = new_entry.dropna(axis=1, how='all')            
            two_class = two_class.dropna(axis=1, how='all')            
            two_class = pd.concat([two_class, new_entry], ignore_index=True)

    # Create two dataframes - one for best accuracies and one for corresponding metrics
    best_acc_df = two_class.loc[two_class.groupby(['task', 'size', 'step'])['accuracy'].idxmax()][['task', 'size', 'step', 'accuracy', 'compute']].reset_index(drop=True)
    best_metric_df = two_class.loc[two_class.groupby(['task', 'size', 'step'])['accuracy'].idxmax()][['task', 'size', 'step', 'metric', 'compute']].reset_index(drop=True)

    # Create pivot tables with size in specified order
    acc_pivot = best_acc_df.pivot(index='task', columns=['size', 'compute'], values='accuracy')[model_sizes]
    metric_pivot = best_metric_df.pivot(index='task', columns=['size', 'compute'], values='metric')[model_sizes]

    # Add average row
    acc_pivot.loc['average'] = acc_pivot.mean()

    return two_class, acc_pivot, metric_pivot


def get_title_from_task(task):
    if isinstance(task, list):
        title_mapping = {
            'mmlu_pro_': 'mmlu_pro',
            'mmlu_abstract_algebra:mc': 'mmlu_mc',
            'mmlu': 'mmlu',
            'minerva': 'minerva',
            'agi_eval': 'agi_eval',
            'bbh': 'bbh',
            'arc_challenge:para': 'olmes_core9_para',
            'arc_challenge:distractors': 'olmes_core9_distractors',
            'arc_challenge:enlarge': 'olmes_core9_enlarge',
            'arc_challenge:mc': 'olmes_core9_mc',
            'arc_challenge': 'olmes_core9',
            'drop': 'olmes_gen',
        }
        for key, title in title_mapping.items():
            if key in task[0]:
                return title
        return 'aggregate'
    return task

def set_title_from_task(ax: plt.Axes, task):
    ax.set_title(get_title_from_task(task))


def run_analysis(df, task, ladder_models, external_ladder_models, eval_ladder_models, metric='primary_score', axes=None, ladder_config_path=DEFAULT_LADDER_CONFIG_PATH):
    results = {}
    
    primary_score_name = PRIMARY_METRICS_OLMES[task] if isinstance(task, str) and task in PRIMARY_METRICS_OLMES else 'primary_score'
    try:
        # Step 1 ladder prediction (base models)
        ax: plt.Axes = axes[0, 0] if axes is not None else None
        rel_error_step_1, _, _ = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish7", "peteish13-highlr"],
            config_path=ladder_config_path,
            run_step2=False, run_stacked=False,
            axes=[ax]
        )
        results.update({
            "rel_error_step_1_7b": rel_error_step_1[0], 
            "rel_error_step_1_13B": rel_error_step_1[1], 
        })
        if ax:
            ax.set_ylabel('Task loss (BPB)')
            ax.legend(fontsize=6)

        # Step 2 ladder prediction (base models)
        ax: plt.Axes = axes[1, 0] if axes is not None else None
        _, rel_error_step_2, _ = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish7", "peteish13-highlr"],
            config_path=ladder_config_path,
            run_step1=False, run_stacked=False,
            axes=[ax]
        )
        results.update({
            "rel_error_step_2_7b": rel_error_step_2[0], 
            "rel_error_step_2_13B": rel_error_step_2[1], 
        })
        if ax:
            ax.set_xlabel('Task loss (BPB)')
            ax.set_ylabel(primary_score_name)
            ax.legend(fontsize=6)

        # Step 2 ladder prediction (external models)
        ax: plt.Axes = axes[2, 0] if axes is not None else None
        _, mean_error_step_2 = run_ladder(
            df,
            task,
            train_models=ladder_models + external_ladder_models,
            eval_models=eval_ladder_models,
            config_path=ladder_config_path,
            run_step1=False, run_stacked=False,
            return_fit_error=True,
            axes=[ax]
        )
        results.update({
            "mean_error_step_2_external": mean_error_step_2, 
        })
        if ax:
            ax.get_legend().remove()
            # ax.legend(fontsize=3, ncols=2)
            ax.set_xlabel('Task loss (BPB)')
            ax.set_ylabel(primary_score_name)
            ax.text(
                x=0.02, y=0.02, s=f'Mean Error={mean_error_step_2*100:.2f}%',
                transform=ax.transAxes,
                va='bottom', ha='left',
                fontsize=8
            )

        # Stacked ladder prediction
        ax: plt.Axes = axes[3, 0] if axes is not None else None
        _, _, rel_error_stacked = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish7", "peteish13-highlr"],
            config_path=ladder_config_path,
            run_step1=False, run_step2=False,
            axes=[ax]
        )
        results.update({
            "rel_error_stacked_7b": rel_error_stacked[0], 
            "rel_error_stacked_13b": rel_error_stacked[1], 
        })
        if ax:
            ax.set_ylabel(primary_score_name)
            ax.legend(fontsize=6)
    except Exception as e:
        print(task, 'failed on ladder fits', e)
        # raise RuntimeError(task, 'failed on ladder fits', e)

    # intermediate checkpoints
    intermediate_models = ['peteish-moreeval-1B-5xC', 'peteish13-highlr']
    intermediate_tv = []
    for j, model in enumerate(intermediate_models):
        ax: plt.Axes = axes[0+(j*2), 1] if axes is not None else None
        tv, _ = compute_total_variation(
            df, models=[model], metric='logits_per_char_corr', tasks=[task], axes=[ax]
        )
        tv_bpb = tv[task]['total_variation'] if not isinstance(task, list) else tv.loc['total_variation']['aggregate']
        if ax and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=6)
            ax.set_ylabel('Task loss (BPB)')
            
            # Get the y-values from the current axis
            lines = ax.get_lines()
            if len(lines) > 0:
                y_data = lines[0].get_ydata()
                # Set top limit 10% above max y value
                y_max = np.max(y_data)
                # Get y-value 10% into the curve for bottom limit
                idx = int(len(y_data) * 0.1)
                y_20_percent = y_data[idx]
                if not (np.isnan(y_20_percent) or np.isnan(y_max) or np.isinf(y_20_percent) or np.isinf(y_max)):
                    ax.set_ylim(bottom=y_20_percent, top=y_max * (0.95 if y_max < 0 else 1.05))

        # 1B intermediate checkpoints
        ax: plt.Axes = axes[1+(j*2), 1] if axes is not None else None
        tv, _ = compute_total_variation(
            df, models=[model], metric=metric, tasks=[task], axes=[ax]
        )
        tv_primary = tv[task]['total_variation'] if not isinstance(task, list) else tv.loc['total_variation']['aggregate']
        if ax and ax.get_legend_handles_labels()[1]:
            ax.set_ylabel(primary_score_name)
            ax.legend(fontsize=6)

        intermediate_tv += [(tv_bpb, tv_primary)]

    results.update({
        "tv_bpb_1b": intermediate_tv[0][0], 
        "tv_primary_1b": intermediate_tv[0][1], 
        "tv_bpb_7b": intermediate_tv[1][0], 
        "tv_primary_7b": intermediate_tv[1][1]
    })

    # Consistent rankings analysis
    try:
        two_class, acc_pivot_bpb_primary, metric_pivot = construct_2class_table(
            df, [task], small_metric='logits_per_byte_corr', target_metric='primary_score'
        )
        two_class_results = acc_pivot_bpb_primary.loc[str(task)].unstack()
        if axes is not None:
            ax: plt.Axes = axes[1, 2]
            plot_task_accuracy(ax, two_class_results, str(task), DDOS_COMPUTE_SIZES)
            ax.set_ylabel(f'Decision Acc (BPB on {primary_score_name})')
            ax.set_ylim(0.75, 1)

        two_class, acc_pivot_best_metric, metric_pivot = construct_2class_table(
            df, [task], target_metric='primary_score'
        )
        two_class_results = acc_pivot_best_metric.loc[str(task)].unstack()
        if axes is not None:
            ax: plt.Axes = axes[2, 2]
            plot_task_accuracy(ax, two_class_results, str(task), DDOS_COMPUTE_SIZES)
            ax.set_ylabel(f'Decision Acc (best on {primary_score_name})')
            ax.set_ylim(0.75, 1)
            
        two_class, acc_pivot_bpb, metric_pivot = construct_2class_table(
            df, [task], 
            small_metric='logits_per_byte_corr', target_metric='logits_per_byte_corr'
        )
        two_class_results = acc_pivot_bpb.loc[str(task)].unstack()
        if axes is not None:
            ax: plt.Axes = axes[3, 2]
            plot_task_accuracy(ax, two_class_results, str(task), DDOS_COMPUTE_SIZES, show_legend=True)
            ax.legend(fontsize=6, ncols=2)
            ax.set_ylabel('Decision Acc (BPB on BPB)')
            ax.set_ylim(0.75, 1)

        results.update({
            "two_class_bpb_4M": acc_pivot_bpb['4M'].loc[str(task)].item(),
            "two_class_bpb_20M": acc_pivot_bpb['20M'].loc[str(task)].item(),
            "two_class_bpb_60M": acc_pivot_bpb['60M'].loc[str(task)].item(),
            "two_class_bpb_90M": acc_pivot_bpb['90M'].loc[str(task)].item(),
            "two_class_bpb_150M": acc_pivot_bpb['150M'].loc[str(task)].item(),
            "two_class_bpb_300M": acc_pivot_bpb['300M'].loc[str(task)].item(),
            "two_class_bpb_530M": acc_pivot_bpb['530M'].loc[str(task)].item(),
            "two_class_bpb_750M": acc_pivot_bpb['750M'].loc[str(task)].item(),

            "two_class_acc_4M": acc_pivot_best_metric['4M'].loc[str(task)].item(),
            "two_class_acc_20M": acc_pivot_best_metric['20M'].loc[str(task)].item(),
            "two_class_acc_60M": acc_pivot_best_metric['60M'].loc[str(task)].item(),
            "two_class_acc_90M": acc_pivot_best_metric['90M'].loc[str(task)].item(),
            "two_class_acc_150M": acc_pivot_best_metric['150M'].loc[str(task)].item(),
            "two_class_acc_300M": acc_pivot_best_metric['300M'].loc[str(task)].item(),
            "two_class_acc_530M": acc_pivot_best_metric['530M'].loc[str(task)].item(),
            "two_class_acc_750M": acc_pivot_best_metric['750M'].loc[str(task)].item(),
        })
    except Exception as e:
        print(task, 'failed on consistent ranking analysis', e)
        # raise RuntimeError(task, 'failed on consistent ranking analysis', e)

    if axes is not None:
        for ax in axes.flat:
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)
            ax.set_title(get_title_from_task(task))

            if not ax.has_data():
                ax.remove()

    # Total cost of evaluation
    try:
        task_as_list = [task] if isinstance(task, str) else task
        total_cost = 0
        for subtask in task_as_list:
            task_results = get_slice(df, task=subtask)
            num_instances = task_results['num_instances'].iloc[0]
            eval_cost = num_instances
            assert (task_results['num_instances'] == num_instances).all(), f"num_instances should be constant across task={subtask} for task_as_list={task_as_list}"
            total_cost += eval_cost
        total_cost = int(total_cost)
        results.update({
            "total_cost": total_cost
        })
    except Exception as e:
        print('Failed to calculate compute cost:', e)

    return results