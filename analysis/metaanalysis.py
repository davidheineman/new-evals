from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os, sys, itertools
sys.path.append(os.path.dirname(os.getcwd()))
from utils import DATA_DIR, PLOT_DIR, ROOT_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from tqdm import tqdm

from dataloader import get_slice, get_nd_array
from ladder_wrapper import run_ladder
from stats import compute_significance, compute_total_variation, kendall_tau_a
from utils.power import calculate_mde
from table import display_task_variants

from ladder_ian import compute_2_class, get_compute, plot_task_accuracy
from utils import get_title_from_task, extract_size
from utils.constants_models import DDOS_MODEL_NAMES
from download.utils_cheap_decisions import PRIMARY_METRICS_OLMES

from ladder_wrapper import sort_experiment_names
from download.preprocess import is_excluded_from_lite
from db_duck import connect_db_backend

DEFAULT_LADDER_CONFIG_PATH = f'{ROOT_DIR}/analysis/utils/ladder_config.json'

ALL_METRICS = ['logits_per_char_corr', 'primary_score']
# REVERSED_METRICS = ['margin_per_byte', 'norm_correct_prob_per_byte', 'correct_prob_per_byte', 'correct_logit_per_byte', 'logits_per_char_corr', 'logits_per_byte_corr']
REVERSED_METRICS = ['margin_per_byte', 'norm_correct_prob_per_byte', 'correct_prob_per_byte', 'correct_logit_per_byte', 'logits_per_byte_corr']

DDOS_SIZES = ['4M', '20M', '60M', '90M', '150M', '300M', '530M', '750M', '1B']
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


def get_df_benchmarks_subset(df_instances: pd.DataFrame, n_instances: int):
    """ Compute benchmark averages for a random subset of instances """
    # Sample n instances from each group
    df_instances_subset = df_instances.groupby(['task', 'model', 'step', 'mix'], dropna=False, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n_instances), random_state=42)
    )

    # Compute aggregate metrics
    df_benchmarks_subset = df_instances_subset.groupby(level=['task', 'model', 'step', 'mix'], dropna=False).agg({
        'primary_score': 'mean',
        'logits_per_byte_corr': 'mean',
        'logits_per_char_corr': 'mean',
        'size': 'first', 
        'token_ratio': 'first'
    }).reset_index()
    
    # Get actual number of instances for each group
    instance_counts = df_instances_subset.groupby(['task', 'model', 'step', 'mix'], dropna=False).size().reset_index(name='num_instances')
    df_benchmarks_subset = df_benchmarks_subset.merge(instance_counts, on=['task', 'model', 'step', 'mix'])

    return df_instances_subset, df_benchmarks_subset


def assert_same_models(df_instances: pd.MultiIndex, df_benchmarks: pd.DataFrame):
    ''' Assert the model sets are the same for different df types '''
    MODELS_BENCHMARKS = list(df_benchmarks['model'].unique())
    MODELS_INSTANCES = df_instances.index.get_level_values('model').unique().to_list()

    benchmarks_set = set(MODELS_BENCHMARKS)
    instances_set = set(MODELS_INSTANCES)

    assert len(benchmarks_set - instances_set) == 0, f"Found models in BENCHMARKS but not in INSTANCES: {benchmarks_set - instances_set}"
    assert len(instances_set - benchmarks_set) == 0, f"Found models in INSTANCES but not in BENCHMARKS: {instances_set - benchmarks_set}"


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
        if _slice.empty:
            raise RuntimeError(f"Empty slice for metric={metric}, size={size}, task={task}")
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


def set_title_from_task(ax: plt.Axes, task):
    ax.set_title(get_title_from_task(task))


def get_task_correlations(df_benchmarks, selected_tasks, pred_metric='logits_per_char_corr', target_metric='primary_score'):
    """Calculate correlation matrix between tasks based on how well models rank on pred_metric vs target_metric."""
    # Get model names from df_benchmarks
    models = sorted(list(df_benchmarks['model'].unique()))
    ladder_models = [model for model in models if "peteish-moreeval" in model]
    ladder_models = sort_experiment_names(ladder_models)
    llama_3_models = [model for model in models if "Llama-3" in model]
    external_models = sorted([
        model for model in models 
        if model not in
            DDOS_MODEL_NAMES + # exclude 1B-5xC models
            ladder_models + # exclude ladder models
            ['peteish13-highlr'] # exclude intermediate checkpoints from 13B
        and not is_excluded_from_lite(model)
    ])
    
    # Get data slice for analysis
    flattened_tasks = [subtask for task in selected_tasks for subtask in (task if isinstance(task, list) else [task])]
    _slice = get_slice(df_benchmarks, model=external_models, task=flattened_tasks)

    # Pre-compute scores dictionary for better performance
    pred_scores_dict = {}
    target_scores_dict = {}
    for task in selected_tasks:
        if isinstance(task, list):
            # For task lists, calculate average score across all subtasks
            task_name = get_title_from_task(task)
            pred_task_scores = []
            target_task_scores = []
            for subtask in task:
                subtask_pred = _slice[_slice['task'] == subtask].set_index('model')[pred_metric]
                subtask_target = _slice[_slice['task'] == subtask].set_index('model')[target_metric]
                if not subtask_pred.empty and not subtask_target.empty:
                    pred_task_scores.append(subtask_pred)
                    target_task_scores.append(subtask_target)
            
            if pred_task_scores and target_task_scores:
                pred_scores_dict[task_name] = pd.concat(pred_task_scores, axis=1).mean(axis=1)
                target_scores_dict[task_name] = pd.concat(target_task_scores, axis=1).mean(axis=1)
        else:
            # Negate scores for paloma tasks (lower is better)
            if task.startswith('paloma_'):
                pred_scores = -_slice[_slice['task'] == task].set_index('model')[pred_metric]
                target_scores = -_slice[_slice['task'] == task].set_index('model')[target_metric]
            else:
                pred_scores = _slice[_slice['task'] == task].set_index('model')[pred_metric]
                target_scores = _slice[_slice['task'] == task].set_index('model')[target_metric]
            pred_scores_dict[task] = pred_scores
            target_scores_dict[task] = target_scores

    # Get task names for display
    task_names = [get_title_from_task(task) if isinstance(task, list) else task for task in selected_tasks]
    n_tasks = len(selected_tasks)

    # Initialize correlation matrix
    corr_matrix = np.zeros((n_tasks, n_tasks))

    # Calculate correlations
    for i in tqdm(range(n_tasks)):
        task1 = selected_tasks[i]
        task1_name = task_names[i]
        task1_pred = pred_scores_dict[task1_name if isinstance(task1, list) else task1]
        
        # Only compute upper triangle for efficiency
        for j in range(i, n_tasks):
            task2 = selected_tasks[j]
            task2_name = task_names[j]
            task2_target = target_scores_dict[task2_name if isinstance(task2, list) else task2]
            
            # Find models with scores for both tasks
            common_models = task1_pred.index.intersection(task2_target.index)
            
            if len(common_models) > 1:
                # Get scores for common models
                pred1 = task1_pred[common_models].dropna()
                target2 = task2_target[common_models].dropna()
                
                # Find common models after dropping NaN values
                valid_models = pred1.index.intersection(target2.index)
                
                if len(valid_models) > 1:
                    # Calculate correlation between rankings on pred_metric for task1 vs target_metric for task2
                    tau, _ = stats.kendalltau(pred1[valid_models], target2[valid_models])
                    tau = abs(tau) # lazy way to deal with inverted metrics
                    corr_matrix[i,j] = corr_matrix[j,i] = tau

    return corr_matrix, task_names


def run_analysis(df, task, ladder_models, external_ladder_models, eval_ladder_models, metric='primary_score', axes=None, small_fig=False, run_irt=False, ladder_config_path=DEFAULT_LADDER_CONFIG_PATH):
    results = {}

    # Observational noise
    observational_models = eval_ladder_models+DDOS_MODEL_NAMES
    _slice = get_slice(df, task=task, model=observational_models)
    numerical_cols     = [col for col in _slice.select_dtypes(include='number').columns if col != 'extracted_size']
    non_numerical_cols = _slice.select_dtypes(exclude='number').columns.tolist() + ['extracted_size']
    _slice = _slice.groupby('model', as_index=False).agg({col: 'mean' for col in numerical_cols} | {col: 'first' for col in non_numerical_cols})
    weight_classes = [
        {
            'label': '7B',
            'weight_range': (6_000_000_000, 8_000_000_000)
        },
        {
            'label': '13B',
            'weight_range': (12_000_000_000, 14_000_000_000)
        }
    ]
    observational_metrics = ['primary_score', 'logits_per_char_corr']
    for observational_metric in observational_metrics:
        for weight_class in weight_classes:
            size_label = weight_class['label']
            weight_min, weight_max = weight_class['weight_range']

            _slice['extracted_size'] = pd.to_numeric(_slice['extracted_size'], errors='coerce').fillna(_slice['extracted_size']).astype('Int64')
            _weight_class_scores = _slice[(_slice['extracted_size'] >= weight_min) & (_slice['extracted_size'] <= weight_max)][observational_metric]

            results.update({
                f'mean:{observational_metric}:{size_label}': _weight_class_scores.mean(),
                f'range:{observational_metric}:{size_label}': _weight_class_scores.max() - _weight_class_scores.min(),
                f'std_dev:{observational_metric}:{size_label}': _weight_class_scores.std()
            })
    
    # Scaling laws
    primary_score_name = PRIMARY_METRICS_OLMES[task] if isinstance(task, str) and task in PRIMARY_METRICS_OLMES else 'primary_score'
    try:
        # Step 1 ladder prediction (base models)
        ax = None
        if not small_fig:
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
            "rel_error:step_1:7B:bpb_to_primary": rel_error_step_1[0], 
            "rel_error:step_1:13B:bpb_to_primary": rel_error_step_1[1], 
        })
        if ax:
            ax.set_ylabel('Task loss (BPB)')
            ax.legend(fontsize=6)

        # Step 2 ladder prediction (base models)
        ax = None
        if not small_fig:
            ax: plt.Axes = axes[1, 0] if axes is not None else None
        _, rel_error_step_2, _ = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish7", "peteish13-highlr"],
            downstream_feature=metric,
            config_path=ladder_config_path,
            run_step1=False, run_stacked=False,
            axes=[ax]
        )
        results.update({
            "rel_error:step_2:7B:bpb_to_primary": rel_error_step_2[0], 
            "rel_error:step_2:13B:bpb_to_primary": rel_error_step_2[1], 
        })
        if ax:
            ax.set_xlabel('Task loss (BPB)')
            # ax.set_ylabel(primary_score_name)
            ax.set_ylabel(metric)
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
            "mean_error:step_2:external:bpb_to_primary": mean_error_step_2, 
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
            ax.set_title('Perplexity -> Task Metric')

        # Stacked ladder prediction
        ax: plt.Axes = axes[3, 0] if axes is not None else None
        _, _, rel_error_stacked = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish7", "peteish13-highlr"],
            downstream_feature=metric,
            config_path=ladder_config_path,
            run_step1=False, run_step2=False,
            axes=[ax]
        )
        results.update({
            "rel_error:stacked:7B:bpb_to_primary": rel_error_stacked[0], 
            "rel_error:stacked:13B:bpb_to_primary": rel_error_stacked[1], 
        })
        if ax:
            # ax.set_ylabel(primary_score_name)
            ax.set_ylabel(metric)
            ax.legend(fontsize=6)
            ax.set_title('Scaling Law Prediction')
        
        # fig, ax = plt.subplots(figsize=(10, 6))
        # plt.savefig(os.path.join(PLOT_DIR, f'debug:{task}:{metric}.pdf'))
        # plt.close()

        # Stacked prediction -- C4 as intermediate feature
        rel_error_step_1, _, rel_error_stacked = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish7", "peteish13-highlr"],
            # Use C4 loss for intermediate feature!
            intermediate_task_name="paloma_c4_en",
            intermediate_feature='logits_per_byte_corr', 
            downstream_feature=metric, # 'primary_score', 
            config_path=ladder_config_path,
        )
        results.update({
            "rel_error:step_1:7B:c4_to_primary": rel_error_step_1[0], 
            "rel_error:step_1:13B:c4_to_primary": rel_error_step_1[1], 
            "rel_error:stacked:7B:c4_to_primary": rel_error_stacked[0], 
            "rel_error:stacked:13B:c4_to_primary": rel_error_stacked[1], 
        })

        if run_irt:
            try:
                # Stacked prediction -- BPB -> IRT ability
                rel_error_step_1, _, rel_error_stacked = run_ladder(
                    df,
                    task, 
                    train_models=ladder_models,
                    eval_models=["peteish7", "peteish13-highlr"],
                    intermediate_task_name=task,
                    # intermediate_feature="logits_per_byte_corr",
                    downstream_feature="theta_primary_score", # theta_bpb, theta_primary_score
                    config_path=ladder_config_path,
                )
                results.update({
                    "rel_error:step_1:7B:bpb_to_irt": rel_error_step_1[0], 
                    "rel_error:step_1:13B:bpb_to_irt": rel_error_step_1[1], 
                    "rel_error:stacked:7B:bpb_to_irt": rel_error_stacked[0], 
                    "rel_error:stacked:13B:bpb_to_irt": rel_error_stacked[1], 
                })
            except Exception as e:
                print(task, 'failed to fit IRT model', e)
    except Exception as e:
        print(task, 'failed on ladder fits', e)
        # raise RuntimeError(task, 'failed on ladder fits', e)

    # Step-to-step noise
    intermediate_models = ['peteish-moreeval-1B-5xC', 'peteish13-highlr']
    intermediate_model_names = ['1B', '13B']
    for j, model in enumerate(intermediate_models):
        model_name = intermediate_model_names[j]

        # logits_per_char_corr intermediate checkpoinrts
        if small_fig:
            ax: plt.Axes = axes[2+j, 1] if axes is not None else None
        else:
            ax: plt.Axes = axes[0+(j*2), 1] if axes is not None else None
        tv, _ = compute_total_variation(
            df, models=[model], metric='logits_per_char_corr', tasks=[task], axes=[ax]
        )
        tv_bpb = tv[task]['total_variation'] if not isinstance(task, list) else tv.loc['total_variation']['aggregate']
        if ax and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=6)
            ax.set_ylabel('Task loss (BPB)')
            ax.set_title('Smoothness')
            
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

        results.update({
            f'tv:logits_per_char_corr:{model_name}': tv_bpb
        })

        # primary_metric intermediate checkpoinrts
        ax = None
        if not small_fig:
            ax: plt.Axes = axes[1+(j*2), 1] if axes is not None else None
        tv, _ = compute_total_variation(
            df, models=[model], metric=metric, tasks=[task], axes=[ax]
        )
        tv_primary = tv[task]['total_variation'] if not isinstance(task, list) else tv.loc['total_variation']['aggregate']
        if ax and ax.get_legend_handles_labels()[1]:
            # ax.set_ylabel(primary_score_name)
            ax.set_ylabel(metric)
            ax.legend(fontsize=6)

        results.update({
            f'tv:{metric}:{model_name}': tv_primary
        })

        # Additional metric calculations
        additional_metrics = ['primary_score', 'logits_per_char_corr']
        if run_irt: 
            additional_metrics += ['theta_bpb', 'theta_primary_score']
        for additional_metric in additional_metrics:
            try:
                tv, _ = compute_total_variation(
                    df, models=[model], metric=additional_metric, tasks=[task]
                )
                tv_result = tv[task]['total_variation'] if not isinstance(task, list) else tv.loc['total_variation']['aggregate']
                results.update({
                    f'tv:{additional_metric}:{model_name}': tv_result
                })
            except Exception as e:
                print(task, f'failed to compute decision accuracy for {additional_metric}', e)

    # Decision accuracy
    try:
        two_class, acc_pivot_bpb_primary, metric_pivot = construct_2class_table(
            df, [task], small_metric='logits_per_byte_corr', target_metric='primary_score'
        )
        two_class_results = acc_pivot_bpb_primary.loc[str(task)].unstack()
        if axes is not None and not small_fig:
            ax: plt.Axes = axes[1, 2]
            plot_task_accuracy(ax, two_class_results, str(task), DDOS_COMPUTE_SIZES)
            ax.set_ylabel(f'Decision Acc (BPB on {primary_score_name})')
            ax.set_ylim(0.75, 1)

        two_class, acc_pivot_best_metric, metric_pivot = construct_2class_table(
            df, [task], small_metric=metric, target_metric=metric
        )
        two_class_results = acc_pivot_best_metric.loc[str(task)].unstack()
        if axes is not None and not small_fig:
            ax: plt.Axes = axes[2, 2]
            plot_task_accuracy(ax, two_class_results, str(task), DDOS_COMPUTE_SIZES)
            ax.set_ylabel(f'Decision Acc (best on {primary_score_name})')
            ax.set_ylim(0.75, 1)

        results.update({
            f"dec_acc:{metric}:4M": acc_pivot_best_metric['4M'].loc[str(task)].item(),
            f"dec_acc:{metric}:20M": acc_pivot_best_metric['20M'].loc[str(task)].item(),
            f"dec_acc:{metric}:60M": acc_pivot_best_metric['60M'].loc[str(task)].item(),
            f"dec_acc:{metric}:90M": acc_pivot_best_metric['90M'].loc[str(task)].item(),
            f"dec_acc:{metric}:150M": acc_pivot_best_metric['150M'].loc[str(task)].item(),
            f"dec_acc:{metric}:300M": acc_pivot_best_metric['300M'].loc[str(task)].item(),
            f"dec_acc:{metric}:530M": acc_pivot_best_metric['530M'].loc[str(task)].item(),
            f"dec_acc:{metric}:750M": acc_pivot_best_metric['750M'].loc[str(task)].item(),
        })
            

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
            ax.set_title('Decision Accuracy')

        results.update({
            "dec_acc:logits_per_byte_corr:4M": acc_pivot_bpb['4M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:20M": acc_pivot_bpb['20M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:60M": acc_pivot_bpb['60M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:90M": acc_pivot_bpb['90M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:150M": acc_pivot_bpb['150M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:300M": acc_pivot_bpb['300M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:530M": acc_pivot_bpb['530M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:750M": acc_pivot_bpb['750M'].loc[str(task)].item(),
        })

        additional_metrics = ['primary_score', 'logits_per_char_corr']
        if run_irt: 
            additional_metrics += ['theta_bpb', 'theta_primary_score']
        for additional_metric in additional_metrics:
            two_class, acc_pivot_bpb, metric_pivot = construct_2class_table(
                df, [task], 
                small_metric=additional_metric, target_metric=additional_metric
            )
            two_class_results = acc_pivot_bpb.loc[str(task)].unstack()
            results.update({
                f"dec_acc:{additional_metric}:4M": acc_pivot_bpb['4M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:20M": acc_pivot_bpb['20M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:60M": acc_pivot_bpb['60M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:90M": acc_pivot_bpb['90M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:150M": acc_pivot_bpb['150M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:300M": acc_pivot_bpb['300M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:530M": acc_pivot_bpb['530M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:750M": acc_pivot_bpb['750M'].loc[str(task)].item(),
            })

        # Compute range and std dev between models at each compute scale
        for additional_metric in additional_metrics:
            for size in DDOS_SIZES:
                scores = get_perf_size(df, size, task, additional_metric)[additional_metric]
                results.update({
                    f'mean:{additional_metric}:{size}': scores.mean(),
                    f'range:{additional_metric}:{size}': scores.max() - scores.min(),
                    f'std_dev:{additional_metric}:{size}': scores.std()
                })
    except Exception as e:
        # print(task, 'failed on consistent ranking analysis', e)
        raise RuntimeError(task, 'failed on consistent ranking analysis', e)

    if axes is not None:
        for ax in axes.flat:
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)
            if not small_fig:
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
            "total_cost": total_cost,
            "total_cost_div_4": total_cost / 4 # Hacky way to estimate BPB vs. RC cost (assumes all tasks have 4 answer choices)
        })
    except Exception as e:
        print('Failed to calculate compute cost:', e)

    return results


def compute_instance_analysis(
    df_instances, 
    task, 
    aggregators=['micro', 'macro'], 
    metrics=['logits_per_byte_corr', 'primary_score'], 
    sizes=DDOS_SIZES, # ['4M', '20M', ..., '750M', '1B'],
    alpha=1e-4, # 0.05
    target_power=0.8,
    quiet=False
    ):
    task_name = get_title_from_task(task)

    if isinstance(df_instances, str):
        df_instances = connect_db_backend(df_instances)
    
    results = {}
    
    for aggregator in aggregators:
        for metric in metrics:
            for size in sizes:
                for binarize in [False, True]:
                    try:
                        models = [model for model in DDOS_MODEL_NAMES if size in model] # e.g., 150M
                        _, out, _ = compute_significance(
                            df_instances, models=models, metric=metric, aggregator=aggregator,
                            step=None, last_n=1, alpha=alpha, tasks=[task], binarize=binarize, quiet=True
                        )
                        plt.close()
                        mixes_A, scores_A, p_values_A, sig_clusters_A = out[task_name]

                        if isinstance(p_values_A, float) and p_values_A == float('-inf'):
                            # If we cannot binarize scores, return
                            continue

                        # Compute metric scores
                        valid_p_values = p_values_A[~np.isnan(p_values_A)]
                        perc_sig = np.sum(valid_p_values <= alpha) / len(valid_p_values)

                        results.update({
                            "task": task_name,
                            f"num_sig_clusters:{metric}:{aggregator}:{size}:{('binary' if binarize else 'non_binary')}": max(sig_clusters_A),
                            f"perc_sig:{metric}:{aggregator}:{size}:{('binary' if binarize else 'non_binary')}": perc_sig,
                        })
                    except Exception as e:
                        raise RuntimeError(task_name, f'failed to compute significance test for aggregator={aggregator} on metric={metric}', e)
    
    # Compute instance-level agreement rate
    aggregator = 'micro'
    for metric in metrics:
        for size in sizes:
            instance_names, mixes, scores = get_nd_array(
                df_instances, 'model', 
                metric=metric, 
                task=task, 
                model=[m for m in DDOS_MODEL_NAMES if size in m],
                return_index=True
            )

            n_models, n_instances = scores.shape

            # Hard agreement - exact matches
            matches = scores[:, None, :] == scores[None, :, :]  # Shape: (n_models, n_models, n_instances)
            mask = np.triu(np.ones((n_models, n_models)), k=1)  # Upper triangular mask to avoid duplicates
            exact_agreement_rate = np.mean(matches[mask.astype(bool)])

            # Soft agreement - average diff
            diffs = np.abs(scores[:, None, :] - scores[None, :, :])  # Shape: (n_models, n_models, n_instances) 
            max_diff = np.max(np.abs(scores))
            normalized_diffs = 1 - (diffs / max_diff)
            soft_agreement_rate = np.mean(normalized_diffs[mask.astype(bool)])

            results.update({
                f'hard_agreement:{metric}:{aggregator}:{size}': exact_agreement_rate,
                f'soft_agreement:{metric}:{aggregator}:{size}': soft_agreement_rate
            })

            # Compute Fisher information using IRT scores
            irt_path = Path(DATA_DIR) / "irt" / f"{task_name}.json"
            if irt_path.exists():
                try:
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/irt') # Add IRT code to PATH
                    from irt_utils.irt_inference import load_irt_params, test_information
                    from stats import compute_irt

                    train_instance_names, discriminations, difficulties = load_irt_params(
                        load_path=irt_path,
                    )
                    irt_params = (difficulties, discriminations, train_instance_names)

                    thetas = compute_irt(irt_params, instance_names, scores, metric)
                    thetas = thetas.tolist()
                    tif = test_information(thetas, discriminations, difficulties)
                    avg_tif = np.mean(tif)
                    
                    results.update({
                        f'mean_information:{metric}:{aggregator}:{size}': avg_tif
                    })
                except Exception as e:
                    print(f'failed to compute fisher information for task_name={task_name} aggregator={aggregator} on metric={metric}: {e}')

            continue

            # Check if results are {0, 1}. If results are [0, 1], then we binarize
            binary_scores = scores
            is_binary = np.all(np.logical_or(scores == 0, scores == 1))
            if not is_binary and np.all((scores >= 0) & (scores <= 1)):
                binary_scores = (scores > 0.5).astype(float) # binarize with threshold 0.5
                is_binary = True

            if is_binary:
                # Compute MDE in parallel (warning: causes lots of deadlocks)
                n_models, n_instances = binary_scores.shape
                mdes = np.full((n_models, n_models), np.nan)

                args = []
                for i in range(n_models):
                    for j in range(i+1, n_models):
                        baseline_acc = np.mean(binary_scores[i])
                        agreement_rate = np.mean(binary_scores[i] == binary_scores[j])
                        args.append((
                            baseline_acc,
                            agreement_rate, 
                            n_instances, 
                            target_power # default=0.8
                        ))

                from utils.power import calculate_mde # need to reimport for pickle
                with ProcessPoolExecutor() as executor:
                    mde_resp = list(tqdm(
                        executor.map(calculate_mde, *zip(*args)),
                        total=len(args),
                        desc="Computing MDE",
                        disable=quiet
                    ))

                idx = 0
                for i in range(n_models):
                    for j in range(i+1, n_models):
                        mdes[i,j] = mde_resp[idx]
                        idx += 1
                
                mean_mde = np.nanmean(mdes)

                results.update({
                    f'mean_mde_binary:{metric}:{aggregator}:{size}': mean_mde
                })

    return results




def compute_metaproperties(df_benchmarks, df_instances, selected_tasks, run_irt=False, run_instance_analysis=False, quiet=False):
    ALPHA=1e-4

    task_names = [get_title_from_task(task) for task in selected_tasks]

    # Get model names from df_benchmarks
    models = sorted(list(df_benchmarks['model'].unique()))
    ladder_models = [model for model in models if "peteish-moreeval" in model]
    ladder_models = sort_experiment_names(ladder_models)
    llama_3_models = [model for model in models if "Llama-3" in model]
    external_models = sorted([
        model for model in models 
        if model not in
            DDOS_MODEL_NAMES + # exclude 1B-5xC models
            ladder_models + # exclude ladder models
            ['peteish13-highlr'] # exclude intermediate checkpoints from 13B
        and not is_excluded_from_lite(model)
    ])

    # Add extracted size
    df_benchmarks['extracted_size'] = df_benchmarks['model'].apply(extract_size)

    benchmark_results = []
    instance_results = []

    # Run benchmark analysis
    benchmark_args = []
    for task in selected_tasks:
        benchmark_args.append({
            'df': df_benchmarks,
            'task': task,
            'ladder_models': ladder_models,
            'eval_ladder_models': ladder_models + llama_3_models,
            'external_ladder_models': external_models,
            'run_irt': run_irt
        })
    
    with ProcessPoolExecutor() as executor:
        futures = []
        for kwargs in benchmark_args:
            futures.append(executor.submit(run_analysis, **kwargs))
        
        benchmark_results = list(tqdm(
            (f.result() for f in futures),
            total=len(benchmark_args),
            desc="Computing benchmark properties"
        ))

    # Run instance analysis
    if run_instance_analysis:
        instance_args = []
        for task in selected_tasks:
            aggregators = ['micro', 'macro', 'irt'] if run_irt else ['micro', 'macro']
            instance_args.append({
                'df_instances': df_instances,
                'task': task,
                'aggregators': aggregators,
                'alpha': ALPHA,
                'quiet': quiet
            })

        with ProcessPoolExecutor() as executor:
            futures = []
            for kwargs in instance_args:
                futures.append(executor.submit(compute_instance_analysis, **kwargs))
            
            instance_results = list(tqdm(
                (f.result() for f in futures),
                total=len(instance_args),
                desc="Computing instance properties"
            ))

    # Create dataframe, filling in missing results as -inf
    all_keys = set().union(*benchmark_results)
    normalized_results = [{key: d.get(key, float('-inf')) for key in all_keys} for d in benchmark_results]
    df_benchmark_results = pd.DataFrame(normalized_results, index=task_names)
    df_instance_results = pd.DataFrame(instance_results, index=task_names)
    df_results = pd.concat([df_benchmark_results, df_instance_results], axis=1)

    # Remove duplicate results if they exist
    n_duplicates = len(df_results.index) - len(df_results.index.unique())
    if n_duplicates > 0:
        print(f"Removing {n_duplicates} duplicates")
        df_results = df_results[~df_results.index.duplicated()]

    return df_results
