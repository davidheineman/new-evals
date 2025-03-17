import os, sys, itertools
sys.path.append(os.path.dirname(os.getcwd()))
from utils import DATA_DIR, ROOT_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from dataloader import get_slice
from ladder_wrapper import run_ladder
from stats import compute_significance, compute_total_variation, kendall_tau_a
from table import display_task_variants

from ladder_ian import compute_2_class, get_compute, plot_task_accuracy
from utils import get_title_from_task
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


def run_analysis(df, task, ladder_models, external_ladder_models, eval_ladder_models, metric='primary_score', axes=None, small_fig=False, run_irt=False, ladder_config_path=DEFAULT_LADDER_CONFIG_PATH):
    results = {}
    
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
            "rel_error:step_1:7b:bpb_to_primary": rel_error_step_1[0], 
            "rel_error:step_1:13b:bpb_to_primary": rel_error_step_1[1], 
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
            "rel_error:step_2:7b:bpb_to_primary": rel_error_step_2[0], 
            "rel_error:step_2:13b:bpb_to_primary": rel_error_step_2[1], 
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
            "rel_error:stacked:7b:bpb_to_primary": rel_error_stacked[0], 
            "rel_error:stacked:13b:bpb_to_primary": rel_error_stacked[1], 
        })
        if ax:
            # ax.set_ylabel(primary_score_name)
            ax.set_ylabel(metric)
            ax.legend(fontsize=6)
            ax.set_title('Scaling Law Prediction')

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
            "rel_error:step_1:7b:c4_to_primary": rel_error_step_1[0], 
            "rel_error:step_1:13b:c4_to_primary": rel_error_step_1[1], 
            "rel_error:stacked:7b:c4_to_primary": rel_error_stacked[0], 
            "rel_error:stacked:13b:c4_to_primary": rel_error_stacked[1], 
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
                    "rel_error:step_1:7b:bpb_to_irt": rel_error_step_1[0], 
                    "rel_error:step_1:13b:bpb_to_irt": rel_error_step_1[1], 
                    "rel_error:stacked:7b:bpb_to_irt": rel_error_stacked[0], 
                    "rel_error:stacked:13b:bpb_to_irt": rel_error_stacked[1], 
                })
            except Exception as e:
                print(task, 'failed to fit IRT model', e)
    except Exception as e:
        print(task, 'failed on ladder fits', e)
        # raise RuntimeError(task, 'failed on ladder fits', e)

    # intermediate checkpoints
    intermediate_models = ['peteish-moreeval-1B-5xC', 'peteish13-highlr']
    intermediate_model_names = ['1b', '13b']
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
                tv_result = tv[task]['total_variation'] if not isinstance(task, list) else tv.loc
                results.update({
                    f'tv:{additional_metric}:{model_name}': tv_result
                })
            except Exception as e:
                print(task, f'failed to compute decision accuracy for {additional_metric}', e)

    # Consistent rankings analysis
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
    except Exception as e:
        print(task, 'failed on consistent ranking analysis', e)
        # raise RuntimeError(task, 'failed on consistent ranking analysis', e)

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


def run_instance_analysis(df_instances, task, aggregators=['micro', 'macro'], metrics=['logits_per_byte_corr', 'primary_score'], alpha=0.05):
    task_name = get_title_from_task(task)

    # ALPHA = 0.01
    ALPHA = 1e-4
    
    results = {}
    
    for aggregator in aggregators:
        for metric in metrics:
            try:
                models = [model for model in DDOS_MODEL_NAMES if '150M' in model]
                _, out, _ = compute_significance(
                    df_instances, models=models, metric=metric, aggregator=aggregator,
                    step=None, last_n=1, alpha=ALPHA, tasks=[task], quiet=True
                )
                mixes_A, scores_A, p_values_A, sig_clusters_A = out[task_name]

                models = [model for model in DDOS_MODEL_NAMES if '1B' in model]
                _, out, _ = compute_significance(
                    df_instances, models=models, metric=metric, aggregator=aggregator,
                    step=None, last_n=1, alpha=ALPHA, tasks=[task], quiet=True
                )
                mixes_B, scores_B, p_values_B, sig_clusters_B = out[task_name]

                # Compute metric scores
                valid_p_values = p_values_A[~np.isnan(p_values_A)]
                perc_sig_150M = np.sum(valid_p_values <= ALPHA) / len(valid_p_values)

                valid_p_values = p_values_B[~np.isnan(p_values_B)]
                perc_sig_1B = np.sum(valid_p_values <= ALPHA) / len(valid_p_values)

                kt_c = kendall_tau_a(mixes_A, mixes_B, sig_clusters_A, sig_clusters_B)
                
                plt.close()
                
                results.update({
                    "task": task_name,
                    f"kt_c:{metric}": kt_c,
                    f"num_sig_clusters:{metric}:{aggregator}:150M": max(sig_clusters_A),
                    f"num_sig_clusters:{metric}:{aggregator}:1B": max(sig_clusters_B),
                    f"perc_sig:{metric}:{aggregator}:150M": perc_sig_150M,
                    f"perc_sig:{metric}:{aggregator}:1B": perc_sig_1B
                })
            except Exception as e:
                print(task, f'failed to compute significance test for aggregator={aggregator} on metric={metric}', e)
    
    return results
