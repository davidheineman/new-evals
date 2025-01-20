import os
import numpy as np
import pandas as pd
from collections import defaultdict

from olmo.scaling.scaling_laws.utils import FinalConfig

from scaling.step1 import fit_step1, predict_step1, plot_step1, str_chinchilla_n_d_fit
from scaling.step2 import fit_step2, predict_step2, plot_step2
from scaling.predict import predict_chained, plot_chained, str_chained_fit

from scaling.step1_flops import fit_step1 as fit_step1_flops, predict_step1 as predict_step1_flops, plot_step1 as plot_step1_flops, str_chinchilla_flops_fit
from scaling.predict_flops import predict_chained_flops, plot_chained as plot_chained_flops, str_chained_fit as str_chained_fit_flops

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

MODEL_COLORS = {
    '150M': 'r',
    '300M': 'orange',
    '530M': 'green',
    '750M': 'blue',
    '1B': 'purple',
}

def add_ladder_data_cheap_decisions(data_by_name):
    # Manual override for Ian's toks seen
    TOKENS_SEEN = {
        '150M': 15003942912,
        '300M': 30006968320,
        '530M': 53009711104,
        '750M': 75012636672,
        '1B': 100015669248,
    }
    for k, v in data_by_name.items():
        n = TOKENS_SEEN[k]
        data_by_name[k]['ds'] = [n]

    # Taken from: https://github.com/allenai/oe-eval-internal/blob/eval-for-consistent-ranking/experiments/eval-for-consistent-ranking/metrics/scaling.py#L172
    MODEL_PARAMETERS = {
        "150M": 151898880,
        "300M": 319980544,
        "530M": 530074944,
        "750M": 681297408,
        "1B": 1176832000,
    }
    for k, v in data_by_name.items():
        d = TOKENS_SEEN[k]
        f = float(d * MODEL_PARAMETERS[k])
        data_by_name[k]["fs"] = [f]

    compute = 6*n*d

    return data_by_name


def get_slice(df, model, task):
    try:
        df = df.loc[(task, model)]
    except KeyError:
        return df.iloc[0:0]
    df = df.reset_index()
    return df


def get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models, step='max'):
    data_by_name = defaultdict(dict)

    for model in train_models + eval_models:
        if model in eval_models:
            mode = 'eval'
        elif model in train_models:
            mode = 'train'

        split_name = model.split('-')
        size, chinchilla = split_name[-2:]
        group = '-'.join(split_name[:-2])

        is_multiindex = isinstance(df.index, pd.MultiIndex)

        if is_multiindex:
            _slice = get_slice(df, model, task_name)
        else:
            # Filter the DataFrame based on the criteria
            _slice = df[
                (df['task'] == task_name) &
                (df['model'] == model)
                # (df['group'] == group) &
                # (df['size'] == size) &
                # (df['chinchilla'] == chinchilla)
            ]

        if len(_slice) == 0:
            raise RuntimeError(f'Got empty slice for {(task_name, group, size, chinchilla)}.')
        
        if step == 'max':
            # Find the entry with the largest value of 'step'
            max_step_entry = _slice.loc[_slice['step'].idxmax()]
            x_val = max_step_entry[x_metric].tolist()
            y_val = max_step_entry[y_metric].tolist()

            # Remove duplicates
            if isinstance(x_val, list):
                assert len(np.unique(x_val)) == 1
                x_val = x_val[0]
            if isinstance(y_val, list):
                assert len(np.unique(y_val)) == 1
                y_val = y_val[0]

            x_val = [x_val]
            y_val = [y_val]
        else:
            _slice = _slice.sort_values(by='step', ascending=True)
            x_val = _slice[x_metric].tolist()
            y_val = _slice[y_metric].tolist()

            x_val = [x_val]
            y_val = [y_val]
        
        if 'xs' not in data_by_name[size]: data_by_name[size]['xs'] = []
        if 'ys' not in data_by_name[size]: data_by_name[size]['ys'] = []

        data_by_name[size]['xs'] += x_val
        data_by_name[size]['ys'] += y_val
        data_by_name[size]['mode'] = mode

    return data_by_name


def create_ladder_config(task_name, train_models, eval_models, color=None):
    # arc_easy:enlarge => arc_easy
    task_root = task_name.split(':')[0] if isinstance(task_name, str) else None

    # Create config
    configs = {}
    for model in train_models + eval_models:
        size = model.split('-')[-2]
        if color == None: color = MODEL_COLORS.get(size, 'k')
        mode = 'eval' if model in eval_models else 'train'
        
        # Create dummy config for new eval points
        configs[size] = FinalConfig(
            paths=None, mode=mode, n=0, label=size, color=color, use_last_n_percentage=100
        )

    task_key = TASK_KEY_MAP.get(task_root, None) # the task key is used to get min/max perf and plot title

    return task_key, configs


def run_ladder(
        df, task_name, train_models, eval_models, x_metric, y_metric,
        use_flops=False, use_helper_points=False, 
        run_step1=True, run_step2=True, run_stacked=True,
        axes=None, add_texts=False, color=None, return_preds=False):
    # Unfortunately there are local references, so we have to be in the OLMo repo
    os.chdir('/Users/dhei/ai2/new-evals/olmo-repos/OLMo')

    abs_error_step_1, abs_error_step_2, abs_error_stacked = None, None, None

    # Get config
    task_key, configs = create_ladder_config(task_name, train_models, eval_models, color=color)

    # Get data
    # data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models)

    # Use avg of final 10% of checkpoints to fit step 1
    data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models, step='all')
    for k1, v1 in data_by_name.items():
        for k2, v2 in v1.items():
            if isinstance(v2, list):
                import math
                data_by_name[k1][k2] = [np.mean(v2[0][math.ceil(0.9 * len(v2[0])):])]

    # which functional form to use for step 1 prediction
    if 'byte' in x_metric:
        y_metric_func = 'rc_bpb'
    else:
        y_metric_func = 'rc_acc'

    assert len(data_by_name) != 0, train_models

    ax_i = 0
    if run_step1 or run_stacked:
        # Add token data
        data_by_name = add_ladder_data_cheap_decisions(data_by_name)
        
        # Fit step 1
        if use_flops:
            step1_coefficients, cov = fit_step1_flops(data_by_name, y_metric_func)
        else:
            step1_coefficients, cov = fit_step1(data_by_name, y_metric_func)

        if use_flops:
            (
                predicted_data_by_name, plotted_predicted_data,
                (step_1_y, step_1_y_pred, rel_error_step_1), all_rel_errors,
            ) = predict_step1_flops(
                configs, data_by_name, step1_coefficients, y_metric=y_metric_func, 
            )
        else:
            (
                predicted_data_by_name, plotted_predicted_data,
                (step_1_y, step_1_y_pred, rel_error_step_1), all_rel_errors,
            ) = predict_step1(
                configs, data_by_name, step1_coefficients, y_metric=y_metric_func, 
            )
        abs_error_step_1 = abs(step_1_y_pred - step_1_y)

        # Plot step 1
        if axes is not None and run_step1:
            ax = axes[ax_i]
            ax_i += 1
            if use_flops:
                plot_step1_flops(
                    configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    task_name, str_chinchilla_flops_fit(step1_coefficients), y_metric_func,
                    step1_coefficients, cov, ax,
                )
            else:
                plot_step1(
                    configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    task_name, str_chinchilla_n_d_fit(step1_coefficients), y_metric_func,
                    step1_coefficients, cov, ax,
                )
            ax.set_ylabel(x_metric)

    # Use intermediate checkpoints to fit step 2
    data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models, step='all')
    for k1, v1 in data_by_name.items():
        for k2, v2 in v1.items():
            if isinstance(v2, list):
                import math
                # use last 90% of checkpoints
                data_by_name[k1][k2] = v2[0][math.ceil(0.1 * len(v2[0])):]

    if run_step2 or run_stacked:
        task_key, configs = create_ladder_config(task_name, train_models, eval_models, color=color)

        _min, _max = None, None
        if task_key is None and use_helper_points:
            _min, _max = 0, 1 # TODO: Use utils.constants_task to get correct values

        # Fit step 2
        step2_coefficients, cov = fit_step2(data_by_name, task_key, y_metric=None, _min=_min, _max=_max, use_log_sigmoid=False, use_helper_points=use_helper_points)

        (
            predicted_data_by_name, plotted_predicted_data,
            (step_2_y, step_2_y_pred, rel_error_step_2, delta_error), all_rel_errors,
        ) = predict_step2(
            configs, data_by_name, step2_coefficients, cov, y_metric=None, use_log_sigmoid=False
        )
        abs_error_step_2 = abs(step_2_y_pred - step_2_y)

        # Plot step 2
        if axes is not None and run_step2:
            ax = axes[ax_i]
            ax_i += 1
            plot_step2(
                configs, data_by_name, predicted_data_by_name, plotted_predicted_data, task_key, None, y_metric_func, 'rc_acc',
                step2_coefficients, cov, use_log_sigmoid=False, add_texts=add_texts, ax=ax
            )
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric)
        
    # Get step 1 data again (necessary if running with intermediate checkpoints)
    data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models)
    data_by_name = add_ladder_data_cheap_decisions(data_by_name)
    
    if run_stacked:
        # Predict stacked
        if use_flops:
            (
                predicted_data_by_name, plotted_predicted_data_by_name, 
                (stacked_y, stacked_y_pred, rel_error_stacked)
            ) = predict_chained_flops(
                data_by_name, step1_coefficients, step2_coefficients
            )
        else:
            (
                predicted_data_by_name, plotted_predicted_data_by_name, 
                (stacked_y, stacked_y_pred, rel_error_stacked)
            ) = predict_chained(
                data_by_name, step1_coefficients, step2_coefficients, use_log_sigmoid=False
            )
        abs_error_stacked = abs(stacked_y_pred - stacked_y)

        # For stacked predictions, the x axis is now the y axis
        for key in data_by_name:
            data_by_name[key]['xs'] = data_by_name[key]['ys']

        # Plot stacked prediction
        if axes is not None:
            ax = axes[ax_i]
            if use_flops:
                plot_chained_flops(
                    configs,
                    data_by_name,
                    predicted_data_by_name,
                    plotted_predicted_data_by_name,
                    task_name,
                    str_chained_fit_flops(step1_coefficients, step2_coefficients),
                    ax,
                )
            else:
                plot_chained(
                    configs,
                    data_by_name,
                    predicted_data_by_name,
                    plotted_predicted_data_by_name,
                    task_name,
                    str_chained_fit(step1_coefficients, step2_coefficients, use_log_sigmoid=False),
                    ax,
                )
            ax.legend(loc='upper left')
            ax.set_ylabel(y_metric)

    if axes is not None:
        for ax in axes:
            ax.set_title(task_name)
            ax.legend(fontsize=8)

    if return_preds:
        return (abs_error_step_1, abs_error_step_2, abs_error_stacked), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    return abs_error_step_1, abs_error_step_2, abs_error_stacked
