import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from olmo.scaling.scaling_laws.utils import FinalConfig

from scaling.step1 import fit_step1, predict_step1, plot_step1, str_chinchilla_n_d_fit
from scaling.step2 import fit_step2, predict_step2, plot_step2
from scaling.predict import predict_chained, plot_chained, str_chained_fit

from scaling.step1_flops import fit_step1 as fit_step1_flops, predict_step1 as predict_step1_flops, plot_step1 as plot_step1_flops, str_chinchilla_flops_fit
from scaling.predict_flops import predict_chained_flops, plot_chained as plot_chained_flops, str_chained_fit as str_chained_fit_flops
from scaling.single_step import fit_single_step, predict_single_step, plot_single_step, str_combined_fit

from concurrent.futures import ProcessPoolExecutor

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

FULL_SCHEDULE = {
    '150M': 38157,
    '300M': 45787,
    '530M': 57786,
    '750M': 63589,
    '1B': 69369,
}

MODEL_TO_BATCH = {
    '150M': 192,
    '300M': 320,
    '530M': 448,
    '750M': 576,
    '1B': 704
}

MODEL_TO_PARAMETERS = {
    '150M': 151898880,
    '300M': 319980544,
    '530M': 530074944,
    '750M': 681297408,
    '1B': 1176832000
}


def get_compute(scale):
    return 2048 * 6 * MODEL_TO_BATCH[scale] * MODEL_TO_PARAMETERS[scale] * FULL_SCHEDULE[scale]


def add_ladder_data_cheap_decisions(data_by_name):
    """ From Ian """
    sequence_length = 2048

    def model_and_step_to_tokens(model, step):
        return MODEL_TO_BATCH[model] * step * sequence_length

    def model_and_step_to_compute(model, step):
        return MODEL_TO_PARAMETERS[model] * model_and_step_to_tokens(model, step) * 6
    
    for k, v in data_by_name.items():
        step = v['step'][-1]
        c = model_and_step_to_compute(k, step)
        n = MODEL_TO_PARAMETERS[k]
        d = model_and_step_to_tokens(k, step)
        f = float(n * d * 6)
        data_by_name[k]['ns'] = [n]
        data_by_name[k]['fs'] = [f]
        data_by_name[k]["ds"] = [d]

    # raise RuntimeError(data_by_name)

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
            step_val = _slice['step'].max().tolist()

            # Remove duplicates
            if isinstance(x_val, list):
                assert len(np.unique(x_val)) == 1
                x_val = x_val[0]
            if isinstance(y_val, list):
                assert len(np.unique(y_val)) == 1
                y_val = y_val[0]
            if isinstance(step_val, list):
                assert len(np.unique(step_val)) == 1
                step_val = step_val[0]

            x_val = [x_val]
            y_val = [y_val]
            step_val = [step_val]
        else:
            _slice = _slice.sort_values(by='step', ascending=True)
            x_val = _slice[x_metric].tolist()
            y_val = _slice[y_metric].tolist()
            step_val = _slice['step'].tolist()

            x_val = [x_val]
            y_val = [y_val]
            step_val = [step_val]
        
        if 'xs' not in data_by_name[size]: data_by_name[size]['xs'] = []
        if 'ys' not in data_by_name[size]: data_by_name[size]['ys'] = []
        if 'step' not in data_by_name[size]: data_by_name[size]['step'] = []

        data_by_name[size]['step'] += step_val
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
        use_flops=False, use_single_step=False, use_two_param=False, use_helper_points=False, 
        last_perc_step_2=0.9,
        run_step1=True, run_step2=True, run_stacked=True,
        axes=None, add_texts=False, color=None, return_preds=False, return_reals=False, return_coeff=False):
    # Unfortunately there are local references, so we have to be in the OLMo repo
    os.chdir('/Users/dhei/ai2/new-evals/olmo-repos/OLMo')

    abs_error_step_1, abs_error_step_2, abs_error_stacked = None, None, None
    step_1_y_pred, step_2_y_pred, stacked_y_pred = None, None, None
    step_1_y, step_2_y, stacked_y = None, None, None

    # Get config
    task_key, configs = create_ladder_config(task_name, train_models, eval_models, color=color)

    # Get data
    # data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models)

    # Use avg of final 10% of checkpoints to fit step 1
    data_by_name = get_ladder_data(df, task_name, x_metric, y_metric, train_models, eval_models, step='all')
    for k1, v1 in data_by_name.items():
        for k2, v2 in v1.items():
            if k2 == 'step': 
                # grab final step
                data_by_name[k1][k2] = [data_by_name[k1][k2][0][-1]]
                continue
            if k2 == 'mode': continue
            if isinstance(v2, list):
                import math
                data_by_name[k1][k2] = [np.mean(v2[0][math.ceil(0.9 * len(v2[0])):])]

    # which functional form to use for step 1 prediction
    if 'byte' in x_metric:
        y_metric_func = 'rc_bpb'
    else:
        y_metric_func = 'rc_acc'

    assert len(data_by_name) != 0, train_models
    if use_two_param:
        assert use_flops, 'we only have a 2 param function for flops version'

    if use_single_step:
        assert not run_stacked and not run_step2, 'Single step prediction will only run step 1!'

    ax_i = 0
    if run_step1 or run_stacked:
        # Add token data
        data_by_name = add_ladder_data_cheap_decisions(data_by_name)
        
        # Fit step 1
        if use_single_step:
            step1_coefficients = fit_single_step(data_by_name, y_metric_func, use_flops=use_flops)
        elif use_flops:
            step1_coefficients, cov = fit_step1_flops(data_by_name, y_metric_func, use_two_param=use_two_param)
        else:
            step1_coefficients, cov = fit_step1(data_by_name, y_metric_func)

        if use_single_step:
            (
                predicted_data_by_name, plotted_predicted_data,
                (step_1_y, step_1_y_pred, rel_error_step_1),
            ) = predict_single_step(
                # configs, data_by_name, step1_coefficients, y_metric=y_metric_func, 
                data_by_name, step1_coefficients, use_flops=use_flops
            )
        elif use_flops:
            (
                predicted_data_by_name, plotted_predicted_data,
                (step_1_y, step_1_y_pred, rel_error_step_1), all_rel_errors,
            ) = predict_step1_flops(
                configs, data_by_name, step1_coefficients, y_metric=y_metric_func, use_two_param=use_two_param
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
            if use_single_step:
                plot_single_step(
                    # configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    # task_name, str_combined_fit(step1_coefficients), y_metric_func,
                    # step1_coefficients, cov, ax,
                    configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    task_name, str_combined_fit(step1_coefficients), use_flops, ax,
                )
            elif use_flops:
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
                data_by_name[k1][k2] = v2[0][math.ceil((1-last_perc_step_2) * len(v2[0])):]
    
    # data_by_name = {k: v for k, v in data_by_name.items() if '150M' not in k and '300M' not in k and '530M' not in k}

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
                data_by_name, step1_coefficients, step2_coefficients, use_two_param=use_two_param
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

    if return_coeff:
        return (step1_coefficients, step2_coefficients), (abs_error_step_1, abs_error_step_2, abs_error_stacked), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    if return_reals:
        return (abs_error_step_1, abs_error_step_2, abs_error_stacked), (step_1_y, step_2_y, stacked_y), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    if return_preds:
        return (abs_error_step_1, abs_error_step_2, abs_error_stacked), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    return abs_error_step_1, abs_error_step_2, abs_error_stacked


def compute_intersection(p1s1, p1s2, p2s1, p2s2, x_range):
    """ Compute the intersection between two scaling law curves """
    from scipy.optimize import root_scalar
    from olmo.scaling.scaling_laws.fitting_functions import sigmoid, chinchilla_flops_fit

    def diff(x):
        y1 = sigmoid(chinchilla_flops_fit(x, p1s1), *p1s2)
        y2 = sigmoid(chinchilla_flops_fit(x, p2s1), *p2s2)
        return y1 - y2
    
    # Split the range into multiple segments to check for multiple intersections
    num_segments = 100
    x_segments = np.logspace(np.log10(x_range[0] + 1e-10), np.log10(x_range[1]), num_segments+1)
    intersections = []
    
    for i in range(num_segments):
        segment_range = (x_segments[i], x_segments[i+1])
        f_a = diff(segment_range[0])
        f_b = diff(segment_range[1])
        
        if np.sign(f_a) != np.sign(f_b):
            try:
                result = root_scalar(diff, bracket=segment_range, method='brentq')
                if result.converged:
                    intersections.append(result.root)
            except ValueError:
                continue
    
    if not intersections:
        raise ValueError(f"No intersections found in the specified range.")
    
    return max(intersections)  # Return the last (rightmost) intersection

def pairwise_intersections(coeffs, x_range):
    """ Compute pairwise intersection between N scaling law fits """
    n = len(coeffs)
    intersections = np.zeros((n, n))

    x_range = (float(x_range[0]), float(x_range[1])) # convert range to floats
    
    for i in range(n):
        for j in range(n):
            if i == j:
                intersections[i, j] = 0
                continue
            
            p1s1, p1s2 = coeffs[i][0], coeffs[i][1]
            p2s1, p2s2 = coeffs[j][0], coeffs[j][1]
            
            try:
                x_intersection = compute_intersection(p1s1, p1s2, p2s1, p2s2, x_range)
                intersections[i, j] = x_intersection
            except ValueError:
                intersections[i, j] = 0
    
    return intersections


def get_perf(coeffs, C):
    """ Get the performance of a scaling law fit for some compute C """
    from olmo.scaling.scaling_laws.fitting_functions import sigmoid, chinchilla_flops_fit

    n = len(coeffs)
    perf = np.zeros((n))
    
    for i in range(n):
        p1s1, p1s2 = coeffs[i][0], coeffs[i][1]
        
        try:
            y1 = sigmoid(chinchilla_flops_fit(C, p1s1), *p1s2)
        except ValueError:
            perf = 0

        perf[i] = y1
    
    return perf


def process_mix(mix, df_multi_index, all_models, all_tasks, setup, x_metric, y_metric):
    warnings.filterwarnings("ignore", category=RuntimeWarning) # supress function fitting warnings

    results = {
        'mix': mix,
        'fitting_results_step_1': {},
        'fitting_results_step_2': {},
        'fitting_results_stacked': {},
        'step_1_y': {},
        'step_2_y': {},
        'stacked_y': {},
        'step_1_y_preds': {},
        'step_2_y_preds': {},
        'stacked_y_preds': {}
    }
    
    models = [model for model in all_models if '-'.join(model.split('-')[:-2]) == mix]

    for task in all_tasks:
        if 'no_750M_no_530M' in setup:
            train_models = [m for m in models if '1B' not in m and '750M' not in m and '530M' not in m]
            eval_models = [m for m in models if '1B' in m]
        elif 'no_750M' in setup:
            train_models = [m for m in models if '1B' not in m and '750M' not in m]
            eval_models = [m for m in models if '1B' in m]
        else:
            train_models = [m for m in models if '1B' not in m]
            eval_models = [m for m in models if '1B' in m]

        assert len(train_models) != 0 and len(eval_models) != 0, f'{mix}: ({train_models}, {eval_models}) {models}'

        use_helper_points = 'helper_points' in setup
        use_single_step   = '1_step' in setup
        use_flops         = '3_param' in setup or '2_param' in setup
        use_two_param     = '2_param' in setup

        last_perc_step_2 = 0.9
        if 'step2=0.5' in setup:
            last_perc_step_2 = 0.5

        run_step1, run_step2, run_stacked = True, True, True
        if use_single_step:
            # Only run 1 step, and have the 1 step be the downstream metric
            run_step2, run_stacked = False, False
            x_metric = y_metric

        try:
            (abs_error_step_1, abs_error_step_2, abs_error_step_stacked), \
                (step_1_y, step_2_y, stacked_y), \
                (step_1_y_pred, step_2_y_pred, stacked_y_pred) = run_ladder(
                df_multi_index,
                task_name=task,
                train_models=train_models,
                eval_models=eval_models,
                use_helper_points=use_helper_points,
                last_perc_step_2=last_perc_step_2,
                x_metric=x_metric,
                y_metric=y_metric,
                use_flops=use_flops,
                use_single_step=use_single_step,
                use_two_param=use_two_param,
                return_reals=True,
                run_step1=run_step1, run_step2=run_step2, run_stacked=run_stacked
            )
        except Exception as e:
            abs_error_step_1 = abs_error_step_2 = abs_error_step_stacked = float('inf')
            step_1_y = step_2_y = stacked_y = float('inf')
            step_1_y_pred = step_2_y_pred = stacked_y_pred = float('inf')
            if task != 'winograde' and task != 'boolq':
                # raise RuntimeError(f'{task}, {setup}: {e}')
                # We expect some of Winograde and BoolQ to fail due to missing data
                print(f'Failed to fit ({setup, mix, task, x_metric, y_metric}): {e}')

        results['fitting_results_step_1'][task] = abs_error_step_1
        results['fitting_results_step_2'][task] = abs_error_step_2
        results['fitting_results_stacked'][task] = abs_error_step_stacked
        results['step_1_y'][task] = step_1_y
        results['step_2_y'][task] = step_2_y
        results['stacked_y'][task] = stacked_y
        results['step_1_y_preds'][task] = step_1_y_pred
        results['step_2_y_preds'][task] = step_2_y_pred
        results['stacked_y_preds'][task] = stacked_y_pred

    return results


def fit_all_mixes(df, all_models, mixes, tasks, y_metrics, setups, x_metric='correct_logit_per_byte', quiet=True):
    all_predictions = []

    df_multi_index = df.set_index(['task', 'model']).sort_index()

    # Use ProcessPoolExecutor for CPU-intensive mix processing
    cpus = os.cpu_count()
    results = []
    with ProcessPoolExecutor(max_workers=cpus) as process_executor:
        total_jobs = len(y_metrics)*len(setups)*len(mixes)

        # Submit all futures upfront
        futures = []
        future_info = {}
        for y_metric in tqdm(y_metrics, desc=f"Submitting {total_jobs} fitting jobs on {cpus} CPUs", total=len(y_metrics)):
            for setup in setups:
                for mix in mixes:
                    future = process_executor.submit(process_mix, mix, df_multi_index, all_models, tasks, setup, x_metric, y_metric)
                    futures.append(future)
                    future_info[future] = (mix, y_metric, setup)

            try:
                for future in tqdm(futures, desc=f"Processing results for {y_metric}", total=total_jobs):
                    result = future.result()
                    mix, y_metric, setup = future_info[future]
                    results.append((result, mix, y_metric, setup))
                    future.done()  # Ensure all futures are complete
                    del future  # Immediately remove future
            finally:
                for future in futures:
                    future.cancel()  # Cancel any remaining futures
                del futures[:]  # Free memory
        process_executor.shutdown(wait=True)  # Shutdown the executor and wait for cleanup

    print(f'Done processing jobs!')

    # Process results as they complete
    for result, mix, y_metric, setup in tqdm(results, desc='Processing all predictions', total=len(results), disable=quiet):
        fitting_results_step_1 = pd.DataFrame(index=[], columns=tasks)
        fitting_results_step_2 = pd.DataFrame(index=[], columns=tasks)
        fitting_results_stacked = pd.DataFrame(index=[], columns=tasks)
        step_1_y = pd.DataFrame(index=[], columns=tasks)
        step_2_y = pd.DataFrame(index=[], columns=tasks)
        stacked_y = pd.DataFrame(index=[], columns=tasks)
        step_1_y_preds = pd.DataFrame(index=[], columns=tasks)
        step_2_y_preds = pd.DataFrame(index=[], columns=tasks)
        stacked_y_preds = pd.DataFrame(index=[], columns=tasks)

        for task in tasks:
            fitting_results_step_1.loc[mix, task] = result['fitting_results_step_1'].get(task, float('inf'))
            fitting_results_step_2.loc[mix, task] = result['fitting_results_step_2'].get(task, float('inf'))
            fitting_results_stacked.loc[mix, task] = result['fitting_results_stacked'].get(task, float('inf'))
            step_1_y.loc[mix, task] = result['step_1_y'].get(task, float('inf'))
            step_2_y.loc[mix, task] = result['step_2_y'].get(task, float('inf'))
            stacked_y.loc[mix, task] = result['stacked_y'].get(task, float('inf'))
            step_1_y_preds.loc[mix, task] = result['step_1_y_preds'].get(task, float('inf'))
            step_2_y_preds.loc[mix, task] = result['step_2_y_preds'].get(task, float('inf'))
            stacked_y_preds.loc[mix, task] = result['stacked_y_preds'].get(task, float('inf'))

        def process_dataframe(df, calculate_abs=False):
            pd.set_option('future.no_silent_downcasting', True)
            df = df.fillna(value=np.nan)
            df['avg'] = df.mean(axis=1, skipna=True)
            if calculate_abs:
                df = df.abs()
            return df.sort_values(by='avg', ascending=False)

        # Process DataFrames sequentially
        fitting_results_step_1 = process_dataframe(fitting_results_step_1, True)
        fitting_results_step_2 = process_dataframe(fitting_results_step_2, True)
        fitting_results_stacked = process_dataframe(fitting_results_stacked, True)
        step_1_y = process_dataframe(step_1_y)
        step_2_y = process_dataframe(step_2_y)
        stacked_y = process_dataframe(stacked_y)
        step_1_y_preds = process_dataframe(step_1_y_preds)
        step_2_y_preds = process_dataframe(step_2_y_preds)
        stacked_y_preds = process_dataframe(stacked_y_preds)

        if not quiet:
            print('Absolute unsigned error for predicting 1B-5xC (stacked):')
            display(fitting_results_stacked.map(lambda x: f'{round(x * 100, 1)}%'))
            print('Predicted performance for 1B-5xC on all mixes:')
            display(stacked_y_preds.map(lambda x: f'{round(x * 100, 1)}%'))

        (step_1_abs_error, step_2_abs_error, stacked_abs_error), \
            (step_1_y_preds, step_2_y_preds, stacked_y_preds), \
            (step_1_y, step_2_y, stacked_y) = \
            (fitting_results_step_1, fitting_results_step_2, fitting_results_stacked), \
            (step_1_y_preds, step_2_y_preds, stacked_y_preds), \
            (step_1_y, step_2_y, stacked_y)
        
        all_predictions += [(
            x, y, y_metric, setup,
            step_1_y.loc[y, x], step_2_y.loc[y, x], stacked_y.loc[y, x],
            step_1_y_preds.loc[y, x], step_2_y_preds.loc[y, x], stacked_y_preds.loc[y, x], 
            step_1_abs_error.loc[y, x], step_2_abs_error.loc[y, x], stacked_abs_error.loc[y, x],
        ) for x in stacked_y_preds.columns for y in stacked_y_preds.index]

    results = pd.DataFrame(all_predictions, columns=[
        'task', 'mix', 'metric', 'setup', 
        'step_1_y', 'step_2_y', 'stacked_y',
        'step_1_pred', 'step_2_pred', 'stacked_pred', 
        'abs_error_step_1', 'abs_error_step_2', 'abs_error_stacked'
    ])

    # For single step setups, only the step_1 pred is reported. Copy over the values
    for col_base in ['y', 'pred', 'abs_error']:
        if col_base in ['y', 'pred']:
            step_1_col = f'step_1_{col_base}'
            step_2_col = f'step_2_{col_base}'
            stacked_col = f'stacked_{col_base}'
        if col_base in ['abs_error']:
            step_1_col = f'{col_base}_step_1'
            step_2_col = f'{col_base}_step_2'
            stacked_col = f'{col_base}_stacked'
        
        # Where step_1 exists but step_2 and stacked are NaN, copy the values
        mask = results[step_1_col].notna() & results[step_2_col].isna() & results[stacked_col].isna()
        results.loc[mask, step_2_col] = results.loc[mask, step_1_col]
        results.loc[mask, stacked_col] = results.loc[mask, step_1_col]

    # Re-compute rel error
    results['rel_error_stacked'] = results['abs_error_stacked'] / results['stacked_pred']

    # remove "avg" task before returning
    results = results[results['task'] != "avg"]

    return results


def clean_data_and_compute_averages(df, quiet=True):
    """ Wrapper around Ian's data cleaning to compute macro averages """
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"], errors='ignore')

    # Preprocess the df into a usuable format
    if not quiet: print('Converting metrics dict to a set of cols...')
    df["metrics"] = df["metrics"].apply(eval)
    metrics_df = df["metrics"].apply(pd.Series)
    df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    # Remove bad mixes
    BAD_MIXES = ["DCLM-baseline-25p", "DCLM-baseline-50p", "DCLM-baseline-75p"]
    for bad_mix in BAD_MIXES:
        df = df[df["group"] != bad_mix]

    df.loc[df['group'] == 'baseline', 'group'] = 'dolma17'

    df['size'] = df['model']
    df['model'] = df['group'] + '-' + df['model'] + '-' + df['chinchilla']

    if not quiet: print('Launching data cleaning!')

    df = ian_clean_data(df, dirty_out=False, quiet=quiet)

    if not quiet: print('Computing macro averages...')

    # Compute MMLU macro-average
    group_cols = ['group', 'model', 'chinchilla', 'step', 'seed']
    agg_cols = [col for col in df.columns if col not in group_cols and col != 'task']
    mmlu_rows = df[df['task'].str.contains("MMLU", case=False)]
    numeric_cols = mmlu_rows[agg_cols].select_dtypes(include=['number']).columns.tolist()
    aggregated = mmlu_rows.groupby(group_cols, as_index=False)[numeric_cols].mean()
    aggregated['task'] = 'mmlu'
    df = df[~df['task'].str.contains("MMLU", case=False)]
    df = pd.concat([df, aggregated], ignore_index=True)

    # Compute olmes macro-average
    group_cols = ['group', 'model', 'chinchilla', 'step', 'seed']
    agg_cols = [col for col in df.columns if col not in group_cols and col != 'task']
    olmes_rows = df # olmes_rows = df[df['task'].str.contains("olmes", case=False)]
    numeric_cols = olmes_rows[agg_cols].select_dtypes(include=['number']).columns.tolist()
    aggregated = olmes_rows.groupby(group_cols, as_index=False)[numeric_cols].mean()
    aggregated['task'] = 'olmes_10_macro_avg'
    df = pd.concat([df, aggregated], ignore_index=True)

    df['size'] = df['model'].str.split('-').str[-2]

    # Remove extra metrics columns that were not used everywhere
    df = df.drop(columns=[
        "predicted_index_per_byte", 
        "acc_per_byte", 
        "sum_logits_corr", 
        "logits_per_token_corr", 
        "logits_per_char_corr", 
        "logits_per_byte_corr"
    ], errors='ignore')

    if not quiet: print('Done!')

    return df


def ian_clean_data(df, dirty_out=False, quiet=True):
    """ Clean data according to https://github.com/allenai/oe-eval-internal/blob/eval-for-consistent-ranking/experiments/eval-for-consistent-ranking/metrics/project/data_exploration_and_cleaning.ipynb """
    
    print(f'Step 0: {len(df)}')
    
    # Ian uses only the size for "model"
    if 'model_full' not in df.columns:
        df['model_full'] = df['model']
        df['model'] = df['model'].apply(lambda x: x.split('-')[-2] if '-' in x else None)
    
    # 1) Clean group names
    # print(len(df['group'].unique()))
    df.loc[df['group'] == 'baseline', 'group'] = 'dolma17'

    bad_mixes = [
        'DCLM-baseline-25p',
        'DCLM-baseline-50p',
        'DCLM-baseline-75p',
    ]

    cannonical_groups = set([
        'DCLM-baseline',
        'c4',
        'dclm_ft7percentile_fw2',
        'dclm_ft7percentile_fw3',
        'dclm_fw_top10',
        'dclm_fw_top3',
        'dolma-v1-6-and-sources-baseline',
        'dolma17',
        'dolma17-25p-DCLM-baseline-75p',
        'dolma17-50p-DCLM-baseline-50p',
        'dolma17-75p-DCLM-baseline-25p',
        'falcon',
        'falcon_and_cc',
        'falcon_and_cc_eli5_oh_top10p',
        'falcon_and_cc_eli5_oh_top20p',
        'falcon_and_cc_og_eli5_oh_top10p',
        'falcon_and_cc_tulu_qc_top10',
        'fineweb_edu_dedup',
        'no_code',
        'no_flan',
        'no_math_no_code',
        'no_reddit',
        'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p',
        'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p',
        'prox_fineweb_pro'
    ])

    df = df[~df['group'].isin(bad_mixes)]

    # print(len(df['group'].unique()))
    assert set(sorted(df['group'].unique())) == cannonical_groups

    print(f'Step 1: {len(df)}')

    # 2) Normalize seeds
    if not quiet:
        print('raw seeds:')
        print(df['seed'].unique())

    def normalize_seeds(df):
        df = df.copy()
        df['seed'] = df['seed'].fillna(6198)
        df['seed'] = df['seed'].astype(int)
        df.loc[df['seed'] == 2, 'seed'] = 6198
        return df

    df = normalize_seeds(df)

    if not quiet:
        print('normalized seeds:')
        print(df['seed'].unique())

    print(f'Step 2: {len(df)}')

    # 3) Throw out 1B seed 14 and 15 cuz we have full seed runs from 4 and 5
    df = df[~((df['model'] == '1B') & (df['seed'].isin([14, 15])))]

    print(f'Step 3: {len(df)}')

    # # 4) Remove all steps without 3 seeds
    # if not dirty_out:
    #     pre_filter_groups = df['group'].unique()
    #     filtered_groups = []
    #     throwaway_groups = []
    #     for name, data in df.groupby(['model', 'group', 'step']): # understand why token and compute effecting this so much? ['model', 'group', 'step', 'tokens', 'compute']
    #         if len(set(data['seed'])) == 3:
    #             filtered_groups.append(data)
    #         else:
    #             throwaway_groups.append(data)
    #     df = pd.concat(filtered_groups)
    #     df_throwaway = pd.concat(throwaway_groups)
    #     df_throwaway['group'].value_counts()

    #     if not quiet: print(len(df['group'].unique()))
    #     post_filter_groups = df['group'].unique()
    #     if not quiet: print(set(pre_filter_groups) - set(post_filter_groups))

    #     # are any steps missing some groups?
    #     missing_groups = []
    #     for (model, step), data in df.groupby(['model', 'step']):
    #         present_groups = set(data['group'].unique())
    #         missing = set(post_filter_groups) - present_groups
    #         if missing:
    #             missing_groups.append((model, step, missing))

    print(f'Step 4: {len(df)}')
    
    # 5) Throw out all steps beyond the end of LR schedule
    full_schedule_last_step_per_model = {
        '150M': 38157,
        '300M': 45787,
        '530M': 57786,
        '750M': 63589,
        '1B': 69369,
    }

    # have to round up for the models that ran to long to make sure we get a checkpoint after the LR fully decays
    def round_up(value, increment):
        return (value + increment - 1) // increment * increment
    for model, step in full_schedule_last_step_per_model.items():
        if model == '1B':
            full_schedule_last_step_per_model[model] = round_up(step, 2500)
        else:
            full_schedule_last_step_per_model[model] = round_up(step, 1250)

    df = df[df.apply(lambda row: row['step'] <= full_schedule_last_step_per_model[row['model']], axis=1)]

    # are there some groups that have fewer steps for a given model size?
    min_last_step_per_model = {}
    for name, data in df.groupby(['model', 'seed']):
        max_per_model = -1
        max_steps = None
        max_group = None
        min_per_model = 100000
        min_group = None
        min_steps = None
        for group in data['group'].unique():
            group_data = data[data['group'] == group]
            if len(group_data['step'].unique()) > max_per_model:
                max_per_model = len(group_data['step'].unique())
                max_group = group
                max_steps = sorted(group_data['step'].unique())
            if len(group_data['step'].unique()) < min_per_model:
                min_per_model = len(group_data['step'].unique())
                min_group = group
                min_steps = sorted(group_data['step'].unique())
        min_last_step_per_model[name] = min_steps[-1]
        if not quiet:
            if max_per_model != min_per_model:
                print(f"max steps for {name}: {max_per_model}")
                print(f"min steps for {name}: {min_per_model}")
                print(f"min group for {name}: {min_group}")


    print(f'Step 5: {len(df)}')
    
    # 6) [RESOLVED] remove groups that don't have targets for 3 seeds for final result
    target_df = df[df['model'] == '1B']
    group_seeds = {}
    for _, row in target_df[['group', 'seed']].iterrows():
        group = row['group']
        seed = row['seed']
        if group not in group_seeds:
            group_seeds[group] = set()
        group_seeds[group].add(seed)

    # group_seeds

    for group in target_df['group'].unique():
        # assert len(target_df['seed'].unique()) == 1, f"Uncomment the next line for more seeds: {target_df['seed'].value_counts()}"
        for seed in {4, 5, 6198}:
        # for seed in {4}:
            latest_step = target_df[(target_df['group'] == group) & (target_df['seed'] == seed)]['step'].max()
            assert latest_step == 69369, f"seed {seed} latest step: {latest_step}"
            if latest_step != 69369:
                print(f"seed {seed} latest step: {latest_step}")

    print(f'Step 6 (excluded): {len(df)}')
    
    # 7) Throw out duplicate rows based on ['model', 'group', 'task', 'step', 'seed']
    def check_duplicates(df):
        for name, data in df.groupby(['model', 'group', 'task', 'step','seed']):
            if len(data) != 1:
                print(f"there are duplicates here in {name}: {data}")
                diff_columns = data.loc[:, (data != data.iloc[0]).any()].columns
                print(f"Different columns: {diff_columns}")
                print(f"The different values are:\n{data[diff_columns]}")
                raise

    # prove to myself that these are just duplicates before dropping them (uncomment to run)
    check_duplicates(df.fillna(0).round(6).drop_duplicates())

    # drop duplicates
    df = df.groupby(['model', 'group', 'task', 'step', 'seed']).first().reset_index()
    # if not dirty_out:
    #     assert all(len(d) == 1 for n,d in df.groupby(['model', 'group', 'task', 'step','seed']) ), f"There are duplicates in the data; max size per models X group X steps X seed X task was {max((len(d) for n, d in df.groupby(['model', 'group', 'task', 'step','seed'])))}"
    #     assert all(len(d) == 3 for n, d in df.groupby(['model', 'group', 'task', 'step'])['seed']), f"Not all models X group X steps X task have 3 seeds; min size was {df.groupby(['model', 'group', 'step'])['seed'].size().min()}"
    #     assert all(d['seed'].nunique() ==3 for n, d in df.groupby(['model', 'group', 'task', 'step'])), f"Not all models X group X steps X task have 3 seeds; min size was {df.groupby(['model', 'group', 'step'])['seed'].nunique().min()}"

    print(f'Step 7: {len(df)}')
    
    # 8) Resolve NaNs by recomputing token and compute values for all rows
    # These columns are not really used now
    columns_to_drop = [
        'eval/c4_en-validation/CrossEntropyLoss',
        'eval/dolma_common-crawl-validation/CrossEntropyLoss',
        'eval/pile-validation/CrossEntropyLoss',
        'eval/wikitext_103-validation/CrossEntropyLoss',
        'train/CrossEntropyLoss',
        'throughput/total_tokens'
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')

    model_to_batch = {
        '150M': 192,
        '300M': 320,
        '530M': 448,
        '750M': 576,
        '1B': 704
    }
    
    model_to_params = {'150M': 151898880, '300M': 319980544, '530M': 530074944, '750M': 681297408, '1B': 1176832000}
    model_to_params = {k: float(v) for k, v in model_to_params.items()}

    sequence_length = 2048

    def model_and_step_to_tokens(model, step):
        return model_to_batch[model] * step * sequence_length

    def model_and_step_to_compute(model, step):
        return model_to_params[model] * model_and_step_to_tokens(model, step) * 6

    # Prove to myself that we can just recompute these values
    # for rows in df.dropna().itertuples():
    #     estimated_compute = model_and_step_to_compute(rows.model, rows.step)
    #     estimated_tokens = model_and_step_to_tokens(rows.model, rows.step)
    #     assert abs(estimated_compute - rows.compute) < 1e-6, f"Compute mismatch for model {rows.model}, step {rows.step}"
    #     assert abs(estimated_tokens - rows.tokens) < 1e-6, f"Tokens mismatch for model {rows.model}, step {rows.step}"

    def recompute_tokens_and_compute(df):
        df = df.copy()
        df['tokens'] = df.apply(lambda row: model_and_step_to_tokens(row['model'], row['step']), axis=1)
        df['compute'] = df.apply(lambda row: model_and_step_to_compute(row['model'], row['step']), axis=1)
        return df

    df = recompute_tokens_and_compute(df)

    print(f'Step 8: {len(df)}')
    
    # 9) Remove all steps that don't have all groups
    if dirty_out:
        df = df[df['seed'] == 6198]

    def remove_incomplete_model_steps(df):
        available_groups = set(df.group.unique())
        df = df.copy()
        # Group by model and step
        grouped = df.groupby(['model', 'step'])
        
        # Filter out groups that don't have all available groups
        complete_groups = [name for name, group in grouped if set(group['group'].unique()) == available_groups]
        
        # Filter the dataframe to keep only the complete groups
        filtered_df = df[df.set_index(['model', 'step']).index.isin(complete_groups)]
        
        return filtered_df

    available_groups = set(df.group.unique())
    # assert all(set(d.group.unique()) == available_groups for n, d in remove_incomplete_model_steps(df).groupby(['model', 'task', 'step', 'seed']))

    print(f'Step 9 (now included!): {len(df)}')
    
    # restore model col
    df['model'] = df['model_full']
    df.drop(columns=['model_full'], errors='ignore', inplace=True)

    return df


def render_result_table(results, index, agg_col='setup', only_use_default_scaling_law=False, raw_values=False):
    """ Convert a df of results to LaTeX """
    from utils.constants_ian import SETUP_NAME_LATEX

    if index != 'metric':
        filtered_results = results[results['metric'] == 'primary_metric']
    else:
        filtered_results = results

    # Select numeric columns for aggregation
    agg_cols = ['abs_error_stacked', 'rel_error_stacked', 'stacked_y']

    # Compute averages for each unique value in 'setup'
    average_by_setup = filtered_results.groupby([index, agg_col])[agg_cols].mean().reset_index()

    # Pivot the table to have 'setup' as columns for comparison
    pivoted_results = average_by_setup.pivot(index=index, columns=agg_col, values=agg_cols).reset_index()

    # Flatten the multi-index columns
    pivoted_results.columns = ['_'.join(col).strip('_') for col in pivoted_results.columns.values]

    # Format percentage columns
    for col in pivoted_results.columns:
        if 'abs_error_stacked' in col or 'rel_error_stacked' in col:
            pivoted_results[col] = (pivoted_results[col] * 100).round(2)

    # Calculate averages for each column and append as a new row
    avg_row = {col: pivoted_results[col].mean() if 'abs_error_stacked' in col or 'rel_error_stacked' in col else 'Average' for col in pivoted_results.columns}
    pivoted_results = pd.concat([pivoted_results, pd.DataFrame([avg_row])], ignore_index=True)

    # Display results
    pivoted_results = pivoted_results.set_index(index)

    # Multiply OLMES Avg col by 100
    for col in pivoted_results.columns:
        if 'stacked_y_' in col and index != 'metric':
            pivoted_results[col] = pivoted_results[col].apply(lambda x: x * 100 if isinstance(x, float) else x)
    
    if not raw_values:
        pivoted_results = pivoted_results.map(lambda x: (f"{x:.2f}" if isinstance(x, float) else x) + "%")

    if only_use_default_scaling_law:
        pivoted_results = pivoted_results[[col for col in pivoted_results.columns if col.endswith("3_param-default") or col.endswith("3_param-no_750M") or col.endswith("3_param-no_750M_no_530M")]]
        pivoted_results = pivoted_results.sort_values('abs_error_stacked_3_param-default')
        pivoted_results.rename(columns=lambda col: col.replace("stacked_y_", "OLMES Avg. ").replace("abs_error_stacked_", "Abs Error ").replace("rel_error_stacked_", "Rel Error ").replace("3_param-", "").replace("no_", "-").replace("_", " "), inplace=True)
        pivoted_results = pd.concat([pivoted_results.loc[pivoted_results.index != "Average"], pivoted_results.loc[["Average"]]])

    if index == 'setup':
        # Create a mapping from index values to their order in SETUP_NAME_LATEX
        setup_order = {k: i for i, k in enumerate(SETUP_NAME_LATEX.keys())}
        
        # Sort the index based on the order in SETUP_NAME_LATEX
        pivoted_results = pivoted_results.reindex(sorted(pivoted_results.index, 
                                                       key=lambda x: setup_order.get(x, float('inf')) if x != 'Average' else float('inf')))

    cols_to_drop = [col for col in pivoted_results.columns if "OLMES Avg." in col and 'default' not in col]
    pivoted_results = pivoted_results.drop(columns=cols_to_drop)

    if index == 'metric':
        cols_to_drop = [col for col in pivoted_results.columns if "Rel Error" in col]
        pivoted_results = pivoted_results.drop(columns=cols_to_drop)
    
    return pivoted_results


def fix_table_rendering(table):
    """ Fix formatting issues with pandas table formatter """
    from utils.constants_ian import DATA_NAME_LATEX, SETUP_NAME_LATEX, TASK_NAME_LATEX

    lines = table.split('\n')
    # Process each line of the table
    for i, line in enumerate(lines):
        # if 'Task' in table: continue
        if not line or line.startswith(' ') or line.startswith('\\'): continue
        if 'Recipe' in line: continue
        
        # Map the data mix name to the latex name
        for key, value in DATA_NAME_LATEX.items():
            if line.split(' ')[0] == key: 
                lines[i] = lines[i].replace(key, value)
        
        # Map the data mix name to the latex name
        for key, value in SETUP_NAME_LATEX.items():
            if line.split(' ')[0] == key: 
                lines[i] = lines[i].replace(key, value)
        
        # Map the data mix name to the latex name
        for key, value in TASK_NAME_LATEX.items():
            if line.split(' ')[0] == key: 
                lines[i] = lines[i].replace(key, value)

    # Add midrule before Average row
    for i, line in enumerate(lines):
        if line.strip().startswith('Average'):
            lines.insert(i, '\\midrule')
            break
    
    # Rejoin and print
    table_str = '\n'.join(lines).replace('%', '\%')

    if 'Scaling Law Functional Form' in table:
        table_str = table_str.replace('\n    ', '\n\\hspace{1em}\\hspace{1em}').replace('\n  ', '\n\\hspace{1em}')

    return table_str
    