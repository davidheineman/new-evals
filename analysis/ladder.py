import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataloader import get_nd_array, get_slice
from download.preprocess import SIZE_PREFIXES, str_find
from collections import defaultdict

from olmo.scaling.scaling_laws.utils import MODEL_FLOPS, FinalConfig
from olmo.scaling.scaling_laws.utils import get_final_configs, get_step2_data_by_name, get_task_sets

from scaling.step2 import main as step2_main
from scaling.step2 import fit_step2, predict_step2, plot_step2
from scaling.variance_analysis import plot_variance_analysis

from olmo.scaling.scaling_laws.utils import get_final_configs, get_step1_data_by_name, get_task_sets
from scaling.step1 import main as step1_main
from scaling.step1 import fit_step1, predict_step1, plot_step1, str_chinchilla_n_d_fit
from scaling.predict import predict_chained, plot_chained, str_chained_fit


from olmo.scaling.scaling_laws.fitting_functions import chinchilla_n_d_fit, sigmoid

DEFAULT_CONIFG_PATH = "scripts/scaling/final.json"
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

COLOR_MAP = {
    'Qwen': 'green',
    'Llama': 'blue',
    'LLaMA': 'blue',
    'Mistral': 'orange',
    '3B': 'black',
    'OLMo': 'pink',
    'pythia': 'brown',
    'gemma': 'teal',
    'phi': 'black',
    'deepseek': 'pink',
    'zephyr': 'green',
    'neo': 'orange',
    'falcon': 'blue'
}


def sort_experiments_corrected(experiments):
    """ Sort ladder model keys """
    def extract_sort_key(entry):
        parts = entry.split('-')
        size_str = parts[-2]  # Extract size portion (e.g., "190M", "1B")
        xc_str = parts[-1]  # Extract xC portion (e.g., "1xC")
        
        if "M" in size_str:
            size = int(size_str.replace("M", "")) * 1e6
        elif "B" in size_str:
            size = int(size_str.replace("B", "")) * 1e9
        else:
            size = 0  # Fallback in case of unexpected format
        
        xc = float(xc_str.replace("xC", ""))  # Extract xC numerical value
        return (size, xc)

    return sorted(experiments, key=extract_sort_key)


def merge_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        raise ValueError(f"Keys of dict1 and dict2 do not match. Seeing:\n{dict1.keys()}\n{dict2.keys()}")
    
    for key in dict1:
        l1, l2 = len(dict1[key]['xs']), len(dict2[key]['xs'])

        # if l2 != l1:
        #     raise ValueError(f"Length of 'xs' in dict1 and 'xs' in dict2 do not match for key '{key}': {l1} != {l2}.")

        sorted_indices = sorted(range(len(dict2[key]['ds'])), key=lambda i: dict2[key]['ds'][i])
        dict2[key]['ds'] = [dict2[key]['ds'][i] for i in sorted_indices]
        dict2[key]['ns'] = [dict2[key]['ns'][i] for i in sorted_indices]
        dict2[key]['ls'] = [dict2[key]['ls'][i] for i in sorted_indices]
        if 'fs' in dict2: dict2[key]['fs'] = [dict2[key]['fs'][i] for i in sorted_indices]

        if l1 != l2:
            # A faustian bargain to allow us to use the wandb tokens w/ oe-eval for intermediate ckpts, since we have different numbers
            # of checkpoints for both. 
            if l1 < l2:
                # Shorter is dict1[key]['xs'], sample dict2[key]['xs']
                indices = np.linspace(0, l2 - 1, l1, dtype=int)
                dict1[key]['ns'] = [dict2[key]['ns'][i] for i in indices]
                dict1[key]['ds'] = [dict2[key]['ds'][i] for i in indices]
                dict1[key]['ls'] = [dict2[key]['ls'][i] for i in indices]
                if 'fs' in dict2: dict1[key]['fs'] = [dict2[key]['fs'][i] for i in indices]
            else:
                raise RuntimeError()
        else:
            dict1[key]['ns'] = dict2[key]['ns']
            dict1[key]['ds'] = dict2[key]['ds']
            dict1[key]['ls'] = dict2[key]['ls']
            if 'fs' in dict2: dict1[key]['fs'] = dict2[key]['fs']
    
    return dict1


def get_ladder_data(df, task_name, train_models, eval_models, step='max'):
    """ Get slices of df and convert to ladder prediction format """
    if isinstance(task_name, list):
        assert len(task_name), f'Task suite needs to be only one list: {task_name}'
        task_name = task_name[0]

    data_by_name = defaultdict(dict)

    for model in train_models + eval_models:
        if model in eval_models:
            mode = 'eval'

            # Fix string names for ladder models
            if model == 'peteish7':
                size = '7B-4T'
            elif 'peteish13-highlr' in model:
                size = '13B-5T'
            else:
                size = model
        elif model in train_models:
            mode = 'train'

            # Fix string names for ladder models
            size = str_find(SIZE_PREFIXES, model)
            if size is not None:  # and size != '-3B-'
                size = size.replace('-', '')
            else:
                size = model
        else:
            raise NameError(model)
        
        loss_task_name = task_name

        # super hacky temporary thing (pls fix soon)
        # if 'gsm' in task_name:
        #     loss_task_name = f'{task_name}:perturb_cot'
        olmes_gen = ["drop", "jeopardy", "naturalqs", "squad", "triviaqa"]
        if 'mmlu_pro_' in task_name:
            loss_task_name = task_name.replace(':cot', '') # this uses the :cot responses, but :rc is easier to predict
        elif ':cot' not in task_name and task_name in olmes_gen:
            loss_task_name = f'{task_name}:perturb_rc'
        elif ':cot' in task_name:
            loss_task_name = task_name.replace(':cot', ':perturb_cot')
        
        if step == 'max':
            m1, corr       = get_nd_array(df, 'model', 'correct_choice', model=model, task=loss_task_name, step=step)
            m2, bpb        = get_nd_array(df, 'model', 'logits_per_byte', model=model, task=loss_task_name, step=step)
            m3, acc        = get_nd_array(df, 'model', 'acc_per_char', model=model, task=task_name, step=step)
            m4, em         = get_nd_array(df, 'model', 'exact_match', model=model, task=task_name, step=step)
            m5, pass_at_1  = get_nd_array(df, 'model', 'pass_at_1', model=model, task=task_name, step=step)
            m6, pass_at_10 = get_nd_array(df, 'model', 'pass_at_10', model=model, task=task_name, step=step)
        else:
            if step == 'all': step = None
            # get results from multiple steps
            m1, corr       = get_nd_array(df, ['model', 'step', 'mix'], 'correct_choice', model=model, task=loss_task_name, step=step)
            m2, bpb        = get_nd_array(df, ['model', 'step', 'mix'], 'logits_per_byte', model=model, task=loss_task_name, step=step)
            m3, acc        = get_nd_array(df, ['model', 'step', 'mix'], 'acc_per_char', model=model, task=task_name, step=step)
            m4, em         = get_nd_array(df, ['model', 'step', 'mix'], 'exact_match', model=model, task=task_name, step=step)
            m5, pass_at_1  = get_nd_array(df, ['model', 'step', 'mix'], 'pass_at_1', model=model, task=task_name, step=step)
            m6, pass_at_10 = get_nd_array(df, ['model', 'step', 'mix'], 'pass_at_10', model=model, task=task_name, step=step)

        # print(loss_task_name)
        # print(em)
        # print(corr)
        # print(bpb)

        # print(acc, em, corr, bpb)
        # print(m1, m2, m3, m4)

        if em.size != 0:
            # if there's exact match, use it as the primary metric
            acc, m3 = em, m4
        if pass_at_1.size != 0 and np.all(~np.isnan(pass_at_1)):
            # if there's pass@k, use it
            acc, m3 = pass_at_1, m5
        if pass_at_10.size != 0 and np.all(~np.isnan(pass_at_10)):
            # if there's pass@k, use it
            acc, m3 = pass_at_10, m6

        # If correct_choice is not set right (nan), set to 0
        corr = np.nan_to_num(corr)

        if bpb.size == 0:
            # raise RuntimeError(f'bpb is empty array: {bpb} on model: {model}')
            print(f'bpb is empty array: {bpb} on model "{model}" for task "{task_name}"')
            continue

        # Ensure the model results are all the same model names
        # assert [_1 == _2 == _3 for _1, _2, _3 in zip(m1, m2, m3)], f'{model} failed'
        if not all(_1 == _2 == _3 for _1, _2, _3 in zip(m1, m2, m3)):
            print(f"{model} result set mismatch task {task_name}.")

            if corr.ndim == 3:
                print('Fixing...')
                # manual fix for olmes gen benchmarks (since the num checkpoints may not match)
                mask = np.isin(m1, m3) # corr indices in acc
                corr = corr[:, mask, :]

                mask = np.isin(m2, m3) # bpb indices in acc
                bpb = bpb[:, mask, :]
            else:
                continue

        if len(corr) == 0 or len(bpb) == 0 or len(acc) == 0: 
            if mode == 'eval':
                print(f'Eval point data not found: {model}')
                continue
                # raise RuntimeError(f'Eval point data not found: {model}')
            # continue 
        
        # if isinstance(task_name, list):
        #     print('Need to compute a weighted average here!!')

        # Get correct logprobs per char
        if bpb.ndim == 2:
            n_choices = bpb[0][0].shape
            correct_bpb = np.empty_like(corr, dtype=np.float64)
            rows, cols = corr.shape
            for i in range(rows):
                for j in range(cols):
                    if corr[i, j] == n_choices and 'enlarge' in task_name: 
                        # print(f'Warning: bpb has {n_choices} choices, but the correct label is {corr[i, j]} (did ChatGPT generate an incorrect ground truth?). re-indexing the correct label...')
                        corr[i, j] -= 1
                    correct_bpb[i, j] = bpb[i, j][corr[i, j].astype(np.int32)]
        elif bpb.ndim == 3:
            # Get correct logprobs per char for 3D arrays
            n_choices = bpb[0][0].shape
            correct_bpb = np.empty_like(corr, dtype=np.float64)
            depth, rows, cols = corr.shape

            for k in range(depth):  # Iterate over the first dimension
                for i in range(rows):
                    for j in range(cols):
                        if corr[k, i, j] == n_choices and 'enlarge' in task_name: 
                            # Adjust the correct label if necessary
                            corr[k, i, j] -= 1
                        correct_bpb[k, i, j] = bpb[k, i, j][corr[k, i, j].astype(np.int32)]
        else:
            raise ValueError(bpb)

        acc = acc.mean(axis=-1)
        correct_bpb = correct_bpb.mean(axis=-1)

        if 'xs' not in data_by_name[size]: data_by_name[size]['xs'] = []
        if 'ys' not in data_by_name[size]: data_by_name[size]['ys'] = []

        # just get a single number
        if correct_bpb.size == 1:
            correct_bpb = correct_bpb.item() 
        else:
            correct_bpb = correct_bpb.squeeze().tolist()
        
        if acc.size == 1:
            acc = acc.item()
        else:
            acc = acc.squeeze().tolist()

        data_by_name[size]['xs'] += [correct_bpb]
        data_by_name[size]['ys'] += [acc]
        data_by_name[size]['mode'] = mode

    return data_by_name


def get_ladder_config(config_path, task_name, train_models, eval_models):
    # arc_easy:enlarge => arc_easy
    task_root = task_name.split(':')[0] if isinstance(task_name, str) else None

    # Get ladder model configs
    configs = get_final_configs(config_path)

    # Add new models to config
    for model in train_models + eval_models:
        if model not in configs.keys():
            # Get model color
            color = 'red'
            for key, value in COLOR_MAP.items():
                if key in model:
                    color = value
                    break

            mode = 'eval' if model in eval_models else 'train'

            model_label = model
            if '-3B-' in model_label:
                model_label = '3B'
                color = 'b'
            
            # Create dummy config for new eval points
            configs[model] = FinalConfig(
                paths=None, mode=mode, n=0, label=model_label, color=color, use_last_n_percentage=100
            )

    task_key = TASK_KEY_MAP.get(task_root, None) # the task key is used to get min/max perf and plot title

    return task_key, configs


def run_ladder_step_1(df, task_name, train_models, eval_models, ax=None, config_path=DEFAULT_CONIFG_PATH):
    # Unfortunately there are local references, so we have to be in the OLMo repo
    os.chdir('/Users/dhei/ai2/new-evals/olmo-repos/OLMo')

    # Get config
    configs = get_final_configs(config_path)

    # Get data
    data_by_name = get_step1_data_by_name(configs, 'arc_easy_test_5shot', y_metric='rc_bpb', moving_avg=5) # we are only using this for the num tokens
    data_by_name_downstream = get_ladder_data(df, task_name, train_models, eval_models)

    data_by_name = merge_dicts(data_by_name_downstream, data_by_name) # merge the 'ns', 'ds', 'ls', 'fs' keys into the step 2 data

    # Fit step 1
    coefficients, cov = fit_step1(data_by_name, 'rc_bpb')

    (
        predicted_data_by_name, plotted_predicted_data,
        (y, y_pred, rel_error), all_rel_errors,
    ) = predict_step1(
        configs, data_by_name, coefficients, y_metric='rc_bpb', 
    )

    plot_step1(
        configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
        task_name, str_chinchilla_n_d_fit(coefficients), 'rc_bpb',
        coefficients, cov, ax,
    )

    ax.set_title(task_name)
    ax.legend(fontsize=6)

    return (y, y_pred, rel_error), all_rel_errors


def run_ladder_step_2(df, task_name, train_models, eval_models, ax=None, config_path=DEFAULT_CONIFG_PATH, add_texts=False):
    # Unfortunately there are local references, so we have to be in the OLMo repo
    os.chdir('/Users/dhei/ai2/new-evals/olmo-repos/OLMo')

    data_by_name = get_ladder_data(df, task_name, train_models, eval_models)
    task_key, configs = get_ladder_config(config_path, task_name, train_models, eval_models)
    
    _min, _max = None, None
    if task_key is None:
        _min, _max = 0, 1 # TODO: Use utils.constants_task to get correct values

    try:
        coefficients, cov = fit_step2(data_by_name, task_key, None, _min=_min, _max=_max, use_log_sigmoid=False)

        (
            predicted_data_by_name, plotted_predicted_data,
            (y, y_pred, rel_error, delta_error), all_rel_errors,
        ) = predict_step2(
            configs, data_by_name, coefficients, cov, y_metric='rc_acc', use_log_sigmoid=False
        )

        plot_step2(
            configs, data_by_name, predicted_data_by_name, plotted_predicted_data, task_key, None, 'rc_bpb', 'rc_acc',
            coefficients, cov, use_log_sigmoid=False, add_texts=add_texts, ax=ax
        )
    except Exception as e:
        print(data_by_name)
        raise RuntimeError(f'Failed to fit: {e}')
    
    ax.set_title(task_name)
    ax.legend(fontsize=6)

    return (y, y_pred, rel_error, delta_error), all_rel_errors


def run_ladder_stacked(df, task_name, train_models, eval_models, ax=None, config_path=DEFAULT_CONIFG_PATH):
    # Get config
    configs = get_final_configs(config_path)

    # Fit step 1
    data_by_name = get_step1_data_by_name(configs, 'arc_easy_test_5shot', y_metric='rc_bpb', moving_avg=5) # we are only using this for the num tokens
    data_by_name_downstream = get_ladder_data(df, task_name, train_models, eval_models)
    data_by_name_step_1 = merge_dicts(data_by_name_downstream, data_by_name) # merge the 'ns', 'ds', 'ls', 'fs' keys into the step 2 data
    step1_coefficients, _ = fit_step1(data_by_name_step_1, 'rc_bpb')

    # workaround 5000. 2 AM. who needs sleep -- there are evals to run.
    for key in data_by_name_step_1:
        data_by_name_step_1[key]['xs'] = data_by_name_step_1[key]['ys']

    # Fit step 2
    data_by_name_step_2 = get_ladder_data(df, task_name, train_models, eval_models)
    task_key, configs = get_ladder_config(config_path, task_name, train_models, eval_models)

    _min, _max = None, None
    if task_key is None:
        _min, _max = 0, 1 # TODO: Use utils.constants_task to get correct values

    step2_coefficients, _ = fit_step2(data_by_name_step_2, task_key, None, _min=_min, _max=_max, use_log_sigmoid=False)

    # make predictions
    predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error) = predict_chained(
        data_by_name_step_1, step1_coefficients, step2_coefficients, use_log_sigmoid=False
    )

    # Plot
    plot_chained(
        configs,
        data_by_name_step_1,
        predicted_data_by_name,
        plotted_predicted_data_by_name,
        task_name,
        str_chained_fit(step1_coefficients, step2_coefficients, use_log_sigmoid=False),
        ax,
    )

    ax.set_title(task_name)
    ax.legend(fontsize=6, loc='upper left')

    n = 6887575552 
    d = 3945065873408 
    target_name = '7B-4T'

    # make predictions
    pred_loss = chinchilla_n_d_fit([n, d], step1_coefficients)
    fit_fn = sigmoid
    pred_acc = fit_fn(pred_loss, *step2_coefficients)
    data = data_by_name_step_2[target_name]
    actual_acc = data["ys"][-1]
    delta_error=pred_acc - actual_acc
    rel_error = np.abs(delta_error) / actual_acc
    # return {"Actual": y, "Pred": y_pred, "Rel Error": rel_error}
    return (y, y_pred, rel_error, delta_error), [rel_error]


def run_variance_analysis(df, tasks, eval_models, last_n_points=10, ax=None, config_path=DEFAULT_CONIFG_PATH):
    # Unfortunately there are local references, so we have to be in the OLMo repo
    os.chdir('/Users/dhei/ai2/new-evals/olmo-repos/OLMo')

    assert len(eval_models) == 1, 'one model at a time for now'

    #################################
    ### TODO: Have this not be copy-pasted from OLMo repo :(
    #################################

    num_tasks = len(tasks)

    if num_tasks < 4:
        n_groups = 1
        fig, axes = plt.subplots(
            num_tasks // n_groups, 2 * n_groups, figsize=(2.75 * 2 * n_groups, 2.5 * (num_tasks // n_groups))
        )
    else:
        # Create a figure with spacing between the two groups of tasks
        n_groups = 2
        fig = plt.figure(figsize=(2.75 * 2 * n_groups, 2.25 * num_tasks // n_groups))
        gs = fig.add_gridspec(
            (num_tasks // n_groups),
            (2 * n_groups) + 1,
            width_ratios=[1, 1, 0, 1, 1],
            wspace=0.4,
            hspace=0.4,
            left=0.05,
            right=0.97,
            bottom=0.05,
            top=0.94,
        )
        axes = []
        for row in range(num_tasks // n_groups):
            row_axes = []
            for col in [0, 1, 3, 4]:
                row_axes.append(fig.add_subplot(gs[row, col]))
            axes.append(row_axes)
        axes = np.array(axes)

    loss_std_devs = {}
    acc_std_devs = {}
    loss_coeffs = {}
    acc_coeffs = {}

    for r, task_name in tqdm(enumerate(tasks), total=len(tasks), desc='Running variance analysis'):
        steps = 'all'

        # Get config
        configs = get_final_configs(config_path)

        # Get data
        data_by_name_downstream = get_ladder_data(df, task_name, [], eval_models, step=steps)

        # Collapse lists of np arrays to lists
        for _, inner_dict in data_by_name_downstream.items():
            for inner_key, value in inner_dict.items():
                if isinstance(value[0], list):
                    assert len(value) == 1
                    inner_dict[inner_key] = value[0]

        # Get tokens for intermediate checkpoints
        try:
            data_by_name = get_step2_data_by_name(configs, 'arc_easy_test_5shot') # we are only using this for the num tokens
            data_by_name = {'peteish-moreeval-1B-5xC': data_by_name['1B']}
            data_by_name = merge_dicts(data_by_name_downstream, data_by_name) # merge the 'ns', 'ds', 'ls', 'fs' keys into the step 2 data
        
            # data_by_name = data_by_name_downstream
        except Exception as e:
            print(f'{task_name} failed: {e}')
            continue

        task_key, configs = get_ladder_config(config_path, task_name, [], eval_models)

        for name, data in data_by_name.items():
            config = configs[name]
            if config.mode == "eval":
                total_points = len(data["xs"])
                # start_point = int(np.ceil(0.1 * total_points)) # skip first 10% of points
                start_point = int(np.ceil(0.3 * total_points)) # skip first 30% of points

                ds = data["ds"][-last_n_points:]
                xs = data["xs"][-last_n_points:]
                ys = data["ys"][-last_n_points:]

                loss_std_dev = np.std(xs)
                loss_coeff_of_var = loss_std_dev / np.mean(xs)
                acc_std_dev = np.std(ys)
                acc_coeff_of_var = acc_std_dev / np.mean(ys)

                loss_std_devs[task_name] = loss_std_dev
                acc_std_devs[task_name] = acc_std_dev
                loss_coeffs[task_name] = loss_coeff_of_var
                acc_coeffs[task_name] = acc_coeff_of_var

                # Step 1
                ax1: plt.Axes = axes[(r * 2) // (2*n_groups)][(r * 2) % (2*n_groups)]

                for ax_ in [ax1]:
                    ax_.scatter(
                        data["ds"][start_point:-last_n_points],
                        data["xs"][start_point:-last_n_points],
                        # color=config.color,
                        color='b',
                        marker="o",
                        s=10,
                        alpha=0.3,
                        # label=config.label
                        label=f"{config.label} (intermediate checkpoints)",
                        # label=f"1B-10xC (intermediate checkpoints)",
                    )
                    ax_.scatter(
                        ds,
                        xs,
                        color="orange",
                        marker="o",
                        s=10,
                        alpha=0.5,
                        label=f"{config.label} (final {last_n_points} checkpoints)",
                        # label=f"1B-10xC (final {last_n_points} checkpoints)",
                    )

                ax1.set_xscale("log")
                ax1.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
                ax1.set_xlabel("Tokens (D)", fontsize=FONTSIZE)
                ax1.set_ylabel("Task loss", fontsize=FONTSIZE)
                ax1.set_title(
                    f"{task_name}\n"
                    + r"(Loss relative SD$_{10}$: "
                    + f"{loss_coeff_of_var*100:.2f}%)",
                    fontsize=FONTSIZE,
                    fontweight="bold",
                )
                ax1.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
                ax1.tick_params(axis="x", which="both", reset=True)
                ax1.tick_params(axis="x", which="minor", labelsize=0)

                # Step 2
                ax2: plt.Axes = axes[(r * 2) // (2*n_groups)][((r * 2) % (2*n_groups))+1]

                for ax_ in [ax2]:
                    ax_.scatter(
                        data["xs"][start_point:-last_n_points],
                        data["ys"][start_point:-last_n_points],
                        # color=config.color,
                        color='b',
                        marker="o",
                        s=10,
                        alpha=0.3,
                        label=f"{config.label} (intermediate checkpoints)",
                        # label=f"1B-10xC (intermediate checkpoints)",
                    )
                    ax_.scatter(
                        xs,
                        ys,
                        color="orange",
                        marker="o",
                        s=10,
                        alpha=0.5,
                        label=f"{config.label} (final {last_n_points} checkpoints)",
                        # label=f"1B-10xC (final {last_n_points} checkpoints)",
                    )

                ax2.legend(loc="upper right", ncols=1, fontsize=10)
                ax2.set_xlabel("Task loss", fontsize=FONTSIZE)
                ax2.set_ylabel("Task accuracy", fontsize=FONTSIZE) # Task RC accuracy
                ax2.set_title(
                    f"{task_name}\n"
                    + r"(Accuracy relative SD$_{10}$: "
                    + f"{acc_coeff_of_var*100:.2f}%)",
                    fontsize=FONTSIZE,
                    fontweight="bold",
                )
                ax2.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
                ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
                break

    
    # Collect all unique handles and labels
    all_handles = []
    all_labels = []
    for row in axes:
        for ax in row:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in all_labels:
                    all_handles.append(handle)
                    all_labels.append(label)

    # Remove redundant labels / legends
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i != len(axes) - 1:
                ax.set_xlabel("")
            if ax.get_legend():
                ax.get_legend().remove()

    # Add shared legend
    legend = fig.legend(
        all_handles,
        all_labels,
        loc="upper center",
        ncol=2,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, 1),  # 1
        handletextpad=0.3,
        columnspacing=0.7,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    if num_tasks < 4:
        fig.tight_layout(h_pad=0.02, rect=[0, 0, 1, 0.95])
    
    df = pd.merge(
        pd.merge(
            pd.DataFrame.from_dict(loss_std_devs, orient="index")
            .reset_index()
            .rename({0: "Loss SD", "index": "Task"}, axis=1),
            pd.DataFrame.from_dict(loss_coeffs, orient="index")
            .reset_index()
            .rename({0: "Loss Rel SD (CV)", "index": "Task"}, axis=1),
        ),
        pd.merge(
            pd.DataFrame.from_dict(acc_std_devs, orient="index")
            .reset_index()
            .rename({0: "Accuracy SD", "index": "Task"}, axis=1),
            pd.DataFrame.from_dict(acc_coeffs, orient="index")
            .reset_index()
            .rename({0: "Accuracy Rel SD (CV)", "index": "Task"}, axis=1),
        ),
    )

    return df 
    