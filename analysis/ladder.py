import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from dataloader import get_nd_array
from download.preprocess import SIZE_PREFIXES, SIZE_PREFIXES_FIX, str_find

from olmo.scaling.scaling_laws.utils import FinalConfig
from olmo.scaling.scaling_laws.utils import get_final_configs, get_step2_data_by_name
from olmo.scaling.scaling_laws.utils import get_final_configs, get_step1_data_by_name
from olmo.scaling.scaling_laws.fitting_functions import chinchilla_n_d_fit, sigmoid

from scaling.step1 import fit_step1, predict_step1, plot_step1, str_chinchilla_n_d_fit
from scaling.step2 import fit_step2, predict_step2, plot_step2
from scaling.predict import predict_chained, plot_chained, str_chained_fit
from scaling.variance_analysis import compute_variance, plot_variance_analysis

from scaling.step1_flops import fit_step1 as fit_step1_flops, predict_step1 as predict_step1_flops, plot_step1 as plot_step1_flops, str_chinchilla_flops_fit
from scaling.predict_flops import predict_chained_flops, plot_chained as plot_chained_flops, str_chained_fit as str_chained_fit_flops

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

def add_ladder_data_cheap_decisions(data_by_name):
    # Manual override for Ian's toks seen
    TOK_SEEN_IAN_5XC = {
        '150M': 15003942912,
        '300M': 30006968320,
        '530M': 53009711104,
        '750M': 75012636672,
        # '750M': 69909479424,
        '1.3B': 100015669248,
    }
    for k, v in data_by_name.items():
        data_by_name[k]['ds'] = [TOK_SEEN_IAN_5XC[k]]

    # Add numbers for Ian's estimated FLOPs
    MODEL_FLOPS = {
        "150M": 1903391232,
        "300M": 3443922944,
        "530M": 5180751744,
        "750M": 6373843968,
        "1.3B": 10109071360,
    }
    for k, v in data_by_name.items():
        d = TOK_SEEN_IAN_5XC[k]
        f = float(d * MODEL_FLOPS[k])
        data_by_name[k]["fs"] = [f]

    return data_by_name


def sort_experiment_names(experiments):
    """ Sort a list of ladder model names by size, then token multiplier """
    def extract_sort_key(entry):
        parts = entry.split('-')
        size_str, xc_str = parts[-2], parts[-1]
        
        if "M" in size_str:
            size = int(size_str.replace("M", "")) * 1e6
        elif "B" in size_str:
            size = int(size_str.replace("B", "")) * 1e9
        else:
            raise ValueError(size_str)
        
        xc = float(xc_str.replace("xC", ""))
        return (size, xc)

    return sorted(experiments, key=extract_sort_key)


def merge_dicts(dict1, dict2, overwrite_xs=False):
    """ Merge the data_by_name of dict2 into dict1 """
    if dict1.keys() != dict2.keys():
        raise ValueError(f"Keys of dict1 and dict2 do not match. Seeing:\n{dict1.keys()}\n{dict2.keys()}")
    
    for key in dict1:
        l1, l2 = len(dict1[key]['xs']), len(dict2[key]['xs'])

        # Sort all values by the number of tokens seen
        indices = sorted(range(len(dict2[key]['ds'])), key=lambda i: dict2[key]['ds'][i])
        dict2[key]['ds'] = [dict2[key]['ds'][i] for i in indices]
        dict2[key]['ns'] = [dict2[key]['ns'][i] for i in indices]
        dict2[key]['ls'] = [dict2[key]['ls'][i] for i in indices]
        if 'fs' in dict2: dict2[key]['fs'] = [dict2[key]['fs'][i] for i in indices]
        if overwrite_xs:  dict2[key]['xs'] = [dict2[key]['xs'][i] for i in indices]

        if l1 != l2:
            # A faustian bargain to allow us to use the wandb tokens w/ oe-eval for intermediate ckpts, since we have different numbers
            # of checkpoints for both. 
            if l1 < l2:
                # Shorter is dict1[key]['xs'], sample dict2[key]['xs']
                indices = np.linspace(0, l2 - 1, l1, dtype=int)
                dict1[key]['ns'] = [dict2[key]['ns'][i] for i in indices]
                dict1[key]['ds'] = [dict2[key]['ds'][i] for i in indices]
                dict1[key]['ls'] = [dict2[key]['ls'][i] for i in indices]
                dict1[key]['fs'] = [None for _ in indices]
                if overwrite_xs:  dict1[key]['xs'] = [dict2[key]['xs'][i] for i in indices]
            else:
                raise RuntimeError(f'different sized lists: {l1}, {l2}')
        else:
            dict1[key]['ns'] = dict2[key]['ns']
            dict1[key]['ds'] = dict2[key]['ds']
            dict1[key]['ls'] = dict2[key]['ls']
            dict1[key]['fs'] = [None for _ in indices]
            if overwrite_xs:  dict1[key]['xs'] = dict2[key]['xs']

    return dict1


def map_corr_labels(bpb, corr, task_name):
    """ Given a tensor of tensors and a tensor of indices, map the indices to their entries """
    if bpb.ndim not in {1, 2, 3}:
        raise ValueError(bpb)

    n_choices = bpb[0].shape[-1]
    correct_bpb = np.empty_like(corr, dtype=np.float64)

    if bpb.ndim == 1:  # Handle 1D case
        for i in range(corr.shape[0]):
            if corr[i] == n_choices and 'enlarge' in task_name:
                corr[i] -= 1
            correct_bpb[i] = bpb[i][int(corr[i])]

    elif bpb.ndim == 2:  # Handle 2D case
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if corr[i, j] == n_choices and 'enlarge' in task_name:
                    corr[i, j] -= 1
                correct_bpb[i, j] = bpb[i, j][corr[i, j].astype(np.int32)]

    else:  # bpb.ndim == 3, Handle 3D case
        for k in range(corr.shape[0]):
            for i in range(corr.shape[1]):
                for j in range(corr.shape[2]):
                    if corr[k, i, j] == n_choices and 'enlarge' in task_name:
                        corr[k, i, j] -= 1
                    correct_bpb[k, i, j] = bpb[k, i, j][corr[k, i, j].astype(np.int32)]

    return correct_bpb


def get_ladder_data(df, task_name, train_models, eval_models, step='max'):
    """ Get slices of df and convert to ladder prediction format """
    if isinstance(task_name, list):
        assert len(task_name), f'Task suite needs to be only one list: {task_name}'
        task_name = task_name[0]

    data_by_name = defaultdict(dict)

    for model in train_models + eval_models:
        if model in eval_models:
            mode = 'eval'
        elif model in train_models:
            mode = 'train'
        
        # Fix string names for ladder models to match ladder
        size = str_find(SIZE_PREFIXES, model)
        if model == 'peteish7':
            size = '7B-4T'
        elif 'peteish13-highlr' in model:
            size = '13B-5T'
        elif size is not None: # and size != '-3B-'
            size = size.replace('-', '')
            size = SIZE_PREFIXES_FIX.get(size, size)
        else:
            size = model
        
        # # super hacky temporary thing (pls fix soon)
        loss_task_name = None
        # if 'mmlu_pro_' in task_name:
        #     loss_task_name = task_name.replace(':cot', '') # this uses the :cot responses, but :rc is easier to predict
        # elif ':cot' not in task_name:
        #     loss_task_name = f'{task_name}:perturb_rc'
        # elif ':cot' in task_name:
        #     loss_task_name = task_name.replace(':cot', ':perturb_cot')
        
        if 'exact_match' in df.columns:
            # Load generative benchmarks
            metric_names = ["correct_choice", "logits_per_byte", "acc_per_char", "exact_match", "pass_at_1", "pass_at_10"]
        else:
            metric_names = ["correct_choice", "logits_per_byte", "acc_per_char"]

        if step == 'max':
            _, scores = get_nd_array(df, "model", metric_names, model=model, task=task_name, step="max")
            if len(scores) == 6:
                corr, bpb, acc, exact_match, pass_at_1, pass_at_10 = scores
            else:
                corr, bpb, acc = scores
                exact_match = np.array([])
                pass_at_1 = np.array([])
                pass_at_10 = np.array([])
        else:
            # get results from multiple steps
            if step == 'all': step = None
            m1, corr        = get_nd_array(df, ['model', 'step', 'mix'], 'correct_choice', model=model, task=task_name, step=step)
            m2, bpb         = get_nd_array(df, ['model', 'step', 'mix'], 'logits_per_byte', model=model, task=task_name, step=step)
            m3, acc         = get_nd_array(df, ['model', 'step', 'mix'], 'acc_per_char', model=model, task=task_name, step=step)
            if len(metric_names) == 6:
                m4, exact_match = get_nd_array(df, ['model', 'step', 'mix'], 'exact_match', model=model, task=task_name, step=step)
                m5, pass_at_1   = get_nd_array(df, ['model', 'step', 'mix'], 'pass_at_1', model=model, task=task_name, step=step)
                m6, pass_at_10  = get_nd_array(df, ['model', 'step', 'mix'], 'pass_at_10', model=model, task=task_name, step=step)
            else:
                exact_match = np.array([])
                pass_at_1 = np.array([])
                pass_at_10 = np.array([])

        if loss_task_name is not None:
            # Eventually I can delete all this when I stop using :perturb_cot for gold perplexity
            if step == 'max':
                m1, acc        = get_nd_array(df, 'model', 'acc_per_char', model=model, task=task_name, step=step)
                m2, bpb        = get_nd_array(df, 'model', 'logits_per_byte', model=model, task=loss_task_name, step=step)
                m3, corr       = get_nd_array(df, 'model', 'correct_choice', model=model, task=loss_task_name, step=step)
            else:
                m1, acc        = get_nd_array(df, ['model', 'step', 'mix'], 'acc_per_char', model=model, task=task_name, step=step)
                m2, bpb        = get_nd_array(df, ['model', 'step', 'mix'], 'logits_per_byte', model=model, task=loss_task_name, step=step)
                m3, corr       = get_nd_array(df, ['model', 'step', 'mix'], 'correct_choice', model=model, task=loss_task_name, step=step)

            # Ensure the model results are all the same model names
            # assert [_1 == _2 == _3 for _1, _2, _3 in zip(m1, m2, m3)], f'{model} failed'
            if not all(_1 == _2 == _3 for _1, _2, _3 in zip(m1, m2, m3)):
                print(f"{model} result set mismatch task {task_name}.")

                if corr.ndim == 3:
                    print('Fixing...')
                    # manual fix for olmes gen benchmarks (since the num checkpoints may not match)
                    mask = np.isin(m3, m1) # corr indices in acc
                    corr = corr[:, mask, :]

                    mask = np.isin(m2, m1) # bpb indices in acc
                    bpb = bpb[:, mask, :]
                else:
                    continue

        if exact_match.size != 0:
            acc = exact_match # if there's exact match, use it as the primary metric
        if pass_at_1.size != 0 and np.all(~np.isnan(pass_at_1.astype(float))):
            acc = pass_at_1 # if there's pass@1, use it
        if pass_at_10.size != 0 and np.all(~np.isnan(pass_at_10.astype(float))):
            acc = pass_at_10 # if there's pass@10, use it

        # If correct_choice is not set right (nan), set to 0
        corr = np.array(corr, dtype=float)
        corr = np.nan_to_num(corr, nan=0)

        if bpb.size == 0:
            # raise RuntimeError(f'bpb is empty array: {bpb} on model: {model}')
            print(f'bpb is empty array: {bpb} on model "{model}" for task "{task_name}"')
            continue

        if len(corr) == 0 or len(bpb) == 0 or len(acc) == 0: 
            if mode == 'eval':
                print(f'Eval point data not found: {model}')
                continue
        
        # if isinstance(task_name, list):
        #     print('Need to compute a weighted average here!!')

        correct_bpb = map_corr_labels(bpb, corr, task_name=task_name)

        acc = acc.mean(axis=-1)
        correct_bpb = correct_bpb.mean(axis=-1)

        if 'xs' not in data_by_name[size]: data_by_name[size]['xs'] = []
        if 'ys' not in data_by_name[size]: data_by_name[size]['ys'] = []

        # Convert output to float / list of floats
        if correct_bpb.size == 1:
            correct_bpb = correct_bpb.item() 
        else:
            correct_bpb = correct_bpb.squeeze().tolist()
        
        if acc.size == 1:
            acc = acc.item()
        else:
            acc = acc.squeeze().tolist()

        if model == 'peteish-moreeval-1B-10xC' and task_name == 'gsm8k':
            # manual fix for broken model
            correct_bpb = 0.5828522670464399
        
        data_by_name[size]['xs'] += [correct_bpb]
        data_by_name[size]['ys'] += [acc]
        data_by_name[size]['mode'] = mode

    return data_by_name


def create_ladder_config(config_path, task_name, train_models, eval_models):
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


def run_ladder(
        df, task_name, train_models, eval_models, config_path,
        y_metric='rc_bpb', use_flops=False, run_step1=True, run_step2=True, run_stacked=True,
        axes=None, add_texts=False, return_preds=False):
    # Unfortunately there are local references, so we have to be in the OLMo repo
    os.chdir('/Users/dhei/ai2/new-evals/olmo-repos/OLMo')

    rel_error_step_1, rel_error_step_2, rel_error_stacked = None, None, None

    # Get config
    configs = get_final_configs(config_path)

    # Get data
    data_by_name = get_ladder_data(df, task_name, train_models, eval_models)

    ax_i = 0
    if run_step1:
        # Add token data
        if 'cheap_decisions' in config_path:
            data_by_name = add_ladder_data_cheap_decisions(data_by_name)
        else:
            data_by_name_tokens = get_step1_data_by_name(configs, 'arc_easy_test_5shot', y_metric=y_metric, moving_avg=1) # we are only using this for the num tokens    
            data_by_name = merge_dicts(data_by_name, data_by_name_tokens, overwrite_xs=(y_metric == 'c4')) # merge the 'ns', 'ds', 'ls', 'fs' keys into the step 2 data

        # Fit step 1
        if use_flops:
            step1_coefficients, cov = fit_step1_flops(data_by_name, y_metric)
        else:
            step1_coefficients, cov = fit_step1(data_by_name, y_metric)

        if use_flops:
            (
                predicted_data_by_name, plotted_predicted_data,
                (y, step_1_y_pred, rel_error_step_1), all_rel_errors,
            ) = predict_step1_flops(
                configs, data_by_name, step1_coefficients, y_metric=y_metric, 
            )
        else:
            (
                predicted_data_by_name, plotted_predicted_data,
                (y, step_1_y_pred, rel_error_step_1), all_rel_errors,
            ) = predict_step1(
                configs, data_by_name, step1_coefficients, y_metric=y_metric, 
            )

        # Plot step 1
        if axes is not None:
            ax = axes[ax_i]
            ax_i += 1
            if use_flops:
                plot_step1_flops(
                    configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    task_name, str_chinchilla_flops_fit(step1_coefficients), y_metric,
                    step1_coefficients, cov, ax,
                )
            else:
                plot_step1(
                    configs, data_by_name, predicted_data_by_name, plotted_predicted_data,
                    task_name, str_chinchilla_n_d_fit(step1_coefficients), y_metric,
                    step1_coefficients, cov, ax,
                )

    if 'cheap_decisions' in config_path:
        # Use intermediate checkpoints to fit step 2
        data_by_name = get_ladder_data(df, task_name, train_models, eval_models, step='all')
        for k1, v1 in data_by_name.items():
            for k2, v2 in v1.items():
                if isinstance(v2, list):
                    import math
                    # use last 90% of checkpoints
                    data_by_name[k1][k2] = v2[0][math.ceil(0.1 * len(v2[0])):]

    if run_step2:
        task_key, configs = create_ladder_config(config_path, task_name, train_models, eval_models)
        # configs = {name: config for name, config in configs.items() if config.paths is not None} # TODO: Any reason for this?

        _min, _max = None, None
        if task_key is None:
            _min, _max = 0, 1 # TODO: Use utils.constants_task to get correct values

        try:
            # Fit step 2
            step2_coefficients, cov = fit_step2(data_by_name, task_key, y_metric=None, _min=_min, _max=_max, use_log_sigmoid=False)

            (
                predicted_data_by_name, plotted_predicted_data,
                (y, step_2_y_pred, rel_error_step_2, delta_error), all_rel_errors,
            ) = predict_step2(
                configs, data_by_name, step2_coefficients, cov, y_metric=None, use_log_sigmoid=False
            )

            # Plot step 2
            if axes is not None:
                ax = axes[ax_i]
                ax_i += 1
                plot_step2(
                    configs, data_by_name, predicted_data_by_name, plotted_predicted_data, task_key, None, y_metric, 'rc_acc',
                    step2_coefficients, cov, use_log_sigmoid=False, add_texts=add_texts, ax=ax
                )
        except Exception as e:
            print(data_by_name)
            raise RuntimeError(f'Step 2 failed to fit: {e}')
        
    if 'cheap_decisions' in config_path:
        # Get step 1 data again (necessary if running with intermediate checkpoints)
        data_by_name = get_ladder_data(df, task_name, train_models, eval_models)
        configs = get_final_configs(config_path)
        if 'cheap_decisions' in config_path:
            data_by_name = add_ladder_data_cheap_decisions(data_by_name)
        else:
            data_by_name_tokens = get_step1_data_by_name(configs, 'arc_easy_test_5shot', y_metric=y_metric, moving_avg=1) # we are only using this for the num tokens
            data_by_name = merge_dicts(data_by_name, data_by_name_tokens, overwrite_xs=(y_metric == 'c4')) # merge the 'ns', 'ds', 'ls', 'fs' keys into the step 2 data

    if run_stacked:
        assert run_step2, 'For now, you must run step 2 to get stacked preds!'

        # Predict stacked
        if use_flops:
            (
                predicted_data_by_name, plotted_predicted_data_by_name, 
                (y, stacked_y_pred, rel_error)
            ) = predict_chained_flops(
                data_by_name, step1_coefficients, step2_coefficients
            )
        else:
            (
                predicted_data_by_name, plotted_predicted_data_by_name, 
                (y, stacked_y_pred, rel_error)
            ) = predict_chained(
                data_by_name, step1_coefficients, step2_coefficients, use_log_sigmoid=False
            )

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

        if 'peteish7' in eval_models:
            # make 7B prediction
            n = 6887575552 
            d = 3945065873408 
            target_name = '7B-4T'

            pred_loss = chinchilla_n_d_fit([n, d], step1_coefficients)
            fit_fn = sigmoid
            pred_acc = fit_fn(pred_loss, *step2_coefficients)
            data = data_by_name[target_name]
            rel_error_stacked = 0
            if "ys" in data:
                actual_acc = data["ys"][-1]
                delta_error=pred_acc - actual_acc
                rel_error_stacked = np.abs(delta_error) / actual_acc if actual_acc > 0 else float('inf')

    if axes is not None:
        for ax in axes:
            ax.set_title(task_name)
            ax.legend(fontsize=6)

    if return_preds:
        return (rel_error_step_1, rel_error_step_2, None), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    return rel_error_step_1, rel_error_step_2, rel_error_stacked


def run_variance_analysis(df, tasks, eval_models, last_n_points=10, ax=None, config_path=DEFAULT_CONIFG_PATH):
    # Unfortunately there are local references, so we have to be in the OLMo repo
    os.chdir('/Users/dhei/ai2/new-evals/olmo-repos/OLMo')

    assert len(eval_models) == 1, 'one model at a time for now'

    variance_results = {}

    for r, task_name in tqdm(enumerate(tasks), total=len(tasks), desc='Running variance analysis'):
        try:
            # Get config
            configs = get_final_configs(config_path)

            # Get data
            data_by_name_downstream = get_ladder_data(df, task_name, [], eval_models, step='all')

            # Collapse lists of np arrays to lists
            for _, inner_dict in data_by_name_downstream.items():
                for inner_key, value in inner_dict.items():
                    if isinstance(value[0], list):
                        assert len(value) == 1
                        inner_dict[inner_key] = value[0]

            # Get N tokens for intermediate checkpoints
            data_by_name = get_step2_data_by_name(configs, 'arc_easy_test_5shot') # we are only using this for the num tokens
            data_by_name = {'1.3B': data_by_name['1.3B']}

            def crop_dict_values(data_by_name_downstream):
                return {key: (value if not isinstance(value, list) else value[:163])
                        for key, value in data_by_name_downstream.items()}
            data_by_name_downstream = {outer_key: crop_dict_values(inner_dict) for outer_key, inner_dict in data_by_name_downstream.items()}

            data_by_name = merge_dicts(data_by_name_downstream, data_by_name) # merge the 'ns', 'ds', 'ls', 'fs' keys into the step 2 data

            task_key, configs = create_ladder_config(config_path, task_name, [], eval_models)

            ####
            # Copy-pasted from OLMo repo (todo: roll into one)
            ####
            assert len(data_by_name) == 1
            name = list(data_by_name.keys())[0]
            data = data_by_name[name]
            
            config = configs[name]
            
            ds = data["ds"][-last_n_points:]
            xs = data["xs"][-last_n_points:]
            ys = data["ys"][-last_n_points:]

            loss_std_dev = np.std(xs)
            loss_coeff_of_var = loss_std_dev / np.mean(xs)
            acc_std_dev = np.std(ys)
            acc_coeff_of_var = acc_std_dev / np.mean(ys)

            variance_results[task_name] = {
                'config': config,
                'data': data,
                'last_n_points': last_n_points,
                'loss_std_dev': loss_std_dev,
                'acc_std_dev': acc_std_dev,
                'loss_coeff_of_var': loss_coeff_of_var,
                'acc_coeff_of_var': acc_coeff_of_var,
            }
            ####
            # Copy-pasted from OLMo repo
            ####
        except Exception as e:
            print(f'{task_name} failed: {e}')
            continue

    fig, df = plot_variance_analysis(config, variance_results, last_n_points)

    return df
    