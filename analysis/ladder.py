import os
import matplotlib.pyplot as plt
import numpy as np
from dataloader import get_nd_array
from download.preprocess import SIZE_PREFIXES, str_find
from collections import defaultdict

from olmo.scaling.scaling_laws.utils import MODEL_FLOPS, FinalConfig
from olmo.scaling.scaling_laws.utils import get_final_configs, get_step2_data_by_name, get_task_sets

from scaling.step2 import main as step2_main
from scaling.step2 import fit_step2, predict_step2, plot_step2


DEFAULT_CONIFG_PATH = "scripts/scaling/final.json"


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
    'OLMo': 'yellow',
    'pythia': 'brown',
    'gemma': 'teal',
    'phi': 'black',
    'deepseek': 'pink',
    'zephyr': 'green',
    'neo': 'orange',
    'falcon': 'blue'
}


def get_ladder_data(df, task_name, train_models, eval_models):
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
            if size is not None and size != '-3B-': 
                size = size.replace('-', '')
            else:
                size = model
        else:
            raise NameError(model)

        m1, corr = get_nd_array(df, 'model', 'correct_choice', model=model, task=task_name, step='max')
        m2, bpb  = get_nd_array(df, 'model', 'logits_per_byte', model=model, task=task_name, step='max')
        m3, acc  = get_nd_array(df, 'model', 'acc_per_char', model=model, task=task_name, step='max')

        # Ensure the model results are all the same model names
        # assert [_1 == _2 == _3 for _1, _2, _3 in zip(m1, m2, m3)], f'{model} failed'
        if not all(_1 == _2 == _3 for _1, _2, _3 in zip(m1, m2, m3)):
            print(f"{model} failed")
            continue

        # if mode == 'eval': 
        #     print(corr)
        #     print(acc)
        #     print(size)

        if len(corr) == 0 or len(bpb) == 0 or len(acc) == 0: 
            if mode == 'eval':
                raise RuntimeError(f'Eval point data not found: {model}')
            # continue
        
        # if isinstance(task_name, list):
        #     print('Need to compute a weighted average here!!')

        # Get correct logprobs per char
        n_choices = bpb[0][0].shape
        correct_bpb = np.empty_like(corr, dtype=np.float64)
        rows, cols = corr.shape
        for i in range(rows):
            for j in range(cols):
                if corr[i, j] == n_choices and 'enlarge' in task_name: 
                    # print(f'Warning: bpb has {n_choices} choices, but the correct label is {corr[i, j]} (did ChatGPT generate an incorrect ground truth?). re-indexing the correct label...')
                    corr[i, j] -= 1
                correct_bpb[i, j] = bpb[i, j][corr[i, j].astype(np.int32)]

        acc = acc.mean(axis=1)
        correct_bpb = correct_bpb.mean(axis=1)

        if 'xs' not in data_by_name[size]: data_by_name[size]['xs'] = []
        if 'ys' not in data_by_name[size]: data_by_name[size]['ys'] = []

        data_by_name[size]['xs'] += [correct_bpb.item()]
        data_by_name[size]['ys'] += [acc.item()]
        data_by_name[size]['mode'] = mode

    return data_by_name


def run_ladder_step_2(df, task_name, train_models, eval_models, ax=None, config_path=DEFAULT_CONIFG_PATH):
    # Unfortunately there are local references, so we have to be in the OLMo repo
    os.chdir('/Users/dhei/ai2/new-evals/olmo-repos/OLMo')

    # arc_easy:enlarge => arc_easy
    task_root = task_name.split(':')[0] if isinstance(task_name, str) else None

    data_by_name = get_ladder_data(df, task_name, train_models, eval_models)

    # Get ladder model configs
    configs = get_final_configs(config_path)

    # Add new models to config
    for model in eval_models:
        if model not in configs.keys():
            # Get model color
            color = 'red'
            for key, value in COLOR_MAP.items():
                if key in model:
                    color = value
                    break
            
            # Create dummy config for new eval points
            configs[model] = FinalConfig(
                paths=None, mode='eval', n=0, label=model, color=color, use_last_n_percentage=100
            )

    task_key = TASK_KEY_MAP.get(task_root, None) # the task key is used to get min/max perf and plot title

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
            coefficients, cov, use_log_sigmoid=False, ax=ax
        )
    except Exception as e:
        print(data_by_name)
        raise RuntimeError(f'Failed to fit: {e}')

    return y, y_pred, rel_error, delta_error