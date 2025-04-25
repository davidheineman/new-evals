import os, sys

from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DATA_DIR, ROOT_DIR
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader import get_nd_array, get_slice, get_instance
from download.hf import pull_predictions_from_hf

from utils import get_title_from_task
from utils.constants_models import DDOS_MODEL_NAMES
from download.preprocess import is_excluded_from_lite

from irt_utils.irt_train import train
from irt_utils.irt_inference import calculate_theta, save_irt_params, load_irt_params

from irt_utils.birnbaum_two_param_logistic import Birnbaum
add_to_registry = Birnbaum.register("2pl_birnbaum")
add_to_registry(Birnbaum)


def train_irt_model(train_instance_names, train_model_names, train_scores):
    # ~/miniconda3/envs/metaeval_pyirt/lib/python3.10/site-packages/py_irt
    irt_model, irt_params = train(
        model_type="2pl_birnbaum", # 1pl, 2pl, 3pl, 4pl, amortized_1pl, 2pl_birnbaum
        predictions=train_scores.T, 
        model_names=train_model_names,
        instance_names=train_instance_names,
        priors="hierarchical", # vague, hierarchical
        epochs=1000,
        # hidden=30,
        # hidden=500,
        hidden=2000,
        # hidden=5000,
        lr=0.05,
        lr_decay=0.999,
        device='cpu', # cpu, cuda, mps
        quiet=True
    )

    disc_map = {irt_params['item_ids'][i]: irt_params['disc'][i] for i in range(len(irt_params['item_ids']))}
    diff_map = {irt_params['item_ids'][i]: irt_params['diff'][i] for i in range(len(irt_params['item_ids']))}

    discriminations = [disc_map[item_id] for item_id in train_instance_names]
    difficulties    = [diff_map[item_id] for item_id in train_instance_names]

    # Convert discrim params back from log space
    discriminations = np.exp(discriminations)

    return discriminations, difficulties


def get_test_data(df, metric, selected_tasks, LADDER_MODELS, new_task_entry_name='irt_aggregate'):
    test_instance_names = None
    full_scores = None
    test_model_names = []

    def update_task_name(selected_tasks, model_names, new_task_entry_name=new_task_entry_name):
        """
        ('peteish-moreeval-1B-5xC', 67500.0, 'peteish-ladder'), -> ('irt_aggregate', 'peteish-moreeval-1B-5xC', 67500.0, 'peteish-ladder'),
        ('peteish-moreeval-1B-5xC'), -> ('irt_aggregate', 'peteish-moreeval-1B-5xC', nan, nan),
        """
        # Update index entry for model for new task, if we are using an aggregate
        if len(selected_tasks) > 1:
            task_name = new_task_entry_name
        else:
            task_name = selected_tasks[0]

        new_model_names = []
        for entry in model_names:
            if isinstance(entry, str):
                entry = (entry,)
            new_entry = (task_name,) + entry
            # while len(new_entry) < 4:
            #     new_entry = new_entry + (float('nan'),)
            new_model_names.append(new_entry)
        return new_model_names

    # ladder models
    instance_names, model_names, scores = get_nd_array(
        df, col=['model'], metric=metric, # TODO: Get multi-index working
        task=selected_tasks, model=LADDER_MODELS,
        mix='peteish-ladder', step=None, return_index=True
    )
    scores = scores.squeeze()
    model_names = update_task_name(selected_tasks, model_names)
    assert scores.shape[0] == len(model_names)
    test_instance_names = instance_names
    test_model_names += model_names
    full_scores = scores

    # data decide models
    instance_names, model_names, scores = get_nd_array(
        df, col=['model'], metric=metric, # TODO: Get multi-index working
        task=selected_tasks, model=DDOS_MODEL_NAMES,
        mix=None, step=None, return_index=True
    )
    scores = scores.squeeze()
    model_names = update_task_name(selected_tasks, model_names)
    assert test_instance_names == instance_names
    assert scores.shape[0] == len(model_names)
    test_model_names += model_names
    full_scores = np.concatenate((full_scores, scores), axis=0)

    # peteish7
    instance_names, model_names, scores = get_nd_array(
        df, col=["model"], metric=metric, 
        task=selected_tasks, model=['peteish7'],
        mix=None, step=None, return_index=True
    )
    model_names = update_task_name(selected_tasks, model_names)
    assert scores.shape[0] == len(model_names)
    assert test_instance_names == instance_names
    test_model_names += model_names
    full_scores = np.concatenate((full_scores, scores), axis=0)

    # 1B models
    instance_names, model_names, scores = get_nd_array(
        df, col=['model', 'step', 'mix'], metric=metric,
        task=selected_tasks, model="peteish-moreeval-1B-5xC",
        mix=None, step=None, return_index=True
    )
    scores = scores.squeeze()
    model_names = update_task_name(selected_tasks, model_names)
    assert test_instance_names == instance_names
    assert scores.shape[0] == len(model_names)
    test_model_names += model_names
    full_scores = np.concatenate((full_scores, scores), axis=0)

    # 13B models
    instance_names, model_names, scores = get_nd_array(
        df, col=['model', 'step'], metric=metric,
        task=selected_tasks, model="peteish13-highlr",
        mix=None, step=None, return_index=True
    )
    scores = scores.squeeze()
    model_names = update_task_name(selected_tasks, model_names)
    assert test_instance_names == instance_names
    assert scores.shape[0] == len(model_names)
    test_model_names += model_names
    full_scores = np.concatenate((full_scores, scores), axis=0)

    assert full_scores.shape[0] == len(test_model_names)
    assert full_scores.shape[1] == len(instance_names)
    assert len(test_model_names) == len(set(test_model_names)), (len(test_model_names), len(set(test_model_names)))

    return test_instance_names, test_model_names, full_scores


def get_train_data(df, metric, tasks, all_models):
    # external models
    train_instance_names, train_model_names, train_scores = get_nd_array(
        df, col=["model"], metric=metric,
        task=tasks, model=all_models,
        mix=None, step=None, return_index=True
    )

    assert len(train_model_names) > 0, f'Found no results for task: {tasks}!'

    # Set NaN values to 0
    train_scores = np.nan_to_num(train_scores, nan=0)
    train_scores = np.round(train_scores).astype(int) # round scores to {0, 1} for IRT model

    return train_instance_names, train_model_names, train_scores


def normalize_scores(scores, _type='acc'):
    if _type == 'acc':
        # Set NaN values to 0 for acc
        scores = np.nan_to_num(scores, nan=0)
        scores = np.round(scores).astype(int) # round scores to {0, 1} for IRT model
    elif _type == 'bpb':
        # Set NaN values to 0 for BPB
        scores = np.nan_to_num(scores, nan=0)

        # Perform column-wise log transformation to scale instances to 0-1 based on all model BPB for that instance
        # scores = (np.log1p(scores) - np.min(np.log1p(scores), axis=0)) / (
        #     np.max(np.log1p(scores), axis=0) - np.min(np.log1p(scores), axis=0)
        # )

        # Perform column-wise min-max norm
        scores = (scores - np.min(scores, axis=0)) / (np.max(scores, axis=0) - np.min(scores, axis=0))

    return scores


def add_to_df_benchmarks(df_benchmarks, ability_dict):
    initial_len = len(df_benchmarks)
    print(f"Initial length: {initial_len}")

    # Update df_benchmarks with each ability dict entry separately
    for key, new_vals in ability_dict.items():
        # Get the first n values that exist in the key tuple
        key_values = key[:len(key)]
        cols = ['task', 'model', 'step', 'mix'][:len(key_values)]
        
        # Create a mask matching all rows where these values match
        mask = pd.Series(True, index=df_benchmarks.index)
        for col, val in zip(cols, key_values):
            mask &= df_benchmarks[col] == val
            
        # Check if task exists
        if mask.any():
            # (EXISTING TASKS) Update existing rows
            for val_col, val in new_vals.items():
                df_benchmarks.loc[mask, val_col] = val
        else:
            # (NEW TAKS) Task doesn't exist, create new rows using last n key values
            new_row = pd.DataFrame(columns=df_benchmarks.columns)

            # Get all rows for this WITHOUT task key
            key_values = key[1:len(key)]
            cols = ['model', 'step', 'mix'][:len(key_values)]
            mask = pd.Series(True, index=df_benchmarks.index)
            for col, val in zip(cols, key_values):
                mask &= df_benchmarks[col] == val

            # Copy over the masked rows and set task to key_values[0]
            new_row = df_benchmarks[mask].copy()

            # Only keep 1 column to add new task under
            new_row = new_row.drop_duplicates(subset=['model', 'step', 'mix'])

            # Set all columns not in ['task', 'model', 'step', 'mix'] to None
            for col in new_row.columns:
                if col not in ['task', 'model', 'step', 'mix', 'size', 'token_ratio', 'model_hash', 'model_config', 'step_str']:
                    new_row[col] = None

            new_row['task'] = new_row['task_name'] = key[0] # set new task name!
            
            # Update with new values
            for val_col, val in new_vals.items():
                new_row[val_col] = val

            # display(new_row)
            # raise RuntimeError
            
            # Explicitly exclude empty/NA columns before concat to avoid FutureWarning
            non_empty_cols = new_row.columns[~new_row.isna().all()] # depreciated in python 3.14 :(
            df_benchmarks = pd.concat([df_benchmarks, new_row[non_empty_cols]], ignore_index=True)

    final_len = len(df_benchmarks)
    print(f"Final length: {final_len}")
    print(f"Difference in length: {final_len - initial_len}")
    return df_benchmarks


def main():
    # Load benchmark scores
    local_path = pull_predictions_from_hf("allenai/ladder-evals", "benchmarks")
    df_benchmarks = pd.read_parquet(local_path)
    df_benchmarks.loc[df_benchmarks['mix'] == 'baseline', 'mix'] = 'dolma17' # fix for the names of one of Ian's data mixes

    # Load instance-level results
    # local_path = pull_predictions_from_hf("allenai/ladder-evals", "instanceslite") # OLMES RC only
    local_path = pull_predictions_from_hf("allenai/ladder-evals", "instancesmedium") # OLMES RC, MC, OLMES GEN, Minerva, MBPP, HumanEval
    # local_path = pull_predictions_from_hf("allenai/ladder-evals", "instances") # All tasks
    COLS = ['step', 'model', 'task', 'mix', 'size', 'token_ratio', 'native_id', 'primary_score', 'logits_per_byte_corr']
    df_instances = pd.read_parquet(local_path, columns=COLS)
    print(f'Loaded {len(df_instances):,} instance results')

    MODELS = sorted(df_instances.index.get_level_values('model').unique().to_list())
    TASKS  = sorted(df_instances.index.get_level_values('task').unique().to_list())

    # Get a list of models to exclude from our "external models"
    LADDER_MODELS = [model for model in MODELS if 'peteish-moreeval' in model]
    external_models = sorted([
        model for model in MODELS 
        if model not in
            DDOS_MODEL_NAMES + # exclude 1B-5xC models
            LADDER_MODELS + # exclude ladder models
            ['peteish13-highlr'] # exclude intermediate checkpoints from 13B
        and not is_excluded_from_lite(model)
    ])

    mmlu      = [t for t in TASKS if 'mmlu' in t and ':' not in t and '_pro_' not in t]
    minerva   = [t for t in TASKS if 'minerva' in t and ':' not in t and 'math_500' not in t]
    mmlu_pro  = [t for t in TASKS if '_pro_' in t and ':rc' in t]
    mmlu_mc   = [t for t in TASKS if 'mmlu' in t and ':mc' in t and '_pro_' not in t]
    olmes     = ['arc_challenge', 'arc_easy', 'boolq', 'csqa', 'hellaswag', 'openbookqa', 'piqa', 'socialiqa', 'winogrande']
    olmes_mc  = [f'{task}:mc' for task in olmes]
    olmes_para        = [f'{task}:para' for task in olmes]
    olmes_distractors = [f'{task}:distractors' for task in olmes]
    olmes_enlarge     = [f'{task}:enlarge' for task in olmes]
    # olmes_gen = ['drop', 'gsm8k', 'jeopardy', 'naturalqs', 'squad', 'triviaqa']
    # olmes_gen = ['drop', 'gsm8k', 'jeopardy', 'squad', 'triviaqa'] # Accidentally excluded GSM, SQuAD has problems at inference
    olmes_gen = ['drop', 'jeopardy', 'triviaqa']
    agi_eval  = [t for t in TASKS if 'agi_eval' in t and ':' not in t]
    bbh       = [t for t in TASKS if 'bbh' in t and ':' not in t]
    paloma    = [t for t in TASKS if 'paloma' in t]
    llm_compression = [t for t in TASKS if 'llm_compression' in t]
    custom_loss = [t for t in TASKS if 'custom_loss' in t]
    
    selected_tasks = []
    selected_tasks += [olmes, mmlu, minerva, olmes_mc, mmlu_mc] # olmes_gen ( causing issues :( )
    selected_tasks += olmes + olmes_gen + mmlu + olmes_mc + mmlu_mc
    selected_tasks += minerva + ['mbpp', 'mbppplus', 'codex_humaneval', 'codex_humanevalplus']
    selected_tasks += ['autobencher', 'autobencher:mc']
    selected_tasks += ['paloma_c4_en', 'paloma_m2d2_s2orc_unsplit']

    selected_task_sets = [[task] if not isinstance(task, list) else task for task in selected_tasks] # convert to lists

    train_irt_models = False
    if train_irt_models:
        for i, task_set in enumerate(selected_task_sets):
            task_name = get_title_from_task((task_set if len(task_set) > 1 else task_set[0]))
            
            print(f'Training IRT model for {task_name} on task set: {task_set} ({i}/{len(selected_task_sets)})')
            
            # Get data
            train_instance_names, train_model_names, train_scores = \
                get_train_data(df_instances, "primary_score", task_set, external_models)

            train_scores = normalize_scores(train_scores, _type='acc')

            # Train IRT model
            discriminations, difficulties = train_irt_model(train_instance_names, train_model_names, train_scores)

            # Eval trained IRT model to save reference scores (not used at inference)
            thetas = calculate_theta(difficulties, discriminations, train_scores)

            # Sort discriminations and difficulties by instance name
            sorted_indices       = np.argsort(train_instance_names)
            train_instance_names = np.array(train_instance_names)[sorted_indices]
            discriminations      = np.array(discriminations)[sorted_indices]
            difficulties         = np.array(difficulties)[sorted_indices]

            save_irt_params(
                save_path=Path(DATA_DIR) / "irt" / f"{task_name}.json",
                train_model_names=train_model_names,
                train_instance_names=train_instance_names,
                discriminations=discriminations,
                difficulties=difficulties,
                thetas=thetas,
            )

    for i, task_set in enumerate(selected_task_sets):
        task_name = get_title_from_task((task_set if len(task_set) > 1 else task_set[0]))

        print(f'Computing IRT model for {task_name} on task set: {task_set} ({i}/{len(selected_task_sets)})')

        try:
            train_instance_names, discriminations, difficulties = load_irt_params(
                load_path=Path(DATA_DIR) / "irt" / f"{task_name}.json",
            )
        except FileNotFoundError as e:
            print(f'Could not find IRT model for {task_name}. Skipping...')
            continue

        test_instance_names, test_model_names, test_scores_acc = \
            get_test_data(
                df_instances, "primary_score", task_set, 
                LADDER_MODELS=LADDER_MODELS, new_task_entry_name=task_name
            )
        test_instance_names, test_model_names, test_scores_bpb = \
            get_test_data(
                df_instances, "logits_per_byte_corr", task_set, 
                LADDER_MODELS=LADDER_MODELS, new_task_entry_name=task_name
            )

        test_scores_acc = normalize_scores(test_scores_acc, _type='acc')
        test_scores_bpb = normalize_scores(test_scores_bpb, _type='bpb')

        # Sort test instances by the train IRT param ordering
        test_to_train_idx = {test_instance: i for i, test_instance in enumerate(train_instance_names)}
        reorder_idx = [test_to_train_idx[test_instance] for test_instance in test_instance_names]
        test_scores_acc = test_scores_acc[:, reorder_idx]
        test_scores_bpb = test_scores_bpb[:, reorder_idx]
        test_instance_names = [test_instance_names[i] for i in reorder_idx]

        assert train_instance_names == test_instance_names, (train_instance_names, test_instance_names)

        # Compute IRT scores
        thetas_acc = calculate_theta(difficulties, discriminations, test_scores_acc)
        thetas_bpb = calculate_theta(difficulties, discriminations, test_scores_bpb, func="gaussian")

        ability_dict = {}
        for name, theta_acc, theta_bpb in zip(test_model_names, thetas_acc, thetas_bpb):
            ability_dict[name] = {
                'theta_primary_score': theta_acc,
                'theta_bpb': theta_bpb
            }
        
        # Add scores back to dataframe
        df_benchmarks = add_to_df_benchmarks(df_benchmarks, ability_dict)

    # Save df with IRT scores
    df_benchmarks.to_parquet(Path(DATA_DIR) / "df_benchmarks_with_irt.parquet")


if __name__ == '__main__':
    main()