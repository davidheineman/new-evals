""" Compute decision acc / prediction error by masking instances """

import numpy as np
import pandas as pd
from scipy import stats

from ladder_wrapper import run_ladder
from datadecide import decision_acc_fast
from utils import ROOT_DIR


def compute_snr(step_scores, datadecide_scores, step_mask, dd_mask):
    """Compute SNR for a subset of instances."""
    # Compute means
    datadecide_means   = np.nanmean(datadecide_scores[:, dd_mask], axis=1)
    step_to_step_means = np.nanmean(step_scores[:, step_mask], axis=1)
    final_30_means = step_to_step_means[-30:]

    # Signal
    # rel_signal = np.std(datadecide_means) / np.mean(datadecide_means)
    rel_signal = np.std(datadecide_means) / np.mean(final_30_means)

    # Noise
    rel_noise = np.std(final_30_means) / np.mean(final_30_means)

    # SNR
    snr = rel_signal / rel_noise
    
    return snr, rel_signal, rel_noise


def compute_decision_acc(scores_large, scores_small, mask=None):
    """ Compute decision accuracy for a subset of instances. """
    if mask is None:
        mask = np.ones(scores_large.shape[1], dtype=bool) # use all scores!
    avg_scores_small = np.nanmean(scores_small[:, mask], axis=1)
    avg_scores_large = np.nanmean(scores_large[:, mask], axis=1)
    return decision_acc_fast(avg_scores_small, avg_scores_large)


def compute_pred_error(
        train_scores, 
        eval_scores, 
        train_bpb, 
        eval_bpb, 
        train_models, # ladder models
        eval_models, # ladder models
        eval_steps, # steps of eval model
        mask=None,
        ladder_config_path=f'{ROOT_DIR}/analysis/utils/ladder_config.json'
    ):
    """ Compute prediction error for a subset of instances. """
    if mask is None:
        mask = np.ones(train_scores.shape[1], dtype=bool) # use all scores!

    avg_scores_train = np.nanmean(train_scores[:, mask], axis=1)
    avg_score_eval = np.nanmean(eval_scores[:, mask], axis=1)
    avg_bpb_train = np.nanmean(train_bpb[:, mask], axis=1)
    avg_bpb_eval = np.nanmean(eval_bpb[:, mask], axis=1)

    # Create new rows for each model and its average score
    new_rows = []
    for model, bpb, score in zip(train_models, avg_bpb_train, avg_scores_train):
        new_row = pd.Series({
            'task': 'custom',
            'model': model,
            'logits_per_byte_corr': bpb,
            'primary_score': score,
            'step': 0
        })
        new_rows.append(new_row)

    # Workaround to support intermediate steps in eval model
    if len(eval_models) != len(eval_steps):
        eval_model_names = [eval_models[0] for _ in range(len(eval_steps))]

    for step, model, bpb, score in zip(eval_steps, eval_model_names, avg_bpb_eval, avg_score_eval):
        new_row = pd.Series({
            'task': 'custom',
            'model': model,
            'logits_per_byte_corr': bpb,
            'primary_score': score,
            'mix': None,
            'step': step
        })
        new_rows.append(new_row)

    df_custom = pd.DataFrame(new_rows)

    # Calculate resampled prediction error
    _, (_, _, stacked_y), (_, _, stacked_y_pred) = run_ladder(
        df_custom,
        'custom',
        train_models=train_models,
        eval_models=eval_models,
        config_path=ladder_config_path,
        run_step1=False, run_step2=False,
        return_reals=True,
    )

    stacked_y_pred = stacked_y_pred[0] # only 1 eval model

    rel_error = np.abs((stacked_y_pred - stacked_y) / stacked_y)
    abs_error = np.abs(stacked_y_pred - stacked_y)
    rel_error = np.mean(rel_error)

    # Calculate margin of error
    _, _, rel_errors_stacked = run_ladder(
        df_custom,
        'custom',
        train_models=train_models,
        eval_models=eval_models,
        config_path=ladder_config_path,
        run_step1=False, run_step2=False,
        last_n_method_train='final', last_n_method_eval='all', last_n=30
    )

    # Calculate margin-of-error using the set of ladder errors
    confidence_level = 0.95
    data = np.array(rel_errors_stacked)
    n = len(data)
    std_error = np.std(data, ddof=1) / np.sqrt(n)
    margin_of_error = std_error * stats.t.ppf((1 + confidence_level) / 2, n - 1)

    return rel_error, margin_of_error