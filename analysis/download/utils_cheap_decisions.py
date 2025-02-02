import re
import json
import numpy as np
import pandas as pd

import sys

sys.path.append('/root/ai2/metaeval/olmo-repos/oe-eval-internal')
from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY


generate_primary_metrics = [
    "em",
    "f1",
    "exact_match",
    "pass_at_1",
    # "prompt_level_loose_acc",
    # "maj_at_1"
]


def get_tasks(file_path):
    with open(file_path, "r") as f:
        tasks = [t.split(":")[0] for t in f.readlines()]
    return tasks


def safe_eval(x):
    """Utility for reading 'metrics' col which is a dict in DataFrame"""
    try:
        result = eval(x)
        # Traverse dict to replace NaN values
        if isinstance(result, dict):
            result = {
                key: (None if (isinstance(value, float) and np.isnan(value)) else value)
                for key, value in result.items()
            }
        return result
    except:
        # If fails, return the original string or handle it as needed
        return x


def log_sum_exp(log_probs):
    """Numerical stable way to compute log(sum(exp(log_probs)))"""
    max_log_prob = np.max(log_probs)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))


def check_finite_and_nan(value, name):
    assert np.isfinite(value), f"{name}: {value} is inf or -inf"
    assert not np.isnan(value), f"{name}: {value} is NaN"


def process_predictions_cheap_decisions(prediction):
    metrics = prediction["metrics"]
    model_outputs = prediction["model_output"]

    # 1. RC tasks
    if all(key in metrics for key in ["acc_raw", "acc_per_char"]):
        correct_idx = metrics["correct_choice"]
        correct_output = model_outputs[correct_idx]

        # Compute correct seq
        correct_logit = correct_output["sum_logits"]
        correct_logit_per_token = correct_output["logits_per_token"]
        correct_logit_per_char = correct_output["logits_per_char"]
        correct_logit_per_byte = correct_output["logits_per_byte"]

        # Compute margin
        correct_prob = np.exp(correct_logit)
        correct_prob_per_token = np.exp(correct_logit_per_token)
        correct_prob_per_char = np.exp(correct_logit_per_char)
        correct_prob_per_byte = np.exp(correct_logit_per_byte)
        incorrect_probs = [
            np.exp(out["sum_logits"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]
        incorrect_probs_per_token = [
            np.exp(out["logits_per_token"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]
        incorrect_probs_per_char = [
            np.exp(out["logits_per_char"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]
        incorrect_probs_per_byte = [
            np.exp(out["logits_per_byte"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]

        # Compute uncond
        if all("sum_logits_uncond" in option for option in model_outputs):
            uncond_logits = np.array(
                [option["sum_logits_uncond"] for option in model_outputs]
            )
            uncond_correct_logit = uncond_logits[correct_idx]
            uncond_correct_prob = np.exp(uncond_correct_logit)
            uncond_correct_prob_per_token = np.exp(
                uncond_correct_logit / correct_output["num_tokens"]
            )
            uncond_correct_prob_per_char = np.exp(
                uncond_correct_logit / correct_output["num_chars"]
            )
            uncond_correct_prob_per_byte = np.exp(
                uncond_correct_logit / correct_output["num_bytes"]
            )
            # sum
            uncond_total_logit = log_sum_exp(uncond_logits)
            uncond_total_prob = np.exp(uncond_total_logit)
        else:
            uncond_correct_prob = None
            uncond_total_prob = None
            uncond_correct_prob_per_token = None
            uncond_correct_prob_per_char = None
            uncond_correct_prob_per_byte = None

        if incorrect_probs and not np.isnan(correct_prob - np.max(incorrect_probs)):
            margin = correct_prob - np.max(incorrect_probs)
            margin_per_token = correct_prob_per_token - np.max(
                incorrect_probs_per_token
            )
            margin_per_char = correct_prob_per_char - np.max(incorrect_probs_per_char)
            margin_per_byte = correct_prob_per_byte - np.max(incorrect_probs_per_byte)
            assert -1 <= margin <= 1, f"Margin out of bounds: {margin}"
            assert (
                -1 <= margin_per_token <= 1
            ), f"Margin per token out of bounds: {margin_per_token}"
            assert (
                -1 <= margin_per_char <= 1
            ), f"Margin per char out of bounds: {margin_per_char}"
        else:
            margin = None
            margin_per_token = None
            margin_per_char = None
            margin_per_byte = None

        # Compute total_logit and total_prob using log-sum-exp trick
        logits = np.array([option["sum_logits"] for option in model_outputs])
        total_logit = log_sum_exp(logits)
        total_prob = np.exp(total_logit)

        logits_per_token = np.array(
            [option["logits_per_token"] for option in model_outputs]
        )
        total_logit_per_token = log_sum_exp(logits_per_token)
        total_prob_per_token = np.exp(total_logit_per_token)

        logits_per_char = np.array(
            [option["logits_per_char"] for option in model_outputs]
        )
        total_logit_per_char = log_sum_exp(logits_per_char)
        total_prob_per_char = np.exp(total_logit_per_char)

        logits_per_byte = np.array(
            [option["logits_per_char"] for option in model_outputs]
        )
        total_logit_per_byte = log_sum_exp(logits_per_byte)
        total_prob_per_byte = np.exp(total_logit_per_byte)

        norm_correct_prob = np.exp(correct_logit - total_logit)
        norm_correct_prob_per_token = np.exp(
            correct_logit_per_token - total_logit_per_token
        )
        norm_correct_prob_per_char = np.exp(
            correct_logit_per_char - total_logit_per_char
        )
        norm_correct_prob_per_byte = np.exp(
            correct_logit_per_byte - total_logit_per_byte
        )

        if not np.isnan(total_prob):
            assert (
                0 <= total_prob <= len(model_outputs)
            ), f"Total probability out of bounds ({len(model_outputs)}): {total_prob}"
            assert (
                0 <= norm_correct_prob <= 1
            ), f"Normalized correct probability out of bounds: {norm_correct_prob}"
            assert (
                0 <= norm_correct_prob_per_token <= 1
            ), f"Normalized correct probability per token out of bounds: {norm_correct_prob_per_token}"
            assert (
                0 <= norm_correct_prob_per_char <= 1
            ), f"Normalized correct probability per char out of bounds: {norm_correct_prob_per_char}"

            # Checks for inf, -inf, and NaNs
            check_finite_and_nan(total_prob, "total_prob")
            check_finite_and_nan(total_prob_per_token, "total_prob_per_token")
            check_finite_and_nan(total_prob_per_char, "total_prob_per_char")
            check_finite_and_nan(norm_correct_prob, "norm_correct_prob")
            check_finite_and_nan(
                norm_correct_prob_per_token, "norm_correct_prob_per_token"
            )
            check_finite_and_nan(
                norm_correct_prob_per_char, "norm_correct_prob_per_char"
            )

        row_dict = {
            "correct_logit": correct_logit,
            "correct_logit_per_token": correct_logit_per_token,
            "correct_logit_per_char": correct_logit_per_char,
            "correct_logit_per_byte": correct_logit_per_byte,
            "correct_prob": correct_prob,
            "correct_prob_per_token": correct_prob_per_token,
            "correct_prob_per_char": correct_prob_per_char,
            "correct_prob_per_byte": correct_prob_per_byte,
            "margin": margin,
            "margin_per_token": margin_per_token,
            "margin_per_char": margin_per_char,
            "margin_per_byte": margin_per_byte,
            "total_prob": total_prob,
            "total_prob_per_token": total_prob_per_token,
            "total_prob_per_char": total_prob_per_char,
            "total_prob_per_byte": total_prob_per_byte,
            "uncond_correct_prob": uncond_correct_prob,
            "uncond_correct_prob_per_token": uncond_correct_prob_per_token,
            "uncond_correct_prob_per_char": uncond_correct_prob_per_char,
            "uncond_correct_prob_per_byte": uncond_correct_prob_per_byte,
            "uncond_total_prob": uncond_total_prob,
            "norm_correct_prob": norm_correct_prob,
            "norm_correct_prob_per_token": norm_correct_prob_per_token,
            "norm_correct_prob_per_char": norm_correct_prob_per_char,
            "norm_correct_prob_per_byte": norm_correct_prob_per_byte,
        }
        metrics.update(row_dict)

    # 2. Generation tasks
    elif any(key in metrics for key in generate_primary_metrics):

        # Case: Codex - Check if model_outputs has 2 elements
        if len(model_outputs) == 2:
            model_outputs = model_outputs[:1]  # pass_at_1

        if len(model_outputs) > 1:
            raise ValueError(
                "Assume generation tasks only have one output (greedy): ",
                len(model_outputs),
            )

        logits = model_outputs[0]["sum_logits"]
        num_tokens = (
            model_outputs[0]["num_tokens"] if model_outputs[0]["num_tokens"] > 0 else 1
        )
        num_chars = (
            len(model_outputs[0]["continuation"])
            if model_outputs[0]["continuation"]
            else 1
        )

        logit_per_token = logits / num_tokens
        logit_per_char = logits / num_chars

        # Case: sum_scores only available in latest version
        if "sum_scores" in model_outputs[0]:
            scores = model_outputs[0]["sum_scores"]
            score_per_token = scores / num_tokens
            score_per_char = scores / num_chars
            check_finite_and_nan(logits, "logit")
            check_finite_and_nan(logit_per_token, "logit_per_token")
            check_finite_and_nan(logit_per_char, "logit_per_char")
        else:
            scores = None
            score_per_token = None
            score_per_char = None

        row_dict = {
            "logit": logits,
            "logit_per_token": logit_per_token,
            "logit_per_char": logit_per_char,
            "score": scores,
            "score_per_token": score_per_token,
            "score_per_char": score_per_char,
        }
        metrics.update(row_dict)

    return metrics


def compute_metrics_from_file(predictions_content: str, task=None) -> list:
    """
    Read predictions from a JSONL file and compute various metrics for each task sample.

    Args:
        predictions_content (str): Content of the predictions file in JSONL format.

    Returns:
        list: A list of dictionaries containing computed metrics for each task sample.
    """
    predictions = [json.loads(line) for line in predictions_content.splitlines()]
    rows = []

    for prediction in predictions:
        metrics = process_predictions_cheap_decisions(prediction, task=task)
        rows.append(metrics)

    return rows


def parse_train_name(path):
    """
    Parse the S3 path to extract the group, model, chinchilla, task, and step.
    Example input path structure: "checkpoints/benb/olmo-150M-no_math_no_code-1xC/step500/mmlu/predictions.jsonl"
    """
    parts = path.split("/")[7:]
    assert re.match(
        r".*-\d+xC(-\d+)?$", parts[3]
    ), f"Invalid model name format: {parts[3]}"

    if re.match(r".*-\d+xC-\d+$", parts[3]):
        group_model_chinchilla = parts[3].rsplit("-", 3)
        seed = group_model_chinchilla[3]
        assert re.match(
            r"\d", seed
        ), f"Invalid model name parsing: {parts[3]} -> {group_model_chinchilla}"
        seed = int(seed)
    elif re.match(r".*-\d+xC$", parts[3]):
        group_model_chinchilla = parts[3].rsplit("-", 2)
        seed = None
    else:
        raise ValueError(f"Invalid model name format: {parts}")

    group = group_model_chinchilla[0]
    model = group_model_chinchilla[1]
    assert re.match(r"\d+[M|B]", model), f"Invalid model size parsing: {model}"
    chinchilla = group_model_chinchilla[2]
    assert re.match(r"\d+xC", chinchilla), f"Invalid chinchilla parsing: {chinchilla}"
    step = int(re.search(r"step(\d+)", parts[4]).group(1))
    if "all_olmes" in path:
        task = None
        if "_rc_tasks" in path:
            metrics_match = re.search(r"task-\d+-(.*?)-metrics\.json", parts[6])
            predictions_match = re.search(r"task-\d+-(.*?)-predictions\.jsonl", parts[6])
            if metrics_match:
                task = metrics_match.group(1)
            elif predictions_match:
                task = predictions_match.group(1)
    else:
        task = parts[5]
    return group, model, chinchilla, task, step, seed

import numpy as np
from typing import Dict, Any

def debug_aggregation(metrics_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Debug metrics aggregation with detailed error reporting.
    
    Args:
        metrics_dict: Dictionary of metrics to aggregate
    Returns:
        Dictionary of mean values for each metric
    """
    mean_metrics = {}
    
    for key, values in metrics_dict.items():
        try:
            # Try to convert to numpy array and get mean
            arr = np.array(values)
            try:
                mean_metrics[key] = np.mean(arr)
            except Exception as e:
                print(f'Couldnt divide on {key}: {arr}')
                mean_metrics[key] = 0
            
        except ValueError as e:
            pass
            
    return mean_metrics


def process_prediction_path(path, rows_list):
    group, model, chinchilla, task, step, seed = parse_train_name(path)

    # # Extract metrics
    # # rows_list = compute_metrics_from_file(predictions, task)
    # rows_list = []
    # for prediction in predictions:
    #     metrics = compute_metrics_from_file(prediction, task=task)
    #     rows_list.append(metrics)
    # if rows_list is None:
    #     print(f"Skipping results for: {task}")
    #     return None

    if task not in TASK_REGISTRY:
        print(f'Could not find "{task}" on path {path}!')

    task_config = TASK_REGISTRY[task].__dict__.get('TASK_CONFIG_DEFAULTS', {})
    
    # Get primary_metric in this order
    primary_metric = task_config.get("primary_metric", None)
    possible_metrics = [
        "primary_metric",
        "acc_raw",
        "exact_match",
        "f1",
        "mc1",
        "pass_at_1",
        "prompt_level_loose_acc",
        "maj_at_1",
    ]

    aggregated_metrics = {}
    for mrow in rows_list:
        if "em" in mrow:
            mrow["exact_match"] = mrow.pop("em")
        if primary_metric is None:
            for metric in possible_metrics:
                if metric in mrow:
                    # Set name forprimary_metric
                    primary_metric = metric
                    break
        if primary_metric is None:
            print(f"Skipping task {task} due to missing primary metric: {mrow}")
            continue

        mrow["primary_metric"] = mrow[primary_metric]
        mrow["acc_raw"] = mrow["acc_raw"]
        mrow["acc_per_char"] = mrow["acc_per_char"]
        mrow["acc_per_token"] = mrow["acc_per_token"]
        mrow["acc_uncond"] = mrow["acc_uncond"]
        for key, value in mrow.items():
            if value is None or isinstance(value, str):
                continue
            if key in aggregated_metrics:
                aggregated_metrics[key].append(value)
            else:
                aggregated_metrics[key] = [value]

    # mean_metrics = {k: np.mean(v) for k, v in aggregated_metrics.items()}
    mean_metrics = debug_aggregation(aggregated_metrics)

    row = {
        "group": group,
        "model": model,
        "task": task,
        "chinchilla": chinchilla,
        "step": step,
        "seed": seed,
        "metrics": mean_metrics,
    }
    return row


def define_compute_proportions(steps=5):
    return [i / steps for i in range(1, steps + 1)]


def unpack_dict_column(df, col_name):
    """
    Unpack a dictionary column in a DataFrame using json_normalize.
    Return a new DataFrame with the unpacked columns joined.
    """
    temp = pd.json_normalize(df[col_name], max_level=1)
    temp = temp.reset_index(drop=True)
    df = df.reset_index(drop=True).drop(columns=[col_name]).join(temp)
    # print(f"Columns from unpacking: {df.columns}")
    return df


def format_tokens(tokens: int):
    if tokens >= 1_000_000_000:  # Check for billions
        return f"{tokens / 1_000_000_000:.1f}B"
    elif tokens >= 1_000_000:  # Check for millions
        return f"{tokens / 1_000_000:.1f}M"
    else:
        return str(tokens)


def find_common_checkpoints(metric_values1: np.array, metric_values2: np.array):
    """Find all common checkpoints between two metric arrays."""
    # Identify non-NaN indices for both arrays
    valid_indices1 = ~np.isnan(metric_values1)
    valid_indices2 = ~np.isnan(metric_values2)
    common_indices = np.where(valid_indices1 & valid_indices2)[0]
    if not len(common_indices):
        raise ValueError("No common checkpoints found between the two mixes.")
    return common_indices


def clean_nans(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    mask = np.isfinite(arr1) & np.isfinite(arr2)
    # Check if any NaNs were removed by comparing the original and filtered lengths
    changed = not np.all(mask)
    # Apply the mask to filter out NaN indices
    filtered_arr1 = arr1[mask].tolist()
    filtered_arr2 = arr2[mask].tolist()
    return filtered_arr1, filtered_arr2, changed
