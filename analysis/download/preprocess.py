import sys
import os
import re
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils import DATA_DIR

# Metrics to use when converting to table:
METRICS_TO_KEEP = [
    "acc_raw",
    "acc_per_char",
    "predicted_index_per_char",
    "predicted_index_raw",
    "correct_choice",
]

MODEL_OUTPUT_TO_KEEP = [
    "sum_logits",
    "logits_per_char",
]

SIZE_PREFIXES = [
    f'-{size}-' for size in ['3B', '1B', '760M', '370M', '190M']
]

def get_mix(model_name):
    """ falcon_and_cc_eli5_oh_top10p-3B-5xC => falcon_and_cc_eli5_oh_top10p """
    mix = None
    for prefix in SIZE_PREFIXES:
        if prefix in model_name:
            mix = model_name.split(prefix)[0]
            
            # manual overrides for model ladder
            mix = mix.replace('-rerun', '')
            mix = mix.replace('-moreeval', '-ladder')
    return mix


def extract_step(input_string):
    if input_string is None: return None
    match = re.search(r'step(\d+)(?:-[a-zA-Z0-9]+)?', input_string)
    return int(match.group(1)) if match else None


def nested_defaultdict():
    return defaultdict(nested_defaultdict)


def fsize(file_path):
    return os.path.getsize(file_path) / (1024 ** 3)


def process_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def process_predictions(file_path):
    """ Process a predictions.jsonl to a list """
    predictions = process_jsonl(file_path)

    processed = []
    for pred in predictions:
        entry = {}
        entry['native_id'] = str(pred['native_id'])
        metrics = pred['metrics']
        model_output = pred['model_output']
        for col in METRICS_TO_KEEP:
            entry[col] = metrics[col] if col in metrics else None
        for col in MODEL_OUTPUT_TO_KEEP:
            entry[col] = [output[col] if col in output else None for output in model_output]
        processed += [entry]
    return processed


def load_file(file_data):
    root, file = file_data
    file_path = os.path.join(root, file)
    path_parts = os.path.normpath(root).split(os.sep)

    # Use last two folders as "path"/"step":
    # E.g., ../peteish-moreeval-1B-0.5xC/step8145-unsharded-hf
    if len(path_parts) >= 2:
        if 'OLMo' in root:
            # Local OLMo runs
            model_name = path_parts[-2]
            step_str = path_parts[-1]
        else:
            # External models (e.g., llama)
            model_name = path_parts[-1]
            step_str = None
    else:
        raise RuntimeError(f'Could not process path: {path_parts}')

    # Get task name: "arc_challenge-metrics.json" => "arc_challenge"
    task = file.rsplit('-', 1)[0]

    # Get step name: "stepXXX-unsharded" => "XXX"
    step = extract_step(step_str)

    # Get mix name
    mix_name = get_mix(model_name)

    # Load predictions
    data = process_jsonl(file_path)
    preprocessed = process_predictions(file_path)

    # # Add metadata to output json
    # if file.endswith('predictions.jsonl'):
    #     predictions_data[model_name][step][task] = data
    # elif file.endswith('metrics.jsonl'):
    #     metrics_data[model_name][step][task] = data
    # else:
    #     raise ValueError(file)

    # Add metadata to parquet file
    for i, entry in enumerate(preprocessed):
        entry.update({
            'model': model_name,
            'mix': mix_name,
            'step': step,
            'step_str': step_str,
            'task': task,
            's3_path': file_path,
        })

        # Convert all entries to str
        # preprocessed[i] = {k: str(v) if k is not None else "" for k, v in entry.items()}

    return preprocessed


def recursive_pull(data_dir):
    predictions_data = nested_defaultdict()
    metrics_data = nested_defaultdict()

    predictions_df = []

    all_files = [
        (root, file)
        for root, _, files in os.walk(data_dir) if 'local_testing' not in root
        for file in files if file.endswith('predictions.jsonl') or file.endswith('metrics.jsonl')
    ]

    # all_files = all_files[:1000]

    with tqdm(total=len(all_files), desc=f"Recursively loading files in {data_dir}") as pbar:
        with ThreadPoolExecutor() as exec:
            for preprocessed in exec.map(load_file, all_files):
                predictions_df += preprocessed
                pbar.update(1)

    return predictions_df, None, None
    # return predictions_df, predictions_data, metrics_data


def verify_df(df):
    # Identify missing models/tasks
    unique_models = df['model'].unique()
    unique_tasks  = df['task'].unique()
    missing_entries = []
    for model in unique_models:
        for task in unique_tasks:
            task_rows = df[(df['model'] == model) & (df['task'] == task)]
            if task_rows.empty:
                missing_entries.append((model, task))

    if missing_entries:
        print("Missing tasks for models:")
        for model, task in missing_entries:
            print(f"  - Model: {model}, Task: {task}")


def main():
    data_dir = Path(DATA_DIR).resolve()
    data_dir.mkdir(exist_ok=True)

    aws_dir = data_dir / "aws"

    predictions_path = data_dir / "all_aws_predictions.json"
    metrics_path     = data_dir / "all_aws_metrics.json"
    parquet_path     = data_dir / "all_aws_predictions.parquet"

    predictions_df, predictions, metrics = recursive_pull(aws_dir)

    # Save predictions to parquet
    import time
    start_time = time.time()
    
    df = pd.DataFrame(predictions_df, columns=predictions_df[0])
    # df = pd.json_normalize(predictions_df)
    print(f"Converted to pandas in: {time.time() - start_time:.4f} seconds")

    # import pyarrow as pa
    # table = pa.Table.from_pylist(predictions_df)
    # df = table.to_pandas()
    # print(f"Converted to pandas in: {time.time() - start_time:.4f} seconds") # conversion for 1000 preds: pyarrow=5.6s, pandas=14s

    verify_df(df)

    # Reset the df index (for faster indexing)
    df.set_index(['task', 'model', 'step', 'mix'], inplace=True)

    print(f"Predictions saved to {parquet_path} ({fsize(parquet_path):.2f} GB)")
    
    # Save to parquet
    df.to_parquet(parquet_path, index=True)

    # Write prediction files
    print('Saving files...')
    if predictions is not None:
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f)
        print(f"Predictions saved to {predictions_path} ({fsize(predictions_path):.2f} GB)")

    if metrics is not None:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        print(f"Metrics saved to {metrics_path} ({fsize(metrics_path):.2f} GB)")
    print('Done!')


if __name__ == '__main__': main()