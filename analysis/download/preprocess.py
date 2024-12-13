import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import psutil
from tqdm import tqdm

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
    "exact_match",
    "f1",
    "recall"
]

MODEL_OUTPUT_TO_KEEP = [
    "sum_logits",
    "logits_per_char",
    "logits_per_byte",
]

SIZE_PREFIXES = [
    f'-{size}-' for size in ['3B', '1B', '760M', '750M', '530M', '370M', '300M', '190M', '150M']
]

CHINHILLA_MULT = [
    '0.5xC', '1xC', '2xC', '5xC', '10xC', '15xC', '20xC'
]

def str_find(str_list, input_string):
    """ Get if a list of strings exists in a string. Return first match """
    hits = [item for item in str_list if item in input_string]
    if len(hits) == 0: 
        return None
    else:
        return hits[0]
    

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


def remove_prefix(input_string):
    return re.sub(r'^task-\d+-', '', input_string)


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
        if pred['native_id'] is not None:
            entry['native_id'] = str(pred['native_id'])
        elif pred['doc_id'] is not None:
            entry['native_id'] = str(pred['doc_id'])
        metrics = pred['metrics']
        model_output = pred['model_output']
        for col in METRICS_TO_KEEP:
            entry[col] = metrics[col] if col in metrics else None
        for col in MODEL_OUTPUT_TO_KEEP:
            entry[col] = [output[col] if col in output else None for output in model_output]
        processed += [entry]
    return processed


def process_chunk(chunk):
    return pd.DataFrame(chunk)


def get_available_cpus(threshold=80):
    cpu_usages = psutil.cpu_percent(percpu=True)
    available_cpus = [i for i, usage in enumerate(cpu_usages) if usage < threshold]
    return available_cpus


def load_df_parallel(data, usage_threshold=80):
    """ Load data as df w/ a CPU pool. Only use CPUs with usage below usage_threshold """
    available_cpus = get_available_cpus(threshold=usage_threshold)
    # num_partitions = len(available_cpus) * 100
    num_partitions = len(data) // 10_000

    print(f'Distributing {num_partitions} chunks across {len(available_cpus)} CPUs')
    
    if num_partitions == 0:
        raise RuntimeError("No CPUs are available below the usage threshold.")
    
    # Use numpy for efficient chunking
    chunk_size = len(data) // num_partitions
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_partitions)]

    print('Launching parallel processing...')
    
    with Pool(processes=len(available_cpus)) as pool:
        dataframes = list(tqdm(pool.imap(process_chunk, chunks), desc='Converting to Pandas dataframe', total=len(chunks)))

    # with Pool(processes=num_partitions) as pool:
    #     dataframes = list(tqdm(pool.map(process_chunk, chunks), desc='Converting to Pandas dataframe', total=len(chunks)))
    
    return pd.concat(dataframes, ignore_index=True)


def load_file(file_data):
    root, file = file_data
    file_path = os.path.join(root, file)

    # Manual override for Ian's folder setup
    root = root.replace('/all_olmes_paraphrase_tasks', '')
    root = root.replace('/all_olmes_rc_tasks', '')

    path_parts = os.path.normpath(root).split(os.sep)

    # Use last two folders as "path"/"step":
    # E.g., ../peteish-moreeval-1B-0.5xC/step8145-unsharded-hf
    if len(path_parts) >= 2:
        if 'step' in path_parts[-1]: # and ('-unsharded' in path_parts[-1] or '-hf' in path_parts[-1])
            # Local OLMo runs (anything that ends in "stepXXX-unsharded")
            model_name = path_parts[-2]
            step_str = path_parts[-1]
        else:
            # External models (e.g., llama)
            model_name = path_parts[-1]
            step_str = None
    else:
        raise RuntimeError(f'Could not process path: {path_parts}')

    # Get task name: "arc_challenge-metrics.json" => "arc_challenge"
    task = remove_prefix(file) # Remove "task-XXX" prefix: task-XXX-task_name.json => task_name.json
    task = task.rsplit('-', 1)[0]

    # Get step name: "stepXXX-unsharded" => "XXX"
    step = extract_step(step_str)

    # Get mix name
    mix_name = get_mix(model_name)

    # Get other metadata
    size = str_find(SIZE_PREFIXES, model_name)
    if size is not None: size = size.replace('-', '')
    token_ratio = str_find(CHINHILLA_MULT, model_name)

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
    for entry in preprocessed:
        entry.update({
            'model': model_name,
            'mix': mix_name,
            'step': step,
            'size': size,
            'token_ratio': token_ratio,
            'step_str': step_str,
            'task': task,
            's3_path': file_path,
        })

    return preprocessed


def process_files_chunk(files_chunk):
    results = []
    for file in files_chunk:
        results.extend(load_file(file))
    return results


def recursive_pull(data_dir):
    predictions_data = nested_defaultdict()
    metrics_data = nested_defaultdict()

    all_files = [
        (root, file)
        for root, _, files in os.walk(data_dir) if 'local_testing' not in root
        for file in files if file.endswith('predictions.jsonl') or file.endswith('metrics.jsonl')
    ]

    # all_files = all_files[:1_000]

    # with tqdm(total=len(all_files), desc=f"Recursively loading files in {data_dir}") as pbar:
    #     with ThreadPoolExecutor() as exec:
    #         for preprocessed in exec.map(load_file, all_files):
    #             predictions_df += preprocessed
    #             pbar.update(1)

    # all_preprocessed = []
    # with tqdm(total=len(all_files), desc=f"Recursively loading files in {data_dir}") as pbar:
    #     with ProcessPoolExecutor() as exec:
    #         for preprocessed in exec.map(load_file, all_files):
    #             all_preprocessed += preprocessed
    #             pbar.update(1)

    chunk_size = 100
    all_preprocessed = []
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    total_files = len(all_files)

    with tqdm(total=total_files, desc=f"Recursively loading files in {data_dir}") as pbar:
        with ProcessPoolExecutor(max_workers=len(get_available_cpus())) as executor:
            futures = {executor.submit(process_files_chunk, chunk): len(chunk) for chunk in file_chunks}
            for future in as_completed(futures):
                all_preprocessed.extend(future.result())
                pbar.update(futures[future])  # Update based on the chunk size
        pbar.close()

    return all_preprocessed, None, None
    # return all_preprocessed, predictions_data, metrics_data


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


def main(folder_name):
    data_dir = Path(DATA_DIR).resolve()
    data_dir.mkdir(exist_ok=True)

    aws_dir = data_dir / folder_name

    predictions_path = data_dir / f"all_{folder_name}_predictions.json"
    metrics_path     = data_dir / f"all_{folder_name}_metrics.json"
    parquet_path     = data_dir / f"all_{folder_name}_predictions.parquet"

    predictions_df, predictions, metrics = recursive_pull(aws_dir)

    # Save predictions to parquet
    import time
    start_time = time.time()

    # df = pd.DataFrame(predictions_df, columns=predictions_df[0])
    # df = pd.json_normalize(predictions_df)

    # import pyarrow as pa # conversion for 1000 preds: pyarrow=5.6s, pandas=14s
    # df = pa.Table.from_pylist(predictions_df).to_pandas()
    
    df = load_df_parallel(predictions_df) # for 6700 preds: 300s (5 min)

    print(f"Converted to pandas in: {time.time() - start_time:.4f} seconds")
    # verify_df(df)

    # Reset the df index (for faster indexing)
    df.set_index(['task', 'model', 'step', 'mix'], inplace=True)
    
    # Save to parquet
    df.to_parquet(parquet_path, index=True)
    print(f"Predictions saved to {parquet_path} ({fsize(parquet_path):.2f} GB)")

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


if __name__ == '__main__': 
    folder_name = "aws" # 30min
    # folder_name = "consistent_ranking" # 3hr
    main(folder_name)