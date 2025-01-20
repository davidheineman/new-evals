import sys
import os
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import boto3

# Add parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils import DATA_DIR

EXCLUDED_FILE_NAMES = [
    'requests.jsonl',
    'recorded-inputs.jsonl',
    'metrics-all.jsonl'
]


def download_file(s3_client, bucket_name, key, local_dir, excluded_file_names):
    local_path = os.path.join(local_dir, key)

    if any(f in key.split('/')[-1] for f in excluded_file_names):
        return # Skip download if there are any str matches with EXCLUDED_FILE_NAMES
    
    # if os.path.exists(local_path):
    #     return  # Skip download if the file already exists
    
    if os.path.exists(local_path):
        s3_head = s3_client.head_object(Bucket=bucket_name, Key=key)
        s3_file_size = s3_head['ContentLength']
        local_file_size = os.path.getsize(local_path)
        if s3_file_size == local_file_size:
            return  # Skip download if the file already exists and has the same size

    # Manual override for Ian's eval setup
    local_path.replace('all_olmes_rc_tasks/', '')
    local_path.replace('all_olmes_paraphrase_tasks/', '')

    # Manual override for Matt J's eval setup
    if 'olmo2_microanneals' in local_path:
        # 1) Seperate checkpoint: [MODEL_NAME]_step2693-hf => [MODEL_NAME]/step2693-hf
        local_path = re.sub(r'([a-zA-Z0-9_-]+)_step(\d+)-hf', r'\1/step\2-hf', local_path)

        # 2) Delete the OLMo 1 evals (__olmo1)
        if '__olmo1' in local_path: return

        # 3) Remove any subfolders after checkpoint: [MODEL_NAME]/step2693-hf/gsm8k__olmes/result.json => step2693-hf/result.json
        local_path = re.sub(r'(step\d+-hf)(/.*)?/([^/]+)$', r'\1/\3', local_path)

        # 4) Get human-readable name from model description
        #    OLMo-medium_peteish7-microanneals_peteish7-weka-microanneal-from928646_automathtext => automathtext
        def get_short(p):
            try:
                # Match "fromXXXXXX_" and extract the part after it
                return re.search(r'from\d+_(.*)', p).group(1)
            except Exception:
                return p
        
        def process_path(input_path):
            """ Perform a substitution on one folder in the path """
            parts = input_path.split('/')
            parts[-3] = get_short(parts[-3])
            final_path = '/'.join(parts)
            return final_path

        local_path = process_path(local_path)

    # Manual override for Luca's eval setup
    if 'olmo2_anneals' in local_path or 'olmo2_soups' in local_path:
        # 1) Remove any subfolders after checkpoint: [MODEL_NAME]/step2693-hf/gsm8k__olmes/result.json => step2693-hf/result.json
        local_path = re.sub(r'([^/]+)/[^/]+/([^/]+)$', r'\1/\2', local_path)

        # 2) Add fake model step: [MODEL_NAME]/result.json => [MODEL_NAME]/step0-hf/result.json
        dir_path, file_name = os.path.split(local_path)
        local_path = os.path.join(dir_path, "step0-hf", file_name)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket_name, key, local_path)


def fetch_page(page):
    return [obj['Key'] for obj in page.get('Contents', [])]


def fetch_keys_for_prefix(bucket_name, prefix):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    keys = []
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    for page in pages:
        keys.extend(fetch_page(page))
    return keys


def mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100, excluded_file_names=EXCLUDED_FILE_NAMES, s3_prefix_list=None):
    """ Recursively download an S3 folder to mirror remote """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    keys = []

    if s3_prefix_list is not None:
        with open(s3_prefix_list, 'r') as file:
            s3_prefixes = [line.strip() for line in file.readlines() if line.strip()]
    else:
        s3_prefixes = [s3_prefix]

    # with ProcessPoolExecutor() as executor:
    with ThreadPoolExecutor(max_workers=100) as executor:
        # TODO: Combine both if branches into one set of results
        if s3_prefix_list is not None:
            future_to_prefix = {}
            with tqdm(total=len(s3_prefixes), desc="Submitting S3 prefix tasks", unit="prefix") as submit_pbar:
                for prefix in s3_prefixes:
                    future = executor.submit(fetch_keys_for_prefix, bucket_name, prefix)
                    future_to_prefix[future] = prefix
                    submit_pbar.update(1)

            with tqdm(total=len(future_to_prefix), desc="Fetching keys from S3 prefixes", unit="prefix") as pbar:
                for future in as_completed(future_to_prefix):
                    try:
                        keys.extend(future.result())
                    except Exception as e:
                        print(f"Error processing prefix {future_to_prefix[future]}: {e}")
                    pbar.update(1)
        else:
            pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
            keys = []
            for result in tqdm(executor.map(fetch_page, pages), desc="Listing S3 entries"):
                keys.extend(result)

            for s3_prefix in tqdm(s3_prefixes, desc="Processing S3 prefixes", unit="prefix"):
                pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
                for result in executor.map(fetch_page, pages):
                    keys.extend(result)

    if max_threads > 1:
        # ProcessPoolExecutor seems not to work with AWS, so we use ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(download_file, s3_client, bucket_name, key, local_dir, excluded_file_names): key for key in keys}
            with tqdm(total=len(futures), desc="Syncing download folder from S3", unit="file") as pbar:
                for _ in as_completed(futures):
                    pbar.update(1)
    else:
        # Sequential implmentation
        for key in tqdm(keys, desc="Syncing download folder from S3", unit="file"):
            download_file(s3_client, bucket_name, key, local_dir, excluded_file_names)


def main():
    """
    Mirror AWS bucket to a local folder
    https://us-east-1.console.aws.amazon.com/s3/buckets/ai2-llm?prefix=eval-results/downstream/metaeval/OLMo-ladder/&region=us-east-1&bucketType=general
    """
    # #### GET REQUESTS FROM IAN'S PROJECT FOR BPB CALCULATION ####
    # bucket_name = 'ai2-llm'
    # s3_prefix = 'eval-results/downstream/eval-for-consistent-ranking-preemption-fixed/baseline-150M-5xC/step38157-unsharded-hf/'
    # folder_name = 'consistent_ranking'
    # local_dir = f'{DATA_DIR}/{folder_name}'
    # mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100, excluded_file_names=[])
    # #############################################################

    # bucket_name = 'ai2-llm'
    # s3_prefix = 'evaluation/microanneal-peteish-7b-postmortem/'
    # folder_name = 'olmo2_microanneals'

    # bucket_name = "ai2-lucas-archival"
    # s3_prefix = "evaluations/2024-09"
    # folder_name = 'olmo2_anneals'

    # bucket_name = "ai2-llm"
    # s3_prefix = "evaluation/peteish7-soup"
    # folder_name = 'olmo2_soups'

    bucket_name = 'ai2-llm'
    s3_prefix = 'eval-results/downstream/metaeval/'
    folder_name = 'aws'

    # bucket_name = 'ai2-llm'
    # s3_prefix = 'eval-results/downstream/eval-for-consistent-ranking-preemption-fixed/'
    # folder_name = 'consistent_ranking'

    # bucket_name = 'ai2-llm'
    # s3_prefix_list = 'analysis/data/cheap_decisions_paths.txt'
    # folder_name = 'consistent_ranking_final'

    local_dir = f'{DATA_DIR}/{folder_name}'

    mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100)
    # mirror_s3_to_local(bucket_name, None, local_dir, max_threads=100, s3_prefix_list=s3_prefix_list)

    # Launch preprocessing job!
    from preprocess import main
    main(folder_name, file_type='metrics')
    main(folder_name, file_type='predictions')

if __name__ == '__main__':
    main()
