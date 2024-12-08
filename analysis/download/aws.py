import sys
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def download_file(s3_client, bucket_name, key, local_dir):
    local_path = os.path.join(local_dir, key)
    if any(f in key.split('/')[-1] for f in EXCLUDED_FILE_NAMES):
        return # Skip download if there are any str matches with EXCLUDED_FILE_NAMES
    if os.path.exists(local_path):
        return  # Skip download if the file already exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Manual override for Ian's eval setup
    local_path.replace('all_olmes_rc_tasks/', '')
    local_path.replace('all_olmes_paraphrase_tasks/', '')

    s3_client.download_file(bucket_name, key, local_path)


def fetch_page(page):
    return [obj['Key'] for obj in page.get('Contents', [])]


def mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=10):
    """ Recursively download an S3 folder to mirror remote """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    keys = []

    with ThreadPoolExecutor() as executor:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
        keys = []
        for result in tqdm(executor.map(fetch_page, pages), desc="Listing S3 entries"):
            keys.extend(result)

    with ThreadPoolExecutor(max_threads) as executor:
        futures = {executor.submit(download_file, s3_client, bucket_name, key, local_dir): key for key in keys}
        with tqdm(total=len(futures), desc="Syncing download folder from S3", unit="file") as pbar:
            for _ in as_completed(futures):
                pbar.update(1)


def main():
    """
    Mirror AWS bucket to a local folder
    https://us-east-1.console.aws.amazon.com/s3/buckets/ai2-llm?prefix=eval-results/downstream/metaeval/OLMo-ladder/&region=us-east-1&bucketType=general
    """
    # bucket_name = 'ai2-llm'
    # s3_prefix = 'eval-results/downstream/metaeval/'
    # local_dir = f'{DATA_DIR}/aws'

    bucket_name = 'ai2-llm'
    s3_prefix = 'eval-results/downstream/eval-for-consistent-ranking-preemption-fixed/'
    local_dir = f'{DATA_DIR}/consistent_ranking'

    mirror_s3_to_local(bucket_name, s3_prefix, local_dir)

if __name__ == '__main__':
    main()
