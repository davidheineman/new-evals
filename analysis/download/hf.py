from huggingface_hub import HfApi, login, hf_hub_download
import os

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DATA_DIR


def push_parquet_to_hf(parquet_file_path, hf_dataset_name, private=True, overwrite=False):
    login(new_session=False)
    api = HfApi()
    
    # Check if the repo exists; create it if not
    try:
        api.repo_info(repo_id=hf_dataset_name, repo_type="dataset")
    except Exception as e:
        api.create_repo(repo_id=hf_dataset_name, private=private, repo_type="dataset", exist_ok=True)

    # Determine the target file path in the repository
    file_name = os.path.basename(parquet_file_path)
    path_in_repo = file_name

    # Check if the file exists in the repository
    repo_files = api.list_repo_files(repo_id=hf_dataset_name, repo_type="dataset")
    if path_in_repo in repo_files:
        if not overwrite:
            print(f"File '{path_in_repo}' already exists in '{hf_dataset_name}'. Skipping upload.")
            return
        print(f"File '{path_in_repo}' exists and will be overwritten.")

    # Upload the file to the repository
    api.upload_file(
        path_or_fileobj=parquet_file_path,
        path_in_repo=path_in_repo,
        repo_id=hf_dataset_name,
        repo_type="dataset"
    )
    print(f"File '{path_in_repo}' uploaded to '{hf_dataset_name}'.")


def download_parquet_from_hf(hf_dataset_name, file_name, local_path):
    print(f'Downloading {file_name} -> {local_path}')
    file_path = hf_hub_download(
        repo_id=hf_dataset_name,
        filename=file_name,
        repo_type="dataset",
        local_dir=local_path
    )
    return file_path


def pull_predictions_from_hf(repo_id, name):
    file_name = f'all_{name}_predictions.parquet'
    local_path = DATA_DIR
    download_parquet_from_hf(repo_id, file_name, local_path)