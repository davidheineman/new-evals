import sys, os
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, login, hf_hub_download
from utils import DATA_DIR


def convert_csv_to_parquet(csv_file_path):
    parquet_file_path = csv_file_path.replace(".csv", ".parquet")
    print(f"Converting '{csv_file_path}' -> '{parquet_file_path}'")
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    df.to_parquet(parquet_file_path, index=False)
    return parquet_file_path


def push_parquet_to_hf(parquet_file_path, hf_dataset_name, split_name='main', private=True, overwrite=False):
    if parquet_file_path.endswith(".csv"):
        parquet_file_path = convert_csv_to_parquet(parquet_file_path)
        parquet_file_path = parquet_file_path.replace('.csv', '.parquet')

    file_name = os.path.basename(parquet_file_path)

    login(new_session=False)
    api = HfApi()
    
    # Check if the repo exists; create it if not
    try:
        api.repo_info(repo_id=hf_dataset_name, repo_type="dataset")
    except Exception as e:
        api.create_repo(repo_id=hf_dataset_name, private=private, repo_type="dataset", exist_ok=True)

    # Determine the target file path in the repository
    
    path_in_repo = os.path.join('data', f'{split_name}-00000-of-00001.parquet') # https://huggingface.co/docs/hub/en/datasets-file-names-and-splits

    # Check if the file exists in the repository
    repo_files = api.list_repo_files(repo_id=hf_dataset_name, repo_type="dataset")
    if path_in_repo in repo_files:
        if not overwrite:
            print(f"File '{path_in_repo}' already exists in '{hf_dataset_name}'. Skipping upload.")
            return
        print(f"File '{path_in_repo}' exists and will be overwritten.")

    print(f"Uploading '{parquet_file_path}' -> '{path_in_repo}' to hf dataset '{hf_dataset_name}'")

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


def pull_predictions_from_hf(repo_id, split_name, local_path=DATA_DIR):
    file_name = f'data/{split_name}-00000-of-00001.parquet'
    download_parquet_from_hf(repo_id, file_name, local_path)
    local_file_name = os.path.join(local_path, file_name)
    return local_file_name


def main():
    # push_parquet_to_hf(
    #     parquet_file_path='analysis/data/all_olmo2_soups_predictions.parquet',
    #     hf_dataset_name='allenai/olmo2-soups-evals',
    #     overwrite=True
    # )

    # push_parquet_to_hf(
    #     parquet_file_path='analysis/data/all_olmo2_anneals_predictions.parquet',
    #     hf_dataset_name='allenai/olmo2-anneals-evals',
    #     overwrite=True
    # )

    # push_parquet_to_hf(
    #     parquet_file_path='analysis/data/all_olmo2_microanneals_predictions.parquet',
    #     hf_dataset_name='allenai/olmo2-microanneals-evals',
    #     overwrite=True
    # )

    push_parquet_to_hf(
        parquet_file_path='analysis/data/aws_metrics.csv',
        hf_dataset_name='allenai/ladder-evals',
        split_name='benchmarks',
        overwrite=True
    )
    push_parquet_to_hf(
        parquet_file_path='analysis/data/all_aws_predictions.parquet',
        hf_dataset_name='allenai/ladder-evals',
        split_name='instances',
        overwrite=True
    )

    # push_parquet_to_hf(
    #     parquet_file_path='analysis/data/all_consistent_ranking_predictions.parquet',
    #     hf_dataset_name='davidheineman/consistent-ranking-evals',
    #     overwrite=True,
    #     private=False,
    # )

    # push_parquet_to_hf(
    #     parquet_file_path='analysis/data/all_consistent_ranking_predictions.parquet',
    #     hf_dataset_name='allenai/consistent-ranking-evals',
    #     overwrite=True
    # )

    # push_parquet_to_hf(
    #     parquet_file_path='analysis/data/all_consistent_ranking_final_predictions.parquet',
    #     hf_dataset_name='davidheineman/consistent-ranking-evals',
    #     overwrite=True,
    #     private=False,
    # )


if __name__ == '__main__': main()