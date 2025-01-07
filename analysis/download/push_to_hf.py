from huggingface_hub import HfApi, login, HfHubHTTPError
import os

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