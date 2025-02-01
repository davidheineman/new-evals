from huggingface_hub import HfApi
from tqdm import tqdm
import numpy as np
import os

WEKA_STEPS = [0, 1000, 10000, 1100, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 19850, 2000, 20000, 21000, 22000, 23000, 239000, 24000, 25000, 26000, 27000, 28000, 29000, 3000, 30000, 31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 4000, 40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 476848, 48000, 49000, 5000, 50000, 51000, 52000, 53000, 54000, 54650, 55000, 56000, 57000, 58000, 59000, 6000, 60000, 61000, 62000, 63000, 64000, 65000, 66000, 67000, 68000, 69000, 7000, 70000, 71000, 72000, 73000, 74000, 75000, 76000, 77000, 78000, 79000, 79700, 8000, 80000, 80550, 81000, 82000, 82600, 83000, 84000, 85000, 85050, 9000]
WEKA_CHECKPOINTS = [f'/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step{step}' for step in WEKA_STEPS]

S3_STEPS = [0, 1000, 10000, 100000, 1100, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 19850, 2000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 3000, 30000, 31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 4000, 40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 5000, 50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 6000, 60000, 61000, 62000, 63000, 64000, 65000, 66000, 67000, 68000, 69000, 7000, 70000, 71000, 72000, 73000, 74000, 75000, 76000, 77000, 78000, 79000, 8000, 80000, 81000, 82000, 83000, 84000, 85000, 86000, 87000, 88000, 89000, 9000, 90000, 91000, 92000, 93000, 93400, 94000, 95000, 96000, 97000, 98000, 99000]
S3_CHECKPOINTS = [f's3://ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step{step}' for step in S3_STEPS]

def get_olmo_branches(num_checkpoints):
    api = HfApi()
    
    # Get all branches for the repo
    branches = api.list_repo_refs("allenai/OLMo-2-1124-13B")
    branches = [branch.name for branch in branches.branches]
    
    # Filter and sort branches that match the pattern
    steps = []
    stage1_branches = []
    for branch in branches + WEKA_CHECKPOINTS + S3_CHECKPOINTS:
        branch = str(branch)
        if branch.startswith('stage1-step'):
            step = int(branch.split('step')[1].split('-')[0])
            if step in steps: 
                continue
            stage1_branches.append(branch)
            steps.append(step)
        elif branch.startswith('/oe-training-default'):
            step = int(branch.split('step')[1])
            if step in steps: 
                continue
            stage1_branches.append(branch)
            steps.append(step)
        elif branch.startswith('s3://'):
            step = int(branch.split('step')[1])
            if step in steps: 
                continue
            stage1_branches.append(branch)
            steps.append(step)
    
    # Sort by step number
    steps = np.array(steps)
    stage1_branches = np.array(stage1_branches)
    sort_idx = np.argsort(steps)
    steps = steps[sort_idx]
    stage1_branches = stage1_branches[sort_idx]

    print(f'Seeing {len(stage1_branches)} branches.')
    
    # Get evenly spaced indices
    total_checkpoints = len(steps)
    if total_checkpoints <= num_checkpoints:
        return stage1_branches.tolist()
        
    ideal_steps = np.linspace(steps[0], steps[-1], num_checkpoints)
    
    selected_indices = []
    for target in ideal_steps:
        idx = np.abs(steps - target).argmin()
        if idx not in selected_indices:  # Avoid duplicates
            selected_indices.append(idx)
    
    # Get final evenly-spaced branches
    selected_branches = stage1_branches[selected_indices].tolist()
    return selected_branches

if __name__ == '__main__':
    """
    Get OLMo 2 11/24 13B ckpts
    - steps 0 to 85050 are on weka -- /oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr
    - steps 85050 to 102500 are on s3? -- https://us-east-1.console.aws.amazon.com/s3/buckets/ai2-llm?region=us-east-1&bucketType=general&prefix=checkpoints/OLMo-medium/peteish13-highlr/&showversions=false
    - steps 102500+ are on hf (and gcp) -- https://huggingface.co/allenai/OLMo-2-1124-13B
    """
    # Get and print evenly spaced branches
    branches = get_olmo_branches(num_checkpoints=150)

    # Create destination directory if it doesn't exist
    dest_dir = '/oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr'
    os.makedirs(dest_dir, exist_ok=True)

    # Process each branch and copy/sync to destination
    for branch in tqdm(branches, total=len(branches), desc='Downloading OLMo 2 models'):
        step = branch.split('step')[1]
        if '-' in step: step = step.split('-')[0]
        step = int(step)

        dest_path = f"{dest_dir}/step{step}"
        
        if branch.startswith('/oe-training-default'):
            # We need to use convert_checkpoints_peteish.sh to convert the checkpoint instead
            continue
        elif branch.startswith('s3://'):
            # AWS S3 sync
            dest_path = f"{dest_dir}/step{step}-sharded"
            if not os.path.exists(dest_path):
                print(f"Syncing {branch} to {dest_path}")
                os.system(f"aws s3 sync {branch} {dest_path}")
        else:
            # Hugging Face download
            if not os.path.exists(dest_path):
                print(f"Downloading {branch} to {dest_path}")
                repo_id = "allenai/OLMo-2-1124-13B"
                os.system(f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {repo_id} --local-dir {dest_path} --revision {branch}")
