import argparse
from huggingface_hub import HfApi
from tqdm import tqdm
import numpy as np
import os

def get_olmo_branches(repo_id, num_checkpoints=None):
    api = HfApi()
    
    # Get all branches for the repo
    branches = api.list_repo_refs(repo_id)
    branches = [branch.name for branch in branches.branches]
    
    # Filter and sort branches that match the pattern
    steps = []
    stage1_branches = []
    for branch in branches:
        branch = str(branch)
        if branch.startswith('stage1-step') or branch.startswith('step'):
            step = int(branch.split('step')[1].split('-')[0])
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

    if num_checkpoints is not None:
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
    else:
        # Return all checkpoints
        selected_branches = stage1_branches.tolist()

    return selected_branches


def download_branches(repo_id, branches, dest_dir):
    """ Pull OLMo intermediate checkpoints from WEKA """
    # Process each branch and copy/sync to destination
    for branch in tqdm(branches, total=len(branches), desc=f'Downloading OLMo 2 models to {dest_dir}'):
        step = branch.split('step')[1]
        if '-' in step: step = step.split('-')[0]
        step = int(step)

        dest_path = f"{dest_dir}/step{step}"

        # Hugging Face download
        if not os.path.exists(dest_path):
            print(f"Downloading {branch} to {dest_path}")
            os.system(f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {repo_id} --local-dir {dest_path} --revision {branch}")


if __name__ == '__main__':
    """
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-2-1124-7B --save-dir OLMo-medium/peteish7 --num-checkpoints 150
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-2-1124-13B --save-dir OLMo-medium/peteish13-highlr --num-checkpoints 150
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-2-0325-32B --save-dir OLMo-large/peteish32 --num-checkpoints 150

    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-190M-5xC --save-dir OLMo-ladder/190M-5xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-370M-5xC --save-dir OLMo-ladder/370M-5xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-760M-5xC --save-dir OLMo-ladder/760M-5xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-1B-5xC --save-dir OLMo-ladder/1B-5xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-3B-5xC --save-dir OLMo-ladder/3B-5xC

    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-190M-1xC --save-dir OLMo-ladder/190M-1xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-370M-1xC --save-dir OLMo-ladder/370M-1xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-760M-1xC --save-dir OLMo-ladder/760M-1xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-1B-1xC --save-dir OLMo-ladder/1B-1xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-3B-1xC --save-dir OLMo-ladder/3B-1xC

    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-190M-0.5xC --save-dir OLMo-ladder/190M-0.5xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-370M-0.5xC --save-dir OLMo-ladder/370M-0.5xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-760M-0.5xC --save-dir OLMo-ladder/760M-0.5xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-1B-0.5xC --save-dir OLMo-ladder/1B-0.5xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-3B-0.5xC --save-dir OLMo-ladder/3B-0.5xC

    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-190M-2xC --save-dir OLMo-ladder/190M-2xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-370M-2xC --save-dir OLMo-ladder/370M-2xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-760M-2xC --save-dir OLMo-ladder/760M-2xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-1B-2xC --save-dir OLMo-ladder/1B-2xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-3B-2xC --save-dir OLMo-ladder/3B-2xC

    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-190M-10xC --save-dir OLMo-ladder/190M-10xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-370M-10xC --save-dir OLMo-ladder/370M-10xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-760M-10xC --save-dir OLMo-ladder/760M-10xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-1B-10xC --save-dir OLMo-ladder/1B-10xC
    python scripts/download_checkpoints_hf.py --repo-id allenai/OLMo-Ladder-3B-10xC --save-dir OLMo-ladder/3B-10xC
    """

    parser = argparse.ArgumentParser(description='Download OLMo checkpoints from Hugging Face')
    parser.add_argument('--repo-id', type=str, required=True,
                      help='Hugging Face repo ID to download from')
    parser.add_argument('--save-dir', type=str, required=True,
                      help='Directory name to save checkpoints under /oe-eval-default/ai2-llm/checkpoints/')
    parser.add_argument('--num-checkpoints', type=int, default=None,
                      help='Number of evenly spaced checkpoints to download (default: None)')

    args = parser.parse_args()

    dest_dir = f'/oe-eval-default/ai2-llm/checkpoints/{args.save_dir}'
    os.makedirs(dest_dir, exist_ok=True) # Create destination directory if it doesn't exist

    # Get and print evenly spaced branches
    branches = get_olmo_branches(args.repo_id, num_checkpoints=args.num_checkpoints)

    # Download branches
    download_branches(args.repo_id, branches, dest_dir)