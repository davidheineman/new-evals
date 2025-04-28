import os, time, sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from analysis.utils.constants_models import MODEL_LIST_MIXES_FINAL, MODEL_LIST_MIXES_FINAL_EXTENDED


MODEL_LIST_ALL = []
MODEL_LIST_ALL += MODEL_LIST_MIXES_FINAL # ian's new mixes
MODEL_LIST_ALL += MODEL_LIST_MIXES_FINAL_EXTENDED


# /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/DCLM-baseline-150M-5xC-2/step38157-unsharded-hf

MODEL_PATHS = [model.replace('weka://', '/') for model in MODEL_LIST_ALL]
MODEL_PATHS = ['/'.join(model.split('/')[:-1]) for model in MODEL_PATHS]

def get_last_n_checkpoints(path, n=5):
    # Get all checkpoint folders
    ckpts = [d for d in os.listdir(path) if d.startswith('step') and d.endswith('-unsharded-hf')]
    
    # Extract step numbers and sort
    ckpts_with_steps = [(int(d.split('-')[0].replace('step','')), d) for d in ckpts]
    ckpts_with_steps.sort(key=lambda x: x[0], reverse=True)
    
    # Get last n checkpoints
    return [os.path.join(path, ckpt[1]) for ckpt in ckpts_with_steps[:n]]

# Get final 5 checkpoints for each model path
final_ckpts = []
for path in MODEL_PATHS:
    if os.path.exists(path):
        final_ckpts.extend(get_last_n_checkpoints(path))


# Save the list of final checkpoints to a text file
output_file = 'final_checkpoints.txt'
with open(output_file, 'w') as f:
    for ckpt in final_ckpts:
        f.write(f"{ckpt}\n")

print(MODEL_PATHS)