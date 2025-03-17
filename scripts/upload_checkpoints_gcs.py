import os, time, sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from analysis.utils import weka_to_gcs
from analysis.utils.constants_models import MODEL_LADDER_LIST, MODEL_LIST_MIXES_FINAL, MODEL_LIST_MIXES_FINAL_EXTENDED, MODEL_LIST_INTERMEDIATE, MODEL_LIST_INTERMEDIATE_13B, MODEL_LIST_MIXES, OE_EVAL_BASE_MODELS, OE_EVAL_INSTRUCT_MODELS, OE_EVAL_ALL_MODELS, OE_EVAL_BASE_MODELS_EXTENDED

MODEL_LIST_ALL = []
# MODEL_LIST_ALL += MODEL_LADDER_LIST
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_13B # 13B intermediate ckpts
MODEL_LIST_ALL += MODEL_LIST_MIXES_FINAL # ian's new mixes
MODEL_LIST_ALL += MODEL_LIST_MIXES_FINAL_EXTENDED # extended set of data mixes

# MODEL_LIST_ALL = MODEL_LIST_ALL[:5]

# Filter for weka models and extract paths after checkpoints/
weka_models = [model for model in MODEL_LIST_ALL if model.startswith('weka://')]
gcs_paths = [weka_to_gcs(model) for model in weka_models]

# Create rsync commands
from concurrent.futures import ThreadPoolExecutor

def check_and_sync(weka_model, gcs_path):
    weka_model = weka_model.replace('weka://', '/')
    check_command = f"gsutil ls {gcs_path}"
    
    if os.system(check_command) != 0:
        command = f"""\
            gsutil -o "GSUtil:parallel_composite_upload_threshold=150M" -m rsync -r {weka_model} {gcs_path} \
        """
        print(f'Executing command:\n{command}')
        os.system(command)
    else:
        print(f'Skipping {gcs_path} - already exists')

with ThreadPoolExecutor() as executor:
    executor.map(lambda x: check_and_sync(*x), zip(weka_models, gcs_paths))
